"""Axis B benchmark: streaming perplexity under constant-memory decode caches.

The question (``docs/week5-plan.md`` §"Axis B", ``docs/notes/
streaming-decode-design.md`` §8): during long *generation-regime* processing --
every token through the decode path, one at a time, against a **bounded** cache
-- does BUG's rank-``r`` running low-rank summary of the whole history preserve
language-model quality better than a constant-size set of retained whole tokens
(MorphKV, ICML'25; StreamingLLM) at **matched stored memory**? And does its
advantage/deficit *grow or shrink with generated length* (the graceful-
degradation slope)?

Protocol (teacher-forced streaming, deterministic)
--------------------------------------------------
Slice a long held-out document into ``prefill + scored`` tokens. Pre-fill the
first ``P`` tokens with full attention (identical for every method -- bounded
methods bound what is *retained*, per the design note), then feed the next
``G`` tokens **one per forward** (q_len=1, exactly the decode regime), scoring
each step's log-prob of the incoming token against the method's bounded cache.
Per-token NLL -> perplexity, reported overall and in position bins (the
quality-vs-generated-length curve). The full cache is the O(T) upper bound;
everything else holds stored memory constant, and the measured stored-float
curves are recorded to prove it.

Matched memory (design note §7)
-------------------------------
One full token costs ``2 n`` floats per layer (K+V, ``n = head_dim *
num_kv_heads``). Budgets are set by the BUG configuration's worst-case stored
floats (sinks + recent ring + U/B/coords for K and V), then MorphKV's
``capacity`` (its score buffer counted -- sum/max fusion needs the last ``R``
attention rows) and StreamingLLM's window are solved to the same budget. At
matched memory BUG attends over ~``n/r`` x more (approximately represented)
history; attended lengths are reported alongside so the memory/compute trade
stays visible.

Runs on CPU at 1B for a smoke (``--g-tokens 512``); the real run is 8B on a pod
(``--model unsloth/Meta-Llama-3.1-8B-Instruct --device cuda --dtype bfloat16``).
Results JSON is echoed between ``===W5_STREAMPPL_JSON_BEGIN/END===`` markers
(``[[vastai-pod-flakiness-jul2026]]``). ``--plot-only`` rebuilds the figure.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import matplotlib
import torch
from perplexity_sweep import load_corpus_ids, load_model
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

from kvdlra.cache import BugStreamingCache, MorphKVCache
from kvdlra.utils.seed import seed_everything

matplotlib.use("Agg")  # headless / CPU-safe backend; set before pyplot import
import matplotlib.pyplot as plt

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
JSON_BEGIN = "===W5_STREAMPPL_JSON_BEGIN==="
JSON_END = "===W5_STREAMPPL_JSON_END==="
N_SINK = 4


def bug_budget_floats(
    n: int, rank: int, coord_budget: int, recent_window: int, absorb_block: int
) -> int:
    """Worst-case stored floats per layer for one BUG cache configuration:
    sinks + recent ring at its high-water mark + (U, B, C) for keys AND values."""
    ring = recent_window + absorb_block - 1
    return 2 * n * N_SINK + 2 * n * ring + 2 * n * rank + 2 * rank * coord_budget + 2 * rank * rank


def solve_bug_variant(
    budget: int,
    n: int,
    rank: int,
    recent_window: int,
    absorb_block: int,
    n_layers: int,
    variant: str,
    quant_bits: int,
    quant_keep_frac: float,
) -> dict[str, int]:
    """Coordinate budgets for a Week-7 BUG variant fitted to the SAME per-layer
    float budget as the baseline (honesty guardrail: every variant pays for its
    own bookkeeping *out of* the coordinate budget, never on top of it).

    Per-column float-equivalent costs (see ``BugStreamingLayer.stored_state_numel``):
    fp32 coord column = ``2*rank`` (K+V); +1 position (int32) and +1 score (fp32)
    each under adaptive retention; quantized column = ``2*rank*bits/32`` (codes,
    bit-packable) + 2 fp32 norms (+ position/score under adaptive retention).
    The ``retention="attn"`` ring score buffer (ring high-water floats) and the
    shared PolarQuant side info (counted once per cache -> ceil-shared per
    layer) come off the top.
    """
    ring = recent_window + absorb_block - 1
    fixed = 2 * n * N_SINK + 2 * n * ring + 2 * n * rank + 2 * rank * rank
    pool = budget - fixed  # the baseline spends all of this on 2*rank*W fp32 coords
    per_f = 2 * rank
    per_q_codes = 2.0 * rank * quant_bits / 32.0  # K+V codes, float-equivalents
    side = rank * rank + 2**quant_bits + (2**quant_bits - 1)  # Pi + centroids + bounds
    side_share = -(-side // n_layers)  # ceil: shared side info charged per layer
    if variant == "bug":
        return {"coord_budget": pool // per_f, "quant_budget": 0}
    if variant == "bugE":  # +1 position per column
        return {"coord_budget": pool // (per_f + 1), "quant_budget": 0}
    if variant == "bugA":  # +1 position +1 score per column, + ring score buffer
        return {"coord_budget": (pool - ring) // (per_f + 2), "quant_budget": 0}
    if variant == "bugD":  # FIFO fp32 tier + FIFO quant tier
        w_f = int(pool * quant_keep_frac) // per_f
        q_pool = pool - w_f * per_f - side_share
        return {"coord_budget": w_f, "quant_budget": int(q_pool // (per_q_codes + 2))}
    if variant == "bugAD":  # both tiers position+score tracked, + ring scores
        pool2 = pool - ring
        w_f = int(pool2 * quant_keep_frac) // (per_f + 2)
        q_pool = pool2 - w_f * (per_f + 2) - side_share
        return {"coord_budget": w_f, "quant_budget": int(q_pool // (per_q_codes + 4))}
    raise ValueError(f"unknown BUG variant {variant!r}")


def morph_capacity_for_budget(budget: int, n: int, h_kv: int, recent_window: int) -> int:
    """Largest MorphKV ``capacity`` whose stored floats fit ``budget`` per layer:
    kept K/V of ``C + R`` tokens (2n each) + the score buffer (h_kv * R rows)."""
    per_token = 2 * n + h_kv * recent_window
    total_tokens = budget // per_token
    return max(1, total_tokens - recent_window)


def sllm_window_for_budget(budget: int, n: int, absorb_block: int) -> int:
    """Largest StreamingLLM recent window whose sinks + ring **high-water mark**
    (window + absorb_block - 1 tokens) fit ``budget``."""
    return max(1, budget // (2 * n) - N_SINK - (absorb_block - 1))


def build_methods(
    model: PreTrainedModel,
    n: int,
    h_kv: int,
    tier: dict[str, int],
    methods: list[str] | None = None,
    quant_bits: int = 4,
    quant_keep_frac: float = 0.5,
    score_decay: float = 0.97,
) -> dict[str, Cache | None]:
    """Caches for one matched-memory tier (``None`` => full DynamicCache).

    ``methods`` picks from: ``full``, ``bug`` (Week-6 baseline), ``bugA`` /
    ``bugE`` (Week-7 A: attention / energy retention), ``bugD`` (Week-7 D:
    quantized age tier), ``bugAD`` (A+D), ``morph``, ``snapkvD``, ``sllm``.
    Every BUG variant is solved to the SAME per-layer float budget as the
    baseline BUG configuration (:func:`solve_bug_variant`)."""
    if methods is None:
        methods = ["full", "bug", "morph", "snapkvD", "sllm"]  # the Week-6 set
    rank, coord, recent, absorb = (
        tier["rank"],
        tier["coord_budget"],
        tier["recent_window"],
        tier["absorb_block"],
    )
    budget = bug_budget_floats(n, rank, coord, recent, absorb)
    n_layers = int(model.config.num_hidden_layers)
    morph_r = tier["morph_recent"]
    morph_c = morph_capacity_for_budget(budget, n, h_kv, morph_r)
    sllm_w = sllm_window_for_budget(budget, n, absorb)

    def bug_common(v: dict[str, int]) -> dict[str, Any]:
        return {
            "rank": rank,
            "coord_budget": v["coord_budget"],
            "recent_window": recent,
            "absorb_block": absorb,
            "n_sink": N_SINK,
        }

    out: dict[str, Cache | None] = {}
    for m in methods:
        if m == "full":
            out["full"] = None
        elif m == "bug":
            out[f"bug-r{rank}"] = BugStreamingCache(model, **bug_common(tier))
        elif m == "bugE":
            v = solve_bug_variant(
                budget, n, rank, recent, absorb, n_layers, m, quant_bits, quant_keep_frac
            )
            out[f"bugE-r{rank}-W{v['coord_budget']}"] = BugStreamingCache(
                model, retention="energy", **bug_common(v)
            )
        elif m == "bugA":
            v = solve_bug_variant(
                budget, n, rank, recent, absorb, n_layers, m, quant_bits, quant_keep_frac
            )
            out[f"bugA-r{rank}-W{v['coord_budget']}"] = BugStreamingCache(
                model, retention="attn", score_decay=score_decay, **bug_common(v)
            )
        elif m == "bugD":
            v = solve_bug_variant(
                budget, n, rank, recent, absorb, n_layers, m, quant_bits, quant_keep_frac
            )
            out[f"bugD-r{rank}-Wf{v['coord_budget']}-Wq{v['quant_budget']}b{quant_bits}"] = (
                BugStreamingCache(
                    model,
                    quant_bits=quant_bits,
                    quant_budget=v["quant_budget"],
                    **bug_common(v),
                )
            )
        elif m == "bugAD":
            v = solve_bug_variant(
                budget, n, rank, recent, absorb, n_layers, m, quant_bits, quant_keep_frac
            )
            out[f"bugAD-r{rank}-Wf{v['coord_budget']}-Wq{v['quant_budget']}b{quant_bits}"] = (
                BugStreamingCache(
                    model,
                    retention="attn",
                    score_decay=score_decay,
                    quant_bits=quant_bits,
                    quant_budget=v["quant_budget"],
                    **bug_common(v),
                )
            )
        elif m == "morph":
            out[f"morph-C{morph_c}-R{morph_r}"] = MorphKVCache(
                model, capacity=morph_c, recent_window=morph_r
            )
        elif m == "snapkvD":
            # SnapKV-style decode eviction = the same windowed-attention top-C scoring
            # applied every `interval` steps (kvpress's DecodingPress+SnapKV mis-rotates
            # buffered window queries under transformers 5.8, so the periodic variant of
            # the faithful MorphKV core stands in -- exact rotations, same score family).
            # Capacity leaves room for the between-prunes overshoot (interval tokens).
            out[f"snapkvD-C{max(1, morph_c - 16)}-i16"] = MorphKVCache(
                model,
                capacity=max(1, morph_c - 16),
                recent_window=morph_r,
                evict_interval=16,
            )
        elif m == "sllm":
            out[f"sllm-w{sllm_w}"] = BugStreamingCache(
                model,
                rank=0,
                coord_budget=0,
                recent_window=sllm_w,
                absorb_block=absorb,
                n_sink=N_SINK,
            )
        else:
            raise ValueError(f"unknown method {m!r}")
    return out


def stream_score(
    model: PreTrainedModel,
    ids: torch.Tensor,
    cache: Cache,
    prefill: int,
    device: str,
) -> dict[str, Any]:
    """Teacher-forced streaming scoring: pre-fill ``ids[:prefill]``, then feed the
    rest one token per forward, scoring each step's next-token log-prob."""
    ids = ids.unsqueeze(0).to(device)
    total = ids.shape[1]
    nlls: list[float] = []
    mems: list[int] = []
    lats: list[float] = []
    attach = (
        cache.attach(model)
        if isinstance(cache, MorphKVCache | BugStreamingCache)
        else nullcontext()
    )  # BugStreamingCache.attach is a no-op unless retention="attn"
    with torch.no_grad(), attach:
        out = model(ids[:, :prefill], past_key_values=cache, use_cache=True)
        for t in range(prefill, total - 1):
            tok = ids[:, t : t + 1]
            t0 = time.perf_counter()
            out = model(tok, past_key_values=cache, use_cache=True)
            lats.append(time.perf_counter() - t0)
            logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            nlls.append(-float(logp[ids[0, t + 1]]))
            if isinstance(cache, BugStreamingCache | MorphKVCache):
                mems.append(int(cache.stored_state_numel()))
            else:
                full_floats = 0
                for layer in cache.layers:
                    if isinstance(layer, DynamicLayer) and layer.keys is not None:
                        assert layer.values is not None
                        full_floats += int(layer.keys.numel()) + int(layer.values.numel())
                mems.append(full_floats)
    lat_sorted = sorted(lats[1:])
    return {
        "nlls": nlls,
        "mem_first": mems[0],
        "mem_last": mems[-1],
        "mem_max": max(mems),
        "lat_ms_p50": 1e3 * lat_sorted[len(lat_sorted) // 2],
        "lat_ms_p90": 1e3 * lat_sorted[int(0.9 * len(lat_sorted))],
    }


def summarize(nlls: list[float], bin_size: int) -> dict[str, Any]:
    """Overall perplexity + per-position-bin perplexities (degradation curve)."""
    t = torch.tensor(nlls, dtype=torch.float64)
    bins = [
        float(torch.exp(t[i : i + bin_size].mean()))
        for i in range(0, len(nlls) - bin_size + 1, bin_size)
    ]
    return {"ppl": float(torch.exp(t.mean())), "ppl_bins": bins}


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    corpus = load_corpus_ids(tokenizer, args.device, corpus=args.corpus)

    tier = {
        "rank": args.rank,
        "coord_budget": args.coord_budget,
        "recent_window": args.recent_window,
        "absorb_block": args.absorb_block,
        "morph_recent": args.morph_recent,
    }
    budget = bug_budget_floats(
        n, args.rank, args.coord_budget, args.recent_window, args.absorb_block
    )
    methods_list = [m.strip() for m in args.methods.split(",") if m.strip()]
    n_layers = int(model.config.num_hidden_layers)
    variant_budgets = {
        m: solve_bug_variant(
            budget,
            n,
            args.rank,
            args.recent_window,
            args.absorb_block,
            n_layers,
            m,
            args.quant_bits,
            args.quant_keep_frac,
        )
        for m in methods_list
        if m.startswith("bug") and m != "bug"
    }
    results: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "corpus": args.corpus,
        "n_features": n,
        "prefill": args.prefill,
        "g_tokens": args.g_tokens,
        "bin_size": args.bin_size,
        "tier": tier,
        "methods": methods_list,
        "quant_bits": args.quant_bits,
        "quant_keep_frac": args.quant_keep_frac,
        "score_decay": args.score_decay,
        "variant_budgets": variant_budgets,
        "budget_floats_per_layer": budget,
        "budget_token_equivalents": budget / (2 * n),
        "docs": {},
    }
    span = args.prefill + args.g_tokens + 1
    for d in range(args.n_docs):
        start = args.doc_stride * d
        ids = corpus[start : start + span]
        if ids.shape[0] < span:
            print(f"[doc {d}] corpus exhausted, stopping at {d} docs", flush=True)
            break
        per_doc: dict[str, Any] = {}
        methods = build_methods(
            model,
            n,
            h_kv,
            tier,
            methods=methods_list,
            quant_bits=args.quant_bits,
            quant_keep_frac=args.quant_keep_frac,
            score_decay=args.score_decay,
        )
        for name, cache in methods.items():
            cache_obj: Cache = cache if cache is not None else DynamicCache()
            print(f"[doc {d}] {name} ...", flush=True)
            rec = stream_score(model, ids, cache_obj, args.prefill, args.device)
            rec.update(summarize(rec.pop("nlls"), args.bin_size))
            attended = (
                cache_obj.attended_length()
                if isinstance(cache_obj, BugStreamingCache | MorphKVCache)
                else args.prefill + args.g_tokens
            )
            rec["attended_length_final"] = attended
            if name != "full":
                # Matched-memory audit: every bounded method must fit the tier
                # budget (morph/snapkvD were solved to it; BUG variants pay their
                # bookkeeping out of the coordinate budget).
                rec["mem_max_over_budget"] = rec["mem_max"] / (n_layers * budget)
                if rec["mem_max"] > n_layers * budget:
                    print(
                        f"[doc {d}] WARNING {name} exceeds budget: "
                        f"{rec['mem_max']} > {n_layers * budget}",
                        flush=True,
                    )
            per_doc[name] = rec
            print(
                f"[doc {d}] {name}: ppl={rec['ppl']:.3f} mem_max={rec['mem_max']}"
                f" attended={attended}",
                flush=True,
            )
        results["docs"][str(d)] = per_doc
    return results


def make_figure(results: dict[str, Any], fig_path: Path) -> None:
    docs = results["docs"]
    if not docs:
        return
    names = list(next(iter(docs.values())))
    colors = {"full": "tab:blue"}
    palette = [
        "tab:orange",
        "tab:brown",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:pink",
        "tab:olive",
        "tab:cyan",
    ]
    for i, name in enumerate(n for n in names if n != "full"):
        colors[name] = palette[i % len(palette)]

    has_full = "full" in names
    n_panels = 3 if has_full else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.2))
    ax_curve, ax_mem = axes[0], axes[-1]
    bin_size = results["bin_size"]
    import math as _math

    for name in names:
        all_bins = [docs[d][name]["ppl_bins"] for d in docs]
        n_bins = min(len(b) for b in all_bins)
        mean_bins = [sum(b[i] for b in all_bins) / len(all_bins) for i in range(n_bins)]
        xs = [(i + 1) * bin_size for i in range(n_bins)]
        ppl = sum(docs[d][name]["ppl"] for d in docs) / len(docs)
        ax_curve.plot(
            xs,
            mean_bins,
            marker="o",
            ms=3,
            color=colors[name],
            label=f"{name} (ppl {ppl:.2f})",
        )
        if has_full and name != "full":
            # Geo-mean over docs of the per-bin ratio to that doc's full cache.
            ratios = []
            for i in range(n_bins):
                logs = [
                    _math.log(docs[d][name]["ppl_bins"][i] / docs[d]["full"]["ppl_bins"][i])
                    for d in docs
                ]
                ratios.append(_math.exp(sum(logs) / len(logs)))
            axes[1].plot(xs, ratios, marker="o", ms=3, color=colors[name], label=name)
    ax_curve.set_xlabel("scored token position (decode steps past prefill)")
    ax_curve.set_ylabel(f"perplexity per {bin_size}-token bin")
    ax_curve.set_title(
        f"streaming ppl vs generated length -- {results['model'].split('/')[-1]}, "
        f"budget ~{results['budget_token_equivalents']:.0f} tok-eq/layer"
    )
    ax_curve.legend(fontsize=8)
    ax_curve.grid(alpha=0.3)

    if has_full:
        axes[1].axhline(1.0, color="tab:blue", lw=1, ls="--")
        axes[1].set_xlabel("scored token position (decode steps past prefill)")
        axes[1].set_ylabel("per-bin ppl ratio to full cache (geo-mean over docs)")
        axes[1].set_title("degradation slope (1.0 = no penalty)")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

    d0 = next(iter(docs.values()))
    xs2 = range(len(names))
    mem_max = [d0[n]["mem_max"] / 1e6 for n in names]
    ax_mem.bar(xs2, mem_max, color=[colors[n] for n in names], alpha=0.85)
    ax_mem.set_xticks(list(xs2))
    ax_mem.set_xticklabels(names, rotation=20, fontsize=8)
    ax_mem.set_ylabel("peak stored cache floats (millions)")
    ax_mem.set_title("measured peak stored memory (doc 0)")
    ax_mem.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"[wrote {fig_path}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--corpus", default="wikitext-103")
    parser.add_argument("--prefill", type=int, default=512)
    parser.add_argument("--g-tokens", type=int, default=2048)
    parser.add_argument("--bin-size", type=int, default=256)
    parser.add_argument("--n-docs", type=int, default=2)
    parser.add_argument("--doc-stride", type=int, default=100_000)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--coord-budget", type=int, default=1024)
    parser.add_argument("--recent-window", type=int, default=64)
    parser.add_argument("--absorb-block", type=int, default=32)
    parser.add_argument("--morph-recent", type=int, default=32)
    parser.add_argument(
        "--methods",
        default="full,bug,morph,snapkvD,sllm",
        help="comma list: full,bug,bugA,bugE,bugD,bugAD,morph,snapkvD,sllm",
    )
    parser.add_argument("--quant-bits", type=int, default=4)
    parser.add_argument("--quant-keep-frac", type=float, default=0.5)
    parser.add_argument("--score-decay", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default="results/w5-streamppl-1b.json")
    parser.add_argument("--fig", default="figures/week5/streamppl_1b.png")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    out_json = Path(args.out_json)
    if args.plot_only:
        results = json.loads(out_json.read_text())
    else:
        results = run(args)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"[wrote {out_json}]")
        print(JSON_BEGIN)
        print(json.dumps(results))
        print(JSON_END)
    make_figure(results, Path(args.fig))


if __name__ == "__main__":
    main()
