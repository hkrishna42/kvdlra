"""Week-7 dominance program, Technique 1: WaterBUG-uniform rank<->coverage sweep.

The only regime where BUG still loses is the **rank squeeze** (mechanism 3): at
an aggressive memory budget a rank-``r`` low-rank summary of the history is too
coarse and whole-token eviction (MorphKV) wins. The free variable nobody has
tuned is the split of a *fixed* per-layer memory budget between **rank** ``r``
(fidelity per retained token) and **coordinate coverage** ``W`` (how many tokens
the low-rank summary spans): lowering ``r`` frees ``2n`` floats of basis + the
``r^2`` core and cheapens every coordinate column from ``2r`` to less, so ``W``
grows. The augmented BUG integrator is already rank-agnostic, so this is a pure
allocation question -- no new state, the cleanest possible matched-memory story.

The honest claim (no dominance): this is a **sweep** over the rank baseline BUG
fixes, so it ``>=`` baseline BUG by construction; it does NOT contain MorphKV
(a rank-16 summary of W tokens is not W exact tokens), so no dominance over
eviction is asserted. The falsifiable question is whether trading rank for
coverage *helps at all* in the squeeze:

  * WIN signature: aggregate ppl is **U-shaped in r** with the minimum at
    ``r < r_ref`` (spending less on rank, more on coverage), and that minimum
    ``<=`` MorphKV -- the aggressive-budget gap closes.
  * KILL signature: **monotone** -- ``r = r_ref`` is best, every smaller ``r``
    strictly worse. That cleanly refutes rank-for-coverage (a publishable
    negative that kills the rank-adaptivity direction).

Matched memory (the load-bearing methodological move). The per-layer float
budget is **frozen once** at a reference config and every swept rank is solved
to *that same budget*: ``W(r) = (budget - 2n*N_SINK - 2n*ring - 2n*r - 2r^2) /
(2r)``. MorphKV's capacity and StreamingLLM's window are solved to the identical
budget (reusing the ``w5_streamppl`` solvers). Rank is a free scalar config with
zero persistent state, so the measured ``mem_max`` is audited ``<= budget`` for
every method and every rank.

Protocol mirrors ``w5_streamppl`` exactly (streaming perplexity, teacher-forced
one-token decode, geo-mean bins). Results JSON between markers. ``--plot-only``
rebuilds the figure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import matplotlib
from perplexity_sweep import load_corpus_ids, load_model
from transformers.cache_utils import Cache, DynamicCache
from w5_streamppl import (
    N_SINK,
    bug_budget_floats,
    morph_capacity_for_budget,
    stream_score,
    summarize,
)

from kvdlra.cache import BugStreamingCache, MorphKVCache
from kvdlra.utils.seed import seed_everything

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
JSON_BEGIN = "===W7_RANKSWEEP_JSON_BEGIN==="
JSON_END = "===W7_RANKSWEEP_JSON_END==="


def coord_for_config(
    budget: int, n: int, rank: int, recent: int, absorb: int, retention: str, hh_budget: int = 0
) -> int:
    """Largest fp32 coordinate coverage ``W`` for ``rank`` (and an optional
    exact heavy-hitter tier of ``hh_budget`` tokens) at a FROZEN ``budget``,
    counting the retention overhead HONESTLY (matched memory).

    fixed = sinks + recent ring + basis(2n*r) + core(2r^2) [+ ring_score(ring)
    when retention="attn"]; each fp32 coord column costs ``2*rank`` (+2 for
    position + score under attn); each exact heavy-hitter costs ``2n`` (+2 for
    position + score, always attn). Mirrors ``BugStreamingLayer.
    stored_state_numel`` exactly, so measured ``mem_max`` audits ``<= budget``."""
    ring = recent + absorb - 1
    fixed = 2 * n * N_SINK + 2 * n * ring + 2 * n * rank + 2 * rank * rank
    if retention == "attn":
        fixed += ring  # ring_score buffer (high-water)
        per_f = 2 * rank + 2  # coords K+V + position + score
        per_hh = 2 * n + 2  # verbatim K+V + position + score
    else:  # fifo: no positions/scores/ring buffer
        per_f = 2 * rank
        per_hh = 2 * n
    return max(1, (budget - fixed - per_hh * hh_budget) // per_f)


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    corpus = load_corpus_ids(tokenizer, args.device, corpus=args.corpus)

    # Freeze the per-layer float budget ONCE at the reference config.
    budget = bug_budget_floats(
        n, args.ref_rank, args.ref_coord, args.recent_window, args.absorb_block
    )
    ranks = [int(x) for x in args.ranks.split(",")]
    results: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "corpus": args.corpus,
        "n_features": n,
        "prefill": args.prefill,
        "g_tokens": args.g_tokens,
        "bin_size": args.bin_size,
        "recent_window": args.recent_window,
        "absorb_block": args.absorb_block,
        "budget_floats_per_layer": budget,
        "budget_token_equivalents": budget / (2 * n),
        "ranks": ranks,
        "ref_rank": args.ref_rank,
        "retention": args.retention,
        "score_decay": args.score_decay,
        "docs": {},
    }
    # Solve every method to the FROZEN budget (honest retention accounting).
    morph_c = morph_capacity_for_budget(budget, n, h_kv, args.morph_recent)
    rw, ab, ret = args.recent_window, args.absorb_block, args.retention
    coords = {r: coord_for_config(budget, n, r, rw, ab, ret) for r in ranks}
    hh_list = [int(x) for x in args.hh_budgets.split(",") if x] if args.hh_budgets else []
    # SLASH variants at the fixed ref_rank: each exact heavy-hitter costs 2n+2,
    # taken out of the coordinate coverage W (matched memory).
    slash_rank = args.slash_rank if args.slash_rank else args.ref_rank
    slash = {
        hh: coord_for_config(budget, n, slash_rank, rw, ab, "attn", hh_budget=hh) for hh in hh_list
    }
    results["hh_budgets"] = hh_list
    print(
        f"budget {budget} floats/layer (~{budget / (2 * n):.0f} tok-eq); "
        f"morph_C={morph_c}; W(r)={coords}; SLASH r{slash_rank} W(hh)={slash}",
        flush=True,
    )

    tag = "bugA" if ret == "attn" else "bug"

    def make_methods() -> dict[str, Cache | None]:
        m: dict[str, Cache | None] = {"full": None}
        for r in ranks:
            m[f"{tag}-r{r}-W{coords[r]}"] = BugStreamingCache(
                model,
                rank=r,
                coord_budget=coords[r],
                recent_window=rw,
                absorb_block=ab,
                n_sink=N_SINK,
                retention=ret,
                score_decay=args.score_decay,
            )
        for hh in hh_list:
            m[f"slash-r{slash_rank}-h{hh}-W{slash[hh]}"] = BugStreamingCache(
                model,
                rank=slash_rank,
                coord_budget=slash[hh],
                recent_window=rw,
                absorb_block=ab,
                n_sink=N_SINK,
                retention="attn",
                score_decay=args.score_decay,
                hh_budget=hh,
            )
        m[f"morph-C{morph_c}-R{args.morph_recent}"] = MorphKVCache(
            model, capacity=morph_c, recent_window=args.morph_recent
        )
        return m

    span = args.prefill + args.g_tokens + 1
    for d in range(args.doc_start, args.doc_start + args.n_docs):
        start = args.doc_stride * d
        ids = corpus[start : start + span]
        if ids.shape[0] < span:
            print(f"[doc {d}] corpus exhausted at {d}", flush=True)
            break
        per_doc: dict[str, Any] = {}
        for name, cache in make_methods().items():
            cache_obj: Cache = cache if cache is not None else DynamicCache()
            rec = stream_score(model, ids, cache_obj, args.prefill, args.device)
            rec.update(summarize(rec.pop("nlls"), args.bin_size))
            if name != "full" and rec["mem_max"] > budget * cfg.num_hidden_layers:
                print(
                    f"[doc {d}] WARN {name} over budget: "
                    f"{rec['mem_max']} > {budget * cfg.num_hidden_layers}",
                    flush=True,
                )
            per_doc[name] = rec
            print(f"[doc {d}] {name}: ppl={rec['ppl']:.3f} mem_max={rec['mem_max']}", flush=True)
        results["docs"][str(d)] = per_doc
    return results


def make_figure(results: dict[str, Any], fig_path: Path) -> None:
    docs = results["docs"]
    if not docs:
        return
    names = list(next(iter(docs.values())))

    def geo(vals: list[float]) -> float:
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    fig, (ax_u, ax_bin) = plt.subplots(1, 2, figsize=(13, 4.6))
    # Panel 1: aggregate ppl vs rank (the U-shape test).
    ranks = results["ranks"]
    tag = (results.get("retention", "attn") == "attn" and "bugA") or "bug"
    bug_names = {r: next(nm for nm in names if nm.startswith(f"{tag}-r{r}-")) for r in ranks}
    aggs = {nm: geo([docs[d][nm]["ppl"] for d in docs]) for nm in names}
    ys_u = [aggs[bug_names[r]] for r in ranks]
    ax_u.plot(ranks, ys_u, "o-", color="tab:orange", label=f"{tag}(r)")
    morph_nm = next(nm for nm in names if nm.startswith("morph"))
    ax_u.axhline(aggs[morph_nm], color="tab:green", ls="--", label=f"morph {aggs[morph_nm]:.2f}")
    ax_u.axhline(aggs["full"], color="tab:blue", ls=":", label=f"full {aggs['full']:.2f}")
    ax_u.set_xlabel("BUG rank r (coverage W grows as r falls, matched memory)")
    ax_u.set_ylabel("aggregate streaming ppl (geo-mean docs)")
    ax_u.set_title(f"rank<->coverage sweep @ ~{results['budget_token_equivalents']:.0f} tok-eq")
    ax_u.legend(fontsize=8)
    ax_u.grid(alpha=0.3)
    # Panel 2: per-bin ratio to full for each rank + morph.
    nb = min(len(docs[d][nm]["ppl_bins"]) for d in docs for nm in names)
    bs = results["bin_size"]
    for r in ranks:
        nm = bug_names[r]
        ys = [
            geo([docs[d][nm]["ppl_bins"][i] / docs[d]["full"]["ppl_bins"][i] for d in docs])
            for i in range(nb)
        ]
        ax_bin.plot([(i + 1) * bs for i in range(nb)], ys, marker=".", label=f"r{r}")
    ys = [
        geo([docs[d][morph_nm]["ppl_bins"][i] / docs[d]["full"]["ppl_bins"][i] for d in docs])
        for i in range(nb)
    ]
    ax_bin.plot([(i + 1) * bs for i in range(nb)], ys, "k--", label="morph")
    ax_bin.set_xlabel("decode position")
    ax_bin.set_ylabel("per-bin ppl ratio to full")
    ax_bin.set_title("degradation slope by rank")
    ax_bin.legend(fontsize=8)
    ax_bin.grid(alpha=0.3)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"[wrote {fig_path}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--corpus", default="wikitext-2")
    parser.add_argument("--prefill", type=int, default=512)
    parser.add_argument("--g-tokens", type=int, default=1024)
    parser.add_argument("--bin-size", type=int, default=128)
    parser.add_argument("--n-docs", type=int, default=2)
    parser.add_argument("--doc-start", type=int, default=0)
    parser.add_argument("--doc-stride", type=int, default=100_000)
    parser.add_argument("--ranks", default="16,24,32,48,64")
    parser.add_argument("--ref-rank", type=int, default=64, help="rank setting the frozen budget")
    parser.add_argument("--ref-coord", type=int, default=64, help="coord setting the frozen budget")
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--absorb-block", type=int, default=16)
    parser.add_argument("--morph-recent", type=int, default=32)
    parser.add_argument("--hh-budgets", default="", help="SLASH exact-tier sizes, e.g. 8,16,24")
    parser.add_argument("--slash-rank", type=int, default=0, help="SLASH rank (0=ref-rank)")
    parser.add_argument("--retention", default="attn", choices=["attn", "fifo", "energy"])
    parser.add_argument("--score-decay", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default="results/w7-ranksweep-1b.json")
    parser.add_argument("--fig", default="figures/week7/ranksweep_1b.png")
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
