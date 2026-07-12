"""Week-9 D3: BUG-informed eviction scoring via low-rank surprise.

Thesis. A token the tracked low-rank basis reconstructs well (``||k - U U^T k||``
small) is *redundant* with the running summary and safe to evict; a high-residual
token is an *outlier* the summary cannot reproduce and should be kept exact. This
is the orthogonal complement of ``retention="energy"`` (which keeps high
*in*-subspace mass ``||U^T k||`` -- a Week-7 negative) and a candidate *better*
eviction signal than attention mass.

Two modes:

* ``--mode probe`` (the GO/NO-GO gate, run FIRST -- nearly free). Instrument a
  ``retention="attn"`` run to also track surprise and log, at every eviction, the
  per-column (attention-mass, surprise, energy) arrays. Pool across layers/docs
  and report Spearman/Pearson correlations. **Decision:** ``|rho(surprise,
  attn)| > 0.7`` -> surprise re-ranks columns the way attention already does ->
  redundant -> D3 is dead (write the honest negative); ``< 0.4`` -> orthogonal ->
  complementary -> proceed to the sweep; in between -> weak go (only a deep-tail /
  retrieval win counts). The surprise<->energy correlation is logged as a
  built-in sanity check (they are near-anti-correlated by Pythagoras).

* ``--mode sweep``. The matched-memory head-to-head: at a frozen per-layer float
  budget, compare streaming ppl of ``bugS`` (surprise) vs ``bugA`` (attention),
  ``bugE`` (energy), ``bug`` (fifo), and ``morph``. Falsifiable target: ``bugS <
  bugA`` in aggregate ppl, or at least in the deep-tail bins (where a dropped
  outlier needed later shows up). Every arm is solved to the SAME budget via
  ``coord_for_config`` (surprise costs ``2r+2``/column, like attn) and audited
  ``mem_max <= budget``.

Protocol mirrors ``w5_streamppl``/``w7_rank_sweep``. Results JSON between markers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import matplotlib
import numpy as np
from perplexity_sweep import load_corpus_ids, load_model
from scipy import stats
from transformers.cache_utils import Cache, DynamicCache
from w5_streamppl import (
    N_SINK,
    bug_budget_floats,
    morph_capacity_for_budget,
    stream_score,
    summarize,
)
from w7_rank_sweep import coord_for_config

from kvdlra.cache import BugStreamingCache, MorphKVCache
from kvdlra.utils.seed import seed_everything

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
JSON_BEGIN = "===W9_SURPRISE_JSON_BEGIN==="
JSON_END = "===W9_SURPRISE_JSON_END==="


# --------------------------------------------------------------------- probe


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Correlation gate: pool (attn-mass, surprise, energy) at eviction events."""
    seed_everything(args.seed)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    corpus = load_corpus_ids(tokenizer, args.device, corpus=args.corpus)

    scores: list[np.ndarray] = []
    surprises: list[np.ndarray] = []
    energies: list[np.ndarray] = []
    span = args.prefill + args.g_tokens + 1
    n_events = 0
    for d in range(args.doc_start, args.doc_start + args.n_docs):
        ids = corpus[args.doc_stride * d : args.doc_stride * d + span]
        if ids.shape[0] < span:
            break
        cache = BugStreamingCache(
            model,
            rank=args.rank,
            coord_budget=args.coord_budget,
            recent_window=args.recent_window,
            absorb_block=args.absorb_block,
            n_sink=N_SINK,
            retention="attn",
            score_decay=args.score_decay,
        )
        sink = cache.enable_surprise_probe()
        stream_score(model, ids, cache, args.prefill, args.device)
        for score, surprise, energy in sink:
            scores.append(score.numpy())
            surprises.append(surprise.numpy())
            energies.append(energy.numpy())
        n_events += len(sink)
        print(f"[doc {d}] {len(sink)} eviction events recorded", flush=True)

    s = np.concatenate(scores)
    u = np.concatenate(surprises)
    e = np.concatenate(energies)
    rho_sa = float(stats.spearmanr(u, s).statistic)
    r_sa = float(stats.pearsonr(u, s).statistic)
    rho_se = float(stats.spearmanr(u, e).statistic)
    if abs(rho_sa) > 0.7:
        gate = "NO-GO (redundant with attention)"
    elif abs(rho_sa) < 0.4:
        gate = "GO (complementary to attention)"
    else:
        gate = "WEAK-GO (only deep-tail/retrieval win counts)"
    result = {
        "mode": "probe",
        "model": args.model,
        "rank": args.rank,
        "coord_budget": args.coord_budget,
        "n_samples": int(s.size),
        "n_events": n_events,
        "spearman_surprise_attn": rho_sa,
        "pearson_surprise_attn": r_sa,
        "spearman_surprise_energy": rho_se,
        "gate": gate,
    }
    print(
        f"\n[probe] n={s.size} samples over {n_events} events\n"
        f"  Spearman(surprise, attn)   = {rho_sa:+.3f}\n"
        f"  Pearson (surprise, attn)   = {r_sa:+.3f}\n"
        f"  Spearman(surprise, energy) = {rho_se:+.3f}  (weak: normalized surprise "
        f"resid/||k|| is scale-free, so it is decoupled from unnormalized energy ||c|| "
        f"because ||k|| varies per token)\n"
        f"  GATE: {gate}",
        flush=True,
    )
    return result


# --------------------------------------------------------------------- sweep


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Matched-memory streaming-ppl comparison of the retention signals."""
    seed_everything(args.seed)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    n_layers = int(cfg.num_hidden_layers)
    corpus = load_corpus_ids(tokenizer, args.device, corpus=args.corpus)

    budget = bug_budget_floats(
        n, args.rank, args.coord_budget, args.recent_window, args.absorb_block
    )
    rw, ab = args.recent_window, args.absorb_block
    morph_c = morph_capacity_for_budget(budget, n, h_kv, args.morph_recent)
    blend_alphas = [float(x) for x in args.blend_alphas.split(",")] if args.blend_alphas else []
    w = {
        ret: coord_for_config(budget, n, args.rank, rw, ab, ret)
        for ret in ("fifo", "attn", "energy", "lowrank_surprise", "blend")
    }
    print(
        f"budget={budget}/layer (~{budget / (2 * n):.0f} tok-eq) morph_C={morph_c} W={w}",
        flush=True,
    )

    def make_methods() -> dict[str, Cache | None]:
        m: dict[str, Cache | None] = {"full": None} if args.with_full else {}
        common = {
            "rank": args.rank,
            "recent_window": rw,
            "absorb_block": ab,
            "n_sink": N_SINK,
            "score_decay": args.score_decay,
        }
        m[f"bug-r{args.rank}-W{w['fifo']}"] = BugStreamingCache(
            model, coord_budget=w["fifo"], retention="fifo", **common
        )
        m[f"bugA-r{args.rank}-W{w['attn']}"] = BugStreamingCache(
            model, coord_budget=w["attn"], retention="attn", **common
        )
        m[f"bugE-r{args.rank}-W{w['energy']}"] = BugStreamingCache(
            model, coord_budget=w["energy"], retention="energy", **common
        )
        m[f"bugS-r{args.rank}-W{w['lowrank_surprise']}"] = BugStreamingCache(
            model, coord_budget=w["lowrank_surprise"], retention="lowrank_surprise", **common
        )
        for alpha in blend_alphas:
            m[f"bugB{alpha:.2f}-r{args.rank}-W{w['blend']}"] = BugStreamingCache(
                model,
                coord_budget=w["blend"],
                retention="blend",
                surprise_blend=alpha,
                **common,
            )
        m[f"morph-C{morph_c}-R{args.morph_recent}"] = MorphKVCache(
            model, capacity=morph_c, recent_window=args.morph_recent
        )
        return m

    results: dict[str, Any] = {
        "mode": "sweep",
        "model": args.model,
        "n_features": n,
        "n_layers": n_layers,
        "prefill": args.prefill,
        "g_tokens": args.g_tokens,
        "bin_size": args.bin_size,
        "rank": args.rank,
        "budget_floats_per_layer": budget,
        "budget_token_equivalents": budget / (2 * n),
        "coord_budgets": w,
        "docs": {},
    }
    span = args.prefill + args.g_tokens + 1
    for d in range(args.doc_start, args.doc_start + args.n_docs):
        ids = corpus[args.doc_stride * d : args.doc_stride * d + span]
        if ids.shape[0] < span:
            print(f"[doc {d}] corpus exhausted", flush=True)
            break
        per_doc: dict[str, Any] = {}
        for name, cache in make_methods().items():
            cache_obj: Cache = cache if cache is not None else DynamicCache()
            rec = stream_score(model, ids, cache_obj, args.prefill, args.device)
            rec.update(summarize(rec.pop("nlls"), args.bin_size))
            if name != "full":
                over = rec["mem_max"] / (n_layers * budget)
                rec["mem_max_over_budget"] = over
                if over > 1.0:
                    print(f"[doc {d}] WARN {name} over budget {over:.3f}x", flush=True)
            per_doc[name] = rec
            print(f"[doc {d}] {name}: ppl={rec['ppl']:.3f} mem_max={rec['mem_max']}", flush=True)
        results["docs"][str(d)] = per_doc
    results["verdict"] = sweep_verdict(results)
    return results


def _geo(vals: list[float]) -> float:
    return float(np.exp(np.mean(np.log(vals))))


def sweep_verdict(results: dict[str, Any]) -> dict[str, Any]:
    docs = results["docs"]
    if not docs:
        return {}
    names = list(next(iter(docs.values())))

    def nm(prefix: str) -> str | None:
        return next((x for x in names if x.startswith(prefix)), None)

    bug_s, bug_a, bug_e = nm("bugS"), nm("bugA"), nm("bugE")
    if bug_s is None or bug_a is None:
        return {}
    agg = {x: _geo([docs[d][x]["ppl"] for d in docs]) for x in names}
    # deep-tail: geo-mean of the last-bin ppl over docs.
    nb = min(len(docs[d][bug_s]["ppl_bins"]) for d in docs)
    blend_names = [x for x in names if x.startswith("bugB")]
    tail_names = [x for x in [bug_s, bug_a, bug_e, *blend_names] if x is not None]
    tail = {x: _geo([docs[d][x]["ppl_bins"][nb - 1] for d in docs]) for x in tail_names}
    out: dict[str, Any] = {
        "agg_ppl": agg,
        "deep_tail_ppl": tail,
        "surprise_beats_attn_agg": bool(agg[bug_s] < agg[bug_a]),
        "surprise_beats_attn_tail": bool(tail[bug_s] < tail[bug_a]),
    }
    if blend_names:
        best_blend = min(blend_names, key=lambda x: agg[x])
        out["best_blend"] = best_blend
        out["blend_beats_attn_agg"] = bool(agg[best_blend] < agg[bug_a])
        out["blend_beats_attn_tail"] = bool(tail[best_blend] < tail[bug_a])
    return out


def make_figure(results: dict[str, Any], fig_path: Path) -> None:
    if results.get("mode") != "sweep":
        return
    docs = results["docs"]
    if not docs:
        return
    names = list(next(iter(docs.values())))
    nb = min(len(docs[d][nm]["ppl_bins"]) for d in docs for nm in names)
    bs = results["bin_size"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        if name == "full":
            continue
        ys = [
            _geo([docs[d][name]["ppl_bins"][i] / docs[d]["full"]["ppl_bins"][i] for d in docs])
            if "full" in names
            else _geo([docs[d][name]["ppl_bins"][i] for d in docs])
            for i in range(nb)
        ]
        ax.plot([(i + 1) * bs for i in range(nb)], ys, marker=".", label=name)
    ax.set_xlabel("decode position")
    ax.set_ylabel("per-bin ppl ratio to full" if "full" in names else "per-bin ppl")
    ax.set_title(
        f"D3 surprise vs attention retention @ ~{results['budget_token_equivalents']:.0f} tok-eq"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"[wrote {fig_path}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="probe", choices=["probe", "sweep"])
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
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--coord-budget", type=int, default=64)
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--absorb-block", type=int, default=16)
    parser.add_argument("--morph-recent", type=int, default=32)
    parser.add_argument("--score-decay", type=float, default=0.97)
    parser.add_argument(
        "--blend-alphas",
        default="",
        help="sweep-mode: alpha weights for retention='blend' (attn vs surprise), e.g. 0.5,0.75",
    )
    parser.add_argument("--with-full", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default="results/w9-surprise-1b.json")
    parser.add_argument("--fig", default="figures/week9/surprise_1b.png")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    out_json = Path(args.out_json)
    if args.plot_only:
        results = json.loads(out_json.read_text())
    else:
        results = run_probe(args) if args.mode == "probe" else run_sweep(args)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"[wrote {out_json}]")
        print(JSON_BEGIN)
        print(json.dumps(results))
        print(JSON_END)
    make_figure(results, Path(args.fig))


if __name__ == "__main__":
    main()
