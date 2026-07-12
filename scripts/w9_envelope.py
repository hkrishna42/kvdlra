"""Week-9 D2: adaptive-SLASH as the {eviction, BUG} envelope across budgets.

The competition question is settled (Weeks 5-8): at a *single* extreme budget
BUG loses to eviction on mean ppl. This script asks the *envelope* question
instead. SLASH -- exact heavy-hitters (``hh_budget``) + a rank-``r`` low-rank
tail + ``coord_budget`` coverage on :class:`BugStreamingCache` -- spans a family
that ranges from eviction-like (large ``hh``, small ``r``) to BUG-like
(``hh=0``). A **budget-adaptive** allocation ``(hh, r, W)`` per *total* memory
budget should trace the Pareto envelope of {eviction, BUG} as the budget sweeps
from moderate to extreme.

The verdict is deliberately two-tiered and honest (``docs/week9-plan.md`` D2):

* **No-regret** -- adaptive-SLASH ppl ``<= min(pure-morph, pure-bugA) + eps`` at
  *every* budget. The strong "vs both" form is expected to FAIL at the extreme
  end: the ``2 n r`` basis is a *fixed* charge SLASH/BUG pay and eviction does
  not, so below a byte-measured budget floor no SLASH allocation can reach morph
  (the overhead-floor wall, ``docs/week7-dominance.md``). The survivable form is
  no-regret *vs BUG* (``<= bugA + eps`` everywhere), since SLASH ``>=`` BUG by
  construction.
* **Crossover strict win** -- adaptive-SLASH *strictly* beats BOTH pure methods
  (``< min(...) - eps``) in the crossover region where the regimes meet. The
  upside bet; if absent, the contribution is "one cache, no regret."

Honest framing (do not overclaim): the SLASH family does NOT *contain* MorphKV
(``rank=0`` degenerates to StreamingLLM, not attention-scored whole-token
eviction), so morph is an **external** baseline the envelope is measured against,
not a special case it reproduces. And the "lean to exact heavy-hitters at the
extreme" intuition is *refuted* by the Week-7 data (``w7-slash-doc0.json``: hh4 <
hh8 < hh12) -- the a-priori schedule therefore keeps ``hh`` small and slowly
growing.

Two allocation rules are reported side by side:

* **a-priori** -- a fixed closed-form ``tau -> (r, hh)`` (``W`` then solved to
  budget). No peeking at ppl; the deployable contribution.
* **per-budget-best** (``--with-oracle``) -- grid-search ``(r, hh)`` and pick the
  min *measured* ppl at each budget. Labelled an ORACLE over allocation: an upper
  bound on what the rule could achieve, NOT deployable. Bounds the gap the
  a-priori rule leaves on the table.

Every arm at every budget is solved to the SAME frozen per-layer float budget via
:func:`w7_rank_sweep.coord_for_config` (which counts hh at ``2 n + 2`` and every
per-column bookkeeping float honestly) and ``morph_capacity_for_budget``; the
measured ``mem_max`` is audited ``<= budget`` for each. A caller-side feasibility
filter rejects any ``(r, hh)`` whose coverage ``W`` falls below one absorb block
(``coord_for_config``'s ``max(1, ...)`` clamp otherwise masks infeasibility).

Protocol mirrors ``w5_streamppl``/``w7_rank_sweep`` exactly (teacher-forced
streaming ppl, geo-mean over docs). Results JSON between markers; ``--plot-only``
rebuilds the figure.

GO/NO-GO gate (the cheap cut): run only the crossover budgets (e.g.
``--budgets-tok-eq 128,180,256,360``), ``--n-docs 1``, a compact ``--bug-ranks``.
If adaptive-SLASH never strictly beats ``min(morph, bugA)`` there, the ceiling is
"no-regret / one cache" -- stop; do not spend the full sweep or an 8B pod
re-confirming a bounded result.
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
JSON_BEGIN = "===W9_ENVELOPE_JSON_BEGIN==="
JSON_END = "===W9_ENVELOPE_JSON_END==="
RANK_GRID = (16, 24, 32, 48, 64, 96, 128)
HH_GRID = (0, 4, 8, 16, 32)


def snap_to(grid: tuple[int, ...], x: int) -> int:
    """Nearest grid value to ``x`` (ties -> the smaller)."""
    return min(grid, key=lambda g: (abs(g - x), g))


def feasible_coord(
    budget: int, n: int, r: int, rw: int, ab: int, retention: str, hh: int
) -> int | None:
    """Coverage ``W`` for ``(r, hh)`` at ``budget``, or ``None`` if infeasible.

    ``coord_for_config`` clamps its result to ``max(1, ...)`` (``w7_rank_sweep``),
    which silently turns an over-budget config into a degenerate ``W=1`` that
    *looks* feasible. A config whose honest coverage is below one absorb block is
    rejected here (the caller-side feasibility filter the D2 spec calls for)."""
    w = coord_for_config(budget, n, r, rw, ab, retention, hh_budget=hh)
    return w if w >= ab else None


def apriori_alloc(tau: float, budget: int, n: int, rw: int, ab: int) -> tuple[int, int, int] | None:
    """The deployable a-priori schedule ``tau -> (r, hh, W)`` (no ppl peeking).

    Rank grows ~``sqrt(tau)`` (more fidelity per token as memory grows) but is
    **floored at 48**: the gate measured a rank-squeeze catastrophe below that
    (r16/r32 plateau at ppl ~17.4 regardless of coverage on 1B). The exact
    heavy-hitter tier stays *small* and grows slowly (the Week-7 data show a large
    exact tier loses at the extreme). ``r`` is stepped down to the largest
    feasible grid rank, reducing ``hh`` only if even ``r=48`` is infeasible."""
    r_target = snap_to(RANK_GRID, min(128, max(48, round(5.0 * math.sqrt(tau)))))
    hh_target = snap_to(HH_GRID, max(0, round(0.9 * math.sqrt(tau)) - 4))
    hh_cands = sorted({h for h in HH_GRID if h <= hh_target} | {0}, reverse=True)
    for hh in hh_cands:
        feasible = [
            r
            for r in RANK_GRID
            if r <= r_target and feasible_coord(budget, n, r, rw, ab, "attn", hh) is not None
        ]
        if feasible:
            r = max(feasible)
            w = feasible_coord(budget, n, r, rw, ab, "attn", hh)
            assert w is not None
            return r, hh, w
    return None


def best_slash_alloc(budget: int, n: int, rw: int, ab: int) -> list[tuple[int, int, int]]:
    """Every feasible ``(r, hh, W)`` in the per-budget-best grid (the oracle
    reference; the caller runs each and keeps the min-ppl one)."""
    out: list[tuple[int, int, int]] = []
    for r in RANK_GRID:
        for hh in HH_GRID:
            w = feasible_coord(budget, n, r, rw, ab, "attn", hh)
            if w is not None:
                out.append((r, hh, w))
    return out


def geo(vals: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def build_methods(
    model: Any,
    args: argparse.Namespace,
    budget: int,
    n: int,
    morph_c: int,
    feasible_bug: list[int],
    apriori: tuple[int, int, int],
    rw: int,
    ab: int,
    ret: str,
) -> dict[str, Cache | None]:
    """Fresh caches for one (budget, doc): pure morph + pure bugA (per feasible
    rank) + a-priori SLASH [+ the per-budget-best oracle grid]. Rebuilt per doc
    because the caches are stateful."""
    ap_r, ap_hh, ap_w = apriori
    m: dict[str, Cache | None] = {}
    if args.with_full:
        m["full"] = None
    m[f"morph-C{morph_c}-R{args.morph_recent}"] = MorphKVCache(
        model, capacity=morph_c, recent_window=args.morph_recent
    )
    for r in feasible_bug:
        w = feasible_coord(budget, n, r, rw, ab, ret, 0)
        assert w is not None
        m[f"bugA-r{r}-W{w}"] = BugStreamingCache(
            model,
            rank=r,
            coord_budget=w,
            recent_window=rw,
            absorb_block=ab,
            n_sink=N_SINK,
            retention=ret,
            score_decay=args.score_decay,
        )
    m[f"slashAP-r{ap_r}-h{ap_hh}-W{ap_w}"] = BugStreamingCache(
        model,
        rank=ap_r,
        coord_budget=ap_w,
        recent_window=rw,
        absorb_block=ab,
        n_sink=N_SINK,
        retention=ret,
        score_decay=args.score_decay,
        hh_budget=ap_hh,
    )
    if args.with_oracle:
        for r, hh, w in best_slash_alloc(budget, n, rw, ab):
            m[f"slashO-r{r}-h{hh}-W{w}"] = BugStreamingCache(
                model,
                rank=r,
                coord_budget=w,
                recent_window=rw,
                absorb_block=ab,
                n_sink=N_SINK,
                retention=ret,
                score_decay=args.score_decay,
                hh_budget=hh,
            )
    return m


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    n_layers = int(cfg.num_hidden_layers)
    corpus = load_corpus_ids(tokenizer, args.device, corpus=args.corpus)

    rw, ab, ret = args.recent_window, args.absorb_block, "attn"
    taus = [float(x) for x in args.budgets_tok_eq.split(",") if x]
    bug_ranks = [int(x) for x in args.bug_ranks.split(",") if x]

    results: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "corpus": args.corpus,
        "n_features": n,
        "n_layers": n_layers,
        "prefill": args.prefill,
        "g_tokens": args.g_tokens,
        "bin_size": args.bin_size,
        "recent_window": rw,
        "absorb_block": ab,
        "morph_recent": args.morph_recent,
        "score_decay": args.score_decay,
        "budgets_tok_eq": taus,
        "bug_ranks": bug_ranks,
        "with_oracle": bool(args.with_oracle),
        "with_full": bool(args.with_full),
        "budgets": {},
    }

    span = args.prefill + args.g_tokens + 1
    for tau in taus:
        budget = round(tau * 2 * n)  # floats per layer
        morph_c = morph_capacity_for_budget(budget, n, h_kv, args.morph_recent)
        alloc = apriori_alloc(tau, budget, n, rw, ab)
        if alloc is None:
            print(f"[tau {tau:.0f}] no feasible a-priori allocation; skipping", flush=True)
            continue
        ap_r, ap_hh, ap_w = alloc
        feasible_bug = [
            r for r in bug_ranks if feasible_coord(budget, n, r, rw, ab, ret, 0) is not None
        ]
        print(
            f"[tau {tau:.0f}] budget={budget}/layer morph_C={morph_c} "
            f"a-priori=(r{ap_r},h{ap_hh},W{ap_w}) bugA_ranks={feasible_bug}",
            flush=True,
        )
        per_budget: dict[str, Any] = {
            "budget_floats_per_layer": budget,
            "morph_capacity": morph_c,
            "apriori": {"rank": ap_r, "hh": ap_hh, "coord_budget": ap_w},
            "docs": {},
        }
        for d in range(args.doc_start, args.doc_start + args.n_docs):
            start = args.doc_stride * d
            ids = corpus[start : start + span]
            if ids.shape[0] < span:
                print(f"[tau {tau:.0f}] corpus exhausted at doc {d}", flush=True)
                break
            doc_rec: dict[str, Any] = {}
            methods = build_methods(
                model, args, budget, n, morph_c, feasible_bug, alloc, rw, ab, ret
            )
            for name, cache in methods.items():
                cache_obj: Cache = cache if cache is not None else DynamicCache()
                rec = stream_score(model, ids, cache_obj, args.prefill, args.device)
                rec.update(summarize(rec.pop("nlls"), args.bin_size))
                if name != "full":
                    over = rec["mem_max"] / (n_layers * budget)
                    rec["mem_max_over_budget"] = over
                    if over > 1.0:
                        print(
                            f"[tau {tau:.0f}] WARN {name} over budget: "
                            f"{rec['mem_max']} > {n_layers * budget} ({over:.3f}x)",
                            flush=True,
                        )
                doc_rec[name] = rec
                print(
                    f"[tau {tau:.0f}][doc {d}] {name}: ppl={rec['ppl']:.3f} "
                    f"mem_max={rec['mem_max']}",
                    flush=True,
                )
            per_budget["docs"][str(d)] = doc_rec
        results["budgets"][f"{tau:.0f}"] = per_budget
    results["verdict"] = verdict(results)
    return results


def _agg_ppl(docs: dict[str, Any], name: str) -> float:
    return geo([docs[d][name]["ppl"] for d in docs])


def verdict(results: dict[str, Any]) -> dict[str, Any]:
    """Per-budget no-regret / crossover-strict-win verdict (geo-mean over docs).

    ``eps`` is a *relative* band, ``eps_rel`` of the local min ppl, widened to the
    per-point doc half-range where that is larger (noise-limited points flagged).
    Raw ``s - min(m, b)`` deltas are always tabled so readers judge independently
    of the ``eps`` choice."""
    eps_rel = results.get("eps_rel", 0.01)
    rows: list[dict[str, Any]] = []
    for tau, pb in results["budgets"].items():
        docs = pb["docs"]
        if not docs:
            continue
        names = list(next(iter(docs.values())))
        morph_nm = next((nm for nm in names if nm.startswith("morph")), None)
        ap_nm = next((nm for nm in names if nm.startswith("slashAP")), None)
        bug_nms = [nm for nm in names if nm.startswith("bugA")]
        if morph_nm is None or ap_nm is None or not bug_nms:
            continue
        m = _agg_ppl(docs, morph_nm)
        s = _agg_ppl(docs, ap_nm)
        b = min(_agg_ppl(docs, nm) for nm in bug_nms)
        best_bug_nm = min(bug_nms, key=lambda nm: _agg_ppl(docs, nm))
        # oracle-best slash (per-budget-best), if run.
        o_nms = [nm for nm in names if nm.startswith("slashO")]
        o = min((_agg_ppl(docs, nm) for nm in o_nms), default=float("nan"))
        lo = min(m, b)
        # per-point doc half-range of the a-priori arm (noise proxy).
        ap_ppls = [docs[d][ap_nm]["ppl"] for d in docs]
        half_range = (max(ap_ppls) - min(ap_ppls)) / 2 if len(ap_ppls) > 1 else 0.0
        eps = max(eps_rel * lo, half_range)
        rows.append(
            {
                "tau": float(tau),
                "morph": m,
                "bugA_best": b,
                "bugA_best_name": best_bug_nm,
                "slash_apriori": s,
                "slash_oracle": o if o_nms else None,
                "min_pure": lo,
                "s_minus_min": s - lo,
                "s_minus_bug": s - b,
                "eps": eps,
                "noise_limited": half_range > eps_rel * lo,
                "no_regret_vs_both": bool(s <= lo + eps),
                "no_regret_vs_bug": bool(s <= b + eps),
                "strict_win_vs_both": bool(s < lo - eps),
            }
        )
    return {
        "eps_rel": eps_rel,
        "rows": rows,
        "no_regret_vs_both_all": all(r["no_regret_vs_both"] for r in rows) if rows else False,
        "no_regret_vs_bug_all": all(r["no_regret_vs_bug"] for r in rows) if rows else False,
        "any_strict_win": any(r["strict_win_vs_both"] for r in rows) if rows else False,
    }


def make_figure(results: dict[str, Any], fig_path: Path) -> None:
    v = results.get("verdict", {})
    rows = v.get("rows", [])
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["tau"])
    taus = [r["tau"] for r in rows]
    fig, (ax_env, ax_delta) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax_env.plot(taus, [r["morph"] for r in rows], "s--", color="tab:green", label="pure morph")
    ax_env.plot(taus, [r["bugA_best"] for r in rows], "o--", color="tab:orange", label="pure bugA")
    ax_env.plot(
        taus, [r["slash_apriori"] for r in rows], "P-", color="tab:red", label="slash a-priori"
    )
    if any(r["slash_oracle"] is not None for r in rows):
        ax_env.plot(
            taus,
            [r["slash_oracle"] for r in rows],
            "*:",
            color="tab:purple",
            label="slash oracle (per-budget best)",
        )
    ax_env.plot(
        taus,
        [r["min_pure"] for r in rows],
        "-",
        color="black",
        lw=0.8,
        alpha=0.5,
        label="lower envelope min(pure)",
    )
    ax_env.set_xscale("log")
    ax_env.set_xlabel("per-layer budget (token-equivalents)")
    ax_env.set_ylabel("streaming ppl (geo-mean docs)")
    ax_env.set_title("cross-budget envelope: adaptive-SLASH vs pure {eviction, BUG}")
    ax_env.legend(fontsize=8)
    ax_env.grid(alpha=0.3, which="both")
    # Panel 2: signed margin of a-priori SLASH to the lower envelope (+eps band).
    ax_delta.axhline(0.0, color="black", lw=0.8)
    ax_delta.plot(
        taus, [r["s_minus_min"] for r in rows], "P-", color="tab:red", label="s - min(pure)"
    )
    ax_delta.fill_between(
        taus,
        [-r["eps"] for r in rows],
        [r["eps"] for r in rows],
        color="gray",
        alpha=0.2,
        label="+/- eps (no-regret band)",
    )
    ax_delta.set_xscale("log")
    ax_delta.set_xlabel("per-layer budget (token-equivalents)")
    ax_delta.set_ylabel("ppl margin to envelope (< 0 = strict win)")
    ax_delta.set_title("no-regret / crossover margin")
    ax_delta.legend(fontsize=8)
    ax_delta.grid(alpha=0.3, which="both")
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
    parser.add_argument("--budgets-tok-eq", default="64,89,128,180,256,360,515,720")
    parser.add_argument("--bug-ranks", default="16,24,32,48,64,96,128")
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--absorb-block", type=int, default=16)
    parser.add_argument("--morph-recent", type=int, default=32)
    parser.add_argument("--score-decay", type=float, default=0.97)
    parser.add_argument("--with-oracle", action="store_true", help="run per-budget-best slash grid")
    parser.add_argument("--with-full", action="store_true", help="run the O(T) full-cache arm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default="results/w9-envelope-1b.json")
    parser.add_argument("--fig", default="figures/week9/envelope_1b.png")
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
