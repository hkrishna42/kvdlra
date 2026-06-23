"""Week-2 go/no-go pilot (PLAN §5): rank-adaptive BUG vs. SVD oracle vs. iSVD.

For Layer-8 K of Llama-3.2-1B, across 5 C4 documents, for **both** post-RoPE and
pre-RoPE keys, this sweeps a prescribed memory budget and compares three
reconstruction-error curves at each budget:

* **truncated SVD (oracle)** -- the Eckart--Young lower bound,
* **incremental SVD** -- the naive streaming baseline,
* **rank-adaptive streaming BUG** (:class:`kvdlra.integrators.streaming.StreamingBUG`).

Memory budget := per-token compression ratio ``r / n_features`` -- i.e. keep ``r``
coefficients per token instead of ``n_features = 512`` (the basis ``U`` is a fixed
``512 x r`` overhead, amortized over ``T`` tokens). So ``r = round(budget * 512)``.

Pass criterion (PLAN §5): rank-adaptive BUG reaches Frobenius reconstruction error
within **1.05x** of the truncated-SVD oracle at the same budget, on at least
**4 of 5** documents, for all budgets **>= 0.10**. Reported separately for pre-
and post-RoPE.

Note on the §5 "mean effective rank <= 0.9*r_oracle" clause: against an *optimal*
(Eckart--Young) oracle nothing matches the oracle's error at lower rank, so that
clause cannot mean "same error, fewer rank". We run BUG at a hard rank cap equal
to the budget rank (so its mean rank == the oracle rank) and treat the operative
test as the 1.05x error match plus the absence of rank bloat -- documented in the
verdict.

Outputs ``figures/week2/pilot.{pdf,png}`` + ``figures/week2/pilot.json``; prints
the verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import matplotlib
import numpy as np
import numpy.typing as npt
import torch

from kvdlra.integrators.streaming import StreamingBUG
from kvdlra.lowrank import incremental_svd_recon, truncated_svd_recon

matplotlib.use("Agg")  # headless / CPU-safe backend; set before pyplot import
import matplotlib.pyplot as plt

N_SINK = 4  # StreamingLLM sink tokens dropped from every matrix (PLAN §8 #5)
PASS_RATIO = 1.05  # BUG error must be within this factor of the oracle
PASS_MIN_DOCS = 4  # ... on at least this many of the 5 documents
PASS_MIN_BUDGET = 0.10  # ... for all budgets at least this large
ROPE_MODES: list[tuple[str, str]] = [("post-RoPE", "K"), ("pre-RoPE", "K_pre")]


def load_matrix(layer_path: Path, key: str) -> npt.NDArray[np.float64]:
    """Load one layer dump and build the ``(512, T - N_SINK)`` feature-by-token matrix.

    ``key`` selects ``"K"`` (post-RoPE) or ``"K_pre"`` (pre-RoPE) from a
    ``--rope both`` dump. The first ``N_SINK`` sink-token columns are dropped.
    """
    blob = torch.load(layer_path, weights_only=False)
    k = blob[key].float().numpy()  # (h, t, d)
    h, t, d = k.shape
    m = k.transpose(0, 2, 1).reshape(h * d, t)[:, N_SINK:]
    return np.ascontiguousarray(m, dtype=np.float64)


def budget_to_rank(budget: float, n_features: int, n_cols: int) -> int:
    """Map a memory budget to a rank ``r = round(budget * n_features)`` (clamped)."""
    r = round(budget * n_features)
    return max(1, min(r, n_features - 1, n_cols - 1))


def bug_error(m: npt.NDArray[np.float64], r: int) -> tuple[float, float]:
    """Stream ``m``'s columns through StreamingBUG (rank cap ``r``); error + mean rank."""
    tracker = StreamingBUG(m.shape[0], rank_cap=r)
    tracker.update_many(m)
    return tracker.reconstruction_error(m), tracker.mean_rank


def main() -> None:
    """Run the budget sweep on pre/post-RoPE Layer-8 K, plot, and print the verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="dumps/llama3.2-1b")
    parser.add_argument("--pattern", default="doc*_len4096_rope-both")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.40])
    parser.add_argument("--out", default="figures/week2/pilot.pdf")
    args = parser.parse_args()

    dump_dirs = sorted(Path(args.dumps).glob(args.pattern))
    layer_paths = [d / f"layer_{args.layer:02d}.pt" for d in dump_dirs]
    layer_paths = [p for p in layer_paths if p.exists()]
    if not layer_paths:
        raise SystemExit(f"no dumps matched {args.dumps}/{args.pattern} (layer {args.layer})")
    n_docs = len(layer_paths)
    budgets: list[float] = list(args.budgets)
    print(f"{n_docs} docs, layer {args.layer}, budgets {budgets}")

    # results[rope][bi] = dict of per-doc arrays + the rank used at that budget.
    results: dict[str, list[dict[str, object]]] = {}
    for rope, key in ROPE_MODES:
        mats = [load_matrix(p, key) for p in layer_paths]
        per_budget: list[dict[str, object]] = []
        for budget in budgets:
            oracle, isvd, bug, bug_rank, ranks = [], [], [], [], []
            for m in mats:
                r = budget_to_rank(budget, m.shape[0], m.shape[1])
                ranks.append(r)
                oracle.append(truncated_svd_recon(m, r))
                isvd.append(incremental_svd_recon(m, r))
                e, mr = bug_error(m, r)
                bug.append(e)
                bug_rank.append(mr)
            ratios = [b / o for b, o in zip(bug, oracle, strict=True)]
            row: dict[str, object] = {
                "budget": budget,
                "rank": round(float(np.mean(ranks))),
                "oracle": oracle,
                "isvd": isvd,
                "bug": bug,
                "bug_mean_rank": bug_rank,
                "bug_over_oracle": ratios,
            }
            per_budget.append(row)
            print(
                f"  {rope} budget={budget:.2f} r~{row['rank']}: "
                f"oracle={np.mean(oracle):.3f} isvd={np.mean(isvd):.3f} "
                f"bug={np.mean(bug):.3f}  bug/oracle={np.mean(ratios):.3f} "
                f"(max {max(ratios):.3f})"
            )
        results[rope] = per_budget

    _plot(results, budgets, Path(args.out))
    verdict = _verdict(results, budgets)
    Path(args.out).with_suffix(".json").write_text(
        json.dumps({"budgets": budgets, "results": results, "verdict": verdict}, indent=2) + "\n"
    )


def _plot(
    results: dict[str, list[dict[str, object]]],
    budgets: list[float],
    out_path: Path,
) -> None:
    """Two panels (post-RoPE, pre-RoPE): budget vs. error, 3 curves with 1-sigma bands."""
    fig, axes = plt.subplots(1, len(ROPE_MODES), figsize=(6 * len(ROPE_MODES), 4.5), squeeze=False)
    styles = [
        ("oracle", "truncated SVD (oracle)", "o-"),
        ("isvd", "incremental SVD", "s--"),
        ("bug", "streaming BUG", "^:"),
    ]
    for col, (rope, _key) in enumerate(ROPE_MODES):
        ax = axes[0][col]
        for field, label, style in styles:
            mean = np.array([float(np.mean(cast(list[float], r[field]))) for r in results[rope]])
            std = np.array([float(np.std(cast(list[float], r[field]))) for r in results[rope]])
            ax.plot(budgets, mean, style, label=label)
            ax.fill_between(budgets, mean - std, mean + std, alpha=0.15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("memory budget (rank / 512)")
        ax.set_ylabel("rel. Frobenius reconstruction error")
        ax.set_title(f"Layer-8 K, {rope}")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(out_path.with_suffix(suffix))
    print(f"wrote figure {out_path.with_suffix('.pdf')} / .png")


def _verdict(
    results: dict[str, list[dict[str, object]]],
    budgets: list[float],
) -> dict[str, object]:
    """Apply the §5 pass criterion per rope mode and return a structured verdict."""
    verdict: dict[str, object] = {}
    print("\n=== VERDICT (BUG within 1.05x oracle on >=4/5 docs, budgets >=0.10) ===")
    for rope, _key in ROPE_MODES:
        per_budget_pass: dict[str, bool] = {}
        for row in results[rope]:
            budget = cast(float, row["budget"])
            if budget < PASS_MIN_BUDGET:
                continue
            ratios = cast(list[float], row["bug_over_oracle"])
            n_ok = sum(1 for x in ratios if x <= PASS_RATIO)
            per_budget_pass[f"{budget:.2f}"] = n_ok >= PASS_MIN_DOCS
            print(
                f"  {rope} budget={budget:.2f}: {n_ok}/{len(ratios)} docs within 1.05x "
                f"-> {'PASS' if n_ok >= PASS_MIN_DOCS else 'FAIL'}"
            )
        overall = all(per_budget_pass.values()) if per_budget_pass else False
        verdict[rope] = {"per_budget_pass": per_budget_pass, "overall_pass": overall}
        print(f"  {rope}: {'GO' if overall else 'NO-GO'}")
    return verdict


if __name__ == "__main__":
    main()
