"""Week-4.5 DLRA integrator ablation (offline; no LLM, no pod).

Bakes off the three streaming DLRA integrators as KV subspace trackers on captured
pre-RoPE KV -- reconstruction error vs the truncated-SVD **oracle** plus per-sweep
**cost** -- and picks ONE winner to carry into the Week-5 comparisons
(``docs/week5-plan.md`` §"Week 4.5"). All three run through the identical
block-streaming harness (same first-block seed, same rank truncation, same
``|| M - U U^T M ||_F / || M ||_F`` metric) and differ only in the per-block step:

* **BUG** -- :func:`kvdlra.integrators.streaming_torch.blocked_bug_subspace`
  (augmented Galerkin, forward, square-root core). The incumbent.
* **PSI** -- :func:`kvdlra.integrators.streaming_variants.psi_subspace`
  (projector-splitting, backward S-step, covariance core).
* **Parallel-BUG** -- :func:`kvdlra.integrators.streaming_variants.parallel_bug_subspace`
  (decoupled in-basis/new-direction update).

Averages over the ``--pattern`` documents (pre-RoPE Layer-``--layer`` K by default,
the operating point) at each rank in ``--ranks``. Writes
``figures/week5/integrator_ablation.{png,pdf}`` + a ``.json`` sidecar and prints
the verdict. Use ``--plot-only`` to re-render from the sidecar.

Decision rule (``docs/week5-plan.md``): ship parallel-BUG only if it *matches* BUG
accuracy at lower cost; otherwise ship plain BUG. Everything clusters near the
oracle in the KV operating range (r <= 128), so read the tails and the cost.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import matplotlib
import numpy as np
import torch

from kvdlra.integrators.streaming_torch import blocked_bug_subspace
from kvdlra.integrators.streaming_variants import parallel_bug_subspace, psi_subspace

matplotlib.use("Agg")  # headless / CPU-safe backend; set before pyplot import
import matplotlib.pyplot as plt

N_SINK = 4  # StreamingLLM sink tokens dropped from every matrix (PLAN §8 #5)
COMPUTE_DTYPE = torch.float64  # offline ablation: fp64 for a clean oracle comparison

# The three integrators under test (name -> subspace-tracker fn). BUG is the
# incumbent; the metric depends only on span(U), returned identically for all.
Tracker = Callable[..., torch.Tensor]
INTEGRATORS: dict[str, Tracker] = {
    "BUG": blocked_bug_subspace,
    "PSI": psi_subspace,
    "parallel-BUG": parallel_bug_subspace,
}
STYLES = {  # (color-cycle marker/linestyle) per series for the figure
    "oracle": ("k", "o-"),
    "BUG": ("C0", "^-"),
    "PSI": ("C1", "s--"),
    "parallel-BUG": ("C2", "d:"),
}


def load_matrix(layer_path: Path, key: str) -> torch.Tensor:
    """Load one layer dump as the ``(512, T - N_SINK)`` feature-by-token matrix (fp64).

    ``key`` selects ``"K"`` (post-RoPE) or ``"K_pre"`` (pre-RoPE) from a
    ``--rope both`` dump; the first ``N_SINK`` sink-token columns are dropped
    (``docs/notes/conventions.md``).
    """
    blob = torch.load(layer_path, weights_only=False)
    k: torch.Tensor = blob[key].to(COMPUTE_DTYPE)  # (h, t, d)
    h, t, d = k.shape
    return k.permute(0, 2, 1).reshape(h * d, t)[:, N_SINK:].contiguous()


def rel_error(m: torch.Tensor, subspace: torch.Tensor) -> float:
    """Relative Frobenius error ``|| M - U U^T M ||_F / || M ||_F`` of a subspace."""
    recon = subspace @ (subspace.mT @ m)
    return float(torch.linalg.norm(m - recon) / torch.linalg.norm(m))


def oracle_error(m: torch.Tensor, r: int) -> float:
    """Eckart--Young lower bound: relative error of the best rank-``r`` recon."""
    u, s, vt = torch.linalg.svd(m, full_matrices=False)
    recon = (u[:, :r] * s[:r]) @ vt[:r]
    return float(torch.linalg.norm(m - recon) / torch.linalg.norm(m))


def run_errors(
    mats: list[torch.Tensor], ranks: list[int], block_size: int
) -> dict[str, list[list[float]]]:
    """Per-integrator (and oracle) rel-error, shape [rank][doc]. NaN if a step fails."""
    out: dict[str, list[list[float]]] = {name: [] for name in ["oracle", *INTEGRATORS]}
    for r in ranks:
        out["oracle"].append([oracle_error(m, r) for m in mats])
        for name, fn in INTEGRATORS.items():
            row: list[float] = []
            for m in mats:
                try:
                    u = fn(m, r, block_size=block_size, compute_dtype=COMPUTE_DTYPE)
                    row.append(rel_error(m, u))
                except RuntimeError:  # e.g. parallel-BUG per-block SVD non-convergence (LAPACK)
                    row.append(float("nan"))
            out[name].append(row)
        agg = ", ".join(f"{n}={np.nanmean(out[n][-1]):.4f}" for n in ["oracle", *INTEGRATORS])
        print(f"  r={r:>4}: {agg}")
    return out


def run_costs(
    mat: torch.Tensor, ranks: list[int], block_size: int, repeats: int = 3
) -> dict[str, list[float]]:
    """Median wall-clock (ms) of one full streaming sweep per integrator, per rank.

    Same harness for all three, so this is a fair *relative* per-sweep cost on the
    dev CPU. (Numpy per-token loops are Python-overhead bound -- see
    ``[[bugpress-numpy-bottleneck]]`` -- but these blocked torch sweeps do a few
    large ops per block, so the timing reflects the linear algebra.)
    """
    costs: dict[str, list[float]] = {name: [] for name in INTEGRATORS}
    for r in ranks:
        for name, fn in INTEGRATORS.items():
            times: list[float] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                try:
                    fn(mat, r, block_size=block_size, compute_dtype=COMPUTE_DTYPE)
                    times.append((time.perf_counter() - t0) * 1e3)
                except RuntimeError:  # SVD non-convergence -> record NaN
                    times.append(float("nan"))
            costs[name].append(float(np.median(times)))
    return costs


def choose_winner(errors: dict[str, list[list[float]]], ranks: list[int]) -> dict[str, object]:
    """Apply the plan's decision rule from the mean error curves; return a verdict dict.

    A rival displaces BUG only if it *both* (1) matches BUG in the operating range
    (ranks <= 128, the useful KV compression band; mean ratio-to-BUG <= 1.02) and
    (2) stays **robust across the whole swept range** -- max error/oracle within
    ROBUST_BOUND, the BUG-like robustness that is the reason to prefer a DLRA
    integrator at all. Matching in the easy regime but destabilizing when pushed
    (the projector-splitting backward step) does *not* qualify. Otherwise BUG wins.
    """
    robust_bound = 1.05  # max error/oracle a "robust" integrator may reach anywhere

    def mean_curve(name: str) -> np.ndarray:
        return np.array([float(np.nanmean(row)) for row in errors[name]])

    bug = mean_curve("BUG")
    op = np.array([r <= 128 for r in ranks])
    summary: dict[str, dict[str, object]] = {}
    for name in INTEGRATORS:
        cur = mean_curve(name)
        ratio_op = float(np.nanmean((cur / bug)[op])) if op.any() else float("nan")
        max_ratio_oracle = float(np.nanmax(cur / mean_curve("oracle")))
        matches_op = ratio_op <= 1.02
        robust = max_ratio_oracle <= robust_bound
        summary[name] = {
            "mean_ratio_to_BUG_op_range": ratio_op,
            "max_ratio_to_oracle_all_ranks": max_ratio_oracle,
            "matches_BUG_op_range": bool(matches_op),
            "robust_all_ranks": bool(robust and bool(np.isfinite(cur).all())),
        }
    # BUG is the incumbent; a rival wins only if it matches BUG in the op range AND
    # is robust everywhere (the plan then prefers it only if also cheaper).
    rivals = [
        n
        for n in ("parallel-BUG", "PSI")
        if summary[n]["matches_BUG_op_range"] and summary[n]["robust_all_ranks"]
    ]
    winner = "BUG" if not rivals else rivals[0]
    return {
        "winner": winner,
        "per_integrator": summary,
        "operating_range_max_rank": 128,
        "robust_bound": robust_bound,
    }


def _plot(
    errors: dict[str, list[list[float]]],
    costs: dict[str, list[float]],
    ranks: list[int],
    out_path: Path,
) -> None:
    """Three panels: error vs rank, error-ratio-to-oracle vs rank, cost vs rank."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    x = np.array(ranks)

    # Panel A: reconstruction error vs rank (log-log), oracle + 3 integrators, bands.
    ax = axes[0]
    for name in ["oracle", *INTEGRATORS]:
        color, style = STYLES[name]
        mean = np.array([float(np.nanmean(row)) for row in errors[name]])
        std = np.array([float(np.nanstd(row)) for row in errors[name]])
        ax.plot(x, mean, style, color=color, label=name, lw=1.8, ms=5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rank r")
    ax.set_ylabel("rel. Frobenius reconstruction error")
    ax.set_title("(a) error vs rank")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    # Panel B: ratio to oracle (the small gaps + PSI's tail, on a linear axis).
    ax = axes[1]
    oracle = np.array([float(np.nanmean(row)) for row in errors["oracle"]])
    for name in INTEGRATORS:
        color, style = STYLES[name]
        mean = np.array([float(np.nanmean(row)) for row in errors[name]])
        ax.plot(x, mean / oracle, style, color=color, label=name, lw=1.8, ms=5)
    ax.axhline(1.0, color="k", lw=0.8, ls="-", alpha=0.6)
    ax.axvline(128, color="grey", lw=0.8, ls=":", alpha=0.7)
    ax.text(128, ax.get_ylim()[1], " KV op. range", color="grey", va="top", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("rank r")
    ax.set_ylabel("error / oracle  (1.0 = optimal)")
    ax.set_title("(b) gap to oracle")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    # Panel C: per-sweep wall-clock cost vs rank.
    ax = axes[2]
    for name in INTEGRATORS:
        color, style = STYLES[name]
        ax.plot(x, costs[name], style, color=color, label=name, lw=1.8, ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("rank r")
    ax.set_ylabel("cost: one streaming sweep (ms, dev CPU)")
    ax.set_title("(c) per-sweep cost")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=140)
    print(f"wrote figure {out_path.with_suffix('.png')} / .pdf")


def main() -> None:
    """Run the ablation (or re-plot), write the sidecar, and print the verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="dumps/llama3.2-1b")
    parser.add_argument("--pattern", default="doc*_len4096_rope-both")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--key", default="K_pre", choices=["K", "K_pre"])
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64, 128, 192, 256, 384])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--out", default="figures/week5/integrator_ablation.png")
    parser.add_argument("--plot-only", action="store_true", help="re-render from the JSON sidecar")
    args = parser.parse_args()

    out_path = Path(args.out)
    json_path = out_path.with_suffix(".json")

    if args.plot_only:
        blob = json.loads(json_path.read_text())
        _plot(blob["errors"], blob["costs"], blob["ranks"], out_path)
        return

    dump_dirs = sorted(Path(args.dumps).glob(args.pattern))
    layer_paths = [d / f"layer_{args.layer:02d}.pt" for d in dump_dirs]
    layer_paths = [p for p in layer_paths if p.exists()]
    if not layer_paths:
        raise SystemExit(f"no dumps matched {args.dumps}/{args.pattern} (layer {args.layer})")
    mats = [load_matrix(p, args.key) for p in layer_paths]
    ranks = [r for r in args.ranks if r < min(m.shape[0] for m in mats)]
    print(
        f"{len(mats)} docs, layer {args.layer}, key {args.key}, "
        f"block_size {args.block_size}, ranks {ranks}"
    )

    errors = run_errors(mats, ranks, args.block_size)
    print("timing per-sweep cost on the largest doc ...")
    biggest = max(mats, key=lambda m: m.shape[1])
    costs = run_costs(biggest, ranks, args.block_size)
    verdict = choose_winner(errors, ranks)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "layer": args.layer,
                    "key": args.key,
                    "block_size": args.block_size,
                    "n_docs": len(mats),
                    "compute_dtype": str(COMPUTE_DTYPE),
                },
                "ranks": ranks,
                "errors": errors,
                "costs": costs,
                "verdict": verdict,
            },
            indent=2,
        )
        + "\n"
    )
    _plot(errors, costs, ranks, out_path)

    print("\n=== VERDICT (docs/week5-plan.md §4.5 decision rule) ===")
    for name, s in cast(dict[str, dict[str, object]], verdict["per_integrator"]).items():
        print(
            f"  {name:>13}: mean err/BUG (r<=128) = {s['mean_ratio_to_BUG_op_range']:.4f}, "
            f"max err/oracle (all r) = {s['max_ratio_to_oracle_all_ranks']:.4f}, "
            f"op-match = {s['matches_BUG_op_range']}, robust = {s['robust_all_ranks']}"
        )
    print(f"  WINNER -> {verdict['winner']}  (carried into Week 5)")


if __name__ == "__main__":
    main()
