"""Week-4.5 DLRA integrator ablation (offline; no LLM, no pod).

Bakes off the three streaming DLRA integrators as KV subspace trackers on captured
pre-RoPE KV -- reconstruction error vs the truncated-SVD **oracle**, per-sweep
**cost**, and streaming **rank-adaptivity** -- and picks ONE winner to carry into
the Week-5 comparisons (``docs/week5-plan.md`` §"Week 4.5"). All three run through
the identical block-streaming harness (same first-block seed, same rank
truncation, same ``|| M - U U^T M ||_F / || M ||_F`` metric, which depends only on
``span(U)``) and differ only in the per-block step:

* **BUG** -- :func:`kvdlra.integrators.streaming_torch.blocked_bug_subspace`
  (augmented Galerkin, forward, square-root core). The incumbent.
* **PSI** -- :func:`kvdlra.integrators.streaming_variants.psi_subspace`
  (projector-splitting, backward S-step, covariance core) -- **fixed-rank**.
* **Parallel-BUG** -- :func:`kvdlra.integrators.streaming_variants.parallel_bug_subspace`
  (decoupled in-basis/new-direction update).

**Fair regime (important).** For the accuracy comparison every method must
actually reach the requested rank, and the incumbent BUG's augmented ``[U | Q]``
basis must fit in ``R^n`` -- i.e. ``rank <= block_size`` (else fixed-rank PSI is
capped below the target and the comparison is apples-to-oranges) **and**
``rank + block_size <= n_features`` (else BUG's augmented basis is rank-deficient
and degenerates). Both hold for the KV operating range ``rank <= 128`` at
``block_size = 128`` (``128 + 128 < 512``), which is the band that matters. A
separate **rank-adaptivity** probe deliberately pushes ``rank_cap`` past
``block_size`` to expose PSI's fixed-rank limitation.

Writes ``figures/week5/integrator_ablation.{png,pdf}`` + a ``.json`` sidecar and
prints the verdict. Use ``--plot-only`` to re-render from the sidecar.
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
STYLES = {  # (color, marker/linestyle) per series for the figure
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
    """Per-integrator (and oracle) rel-error, shape [rank][doc]. Also asserts the
    fair regime (every method reaches the rank; BUG's augmented basis fits in R^n)."""
    n = mats[0].shape[0]
    out: dict[str, list[list[float]]] = {name: [] for name in ["oracle", *INTEGRATORS]}
    for r in ranks:
        if r > block_size or r + block_size > n:
            raise SystemExit(
                f"unfair rank r={r} at block_size={block_size} (need r<=block_size and "
                f"r+block_size<=n_features={n}); restrict --ranks or lower --block-size"
            )
        out["oracle"].append([oracle_error(m, r) for m in mats])
        for name, fn in INTEGRATORS.items():
            row: list[float] = []
            for m in mats:
                u = fn(m, r, block_size=block_size, compute_dtype=COMPUTE_DTYPE)
                assert u.shape[1] == r, f"{name} reached rank {u.shape[1]} != {r}"  # fair regime
                row.append(rel_error(m, u))
            out[name].append(row)
        agg = ", ".join(f"{n2}={np.nanmean(out[n2][-1]):.4f}" for n2 in ["oracle", *INTEGRATORS])
        print(f"  r={r:>4}: {agg}")
    return out


def run_adaptivity(
    mat: torch.Tensor, rank_caps: list[int], block_size: int
) -> dict[str, list[int]]:
    """Achieved rank (basis columns) vs requested rank_cap at a streaming block size.

    Pushes rank_cap past ``block_size`` to expose which integrators are
    rank-adaptive (BUG, parallel-BUG grow to rank_cap) vs fixed-rank (PSI caps at
    ``block_size``). Ranks with ``rank_cap + block_size > n_features`` are skipped
    for BUG-family degeneracy but still probed for PSI (which stays <= block_size).
    """
    achieved: dict[str, list[int]] = {name: [] for name in INTEGRATORS}
    for rc in rank_caps:
        for name, fn in INTEGRATORS.items():
            u = fn(mat, rc, block_size=block_size, compute_dtype=COMPUTE_DTYPE)
            achieved[name].append(int(u.shape[1]))
    return achieved


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
                fn(mat, r, block_size=block_size, compute_dtype=COMPUTE_DTYPE)
                times.append((time.perf_counter() - t0) * 1e3)
            costs[name].append(float(np.median(times)))
    return costs


def choose_winner(
    errors: dict[str, list[list[float]]],
    adaptivity: dict[str, list[int]],
    rank_caps: list[int],
    block_size: int,
) -> dict[str, object]:
    """Pick the integrator to carry into Week 5 from accuracy + rank-adaptivity.

    In the fair operating range all three cluster near the oracle (a rigor result,
    not a competitive lever). BUG is chosen because it is (1) at least as accurate
    as every rival there **and** (2) rank-adaptive at a streaming block size, which
    fixed-rank PSI is not. Parallel-BUG is rank-adaptive but measurably less
    accurate. This is a design-validation (near-tie) verdict, NOT a claim that any
    rival numerically blows up.
    """

    def mean_curve(name: str) -> np.ndarray:
        return np.array([float(np.nanmean(row)) for row in errors[name]])

    oracle = mean_curve("oracle")
    summary: dict[str, object] = {}
    for name in INTEGRATORS:
        cur = mean_curve(name)
        # rank-adaptive := reaches a rank_cap strictly greater than block_size.
        adaptive = any(
            rc > block_size and ach > block_size
            for rc, ach in zip(rank_caps, adaptivity[name], strict=True)
        )
        summary[name] = {
            "mean_err_over_oracle": float(np.nanmean(cur / oracle)),
            "max_err_over_oracle": float(np.nanmax(cur / oracle)),
            "rank_adaptive": bool(adaptive),
        }
    # Winner: accurate in the operating range (err/oracle <= 1.05) AND rank-adaptive.
    # Among such, prefer the most accurate. BUG and parallel-BUG are adaptive; PSI
    # is not. BUG is the most accurate adaptive method -> winner.
    accurate_adaptive = [
        n
        for n in INTEGRATORS
        if cast(dict[str, object], summary[n])["rank_adaptive"]
        and cast(float, cast(dict[str, object], summary[n])["max_err_over_oracle"]) <= 1.05
    ]
    winner = min(
        accurate_adaptive or list(INTEGRATORS),
        key=lambda n: cast(float, cast(dict[str, object], summary[n])["mean_err_over_oracle"]),
    )
    return {"winner": winner, "per_integrator": summary, "block_size": block_size}


def _plot(
    errors: dict[str, list[list[float]]],
    costs: dict[str, list[float]],
    adaptivity: dict[str, list[int]],
    ranks: list[int],
    rank_caps: list[int],
    block_size: int,
    out_path: Path,
) -> None:
    """Three panels: error vs rank (fair), streaming rank-adaptivity, per-sweep cost."""
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
    ax.set_xlabel("rank r  (fair: r <= block_size, all methods reach r)")
    ax.set_ylabel("rel. Frobenius reconstruction error")
    ax.set_title("(a) accuracy vs rank — near-tie")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    # Panel B: streaming rank-adaptivity — achieved rank vs requested rank_cap.
    ax = axes[1]
    xc = np.array(rank_caps)
    ax.plot(xc, xc, "-", color="grey", lw=1, alpha=0.6, label="requested (ideal)")
    for name in INTEGRATORS:
        color, style = STYLES[name]
        ax.plot(xc, adaptivity[name], style, color=color, label=name, lw=1.8, ms=6)
    ax.axvline(block_size, color="grey", lw=0.8, ls=":", alpha=0.7)
    ax.text(block_size, ax.get_ylim()[1], " block_size", color="grey", va="top", fontsize=8)
    ax.set_xlabel("requested rank_cap")
    ax.set_ylabel(f"achieved rank (block_size={block_size})")
    ax.set_title("(b) streaming rank-adaptivity — PSI is fixed-rank")
    ax.grid(True, alpha=0.3)
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
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--rank-caps",
        type=int,
        nargs="+",
        default=[64, 128, 192, 256, 384],
        help="rank_caps for the rank-adaptivity probe (some > block_size, to expose PSI's cap)",
    )
    parser.add_argument("--out", default="figures/week5/integrator_ablation.png")
    parser.add_argument("--plot-only", action="store_true", help="re-render from the JSON sidecar")
    args = parser.parse_args()

    out_path = Path(args.out)
    json_path = out_path.with_suffix(".json")

    if args.plot_only:
        blob = json.loads(json_path.read_text())
        _plot(
            blob["errors"],
            blob["costs"],
            blob["adaptivity"],
            blob["ranks"],
            blob["rank_caps"],
            blob["config"]["block_size"],
            out_path,
        )
        return

    dump_dirs = sorted(Path(args.dumps).glob(args.pattern))
    layer_paths = [d / f"layer_{args.layer:02d}.pt" for d in dump_dirs]
    layer_paths = [p for p in layer_paths if p.exists()]
    if not layer_paths:
        raise SystemExit(f"no dumps matched {args.dumps}/{args.pattern} (layer {args.layer})")
    mats = [load_matrix(p, args.key) for p in layer_paths]
    ranks = [r for r in args.ranks if r <= args.block_size]
    print(
        f"{len(mats)} docs, layer {args.layer}, key {args.key}, block_size {args.block_size}, "
        f"ranks {ranks} (fair), rank_caps {args.rank_caps} (adaptivity)"
    )

    errors = run_errors(mats, ranks, args.block_size)
    biggest = max(mats, key=lambda m: m.shape[1])
    print("streaming rank-adaptivity probe ...")
    adaptivity = run_adaptivity(biggest, args.rank_caps, args.block_size)
    for name, ach in adaptivity.items():
        print(f"  {name:>13}: achieved {ach} for rank_caps {args.rank_caps}")
    print("timing per-sweep cost ...")
    costs = run_costs(biggest, ranks, args.block_size)
    verdict = choose_winner(errors, adaptivity, args.rank_caps, args.block_size)

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
                "rank_caps": args.rank_caps,
                "errors": errors,
                "costs": costs,
                "adaptivity": adaptivity,
                "verdict": verdict,
            },
            indent=2,
        )
        + "\n"
    )
    _plot(errors, costs, adaptivity, ranks, args.rank_caps, args.block_size, out_path)

    print("\n=== VERDICT (docs/week5-plan.md §4.5) ===")
    for name, s in cast(dict[str, dict[str, object]], verdict["per_integrator"]).items():
        print(
            f"  {name:>13}: mean err/oracle = {s['mean_err_over_oracle']:.4f}, "
            f"max err/oracle = {s['max_err_over_oracle']:.4f}, "
            f"rank-adaptive = {s['rank_adaptive']}"
        )
    print(f"  WINNER -> {verdict['winner']}  (carried into Week 5)")


if __name__ == "__main__":
    main()
