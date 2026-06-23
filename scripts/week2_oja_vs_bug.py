"""Week-2 Thursday deliverable: BUG vs Oja vs SVD-oracle at matched memory.

Compares three left-subspace trackers on Layer-8 K of Llama-3.2-1B, across the 5
pilot C4 documents (``dumps/llama3.2-1b/doc*_len4096_rope-both``), at ranks
``r = 16, 32, 64, 128``:

* **truncated SVD (oracle)** -- :func:`kvdlra.lowrank.truncated_svd_recon`, the
  Eckart--Young lower bound (the best any rank-``r`` projector can do).
* **streaming BUG** -- :class:`kvdlra.integrators.streaming.StreamingBUG` with
  ``rank_cap = r``, the augmented rank-adaptive BUG tracker.
* **Oja** -- :class:`kvdlra.integrators.oja.OjaTracker` with ``rank = r`` and a
  **fairly tuned** learning rate (see "Fair Oja tuning" below).

All three are scored on the **same** metric, the relative Frobenius
reconstruction error of the orthogonal projection onto the tracked basis,
``||M - U U^T M||_F / ||M||_F`` (identical in ``StreamingBUG`` and ``OjaTracker``;
the oracle is the same error for the truncated-SVD basis). Output: a printed
table, a grouped bar chart ``figures/week2/oja_vs_bug.{pdf,png}`` (groups = ranks;
bars = BUG / Oja / oracle), and a metrics JSON sidecar.

Fair, matched-memory, matched-compute comparison
------------------------------------------------
*Matched memory.* At rank ``r`` all three keep an identical ``512 x r`` basis, so
"rank" is the memory axis and the comparison is apples-to-apples per rank.

*Matched compute (single streaming pass).* BUG and Oja each see every token
**exactly once, in order** -- the honest online setting. We do NOT give Oja extra
epochs over the stream (multi-pass Oja would not be a streaming method and would
be an unfair advantage over single-pass BUG). The oracle, by definition, sees the
whole matrix at once; it is the lower bound, not a streaming competitor.

*Fair Oja tuning.* Oja's rule is famously step-size sensitive, so per rope mode we
sweep a grid of ``(eta0, decay)`` for the schedule ``eta_t = eta0/(1 + decay*t)``
(with per-step token normalization; see :mod:`kvdlra.integrators.oja`) and pick the
configuration with the lowest **mean-over-docs** error -- i.e. Oja is shown at its
single-pass best, tuned on the same data it is scored on (a generous,
in-favour-of-the-baseline protocol). The schedule is tuned once at a cheap
reference rank (:data:`OJA_TUNE_RANK`) and reused at every rank: the normalized-Oja
optimum is essentially rank-independent (verified on this data, <0.01 error
penalty), and tuning at the small rank avoids 100+ full-stream Oja fits at r=128.
The chosen ``(eta0, decay)`` is printed and saved in the JSON.

Honest verdict. The script prints, per rank, whether BUG beats Oja and by how much
(ratio ``oja/bug`` and gap to the oracle), and a one-line overall verdict. If Oja
matches or beats BUG, that is reported as-is.
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

from kvdlra.integrators.oja import OjaTracker
from kvdlra.integrators.streaming import StreamingBUG
from kvdlra.lowrank import truncated_svd_recon

matplotlib.use("Agg")  # headless / CPU-safe backend; set before pyplot import
import matplotlib.pyplot as plt

N_SINK = 4  # StreamingLLM sink tokens dropped from every matrix (PLAN §8 #5)
RANKS: list[int] = [16, 32, 64, 128]
ROPE_MODES: list[tuple[str, str]] = [("post-RoPE", "K"), ("pre-RoPE", "K_pre")]

# Fair single-pass Oja tuning grid for the schedule eta_t = eta0 / (1 + decay*t),
# with per-step token normalization (OjaTracker default). The grid brackets the
# single-pass optimum found on real Layer-8 K: post-RoPE error is nearly flat in
# the step size (Oja saturates around its best regardless), while pre-RoPE prefers
# the larger sustained steps of a small decay -- both regimes are covered.
ETA0_GRID: list[float] = [1.0, 2.0, 5.0, 10.0, 20.0]
DECAY_GRID: list[float] = [1e-3, 3e-3, 1e-2, 3e-2]

# Rank at which the (eta0, decay) schedule is tuned. The normalized-Oja optimum is
# essentially rank-independent (the step size tracks the data's gradient scale, not
# the basis width), so we tune once at the cheapest rank and reuse the winning
# schedule at every rank -- verified on Layer-8 K, where the r=16-tuned schedule
# costs <0.01 extra reconstruction error at r=64 vs re-tuning there. Tuning at the
# small rank avoids 100+ full-stream Oja fits at r=128 (each a per-token 512xr QR).
OJA_TUNE_RANK = 16


def load_matrix(layer_path: Path, key: str) -> npt.NDArray[np.float64]:
    """Load one layer dump and build the ``(512, T - N_SINK)`` feature-by-token matrix.

    ``key`` selects ``"K"`` (post-RoPE) or ``"K_pre"`` (pre-RoPE) from a
    ``--rope both`` dump. The first ``N_SINK`` sink-token columns are dropped.
    Mirrors ``scripts/week2_pilot.py`` exactly (512 features, drop 4 sinks).
    """
    blob = torch.load(layer_path, weights_only=False)
    k = blob[key].float().numpy()  # (h, t, d)
    h, t, d = k.shape
    m = k.transpose(0, 2, 1).reshape(h * d, t)[:, N_SINK:]
    return np.ascontiguousarray(m, dtype=np.float64)


def bug_error(m: npt.NDArray[np.float64], r: int) -> float:
    """Single streaming pass of ``m``'s columns through StreamingBUG (rank cap ``r``)."""
    tracker = StreamingBUG(m.shape[0], rank_cap=r)
    tracker.update_many(m)
    return tracker.reconstruction_error(m)


def oja_error(m: npt.NDArray[np.float64], r: int, eta0: float, decay: float) -> float:
    """Single streaming pass of ``m`` through OjaTracker at ``(r, eta0, decay)``."""
    tracker = OjaTracker(m.shape[0], rank=r, eta0=eta0, decay=decay, normalize=True)
    tracker.update_many(m)
    return tracker.reconstruction_error(m)


def tune_oja_schedule(mats: list[npt.NDArray[np.float64]], tune_rank: int) -> tuple[float, float]:
    """Pick the ``(eta0, decay)`` minimizing mean-over-docs Oja error at ``tune_rank``.

    Sweeps :data:`ETA0_GRID` x :data:`DECAY_GRID` at the (cheap) reference rank and
    returns the best schedule. This tunes Oja *in its own favour* on the evaluation
    data, so the reported Oja curve is a single-pass best case for the baseline; the
    schedule is reused at every rank (see :data:`OJA_TUNE_RANK`).
    """
    best_mean = np.inf
    best: tuple[float, float] = (ETA0_GRID[0], DECAY_GRID[0])
    for eta0 in ETA0_GRID:
        for decay in DECAY_GRID:
            mean = float(np.mean([oja_error(m, tune_rank, eta0, decay) for m in mats]))
            if mean < best_mean:
                best_mean = mean
                best = (eta0, decay)
    return best


def run_rope_mode(layer_paths: list[Path], key: str) -> list[dict[str, object]]:
    """Compute oracle / BUG / tuned-Oja per-doc errors for every rank, one rope mode."""
    mats = [load_matrix(p, key) for p in layer_paths]
    # Tune the Oja schedule once at the cheap reference rank, then reuse it (the
    # normalized-Oja optimum is essentially rank-independent; see OJA_TUNE_RANK).
    eta0, decay = tune_oja_schedule(mats, OJA_TUNE_RANK)
    rows: list[dict[str, object]] = []
    for r in RANKS:
        oracle = [truncated_svd_recon(m, r) for m in mats]
        bug = [bug_error(m, r) for m in mats]
        oja = [oja_error(m, r, eta0, decay) for m in mats]
        rows.append(
            {
                "rank": r,
                "oracle": oracle,
                "bug": bug,
                "oja": oja,
                "oja_eta0": eta0,
                "oja_decay": decay,
            }
        )
    return rows


def _mean(row: dict[str, object], field: str) -> float:
    """Mean over documents of ``row[field]`` (a per-doc list)."""
    return float(np.mean(cast("list[float]", row[field])))


def main() -> None:
    """Run the BUG-vs-Oja-vs-oracle comparison, print the table, plot, and the verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", default="dumps/llama3.2-1b")
    parser.add_argument("--pattern", default="doc*_len4096_rope-both")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--out", default="figures/week2/oja_vs_bug.pdf")
    args = parser.parse_args()

    dump_dirs = sorted(Path(args.dumps).glob(args.pattern))
    layer_paths = [d / f"layer_{args.layer:02d}.pt" for d in dump_dirs]
    layer_paths = [p for p in layer_paths if p.exists()]
    if not layer_paths:
        raise SystemExit(f"no dumps matched {args.dumps}/{args.pattern} (layer {args.layer})")
    n_docs = len(layer_paths)
    print(f"{n_docs} docs, layer {args.layer}, ranks {RANKS}")
    print(
        f"single streaming pass for BUG and Oja; Oja eta tuned per rope-mode "
        f"at r={OJA_TUNE_RANK} (fair, in-favour)\n"
    )

    results: dict[str, list[dict[str, object]]] = {}
    for rope, key in ROPE_MODES:
        rows = run_rope_mode(layer_paths, key)
        results[rope] = rows
        _print_table(rope, rows)

    verdict = _verdict(results)
    _plot(results, Path(args.out))

    out_json = Path(args.out).with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "layer": args.layer,
                "n_docs": n_docs,
                "ranks": RANKS,
                "single_pass": True,
                "oja_eta0_grid": ETA0_GRID,
                "oja_decay_grid": DECAY_GRID,
                "results": results,
                "verdict": verdict,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote metrics {out_json}")


def _print_table(rope: str, rows: list[dict[str, object]]) -> None:
    """Print the per-rank mean-over-docs table for one rope mode."""
    print(f"=== Layer-8 K, {rope}: mean rel. Frobenius error over docs ===")
    print(
        f"  {'rank':>4}  {'oracle':>7}  {'BUG':>7}  {'Oja':>7}  "
        f"{'oja/bug':>8}  {'bug/orc':>8}  {'oja/orc':>8}  {'eta0':>5}  {'decay':>6}"
    )
    for row in rows:
        r = cast(int, row["rank"])
        orc, bug, oja = _mean(row, "oracle"), _mean(row, "bug"), _mean(row, "oja")
        eta0, decay = cast(float, row["oja_eta0"]), cast(float, row["oja_decay"])
        print(
            f"  {r:>4}  {orc:>7.4f}  {bug:>7.4f}  {oja:>7.4f}  "
            f"{oja / bug:>8.3f}  {bug / orc:>8.3f}  {oja / orc:>8.3f}  "
            f"{eta0:>5.1f}  {decay:>6.0e}"
        )
    print()


def _verdict(results: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    """Honest verdict: does BUG beat Oja, and by how much, per rope mode and rank?"""
    print("=== VERDICT (single-pass, matched memory) ===")
    verdict: dict[str, object] = {}
    for rope, _key in ROPE_MODES:
        rows = results[rope]
        per_rank: list[dict[str, object]] = []
        bug_wins = 0
        for row in rows:
            r = cast(int, row["rank"])
            bug, oja = _mean(row, "bug"), _mean(row, "oja")
            bug_better = bug < oja
            bug_wins += int(bug_better)
            pct = 100.0 * (oja - bug) / oja if oja > 0 else 0.0
            per_rank.append(
                {
                    "rank": r,
                    "bug_mean": bug,
                    "oja_mean": oja,
                    "bug_beats_oja": bug_better,
                    "bug_lower_by_pct": pct,
                }
            )
            winner = "BUG" if bug_better else "Oja"
            print(
                f"  {rope} r={r:>3}: BUG={bug:.4f} Oja={oja:.4f} "
                f"-> {winner} better (BUG is {pct:+.1f}% rel. to Oja)"
            )
        overall_bug = bug_wins == len(rows)
        verdict[rope] = {"per_rank": per_rank, "bug_beats_oja_all_ranks": overall_bug}
        print(f"  {rope}: {'BUG beats Oja at every rank' if overall_bug else 'mixed'}\n")
    return verdict


def _plot(results: dict[str, list[dict[str, object]]], out_path: Path) -> None:
    """Grouped bar chart per rope mode: groups = ranks, bars = BUG / Oja / oracle."""
    fig, axes = plt.subplots(1, len(ROPE_MODES), figsize=(6 * len(ROPE_MODES), 4.5), squeeze=False)
    series = [
        ("bug", "streaming BUG", "#1f77b4"),
        ("oja", "Oja (tuned)", "#ff7f0e"),
        ("oracle", "truncated SVD (oracle)", "#2ca02c"),
    ]
    width = 0.26
    x = np.arange(len(RANKS))
    for col, (rope, _key) in enumerate(ROPE_MODES):
        ax = axes[0][col]
        rows = results[rope]
        for i, (field, label, color) in enumerate(series):
            means = np.array([_mean(r, field) for r in rows])
            stds = np.array([float(np.std(cast("list[float]", r[field]))) for r in rows])
            ax.bar(
                x + (i - 1) * width,
                means,
                width,
                yerr=stds,
                capsize=3,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in RANKS])
        ax.set_xlabel("rank (memory = 512 x rank)")
        ax.set_ylabel("rel. Frobenius reconstruction error")
        ax.set_title(f"Layer-8 K, {rope}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(out_path.with_suffix(suffix))
    print(f"\nwrote figure {out_path.with_suffix('.pdf')} / .png")


if __name__ == "__main__":
    main()
