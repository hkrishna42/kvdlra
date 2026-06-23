"""Week-2 streaming rank-adaptive BUG: validate against the truncated-SVD oracle.

Streams the columns (tokens) of Layer-8 of the Llama-3.2-1B K-cache through
:class:`kvdlra.integrators.streaming.StreamingBUG` and answers the Week-2 pilot
question (``docs/PLAN.md`` §5): is the *streaming* rank-adaptive BUG tracker's
final reconstruction error vs. rank competitive with the truncated-SVD oracle and
no worse than incremental SVD?

It does three things:

1. **Validation table.** At ranks ``{8, 16, 32, 64, 128}`` it tabulates the
   final relative-Frobenius reconstruction error of

       * ``StreamingBUG`` (this project's streaming tracker, ``rank_cap = r``),
       * truncated SVD (the Eckart--Young oracle / lower bound), and
       * Brand-style incremental SVD (the naive streaming baseline),

   together with the BUG/oracle ratio, and prints it.

2. **Mean rank vs. token index.** It streams once with an adaptive tolerance
   ``theta`` and records the tracked rank after every token, then plots the
   running mean rank against the token index and saves it to ``figures/week2/``.

3. **Metrics JSON.** Everything above (and the run configuration) is written to a
   sidecar JSON so the numbers are auditable without re-running the model (the
   raw KV dump is gitignored).

Matrix convention (``docs/notes/conventions.md``): from a per-layer K tensor of
shape ``(H, T, D) = (8, T, 64)`` we build ``M = K.transpose(0, 2, 1).reshape(H*D,
T)`` -- rows = 512 features, columns = T tokens -- and drop the first ``n_sink =
4`` sink-token columns (StreamingLLM; PLAN §8 pitfall #5), giving ``M = (512,
2044)`` for the 2048-token dump.

The cached keys are **post-RoPE** (``docs/notes/rope-pitfall.md``); a shallow
spectrum is expected and is that pitfall, not a tracker failure. The point of
this experiment is the *relative* gap between the streaming tracker and the
oracle, which is RoPE-independent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
import torch

matplotlib.use("Agg")  # headless / CPU-safe backend; set before pyplot import
import matplotlib.pyplot as plt

# --- bootstrap: make the in-repo kvdlra package importable when run directly ---
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kvdlra.integrators.streaming import StreamingBUG  # noqa: E402

# Number of leading attention-sink tokens to drop (StreamingLLM; PLAN §8 #5).
N_SINK = 4
# Ranks at which the validation table is tabulated (PLAN §5 pilot grid).
PILOT_RANKS = [8, 16, 32, 64, 128]


def load_layer_matrix(dump_dir: Path, layer: int) -> npt.NDArray[np.float64]:
    """Load one layer dump and build the ``(512, T - N_SINK)`` feature-by-token matrix."""
    blob = torch.load(dump_dir / f"layer_{layer:02d}.pt", weights_only=False)
    k = blob["K"].float().numpy()  # (H, T, D)
    h, t, d = k.shape
    m = k.transpose(0, 2, 1).reshape(h * d, t)  # rows = features, cols = tokens
    if m.shape[1] > N_SINK:
        m = m[:, N_SINK:]
    return np.ascontiguousarray(m, dtype=np.float64)


def truncated_svd_error(m: npt.NDArray[np.float64], r: int) -> float:
    """Relative Frobenius error of the best rank-``r`` (truncated-SVD) recon (oracle)."""
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    mr = (u[:, :r] * s[:r]) @ vt[:r]
    return float(np.linalg.norm(m - mr) / np.linalg.norm(m))


def incremental_svd_error(m: npt.NDArray[np.float64], r: int) -> float:
    """Brand-style incremental SVD: stream columns of ``m``, keep top-``r`` (baseline)."""
    u: npt.NDArray[np.float64] = np.zeros((m.shape[0], 0))
    s: npt.NDArray[np.float64] = np.zeros(0)
    vt: npt.NDArray[np.float64] = np.zeros((0, 0))
    for j in range(m.shape[1]):
        c = m[:, j : j + 1]
        if u.size == 0:
            u, sj, vtj = np.linalg.svd(c, full_matrices=False)
            s = sj
            vt = np.zeros((sj.shape[0], 1))
            vt[:, -1] = vtj[:, 0]
            continue
        proj = u.T @ c
        resid = c - u @ proj
        resid_norm = float(np.linalg.norm(resid))
        resid_unit = resid / (resid_norm + 1e-12)
        core = np.block([[np.diag(s), proj], [np.zeros((1, s.size)), np.array([[resid_norm]])]])
        up, sp, vtp = np.linalg.svd(core, full_matrices=False)
        u = np.hstack([u, resid_unit]) @ up
        s = sp
        vt_ext = np.block(
            [
                [vt, np.zeros((vt.shape[0], 1))],
                [np.zeros((1, vt.shape[1])), np.array([[1.0]])],
            ]
        )
        vt = vtp @ vt_ext
        if s.size > r:
            u, s, vt = u[:, :r], s[:r], vt[:r, :]
    mr = (u * s) @ vt
    return float(np.linalg.norm(m - mr) / np.linalg.norm(m))


def streaming_bug_error(m: npt.NDArray[np.float64], r: int) -> float:
    """Stream ``m`` through :class:`StreamingBUG` at ``rank_cap = r``; return rel. error."""
    tracker = StreamingBUG(n_features=m.shape[0], rank_cap=r)
    tracker.update_many(m)
    return tracker.reconstruction_error(m)


def build_validation_table(m: npt.NDArray[np.float64], ranks: list[int]) -> list[dict[str, float]]:
    """Compute (and return) the BUG vs. oracle vs. iSVD error table over ``ranks``."""
    rows: list[dict[str, float]] = []
    for r in ranks:
        if r >= min(m.shape):
            continue
        e_oracle = truncated_svd_error(m, r)
        e_isvd = incremental_svd_error(m, r)
        e_bug = streaming_bug_error(m, r)
        rows.append(
            {
                "rank": float(r),
                "bug": e_bug,
                "oracle": e_oracle,
                "isvd": e_isvd,
                "bug_over_oracle": e_bug / e_oracle if e_oracle > 0 else float("nan"),
            }
        )
    return rows


def print_validation_table(rows: list[dict[str, float]]) -> None:
    """Pretty-print the validation table to stdout."""
    print("\nValidation: final rel-Frobenius reconstruction error vs. rank (Layer 8)")
    print(f"{'rank':>6} | {'BUG':>10} | {'oracle SVD':>10} | {'incr SVD':>10} | {'BUG/oracle':>10}")
    print("-" * 62)
    for row in rows:
        print(
            f"{int(row['rank']):>6} | {row['bug']:>10.4e} | {row['oracle']:>10.4e} | "
            f"{row['isvd']:>10.4e} | {row['bug_over_oracle']:>10.3f}"
        )


def streaming_rank_trajectory(
    m: npt.NDArray[np.float64], theta: float, rank_cap: int
) -> tuple[list[int], float]:
    """Stream once with adaptive ``theta``; return the per-token rank history + final error."""
    tracker = StreamingBUG(n_features=m.shape[0], theta=theta, rank_cap=rank_cap)
    tracker.update_many(m)
    return tracker.rank_history, tracker.reconstruction_error(m)


def plot_mean_rank(rank_history: list[int], out_path: Path, theta: float) -> None:
    """Plot running mean rank (and instantaneous rank) vs. token index; save to ``out_path``."""
    tokens = np.arange(1, len(rank_history) + 1)
    inst = np.asarray(rank_history, dtype=np.float64)
    running_mean = np.cumsum(inst) / tokens

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tokens, inst, color="0.7", lw=0.8, label="instantaneous rank")
    ax.plot(tokens, running_mean, color="C0", lw=2.0, label="running mean rank")
    ax.set_xlabel("token index")
    ax.set_ylabel("tracked rank")
    ax.set_title(rf"Streaming BUG rank vs. token (Layer 8, $\vartheta$={theta:g})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main() -> None:
    """Parse arguments, run the validation + rank-trajectory, save the figure + JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump",
        default="dumps/llama3.2-1b/doc15_d1b3a6eb_len2048",
        help="dir of layer_XX.pt files",
    )
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=PILOT_RANKS, help="rank grid for the table"
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=20.0,
        help="adaptive Frobenius-tail tolerance for the rank-trajectory plot",
    )
    parser.add_argument(
        "--theta-rank-cap",
        type=int,
        default=128,
        help="rank cap applied alongside theta in the trajectory run",
    )
    parser.add_argument("--fig", default="figures/week2/streaming_bug_rank.pdf")
    parser.add_argument("--metrics", default="figures/week2/streaming_bug_metrics.json")
    args = parser.parse_args()

    dump_dir = Path(args.dump)
    m = load_layer_matrix(dump_dir, args.layer)
    print(
        f"Layer {args.layer}: M.shape={m.shape} "
        f"(rows=features, cols=tokens; dropped {N_SINK} sinks)"
    )

    # --- 1. Validation table ---
    t0 = time.perf_counter()
    rows = build_validation_table(m, list(args.ranks))
    print_validation_table(rows)
    print(f"\n(table computed in {time.perf_counter() - t0:.1f}s)")

    # --- 2. Mean-rank trajectory with adaptive theta ---
    rank_history, theta_err = streaming_rank_trajectory(m, args.theta, args.theta_rank_cap)
    mean_rank = float(np.mean(rank_history))
    print(
        f"\nAdaptive run (theta={args.theta:g}, cap={args.theta_rank_cap}): "
        f"final_rank={rank_history[-1]}, mean_rank={mean_rank:.1f}, rel_err={theta_err:.4e}"
    )

    fig_path = Path(args.fig)
    plot_mean_rank(rank_history, fig_path, args.theta)
    print(f"wrote figure {fig_path} (+ .png)")

    # --- 3. Verdict + metrics JSON ---
    # Pilot-style check on the printed table: BUG within ~1.5x of the oracle and
    # no worse than incremental SVD, at every tabulated rank.
    within_oracle = all(row["bug_over_oracle"] <= 1.5 for row in rows)
    le_isvd = all(row["bug"] <= row["isvd"] + 1e-6 for row in rows)
    verdict = bool(within_oracle and le_isvd)
    print(
        f"\nVERDICT: BUG within 1.5x oracle at every rank: {within_oracle}; "
        f"BUG <= incremental SVD at every rank: {le_isvd}  =>  competitive: {verdict}"
    )

    metrics: dict[str, object] = {
        "dump": str(dump_dir),
        "layer": args.layer,
        "matrix_shape": list(m.shape),
        "n_sink_dropped": N_SINK,
        "table": rows,
        "adaptive": {
            "theta": args.theta,
            "rank_cap": args.theta_rank_cap,
            "final_rank": rank_history[-1],
            "mean_rank": mean_rank,
            "rel_err": theta_err,
            "rank_history": rank_history,
        },
        "verdict": {
            "bug_within_1p5x_oracle": within_oracle,
            "bug_le_incremental_svd": le_isvd,
            "competitive": verdict,
        },
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote metrics {metrics_path}")


if __name__ == "__main__":
    main()
