"""Week-2 rank-policy comparison (``docs/PLAN.md`` Week-2 "Wed" row).

Compares three rank policies of the streaming rank-adaptive BUG tracker
(:class:`kvdlra.integrators.streaming.StreamingBUG`) on **Layer-8 K** of the
Llama-3.2-1B cache, as **reconstruction error vs. seq_len**:

* **fixed r = 16** -- hard ``rank_cap = 16`` (a small, cheap budget),
* **fixed r = 32** -- hard ``rank_cap = 32`` (twice the budget),
* **rank-adaptive theta = 1e-2** -- no hard cap; each token keeps the smallest
  rank whose discarded Frobenius-tail is ``<= theta`` (Ceruti--Kusch--Lubich 2022,
  arXiv:2104.05247 §2). Its rank typically *grows* with context.

Each policy is streamed **once**, token by token, over the columns of
``M = (512, T)``. At a grid of prefix lengths ``t`` (every ``--stride`` tokens)
we freeze the tracker's CURRENT left basis ``U`` (after exactly ``t`` tokens) and
record its reconstruction error on the data seen so far,
``tracker.reconstruction_error(M[:, :t]) = ||M_t - U U^T M_t||_F / ||M_t||_F``.
We also record the adaptive policy's tracked rank at each ``t``.

Why we normalize ``M`` to unit Frobenius norm before streaming
--------------------------------------------------------------
``StreamingBUG``'s ``theta`` is an **absolute** Frobenius-tail tolerance: it keeps
the smallest rank whose discarded singular tail ``(sum_{j>r} sigma_j^2)^{1/2}`` is
``<= theta``. The raw post-RoPE Layer-8 K matrix has ``||M||_F ~ 4.1e3`` and even
its *smallest* singular value is ``~6.3``, so an absolute ``theta = 1e-2`` lies far
below a single discarded direction and degenerately forces **full rank r = 512**
(error at machine precision) -- an uninformative comparison. The intended,
dimensionless reading of "theta = 1e-2" is *relative*: "discard at most 1% of the
Frobenius energy." We realize that exactly by rescaling the streamed matrix to
``||M||_F = 1`` (``M_hat = M / ||M||_F``) before feeding it to the tracker. Because
the reported reconstruction error is a **ratio** (``||M-UU^T M||/||M||``) it is
scale-invariant -- the three error curves are bit-identical with or without the
rescale -- so the normalization changes *only* the adaptive rank decision, turning
``theta = 1e-2`` into the sensible "<=1% energy discarded" criterion (whose rank
then grows with context, as the PLAN predicts). The fixed-``rank_cap`` policies are
unaffected by ``theta`` and so are scale-invariant in rank too.

Matrix convention (``docs/notes/conventions.md``; matches
``scripts/week2_pilot.py:load_matrix``): from a per-layer K tensor of shape
``(H, T, D) = (8, T, 64)`` we build ``M = K.transpose(0, 2, 1).reshape(H*D, T)``
-- rows = 512 features, columns = T tokens -- and drop the first ``N_SINK = 4``
attention-sink columns (StreamingLLM; PLAN §8 pitfall #5). We use the **post-RoPE**
``K`` of a 4096-token pilot dump (``doc*_len4096_rope-both``). The post-RoPE
spectrum is shallow (``docs/notes/rope-pitfall.md``) -- that is the RoPE pitfall,
not a tracker failure; the point here is the *relative ordering* of the three
policies and the adaptive rank trajectory, which is RoPE-independent.

Outputs ``figures/week2/rank_policy.{pdf,png}`` (panel 1: error vs. seq_len, three
curves; panel 2: adaptive rank vs. seq_len), a metrics JSON, and a one-line smoke
assertion (fixed r=32 must end no worse than fixed r=16). No model is loaded; the
raw KV dump is gitignored, so the committed JSON is the auditable artifact.
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
# Default 4096-token pilot dump (post-RoPE K). Overridable via --dump.
DEFAULT_DUMP = "dumps/llama3.2-1b/doc63_40468cde_len4096_rope-both"
# The three policies, as (label, kwargs-for-StreamingBUG).
THETA = 1e-2
POLICIES: list[tuple[str, dict[str, object]]] = [
    ("fixed r=16", {"rank_cap": 16}),
    ("fixed r=32", {"rank_cap": 32}),
    (f"adaptive theta={THETA:g}", {"theta": THETA}),
]
# Plot styles, keyed by policy label.
STYLES: dict[str, str] = {
    "fixed r=16": "o-",
    "fixed r=32": "s--",
    f"adaptive theta={THETA:g}": "^:",
}


def load_layer_matrix(dump_dir: Path, layer: int, key: str = "K") -> npt.NDArray[np.float64]:
    """Load one layer dump and build the ``(512, T - N_SINK)`` feature-by-token matrix.

    ``key`` selects ``"K"`` (post-RoPE) or ``"K_pre"`` (pre-RoPE) from a
    ``--rope both`` dump. Matches ``scripts/week2_pilot.py:load_matrix``.
    """
    blob = torch.load(dump_dir / f"layer_{layer:02d}.pt", weights_only=False)
    k = blob[key].float().numpy()  # (H, T, D)
    h, t, d = k.shape
    m = k.transpose(0, 2, 1).reshape(h * d, t)  # rows = features, cols = tokens
    if m.shape[1] > N_SINK:
        m = m[:, N_SINK:]
    return np.ascontiguousarray(m, dtype=np.float64)


def prefix_grid(n_cols: int, stride: int) -> list[int]:
    """Prefix lengths ``t`` at which to evaluate: every ``stride`` tokens, plus the end.

    Starts at ``stride`` (the first non-trivial prefix) and always includes the
    full length ``n_cols`` so the final error of every policy is recorded.
    """
    grid = list(range(stride, n_cols + 1, stride))
    if not grid or grid[-1] != n_cols:
        grid.append(n_cols)
    return grid


def stream_policy(
    m: npt.NDArray[np.float64], grid: list[int], **bug_kwargs: object
) -> tuple[list[float], list[int]]:
    """Stream ``m`` once through ``StreamingBUG`` with ``bug_kwargs``; evaluate on the grid.

    Streams the columns of ``m`` one at a time. At each prefix length ``t`` in
    ``grid`` (after exactly ``t`` :meth:`update` calls), it records the tracker's
    reconstruction error on the data seen so far -- ``reconstruction_error(m[:,
    :t])`` using the CURRENT basis ``U`` -- and the current tracked rank.

    Returns ``(errors, ranks)``, each aligned with ``grid``.
    """
    tracker = StreamingBUG(n_features=m.shape[0], **bug_kwargs)
    grid_set = set(grid)
    errors: list[float] = []
    ranks: list[int] = []
    n_cols = m.shape[1]
    for t in range(1, n_cols + 1):
        tracker.update(m[:, t - 1])
        if t in grid_set:
            errors.append(tracker.reconstruction_error(m[:, :t]))
            ranks.append(tracker.rank)
    return errors, ranks


def plot_rank_policy(
    grid: list[int],
    errors_by_policy: dict[str, list[float]],
    adaptive_label: str,
    adaptive_ranks: list[int],
    fixed_ranks: dict[str, int],
    out_path: Path,
) -> None:
    """Two panels: (1) error vs. seq_len for the 3 policies, (2) adaptive rank vs. seq_len."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for label, errs in errors_by_policy.items():
        ax.plot(grid, errs, STYLES.get(label, "-"), markersize=4, label=label)
    ax.set_xlabel("seq_len (tokens streamed)")
    ax.set_ylabel(r"rel. Frobenius error  $\||M_t - UU^\top M_t\||_F / \||M_t\||_F$")
    ax.set_title("Layer-8 K (post-RoPE): reconstruction error vs. seq_len")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    ax2.plot(grid, adaptive_ranks, "^:", color="C2", markersize=4, label=adaptive_label)
    for label, r in fixed_ranks.items():
        ax2.axhline(r, ls="--", lw=1.0, alpha=0.6, label=f"{label} (cap)")
    ax2.set_xlabel("seq_len (tokens streamed)")
    ax2.set_ylabel("tracked rank r")
    ax2.set_title("Rank-adaptive policy: rank vs. seq_len")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main() -> None:
    """Parse arguments, stream the three policies, plot, write JSON, smoke-assert."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default=DEFAULT_DUMP, help="dir of layer_XX.pt files")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument(
        "--key", default="K", choices=["K", "K_pre"], help="post-RoPE (K) or pre-RoPE (K_pre)"
    )
    parser.add_argument(
        "--stride", type=int, default=128, help="evaluate the error every STRIDE tokens"
    )
    parser.add_argument("--fig", default="figures/week2/rank_policy.pdf")
    parser.add_argument("--metrics", default="figures/week2/rank_policy_metrics.json")
    args = parser.parse_args()

    dump_dir = Path(args.dump)
    m_raw = load_layer_matrix(dump_dir, args.layer, args.key)
    # Normalize to unit Frobenius norm so the absolute ``theta`` becomes a relative
    # "fraction of energy discarded" tolerance (see the module docstring). The
    # reported error is scale-invariant; only the adaptive rank decision changes.
    fro = float(np.linalg.norm(m_raw))
    m = m_raw / fro if fro > 0.0 else m_raw
    grid = prefix_grid(m.shape[1], args.stride)
    print(
        f"Layer {args.layer} ({args.key}): M.shape={m.shape} "
        f"(rows=features, cols=tokens; dropped {N_SINK} sinks); ||M||_F={fro:.3e} "
        f"-> normalized to 1 for theta; {len(grid)} grid points "
        f"(stride {args.stride}, up to t={grid[-1]})"
    )

    # --- Stream each policy once, evaluating on the shared prefix grid. ---
    t0 = time.perf_counter()
    errors_by_policy: dict[str, list[float]] = {}
    ranks_by_policy: dict[str, list[int]] = {}
    for label, kwargs in POLICIES:
        errs, ranks = stream_policy(m, grid, **kwargs)
        errors_by_policy[label] = errs
        ranks_by_policy[label] = ranks
        print(
            f"  {label:>22}: final rel-err={errs[-1]:.4e}  final rank={ranks[-1]}  "
            f"(rank range {min(ranks)}..{max(ranks)})"
        )
    print(f"(streamed 3 policies in {time.perf_counter() - t0:.1f}s)")

    adaptive_label = POLICIES[-1][0]
    fixed_labels = [lbl for lbl, _ in POLICIES[:-1]]
    fixed_ranks = {lbl: ranks_by_policy[lbl][-1] for lbl in fixed_labels}

    # --- Plot ---
    fig_path = Path(args.fig)
    plot_rank_policy(
        grid,
        errors_by_policy,
        adaptive_label,
        ranks_by_policy[adaptive_label],
        fixed_ranks,
        fig_path,
    )
    print(f"wrote figure {fig_path} (+ .png)")

    # --- Honest comparison summary (PLAN: where does adaptive land?) ---
    final_err = {lbl: errors_by_policy[lbl][-1] for lbl, _ in POLICIES}
    adaptive_final_rank = ranks_by_policy[adaptive_label][-1]
    r32_final_err = final_err["fixed r=32"]
    adaptive_final_err = final_err[adaptive_label]
    # If the adaptive policy ends at a rank >= 32 yet is worse than fixed r=32,
    # that is the honest "worse at equal/greater rank" caveat the PLAN asks for.
    adaptive_ge_r32_rank = adaptive_final_rank >= fixed_ranks["fixed r=32"]
    adaptive_worse_than_r32 = adaptive_final_err > r32_final_err
    print(
        f"\nadaptive theta={THETA:g} ends at rank {adaptive_final_rank} "
        f"(fixed caps: r=16, r=32); final rel-err {adaptive_final_err:.4e} vs "
        f"r=32 {r32_final_err:.4e}"
    )
    if adaptive_ge_r32_rank and adaptive_worse_than_r32:
        print(
            "  NOTE: adaptive uses >= r=32 rank yet has higher error than fixed r=32 "
            "-- worse at equal/greater rank (honest caveat)."
        )

    # --- Metrics JSON ---
    metrics: dict[str, object] = {
        "dump": str(dump_dir),
        "layer": args.layer,
        "key": args.key,
        "matrix_shape": list(m.shape),
        "n_sink_dropped": N_SINK,
        "frobenius_norm_raw": fro,
        "normalized_to_unit_frobenius": True,
        "stride": args.stride,
        "theta": THETA,
        "seq_len_grid": grid,
        "error_vs_seq_len": errors_by_policy,
        "rank_vs_seq_len": ranks_by_policy,
        "final": {
            "error": final_err,
            "rank": {lbl: ranks_by_policy[lbl][-1] for lbl, _ in POLICIES},
        },
        "adaptive_vs_r32": {
            "adaptive_final_rank": adaptive_final_rank,
            "r32_cap": fixed_ranks["fixed r=32"],
            "adaptive_uses_ge_r32_rank": bool(adaptive_ge_r32_rank),
            "adaptive_worse_than_r32": bool(adaptive_worse_than_r32),
        },
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote metrics {metrics_path}")

    # --- Smoke assertion: more rank cannot hurt the final fit (r=32 <= r=16). ---
    assert final_err["fixed r=32"] <= final_err["fixed r=16"] + 1e-9, (
        f"fixed r=32 final error {final_err['fixed r=32']:.4e} should be <= "
        f"fixed r=16 {final_err['fixed r=16']:.4e}"
    )
    print("smoke OK: fixed r=32 final error <= fixed r=16 final error")


if __name__ == "__main__":
    main()
