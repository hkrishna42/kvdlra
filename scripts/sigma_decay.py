"""Singular-value decay + reconstruction-error-vs-rank on captured KV.

PLAN §3 Script #3 (enhanced). For each requested layer this loads the per-layer
key dump written by ``scripts/capture_kv.py``, builds the feature-by-token matrix

    m = k.transpose(0, 2, 1).reshape(h * d, t)      # rows = features, cols = tokens

per ``docs/notes/conventions.md`` (rows = ``head_dim * num_kv_heads = 64 * 8 =
512`` features; columns = ``T`` tokens; "append a token" = "append a column"),
and plots four curves:

* the singular-value decay ``sigma_i / sigma_1`` of ``m``;
* relative Frobenius reconstruction error vs. rank for
    - **truncated SVD** (the oracle / irreducible lower bound at each rank),
    - **incremental SVD** (Brand-style streaming baseline),
    - a **BUG-based reconstruction** (one Ceruti--Lubich basis-update & Galerkin
      sweep over the token stream; the substep ODEs are advanced with
      :func:`kvdlra.integrators.bug.rk2_step`, the same RK2 stepper used by the
      validated synthetic integrator).

Sink tokens
-----------
StreamingLLM (arXiv 2309.17453, Fig. 2) shows the first 4 tokens are outliers in
any low-rank basis (PLAN §8 pitfall #5). By default the first 4 *columns*
(sink-token positions) are **excluded** from ``m`` before factoring; pass
``--keep-sinks`` to include them.

RoPE caveat
-----------
The cached keys are **post-RoPE** (see ``docs/notes/rope-pitfall.md``). RoPE
smears low-rank structure across positions, so a shallow decay or a
disappointing reconstruction-error curve is expected for post-RoPE keys and is
*this pitfall*, not a failure of DLRA. Report it as-is.

The output PDF is written under ``figures/`` (the project's tracked figures dir;
``figs/`` is gitignored).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
import torch

matplotlib.use("Agg")  # headless / CPU-safe backend; set before pyplot import
import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import matplotlib.pyplot as plt

from kvdlra.integrators.bug import rk2_step

# Number of leading attention-sink tokens to drop by default (StreamingLLM).
n_sink = 4


def truncated_svd_recon(m: npt.NDArray[np.float64], r: int) -> float:
    """Relative Frobenius error of the best rank-``r`` (truncated-SVD) recon.

    This is the oracle / irreducible lower bound at rank ``r``: the Eckart--Young
    optimal rank-``r`` approximation of ``m``.
    """
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    mr = (u[:, :r] * s[:r]) @ vt[:r]
    return float(np.linalg.norm(m - mr, ord="fro") / np.linalg.norm(m, ord="fro"))


def incremental_svd_recon(m: npt.NDArray[np.float64], r: int) -> float:
    """Brand-style incremental SVD: stream columns of ``m``, keep top-``r``.

    A locally greedy streaming baseline -- it never revisits a column once
    absorbed -- which is exactly the "naive streaming" curve the BUG integrator
    is meant to beat at matched rank.
    """
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
        # Project the new column onto the current basis and take the residual.
        proj = u.T @ c
        resid = c - u @ proj
        resid_norm = float(np.linalg.norm(resid))
        resid_unit = resid / (resid_norm + 1e-12)
        # Build and re-decompose the small (k+1)x(k+1) core.
        core = np.block([[np.diag(s), proj], [np.zeros((1, s.size)), np.array([[resid_norm]])]])
        up, sp, vtp = np.linalg.svd(core, full_matrices=False)
        # Absorb the rotation back into the bases.
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
    return float(np.linalg.norm(m - mr, ord="fro") / np.linalg.norm(m, ord="fro"))


def bug_recon(m: npt.NDArray[np.float64], r: int, h: float = 1.0) -> float:
    """One Ceruti--Lubich BUG sweep toward target ``m``; return rel. Frob. error.

    Treats the columns of ``m`` as a token stream and performs a single
    basis-update & Galerkin step (arXiv:2010.02022 §3.1 ordering: K- and L-steps
    from the OLD factors in parallel, then a forward-in-time S-step) for the
    relaxation field ``f(y) = m - y``. Each substep ODE is advanced with one
    :func:`kvdlra.integrators.bug.rk2_step` (the validated RK2 stepper), rather
    than a hand-rolled Euler step, so this script reuses the integrator library.
    """
    n, t = m.shape
    rng = np.random.default_rng(0)
    u, _ = np.linalg.qr(rng.standard_normal((n, r)))
    v, _ = np.linalg.qr(rng.standard_normal((t, r)))
    s = np.zeros((r, r))

    def f(y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return m - y

    # K-step: K' = f(K V^T) V, K(0) = U S  (uses OLD U, S, V).
    k1 = rk2_step(u @ s, lambda k: f(k @ v.T) @ v, h)
    u1, _ = np.linalg.qr(k1)
    mm = u1.T @ u

    # L-step: L' = f(U L^T)^T U, L(0) = V S^T  (uses OLD U, S, V).
    l1 = rk2_step(v @ s.T, lambda lf: f(u @ lf.T).T @ u, h)
    v1, _ = np.linalg.qr(l1)
    nn = v1.T @ v

    # S-step (forward in time): S' = U1^T f(U1 S V1^T) V1, S(0) = M S N^T.
    s0 = mm @ s @ nn.T
    s1 = rk2_step(s0, lambda sm: u1.T @ f(u1 @ sm @ v1.T) @ v1, h)

    mr = u1 @ s1 @ v1.T
    return float(np.linalg.norm(m - mr, ord="fro") / np.linalg.norm(m, ord="fro"))


def load_layer_matrix(dump_dir: Path, layer: int, keep_sinks: bool) -> npt.NDArray[np.float64]:
    """Load one layer dump and build the ``(h*d, t)`` feature-by-token matrix.

    Drops the first ``n_sink`` token columns unless ``keep_sinks`` is set.
    """
    blob = torch.load(dump_dir / f"layer_{layer:02d}.pt", weights_only=False)
    k = blob["K"].float().numpy()  # (h, t, d)
    h, t, d = k.shape
    m = k.transpose(0, 2, 1).reshape(h * d, t)  # rows = features, cols = tokens
    if not keep_sinks and m.shape[1] > n_sink:
        m = m[:, n_sink:]
    return np.ascontiguousarray(m, dtype=np.float64)


def make_ranks(m: npt.NDArray[np.float64]) -> list[int]:
    """Default rank grid, clamped to be strictly below ``min(m.shape)``."""
    grid = [4, 8, 16, 32, 64, 128]
    return [r for r in grid if r < min(m.shape)]


def main() -> None:
    """Parse arguments, compute curves per layer, and save the figure PDF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, help="dir of layer_XX.pt files")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 8, 15])
    parser.add_argument("--out", default="figures/week1/sv_decay.pdf")
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=None,
        help="explicit rank grid (default: 4 8 16 32 64 128, clamped)",
    )
    parser.add_argument(
        "--keep-sinks",
        action="store_true",
        help=f"include the first {n_sink} sink-token columns (default: exclude)",
    )
    args = parser.parse_args()

    dump_dir = Path(args.dump)
    layers: list[int] = list(args.layers)
    recon_fns: list[tuple[str, str, Callable[[npt.NDArray[np.float64], int], float]]] = [
        ("truncated SVD (oracle)", "o-", truncated_svd_recon),
        ("incremental SVD", "s--", incremental_svd_recon),
        ("BUG (1 sweep, RK2)", "^:", bug_recon),
    ]

    fig, axes = plt.subplots(len(layers), 2, figsize=(11, 4 * len(layers)), squeeze=False)

    layer_metrics: dict[str, dict[str, object]] = {}

    for row, layer in enumerate(layers):
        m = load_layer_matrix(dump_dir, layer, args.keep_sinks)
        sigma = np.linalg.svd(m, compute_uv=False)
        ranks = list(args.ranks) if args.ranks is not None else make_ranks(m)
        ranks = [r for r in ranks if r < min(m.shape)]
        sinks = "with sinks" if args.keep_sinks else f"no first-{n_sink} sinks"
        print(f"layer {layer}: m.shape={m.shape} ({sinks}); ranks={ranks}")

        ax_sigma = axes[row][0]
        ax_sigma.semilogy(sigma / sigma[0])
        ax_sigma.set_xlabel("index i")
        ax_sigma.set_ylabel(r"$\sigma_i / \sigma_1$")
        ax_sigma.set_title(f"layer {layer} K-cache singular values ({sinks})")
        ax_sigma.grid(True, alpha=0.3)

        errors: dict[str, list[float]] = {}
        ax_err = axes[row][1]
        for label, style, fn in recon_fns:
            errs = [fn(m, r) for r in ranks]
            errors[label] = errs
            ax_err.semilogy(ranks, errs, style, label=label)
            pairs = ", ".join(f"r={r}:{e:.3e}" for r, e in zip(ranks, errs, strict=True))
            print(f"  {label}: {pairs}")
        ax_err.set_xlabel("rank r")
        ax_err.set_ylabel("rel. Frobenius error")
        ax_err.set_title(f"layer {layer} reconstruction error vs. rank")
        ax_err.legend()
        ax_err.grid(True, alpha=0.3)

        layer_metrics[str(layer)] = {
            "shape": list(m.shape),
            "ranks": ranks,
            "sigma_normalized": (sigma / sigma[0]).tolist(),
            "errors": errors,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote figure {out_path}")

    # Sidecar metrics so the figure's numbers are reproducible / auditable from the
    # repo without re-running the model (the raw KV dump is gitignored).
    metrics: dict[str, object] = {
        "dump": str(dump_dir),
        "keep_sinks": bool(args.keep_sinks),
        "n_sink": n_sink,
        "layers": layer_metrics,
    }
    metrics_path = out_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote metrics {metrics_path}")


if __name__ == "__main__":
    main()
