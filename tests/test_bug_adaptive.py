"""Tests for the rank-adaptive (augmented) BUG integrator (numpy float64).

Validates :func:`kvdlra.integrators.bug_adaptive.rank_adaptive_bug_step`, the
rank-adaptive integrator of Ceruti--Kusch--Lubich 2022 (arXiv:2104.05247 §2),
three complementary ways:

* :func:`test_adaptive_tracks_exact` -- on the standard stiff Lyapunov flow,
  started at the true rank with a tight ``theta``, the rank-adaptive step tracks
  the exact analytic solution ``Y(t) = e^{-tB} Y0 e^{-tB}`` (B symmetric) about
  as well as fixed-rank BUG. This is the accuracy guarantee.
* :func:`test_rank_grows_when_underresolved` -- started at a deliberately
  too-small rank with tight ``theta``, the effective rank (size of ``S``)
  *increases* over several steps, demonstrating basis augmentation.
* :func:`test_rank_shrinks_when_overresolved` -- started at an inflated rank
  whose core has only a few significant singular values, a moderate ``theta``
  *shrinks* the effective rank below the initial one, demonstrating truncation.

The reusable pieces (``build_operator``, ``F``, exact reference pattern) follow
``tests/test_bug_synthetic.py``; ``build_operator``/``F`` are imported from
:mod:`kvdlra.integrators.bug` rather than redefined.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from kvdlra.integrators.bug import build_operator
from kvdlra.integrators.bug_adaptive import rank_adaptive_bug_step


def _exact_Y(B: np.ndarray, Y0: np.ndarray, T: float) -> np.ndarray:
    """Exact solution of ``Y' = -(B Y + Y B^T)`` at time ``T``.

    For symmetric ``B`` the flow has the closed form
    ``Y(T) = e^{-TB} Y0 e^{-TB^T}`` (same reference as the fixed-rank tests).
    """
    E = sla.expm(-T * B)
    return np.asarray(E @ Y0 @ E.T)


def test_adaptive_tracks_exact() -> None:
    """Rank-adaptive BUG tracks the exact solution at least as well as fixed-rank.

    Started at the true rank ``r=8`` with a tight tolerance ``theta=1e-10``, the
    augmented step is free to *grow* the rank (here 8 -> 10) to capture more of
    the geometrically-decaying spectrum, so it is markedly more accurate than
    fixed-rank BUG: observed rel. error ~6.2e-8 vs. ~4.1e-5 for plain
    :func:`~kvdlra.integrators.bug.bug_step` on the identical problem. The bound
    1e-6 sits ~16x above the observed value -- loose enough to be robust to BLAS
    rounding, but tight enough that any regression toward fixed-rank accuracy
    (e.g. dropping the basis augmentation) or a broken projection (wrong substep
    order) would trip it by orders of magnitude.
    """
    rng = np.random.default_rng(0)
    n, r, T, h, theta = 100, 8, 0.05, 1e-3, 1e-10
    B = build_operator(n)
    U, _ = np.linalg.qr(rng.standard_normal((n, r)))
    V, _ = np.linalg.qr(rng.standard_normal((n, r)))
    S = np.diag(10.0 ** (-np.arange(r)))
    Y0 = U @ S @ V.T

    nsteps = round(T / h)
    ranks: list[int] = [S.shape[0]]
    for _ in range(nsteps):
        U, S, V = rank_adaptive_bug_step(U, S, V, B, h, theta)
        ranks.append(S.shape[0])
    Y_num = U @ S @ V.T

    Y_exact = _exact_Y(B, Y0, T)
    rel_err = float(np.linalg.norm(Y_num - Y_exact, "fro") / np.linalg.norm(Y_exact, "fro"))
    print(f"[tracks_exact] rel_err={rel_err:.3e}  rank trajectory={ranks}")
    assert rel_err < 1e-6, f"rank-adaptive BUG diverged from exact: {rel_err}"


def test_rank_grows_when_underresolved() -> None:
    """Rank grows when the integrator starts below the solution's true rank.

    The exact flow ``Y(t) = e^{-tB} Y0 e^{-tB}`` started from a full-rank-ish
    ``Y0`` has many non-negligible singular values. Starting the integrator at
    rank ``r=2`` with a tight ``theta`` forces it to add directions: each step
    augments by at most ``r`` new basis vectors and the tight tolerance keeps
    them, so after several steps the effective rank must exceed the initial 2.
    """
    rng = np.random.default_rng(1)
    n, r0, h, theta = 100, 2, 1e-3, 1e-12
    nsteps = 12
    B = build_operator(n)
    # A genuinely high-rank initial Y0 (rank-20 core with slowly decaying values)
    # so that a rank-2 approximation is badly under-resolved.
    r_true = 20
    Uf, _ = np.linalg.qr(rng.standard_normal((n, r_true)))
    Vf, _ = np.linalg.qr(rng.standard_normal((n, r_true)))
    Sf = np.diag(2.0 ** (-np.arange(r_true)))  # 1, 1/2, 1/4, ... down to ~2e-6
    Y0 = Uf @ Sf @ Vf.T

    # Initialize the integrator at rank r0 = 2 via a truncated SVD of Y0.
    P, sig, Qt = np.linalg.svd(Y0, full_matrices=False)
    U = P[:, :r0]
    S = np.diag(sig[:r0])
    V = Qt[:r0, :].T

    ranks: list[int] = [S.shape[0]]
    for _ in range(nsteps):
        U, S, V = rank_adaptive_bug_step(U, S, V, B, h, theta)
        ranks.append(S.shape[0])
    print(f"[rank_grows] rank trajectory={ranks}")
    assert ranks[-1] > ranks[0], f"expected rank to grow above {ranks[0]}, got {ranks}"


def test_rank_shrinks_when_overresolved() -> None:
    """Rank shrinks when the initial core has only a few significant values.

    Start at an inflated rank ``r=8`` whose core ``S0`` has only ~3 singular
    values above ``theta`` (the remainder are ~1e-13). The contractive Lyapunov
    flow does not create new directions, so the moderate tolerance
    ``theta=1e-6`` truncates the negligible tail and the effective rank drops to
    <= 4 within a few steps.
    """
    rng = np.random.default_rng(2)
    n, r0, h, theta = 100, 8, 1e-3, 1e-6
    nsteps = 5
    B = build_operator(n)
    U, _ = np.linalg.qr(rng.standard_normal((n, r0)))
    V, _ = np.linalg.qr(rng.standard_normal((n, r0)))
    # Only the first 3 singular values are significant; the rest are ~1e-13.
    S = np.diag(np.array([1.0, 0.5, 0.25] + [1e-13] * (r0 - 3)))

    ranks: list[int] = [S.shape[0]]
    for _ in range(nsteps):
        U, S, V = rank_adaptive_bug_step(U, S, V, B, h, theta)
        ranks.append(S.shape[0])
    print(f"[rank_shrinks] rank trajectory={ranks}")
    assert ranks[-1] < ranks[0], f"expected rank to shrink below {ranks[0]}, got {ranks}"
    assert ranks[-1] <= 4, f"expected final rank <= 4, got {ranks[-1]}"


if __name__ == "__main__":
    test_adaptive_tracks_exact()
    test_rank_grows_when_underresolved()
    test_rank_shrinks_when_overresolved()
    print("OK")
