"""Tests for the reusable :class:`kvdlra.integrators.bug_class.BUG` integrator.

The fixed-rank BUG step of :mod:`kvdlra.integrators.bug` (Ceruti--Lubich 2022,
arXiv:2010.02022 §3.1) is validated functionally in
``tests/test_bug_synthetic.py``. Here we check the *class* refactor
(:class:`LowRankState` + :class:`BUG`) that exposes the same step for a generic
vector field ``rhs(Y) -> Ydot``:

#. it reproduces the functional ``bug_step`` trajectory on the synthetic
   Lyapunov flow to round-off;
#. it is first order in time vs. the exact ``expm`` solution (error ratio ~2
   when ``h`` halves);
#. it matches ``bug_step`` step-for-step on a random rank-``r`` start;
#. :class:`LowRankState` invariants (orthonormal ``U, V``; ``r x r`` ``S``) are
   preserved across a step;
#. a non-Lyapunov linear field ``Y' = A @ Y`` is tracked accurately;
#. a rank-1 start works;
#. different ``(n, r, h)`` work;
#. a zero field leaves the reconstruction unchanged (up to QR gauge);
#. :meth:`BUG.integrate` over ``N`` steps equals ``N`` manual steps;
#. reconstruction stays finite (no NaNs) on a stiff flow.

The robust-error theory (Kieri--Lubich--Walach 2016, SIAM J. Numer. Anal.
54(2), §2 / Thm 2.1) guarantees the error bound is independent of sigma_min;
case 10 exercises a stiff operator with a wide singular-value spread.

GAUGE-FREEDOM CAVEAT: a factorization ``(U, S, V)`` is only unique up to a
shared orthogonal change of basis ``U -> U Q``, ``S -> Q^T S R``, ``V -> V R``.
The QR sign/ordering in ``bug_step`` and ``BUG.step`` is *identical* (same code
path on the same arrays), so step-for-step comparisons of the raw factors hold
to round-off (cases 1, 3). Comparisons that could legitimately differ in gauge
(case 8, zero field) are made on the gauge-invariant reconstruction
``U @ S @ V.T`` instead.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from kvdlra.integrators.bug import F, bug_step, build_operator
from kvdlra.integrators.bug_class import BUG, LowRankState


def _lyapunov_start(n: int, r: int) -> tuple[LowRankState, np.ndarray]:
    """Deterministic Lyapunov start ``(state, B)`` matching the synthetic test.

    ``U0, V0`` are orthonormal ``(n, r)`` (QR of fixed-seed Gaussians); ``S0`` is
    the diagonal core with geometric singular values ``10**(-i)``; ``B`` is the
    Lyapunov operator from :func:`build_operator`.
    """
    rng = np.random.default_rng(0)
    B = build_operator(n)
    U0, _ = np.linalg.qr(rng.standard_normal((n, r)))
    V0, _ = np.linalg.qr(rng.standard_normal((n, r)))
    S0 = np.diag(10.0 ** (-np.arange(r))).astype(np.float64)
    return LowRankState(U=U0, S=S0, V=V0), B


def _random_orthonormal(n: int, r: int, seed: int) -> np.ndarray:
    """Orthonormal ``(n, r)`` factor from a fixed seed (QR of a Gaussian)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, r)))
    return np.asarray(Q, dtype=np.float64)


def _exact_lyapunov(B: np.ndarray, Y0: np.ndarray, T: float) -> np.ndarray:
    """Exact ``Y(T) = e^{-TB} Y0 e^{-TB^T}`` of ``Y' = -(B Y + Y B^T)``."""
    E = sla.expm(-T * B)
    return np.asarray(E @ Y0 @ E.T)


# --------------------------------------------------------------------------- #
# 1. Class trajectory matches the functional bug_step on the Lyapunov flow.
# --------------------------------------------------------------------------- #
def test_class_matches_bug_step_trajectory_lyapunov() -> None:
    """Many class steps equal many ``bug_step`` calls to ~1e-12 (Lyapunov flow)."""
    n, r, h, nsteps = 100, 8, 1e-3, 100
    state, B = _lyapunov_start(n, r)

    def rhs(Y: np.ndarray) -> np.ndarray:
        return F(Y, B)

    integrator = BUG()
    final = integrator.integrate(state, rhs, h, nsteps)

    U, S, V = state.U, state.S, state.V
    for _ in range(nsteps):
        U, S, V = bug_step(U, S, V, B, h)

    assert np.allclose(final.full(), U @ S @ V.T, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# 2. First-order convergence to the exact solution (error ratio ~2).
# --------------------------------------------------------------------------- #
def test_first_order_convergence_vs_expm() -> None:
    """Halving ``h`` halves the error vs. the exact ``expm`` solution."""
    n, r, T = 100, 8, 0.1
    state, B = _lyapunov_start(n, r)
    Y_exact = _exact_lyapunov(B, state.full(), T)

    def rhs(Y: np.ndarray) -> np.ndarray:
        return F(Y, B)

    integrator = BUG()
    errs: list[float] = []
    for h in (1e-3, 5e-4):
        final = integrator.integrate(state, rhs, h, round(T / h))
        errs.append(float(np.linalg.norm(final.full() - Y_exact, ord="fro")))
    ratio = errs[0] / errs[1]
    assert 1.8 < ratio < 2.2, f"expected first-order ratio ~2, got {ratio:.3f}"


# --------------------------------------------------------------------------- #
# 3. Class step == bug_step step-for-step on a random rank-r start.
# --------------------------------------------------------------------------- #
def test_class_step_equals_bug_step_random_start() -> None:
    """A single class step equals a single ``bug_step`` on random factors."""
    n, r, h = 60, 5, 1e-2
    B = build_operator(n)
    U0 = _random_orthonormal(n, r, seed=1)
    V0 = _random_orthonormal(n, r, seed=2)
    rng = np.random.default_rng(3)
    S0 = rng.standard_normal((r, r)).astype(np.float64)
    state = LowRankState(U=U0, S=S0, V=V0)

    def rhs(Y: np.ndarray) -> np.ndarray:
        return F(Y, B)

    new = BUG().step(state, rhs, h)
    U1, S1, V1 = bug_step(U0, S0, V0, B, h)

    assert np.allclose(new.U, U1, atol=1e-12, rtol=0.0)
    assert np.allclose(new.S, S1, atol=1e-12, rtol=0.0)
    assert np.allclose(new.V, V1, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# 4. LowRankState invariants preserved after a step.
# --------------------------------------------------------------------------- #
def test_lowrank_state_invariants_preserved() -> None:
    """After a step ``U, V`` stay orthonormal ``(n, r)`` and ``S`` stays ``r x r``."""
    n, r, h = 50, 6, 1e-2
    state, B = _lyapunov_start(n, r)

    def rhs(Y: np.ndarray) -> np.ndarray:
        return F(Y, B)

    new = BUG().step(state, rhs, h)
    assert new.rank == r
    assert new.U.shape == (n, r)
    assert new.V.shape == (n, r)
    assert new.S.shape == (r, r)
    eye = np.eye(r)
    assert np.allclose(new.U.T @ new.U, eye, atol=1e-12)
    assert np.allclose(new.V.T @ new.V, eye, atol=1e-12)


# --------------------------------------------------------------------------- #
# 5. Non-Lyapunov linear field Y' = A @ Y tracked accurately.
# --------------------------------------------------------------------------- #
def test_non_lyapunov_linear_field_tracked() -> None:
    """``Y' = A @ Y`` (left-multiplication only) tracked vs. exact ``e^{tA} Y0``.

    This is a genuinely non-Lyapunov field: it acts on the left only, so the
    column space follows ``e^{tA}`` while the row space is invariant. The exact
    solution ``Y(t) = e^{tA} Y0`` is exactly rank ``r`` for all ``t``, so the
    fixed-rank BUG integrator can track it to its O(h) discretization error.
    """
    n, r, T, h = 40, 4, 0.05, 5e-4
    rng = np.random.default_rng(7)
    # Mildly stiff but stable: symmetric negative-definite A keeps the flow bounded.
    A = rng.standard_normal((n, n))
    A = -(A @ A.T) / n
    U0 = _random_orthonormal(n, r, seed=8)
    V0 = _random_orthonormal(n, r, seed=9)
    S0 = np.diag(10.0 ** (-np.arange(r))).astype(np.float64)
    state = LowRankState(U=U0, S=S0, V=V0)

    def rhs(Y: np.ndarray) -> np.ndarray:
        return np.asarray(A @ Y)

    final = BUG().integrate(state, rhs, h, round(T / h))
    Y_exact = sla.expm(T * A) @ state.full()
    err = float(np.linalg.norm(final.full() - Y_exact, ord="fro"))
    assert err < 1e-5, f"linear-field tracking error too large: {err:.3e}"


# --------------------------------------------------------------------------- #
# 6. Rank-1 start.
# --------------------------------------------------------------------------- #
def test_rank_one_start() -> None:
    """A rank-1 state steps without error and matches ``bug_step``."""
    n, r, h = 30, 1, 1e-2
    B = build_operator(n)
    U0 = _random_orthonormal(n, r, seed=10)
    V0 = _random_orthonormal(n, r, seed=11)
    S0 = np.array([[2.0]], dtype=np.float64)
    state = LowRankState(U=U0, S=S0, V=V0)
    assert state.rank == 1

    def rhs(Y: np.ndarray) -> np.ndarray:
        return F(Y, B)

    new = BUG().step(state, rhs, h)
    U1, S1, V1 = bug_step(U0, S0, V0, B, h)
    assert new.rank == 1
    assert new.S.shape == (1, 1)
    assert np.allclose(new.full(), U1 @ S1 @ V1.T, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# 7. Different (n, r, h) still match the functional step.
# --------------------------------------------------------------------------- #
def test_various_shapes_match_bug_step() -> None:
    """Several ``(n, r, h)`` combinations agree with ``bug_step`` to round-off."""
    integrator = BUG()
    for n, r, h, seed in ((20, 3, 5e-3, 21), (80, 10, 2e-3, 22), (45, 7, 1e-2, 23)):
        B = build_operator(n)
        U0 = _random_orthonormal(n, r, seed=seed)
        V0 = _random_orthonormal(n, r, seed=seed + 100)
        S0 = np.diag(2.0 ** (-np.arange(r))).astype(np.float64)
        state = LowRankState(U=U0, S=S0, V=V0)

        def rhs(Y: np.ndarray, B: np.ndarray = B) -> np.ndarray:
            return F(Y, B)

        new = integrator.step(state, rhs, h)
        U1, S1, V1 = bug_step(U0, S0, V0, B, h)
        assert np.allclose(new.full(), U1 @ S1 @ V1.T, atol=1e-12, rtol=0.0), (
            f"mismatch at n={n}, r={r}, h={h}"
        )


# --------------------------------------------------------------------------- #
# 8. Zero field leaves the reconstruction unchanged (up to QR gauge).
# --------------------------------------------------------------------------- #
def test_zero_field_leaves_state_unchanged() -> None:
    """With ``rhs == 0`` the reconstruction ``U S V^T`` is unchanged.

    The raw factors may be re-gauged by the QR (``U S V^T`` is invariant under a
    shared orthogonal change of basis), so we compare the gauge-invariant dense
    reconstruction rather than the factors.
    """
    n, r, h = 35, 4, 0.1
    U0 = _random_orthonormal(n, r, seed=12)
    V0 = _random_orthonormal(n, r, seed=13)
    rng = np.random.default_rng(14)
    S0 = rng.standard_normal((r, r)).astype(np.float64)
    state = LowRankState(U=U0, S=S0, V=V0)
    before = state.full()

    def rhs(Y: np.ndarray) -> np.ndarray:
        return np.zeros_like(Y)

    new = BUG().step(state, rhs, h)
    assert np.allclose(new.full(), before, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# 9. integrate(N) equals N manual steps.
# --------------------------------------------------------------------------- #
def test_integrate_equals_manual_steps() -> None:
    """``integrate(state, rhs, h, N)`` equals applying ``step`` ``N`` times."""
    n, r, h, nsteps = 50, 6, 2e-3, 17
    state, B = _lyapunov_start(n, r)

    def rhs(Y: np.ndarray) -> np.ndarray:
        return F(Y, B)

    integrator = BUG()
    bulk = integrator.integrate(state, rhs, h, nsteps)

    manual = state
    for _ in range(nsteps):
        manual = integrator.step(manual, rhs, h)

    assert np.allclose(bulk.full(), manual.full(), atol=1e-12, rtol=0.0)
    # integrate(.., 0) is the identity.
    assert integrator.integrate(state, rhs, h, 0) is state


# --------------------------------------------------------------------------- #
# 10. Reconstruction stays finite (no NaNs) on a stiff flow.
# --------------------------------------------------------------------------- #
def test_stiff_flow_stays_finite() -> None:
    """A stiff Lyapunov flow with tiny singular values produces no NaNs/Infs.

    The state starts with singular values spanning ``1`` down to ``1e-10``;
    robustness to such small sigma_min is exactly the Kieri--Lubich--Walach
    (2016, §2 / Thm 2.1) guarantee of the forward S-step. We only assert the
    reconstruction stays finite and orthonormality is retained -- accuracy is
    pinned by the convergence test.
    """
    n, r, h, nsteps = 100, 8, 1e-3, 100
    rng = np.random.default_rng(0)
    B = build_operator(n)
    U0, _ = np.linalg.qr(rng.standard_normal((n, r)))
    V0, _ = np.linalg.qr(rng.standard_normal((n, r)))
    # Singular values from 1e0 down to 1e-10 -> sigma_min is tiny (stiff).
    S0 = np.diag(np.logspace(0, -10, r)).astype(np.float64)
    state = LowRankState(U=U0, S=S0, V=V0)

    def rhs(Y: np.ndarray) -> np.ndarray:
        return F(Y, B)

    final = BUG().integrate(state, rhs, h, nsteps)
    recon = final.full()
    assert np.all(np.isfinite(recon))
    assert np.allclose(final.U.T @ final.U, np.eye(r), atol=1e-10)
    assert np.allclose(final.V.T @ final.V, np.eye(r), atol=1e-10)
