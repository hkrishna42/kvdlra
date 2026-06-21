"""Gold "the math is correct" tests for the BUG integrator (numpy float64).

Based on Script #1 of the project plan (§3): a from-scratch BUG integrator
(Ceruti--Lubich 2022, arXiv:2010.02022 §3.1) on a stiff matrix Lyapunov flow,
validated against a dense ``scipy.integrate.solve_ivp`` RK45 reference.

Correctness is asserted three complementary ways:

* :func:`test_bug_tracks_reference` -- absolute error at the plan's ``h=1e-3``
  is small (a few times 1e-5) vs. the ode45-grade reference.
* :func:`test_bug_first_order_convergence` -- halving ``h`` halves the error,
  i.e. the basic (unconventional) BUG integrator is **first order** in time.
  This is the rigorous correctness check: a wrong substep ordering (K & L from
  the old factors, then S forward) would not give clean O(h) convergence.
* :func:`test_bug_meets_plan_threshold_at_fine_h` -- the plan's headline
  ``< 1e-5`` target *is* reached, at the finer ``h~6e-5`` that first-order
  convergence requires.

PLAN DEVIATION (documented): Script #1 asserts ``err < 1e-5`` *at h=1e-3*. The
basic BUG integrator is first order, so at ``h=1e-3`` the error is ~8.4e-5 and
scales as exactly O(h) (ratio 2.00 per halving, verified); ``< 1e-5`` is only
attainable near ``h~6e-5``. We keep the plan's problem and reference but assert
against the method's true first-order behaviour rather than the optimistic
constant. The exact analytic reference ``Y(t) = e^{-tB} Y0 e^{-tB}`` (B is
symmetric) agrees with the RK45 reference to ~1e-11.

To stay DRY, the reusable pieces (``build_operator``, ``F``, ``bug_step``) are
imported from :mod:`kvdlra.integrators.bug` rather than redefined here.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp

from kvdlra.integrators.bug import F, bug_step, build_operator


def _initial_state(n: int, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic start state ``(U0, S0, V0, B)``.

    ``U0, V0`` are orthonormal ``(n, r)``; ``S0`` is the diagonal core with
    geometric singular values ``10**(-i)``; ``B`` is the Lyapunov operator.
    """
    rng = np.random.default_rng(0)
    B = build_operator(n)
    U0, _ = np.linalg.qr(rng.standard_normal((n, r)))
    V0, _ = np.linalg.qr(rng.standard_normal((n, r)))
    S0 = np.diag(10.0 ** (-np.arange(r)))
    return U0, S0, V0, B


def _exact_Y(B: np.ndarray, Y0: np.ndarray, T: float) -> np.ndarray:
    """Exact solution of ``Y' = -(B Y + Y B^T)`` at time ``T``.

    For symmetric ``B`` the flow has the closed form
    ``Y(T) = e^{-TB} Y0 e^{-TB^T}``.
    """
    E = sla.expm(-T * B)
    return np.asarray(E @ Y0 @ E.T)


def _integrate_bug(
    U: np.ndarray,
    S: np.ndarray,
    V: np.ndarray,
    B: np.ndarray,
    h: float,
    nsteps: int,
) -> np.ndarray:
    """Run ``nsteps`` BUG steps and return the reconstructed ``Y = U S V^T``."""
    for _ in range(nsteps):
        U, S, V = bug_step(U, S, V, B, h)
    return np.asarray(U @ S @ V.T)


def test_bug_tracks_reference() -> None:
    """BUG tracks a dense RK45 reference closely at the plan's ``h=1e-3``.

    Basic (unconventional) BUG is first order, so the error here is ~8.4e-5;
    the tighter behaviour is pinned by the convergence and fine-``h`` tests.
    """
    n, r, T, h = 100, 8, 0.1, 1e-3
    U0, S0, V0, B = _initial_state(n, r)
    Y0 = U0 @ S0 @ V0.T

    # Dense reference (ode45-grade): solve the full n^2 ODE.
    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        Y = y.reshape(n, n)
        return F(Y, B).reshape(-1)

    sol = solve_ivp(
        rhs,
        (0.0, T),
        Y0.reshape(-1),
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        t_eval=[T],
    )
    Y_ref = sol.y[:, -1].reshape(n, n)

    Y_bug = _integrate_bug(U0, S0, V0, B, h, round(T / h))
    err = float(np.linalg.norm(Y_bug - Y_ref, ord="fro"))
    print(f"BUG error vs. reference (h={h}): {err:.3e}")
    assert err < 2e-4, f"BUG diverged: {err}"


def test_bug_first_order_convergence() -> None:
    """Halving ``h`` halves the error: basic BUG is first order in time.

    This is the real correctness guarantee. It validates the substep ordering
    (K & L computed in parallel from the OLD factors, then S integrated forward)
    by confirming clean O(h) convergence to the exact analytic solution.
    """
    n, r, T = 100, 8, 0.1
    U0, S0, V0, B = _initial_state(n, r)
    Y0 = U0 @ S0 @ V0.T
    Y_exact = _exact_Y(B, Y0, T)

    errs: list[float] = []
    for h in (1e-3, 5e-4):
        Y_bug = _integrate_bug(U0, S0, V0, B, h, round(T / h))
        errs.append(float(np.linalg.norm(Y_bug - Y_exact, ord="fro")))
    ratio = errs[0] / errs[1]
    print(f"convergence errs={errs[0]:.3e}, {errs[1]:.3e}  ratio={ratio:.2f}")
    assert 1.8 < ratio < 2.2, f"expected first-order ratio ~2, got {ratio:.3f}"


def test_bug_meets_plan_threshold_at_fine_h() -> None:
    """The plan's headline ``err < 1e-5`` is reached at the finer ``h`` it needs.

    First-order BUG hits ~1e-5 near ``h~6e-5`` (8x finer than the plan's
    ``h=1e-3``), exactly as the O(h) convergence predicts.
    """
    n, r, T, h = 100, 8, 0.1, 6.25e-5
    U0, S0, V0, B = _initial_state(n, r)
    Y0 = U0 @ S0 @ V0.T
    Y_exact = _exact_Y(B, Y0, T)

    Y_bug = _integrate_bug(U0, S0, V0, B, h, round(T / h))
    err = float(np.linalg.norm(Y_bug - Y_exact, ord="fro"))
    print(f"BUG error vs. exact (h={h}): {err:.3e}")
    assert err < 1e-5, f"expected < 1e-5 at h={h}, got {err}"


if __name__ == "__main__":
    test_bug_tracks_reference()
    test_bug_first_order_convergence()
    test_bug_meets_plan_threshold_at_fine_h()
    print("OK")
