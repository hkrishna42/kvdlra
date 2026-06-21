"""From-scratch BUG integrator (Ceruti--Lubich 2022, arXiv:2010.02022 §3.1).

This module implements the "basis-update & Galerkin" (BUG) low-rank integrator
of Ceruti & Lubich, *An unconventional robust integrator for dynamical low-rank
approximation* (BIT Numer. Math., 2022; arXiv:2010.02022). The unconventional
(BUG) step described in §3.1 of that paper has three substeps:

    1. **K-step** and **L-step**, computed *in parallel* from the OLD factors
       ``(U, S, V)``. Each produces an updated orthonormal basis (``U1`` from a
       QR of the K-factor, ``V1`` from a QR of the L-factor).
    2. **S-step**, integrated *forward in time* using the NEW bases
       ``(U1, V1)``, starting from the old core projected onto the new bases.

The forward (rather than backward) S-step is what makes the integrator robust
to small singular values without the projector-splitting minus-sign instability
of the original KSL scheme (Lubich--Oseledets 2014). Getting the substep order
right -- K & L from the old factors, then S forward -- is the crux of the method.

Test / validation problem (Ceruti--Lubich §6; cf. Ceruti--Kusch--Lubich §5.1):
a stiff matrix Lyapunov flow

    Y'(t) = F(Y) = -(B Y + Y B^T),
    B = V - 0.5 D,
    D = tridiag(-1, 2, -1),
    V = diag{ 1 - cos(2 pi j / n) }.

All substep ODEs are advanced with a single explicit RK2 (Heun/trapezoidal)
step, matching the reference script. The library functions here are deliberately
generic: :func:`F` and :func:`rk2_step` accept an arbitrary right-hand side so
they can be reused for streaming KV-cache flows elsewhere in the project.

NOTE: This is library code -- it contains no I/O or ``print`` calls. The
companion test (``tests/test_bug_synthetic.py``) drives it against a dense
``scipy.integrate.solve_ivp`` RK45 reference.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

__all__ = ["F", "bug_step", "build_operator", "rk2_step"]


def build_operator(n: int) -> npt.NDArray[np.float64]:
    """Build the Lyapunov operator ``B`` for the synthetic test flow.

    Constructs ``B = V - 0.5 * D`` where ``D = tridiag(-1, 2, -1)`` is the
    standard 1-D Laplacian stencil and ``V = diag{1 - cos(2 pi j / n)}`` with
    ``j`` ranging symmetrically over ``[-n//2, n//2)``. The flow right-hand side
    is then ``F(Y) = -(B @ Y + Y @ B.T)`` (see :func:`F`).

    Parameters
    ----------
    n:
        Dimension of the (square) state matrix ``Y`` is ``n x n``.

    Returns
    -------
    npt.NDArray[np.float64]
        The ``(n, n)`` operator matrix ``B`` in float64.
    """
    D = np.diag(2.0 * np.ones(n)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)
    j = np.arange(-n // 2, n // 2)
    V = np.diag(1.0 - np.cos(2 * np.pi * j / n))
    return (V - 0.5 * D).astype(np.float64)  # B; then F(Y) = -(B Y + Y B^T)


def F(Y: npt.NDArray[np.float64], B: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Lyapunov right-hand side ``F(Y) = -(B @ Y + Y @ B.T)``.

    This is the full-space vector field of the matrix ODE ``Y' = F(Y)``. It is
    kept generic (a plain function of ``Y`` and the operator ``B``) so that the
    same RHS can be reused for the dense reference solve and for every BUG
    substep via small ``lambda`` wrappers.

    Parameters
    ----------
    Y:
        Current state matrix, shape ``(m, p)`` (square ``(n, n)`` in the test).
    B:
        Operator matrix from :func:`build_operator`, shape ``(m, m)``.

    Returns
    -------
    npt.NDArray[np.float64]
        ``F(Y)`` with the same shape as ``Y``.
    """
    return -(B @ Y + Y @ B.T)


def rk2_step(
    Y0: npt.NDArray[np.float64],
    rhs: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    h: float,
) -> npt.NDArray[np.float64]:
    """Advance ``Y' = rhs(Y)`` by one explicit RK2 (Heun / trapezoidal) step.

    Uses the two-stage scheme

        k1 = rhs(Y0)
        k2 = rhs(Y0 + h * k1)
        Y1 = Y0 + 0.5 * h * (k1 + k2),

    which is second-order accurate. ``rhs`` is an arbitrary callable so this
    helper can advance the K-, L-, and S-substep ODEs (and any other matrix
    flow) without modification.

    Parameters
    ----------
    Y0:
        State at the start of the step.
    rhs:
        Vector field; maps a state array to its time derivative of equal shape.
    h:
        Step size.

    Returns
    -------
    npt.NDArray[np.float64]
        State after one RK2 step, same shape as ``Y0``.
    """
    k1 = rhs(Y0)
    k2 = rhs(Y0 + h * k1)
    return Y0 + 0.5 * h * (k1 + k2)


def bug_step(
    U: npt.NDArray[np.float64],
    S: npt.NDArray[np.float64],
    V: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    h: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Perform one BUG (basis-update & Galerkin) step.

    Implements exactly the unconventional integrator of Ceruti--Lubich 2022,
    arXiv:2010.02022 §3.1, for the Lyapunov vector field :func:`F`. The current
    low-rank factorization is ``Y = U @ S @ V.T`` with ``U, V`` orthonormal
    (``n x r``) and ``S`` the ``r x r`` core. The three substeps are:

    **K-step** (basis update for the range / left factor)
        Integrate ``K' = F(K V^T) V`` with ``K(0) = U @ S`` over ``[0, h]``,
        then orthonormalize: ``U1, _ = qr(K1)``. Record ``M = U1.T @ U``.

    **L-step** (basis update for the co-range / right factor)
        Integrate ``L' = F(U L^T)^T U`` with ``L(0) = V @ S.T`` over ``[0, h]``,
        then orthonormalize: ``V1, _ = qr(L1)``. Record ``N = V1.T @ V``.

        The K-step and L-step are computed **in parallel from the OLD factors**
        ``(U, S, V)`` -- neither uses the other's output. This independence is
        the defining feature of the BUG integrator.

    **S-step** (Galerkin core update, integrated FORWARD in time)
        Using the NEW bases ``(U1, V1)``, integrate
        ``S' = U1^T F(U1 S V1^T) V1`` with ``S(0) = M @ S @ N.T`` over
        ``[0, h]``. The forward time direction (not backward, as in KSL) is what
        gives the method its robustness to small singular values.

    Each substep ODE is advanced with a single :func:`rk2_step`.

    Parameters
    ----------
    U:
        Left orthonormal factor, shape ``(n, r)``.
    S:
        Core matrix, shape ``(r, r)``.
    V:
        Right orthonormal factor, shape ``(n, r)``.
    B:
        Lyapunov operator, shape ``(n, n)`` (see :func:`build_operator`).
    h:
        Step size.

    Returns
    -------
    tuple of npt.NDArray[np.float64]
        ``(U1, S1, V1)`` -- the updated factors after one step, with
        ``Y_new approx U1 @ S1 @ V1.T``. ``U1`` and ``V1`` are orthonormal.

    References
    ----------
    G. Ceruti and C. Lubich, "An unconventional robust integrator for
    dynamical low-rank approximation," BIT Numer. Math. 62 (2022) 23-44,
    arXiv:2010.02022, §3.1.
    """
    # --- K-step: K' = F(K V^T) V, K(0) = U S  (uses OLD U, S, V) ---
    K0 = U @ S
    K1 = rk2_step(K0, lambda K: F(K @ V.T, B) @ V, h)
    U1, _ = np.linalg.qr(K1)
    M = U1.T @ U  # r x r

    # --- L-step: L' = F(U L^T)^T U, L(0) = V S^T  (uses OLD U, S, V) ---
    L0 = V @ S.T
    L1 = rk2_step(L0, lambda L: F(U @ L.T, B).T @ U, h)
    V1, _ = np.linalg.qr(L1)
    N = V1.T @ V  # r x r

    # --- S-step (forward in time): S' = U1^T F(U1 S V1^T) V1, S(0) = M S N^T ---
    S0 = M @ S @ N.T
    S1 = rk2_step(S0, lambda Sm: U1.T @ F(U1 @ Sm @ V1.T, B) @ V1, h)
    return U1, S1, V1
