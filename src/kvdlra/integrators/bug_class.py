"""Reusable fixed-rank BUG integrator class (Ceruti--Lubich 2022, §3.1).

This module wraps the from-scratch "basis-update & Galerkin" (BUG) step of
:mod:`kvdlra.integrators.bug` in a small, reusable, fully typed API so it can be
driven by an *arbitrary* matrix vector field ``rhs(Y) -> Ydot`` rather than only
the Lyapunov flow used to validate it. Two objects are exported:

* :class:`LowRankState` -- a frozen dataclass holding the factorization
  ``Y = U @ S @ V.T`` with ``U, V`` orthonormal ``(n, r)`` and ``S`` the
  ``r x r`` core, plus :meth:`LowRankState.full` (the dense reconstruction) and
  a :attr:`LowRankState.rank` property.
* :class:`BUG` -- the integrator. :meth:`BUG.step` performs one unconventional
  BUG step (Ceruti--Lubich 2022, arXiv:2010.02022 §3.1) for a generic ``rhs``;
  :meth:`BUG.integrate` chains several steps.

The unconventional (BUG) step has three substeps:

    1. **K-step** and **L-step**, computed *in parallel from the OLD factors*
       ``(U, S, V)``. Each integrates its substep ODE forward over ``[0, h]``
       and ends in a QR that yields a new orthonormal basis (``U1`` from the
       K-factor, ``V1`` from the L-factor). Neither uses the other's output.
    2. **S-step**, integrated *forward in time* with the NEW bases ``(U1, V1)``,
       starting from the old core projected onto them, ``S(0) = M @ S @ N.T``
       with ``M = U1.T @ U`` and ``N = V1.T @ V``.

The forward (not backward) S-step is what makes the integrator robust to small
singular values: the error bound of Kieri--Lubich--Walach 2016 (SIAM J. Numer.
Anal. 54(2):1020--1038, §2 / Thm 2.1) is *independent of the smallest singular
value* sigma_min, unlike the projector-splitting KSL scheme whose backward
S-step is unstable when sigma_min is tiny. Getting the substep order right --
K & L from the old factors, then S forward -- is the crux of the method.

The dense vector field ``rhs`` is supplied by the caller: for the synthetic
Lyapunov flow it is ``rhs = lambda Y: F(Y, B)`` with :func:`kvdlra.integrators.bug.F`,
but any ``Callable[[NDArray], NDArray]`` of matching shape works (e.g. a linear
flow ``rhs = lambda Y: A @ Y`` or a streaming KV-cache target field). Every
substep ODE is advanced with the single-source-of-truth explicit RK2 stepper
:func:`kvdlra.integrators.bug.rk2_step` (imported, not redefined).

NOTE: This is library code -- it contains no I/O or ``print`` calls.

References
----------
G. Ceruti and C. Lubich, "An unconventional robust integrator for dynamical
low-rank approximation," BIT Numer. Math. 62 (2022) 23--44, arXiv:2010.02022,
§3.1.

E. Kieri, C. Lubich and H. Walach, "Discretized dynamical low-rank
approximation in the presence of small singular values," SIAM J. Numer. Anal.
54(2) (2016) 1020--1038, §2 / Thm 2.1 (robust error bound independent of
sigma_min).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from kvdlra.integrators.bug import rk2_step

__all__ = ["BUG", "LowRankState"]

# Type alias for a dense matrix vector field Y -> Ydot (same shape in and out).
RHS = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


@dataclass(frozen=True)
class LowRankState:
    """Immutable low-rank factorization ``Y = U @ S @ V.T``.

    A small typed holder for the three factors of a rank-``r`` matrix in the
    convention of Ceruti--Lubich 2022 (arXiv:2010.02022): ``U`` and ``V`` are
    orthonormal ``(n, r)`` bases of the column and row spaces and ``S`` is the
    ``r x r`` core. The class is a frozen dataclass so a state is never mutated
    in place; :meth:`BUG.step` returns a *new* :class:`LowRankState`.

    The factors are stored as-is and are *not* re-orthonormalized on
    construction -- callers are expected to pass orthonormal ``U, V`` (as the
    integrator does after each QR). :meth:`full` materializes the dense matrix.

    Attributes
    ----------
    U:
        Left factor, shape ``(n, r)`` (orthonormal columns by convention).
    S:
        Core matrix, shape ``(r, r)``.
    V:
        Right factor, shape ``(n, r)`` (orthonormal columns by convention).
    """

    U: npt.NDArray[np.float64]
    S: npt.NDArray[np.float64]
    V: npt.NDArray[np.float64]

    @property
    def rank(self) -> int:
        """Rank ``r`` of the factorization (number of core columns)."""
        return int(self.S.shape[1])

    def full(self) -> npt.NDArray[np.float64]:
        """Dense reconstruction ``U @ S @ V.T``, shape ``(n, n)``.

        Returns
        -------
        npt.NDArray[np.float64]
            The materialized rank-``r`` matrix in float64.
        """
        return np.asarray(self.U @ self.S @ self.V.T, dtype=np.float64)


class BUG:
    """Fixed-rank unconventional BUG integrator for a generic vector field.

    Implements the unconventional "basis-update & Galerkin" integrator of
    Ceruti--Lubich 2022 (arXiv:2010.02022 §3.1) as a reusable object operating
    on a :class:`LowRankState`. Unlike :func:`kvdlra.integrators.bug.bug_step`,
    which is hard-wired to the Lyapunov field ``F(Y, B)``, this class takes the
    dense vector field ``rhs(Y) -> Ydot`` as an argument to :meth:`step`, so it
    works for any matrix ODE ``Y' = rhs(Y)`` (the synthetic Lyapunov flow, a
    linear flow ``A @ Y``, a streaming KV-cache target field, ...).

    The robust error bound of Kieri--Lubich--Walach 2016 (SIAM J. Numer. Anal.
    54(2), §2 / Thm 2.1) applies: because the S-step is integrated *forward* in
    time, the discretization error is independent of the smallest singular value
    sigma_min, so the method does not suffer the KSL backward-step instability
    when the low-rank approximation is nearly rank-deficient.

    The integrator is stateless -- it holds no problem data -- so a single
    instance can step many different states and fields.
    """

    def step(
        self,
        state: LowRankState,
        rhs: RHS,
        h: float,
    ) -> LowRankState:
        """Advance one unconventional BUG step of ``Y' = rhs(Y)`` over ``[0, h]``.

        Performs the three substeps of Ceruti--Lubich 2022 (arXiv:2010.02022
        §3.1) for a *generic* dense vector field ``rhs``. With
        ``Y = U @ S @ V.T`` the old factorization:

        **K-step** (left basis update)
            Integrate ``K' = rhs(K V^T) V`` with ``K(0) = U @ S`` over
            ``[0, h]``, then ``U1, _ = qr(K1)`` and record ``M = U1.T @ U``.

        **L-step** (right basis update)
            Integrate ``L' = rhs(U L^T)^T U`` with ``L(0) = V @ S.T`` over
            ``[0, h]``, then ``V1, _ = qr(L1)`` and record ``N = V1.T @ V``.

            The K- and L-steps are computed **in parallel from the OLD factors**
            ``(U, S, V)`` -- neither uses the other's output. This independence
            is the defining feature of the BUG family.

        **S-step** (Galerkin core update, integrated FORWARD in time)
            Using the NEW bases ``(U1, V1)``, integrate
            ``S' = U1^T rhs(U1 S V1^T) V1`` with ``S(0) = M @ S @ N.T`` over
            ``[0, h]``. The forward time direction (not backward, as in KSL) is
            what gives the method its robustness to small singular values
            (Kieri--Lubich--Walach 2016, §2 / Thm 2.1).

        Each substep ODE is advanced with a single
        :func:`kvdlra.integrators.bug.rk2_step` (the project's single source of
        truth for the RK2 stepper). For the Lyapunov field this reproduces
        :func:`kvdlra.integrators.bug.bug_step` step-for-step (to floating-point
        round-off) when ``rhs = lambda Y: F(Y, B)``.

        Parameters
        ----------
        state:
            Current low-rank factorization with orthonormal ``U, V`` (``n x r``)
            and core ``S`` (``r x r``).
        rhs:
            Dense matrix vector field; maps an ``(n, n)`` state to its time
            derivative of the same shape.
        h:
            Step size.

        Returns
        -------
        LowRankState
            The updated factorization after one step, with ``U1, V1``
            orthonormal ``(n, r)`` and core ``S1`` (``r x r``). The rank is
            preserved (fixed-rank integrator).
        """
        U, S, V = state.U, state.S, state.V

        # --- K-step: K' = rhs(K V^T) V, K(0) = U S  (uses OLD U, S, V) ---
        K0 = U @ S
        K1 = rk2_step(K0, lambda K: rhs(K @ V.T) @ V, h)
        U1, _ = np.linalg.qr(K1)
        M = U1.T @ U  # r x r

        # --- L-step: L' = rhs(U L^T)^T U, L(0) = V S^T  (uses OLD U, S, V) ---
        L0 = V @ S.T
        L1 = rk2_step(L0, lambda L: rhs(U @ L.T).T @ U, h)
        V1, _ = np.linalg.qr(L1)
        N = V1.T @ V  # r x r

        # --- S-step (forward in time): S' = U1^T rhs(U1 S V1^T) V1, S(0) = M S N^T ---
        S0 = M @ S @ N.T
        S1 = rk2_step(S0, lambda Sm: U1.T @ rhs(U1 @ Sm @ V1.T) @ V1, h)
        return LowRankState(U=U1, S=S1, V=V1)

    def integrate(
        self,
        state: LowRankState,
        rhs: RHS,
        h: float,
        nsteps: int,
    ) -> LowRankState:
        """Apply :meth:`step` ``nsteps`` times with fixed step size ``h``.

        Convenience driver that chains ``nsteps`` unconventional BUG steps of
        ``Y' = rhs(Y)`` starting from ``state``. ``nsteps`` must be
        non-negative; ``nsteps == 0`` returns the input ``state`` unchanged.

        Parameters
        ----------
        state:
            Initial low-rank factorization.
        rhs:
            Dense matrix vector field (see :meth:`step`).
        h:
            Step size for every step.
        nsteps:
            Number of BUG steps to take (``>= 0``).

        Returns
        -------
        LowRankState
            The factorization after ``nsteps`` steps.
        """
        for _ in range(nsteps):
            state = self.step(state, rhs, h)
        return state
