"""Rank-adaptive (augmented) BUG integrator (Ceruti--Kusch--Lubich 2022).

This module implements the *rank-adaptive* basis-update & Galerkin (BUG)
integrator of G. Ceruti, J. Kusch and C. Lubich, *A rank-adaptive robust
integrator for dynamical low-rank approximation* (BIT Numer. Math., 2022;
arXiv:2104.05247). It extends the fixed-rank BUG step of
:mod:`kvdlra.integrators.bug` (arXiv:2010.02022 §3.1) so that the rank of the
approximation can grow or shrink from one step to the next.

The construction (arXiv:2104.05247 §2, Eqs. (3)--(6)) augments each updated
basis with the *old* basis before the Galerkin step:

    1. **K-step.** Integrate ``K' = F(K V^T) V`` with ``K(0) = U @ S`` over
       ``[0, h]`` (exactly as in fixed-rank BUG), then orthonormalize the
       *augmented* stack ``[K1 | U]`` (an ``n x 2r`` matrix) to obtain
       ``Uhat`` with up to ``2r`` orthonormal columns.
    2. **L-step.** Integrate ``L' = F(U L^T)^T U`` with ``L(0) = V @ S^T``,
       then orthonormalize ``[L1 | V]`` to obtain ``Vhat``.

       The K- and L-steps are computed *in parallel from the OLD factors*
       ``(U, S, V)`` -- the defining feature of the BUG family.
    3. **Augmented Galerkin S-step.** Project the old core onto the augmented
       bases, ``Shat0 = (Uhat^T U) S (V^T Vhat)`` (shape ``2r x 2r``), and
       integrate it *forward in time*,
       ``Shat' = Uhat^T F(Uhat Shat Vhat^T) Vhat`` over ``[0, h]``.
    4. **Truncation.** Take the SVD ``Shat1 = P diag(sigma) Q^T`` and discard the
       smallest singular values, keeping the *smallest* new rank ``r1`` whose
       discarded Frobenius tail satisfies the criterion

           ( sum_{j > r1} sigma_j^2 )^{1/2} <= theta.

       The new factors are ``U1 = Uhat P[:, :r1]``, ``S1 = diag(sigma[:r1])``,
       ``V1 = Vhat Q[:, :r1]``; their rank ``r1`` may differ from the input ``r``.

Including the old basis in the augmentation guarantees that the new subspace
*contains* the old one, so the rank-adaptive step is at least as accurate as the
fixed-rank step and inherits its robustness to small singular values (the
forward, not backward, S-step). The augmentation adds at most ``r`` new
directions per step, so a rank deficit of more than ``r`` takes several steps to
fill in.

The fixed-rank vector field :func:`F` and the explicit RK2 stepper
:func:`rk2_step` are imported from :mod:`kvdlra.integrators.bug` (single source
of truth -- they are not redefined here).

NOTE: This is library code -- it contains no I/O or ``print`` calls. The
companion test ``tests/test_bug_adaptive.py`` drives it against the exact
``Y(t) = e^{-tB} Y0 e^{-tB}`` reference and checks rank growth/shrinkage.

References
----------
G. Ceruti, J. Kusch and C. Lubich, "A rank-adaptive robust integrator for
dynamical low-rank approximation," BIT Numer. Math. 62 (2022) 1149-1174,
arXiv:2104.05247, §2, Eqs. (3)--(6).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kvdlra.integrators.bug import F, rk2_step

__all__ = ["rank_adaptive_bug_step", "truncation_rank"]


def truncation_rank(sigma: npt.NDArray[np.float64], theta: float) -> int:
    """Smallest rank ``r1`` whose discarded singular-value tail is ``<= theta``.

    Implements the truncation criterion of Ceruti--Kusch--Lubich 2022
    (arXiv:2104.05247 §2): given the singular values ``sigma`` (assumed sorted
    in non-increasing order, as returned by :func:`numpy.linalg.svd`), choose the
    smallest ``r1`` in ``[1, len(sigma)]`` such that the Frobenius norm of the
    discarded tail satisfies

        ( sum_{j > r1} sigma_j^2 )^{1/2} <= theta.

    The tail norm is computed from the *largest* discarded values down, which is
    numerically stable. At least one singular value is always retained
    (``r1 >= 1``) even if ``theta`` exceeds the full Frobenius norm.

    Parameters
    ----------
    sigma:
        Non-increasing singular values, shape ``(k,)``.
    theta:
        Non-negative truncation tolerance on the discarded Frobenius tail.

    Returns
    -------
    int
        The chosen rank ``r1`` with ``1 <= r1 <= len(sigma)``.
    """
    k = int(sigma.shape[0])
    if k == 0:
        return 0
    # tail_norm[i] = sqrt(sum_{j >= i} sigma_j^2), i.e. the discarded Frobenius
    # norm if we keep the first i singular values. Built via a reverse cumulative
    # sum so it is evaluated from the smallest contributions upward.
    sq = sigma.astype(np.float64) ** 2
    tail_sq = np.concatenate([np.cumsum(sq[::-1])[::-1], np.zeros(1)])
    tail_norm = np.sqrt(tail_sq)  # length k + 1; tail_norm[i] for keeping i values
    # Smallest r1 in [1, k] with tail_norm[r1] <= theta; fall back to k.
    for r1 in range(1, k + 1):
        if tail_norm[r1] <= theta:
            return r1
    return k


def rank_adaptive_bug_step(
    U: npt.NDArray[np.float64],
    S: npt.NDArray[np.float64],
    V: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    h: float,
    theta: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Perform one rank-adaptive (augmented) BUG step.

    Implements the rank-adaptive integrator of Ceruti--Kusch--Lubich 2022,
    arXiv:2104.05247 §2 (Eqs. (3)--(6)), for the Lyapunov vector field :func:`F`.
    The current low-rank factorization is ``Y0 = U @ S @ V.T`` with ``U, V``
    orthonormal ``(n, r)`` and ``S`` the ``r x r`` core. One step produces new
    factors ``(U1, S1, V1)`` whose rank ``r1`` may grow or shrink relative to
    ``r`` according to the truncation tolerance ``theta``.

    **K-step / L-step (augmented bases).** As in fixed-rank BUG, integrate the
    K- and L-ODEs forward from the OLD factors. Then orthonormalize the
    *augmented* stacks ``[K1 | U]`` and ``[L1 | V]`` (economic QR of ``n x 2r``
    matrices) to obtain ``Uhat`` and ``Vhat`` with up to ``2r`` columns. Because
    the old bases are included, ``range(U) <= range(Uhat)`` and likewise for
    ``V``, so no accuracy of the old approximation is lost.

    **Augmented Galerkin S-step (forward in time).** Project the old core onto
    the augmented bases, ``Shat0 = (Uhat^T U) S (V^T Vhat)``, and integrate
    ``Shat' = Uhat^T F(Uhat Shat Vhat^T) Vhat`` forward over ``[0, h]``.

    **Truncation.** SVD ``Shat1 = P diag(sigma) Q^T`` and keep the smallest rank
    ``r1`` whose discarded tail obeys ``(sum_{j>r1} sigma_j^2)^{1/2} <= theta``
    (see :func:`truncation_rank`). Return ``U1 = Uhat P[:, :r1]``,
    ``S1 = diag(sigma[:r1])``, ``V1 = Vhat Q[:, :r1]``.

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
        Lyapunov operator, shape ``(n, n)`` (see
        :func:`kvdlra.integrators.bug.build_operator`).
    h:
        Step size.
    theta:
        Truncation tolerance on the discarded Frobenius tail (Eq. of §2).

    Returns
    -------
    tuple of npt.NDArray[np.float64]
        ``(U1, S1, V1)`` -- the updated factors with ``Y_new approx
        U1 @ S1 @ V1.T``. ``U1`` (``n x r1``) and ``V1`` (``n x r1``) are
        orthonormal and ``S1`` is ``r1 x r1`` diagonal; ``r1`` may differ from
        the input rank ``r``.

    References
    ----------
    G. Ceruti, J. Kusch and C. Lubich, "A rank-adaptive robust integrator for
    dynamical low-rank approximation," BIT Numer. Math. 62 (2022) 1149-1174,
    arXiv:2104.05247, §2, Eqs. (3)--(6).
    """
    # --- K-step: K' = F(K V^T) V, K(0) = U S  (uses OLD U, S, V) ---
    K0 = U @ S
    K1 = rk2_step(K0, lambda K: F(K @ V.T, B) @ V, h)
    # Augmented left basis: orthonormalize the n x 2r stack [K1 | U].
    Uhat, _ = np.linalg.qr(np.concatenate([K1, U], axis=1))

    # --- L-step: L' = F(U L^T)^T U, L(0) = V S^T  (uses OLD U, S, V) ---
    L0 = V @ S.T
    L1 = rk2_step(L0, lambda L: F(U @ L.T, B).T @ U, h)
    # Augmented right basis: orthonormalize the n x 2r stack [L1 | V].
    Vhat, _ = np.linalg.qr(np.concatenate([L1, V], axis=1))

    # --- Augmented Galerkin S-step (forward in time) ---
    # Project the old core onto the augmented bases, then integrate forward.
    Shat0 = (Uhat.T @ U) @ S @ (V.T @ Vhat)
    Shat1 = rk2_step(Shat0, lambda Sm: Uhat.T @ F(Uhat @ Sm @ Vhat.T, B) @ Vhat, h)

    # --- Truncate to the smallest rank meeting the Frobenius-tail criterion ---
    P, sigma, Qt = np.linalg.svd(Shat1, full_matrices=False)
    r1 = truncation_rank(sigma, theta)
    U1 = Uhat @ P[:, :r1]
    S1 = np.diag(sigma[:r1])
    V1 = Vhat @ Qt[:r1, :].T  # V1 = Vhat @ Q[:, :r1] with Q = Qt.T
    return U1, S1, V1
