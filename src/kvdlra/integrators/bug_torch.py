"""Torch port of the BUG integrator for mixed-precision testing.

This mirrors :func:`kvdlra.integrators.bug.bug_step` (Ceruti--Lubich 2022,
arXiv:2010.02022 §3.1) using PyTorch tensors, and exists specifically to study
**§8 pitfall #4 of the project plan: "fp16/bf16 in the BUG core."**

The pitfall, in brief: the K-step and L-step each end in a QR factorization.
QR (and the matmuls feeding it) on fp16/bf16 matrices with condition numbers
> 1e3 -- typical for KV-cache streams -- loses 2-3 decimal digits *per step*,
and bf16's 8-bit mantissa makes the orthonormalization especially lossy. The
fix is to run the integrator's linear algebra (QR, matmuls, RK2 substeps) in
**fp32**, regardless of how the factors are *stored*. Storage may be bf16 (it
shares fp32's exponent range, so it won't overflow), but the math is fp32.

Accordingly, :func:`bug_step_torch` upcasts the input factors to
``compute_dtype`` (default ``torch.float32``), runs the entire BUG step there,
and casts the returned factors back to the *storage* dtype of the inputs. The
model forward elsewhere stays bf16; only this integrator core is fp32.

NOTE: library code -- no ``print``/I/O here.

References
----------
G. Ceruti and C. Lubich, "An unconventional robust integrator for dynamical
low-rank approximation," BIT Numer. Math. 62 (2022) 23-44, arXiv:2010.02022,
§3.1. Project plan §8, pitfall #4 ("fp16 in the BUG core").
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = ["F_torch", "bug_step_torch", "rk2_step_torch"]


def F_torch(Y: Tensor, B: Tensor) -> Tensor:
    """Lyapunov right-hand side ``F(Y) = -(B @ Y + Y @ B.T)`` (torch).

    Torch analogue of :func:`kvdlra.integrators.bug.F`. ``Y`` and ``B`` are
    assumed to already be in the desired compute dtype/device.

    Parameters
    ----------
    Y:
        Current state matrix, shape ``(m, p)``.
    B:
        Operator matrix, shape ``(m, m)``.

    Returns
    -------
    Tensor
        ``F(Y)`` with the same shape/dtype as ``Y``.
    """
    return -(B @ Y + Y @ B.mT)


def rk2_step_torch(
    Y0: Tensor,
    rhs: Callable[[Tensor], Tensor],
    h: float,
) -> Tensor:
    """One explicit RK2 (Heun / trapezoidal) step for ``Y' = rhs(Y)`` (torch).

    Torch analogue of :func:`kvdlra.integrators.bug.rk2_step`::

        k1 = rhs(Y0)
        k2 = rhs(Y0 + h * k1)
        Y1 = Y0 + 0.5 * h * (k1 + k2)

    Parameters
    ----------
    Y0:
        State at the start of the step.
    rhs:
        Vector field mapping a state tensor to its derivative of equal shape.
    h:
        Step size.

    Returns
    -------
    Tensor
        State after one RK2 step.
    """
    k1 = rhs(Y0)
    k2 = rhs(Y0 + h * k1)
    return Y0 + 0.5 * h * (k1 + k2)


def bug_step_torch(
    U: Tensor,
    S: Tensor,
    V: Tensor,
    B: Tensor,
    h: float,
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    """One BUG step in ``compute_dtype``, returned in the inputs' storage dtype.

    Mirrors :func:`kvdlra.integrators.bug.bug_step` (Ceruti--Lubich 2022,
    arXiv:2010.02022 §3.1): the K-step and L-step are computed **in parallel
    from the OLD factors** ``(U, S, V)``, then the S-step is integrated
    **forward in time** using the new bases ``(U1, V1)``. Each substep ODE is
    advanced with a single :func:`rk2_step_torch`.

    Mixed precision (project plan §8, pitfall #4 -- "fp16/bf16 in the BUG
    core"): the *storage* dtype is taken from ``U`` (e.g. ``bfloat16``), but
    **all** linear algebra -- QR, matmuls, and the RK2 substeps -- is performed
    in ``compute_dtype`` (default ``torch.float32``). The returned factors
    ``(U1, S1, V1)`` are cast back to the storage dtype. This keeps the
    error-sensitive QR/orthonormalization in fp32 while allowing bf16 storage.

    Parameters
    ----------
    U:
        Left orthonormal factor, shape ``(n, r)``. Its dtype defines the
        *storage* dtype of the returned factors.
    S:
        Core matrix, shape ``(r, r)``.
    V:
        Right orthonormal factor, shape ``(n, r)``.
    B:
        Lyapunov operator, shape ``(n, n)``.
    h:
        Step size.
    compute_dtype:
        Dtype in which to run QR / matmuls / RK2 substeps. Default
        ``torch.float32``. Passing ``torch.bfloat16`` here runs the QR in bf16
        and is expected to be materially less accurate -- the
        ``test_bug_pure_bfloat16_qr_is_worse`` test exercises exactly that
        degradation to motivate keeping the core in fp32.

    Returns
    -------
    tuple of Tensor
        ``(U1, S1, V1)`` cast back to the storage dtype of ``U``, with
        ``Y_new approx U1 @ S1 @ V1.T``.

    References
    ----------
    G. Ceruti and C. Lubich, arXiv:2010.02022 §3.1. Project plan §8, pitfall #4.
    """
    storage_dtype = U.dtype

    # Upcast everything into the compute dtype: the QR/matmuls below are the
    # precision-critical part (pitfall #4), so they must not run in bf16/fp16
    # storage precision.
    Uc = U.to(compute_dtype)
    Sc = S.to(compute_dtype)
    Vc = V.to(compute_dtype)
    Bc = B.to(compute_dtype)

    # --- K-step: K' = F(K V^T) V, K(0) = U S  (uses OLD U, S, V) ---
    K0 = Uc @ Sc
    K1 = rk2_step_torch(K0, lambda K: F_torch(K @ Vc.mT, Bc) @ Vc, h)
    U1, _ = torch.linalg.qr(K1)
    M = U1.mT @ Uc  # r x r

    # --- L-step: L' = F(U L^T)^T U, L(0) = V S^T  (uses OLD U, S, V) ---
    L0 = Vc @ Sc.mT
    L1 = rk2_step_torch(L0, lambda L: F_torch(Uc @ L.mT, Bc).mT @ Uc, h)
    V1, _ = torch.linalg.qr(L1)
    N = V1.mT @ Vc  # r x r

    # --- S-step (forward in time): S' = U1^T F(U1 S V1^T) V1, S(0) = M S N^T --
    S0 = M @ Sc @ N.mT
    S1 = rk2_step_torch(S0, lambda Sm: U1.mT @ F_torch(U1 @ Sm @ V1.mT, Bc) @ V1, h)

    # Cast the updated factors back to the storage dtype (bf16/fp16/fp32).
    return (
        U1.to(storage_dtype),
        S1.to(storage_dtype),
        V1.to(storage_dtype),
    )
