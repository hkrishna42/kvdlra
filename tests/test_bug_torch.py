"""Mixed-precision (fp32 / bf16) tests for the torch BUG port.

Strategy: the high-precision **numpy float64** BUG trajectory is treated as
ground truth, and we measure how much each torch dtype path degrades it. The
underlying algorithm correctness (fp64 BUG vs. a dense RK45 reference) is
already pinned by ``tests/test_bug_synthetic.py``; here we isolate the
*precision* behavior of :func:`kvdlra.integrators.bug_torch.bug_step_torch`.

These tests document **§8 pitfall #4 of the project plan ("fp16/bf16 in the
BUG core")**:

* fp32 storage + fp32 compute  -> tight  (rel err < 1e-3)
* bf16 storage + fp32 compute  -> loose but usable (rel err < 1e-1; ~6.4e-2)
* bf16 storage + bf16 compute  -> materially worse (QR in bf16 is the problem)

The third test is the *why*: it shows that moving the QR/matmul core from fp32
down to bf16 is what blows up the error, which is exactly the failure mode the
"keep the core in fp32" guidance is meant to avoid.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kvdlra.integrators.bug import bug_step, build_operator
from kvdlra.integrators.bug_torch import bug_step_torch

# Small-but-stiff synthetic problem (same family as the synthetic test, scaled
# down for torch-loop speed). Params chosen so the fp64 BUG trajectory is itself
# extremely accurate, making it a clean ground truth for the dtype comparison.
N: int = 64
R: int = 6
T: float = 0.05
H: float = 1e-3
NSTEPS: int = round(T / H)


def _initial_factors() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic ``(U0, S0, V0, B)`` start state in float64.

    Returns orthonormal ``U0, V0`` (shape ``(N, R)``), a diagonal core ``S0``
    with geometric singular values ``10**(-i)``, and the Lyapunov operator
    ``B`` (shape ``(N, N)``).
    """
    rng = np.random.default_rng(0)
    B = build_operator(N)
    U0, _ = np.linalg.qr(rng.standard_normal((N, R)))
    V0, _ = np.linalg.qr(rng.standard_normal((N, R)))
    S0 = np.diag(10.0 ** (-np.arange(R)))
    return U0, S0, V0, B


def _reference_Y() -> np.ndarray:
    """Ground-truth final ``Y = U S V^T`` from the numpy float64 BUG trajectory."""
    U, S, V, B = _initial_factors()
    for _ in range(NSTEPS):
        U, S, V = bug_step(U, S, V, B, H)
    return np.asarray(U @ S @ V.T)


def _run_torch_trajectory(
    storage_dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> np.ndarray:
    """Run the full BUG trajectory with the torch port at given dtypes.

    Parameters
    ----------
    storage_dtype:
        Dtype of the persisted factors (e.g. ``torch.bfloat16``).
    compute_dtype:
        Dtype used inside :func:`bug_step_torch` for QR/matmuls/RK2.

    Returns
    -------
    np.ndarray
        Final ``Y = U S V^T`` reconstructed in float64 (for comparison).
    """
    U0, S0, V0, B = _initial_factors()
    U = torch.from_numpy(U0).to(storage_dtype)
    S = torch.from_numpy(S0).to(storage_dtype)
    V = torch.from_numpy(V0).to(storage_dtype)
    # B is the operator; carry it in the storage dtype too so bug_step_torch
    # upcasts it like the factors. (It is upcast to compute_dtype internally.)
    Bt = torch.from_numpy(B).to(storage_dtype)

    for _ in range(NSTEPS):
        U, S, V = bug_step_torch(U, S, V, Bt, H, compute_dtype=compute_dtype)

    Y = U.to(torch.float64) @ S.to(torch.float64) @ V.to(torch.float64).mT
    return Y.numpy()


def _rel_fro_error(Y: np.ndarray, Y_ref: np.ndarray) -> float:
    """Relative Frobenius error ``||Y - Y_ref||_F / ||Y_ref||_F``."""
    return float(np.linalg.norm(Y - Y_ref, ord="fro") / np.linalg.norm(Y_ref, ord="fro"))


def test_bug_torch_float32() -> None:
    """fp32 storage + fp32 compute should closely track the fp64 trajectory."""
    torch.manual_seed(0)
    Y_ref = _reference_Y()
    Y = _run_torch_trajectory(torch.float32, torch.float32)
    err = _rel_fro_error(Y, Y_ref)
    print(f"[float32 storage / fp32 core] rel Fro error: {err:.3e}")
    assert err < 1e-3, f"fp32 BUG drifted too far from fp64: {err}"


def test_bug_torch_bfloat16_fp32core() -> None:
    """bf16 storage + fp32 compute core: the recommended mixed-precision mode.

    Storing factors in bf16 but doing the QR/matmuls in fp32 (pitfall #4's
    prescription) should stay well within a loose tolerance.
    """
    torch.manual_seed(0)
    Y_ref = _reference_Y()
    Y = _run_torch_trajectory(torch.bfloat16, torch.float32)
    err = _rel_fro_error(Y, Y_ref)
    print(f"[bfloat16 storage / fp32 core] rel Fro error: {err:.3e}")
    # Measured ~6.4e-2 on Mac CPU: bf16's ~2^-8 mantissa injects ~0.4%/element
    # storage noise that accumulates over the trajectory. The 1e-1 ceiling keeps
    # headroom for CUDA bf16 variance while still catching a regressed core.
    assert err < 1e-1, f"bf16-storage/fp32-core BUG too inaccurate: {err}"


def test_bug_pure_bfloat16_qr_is_worse() -> None:
    """Running the QR/matmul core in bf16 is materially worse than fp32 core.

    This is the concrete demonstration of §8 pitfall #4 ("fp16/bf16 in the BUG
    core"): the K- and L-steps each end in a QR, and a bf16 QR loses several
    mantissa bits per step. We therefore keep ``compute_dtype=float32`` in
    production; this test exists to document *why*. (Skips where a backend lacks
    a bf16 QR kernel, e.g. CPU LAPACK -- it runs on CUDA.)
    """
    torch.manual_seed(0)
    Y_ref = _reference_Y()

    err_fp32_core = _rel_fro_error(_run_torch_trajectory(torch.bfloat16, torch.float32), Y_ref)

    try:
        err_bf16_core = _rel_fro_error(_run_torch_trajectory(torch.bfloat16, torch.bfloat16), Y_ref)
    except (RuntimeError, NotImplementedError) as exc:  # pragma: no cover
        # Some backends lack a bf16 QR kernel; skip rather than fail there.
        pytest.skip(f"bf16 QR unsupported on this platform: {exc}")

    print(f"[bf16 core] rel Fro error: {err_bf16_core:.3e}  (vs fp32 core: {err_fp32_core:.3e})")

    # bf16-core must be clearly worse: either >= 2x the fp32-core error, or it
    # blows past the loose bf16-storage tolerance entirely.
    assert (err_bf16_core >= 2.0 * err_fp32_core) or (err_bf16_core > 1e-1), (
        "Expected bf16 QR core to be materially worse than fp32 core, but got "
        f"bf16-core={err_bf16_core:.3e} vs fp32-core={err_fp32_core:.3e}"
    )


def test_bf16_orthogonalization_loses_precision() -> None:
    """bf16 orthogonalization loses orthogonality vs. fp32 -- pitfall #4's mechanism.

    The integration test above (``test_bug_pure_bfloat16_qr_is_worse``) skips on
    CPU because LAPACK has no bf16 ``geqrf`` -- and CI is CPU-only, so that test
    never actually runs. This test exercises the *same mechanism* on CPU with a
    manual classical Gram-Schmidt (elementwise ops only, bf16-capable on CPU):
    orthonormalizing an ill-conditioned matrix in bf16 leaves a far larger
    ``||Q^T Q - I||`` than fp32, which is exactly why the BUG core's QR must stay
    in fp32 regardless of bf16 storage.
    """
    torch.manual_seed(0)
    n, r = 128, 16
    # Build an ill-conditioned (cond ~ 1e3, typical for KV streams) tall matrix.
    base = torch.randn(n, r, dtype=torch.float64)
    q0, _ = torch.linalg.qr(base)
    sigma = torch.logspace(0, -3, r, dtype=torch.float64)
    a = q0 * sigma  # columns scaled to a 1e3 condition number

    def gram_schmidt(mat: torch.Tensor) -> torch.Tensor:
        """Classical Gram-Schmidt in ``mat.dtype`` (no LAPACK; bf16-safe on CPU)."""
        q = torch.zeros_like(mat)
        for j in range(mat.shape[1]):
            v = mat[:, j].clone()
            for i in range(j):
                v = v - (q[:, i] @ v) * q[:, i]
            q[:, j] = v / v.norm()
        return q

    def orth_error(dtype: torch.dtype) -> float:
        q = gram_schmidt(a.to(dtype)).to(torch.float64)
        eye = torch.eye(r, dtype=torch.float64)
        return float((q.mT @ q - eye).norm())

    err_fp32 = orth_error(torch.float32)
    err_bf16 = orth_error(torch.bfloat16)
    print(f"orthogonality ||Q^T Q - I||: fp32={err_fp32:.2e}  bf16={err_bf16:.2e}")
    # bf16's 8-bit mantissa loses orthogonality by orders of magnitude here.
    assert err_bf16 > 10.0 * err_fp32, (
        f"expected bf16 Gram-Schmidt much worse than fp32, got "
        f"fp32={err_fp32:.2e}, bf16={err_bf16:.2e}"
    )


if __name__ == "__main__":
    test_bug_torch_float32()
    test_bug_torch_bfloat16_fp32core()
    test_bug_pure_bfloat16_qr_is_worse()
    test_bf16_orthogonalization_loses_precision()
    print("OK")
