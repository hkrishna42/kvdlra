"""FD numerics: the shared augmentation, the SVD core's eigh fallback, and the streams
that crashed the Week-20 swap pod.

The archived FD arm died in ``linalg.svd`` ("too many repeated singular values"): its own
augmentation kept the residual's plain QR, so a stream whose block already lies in the
tracked subspace fed the core sixteen junk directions per step, and FD's shrinkage then
zeroed most of the spectrum. ``fd_step`` now runs the same rank-revealing ``_augment`` the
incremental-SVD step does, and ``_svd_core`` falls back to an eigendecomposition of the
Gram when LAPACK still refuses to converge.
"""

from __future__ import annotations

import pytest
import torch

from kvdlra.tracker.isvd import _svd_core, fd_step, isvd_step, orth_error


def test_svd_core_falls_back_to_eigh_on_linalg_error(monkeypatch: pytest.MonkeyPatch) -> None:
    g = torch.Generator().manual_seed(0)
    b = torch.randn(40, 56, generator=g)
    u_ref, s_ref, _ = torch.linalg.svd(b, full_matrices=False)
    real_svd = torch.linalg.svd

    def boom(*a: object, **k: object) -> tuple[torch.Tensor, ...]:
        raise torch.linalg.LinAlgError(  # type: ignore[attr-defined]
            "linalg.svd failed to converge"
        )

    monkeypatch.setattr(torch.linalg, "svd", boom)
    u, s = _svd_core(b)
    monkeypatch.setattr(torch.linalg, "svd", real_svd)
    assert torch.allclose(s, s_ref, atol=1e-5)
    assert torch.allclose((u * s) @ (u * s).mT, (u_ref * s_ref) @ (u_ref * s_ref).mT, atol=1e-4)


def test_svd_core_second_attempt_subtracts_its_own_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """R1: the jittered retry adds ``jitter`` to the GRAM to make ``eigh`` converge; that
    jitter must not survive into the recovered singular values. Un-subtracted, every
    numerically-zero singular value comes back as ``sqrt(jitter)`` instead of 0 -- exactly
    the tail FD's shrinkage is supposed to zero. float64 + a genuinely rank-10-in-40 ``b_fac``
    separates the true (near machine-epsilon) tail from ``jitter`` (~1e-6) by nine orders of
    magnitude, so the two behaviours cannot be confused.
    """
    g = torch.Generator().manual_seed(3)
    q = torch.linalg.qr(torch.randn(40, 10, generator=g, dtype=torch.float64))[0]
    b = q @ torch.randn(10, 56, generator=g, dtype=torch.float64)  # true rank 10 in 40 rows
    _, s_ref, _ = torch.linalg.svd(b, full_matrices=False)
    real_eigh = torch.linalg.eigh
    eigh_calls = 0

    def boom_svd(*a: object, **k: object) -> tuple[torch.Tensor, ...]:
        raise torch.linalg.LinAlgError("linalg.svd failed to converge")  # type: ignore[attr-defined]

    def eigh_fails_once(*a: object, **k: object) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal eigh_calls
        eigh_calls += 1
        if eigh_calls == 1:
            raise torch.linalg.LinAlgError("linalg.eigh failed to converge")  # type: ignore[attr-defined]
        return real_eigh(*a, **k)  # type: ignore[no-any-return]

    monkeypatch.setattr(torch.linalg, "svd", boom_svd)
    monkeypatch.setattr(torch.linalg, "eigh", eigh_fails_once)
    _, s = _svd_core(b)
    assert eigh_calls == 2  # the plain Gram failed once; the jittered retry succeeded
    assert torch.allclose(s[:10], s_ref[:10], atol=1e-4)
    assert torch.allclose(s[10:], torch.zeros_like(s[10:]), atol=1e-6)


def test_fd_shrinkage_repeated_zero_spectrum_does_not_raise() -> None:
    # a stream whose augmented core carries many exact zeros after shrinkage (the swap-pod
    # crash class): rank 6 in 256 features, so every block after the first is already in
    # the tracked subspace and the residual is numerically null.
    g = torch.Generator().manual_seed(1)
    q = torch.linalg.qr(torch.randn(256, 6, generator=g))[0]
    m = q @ torch.randn(6, 16 * 300, generator=g)
    u = b = None
    for s in range(0, m.shape[1], 16):
        u, b, _ = fd_step(u, b, m[:, s : s + 16], 64)
    assert u is not None and b is not None
    assert orth_error(u) < 1e-4 and torch.isfinite(b).all()


def test_fd_matches_isvd_subspace_when_no_tail() -> None:
    """With ``rank_cap == n_features`` the augmented core never has a tail, so FD's
    shrinkage subtracts zero and the two steps must track the same subspace -- the pin
    that FD differs from incremental SVD in the shrinkage and in nothing else."""
    g = torch.Generator().manual_seed(2)
    m = torch.randn(64, 320, generator=g)
    ua = ba = ub = bb = None
    for s in range(0, 320, 16):
        ua, ba, _ = isvd_step(ua, ba, m[:, s : s + 16], 64)  # cap == n: nothing discarded
        ub, bb, _ = fd_step(ub, bb, m[:, s : s + 16], 64)
    assert ua is not None and ub is not None
    assert torch.allclose(ua @ ua.mT, ub @ ub.mT, atol=1e-5)


def test_fd_survives_the_ratchet_stream() -> None:
    from tests.test_orth_guard import ratchet_stream

    m = ratchet_stream(t=150 * 16)
    u = b = None
    for s in range(0, m.shape[1], 16):
        u, b, _ = fd_step(u, b, m[:, s : s + 16], 256)
    assert u is not None and torch.isfinite(u).all()
