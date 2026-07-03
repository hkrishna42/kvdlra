"""Parity + correctness tests for the blocked torch BUG tracker.

Validates :func:`kvdlra.integrators.streaming_torch.blocked_bug_subspace`
against the Week-2-validated numpy tracker
:class:`kvdlra.integrators.streaming.StreamingBUG` and the truncated-SVD oracle.
The load-bearing guarantees (see the module docstring):

* ``block_size == 1`` reproduces the numpy per-token tracker to ~fp precision;
* ``block_size == T`` equals the Eckart--Young (truncated-SVD) oracle;
* intermediate block sizes land in the ``[oracle, numpy-stream]`` fidelity band;
* an exactly rank-``r`` input is reconstructed exactly at any block size.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import torch

from kvdlra.integrators.streaming import StreamingBUG
from kvdlra.integrators.streaming_torch import blocked_bug_project, blocked_bug_subspace


def _rel_err(m: npt.NDArray[np.float64], u: npt.NDArray[np.float64]) -> float:
    resid = m - u @ (u.T @ m)
    return float(np.linalg.norm(resid) / np.linalg.norm(m))


def _oracle_err(m: npt.NDArray[np.float64], r: int) -> float:
    u, _, _ = np.linalg.svd(m, full_matrices=False)
    return _rel_err(m, u[:, :r])


def _heavy_tailed(n: int, t: int, true_rank: int, seed: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, true_rank))
    b = rng.standard_normal((true_rank, t))
    return a @ b + 0.05 * rng.standard_normal((n, t))


def test_block1_matches_numpy_streaming() -> None:
    # block_size == 1 is the per-token tracker: same algorithm, so the
    # reconstruction error must match StreamingBUG to ~fp precision.
    m = _heavy_tailed(512, 600, true_rank=40, seed=0)
    mt = torch.from_numpy(m)
    for r in (16, 32, 64):
        sb = StreamingBUG(n_features=512, rank_cap=r)
        sb.update_many(m)
        u = blocked_bug_subspace(mt, rank_cap=r, block_size=1, compute_dtype=torch.float64)
        assert _rel_err(m, u.numpy()) == pytest.approx(sb.reconstruction_error(m), abs=1e-4)


def test_blockT_equals_oracle() -> None:
    # A single augmented step over all columns == the truncated-SVD oracle.
    m = _heavy_tailed(512, 400, true_rank=40, seed=1)
    mt = torch.from_numpy(m)
    for r in (16, 32, 64):
        u = blocked_bug_subspace(mt, rank_cap=r, block_size=400, compute_dtype=torch.float64)
        assert _rel_err(m, u.numpy()) == pytest.approx(_oracle_err(m, r), abs=1e-6)


def test_intermediate_blocks_in_fidelity_band() -> None:
    # Intermediate block sizes sit between the oracle (best) and the per-token
    # tracker, within a small tolerance -- never worse than streaming, never
    # better than the oracle.
    m = _heavy_tailed(512, 512, true_rank=40, seed=2)
    mt = torch.from_numpy(m)
    r = 32
    sb = StreamingBUG(n_features=512, rank_cap=r)
    sb.update_many(m)
    stream_err = sb.reconstruction_error(m)
    oracle_err = _oracle_err(m, r)
    for bs in (16, 64, 128):
        e = _rel_err(m, blocked_bug_subspace(mt, r, bs, compute_dtype=torch.float64).numpy())
        assert oracle_err - 1e-6 <= e <= stream_err + 1e-3


def test_exact_rank_r_reconstructed_exactly() -> None:
    rng = np.random.default_rng(3)
    r, n, t = 5, 512, 200
    m = rng.standard_normal((n, r)) @ rng.standard_normal((r, t))  # exactly rank r
    mt = torch.from_numpy(m)
    for bs in (1, 32, 200):
        recon = blocked_bug_project(mt, rank_cap=r, block_size=bs, compute_dtype=torch.float64)
        assert torch.allclose(recon, mt, atol=1e-8)


def test_rank_cap_respected() -> None:
    m = torch.from_numpy(_heavy_tailed(128, 300, true_rank=40, seed=4))
    for r in (8, 16, 40):
        u = blocked_bug_subspace(m, rank_cap=r, block_size=64)
        assert u.shape[1] <= r


def test_bf16_storage_runs_fp32_core() -> None:
    # bf16 storage must be accepted (core runs in fp32); the projection comes
    # back in the input's storage dtype and is a sane low-rank approximation.
    m64 = _heavy_tailed(512, 256, true_rank=30, seed=5)
    m_bf16 = torch.from_numpy(m64).to(torch.bfloat16)
    recon = blocked_bug_project(m_bf16, rank_cap=64, block_size=64)
    assert recon.dtype == torch.bfloat16
    # Direct reconstruction error vs the original fp64 data (recon is the
    # approximation of M, not a basis -- compare it to M directly).
    recon64 = recon.to(torch.float64).numpy()
    err = float(np.linalg.norm(m64 - recon64) / np.linalg.norm(m64))
    assert 0.0 <= err < 0.2  # rank-64 on ~rank-30 data: small error even via bf16


@pytest.mark.parametrize("block_size", [64, 104])  # == n_features and > n_features
def test_large_block_does_not_degenerate(block_size: int) -> None:
    # Regression (docs/week5.md "Follow-up found during the ablation"): when
    # rank_cap + block_size > n_features the augmented [U | Q] basis would exceed
    # R^n and degenerate -- a silent ~5x-oracle error at block_size == n_features
    # and a hard torch.cat shape crash at block_size > n_features. Capping the
    # residual QR to n_features - rank new directions keeps [U | Q] within R^n, so
    # the tracker stays near-oracle and never crashes.
    n = 64
    m = torch.from_numpy(_heavy_tailed(n, 300, true_rank=48, seed=11))
    r = 32  # r + block_size > n for both block sizes -> exercises the cap
    u = blocked_bug_subspace(m, rank_cap=r, block_size=block_size, compute_dtype=torch.float64)
    assert u.shape[1] <= r
    gram = u.mT @ u  # basis stays genuinely orthonormal (no degenerate directions)
    assert torch.allclose(gram, torch.eye(u.shape[1], dtype=torch.float64), atol=1e-9)
    err = _rel_err(m.numpy(), u.numpy())
    assert err <= _oracle_err(m.numpy(), r) * 1.05  # near-oracle, not the ~5x blow-up


def test_guards() -> None:
    m = torch.zeros(16, 10)
    with pytest.raises(ValueError, match="rank_cap"):
        blocked_bug_subspace(m, rank_cap=0)
    with pytest.raises(ValueError, match="block_size"):
        blocked_bug_subspace(m, rank_cap=4, block_size=0)
