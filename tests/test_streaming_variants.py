"""Tests for the Week-4.5 alternative DLRA integrators (PSI + parallel-BUG).

Validates that :mod:`kvdlra.integrators.streaming_variants` runs through the same
block-streaming subspace-tracking contract as
:func:`kvdlra.integrators.streaming_torch.blocked_bug_subspace`:

* a single block (``block_size = T``) reproduces the truncated-SVD **oracle**;
* an exactly rank-``r`` input is recovered exactly at ``rank_cap >= r``;
* the returned basis is orthonormal and respects ``rank_cap``;
* in the KV operating range (moderate rank) both stay near the oracle -- PSI ties
  it closely, parallel-BUG trails by the cross-coupling it drops;
* guards reject bad ``rank_cap`` / ``block_size``.

The reconstruction metric depends only on ``span(U)``, so it is identical to the
incumbent's and the three integrators are compared apples-to-apples.
"""

from __future__ import annotations

import pytest
import torch

from kvdlra.integrators.streaming_torch import blocked_bug_subspace
from kvdlra.integrators.streaming_variants import (
    parallel_bug_project,
    parallel_bug_subspace,
    psi_project,
    psi_subspace,
)

# (name -> (subspace_fn, project_fn)) for the two new integrators.
VARIANTS = {
    "psi": (psi_subspace, psi_project),
    "parallel": (parallel_bug_subspace, parallel_bug_project),
}


def _rel_error(m: torch.Tensor, u: torch.Tensor) -> float:
    recon = u @ (u.mT @ m)
    return float(torch.linalg.norm(m - recon) / torch.linalg.norm(m))


def _oracle_error(m: torch.Tensor, r: int) -> float:
    u, s, vt = torch.linalg.svd(m, full_matrices=False)
    recon = (u[:, :r] * s[:r]) @ vt[:r]
    return float(torch.linalg.norm(m - recon) / torch.linalg.norm(m))


def _heavy_tailed(n: int, t: int, seed: int = 0) -> torch.Tensor:
    """A full-rank matrix with a geometrically decaying spectrum (KV-like)."""
    g = torch.Generator().manual_seed(seed)
    u, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(t, n, generator=g, dtype=torch.float64))
    sigma = torch.pow(0.9, torch.arange(n, dtype=torch.float64))
    m: torch.Tensor = (u * sigma) @ v.mT
    return m


@pytest.mark.parametrize("name", list(VARIANTS))
def test_block_t_equals_oracle(name: str) -> None:
    # One block (block_size == T) is just the seed: a truncated SVD == the oracle.
    subspace_fn, _ = VARIANTS[name]
    m = _heavy_tailed(48, 120, seed=1)
    for r in (4, 12, 24):
        u = subspace_fn(m, r, block_size=m.shape[1], compute_dtype=torch.float64)
        assert u.shape[1] <= r
        assert _rel_error(m, u) == pytest.approx(_oracle_error(m, r), abs=1e-10)


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("block_size", [30, 200])
def test_exact_low_rank_recovered(name: str, block_size: int) -> None:
    # An exactly rank-8 stream is recovered exactly at rank_cap >= 8, any block.
    subspace_fn, _ = VARIANTS[name]
    g = torch.Generator().manual_seed(2)
    a = torch.randn(64, 8, generator=g, dtype=torch.float64)
    b = torch.randn(8, 200, generator=g, dtype=torch.float64)
    m = a @ b
    u = subspace_fn(m, 8, block_size=block_size, compute_dtype=torch.float64)
    assert _rel_error(m, u) < 1e-8


@pytest.mark.parametrize("name", list(VARIANTS))
def test_basis_orthonormal_and_rank_capped(name: str) -> None:
    subspace_fn, _ = VARIANTS[name]
    m = _heavy_tailed(60, 250, seed=3)
    u = subspace_fn(m, 16, block_size=64, compute_dtype=torch.float64)
    assert u.shape[1] <= 16
    gram = u.mT @ u
    assert torch.allclose(gram, torch.eye(u.shape[1], dtype=torch.float64), atol=1e-9)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_project_shape_and_dtype(name: str) -> None:
    _, project_fn = VARIANTS[name]
    m = _heavy_tailed(40, 100, seed=4).to(torch.float32)
    recon = project_fn(m, 8, block_size=32)
    assert recon.shape == m.shape
    assert recon.dtype == m.dtype


def test_operating_range_near_oracle() -> None:
    # In the KV operating range (moderate rank), PSI ties the oracle closely while
    # parallel-BUG trails by the cross-coupling it decouples away -- but both track
    # far better than a trivial bound. Same matrix, same rank for all three.
    m = _heavy_tailed(128, 400, seed=5)
    r = 24
    oracle = _oracle_error(m, r)
    bug = _rel_error(m, blocked_bug_subspace(m, r, block_size=64, compute_dtype=torch.float64))
    psi = _rel_error(m, psi_subspace(m, r, block_size=64, compute_dtype=torch.float64))
    par = _rel_error(m, parallel_bug_subspace(m, r, block_size=64, compute_dtype=torch.float64))
    assert bug / oracle < 1.05  # incumbent hugs the oracle
    assert psi / oracle < 1.10  # PSI ties it closely in the operating range
    assert par / oracle < 1.60  # parallel-BUG trails (decoupled cross terms)
    assert bug <= psi + 1e-9  # BUG is at least as good as its rivals here
    assert bug <= par + 1e-9


def test_psi_is_fixed_rank() -> None:
    # PSI is basic (fixed-rank) projector-splitting: it cannot exceed the seed
    # rank, so at a streaming block size it is capped at rank <= block_size no
    # matter how large rank_cap is. (This is why an ablation that sweeps rank past
    # block_size must not read PSI's error as "instability" -- it is a rank cap.)
    m = _heavy_tailed(256, 800, seed=7)
    block = 64
    for rank_cap in (64, 128, 256):
        u = psi_subspace(m, rank_cap, block_size=block, compute_dtype=torch.float64)
        assert u.shape[1] == min(rank_cap, block)


def test_bug_and_parallel_are_rank_adaptive() -> None:
    # BUG and parallel-BUG grow the basis across blocks, so they reach any
    # rank_cap (up to rank_cap + block_size <= n_features) even when it exceeds the
    # streaming block size -- PSI cannot. This rank-adaptivity is BUG's genuine
    # streaming edge over fixed-rank projector-splitting.
    m = _heavy_tailed(512, 900, seed=8)
    block = 64
    for rank_cap in (128, 192, 256):  # all > block, all with rank_cap + block <= 512
        for fn in (blocked_bug_subspace, parallel_bug_subspace):
            u = fn(m, rank_cap, block_size=block, compute_dtype=torch.float64)
            assert u.shape[1] == rank_cap


@pytest.mark.parametrize("name", list(VARIANTS))
def test_guards(name: str) -> None:
    subspace_fn, _ = VARIANTS[name]
    m = _heavy_tailed(16, 40, seed=6)
    with pytest.raises(ValueError, match="rank_cap"):
        subspace_fn(m, 0)
    with pytest.raises(ValueError, match="block_size"):
        subspace_fn(m, 4, block_size=0)
