"""Tests for :class:`kvdlra.integrators.frequent_directions.FrequentDirections`.

The FD sketch is the non-DLRA subspace-tracker baseline (``docs/
week7-dominance.md``). These pin: exact recovery of a genuinely low-rank stream,
the deterministic covariance-error bound, API parity with the other trackers,
and scale-invariance of the metric.
"""

from __future__ import annotations

import numpy as np
import pytest

from kvdlra.integrators.frequent_directions import FrequentDirections
from kvdlra.lowrank import truncated_svd_recon


def test_exact_recovery_of_low_rank_stream() -> None:
    # A rank-5 matrix streamed through an FD sketch of rank>=5 must reconstruct
    # to ~machine precision (the sketch's top directions span the whole range).
    rng = np.random.default_rng(0)
    u = np.linalg.qr(rng.standard_normal((64, 5)))[0]
    m = u @ rng.standard_normal((5, 400))
    fd = FrequentDirections(64, rank=5, ell=12)
    fd.update_many(m)
    assert fd.reconstruction_error(m) < 1e-9


def test_covariance_bound_holds() -> None:
    # Ghashami et al.: ||M M^T - B^T B||_2 <= ||M - M_k||_F^2 / (ell - k).
    rng = np.random.default_rng(1)
    m = rng.standard_normal((40, 600))
    ell, k = 20, 8
    fd = FrequentDirections(40, rank=k, ell=ell)
    fd.update_many(m)
    b = fd._sketch
    cov_err = float(np.linalg.norm(m @ m.T - b.T @ b, 2))
    _u, s, _vt = np.linalg.svd(m, full_matrices=False)
    tail = float(np.sum(s[k:] ** 2))  # ||M - M_k||_F^2
    assert cov_err <= tail / (ell - k) + 1e-6


def test_near_oracle_on_heavy_tailed_spectrum() -> None:
    # On a heavy-tailed (KV-like) spectrum FD with ell=2r should track close to
    # the truncated-SVD oracle at the same projection rank.
    rng = np.random.default_rng(2)
    scales = np.geomspace(1.0, 1e-3, 32)
    m = (rng.standard_normal((64, 32)) * scales) @ rng.standard_normal((32, 500))
    r = 8
    fd = FrequentDirections(64, rank=r, ell=2 * r)
    fd.update_many(m)
    oracle = truncated_svd_recon(m, r)
    assert fd.reconstruction_error(m) <= oracle + 0.03


def test_scale_invariance() -> None:
    rng = np.random.default_rng(3)
    m = rng.standard_normal((32, 300))
    fd1 = FrequentDirections(32, rank=6)
    fd1.update_many(m)
    fd2 = FrequentDirections(32, rank=6)
    fd2.update_many(1e4 * m)
    assert abs(fd1.reconstruction_error(m) - fd2.reconstruction_error(1e4 * m)) < 1e-9


def test_api_parity_and_validation() -> None:
    fd = FrequentDirections(16, rank=4)
    assert fd.ell == 8  # default 2r
    m = np.random.default_rng(4).standard_normal((16, 50))
    u = fd.subspace()  # empty sketch -> trivial basis
    assert u.shape[0] == 16
    fd.update_many(m)
    assert fd.subspace().shape == (16, 4)
    assert fd.project(m).shape == m.shape
    with pytest.raises(ValueError, match="rank must be <="):
        FrequentDirections(4, rank=8)
    with pytest.raises(ValueError, match="ell"):
        FrequentDirections(16, rank=4, ell=4)
    with pytest.raises(ValueError, match="shape"):
        fd.update_many(np.zeros((8, 3)))
