"""Tests for the streaming rank-adaptive BUG tracker (:mod:`kvdlra.integrators.streaming`).

:class:`kvdlra.integrators.streaming.StreamingBUG` processes a feature-by-token
matrix one column (token) at a time with augmented-BUG dynamics and tracks the
left (feature) subspace ``U``; its reconstruction model is the orthogonal
projection ``U U^T M`` (see that module's docstring for the BUG derivation). The
deterministic checks here pin its core contract for the Week-2 pilot:

#. the hard ``rank_cap`` is never exceeded across a stream;
#. an exactly-rank-``r`` input is reconstructed to ~machine precision (even with
   a larger cap), which is the augmented step's *exactness* property;
#. reconstruction error is non-increasing as the rank cap rises, and approaches
   the truncated-SVD oracle (within a small constant) on a synthetic low-rank +
   noise matrix;
#. the adaptive Frobenius-tail tolerance ``theta`` trades rank for accuracy
   monotonically;
#. ``mean_rank`` / ``rank_history`` bookkeeping is correct;
#. input validation and the no-new-direction branch behave.

All inputs are fixed-seed; no model or KV dump is required (the real-KV
validation lives in ``experiments/2026-w2-streaming-bug/run.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from kvdlra.integrators.streaming import StreamingBUG


def _orthonormal(n: int, r: int, seed: int) -> np.ndarray:
    """Orthonormal ``(n, r)`` factor from a fixed seed (QR of a Gaussian)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, r)))
    return np.asarray(q, dtype=np.float64)


def _exact_rank_matrix(n: int, t: int, r: int, seed: int) -> np.ndarray:
    """A deterministic exactly-rank-``r`` matrix ``(n, t)`` with decaying spectrum."""
    u = _orthonormal(n, r, seed)
    rng = np.random.default_rng(seed + 1)
    coeffs = rng.standard_normal((r, t))
    sigma = np.diag(2.0 ** -np.arange(r).astype(np.float64))  # decaying singular scale
    return np.asarray(u @ sigma @ coeffs, dtype=np.float64)


def _oracle_error(M: np.ndarray, r: int) -> float:
    """Truncated-SVD (Eckart--Young) relative Frobenius error at rank ``r``."""
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    mr = (u[:, :r] * s[:r]) @ vt[:r]
    return float(np.linalg.norm(M - mr) / np.linalg.norm(M))


# --------------------------------------------------------------------------- #
# 1. The hard rank cap is never exceeded.
# --------------------------------------------------------------------------- #
def test_rank_cap_respected() -> None:
    """Across a full stream the tracked rank stays ``<= rank_cap``."""
    n, t, cap = 50, 200, 12
    rng = np.random.default_rng(0)
    M = rng.standard_normal((n, t))  # full-rank input -> tries to grow past the cap
    tracker = StreamingBUG(n_features=n, rank_cap=cap)
    tracker.update_many(M)
    assert all(r <= cap for r in tracker.rank_history)
    assert tracker.rank <= cap
    assert max(tracker.rank_history) == cap  # the cap is actually reached


# --------------------------------------------------------------------------- #
# 2. Exactly-rank-r input is reconstructed ~exactly.
# --------------------------------------------------------------------------- #
def test_exact_rank_input_reconstructed_exactly() -> None:
    """An exactly-rank-``r`` stream is captured to machine precision."""
    n, t, r = 40, 120, 5
    M = _exact_rank_matrix(n, t, r, seed=3)
    # Cap strictly larger than the true rank: the tracker must still nail it.
    tracker = StreamingBUG(n_features=n, rank_cap=8)
    tracker.update_many(M)
    err = tracker.reconstruction_error(M)
    assert err < 1e-10, f"exact rank-{r} input not reconstructed: err={err:.2e}"
    # And it should not have inflated the rank beyond the true rank.
    assert tracker.rank <= r + 1


def test_exact_rank_input_with_tight_cap() -> None:
    """With ``rank_cap == r`` the exactly-rank-``r`` input is still exact."""
    n, t, r = 30, 90, 4
    M = _exact_rank_matrix(n, t, r, seed=5)
    tracker = StreamingBUG(n_features=n, rank_cap=r)
    tracker.update_many(M)
    assert tracker.reconstruction_error(M) < 1e-10


# --------------------------------------------------------------------------- #
# 3. Error decreases as the rank cap rises, and tracks the SVD oracle.
# --------------------------------------------------------------------------- #
def test_error_decreases_with_rank() -> None:
    """Higher rank cap -> non-increasing reconstruction error (low-rank + noise)."""
    n, t, r = 60, 300, 6
    base = _exact_rank_matrix(n, t, r, seed=7)
    rng = np.random.default_rng(8)
    M = base + 0.02 * rng.standard_normal((n, t))  # full-rank but low-rank-dominated

    caps = [2, 4, 8, 16, 32]
    errs = []
    for cap in caps:
        tracker = StreamingBUG(n_features=n, rank_cap=cap)
        tracker.update_many(M)
        errs.append(tracker.reconstruction_error(M))

    # Strictly non-increasing in rank (allow a tiny float slack).
    for lo, hi in zip(errs[1:], errs[:-1], strict=True):
        assert lo <= hi + 1e-9, f"error not monotone in rank: {errs}"


def test_competitive_with_svd_oracle() -> None:
    """Streaming-BUG error is within a small constant of the truncated-SVD oracle.

    The pilot criterion (PLAN §5) is ~1.5x of the oracle on real KV; on this
    clean synthetic low-rank+noise problem the tracker should sit essentially on
    the oracle, so we use a tight 1.2x bound here.
    """
    n, t, r = 60, 300, 6
    base = _exact_rank_matrix(n, t, r, seed=7)
    rng = np.random.default_rng(8)
    M = base + 0.02 * rng.standard_normal((n, t))

    for cap in (4, 8, 16):
        tracker = StreamingBUG(n_features=n, rank_cap=cap)
        tracker.update_many(M)
        err = tracker.reconstruction_error(M)
        oracle = _oracle_error(M, cap)
        assert err <= 1.2 * oracle + 1e-9, (
            f"rank {cap}: BUG err {err:.4e} not within 1.2x oracle {oracle:.4e}"
        )


# --------------------------------------------------------------------------- #
# 4. The adaptive theta tolerance trades rank for accuracy monotonically.
# --------------------------------------------------------------------------- #
def test_theta_monotonicity() -> None:
    """Smaller ``theta`` -> larger mean rank and smaller reconstruction error."""
    n, t, r = 50, 250, 8
    base = _exact_rank_matrix(n, t, r, seed=11)
    rng = np.random.default_rng(12)
    M = base + 0.01 * rng.standard_normal((n, t))

    thetas = [1.0, 0.3, 0.1]  # decreasing tolerance
    mean_ranks, errs = [], []
    for theta in thetas:
        tracker = StreamingBUG(n_features=n, theta=theta, rank_cap=n)
        tracker.update_many(M)
        mean_ranks.append(tracker.mean_rank)
        errs.append(tracker.reconstruction_error(M))

    # Tighter tolerance keeps more rank and lowers the error.
    assert mean_ranks[0] <= mean_ranks[1] <= mean_ranks[2]
    assert errs[0] >= errs[1] >= errs[2] - 1e-9


def test_theta_clamped_by_rank_cap() -> None:
    """When both are set, ``rank_cap`` bounds the adaptive ``theta`` choice."""
    n, t, cap = 40, 200, 5
    rng = np.random.default_rng(13)
    M = rng.standard_normal((n, t))  # full-rank -> theta alone would grow rank large
    tracker = StreamingBUG(n_features=n, theta=1e-6, rank_cap=cap)
    tracker.update_many(M)
    assert tracker.rank <= cap
    assert all(r <= cap for r in tracker.rank_history)


# --------------------------------------------------------------------------- #
# 5. Bookkeeping: mean_rank / rank_history / n_tokens_seen.
# --------------------------------------------------------------------------- #
def test_rank_bookkeeping() -> None:
    """``rank_history`` has one entry per token; ``mean_rank`` is their average."""
    n, t, cap = 30, 50, 6
    M = _exact_rank_matrix(n, t, 4, seed=15)
    tracker = StreamingBUG(n_features=n, rank_cap=cap)
    assert tracker.mean_rank == 0.0  # no tokens yet
    tracker.update_many(M)
    assert tracker.n_tokens_seen == t
    assert len(tracker.rank_history) == t
    assert tracker.mean_rank == pytest.approx(float(np.mean(tracker.rank_history)))


# --------------------------------------------------------------------------- #
# 6. Input validation and the no-new-direction branch.
# --------------------------------------------------------------------------- #
def test_constructor_validation() -> None:
    """The constructor rejects missing knobs and out-of-range values."""
    with pytest.raises(ValueError):
        StreamingBUG(n_features=10)  # neither rank_cap nor theta
    with pytest.raises(ValueError):
        StreamingBUG(n_features=10, rank_cap=0)
    with pytest.raises(ValueError):
        StreamingBUG(n_features=10, theta=-1.0)
    with pytest.raises(ValueError):
        StreamingBUG(n_features=0, rank_cap=2)


def test_update_shape_validation() -> None:
    """``update`` rejects a column with the wrong number of features."""
    tracker = StreamingBUG(n_features=8, rank_cap=3)
    with pytest.raises(ValueError):
        tracker.update(np.ones(7))
    # Column-vector and flat-vector forms are both accepted.
    tracker.update(np.ones((8, 1)))
    tracker.update(np.ones(8))
    assert tracker.n_tokens_seen == 2


def test_repeated_direction_does_not_grow_rank() -> None:
    """Streaming the same direction repeatedly keeps the rank at 1 and error ~0."""
    n = 20
    rng = np.random.default_rng(17)
    v = rng.standard_normal((n, 1))
    M = v @ rng.standard_normal((1, 100))  # every column is a scalar multiple of v
    tracker = StreamingBUG(n_features=n, rank_cap=5)
    tracker.update_many(M)
    assert tracker.rank == 1
    assert tracker.reconstruction_error(M) < 1e-10


def test_project_lies_in_subspace() -> None:
    """``project(M)`` columns lie in ``range(U)`` (residual orthogonal to U)."""
    n, t = 25, 80
    M = _exact_rank_matrix(n, t, 4, seed=19)
    tracker = StreamingBUG(n_features=n, rank_cap=6)
    tracker.update_many(M)
    proj = tracker.project(M)
    resid = M - proj
    # U^T resid == 0 (projection residual is orthogonal to the tracked basis).
    assert np.allclose(tracker.U.T @ resid, 0.0, atol=1e-10)
