"""Tests for the Oja's-rule online subspace tracker (:mod:`kvdlra.integrators.oja`).

:class:`kvdlra.integrators.oja.OjaTracker` processes a feature-by-token matrix one
column (token) at a time with the Oja update ``U <- orth(U + eta * k (k^T U))``
(OjaKV, arXiv:2509.21623) and tracks the left (feature) subspace ``U``; its
reconstruction model is the orthogonal projection ``U U^T M`` -- the *same* metric
as :class:`kvdlra.integrators.streaming.StreamingBUG`, so the two are directly
comparable. The deterministic checks here pin its core contract for the Week-2
Oja-vs-BUG comparison:

#. ``U`` stays orthonormal (``U^T U == I``) throughout and after a full stream;
#. an exactly-rank-``r`` input is tracked well -- the converged subspace's
   reconstruction error is small (near the truncated-SVD oracle of 0);
#. reconstruction error is (essentially) non-increasing as the rank rises and sits
   within a small constant of the truncated-SVD oracle on a low-rank+noise matrix;
#. the metric matches ``StreamingBUG`` exactly (same projection, same formula);
#. bookkeeping, the eta schedule, input validation, and the zero-token branch
   behave.

All inputs are fixed-seed; no model or KV dump is required (the real-KV
comparison lives in ``scripts/week2_oja_vs_bug.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from kvdlra.integrators.oja import OjaTracker
from kvdlra.integrators.streaming import StreamingBUG


def _orthonormal(n: int, r: int, seed: int) -> np.ndarray:
    """Orthonormal ``(n, r)`` factor from a fixed seed (QR of a Gaussian)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, r)))
    return np.asarray(q, dtype=np.float64)


def _exact_rank_matrix(n: int, t: int, r: int, seed: int, base: float = 2.0) -> np.ndarray:
    """A deterministic exactly-rank-``r`` matrix ``(n, t)`` with a decaying spectrum.

    The singular scale decays as ``base ** -j``. A gentler ``base`` (closer to 1)
    excites the trailing directions more strongly, so a stochastic online tracker
    converges into the *full* rank-``r`` subspace in fewer passes -- used by the
    Oja exactness tests, where the steep default ``base=2`` would leave the weakest
    direction (``2 ** -(r-1)``) starved of gradient signal for many epochs.
    """
    u = _orthonormal(n, r, seed)
    rng = np.random.default_rng(seed + 1)
    coeffs = rng.standard_normal((r, t))
    sigma = np.diag(base ** -np.arange(r).astype(np.float64))  # decaying singular scale
    return np.asarray(u @ sigma @ coeffs, dtype=np.float64)


def _oracle_error(M: np.ndarray, r: int) -> float:
    """Truncated-SVD (Eckart--Young) relative Frobenius error at rank ``r``."""
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    mr = (u[:, :r] * s[:r]) @ vt[:r]
    return float(np.linalg.norm(M - mr) / np.linalg.norm(M))


# --------------------------------------------------------------------------- #
# 1. U stays orthonormal throughout.
# --------------------------------------------------------------------------- #
def test_initial_basis_orthonormal() -> None:
    """The seed basis is orthonormal: ``U^T U == I_r``."""
    n, r = 64, 8
    tracker = OjaTracker(n_features=n, rank=r, seed=0)
    assert tracker.U.shape == (n, r)
    assert np.allclose(tracker.U.T @ tracker.U, np.eye(r), atol=1e-12)


def test_basis_stays_orthonormal_after_stream() -> None:
    """After streaming a full-rank matrix, ``U`` is still orthonormal."""
    n, t, r = 50, 200, 10
    rng = np.random.default_rng(1)
    M = rng.standard_normal((n, t))
    tracker = OjaTracker(n_features=n, rank=r, eta0=0.5, decay=1e-2)
    tracker.update_many(M)
    assert tracker.U.shape == (n, r)
    assert np.allclose(tracker.U.T @ tracker.U, np.eye(r), atol=1e-10)
    assert tracker.n_tokens_seen == t


# --------------------------------------------------------------------------- #
# 2. An exactly-rank-r input is tracked well.
# --------------------------------------------------------------------------- #
def test_exact_rank_input_tracked_well() -> None:
    """An exactly-rank-``r`` stream is captured to small error by an Oja rank-``r``.

    The data lives in an ``r``-dimensional subspace, so once Oja's basis rotates
    into it the projection residual must be tiny. We stream the (fixed) rank-``r``
    matrix a few times so the schedule converges, then check the reconstruction.
    """
    n, t, r = 40, 150, 4
    M = _exact_rank_matrix(n, t, r, seed=3, base=np.sqrt(2.0))
    tracker = OjaTracker(n_features=n, rank=r, eta0=1.0, decay=1e-3)
    # A handful of passes lets the converging schedule settle into the subspace.
    for _ in range(8):
        tracker.update_many(M)
    err = tracker.reconstruction_error(M)
    assert err < 1e-4, f"exact rank-{r} input not tracked: err={err:.2e}"


def test_exact_rank_input_beats_lower_rank() -> None:
    """Matched rank captures an exactly-rank-``r`` input far better than rank ``r-1``."""
    n, t, r = 36, 150, 5
    M = _exact_rank_matrix(n, t, r, seed=5, base=np.sqrt(2.0))

    full = OjaTracker(n_features=n, rank=r, eta0=1.0, decay=1e-3)
    deficient = OjaTracker(n_features=n, rank=r - 1, eta0=1.0, decay=1e-3)
    for _ in range(8):
        full.update_many(M)
        deficient.update_many(M)
    # Rank r nails the rank-r data; rank r-1 cannot (an irreducible tail remains).
    assert full.reconstruction_error(M) < deficient.reconstruction_error(M)
    assert full.reconstruction_error(M) < 1e-4


# --------------------------------------------------------------------------- #
# 3. Error decreases with rank and tracks the SVD oracle.
# --------------------------------------------------------------------------- #
def test_error_decreases_with_rank() -> None:
    """Higher rank -> (essentially) non-increasing reconstruction error."""
    n, t, r = 60, 300, 6
    base = _exact_rank_matrix(n, t, r, seed=7)
    rng = np.random.default_rng(8)
    M = base + 0.02 * rng.standard_normal((n, t))  # full-rank, low-rank-dominated

    ranks = [2, 4, 8, 16, 32]
    errs = []
    for rk in ranks:
        tracker = OjaTracker(n_features=n, rank=rk, eta0=1.0, decay=1e-3)
        for _ in range(5):
            tracker.update_many(M)
        errs.append(tracker.reconstruction_error(M))

    # Non-increasing in rank (small slack for the stochastic tracker).
    for lo, hi in zip(errs[1:], errs[:-1], strict=True):
        assert lo <= hi + 1e-2, f"error not monotone in rank: {errs}"


def test_competitive_with_svd_oracle() -> None:
    """Oja error is within a modest constant of the truncated-SVD oracle.

    Oja is an *online* tracker that never revisits a token, so it lags the oracle;
    on this clean low-rank+noise problem (with multiple passes to converge) it
    should still land within a small factor. We use a loose 2x bound -- this is a
    sanity floor, not the headline comparison (that is BUG vs Oja vs oracle on real
    KV in ``scripts/week2_oja_vs_bug.py``).
    """
    n, t, r = 60, 300, 6
    base = _exact_rank_matrix(n, t, r, seed=7)
    rng = np.random.default_rng(8)
    M = base + 0.02 * rng.standard_normal((n, t))

    for rk in (4, 8, 16):
        tracker = OjaTracker(n_features=n, rank=rk, eta0=1.0, decay=1e-3)
        for _ in range(8):
            tracker.update_many(M)
        err = tracker.reconstruction_error(M)
        oracle = _oracle_error(M, rk)
        assert err <= 2.0 * oracle + 1e-2, (
            f"rank {rk}: Oja err {err:.4e} not within 2x oracle {oracle:.4e}"
        )


# --------------------------------------------------------------------------- #
# 4. The metric matches StreamingBUG exactly.
# --------------------------------------------------------------------------- #
def test_metric_matches_streaming_bug() -> None:
    """Given the *same* ``U``, Oja and BUG report identical reconstruction error.

    Both define the error as ``||M - U U^T M||_F / ||M||_F``; this pins that the
    two implementations agree to machine precision for a shared basis, so the
    Week-2 comparison is apples-to-apples.
    """
    n, t, r = 32, 90, 5
    M = _exact_rank_matrix(n, t, r, seed=21)
    oja = OjaTracker(n_features=n, rank=r, eta0=0.7, decay=1e-2)
    oja.update_many(M)

    bug = StreamingBUG(n_features=n, rank_cap=r)
    bug.U = oja.U.copy()  # force the same basis into the BUG metric
    assert oja.reconstruction_error(M) == pytest.approx(bug.reconstruction_error(M), abs=1e-12)


def test_project_lies_in_subspace() -> None:
    """``project(M)`` columns lie in ``range(U)`` (residual orthogonal to U)."""
    n, t, r = 25, 80, 4
    M = _exact_rank_matrix(n, t, r, seed=19)
    tracker = OjaTracker(n_features=n, rank=r, eta0=1.0, decay=1e-3)
    tracker.update_many(M)
    resid = M - tracker.project(M)
    assert np.allclose(tracker.U.T @ resid, 0.0, atol=1e-10)


def test_reconstruction_error_zero_matrix() -> None:
    """An all-zero ``M`` yields exactly ``0.0`` (no division by zero)."""
    n, r = 16, 4
    tracker = OjaTracker(n_features=n, rank=r)
    assert tracker.reconstruction_error(np.zeros((n, 20))) == 0.0


# --------------------------------------------------------------------------- #
# 5. Schedule, bookkeeping, validation, and the zero-token branch.
# --------------------------------------------------------------------------- #
def test_eta_schedule_decays() -> None:
    """``eta_t = eta0 / (1 + decay * t)`` decreases with ``t`` (and is flat at decay=0)."""
    tracker = OjaTracker(n_features=8, rank=2, eta0=2.0, decay=0.5)
    assert tracker._eta(0) == pytest.approx(2.0)
    assert tracker._eta(1) == pytest.approx(2.0 / 1.5)
    assert tracker._eta(10) < tracker._eta(1)

    flat = OjaTracker(n_features=8, rank=2, eta0=2.0, decay=0.0)
    assert flat._eta(0) == flat._eta(100) == pytest.approx(2.0)


def test_token_counter() -> None:
    """``n_tokens_seen`` counts every processed column, zero tokens included."""
    n, r = 12, 3
    tracker = OjaTracker(n_features=n, rank=r)
    tracker.update(np.zeros(n))  # zero token: skipped update but still counted
    u_after_zero = tracker.U.copy()
    rng = np.random.default_rng(2)
    tracker.update(rng.standard_normal(n))
    assert tracker.n_tokens_seen == 2
    # The zero token must not have moved the basis.
    assert np.allclose(u_after_zero, OjaTracker(n_features=n, rank=r).U)


def test_normalize_makes_update_scale_invariant() -> None:
    """With ``normalize=True`` a token and its rescaling drive the basis identically."""
    n, r = 20, 4
    rng = np.random.default_rng(4)
    k = rng.standard_normal(n)

    a = OjaTracker(n_features=n, rank=r, eta0=1.0, decay=0.0, normalize=True)
    b = OjaTracker(n_features=n, rank=r, eta0=1.0, decay=0.0, normalize=True)
    a.update(k)
    b.update(1000.0 * k)  # same direction, very different magnitude
    assert np.allclose(a.U, b.U, atol=1e-10)


def test_accepts_flat_and_column_vectors() -> None:
    """``update`` accepts both ``(n,)`` and ``(n, 1)`` token shapes."""
    n, r = 8, 2
    tracker = OjaTracker(n_features=n, rank=r)
    rng = np.random.default_rng(6)
    tracker.update(rng.standard_normal(n))
    tracker.update(rng.standard_normal((n, 1)))
    assert tracker.n_tokens_seen == 2
    assert np.allclose(tracker.U.T @ tracker.U, np.eye(r), atol=1e-10)


def test_constructor_validation() -> None:
    """The constructor rejects out-of-range arguments."""
    with pytest.raises(ValueError):
        OjaTracker(n_features=0, rank=2)
    with pytest.raises(ValueError):
        OjaTracker(n_features=10, rank=0)
    with pytest.raises(ValueError):
        OjaTracker(n_features=4, rank=5)  # rank > n_features
    with pytest.raises(ValueError):
        OjaTracker(n_features=10, rank=2, eta0=0.0)
    with pytest.raises(ValueError):
        OjaTracker(n_features=10, rank=2, decay=-1.0)


def test_update_shape_validation() -> None:
    """``update``/``update_many`` reject the wrong number of features."""
    tracker = OjaTracker(n_features=8, rank=3)
    with pytest.raises(ValueError):
        tracker.update(np.ones(7))
    with pytest.raises(ValueError):
        tracker.update_many(np.ones((7, 10)))
