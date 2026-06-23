"""Oja's-rule online subspace tracker for a column-streamed matrix (OjaKV baseline).

This module implements :class:`OjaTracker`, the **online PCA / subspace-tracking
baseline** for the Week-2 pilot: it processes a feature-by-token matrix ``M``
(rows = features, columns = tokens; see ``docs/notes/conventions.md``) one column
(token) at a time and maintains an orthonormal basis ``U`` (``n_features x rank``)
of the tracked left (feature) subspace via **Oja's rule**. It is the direct
counterpart of :class:`kvdlra.integrators.streaming.StreamingBUG` (the augmented
rank-adaptive BUG tracker) and is compared against it -- and against the
truncated-SVD oracle -- at matched memory in ``scripts/week2_oja_vs_bug.py``.

Oja's rule (OjaKV, arXiv:2509.21623)
------------------------------------
For a unit-norm token ``k_t`` the per-step update is

    U <- orth( U + eta_t * k_t (k_t^T U) ),

i.e. each column of ``U`` is nudged toward ``k_t`` in proportion to its current
alignment ``k_t^T U`` with that column, then the basis is re-orthonormalized
(economic QR). This is the classic rank-``r`` generalization of Oja's (1982)
stochastic gradient ascent on the Rayleigh quotient ``tr(U^T C U)`` with
``C = E[k k^T]``: the gradient of ``1/2 tr(U^T k k^T U)`` w.r.t. ``U`` is exactly
``k (k^T U)``, and the ``orth(.)`` is the retraction back onto the Stiefel
manifold of orthonormal frames. Its fixed point is the top-``r`` eigenspace of
``C`` -- which (for centred, unit-scaled data) is the top-``r`` *left* singular
subspace of ``M`` -- so a well-tuned tracker converges toward the same subspace
that :func:`kvdlra.lowrank.truncated_svd_recon` finds in one shot. Unlike the
augmented BUG step, Oja **never revisits** a token and its subspace can only be as
good as the running gradient average, so it typically lags the BUG/SVD optimum.

Reconstruction model. The model for the data is the orthogonal projection of the
columns onto ``U``, ``U @ U.T @ M``; :meth:`reconstruction_error` reports its
relative Frobenius error ``||M - U U^T M||_F / ||M||_F`` -- **identical** to
:meth:`kvdlra.integrators.streaming.StreamingBUG.reconstruction_error`, so the two
trackers are compared on exactly the same metric at the same rank.

The eta schedule (the one knob that matters) -- DOCUMENTED CHOICE
----------------------------------------------------------------
Oja's rule is notoriously **step-size sensitive**: the raw gradient ``k (k^T U)``
scales with ``||k_t||^2``, so a fixed ``eta`` on raw KV vectors (whose norms vary
by orders of magnitude across tokens and grow with RoPE position) either diverges
(too large) or never moves (too small), and the "right" value depends on the
absolute data scale. We remove both problems with two coupled choices, applied
together by default:

1.  **Per-step data normalization** (``normalize=True``, default). Each token is
    L2-normalized to ``k_hat = k_t / ||k_t||`` *before* the update, so the
    gradient term ``k_hat (k_hat^T U)`` has spectral norm ``<= 1`` regardless of
    the token's magnitude. This makes a single ``eta0`` transfer across layers,
    documents and RoPE scales (the update becomes scale-free), and is the
    standard "normalized Hebbian / Oja" remedy. It changes *which* second moment
    is tracked -- the top subspace of ``sum_t k_hat k_hat^T`` (direction-only,
    each token weighted equally) rather than of ``M M^T`` (magnitude-weighted) --
    but on these KV streams the leading directions coincide closely, and this is
    the standard, robust way to run Oja; the choice is reported in the verdict.

2.  **Robbins--Monro decaying step** ``eta_t = eta0 / (1 + decay * t)`` (``t``
    0-indexed over tokens). A decreasing step that satisfies ``sum eta_t = inf``,
    ``sum eta_t^2 < inf`` is the classical condition for Oja's rule to converge
    a.s. to the principal subspace: large early steps explore, shrinking later
    steps average out the per-token gradient noise and let ``U`` settle. With
    ``decay = 0`` this reduces to a constant step (kept as an option for
    diagnostics, but it does not converge as tightly).

``eta0`` and ``decay`` are *tuned fairly* by the comparison script (it sweeps a
grid of ``eta0`` per rank and reports the best together with the schedule), so the
baseline is shown at its strongest. Defaults ``eta0 = 1.0``, ``decay = 1e-3`` are
a reasonable normalized-Oja operating point but are expected to be overridden by
the sweep.

Numerics. All linear algebra is float64. The basis is re-orthonormalized every
step by economic QR (with a sign fix so the frame is deterministic); per-token
cost is ``O(n r^2)`` (the QR), i.e. ``O(T n r^2)`` for the whole stream.

NOTE: This is library code -- it contains no I/O or ``print`` calls. The
companion test ``tests/test_oja.py`` checks orthonormality of ``U``, monotone
error decrease with rank, and near-exact tracking of an exactly-rank-``r`` input.

References
----------
E. Oja, "Simplified neuron model as a principal component analyzer,"
J. Math. Biol. 15 (1982) 267--273 (the original single-unit rule); the
subspace/multi-unit generalization is Oja 1989/1992.

OjaKV: online low-rank KV-cache compression with Oja's rule, arXiv:2509.21623 --
the ``U <- orth(U + eta * k (k^T U))`` online subspace update used here.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["OjaTracker"]

# A token whose Euclidean norm is below this is treated as zero (no direction to
# learn from) and skipped, avoiding a 0/0 in the per-step normalization.
_NORM_EPS = 1e-12


def _orthonormalize(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Economic-QR orthonormalization with a deterministic sign convention.

    Returns an orthonormal basis for ``range(a)`` (the ``Q`` factor of a thin QR),
    with each column's sign fixed so that its largest-magnitude entry is positive.
    The sign fix makes the frame reproducible (QR is only unique up to per-column
    signs) without changing the spanned subspace, which is all the reconstruction
    error depends on.
    """
    q, r = np.linalg.qr(a)
    # Fix column signs: make the pivot (diagonal of R) non-negative.
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return np.asarray(q * signs, dtype=np.float64)


class OjaTracker:
    """Oja's-rule online subspace tracker of a column-streamed matrix's range.

    Processes the columns (tokens) of a feature-by-token matrix one at a time with
    the Oja update ``U <- orth(U + eta_t * k_t (k_t^T U))`` (OjaKV,
    arXiv:2509.21623) and maintains an ``(n_features, rank)`` orthonormal left
    basis :attr:`U`. The reconstruction model for the data is the orthogonal
    projection ``U @ U.T @ M``; :meth:`reconstruction_error` reports its relative
    Frobenius error against a supplied ``M`` -- the **same metric** as
    :class:`kvdlra.integrators.streaming.StreamingBUG`, so the two trackers are
    directly comparable at the same ``rank``.

    Unlike the rank-adaptive BUG tracker, the rank here is **fixed** at ``rank``
    for the whole stream (Oja's rule maintains a frame of a chosen size); compare
    it against ``StreamingBUG(rank_cap=rank)`` for a matched-memory comparison.

    See the module docstring for the eta schedule and the per-step normalization
    choice (both documented there).

    Parameters
    ----------
    n_features:
        Number of rows (features) of the streamed matrix -- ``head_dim *
        num_kv_heads = 512`` for the Llama-3.2-1B K-cache (see
        ``docs/notes/conventions.md``).
    rank:
        Fixed rank ``r >= 1`` of the tracked subspace (columns of :attr:`U`).
    eta0:
        Base learning rate of the schedule ``eta_t = eta0 / (1 + decay * t)``
        (``> 0``). With per-step normalization on, ``eta0`` transfers across data
        scales; it is tuned by the comparison script.
    decay:
        Non-negative Robbins--Monro decay rate. ``decay = 0`` gives a constant
        step ``eta0``; ``decay > 0`` gives the converging decreasing schedule.
    normalize:
        If ``True`` (default), L2-normalize each token before the update so the
        gradient is scale-free (the documented robust choice). If ``False``, use
        raw tokens (Oja becomes magnitude-weighted and eta-fragile).
    seed:
        Seed for the deterministic random orthonormal initial basis.

    Attributes
    ----------
    U:
        Current left orthonormal basis, shape ``(n_features, rank)``.
    n_tokens_seen:
        Number of tokens (columns) processed so far.
    """

    U: npt.NDArray[np.float64]

    def __init__(
        self,
        n_features: int,
        rank: int,
        eta0: float = 1.0,
        decay: float = 1e-3,
        normalize: bool = True,
        seed: int = 0,
    ) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if rank > n_features:
            raise ValueError(f"rank must be <= n_features ({n_features}), got {rank}")
        if eta0 <= 0.0:
            raise ValueError(f"eta0 must be > 0, got {eta0}")
        if decay < 0.0:
            raise ValueError(f"decay must be >= 0, got {decay}")

        self.n_features = int(n_features)
        self.rank = int(rank)
        self.eta0 = float(eta0)
        self.decay = float(decay)
        self.normalize = bool(normalize)
        self.n_tokens_seen = 0

        # Deterministic orthonormal seed basis (QR of a Gaussian). Oja's rule
        # rotates this toward the data's principal subspace as tokens stream in.
        rng = np.random.default_rng(seed)
        q = _orthonormalize(rng.standard_normal((self.n_features, self.rank)))
        self.U = np.asarray(q, dtype=np.float64)

    def _eta(self, t: int) -> float:
        """Step size at token index ``t`` (0-indexed): ``eta0 / (1 + decay * t)``."""
        return self.eta0 / (1.0 + self.decay * t)

    def update(self, k_t: npt.NDArray[np.float64]) -> None:
        """Process one token (one column ``k_t``) with a single Oja step.

        Computes ``U <- orth(U + eta_t * k_hat (k_hat^T U))`` where ``k_hat`` is
        ``k_t`` (optionally L2-normalized, see :attr:`normalize`) and ``eta_t``
        follows the decaying schedule. A (numerically) zero token is skipped (it
        carries no direction), but still advances the token counter and schedule.

        Parameters
        ----------
        k_t:
            The incoming token's feature vector, shape ``(n_features,)`` or
            ``(n_features, 1)``.

        Raises
        ------
        ValueError
            If ``k_t`` does not have ``n_features`` entries.
        """
        c = np.asarray(k_t, dtype=np.float64).reshape(-1, 1)
        if c.shape[0] != self.n_features:
            raise ValueError(
                f"k_t must have {self.n_features} features, got shape {np.asarray(k_t).shape}"
            )

        norm = float(np.linalg.norm(c))
        if norm <= _NORM_EPS:
            # All-zero (or numerically zero) token: no direction to learn from.
            self.n_tokens_seen += 1
            return
        if self.normalize:
            c = c / norm

        eta = self._eta(self.n_tokens_seen)
        # Oja step: nudge each basis column toward c by its alignment c^T U.
        coords = c.T @ self.U  # (1, r) = k_hat^T U
        u_new = self.U + eta * (c @ coords)  # (n, r)
        self.U = _orthonormalize(u_new)
        self.n_tokens_seen += 1

    def update_many(self, M: npt.NDArray[np.float64]) -> None:
        """Stream every column of ``M`` through :meth:`update`, left to right.

        Parameters
        ----------
        M:
            Feature-by-token matrix, shape ``(n_features, T)``. Its columns are
            processed in order, one :meth:`update` per column.

        Raises
        ------
        ValueError
            If ``M`` does not have ``n_features`` rows.
        """
        arr = np.asarray(M, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != self.n_features:
            raise ValueError(f"M must have shape ({self.n_features}, T), got {arr.shape}")
        for t in range(arr.shape[1]):
            self.update(arr[:, t])

    def project(self, M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Orthogonal projection ``U @ U.T @ M`` of ``M`` onto the tracked subspace.

        This is the tracker's reconstruction model for the data: the best
        approximation of ``M`` whose columns lie in ``range(U)``.

        Parameters
        ----------
        M:
            Feature-by-token matrix, shape ``(n_features, T)``.

        Returns
        -------
        npt.NDArray[np.float64]
            The projected matrix ``U @ (U.T @ M)``, shape ``(n_features, T)``.
        """
        arr = np.asarray(M, dtype=np.float64)
        return np.asarray(self.U @ (self.U.T @ arr), dtype=np.float64)

    def reconstruction_error(self, M: npt.NDArray[np.float64]) -> float:
        """Relative Frobenius error ``||M - U U^T M||_F / ||M||_F``.

        The reconstruction is the orthogonal projection of ``M`` onto the tracked
        left subspace (:meth:`project`). This is **the same metric** as
        :meth:`kvdlra.integrators.streaming.StreamingBUG.reconstruction_error`, so
        Oja and BUG are compared on identical footing at the same rank against the
        truncated-SVD oracle.

        Parameters
        ----------
        M:
            Feature-by-token matrix, shape ``(n_features, T)`` (typically the same
            matrix that was streamed, evaluated after streaming).

        Returns
        -------
        float
            Relative Frobenius reconstruction error in ``[0, 1]``. Returns ``0.0``
            for an all-zero ``M``.
        """
        arr = np.asarray(M, dtype=np.float64)
        denom = float(np.linalg.norm(arr))
        if denom == 0.0:
            return 0.0
        resid = arr - self.project(arr)
        return float(np.linalg.norm(resid) / denom)
