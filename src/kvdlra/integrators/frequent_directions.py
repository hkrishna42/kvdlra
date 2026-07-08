"""Frequent Directions -- a deterministic streaming low-rank sketch (Liberty 2013).

A non-DLRA alternative to :class:`kvdlra.integrators.streaming.StreamingBUG` for
tracking the dominant **left (feature) subspace** of a streaming feature-by-token
matrix ``M`` (rows = features ``n``, columns = tokens). Where BUG time-steps the
subspace on the rank-``r`` manifold (a dynamical-low-rank integrator), Frequent
Directions (FD) maintains a small **sketch** ``B`` (``ell x n``) such that
``B^T B`` approximates the feature second moment ``M M^T``, by periodically
SVD-ing the sketch and **deterministically shrinking** every singular value by
the ``ell``-th one (``sigma_i^2 <- max(sigma_i^2 - sigma_ell^2, 0)``). The tracked
subspace is the top-``rank`` right singular vectors of the sketch.

Why it is the natural non-DLRA competitor for the KV use case
-------------------------------------------------------------
Our problem is not integrating an ODE -- it is *sketching a streaming data
matrix* -- and FD is the canonical deterministic algorithm for exactly that,
with a provable covariance-error bound (Ghashami--Liberty--Phillips--Woodruff
2016): for a sketch of ``ell`` rows and any ``k < ell``,

    ||M M^T - B^T B||_2 <= ||M - M_k||_F^2 / (ell - k),

so the projection error of the top-``rank`` sketch directions is bounded relative
to the truncated-SVD oracle. Its shrinkage step is a principled, deterministic
cousin of BUG's truncation -- the ablation this class enables (``docs/
week7-dominance.md``): swap the subspace tracker, keep everything else, and ask
whether BUG's DLRA machinery leaves any reconstruction accuracy on the table
versus a pure streaming sketch.

API parity with :class:`StreamingBUG` / :class:`OjaTracker`: ``update_many(M)``,
``project(M)``, ``reconstruction_error(M)`` -- identical relative-Frobenius
metric, so the three are compared on identical footing at the same rank against
the truncated-SVD oracle (:func:`kvdlra.lowrank.truncated_svd_recon`).

NOTE: numpy fp64 library code -- no ``print``/I/O.

Reference: E. Liberty, "Simple and Deterministic Matrix Sketching," KDD 2013;
M. Ghashami, E. Liberty, J. M. Phillips, D. P. Woodruff, "Frequent Directions:
Simple and Deterministic Matrix Sketching," SIAM J. Comput. 45(5) 2016.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["FrequentDirections"]


class FrequentDirections:
    """Deterministic streaming sketch of the left subspace of ``M`` (``n x T``).

    Parameters
    ----------
    n_features:
        Feature dimension ``n`` (rows of ``M``).
    rank:
        Target subspace rank ``r`` (``project`` uses the top-``r`` sketch
        directions). Must be ``>= 1`` and ``<= n_features``.
    ell:
        Sketch size (number of rows kept); ``None`` => ``2 * rank`` (the standard
        choice giving the ``ell - k`` denominator in the error bound). The sketch
        stores ``ell x n`` floats -- so at ``ell = 2r`` its memory is comparable
        to BUG's ``n x r`` basis plus core, matched at the subspace level.
    """

    def __init__(self, n_features: int, rank: int, ell: int | None = None) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if rank > n_features:
            raise ValueError(f"rank must be <= n_features ({n_features}), got {rank}")
        self.n_features = int(n_features)
        self.rank = int(rank)
        self.ell = int(ell) if ell is not None else 2 * self.rank
        if self.ell <= self.rank:
            raise ValueError(f"ell ({self.ell}) must be > rank ({self.rank})")
        self.ell = min(self.ell, self.n_features)
        self.n_tokens_seen = 0
        # Sketch rows accumulate here; shrunk back to <= ell-1 rows when full.
        self._sketch = np.zeros((0, self.n_features), dtype=np.float64)

    def _shrink(self) -> None:
        """SVD the sketch and subtract the ``ell``-th squared singular value from
        all (the Frequent-Directions shrinkage), leaving ``<= ell - 1`` rows."""
        _u, s, vt = np.linalg.svd(self._sketch, full_matrices=False)
        keep = self.ell - 1
        if s.shape[0] <= keep:
            return  # nothing to shrink yet
        delta = float(s[keep] ** 2)
        s2 = np.maximum(s[:keep] ** 2 - delta, 0.0)
        self._sketch = np.ascontiguousarray((np.sqrt(s2)[:, None]) * vt[:keep])

    def update_many(self, M: npt.NDArray[np.float64]) -> None:
        """Insert every column of ``M`` (``n x T``) as a sketch row, shrinking
        whenever the sketch reaches ``ell`` rows (the streaming FD schedule)."""
        arr = np.asarray(M, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != self.n_features:
            raise ValueError(f"M must have shape ({self.n_features}, T), got {arr.shape}")
        rows = arr.T  # (T, n): tokens as rows, so B^T B tracks M M^T
        for start in range(0, rows.shape[0], self.ell):
            block = rows[start : start + self.ell]
            self._sketch = np.vstack([self._sketch, block])
            self.n_tokens_seen += int(block.shape[0])
            if self._sketch.shape[0] >= self.ell:
                self._shrink()

    def subspace(self) -> npt.NDArray[np.float64]:
        """Current tracked basis ``U`` (``n x r'``, ``r' <= rank``): the top-``rank``
        right singular vectors of the sketch (the dominant feature directions)."""
        if self._sketch.shape[0] == 0:
            return np.zeros((self.n_features, 1), dtype=np.float64)
        _u, _s, vt = np.linalg.svd(self._sketch, full_matrices=False)
        keep = min(self.rank, vt.shape[0])
        return np.ascontiguousarray(vt[:keep].T)

    def project(self, M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Orthogonal projection ``U @ (U^T @ M)`` onto the tracked subspace
        (the same reconstruction model as BUG/Oja)."""
        u = self.subspace()
        arr = np.asarray(M, dtype=np.float64)
        return np.asarray(u @ (u.T @ arr), dtype=np.float64)

    def reconstruction_error(self, M: npt.NDArray[np.float64]) -> float:
        """Relative Frobenius error ``||M - U U^T M||_F / ||M||_F`` -- **the same
        metric** as :meth:`StreamingBUG.reconstruction_error`, so FD, BUG and Oja
        are compared on identical footing against the truncated-SVD oracle."""
        arr = np.asarray(M, dtype=np.float64)
        denom = float(np.linalg.norm(arr))
        if denom == 0.0:
            return 0.0
        return float(np.linalg.norm(arr - self.project(arr)) / denom)
