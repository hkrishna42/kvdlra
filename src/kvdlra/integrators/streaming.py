"""Streaming rank-adaptive BUG subspace tracker for a column-streamed matrix.

This module implements :class:`StreamingBUG`, a *real streaming integrator* that
processes a feature-by-token matrix ``M`` (rows = features, columns = tokens; see
``docs/notes/conventions.md``) **one column (token) at a time** with
basis-update & Galerkin (BUG) dynamics, and tracks the left (feature) subspace
``U`` of the accumulated data. It is the "rank-adaptive BUG" curve of the Week-2
go/no-go pilot (``docs/PLAN.md`` §5), and replaces the crude single-sweep
placeholder ``bug_recon`` in ``scripts/sigma_decay.py`` with a genuine
token-at-a-time tracker.

Formulation (subspace tracking, PLAN option (a))
------------------------------------------------
After ``t`` tokens the accumulated matrix is ``M_t = M[:, :t]``. We maintain a
rank-``r`` factorization of its *range* via

    * ``U`` -- an ``(n, r)`` orthonormal basis of the tracked left (feature)
      subspace, and
    * ``B`` -- an ``(r, k)`` square-root core with ``B @ B.T = U.T @ M_t @ M_t.T
      @ U`` (the accumulated second moment expressed in ``U``-coordinates).

The reconstruction is the orthogonal projection of the data onto ``U``: the model
for the data is ``U @ U.T @ M_t``, and the reconstruction error reported by
:meth:`StreamingBUG.reconstruction_error` is ``||M - U U^T M||_F / ||M||_F``. The
oracle for *this* error at rank ``r`` is the top-``r`` left singular subspace of
``M`` (Eckart--Young), so a tracker whose ``U`` converges to that subspace
attains the truncated-SVD lower bound.

Why this is a BUG step (not just incremental PCA)
-------------------------------------------------
The per-token update below is exactly the *augmented* (rank-adaptive) BUG step of
Ceruti--Kusch--Lubich 2022 (arXiv:2104.05247 §2, Eqs. (3)--(6); implemented in
:mod:`kvdlra.integrators.bug_adaptive`) specialized to the streaming relaxation
field ``F(Y) = Y_target - Y`` with the linear K-, L- and S-substep ODEs
integrated **exactly** (their Galerkin steady state) rather than by a single
explicit RK2 step. Concretely, the augmented BUG construction --

    1. **range-augment** the left basis with the new column's residual direction
       (``[U | U_new]`` ↔ the K-step's ``[K1 | U]`` augmentation),
    2. **Galerkin-project** the grown target onto the augmented basis (the
       forward S-step's steady state), and
    3. **truncate** by the Frobenius-tail criterion
       ``(sum_{j>r1} sigma_j^2)^{1/2} <= theta`` (Eq. of §2)

-- collapses, for a *rank-one* data increment, to: append the incoming column's
``U``-coordinates as a new column of the square-root core ``B``, re-SVD the small
augmented factor, and keep the leading directions. Solving the linear substeps
exactly (a closed form for this field) is the natural choice for the data-fitting
setting and is what makes the tracker reach the SVD oracle rather than lag it by
the O(h) error of one explicit step. The Frobenius-tail truncation reuses
:func:`kvdlra.integrators.bug_adaptive.truncation_rank` (single source of truth).

Numerically the update is kept in **square-root form** (we factor ``B``, never
form ``B @ B.T``), per PLAN §8 pitfall #4 -- squaring the data would halve the
significant digits of the small singular values. All linear algebra is float64.

Rank control
------------
Two knobs (either or both):

    * ``rank_cap`` -- a hard upper bound ``r <= rank_cap`` on the tracked rank.
    * ``theta`` -- an adaptive Frobenius-tail tolerance: each step keeps the
      smallest rank whose discarded singular-value tail is ``<= theta`` (then
      clamped to ``rank_cap`` if both are set). Larger ``theta`` -> smaller rank.

Per-token cost is ``O(n r + r^3)`` (one rank-one basis augmentation plus a small
``(r+1) x (k+1)`` SVD); the whole stream is ``O(T (n r + r^3))``.

NOTE: This is library code -- it contains no I/O or ``print`` calls. The
companion test ``tests/test_streaming.py`` checks the rank cap, error monotonicity
in rank, and exact reconstruction of an exactly-rank-``r`` input.

References
----------
G. Ceruti, J. Kusch and C. Lubich, "A rank-adaptive robust integrator for
dynamical low-rank approximation," BIT Numer. Math. 62 (2022) 1149--1174,
arXiv:2104.05247, §2, Eqs. (3)--(6).

G. Ceruti and C. Lubich, "An unconventional robust integrator for dynamical
low-rank approximation," BIT Numer. Math. 62 (2022) 23--44, arXiv:2010.02022,
§3.1 (the underlying fixed-rank BUG step).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from kvdlra.integrators.bug_adaptive import truncation_rank

__all__ = ["StreamingBUG"]

# A residual whose Euclidean norm is below this (relative to the running scale)
# adds no genuinely new direction, so the basis is not augmented on that token.
_RESID_EPS = 1e-12


class StreamingBUG:
    """Streaming rank-adaptive BUG tracker of a column-streamed matrix's range.

    Processes the columns (tokens) of a feature-by-token matrix one at a time via
    the augmented BUG step (Ceruti--Kusch--Lubich 2022, arXiv:2104.05247 §2)
    specialized to the streaming relaxation field with exact substep solves; see
    the module docstring for the derivation. Maintains an ``(n, r)`` orthonormal
    left basis :attr:`U` and an ``(r, k)`` square-root core :attr:`B` such that
    ``B @ B.T`` is the accumulated second moment ``U.T @ M_t @ M_t.T @ U`` after
    ``t`` tokens.

    The tracked rank ``r`` is bounded by ``rank_cap`` and/or chosen adaptively by
    the Frobenius-tail tolerance ``theta``. The reconstruction model for the data
    is the orthogonal projection ``U @ U.T @ M``; :meth:`reconstruction_error`
    reports its relative Frobenius error against a supplied ``M``.

    Parameters
    ----------
    n_features:
        Number of rows (features) of the streamed matrix -- ``head_dim *
        num_kv_heads = 512`` for the Llama-3.2-1B K-cache (see
        ``docs/notes/conventions.md``).
    rank_cap:
        Hard upper bound on the tracked rank ``r`` (``>= 1``). Required unless
        ``theta`` is given; if both are given, the per-step rank is the adaptive
        choice clamped to ``rank_cap``.
    theta:
        Optional non-negative Frobenius-tail truncation tolerance. When set, each
        step keeps the smallest rank whose discarded singular-value tail is
        ``<= theta`` (see :func:`kvdlra.integrators.bug_adaptive.truncation_rank`).
    seed:
        Seed for the deterministic random initial basis (a tiny detail: the basis
        is overwritten by data within the first ``rank_cap`` tokens, but a fixed
        seed makes runs bit-reproducible).

    Attributes
    ----------
    U:
        Current left orthonormal basis, shape ``(n_features, r)``.
    B:
        Current square-root core, shape ``(r, k)`` with ``k <= r``.
    n_tokens_seen:
        Number of tokens (columns) processed so far.
    """

    U: npt.NDArray[np.float64]
    B: npt.NDArray[np.float64]

    def __init__(
        self,
        n_features: int,
        rank_cap: int | None = None,
        theta: float | None = None,
        seed: int = 0,
    ) -> None:
        if rank_cap is None and theta is None:
            raise ValueError("provide at least one of rank_cap or theta")
        if rank_cap is not None and rank_cap < 1:
            raise ValueError(f"rank_cap must be >= 1, got {rank_cap}")
        if theta is not None and theta < 0.0:
            raise ValueError(f"theta must be >= 0, got {theta}")
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")

        self.n_features = int(n_features)
        self.rank_cap = int(rank_cap) if rank_cap is not None else None
        self.theta = float(theta) if theta is not None else None
        self.n_tokens_seen = 0
        self._rank_history: list[int] = []

        # Deterministic orthonormal seed basis. We start at rank 1 (a single
        # direction); the augmented step grows the rank as new directions arrive,
        # up to the cap / tolerance. Starting small keeps the early-token ranks
        # honest (no phantom directions are reported before data fills them).
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.standard_normal((self.n_features, 1)))
        self.U = np.asarray(q, dtype=np.float64)
        self.B = np.zeros((1, 0), dtype=np.float64)

    @property
    def rank(self) -> int:
        """Current tracked rank ``r`` (number of columns of :attr:`U`)."""
        return int(self.U.shape[1])

    @property
    def mean_rank(self) -> float:
        """Mean tracked rank over all tokens processed so far (``0.0`` if none)."""
        if not self._rank_history:
            return 0.0
        return float(np.mean(self._rank_history))

    @property
    def rank_history(self) -> list[int]:
        """Per-token tracked rank, one entry per :meth:`update` call (a copy)."""
        return list(self._rank_history)

    def _max_rank_this_step(self, n_singular: int) -> int:
        """Per-step rank cap from ``rank_cap`` and the available singular count."""
        cap = n_singular
        if self.rank_cap is not None:
            cap = min(cap, self.rank_cap)
        return max(1, cap)

    def update(self, k_t: npt.NDArray[np.float64]) -> None:
        """Process one token (one column ``k_t``) with an augmented BUG step.

        Performs the augmented (rank-adaptive) BUG update specialized to the
        rank-one data increment ``k_t`` (see the module docstring): split ``k_t``
        into its component in the current subspace and an out-of-subspace
        residual, range-augment ``U`` with the residual direction, append the
        column's coordinates to the square-root core ``B``, re-SVD the augmented
        factor, and keep the leading directions (bounded by ``rank_cap`` and/or
        chosen by the Frobenius-tail tolerance ``theta``).

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

        # Coordinates of the new column in the current basis, and its residual.
        a = self.U.T @ c  # (r, 1)
        resid = c - self.U @ a  # (n, 1)
        resid_norm = float(np.linalg.norm(resid))

        if resid_norm > _RESID_EPS:
            # The column has a genuinely new direction: augment the range (the
            # K-step's [K1 | U] augmentation, with U_new = the residual unit
            # vector), and grow the square-root core by the new row + column.
            p = resid / resid_norm
            u_aug = np.concatenate([self.U, p], axis=1)  # (n, r+1)
            b_grown = np.vstack([self.B, np.zeros((1, self.B.shape[1]))])  # (r+1, k)
            new_col = np.concatenate([a, np.array([[resid_norm]])], axis=0)  # (r+1, 1)
            b_fac = np.concatenate([b_grown, new_col], axis=1)  # (r+1, k+1)
        else:
            # No new direction; the column lies (numerically) in range(U). Just
            # append its in-basis coordinates as a new core column.
            u_aug = self.U
            b_fac = np.concatenate([self.B, a], axis=1)  # (r, k+1)

        # Galerkin + Frobenius-tail truncation == SVD of the small square-root
        # core (its left singular vectors rotate u_aug; its singular values are
        # those of the accumulated model in u_aug-coordinates).
        u_loc, sigma, _ = np.linalg.svd(b_fac, full_matrices=False)

        keep = self._max_rank_this_step(int(sigma.shape[0]))
        if self.theta is not None:
            keep_theta = truncation_rank(sigma, self.theta)
            keep = min(keep, keep_theta)
            keep = max(1, keep)

        self.U = u_aug @ u_loc[:, :keep]
        self.B = np.diag(sigma[:keep])
        self.n_tokens_seen += 1
        self._rank_history.append(keep)

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
        left subspace (:meth:`project`). Compared at rank ``r`` against the
        truncated-SVD oracle (the top-``r`` left singular subspace), this is the
        quantity the Week-2 pilot (PLAN §5) requires to be competitive.

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
