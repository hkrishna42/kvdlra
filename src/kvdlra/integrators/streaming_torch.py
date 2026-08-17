"""Blocked (chunked) streaming BUG subspace tracker in PyTorch.

A GPU-capable, ``torch`` reimplementation of the augmented rank-adaptive BUG
subspace tracker in :class:`kvdlra.integrators.streaming.StreamingBUG`, processing
the feature-by-token matrix ``M`` (rows = features, columns = tokens;
``docs/notes/conventions.md``) **a block of columns at a time** instead of one
column at a time.

Why blocked (the motivation)
----------------------------
The numpy per-token tracker does one small SVD per token in a Python loop -- fine
for the Week-2 *proof* that streaming BUG tracks the SVD oracle, but far too slow
for Week-3/4 perplexity sweeps and 8B (the loop is CPU-bound; a GPU does not help
it). Processing ``block_size`` columns per augmented-BUG step turns ``T`` Python
iterations into ``ceil(T / block_size)`` and runs the QR/SVD/matmuls as batched
``torch`` ops on the tensor's own device (GPU). It is the **same** augmented BUG
integrator (Ceruti--Kusch--Lubich 2022, arXiv:2104.05247 §2) with a rank-``b``
data increment per step rather than rank-1; the block size is a speed/fidelity
knob:

    * ``block_size = 1``   -> the per-token streaming tracker (matches numpy).
    * ``block_size = T``   -> a single augmented step == the truncated-SVD oracle.
    * intermediate         -> a coarse streaming integration; Week-2 showed BUG
      and the oracle agree to ~1-3% on real KV, so all block sizes land in that
      band. :func:`blocked_bug_subspace` is validated for reconstruction-error
      parity against :class:`StreamingBUG` in ``tests/test_streaming_torch.py``.

The blocked augmented-BUG step (for a block ``C`` of ``b`` columns)
------------------------------------------------------------------
Given the current orthonormal basis ``U`` (``n x r``) and square-root core ``B``
(``r x r`` diagonal, ``B @ B.T`` == the accumulated second moment in
``U``-coordinates):

    1. coordinates ``A = U^T C`` (``r x b``) and out-of-basis residual
       ``R_perp = C - U A`` (``n x b``);
    2. **range-augment**: ``Q, R = qr(R_perp)`` and ``U_aug = [U | Q]``
       (the block analogue of the K-step's ``[U | U_new]``);
    3. **Galerkin core**: ``B_aug = [[B, A], [0, R]]`` (grow the square-root
       core by the block's coordinates + residual triangle);
    4. **truncate**: SVD ``B_aug = u_loc @ diag(sigma) @ *``; keep the leading
       ``keep`` directions (``rank_cap`` and/or the Frobenius-tail ``theta``);
       ``U <- U_aug @ u_loc[:, :keep]``, ``B <- diag(sigma[:keep])``.

Mixed precision (``docs/PLAN.md`` §8 pitfall #4): the QR/SVD/matmuls run in
``compute_dtype`` (default ``float32``) regardless of the storage dtype of ``M``;
bf16 storage is safe, bf16 *core* math is not.

NOTE: library code -- no ``print``/I/O here.

References
----------
G. Ceruti, J. Kusch and C. Lubich, "A rank-adaptive robust integrator for
dynamical low-rank approximation," BIT Numer. Math. 62 (2022) 1149--1174,
arXiv:2104.05247, §2. See also :mod:`kvdlra.integrators.streaming` (the numpy
per-token tracker this mirrors) and :mod:`kvdlra.integrators.bug_torch` (the
fp32-core convention).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["augmented_bug_step", "blocked_bug_project", "blocked_bug_subspace"]


def _truncation_rank(sigma: Tensor, theta: float) -> int:
    """Smallest ``k`` whose discarded singular tail ``(sum_{j>k} sigma_j^2)^{1/2}``
    is ``<= theta`` (torch mirror of
    :func:`kvdlra.integrators.bug_adaptive.truncation_rank`)."""
    # sigma is sorted descending (torch.linalg.svd convention).
    tail_sq = torch.flip(torch.cumsum(torch.flip(sigma**2, (0,)), 0), (0,))
    # tail_sq[k] == sum_{j>=k} sigma_j^2; we want the smallest k with the tail
    # *beyond* k (i.e. tail_sq[k]) <= theta^2.
    ok = tail_sq <= theta**2
    idx = torch.nonzero(ok, as_tuple=False)
    if idx.numel() == 0:
        return int(sigma.shape[0])
    return max(1, int(idx[0].item()))


def augmented_bug_step(
    u: Tensor | None,
    b_core: Tensor | None,
    block: Tensor,
    rank_cap: int,
    *,
    theta: float | None = None,
    min_sv_frac: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """One augmented rank-adaptive BUG step on a block of ``b`` new columns.

    The loop body of :func:`blocked_bug_subspace`, exposed as a *stateful* single
    step so callers that receive columns over time (the decode-time streaming
    cache, :mod:`kvdlra.cache`) can advance the tracked pair ``(U, B)`` one data
    increment at a time with exactly the validated integrator math.

    Parameters
    ----------
    u:
        Current orthonormal basis ``(n, r)``, or ``None`` for the seeding step
        (the first block, whose reduced QR initializes the basis).
    b_core:
        Current square-root core ``(r, r)``; must be ``None`` iff ``u`` is.
    block:
        New columns ``(n, b)`` in the dtype/device the step should run in
        (callers pass ``compute_dtype``; PLAN §8 pitfall #4).
    rank_cap:
        Hard upper bound on the tracked rank (``>= 1``).
    theta:
        Optional Frobenius-tail truncation tolerance (see :func:`_truncation_rank`).
    min_sv_frac:
        Optional relative singular-value floor in ``[0, 1)``: after truncation,
        additionally drop directions whose singular value is ``<= min_sv_frac``
        times the leading one. ``0.0`` (default) is a no-op, bit-for-bit the
        archived path. Caps the tracked rank at the block's numerical rank so a
        rank-deficient stream cannot pad the basis to ``rank_cap`` with near-null
        tail directions (the Week-17 high-rank stability fix, docs/week17).

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(u_new, b_new, rot)`` -- the updated basis ``(n, r')`` and core
        ``(r', r')``, plus ``rot = u_new^T u_old`` of shape ``(r', r)`` (``(r', 0)``
        on the seeding step). Because ``u_new = [u | q] @ u_loc[:, :r']`` with
        ``q ⟂ u``, this is exactly ``u_loc[:r, :r']^T`` -- the map that re-expresses
        coordinates held in the old basis in the new one (used by the streaming
        cache to carry per-token coordinates across basis updates).

    Notes
    -----
    The number of *admitted* new directions is clamped to ``n - r`` so the
    augmented basis ``[u | q]`` always fits in ``R^n``. This fixes the latent
    degeneracy of the original blocked sweep when ``rank_cap + block_size >
    n_features`` (the ablation follow-up in ``docs/week5.md``): the residual has
    rank at most ``n - r`` mathematically, so the discarded QR columns are
    numerically null and the clamp is exact up to roundoff.

    Additionally (Week-7 fix), the residual is **re-orthogonalized** against
    ``u`` once ("twice is enough", Parlett/Kahan) and rank-revealed by SVD:
    only directions with singular value above ``100 * eps(dtype) * ||block||_F``
    are admitted. Without this, a *numerically null* residual (the incoming
    block already lies in the tracked subspace -- e.g. data of effective
    dimension < ``rank_cap``) makes the plain QR return junk directions whose
    orthogonality against ``u`` is destroyed by cancellation, silently breaking
    the basis (found by the Week-7 full-rank parity tests on a tiny model; the
    pathology never fired at r=128 on real KV, so the *archived* Week-3..6
    results stand as-is). Note the change applies on every step, so reruns of
    older configs are **fp-equivalent, not bit-identical** (QR -> SVD residual
    factorization + the coordinate refinement term) -- compare methods within
    one run, never fresh curves against archived JSON at bit level.
    """
    if (u is None) != (b_core is None):
        raise ValueError("u and b_core must be provided together (or both None)")
    if rank_cap < 1:
        raise ValueError(f"rank_cap must be >= 1, got {rank_cap}")

    n = block.shape[0]
    if u is None:
        # Seeding step: the basis is the reduced QR of the first block.
        q, r = torch.linalg.qr(block, mode="reduced")
        u_aug = q
        b_fac = r
        r_old = 0
    else:
        assert b_core is not None  # by the pairing check above
        r_old = u.shape[1]
        a = u.mT @ block  # (r, b)
        r_perp = block - u @ a  # (n, b)
        # Re-orthogonalize once: removes the O(eps * ||block||) component of
        # r_perp along span(u) that cancellation leaves behind (and refines the
        # coordinates a accordingly), so admitted directions are orthogonal to
        # u at machine level even when the true residual is tiny.
        a2 = u.mT @ r_perp
        r_perp = r_perp - u @ a2
        a = a + a2
        # Rank-revealing factorization of the residual: admit only directions
        # that carry genuine new energy (and at most n - r_old of them).
        q, s, vh = torch.linalg.svd(r_perp, full_matrices=False)
        tol = 100.0 * torch.finfo(block.dtype).eps * torch.linalg.norm(block)
        m = int((s > tol).sum().item())
        m = min(m, max(0, n - r_old))
        q = q[:, :m]
        u_aug = torch.cat([u, q], dim=1)  # (n, r+m)
        top = torch.cat([b_core, a], dim=1)  # (r, r+b)
        zeros = torch.zeros(m, r_old, dtype=block.dtype, device=block.device)
        bot = torch.cat([zeros, s[:m].unsqueeze(1) * vh[:m]], dim=1)  # (m, r+b)
        b_fac = torch.cat([top, bot], dim=0)  # (r+m, r+b)

    u_loc, sigma, _ = torch.linalg.svd(b_fac, full_matrices=False)
    keep = min(rank_cap, int(sigma.shape[0]))
    if theta is not None:
        keep = max(1, min(keep, _truncation_rank(sigma, theta)))
    if min_sv_frac > 0.0 and sigma.numel() > 0 and sigma[0] > 0:
        # Week-17: relative singular-value floor -- drop directions whose singular
        # value is <= min_sv_frac of the leading one, so a rank-deficient block
        # does not pad the basis to rank_cap with near-null tail directions (the
        # high-rank divergence substrate: docs/week17). Self-scaling relative to
        # sigma[0] so no per-stream tuning; 0.0 = off = the bit-for-bit archived path.
        keep = max(1, min(keep, int((sigma > min_sv_frac * sigma[0]).sum().item())))
    u_new = u_aug @ u_loc[:, :keep]  # (n, keep)
    b_new = torch.diag(sigma[:keep])  # (keep, keep)
    rot = u_loc[:r_old, :keep].mT  # (keep, r_old)
    return u_new, b_new, rot


def blocked_bug_subspace(
    M: Tensor,
    rank_cap: int,
    block_size: int = 128,
    *,
    theta: float | None = None,
    min_sv_frac: float = 0.0,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Track the left (feature) subspace of ``M`` with a blocked augmented-BUG sweep.

    Parameters
    ----------
    M:
        Feature-by-token matrix, shape ``(n, T)``, on any device/dtype.
    rank_cap:
        Hard upper bound on the tracked rank ``r`` (``>= 1``).
    block_size:
        Number of columns consumed per augmented-BUG step (``>= 1``).
    theta:
        Optional Frobenius-tail truncation tolerance (see :func:`_truncation_rank`).
    min_sv_frac:
        Optional relative singular-value floor (see :func:`augmented_bug_step`);
        ``0.0`` (default) is a no-op.
    compute_dtype:
        Dtype for the QR/SVD/matmuls (default ``float32``); pass ``float64`` to
        compare against the numpy tracker at matched precision.

    Returns
    -------
    Tensor
        Orthonormal basis ``U`` of shape ``(n, r)`` in ``compute_dtype`` on
        ``M``'s device, with ``r <= rank_cap``.
    """
    if rank_cap < 1:
        raise ValueError(f"rank_cap must be >= 1, got {rank_cap}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    n, t = M.shape
    mc = M.to(compute_dtype)
    u: Tensor | None = None
    b_core: Tensor | None = None

    for start in range(0, t, block_size):
        block = mc[:, start : start + block_size]  # (n, b)
        u, b_core, _ = augmented_bug_step(
            u, b_core, block, rank_cap, theta=theta, min_sv_frac=min_sv_frac
        )

    if u is None:  # empty M (T == 0): return a trivial 1-column basis
        u = torch.zeros(n, 1, dtype=compute_dtype, device=M.device)
    return u


def blocked_bug_project(
    M: Tensor,
    rank_cap: int,
    block_size: int = 128,
    *,
    theta: float | None = None,
    min_sv_frac: float = 0.0,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Orthogonal projection ``U (U^T M)`` of ``M`` onto the tracked subspace.

    The reconstruction model for the data (mirrors
    :meth:`kvdlra.integrators.streaming.StreamingBUG.project`), returned in ``M``'s
    original storage dtype.
    """
    u = blocked_bug_subspace(
        M, rank_cap, block_size, theta=theta, min_sv_frac=min_sv_frac, compute_dtype=compute_dtype
    )
    mc = M.to(compute_dtype)
    recon = u @ (u.mT @ mc)
    return recon.to(M.dtype)
