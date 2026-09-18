"""Blocked (chunked) streaming subspace tracker in PyTorch.

Block incremental SVD with rank truncation (Brand 2006): each step range-augments
the orthonormal basis with the block's out-of-basis residual, rebuilds the
square-root core, and truncates back to the rank cap. The augmented step and its
rank-adaptive truncation criterion are taken from Ceruti--Kusch--Lubich
(arXiv:2104.05247 §2); at the shipped settings (``theta=None, min_sv_frac=0``) it
IS fixed-rank incremental SVD. No error bound is claimed here: the robustness
results that step comes with are for an ODE flow, not a column stream.

It processes the feature-by-token matrix ``M`` (rows = features, columns =
tokens) **a block of columns at a time** instead of one column at a time.

Why blocked (the motivation)
----------------------------
The numpy per-token tracker does one small SVD per token in a Python loop -- fine
for the Week-2 *proof* that streaming BUG tracks the SVD oracle, but far too slow
for Week-3/4 perplexity sweeps and 8B (the loop is CPU-bound; a GPU does not help
it). Processing ``block_size`` columns per augmented-BUG step turns ``T`` Python
iterations into ``ceil(T / block_size)`` and runs the QR/SVD/matmuls as batched
``torch`` ops on the tensor's own device (GPU). It is the **same** augmented
step (Ceruti--Kusch--Lubich 2022, arXiv:2104.05247 §2) with a rank-``b``
data increment per step rather than rank-1; the block size is a speed/fidelity
knob:

    * ``block_size = 1``   -> the per-token streaming tracker.
    * ``block_size = T``   -> a single augmented step == the truncated-SVD oracle.
    * intermediate         -> a coarse streaming integration; Week-2 showed BUG
      and the oracle agree to ~1-3% on real KV, so all block sizes land in that
      band. ``tests/test_isvd.py`` pins the two endpoints and the band between
      them.

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

Mixed precision (PLAN §8 pitfall #4): the QR/SVD/matmuls run in
``compute_dtype`` (default ``float32``) regardless of the storage dtype of ``M``;
bf16 storage is safe, bf16 *core* math is not.

NOTE: library code -- no ``print``/I/O here.

References
----------
M. Brand, "Fast low-rank modifications of the thin singular value decomposition,"
Linear Algebra Appl. 415 (2006) 20--30 -- the incremental SVD this reduces to at
``theta=None, min_sv_frac=0``. G. Ceruti, J. Kusch and C. Lubich, "A rank-adaptive
robust integrator for dynamical low-rank approximation," BIT Numer. Math. 62
(2022) 1149--1174, arXiv:2104.05247, §2 -- the augmented step and the
Frobenius-tail truncation criterion. The core runs in fp32 even when the data is
stored in bf16 (PLAN §8 pitfall #4).
"""

from __future__ import annotations

import json

import torch
from torch import Tensor

__all__ = [
    "augmented_bug_step",
    "blocked_bug_project",
    "blocked_bug_subspace",
    "eff_rank",
    "isvd_step",
    "orth_error",
    "reorthonormalize",
]


def orth_error(u: Tensor) -> float:
    """Departure of ``u`` from orthonormality: ``‖UᵀU - I‖_F`` (0 for an exact basis).

    The divergence monitor (CODE_AUDIT Part A §Q4): the tracked basis has no
    orthonormality guarantee of its own, and every quantity the cache derives from it
    -- coordinates, reconstruction, the Pythagoras identity behind the surprise scores
    -- assumes ``UᵀU = I``. Computed in ``u``'s own dtype (no promotion), and outside
    autograd: it is a measurement, and the caller may well be inside a grad-enabled
    forward (nothing can differentiate through the returned float anyway).
    """
    with torch.no_grad():
        eye = torch.eye(u.shape[1], dtype=u.dtype, device=u.device)
        return float(torch.linalg.norm(u.mT @ u - eye))


@torch.no_grad()
def reorthonormalize(u: Tensor, c: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Restore orthonormality of ``u`` while preserving what it represents.

    Thin QR ``u = q r``, then re-diagonalize the core: the returned basis is orthonormal,
    ``u @ c == u_new @ c_new`` exactly (to roundoff) for stored coordinates ``c``
    ``(r, cols)``, ``b_new`` is diagonal (the core's *stored* form -- the accounting bills
    its ``r`` diagonal entries, not ``r²``), and the second moment is carried exactly:
    ``u_new b_new b_newᵀ u_newᵀ == u b bᵀ uᵀ``.

    Returns ``(u_new, c_new, b_new, rot)`` where ``rot`` maps old coordinates to new
    (``rot @ c == c_new``) -- the same contract as the step's own ``rot``, so callers
    rotate every other coordinate tier (e.g. the quantized one) with it.
    """
    q, r = torch.linalg.qr(u, mode="reduced")
    u_loc, sigma, _ = torch.linalg.svd(r @ b, full_matrices=False)  # (r, r), cheap
    rot = u_loc.mT @ r
    return (q @ u_loc).contiguous(), rot @ c, torch.diag(sigma), rot


def eff_rank(b: Tensor, rel: float = 1e-6) -> int:
    """Live directions in a diagonal core ``b``: diagonal entries above ``rel`` times the
    leading one. A basis padded to ``rank_cap`` with near-null tail directions (the
    high-rank divergence substrate) reports an effective rank far below its stored one.

    Diagonal is the incremental-SVD and FD contract, not a universal one: the Oja
    tracker's ``_carry_core`` returns a triangular R-factor, whose diagonal entries are
    not its singular values, so on that tracker this counts the diagonal it is given and
    only bounds the live rank. It is a diagnostic either way -- nothing is billed or
    truncated from it (``stored_state_numel`` counts the stored ``r``)."""
    with torch.no_grad():  # a measurement, like orth_error: never part of a graph
        d = torch.diagonal(b).abs()
        if d.numel() == 0 or float(d.max()) == 0.0:
            return 0
        return int((d > rel * d.max()).sum().item())


def _truncation_rank(sigma: Tensor, theta: float) -> int:
    """Smallest ``k`` whose discarded singular tail ``(sum_{j>k} sigma_j^2)^{1/2}``
    is ``<= theta`` (the rank-adaptive truncation criterion, arXiv:2104.05247 §2)."""
    # sigma is sorted descending (torch.linalg.svd convention).
    tail_sq = torch.flip(torch.cumsum(torch.flip(sigma**2, (0,)), 0), (0,))
    # tail_sq[k] == sum_{j>=k} sigma_j^2; we want the smallest k with the tail
    # *beyond* k (i.e. tail_sq[k]) <= theta^2.
    ok = tail_sq <= theta**2
    idx = torch.nonzero(ok, as_tuple=False)
    if idx.numel() == 0:
        return int(sigma.shape[0])
    return max(1, int(idx[0].item()))


def _augment(u: Tensor | None, b_core: Tensor | None, block: Tensor) -> tuple[Tensor, Tensor, int]:
    """Range-augment ``u`` by the block's out-of-basis residual: steps 1-3 of the
    blocked step, shared by every tracker that augments before it truncates.

    Returns ``(u_aug, b_fac, r_old)`` -- the grown orthonormal basis ``(n, r + m)``, the
    grown square-root core ``(r + m, r + b)`` and the incoming rank, so the caller's own
    truncation can build ``rot = u_loc[:r_old, :keep].mT``. ``u is None`` seeds from the
    block's reduced QR. The admitted directions ``u_aug[:, r_old:]`` are the residual's
    left singular vectors, ordered by singular value.

    The residual is re-orthogonalized against ``u`` once ("twice is enough",
    Parlett/Kahan) and rank-revealed by SVD: only directions with singular value above
    ``100 * eps(dtype) * ||block||_F`` are admitted, and at most ``n - r`` of them so the
    augmented basis still fits in ``R^n``. Without the rank-revealing admission a
    numerically null residual (the block already lies in the tracked subspace) makes the
    plain QR hand back junk directions whose orthogonality against ``u`` cancellation has
    destroyed -- the Week-7 fix, and the defect that crashed the Week-20 FD arm.
    """
    n = block.shape[0]
    if u is None:
        # Seeding step: the basis is the reduced QR of the first block.
        q, r = torch.linalg.qr(block, mode="reduced")
        return q, r, 0
    assert b_core is not None  # callers check the pairing
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
    return u_aug, b_fac, r_old


# ``torch.linalg`` re-exports the C++ exception under a different name
# (``_LinAlgError as LinAlgError``), which mypy does not count as an explicit export.
_LinAlgError: type[Exception] = torch.linalg.LinAlgError  # type: ignore[attr-defined]


# Gram-path entries this process has taken. Counted for the tests; the first one also
# prints the ``[diag]`` line below, once, so a run that fell back says so in its log.
FALLBACKS = 0


def _svd_core(b_fac: Tensor) -> tuple[Tensor, Tensor]:
    """Left factors and singular values of the augmented core, untruncated.

    ``torch.linalg.svd(b_fac, full_matrices=False)`` with a fallback: LAPACK's divide-and-
    conquer driver raises ``LinAlgError`` ("too many repeated singular values") on cores
    whose spectrum carries repeated exact zeros -- which is what killed the Week-20 FD arm,
    whose shrinkage zeroes the whole tail. On CUDA the first retry is cuSOLVER's ``gesvd``
    driver, which converges on spectra its default (``gesvdj``/DC) refuses; CPU LAPACK
    takes no ``driver`` argument, so that branch is unreachable -- and untested -- here.
    Then the eigendecomposition of the Gram ``b_fac b_facᵀ``, which computes the same
    factors by a different driver (at half the precision -- it squares the condition
    number -- which is why it is a fallback and not the path); a second failure retries it
    once on a jittered Gram, and a third re-raises.

    ``b_fac`` is never TALLER than it is wide at any call site -- it is ``(r + m, r + b)``
    with ``m <= b`` -- so the Gram's ``r + m`` eigenpairs are exactly the ``min(rows, cols)``
    factors ``full_matrices=False`` returns.
    """
    global FALLBACKS
    try:
        u_loc, sigma, _ = torch.linalg.svd(b_fac, full_matrices=False)
        return u_loc, sigma
    except _LinAlgError:
        if b_fac.is_cuda:  # a second driver before a second algorithm: full precision
            try:
                u_loc, sigma, _ = torch.linalg.svd(b_fac, full_matrices=False, driver="gesvd")
                return u_loc, sigma
            except _LinAlgError:
                pass
        gram = b_fac @ b_fac.mT
        jitter: Tensor | float = 0.0  # 0.0 on the first attempt: nothing was added to the Gram
        attempt = 1
        try:
            evals, evecs = torch.linalg.eigh(gram)
        except _LinAlgError:
            attempt = 2
            dim = gram.shape[0]
            jitter = 1e-7 * torch.diagonal(gram).sum() / dim
            eye = torch.eye(dim, dtype=gram.dtype, device=gram.device)
            evals, evecs = torch.linalg.eigh(gram + jitter * eye)  # a third failure raises
        FALLBACKS += 1
        if FALLBACKS == 1:  # once per process: the trace, not a per-step log
            payload = {"event": "svd_fallback", "attempt": attempt, "shape": list(b_fac.shape)}
            print("[diag] " + json.dumps(payload, sort_keys=True), flush=True)
        # eigh returns ascending eigenvalues; svd returns descending singular values. Subtract
        # the jitter added to the Gram (0.0 unless the second attempt fired) so a numerically
        # zero singular value is recovered as exactly 0, not sqrt(jitter).
        sigma = torch.sqrt(torch.clamp(torch.flip(evals, (0,)) - jitter, min=0.0))
        return torch.flip(evecs, (1,)), sigma


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
    increment at a time with the same block incremental-SVD step.

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
        tail directions (the Week-17 high-rank stability fix).

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
    n_features`` (the Week-5 ablation follow-up): the residual has
    rank at most ``n - r`` mathematically, so the discarded QR columns are
    numerically null and the clamp is exact up to roundoff.

    Additionally (Week-7 fix), the residual is **re-orthogonalized** against
    ``u`` once and rank-revealed by SVD (see :func:`_augment`, which holds the
    augmentation this step shares with :func:`fd_step`). Without this, a
    *numerically null* residual (the incoming block already lies in the tracked
    subspace -- e.g. data of effective dimension < ``rank_cap``) makes the plain
    QR return junk directions whose orthogonality against ``u`` is destroyed by
    cancellation, silently breaking the basis (found by the Week-7 full-rank
    parity tests on a tiny model; the pathology never fired at r=128 on real KV,
    so the *archived* Week-3..6 results stand as-is). Note the change applies on
    every step, so reruns of older configs are **fp-equivalent, not
    bit-identical** (QR -> SVD residual factorization + the coordinate
    refinement term) -- compare methods within one run, never fresh curves
    against archived JSON at bit level.
    """
    if (u is None) != (b_core is None):
        raise ValueError("u and b_core must be provided together (or both None)")
    if rank_cap < 1:
        raise ValueError(f"rank_cap must be >= 1, got {rank_cap}")

    u_aug, b_fac, r_old = _augment(u, b_core, block)
    u_loc, sigma = _svd_core(b_fac)
    keep = min(rank_cap, int(sigma.shape[0]))
    if theta is not None:
        keep = max(1, min(keep, _truncation_rank(sigma, theta)))
    if min_sv_frac > 0.0 and sigma.numel() > 0 and sigma[0] > 0:
        # Week-17: relative singular-value floor -- drop directions whose singular
        # value is <= min_sv_frac of the leading one, so a rank-deficient block
        # does not pad the basis to rank_cap with near-null tail directions (the
        # high-rank divergence substrate). Self-scaling relative to
        # sigma[0] so no per-stream tuning; 0.0 = off = the bit-for-bit archived path.
        keep = max(1, min(keep, int((sigma > min_sv_frac * sigma[0]).sum().item())))
    u_new = u_aug @ u_loc[:, :keep]  # (n, keep)
    b_new = torch.diag(sigma[:keep])  # (keep, keep)
    rot = u_loc[:r_old, :keep].mT  # (keep, r_old)
    return u_new, b_new, rot


# ``isvd_step`` is an alias for ``augmented_bug_step`` (the function keeps its original
# name, defined above -- the step IS block incremental SVD at the shipped settings); new
# code should call it by the new name, ``isvd_step``.
isvd_step = augmented_bug_step


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

    The reconstruction model for the data, returned in ``M``'s original storage
    dtype.
    """
    u = blocked_bug_subspace(
        M, rank_cap, block_size, theta=theta, min_sv_frac=min_sv_frac, compute_dtype=compute_dtype
    )
    mc = M.to(compute_dtype)
    recon = u @ (u.mT @ mc)
    return recon.to(M.dtype)


# ----------------------------------------------------------------------------
# Week-20 tracker-swap ablation: drop-in alternatives to ``augmented_bug_step``
# with the SAME ``(u, b_core, block, rank_cap) -> (u_new, b_new, rot)`` contract,
# so ``BugStreamingCache(tracker=...)`` can swap the gist tracker while every
# other part of the cache (sinks, ring, surprise tier, seed, accounting) is held
# fixed. ``rot = u_new^T u_old`` carries stored coordinates across the basis
# change exactly as the BUG step does. Note that ``augmented_bug_step`` with
# ``theta=None, min_sv_frac=0`` (the r64 configuration's defaults) IS fixed-rank
# incremental SVD (Brand 2006), so that arm needs no new code.
# ----------------------------------------------------------------------------


def _carry_core(u_old: Tensor, b_old: Tensor, u_new: Tensor, block: Tensor) -> Tensor:
    """Square-root core of the projected data in the NEW basis, ``(r', r')``:
    ``m = [rot @ b_old | u_new^T block]`` has ``m m^T = P (old data + block) P^T``;
    the R-factor of ``m^T`` gives ``b_new`` with ``b_new b_new^T = m m^T``."""
    rot = u_new.mT @ u_old
    m = torch.cat([rot @ b_old, u_new.mT @ block], dim=1)
    core: Tensor = torch.linalg.qr(m.mT, mode="reduced")[1].mT.contiguous()
    return core


def oja_step(
    u: Tensor | None,
    b_core: Tensor | None,
    block: Tensor,
    rank_cap: int,
    *,
    n_seen: int = 0,
    eta0: float,
    decay: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Oja's-rule subspace tracker (the OjaKV baseline), one block of columns.

    Oja's rule as the tracker's drop-in alternative (Week-20 swap): each
    column is L2-normalized and applied sequentially,
    ``U <- orth(U + eta_t * c (c^T U))`` with ``eta_t = eta0 / (1 + decay * t)``
    (``t = n_seen`` at the block's first column, advancing one per column).
    Seeding is the reduced QR of the first block (as the incremental-SVD step),
    so the two trackers start identical and differ only in how they advance.

    ``eta0`` and ``decay`` are REQUIRED: the schedule is the whole content of the
    arm, and the Week-20 swap pod ran whatever defaults this signature happened to
    carry (``1.0, 1e-3``), which is why its cell is void. There is no schedule to
    fall back on -- an arm names one or the call fails.

    The basis GROWS to ``rank_cap`` instead of being pinned at the seeding block's
    width. Oja's update spans nothing new (it re-weights directions already in
    ``U``), so a basis seeded from a 16-column block and then only re-orthonormalized
    stayed rank 16 forever while the accounting billed ``rank_cap`` -- the Week-20
    arm tracked a rank-16 gist billed as rank 64. Each block therefore first takes
    the leading out-of-basis directions from :func:`_augment` (ordered by residual
    singular value) up to ``rank_cap`` columns, and the Oja updates run on the grown
    basis. The rank the arm is billed for is the rank it tracks: ``min(rank_cap, n)``
    once enough columns have been seen.
    """
    if (u is None) != (b_core is None):
        raise ValueError("u and b_core must be provided together (or both None)")
    if rank_cap < 1:
        raise ValueError(f"rank_cap must be >= 1, got {rank_cap}")
    if u is None:
        q, r = torch.linalg.qr(block, mode="reduced")
        k = min(rank_cap, q.shape[1])
        u_new, b_new = q[:, :k].contiguous(), r[:k, :k].contiguous()
        return u_new, b_new, u_new.new_zeros((k, 0))
    assert b_core is not None
    u_cur = u
    if u_cur.shape[1] < rank_cap:  # grow first, then learn in the grown basis
        u_cur = _augment(u_cur, b_core, block)[0][:, :rank_cap]
    t = n_seen
    for j in range(block.shape[1]):
        c = block[:, j : j + 1]
        norm = torch.linalg.vector_norm(c)
        if float(norm.detach()) > 1e-12:  # detach: a measurement, never a graph node
            c = c / norm
            eta = eta0 / (1.0 + decay * t)
            u_cur = torch.linalg.qr(u_cur + eta * (c @ (c.mT @ u_cur)), mode="reduced")[0]
        t += 1
    u_new = u_cur.contiguous()
    return u_new, _carry_core(u, b_core, u_new, block), u_new.mT @ u


def fd_step(
    u: Tensor | None,
    b_core: Tensor | None,
    block: Tensor,
    rank_cap: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Frequent Directions (Liberty 2013) on the column stream, one block.

    Exactly the incremental-SVD step's augmentation (:func:`_augment`) and core
    factorization (:func:`_svd_core`); the one algorithmic difference is the
    truncation, which is FD's *shrinkage*: subtract the ``(rank_cap+1)``-th
    squared singular value from every kept one before dropping the tail. So this
    arm isolates the shrinkage and nothing else -- with ``rank_cap`` at the
    feature count there is no tail, and the two steps track the same subspace.
    Seeding = reduced QR of the first block. No ``theta`` and no ``min_sv_frac``:
    FD's rank is its own, fixed at ``rank_cap``.

    The Week-20 arm crashed here. Recorded (results/w19_harvest/swap-llama.raw:69-71, kept
    locally, not in the paper-v1 archive): the pod's log reads ``linalg.svd: The algorithm
    failed to converge because the input matrix is ill-conditioned or has too many repeated
    singular values``. Reproduced (tests/test_fd_numerics.py): a rank-deficient augmented
    core after shrinkage -- the arm's own augmentation kept the residual's plain QR,
    admitting ``b`` junk directions per step once the block already lay in the tracked
    subspace, and the shrinkage then zeroed their whole spectrum before the next core SVD.
    That is the same CLASS of failure as the recorded one, not a verified-identical trace;
    no more than that is claimed. The shared rank-revealing ``_augment`` keeps the junk
    directions out, and ``_svd_core``'s eigh fallback catches what still reaches it.
    """
    if (u is None) != (b_core is None):
        raise ValueError("u and b_core must be provided together (or both None)")
    if rank_cap < 1:
        raise ValueError(f"rank_cap must be >= 1, got {rank_cap}")
    u_aug, b_fac, r_old = _augment(u, b_core, block)
    u_loc, s = _svd_core(b_fac)
    k = min(rank_cap, int(s.shape[0]))
    # FD shrinkage: delta = sigma_{k+1}^2 (0 if the augmented core has no tail).
    delta = s[k] ** 2 if s.shape[0] > k else s.new_zeros(())
    s_new = torch.sqrt(torch.clamp(s[:k] ** 2 - delta, min=0.0))
    u_new = (u_aug @ u_loc[:, :k]).contiguous()
    b_new = torch.diag(s_new).contiguous()
    return u_new, b_new, u_loc[:r_old, :k].mT  # == u_new^T u, and (k, 0) when seeding
