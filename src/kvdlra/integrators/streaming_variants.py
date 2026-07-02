"""Alternative DLRA integrators as block-streaming KV subspace trackers.

Week-4.5 integrator ablation (``docs/week5-plan.md`` §"Week 4.5"). Before BUG
enters the expensive Week-5 comparisons we pick the best DLRA integrator for the
streaming-KV subspace-tracking problem via an *offline* bake-off (reconstruction
error vs the SVD oracle **+ per-step cost**, on captured KV; no LLM, no pod). The
incumbent -- the augmented rank-adaptive BUG step -- is
:func:`kvdlra.integrators.streaming_torch.blocked_bug_subspace`. This module adds
its two rivals so all three run through the **same** block-streaming harness
(identical first-block seeding, identical rank/``theta`` truncation, identical
``|| M - U U^T M ||_F / || M ||_F`` metric) and differ *only* in the per-block
integrator step:

* **Projector-Splitting (PSI)** -- :func:`psi_step` / :func:`psi_subspace`.
  Lubich--Oseledets 2014 (arXiv:1301.1058); the K/S/L split with a **backward**
  S-substep. Implemented in its natural *covariance-core* form (the backward
  substep ``S~ = S^ - P1^T (dG) U0`` is defined on the core of the tracked second
  moment ``G = sum_t k_t k_t^T``), which -- unlike BUG's square-root core -- forms
  ``S = B B^T`` and so squares the spectrum. The backward substep can render the
  core indefinite; the reconstruction metric depends only on ``span(U)`` (an
  orthonormal basis is maintained regardless), so PSI never NaNs the metric -- it
  can only track a worse *subspace*. Both the backward step and the covariance
  squaring are intrinsic to projector-splitting for this problem, and together are
  exactly *why* the project chose the square-root augmented BUG step; the ablation
  measures whether they cost anything on real KV.

* **Parallel-BUG** -- :func:`parallel_bug_step` / :func:`parallel_bug_subspace`.
  Ceruti--Kusch--Lubich parallel rank-adaptive integrator (arXiv:2304.05660),
  specialized to the streaming increment. Its defining move is to compute the
  basis update and the core update *independently* (the K- and L-steps do not wait
  on each other). For the one-sided symmetric KV problem this specializes to a
  **decoupled** update: the in-subspace rotation (an ``r x (r+b)`` SVD) and the
  new-direction admission (a ``b x b`` SVD) are computed and truncation-ranked
  independently, dropping the in-basis<->new-direction cross-coupling that BUG's
  joint ``(r+b) x (r+b)`` SVD retains. Cheaper per step when the block ``b`` is a
  sizeable fraction of ``r`` (``r^3 + b^3`` vs ``(r+b)^3``); the measured accuracy
  gap is exactly the discarded cross-coupling energy.

Both mirror :mod:`kvdlra.integrators.streaming_torch`'s conventions: rows =
features, columns = tokens (``docs/notes/conventions.md``); mixed precision with
the QR/SVD/matmuls in ``compute_dtype`` (default ``float32``; pass ``float64`` for
the offline ablation) regardless of ``M``'s storage dtype (PLAN §8 pitfall #4).

NOTE: library code -- no ``print``/I/O here. Parity + sanity tests live in
``tests/test_streaming_variants.py``.

References
----------
C. Lubich and I. V. Oseledets, "A projector-splitting integrator for dynamical
low-rank approximation," BIT Numer. Math. 54 (2014) 171--188, arXiv:1301.1058.

G. Ceruti, J. Kusch and C. Lubich, "A parallel rank-adaptive integrator for
dynamical low-rank approximation," SIAM J. Sci. Comput. 46 (2024) B205--B228,
arXiv:2304.05660.

See also :mod:`kvdlra.integrators.streaming_torch` (the incumbent BUG tracker and
the shared blocked harness) and :mod:`kvdlra.integrators.streaming` (the numpy
per-token reference).
"""

from __future__ import annotations

import torch
from torch import Tensor

from kvdlra.integrators.streaming_torch import _truncation_rank

__all__ = [
    "parallel_bug_project",
    "parallel_bug_step",
    "parallel_bug_subspace",
    "psi_project",
    "psi_step",
    "psi_subspace",
]


def _seed_block(block: Tensor, rank_cap: int, theta: float | None) -> tuple[Tensor, Tensor]:
    """Seed a tracker from the first block: truncated SVD of its columns.

    Returns ``(U, sigma)`` with ``U`` an ``(n, keep)`` orthonormal basis and
    ``sigma`` the kept singular values (``keep <= rank_cap``). Identical to the
    first-block branch of :func:`blocked_bug_subspace`, so every integrator starts
    from the same rank-``keep`` truncated-SVD state (the oracle for block 1).
    """
    q, r = torch.linalg.qr(block, mode="reduced")  # q:(n,b) r:(b,b)
    u_loc, sigma, _ = torch.linalg.svd(r, full_matrices=False)
    keep = min(rank_cap, int(sigma.shape[0]))
    if theta is not None:
        keep = max(1, min(keep, _truncation_rank(sigma, theta)))
    return q @ u_loc[:, :keep], sigma[:keep]


def parallel_bug_step(
    u: Tensor,
    b_core: Tensor,
    block: Tensor,
    rank_cap: int,
    theta: float | None = None,
) -> tuple[Tensor, Tensor]:
    """One decoupled parallel-BUG block step (square-root core).

    Given the current orthonormal basis ``u`` ``(n, r)`` and square-root core
    ``b_core`` ``(r, r)`` (``b_core @ b_core.T == U^T G U``), absorb the block
    ``block`` ``(n, b)`` by updating the in-subspace rotation and the new-direction
    admission **independently** (the parallel integrator's decoupling), then
    truncation-rank the two candidate sets together. Drops the
    in-basis<->new-direction cross term that :func:`blocked_bug_subspace` retains.

    Returns the new ``(u, b_core)`` with ``u`` orthonormal ``(n, keep)`` and
    ``b_core`` diagonal ``(keep, keep)``, ``keep <= rank_cap``.
    """
    a = u.mT @ block  # (r, b) coordinates in the current basis
    r_perp = block - u @ a  # (n, b) out-of-basis residual
    q, rqr = torch.linalg.qr(r_perp, mode="reduced")  # q:(n,b) rqr:(b,b)

    # In-subspace update: singular values/rotation of the grown square-root core
    # restricted to the OLD basis (r x (r+b)); ignores the new directions.
    u_in, sig_in, _ = torch.linalg.svd(torch.cat([b_core, a], dim=1), full_matrices=False)
    # New-direction update: energy of the residual block, independent of the above.
    u_new, sig_new, _ = torch.linalg.svd(rqr, full_matrices=False)

    cand_u = torch.cat([u @ u_in, q @ u_new], dim=1)  # (n, r+b) candidate directions
    cand_sig = torch.cat([sig_in, sig_new])  # (r+b,) their singular values
    sig_sorted, order = torch.sort(cand_sig, descending=True)
    keep = min(rank_cap, int(cand_sig.shape[0]))
    if theta is not None:
        keep = max(1, min(keep, _truncation_rank(sig_sorted, theta)))
    sel = order[:keep]
    return cand_u[:, sel], torch.diag(cand_sig[sel])


def psi_step(
    u: Tensor,
    s_cov: Tensor,
    block: Tensor,
    rank_cap: int,
) -> tuple[Tensor, Tensor]:
    """One projector-splitting (PSI) block step, covariance-core form (fixed rank).

    Tracks the second moment ``G = sum_t k_t k_t^T`` as ``Y = U S U^T`` with ``S``
    the ``(r, r)`` covariance core. Absorbs the block's rank-``b`` increment
    ``dG = C C^T`` (``C = block``) with the first-order Lie--Trotter K/S/L split of
    Lubich--Oseledets 2014, whose middle S-substep is integrated **backward** in
    time (the ``- P1^T dG U0`` subtraction) -- the family's characteristic
    (in)stability source. The step propagates rank ``r`` (basic projector-splitting
    is fixed-rank; the ablation sweeps ``rank_cap`` explicitly), but the augmented
    ``[U | Q]`` frame lets genuinely new directions rotate into the basis.

    Returns ``(u, s_cov)`` with ``u`` orthonormal ``(n, r)`` and ``s_cov`` the
    ``(r, r)`` covariance core (upper-triangular; may be indefinite -- only
    ``span(u)`` enters the reconstruction metric).
    """
    r = u.shape[1]
    a = u.mT @ block  # (r, b)
    r_perp = block - u @ a  # (n, b)
    q, rqr = torch.linalg.qr(r_perp, mode="reduced")  # q:(n,k) rqr:(k,b), k = min(n,b)
    k = q.shape[1]  # number of genuinely new orthonormal directions this block
    u_aug = torch.cat([u, q], dim=1)  # (n, r+k) fixed frame for this step
    c_hat = torch.cat([a, rqr], dim=0)  # (r+k, b): coords of C in u_aug
    c_top = c_hat[:r, :]  # (r, b): its projection onto the OLD basis

    zeros_br = torch.zeros(k, r, dtype=block.dtype, device=block.device)
    # --- K-step: K1 = U0 S0 + dG U0 in u_aug coords (U0 = first r frame axes) ---
    u0_s0 = torch.cat([s_cov, zeros_br], dim=0)  # (r+k, r): [[S0],[0]]
    k1 = u0_s0 + c_hat @ c_top.mT  # (r+k, r)
    p1, s_hat = torch.linalg.qr(k1, mode="reduced")  # p1:(r+k,r) s_hat:(r,r)
    # --- S-step (BACKWARD): S~ = S^ - P1^T dG U0 ---
    s_tilde = s_hat - (p1.mT @ c_hat) @ c_top.mT  # (r, r)
    # --- L-step: L1 = V0 S~^T + dG P1 (symmetric V0 = U0) ---
    s_tilde_top = torch.cat([s_tilde.mT, zeros_br], dim=0)  # (r+k, r)
    l1 = s_tilde_top + c_hat @ (c_hat.mT @ p1)  # (r+k, r)
    p2, s2 = torch.linalg.qr(l1, mode="reduced")  # p2:(r+k,r) s2:(r,r)
    return u_aug @ p2, s2


def parallel_bug_subspace(
    M: Tensor,
    rank_cap: int,
    block_size: int = 128,
    *,
    theta: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Track the left subspace of ``M`` with the decoupled parallel-BUG integrator.

    Signature/semantics mirror
    :func:`kvdlra.integrators.streaming_torch.blocked_bug_subspace`; the per-block
    rule is :func:`parallel_bug_step`. Returns the orthonormal basis ``U``
    ``(n, r)`` in ``compute_dtype`` on ``M``'s device, ``r <= rank_cap``.
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
        block = mc[:, start : start + block_size]
        if u is None:
            u, sigma = _seed_block(block, rank_cap, theta)
            b_core = torch.diag(sigma)
        else:
            assert b_core is not None
            u, b_core = parallel_bug_step(u, b_core, block, rank_cap, theta)
    if u is None:
        u = torch.zeros(n, 1, dtype=compute_dtype, device=M.device)
    return u


def psi_subspace(
    M: Tensor,
    rank_cap: int,
    block_size: int = 128,
    *,
    theta: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Track the left subspace of ``M`` with the projector-splitting integrator.

    Signature/semantics mirror
    :func:`kvdlra.integrators.streaming_torch.blocked_bug_subspace`; the per-block
    rule is :func:`psi_step` (fixed-rank; ``theta`` only affects the first-block
    seed rank). Returns the orthonormal basis ``U`` ``(n, r)`` in ``compute_dtype``
    on ``M``'s device, ``r <= rank_cap``.
    """
    if rank_cap < 1:
        raise ValueError(f"rank_cap must be >= 1, got {rank_cap}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    n, t = M.shape
    mc = M.to(compute_dtype)
    u: Tensor | None = None
    s_cov: Tensor | None = None
    for start in range(0, t, block_size):
        block = mc[:, start : start + block_size]
        if u is None:
            u, sigma = _seed_block(block, rank_cap, theta)
            s_cov = torch.diag(sigma**2)  # covariance core: eigenvalues are sigma^2
        else:
            assert s_cov is not None
            u, s_cov = psi_step(u, s_cov, block, rank_cap)
    if u is None:
        u = torch.zeros(n, 1, dtype=compute_dtype, device=M.device)
    return u


def _project(subspace: Tensor, M: Tensor, compute_dtype: torch.dtype) -> Tensor:
    """Orthogonal projection ``U (U^T M)`` returned in ``M``'s storage dtype."""
    mc = M.to(compute_dtype)
    return (subspace @ (subspace.mT @ mc)).to(M.dtype)


def parallel_bug_project(
    M: Tensor,
    rank_cap: int,
    block_size: int = 128,
    *,
    theta: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Orthogonal projection of ``M`` onto the parallel-BUG-tracked subspace."""
    u = parallel_bug_subspace(M, rank_cap, block_size, theta=theta, compute_dtype=compute_dtype)
    return _project(u, M, compute_dtype)


def psi_project(
    M: Tensor,
    rank_cap: int,
    block_size: int = 128,
    *,
    theta: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Orthogonal projection of ``M`` onto the PSI-tracked subspace."""
    u = psi_subspace(M, rank_cap, block_size, theta=theta, compute_dtype=compute_dtype)
    return _project(u, M, compute_dtype)
