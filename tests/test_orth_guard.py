"""Orthonormality tripwire + guard (CODE_AUDIT Part A §Q4).

The audit's synthetic ratchet stream: n=512, cap=256, block 16, a rank-40 signal + 4
massive-activation channels x1e3 + 1e-2 noise, bf16-rounded with an fp32 core (the pod
path). Both directions are pinned here:

* the **pre-Week-7** form of the augmented step -- plain residual QR, no
  re-orthogonalization and no rank-revealing residual admission -- ratchets
  ``‖UᵀU - I‖`` past 1 within ten steps on that stream (measured max 4.5e+01 at 50 steps,
  seeds 0-2);
* the **shipped** step holds at roundoff (measured max 1.1e-04 at 50 steps, 2.4e-04 at
  200, 6.9e-04 at 1400, seeds 0-2), i.e. below the ``orth_fix_tol=1e-3`` default, so the
  guard never fires on it.

The divergence CODE_AUDIT.md:172 records was observed through the *production layer*
(bf16 storage, RoPE round trip, real spectra: 6e-4 at 8K -> 35.1 at 36K), which is why
the guard measures the cache's stored basis rather than the bare tracker.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from itertools import pairwise
from typing import Any

import pytest
import torch
from torch import Tensor
from transformers import LlamaForCausalLM

from kvdlra.cache import BugStreamingCache, OrthonormalityError
from kvdlra.tracker import isvd
from kvdlra.tracker.isvd import eff_rank, isvd_step, orth_error, reorthonormalize
from tests.conftest import N_FEATURES
from tests.test_accounting import _drive

# ``‖QᵀQ - I‖_F`` after an fp32 thin QR sits at ~r x eps (~8e-6 at r=64) -- two decades
# below ``orth_fix_tol=1e-3``, so "restored to roundoff" is what a repair can assert.
ROUNDOFF = 1e-5

DIAG_KEYS = {
    "layer",
    "absorbs",
    "tokens_seen",
    "orth_err_k",
    "orth_err_v",
    "eff_rank_k",
    "eff_rank_v",
    "rank_k",
    "rank_v",
    "fixed_k",
    "fixed_v",
}


def ratchet_stream(n: int = 512, t: int = 200 * 16, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(n, 40, generator=g))[0]
    sig = q @ (torch.randn(40, t, generator=g) * torch.linspace(3.0, 0.5, 40).unsqueeze(1))
    m = sig + 1e-2 * torch.randn(n, t, generator=g)
    m[:4] *= 1e3  # four massive-activation channels
    out: Tensor = m.to(torch.bfloat16).to(torch.float32)  # bf16-rounded, fp32 core (pod path)
    return out


def rank_deficient_block(cols: int = 24, rank: int = 10, seed: int = 5) -> Tensor:
    """A seeding block of ``cols`` columns spanning only ``rank`` directions -- the pods'
    trigger, a PG-19 excerpt's opening whose repeated tokens give duplicate value vectors
    (D-011 addendum 5: ``eff_rank_v`` 93 of 128 columns)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(N_FEATURES, rank, generator=g)
    return torch.cat([base, base[:, :1].expand(-1, cols - rank)], dim=1)


def null_space_garbage_svd(rank: int) -> Callable[..., tuple[Tensor, Tensor, Tensor]]:
    """``torch.linalg.svd`` as the pods' GPU driver returned it for a rank-deficient
    matrix: the left columns beyond ``rank`` -- the null space, which no data pins down --
    come back duplicated instead of orthonormal (‖UᵀU - I‖ 0.5-0.9 on the three pods).
    CPU LAPACK does not do this, so a stub is the only way to run that failure here."""
    real = torch.linalg.svd

    def patched(mat: Tensor, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor]:
        u, s, vh = real(mat, *args, **kwargs)
        u = u.clone()
        u[:, rank:] = u[:, :1]
        return u, s, vh

    return patched


def _failed_repair(u: Tensor, c: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """A ``reorthonormalize`` that does not repair: the basis comes back merely rescaled,
    still non-orthonormal. An unrepairable basis is the only thing that aborts a trial now,
    and half-scaling makes the POST-repair error it is judged on measurably smaller than
    the PRE-repair error the diagnostic window keeps."""
    eye = torch.eye(u.shape[1], dtype=u.dtype, device=u.device)
    return u * 0.5, c, b, eye


def _rank_inflating_repair(
    u: Tensor, c: Tensor, b: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """A ``reorthonormalize`` whose repaired core reports a FULL effective rank -- the
    opposite of the deficient one that made the repair fire in the first place -- so a
    diag row read AFTER the repair would show a rank that never identified anything."""
    q: Tensor = torch.linalg.qr(u, mode="reduced")[0]
    full_rank_core = torch.eye(b.shape[0], dtype=b.dtype, device=b.device)
    rot = torch.eye(u.shape[1], dtype=u.dtype, device=u.device)
    return q, c, full_rank_core, rot


def _pre_w7_step(
    u: Tensor | None, b: Tensor | None, block: Tensor, cap: int
) -> tuple[Tensor, Tensor]:
    """The augmented step as it stood before the Week-7 robustness fix: the residual is
    factorized by a plain QR, with neither the ``resid - u (uᵀ resid)`` re-orthogonalization
    nor the rank-revealing admission test that keeps near-null directions out of the frame."""
    if u is None or b is None:
        q, r = torch.linalg.qr(block, mode="reduced")
        k = min(cap, q.shape[1])
        return q[:, :k], r[:k, :k]
    coords = u.mT @ block
    q, rr = torch.linalg.qr(block - u @ coords, mode="reduced")
    top = torch.cat([b, coords], dim=1)
    bot = torch.cat([rr.new_zeros((rr.shape[0], u.shape[1])), rr], dim=1)
    u_loc, sigma, _ = torch.linalg.svd(torch.cat([top, bot], dim=0), full_matrices=False)
    keep = min(cap, int(sigma.shape[0]))
    return torch.cat([u, q], dim=1) @ u_loc[:, :keep], torch.diag(sigma[:keep])


def _max_orth_error(
    step: Callable[[Tensor | None, Tensor | None, Tensor, int], tuple[Tensor, ...]],
    m: Tensor,
    cap: int = 256,
    block: int = 16,
) -> float:
    u: Tensor | None = None
    b: Tensor | None = None
    worst = 0.0
    for start in range(0, m.shape[1], block):
        out = step(u, b, m[:, start : start + block], cap)
        u, b = out[0], out[1]
        worst = max(worst, orth_error(u))
    return worst


def _decode_steps(model: LlamaForCausalLM, cache: BugStreamingCache, n: int) -> None:
    """Continue an already-prefilled cache for ``n`` single-token decode steps."""
    with torch.no_grad(), cache.attach(model):
        for _ in range(n):
            pos = torch.tensor([[cache.get_seq_length()]])
            model(torch.tensor([[7]]), past_key_values=cache, use_cache=True, position_ids=pos)


# --------------------------------------------------------------- the tracker


def test_pre_w7_augmentation_ratchets_past_one() -> None:
    assert _max_orth_error(_pre_w7_step, ratchet_stream(t=50 * 16)) > 1.0


def test_shipped_step_stays_below_fix_tol() -> None:
    assert _max_orth_error(isvd_step, ratchet_stream(t=200 * 16)) < 1e-3


def test_guard_restores_after_injection() -> None:
    g = torch.Generator().manual_seed(3)
    u = torch.linalg.qr(torch.randn(128, 64, generator=g))[0]
    c = torch.randn(64, 200, generator=g)
    b = torch.diag(torch.rand(64, generator=g))
    injected = u * 3.0  # the tripwire's own injection: far above orth_abort_tol
    assert orth_error(injected) > 1e-1
    q, c_new, b_new, rot = reorthonormalize(injected, c, b)
    assert orth_error(q) < ROUNDOFF
    assert torch.allclose(q @ c_new, injected @ c, atol=1e-5)
    assert torch.allclose(rot @ c, c_new)
    assert torch.allclose(b_new, torch.diag(torch.diagonal(b_new)))  # core stays diagonal


def test_reorthonormalize_preserves_reconstruction() -> None:
    g = torch.Generator().manual_seed(1)
    u = torch.linalg.qr(torch.randn(64, 8, generator=g))[0]
    u = u + 1e-3 * torch.randn(64, 8, generator=g)  # perturbed off orthonormality
    c = torch.randn(8, 50, generator=g)
    b = torch.diag(torch.rand(8, generator=g))
    q, c_new, b_new, rot = reorthonormalize(u, c, b)
    assert orth_error(q) < ROUNDOFF
    assert torch.allclose(q @ c_new, u @ c, atol=1e-6)
    assert torch.allclose(rot @ c, c_new)
    # The core is re-diagonalized, so it is the second moment ``U B Bᵀ Uᵀ`` -- what the
    # core stands for -- that carries across unchanged, not ``U B`` itself.
    assert torch.allclose(q @ b_new @ b_new.mT @ q.mT, u @ b @ b.mT @ u.mT, atol=1e-6)
    assert torch.allclose(b_new, torch.diag(torch.diagonal(b_new)))


def test_eff_rank_counts_above_floor() -> None:
    b = torch.diag(torch.tensor([3.0, 1.0, 1e-9, 0.0]))
    assert eff_rank(b) == 2
    assert eff_rank(torch.zeros(3, 3)) == 0  # an all-zero core has no live direction
    assert eff_rank(torch.zeros(0, 0)) == 0  # ... and neither has an empty one


def test_defaults_are_bit_identical_on_benign_stream() -> None:
    g = torch.Generator().manual_seed(2)
    m = torch.randn(128, 512, generator=g)
    u1 = b1 = u2 = b2 = None
    for s in range(0, 512, 16):
        u1, b1, _ = isvd_step(u1, b1, m[:, s : s + 16], 16)
        u2, b2, _ = isvd_step(u2, b2, m[:, s : s + 16], 16)
        if orth_error(u2) > 1e-3:  # never fires on a benign stream
            u2, _, b2, _ = reorthonormalize(u2, u2.new_zeros(16, 0), b2)
    assert u1 is not None and b1 is not None and u2 is not None and b2 is not None
    assert torch.equal(u1, u2) and torch.equal(b1, b2)


# ----------------------------------------------------------------- the cache


def test_cache_tripwire_records_and_aborts(
    tiny_model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=64,
        recent_window=8,
        absorb_block=4,
        orth_fix_tol=1e-3,
        orth_abort_tol=1e-1,
        diag_every=1,
    )
    _drive(tiny_model, cache)
    rows: list[dict[str, Any]] = cache.drain_diag()
    assert rows and rows[0].keys() == DIAG_KEYS
    assert cache.drain_diag() == []  # drained
    assert {row["layer"] for row in rows} == {0, 1}
    for row in rows:
        assert not row["fixed_k"] and not row["fixed_v"]  # benign stream: the guard sleeps
        assert row["orth_err_k"] < 1e-3 and row["orth_err_v"] < 1e-3
        assert 1 <= row["eff_rank_k"] <= row["rank_k"]
        assert row["absorbs"] > 0
        # positions-derived (fix1 Important #1): real even for pre-fill's own rows, not
        # just decode's -- the cache's counter alone would read 0 for every pre-fill row.
        assert row["tokens_seen"] > 0

    # Force an abort: corrupt U, make the repair fail, and absorb once more. A repair that
    # WORKS no longer aborts (see ``test_guard_repairs_before_it_aborts``), so an
    # unrepairable basis is the only way into this path.
    monkeypatch.setattr("kvdlra.cache.bug_cache.reorthonormalize", _failed_repair)
    layer = cache._bug_layers()[0]
    assert layer.u_k is not None
    layer.u_k = layer.u_k * 3.0
    pattern = r"layer=(\d+) orth_err=(\S+) > abort_tol=(\S+) at absorb (\d+)"
    with pytest.raises(OrthonormalityError, match=pattern) as excinfo:
        _decode_steps(tiny_model, cache, layer.absorb_block + 1)
    matched = re.search(pattern, str(excinfo.value))
    assert matched is not None
    # fix1 Minor #3: the window the trial died in is still in the drained rows, not lost.
    fatal_rows: list[dict[str, Any]] = [row for row in cache.drain_diag() if row["layer"] == 0]
    assert fatal_rows
    fatal = fatal_rows[-1]
    assert int(fatal["absorbs"]) == int(matched.group(4))
    assert fatal["fixed_k"]  # the repair was attempted -- it just did not restore anything
    # The message reports the POST-repair error (what the repair could not fix); the diag
    # window keeps the PRE-repair error, so the ratchet trace is the raw measurement.
    assert float(matched.group(2)) == pytest.approx(orth_error(layer.u_k), rel=1e-3)
    window = max(float(fatal["orth_err_k"]), float(fatal["orth_err_v"]))
    assert window > 1.5 * float(matched.group(2))


def test_qr_every_forces_fixes_and_keeps_the_basis_orthonormal(
    tiny_model: LlamaForCausalLM,
) -> None:
    """``qr_every=1`` re-orthonormalizes unconditionally -- including on the seeding
    absorb, where the coordinate buffer is still empty, and on every rank change."""
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=64,
        recent_window=8,
        absorb_block=4,
        qr_every=1,
        diag_every=1,
    )
    _drive(tiny_model, cache)
    rows = cache.drain_diag()
    assert rows and all(row["fixed_k"] and row["fixed_v"] for row in rows)
    for layer in cache._bug_layers():
        assert layer.u_k is not None and layer.u_v is not None
        assert orth_error(layer.u_k) < ROUNDOFF and orth_error(layer.u_v) < ROUNDOFF
        assert layer.b_k is not None
        assert torch.allclose(layer.b_k, torch.diag(torch.diagonal(layer.b_k)))


def test_quant_tier_rotation_skips_the_untouched_side(tiny_model: LlamaForCausalLM) -> None:
    """A fix on one stream only must not re-code the other: an identity rotation is not a
    no-op through the quantizer (it dequantizes and requantizes, injecting distortion)."""
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=16,
        recent_window=8,
        absorb_block=4,
        quant_bits=4,
        quant_budget=16,
    )
    _drive(tiny_model, cache)
    layer = cache._bug_layers()[0]
    assert layer._q_len() > 0 and layer.qk_codes is not None and layer.qv_codes is not None
    before_k, before_v = layer.qk_codes.clone(), layer.qv_codes.clone()
    layer._rotate_quant_tier(torch.eye(int(layer.rank)) * -1.0, None)
    assert layer.qv_codes is not None and torch.equal(layer.qv_codes, before_v)  # V untouched
    assert layer.qk_codes is not None and not torch.equal(layer.qk_codes, before_k)


def test_orth_knobs_validate(tiny_model: LlamaForCausalLM) -> None:
    for kw in (
        {"orth_fix_tol": 0.0},
        {"orth_abort_tol": -1.0},
        {"qr_every": 0},
        {"diag_every": 0},
        # A stream below the repair threshold is never repaired, so an abort threshold
        # below it could only fire on a basis the guard had decided not to touch.
        {"orth_abort_tol": 1e-4, "orth_fix_tol": 1e-3},
    ):
        with pytest.raises(ValueError):
            BugStreamingCache(tiny_model, rank=8, coord_budget=16, **kw)


# ------------------------------------------------------- fix round 1 coverage


def test_tokens_seen_from_positions_during_prefill(tiny_model: LlamaForCausalLM) -> None:
    """fix1 Important #1: single-shot pre-fill's absorb loop runs entirely before
    ``cumulative_length`` is advanced (``_prefill``), so ``tokens_seen`` must come from
    each block's own ``positions`` -- not the still-zero counter -- or every pre-fill row
    would plot at x=0 in a committed ``diag.jsonl``."""
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=64,
        recent_window=8,
        absorb_block=4,
        prefill_block_size=32,
        diag_every=1,
    )
    _drive(tiny_model, cache, t=200, n_new=0)  # prefill-only
    rows: list[dict[str, Any]] = [row for row in cache.drain_diag() if row["layer"] == 0]
    assert len(rows) > 1  # several absorbs inside the one prefill call
    seen = [int(row["tokens_seen"]) for row in rows]
    assert all(v > 0 for v in seen)
    assert all(b > a for a, b in pairwise(seen))  # strictly increasing


def test_rank_changed_is_per_stream(tiny_model: LlamaForCausalLM) -> None:
    """fix1 Minor #2: a V-side-only rank change must not force a K-side repair -- forced
    repairs under ``qr_every`` are attributable per stream."""
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=64,
        recent_window=8,
        absorb_block=4,
        qr_every=10_000,  # so the periodic trigger cannot fire on its own here
        diag_every=1,
    )
    _drive(tiny_model, cache)
    cache.drain_diag()  # discard the drive's own rows
    layer = cache._bug_layers()[0]
    layer._guard_orthonormality(
        rank_changed_k=False, rank_changed_v=True, positions=torch.tensor([0])
    )
    rows = [row for row in cache.drain_diag() if row["layer"] == 0]
    assert rows and rows[-1]["fixed_v"] and not rows[-1]["fixed_k"]


def test_orth_fix_tol_none_disables_repair(tiny_model: LlamaForCausalLM) -> None:
    """fix1 Minor #6: ``orth_fix_tol=None`` must never repair, however divergent the
    basis -- ``fixed_k`` stays False and the corrupted ``u_k`` is left untouched."""
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=64,
        recent_window=8,
        absorb_block=4,
        orth_fix_tol=None,
        orth_abort_tol=None,
        diag_every=1,
    )
    _drive(tiny_model, cache)
    cache.drain_diag()
    layer = cache._bug_layers()[0]
    assert layer.u_k is not None
    corrupted = layer.u_k * 3.0
    layer.u_k = corrupted
    layer._guard_orthonormality(
        rank_changed_k=False, rank_changed_v=False, positions=torch.tensor([0])
    )
    assert torch.equal(layer.u_k, corrupted)  # never repaired
    rows = [row for row in cache.drain_diag() if row["layer"] == 0]
    assert rows and not rows[-1]["fixed_k"]


def test_orth_abort_tol_none_disables_the_raise(tiny_model: LlamaForCausalLM) -> None:
    """fix1 Minor #6: ``orth_abort_tol=None`` must never raise, however divergent the
    basis -- the ordinary repair still runs (``orth_fix_tol`` stays at its default)."""
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=64,
        recent_window=8,
        absorb_block=4,
        orth_abort_tol=None,
        diag_every=1,
    )
    _drive(tiny_model, cache)
    cache.drain_diag()
    layer = cache._bug_layers()[0]
    assert layer.u_k is not None
    layer.u_k = layer.u_k * 3.0
    assert orth_error(layer.u_k) > 1e-1  # would abort at the default orth_abort_tol
    layer._guard_orthonormality(  # must not raise
        rank_changed_k=False, rank_changed_v=False, positions=torch.tensor([0])
    )
    assert orth_error(layer.u_k) < ROUNDOFF  # repaired instead of aborting
    rows = [row for row in cache.drain_diag() if row["layer"] == 0]
    assert rows and rows[-1]["fixed_k"]


# ------------------------------------------- repair before abort (D-011 addendum 5)


def test_guard_repairs_before_it_aborts(tiny_model: LlamaForCausalLM) -> None:
    """The harness defect that failed every guarded arm on three Table-4 pods: the guard
    checked ``orth_abort_tol`` BEFORE it tried the thin QR that fixes a one-step numerical
    event, so a repairable basis aborted the arm at absorb 1. At the DEFAULT knobs a basis
    far above the abort tolerance is now repaired, the absorb completes, and the window
    still carries the raw pre-repair error."""
    cache = BugStreamingCache(
        tiny_model, rank=8, coord_budget=64, recent_window=8, absorb_block=4, diag_every=1
    )
    _drive(tiny_model, cache)
    cache.drain_diag()
    layer = cache._bug_layers()[0]
    assert layer.u_k is not None
    layer.u_k = layer.u_k * 3.0
    assert orth_error(layer.u_k) > 1e-1  # the default orth_abort_tol

    _decode_steps(tiny_model, cache, layer.absorb_block + 1)  # must not raise

    assert layer.u_k is not None and orth_error(layer.u_k) < ROUNDOFF
    rows: list[dict[str, Any]] = cache.drain_diag()
    repaired = [row for row in rows if row["layer"] == 0 and row["fixed_k"]]
    assert repaired
    assert float(repaired[-1]["orth_err_k"]) > 1e-1  # pre-repair: the ratchet trace


def test_rank_deficient_seeding_block_absorbs_and_stays_orthonormal(
    tiny_model: LlamaForCausalLM,
) -> None:
    """The pods' trigger through the cache: a seeding block that spans fewer directions
    than the rank cap (theirs: ``eff_rank_v`` 93 of a 128-column block at rank 256). On CPU
    the SVD of that block's core comes back orthonormal, so this pins the CACHE path -- the
    seeding absorb with an empty coordinate buffer, the guard, the effective-rank row --
    not the CUDA driver failure, which needs the stub the next test installs."""
    cache = BugStreamingCache(
        tiny_model, rank=16, coord_budget=64, recent_window=8, absorb_block=4, diag_every=1
    )
    layer = cache._bug_layers()[0]
    block_v = rank_deficient_block()
    g = torch.Generator().manual_seed(6)
    block_k = torch.randn(N_FEATURES, block_v.shape[1], generator=g)

    layer._absorb_columns(block_k, block_v, torch.arange(block_v.shape[1]))

    assert layer.u_v is not None and orth_error(layer.u_v) < ROUNDOFF
    assert layer.b_v is not None and eff_rank(layer.b_v) < int(layer.u_v.shape[1])  # deficient
    (row,) = [r for r in cache.drain_diag() if r["layer"] == 0]
    assert float(row["orth_err_v"]) < ROUNDOFF  # type: ignore[arg-type]


def test_rank_deficient_seeding_block_survives_a_garbage_driver(
    tiny_model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pod failure end to end (D-011 addendum 5): a rank-deficient seeding block plus a
    driver whose null-space left columns are not orthonormal put ‖UᵀU - I‖ = 0.921 into the
    stored V basis at absorb 1, and the guard aborted the arm before trying the repair.
    ``_svd_core`` now hands the step an orthonormal left factor, so the garbage never
    reaches the basis and there is nothing to repair or abort."""
    monkeypatch.setattr(isvd, "NONORTHONORMAL", 0)
    cache = BugStreamingCache(
        tiny_model, rank=16, coord_budget=64, recent_window=8, absorb_block=4, diag_every=1
    )
    layer = cache._bug_layers()[0]
    block_v = rank_deficient_block()
    g = torch.Generator().manual_seed(6)
    block_k = torch.randn(N_FEATURES, block_v.shape[1], generator=g)
    monkeypatch.setattr(torch.linalg, "svd", null_space_garbage_svd(10))

    layer._absorb_columns(block_k, block_v, torch.arange(block_v.shape[1]))

    assert layer.u_v is not None and orth_error(layer.u_v) < ROUNDOFF
    assert isvd.NONORTHONORMAL > 0  # the core caught the driver, not the guard
    (row,) = [r for r in cache.drain_diag() if r["layer"] == 0]
    assert not row["fixed_v"]
    assert float(row["orth_err_v"]) < ROUNDOFF  # type: ignore[arg-type]


# --------------------------------------------------------------- fix1 (round 1) items


def test_fatal_row_keeps_the_pre_repair_effective_rank(
    tiny_model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fix1 A2: ``eff_rank`` must be folded into the diag window BEFORE the repair, the
    same rule the pre-repair error already follows -- a repair that re-derives the core
    from a fresh SVD can read back as full rank even when the defect that triggered it
    was a rank deficiency (the pods' own ``eff_rank_v`` 93 of 128)."""
    cache = BugStreamingCache(
        tiny_model, rank=8, coord_budget=64, recent_window=8, absorb_block=4, diag_every=1
    )
    _drive(tiny_model, cache)
    cache.drain_diag()
    layer = cache._bug_layers()[0]
    assert layer.u_k is not None
    layer.b_k = torch.diag(torch.tensor([5.0, 3.0] + [1e-9] * 6))  # eff_rank 2 of 8
    layer.u_k = layer.u_k * 3.0  # forces the repair (same injection as elsewhere)
    monkeypatch.setattr("kvdlra.cache.bug_cache.reorthonormalize", _rank_inflating_repair)

    layer._guard_orthonormality(
        rank_changed_k=False, rank_changed_v=False, positions=torch.tensor([0])
    )

    rows = [row for row in cache.drain_diag() if row["layer"] == 0]
    assert rows and rows[-1]["fixed_k"]
    assert int(rows[-1]["eff_rank_k"]) == 2  # type: ignore[call-overload]  # pre-repair, not 8


def test_nan_orth_error_forces_a_repair(
    tiny_model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fix1 A3: a NaN measurement must not slip past a plain ``>`` comparison (``NaN >
    tol`` is always False) -- treated as "not below tolerance", it has to repair."""
    cache = BugStreamingCache(
        tiny_model, rank=8, coord_budget=64, recent_window=8, absorb_block=4, diag_every=1
    )
    _drive(tiny_model, cache)
    cache.drain_diag()
    layer = cache._bug_layers()[0]
    calls = 0

    def nan_once(u: Tensor) -> float:
        nonlocal calls
        calls += 1
        return float("nan") if calls == 1 else orth_error(u)

    monkeypatch.setattr("kvdlra.cache.bug_cache.orth_error", nan_once)
    layer._guard_orthonormality(  # must not raise: the repaired basis is clean
        rank_changed_k=False, rank_changed_v=False, positions=torch.tensor([0])
    )
    rows = [row for row in cache.drain_diag() if row["layer"] == 0]
    assert rows and rows[-1]["fixed_k"]  # the NaN measurement was treated as "repair"


def test_nan_post_repair_error_aborts(
    tiny_model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fix1 A3: a repair whose OWN post-measurement comes back NaN must abort -- NaN is
    never "at or below" any tolerance, so it can never be silently accepted as fixed."""
    cache = BugStreamingCache(
        tiny_model, rank=8, coord_budget=64, recent_window=8, absorb_block=4, diag_every=1
    )
    _drive(tiny_model, cache)
    cache.drain_diag()
    layer = cache._bug_layers()[0]
    assert layer.u_k is not None
    layer.u_k = layer.u_k * 3.0  # real corruption: forces the repair without a stub
    calls = 0

    def nan_after_repair(u: Tensor) -> float:
        nonlocal calls
        calls += 1
        return orth_error(u) if calls == 1 else float("nan")

    monkeypatch.setattr("kvdlra.cache.bug_cache.orth_error", nan_after_repair)
    with pytest.raises(OrthonormalityError, match=r"orth_err=nan"):
        layer._guard_orthonormality(
            rank_changed_k=False, rank_changed_v=False, positions=torch.tensor([0])
        )
