"""bf16 gist storage (L5.1): ``gist_dtype``, the cheapest halving of the stored state.

The gist -- the basis ``U``, the diagonal core ``B`` and the coordinate tier ``C`` -- is
what ``ratio_stored_bits`` bills at 32 bits per element; everything else the cache keeps
(sinks, ring, exact tier) is already model-dtype. Storing the three in bf16 halves that
part of the bill. The knob is default-off and the fp32 path is bit-identical
(``tests/test_golden_cache.py`` is the standing pin; the first test below pins the
default itself).

What bf16 storage costs, and where it is paid:

* **The basis stops being orthonormal.** A bf16-rounded orthonormal basis has
  ``‖UᵀU - I‖_F`` ~ 3.8e-3 at the tiny model's ``n=32, r=8`` (4.9e-3 at ``n=1024,
  r=64``, 7.0e-3 at ``n=512``) -- above ``orth_fix_tol`` (1e-3) and far below
  ``orth_abort_tol`` (1e-1). The step propagates that error into the basis it returns,
  so the guard repairs with a thin QR on **every** absorb after the first. That is the
  design, not a defect: the repair is what stops U's rounding from compounding.
* **The coordinates are re-rounded at every carry** (``rot @ C`` then back to bf16), and
  nothing repairs those -- so their error compounds with the absorb count. That is the
  risk ``prereg/bf16_gist.md`` names; the drift curve is measured on a synthetic stream
  in the task report, not here (1000 absorbs is not a 90-second suite).

Everything inside one absorb -- the step, the coordinate carry, the quantized-tier
rotation, the guard and the new coordinates -- runs on fp32 working copies; bf16 exists
only at the store boundary.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import LlamaForCausalLM

from kvdlra import accounting as acc
from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer
from kvdlra.eval.config import arm_kwargs, load_arm
from kvdlra.eval.frontier import _footprint, _prefill_chunked, build_arm
from tests.conftest import N_FEATURES

T16K = 16384
RANK = 8
# 512 tokens in two chunks through the production prefill helper: 44 absorbs at
# ``prefill_block_size=8`` / ``absorb_block=16``, enough for the coordinates' re-rounding
# to compound, and the second chunk attends the reconstructed middle (so the read path's
# own upcast is exercised, not only the write path's). ``coord_budget`` holds the whole
# stream, so nothing is evicted and the two arms' coordinate tiers stay column-aligned.
T, CHUNK = 512, 256


def _cache(model: LlamaForCausalLM, **kw: Any) -> BugStreamingCache:
    base: dict[str, Any] = {
        "rank": RANK, "coord_budget": T, "recent_window": 32, "absorb_block": 16,
        "n_sink": 4, "prefill_block_size": 8, "diag_every": 1,
    }  # fmt: skip
    return BugStreamingCache(model, **{**base, **kw})


@torch.no_grad()
def _run(model: LlamaForCausalLM, **kw: Any) -> BugStreamingCache:
    """Drive a cache over a fixed id stream, exactly as ``test_golden_cache._run`` does."""
    cache = _cache(model, **kw)
    ids = torch.randint(0, 256, (1, T), generator=torch.Generator().manual_seed(1))
    with cache.attach(model):
        _prefill_chunked(model, cache, ids, CHUNK)
    return cache


@pytest.fixture(scope="module")
def fp32_cache(tiny_model: LlamaForCausalLM) -> BugStreamingCache:
    return _run(tiny_model)


@pytest.fixture(scope="module")
def bf16_cache(tiny_model: LlamaForCausalLM) -> BugStreamingCache:
    return _run(tiny_model, gist_dtype=torch.bfloat16)


def _layer(cache: BugStreamingCache) -> BugStreamingLayer:
    return cache._bug_layers()[0]


# ------------------------------------------------------------------ the default


def test_default_gist_dtype_is_fp32_and_bit_identical(
    tiny_model: LlamaForCausalLM, fp32_cache: BugStreamingCache
) -> None:
    """The default stores fp32 and is bit-for-bit what ``gist_dtype=torch.float32`` gives
    -- so the knob's presence cannot move an existing arm (the r64 golden is the other
    half of that pin)."""
    default, explicit = _layer(fp32_cache), _layer(_run(tiny_model, gist_dtype=torch.float32))
    assert default.gist_dtype == torch.float32
    for name in ("u_k", "b_k", "c_k", "u_v", "b_v", "c_v"):
        a, b = getattr(default, name), getattr(explicit, name)
        assert a is not None and a.dtype == torch.float32, name
        assert torch.equal(a, b), name


def test_an_unsupported_gist_dtype_is_refused(tiny_model: LlamaForCausalLM) -> None:
    """Only the two dtypes the accounting can bill (32 bits, 16 bits) are accepted: fp16
    would be stored at 16 bits and billed at 32, a silent mis-bill."""
    for bad in ("float16", torch.float16, "bf16"):
        with pytest.raises(ValueError, match="gist_dtype"):
            _cache(tiny_model, gist_dtype=bad)


# -------------------------------------------------------------- bf16 storage


def test_bf16_storage_halves_the_gist_bits_and_keeps_the_reconstruction_close(
    fp32_cache: BugStreamingCache, bf16_cache: BugStreamingCache
) -> None:
    """The three tensors are stored bf16, the same number of them (the element counts --
    ``float_equiv`` / ``stored_state_numel`` -- do not move), the stored bill drops by
    exactly 16 bits per U and C element, and the reconstruction the stored state yields
    is within the drift budget of the fp32 arm's over 44 absorbs."""
    l32, l16 = _layer(fp32_cache), _layer(bf16_cache)
    assert l16.u_k is not None and l16.c_k is not None and l16.b_k is not None
    assert l32.u_k is not None and l32.c_k is not None
    assert {t.dtype for t in (l16.u_k, l16.b_k, l16.c_k)} == {torch.bfloat16}
    assert l16.u_k.shape == l32.u_k.shape == (N_FEATURES, RANK)
    assert l16._f_len() == l32._f_len() > 0

    k32 = l32.u_k @ l32.c_k
    k16 = l16.u_k.float() @ l16.c_k.float()  # what attention reconstructs from the store
    rel = float(torch.linalg.norm(k32 - k16) / torch.linalg.norm(k32))
    # The bf16 drift budget on this stream: measured 1.4e-2 over these 44 absorbs. It
    # grows with the absorb count (the coordinates are re-rounded at every carry) --
    # 1.0e-2 / 2.3e-2 / 5.0e-2 at 40 / 200 / 1000 absorbs on the report's n=1024 stream.
    assert rel < 5e-2, rel

    fp32 = acc.bug_footprint(
        N_FEATURES, rank=RANK, coord_count=l32._f_len(), recent_len=l32._recent_len()
    )
    fp16 = acc.bug_footprint(
        N_FEATURES, rank=RANK, coord_count=l16._f_len(), recent_len=l16._recent_len(),
        gist_bits=16,
    )  # fmt: skip
    assert fp16.float_equiv() == fp32.float_equiv()  # element counts (the anti-drift pin)
    gist_words = 2 * N_FEATURES * RANK + 2 * RANK * l32._f_len()  # U and C, K + V
    assert fp16.stored_bits() < fp32.stored_bits()
    assert fp32.stored_bits() - fp16.stored_bits() == pytest.approx(16 * gist_words, abs=1e-6)


def test_the_frontier_bills_a_bf16_gist_at_sixteen_bits(
    fp32_cache: BugStreamingCache, bf16_cache: BugStreamingCache
) -> None:
    """The live cache's ``gist_dtype`` reaches the bill: same float-equivalents, fewer
    stored bits, and the difference is the same U+C words the formula test computes."""
    arm = {"kind": "bug", "retention": "fifo"}
    fp32 = _footprint(arm, fp32_cache, T, N_FEATURES, 2)
    fp16 = _footprint(arm, bf16_cache, T, N_FEATURES, 2)
    assert fp16.float_equiv() == fp32.float_equiv()
    gist_words = 2 * N_FEATURES * RANK + 2 * RANK * _layer(fp32_cache)._f_len()
    assert fp32.stored_bits() - fp16.stored_bits() == pytest.approx(16 * gist_words, abs=1e-6)


def test_the_guard_repairs_every_absorb_under_bf16_storage(
    fp32_cache: BugStreamingCache, bf16_cache: BugStreamingCache
) -> None:
    """The guard and the surprise scores read an fp32 upcast of the stored basis, so the
    error the diag row carries is the bf16 rounding propagated through the step -- a few
    1e-3 at this width, above ``orth_fix_tol`` and two decades below the 1e-1 abort. So
    the repair fires on every absorb but the first (which builds its basis from nothing),
    and never at the fp32 default on the same stream. ``diag_every=1`` => one row per
    absorb per layer."""
    rows16: list[dict[str, Any]] = bf16_cache.drain_diag()
    rows32: list[dict[str, Any]] = fp32_cache.drain_diag()
    assert rows16 and len(rows16) == len(rows32)
    assert all(r["orth_err_k"] < 1e-2 and r["orth_err_v"] < 1e-2 for r in rows16)
    assert all(r["fixed_k"] and r["fixed_v"] for r in rows16 if r["absorbs"] > 1)
    assert not any(r["fixed_k"] or r["fixed_v"] for r in rows32)
    assert max(r["orth_err_k"] for r in rows16) > 1e-3  # the bf16 rounding, propagated
    assert max(r["orth_err_k"] for r in rows32) < 1e-3  # ...and the fp32 arm's roundoff


@pytest.mark.parametrize(
    "tiers",
    [{}, {"coord_budget": 64, "quant_bits": 4, "quant_budget": 64}],
    ids=["exact-tier", "quant-tier"],
)
def test_every_tier_of_the_deployed_shape_runs_against_a_bf16_gist(
    tiny_model: LlamaForCausalLM, tiers: dict[str, Any]
) -> None:
    """The r64 arm's shape -- surprise retention, a surprise-selected exact tier, the
    warm-up seed -- and the same shape composed with the coded tier. Both walk code the
    fifo tests above do not: SLASH scores its candidate pool against the STORED basis
    *before* the absorb upcasts it (the one place a bf16 gist meets an fp32 block outside
    the absorb window), and the coded tier is dequantized against the stored basis when
    the middle is rebuilt."""
    cache = _run(
        tiny_model, gist_dtype=torch.bfloat16, retention="lowrank_surprise",
        hh_select="surprise", hh_budget=16, hh_neighbor=1, seed_hh_warmup=True, **tiers,
    )  # fmt: skip
    layer = _layer(cache)
    assert layer.u_k is not None and layer.c_k is not None
    assert {layer.u_k.dtype, layer.c_k.dtype} == {torch.bfloat16}
    assert layer._hh_len() == 16 and layer._f_len() > 0
    assert layer._q_len() == (64 if tiers else 0)
    layer._ensure_mid_cache()  # the read path, over every tier the arm holds
    assert layer._mid_k_cache is not None and bool(torch.isfinite(layer._mid_k_cache).all())


# ------------------------------------------------------------------- the arm


def test_the_bf16_arm_is_the_r64_arm_plus_the_knob(tiny_model: LlamaForCausalLM) -> None:
    """The Gate-1 bf16 arm must differ from ``isvd_r64_h256_seed`` in ``gist_dtype`` and
    nothing else -- otherwise its cell measures two changes. The YAML carries the dtype as
    a string (omegaconf has no torch dtypes); the constructor resolves it."""
    arm = build_arm(load_arm("isvd_r64_h256_seed_bf16"), tiny_model, T16K)
    assert arm["kwargs"]["gist_dtype"] == "bfloat16"
    ref = arm_kwargs(load_arm("isvd_r64_h256_seed"), T16K)
    assert {k: v for k, v in arm["kwargs"].items() if k != "gist_dtype"} == ref
    assert _layer(arm["make"]()).gist_dtype == torch.bfloat16
