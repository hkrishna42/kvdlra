"""Cross-method memory accounting -- the anti-drift pin.

The load-bearing invariant (``docs/week10-plan.md``): the *formula* path in
``kvdlra.accounting`` (used to count SnapKV / ShadowKV, which have no
``stored_state_numel``) must reproduce the *measured* ``stored_state_numel`` of the
streaming caches byte-for-byte, so BUG / MorphKV / SnapKV / ShadowKV all land on
one honest float-equivalent axis. If a new cache tier is added to
``stored_state_numel`` but not here, these tests fail loudly.

Hermetic: a tiny random-weight Llama (2 layers, 2 KV heads x head_dim 16 =>
n_features 32), mirroring ``tests/test_bug_cache.py``.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra import accounting as acc
from kvdlra.cache import BugStreamingCache, MorphKVCache
from kvdlra.cache.bug_cache import BugStreamingLayer
from kvdlra.cache.morph_cache import MorphKVLayer

H, D = 2, 16
N_FEATURES = H * D


def _tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=H,
        head_dim=D,
        max_position_embeddings=4096,
    )


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    model = LlamaForCausalLM(_tiny_config())  # type: ignore[no-untyped-call]
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _drive(model: LlamaForCausalLM, cache: object, t: int = 200, n_new: int = 20) -> None:
    """Prefill ``t`` tokens then ``n_new`` one-token decode steps at true positions
    (drives the cache to a saturated steady state) under the score-hook attach."""
    g = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, 256, (1, t), generator=g)
    with torch.no_grad(), cache.attach(model):  # type: ignore[attr-defined]
        out = model(prompt, past_key_values=cache, use_cache=True)
        pos = t
        nxt = int(out.logits[0, -1].argmax())
        for _ in range(n_new):
            cur = torch.tensor([[nxt]])
            position_ids = torch.arange(pos, pos + 1).unsqueeze(0)
            out = model(cur, past_key_values=cache, use_cache=True, position_ids=position_ids)
            pos += 1
            nxt = int(out.logits[0, -1].argmax())


def _bug_layer(cache: BugStreamingCache) -> BugStreamingLayer:
    return next(layer for layer in cache.layers if isinstance(layer, BugStreamingLayer))


# ---------------------------------------------------- BUG anti-drift pin


@pytest.mark.parametrize(
    "rank,retention,hh_budget,hh_select,hh_retain",
    [
        (8, "fifo", 0, "attn", True),
        (8, "attn", 0, "attn", True),
        (8, "attn", 6, "attn", True),
        # Week-11 SurpriseSLASH: surprise-selected exact tier (no hh_score) on a
        # lowrank_surprise tail -- the tier the retrieval arms deploy.
        (8, "lowrank_surprise", 6, "surprise", True),
        # Week-11 balanced-config shape: high rank (n/2) + exact tier bigger than
        # the coord tier, mirroring bugS-r128-h1024 (rank >> typical, hh dominant).
        (16, "lowrank_surprise", 12, "surprise", True),
        # Week-12 select-and-discard ablation (bugSdrop): the pool is invisible
        # to attention but is LIVE stored state (feeds re-selection/demotion), so
        # the SAME formula must hold -- the discard arm is not a cheaper point.
        (8, "lowrank_surprise", 6, "surprise", False),
    ],
)
def test_bug_footprint_matches_stored_state_numel(
    tiny_model: LlamaForCausalLM,
    rank: int,
    retention: str,
    hh_budget: int,
    hh_select: str,
    hh_retain: bool,
) -> None:
    """``bug_footprint`` computed from the live layer's actual counts equals the
    measured ``stored_state_numel`` -- the pin across the frontier configs."""
    cache = BugStreamingCache(
        tiny_model,
        rank=rank,
        coord_budget=24,
        recent_window=8,
        absorb_block=4,
        n_sink=4,
        retention=retention,
        hh_budget=hh_budget,
        hh_select=hh_select,
        hh_retain=hh_retain,
    )
    _drive(tiny_model, cache)
    layer = _bug_layer(cache)
    coord_count = layer._f_len() + layer._q_len()
    fp = acc.bug_footprint(
        N_FEATURES,
        rank=rank,
        coord_count=coord_count,
        recent_len=layer._recent_len(),
        n_sink=4,
        retention=retention,
        hh_count=layer._hh_len(),
        hh_select=hh_select,
        u_present=layer.u_k is not None,
    )
    assert fp.float_equiv() == layer.stored_state_numel()


def test_balanced_config_ratio_pin() -> None:
    """The Week-11 'balanced' operating point bugS-r128-h1024 at 32K on 8B
    (n=1024) computes to ~0.15x by the accounting formula -- the memory-cost
    claim attached to the config in docs/week11*. Arm-style counts as built by
    ``w10_frontier.build_arms`` (coord tier = everything not verbatim)."""
    n, t, rank, hh = 1024, 32768, 128, 1024
    rw, sink = 32, 4
    coord = t - hh - rw - sink
    fp = acc.bug_footprint(
        n,
        rank=rank,
        coord_count=coord,
        recent_len=rw,
        n_sink=sink,
        retention="lowrank_surprise",
        hh_count=hh,
        hh_select="surprise",
    )
    ratio = fp.ratio_fp16(t, n)
    assert 0.12 < ratio < 0.20  # "~0.15x": rank is the ppl lever, paid in memory


def test_r192_h1024_ratio_pin() -> None:
    """Week-12 sweet-spot probe bugS-r192-h1024 at 32K on 8B: the formula puts it
    at ~0.22x -- between r128 (~0.16x) and r256 (~0.28x), as the memory cost the
    r192 RULER/ppl point is bought at. Anti-drift for the new rank knob."""
    n, t, rank, hh = 1024, 32768, 192, 1024
    rw, sink = 32, 4
    coord = t - hh - rw - sink
    fp = acc.bug_footprint(
        n,
        rank=rank,
        coord_count=coord,
        recent_len=rw,
        n_sink=sink,
        retention="lowrank_surprise",
        hh_count=hh,
        hh_select="surprise",
    )
    ratio = fp.ratio_fp16(t, n)
    assert 0.19 < ratio < 0.25  # computed 0.2216 at authoring time


def test_bug_footprint_saturated_matches_bug_budget_floats() -> None:
    """The high-water helper reproduces ``w5_streamppl.bug_budget_floats`` (fifo)."""
    from w5_streamppl import bug_budget_floats  # scripts on sys.path via conftest/_paths

    for rank in (8, 16, 32):
        for w in (24, 100):
            fp = acc.bug_footprint_saturated(
                N_FEATURES, rank=rank, coord_budget=w, recent_window=8, absorb_block=4
            )
            assert fp.float_equiv() == bug_budget_floats(N_FEATURES, rank, w, 8, 4)


# ---------------------------------------------------- MorphKV anti-drift pin


def test_morph_footprint_matches_stored_state_numel(tiny_model: LlamaForCausalLM) -> None:
    cache = MorphKVCache(tiny_model, capacity=32, recent_window=8)
    _drive(tiny_model, cache)
    layer = next(la for la in cache.layers if isinstance(la, MorphKVLayer))
    assert layer.keys is not None
    kept_len = int(layer.keys.shape[2])
    fp = acc.morph_footprint(N_FEATURES, H, kept_len, recent_window=8)
    assert fp.float_equiv() == layer.stored_state_numel()


# ---------------------------------------------------- eviction / continuity


def test_evict_pure_fp16_ratio_is_keep_frac() -> None:
    for keep in (0.1, 0.25, 0.5, 0.9):
        fp = acc.evict_footprint(4096, 512, keep)
        assert fp.ratio_fp16(4096, 512) == pytest.approx(keep)


def test_bug_prefill_ratio_matches_kv_memory_ratio() -> None:
    """Continuity: the prefill helper reproduces ``kv_memory_ratio`` (fp case) to
    the digit -- so the Phase-7 delegator refactor cannot shift Week-4 numbers."""
    from w4_hybrid_sweep import kv_memory_ratio

    for t in (1024, 4096):
        for rank in (32, 64, 128):
            fp = acc.bug_prefill_footprint(t, 512, rank)
            assert fp.ratio_fp16(t, 512) == pytest.approx(kv_memory_ratio(t, 512, rank, None))


# ---------------------------------------------------- ShadowKV honest split


def test_shadow_cpu_offload_is_counted_and_half() -> None:
    """The offloaded value cache is counted in the total (never hidden) and is
    exactly half the full fp16 K+V cache -- so a zeroed CPU term fails loudly."""
    t, n, h_kv, head_dim = 2048, 512, 8, 64
    fp = acc.shadow_footprint(t, n, h_kv, head_dim, rank_s=64)
    assert fp.cpu_ratio_fp16(t, n) == pytest.approx(0.5)
    assert fp.cpu_verbatim_elems == t * n
    # total = GPU-resident + CPU-offloaded; GPU-only is strictly less than total.
    assert fp.gpu_ratio_fp16(t, n) < fp.ratio_fp16(t, n)
    assert fp.ratio_fp16(t, n) == pytest.approx(fp.gpu_ratio_fp16(t, n) + fp.cpu_ratio_fp16(t, n))


def test_think_ratio_is_one_minus_half_cr() -> None:
    """ThinK prunes only KEY channels -> ratio 1 - cr/2 (K is half the cache)."""
    t, n, head_dim, h_kv = 4096, 512, 64, 8
    for cr in (0.3, 0.5, 0.7):
        fp = acc.think_footprint(t, n, head_dim, h_kv, cr)
        assert fp.ratio_fp16(t, n) == pytest.approx(1.0 - cr / 2, abs=2e-3)


def test_palu_ratio_tracks_rank_ratio() -> None:
    """Palu low-rank K+V latents -> ratio ~ rank_ratio at long t (basis + exact
    sinks amortize)."""
    t, n, head_dim, h_kv = 8192, 512, 64, 8
    for rr in (0.25, 0.5):
        fp = acc.palu_footprint(t, n, head_dim, h_kv, rr)
        assert fp.ratio_fp16(t, n) == pytest.approx(rr, abs=0.03)


def test_palu_footprint_counts_sinks() -> None:
    """Week-15: ``PaluPress`` keeps the ``n_sink`` leading columns exact, so the
    footprint counts them verbatim (``2*n*n_sink``, K+V at full feature width)
    and pays the per-token latent only over ``t - n_sink`` columns -- the exact
    sinks are stored, never free (the one-unit ethos)."""
    t, n, head_dim, h_kv, rr, sink = 8192, 512, 64, 8, 0.5, 4
    r = round(rr * head_dim)  # per-head rank (group=1)
    fp = acc.palu_footprint(t, n, head_dim, h_kv, rr)  # default n_sink=4
    expected = 2 * n * sink + 2 * (t - sink) * r * h_kv + 2 * r * head_dim * h_kv
    assert fp.float_equiv() == expected
    # n_sink=0 reproduces the pre-fix latent+basis-only formula ...
    fp0 = acc.palu_footprint(t, n, head_dim, h_kv, rr, n_sink=0)
    assert fp0.float_equiv() == 2 * t * r * h_kv + 2 * r * head_dim * h_kv
    # ... and the delta is exactly (verbatim sinks added) - (sink latents removed).
    assert fp.float_equiv() - fp0.float_equiv() == 2 * n * sink - 2 * sink * r * h_kv


def test_full_cache_ratio_is_one() -> None:
    fp = acc.full_cache_footprint(4096, 512)
    assert fp.ratio_fp16(4096, 512) == pytest.approx(1.0)
    assert fp.tok_equiv(512) == pytest.approx(4096.0)


# ---------------------------------------------------- matched-memory gate


def test_assert_all_within_gate() -> None:
    acc.assert_all_within({"a": 100.0, "b": 200.0}, budget_per_layer=200.0)
    with pytest.raises(ValueError, match="matched-memory audit FAILED"):
        acc.assert_all_within({"a": 100.0, "b": 201.0}, budget_per_layer=200.0)


# ---------------------------------------------- Week-18 dual memory billing


def test_ratio_stored_bits_equals_fp16_for_baselines() -> None:
    """Every method with no fp32-at-rest state (ThinK/Palu/eviction/full) bills the
    same honest ratio as ratio_fp16: fp32_verbatim_elems is 0, so the two coincide.
    This is what makes the dual-billing safe to report for the baselines."""
    t, n, head_dim, h_kv = 16384, 1024, 128, 8
    fps = {
        "think": acc.think_footprint(t, n, head_dim, h_kv, key_channel_ratio=0.5),
        "palu": acc.palu_footprint(t, n, head_dim, h_kv, rank_ratio=0.5),
        "evict": acc.evict_footprint(t, n, keep_frac=0.1),
        "full": acc.full_cache_footprint(t, n),
    }
    for name, fp in fps.items():
        assert fp.fp32_verbatim_elems == 0.0, name
        assert fp.ratio_stored_bits(t, n) == pytest.approx(fp.ratio_fp16(t, n)), name


def test_bug_ratio_stored_bits_exceeds_fp16_honest_band() -> None:
    """BUG's basis U and coordinates C are fp32 at rest, so the honest stored-bits
    ratio is strictly above the fp16-equivalent headline. Pins the review's numbers:
    r64-h256 @16K is 0.085x (fp16) / 0.150x (honest) on Llama-8B (n=1024) and
    0.148x / 0.275x on Qwen-7B (n=512). ratio_fp16 itself is unchanged."""
    t, rank, hh, rw, sink = 16384, 64, 256, 32, 4
    coord = t - hh - rw - sink
    for n, fp16_exp, stored_exp in ((1024, 0.085, 0.150), (512, 0.148, 0.275)):
        fp = acc.bug_footprint(
            n,
            rank=rank,
            coord_count=coord,
            recent_len=rw,
            n_sink=sink,
            retention="lowrank_surprise",
            hh_count=hh,
            hh_select="surprise",
        )
        assert fp.ratio_fp16(t, n) == pytest.approx(fp16_exp, abs=2e-3)
        assert fp.ratio_stored_bits(t, n) == pytest.approx(stored_exp, abs=2e-3)
        assert fp.ratio_stored_bits(t, n) > fp.ratio_fp16(t, n)


def test_stored_bits_matches_manual_split() -> None:
    """stored_bits() = (verbatim - fp32_verbatim)*16 + fp32_verbatim*32 + codes +
    aux*32. A frozen algebra pin so the honest formula can't silently drift."""
    fp = acc.Footprint(
        verbatim_elems=1000.0, quant_code_bits=640.0, aux_words=50.0, fp32_verbatim_elems=300.0
    )
    expected = (1000.0 - 300.0) * 16 + 300.0 * 32 + 640.0 + 50.0 * 32
    assert fp.stored_bits() == pytest.approx(expected)
    # bits(16) (the fp16-equivalent) is unchanged by the new field.
    assert fp.bits(16) == 1000.0 * 16 + 640.0 + 50.0 * 32


# ---------------------------------------------- Week-18 quantized-KV baseline


def test_quant_footprint_asymptotic_ratios() -> None:
    """KIVI-style QuantizedCache baseline sits just above the pure code-bit asymptote
    (the fp32 scale+shift at group-64 and the fp16 residual window add overhead):
    2-bit/g64 -> ~0.19x, 4-bit -> ~0.31x. The residual is a vanishing fraction as
    t grows, so a huge context recovers the (nbits + 2*32/64)/16 asymptote."""
    n = 1024
    for t in (16384, 65536):
        fp2 = acc.quant_footprint(t, n, nbits=2, group=64, residual_length=128)
        fp4 = acc.quant_footprint(t, n, nbits=4, group=64, residual_length=128)
        assert 0.185 < fp2.ratio_fp16(t, n) < 0.200  # near the KIVI 2-bit band top edge
        assert 0.310 < fp4.ratio_fp16(t, n) < 0.320
    huge = 4_000_000
    r2 = acc.quant_footprint(huge, n, nbits=2).ratio_fp16(huge, n)
    r4 = acc.quant_footprint(huge, n, nbits=4).ratio_fp16(huge, n)
    assert r2 == pytest.approx(0.1875, abs=2e-3)
    assert r4 == pytest.approx(0.3125, abs=2e-3)


def test_quant_footprint_stored_equals_fp16() -> None:
    """The quant baseline has no fp32-at-rest state (residual is model dtype), so its
    honest ratio_stored_bits equals ratio_fp16 -- billed on the same footing as
    ThinK/Palu, unlike BUG."""
    t, n = 16384, 1024
    fp = acc.quant_footprint(t, n, nbits=2)
    assert fp.fp32_verbatim_elems == 0.0
    assert fp.ratio_stored_bits(t, n) == pytest.approx(fp.ratio_fp16(t, n))


def test_quant_footprint_components_match_quanto() -> None:
    """Pin the component algebra against the Week-18 quanto probe: uint8 codes at nbits,
    fp32 scale+shift (2 aux words) per group, no zeropoint; residual verbatim fp16."""
    t, n, nbits, group, resid = 1000, 512, 2, 64, 128
    fp = acc.quant_footprint(t, n, nbits=nbits, group=group, residual_length=resid)
    payload = t - resid
    assert fp.verbatim_elems == 2 * resid * n
    assert fp.quant_code_bits == 2 * payload * n * nbits
    import math as _m

    assert fp.aux_words == 2 * _m.ceil(payload * n / group) * 2


def test_bug_quant_footprint_matches_stored_state_numel(tiny_model: LlamaForCausalLM) -> None:
    """Week-18: with a coordinate-quant tier the coded columns must be billed at nbits,
    not as fp32 coords. Drive a small-budget quant config so the middle overflows into
    the quantized tier, then pin the CORRECTED split (coord_count=_f_len(),
    quant_count=_q_len(), quant_bits) against the measured stored_state_numel."""
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=16,
        recent_window=8,
        absorb_block=4,
        n_sink=4,
        quant_bits=4,
        quant_budget=32,  # small -> the prefill middle demotes into the quant tier
    )
    _drive(tiny_model, cache)
    layer = _bug_layer(cache)
    assert layer._q_len() > 0, "test did not exercise the quant tier"
    fp = acc.bug_footprint(
        N_FEATURES,
        rank=8,
        coord_count=layer._f_len(),  # fp coords only
        recent_len=layer._recent_len(),
        n_sink=4,
        hh_count=layer._hh_len(),
        u_present=layer.u_k is not None,
        quant_count=layer._q_len(),
        quant_bits=layer.quant_bits,
    )
    assert fp.float_equiv() == layer.stored_state_numel()
    # And the old lumped billing (coords + quant as fp32) OVER-counts -> the bug we fixed.
    lumped = acc.bug_footprint(
        N_FEATURES,
        rank=8,
        coord_count=layer._f_len() + layer._q_len(),
        recent_len=layer._recent_len(),
        n_sink=4,
        hh_count=layer._hh_len(),
        u_present=layer.u_k is not None,
    )
    assert lumped.float_equiv() > fp.float_equiv()
