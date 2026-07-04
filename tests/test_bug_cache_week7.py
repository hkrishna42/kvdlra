"""Week-7 tests: adaptive coordinate retention (A) + quantized age tier (D).

Extends the :mod:`tests.test_bug_cache` correctness ladder to the Week-7 knobs
(``docs/week7-plan.md`` tier 1):

* retention selection actually keeps the highest-scored / highest-energy
  columns and preserves chronological order among survivors;
* non-contiguous middles are re-rotated at their **true** tracked positions;
* ``retention="attn"`` without any observations reduces to FIFO **bitwise**
  (the stable-sort tiebreak) -- and the attach() hook records real scores;
* the quantized tier bounds reconstruction error, keeps memory constant, and
  is **counted honestly** (codes at ``bits/32`` float-equivalents + norms +
  positions + scores + shared side info);
* exact-mode parity with ``DynamicCache`` still holds for every variant when
  nothing overflows;
* the mask-size == returned-length invariant holds for every variant.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer, _QuantBank, _RopeAngles

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
        max_position_embeddings=2048,
    )


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    model = LlamaForCausalLM(_tiny_config())  # type: ignore[no-untyped-call]
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _prompt(t: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, t), generator=g)


def _rope(model: LlamaForCausalLM) -> _RopeAngles:
    return _RopeAngles(cast(nn.Module, model.model.rotary_emb))


def _kv(t: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(1, H, t, D, generator=g),
        torch.randn(1, H, t, D, generator=g),
    )


# --------------------------------------------------------------------------
# Constructor validation
# --------------------------------------------------------------------------


def test_week7_constructor_validation(tiny_model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError, match="retention"):
        BugStreamingCache(tiny_model, rank=8, coord_budget=16, retention="lru")
    with pytest.raises(ValueError, match="score_decay"):
        BugStreamingCache(tiny_model, rank=8, coord_budget=16, score_decay=0.0)
    with pytest.raises(ValueError, match="set together"):
        BugStreamingCache(tiny_model, rank=8, coord_budget=16, quant_bits=4)
    with pytest.raises(ValueError, match="set together"):
        BugStreamingCache(tiny_model, rank=8, coord_budget=16, quant_budget=32)
    with pytest.raises(ValueError, match=r"\[1, 8\]"):
        BugStreamingCache(tiny_model, rank=8, coord_budget=16, quant_bits=16, quant_budget=32)
    with pytest.raises(ValueError, match="low-rank middle"):
        BugStreamingCache(tiny_model, rank=0, coord_budget=0, quant_bits=4, quant_budget=32)


# --------------------------------------------------------------------------
# Retention selection (A)
# --------------------------------------------------------------------------


def test_attn_retention_keeps_top_scored_columns(tiny_model: LlamaForCausalLM) -> None:
    # Drive the layer directly; set scores by hand (the attach() hook's job) and
    # check the eviction at the next absorb keeps exactly the top-scored columns
    # while preserving chronological order among survivors.
    layer = BugStreamingLayer(
        rope=_rope(tiny_model),
        rank=8,
        coord_budget=12,
        recent_window=4,
        absorb_block=4,
        n_sink=2,
        retention="attn",
    )
    g = torch.Generator().manual_seed(0)
    layer.update(*_kv(18, g))  # prefill: mid = 18 - 2 - 4 = 12 columns, exactly at cap
    assert layer._f_len() == 12
    assert layer.mid_pos is not None
    assert torch.equal(layer.mid_pos, torch.arange(2, 14))

    # Hand-set scores: columns at mid_pos 4,5,6,7 lowest => they must be evicted
    # when the next absorb pushes 4 new columns in.
    score = torch.full((12,), 10.0)
    score[2:6] = 0.5  # positions 4..7
    layer.mid_score = score.clone()
    layer.ring_score = torch.full((4,), 7.0)  # graduating tokens carry this score
    layer._seen_observation = True

    for _ in range(4):  # ring 4 -> 8 => absorb of 4
        layer.update(*_kv(1, g))
    assert layer._f_len() == 12
    kept = layer.mid_pos.tolist()
    assert kept == [2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # 4..7 gone, order kept
    # The graduated block carried its ring scores (7.0), not zeros.
    assert layer.mid_score is not None
    assert torch.allclose(layer.mid_score[-4:], torch.full((4,), 7.0))


def test_energy_retention_keeps_high_norm_columns(tiny_model: LlamaForCausalLM) -> None:
    # Feed a prefill whose middle has two blocks of hugely different scale; on
    # overflow the low-energy columns must be the ones dropped, regardless of age.
    layer = BugStreamingLayer(
        rope=_rope(tiny_model),
        rank=8,
        coord_budget=8,
        recent_window=4,
        absorb_block=4,
        n_sink=2,
        retention="energy",
    )
    g = torch.Generator().manual_seed(1)
    k, v = _kv(22, g)
    k[:, :, 2:10, :] *= 100.0  # OLD middle tokens (positions 2..9) are high-energy
    layer.update(k, v)
    # mid = 22 - 2 - 4 = 16 -> overflow 8; the 8 kept must be the loud old ones.
    assert layer._f_len() == 8
    assert layer.mid_pos is not None
    assert layer.mid_pos.tolist() == list(range(2, 10))


def test_attn_without_observations_is_bitwise_fifo(tiny_model: LlamaForCausalLM) -> None:
    # No attach() => all scores zero => stable-sort tiebreak == FIFO. The whole
    # teacher-forced logit trajectory must match the fifo cache bitwise (this
    # also pins the gather-based re-rotation path against the contiguous one).
    tiny_model.config._attn_implementation = "sdpa"
    stream = _prompt(70, seed=9)
    outs = []
    for retention in ("fifo", "attn"):
        cache = BugStreamingCache(
            tiny_model,
            rank=8,
            coord_budget=16,
            recent_window=8,
            absorb_block=4,
            retention=retention,
        )
        logits = []
        with torch.no_grad():
            out = tiny_model(stream[:, :30], past_key_values=cache, use_cache=True)
            for t in range(30, 70):
                out = tiny_model(stream[:, t : t + 1], past_key_values=cache, use_cache=True)
                logits.append(out.logits)
        outs.append(torch.cat(logits))
    assert torch.equal(outs[0], outs[1])


def test_noncontiguous_middle_rerotated_at_true_positions(
    tiny_model: LlamaForCausalLM,
) -> None:
    # After adaptive eviction the middle is non-contiguous; every reconstructed
    # key column must equal RoPE(U c_j) evaluated at that column's TRUE position.
    layer = BugStreamingLayer(
        rope=_rope(tiny_model),
        rank=8,
        coord_budget=8,
        recent_window=4,
        absorb_block=4,
        n_sink=2,
        retention="energy",
    )
    g = torch.Generator().manual_seed(2)
    k, v = _kv(22, g)
    k[:, :, 2:6, :] *= 50.0
    k[:, :, 10:14, :] *= 50.0  # two loud islands -> non-contiguous survivors
    layer.update(k, v)
    for _ in range(6):
        layer.update(*_kv(1, g))
    assert layer.mid_pos is not None
    pos = layer.mid_pos.tolist()
    assert pos != list(range(pos[0], pos[0] + len(pos)))  # really non-contiguous
    layer._ensure_mid_cache()
    assert layer._mid_k_cache is not None
    assert layer.u_k is not None and layer.c_k is not None
    for j in range(layer._f_len()):
        col_pre = (layer.u_k @ layer.c_k[:, j : j + 1]).reshape(N_FEATURES, 1)
        expect = layer._mat_rope(col_pre, int(pos[j]), inverse=False)
        got = layer._mid_k_cache[:, j : j + 1].to(torch.float32)
        assert torch.allclose(got, expect, atol=1e-5)


# --------------------------------------------------------------------------
# Quantized age tier (D)
# --------------------------------------------------------------------------


def test_quant_tier_demotes_then_drops_fifo(tiny_model: LlamaForCausalLM) -> None:
    layer = BugStreamingLayer(
        rope=_rope(tiny_model),
        rank=8,
        coord_budget=8,
        recent_window=4,
        absorb_block=4,
        n_sink=2,
        quant_bits=4,
        quant_budget=8,
    )
    g = torch.Generator().manual_seed(3)
    layer.update(*_kv(14, g))  # mid = 8: fills the fp32 tier exactly
    assert layer._f_len() == 8 and layer._q_len() == 0
    for _ in range(8):  # two absorbs of 4 -> 8 demotions into the quant tier
        layer.update(*_kv(1, g))
    assert layer._f_len() == 8 and layer._q_len() == 8
    assert layer._mid_len() == 16 and layer.attended_length() == 2 + 16 + layer._recent_len()
    before = layer.qk_norm
    for _ in range(4):  # one more absorb -> quant tier overflow -> oldest dropped
        layer.update(*_kv(1, g))
    assert layer._q_len() == 8  # capped
    assert before is not None and layer.qk_norm is not None


def test_quant_roundtrip_error_bounded(tiny_model: LlamaForCausalLM) -> None:
    # A demoted column's dequantized coordinates must stay close to the fp32
    # coordinates it was quantized from (PolarQuant thm-1 distortion, dim=rank).
    bank = _QuantBank(bits=8)
    layer = BugStreamingLayer(
        rope=_rope(tiny_model),
        rank=8,
        coord_budget=4,
        recent_window=4,
        absorb_block=4,
        n_sink=2,
        quant_bits=8,
        quant_budget=8,
        quant_bank=bank,
    )
    g = torch.Generator().manual_seed(4)
    layer.update(*_kv(10, g))  # mid = 4, at cap
    assert layer.c_k is not None
    # Next absorb will (1) rotate these coords, (2) demote them. Capture what the
    # demoted values *should* be by replaying the rotation on a snapshot.
    snap_ck = layer.c_k.clone()
    demote_expected: dict[str, torch.Tensor] = {}
    orig_absorb = layer._absorb_columns

    def spy(block_k: torch.Tensor, block_v: torch.Tensor, start: int, scores: object) -> None:
        from kvdlra.integrators.streaming_torch import augmented_bug_step

        assert layer.u_k is not None and layer.b_k is not None
        _, _, rot_k = augmented_bug_step(
            layer.u_k.clone(), layer.b_k.clone(), block_k, layer.rank, theta=layer.theta
        )
        demote_expected["ck"] = rot_k @ snap_ck
        orig_absorb(block_k, block_v, start, cast(torch.Tensor | None, scores))

    layer._absorb_columns = spy  # type: ignore[method-assign]
    for _ in range(4):
        layer.update(*_kv(1, g))
    layer._absorb_columns = orig_absorb  # type: ignore[method-assign]
    assert layer._q_len() == 4
    assert layer.qk_codes is not None and layer.qk_norm is not None
    got = layer._dequantize(layer.qk_codes, layer.qk_norm)
    want = demote_expected["ck"]
    rel = (got - want).norm(dim=0) / want.norm(dim=0).clamp_min(1e-9)
    assert float(rel.max()) < 0.05  # 8-bit: sqrt(2.7)*4^-8-ish per coordinate


def test_quant_carry_norms_exact_under_identity_rotation(
    tiny_model: LlamaForCausalLM,
) -> None:
    # Adversarial-review regression (Week 7): the dequantize -> rot -> requantize
    # carry must NOT drift column norms when rot == I. (Plain PolarQuant
    # dequantize scales the raw centroid vector, whose norm != 1, so the naive
    # cycle multiplied each norm by ||centroids[codes]|| per absorb -> g^k
    # exponential drift; _dequantize now renormalizes the direction.)
    layer = BugStreamingLayer(
        rope=_rope(tiny_model),
        rank=8,
        coord_budget=4,
        recent_window=4,
        absorb_block=4,
        n_sink=2,
        quant_bits=4,
        quant_budget=16,
    )
    g = torch.Generator().manual_seed(12)
    layer.update(*_kv(10, g))
    for _ in range(8):  # push two blocks through -> populate the quant tier
        layer.update(*_kv(1, g))
    assert layer._q_len() > 0
    assert layer.qk_norm is not None
    norms0 = layer.qk_norm.clone()
    deq0 = layer._dequantize(cast(torch.Tensor, layer.qk_codes), layer.qk_norm)
    eye = torch.eye(layer.rank if layer.c_k is None else int(layer.c_k.shape[0]))
    for _ in range(50):
        layer._rotate_quant_tier(eye, eye)
    assert layer.qk_norm is not None
    # Norms exactly preserved (fp32 roundoff only), no exponential drift.
    assert torch.allclose(layer.qk_norm, norms0, rtol=1e-5, atol=1e-7)
    # Reconstruction stays within a small multiple of the one-shot distortion.
    deq50 = layer._dequantize(cast(torch.Tensor, layer.qk_codes), layer.qk_norm)
    rel = (deq50 - deq0).norm(dim=0) / deq0.norm(dim=0).clamp_min(1e-9)
    assert float(rel.max()) < 0.5  # bounded jitter, not the ~3x drift of the bug
    # And every reconstructed column's norm equals its stored norm exactly.
    assert torch.allclose(deq50.norm(dim=0), layer.qk_norm, rtol=1e-5, atol=1e-7)


def test_quant_memory_counted_honestly(tiny_model: LlamaForCausalLM) -> None:
    # stored_state_numel must equal: base fp32 tensors + ceil(codes*bits/32)
    # + fp32 norms + (positions + scores if tracked); side info counted once at
    # the cache level. Recomputed here independently from the tier shapes.
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=8,
        recent_window=4,
        absorb_block=4,
        n_sink=2,
        retention="attn",
        quant_bits=4,
        quant_budget=8,
    )
    ids = _prompt(20, seed=5)
    with torch.no_grad():
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        for _ in range(40):
            tok = out.logits[:, -1:].argmax(-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
    total_expected = 0
    for layer in cache.layers:
        assert isinstance(layer, BugStreamingLayer)
        base = sum(
            int(t.numel())
            for t in (
                layer.sink_k,
                layer.sink_v,
                layer.recent_k,
                layer.recent_v,
                layer.u_k,
                layer.b_k,
                layer.c_k,
                layer.u_v,
                layer.b_v,
                layer.c_v,
            )
            if t is not None
        )
        q, f, rlen = layer._q_len(), layer._f_len(), layer._recent_len()
        assert q == 8 and f == 8  # both tiers full after 40 steps
        r_k = int(cast(torch.Tensor, layer.qk_codes).shape[1])
        r_v = int(cast(torch.Tensor, layer.qv_codes).shape[1])
        codes = -(-(q * (r_k + r_v) * 4) // 32)  # ceil
        norms = 2 * q
        positions = f + q
        scores = f + q + rlen
        total_expected += base + codes + norms + positions + scores
    bank = cache._quant_bank
    assert bank is not None
    side = bank.side_info_numel()
    assert side > 0  # Pi + codebook really counted
    assert cache.stored_state_numel() == total_expected + side


def test_variants_report_more_memory_than_baseline_at_same_config(
    tiny_model: LlamaForCausalLM,
) -> None:
    # Honesty canary: at identical (rank, coord_budget), the attn variant must
    # report MORE stored floats than fifo (positions + scores are not free).
    stored = {}
    for retention in ("fifo", "attn"):
        cache = BugStreamingCache(
            tiny_model, rank=8, coord_budget=16, recent_window=4, retention=retention
        )
        ids = _prompt(40, seed=6)
        with torch.no_grad():
            out = tiny_model(ids, past_key_values=cache, use_cache=True)
            for _ in range(20):
                tok = out.logits[:, -1:].argmax(-1)
                out = tiny_model(tok, past_key_values=cache, use_cache=True)
        stored[retention] = cache.stored_state_numel()
    assert stored["attn"] > stored["fifo"]


def test_memory_constant_with_quant_and_retention(tiny_model: LlamaForCausalLM) -> None:
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=16,
        recent_window=8,
        absorb_block=4,
        retention="energy",
        quant_bits=4,
        quant_budget=16,
    )
    ids = _prompt(40, seed=7)
    mems = []
    with torch.no_grad():
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        for _ in range(200):
            tok = out.logits[:, -1:].argmax(-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
            mems.append(cache.stored_state_numel())
    assert max(mems[80:]) <= max(mems[:80])  # bound reached early, never exceeded
    layer = cache.layers[0]
    assert isinstance(layer, BugStreamingLayer)
    assert layer.cumulative_length == 240
    assert layer._q_len() == 16 and layer._f_len() == 16


# --------------------------------------------------------------------------
# End-to-end: parity, masks, attach() scoring
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retention": "attn"},
        {"retention": "energy"},
        {"quant_bits": 4, "quant_budget": 4096},
    ],
    ids=["attn", "energy", "quant"],
)
def test_exact_mode_parity_all_variants(
    tiny_model: LlamaForCausalLM, kwargs: dict[str, object]
) -> None:
    # Generous budgets => nothing overflows => no variant machinery ever fires
    # destructively; logits must match DynamicCache to fp tolerance.
    tiny_model.config._attn_implementation = "sdpa"
    cache = BugStreamingCache(
        tiny_model,
        rank=N_FEATURES,
        coord_budget=4096,
        recent_window=8,
        absorb_block=4,
        **cast(dict[str, Any], kwargs),
    )
    ref = DynamicCache()
    stream = _prompt(60, seed=8)
    with torch.no_grad():
        out_a = tiny_model(stream[:, :25], past_key_values=cache, use_cache=True)
        out_b = tiny_model(stream[:, :25], past_key_values=ref, use_cache=True)
        assert torch.equal(out_a.logits, out_b.logits)
        for t in range(25, 60):
            tok = stream[:, t : t + 1]
            out_a = tiny_model(tok, past_key_values=cache, use_cache=True)
            out_b = tiny_model(tok, past_key_values=ref, use_cache=True)
            assert torch.allclose(out_a.logits, out_b.logits, atol=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retention": "attn"},
        {"retention": "energy"},
        {"quant_bits": 4, "quant_budget": 8},
        {"retention": "attn", "quant_bits": 4, "quant_budget": 8},
    ],
    ids=["attn", "energy", "quant", "attn+quant"],
)
def test_mask_sizes_consistent_all_variants(
    tiny_model: LlamaForCausalLM, kwargs: dict[str, object]
) -> None:
    tiny_model.config._attn_implementation = "sdpa"
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=12,
        recent_window=8,
        absorb_block=4,
        **cast(dict[str, Any], kwargs),
    )
    layer = cache.layers[0]
    assert isinstance(layer, BugStreamingLayer)
    ids = _prompt(40, seed=10)
    with torch.no_grad(), cache.attach(tiny_model):
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        for _ in range(70):
            kv_length, kv_offset = layer.get_mask_sizes(1)
            tok = out.logits[:, -1:].argmax(-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
            assert kv_length == layer.attended_length()
            assert kv_offset + kv_length == layer.cumulative_length


def test_attach_hook_records_and_seeds_scores(tiny_model: LlamaForCausalLM) -> None:
    tiny_model.config._attn_implementation = "sdpa"
    cache = BugStreamingCache(
        tiny_model, rank=8, coord_budget=12, recent_window=8, absorb_block=4, retention="attn"
    )
    ids = _prompt(40, seed=11)
    with torch.no_grad(), cache.attach(tiny_model):
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        layer = cache.layers[0]
        assert isinstance(layer, BugStreamingLayer)
        # Pre-fill seeding happened: scores exist and are not all zero.
        assert layer._seen_observation
        assert layer.mid_score is not None and float(layer.mid_score.sum()) > 0
        assert layer.ring_score is not None and float(layer.ring_score.sum()) > 0
        seeded = layer.mid_score.clone()
        for _ in range(12):
            tok = out.logits[:, -1:].argmax(-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
        # Decode observations kept flowing (scores moved from the seed).
        assert layer.mid_score is not None
        assert not torch.equal(layer.mid_score[: min(8, seeded.shape[0])], seeded[:8])
    # Off-cache traffic is ignored: a different cache forward must not observe.
    other = BugStreamingCache(
        tiny_model, rank=8, coord_budget=12, recent_window=8, absorb_block=4, retention="attn"
    )
    with torch.no_grad(), cache.attach(tiny_model):
        tiny_model(ids, past_key_values=other, use_cache=True)
    other_layer = other.layers[0]
    assert isinstance(other_layer, BugStreamingLayer)
    assert not other_layer._seen_observation
