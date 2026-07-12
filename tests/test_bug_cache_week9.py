"""Tests for the Week-9 D3 ``retention="lowrank_surprise"`` mode.

Low-rank surprise keeps the highest-residual (outlier) coordinate columns and
evicts the ones the basis already reconstructs (redundant). The residual is the
out-of-subspace half of ``||k||^2`` (Pythagoras; ``u_k`` orthonormal), so it is
the orthogonal complement of ``retention="energy"`` (the in-subspace half). It is
a *stored snapshot* at graduation -- not recomputable later (``U C`` has zero
residual) -- so it costs one fp32 scalar per column, exactly like ``mid_score``.

The correctness ladder mirrors ``test_bug_cache_week7.py``:
1. mode is valid; incompatible with SLASH (``hh_budget``);
2. eviction keeps the highest-surprise columns (direct, hand-set scores);
3. computed surprise is normalized in ``[0, 1]`` and populated on a real stream;
4. with no eviction, the trajectory is **bitwise** identical to FIFO;
5. memory is counted and equals the ``attn`` per-column cost;
6. mask sizes stay consistent every decode step;
7. the correlation probe records (score, surprise, energy) triples.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer, _RopeAngles

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
# Mode validity + SLASH incompatibility
# --------------------------------------------------------------------------


def test_surprise_mode_valid_and_slash_incompatible(tiny_model: LlamaForCausalLM) -> None:
    # A plain surprise cache constructs fine.
    BugStreamingCache(tiny_model, rank=8, coord_budget=16, retention="lowrank_surprise")
    # SLASH exact heavy-hitters require retention="attn"; surprise is disallowed.
    with pytest.raises(ValueError, match="requires retention='attn'"):
        BugStreamingCache(
            tiny_model,
            rank=8,
            coord_budget=16,
            retention="lowrank_surprise",
            hh_budget=4,
        )
    # merge fuses columns without a single graduation residual -> incompatible.
    with pytest.raises(ValueError, match="mutually exclusive"):
        BugStreamingCache(
            tiny_model,
            rank=8,
            coord_budget=16,
            retention="lowrank_surprise",
            merge=True,
        )


# --------------------------------------------------------------------------
# Eviction keeps the highest-surprise columns
# --------------------------------------------------------------------------


def test_surprise_retention_keeps_high_residual_columns(tiny_model: LlamaForCausalLM) -> None:
    # Mirror the attn hand-set test: force four old columns to the lowest surprise
    # and confirm the next absorb evicts exactly those (the most-redundant), while
    # keeping every high-surprise survivor in chronological order.
    layer = BugStreamingLayer(
        rope=_rope(tiny_model),
        rank=8,
        coord_budget=12,
        recent_window=4,
        absorb_block=4,
        n_sink=2,
        retention="lowrank_surprise",
    )
    g = torch.Generator().manual_seed(0)
    layer.update(*_kv(18, g))  # prefill: mid = 18 - 2 - 4 = 12 columns, at cap
    assert layer._f_len() == 12
    assert layer.mid_pos is not None
    assert torch.equal(layer.mid_pos, torch.arange(2, 14))
    assert layer.mid_surprise is not None and layer.mid_surprise.shape == (12,)

    # Hand-set surprise: positions 4..7 lowest (0.0), everything else above the
    # normalized [0,1] range so the four zeros are strictly the lowest four even
    # after the four graduating columns (computed surprise in [0,1]) arrive.
    surp = torch.full((12,), 2.0)
    surp[2:6] = 0.0  # mid_pos 4..7
    layer.mid_surprise = surp.clone()

    for _ in range(4):  # ring 4 -> 8 => one absorb of 4
        layer.update(*_kv(1, g))
    assert layer._f_len() == 12
    kept = layer.mid_pos.tolist()
    assert kept == [2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # 4..7 gone, order kept


def test_surprise_computed_normalized_and_populated(tiny_model: LlamaForCausalLM) -> None:
    # On a real streamed run the computed surprise must be populated, one scalar
    # per fp32 column, all in [0, 1] (it is a sine of an angle).
    layer = BugStreamingLayer(
        rope=_rope(tiny_model),
        rank=8,
        coord_budget=16,
        recent_window=8,
        absorb_block=4,
        n_sink=2,
        retention="lowrank_surprise",
    )
    g = torch.Generator().manual_seed(3)
    layer.update(*_kv(24, g))
    for _ in range(30):
        layer.update(*_kv(1, g))
    assert layer.mid_surprise is not None
    assert layer.mid_surprise.shape == (layer._f_len(),)
    assert torch.all(layer.mid_surprise >= 0.0)
    assert torch.all(layer.mid_surprise <= 1.0 + 1e-5)


# --------------------------------------------------------------------------
# No-eviction => bitwise FIFO
# --------------------------------------------------------------------------


def test_surprise_without_evict_is_bitwise_fifo(tiny_model: LlamaForCausalLM) -> None:
    # coord_budget large enough that no eviction ever fires => the surprise sort
    # key is never used => the whole teacher-forced logit trajectory must match
    # the fifo cache bitwise.
    tiny_model.config._attn_implementation = "sdpa"
    stream = _prompt(70, seed=9)
    outs = []
    for retention in ("fifo", "lowrank_surprise"):
        cache = BugStreamingCache(
            tiny_model,
            rank=8,
            coord_budget=4096,
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


# --------------------------------------------------------------------------
# Honest memory: surprise costs one fp32/column, == attn
# --------------------------------------------------------------------------


def test_surprise_memory_counted_between_fifo_and_attn(tiny_model: LlamaForCausalLM) -> None:
    # Per column the surprise cache costs the SAME as attn (mid_pos + one fp32
    # scalar = 2r+2), so both exceed positionless fifo. But surprise needs no
    # ``ring_score`` high-water buffer (it reads its signal off the graduating K
    # columns, not an EMA over the recent ring), so its *total* is strictly below
    # attn by exactly that buffer -- an honest, tiny edge, not a cost.
    stream = _prompt(80, seed=5)

    def peak(retention: str) -> int:
        cache = BugStreamingCache(
            tiny_model,
            rank=8,
            coord_budget=16,
            recent_window=8,
            absorb_block=4,
            retention=retention,
        )
        mems = []
        with torch.no_grad():
            tiny_model(stream[:, :30], past_key_values=cache, use_cache=True)
            for t in range(30, 80):
                tiny_model(stream[:, t : t + 1], past_key_values=cache, use_cache=True)
                mems.append(cache.stored_state_numel())
        return max(mems)

    fifo, attn, surprise = peak("fifo"), peak("attn"), peak("lowrank_surprise")
    assert fifo < surprise < attn  # positions + surprise scalar counted; no ring buffer


# --------------------------------------------------------------------------
# Mask consistency
# --------------------------------------------------------------------------


def test_surprise_mask_consistency(tiny_model: LlamaForCausalLM) -> None:
    tiny_model.config._attn_implementation = "sdpa"
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=12,
        recent_window=8,
        absorb_block=4,
        retention="lowrank_surprise",
    )
    layer = cache.layers[0]
    assert isinstance(layer, BugStreamingLayer)
    ids = _prompt(40)
    with torch.no_grad():
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        for _ in range(70):
            kv_length, kv_offset = layer.get_mask_sizes(1)
            tok = out.logits[:, -1:].argmax(-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
            assert kv_length == layer.attended_length()
            assert kv_offset + kv_length == layer.cumulative_length


# --------------------------------------------------------------------------
# Blend retention (attn mass + surprise on a common rank scale)
# --------------------------------------------------------------------------


def test_blend_validation_and_alpha1_matches_attn(tiny_model: LlamaForCausalLM) -> None:
    # surprise_blend must be in [0, 1].
    with pytest.raises(ValueError, match="surprise_blend"):
        BugStreamingCache(
            tiny_model, rank=8, coord_budget=16, retention="blend", surprise_blend=2.0
        )
    # alpha=1 => the blend key is exactly rank(attn), so eviction order equals the
    # pure attn cache's; the whole teacher-forced trajectory must match bitwise.
    tiny_model.config._attn_implementation = "sdpa"
    stream = _prompt(80, seed=4)
    outs = []
    configs: tuple[tuple[str, dict[str, Any]], ...] = (
        ("attn", {}),
        ("blend", {"surprise_blend": 1.0}),
    )
    for retention, kw in configs:
        cache = BugStreamingCache(
            tiny_model,
            rank=8,
            coord_budget=12,
            recent_window=8,
            absorb_block=4,
            retention=retention,
            **kw,
        )
        logits = []
        with torch.no_grad(), cache.attach(tiny_model):
            out = tiny_model(stream[:, :30], past_key_values=cache, use_cache=True)
            for t in range(30, 80):
                out = tiny_model(stream[:, t : t + 1], past_key_values=cache, use_cache=True)
                logits.append(out.logits)
        outs.append(torch.cat(logits))
    assert torch.equal(outs[0], outs[1])


def test_blend_memory_counts_both_scalars(tiny_model: LlamaForCausalLM) -> None:
    # Blend stores mid_pos + mid_score + mid_surprise (2r+3), strictly more than
    # attn (2r+2). Both audited in stored_state_numel.
    stream = _prompt(80, seed=6)

    def peak(retention: str, **kw: Any) -> int:
        cache = BugStreamingCache(
            tiny_model,
            rank=8,
            coord_budget=16,
            recent_window=8,
            absorb_block=4,
            retention=retention,
            **kw,
        )
        mems = []
        with torch.no_grad(), cache.attach(tiny_model):
            tiny_model(stream[:, :30], past_key_values=cache, use_cache=True)
            for t in range(30, 80):
                tiny_model(stream[:, t : t + 1], past_key_values=cache, use_cache=True)
                mems.append(cache.stored_state_numel())
        return max(mems)

    assert peak("blend", surprise_blend=0.5) > peak("attn")


# --------------------------------------------------------------------------
# Correlation probe (the GO/NO-GO gate machinery)
# --------------------------------------------------------------------------


def test_surprise_probe_records_triples(tiny_model: LlamaForCausalLM) -> None:
    # An attn cache with the probe enabled must record (score, surprise, energy)
    # arrays of equal length at each eviction event, without changing the deployed
    # config's counted memory (the surprise buffer is instrumentation only).
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=12,
        recent_window=8,
        absorb_block=4,
        retention="attn",
    )
    sink = cache.enable_surprise_probe()
    mem_before = None
    stream = _prompt(90, seed=11)
    with torch.no_grad(), cache.attach(tiny_model):
        tiny_model(stream[:, :30], past_key_values=cache, use_cache=True)
        for t in range(30, 90):
            tiny_model(stream[:, t : t + 1], past_key_values=cache, use_cache=True)
            if mem_before is None:
                mem_before = cache.stored_state_numel()
    assert len(sink) > 0  # evictions happened and were recorded
    for score, surprise, energy in sink:
        assert score.shape == surprise.shape == energy.shape
        assert score.ndim == 1 and score.numel() > 0
        assert torch.all(surprise >= 0.0) and torch.all(surprise <= 1.0 + 1e-5)
    # The probe's surprise buffer is NOT counted against the attn config's budget.
    layer = cache.layers[0]
    assert isinstance(layer, BugStreamingLayer)
    assert layer._probe_surprise and not layer.track_surprise
    assert layer.mid_surprise is not None  # tracked for the probe...
    plain = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=12,
        recent_window=8,
        absorb_block=4,
        retention="attn",
    )
    with torch.no_grad(), plain.attach(tiny_model):
        tiny_model(stream[:, :30], past_key_values=plain, use_cache=True)
        for t in range(30, 90):
            tiny_model(stream[:, t : t + 1], past_key_values=plain, use_cache=True)
    assert cache.stored_state_numel() == plain.stored_state_numel()  # ...but not counted
