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
    "retention,hh_budget",
    [("fifo", 0), ("attn", 0), ("attn", 6)],
)
def test_bug_footprint_matches_stored_state_numel(
    tiny_model: LlamaForCausalLM, retention: str, hh_budget: int
) -> None:
    """``bug_footprint`` computed from the live layer's actual counts equals the
    measured ``stored_state_numel`` -- the pin across the frontier configs."""
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=24,
        recent_window=8,
        absorb_block=4,
        n_sink=4,
        retention=retention,
        hh_budget=hh_budget,
    )
    _drive(tiny_model, cache)
    layer = _bug_layer(cache)
    coord_count = layer._f_len() + layer._q_len()
    fp = acc.bug_footprint(
        N_FEATURES,
        rank=8,
        coord_count=coord_count,
        recent_len=layer._recent_len(),
        n_sink=4,
        retention=retention,
        hh_count=layer._hh_len(),
        u_present=layer.u_k is not None,
    )
    assert fp.float_equiv() == layer.stored_state_numel()


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


def test_full_cache_ratio_is_one() -> None:
    fp = acc.full_cache_footprint(4096, 512)
    assert fp.ratio_fp16(4096, 512) == pytest.approx(1.0)
    assert fp.tok_equiv(512) == pytest.approx(4096.0)


# ---------------------------------------------------- matched-memory gate


def test_assert_all_within_gate() -> None:
    acc.assert_all_within({"a": 100.0, "b": 200.0}, budget_per_layer=200.0)
    with pytest.raises(ValueError, match="matched-memory audit FAILED"):
        acc.assert_all_within({"a": 100.0, "b": 201.0}, budget_per_layer=200.0)
