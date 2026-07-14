"""Week-10 Phase 1b: frozen-scoring (``score``) mode on BUG + MorphKV.

Score mode lets a ``q_len = W`` continuation window attend to the *compressed*
cache without mutating it, so one forward measures perplexity over the compressed
cache (``docs/week10-plan.md``). Pins:

* the default mode is ``"normal"`` -- the ``q_len == 1`` decode invariant and every
  existing call site are untouched;
* the ``q_len != 1`` raise still fires outside score mode;
* a frozen-scoring forward is **non-mutating** (``stored_state_numel`` and
  ``cumulative_length`` unchanged);
* in a LOSSLESS config the frozen-scoring window logits match a full
  ``DynamicCache`` -- i.e. the ``get_mask_sizes`` math is correct end to end.

Hermetic tiny Llama, mirroring ``tests/test_bug_cache.py``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from kvdlra.cache import BugStreamingCache, MorphKVCache
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
    model.config._attn_implementation = "sdpa"
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _prompt(t: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, t), generator=g)


def _pos(start: int, n: int) -> torch.Tensor:
    return torch.arange(start, start + n).unsqueeze(0)


def _make_bug(
    model: LlamaForCausalLM,
    *,
    rank: int = 8,
    coord_budget: int = 128,
    recent_window: int = 8,
    absorb_block: int = 4,
) -> BugStreamingCache:
    return BugStreamingCache(
        model,
        rank=rank,
        coord_budget=coord_budget,
        recent_window=recent_window,
        absorb_block=absorb_block,
    )


def _build(model: LlamaForCausalLM, kind: str) -> BugStreamingCache | MorphKVCache:
    if kind == "bug":
        return _make_bug(model)
    return MorphKVCache(model, capacity=32, recent_window=8)


# ---------------------------------------------------------- default / invariant


def test_default_mode_is_normal(tiny_model: LlamaForCausalLM) -> None:
    bug = _make_bug(tiny_model)
    assert all(layer._mode == "normal" for layer in bug._bug_layers())
    morph = MorphKVCache(tiny_model, capacity=32, recent_window=8)
    assert all(la._mode == "normal" for la in morph.layers if isinstance(la, MorphKVLayer))


@pytest.mark.parametrize("kind", ["bug", "morph"])
def test_qlen_raise_still_fires_outside_score(tiny_model: LlamaForCausalLM, kind: str) -> None:
    cache = _build(tiny_model, kind)
    ids = _prompt(40)
    with torch.no_grad():
        tiny_model(ids, past_key_values=cache, use_cache=True)  # single-shot prefill
        with pytest.raises(NotImplementedError):  # a q_len=3 continuation, normal mode
            tiny_model(
                _prompt(3, seed=2), past_key_values=cache, use_cache=True, position_ids=_pos(40, 3)
            )


# ---------------------------------------------------------- non-mutating score


@pytest.mark.parametrize("kind", ["bug", "morph"])
def test_score_forward_is_non_mutating(tiny_model: LlamaForCausalLM, kind: str) -> None:
    cache = _build(tiny_model, kind)
    ids = _prompt(60)
    with torch.no_grad(), cache.attach(tiny_model):
        tiny_model(ids, past_key_values=cache, use_cache=True)
    layer0 = cast(Any, cache.layers[0])
    before_numel = cache.stored_state_numel()
    before_cum = layer0.cumulative_length
    win = _prompt(16, seed=3)
    with torch.no_grad(), cache.frozen_scoring():
        out = tiny_model(win, past_key_values=cache, use_cache=True, position_ids=_pos(60, 16))
    assert out.logits.shape[1] == 16
    assert cache.stored_state_numel() == before_numel
    assert layer0.cumulative_length == before_cum
    # mode is restored to normal on exit -> the q_len==1 raise path is live again.
    assert all(cast(Any, layer)._mode == "normal" for layer in cache.layers)


# ---------------------------------------------------------- lossless correctness


def _reference_window_logits(
    model: LlamaForCausalLM, ids: torch.Tensor, win: torch.Tensor
) -> torch.Tensor:
    ref = DynamicCache()
    t, w = int(ids.shape[1]), int(win.shape[1])
    with torch.no_grad():
        model(ids, past_key_values=ref, use_cache=True)
        out = model(win, past_key_values=ref, use_cache=True, position_ids=_pos(t, w))
    return cast(torch.Tensor, out.logits)


def test_bug_frozen_scoring_matches_dynamic_cache_when_lossless(
    tiny_model: LlamaForCausalLM,
) -> None:
    """BUG exact mode (full rank, coord budget >> mid) losslessly retains the
    prompt, so its frozen-scoring window logits match a full DynamicCache."""
    ids, win = _prompt(50), _prompt(16, seed=7)
    ref_logits = _reference_window_logits(tiny_model, ids, win)
    bug = _make_bug(tiny_model, rank=N_FEATURES, coord_budget=4096)
    with torch.no_grad(), bug.attach(tiny_model):
        tiny_model(ids, past_key_values=bug, use_cache=True)
    with torch.no_grad(), bug.frozen_scoring():
        out = tiny_model(win, past_key_values=bug, use_cache=True, position_ids=_pos(50, 16))
    assert torch.allclose(out.logits, ref_logits, atol=2e-3, rtol=2e-3)


def test_morph_frozen_scoring_matches_dynamic_cache_when_no_eviction(
    tiny_model: LlamaForCausalLM,
) -> None:
    """MorphKV with capacity >= context keeps every token verbatim, so frozen
    scoring is bit-for-bit the full-cache window forward."""
    ids, win = _prompt(50), _prompt(16, seed=7)
    ref_logits = _reference_window_logits(tiny_model, ids, win)
    morph = MorphKVCache(tiny_model, capacity=128, recent_window=8)
    with torch.no_grad(), morph.attach(tiny_model):
        tiny_model(ids, past_key_values=morph, use_cache=True)
    with torch.no_grad(), morph.frozen_scoring():
        out = tiny_model(win, past_key_values=morph, use_cache=True, position_ids=_pos(50, 16))
    assert torch.allclose(out.logits, ref_logits, atol=1e-4, rtol=1e-4)
