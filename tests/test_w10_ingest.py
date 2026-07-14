"""Week-10 Phase 3: chunked ``ingest`` prefill on BUG + MorphKV (unlocks 32K/64K).

Chunked ingest appends each prompt chunk to the recent ring and defers the absorb
to ``consolidate()`` (BUG) / the attach hook's per-chunk seed+evict (MorphKV), so
no forward ever holds the full-``T`` K/V. Appending a block then consolidating is
equivalent to feeding the chunk one token at a time; it differs from single-shot
prefill only in that each chunk attends to a *compressed* past (the documented
deviation -- **exact at a lossless config**). Pins:

* lossless config (BUG full rank / MorphKV capacity >= T): chunked-ingest frozen
  scoring matches a full ``DynamicCache`` -> the ``get_mask_sizes`` math is correct
  end to end for ``q_len > 1`` ingest;
* chunked ingest reaches the SAME bounded compressed state as single-shot (memory
  is O(budget), not O(T)) -- the OOM-safety guarantee;
* ``ingesting()`` restores ``"normal"`` on exit.

Hermetic tiny Llama, mirroring ``tests/test_bug_cache.py``.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from kvdlra.cache import BugStreamingCache, MorphKVCache

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


def _make_bug(model: LlamaForCausalLM, *, rank: int, coord_budget: int) -> BugStreamingCache:
    return BugStreamingCache(
        model, rank=rank, coord_budget=coord_budget, recent_window=8, absorb_block=4
    )


def _chunked_prefill(
    model: LlamaForCausalLM, cache: BugStreamingCache | MorphKVCache, ids: torch.Tensor, chunk: int
) -> None:
    """Drive an OOM-safe chunked prefill: first chunk single-shot-prefills, later
    chunks ingest; consolidate after each (mirrors w10_frontier._prefill_chunked)."""
    t = int(ids.shape[1])
    with cache.attach(model), cache.ingesting():
        for start in range(0, t, chunk):
            stop = min(t, start + chunk)
            model(
                ids[:, start:stop],
                past_key_values=cache,
                use_cache=True,
                position_ids=_pos(start, stop - start),
            )
            cache.consolidate()


def _dynamic_window_logits(
    model: LlamaForCausalLM, ids: torch.Tensor, win: torch.Tensor
) -> torch.Tensor:
    ref = DynamicCache()
    t, w = int(ids.shape[1]), int(win.shape[1])
    with torch.no_grad():
        model(ids, past_key_values=ref, use_cache=True)
        out = model(win, past_key_values=ref, use_cache=True, position_ids=_pos(t, w))
    return cast(torch.Tensor, out.logits)


def _single_shot(
    model: LlamaForCausalLM, cache: BugStreamingCache | MorphKVCache, ids: torch.Tensor
) -> None:
    with torch.no_grad(), cache.attach(model):
        model(ids, past_key_values=cache, use_cache=True)


# --------------------------------------------------------- lossless oracle


def test_bug_chunked_ingest_lossless_matches_dynamic_cache(tiny_model: LlamaForCausalLM) -> None:
    """BUG full-rank + full coord budget => the compressed past IS the exact past,
    so chunked-ingest frozen scoring matches a full DynamicCache."""
    ids, win = _prompt(64), _prompt(16, seed=9)
    ref = _dynamic_window_logits(tiny_model, ids, win)
    bug = _make_bug(tiny_model, rank=N_FEATURES, coord_budget=4096)
    with torch.no_grad():
        _chunked_prefill(tiny_model, bug, ids, chunk=16)
        with bug.frozen_scoring():
            out = tiny_model(win, past_key_values=bug, use_cache=True, position_ids=_pos(64, 16))
    assert torch.allclose(out.logits, ref, atol=2e-3, rtol=2e-3)


def test_morph_chunked_ingest_lossless_matches_dynamic_cache(tiny_model: LlamaForCausalLM) -> None:
    """MorphKV capacity >= T keeps every token, so chunked ingest (append + hook
    seed/evict, no actual eviction) matches the full DynamicCache."""
    ids, win = _prompt(64), _prompt(16, seed=9)
    ref = _dynamic_window_logits(tiny_model, ids, win)
    morph = MorphKVCache(tiny_model, capacity=256, recent_window=8)
    with torch.no_grad():
        _chunked_prefill(tiny_model, morph, ids, chunk=16)
        with morph.frozen_scoring():
            out = tiny_model(win, past_key_values=morph, use_cache=True, position_ids=_pos(64, 16))
    assert torch.allclose(out.logits, ref, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------- bounded state (OOM-safety)


def test_bug_chunked_ingest_reaches_bounded_state(tiny_model: LlamaForCausalLM) -> None:
    """Chunked ingest of T tokens at a real (lossy) budget reaches a bounded
    compressed state ~= single-shot's -- O(budget), NOT O(T)."""
    ids = _prompt(160)
    single = _make_bug(tiny_model, rank=8, coord_budget=32)
    _single_shot(tiny_model, single, ids)
    chunked = _make_bug(tiny_model, rank=8, coord_budget=32)
    _chunked_prefill(tiny_model, chunked, ids, chunk=32)
    full_kv = 2 * N_FEATURES * int(ids.shape[1]) * tiny_model.config.num_hidden_layers
    # bounded well under the full cache, and within a few tokens of single-shot.
    assert chunked.stored_state_numel() < full_kv // 2
    rel = abs(chunked.stored_state_numel() - single.stored_state_numel())
    assert rel <= 4 * 2 * N_FEATURES * tiny_model.config.num_hidden_layers  # <= ~4 tokens slack


# --------------------------------------------------------- mode restore


def test_ingesting_restores_normal_mode(tiny_model: LlamaForCausalLM) -> None:
    bug: BugStreamingCache | MorphKVCache = _make_bug(tiny_model, rank=8, coord_budget=32)
    _chunked_prefill(tiny_model, bug, _prompt(48), chunk=16)
    assert all(cast(Any, layer)._mode == "normal" for layer in bug.layers)
    # the q_len==1 decode invariant is live again after ingest.
    with torch.no_grad(), pytest.raises(NotImplementedError):
        tiny_model(
            _prompt(3, seed=2), past_key_values=bug, use_cache=True, position_ids=_pos(48, 3)
        )
