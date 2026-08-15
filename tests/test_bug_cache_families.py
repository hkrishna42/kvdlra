"""Non-Llama family pins for :class:`kvdlra.cache.BugStreamingCache` (Week-16 Tier-2).

Before spending GPU on Mistral-7B-v0.3 and Qwen2.5-7B we prove, hermetically on tiny
random-weight models (no download), that the cache's family-agnostic assumptions hold:

* **RoPE discovery + round trip** -- the cache reads each model's own ``model.rotary_emb``;
* **GQA grouping** at a non-1 ratio inferred from tensor shapes (Qwen 7:1, Mistral 4:1);
* **QKV bias** (Qwen2 carries q/k/v bias; it must flow through the ``k_proj`` capture).

The check is exact-mode parity vs :class:`DynamicCache` (rank == n_features, generous
budgets => nothing truncated), the same correctness ladder used for Llama in
``test_bug_cache.py``. If a family broke the cache, this fails loud at $0 CPU rather than
as silent 0s on a pod.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import (
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from transformers.cache_utils import DynamicCache

from kvdlra.cache import BugStreamingCache


def _qwen() -> Qwen2ForCausalLM:
    # 7 query : 1 KV head (Qwen2.5-7B's 28:4 ratio in miniature); head_dim 16 => n=16.
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=112,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=7,
        num_key_value_heads=1,
        max_position_embeddings=2048,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)  # type: ignore[no-untyped-call]
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _mistral() -> MistralForCausalLM:
    # 8 query : 2 KV head (Mistral-7B's 32:8 = 4:1 ratio); head_dim 16 => n=32; no SWA (v0.3).
    cfg = MistralConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        sliding_window=None,
    )
    torch.manual_seed(0)
    model = MistralForCausalLM(cfg)  # type: ignore[no-untyped-call]
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _n_features(model: Any) -> int:
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return int(head_dim * cfg.num_key_value_heads)


def _assert_exact_parity(model: Any) -> None:
    """Teacher-forced per-step logit parity of exact-mode BugStreamingCache vs
    DynamicCache -- pins masks, positions, the RoPE round trip and GQA grouping."""
    model.config._attn_implementation = "sdpa"
    n = _n_features(model)
    cache = BugStreamingCache(model, rank=n, coord_budget=4096, recent_window=8, absorb_block=4)
    ref = DynamicCache()
    g = torch.Generator().manual_seed(7)
    stream = torch.randint(0, int(model.config.vocab_size), (1, 90), generator=g)
    with torch.no_grad():
        out_a = model(stream[:, :33], past_key_values=cache, use_cache=True)
        out_b = model(stream[:, :33], past_key_values=ref, use_cache=True)
        assert torch.allclose(out_a.logits, out_b.logits, atol=1e-4)
        for t in range(33, 90):
            tok = stream[:, t : t + 1]
            out_a = model(tok, past_key_values=cache, use_cache=True)
            out_b = model(tok, past_key_values=ref, use_cache=True)
            assert torch.allclose(out_a.logits, out_b.logits, atol=1e-4)


def test_qwen2_has_qkv_bias() -> None:
    # Qwen2.5 carries q/k/v bias; the cache captures K via k_proj(hidden_states) so the
    # bias is included automatically -- assert the structural precondition holds.
    model = _qwen()
    k_proj = model.model.layers[0].self_attn.k_proj
    assert k_proj.bias is not None


def test_qwen2_exact_mode_parity_with_dynamic_cache() -> None:
    _assert_exact_parity(_qwen())


def test_mistral_exact_mode_parity_with_dynamic_cache() -> None:
    _assert_exact_parity(_mistral())
