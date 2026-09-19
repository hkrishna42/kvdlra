"""Shared hermetic fixtures.

The tiny random-weight Llama the cache tests run on (no download): 2 layers, 2 KV heads
x head_dim 16 => ``n_features = 32``. It lived in ``tests/test_bug_cache.py``; it is here
so any test module can request it.

A module whose model differs says so at module level -- ``TINY_MPE`` (the context ceiling)
and ``TINY_SDPA`` (the attention kernel). Those two are the ONLY way the thirteen private
copies of this fixture differed from each other, so they are the whole knob. The scope is
per REQUESTING module, so a test that mutates its model (several set ``sdpa`` mid-file)
still cannot reach another module's.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

# Tiny Llama: 2 layers, 2 KV heads x head_dim 16 => n_features = 32.
H, D = 2, 16
N_FEATURES = H * D


def _tiny_config(max_position_embeddings: int = 2048) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=H,
        head_dim=D,
        max_position_embeddings=max_position_embeddings,
    )


@pytest.fixture(scope="module")
def tiny_model(request: pytest.FixtureRequest) -> LlamaForCausalLM:
    torch.manual_seed(0)
    cfg = _tiny_config(int(getattr(request.module, "TINY_MPE", 2048)))
    model = LlamaForCausalLM(cfg)  # type: ignore[no-untyped-call]
    if bool(getattr(request.module, "TINY_SDPA", False)):
        model.config._attn_implementation = "sdpa"
    model.eval()  # type: ignore[no-untyped-call]
    return model
