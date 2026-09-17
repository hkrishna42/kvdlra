"""Shared hermetic fixtures.

The tiny random-weight Llama the cache tests run on (no download): 2 layers, 2 KV heads
x head_dim 16 => ``n_features = 32``. It lived in ``tests/test_bug_cache.py``; it is here
so any test module can request it.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

# Tiny Llama: 2 layers, 2 KV heads x head_dim 16 => n_features = 32.
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
