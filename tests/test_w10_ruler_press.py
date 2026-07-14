"""Week-10/11 regression: the RULER press path runs scorer presses SINGLE-SHOT.

``w10_ruler.retrieve`` used to wrap kvpress scorer presses in ``ChunkPress`` for the
OOM-safe path. But ``SnapKVPress.compress`` asserts ``q_len > window_size`` (64), so
any ChunkPress chunk shorter than 64 (always true for the final chunk, and for small
``--chunk`` values) raised ``AssertionError: Query length ... should be greater than
the window size 64`` -- which is why SnapKV/MorphKV were absent from every Week-10
RULER result and only ExpectedAttention (no such assert) survived.

The fix runs scorer presses single-shot (full-``T`` prefill, ``logits_to_keep=1`` +
sdpa is memory-safe). This test drives ``retrieve`` with a scorer press and a sub-64
``chunk`` and asserts it returns a result row instead of raising.

Hermetic tiny Llama, no network (a stub tokenizer supplies ``.decode``).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.press.compat import install_kvpress_prefill_compat

H, D = 2, 16  # KV heads x head_dim -> n_features 32; num query heads 4


class _StubTok:
    """Minimal tokenizer: ``retrieve`` -> ``_decode`` only calls ``.decode``."""

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=H,
        head_dim=D,
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)  # type: ignore[no-untyped-call]
    model.config._attn_implementation = "sdpa"
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _prompt(t: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, t), generator=g)


def test_retrieve_snapkv_single_shot_survives_subwindow_chunk(
    tiny_model: LlamaForCausalLM,
) -> None:
    install_kvpress_prefill_compat()  # transformers cache_position shim
    from kvpress import SnapKVPress
    from w10_ruler import retrieve

    n, h_kv = H * D, H
    ctx = 128  # > SnapKV window_size (64) so single-shot q_len passes the assert
    hay, query = _prompt(ctx, seed=5), _prompt(6, seed=6)
    arm: dict[str, Any] = {
        "name": "snapkv-k0.5",
        "kind": "press",
        "rank": None,
        "keep": 0.5,
        "make": lambda: SnapKVPress(compression_ratio=0.5),
    }
    # chunk=48 < window(64): the old ChunkPress wrap raised "Query length 48 should be
    # greater than the window size 64"; the single-shot press path must not.
    hit, ratio, frac = retrieve(
        tiny_model, _StubTok(), arm, hay, query, ["needle"], "cpu", 48, n, h_kv, 4
    )
    assert isinstance(hit, bool)
    assert ratio > 0.0  # a real compressed-cache memory ratio was measured
    assert 0.0 <= frac <= 1.0
