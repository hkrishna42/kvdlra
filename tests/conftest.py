"""Shared hermetic fixtures.

The tiny random-weight Llama the cache tests run on (no download): 2 layers, 2 KV heads
x head_dim 16 => ``n_features = 32``. It lived in ``tests/test_bug_cache.py``; it is here
so any test module can request it.

A module whose model differs says so at module level -- ``TINY_MPE`` (the context ceiling)
and ``TINY_SDPA`` (the attention kernel). Those two are the ONLY way the thirteen private
copies of this fixture differed from each other, so they are the whole knob. The scope is
per REQUESTING module, so a test that mutates its model (several set ``sdpa`` mid-file)
still cannot reach another module's.

``tok`` is the whitespace tokenizer the generator tests share (``WhitespaceTok`` below).
"""

from __future__ import annotations

import zlib
from typing import Any

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


# --------------------------------------------------------------- the `tok` fixture


class _Enc(dict[str, Any]):
    """The two faces of HF's ``BatchEncoding`` the eval code reads: ``enc.input_ids``
    (``ruler._filler_cached``) and ``enc["input_ids"]`` (``templated_official``)."""

    @property
    def input_ids(self) -> Any:
        return self["input_ids"]


class WhitespaceTok:
    """A whitespace tokenizer with a chat template and STABLE ids -- ``crc32(word)`` -- so
    a golden hash computed on one machine reproduces on CI (a grow-on-demand vocabulary
    would depend on call order). Enough of the HF surface for both generators:
    ``__call__`` (``.input_ids`` a list, or a ``[1, n]`` tensor under
    ``return_tensors="pt"``), ``apply_chat_template`` (the content words plus a fixed
    3-token generation header) and ``decode`` (through the inverse map of every id it has
    handed out). The two private stubs in test_w10_ruler_filler / test_ruler_template_tail
    stay where they are: each pins a narrower surface on purpose."""

    name_or_path = "tests/tok"
    header = ("<eot>", "<asst>", "<hdr>")

    def __init__(self) -> None:
        self._words: dict[int, str] = {}

    def _ids(self, words: list[str]) -> list[int]:
        ids = [zlib.crc32(w.encode()) & 0x7FFFFFFF for w in words]
        self._words.update(zip(ids, words, strict=True))
        return ids

    def __call__(self, text: str, return_tensors: str | None = None, **_: Any) -> _Enc:
        ids = self._ids(text.split())
        return _Enc(input_ids=torch.tensor([ids]) if return_tensors else ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
        return_tensors: str | None = None,
        return_dict: bool = False,
    ) -> dict[str, torch.Tensor]:
        words = messages[0]["content"].split()
        if add_generation_prompt:
            words += self.header
        return {"input_ids": torch.tensor([self._ids(words)])}

    def decode(self, ids: Any) -> str:
        ids = ids.tolist() if hasattr(ids, "tolist") else ids
        return " ".join(self._words.get(int(i), "") for i in ids)


@pytest.fixture(scope="module")
def tok() -> WhitespaceTok:
    return WhitespaceTok()
