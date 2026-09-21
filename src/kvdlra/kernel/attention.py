"""The transformers 5.8.0 glue: a factored-attention function registered under
``KERNEL_ATTN`` for one cache's attach scope.

`LlamaAttention.forward` (modeling_llama.py:270-285) calls ``past_key_values.update(k, v,
layer_idx)`` and then ``ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation,
...)(self, q, k, v, attention_mask, dropout=..., scaling=self.scaling, **kwargs)``. In kernel
mode the layer's update returns only the dense tokens and keeps the factored middle
(`BugStreamingLayer.kernel_middles`), which the function below attends with the dense
tokens in one online softmax (`kvdlra.kernel.factored_attention`). Everything else -- a
prefill, a chunked-ingest chunk, a frozen-scoring window, any ``q_len > 1`` forward, the
reconstruct arm -- delegates to ``sdpa_attention_forward`` unchanged.

Two registrations, both required: the attention function (`AttentionInterface`), and
``sdpa_mask`` as the mask builder (`AttentionMaskInterface`) -- `create_causal_mask` returns
``None`` for any name absent from that mapping (masking_utils.py:836), which would run every
ingest chunk unmasked. With it, a decode forward gets no mask (q_len 1, no padding:
masking_utils.py:233-275) and an ingest forward gets exactly sdpa's mask. The scope also
switches ``model.config._attn_implementation`` (the setter stores any string,
configuration_utils.py:371-390) and restores it; on exit the name is re-bound to a stub, so
the class-level mapping never keeps a finished cache -- and its GPU state -- alive.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
from torch import Tensor
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
from transformers.modeling_utils import AttentionInterface

from kvdlra.kernel import FactoredMiddle, factored_attention

__all__ = ["KERNEL_ATTN", "attach_kernel", "factored_attention_forward"]

KERNEL_ATTN = "kvdlra_factored"


def factored_attention_forward(
    module: Any,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    *,
    cache: Any,
    inv_freq: Tensor,
    attention_scaling: float,
    **kwargs: Any,
) -> tuple[Tensor, None]:
    """The attention function transformers calls with (module, q, k, v, mask, ...).
    ``cache``, ``inv_freq`` and ``attention_scaling`` are bound by `attach_kernel`."""
    layer = cache.layers[module.layer_idx]
    mids = layer.kernel_middles() if hasattr(layer, "kernel_middles") else None
    if mids is None or query.shape[2] != 1:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
        )
    if attention_mask is not None:
        raise NotImplementedError(
            "the kernel decode step takes no attention mask (transformers builds none for "
            f"q_len 1 without padding); got one of shape {tuple(attention_mask.shape)}"
        )
    scale = float(module.scaling) if scaling is None else float(scaling)
    if any(m is None for m in mids) != all(m is None for m in mids):
        raise NotImplementedError("rows disagree on whether a middle exists (ragged rows)")
    mid = None if mids[0] is None else FactoredMiddle.cat([m for m in mids if m is not None])
    out = factored_attention(
        query, key, value, mid, inv_freq=inv_freq, attention_scaling=attention_scaling,
        scaling=scale, operand_dtype=layer.kernel_operand_dtype,
    )  # fmt: skip
    if cache.kernel_compare is not None:
        if layer._rows:
            raise NotImplementedError("kernel_compare reads one row's _decode_peek: batch 1 only")
        k_full, v_full = layer._decode_peek()  # reconstruct-then-attend, on the same query
        ref, _ = sdpa_attention_forward(
            module, query.float(), k_full.float(), v_full.float(), None, dropout=0.0, scaling=scale
        )  # fp32 over the bf16 stored representation -- the reference of test_kernel_reference.py
        cache.kernel_compare[int(module.layer_idx)] = float(
            (out.transpose(1, 2).float() - ref.float()).abs().max()
        )
    return out.transpose(1, 2).contiguous(), None


def _unbound(*args: Any, **kwargs: Any) -> tuple[Tensor, None]:
    raise RuntimeError(f"{KERNEL_ATTN}: no cache attached (use `with cache.attach(model):`)")


@contextmanager
def attach_kernel(cache: Any, model: Any) -> Iterator[None]:
    """Bind `factored_attention_forward` to ``cache`` and the model's rotary constants under
    ``KERNEL_ATTN``, switch ``model.config._attn_implementation`` to it, restore on exit."""
    base = getattr(model, "model", model)
    rotary = base.rotary_emb
    AttentionInterface.register(
        KERNEL_ATTN,
        partial(
            factored_attention_forward,
            cache=cache,
            inv_freq=rotary.inv_freq.to(torch.float32),
            attention_scaling=float(getattr(rotary, "attention_scaling", 1.0)),
        ),
    )
    AttentionMaskInterface.register(KERNEL_ATTN, sdpa_mask)
    previous = model.config._attn_implementation
    model.config._attn_implementation = KERNEL_ATTN
    try:
        yield
    finally:
        model.config._attn_implementation = previous
        AttentionInterface.register(KERNEL_ATTN, _unbound)
