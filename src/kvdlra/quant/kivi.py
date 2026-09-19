"""The faithful KIVI arm (Liu et al. 2024): G=32, R=128, full-precision single-shot prefill.

KIVI's protocol: (i) attend over the whole prompt in full precision; (ii) quantize every
token but the trailing ``T mod R`` -- keys per channel, values per token, groups of 32;
(iii) decode against the quantized store plus an fp16 residual that is folded in every
``R`` tokens. The streaming arm (``kivi_cache``, the arm the paper-v1 tables used) breaks
(i): chunked 4096 prefill, later chunks attending to already-quantized context. Here the
prefill runs into a plain ``DynamicCache`` and ``quantize_after_prefill`` builds the
QuantizedCache's state from it post hoc, so (i)-(iii) all hold.

The cache object is ``kivi_cache.make_quant_cache``'s (the per-channel key path, the
residual mechanics); only the defaults differ. Deviation kept, recorded in the arm YAMLs:
no fused 2-bit kernel -- transformers dequantizes the store at every decode step.
"""

from __future__ import annotations

from typing import Any

from transformers.cache_utils import DynamicCache, QuantizedCache

from kvdlra.quant.kivi_cache import make_quant_cache


def make_kivi(
    config: Any, *, nbits: int, group: int = 32, residual: int = 128, backend: str = "quanto"
) -> QuantizedCache:
    """KIVI's published operating point: per-channel keys, per-token values, G=32, R=128."""
    return make_quant_cache(
        config, nbits=nbits, scheme="kivi", backend=backend, group=group, residual=residual
    )


def quantize_after_prefill(cache: QuantizedCache, dyn: DynamicCache) -> None:
    """Move every layer's fp16 K/V from ``dyn`` into ``cache``, KIVI's way: tokens
    ``[0, T - T mod R)`` quantized in ONE call per layer, the trailing ``T mod R`` kept
    fp16 as the layer's residual (``R = 0``: everything quantized). ``dyn`` is consumed
    layer by layer, so each fp16 slab is freed as its quantized store appears.

    The layer ends up exactly as its own ``update`` leaves it after a residual fold:
    ``_quantized_keys/_values``, ``keys/values`` holding the residual, and
    ``cumulative_length = T`` -- what ``get_seq_length``/``get_mask_sizes`` read -- so
    decode continues from the post-hoc state as it would from a streamed one.
    """
    # ponytail: one whole-layer quantize per layer; its temporaries are a full fp16 copy
    # of the slab plus the backend's workspace. If a 64K layer OOMs on the smoke pod,
    # quantize the slab in slices (a multiple of the group) and cat the codes.
    prefilled: list[Any] = dyn.layers
    for i, dl in enumerate(prefilled):
        layer: Any = cache.layers[i]
        k, v = dl.keys, dl.values
        t = int(k.shape[-2])
        cut = t - t % layer.residual_length if layer.residual_length else t
        layer.lazy_initialization(k, v)  # dtype/device -- hqq quantizes onto them
        layer._quantized_keys = layer._quantize(k[..., :cut, :].contiguous(), axis=layer.axis_key)
        layer._quantized_values = layer._quantize(
            v[..., :cut, :].contiguous(), axis=layer.axis_value
        )
        if cut < t:  # else the 1-D empty `lazy_initialization` left: upstream's own form
            layer.keys, layer.values = k[..., cut:, :].clone(), v[..., cut:, :].clone()
        layer.cumulative_length = t
        dl.keys = dl.values = k.new_empty(0)  # the fp16 slab goes now, not with ``dyn``


def residual_tokens(cache: Any) -> int:
    """Tokens layer 0 holds fp16 in its residual (``T mod R`` right after prefill)."""
    layer: Any = cache.layers[0]
    return int(layer.keys.shape[-2]) if layer.keys.dim() == 4 else 0
