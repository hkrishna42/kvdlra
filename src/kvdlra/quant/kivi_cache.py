"""KIVI-faithful quantized-KV baseline caches (Week-19 A1).

transformers' ``QuantizedCache`` groups along the axis you name, but the backends'
axis semantics are easy to misread. For a ``(B, H, T, D)`` KV tensor:

* optimum-quanto ``axis=0``  -> groups of ``g`` *consecutive elements* = per-TOKEN groups
  (one scale per token per ``g`` channels);  ``axis=-1`` -> one scale per head-dim
  CHANNEL per ``g`` consecutive tokens = per-CHANNEL groups, which needs ``B*H*T`` to be a
  multiple of ``g`` (the Week-18 "Group size (64) must be a divisor of (65588)" SKIP).
* hqq ``axis=1`` -> consecutive-element groups (per-token); ``axis=0`` is a strided
  grouping with no KV meaning, so per-channel keys are done by transposing to
  ``(B, H, D, T)`` and grouping consecutively along ``T``.

KIVI (Liu et al., 2024) quantizes **keys per-channel** (outlier channels -- e.g. the
Qwen2.5 key-bias channel -- get their own scale) and **values per-token**. The Week-18
default quantized keys per-token, which one outlier channel reduces to noise at 2 AND
4 bits (retrieval 0.00). ``scheme="kivi"`` here is the faithful configuration;
``scheme="token"`` is the upstream default, bit-identical to the Week-18 arms.

The per-channel key path pads ``T`` to a multiple of the group by edge-replicating the
last token (no range distortion; groups never straddle heads) and slices the padding
off on dequantize. ``flush()`` folds the fp16 residual into the quantized store after a
chunked prefill, so decode starts from the same fully-quantized state single-shot
prefill produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers.cache_utils import HQQQuantizedLayer, QuantizedCache, QuantoQuantizedLayer

SCHEMES = ("token", "kivi")
BACKENDS = ("quanto", "hqq")
_NBITS = {"quanto": (2, 4), "hqq": (1, 2, 3, 4, 8)}


@dataclass
class _PerChannel:
    """A per-channel-quantized key store: the backend's object + the true token count."""

    q: Any
    t: int


class _KiviKeysMixin:
    """Per-channel KEY quantization + residual flush on top of a transformers layer.

    Set ``kivi=True`` to route the *key* quantize/dequantize calls (recognised by
    ``axis == self.axis_key``; the factory guarantees ``axis_key != axis_value`` in the
    kivi scheme) through the padded per-channel path. ``kivi=False`` never intercepts
    -> bit-identical to the upstream layer."""

    kivi: bool = False
    # backend-specific: how to present (B, H, Tp, D) so the backend groups along tokens
    _chan_axis: int = -1
    _transpose: bool = False

    # axis_key / axis_value / q_group_size / keys / values / _quantized_* come from the
    # transformers layer we are mixed into (accessed through ``cast(Any, self)``).
    axis_key: int
    axis_value: int
    q_group_size: int

    def _quantize(self, tensor: torch.Tensor, axis: int) -> Any:
        if not (self.kivi and axis == self.axis_key):
            return super()._quantize(tensor, axis)  # type: ignore[misc]
        t = int(tensor.shape[-2])
        pad = (-t) % self.q_group_size
        if pad:
            edge = tensor[..., -1:, :].expand(*tensor.shape[:-2], pad, tensor.shape[-1])
            tensor = torch.cat([tensor, edge], dim=-2)
        if self._transpose:
            tensor = tensor.transpose(-1, -2)
        q = super()._quantize(tensor.contiguous(), self._chan_axis)  # type: ignore[misc]
        return _PerChannel(q, t)

    def _dequantize(self, qtensor: Any) -> torch.Tensor:
        if not isinstance(qtensor, _PerChannel):
            return super()._dequantize(qtensor)  # type: ignore[misc, no-any-return]
        out: torch.Tensor = super()._dequantize(qtensor.q)  # type: ignore[misc]
        if self._transpose:
            out = out.transpose(-1, -2)
        return out[..., : qtensor.t, :].contiguous()

    def flush(self) -> None:
        """Fold the fp16 residual into the quantized store (no-op when empty)."""
        me = cast(Any, self)
        if me.keys.dim() != 4 or me.keys.shape[-2] == 0:
            return
        keys = torch.cat([me._dequantize(me._quantized_keys), me.keys], dim=-2)
        values = torch.cat([me._dequantize(me._quantized_values), me.values], dim=-2)
        me._quantized_keys = me._quantize(keys.contiguous(), axis=self.axis_key)
        me._quantized_values = me._quantize(values.contiguous(), axis=self.axis_value)
        me.keys = torch.tensor([], dtype=keys.dtype, device=keys.device)
        me.values = torch.tensor([], dtype=values.dtype, device=values.device)


class QuantoKiviLayer(_KiviKeysMixin, QuantoQuantizedLayer):  # type: ignore[no-untyped-call]
    _chan_axis = -1
    _transpose = False


class HqqKiviLayer(_KiviKeysMixin, HQQQuantizedLayer):  # type: ignore[no-untyped-call]
    _chan_axis = 1
    _transpose = True


def make_quant_cache(
    config: Any,
    *,
    nbits: int,
    scheme: str = "token",
    backend: str = "quanto",
    group: int = 64,
    residual: int = 128,
) -> QuantizedCache:
    """Build the quant baseline arm's cache. ``scheme="token"`` = transformers' default
    axes (bit-identical to the Week-18 arms); ``scheme="kivi"`` = per-channel keys +
    per-token values. Fails loud on an unsupported bit width for the backend."""
    if scheme not in SCHEMES:
        raise ValueError(f"quant scheme must be one of {SCHEMES}, got {scheme!r}")
    if backend not in BACKENDS:
        raise ValueError(f"quant backend must be one of {BACKENDS}, got {backend!r}")
    if nbits not in _NBITS[backend]:
        raise ValueError(f"{backend} supports nbits in {_NBITS[backend]}, got nbits={nbits}")
    # (axis_key, axis_value) per scheme, in each backend's own semantics (see module doc).
    if backend == "quanto":
        layer_cls: type[Any] = QuantoKiviLayer
        ak, av = (-1, 0) if scheme == "kivi" else (0, 0)
    else:
        layer_cls = HqqKiviLayer
        ak, av = (0, 1) if scheme == "kivi" else (1, 1)
    assert scheme == "token" or ak != av  # the mixin recognises key calls by axis
    cache = QuantizedCache(
        backend=backend,
        config=config,
        nbits=nbits,
        axis_key=ak,
        axis_value=av,
        q_group_size=group,
        residual_length=residual,
    )
    layers = []
    for _ in cache.layers:
        layer = layer_cls(nbits, ak, av, group, residual)
        layer.kivi = scheme == "kivi"
        layers.append(layer)
    cache.layers = layers
    return cache


def flush(cache: Any) -> None:
    """Fold every layer's fp16 residual into its quantized store (after chunked prefill)."""
    for layer in cache.layers:
        layer.flush()


def aux_words(cache: Any) -> float:
    """32-bit words the backend actually stores per group for (scale, zero), read off
    layer 0's quantized keys after prefill -- so the baseline is billed at its real
    aux precision (fp32 pairs on an fp32 model, 16-bit pairs on a bf16 model)."""
    q: Any = cache.layers[0]._quantized_keys
    if isinstance(q, _PerChannel):
        q = q.q
    if isinstance(q, tuple):  # hqq: (W_q, meta)
        scale, zero = q[1]["scale"], q[1]["zero"]
    else:  # quanto WeightQBitsTensor
        scale, zero = q._scale, q._shift
    return float((scale.element_size() + zero.element_size()) * 8 / 32)
