"""``TurboQuantPress`` -- in-place PolarQuant quantization of the KV cache.

A standalone kvpress press that quantizes each token's key (and value) vector
with :class:`kvdlra.quant.PolarQuant` (TurboQuant §2), reconstructing in place
(same shape, ThinK-style -- so it measures the *quality* cost of ``bits``-bit KV
quantization). Its purpose is the **fairness control** for Week 4: quantization is
orthogonal to *how* you shrink the cache, so an honest comparison quantizes the
eviction baselines too. Compose it after an eviction press:

    ComposedPress([SnapKVPress(compression_ratio=0.6), TurboQuantPress(bits=4)])

gives "SnapKV x TurboQuant" -- evict tokens, then quantize the survivors -- the
apples-to-apples counterpart of ``BUGPress(rank=r, quant_bits=b)``
("BUG x TurboQuant"). Used by ``scripts/w4_fair.py``.

Quantizes the **per-token joint feature vector** (``num_kv_heads * head_dim`` =
512 for Llama-3.2-1B), matching the BUG feature convention. These are post-RoPE
keys; we quantize/dequantize in place (no re-rotation), so the PolarQuant rotation
Pi and RoPE never interact (see ``docs/notes/turboquant-rope-interaction.md``).
``compression_ratio`` is 0 (quantization removes no tokens); the real bit cost is
accounted for by the caller's memory model.

NOTE: library code -- no ``print``/I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from kvdlra.quant import PolarQuant

try:
    from kvpress.presses.base_press import BasePress
except ImportError as exc:  # pragma: no cover
    raise ImportError("TurboQuantPress requires kvpress (kvpress==0.5.1).") from exc

__all__ = ["TurboQuantPress"]


@dataclass
class TurboQuantPress(BasePress):  # type: ignore[misc]
    """PolarQuant the KV cache in place at ``bits`` bits/coordinate.

    Parameters
    ----------
    bits:
        Bits per coordinate for PolarQuant (typical: 2-4).
    compress_values:
        If ``True`` (default) also quantize values; else keys only.
    """

    bits: int = 4
    compress_values: bool = True
    _pq_cache: dict[tuple[int, str], PolarQuant] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ValueError(f"bits must be >= 1, got {self.bits}")

    @property
    def compression_ratio(self) -> float:
        """0 -- quantization removes no tokens (bit cost tracked by the caller)."""
        return 0.0

    @compression_ratio.setter
    def compression_ratio(self, value: float) -> None:
        if value not in (None, 0.0):
            raise AttributeError(f"compression_ratio is fixed at 0 for {type(self).__name__}")

    def _get_pq(self, dim: int, device: torch.device) -> PolarQuant:
        key = (dim, str(device))
        pq = self._pq_cache.get(key)
        if pq is None:
            pq = PolarQuant(dim=dim, bits=self.bits, device=device)
            self._pq_cache[key] = pq
        return pq

    def _quant(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize+dequantize the per-token joint vector of ``(bsz, H, T, D)``."""
        bsz, h, t, d = x.shape
        vecs = x.permute(0, 2, 1, 3).reshape(bsz, t, h * d)  # (bsz, T, H*D) per-token
        pq = self._get_pq(h * d, x.device)
        codes, norm = pq.quantize(vecs)
        rec = pq.dequantize(codes, norm)
        return rec.reshape(bsz, t, h, d).permute(0, 2, 1, 3)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys = self._quant(keys)
        if self.compress_values:
            values = self._quant(values)
        return keys, values
