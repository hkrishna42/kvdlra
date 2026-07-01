"""``BUGPress`` -- a streaming dynamical-low-rank KV-cache press for kvpress.

This is the Week-3 bridge between the validated streaming BUG tracker
(:class:`kvdlra.integrators.streaming.StreamingBUG`, the Week-2 go/no-go winner)
and NVIDIA's ``kvpress`` compression framework. It plugs the tracker into the
:class:`kvpress.BasePress` ``compress`` hook so we can measure the *generation*
and *perplexity* cost of replacing a layer's KV cache with its rank-``r``
dynamical-low-rank reconstruction ``U Uᵀ M`` -- the quantity Weeks 1-2 only
measured as an offline reconstruction error.

Mechanism (why the tensor shape does not change)
-------------------------------------------------
Like :class:`kvpress.ThinKPress` (channel-wise key compression), ``BUGPress``
compresses along the *feature* direction, not the *sequence* direction: it
replaces ``keys``/``values`` with a low-rank reconstruction of the **same
shape**. There is therefore **no literal memory saving in this in-place form**
(the cache tensors keep shape ``(batch, num_kv_heads, seq_len, head_dim)``); the
compression is *nominal* -- a rank-``r`` factorization of the ``(512, T)``
feature-by-token matrix stores ``U`` (``512 x r``) plus per-token coordinates
(``r`` each), i.e. an asymptotic per-token budget of ``r / 512``. Storing the
cache in genuinely factored form is a Week-4+ concern; here we isolate the
*accuracy* question. See :attr:`compression_ratio`.

Matrix convention (``docs/notes/conventions.md``)
-------------------------------------------------
A per-layer key tensor ``K`` of shape ``(H, T, D) = (8, T, 64)`` (batch squeezed)
is factored as ``M = K.transpose(0, 2, 1).reshape(H*D, T)`` -- **rows = features**
(``head_dim * num_kv_heads = 512``), **columns = tokens** (``T``); a new token is
a new column. The rank-``r`` reconstruction is reshaped straight back.

RoPE operating point (Week-2 finding; ``docs/notes/rope-pitfall.md``)
--------------------------------------------------------------------
HuggingFace caches *post-RoPE* keys, which are markedly less low-rank; pre-RoPE
keys roughly halve the reconstruction error at matched rank. With
``pre_rope=True`` (default) ``BUGPress`` recomputes the pre-RoPE keys from the
layer's hidden states (via kvpress's :func:`get_prerope_key_states`), factors
*those*, then re-applies the layer's own RoPE (``kwargs["position_embeddings"]``)
to the reconstruction before writing it back -- so attention still sees correctly
rotated keys. Values carry no RoPE and are factored directly.

Attention sinks (``docs/PLAN.md`` §8 pitfall #5)
------------------------------------------------
The first ``n_sink`` token columns are known high-norm outliers that dominate the
spectrum; Week-2 excluded them from the low-rank model. Here they are kept
**exact** (StreamingLLM-style) -- only columns ``n_sink:`` are reconstructed.

References
----------
G. Ceruti, J. Kusch, C. Lubich, arXiv:2104.05247 (rank-adaptive BUG).
Y. Xu et al., "ThinK", arXiv:2407.21018 (the same-shape channel-compression
pattern this press mirrors).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import rotate_half

from kvdlra.integrators.streaming import StreamingBUG

try:  # kvpress is an optional heavy dependency; keep import errors legible.
    from kvpress.presses.base_press import BasePress
    from kvpress.utils import extract_keys_and_values, get_prerope_key_states
except ImportError as exc:  # pragma: no cover - exercised only without kvpress
    raise ImportError(
        "BUGPress requires kvpress; install the project with its pinned deps "
        "(kvpress==0.5.1). See pyproject.toml."
    ) from exc

logger = logging.getLogger(__name__)

__all__ = ["BUGPress"]


@dataclass
class BUGPress(BasePress):  # type: ignore[misc]
    """Dynamical-low-rank (streaming BUG) KV-cache press.

    Replaces each layer's keys and (optionally) values with their rank-``rank``
    streaming-BUG reconstruction during pre-fill, keeping the cache tensor shape
    unchanged (see the module docstring for why memory is only *nominally*
    reduced).

    Parameters
    ----------
    rank:
        Target tracked rank ``r`` of the ``(512, T)`` feature-by-token
        factorization (the ``rank_cap`` handed to :class:`StreamingBUG`). Typical
        Week-3 sweep values: 16, 32, 64. ``rank >= min(512, T - n_sink)`` recovers
        the input essentially exactly (used by the parity tests).
    pre_rope:
        If ``True`` (default, the Week-2 operating point) factor the *pre-RoPE*
        keys and re-apply RoPE to the reconstruction; if ``False`` factor the
        cached post-RoPE keys directly. No effect on values (RoPE-free).
    compress_values:
        If ``True`` (default) also low-rank the values; if ``False`` leave values
        untouched (keys-only compression).
    n_sink:
        Number of leading attention-sink token columns kept exact (default 4).
    """

    rank: int = 32
    pre_rope: bool = True
    compress_values: bool = True
    n_sink: int = 4
    # ``head_dim * num_kv_heads``; set from the model in ``post_init_from_model``
    # but defaulted to the Llama-3.2-1B value so the press is usable stand-alone.
    n_features: int = field(default=512)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.n_sink < 0:
            raise ValueError(f"n_sink must be >= 0, got {self.n_sink}")

    def post_init_from_model(self, model: PreTrainedModel) -> None:
        """Pin ``n_features = head_dim * num_kv_heads`` from the model config."""
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.n_features = int(head_dim * cfg.num_key_value_heads)

    @property
    def compression_ratio(self) -> float:
        """Nominal per-token memory reduction of the rank-``r`` factorization.

        ``1 - rank / n_features`` -- the asymptotic (large-``T``) fraction of
        per-token key memory removed by storing ``r`` coordinates instead of
        ``n_features`` raw features. **Nominal only**: this in-place press keeps
        the cache tensor shape, so it realizes no literal saving (same caveat as
        :class:`kvpress.ThinKPress`). Reported for the perplexity-vs-compression
        figure. Read-only.
        """
        return max(0.0, 1.0 - self.rank / self.n_features)

    @compression_ratio.setter
    def compression_ratio(self, value: float) -> None:
        raise AttributeError(f"compression_ratio cannot be set for {type(self).__name__}")

    def _lowrank_reconstruct(self, mat: torch.Tensor) -> torch.Tensor:
        """Rank-``rank`` streaming-BUG reconstruction of a ``(features, T)`` matrix.

        Keeps the first ``n_sink`` columns exact and reconstructs the rest via the
        orthogonal projection ``U Uᵀ M`` onto the tracked subspace. The BUG core
        runs in numpy float64 (:class:`StreamingBUG`, PLAN §8 pitfall #4 -- keep
        the core in high precision); the result is cast back to ``mat``'s dtype
        and device.
        """
        n_features, t = mat.shape
        if t <= self.n_sink:
            return mat  # only sinks present; nothing to compress

        mat_np = mat.detach().to(torch.float64).cpu().numpy()
        payload = np.ascontiguousarray(mat_np[:, self.n_sink :])

        tracker = StreamingBUG(n_features=n_features, rank_cap=self.rank)
        tracker.update_many(payload)
        payload_hat = tracker.project(payload)

        out = mat_np.copy()
        out[:, self.n_sink :] = payload_hat
        return torch.from_numpy(out).to(dtype=mat.dtype, device=mat.device)

    def _compress_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Apply :meth:`_lowrank_reconstruct` per batch element.

        ``x`` has shape ``(bsz, H, T, D)``; factored as ``(H*D, T)`` per batch
        element (the ``docs/notes/conventions.md`` layout) and reshaped back.
        """
        bsz, h, t, d = x.shape
        out = torch.empty_like(x)
        for b in range(bsz):
            mat = x[b].permute(0, 2, 1).reshape(h * d, t)  # (H*D, T)
            mat_hat = self._lowrank_reconstruct(mat)
            out[b] = mat_hat.reshape(h, d, t).permute(0, 2, 1)  # back to (H, T, D)
        return out

    def forward_hook(
        self,
        module: nn.Module,
        input: list[torch.Tensor],
        kwargs: dict[str, Any],
        output: list[Any],
    ) -> list[Any]:
        """Compress the layer's KV cache once, at the end of pre-fill.

        Overrides :meth:`kvpress.BasePress.forward_hook` for a
        version-robust pre-fill check. The upstream hook keys off
        ``kwargs["cache_position"]``, but transformers >= 5.8 does not thread
        ``cache_position`` into the attention module's kwargs (verified: the
        module receives only ``hidden_states``, ``attention_mask``,
        ``past_key_values``, ``position_embeddings``, ``position_ids``,
        ``use_cache``). We therefore detect pre-fill by the query length
        (``q_len > 1``); a length-1 forward is a decode step and is skipped.
        Only the unquantized cache path is supported (Week-3 scope).
        """
        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]

        cache_position = kwargs.get("cache_position")
        if cache_position is not None:
            if cache_position[-1] > q_len:  # past pre-fill (upstream semantics)
                return output
        elif q_len <= 1:  # decode step: nothing to (re)compress
            return output

        cache = kwargs["past_key_values"]
        cache_layer = cache.layers[module.layer_idx]
        keys, values = extract_keys_and_values(cache, module.layer_idx)
        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)
        cache_layer.keys = keys
        cache_layer.values = values
        return output

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace keys/values with their rank-``rank`` low-rank reconstruction.

        Called once per layer at the end of pre-fill (see
        :meth:`kvpress.BasePress.forward_hook`). Returns same-shape tensors.

        Only **single-shot pre-fill** is supported: the compressed forward must
        cover the whole cache. In the pre-RoPE path the reconstruction is built
        from ``hidden_states``/``position_embeddings``, which span only the
        *current* forward's tokens; if that forward does not cover the entire
        cache (chunked or continued pre-fill), the returned keys would be shorter
        than the values and the cache would silently desync. We detect that and
        raise rather than corrupt. (Supporting streaming across forwards would
        require carrying per-layer :class:`StreamingBUG` state between calls -- a
        later extension.)
        """
        q_len = hidden_states.shape[1]
        cache_len = keys.shape[2]
        if q_len != cache_len:
            raise NotImplementedError(
                "BUGPress supports only single-shot pre-fill: the compressed forward "
                f"must cover the whole cache, but q_len={q_len} != cache_len={cache_len}. "
                "Chunked/continued pre-fill would desync keys and values."
            )

        if self.pre_rope:
            # Factor the pre-RoPE keys (Week-2 operating point), then re-rotate.
            cos, sin = kwargs["position_embeddings"]
            keys_pre = get_prerope_key_states(module, kwargs["hidden_states"])
            keys_hat = self._compress_tensor(keys_pre)
            rot: torch.Tensor = rotate_half(keys_hat)  # type: ignore[no-untyped-call]
            keys = (keys_hat * cos.unsqueeze(1)) + (rot * sin.unsqueeze(1))
        else:
            keys = self._compress_tensor(keys)

        if self.compress_values:
            values = self._compress_tensor(values)

        return keys, values
