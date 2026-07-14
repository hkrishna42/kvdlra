"""``ShadowKVCache`` -- a faithful in-repo port of ShadowKV (arXiv:2410.21465).

ShadowKV (Sun et al.) is a *high-throughput long-context* KV cache: it keeps only
a **low-rank factorisation of the keys** on the accelerator, **offloads the full
value cache to CPU**, and at every decode step reconstructs and attends to only a
**query-selected sparse subset** of key/value chunks -- picked by scoring the
current query against one *landmark* (mean key) per chunk. The paper's headline is
a GPU-memory win from the value offload; under this repo's device-agnostic
float-equivalent accounting (``kvdlra.accounting``: a CPU float costs the same as a
GPU float) that offload buys *nothing* on the honest memory axis, so this port's
only fair savings are (a) the low-rank keys and (b) sparse decode. Counting the
offloaded values as "free" is the forbidden misreport; :meth:`stored_state_numel`
counts them at 1 float/elem, identical to a GPU float.

The four ShadowKV pillars, and how each maps onto this repo's cache pattern
--------------------------------------------------------------------------
1. **Landmarks.** At single-shot pre-fill the retained *middle* keys (everything
   but the ``n_sink`` sinks and the ``recent_window`` local tokens) are partitioned
   into chunks of ``chunk`` tokens; ``landmark[c]`` is the mean of the chunk's
   **post-RoPE** keys, per KV head. This is the query-similarity index used for
   selection.
2. **Low-rank K.** The middle keys are un-rotated to **pre-RoPE** (the exact RoPE
   round-trip from :mod:`kvdlra.cache.bug_cache`) and factorised per KV head by SVD
   into per-token coefficients ``coeff`` (``T_mid x r``) and a small basis
   ``basis`` (``r x head_dim``). Selected chunks are reconstructed on demand
   (``coeff[chunk] @ basis``) and re-rotated at their true positions.
3. **V offload to CPU.** The **entire** value cache is stored verbatim on CPU
   (``value.to("cpu")``); the selected chunks (+ sinks + recent) are gathered back
   to the device each step. The CPU values are counted in
   :meth:`stored_state_numel` (``T * n`` floats) -- never hidden.
4. **Top-k chunk retrieval at decode (the query-aware, PRE-attention step).**
   transformers 5.8 passes no kwargs to ``cache.update()``, so the layer cannot see
   the current query there. ShadowKV's selection is resolved -- like MorphKV's
   eviction -- from a **forward hook**, but ShadowKV's is a *pre*-attention hook
   (MorphKV's is an observe-only post-hook): :meth:`ShadowKVCache.attach` (decode)
   and :meth:`ShadowKVCache.frozen_scoring` (windowed scoring) register a
   ``forward_pre_hook`` that recomputes ``q = q_proj(hidden_states)`` with the
   step's RoPE, scores it against the landmarks per KV head, picks the top-``top_k``
   chunk ids, and stashes them on the layer; ``update()`` then reconstructs +
   re-rotates those key chunks, fetches those value chunks from CPU, and returns
   ``[sink | selected | recent]``. ``get_mask_sizes`` reports exactly that assembled
   length (``n_sink + min(top_k, n_chunks) * chunk + recent`` -- the selected *count*
   is query-independent, so the mask is stable) with a consistent ``kv_offset``;
   ``get_seq_length`` stays cumulative (true positions/RoPE keep advancing).

Selection is **per KV head** (each head scores its own landmarks and keeps its own
top-k chunk ids); the returned ``(1, H_kv, L, D)`` tensor already carries per-head
keys, and because every returned slot is strictly-past the shared causal mask marks
them all visible, so per-head chunk sets need no per-head mask.

Faithful-core deviations, documented (an honest labelled gap beats a hand-wavy
match):

1. **No outlier-chunk tier.** ShadowKV keeps a small fraction of high-variance
   chunks in full precision (the low-rank basis reconstructs them poorly); v1 omits
   that exact tier -- every middle chunk is low-rank. ``accounting.shadow_footprint``
   carries an ``outlier_frac`` knob (default 0) for when it is added.
2. **Chunk-aligned middle.** The middle length is trimmed to a multiple of
   ``chunk`` by folding the remainder into the recent window, so every chunk is
   exactly ``chunk`` tokens and the selected *count* ``min(top_k, n_chunks) * chunk``
   is fixed -- which is what makes ``get_mask_sizes`` predictable.
3. **Windowed / multi-query selection aggregates.** A ``q_len > 1`` scoring window
   (frozen scoring) produces one selection by summing the landmark scores over the
   window's queries (and over the GQA query-head group) -- a single fixed top-k for
   the whole window, since the returned mask is fixed.
4. **Decode keeps generated tokens verbatim.** ShadowKV targets long context + short
   generation; generated tokens are appended to the recent buffer (and their values
   to CPU) verbatim, with no re-chunking -- the compression is of the *pre-fill*
   context. Memory therefore grows by the (short) generation length, not with a
   bounded ring.
5. **Chunked pre-fill is not supported** (:meth:`ingesting` raises on a second
   chunk, like the ``BUGPress`` scope guard); single-shot pre-fill only.

Scope guards: batch size 1; single-shot pre-fill; pre-fill attention is
full/standard (compression bounds what is *retained for decode*, the same protocol
as every other frontier method).

References
----------
Sun et al., arXiv:2410.21465 (ShadowKV).
Xiao et al., arXiv:2309.17453 (StreamingLLM; sinks + recent window).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, CacheLayerMixin, LinearAttentionCacheLayerMixin
from transformers.models.llama.modeling_llama import rotate_half

from kvdlra.cache.bug_cache import _rope_unapply, _RopeAngles

logger = logging.getLogger(__name__)

__all__ = ["ShadowKVCache", "ShadowKVLayer"]


class ShadowKVLayer(CacheLayerMixin):  # type: ignore[no-untyped-call]
    """One layer's ShadowKV state: verbatim sinks + recent, low-rank middle keys,
    per-chunk landmarks, and the full value cache offloaded to CPU (see the module
    docstring). Tensors are per KV head ``(H_kv, ..., D)``; the ``(1, H, L, D)`` HF
    layout is assembled only at the ``update()`` boundary."""

    is_sliding = False

    def __init__(
        self,
        rope: _RopeAngles,
        rank_s: int,
        top_k: int,
        chunk: int = 8,
        recent_window: int = 32,
        n_sink: int = 4,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        if rank_s < 1:
            raise ValueError(f"rank_s must be >= 1, got {rank_s}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")
        if recent_window < 1:
            raise ValueError(f"recent_window must be >= 1, got {recent_window}")
        if n_sink < 0:
            raise ValueError(f"n_sink must be >= 0, got {n_sink}")
        self.rope = rope
        self.rank_s = rank_s
        self.top_k = top_k
        self.chunk = chunk
        self.recent_window = recent_window
        self.n_sink_cfg = n_sink
        # Week-10 harness mode: "normal" (decode) or "score" (non-mutating frozen
        # continuation-window forward). Set only by ShadowKVCache context managers.
        self._mode = "normal"
        self._reset_state()

    def _reset_state(self) -> None:
        self.cumulative_length = 0
        self.n_sink = 0
        self.mid_len = 0
        self.n_chunks = 0
        self.rank_eff = 0
        self.sink_k: Tensor | None = None  # (H_kv, n_sink, D) post-RoPE verbatim
        self.recent_k: Tensor | None = None  # (H_kv, recent_len, D) post-RoPE verbatim
        self.value_cpu: Tensor | None = None  # (H_kv, cumulative, D) verbatim, on CPU
        self.mid_coeff: Tensor | None = None  # (H_kv, T_mid, r) per-token coefficients
        self.mid_basis: Tensor | None = None  # (H_kv, r, D) small key basis (pre-RoPE)
        self.landmarks: Tensor | None = None  # (H_kv, n_chunks, D) post-RoPE chunk means
        self._selected: Tensor | None = None  # (H_kv, k_eff) planned chunk ids (int64)
        self._warned_unattached = False

    def lazy_initialization(self, key_states: Tensor, value_states: Tensor) -> None:
        if key_states.shape[0] != 1:
            raise NotImplementedError(
                f"ShadowKVLayer supports batch size 1, got {key_states.shape[0]}"
            )
        self.dtype, self.device = key_states.dtype, key_states.device
        self.num_kv_heads = int(key_states.shape[1])
        self.head_dim = int(key_states.shape[3])
        self.n_features = self.num_kv_heads * self.head_dim
        self.is_initialized = True

    # --------------------------------------------------------------- lengths

    def _recent_len(self) -> int:
        return 0 if self.recent_k is None else int(self.recent_k.shape[1])

    def _k_eff(self) -> int:
        return min(self.top_k, self.n_chunks)

    def _n_selected(self) -> int:
        """Selected middle tokens returned this step -- ``min(top_k, n_chunks) *
        chunk``, independent of *which* chunks (so ``get_mask_sizes`` is stable)."""
        return self._k_eff() * self.chunk

    def attended_length(self) -> int:
        """Tokens attention currently sees: sinks + selected middle + recent."""
        return self.n_sink + self._n_selected() + self._recent_len()

    # ----------------------------------------------------------- selection

    def _select_chunks(self, qf: Tensor) -> Tensor:
        """Score post-RoPE queries ``qf`` ``(H_kv, groups, q_len, D)`` against the
        per-head landmarks and return the top-``k_eff`` chunk ids ``(H_kv, k_eff)``
        (ascending, chronological). Scores sum over the GQA query-head group and the
        query window, i.e. one selection per KV head for the whole ``q_len``."""
        assert self.landmarks is not None
        # score[h, c] = sum_{g, q} <q_{h,g,q}, landmark_{h,c}>
        scores = torch.einsum("hgqd,hcd->hc", qf, self.landmarks.to(qf.dtype))
        k_eff = self._k_eff()
        if k_eff == 0:
            return torch.zeros(self.num_kv_heads, 0, dtype=torch.int64, device=scores.device)
        idx = torch.topk(scores, k_eff, dim=1).indices  # (H_kv, k_eff)
        sorted_idx: Tensor = idx.sort(dim=1).values
        return sorted_idx

    def _plan_selection(
        self, module: nn.Module, hidden_states: Tensor, cos: Tensor, sin: Tensor
    ) -> None:
        """PRE-attention hook body: recompute ``q`` with the step's RoPE, score it
        against the landmarks, and stash the selected chunk ids for ``update()``."""
        if self.n_chunks == 0:
            self._selected = torch.zeros(
                self.num_kv_heads, 0, dtype=torch.int64, device=self.device
            )
            return
        q_len = int(hidden_states.shape[1])
        q_proj = cast(nn.Linear, module.q_proj)
        q = q_proj(hidden_states)  # (1, q_len, H_q*D)
        q = q.view(1, q_len, -1, self.head_dim).transpose(1, 2)  # (1, H_q, q_len, D)
        q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)  # type: ignore[no-untyped-call]
        groups = q.shape[1] // self.num_kv_heads
        qf = q.to(torch.float32).view(self.num_kv_heads, groups, q_len, self.head_dim)
        self._selected = self._select_chunks(qf)

    def _selected_chunks(self) -> Tensor:
        """The planned selection, or a fall-back (the most recent ``k_eff`` chunks)
        if no pre-attention hook ran -- warned once, mirroring MorphKV/BUG."""
        k_eff = self._k_eff()
        if self._selected is not None and int(self._selected.shape[1]) == k_eff:
            return self._selected
        if not self._warned_unattached and k_eff < self.n_chunks:
            logger.warning(
                "ShadowKV selecting with no planned chunks (cache not attach()ed / "
                "frozen_scoring()?) -- falling back to the most recent %d chunks",
                k_eff,
            )
            self._warned_unattached = True
        recent = torch.arange(self.n_chunks - k_eff, self.n_chunks, device=self.device)
        return recent.unsqueeze(0).expand(self.num_kv_heads, -1)

    # ------------------------------------------------------ reconstruction

    def _reconstruct_selected(self, selected: Tensor) -> tuple[Tensor, Tensor]:
        """Reconstruct + re-rotate the selected middle keys and gather their values
        from CPU. ``selected`` is ``(H_kv, k_eff)`` chunk ids; returns key/value
        ``(H_kv, n_sel, D)`` on-device (keys post-RoPE at true positions)."""
        assert self.mid_coeff is not None and self.mid_basis is not None
        assert self.value_cpu is not None
        n_sel = self._n_selected()
        if n_sel == 0:
            empty = torch.empty(self.num_kv_heads, 0, self.head_dim, device=self.device)
            return empty.to(self.dtype), empty.to(self.dtype)
        col = torch.arange(self.chunk, device=self.device)
        key_heads: list[Tensor] = []
        val_heads: list[Tensor] = []
        for h in range(self.num_kv_heads):
            # chunk ids -> middle-token indices [c*chunk, c*chunk + chunk)
            tok = (selected[h].unsqueeze(1) * self.chunk + col).reshape(-1)  # (n_sel,)
            recon = self.mid_coeff[h][tok] @ self.mid_basis[h]  # (n_sel, D) pre-RoPE
            pos = tok + self.n_sink  # true positions of the middle tokens
            cs, sn = self.rope.cos_sin_at(pos)  # (n_sel, D)
            post = recon * cs + rotate_half(recon) * sn  # type: ignore[no-untyped-call]
            key_heads.append(post.to(self.dtype))
            val_heads.append(self.value_cpu[h][pos].to(self.device))
        return torch.stack(key_heads, dim=0), torch.stack(val_heads, dim=0)

    def _assemble(self) -> tuple[Tensor, Tensor]:
        """Assemble the retained ``[sink | selected | recent]`` K/V in HF layout
        ``(1, H_kv, L, D)`` (also used by tests for introspection)."""
        assert self.value_cpu is not None
        parts_k: list[Tensor] = []
        parts_v: list[Tensor] = []
        if self.sink_k is not None and self.sink_k.shape[1] > 0:
            parts_k.append(self.sink_k)
            parts_v.append(self.value_cpu[:, : self.n_sink].to(self.device))
        if self._n_selected() > 0:
            mid_k, mid_v = self._reconstruct_selected(self._selected_chunks())
            parts_k.append(mid_k)
            parts_v.append(mid_v)
        rlen = self._recent_len()
        if rlen > 0:
            assert self.recent_k is not None
            parts_k.append(self.recent_k)
            parts_v.append(self.value_cpu[:, self.cumulative_length - rlen :].to(self.device))
        k_ret = torch.cat(parts_k, dim=1) if len(parts_k) > 1 else parts_k[0]
        v_ret = torch.cat(parts_v, dim=1) if len(parts_v) > 1 else parts_v[0]
        return k_ret.unsqueeze(0), v_ret.unsqueeze(0)

    # -------------------------------------------------------------- update

    def update(
        self, key_states: Tensor, value_states: Tensor, *args: object, **kwargs: object
    ) -> tuple[Tensor, Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        q_len = int(key_states.shape[2])
        if self.cumulative_length == 0:
            return self._prefill(key_states, value_states)
        if self._mode == "score":
            return self._score_forward(key_states, value_states)
        if self._mode == "ingest":
            raise NotImplementedError(
                "ShadowKVCache supports single-shot pre-fill only; chunked ingest is "
                "not implemented (documented deviation #5)."
            )
        if q_len != 1:
            raise NotImplementedError(
                "ShadowKVCache supports single-shot pre-fill + one-token decode steps "
                f"only, got q_len={q_len} with {self.cumulative_length} cached tokens."
            )
        return self._decode_step(key_states, value_states)

    def _prefill(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        """Compress the prompt: verbatim sinks + recent keys, SVD low-rank middle
        keys + per-chunk landmarks, and the whole value cache offloaded to CPU.
        Returns the full K/V (standard full pre-fill attention)."""
        k = key_states[0]  # (H_kv, T, D) post-RoPE
        v = value_states[0]
        t = int(k.shape[1])
        self.value_cpu = v.detach().to("cpu").clone()  # full V offloaded, counted
        self.n_sink = min(self.n_sink_cfg, t)
        mid_full = t - self.n_sink - self.recent_window
        if mid_full <= 0:
            # No middle: degenerate to sinks + (all remaining as) recent.
            self.n_sink = min(self.n_sink_cfg, t)
            self.sink_k = k[:, : self.n_sink].clone() if self.n_sink > 0 else None
            self.recent_k = k[:, self.n_sink :].clone()
            self.mid_len = 0
            self.n_chunks = 0
            self.cumulative_length = t
            return key_states, value_states
        # Trim the middle to a whole number of chunks; fold the remainder into recent
        # so every chunk is exactly ``chunk`` tokens (deviation #2).
        self.mid_len = (mid_full // self.chunk) * self.chunk
        self.n_chunks = self.mid_len // self.chunk
        recent_len = t - self.n_sink - self.mid_len
        self.sink_k = k[:, : self.n_sink].clone() if self.n_sink > 0 else None
        mid_post = k[:, self.n_sink : self.n_sink + self.mid_len]  # (H_kv, T_mid, D)
        self.recent_k = k[:, self.n_sink + self.mid_len :].clone()  # (H_kv, recent_len, D)
        assert recent_len == self._recent_len()

        # Landmarks: mean of the chunk's POST-RoPE keys, per KV head (pillar 1).
        chunked = mid_post.reshape(self.num_kv_heads, self.n_chunks, self.chunk, self.head_dim)
        self.landmarks = chunked.mean(dim=2).to(torch.float32)  # (H_kv, n_chunks, D)

        # Low-rank K: un-rotate the middle to pre-RoPE, then per-head SVD (pillar 2).
        cs, sn = self.rope.cos_sin(self.n_sink, self.mid_len, k.device)  # (T_mid, D)
        mid_pre = _rope_unapply(
            mid_post.to(torch.float32), cs, sn, self.rope.scale_sq
        )  # (H_kv, T_mid, D)
        self.rank_eff = min(self.rank_s, self.head_dim, self.mid_len)
        coeff_heads: list[Tensor] = []
        basis_heads: list[Tensor] = []
        for h in range(self.num_kv_heads):
            # A = coeff @ basis with coeff = U*S (T_mid, r), basis = Vh (r, D).
            u, s, vh = torch.linalg.svd(mid_pre[h], full_matrices=False)
            r = self.rank_eff
            coeff_heads.append(u[:, :r] * s[:r])
            basis_heads.append(vh[:r])
        self.mid_coeff = torch.stack(coeff_heads, dim=0)  # (H_kv, T_mid, r)
        self.mid_basis = torch.stack(basis_heads, dim=0)  # (H_kv, r, D)
        self.cumulative_length = t
        return key_states, value_states

    def _decode_step(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        """Append the new token's key to the recent buffer (verbatim, GPU) and its
        value to the CPU cache (verbatim), then return ``[sink | selected | recent]``
        using the chunks the pre-attention hook planned."""
        assert self.recent_k is not None and self.value_cpu is not None
        self.recent_k = torch.cat([self.recent_k, key_states[0]], dim=1)
        self.value_cpu = torch.cat([self.value_cpu, value_states[0].detach().to("cpu")], dim=1)
        self.cumulative_length += 1
        return self._assemble()

    def _score_forward(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        """Frozen continuation scoring (non-mutating): attend the ``q_len`` window to
        ``[retained | window]`` without touching the compressed state, so one forward
        scores perplexity over the compressed cache. Mirrors
        ``BugStreamingLayer._score_forward``."""
        k_ret, v_ret = self._assemble()
        return (
            torch.cat([k_ret, key_states], dim=2),
            torch.cat([v_ret, value_states], dim=2),
        )

    # ------------------------------------------------------------ cache API

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Report exactly the K/V length ``update()`` will return this step. The
        selected count is query-independent, so this is stable across steps."""
        if self._mode == "score":
            attended = self.attended_length()
            return attended + query_length, self.cumulative_length - attended
        if self.cumulative_length == 0:  # single-shot pre-fill returns the full input
            return query_length, 0
        # Decode: the query's key joins the recent buffer, so the returned block is
        # sinks + selected + (recent + query_length).
        kv_length = self.n_sink + self._n_selected() + self._recent_len() + query_length
        kv_offset = self.cumulative_length + query_length - kv_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self) -> None:
        self._reset_state()

    def stored_state_numel(self) -> int:
        """Float-equivalents of the *stored* per-layer state (the honest memory):
        verbatim sink/recent keys, the low-rank key coefficients + basis, the
        per-chunk landmarks, and the **CPU-offloaded value cache** -- every element
        at 1 each, a CPU float identical to a GPU float (``kvdlra.accounting``). A
        zeroed CPU-value term would fail the anti-drift pin loudly."""
        total = 0
        for tsr in (
            self.sink_k,
            self.recent_k,
            self.mid_coeff,
            self.mid_basis,
            self.landmarks,
            self.value_cpu,  # offloaded V -- counted, never hidden
        ):
            if tsr is not None:
                total += int(tsr.numel())
        return total

    def cpu_value_numel(self) -> int:
        """Float-equivalents of the CPU-offloaded value cache alone (``T * n``)."""
        return 0 if self.value_cpu is None else int(self.value_cpu.numel())


class ShadowKVCache(Cache):
    """Model-level ShadowKV cache (one :class:`ShadowKVLayer` per model layer).

    Pass as ``past_key_values`` to ``model(...)`` / ``model.generate(...)``. Wrap
    the decode forward in :meth:`attach` (so the pre-attention hook plans the
    per-step chunk selection) and any windowed scoring in :meth:`frozen_scoring`.

    Parameters
    ----------
    model:
        The (Llama-family) model; supplies the layer count and the rotary embedding
        module used for the exact RoPE round-trip.
    rank_s:
        Per-KV-head SVD rank of the low-rank key factorisation (clamped to
        ``min(head_dim, T_mid)``).
    top_k:
        Chunks selected per KV head per step (the sparse-decode budget).
    chunk:
        Tokens per chunk (landmark granularity); the paper default is 8.
    recent_window:
        Local tokens kept verbatim on the accelerator (StreamingLLM window).
    n_sink:
        Attention-sink tokens kept verbatim (StreamingLLM sinks).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        rank_s: int = 160,
        top_k: int = 32,
        chunk: int = 8,
        recent_window: int = 32,
        n_sink: int = 4,
    ) -> None:
        base = getattr(model, "model", model)
        rotary = getattr(base, "rotary_emb", None)
        if rotary is None:
            raise ValueError(
                f"could not find a `rotary_emb` module on {type(model).__name__}; "
                "ShadowKVCache needs the model's own rotary embedding for the exact "
                "RoPE round trip"
            )
        rope = _RopeAngles(rotary)
        self._model = model  # for frozen_scoring's pre-hook (no model arg there)
        self.chunk = chunk
        n_layers = int(model.config.num_hidden_layers)
        layers: list[CacheLayerMixin | LinearAttentionCacheLayerMixin] = [
            ShadowKVLayer(
                rope=rope,
                rank_s=rank_s,
                top_k=top_k,
                chunk=chunk,
                recent_window=recent_window,
                n_sink=n_sink,
            )
            for _ in range(n_layers)
        ]
        super().__init__(layers=layers)

    # -------------------------------------------------- pre-attention hook

    def _install_hooks(self, model: PreTrainedModel) -> list[Any]:
        """Register the ``forward_pre_hook`` that plans per-step chunk selection."""
        handles: list[Any] = []

        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if kwargs.get("past_key_values") is not self:
                return None  # a different cache is in play; stay out
            layer = self.layers[module.layer_idx]
            if not isinstance(layer, ShadowKVLayer):
                return None
            if not layer.is_initialized or layer.cumulative_length == 0:
                return None  # pre-fill: nothing to select yet
            hidden_states = kwargs["hidden_states"]
            cos, sin = kwargs["position_embeddings"]
            layer._plan_selection(module, hidden_states, cos, sin)
            return None

        for module in model.modules():
            if hasattr(module, "layer_idx") and hasattr(module, "q_proj"):
                handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
        return handles

    @contextmanager
    def attach(self, model: PreTrainedModel) -> Iterator[None]:
        """Register the pre-attention selection hooks for a decode forward/generate
        (harmless during pre-fill, where the hook sees ``cumulative_length == 0`` and
        skips). Symmetric with :meth:`MorphKVCache.attach`, but ShadowKV's hook is a
        *pre*-hook (query-aware, before attention) rather than an observe-only
        post-hook."""
        handles = self._install_hooks(model)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _set_mode(self, mode: str) -> None:
        for layer in self.layers:
            if isinstance(layer, ShadowKVLayer):
                layer._mode = mode

    @contextmanager
    def frozen_scoring(self) -> Iterator[None]:
        """Week-10: score a continuation window over the compressed cache without
        mutating it. Installs the selection hooks (using the stored model reference)
        so the window's queries pick their chunks, sets every layer to ``"score"``
        (non-mutating ``[retained | window]`` return), and restores ``"normal"`` on
        exit. Symmetric with :meth:`BugStreamingCache.frozen_scoring`."""
        handles = self._install_hooks(self._model)
        self._set_mode("score")
        try:
            yield
        finally:
            self._set_mode("normal")
            for handle in handles:
                handle.remove()

    @contextmanager
    def ingesting(self) -> Iterator[None]:
        """Week-10 chunked pre-fill is NOT supported by ShadowKV (deviation #5): the
        first chunk single-shot-prefills, but a second chunk raises in ``update()``.
        Provided so the harness's ``getattr(cache, "ingesting")`` probe succeeds."""
        self._set_mode("ingest")
        try:
            yield
        finally:
            self._set_mode("normal")

    def consolidate(self) -> None:
        """No-op: ShadowKV compresses at single-shot pre-fill (no deferred absorb)."""

    def stored_state_numel(self) -> int:
        """Total stored float-equivalents across layers (GPU low-rank keys +
        landmarks + verbatim sinks/recent, and the CPU-offloaded value cache)."""
        return sum(
            layer.stored_state_numel() for layer in self.layers if isinstance(layer, ShadowKVLayer)
        )

    def attended_length(self) -> int:
        first = self.layers[0]
        assert isinstance(first, ShadowKVLayer)
        return first.attended_length()
