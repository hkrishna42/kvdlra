"""``MorphKVCache`` -- a faithful-core reimplementation of MorphKV (ICML'25).

MorphKV (Ghadia et al., arXiv:2503.00979) maintains a **constant-size** KV cache
during long generation: ``R`` most-recent tokens (local coherence) plus the
top-``C`` older ("distant") tokens, re-ranked *every decode step* by fusing the
attention patterns of the recent tokens -- correlation-aware selection, in
contrast to StreamingLLM's position-only window or SnapKV's one-shot prefill
scores. It is the published competitor for Week-5 Axis B
(``docs/week5-plan.md``): constant-size *whole-token eviction* vs the BUG
cache's constant-size *low-rank running summary* at matched memory.

The algorithm (paper §3, Algorithm 1), as implemented here:

* cache = ``R`` recent + up to ``C`` distant tokens, per layer;
* each decode step, the new token's attention row over the kept tokens is
  recorded; the last ``R`` rows form the score window;
* fusion ``F[k] = sum_r AW_r[k]`` (``"sum"``) or ``max_r AW_r[k]`` (``"max"``)
  over the score window, distant slots only;
* GQA: query-head rows are softmaxed per head then **summed within each KV-head
  group** ("aggregate attention weights from the grouped query heads, then
  compute fusion scores for the shared key-value heads");
* selection is **per KV head** (each head keeps its own top-``C``; kept tensors
  stay rectangular), chronological order preserved;
* eviction runs every decode step once capacity is exceeded (the paper's
  fine-grained default).

kvpress 0.5.1 has no MorphKV, so this is a reimplementation. Faithful-core
deviations, documented:

1. **Prompt pruning at pre-fill end.** The paper is generation-focused and does
   not pin down prompt handling; we initialize the score window with the last
   ``R`` prompt tokens' (recomputed, causal) attention rows and prune to
   capacity immediately, so memory is constant from the first decode step --
   the same protocol as every other Axis-B method (full pre-fill attention,
   bounded *retained* state).
2. **Score-window memory is counted.** Sum/max fusion over a sliding window
   needs the last ``R`` rows (``H_kv x R x L`` floats); ``stored_state_numel``
   includes them. (A recompute-each-step variant could trade this memory for
   compute; the paper does not discuss the buffer's cost.)
3. Attention rows are **recomputed** from the layer's own ``q_proj`` + RoPE in a
   forward hook (transformers 5.8 neither passes attention weights to caches
   nor materializes them under sdpa). Recomputation is exact up to fp roundoff.

Usage::

    cache = MorphKVCache(model, capacity=600, recent_window=32)
    with cache.attach(model):
        model.generate(..., past_key_values=cache)

The mask/position plumbing mirrors :class:`kvdlra.cache.bug_cache.
BugStreamingLayer`: ``get_seq_length`` stays cumulative (true positions keep
advancing), ``get_mask_sizes`` reports exactly the returned length, and kept
tokens keep their true (post-RoPE) rotations. Batch size 1; single-shot
pre-fill.
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

logger = logging.getLogger(__name__)

__all__ = ["MorphKVCache", "MorphKVLayer"]


class MorphKVLayer(CacheLayerMixin):  # type: ignore[no-untyped-call]
    """One layer's MorphKV state: kept K/V + the recent-window score buffer."""

    is_sliding = False

    def __init__(
        self,
        capacity: int,
        recent_window: int,
        fusion: str = "sum",
        evict_interval: int = 1,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if recent_window < 1:
            raise ValueError(f"recent_window must be >= 1, got {recent_window}")
        if fusion not in ("sum", "max"):
            raise ValueError(f"fusion must be 'sum' or 'max', got {fusion!r}")
        if evict_interval < 1:
            raise ValueError(f"evict_interval must be >= 1, got {evict_interval}")
        self.capacity = capacity  # C: distant tokens kept
        self.recent_window = recent_window  # R: recent tokens kept + score window
        self.fusion = fusion
        # 1 = MorphKV's per-step fine-grained eviction; > 1 = the coarse-grained
        # variant (paper §3) == SnapKV-style windowed-attention eviction applied
        # periodically during decode (the cache overshoots capacity by at most
        # evict_interval - 1 tokens between prunes; count that in memory budgets).
        self.evict_interval = evict_interval
        self._obs_count = 0
        self.cumulative_length = 0
        self.keys: Tensor | None = None  # (1, H_kv, L, D) post-RoPE verbatim
        self.values: Tensor | None = None
        # Score buffer: last <=R recorded attention rows over the kept slots,
        # (H_kv, <=R, L), aligned column-for-column with keys/values.
        self.score_rows: Tensor | None = None
        # Week-10 harness mode (default "normal" preserves the q_len==1 decode
        # invariant). "score": non-mutating frozen continuation-window forward.
        # Set only by MorphKVCache.frozen_scoring().
        self._mode = "normal"

    def lazy_initialization(self, key_states: Tensor, value_states: Tensor) -> None:
        if key_states.shape[0] != 1:
            raise NotImplementedError(
                f"MorphKVLayer supports batch size 1, got {key_states.shape[0]}"
            )
        self.dtype, self.device = key_states.dtype, key_states.device
        self.num_heads = int(key_states.shape[1])
        self.head_dim = int(key_states.shape[3])
        self.is_initialized = True

    # -------------------------------------------------------------- update

    def update(
        self, key_states: Tensor, value_states: Tensor, *args: object, **kwargs: object
    ) -> tuple[Tensor, Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        q_len = int(key_states.shape[2])
        if self.cumulative_length == 0:
            self.keys = key_states
            self.values = value_states
            self.cumulative_length = q_len
            return key_states, value_states
        if self._mode == "score":
            # Week-10 frozen continuation scoring (non-mutating): attend the window
            # to [kept | window] without appending or evicting. get_mask_sizes
            # already reports kept + q_len with offset cumulative - kept.
            assert self.keys is not None and self.values is not None
            return (
                torch.cat([self.keys, key_states], dim=2),
                torch.cat([self.values, value_states], dim=2),
            )
        if self._mode == "ingest":
            # Week-10 chunked pre-fill (OOM-safe): append the q_len block; the
            # attach() hook's prefill branch (q_len > 1) then seeds scores from the
            # chunk's window attention and evicts back to capacity, so the kept set
            # stays bounded (peak = capacity + recent + chunk, never the full T).
            assert self.keys is not None and self.values is not None
            self.keys = torch.cat([self.keys, key_states], dim=2)
            self.values = torch.cat([self.values, value_states], dim=2)
            self.cumulative_length += q_len
            return self.keys, self.values
        if q_len != 1:
            raise NotImplementedError(
                "MorphKVCache supports single-shot pre-fill + one-token decode steps "
                f"only, got q_len={q_len} with {self.cumulative_length} cached tokens."
            )
        assert self.keys is not None and self.values is not None
        self.keys = torch.cat([self.keys, key_states], dim=2)
        self.values = torch.cat([self.values, value_states], dim=2)
        self.cumulative_length += 1
        return self.keys, self.values

    # ------------------------------------------------- scoring + eviction

    def observe(self, attn_row: Tensor) -> None:
        """Record one decode step's aggregated attention row ``(H_kv, L)`` over the
        kept slots (called by the attach() hook after attention), then evict."""
        assert self.keys is not None
        length = int(self.keys.shape[2])
        if attn_row.shape != (self.num_heads, length):
            raise ValueError(
                f"attn_row must be (H_kv={self.num_heads}, L={length}), got {tuple(attn_row.shape)}"
            )
        row = attn_row.to(torch.float32).unsqueeze(1)  # (H_kv, 1, L)
        if self.score_rows is None:
            self.score_rows = row
        else:
            buf = self.score_rows
            if buf.shape[2] < length:  # older rows never saw the newer tokens
                pad = torch.zeros(
                    self.num_heads,
                    buf.shape[1],
                    length - buf.shape[2],
                    dtype=buf.dtype,
                    device=buf.device,
                )
                buf = torch.cat([buf, pad], dim=2)
            self.score_rows = torch.cat([buf, row], dim=1)[:, -self.recent_window :, :]
        self._obs_count += 1
        if self._obs_count % self.evict_interval == 0:
            self._evict()

    def seed_scores(self, rows: Tensor) -> None:
        """Initialize the score window from the last ``R`` *prompt* tokens' causal
        attention rows ``(H_kv, R', T)`` (deviation #1), then prune to capacity."""
        self.score_rows = rows.to(torch.float32)[:, -self.recent_window :, :]
        self._evict()

    def consolidate(self) -> None:
        """Week-10 chunked pre-fill: the attach() hook already seeds+evicts per
        chunk; this defensively prunes to capacity so the kept set stays bounded
        even without a firing hook (no-op once evicted)."""
        self._evict()

    def _evict(self) -> None:
        """Per-KV-head: keep the last ``R`` slots + top-``C`` fused distant slots."""
        assert self.keys is not None and self.values is not None
        length = int(self.keys.shape[2])
        budget = self.capacity + self.recent_window
        if length <= budget or self.score_rows is None:
            return
        n_distant = length - self.recent_window
        distant = self.score_rows[:, :, :n_distant]  # (H_kv, r, n_distant)
        fused = distant.sum(dim=1) if self.fusion == "sum" else distant.amax(dim=1)
        top = torch.topk(fused, k=self.capacity, dim=1).indices  # (H_kv, C)
        recent_idx = (
            torch.arange(n_distant, length, device=top.device)
            .unsqueeze(0)
            .expand(self.num_heads, -1)
        )
        keep = torch.cat([top, recent_idx], dim=1).sort(dim=1).values  # (H_kv, C+R)

        gather_kv = keep.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, self.head_dim)
        self.keys = self.keys.gather(2, gather_kv)
        self.values = self.values.gather(2, gather_kv)
        gather_rows = keep.unsqueeze(1).expand(-1, self.score_rows.shape[1], -1)
        self.score_rows = self.score_rows.gather(2, gather_rows)

    # ------------------------------------------------------------ cache API

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        kept = 0 if self.keys is None else int(self.keys.shape[2])
        kv_length = kept + query_length
        kv_offset = self.cumulative_length + query_length - kv_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self) -> None:
        self.cumulative_length = 0
        self.keys = None
        self.values = None
        self.score_rows = None
        self._obs_count = 0

    def stored_state_numel(self) -> int:
        """Kept K/V floats + the score buffer (deviation #2: counted honestly)."""
        total = 0
        for t in (self.keys, self.values, self.score_rows):
            if t is not None:
                total += int(t.numel())
        return total


def _aggregated_attention_row(
    module: nn.Module, hidden_states: Tensor, keys: Tensor, cos: Tensor, sin: Tensor
) -> Tensor:
    """The current token's attention over ``keys``, softmaxed per query head and
    summed within each KV-head group (the paper's GQA aggregation): ``(H_kv, L)``.

    Recomputed from the layer's own ``q_proj`` + the step's RoPE embeddings, so
    it matches what attention actually used up to fp roundoff.
    """
    h_kv, head_dim = int(keys.shape[1]), int(keys.shape[3])
    q_proj = cast(nn.Linear, module.q_proj)
    q = q_proj(hidden_states)  # (1, 1, H_q*D)
    q = q.view(1, 1, -1, head_dim).transpose(1, 2)  # (1, H_q, 1, D)
    q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)  # type: ignore[no-untyped-call]
    groups = q.shape[1] // h_kv
    qf = q.to(torch.float32).view(h_kv, groups, head_dim)
    kf = keys[0].to(torch.float32)  # (H_kv, L, D)
    logits = torch.einsum("hgd,hld->hgl", qf, kf) / head_dim**0.5
    attn = torch.softmax(logits, dim=-1)  # per query head
    row: Tensor = attn.sum(dim=1)  # sum over the group -> (H_kv, L)
    return row


def _window_attention_rows(
    module: nn.Module,
    hidden_states: Tensor,
    keys: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    window: int,
) -> Tensor:
    """Causal attention rows of the last ``window`` prompt tokens over all prompt
    keys, GQA-aggregated: ``(H_kv, window, T)`` (pre-fill score seeding)."""
    h_kv, t, head_dim = int(keys.shape[1]), int(keys.shape[2]), int(keys.shape[3])
    # Bound the window by the actual hidden-states length too: during chunked
    # ingest the final chunk may carry fewer than ``window`` tokens, in which
    # case only its last ``w`` positions (t-w..t-1) can seed scores. Slicing q,
    # cos/sin and ``query_pos`` all off this ``w`` keeps the rows consistent.
    w = min(window, int(hidden_states.shape[1]), t)
    cos, sin = position_embeddings
    q_proj = cast(nn.Linear, module.q_proj)
    q = q_proj(hidden_states[:, -w:])  # (1, w, H_q*D)
    q = q.view(1, w, -1, head_dim).transpose(1, 2)  # (1, H_q, w, D)
    cos_w, sin_w = cos[:, -w:].unsqueeze(1), sin[:, -w:].unsqueeze(1)
    q = q * cos_w + rotate_half(q) * sin_w  # type: ignore[no-untyped-call]
    groups = q.shape[1] // h_kv
    qf = q.to(torch.float32).view(h_kv, groups, w, head_dim)
    kf = keys[0].to(torch.float32)  # (H_kv, T, D)
    logits = torch.einsum("hgwd,htd->hgwt", qf, kf) / head_dim**0.5
    # Causal within the window: query at position t-w+j sees keys <= t-w+j.
    key_pos = torch.arange(t, device=keys.device)
    query_pos = torch.arange(t - w, t, device=keys.device)
    future = key_pos[None, None, None, :] > query_pos[None, None, :, None]
    masked = logits.masked_fill(future, -torch.inf)
    attn = torch.softmax(masked, dim=-1)
    rows: Tensor = attn.sum(dim=1)  # (H_kv, w, T)
    return rows


class MorphKVCache(Cache):
    """Model-level MorphKV cache; pair with :meth:`attach` for score observation.

    Parameters
    ----------
    model:
        The (Llama-family) model; supplies the layer count.
    capacity:
        ``C`` -- distant tokens kept per layer per KV head.
    recent_window:
        ``R`` -- recent tokens kept; also the score-fusion window.
    fusion:
        ``"sum"`` (consistently-attended tokens) or ``"max"`` (strongly-attended).
    evict_interval:
        ``1`` (default) = MorphKV's per-step eviction; ``k > 1`` = the paper's
        coarse-grained variant, i.e. SnapKV-style windowed-attention eviction
        applied every ``k`` decode steps (used as the "SnapKV-decode" baseline).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        capacity: int = 600,
        recent_window: int = 32,
        fusion: str = "sum",
        evict_interval: int = 1,
    ) -> None:
        n_layers = int(model.config.num_hidden_layers)
        layers: list[CacheLayerMixin | LinearAttentionCacheLayerMixin] = [
            MorphKVLayer(
                capacity=capacity,
                recent_window=recent_window,
                fusion=fusion,
                evict_interval=evict_interval,
            )
            for _ in range(n_layers)
        ]
        super().__init__(layers=layers)

    @contextmanager
    def attach(self, model: PreTrainedModel) -> Iterator[None]:
        """Register per-layer forward hooks that record attention rows into this
        cache (and thereby drive eviction). Use around any forward/generate that
        passes this cache as ``past_key_values``."""
        handles = []

        def make_hook(cache: MorphKVCache) -> Any:
            def hook(
                module: nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                output: Any,
            ) -> Any:
                if kwargs.get("past_key_values") is not cache:
                    return output  # a different cache is in play; stay out
                layer = cache.layers[module.layer_idx]
                assert isinstance(layer, MorphKVLayer)
                if layer.keys is None:
                    return output
                if layer._mode == "score":
                    return output  # frozen scoring is non-mutating: never observe/evict
                hidden_states = kwargs["hidden_states"]
                cos, sin = kwargs["position_embeddings"]
                if hidden_states.shape[1] == 1:  # decode step
                    row = _aggregated_attention_row(module, hidden_states, layer.keys, cos, sin)
                    layer.observe(row)
                else:  # pre-fill: seed the score window, prune to capacity
                    rows = _window_attention_rows(
                        module, hidden_states, layer.keys, (cos, sin), layer.recent_window
                    )
                    layer.seed_scores(rows)
                return output

            return hook

        for module in model.modules():
            if hasattr(module, "layer_idx") and hasattr(module, "q_proj"):
                handles.append(module.register_forward_hook(make_hook(self), with_kwargs=True))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _set_mode(self, mode: str) -> None:
        for layer in self.layers:
            if isinstance(layer, MorphKVLayer):
                layer._mode = mode

    @contextmanager
    def frozen_scoring(self) -> Iterator[None]:
        """Week-10: score a continuation window over the compressed cache without
        mutating it (non-mutating [kept | window] return); restores "normal" on
        exit. Symmetric with :meth:`BugStreamingCache.frozen_scoring`."""
        self._set_mode("score")
        try:
            yield
        finally:
            self._set_mode("normal")

    @contextmanager
    def ingesting(self) -> Iterator[None]:
        """Week-10 chunked pre-fill (OOM-safe). Feed the prompt in chunks under an
        active :meth:`attach`; each chunk is appended and the hook seeds+evicts back
        to capacity. Call :meth:`consolidate` after each chunk. Restores "normal"."""
        self._set_mode("ingest")
        try:
            yield
        finally:
            self._set_mode("normal")

    def consolidate(self) -> None:
        """Prune every layer to capacity after a chunked-ingest forward."""
        for layer in self.layers:
            if isinstance(layer, MorphKVLayer):
                layer.consolidate()

    def stored_state_numel(self) -> int:
        """Total stored floats across layers (K/V + score buffers)."""
        return sum(
            layer.stored_state_numel() for layer in self.layers if isinstance(layer, MorphKVLayer)
        )

    def attended_length(self) -> int:
        first = self.layers[0]
        assert isinstance(first, MorphKVLayer)
        return 0 if first.keys is None else int(first.keys.shape[2])
