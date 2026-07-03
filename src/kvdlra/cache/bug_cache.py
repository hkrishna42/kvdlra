"""``BugStreamingCache`` -- a constant-memory decode-time streaming BUG KV cache.

Week-5 Axis B (``docs/week5-plan.md`` §"New capability to build",
``docs/notes/streaming-decode-design.md``): the first use of BUG *as a streaming
integrator during generation*. Everywhere else in this project BUG compresses
the pre-fill and is static during decode (:class:`kvdlra.press.BUGPress`); this
cache advances the tracked subspace **per generated token** at a fixed rank cap,
so the stored cache size is *bounded* -- independent of how many tokens have
been generated -- while attention keeps seeing a running low-rank reconstruction
of the whole retained history.

What is stored per layer (all bounded; see the design note §4)
--------------------------------------------------------------
* the first ``n_sink`` tokens' K/V **verbatim** (K post-RoPE, bit-exact --
  attention sinks, PLAN §8 pitfall #5);
* a **recent ring** of the last ``recent_window``..``recent_window +
  absorb_block - 1`` tokens' K/V verbatim (local context is exact);
* the **middle** tokens as BUG state: an orthonormal feature basis ``U`` (``n x
  r``, tracked on *pre-RoPE* keys -- the Week-2 operating point), a square-root
  core ``B`` (``r x r``, steers the basis; attention never sees it), and up to
  ``coord_budget`` per-token **coordinate** columns ``C`` (``r`` floats per
  token instead of ``n``). When the coordinate buffer is full the *oldest*
  columns are dropped -- that is the honest memory bound: softmax attention
  needs per-token information for every attendable token, so "constant memory"
  can only mean bounding the attended set. At matched memory the BUG cache
  retains a ~``n/r`` x longer (but only approximately represented) history than
  whole-token eviction; that trade is exactly the Axis-B experiment.

The per-token cycle (design note §5)
------------------------------------
``update()`` receives the new token's **post-RoPE** key (transformers >= 5.8
passes no kwargs to the cache) and pushes it verbatim into the recent ring.
When the ring overflows, the oldest ``absorb_block`` tokens *graduate*: their
keys are exactly un-rotated to pre-RoPE (the model's own rotary embedding, so
angles are bit-identical; the inverse divides by ``attention_scaling**2``), one
augmented rank-adaptive BUG step (:func:`kvdlra.integrators.streaming_torch.
augmented_bug_step` -- the validated integrator math, fp32 core per PLAN §8
pitfall #4) advances ``(U, B)``, existing coordinates are re-expressed in the
new basis (``C <- rot @ C`` -- each truncation projects old tokens onto the new
subspace; the graceful-degradation mechanism the DLRA robustness bound governs),
and the graduating coordinates are appended. Attention then sees ``[sinks |
RoPE(U C, true positions) | recent]``; the middle reconstruction only changes on
absorb events and is cached in between, so the steady-state per-step cost is a
concat plus attention over a constant-length cache. Amortized per-token update
cost: ``O(n r + (r+b)^3 / b + r^2 W / b)`` -- bounded, independent of generated
length.

Positions and masks
-------------------
``get_seq_length()`` keeps returning the *cumulative* token count, so the model
keeps advancing true positions (transformers 5.8 derives ``position_ids`` from
it) and the query is rotated at its true position. Retained tokens keep their
true rotations too (sinks/recents verbatim; the middle re-rotated at its true
positions), so relative-position geometry is preserved. ``get_mask_sizes``
reports exactly the length ``update()`` will return this step, with ``kv_offset
= cumulative + q - length`` so the causal mask sees every returned (strictly
past) token as visible.

Scope guards: batch size 1; single-shot pre-fill (chunked pre-fill raises, as
in ``BUGPress``); pre-fill attention is full/standard (the same protocol as all
Axis-B baselines -- compression bounds what is *retained for decode*).

``rank=0`` or ``coord_budget=0`` disables the low-rank middle entirely:
graduating tokens are simply dropped and the cache degenerates to **sinks +
recent window** -- i.e. the StreamingLLM baseline falls out of the same
implementation.

References
----------
G. Ceruti, J. Kusch, C. Lubich, arXiv:2104.05247 (rank-adaptive BUG).
Xiao et al., arXiv:2309.17453 (StreamingLLM; sinks + recent window).
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, CacheLayerMixin, LinearAttentionCacheLayerMixin
from transformers.models.llama.modeling_llama import rotate_half

from kvdlra.integrators.streaming_torch import augmented_bug_step

logger = logging.getLogger(__name__)

__all__ = ["BugStreamingCache", "BugStreamingLayer"]


class _RopeAngles:
    """cos/sin provider for arbitrary position ranges, via the model's own rotary.

    Using the model's ``rotary_emb`` module (rather than re-deriving frequencies)
    guarantees the angles -- including any RoPE scaling (Llama-3 ``llama3`` rope,
    linear scaling, ...) -- are identical to what attention applied to the keys,
    so the un-rotate/re-rotate round trip is exact up to floating-point roundoff.

    All layers absorb on the same schedule and therefore request the same
    position ranges within a decode step, so results are memoized (small LRU).
    """

    def __init__(self, rotary_emb: nn.Module) -> None:
        self._rotary = rotary_emb
        # cos/sin returned by the module are scaled by ``attention_scaling``;
        # the exact inverse rotation divides by its square (design note §3).
        self.scale_sq = float(getattr(rotary_emb, "attention_scaling", 1.0)) ** 2
        self._memo: OrderedDict[tuple[int, int, str], tuple[Tensor, Tensor]] = OrderedDict()

    def cos_sin(self, start: int, length: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """fp32 ``(cos, sin)`` of shape ``(length, head_dim)`` for positions
        ``start .. start+length-1`` (scaled exactly as the model scales them)."""
        key = (start, length, str(device))
        hit = self._memo.get(key)
        if hit is not None:
            self._memo.move_to_end(key)
            return hit
        positions = torch.arange(start, start + length, device=device).unsqueeze(0)
        probe = torch.empty(1, dtype=torch.float32, device=device)
        cos, sin = self._rotary(probe, positions)
        out = (cos.squeeze(0).to(torch.float32), sin.squeeze(0).to(torch.float32))
        self._memo[key] = out
        while len(self._memo) > 8:
            self._memo.popitem(last=False)
        return out


def _rope_apply(x_htd: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to ``(H, T, D)`` given ``(T, D)`` cos/sin (broadcast over heads)."""
    rot: Tensor = rotate_half(x_htd)  # type: ignore[no-untyped-call]
    return x_htd * cos + rot * sin


def _rope_unapply(x_htd: Tensor, cos: Tensor, sin: Tensor, scale_sq: float) -> Tensor:
    """Exact inverse of :func:`_rope_apply` (rotation transposed, / scaling^2)."""
    rot: Tensor = rotate_half(x_htd)  # type: ignore[no-untyped-call]
    return (x_htd * cos - rot * sin) / scale_sq


class BugStreamingLayer(CacheLayerMixin):  # type: ignore[no-untyped-call]
    """One layer's constant-memory streaming-BUG KV state (see module docstring).

    Internally tokens live as feature-by-token matrices ``(n, T)`` with ``n =
    num_kv_heads * head_dim`` (``docs/notes/conventions.md``); conversion to the
    HF layout ``(1, H, T, D)`` happens only at the ``update()`` boundary.
    """

    is_sliding = False

    def __init__(
        self,
        rope: _RopeAngles,
        rank: int,
        coord_budget: int,
        recent_window: int = 64,
        absorb_block: int = 32,
        n_sink: int = 4,
        theta: float | None = None,
        prefill_block_size: int = 128,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        if rank < 0 or coord_budget < 0:
            raise ValueError("rank and coord_budget must be >= 0")
        if recent_window < 1:
            raise ValueError(f"recent_window must be >= 1, got {recent_window}")
        if absorb_block < 1:
            raise ValueError(f"absorb_block must be >= 1, got {absorb_block}")
        if n_sink < 0:
            raise ValueError(f"n_sink must be >= 0, got {n_sink}")
        if prefill_block_size < 1:
            raise ValueError(f"prefill_block_size must be >= 1, got {prefill_block_size}")
        self.rope = rope
        self.rank = rank
        self.coord_budget = coord_budget
        self.recent_window = recent_window
        self.absorb_block = absorb_block
        self.n_sink = n_sink
        self.theta = theta
        self.prefill_block_size = prefill_block_size
        # rank=0 / coord_budget=0 => no low-rank middle => StreamingLLM baseline.
        self.lowrank_enabled = rank >= 1 and coord_budget >= 1
        self.cumulative_length = 0
        self._reset_state()

    # ------------------------------------------------------------------ state

    def _reset_state(self) -> None:
        self.cumulative_length = 0
        self.sink_k: Tensor | None = None  # (n, <=n_sink) post-RoPE, verbatim
        self.sink_v: Tensor | None = None
        self.recent_k: Tensor | None = None  # (n, <recent_window+absorb_block) verbatim
        self.recent_v: Tensor | None = None
        self.u_k: Tensor | None = None  # (n, r) fp32, pre-RoPE key basis
        self.b_k: Tensor | None = None  # (r, r) fp32 square-root core
        self.c_k: Tensor | None = None  # (r, <=coord_budget) fp32 coords, current basis
        self.u_v: Tensor | None = None
        self.b_v: Tensor | None = None
        self.c_v: Tensor | None = None
        self._mid_k_cache: Tensor | None = None  # (n, mid_len) storage dtype, post-RoPE
        self._mid_v_cache: Tensor | None = None

    def lazy_initialization(self, key_states: Tensor, value_states: Tensor) -> None:
        if key_states.shape[0] != 1:
            raise NotImplementedError(
                f"BugStreamingLayer supports batch size 1, got {key_states.shape[0]}"
            )
        self.dtype, self.device = key_states.dtype, key_states.device
        self.num_heads = int(key_states.shape[1])
        self.head_dim = int(key_states.shape[3])
        self.n_features = self.num_heads * self.head_dim
        self.is_initialized = True

    # ----------------------------------------------------------- conversions

    def _to_mat(self, x: Tensor) -> Tensor:
        """``(1, H, T, D)`` -> feature-by-token ``(H*D, T)``."""
        h, t, d = x.shape[1], x.shape[2], x.shape[3]
        return x[0].permute(0, 2, 1).reshape(h * d, t)

    def _to_hf(self, mat: Tensor) -> Tensor:
        """Feature-by-token ``(H*D, T)`` -> ``(1, H, T, D)``."""
        t = mat.shape[1]
        return mat.reshape(self.num_heads, self.head_dim, t).permute(0, 2, 1).unsqueeze(0)

    def _mat_rope(self, mat: Tensor, start: int, *, inverse: bool) -> Tensor:
        """(Un-)rotate a feature-by-token block whose columns sit at positions
        ``start .. start+T-1``, in fp32."""
        t = mat.shape[1]
        cos, sin = self.rope.cos_sin(start, t, mat.device)
        htd = mat.to(torch.float32).reshape(self.num_heads, self.head_dim, t).permute(0, 2, 1)
        if inverse:
            out = _rope_unapply(htd, cos, sin, self.rope.scale_sq)
        else:
            out = _rope_apply(htd, cos, sin)
        return out.permute(0, 2, 1).reshape(self.n_features, t)

    # -------------------------------------------------------------- lengths

    def _sink_len(self) -> int:
        return 0 if self.sink_k is None else int(self.sink_k.shape[1])

    def _recent_len(self) -> int:
        return 0 if self.recent_k is None else int(self.recent_k.shape[1])

    def _mid_len(self) -> int:
        return 0 if self.c_k is None else int(self.c_k.shape[1])

    def _post_update_lengths(self, query_length: int) -> tuple[int, int, int]:
        """Predict ``(sink, mid, recent)`` lengths *after* an ``update`` of
        ``query_length`` tokens -- the single source of truth shared by
        ``update()``'s absorb loop and ``get_mask_sizes`` (mask consistency)."""
        if self.cumulative_length == 0:
            # Pre-fill returns the full input.
            return 0, 0, query_length
        recent = self._recent_len() + query_length
        mid = self._mid_len()
        while recent >= self.recent_window + self.absorb_block:
            recent -= self.absorb_block
            if self.lowrank_enabled:
                mid = min(self.coord_budget, mid + self.absorb_block)
        return self._sink_len(), mid, recent

    # ------------------------------------------------------------- BUG step

    def _absorb_block_into_stream(self, m: int) -> None:
        """Graduate the oldest ``m`` recent tokens into the low-rank stream."""
        assert self.recent_k is not None and self.recent_v is not None
        grad_k = self.recent_k[:, :m]
        grad_v = self.recent_v[:, :m]
        self.recent_k = self.recent_k[:, m:]
        self.recent_v = self.recent_v[:, m:]
        self._mid_k_cache = None
        self._mid_v_cache = None
        if not self.lowrank_enabled:
            return  # StreamingLLM mode: graduating tokens are dropped.

        # Positions: recents are the last `_recent_len (pre-slice) ` tokens; the
        # graduating block starts where the middle currently ends.
        grad_start = self.cumulative_length - self._recent_len() - m

        block_k = self._mat_rope(grad_k, grad_start, inverse=True)  # pre-RoPE, fp32
        self.u_k, self.b_k, self.c_k = self._advance(self.u_k, self.b_k, self.c_k, block_k)
        block_v = grad_v.to(torch.float32)
        self.u_v, self.b_v, self.c_v = self._advance(self.u_v, self.b_v, self.c_v, block_v)

    def _advance(
        self, u: Tensor | None, b_core: Tensor | None, coords: Tensor | None, block: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One BUG step + coordinate carry: rotate held coords into the new basis,
        append the block's coords, and enforce the coordinate budget (drop oldest)."""
        u_new, b_new, rot = augmented_bug_step(u, b_core, block, self.rank, theta=self.theta)
        parts = [] if coords is None else [rot @ coords]
        parts.append(u_new.mT @ block)
        c_new = torch.cat(parts, dim=1)
        if c_new.shape[1] > self.coord_budget:
            c_new = c_new[:, -self.coord_budget :]
        return u_new, b_new, c_new

    # -------------------------------------------------------------- update

    def update(
        self, key_states: Tensor, value_states: Tensor, *args: object, **kwargs: object
    ) -> tuple[Tensor, Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        q_len = int(key_states.shape[2])
        if self.cumulative_length == 0:
            return self._prefill(key_states, value_states)
        if q_len != 1:
            raise NotImplementedError(
                "BugStreamingCache supports single-shot pre-fill + one-token decode "
                f"steps only, got q_len={q_len} with {self.cumulative_length} cached "
                "tokens (chunked/continued pre-fill would desync the streaming state)."
            )
        return self._decode_step(key_states, value_states)

    def _prefill(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        """Compress the prompt into the bounded state; return the full K/V (standard
        full pre-fill attention -- the same protocol as every Axis-B baseline)."""
        k_mat = self._to_mat(key_states)
        v_mat = self._to_mat(value_states)
        t = k_mat.shape[1]
        n_sink = min(self.n_sink, t)
        recent = min(self.recent_window, t - n_sink)
        mid = t - n_sink - recent

        self.sink_k = k_mat[:, :n_sink].clone()
        self.sink_v = v_mat[:, :n_sink].clone()
        self.recent_k = k_mat[:, t - recent :].clone()
        self.recent_v = v_mat[:, t - recent :].clone()

        if mid > 0 and self.lowrank_enabled:
            k_pre = self._mat_rope(k_mat[:, n_sink : n_sink + mid], n_sink, inverse=True)
            v_mid = v_mat[:, n_sink : n_sink + mid].to(torch.float32)
            for start in range(0, mid, self.prefill_block_size):
                stop = min(mid, start + self.prefill_block_size)
                self.u_k, self.b_k, self.c_k = self._advance(
                    self.u_k, self.b_k, self.c_k, k_pre[:, start:stop]
                )
                self.u_v, self.b_v, self.c_v = self._advance(
                    self.u_v, self.b_v, self.c_v, v_mid[:, start:stop]
                )
        self.cumulative_length = t
        return key_states, value_states

    def _decode_step(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        assert self.recent_k is not None and self.recent_v is not None
        self.recent_k = torch.cat([self.recent_k, self._to_mat(key_states)], dim=1)
        self.recent_v = torch.cat([self.recent_v, self._to_mat(value_states)], dim=1)
        self.cumulative_length += 1
        while self._recent_len() >= self.recent_window + self.absorb_block:
            self._absorb_block_into_stream(self.absorb_block)
        return self._decode_peek()

    def _decode_peek(self) -> tuple[Tensor, Tensor]:
        """Assemble the currently retained ``[sinks | middle-hat | recent]`` K/V in
        the HF ``(1, H, L, D)`` layout (also used by tests for introspection)."""
        assert self.recent_k is not None and self.recent_v is not None
        parts_k: list[Tensor] = []
        parts_v: list[Tensor] = []
        if self.sink_k is not None and self.sink_k.shape[1] > 0:
            assert self.sink_v is not None
            parts_k.append(self.sink_k)
            parts_v.append(self.sink_v)
        if self._mid_len() > 0:
            self._ensure_mid_cache()
            assert self._mid_k_cache is not None and self._mid_v_cache is not None
            parts_k.append(self._mid_k_cache)
            parts_v.append(self._mid_v_cache)
        parts_k.append(self.recent_k)
        parts_v.append(self.recent_v)
        k_ret = torch.cat(parts_k, dim=1) if len(parts_k) > 1 else parts_k[0]
        v_ret = torch.cat(parts_v, dim=1) if len(parts_v) > 1 else parts_v[0]
        return self._to_hf(k_ret), self._to_hf(v_ret)

    def _ensure_mid_cache(self) -> None:
        """(Re)build the middle reconstruction; only changes on absorb events."""
        if self._mid_k_cache is not None:
            return
        assert self.u_k is not None and self.c_k is not None
        assert self.u_v is not None and self.c_v is not None
        mid_len = self._mid_len()
        mid_start = self.cumulative_length - self._recent_len() - mid_len
        k_pre_hat = self.u_k @ self.c_k  # (n, mid_len) fp32
        k_hat = self._mat_rope(k_pre_hat, mid_start, inverse=False)
        self._mid_k_cache = k_hat.to(self.dtype)
        self._mid_v_cache = (self.u_v @ self.c_v).to(self.dtype)

    # ------------------------------------------------------------ cache API

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Report exactly the K/V length ``update()`` will return this step; the
        offset places the (strictly past) returned block right below the query's
        true position so the causal mask sees it all as visible."""
        sink, mid, recent = self._post_update_lengths(query_length)
        kv_length = sink + mid + recent
        kv_offset = self.cumulative_length + query_length - kv_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """*Cumulative* token count -- keeps true positions advancing."""
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self) -> None:
        self._reset_state()

    # ---------------------------------------------------------- accounting

    def stored_state_numel(self) -> int:
        """Float entries of the *stored* per-layer state (the honest memory)."""
        tensors = (
            self.sink_k,
            self.sink_v,
            self.recent_k,
            self.recent_v,
            self.u_k,
            self.b_k,
            self.c_k,
            self.u_v,
            self.b_v,
            self.c_v,
        )
        return sum(t.numel() for t in tensors if t is not None)

    def workspace_numel(self) -> int:
        """Float entries of the cached middle reconstruction (bounded derived
        state, avoidable by recomputing each step; reported separately)."""
        tensors = (self._mid_k_cache, self._mid_v_cache)
        return sum(t.numel() for t in tensors if t is not None)

    def attended_length(self) -> int:
        """Tokens attention currently sees (sinks + middle + recent)."""
        return self._sink_len() + self._mid_len() + self._recent_len()


class BugStreamingCache(Cache):
    """Model-level constant-memory streaming-BUG cache (one layer per model layer).

    Pass as ``past_key_values`` to ``model.generate(...)`` / ``model(...)``.

    Parameters
    ----------
    model:
        The (Llama-family) model; supplies the layer count and the rotary
        embedding module used for exact un-/re-rotation.
    rank:
        BUG rank cap ``r`` per layer (0 disables the low-rank middle).
    coord_budget:
        Max retained middle-token coordinate columns ``W`` per layer (0 disables
        the low-rank middle; with ``rank=0`` this cache *is* StreamingLLM).
    recent_window, absorb_block, n_sink, theta, prefill_block_size:
        See :class:`BugStreamingLayer`.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        rank: int = 128,
        coord_budget: int = 1024,
        recent_window: int = 64,
        absorb_block: int = 32,
        n_sink: int = 4,
        theta: float | None = None,
        prefill_block_size: int = 128,
    ) -> None:
        base = getattr(model, "model", model)
        rotary = getattr(base, "rotary_emb", None)
        if rotary is None:
            raise ValueError(
                f"could not find a `rotary_emb` module on {type(model).__name__}; "
                "BugStreamingCache needs the model's own rotary embedding for the "
                "exact RoPE round trip"
            )
        rope = _RopeAngles(rotary)
        n_layers = int(model.config.num_hidden_layers)
        layers: list[CacheLayerMixin | LinearAttentionCacheLayerMixin] = [
            BugStreamingLayer(
                rope=rope,
                rank=rank,
                coord_budget=coord_budget,
                recent_window=recent_window,
                absorb_block=absorb_block,
                n_sink=n_sink,
                theta=theta,
                prefill_block_size=prefill_block_size,
            )
            for _ in range(n_layers)
        ]
        super().__init__(layers=layers)

    def _bug_layers(self) -> list[BugStreamingLayer]:
        return [layer for layer in self.layers if isinstance(layer, BugStreamingLayer)]

    def stored_state_numel(self) -> int:
        """Total stored float entries across layers (the constant-memory claim)."""
        return sum(layer.stored_state_numel() for layer in self._bug_layers())

    def workspace_numel(self) -> int:
        """Total cached-reconstruction float entries across layers (bounded)."""
        return sum(layer.workspace_numel() for layer in self._bug_layers())

    def attended_length(self) -> int:
        """Tokens attention sees per layer (identical across layers)."""
        if not self.layers:
            return 0
        first = self.layers[0]
        assert isinstance(first, BugStreamingLayer)
        return first.attended_length()
