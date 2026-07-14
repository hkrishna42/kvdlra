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
  token instead of ``n``). When the coordinate buffer is full, columns are
  *evicted* (see "Week-7 retention" below) -- that is the honest memory bound:
  softmax attention needs per-token information for every attendable token, so
  "constant memory" can only mean bounding the attended set. At matched memory
  the BUG cache retains a ~``n/r`` x longer (but only approximately represented)
  history than whole-token eviction; that trade is exactly the Axis-B experiment.

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

Week-7 retention (``docs/week7-plan.md`` tier 1)
------------------------------------------------
Week 6 measured the deep-horizon loss mechanisms: FIFO coordinate eviction
(adaptive *subspace*, non-adaptive *retention*), erosion by repeated
projection, and the fixed-rank squeeze. Two independent knobs on the same
coordinate buffer attack the first two:

* **(A) adaptive coordinate retention** -- ``retention``:

  - ``"fifo"`` (default): drop the oldest column (the Week-6 behaviour; note
    the Week-7 integrator robustness fix makes reruns *fp-equivalent*, not
    bit-identical, to the archived Week-6 numbers -- rerun baselines in-sweep);
  - ``"attn"``: drop the column with the lowest **EMA-accumulated attention
    mass** (decay ``score_decay`` per decode step -- the O(1)-per-column
    analogue of MorphKV's sum-fusion over its recent window). Scores are
    observed by the :meth:`BugStreamingCache.attach` hook, which recomputes the
    step's aggregated attention row over this cache's returned K with exactly
    MorphKV's machinery; tokens accumulate score while still in the recent
    ring and carry it into the middle at graduation; prompt scores are seeded
    from the last ``recent_window`` prompt queries' causal rows, ``score_decay``
    -collapsed. Without :meth:`~BugStreamingCache.attach` all scores stay zero
    and the stable-sort tiebreak reduces to FIFO (a warning is logged once).
  - ``"energy"``: drop the column with the smallest coordinate norm
    ``||c_s||_2`` (K stream; quantized columns use their stored norm) -- a
    zero-extra-memory proxy: erosion shrinks exactly the columns whose
    out-of-subspace mass the basis has drifted away from.
  - ``"lowrank_surprise"`` (Week-9 D3): drop the column with the smallest
    **low-rank reconstruction residual** ``||k - U U^T k|| / ||k||`` -- i.e. keep
    the *outliers* the low-rank summary cannot reproduce and evict the columns it
    already predicts (redundant). By Pythagoras (``U`` orthonormal) this residual
    is the out-of-subspace half of ``||k||^2`` and ``"energy"`` keeps the
    in-subspace half, so the two are near-anti-correlated -- surprise is the
    *orthogonal complement* of energy, not a variant of it. Unlike ``"energy"``
    the residual is NOT recomputable after absorption (``U C`` lies in the
    subspace, residual == 0), so it is *stored* per column as one fp32 scalar
    (``mid_surprise``/``q_surprise``), a snapshot at graduation -- exactly the
    per-column cost of an ``"attn"`` score.
  - ``"blend"`` (Week-9 D3): combine attention mass and surprise on a common
    **rank** scale (their raw units are incomparable), ``score = surprise_blend *
    rank(attn) + (1 - surprise_blend) * rank(surprise)``. ``surprise_blend=1`` is
    pure ``"attn"``, ``0`` is pure surprise. Needs :meth:`attach` (for the attn
    mass) and stores both a score and a surprise scalar per column (``2r+3``).

  Non-FIFO retention makes the retained middle **non-contiguous in position**,
  so true per-column positions are tracked (``mid_pos``/``q_pos``) and the
  reconstruction is re-rotated at those positions (gathered cos/sin from the
  model's own rotary module). The position arrays and (for ``"attn"``) the
  score buffers are **counted** in ``stored_state_numel`` -- one float
  equivalent per int32 position, one per fp32 score.

* **(D) quantize-instead-of-drop (age-tiered precision)** -- ``quant_bits`` +
  ``quant_budget``: columns evicted from the fp32 coordinate buffer are
  **demoted** to a second tier of up to ``quant_budget`` PolarQuant-quantized
  columns (``quant_bits`` bits/coordinate, the Week-4 machinery already
  validated on BUG coordinates) instead of dropped; the quantized tier evicts
  by the same ``retention`` rule. Across basis updates the tier is carried by
  dequantize -> ``rot @`` -> requantize with **exact norm carry** (the decoded
  direction is renormalized, so magnitudes see only the true rotation-induced
  contraction -- see :meth:`BugStreamingLayer._dequantize`); the direction is
  re-coded per absorb with error bounded by the one-shot PolarQuant distortion
  per event -- whether that per-event jitter compounds visibly over deep
  horizons is exactly what the Week-7 bin curves falsify.
  Accounting (the Week-4 fairness convention, ``scripts/w4_fair.py``): codes
  count ``quant_bits/32`` float-equivalents each (bit-packable), plus one fp32
  norm per column per K/V stream, plus the shared rotation/codebook side
  information counted **once** per cache (:class:`_QuantBank`).

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
Ghadia et al., arXiv:2503.00979 (MorphKV; the attention-scored retention rule).
Zandieh et al., arXiv:2504.19874 (TurboQuant/PolarQuant; the quantized tier).
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, CacheLayerMixin, LinearAttentionCacheLayerMixin
from transformers.models.llama.modeling_llama import rotate_half

from kvdlra.cache.morph_cache import _aggregated_attention_row, _window_attention_rows
from kvdlra.integrators.streaming_torch import augmented_bug_step
from kvdlra.quant import PolarQuant, ProductQuantizer

logger = logging.getLogger(__name__)

__all__ = ["BugStreamingCache", "BugStreamingLayer"]

RETENTION_MODES = ("fifo", "attn", "energy", "lowrank_surprise", "blend")


class _RopeAngles:
    """cos/sin provider for arbitrary position ranges, via the model's own rotary.

    Using the model's ``rotary_emb`` module (rather than re-deriving frequencies)
    guarantees the angles -- including any RoPE scaling (Llama-3 ``llama3`` rope,
    linear scaling, ...) -- are identical to what attention applied to the keys,
    so the un-rotate/re-rotate round trip is exact up to floating-point roundoff.

    All layers absorb on the same schedule and therefore request the same
    position ranges within a decode step, so contiguous-range results are
    memoized (small LRU). Arbitrary (non-contiguous) position vectors -- needed
    when adaptive retention makes the middle non-contiguous -- go through
    :meth:`cos_sin_at` (no memo; rebuilt once per absorb event per layer).
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
        out = self._cos_sin_for(positions)
        self._memo[key] = out
        while len(self._memo) > 8:
            self._memo.popitem(last=False)
        return out

    def cos_sin_at(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """fp32 ``(cos, sin)`` of shape ``(T, head_dim)`` for an arbitrary int64
        position vector (identical rotary module call as :meth:`cos_sin`)."""
        return self._cos_sin_for(positions.unsqueeze(0))

    def _cos_sin_for(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        probe = torch.empty(1, dtype=torch.float32, device=positions.device)
        cos, sin = self._rotary(probe, positions)
        return cos.squeeze(0).to(torch.float32), sin.squeeze(0).to(torch.float32)


class _QuantBank:
    """Shared :class:`PolarQuant` instances keyed by coordinate dimension.

    The rotation ``Pi`` and Lloyd--Max codebook are *data-oblivious* side
    information shared by every layer and both K/V coordinate streams;
    :meth:`side_info_numel` reports their float cost **once** so the matched-
    memory budget can count it honestly (Week-7 guardrail: every variant counts
    ALL its memory).
    """

    def __init__(self, bits: int, seed: int = 0) -> None:
        self.bits = bits
        self.seed = seed
        self._bank: dict[tuple[int, str], PolarQuant] = {}

    def get(self, dim: int, device: torch.device) -> PolarQuant:
        key = (dim, str(device))
        pq = self._bank.get(key)
        if pq is None:
            pq = PolarQuant(dim=dim, bits=self.bits, seed=self.seed, device=device)
            self._bank[key] = pq
        return pq

    def side_info_numel(self) -> int:
        """Float entries of the shared rotations + codebooks (per distinct dim)."""
        seen: set[int] = set()
        total = 0
        for (dim, _), pq in self._bank.items():
            if dim in seen:  # same dim on another device: identical side info
                continue
            seen.add(dim)
            total += int(pq.Pi.numel() + pq.centroids.numel() + pq.boundaries.numel())
        return total


def _rank_normalize(x: Tensor) -> Tensor:
    """Percentile ranks of ``x`` in ``[0, 1]`` (ascending; ties by position via the
    stable double-argsort). Used to blend scores whose raw units are incomparable
    (Week-9 D3 blend retention)."""
    m = int(x.numel())
    if m <= 1:
        return torch.zeros_like(x, dtype=torch.float32)
    ranks = torch.argsort(torch.argsort(x, stable=True), stable=True)
    return ranks.to(torch.float32) / (m - 1)


def _rope_apply(x_htd: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to ``(H, T, D)`` given ``(T, D)`` cos/sin (broadcast over heads)."""
    rot: Tensor = rotate_half(x_htd)  # type: ignore[no-untyped-call]
    return x_htd * cos + rot * sin


def _rope_unapply(x_htd: Tensor, cos: Tensor, sin: Tensor, scale_sq: float) -> Tensor:
    """Exact inverse of :func:`_rope_apply` (rotation transposed, / scaling^2)."""
    rot: Tensor = rotate_half(x_htd)  # type: ignore[no-untyped-call]
    return (x_htd * cos - rot * sin) / scale_sq


def _prompt_seed_scores(
    module: nn.Module,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    window: int,
    decay: float,
) -> Tensor:
    """``(T,)`` score seed over prompt positions: the last ``window`` prompt
    queries' causal attention rows (recomputed exactly, GQA-aggregated, summed
    over KV heads), collapsed with ``decay``-weights so the seed equals what the
    per-step EMA would have accumulated had it run over those steps."""
    cos, sin = position_embeddings
    k_proj = cast(nn.Linear, module.k_proj)
    head_dim = int(cast(int, module.head_dim))
    t = int(hidden_states.shape[1])
    k = k_proj(hidden_states).view(1, t, -1, head_dim).transpose(1, 2)  # (1, H_kv, T, D)
    k = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)  # type: ignore[no-untyped-call]
    rows = _window_attention_rows(module, hidden_states, k, (cos, sin), window)  # (H_kv, w, T)
    mass = rows.sum(dim=0)  # (w, T)
    w = int(mass.shape[0])
    weights = decay ** torch.arange(
        w - 1, -1, -1, dtype=torch.float32, device=mass.device
    )  # newest row gets weight 1
    return (weights.unsqueeze(1) * mass).sum(dim=0)  # (T,)


class BugStreamingLayer(CacheLayerMixin):  # type: ignore[no-untyped-call]
    """One layer's constant-memory streaming-BUG KV state (see module docstring).

    Internally tokens live as feature-by-token matrices ``(n, T)`` with ``n =
    num_kv_heads * head_dim`` (``docs/notes/conventions.md``); conversion to the
    HF layout ``(1, H, T, D)`` happens only at the ``update()`` boundary. The
    retained middle is ``[quantized tier | fp32 tier]`` (each chronological
    under FIFO; position-tracked under adaptive retention).
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
        retention: str = "fifo",
        score_decay: float = 0.97,
        surprise_blend: float = 0.5,
        quant_bits: int | None = None,
        quant_budget: int = 0,
        quant_bank: _QuantBank | None = None,
        hh_budget: int = 0,
        merge: bool = False,
        coord_codebook: ProductQuantizer | None = None,
        anchor_rank: int = 0,
        anchor_seal_absorbs: int = 4,
        code_budget: int = 0,
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
        if retention not in RETENTION_MODES:
            raise ValueError(f"retention must be one of {RETENTION_MODES}, got {retention!r}")
        if not 0.0 < score_decay <= 1.0:
            raise ValueError(f"score_decay must be in (0, 1], got {score_decay}")
        if not 0.0 <= surprise_blend <= 1.0:
            raise ValueError(f"surprise_blend must be in [0, 1], got {surprise_blend}")
        if quant_budget < 0:
            raise ValueError(f"quant_budget must be >= 0, got {quant_budget}")
        if (quant_bits is None) != (quant_budget == 0):
            raise ValueError("quant_bits and quant_budget (> 0) must be set together")
        if quant_bits is not None and not 1 <= quant_bits <= 8:
            raise ValueError(f"quant_bits must be in [1, 8], got {quant_bits}")
        if hh_budget < 0:
            raise ValueError(f"hh_budget must be >= 0, got {hh_budget}")
        if hh_budget > 0 and retention != "attn":
            raise ValueError("hh_budget > 0 (SLASH exact heavy-hitters) requires retention='attn'")
        self.rope = rope
        self.rank = rank
        self.coord_budget = coord_budget
        self.recent_window = recent_window
        self.absorb_block = absorb_block
        self.n_sink = n_sink
        self.theta = theta
        self.prefill_block_size = prefill_block_size
        self.retention = retention
        self.score_decay = score_decay
        self.quant_bits = quant_bits
        self.quant_budget = quant_budget
        self.hh_budget = hh_budget
        self.merge = merge
        # rank=0 / coord_budget=0 => no low-rank middle => StreamingLLM baseline.
        self.lowrank_enabled = rank >= 1 and coord_budget >= 1
        if quant_budget > 0 and not self.lowrank_enabled:
            raise ValueError("quant_budget > 0 requires an enabled low-rank middle")
        if hh_budget > 0 and not self.lowrank_enabled:
            raise ValueError("hh_budget > 0 requires an enabled low-rank middle")
        if merge and not self.lowrank_enabled:
            raise ValueError("merge requires an enabled low-rank middle")
        if merge and quant_budget > 0:
            raise ValueError("merge and the quantized tier are mutually exclusive")
        if merge and retention == "lowrank_surprise":
            # _merge_down does not fuse the mid_surprise buffer (a merged
            # super-column has no single graduation-time residual), so its length
            # would diverge from the coordinates. Week-9 D3 uses neither together.
            raise ValueError("merge and retention='lowrank_surprise' are mutually exclusive")
        # Week-8 CodeBUG: product-quantized coordinate tier against a *frozen*
        # reduced-rank anchor U_ref -- code once at graduation, never rotate the
        # codes (the structural inverse of variant D's per-absorb requantize).
        self.codebook = coord_codebook
        self._coded_mode = coord_codebook is not None
        self.anchor_rank = anchor_rank if anchor_rank > 0 else min(8, rank)
        self.anchor_seal_absorbs = anchor_seal_absorbs
        self.code_budget = code_budget
        if self._coded_mode:
            assert coord_codebook is not None
            if not self.lowrank_enabled:
                raise ValueError("coord_codebook requires an enabled low-rank middle")
            if code_budget < 1:
                raise ValueError("coord_codebook requires code_budget >= 1")
            if quant_bits is not None or quant_budget > 0:
                raise ValueError("coord_codebook and the PolarQuant tier are mutually exclusive")
            if merge:
                raise ValueError("coord_codebook and merge are mutually exclusive")
            if not 1 <= self.anchor_rank <= rank:
                raise ValueError(f"anchor_rank must be in [1, rank={rank}], got {self.anchor_rank}")
            if coord_codebook.dim != self.anchor_rank:
                raise ValueError(
                    f"coord_codebook.dim ({coord_codebook.dim}) must equal "
                    f"anchor_rank ({self.anchor_rank})"
                )
            if anchor_seal_absorbs < 0:
                raise ValueError(f"anchor_seal_absorbs must be >= 0, got {anchor_seal_absorbs}")
        elif code_budget > 0:
            raise ValueError("code_budget > 0 requires coord_codebook")
        # The second tier's column budget: PolarQuant (variant D) or CodeBUG.
        self._second_tier_budget = quant_budget + code_budget
        # Calibration hook: when set to a list, the exact anchor-frame coordinate
        # vectors that would be coded are appended here (pre-quantization), so
        # ``scripts/calibrate_codebook.py`` can fit the PQ on the true deployment
        # distribution. None in normal operation (zero overhead).
        self._calib_sink: list[Tensor] | None = None
        # Ablation ONLY (``scripts/w8_carry_drift.py``): when True, the coded tier
        # is re-anchored to the live basis and RE-CODED every absorb -- the
        # variant-D analog that SHOULD compound. The falsifiable OFF control that
        # proves the frozen-anchor ON path does not. Never set in a real run.
        self._debug_recode_coded = False
        # Drift experiment ONLY: when a dict, each coded column's true rank-r
        # feature vector (K stream) is recorded at graduation keyed by position,
        # so the readout reconstruction error can be measured. None in prod.
        self._drift_truth: dict[int, Tensor] | None = None
        # SLASH exact heavy-hitter tier (Week-7 dominance program).
        self.hh_enabled = hh_budget >= 1 and self.lowrank_enabled
        self._quant_bank = (
            quant_bank
            if quant_bank is not None
            else (_QuantBank(quant_bits) if quant_bits is not None else None)
        )
        # Adaptive retention needs true per-column positions (the middle stops
        # being contiguous); attention scoring additionally needs score buffers.
        self.track_positions = self.lowrank_enabled and (retention != "fifo" or merge)
        self.track_scores = self.lowrank_enabled and retention in ("attn", "blend")
        self.surprise_blend = surprise_blend
        # Week-9 D3: low-rank surprise retention keeps the highest-residual
        # (outlier) columns and evicts the ones the basis already reconstructs
        # (redundant). Unlike ``energy`` (which reads ``||c_s||`` from the stored
        # coordinates on the fly, zero extra bytes), the residual ``||k - U U^T
        # k||`` is NOT recomputable after absorption (``U C`` lies in the
        # subspace, residual == 0), so it must be *stored* per column -- one fp32
        # scalar, exactly parallel to ``mid_score``. Counted in
        # ``stored_state_numel`` (the honest-memory guardrail).
        self.track_surprise = self.lowrank_enabled and retention in ("lowrank_surprise", "blend")
        # Correlation probe (``scripts/w9_surprise.py`` GO/NO-GO gate): when True,
        # surprise is tracked under *any* retention so a ``retention="attn"`` run
        # can log (attention-mass, surprise) pairs; when ``_probe_sink`` is a list,
        # the full (score, surprise, energy) column arrays are recorded at each
        # eviction event. Both off (no overhead) in normal operation.
        self._probe_surprise = False
        self._probe_sink: list[tuple[Tensor, Tensor, Tensor]] | None = None
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
        # Quantized age tier (Week-7 D / Week-8 CodeBUG): codes/norms, rows =
        # columns. Variant D: PolarQuant, shape ``(Wq, r)`` in the *live* basis,
        # rotated per absorb. CodeBUG: ProductQuantizer, shape ``(W, M)`` in the
        # *frozen* anchor ``u_ref``, never rotated.
        self.qk_codes: Tensor | None = None  # (Wq, r) or (W, M) uint8 codes, K
        self.qk_norm: Tensor | None = None  # (Wq,) fp32
        self.qv_codes: Tensor | None = None
        self.qv_norm: Tensor | None = None
        # CodeBUG frozen anchor: the reduced-rank basis coded columns live in.
        self.u_ref_k: Tensor | None = None  # (n, r_a) fp32, frozen at seal
        self.u_ref_v: Tensor | None = None
        self.sealed = False
        self._n_absorbs = 0
        # Retention bookkeeping (Week-7 A): positions + EMA attention scores.
        self.mid_pos: Tensor | None = None  # (f_len,) int64, fp32-tier positions
        self.q_pos: Tensor | None = None  # (q_len,) int64, quant-tier positions
        self.mid_score: Tensor | None = None  # (f_len,) fp32 EMA attention mass
        self.q_score: Tensor | None = None  # (q_len,) fp32
        # Retention bookkeeping (Week-9 D3): per-column low-rank surprise (the
        # residual ||k - U U^T k|| at graduation, normalized -- a snapshot, never
        # recomputed as the basis drifts, the honest analogue of how mid_score
        # accumulates but is never recomputed).
        self.mid_surprise: Tensor | None = None  # (f_len,) fp32
        self.q_surprise: Tensor | None = None  # (q_len,) fp32
        # SLASH exact heavy-hitter tier (Week-7 dominance): verbatim K/V + posns.
        self.hh_k: Tensor | None = None  # (n, <=hh_budget) post-RoPE, verbatim
        self.hh_v: Tensor | None = None
        self.hh_pos: Tensor | None = None  # (hh_len,) int64 true positions
        self.hh_score: Tensor | None = None  # (hh_len,) fp32 EMA attention mass
        # Hierarchical merge (Week-7 dominance): token count each fp32 column
        # represents (1 = a single token; >1 = a merged centroid super-column).
        self.mid_weight: Tensor | None = None  # (f_len,) fp32
        self.ring_score: Tensor | None = None  # (recent_len,) fp32
        self._seen_observation = False
        self._warned_unattached = False
        self._mid_k_cache: Tensor | None = None  # (n, mid_len) storage dtype, post-RoPE
        self._mid_v_cache: Tensor | None = None
        # Week-10 harness mode (default "normal" preserves every existing call
        # site / the q_len==1 decode invariant). "score": non-mutating frozen
        # continuation-window forward (Phase 1b). "ingest": chunked pre-fill
        # (Phase 3). Set only by BugStreamingCache context managers.
        self._mode = "normal"

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
        cos, sin = self.rope.cos_sin(start, mat.shape[1], mat.device)
        return self._mat_rope_with(mat, cos, sin, inverse=inverse)

    def _mat_rope_at(self, mat: Tensor, positions: Tensor, *, inverse: bool) -> Tensor:
        """(Un-)rotate a feature-by-token block at *arbitrary* per-column positions."""
        cos, sin = self.rope.cos_sin_at(positions.to(mat.device))
        return self._mat_rope_with(mat, cos, sin, inverse=inverse)

    def _mat_rope_with(self, mat: Tensor, cos: Tensor, sin: Tensor, *, inverse: bool) -> Tensor:
        t = mat.shape[1]
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

    def _f_len(self) -> int:
        """fp32 coordinate columns currently held."""
        return 0 if self.c_k is None else int(self.c_k.shape[1])

    def _q_len(self) -> int:
        """Second-tier (quantized / coded) coordinate columns currently held."""
        return 0 if self.qk_codes is None else int(self.qk_codes.shape[0])

    def _hh_len(self) -> int:
        """Exact heavy-hitter tokens currently held (SLASH)."""
        return 0 if self.hh_k is None else int(self.hh_k.shape[1])

    def _mid_len(self) -> int:
        return self._hh_len() + self._q_len() + self._f_len()

    def _post_update_lengths(self, query_length: int) -> tuple[int, int, int]:
        """Predict ``(sink, mid, recent)`` lengths *after* an ``update`` of
        ``query_length`` tokens -- the single source of truth shared by
        ``update()``'s absorb loop and ``get_mask_sizes`` (mask consistency)."""
        if self.cumulative_length == 0:
            # Pre-fill returns the full input.
            return 0, 0, query_length
        recent = self._recent_len() + query_length
        mid = self._mid_len()
        mid_cap = self.coord_budget + self._second_tier_budget + self.hh_budget
        while recent >= self.recent_window + self.absorb_block:
            recent -= self.absorb_block
            if self.lowrank_enabled:
                mid = min(mid_cap, mid + self.absorb_block)
        return self._sink_len(), mid, recent

    # ------------------------------------------------------------- BUG step

    def _absorb_block_into_stream(self, m: int) -> None:
        """Graduate the oldest ``m`` recent tokens into the low-rank stream."""
        assert self.recent_k is not None and self.recent_v is not None
        grad_k = self.recent_k[:, :m]
        grad_v = self.recent_v[:, :m]
        self.recent_k = self.recent_k[:, m:]
        self.recent_v = self.recent_v[:, m:]
        grad_score: Tensor | None = None
        if self.track_scores and self.ring_score is not None:
            grad_score = self.ring_score[:m]
            self.ring_score = self.ring_score[m:]
        self._mid_k_cache = None
        self._mid_v_cache = None
        if not self.lowrank_enabled:
            return  # StreamingLLM mode: graduating tokens are dropped.

        # Positions: recents are the last `_recent_len (post-slice)` tokens; the
        # graduating block starts where the middle currently ends.
        grad_start = self.cumulative_length - self._recent_len() - m
        grad_pos = torch.arange(grad_start, grad_start + m, dtype=torch.int64, device=grad_k.device)
        if self.hh_enabled:
            self._absorb_block_slash(grad_k, grad_v, grad_pos, grad_score)
            return
        block_k = self._mat_rope(grad_k, grad_start, inverse=True)  # pre-RoPE, fp32
        block_v = grad_v.to(torch.float32)
        self._absorb_columns(block_k, block_v, grad_pos, grad_score)

    def _absorb_block_slash(
        self, grad_k: Tensor, grad_v: Tensor, grad_pos: Tensor, grad_score: Tensor | None
    ) -> None:
        """SLASH (Week-7 dominance program): split the graduating block + the
        current exact heavy-hitter tier into (i) the top-``hh_budget`` tokens by
        recent-attention score, kept **verbatim** (post-RoPE K, raw V -- exactly
        like sinks), and (ii) the rest, absorbed into the low-rank tail via
        :meth:`_absorb_columns`. Because the exact peaks never pass through
        ``augmented_bug_step``, the tracked rank-``r`` basis summarizes the
        *outlier-removed residual* spectrum (a robust-PCA decomposition matched
        to attention's heavy-tailed structure) -- neither pure eviction nor pure
        BUG has both. Demoted former-heavy-hitters re-enter the tail at their
        true (non-contiguous) positions."""
        sc = (
            grad_score.to(torch.float32)
            if grad_score is not None
            else torch.zeros(grad_k.shape[1], dtype=torch.float32, device=grad_k.device)
        )
        # Candidate pool = current exact tier + graduating block (all post-RoPE).
        if self.hh_k is not None:
            assert self.hh_v is not None and self.hh_pos is not None and self.hh_score is not None
            cand_k = torch.cat([self.hh_k, grad_k], dim=1)
            cand_v = torch.cat([self.hh_v, grad_v], dim=1)
            cand_pos = torch.cat([self.hh_pos, grad_pos])
            cand_score = torch.cat([self.hh_score, sc])
        else:
            cand_k, cand_v, cand_pos, cand_score = grad_k, grad_v, grad_pos, sc
        n_cand = int(cand_k.shape[1])
        keep_n = min(self.hh_budget, n_cand)
        # Highest recent-attention scores stay exact; ties fall back to recency.
        order = torch.argsort(cand_score, stable=True, descending=True)
        keep = order[:keep_n].sort().values  # chronological for clean assembly
        demote = order[keep_n:].sort().values
        self.hh_k = cand_k[:, keep].clone()
        self.hh_v = cand_v[:, keep].clone()
        self.hh_pos = cand_pos[keep].clone()
        self.hh_score = cand_score[keep].clone()
        if demote.numel() > 0:
            dem_pos = cand_pos[demote]
            dem_k_pre = self._mat_rope_at(cand_k[:, demote], dem_pos, inverse=True)
            dem_v = cand_v[:, demote].to(torch.float32)
            self._absorb_columns(dem_k_pre, dem_v, dem_pos, cand_score[demote])

    def _absorb_columns(
        self, block_k: Tensor, block_v: Tensor, positions: Tensor, scores: Tensor | None
    ) -> None:
        """One augmented BUG step + coordinate carry + budget enforcement for a
        block of ``m`` new columns at the given ``positions`` (``(m,)`` int64; may
        be non-contiguous when heavy-hitters are demoted back into the tail)."""
        m = int(block_k.shape[1])
        if m == 0:
            return
        # Week-9 D3: low-rank surprise = out-of-subspace fraction of the incoming
        # K columns, measured against the basis built from *strictly older*
        # history (pre-step u_k -- the true novelty signal). By Pythagoras (u_k
        # orthonormal): ||k||^2 = ||u_k^T k||^2 + ||k - u_k u_k^T k||^2, so the
        # residual norm is sqrt(||k||^2 - ||c_old||^2). Normalized to [0, 1] =
        # sin of the angle between k and span(u_k) (scale-free across columns).
        surprise: Tensor | None = None
        if self.track_surprise or self._probe_surprise:
            k_norm = block_k.norm(dim=0)
            if self.u_k is None:  # first absorb: nothing older to predict from
                surprise = torch.ones(m, dtype=torch.float32, device=block_k.device)
            else:
                c_old = self.u_k.mT @ block_k  # (r, m) coords in the OLD basis
                resid = (k_norm**2 - c_old.norm(dim=0) ** 2).clamp_min(0.0).sqrt()
                surprise = (resid / k_norm.clamp_min(1e-12)).to(torch.float32)
        self.u_k, self.b_k, rot_k = augmented_bug_step(
            self.u_k, self.b_k, block_k, self.rank, theta=self.theta
        )
        self.u_v, self.b_v, rot_v = augmented_bug_step(
            self.u_v, self.b_v, block_v, self.rank, theta=self.theta
        )
        # Carry held fp32 coordinates into the new basis.
        if self.c_k is not None:
            assert self.c_v is not None
            self.c_k = rot_k @ self.c_k
            self.c_v = rot_v @ self.c_v
        # Second tier: variant D re-expresses its codes into the new basis
        # (dequantize -> rot @ -> requantize, the compounding path). CodeBUG does
        # NOT: coded columns live in the *frozen* anchor ``u_ref`` and are never
        # rotated, so no per-absorb quantization noise is re-injected.
        if self._q_len() > 0:
            if not self._coded_mode:
                self._rotate_quant_tier(rot_k, rot_v)
            elif self._debug_recode_coded and self.sealed:
                self._recode_coded_tier()  # ablation-only variant-D analog
        # Append the graduating block's coordinates (fp32 tier).
        new_ck = self.u_k.mT @ block_k
        new_cv = self.u_v.mT @ block_v
        self.c_k = new_ck if self.c_k is None else torch.cat([self.c_k, new_ck], dim=1)
        self.c_v = new_cv if self.c_v is None else torch.cat([self.c_v, new_cv], dim=1)
        if self.track_positions:
            pos = positions.to(dtype=torch.int64, device=block_k.device)
            self.mid_pos = pos if self.mid_pos is None else torch.cat([self.mid_pos, pos])
        if self.track_scores:
            sc = (
                scores.to(torch.float32)
                if scores is not None
                else torch.zeros(m, dtype=torch.float32, device=block_k.device)
            )
            self.mid_score = sc if self.mid_score is None else torch.cat([self.mid_score, sc])
        if (self.track_surprise or self._probe_surprise) and surprise is not None:
            self.mid_surprise = (
                surprise if self.mid_surprise is None else torch.cat([self.mid_surprise, surprise])
            )
        if self.merge:
            w = torch.ones(m, dtype=torch.float32, device=block_k.device)
            self.mid_weight = w if self.mid_weight is None else torch.cat([self.mid_weight, w])
        # CodeBUG: seal the frozen anchor after a short decode warmup, so U_ref
        # reflects decode statistics (not just the prompt). Sealed once, then the
        # coded tier is opened for demotion in the enforce step below.
        self._n_absorbs += 1
        if (
            self._coded_mode
            and not self.sealed
            and self._n_absorbs >= self.anchor_seal_absorbs
            and self.u_k is not None
            and int(self.u_k.shape[1]) >= self.anchor_rank
        ):
            self._seal_anchor()
        self._enforce_budgets()

    # ------------------------------------------------- retention + quant tier

    def _split_indices(self, n_out: int, *, tier: str) -> tuple[Tensor, Tensor]:
        """``(evict_idx, keep_idx)`` (both ascending) for ``n_out`` evictions from
        ``tier`` ("fp32" or "quant") under the configured retention rule. Ties
        (e.g. all-zero scores) fall back to FIFO via the stable sort."""
        length = self._f_len() if tier == "fp32" else self._q_len()
        device = self.c_k.device if self.c_k is not None else torch.device("cpu")
        if self.retention == "fifo":
            evict = torch.arange(n_out, device=device)
            keep = torch.arange(n_out, length, device=device)
            return evict, keep
        if self.retention == "attn":
            scores = self.mid_score if tier == "fp32" else self.q_score
            assert scores is not None
            # Pre-fill overflow evicts before the attach() hook can seed scores
            # (update() runs inside the attention forward; the hook fires after)
            # -- that is documented FIFO-by-design, so only warn for score-less
            # evictions during decode (cumulative_length > 0).
            if (
                not self._seen_observation
                and not self._warned_unattached
                and self.cumulative_length > 0
            ):
                logger.warning(
                    "retention='attn' evicting with no recorded attention scores "
                    "(cache not attach()ed?) -- falling back to FIFO order"
                )
                self._warned_unattached = True
        elif self.retention == "lowrank_surprise":
            # Evict the LOWEST-surprise (best-reconstructed = most redundant)
            # columns; the ascending argsort below keeps the high-residual
            # outliers the low-rank summary cannot reproduce.
            scores = self.mid_surprise if tier == "fp32" else self.q_surprise
            assert scores is not None
        elif self.retention == "blend":
            # Week-9 D3 blend: combine attention mass and surprise on a common
            # RANK scale (their raw units are incomparable -- attn mass is a
            # heavy-tailed growing EMA, surprise a bounded feature-space ratio).
            # score = alpha * rank(attn) + (1 - alpha) * rank(surprise); evict the
            # lowest blended rank. alpha=1 => pure attn, alpha=0 => pure surprise.
            attn = self.mid_score if tier == "fp32" else self.q_score
            surp = self.mid_surprise if tier == "fp32" else self.q_surprise
            assert attn is not None and surp is not None
            a = self.surprise_blend
            scores = a * _rank_normalize(attn) + (1.0 - a) * _rank_normalize(surp)
        else:  # "energy"
            if tier == "fp32":
                assert self.c_k is not None
                scores = self.c_k.norm(dim=0)
            else:
                assert self.qk_norm is not None
                scores = self.qk_norm
        order = torch.argsort(scores, stable=True)  # ascending; ties keep age order
        evict = order[:n_out].sort().values
        keep = order[n_out:].sort().values
        return evict, keep

    def _enforce_budgets(self) -> None:
        """Evict down to ``coord_budget`` (demoting to the quantized tier when
        enabled) and then the quantized tier down to ``quant_budget``."""
        f_over = self._f_len() - self.coord_budget
        if f_over > 0 and self.merge:
            self._merge_down(f_over)
            f_over = 0
        if f_over > 0:
            assert self.c_k is not None and self.c_v is not None
            self._probe_record()
            evict, keep = self._split_indices(f_over, tier="fp32")
            out_ck = self.c_k[:, evict]
            out_cv = self.c_v[:, evict]
            self.c_k = self.c_k[:, keep]
            self.c_v = self.c_v[:, keep]
            out_pos: Tensor | None = None
            out_score: Tensor | None = None
            out_surprise: Tensor | None = None
            if self.track_positions:
                assert self.mid_pos is not None
                out_pos = self.mid_pos[evict]
                self.mid_pos = self.mid_pos[keep]
            if self.track_scores:
                assert self.mid_score is not None
                out_score = self.mid_score[evict]
                self.mid_score = self.mid_score[keep]
            if self.track_surprise:
                assert self.mid_surprise is not None
                out_surprise = self.mid_surprise[evict]
                self.mid_surprise = self.mid_surprise[keep]
            elif self._probe_surprise and self.mid_surprise is not None:
                self.mid_surprise = self.mid_surprise[keep]
            if self._coded_mode:
                # Demote into the CodeBUG coded tier -- but only once the anchor
                # is sealed; during the short pre-seal warmup demoted columns are
                # dropped (the fp32 budget covers the warmup window).
                if self.sealed:
                    self._append_coded(out_ck, out_cv, out_pos, out_score, out_surprise)
            elif self.quant_bits is not None:
                self._append_quant(out_ck, out_cv, out_pos, out_score, out_surprise)
        q_over = self._q_len() - self._second_tier_budget
        if q_over > 0:
            assert self.qk_codes is not None and self.qk_norm is not None
            assert self.qv_codes is not None and self.qv_norm is not None
            evict, keep = self._split_indices(q_over, tier="quant")
            self.qk_codes = self.qk_codes[keep]
            self.qk_norm = self.qk_norm[keep]
            self.qv_codes = self.qv_codes[keep]
            self.qv_norm = self.qv_norm[keep]
            if self.track_positions:
                assert self.q_pos is not None
                self.q_pos = self.q_pos[keep]
            if self.track_scores:
                assert self.q_score is not None
                self.q_score = self.q_score[keep]
            if self.track_surprise:
                assert self.q_surprise is not None
                self.q_surprise = self.q_surprise[keep]

    def _probe_record(self) -> None:
        """GO/NO-GO correlation probe: snapshot the fp32 tier's (attention-mass,
        surprise, energy) per-column arrays at an eviction event, for measuring
        whether surprise is complementary to attention mass or redundant. Active
        only when a ``_probe_sink`` list is attached (zero overhead otherwise)."""
        if self._probe_sink is None or self.c_k is None:
            return
        if self.mid_score is None or self.mid_surprise is None:
            return
        energy = self.c_k.norm(dim=0)
        self._probe_sink.append((self.mid_score.clone(), self.mid_surprise.clone(), energy.clone()))

    def _merge_down(self, n_out: int) -> None:
        """Hierarchical **dyadic** merge (Week-7 dominance): instead of *dropping*
        the oldest ``n_out`` fp32 coordinate columns, collapse ``n_out`` adjacent
        **equal-weight** pairs into weighted centroids -- so the middle keeps an
        unbounded history at geometrically decaying resolution (recent = per-token,
        old = coarse super-columns with weights ``..4,2,1``, a bounded count per
        level like a binary counter), constant memory. Merging only equal-weight
        neighbours keeps the pyramid balanced (a Compressive-Transformer / exponential
        -histogram schedule) rather than letting one blob absorb all history.

        Centroids are token-count-weighted -- a uniform-attention approximation
        within a group; the log-count softmax correction that would make merged
        *keys* exact is not applied (a documented, falsifiable approximation).
        The reconstruction ``U C`` and position re-rotation treat a merged column
        like any other, at its count-weighted mean position."""
        assert self.c_k is not None and self.c_v is not None and self.mid_weight is not None
        for _ in range(n_out):
            length = self._f_len()
            if length < 2:
                return
            w = self.mid_weight
            eq = (w[:-1] == w[1:]).nonzero(as_tuple=False).flatten()
            i = int(eq[0].item()) if eq.numel() > 0 else 0  # oldest equal pair, else oldest
            wsum = w[i] + w[i + 1]
            wi, wj = w[i].clone(), w[i + 1].clone()

            def _merge_col(
                c: Tensor, j: int = i, s: Tensor = wsum, a: Tensor = wi, b: Tensor = wj
            ) -> Tensor:
                cen = (c[:, j : j + 1] * a + c[:, j + 1 : j + 2] * b) / s
                return torch.cat([c[:, :j], cen, c[:, j + 2 :]], dim=1)

            self.c_k = _merge_col(self.c_k)
            self.c_v = _merge_col(self.c_v)
            self.mid_weight = torch.cat([w[:i], wsum.reshape(1), w[i + 2 :]])
            if self.track_positions:
                assert self.mid_pos is not None
                p = self.mid_pos
                mp = ((p[i] * w[i] + p[i + 1] * w[i + 1]) / wsum).round().to(torch.int64)
                self.mid_pos = torch.cat([p[:i], mp.reshape(1), p[i + 2 :]])
            if self.track_scores:
                assert self.mid_score is not None
                sc = self.mid_score
                self.mid_score = torch.cat([sc[:i], (sc[i] + sc[i + 1]).reshape(1), sc[i + 2 :]])

    def _quantize_cols(self, cols: Tensor, stream: str) -> tuple[Tensor, Tensor]:
        """PolarQuant coordinate columns ``(r, m)`` -> ``(codes (m, r) uint8,
        norms (m,) fp32)``. ``stream`` is only for error messages."""
        assert self._quant_bank is not None, f"quant bank missing ({stream})"
        pq = self._quant_bank.get(int(cols.shape[0]), cols.device)
        codes, norm = pq.quantize(cols.mT)  # rows = vectors
        return codes.to(torch.uint8), norm.squeeze(-1).to(torch.float32)

    def _dequantize(self, codes: Tensor, norm: Tensor) -> Tensor:
        """Inverse of :meth:`_quantize_cols`: ``(m, r)+(m,)`` -> columns ``(r, m)``.

        The decoded *direction* is renormalized to unit length before scaling,
        so every reconstructed column has Euclidean norm exactly equal to its
        stored ``norm``. Plain ``PolarQuant.dequantize`` scales the raw centroid
        vector, whose norm is ``||centroids[codes]|| != 1`` (Lloyd--Max
        centroids do not lie on the sphere) -- under the per-absorb
        dequantize -> rot -> requantize carry that mismatch compounds into
        exponential per-column norm drift (adversarial-review finding, Week 7);
        renormalizing makes the norm carry exact and leaves only the bounded
        per-event direction error."""
        assert self._quant_bank is not None
        pq = self._quant_bank.get(int(codes.shape[1]), codes.device)
        unit = torch.ones(codes.shape[0], 1, dtype=norm.dtype, device=norm.device)
        direction = pq.dequantize(codes.to(torch.int64), unit)  # (m, r), ~unit rows
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        out: Tensor = (direction * norm.unsqueeze(-1)).mT
        return out

    def _append_quant(
        self,
        ck: Tensor,
        cv: Tensor,
        pos: Tensor | None,
        score: Tensor | None,
        surprise: Tensor | None = None,
    ) -> None:
        """Demote evicted fp32 coordinate columns into the quantized tier."""
        codes_k, norm_k = self._quantize_cols(ck, "K")
        codes_v, norm_v = self._quantize_cols(cv, "V")
        if self.qk_codes is None:
            self.qk_codes, self.qk_norm = codes_k, norm_k
            self.qv_codes, self.qv_norm = codes_v, norm_v
        else:
            assert self.qk_norm is not None and self.qv_codes is not None
            assert self.qv_norm is not None
            self.qk_codes = torch.cat([self.qk_codes, codes_k])
            self.qk_norm = torch.cat([self.qk_norm, norm_k])
            self.qv_codes = torch.cat([self.qv_codes, codes_v])
            self.qv_norm = torch.cat([self.qv_norm, norm_v])
        if self.track_positions:
            assert pos is not None
            self.q_pos = pos if self.q_pos is None else torch.cat([self.q_pos, pos])
        if self.track_scores:
            assert score is not None
            self.q_score = score if self.q_score is None else torch.cat([self.q_score, score])
        if self.track_surprise:
            assert surprise is not None
            self.q_surprise = (
                surprise if self.q_surprise is None else torch.cat([self.q_surprise, surprise])
            )

    def _rotate_quant_tier(self, rot_k: Tensor, rot_v: Tensor) -> None:
        """Carry the quantized tier across a basis update: dequantize -> ``rot @``
        -> requantize. Column *norms* are carried exactly (:meth:`_dequantize`
        renormalizes the decoded direction, so the requantized norm is
        ``norm_old * ||rot @ u_hat||`` -- the true rotation-induced contraction,
        with no quantizer-induced drift). The *direction* is re-coded each
        absorb with error bounded by the one-shot PolarQuant distortion per
        event; whether that per-event jitter compounds visibly over deep
        horizons is exactly what the Week-7 bin curves falsify."""
        assert self.qk_codes is not None and self.qk_norm is not None
        assert self.qv_codes is not None and self.qv_norm is not None
        ck = rot_k @ self._dequantize(self.qk_codes, self.qk_norm)
        cv = rot_v @ self._dequantize(self.qv_codes, self.qv_norm)
        self.qk_codes, self.qk_norm = self._quantize_cols(ck, "K")
        self.qv_codes, self.qv_norm = self._quantize_cols(cv, "V")

    # -------------------------------------------------- CodeBUG frozen anchor

    def _seal_anchor(self) -> None:
        """Freeze the CodeBUG anchor: snapshot the top-``anchor_rank`` columns of
        the current live basis as the read-only ``u_ref`` the coded tier lives
        in. Columns are energy-sorted (``b = diag(sigma)`` descending, nothing
        rotates it), so the leading ``r_a`` capture the dominant subspace; a
        subset of orthonormal columns is itself orthonormal, so no
        re-orthonormalization is needed. Called once, after
        ``anchor_seal_absorbs`` absorbs."""
        assert self.u_k is not None and self.u_v is not None
        # Guaranteed by the seal gate: the basis has >= anchor_rank columns.
        ra = self.anchor_rank
        self.u_ref_k = self.u_k[:, :ra].clone()
        self.u_ref_v = self.u_v[:, :ra].clone()
        self.sealed = True

    def _append_coded(
        self,
        ck: Tensor,
        cv: Tensor,
        pos: Tensor | None,
        score: Tensor | None,
        surprise: Tensor | None = None,
    ) -> None:
        """Demote evicted fp32 columns into the CodeBUG coded tier: re-express
        each from the *live* basis into the *frozen* anchor **once**
        (``c_ref = u_ref^T (u_live @ c_live)``), PQ-encode, and append. This is
        the only point a coded column is ever encoded; thereafter it is inert
        (never rotated) -- the structural non-compounding guarantee."""
        assert self.codebook is not None
        assert self.u_ref_k is not None and self.u_ref_v is not None
        assert self.u_k is not None and self.u_v is not None
        ref_k = (self.u_ref_k.mT @ self.u_k) @ ck  # (r_a, m) anchor coordinates
        ref_v = (self.u_ref_v.mT @ self.u_v) @ cv
        if self._calib_sink is not None:  # calibration: record true anchor coords
            self._calib_sink.append(ref_k.mT.detach().to("cpu", torch.float32))
            self._calib_sink.append(ref_v.mT.detach().to("cpu", torch.float32))
        if self._drift_truth is not None and pos is not None:  # drift experiment
            xk = (self.u_k @ ck).detach().to("cpu", torch.float32)  # (n, m) rank-r truth
            for j, p in enumerate(pos.tolist()):
                self._drift_truth[int(p)] = xk[:, j]
        codes_k, norm_k = self.codebook.quantize(ref_k.mT)  # (m, M) uint8, (m, 1)
        codes_v, norm_v = self.codebook.quantize(ref_v.mT)
        codes_k = codes_k.to(dtype=torch.uint8, device=self.device)
        codes_v = codes_v.to(dtype=torch.uint8, device=self.device)
        norm_k = norm_k.squeeze(-1).to(dtype=torch.float32, device=self.device)
        norm_v = norm_v.squeeze(-1).to(dtype=torch.float32, device=self.device)
        if self.qk_codes is None:
            self.qk_codes, self.qk_norm = codes_k, norm_k
            self.qv_codes, self.qv_norm = codes_v, norm_v
        else:
            assert self.qk_norm is not None and self.qv_codes is not None
            assert self.qv_norm is not None
            self.qk_codes = torch.cat([self.qk_codes, codes_k])
            self.qk_norm = torch.cat([self.qk_norm, norm_k])
            self.qv_codes = torch.cat([self.qv_codes, codes_v])
            self.qv_norm = torch.cat([self.qv_norm, norm_v])
        if self.track_positions:
            assert pos is not None
            self.q_pos = pos if self.q_pos is None else torch.cat([self.q_pos, pos])
        if self.track_scores:
            assert score is not None
            self.q_score = score if self.q_score is None else torch.cat([self.q_score, score])
        if self.track_surprise:
            assert surprise is not None
            self.q_surprise = (
                surprise if self.q_surprise is None else torch.cat([self.q_surprise, surprise])
            )

    def _decode_coded(self, codes: Tensor, norm: Tensor | None) -> Tensor:
        """Inverse of the coded-tier encode: ``(m, M)`` codes -> ``(r_a, m)``
        anchor coordinates. In ``normalize`` mode the decoded unit direction is
        scaled by the exact stored norm (magnitude carried losslessly); otherwise
        the raw centroids carry magnitude."""
        assert self.codebook is not None
        idx = codes.to(torch.int64)
        if self.codebook.normalize:
            assert norm is not None
            recon = self.codebook.dequantize(idx, norm.unsqueeze(-1))
        else:
            recon = self.codebook.decode(idx)
        return recon.mT.to(device=self.device)  # (r_a, m)

    def _recode_coded_tier(self) -> None:
        """ABLATION ONLY (``_debug_recode_coded``): the variant-D analog. Decode
        the coded tier, re-anchor ``u_ref`` to the current live basis, and
        RE-CODE every column against it -- re-injecting PQ distortion every
        absorb. This is the falsifiable OFF control that should compound; the
        real (frozen-anchor) path never calls it."""
        assert self.codebook is not None and self.u_ref_k is not None
        assert self.u_ref_v is not None and self.u_k is not None and self.u_v is not None
        assert self.qk_codes is not None and self.qv_codes is not None
        full_k = self.u_ref_k @ self._decode_coded(self.qk_codes, self.qk_norm)  # (n, W)
        full_v = self.u_ref_v @ self._decode_coded(self.qv_codes, self.qv_norm)
        ra = self.anchor_rank
        self.u_ref_k = self.u_k[:, :ra].clone()
        self.u_ref_v = self.u_v[:, :ra].clone()
        ref_k = self.u_ref_k.mT @ full_k  # (r_a, W) in the re-anchored frame
        ref_v = self.u_ref_v.mT @ full_v
        codes_k, norm_k = self.codebook.quantize(ref_k.mT)
        codes_v, norm_v = self.codebook.quantize(ref_v.mT)
        self.qk_codes = codes_k.to(dtype=torch.uint8, device=self.device)
        self.qv_codes = codes_v.to(dtype=torch.uint8, device=self.device)
        self.qk_norm = norm_k.squeeze(-1).to(dtype=torch.float32, device=self.device)
        self.qv_norm = norm_v.squeeze(-1).to(dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------ scoring

    def observe_attention(self, mass: Tensor) -> None:
        """EMA-update retention scores from one decode step's attention mass over
        the returned ``[sinks | hh | quant | fp32 | recent]`` columns (``(L,)``,
        aggregated over all query heads). Called by the attach() hook."""
        if not self.track_scores:
            return
        s, hh = self._sink_len(), self._hh_len()
        q, f, rlen = self._q_len(), self._f_len(), self._recent_len()
        if mass.shape != (s + hh + q + f + rlen,):
            raise ValueError(
                f"attention mass must have length {s + hh + q + f + rlen} "
                f"(sinks {s} + hh {hh} + quant {q} + fp32 {f} + recent {rlen}), "
                f"got {tuple(mass.shape)}"
            )
        mass = mass.to(torch.float32)
        g = self.score_decay
        off = s
        if hh > 0:
            assert self.hh_score is not None
            self.hh_score = g * self.hh_score + mass[off : off + hh]
            off += hh
        if q > 0:
            assert self.q_score is not None
            self.q_score = g * self.q_score + mass[off : off + q]
            off += q
        if f > 0:
            assert self.mid_score is not None
            self.mid_score = g * self.mid_score + mass[off : off + f]
            off += f
        if self.ring_score is not None and rlen > 0:
            self.ring_score = g * self.ring_score + mass[off:]
        self._seen_observation = True

    def seed_scores(self, seed: Tensor) -> None:
        """Initialize retention scores from prompt attention: ``seed`` is a
        ``(T,)`` per-position mass over the prompt (see
        :func:`_prompt_seed_scores`); mapped to retained columns by position."""
        if not self.track_scores:
            return
        rlen = self._recent_len()
        if rlen > 0:
            self.ring_score = (
                seed[self.cumulative_length - rlen : self.cumulative_length]
                .to(torch.float32)
                .clone()
            )
        if self._f_len() > 0:
            assert self.mid_pos is not None
            self.mid_score = seed[self.mid_pos].to(torch.float32).clone()
        if self._q_len() > 0:
            assert self.q_pos is not None
            self.q_score = seed[self.q_pos].to(torch.float32).clone()
        self._seen_observation = True

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
            return self._ingest_chunk(key_states, value_states)
        if q_len != 1:
            raise NotImplementedError(
                "BugStreamingCache supports single-shot pre-fill + one-token decode "
                f"steps only, got q_len={q_len} with {self.cumulative_length} cached "
                "tokens (chunked/continued pre-fill would desync the streaming state)."
            )
        return self._decode_step(key_states, value_states)

    def _score_forward(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        """Week-10 frozen continuation scoring (non-mutating): attend the ``q_len``
        window to ``[retained | window]`` without touching the streaming state, so
        one forward scores perplexity over the compressed cache. The retained block
        is strictly-past (true-position RoPE preserved); ``get_mask_sizes`` reports
        ``attended_length() + q_len`` so the causal mask sees it all. No absorb, no
        recent append, no score/position mutation -- ``stored_state_numel`` is
        unchanged by this call."""
        k_ret, v_ret = self._decode_peek()
        return (
            torch.cat([k_ret, key_states], dim=2),
            torch.cat([v_ret, value_states], dim=2),
        )

    def _ingest_chunk(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        """Week-10 chunked pre-fill (OOM-safe for 32K/64K): append a ``q_len``-token
        block to the recent ring, advance ``cumulative_length``, and DEFER the
        absorb to :meth:`consolidate` (called by the harness after the forward). The
        returned ``[sink | hh | mid-hat | recent+chunk]`` lets the chunk attend to
        the compressed-so-far cache plus itself -- so no forward ever holds the full
        ``T`` K/V. Appending the block then consolidating is equivalent to feeding
        the chunk one token at a time via :meth:`_decode_step` (same recent buffer,
        same graduating blocks, same order); it differs from single-shot
        :meth:`_prefill` only in that each chunk saw a *compressed* past (the
        documented deviation, exact at a lossless config)."""
        assert self.recent_k is not None and self.recent_v is not None
        q = int(key_states.shape[2])
        self.recent_k = torch.cat([self.recent_k, self._to_mat(key_states)], dim=1)
        self.recent_v = torch.cat([self.recent_v, self._to_mat(value_states)], dim=1)
        if self.track_scores and self.ring_score is not None:
            zeros = torch.zeros(q, dtype=torch.float32, device=self.ring_score.device)
            self.ring_score = torch.cat([self.ring_score, zeros])
        self.cumulative_length += q
        return self._decode_peek()

    def consolidate(self) -> None:
        """Run the deferred absorb for an over-full recent ring after a chunked
        :meth:`_ingest_chunk` (same block schedule / loop as :meth:`_decode_step`)."""
        while self._recent_len() >= self.recent_window + self.absorb_block:
            self._absorb_block_into_stream(self.absorb_block)

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
        if self.track_scores:
            # Zero until the attach() hook seeds them from the prompt's rows.
            self.ring_score = torch.zeros(recent, dtype=torch.float32, device=k_mat.device)

        if mid > 0 and self.lowrank_enabled:
            k_pre = self._mat_rope(k_mat[:, n_sink : n_sink + mid], n_sink, inverse=True)
            v_mid = v_mat[:, n_sink : n_sink + mid].to(torch.float32)
            for start in range(0, mid, self.prefill_block_size):
                stop = min(mid, start + self.prefill_block_size)
                pos = torch.arange(
                    n_sink + start, n_sink + stop, dtype=torch.int64, device=k_pre.device
                )
                self._absorb_columns(k_pre[:, start:stop], v_mid[:, start:stop], pos, None)
        self.cumulative_length = t
        return key_states, value_states

    def _decode_step(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        assert self.recent_k is not None and self.recent_v is not None
        self.recent_k = torch.cat([self.recent_k, self._to_mat(key_states)], dim=1)
        self.recent_v = torch.cat([self.recent_v, self._to_mat(value_states)], dim=1)
        if self.track_scores:
            assert self.ring_score is not None
            zero = torch.zeros(1, dtype=torch.float32, device=self.ring_score.device)
            self.ring_score = torch.cat([self.ring_score, zero])
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
        if self._hh_len() > 0:  # SLASH exact tier: verbatim post-RoPE, like sinks
            assert self.hh_k is not None and self.hh_v is not None
            parts_k.append(self.hh_k)
            parts_v.append(self.hh_v)
        if self._q_len() + self._f_len() > 0:  # the low-rank (reconstructed) tail
            self._ensure_mid_cache()
            assert self._mid_k_cache is not None and self._mid_v_cache is not None
            parts_k.append(self._mid_k_cache)
            parts_v.append(self._mid_v_cache)
        parts_k.append(self.recent_k)
        parts_v.append(self.recent_v)
        k_ret = torch.cat(parts_k, dim=1) if len(parts_k) > 1 else parts_k[0]
        v_ret = torch.cat(parts_v, dim=1) if len(parts_v) > 1 else parts_v[0]
        return self._to_hf(k_ret), self._to_hf(v_ret)

    def _mid_positions(self) -> Tensor:
        """True positions of the retained **low-rank** columns (assembly order
        ``[quant | fp32]``), for re-rotating their reconstruction. The exact
        heavy-hitter tier is stored verbatim post-RoPE and is not included here.
        Contiguous by construction under FIFO."""
        device = self.c_k.device if self.c_k is not None else torch.device("cpu")
        if not self.track_positions:
            lr_len = self._q_len() + self._f_len()
            mid_start = self.cumulative_length - self._recent_len() - lr_len
            return torch.arange(mid_start, mid_start + lr_len, dtype=torch.int64, device=device)
        parts = []
        if self._q_len() > 0:
            assert self.q_pos is not None
            parts.append(self.q_pos)
        if self._f_len() > 0:
            assert self.mid_pos is not None
            parts.append(self.mid_pos)
        return torch.cat(parts) if len(parts) > 1 else parts[0]

    def _ensure_mid_cache(self) -> None:
        """(Re)build the middle reconstruction; only changes on absorb events."""
        if self._mid_k_cache is not None:
            return
        assert self.u_k is not None and self.u_v is not None
        parts_k: list[Tensor] = []
        parts_v: list[Tensor] = []
        if self._q_len() > 0:
            assert self.qk_codes is not None and self.qv_codes is not None
            if self._coded_mode:
                # CodeBUG: reconstruct against the FROZEN anchor (never u_k).
                assert self.u_ref_k is not None and self.u_ref_v is not None
                parts_k.append(self.u_ref_k @ self._decode_coded(self.qk_codes, self.qk_norm))
                parts_v.append(self.u_ref_v @ self._decode_coded(self.qv_codes, self.qv_norm))
            else:
                assert self.qk_norm is not None and self.qv_norm is not None
                parts_k.append(self.u_k @ self._dequantize(self.qk_codes, self.qk_norm))
                parts_v.append(self.u_v @ self._dequantize(self.qv_codes, self.qv_norm))
        if self._f_len() > 0:
            assert self.c_k is not None and self.c_v is not None
            parts_k.append(self.u_k @ self.c_k)
            parts_v.append(self.u_v @ self.c_v)
        k_pre_hat = torch.cat(parts_k, dim=1) if len(parts_k) > 1 else parts_k[0]
        v_hat = torch.cat(parts_v, dim=1) if len(parts_v) > 1 else parts_v[0]
        if self.track_positions:
            k_hat = self._mat_rope_at(k_pre_hat, self._mid_positions(), inverse=False)
        else:
            # Contiguous middle: the memoized range path (bit-identical to Week 6).
            mid_len = self._mid_len()
            mid_start = self.cumulative_length - self._recent_len() - mid_len
            k_hat = self._mat_rope(k_pre_hat, mid_start, inverse=False)
        self._mid_k_cache = k_hat.to(self.dtype)
        self._mid_v_cache = v_hat.to(self.dtype)

    # ------------------------------------------------------------ cache API

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Report exactly the K/V length ``update()`` will return this step; the
        offset places the (strictly past) returned block right below the query's
        true position so the causal mask sees it all as visible."""
        if self._mode in ("score", "ingest"):
            # score: returns [retained | window] (non-mutating). ingest: returns
            # [retained | chunk] then defers absorb. Both: the returned block is
            # strictly-past + the query block, so kv_length = attended + q_len and
            # kv_offset places it right below the query's true position. In ingest,
            # get_mask_sizes runs before update() advances cumulative_length, so
            # attended_length()/cumulative_length are pre-chunk here.
            attended = self.attended_length()
            return attended + query_length, self.cumulative_length - attended
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
        """Float-equivalents of the *stored* per-layer state (the honest memory):
        fp32/verbatim tensors at 1 each; quantized codes at ``quant_bits/32``
        each (bit-packable) + their fp32 norms; retention positions (int32) and
        scores at 1 each. Shared quantizer side info is counted once at the
        cache level (:meth:`BugStreamingCache.stored_state_numel`)."""
        tensors = (
            self.sink_k,
            self.sink_v,
            self.recent_k,
            self.recent_v,
            self.u_k,
            self.c_k,
            self.u_v,
            self.c_v,
            self.hh_k,  # SLASH exact tier: full 2n floats/token, like a whole token
            self.hh_v,
            self.u_ref_k,  # CodeBUG frozen anchor (n, r_a) fp32 -- counted, not free
            self.u_ref_v,
        )
        total = sum(t.numel() for t in tensors if t is not None)
        # The square-root core B is *provably diagonal* (``augmented_bug_step``
        # returns ``diag(sigma)`` every step and nothing rotates it between
        # steps), so its deployable footprint is its ``r`` diagonal entries, not
        # ``r^2`` -- counted honestly here (same convention as quant codes at
        # bits/32 rather than their uint8 storage). ``tests`` pin the diagonality.
        for core in (self.b_k, self.b_v):
            if core is not None:
                total += int(min(core.shape))
        if self.qk_codes is not None:
            assert self.qv_codes is not None
            # bits/code: PolarQuant (variant D, r codes/token) or ProductQuantizer
            # (CodeBUG, M codes/token). Codes counted at bits/32 (bit-packable).
            bits = self.codebook.bits if self.codebook is not None else self.quant_bits
            assert bits is not None
            code_entries = int(self.qk_codes.numel() + self.qv_codes.numel())
            total += math.ceil(code_entries * bits / 32)
            # Per-column norms (fp32): variant D always; CodeBUG in normalize mode.
            for nrm in (self.qk_norm, self.qv_norm):
                if nrm is not None:
                    total += int(nrm.numel())
        bookkeeping = (
            self.mid_pos,
            self.q_pos,
            self.mid_score,
            self.q_score,
            self.ring_score,
            self.hh_pos,
            self.hh_score,
            self.mid_weight,
        )
        total += sum(t.numel() for t in bookkeeping if t is not None)
        # Week-9 D3: the per-column surprise buffers are stored state under
        # retention="lowrank_surprise" and are counted. Under the correlation
        # probe (``_probe_surprise`` on a retention="attn" run) they are
        # instrumentation only -- NOT part of the deployed config's footprint --
        # so they are excluded from the honest budget in that case.
        if self.track_surprise:
            for surp in (self.mid_surprise, self.q_surprise):
                if surp is not None:
                    total += int(surp.numel())
        return int(total)

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
    With ``retention="attn"``, wrap the forward/generate in :meth:`attach` so
    per-step attention rows drive the retention scores (mirrors
    :meth:`MorphKVCache.attach`).

    Parameters
    ----------
    model:
        The (Llama-family) model; supplies the layer count and the rotary
        embedding module used for exact un-/re-rotation.
    rank:
        BUG rank cap ``r`` per layer (0 disables the low-rank middle).
    coord_budget:
        Max retained fp32 middle-token coordinate columns per layer (0 disables
        the low-rank middle; with ``rank=0`` this cache *is* StreamingLLM).
    retention:
        Coordinate eviction rule: ``"fifo"`` (Week-6 baseline), ``"attn"``
        (EMA attention mass; needs :meth:`attach`), ``"energy"`` (``||c_s||_2``),
        ``"lowrank_surprise"`` (Week-9 D3: keep the highest-residual outliers), or
        ``"blend"`` (Week-9 D3: rank-blend attn mass and surprise, weight
        ``surprise_blend``). See the module docstring.
    score_decay:
        EMA decay per decode step for ``retention="attn"`` scores.
    quant_bits, quant_budget:
        Week-7 D: demote evicted fp32 coordinates into a PolarQuant tier of
        ``quant_budget`` columns at ``quant_bits`` bits/coordinate (both set,
        or neither). Codes count ``quant_bits/32`` float-equivalents each.
    quant_seed:
        Seed for the shared PolarQuant rotation/codebook.
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
        retention: str = "fifo",
        score_decay: float = 0.97,
        surprise_blend: float = 0.5,
        quant_bits: int | None = None,
        quant_budget: int = 0,
        quant_seed: int = 0,
        hh_budget: int = 0,
        merge: bool = False,
        coord_codebook: ProductQuantizer | None = None,
        anchor_rank: int = 0,
        anchor_seal_absorbs: int = 4,
        code_budget: int = 0,
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
        self._retention = retention
        self._quant_bank = _QuantBank(quant_bits, seed=quant_seed) if quant_bits else None
        self._codebook = coord_codebook  # shared PQ side info, counted once
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
                retention=retention,
                score_decay=score_decay,
                surprise_blend=surprise_blend,
                quant_bits=quant_bits,
                quant_budget=quant_budget,
                quant_bank=self._quant_bank,
                hh_budget=hh_budget,
                merge=merge,
                coord_codebook=coord_codebook,
                anchor_rank=anchor_rank,
                anchor_seal_absorbs=anchor_seal_absorbs,
                code_budget=code_budget,
            )
            for _ in range(n_layers)
        ]
        super().__init__(layers=layers)

    @contextmanager
    def attach(self, model: PreTrainedModel) -> Iterator[None]:
        """Register per-layer hooks that record attention rows into the retention
        scores (``retention="attn"``; a no-op otherwise). Per decode step the
        aggregated attention row over this cache's *returned* K is recomputed
        with MorphKV's exact machinery and EMA'd into per-column scores; at
        pre-fill end the scores are seeded from the last ``recent_window``
        prompt queries' causal rows. Also active for ``retention="blend"``, which
        needs the same attention-mass scores as one of its two blended signals."""
        if self._retention not in ("attn", "blend"):
            yield
            return
        handles = []

        def make_hook(cache: BugStreamingCache) -> Any:
            def hook(
                module: nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                output: Any,
            ) -> Any:
                if kwargs.get("past_key_values") is not cache:
                    return output  # a different cache is in play; stay out
                layer = cache.layers[module.layer_idx]
                assert isinstance(layer, BugStreamingLayer)
                if not layer.is_initialized or layer.cumulative_length == 0:
                    return output
                if layer._mode == "score":
                    return output  # frozen scoring is non-mutating: never re-seed
                hidden_states = kwargs["hidden_states"]
                cos, sin = kwargs["position_embeddings"]
                if hidden_states.shape[1] == 1:  # decode step
                    keys, _ = layer._decode_peek()  # memoized middle; cheap concat
                    row = _aggregated_attention_row(module, hidden_states, keys, cos, sin)
                    layer.observe_attention(row.sum(dim=0))
                else:  # pre-fill: seed scores from the prompt's own attention
                    seed = _prompt_seed_scores(
                        module,
                        hidden_states,
                        (cos, sin),
                        layer.recent_window,
                        layer.score_decay,
                    )
                    layer.seed_scores(seed)
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

    def _bug_layers(self) -> list[BugStreamingLayer]:
        return [layer for layer in self.layers if isinstance(layer, BugStreamingLayer)]

    def _set_mode(self, mode: str) -> None:
        for layer in self._bug_layers():
            layer._mode = mode

    @contextmanager
    def frozen_scoring(self) -> Iterator[None]:
        """Week-10: score a continuation window over the compressed cache without
        mutating it. Inside the block every BUG layer returns ``[retained | window]``
        (non-mutating) so a ``q_len=W`` teacher-forced forward measures perplexity
        against the compressed cache; state (and ``stored_state_numel``) is
        unchanged on exit. Restores ``"normal"`` even on exception."""
        self._set_mode("score")
        try:
            yield
        finally:
            self._set_mode("normal")

    @contextmanager
    def ingesting(self) -> Iterator[None]:
        """Week-10 chunked pre-fill (OOM-safe for 32K/64K). Inside the block, feed
        the prompt in chunks; the FIRST chunk hits the normal single-shot
        ``_prefill`` (cumulative_length == 0), each later chunk is appended to the
        recent ring with its absorb deferred. Call :meth:`consolidate` after each
        chunk forward to run the deferred absorb. Restores ``"normal"`` on exit."""
        self._set_mode("ingest")
        try:
            yield
        finally:
            self._set_mode("normal")

    def consolidate(self) -> None:
        """Run each BUG layer's deferred absorb after a chunked-ingest forward."""
        for layer in self._bug_layers():
            layer.consolidate()

    def enable_surprise_probe(self) -> list[tuple[Tensor, Tensor, Tensor]]:
        """Week-9 D3 GO/NO-GO gate: instrument every layer to also track low-rank
        surprise (under any retention) and record the per-column (attention-mass,
        surprise, energy) arrays at each eviction into a shared sink, returned
        here. Run a ``retention="attn"`` cache through a stream, then measure the
        surprise<->attention correlation on the pooled sink (redundant if high,
        complementary if low). Instrumentation only -- not counted in
        :meth:`stored_state_numel` (the deployed config stores no surprise)."""
        sink: list[tuple[Tensor, Tensor, Tensor]] = []
        for layer in self._bug_layers():
            layer._probe_surprise = True
            layer._probe_sink = sink
        return sink

    def stored_state_numel(self) -> int:
        """Total stored float-equivalents across layers (the constant-memory
        claim), plus the shared quantizer side info counted once."""
        total = sum(layer.stored_state_numel() for layer in self._bug_layers())
        if self._quant_bank is not None:
            total += self._quant_bank.side_info_numel()
        if self._codebook is not None:
            # Shared, calibrated PQ codebook: side info counted ONCE per cache
            # (amortized across every coded token and every layer).
            total += self._codebook.codebook_numel()
        return total

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
