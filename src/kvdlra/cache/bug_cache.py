"""``BugStreamingCache`` -- a constant-memory decode-time streaming BUG KV cache.

Week-5 Axis B: the first use of the tracked basis *during generation* rather than
over a finished prefill. Everywhere else in this project BUG compresses
the pre-fill and is static during decode (:class:`kvdlra.baselines.lowrank_press.BUGPress`); this
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
  token instead of ``n``). The three are stored in ``gist_dtype`` -- fp32 by
  default, bf16 under L5.1, which halves their bytes -- and are upcast to fp32
  for every computation. When the coordinate buffer is full, columns are
  *evicted* (see "Week-7 retention" below) -- that is the memory bound:
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
augmented rank-truncating step (:func:`kvdlra.tracker.isvd.
augmented_bug_step` -- block incremental SVD, fp32 core per PLAN §8
pitfall #4) advances ``(U, B)``, existing coordinates are re-expressed in the
new basis (``C <- rot @ C`` -- each truncation projects old tokens onto the new
subspace, which is where repeated projection erodes them; no error bound is
claimed for it, the rank-adaptive robustness results are for an ODE flow, not a
column stream),
and the graduating coordinates are appended. Attention then sees ``[sinks |
RoPE(U C, true positions) | recent]``; the middle reconstruction only changes on
absorb events and is cached in between, so the steady-state per-step cost is a
concat plus attention over a constant-length cache. Amortized per-token update
cost: ``O(n r + (r+b)^3 / b + r^2 W / b)`` -- bounded, independent of generated
length.

Week-7 retention (tier 1)
-------------------------
Week 6 measured the deep-horizon loss mechanisms: FIFO coordinate eviction
(adaptive *subspace*, non-adaptive *retention*), erosion by repeated
projection, and the fixed-rank squeeze. Two independent knobs on the same
coordinate buffer attack the first two:

* **(A) adaptive coordinate retention** -- ``retention``:

  - ``"fifo"`` (default): drop the oldest column (the Week-6 behaviour; note
    the Week-7 tracker robustness fix makes reruns *fp-equivalent*, not
    bit-identical, to the archived Week-6 numbers -- rerun baselines in-sweep);
  - ``"lowrank_surprise"`` (Week-9 D3): drop the column with the smallest
    **low-rank reconstruction residual** ``||k - U U^T k|| / ||k||`` -- i.e. keep
    the *outliers* the low-rank summary cannot reproduce and evict the columns it
    already predicts (redundant). By Pythagoras (``U`` orthonormal) this residual
    is the out-of-subspace half of ``||k||^2``. It is NOT recomputable after
    absorption (``U C`` lies in the subspace, residual == 0), so it is *stored*
    per column as one fp32 scalar (``mid_surprise``/``q_surprise``), a snapshot
    at graduation.

  Non-FIFO retention makes the retained middle **non-contiguous in position**,
  so true per-column positions are tracked (``mid_pos``/``q_pos``) and the
  reconstruction is re-rotated at those positions (gathered cos/sin from the
  model's own rotary module). The position arrays and the surprise snapshots are
  **counted** in ``stored_state_numel`` -- one float equivalent per int32
  position, one per fp32 scalar.

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
  Accounting (the Week-4 fairness convention): codes
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

Scope guards: single-shot pre-fill (chunked pre-fill raises, as in ``BUGPress``); pre-fill
attention is full/standard (the same protocol as all Axis-B baselines -- compression bounds
what is *retained for decode*). Batch > 1 is served by composition: at the first ``update``
with B rows the layer builds B independent row layers from its own constructor kwargs and
drives each through the batch-1 path on its ``(1, H, T, D)`` slice (``_update_rows``); every
reader fans out over the rows. Rows must stay equal-length (the latency task's random
tensor is the only batched input); a ragged result raises instead of padding.

``rank=0`` or ``coord_budget=0`` disables the low-rank middle entirely:
graduating tokens are simply dropped and the cache degenerates to **sinks +
recent window** -- i.e. the StreamingLLM baseline falls out of the same
implementation.

References
----------
G. Ceruti, J. Kusch, C. Lubich, arXiv:2104.05247 (rank-adaptive BUG).
Xiao et al., arXiv:2309.17453 (StreamingLLM; sinks + recent window).
Zandieh et al., arXiv:2504.19874 (TurboQuant/PolarQuant; the quantized tier).
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, CacheLayerMixin, LinearAttentionCacheLayerMixin
from transformers.models.llama.modeling_llama import rotate_half

from kvdlra.kernel.reference import FactoredMiddle
from kvdlra.quant import PolarQuant
from kvdlra.tracker import TRACKERS
from kvdlra.tracker.isvd import eff_rank, orth_error, reorthonormalize

__all__ = ["BugStreamingCache", "BugStreamingLayer", "OrthonormalityError"]

RETENTION_MODES = ("fifo", "lowrank_surprise")

# What the gist (U, B, C) may be STORED as (L5.1). Two entries, because the accounting
# bills exactly two widths: ``bug_footprint(gist_bits=32)`` and ``gist_bits=16``. fp16 is
# absent on purpose -- it would store at 16 bits and bill at 32, a silent mis-bill. The
# string form is what a YAML ``cache:`` block carries (omegaconf has no torch dtypes).
GIST_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}

# How a decode step is attended (Task L4.4; ADR 0001 option (iii)). "reconstruct" (default)
# is today's path: the middle is rebuilt as an n x T block (`_ensure_mid_cache`) and returned
# with the dense tokens. "kernel": `update()` returns only [sinks | exact tier | recent] and
# keeps the factored middle on the layer (`_factored_middle`) for the attention function
# `BugStreamingCache.attach` registers, which attends both from (U, C) tile by tile.
DECODE_ATTENTION = ("reconstruct", "kernel")

# Which basis the gist tracks (Task L4.9; ADR 0001 option (i)). "pre" (default) un-rotates
# keys at ingest and re-rotates the reconstruction at their true positions; "post" tracks the
# rotated keys directly, making both sites identities. V is never rotated either way.
ROPE_BASES = ("pre", "post")


class OrthonormalityError(RuntimeError):
    """The tracked basis lost orthonormality beyond ``orth_abort_tol``.

    Divergence of this cache *is* loss of orthonormality (CODE_AUDIT Part A §Q4: 6e-4 at
    8K -> 35.1 at 36K on Qwen r256, ending in four-digit perplexity). Past the abort
    tolerance the stored gist no longer represents the keys it was built from, so the
    trial is failed loudly rather than silently reported as a number."""


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
    memory budget can count it in the same unit (Week-7 guardrail: every variant counts
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


def _rope_apply(x_htd: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to ``(H, T, D)`` given ``(T, D)`` cos/sin (broadcast over heads)."""
    rot: Tensor = rotate_half(x_htd)  # type: ignore[no-untyped-call]
    return x_htd * cos + rot * sin


def _rope_unapply(x_htd: Tensor, cos: Tensor, sin: Tensor, scale_sq: float) -> Tensor:
    """Exact inverse of :func:`_rope_apply` (rotation transposed, / scaling^2)."""
    rot: Tensor = rotate_half(x_htd)  # type: ignore[no-untyped-call]
    return (x_htd * cos - rot * sin) / scale_sq


def _nan_max(a: float, b: float) -> float:
    """NaN-sticky ``max``: plain ``max(a, b)`` silently drops a NaN landing in the
    SECOND argument (``max(0.0, nan) == 0.0``) -- once either reading is NaN the fold
    stays NaN, the way a window's worst-case error should when it is fed one."""
    return float("nan") if (a != a or b != b) else max(a, b)


class BugStreamingLayer(CacheLayerMixin):  # type: ignore[no-untyped-call]
    """One layer's constant-memory streaming-BUG KV state (see module docstring).

    Internally tokens live as feature-by-token matrices ``(n, T)`` with ``n =
    num_kv_heads * head_dim``; conversion to the
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
        min_sv_frac: float = 0.0,
        gist_dtype: torch.dtype | str = torch.float32,
        tracker: str = "isvd",
        oja_eta0: float = 20.0,
        oja_decay: float = 0.03,
        freeze_after: int = 4096,
        basis_seed: int = 0,
        prefill_block_size: int = 128,
        retention: str = "fifo",
        quant_bits: int | None = None,
        quant_budget: int = 0,
        quant_bank: _QuantBank | None = None,
        hh_budget: int = 0,
        hh_select: str = "attn",
        hh_neighbor: int = 0,
        hh_retain: bool = True,
        score_rank: int | None = None,
        seed_hh_warmup: bool = False,
        orth_fix_tol: float | None = 1e-3,
        orth_abort_tol: float | None = 1e-1,
        qr_every: int | None = None,
        diag_every: int = 64,
        decode_attention: str = "reconstruct",
        kernel_operand_dtype: torch.dtype | str = torch.bfloat16,
        rope_basis: str = "pre",
        layer_idx: int = 0,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        # The kwargs this layer was built with, so a batch > 1 update can build one
        # independent row layer per batch element (`_update_rows`). `locals()` here is
        # exactly the parameter list (plus `self` and the zero-arg-super `__class__` cell).
        self._ctor_kwargs: dict[str, Any] = {
            k: v for k, v in locals().items() if k not in ("self", "__class__")
        }
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
        if quant_budget < 0:
            raise ValueError(f"quant_budget must be >= 0, got {quant_budget}")
        if (quant_bits is None) != (quant_budget == 0):
            raise ValueError("quant_bits and quant_budget (> 0) must be set together")
        if quant_bits is not None and not 1 <= quant_bits <= 8:
            raise ValueError(f"quant_bits must be in [1, 8], got {quant_bits}")
        if freeze_after < 0:
            raise ValueError(f"freeze_after must be >= 0, got {freeze_after}")
        if hh_budget < 0:
            raise ValueError(f"hh_budget must be >= 0, got {hh_budget}")
        if hh_select not in ("attn", "surprise"):
            raise ValueError(f"hh_select must be 'attn' or 'surprise', got {hh_select!r}")
        if hh_neighbor < 0:
            raise ValueError(f"hh_neighbor must be >= 0, got {hh_neighbor}")
        if hh_neighbor > 0 and hh_select != "surprise":
            raise ValueError("hh_neighbor (span expansion) requires hh_select='surprise'")
        # Week-12 H1 ablation: ``hh_retain=False`` keeps the surprise selection +
        # withholding-from-absorption mechanics bit-identical (the pool still
        # lives in hh_k/hh_v and feeds re-selection/demotion each absorb) but the
        # pool is invisible to attention. Discarding an *empty* tier would be the
        # silent hh=0 degeneration to plain BUG; an attn-selected tier receives
        # no attention mass once invisible -- both are config errors, fail loud.
        if not hh_retain and hh_budget < 1:
            raise ValueError(
                "hh_retain=False (select-and-discard ablation) requires hh_budget >= 1: "
                "with hh_budget=0 the SLASH path is disabled entirely and the config "
                "silently degenerates to plain BUG"
            )
        if not hh_retain and hh_select != "surprise":
            raise ValueError(
                "hh_retain=False requires hh_select='surprise': a discarded tier is "
                "invisible to attention, so attention-mass selection is incoherent"
            )
        # Week-11 SurpriseSLASH: the exact heavy-hitter tier is selected by
        # low-rank surprise (the out-of-subspace residual, computed from the
        # pre-step basis + stored keys, so it is *attach-free* and imposes no
        # constraint on the tail retention rule). The Week-7 "attn" selector read
        # the EMA attention mass of the retired retention="attn" mode, so an
        # enabled tier now requires hh_select='surprise'.
        if hh_budget > 0 and hh_select == "attn":
            raise ValueError(
                "hh_budget > 0 (SLASH exact heavy-hitters) requires hh_select='surprise': "
                "attention-mass selection retired with retention='attn'"
            )
        # Week-15 T2 (score-rank decoupling): cap SurpriseSLASH *selection* scoring
        # to the leading ``score_rank`` basis columns (``u_k[:, :s]`` -- energy-
        # ordered by the integrator's SVD, so a leading-column subview is the
        # dominant subspace and orthonormal for free).
        # Applied ONLY at the SLASH selection site; the tail-retention surprise
        # snapshot stays uncapped. Storage rank (and so accounting) is unchanged.
        if score_rank is not None:
            if not 1 <= score_rank <= rank:
                raise ValueError(f"score_rank must be in [1, rank={rank}], got {score_rank}")
            if hh_select != "surprise":
                raise ValueError(
                    "score_rank caps SurpriseSLASH selection scoring; it requires "
                    f"hh_select='surprise', got {hh_select!r}"
                )
            if hh_budget < 1:
                raise ValueError(
                    "score_rank requires an enabled SLASH exact tier (hh_budget >= 1); "
                    f"got hh_budget={hh_budget}"
                )
        if not 0.0 <= min_sv_frac < 1.0:
            raise ValueError(f"min_sv_frac must be in [0, 1), got {min_sv_frac}")
        resolved = GIST_DTYPES.get(gist_dtype) if isinstance(gist_dtype, str) else gist_dtype
        if resolved is None or resolved not in GIST_DTYPES.values():
            raise ValueError(f"gist_dtype must be one of {sorted(GIST_DTYPES)}, got {gist_dtype!r}")
        if orth_fix_tol is not None and orth_fix_tol <= 0.0:
            raise ValueError(f"orth_fix_tol must be > 0 (None disables), got {orth_fix_tol}")
        if orth_abort_tol is not None and orth_abort_tol <= 0.0:
            raise ValueError(f"orth_abort_tol must be > 0 (None disables), got {orth_abort_tol}")
        if (
            orth_abort_tol is not None
            and orth_fix_tol is not None
            and orth_abort_tol < orth_fix_tol
        ):
            # The guard repairs first and judges the REPAIRED basis, so a stream the repair
            # threshold leaves alone must be below the abort threshold too -- otherwise the
            # only way to abort is a basis nothing ever tried to fix.
            raise ValueError(
                f"orth_abort_tol ({orth_abort_tol}) must be >= orth_fix_tol ({orth_fix_tol})"
            )
        if qr_every is not None and qr_every < 1:
            raise ValueError(f"qr_every must be >= 1 (None disables), got {qr_every}")
        if diag_every < 1:
            raise ValueError(f"diag_every must be >= 1, got {diag_every}")
        if decode_attention not in DECODE_ATTENTION:
            raise ValueError(
                f"decode_attention must be one of {DECODE_ATTENTION}, got {decode_attention!r}"
            )
        op = (
            GIST_DTYPES.get(kernel_operand_dtype)
            if isinstance(kernel_operand_dtype, str)
            else kernel_operand_dtype
        )
        if op is None or op not in GIST_DTYPES.values():
            raise ValueError(
                f"kernel_operand_dtype must be one of {sorted(GIST_DTYPES)}, "
                f"got {kernel_operand_dtype!r}"
            )
        if rope_basis not in ROPE_BASES:
            raise ValueError(f"rope_basis must be one of {ROPE_BASES}, got {rope_basis!r}")
        if rope_basis == "post" and decode_attention == "kernel":
            raise ValueError(
                "rope_basis='post' is the reconstruct-path accuracy row (ADR 0001 option (i)); "
                "decode_attention='kernel' re-rotates in-tile and needs the pre-RoPE basis"
            )
        if tracker == "bug":
            warnings.warn(
                "tracker='bug' is deprecated; the shipped step is block incremental SVD, "
                "so its name is 'isvd'. The alias will be removed.",
                DeprecationWarning,
                stacklevel=2,
            )
            tracker = "isvd"
        self._ctor_kwargs["tracker"] = tracker  # rows must not re-trigger the alias warning
        if tracker not in TRACKERS:
            raise ValueError(f"tracker must be one of {sorted(TRACKERS)}, got {tracker!r}")
        # The gist tracker (Week-20 swap ablation, extended by the L3 Gate-1 controls).
        # "isvd" = the shipped augmented step, which at theta=None and min_sv_frac=0 IS
        # fixed-rank incremental SVD; "oja" = Oja's rule (the OjaKV baseline), whose
        # schedule the two knobs below set; "fd" = Frequent Directions; "frozen" = the
        # incremental-SVD step for the first ``freeze_after`` tokens and a fixed basis
        # after that; "random" = one seeded orthonormal draw per layer and stream, never
        # updated. Everything else in the cache is held fixed across the five.
        self.tracker = tracker
        self.oja_eta0 = oja_eta0
        self.oja_decay = oja_decay
        self.freeze_after = freeze_after
        self.basis_seed = basis_seed
        self.rope = rope
        self.rank = rank
        self.coord_budget = coord_budget
        self.recent_window = recent_window
        self.absorb_block = absorb_block
        self.n_sink = n_sink
        self.theta = theta
        self.min_sv_frac = min_sv_frac
        # L5.1: the dtype the gist (U, B, C) is STORED as between absorbs. fp32 (default)
        # is bit-identical to the pre-L5.1 cache; bfloat16 halves those bytes and is billed
        # at 16 bits (``bug_footprint(gist_bits=16)``). Every computation still runs in
        # fp32 -- see :meth:`_cast_gist` and :meth:`_absorb_columns`.
        self.gist_dtype: torch.dtype = resolved
        self.prefill_block_size = prefill_block_size
        self.retention = retention
        self.quant_bits = quant_bits
        self.quant_budget = quant_budget
        self.hh_budget = hh_budget
        self.hh_select = hh_select
        self.hh_neighbor = hh_neighbor
        self.hh_retain = hh_retain
        self.score_rank = score_rank
        # Week-13 T-B: seed the exact tier from the first ingest chunk's outliers
        # (default off). Only fires under chunked ingest (self._mode == "ingest") so
        # single-shot prefill keeps the tier empty, matching the deployed contract.
        self.seed_hh_warmup = seed_hh_warmup
        # Orthonormality guard (CODE_AUDIT Part A §Q4). Divergence of this cache is loss
        # of orthonormality of the tracked basis, and nothing used to watch for it. Every
        # absorb measures ``‖UᵀU - I‖`` on the STORED basis of each stream: above
        # ``orth_fix_tol`` the basis is re-orthonormalized in place (coordinates and the
        # quantized tier carried exactly, so the retained history survives the repair), and
        # only a basis still above ``orth_abort_tol`` AFTER that repair fails the trial --
        # an unrepairable basis, not a numerical event a thin QR fixes (D-011 addendum 5).
        # ``qr_every`` is the experimental factor: re-orthonormalize unconditionally every
        # k absorbs and after that stream's own rank change, whatever the measurement says.
        # ``orth_fix_tol=None`` with ``orth_abort_tol`` set (and ``qr_every`` unset): measure
        # and abort without ever repairing -- the repair branch cannot fire with no finite
        # fix threshold, so the abort check sees the raw, unrepaired error (fix1 A4).
        # How often the fix fires is a property of the run, not of the defaults. On CPU in
        # fp32 -- the tests, the r64 golden -- it does not: the measured ceiling is ~7e-4
        # over 1400 adversarial steps and 6e-5 over the golden, both under the 1e-3
        # threshold, and that path is the bit-identical one. On the pods it does: at bf16
        # model dtype the L2 filler-realism diag rows report a repair in essentially every
        # 64-absorb window (``fixed_k`` 100 %, ``fixed_v`` 99.2 %, pre-repair max
        # ``orth_err_k`` at the 1e-3 threshold), and under ``gist_dtype=bfloat16`` the
        # stored basis is re-rounded every absorb (‖UᵀU - I‖_F ~ 5e-3 at n=1024, r=64), so
        # the repair fires every absorb but the first (which builds its basis from
        # nothing, so the step returns a fresh orthonormal factor) by design.
        self.orth_fix_tol = orth_fix_tol
        self.orth_abort_tol = orth_abort_tol
        self.qr_every = qr_every
        self.diag_every = diag_every
        self.decode_attention = decode_attention
        # The dtype the kernel rounds U, C, K-hat, V-hat and P to at the tensor-core inputs
        # (fp32 accumulation throughout). bf16 is the pod; fp32 makes the tiny-model CPU
        # check exact.
        self.kernel_operand_dtype: torch.dtype = op
        # ADR 0001 §3 (i): "post" tracks post-RoPE keys -- the ingest un-rotation and the
        # reconstruct re-rotation are identities; V is never rotated either way.
        self.rope_basis = rope_basis
        self.layer_idx = layer_idx
        # Per-layer diagnostic rows, drained by ``BugStreamingCache.drain_diag()``. Not
        # cleared by ``reset()``: a drained-once contract must not lose a finished
        # stream's measurements to the next stream's setup.
        self.diag: list[dict[str, object]] = []
        # rank=0 / coord_budget=0 => no low-rank middle => StreamingLLM baseline.
        self.lowrank_enabled = rank >= 1 and coord_budget >= 1
        if quant_budget > 0 and not self.lowrank_enabled:
            raise ValueError("quant_budget > 0 requires an enabled low-rank middle")
        if hh_budget > 0 and not self.lowrank_enabled:
            raise ValueError("hh_budget > 0 requires an enabled low-rank middle")
        # SLASH exact heavy-hitter tier (Week-7 dominance program).
        self.hh_enabled = hh_budget >= 1 and self.lowrank_enabled
        self._quant_bank = (
            quant_bank
            if quant_bank is not None
            else (_QuantBank(quant_bits) if quant_bits is not None else None)
        )
        # Adaptive retention needs true per-column positions (the middle stops
        # being contiguous).
        self.track_positions = self.lowrank_enabled and retention != "fifo"
        # Week-9 D3: low-rank surprise retention keeps the highest-residual
        # (outlier) columns and evicts the ones the basis already reconstructs
        # (redundant). The residual ``||k - U U^T k||`` is NOT recomputable after
        # absorption (``U C`` lies in the subspace, residual == 0), so it must be
        # *stored* per column -- one fp32 scalar. Counted in
        # ``stored_state_numel`` (the stored-state guardrail).
        self.track_surprise = self.lowrank_enabled and retention == "lowrank_surprise"
        self.cumulative_length = 0
        self._reset_state()

    # ------------------------------------------------------------------ state

    def _reset_state(self) -> None:
        self.cumulative_length = 0
        # Batch > 1: one independent layer per row, built at the first batched update.
        self._rows: list[BugStreamingLayer] | None = None
        # Kernel path: set by a kernel-mode decode step, cleared by every other update.
        self._kernel_step = False
        self._kernel_mid: FactoredMiddle | None = None
        self.sink_k: Tensor | None = None  # (n, <=n_sink) post-RoPE, verbatim
        self.sink_v: Tensor | None = None
        self.recent_k: Tensor | None = None  # (n, <recent_window+absorb_block) verbatim
        self.recent_v: Tensor | None = None
        # The gist, stored in ``gist_dtype`` (fp32 by default, bf16 under L5.1) and
        # upcast to fp32 for every computation (:meth:`_cast_gist`).
        self.u_k: Tensor | None = None  # (n, r) pre-RoPE key basis
        self.b_k: Tensor | None = None  # (r, r) square-root core
        self.c_k: Tensor | None = None  # (r, <=coord_budget) coords, current basis
        self.u_v: Tensor | None = None
        self.b_v: Tensor | None = None
        self.c_v: Tensor | None = None
        # Quantized age tier (Week-7 D): PolarQuant codes/norms, rows = columns,
        # shape ``(Wq, r)`` in the *live* basis, rotated per absorb.
        self.qk_codes: Tensor | None = None  # (Wq, r) uint8 codes, K
        self.qk_norm: Tensor | None = None  # (Wq,) fp32
        self.qv_codes: Tensor | None = None
        self.qv_norm: Tensor | None = None
        # Retention bookkeeping (Week-7 A): true per-column positions.
        self.mid_pos: Tensor | None = None  # (f_len,) int64, fp32-tier positions
        self.q_pos: Tensor | None = None  # (q_len,) int64, quant-tier positions
        # Retention bookkeeping (Week-9 D3): per-column low-rank surprise (the
        # residual ||k - U U^T k|| at graduation, normalized -- a snapshot, never
        # recomputed as the basis drifts).
        self.mid_surprise: Tensor | None = None  # (f_len,) fp32
        self.q_surprise: Tensor | None = None  # (q_len,) fp32
        # SLASH exact heavy-hitter tier (Week-7 dominance): verbatim K/V + posns.
        self.hh_k: Tensor | None = None  # (n, <=hh_budget) post-RoPE, verbatim
        self.hh_v: Tensor | None = None
        self.hh_pos: Tensor | None = None  # (hh_len,) int64 true positions
        self._mid_k_cache: Tensor | None = None  # (n, mid_len) storage dtype, post-RoPE
        self._mid_v_cache: Tensor | None = None
        # Orthonormality guard bookkeeping: absorbs so far and the current diag window
        # (max orth_error / min effective rank per stream since the last emitted row).
        self._absorbs = 0
        self._diag_window = 0  # absorbs since the last flush -- 0 means nothing to flush
        self._diag_tokens = 0  # ``tokens_seen`` of the last absorb in the open window
        self._diag_max_err_k = 0.0
        self._diag_max_err_v = 0.0
        # Seeded at the cap: ``eff_rank`` counts the diagonal of an ``(r, r)`` core, so
        # the configured rank is an exact upper bound and plain ``min`` is the fold.
        self._diag_min_rank_k = self.rank
        self._diag_min_rank_v = self.rank
        self._diag_fixed_k = False
        self._diag_fixed_v = False
        # Running maximum behind ``_tokens_seen`` (fix1 R2): reset with the rest of the
        # layer's state so a new stream starts its own frontier at 0.
        self._tokens_seen_max = 0
        # Week-10 harness mode (default "normal" preserves every existing call
        # site / the q_len==1 decode invariant). "score": non-mutating frozen
        # continuation-window forward (Phase 1b). "ingest": chunked pre-fill
        # (Phase 3). Set only by BugStreamingCache context managers.
        self._mode = "normal"

    def lazy_initialization(self, key_states: Tensor, value_states: Tensor) -> None:
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

    def _hh_attended_len(self) -> int:
        """Exact-tier tokens attention can see: the full tier normally, 0 under
        the Week-12 select-and-discard ablation (the pool still exists and feeds
        selection/demotion, but is never returned to attention)."""
        return self._hh_len() if self.hh_retain else 0

    def _mid_len(self) -> int:
        return self._hh_attended_len() + self._q_len() + self._f_len()

    def _post_update_lengths(self, query_length: int) -> tuple[int, int, int]:
        """Predict ``(sink, mid, recent)`` lengths *after* an ``update`` of
        ``query_length`` tokens -- the single source of truth shared by
        ``update()``'s absorb loop and ``get_mask_sizes`` (mask consistency)."""
        if self.cumulative_length == 0:
            # Pre-fill returns the full input.
            return 0, 0, query_length
        recent = self._recent_len() + query_length
        mid = self._mid_len()
        mid_cap = self.coord_budget + self.quant_budget + (self.hh_budget if self.hh_retain else 0)
        # Under hh_retain=False the *visible* middle grows per absorb by the
        # DEMOTED count (candidates minus what the pool keeps) -- 0 while the
        # pool is still filling -- so the invisible pool must be simulated here
        # or decode-time mask sizes desync from the returned K length.
        wh = self._hh_len() if (self.hh_enabled and not self.hh_retain) else -1
        while recent >= self.recent_window + self.absorb_block:
            recent -= self.absorb_block
            if self.lowrank_enabled:
                if wh >= 0:
                    new_wh = min(self.hh_budget, wh + self.absorb_block)
                    mid = min(mid_cap, mid + (wh + self.absorb_block - new_wh))
                    wh = new_wh
                else:
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
        self._mid_k_cache = None
        self._mid_v_cache = None
        if not self.lowrank_enabled:
            return  # StreamingLLM mode: graduating tokens are dropped.

        # Positions: recents are the last `_recent_len (post-slice)` tokens; the
        # graduating block starts where the middle currently ends.
        grad_start = self.cumulative_length - self._recent_len() - m
        grad_pos = torch.arange(grad_start, grad_start + m, dtype=torch.int64, device=grad_k.device)
        if self.hh_enabled:
            self._absorb_block_slash(grad_k, grad_v, grad_pos)
            return
        block_k = (
            grad_k.to(torch.float32)
            if self.rope_basis == "post"
            else self._mat_rope(grad_k, grad_start, inverse=True)  # pre-RoPE, fp32
        )
        block_v = grad_v.to(torch.float32)
        self._absorb_columns(block_k, block_v, grad_pos)

    def _absorb_block_slash(self, grad_k: Tensor, grad_v: Tensor, grad_pos: Tensor) -> None:
        """SLASH: split the graduating block + the current exact heavy-hitter tier
        into (i) the top-``hh_budget`` tokens, kept **verbatim** (post-RoPE K, raw
        V -- exactly like sinks), and (ii) the rest, absorbed into the low-rank
        tail via :meth:`_absorb_columns`. Because the exact peaks never pass
        through ``augmented_bug_step``, the tracked rank-``r`` basis summarizes the
        *outlier-removed residual* spectrum. Demoted former-heavy-hitters re-enter
        the tail at their true (non-contiguous) positions.

        Selection rule (``hh_select="surprise"``, Week-11 SurpriseSLASH): keep the
        highest low-rank *surprise* (out-of-subspace residual) tokens, recomputed
        for the whole candidate pool against the current basis each absorb (so a
        token now captured by the grown basis is demoted, while a persistently
        un-representable outlier -- a sharp needle -- keeps residual ~1 and is
        never demoted). Attach-free (no attention hook) and stores no score."""
        # Candidate pool = current exact tier + graduating block (all post-RoPE).
        if self.hh_k is not None:
            assert self.hh_v is not None and self.hh_pos is not None
            cand_k = torch.cat([self.hh_k, grad_k], dim=1)
            cand_v = torch.cat([self.hh_v, grad_v], dim=1)
            cand_pos = torch.cat([self.hh_pos, grad_pos])
        else:
            cand_k, cand_v, cand_pos = grad_k, grad_v, grad_pos
        n_cand = int(cand_k.shape[1])
        keep_n = min(self.hh_budget, n_cand)
        # Score the whole pool by its CURRENT out-of-subspace residual; the raw
        # cand_k_pre is kept for the demote path.
        cand_k_pre = (
            cand_k.to(torch.float32)
            if self.rope_basis == "post"
            else self._mat_rope_at(cand_k, cand_pos, inverse=True)
        )
        # Week-15 T2: selection may score against the leading score_rank basis
        # columns only (None = full basis). The tail-retention surprise snapshot
        # in _absorb_columns stays UNCAPPED.
        sel = self._surprise_scores(cand_k_pre, cap=self.score_rank)
        if self.hh_neighbor > 0:  # span-boost so needle neighbours are kept too
            sel = self._span_boost(sel, cand_pos, self.hh_neighbor)
        # Highest selection score stays exact; ties fall back to recency.
        order = torch.argsort(sel, stable=True, descending=True)
        keep = order[:keep_n].sort().values  # chronological for clean assembly
        demote = order[keep_n:].sort().values
        self.hh_k = cand_k[:, keep].clone()
        self.hh_v = cand_v[:, keep].clone()
        self.hh_pos = cand_pos[keep].clone()
        if demote.numel() > 0:
            dem_pos = cand_pos[demote]
            dem_k_pre = cand_k_pre[:, demote]  # already un-rotated above
            dem_v = cand_v[:, demote].to(torch.float32)
            self._absorb_columns(dem_k_pre, dem_v, dem_pos)

    def _cast_gist(self, dtype: torch.dtype) -> None:
        """Cast the stored gist -- ``U``, ``B`` and the coordinate tier of both streams --
        to ``dtype``. Called twice per absorb under a non-fp32 ``gist_dtype``: up to fp32
        before the step, back down once the new coordinates have landed. The quantized
        tier is not touched: its codes carry their own width."""
        for name in ("u_k", "b_k", "c_k", "u_v", "b_v", "c_v"):
            stored: Tensor | None = getattr(self, name)
            if stored is not None:
                setattr(self, name, stored.to(dtype))

    def _absorb_columns(self, block_k: Tensor, block_v: Tensor, positions: Tensor) -> None:
        """One augmented BUG step + coordinate carry + budget enforcement for a
        block of ``m`` new columns at the given ``positions`` (``(m,)`` int64; may
        be non-contiguous when heavy-hitters are demoted back into the tail).

        Under ``gist_dtype=bfloat16`` the whole method runs on fp32 working copies of the
        gist -- the step, the coordinate carry, the quantized-tier rotation, the guard's
        repair and the new coordinates -- and the store is rounded back once, at the end.
        What that costs, and why the guard then repairs every absorb but the first, is in
        the ``orth_fix_tol`` comment in ``__init__`` and in the bf16 arm's ``doc:``.

        If :class:`OrthonormalityError` aborts out of :meth:`_guard_orthonormality`, the
        final downcast below never runs and the gist is left in fp32 even under
        ``gist_dtype=bfloat16`` -- inert, since the runners (``kvdlra.eval.frontier``)
        build a fresh cache per sample and record the failure instead of reusing the
        aborted one, so no footprint is ever taken on that path."""
        m = int(block_k.shape[1])
        if m == 0:
            return
        if self.gist_dtype != torch.float32:
            self._cast_gist(torch.float32)
        # Week-9 D3: low-rank surprise = out-of-subspace fraction of the incoming
        # K columns, measured against the basis built from *strictly older*
        # history (pre-step u_k -- the true novelty signal). By Pythagoras (u_k
        # orthonormal): ||k||^2 = ||u_k^T k||^2 + ||k - u_k u_k^T k||^2, so the
        # residual norm is sqrt(||k||^2 - ||c_old||^2). Normalized to [0, 1] =
        # sin of the angle between k and span(u_k) (scale-free across columns).
        surprise: Tensor | None = None
        if self.track_surprise:
            surprise = self._surprise_scores(block_k)
        r_old_k = 0 if self.u_k is None else int(self.u_k.shape[1])
        r_old_v = 0 if self.u_v is None else int(self.u_v.shape[1])
        # Looked up in TRACKERS at call time, not bound at import: the dispatch table is
        # the seam the reconstruction study and the tests substitute a step through.
        if self.tracker == "oja":  # Oja's rule, on the arm's own schedule
            # The decay needs TOKENS SEEN, which only grows. Deriving it from the
            # coordinate tiers' occupancy froze it the moment they hit their budget, so
            # eta stopped decaying for the whole saturated tail of every long context.
            n_seen = self._tokens_seen(positions)
            step = TRACKERS["oja"]
            self.u_k, self.b_k, rot_k = step(
                self.u_k,
                self.b_k,
                block_k,
                self.rank,
                n_seen=n_seen,
                eta0=self.oja_eta0,
                decay=self.oja_decay,
            )
            self.u_v, self.b_v, rot_v = step(
                self.u_v,
                self.b_v,
                block_v,
                self.rank,
                n_seen=n_seen,
                eta0=self.oja_eta0,
                decay=self.oja_decay,
            )
        elif self.tracker == "fd":  # Frequent Directions shrinkage
            step = TRACKERS["fd"]
            self.u_k, self.b_k, rot_k = step(self.u_k, self.b_k, block_k, self.rank)
            self.u_v, self.b_v, rot_v = step(self.u_v, self.b_v, block_v, self.rank)
        elif self.tracker == "frozen":  # Gate-1: track the warm-up window, then stop
            # The SAME monotone frontier Oja's decay reads, for the same reason: derived
            # from the coordinate tiers' occupancy it would stop at their budget, and a
            # context longer than the budget would never reach the freeze at all.
            n_seen = self._tokens_seen(positions)
            step = TRACKERS["frozen"]
            self.u_k, self.b_k, rot_k = step(
                self.u_k,
                self.b_k,
                block_k,
                self.rank,
                n_seen=n_seen,
                freeze_after=self.freeze_after,
                theta=self.theta,
                min_sv_frac=self.min_sv_frac,
            )
            self.u_v, self.b_v, rot_v = step(
                self.u_v,
                self.b_v,
                block_v,
                self.rank,
                n_seen=n_seen,
                freeze_after=self.freeze_after,
                theta=self.theta,
                min_sv_frac=self.min_sv_frac,
            )
        elif self.tracker == "random":  # Gate-1 floor: one seeded draw, never updated
            # Per layer AND per stream, so the 2L bases of a model are 2L independent
            # draws and K and V never share one.
            step = TRACKERS["random"]
            seed = self.basis_seed + 2 * self.layer_idx
            self.u_k, self.b_k, rot_k = step(self.u_k, self.b_k, block_k, self.rank, seed=seed)
            self.u_v, self.b_v, rot_v = step(self.u_v, self.b_v, block_v, self.rank, seed=seed + 1)
        else:  # "isvd" today -- unchanged, bit-identical; looked up by name (fix1 R3) so a
            # fourth TRACKERS entry lands here and fails loudly on theta=/min_sv_frac=
            # instead of silently running the incremental-SVD step.
            step = TRACKERS[self.tracker]
            self.u_k, self.b_k, rot_k = step(
                self.u_k,
                self.b_k,
                block_k,
                self.rank,
                theta=self.theta,
                min_sv_frac=self.min_sv_frac,
            )
            self.u_v, self.b_v, rot_v = step(
                self.u_v,
                self.b_v,
                block_v,
                self.rank,
                theta=self.theta,
                min_sv_frac=self.min_sv_frac,
            )
        # Carry held fp32 coordinates into the new basis.
        if self.c_k is not None:
            assert self.c_v is not None
            self.c_k = rot_k @ self.c_k
            self.c_v = rot_v @ self.c_v
        # Second tier: variant D re-expresses its codes into the new basis
        # (dequantize -> rot @ -> requantize, the compounding path).
        if self._q_len() > 0:
            self._rotate_quant_tier(rot_k, rot_v)
        # Orthonormality tripwire + guard, BEFORE the new coordinates are taken: they must
        # be measured in the basis the repair leaves behind, not in the broken one.
        self._guard_orthonormality(
            rank_changed_k=int(self.u_k.shape[1]) != r_old_k,
            rank_changed_v=int(self.u_v.shape[1]) != r_old_v,
            positions=positions,
        )
        # Append the graduating block's coordinates (fp32 tier).
        new_ck = self.u_k.mT @ block_k
        new_cv = self.u_v.mT @ block_v
        self.c_k = new_ck if self.c_k is None else torch.cat([self.c_k, new_ck], dim=1)
        self.c_v = new_cv if self.c_v is None else torch.cat([self.c_v, new_cv], dim=1)
        if self.track_positions:
            pos = positions.to(dtype=torch.int64, device=block_k.device)
            self.mid_pos = pos if self.mid_pos is None else torch.cat([self.mid_pos, pos])
        if self.track_surprise and surprise is not None:
            self.mid_surprise = (
                surprise if self.mid_surprise is None else torch.cat([self.mid_surprise, surprise])
            )
        self._enforce_budgets()
        if self.gist_dtype != torch.float32:  # round the store back down, once per absorb
            self._cast_gist(self.gist_dtype)

    def _tokens_seen(self, positions: Tensor) -> int:
        """Monotone tokens-seen frontier at the absorb of ``positions``: a per-layer running
        maximum of the cache's own counter and this block's own positions (L1.1 fix1
        Important #1; made a running maximum in fix1 R2).

        Single-shot pre-fill only advances the counter after its whole absorb loop, so the
        counter alone reads 0 for every absorb inside that loop and ``positions`` is what is
        exact there; on the heavy-hitter demote path a demoted token's position can be OLDER
        than every position absorbed so far in the SAME prefill call (a later sub-block can
        demote a now-unsurprising resident that arrived before the previous sub-block's own
        demotees), so ``max(cumulative_length, positions.max() + 1)`` alone can dip between
        two calls within one prefill. The running maximum is what never dips. Both the
        diagnostic row's ``tokens_seen`` and Oja's learning-rate decay read this method -- one
        expression, so a schedule and the row that reports it cannot disagree.
        """
        candidate = max(self.cumulative_length, int(positions.max()) + 1)
        self._tokens_seen_max = max(self._tokens_seen_max, candidate)
        return self._tokens_seen_max

    def _guard_orthonormality(
        self, rank_changed_k: bool, rank_changed_v: bool, positions: Tensor
    ) -> None:
        """One absorb's orthonormality tripwire, guard and diagnostic row.

        ``‖UᵀU - I‖`` is measured on each stream's basis as the step just returned it --
        the fp32 working copy of the stored basis, which under ``gist_dtype=bfloat16``
        carries the previous absorb's rounding (~5e-3 at n=1024, r=64) through the step.
        Above ``orth_fix_tol`` -- or whenever ``qr_every`` says so for that stream -- the
        offending stream is re-orthonormalized, its coordinates and (once, for both streams
        together) its quantized tier carried into the repaired basis. The trial fails loudly
        only if the REPAIRED basis is still above ``orth_abort_tol``: a rank-deficient block
        whose SVD came back non-orthonormal is a one-step numerical event a thin QR fixes,
        and checking the abort tolerance first failed every guarded arm on three Table-4
        pods at absorb 1 (D-011 addendum 5). The window keeps the PRE-repair error, so the
        ratchet trace is the raw measurement, and the fatal window is flushed BEFORE the
        raise, so a failed trial's drained rows contain it (fix1 Minor #3) rather than a gap.
        Every ``diag_every`` absorbs one row per layer records the window's worst error, its
        lowest effective rank, the live rank and whether a repair fired;
        ``BugStreamingCache.drain_diag()`` collects them. ``tokens_seen`` is
        :meth:`_tokens_seen` of this absorb -- see its docstring for why the cache's own
        counter is not it.
        """
        assert self.u_k is not None and self.u_v is not None
        assert self.b_k is not None and self.b_v is not None
        self._absorbs += 1
        err_k = orth_error(self.u_k)
        err_v = orth_error(self.u_v)
        self._diag_window += 1
        self._diag_tokens = self._tokens_seen(positions)
        # The PRE-repair error AND effective rank are what the window carries on every
        # path (fix1 Minor #3, fix1 A2): the ratchet trace is the raw measurement, not
        # what a repair leaves behind -- the pods' own defect showed up as a pre-repair
        # `eff_rank_v` 93 of 128, and a repair that re-derives the core from a fresh SVD
        # can read back as full rank with nothing about the defect having changed. The
        # fold is NaN-sticky (fix2, ``_nan_max``): plain ``max`` drops a NaN landing in
        # the second argument, losing it from the window instead of reporting it.
        self._diag_max_err_k = _nan_max(self._diag_max_err_k, err_k)
        self._diag_max_err_v = _nan_max(self._diag_max_err_v, err_v)
        self._diag_min_rank_k = min(self._diag_min_rank_k, eff_rank(self.b_k))
        self._diag_min_rank_v = min(self._diag_min_rank_v, eff_rank(self.b_v))
        # ``qr_every`` (the experimental factor) repairs unconditionally every k absorbs
        # and after that stream's own rank change, whatever the measurement says. NaN is
        # never "at or below" a tolerance (fix1 A3): ``not (err <= tol)`` repairs on
        # a NaN measurement, where the old ``err > tol`` would have silently let it pass.
        forced_k = self.qr_every is not None and (
            self._absorbs % self.qr_every == 0 or rank_changed_k
        )
        forced_v = self.qr_every is not None and (
            self._absorbs % self.qr_every == 0 or rank_changed_v
        )
        fixed_k = forced_k or (self.orth_fix_tol is not None and not (err_k <= self.orth_fix_tol))
        fixed_v = forced_v or (self.orth_fix_tol is not None and not (err_v <= self.orth_fix_tol))
        rot_fix_k: Tensor | None = None
        rot_fix_v: Tensor | None = None
        if fixed_k:
            # Before the first graduating block there are no coordinates to carry yet.
            c_k = self.c_k if self.c_k is not None else self.u_k.new_zeros(self.u_k.shape[1], 0)
            self.u_k, c_k, self.b_k, rot_fix_k = reorthonormalize(self.u_k, c_k, self.b_k)
            if self.c_k is not None:
                self.c_k = c_k
        if fixed_v:
            c_v = self.c_v if self.c_v is not None else self.u_v.new_zeros(self.u_v.shape[1], 0)
            self.u_v, c_v, self.b_v, rot_fix_v = reorthonormalize(self.u_v, c_v, self.b_v)
            if self.c_v is not None:
                self.c_v = c_v
        if (fixed_k or fixed_v) and self._q_len() > 0:
            # One rotation for both streams. The unfixed side passes ``None``: an identity
            # rotation is not a no-op through the quantizer, it re-codes and injects
            # distortion the tier never had to take.
            self._rotate_quant_tier(rot_fix_k, rot_fix_v)
        self._diag_fixed_k = self._diag_fixed_k or fixed_k
        self._diag_fixed_v = self._diag_fixed_v or fixed_v
        if self.orth_abort_tol is not None:
            # Only a repaired stream is re-measured: an untouched one still carries the
            # error measured above, and (``orth_abort_tol >= orth_fix_tol``, validated in
            # the constructor) it cannot be over the abort tolerance without a repair.
            post_k = orth_error(self.u_k) if fixed_k else err_k
            post_v = orth_error(self.u_v) if fixed_v else err_v
            # NaN is never "at or below" the abort tolerance either (fix1 A3): a
            # repair whose OWN post-measurement comes back NaN must abort, not be read as
            # silently within bounds. Checked per stream, not aggregated by ``max()``
            # (fix2): ``max(post_k, post_v)`` silently drops a NaN landing in the second
            # argument (``max(0.0, nan) == 0.0``), so a NaN-V reading next to a finite K
            # never aborted.
            if any(not (e <= self.orth_abort_tol) for e in (post_k, post_v)):
                self._flush_diag_window()
                raise OrthonormalityError(
                    f"layer={self.layer_idx} orth_err_k={post_k:.3e} orth_err_v={post_v:.3e} > "
                    f"abort_tol={self.orth_abort_tol} at absorb {self._absorbs}"
                )
        if self._absorbs % self.diag_every == 0:
            self._flush_diag_window()

    def _flush_diag_window(self) -> None:
        """Append one row from the current window and reset it -- shared by the periodic
        (``diag_every``) flush, the abort path (fix1 Minor #3) and the partial-window
        flush :meth:`BugStreamingCache.drain_diag` does, so there is exactly one place
        that resets the window counters."""
        assert self.u_k is not None and self.u_v is not None
        self.diag.append(
            {
                "layer": self.layer_idx,
                "absorbs": self._absorbs,
                "tokens_seen": self._diag_tokens,
                "orth_err_k": self._diag_max_err_k,
                "orth_err_v": self._diag_max_err_v,
                "eff_rank_k": self._diag_min_rank_k,
                "eff_rank_v": self._diag_min_rank_v,
                "rank_k": int(self.u_k.shape[1]),
                "rank_v": int(self.u_v.shape[1]),
                "fixed_k": self._diag_fixed_k,
                "fixed_v": self._diag_fixed_v,
            }
        )
        self._diag_window = 0
        self._diag_max_err_k = self._diag_max_err_v = 0.0
        self._diag_min_rank_k = self._diag_min_rank_v = self.rank
        self._diag_fixed_k = self._diag_fixed_v = False

    def _surprise_scores(self, block_k: Tensor, cap: int | None = None) -> Tensor:
        """Low-rank surprise of pre-RoPE K columns ``block_k`` ``(n, m)``: the
        out-of-subspace residual fraction ``||k - U Uᵀk|| / ||k||`` against the
        current (pre-step) basis ``self.u_k``. By Pythagoras (``u_k``
        orthonormal): ``||k||² = ||u_kᵀk||² + ||k - u_k u_kᵀk||²``, so the residual
        norm is ``sqrt(||k||² - ||c_old||²)``; normalized to ``[0, 1]`` (sin of the
        angle to ``span(u_k)``, scale-free across columns). ``1.0`` when there is
        no basis yet. Shared by ``retention="lowrank_surprise"`` graduation
        snapshots and Week-11 SurpriseSLASH exact-tier selection.

        Week-15 T2: ``cap`` scores against only the leading ``cap`` basis columns
        ``u_k[:, :cap]`` -- the energy-dominant subspace (the integrator's SVD
        orders columns by accumulated-history singular value every step), itself
        orthonormal, so the Pythagoras identity holds unchanged. This decouples
        *selection* rank from *storage* rank: a big basis fits needles (residual
        -> 0 -> never selected -- the measured rank-retrieval coupling), while
        the leading-``cap`` subview keeps them surprising. ``None`` = full basis
        (bit-for-bit the pre-Week-15 behaviour); a no-op until ``u_k`` has more
        than ``cap`` columns (effective cap ``min(cap, u_k.shape[1])``)."""
        m = int(block_k.shape[1])
        k_norm = block_k.norm(dim=0)
        if self.u_k is None:  # first absorb: nothing older to predict from
            return torch.ones(m, dtype=torch.float32, device=block_k.device)
        u = self.u_k if cap is None else self.u_k[:, : min(cap, int(self.u_k.shape[1]))]
        # fp32 for the product (a no-op at the fp32 default): under ``gist_dtype=bfloat16``
        # the stored basis is bf16 while ``block_k`` is fp32. This is also the SLASH
        # selection path, which reads the basis BEFORE the absorb's upcast.
        c_old = u.to(torch.float32).mT @ block_k  # (r, m) coords in the OLD (capped) basis
        resid = (k_norm**2 - c_old.norm(dim=0) ** 2).clamp_min(0.0).sqrt()
        return cast(Tensor, (resid / k_norm.clamp_min(1e-12)).to(torch.float32))

    def _span_boost(self, sel: Tensor, positions: Tensor, w: int) -> Tensor:
        """Position-windowed max of ``sel``: each token inherits the highest
        surprise among tokens within ``+-w`` positions (SnapKV-style span
        expansion). A non-outlier token adjacent to a sharp needle outlier is thus
        pulled toward the top-``hh_budget``, so the exact tier keeps the whole
        contiguous needle span verbatim -- while the hard ``hh_budget`` cap (and so
        the mask-length machinery) is untouched (it re-ranks, never grows the tier).
        """
        n = int(sel.shape[0])
        if w <= 0 or n <= 1:
            return sel
        order = torch.argsort(positions)  # ascending true position
        pos_s = positions[order]
        boosted_s = sel[order].clone()
        src_s = sel[order]
        # Two tokens within +-w positions are within +-w sorted offsets (positions
        # are unique), so a small offset sweep with a gap guard is exact.
        for off in range(1, min(w, n - 1) + 1):
            near = (pos_s[off:] - pos_s[:-off]) <= w  # gap between sorted i, i+off
            boosted_s[:-off] = torch.where(
                near, torch.maximum(boosted_s[:-off], src_s[off:]), boosted_s[:-off]
            )
            boosted_s[off:] = torch.where(
                near, torch.maximum(boosted_s[off:], src_s[:-off]), boosted_s[off:]
            )
        out = torch.empty_like(sel)
        out[order] = boosted_s
        return out

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
        # "lowrank_surprise": evict the LOWEST-surprise (best-reconstructed =
        # most redundant) columns; the ascending argsort below keeps the
        # high-residual outliers the low-rank summary cannot reproduce.
        scores = self.mid_surprise if tier == "fp32" else self.q_surprise
        assert scores is not None
        order = torch.argsort(scores, stable=True)  # ascending; ties keep age order
        evict = order[:n_out].sort().values
        keep = order[n_out:].sort().values
        return evict, keep

    def _enforce_budgets(self) -> None:
        """Evict down to ``coord_budget`` (demoting to the quantized tier when
        enabled) and then the quantized tier down to ``quant_budget``."""
        f_over = self._f_len() - self.coord_budget
        if f_over > 0:
            assert self.c_k is not None and self.c_v is not None
            evict, keep = self._split_indices(f_over, tier="fp32")
            out_ck = self.c_k[:, evict]
            out_cv = self.c_v[:, evict]
            self.c_k = self.c_k[:, keep]
            self.c_v = self.c_v[:, keep]
            out_pos: Tensor | None = None
            out_surprise: Tensor | None = None
            if self.track_positions:
                assert self.mid_pos is not None
                out_pos = self.mid_pos[evict]
                self.mid_pos = self.mid_pos[keep]
            if self.track_surprise:
                assert self.mid_surprise is not None
                out_surprise = self.mid_surprise[evict]
                self.mid_surprise = self.mid_surprise[keep]
            if self.quant_bits is not None:
                self._append_quant(out_ck, out_cv, out_pos, out_surprise)
        q_over = self._q_len() - self.quant_budget
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
            if self.track_surprise:
                assert self.q_surprise is not None
                self.q_surprise = self.q_surprise[keep]

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
        if self.track_surprise:
            assert surprise is not None
            self.q_surprise = (
                surprise if self.q_surprise is None else torch.cat([self.q_surprise, surprise])
            )

    def _rotate_quant_tier(self, rot_k: Tensor | None, rot_v: Tensor | None) -> None:
        """Carry the quantized tier across a basis update: dequantize -> ``rot @``
        -> requantize. Column *norms* are carried exactly (:meth:`_dequantize`
        renormalizes the decoded direction, so the requantized norm is
        ``norm_old * ||rot @ u_hat||`` -- the true rotation-induced contraction,
        with no quantizer-induced drift). The *direction* is re-coded each
        absorb with error bounded by the one-shot PolarQuant distortion per
        event; whether that per-event jitter compounds visibly over deep
        horizons is exactly what the Week-7 bin curves falsify.

        ``None`` leaves that stream's tier untouched (the orthonormality guard can repair
        one stream and not the other): passing an identity rotation instead would still
        dequantize and re-code the tier, i.e. spend a re-coding event on nothing."""
        assert self.qk_codes is not None and self.qk_norm is not None
        assert self.qv_codes is not None and self.qv_norm is not None
        if rot_k is not None:
            ck = rot_k @ self._dequantize(self.qk_codes, self.qk_norm)
            self.qk_codes, self.qk_norm = self._quantize_cols(ck, "K")
        if rot_v is not None:
            cv = rot_v @ self._dequantize(self.qv_codes, self.qv_norm)
            self.qv_codes, self.qv_norm = self._quantize_cols(cv, "V")

    # -------------------------------------------------------------- update

    def update(
        self, key_states: Tensor, value_states: Tensor, *args: object, **kwargs: object
    ) -> tuple[Tensor, Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self._kernel_step = False
        if self._rows is not None or int(key_states.shape[0]) != 1:
            return self._update_rows(key_states, value_states)
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

    def _update_rows(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        """Batch > 1: one independent ``BugStreamingLayer`` per row, built from this layer's
        own constructor kwargs at the first update (when B is known) and each driven through
        the unchanged batch-1 path on its ``(1, H, T, D)`` slice; the returned K/V are the
        rows' outputs concatenated on dim 0. Rows share the quant bank (side information by
        design) and nothing else. Every row sees the same token count, so their returned
        lengths agree by construction; a disagreement is a bug and raises rather than pads."""
        b = int(key_states.shape[0])
        if self._rows is None:
            if self.cumulative_length != 0:
                # R-L4-8: the rows would start empty while this layer already carries a
                # batch-1 stream, so the batched run would silently attend to nothing.
                raise NotImplementedError(
                    f"BugStreamingLayer: a batch-{b} update cannot follow batch-1 updates on "
                    f"the same layer (cumulative_length={self.cumulative_length}); reset() first"
                )
            self._rows = [BugStreamingLayer(**self._ctor_kwargs) for _ in range(b)]
            for row in self._rows:
                row._mode = self._mode
        if b != len(self._rows):
            raise NotImplementedError(
                f"BugStreamingLayer was built for batch {len(self._rows)}, got batch {b}: "
                "a cache serves one batch of equal-length rows for its whole life"
            )
        outs = [
            row.update(key_states[i : i + 1], value_states[i : i + 1])
            for i, row in enumerate(self._rows)
        ]
        lengths = [int(k.shape[2]) for k, _ in outs]
        if len(set(lengths)) != 1:
            raise NotImplementedError(
                f"ragged rows: the per-row returned lengths differ ({lengths}); equal-length "
                "rows are the only batched input this cache serves"
            )
        return torch.cat([k for k, _ in outs]), torch.cat([v for _, v in outs])

    def _score_forward(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        """Week-10 frozen continuation scoring (non-mutating): attend the ``q_len``
        window to ``[retained | window]`` without touching the streaming state, so
        one forward scores perplexity over the compressed cache. The retained block
        is strictly-past (true-position RoPE preserved); ``get_mask_sizes`` reports
        ``attended_length() + q_len`` so the causal mask sees it all. No absorb, no
        recent append, no position mutation -- ``stored_state_numel`` is
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
        self.cumulative_length += q
        return self._decode_peek()

    def consolidate(self) -> None:
        """Run the deferred absorb for an over-full recent ring after a chunked
        :meth:`_ingest_chunk` (same block schedule / loop as :meth:`_decode_step`)."""
        for row in self._rows or ():
            row.consolidate()
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

        if mid > 0 and self.lowrank_enabled:
            # Week-13 T-B: when seeding (chunked ingest + an exact tier), route the
            # first chunk's middle through the SLASH path in sub-blocks. Each
            # sub-block's surprise is scored against the basis of STRICTLY OLDER
            # sub-blocks (which have not seen it), so an early outlier is caught
            # verbatim into the exact tier exactly as in steady-state decode --
            # shrinking the warm-up window from the whole first chunk to one
            # sub-block. Scoring against a needle-free basis is the crux: the
            # rejected "warm-then-select" variant scored the middle against a basis
            # that had already absorbed the needle, so the needle no longer read as
            # surprising (CPU-verified failure at realistic rank/diversity). Off (the
            # default) absorbs the middle directly, bit-for-bit the prior path.
            seed = self.seed_hh_warmup and self.hh_enabled and self._mode == "ingest"
            k_pre = (
                k_mat[:, n_sink : n_sink + mid].to(torch.float32)
                if self.rope_basis == "post"
                else self._mat_rope(k_mat[:, n_sink : n_sink + mid], n_sink, inverse=True)
            )
            v_mid = v_mat[:, n_sink : n_sink + mid].to(torch.float32)
            for start in range(0, mid, self.prefill_block_size):
                stop = min(mid, start + self.prefill_block_size)
                pos = torch.arange(
                    n_sink + start, n_sink + stop, dtype=torch.int64, device=k_pre.device
                )
                if seed:
                    # post-RoPE K + raw V, exactly as the steady-state graduating
                    # block feeds _absorb_block_slash from the recent ring.
                    self._absorb_block_slash(
                        k_mat[:, n_sink + start : n_sink + stop],
                        v_mat[:, n_sink + start : n_sink + stop],
                        pos,
                    )
                else:
                    self._absorb_columns(k_pre[:, start:stop], v_mid[:, start:stop], pos)
        self.cumulative_length = t
        return key_states, value_states

    def _decode_step(self, key_states: Tensor, value_states: Tensor) -> tuple[Tensor, Tensor]:
        assert self.recent_k is not None and self.recent_v is not None
        self.recent_k = torch.cat([self.recent_k, self._to_mat(key_states)], dim=1)
        self.recent_v = torch.cat([self.recent_v, self._to_mat(value_states)], dim=1)
        self.cumulative_length += 1
        while self._recent_len() >= self.recent_window + self.absorb_block:
            self._absorb_block_into_stream(self.absorb_block)
        if self.decode_attention == "kernel":
            self._kernel_mid = self._factored_middle()
            self._kernel_step = True
            return self._decode_peek(dense_only=True)
        return self._decode_peek()

    def _decode_peek(self, *, dense_only: bool = False) -> tuple[Tensor, Tensor]:
        """Assemble the currently retained ``[sinks | middle-hat | recent]`` K/V in
        the HF ``(1, H, L, D)`` layout (also used by tests for introspection).

        ``dense_only`` (the kernel path) leaves the low-rank tail out: ``[sinks | exact tier
        | recent]`` -- the attention function attends the tail from (U, C)."""
        assert self.recent_k is not None and self.recent_v is not None
        parts_k: list[Tensor] = []
        parts_v: list[Tensor] = []
        if self.sink_k is not None and self.sink_k.shape[1] > 0:
            assert self.sink_v is not None
            parts_k.append(self.sink_k)
            parts_v.append(self.sink_v)
        if self._hh_attended_len() > 0:  # SLASH exact tier: verbatim post-RoPE, like sinks
            assert self.hh_k is not None and self.hh_v is not None
            parts_k.append(self.hh_k)
            parts_v.append(self.hh_v)
        if not dense_only and self._q_len() + self._f_len() > 0:  # the low-rank (recon) tail
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
        Contiguous by construction under FIFO -- the single source of truth for
        ``mid_start`` that `_ensure_mid_cache` and `_factored_middle` both read
        through (R-L4-18): it must leave room for `_hh_attended_len()` columns
        ahead of this block, not just this block's own length, or a live exact
        tier shifts the kernel middle's positions off the reconstruct path's."""
        device = self.c_k.device if self.c_k is not None else torch.device("cpu")
        if not self.track_positions:
            lr_len = self._q_len() + self._f_len()
            mid_start = self.cumulative_length - self._recent_len() - self._mid_len()
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
        # fp32 for the products (no-ops at the fp32 default): the STORED gist may be bf16,
        # while the dequantized tier and the RoPE round trip below are fp32. What is
        # reconstructed is the stored representation -- the bf16 rounding is inside it,
        # which is the point of the arm.
        u_k, u_v = self.u_k.to(torch.float32), self.u_v.to(torch.float32)
        parts_k: list[Tensor] = []
        parts_v: list[Tensor] = []
        if self._q_len() > 0:
            assert self.qk_codes is not None and self.qv_codes is not None
            assert self.qk_norm is not None and self.qv_norm is not None
            parts_k.append(u_k @ self._dequantize(self.qk_codes, self.qk_norm))
            parts_v.append(u_v @ self._dequantize(self.qv_codes, self.qv_norm))
        if self._f_len() > 0:
            assert self.c_k is not None and self.c_v is not None
            parts_k.append(u_k @ self.c_k.to(torch.float32))
            parts_v.append(u_v @ self.c_v.to(torch.float32))
        k_pre_hat = torch.cat(parts_k, dim=1) if len(parts_k) > 1 else parts_k[0]
        v_hat = torch.cat(parts_v, dim=1) if len(parts_v) > 1 else parts_v[0]
        if self.rope_basis == "post":
            k_hat = k_pre_hat  # stored post-RoPE: nothing to re-rotate
        elif self.track_positions:
            k_hat = self._mat_rope_at(k_pre_hat, self._mid_positions(), inverse=False)
        else:
            # Contiguous middle: the memoized range path (bit-identical to Week 6). The
            # start comes from `_mid_positions()` (R-L4-18: one source of truth for both
            # this reconstruct path and the kernel path's `_factored_middle`) -- taken as
            # a scalar rather than switching to `_mat_rope_at` so this hot path keeps
            # `_mat_rope`'s per-(start, length) memo across layers.
            mid_start = int(self._mid_positions()[0])
            k_hat = self._mat_rope(k_pre_hat, mid_start, inverse=False)
        self._mid_k_cache = k_hat.to(self.dtype)
        self._mid_v_cache = v_hat.to(self.dtype)

    def _mid_coords(self) -> tuple[Tensor, Tensor]:
        """The retained low-rank columns' coordinates in the current basis, fp32 ``(r, T)``
        each, assembly order ``[quant | fp32]`` -- the same columns `_ensure_mid_cache`
        multiplies out, left factored here. The quantized tier is dequantized into fp32
        coordinates (`_dequantize`); the fp32 tier is upcast from ``gist_dtype``."""
        parts_k: list[Tensor] = []
        parts_v: list[Tensor] = []
        if self._q_len() > 0:
            assert self.qk_codes is not None and self.qv_codes is not None
            assert self.qk_norm is not None and self.qv_norm is not None
            parts_k.append(self._dequantize(self.qk_codes, self.qk_norm))
            parts_v.append(self._dequantize(self.qv_codes, self.qv_norm))
        if self._f_len() > 0:
            assert self.c_k is not None and self.c_v is not None
            parts_k.append(self.c_k.to(torch.float32))
            parts_v.append(self.c_v.to(torch.float32))
        c_k = torch.cat(parts_k, dim=1) if len(parts_k) > 1 else parts_k[0]
        c_v = torch.cat(parts_v, dim=1) if len(parts_v) > 1 else parts_v[0]
        return c_k, c_v

    def _factored_middle(self) -> FactoredMiddle | None:
        """The low-rank middle as the kernel attends it (batch dim 1), or ``None`` before the
        first absorb (nothing factored yet: attention then sees the dense tokens only)."""
        if self._q_len() + self._f_len() == 0:
            return None
        assert self.u_k is not None and self.u_v is not None
        c_k, c_v = self._mid_coords()
        return FactoredMiddle(
            u_k=self.u_k.unsqueeze(0),
            u_v=self.u_v.unsqueeze(0),
            c_k=c_k.unsqueeze(0),
            c_v=c_v.unsqueeze(0),
            positions=self._mid_positions().unsqueeze(0),
        )

    def kernel_middles(self) -> list[FactoredMiddle | None] | None:
        """What the registered attention function attends this step: one middle per batch
        row (``None`` for a row with nothing factored yet), or ``None`` when the last update
        was not a kernel-mode decode step (prefill, ingest, score, or the reconstruct path)
        -- the function then delegates to sdpa on the returned K/V."""
        layers = self._rows or [self]
        if not all(la._kernel_step for la in layers):
            return None
        return [la._kernel_mid for la in layers]

    # ------------------------------------------------------------ cache API

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Report exactly the K/V length ``update()`` will return this step; the
        offset places the (strictly past) returned block right below the query's
        true position so the causal mask sees it all as visible."""
        if self._rows:  # every row reports the same sizes (they share a token count)
            return self._rows[0].get_mask_sizes(query_length)
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
        return self._rows[0].cumulative_length if self._rows else self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self) -> None:
        for row in self._rows or ():  # the drained-once contract survives a reset
            self.diag.extend(row.drain_diag())
        self._reset_state()

    # ---------------------------------------------------------- accounting

    def stored_state_numel(self) -> int:
        """Float-equivalents of the *stored* per-layer state (what the accounting
        bills): fp32/verbatim tensors at 1 each; quantized codes at ``quant_bits/32``
        each (bit-packable) + their fp32 norms; retention positions (int32) and
        surprise snapshots at 1 each. Shared quantizer side info is counted once
        at the cache level (:meth:`BugStreamingCache.stored_state_numel`)."""
        if self._rows:
            return sum(row.stored_state_numel() for row in self._rows)
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
        )
        total = sum(t.numel() for t in tensors if t is not None)
        # The square-root core B is *provably diagonal*: both writers keep it so.
        # ``augmented_bug_step`` returns ``diag(sigma)`` every ordinary step, and the
        # second writer -- ``reorthonormalize`` (isvd.py), called from the orthonormality
        # guard's ``_guard_orthonormality`` whenever it repairs a stream -- re-diagonalizes
        # it via its own SVD and returns ``torch.diag(sigma)`` at the same r. So its
        # deployable footprint is its ``r`` diagonal entries, not ``r^2`` -- counted that
        # way here (same convention as quant codes at bits/32 rather than their uint8
        # storage). ``tests`` pin the diagonality.
        for core in (self.b_k, self.b_v):
            if core is not None:
                total += int(min(core.shape))
        if self.qk_codes is not None:
            assert self.qv_codes is not None
            # bits/code: PolarQuant (variant D, r codes/token). Codes counted at
            # bits/32 (bit-packable).
            bits = self.quant_bits
            assert bits is not None
            code_entries = int(self.qk_codes.numel() + self.qv_codes.numel())
            total += math.ceil(code_entries * bits / 32)
            # Per-column norms (fp32).
            for nrm in (self.qk_norm, self.qv_norm):
                if nrm is not None:
                    total += int(nrm.numel())
        bookkeeping = (self.mid_pos, self.q_pos, self.hh_pos)
        total += sum(t.numel() for t in bookkeeping if t is not None)
        # Week-9 D3: the per-column surprise buffers are stored state under
        # retention="lowrank_surprise" and are counted.
        if self.track_surprise:
            for surp in (self.mid_surprise, self.q_surprise):
                if surp is not None:
                    total += int(surp.numel())
        return int(total)

    def workspace_numel(self) -> int:
        """Float entries of the cached middle reconstruction (bounded derived
        state, avoidable by recomputing each step; reported separately)."""
        if self._rows:
            return sum(row.workspace_numel() for row in self._rows)
        tensors = (self._mid_k_cache, self._mid_v_cache)
        return sum(t.numel() for t in tensors if t is not None)

    def attended_length(self) -> int:
        """Tokens attention currently sees (sinks + middle + recent)."""
        if self._rows:
            return self._rows[0].attended_length()
        return self._sink_len() + self._mid_len() + self._recent_len()

    def drain_diag(self) -> list[dict[str, object]]:
        """This layer's diagnostic rows (its rows' first, in row order), the open window
        flushed first; the rows are cleared. The cache's ``drain_diag`` calls this."""
        rows: list[dict[str, object]] = []
        for row in self._rows or ():
            rows.extend(row.drain_diag())
        if self._diag_window:
            self._flush_diag_window()
        rows.extend(self.diag)
        self.diag.clear()
        return rows


class BugStreamingCache(Cache):
    """Model-level constant-memory streaming-BUG cache (one layer per model layer).

    Pass as ``past_key_values`` to ``model.generate(...)`` / ``model(...)``.
    Every retention rule reads its signal off the stored keys, so no attention
    hook is needed; :meth:`attach` exists only so a harness can wrap any
    streaming cache the same way.

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
        Coordinate eviction rule: ``"fifo"`` (Week-6 baseline) or
        ``"lowrank_surprise"`` (Week-9 D3: keep the highest-residual outliers).
        See the module docstring.
    quant_bits, quant_budget:
        Week-7 D: demote evicted fp32 coordinates into a PolarQuant tier of
        ``quant_budget`` columns at ``quant_bits`` bits/coordinate (both set,
        or neither). Codes count ``quant_bits/32`` float-equivalents each.
    quant_seed:
        Seed for the shared PolarQuant rotation/codebook.
    gist_dtype:
        L5.1 (default ``torch.float32`` = off, and bit-identical to the pre-L5.1 cache):
        the dtype the gist -- the basis ``U``, the core ``B`` and the coordinate tier
        ``C`` -- is STORED as between absorbs. ``torch.bfloat16`` (or the string
        ``"bfloat16"``, which is what a YAML ``cache:`` block carries) halves those bytes
        and is billed at 16 bits (``accounting.bug_footprint(gist_bits=16)``). Every
        computation still runs in fp32: see :meth:`BugStreamingLayer._absorb_columns`.
    score_rank:
        Week-15 T2 (default ``None`` = off): cap SurpriseSLASH *selection*
        scoring to the leading ``score_rank`` basis columns (``1 <= score_rank
        <= rank``; requires ``hh_select='surprise'`` and ``hh_budget >= 1``).
        Storage rank, tail-retention snapshots and accounting are unchanged.
    orth_fix_tol, orth_abort_tol, qr_every, diag_every:
        Orthonormality guard (CODE_AUDIT Part A §Q4). Each absorb measures
        ``‖UᵀU - I‖`` on both bases: above ``orth_fix_tol`` (default 1e-3, above the ~7e-4
        a benign fp32 stream reaches over 1400 adversarial steps -- but NOT above what a
        bf16 model dtype or ``gist_dtype`` reaches, where it fires routinely) it
        re-orthonormalizes the basis in place, carrying
        coordinates and the quantized tier; if the repaired basis is STILL above
        ``orth_abort_tol`` (default 1e-1, and never below ``orth_fix_tol``) it raises
        :class:`OrthonormalityError`. ``None`` disables either; ``orth_fix_tol=None`` with
        ``orth_abort_tol`` set measures and aborts without ever repairing (fix1 A4).
        ``qr_every`` (default ``None``) repairs unconditionally every k absorbs and after
        that stream's own rank change. ``diag_every`` (default 64) sets the diagnostic
        window; see :meth:`drain_diag`.
    decode_attention, kernel_operand_dtype:
        L4.4 (default ``"reconstruct"`` = today's path, bit-identical). ``"kernel"``: each
        decode step returns only the dense tokens and :meth:`attach` registers the
        factored-attention function (``kvdlra.kernel.attention``) that attends the low-rank
        middle from (U, C) tile by tile, RoPE in-kernel at the true positions. Prefill,
        chunked ingest, the absorb step and the stored state are unchanged, so the footprint
        is the reconstruct arm's; only decode-time work and workspace differ.
        ``kernel_operand_dtype`` (``"bfloat16"`` default, ``"float32"``) is the tensor-core
        input width the kernel rounds to; accumulation is fp32 either way.
    rope_basis:
        ``"pre"`` (default, the shipped operating point) or ``"post"`` -- track post-RoPE keys
        (ADR 0001 §3 option (i)); the accuracy row ``isvd_postrope_r128``.
    recent_window, absorb_block, n_sink, theta, min_sv_frac, prefill_block_size:
        See :class:`BugStreamingLayer`.
    tracker, oja_eta0, oja_decay, freeze_after, basis_seed:
        Which gist tracker the layers run (a key of :data:`kvdlra.tracker.TRACKERS`;
        ``"bug"`` is a deprecated alias for ``"isvd"``) and the knobs the swapped ones
        need: for ``"oja"``, the learning-rate schedule ``eta0 / (1 + decay *
        tokens_seen)``; for ``"frozen"``, the tokens-seen frontier past which the basis
        stops updating (default 4096); for ``"random"``, the base seed of the per-layer,
        per-stream orthonormal draw (``basis_seed + 2*layer_idx``, ``+1`` for V). See
        :class:`BugStreamingLayer`.
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
        min_sv_frac: float = 0.0,
        gist_dtype: torch.dtype | str = torch.float32,
        tracker: str = "isvd",
        oja_eta0: float = 20.0,
        oja_decay: float = 0.03,
        freeze_after: int = 4096,
        basis_seed: int = 0,
        prefill_block_size: int = 128,
        retention: str = "fifo",
        quant_bits: int | None = None,
        quant_budget: int = 0,
        quant_seed: int = 0,
        hh_budget: int = 0,
        hh_select: str = "attn",
        hh_neighbor: int = 0,
        hh_retain: bool = True,
        score_rank: int | None = None,
        seed_hh_warmup: bool = False,
        orth_fix_tol: float | None = 1e-3,
        orth_abort_tol: float | None = 1e-1,
        qr_every: int | None = None,
        diag_every: int = 64,
        decode_attention: str = "reconstruct",
        kernel_operand_dtype: torch.dtype | str = torch.bfloat16,
        rope_basis: str = "pre",
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
        self._quant_bank = _QuantBank(quant_bits, seed=quant_seed) if quant_bits else None
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
                min_sv_frac=min_sv_frac,
                gist_dtype=gist_dtype,
                tracker=tracker,
                oja_eta0=oja_eta0,
                oja_decay=oja_decay,
                freeze_after=freeze_after,
                basis_seed=basis_seed,
                prefill_block_size=prefill_block_size,
                retention=retention,
                quant_bits=quant_bits,
                quant_budget=quant_budget,
                quant_bank=self._quant_bank,
                hh_budget=hh_budget,
                hh_select=hh_select,
                hh_neighbor=hh_neighbor,
                hh_retain=hh_retain,
                score_rank=score_rank,
                seed_hh_warmup=seed_hh_warmup,
                orth_fix_tol=orth_fix_tol,
                orth_abort_tol=orth_abort_tol,
                qr_every=qr_every,
                diag_every=diag_every,
                decode_attention=decode_attention,
                kernel_operand_dtype=kernel_operand_dtype,
                rope_basis=rope_basis,
                layer_idx=layer_idx,
            )
            for layer_idx in range(n_layers)
        ]
        super().__init__(layers=layers)
        self.decode_attention = decode_attention
        # Set to {} by a caller before ONE decode forward to have the kernel attention
        # function also run reconstruct-then-attend and record `layer_idx -> max|d|` here
        # (the on-pod single-layer check, `kvdlra.eval.kernel_check`); None = off.
        self.kernel_compare: dict[int, float] | None = None
        # Which backend actually attended (`kvdlra.kernel.select_backend`), set by the
        # attention function on the first kernel decode step and never changed after:
        # `backend="auto"` degrades to the torch reference on a live rank the Triton kernel
        # refuses (R-L4-22), and a ms/token that is the reference's must say so.
        self.kernel_backend: str | None = None

    @contextmanager
    def attach(self, model: PreTrainedModel) -> Iterator[None]:
        """Kernel mode: register the factored-attention function for this cache and switch
        the model's attention implementation to it for the scope (``kvdlra.kernel.attention``).

        Reconstruct mode: a no-op, kept so the harness can wrap any streaming cache
        uniformly (``with cache.attach(model): ...``, as :class:`ShadowKVCache` and the
        kvpress presses need); the surviving retention rule
        (``retention="lowrank_surprise"``) and the SurpriseSLASH exact tier both read their
        signal off the stored keys, so no attention hook is needed there; the Week-7
        ``retention="attn"`` mode that did is retired."""
        if self.decode_attention != "kernel":
            del model
            yield
            return
        # lazily: the glue imports transformers' attention integrations, kept off this
        # module's import path.
        from kvdlra.kernel.attention import attach_kernel

        with attach_kernel(self, model):
            yield

    def _bug_layers(self) -> list[BugStreamingLayer]:
        return [layer for layer in self.layers if isinstance(layer, BugStreamingLayer)]

    def _set_mode(self, mode: str) -> None:
        for layer in self._bug_layers():
            layer._mode = mode
            for row in layer._rows or ():
                row._mode = mode

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

    def drain_diag(self) -> list[dict[str, object]]:
        """Every layer's diagnostic rows since the last drain, and clear them.

        One row per layer per ``diag_every`` absorbs: ``{layer, absorbs, tokens_seen,
        orth_err_k, orth_err_v, eff_rank_k, eff_rank_v, rank_k, rank_v, fixed_k,
        fixed_v}`` -- the orthonormality window maxima, the effective-rank window minima,
        the live rank and whether the guard repaired that stream. Draining is what a
        runner does once per trial to write them out.

        The OPEN window is flushed first, so every drained layer that absorbed anything
        yields at least one row whatever ``diag_every`` is. Without it the periodic flush
        is the only emitter and a sample shorter than one window says nothing at all: at
        the pods' ``diag_every: 4096`` that is every 16K sample, and even at the default
        64 the tail of every sample is dropped (fix1 PR-29 A/D)."""
        rows: list[dict[str, object]] = []
        for layer in self._bug_layers():
            rows.extend(layer.drain_diag())
        return rows

    def stored_state_numel(self) -> int:
        """Total stored float-equivalents across layers (the constant-memory
        claim), plus the shared quantizer side info counted once."""
        total = sum(layer.stored_state_numel() for layer in self._bug_layers())
        if self._quant_bank is not None:
            total += self._quant_bank.side_info_numel()
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
