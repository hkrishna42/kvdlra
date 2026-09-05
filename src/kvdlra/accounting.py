"""Cross-method KV-cache memory accounting -- the single honest source of truth.

Every Week-10 frontier arm stores something different (BUG low-rank factors,
eviction survivors, ShadowKV low-rank keys + CPU-offloaded values). This module
counts each **honestly, in the same unit**, reusing the repo's existing
conventions so the *measured* caches
(:meth:`kvdlra.cache.BugStreamingCache.stored_state_numel`,
:meth:`kvdlra.cache.MorphKVCache.stored_state_numel`) and the *formula-only*
presses (SnapKV / ExpectedAttention via kvpress, ShadowKV) land on ONE axis. See
``docs/week10-plan.md`` (the "Memory accounting -- the core deliverable" section)
and ``docs/week10-kickoff.md``.

Conventions (identical to ``stored_state_numel`` / ``kv_memory_ratio`` /
``evict_quant_memory``):

* fp / verbatim elements count as **1 float-equivalent** each -- *regardless of
  device*, so a CPU-offloaded float counts the same as a GPU float (this is what
  neutralises ShadowKV's value-offload on the honest memory axis);
* quantized codes count at ``bits/32`` float-equivalents (bit-packable), ``ceil``
  over all codes, plus one fp32 norm per quantised column;
* the diagonal BUG core costs ``r`` per stream, not ``r**2``;
* retention positions (int32) and scores/norms (fp32) cost 1 each;
* the full reference cache is fp16 K+V = ``2 * T * n`` elements / layer, i.e.
  ``2 * T * n * 16`` bits / layer.

Two rendered numbers matter for the frontier: :meth:`Footprint.float_equiv` (the
repo's fp32-word unit -- the matched-memory / audit axis, byte-identical to
``stored_state_numel``) and :meth:`Footprint.ratio_fp16` (stored bits / full-fp16
bits -- the deployment headline, byte-identical to ``kv_memory_ratio``). A third,
truly-measured number, ``torch.cuda.max_memory_allocated`` (:func:`measure_peak_gpu`),
is reported *alongside* -- never instead of -- the analytic footprint.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

FP16_BITS = 16
FP32_BITS = 32
N_SINK = 4

# Retention modes that track per-coordinate positions / scores / surprise, and the
# ring-score high-water buffer -- mirrors ``coord_for_config`` /
# ``BugStreamingLayer.stored_state_numel``.
_TRACK_SCORE = ("attn", "blend")
_TRACK_SURPRISE = ("lowrank_surprise", "blend")


@dataclass(frozen=True)
class Footprint:
    """A method's per-layer stored footprint as a component breakdown, rendered
    into the repo's two units. ``verbatim_elems`` are fp elements (stored at the
    deploy dtype); ``quant_code_bits`` are packed quantiser code bits (already in
    bits); ``aux_words`` are fp32 bookkeeping words (positions/scores/norms/diag
    core) counted 1 each. For ShadowKV, ``gpu_verbatim_elems``/``cpu_verbatim_elems``
    split ``verbatim_elems`` by device (``gpu_verbatim_elems is None`` => all on
    GPU); the offloaded value cache is counted in the total, never hidden."""

    verbatim_elems: float
    quant_code_bits: float = 0.0
    aux_words: float = 0.0
    gpu_verbatim_elems: float | None = None
    cpu_verbatim_elems: float = 0.0
    fp32_verbatim_elems: float = 0.0

    def float_equiv(self) -> float:
        """Float-equivalents / layer (fp32-word unit), byte-identical to
        ``stored_state_numel``: verbatim at 1 each, codes ``ceil(bits/32)``, aux at
        1 each. Integer-valued for measured (BUG/MorphKV) configs."""
        codes = math.ceil(self.quant_code_bits / 32) if self.quant_code_bits else 0
        return self.verbatim_elems + codes + self.aux_words

    def bits(self, store_bits: int = FP16_BITS) -> float:
        """Stored bits / layer at ``store_bits`` for verbatim elements (codes keep
        their native bit width; aux words are fp32)."""
        return self.verbatim_elems * store_bits + self.quant_code_bits + self.aux_words * FP32_BITS

    def stored_bits(self) -> float:
        """Honest at-rest bits / layer: the ``fp32_verbatim_elems`` subset of
        ``verbatim_elems`` billed at its actual 32 bits, the remainder at 16, codes
        native, aux at 32. Equals ``bits(16)`` for any method with no fp32-at-rest
        state (ThinK/Palu/eviction/full/ShadowKV, ``fp32_verbatim_elems == 0``);
        for BUG the fp32 basis ``U`` and coordinates ``C`` push it above ``bits(16)``.
        This is what a naive ``ratio_fp16`` under-bills (Week-18 dual-billing)."""
        fp16_part = (self.verbatim_elems - self.fp32_verbatim_elems) * FP16_BITS
        return (
            fp16_part
            + self.fp32_verbatim_elems * FP32_BITS
            + self.quant_code_bits
            + self.aux_words * FP32_BITS
        )

    def ratio_fp16(self, t: int, n: int) -> float:
        """Stored bits / full-fp16-cache bits at context ``t`` (K+V, ``2*t*n*16``
        bits/layer) -- byte-identical to ``kv_memory_ratio`` for the BUG prefill
        model, and ``keep_frac`` for pure-fp16 eviction. This is the *fp16-equivalent*
        headline (BUG's fp32 state billed as if fp16); see :meth:`ratio_stored_bits`
        for the honest at-rest number."""
        return self.bits(FP16_BITS) / (2 * t * n * FP16_BITS)

    def ratio_stored_bits(self, t: int, n: int) -> float:
        """Honest at-rest stored-bits ratio (:meth:`stored_bits` / full-fp16-cache
        bits). Equals :meth:`ratio_fp16` for every method with no fp32-at-rest state;
        for BUG r64 it is ~0.15x/0.27x vs the 0.085x/0.149x fp16-equivalent number."""
        return self.stored_bits() / (2 * t * n * FP16_BITS)

    def tok_equiv(self, n: int) -> float:
        """Token-equivalents / layer = float_equiv / ``2n`` (the ``tau`` x-axis of
        the Week-7/9 figures: one full token is ``2n`` floats)."""
        return self.float_equiv() / (2 * n)

    def gpu_ratio_fp16(self, t: int, n: int) -> float:
        """GPU-resident stored bits / full-fp16-cache bits. Equals
        :meth:`ratio_fp16` for every all-GPU method; for ShadowKV it is the
        (flattering) GPU-only number, reported *beside* the honest total."""
        gpu = self.verbatim_elems if self.gpu_verbatim_elems is None else self.gpu_verbatim_elems
        gpu_bits = gpu * FP16_BITS + self.quant_code_bits + self.aux_words * FP32_BITS
        return gpu_bits / (2 * t * n * FP16_BITS)

    def cpu_ratio_fp16(self, t: int, n: int) -> float:
        """CPU-offloaded stored bits / full-fp16-cache bits (0 for every method but
        ShadowKV, whose offloaded value cache is ``t*n`` -> ``0.5`` at rank << t)."""
        return self.cpu_verbatim_elems * FP16_BITS / (2 * t * n * FP16_BITS)


# --------------------------------------------------------------------------- BUG


def bug_footprint(
    n: int,
    rank: int,
    coord_count: int,
    recent_len: int,
    *,
    n_sink: int = N_SINK,
    retention: str = "fifo",
    hh_count: int = 0,
    hh_select: str = "attn",
    merge: bool = False,
    u_present: bool = True,
    quant_count: int = 0,
    quant_bits: int | None = None,
    w_key: bool = False,
) -> Footprint:
    """Per-layer footprint of one ``BugStreamingCache`` layer state, mirroring
    :meth:`BugStreamingLayer.stored_state_numel` to the float.

    ``coord_count`` fp32 coordinate columns (K+V, ``2*rank`` each), ``recent_len``
    verbatim recent-ring tokens (``2n``), ``n_sink`` verbatim sinks (``2n``),
    ``hh_count`` verbatim SLASH heavy-hitters (``2n`` + 2 aux for ``hh_select=
    "attn"``, ``2n`` + 1 aux for ``"surprise"``), the basis ``U``
    (``2*n*rank``) and diagonal core (``2*rank``) when ``u_present``. Adaptive
    retention adds 1 position and/or 1 score per column and a ``recent_len``
    ring-score buffer (``retention in {attn, blend}``). ``quant_count`` columns
    are stored as ``2*rank*quant_bits`` code bits + ``2`` fp32 norms each.

    Its ``float_equiv()`` equals the live cache's ``stored_state_numel()``; the
    anti-drift test (``tests/test_accounting.py``) pins this so the formula (used
    for SnapKV/ShadowKV) and the measured path cannot diverge."""
    track_pos = retention != "fifo" or merge
    track_score = retention in _TRACK_SCORE
    track_surprise = retention in _TRACK_SURPRISE
    n_cols = coord_count + quant_count  # all low-rank columns carry bookkeeping

    verbatim = 2 * n * n_sink + 2 * n * recent_len + 2 * rank * coord_count + 2 * n * hh_count
    # fp32-at-rest subset of ``verbatim`` (bug_cache.py:575-577): the coordinate
    # columns C and the basis U. Sinks/recent/hh are verbatim KV in the model dtype
    # (fp16/bf16), so they are NOT fp32-at-rest. Reported via ratio_stored_bits.
    fp32_verbatim = 2.0 * rank * coord_count
    aux = 0.0
    if u_present:
        verbatim += 2 * n * rank  # basis U (K + V)
        fp32_verbatim += 2 * n * rank  # ...also fp32 at rest
        aux += 2 * rank  # diagonal core (K + V)
    aux += n_cols * (int(track_pos) + int(track_score) + int(merge) + int(track_surprise))
    if track_score:
        aux += recent_len  # ring-score buffer (high-water = recent_len)
    # hh tier: always the int64 positions; the fp32 selection score only when the
    # exact tier is attention-selected (Week-11 SurpriseSLASH recomputes surprise
    # from the basis each absorb, so hh_select="surprise" stores no score).
    aux += hh_count * (2 if hh_select == "attn" else 1)  # hh_pos (+ hh_score for attn)

    code_bits = 0.0
    if quant_count and quant_bits is not None:
        code_bits = 2 * rank * quant_count * quant_bits  # K + V codes
        aux += 2 * quant_count  # one fp32 norm per K,V quantised column
    if w_key:
        aux += n  # Week-12 Q-BUG: frozen per-feature key-whitening diagonal (fp32)
    return Footprint(
        verbatim_elems=verbatim,
        quant_code_bits=code_bits,
        aux_words=aux,
        fp32_verbatim_elems=fp32_verbatim,
    )


def bug_footprint_saturated(
    n: int,
    rank: int,
    coord_budget: int,
    recent_window: int,
    absorb_block: int,
    *,
    n_sink: int = N_SINK,
    retention: str = "fifo",
    hh_budget: int = 0,
) -> Footprint:
    """High-water footprint of a saturated BUG config -- the budget
    ``bug_budget_floats`` / ``coord_for_config`` allocate against: ``coord_budget``
    coordinate columns and the recent ring at its high-water
    (``recent_window + absorb_block - 1``)."""
    ring = recent_window + absorb_block - 1
    return bug_footprint(
        n,
        rank,
        coord_budget,
        ring,
        n_sink=n_sink,
        retention=retention,
        hh_count=hh_budget,
    )


def bug_prefill_footprint(
    t: int, n: int, rank: int, *, n_sink: int = N_SINK, quant_bits: int | None = None
) -> Footprint:
    """The Week-4 ``BUGPress`` *prefill* model (distinct from the streaming
    :func:`bug_footprint`): every non-sink token kept as a coordinate, basis ``U``,
    exact sinks -- **no** recent ring or diagonal core. Its ``ratio_fp16`` is
    byte-identical to ``scripts/w4_hybrid_sweep.kv_memory_ratio`` for the fp case.
    Kept for the Phase-7 delegator consolidation / continuity, not the frontier
    (the Week-10 frontier stores the streaming cache, so it uses
    :func:`bug_footprint`)."""
    t_pay = t - n_sink
    verbatim = 2 * n * n_sink + 2 * n * rank  # sinks + basis U (K+V)
    if quant_bits is None:
        # fp32-at-rest: basis U + the fp coordinate columns (sinks are model-dtype).
        return Footprint(
            verbatim_elems=verbatim + 2 * rank * t_pay,
            fp32_verbatim_elems=2.0 * n * rank + 2.0 * rank * t_pay,
        )
    # kv_memory_ratio counts the per-token norm at fp16; fold it into verbatim so
    # bits(16) reproduces it exactly (norms are fp16 here, not the fp32 aux word).
    # Only U is fp32 at rest here; coordinates live as quant codes.
    return Footprint(
        verbatim_elems=verbatim + t_pay,
        quant_code_bits=2 * rank * t_pay * quant_bits,
        fp32_verbatim_elems=2.0 * n * rank,
    )


# ------------------------------------------------------------------ eviction


def evict_footprint(
    t: int, n: int, keep_frac: float, *, quant_bits: int | None = None
) -> Footprint:
    """Per-layer footprint of prefill eviction (SnapKV / ExpectedAttention): the
    kept fraction of tokens stored verbatim (K+V, ``2n`` each), or -- if
    ``quant_bits`` -- as ``2*n*kept*quant_bits`` code bits + one fp32 norm per K,V
    column. Pure-fp16 ``ratio_fp16`` == ``keep_frac`` exactly.

    Note: this is the corrected convention (no spurious ``+FP16`` norm term for the
    pure-fp16 case that ``scripts/w4_fair.evict_quant_memory`` carries); the
    published scripts keep their number until the Phase-7 delegator consolidation
    (see ``docs/week10-plan.md``)."""
    kept = keep_frac * t
    if quant_bits is None:
        return Footprint(verbatim_elems=2 * n * kept)
    return Footprint(
        verbatim_elems=0.0, quant_code_bits=2 * n * kept * quant_bits, aux_words=2 * kept
    )


# ------------------------------------------------------------------- MorphKV


def morph_footprint(n: int, h_kv: int, kept_len: int, recent_window: int) -> Footprint:
    """Per-layer footprint of a MorphKV layer, mirroring
    :meth:`MorphKVLayer.stored_state_numel`: kept K/V (``2n`` each over ``kept_len``
    tokens) + the score buffer ``(H_kv, R, kept_len)``. ``kept_len`` should include
    any ``evict_interval-1`` overshoot when auditing the measured high-water."""
    return Footprint(verbatim_elems=2 * n * kept_len, aux_words=h_kv * recent_window * kept_len)


# ------------------------------------------------------------------ ThinK


def think_footprint(
    t: int, n: int, head_dim: int, h_kv: int, key_channel_ratio: float, window_size: int = 32
) -> Footprint:
    """Per-layer footprint of ThinK (arXiv:2407.21018): prune a ``key_channel_ratio``
    fraction of the KEY channels (dimensions), values untouched. So only K is
    compressed to ``(1-ratio)*head_dim`` channels/token; ``ratio_fp16 -> 1 - ratio/2``
    (K is half the cache). Plus the kept-channel index set (per KV head, one-time).

    Note: the kvpress ``ThinKPress`` *zeros* pruned channels (same tensor shape, no
    measured gain), so this analytic footprint -- not the DynamicCache numel -- is
    the honest deployable memory."""
    kept_ch = max(1, round((1.0 - key_channel_ratio) * head_dim))
    verbatim = t * kept_ch * h_kv + t * n  # pruned K + full V
    aux = h_kv * kept_ch  # kept-channel indices, per head (one-time)
    return Footprint(verbatim_elems=verbatim, aux_words=aux)


# ------------------------------------------------------------------ Palu


def palu_footprint(
    t: int,
    n: int,
    head_dim: int,
    h_kv: int,
    rank_ratio: float,
    *,
    group: int = 1,
    n_sink: int = N_SINK,
) -> Footprint:
    """Per-layer footprint of Palu (arXiv:2407.21118): low-rank projection of K AND
    V into a rank-``r`` latent per head-group, ``r = rank_ratio * head_dim * group``.
    Stores the ``n_sink`` leading token columns **verbatim** (the sink exemption
    :class:`kvdlra.press.PaluPress` applies since the Week-15 audit fix -- only
    columns ``n_sink:`` are low-ranked, K+V), the per-token latent ``H``
    (``(t-n_sink)*r``) for K and V, plus the reconstruction basis ``B``
    (``r*head_dim*group``) per group -- all counted. ``group`` = KV heads sharing
    one projection (Palu's grouped low-rank; group=1 = per-head).

    ratio_fp16 ~ (r / (head_dim*group)) = rank_ratio at long t (the basis and the
    tiny exact-sink block amortize), i.e. it compresses BOTH K and V to the rank
    fraction -- the low-rank analogue of BUG with a *static* (weight-SVD) subspace
    rather than the streaming tracker."""
    n_groups = max(1, h_kv // group)
    r = max(1, round(rank_ratio * head_dim * group))
    t_sink = min(n_sink, t)
    t_pay = t - t_sink  # only columns n_sink: carry the low-rank latent
    sinks = 2 * n * t_sink  # K + V exact sink columns (full feature width)
    latent = 2 * t_pay * r * n_groups  # K + V per-token latents H (t_pay x r) per group
    basis = 2 * r * head_dim * group * n_groups  # K + V reconstruction bases B
    return Footprint(verbatim_elems=sinks + latent + basis)


# ------------------------------------------------------------------ ShadowKV


def shadow_footprint(
    t: int,
    n: int,
    h_kv: int,
    head_dim: int,
    *,
    rank_s: int = 160,
    chunk: int = 8,
    outlier_frac: float = 0.0,
    sparse_budget: int | None = None,
    n_sink: int = 0,
    recent_len: int = 0,
) -> Footprint:
    """Per-layer footprint of ShadowKV (arXiv:2410.21465), the fairness crux --
    reconciled with the in-repo port :class:`kvdlra.cache.ShadowKVCache` so its
    ``float_equiv()`` equals the live cache's ``stored_state_numel()`` (the
    ``tests/test_shadow_cache.py`` anti-drift pin).

    The port keeps ``n_sink`` sinks + ``recent_len`` recent keys **verbatim** on the
    accelerator and low-ranks only the *middle* (``mid_len = t - n_sink -
    recent_len``, trimmed to a whole number of ``chunk``-token chunks).

    **GPU-resident:** the verbatim sink/recent keys (``n*(n_sink + recent_len)``), a
    rank-``rank_s`` per-KV-head factorisation of the middle keys (coefficients
    ``mid_len*rank_eff`` + basis ``rank_eff*head_dim`` per head, with ``rank_eff =
    min(rank_s, head_dim, mid_len)``), one mean-pooled landmark per chunk
    (``n_chunks*n``), and an optional exact ``outlier_frac`` of the middle.
    **CPU-offloaded:** the FULL value cache (``t*n``), fetched sparsely each step.
    Under this repo's device-agnostic float rule the offloaded V counts at 1
    float/elem in the total (``cpu_ratio_fp16 -> 0.5``), so the offload buys ZERO on
    the honest memory axis -- its only fair savings are low-rank K and sparse decode.
    Reporting V-offload as "free" is forbidden.

    With the defaults ``n_sink = recent_len = 0`` the whole context is the middle,
    recovering the provisional whole-context formula (the value counted, never
    hidden). ``sparse_budget`` (top-k * chunk tokens fetched/step) does not change
    the stored CPU footprint (always ``t*n``); it drives the separate per-step
    CPU<->GPU bandwidth line, not this footprint."""
    mid_len = max(0, t - n_sink - recent_len)
    mid_len = (mid_len // chunk) * chunk  # whole chunks only (port deviation #2)
    n_chunks = mid_len // chunk
    rank_eff = min(rank_s, head_dim, mid_len) if mid_len > 0 else 0
    lowrank_k = h_kv * (mid_len * rank_eff + rank_eff * head_dim)
    landmarks = float(n_chunks * n)
    sink_recent_k = float(n * (n_sink + recent_len))
    outliers = outlier_frac * mid_len * n
    gpu = lowrank_k + landmarks + sink_recent_k + outliers
    cpu = float(t * n)  # full value cache, offloaded -- counted, never hidden
    return Footprint(verbatim_elems=gpu + cpu, gpu_verbatim_elems=gpu, cpu_verbatim_elems=cpu)


def shadow_bandwidth_per_token(n: int, sparse_budget: int, chunk: int) -> float:
    """Elements fetched CPU->GPU per decode step for ShadowKV's sparse-V read
    (``sparse_budget`` selected tokens x ``n`` features). The PCIe traffic the
    offload pays back every step -- reported as an explicit line so the offload's
    cost is never hidden behind its GPU-memory saving."""
    return float(sparse_budget * n)


# ---------------------------------------------------------------- full cache


def full_cache_footprint(t: int, n: int) -> Footprint:
    """The uncompressed fp16 reference: K+V verbatim, ``2*t*n`` elements/layer
    (``ratio_fp16 == 1.0``)."""
    return Footprint(verbatim_elems=2 * t * n)


# ------------------------------------------------------------- quantized KV (KIVI)


def quant_footprint(
    t: int,
    n: int,
    *,
    nbits: int,
    group: int = 64,
    residual_length: int = 128,
    scale_words: float = 2,
) -> Footprint:
    """KIVI-style 2/4-bit KV baseline (transformers ``QuantizedCache`` / quanto backend,
    arm ``quant-{nbits}bit``). The ``residual_length`` most-recent tokens stay verbatim
    in the model dtype; older tokens are quantized to ``nbits`` with a per-``group``
    scale+shift. Billed honestly to match what quanto actually stores (verified via the
    Week-18 probe: ``_data`` uint8 codes at ``nbits``, ``_scale``+``_shift`` fp32, one
    pair per group, no zeropoint): code bits at ``nbits``, ``scale_words`` fp32 aux words
    per group, residual as fp16 verbatim. Asymptote ``(nbits + scale_words*32/group)/16``
    -> 2-bit/g64 = 0.1875x, 4-bit/g64 = 0.3125x (the KIVI/KVQuant 0.125-0.19x band).

    ``fp32_verbatim_elems`` is 0 (the residual is model-dtype), so its honest
    ``ratio_stored_bits`` equals ``ratio_fp16`` -- billed on the same footing as ThinK/
    Palu, unlike BUG whose fp32 state splits the two."""
    resid = min(residual_length, t)
    payload = max(0, t - resid)
    verbatim = 2 * resid * n  # K+V fp16 residual window
    code_bits = 2 * payload * n * nbits  # K+V quantized codes at nbits
    n_groups = 2 * math.ceil(payload * n / group) if payload else 0  # K+V groups
    aux = float(n_groups * scale_words)  # fp32 scale + shift per group
    return Footprint(verbatim_elems=verbatim, quant_code_bits=code_bits, aux_words=aux)


# ------------------------------------------------------------- peak-GPU probe


@contextmanager
def measure_peak_gpu(device: str) -> Iterator[Callable[[], int | None]]:
    """Yield a getter for ``torch.cuda.max_memory_allocated`` over the block, or a
    getter returning ``None`` on CPU (honestly unmeasured, not faked). The peak
    includes activations/workspace, so it is reported *beside* the analytic
    footprint, never as the deployable-cache claim."""
    import torch

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        yield lambda: int(torch.cuda.max_memory_allocated())
    else:
        yield lambda: None


# ----------------------------------------------------------------- report


@dataclass(frozen=True)
class MemoryReport:
    """One arm's three memory numbers + the honest ShadowKV split."""

    arm: str
    t: int
    n: int
    float_equiv_per_layer: float
    tok_equiv_per_layer: float
    ratio_fp16: float
    gpu_ratio_fp16: float
    cpu_ratio_fp16: float
    bytes_native: int
    peak_gpu_bytes: int | None
    detail: dict[str, Any] = field(default_factory=dict)


def report(
    arm: str,
    footprint: Footprint,
    t: int,
    n: int,
    *,
    store_bits: int = FP16_BITS,
    peak_gpu_bytes: int | None = None,
    detail: dict[str, Any] | None = None,
) -> MemoryReport:
    """Render a :class:`Footprint` into a :class:`MemoryReport` (the three required
    numbers + the GPU/CPU split); ``bytes_native`` is the deployable stored bytes
    at ``store_bits``, ``peak_gpu_bytes`` the measured operational peak (or None)."""
    return MemoryReport(
        arm=arm,
        t=t,
        n=n,
        float_equiv_per_layer=footprint.float_equiv(),
        tok_equiv_per_layer=footprint.tok_equiv(n),
        ratio_fp16=footprint.ratio_fp16(t, n),
        gpu_ratio_fp16=footprint.gpu_ratio_fp16(t, n),
        cpu_ratio_fp16=footprint.cpu_ratio_fp16(t, n),
        bytes_native=int(footprint.bits(store_bits) / 8),
        peak_gpu_bytes=peak_gpu_bytes,
        detail=detail or {},
    )


# --------------------------------------------------------- matched-memory audit


def audit_matched(
    mem_max_per_layer: float, budget_per_layer: float, arm: str, *, tol: float = 0.0
) -> dict[str, Any]:
    """Audit one arm's high-water per-layer footprint against a matched budget."""
    within = mem_max_per_layer <= budget_per_layer + tol
    over = mem_max_per_layer / budget_per_layer if budget_per_layer else math.inf
    return {"arm": arm, "mem_max": mem_max_per_layer, "over_budget": over, "within": within}


def assert_all_within(
    mem_by_arm: dict[str, float], budget_per_layer: float, *, tol: float = 0.0
) -> None:
    """Raise if any arm's per-layer footprint exceeds the matched budget -- the
    gate called before any "matched-memory" frontier or table is emitted, so a
    matched label can never be printed for an over-budget arm (formalises the
    ad-hoc ``mem_max > budget`` checks in w5/w7/w9)."""
    over = {a: m for a, m in mem_by_arm.items() if m > budget_per_layer + tol}
    if over:
        raise ValueError(
            f"matched-memory audit FAILED: {over} exceed budget {budget_per_layer}/layer"
        )
