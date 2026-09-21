"""Cross-method KV-cache memory accounting -- one unit for every method.

Every Week-10 frontier arm stores something different (BUG low-rank factors,
eviction survivors, ShadowKV low-rank keys + CPU-offloaded values). This module
counts each **in the same unit**, reusing the repo's existing
conventions so the *measured* caches
(:meth:`kvdlra.cache.BugStreamingCache.stored_state_numel`) and the
*formula-only* presses (SnapKV / ExpectedAttention via kvpress, ShadowKV) land on
ONE axis.

Conventions (identical to ``stored_state_numel`` / ``kv_memory_ratio`` /
``evict_quant_memory``):

* fp / verbatim elements count as **1 float-equivalent** each -- *regardless of
  device*, so a CPU-offloaded float counts the same as a GPU float (this is what
  neutralises ShadowKV's value-offload on the stored-state axis);
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
_TRACK_SURPRISE = ("lowrank_surprise",)


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
        1 each. Integer-valued for measured (BUG) configs."""
        codes = math.ceil(self.quant_code_bits / 32) if self.quant_code_bits else 0
        return self.verbatim_elems + codes + self.aux_words

    def bits(self, store_bits: int = FP16_BITS) -> float:
        """Stored bits / layer at ``store_bits`` for verbatim elements (codes keep
        their native bit width; aux words are fp32)."""
        return self.verbatim_elems * store_bits + self.quant_code_bits + self.aux_words * FP32_BITS

    def stored_bits(self) -> float:
        """At-rest bits / layer: the ``fp32_verbatim_elems`` subset of
        ``verbatim_elems`` billed at its actual 32 bits, the remainder at 16, codes
        native, aux at 32. Equals ``bits(16)`` for any method with no fp32-at-rest
        state (ThinK/low-rank/eviction/full/ShadowKV, ``fp32_verbatim_elems == 0``);
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
        for the at-rest number."""
        return self.bits(FP16_BITS) / (2 * t * n * FP16_BITS)

    def ratio_stored_bits(self, t: int, n: int) -> float:
        """At-rest stored-bits ratio (:meth:`stored_bits` / full-fp16-cache
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
        (flattering) GPU-only number, reported *beside* the device-agnostic total."""
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
    rank: float,
    coord_count: int,
    recent_len: int,
    *,
    n_sink: int = N_SINK,
    retention: str = "fifo",
    hh_count: int = 0,
    u_present: bool = True,
    quant_count: int = 0,
    quant_bits: int | None = None,
    gist_bits: int = FP32_BITS,
) -> Footprint:
    """Per-layer footprint of one ``BugStreamingCache`` layer state, mirroring
    :meth:`BugStreamingLayer.stored_state_numel` to the float.

    ``rank`` is the **live tracked rank** -- the columns the layer's bases actually hold
    (``u_k.shape[1]``) -- not the configured cap the arm was built with: the Week-17
    singular-value floor (``min_sv_frac``) drops near-null tail directions, so a
    floor-on layer tracks fewer, and the cap would bill a basis that is not stored
    (audit finding 0.2). Every rank term below is ``2*rank*x``, i.e. symmetric in the K
    and V streams, so where the floor collapsed the two to different widths their MEAN
    is the value that reproduces ``stored_state_numel`` -- a half-integer ``rank`` is
    well defined here and leaves ``float_equiv()`` integral.

    ``coord_count`` fp32 coordinate columns (K+V, ``2*rank`` each), ``recent_len``
    verbatim recent-ring tokens (``2n``), ``n_sink`` verbatim sinks (``2n``),
    ``hh_count`` verbatim SLASH heavy-hitters (``2n`` + 1 aux for the position),
    the basis ``U`` (``2*n*rank``) and diagonal core (``2*rank``) when
    ``u_present``. Adaptive retention adds 1 position and 1 surprise snapshot per
    column. ``quant_count`` columns are stored as ``2*rank*quant_bits`` code bits
    + ``2`` fp32 norms each.

    ``gist_bits`` is the width the gist is STORED at (``BugStreamingLayer.gist_dtype``,
    L5.1): 32 by default; at 16 the basis and the coordinates leave the fp32-at-rest
    subset, so ``stored_bits()`` bills them at 16 like every other verbatim element.
    Element counts do not move with it -- ``float_equiv()`` and ``stored_state_numel()``
    count elements, not bytes.

    Its ``float_equiv()`` equals the live cache's ``stored_state_numel()``; the
    anti-drift test (``tests/test_accounting.py``) pins this so the formula (used
    for SnapKV/ShadowKV) and the measured path cannot diverge."""
    if gist_bits not in (FP16_BITS, FP32_BITS):
        raise ValueError(f"gist_bits must be {FP16_BITS} or {FP32_BITS}, got {gist_bits}")
    gist_fp32 = gist_bits == FP32_BITS
    track_pos = retention != "fifo"
    track_surprise = retention in _TRACK_SURPRISE
    n_cols = coord_count + quant_count  # all low-rank columns carry bookkeeping

    verbatim = 2 * n * n_sink + 2 * n * recent_len + 2 * rank * coord_count + 2 * n * hh_count
    # fp32-at-rest subset of ``verbatim`` (``BugStreamingLayer._reset_state``, where U and
    # C are declared): the coordinate columns C and the basis U. Sinks/recent/hh are
    # verbatim KV in the model dtype (fp16/bf16), so they are NOT fp32-at-rest. Reported
    # via ratio_stored_bits. A ``gist_bits=16`` arm stores C and U in bf16 too, so the
    # subset is empty there.
    fp32_verbatim = 2.0 * rank * coord_count if gist_fp32 else 0.0
    aux = 0.0
    if u_present:
        verbatim += 2 * n * rank  # basis U (K + V)
        if gist_fp32:
            fp32_verbatim += 2 * n * rank  # ...also fp32 at rest
        # The diagonal core (K + V). Left in ``aux_words`` at 32 bits even under a bf16
        # gist -- conservative, and 2r words is noise beside the 2nr the basis costs.
        aux += 2 * rank
    aux += n_cols * (int(track_pos) + int(track_surprise))
    # hh tier: the int64 positions. Week-11 SurpriseSLASH recomputes the selection
    # score from the basis each absorb, so the exact tier stores no score.
    aux += hh_count  # hh_pos

    code_bits = 0.0
    if quant_count and quant_bits is not None:
        code_bits = 2 * rank * quant_count * quant_bits  # K + V codes
        aux += 2 * quant_count  # one fp32 norm per K,V quantised column
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
    byte-identical to ``scripts/w4_hybrid_sweep.kv_memory_ratio`` @ ``paper-v1-archive``
    for the fp case.
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
    pure-fp16 case that ``scripts/w4_fair.evict_quant_memory`` @ ``paper-v1-archive``
    carries); the published scripts keep their number until the Phase-7 delegator
    consolidation."""
    kept = keep_frac * t
    if quant_bits is None:
        return Footprint(verbatim_elems=2 * n * kept)
    return Footprint(
        verbatim_elems=0.0, quant_code_bits=2 * n * kept * quant_bits, aux_words=2 * kept
    )


# ------------------------------------------------------------------ ThinK


def think_footprint(
    t: float, n: int, head_dim: int, h_kv: int, key_channel_ratio: float, window_size: int = 32
) -> Footprint:
    """Per-layer footprint of ThinK (arXiv:2407.21018): prune a ``key_channel_ratio``
    fraction of the KEY channels (dimensions), values untouched. So only K is
    compressed to ``(1-ratio)*head_dim`` channels/token; ``ratio_fp16 -> 1 - ratio/2``
    (K is half the cache). Plus the kept-channel index set (per KV head, one-time).

    Note: the kvpress ``ThinKPress`` *zeros* pruned channels (same tensor shape, no
    measured gain), so this analytic footprint -- not the DynamicCache numel -- is
    the deployable memory."""
    kept_ch = max(1, round((1.0 - key_channel_ratio) * head_dim))
    verbatim = t * kept_ch * h_kv + t * n  # pruned K + full V
    aux = h_kv * kept_ch  # kept-channel indices, per head (one-time)
    return Footprint(verbatim_elems=verbatim, aux_words=aux)


def think_evict_footprint(
    t: int, n: int, head_dim: int, h_kv: int, key_channel_ratio: float, keep_frac: float
) -> Footprint:
    """ThinK composed with an eviction press, the pairing its paper evaluates (SnapKV/H2O,
    then ThinK): the ``keep_frac`` kept tokens billed as `think_footprint` bills a token.
    ``key_channel_ratio=0`` is `evict_footprint`; ``keep_frac=1`` is `think_footprint`."""
    if key_channel_ratio == 0:  # nothing pruned -> no channel index set to store
        return evict_footprint(t, n, keep_frac)
    return think_footprint(keep_frac * t, n, head_dim, h_kv, key_channel_ratio)


# ------------------------------------------------------------------ SVD oracle


def lowrank_footprint(
    t: int,
    n: int,
    head_dim: int,
    h_kv: int,
    rank_ratio: float,
    *,
    group: int = 1,
    n_sink: int = N_SINK,
) -> Footprint:
    """Per-layer footprint of a static low-rank projection of K AND V into a
    rank-``r`` latent per head-group, ``r = rank_ratio * head_dim * group`` -- the
    memory model that paper (arXiv:2407.21118) would have; used to bill the SVD
    oracle (:class:`kvdlra.baselines.svd_oracle.SVDOraclePress`), which reconstructs
    same-shape K/V. Stores the ``n_sink`` leading token columns **verbatim** (the
    sink exemption applies since the Week-15 audit fix -- only columns ``n_sink:``
    are low-ranked, K+V), the per-token latent ``H`` (``(t-n_sink)*r``) for K and V,
    plus the reconstruction basis ``B`` (``r*head_dim*group``) per group -- all
    counted. ``group`` = KV heads sharing one projection (that paper's grouped
    low-rank; group=1 = per-head).

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
    the stored-state axis -- its only fair savings are low-rank K and sparse decode.
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
    scale+shift. Billed to match what quanto actually stores (verified via the
    Week-18 probe: ``_data`` uint8 codes at ``nbits``, ``_scale``+``_shift`` fp32, one
    pair per group, no zeropoint): code bits at ``nbits``, ``scale_words`` fp32 aux words
    per group, residual as fp16 verbatim. Asymptote ``(nbits + scale_words*32/group)/16``
    -> 2-bit/g64 = 0.1875x, 4-bit/g64 = 0.3125x (the KIVI/KVQuant 0.125-0.19x band).

    ``fp32_verbatim_elems`` is 0 (the residual is model-dtype), so its
    ``ratio_stored_bits`` equals ``ratio_fp16`` -- billed on the same footing as
    ThinK and the SVD oracle, unlike BUG whose fp32 state splits the two."""
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
    getter returning ``None`` on CPU (unmeasured, not faked). The peak
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
    """One arm's three memory numbers + the ShadowKV device split."""

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
