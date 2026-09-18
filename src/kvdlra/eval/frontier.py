"""The long-context perplexity axis, the arm builder, and the memory accounting.

One protocol across every method: prefill ``T`` tokens with the arm's compression
active, then score teacher-forced perplexity on a frozen ``W``-token continuation
window attending to the *compressed* cache (the "compress-then-score" deviation
documented in ``window_nll`` / Week-4). Memory is counted per ``kvdlra.accounting``
(float-equivalents per layer, the ``stored_state_numel`` unit) and cross-checked
against the live cache.

This module also owns the two pieces every other eval axis shares: the prefill
helpers (``_prefill_chunked`` for a streaming cache, ``_prefill_plain`` for a
QuantizedCache) and ``_footprint``, which maps an arm plus its post-prefill cache to
a ``Footprint``.

The ``[pplw]`` and ``ppl=`` lines this module prints are the pod's stdout contract:
``kvdlra.eval.records`` parses them back out of a harvested log, so their format is
frozen (``tests/test_w15_pplw.py``).
"""

from __future__ import annotations

import gc
from contextlib import nullcontext
from typing import Any, cast

import torch
from torch.nn.functional import cross_entropy
from transformers.cache_utils import Cache, DynamicCache

from kvdlra import accounting as acc
from kvdlra.cache import BugStreamingCache, ShadowKVCache
from kvdlra.eval.config import ArmCfg, arm_kwargs
from kvdlra.eval.records import emit_diag
from kvdlra.quant.kivi_cache import aux_words, flush, make_quant_cache

N_SINK = 4
# One scored window: (context ids, continuation ids), both 1-D.
Sample = tuple[torch.Tensor, torch.Tensor]


# --------------------------------------------------------------------- scoring


@torch.no_grad()
def _score_window(
    model: Any, cache: Cache, ctx_len: int, win_ids: torch.Tensor
) -> tuple[float, int]:
    """Summed NLL (nats) + scored-token count for the frozen continuation window,
    at TRUE positions -- the Week-3 ``window_nll`` scorer, byte for byte."""
    win = win_ids.unsqueeze(0)
    win_len = int(win_ids.shape[0])
    pos = torch.arange(ctx_len, ctx_len + win_len, device=win_ids.device).unsqueeze(0)
    out = model(win, past_key_values=cache, use_cache=True, position_ids=pos)
    return _nll_sum(out.logits[0][:-1], win_ids[1:]), int(win_ids[1:].shape[0])


def _nll_sum(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Summed NLL in fp32. Week-19 exit-gate fix: on a bf16 model ``cross_entropy``
    log-softmaxes and sums in bf16, quantizing a 511-token window to the bf16 ulp
    (8 nats at ~1100, ~1.6%); every Week-15..19 window sum was a multiple of that ulp.
    The archived numbers stand as recorded (their resolution is disclosed in the
    paper); new runs accumulate exactly."""
    return float(cross_entropy(logits.float(), targets, reduction="sum"))


def _prefill_chunked(model: Any, cache: Cache, ctx: torch.Tensor, chunk: int) -> None:
    """OOM-safe chunked prefill for a streaming cache (Phase 3): the first chunk
    single-shot-prefills (cumulative_length == 0), each later chunk ingests; absorb
    is consolidated after every chunk. ``logits_to_keep=1`` drops the P x vocab
    logits so no forward's activations scale with the full ``T``."""
    t = int(ctx.shape[1])
    ingesting = getattr(cache, "ingesting")  # noqa: B009 (BugStreamingCache/ShadowKVCache)
    consolidate = getattr(cache, "consolidate")  # noqa: B009
    with ingesting():
        for start in range(0, t, chunk):
            stop = min(t, start + chunk)
            pos = torch.arange(start, stop, device=ctx.device).unsqueeze(0)
            model(
                ctx[:, start:stop],
                past_key_values=cache,
                use_cache=True,
                position_ids=pos,
                logits_to_keep=1,
            )
            consolidate()


@torch.no_grad()
def score_streaming(
    model: Any,
    cache: BugStreamingCache | ShadowKVCache,
    ctx_ids: torch.Tensor,
    win_ids: torch.Tensor,
    chunk: int = 0,
    *,
    arm: str,
    idx: int | None = None,
) -> tuple[float, int]:
    """Prefill into a streaming cache (compresses) then frozen-window score over the
    compressed cache (non-mutating). ``chunk > 0`` (and < ctx) uses OOM-safe chunked
    ingest; otherwise single-shot. ``arm`` / ``idx`` only label the diagnostics."""
    ctx = ctx_ids.unsqueeze(0)
    ctx_len = int(ctx_ids.shape[0])
    try:
        with cache.attach(model):
            if 0 < chunk < ctx_len:
                _prefill_chunked(model, cache, ctx, chunk)
            else:
                model(ctx, past_key_values=cache, use_cache=True, logits_to_keep=1)
        with cache.frozen_scoring():
            scored = _score_window(model, cache, ctx_len, win_ids)
    finally:
        # The tripwire's rows leave the library here -- drained after the sample and
        # before the cache is dropped, since nothing else ever reads them again. In a
        # `finally` because the window worth reading most is the last one before an
        # `OrthonormalityError`: the cache flushes it before raising (L1.1), and an emit
        # on the success path alone would let it die with the cache, leaving the trial
        # recorded as an error with no diagnostics behind it.
        if isinstance(cache, BugStreamingCache):
            emit_diag(
                cache.drain_diag(),
                model=str(model.name_or_path),
                source="ppl",
                arm=arm,
                ctx=ctx_len,
                idx=idx,
            )
    return scored


@torch.no_grad()
def score_press(
    model: Any, press: Any, ctx_ids: torch.Tensor, win_ids: torch.Tensor, chunk: int = 0
) -> tuple[float, int, DynamicCache]:
    """Single-shot (or ChunkPress-chunked) prefill through a kvpress prefill press
    (or None=full), then score. Returns the (compressed) DynamicCache so its kept
    memory is measured."""
    cache = DynamicCache()
    ctx = ctx_ids.unsqueeze(0)
    ctx_len = int(ctx_ids.shape[0])
    active = press
    if press is not None and 0 < chunk < ctx_len:
        from kvpress import ChunkPress

        active = ChunkPress(press=press, chunk_length=chunk)
    with active(model) if active is not None else nullcontext():
        model(ctx, past_key_values=cache, use_cache=True, logits_to_keep=1)
    nll, ntok = _score_window(model, cache, ctx_len, win_ids)
    return nll, ntok, cache


@torch.no_grad()
def _prefill_plain(model: Any, cache: Any, ctx: torch.Tensor, chunk: int) -> None:
    """Chunked (or single-shot when ``chunk`` is 0 / >= T) prefill into a plain HF cache
    that updates incrementally (the QuantizedCache baseline), then flush the fp16
    residual so decode starts fully quantized exactly as after a single-shot prefill.
    Week-19: the single-shot 16K/32K quant prefill OOM'd even on 80GB; ``logits_to_keep=1``
    keeps every forward's activations bounded by the chunk, not by T."""
    t = int(ctx.shape[1])
    step = chunk if 0 < chunk < t else t
    for start in range(0, t, step):
        stop = min(t, start + step)
        pos = torch.arange(start, stop, device=ctx.device).unsqueeze(0)
        model(
            ctx[:, start:stop],
            past_key_values=cache,
            use_cache=True,
            position_ids=pos,
            logits_to_keep=1,
        )
    flush(cache)


@torch.no_grad()
def score_quant(
    model: Any, cache: Any, ctx_ids: torch.Tensor, win_ids: torch.Tensor, chunk: int = 0
) -> tuple[float, int]:
    """Prefill into a caller-supplied QuantizedCache (KIVI baseline; chunked when
    ``chunk`` > 0) then score the window; returns nll + token count."""
    ctx = ctx_ids.unsqueeze(0)
    ctx_len = int(ctx_ids.shape[0])
    _prefill_plain(model, cache, ctx, chunk)
    return _score_window(model, cache, ctx_len, win_ids)


# ------------------------------------------------------------------- the arms

# The YAML `kind` a config declares -> the dispatch key an arm dict carries, which is
# what `retrieve`, `generate` and `_footprint` branch on. They differ in one place:
# `composite` reads better in a config, `press_quant` says what the dispatch does.
KIND = {"composite": "press_quant"}


def build_arm(cfg: ArmCfg, model: Any, t: int) -> dict[str, Any]:
    """One arm of ``configs/arms/<name>.yaml``, resolved for context length ``t``.

    The dict is the harness's unit of work: a ``name`` (the string every record and log
    row carries -- ``legacy_name`` where the config sets one, so a re-run is comparable
    with the archive key for key), a dispatch ``kind``, whatever identifying parameters
    ``_footprint`` needs to bill it, and a zero-argument factory. Caches and presses are
    stateful, so the factory is called once per sample rather than shared.

    ``kwargs`` is the cache constructor's keyword set verbatim (``config.arm_kwargs``);
    ``make`` closes over a COPY of it, so mutating the stored dict cannot change what a
    later ``make()`` builds. ``model`` is only captured, never touched, so an arm can be
    built without one -- which is what ``tests/test_config_parity.py`` does.
    """
    kind = KIND.get(cfg.kind, cfg.kind)
    arm: dict[str, Any] = {
        "name": cfg.legacy_name or cfg.name,
        "kind": kind,
        "rank": None,
        "chunkable": cfg.chunkable,
    }
    if kind == "full":
        return {**arm, "make": lambda: None}
    if kind == "bug":
        kw = arm_kwargs(cfg, t)
        arm["rank"] = int(kw["rank"])
        arm["kwargs"] = kw
        # The surprise tier's three knobs are lifted out of the kwargs because
        # `_footprint` bills the position/surprise buffers off them; a fifo arm carries
        # retention="fifo" and no tier, exactly as the legacy dict did by omission.
        for key in ("retention", "hh_select", "hh_budget"):
            if key in kw:
                arm[key] = kw[key]
        return {**arm, "make": lambda kw=dict(kw): BugStreamingCache(model, **kw)}
    if kind == "shadow":
        kw = arm_kwargs(cfg, t)
        arm["rank_s"] = int(kw["rank_s"])
        arm["kwargs"] = kw
        return {**arm, "make": lambda kw=dict(kw): ShadowKVCache(model, **kw)}
    if kind == "quant":
        return {**arm, **_quant_fields(cfg), "make": _quant_factory(cfg, model)}
    if kind == "press_quant":
        # Eviction x quantization (MiniKV-style): the press prunes to the keep fraction
        # during single-shot prefill and the kvpress forward_hook's QuantizedCache branch
        # re-quantizes the survivors, so the two compression axes multiply.
        return {
            **arm,
            "keep": float(cfg.press["keep"]),
            **_quant_fields(cfg),
            "make_press": _evict_factory(cfg),
            "make_cache": _quant_factory(cfg, model),
        }
    if kind == "press":
        return {**arm, **_press(cfg)}
    if kind == "quant_faithful":
        raise NotImplementedError("L2")  # faithful KIVI (G=32, R=128, fp prefill)
    raise ValueError(f"unknown arm kind {cfg.kind!r} in configs/arms/{cfg.name}.yaml")


def _quant_fields(cfg: ArmCfg) -> dict[str, Any]:
    """The quantized tier's identifying parameters, as `_footprint` reads them."""
    q = cfg.quant
    return {
        "nbits": int(q["nbits"]),
        "quant_group": int(q["group"]),
        "quant_residual": int(q["residual"]),
        "quant_scheme": str(q["scheme"]),
        "quant_backend": str(q["backend"]),
    }


def _quant_factory(cfg: ArmCfg, model: Any) -> Any:
    q = cfg.quant
    return lambda: make_quant_cache(
        model.config,
        nbits=int(q["nbits"]),
        scheme=str(q["scheme"]),
        backend=str(q["backend"]),
        group=int(q["group"]),
        residual=int(q["residual"]),
    )


def _evict_factory(cfg: ArmCfg) -> Any:
    """SnapKV for a config named ``snapkv*``, ExpectedAttention otherwise -- the two
    scorer presses take the same single parameter, so the name is what separates them."""
    from kvpress import ExpectedAttentionPress, SnapKVPress

    cls = SnapKVPress if cfg.name.startswith("snapkv") else ExpectedAttentionPress
    return lambda: cls(compression_ratio=1.0 - float(cfg.press["keep"]))


def _press(cfg: ArmCfg) -> dict[str, Any]:
    """A prefill press, dispatched on which parameter its ``press:`` block carries:
    ``ratio`` is ThinK's channel-wise key pruning, ``rank`` the per-sequence SVD oracle
    (the rows published as ``palu-*``), ``keep`` an eviction press's kept fraction.
    ``press_type`` is what `_footprint` branches on for the two analytic footprints."""
    p = cfg.press
    if "ratio" in p:
        from kvpress import ThinKPress

        ratio = float(p["ratio"])
        return {
            "press_type": "think",
            "think_ratio": ratio,
            "make": lambda: ThinKPress(key_channel_compression_ratio=ratio),
        }
    if "rank" in p:
        from kvdlra.baselines.svd_oracle import SVDOraclePress

        ratio, group = float(p["rank"]), int(p["group"])
        return {
            "press_type": "palu",
            "palu_rank_ratio": ratio,
            "palu_group": group,
            "make": lambda: SVDOraclePress(rank_ratio=ratio, group=group),
        }
    return {"keep": float(p["keep"]), "make": _evict_factory(cfg)}


def _tracked_rank(u: torch.Tensor | None) -> int:
    """Columns a stored basis actually holds; 0 when the layer has yet to absorb one.

    The rank that is BILLED -- as against ``arm["rank"]``, the configured *cap*: the
    Week-17 relative singular-value floor (``min_sv_frac``) drops near-null tail
    directions, so a floor-on layer tracks fewer columns than the cap and the cap
    over-bills it (audit finding 0.2). The K and V streams collapse independently.
    """
    return 0 if u is None else int(u.shape[1])


def _footprint(arm: dict[str, Any], cache: Cache, t: int, n: int, h_kv: int) -> acc.Footprint:
    """Per-layer footprint of the arm's *post-prefill* state."""
    kind = arm["kind"]
    if kind == "bug":
        assert isinstance(cache, BugStreamingCache)
        layers = cache._bug_layers()
        layer = layers[0]
        # The LIVE tracked rank, never arm["rank"]. Every rank term in `bug_footprint`
        # is `2*rank*x` -- symmetric in the two streams -- so where the floor collapsed
        # K and V to different widths (measured 21 vs 23 on one tiny-model layer) their
        # MEAN is what reproduces the measured `stored_state_numel` exactly, and it can
        # be a half-integer. No basis yet => rank 0, billed with u_present=False so the
        # basis and core terms drop out together.
        #
        # Averaged over ALL the layers, not read off layer 0: the floor collapses every
        # layer's streams independently (measured 21/23 and 20/23 on the two tiny-model
        # layers), and this one footprint is what every axis multiplies by the layer
        # count. Every other term below is layer-invariant (the tier lengths are driven
        # by the token count, which every layer shares), so the mean rank is exactly the
        # mean of the per-layer `stored_state_numel()` -- pinned in
        # tests/test_effective_rank_billing.py. Floor off, every layer sits at the cap
        # and the bill is byte-identical to the one-layer read.
        rank = sum(_tracked_rank(la.u_k) + _tracked_rank(la.u_v) for la in layers) / (
            2 * len(layers)
        )

        # Thread the arm's retention + hh_select so surprise arms count their
        # position/surprise buffers too (fifo default keeps existing arms
        # byte-identical); the anti-drift pin guards this against drift.
        # Split the fp32 coordinate tier from the quantized tier so the coded columns
        # are billed at their nbits, not as fp32 coords (Week-18: the old
        # _f_len()+_q_len() lumped them, mis-billing any bug quant arm). q_len==0 for
        # every non-quant arm -> byte-identical there; the anti-drift pin covers both.
        q_len = layer._q_len()
        return acc.bug_footprint(
            n,
            rank=rank,
            coord_count=layer._f_len(),
            recent_len=layer._recent_len(),
            n_sink=N_SINK,
            retention=arm.get("retention", "fifo"),
            hh_count=layer._hh_len(),
            u_present=layer.u_k is not None,
            quant_count=q_len,
            quant_bits=layer.quant_bits if q_len else None,
        )
    if kind == "shadow":
        from kvdlra.cache import ShadowKVLayer

        slayer = next(la for la in cache.layers if isinstance(la, ShadowKVLayer))
        return acc.shadow_footprint(
            t,
            n,
            h_kv,
            n // h_kv,
            rank_s=int(arm["rank_s"]),
            chunk=slayer.chunk,
            n_sink=slayer.n_sink,
            recent_len=slayer._recent_len(),
        )
    if kind == "full":
        return acc.full_cache_footprint(t, n)
    if kind == "quant":  # QuantizedCache is NOT a DynamicCache subclass -> branch first
        return acc.quant_footprint(
            t,
            n,
            nbits=int(arm["nbits"]),
            group=int(arm["quant_group"]),
            residual_length=int(arm["quant_residual"]),
            scale_words=aux_words(cache),  # billed at the backend's real aux precision
        )
    if kind == "press_quant":  # Week-20 composite: kept fraction, survivors quantized.
        # The forward_hook prunes then re-quantizes and empties the fp16 residual
        # (compat.py: cl.keys = zeros(0), cl.cumulative_length = kept), so bill the
        # measured kept-token count at nbits with NO residual -- the composite
        # footprint (k*nbits/16 + aux).
        kept = int(getattr(cache.layers[0], "cumulative_length", 0))
        assert kept > 0, "press_quant: empty cache -- forward_hook did not fire?"
        return acc.quant_footprint(
            kept,
            n,
            nbits=int(arm["nbits"]),
            group=int(arm["quant_group"]),
            residual_length=0,
            scale_words=aux_words(cache),
        )
    if arm.get("press_type") == "think":
        # ThinK zeros channels (no measured gain) -> analytic footprint (K pruned).
        head_dim = n // h_kv
        return acc.think_footprint(t, n, head_dim, h_kv, float(arm["think_ratio"]))
    if arm.get("press_type") == "palu":
        # Palu reconstructs same-shape K/V (Mode A); analytic low-rank footprint.
        head_dim = n // h_kv
        return acc.palu_footprint(
            t, n, head_dim, h_kv, float(arm["palu_rank_ratio"]), group=int(arm["palu_group"])
        )
    # eviction press: measure kept fraction from the compressed DynamicCache
    assert isinstance(cache, DynamicCache)
    kept = int(cast(Any, cache.layers[0]).keys.shape[2])
    return acc.evict_footprint(t, n, kept / t)


# -------------------------------------------------------------------- runner


def windows(ids: torch.Tensor, t: int, window: int, n_samples: int) -> list[Sample]:
    """Up to ``n_samples`` non-overlapping (context, continuation) slices of ``ids``."""
    span = t + window
    return [(ids[s : s + t], ids[s + t : s + span]) for s in range(0, ids.shape[0] - span, span)][
        :n_samples
    ]


def run_ppl(
    arms: list[dict[str, Any]],
    model: Any,
    samples: list[Sample],
    t: int,
    *,
    chunk: int,
    n: int,
    h_kv: int,
    device: str,
    corpus: str = "wikitext-103",
) -> list[dict[str, Any]]:
    """Score every arm on the same windows at context length ``t``; one row per arm.

    ``corpus`` only labels: the windows are already cut, and it rides onto the row and
    the ``[pplw]`` line so a pooled number can never be read as belonging to a corpus it
    was not measured on. The default is `config.TaskCfg.corpus`'s -- what v1 scored --
    and `kvdlra.eval.runner` always passes the task's.

    An arm that raises does not take the sweep down with it: its row carries
    ``status`` OOM/error and the loop moves on, which is the same rule the trial runner
    applies (a failure is recorded, never silently dropped).
    """
    rows: list[dict[str, Any]] = []
    for arm in arms:
        peak_ctx = acc.measure_peak_gpu(device)
        try:
            with peak_ctx as peak_get:
                total_nll, total_tok = 0.0, 0
                window_nlls: list[float] = []  # per-window MEAN nll (nats/token)
                window_toks: list[int] = []  # per-window scored-token counts
                fp: acc.Footprint | None = None
                eff_rank: int | None = None  # bug arms only: the live tracked rank
                arm_chunk = chunk if arm.get("chunkable", True) else 0
                for sample_idx, (ctx_ids, win_ids) in enumerate(samples):
                    if arm["kind"] == "press" or arm["kind"] == "full":
                        press = arm["make"]()
                        nll, ntok, cache = score_press(model, press, ctx_ids, win_ids, arm_chunk)
                    elif arm["kind"] == "quant":
                        cache = arm["make"]()
                        nll, ntok = score_quant(model, cache, ctx_ids, win_ids, arm_chunk)
                    else:
                        cache = arm["make"]()
                        nll, ntok = score_streaming(
                            model,
                            cache,
                            ctx_ids,
                            win_ids,
                            arm_chunk,
                            arm=str(arm["name"]),
                            idx=sample_idx,
                        )
                    total_nll += nll
                    total_tok += ntok
                    window_nlls.append(nll / ntok)
                    window_toks.append(ntok)
                    if fp is None:
                        fp = _footprint(arm, cache, t, n, h_kv)
                        if isinstance(cache, BugStreamingCache):
                            # The narrowest K basis in the cache: the ppl axis's
                            # one-number view of how far the floor collapsed the gist
                            # (`diag.jsonl` carries the per-layer K and V ranks).
                            eff_rank = min(_tracked_rank(la.u_k) for la in cache._bug_layers())
                    del cache
                    gc.collect()
                peak = peak_get()
            assert fp is not None
            # Pooled ppl: BYTE-IDENTICAL to the pre-Week-15 computation (summed
            # nats / summed tokens, then exp). window_nlls/window_toks are a pure
            # ADDITION so error bars exist (Week-15 intervals); the
            # pin: ppl == exp(sum(nll_i*tok_i)/sum(tok_i)) recomputed from them.
            ppl = float(torch.tensor(total_nll / total_tok).exp())
            _log_pplw(t, arm["name"], window_nlls, window_toks[0], corpus)
            row = {
                "method": arm["name"],
                "kind": arm["kind"],
                "rank": arm["rank"],
                "T": t,
                "corpus": corpus,
                "ppl": ppl,
                "window_nlls": window_nlls,
                "window_toks": window_toks,
                "float_equiv_per_layer": fp.float_equiv(),
                "tok_equiv_per_layer": fp.tok_equiv(n),
                "ratio_fp16": fp.ratio_fp16(t, n),
                "ratio_stored_bits": fp.ratio_stored_bits(t, n),
                "gpu_ratio_fp16": fp.gpu_ratio_fp16(t, n),
                "cpu_ratio_fp16": fp.cpu_ratio_fp16(t, n),
                "peak_gpu_bytes": peak,
                "eff_rank": eff_rank,
                "status": "ok",
            }
        except Exception as exc:
            oom = isinstance(exc, torch.cuda.OutOfMemoryError)
            row = {
                "method": arm["name"],
                "kind": arm["kind"],
                "T": t,
                "status": "OOM" if oom else "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if device.startswith("cuda"):
                # Week-19: say WHERE the memory went (allocated vs peak) so an OOM
                # in the pod log is diagnosable without a re-run.
                row["mem_alloc_gb"] = torch.cuda.memory_allocated() / 2**30
                row["mem_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                torch.cuda.empty_cache()
                print(
                    f"  {arm['name']:14s} [T={t}] mem alloc={row['mem_alloc_gb']:.1f}GB "
                    f"peak={row['mem_peak_gb']:.1f}GB",
                    flush=True,
                )
            print(f"  {arm['name']:14s} [T={t}] {row['status']}: {row['error'][:110]}")
        rows.append(row)
        _log_row(row)
    return rows


def _log_pplw(t: int, name: str, window_nlls: list[float], ntok: int, corpus: str) -> None:
    """The ``[pplw]`` per-window NLL line -- one per (arm, T), greppable ``^\\[pplw``.

    `vastai logs` truncates a line at ~500 chars, so a would-be >400-char line splits
    into ``part=i/N`` lines of 8 values each; dropping the fragments loses the whole
    sweep silently, so `records.parse_pplw_lines` reassembles them and raises on a gap.
    Windows are uniform-length by construction (exact slices), so one ``ntok`` covers
    the group. Format (the harvest regex, `records.PPLW_RE`)::

        ^\\[pplw\\] T=(\\d+) (\\S+) ntok=(\\d+)(?: part=(\\d+)/(\\d+))? nlls=([0-9.,]+)
        (?: corpus=(\\S+))?$          # one pattern; wrapped here only to fit the line limit

    ``corpus=`` goes LAST and every fragment repeats it, so the group is self-describing
    however the log was truncated -- and so the archived lines, which have none, keep
    matching (`records.PPLW_RE` takes it as an optional trailing group).
    """
    vals = [f"{v:.6f}" for v in window_nlls]
    head = f"[pplw] T={t} {name} ntok={ntok}"
    tail = f" corpus={corpus}"
    line = f"{head} nlls={','.join(vals)}{tail}"
    if len(line) <= 400:
        print(line, flush=True)
        return
    groups = [vals[i : i + 8] for i in range(0, len(vals), 8)]
    for pi, grp in enumerate(groups, 1):
        print(f"{head} part={pi}/{len(groups)} nlls={','.join(grp)}{tail}", flush=True)


def _log_row(row: dict[str, Any]) -> None:
    if row["status"] != "ok":
        print(f"  {row['method']:14s} [T={row['T']}] {row['status']}", flush=True)
        return
    # `sbits=` appended after `ratio=` -- records.PPL_RE captures ratio= and ignores the
    # tail, so archived ppl lines keep parsing unchanged.
    # `eff_rank=` (bug arms only) goes next: records.PPL_RE is a prefix match, so a
    # field appended after `sbits=` leaves every archived and new line parsing the same.
    # `corpus=` goes LAST, same rule as `_log_pplw`'s -- records.PPL_RE takes it as an
    # optional trailing group, so archived lines (neither field) keep parsing too.
    eff = row.get("eff_rank")
    print(
        f"  {row['method']:14s} [T={row['T']}] ppl={row['ppl']:.3f} "
        f"tok_eq/layer={row['tok_equiv_per_layer']:.1f} ratio={row['ratio_fp16']:.3f} "
        f"sbits={row.get('ratio_stored_bits', float('nan')):.3f}"
        f"{'' if eff is None else f' eff_rank={eff}'}"
        f" corpus={row['corpus']}",
        flush=True,
    )
