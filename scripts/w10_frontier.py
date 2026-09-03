"""Week-10: the definitive long-context frontier -- perplexity vs honestly-counted
KV-cache memory, one consistent protocol across every method.

Compares **BUG** (``BugStreamingCache``, ranks {32,64,128,256}) against **MorphKV**
(``MorphKVCache``, decode eviction) and the kvpress prefill presses **SnapKV** /
**ExpectedAttention**, on ONE protocol: prefill ``T`` tokens with each method's
compression active, then score teacher-forced perplexity on a frozen ``W``-token
continuation window attending to the *compressed* cache (the "compress-then-score"
deviation documented in ``perplexity_sweep.window_nll`` / ``w4_fair``, now uniform
for streaming caches and presses alike). Memory is counted honestly per
``kvdlra.accounting`` (float-equivalents/layer, the ``stored_state_numel`` unit)
and cross-checked against the live cache.

Phase 2 (this file's default) banks the FIRST real 3-method frontier at moderate
``T`` with **single-shot prefill** on the Mac CPU / 1B, where single-shot fits.
Chunked ``ingest`` prefill (Phase 3) unlocks 32K/64K; ShadowKV (Phase 6) and the
RULER secondary axis (Phase 4) land later. See ``docs/week10-plan.md``.

Example (CPU smoke, 1B)
-----------------------
    uv run python scripts/w10_frontier.py --device cpu --T 1024 \
        --ranks 32 64 --n-samples 1 --methods bug morph snapkv
"""

from __future__ import annotations

import argparse
import gc
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import matplotlib
import torch
from perplexity_sweep import load_corpus_ids, load_model
from torch.nn.functional import cross_entropy
from transformers.cache_utils import Cache, DynamicCache

from kvdlra import accounting as acc
from kvdlra.cache import BugStreamingCache, MorphKVCache, ShadowKVCache
from kvdlra.press.compat import install_kvpress_prefill_compat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
JSON_BEGIN = "===W10_FRONTIER_JSON_BEGIN==="
JSON_END = "===W10_FRONTIER_JSON_END==="
N_SINK = 4


# --------------------------------------------------------------------- scoring


@torch.no_grad()
def _score_window(
    model: Any, cache: Cache, ctx_len: int, win_ids: torch.Tensor
) -> tuple[float, int]:
    """Summed NLL (nats) + scored-token count for the frozen continuation window,
    at TRUE positions -- byte-for-byte ``perplexity_sweep.window_nll``'s scorer."""
    win = win_ids.unsqueeze(0)
    win_len = int(win_ids.shape[0])
    pos = torch.arange(ctx_len, ctx_len + win_len, device=win_ids.device).unsqueeze(0)
    out = model(win, past_key_values=cache, use_cache=True, position_ids=pos)
    logits = out.logits[0]
    nll = cross_entropy(logits[:-1], win_ids[1:], reduction="sum")
    return float(nll), int(win_ids[1:].shape[0])


def _prefill_chunked(model: Any, cache: Cache, ctx: torch.Tensor, chunk: int) -> None:
    """OOM-safe chunked prefill for a streaming cache (Phase 3): the first chunk
    single-shot-prefills (cumulative_length == 0), each later chunk ingests; absorb
    is consolidated after every chunk. ``logits_to_keep=1`` drops the P x vocab
    logits so no forward's activations scale with the full ``T``."""
    t = int(ctx.shape[1])
    ingesting = getattr(cache, "ingesting")  # noqa: B009 (BugStreamingCache/MorphKVCache)
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
    cache: BugStreamingCache | MorphKVCache | ShadowKVCache,
    ctx_ids: torch.Tensor,
    win_ids: torch.Tensor,
    chunk: int = 0,
) -> tuple[float, int]:
    """Prefill into a streaming cache (compresses) then frozen-window score over the
    compressed cache (non-mutating). ``chunk > 0`` (and < ctx) uses OOM-safe chunked
    ingest; otherwise single-shot."""
    ctx = ctx_ids.unsqueeze(0)
    ctx_len = int(ctx_ids.shape[0])
    with cache.attach(model):
        if 0 < chunk < ctx_len:
            _prefill_chunked(model, cache, ctx, chunk)
        else:
            model(ctx, past_key_values=cache, use_cache=True, logits_to_keep=1)
    with cache.frozen_scoring():
        return _score_window(model, cache, ctx_len, win_ids)


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


# ------------------------------------------------------------------- the arms


def _load_wkey(path: str | None) -> list[Any] | None:
    """Load a calibrated Q-BUG whitening file -> per-layer list of ``(n,)`` vectors.

    The file (``scripts/w12_calibrate_qkey.py``) holds ``{"w_key": (L, n)}``; the
    cache takes a per-layer list, so unbind the layer axis. ``None`` when unset."""
    if not path:
        return None
    blob = torch.load(path, weights_only=False)
    w = blob["w_key"] if isinstance(blob, dict) else blob
    return [row.contiguous() for row in w]


def build_arms(args: argparse.Namespace, model: Any, t: int) -> list[dict[str, Any]]:
    """Each arm: name, kind ("bug"|"morph"|"press"|"full"), and a zero-arg factory
    for a fresh cache/press (stateful -> rebuilt per sample), for context length ``t``."""
    rw, ab = args.recent_window, args.absorb_block
    msf = float(getattr(args, "min_sv_frac", 0.0))  # Week-17 integrator floor (0.0 = off)
    fsuf = f"-f{msf:g}" if msf > 0.0 else ""
    arms: list[dict[str, Any]] = []
    want = set(args.methods)

    # Load-bearing invariant: single-shot _prefill bypasses the SLASH exact tier
    # (it only populates via chunked ingest), so a SurpriseSLASH arm run without
    # --chunk would silently retrieve nothing -- a wrong-reason zero. Fail loud.
    if want & {"bugslash", "bugevict"} and not getattr(args, "chunk", 0):
        raise ValueError(
            "bugslash/bugevict require chunked prefill (--chunk > 0): single-shot "
            "prefill bypasses the SLASH exact tier and the needle would be smeared."
        )

    if "full" in want:
        arms.append({"name": "full", "kind": "full", "rank": None, "make": lambda: None})

    if "bug" in want:
        # coord_budget >= mid so the whole middle is retained as rank-r coords
        # (BUG-as-prefill-compressor: memory ~ rank/n, one point per rank).
        cb = t + rw + ab
        for r in args.ranks:
            arms.append(
                {
                    "name": f"bug-r{r}{fsuf}",
                    "kind": "bug",
                    "rank": r,
                    "make": (
                        lambda r=r, cb=cb: BugStreamingCache(
                            model,
                            rank=r,
                            coord_budget=cb,
                            recent_window=rw,
                            absorb_block=ab,
                            n_sink=N_SINK,
                            retention="fifo",
                            min_sv_frac=msf,
                        )
                    ),
                }
            )

    if "bugslash" in want:
        # Week-11 SurpriseSLASH: BUG low-rank gist + a surprise-selected EXACT
        # outlier tier (hh_budget tokens kept verbatim). The needle is a
        # low-attention high-residual outlier the low-rank basis fits worst, so a
        # surprise-scored exact tier catches it where plain BUG (0% at 32K) can't.
        # MUST prefill via chunked ingest (--chunk > 0) -- single-shot bypasses the
        # exact tier. coord_budget >= mid keeps the whole (non-hh) bulk low-rank.
        cb = t + rw + ab
        # Week-12 H1 ablation: --hh-discard keeps selection + withholding
        # identical but hides the pool from attention (hh_retain=False). The
        # name prefix is deliberately NOT a superstring/substring of "bugS-"
        # so ad-hoc greps over mixed logs cannot pool the two arm families.
        retain = not getattr(args, "hh_discard", False)
        # Week-12 Q-BUG: --qwhiten-file loads a calibrated per-layer key-whitening
        # diagonal (scripts/w12_calibrate_qkey.py) -> arm name gains a "Q".
        w_key = _load_wkey(getattr(args, "qwhiten_file", None))
        # Week-13 T-B: --warmup-seed seeds the exact tier from the first ingest
        # chunk's outliers -> arm name gains a "seed" suffix (bugS vs bugSseed A/B).
        warmup = bool(getattr(args, "warmup_seed", False))
        # Week-15 T2: --score-rank caps the SLASH surprise-scoring basis at a
        # leading-column subview (selection-rank decoupled from storage-rank) ->
        # arm name gains a "-s{k}" SUFFIX after -h{hh} (suffix, not prefix, so the
        # bugS-* prefix-grep discipline over mixed logs stays intact).
        score_rank = getattr(args, "score_rank", None)
        prefix = "bugS" if retain else "bugSdrop"
        if w_key is not None:
            if not retain:
                raise ValueError("--qwhiten-file (Q-BUG) is not combined with --hh-discard")
            prefix = "bugSQ"
        if warmup:
            prefix += "seed"
        suffix = f"-s{score_rank}" if score_rank is not None else ""
        for r in args.ranks:
            for hh in args.hh_budgets:
                arms.append(
                    {
                        "name": f"{prefix}-r{r}-h{hh}{suffix}{fsuf}",
                        "kind": "bug",
                        "rank": r,
                        "retention": "lowrank_surprise",
                        "hh_select": "surprise",
                        "hh_budget": hh,
                        "make": (
                            lambda r=r, cb=cb, hh=hh: BugStreamingCache(
                                model,
                                rank=r,
                                coord_budget=cb,
                                recent_window=rw,
                                absorb_block=ab,
                                n_sink=N_SINK,
                                retention="lowrank_surprise",
                                hh_budget=hh,
                                hh_select="surprise",
                                hh_neighbor=args.hh_neighbor,
                                hh_retain=retain,
                                seed_hh_warmup=warmup,
                                w_key=w_key,
                                score_rank=score_rank,
                                min_sv_frac=msf,
                            )
                        ),
                    }
                )

    if "bugevict" in want:
        # Attribution control: a degenerate rank-1 BUG (negligible gist) with a
        # surprise-selected exact tier == ~pure surprise-selected verbatim
        # eviction. If bugslash retrieves at the SAME hh_budget as this, the win
        # is the SELECTION RULE, not BUG's gist (the honesty crux, Week-7/8 wall).
        for hh in args.hh_budgets:
            arms.append(
                {
                    "name": f"bugEVICT-h{hh}",
                    "kind": "bug",
                    "rank": 1,
                    "retention": "lowrank_surprise",
                    "hh_select": "surprise",
                    "hh_budget": hh,
                    "make": (
                        lambda hh=hh: BugStreamingCache(
                            model,
                            rank=1,
                            coord_budget=1,
                            recent_window=rw,
                            absorb_block=ab,
                            n_sink=N_SINK,
                            retention="lowrank_surprise",
                            hh_budget=hh,
                            hh_select="surprise",
                            hh_neighbor=args.hh_neighbor,
                        )
                    ),
                }
            )

    if "morph" in want:
        for keep in args.morph_keeps:
            arms.append(
                {
                    "name": f"morph-k{keep}",
                    "kind": "morph",
                    "rank": None,
                    "keep": keep,
                    "make": (
                        lambda keep=keep: MorphKVCache(
                            model,
                            capacity=max(1, int(keep * t) - rw),
                            recent_window=rw,
                        )
                    ),
                }
            )

    if want & {"snapkv", "ea"}:
        from kvpress import ExpectedAttentionPress, SnapKVPress

        for keep in args.evict_keeps:
            cr = 1.0 - keep
            if "snapkv" in want:
                arms.append(
                    {
                        "name": f"snapkv-k{keep}",
                        "kind": "press",
                        "rank": None,
                        "keep": keep,
                        "make": lambda cr=cr: SnapKVPress(compression_ratio=cr),
                    }
                )
            if "ea" in want:
                arms.append(
                    {
                        "name": f"ea-k{keep}",
                        "kind": "press",
                        "rank": None,
                        "keep": keep,
                        "make": lambda cr=cr: ExpectedAttentionPress(compression_ratio=cr),
                    }
                )

    if "think" in want:  # ThinK: channel-wise KEY low-rank (kvpress drop-in)
        from kvpress import ThinKPress

        for cr in args.think_ratios:
            arms.append(
                {
                    "name": f"think-c{cr}",
                    "kind": "press",
                    "press_type": "think",
                    "rank": None,
                    "think_ratio": cr,
                    "chunkable": False,  # ThinKPress is not a ScorerPress -> no ChunkPress
                    "make": lambda cr=cr: ThinKPress(key_channel_compression_ratio=cr),
                }
            )

    if "palu" in want:  # Palu: low-rank projection of K+V (reconstruct-then-attend)
        from kvdlra.press import PaluPress

        for rr in args.palu_ranks:
            arms.append(
                {
                    "name": f"palu-r{rr}",
                    "kind": "press",
                    "press_type": "palu",
                    "rank": None,
                    "palu_rank_ratio": rr,
                    "palu_group": args.palu_group,
                    "chunkable": False,  # single-shot prefill only (SVD over the whole T)
                    "make": lambda rr=rr: PaluPress(rank_ratio=rr, group=args.palu_group),
                }
            )

    if "shadow" in want:  # ShadowKV: low-rank K on GPU + V offloaded to CPU + sparse decode
        from kvdlra.cache import ShadowKVCache

        for rs in args.shadow_ranks:
            arms.append(
                {
                    "name": f"shadow-r{rs}",
                    "kind": "shadow",
                    "rank": None,
                    "rank_s": rs,
                    "chunkable": False,  # single-shot prefill only (port scope guard)
                    "make": (
                        lambda rs=rs: ShadowKVCache(
                            model,
                            rank_s=rs,
                            top_k=args.shadow_topk,
                            chunk=8,
                            recent_window=rw,
                            n_sink=N_SINK,
                        )
                    ),
                }
            )
    return arms


def _footprint(arm: dict[str, Any], cache: Cache, t: int, n: int, h_kv: int) -> acc.Footprint:
    """Honest per-layer footprint of the arm's *post-prefill* state."""
    kind = arm["kind"]
    if kind == "bug":
        assert isinstance(cache, BugStreamingCache)
        layer = cache._bug_layers()[0]
        # Thread the arm's retention + hh_select so surprise arms count their
        # position/surprise buffers honestly (fifo default keeps existing arms
        # byte-identical); the anti-drift pin guards this against drift.
        return acc.bug_footprint(
            n,
            rank=int(arm["rank"]),
            coord_count=layer._f_len() + layer._q_len(),
            recent_len=layer._recent_len(),
            n_sink=N_SINK,
            retention=arm.get("retention", "fifo"),
            hh_count=layer._hh_len(),
            hh_select=arm.get("hh_select", "attn"),
            u_present=layer.u_k is not None,
            w_key=layer.w_key is not None,  # Q-BUG: count the frozen whitening diagonal
        )
    if kind == "morph":
        assert isinstance(cache, MorphKVCache)
        mlayer = cast(Any, cache.layers[0])
        kept = int(mlayer.keys.shape[2])
        return acc.morph_footprint(n, h_kv, kept, recent_window=mlayer.recent_window)
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    install_kvpress_prefill_compat()
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    # Memory-efficient attention (never eager at long T -- eager materializes
    # O(T^2); the OOM discipline for the 32K/64K GPU run, harmless on CPU).
    model.config._attn_implementation = "sdpa"
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    n_layers = int(cfg.num_hidden_layers)
    ids = load_corpus_ids(tokenizer, args.device, corpus=args.corpus)
    print(
        f"model={args.model} n={n} h_kv={h_kv} layers={n_layers} corpus={args.corpus}", flush=True
    )

    per_t: dict[int, list[dict[str, Any]]] = {}
    for t in args.T:
        window = t + args.window
        samples = [
            (ids[s : s + t], ids[s + t : s + window])
            for s in range(0, ids.shape[0] - window, window)
        ][: args.n_samples]
        if not samples:
            print(f"[T={t}] corpus too short for {args.n_samples} windows; skipping", flush=True)
            continue
        print(f"[T={t}] {len(samples)} window(s) of {t}+{args.window}", flush=True)
        rows: list[dict[str, Any]] = []
        for arm in build_arms(args, model, t):
            peak_ctx = acc.measure_peak_gpu(args.device)
            try:
                with peak_ctx as peak_get:
                    total_nll, total_tok = 0.0, 0
                    window_nlls: list[float] = []  # per-window MEAN nll (nats/token)
                    window_toks: list[int] = []  # per-window scored-token counts
                    fp: acc.Footprint | None = None
                    arm_chunk = args.chunk if arm.get("chunkable", True) else 0
                    for ctx_ids, win_ids in samples:
                        if arm["kind"] == "press" or arm["kind"] == "full":
                            press = arm["make"]()
                            nll, ntok, cache = score_press(
                                model, press, ctx_ids, win_ids, arm_chunk
                            )
                        else:
                            cache = arm["make"]()
                            nll, ntok = score_streaming(model, cache, ctx_ids, win_ids, arm_chunk)
                        total_nll += nll
                        total_tok += ntok
                        window_nlls.append(nll / ntok)
                        window_toks.append(ntok)
                        if fp is None:
                            fp = _footprint(arm, cache, t, n, h_kv)
                        del cache
                        gc.collect()
                    peak = peak_get()
                assert fp is not None
                # Pooled ppl: BYTE-IDENTICAL to the pre-Week-15 computation (summed
                # nats / summed tokens, then exp). window_nlls/window_toks are a pure
                # ADDITION so error bars exist (docs/week15-significance.md); the
                # pin: ppl == exp(sum(nll_i*tok_i)/sum(tok_i)) recomputed from them.
                ppl = float(torch.tensor(total_nll / total_tok).exp())
                # [pplw] per-window mean NLLs -- one line per (arm, T), greppable
                # ^\[pplw (the ^\[niah harvest discipline). vast logs truncate at
                # ~500 chars, so a would-be >400-char line splits into part=i/N
                # lines of 8 values each. Windows are uniform-length by
                # construction (exact slices); ntok is that per-window count, and
                # the JSON row stays authoritative for exact per-window weights.
                # Format (harvest regex):
                #   ^\[pplw\] T=(\d+) (\S+) ntok=(\d+)
                #     (?: part=(\d+)/(\d+))? nlls=([0-9.,]+)$
                vals = [f"{v:.6f}" for v in window_nlls]
                head = f"[pplw] T={t} {arm['name']} ntok={window_toks[0]}"
                line = f"{head} nlls={','.join(vals)}"
                if len(line) <= 400:
                    print(line, flush=True)
                else:
                    groups = [vals[i : i + 8] for i in range(0, len(vals), 8)]
                    for pi, grp in enumerate(groups, 1):
                        part = f"{head} part={pi}/{len(groups)} nlls={','.join(grp)}"
                        print(part, flush=True)
                row = {
                    "method": arm["name"],
                    "kind": arm["kind"],
                    "rank": arm["rank"],
                    "T": t,
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
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(f"  {arm['name']:14s} [T={t}] {row['status']}: {row['error'][:110]}")
            rows.append(row)
            _log_row(row)
        per_t[t] = rows
    return {
        "model": args.model,
        "n_features": n,
        "n_layers": n_layers,
        "window": args.window,
        "corpus": args.corpus,
        "per_T": {str(t): rows for t, rows in per_t.items()},
    }


def _log_row(row: dict[str, Any]) -> None:
    if row["status"] != "ok":
        print(f"  {row['method']:14s} [T={row['T']}] {row['status']}", flush=True)
        return
    # `sbits=` appended after `ratio=` -- PPL_RE (w11_merge) captures ratio= and
    # ignores the tail, so archived ppl lines keep parsing unchanged.
    print(
        f"  {row['method']:14s} [T={row['T']}] ppl={row['ppl']:.3f} "
        f"tok_eq/layer={row['tok_equiv_per_layer']:.1f} ratio={row['ratio_fp16']:.3f} "
        f"sbits={row.get('ratio_stored_bits', float('nan')):.3f}",
        flush=True,
    )


# --------------------------------------------------------------------- plot


def _plot(blob: dict[str, Any], out: Path) -> None:
    per_t = blob["per_T"]
    fig, axes = plt.subplots(1, len(per_t), figsize=(6.5 * len(per_t), 5), squeeze=False)
    kind_style = {
        "bug": ("tab:orange", "o"),
        "morph": ("tab:green", "s"),
        "press": ("tab:red", "^"),
        "full": ("tab:blue", "*"),
    }
    ordered = sorted(per_t.items(), key=lambda kv: int(kv[0]))
    for ax, (t, rows) in zip(axes[0], ordered, strict=False):
        ok = [r for r in rows if r["status"] == "ok"]
        for kind, (c, m) in kind_style.items():
            pts = sorted(
                (r["tok_equiv_per_layer"], r["ppl"], r["method"]) for r in ok if r["kind"] == kind
            )
            if not pts:
                continue
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                marker=m,
                color=c,
                lw=1.8,
                ms=8,
                label=kind,
                alpha=0.9,
            )
        full = [r for r in ok if r["kind"] == "full"]
        if full:
            ax.axhline(full[0]["ppl"], ls=":", c="grey", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("KV memory (float-equiv token-eq / layer)  ->  more compression left")
        ax.set_ylabel("perplexity (lower is better)")
        ax.set_title(f"T = {t}")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Week-10 frontier: BUG vs eviction -- {blob['model'].split('/')[-1]}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=150)
    print(f"[wrote {out.with_suffix('.png')}]", flush=True)


# --------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--T", type=int, nargs="+", default=[2048])
    parser.add_argument("--window", type=int, default=512, help="continuation scoring window W")
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--morph-keeps", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--evict-keeps", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--think-ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--palu-ranks", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--palu-group", type=int, default=1)
    parser.add_argument("--shadow-ranks", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--shadow-topk", type=int, default=256)
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--absorb-block", type=int, default=16)
    parser.add_argument(
        "--hh-budgets",
        type=int,
        nargs="+",
        default=[256, 1024, 2048],
        help="Week-11 SurpriseSLASH exact-tier sizes (bugslash/bugevict verbatim tokens)",
    )
    parser.add_argument(
        "--hh-neighbor", type=int, default=0, help="SurpriseSLASH span-expansion window (0=off)"
    )
    parser.add_argument(
        "--hh-discard",
        action="store_true",
        help="Week-12 H1 ablation: bugslash arms select-and-DISCARD (hh_retain=False; "
        "pool invisible to attention) -> bugSdrop-* arm names",
    )
    parser.add_argument(
        "--qwhiten-file",
        default=None,
        help="Week-12 Q-BUG: calibrated per-layer key-whitening diagonal "
        "(scripts/w12_calibrate_qkey.py) -> bugSQ-* arms (query-metric gist)",
    )
    parser.add_argument(
        "--warmup-seed",
        action="store_true",
        help="Week-13 T-B: seed the exact tier from the first ingest chunk's outliers "
        "-> bugSseed-* arms (fixes the warm-up window; requires --chunk>0)",
    )
    parser.add_argument(
        "--score-rank",
        type=int,
        default=None,
        help="Week-15 T2: cap the SLASH surprise-scoring basis at this many leading "
        "columns (selection-rank decoupled from storage-rank) -> '-s{k}' arm suffix; "
        "storage/footprint unchanged; requires 1 <= k <= rank",
    )
    parser.add_argument(
        "--min-sv-frac",
        type=float,
        default=0.0,
        help="Week-17: relative singular-value floor for the streaming integrator "
        "(0.0=off; e.g. 1e-2 caps the tracked rank at the block's numerical rank to "
        "stop high-rank divergence) -> '-f{v}' arm suffix; storage/footprint unchanged",
    )
    parser.add_argument(
        "--chunk", type=int, default=0, help="chunked-prefill block size (0=single-shot)"
    )
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument(
        "--corpus", default="wikitext-103", choices=["wikitext-2", "wikitext-103", "pg19"]
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "full",
            "bug",
            "morph",
            "snapkv",
            "ea",
            "think",
            "palu",
            "shadow",
        ],
    )
    parser.add_argument("--no-ruler", action="store_true", help="skip RULER (Phase 4)")
    parser.add_argument("--out-json", default="results/w10-frontier-1b.json")
    parser.add_argument("--out-fig", default="figures/week10/frontier_longctx")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    out_json = Path(args.out_json)
    if args.plot_only:
        _plot(json.loads(out_json.read_text()), Path(args.out_fig))
        return

    blob = run(args)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(blob, indent=2) + "\n")
    _plot(blob, Path(args.out_fig))
    print(JSON_BEGIN)
    print(json.dumps(blob))
    print(JSON_END)
    print(f"[wrote {out_json}]", flush=True)


if __name__ == "__main__":
    main()
