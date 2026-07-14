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
from kvdlra.cache import BugStreamingCache, MorphKVCache
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


@torch.no_grad()
def score_streaming(
    model: Any,
    cache: BugStreamingCache | MorphKVCache,
    ctx_ids: torch.Tensor,
    win_ids: torch.Tensor,
) -> tuple[float, int]:
    """Single-shot prefill into a streaming cache (compresses), then frozen-window
    score over the compressed cache (non-mutating)."""
    ctx = ctx_ids.unsqueeze(0)
    with cache.attach(model):
        model(ctx, past_key_values=cache, use_cache=True)
    with cache.frozen_scoring():
        return _score_window(model, cache, int(ctx_ids.shape[0]), win_ids)


@torch.no_grad()
def score_press(
    model: Any, press: Any, ctx_ids: torch.Tensor, win_ids: torch.Tensor
) -> tuple[float, int, DynamicCache]:
    """Single-shot prefill through a kvpress prefill press (or None=full), then
    score. Returns the (compressed) DynamicCache so its kept memory is measured."""
    cache = DynamicCache()
    ctx = ctx_ids.unsqueeze(0)
    with press(model) if press is not None else nullcontext():
        model(ctx, past_key_values=cache, use_cache=True)
    nll, ntok = _score_window(model, cache, int(ctx_ids.shape[0]), win_ids)
    return nll, ntok, cache


# ------------------------------------------------------------------- the arms


def build_arms(args: argparse.Namespace, model: Any, t: int) -> list[dict[str, Any]]:
    """Each arm: name, kind ("bug"|"morph"|"press"|"full"), and a zero-arg factory
    for a fresh cache/press (stateful -> rebuilt per sample), for context length ``t``."""
    rw, ab = args.recent_window, args.absorb_block
    arms: list[dict[str, Any]] = []
    want = set(args.methods)

    if "full" in want:
        arms.append({"name": "full", "kind": "full", "rank": None, "make": lambda: None})

    if "bug" in want:
        # coord_budget >= mid so the whole middle is retained as rank-r coords
        # (BUG-as-prefill-compressor: memory ~ rank/n, one point per rank).
        cb = t + rw + ab
        for r in args.ranks:
            arms.append(
                {
                    "name": f"bug-r{r}",
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
    return arms


def _footprint(arm: dict[str, Any], cache: Cache, t: int, n: int, h_kv: int) -> acc.Footprint:
    """Honest per-layer footprint of the arm's *post-prefill* state."""
    kind = arm["kind"]
    if kind == "bug":
        assert isinstance(cache, BugStreamingCache)
        layer = cache._bug_layers()[0]
        return acc.bug_footprint(
            n,
            rank=int(arm["rank"]),
            coord_count=layer._f_len() + layer._q_len(),
            recent_len=layer._recent_len(),
            n_sink=N_SINK,
            retention="fifo",
            hh_count=layer._hh_len(),
            u_present=layer.u_k is not None,
        )
    if kind == "morph":
        assert isinstance(cache, MorphKVCache)
        mlayer = cast(Any, cache.layers[0])
        kept = int(mlayer.keys.shape[2])
        return acc.morph_footprint(n, h_kv, kept, recent_window=mlayer.recent_window)
    if kind == "full":
        return acc.full_cache_footprint(t, n)
    # press: measure kept fraction from the compressed DynamicCache
    assert isinstance(cache, DynamicCache)
    kept = int(cast(Any, cache.layers[0]).keys.shape[2])
    return acc.evict_footprint(t, n, kept / t)


# -------------------------------------------------------------------- runner


def run(args: argparse.Namespace) -> dict[str, Any]:
    install_kvpress_prefill_compat()
    model, tokenizer = load_model(args.model, args.device, args.dtype)
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
                    fp: acc.Footprint | None = None
                    for ctx_ids, win_ids in samples:
                        if arm["kind"] == "press" or arm["kind"] == "full":
                            press = arm["make"]()
                            nll, ntok, cache = score_press(model, press, ctx_ids, win_ids)
                        else:
                            cache = arm["make"]()
                            nll, ntok = score_streaming(model, cache, ctx_ids, win_ids)
                        total_nll += nll
                        total_tok += ntok
                        if fp is None:
                            fp = _footprint(arm, cache, t, n, h_kv)
                        del cache
                        gc.collect()
                    peak = peak_get()
                assert fp is not None
                ppl = float(torch.tensor(total_nll / total_tok).exp())
                row = {
                    "method": arm["name"],
                    "kind": arm["kind"],
                    "rank": arm["rank"],
                    "T": t,
                    "ppl": ppl,
                    "float_equiv_per_layer": fp.float_equiv(),
                    "tok_equiv_per_layer": fp.tok_equiv(n),
                    "ratio_fp16": fp.ratio_fp16(t, n),
                    "gpu_ratio_fp16": fp.gpu_ratio_fp16(t, n),
                    "cpu_ratio_fp16": fp.cpu_ratio_fp16(t, n),
                    "peak_gpu_bytes": peak,
                    "status": "ok",
                }
            except torch.cuda.OutOfMemoryError:  # pragma: no cover - GPU-only path
                row = {"method": arm["name"], "kind": arm["kind"], "T": t, "status": "OOM"}
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
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
    print(
        f"  {row['method']:14s} [T={row['T']}] ppl={row['ppl']:.3f} "
        f"tok_eq/layer={row['tok_equiv_per_layer']:.1f} ratio={row['ratio_fp16']:.3f}",
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
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--absorb-block", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument(
        "--corpus", default="wikitext-103", choices=["wikitext-2", "wikitext-103", "pg19"]
    )
    parser.add_argument("--methods", nargs="+", default=["full", "bug", "morph", "snapkv", "ea"])
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
