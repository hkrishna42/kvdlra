"""Week-5 hybrid press: exact-tokens + low-rank-residual vs pure BUG vs eviction.

Tests the recommendation from the Axis-C post-mortem: pure BUG smears its budget
across *all* tokens and never quite passes eviction on perplexity, because a few
high-norm tokens carry most of the signal (which is exactly what SnapKV keeps).
The **hybrid** (`BUGPress(n_exact=k)`) keeps the top-``k`` high-norm tokens *exact*
and low-ranks only the rest — matching eviction's strength on the few important
tokens while keeping a cheap summary of everything else (BUG's strength). Removing
the outliers also leaves a *more* low-rank residual, so BUG's rank goes further.

This puts **BUG x TurboQuant, BUG-hybrid x TurboQuant, SnapKV x TurboQuant, and
ExpectedAttention x TurboQuant** on one fair perplexity-vs-memory axis (every
mechanism quantized, position-fair) at a few context lengths, and asks: **does the
hybrid pass SnapKV where pure BUG could not?**

Reuses `perplexity_sweep.evaluate` (position-fair), the memory models
(`w4_hybrid_sweep.kv_memory_ratio`, `w4_fair.evict_quant_memory`), and the frontier
helpers from `w5_longctx`. Emits the full results JSON to stdout between
`===W5_HYBRID_JSON_BEGIN/END===` for `vastai logs` scraping (never post-hoc SSH,
`[[vastai-pod-flakiness-jul2026]]`); each ctx runs in try/except (OOM-safe). Use
`--plot-only` to rebuild figures from `--out-json`.

Example (CPU smoke, 1B)
-----------------------
    uv run python scripts/w5_hybrid.py --context-lens 1024 --ranks 128 \
        --exact-fracs 0.05 --evict-ratios 0.5 0.85 --bits 4 --n-windows 2

Example (pod, 8B)
-----------------
    python scripts/w5_hybrid.py --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --device cuda --dtype bfloat16 --context-lens 1024 8192 32768 \
        --ranks 128 256 --exact-fracs 0.03 0.06 --evict-ratios 0.5 0.7 0.85 \
        --bits 4 --n-windows 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import matplotlib
import numpy as np
import torch
from kvpress import ComposedPress, ExpectedAttentionPress, SnapKVPress
from perplexity_sweep import evaluate, load_model, load_wikitext_ids
from transformers import PreTrainedModel
from w4_fair import evict_quant_memory
from w4_hybrid_sweep import FP16, kv_memory_ratio
from w5_longctx import pareto, ppl_at_budget

from kvdlra.press import BUGPress, TurboQuantPress
from kvdlra.press.compat import install_kvpress_prefill_compat

matplotlib.use("Agg")  # headless / CPU-safe backend; set before pyplot import
import matplotlib.pyplot as plt

METHOD_STYLE = {
    "BUG x TurboQuant": ("tab:orange", "s", "-"),
    "BUG-hybrid x TurboQuant": ("tab:blue", "o", "-"),
    "SnapKV x TurboQuant": ("tab:green", "^", "--"),
    "ExpectedAttn x TurboQuant": ("tab:red", "v", "--"),
}
N_SINK = 4


def hybrid_memory_ratio(
    t: int, n_features: int, rank: int, n_exact: int, quant_bits: int, n_sink: int = N_SINK
) -> float:
    """Stored memory / full fp16 for the hybrid. Sinks (``n_sink``) stay fp16-exact;
    the ``n_exact`` kept tokens are PolarQuant-quantized to ``quant_bits`` (+ a
    per-token fp16 norm), matching how eviction ``x TurboQuant`` stores ITS kept
    tokens; the rest is the low-rank ``U`` + quantized coords + norms."""
    t_lr = max(0, t - n_sink - n_exact)
    full = t * n_features * FP16
    u_bits = n_features * rank * FP16  # basis U (fp16), fixed -> amortizes with t
    coord_bits = rank * t_lr * quant_bits
    norm_bits = t_lr * FP16  # per-token coord-vector norm
    sink_bits = n_sink * n_features * FP16  # sinks kept fp16-exact
    kept_bits = n_exact * (n_features * quant_bits + FP16)  # quantized + per-token norm
    return (u_bits + coord_bits + norm_bits + sink_bits + kept_bits) / full


def _fronts(rec: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    return {m: pareto([(p["mem"], p["ppl"]) for p in rec["series"][m]]) for m in METHOD_STYLE}


def frontier_gap(rec: dict[str, Any], method: str, rival: str) -> float | None:
    """Min ppl gap ``method - rival`` over the overlapping memory range of their
    Pareto frontiers (negative = ``method`` passes ``rival`` somewhere)."""
    fr = _fronts(rec)
    a, b = fr[method], fr[rival]
    if len(a) < 2 or len(b) < 2:
        return None
    lo, hi = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
    if lo >= hi:
        return None
    gaps = [
        av - bv
        for x in np.linspace(lo, hi, 25)
        if (av := ppl_at_budget(a, float(x))) is not None
        and (bv := ppl_at_budget(b, float(x))) is not None
    ]
    return min(gaps) if gaps else None


def run_one_ctx(
    model: PreTrainedModel, ids: torch.Tensor, ctx: int, n_features: int, args: argparse.Namespace
) -> dict[str, Any]:
    """Fair sweep at one context length across all four method families."""
    eval_args = argparse.Namespace(
        context_len=ctx,
        target_len=args.target_len,
        stride=ctx + args.target_len,
        n_windows=args.n_windows,
    )
    base_ppl, _, n_win = evaluate(model, ids, None, eval_args)
    print(f"[ctx {ctx}] baseline ppl={base_ppl:.4f} ({n_win} windows)")
    series: dict[str, list[tuple[float, float]]] = {m: [] for m in METHOD_STYLE}

    def record(name: str, press: Any, mem: float, tag: str) -> None:
        ppl, _, _ = evaluate(model, ids, press, eval_args)
        series[name].append((mem, ppl))
        d = ppl - base_ppl
        print(f"[ctx {ctx}] {name:26s} {tag:16s} mem={mem:.3f}x ppl={ppl:.4f} (+{d:.3f})")

    bits = args.bits
    for rank in args.ranks:
        record(
            "BUG x TurboQuant",
            BUGPress(rank=rank, quant_bits=bits),
            kv_memory_ratio(ctx, n_features, rank, bits),
            f"r{rank}/{bits}b",
        )
        for frac in args.exact_fracs:
            n_exact = round(frac * (ctx - N_SINK))
            record(
                "BUG-hybrid x TurboQuant",
                BUGPress(rank=rank, quant_bits=bits, n_exact=n_exact),
                hybrid_memory_ratio(ctx, n_features, rank, n_exact, bits),
                f"r{rank}/x{frac:.2f}",
            )
    for cr in args.evict_ratios:
        mem = evict_quant_memory(1 - cr, bits, n_features)
        record(
            "SnapKV x TurboQuant",
            ComposedPress([SnapKVPress(compression_ratio=cr), TurboQuantPress(bits=bits)]),
            mem,
            f"cr{cr}/{bits}b",
        )
        record(
            "ExpectedAttn x TurboQuant",
            ComposedPress(
                [ExpectedAttentionPress(compression_ratio=cr), TurboQuantPress(bits=bits)]
            ),
            mem,
            f"cr{cr}/{bits}b",
        )

    return {
        "ctx": ctx,
        "baseline_ppl": base_ppl,
        "n_windows": n_win,
        "series": {m: [{"mem": mm, "ppl": pp} for mm, pp in v] for m, v in series.items()},
    }


def verdict(per_ctx: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-ctx min frontier-gap of BUG and BUG-hybrid vs each eviction rival."""
    methods = {"bug": "BUG x TurboQuant", "hybrid": "BUG-hybrid x TurboQuant"}
    rivals = {"snap": "SnapKV x TurboQuant", "ea": "ExpectedAttn x TurboQuant"}
    rows: list[dict[str, Any]] = []
    for rec in per_ctx:
        row: dict[str, Any] = {"ctx": rec["ctx"]}
        for mkey, m in methods.items():
            for rkey, riv in rivals.items():
                row[f"{mkey}_minus_{rkey}"] = frontier_gap(rec, m, riv)
        rows.append(row)
    return {"per_ctx_gap": rows, "note": "negative = that method's frontier passes the rival"}


def _plot(per_ctx: list[dict[str, Any]], out_path: Path) -> None:
    """Per-ctx fair frontiers + a summary of min-gap-vs-ctx (hybrid vs BUG vs SnapKV)."""
    # Figure 1: per-ctx frontiers.
    fig, axes = plt.subplots(1, len(per_ctx), figsize=(4.6 * len(per_ctx), 4.4), squeeze=False)
    for j, rec in enumerate(per_ctx):
        ax = axes[0][j]
        for m, (color, marker, ls) in METHOD_STYLE.items():
            pts = [(p["mem"], p["ppl"]) for p in rec["series"][m]]
            if not pts:
                continue
            ax.scatter(
                [p[0] for p in pts], [p[1] for p in pts], c=color, marker=marker, s=22, alpha=0.35
            )
            fr = pareto(pts)
            ax.plot(
                [p[0] for p in fr],
                [p[1] for p in fr],
                c=color,
                marker=marker,
                ls=ls,
                label=m,
                alpha=0.9,
            )
        ax.axhline(float(rec["baseline_ppl"]), ls=":", c="grey", lw=1)
        ax.set_xlabel("stored KV memory / full fp16")
        ax.set_ylabel("WikiText-2 perplexity")
        ax.set_title(f"ctx {int(rec['ctx'])}")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Hybrid (exact + low-rank) vs pure BUG vs eviction — fair, xTurboQuant")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=140)

    # Figure 2: min frontier-gap vs ctx (does hybrid close what BUG could not?).
    ctxs = [int(r["ctx"]) for r in per_ctx]
    fig2, ax = plt.subplots(figsize=(7.5, 5))
    for method, color, marker, lab in [
        ("BUG x TurboQuant", "tab:orange", "s", "BUG - SnapKV"),
        ("BUG-hybrid x TurboQuant", "tab:blue", "o", "BUG-hybrid - SnapKV"),
    ]:
        gaps = [frontier_gap(r, method, "SnapKV x TurboQuant") for r in per_ctx]
        xs = [c for c, g in zip(ctxs, gaps, strict=True) if g is not None]
        ys = [g for g in gaps if g is not None]
        if xs:
            ax.plot(xs, ys, marker=marker, color=color, lw=1.9, ms=7, label=lab)
    ax.axhline(0.0, color="k", lw=1, alpha=0.7)
    ax.text(ctxs[0], 0.0, " passes SnapKV below this line", va="bottom", fontsize=8, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length T")
    ax.set_ylabel("min ppl gap at matched memory (method - SnapKV)")
    ax.set_title("Does the hybrid pass SnapKV where pure BUG could not?")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig2.tight_layout()
    gap_path = out_path.with_name(out_path.stem + "_gap")
    for suffix in (".png", ".pdf"):
        fig2.savefig(gap_path.with_suffix(suffix), dpi=140)
    print(f"wrote {out_path.with_suffix('.png')} and {gap_path.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[1024, 8192, 32768])
    parser.add_argument("--target-len", type=int, default=256)
    parser.add_argument("--ranks", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--exact-fracs", type=float, nargs="+", default=[0.03, 0.06])
    parser.add_argument("--evict-ratios", type=float, nargs="+", default=[0.5, 0.7, 0.85])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--n-windows", type=int, default=4)
    parser.add_argument("--out-json", default="results/w5-hybrid.json")
    parser.add_argument("--out-fig", default="figures/week5/hybrid_frontiers.png")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    out_fig = Path(args.out_fig)
    if args.plot_only:
        blob = json.loads(Path(args.out_json).read_text())
        _plot(blob["per_ctx"], out_fig)
        print(json.dumps(verdict(blob["per_ctx"]), indent=2))
        return

    install_kvpress_prefill_compat()
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    ids = load_wikitext_ids(tokenizer, args.device)
    n_features = int(model.config.head_dim * model.config.num_key_value_heads)
    print(f"model={args.model} n_features={n_features} ctxs={args.context_lens}")

    per_ctx: list[dict[str, Any]] = []
    for ctx in args.context_lens:
        try:
            per_ctx.append(run_one_ctx(model, ids, ctx, n_features, args))
        except RuntimeError as exc:  # CUDA OOM subclasses RuntimeError
            print(f"[ctx {ctx}] FAILED ({type(exc).__name__}: {exc}); keeping prior results")
            break

    blob = {
        "model": args.model,
        "dtype": args.dtype,
        "n_features": n_features,
        "ranks": args.ranks,
        "exact_fracs": args.exact_fracs,
        "evict_ratios": args.evict_ratios,
        "bits": args.bits,
        "per_ctx": per_ctx,
        "verdict": verdict(per_ctx),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(blob, indent=2) + "\n")
    if per_ctx:
        _plot(per_ctx, out_fig)

    print("===W5_HYBRID_JSON_BEGIN===")
    print(json.dumps(blob))
    print("===W5_HYBRID_JSON_END===")
    print(f"[wrote {out_json}]")


if __name__ == "__main__":
    main()
