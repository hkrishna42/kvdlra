"""Axis C / Experiment A: long-context scaling -- the amortization test.

The headline open question from Weeks 3-4 (``docs/week5-plan.md`` §"Axis C"). BUG's
one structural handicap is its *fixed* ``U`` basis (``n_features x rank`` fp16 per
layer). That overhead **amortizes as context grows**: BUG's stored-memory ratio
``kv_memory_ratio(T, ...)`` shrinks with ``T`` (the ``U`` term is fixed while the
full cache grows ``~T``), whereas eviction's memory ratio is **independent of
``T``** (it keeps a fixed *fraction* of tokens). So BUG's fair perplexity-vs-memory
frontier should improve *relative to* SnapKV/ExpectedAttention as ``T`` grows.

This script tests that, falsifiably, on ONE model (8B on a pod; 1B for a CPU
smoke): sweep ``ctx in {1K, 4K, 8K, 16K, 32K}`` and at each length redo the *fair*
comparison -- BUG x TurboQuant vs SnapKV x TurboQuant vs ExpectedAttention x
TurboQuant, every mechanism quantized, position-fair (``perplexity_sweep`` passes
explicit ``position_ids`` so eviction is not penalized). It then interpolates each
method's Pareto frontier at fixed memory budgets and plots **ppl vs T** per method
(the crossover) plus a per-ctx frontier grid.

*Win = BUG's frontier passes SnapKV's (and closes on EA's) as ``T`` grows; no
movement falsifies the amortization thesis -- a valuable negative either way.*

Infra (``[[vastai-pod-flakiness-jul2026]]``): the full results JSON is printed to
stdout between ``===W5_LONGCTX_JSON_BEGIN/END===`` markers so it survives in
``vastai logs`` -- never rely on post-hoc SSH to retrieve pod results. Each ctx is
run in a try/except so an OOM at 32K still yields the shorter-ctx results and a
partial figure. Use ``--plot-only`` to rebuild figures from the saved JSON.

Example (CPU smoke, 1B)
-----------------------
    uv run python scripts/w5_longctx.py --context-lens 1024 2048 \
        --ranks 64 128 --evict-ratios 0.5 0.7 --bits 4 --n-windows 2

Example (pod, 8B) -- emit results to ``vastai logs``
---------------------------------------------------
    python scripts/w5_longctx.py --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --device cuda --dtype bfloat16 \
        --context-lens 1024 4096 8192 16384 32768 \
        --ranks 64 128 256 --evict-ratios 0.5 0.7 0.85 --bits 4 --n-windows 6
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
from w4_hybrid_sweep import kv_memory_ratio

from kvdlra.press import BUGPress, TurboQuantPress
from kvdlra.press.compat import install_kvpress_prefill_compat

matplotlib.use("Agg")  # headless / CPU-safe backend; set before pyplot import
import matplotlib.pyplot as plt

# Methods and their plot styling. "BUG x TurboQuant" is low-rank (amortizing U);
# the two eviction methods keep a fixed token fraction (T-independent memory).
METHOD_STYLE = {
    "BUG x TurboQuant": ("tab:orange", "s", "-"),
    "SnapKV x TurboQuant": ("tab:green", "^", "--"),
    "ExpectedAttn x TurboQuant": ("tab:red", "v", "--"),
    "TurboQuant only": ("tab:purple", "D", ":"),
}
# Fixed memory budgets at which we read each method's frontier for the crossover.
BUDGETS = [0.10, 0.15, 0.20, 0.25]


def pareto(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower-left envelope: sorted by memory, keep a point only if it lowers ppl."""
    out: list[tuple[float, float]] = []
    best = float("inf")
    for mem, ppl in sorted(pts):
        if ppl < best:
            out.append((mem, ppl))
            best = ppl
    return out


def ppl_at_budget(front: list[tuple[float, float]], budget: float) -> float | None:
    """Interpolate a method's Pareto frontier ppl at a memory ``budget``.

    Linear interpolation in memory; returns ``None`` if ``budget`` is outside the
    frontier's achieved memory range (no extrapolation -- an honest gap).
    """
    if len(front) < 2:
        return None
    mems = [m for m, _ in front]
    ppls = [p for _, p in front]
    if budget < mems[0] or budget > mems[-1]:
        return None
    return float(np.interp(budget, mems, ppls))


def _fronts(rec: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    """Pareto frontier per method for one ctx record."""
    return {m: pareto([(p["mem"], p["ppl"]) for p in rec["series"][m]]) for m in METHOD_STYLE}


def run_one_ctx(
    model: PreTrainedModel,
    ids: torch.Tensor,
    ctx: int,
    n_features: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Fair sweep at a single context length; returns baseline + per-method points."""
    eval_args = argparse.Namespace(
        context_len=ctx,
        target_len=args.target_len,
        stride=ctx + args.target_len,
        n_windows=args.n_windows,
    )
    base_ppl, n_tok, n_win = evaluate(model, ids, None, eval_args)
    print(f"[ctx {ctx}] baseline ppl={base_ppl:.4f} ({n_win} windows, {n_tok} tokens)")
    series: dict[str, list[tuple[float, float]]] = {m: [] for m in METHOD_STYLE}

    def record(name: str, press: Any, mem: float, tag: str) -> None:
        ppl, _, _ = evaluate(model, ids, press, eval_args)
        series[name].append((mem, ppl))
        d = ppl - base_ppl
        print(f"[ctx {ctx}] {name:24s} {tag:11s} mem={mem:.3f}x ppl={ppl:.4f} (+{d:.3f})")

    # BUG x TurboQuant: low-rank, U amortizes -> memory shrinks with ctx.
    for rank in args.ranks:
        for bits in args.bits:
            record(
                "BUG x TurboQuant",
                BUGPress(rank=rank, quant_bits=bits),
                kv_memory_ratio(ctx, n_features, rank, bits),
                f"r{rank}/{bits}b",
            )
    # Eviction x TurboQuant: keep a fixed fraction (memory independent of ctx).
    for cr in args.evict_ratios:
        for bits in args.bits:
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
    # Pure-quantization floor (all tokens kept), for reference.
    for bits in args.bits:
        record(
            "TurboQuant only",
            TurboQuantPress(bits=bits),
            evict_quant_memory(1.0, bits, n_features),
            f"{bits}b",
        )

    return {
        "ctx": ctx,
        "baseline_ppl": base_ppl,
        "n_windows": n_win,
        "series": {m: [{"mem": mm, "ppl": pp} for mm, pp in v] for m, v in series.items()},
    }


def crossover_verdict(per_ctx: list[dict[str, Any]]) -> dict[str, Any]:
    """Does BUG's frontier pass SnapKV's (and close on EA's) as ctx grows?

    For each ctx and budget, report ppl(method) on the Pareto frontier and the gap
    ``BUG - SnapKV`` (negative = BUG wins) and ``BUG - EA``. Amortization predicts
    these gaps decrease (become more negative) with ctx.
    """
    rows: list[dict[str, Any]] = []
    for rec in per_ctx:
        fronts = _fronts(rec)
        for b in BUDGETS:
            vals = {m: ppl_at_budget(fronts[m], b) for m in METHOD_STYLE}
            bug = vals["BUG x TurboQuant"]
            snap = vals["SnapKV x TurboQuant"]
            ea = vals["ExpectedAttn x TurboQuant"]
            rows.append(
                {
                    "ctx": rec["ctx"],
                    "budget": b,
                    "ppl": vals,
                    "bug_minus_snapkv": (None if bug is None or snap is None else bug - snap),
                    "bug_minus_ea": (None if bug is None or ea is None else bug - ea),
                }
            )
    return {"per_budget": rows, "note": "bug_minus_snapkv<0 means BUG wins at that (ctx,budget)"}


def _plot(per_ctx: list[dict[str, Any]], out_path: Path) -> None:
    """Two figures: crossover (ppl-vs-ctx per method per budget) + per-ctx frontiers."""
    ctxs = [int(r["ctx"]) for r in per_ctx]
    fronts_by_ctx = {int(r["ctx"]): _fronts(r) for r in per_ctx}

    # Figure 1: crossover -- ppl at each fixed budget vs ctx, one panel per budget.
    fig, axes = plt.subplots(1, len(BUDGETS), figsize=(4.6 * len(BUDGETS), 4.4), squeeze=False)
    for j, budget in enumerate(BUDGETS):
        ax = axes[0][j]
        for m, (color, marker, ls) in METHOD_STYLE.items():
            ys = [ppl_at_budget(fronts_by_ctx[c][m], budget) for c in ctxs]
            xs = [c for c, y in zip(ctxs, ys, strict=True) if y is not None]
            yv = [y for y in ys if y is not None]
            if xs:
                ax.plot(xs, yv, marker=marker, ls=ls, color=color, label=m, alpha=0.9)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length T")
        ax.set_ylabel("WikiText-2 perplexity")
        ax.set_title(f"budget {budget:.2f}x")
        ax.grid(True, alpha=0.3, which="both")
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Axis C: perplexity at fixed memory vs context (does BUG pass SnapKV as T grows?)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=140)

    # Figure 2: per-ctx fair frontiers (small multiples).
    fig2, axes2 = plt.subplots(1, len(per_ctx), figsize=(4.4 * len(per_ctx), 4.2), squeeze=False)
    for j, rec in enumerate(per_ctx):
        ax = axes2[0][j]
        for m, (color, marker, ls) in METHOD_STYLE.items():
            pts = [(p["mem"], p["ppl"]) for p in rec["series"][m]]
            if not pts:
                continue
            ax.scatter(
                [p[0] for p in pts], [p[1] for p in pts], c=color, marker=marker, s=20, alpha=0.3
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
        ax.set_ylabel("perplexity")
        ax.set_title(f"ctx {int(rec['ctx'])}")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)
    fig2.tight_layout()
    grid_path = out_path.with_name(out_path.stem + "_frontiers")
    for suffix in (".png", ".pdf"):
        fig2.savefig(grid_path.with_suffix(suffix), dpi=140)
    print(f"wrote {out_path.with_suffix('.png')} and {grid_path.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument(
        "--context-lens", type=int, nargs="+", default=[1024, 4096, 8192, 16384, 32768]
    )
    parser.add_argument("--target-len", type=int, default=256)
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--evict-ratios", type=float, nargs="+", default=[0.5, 0.7, 0.85])
    parser.add_argument("--bits", type=int, nargs="+", default=[4])
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument("--out-json", default="results/w5-longctx.json")
    parser.add_argument("--out-fig", default="figures/week5/longctx_crossover.png")
    parser.add_argument("--plot-only", action="store_true", help="rebuild figures from --out-json")
    args = parser.parse_args()

    out_fig = Path(args.out_fig)
    if args.plot_only:
        blob = json.loads(Path(args.out_json).read_text())
        _plot(blob["per_ctx"], out_fig)
        print(crossover_verdict(blob["per_ctx"])["note"])
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
        except RuntimeError as exc:  # CUDA OOM subclasses RuntimeError -> keep shorter ctxs
            print(f"[ctx {ctx}] FAILED ({type(exc).__name__}: {exc}); keeping prior results")
            break

    blob = {
        "model": args.model,
        "dtype": args.dtype,
        "n_features": n_features,
        "target_len": args.target_len,
        "ranks": args.ranks,
        "evict_ratios": args.evict_ratios,
        "bits": args.bits,
        "per_ctx": per_ctx,
        "verdict": crossover_verdict(per_ctx),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(blob, indent=2) + "\n")
    if per_ctx:
        _plot(per_ctx, out_fig)

    # Emit results to stdout so they survive in `vastai logs` (never rely on SSH).
    print("===W5_LONGCTX_JSON_BEGIN===")
    print(json.dumps(blob))
    print("===W5_LONGCTX_JSON_END===")
    print(f"[wrote {out_json}]")


if __name__ == "__main__":
    main()
