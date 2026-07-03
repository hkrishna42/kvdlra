"""Week-5 clean fp16 long-context frontier: pure BUG vs SnapKV/EA vs BUG-hybrid.

Every prior Week-5 comparison composed each mechanism with 4-bit TurboQuant; that
quantization floor made eviction near-lossless at long context and confounded the
low-rank-vs-eviction question. This run removes TurboQuant entirely (**all fp16**)
to isolate the mechanism at long context. Without quant, fp16 eviction can only keep
a ``keep_frac`` *fraction of tokens*, while fp16 BUG keeps a rank-``r`` *summary of
every token* at the same memory (BUG's fp16 floor is ``rank / n_features``). At
32K-64K this is the cleanest test of whether "summarize everything" beats "keep a
fraction" -- on perplexity, against SnapKV **and** ExpectedAttention.

Uses WikiText-103 train (streamed, >100M tokens -> many clean windows) not WikiText-2
(only 8/4 windows at 32K/64K -- too noisy); see ``perplexity_sweep.load_corpus_ids``
(PG19 is also selectable but needs trust_remote_code). Reuses the frontier helpers
from ``w5_hybrid``/``w5_longctx`` and the position-fair ``evaluate``. Emits the full
results JSON to stdout between ``===W5_FP16_JSON_BEGIN/END===`` for ``vastai logs``
scraping; each ctx runs OOM-safe; ``--plot-only`` rebuilds figures.

Example (CPU smoke, 1B)
-----------------------
    uv run python scripts/w5_fp16_longctx.py --model unsloth/Llama-3.2-1B-Instruct \
        --context-lens 1024 2048 --corpus wikitext-103 --max-tokens 200000 \
        --ranks 64 128 --n-windows 2

Example (pod, 8B)
-----------------
    python scripts/w5_fp16_longctx.py --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --device cuda --dtype bfloat16 --context-lens 32768 65536 --corpus wikitext-103 \
        --ranks 64 128 256 --evict-ratios 0.75 0.875 0.94 --exact-fracs 0.03 \
        --n-windows 8
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
from kvpress import ExpectedAttentionPress, SnapKVPress
from perplexity_sweep import evaluate, load_corpus_ids, load_model
from transformers import PreTrainedModel
from w4_fair import evict_quant_memory
from w4_hybrid_sweep import kv_memory_ratio
from w5_hybrid import hybrid_memory_ratio
from w5_longctx import pareto, ppl_at_budget

from kvdlra.press import BUGPress
from kvdlra.press.compat import install_kvpress_prefill_compat

matplotlib.use("Agg")  # headless / CPU-safe backend; set before pyplot import
import matplotlib.pyplot as plt

# All fp16 (no TurboQuant). BUG/hybrid = low-rank; SnapKV/EA = eviction.
METHOD_STYLE = {
    "BUG (fp16)": ("tab:orange", "s", "-"),
    "BUG-hybrid (fp16)": ("tab:blue", "o", "-"),
    "SnapKV (fp16)": ("tab:green", "^", "--"),
    "ExpectedAttn (fp16)": ("tab:red", "v", "--"),
}
N_SINK = 4
BUG_BLOCK = 512  # larger blocks -> 4x fewer SVDs than the default 128 (cusolver-stall guard)


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
    """Fair fp16 sweep at one context length across all four method families."""
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
        print(f"[ctx {ctx}] {name:20s} {tag:14s} mem={mem:.3f}x ppl={ppl:.4f} (+{d:.3f})")

    for rank in args.ranks:
        record(
            "BUG (fp16)",
            BUGPress(rank=rank, quant_bits=None, block_size=BUG_BLOCK),
            kv_memory_ratio(ctx, n_features, rank, None),
            f"r{rank}",
        )
        for frac in args.exact_fracs:
            n_exact = round(frac * (ctx - N_SINK))
            record(
                "BUG-hybrid (fp16)",
                BUGPress(rank=rank, quant_bits=None, n_exact=n_exact, block_size=BUG_BLOCK),
                hybrid_memory_ratio(ctx, n_features, rank, n_exact, 16),
                f"r{rank}/x{frac:.2f}",
            )
    for cr in args.evict_ratios:
        mem = evict_quant_memory(1 - cr, 16, n_features)
        record("SnapKV (fp16)", SnapKVPress(compression_ratio=cr), mem, f"keep{1 - cr:.2f}")
        record(
            "ExpectedAttn (fp16)",
            ExpectedAttentionPress(compression_ratio=cr),
            mem,
            f"keep{1 - cr:.2f}",
        )

    return {
        "ctx": ctx,
        "baseline_ppl": base_ppl,
        "n_windows": n_win,
        "series": {m: [{"mem": mm, "ppl": pp} for mm, pp in v] for m, v in series.items()},
    }


def verdict(per_ctx: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-ctx min frontier-gap of BUG and BUG-hybrid vs each eviction rival."""
    methods = {"bug": "BUG (fp16)", "hybrid": "BUG-hybrid (fp16)"}
    rivals = {"snap": "SnapKV (fp16)", "ea": "ExpectedAttn (fp16)"}
    rows: list[dict[str, Any]] = []
    for rec in per_ctx:
        row: dict[str, Any] = {"ctx": rec["ctx"]}
        for mkey, m in methods.items():
            for rkey, riv in rivals.items():
                row[f"{mkey}_minus_{rkey}"] = frontier_gap(rec, m, riv)
        rows.append(row)
    return {"per_ctx_gap": rows, "note": "negative = that method's fp16 frontier passes the rival"}


def _plot(per_ctx: list[dict[str, Any]], out_path: Path) -> None:
    """Per-ctx fair fp16 frontiers + a min-gap-vs-ctx summary (BUG/hybrid vs eviction)."""
    fig, axes = plt.subplots(1, len(per_ctx), figsize=(4.8 * len(per_ctx), 4.4), squeeze=False)
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
        ax.set_ylabel("perplexity")
        ax.set_title(f"ctx {int(rec['ctx'])}")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("fp16 (no TurboQuant): pure BUG vs eviction vs BUG-hybrid, long context")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=140)

    # Summary: min frontier-gap vs ctx (does fp16 BUG/hybrid pass SnapKV/EA?).
    ctxs = [int(r["ctx"]) for r in per_ctx]
    fig2, ax = plt.subplots(figsize=(7.5, 5))
    lines = [
        ("BUG (fp16)", "SnapKV (fp16)", "tab:orange", "s", "BUG - SnapKV"),
        ("BUG (fp16)", "ExpectedAttn (fp16)", "tab:brown", "s", "BUG - EA"),
        ("BUG-hybrid (fp16)", "SnapKV (fp16)", "tab:blue", "o", "hybrid - SnapKV"),
        ("BUG-hybrid (fp16)", "ExpectedAttn (fp16)", "tab:cyan", "o", "hybrid - EA"),
    ]
    for method, rival, color, marker, lab in lines:
        gaps = [frontier_gap(r, method, rival) for r in per_ctx]
        xs = [c for c, g in zip(ctxs, gaps, strict=True) if g is not None]
        ys = [g for g in gaps if g is not None]
        if xs:
            ax.plot(xs, ys, marker=marker, color=color, lw=1.9, ms=7, label=lab)
    ax.axhline(0.0, color="k", lw=1, alpha=0.7)
    ax.text(ctxs[0], 0.0, " BUG passes rival below this line", va="bottom", fontsize=8, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length T")
    ax.set_ylabel("min ppl gap at matched memory (method - rival)")
    ax.set_title("fp16: does BUG/hybrid pass eviction at 32K-64K?")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
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
    parser.add_argument("--context-lens", type=int, nargs="+", default=[32768, 65536])
    parser.add_argument(
        "--corpus", default="wikitext-103", choices=["wikitext-103", "wikitext-2", "pg19"]
    )
    parser.add_argument("--max-tokens", type=int, default=3_000_000)
    parser.add_argument("--target-len", type=int, default=256)
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--evict-ratios", type=float, nargs="+", default=[0.75, 0.875, 0.94])
    parser.add_argument("--exact-fracs", type=float, nargs="+", default=[0.03])
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--out-json", default="results/w5-fp16-longctx.json")
    parser.add_argument("--out-fig", default="figures/week5/fp16_longctx.png")
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
    ids = load_corpus_ids(tokenizer, args.device, args.corpus, args.max_tokens)
    n_features = int(model.config.head_dim * model.config.num_key_value_heads)
    print(
        f"model={args.model} n_features={n_features} corpus={args.corpus} "
        f"tokens={ids.shape[0]} ctxs={args.context_lens}"
    )

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
        "corpus": args.corpus,
        "n_features": n_features,
        "ranks": args.ranks,
        "evict_ratios": args.evict_ratios,
        "exact_fracs": args.exact_fracs,
        "per_ctx": per_ctx,
        "verdict": verdict(per_ctx),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(blob, indent=2) + "\n")
    if per_ctx:
        _plot(per_ctx, out_fig)

    print("===W5_FP16_JSON_BEGIN===")
    print(json.dumps(blob))
    print("===W5_FP16_JSON_END===")
    print(f"[wrote {out_json}]")


if __name__ == "__main__":
    main()
