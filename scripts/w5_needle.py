"""Experiment B: long-context needle-in-a-haystack -- BUG's expected home turf.

This is the retrieval counterpart to ``w5_longctx.py`` (Axis C amortization). Where
``w4_needle.py`` ran a *single* short context (1600 tokens) on SnapKV's home turf,
Week-5 asks the sharper question (``docs/week5-plan.md`` §"Experiment B"): what
happens to retrieval **as the haystack grows to 4K-32K tokens**?

The thesis. Eviction (SnapKV, ExpectedAttention) keeps a *fixed fraction* of the
"important" tokens -- but the needle ("The secret passcode is NNNNN.") looks
unimportant until the question is asked, so a longer haystack means the evicted
fraction is larger in absolute terms and the page holding the fact is more likely
thrown away. BUG instead keeps a *low-rank summary of everything*, and its fixed
``U`` overhead amortizes as context grows (``kv_memory_ratio`` shrinks with ``T``).
So BUG's retrieval accuracy should *hold up* as ``T`` grows while eviction's
*degrades* -- at ROUGHLY MATCHED memory. No divergence falsifies the thesis.

Protocol (reused verbatim from ``w4_needle.py``). ``build_haystack`` builds filler
sentences to a target length, inserts the needle at ``depth``, chat-templates, and
splits off the last ``_TAIL_K`` tokens as the post-compression query. ``retrieve``
pre-fills the haystack **with the press** (compressing the context cache) and then
greedy-decodes the answer with explicit TRUE ``position_ids`` -- the fairness fix
that stops eviction's shrunken cache from breaking RoPE. Accuracy is the hit rate
over ``depths x passcodes`` trials at each context length.

Memory is computed with the exact same models as ``w4_needle``/``w4_fair``: BUG via
``kv_memory_ratio(ctx, n_features, rank, quant_bits)`` (amortizing), eviction via
``evict_quant_memory(1 - cr, bits, n_features)`` (ctx-independent).

Infra (``[[vastai-pod-flakiness-jul2026]]``, mirrors ``w5_longctx``). Each context
length runs in a try/except so a 32K OOM still yields shorter-ctx results; the model
is loaded ONCE and reused across lengths; the full results JSON is printed to stdout
between ``===W5_NEEDLE_JSON_BEGIN/END===`` markers so it survives in ``vastai logs``
(never rely on post-hoc SSH). ``--plot-only`` rebuilds the figure from ``--out-json``.

Example (CPU smoke, 1B)
-----------------------
    python scripts/w5_needle.py --context-lens 512 1024 \
        --depths 0.5 --passcodes 48213

Example (pod, 8B) -- emit results to ``vastai logs``
---------------------------------------------------
    python scripts/w5_needle.py --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --device cuda --dtype bfloat16 \
        --context-lens 4096 8192 16384 32768
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import matplotlib
from kvpress import ExpectedAttentionPress, SnapKVPress
from perplexity_sweep import load_model
from w4_fair import evict_quant_memory
from w4_hybrid_sweep import kv_memory_ratio
from w4_needle import build_haystack, retrieve

from kvdlra.press import BUGPress
from kvdlra.press.compat import install_kvpress_prefill_compat

matplotlib.use("Agg")  # headless / CPU-safe backend; set before pyplot import
import matplotlib.pyplot as plt

# Methods and their plot styling. BUG keeps a low-rank summary of *everything* (U
# amortizes with ctx); the eviction methods keep a fixed token fraction and so are
# expected to drop the needle more often as the haystack grows.
METHOD_STYLE = {
    "baseline": ("tab:blue", "o", "-"),
    "BUG r128/4b": ("tab:orange", "s", "-"),
    "BUG r256/4b": ("tab:brown", "P", "-"),
    "SnapKV keep0.50": ("tab:green", "^", "--"),
    "SnapKV keep0.15": ("tab:olive", "v", "--"),
    "ExpectedAttn keep0.50": ("tab:red", "D", ":"),
    "ExpectedAttn keep0.15": ("tab:pink", "X", ":"),
}


def build_methods(ctx: int, n_features: int, bits: int = 4) -> list[tuple[str, Any, float]]:
    """Return (name, press, memory-ratio) triples at a single context length.

    Memory uses the same models as ``w4_needle``: BUG via ``kv_memory_ratio`` (the
    ``U`` overhead amortizes, so the ratio depends on ``ctx``), eviction via
    ``evict_quant_memory(1 - cr, ...)`` (a fixed kept fraction, ctx-independent).
    """

    def bug_mem(rank: int) -> float:
        return kv_memory_ratio(ctx, n_features, rank, bits)

    def evict_mem(cr: float) -> float:
        return evict_quant_memory(1 - cr, bits, n_features)

    return [
        ("baseline", None, 1.0),
        ("BUG r128/4b", BUGPress(rank=128, quant_bits=bits), bug_mem(128)),
        ("BUG r256/4b", BUGPress(rank=256, quant_bits=bits), bug_mem(256)),
        ("SnapKV keep0.50", SnapKVPress(compression_ratio=0.5), evict_mem(0.5)),
        ("SnapKV keep0.15", SnapKVPress(compression_ratio=0.85), evict_mem(0.85)),
        (
            "ExpectedAttn keep0.50",
            ExpectedAttentionPress(compression_ratio=0.5),
            evict_mem(0.5),
        ),
        (
            "ExpectedAttn keep0.15",
            ExpectedAttentionPress(compression_ratio=0.85),
            evict_mem(0.85),
        ),
    ]


def run_one_ctx(
    model: Any,
    tokenizer: Any,
    ctx: int,
    n_features: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Score needle retrieval for every method at a single context length."""
    methods = build_methods(ctx, n_features, args.bits)
    rows: list[dict[str, Any]] = []
    for name, press, mem in methods:
        hits, total = 0, 0
        for passcode in args.passcodes:
            for depth in args.depths:
                hay, q, target = build_haystack(tokenizer, ctx, depth, passcode)
                ok = retrieve(model, tokenizer, hay, q, target, press, args.device)
                hits += int(ok)
                total += 1
        acc = hits / total
        rows.append(
            {
                "method": name,
                "ctx": ctx,
                "memory": mem,
                "accuracy": acc,
                "hits": hits,
                "total": total,
            }
        )
        print(f"[ctx {ctx}] {name:24s} mem={mem:.3f}x  retrieval acc={acc:.2f} ({hits}/{total})")
    return rows


def _plot(per_ctx: list[dict[str, Any]], out_path: Path) -> None:
    """Two panels: accuracy vs context length (headline) + accuracy vs memory."""
    records: list[dict[str, Any]] = [r for rec in per_ctx for r in rec["rows"]]
    ctxs = sorted({int(r["ctx"]) for r in records})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1 (headline): accuracy vs context length, one line per method. BUG should
    # hold up as ctx grows; eviction should degrade as it drops the needle's page.
    for name, (color, marker, ls) in METHOD_STYLE.items():
        xs, ys = [], []
        for c in ctxs:
            hit = [r for r in records if r["method"] == name and int(r["ctx"]) == c]
            if hit:
                xs.append(c)
                ys.append(float(hit[0]["accuracy"]))
        if xs:
            ax1.plot(xs, ys, marker=marker, color=color, ls=ls, lw=1.9, ms=7, label=name)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("context length T")
    ax1.set_ylabel("needle retrieval accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Experiment B: retrieval vs context (matched memory)")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(fontsize=7)

    # Panel 2: accuracy vs stored memory, per ctx (marker size grows with ctx) -- the
    # matched-memory framing (do BUG's points sit above eviction at equal memory?).
    sizes = {c: 40 + 70 * i for i, c in enumerate(ctxs)}
    for name, (color, marker, _ls) in METHOD_STYLE.items():
        pts = [r for r in records if r["method"] == name]
        if not pts:
            continue
        ax2.scatter(
            [float(r["memory"]) for r in pts],
            [float(r["accuracy"]) for r in pts],
            s=[sizes[int(r["ctx"])] for r in pts],
            c=color,
            marker=marker,
            alpha=0.75,
            label=name,
        )
    ax2.set_xscale("log")
    ax2.set_xlabel("stored KV memory / full fp16 cache")
    ax2.set_ylabel("needle retrieval accuracy")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("accuracy vs memory (marker size ~ ctx)")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=140)
    print(f"wrote {out_path.with_suffix('.png')} and {out_path.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--passcodes", nargs="+", default=["48213", "70561"])
    parser.add_argument(
        "--bits", type=int, default=4, help="TurboQuant bits for BUG/eviction memory"
    )
    parser.add_argument("--out-json", default="results/w5-needle.json")
    parser.add_argument("--out-fig", default="figures/week5/needle_longctx.png")
    parser.add_argument("--plot-only", action="store_true", help="rebuild figures from --out-json")
    args = parser.parse_args()

    out_fig = Path(args.out_fig)
    if args.plot_only:
        blob = json.loads(Path(args.out_json).read_text())
        _plot(blob["per_ctx"], out_fig)
        return

    install_kvpress_prefill_compat()
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    n_features = int(model.config.head_dim * model.config.num_key_value_heads)
    print(f"model={args.model} n_features={n_features} ctxs={args.context_lens}")

    per_ctx: list[dict[str, Any]] = []
    for ctx in args.context_lens:
        try:
            rows = run_one_ctx(model, tokenizer, ctx, n_features, args)
            per_ctx.append({"ctx": ctx, "rows": rows})
        except RuntimeError as exc:  # CUDA OOM subclasses RuntimeError -> keep shorter ctxs
            print(f"[ctx {ctx}] FAILED ({type(exc).__name__}: {exc}); keeping prior results")
            break

    blob = {
        "model": args.model,
        "dtype": args.dtype,
        "n_features": n_features,
        "depths": args.depths,
        "passcodes": args.passcodes,
        "bits": args.bits,
        "per_ctx": per_ctx,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(blob, indent=2) + "\n")
    if per_ctx:
        _plot(per_ctx, out_fig)

    # Emit results to stdout so they survive in `vastai logs` (never rely on SSH).
    print("===W5_NEEDLE_JSON_BEGIN===")
    print(json.dumps(blob))
    print("===W5_NEEDLE_JSON_END===")
    print(f"[wrote {out_json}]")


if __name__ == "__main__":
    main()
