"""Fair comparison: every mechanism composed with the SAME quantization.

The Week-4 hero figure compared BUG+TurboQuant (quantized) against fp16 eviction
baselines -- which flatters BUG, since quantization is orthogonal to eviction and
you can quantize the tokens SnapKV keeps too. This script closes that gap: it puts
**BUG x TurboQuant, SnapKV x TurboQuant, ExpectedAttention x TurboQuant, and pure
TurboQuant** on one perplexity-vs-memory axis, so the remaining difference is the
*mechanism* (low-rank vs eviction vs nothing), not who got quantized.

* BUG x TurboQuant points are reused from ``results/w4-hybrid.json`` (BUGPress
  quantizes its rank-r coordinates).
* SnapKV/ExpectedAttention x TurboQuant = ``ComposedPress([<evict>,
  TurboQuantPress])`` -- evict tokens, then PolarQuant the survivors.
* pure TurboQuant = ``TurboQuantPress`` alone (quantize all tokens; the
  quantization-only floor).

Same prefill-then-score protocol as ``perplexity_sweep.py``. Writes
``results/w4-fair.json`` and ``figures/week4/fair.{pdf,png}``. Runs on CPU (1B) or
``--device cuda`` (8B on a pod).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import matplotlib
from kvpress import ComposedPress, ExpectedAttentionPress, SnapKVPress
from perplexity_sweep import evaluate, load_model, load_wikitext_ids
from w4_hybrid_sweep import FP16

from kvdlra.press import TurboQuantPress
from kvdlra.press.compat import install_kvpress_prefill_compat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evict_quant_memory(keep_frac: float, bits: int, n_features: int, n_sink: int = 0) -> float:
    """Stored memory for keep-then-quantize: kept fraction x (bits + norm) / full fp16."""
    per_tok = n_features * bits + FP16  # quantized vector + one fp16 norm
    return keep_frac * per_tok / (n_features * FP16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--target-len", type=int, default=512)
    parser.add_argument("--n-windows", type=int, default=16)
    parser.add_argument("--hybrid-json", default="results/w4-hybrid.json")
    parser.add_argument("--out-json", default="results/w4-fair.json")
    parser.add_argument("--out-fig", default="figures/week4/fair")
    parser.add_argument(
        "--plot-only", action="store_true", help="re-plot from --out-json without loading the model"
    )
    args = parser.parse_args()

    if args.plot_only:
        meta = json.loads(Path(args.out_json).read_text())
        loaded = {k: [(p["ratio"], p["ppl"]) for p in v] for k, v in meta["series"].items()}
        _plot(loaded, float(meta["baseline_ppl"]), args)
        print(f"[replotted {args.out_fig}.{{pdf,png}}]")
        return

    install_kvpress_prefill_compat()
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    n_features = int(model.config.head_dim * model.config.num_key_value_heads)
    eval_args = argparse.Namespace(
        context_len=args.context_len,
        target_len=args.target_len,
        stride=args.context_len + args.target_len,
        n_windows=args.n_windows,
    )
    base_ppl, n_tok, n_win = evaluate(
        model, ids := load_wikitext_ids(tokenizer, args.device), None, eval_args
    )
    print(f"baseline: ppl={base_ppl:.4f} ({n_win} windows, {n_tok} tokens)")

    series: dict[str, list[tuple[float, float]]] = {
        "BUG x TurboQuant": [],
        "SnapKV x TurboQuant": [],
        "ExpectedAttn x TurboQuant": [],
        "TurboQuant only": [],
    }

    # BUG x TurboQuant: reuse the hybrid sweep's quantized points.
    hyb = json.loads(Path(args.hybrid_json).read_text())
    for r in hyb["results"]:
        if r["method"] == "BUG+TurboQuant":
            series["BUG x TurboQuant"].append((float(r["ratio"]), float(r["ppl"])))

    # Eviction x TurboQuant (evict, then quantize survivors) + pure TurboQuant.
    def run(name: str, press: Any, mem: float) -> None:
        ppl, _, _ = evaluate(model, ids, press, eval_args)
        series[name].append((mem, ppl))
        print(f"{name:26s} mem={mem:.3f}x  ppl={ppl:.4f}  (+{ppl - base_ppl:.3f})")

    for cr in (0.3, 0.5, 0.7):  # keep 0.7/0.5/0.3
        for bits in (4, 2):
            keep = 1 - cr
            run(
                "SnapKV x TurboQuant",
                ComposedPress([SnapKVPress(compression_ratio=cr), TurboQuantPress(bits=bits)]),
                evict_quant_memory(keep, bits, n_features),
            )
            run(
                "ExpectedAttn x TurboQuant",
                ComposedPress(
                    [ExpectedAttentionPress(compression_ratio=cr), TurboQuantPress(bits=bits)]
                ),
                evict_quant_memory(keep, bits, n_features),
            )
    for bits in (4, 3, 2):  # pure quantization floor (all tokens kept)
        run(
            "TurboQuant only", TurboQuantPress(bits=bits), evict_quant_memory(1.0, bits, n_features)
        )

    meta = {
        "model": args.model,
        "context_len": args.context_len,
        "baseline_ppl": base_ppl,
        "series": {k: [{"ratio": r, "ppl": p} for r, p in v] for k, v in series.items()},
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(meta, indent=2) + "\n")
    _plot(series, base_ppl, args)
    print(f"[wrote {args.out_json} and {args.out_fig}.{{pdf,png}}]")


def _pareto(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower-left envelope: cheaper points kept only if strictly better (lower ppl)."""
    out: list[tuple[float, float]] = []
    best = float("inf")
    for ratio, ppl in sorted(pts):
        if ppl < best:
            out.append((ratio, ppl))
            best = ppl
    return out


def _plot(
    series: dict[str, list[tuple[float, float]]], base_ppl: float, args: argparse.Namespace
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    styles = {
        "BUG x TurboQuant": ("tab:orange", "s", "-"),
        "SnapKV x TurboQuant": ("tab:green", "^", "--"),
        "ExpectedAttn x TurboQuant": ("tab:red", "v", "--"),
        "TurboQuant only": ("tab:purple", "D", ":"),
    }
    for name, pts in series.items():
        if not pts:
            continue
        c, m, ls = styles[name]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=c, marker=m, alpha=0.35, s=25)
        front = _pareto(pts)  # plot each method's best-achievable (Pareto) frontier
        ax.plot(
            [p[0] for p in front],
            [p[1] for p in front],
            c=c,
            marker=m,
            ls=ls,
            label=name,
            alpha=0.9,
        )
    ax.axhline(base_ppl, ls=":", c="grey", lw=1, label=f"baseline ({base_ppl:.2f})")
    ax.set_xlabel("stored KV memory / full fp16 cache")
    ax.set_ylabel("WikiText-2 perplexity")
    ax.set_title(
        f"Fair: every mechanism x TurboQuant — {args.model.split('/')[-1]}, ctx {args.context_len}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(args.out_fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=150)


if __name__ == "__main__":
    main()
