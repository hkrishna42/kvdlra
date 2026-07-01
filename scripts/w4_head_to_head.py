"""Week-4 hero figure: BUG / BUG+TurboQuant vs. SnapKV / ExpectedAttention.

The actual "does it beat SOTA?" comparison. Plots WikiText-2 perplexity against
stored-KV memory ratio for four methods on one axis:

* **BUG** and **BUG+TurboQuant** -- reused from ``results/w4-hybrid.json`` (run
  ``w4_hybrid_sweep.py`` first), memory via the honest factored model.
* **SnapKV** and **ExpectedAttention** -- stock kvpress eviction presses run here
  under the same prefill-then-score protocol; memory ratio = fraction of tokens
  kept = ``1 - compression_ratio``.

Requires the transformers>=5.8 compat shim (stock presses otherwise KeyError on
``cache_position``). Writes ``results/w4-head-to-head.json`` and the hero figure
``figures/week4/hero.{pdf,png}``.

Example
-------
    uv run python scripts/w4_head_to_head.py --ratios 0.5 0.75 0.9 \
        --context-len 1024 --target-len 512 --n-windows 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import matplotlib
from kvpress import ExpectedAttentionPress, SnapKVPress
from perplexity_sweep import evaluate, load_model, load_wikitext_ids

from kvdlra.press.compat import install_kvpress_prefill_compat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 0.9],
        help="eviction compression_ratios for SnapKV / ExpectedAttention",
    )
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--target-len", type=int, default=512)
    parser.add_argument("--n-windows", type=int, default=16)
    parser.add_argument("--hybrid-json", default="results/w4-hybrid.json")
    parser.add_argument("--out-json", default="results/w4-head-to-head.json")
    parser.add_argument("--out-fig", default="figures/week4/hero")
    parser.add_argument(
        "--plot-only", action="store_true", help="re-plot from --out-json without loading the model"
    )
    args = parser.parse_args()

    if args.plot_only:
        meta = json.loads(Path(args.out_json).read_text())
        loaded = {k: [(p["ratio"], p["ppl"]) for p in v] for k, v in meta["series"].items()}
        plot_hero(loaded, float(meta["baseline_ppl"]), args)
        print(f"[replotted {args.out_fig}.{{pdf,png}}]")
        return

    install_kvpress_prefill_compat()  # stock presses need this on transformers>=5.8
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    ids = load_wikitext_ids(tokenizer, args.device)
    eval_args = argparse.Namespace(
        context_len=args.context_len,
        target_len=args.target_len,
        stride=args.context_len + args.target_len,
        n_windows=args.n_windows,
    )

    # BUG / BUG+TurboQuant points from the hybrid sweep.
    hybrid = json.loads(Path(args.hybrid_json).read_text())
    series: dict[str, list[tuple[float, float]]] = {"BUG": [], "BUG+TurboQuant": []}
    base_ppl = float(hybrid["baseline_ppl"])
    for r in hybrid["results"]:
        if r["method"] in series:
            series[r["method"]].append((float(r["ratio"]), float(r["ppl"])))

    # Eviction baselines, same protocol.
    for name, cls in [("SnapKV", SnapKVPress), ("ExpectedAttention", ExpectedAttentionPress)]:
        series[name] = []
        for c in args.ratios:
            ppl, _, _ = evaluate(model, ids, cls(compression_ratio=c), eval_args)
            series[name].append((1.0 - c, ppl))  # memory kept = 1 - compression_ratio
            print(f"{name:18s} c={c:.2f} (mem={1 - c:.2f}x): ppl={ppl:.4f}")

    meta = {
        "model": args.model,
        "context_len": args.context_len,
        "baseline_ppl": base_ppl,
        "series": {k: [{"ratio": r, "ppl": p} for r, p in v] for k, v in series.items()},
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(meta, indent=2) + "\n")

    plot_hero(series, base_ppl, args)
    print(f"[wrote {out_json} and {args.out_fig}.{{pdf,png}}]")


def _pareto_frontier(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower-left envelope: points not dominated on (memory, ppl), sorted by memory."""
    frontier: list[tuple[float, float]] = []
    best_ppl = float("inf")
    for ratio, ppl in sorted(pts):  # ascending memory
        if ppl < best_ppl:  # cheaper points must be better to stay on the frontier
            frontier.append((ratio, ppl))
            best_ppl = ppl
    return frontier


def plot_hero(
    series: dict[str, list[tuple[float, float]]], base_ppl: float, args: argparse.Namespace
) -> None:
    """Hero figure: perplexity vs. stored memory, four methods on one axis.

    BUG-family points are scattered (they span rank x bits, not a single knob) with
    their Pareto frontier drawn as a line; the single-knob eviction methods are
    plotted as monotone lines.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, colour, marker in [("BUG", "tab:blue", "o"), ("BUG+TurboQuant", "tab:orange", "s")]:
        pts = series.get(name, [])
        if not pts:
            continue
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            c=colour,
            marker=marker,
            label=name,
            alpha=0.8,
            zorder=3,
        )
    frontier = _pareto_frontier(series.get("BUG", []) + series.get("BUG+TurboQuant", []))
    if frontier:
        ax.plot(
            [p[0] for p in frontier],
            [p[1] for p in frontier],
            c="tab:orange",
            lw=1.5,
            ls="-",
            label="BUG family (Pareto)",
            zorder=2,
        )
    evict = [("SnapKV", "tab:green", "^"), ("ExpectedAttention", "tab:red", "v")]
    for name, colour, marker in evict:
        pts = sorted(series.get(name, []))
        if pts:
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                c=colour,
                marker=marker,
                ls="--",
                label=name,
                alpha=0.85,
            )
    ax.axhline(base_ppl, ls=":", c="grey", lw=1, label=f"baseline ({base_ppl:.2f})")
    ax.set_xlabel("stored KV memory / full fp16 cache")
    ax.set_ylabel("WikiText-2 perplexity")
    ax.set_title(f"KV-cache compression — {args.model.split('/')[-1]}, ctx {args.context_len}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(args.out_fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=150)


if __name__ == "__main__":
    main()
