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
    args = parser.parse_args()

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

    fig, ax = plt.subplots(figsize=(7.5, 5))
    styles = {
        "BUG": ("o", "-"),
        "BUG+TurboQuant": ("s", "-"),
        "SnapKV": ("^", "--"),
        "ExpectedAttention": ("v", "--"),
    }
    for name, pts in series.items():
        pts = sorted(pts)
        if not pts:
            continue
        marker, ls = styles[name]
        ax.plot(
            [p[0] for p in pts], [p[1] for p in pts], marker=marker, ls=ls, label=name, alpha=0.85
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
    print(f"[wrote {out_json} and {out}.{{pdf,png}}]")


if __name__ == "__main__":
    main()
