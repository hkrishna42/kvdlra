"""Week-4 hybrid sweep: perplexity vs. *memory* for BUG and BUG+TurboQuant.

Sweeps the BUG rank ``r`` and the PolarQuant bit-width ``b`` and plots WikiText-2
perplexity against the **actual stored-cache memory ratio** (not the nominal
rank fraction), so BUG (fp factors) and BUG+TurboQuant (quantized factors) live
on one honest axis. Uses the same prefill-then-score protocol as
``perplexity_sweep.py`` (imported), so numbers are comparable.

Memory model (`kv_memory_ratio`, per compressed layer, K or V; both identical):
stored = U (n_features x r, fp16)  +  coordinates (r x T', b bits; fp16 if no
quant) + per-token norms (T', fp16, quant only) + exact sinks (n_sink x
n_features, fp16); full = T x n_features x fp16. The fixed ``U`` term amortizes
as ``T`` grows — reported honestly at the sweep's context length.

Writes ``results/w4-hybrid.json`` and ``figures/week4/hybrid.{pdf,png}``.

Example
-------
    uv run python scripts/w4_hybrid_sweep.py --ranks 32 64 128 --bits fp 4 3 2 \
        --context-len 1024 --target-len 512 --n-windows 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import matplotlib
from perplexity_sweep import evaluate, load_model, load_wikitext_ids

from kvdlra.press import BUGPress

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FP16 = 16


def kv_memory_ratio(
    t: int, n_features: int, rank: int, quant_bits: int | None, n_sink: int = 4
) -> float:
    """Stored compressed-cache bits / full fp16-cache bits for one layer's K (or V)."""
    t_pay = t - n_sink
    full = t * n_features * FP16
    u_bits = n_features * rank * FP16  # basis U (fp16), fixed cost
    if quant_bits is None:
        coord_bits = rank * t_pay * FP16
        norm_bits = 0
    else:
        coord_bits = rank * t_pay * quant_bits
        norm_bits = t_pay * FP16  # per-token coordinate-vector norm
    sink_bits = n_sink * n_features * FP16  # sinks kept exact
    return (u_bits + coord_bits + norm_bits + sink_bits) / full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument(
        "--bits",
        nargs="+",
        default=["fp", "4", "3", "2"],
        help="'fp' (no quant) or an integer bit-width",
    )
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--target-len", type=int, default=512)
    parser.add_argument("--n-windows", type=int, default=16)
    parser.add_argument("--out-json", default="results/w4-hybrid.json")
    parser.add_argument("--out-fig", default="figures/week4/hybrid")
    parser.add_argument(
        "--plot-only", action="store_true", help="re-plot from --out-json without loading the model"
    )
    args = parser.parse_args()
    args.stride = args.context_len + args.target_len

    if args.plot_only:
        meta = json.loads(Path(args.out_json).read_text())
        args.model = meta.get("model", args.model)
        args.context_len = meta.get("context_len", args.context_len)
        _plot(meta["results"], float(meta["baseline_ppl"]), args)
        print(f"[replotted {args.out_fig}.{{pdf,png}}]")
        return

    model, tokenizer = load_model(args.model, args.device, args.dtype)
    ids = load_wikitext_ids(tokenizer, args.device)
    n_features = int(model.config.head_dim * model.config.num_key_value_heads)
    eval_args = argparse.Namespace(
        context_len=args.context_len,
        target_len=args.target_len,
        stride=args.stride,
        n_windows=args.n_windows,
    )

    base_ppl, n_tok, n_win = evaluate(model, ids, None, eval_args)
    print(f"baseline: ppl={base_ppl:.4f} ({n_win} windows, {n_tok} tokens)")
    rows: list[dict[str, object]] = [
        {"method": "baseline", "rank": None, "bits": None, "ratio": 1.0, "ppl": base_ppl}
    ]

    for rank in args.ranks:
        for bit_spec in args.bits:
            quant_bits = None if bit_spec == "fp" else int(bit_spec)
            press = BUGPress(rank=rank, quant_bits=quant_bits)
            ppl, _, _ = evaluate(model, ids, press, eval_args)
            ratio = kv_memory_ratio(args.context_len, n_features, rank, quant_bits)
            method = "BUG" if quant_bits is None else "BUG+TurboQuant"
            rows.append(
                {"method": method, "rank": rank, "bits": quant_bits, "ratio": ratio, "ppl": ppl}
            )
            print(
                f"{method:16s} r={rank:>3} b={bit_spec:>2}: "
                f"mem={ratio:.3f}x  ppl={ppl:.4f}  (+{ppl - base_ppl:.3f})"
            )

    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "n_features": n_features,
        "context_len": args.context_len,
        "target_len": args.target_len,
        "n_windows": n_win,
        "scored_tokens": n_tok,
        "baseline_ppl": base_ppl,
        "results": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(meta, indent=2) + "\n")

    _plot(rows, base_ppl, args)
    print(f"[wrote {out_json} and {args.out_fig}.{{pdf,png}}]")


def _plot(rows: list[dict[str, object]], base_ppl: float, args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    def _f(r: dict[str, object], key: str) -> float:
        return float(cast(float, r[key]))

    for method, marker in [("BUG", "o"), ("BUG+TurboQuant", "s")]:
        pts = sorted((r for r in rows if r["method"] == method), key=lambda r: _f(r, "ratio"))
        if not pts:
            continue
        ax.plot(
            [_f(r, "ratio") for r in pts],
            [_f(r, "ppl") for r in pts],
            marker=marker,
            label=method,
            alpha=0.8,
        )
        for r in pts:
            label = f"r{r['rank']}" + (f"/{r['bits']}b" if r["bits"] else "")
            ax.annotate(
                label,
                (_f(r, "ratio"), _f(r, "ppl")),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.axhline(base_ppl, ls="--", c="grey", lw=1, label=f"baseline ({base_ppl:.2f})")
    ax.set_xlabel("stored KV memory / full fp16 cache")
    ax.set_ylabel("WikiText-2 perplexity")
    ax.set_title(f"BUG vs BUG+TurboQuant — {args.model.split('/')[-1]}, ctx {args.context_len}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(args.out_fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=150)


if __name__ == "__main__":
    main()
