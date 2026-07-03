"""Week-5 Experiment B (harder): RULER-lite multi-key retrieval under compression.

The single-passcode needle (`w5_needle.py`) **saturated** at 8B -- every method,
eviction included, recovered one salient fact. This is the harder, discriminating
version: insert **many distinct key->value pairs** ("The <label> code is <NNNNN>.")
scattered through the haystack, then ask for **one specific** key's value. Now
compression must preserve *the queried key among many distractors* -- exactly where
eviction is expected to fail (it keeps "important-looking" tokens, but at
compression time it cannot know which key will be asked, so it drops the un-cued
one), while BUG keeps a low-rank summary of *all* keys. We also push to **aggressive
memory** (eviction keep 5-15%), where eviction must discard most of the context.

Protocol reuses `w4_needle.retrieve` verbatim (compressed prefill + true-position
greedy decode -- the eviction fairness fix). Accuracy is the hit rate over
(queried-key x seed) trials at each context length. Methods are compared at roughly
matched, aggressive memory.

Infra (mirrors `w5_needle`/`w5_longctx`): each ctx in try/except (OOM-safe), model
loaded once, full results JSON printed between `===W5_RULER_JSON_BEGIN/END===` for
`vastai logs` scraping; `--plot-only` rebuilds the figure.

Example (CPU smoke, 1B)
-----------------------
    uv run python scripts/w5_ruler.py --context-lens 2048 --n-keys 8 \
        --n-queries 3 --seeds 0 1

Example (pod, 8B)
-----------------
    python scripts/w5_ruler.py --model unsloth/Meta-Llama-3.1-8B-Instruct \
        --device cuda --dtype bfloat16 --context-lens 4096 16384 --n-keys 12 \
        --n-queries 4 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import matplotlib
import torch
from kvpress import ExpectedAttentionPress, SnapKVPress
from perplexity_sweep import load_model
from w4_fair import evict_quant_memory
from w4_hybrid_sweep import kv_memory_ratio
from w4_needle import _FILLER, retrieve

from kvdlra.press import BUGPress
from kvdlra.press.compat import install_kvpress_prefill_compat

matplotlib.use("Agg")  # headless / CPU-safe backend; set before pyplot import
import matplotlib.pyplot as plt

METHOD_STYLE = {
    "baseline": ("tab:blue", "o", "-"),
    "BUG r128/4b": ("tab:orange", "s", "-"),
    "BUG-hybrid r128/x0.05": ("tab:cyan", "*", "-"),
    "SnapKV keep0.15": ("tab:green", "^", "--"),
    "SnapKV keep0.05": ("tab:olive", "v", "--"),
    "ExpectedAttn keep0.15": ("tab:red", "D", ":"),
    "ExpectedAttn keep0.05": ("tab:pink", "X", ":"),
}
# Distinct key labels (NATO-ish); codes are distinct 5-digit numbers per trial.
_LABELS = [
    "amber",
    "bronze",
    "coral",
    "delta",
    "ember",
    "flint",
    "garnet",
    "hazel",
    "indigo",
    "jade",
    "kelp",
    "lunar",
    "maple",
    "nickel",
    "onyx",
    "pearl",
]
_TAIL_K = 48  # question + assistant header fed AFTER compression (as in w4_needle)


def build_multikey_haystack(
    tok: Any, context_len: int, n_keys: int, query_idx: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Haystack with ``n_keys`` distinct 'The <label> code is <NNNNN>.' facts scattered
    through filler; the query asks for the ``query_idx``-th label's code. Returns
    (prefill_ids, query_ids, target_code)."""
    g = torch.Generator().manual_seed(seed * 131 + query_idx)
    labels = _LABELS[:n_keys]
    # distinct 5-digit codes
    codes = (10000 + torch.randperm(89999, generator=g)[:n_keys]).tolist()

    sentences: list[str] = []
    i = 0
    while len(tok(" ".join(sentences)).input_ids) < context_len:
        sentences.append(_FILLER[i % len(_FILLER)])
        i += 1
    # Insert each key fact at an evenly spread depth (jittered by seed).
    n = len(sentences)
    for k, (lab, code) in enumerate(zip(labels, codes, strict=True)):
        depth = (k + 1) / (n_keys + 1)
        pos = min(n, max(0, int(depth * n) + (k % 3)))
        sentences.insert(pos, f"The {lab} code is {code}.")
    body = " ".join(sentences)
    target = str(codes[query_idx])
    question = f"\n\nWhat is the {labels[query_idx]} code? Reply with only the number."
    msgs = [{"role": "user", "content": body + question}]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"]
    return ids[:, :-_TAIL_K], ids[:, -_TAIL_K:], target


def build_methods(ctx: int, n_features: int, bits: int = 4) -> list[tuple[str, Any, float]]:
    """Methods at aggressive matched memory. BUG (+hybrid) vs SnapKV/EA at keep 5-15%."""
    n_exact = round(0.05 * (ctx - 4))

    def bug_mem(rank: int) -> float:
        return kv_memory_ratio(ctx, n_features, rank, bits)

    return [
        ("baseline", None, 1.0),
        ("BUG r128/4b", BUGPress(rank=128, quant_bits=bits), bug_mem(128)),
        (
            "BUG-hybrid r128/x0.05",
            BUGPress(rank=128, quant_bits=bits, n_exact=n_exact),
            bug_mem(128) + n_exact * (n_features * bits + 16) / (ctx * n_features * 16),
        ),
        (
            "SnapKV keep0.15",
            SnapKVPress(compression_ratio=0.85),
            evict_quant_memory(0.15, bits, n_features),
        ),
        (
            "SnapKV keep0.05",
            SnapKVPress(compression_ratio=0.95),
            evict_quant_memory(0.05, bits, n_features),
        ),
        (
            "ExpectedAttn keep0.15",
            ExpectedAttentionPress(compression_ratio=0.85),
            evict_quant_memory(0.15, bits, n_features),
        ),
        (
            "ExpectedAttn keep0.05",
            ExpectedAttentionPress(compression_ratio=0.95),
            evict_quant_memory(0.05, bits, n_features),
        ),
    ]


def run_one_ctx(
    model: Any, tokenizer: Any, ctx: int, n_features: int, args: argparse.Namespace
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, press, mem in build_methods(ctx, n_features, args.bits):
        hits, total = 0, 0
        for seed in args.seeds:
            for q in range(args.n_queries):
                hay, query, target = build_multikey_haystack(tokenizer, ctx, args.n_keys, q, seed)
                ok = retrieve(model, tokenizer, hay, query, target, press, args.device)
                hits += int(ok)
                total += 1
        acc = hits / total if total else 0.0
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
        print(f"[ctx {ctx}] {name:24s} mem={mem:.3f}x  acc={acc:.2f} ({hits}/{total})")
    return {"ctx": ctx, "rows": rows}


def _plot(per_ctx: list[dict[str, Any]], out_path: Path) -> None:
    records = [r for rec in per_ctx for r in rec["rows"]]
    ctxs = sorted({int(r["ctx"]) for r in records})
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, (color, marker, ls) in METHOD_STYLE.items():
        xs, ys = [], []
        for c in ctxs:
            hit = [r for r in records if r["method"] == name and int(r["ctx"]) == c]
            if hit:
                xs.append(c)
                ys.append(float(hit[0]["accuracy"]))
        if xs:
            ax.plot(xs, ys, marker=marker, color=color, ls=ls, lw=1.9, ms=8, label=name)
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("context length T")
    ax.set_ylabel("multi-key retrieval accuracy")
    ax.set_title("RULER-lite: retrieve 1 of N keys under aggressive compression")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=140)
    print(f"wrote {out_path.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--n-keys", type=int, default=12)
    parser.add_argument("--n-queries", type=int, default=4, help="distinct keys queried per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--out-json", default="results/w5-ruler.json")
    parser.add_argument("--out-fig", default="figures/week5/ruler_longctx.png")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    out_fig = Path(args.out_fig)
    if args.plot_only:
        blob = json.loads(Path(args.out_json).read_text())
        _plot(blob["per_ctx"], out_fig)
        return

    install_kvpress_prefill_compat()
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    n_features = int(model.config.head_dim * model.config.num_key_value_heads)
    print(
        f"model={args.model} n_features={n_features} n_keys={args.n_keys} ctxs={args.context_lens}"
    )

    per_ctx: list[dict[str, Any]] = []
    for ctx in args.context_lens:
        try:
            per_ctx.append(run_one_ctx(model, tokenizer, ctx, n_features, args))
        except RuntimeError as exc:  # CUDA OOM subclasses RuntimeError
            print(f"[ctx {ctx}] FAILED ({type(exc).__name__}: {exc}); keeping prior results")
            break

    blob = {
        "model": args.model,
        "n_features": n_features,
        "n_keys": args.n_keys,
        "n_queries": args.n_queries,
        "seeds": args.seeds,
        "per_ctx": per_ctx,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(blob, indent=2) + "\n")
    if per_ctx:
        _plot(per_ctx, out_fig)

    print("===W5_RULER_JSON_BEGIN===")
    print(json.dumps(blob))
    print("===W5_RULER_JSON_END===")
    print(f"[wrote {out_json}]")


if __name__ == "__main__":
    main()
