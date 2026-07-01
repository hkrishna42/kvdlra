"""Needle-in-a-haystack retrieval under KV compression (SnapKV's home turf).

Long-context KV compressors are really designed for *retrieval*: compress a long
prompt, then answer a question whose evidence sits somewhere inside it. Eviction
methods (SnapKV, Expected Attention) explicitly **keep the important tokens**, so
they should shine here; BUG's low-rank reconstruction smears all tokens equally
and may blur a sharp fact. This script tests that honestly.

Protocol. Build a haystack of filler sentences with one needle
("The secret passcode is NNNNN.") inserted at a given depth, pre-fill it **with
the press** (compressing the context cache), then greedy-decode the answer to
"What is the secret passcode?" and check whether NNNNN appears. Retrieval is
scored across several depths x trials; accuracy is the hit rate.

Fairness (same issue as perplexity_sweep). Eviction shrinks the cache below the
true context length, so the query/answer tokens must be positioned at their TRUE
sequence positions or RoPE breaks. We decode manually with explicit
``position_ids`` (a no-op for BUG/baseline; a correction for eviction) instead of
``model.generate`` (which derives positions from the shrunken cache length).

Writes ``results/w4-needle.json`` and ``figures/week4/needle.{pdf,png}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import matplotlib
import torch
from kvpress import ExpectedAttentionPress, SnapKVPress
from perplexity_sweep import load_model
from w4_hybrid_sweep import kv_memory_ratio

from kvdlra.press import BUGPress
from kvdlra.press.compat import install_kvpress_prefill_compat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Neutral filler sentences (cycled to build the haystack). Deliberately bland so
# the needle is the only distinctive fact.
_FILLER = [
    "The weather in the valley was mild and unremarkable that afternoon.",
    "A gentle breeze moved through the tall grass near the river.",
    "The library kept its usual hours throughout the quiet week.",
    "Several birds gathered on the fence before flying away together.",
    "The old clock in the hallway ticked steadily as always.",
    "Rows of books lined the shelves from floor to ceiling.",
    "The train arrived on schedule and departed a few minutes later.",
    "Sunlight fell across the wooden floor in long thin stripes.",
    "The market sold fruit, bread, and flowers every morning.",
    "A narrow path wound between the hills toward the coast.",
]


# Tokens at the end (question + assistant header) fed *after* compression, so the
# answer is generated attending to the compressed haystack (not a pre-compression
# boundary logit). The needle sits mid-haystack, far inside the compressed region.
_TAIL_K = 48


def build_haystack(
    tok: Any, context_len: int, depth: float, passcode: str
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return (prefill_ids, query_ids, target): a chat-templated needle prompt split
    so the haystack (with the needle at ``depth``) is the compressed prefill and the
    question + assistant header are the post-compression query."""
    # Build filler sentences to ~context_len tokens, insert the needle by sentence.
    sentences: list[str] = []
    i = 0
    while len(tok(" ".join(sentences)).input_ids) < context_len:
        sentences.append(_FILLER[i % len(_FILLER)])
        i += 1
    sentences.insert(int(depth * len(sentences)), f"The secret passcode is {passcode}.")
    body = " ".join(sentences)
    question = "\n\nWhat is the secret passcode? Reply with only the number."
    msgs = [{"role": "user", "content": body + question}]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"]
    return ids[:, :-_TAIL_K], ids[:, -_TAIL_K:], passcode


@torch.no_grad()
def retrieve(
    model: Any,
    tokenizer: Any,
    haystack_ids: torch.Tensor,
    query_ids: torch.Tensor,
    target: str,
    press: Any,
    device: str,
    max_new: int = 12,
) -> bool:
    """Pre-fill the haystack (compressed), then greedy-decode; True if target hit.

    Manual decode with explicit true positions so eviction (shrunken cache) is
    scored fairly.
    """
    from contextlib import nullcontext

    from transformers import DynamicCache

    cache = DynamicCache()
    hay = haystack_ids.to(device)
    ctx_len = hay.shape[1]
    cm = press(model) if press is not None else nullcontext()
    with cm:
        model(hay, past_key_values=cache, use_cache=True)

    # Decode the answer, positioning tokens at their TRUE sequence positions.
    cur = query_ids.to(device)
    start = ctx_len
    generated: list[int] = []
    for _ in range(max_new):
        q_len = cur.shape[1]
        position_ids = torch.arange(start, start + q_len, device=device).unsqueeze(0)
        out = model(cur, past_key_values=cache, use_cache=True, position_ids=position_ids)
        nxt = int(out.logits[0, -1].argmax())
        generated.append(nxt)
        start += q_len
        cur = torch.tensor([[nxt]], device=device)
    text = tokenizer.decode(generated)
    return target in text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--context-len", type=int, default=1600)
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--passcodes", nargs="+", default=["48213", "70561", "92014"])
    parser.add_argument("--out-json", default="results/w4-needle.json")
    parser.add_argument("--out-fig", default="figures/week4/needle")
    args = parser.parse_args()

    install_kvpress_prefill_compat()
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    n_features = int(model.config.head_dim * model.config.num_key_value_heads)

    # Methods at (roughly) matched aggressive memory ~0.10-0.20x.
    def bug_mem(rank: int, bits: int | None) -> float:
        return kv_memory_ratio(args.context_len, n_features, rank, bits)

    methods: list[tuple[str, Any, float]] = [
        ("baseline", None, 1.0),
        ("BUG r64", BUGPress(rank=64), bug_mem(64, None)),
        ("BUG+TQ r64/4b", BUGPress(rank=64, quant_bits=4), bug_mem(64, 4)),
        ("BUG+TQ r128/4b", BUGPress(rank=128, quant_bits=4), bug_mem(128, 4)),
        ("SnapKV keep0.20", SnapKVPress(compression_ratio=0.8), 0.20),
        ("SnapKV keep0.50", SnapKVPress(compression_ratio=0.5), 0.50),
        ("ExpectedAttn keep0.20", ExpectedAttentionPress(compression_ratio=0.8), 0.20),
        ("ExpectedAttn keep0.50", ExpectedAttentionPress(compression_ratio=0.5), 0.50),
    ]

    rows: list[dict[str, object]] = []
    for name, press, mem in methods:
        hits, total = 0, 0
        for passcode in args.passcodes:
            for depth in args.depths:
                hay, q, target = build_haystack(tokenizer, args.context_len, depth, passcode)
                ok = retrieve(model, tokenizer, hay, q, target, press, args.device)
                hits += int(ok)
                total += 1
        acc = hits / total
        rows.append({"method": name, "memory": mem, "accuracy": acc, "n": total})
        print(f"{name:24s} mem={mem:.3f}x  retrieval acc={acc:.2f} ({hits}/{total})")

    meta = {
        "model": args.model,
        "context_len": args.context_len,
        "depths": args.depths,
        "passcodes": args.passcodes,
        "results": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(meta, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        ax.scatter(
            float(cast(float, r["memory"])),
            float(cast(float, r["accuracy"])),
            s=80,
            label=r["method"],
        )
    ax.set_xlabel("stored KV memory / full fp16 cache")
    ax.set_ylabel(f"needle retrieval accuracy ({rows[0]['n']} trials)")
    ax.set_title(
        f"Needle-in-haystack under compression — {args.model.split('/')[-1]}, "
        f"ctx {args.context_len}"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(args.out_fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=150)
    print(f"[wrote {out_json} and {out}.{{pdf,png}}]")


if __name__ == "__main__":
    main()
