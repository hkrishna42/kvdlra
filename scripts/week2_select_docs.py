"""Reproducibly select the Week-2 pilot documents (a pre-registered rule).

Selection rule: the first ``--n`` C4 (``en``, streaming ``train`` split, default
order) documents whose ``--model`` tokenization reaches at least ``--seq-len``
tokens. This is fully deterministic given (dataset, tokenizer, n, seq_len) -- no
RNG and no manual choice -- so the pilot's document sample is auditable rather
than cherry-picked. Writes a manifest and prints the indices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main() -> None:
    """Stream C4, select the first ``n`` docs reaching ``seq_len`` tokens, write a manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--scan-limit", type=int, default=2000)
    parser.add_argument("--out", default="figures/week2/pilot_docs.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    selected: list[dict[str, int]] = []
    for i, example in enumerate(dataset):
        if i >= args.scan_limit:
            break
        n_tokens = len(tokenizer(example["text"], truncation=False).input_ids)
        if n_tokens >= args.seq_len:
            selected.append({"doc_idx": i, "n_tokens": n_tokens})
            if len(selected) >= args.n:
                break

    rule = (
        f"first {args.n} C4 (en, streaming train split, default order) documents whose "
        f"{args.model} tokenization reaches >= {args.seq_len} tokens"
    )
    manifest = {
        "selection_rule": rule,
        "model": args.model,
        "seq_len": args.seq_len,
        "n": args.n,
        "scan_limit": args.scan_limit,
        "documents": selected,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"selected doc indices: {[d['doc_idx'] for d in selected]}")
    print(f"wrote manifest {out}")


if __name__ == "__main__":
    main()
