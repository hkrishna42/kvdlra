"""Week-3 perplexity sweep: language-modeling loss under a compressed KV cache.

Measures how much replacing a *context's* KV cache with its rank-``r``
dynamical-low-rank reconstruction (:class:`kvdlra.press.BUGPress`) degrades the
model's predictions of the *following* tokens, on WikiText-2.

Why not a plain rolling-window perplexity (and not lm-eval-harness directly)
----------------------------------------------------------------------------
``kvpress`` presses compress in a **post-attention forward hook** -- the cache is
replaced only *after* a layer's attention has run, so it affects **subsequent**
forwards, not the one that triggered it. A single teacher-forced pass over a
window would therefore not feel the compression at all (attention within that
pass sees the uncompressed keys). To measure the compression's real effect we use
a **prefill-then-score** protocol that mirrors generation:

    1. Pre-fill a ``context`` of ``context_len`` tokens **with the press active**
       -> the per-layer cache is compressed in place.
    2. **Exit** the press context (remove the hooks), then forward the next
       ``target_len`` tokens with that compressed cache as ``past_key_values``.
       These target tokens attend to the *compressed* context (and are appended
       raw, exactly as in decoding).
    3. Cross-entropy of the target tokens (predicting ``target[i+1]`` from the
       logits at ``target[i]``) -> summed NLL / token count -> perplexity.

The baseline is the identical protocol with **no** press (full-precision context
cache), so the reported ratio isolates the cost of compression. This is a more
faithful measurement for a streaming cache compressor than lm-eval's default
rolling perplexity; the deviation from ``docs/PLAN.md`` (which named lm-eval) is
deliberate and documented here.

Runs on CPU for a quick smoke (small ``--n-windows``); use ``--device cuda`` and
larger windows on the GPU pod for the full sweep. Writes ``results/w3-ppl.json``
and ``results/w3-ppl.csv``.

Example
-------
    uv run python scripts/perplexity_sweep.py --ranks 16 32 64 \
        --context-len 1024 --target-len 512 --n-windows 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from kvdlra.press import BUGPress
from kvdlra.utils.seed import seed_everything

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"


_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def load_model(
    model_id: str, device: str, dtype: str = "float32"
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    # BUG's core is fp64 numpy regardless of storage dtype (PLAN §8 #4), so bf16
    # storage is safe and is required to fit an 8B model on a 24 GB card.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=_DTYPES[dtype])
    model.to(device)  # type: ignore[arg-type]
    model.eval()  # type: ignore[no-untyped-call]
    return model, tokenizer


def load_corpus_ids(
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    corpus: str = "wikitext-2",
    max_tokens: int = 3_000_000,
) -> torch.Tensor:
    """Tokenize a held-out corpus into one 1-D token tensor (truncated to ``max_tokens``).

    ``corpus``:

    * ``"wikitext-2"`` -- the WikiText-2 raw test split (~280K tokens). Uses the
      **namespaced** ``Salesforce/wikitext`` repo id, not the bare ``wikitext``
      alias, so ``datasets==2.21`` + newer ``huggingface_hub`` (as resolved on the
      GPU pod) does not build an ``hf://`` URI it rejects with ``HfUriError``.
    * ``"wikitext-103"`` -- the WikiText-103 raw **train** split (same
      ``Salesforce/wikitext`` repo, no remote code), **streamed** and concatenated
      until ~``max_tokens`` tokens (>100M available). WikiText-2 yields only 8/4
      non-overlapping windows at 32K/64K context; wikitext-103 gives hundreds, so
      long-context perplexity is not noise. Using the train split is fine for a
      *relative* method comparison (every method scores the identical text);
      absolute ppl is not a held-out benchmark number.
    * ``"pg19"`` -- PG19 test (long Project-Gutenberg books). Needs
      ``trust_remote_code=True`` (legacy loader); kept as an option but not the
      default (fragile on a fresh pod). Same streaming/concatenation.

    Absolute ppl is not comparable across corpora -- only the relative method
    frontier within one corpus is.
    """
    if corpus == "wikitext-2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(line for line in ds["text"] if line.strip())
    elif corpus in ("wikitext-103", "pg19"):
        if corpus == "wikitext-103":
            stream = load_dataset(
                "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
            )
        else:
            stream = load_dataset(
                "deepmind/pg19", split="test", streaming=True, trust_remote_code=True
            )
        parts: list[str] = []
        approx_tokens = 0
        for example in stream:
            chunk = cast(str, example["text"])
            if not chunk.strip():
                continue
            parts.append(chunk)
            approx_tokens += len(chunk) // 4  # ~4 chars/token; stop once we have enough
            if approx_tokens >= max_tokens:
                break
        text = "\n\n".join(parts)
    else:
        raise ValueError(f"unknown corpus {corpus!r} (use 'wikitext-2', 'wikitext-103', 'pg19')")
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)[0]
    if ids.shape[0] > max_tokens:
        ids = ids[:max_tokens]
    return cast(torch.Tensor, ids)


def load_wikitext_ids(tokenizer: PreTrainedTokenizerBase, device: str) -> torch.Tensor:
    """WikiText-2 raw test split as one 1-D token tensor (see :func:`load_corpus_ids`)."""
    return load_corpus_ids(tokenizer, device, "wikitext-2")


@torch.no_grad()
def window_nll(
    model: PreTrainedModel,
    context_ids: torch.Tensor,
    target_ids: torch.Tensor,
    press: BUGPress | None,
) -> tuple[float, int]:
    """Summed NLL (nats) and token count for one prefill-then-score window.

    ``press is None`` -> uncompressed baseline. The press is applied **only** to
    the pre-fill forward; the target forward runs with hooks removed so the
    targets attend to the compressed cache without being recompressed.
    """
    cache = DynamicCache()
    ctx = context_ids.unsqueeze(0)
    ctx_manager = press(model) if press is not None else nullcontext()
    with ctx_manager:
        model(ctx, past_key_values=cache, use_cache=True)

    # Score the targets at their TRUE sequence positions. Eviction presses shrink
    # the cache below context_len, so if we let the model derive positions from
    # the (shrunken) cache length the targets would get wrong RoPE positions and
    # wrong relative distances to the kept keys -- unfairly wrecking eviction
    # baselines. Passing explicit position_ids fixes RoPE while the causal mask
    # still lets targets attend to all kept cache entries. No-op for BUG (which
    # keeps full seq_len) and for the uncompressed baseline.
    tgt = target_ids.unsqueeze(0)
    ctx_len, tgt_len = context_ids.shape[0], target_ids.shape[0]
    position_ids = torch.arange(ctx_len, ctx_len + tgt_len, device=target_ids.device).unsqueeze(0)
    out = model(tgt, past_key_values=cache, use_cache=True, position_ids=position_ids)
    logits = out.logits[0]  # (target_len, vocab)
    # Predict target[i+1] from the logits at target[i]; scores target_len-1
    # tokens, all attending to the (compressed) context cache.
    pred = logits[:-1]
    gold = target_ids[1:]
    nll = torch.nn.functional.cross_entropy(pred, gold, reduction="sum")
    return float(nll), int(gold.shape[0])


def evaluate(
    model: PreTrainedModel,
    ids: torch.Tensor,
    press: BUGPress | None,
    args: argparse.Namespace,
) -> tuple[float, int, int]:
    """Aggregate perplexity over up to ``n_windows`` windows. Returns (ppl, tokens, windows)."""
    window = args.context_len + args.target_len
    total_nll, total_tok, n_win = 0.0, 0, 0
    for start in range(0, ids.shape[0] - window, args.stride):
        if n_win >= args.n_windows:
            break
        ctx = ids[start : start + args.context_len]
        tgt = ids[start + args.context_len : start + window]
        nll, ntok = window_nll(model, ctx, tgt, press)
        total_nll += nll
        total_tok += ntok
        n_win += 1
    ppl = math.exp(total_nll / total_tok) if total_tok else float("nan")
    return ppl, total_tok, n_win


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=list(_DTYPES))
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--target-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=None, help="default: context_len+target_len")
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--no-values", action="store_true", help="compress keys only")
    rope_group = parser.add_mutually_exclusive_group()
    rope_group.add_argument("--pre-rope", dest="pre_rope", action="store_true", default=True)
    rope_group.add_argument("--post-rope", dest="pre_rope", action="store_false")
    parser.add_argument("--out-json", default="results/w3-ppl.json")
    parser.add_argument("--out-csv", default="results/w3-ppl.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.stride is None:
        args.stride = args.context_len + args.target_len

    seed_everything(args.seed)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    ids = load_wikitext_ids(tokenizer, args.device)

    rope = "pre" if args.pre_rope else "post"
    rows: list[dict[str, object]] = []

    base_ppl, n_tok, n_win = evaluate(model, ids, None, args)
    rows.append({"config": "baseline", "rank": None, "compression_ratio": 0.0, "ppl": base_ppl})
    print(f"baseline: ppl={base_ppl:.4f}  ({n_win} windows, {n_tok} scored tokens)")

    for rank in args.ranks:
        press = BUGPress(rank=rank, pre_rope=args.pre_rope, compress_values=not args.no_values)
        ppl, _, _ = evaluate(model, ids, press, args)
        rows.append(
            {
                "config": f"bug-r{rank}",
                "rank": rank,
                "compression_ratio": press.compression_ratio,
                "ppl": ppl,
            }
        )
        print(
            f"bug r={rank:>3} ({rope}-RoPE): ppl={ppl:.4f}  "
            f"delta=+{ppl - base_ppl:.4f}  cr={press.compression_ratio:.3f}"
        )

    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "rope": rope,
        "compress_values": not args.no_values,
        "context_len": args.context_len,
        "target_len": args.target_len,
        "stride": args.stride,
        "n_windows": n_win,
        "scored_tokens": n_tok,
        "results": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(meta, indent=2) + "\n")
    with Path(args.out_csv).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "rank", "compression_ratio", "ppl"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[wrote {out_json} and {args.out_csv}]")


if __name__ == "__main__":
    main()
