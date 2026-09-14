"""Model + corpus loading and the needle-generator's fixed word lists.

Everything the eval modules read from disk or the Hub lives here: the tokenizer/model
pair, the held-out token stream a perplexity sweep scores, the natural-text sentence
pool the realistic RULER filler draws from, and the two frozen word lists the in-house
generator uses (``FILLER``, the ten cycled sentences of the archived haystack, and
``LABELS``, the needle key names).

The bodies are the Week-3..18 ones, unchanged: the corpora, the splits, the sentence
split rule and the two lists are what produced every archived number, and a change here
would silently move them. WikiText-103 TRAIN is the perplexity corpus v1 used -- a known
defect recorded in ``docs/plan/CODE_AUDIT.md``, not something this move fixes.
"""

from __future__ import annotations

import re
from typing import cast

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

# The archived in-house haystack: ten sentences cycled to the context length. Suspected
# of flattering a low-rank gist (CLAUDE.md), which is why `filler: wikitext` exists --
# but it is the corpus every v1 RULER cell ran on, so it is frozen verbatim.
FILLER = [
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

# Distinct key labels (NATO-ish) for the multi-key / multi-value needles; codes are
# distinct 5-digit numbers per trial. The LENGTH is load-bearing: niah_multivalue picks
# LABELS[trial % len(LABELS)], so adding or dropping one re-labels every trial.
LABELS = [
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


def load_model(
    model_id: str, device: str, dtype: str = "float32"
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    # The tracker's core is fp64 regardless of storage dtype (PLAN §8 #4), so bf16
    # storage is safe and is required to fit an 8B model on a 24 GB card.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPES[dtype])
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


def load_corpus_sentences(corpus: str = "wikitext-2", max_sentences: int = 20_000) -> list[str]:
    """A pool of natural-text sentences from ``corpus`` (Week-18 realistic RULER filler).

    Returns up to ``max_sentences`` sentence strings split on terminal punctuation
    from the same corpora as :func:`load_corpus_ids` (``wikitext``/``wikitext-2``,
    ``wikitext-103``, ``pg19``). Used as the needle-in-a-haystack filler pool so the
    RULER benchmark is not the self-authored 10-sentence cycle the review flagged. The
    caller seed-shuffles this pool per trial; absolute content is corpus-relative only.
    """
    name = "wikitext-2" if corpus == "wikitext" else corpus
    if name == "wikitext-2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(line for line in ds["text"] if line.strip())
    elif name in ("wikitext-103", "pg19"):
        if name == "wikitext-103":
            stream = load_dataset(
                "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
            )
        else:
            stream = load_dataset(
                "deepmind/pg19", split="test", streaming=True, trust_remote_code=True
            )
        parts: list[str] = []
        for example in stream:
            chunk = cast(str, example["text"])
            if chunk.strip():
                parts.append(chunk)
            if len(parts) >= max_sentences // 4:  # each chunk yields several sentences
                break
        text = "\n".join(parts)
    else:
        raise ValueError(f"unknown corpus {corpus!r} (use 'wikitext-2', 'wikitext-103', 'pg19')")
    # Split on sentence-terminal punctuation; keep sentences long enough to be real
    # filler (drops headers like "= Valkyria =" and one-word fragments).
    raw = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    sentences = [s.strip() for s in raw if len(s.strip()) >= 20 and " " in s.strip()]
    if not sentences:
        raise RuntimeError(f"no usable filler sentences from corpus {corpus!r}")
    return sentences[:max_sentences]
