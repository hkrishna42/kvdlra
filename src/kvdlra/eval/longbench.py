"""The realistic long-context axis: LongBench token-F1 vs memory.

Complements the synthetic RULER frontier with realistic long-context QA from LongBench
(arXiv:2308.14508). Where RULER holds the query out (compress-then-query, eviction's
Achilles heel), LongBench puts the question **in the prompt** -- the standard
compress-the-whole-prompt-then-generate protocol, which is *fairer to eviction*. Having
both is the balanced long-context picture.

Focused QA subset (token-F1 scored, no extra deps): ``qasper``, ``multifieldqa_en``,
``hotpotqa``, ``2wikimqa`` -- realistic ~4-18K-token contexts. Same arms as the ppl /
RULER axes, OOM-safe chunked-ingest prefill / ``ChunkPress``, greedy generation at TRUE
positions, memory counted by ``kvdlra.accounting``.

Note: contexts are middle-truncated to ``max_len`` tokens (LongBench-style) to set the
compression budget; absolute F1 is template-sensitive, only the *within-setup* method
ordering is the claim. Most LongBench tasks live < 32K, so LongBench carries the
realistic-tasks story while RULER carries 64K.
"""

from __future__ import annotations

import functools
import gc
import re
import string
from collections import Counter
from typing import Any

import torch
from datasets import load_dataset
from transformers.cache_utils import Cache, DynamicCache

from kvdlra.cache import BugStreamingCache
from kvdlra.eval.config import TaskCfg
from kvdlra.eval.frontier import _footprint, _prefill_chunked
from kvdlra.eval.records import emit_diag
from kvdlra.eval.ruler import _decode, prompt_sha256

# One templated example: (prompt ids, reference answers).
Example = tuple[torch.Tensor, list[str]]

QA_TASKS = ("qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "narrativeqa")
_PROMPT = (
    "{context}\n\nAnswer the question based on the passage above. "
    "Be concise and only give the answer, no other words.\n\nQuestion: {q}\nAnswer:"
)


# ------------------------------------------------------------- token-F1 metric


def _normalize(s: str) -> str:
    """SQuAD/LongBench answer normalization: lowercase, drop punctuation + articles."""
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred: str, ref: str) -> float:
    p, r = _normalize(pred).split(), _normalize(ref).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def qa_f1_max(pred: str, refs: list[str]) -> float:
    return max((qa_f1(pred, ref) for ref in refs), default=0.0)


# ------------------------------------------------------------- example builder


def build_example(tok: Any, ex: dict[str, Any], max_len: int) -> tuple[torch.Tensor, list[str]]:
    """Chat-templated (context + question) prompt, context middle-truncated so the
    prompt is ~``max_len`` tokens. Returns (prompt_ids, reference answers)."""
    context = str(ex["context"])
    budget = max_len * 4  # ~4 chars/token; middle-truncate to keep head + tail
    if len(context) > budget:
        h = budget // 2
        context = context[:h] + context[-h:]
    prompt = _PROMPT.format(context=context, q=str(ex["input"]))
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )["input_ids"]
    if ids.shape[1] > max_len:  # final guard (token-exact middle truncation)
        h = max_len // 2
        ids = torch.cat([ids[:, :h], ids[:, -h:]], dim=1)
    return ids, [str(a) for a in ex["answers"]]


# ------------------------------------------------------------- generation


@torch.no_grad()
def generate(
    model: Any,
    tok: Any,
    arm: dict[str, Any],
    prompt_ids: torch.Tensor,
    device: str,
    chunk: int,
    n: int,
    h_kv: int,
    max_new: int,
) -> tuple[str, float, float]:
    """Prefill the whole prompt except its last token (chunked, OOM-safe), then
    greedy-generate the answer from the last token at TRUE positions. Returns
    (answer_text, fp16 memory ratio, stored-bits ratio of the compressed prompt)."""
    prompt_ids = prompt_ids.to(device)
    pre, last = prompt_ids[:, :-1], prompt_ids[:, -1:]
    ctx_len = int(pre.shape[1])
    streaming = arm["kind"] in ("bug", "shadow")
    if streaming:
        cache: Cache = arm["make"]()
        # Week-15 A1 fix: attach() covers prefill AND decode (uniform for all
        # streaming arms) -- decode outside attach left ShadowKV's selection hook
        # unregistered, silently degrading it to most-recent-chunks retention
        # (the same defect as the RULER harness; those published rows are VOID).
        with cache.attach(model):  # type: ignore[attr-defined]
            if 0 < chunk < ctx_len:
                _prefill_chunked(model, cache, pre, chunk)
            else:
                model(pre, past_key_values=cache, use_cache=True, logits_to_keep=1)
            fp = _footprint(arm, cache, ctx_len, n, h_kv)
            text = _decode(model, tok, cache, last, ctx_len, device, block=False, max_new=max_new)
    else:
        cache = DynamicCache()
        press = arm["make"]()
        active = press
        if press is not None and 0 < chunk < ctx_len:
            from kvpress import ChunkPress

            active = ChunkPress(press=press, chunk_length=chunk)
        from contextlib import nullcontext

        with active(model) if active is not None else nullcontext():
            model(pre, past_key_values=cache, use_cache=True, logits_to_keep=1)
        fp = _footprint(arm, cache, ctx_len, n, h_kv)
        text = _decode(model, tok, cache, last, ctx_len, device, block=True, max_new=max_new)
    if isinstance(cache, BugStreamingCache):  # the tripwire's rows, before the cache goes
        emit_diag(cache.drain_diag(), model=str(model.name_or_path), source="longbench")
    del cache
    gc.collect()
    return text, fp.ratio_fp16(ctx_len, n), fp.ratio_stored_bits(ctx_len, n)


# ------------------------------------------------------------- one trial

MAX_NEW = 48  # the v1 answer budget; LongBench QA answers are short


@functools.lru_cache(maxsize=8)
def load_examples(tok: Any, task: str, max_len: int, n: int) -> tuple[Example, ...]:
    """The first ``n`` test examples of a LongBench QA subset, already templated and
    middle-truncated to ``max_len``. Memoized: every arm of a cell rebuilds the same set,
    and each rebuild re-reads the dataset."""
    ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
    return tuple(build_example(tok, ds[i], max_len) for i in range(n))


def run_trial(
    arm: dict[str, Any],
    model: Any,
    tok: Any,
    task: TaskCfg,
    sub: str,
    seed: int,
    trial: int,
    *,
    device: str,
    chunk: int,
    n: int,
    h_kv: int,
    pool: list[str] | None = None,
) -> tuple[int, float, dict[str, Any]]:
    """Mean token-F1 over the subset's ``n_samples`` examples, as one trial record.

    LongBench's unit is a mean over examples, not a Bernoulli draw, so a cell is one
    record: ``frac`` is that mean F1 and ``hit`` is the exact-match case (F1 == 1), the
    same frac>=1 rule the needle tasks use. ``seed`` and ``pool`` are unused -- the
    examples are the dataset's own, in its own order -- and are accepted so every
    generator's ``run_trial`` has one signature.
    """
    examples = load_examples(tok, sub, task.ctx, task.n_samples)
    f1s, ratios, sbits = [], [], []
    for prompt_ids, answers in examples:
        text, ratio, sratio = generate(model, tok, arm, prompt_ids, device, chunk, n, h_kv, MAX_NEW)
        f1s.append(qa_f1_max(text, answers))
        ratios.append(ratio)
        sbits.append(sratio)
    f1 = sum(f1s) / len(f1s)
    return (
        int(f1 >= 1.0),
        f1,
        {
            "haystack_id": f"{sub}:0-{task.n_samples - 1}",  # the dataset's own examples
            "depth": None,
            "code_family": None,
            "prompt_sha256": prompt_sha256(*(p for p, _ in examples)),
            "ratio": sum(ratios) / len(ratios),
            "sbits": sum(sbits) / len(sbits),
        },
    )
