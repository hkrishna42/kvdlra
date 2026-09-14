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

import argparse
import gc
import re
import string
from collections import Counter
from typing import Any

import torch
from datasets import load_dataset
from transformers.cache_utils import Cache, DynamicCache

from kvdlra.baselines.compat import install_kvpress_prefill_compat
from kvdlra.eval.data import load_model
from kvdlra.eval.frontier import _footprint, _prefill_chunked, build_arms
from kvdlra.eval.ruler import _decode

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
) -> tuple[str, float]:
    """Prefill the whole prompt except its last token (chunked, OOM-safe), then
    greedy-generate the answer from the last token at TRUE positions. Returns
    (answer_text, memory-ratio of the compressed prompt)."""
    prompt_ids = prompt_ids.to(device)
    pre, last = prompt_ids[:, :-1], prompt_ids[:, -1:]
    ctx_len = int(pre.shape[1])
    streaming = arm["kind"] in ("bug", "morph", "shadow")
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
    del cache
    gc.collect()
    return text, fp.ratio_fp16(ctx_len, n)


# ------------------------------------------------------------- runner


def run(args: argparse.Namespace) -> dict[str, Any]:
    install_kvpress_prefill_compat()
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    model.config._attn_implementation = "sdpa"
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    print(f"model={args.model} n={n} tasks={args.tasks} max_len={args.max_len}", flush=True)

    results: list[dict[str, Any]] = []
    for task in args.tasks:
        try:
            # Per-task load is isolated: one task's dataset load/build failing
            # (network, HF schema drift, an empty split) must skip only that task,
            # not abort the whole axis before a single arm runs (the lb:[] bug).
            ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
            examples = [
                build_example(tokenizer, ds[i], args.max_len) for i in range(args.n_examples)
            ]
            avg_len = sum(int(p.shape[1]) for p, _ in examples) / len(examples)
        except Exception as exc:
            print(f"[{task}] SKIP {type(exc).__name__}: {exc}", flush=True)
            continue
        for arm in build_arms(args, model, args.max_len):
            arm_chunk = args.chunk if arm.get("chunkable", True) else 0
            f1s, ratios = [], []
            try:
                for prompt_ids, answers in examples:
                    text, ratio = generate(
                        model,
                        tokenizer,
                        arm,
                        prompt_ids,
                        args.device,
                        arm_chunk,
                        n,
                        h_kv,
                        args.max_new,
                    )
                    f1s.append(qa_f1_max(text, answers))
                    ratios.append(ratio)
            except Exception as exc:
                print(f"[{task}] {arm['name']:14s} SKIP {type(exc).__name__}: {exc}", flush=True)
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue
            row = {
                "task": task,
                "avg_len": avg_len,
                "method": arm["name"],
                "kind": arm["kind"],
                "rank": arm["rank"],
                "f1": sum(f1s) / len(f1s),
                "ratio_fp16": sum(ratios) / len(ratios),
                "n": len(f1s),
            }
            results.append(row)
            print(
                f"[{task} len~{avg_len:.0f}] {arm['name']:14s} f1={row['f1']:.3f} "
                f"ratio={row['ratio_fp16']:.3f}",
                flush=True,
            )
    return {
        "model": args.model,
        "n_features": n,
        "max_len": args.max_len,
        "tasks": args.tasks,
        "n_examples": args.n_examples,
        "results": results,
    }
