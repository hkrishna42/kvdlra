"""Week-10 PRIMARY long-context axis (realistic tasks): LongBench F1 vs memory.

Complements the synthetic RULER frontier (``w10_ruler.py``) with *realistic*
long-context QA from LongBench (arXiv:2308.14508). Where RULER holds the query out
(compress-then-query, eviction's Achilles heel), LongBench puts the question
**in the prompt** -- the standard compress-the-whole-prompt-then-generate protocol,
which is *fairer to eviction* (it can keep query-relevant tokens at prefill). Having
both is the balanced long-context picture.

Focused QA subset (token-F1 scored, no extra deps): ``qasper``, ``multifieldqa_en``,
``hotpotqa``, ``2wikimqa`` -- realistic ~4-18K-token contexts. Same arms as the ppl
/ RULER frontiers (BUG ``BugStreamingCache`` ranks / MorphKV / SnapKV / EA), OOM-safe
chunked-ingest prefill / ``ChunkPress``, greedy generation at TRUE positions. Memory
counted honestly (``kvdlra.accounting``). ``--plot-only`` rebuilds figures.

Note: contexts are middle-truncated to ``--max-len`` tokens (LongBench-style) to set
the compression budget; absolute F1 is template-sensitive, only the *within-setup*
method ordering is the claim. Most LongBench tasks live < 32K, so LongBench carries
the realistic-tasks story while RULER carries 64K.

Example (CPU smoke, 1B)
-----------------------
    uv run python scripts/w10_longbench.py --device cpu --tasks qasper \
        --max-len 2048 --n-examples 3 --ranks 128 --methods full bug morph snapkv
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import matplotlib
import torch
from datasets import load_dataset
from perplexity_sweep import load_model
from transformers.cache_utils import Cache, DynamicCache
from w10_frontier import _footprint, _prefill_chunked, build_arms
from w10_ruler import _decode

from kvdlra.press.compat import install_kvpress_prefill_compat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
JSON_BEGIN = "===W10_LONGBENCH_JSON_BEGIN==="
JSON_END = "===W10_LONGBENCH_JSON_END==="
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
    streaming = arm["kind"] in ("bug", "morph")
    if streaming:
        cache: Cache = arm["make"]()
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
        ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
        examples = [build_example(tokenizer, ds[i], args.max_len) for i in range(args.n_examples)]
        avg_len = sum(int(p.shape[1]) for p, _ in examples) / len(examples)
        for arm in build_arms(args, model, args.max_len):
            arm_chunk = args.chunk if arm.get("chunkable", True) else 0
            f1s, ratios = [], []
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


# ------------------------------------------------------------- plot


def _plot(blob: dict[str, Any], out: Path) -> None:
    results = blob["results"]
    tasks = blob["tasks"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.8 * len(tasks), 4.4), squeeze=False)
    kind_style = {
        "bug": ("tab:orange", "o"),
        "morph": ("tab:green", "s"),
        "press": ("tab:red", "^"),
        "full": ("tab:blue", "*"),
    }
    for j, task in enumerate(tasks):
        ax = axes[0][j]
        rows = [r for r in results if r["task"] == task]
        for kind, (c, m) in kind_style.items():
            pts = sorted((r["ratio_fp16"], r["f1"]) for r in rows if r["kind"] == kind)
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker=m,
                    color=c,
                    lw=1.8,
                    ms=7,
                    label=kind,
                    alpha=0.9,
                )
        ax.set_ylim(-0.02, 1.02)
        ax.set_xscale("log")
        ax.set_title(task, fontsize=10)
        ax.set_xlabel("KV memory / full fp16")
        ax.set_ylabel("token-F1")
        ax.grid(True, which="both", alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"Week-10 LongBench: QA F1 vs memory -- {blob['model'].split('/')[-1]}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=150)
    print(f"[wrote {out.with_suffix('.png')}]", flush=True)


# ------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument(
        "--tasks", nargs="+", default=["qasper", "multifieldqa_en"], choices=list(QA_TASKS)
    )
    parser.add_argument("--max-len", type=int, default=4096, help="prompt token budget")
    parser.add_argument("--n-examples", type=int, default=20)
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--morph-keeps", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--evict-keeps", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--think-ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--palu-ranks", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--palu-group", type=int, default=1)
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--absorb-block", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=0, help="chunked-prefill block size")
    parser.add_argument(
        "--methods", nargs="+", default=["full", "bug", "morph", "snapkv", "ea", "think", "palu"]
    )
    parser.add_argument("--out-json", default="results/w10-longbench-1b.json")
    parser.add_argument("--out-fig", default="figures/week10/longbench_f1")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    out_json = Path(args.out_json)
    if args.plot_only:
        _plot(json.loads(out_json.read_text()), Path(args.out_fig))
        return

    blob = run(args)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(blob, indent=2) + "\n")
    _plot(blob, Path(args.out_fig))
    print(JSON_BEGIN)
    print(json.dumps(blob))
    print(JSON_END)
    print(f"[wrote {out_json}]", flush=True)


if __name__ == "__main__":
    main()
