"""Week-19 A2: the official-benchmark anchor -- NVIDIA RULER prompts through OUR arms.

The retrieval evidence so far comes from an in-repo generator (``w10_ruler.py``:
cyclic/WikiText filler, our needle/query templates). This runs the SAME arms and the
SAME decode-at-true-positions protocol on prompts produced by the official RULER
generator (github.com/NVIDIA/RULER ``scripts/data/prepare.py``, pinned by commit on
the pod): their haystacks (Paul Graham essays / noise), their needle types (words,
numbers, uuids), their templates, their ``tokens_to_generate``, and their scoring
rule (``string_match_all``: every reference output must appear in the prediction).

Protocol (mirrors RULER's ``meta-llama3`` template for instruct models): the task
prompt minus its trailing answer prefix is the user turn of the tokenizer's chat
template; the answer prefix is appended after the assistant header, priming the
completion. The haystack + question up to the template-derived tail is the
compressed prefill; the tail (question + assistant header + answer prefix) is decoded
at true positions (``w10_ruler._tail_len``), exactly as in the in-repo harness.

Rows print in the ``w10_ruler`` format (``[<task> ctx<T>] <arm> acc=... n=...`` plus
per-trial ``[trial]`` lines) so ``w18_intervals.py`` ingests them unchanged.

Usage (pod)
-----------
    python scripts/data/prepare.py --save_dir data --benchmark synthetic --task niah_single_2 \
        --tokenizer_path $MODEL --tokenizer_type hf --max_seq_length 16384 --num_samples 12 \
        --model_template_type base          # in the RULER checkout
    PYTHONPATH=src python scripts/w19_official_ruler.py --model $MODEL --device cuda \
        --dtype bfloat16 --chunk 4096 --data-dir <RULER>/data --tasks niah_single_2 vt \
        --methods full bugslash --ranks 64 --hh-budgets 256 --hh-neighbor 1 --warmup-seed
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from perplexity_sweep import load_model
from w10_frontier import build_arms, build_parser
from w10_ruler import _tail_len, retrieve

from kvdlra.press.compat import install_kvpress_prefill_compat

JSON_BEGIN = "===W19_OFFICIAL_RULER_JSON_BEGIN==="
JSON_END = "===W19_OFFICIAL_RULER_JSON_END==="
# RULER synthetic.yaml tokens_to_generate (the official generation budgets).
TOKENS_TO_GENERATE = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}


def split_input(text: str) -> tuple[str, str, str]:
    """Split an official RULER ``input`` into (body, question, answer_prefix).

    RULER templates end with ``{context}\\n<question><answer_prefix>``: the question is
    the last line; the answer prefix follows its terminal ``?`` (niah/cwe/fwe/qa) or the
    ``Answer:`` cue (vt, whose question ends in ``.``). Fails loud when neither cue is
    present -- a silent mis-split would decode the wrong tail."""
    body, _, last = text.rpartition("\n")
    if "?" in last:
        cut = last.rindex("?") + 1
    elif " Answer:" in last:
        cut = last.rindex(" Answer:")
    else:
        raise ValueError(f"cannot locate the answer prefix in the last line: {last[:120]!r}")
    return body, last[:cut], last[cut:]


def templated_official(
    tok: Any, body: str, question: str, prefix: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """(compressed-prefill ids, decoded-query ids) for one official prompt: the chat
    template around ``body + question`` with generation prompt, then ``prefix`` appended
    as plain continuation tokens (RULER's meta-llama3 protocol primes the assistant).
    The tail is template-derived (``_tail_len``) so the whole question + header + prefix
    is decoded at true positions; tripwire if the question is not inside it."""

    def _chat(text: str) -> torch.Tensor:
        out = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )["input_ids"]
        return cast(torch.Tensor, out)

    full = _chat(body + "\n" + question)
    # floor 1 (not the in-repo _TAIL_K=48): official needles sit at ANY depth, so the
    # decoded tail must be exactly question + header + prefix -- never body text.
    tail_k = _tail_len(full[0], _chat(body)[0], 1)
    pre_ids = tok(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
    full = torch.cat([full, pre_ids.to(full.dtype)], dim=1)
    tail_k += int(pre_ids.shape[1])
    pre, query = full[:, :-tail_k], full[:, -tail_k:]
    q_norm = " ".join(question.split())
    if q_norm not in " ".join(tok.decode(query[0]).split()):
        raise ValueError(
            f"official-RULER template mis-slice for {getattr(tok, 'name_or_path', '?')!r}: "
            f"the question is not fully inside the decoded query (tail_k={tail_k})"
        )
    return pre, query


def load_task(data_dir: Path, task: str, n: int | None) -> list[dict[str, Any]]:
    """The first ``n`` records of ``<data_dir>/<task>/validation.jsonl``."""
    path = data_dir / task / "validation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:n] if n else rows


def run(args: Any) -> dict[str, Any]:
    install_kvpress_prefill_compat()
    model, tok = load_model(args.model, args.device, args.dtype)
    model.config._attn_implementation = "sdpa"
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_feat = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    ctx = int(args.context_len)
    print(f"model={args.model} n={n_feat} tasks={args.tasks} ctx={ctx} data={args.data_dir}")
    results: list[dict[str, Any]] = []
    for task in args.tasks:
        records = load_task(Path(args.data_dir), task, args.n_examples)
        max_new = TOKENS_TO_GENERATE[task.split("_")[0]]
        for arm in build_arms(args, model, ctx):
            arm_chunk = args.chunk if arm.get("chunkable", True) else 0
            hits, ratios, fracs, sbits = 0, [], [], []
            for rec in records:
                try:
                    body, question, prefix = split_input(rec["input"])
                    hay, query = templated_official(tok, body, question, prefix)
                    hit, ratio, frac, sratio = retrieve(
                        model, tok, arm, hay, query, list(rec["outputs"]), args.device,
                        arm_chunk, n_feat, h_kv, max_new,
                    )  # fmt: skip
                except Exception as exc:  # one bad example is skipped, logged; arm survives
                    print(f"[{task} ctx{ctx}] {arm['name']:14s} SKIP {type(exc).__name__}: {exc}")
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    continue
                hits += int(hit)
                ratios.append(ratio)
                fracs.append(frac)
                sbits.append(sratio)
                print(
                    f"[trial] task={task} ctx={ctx} arm={arm['name']} seed=0 "
                    f"trial={rec['index']} hit={int(hit)} frac={frac:.3f}",
                    flush=True,
                )
                gc.collect()
            if not ratios:
                continue
            total = len(ratios)
            row = {
                "task": task,
                "ctx": ctx,
                "method": arm["name"],
                "kind": arm["kind"],
                "rank": arm["rank"],
                "accuracy": hits / total,
                "recall_frac": sum(fracs) / total,
                "ratio_fp16": sum(ratios) / total,
                "ratio_stored_bits": sum(sbits) / total,
                "hits": hits,
                "total": total,
            }
            results.append(row)
            print(
                f"[{task} ctx{ctx}] {arm['name']:14s} acc={row['accuracy']:.2f} "
                f"recall={row['recall_frac']:.2f} ratio={row['ratio_fp16']:.3f} "
                f"sbits={row['ratio_stored_bits']:.3f} n={total}",
                flush=True,
            )
    return {"model": args.model, "benchmark": "ruler-official", "ctx": ctx, "results": results}


def main() -> None:
    parser = build_parser()  # every arm flag, identical to the ppl/RULER harnesses
    parser.add_argument("--data-dir", required=True, help="RULER prepare.py --save_dir")
    parser.add_argument("--tasks", nargs="+", default=["niah_single_2"])
    parser.add_argument("--context-len", type=int, default=16384, help="prepare max_seq_length")
    parser.add_argument("--n-examples", type=int, default=None, help="first N records per task")
    parser.add_argument("--out-json", default="results/w19-official-ruler.json")
    args = parser.parse_args()
    blob = run(args)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    print(JSON_BEGIN)
    print(json.dumps(blob))
    print(JSON_END)
    print(f"[wrote {out}]", flush=True)


if __name__ == "__main__":
    main()
