"""The external retrieval anchor: NVIDIA RULER prompts through our arms.

The in-house evidence comes from our own generator (``kvdlra.eval.ruler``: cyclic or
WikiText filler, our needle/query templates). This runs the SAME arms and the SAME
decode-at-true-positions protocol on prompts produced by the official RULER generator
(github.com/NVIDIA/RULER ``scripts/data/prepare.py``, pinned by commit on the pod):
their haystacks (Paul Graham essays / noise), their needle types (words, numbers,
uuids), their templates, their ``tokens_to_generate``, and their scoring rule
(``string_match_all``: every reference output must appear in the prediction).

Protocol (mirrors RULER's ``meta-llama3`` template for instruct models): the task
prompt minus its trailing answer prefix is the user turn of the tokenizer's chat
template; the answer prefix is appended after the assistant header, priming the
completion. The haystack + question up to the template-derived tail is the compressed
prefill; the tail (question + assistant header + answer prefix) is decoded at true
positions (``ruler._tail_len``), exactly as in the in-house harness.

Rows print in the in-house format (``[<task> ctx<T>] <arm> acc=... n=...`` plus
per-trial ``[trial]`` lines), which is the pod's stdout contract.

The pod prepares the prompts first::

    python scripts/data/prepare.py --save_dir data --benchmark synthetic \
        --task niah_single_2 --tokenizer_path $MODEL --tokenizer_type hf \
        --max_seq_length 16384 --num_samples 12 --model_template_type base
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, cast

import torch

from kvdlra.eval.config import TaskCfg
from kvdlra.eval.ruler import _tail_len, prompt_sha256, retrieve

# Where the pod left the prompts the RULER generator built (the pod bootstrap clones
# NVIDIA/RULER at a pinned commit and runs its prepare.py into this directory).
DATA_DIR = Path(os.environ.get("RULER_DATA", "/root/ruler_data"))

# RULER synthetic.yaml tokens_to_generate (the official generation budgets).
TOKENS_TO_GENERATE = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}


def split_input(text: str) -> tuple[str, str]:
    """Split an official RULER ``input`` into (body, question): every RULER template ends
    with ``{context}\n<question>``, and the generator (c3f5e3b) strips the answer prefix
    into the record's own ``answer_prefix`` field. Fails loud on a single-line input --
    a silent mis-split would decode the wrong tail."""
    body, sep, question = text.rpartition("\n")
    if not sep or not question.strip():
        raise ValueError(f"no question line at the end of the input: {text[-120:]!r}")
    return body, question


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


@functools.lru_cache(maxsize=16)
def load_records(data_dir: Path, task: str, n: int | None) -> list[dict[str, Any]]:
    """The first ``n`` records of ``<data_dir>/<task>/validation.jsonl`` (all of them for
    ``n=None``). Memoized: every arm of a cell re-reads the same file."""
    path = data_dir / task / "validation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:n] if n else rows


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
    """Retrieve the ``trial``-th official record of sub-task ``sub`` through ``arm``.

    ``trial`` selects a record by POSITION: the generator wrote ``n_trials`` samples per
    task at the pinned seed, in order. What the record is CALLED is RULER's business --
    their ``index`` is a sparse id (11779, 76228 in the archived w19_a2 / w19_q4off
    rows), and it is what the v1 records carry as their ``trial``, so the meta hands it
    back for the runner to record in place of the loop counter.

    ``pool`` is unused -- the haystack comes from RULER's own generator, not from ours --
    and is accepted so every generator's ``run_trial`` has one signature.
    """
    rec = load_records(DATA_DIR, sub, None)[trial]
    body, question = split_input(rec["input"])
    hay, query = templated_official(tok, body, question, rec.get("answer_prefix", ""))
    max_new = TOKENS_TO_GENERATE[sub.split("_")[0]]
    hit, ratio, frac, sbits = retrieve(
        model, tok, arm, hay, query, list(rec["outputs"]), device, chunk, n, h_kv, max_new
    )
    return (
        int(hit),
        frac,
        {
            # RULER's own record id -- their haystack, not ours, so their index names it.
            "trial": rec["index"],
            "haystack_id": f"{sub}:{rec['index']}",
            "depth": None,  # official needles sit wherever their generator put them
            "code_family": None,
            "prompt_sha256": prompt_sha256(hay, query),
            "ratio": ratio,
            "sbits": sbits,
        },
    )
