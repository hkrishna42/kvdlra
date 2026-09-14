"""The external anchor: official NVIDIA-RULER prompts through our arms.

Hermetic: a whitespace fake tokenizer with a chat template + the tiny-Llama model; the
real generator/tokenizer path runs on the pod, which writes the prompts into
``official_ruler.DATA_DIR`` before ``pod.py run`` reads them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.eval import official_ruler as official

# Shaped like RULER c3f5e3b records: the answer prefix is its own field, not in `input`.
NIAH = (
    "Some special magic numbers are hidden within the following text. Make sure to memorize it."
    "\n"
    + "The grass is green. The sky is blue. "
    * 40
    + "One of the special magic numbers for apple is: 7."
    "\nWhat are all the special magic numbers for apple mentioned in the provided text?"
)
NIAH_PREFIX = " The special magic numbers for apple mentioned in the provided text are"
VT = (
    "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
    "VAR ABC = 12345 VAR XYZ = VAR ABC\n"
    "Question: Find all variables that are assigned the value 12345 in the text above."
)


def test_split_input_takes_the_last_line_as_the_question() -> None:
    body, question = official.split_input(NIAH)
    assert body.endswith("apple is: 7.")
    assert (
        question
        == "What are all the special magic numbers for apple mentioned in the provided text?"
    )
    body, question = official.split_input(VT)
    assert body.endswith("VAR XYZ = VAR ABC")
    assert question.startswith("Question: Find all") and question.endswith("above.")


def test_split_input_fails_loud_on_a_single_line() -> None:
    with pytest.raises(ValueError, match="question line"):
        official.split_input("no newline anywhere in this prompt")


class _FakeTok:
    """Whitespace tokenizer with a chat template and a plain-text ``__call__``."""

    name_or_path = "fake/model"

    def __init__(self) -> None:
        self.header = ["<assistant>"]
        self._vocab: dict[str, int] = {}
        self._inv: dict[int, str] = {}

    def _id(self, w: str) -> int:
        if w not in self._vocab:
            self._vocab[w] = len(self._vocab) + 1
            self._inv[self._vocab[w]] = w
        return self._vocab[w]

    def apply_chat_template(self, messages: list[dict[str, str]], **kw: Any) -> dict[str, Any]:
        words = messages[0]["content"].split() + (
            self.header if kw.get("add_generation_prompt") else []
        )
        return {"input_ids": torch.tensor([[self._id(w) for w in words]])}

    def __call__(self, text: str, **kw: Any) -> dict[str, Any]:
        return {"input_ids": torch.tensor([[self._id(w) for w in text.split()]])}

    def decode(self, ids: Any) -> str:
        seq = ids if isinstance(ids, list) else ids.tolist()
        return " ".join(self._inv.get(int(i), "") for i in seq)


def test_templated_official_puts_question_header_and_prefix_in_the_decoded_tail() -> None:
    tok = _FakeTok()
    body, question = official.split_input(NIAH)
    pre, query = official.templated_official(tok, body, question, NIAH_PREFIX)
    full_words = (body + "\n" + question).split() + tok.header + NIAH_PREFIX.split()
    assert pre.shape[1] + query.shape[1] == len(full_words)
    tail = tok.decode(query[0])
    assert question in tail and "<assistant>" in tail and NIAH_PREFIX.strip() in tail
    assert "apple is: 7." not in tail  # the needle stays in the compressed prefill


def _tiny() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg)  # type: ignore[no-untyped-call]
    m.config._attn_implementation = "sdpa"
    m.eval()  # type: ignore[no-untyped-call]
    return m


def test_run_trial_reads_the_indexed_official_record(tmp_path: Path, monkeypatch: Any) -> None:
    """One trial IS one official record, at the trial index: the generator wrote the
    samples in order at the pinned seed, so ``trial`` selects one without any
    re-sampling of ours. The meta carries RULER's own record id."""
    from kvdlra.eval.config import ArmCfg, load_task
    from kvdlra.eval.frontier import build_arm

    data = tmp_path / "niah_single_2"
    data.mkdir()
    recs = [
        {"index": i, "input": NIAH, "outputs": ["7"], "length": 40, "answer_prefix": NIAH_PREFIX}
        for i in range(2)
    ]
    (data / "validation.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    monkeypatch.setattr(official, "DATA_DIR", tmp_path)
    official.load_records.cache_clear()

    model, tok = _tiny(), _FakeTok()
    task = load_task("ruler_official_16k")
    task.ctx = 64
    arm = build_arm(ArmCfg(name="full", kind="full"), model, task.ctx)
    for trial in (0, 1):
        hit, frac, meta = official.run_trial(
            arm, model, tok, task, "niah_single_2", 42, trial,
            device="cpu", chunk=0, n=64, h_kv=2,
        )  # fmt: skip
        assert hit in (0, 1) and 0.0 <= frac <= 1.0
        assert meta["haystack_id"] == f"niah_single_2:{trial}"
        assert meta["depth"] is None and len(meta["prompt_sha256"]) == 64
        assert 0.0 < meta["ratio"] <= 1.0


def test_load_records_reads_the_generators_validation_file(tmp_path: Path) -> None:
    data = tmp_path / "vt"
    data.mkdir()
    rows = [{"index": i, "input": VT, "outputs": ["12345"]} for i in range(3)]
    (data / "validation.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    official.load_records.cache_clear()
    assert [r["index"] for r in official.load_records(tmp_path, "vt", None)] == [0, 1, 2]
    assert [r["index"] for r in official.load_records(tmp_path, "vt", 2)] == [0, 1]
