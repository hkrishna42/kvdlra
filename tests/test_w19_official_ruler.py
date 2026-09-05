"""Week-19 A2: official NVIDIA-RULER prompts through our arms (``w19_official_ruler``).

Hermetic: a whitespace fake tokenizer with a chat template + the tiny-Llama model; the
real generator/tokenizer path runs on the pod (MODE a2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import w19_official_ruler as official
from transformers import LlamaConfig, LlamaForCausalLM
from w10_frontier import build_parser

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


def test_run_emits_intervals_compatible_rows(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    from w18_intervals import ROW

    data = tmp_path / "niah_single_2"
    data.mkdir()
    recs = [
        {"index": i, "input": NIAH, "outputs": ["7"], "length": 40, "answer_prefix": NIAH_PREFIX}
        for i in range(2)
    ]
    (data / "validation.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    monkeypatch.setattr(official, "load_model", lambda *a, **k: (_tiny(), _FakeTok()))
    args = build_parser().parse_args(
        [
            "--methods",
            "full",
            "quant",
            "--quant-nbits",
            "4",
            "--quant-scheme",
            "kivi",
            "--chunk",
            "0",
        ]
    )
    args.data_dir, args.tasks, args.context_len, args.n_examples = (
        str(tmp_path),
        ["niah_single_2"],
        64,
        None,
    )
    blob = official.run(args)
    names = [r["method"] for r in blob["results"]]
    assert names == ["full", "quant-4bit-kivi"]
    assert all(
        r["total"] == 2 and r["task"] == "niah_single_2" and r["ctx"] == 64 for r in blob["results"]
    )
    out = capsys.readouterr().out
    rows = [ln for ln in out.splitlines() if ln.startswith("[niah_single_2 ctx64]")]
    assert len(rows) == 2 and all(ROW.search(ln) for ln in rows), rows
    assert sum(ln.startswith("[trial] task=niah_single_2") for ln in out.splitlines()) == 4
