"""Pins for the template-derived RULER query tail (``scripts/w10_ruler.py``).

The Week-16 Tier-2 generality push replaced the hardcoded ``_TAIL_K = 48`` slice --
tuned to Llama's chat-template token count -- with a template-derived tail so the whole
question + generation header lands in the decoded-at-true-position query for ANY family
(Mistral ``[/INST]`` / Qwen ``<|im_start|>assistant`` suffixes tokenize to different
lengths; a wrong slice = silent needle failure = false 0s). These tests pin:

* the pure divergence / tail-length arithmetic;
* Llama identity (a short question stays at the 48-token floor -> validated slices unchanged);
* the new-family fix (a long question expands the tail so a naive 48 would have dropped it);
* the fail-loud tripwire when the question is not recoverable from the query.

The real-tokenizer path is exercised at $0 CPU by ``scripts/w16_tier2_probe.py`` (G-TAIL).
"""

from __future__ import annotations

import pytest
import torch
from w10_ruler import _first_divergence, _tail_len, _templated

# --------------------------------------------------------------- pure arithmetic


def test_first_divergence_basic() -> None:
    a = torch.tensor([1, 2, 3, 9, 9])
    b = torch.tensor([1, 2, 3, 4, 4])
    assert _first_divergence(a, b) == 3


def test_first_divergence_prefix_returns_shorter_len() -> None:
    assert _first_divergence(torch.tensor([1, 2, 3, 4, 5]), torch.tensor([1, 2, 3])) == 3


def test_first_divergence_identical() -> None:
    assert _first_divergence(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])) == 3


def test_tail_len_floor_dominates_for_short_question() -> None:
    # body = 100 tokens, question+header = 20 -> derived 20 < floor 48 -> 48 (Llama identity)
    full = torch.arange(120)
    body = torch.cat([torch.arange(100), torch.arange(1000, 1005)])  # shared 100 + 5 header
    assert _tail_len(full, body, 48) == 48


def test_tail_len_derived_dominates_for_long_question() -> None:
    # question+header = 60 > floor 48 -> tail expands to cover the whole question (the fix)
    full = torch.arange(160)
    body = torch.cat([torch.arange(100), torch.arange(2000, 2005)])
    assert _tail_len(full, body, 48) == 60


def test_tail_len_clamped_to_leave_one_token() -> None:
    full = torch.arange(30)  # shorter than the floor
    body = torch.cat([torch.arange(10), torch.arange(3000, 3005)])
    assert _tail_len(full, body, 48) == 29  # min(max(48, 25), 29)


# ------------------------------------------------------ _templated with a fake tokenizer


class _FakeTok:
    """Deterministic whitespace tokenizer with a chat template -- enough of the HF surface
    (``apply_chat_template`` + ``decode``) for the slicing logic, no download."""

    name_or_path: str = "fake/model"

    def __init__(self, header: list[str], lossy: bool = False) -> None:
        self.header: list[str] = list(header)
        self.lossy: bool = lossy
        self._vocab: dict[str, int] = {}
        self._inv: dict[int, str] = {}

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            idx = len(self._vocab) + 1
            self._vocab[tok] = idx
            self._inv[idx] = tok
        return self._vocab[tok]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
        return_tensors: str | None = None,
        return_dict: bool = False,
    ) -> dict[str, torch.Tensor]:
        words = messages[0]["content"].split()
        if add_generation_prompt:
            words = words + self.header
        return {"input_ids": torch.tensor([[self._id(w) for w in words]])}

    def decode(self, ids: torch.Tensor) -> str:
        out: list[str] = []
        for i in ids.tolist():
            word = self._inv.get(int(i), "")
            if self.lossy and word not in self.header:
                continue  # simulate a template whose content does not round-trip
            out.append(word)
        return " ".join(out)


def _body(needle: str, n_side: int = 100) -> str:
    return " ".join(["fill"] * n_side + [needle] + ["fill"] * n_side)


def test_templated_short_question_is_llama_identity() -> None:
    tok = _FakeTok(header=["<eot>", "<asst>", "<hdr>"])
    body = _body("SECRET42")
    question = "\n\n What is the secret passcode? Reply only."
    pre, query = _templated(tok, body, question)
    assert query.shape[1] == 48  # floored -> identical to the pre-fix Llama slice
    assert "SECRET42" in tok.decode(pre[0])  # needle stays compressed
    assert "passcode?" in tok.decode(query[0])  # question decoded at true positions


def test_templated_long_question_expands_tail_and_a_naive_48_would_drop_it() -> None:
    tok = _FakeTok(header=["<eot>", "<asst>", "<hdr>"])
    body = _body("SECRET42")
    question = "\n\n " + " ".join(f"q{i}" for i in range(60)) + " ?"  # 61-token question
    _pre, query = _templated(tok, body, question)
    assert query.shape[1] > 48  # tail expanded to cover the whole question (the fix)
    assert "q0" in tok.decode(query[0]) and "q59" in tok.decode(query[0])
    # a naive fixed-48 tail would have left the FRONT of the question in the prefill:
    full = tok.apply_chat_template(
        [{"role": "user", "content": body + question}], add_generation_prompt=True
    )["input_ids"]
    assert "q0" not in tok.decode(full[0, -48:])


def test_templated_raises_when_question_not_in_query() -> None:
    tok = _FakeTok(header=["<eot>", "<asst>", "<hdr>"], lossy=True)
    with pytest.raises(ValueError, match="mis-slice"):
        _templated(tok, _body("SECRET42"), "\n\n What is the secret passcode? Reply only.")
