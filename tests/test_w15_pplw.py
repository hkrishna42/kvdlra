"""Week-15 W-C: per-window NLL emission from the ppl harness (``w10_frontier``).

Every published ppl so far pooled NLL across windows and discarded the
per-window values (no error bars). Week-15 keeps them. Pins:

1. ``test_window_nll_consistency`` -- the pooled ``ppl`` in the emitted row is
   exactly recomputable from the new ``window_nlls``/``window_toks`` fields as
   ``exp(sum(nll_i * tok_i) / sum(tok_i))``, so per-window error bars can never
   disagree with the published pooled number.
2. ``test_pplw_line_format`` -- the new ``[pplw]`` printed line matches its
   documented harvest regex (``^\\[pplw`` grep discipline), and the pooled
   ``  <method> [T=..] ppl=..`` line still matches ``w11_merge.PPL_RE``
   byte-compatibly (the ``[pplw]`` line itself never does).
3. ``test_pplw_line_splits_when_long`` -- >400-char lines (vast logs truncate
   at ~500) split into ``part=i/N`` lines of 8 values that reassemble exactly.

Hermetic: tiny random-weight Llama + random-token corpus, loaders monkeypatched
so ``w10_frontier.run`` executes its real eval loop with no downloads.
"""

from __future__ import annotations

import argparse
import math
import re
from typing import Any

import pytest
import torch
import w10_frontier
from transformers import LlamaConfig, LlamaForCausalLM
from w11_merge import PPL_RE

# The documented [pplw] harvest regex (scripts/w10_frontier.py, run()):
#   [pplw] T=<T> <method> ntok=<per-window scored tokens> [part=<i>/<N>]
#   nlls=<comma-joined per-window mean NLLs, 6 decimals>
PPLW_RE = re.compile(
    r"^\[pplw\] T=(\d+) (\S+) ntok=(\d+)(?: part=(\d+)/(\d+))? nlls=([0-9.,]+)$",
    re.M,
)


def _tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    model = LlamaForCausalLM(  # type: ignore[no-untyped-call]
        LlamaConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=8192,
        )
    )
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    methods: list[str],
    t: int = 64,
    window: int = 16,
    n_samples: int = 3,
) -> dict[str, Any]:
    """Drive the REAL ``w10_frontier.run`` eval loop hermetically (cpu, no I/O)."""
    model = _tiny_model()
    n_ids = (t + window) * (n_samples + 1)  # enough exact windows for n_samples
    ids = torch.randint(0, 256, (n_ids,))
    monkeypatch.setattr(w10_frontier, "load_model", lambda *a, **k: (model, None))
    monkeypatch.setattr(w10_frontier, "load_corpus_ids", lambda *a, **k: ids)
    monkeypatch.setattr(w10_frontier, "install_kvpress_prefill_compat", lambda: None)
    args = argparse.Namespace(
        model="tiny-hermetic",
        device="cpu",
        dtype="float32",
        T=[t],
        window=window,
        n_samples=n_samples,
        ranks=[8],
        recent_window=8,
        absorb_block=4,
        chunk=0,
        corpus="wikitext-103",
        methods=methods,
    )
    return w10_frontier.run(args)


def _ok_rows(blob: dict[str, Any], t: int) -> list[dict[str, Any]]:
    rows = [r for r in blob["per_T"][str(t)] if r["status"] == "ok"]
    assert rows, "hermetic run produced no ok rows"
    return rows


def test_window_nll_consistency(monkeypatch: pytest.MonkeyPatch) -> None:
    # full exercises score_press(None); bug (rank 8) exercises score_streaming.
    blob = _run(monkeypatch, methods=["full", "bug"], n_samples=3)
    rows = _ok_rows(blob, 64)
    assert {r["method"] for r in rows} == {"full", "bug-r8"}
    for row in rows:
        nlls, toks = row["window_nlls"], row["window_toks"]
        assert len(nlls) == len(toks) == 3
        assert all(tok == 16 - 1 for tok in toks)  # scored tokens = window - 1
        assert all(math.isfinite(v) and v > 0 for v in nlls)
        assert len(set(nlls)) > 1  # genuinely per-window, not one value repeated
        # THE pin: pooled ppl == exp(sum(nll_i * tok_i) / sum(tok_i)).
        pooled = math.exp(sum(v * tok for v, tok in zip(nlls, toks, strict=True)) / sum(toks))
        # run() exponentiates through a float32 tensor; allow only that rounding.
        assert row["ppl"] == pytest.approx(pooled, rel=1e-5)


def test_pplw_line_format(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    blob = _run(monkeypatch, methods=["full", "bug"], n_samples=3)
    out = capsys.readouterr().out
    rows = _ok_rows(blob, 64)

    pplw = {m.group(2): m for m in PPLW_RE.finditer(out)}
    pooled = {m.group(1): m for m in PPL_RE.finditer(out)}
    for row in rows:
        # -- new [pplw] line: exactly one per (arm, T), matching the documented regex
        m = pplw[row["method"]]
        assert out.count(f"[pplw] T=64 {row['method']} ") == 1
        assert m.group(1) == "64"
        assert m.group(3) == str(row["window_toks"][0])
        assert m.group(4) is None  # 3 windows: single line, no part index
        assert m.group(6).split(",") == [f"{v:.6f}" for v in row["window_nlls"]]
        # -- pooled line: still byte-compatible with the w11_merge harvest
        p = pooled[row["method"]]
        assert p.group(2) == "64"
        assert p.group(3) == f"{row['ppl']:.3f}"

    # The [pplw] lines themselves must never be picked up by PPL_RE (it anchors
    # on leading whitespace) -- pooling stays uncontaminated.
    for line in out.splitlines():
        if line.startswith("[pplw]"):
            assert PPL_RE.match(line) is None


def test_pplw_line_splits_when_long(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # 48 windows -> single line would be ~460 chars > 400 -> 6 part-lines of 8.
    blob = _run(monkeypatch, methods=["full"], t=32, window=8, n_samples=48)
    out = capsys.readouterr().out
    (row,) = _ok_rows(blob, 32)
    assert len(row["window_nlls"]) == 48

    parts = [m for m in PPLW_RE.finditer(out) if m.group(2) == "full"]
    assert [(m.group(4), m.group(5)) for m in parts] == [(str(i), "6") for i in range(1, 7)]
    assert all(len(m.group(0)) <= 400 for m in parts)
    joined: list[str] = []
    for m in parts:
        vals = m.group(6).split(",")
        assert len(vals) <= 8
        joined += vals
    assert joined == [f"{v:.6f}" for v in row["window_nlls"]]
    # Equal-weight recompute from the PRINTED values (uniform windows) matches.
    printed_pooled = math.exp(sum(float(v) for v in joined) / len(joined))
    assert row["ppl"] == pytest.approx(printed_pooled, rel=1e-4)
