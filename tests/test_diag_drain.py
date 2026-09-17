"""The diagnostics drain: cache rows -> ``[diag]`` lines -> ``results/<pod>/diag.jsonl``.

The tripwire (L1.1) accumulates per-layer orthonormality / rank rows inside the cache
and the reader side (``records.parse_diag_lines``, `pod.py harvest`, `check`'s
``diag_skipped`` rule) was written in L0 -- but nothing emitted them, so both ends were
wired to a gap. Library code still does no I/O of its own: the eval axes drain the cache
they just finished with (``BugStreamingCache.drain_diag``) and hand the rows to
``records.emit_diag``, which prints the pod's stdout contract AND buffers the rows for
the runner to write at the end of the pod -- the same two artifacts, log and file, every
other record type leaves.

A trial that aborts on ``OrthonormalityError`` is RECORDED as an error and counted, like
any other raising trial: divergence must be visible in the results, never a silent gap.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.eval import frontier, longbench, records, ruler
from kvdlra.eval.config import ArmCfg, PodCfg, load_pod
from kvdlra.eval.records import DIAG_ROWS, emit_diag, parse_diag_lines
from kvdlra.eval.runner import _log_ppl_errors, run_pod
from tests.conftest import N_FEATURES

# One row as `BugStreamingCache.drain_diag` builds it (L1.1's 11 fields).
ROW: dict[str, object] = {
    "layer": 3,
    "absorbs": 64,
    "tokens_seen": 4096,
    "orth_err_k": 6.9e-4,
    "orth_err_v": 5.8e-4,
    "eff_rank_k": 61,
    "eff_rank_v": 64,
    "rank_k": 64,
    "rank_v": 64,
    "fixed_k": False,
    "fixed_v": True,
}


@pytest.fixture(autouse=True)
def _drain_buffer() -> Any:
    """``DIAG_ROWS`` is process-global (the runner drains it at `_finish`), so a test
    that asserts on it starts and leaves it empty."""
    DIAG_ROWS.clear()
    yield
    DIAG_ROWS.clear()


def test_emit_diag_prints_what_the_harvest_parses_back(capsys: pytest.CaptureFixture[str]) -> None:
    """Round trip: what `emit_diag` prints is exactly what `parse_diag_lines` reads,
    and the in-process buffer carries the same payload stamped the same way."""
    rows = [ROW, {**ROW, "layer": 4}]
    emit_diag(rows, model="meta-llama/Llama-3.2-1B", source="ppl")

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == 2 and all(ln.startswith("[diag] {") for ln in lines)
    # `vastai logs` truncates a line at ~500 chars; an 11-field row must not need the
    # `part=i/N` splitting the [pplw] contract falls back on.
    assert max(len(ln) for ln in lines) < 400

    parsed, skipped = parse_diag_lines(out, model="meta-llama/Llama-3.2-1B", source="pod.log")
    assert skipped == 0
    assert parsed == [
        {**r, "model": "meta-llama/Llama-3.2-1B", "source": f"pod.log:{i}"}
        for i, r in enumerate(rows, 1)
    ]
    buffered = [{**r, "model": "meta-llama/Llama-3.2-1B", "source": "ppl"} for r in rows]
    assert list(DIAG_ROWS) == buffered


def test_emit_diag_of_nothing_prints_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    """Every call site drains unconditionally; a cache with no rows yet is not a line."""
    emit_diag([], model="M", source="ppl")
    assert capsys.readouterr().out == "" and DIAG_ROWS == []


def test_score_streaming_drains_the_cache(
    tiny_model: LlamaForCausalLM, capsys: pytest.CaptureFixture[str]
) -> None:
    """The perplexity axis emits the rows of the cache it just scored, and leaves the
    cache drained -- the rows are handed on exactly once."""
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=24,
        recent_window=8,
        absorb_block=4,
        n_sink=4,
        prefill_block_size=8,  # several absorbs out of one 64-token prefill...
        diag_every=2,  # ...and a row every second one, so a short sample says something
    )
    ids = torch.randint(0, 256, (80,), generator=torch.Generator().manual_seed(0))
    frontier.score_streaming(tiny_model, cache, ids[:64], ids[64:])

    assert cache.drain_diag() == [], "the axis did not drain the cache"
    out = capsys.readouterr().out
    assert out.count("[diag] ") == len(DIAG_ROWS) > 0
    assert {r["layer"] for r in DIAG_ROWS} == {0, 1}  # every layer, not just the first
    for row in DIAG_ROWS:
        assert row["source"] == "ppl" and row["model"] == tiny_model.name_or_path
        assert set(row) == set(ROW) | {"model", "source"}
    parsed, skipped = parse_diag_lines(out, model="M", source="pod.log")
    assert skipped == 0 and len(parsed) == len(DIAG_ROWS)


def test_retrieval_and_longbench_axes_drain_too(
    tiny_model: LlamaForCausalLM, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other two axes that build a streaming cache emit the same rows under their own
    ``source``. `_decode` only ever asks the tokenizer to decode, so a stub stands in for
    one (no prompt is built here -- this is about the drain, not the generator)."""
    cfg = ArmCfg(
        name="bug-r8",
        kind="bug",
        cache={
            "rank": 8,
            "coord_budget": 24,
            "recent_window": 8,
            "absorb_block": 4,
            "n_sink": 4,
            "prefill_block_size": 8,
            "diag_every": 2,
        },
    )
    arm = frontier.build_arm(cfg, tiny_model, 64)
    tok = SimpleNamespace(decode=lambda ids: "")
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 256, (1, 68), generator=g)

    ruler.retrieve(
        tiny_model, tok, arm, ids[:, :64], ids[:, 64:], ["x"], "cpu", 0, N_FEATURES, 2, 2
    )
    assert {r["source"] for r in DIAG_ROWS} == {"ruler"} and len(DIAG_ROWS) > 0
    n_ruler = len(DIAG_ROWS)

    longbench.generate(tiny_model, tok, arm, ids[:, :65], "cpu", 0, N_FEATURES, 2, 2)
    assert {r["source"] for r in DIAG_ROWS[n_ruler:]} == {"longbench"}
    assert len(DIAG_ROWS) > n_ruler
    assert capsys.readouterr().out.count("[diag] ") == len(DIAG_ROWS)


def test_full_arm_emits_no_diag_rows(
    tiny_model: LlamaForCausalLM, capsys: pytest.CaptureFixture[str]
) -> None:
    """An arm with no streaming cache has nothing to say: no line, no buffered row."""
    ids = torch.randint(0, 256, (240,), generator=torch.Generator().manual_seed(0))
    arm = frontier.build_arm(ArmCfg(name="full", kind="full"), tiny_model, 64)
    frontier.run_ppl(
        [arm],
        tiny_model,
        frontier.windows(ids, 64, 16, 2),
        64,
        chunk=0,
        n=N_FEATURES,
        h_kv=2,
        device="cpu",
    )
    assert "[diag]" not in capsys.readouterr().out and DIAG_ROWS == []


def test_finish_writes_diag_jsonl_and_clears_the_buffer(tmp_path: Path, monkeypatch: Any) -> None:
    """The pod's last act: the buffered rows become ``results/<pod>/diag.jsonl``, counted
    in the manifest exactly as `pod.py harvest` counts them off a log."""

    def fake_trial(*a: Any, **k: Any) -> tuple[int, float, dict[str, Any]]:
        emit_diag([ROW], model="tiny", source="ruler")
        return 1, 1.0, {"haystack_id": "h0", "depth": 0.5, "prompt_sha256": "x", "ratio": 0.15,
                        "sbits": 0.15}  # fmt: skip

    monkeypatch.setattr("kvdlra.eval.ruler.run_trial", fake_trial)
    cfg: PodCfg = load_pod("w18_g1")
    cfg.arms, cfg.tasks = ["full"], ["ruler_inhouse_16k"]
    run_pod(cfg, out=tmp_path, model=None, dry_model=True)

    rows = [json.loads(x) for x in (tmp_path / "diag.jsonl").read_text().splitlines()]
    assert rows and all(r == {**ROW, "model": "tiny", "source": "ruler"} for r in rows)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["records"]["diag.jsonl"] == len(rows)
    assert DIAG_ROWS == [], "the buffer must not survive into the next pod"


def test_orthonormality_abort_is_recorded_as_an_error(
    tiny_model: LlamaForCausalLM, capsys: pytest.CaptureFixture[str]
) -> None:
    """A diverged tracker aborts the trial -- and the abort is a RECORDED, counted error
    carrying the tripwire's message, never a silently dropped arm."""
    t, window = 64, 16
    cfg = ArmCfg(
        name="bug-r8-tripwire",
        kind="bug",
        cache={
            "rank": 8,
            "coord_budget": 24,
            "recent_window": 8,
            "absorb_block": 4,
            "n_sink": 4,
            "orth_abort_tol": 1e-12,  # below the fp32 QR floor: the first absorb aborts
        },
    )
    arm = frontier.build_arm(cfg, tiny_model, t)
    ids = torch.randint(0, 256, ((t + window) * 3,), generator=torch.Generator().manual_seed(0))
    (row,) = frontier.run_ppl(
        [arm],
        tiny_model,
        frontier.windows(ids, t, window, 2),
        t,
        chunk=0,
        n=N_FEATURES,
        h_kv=2,
        device="cpu",
    )
    assert row["status"] == "error"
    assert row["error"].startswith("OrthonormalityError:") and "abort_tol=" in row["error"]

    # ...and counted: the ppl axis writes no per-trial record, so the `[error]` line is
    # what a harvest of the log counts the failure from.
    capsys.readouterr()
    assert _log_ppl_errors([row]) == 1
    errors = records.parse_error_lines(capsys.readouterr().out, source="pod.log")
    assert len(errors) == 1 and errors[0]["axis"] == "ppl" and errors[0]["arm"] == cfg.name
    assert "OrthonormalityError" in str(errors[0]["error"])
