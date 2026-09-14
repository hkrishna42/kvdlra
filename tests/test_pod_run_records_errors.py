"""A trial that raises is RECORDED, and the cell keeps its full n.

The v1 harness caught every exception, printed SKIP and dropped the trial, so a cell
that lost trials to OOM or a bad edge case looked like a clean, smaller run. The rule
now: the record is written with ``error`` set and ``hit=0``, the loop continues, the
manifest counts it, and `scripts/pod.py check` -- which requires exactly
``n_trials x len(seeds)`` records per configured cell -- still passes the shape while
the error stays visible in the row.

Hermetic: the generator is substituted, so no model is loaded and no prompt is built.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kvdlra.eval.config import load_pod, load_task
from kvdlra.eval.runner import run_pod


def test_raising_trial_is_recorded_not_skipped(tmp_path: Path, monkeypatch: Any) -> None:
    calls = {"n": 0}

    def fake_trial(*a: Any, **k: Any) -> tuple[int, float, dict[str, Any]]:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return 1, 1.0, {"haystack_id": "h0", "depth": 0.5, "prompt_sha256": "x", "ratio": 0.15,
                        "sbits": 0.15}  # fmt: skip

    monkeypatch.setattr("kvdlra.eval.ruler.run_trial", fake_trial)
    pod = load_pod("w18_g1")
    pod.arms = ["full"]
    pod.tasks = ["ruler_inhouse_16k"]
    run_pod(pod, out=tmp_path, model=None, dry_model=True)

    rows = [json.loads(x) for x in (tmp_path / "trials.jsonl").read_text().splitlines()]
    t = load_task("ruler_inhouse_16k")
    assert len(rows) == len(t.tasks) * t.n_trials * len(t.seeds)
    errs = [r for r in rows if r["error"]]
    assert len(errs) == 1 and errs[0]["hit"] == 0 and errs[0]["frac"] == 0.0
    assert "boom" in errs[0]["error"] and errs[0]["error"].startswith("RuntimeError:")
    assert json.loads((tmp_path / "manifest.json").read_text())["errors"] == 1


def test_no_cell_loses_a_record_to_the_error(tmp_path: Path, monkeypatch: Any) -> None:
    """The point of recording rather than skipping: every configured cell is still full.

    Keyed the way `pod.py check` keys them -- (arm, generator, sub-task, ctx) -- so the
    failing cell is short only if the record was dropped.
    """
    from collections import Counter

    def fake_trial(*a: Any, **k: Any) -> tuple[int, float, dict[str, Any]]:
        raise RuntimeError("everything is on fire")

    monkeypatch.setattr("kvdlra.eval.ruler.run_trial", fake_trial)
    pod = load_pod("w18_g1")
    pod.arms = ["full"]
    pod.tasks = ["ruler_inhouse_16k"]
    run_pod(pod, out=tmp_path, model=None, dry_model=True)

    rows = [json.loads(x) for x in (tmp_path / "trials.jsonl").read_text().splitlines()]
    t = load_task("ruler_inhouse_16k")
    counts = Counter((r["arm"], r["generator"], r["task"], r["ctx"]) for r in rows)
    assert set(counts.values()) == {t.n_trials * len(t.seeds)}
    assert set(counts) == {("full", "inhouse", sub, t.ctx) for sub in t.tasks}
    assert all(r["error"] and r["hit"] == 0 for r in rows)
    assert json.loads((tmp_path / "manifest.json").read_text())["errors"] == len(rows)


def test_trial_rows_stream_out_and_carry_the_generators_metadata(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rows are appended as they complete (a killed pod leaves a readable file), each
    carrying the generator that built it and whatever that generator knows about the
    prompt. The ``[trial]`` and ``[<task> ctx<T>]`` stdout lines are the pod's contract,
    so a harvest of the log alone rebuilds the same cells."""
    seen: list[int] = []

    def fake_trial(*a: Any, **k: Any) -> tuple[int, float, dict[str, Any]]:
        # Every append is visible to a reader before the next trial starts.
        seen.append(len((tmp_path / "trials.jsonl").read_text().splitlines()))
        return 1, 1.0, {"haystack_id": "h0", "depth": 0.5, "code_family": "amber",
                        "prompt_sha256": "abc", "ratio": 0.15, "sbits": 0.14}  # fmt: skip

    monkeypatch.setattr("kvdlra.eval.ruler.run_trial", fake_trial)
    pod = load_pod("w18_g1")
    pod.arms = ["full"]
    pod.tasks = ["ruler_inhouse_16k"]
    run_pod(pod, out=tmp_path, model=None, dry_model=True)

    assert seen == list(range(len(seen)))  # one row on disk per completed trial
    rows = [json.loads(x) for x in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert all(r["generator"] == "inhouse" and r["error"] is None for r in rows)
    assert rows[0]["haystack_id"] == "h0" and rows[0]["code_family"] == "amber"
    assert rows[0]["depth"] == 0.5 and rows[0]["prompt_sha256"] == "abc"
    assert rows[0]["model"] == pod.model and rows[0]["arm"] == "full"

    out = capsys.readouterr().out
    from kvdlra.eval.records import parse_cell_lines, parse_trial_lines

    assert len(parse_trial_lines(out, "m", "log")) == len(rows)
    cells = parse_cell_lines(out, "m", "log")
    assert len(cells) == len(load_task("ruler_inhouse_16k").tasks)
    assert all(c["n"] == 12 and c["acc"] == 1.0 and c["ratio"] == 0.15 for c in cells)
