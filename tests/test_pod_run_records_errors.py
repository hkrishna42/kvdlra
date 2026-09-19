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

import pod
import pytest
import torch

from kvdlra.eval.config import PodCfg, load_pod, load_task
from kvdlra.eval.records import parse_cell_lines, parse_error_lines
from kvdlra.eval.runner import run_pod


def _cfg(task: str) -> PodCfg:
    """w18_g1 narrowed to one arm and one task -- the pod every test here drives. The
    arm is `full` because none of these tests is about compression: the generator is
    substituted, so what runs is the loop's bookkeeping."""
    cfg = load_pod("w18_g1")
    cfg.arms = ["full"]
    cfg.tasks = [task]
    return cfg


def test_raising_trial_is_recorded_not_skipped(tmp_path: Path, monkeypatch: Any) -> None:
    calls = {"n": 0}

    def fake_trial(*a: Any, **k: Any) -> tuple[int, float, dict[str, Any]]:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return 1, 1.0, {"haystack_id": "h0", "depth": 0.5, "prompt_sha256": "x", "ratio": 0.15,
                        "sbits": 0.15}  # fmt: skip

    monkeypatch.setattr("kvdlra.eval.ruler.run_trial", fake_trial)
    cfg = _cfg("ruler_inhouse_16k")
    run_pod(cfg, out=tmp_path, model=None, dry_model=True)

    rows = [json.loads(x) for x in (tmp_path / "trials.jsonl").read_text().splitlines()]
    t = load_task("ruler_inhouse_16k")
    assert len(rows) == len(t.tasks) * t.n_trials * len(t.seeds)
    errs = [r for r in rows if r["error"]]
    assert len(errs) == 1 and errs[0]["hit"] == 0 and errs[0]["frac"] == 0.0
    assert "boom" in errs[0]["error"] and errs[0]["error"].startswith("RuntimeError:")
    assert json.loads((tmp_path / "manifest.json").read_text())["errors"] == 1


def test_no_cell_loses_a_record_to_the_error(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The point of recording rather than skipping: every configured cell is still full.

    Keyed the way `pod.py check` keys them -- (arm, generator, sub-task, ctx) -- so the
    failing cell is short only if the record was dropped. A cell with no surviving trial
    has no ratio to average: it used to print `ratio=nan sbits=nan`, which `CELL_RE`
    could not parse at all, so the whole cell vanished from a log harvest.
    """
    from collections import Counter

    def fake_trial(*a: Any, **k: Any) -> tuple[int, float, dict[str, Any]]:
        raise RuntimeError("everything is on fire")

    monkeypatch.setattr("kvdlra.eval.ruler.run_trial", fake_trial)
    cfg = _cfg("ruler_inhouse_16k")
    run_pod(cfg, out=tmp_path, model=None, dry_model=True)

    rows = [json.loads(x) for x in (tmp_path / "trials.jsonl").read_text().splitlines()]
    t = load_task("ruler_inhouse_16k")
    counts = Counter((r["arm"], r["generator"], r["task"], r["ctx"]) for r in rows)
    assert set(counts.values()) == {t.n_trials * len(t.seeds)}
    assert set(counts) == {("full", "inhouse", sub, t.ctx) for sub in t.tasks}
    assert all(r["error"] and r["hit"] == 0 for r in rows)
    assert json.loads((tmp_path / "manifest.json").read_text())["errors"] == len(rows)

    cells = parse_cell_lines(capsys.readouterr().out, "m", "log")
    assert len(cells) == len(t.tasks)
    assert all(c["n"] == 12 and c["acc"] == 0.0 for c in cells)
    assert all(c["ratio"] is None and c["sbits"] is None and c["errors"] == 12 for c in cells)


def test_check_refuses_a_pod_whose_trials_raised(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ruling R29. Recording rather than skipping keeps every cell full -- which is the
    point, and which also means the cell rule has nothing left to complain about. The
    error count is what fails the pod; there is no tolerance knob."""

    def fake_trial(*a: Any, **k: Any) -> tuple[int, float, dict[str, Any]]:
        raise RuntimeError("everything is on fire")

    monkeypatch.setattr("kvdlra.eval.ruler.run_trial", fake_trial)
    cfg = _cfg("ruler_inhouse_16k")
    # What `pod.py run` writes before handing over: the manifest `_finish` folds into.
    (tmp_path / "env.txt").write_text("\n".join(pod.env_lines()) + "\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(pod.manifest("w18_g1", pod._head(), "test", dry_run=False))
    )
    run_pod(cfg, out=tmp_path, model=None, dry_model=True)

    assert pod.check(tmp_path) == 1
    out = capsys.readouterr().out
    assert "CHECK FAIL errors: 48 trial(s) raised (see the error field in trials.jsonl)" in out


def test_a_generator_may_name_the_trial_itself(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The official generator's trial is RULER's own record id (76228), not the loop
    counter -- which is what the archived v1 rows carry, so a re-run joins to them."""

    def fake_trial(*a: Any, **k: Any) -> tuple[int, float, dict[str, Any]]:
        return 1, 1.0, {"trial": 76228, "ratio": 0.15, "sbits": 0.15}

    monkeypatch.setattr("kvdlra.eval.ruler.run_trial", fake_trial)
    cfg = _cfg("ruler_inhouse_16k")
    run_pod(cfg, out=tmp_path, model=None, dry_model=True)

    rows = [json.loads(x) for x in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert {r["trial"] for r in rows} == {76228}
    assert "trial=76228" in capsys.readouterr().out


def test_a_perplexity_arm_that_raises_is_logged_and_counted(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The perplexity axis has no per-trial record to hang an error on, so a failed arm
    used to vanish twice over: filtered out of the rows, and absent from the manifest's
    count. It prints an `[error]` line and increments the count -- which is what lets a
    harvest of the log alone rebuild the manifest the run wrote."""
    from kvdlra.eval import frontier

    failed = [{"method": "full", "T": 16384, "status": "error", "error": "RuntimeError: boom"}]
    # A tensor, not a sentinel: the runner digests the ids into the manifest's
    # dataset_sha256 before it cuts windows out of them.
    monkeypatch.setattr("kvdlra.eval.runner.load_corpus_ids", lambda *a, **k: torch.arange(4))
    monkeypatch.setattr(frontier, "windows", lambda *a, **k: [("ctx", "win")])
    monkeypatch.setattr(frontier, "run_ppl", lambda *a, **k: failed)
    cfg = _cfg("ppl_16k")
    run_pod(cfg, out=tmp_path, model=None, dry_model=True)

    out = capsys.readouterr().out
    assert "[error] axis=ppl arm=full ctx=16384 error=RuntimeError: boom" in out
    assert "[stage] load_corpus_ids wikitext-103 (" in out  # the corpus load is timed
    (err,) = parse_error_lines(out, "log")
    assert err["axis"] == "ppl" and err["arm"] == "full" and err["ctx"] == 16384
    assert err["error"] == "RuntimeError: boom" and str(err["source"]).startswith("log:")
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["errors"] == 1
    # The corpus digest reaches the log in the form the harvest parses (L2.9a): the manifest
    # this run wrote stays on the pod, and the harvested one is rebuilt from these lines.
    assert dict(pod.DIGEST_RE.findall(out)) == m["dataset_sha256"] != {}
    assert not (tmp_path / "ppl.jsonl").exists()  # a failed arm writes no record


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
    cfg = _cfg("ruler_inhouse_16k")
    run_pod(cfg, out=tmp_path, model=None, dry_model=True)

    assert seen == list(range(len(seen)))  # one row on disk per completed trial
    rows = [json.loads(x) for x in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert all(r["generator"] == "inhouse" and r["error"] is None for r in rows)
    assert rows[0]["haystack_id"] == "h0" and rows[0]["code_family"] == "amber"
    assert rows[0]["depth"] == 0.5 and rows[0]["prompt_sha256"] == "abc"
    assert rows[0]["model"] == cfg.model and rows[0]["arm"] == "full"

    out = capsys.readouterr().out
    from kvdlra.eval.records import parse_cell_lines, parse_trial_lines

    assert len(parse_trial_lines(out, "m", "log")) == len(rows)
    cells = parse_cell_lines(out, "m", "log")
    subs = load_task("ruler_inhouse_16k").tasks
    assert len(cells) == len(subs)
    assert all(c["n"] == 12 and c["acc"] == 1.0 and c["ratio"] == 0.15 for c in cells)

    # One `[stage] cell` line per cell, carrying its own elapsed seconds: no other row
    # the harvest keeps has a clock, and a per-arm rate is what sizes the next pod.
    timings = pod.CELL_S_RE.findall(out)
    assert [(a, t, c) for a, t, c, _ in timings] == [("full", sub, "16384") for sub in subs]
    assert all(float(s) >= 0.0 for *_, s in timings)
    assert out.count(" n=12\n") >= len(subs)  # the cell's record count rides the line
