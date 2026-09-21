"""The `latency` axis is a wired generator, not a module nobody calls.

`kvdlra.eval.latency` measured the decode cost the paper reports and then had no
importer at all: the numbers were regenerable only by a script that no longer exists.
Here the runner drives it from `configs/tasks/latency_16k_32k_64k.yaml` -- one row per
(arm, ctx, batch) in `results/<pod>/latency.jsonl` -- and `scripts/pod.py check`
requires that grid, so a point that OOMed cannot leave a short file behind and pass.

Hermetic: `run_latency` is substituted, so no model is loaded and nothing is timed -- with
one exception, the last test, which measures two decode steps on the tiny model so the REAL
print meets the real harvest regex (everything else here would pass on a print that no
parser can read).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pod
import pytest
from transformers import LlamaForCausalLM

from kvdlra.eval.config import load_arm, load_pod, load_task
from kvdlra.eval.latency import run_latency
from kvdlra.eval.records import parse_error_lines, parse_latency_lines
from kvdlra.eval.runner import run_pod
from tests.conftest import tiny_cache

TINY_SDPA = True
POD = "w19_sysfix_latency"


def _fake_run_latency(
    model: Any,
    arms: list[dict[str, Any]],
    ctx: int,
    device: str,
    chunk: int,
    n_steps: int,
    warmup: int,
    batch: int = 1,
) -> list[dict[str, Any]]:
    """The module's own row shape, with fixed numbers instead of measured ones."""
    (arm,) = arms
    return [
        {
            "method": arm["name"],
            "kind": arm["kind"],
            "ctx": ctx,
            "batch": batch,
            "ms_per_tok_p50": 100.0,
            "ms_per_tok_mean": 110.0,
            "ms_per_tok_max": 300.0,
            "spikes_gt_2x": 4,
            "n_steps": n_steps - warmup,
            "resident_gb": 15.0,
            "peak_gb": 18.0,
            "weights_gb": 14.0,
            "kv_resident_gb": 1.0,
            "kv_peak_gb": 4.0,
            "backend": None,
            "per_step_ms": [],
        }
    ]


def _launch(tmp_path: Path) -> None:
    """What `pod.py run` writes before handing over: the manifest `_finish` folds into."""
    (tmp_path / "env.txt").write_text("\n".join(pod.env_lines()) + "\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(pod.manifest(POD, pod._head(), "test", dry_run=False))
    )


def _check_fails_when_a_point_is_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Drop the last decode row and demand the exact failure. The axis writes no trial,
    so this row is the only thing that CAN fail -- exactly the hole that let a pod with a
    dead 64K arm look clean -- and the rule has to hold for a `run_pod`-written file and a
    harvested one alike, which is why both tests below end here."""
    lat = tmp_path / "latency.jsonl"
    kept = lat.read_text().splitlines()
    lat.write_text("\n".join(kept[:-1]) + "\n")
    capsys.readouterr()
    assert pod.check(tmp_path) == 1
    dropped = json.loads(kept[-1])
    assert (
        f"CHECK FAIL latency: {dropped['arm']} ctx={dropped['ctx']}"
        f" batch={dropped['batch']} has no decode record" in capsys.readouterr().out
    )


def test_the_loop_writes_one_row_per_arm_ctx_batch(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("kvdlra.eval.latency.run_latency", _fake_run_latency)
    cfg = load_pod(POD)
    run_pod(cfg, out=tmp_path, model=None, dry_model=True)
    # L3.3a: this axis prints `_cell`'s timing line too -- one per (arm, ctx) sweep over
    # the batch sizes, keyed by the TASK name, which is the only clock a harvest carries.
    timings = pod.CELL_S_RE.findall(capsys.readouterr().out)

    task = load_task(cfg.tasks[0])
    rows = [json.loads(x) for x in (tmp_path / "latency.jsonl").read_text().splitlines()]
    names = [load_arm(a).legacy_name or a for a in cfg.arms]
    assert {(a, t, c) for a, t, c, _ in timings} == {
        (a, task.name, str(c)) for a in names for c in task.ctxs or [task.ctx]
    }
    assert {(r["arm"], r["ctx"], r["batch"]) for r in rows} == {
        (a, c, b) for a in names for c in task.ctxs or [task.ctx] for b in task.batch_sizes
    }
    assert len(rows) == len(names) * len(task.ctxs or []) * len(task.batch_sizes)
    r = rows[0]
    assert r["model"] == cfg.model and r["source"] == f"{cfg.name}:run"
    assert (r["ms_per_token_p50"], r["ms_mean"], r["ms_max"]) == (100.0, 110.0, 300.0)
    assert (r["spikes"], r["resident_gb"], r["peak_gb"], r["kv_peak_gb"]) == (4, 15.0, 18.0, 4.0)
    assert r["kv_resident_gb"] == 1.0
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["records"] == {"trials.jsonl": 0, "latency.jsonl": len(rows)} and m["errors"] == 0


def test_check_passes_on_the_full_grid_and_fails_without_it(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("kvdlra.eval.latency.run_latency", _fake_run_latency)
    _launch(tmp_path)
    run_pod(load_pod(POD), out=tmp_path, model=None, dry_model=True)
    assert pod.check(tmp_path) == 0
    _check_fails_when_a_point_is_dropped(tmp_path, capsys)


def test_a_point_that_raises_is_logged_and_counted(
    tmp_path: Path, monkeypatch: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """No row of its own to carry the failure, so it is an `[error]` line and a count --
    the rule the perplexity axis already follows."""

    def boom(*a: Any, **k: Any) -> list[dict[str, Any]]:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("kvdlra.eval.latency.run_latency", boom)
    cfg = load_pod(POD)
    run_pod(cfg, out=tmp_path, model=None, dry_model=True)

    out = capsys.readouterr().out
    errs = parse_error_lines(out, "log")
    task = load_task(cfg.tasks[0])
    n = len(cfg.arms) * len(task.ctxs or []) * len(task.batch_sizes)
    assert len(errs) == n and {e["axis"] for e in errs} == {"latency"}
    assert errs[0]["error"] == "RuntimeError: CUDA out of memory"
    assert {e["batch"] for e in errs} == set(task.batch_sizes)  # batch=, not None
    assert not (tmp_path / "latency.jsonl").exists()  # a failed point writes no record
    assert json.loads((tmp_path / "manifest.json").read_text())["errors"] == n


def test_harvest_rebuilds_latency_jsonl_from_the_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`records.parse_latency_lines` is the harvest-side counterpart to
    `kvdlra.eval.latency.run_latency`'s print -- without it a results directory that
    never made it off the instance had no way back to a checkable `latency.jsonl` (only
    `run_pod` ever wrote one). Mirrors `test_check_passes_on_the_full_grid_and_fails_
    without_it` above, but the file comes from a harvested log instead of a live run."""
    _launch(tmp_path)
    cfg = load_pod(POD)
    task = load_task(cfg.tasks[0])
    names = [load_arm(a).legacy_name or a for a in cfg.arms]
    ctxs = task.ctxs or [task.ctx]
    lines = [
        f"[latency ctx{ctx}] {arm:22s} ms/tok=100.00 mean=110.00 max=300.00 spikes=4 "
        f"resident_gb=15.00 peak_gb=18.00 weights_gb=14.00 kv_peak_gb=4.00 batch=1"
        for ctx in ctxs
        for arm in names
    ]
    log = tmp_path / "pod.log"
    log.write_text("\n".join(lines) + "\n")

    assert pod.harvest(POD, log, tmp_path, force=False) == 0
    rows = [json.loads(x) for x in (tmp_path / "latency.jsonl").read_text().splitlines()]
    got = {(r["arm"], r["ctx"], r["batch"]) for r in rows}
    assert got == {(a, c, 1) for a in names for c in ctxs}
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["records"]["latency.jsonl"] == len(rows) == 9

    capsys.readouterr()
    assert pod.check(tmp_path) == 0
    _check_fails_when_a_point_is_dropped(tmp_path, capsys)


def test_the_printed_latency_line_is_what_the_harvest_regex_reads(
    tiny_model: LlamaForCausalLM, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run_latency`'s print and `records.LATENCY_RE` are one contract in two places, and
    every other test in this module substitutes the print away. Two measured decode steps on
    the tiny model -- a full arm and a kernel arm -- through the real print and back through
    the real parser: one record per arm, the batch and the resident reading recovered, and
    the kernel arm's attested backend with them."""
    arms: list[dict[str, Any]] = [
        {"name": "full", "kind": "full", "make": lambda: None},
        {
            "name": "tiny_kernel",
            "kind": "bug",
            "make": lambda: tiny_cache(
                tiny_model, decode_attention="kernel", kernel_operand_dtype="float32"
            ),
        },
    ]
    rows = run_latency(tiny_model, arms, 64, "cpu", chunk=32, n_steps=2, warmup=0, batch=1)
    printed = [x for x in capsys.readouterr().out.splitlines() if x.startswith("[latency ")]
    got = parse_latency_lines("\n".join(printed), model="tiny", source="log")
    assert len(printed) == len(got) == len(arms)
    assert [g["arm"] for g in got] == [r["method"] for r in rows] == ["full", "tiny_kernel"]
    assert all(g["batch"] == 1 and isinstance(g["kv_resident_gb"], float) for g in got)
    assert [g["backend"] for g in got] == [None, "reference"]  # a CPU query: never triton
