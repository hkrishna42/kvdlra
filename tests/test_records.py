"""The record schema behind `results/paper-v1/`: parse, round-trip, archive counts.

`parse_trial_lines` / `parse_cell_lines` / `parse_ppl_lines` are the single readers for
the three formats the Week-18/19 pods emitted; `scripts/tables.py convert-v1` is the
only writer of the archive. These pins are what let Task 9 delete the original line
files: every source gets a verbatim `raw/` copy and a manifest whose line/parsed/
unconverted counts add up, so `test_v1_archive_manifests_add_up_and_match_jsonl` can
re-derive the whole count audit from the archive alone, forever.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tables

from kvdlra.eval.records import (
    parse_cell_lines,
    parse_diag_lines,
    parse_error_lines,
    parse_latency_lines,
    parse_ppl_lines,
    parse_pplw_lines,
    parse_trial_lines,
    read_jsonl,
    write_jsonl,
)

ARCHIVE = Path(__file__).parent.parent / "results" / "paper-v1"

TRIAL = "[trial] task=niah_single ctx=16384 arm=bugSseed-r64-h256 seed=1 trial=3 hit=1 frac=1.000\n"
CELL = (
    "[niah_single ctx16384] bugSseed-r64-h256 acc=1.000 recall=1.000 ratio=0.151 sbits=0.151 n=12\n"
)
PPL = "  bugSseed-r64-h256 [T=16384] ppl=5.308 tok_eq/layer=1377.7 ratio=0.085 sbits=0.150\n"
PPLW = "[pplw] T=16384 bugSseed-r64-h256 ntok=511 nlls=1.573386,1.236791\n"
# A >400-char line splits into part=i/N lines of 8 values (kvdlra.eval.frontier._log_pplw);
# vast.ai truncates a log line at ~500 chars, so the parts ARE the artifact.
PPLW_SPLIT = (
    "[pplw] T=32768 quant-2bit-kivi ntok=255 part=1/3 nlls=1.000000,2.000000\n"
    "[pplw] T=32768 quant-2bit-kivi ntok=255 part=2/3 nlls=3.000000,4.000000\n"
    "[pplw] T=32768 quant-2bit-kivi ntok=255 part=3/3 nlls=5.000000\n"
)
# kvdlra.eval.latency.run_latency's own print (weights_gb= between peak_gb= and
# kv_peak_gb=, batch= last).
LATENCY = (
    "[latency ctx16384] bugSseed-r64-h256      ms/tok=103.25 mean=117.57 max=309.70 "
    "spikes=4 resident_gb=15.79 peak_gb=18.20 weights_gb=14.96 kv_peak_gb=3.25 batch=1\n"
)


def test_parse_trial_lines_schema() -> None:
    rows = parse_trial_lines(TRIAL, model="M", source="f.txt")
    assert rows == [
        {
            "model": "M",
            "arm": "bugSseed-r64-h256",
            "task": "niah_single",
            "ctx": 16384,
            "seed": 1,
            "trial": 3,
            "hit": 1,
            "frac": 1.0,
            "generator": None,
            "haystack_id": None,
            "depth": None,
            "code_family": None,
            "prompt_sha256": None,
            "error": None,
            "source": "f.txt:1",
        }
    ]


def test_parse_lines_ignore_noise_and_cite_the_source_line() -> None:
    """Pod logs interleave progress chatter; only the two emitter formats are read,
    and `source` cites the 1-based line in the original file (the audit trail)."""
    noise = "loading shards...\n"
    assert parse_trial_lines(noise + TRIAL, model="M", source="f")[0]["source"] == "f:2"
    assert parse_cell_lines(noise + CELL, model="M", source="f")[0]["source"] == "f:2"
    assert parse_trial_lines(CELL, model="M", source="f") == []  # a cell is not a trial
    assert parse_cell_lines(TRIAL, model="M", source="f") == []  # and vice versa


def test_parse_cell_lines_schema() -> None:
    """`sbits` (fp32-at-rest stored bits) is the memory convention half the v1 tables
    print; dropping it made those columns unregenerable (task-3 review, ruling R7)."""
    rows = parse_cell_lines(CELL, model="M", source="f.txt")
    assert rows == [
        {
            "model": "M",
            "arm": "bugSseed-r64-h256",
            "task": "niah_single",
            "ctx": 16384,
            "acc": 1.0,
            "n": 12,
            "hits": 12,
            "ratio": 0.151,
            "sbits": 0.151,
            "errors": None,
            "source": "f.txt:1",
        }
    ]


def test_parse_cell_lines_without_n_or_sbits() -> None:
    """Pre-Week-18 rows carry no `n=`, so the Bernoulli count is unrecoverable, and no
    `sbits=`, so only the float-equivalent ratio is known."""
    old = "[niah_single ctx16384] bug-r64 acc=0.500 recall=0.500 ratio=0.068\n"
    (row,) = parse_cell_lines(old, model="M", source="f")
    assert row["n"] is None and row["hits"] is None and row["acc"] == 0.5
    assert row["sbits"] is None and row["ratio"] == 0.068


def test_parse_trial_lines_reads_the_generator_the_row_carries() -> None:
    """The runner prints ``generator=`` after ``frac=``; a row that has it is not
    guesswork, and a pre-Week-21 row that does not stays ``None`` (`pod.py harvest`
    fills those from the pod's task configs)."""
    (row,) = parse_trial_lines(TRIAL.rstrip("\n") + " generator=official_ruler\n", "M", "f")
    assert row["generator"] == "official_ruler" and row["error"] is None
    assert parse_trial_lines(TRIAL, "M", "f")[0]["generator"] is None


def test_parse_trial_lines_reads_the_error_a_raised_trial_printed() -> None:
    """A raised trial is recorded, not skipped -- and `error=` is the last field on the
    line, so a harvest of the log counts the same failures the run's records hold."""
    raised = (
        "[trial] task=vt ctx=16384 arm=bug-r64 seed=0 trial=2 hit=0 frac=0.000"
        " generator=inhouse error=RuntimeError: CUDA out of memory\n"
    )
    (row,) = parse_trial_lines(raised, "M", "f")
    assert row["error"] == "RuntimeError: CUDA out of memory"
    assert row["hit"] == 0 and row["generator"] == "inhouse"


def test_parse_cell_lines_of_a_wholly_failed_cell() -> None:
    """A cell whose every trial raised has no ratio to average, so the emitter prints
    none (it used to print ``ratio=nan sbits=nan``, which no reader could parse) and
    appends ``errors=``. The row still carries acc/n -- the cell kept its full n."""
    dead = "[niah_single ctx16384] bug-r64 acc=0.00 recall=0.00 n=12 errors=12\n"
    (row,) = parse_cell_lines(dead, model="M", source="f")
    assert row["ratio"] is None and row["sbits"] is None
    assert row["n"] == 12 and row["hits"] == 0 and row["errors"] == 12
    assert parse_cell_lines(CELL, "M", "f")[0]["errors"] is None  # the ok form


def test_parse_error_lines_schema() -> None:
    """The perplexity axis has no per-trial record to hang an error on -- an arm that
    raises produces no row at all -- so the failure is its own log line, and that line
    is what makes the harvested manifest's error count match the run's. The ppl axis
    has no batch sweep, so its line carries no `batch=` and the field reads None."""
    text = "noise\n[error] axis=ppl arm=bugSseed-r64-h256 ctx=32768 error=RuntimeError: boom\n"
    assert parse_error_lines(text, source="f.txt") == [
        {
            "axis": "ppl", "arm": "bugSseed-r64-h256", "ctx": 32768, "batch": None,
            "error": "RuntimeError: boom", "source": "f.txt:2",
        }
    ]  # fmt: skip
    assert parse_error_lines(TRIAL + CELL, source="f") == []


def test_parse_error_lines_reads_the_batch_a_latency_point_carries() -> None:
    """`batch=` sits between `ctx=` and `error=` on a `latency`-axis line (the decode
    axis sweeps batch sizes); `error=` must stay last so the exception message is never
    truncated by the added group."""
    text = "[error] axis=latency arm=full ctx=16384 batch=1 error=RuntimeError: CUDA out of mem\n"
    (row,) = parse_error_lines(text, source="f")
    assert row["axis"] == "latency" and row["batch"] == 1
    assert row["error"] == "RuntimeError: CUDA out of mem"


def test_parse_ppl_lines_schema() -> None:
    rows = parse_ppl_lines(PPL, model="M", source="f.txt")
    assert rows == [
        {
            "model": "M",
            "arm": "bugSseed-r64-h256",
            "ctx": 16384,
            "ppl": 5.308,
            "ratio": 0.085,
            "sbits": 0.150,
            "tok_eq": 1377.7,
            "source": "f.txt:1",
        }
    ]


def test_parse_ppl_lines_no_leading_space_and_no_sbits() -> None:
    """w11's ppl lines have no leading whitespace and never printed sbits=."""
    old = "bug-r32        [T=16384] ppl=4.566 tok_eq/layer=578.9 ratio=0.035\n"
    (row,) = parse_ppl_lines(old, model="M", source="f")
    assert row["arm"] == "bug-r32" and row["sbits"] is None and row["tok_eq"] == 578.9


def test_parse_ppl_lines_ignores_trial_and_cell_lines() -> None:
    assert parse_ppl_lines(TRIAL, model="M", source="f") == []
    assert parse_ppl_lines(CELL, model="M", source="f") == []


def test_parse_latency_lines_schema() -> None:
    """The harvest-side counterpart to `latency.run_latency`'s print -- without this a
    results directory that never made it off the instance had no way back to a
    checkable `latency.jsonl`. `weights_gb=` prints between `peak_gb=` and
    `kv_peak_gb=` but is not a field of `LatencyRecord`, so it is matched, not kept."""
    rows = parse_latency_lines(LATENCY, model="M", source="f.txt")
    assert rows == [
        {
            "model": "M",
            "arm": "bugSseed-r64-h256",
            "ctx": 16384,
            "batch": 1,
            "ms_per_token_p50": 103.25,
            "ms_mean": 117.57,
            "ms_max": 309.70,
            "spikes": 4,
            "resident_gb": 15.79,
            "peak_gb": 18.20,
            "kv_peak_gb": 3.25,
            "source": "f.txt:1",
        }
    ]


def test_parse_latency_lines_ignores_trial_and_ppl_lines() -> None:
    assert parse_latency_lines(TRIAL, model="M", source="f") == []
    assert parse_latency_lines(PPL, model="M", source="f") == []


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = parse_trial_lines(TRIAL, model="M", source="f.txt")
    write_jsonl(tmp_path / "t.jsonl", rows)
    assert read_jsonl(tmp_path / "t.jsonl") == rows


def test_emit_refuses_untagged_per_trial_source(tmp_path: Path) -> None:
    """A trials.jsonl whose model is unknown is not archivable evidence -- fail loud."""
    src = tmp_path / "nomodel-trials.txt"
    src.write_text(TRIAL)
    with pytest.raises(SystemExit, match="no model tag"):
        tables._emit(tmp_path / "pod", src, "deadbee", per_trial=True)


def test_emit_creates_pod_dir_and_raw_copy_even_when_source_yields_no_records(
    tmp_path: Path,
) -> None:
    """Some archived `-trials.txt` files hold storage tables, not [trial] lines. The
    pod dir and a verbatim raw/ copy must still exist -- Task 9's deletion gate is
    "every source has an archived counterpart", not "produced a non-empty JSONL"
    (fix round 1, review Important #1)."""
    src = tmp_path / "llama-trials.txt"
    src.write_text("[ctx 16384] bug-r64 stored_ratio=0.0685\n")
    tables._emit(tmp_path / "pod", src, "deadbee", per_trial=True)
    pod = tmp_path / "pod"
    assert not (pod / "trials.jsonl").exists()
    assert (pod / "raw" / "llama-trials.txt").read_text() == src.read_text()
    manifest = json.loads((pod / "manifest.json").read_text())
    (entry,) = manifest["source_files"]
    assert entry["lines"] == 1
    assert entry["parsed"] == {"trials": 0, "cells": 0, "ppl": 0}
    assert entry["unconverted"] == 1
    assert entry["raw"] == "raw/llama-trials.txt"


def test_emit_raises_rather_than_silently_clobber_a_second_contributor(tmp_path: Path) -> None:
    """Every (pod, artifact) pair in the real archive has exactly one contributing
    source (verified across all 98 source files); if that ever stops being true,
    fail loud rather than silently drop the first source's rows."""
    pod = tmp_path / "pod"
    first = tmp_path / "a-llama-trials.txt"
    first.write_text(TRIAL)
    tables._emit(pod, first, "deadbee", per_trial=True)
    second = tmp_path / "b-llama-trials.txt"
    second.write_text(TRIAL)
    with pytest.raises(SystemExit, match="clobber"):
        tables._emit(pod, second, "deadbee", per_trial=True)


def test_v1_archive_counts_match_sources() -> None:
    # every archived pod has exactly as many records as [trial] lines in its source
    assert (ARCHIVE / "w18-g1-llama/trials.jsonl").exists()
    assert len(read_jsonl(ARCHIVE / "w18-g1-llama/trials.jsonl")) == 192
    assert len(read_jsonl(ARCHIVE / "w19-a2-llama/trials.jsonl")) == 756


def test_v1_archive_manifest_keeps_both_artifacts() -> None:
    """24 pods have both a per-trial and an aggregate source; the manifest must name
    both (a single-source manifest would drop half the provenance)."""
    m = json.loads((ARCHIVE / "w19-a2-llama" / "manifest.json").read_text())
    assert m["git_sha"] == "ee8c0ab"
    assert set(m["records"]) == {"trials.jsonl", "cells.jsonl"}
    assert m["records"]["trials.jsonl"] == 756
    assert len(m["source_files"]) == 2


def test_v1_archive_models_are_the_pod_hf_ids() -> None:
    rows = read_jsonl(ARCHIVE / "w18-g1-llama" / "trials.jsonl")
    assert {r["model"] for r in rows} == {"unsloth/Meta-Llama-3.1-8B-Instruct"}


def test_v1_archive_ppl_pod_exists() -> None:
    """The ppl parser (fix round 1) recovers perplexity rows the original archive
    silently dropped -- see task-1-report.md Concerns / review Important #1."""
    rows = read_jsonl(ARCHIVE / "w17-qwen" / "ppl.jsonl")
    assert len(rows) == 17
    assert {r["model"] for r in rows} == {"Qwen/Qwen2.5-7B-Instruct"}


def test_v1_archive_previously_missing_pods_now_exist() -> None:
    """These 4 pod names were entirely absent before fix round 1 (both of their
    sources produced zero rows under the old two-format parser). They must now exist
    with a raw/ copy of every source that named them, per review ruling R4."""
    expect = {
        "w18-g5-llama": {"g5-llama-trials.txt", "w18-g5-llama-lines.txt"},
        "w19-a3-llama": {"a3-llama-trials.txt", "w19-a3-llama-lines.txt"},
        "w19-a3-llama2": {"a3-llama2-trials.txt", "w19-a3-llama2-lines.txt"},
        "w19-forkdiag-qwen": {"forkdiag-qwen-trials.txt", "w19-forkdiag-qwen-lines.txt"},
    }
    for pod, raw_names in expect.items():
        pod_dir = ARCHIVE / pod
        assert pod_dir.is_dir(), pod
        assert {p.name for p in (pod_dir / "raw").iterdir()} == raw_names, pod


def test_v1_archive_manifests_add_up_and_match_jsonl() -> None:
    """The manifest is the whole count audit, forever repeatable from the archive
    alone: every raw/ copy exists, every source's parsed+unconverted equals its line
    count, every per-trial source's parsed.trials equals its raw copy's [trial] line
    count, and every records[] count equals its JSONL file's actual line count."""
    manifests = sorted(ARCHIVE.glob("*/manifest.json"))
    assert len(manifests) == 74  # one dir per unique pod name across all 98 sources
    for manifest_path in manifests:
        pod_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["source_files"]:
            raw_path = pod_dir / entry["raw"]
            assert raw_path.is_file(), f"{pod_dir.name}: missing {entry['raw']}"
            assert sum(entry["parsed"].values()) + entry["unconverted"] == entry["lines"], (
                pod_dir.name,
                entry["path"],
            )
            n_trial_lines = sum(
                1 for line in raw_path.read_text().splitlines() if line.startswith("[trial]")
            )
            assert entry["parsed"]["trials"] == n_trial_lines, pod_dir.name
        for name, n in manifest["records"].items():
            assert len(read_jsonl(pod_dir / name)) == n, (pod_dir.name, name)


def test_parse_pplw_lines_schema() -> None:
    """One row per window, carrying the window's NLL SUM: the printed value is a
    per-token mean over `ntok` tokens, and a sum is what pools without re-weighting."""
    rows = parse_pplw_lines(PPLW, model="M", source="f.txt")
    assert rows == [
        {
            "model": "M", "arm": "bugSseed-r64-h256", "ctx": 16384, "window_idx": 0,
            "ntok": 511, "nll_sum_nats": 1.573386 * 511, "source": "f.txt:1",
        },
        {
            "model": "M", "arm": "bugSseed-r64-h256", "ctx": 16384, "window_idx": 1,
            "ntok": 511, "nll_sum_nats": 1.236791 * 511, "source": "f.txt:1",
        },
    ]  # fmt: skip


def test_parse_pplw_lines_reassembles_split_parts() -> None:
    rows = parse_pplw_lines(PPLW_SPLIT, model="M", source="f.txt")
    assert [r["window_idx"] for r in rows] == [0, 1, 2, 3, 4]
    assert [r["nll_sum_nats"] for r in rows] == [v * 255 for v in (1.0, 2.0, 3.0, 4.0, 5.0)]
    assert {r["source"] for r in rows} == {"f.txt:1"}  # the line the group started on


def test_parse_pplw_lines_fails_loud_on_a_missing_part() -> None:
    """A dropped fragment would silently shorten the window series -- the same
    "never silently reduces n" rule the trial records live by."""
    with pytest.raises(SystemExit, match="part"):
        parse_pplw_lines(PPLW_SPLIT.splitlines(True)[0], model="M", source="f.txt")


def test_parse_diag_lines_carries_the_payload_and_its_source() -> None:
    rows, skipped = parse_diag_lines('[diag] {"layer": 0, "rank": 64}\n', model="M", source="f.txt")
    assert rows == [{"model": "M", "layer": 0, "rank": 64, "source": "f.txt:1"}] and skipped == 0


def test_parse_diag_lines_counts_a_payload_it_cannot_parse() -> None:
    """`vastai logs` cuts a long line in half. The halved payload cannot become a record
    -- but the COUNT is what tells a harvest the log came back truncated, and dropping it
    silently made a half-fetched log look like a pod that printed no diagnostics."""
    text = '[diag] {"layer": 0, "rank": 64}\n[diag] {"layer": 1, "rank": 6\n'
    rows, skipped = parse_diag_lines(text, model="M", source="f.txt")
    assert [r["layer"] for r in rows] == [0] and skipped == 1
