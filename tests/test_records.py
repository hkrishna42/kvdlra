"""The record schema behind `results/paper-v1/`: parse, round-trip, archive counts.

`parse_trial_lines` / `parse_cell_lines` are the single readers for the two formats
the Week-18/19 pods emitted; `scripts/tables.py convert-v1` is the only writer of the
archive. These pins are what let Task 9 delete the original line files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tables

from kvdlra.eval.records import parse_cell_lines, parse_trial_lines, read_jsonl, write_jsonl

ARCHIVE = Path(__file__).parent.parent / "results" / "paper-v1"

TRIAL = "[trial] task=niah_single ctx=16384 arm=bugSseed-r64-h256 seed=1 trial=3 hit=1 frac=1.000\n"
CELL = (
    "[niah_single ctx16384] bugSseed-r64-h256 acc=1.000 recall=1.000 ratio=0.151 sbits=0.151 n=12\n"
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
            "haystack_id": None,
            "depth": None,
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


def test_parse_cell_lines_recovers_hits() -> None:
    rows = parse_cell_lines(CELL, model="M", source="f.txt")
    assert rows[0]["hits"] == 12 and rows[0]["n"] == 12 and rows[0]["ratio"] == 0.151


def test_parse_cell_lines_without_n_has_no_hits() -> None:
    """Pre-Week-18 rows carry no `n=`, so the Bernoulli count is unrecoverable."""
    old = "[niah_single ctx16384] bug-r64 acc=0.500 recall=0.500 ratio=0.068\n"
    (row,) = parse_cell_lines(old, model="M", source="f")
    assert row["n"] is None and row["hits"] is None and row["acc"] == 0.5


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


def test_emit_skips_empty_sources(tmp_path: Path) -> None:
    """Some archived `-trials.txt` files hold storage tables, not [trial] lines."""
    src = tmp_path / "llama-trials.txt"
    src.write_text("[ctx 16384] bug-r64 stored_ratio=0.0685\n")
    tables._emit(tmp_path / "pod", src, "deadbee", per_trial=True)
    assert not (tmp_path / "pod").exists()


def test_v1_archive_counts_match_sources() -> None:
    # every archived pod has exactly as many records as [trial] lines in its source
    assert (ARCHIVE / "w18-g1-llama/trials.jsonl").exists()
    assert len(read_jsonl(ARCHIVE / "w18-g1-llama/trials.jsonl")) == 192
    assert len(read_jsonl(ARCHIVE / "w19-a2-llama/trials.jsonl")) == 756


def test_v1_archive_manifest_keeps_both_artifacts() -> None:
    """24 pods have both a per-trial and an aggregate source; the manifest must name
    both (a single-source manifest would drop half the provenance)."""
    import json

    m = json.loads((ARCHIVE / "w19-a2-llama" / "manifest.json").read_text())
    assert m["git_sha"] == "ee8c0ab"
    assert set(m["records"]) == {"trials.jsonl", "cells.jsonl"}
    assert m["records"]["trials.jsonl"] == 756
    assert len(m["source_files"]) == 2


def test_v1_archive_models_are_the_pod_hf_ids() -> None:
    rows = read_jsonl(ARCHIVE / "w18-g1-llama" / "trials.jsonl")
    assert {r["model"] for r in rows} == {"unsloth/Meta-Llama-3.1-8B-Instruct"}
