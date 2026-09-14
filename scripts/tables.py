"""Tables entrypoint. `convert-v1` archives the paper-v1 line files as JSONL records;
`build` (Task 3) regenerates every paper-v1 table from results/paper-v1/.

The archive is the evidence behind the v1 tables: one directory per pod holding the
per-trial Bernoulli outcomes (`trials.jsonl`), the pooled cells (`cells.jsonl`) and a
`manifest.json` naming every source file and the git SHA they were archived at. It is
written once, from the line files at tag `paper-v1-archive`, and read from then on.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _paths  # noqa: F401

from kvdlra.eval.records import (
    CellRecord,
    TrialRecord,
    parse_cell_lines,
    parse_trial_lines,
    write_jsonl,
)

MODEL_BY_TAG = {  # the exact HF ids the Week-18/19 pods ran (docs/week18-kickoff.md)
    "llama": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}
NOTE = "paper-v1 archive; haystack_id/depth/prompt_sha256 were not recorded and are null"


def _tag(stem: str) -> str:
    for t in MODEL_BY_TAG:
        if t in stem:
            return t
    raise SystemExit(f"no model tag in {stem}")


def convert_v1(out_root: Path) -> None:
    """Archive every paper-v1 line file under ``out_root/<pod>/``.

    Per-trial sources first, then the aggregate ones. 24 pod names carry both kinds of
    source file and 20 of them end up with a trials.jsonl AND a cells.jsonl under one
    merged manifest (the other 4 per-trial files hold storage tables, not [trial] lines).
    """
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "paper-v1-archive^{commit}"], text=True
    ).strip()
    for src in sorted(Path("results/w18_pertrial").glob("*-trials.txt")):
        stem = src.name.removesuffix("-trials.txt")
        pod = f"w18-g1-{stem}" if stem in MODEL_BY_TAG else f"w18-{stem}"
        _emit(out_root / pod, src, sha, per_trial=True)
    for src in sorted(Path("results/w19_pertrial").glob("*-trials.txt")):
        _emit(out_root / f"w19-{src.name.removesuffix('-trials.txt')}", src, sha, per_trial=True)
    for src in sorted(Path("results").glob("*-lines.txt")):
        _emit(out_root / src.name.removesuffix("-lines.txt"), src, sha, per_trial=False)


def _emit(pod_dir: Path, src: Path, sha: str, per_trial: bool) -> None:
    """Write one artifact into ``pod_dir`` and merge its provenance into the manifest.

    The merge is what keeps both halves of a pod that has a per-trial AND an aggregate
    source: overwriting would drop whichever was archived first.
    """
    text, model, cite = src.read_text(), _model(src, per_trial), str(src)
    rows: list[TrialRecord] | list[CellRecord] = (
        parse_trial_lines(text, model, cite) if per_trial else parse_cell_lines(text, model, cite)
    )
    if not rows:
        return  # e.g. a *-trials.txt holding a storage table, not [trial] lines
    name = "trials.jsonl" if per_trial else "cells.jsonl"
    write_jsonl(pod_dir / name, rows)
    path = pod_dir / "manifest.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(
        json.dumps(
            {
                "pod": pod_dir.name,
                "git_sha": sha,
                "source_files": sorted({*old.get("source_files", []), cite}),
                "converted_by": "scripts/tables.py convert-v1",
                "records": {**old.get("records", {}), name: len(rows)},
                "note": NOTE,
            },
            indent=2,
        )
        + "\n"
    )


def _model(src: Path, per_trial: bool) -> str:
    """The HF id the pod ran. Per-trial evidence must be attributable, so ``_tag``
    raises for it; aggregate rows may be "unknown" (the pre-Week-16 files pool
    several pods and never named a model)."""
    if not per_trial and not any(t in src.stem for t in MODEL_BY_TAG):
        return "unknown"
    return MODEL_BY_TAG[_tag(src.stem)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert-v1", help="archive the paper-v1 line files as JSONL")
    c.add_argument("--out", default="results/paper-v1")
    a = ap.parse_args()
    if a.cmd == "convert-v1":
        convert_v1(Path(a.out))


if __name__ == "__main__":
    main()
