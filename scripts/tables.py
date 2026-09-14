"""Tables entrypoint. `convert-v1` archives the paper-v1 line files as JSONL records;
`build` (Task 3) regenerates every paper-v1 table from results/paper-v1/.

The archive is the evidence behind the v1 tables: one directory per pod holding a
verbatim `raw/` copy of every source file that named it, the parsed records
(`trials.jsonl` / `cells.jsonl` / `ppl.jsonl`, whichever formats that pod's sources
contain) and a `manifest.json` self-describing every source's line/parsed/unconverted
counts. A pod directory exists for every source file the converter reads -- even one
whose lines match no known format -- because completeness lives in the raw/ copy, not
in whether a row was parsed. It is written once, from the line files at tag
`paper-v1-archive`, and read from then on.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import _paths  # noqa: F401

from kvdlra.eval.records import (
    CellRecord,
    PplRecord,
    TrialRecord,
    parse_cell_lines,
    parse_ppl_lines,
    parse_trial_lines,
    write_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
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

    Per-trial sources first, then the aggregate ones. Globs are resolved from
    ``REPO_ROOT``, not the process cwd, so this runs the same from any directory.
    24 pod names carry both kinds of source file and 20 of them end up with a
    trials.jsonl AND a cells.jsonl under one merged manifest (the other 4 per-trial
    files hold storage tables, not [trial] lines -- they still get an archived raw/
    copy and a manifest, just no trials.jsonl).
    """
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "paper-v1-archive^{commit}"], text=True
    ).strip()
    for src in sorted((REPO_ROOT / "results" / "w18_pertrial").glob("*-trials.txt")):
        stem = src.name.removesuffix("-trials.txt")
        pod = f"w18-g1-{stem}" if stem in MODEL_BY_TAG else f"w18-{stem}"
        _emit(out_root / pod, src, sha, per_trial=True)
    for src in sorted((REPO_ROOT / "results" / "w19_pertrial").glob("*-trials.txt")):
        _emit(out_root / f"w19-{src.name.removesuffix('-trials.txt')}", src, sha, per_trial=True)
    for src in sorted((REPO_ROOT / "results").glob("*-lines.txt")):
        _emit(out_root / src.name.removesuffix("-lines.txt"), src, sha, per_trial=False)


def _emit(pod_dir: Path, src: Path, sha: str, per_trial: bool) -> None:
    """Archive one source file into ``pod_dir``: a verbatim ``raw/`` copy always, plus
    whichever of trials/cells/ppl its lines parse as, and a manifest entry recording
    that source's line/parsed/unconverted counts (merged with any sibling source's).

    Every (pod, artifact) pair in the real archive has exactly one contributing
    source (checked across all 98 source files); if a source ever collided with an
    existing artifact this raises rather than silently dropping the earlier rows.
    """
    pod_dir.mkdir(parents=True, exist_ok=True)
    (pod_dir / "raw").mkdir(exist_ok=True)
    shutil.copyfile(src, pod_dir / "raw" / src.name)

    try:
        cite = str(src.relative_to(REPO_ROOT))
    except ValueError:
        cite = str(src)  # e.g. a test's tmp_path source, outside the repo
    text = src.read_text()
    model = _model(src, per_trial)
    rows_by_artifact: dict[str, list[TrialRecord] | list[CellRecord] | list[PplRecord]] = {
        "trials.jsonl": parse_trial_lines(text, model, cite),
        "cells.jsonl": parse_cell_lines(text, model, cite),
        "ppl.jsonl": parse_ppl_lines(text, model, cite),
    }

    manifest_path = pod_dir / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records: dict[str, int] = dict(old.get("records", {}))
    for name, rows in rows_by_artifact.items():
        if not rows:
            continue
        path = pod_dir / name
        if path.exists():
            raise SystemExit(f"{pod_dir.name}/{name}: {cite} would clobber existing rows")
        write_jsonl(path, rows)
        records[name] = len(rows)

    n_lines = sum(1 for line in text.splitlines() if line.strip())
    parsed = {name.removesuffix(".jsonl"): len(rows) for name, rows in rows_by_artifact.items()}
    source_files = [e for e in old.get("source_files", []) if e["path"] != cite]
    source_files.append(
        {
            "path": cite,
            "lines": n_lines,
            "parsed": parsed,
            "unconverted": n_lines - sum(parsed.values()),
            "raw": f"raw/{src.name}",
        }
    )
    manifest_path.write_text(
        json.dumps(
            {
                "pod": pod_dir.name,
                "git_sha": sha,
                "source_files": sorted(source_files, key=lambda e: e["path"]),
                "converted_by": "scripts/tables.py convert-v1",
                "records": records,
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
    c.add_argument("--out", default=None, help="default: <repo_root>/results/paper-v1")
    a = ap.parse_args()
    if a.cmd == "convert-v1":
        out = Path(a.out) if a.out else REPO_ROOT / "results" / "paper-v1"
        convert_v1(out)


if __name__ == "__main__":
    main()
