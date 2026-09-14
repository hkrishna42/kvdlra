"""`make tables` regenerates the paper-v1 tables: the build output must equal the
hand-transcribed golden (docs/plan/paper-v1-tables.md), byte for byte, and the six
committed .tex deliverables must be exactly what the current build emits."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import tables

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "docs" / "plan" / "paper-v1-tables.md"
COMMITTED = REPO_ROOT / "docs" / "paper" / "tables"


def _build(out: Path) -> None:
    """Build into a throwaway dir, from any cwd (CI runs pytest from the repo root,
    `make test` from wherever the user stands)."""
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "tables.py"), "build", "--out", str(out)],
        check=True,
    )


def test_make_tables_matches_golden(tmp_path: Path) -> None:
    out = tmp_path / "tables"
    _build(out)
    got = "".join(p.read_text() for p in sorted(out.glob("table*.md")))
    assert got == GOLDEN.read_text()


def test_committed_tex_files_are_the_built_ones(tmp_path: Path) -> None:
    """The .tex deliverables are generated, never maintained: a stale committed file
    is a hand-edited paper table, which is what `make tables` exists to prevent."""
    out = tmp_path / "tables"
    _build(out)
    built = sorted(out.glob("table*.tex"))
    assert [p.name for p in built] == [f"table{n}.tex" for n in (1, 2, 3, 6, 7, 8)]
    for p in built:
        assert (COMMITTED / p.name).read_text() == p.read_text(), p.name


def test_tex_bolds_and_escapes() -> None:
    """The markdown is the contract; the .tex mirror must carry the same emphasis and
    survive the underscores every arm and task name contains."""
    assert tables._tex("**1.00 (12/12)**") == r"\textbf{1.00 (12/12)}"
    assert tables._tex("niah_multikey") == r"niah\_multikey"
