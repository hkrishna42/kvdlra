"""`make tables` regenerates the paper-v1 tables: the build output must equal the
hand-transcribed golden (docs/plan/paper-v1-tables.md), byte for byte."""

import subprocess
import sys
from pathlib import Path


def test_make_tables_matches_golden(tmp_path: Path) -> None:
    out = tmp_path / "tables"
    subprocess.run([sys.executable, "scripts/tables.py", "build", "--out", str(out)], check=True)
    got = "".join(p.read_text() for p in sorted(out.glob("table*.md")))
    assert got == Path("docs/plan/paper-v1-tables.md").read_text()
