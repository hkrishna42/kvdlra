"""L2.4 (PR-L2-13): the `baselines` table surfaces the two v1 rows the paper omitted --
ShadowKV post-fix at Llama 16K (`shadow-r64`, 0.815x) and plain eviction at 0.25x
(`ea-k0.25`) -- from the archived aggregate cells. Those pre-Week-18 rows carry no
``hits``/``n``, so the table prints the archived point estimates with ``n`` shown as
``---`` and NO Wilson interval: an interval needs an n the record does not hold."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_baselines_table_has_shadow_and_ea025(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "tables.py"), "build", "--out", str(tmp_path)],
        check=True,
    )
    md = (tmp_path / "table_baselines.md").read_text()
    assert "shadow-r64" in md and "ea-k0.25" in md and "0.815" in md
    assert "[" not in md.split("\n\n", 1)[1], "no Wilson interval: the archive holds no n"
    rows = [ln for ln in md.splitlines() if ln.startswith(("| shadow-r64", "| ea-k0.25"))]
    assert len(rows) == 2 and all("| --- |" in r for r in rows)  # n shown as ---
    assert "| shadow-r64 | 0.815x | 1.00 | 1.00 | --- | 0.00 | --- |" in md
    assert "| ea-k0.25 | 0.250x | 1.00 | 0.88 | 1.00 | 0.50 | --- |" in md
