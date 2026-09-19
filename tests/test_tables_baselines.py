"""L2.4 (PR-L2-13): the `baselines` table surfaces the two v1 rows the paper omitted --
ShadowKV post-fix at Llama 16K (`shadow-r64`, 0.815x) and plain eviction at 0.25x
(`ea-k0.25`) -- from the archived aggregate cells. Those pre-Week-18 rows carry no
``hits``/``n``, so the table prints the archived point estimates with ``n`` shown as
``---`` and NO Wilson interval: an interval needs an n the record does not hold."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import tables

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


def test_a_duplicate_archive_row_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two archived rows for one (arm, task, ctx) would silently overwrite each other in
    the task-keyed dict; like the function's other guards, it refuses instead."""
    real = tables._cells

    def with_twin(pod: str) -> tuple[Any, ...]:
        rows = real(pod)
        twin = next(r for r in rows if r["arm"] == "shadow-r64" and r["ctx"] == tables.K16)
        return (*rows, dict(twin)) if pod == "w15-confirm" else rows

    monkeypatch.setattr(tables, "_cells", with_twin)
    with pytest.raises(SystemExit, match=r"duplicate .*w15-confirm shadow-r64"):
        tables.table_baselines()
