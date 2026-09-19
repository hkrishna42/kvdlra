"""``palu`` (case-insensitive) is confined to the SVD-oracle module's own disclaimer
and the archive keys that carry the retired legacy arm string ``palu-r0.5`` (CLAUDE.md
settled facts; DECISIONS D-004): the press module's docstring, the arm YAML's
``legacy_name`` + doc disclaimer, ``scripts/tables.py``'s v1 row keys, and the golden's
archived top-level entry.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED = (
    "src/kvdlra/baselines/svd_oracle.py",  # the disclaimer
    "configs/arms/svd_oracle_r0.5.yaml",  # legacy_name + doc disclaimer
    "scripts/tables.py",  # v1 row keys (archived records)
    "tests/golden/legacy_arm_kwargs.json",  # the archived arm string
    "tests/test_palu_rename.py",
)


def test_palu_appears_only_where_allowed() -> None:
    out = subprocess.run(
        ["git", "grep", "-n", "-i", "palu", "--", "src", "configs", "scripts", "tests"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.splitlines()
    bad = [line for line in out if not line.startswith(ALLOWED)]
    assert not bad, "\n".join(bad)
