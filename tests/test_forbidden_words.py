"""CLAUDE.md: DLRA (as a word), 'BUG integrator', honest(ly), marquee, flagship never appear
in src/, configs/, scripts/, docs/paper/. `kvdlra` / `kv-dlra` identifiers are not hits."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"(?<![\w-])(dlra|marquee|flagship|honest(?:ly)?|bug integrator)\b", re.I)
SCOPE = ("src", "configs", "scripts", "docs/paper")


def test_no_forbidden_words() -> None:
    hits = []
    for top in SCOPE:
        for p in (ROOT / top).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".sh", ".md", ".tex", ".txt"}:
                for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                    if PAT.search(line):
                        hits.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:80]}")
    assert not hits, "\n".join(hits)
