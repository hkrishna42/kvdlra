"""Every .py under src/ and scripts/ must be in the static import closure of the four
entrypoints + tests (+ what scripts/pod/*.sh invokes). Unreachable code is deleted, not kept."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, SCR, TESTS = ROOT / "src", ROOT / "scripts", ROOT / "tests"
ENTRY = [SCR / f"{n}.py" for n in ("pod", "tables", "figures", "dump_kv")]


def _resolve(name: str) -> Path | None:
    for base in (SRC, SCR):
        p = base / Path(*name.split("."))
        if p.with_suffix(".py").exists():
            return p.with_suffix(".py")
        if (p / "__init__.py").exists():
            return p / "__init__.py"
    return None


def _with_packages(path: Path) -> set[Path]:
    """``path`` plus every package ``__init__.py`` above it, because importing
    ``kvdlra.tracker.isvd`` executes ``kvdlra/__init__.py`` and
    ``kvdlra/tracker/__init__.py`` first. A package marker nobody names directly is
    reached, not orphaned -- deleting one to satisfy this test would make the package a
    namespace package on the pod's editable install, which is a real change, not a
    cleanup."""
    out, d = {path}, path.parent
    while d not in (SRC, SCR) and (SRC in d.parents or SCR in d.parents):
        if (d / "__init__.py").exists():
            out.add(d / "__init__.py")
        d = d.parent
    return out


def _imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text())
    out: set[Path] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for n in names:
            r = _resolve(n)
            if r:
                out |= _with_packages(r)
    return out


def test_everything_is_reachable() -> None:
    sh_invoked = {SCR / m for sh in (SCR / "pod").glob("*.sh")
                  for m in re.findall(r"scripts/([A-Za-z0-9_]+\.py)", sh.read_text())}  # fmt: skip
    seen: set[Path] = set()
    todo = set(ENTRY) | set(TESTS.glob("test_*.py")) | sh_invoked
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        todo |= _imports(p) - seen
    all_py = {p for base in (SRC, SCR) for p in base.rglob("*.py") if "__pycache__" not in p.parts}
    unreachable = sorted(str(p.relative_to(ROOT)) for p in all_py - seen)
    assert not unreachable, f"unreachable: {unreachable}"
