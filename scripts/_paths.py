"""Make the in-repo ``kvdlra`` package importable when scripts run directly.

When these scripts are launched as files (``python scripts/foo.py`` or
``uv run python scripts/foo.py``), ``sys.path[0]`` is the ``scripts/`` directory,
so ``import kvdlra`` resolves only if the package is installed. This module
**always** prepends *this* repo's ``src/`` directory, so a script run from a git
worktree imports that worktree's sources -- an editable install's ``.pth`` points
at whichever checkout was installed, which is the wrong tree for every other one.

Import this module (``import _paths``) before importing anything from ``kvdlra``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_kvdlra_importable() -> None:
    """Prepend this repo's ``<repo>/src`` to ``sys.path``, ahead of any install."""
    src_dir = Path(__file__).resolve().parent.parent / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))


_ensure_kvdlra_importable()
