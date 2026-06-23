"""Make the in-repo ``kvdlra`` package importable when scripts run directly.

When these scripts are launched as files (``python scripts/foo.py`` or
``uv run python scripts/foo.py``), ``sys.path[0]`` is the ``scripts/`` directory,
so ``import kvdlra`` resolves only if the package is installed. The hatchling
editable install's import hook is not reliably honored on every setup (it can be
dropped by an environment re-sync), so this module prepends the repo's ``src/``
directory to ``sys.path`` **only if** ``kvdlra`` is not already importable -- a
no-op when the install works (e.g. in CI / on the pod).

Import this module (``import _paths``) before importing anything from ``kvdlra``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _ensure_kvdlra_importable() -> None:
    """Prepend ``<repo>/src`` to ``sys.path`` if ``kvdlra`` is not importable."""
    if importlib.util.find_spec("kvdlra") is not None:
        return
    src_dir = Path(__file__).resolve().parent.parent / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))


_ensure_kvdlra_importable()
