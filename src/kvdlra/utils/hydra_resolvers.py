"""Custom OmegaConf resolvers for Hydra configs.

Transcribed from PLAN §4 ("Config naming"). Registering the ``git_sha``
resolver lets every YAML carry ``version: ${git_sha:}`` so that every figure
traces back to a commit hash -- the core reproducibility rule of the project.
"""

from __future__ import annotations

import subprocess

from omegaconf import OmegaConf


def _git_sha() -> str:
    """Return the short git SHA of the current HEAD."""
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()


OmegaConf.register_new_resolver("git_sha", _git_sha)
