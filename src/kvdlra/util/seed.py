"""Seeding utilities for reproducible runs.

Transcribed from PLAN §4 ("Seeding everything"). The BUG core should run in
fp32; only the model forward should run in bf16. See PLAN §4 determinism
caveats.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 0, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: The seed value applied to all RNGs.
        deterministic: If True, force deterministic algorithms and set the
            cuBLAS workspace config. This trades speed for bit-reproducibility
            and should be used when exact reproducibility matters.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
