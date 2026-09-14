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


def seed_everything(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    The one caller (`scripts/dump_kv.py`) passes a seed and nothing else. The
    bit-reproducibility switch that used to sit here -- deterministic algorithms plus
    the cuBLAS workspace config -- was never passed by anything in three weeks of runs,
    so it is not a knob, it is an untested branch; a lane that needs it adds it back
    with the caller that wants it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
