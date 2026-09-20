"""The single statistics module. Wilson + exact McNemar are the paper's verified tools
(ported from ``scripts/w15_intervals.py`` and ``w18_intervals.py`` @
``paper-v1-archive``); Holm/BH/bootstrap/TOST are
the corrections the review panel asked for. scipy does the arithmetic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import numpy.typing as npt
from scipy.stats import binomtest, false_discovery_control, ttest_1samp
from scipy.stats import t as student_t

Key = tuple[int, int]  # (seed, trial)


class McNemar(TypedDict):
    n_paired: int
    a_favored: int
    b_favored: int
    p_value: float


def wilson(hits: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n <= 0 or not 0 <= hits <= n:
        raise ValueError(f"bad cell: hits={hits} n={n}")
    ci = binomtest(hits, n).proportion_ci(confidence_level=conf, method="wilson")
    return float(ci.low), float(ci.high)


def mcnemar_exact(a: dict[Key, int], b: dict[Key, int]) -> McNemar | None:
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    a_f = sum(1 for k in shared if a[k] == 1 and b[k] == 0)
    b_f = sum(1 for k in shared if a[k] == 0 and b[k] == 1)
    p = 1.0 if a_f + b_f == 0 else float(binomtest(min(a_f, b_f), a_f + b_f, 0.5).pvalue)
    return {"n_paired": len(shared), "a_favored": a_f, "b_favored": b_f, "p_value": p}


def holm(p: Sequence[float]) -> list[float]:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def bh(p: Sequence[float]) -> list[float]:
    adj = false_discovery_control(np.asarray(p, dtype=float), method="bh")
    return [float(x) for x in adj]


def paired_bootstrap(
    d: npt.ArrayLike, n_boot: int = 10_000, seed: int = 0, conf: float = 0.95
) -> tuple[float, float, float]:
    x = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(x.mean()), float(lo), float(hi)


def tost_decidable(d: npt.ArrayLike, delta: float, alpha: float = 0.05) -> bool:
    """Could :func:`tost` have fired at all on this spread? ``t(1-alpha, n-1)*s/sqrt(n) <
    delta``, which is the equivalence condition at its most favourable point estimate
    (d_bar = 0). ``False`` means the interval is too wide to place against the margin, so
    the TOST's ``False`` is "not decidable" rather than "not equivalent" -- a third state
    the reading has to keep apart (``prereg/gate1_tracker_swap_v2.md`` section 6)."""
    x = np.asarray(d, dtype=float)
    half = float(student_t.ppf(1 - alpha, x.size - 1)) * float(x.std(ddof=1)) / np.sqrt(x.size)
    return bool(half < delta)


def tost(d: npt.ArrayLike, delta: float, alpha: float = 0.05) -> tuple[float, float, bool]:
    x = np.asarray(d, dtype=float)
    p_lo = float(ttest_1samp(x, -delta, alternative="greater").pvalue)
    p_hi = float(ttest_1samp(x, delta, alternative="less").pvalue)
    return p_lo, p_hi, max(p_lo, p_hi) < alpha
