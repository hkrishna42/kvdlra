import numpy as np
import pytest

from kvdlra.eval.stats import bh, holm, mcnemar_exact, paired_bootstrap, tost, wilson


def test_wilson_12_of_12_lower_bound() -> None:
    lo, hi = wilson(12, 12)
    assert lo == pytest.approx(0.758, abs=5e-4) and hi == 1.0


def test_wilson_rejects_bad_cells() -> None:
    with pytest.raises(ValueError):
        wilson(13, 12)


def test_mcnemar_10_0_and_6_0() -> None:
    a = {(s, t): 1 for s in (0, 1) for t in range(8)}  # 16/16
    b = dict(a)
    b.update(dict.fromkeys(list(a)[:10], 0))  # 6/16, all discordant one way
    r = mcnemar_exact(a, b)
    assert r is not None and r["a_favored"] == 10 and r["b_favored"] == 0
    assert r["p_value"] == pytest.approx(1.95e-3, rel=1e-2)
    b6 = dict(a)
    b6.update(dict.fromkeys(list(a)[:6], 0))
    assert mcnemar_exact(a, b6)["p_value"] == pytest.approx(3.13e-2, rel=1e-2)  # type: ignore[index]


def test_mcnemar_no_shared_trials_is_none() -> None:
    assert mcnemar_exact({(0, 0): 1}, {(1, 0): 1}) is None


def test_holm_and_bh_known_vectors() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert bh([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_paired_bootstrap_symmetric_contains_zero() -> None:
    d = np.random.default_rng(0).normal(0.0, 1.0, 64)
    mean, lo, hi = paired_bootstrap(d, n_boot=2000, seed=0)
    assert lo < 0.0 < hi and abs(mean - d.mean()) < 1e-12


def test_tost_equivalence() -> None:
    rng = np.random.default_rng(1)
    tight = rng.normal(0.0, 0.01, 32)
    p_lo, p_hi, ok = tost(tight, delta=0.05)
    assert ok and max(p_lo, p_hi) < 0.05
    shifted = tight + 0.10
    assert not tost(shifted, delta=0.05)[2]
