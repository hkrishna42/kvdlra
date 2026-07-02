"""Tests for :class:`kvdlra.quant.qjl.QJL` (TurboQuant §3).

The two load-bearing guarantees:

* **Unbiased** inner-product estimator: ``E<y, Q^{-1}(Q(x))> = <y, x/||x||>``,
  so ``||x|| * <y, Q^{-1}(Q(x))>`` is an unbiased estimate of ``<y, x>``.
* **Variance bound** (Lemma 4): ``Var(<y, Q^{-1}(Q(x))>) <= (pi / 2m) ||y||^2``.

Estimates are averaged over many independent sketches (fixed seed sequence, so
the measured mean/variance are deterministic -- no flakiness).
"""

from __future__ import annotations

import math

import pytest
import torch

from kvdlra.quant.qjl import QJL

DIM = 64


def _fixed_xy(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(DIM, generator=g), torch.randn(DIM, generator=g)


def _estimator_samples(x: torch.Tensor, y: torch.Tensor, m: int, k: int) -> torch.Tensor:
    """``k`` independent estimates of ``<y, x>`` from independent QJL sketches."""
    out = []
    for s in range(k):
        q = QJL(DIM, m=m, seed=s)
        codes, norm = q.sign_hash(x)
        out.append(q.estimate_inner(y, codes, norm).item())
    return torch.tensor(out)


def test_estimator_is_unbiased() -> None:
    x, y = _fixed_xy(0)
    true = float(torch.dot(y, x))
    est = _estimator_samples(x, y, m=DIM, k=3000)
    assert float(est.mean()) == pytest.approx(true, rel=0.02)


def test_variance_matches_lemma4_bound() -> None:
    x, y = _fixed_xy(1)
    est = _estimator_samples(x, y, m=DIM, k=3000)
    # estimate_inner scales the direction estimator by ||x||; undo it to compare
    # against the direction-estimator bound pi/(2m)||y||^2 (Lemma 4).
    dir_var = float((est / float(x.norm())).var())
    bound = math.pi / (2 * DIM) * float(y.norm()) ** 2
    assert dir_var <= 1.15 * bound  # Lemma 4 upper bound (+ MC slack)
    assert dir_var >= 0.70 * bound  # ... and near-tight, not accidentally tiny


def test_variance_decreases_with_more_rows() -> None:
    x, y = _fixed_xy(2)
    variances = [float(_estimator_samples(x, y, m=m, k=1500).var()) for m in (16, 64, 256)]
    assert variances[0] > variances[1] > variances[2]


def test_sign_hash_shapes_and_values() -> None:
    q = QJL(DIM, m=32, seed=0)
    x = torch.randn(5, 3, DIM)
    codes, norm = q.sign_hash(x)
    assert codes.shape == (5, 3, 32) and codes.dtype == torch.int8
    assert set(codes.flatten().tolist()) <= {-1, 1}
    assert norm.shape == (5, 3, 1)


def test_estimate_inner_batched() -> None:
    # A batch of keys, one query: estimates should track the true inner products.
    # Seed the draw so the single-sketch correlation is deterministic regardless
    # of test order (the QJL sketch is seeded, but these operands were not).
    torch.manual_seed(0)
    q = QJL(DIM, m=DIM, seed=0)
    y = torch.randn(DIM)
    x = torch.randn(50, DIM)
    codes, norm = q.sign_hash(x)
    est = q.estimate_inner(y, codes, norm)  # (50,)
    true = x @ y
    assert est.shape == (50,)
    # single-sketch estimates are noisy but correlated with the truth
    corr = float(torch.corrcoef(torch.stack([est, true]))[0, 1])
    assert corr > 0.6


def test_guards() -> None:
    with pytest.raises(ValueError, match="dim"):
        QJL(0)
    with pytest.raises(ValueError, match="m"):
        QJL(8, m=0)
