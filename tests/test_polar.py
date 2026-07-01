"""Tests for :class:`kvdlra.quant.polar.PolarQuant` (TurboQuant §2).

The load-bearing guarantee (their Thm 1): for unit vectors the normalized
distortion is within the constant ``sqrt(3)pi/2 ~ 2.7`` of the information-
theoretic floor ``4^{-b}``. We check the measured distortion sits in
``[4^{-b}, ~2.7 * 4^{-b}]`` and the usual quantizer sanity (monotone in bits,
scale-invariant, orthonormal rotation, valid codes).
"""

from __future__ import annotations

from typing import cast

import pytest
import torch

from kvdlra.quant import PolarQuant

DIM = 64
FLOOR_CONST = 2.7  # sqrt(3) pi / 2


def _unit_vectors(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.randn(n, d, generator=torch.Generator().manual_seed(seed))
    return cast(torch.Tensor, g / g.norm(dim=1, keepdim=True))


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_distortion_within_2p7x_of_floor(bits: int) -> None:
    pq = PolarQuant(dim=DIM, bits=bits, seed=1, n_fit_samples=100_000)
    u = _unit_vectors(4000, DIM, seed=7)
    d = pq.distortion(u)
    floor = 4.0**-bits
    assert floor <= d <= (FLOOR_CONST + 0.2) * floor  # above floor, within ~2.7x


def test_distortion_monotone_in_bits() -> None:
    u = _unit_vectors(3000, DIM, seed=8)
    errs = [
        PolarQuant(dim=DIM, bits=b, seed=1, n_fit_samples=80_000).distortion(u)
        for b in (2, 3, 4, 5)
    ]
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))


def test_scale_invariance() -> None:
    # PolarQuant normalizes then stores the norm, so the *relative* error must be
    # invariant to the input scale.
    pq = PolarQuant(dim=DIM, bits=3, seed=1, n_fit_samples=80_000)
    x = torch.randn(1000, DIM, generator=torch.Generator().manual_seed(9))

    def rel_mse(v: torch.Tensor) -> float:
        c, n = pq.quantize(v)
        vh = pq.dequantize(c, n)
        return float(((v - vh).pow(2).sum(1) / v.pow(2).sum(1)).mean())

    assert rel_mse(x) == pytest.approx(rel_mse(x * 100.0), rel=1e-4)


def test_rotation_is_orthonormal() -> None:
    pq = PolarQuant(dim=DIM, bits=3, seed=2, n_fit_samples=10_000)
    ident = pq.Pi.mT @ pq.Pi
    assert torch.allclose(ident, torch.eye(DIM), atol=1e-5)


def test_roundtrip_shape_and_codes() -> None:
    pq = PolarQuant(dim=DIM, bits=4, seed=1, n_fit_samples=10_000)
    x = torch.randn(5, 3, DIM)  # arbitrary leading dims
    codes, norm = pq.quantize(x)
    x_hat = pq.dequantize(codes, norm)
    assert codes.shape == x.shape and codes.dtype == torch.int64
    assert norm.shape == (5, 3, 1)
    assert int(codes.min()) >= 0 and int(codes.max()) < pq.levels
    assert x_hat.shape == x.shape


def test_high_bits_low_error() -> None:
    pq = PolarQuant(dim=DIM, bits=6, seed=1, n_fit_samples=120_000)
    x = torch.randn(500, DIM, generator=torch.Generator().manual_seed(3))
    c, n = pq.quantize(x)
    xh = pq.dequantize(c, n)
    rel = float(((x - xh).pow(2).sum(1) / x.pow(2).sum(1)).mean())
    assert rel < 0.01  # 6 bits/coord -> <1% energy error


def test_guards() -> None:
    with pytest.raises(ValueError, match="dim"):
        PolarQuant(dim=0, bits=3)
    with pytest.raises(ValueError, match="bits"):
        PolarQuant(dim=8, bits=0)
