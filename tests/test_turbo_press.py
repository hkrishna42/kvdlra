"""Hermetic tests for :class:`kvdlra.press.turbo_press.TurboQuantPress`."""

from __future__ import annotations

import pytest
import torch

from kvdlra.press import TurboQuantPress

H, D = 8, 64


def _kv(t: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, H, t, D, generator=g, dtype=torch.float32)


def test_shape_preserved_and_lossy() -> None:
    press = TurboQuantPress(bits=4)
    x = _kv(60)
    out = press._quant(x)
    assert out.shape == x.shape
    rel = float((out - x).norm() / x.norm())
    assert 0.0 < rel < 1.0  # genuinely quantized, not exact, not garbage


def test_more_bits_less_error() -> None:
    x = _kv(80, seed=1)
    errs = [float((TurboQuantPress(bits=b)._quant(x) - x).norm() / x.norm()) for b in (2, 3, 5)]
    assert errs[0] > errs[1] > errs[2]


def test_compression_ratio_is_zero_and_fixed() -> None:
    press = TurboQuantPress(bits=4)
    assert press.compression_ratio == 0.0
    press.compression_ratio = 0.0  # allowed (no-op, ComposedPress may set it)
    with pytest.raises(AttributeError):
        press.compression_ratio = 0.5


def test_bits_guard() -> None:
    with pytest.raises(ValueError, match="bits"):
        TurboQuantPress(bits=0)
