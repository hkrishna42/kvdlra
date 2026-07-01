"""Unit tests for :class:`kvdlra.press.bug_press.BUGPress`.

These are hermetic (no model download): they exercise the math-bearing internals
-- the per-batch reshape/reconstruct (:meth:`BUGPress._compress_tensor`), the
sink-preserving low-rank reconstruction (:meth:`BUGPress._lowrank_reconstruct`),
the nominal ``compression_ratio``, and config-driven ``n_features`` -- plus the
constructor guards. End-to-end generation through the kvpress hook is validated
separately by ``scripts/generate_with_press.py`` (it loads a real model).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from kvdlra.press import BUGPress

# Llama-3.2-1B shape constants (docs/notes/conventions.md).
H, D = 8, 64
N_FEATURES = H * D  # 512


def _random_kv(bsz: int, t: int, seed: int = 0) -> torch.Tensor:
    """A ``(bsz, H, T, D)`` key/value-shaped tensor of float64 noise."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(bsz, H, t, D, generator=g, dtype=torch.float64)


def test_shape_preserved() -> None:
    press = BUGPress(rank=8)
    x = _random_kv(bsz=2, t=40)
    out = press._compress_tensor(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_full_rank_recovers_input() -> None:
    # rank_cap >= number of reconstructed columns => the tracked subspace spans
    # all payload columns => the projection is the identity (exact recovery).
    t = 40
    press = BUGPress(rank=t, n_sink=4)  # rank >= t - n_sink
    x = _random_kv(bsz=1, t=t, seed=1)
    out = press._compress_tensor(x)
    assert torch.allclose(out, x, atol=1e-9)


def test_sinks_preserved_exactly_at_low_rank() -> None:
    # The first n_sink token columns must be byte-for-byte preserved even when
    # the low-rank model is very aggressive.
    n_sink = 4
    press = BUGPress(rank=2, n_sink=n_sink)
    x = _random_kv(bsz=1, t=60, seed=2)
    out = press._compress_tensor(x)
    assert torch.allclose(out[:, :, :n_sink, :], x[:, :, :n_sink, :], atol=0.0)
    # ... while the non-sink region is genuinely altered by the rank-2 model.
    assert not torch.allclose(out[:, :, n_sink:, :], x[:, :, n_sink:, :])


def test_exact_rank_r_input_reconstructed_exactly() -> None:
    # An exactly rank-r feature-by-token matrix must be reconstructed exactly by
    # a rank-r tracker (mirrors tests/test_streaming.py).
    r, t = 5, 50
    rng = np.random.default_rng(3)
    left = rng.standard_normal((N_FEATURES, r))
    right = rng.standard_normal((r, t))
    mat = left @ right  # exactly rank r, shape (512, T)
    # Reshape into (1, H, T, D) using the inverse of the press's own reshape.
    x = torch.from_numpy(mat).reshape(H, D, t).permute(0, 2, 1).unsqueeze(0)
    press = BUGPress(rank=r, n_sink=0)
    out = press._compress_tensor(x)
    assert torch.allclose(out, x, atol=1e-8)


def test_low_rank_is_lossy_on_full_rank_input() -> None:
    press = BUGPress(rank=4, n_sink=0)
    x = _random_kv(bsz=1, t=80, seed=4)  # full-rank noise
    out = press._compress_tensor(x)
    rel_err = (out - x).norm() / x.norm()
    assert 0.0 < rel_err < 1.0  # genuinely compressed, not exact, not garbage


def test_short_context_noop() -> None:
    # T <= n_sink: nothing to compress, returned unchanged.
    press = BUGPress(rank=8, n_sink=4)
    x = _random_kv(bsz=1, t=3)
    out = press._compress_tensor(x)
    assert torch.allclose(out, x, atol=0.0)


def test_compression_ratio_value_and_readonly() -> None:
    press = BUGPress(rank=32, n_sink=4)
    press.n_features = 512
    assert press.compression_ratio == pytest.approx(1.0 - 32 / 512)
    with pytest.raises(AttributeError):
        press.compression_ratio = 0.5


def test_post_init_from_model_sets_n_features() -> None:
    # No model download: a duck-typed config is all post_init_from_model reads.
    cfg = SimpleNamespace(
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        hidden_size=2048,
    )
    press = BUGPress(rank=16)
    press.post_init_from_model(cast(Any, SimpleNamespace(config=cfg)))
    assert press.n_features == 512


def test_constructor_guards() -> None:
    with pytest.raises(ValueError, match="rank"):
        BUGPress(rank=0)
    with pytest.raises(ValueError, match="n_sink"):
        BUGPress(rank=8, n_sink=-1)
