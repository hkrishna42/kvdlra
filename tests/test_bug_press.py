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


@pytest.mark.parametrize(("backend", "atol"), [("numpy", 1e-9), ("torch", 1e-4)])
def test_full_rank_recovers_input(backend: str, atol: float) -> None:
    # rank_cap >= number of reconstructed columns => the tracked subspace spans
    # all payload columns => the projection is the identity (exact recovery).
    # (torch backend computes in fp32, hence the looser tolerance.)
    t = 40
    press = BUGPress(rank=t, n_sink=4, backend=backend)  # rank >= t - n_sink
    x = _random_kv(bsz=1, t=t, seed=1)
    out = press._compress_tensor(x)
    assert torch.allclose(out, x, atol=atol)


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


@pytest.mark.parametrize(("backend", "atol"), [("numpy", 1e-8), ("torch", 1e-4)])
def test_exact_rank_r_input_reconstructed_exactly(backend: str, atol: float) -> None:
    # An exactly rank-r feature-by-token matrix must be reconstructed exactly by
    # a rank-r tracker (mirrors tests/test_streaming.py).
    r, t = 5, 50
    rng = np.random.default_rng(3)
    left = rng.standard_normal((N_FEATURES, r))
    right = rng.standard_normal((r, t))
    mat = left @ right  # exactly rank r, shape (512, T)
    # Reshape into (1, H, T, D) using the inverse of the press's own reshape.
    x = torch.from_numpy(mat).reshape(H, D, t).permute(0, 2, 1).unsqueeze(0)
    press = BUGPress(rank=r, n_sink=0, backend=backend)
    out = press._compress_tensor(x)
    assert torch.allclose(out, x, atol=atol)


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


def test_quant_path_runs_and_is_lossier_than_fp() -> None:
    # TurboQuant on the coordinates: same shape, and (on full-rank data) lossier
    # than the un-quantized fp reconstruction but still a sane approximation.
    x = _random_kv(bsz=1, t=200, seed=11).to(torch.float32)
    fp = BUGPress(rank=32)
    q4 = BUGPress(rank=32, quant_bits=4)
    out_fp, out_q = fp._compress_tensor(x), q4._compress_tensor(x)
    assert out_q.shape == x.shape
    err_fp = (out_fp - x).norm() / x.norm()
    err_q = (out_q - x).norm() / x.norm()
    assert err_fp < err_q < 1.0  # quantization adds loss, but not garbage


def test_quant_more_bits_less_error() -> None:
    x = _random_kv(bsz=1, t=200, seed=12).to(torch.float32)
    errs = []
    for b in (2, 3, 5):
        out = BUGPress(rank=32, quant_bits=b)._compress_tensor(x)
        errs.append(float((out - x).norm() / x.norm()))
    assert errs[0] > errs[1] > errs[2]


def test_quant_requires_torch_backend() -> None:
    with pytest.raises(ValueError, match="quant_bits requires backend='torch'"):
        BUGPress(rank=16, quant_bits=4, backend="numpy")
    with pytest.raises(ValueError, match="quant_bits must be"):
        BUGPress(rank=16, quant_bits=0)


def test_compress_rejects_partial_cache_forward() -> None:
    # Guard against silent keys/values desync: if a q_len>1 forward does not
    # cover the whole cache (chunked/continued prefill), compress must raise
    # rather than truncate. The check fires before any module access, so a
    # dummy module/kwargs suffices.
    press = BUGPress(rank=8)
    hidden = torch.zeros(1, 8, 16)  # current forward: q_len = 8
    keys = _random_kv(bsz=1, t=16)  # but the cache already holds 16 tokens
    values = _random_kv(bsz=1, t=16)
    with pytest.raises(NotImplementedError, match="single-shot"):
        press.compress(cast(Any, None), hidden, keys, values, cast(Any, None), {})
