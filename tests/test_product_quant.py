"""Tests for :class:`kvdlra.quant.product_quant.ProductQuantizer` (Jegou 2011).

Product quantization has no data-oblivious distortion bound (unlike PolarQuant),
so the guarantees we check are operational: valid codes, round-trip shape,
distortion **monotone decreasing** in ``bits`` (finer codebooks) and in
``subspaces`` (more, smaller subvectors), calibration determinism, exact norm
carry in ``normalize`` mode, and honest codebook accounting. These are the
properties the CodeBUG cache and its matched-memory audit rely on.
"""

from __future__ import annotations

import pytest
import torch

from kvdlra.quant import ProductQuantizer

DIM = 24  # the BUG rank r used at the aggressive budget


def _clustered(n: int, d: int, seed: int, n_modes: int = 12) -> torch.Tensor:
    """Data with per-coordinate cluster structure (PQ's favourable regime)."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_modes, d, generator=g) * 3.0
    pick = torch.randint(0, n_modes, (n,), generator=g)
    return centers[pick] + 0.5 * torch.randn(n, d, generator=g)


def test_roundtrip_shape_and_codes() -> None:
    x = _clustered(4000, DIM, seed=0)
    pq = ProductQuantizer(dim=DIM, bits=4, subspaces=6, seed=1).fit(x, iters=10)
    y = torch.randn(5, 3, DIM)  # arbitrary leading dims
    codes, norm = pq.quantize(y)
    y_hat = pq.dequantize(codes, norm)
    assert codes.shape == (5, 3, 6) and codes.dtype == torch.uint8
    assert int(codes.min()) >= 0 and int(codes.max()) < pq.levels
    assert norm.shape == (5, 3, 1)
    assert y_hat.shape == y.shape


def test_encode_decode_native() -> None:
    x = _clustered(3000, DIM, seed=2)
    pq = ProductQuantizer(dim=DIM, bits=4, subspaces=8, seed=1).fit(x, iters=10)
    codes = pq.encode(x)
    assert codes.shape == (3000, 8) and codes.dtype == torch.uint8
    recon = pq.decode(codes)
    assert recon.shape == x.shape
    # decode(encode(x)) is exactly dequantize(quantize(x)) in raw (non-norm) mode.
    q_codes, q_norm = pq.quantize(x)
    assert torch.equal(codes, q_codes)
    assert torch.allclose(recon, pq.dequantize(q_codes, q_norm))


def test_distortion_monotone_in_bits() -> None:
    train = _clustered(6000, DIM, seed=3)
    test = _clustered(2000, DIM, seed=4)
    errs = [
        ProductQuantizer(dim=DIM, bits=b, subspaces=6, seed=1).fit(train, iters=12).distortion(test)
        for b in (2, 3, 4, 5)
    ]
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)), errs


def test_distortion_monotone_in_subspaces() -> None:
    # More subspaces at fixed bits/subspace = more total code bits = lower error.
    train = _clustered(6000, DIM, seed=5)
    test = _clustered(2000, DIM, seed=6)
    errs = [
        ProductQuantizer(dim=DIM, bits=4, subspaces=m, seed=1).fit(train, iters=12).distortion(test)
        for m in (2, 3, 6, 12)
    ]
    assert all(errs[i] >= errs[i + 1] for i in range(len(errs) - 1)), errs


def test_calibration_helps() -> None:
    # A PQ fit on-distribution beats one fit on unrelated (shifted) data.
    train = _clustered(6000, DIM, seed=7)
    test = _clustered(2000, DIM, seed=8)
    wrong = torch.randn(6000, DIM) * 10.0 + 50.0  # different distribution
    good = ProductQuantizer(dim=DIM, bits=4, subspaces=6, seed=1).fit(train, iters=12)
    bad = ProductQuantizer(dim=DIM, bits=4, subspaces=6, seed=1).fit(wrong, iters=12)
    assert good.distortion(test) < bad.distortion(test)


def test_high_capacity_low_error() -> None:
    x = _clustered(6000, DIM, seed=9, n_modes=8)
    pq = ProductQuantizer(dim=DIM, bits=6, subspaces=12, seed=1).fit(x, iters=20)
    rel = pq.distortion(x) / float((x**2).sum(dim=-1).mean())
    assert rel < 0.05  # high-bit, many-subspace PQ recovers clustered data well


def test_fit_deterministic() -> None:
    x = _clustered(3000, DIM, seed=10)
    a = ProductQuantizer(dim=DIM, bits=4, subspaces=6, seed=42).fit(x, iters=10)
    b = ProductQuantizer(dim=DIM, bits=4, subspaces=6, seed=42).fit(x, iters=10)
    for ca, cb in zip(a.centroids, b.centroids, strict=True):
        assert torch.equal(ca, cb)


def test_normalize_scale_invariance_and_exact_norm() -> None:
    x = _clustered(4000, DIM, seed=11)
    pq = ProductQuantizer(dim=DIM, bits=4, subspaces=6, normalize=True, seed=1).fit(x, iters=12)
    y = torch.randn(500, DIM, generator=torch.Generator().manual_seed(12))

    def rel_mse(v: torch.Tensor) -> float:
        c, n = pq.quantize(v)
        vh = pq.dequantize(c, n)
        return float(((v - vh).pow(2).sum(1) / v.pow(2).sum(1)).mean())

    assert rel_mse(y) == pytest.approx(rel_mse(y * 100.0), rel=1e-4)
    # In normalize mode the stored norm is the exact per-vector norm.
    codes, norm = pq.quantize(y)
    assert torch.allclose(norm.squeeze(-1), y.norm(dim=-1), atol=1e-4)
    # Reconstructed vectors preserve that norm exactly (direction-only quantized).
    y_hat = pq.dequantize(codes, norm)
    assert torch.allclose(y_hat.norm(dim=-1), y.norm(dim=-1), atol=1e-3)


def test_raw_mode_norm_is_ones() -> None:
    x = _clustered(2000, DIM, seed=13)
    pq = ProductQuantizer(dim=DIM, bits=4, subspaces=6, normalize=False, seed=1).fit(x, iters=8)
    _codes, norm = pq.quantize(x[:10])
    assert torch.allclose(norm, torch.ones_like(norm))


def test_codebook_numel() -> None:
    pq = ProductQuantizer(dim=DIM, bits=8, subspaces=8, seed=1)
    assert pq.codebook_numel() == 256 * DIM  # K * dim, independent of M
    pq2 = ProductQuantizer(dim=DIM, bits=4, subspaces=3, seed=1)
    assert pq2.codebook_numel() == 16 * DIM


def test_uneven_subspaces() -> None:
    # dim=24 with subspaces=5 -> splits [5,5,5,5,4] (first dim%M get one extra... here 24%5=4).
    pq = ProductQuantizer(dim=24, bits=3, subspaces=5, seed=1)
    sizes = [b - a for a, b in pq.splits]
    assert sizes == [5, 5, 5, 5, 4] and sum(sizes) == 24
    x = _clustered(2000, 24, seed=14)
    pq.fit(x, iters=8)
    codes = pq.encode(x[:7])
    assert codes.shape == (7, 5)
    assert pq.decode(codes).shape == (7, 24)


def test_state_roundtrip() -> None:
    x = _clustered(2000, DIM, seed=15)
    pq = ProductQuantizer(dim=DIM, bits=4, subspaces=6, seed=1).fit(x, iters=8)
    rebuilt = ProductQuantizer.from_state(pq.state())
    assert rebuilt.fitted
    assert torch.equal(pq.encode(x[:20]), rebuilt.encode(x[:20]))


def test_guards() -> None:
    with pytest.raises(ValueError, match="dim"):
        ProductQuantizer(dim=0, bits=4, subspaces=2)
    with pytest.raises(ValueError, match="bits"):
        ProductQuantizer(dim=8, bits=0, subspaces=2)
    with pytest.raises(ValueError, match="bits"):
        ProductQuantizer(dim=8, bits=9, subspaces=2)
    with pytest.raises(ValueError, match="subspaces"):
        ProductQuantizer(dim=8, bits=4, subspaces=9)
    with pytest.raises(RuntimeError, match="fit"):
        ProductQuantizer(dim=8, bits=4, subspaces=2).encode(torch.randn(3, 8))
