"""PolarQuant: rotated per-coordinate scalar quantization (TurboQuant §2).

Quantizes a vector ``x`` by (1) normalizing to the unit sphere, (2) applying a
**fixed, data-oblivious random rotation** ``Pi`` (the QR factor of a Gaussian
matrix) so the vector is uniform on the sphere and its coordinates become
near-independent and Beta-distributed, and (3) applying an **MSE-optimal 1-D
scalar quantizer (Lloyd--Max)** to each coordinate independently. The rotation
turns a hard vector-quantization problem into ``d`` easy scalar ones with almost
no loss because, on the sphere, the coordinates are near-independent
(TurboQuant, arXiv:2504.19874 §2).

Guarantee (their Thm 1): for unit vectors the normalized distortion obeys
``E||u - u_hat||^2 <= (sqrt(3) pi / 2) * 4^{-b} approx 2.7 * 4^{-b}`` at ``b``
bits/coordinate, i.e. within the constant ``2.7`` of the information-theoretic
floor ``4^{-b}``. ``tests/test_polar.py`` checks the measured distortion against
this bound.

Deviation from the paper's "3.5 bits/channel": we use **integer** bit-widths
``b`` (the codebook has ``2^b`` levels); non-integer rates in the paper come from
mixed/entropy schemes we do not need for the memory-budget sweep (we vary the
integer ``b`` and the BUG rank ``r`` and report the actual bit cost).

Mixed precision (PLAN §8 #4): the rotation/codebook math is fp32; only the
integer ``codes`` are the compact representation.

NOTE: library code -- no ``print``/I/O.

Reference: A. Zandieh, M. Daliri, M. Hadian, V. Mirrokni, "TurboQuant: Online
Vector Quantization with Near-optimal Distortion Rate," arXiv:2504.19874, §2.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

__all__ = ["PolarQuant"]


def _fit_lloyd_max(samples: Tensor, levels: int, iters: int = 50) -> Tensor:
    """1-D Lloyd--Max quantizer: ``levels`` centroids minimizing MSE on ``samples``.

    Standard Lloyd iteration (== 1-D k-means): assign each sample to its nearest
    centroid, move each centroid to the mean of its cell, repeat. Initialized at
    evenly spaced sample quantiles (deterministic given ``samples``).
    """
    s, _ = torch.sort(samples.flatten())
    # Initialize centroids at the midpoints of ``levels`` equal-mass quantile bins.
    qs = (torch.arange(levels, dtype=s.dtype, device=s.device) + 0.5) / levels
    centroids = torch.quantile(s, qs)
    for _ in range(iters):
        boundaries = (centroids[1:] + centroids[:-1]) / 2  # (levels-1,)
        idx = torch.bucketize(s, boundaries)  # cell index per sample
        new = centroids.clone()
        for k in range(levels):
            mask = idx == k
            if bool(mask.any()):
                new[k] = s[mask].mean()
        if torch.allclose(new, centroids, atol=1e-9):
            centroids = new
            break
        centroids = new
    return centroids


class PolarQuant:
    """Rotated per-coordinate Lloyd--Max scalar quantizer (TurboQuant §2).

    Parameters
    ----------
    dim:
        Vector dimension ``d`` (quantized vectors are the last-axis rows).
    bits:
        Bits per coordinate ``b``; the codebook has ``2^b`` Lloyd--Max levels.
    seed:
        Seed for the shared rotation ``Pi`` and the Monte-Carlo Lloyd fit
        (makes the quantizer bit-reproducible; ``Pi``/codebook are *not* stored
        per-vector -- they are shared side information).
    n_fit_samples:
        Number of unit-sphere coordinate samples used to fit the Lloyd--Max
        codebook for the (Beta) coordinate marginal.
    device, dtype:
        Where/what precision to build ``Pi`` and the codebook (fp32 recommended).
    """

    def __init__(
        self,
        dim: int,
        bits: int,
        *,
        seed: int = 0,
        n_fit_samples: int = 200_000,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        self.dim = int(dim)
        self.bits = int(bits)
        self.levels = 2**bits
        self.device = torch.device(device)
        self.dtype = dtype

        gen = torch.Generator(device="cpu").manual_seed(seed)
        # Fixed rotation Pi = Q from QR of a Gaussian matrix (orthonormal).
        g = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
        q, _ = torch.linalg.qr(g)
        self.Pi = q.to(device=self.device, dtype=self.dtype)  # (d, d)

        # Coordinate marginal of a uniform unit vector: sample and Lloyd-fit.
        gs = torch.randn(n_fit_samples, dim, generator=gen, dtype=torch.float64)
        coords = (gs / gs.norm(dim=1, keepdim=True))[:, 0]  # one coord per sample
        centroids = _fit_lloyd_max(coords, self.levels)
        self.centroids = centroids.to(device=self.device, dtype=self.dtype)  # (levels,)
        self.boundaries = (self.centroids[1:] + self.centroids[:-1]) / 2  # (levels-1,)

    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize rows of ``x`` (``(..., d)``) to ``(codes, norm)``.

        ``codes`` are integer level indices ``(..., d)`` in ``[0, 2^b)``; ``norm``
        is the per-vector Euclidean norm ``(..., 1)`` (stored full precision).
        """
        if x.shape[-1] != self.dim:
            raise ValueError(f"last dim must be {self.dim}, got {x.shape[-1]}")
        xc = x.to(self.dtype)
        norm = xc.norm(dim=-1, keepdim=True)
        u = xc / norm.clamp_min(1e-12)
        u_rot = u @ self.Pi.mT  # rotate rows: (Pi u) per vector
        codes = torch.bucketize(u_rot, self.boundaries)  # (..., d) in [0, levels)
        return codes.to(torch.int64), norm

    def dequantize(self, codes: Tensor, norm: Tensor) -> Tensor:
        """Reconstruct ``x_hat`` from ``(codes, norm)`` (inverse of :meth:`quantize`)."""
        u_rot = self.centroids[codes]  # (..., d)
        u = u_rot @ self.Pi  # inverse rotation (Pi orthonormal: Pi^T == Pi^{-1})
        return cast(Tensor, u * norm)

    def distortion(self, x: Tensor) -> float:
        """Mean squared reconstruction error ``E||x - x_hat||^2`` over rows of ``x``."""
        codes, norm = self.quantize(x)
        x_hat = self.dequantize(codes, norm)
        return float(((x.to(self.dtype) - x_hat) ** 2).sum(dim=-1).mean())
