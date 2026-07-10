"""ProductQuantizer: shared, calibrated product quantization of coordinate vectors.

Product quantization (Jegou, Douze, Schmid, "Product Quantization for Nearest
Neighbor Search," IEEE TPAMI 2011) splits a ``dim``-vector into ``M`` contiguous
subvectors and quantizes each independently against a per-subspace codebook of
``K = 2**bits`` centroids fit by k-means. A vector is stored as ``M`` integer
codes (``M * bits`` bits total); the ``M`` codebooks are **shared side
information** (``sum_m K * subdim_m = K * dim`` floats), calibrated once on a
held-out corpus and amortized across every coded vector and every layer.

Role in kvdlra (Week 8, CodeBUG): the streaming BUG cache summarizes a retained
token as an ``r``-dimensional coordinate column ``C[:, s] = U^T k_s``. Storing
that column as ``r`` fp32 floats is the per-token overhead that lets whole-token
eviction win the extreme-compression regime (``docs/week7-dominance.md``).
Replacing it with a product-quantized **code** costs ``M * bits / 32``
float-equivalents (e.g. ``M=8, bits=8`` -> 8 bytes = 2 float-eq) instead of
``r`` floats -- an order-of-magnitude more coverage at a fixed memory budget.

API parity with :class:`kvdlra.quant.polar.PolarQuant`: ``quantize`` /
``dequantize`` / ``distortion`` behave the same way (rows are the quantized
vectors), so the two are drop-in interchangeable in the cache's quantized tier.
PQ additionally exposes ``fit`` (calibration), ``encode`` / ``decode`` (the
native code interface), and ``codebook_numel`` (the once-per-cache side-info
float count for honest memory accounting).

Distortion: unlike PolarQuant, PQ has no closed-form data-oblivious bound -- its
quality depends on the calibration distribution matching the deployment one. The
guarantee we test (``tests/test_product_quant.py``) is the operational one:
distortion decreases monotonically with ``bits`` (finer codebooks) and with
``subspaces`` (more, smaller subvectors), and a calibrated PQ beats an
uncalibrated scalar baseline on the same data.

Mixed precision (PLAN §8 #4): centroids/codebook math is fp32; only the integer
``codes`` (uint8) are the compact representation.

NOTE: library code -- no ``print``/I/O (the calibration *script* owns I/O).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

__all__ = ["ProductQuantizer"]


def _kmeans(
    samples: Tensor,
    n_clusters: int,
    *,
    iters: int,
    generator: torch.Generator,
) -> Tensor:
    """k-means (k-means++ init + Lloyd) on ``(N, d)`` rows -> ``(K, d)`` centroids.

    Deterministic given ``generator``. If ``N <= n_clusters`` the samples are
    padded by repetition so every centroid is seeded (degenerate but valid; the
    calibration corpus is always far larger than ``K`` in practice).
    """
    n, _d = samples.shape
    if n < n_clusters:
        reps = (n_clusters + n - 1) // n + 1
        samples = samples.repeat(reps, 1)
        n = samples.shape[0]

    # --- k-means++ seeding (D^2 sampling), deterministic via ``generator``. ---
    first = int(torch.randint(0, n, (1,), generator=generator).item())
    centroids = samples[first : first + 1].clone()  # (1, d)
    closest_sq = ((samples - centroids[0]) ** 2).sum(dim=1)  # (N,)
    for _ in range(1, n_clusters):
        total = float(closest_sq.sum())
        if total <= 0.0:  # all points already coincide with a centroid
            idx = int(torch.randint(0, n, (1,), generator=generator).item())
        else:
            probs = closest_sq / closest_sq.sum()
            idx = int(torch.multinomial(probs, 1, generator=generator).item())
        centroids = torch.cat([centroids, samples[idx : idx + 1]], dim=0)
        new_sq = ((samples - samples[idx]) ** 2).sum(dim=1)
        closest_sq = torch.minimum(closest_sq, new_sq)

    # --- Lloyd iterations (vectorized assignment + mean update). ---
    for _ in range(iters):
        dists = torch.cdist(samples, centroids)  # (N, K)
        assign = dists.argmin(dim=1)  # (N,)
        moved = False
        for k in range(n_clusters):
            mask = assign == k
            if bool(mask.any()):
                mean = samples[mask].mean(dim=0)
                if not torch.allclose(mean, centroids[k], atol=1e-9):
                    centroids[k] = mean
                    moved = True
            else:
                # Re-seed an empty cluster at the worst-fit point (keeps K live).
                worst = int(dists.gather(1, assign[:, None]).squeeze(1).argmax())
                centroids[k] = samples[worst]
                moved = True
        if not moved:
            break
    return centroids


class ProductQuantizer:
    """Calibrated product quantizer (Jegou et al. 2011); PolarQuant-parity API.

    Parameters
    ----------
    dim:
        Vector dimension ``d`` (the last axis of quantized tensors). For the BUG
        coordinate tier this is the rank ``r``.
    bits:
        Bits per subspace code; each subspace codebook has ``K = 2**bits``
        centroids. Codes are ``uint8`` so ``1 <= bits <= 8``.
    subspaces:
        Number ``M`` of contiguous subvectors ``dim`` is split into (as evenly as
        possible; the first ``dim % M`` subspaces get one extra coordinate). Total
        code cost per vector is ``M * bits`` bits.
    normalize:
        If ``True``, unit-normalize each vector and PQ the direction, returning
        the norm separately (fp32, exact) -- matches PolarQuant's sphere model and
        makes magnitude exact at the cost of one extra float per vector. If
        ``False`` (default), PQ the raw vector so the centroids carry magnitude
        (cheapest; relies on calibration covering the magnitude distribution).
    seed:
        Seed for the deterministic k-means fit.
    device, dtype:
        Where/what precision to hold the centroids (fp32 recommended).
    """

    def __init__(
        self,
        dim: int,
        bits: int = 8,
        subspaces: int = 8,
        *,
        normalize: bool = False,
        seed: int = 0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if not 1 <= bits <= 8:
            raise ValueError(f"bits must be in [1, 8] (uint8 codes), got {bits}")
        if not 1 <= subspaces <= dim:
            raise ValueError(f"subspaces must be in [1, dim={dim}], got {subspaces}")
        self.dim = int(dim)
        self.bits = int(bits)
        self.subspaces = int(subspaces)
        self.levels = 2**bits  # K centroids per subspace
        self.normalize = bool(normalize)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype

        # Contiguous subspace boundaries; first (dim % M) subspaces are 1 longer.
        base, extra = divmod(self.dim, self.subspaces)
        bounds = [0]
        for m in range(self.subspaces):
            bounds.append(bounds[-1] + base + (1 if m < extra else 0))
        self.splits: list[tuple[int, int]] = [
            (bounds[m], bounds[m + 1]) for m in range(self.subspaces)
        ]
        # One (K, subdim_m) centroid tensor per subspace; empty until ``fit``.
        self.centroids: list[Tensor] = []

    # ------------------------------------------------------------------ fit ---
    def fit(self, samples: Tensor, *, iters: int = 25, max_fit: int = 50_000) -> ProductQuantizer:
        """Calibrate the ``M`` codebooks by k-means on ``samples`` ``(N, dim)``.

        ``samples`` are the vectors to be quantized (e.g. captured BUG coordinate
        columns, rows = tokens). At most ``max_fit`` rows (deterministically
        strided) are used per subspace to bound fit time/memory. Returns ``self``.
        """
        if samples.ndim != 2 or samples.shape[1] != self.dim:
            raise ValueError(f"samples must be (N, {self.dim}), got {tuple(samples.shape)}")
        x = samples.to(device=self.device, dtype=self.dtype)
        if self.normalize:
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if x.shape[0] > max_fit:
            stride = x.shape[0] // max_fit
            x = x[::stride][:max_fit]
        self.centroids = []
        for m, (a, b) in enumerate(self.splits):
            sub = x[:, a:b].cpu()  # k-means on CPU for determinism
            gen_m = torch.Generator(device="cpu").manual_seed(self.seed + m)
            cents = _kmeans(sub, self.levels, iters=iters, generator=gen_m)
            self.centroids.append(cents.to(device=self.device, dtype=self.dtype))
        return self

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has populated the codebooks."""
        return len(self.centroids) == self.subspaces

    def _require_fit(self) -> None:
        if not self.fitted:
            raise RuntimeError("ProductQuantizer used before fit(); call fit(samples) first")

    # --------------------------------------------------------------- encode ---
    def encode(self, x: Tensor) -> Tensor:
        """Encode rows of ``x`` ``(..., dim)`` to ``uint8`` codes ``(..., M)``."""
        self._require_fit()
        if x.shape[-1] != self.dim:
            raise ValueError(f"last dim must be {self.dim}, got {x.shape[-1]}")
        xc = x.to(device=self.device, dtype=self.dtype)
        codes = torch.empty((*xc.shape[:-1], self.subspaces), dtype=torch.uint8, device=self.device)
        for m, (a, b) in enumerate(self.splits):
            sub = xc[..., a:b]  # (..., subdim)
            flat = sub.reshape(-1, b - a)  # (P, subdim)
            dists = torch.cdist(flat, self.centroids[m])  # (P, K)
            codes[..., m] = dists.argmin(dim=1).reshape(sub.shape[:-1]).to(torch.uint8)
        return codes

    def decode(self, codes: Tensor) -> Tensor:
        """Reconstruct vectors ``(..., dim)`` from ``uint8`` codes ``(..., M)``."""
        self._require_fit()
        if codes.shape[-1] != self.subspaces:
            raise ValueError(f"last dim must be M={self.subspaces}, got {codes.shape[-1]}")
        idx = codes.to(torch.int64)
        parts = [self.centroids[m][idx[..., m]] for m in range(self.subspaces)]
        return torch.cat(parts, dim=-1)

    # ---------------------------------------------------- PolarQuant parity ---
    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize rows of ``x`` ``(..., dim)`` to ``(codes, norm)``.

        ``codes`` are ``uint8`` ``(..., M)``; ``norm`` is ``(..., 1)`` -- the
        per-vector Euclidean norm when ``normalize`` is set (stored fp32, exact),
        else all-ones (raw PQ carries magnitude in the centroids). Mirrors
        :meth:`PolarQuant.quantize` so the two are interchangeable.
        """
        if x.shape[-1] != self.dim:
            raise ValueError(f"last dim must be {self.dim}, got {x.shape[-1]}")
        xc = x.to(device=self.device, dtype=self.dtype)
        if self.normalize:
            norm = xc.norm(dim=-1, keepdim=True)
            codes = self.encode(xc / norm.clamp_min(1e-12))
            return codes, norm
        codes = self.encode(xc)
        norm = torch.ones((*xc.shape[:-1], 1), dtype=self.dtype, device=self.device)
        return codes, norm

    def dequantize(self, codes: Tensor, norm: Tensor) -> Tensor:
        """Reconstruct ``x_hat`` from ``(codes, norm)`` (inverse of :meth:`quantize`)."""
        recon = self.decode(codes)
        if self.normalize:
            recon = recon / recon.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return cast(Tensor, recon * norm)
        return recon

    def distortion(self, x: Tensor) -> float:
        """Mean squared reconstruction error ``E||x - x_hat||^2`` over rows of ``x``."""
        codes, norm = self.quantize(x)
        x_hat = self.dequantize(codes, norm)
        return float(((x.to(self.dtype) - x_hat) ** 2).sum(dim=-1).mean())

    # -------------------------------------------------------- accounting/io ---
    def codebook_numel(self) -> int:
        """Float count of the shared codebooks (``sum_m K * subdim_m = K * dim``).

        This is the once-per-cache side information -- counted a single time in
        the matched-memory audit regardless of how many vectors/layers use it.
        """
        return int(self.levels * self.dim)

    def state(self) -> dict[str, object]:
        """Serializable state (for the calibration script to save/load)."""
        return {
            "dim": self.dim,
            "bits": self.bits,
            "subspaces": self.subspaces,
            "normalize": self.normalize,
            "seed": self.seed,
            "centroids": [c.cpu() for c in self.centroids],
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, object],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> ProductQuantizer:
        """Rebuild a fitted quantizer from :meth:`state`."""
        pq = cls(
            dim=int(cast(int, state["dim"])),
            bits=int(cast(int, state["bits"])),
            subspaces=int(cast(int, state["subspaces"])),
            normalize=bool(cast(bool, state["normalize"])),
            seed=int(cast(int, state["seed"])),
            device=device,
            dtype=dtype,
        )
        cents = cast("list[Tensor]", state["centroids"])
        pq.centroids = [c.to(device=pq.device, dtype=pq.dtype) for c in cents]
        return pq
