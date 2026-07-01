"""QJL: 1-bit Quantized Johnson--Lindenstrauss residual quantizer (TurboQuant §3).

MSE-optimal scalar quantizers (:class:`kvdlra.quant.polar.PolarQuant`) *bias*
inner-product estimates. TurboQuant removes that bias by quantizing the residual
with a **1-bit sign hash** whose dequantization gives an **unbiased** inner-product
estimator.

Construction (their Def. 1)
---------------------------
Draw a fixed Gaussian sketch ``S in R^{m x d}`` (i.i.d. ``N(0,1)``, seeded/shared).

    * quantize:   ``Q(x)   = sign(S x) in {-1,+1}^m``   (1 bit per row; direction
      only -- ``sign`` is scale-invariant, so the norm ``||x||`` is stored aside),
    * dequantize: ``Q^{-1}(z) = sqrt(pi/2) / m * S^T z``.

Then for any query ``y`` (with ``a = S x``, ``b = S y`` jointly Gaussian,
``E[b_i sign(a_i)] = sqrt(2/pi) <x,y> / ||x||``):

    ``E[ <y, Q^{-1}(Q(x))> ] = <y, x / ||x||>``      (unbiased **direction** dot),

so the unbiased estimator of the full inner product is
``<y, x> ~ ||x|| * <y, Q^{-1}(Q(x))>`` -- exactly the ``||r|| * <y, Q_qjl^{-1}
(Q_qjl(r))>`` residual term in TurboQuant's two-stage estimator.

Variance (their Lemma 4): ``Var( <y, Q^{-1}(Q(x))> ) <= (pi / 2m) ||y||^2``.

``tests/test_qjl.py`` checks both the unbiasedness and the variance bound.

NOTE: library code -- no ``print``/I/O. fp32 math (PLAN §8 #4).

Reference: A. Zandieh, M. Daliri, M. Hadian, V. Mirrokni, "TurboQuant",
arXiv:2504.19874, §3; and the original QJL, arXiv:2406.03482.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["QJL"]


class QJL:
    """1-bit Quantized-JL sign hash + unbiased inner-product estimator.

    Parameters
    ----------
    dim:
        Input dimension ``d``.
    m:
        Number of sketch rows / sign bits per vector (default ``dim``). More rows
        -> lower estimator variance (``~ 1/m``).
    seed:
        Seed for the shared Gaussian sketch ``S`` (bit-reproducible; ``S`` is
        shared side information, not stored per-vector).
    device, dtype:
        Where/what precision to build ``S`` (fp32 recommended).
    """

    def __init__(
        self,
        dim: int,
        m: int | None = None,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = int(dim)
        self.m = int(m) if m is not None else int(dim)
        if self.m < 1:
            raise ValueError(f"m must be >= 1, got {self.m}")
        self.device = torch.device(device)
        self.dtype = dtype
        gen = torch.Generator(device="cpu").manual_seed(seed)
        s = torch.randn(self.m, dim, generator=gen, dtype=torch.float64)
        self.S = s.to(device=self.device, dtype=self.dtype)  # (m, d)
        self._scale = math.sqrt(math.pi / 2.0) / self.m

    def sign_hash(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(codes, norm)``: ``codes = sign(S x) in {-1,+1}^m`` (int8) and
        the per-vector norm ``||x||`` (stored full precision)."""
        if x.shape[-1] != self.dim:
            raise ValueError(f"last dim must be {self.dim}, got {x.shape[-1]}")
        xc = x.to(self.dtype)
        norm = xc.norm(dim=-1, keepdim=True)
        proj = xc @ self.S.mT  # (..., m)
        codes = torch.where(proj >= 0, 1, -1).to(torch.int8)
        return codes, norm

    def dequantize_direction(self, codes: Tensor) -> Tensor:
        """``Q^{-1}(z) = sqrt(pi/2)/m * S^T z`` -- an unbiased estimate of the unit
        direction ``x/||x||`` (in the inner-product sense ``E<y, .> = <y, x/||x||>``)."""
        z = codes.to(self.dtype)
        return self._scale * (z @ self.S)  # (..., d)

    def estimate_inner(self, y: Tensor, codes: Tensor, norm: Tensor) -> Tensor:
        """Unbiased estimate of ``<y, x>`` from ``x``'s ``(codes, norm)``.

        ``<y, x> ~ ||x|| * <y, Q^{-1}(Q(x))>``. ``y`` broadcasts against the code
        batch; the dot is over the feature axis.
        """
        direction = self.dequantize_direction(codes)  # (..., d), estimates x/||x||
        inner_dir = (y.to(self.dtype) * direction).sum(dim=-1, keepdim=True)  # <y, x/||x||>
        return (norm * inner_dir).squeeze(-1)
