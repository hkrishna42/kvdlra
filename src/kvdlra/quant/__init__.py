"""TurboQuant vector quantizers (arXiv:2504.19874): PolarQuant + QJL residual."""

from __future__ import annotations

from kvdlra.quant.polar import PolarQuant
from kvdlra.quant.product_quant import ProductQuantizer
from kvdlra.quant.qjl import QJL

__all__ = ["QJL", "PolarQuant", "ProductQuantizer"]
