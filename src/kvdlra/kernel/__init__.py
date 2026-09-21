"""The factored-attention kernel (ADR 0001 option (iii)): decode-time attention that
materializes each K/V tile from the tracked ``(U, C)`` inside attention instead of
reconstructing the whole middle. ``kvdlra.kernel.reference`` is the torch tile loop (the
numerics contract, CPU-tested); ``kvdlra.kernel.triton_kernel`` the GPU one, imported only
when a CUDA tensor asks for it, so this package imports without triton."""

from __future__ import annotations

import importlib.util

import torch
from torch import Tensor

from kvdlra.kernel import reference
from kvdlra.kernel.reference import FactoredMiddle, rope_cos_sin

__all__ = ["FactoredMiddle", "factored_attention", "rope_cos_sin"]


def _triton_available() -> bool:
    return (
        importlib.util.find_spec("triton") is not None
        and importlib.util.find_spec("kvdlra.kernel.triton_kernel") is not None
    )


def factored_attention(
    query: Tensor,
    dense_k: Tensor,
    dense_v: Tensor,
    mid: FactoredMiddle | None,
    *,
    inv_freq: Tensor,
    attention_scaling: float,
    scaling: float,
    operand_dtype: torch.dtype = torch.bfloat16,
    tile: int = 64,
    backend: str = "auto",
) -> Tensor:
    """`reference.factored_attention`, or the Triton kernel on a CUDA query when
    ``backend="auto"`` and triton is installed (``"reference"`` / ``"triton"`` force one)."""
    if backend == "auto":
        backend = (
            "triton"
            if query.is_cuda and operand_dtype is torch.bfloat16 and _triton_available()
            else "reference"
        )
    if backend == "triton":
        # GPU-only: the import needs triton, and the module is absent from a CPU tree --
        # `import_module` keeps this file type-checkable there.
        triton_kernel = importlib.import_module("kvdlra.kernel.triton_kernel")
        out: Tensor = triton_kernel.factored_attention(
            query, dense_k, dense_v, mid, inv_freq=inv_freq, attention_scaling=attention_scaling,
            scaling=scaling, operand_dtype=operand_dtype, tile=tile,
        )  # fmt: skip
        return out
    if backend != "reference":
        raise ValueError(f"backend must be auto/reference/triton, got {backend!r}")
    return reference.factored_attention(
        query, dense_k, dense_v, mid, inv_freq=inv_freq, attention_scaling=attention_scaling,
        scaling=scaling, operand_dtype=operand_dtype, tile=tile,
    )  # fmt: skip
