"""The factored-attention kernel (ADR 0001 option (iii)): decode-time attention that
materializes each K/V tile from the tracked ``(U, C)`` inside attention instead of
reconstructing the whole middle. ``kvdlra.kernel.reference`` is the torch tile loop (the
numerics contract, CPU-tested); ``kvdlra.kernel.triton_kernel`` the GPU one, imported only
when a CUDA tensor asks for it, so this package imports without triton.

The Triton backend's launch geometry and shape refusals are stated HERE, not beside the
kernel: ``triton_kernel`` cannot be imported without triton, and ``backend="auto"`` has to
decide whether the kernel would accept a call BEFORE importing it. ``G_PAD``,
``TARGET_PROGRAMS``, ``n_splits_for`` and ``_shapes_eligible`` are integer arithmetic over
shapes, `triton_kernel` imports them back (one definition, no drift), and both the split
count and the choice of backend are testable on a CPU (R-L4-21, R-L4-22).
"""

from __future__ import annotations

import importlib.util

import torch
from torch import Tensor

from kvdlra.kernel import reference
from kvdlra.kernel.reference import FactoredMiddle, rope_cos_sin

__all__ = [
    "G_PAD",
    "TARGET_PROGRAMS",
    "FactoredMiddle",
    "factored_attention",
    "n_splits_for",
    "rope_cos_sin",
    "select_backend",
]

G_PAD = 16  # tl.dot needs M >= 16: the G query heads of one KV head are padded to 16 rows
TARGET_PROGRAMS = 128  # ~ the SM count (A100 108, H100 132): B*H_kv*n_splits reaches it


def _pow2(x: int, at_least: int) -> bool:
    return x >= at_least and (x & (x - 1)) == 0


def n_splits_for(b: int, h_kv: int, n_tiles: int) -> int:
    """How many programs the Triton backend splits one KV head's factored tiles across:
    enough that ``B * H_kv * n_splits`` reaches `TARGET_PROGRAMS` (batch 1 on 8 KV heads
    would otherwise leave a tenth of the card idle), never more splits than there are
    tiles, and never fewer than one -- split 0 also carries the dense tokens, so it runs
    even when the middle is empty."""
    return max(1, min(n_tiles, -(-TARGET_PROGRAMS // (b * h_kv))))


def _shapes_eligible(query: Tensor, mid: FactoredMiddle | None, h_kv: int) -> bool:
    """Whether `kvdlra.kernel.triton_kernel.factored_attention` accepts these shapes -- its
    own refusals, minus the dtype and device ones: ``H_q / H_kv <= G_PAD``, ``D`` a power of
    two >= 32 (the kernel's index vectors need power-of-two extents), and, where there are
    columns to factor, ``r`` a power of two >= 16. ``h_kv`` is the dense K/V's head count,
    which is where the kernel reads its own ``h_kv`` from."""
    h_q, d = int(query.shape[1]), int(query.shape[-1])
    if h_q % h_kv or h_q // h_kv > G_PAD or not _pow2(d, 32):
        return False
    return mid is None or not mid.n_columns or _pow2(int(mid.u_k.shape[-1]), 16)


def _triton_eligible(
    query: Tensor, dense_k: Tensor, mid: FactoredMiddle | None, operand_dtype: torch.dtype
) -> bool:
    """Whether ``backend="auto"`` hands this call to the Triton kernel: a CUDA bf16 call,
    triton importable, and shapes the kernel accepts.

    R-L4-22: the shape test is what keeps ``"auto"`` from raising the kernel's ValueError
    mid-decode. ``min_sv_frac > 0`` shrinks the live rank to an arbitrary integer and r192
    arms exist, so a rank the kernel refuses is reachable -- ``"auto"`` degrades to the
    reference there. An explicit ``backend="triton"`` still raises."""
    return (
        query.is_cuda
        and operand_dtype is torch.bfloat16
        and _triton_available()
        and _shapes_eligible(query, mid, int(dense_k.shape[1]))
    )


def _triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


def select_backend(
    query: Tensor,
    dense_k: Tensor,
    mid: FactoredMiddle | None,
    operand_dtype: torch.dtype,
    backend: str = "auto",
) -> str:
    """Which backend `factored_attention` would run this call on: ``"triton"`` or
    ``"reference"``, never ``"auto"``.

    Pure, and the ONE place the choice is made, so a caller can attest what attended
    (`kvdlra.kernel.attention` records it on the cache, `kvdlra.eval.kernel_check` and
    `kvdlra.eval.latency` print it). R-L4-22 lets ``"auto"`` degrade to the reference on a
    rank the kernel refuses; without this the degradation leaves no trace in the record."""
    if backend == "auto":
        return "triton" if _triton_eligible(query, dense_k, mid, operand_dtype) else "reference"
    if backend not in ("reference", "triton"):
        raise ValueError(f"backend must be auto/reference/triton, got {backend!r}")
    return backend


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
    ``backend="auto"`` and `_triton_eligible` (``"reference"`` / ``"triton"`` force one)."""
    backend = select_backend(query, dense_k, mid, operand_dtype, backend)
    if backend == "triton":
        # GPU-only: the import needs triton. `import_module` keeps this file importable
        # (and type-checkable) on a tree that has none.
        triton_kernel = importlib.import_module("kvdlra.kernel.triton_kernel")
        out: Tensor = triton_kernel.factored_attention(
            query, dense_k, dense_v, mid, inv_freq=inv_freq, attention_scaling=attention_scaling,
            scaling=scaling, operand_dtype=operand_dtype, tile=tile,
        )  # fmt: skip
        return out
    return reference.factored_attention(
        query, dense_k, dense_v, mid, inv_freq=inv_freq, attention_scaling=attention_scaling,
        scaling=scaling, operand_dtype=operand_dtype, tile=tile,
    )  # fmt: skip
