"""The factored-attention kernel (ADR 0001 option (iii)): decode-time attention that
materializes each K/V tile from the tracked ``(U, C)`` inside attention instead of
reconstructing the whole middle. ``kvdlra.kernel.reference`` is the torch tile loop (the
numerics contract, CPU-tested); ``kvdlra.kernel.triton_kernel`` the GPU one, imported only
when a CUDA tensor asks for it, so this package imports without triton."""

from __future__ import annotations

__all__: list[str] = []
