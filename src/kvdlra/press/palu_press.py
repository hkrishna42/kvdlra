"""Palu: low-rank projection of the KV cache (arXiv:2407.21118), as a press.

Palu compresses the KV cache by projecting keys and values onto a **low-rank
subspace per head-group** and storing the compact latent + a small reconstruction
basis. This is the canonical "low-rank KV cache" baseline, and the natural foil to
BUG: both are low-rank, but Palu uses a *static* per-head-group subspace (here the
Eckart--Young-optimal truncated SVD, computed once at pre-fill) whereas BUG tracks
one *streaming* rank-``r`` subspace pooled over **all** KV heads
(``n_features = head_dim * num_kv_heads``). Palu's per-head granularity is the
standard low-rank-KV operating point.

Post-hoc realization (no fine-tuning): a reconstruct-then-attend press (Mode A,
same-shape output, like :class:`BUGPress`). Per head-group of ``group`` KV heads we
form the pre-RoPE key matrix ``(group*head_dim, T)`` and its value matrix, truncate
each to rank ``r = round(rank_ratio * group * head_dim)``, and write back the rank-r
reconstruction (keys re-rotated to post-RoPE). ``rank_ratio in (0, 1]``;
``rank_ratio = 1`` is lossless. Memory is counted honestly by
:func:`kvdlra.accounting.palu_footprint` (per-token latent ``r*T`` + basis
``r*group*head_dim``, K+V) -- **not** the same-shape DynamicCache tensor, which is
uncompressed by construction (Mode A).

Faithfulness note: real Palu low-rank-decomposes the projection *weights* offline
(and can fine-tune / quantize). This post-hoc *activation*-SVD is the Eckart--Young
upper bound on that scheme -- a strong, principled low-rank baseline -- and is
labelled as such. Single-shot pre-fill only (inherits :class:`BUGPress`'s guard).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from kvdlra.press.bug_press import BUGPress


def _svd_lowrank_recon(mat: torch.Tensor, r: int) -> torch.Tensor:
    """Best rank-``r`` reconstruction of a ``(features, tokens)`` matrix via truncated
    SVD (Eckart--Young), computed in fp32 (SVD is unstable in bf16, PLAN §8 #4)."""
    u, s, vh = torch.linalg.svd(mat.to(torch.float32), full_matrices=False)
    r = min(r, int(s.shape[0]))
    return cast(torch.Tensor, ((u[:, :r] * s[:r]) @ vh[:r]).to(mat.dtype))


@dataclass
class PaluPress(BUGPress):
    """Low-rank-projection press (Palu-style). ``rank_ratio`` sets the per-group rank
    ``r = round(rank_ratio * group * head_dim)``; ``group`` KV heads share a subspace
    (``group = 1`` = per-head, the default). Reuses :class:`BUGPress`'s pre-RoPE
    round-trip + single-shot-prefill machinery; only the low-rank step differs."""

    rank_ratio: float = 0.5
    group: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0.0 < self.rank_ratio <= 1.0):
            raise ValueError(f"rank_ratio must be in (0, 1], got {self.rank_ratio}")
        if self.group < 1:
            raise ValueError(f"group must be >= 1, got {self.group}")

    def _compress_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Per-head-group truncated-SVD low-rank reconstruction of ``(bsz, H, T, D)``."""
        bsz, h, t, d = x.shape
        out = torch.empty_like(x)
        for b in range(bsz):
            for g0 in range(0, h, self.group):
                g1 = min(h, g0 + self.group)
                gd = (g1 - g0) * d
                r = max(1, min(gd, round(self.rank_ratio * gd)))
                blk = x[b, g0:g1].permute(0, 2, 1).reshape(gd, t)  # (group*D, T)
                recon = _svd_lowrank_recon(blk, r)
                out[b, g0:g1] = recon.reshape(g1 - g0, d, t).permute(0, 2, 1).to(x.dtype)
        return out
