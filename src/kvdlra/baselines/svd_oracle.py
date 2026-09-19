"""Per-sequence, per-head truncated SVD of the prefill K and V (rank ratio
``rank_ratio``), sinks kept exact. It is an *upper bound* for static low-rank
methods on this sequence (Eckart--Young in Frobenius norm on the sequence's own
K/V) -- it is not Palu (no weight decomposition, no grouped heads, no Fisher rank
allocation, no fine-tuning, cannot be computed online). The paper-v1 records name
it ``palu-r0.5``; that name is retired.

Mechanics: a reconstruct-then-attend press (Mode A, same-shape output, like
:class:`BUGPress`). Per group we form the pre-RoPE key matrix ``(group*head_dim, T)``
and its value matrix, keep the ``n_sink`` leading token columns **exact** (the
:class:`BUGPress` sink exemption -- only columns ``n_sink:`` are reconstructed;
Week-15 audit fix), truncate the rest to rank ``r = round(rank_ratio * group *
head_dim)``, and write back the rank-r reconstruction (keys re-rotated to post-RoPE).
``rank_ratio in (0, 1]``; ``rank_ratio = 1`` is lossless. Memory is billed by
:func:`kvdlra.accounting.lowrank_footprint` (exact sinks + per-token latent
``r*(T - n_sink)`` + basis ``r*group*head_dim``, K+V) -- **not** the same-shape
DynamicCache tensor, which is uncompressed by construction (Mode A). Single-shot
pre-fill only (inherits :class:`BUGPress`'s guard).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from kvdlra.baselines.lowrank_press import BUGPress


def _svd_lowrank_recon(mat: torch.Tensor, r: int) -> torch.Tensor:
    """Best rank-``r`` reconstruction of a ``(features, tokens)`` matrix via truncated
    SVD (Eckart--Young), computed in fp32 (SVD is unstable in bf16, PLAN §8 #4)."""
    u, s, vh = torch.linalg.svd(mat.to(torch.float32), full_matrices=False)
    r = min(r, int(s.shape[0]))
    return cast(torch.Tensor, ((u[:, :r] * s[:r]) @ vh[:r]).to(mat.dtype))


@dataclass
class SVDOraclePress(BUGPress):
    """Static truncated-SVD press. ``rank_ratio`` sets the per-group rank
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
        """Per-head-group truncated-SVD low-rank reconstruction of ``(bsz, H, T, D)``.

        The ``n_sink`` leading token columns (attention sinks) are kept **exact**
        and only columns ``n_sink:`` are low-ranked -- the same contract as
        :class:`BUGPress` ("only columns ``n_sink:`` are reconstructed"), applied
        to K and V alike. Week-15 audit fix: without this carve-out this arm was the
        ONLY frontier arm that low-ranked the sinks; the high-norm sink columns
        dominate each group's spectrum, so the truncated SVD spent its rank on
        them and wrecked fluency (the incoherent-for-Eckart--Young 1B palu-r0.5
        ppl signature). The carve-out is inline by design -- calling
        ``BUGPress._lowrank_reconstruct`` would drag BUG's streaming-projector
        semantics into this static Eckart--Young baseline.
        """
        bsz, h, t, d = x.shape
        out = x.clone()  # sinks (columns :n_sink) stay bit-exact
        s = min(self.n_sink, t)
        if t <= s:
            return out  # only sinks present; nothing to compress
        for b in range(bsz):
            for g0 in range(0, h, self.group):
                g1 = min(h, g0 + self.group)
                gd = (g1 - g0) * d
                r = max(1, min(gd, round(self.rank_ratio * gd)))
                blk = x[b, g0:g1].permute(0, 2, 1).reshape(gd, t)  # (group*D, T)
                recon = _svd_lowrank_recon(blk[:, s:], r)  # non-sink block only
                out[b, g0:g1, s:] = recon.reshape(g1 - g0, d, t - s).permute(0, 2, 1).to(x.dtype)
        return out
