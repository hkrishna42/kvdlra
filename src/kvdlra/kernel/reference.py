"""The factored-attention tile loop in plain torch (ADR 0001 option (iii)) -- the numerics
contract `kvdlra.kernel.triton_kernel` reproduces on CUDA, and the CPU-tested backend.

Per tile of ``tile`` low-rank columns, per KV head: ``K̂ = U_k[head] @ C_k[:, tile]`` and
``V̂`` likewise, with U and C rounded to ``operand_dtype`` (bf16 on the pod; passed AS STORED
and rounded per tile, so no per-step cast copy) and fp32 accumulation; RoPE on K̂ in fp32 at
the tile's TRUE positions from the model's own ``inv_freq`` and ``attention_scaling`` (the
same products ``LlamaRotaryEmbedding.forward`` forms: ``cos = cat(ang, ang).cos() *
attention_scaling``, ``ang = pos * inv_freq``); K̂ and V̂ rounded to ``operand_dtype`` for the
score and PV dots (the tensor-core inputs), scores and the online softmax (running max and
sum) in fp32, P rounded to ``operand_dtype`` before PV -- FlashAttention's rounding points.
The dense tokens (sinks, exact tier, recent ring, the new token: post-RoPE, model dtype) are
further tiles of the same loop, so nothing n-dimensional is ever written for the middle.

Layouts: ``U`` rows are head-major (``h*D + d``, `bug_cache._to_mat`); query heads ``h*G +
j`` attend KV head ``h`` (`repeat_kv`); decode only (``q_len == 1``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers.models.llama.modeling_llama import rotate_half

__all__ = ["FactoredMiddle", "factored_attention", "rope_cos_sin"]


@dataclass(frozen=True)
class FactoredMiddle:
    """The low-rank middle of one or more batch rows, as stored: bases ``(B, n, r)``,
    coordinates ``(B, r, T)`` (the quantized tier, if any, already dequantized into them --
    `BugStreamingLayer._mid_coords`), true positions ``(B, T)`` (int)."""

    u_k: Tensor
    u_v: Tensor
    c_k: Tensor
    c_v: Tensor
    positions: Tensor

    def __post_init__(self) -> None:
        b, n, r = self.u_k.shape
        if self.u_v.shape != (b, n, r):
            raise ValueError(f"u_v {tuple(self.u_v.shape)} != u_k {(b, n, r)}")
        t = self.c_k.shape[-1]
        if self.c_k.shape != (b, r, t) or self.c_v.shape != (b, r, t):
            raise ValueError(f"c_k/c_v must be (B, r, T) = {(b, r, t)}")
        if self.positions.shape != (b, t) or self.positions.is_floating_point():
            raise ValueError(f"positions must be an int (B, T) = {(b, t)} tensor")

    @property
    def n_columns(self) -> int:
        return int(self.c_k.shape[-1])

    @staticmethod
    def cat(mids: Sequence[FactoredMiddle]) -> FactoredMiddle:
        """Batch rows' middles (each B = 1) stacked into one (B = len(mids)) middle.

        ponytail: a per-step copy of every row's C (r x T fp32, 8 MB per layer at 32K r64);
        pass per-row pointers to the kernel if batch > 1 decode ever shows it."""
        ranks = [int(m.u_k.shape[-1]) for m in mids]
        if len(set(ranks)) > 1:
            raise ValueError(f"rows have different live ranks (min_sv_frac > 0): {ranks}")
        return FactoredMiddle(
            torch.cat([m.u_k for m in mids]), torch.cat([m.u_v for m in mids]),
            torch.cat([m.c_k for m in mids]), torch.cat([m.c_v for m in mids]),
            torch.cat([m.positions for m in mids]),
        )  # fmt: skip


def rope_cos_sin(
    positions: Tensor, inv_freq: Tensor, attention_scaling: float
) -> tuple[Tensor, Tensor]:
    """fp32 ``(cos, sin)`` of shape ``(B, M, D)`` at int positions ``(B, M)``, exactly as
    ``LlamaRotaryEmbedding.forward`` forms them (its ``inv_freq @ positions`` matmul is one
    product per element, so the outer product below is bit-identical)."""
    ang = positions.to(torch.float32)[..., None] * inv_freq.to(torch.float32)
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos() * attention_scaling, emb.sin() * attention_scaling


class _OnlineSoftmax:
    """Running (max, sum, numerator) of one softmax over tiles, fp32."""

    def __init__(self, q: Tensor, scaling: float, operand_dtype: torch.dtype) -> None:
        self.q, self.scaling, self.od = q, scaling, operand_dtype  # q: (B, H_kv, G, D) fp32
        b, h, g, _ = q.shape
        self.m = q.new_full((b, h, g), float("-inf"))
        self.l = q.new_zeros((b, h, g))
        self.acc = q.new_zeros(q.shape)

    def push(self, k: Tensor, v: Tensor) -> None:
        """One tile: ``k``, ``v`` are ``(B, H_kv, M, D)`` fp32, already rounded through the
        operand dtype."""
        s = torch.einsum("bhgd,bhmd->bhgm", self.q, k) * self.scaling
        m_new = torch.maximum(self.m, s.amax(-1))
        alpha = torch.exp(self.m - m_new)
        p = torch.exp(s - m_new[..., None])
        self.l = alpha * self.l + p.sum(-1)
        p_rounded = p.to(self.od).to(torch.float32)
        self.acc = alpha[..., None] * self.acc + torch.einsum("bhgm,bhmd->bhgd", p_rounded, v)
        self.m = m_new

    def finish(self) -> Tensor:
        return self.acc / self.l[..., None]


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
) -> Tensor:
    """Decode attention of ``query`` ``(B, H_q, 1, D)`` over the dense tokens ``dense_k`` /
    ``dense_v`` ``(B, H_kv, L, D)`` and the factored middle, returned as ``(B, H_q, 1, D)``
    in ``query.dtype``. ``mid=None`` (or an empty middle) is plain attention over the dense
    tokens. ``scaling`` is the softmax scale (``head_dim ** -0.5``)."""
    b, h_q, q_len, d = query.shape
    if q_len != 1:
        raise ValueError(f"decode-only: q_len must be 1, got {q_len}")
    h_kv = int(dense_k.shape[1])
    if h_q % h_kv:
        raise ValueError(f"{h_q} query heads do not group over {h_kv} KV heads")
    if tile < 1:
        raise ValueError(f"tile must be >= 1, got {tile}")
    g = h_q // h_kv
    od = operand_dtype

    def rounded(x: Tensor) -> Tensor:
        return x.to(od).to(torch.float32)

    run = _OnlineSoftmax(rounded(query[:, :, 0].reshape(b, h_kv, g, d)), scaling, od)
    if mid is not None and mid.n_columns:
        n, r = int(mid.u_k.shape[1]), int(mid.u_k.shape[2])
        if n != h_kv * d:
            raise ValueError(f"U has {n} rows, expected H_kv * D = {h_kv * d}")
        u_k = rounded(mid.u_k).reshape(b, h_kv, d, r)
        u_v = rounded(mid.u_v).reshape(b, h_kv, d, r)
        for s0 in range(0, mid.n_columns, tile):
            c_k = rounded(mid.c_k[:, :, s0 : s0 + tile])  # (B, r, M)
            c_v = rounded(mid.c_v[:, :, s0 : s0 + tile])
            k_hat = torch.einsum("bhdr,brm->bhmd", u_k, c_k)  # (B, H_kv, M, D), fp32 accumulate
            v_hat = torch.einsum("bhdr,brm->bhmd", u_v, c_v)
            cos, sin = rope_cos_sin(mid.positions[:, s0 : s0 + tile], inv_freq, attention_scaling)
            rot = rotate_half(k_hat)  # type: ignore[no-untyped-call]  # unannotated upstream
            k_hat = k_hat * cos[:, None] + rot * sin[:, None]
            run.push(rounded(k_hat), rounded(v_hat))
    for s0 in range(0, int(dense_k.shape[2]), tile):
        run.push(rounded(dense_k[:, :, s0 : s0 + tile]), rounded(dense_v[:, :, s0 : s0 + tile]))
    return run.finish().reshape(b, h_q, 1, d).to(query.dtype)
