"""The Triton backend of `kvdlra.kernel.factored_attention` (ADR 0001 §3 option (iii), §5):
the reference tile loop on the GPU, blocked one KV head x M tokens per program, with a
flash-decoding split of the factored tiles across programs and a torch merge.

Rounding points are the reference's: U and C tiles rounded to bf16 as loaded, K̂/V̂ by
`tl.dot` in fp32 and rounded to bf16 for the score/PV dots, RoPE in fp32 from the tile's int
positions (`ang = inv_freq * pos`, the model's own products), scores and the online softmax
in fp32 (the running SUM over the UNROUNDED p), P rounded to bf16. Only bf16 operands: fp32
operands need the reference.

Imported lazily by `kvdlra.kernel.factored_attention` (needs triton, hence CUDA).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from kvdlra.kernel.reference import FactoredMiddle

__all__ = ["G_PAD", "TARGET_PROGRAMS", "factored_attention"]

G_PAD = 16  # tl.dot needs M >= 16: the G query heads of one KV head are padded to 16 rows
TARGET_PROGRAMS = 128  # ~ the SM count (A100 108, H100 132): B*H_kv*n_splits reaches it


@triton.jit  # type: ignore[untyped-decorator]
def _tiles_kernel(  # type: ignore[no-untyped-def]
    Q, DK, DV, UK, UV, CK, CV, POS, INV_FREQ, M_OUT, L_OUT, ACC_OUT,  # noqa: N803
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_ub, stride_un, stride_ur,
    stride_cb, stride_cr, stride_ct,
    stride_pb, stride_pt,
    n_dense, n_mid, tiles_per_split, n_splits, attention_scaling, sm_scale,
    H_KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,  # noqa: N803
    R: tl.constexpr, M: tl.constexpr, GP: tl.constexpr,  # noqa: N803
):  # fmt: skip
    # Q (B, H_q, 1, D) -- the q_len dim is 1, so its stride is never passed; DK/DV
    # (B, H_kv, L, D); UK/UV (B, H_kv*D, R) head-major rows; CK/CV (B, R, T); POS (B, T) int32.
    bh = tl.program_id(0)
    split = tl.program_id(1)
    b = bh // H_KV  # grid dim 0 enumerates (b, h) with h fastest
    h = bh % H_KV
    g = tl.arange(0, GP)
    dh = tl.arange(0, HALF)
    dd = tl.arange(0, D)
    rr = tl.arange(0, R)
    mm = tl.arange(0, M)
    row_ok = (g < G)[:, None]  # (GP, 1): rows G..GP-1 are padding, dropped after the merge
    q_base = Q + b * stride_qb + (h * G + g)[:, None] * stride_qh  # query head h*G + j
    q_lo = tl.load(q_base + dh[None, :] * stride_qd, mask=row_ok, other=0.0).to(tl.bfloat16)
    q_hi = tl.load(q_base + (HALF + dh)[None, :] * stride_qd, mask=row_ok, other=0.0)
    q_hi = q_hi.to(tl.bfloat16)  # (GP, HALF): the second half of the head dim
    q_all = tl.load(q_base + dd[None, :] * stride_qd, mask=row_ok, other=0.0).to(tl.bfloat16)
    m_i = tl.full((GP,), float("-inf"), tl.float32)
    l_i = tl.zeros((GP,), tl.float32)
    acc = tl.zeros((GP, D), tl.float32)

    # --- the factored tiles of this split: K̂ = U_k[head] C_k[:, tile], RoPE'd in fp32
    inv = tl.load(INV_FREQ + dh)  # (HALF,) fp32
    u_base = UK + b * stride_ub + rr[None, :] * stride_ur
    uk_lo = tl.load(u_base + (h * D + dh)[:, None] * stride_un).to(tl.bfloat16)  # (HALF, R)
    uk_hi = tl.load(u_base + (h * D + HALF + dh)[:, None] * stride_un).to(tl.bfloat16)
    uvT = tl.load(  # noqa: N806
        UV + b * stride_ub + (h * D + dd)[None, :] * stride_un + rr[:, None] * stride_ur
    ).to(tl.bfloat16)  # (R, D): uvT[j, d] = U_v[b, h*D + d, j]
    n_tiles = tl.cdiv(n_mid, M)
    t_first = split * tiles_per_split
    t_last = tl.minimum(t_first + tiles_per_split, n_tiles)  # empty where t_last <= t_first
    for t in range(t_first, t_last):
        cols = t * M + mm  # (M,) column indices into the middle
        valid = cols < n_mid  # (M,) the last tile is partial
        ck = tl.load(
            CK + b * stride_cb + rr[:, None] * stride_cr + cols[None, :] * stride_ct,
            mask=valid[None, :],
            other=0.0,
        ).to(tl.bfloat16)  # (R, M) = C_k[b, :, tile]
        cvT = tl.load(  # noqa: N806
            CV + b * stride_cb + rr[None, :] * stride_cr + cols[:, None] * stride_ct,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)  # (M, R) = C_v[b, :, tile]^T
        k1 = tl.dot(uk_lo, ck)  # (HALF, M) fp32: pre-RoPE K̂, first half of the head dim
        k2 = tl.dot(uk_hi, ck)  # second half
        pos = tl.load(POS + b * stride_pb + cols * stride_pt, mask=valid, other=0).to(tl.float32)
        ang = inv[:, None] * pos[None, :]  # (HALF, M): the products modeling_llama.py:130 forms
        cos = tl.cos(ang) * attention_scaling
        sin = tl.sin(ang) * attention_scaling
        # x*cos + rotate_half(x)*sin, rotate_half(x) = [-x2, x1]:
        k1r = (k1 * cos - k2 * sin).to(tl.bfloat16)
        k2r = (k2 * cos + k1 * sin).to(tl.bfloat16)
        s = (tl.dot(q_lo, k1r) + tl.dot(q_hi, k2r)) * sm_scale  # (GP, M) fp32
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, 1)  # the running sum takes p UNROUNDED
        v_hat = tl.dot(cvT, uvT).to(tl.bfloat16)  # (M, D)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_hat)
        m_i = m_new

    # --- the dense tiles (sinks | exact tier | recent | new token), split 0 only
    if split == 0:
        for t in range(0, tl.cdiv(n_dense, M)):
            cols = t * M + mm
            valid = cols < n_dense
            kT = tl.load(  # noqa: N806
                DK
                + b * stride_kb
                + h * stride_kh
                + cols[None, :] * stride_kt
                + dd[:, None] * stride_kd,
                mask=valid[None, :],
                other=0.0,
            ).to(tl.bfloat16)  # (D, M): K^T of this tile, loaded transposed by the strides
            v = tl.load(
                DV
                + b * stride_kb
                + h * stride_kh
                + cols[:, None] * stride_kt
                + dd[None, :] * stride_kd,
                mask=valid[:, None],
                other=0.0,
            ).to(tl.bfloat16)  # (M, D)
            s = tl.dot(q_all, kT) * sm_scale
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = alpha * l_i + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new

    # M_OUT/L_OUT (B, H_kv, n_splits, GP); ACC_OUT (B, H_kv, n_splits, GP, D), both contiguous.
    base = (bh * n_splits + split) * GP
    tl.store(M_OUT + base + g, m_i)
    tl.store(L_OUT + base + g, l_i)
    tl.store(ACC_OUT + (base + g)[:, None] * D + dd[None, :], acc)


def _pow2(x: int, at_least: int) -> bool:
    return x >= at_least and (x & (x - 1)) == 0


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
    """`kvdlra.kernel.reference.factored_attention`, on CUDA. Constraints: bf16 operands,
    ``D`` a power of two >= 32, ``r`` a power of two >= 16, ``tile`` a power of two >= 16,
    ``H_q / H_kv <= 16`` (Llama-3.1-8B: D 128, r 64, G 4)."""
    if operand_dtype is not torch.bfloat16:
        raise ValueError(
            "the Triton backend computes with bf16 tile operands; use the reference for fp32"
        )
    if not query.is_cuda:
        raise ValueError("the Triton backend needs CUDA tensors")
    b, h_q, q_len, d = query.shape
    if q_len != 1:
        raise ValueError(f"decode-only: q_len must be 1, got {q_len}")
    h_kv = int(dense_k.shape[1])
    g = h_q // h_kv
    if h_q % h_kv or g > G_PAD:
        raise ValueError(f"{h_q} query heads over {h_kv} KV heads: need G <= {G_PAD}")
    if not _pow2(d, 32) or not _pow2(tile, 16):
        raise ValueError(f"D ({d}) and tile ({tile}) must be powers of two >= 32 and 16")
    if dense_v.shape != dense_k.shape:  # one set of strides addresses both
        raise ValueError(f"dense_v {tuple(dense_v.shape)} != dense_k {tuple(dense_k.shape)}")
    dev = query.device
    if mid is None or mid.n_columns == 0:
        n_mid, r = 0, 16
        u_k = u_v = query.new_zeros((b, h_kv * d, r))
        c_k = c_v = query.new_zeros((b, r, 1))
        pos = torch.zeros((b, 1), dtype=torch.int32, device=dev)
    else:
        n_mid, r = mid.n_columns, int(mid.u_k.shape[-1])
        if not _pow2(r, 16):
            raise ValueError(f"r ({r}) must be a power of two >= 16")
        if int(mid.u_k.shape[1]) != h_kv * d:
            raise ValueError(f"U has {int(mid.u_k.shape[1])} rows, expected {h_kv * d}")
        u_k, u_v = mid.u_k.contiguous(), mid.u_v.contiguous()
        c_k, c_v = mid.c_k.contiguous(), mid.c_v.contiguous()
        pos = mid.positions.to(torch.int32).contiguous()
    n_tiles = -(-n_mid // tile)
    n_splits = max(1, min(n_tiles, -(-TARGET_PROGRAMS // (b * h_kv))))
    tiles_per_split = -(-n_tiles // n_splits) if n_tiles else 0
    n_dense = int(dense_k.shape[2])
    q = query.contiguous()
    dk, dv = dense_k.contiguous(), dense_v.contiguous()
    m_out = torch.empty((b, h_kv, n_splits, G_PAD), dtype=torch.float32, device=dev)
    l_out = torch.empty_like(m_out)
    acc_out = torch.empty((b, h_kv, n_splits, G_PAD, d), dtype=torch.float32, device=dev)
    _tiles_kernel[(b * h_kv, n_splits)](
        q, dk, dv, u_k, u_v, c_k, c_v, pos, inv_freq.to(torch.float32).contiguous(),
        m_out, l_out, acc_out,
        q.stride(0), q.stride(1), q.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        u_k.stride(0), u_k.stride(1), u_k.stride(2),
        c_k.stride(0), c_k.stride(1), c_k.stride(2),
        pos.stride(0), pos.stride(1),
        n_dense, n_mid, tiles_per_split, n_splits, float(attention_scaling), float(scaling),
        H_KV=h_kv, G=g, D=d, HALF=d // 2, R=r, M=tile, GP=G_PAD, num_warps=4,
    )  # fmt: skip
    # The flash-decoding merge of the splits, fp32: a split that ran no tile carries
    # m = -inf, l = 0, acc = 0 and weighs exp(-inf) = 0 (split 0 always has the dense tokens).
    m_max = m_out.amax(dim=2, keepdim=True)
    w = torch.exp(m_out - m_max)  # (b, h_kv, S, GP)
    l_all = (w * l_out).sum(dim=2)  # (b, h_kv, GP)
    acc = (w[..., None] * acc_out).sum(dim=2)  # (b, h_kv, GP, d)
    out = (acc / l_all[..., None])[:, :, :g]  # drop the padded query rows
    return out.reshape(b, h_q, 1, d).to(query.dtype)
