"""max|Δ| < 1e-2 in bf16 vs reconstruct-then-attend (GATES §G4 line 2; prereg/kernel_smoke.md
§4; ADR 0001 §6 Week 2) -- on random tensors at the 8B shapes (n = 1024, r = 64, D = 128,
H_kv = 8, GQA 4) and on one layer of the 1B dumps, driven through the production layer.

The reference is today's decode path as `_ensure_mid_cache` + `_decode_peek` + sdpa compute
it: fp32 U@C, fp32 RoPE at the true positions, cast to the model dtype (bf16 on the pod), then
attention -- taken here as exact fp32 attention over that bf16 stored representation, so the
difference measured is the kernel's own rounding (bf16 U and C before the r-term dot, bf16
K̂/V̂/P at the tensor-core inputs). Expected ~1e-3; the bar is 1e-2. In fp32 operand mode the
two paths differ only by summation order and agree to 1e-4 -- the plumbing pin.

BACKENDS grows a gpu-marked "triton" entry in Task 5; the same assertions then run on CUDA.
The 1B-dump test is skipped where the gitignored 4.7 GB tree is absent (CI). The Triton
backend's split count and the shape refusals `backend="auto"` degrades on are integer
arithmetic over shapes, so they live in `kvdlra.kernel` and are tested here, on the CPU
(R-L4-21, R-L4-22) -- the kernel module itself cannot be imported without triton.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention as sdpa
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, rotate_half

from kvdlra.cache.bug_cache import BugStreamingLayer, _RopeAngles
from kvdlra.kernel import (
    FactoredMiddle,
    _shapes_eligible,
    _triton_eligible,
    factored_attention,
    n_splits_for,
)
from kvdlra.kernel.reference import rope_cos_sin

BACKENDS: list[object] = ["reference"]
BACKENDS.append(pytest.param("triton", marks=pytest.mark.gpu))
TOL = 1e-2  # the pre-registered tripwire (prereg/kernel_smoke.md §4)
DUMP = Path(__file__).resolve().parents[1] / "dumps/llama3.2-1b/doc411_047d060d_len4096_rope-both"
N, R, D, H_KV, H_Q, T_MID, N_DENSE = 1024, 64, 128, 8, 32, 2048, 307  # the 8B shapes (ADR §3)


def llama_rope(head_dim: int, hidden: int, factor: float) -> LlamaRotaryEmbedding:
    """A Llama-3 rotary module from its config alone (no weights, no download): the
    llama3-scaled inv_freq and attention_scaling the real models carry."""
    cfg = LlamaConfig(
        hidden_size=hidden, num_attention_heads=32, num_key_value_heads=8, head_dim=head_dim,
        max_position_embeddings=131072,
        rope_parameters={
            "rope_type": "llama3", "rope_theta": 500000.0,
            "factor": factor, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
        },
    )  # fmt: skip
    return LlamaRotaryEmbedding(cfg)


def reconstruct_then_attend(
    q: Tensor, dense_k: Tensor, dense_v: Tensor, mid: FactoredMiddle | None,
    rope: _RopeAngles, scaling: float, model_dtype: torch.dtype,
) -> Tensor:  # fmt: skip
    """Today's path on one batch row: fp32 U@C, the model's own cos/sin at the true
    positions, cast to the model dtype, exact fp32 attention over [dense | middle]."""
    ks, vs = [dense_k], [dense_v]
    if mid is not None and mid.n_columns:
        h_kv, d = dense_k.shape[1], dense_k.shape[3]
        cos, sin = rope.cos_sin_at(mid.positions[0])
        k_pre = (mid.u_k[0].float() @ mid.c_k[0].float()).reshape(h_kv, d, -1).permute(0, 2, 1)
        k_hat = (k_pre * cos + rotate_half(k_pre) * sin).to(model_dtype)  # type: ignore[no-untyped-call]
        v_hat = (mid.u_v[0].float() @ mid.c_v[0].float()).reshape(h_kv, d, -1).permute(0, 2, 1)
        ks.append(k_hat[None])
        vs.append(v_hat.to(model_dtype)[None])
    k, v = torch.cat(ks, dim=2).float(), torch.cat(vs, dim=2).float()
    return sdpa(q.float(), k, v, scale=scaling, enable_gqa=True)


def random_case(
    seed: int, contiguous: bool, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor, FactoredMiddle]:
    g = torch.Generator().manual_seed(seed)
    u_k = torch.linalg.qr(torch.randn(N, R, generator=g))[0]
    u_v = torch.linalg.qr(torch.randn(N, R, generator=g))[0]
    spectrum = torch.linspace(6.0, 0.3, R).unsqueeze(1)  # far from uniformly scaled (ADR §5)
    c_k = torch.randn(R, T_MID, generator=g) * spectrum
    c_v = torch.randn(R, T_MID, generator=g) * spectrum * 8.0  # x8: |out| ~ O(1), see below
    if contiguous:
        pos = torch.arange(N_DENSE, N_DENSE + T_MID)
    else:  # scattered true positions up to 64K: the fp32 angles the ADR insists on
        pos = torch.randperm(65536, generator=g)[:T_MID].sort().values
    dense_k = torch.randn(1, H_KV, N_DENSE, D, generator=g).to(dtype)
    # dense_v and c_v are scaled x8 so |out| is O(1): unscaled, |out| <= 0.11 and the
    # pre-registered absolute 1e-2 bar is a ~10% relative bar that cannot fail a nearly-right
    # kernel. The absolute bar itself is unchanged.
    dense_v = (torch.randn(1, H_KV, N_DENSE, D, generator=g) * 8.0).to(dtype)
    q = torch.randn(1, H_Q, 1, D, generator=g).to(dtype)
    mid = FactoredMiddle(u_k[None], u_v[None], c_k[None], c_v[None], pos[None])
    return q, dense_k, dense_v, mid


def _mid_to(m: FactoredMiddle, device: str) -> FactoredMiddle:
    return FactoredMiddle(*(t.to(device) for t in (m.u_k, m.u_v, m.c_k, m.c_v, m.positions)))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("contiguous", [True, False], ids=["contiguous", "scattered_to_64k"])
def test_random_8b_shapes_within_tolerance(backend: str, contiguous: bool) -> None:
    dev = "cuda" if backend == "triton" else "cpu"
    emb = llama_rope(D, 4096, 8.0)  # Llama-3.1-8B's rotary
    q, dk, dv, mid = random_case(0, contiguous, torch.bfloat16)
    out = factored_attention(
        q.to(dev), dk.to(dev), dv.to(dev), _mid_to(mid, dev),
        inv_freq=emb.inv_freq.to(dev), attention_scaling=emb.attention_scaling,
        scaling=D**-0.5, backend=backend,
    )  # fmt: skip
    ref = reconstruct_then_attend(q, dk, dv, mid, _RopeAngles(emb), D**-0.5, torch.bfloat16)
    assert out.shape == q.shape and out.dtype == q.dtype
    diff = float((out.float().cpu() - ref).abs().max())
    assert diff < TOL, diff


def test_fp32_operands_reproduce_reconstruct_to_1e_4() -> None:
    """No rounding anywhere: the tile loop, RoPE, GQA mapping and online softmax are the
    reconstruct path's arithmetic in a different order."""
    emb = llama_rope(D, 4096, 8.0)
    q, dk, dv, mid = random_case(1, False, torch.float32)
    out = factored_attention(
        q, dk, dv, mid, inv_freq=emb.inv_freq, attention_scaling=emb.attention_scaling,
        scaling=D**-0.5, operand_dtype=torch.float32, backend="reference",
    )  # fmt: skip
    ref = reconstruct_then_attend(q, dk, dv, mid, _RopeAngles(emb), D**-0.5, torch.float32)
    assert float((out - ref).abs().max()) < 1e-4


def test_no_middle_is_plain_attention_and_tiles_are_a_detail() -> None:
    emb = llama_rope(D, 4096, 8.0)
    q, dk, dv, mid = random_case(2, True, torch.float32)
    ref = sdpa(q, dk, dv, scale=D**-0.5, enable_gqa=True)
    for tile in (16, 64, 1000):
        out = factored_attention(
            q, dk, dv, None, inv_freq=emb.inv_freq, attention_scaling=emb.attention_scaling,
            scaling=D**-0.5, operand_dtype=torch.float32, tile=tile, backend="reference",
        )  # fmt: skip
        assert float((out - ref).abs().max()) < 1e-5
    a = factored_attention(q, dk, dv, mid, inv_freq=emb.inv_freq, attention_scaling=1.0,
                           scaling=D**-0.5, operand_dtype=torch.float32, tile=64,
                           backend="reference")  # fmt: skip
    b = factored_attention(q, dk, dv, mid, inv_freq=emb.inv_freq, attention_scaling=1.0,
                           scaling=D**-0.5, operand_dtype=torch.float32, tile=48,
                           backend="reference")  # fmt: skip
    assert float((a - b).abs().max()) < 1e-5
    with pytest.raises(ValueError, match="q_len"):
        factored_attention(q.expand(1, H_Q, 2, D), dk, dv, None, inv_freq=emb.inv_freq,
                           attention_scaling=1.0, scaling=D**-0.5, backend="reference")  # fmt: skip


def test_rope_cos_sin_is_the_models_own() -> None:
    emb = llama_rope(D, 4096, 8.0)
    pos = torch.tensor([[0, 1, 4095, 32767, 131071]])
    cos, sin = rope_cos_sin(pos, emb.inv_freq, emb.attention_scaling)
    want_cos, want_sin = emb(torch.empty(1), pos)
    assert torch.equal(cos, want_cos) and torch.equal(sin, want_sin)  # bit for bit


def test_n_splits_for_is_the_kernels_own_split_count() -> None:
    """The Triton backend's launch geometry, on the CPU (the kernel itself cannot be
    imported without triton): B = 16 rows at H_kv = 8 reaches TARGET_PROGRAMS in ONE split
    -- the premise of the tight bar in tests/test_kernel_triton.py -- while a single row
    cuts the same 32 tiles 16 ways, and no call ever asks for more splits than tiles."""
    assert n_splits_for(16, H_KV, 32) == 1 and n_splits_for(1, H_KV, 32) == 16
    assert n_splits_for(2, H_KV, 32) == 8 and n_splits_for(64, H_KV, 32) == 1
    assert n_splits_for(1, H_KV, 4) == 4 and n_splits_for(1, H_KV, 0) == 1  # empty middle


def test_auto_falls_back_to_the_reference_where_the_kernel_would_refuse() -> None:
    """R-L4-22: `min_sv_frac > 0` shrinks the live rank to an arbitrary integer and r192
    arms exist, so `backend="auto"` degrades to the reference on a shape the Triton kernel
    refuses instead of raising its ValueError mid-decode (an explicit `backend="triton"`
    still raises). The shape half of that decision is the kernel's own refusals, which is
    what `_shapes_eligible` pins here; `_triton_eligible` is False on this tree whatever
    the shapes, there being neither CUDA nor triton."""
    q, dk, _, mid = random_case(7, True, torch.bfloat16)
    r48 = FactoredMiddle(
        mid.u_k[:, :, :48], mid.u_v[:, :, :48], mid.c_k[:, :48], mid.c_v[:, :48], mid.positions
    )
    assert _shapes_eligible(q, mid, H_KV)  # the 8B shapes: r 64, D 128, G 4
    assert _shapes_eligible(q, None, H_KV)  # no middle: no rank to refuse
    assert not _shapes_eligible(q, r48, H_KV)  # a live rank between the powers of two
    assert not _shapes_eligible(q[:, :, :, :96], mid, H_KV)  # D 96
    assert not _shapes_eligible(q, mid, 1)  # G = 32 query heads per KV head
    assert not _triton_eligible(q, dk, r48, torch.bfloat16)
    assert not _triton_eligible(q, dk, mid, torch.bfloat16)


@pytest.mark.skipif(not (DUMP / "layer_08.pt").is_file(), reason="1B dumps absent (gitignored)")
@pytest.mark.parametrize("backend", BACKENDS)
def test_one_layer_of_the_1b_dump(backend: str) -> None:
    """Layer 8 of Llama-3.2-1B on a real C4 document: the r64 configuration's own prefill of
    tokens 0..4094 (bf16 K/V, the pod's stored representation) and the decode step that
    pushes token 4095, attended by that token's real query. The rotary is rebuilt from the
    config and pinned bit-for-bit against the dump's own rope.pt."""
    dev = "cuda" if backend == "triton" else "cpu"
    blob = torch.load(DUMP / "layer_08.pt", map_location="cpu", weights_only=True)
    rope_pt = torch.load(DUMP / "rope.pt", map_location="cpu", weights_only=True)
    emb = llama_rope(64, 2048, 32.0)  # Llama-3.2-1B: llama3 rope, factor 32
    cos, sin = emb(torch.empty(1), torch.arange(4096).unsqueeze(0))
    assert torch.equal(cos[0], rope_pt["cos"]) and torch.equal(sin[0], rope_pt["sin"])
    k_post, v = blob["K"].to(torch.bfloat16)[None], blob["V"].to(torch.bfloat16)[None]
    layer = BugStreamingLayer(
        rope=_RopeAngles(emb), rank=64, coord_budget=4096 + 48, recent_window=32, absorb_block=16,
        n_sink=4, retention="lowrank_surprise", hh_budget=256, hh_select="surprise", hh_neighbor=1,
        seed_hh_warmup=True,
    )  # fmt: skip
    layer._mode = "ingest"  # the first chunk of a chunked prefill: the warm-up seed fills the tier
    with torch.no_grad():
        layer.update(k_post[:, :, :4095], v[:, :, :4095])  # single-shot prefill of tokens 0..4094
    layer._mode = "normal"
    with torch.no_grad():
        layer.update(k_post[:, :, 4095:], v[:, :, 4095:])  # the decode step that pushes token 4095
    # What a kernel decode step hands attention, assembled by the layer itself.
    mid = layer._factored_middle()
    dense_k, dense_v = layer._decode_peek(dense_only=True)
    assert mid is not None and layer._hh_len() == 256 and mid.n_columns > 3000
    assert dense_k.shape[2] == 4 + 256 + layer._recent_len()
    q_pre = blob["Q_pre"][:, 4095]  # (32, 64), pre-RoPE, all query heads
    rot = rotate_half(q_pre)  # type: ignore[no-untyped-call]
    q = (q_pre * cos[0, 4095] + rot * sin[0, 4095]).to(torch.bfloat16)[None, :, None]
    out = factored_attention(
        q.to(dev), dense_k.to(dev), dense_v.to(dev), _mid_to(mid, dev),
        inv_freq=emb.inv_freq.to(dev), attention_scaling=emb.attention_scaling,
        scaling=64**-0.5, backend=backend,
    )  # fmt: skip
    k_full, v_full = layer._decode_peek()  # today's path: bf16 [sinks | tier | mid-hat | recent]
    ref = sdpa(q.float(), k_full.float(), v_full.float(), scale=64**-0.5, enable_gqa=True)
    diff = float((out.float().cpu() - ref).abs().max())
    assert diff < TOL, diff
