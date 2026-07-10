"""Tests for Week-8 CodeBUG: product-quantized coordinates against a frozen anchor.

CodeBUG replaces the *storage* of the low-rank coordinate tail with codes into a
shared, calibrated product-quantization codebook, expressed once against a
**frozen** reduced-rank anchor ``u_ref`` and NEVER rotated (the structural
inverse of Week-7 variant D's per-absorb dequantize->rot->requantize, which
exploded 14.7x). The correctness ladder:

1. **runs end-to-end** and populates the coded tier (finite output);
2. **codes are frozen after graduation** -- ``_rotate_quant_tier`` is never
   called and a coded column's reconstruction is bit-stable across absorbs (the
   non-compounding guarantee, proven structurally; the *magnitude* on real KV is
   measured by ``scripts/w8_carry_drift.py``);
3. **mask consistency** (``get_mask_sizes`` == returned length, every step);
4. **constant memory** in generated length;
5. **honest accounting** -- the frozen anchor is counted, codes at ``bits/32``,
   and the shared codebook counted **once** at the cache level;
6. **guards** on the new configuration surface.
"""

from __future__ import annotations

import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.quant import ProductQuantizer

H, D = 2, 16
N_FEATURES = H * D


def _tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=H,
        head_dim=D,
        max_position_embeddings=2048,
    )


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    model = LlamaForCausalLM(_tiny_config())  # type: ignore[no-untyped-call]
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _codebook(dim: int, *, normalize: bool = True, seed: int = 0) -> ProductQuantizer:
    g = torch.Generator().manual_seed(seed)
    samples = torch.randn(3000, dim, generator=g)
    return ProductQuantizer(
        dim=dim, bits=4, subspaces=min(4, dim), normalize=normalize, seed=seed
    ).fit(samples, iters=8)


def _codebug_cache(
    model: LlamaForCausalLM,
    *,
    rank: int = 8,
    anchor_rank: int = 4,
    coord_budget: int = 8,
    code_budget: int = 40,
    retention: str = "fifo",
    normalize: bool = True,
    hh_budget: int = 0,
    seal: int = 1,
) -> BugStreamingCache:
    cb = _codebook(anchor_rank, normalize=normalize)
    return BugStreamingCache(
        model,
        rank=rank,
        coord_budget=coord_budget,
        recent_window=4,
        absorb_block=4,
        n_sink=2,
        retention=retention,
        coord_codebook=cb,
        anchor_rank=anchor_rank,
        anchor_seal_absorbs=seal,
        code_budget=code_budget,
        hh_budget=hh_budget,
    )


def _prompt(t: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, t), generator=g)


def _generate(
    model: LlamaForCausalLM, prompt: torch.Tensor, cache: object, n_new: int
) -> torch.Tensor:
    with torch.no_grad():
        out = model.generate(  # type: ignore[misc]
            prompt, max_new_tokens=n_new, do_sample=False, past_key_values=cache, pad_token_id=0
        )
    return out[0, prompt.shape[1] :]


# --------------------------------------------------------------------------
# 1. runs + populates the coded tier
# --------------------------------------------------------------------------


def test_codebug_runs_and_populates_coded_tier(tiny_model: LlamaForCausalLM) -> None:
    cache = _codebug_cache(tiny_model)
    out = _generate(tiny_model, _prompt(8), cache, n_new=80)
    assert out.shape == (80,)
    layer = cache._bug_layers()[0]
    assert layer.sealed  # anchor frozen after warmup
    assert layer.u_ref_k is not None and layer.u_ref_k.shape == (N_FEATURES, layer.anchor_rank)
    assert layer._q_len() > 0  # coded tier actually got populated
    assert layer.qk_codes is not None and layer.qk_codes.dtype == torch.uint8
    assert layer.codebook is not None
    # coded codes have M columns (subspaces), not r columns.
    assert layer.qk_codes.shape[1] == layer.codebook.subspaces


# --------------------------------------------------------------------------
# 2. the non-compounding guarantee (structural)
# --------------------------------------------------------------------------


def test_codebug_never_rotates_codes(
    tiny_model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CodeBUG must NEVER call the variant-D requantize-carry path."""
    cache = _codebug_cache(tiny_model)

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("CodeBUG rotated the coded tier (variant-D path)!")

    for layer in cache._bug_layers():
        monkeypatch.setattr(layer, "_rotate_quant_tier", _boom)
    out = _generate(tiny_model, _prompt(8), cache, n_new=80)
    assert torch.isfinite(out.float()).all()


def test_codebug_coded_column_reconstruction_is_bit_stable(tiny_model: LlamaForCausalLM) -> None:
    """The oldest coded column's pre-RoPE reconstruction ``u_ref @ decode(codes)``
    is identical across later absorbs -- zero drift by construction (codes and
    u_ref both frozen). code_budget is large enough that column 0 is not evicted."""
    cache = _codebug_cache(tiny_model, code_budget=1000)
    layer = cache._bug_layers()[0]
    _generate(tiny_model, _prompt(8), cache, n_new=40)
    assert layer._q_len() > 0 and layer.u_ref_k is not None

    def recon0() -> torch.Tensor:
        assert layer.qk_codes is not None and layer.u_ref_k is not None
        norm0 = layer.qk_norm[:1] if layer.qk_norm is not None else None
        col = layer._decode_coded(layer.qk_codes[:1], norm0)
        recon: torch.Tensor = layer.u_ref_k @ col
        return recon.clone()

    before = recon0()
    n_before = layer._q_len()
    # Force many more absorbs by continuing to decode single tokens on the SAME
    # cache (generate() again would re-prefill and trip the single-shot guard).
    cur = _prompt(1, seed=2)
    with torch.no_grad():
        for _ in range(50):
            out = tiny_model(cur, past_key_values=cache, use_cache=True)
            cur = out.logits[:, -1:].argmax(-1)
    assert layer._q_len() > n_before  # tier grew (more graduations coded)
    after = recon0()
    assert torch.equal(before, after)  # column 0's reconstruction never changed


# --------------------------------------------------------------------------
# 3. mask consistency
# --------------------------------------------------------------------------


def test_codebug_mask_consistency(tiny_model: LlamaForCausalLM) -> None:
    cache = _codebug_cache(tiny_model)
    layer0 = cache._bug_layers()[0]
    prompt = _prompt(8)
    with torch.no_grad():
        tiny_model(prompt, past_key_values=cache, use_cache=True)  # prefill
        cur = prompt[:, -1:]
        for _ in range(60):
            kv_len, _off = layer0.get_mask_sizes(1)
            out = tiny_model(cur, past_key_values=cache, use_cache=True)
            # the returned K length equals what the mask was told to expect
            assert layer0.attended_length() == kv_len
            cur = out.logits[:, -1:].argmax(-1)


# --------------------------------------------------------------------------
# 4. constant memory
# --------------------------------------------------------------------------


def test_codebug_constant_memory(tiny_model: LlamaForCausalLM) -> None:
    def mem_after(n_new: int) -> int:
        cache = _codebug_cache(tiny_model)
        _generate(tiny_model, _prompt(8), cache, n_new=n_new)
        return int(cache.stored_state_numel())

    m_short, m_long = mem_after(60), mem_after(140)
    assert m_short == m_long  # saturated tiers => identical footprint


# --------------------------------------------------------------------------
# 5. honest accounting
# --------------------------------------------------------------------------


def test_codebug_codebook_counted_once(tiny_model: LlamaForCausalLM) -> None:
    cache = _codebug_cache(tiny_model)
    _generate(tiny_model, _prompt(8), cache, n_new=60)
    codebook = cache._codebook
    assert codebook is not None
    layer_sum = sum(layer.stored_state_numel() for layer in cache._bug_layers())
    # cache-level total = per-layer state + the shared codebook counted ONCE.
    assert cache.stored_state_numel() == layer_sum + codebook.codebook_numel()
    # the codebook float count is K * dim, independent of the 2 layers.
    assert codebook.codebook_numel() == codebook.levels * codebook.dim


def test_codebug_anchor_and_codes_are_counted(tiny_model: LlamaForCausalLM) -> None:
    cache = _codebug_cache(tiny_model)
    _generate(tiny_model, _prompt(8), cache, n_new=60)
    layer = cache._bug_layers()[0]
    assert layer.u_ref_k is not None and layer.u_ref_v is not None
    assert layer.qk_codes is not None and layer.qv_codes is not None
    assert layer.codebook is not None
    # Rebuild the expected per-layer float count from its parts and match exactly.
    n, ra = N_FEATURES, layer.anchor_rank
    expected = 0
    for t in (
        layer.sink_k,
        layer.sink_v,
        layer.recent_k,
        layer.recent_v,
        layer.u_k,
        layer.c_k,
        layer.u_v,
        layer.c_v,
        layer.u_ref_k,
        layer.u_ref_v,
    ):
        if t is not None:
            expected += t.numel()
    for core in (layer.b_k, layer.b_v):
        if core is not None:
            expected += min(core.shape)
    bits = layer.codebook.bits
    expected += math.ceil((layer.qk_codes.numel() + layer.qv_codes.numel()) * bits / 32)
    for nrm in (layer.qk_norm, layer.qv_norm):
        if nrm is not None:
            expected += nrm.numel()
    for bk in (layer.mid_pos, layer.q_pos, layer.mid_score, layer.q_score, layer.ring_score):
        if bk is not None:
            expected += bk.numel()
    assert layer.stored_state_numel() == expected
    # anchor really is (n, r_a) and contributes 2*n*r_a floats (K+V).
    assert layer.u_ref_k.shape == (n, ra) and layer.u_ref_v.shape == (n, ra)


def test_codebug_cheaper_than_fp32_coords(tiny_model: LlamaForCausalLM) -> None:
    # A coded column must cost far less than an fp32 coordinate column (r floats):
    # that density gain is the entire point. Compare code+norm bits to r floats.
    cache = _codebug_cache(tiny_model, normalize=True)
    _generate(tiny_model, _prompt(8), cache, n_new=80)
    layer = cache._bug_layers()[0]
    assert layer.qk_codes is not None and layer.codebook is not None
    per_coded = layer.codebook.subspaces * layer.codebook.bits / 32 + 1  # codes + 1 norm (K)
    assert per_coded < layer.rank  # cheaper than r fp32 floats/token


# --------------------------------------------------------------------------
# 6. guards + variants
# --------------------------------------------------------------------------


def test_codebug_attn_and_slash_run(tiny_model: LlamaForCausalLM) -> None:
    # attn retention (needs attach) + the SLASH exact-hedge tier compose.
    cache = _codebug_cache(tiny_model, retention="attn", hh_budget=2)
    with cache.attach(tiny_model):
        out = _generate(tiny_model, _prompt(8), cache, n_new=80)
    assert torch.isfinite(out.float()).all()
    layer = cache._bug_layers()[0]
    assert layer._q_len() > 0 and layer._hh_len() <= 2


def test_codebug_experiment_hooks_run(tiny_model: LlamaForCausalLM) -> None:
    # The drift-experiment hooks (`scripts/w8_carry_drift.py`): the OFF recode
    # control (`_debug_recode_coded`) and the ground-truth capture (`_drift_truth`,
    # which records positions -> needs retention="attn") must run end to end and
    # populate. (Both off by default in production.)
    cache = _codebug_cache(tiny_model, code_budget=1000, retention="attn")
    for layer in cache._bug_layers():
        layer._debug_recode_coded = True
        layer._drift_truth = {}
    with cache.attach(tiny_model):
        out = _generate(tiny_model, _prompt(8), cache, n_new=60)
    assert torch.isfinite(out.float()).all()
    layer = cache._bug_layers()[0]
    assert layer._q_len() > 0
    assert layer._drift_truth is not None and len(layer._drift_truth) > 0  # truth captured
    assert layer.qk_codes is not None and layer.u_ref_k is not None  # recode kept tier valid


def test_codebug_raw_mode_runs(tiny_model: LlamaForCausalLM) -> None:
    cache = _codebug_cache(tiny_model, normalize=False)
    out = _generate(tiny_model, _prompt(8), cache, n_new=60)
    assert torch.isfinite(out.float()).all()
    assert cache._bug_layers()[0]._q_len() > 0


def test_codebug_validation(tiny_model: LlamaForCausalLM) -> None:
    cb = _codebook(4)
    with pytest.raises(ValueError, match="code_budget"):
        BugStreamingCache(tiny_model, rank=8, coord_budget=8, coord_codebook=cb, anchor_rank=4)
    with pytest.raises(ValueError, match="coord_codebook"):  # code_budget without a codebook
        BugStreamingCache(tiny_model, rank=8, coord_budget=8, code_budget=10)
    with pytest.raises(ValueError, match="mutually exclusive"):  # vs PolarQuant tier
        BugStreamingCache(
            tiny_model,
            rank=8,
            coord_budget=8,
            coord_codebook=cb,
            anchor_rank=4,
            code_budget=10,
            quant_bits=4,
            quant_budget=8,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):  # vs merge
        BugStreamingCache(
            tiny_model,
            rank=8,
            coord_budget=8,
            coord_codebook=cb,
            anchor_rank=4,
            code_budget=10,
            merge=True,
        )
    with pytest.raises(ValueError, match="anchor_rank"):  # anchor_rank > rank
        BugStreamingCache(
            tiny_model,
            rank=4,
            coord_budget=8,
            coord_codebook=_codebook(8),
            anchor_rank=8,
            code_budget=10,
        )
    with pytest.raises(ValueError, match="dim"):  # codebook.dim != anchor_rank
        BugStreamingCache(
            tiny_model,
            rank=8,
            coord_budget=8,
            coord_codebook=_codebook(6),
            anchor_rank=4,
            code_budget=10,
        )
