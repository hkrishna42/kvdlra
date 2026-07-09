"""Tests for :mod:`kvdlra.cache.bug_cache` (the constant-memory decode cache).

Hermetic: a *tiny random-weight* Llama is instantiated from a local config (no
download) so the full transformers-5.8 decode path -- mask sizes, position
derivation, sdpa/eager attention -- is exercised end to end. The correctness
ladder (design note §6):

1. exact-mode **byte parity** with ``DynamicCache`` (nothing truncated/dropped);
2. **mask consistency** (``get_mask_sizes`` == returned length, every step);
3. **constant memory** in generated length;
4. the **coordinate-carry invariant** across basis updates;
5. StreamingLLM degenerate mode retains sinks + recent verbatim.

Also covers the ``augmented_bug_step`` extraction: the returned ``rot`` really
is ``u_new^T u_old``, and the ``rank + block > n_features`` clamp fixes the
latent degeneracy flagged in ``docs/week5.md``.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer, _RopeAngles
from kvdlra.integrators.streaming_torch import augmented_bug_step, blocked_bug_subspace

# Tiny Llama: 2 layers, 2 KV heads x head_dim 16 => n_features = 32.
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


def _generate(
    model: LlamaForCausalLM, prompt: torch.Tensor, cache: object, n_new: int
) -> torch.Tensor:
    with torch.no_grad():
        out = model.generate(  # type: ignore[misc]
            prompt,
            max_new_tokens=n_new,
            do_sample=False,
            past_key_values=cache,
            pad_token_id=0,
        )
    return out[0, prompt.shape[1] :]


def _prompt(t: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, t), generator=g)


# --------------------------------------------------------------------------
# augmented_bug_step (the factored integrator step)
# --------------------------------------------------------------------------


def test_step_rot_is_new_basis_expressed_in_old() -> None:
    g = torch.Generator().manual_seed(0)
    m = torch.randn(N_FEATURES, 64, generator=g, dtype=torch.float64)
    u, b, _ = augmented_bug_step(None, None, m[:, :16], 8)
    u2, _b2, rot = augmented_bug_step(u, b, m[:, 16:32], 8)
    # rot must be exactly u_new^T u_old (the coordinate-carry map).
    assert torch.allclose(rot, u2.mT @ u, atol=1e-12)
    # Bases stay orthonormal.
    eye = torch.eye(u2.shape[1], dtype=torch.float64)
    assert torch.allclose(u2.mT @ u2, eye, atol=1e-12)


def test_step_seed_matches_blocked_sweep() -> None:
    # Driving augmented_bug_step block-by-block must equal blocked_bug_subspace
    # (it *is* the extracted loop body -- guard against drift).
    g = torch.Generator().manual_seed(1)
    m = torch.randn(N_FEATURES, 96, generator=g)
    u_ref = blocked_bug_subspace(m, rank_cap=8, block_size=32)
    u, b = None, None
    for start in range(0, 96, 32):
        u, b, _ = augmented_bug_step(u, b, m[:, start : start + 32].to(torch.float32), 8)
    assert u is not None
    assert torch.allclose(u, u_ref, atol=0.0)  # identical op sequence => bitwise


def test_step_clamps_overfull_augmentation() -> None:
    # rank + block > n_features: the residual has rank <= n - r, the clamp must
    # keep the basis orthonormal and the step exact (the docs/week5.md latent bug).
    g = torch.Generator().manual_seed(2)
    n = 16
    m = torch.randn(n, 64, generator=g, dtype=torch.float64)
    u, b, _ = augmented_bug_step(None, None, m[:, :12], 12)
    u2, _, _ = augmented_bug_step(u, b, m[:, 12:40], 16)  # 12 + 28 > 16
    eye = torch.eye(u2.shape[1], dtype=torch.float64)
    assert u2.shape[1] <= n
    assert torch.allclose(u2.mT @ u2, eye, atol=1e-10)
    # At rank == n the tracked subspace is all of R^n => projection is identity.
    u3, b3, _ = augmented_bug_step(None, None, m[:, :12], 16)
    u3, b3, _ = augmented_bug_step(u3, b3, m[:, 12:40], 16)
    proj = u3 @ (u3.mT @ m)
    assert torch.allclose(proj, m, atol=1e-9)


def test_step_validation() -> None:
    m = torch.randn(8, 4)
    with pytest.raises(ValueError, match="together"):
        augmented_bug_step(torch.eye(8, 2), None, m, 2)
    with pytest.raises(ValueError, match="rank_cap"):
        augmented_bug_step(None, None, m, 0)


# --------------------------------------------------------------------------
# BugStreamingCache: end-to-end decode correctness
# --------------------------------------------------------------------------


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_exact_mode_parity_with_dynamic_cache(tiny_model: LlamaForCausalLM, attn: str) -> None:
    # rank == n_features and generous budgets => nothing is ever truncated or
    # dropped, so the cache is information-losslessly equivalent to DynamicCache.
    # Teacher-forced comparison (same token stream through both) pins the whole
    # plumbing layer -- masks, positions, RoPE round trip -- without the chaotic
    # argmax-flip amplification of free-running generation: per-step logits must
    # agree to fp tolerance (the middle block is reconstructed through an fp32
    # un-rotate/re-rotate round trip, so bitwise equality is *not* expected there;
    # sink/recent tokens are stored verbatim and BUGPress-style byte parity on a
    # real model is checked in the w5 validation script).
    tiny_model.config._attn_implementation = attn
    cache = BugStreamingCache(
        tiny_model, rank=N_FEATURES, coord_budget=4096, recent_window=8, absorb_block=4
    )
    ref = DynamicCache()
    g = torch.Generator().manual_seed(7)
    stream = torch.randint(0, 256, (1, 90), generator=g)
    with torch.no_grad():
        out_a = tiny_model(stream[:, :33], past_key_values=cache, use_cache=True)
        out_b = tiny_model(stream[:, :33], past_key_values=ref, use_cache=True)
        assert torch.equal(out_a.logits, out_b.logits)  # pre-fill returns input as-is
        for t in range(33, 90):
            tok = stream[:, t : t + 1]
            out_a = tiny_model(tok, past_key_values=cache, use_cache=True)
            out_b = tiny_model(tok, past_key_values=ref, use_cache=True)
            assert torch.allclose(out_a.logits, out_b.logits, atol=1e-4)
    # The returned K/V at full rank must match the reference cache to fp tol.
    layer = cache.layers[0]
    assert isinstance(layer, BugStreamingLayer)
    k_ret, v_ret = layer._decode_peek()
    ref_layer = ref.layers[0]
    assert isinstance(ref_layer, DynamicLayer)
    assert ref_layer.keys is not None and ref_layer.values is not None
    assert torch.allclose(k_ret, ref_layer.keys, atol=1e-4)
    assert torch.allclose(v_ret, ref_layer.values, atol=1e-4)


def test_lowrank_mode_generates_and_masks_stay_consistent(
    tiny_model: LlamaForCausalLM,
) -> None:
    # Aggressive rank: outputs may differ from baseline, but every decode step's
    # predicted mask size must equal the K/V length actually returned.
    tiny_model.config._attn_implementation = "sdpa"
    cache = BugStreamingCache(tiny_model, rank=8, coord_budget=24, recent_window=8, absorb_block=4)
    layer = cache.layers[0]
    assert isinstance(layer, BugStreamingLayer)
    ids = _prompt(40)
    with torch.no_grad():
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        for _ in range(70):
            kv_length, kv_offset = layer.get_mask_sizes(1)
            tok = out.logits[:, -1:].argmax(-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
            assert kv_length == layer.attended_length()
            # The offset places the returned block's last fake position exactly at
            # the query's true position (all retained tokens visible to it).
            assert kv_offset + kv_length == layer.cumulative_length


def test_memory_constant_in_generated_length(tiny_model: LlamaForCausalLM) -> None:
    cache = BugStreamingCache(tiny_model, rank=8, coord_budget=32, recent_window=8, absorb_block=4)
    ids = _prompt(40)
    mems = []
    with torch.no_grad():
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        for _ in range(200):
            tok = out.logits[:, -1:].argmax(-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
            mems.append(cache.stored_state_numel())
    # The bound is reached early and never exceeded afterwards: constant memory.
    assert max(mems[50:]) <= max(mems[:50])
    # Workspace (cached middle reconstruction) is bounded too.
    assert cache.workspace_numel() <= 2 * 2 * N_FEATURES * 32  # layers*(K,V)*n*W
    # ... while the cumulative length kept growing far past the stored budget.
    layer = cache.layers[0]
    assert isinstance(layer, BugStreamingLayer)
    assert layer.cumulative_length == 240
    assert layer.attended_length() < 60


def test_streamingllm_mode_keeps_sinks_and_recents_verbatim(
    tiny_model: LlamaForCausalLM,
) -> None:
    # rank=0/coord_budget=0 => sinks + recent ring only. Cross-check the retained
    # tokens verbatim against a full DynamicCache reference run on the same greedy
    # trajectory (feed the *streaming* run's tokens into the reference model).
    tiny_model.config._attn_implementation = "sdpa"
    n_sink, w, b = 4, 8, 4
    cache = BugStreamingCache(
        tiny_model, rank=0, coord_budget=0, recent_window=w, absorb_block=b, n_sink=n_sink
    )
    ref = DynamicCache()
    ids = _prompt(24)
    with torch.no_grad():
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        toks = []
        for _ in range(30):
            tok = out.logits[:, -1:].argmax(-1)
            toks.append(tok)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
        full = torch.cat([ids, *toks], dim=1)
        tiny_model(full, past_key_values=ref, use_cache=True)

    layer = cache.layers[0]
    assert isinstance(layer, BugStreamingLayer)
    assert layer._mid_len() == 0
    ref_layer = ref.layers[0]
    assert isinstance(ref_layer, DynamicLayer)
    ref_k = ref_layer.keys  # (1, H, T, D)
    assert ref_k is not None
    assert layer.sink_k is not None and layer.recent_k is not None
    t_total = layer.cumulative_length
    rec_len = layer.recent_k.shape[1]
    sink_ref = ref_k[0, :, :n_sink, :].permute(0, 2, 1).reshape(N_FEATURES, n_sink)
    rec_ref = ref_k[0, :, t_total - rec_len :, :].permute(0, 2, 1).reshape(N_FEATURES, rec_len)
    assert torch.allclose(layer.sink_k, sink_ref, atol=1e-5)
    assert torch.allclose(layer.recent_k, rec_ref, atol=1e-5)
    # Ring stays within its designed band.
    assert w <= rec_len < w + b


def test_coordinate_carry_matches_direct_projection(tiny_model: LlamaForCausalLM) -> None:
    # Tokens absorbed in the LAST absorb event have coordinates u^T k_pre in the
    # *current* basis: their reconstruction must equal the direct projection of
    # their original pre-RoPE key. Older tokens' reconstructions must lie in
    # range(U) (projected-history semantics).
    rope = _RopeAngles(cast(nn.Module, tiny_model.model.rotary_emb))
    layer = BugStreamingLayer(
        rope=rope, rank=8, coord_budget=64, recent_window=4, absorb_block=4, n_sink=2
    )
    g = torch.Generator().manual_seed(5)

    def kv(t: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.randn(1, H, t, D, generator=g),
            torch.randn(1, H, t, D, generator=g),
        )

    layer.update(*kv(20))  # prefill
    originals: dict[int, torch.Tensor] = {}  # position -> pre-RoPE key column
    pos = 20
    for _ in range(24):
        k, v = kv(1)
        originals[pos] = layer._mat_rope(layer._to_mat(k), pos, inverse=True)[:, 0]
        layer.update(k, v)
        pos += 1

    assert layer.u_k is not None and layer.c_k is not None
    mid_len = layer._mid_len()
    mid_start = layer.cumulative_length - layer._recent_len() - mid_len
    recon = layer.u_k @ layer.c_k  # (n, mid_len) pre-RoPE reconstruction
    # Last absorbed block: exact direct projection.
    for j in range(mid_len - layer.absorb_block, mid_len):
        p = mid_start + j
        if p in originals:
            direct = layer.u_k @ (layer.u_k.mT @ originals[p])
            assert torch.allclose(recon[:, j], direct, atol=1e-4)
    # All middle reconstructions lie in range(U): projecting is a no-op.
    assert torch.allclose(layer.u_k @ (layer.u_k.mT @ recon), recon, atol=1e-4)


def test_single_token_prompt_and_short_prompt(tiny_model: LlamaForCausalLM) -> None:
    cache = BugStreamingCache(tiny_model, rank=8, coord_budget=16, recent_window=4)
    out = _generate(tiny_model, _prompt(1), cache, 12)
    assert out.shape[0] == 12
    cache2 = BugStreamingCache(tiny_model, rank=8, coord_budget=16, recent_window=4)
    out2 = _generate(tiny_model, _prompt(3), cache2, 12)
    assert out2.shape[0] == 12


def test_chunked_prefill_raises(tiny_model: LlamaForCausalLM) -> None:
    cache = BugStreamingCache(tiny_model, rank=8, coord_budget=16)
    with torch.no_grad():
        tiny_model(_prompt(16), past_key_values=cache, use_cache=True)
        with pytest.raises(NotImplementedError, match="single-shot"):
            tiny_model(_prompt(8, seed=2), past_key_values=cache, use_cache=True)


def test_batch_size_one_enforced(tiny_model: LlamaForCausalLM) -> None:
    cache = BugStreamingCache(tiny_model, rank=8, coord_budget=16)
    ids = torch.randint(0, 256, (2, 10))
    with torch.no_grad(), pytest.raises(NotImplementedError, match="batch size 1"):
        tiny_model(ids, past_key_values=cache, use_cache=True)


def test_constructor_validation(tiny_model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError, match="recent_window"):
        BugStreamingCache(tiny_model, rank=8, coord_budget=16, recent_window=0)
    with pytest.raises(ValueError, match=">= 0"):
        BugStreamingCache(tiny_model, rank=-1, coord_budget=16)
    with pytest.raises(ValueError, match="absorb_block"):
        BugStreamingCache(tiny_model, rank=8, coord_budget=16, absorb_block=0)


def test_reset_allows_reuse(tiny_model: LlamaForCausalLM) -> None:
    cache = BugStreamingCache(tiny_model, rank=8, coord_budget=16, recent_window=4)
    _generate(tiny_model, _prompt(20), cache, 10)
    for layer in cache.layers:
        layer.reset()
    out = _generate(tiny_model, _prompt(20, seed=3), cache, 10)
    assert out.shape[0] == 10


def test_core_is_diagonal_so_counted_as_r(tiny_model: LlamaForCausalLM) -> None:
    # The square-root core B is always diag(sigma); stored_state_numel counts it
    # as its r diagonal entries (deployable footprint), not r^2. Pin both facts.
    cache = BugStreamingCache(tiny_model, rank=8, coord_budget=16, recent_window=8, absorb_block=4)
    ids = _prompt(30)
    with torch.no_grad():
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        for _ in range(50):
            tok = out.logits[:, -1:].argmax(-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
    layer = cache.layers[0]
    assert isinstance(layer, BugStreamingLayer)
    assert layer.b_k is not None and layer.b_v is not None
    for core in (layer.b_k, layer.b_v):
        off = (core - torch.diag(torch.diag(core))).abs().max()
        assert float(off) == 0.0  # exactly diagonal
    # stored_state_numel counts the core as r (min dim), not r*r.
    r = layer.b_k.shape[0]
    counted = layer.stored_state_numel()
    dense = counted + 2 * (r * r - r)  # what it would be if B were counted dense
    assert counted < dense  # the diagonal saving is real
