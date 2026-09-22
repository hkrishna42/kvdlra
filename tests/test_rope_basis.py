"""`rope_basis: post` -- the gist tracks post-RoPE keys (ADR 0001 §3 option (i)): an identity
in the shared rotation helper (`_mat_rope_with`), reached from the three ingest un-rotations
and the reconstruct re-rotation. Default "pre" is
today's path (the golden). Pins: at full rank the post basis stores the rotated keys exactly
and the decode logits match DynamicCache's; the middle's reconstruction IS the post-RoPE keys
(no rotation applied); the kernel refuses the post basis; the arm/pod files are what
prereg/postrope_r128.md says."""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer
from kvdlra.eval.config import arm_kwargs, load_arm, load_pod, load_task
from kvdlra.eval.frontier import build_arm
from tests.conftest import N_FEATURES, tiny_cache

TINY_SDPA = True


def test_default_is_pre_and_the_knob_is_validated(tiny_model: LlamaForCausalLM) -> None:
    assert all(la.rope_basis == "pre" for la in tiny_cache(tiny_model)._bug_layers())
    with pytest.raises(ValueError, match="rope_basis"):
        tiny_cache(tiny_model, rope_basis="none")
    with pytest.raises(ValueError, match="rope_basis"):
        tiny_cache(tiny_model, rope_basis="post", decode_attention="kernel")


def test_post_basis_at_full_rank_matches_dynamic_cache(tiny_model: LlamaForCausalLM) -> None:
    cache = BugStreamingCache(tiny_model, rank=N_FEATURES, coord_budget=4096, recent_window=8,
                              absorb_block=4, rope_basis="post")  # fmt: skip
    ref = DynamicCache()
    stream = torch.randint(0, 256, (1, 90), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        a = tiny_model(stream[:, :33], past_key_values=cache, use_cache=True)
        b = tiny_model(stream[:, :33], past_key_values=ref, use_cache=True)
        assert torch.equal(a.logits, b.logits)
        for t in range(33, 90):
            tok = stream[:, t : t + 1]
            a = tiny_model(tok, past_key_values=cache, use_cache=True)
            b = tiny_model(tok, past_key_values=ref, use_cache=True)
            assert torch.allclose(a.logits, b.logits, atol=1e-4)
    layer, ref_layer = cache.layers[0], ref.layers[0]
    assert isinstance(layer, BugStreamingLayer) and isinstance(ref_layer, DynamicLayer)
    assert ref_layer.keys is not None
    # the middle's reconstruction is the stored post-RoPE key, no rotation applied
    layer._ensure_mid_cache()
    assert layer._mid_k_cache is not None and layer.u_k is not None and layer.c_k is not None
    assert torch.allclose(layer._mid_k_cache, layer.u_k @ layer.c_k, atol=1e-5)
    start = layer.cumulative_length - layer._recent_len() - layer._mid_len()
    mid = ref_layer.keys[0, :, start : start + layer._mid_len(), :]
    want = mid.permute(0, 2, 1).reshape(N_FEATURES, -1)
    assert torch.allclose(layer._mid_k_cache, want, atol=1e-4)


def test_post_basis_differs_from_pre_at_low_rank(tiny_model: LlamaForCausalLM) -> None:
    ids = torch.randint(0, 256, (1, 60), generator=torch.Generator().manual_seed(3))
    outs = []
    for basis in ("pre", "post"):
        cache = tiny_cache(tiny_model, rope_basis=basis)
        with torch.no_grad():
            out = tiny_model(ids, past_key_values=cache, use_cache=True)
            for _ in range(20):
                nxt = out.logits[:, -1:].argmax(-1)
                out = tiny_model(nxt, past_key_values=cache, use_cache=True)
        outs.append(out.logits)
    assert not torch.allclose(outs[0], outs[1], atol=1e-3)  # a different basis, not a no-op


def test_the_postrope_arm_and_pod_are_their_prereg_design() -> None:
    arm, base = load_arm("isvd_postrope_r128"), load_arm("isvd_r64_h256_seed")
    assert arm.kind == "bug" and arm.legacy_name is None
    assert arm.cache == {**base.cache, "rank": 128, "rope_basis": "post"}
    want = {**arm_kwargs(base, 16384), "rank": 128, "rope_basis": "post"}
    assert build_arm(arm, None, 16384)["kwargs"] == want
    p = load_pod("postrope_r128")
    assert p.prereg == "prereg/postrope_r128.md" and p.model == "unsloth/Meta-Llama-3.1-8B-Instruct"
    assert p.arms == ["isvd_postrope_r128"] and p.tasks == ["ruler_v2_16k_g1", "ppl_16k_pg19val"]
    assert p.gpu_budget_h == 24.0 and p.dtype == "bfloat16" and "-devel" in p.image
    stage1 = load_pod("gate1_v2_stage1_llama")
    assert p.tasks == stage1.tasks and p.model == stage1.model  # byte-identical prompts and windows
    assert load_task("ruler_v2_16k_g1").n_trials == 24
