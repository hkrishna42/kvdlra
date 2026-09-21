"""`decode_attention: kernel` -- default off and bit-identical (the golden), on: a decode
step returns only the dense tokens, the factored middle rides on the layer, and the attention
function registered for the attach scope attends both (prereg/kernel_smoke.md §3's "a
bug-kind arm carrying a config knob"; ADR 0001 §5).

The full-model greedy token-exact check of GATES §G4 line 2 runs here on the tiny model with
fp32 tile operands: the two paths then differ only by summation order, so the 16 committed
prompts must agree on every greedy token AND on every step's logits to 1e-4 -- expected 16/16
against the pre-registered >= 14/16 (the pod repeats the check on Llama-3.1-8B in bf16, where
the bar applies). Chunked prefill under the registered name exercises the ingest mask the
function must receive for `q_len > 1`. The `exact_tier` parametrization repeats this on 4 of
the 16 prompts (a live exact tier exercises different code paths without doubling the
module's CPU budget); the 16-prompt precondition itself is pinned by the `plain` variant.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from transformers import LlamaForCausalLM
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, rotate_half

from kvdlra.cache import BugStreamingCache
from kvdlra.eval.config import arm_kwargs, load_arm
from kvdlra.eval.frontier import build_arm
from kvdlra.kernel import factored_attention
from kvdlra.kernel.attention import KERNEL_ATTN
from kvdlra.kernel.prompts import TINY_N_NEW, TINY_PROMPT_TOKENS, tiny_prompts
from kvdlra.kernel.reference import rope_cos_sin
from tests.conftest import tiny_cache
from tests.test_batched_cache import _teacher_forced

TINY_SDPA = True
CHUNK = 32
TIER = {
    "retention": "lowrank_surprise", "hh_budget": 8, "hh_select": "surprise",
    "hh_neighbor": 1, "seed_hh_warmup": True,
}  # fmt: skip


@torch.no_grad()
def _greedy(
    model: LlamaForCausalLM, cache: BugStreamingCache, ids: torch.Tensor, n_new: int
) -> tuple[list[int], list[torch.Tensor]]:
    """Chunked prefill, then greedy decode at true positions under the cache's attach scope
    (the first token is the prefill's own prediction). Returns the tokens fed and the
    per-step logits."""
    t = int(ids.shape[1])
    toks: list[int] = []
    logits: list[torch.Tensor] = []
    with cache.attach(model), cache.ingesting():
        for start in range(0, t, CHUNK):
            stop = min(t, start + CHUNK)
            pos = torch.arange(start, stop).unsqueeze(0)
            out = model(ids[:, start:stop], past_key_values=cache, use_cache=True,
                        position_ids=pos, logits_to_keep=1)  # fmt: skip
            cache.consolidate()
    tok = out.logits[:, -1].argmax(-1).view(1, 1)
    with cache.attach(model):
        for s in range(n_new):
            toks.append(int(tok))
            pos = torch.tensor([[t + s]])
            out = model(tok, past_key_values=cache, use_cache=True, position_ids=pos)
            logits.append(out.logits[:, -1].float())
            tok = out.logits[:, -1].argmax(-1).view(1, 1)
    return toks, logits


def test_default_is_reconstruct_and_the_knob_is_validated(tiny_model: LlamaForCausalLM) -> None:
    cache = tiny_cache(tiny_model)
    assert cache.decode_attention == "reconstruct"
    assert all(la.decode_attention == "reconstruct" for la in cache._bug_layers())
    assert all(la.kernel_operand_dtype is torch.bfloat16 for la in cache._bug_layers())
    with pytest.raises(ValueError, match="decode_attention"):
        tiny_cache(tiny_model, decode_attention="triton")
    with pytest.raises(ValueError, match="kernel_operand_dtype"):
        tiny_cache(tiny_model, kernel_operand_dtype="float16")
    cache = tiny_cache(tiny_model, decode_attention="kernel", kernel_operand_dtype="float32")
    assert all(la.kernel_operand_dtype is torch.float32 for la in cache._bug_layers())


def test_reconstruct_arm_never_registers_and_a_decode_step_returns_everything(
    tiny_model: LlamaForCausalLM,
) -> None:
    cache = tiny_cache(tiny_model)
    before = tiny_model.config._attn_implementation
    with cache.attach(tiny_model):
        assert tiny_model.config._attn_implementation == before
    _greedy(tiny_model, cache, tiny_prompts()[0][None], 6)
    layer = cache._bug_layers()[0]
    assert layer.kernel_middles() is None and not layer._kernel_step
    k, _ = layer._decode_peek()
    assert int(k.shape[2]) == layer.attended_length()


def test_kernel_step_returns_dense_only_and_the_scope_registers(
    tiny_model: LlamaForCausalLM,
) -> None:
    cache = tiny_cache(tiny_model, decode_attention="kernel", kernel_operand_dtype="float32")
    before = tiny_model.config._attn_implementation
    with cache.attach(tiny_model):
        assert tiny_model.config._attn_implementation == KERNEL_ATTN
        assert KERNEL_ATTN in ALL_ATTENTION_FUNCTIONS
        assert KERNEL_ATTN in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    assert tiny_model.config._attn_implementation == before
    _greedy(tiny_model, cache, tiny_prompts()[1][None], 8)
    layer = cache._bug_layers()[0]
    mids = layer.kernel_middles()
    assert layer._kernel_step and mids is not None and len(mids) == 1
    mid = mids[0]
    assert mid is not None and mid.n_columns == layer._mid_len() > 0  # holds only without an
    # exact tier -- `_mid_len()` includes `hh`; this test's cache carries no exact tier
    k_dense, _ = layer._decode_peek(dense_only=True)
    assert int(k_dense.shape[2]) + mid.n_columns == layer.attended_length()
    assert layer.get_mask_sizes(1)[0] == layer.attended_length() + 1  # the mask reports the
    # returned block incl. the new token
    # outside the scope the stub refuses rather than attending with a stale cache
    fn = ALL_ATTENTION_FUNCTIONS[KERNEL_ATTN]
    with pytest.raises(RuntimeError, match="no cache attached"):
        fn(None, None, None, None, None)


def test_batch_1_hands_the_layers_own_middle_to_the_kernel(
    tiny_model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At batch 1 there is one row, and `FactoredMiddle.cat` of one row copies every tensor
    in it -- C_k, C_v, U_k, U_v and the positions, every layer every step (~17 MB per layer
    per step at 32K, r64), for a middle that is already exactly what the kernel attends.
    What reaches `factored_attention` is therefore the layer's own `_factored_middle()`
    object, and its tensors are the layer's own; the selected backend is attested once on
    the cache (R-L4-22: `auto` degrades to the reference silently otherwise)."""
    seen: list[Any] = []

    def spy(*a: Any, **kw: Any) -> torch.Tensor:
        seen.append(a[3])  # the `mid` positional
        return factored_attention(*a, **kw)

    monkeypatch.setattr("kvdlra.kernel.attention.factored_attention", spy)
    cache = tiny_cache(tiny_model, decode_attention="kernel", kernel_operand_dtype="float32")
    _greedy(tiny_model, cache, tiny_prompts()[1][None], 1)  # one step: one call per layer
    layer = cache._bug_layers()[0]
    assert len(seen) == len(cache._bug_layers())
    assert layer._kernel_mid is not None and layer.u_k is not None
    assert seen[0] is layer._kernel_mid  # the object itself, not a copy of it
    assert seen[0].u_k.data_ptr() == layer.u_k.data_ptr()  # ... and its tensors are the layer's
    assert cache.kernel_backend == "reference"  # on a CPU query `auto` cannot pick triton


def test_kernel_middle_positions_match_reconstruct_under_fifo_with_an_exact_tier(
    tiny_model: LlamaForCausalLM,
) -> None:
    """R-L4-18: `_factored_middle` must position the kernel's middle exactly as
    `_ensure_mid_cache` positions the reconstruct path's -- both must leave room for
    `_hh_attended_len()` columns ahead of the low-rank block, not just the block's own
    length, or a live exact tier under fifo retention shifts the kernel middle off the
    reconstruct path's (the bug this test pins: unfixed, K differed by 0.75 max abs; V,
    never RoPE'd, was already identical). Rebuilds the kernel's middle by hand -- U @ C,
    in-kernel RoPE via `rope_cos_sin` at `mid.positions` and the model's own
    `inv_freq`/`attention_scaling` -- and checks it against `_mid_k_cache`/`_mid_v_cache`
    after a plain `_decode_peek()` forces the reconstruct path to (re)build them."""
    cache = tiny_cache(
        tiny_model, retention="fifo", hh_budget=8, hh_select="surprise",
        decode_attention="kernel", kernel_operand_dtype="float32",
    )  # fmt: skip
    _greedy(tiny_model, cache, tiny_prompts()[0][None], 1)
    layer = cache._bug_layers()[0]
    assert layer._hh_attended_len() > 0  # the tier must be live for R-L4-18 to bite
    mids = layer.kernel_middles()
    assert mids is not None
    mid = mids[0]
    assert mid is not None and mid.n_columns > 0
    layer._decode_peek()  # off the kernel step: forces `_ensure_mid_cache` (reconstruct)
    assert layer._mid_k_cache is not None and layer._mid_v_cache is not None
    rotary = cast(LlamaRotaryEmbedding, tiny_model.model.rotary_emb)
    inv_freq = rotary.inv_freq
    scaling = float(getattr(rotary, "attention_scaling", 1.0))
    h, d = layer.num_heads, layer.head_dim
    k_pre = (mid.u_k[0] @ mid.c_k[0]).reshape(h, d, -1).permute(0, 2, 1)  # (H, T, D)
    cos, sin = rope_cos_sin(mid.positions, inv_freq, scaling)
    rot = rotate_half(k_pre)  # type: ignore[no-untyped-call]  # unannotated upstream
    k_hat = (k_pre * cos[0] + rot * sin[0]).permute(0, 2, 1).reshape(h * d, -1)
    v_hat = mid.u_v[0] @ mid.c_v[0]
    k_diff = float((k_hat - layer._mid_k_cache.float()).abs().max())
    v_diff = float((v_hat - layer._mid_v_cache.float()).abs().max())
    assert k_diff < 1e-5, k_diff
    assert v_diff < 1e-5, v_diff


@pytest.mark.parametrize(
    ("extra", "prompts"),
    [({}, tiny_prompts()), (TIER, tiny_prompts()[:4])],
    ids=["plain", "exact_tier"],
)
def test_the_16_prompts_are_token_exact_in_fp32(
    tiny_model: LlamaForCausalLM, extra: dict[str, Any], prompts: list[torch.Tensor]
) -> None:
    mismatches: list[tuple[int, int]] = []
    for i, prompt in enumerate(prompts):
        ids = prompt[None]
        assert ids.shape == (1, TINY_PROMPT_TOKENS)
        kern = tiny_cache(
            tiny_model, decode_attention="kernel", kernel_operand_dtype="float32", **extra
        )
        recon = tiny_cache(tiny_model, decode_attention="reconstruct", **extra)
        toks_k, logits_k = _greedy(tiny_model, kern, ids, TINY_N_NEW)
        if not extra:  # "plain": the kernel path never builds the reconstruct workspace --
            assert kern.workspace_numel() == 0  # the observable proof it ran, uninstrumented
        toks_r, logits_r = _greedy(tiny_model, recon, ids, TINY_N_NEW)
        steps = enumerate(zip(toks_k, toks_r, strict=True))
        first = next((s for s, (a, b) in steps if a != b), None)
        if first is not None:
            mismatches.append((i, first))
        for s, (lk, lr) in enumerate(zip(logits_k, logits_r, strict=True)):
            assert torch.allclose(lk, lr, atol=1e-4), (i, s, float((lk - lr).abs().max()))
    # all-agree: 16/16 on "plain" (the pre-registered >= 14/16 precondition itself); 4/4 on
    # "exact_tier" (a narrower plumbing check kept small for the CPU budget)
    assert mismatches == [], mismatches


def test_batched_kernel_decode_equals_two_batch1_kernel_decodes(
    tiny_model: LlamaForCausalLM,
) -> None:
    ps = tiny_prompts()
    ids = torch.stack([ps[2], ps[3]])
    stream = torch.randint(0, 256, (2, 8), generator=torch.Generator().manual_seed(4))
    kw = {"decode_attention": "kernel", "kernel_operand_dtype": "float32"}

    def run(cache: BugStreamingCache, x: torch.Tensor, st: torch.Tensor) -> list[torch.Tensor]:
        return _teacher_forced(tiny_model, cache, x, st, CHUNK)

    both = run(tiny_cache(tiny_model, **kw), ids, stream)
    singles = [
        run(tiny_cache(tiny_model, **kw), ids[b : b + 1], stream[b : b + 1]) for b in range(2)
    ]
    for s in range(stream.shape[1]):
        for b in range(2):
            assert torch.allclose(both[s][b], singles[b][s][0], atol=1e-4), (s, b)


def test_the_kernel_arm_is_arm_2_plus_the_knob() -> None:
    kern, base = load_arm("isvd_r64_h256_seed_kernel"), load_arm("isvd_r64_h256_seed")
    assert kern.kind == "bug" and kern.legacy_name is None and kern.chunkable
    assert kern.cache == {**base.cache, "decode_attention": "kernel"}
    for t in (16384, 32768):
        want = {**arm_kwargs(base, t), "decode_attention": "kernel"}
        assert build_arm(kern, None, t)["kwargs"] == want
    assert build_arm(kern, None, 16384)["name"] == "isvd_r64_h256_seed_kernel"
