"""The Triton backend reproduces the torch reference (the numerics contract) on CUDA, and
the kernel arm decodes end to end on a GPU model. All gpu-marked: skipped on the CPU gate,
run on the kernel-smoke pod before the pod's own tasks (`pytest -m gpu`).

The reference comparison is a DIRECT one (same inputs, same bf16 operand dtype) at 2e-4
(R-L4-17): the pre-registered 1e-2 bar against reconstruct-then-attend cannot see a
misplaced rounding point -- the review's mutation study moved the output by ~5.6e-4 -- so
the tile loop's own rounding points are pinned an order of magnitude tighter here. The
measured value rides in the assert message, so a pod log carries it.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.eval.frontier import _prefill_chunked
from kvdlra.kernel import FactoredMiddle, factored_attention
from tests.test_kernel_reference import llama_rope, random_case

pytestmark = pytest.mark.gpu
D = 128
REF_TOL = 2e-4  # R-L4-17: the rounding points, not just the reconstruct bar


def _cuda(mid: FactoredMiddle) -> FactoredMiddle:
    return FactoredMiddle(*(t.cuda() for t in (mid.u_k, mid.u_v, mid.c_k, mid.c_v, mid.positions)))


@pytest.mark.parametrize("contiguous", [True, False])
def test_triton_matches_the_reference(contiguous: bool) -> None:
    emb = llama_rope(D, 4096, 8.0)
    q, dk, dv, mid = random_case(3, contiguous, torch.bfloat16)
    kw: dict[str, Any] = {
        "inv_freq": emb.inv_freq.cuda(),
        "attention_scaling": emb.attention_scaling,
        "scaling": D**-0.5,
    }
    tri = factored_attention(q.cuda(), dk.cuda(), dv.cuda(), _cuda(mid), backend="triton", **kw)
    ref = factored_attention(q, dk, dv, mid, backend="reference", inv_freq=emb.inv_freq,
                             attention_scaling=emb.attention_scaling, scaling=D**-0.5)  # fmt: skip
    assert tri.dtype == q.dtype and tri.shape == q.shape
    delta = float((tri.float().cpu() - ref.float()).abs().max())
    print(f"triton vs reference (contiguous={contiguous}): max|d| = {delta:.3e}")
    assert delta <= REF_TOL, f"triton vs reference max|d| = {delta:.3e} > {REF_TOL:.0e}"
    auto = factored_attention(q.cuda(), dk.cuda(), dv.cuda(), _cuda(mid), backend="auto", **kw)
    assert torch.equal(auto, tri)  # auto picks triton on a CUDA query


def test_triton_batch_rows_and_splits_are_independent() -> None:
    """B = 2 (the stacked middles of two rows, different columns and positions) equals two
    B = 1 calls; n_splits > 1 on both (2048 columns = 32 tiles, 128 / 16 = 8 splits)."""
    emb = llama_rope(D, 4096, 8.0)
    a, b = random_case(4, True, torch.bfloat16), random_case(5, False, torch.bfloat16)
    kw: dict[str, Any] = {
        "inv_freq": emb.inv_freq.cuda(),
        "attention_scaling": emb.attention_scaling,
        "scaling": D**-0.5,
    }
    both = FactoredMiddle.cat([_cuda(a[3]), _cuda(b[3])])
    q = torch.cat([a[0], b[0]]).cuda()
    dk, dv = torch.cat([a[1], b[1]]).cuda(), torch.cat([a[2], b[2]]).cuda()
    out = factored_attention(q, dk, dv, both, backend="triton", **kw)
    for i, case in enumerate((a, b)):
        one = factored_attention(case[0].cuda(), case[1].cuda(), case[2].cuda(), _cuda(case[3]),
                                 backend="triton", **kw)  # fmt: skip
        assert float((out[i : i + 1].float() - one.float()).abs().max()) < 1e-3


def test_triton_no_middle_and_refusals() -> None:
    # The only static import of the module anywhere: `kvdlra.kernel.factored_attention`
    # reaches it through `importlib.import_module` (it needs triton, absent from a CPU
    # tree), which tests/test_reachability.py's AST walk cannot see. Inside a gpu-marked
    # body it runs on the pod only, and pins the two documented constants.
    from kvdlra.kernel import triton_kernel

    assert (triton_kernel.G_PAD, triton_kernel.TARGET_PROGRAMS) == (16, 128)
    emb = llama_rope(D, 4096, 8.0)
    q, dk, dv, mid = random_case(6, True, torch.bfloat16)
    kw: dict[str, Any] = {
        "inv_freq": emb.inv_freq.cuda(),
        "attention_scaling": emb.attention_scaling,
        "scaling": D**-0.5,
    }
    out = factored_attention(q.cuda(), dk.cuda(), dv.cuda(), None, backend="triton", **kw)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), dk.float(), dv.float(), scale=D**-0.5, enable_gqa=True
    )
    assert float((out.float().cpu() - ref).abs().max()) < 1e-2
    with pytest.raises(ValueError, match="bf16"):
        factored_attention(q.cuda(), dk.cuda(), dv.cuda(), _cuda(mid), backend="triton",
                           operand_dtype=torch.float32, **kw)  # fmt: skip


def test_kernel_arm_decodes_on_a_cuda_model() -> None:
    """A random-weight Llama with head_dim 64 (the kernel needs D >= 32, r >= 16) on CUDA in
    bf16: chunked prefill + 24 greedy steps under the kernel; the reconstruct twin's logits
    stay within 5e-2 of it (bf16 through 2 layers) and the tokens mostly agree."""
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=8, num_key_value_heads=2, head_dim=64,
                      max_position_embeddings=4096)  # fmt: skip
    model = LlamaForCausalLM(cfg).to("cuda", torch.bfloat16).eval()  # type: ignore[no-untyped-call,arg-type]
    model.config._attn_implementation = "sdpa"
    ids = torch.randint(0, 256, (1, 256), generator=torch.Generator().manual_seed(1)).cuda()
    kw: dict[str, Any] = {
        "rank": 16,
        "coord_budget": 512,
        "recent_window": 32,
        "absorb_block": 16,
        "n_sink": 4,
    }

    @torch.no_grad()
    def run(cache: BugStreamingCache) -> tuple[list[int], list[torch.Tensor]]:
        toks, logits = [], []
        with cache.attach(model):
            _prefill_chunked(model, cache, ids, 128)
            tok = torch.zeros((1, 1), dtype=torch.long, device="cuda")
            for s in range(24):
                out = model(tok, past_key_values=cache, use_cache=True,
                            position_ids=torch.tensor([[256 + s]], device="cuda"))  # fmt: skip
                logits.append(out.logits[:, -1].float())
                tok = out.logits[:, -1].argmax(-1).view(1, 1)
                toks.append(int(tok))
        return toks, logits

    tk, lk = run(BugStreamingCache(model, decode_attention="kernel", **kw))
    tr, lr = run(BugStreamingCache(model, decode_attention="reconstruct", **kw))
    assert max(float((a - b).abs().max()) for a, b in zip(lk, lr, strict=True)) < 5e-2
    assert sum(a == b for a, b in zip(tk, tr, strict=True)) >= 18
