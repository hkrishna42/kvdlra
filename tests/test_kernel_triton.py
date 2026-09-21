"""The Triton backend reproduces the torch reference (the numerics contract) on CUDA, and
the kernel arm decodes end to end on a GPU model. All gpu-marked: skipped on the CPU gate,
run on the kernel-smoke pod before the pod's own tasks (`pytest -m gpu`).

The reference comparison is a DIRECT one -- same inputs, same bf16 operand dtype -- because
the pre-registered 1e-2 bar against reconstruct-then-attend cannot see a misplaced rounding
point (the review's mutation study moved the output by ~5.6e-4). R-L4-21 (amended by
R-L4-26 for the tight case) sets what it is measured on and where the bars sit:

* fp32 INPUTS (`random_case(..., torch.float32)`), so both backends return fp32 and the
  comparison is not quantized to a bf16 output ulp (3.9e-3 at |out| ~ 0.8 -- 2e-4 would sit
  in a dead zone where the only measurable values are 0 and one ulp). The OPERANDS stay
  bf16 on both sides: the kernel refuses anything else, and the reference rounds the same
  tensors at the same points either way, so every rounding point of the contract is still
  exercised.
* the tight case (B = 16 rows puts B*H_kv at TARGET_PROGRAMS, one split over the whole
  32-tile middle -- no per-split renormalization of P) asserts BOTH `TIGHT_MAX = 2e-3` on
  max|Δ| and `TIGHT_RMS = 1e-4` on rms(Δ) (R-L4-26, amending R-L4-21's single 2e-4 max
  bar). A one-row CPU emulation of the blocking measured the residual at 6.1e-5, but that
  is a PER-ROW number: the max over the whole B = 16 output runs over 16x the elements and
  reaches ~4e-4-9e-4 -- above the old single bar, though the kernel did nothing wrong. rms(Δ)
  over the same output stays ~1e-5, 30-75x under its own max, because the residual is a
  sparse handful of bf16 rounding flips per row; a MISPLACED rounding point instead shifts
  EVERY element, which rms (with ~8-10x headroom over the emulated floor) still catches
  even though max now has to tolerate the legitimate per-row scatter.
* across splits the per-split running max re-draws the P->bf16 rounding on every attention
  weight (measured 1.8e-3 on fp32 outputs at 16 splits), so the multi-split configuration
  and the batch-independence test take `SPLIT_TOL` -- ~2x that, while a layout, GQA or RoPE
  defect measures ~1.

All three bars print the measured value and carry it in the assert message, so a pod log
holds the number whether the item passes (`-rA`) or fails.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from kvdlra.cache import BugStreamingCache
from kvdlra.eval.frontier import _prefill_chunked
from kvdlra.kernel import FactoredMiddle, factored_attention, n_splits_for
from tests.test_kernel_reference import H_KV, llama_rope, random_case

pytestmark = pytest.mark.gpu
D = 128
TILE = 64  # the wrapper's default, the tile the split count is computed over
TIGHT_MAX = 2e-3  # R-L4-26: one split, fp32 outputs -- max|d| over the whole B=16 output
TIGHT_RMS = 1e-4  # R-L4-26: rms(d) over the same output -- catches a misplaced rounding point
SPLIT_TOL = 4e-3  # R-L4-21: with the per-split P-rounding draw in the way


def _cuda(mid: FactoredMiddle) -> FactoredMiddle:
    return FactoredMiddle(*(t.cuda() for t in (mid.u_k, mid.u_v, mid.c_k, mid.c_v, mid.positions)))


def _kw(emb: LlamaRotaryEmbedding) -> dict[str, Any]:
    return {
        "inv_freq": emb.inv_freq.cuda(),
        "attention_scaling": emb.attention_scaling,
        "scaling": D**-0.5,
    }


def _triton_vs_reference(
    q: Tensor, dk: Tensor, dv: Tensor, mid: FactoredMiddle, emb: LlamaRotaryEmbedding
) -> tuple[Tensor, Tensor]:
    """Both backends on the same (fp32) inputs with bf16 operands: the kernel's rounding
    points against the reference's, with no output quantization in the way. Returns the
    triton output and the raw (fp32, on CPU) elementwise difference -- callers reduce it
    to max|Δ| and/or rms(Δ) themselves (R-L4-26)."""
    tri = factored_attention(q.cuda(), dk.cuda(), dv.cuda(), _cuda(mid),
                             backend="triton", **_kw(emb))  # fmt: skip
    ref = factored_attention(q, dk, dv, mid, backend="reference", inv_freq=emb.inv_freq,
                             attention_scaling=emb.attention_scaling, scaling=D**-0.5)  # fmt: skip
    assert tri.dtype == q.dtype and tri.shape == q.shape
    return tri, tri.float().cpu() - ref.float()


@pytest.mark.parametrize("contiguous", [True, False])
def test_triton_matches_the_reference(contiguous: bool) -> None:
    """Two configurations of the same comparison (R-L4-21/R-L4-26): B = 16 rows, where the
    whole 32-tile middle runs in a single split and BOTH the max and rms bars apply; and
    the single row, where the same middle is cut across 16 splits and the bar carries the
    P-rounding draw."""
    emb = llama_rope(D, 4096, 8.0)
    rows = [random_case(seed, contiguous, torch.float32) for seed in range(100, 116)]
    mid = FactoredMiddle.cat([r[3] for r in rows])
    q = torch.cat([r[0] for r in rows])
    dk, dv = torch.cat([r[1] for r in rows]), torch.cat([r[2] for r in rows])
    n_tiles = -(-mid.n_columns // TILE)
    assert n_splits_for(len(rows), H_KV, n_tiles) == 1, "the tight bar is the one-split bar"
    _, d = _triton_vs_reference(q, dk, dv, mid, emb)
    mx, rms = float(d.abs().max()), float((d.float() ** 2).mean().sqrt())
    print(
        f"triton vs reference, 1 split (contiguous={contiguous}): max|d| = {mx:.3e} rms = {rms:.3e}"
    )
    assert mx <= TIGHT_MAX, f"1 split, B={len(rows)}: max|d| = {mx:.3e} > {TIGHT_MAX:.0e}"
    assert rms <= TIGHT_RMS, f"1 split, B={len(rows)}: rms(d) = {rms:.3e} > {TIGHT_RMS:.0e}"

    q1, dk1, dv1, mid1 = rows[0]
    assert n_splits_for(1, H_KV, n_tiles) == 16
    tri1, d1 = _triton_vs_reference(q1, dk1, dv1, mid1, emb)
    split_delta = float(d1.abs().max())
    print(f"triton vs reference, 16 splits (contiguous={contiguous}): max|d| = {split_delta:.3e}")
    assert split_delta <= SPLIT_TOL, f"16 splits, B=1: max|d| = {split_delta:.3e} > {SPLIT_TOL:.0e}"
    auto = factored_attention(q1.cuda(), dk1.cuda(), dv1.cuda(), _cuda(mid1), backend="auto",
                              **_kw(emb))  # fmt: skip
    assert torch.equal(auto, tri1)  # auto picks triton on a CUDA query at these shapes


def test_triton_batch_rows_and_splits_are_independent() -> None:
    """B = 2 (the stacked middles of two rows, different columns and positions) equals two
    B = 1 calls; the split count differs between them (2048 columns = 32 tiles, so 8 splits
    at B = 2 and 16 at B = 1), which is a different P-rounding draw -- hence `SPLIT_TOL`,
    while batch cross-talk is O(1)."""
    emb = llama_rope(D, 4096, 8.0)
    a, b = random_case(4, True, torch.float32), random_case(5, False, torch.float32)
    kw = _kw(emb)
    both = FactoredMiddle.cat([_cuda(a[3]), _cuda(b[3])])
    q = torch.cat([a[0], b[0]]).cuda()
    dk, dv = torch.cat([a[1], b[1]]).cuda(), torch.cat([a[2], b[2]]).cuda()
    out = factored_attention(q, dk, dv, both, backend="triton", **kw)
    for i, case in enumerate((a, b)):
        one = factored_attention(case[0].cuda(), case[1].cuda(), case[2].cuda(), _cuda(case[3]),
                                 backend="triton", **kw)  # fmt: skip
        delta = float((out[i : i + 1].float() - one.float()).abs().max())
        print(f"triton B=2 row {i} vs its own B=1 call: max|d| = {delta:.3e}")
        assert delta <= SPLIT_TOL, f"row {i}: B=2 vs B=1 max|d| = {delta:.3e} > {SPLIT_TOL:.0e}"


def test_triton_no_middle_and_refusals() -> None:
    # The only static import of the module anywhere: `kvdlra.kernel.factored_attention`
    # reaches it through `importlib.import_module` (it needs triton, absent from a CPU
    # tree), which tests/test_reachability.py's AST walk cannot see. Inside a gpu-marked
    # body it runs on the pod only, and pins the two documented constants.
    from kvdlra.kernel import triton_kernel

    assert (triton_kernel.G_PAD, triton_kernel.TARGET_PROGRAMS) == (16, 128)
    emb = llama_rope(D, 4096, 8.0)
    q, dk, dv, mid = random_case(6, True, torch.bfloat16)
    kw = _kw(emb)
    out = factored_attention(q.cuda(), dk.cuda(), dv.cuda(), None, backend="triton", **kw)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), dk.float(), dv.float(), scale=D**-0.5, enable_gqa=True
    )
    assert float((out.float().cpu() - ref).abs().max()) < 1e-2
    with pytest.raises(ValueError, match="bf16"):
        factored_attention(q.cuda(), dk.cuda(), dv.cuda(), _cuda(mid), backend="triton",
                           operand_dtype=torch.float32, **kw)  # fmt: skip
    # M-5/M-6/M-7: the Python-side checks that stand between a bad call and an unmasked
    # `tl.load` of the wrong memory. `inv_freq` is read as (D // 2,) with no mask; a host
    # tensor reaches the kernel as a host pointer; the middle is indexed by the query's b.
    with pytest.raises(ValueError, match="inv_freq"):
        factored_attention(q.cuda(), dk.cuda(), dv.cuda(), _cuda(mid), backend="triton",
                           **{**kw, "inv_freq": emb.inv_freq[: D // 4].cuda()})  # fmt: skip
    with pytest.raises(ValueError, match="on the host"):
        factored_attention(q.cuda(), dk.cuda(), dv.cuda(), mid, backend="triton", **kw)
    with pytest.raises(ValueError, match="batch rows"):
        factored_attention(q.cuda(), dk.cuda(), dv.cuda(),
                           FactoredMiddle.cat([_cuda(mid), _cuda(mid)]),
                           backend="triton", **kw)  # fmt: skip
    # P-1: r = 48 sits between the kernel's power-of-two floors (FactoredMiddle itself has
    # no such constraint -- __post_init__ only ties shapes together). "auto" has to fall
    # back to the reference instead of raising the kernel's own refusal mid-decode
    # (R-L4-22); an explicit backend="triton" still raises it.
    r48 = FactoredMiddle(
        mid.u_k[:, :, :48], mid.u_v[:, :, :48], mid.c_k[:, :48], mid.c_v[:, :48], mid.positions
    )
    cuda_r48 = _cuda(r48)
    auto48 = factored_attention(q.cuda(), dk.cuda(), dv.cuda(), cuda_r48, backend="auto", **kw)
    ref48 = factored_attention(q.cuda(), dk.cuda(), dv.cuda(), cuda_r48, backend="reference", **kw)
    assert torch.equal(auto48, ref48)  # the fallback took the reference
    with pytest.raises(ValueError, match="power of two"):
        factored_attention(q.cuda(), dk.cuda(), dv.cuda(), cuda_r48, backend="triton", **kw)


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
