"""Week-19 A1: a KIVI-faithful quantized-KV baseline (the panel's decisive experiment).

Week-18 ran transformers' QuantizedCache at its default axes and got 0.00 retrieval at
2 AND 4 bits. In optimum-quanto's grouping semantics axis 0 = per-TOKEN groups (64
consecutive elements of one token row) and axis -1 = per-CHANNEL groups (64 consecutive
tokens of one head-dim channel). KIVI quantizes keys per-channel (outlier channels get
their own scale) and values per-token; the W18 default quantized keys per-token, which
one large key-bias channel (Qwen2.5) reduces to noise. The per-channel axis needs
B*H*T divisible by the group size (the "group 64 must divide 65588" SKIP), so the
per-channel path pads T by edge-replication and slices it off on dequantize.

`kvdlra.quant.kivi_cache.make_quant_cache` builds the arm's cache for either scheme
("token" = upstream default, bit-identical; "kivi" = per-channel keys + per-token
values) on either backend ("quanto" 2/4-bit; "hqq" 1-8 bit, the 8-bit control).
"""

from __future__ import annotations

import argparse
from typing import Any, cast

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import QuantizedCache

from kvdlra.quant.kivi_cache import aux_words, make_quant_cache

H, D = 2, 16  # KV heads x head_dim
# Not a multiple of the group -> exercises the padding path. Even, because the tiny D=16
# needs B*H*T*D divisible by 64 for the per-token axis (D=128 satisfies it for any T).
T_ODD = 78


def _cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=H,
        head_dim=D,
        max_position_embeddings=4096,
    )


def _kv(t: int = T_ODD, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-variance keys with one Qwen-like outlier channel (+50), plain values."""
    g = torch.Generator().manual_seed(0)
    k = torch.randn(1, H, t, D, generator=g)
    k[..., 3] += 50.0
    v = torch.randn(1, H, t, D, generator=g)
    return k.to(dtype), v.to(dtype)


def _prefill_dequant(cache: Any, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, Any]:
    layer: Any = cache.layers[0]
    layer.update(k, v)  # prefill: quantizes everything, returns exact states
    return layer._dequantize(layer._quantized_keys), layer


def test_kivi_keys_are_per_channel_and_odd_T_round_trips() -> None:
    """Per-channel key quant must (a) restore the true odd T after padding and (b)
    isolate the outlier channel: non-outlier error far below the per-token scheme."""
    k, v = _kv()
    deq_tok, _ = _prefill_dequant(
        make_quant_cache(_cfg(), nbits=4, scheme="token", backend="quanto", group=64), k, v
    )
    deq_kivi, _ = _prefill_dequant(
        make_quant_cache(_cfg(), nbits=4, scheme="kivi", backend="quanto", group=64), k, v
    )
    assert deq_kivi.shape == k.shape == deq_tok.shape
    keep = [c for c in range(D) if c != 3]
    err_tok = (deq_tok - k)[..., keep].abs().mean()
    err_kivi = (deq_kivi - k)[..., keep].abs().mean()
    assert err_kivi < 0.2 * err_tok, (float(err_kivi), float(err_tok))


def test_token_scheme_is_bit_identical_to_upstream_default() -> None:
    """scheme='token' (the default arm) must reproduce transformers' own QuantizedCache
    exactly -- the W18 quant-{2,4}bit rows keep their meaning."""
    k, v = _kv()
    ours, _ = _prefill_dequant(
        make_quant_cache(_cfg(), nbits=2, scheme="token", backend="quanto", group=64), k, v
    )
    upstream = QuantizedCache(
        backend="quanto", config=_cfg(), nbits=2, axis_key=0, axis_value=0, q_group_size=64
    )
    theirs, _ = _prefill_dequant(upstream, k, v)
    assert torch.equal(ours, theirs)


def test_hqq_8bit_kivi_control_is_near_lossless() -> None:
    """The 8-bit control (hqq backend, per-channel keys): a decode path that scores 0
    with THIS cache is broken, so its round-trip error must be tiny even with the
    outlier channel, and the odd T must round-trip through the transpose+pad path."""
    k, v = _kv()
    deq, layer = _prefill_dequant(
        make_quant_cache(_cfg(), nbits=8, scheme="kivi", backend="hqq", group=64), k, v
    )
    assert deq.shape == k.shape
    rel = ((deq - k).abs().mean() / k.abs().mean()).item()
    assert rel < 0.01, rel
    deq_v = layer._dequantize(layer._quantized_values)
    assert deq_v.shape == v.shape


def test_flush_quantizes_the_residual_after_chunked_prefill() -> None:
    """Chunked prefill leaves the last chunk in the fp16 residual; flush() folds it
    into the quantized store so decode starts from the same fully-quantized state a
    single-shot prefill produces (fair to the baseline, comparable across chunkings)."""
    k, v = _kv(200)
    cache = make_quant_cache(_cfg(), nbits=4, scheme="kivi", backend="quanto", group=64)
    layer: Any = cache.layers[0]
    layer.update(k[..., :100, :], v[..., :100, :])
    layer.update(k[..., 100:, :], v[..., 100:, :])
    assert layer.keys.shape[-2] == 100  # second chunk parked in the residual
    layer.flush()
    assert layer.keys.numel() == 0
    assert layer._dequantize(layer._quantized_keys).shape[-2] == 200
    assert layer.get_seq_length() == 200
    layer.flush()  # idempotent on an empty residual
    assert layer._dequantize(layer._quantized_keys).shape[-2] == 200


def test_aux_words_bills_the_stored_scale_and_zero_dtype() -> None:
    """Accounting must bill the aux (scale+zero per group) at what the backend actually
    stores: 2 fp32 words on an fp32 model, 1 word (2 x 16-bit) on a bf16 model."""
    k32, v32 = _kv()
    c32 = make_quant_cache(_cfg(), nbits=4, scheme="kivi", backend="quanto", group=64)
    cast(Any, c32.layers[0]).update(k32, v32)
    assert aux_words(c32) == 2.0
    k16, v16 = _kv(dtype=torch.bfloat16)
    c16 = make_quant_cache(_cfg(), nbits=4, scheme="kivi", backend="quanto", group=64)
    cast(Any, c16.layers[0]).update(k16, v16)
    assert aux_words(c16) == 1.0


def test_invalid_scheme_backend_combos_fail_loud() -> None:
    with pytest.raises(ValueError, match="nbits"):
        make_quant_cache(_cfg(), nbits=8, scheme="kivi", backend="quanto")  # quanto: 2/4 only
    with pytest.raises(ValueError, match="scheme"):
        make_quant_cache(_cfg(), nbits=4, scheme="channel", backend="quanto")


# ------------------------------------------------------------ harness plumbing


class _StubTok:
    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


def _model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    m = LlamaForCausalLM(_cfg())  # type: ignore[no-untyped-call]
    m.config._attn_implementation = "sdpa"
    m.eval()  # type: ignore[no-untyped-call]
    return m


def _args(**kw: Any) -> argparse.Namespace:
    from w10_frontier import build_parser

    ns = build_parser().parse_args([])
    ns.methods = ["quant"]
    ns.chunk = 0
    for key, val in kw.items():
        setattr(ns, key, val)
    return ns


def test_build_arms_names_encode_scheme_and_backend() -> None:
    from w10_frontier import build_arms

    m = _model()
    assert [a["name"] for a in build_arms(_args(), m, 200)] == ["quant-2bit", "quant-4bit"]
    kivi = build_arms(_args(quant_scheme="kivi"), m, 200)
    assert [a["name"] for a in kivi] == ["quant-2bit-kivi", "quant-4bit-kivi"]
    hqq8 = build_arms(_args(quant_scheme="kivi", quant_backend="hqq", quant_nbits=[8]), m, 200)
    assert [a["name"] for a in hqq8] == ["quant-8bit-kivi-hqq"]
    assert all(a["chunkable"] is True for a in kivi + hqq8)  # quant now honors --chunk


def test_quant_arm_retrieves_with_chunked_prefill() -> None:
    """retrieve() with --chunk > 0 must run the quant arm through the chunked prefill
    (+ flush) and still return a measurement; single-shot (chunk=0) too."""
    from w10_frontier import build_arms
    from w10_ruler import retrieve

    from kvdlra.press.compat import install_kvpress_prefill_compat

    install_kvpress_prefill_compat()
    m = _model()
    hay = torch.randint(0, 256, (1, 200))
    query = torch.randint(0, 256, (1, 8))
    for chunk in (64, 0):
        for arm in build_arms(_args(quant_scheme="kivi", quant_nbits=[4]), m, 200):
            hit, ratio, _frac, sratio = retrieve(
                m, _StubTok(), arm, hay, query, ["1"], "cpu", chunk, H * D, H, 4
            )
            assert isinstance(hit, bool)
            assert 0.0 < ratio <= 1.0 and ratio == sratio


def test_score_quant_runs_without_autograd() -> None:
    """The ppl path must not retain the prefill graph: the W18/W19 quant-ppl OOMs (38 GB
    allocated during a 4K chunk on Qwen-7B) were an undecorated score_quant building
    autograd history across the whole prefill. Dequantized state must carry no grad."""
    from w10_frontier import build_arms, score_quant

    m = _model()
    arm = build_arms(_args(quant_scheme="kivi", quant_nbits=[4]), m, 256)[0]
    cache = arm["make"]()
    ctx = torch.randint(0, 256, (256,))
    win = torch.randint(0, 256, (16,))
    nll, ntok = score_quant(m, cache, ctx, win, chunk=64)
    assert ntok == 15 and nll > 0.0
    layer: Any = cache.layers[0]
    assert not layer._dequantize(layer._quantized_keys).requires_grad
