"""L2.2: the faithful KIVI arm -- G=32, R=128, full-precision single-shot prefill, then the
quantized store built post hoc (Liu et al. 2024). The streaming arm (``kivi_cache``: G=64,
chunked prefill, later chunks attending to already-quantized context) is the arm the
paper-v1 tables used and stays as it is; CLAUDE.md names this one as a separate arm.

hqq backend throughout: pure torch on CPU, no JIT. Two layers, T <= 300: the suite's
90-second gate.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from kvdlra import accounting as acc
from kvdlra.eval import frontier, longbench, ruler
from kvdlra.eval.config import load_arm
from kvdlra.eval.frontier import build_arm
from kvdlra.quant.kivi import make_kivi, quantize_after_prefill, residual_tokens
from kvdlra.quant.kivi_cache import aux_words, make_quant_cache

H_KV, D = 2, 16  # KV heads x head_dim -> n = 32 features per layer
TOK = SimpleNamespace(decode=lambda ids: " ".join(str(i) for i in ids))


def _tiny() -> tuple[Qwen2ForCausalLM, Qwen2Config]:
    cfg = Qwen2Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=H_KV,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=1024,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).eval(), cfg  # type: ignore[no-untyped-call]


def _deq(cache: Any) -> tuple[torch.Tensor, torch.Tensor]:
    layer: Any = cache.layers[0]
    return layer._dequantize(layer._quantized_keys), layer._dequantize(layer._quantized_values)


def _faithful_arm(model: Any, t: int) -> dict[str, Any]:
    cfg = load_arm("kivi2_faithful")
    cfg.quant["backend"] = "hqq"  # the YAML's quanto JIT-compiles on CPU; hqq does not
    return build_arm(cfg, model, t)


def _want(t: int) -> acc.Footprint:
    """What the faithful arm must be billed at T: 2-bit, G=32, the ACTUAL fp16 residual
    (T mod 128), hqq's fp32 scale+zero pair on an fp32 model."""
    return acc.quant_footprint(
        t, H_KV * D, nbits=2, group=32, residual_length=t % 128, scale_words=2
    )


def test_faithful_prefill_never_calls_quantized_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefill attention sees fp16 only (the QuantizedCache's update never runs), the
    trailing T mod R tokens stay fp16 in the residual, and decode continues from the
    post-hoc state: the length advances and the residual grows -- a re-initialised layer
    would have started over with an empty residual."""
    model, cfg = _tiny()
    cache = make_kivi(cfg, nbits=4, backend="hqq")
    calls = {"n": 0}
    orig = type(cache).update

    def counting(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(type(cache), "update", counting)
    ids = torch.randint(0, 256, (1, 300))
    dyn = DynamicCache()
    with torch.no_grad():
        model(ids, past_key_values=dyn, use_cache=True, logits_to_keep=1)
    quantize_after_prefill(cache, dyn)
    assert calls["n"] == 0
    assert residual_tokens(cache) == 300 % 128
    assert cache.get_seq_length() == 300
    with torch.no_grad():
        model(ids[:, :1], past_key_values=cache, use_cache=True, position_ids=torch.tensor([[300]]))
    assert cache.get_seq_length() == 301 and residual_tokens(cache) == 300 % 128 + 1


def test_posthoc_quantization_equals_the_upstream_update_on_the_same_slab() -> None:
    """ONE quantize call per layer over the slab -- bit-identical to what the layer's own
    update produces from the same slab (residual 0: everything quantized, both ways)."""
    _, cfg = _tiny()
    torch.manual_seed(0)
    k, v = torch.randn(1, 2, 256, 32), torch.randn(1, 2, 256, 32)
    upstream = make_quant_cache(cfg, nbits=4, scheme="kivi", backend="hqq", group=32, residual=0)
    upstream.update(k, v, 0)
    dyn = DynamicCache()
    dyn.update(k, v, 0)
    post = make_kivi(cfg, nbits=4, backend="hqq", residual=0)
    quantize_after_prefill(post, dyn)
    for theirs, ours in zip(_deq(upstream), _deq(post), strict=True):
        assert torch.equal(theirs, ours)
    assert residual_tokens(post) == 0


def test_per_channel_keys_survive_an_outlier_channel() -> None:
    """Per-channel keys give the outlier channel its own scale; per-token keys let it set
    the scale of every other channel in its token (the Week-18 0.00 retrieval)."""
    _, cfg = _tiny()
    torch.manual_seed(1)
    k = torch.randn(1, 2, 256, 32)
    k[..., 7] *= 200.0  # one massive channel (Qwen-style key bias)
    v = torch.randn(1, 2, 256, 32)
    dyn = DynamicCache()
    dyn.update(k, v, 0)
    faithful = make_kivi(cfg, nbits=2, backend="hqq", residual=0)
    quantize_after_prefill(faithful, dyn)
    token = make_quant_cache(cfg, nbits=2, scheme="token", backend="hqq", group=32, residual=0)
    token.update(k, v, 0)
    keep = [c for c in range(32) if c != 7]

    def err(cache: Any) -> float:
        return float(torch.linalg.norm((_deq(cache)[0] - k)[..., keep]))

    assert err(faithful) < 0.5 * err(token)


def test_bytes_include_scales_zeros_and_actual_residual() -> None:
    _, cfg = _tiny()
    dyn = DynamicCache()
    dyn.update(torch.randn(1, 2, 300, 32), torch.randn(1, 2, 300, 32), 0)
    c = make_kivi(cfg, nbits=2, backend="hqq")
    quantize_after_prefill(c, dyn)
    fp = acc.quant_footprint(
        t=300, n=64, nbits=2, group=32, residual_length=residual_tokens(c), scale_words=aux_words(c)
    )
    assert fp.aux_words > 0 and fp.verbatim_elems == 2 * 64 * (300 % 128)


# ------------------------------------------------------------ config + harness


@pytest.mark.parametrize("name, nbits", [("kivi2_faithful", 2), ("kivi4_faithful", 4)])
def test_the_faithful_arm_resolves_from_its_yaml(name: str, nbits: int) -> None:
    cfg = load_arm(name)
    assert cfg.chunkable is False and "chunk" not in cfg.quant  # single-shot by construction
    arm = build_arm(cfg, model=None, t=16384)
    assert (arm["name"], arm["kind"], arm["chunkable"]) == (name, "quant_faithful", False)
    assert (arm["nbits"], arm["quant_group"], arm["quant_residual"]) == (nbits, 32, 128)
    assert (arm["quant_scheme"], arm["quant_backend"]) == ("kivi", "quanto")
    assert callable(arm["make"])


def test_the_faithful_arm_refuses_a_yaml_that_is_not_the_kivi_scheme() -> None:
    """The factory pins the kivi scheme; a YAML saying otherwise would be mislabelled."""
    cfg = load_arm("kivi2_faithful")
    cfg.quant["scheme"] = "token"
    with pytest.raises(ValueError, match="kivi"):
        build_arm(cfg, model=None, t=16384)


def test_retrieve_runs_the_faithful_arm_and_bills_the_actual_residual() -> None:
    """The retrieval axis: fp16 single-shot prefill, post-hoc quantization, block decode;
    the footprint bills the T mod R residual the prefill actually left, not the
    configured 128 the streaming arm is billed."""
    model, _ = _tiny()
    ids = torch.randint(0, 256, (1, 208))
    hit, ratio, _frac, sbits = ruler.retrieve(
        model, TOK, _faithful_arm(model, 200), ids[:, :200], ids[:, 200:], ["1"], "cpu", 0, 32, 2, 4
    )
    assert isinstance(hit, bool) and ratio == sbits == _want(200).ratio_fp16(200, 32)


def test_run_ppl_scores_the_faithful_arm_over_the_quantized_cache() -> None:
    """The perplexity axis takes the kind (no fall-through into the press branch): the row
    is ok and billed like the retrieval axis; ``chunkable: false`` ignores the sweep's chunk."""
    model, _ = _tiny()
    t, window = 160, 16
    ids = torch.randint(0, 256, ((t + window) * 2,))
    (row,) = frontier.run_ppl(
        [_faithful_arm(model, t)], model, frontier.windows(ids, t, window, 1), t,
        chunk=64, n=32, h_kv=2, device="cpu",
    )  # fmt: skip
    assert row["status"] == "ok", row
    assert row["kind"] == "quant_faithful" and row["ratio_fp16"] == _want(t).ratio_fp16(t, 32)


def test_longbench_generates_with_the_faithful_arm() -> None:
    model, _ = _tiny()
    ids = torch.randint(0, 256, (1, 201))  # 200 prefilled + the last token generated from
    text, ratio, sbits = longbench.generate(
        model, TOK, _faithful_arm(model, 200), ids, "cpu", 0, 32, 2, 3
    )
    assert isinstance(text, str) and ratio == sbits == _want(200).ratio_fp16(200, 32)
