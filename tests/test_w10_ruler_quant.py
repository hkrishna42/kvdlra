"""Week-18 W1: the KIVI-style quantized-KV baseline arm (the panel's #1 blocking gap).

`quant-2bit`/`quant-4bit` wrap transformers' QuantizedCache (quanto backend). The arm
supplies its OWN cache object (a QuantizedCache is NOT a DynamicCache subclass), so it
needs a dedicated branch in build_arm, _footprint (else it trips the DynamicCache
assert), and retrieve. These hermetic tiny-Llama tests exercise all three on CPU.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.baselines.compat import install_kvpress_prefill_compat
from kvdlra.eval.config import ArmCfg
from kvdlra.eval.frontier import _footprint, build_arm
from kvdlra.eval.ruler import retrieve

H, D = 2, 16  # KV heads x head_dim -> n_features 32


class _StubTok:
    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


def _model() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=H,
        head_dim=D,
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg)  # type: ignore[no-untyped-call]
    m.config._attn_implementation = "sdpa"
    m.eval()  # type: ignore[no-untyped-call]
    return m


def _quant_arms(model: Any, t: int = 200) -> list[dict[str, Any]]:
    """The 2- and 4-bit arms at the transformers-default (per-token) axes."""
    return [
        build_arm(
            ArmCfg(
                name=f"quant-{b}bit",
                kind="quant",
                quant={
                    "nbits": b,
                    "scheme": "token",
                    "backend": "quanto",
                    "group": 64,
                    "residual": 128,
                },
            ),
            model,
            t,
        )
        for b in (2, 4)
    ]


def test_build_arm_creates_quant_arms() -> None:
    arms = _quant_arms(_model())
    names = [a["name"] for a in arms]
    assert names == ["quant-2bit", "quant-4bit"]
    assert all(a["kind"] == "quant" and a["chunkable"] is True for a in arms)  # Week-19: chunked


def test_quant_arm_runs_through_retrieve_and_footprint() -> None:
    """The arm supplies a QuantizedCache; retrieve() must run it (not trip the
    DynamicCache assert in _footprint) and return a 4-tuple with sbits == fp16."""
    install_kvpress_prefill_compat()
    model = _model()
    n = H * D
    hay = torch.randint(0, 256, (1, 200))
    query = torch.randint(0, 256, (1, 8))
    for arm in _quant_arms(model):
        hit, ratio, _frac, sratio = retrieve(
            model, _StubTok(), arm, hay, query, ["1"], "cpu", 0, n, H, 4
        )
        assert isinstance(hit, bool)
        assert 0.0 < ratio <= 1.0
        # quant has no fp32-at-rest state -> honest ratio equals the fp16 ratio.
        assert ratio == sratio


def test_quant_footprint_dispatch_matches_accounting() -> None:
    """_footprint routes kind='quant' to acc.quant_footprint with the arm's nbits/
    group/residual (proving the branch precedes the DynamicCache assert)."""
    import kvdlra.accounting as acc

    arm: dict[str, Any] = {
        "kind": "quant",
        "nbits": 2,
        "quant_group": 64,
        "quant_residual": 128,
        "name": "quant-2bit",
    }
    # Week-19: the aux (scale+zero) precision is read off the real cache after prefill.
    cache = _quant_arms(_model())[0]["make"]()
    cast(Any, cache.layers[0]).update(torch.randn(1, H, 200, D), torch.randn(1, H, 200, D))
    fp = _footprint(arm, cast(Any, cache), 16384, 1024, H)
    assert fp.ratio_fp16(16384, 1024) == acc.quant_footprint(
        16384, 1024, nbits=2, group=64, residual_length=128, scale_words=2
    ).ratio_fp16(16384, 1024)


# ---------------------------------------------- Week-18 W2: BUG x quant compose


def _compose_arm(model: Any, name: str, *, seed: bool, t: int = 160) -> dict[str, Any]:
    """The composed arm: 32 fp32 coordinate columns kept, everything demoted from them
    coded at 4 bits (quant_budget=null resolves to the whole context, so nothing is
    dropped) -- the configs/arms/isvd_r64_h256_seed_q4 shape at tiny scale."""
    return build_arm(
        ArmCfg(
            name=name,
            kind="bug",
            cache={
                "rank": 16,
                "coord_budget": 32,
                "recent_window": 16,
                "absorb_block": 8,
                "n_sink": 4,
                "retention": "lowrank_surprise",
                "hh_budget": 16,
                "hh_select": "surprise",
                "hh_neighbor": 0,
                "seed_hh_warmup": seed,
                "quant_bits": 4,
                "quant_budget": None,
            },
        ),
        model,
        t,
    )


def test_bug_quant_compose_arm_builds() -> None:
    """A coordinate tier composed with 4-bit coding is a cache arm like any other."""
    arm = _compose_arm(_model(), "bugS-r16-h16-q4", seed=False)
    assert arm["name"] == "bugS-r16-h16-q4" and arm["kind"] == "bug"
    assert arm["kwargs"]["quant_bits"] == 4 and arm["kwargs"]["coord_budget"] == 32


def test_seed_plus_quant_builds_and_seeds_the_exact_tier() -> None:
    """Week-19: the seeded compose arm (the sub-cliff candidate). The
    warm-up seed only routes the first chunk's sub-blocks through _absorb_block_slash --
    the same graduation path the unseeded q4 arm runs with its quant tier every step --
    so the combination is wired: it builds, its first-chunk ingest populates the exact
    tier (the seed effect), and the demoted columns reach the quant tier (billed)."""
    from kvdlra.eval.frontier import _prefill_chunked

    model = _model()
    arm = _compose_arm(model, "bugSseed-r16-h16-q4", seed=True)
    assert arm["name"] == "bugSseed-r16-h16-q4"
    cache = arm["make"]()
    hay = torch.randint(0, 256, (1, 160))
    with cache.attach(model):
        _prefill_chunked(model, cache, hay, 40)
    layer = cache._bug_layers()[0]
    assert layer._hh_len() > 0  # seeded during the first chunk, not only at graduation
    assert layer._q_len() > 0  # demoted coordinates landed in the 4-bit tier
    fp = _footprint(arm, cache, 160, H * D, H)
    assert 0.0 < fp.ratio_stored_bits(160, H * D) < 1.0
