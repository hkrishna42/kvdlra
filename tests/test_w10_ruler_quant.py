"""Week-18 W1: the KIVI-style quantized-KV baseline arm (the panel's #1 blocking gap).

`quant-2bit`/`quant-4bit` wrap transformers' QuantizedCache (quanto backend). The arm
supplies its OWN cache object (a QuantizedCache is NOT a DynamicCache subclass), so it
needs a dedicated branch in build_arms, _footprint (else it trips the DynamicCache
assert), and retrieve. These hermetic tiny-Llama tests exercise all three on CPU.
"""

from __future__ import annotations

import argparse
from typing import Any, cast

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from w10_frontier import _footprint, build_arms
from w10_ruler import retrieve

from kvdlra.press.compat import install_kvpress_prefill_compat

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


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        methods=["quant"],
        quant_nbits=[2, 4],
        quant_group=64,
        quant_residual=128,
        ranks=[64],
        hh_budgets=[256],
        chunk=0,
        recent_window=32,
        absorb_block=16,
        morph_keeps=[0.1],
        evict_keeps=[0.1],
        think_ratios=[0.5],
        palu_ranks=[0.5],
        palu_group=1,
        shadow_ranks=[64],
        shadow_topk=256,
        hh_neighbor=0,
        hh_discard=False,
        qwhiten_file=None,
        warmup_seed=False,
        score_rank=None,
        min_sv_frac=0.0,
    )


def test_build_arms_creates_quant_arms() -> None:
    arms = build_arms(_args(), _model(), 200)
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
    for arm in build_arms(_args(), model, 200):
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
    cache = build_arms(_args(), _model(), 200)[0]["make"]()
    cast(Any, cache.layers[0]).update(torch.randn(1, H, 200, D), torch.randn(1, H, 200, D))
    fp = _footprint(arm, cast(Any, cache), 16384, 1024, H)
    assert fp.ratio_fp16(16384, 1024) == acc.quant_footprint(
        16384, 1024, nbits=2, group=64, residual_length=128, scale_words=2
    ).ratio_fp16(16384, 1024)


# ---------------------------------------------- Week-18 W2: BUG x quant compose


def _bug_args(**kw: Any) -> argparse.Namespace:
    d: dict[str, Any] = {
        "methods": ["bugslash"],
        "ranks": [16],
        "hh_budgets": [16],
        "chunk": 40,
        "recent_window": 16,
        "absorb_block": 8,
        "morph_keeps": [0.1],
        "evict_keeps": [0.1],
        "think_ratios": [0.5],
        "palu_ranks": [0.5],
        "palu_group": 1,
        "shadow_ranks": [64],
        "shadow_topk": 256,
        "hh_neighbor": 0,
        "hh_discard": False,
        "qwhiten_file": None,
        "warmup_seed": False,
        "score_rank": None,
        "min_sv_frac": 0.0,
        "bug_quant_bits": None,
        "bug_quant_budget": 0,
    }
    d.update(kw)
    return argparse.Namespace(**d)


def test_bug_quant_compose_arm_builds_with_q_suffix() -> None:
    """--bug-quant-bits produces a bugS-...-q{bits} compose arm (no seed)."""
    arms = build_arms(_bug_args(bug_quant_bits=4, bug_quant_budget=32), _model(), 160)
    assert arms[0]["name"] == "bugS-r16-h16-q4"
    assert arms[0]["kind"] == "bug"


def test_seed_plus_quant_fails_loud() -> None:
    """The seed+quant combo is fenced until GPU-validated: build_arms raises a clear
    error rather than silently dropping the seed or hitting the deep cache guard."""
    import pytest

    args = _bug_args(bug_quant_bits=4, bug_quant_budget=32, warmup_seed=True)
    with pytest.raises(ValueError, match="not yet combined with --warmup-seed"):
        build_arms(args, _model(), 160)


def test_plot_survives_empty_results() -> None:
    """A RULER run where every arm SKIPs (e.g. quant on a -runtime pod with no CUDA
    kernel) yields empty results; _plot must skip cleanly, not crash plt.subplots on a
    0-row grid (the non-fatal Traceback seen on the Week-18 G1 runtime pods)."""
    from pathlib import Path

    import w10_ruler

    w10_ruler._plot({"results": [], "tasks": ["niah_single"], "model": "m"}, Path("/tmp/w18_empty"))
