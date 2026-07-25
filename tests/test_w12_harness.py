"""Week-12 harness: the ``--hh-discard`` flag builds ``bugSdrop-*`` arms.

Pins for the T1 ablation plumbing: the flag flips ``hh_retain`` on the built
caches, the arm-name family cannot be cross-grepped with the retain family
(``bugS-r...`` is not a substring of ``bugSdrop-r...`` and vice versa -- mixed
logs stay separable), and the chunked-ingest guard still protects the discard
arms (single-shot prefill would bypass the SLASH pool entirely).
"""

from __future__ import annotations

import argparse

import pytest
import torch
import w10_frontier
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer


def _tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    model = LlamaForCausalLM(  # type: ignore[no-untyped-call]
        LlamaConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=4096,
        )
    )
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _args(**over: object) -> argparse.Namespace:
    ns = argparse.Namespace(
        methods=["bugslash"],
        ranks=[8],
        hh_budgets=[4],
        hh_neighbor=1,
        chunk=16,
        recent_window=8,
        absorb_block=4,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _first_layer(cache: BugStreamingCache) -> BugStreamingLayer:
    return next(layer for layer in cache.layers if isinstance(layer, BugStreamingLayer))


def test_hh_discard_builds_bugsdrop_arms() -> None:
    model = _tiny_model()
    arms = w10_frontier.build_arms(_args(hh_discard=True), model, t=64)
    assert [a["name"] for a in arms] == ["bugSdrop-r8-h4"]
    layer = _first_layer(arms[0]["make"]())
    assert layer.hh_retain is False
    assert layer.hh_budget == 4 and layer.hh_select == "surprise"


def test_default_stays_bugs_with_retain() -> None:
    model = _tiny_model()
    # No hh_discard attribute at all (older callers): getattr-default keeps bugS.
    arms = w10_frontier.build_arms(_args(), model, t=64)
    assert [a["name"] for a in arms] == ["bugS-r8-h4"]
    assert _first_layer(arms[0]["make"]()).hh_retain is True


def test_arm_families_are_not_cross_greppable() -> None:
    retain, discard = "bugS-r128-h1024", "bugSdrop-r128-h1024"
    assert retain not in discard
    assert discard not in retain
    assert "bugS-" not in discard  # the family-prefix grep stays clean too


def test_discard_still_requires_chunk() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="chunked prefill"):
        w10_frontier.build_arms(_args(hh_discard=True, chunk=0), model, t=64)
