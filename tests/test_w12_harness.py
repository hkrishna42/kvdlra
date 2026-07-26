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


def test_qwhiten_builds_bugsq_arms(tmp_path: object) -> None:
    import pathlib

    model = _tiny_model()
    n_feat = 2 * 16  # num_kv_heads * head_dim
    wfile = pathlib.Path(str(tmp_path)) / "wkey.pt"
    torch.save({"w_key": torch.rand(2, n_feat) + 0.5}, wfile)  # (n_layers, n_features)
    arms = w10_frontier.build_arms(_args(qwhiten_file=str(wfile)), model, t=64)
    assert [a["name"] for a in arms] == ["bugSQ-r8-h4"]
    layer = _first_layer(arms[0]["make"]())
    assert layer.w_key is not None and tuple(layer.w_key.shape) == (n_feat,)
    assert layer.hh_retain is True and layer.hh_select == "surprise"


def test_warmup_seed_builds_bugsseed_arms() -> None:
    # Week-13 T-B: --warmup-seed flips seed_hh_warmup and renames the arm family so
    # bugS vs bugSseed is a clean A/B (and stays separable from bugS-* under an
    # anchored/hyphenated grep, like bugSdrop).
    model = _tiny_model()
    arms = w10_frontier.build_arms(_args(warmup_seed=True), model, t=64)
    assert [a["name"] for a in arms] == ["bugSseed-r8-h4"]
    layer = _first_layer(arms[0]["make"]())
    assert layer.seed_hh_warmup is True
    assert layer.hh_retain is True and layer.hh_select == "surprise"
    assert "bugS-" not in "bugSseed-r8-h4"


def test_qwhiten_not_combined_with_discard(tmp_path: object) -> None:
    import pathlib

    model = _tiny_model()
    wfile = pathlib.Path(str(tmp_path)) / "wkey.pt"
    torch.save({"w_key": torch.rand(2, 32) + 0.5}, wfile)
    with pytest.raises(ValueError, match="not combined with --hh-discard"):
        w10_frontier.build_arms(_args(qwhiten_file=str(wfile), hh_discard=True), model, t=64)


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
