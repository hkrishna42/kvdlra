"""The exact-tier config fields reach the cache, and the arm families stay separable.

The Week-12 T1 ablation (``hh_retain=False``) and the Week-13 warm-up seed were CLI
flags that renamed the arm family; they are now fields of ``configs/arms/*.yaml`` and
the family name is the config's own ``legacy_name``. What is still worth pinning is
that ``build_arm`` threads those fields into the constructed layer, and that the two
family names cannot be cross-grepped in a mixed log (``bugS-r...`` is not a substring
of ``bugSdrop-r...`` and vice versa).
"""

from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer
from kvdlra.eval.config import ArmCfg
from kvdlra.eval.frontier import build_arm

CACHE = {
    "rank": 8,
    "coord_budget": None,
    "recent_window": 8,
    "absorb_block": 4,
    "n_sink": 4,
    "retention": "lowrank_surprise",
    "hh_budget": 4,
    "hh_select": "surprise",
    "hh_neighbor": 1,
    "hh_retain": True,
    "seed_hh_warmup": False,
}


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


def _layer(name: str, **over: object) -> BugStreamingLayer:
    cfg = ArmCfg(name=name, kind="bug", cache={**CACHE, **over})
    cache = build_arm(cfg, _tiny_model(), t=64)["make"]()
    assert isinstance(cache, BugStreamingCache)
    return next(ly for ly in cache.layers if isinstance(ly, BugStreamingLayer))


def test_hh_retain_false_builds_a_select_and_discard_tier() -> None:
    layer = _layer("bugSdrop-r8-h4", hh_retain=False)
    assert layer.hh_retain is False
    assert layer.hh_budget == 4 and layer.hh_select == "surprise"


def test_seed_hh_warmup_reaches_the_layer() -> None:
    layer = _layer("bugSseed-r8-h4", seed_hh_warmup=True)
    assert layer.seed_hh_warmup is True
    assert layer.hh_retain is True and layer.hh_select == "surprise"


def test_the_default_retains_the_tier() -> None:
    assert _layer("bugS-r8-h4").hh_retain is True


def test_arm_families_are_not_cross_greppable() -> None:
    retain, discard = "bugS-r128-h1024", "bugSdrop-r128-h1024"
    assert retain not in discard
    assert discard not in retain
    assert "bugS-" not in discard  # the family-prefix grep stays clean too


def test_the_arm_name_is_the_configs_legacy_name_when_it_has_one() -> None:
    """Records and log rows key on this string, so it is the config's claim about which
    archived rows a re-run is comparable with -- not a name the builder invented."""
    plain = ArmCfg(name="isvd_r8", kind="bug", cache=CACHE)
    assert build_arm(plain, model=None, t=64)["name"] == "isvd_r8"
    plain.legacy_name = "bugS-r8-h4"
    assert build_arm(plain, model=None, t=64)["name"] == "bugS-r8-h4"
