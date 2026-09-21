"""Every v1 arm built from YAML must produce exactly what the legacy CLI path produced.

``configs/arms/*.yaml`` + ``frontier.build_arm`` replace the flag soup of the argparse
builder that produced every archived number. That builder was the oracle while both
existed; L0.8a froze its output to ``tests/golden/legacy_arm_kwargs.json`` and L0.8c
deleted it, so the golden is the oracle now and parity is checked two ways: a cache
arm's YAML must resolve to the exact keyword set the legacy ``make`` lambda passed to
``BugStreamingCache`` / ``ShadowKVCache``, key for key and at both context lengths; a
press/quant arm must carry every identifying field the legacy arm dict did, and no
others. ``make`` is never called -- the kwargs are the contract, and comparing them
needs no model.

L1.3b renamed the cache's tracker string from ``bug`` to ``isvd``; the eight ``tracker``
values were rewritten by hand, and the Oja arm gained the two schedule knobs its config now
names (``oja_eta0``/``oja_decay``) because that arm deliberately is no longer the legacy arm
-- the Week-20 cell it froze ran the library defaults and is void. L2.1 (the ``svd_oracle``
rename) hand-rewrote the ``svd_oracle_r0.5`` arm's golden ``params`` keys to
``oracle_group``/``oracle_rank_ratio`` (matching frontier's ``_press``/``_footprint``); the
entry's top-level key is untouched -- the archived legacy arm string. Nothing else in the
golden changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kvdlra.eval.config import ROOT, arm_kwargs, config_hash, load_arm, load_pod, load_task
from kvdlra.eval.frontier import build_arm

GOLDEN = json.loads((Path(__file__).parent / "golden" / "legacy_arm_kwargs.json").read_text())

# Structure, not parameters: the name, the dispatch kind, the factories, the captured
# kwargs, and the analytic-footprint tag. Everything else in an arm dict is a parameter
# the legacy dict also carried, and the golden holds it.
IGNORED = {"name", "kind", "rank", "rank_s", "chunkable", "kwargs", "press_type",
           "make", "make_press", "make_cache", "retention", "hh_select", "hh_budget"}  # fmt: skip

CACHE = sorted(k for k, v in GOLDEN.items() if "kwargs" in v)
PARAMS = sorted(k for k, v in GOLDEN.items() if "params" in v)

# Arm stems the golden does not hold, and why each one is absent rather than missed.
# `kivi2_singleshot` is v1's single-shot control: the legacy quant branch always set
# chunkable=True and the pod forced single-shot with `--chunk 0`, so there is no legacy
# dict to freeze. The Table-4 cells are the L1.6 guard/floor arms
# (`prereg/hygiene_table4.md`) -- post-v1 arms, no `legacy_name`, nothing to be parity
# with. `oja_r64_h256_seed_tuned` (L1.4b) is the Week-2 Oja arm's schedule re-tuned on
# the 1B stored-representation study (`results/recon_1b/`); it is a new arm, not a
# rebuild of the legacy one. The two `kivi*_faithful` arms are L2.2's KIVI at its published
# operating point (G=32, R=128, fp16 single-shot prefill) -- a different arm from the
# streaming mixin the tables used. A plain set on purpose: the next lane's arm is one line
# here.
POST_V1 = {
    "kivi2_singleshot",
    "kivi2_faithful",
    "kivi4_faithful",
    "isvd_r128_noguard",
    "isvd_r128_tol",
    "isvd_r128_qr64",
    "isvd_r128_f0.01_tol",
    "isvd_r128_f0.01_qr64",
    "isvd_r256_noguard",
    "isvd_r256_tol",
    "isvd_r256_qr64",
    "isvd_r256_f0.01_tol",
    "isvd_r256_f0.01_qr64",
    "oja_r64_h256_seed_tuned",
    # L2.4: the matched-budget eviction/structured baselines at k in {0.10, 0.15, 0.25}
    # (prereg to come); `ea_k0.10` is the archived `ea_k0.1` operating point under the
    # new naming, keyed by its own name. `think_c0.5_snapkv_k0.15` is ThinK composed as
    # its paper intends.
    "snapkv_k0.10",
    "snapkv_k0.15",
    "snapkv_k0.25",
    "pyramidkv_k0.10",
    "pyramidkv_k0.15",
    "pyramidkv_k0.25",
    "ea_k0.10",
    "ea_k0.15",
    "think_c0.5_snapkv_k0.15",
    # L3.1: the Gate-1 tracker controls at the r64 operating point (learn-then-freeze,
    # fixed random basis) and the two byte-matched no-gist arms, one per layer width
    # (`tests/test_gate1_arms.py`). New arms, no legacy counterpart.
    "frozen_r64_h256_seed",
    "random_r64_h256_seed",
    "nogist_h2423",
    "nogist_h4460",
    # L5.1: the r64 arm with the gist stored in bf16 (`tests/test_bf16_gist.py`), read by
    # `prereg/bf16_gist.md`. New arm, no legacy counterpart.
    "isvd_r64_h256_seed_bf16",
    # L4.4: the r64 arm on the factored-attention kernel (`decode_attention: kernel`),
    # arm 3 of prereg/kernel_smoke.md. New arm, no legacy counterpart.
    "isvd_r64_h256_seed_kernel",
}


@pytest.mark.parametrize("t", [16384, 32768], ids=["16k", "32k"])
@pytest.mark.parametrize("legacy", CACHE)
def test_yaml_cache_arm_matches_the_frozen_legacy_kwargs(legacy: str, t: int) -> None:
    """The YAML resolves to the legacy lambda's keyword set, key for key, at 16K and 32K.

    Both lengths matter: a budget pinned to the 16K value (e.g. hand-typed instead of left
    ``null`` for ``arm_kwargs`` to resolve) would still pass at one length and only show up
    once the context actually moves the derived ``coord_budget``/``quant_budget``.
    """
    want = GOLDEN[legacy]
    cfg = load_arm(want["config"])
    arm = build_arm(cfg, model=None, t=t)
    assert arm["kwargs"] == want["kwargs"][str(t)]
    assert arm["kwargs"] == arm_kwargs(cfg, t)  # build_arm resolves, it does not re-derive
    assert arm["name"] == legacy and arm["kind"] == want["kind"]
    assert arm["chunkable"] == want["chunkable"]


@pytest.mark.parametrize("legacy", PARAMS)
def test_yaml_press_arm_matches_the_frozen_legacy_params(legacy: str) -> None:
    """The press/quant/composite YAMLs carry every identifying field of the legacy dict,
    and no field the legacy dict did not have."""
    want = GOLDEN[legacy]
    arm = build_arm(load_arm(want["config"]), model=None, t=16384)
    assert arm["name"] == legacy and arm["kind"] == want["kind"]
    assert arm["chunkable"] == want["chunkable"], f"{legacy}: chunkable drifted from legacy"
    assert {k: v for k, v in arm.items() if k not in IGNORED} == want["params"]


def test_the_golden_covers_every_arm_config_with_a_legacy_name() -> None:
    """One exception, and it is a real one: the ss2 control has no legacy counterpart --
    the legacy quant branch always set chunkable=True and the pod forced single-shot with
    --chunk 0, so there is no legacy dict to freeze. It is pinned against its streaming
    twin instead (below)."""
    named = {load_arm(p.stem).legacy_name for p in (ROOT / "arms").glob("*.yaml")} - {None}
    assert named - set(GOLDEN) == {"quant-2bit-kivi#chunk0"}
    assert {v["config"] for v in GOLDEN.values()} | POST_V1 == {
        p.stem for p in (ROOT / "arms").glob("*.yaml")
    }


def test_build_arm_copies_the_kwargs_into_the_factory() -> None:
    """The stored ``kwargs`` and the factory's must not be the same dict: a caller that
    inspects and edits one would silently change what the next ``make()`` constructs."""
    arm = build_arm(load_arm("isvd_r64"), model=None, t=1024)
    arm["kwargs"]["rank"] = 999
    assert arm["make"].__defaults__[0]["rank"] == 64  # the factory's copy is untouched


def test_build_arm_rejects_an_unknown_kind() -> None:
    cfg = load_arm("full")
    cfg.kind = "telepathy"
    with pytest.raises(ValueError, match="unknown arm kind"):
        build_arm(cfg, model=None, t=1024)


def test_every_arm_yaml_loads_and_names_match() -> None:
    for p in (ROOT / "arms").glob("*.yaml"):
        assert load_arm(p.stem).name == p.stem


def test_every_task_and_pod_yaml_loads_and_resolves() -> None:
    """A pod may only name arms and tasks that exist -- the config set is closed."""
    for p in (ROOT / "tasks").glob("*.yaml"):
        assert load_task(p.stem).name == p.stem
    for p in (ROOT / "pods").glob("*.yaml"):
        pod = load_pod(p.stem)
        assert pod.name == p.stem and pod.arms and pod.tasks
        for a in pod.arms:
            load_arm(a)
        for t in pod.tasks:
            load_task(t)


def test_kivi2_singleshot_differs_from_the_streaming_arm_only_in_chunk() -> None:
    """The ss2 control is the same 2-bit cache under single-shot prefill (Week-19)."""
    ss, st = load_arm("kivi2_singleshot"), load_arm("kivi2_streaming")
    assert (ss.quant["chunk"], st.quant["chunk"]) == (0, 4096)
    assert {k: v for k, v in ss.quant.items() if k != "chunk"} == {
        k: v for k, v in st.quant.items() if k != "chunk"
    }


def test_config_hash_is_stable_and_sensitive() -> None:
    pod = load_pod("w18_g1")
    h1 = config_hash(pod)
    assert h1 == config_hash(load_pod("w18_g1"))
    pod.gpu_budget_h = pod.gpu_budget_h + 1
    assert config_hash(pod) != h1


@pytest.mark.parametrize("loader", ["load_arm", "load_task"])
def test_config_hash_covers_the_referenced_configs(
    loader: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing an arm or a task file must change every pod hash that names it."""
    from kvdlra.eval import config as mod

    pod = load_pod("w18_g1")
    h1 = config_hash(pod)
    real = getattr(mod, loader)

    def perturbed(name: str) -> Any:
        cfg = real(name)
        cfg.doc = f"{cfg.doc} (perturbed)"
        return cfg

    monkeypatch.setattr(mod, loader, perturbed)
    assert config_hash(pod) != h1


def test_load_arm_rejects_an_unknown_name() -> None:
    with pytest.raises(FileNotFoundError):
        load_arm("no_such_arm")


def test_a_task_config_that_can_produce_no_records_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``n_trials x len(seeds)`` is the count `pod.py check` demands of every cell and
    ``n_samples`` the count it demands of every sweep, so a zero in either configures a
    task that runs nothing AND a gate that asks for nothing. Refused at load, naming the
    file -- which is the only place the operator can still fix it."""
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "broken.yaml").write_text(
        "name: broken\ngenerator: ppl\nctx: 0\nn_trials: 0\nseeds: []\nn_samples: 0\n"
    )
    monkeypatch.setattr("kvdlra.eval.config.ROOT", tmp_path)
    with pytest.raises(ValueError, match=r"broken\.yaml") as e:
        load_task("broken")
    assert all(k in str(e.value) for k in ("n_trials", "seeds", "ctx", "n_samples"))
