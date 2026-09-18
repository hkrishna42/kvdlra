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
-- the Week-20 cell it froze ran the library defaults and is void. Nothing else in the
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
    assert {v["config"] for v in GOLDEN.values()} | {"kivi2_singleshot"} == {
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


def test_faithful_quant_is_reserved_not_silently_wrong() -> None:
    """A faithful KIVI (G=32, R=128, full-precision prefill) is a different arm from the
    QuantizedCache mixin, and L2 owns it. Until then it must refuse, not approximate."""
    cfg = load_arm("kivi2_streaming")
    cfg.kind = "quant_faithful"
    with pytest.raises(NotImplementedError, match="L2"):
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
