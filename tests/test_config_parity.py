"""Every v1 arm built from YAML must produce exactly what the legacy CLI path produced.

``configs/arms/*.yaml`` replaces the flag soup of ``scripts/w10_frontier.py`` (Task 8
deletes the builder). Until then that builder is the oracle, and parity is checked two
ways: a cache arm's YAML must reproduce the exact keyword set its ``make`` lambda
passes to ``BugStreamingCache`` / ``ShadowKVCache``, key for key; a press/quant arm's
YAML must reproduce every identifying field of the legacy arm dict. ``make`` is never
called -- the kwargs are the contract, and a real model is not needed to compare them.
"""

from __future__ import annotations

from typing import Any

import pytest

from kvdlra.eval.config import ROOT, arm_kwargs, config_hash, load_arm, load_pod, load_task

T = 16384  # the context the v1 16K cells ran at; the budgets resolve against it

RH = "--ranks 64 --hh-budgets 256 --hh-neighbor 1 --warmup-seed"  # scripts/pod/w18.sh
QC = f"{RH} --bug-quant-bits 4 --bug-quant-budget 512"  # scripts/pod/w19.sh (a1q)

CACHE = {  # legacy arm name -> the CLI flags that produced it (scripts/pod/w18.sh, w19.sh)
    "bug-r64": "--methods bug --ranks 64",
    "bug-r128": "--methods bug --ranks 128",
    "bug-r256": "--methods bug --ranks 256",
    "bug-r256-f0.01": "--methods bug --ranks 256 --min-sv-frac 0.01",
    "bugSseed-r64-h256": f"--methods bugslash {RH}",
    "bugSseed-r64-h256-q4": f"--methods bugslash {QC}",
    "bugSseed-r64-h256-oja": f"--methods bugslash {RH} --tracker oja",
    "bugSseed-r64-h256-fd": f"--methods bugslash {RH} --tracker fd",
    "bugSseed-r128-h1024-s32": "--methods bugslash --ranks 128 --hh-budgets 1024 "
    "--hh-neighbor 1 --warmup-seed --score-rank 32",
    "bugSseed-r256-h1024": "--methods bugslash --ranks 256 --hh-budgets 1024 "
    "--hh-neighbor 1 --warmup-seed",
    "bugEVICT-h256": "--methods bugevict --hh-budgets 256",
    "shadow-r64": "--methods shadow --shadow-ranks 64",
    "shadow-r128": "--methods shadow --shadow-ranks 128",
}

KIVI = {  # the quant arms' identifying fields, shared by the quant and composite arms
    "nbits": "nbits",
    "quant_scheme": "scheme",
    "quant_backend": "backend",
    "quant_group": "group",
    "quant_residual": "residual",
}
PARAMS: dict[str, tuple[str, dict[str, str]]] = {  # legacy arm -> (flags, legacy field -> YAML key)
    "full": ("--methods full", {}),
    "quant-2bit-kivi": ("--methods quant --quant-scheme kivi --quant-nbits 2", KIVI),
    "quant-4bit-kivi": ("--methods quant --quant-scheme kivi --quant-nbits 4", KIVI),
    "quant-8bit-kivi-hqq": (
        "--methods quant --quant-scheme kivi --quant-backend hqq --quant-nbits 8",
        KIVI,
    ),
    "think-c0.5": ("--methods think --think-ratios 0.5", {"think_ratio": "ratio"}),
    "palu-r0.5": (
        "--methods palu --palu-ranks 0.5",
        {"palu_rank_ratio": "rank", "palu_group": "group"},
    ),
    "ea-k0.1": ("--methods ea --evict-keeps 0.1", {"keep": "keep"}),
    "ea-k0.25": ("--methods ea --evict-keeps 0.25", {"keep": "keep"}),
    "ea-k0.5": ("--methods ea --evict-keeps 0.5", {"keep": "keep"}),
    "snapkv-k0.1": ("--methods snapkv --evict-keeps 0.1", {"keep": "keep"}),
    **{
        f"ea-k{k}-q{b}-kivi": (
            f"--methods composite --evict-keeps {k} --quant-nbits {b} --quant-scheme kivi",
            {"keep": "keep", **KIVI},
        )
        for k in (0.1, 0.25)
        for b in (2, 4)
    },
}

# Legacy arm-dict fields that are structure, not parameters: the name, the dispatch kind,
# the factories, the captured kwargs, and the flags that only steer the harness. `chunkable`
# used to sit in this set too, which let YAML<->legacy drift on the single-shot guard pass
# silently; it is compared explicitly below instead, since it lives at the top of ArmCfg
# rather than inside press:/quant: (so it cannot go through the `fields`/`blocks` mapping).
IGNORED = {"name", "kind", "rank", "press_type", "make", "make_press", "make_cache"}
KIND = {"composite": "press_quant"}  # the legacy dispatch key for the YAML `composite` kind


def _legacy(flags: str, name: str, t: int = T) -> dict[str, Any]:
    """The legacy arm dict named ``name``, built by the CLI path ``flags`` produced."""
    import w10_frontier  # scripts/ is on pythonpath; deleted in Task 8 with this import

    ns = w10_frontier.build_parser().parse_args([*flags.split(), "--chunk", "4096"])
    arms = w10_frontier.build_arms(ns, model=None, t=t)
    return next(a for a in arms if a["name"] == name)


def _yaml_name(legacy: str) -> str:
    """The config whose ``legacy_name`` is ``legacy`` (exactly one, or the test lies)."""
    files = sorted((ROOT / "arms").glob("*.yaml"))
    hits = [p.stem for p in files if load_arm(p.stem).legacy_name == legacy]
    assert len(hits) == 1, f"{legacy}: expected exactly one config, got {hits}"
    return hits[0]


@pytest.mark.parametrize("t", [16384, 32768], ids=["16k", "32k"])
@pytest.mark.parametrize("legacy", sorted(CACHE))
def test_yaml_cache_arm_matches_legacy_build_arms(legacy: str, t: int) -> None:
    """The YAML resolves to the legacy lambda's keyword set, key for key, at 16K and 32K.

    Both lengths matter: a budget pinned to the 16K value (e.g. hand-typed instead of left
    ``null`` for ``arm_kwargs`` to resolve) would still pass at ``t=T`` and only show up once
    the context length actually moves the derived ``coord_budget``/``quant_budget``.
    """
    got = arm_kwargs(load_arm(_yaml_name(legacy)), t=t)
    want = _legacy(CACHE[legacy], legacy, t=t)["kwargs"]
    assert got == want


@pytest.mark.parametrize("legacy", sorted(PARAMS))
def test_yaml_press_arm_matches_legacy_build_arms(legacy: str) -> None:
    """The press/quant/composite YAMLs carry every identifying field of the legacy dict."""
    flags, fields = PARAMS[legacy]
    arm, cfg = _legacy(flags, legacy), load_arm(_yaml_name(legacy))
    assert KIND.get(cfg.kind, cfg.kind) == arm["kind"]
    # The single-shot guard: not routed through `fields`/`blocks` (see IGNORED above), so it
    # needs its own line, with the legacy dict's implicit default (True, ArmCfg's own default)
    # where a branch never sets the key at all (e.g. ea-k*/snapkv-k*). kivi2_singleshot is NOT
    # covered here: it isn't in PARAMS, because its chunkable=false (chunk=0, single-shot) is a
    # deliberate config-level choice with no legacy build_arms counterpart -- the legacy "quant"
    # branch always sets chunkable=True regardless of the pod-level --chunk value. See
    # test_kivi2_singleshot_differs_from_the_streaming_arm_only_in_chunk below.
    assert cfg.chunkable == arm.get("chunkable", True), f"{legacy}: chunkable drifted from legacy"
    assert set(fields) == set(arm) - IGNORED - {"chunkable"}, "a legacy field is unaccounted for"
    blocks = {**cfg.press, **cfg.quant}
    assert set(blocks) - set(fields.values()) <= {"chunk"}, "an unidentified YAML field"
    assert {k: blocks[k] for k in fields.values()} == {v: arm[k] for k, v in fields.items()}


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
