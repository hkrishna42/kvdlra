"""YAML configs (omegaconf) for arms, tasks and pods. Every experiment is a YAML; the
CLI flag soup of w10_ruler/w10_frontier is retired in Task 8.

Three flat directories under ``configs/`` -- no groups, no compose, no plugins. A file
is loaded by merging it onto the structured schema, so an unknown key or a wrong type
fails at load time and ``to_object`` hands back a plain dataclass.

``cache:`` holds the cache constructor's keyword arguments *verbatim*: the YAML is the
call, and ``arm_kwargs`` only resolves the two that depend on the context length. A key
the v1 arm did not pass is simply absent from the file, so the resolved dict is the
legacy keyword set key for key (``tests/test_config_parity.py``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[3] / "configs"


@dataclass
class ArmCfg:
    """One compression method at one operating point."""

    name: str
    kind: str  # bug | full | press | quant | composite | shadow
    legacy_name: str | None = None  # the arm string in results/paper-v1 records
    chunkable: bool = True
    doc: str = ""
    cache: dict[str, Any] = field(default_factory=dict)  # streaming-cache kwargs, verbatim
    press: dict[str, Any] = field(default_factory=dict)  # kvpress / oracle press kwargs
    quant: dict[str, Any] = field(default_factory=dict)  # nbits, scheme, backend, group, ...


@dataclass
class TaskCfg:
    """One evaluation protocol at one context length."""

    name: str
    generator: str  # inhouse | official_ruler | longbench | ppl
    ctx: int
    doc: str = ""
    tasks: list[str] = field(default_factory=list)
    n_trials: int = 6
    seeds: list[int] = field(default_factory=lambda: [0, 1])
    filler: str = "cycle"
    depths: list[float] | None = None
    chunk: int = 4096
    window: int = 512
    n_samples: int = 4


@dataclass
class PodCfg:
    """One GPU run: a model, a set of arms, a set of tasks."""

    name: str
    model: str
    arms: list[str]
    tasks: list[str]
    doc: str = ""
    dtype: str = "bfloat16"
    prereg: str | None = None
    gpu_budget_h: float = 0.0
    image: str = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel"


def _load(kind: str, name: str, schema: type) -> Any:
    p = ROOT / kind / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(p)
    cfg = OmegaConf.merge(OmegaConf.structured(schema), OmegaConf.load(p))
    return OmegaConf.to_object(cfg)


def load_arm(name: str) -> ArmCfg:
    return _load("arms", name, ArmCfg)  # type: ignore[no-any-return]


def load_task(name: str) -> TaskCfg:
    return _load("tasks", name, TaskCfg)  # type: ignore[no-any-return]


def load_pod(name: str) -> PodCfg:
    return _load("pods", name, PodCfg)  # type: ignore[no-any-return]


def arm_kwargs(arm: ArmCfg, t: int) -> dict[str, Any]:
    """The exact cache-constructor kwargs for context length ``t``.

    Only the two tier budgets depend on ``t``: ``null`` means "size this tier to the
    whole prefill" -- ``t + recent_window + absorb_block``, which keeps the entire
    non-exact middle as rank-r coordinates. A literal pins the tier instead (the
    surprise-eviction control pins the gist at 1; the composed arm pins the fp32
    coordinate tier at 512 columns and lets everything demoted from it be coded).
    """
    kw = dict(arm.cache)
    cb = t + int(kw.get("recent_window", 32)) + int(kw.get("absorb_block", 16))
    for k in ("coord_budget", "quant_budget"):
        if k in kw and kw[k] is None:
            kw[k] = cb
    if "tracker" in kw:  # the shipped step's legacy string, until L1 renames it in-cache
        kw["tracker"] = {"isvd": "bug"}.get(kw["tracker"], kw["tracker"])
    return kw


def config_hash(pod: PodCfg) -> str:
    """sha256 of the pod together with every arm and task it names."""
    flat = OmegaConf.to_container(OmegaConf.structured(pod))
    arms = {a: OmegaConf.to_container(OmegaConf.structured(load_arm(a))) for a in pod.arms}
    tasks = {t: OmegaConf.to_container(OmegaConf.structured(load_task(t))) for t in pod.tasks}
    blob: DictConfig = OmegaConf.create({"pod": flat, "arms": arms, "tasks": tasks})
    return hashlib.sha256(OmegaConf.to_yaml(blob, sort_keys=True).encode()).hexdigest()
