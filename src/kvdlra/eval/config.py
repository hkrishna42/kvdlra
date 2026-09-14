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
    generator: str  # inhouse | official_ruler | longbench | ppl | latency
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
    # `latency` only. That axis is one measurement per (arm, context, batch) rather than
    # a set of trials, and it sweeps context lengths WITHIN one task -- each one rebuilds
    # the arms, because a cache arm's budgets resolve against the context length. `ctx`
    # stays the scalar every other generator reads; `ctxs` is the sweep when it is set.
    ctxs: list[int] | None = None
    batch_sizes: list[int] = field(default_factory=lambda: [1])
    n_steps: int = 64  # timed decode forwards per point
    warmup: int = 8  # of which the first this many are discarded before the p50


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
    # A missing file is OmegaConf.load's own FileNotFoundError, naming the same path.
    p = ROOT / kind / f"{name}.yaml"
    cfg = OmegaConf.merge(OmegaConf.structured(schema), OmegaConf.load(p))
    return OmegaConf.to_object(cfg)


def load_arm(name: str) -> ArmCfg:
    return _load("arms", name, ArmCfg)  # type: ignore[no-any-return]


def load_task(name: str) -> TaskCfg:
    """The task config, refusing a grid that cannot produce the records `check` counts.

    A cell is ``n_trials x len(seeds)`` records and a perplexity sweep is ``n_samples``
    windows, so a zero in either is a task that runs nothing AND a `scripts/pod.py check`
    rule that asks for nothing -- the one shape in which an empty pod passes its own gate.
    A non-positive context length is the same failure one step earlier: there is no prompt
    to build. Refused at load time, where the file can still be named.
    """
    t: TaskCfg = _load("tasks", name, TaskCfg)
    bad = []
    if t.n_trials < 1:
        bad.append(f"n_trials={t.n_trials} must be >= 1")
    if not t.seeds:
        bad.append("seeds is empty")
    if min([t.ctx, *(t.ctxs or [])]) <= 0:
        bad.append(f"ctx must be positive (ctx={t.ctx}, ctxs={t.ctxs})")
    if t.generator == "ppl" and t.n_samples < 1:
        bad.append(f"n_samples={t.n_samples} must be >= 1 for a ppl task")
    if bad:
        raise ValueError(f"{ROOT / 'tasks' / f'{name}.yaml'}: " + "; ".join(bad))
    return t


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
