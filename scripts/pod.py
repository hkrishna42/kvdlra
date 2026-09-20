"""The pod entrypoint: `launch` from the laptop, `run` on the GPU, `harvest` the log,
`check` what came back.

One directory per pod, `results/<pod>/`, holding `manifest.json` (git SHA, config hash,
model, library versions, GPU, wall clock, command line), `env.txt` (the pinned set as
`name==version`), and the records a harvest parsed out of the log: `trials.jsonl`,
`ppl.jsonl` (aggregate perplexity), `pplw.jsonl` (the per-window NLLs behind it -- a
different schema, hence a different file), `latency.jsonl` (measured decode cost, one row
per arm x ctx x batch), `diag.jsonl`. `check` is the gate every citable number passes: it
re-derives the config hash from `configs/pods/<pod>.yaml`, resolves the SHA, enforces the
pre-registration commit order, and requires EVERY cell the config calls for -- arm x
generator x sub-task x ctx -- to hold exactly `n_trials x len(seeds)` records, plus one
perplexity record AND `n_samples` per-window records per (arm, ctx) for every `ppl` task,
and one decode record per (arm, ctx, batch) for every `latency` task. A trial that raised
is recorded with `error` and still counted, so a cell can never silently shrink; a cell
with no records at all is the loudest failure there is, which is what makes a pod that
produced nothing impossible to pass off as a clean run. And because recording rather than
skipping keeps every cell full, the error COUNT is its own rule: any recorded failure
fails the pod (ruling R29), with no tolerance knob.

`run` writes the manifest and the environment, materializes any haystack source a
generator-v2 task names that is not on disk (`prepare` is that step alone) and records
the sources' digests, then hands the pod config to `kvdlra.eval.runner.run_pod`, which
is the eval loop; `--dry-run` stops after the manifest, which is what the tests and
`make check` exercise. Archived paper-v1 manifests carry
`converted_by` and are skipped: they are static evidence of runs that happened before this
entrypoint existed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import shlex
import subprocess
import sys
import time
import tomllib
from collections import Counter
from datetime import UTC, datetime
from importlib import metadata as md
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401

from kvdlra.eval.config import PodCfg, TaskV2Cfg, config_hash, load_arm, load_pod, load_task
from kvdlra.eval.records import (
    TrialRecord,
    parse_diag_lines,
    parse_error_lines,
    parse_latency_lines,
    parse_ppl_lines,
    parse_pplw_lines,
    parse_trial_lines,
    read_jsonl,
    replayed,
    write_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# env.txt: the library set a re-run must reproduce. `cuda` is torch's build, not a
# distribution; `gpu` is not a version at all and is written as `gpu=<name|none>`.
ENV_PKGS = (
    "torch",
    "cuda",
    "triton",
    "transformers",
    "kvpress",
    "optimum-quanto",
    "hqq",
    "omegaconf",
)
PINNED = ("torch", "transformers", "kvpress", "hqq")  # checked against the pyproject pins
TS_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})")


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)


def _head() -> str:
    return _git("rev-parse", "HEAD").stdout.strip()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dist(pkg: str) -> str:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "none"


def versions() -> dict[str, str]:
    """Installed versions of the pinned set, plus torch's CUDA build and the GPU name.

    Missing is `"none"`, never an exception: a CPU laptop writes the same file shape a
    pod does, so `check` compares like with like.
    """
    out = {p: _dist(p) for p in ENV_PKGS if p != "cuda"}
    out["cuda"], out["gpu"] = "none", "none"
    try:
        import torch
    except ImportError:  # a torch-less laptop records `none`; anything else is a bug
        return out
    out["cuda"] = torch.version.cuda or "none"
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


def env_lines() -> list[str]:
    v = versions()
    return [f"{p}=={v[p]}" for p in ENV_PKGS] + [f"gpu={v['gpu']}"]


def manifest(name: str, sha: str, command_line: str, dry_run: bool) -> dict[str, Any]:
    """The launch-time half of the manifest; `harvest` fills the rest."""
    pod = load_pod(name)
    v = versions()
    return {
        "pod": name,
        "git_sha": sha,
        "config_hash": config_hash(pod),
        "model": pod.model,
        # Resolved by the run that downloads the weights; a dry run and a laptop-side
        # launch have no revision to record.
        "model_revision": None,
        "dataset_sha256": {},
        "torch": v["torch"],
        "cuda": v["cuda"],
        "triton": v["triton"],
        "transformers": v["transformers"],
        "kvpress": v["kvpress"],
        "gpu": v["gpu"],
        "launched_at": _now(),
        "harvested_at": None,
        "wall_clock_s": None,
        "command_line": command_line,
        "errors": 0,
        "records": {},
        # ALL_DONE / RUN_FAILED / BOOT_FAILED, read off the log's markers by `harvest`.
        # The watchdog destroys the instance on all three; this is where the failure shows.
        "status": None,
        # True when the log carries boot.sh's ===RUN_TIMEOUT_ marker: the run hit the
        # `--max-hours` bar. Its status is still RUN_FAILED; this keeps the reason.
        "timeout": False,
        "dry_run": dry_run,
    }


def _write_manifest(out: Path, m: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def _read_manifest(out: Path) -> dict[str, Any] | None:
    p = out / "manifest.json"
    return cast(dict[str, Any], json.loads(p.read_text())) if p.is_file() else None


# --- run / prepare --------------------------------------------------------------


def _haystack_sha256(pod: PodCfg) -> dict[str, str]:
    """``haystack:<source>`` -> sha256 of ``data/haystacks/<source>.jsonl`` for every
    source the pod's v2 tasks name, materializing (`kvdlra.eval.haystacks.ensure`) the
    ones not on disk. Empty for a pod with no v2 task."""
    # Imported here: `check` and `--dry-run` never need the eval stack it pulls in.
    from kvdlra.eval.haystacks import ensure

    tasks = [load_task(t) for t in pod.tasks]
    sources = sorted({s for t in tasks if isinstance(t, TaskV2Cfg) for s in t.haystacks})
    return {f"haystack:{s}": sha for s, sha in ensure(sources).items()}


def prepare(name: str) -> int:
    """Materialize the haystacks a pod's v2 tasks name (`run` does it too, before the
    model loads; this is the same step on its own, for a laptop or a warm pod)."""
    for key, sha in _haystack_sha256(load_pod(name)).items():
        print(f"{key} {sha}")
    return 0


# The manifest fields only the POD can fill in: `launch` writes the laptop's own
# (`gpu: none`, `cuda: none`, `model_revision: null`), and the manifest `run` writes with
# the real ones dies with the instance. `_stage_lines` is how they reach the log, and
# `STAGE_RE` reads them back -- one tuple, so the two cannot drift apart.
STAGE_KEYS = ("gpu", "cuda", "torch", "model_revision")


def _stage_lines(m: dict[str, Any]) -> list[str]:
    """The pod's own environment as `[stage] <key> <value>` lines, in the digest lines'
    pattern. A value the run has none of -- a model with no resolved revision (`None`),
    or a gpu/cuda/torch read that fell back to `versions()`'s "none" sentinel (never
    `None` there, so it needs its own check) -- prints NO line: the string "None"/"none"
    in the manifest would read as a real value, and a missing line correctly leaves the
    launch-time value standing."""
    return [f"[stage] {k} {m[k]}" for k in STAGE_KEYS if m.get(k) not in (None, "none")]


def run(name: str, out: Path, dry_run: bool) -> int:
    # The manifest first: it is what loads the pod config, so an unknown pod name raises
    # before a half-written directory exists on disk.
    pod = load_pod(name)
    m = manifest(name, _head(), shlex.join(sys.argv), dry_run)
    out.mkdir(parents=True, exist_ok=True)
    (out / "env.txt").write_text("\n".join(env_lines()) + "\n")
    _write_manifest(out, m)
    if dry_run:
        (out / "trials.jsonl").write_text("")
        print(f"{out}: manifest.json, env.txt, empty trials.jsonl (dry run)")
        return 0

    # Imported here, not at module scope: `--dry-run` and `check` must work on a laptop
    # without pulling in the eval stack (and, through it, kvpress).
    import torch

    from kvdlra.eval.data import load_model
    from kvdlra.eval.runner import run_pod

    # Everything from here to the end of the pod prints inside `replayed`, which repeats
    # the compact record lines between its markers when the block exits (L3.4a): the log
    # dies with the instance and `vastai logs` returns only its tail, so the digests and
    # the rows have to be at the END of the log as well as where they happened. The
    # digest lines are the reason the block starts here rather than at `run_pod`: nothing
    # else carries `dataset_sha256` off the pod.
    with replayed():
        # The haystacks before the weights: a source that will not download fails the pod
        # in seconds, not after the model load; their digests are evidence, so the manifest
        # carries them from here on -- and the log too: this manifest dies with the
        # instance, and `harvest` rebuilds `dataset_sha256` from these lines (`DIGEST_RE`).
        m["dataset_sha256"] = {**m["dataset_sha256"], **_haystack_sha256(pod)}
        for key, sha in m["dataset_sha256"].items():
            print(f"[stage] dataset_sha256 {key} {sha}", flush=True)
        _write_manifest(out, m)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.perf_counter()
        loaded = load_model(pod.model, device, pod.dtype)
        print(f"[stage] load_model {pod.model} ({time.perf_counter() - t0:.1f} s)", flush=True)
        m["model_revision"] = getattr(loaded[0].config, "_commit_hash", None)
        _write_manifest(out, m)
        for line in _stage_lines(m):
            print(line, flush=True)
        t_run = time.perf_counter()
        run_pod(pod, out, loaded)
    # The run's OWN span, printed for the same reason as the lines above: `_finish` puts
    # it in the manifest on the pod, and boot.sh's timestamps (which `_wall_clock_s`
    # reads, and which cover the boot too) are not always in the fetched log.
    print(f"[stage] wall_clock_s {time.perf_counter() - t_run:.1f}", flush=True)
    return 0


# --- launch -------------------------------------------------------------------


def launch_command(name: str, offer: str, sha: str, max_hours: float | None = None) -> list[str]:
    """The `vastai create instance` the boot-script header documents.

    `boot.sh` needs five things from the environment: which pod config to run, which
    commit to check out, which weights to pull and in what dtype, and the bar in hours
    it enforces on the run with `timeout` (default: the pod's pre-registered
    `gpu_budget_h`). Everything else it reads from the SHA-pinned clone. A bar of zero is
    refused, not passed on: `timeout 0h` disables the limit, and every v1 pod config
    carries `gpu_budget_h: 0.0`.
    """
    pod = load_pod(name)
    hours = pod.gpu_budget_h if max_hours is None else max_hours
    if not (math.isfinite(hours) and hours > 0):  # catches <= 0, nan, AND +/-inf
        # (nan > 0 is False; +inf > 0 is True but not finite -- `timeout infh` is no bar)
        raise ValueError(
            f"--max-hours {hours:g} is no bar: pre-register a finite gpu_budget_h > 0 in"
            f" configs/pods/{name}.yaml or pass --max-hours"
        )
    return [
        "vastai",
        "create",
        "instance",
        offer,
        "--image",
        pod.image,
        "--disk",
        "80",
        "--env",
        f"-e POD={name} -e SHA={sha} -e MODEL={pod.model} -e DTYPE={pod.dtype}"
        f" -e MAX_HOURS={hours:g}",
        "--onstart",
        "scripts/pod/boot.sh",
        "--label",
        f"kvdlra-{name}",
    ]


def prereg_error(path: Path, sha: str) -> str | None:
    """Why `path` is not a valid pre-registration for a launch at `sha`, or None.

    The rule (CLAUDE.md, `prereg/README.md`): the pre-registration is committed
    *before* the launch commit. "Before" is strict -- a prereg committed by the launch
    commit itself was written with the results in hand as far as this repo can tell.
    """
    if not path.is_file():
        return f"{path} is missing"
    first = _git("log", "--reverse", "--format=%H", "--", str(path)).stdout.split("\n")[0].strip()
    if not first:
        return f"{path} is uncommitted"
    if _git("merge-base", "--is-ancestor", first, sha).returncode != 0:
        return f"{path} first commit {first[:8]} is not an ancestor of {sha[:8]}"
    if first == _git("rev-parse", f"{sha}^{{commit}}").stdout.strip():
        return f"{path} was committed by {sha[:8]} itself, not before it"
    return None


def unpushed_error(sha: str, root: Path = REPO_ROOT) -> str | None:
    """Why launching at `sha` would boot into a CHECKOUT_FAILED, or None.

    `boot.sh` clones from GitHub and checks out `$SHA`; a commit that exists only on
    this laptop is a pod that boots, fails, and bills until the watchdog notices.
    Containment in any remote branch is the test -- which branch is not this file's
    business."""
    contains = subprocess.run(
        ["git", "-C", str(root), "branch", "-r", "--contains", sha], capture_output=True, text=True
    )
    if contains.stdout.strip():
        return None
    return f"HEAD {sha} is not on any remote branch; push first (the pod clones from GitHub)"


def launch_refusals(pod: PodCfg, sha: str) -> list[str]:
    reasons = []
    if _git("status", "--porcelain").stdout.strip():
        reasons.append("the working tree is dirty; commit and push the launch commit first")
    unpushed = unpushed_error(sha)
    if unpushed:
        reasons.append(unpushed)
    if not pod.prereg:
        reasons.append(f"pod {pod.name} names no prereg; write prereg/{pod.name}.md and commit it")
    else:
        err = prereg_error(REPO_ROOT / pod.prereg, sha)
        if err:
            reasons.append(f"prereg {err}")
    return reasons


def pods_line(name: str, instance: str, model: str) -> str:
    """One `results/<pod>/pods.txt` row for the watchdog: `<label>:<id>:<pod>:<tag>`.

    The label is `<pod>-<instance>`, NOT the pod name. The watchdog remembers harvested
    labels in `done.txt` and skips them forever, so a relaunch under a label the first
    instance already retired is never harvested and never destroyed -- it bills until
    the credit floor. Field 3 stays the pod name: it is what the `===ALL_DONE_<pod>`
    marker carries and what `pod.py harvest --pod` is given.
    """
    return f"{name}-{instance}:{instance}:{name}:{model.split('/')[-1]}\n"


def pod_name(arg: str) -> str:
    """`--pod` takes a pod name or a watchdog label (`<pod>-<instance id>`); both name
    the same `configs/pods/<pod>.yaml` and the same `results/<pod>/`. No pod name
    contains a hyphen, so the split is unambiguous."""
    m = re.fullmatch(r"(.+)-\d+", arg)
    return m.group(1) if m else arg


def launch(name: str, offer: str, dry_run: bool, max_hours: float | None = None) -> int:
    pod = load_pod(name)
    sha = _head()
    reasons = launch_refusals(pod, sha)
    try:
        cmd = launch_command(name, offer, sha, max_hours)
    except ValueError as exc:  # no enforceable bar (see launch_command)
        reasons.append(str(exc))
        cmd = []
    for r in reasons:
        print(f"REFUSE: {r}")
    if reasons:
        return 1
    out = REPO_ROOT / "results" / name
    m = manifest(name, sha, shlex.join(cmd), dry_run=False)
    if dry_run:
        print(shlex.join(cmd))
        print(json.dumps(m, indent=2, sort_keys=True))
        return 0
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout + proc.stderr)
    found = re.search(r"new_contract\D+(\d+)", proc.stdout)
    if proc.returncode != 0 or not found:
        print("REFUSE: no instance id in the vastai output; nothing was recorded")
        return 1
    instance = found.group(1)
    m["command_line"] = shlex.join(cmd)
    _write_manifest(out, m)
    with (out / "pods.txt").open("a") as f:
        f.write(pods_line(name, instance, pod.model))
    print(f"launched {name} as instance {instance}; watch with scripts/pod/watchdog.sh {name}")
    return 0


# --- harvest ------------------------------------------------------------------


def _shrink_refusal(out: Path, name: str, n_new: int) -> str | None:
    """Why writing `n_new` rows over `out/name` is a harvest to redo, or None.

    A re-harvest whose parse came back SHORTER than what is on disk is the 5000-line
    fallback fetch, a truncated log, or a `[pplw]` group whose fragments did not all
    arrive -- never a pod that produced less. The rule covered `trials.jsonl` alone,
    so a fetch carrying every `[trial]` line but half a sweep silently replaced a good
    `pplw.jsonl`; it covers every record file the harvest writes now. `--force` is the
    override, as before.
    """
    p = out / name
    n_old = len(read_jsonl(p)) if p.is_file() else 0
    if n_old <= n_new:
        return None
    return f"REFUSE: {name} would shrink from {n_old} to {n_new} rows; pass --force to overwrite"


# `run` (the haystack sources) and `runner._ppl_rows` (the perplexity corpora) print one
# `[stage] dataset_sha256 <key> <sha>` line per digest, because the manifest they write
# it into stays on the destroyed instance: every harvested manifest carried
# `dataset_sha256: {}`. The watchdog keeps `[stage]` rows; the last line for a key wins.
DIGEST_RE = re.compile(r"^\[stage\] dataset_sha256 (\S+) ([0-9a-f]{64})\s*$", re.M)
# THE rationale for the `[stage] cell` line, in one place (every emitter in
# `kvdlra.eval.runner` points here). One line per completed cell, on all three axes:
# retrieval (`_cell`, per arm x sub-task), perplexity (`_ppl_rows`, per arm x ctx sweep,
# keyed by the ppl TASK name so the two axes share one namespace without pooling) and
# latency (per arm x ctx). It is the ONLY clock a harvest carries -- `[trial]` and cell
# rows have no timestamp, a harvested `wall_clock_s` is null, and the watchdog's per-poll
# `sort -u` destroys arrival order -- so the per-arm min/sample that sizes the next pod
# is read from here. The seconds ride the line, so the dedupe cannot hurt it; an arm that
# failed still prints the seconds it burned; last line for a key wins.
CELL_S_RE = re.compile(
    r"^\[stage\] cell arm=(\S+) task=(\S+) ctx=(\d+) elapsed_s=([0-9.]+) n=\d+\s*$", re.M
)
# `run`'s `_stage_lines` (the GPU name carries spaces, so the value runs to end of line)
# and its post-`run_pod` span. Last line for a key wins; an absent key leaves the
# launch-time value, which is the laptop's and says so (`gpu: none`).
STAGE_RE = re.compile(rf"^\[stage\] ({'|'.join(STAGE_KEYS)}) (\S.*?)\s*$", re.M)
WALL_S_RE = re.compile(r"^\[stage\] wall_clock_s ([0-9.]+)\s*$", re.M)


def _env_from_log(text: str) -> list[str] | None:
    """`env_lines()` rebuilt from boot.sh's ENV block, or None if the log carries none.

    `run` writes env.txt ON THE POD, and a log harvest never brings that file back -- so
    every harvested pod failed `check` on `env: env.txt is missing`, which is a missing
    file, not a missing environment: boot.sh prints the same set into the log. The block
    is read line by line rather than between its `===ENV_` markers because the watchdog
    dedupes the rows with `sort -u`, which scatters the block's contents away from them.

    A log with no `torch=` line gets NO env.txt: an environment invented from the laptop
    running the harvest is worse than the missing-file failure it would silence. What the
    block does not print is `unrecorded`, which `_env_fails` treats as the absent line it
    is.
    """
    got: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("device="):  # the GPU name has spaces; not a k=v token list
            got["device"] = line[len("device=") :].strip()
        elif line.startswith(("torch=", "transformers=", "triton=")):
            for tok in line.split():
                key, _, val = tok.partition("=")
                if val:
                    got[key] = val
    if "torch" not in got:
        return None
    got["cuda"] = got.get("cuda_build", "unrecorded")  # torch's build, not a distribution
    return [f"{p}=={got.get(p, 'unrecorded')}" for p in ENV_PKGS] + [
        f"gpu={got.get('device', 'none')}"
    ]


def _wall_clock_s(text: str) -> float | None:
    """Seconds from the environment header to ALL_DONE, when the log carries timestamps
    (`vastai logs` does not always), else None."""
    start = end = None
    for line in text.splitlines():
        m = TS_RE.match(line)
        if not m:
            continue
        if "ENV_BEGIN" in line and start is None:
            start = m.group(1)
        if "ALL_DONE" in line:
            end = m.group(1)
    if not (start and end):
        return None
    with contextlib.suppress(ValueError):
        a = datetime.fromisoformat(start.replace(" ", "T"))
        b = datetime.fromisoformat(end.replace(" ", "T"))
        return (b - a).total_seconds()
    return None


def _fetch_log(out: Path, name: str) -> str:
    """The pod's log, through the vast.ai CLI. The 30000-line fetch came back EMPTY for
    three finished Week-20 pods (never harvested, never destroyed, idle overnight); a
    smaller tail worked, so fall back before giving up."""
    pods = out / "pods.txt"
    if not pods.is_file():
        raise SystemExit(f"no {pods}: launch {name} first, or pass --log")
    instance = [x for x in pods.read_text().splitlines() if x.strip()][-1].split(":")[1]
    for tail in ("30000", "5000"):
        text = subprocess.run(
            ["vastai", "logs", instance, "--tail", tail], capture_output=True, text=True
        ).stdout
        if text.strip():
            return text
    raise SystemExit(f"empty log fetch for instance {instance}")


# boot.sh's pre-run failures. The watchdog destroys the instance on any of them (no
# idle billing) and the manifest keeps the reason. QUANTO/HQQ_MISSING do not stop the
# run, but a pod whose quant backend is absent is not the pod that was configured.
BOOT_FAILURES = (
    "CLONE_FAILED",
    "CHECKOUT_FAILED",
    "DEPS_FAILED",
    "MODEL_FAILED",
    "QUANTO_MISSING",
    "HQQ_MISSING",
)


def _status(text: str) -> str | None:
    """The end marker `boot.sh` printed. A run whose `pod.py run` exited non-zero prints
    `===RUN_FAILED_` instead of `===ALL_DONE_`; the watchdog destroys the instance on
    either (no idle billing), so the manifest is where the failure has to survive. A
    boot failure outranks both: a log carrying one never ran what the config asked for,
    whatever it printed afterwards."""
    if any(f"==={m}_" in text for m in BOOT_FAILURES):
        return "BOOT_FAILED"
    for marker in ("RUN_FAILED", "ALL_DONE"):
        if f"==={marker}_" in text:
            return marker
    return None


def _generators_by_subtask(pod: PodCfg) -> dict[str, set[str]]:
    """Sub-task name -> the generators that produce it in this pod (`ppl` has none)."""
    out: dict[str, set[str]] = {}
    for tname in pod.tasks:
        t = load_task(tname)
        if t.generator != "ppl":
            for sub in t.tasks:
                out.setdefault(sub, set()).add(t.generator)
    return out


def _fill_generators(name: str, trials: list[TrialRecord]) -> None:
    """Name the generator behind every harvested row, from the pod's own task configs.

    No v1 `[trial]` line carries `generator=`, and `check` keys a cell by the generator
    -- so a harvest that left it null keyed every cell "None" and rejected a complete,
    correct pod. A row that names its own generator (the runner prints it) is taken as
    it stands; the configs only fill the gap.

    When two of a pod's tasks own the same sub-task name -- w19_fork runs the in-house
    and the official 16K tasks, which both call a sub-task `vt` -- a row that does not
    name its generator is unattributable, and guessing would pool two benchmarks into
    one cell and call the doubled count complete. So it stops.
    """
    gens = _generators_by_subtask(load_pod(name))
    for r in trials:
        if r["generator"] is not None:
            continue
        owners = gens.get(r["task"], set())
        if len(owners) > 1:
            raise SystemExit(
                f"pod {name}: sub-task {r['task']} belongs to more than one generator;"
                " the runner must emit generator= in the [trial] row"
            )
        r["generator"] = next(iter(owners), None)


def harvest(name: str, log: Path | None, out: Path, force: bool) -> int:
    model = load_pod(name).model
    out.mkdir(parents=True, exist_ok=True)
    text = log.read_text() if log else _fetch_log(out, name)
    source = str(log) if log else f"vastai logs ({_now()})"
    # Exact-duplicate lines, dropped in place (L3.4a). The pod repeats its record lines at
    # the end of the run (`records.replayed`), and a fetch of the pod's own stdout carries
    # both copies -- the watchdog's `sort -u` collapses them in `<label>.raw`, a hand-fetched
    # dump does not. Order-preserving, so the `source:<line>` a record cites is still the
    # line it was first printed on. A `[diag]` row this drops is a row printed twice
    # byte-for-byte, which is what the watchdog path has always kept one of.
    text = "\n".join(dict.fromkeys(text.splitlines()))

    # Parse EVERY artifact before writing ANY of them. An incomplete [pplw] part set
    # raises SystemExit, and the 5000-line fallback fetch returns a shorter log than the
    # one already harvested -- either used to land after trials.jsonl had been replaced,
    # leaving a half-updated directory that looks like a complete, smaller run.
    # Two artifacts of one sweep, two schemas, two files -- parsed independently, never
    # one instead of the other.
    trials = parse_trial_lines(text, model, source)
    _fill_generators(name, trials)
    pplw = parse_pplw_lines(text, model, source)
    ppl = parse_ppl_lines(text, model, source)
    lat = parse_latency_lines(text, model, source)
    diag, diag_skipped = parse_diag_lines(text, model, source)

    # Every file this harvest is about to write, checked against what is on disk BEFORE
    # any of them is written -- the same all-or-nothing rule the parse above follows.
    parsed: dict[str, list[Any]] = {
        "trials.jsonl": trials,
        "pplw.jsonl": pplw,
        "ppl.jsonl": ppl,
        "latency.jsonl": lat,
        "diag.jsonl": diag,
    }
    refusals = [r for r in (_shrink_refusal(out, f, len(x)) for f, x in parsed.items()) if r]
    if refusals and not force:
        for r in refusals:
            print(r)
        return 1

    # `trials.jsonl` is written even when it is empty: it is the file `check` reads to
    # tell a pod that produced nothing from one that was never harvested. The other four
    # are written only when the log carried rows -- an absent file is not a short one.
    write_jsonl(out / "trials.jsonl", trials)
    records = {"trials.jsonl": len(trials)}
    for fname, rows in parsed.items():
        if fname != "trials.jsonl" and rows:
            write_jsonl(out / fname, rows)
            records[fname] = len(rows)

    m = _read_manifest(out) or manifest(name, _head(), source, False)
    m["harvested_at"] = _now()
    m["dataset_sha256"] = {**m.get("dataset_sha256", {}), **dict(DIGEST_RE.findall(text))}
    cells_s = {f"{a}/{t}/{c}": float(s) for a, t, c, s in CELL_S_RE.findall(text)}
    if cells_s:  # a log printed before L3.1c leaves the key absent, not empty
        m["cell_elapsed_s"] = {**m.get("cell_elapsed_s", {}), **cells_s}
    m.update(dict(STAGE_RE.findall(text)))  # the pod's card, CUDA build, torch, revision
    m["records"] = records
    # Both axes: a perplexity arm that raised has no record to carry the failure, only
    # the `[error]` line, so counting trial rows alone called such a pod clean.
    m["errors"] = sum(1 for t in trials if t["error"] is not None) + len(
        parse_error_lines(text, source)
    )
    # boot.sh's timestamps first (they span the boot too, and are the billable clock);
    # `run`'s own printed span is the fallback for the logs that come back without them,
    # which is every log `vastai logs` has returned so far.
    ran = WALL_S_RE.findall(text)
    m["wall_clock_s"] = (
        _wall_clock_s(text) or (float(ran[-1]) if ran else None) or m.get("wall_clock_s")
    )
    m["status"] = _status(text)
    m["timeout"] = "===RUN_TIMEOUT_" in text  # the RUN_FAILED was boot.sh's `timeout`
    # A `[diag]` line the fetch cut in half parses into nothing. Counted in the manifest
    # (and failed by `check`) rather than dropped: the skip is evidence the log came back
    # truncated, which is a harvest to redo, not a pod that printed no diagnostics.
    m["diag_skipped"] = diag_skipped
    _write_manifest(out, m)
    print(f"{out}: " + ", ".join(f"{k}={v}" for k, v in records.items()))
    # Never written over an env.txt `run` left on the pod: that file is the environment,
    # this one only a reconstruction of it. A refused harvest returned above and writes
    # no env.txt either -- the environment belongs to the records it came back with.
    epath = out / "env.txt"
    env = None if epath.is_file() else _env_from_log(text)
    if env:
        epath.write_text("\n".join(env) + "\n")
        print(f"harvest: wrote {epath} from the log's ENV block")
    if diag_skipped:
        print(
            f"harvest: {diag_skipped} [diag] line(s) skipped"
            " (payload not JSON -- truncated by the log fetch?)"
        )
    return 0


# --- check --------------------------------------------------------------------


def _pyproject_pins() -> dict[str, str]:
    deps = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    pins = (d.split("==") for d in deps if "==" in d)
    return {name: ver for name, ver in pins if name in PINNED}


def _expected_cells(pod: PodCfg) -> dict[tuple[str, str, str, int], tuple[int, str]]:
    """Every cell the config calls for: arm x generator x sub-task x ctx -> (n, task).

    The arm key is the string a record carries, which is `legacy_name` where one is set
    (the v1 arm names live on in the records) and the config name otherwise. The
    GENERATOR is part of the key because the in-house and official RULER generators
    reuse the sub-task names `niah_multivalue` and `vt` at the same context length, and
    a pod naming both (w19_fork) would otherwise pool two different benchmarks into one
    cell and call the doubled count complete.

    A cell is `n_trials x len(seeds)` records. A task's `depths` grid does NOT multiply
    it: the generator sweeps the grid across trial indices (`depths[trial % len]`), so
    pinning depths changes which needle each trial places, not how many trials run.

    `ppl` and `latency` tasks contribute no cells: a perplexity sweep is one number per
    (arm, ctx) and a decode measurement one row per (arm, ctx, batch), not sets of
    Bernoulli trials -- each is checked through its own file instead of here.
    """
    arms = [load_arm(a) for a in pod.arms]
    expect = {}
    for tname in pod.tasks:
        t = load_task(tname)
        if t.generator in ("ppl", "latency"):
            continue
        for sub in t.tasks:
            for arm in arms:
                expect[(arm.legacy_name or arm.name, t.generator, sub, t.ctx)] = (
                    t.n_trials * len(t.seeds),
                    tname,
                )
    return expect


def _cell_fails(pod: PodCfg, trials: list[TrialRecord]) -> list[str]:
    """Every configured cell holds exactly `n_trials x len(seeds)` records, errors
    counted. Checking only the cells PRESENT is how a pod that produced nothing, or one
    arm of three, used to pass -- so the expected set comes from the config, not from
    the records, and a cell with no rows fails like a short one."""
    expect = _expected_cells(pod)
    counts = Counter((r["arm"], str(r.get("generator")), r["task"], r["ctx"]) for r in trials)
    fails = []
    for key in sorted(expect.keys() | counts.keys()):
        arm, gen, task, ctx = key
        want, n = expect.get(key), counts.get(key, 0)
        if want is None:
            fails.append(
                f"cells: {arm} {gen}/{task} ctx={ctx} is no cell of any task in {pod.name}"
            )
        elif n == 0:
            fails.append(f"cells: {arm} {gen}/{task} ctx={ctx} has 0 of {want[0]} records")
        elif n != want[0]:
            fails.append(
                f"cells: {arm} {gen}/{task} ctx={ctx} has n={n}, expected {want[0]}"
                f" ({want[1]}: n_trials x seeds)"
            )
    return fails


def _ppl_fails(pod: PodCfg, d: Path) -> list[str]:
    """Every `ppl` task holds one `PplRecord` per (arm, ctx) in `ppl.jsonl`.

    A perplexity sweep is one number per (arm, ctx), not a set of Bernoulli trials, so
    `_expected_cells` skips it -- which left nothing at all looking at `ppl.jsonl`, and
    a pod whose entire sweep died passed `check` clean.
    """
    p = d / "ppl.jsonl"
    got = {(r["arm"], r["ctx"]) for r in read_jsonl(p)} if p.is_file() else set()
    ctxs = [t.ctx for t in (load_task(x) for x in pod.tasks) if t.generator == "ppl"]
    arms = [load_arm(a) for a in pod.arms]
    want = {(a.legacy_name or a.name, c) for a in arms for c in ctxs}
    return [f"ppl: {arm} ctx={ctx} has no perplexity record" for arm, ctx in sorted(want - got)]


def _pplw_fails(pod: PodCfg, d: Path) -> list[str]:
    """Every `ppl` task holds `n_samples` per-window rows per (arm, ctx) in `pplw.jsonl`.

    The sweep leaves two artifacts and `_ppl_fails` only looks at the aggregate one, so a
    run whose per-window file came back short -- a truncated fetch, a `[pplw]` group whose
    fragments never all arrived -- still passed with the pooled number intact and nothing
    left to re-pool it from. The count is the protocol the config pre-registers, not
    whatever came back.
    """
    p = d / "pplw.jsonl"
    got = Counter((r["arm"], r["ctx"]) for r in read_jsonl(p)) if p.is_file() else Counter()
    tasks = [t for t in (load_task(x) for x in pod.tasks) if t.generator == "ppl"]
    arms = [a.legacy_name or a.name for a in (load_arm(x) for x in pod.arms)]
    return [
        f"pplw: {arm} ctx={t.ctx} has {got[(arm, t.ctx)]} of {t.n_samples} window rows"
        for t in tasks
        for arm in sorted(arms)
        if got[(arm, t.ctx)] != t.n_samples
    ]


def _latency_fails(pod: PodCfg, d: Path) -> list[str]:
    """Every `latency` task holds one `LatencyRecord` per (arm, ctx, batch).

    The decode axis writes one row per point and no trial at all, so without this rule a
    pod whose 64K points OOMed -- or whose whole task never ran -- would show an empty
    `latency.jsonl` and nothing to fail on, exactly the hole `_ppl_fails` closes for the
    perplexity axis.
    """
    p = d / "latency.jsonl"
    got = {(r["arm"], r["ctx"], r["batch"]) for r in read_jsonl(p)} if p.is_file() else set()
    tasks = [t for t in (load_task(x) for x in pod.tasks) if t.generator == "latency"]
    arms = [load_arm(a) for a in pod.arms]
    want = {
        (a.legacy_name or a.name, c, b)
        for t in tasks
        for c in (t.ctxs or [t.ctx])
        for b in t.batch_sizes
        for a in arms
    }
    return [
        f"latency: {arm} ctx={ctx} batch={batch} has no decode record"
        for arm, ctx, batch in sorted(want - got)
    ]


def _env_fails(d: Path) -> list[str]:
    p = d / "env.txt"
    if not p.is_file():
        return ["env: env.txt is missing"]
    got = dict(
        cast(tuple[str, str], tuple(x.split("==", 1)))
        for x in p.read_text().splitlines()
        if "==" in x
    )
    fails = []
    for pkg, pin in _pyproject_pins().items():
        ver = got.get(pkg, "unrecorded")  # a harvest writes `unrecorded`: the same gap
        if ver == "unrecorded":
            fails.append(f"env: {pkg} is not recorded in env.txt")
        # A pod reports torch as a PEP 440 LOCAL version (`2.11.0+cu128`); the pyproject
        # pin is the public one, and the CUDA build is recorded separately as `cuda`.
        elif ver.split("+", 1)[0] != pin:
            fails.append(f"env: {pkg}=={ver} != the pyproject pin {pin}")
    return fails


def check(d: Path, log: Path | None = None) -> int:
    m = _read_manifest(d)
    if m is None:
        print(f"CHECK FAIL manifest: {d / 'manifest.json'} does not exist")
        return 1
    if "converted_by" in m:  # a paper-v1 archive directory: static evidence, not a run
        print(f"{d}: archive, skipped")
        return 0
    name = str(m.get("pod", ""))
    try:
        pod = load_pod(name)
    except FileNotFoundError:
        print(f"CHECK FAIL config_hash: no configs/pods/{name}.yaml for this manifest")
        return 1

    fails = []
    if m.get("config_hash") != config_hash(pod):
        fails.append(f"config_hash: manifest {m.get('config_hash')} != configs/pods/{name}.yaml")
    sha = str(m.get("git_sha", ""))
    if _git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        fails.append(f"git_sha: {sha!r} does not resolve in this clone")
    elif pod.prereg:
        err = prereg_error(REPO_ROOT / pod.prereg, sha)
        if err:
            fails.append(f"prereg: {err}")

    tpath = d / "trials.jsonl"
    trials = [cast(TrialRecord, r) for r in read_jsonl(tpath)] if tpath.is_file() else []
    n_err = sum(1 for t in trials if t["error"] is not None)
    m_err = int(m.get("errors") or 0)
    # Ruling R29. Recording a raised trial rather than skipping it keeps the cell full,
    # which is the point -- and which leaves the cell rule with nothing to say about a
    # pod whose every trial failed. So the count is its own rule, with no tolerance knob.
    if n_err:
        fails.append(f"errors: {n_err} trial(s) raised (see the error field in trials.jsonl)")
    # The manifest counts all three axes; a perplexity or decode point that raised leaves
    # an `[error]` log line and no record at all, so the excess over the trial rows is the
    # non-trial count, not a disagreement -- name it instead of reporting a mismatch.
    # FEWER than the rows is a real one: the manifest cannot have counted what it has not
    # seen. Since R39 the watchdog's row filter keeps `[error]` lines, so when a harvested
    # log is on hand (`--log`), its `[error]` line count is cross-checked against
    # `m_err - n_err`; with no log passed, there is nothing to check it against, and the
    # cross-check is skipped.
    if m_err > n_err:
        fails.append(
            f"errors: {n_err} trial error(s) + {m_err - n_err} perplexity/latency error(s)"
        )
        if log and log.is_file():
            lines = log.read_text().splitlines()
            n_log_err = sum(1 for ln in lines if ln.startswith("[error]"))
            if n_log_err != m_err - n_err:
                fails.append(
                    f"errors: {log} has {n_log_err} [error] line(s), wants {m_err - n_err}"
                )
    elif m_err < n_err:
        fails.append(
            f"errors: manifest/rows mismatch -- the manifest counts {m_err},"
            f" the trial records {n_err}"
        )
    n_diag_skipped = int(m.get("diag_skipped") or 0)
    if n_diag_skipped:
        fails.append(f"diag: {n_diag_skipped} [diag] line(s) were unparseable")
    if trials or not m.get("dry_run"):  # a dry run has no records to count
        fails += _cell_fails(pod, trials)
        fails += _ppl_fails(pod, d)
        fails += _pplw_fails(pod, d)
        fails += _latency_fails(pod, d)
    fails += _env_fails(d)

    for f in fails:
        print(f"CHECK FAIL {f}")
    if not fails:
        print(f"{d}: OK ({len(trials)} trials, {n_err} errors)")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate a pod config (on the GPU)")
    r.add_argument("--pod", required=True)
    r.add_argument("--out", default=None, help="default: results/<pod>")
    r.add_argument("--dry-run", action="store_true", help="manifest + env only, no eval")
    pr = sub.add_parser("prepare", help="materialize the haystacks a pod's v2 tasks name")
    pr.add_argument("--pod", required=True)
    ln = sub.add_parser("launch", help="create the vast.ai instance that runs a pod")
    ln.add_argument("--pod", required=True)
    ln.add_argument("--offer", required=True)
    ln.add_argument("--dry-run", action="store_true", help="print the command, launch nothing")
    ln.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="the bar boot.sh enforces on the run with `timeout`; default: the pod's gpu_budget_h",
    )
    h = sub.add_parser("harvest", help="parse a pod's log into records")
    h.add_argument("--pod", required=True, help="pod name, or a watchdog label <pod>-<instance>")
    h.add_argument("--log", default=None, help="default: fetch it with the vast.ai CLI")
    h.add_argument("--out", default=None, help="default: results/<pod>")
    h.add_argument(
        "--force", action="store_true", help="overwrite even if the parse has fewer trials"
    )
    c = sub.add_parser("check", help="verify a results directory")
    c.add_argument("dir")
    c.add_argument("--log", default=None, help="cross-check its [error] lines vs manifest.errors")
    a = ap.parse_args()
    if a.cmd == "check":
        return check(Path(a.dir), Path(a.log) if a.log else None)
    name = pod_name(a.pod)
    out = Path(a.out) if getattr(a, "out", None) else REPO_ROOT / "results" / name
    if a.cmd == "run":
        return run(name, out, a.dry_run)
    if a.cmd == "prepare":
        return prepare(name)
    if a.cmd == "launch":
        return launch(name, a.offer, a.dry_run, a.max_hours)
    return harvest(name, Path(a.log) if a.log else None, out, a.force)


if __name__ == "__main__":
    raise SystemExit(main())
