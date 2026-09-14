"""The pod entrypoint: `launch` from the laptop, `run` on the GPU, `harvest` the log,
`check` what came back.

One directory per pod, `results/<pod>/`, holding `manifest.json` (git SHA, config hash,
model, library versions, GPU, wall clock, command line), `env.txt` (the pinned set as
`name==version`), and the records a harvest parsed out of the log: `trials.jsonl`,
`ppl.jsonl` (aggregate perplexity), `pplw.jsonl` (the per-window NLLs behind it -- a
different schema, hence a different file), `diag.jsonl`. `check` is the gate every citable
number passes: it re-derives the config hash from `configs/pods/<pod>.yaml`, resolves the
SHA, enforces the pre-registration commit order, and requires EVERY cell the config calls
for -- arm x task x sub-task x ctx -- to hold exactly `n_trials x len(seeds)` records. A
trial that raised is recorded with `error` and still counted, so a cell can never silently
shrink; a cell with no records at all is the loudest failure there is, which is what makes
a pod that produced nothing impossible to pass off as a clean run.

`run`'s eval loop lands in Task 8; here `run` writes the manifest and the environment and
`--dry-run` stops there, which is what the tests and `make check` exercise. Archived
paper-v1 manifests carry `converted_by` and are skipped: they are static evidence of runs
that happened before this entrypoint existed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata as md
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import tomllib

from kvdlra.eval.config import PodCfg, config_hash, load_arm, load_pod, load_task
from kvdlra.eval.records import (
    TrialRecord,
    parse_diag_lines,
    parse_ppl_lines,
    parse_pplw_lines,
    parse_trial_lines,
    read_jsonl,
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
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    with contextlib.suppress(Exception):
        import torch

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
        # Resolved by the run that downloads the weights (Task 8); a dry run and a
        # laptop-side launch have no revision to record.
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
        # ALL_DONE / RUN_FAILED, read off the log's end marker by `harvest`. The
        # watchdog destroys the instance either way; this is where the failure shows.
        "status": None,
        "dry_run": dry_run,
    }


def _write_manifest(out: Path, m: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def _read_manifest(out: Path) -> dict[str, Any] | None:
    p = out / "manifest.json"
    return cast(dict[str, Any], json.loads(p.read_text())) if p.is_file() else None


# --- run ----------------------------------------------------------------------


def run(name: str, out: Path, dry_run: bool) -> int:
    # The manifest first: it is what loads the pod config, so an unknown pod name raises
    # before a half-written directory exists on disk.
    m = manifest(name, _head(), shlex.join(sys.argv), dry_run)
    out.mkdir(parents=True, exist_ok=True)
    (out / "env.txt").write_text("\n".join(env_lines()) + "\n")
    _write_manifest(out, m)
    if dry_run:
        (out / "trials.jsonl").write_text("")
        print(f"{out}: manifest.json, env.txt, empty trials.jsonl (dry run)")
        return 0
    print(f"{out}: manifest.json + env.txt written; the eval loop lands in Task 8")
    return 2


# --- launch -------------------------------------------------------------------


def launch_command(name: str, offer: str, sha: str) -> list[str]:
    """The `vastai create instance` the boot-script header documents.

    `boot.sh` needs four things from the environment: which pod config to run, which
    commit to check out, which weights to pull and in what dtype. Everything else it
    reads from the SHA-pinned clone.
    """
    pod = load_pod(name)
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
        f"-e POD={name} -e SHA={sha} -e MODEL={pod.model} -e DTYPE={pod.dtype}",
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


def launch_refusals(pod: PodCfg, sha: str) -> list[str]:
    reasons = []
    if _git("status", "--porcelain").stdout.strip():
        reasons.append("the working tree is dirty; commit and push the launch commit first")
    if not pod.prereg:
        reasons.append(f"pod {pod.name} names no prereg; write prereg/{pod.name}.md and commit it")
    else:
        err = prereg_error(REPO_ROOT / pod.prereg, sha)
        if err:
            reasons.append(f"prereg {err}")
    return reasons


def launch(name: str, offer: str, dry_run: bool) -> int:
    pod = load_pod(name)
    sha = _head()
    reasons = launch_refusals(pod, sha)
    for r in reasons:
        print(f"REFUSE: {r}")
    if reasons:
        return 1
    cmd = launch_command(name, offer, sha)
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
    with (out / "pods.txt").open("a") as f:  # <pod>:<id>:<mode>:<tag> for the watchdog
        f.write(f"{name}:{instance}:{name}:{pod.model.split('/')[-1]}\n")
    print(f"launched {name} as instance {instance}; watch with scripts/pod/watchdog.sh {name}")
    return 0


# --- harvest ------------------------------------------------------------------


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return len(rows)


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


def _status(text: str) -> str | None:
    """The end marker `boot.sh` printed. A run whose `pod.py run` exited non-zero prints
    `===RUN_FAILED_` instead of `===ALL_DONE_`; the watchdog destroys the instance on
    either (no idle billing), so the manifest is where the failure has to survive."""
    for marker in ("RUN_FAILED", "ALL_DONE"):
        if f"==={marker}_" in text:
            return marker
    return None


def harvest(name: str, log: Path | None, out: Path) -> int:
    model = load_pod(name).model
    out.mkdir(parents=True, exist_ok=True)
    text = log.read_text() if log else _fetch_log(out, name)
    source = str(log) if log else f"vastai logs ({_now()})"
    trials = parse_trial_lines(text, model, source)
    write_jsonl(out / "trials.jsonl", trials)
    records = {"trials.jsonl": len(trials)}

    # Two artifacts of one sweep, two schemas, two files -- parsed independently, never
    # one instead of the other.
    pplw = parse_pplw_lines(text, model, source)
    if pplw:
        write_jsonl(out / "pplw.jsonl", pplw)
        records["pplw.jsonl"] = len(pplw)
    ppl = parse_ppl_lines(text, model, source)
    if ppl:
        write_jsonl(out / "ppl.jsonl", ppl)
        records["ppl.jsonl"] = len(ppl)
    diag = parse_diag_lines(text, source)
    if diag:
        records["diag.jsonl"] = _jsonl(out / "diag.jsonl", diag)

    m = _read_manifest(out) or manifest(name, _head(), source, False)
    m["harvested_at"] = _now()
    m["records"] = records
    m["errors"] = sum(1 for t in trials if t["error"] is not None)
    m["wall_clock_s"] = _wall_clock_s(text) or m.get("wall_clock_s")
    m["status"] = _status(text)
    _write_manifest(out, m)
    print(f"{out}: " + ", ".join(f"{k}={v}" for k, v in records.items()))
    return 0


# --- check --------------------------------------------------------------------


def _pyproject_pins() -> dict[str, str]:
    deps = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    pins = (d.split("==") for d in deps if "==" in d)
    return {name: ver for name, ver in pins if name in PINNED}


def _expected_cells(pod: PodCfg) -> dict[tuple[str, str, int], tuple[int, str]]:
    """Every cell the config calls for: arm x task x sub-task x ctx -> (n, task name).

    The arm key is the string a record carries, which is `legacy_name` where one is set
    (the v1 arm names live on in the records) and the config name otherwise. `ppl` tasks
    contribute no cells: a perplexity sweep is one number per (arm, ctx), not a set of
    Bernoulli trials, and it is checked through `ppl.jsonl` rather than here.
    """
    arms = [load_arm(a) for a in pod.arms]
    expect = {}
    for tname in pod.tasks:
        t = load_task(tname)
        if t.generator == "ppl":
            continue
        for sub in t.tasks:
            for arm in arms:
                expect[(arm.legacy_name or arm.name, sub, t.ctx)] = (
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
    counts = Counter((r["arm"], r["task"], r["ctx"]) for r in trials)
    fails = []
    for key in sorted(expect.keys() | counts.keys()):
        arm, task, ctx = key
        want, n = expect.get(key), counts.get(key, 0)
        if want is None:
            fails.append(f"cells: {arm} {task} ctx={ctx} is no cell of any task in {pod.name}")
        elif n == 0:
            fails.append(f"cells: {arm} {task} ctx={ctx} has 0 of {want[0]} records")
        elif n != want[0]:
            fails.append(
                f"cells: {arm} {task} ctx={ctx} has n={n}, expected {want[0]}"
                f" ({want[1]}: n_trials x seeds)"
            )
    return fails


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
        if pkg not in got:
            fails.append(f"env: {pkg} is not recorded in env.txt")
        elif got[pkg] != pin:
            fails.append(f"env: {pkg}=={got[pkg]} != the pyproject pin {pin}")
    return fails


def check(d: Path) -> int:
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
    if n_err != m.get("errors"):
        fails.append(f"errors: manifest says {m.get('errors')}, records hold {n_err}")
    if trials or not m.get("dry_run"):  # a dry run has no records to count
        fails += _cell_fails(pod, trials)
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
    ln = sub.add_parser("launch", help="create the vast.ai instance that runs a pod")
    ln.add_argument("--pod", required=True)
    ln.add_argument("--offer", required=True)
    ln.add_argument("--dry-run", action="store_true", help="print the command, launch nothing")
    h = sub.add_parser("harvest", help="parse a pod's log into records")
    h.add_argument("--pod", required=True)
    h.add_argument("--log", default=None, help="default: fetch it with the vast.ai CLI")
    h.add_argument("--out", default=None, help="default: results/<pod>")
    c = sub.add_parser("check", help="verify a results directory")
    c.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "check":
        return check(Path(a.dir))
    out = Path(a.out) if getattr(a, "out", None) else REPO_ROOT / "results" / a.pod
    if a.cmd == "run":
        return run(a.pod, out, a.dry_run)
    if a.cmd == "launch":
        return launch(a.pod, a.offer, a.dry_run)
    return harvest(a.pod, Path(a.log) if a.log else None, out)


if __name__ == "__main__":
    raise SystemExit(main())
