"""`scripts/pod.py`: the manifest a pod writes, the rules `check` enforces, the log a
harvest turns into records, and the refusals that stand between a dirty tree and a
GPU bill. Every test is CPU-only and network-free -- `launch` is only ever exercised
through its refusals and `--dry-run`, which print the command instead of running it."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pod
import pytest

from kvdlra.eval.config import TaskV2Cfg, config_hash, load_arm, load_pod, load_task
from kvdlra.eval.frontier import build_arm

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """The CLI as a pod (or a human) invokes it, from the repo root whatever the
    process cwd is -- `make test` and CI stand in different places."""
    return subprocess.run(
        [sys.executable, "scripts/pod.py", *args], capture_output=True, text=True, cwd=REPO_ROOT
    )


def test_dry_run_writes_manifest_and_env(tmp_path: Path) -> None:
    r = _run("run", "--pod", "w18_g1", "--out", str(tmp_path), "--dry-run")
    assert r.returncode == 0, r.stderr
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["config_hash"] == config_hash(load_pod("w18_g1")) and (tmp_path / "env.txt").exists()


def test_check_rejects_tampered_config_hash(tmp_path: Path) -> None:
    _run("run", "--pod", "w18_g1", "--out", str(tmp_path), "--dry-run")
    m = json.loads((tmp_path / "manifest.json").read_text())
    m["config_hash"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    r = _run("check", str(tmp_path))
    assert r.returncode == 1 and "config_hash" in r.stdout + r.stderr


def test_check_counts_errors(tmp_path: Path) -> None:
    _run("run", "--pod", "w18_g1", "--out", str(tmp_path), "--dry-run")
    (tmp_path / "trials.jsonl").write_text(
        json.dumps(
            {
                "model": "m", "arm": "a", "task": "t", "ctx": 1, "seed": 0, "trial": 0,
                "hit": 0, "frac": 0.0, "generator": "inhouse", "haystack_id": None,
                "depth": None, "code_family": None, "prompt_sha256": None,
                "error": "RuntimeError: boom", "source": "x",
            }
        )
        + "\n"
    )  # fmt: skip
    r = _run("check", str(tmp_path))
    assert r.returncode == 1 and "errors" in r.stdout + r.stderr  # manifest 0, records 1


@pytest.fixture(scope="module")
def dry_pod(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One dry run for the tests below to copy: `run` imports torch to record the GPU,
    which costs more than the rest of this file put together."""
    d = tmp_path_factory.mktemp("dry")
    _run("run", "--pod", "w18_g1", "--out", str(d), "--dry-run")
    return d


def _copy(dry_pod: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(dry_pod, tmp_path / "pod"))


def test_dry_run_manifest_is_checkable(dry_pod: Path, tmp_path: Path) -> None:
    """The cell-count rule is the one rule a dry run cannot satisfy (it has no
    records); with `dry_run: true` and an empty trials.jsonl the manifest must pass."""
    tmp_path = _copy(dry_pod, tmp_path)
    r = _run("check", str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "trials.jsonl").read_text() == ""


def test_check_skips_the_paper_v1_archive() -> None:
    """A manifest carrying `converted_by` is static archived evidence, not a pod run."""
    r = _run("check", str(REPO_ROOT / "results" / "paper-v1" / "w19-a1-llama"))
    assert r.returncode == 0 and "archive, skipped" in r.stdout


def test_check_rejects_a_short_cell(dry_pod: Path, tmp_path: Path) -> None:
    """6 trials x 2 seeds = 12 records per RULER cell; 2 is a silently reduced n."""
    tmp_path = _copy(dry_pod, tmp_path)
    rows = [
        {
            "model": "m", "arm": "bugSseed-r64-h256", "task": "niah_single", "ctx": 16384,
            "seed": 0, "trial": t, "hit": 1, "frac": 1.0, "generator": "inhouse",
            "haystack_id": None, "depth": None, "code_family": None,
            "prompt_sha256": None, "error": None, "source": "x",
        }
        for t in (0, 1)
    ]  # fmt: skip
    (tmp_path / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = _run("check", str(tmp_path))
    assert r.returncode == 1 and "CHECK FAIL cells" in r.stdout + r.stderr


# A whole [pplw] line, a 3-part split one (>400 chars splits into part=i/N groups of
# 8 -- vast.ai truncates a log line at ~500), and the aggregate ppl line, which is a
# different artifact and is parsed alongside, never instead.
LOG = """===RUN_SHA_deadbeef===
===ENV_BEGIN===
run_sha=deadbeef
===ENV_END===
[trial] task=niah_single ctx=16384 arm=bugSseed-r64-h256 seed=0 trial=0 hit=1 frac=1.000
[trial] task=niah_single ctx=16384 arm=bugSseed-r64-h256 seed=0 trial=1 hit=0 frac=0.000
[pplw] T=16384 bugSseed-r64-h256 ntok=511 nlls=1.573386,1.236791
[pplw] T=32768 quant-2bit-kivi ntok=255 part=1/3 nlls=1.000000,2.000000
[pplw] T=32768 quant-2bit-kivi ntok=255 part=2/3 nlls=3.000000,4.000000
[pplw] T=32768 quant-2bit-kivi ntok=255 part=3/3 nlls=5.000000
  bugSseed-r64-h256 [T=16384] ppl=5.403 tok_eq/layer=1398.0 ratio=0.163 sbits=0.163
[diag] {"layer": 0, "rank": 64}
===ALL_DONE_w18_g1_deadbeef===
"""


def _rows(d: Path, name: str) -> list[dict[str, object]]:
    return [json.loads(x) for x in (d / name).read_text().splitlines()]


def test_harvest_parses_a_log_into_records(dry_pod: Path, tmp_path: Path) -> None:
    """Two schemas, two files: per-window rows in `pplw.jsonl`, the aggregate
    `PplRecord` rows in `ppl.jsonl`. A split [pplw] group reassembles in part order."""
    tmp_path = _copy(dry_pod, tmp_path)
    log = tmp_path / "pod.log"
    log.write_text(LOG)
    r = _run("harvest", "--pod", "w18_g1", "--log", str(log), "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    trials = _rows(tmp_path, "trials.jsonl")
    assert [t["hit"] for t in trials] == [1, 0]
    assert trials[0]["model"] == load_pod("w18_g1").model
    # No `generator=` in the line (no v1 log has one); `niah_single` belongs to exactly
    # one generator in this pod's tasks, so the harvest fills it from the config.
    assert {t["generator"] for t in trials} == {"inhouse"}

    pplw = _rows(tmp_path, "pplw.jsonl")
    assert [(w["arm"], w["ctx"], w["window_idx"]) for w in pplw] == [
        ("bugSseed-r64-h256", 16384, 0), ("bugSseed-r64-h256", 16384, 1),
        *[("quant-2bit-kivi", 32768, i) for i in range(5)],
    ]  # fmt: skip
    assert pplw[0] == {
        "model": load_pod("w18_g1").model, "arm": "bugSseed-r64-h256", "ctx": 16384,
        "window_idx": 0, "ntok": 511, "nll_sum_nats": 1.573386 * 511,
        "corpus": None,  # an archived-format [pplw] line: no corpus= field to recover
        "source": f"{log}:7",
    }  # fmt: skip
    assert [w["nll_sum_nats"] for w in pplw[2:]] == [v * 255 for v in (1.0, 2.0, 3.0, 4.0, 5.0)]

    # The aggregate line is a DIFFERENT artifact, parsed alongside -- never instead.
    (ppl,) = _rows(tmp_path, "ppl.jsonl")
    assert ppl["ppl"] == 5.403 and ppl["sbits"] == 0.163 and ppl["ctx"] == 16384

    assert _rows(tmp_path, "diag.jsonl") == [
        {"model": load_pod("w18_g1").model, "layer": 0, "rank": 64, "source": f"{log}:12"}
    ]
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["records"] == {
        "trials.jsonl": 2, "pplw.jsonl": 7, "ppl.jsonl": 1, "diag.jsonl": 1,
    }  # fmt: skip
    assert m["errors"] == 0 and m["harvested_at"] and m["status"] == "ALL_DONE"


def test_harvest_parses_the_aggregate_ppl_lines_with_no_pplw(dry_pod: Path, tmp_path: Path) -> None:
    tmp_path = _copy(dry_pod, tmp_path)
    log = tmp_path / "pod.log"
    log.write_text("\n".join(x for x in LOG.splitlines() if not x.startswith("[pplw]")))
    _run("harvest", "--pod", "w18_g1", "--log", str(log), "--out", str(tmp_path))
    ppl = _rows(tmp_path, "ppl.jsonl")
    assert len(ppl) == 1 and ppl[0]["ppl"] == 5.403 and ppl[0]["sbits"] == 0.163
    assert not (tmp_path / "pplw.jsonl").exists()


def test_harvest_fails_loud_on_an_incomplete_pplw_part_set(dry_pod: Path, tmp_path: Path) -> None:
    tmp_path = _copy(dry_pod, tmp_path)
    log = tmp_path / "pod.log"
    log.write_text("\n".join(x for x in LOG.splitlines() if "part=2/3" not in x))
    r = _run("harvest", "--pod", "w18_g1", "--log", str(log), "--out", str(tmp_path))
    assert r.returncode != 0 and "part" in r.stdout + r.stderr


def test_harvest_records_a_failed_run(dry_pod: Path, tmp_path: Path) -> None:
    """boot.sh prints ===RUN_FAILED_ instead of ===ALL_DONE_ when `run` exits non-zero;
    the watchdog still destroys the instance, so the failure has to be visible here."""
    tmp_path = _copy(dry_pod, tmp_path)
    log = tmp_path / "pod.log"
    log.write_text(LOG.replace("===ALL_DONE_w18_g1", "===RUN_FAILED_w18_g1"))
    _run("harvest", "--pod", "w18_g1", "--log", str(log), "--out", str(tmp_path))
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["status"] == "RUN_FAILED"
    assert m["timeout"] is False  # a plain RUN_FAILED with no RUN_TIMEOUT marker (e.g. a 137 KILL)


def test_harvest_records_a_timeout_as_a_failed_run(dry_pod: Path, tmp_path: Path) -> None:
    """A run that reaches `--max-hours` prints ===RUN_TIMEOUT_ and then the same
    ===RUN_FAILED_ as any failed run: the status stays RUN_FAILED (the watchdog and
    `check` know one failure kind) and the reason survives as `timeout: true`."""
    tmp_path = _copy(dry_pod, tmp_path)
    log = tmp_path / "pod.log"
    log.write_text(
        LOG.replace(
            "===ALL_DONE_w18_g1_deadbeef===",
            "===RUN_TIMEOUT_w18_g1_18.3h===\n===RUN_FAILED_w18_g1_deadbeef===",
        )
    )
    _run("harvest", "--pod", "w18_g1", "--log", str(log), "--out", str(tmp_path))
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert (m["status"], m["timeout"]) == ("RUN_FAILED", True)


def test_harvest_refuses_to_shrink_an_existing_trials_file(dry_pod: Path, tmp_path: Path) -> None:
    """A short log -- the 5000-line fallback, a truncated fetch -- must not overwrite a
    good harvest. The files on disk are left exactly as they were."""
    tmp_path = _copy(dry_pod, tmp_path)
    log = tmp_path / "pod.log"
    log.write_text(LOG)
    first = _run("harvest", "--pod", "w18_g1", "--log", str(log), "--out", str(tmp_path))
    assert first.returncode == 0, first.stderr
    before = {p.name: p.read_text() for p in tmp_path.glob("*.jsonl")}

    short = tmp_path / "short.log"
    short.write_text("\n".join(LOG.splitlines()[:5]))  # one [trial] line, no pplw/ppl/diag
    r = _run("harvest", "--pod", "w18_g1", "--log", str(short), "--out", str(tmp_path))
    assert r.returncode == 1
    assert "REFUSE: trials.jsonl would shrink from 2 to 1 rows" in r.stdout + r.stderr
    assert {p.name: p.read_text() for p in tmp_path.glob("*.jsonl")} == before

    forced = _run(
        "harvest", "--pod", "w18_g1", "--log", str(short), "--out", str(tmp_path), "--force"
    )
    assert forced.returncode == 0, forced.stderr
    assert len(_rows(tmp_path, "trials.jsonl")) == 1


def test_harvest_writes_nothing_when_a_pplw_part_set_is_incomplete(
    dry_pod: Path, tmp_path: Path
) -> None:
    """Everything is parsed before anything is written: the SystemExit an incomplete
    [pplw] group raises used to land AFTER trials.jsonl had already been replaced."""
    tmp_path = _copy(dry_pod, tmp_path)
    good, bad = tmp_path / "good.log", tmp_path / "bad.log"
    good.write_text(LOG)
    _run("harvest", "--pod", "w18_g1", "--log", str(good), "--out", str(tmp_path))
    before = {p.name: p.read_text() for p in tmp_path.glob("*.jsonl")}

    bad.write_text("\n".join(x for x in LOG.splitlines() if "part=2/3" not in x))
    r = _run("harvest", "--pod", "w18_g1", "--log", str(bad), "--out", str(tmp_path))
    assert r.returncode != 0 and "part" in r.stdout + r.stderr
    assert {p.name: p.read_text() for p in tmp_path.glob("*.jsonl")} == before


def test_pods_line_is_unique_per_instance() -> None:
    """The watchdog remembers harvested LABELS in done.txt. A label of just the pod name
    made a relaunched pod "done" before it started -- never harvested, never destroyed,
    billing until the credit floor. The label carries the instance id."""
    a = pod.pods_line("w18_g1", "111", "unsloth/Meta-Llama-3.1-8B-Instruct")
    b = pod.pods_line("w18_g1", "222", "unsloth/Meta-Llama-3.1-8B-Instruct")
    assert a.split(":")[0] != b.split(":")[0]
    assert a == "w18_g1-111:111:w18_g1:Meta-Llama-3.1-8B-Instruct\n"
    # <label>:<id>:<pod>:<tag> -- field 3 is what the watchdog matches ===ALL_DONE_ on.
    assert a.rstrip("\n").split(":")[2] == "w18_g1"
    assert pod.pod_name("w18_g1-111") == "w18_g1" and pod.pod_name("w18_g1") == "w18_g1"


def test_launch_refuses_a_sha_that_is_on_no_remote_branch(tmp_path: Path) -> None:
    """The pod clones from GitHub, so an unpushed SHA boots into a CHECKOUT_FAILED and
    bills for the privilege. A repo with no remotes is the strongest form of that."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for cmd in (
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "commit", "-q", "--allow-empty", "-m", "x"],
    ):
        subprocess.run(cmd, cwd=tmp_path, check=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True
    ).stdout.strip()
    err = pod.unpushed_error(sha, tmp_path)
    assert err is not None and err.startswith(f"HEAD {sha} is not on any remote branch")
    assert "push first" in err


def test_launch_refuses_a_pod_with_no_prereg() -> None:
    """Every v1 pod config records `prereg: null`; a re-run must fill it before launch.
    The refusal is checked before `vastai` is reached, so `--dry-run` proves it."""
    r = _run("launch", "--pod", "w18_g1", "--offer", "12345678", "--dry-run")
    assert r.returncode == 1 and "prereg" in r.stdout + r.stderr
    # ...and its `gpu_budget_h: 0.0` is the second refusal: no bar for boot.sh to enforce.
    assert "max-hours" in r.stdout + r.stderr


def test_prereg_refusal_reasons(tmp_path: Path) -> None:
    """Missing and uncommitted both refuse; the message names which."""
    missing = tmp_path / "nope.md"
    assert "missing" in (pod.prereg_error(missing, "HEAD") or "")
    uncommitted = tmp_path / "w99.md"
    uncommitted.write_text("# pre-registration\n")
    assert "uncommitted" in (pod.prereg_error(uncommitted, "HEAD") or "")


def test_launch_dry_run_prints_the_vastai_command(tmp_path: Path) -> None:
    """The command is the one the boot-script header documents: image from the pod
    YAML, --disk 80, the POD/SHA/MODEL/DTYPE/MAX_HOURS env, boot.sh as --onstart."""
    cmd = pod.launch_command("w18_g1", "12345678", "deadbeef", max_hours=2.5)
    assert cmd[:4] == ["vastai", "create", "instance", "12345678"]
    joined = " ".join(cmd)
    assert "--disk 80" in joined and "--onstart scripts/pod/boot.sh" in joined
    assert "--label kvdlra-w18_g1" in joined
    assert "-e POD=w18_g1 -e SHA=deadbeef" in joined
    assert "-e DTYPE=bfloat16 -e MAX_HOURS=2.5" in joined  # inside the one --env string
    assert load_pod("w18_g1").image in joined


def test_launch_max_hours_defaults_to_the_pod_budget() -> None:
    """The bar boot.sh enforces on the pod (`timeout`) is the pre-registered one unless
    the launch says otherwise. A pod with no budget -- every v1 pod -- has no bar to
    enforce and is refused rather than launched open-ended: `timeout 0h` DISABLES the
    limit, so a zero can never reach the command."""
    joined = " ".join(pod.launch_command("filler_realism", "1", "deadbeef"))
    hours = float(joined.split("MAX_HOURS=")[1].split()[0])
    assert hours == load_pod("filler_realism").gpu_budget_h == 18.3
    with pytest.raises(ValueError, match="max-hours"):
        pod.launch_command("w18_g1", "1", "deadbeef")  # gpu_budget_h: 0.0
    # nan <= 0 is False and +inf > 0 is True: each needs its own check (isfinite AND > 0)
    for bad in (0.0, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="max-hours"):
            pod.launch_command("filler_realism", "1", "deadbeef", max_hours=bad)


# w18_g1's expected cell set, spelled out rather than re-derived: three arms by their
# `legacy_name` (the string a record carries), the four in-house RULER sub-tasks, two
# context lengths, 6 trials x 2 seeds each. The two ppl tasks contribute no cells.
#
# The rows go in as the LOG a pod prints and come back through `harvest`, never as
# hand-written JSON: hand-writing `"generator": "inhouse"` is what hid the bug where the
# real parser left it null and `check` then rejected a complete, correct run.
ARMS = ("quant-2bit-kivi", "quant-4bit-kivi", "bugSseed-r64-h256")
SUBTASKS = ("niah_single", "niah_multikey", "niah_multivalue", "vt")
CTXS = (16384, 32768)


def _trial_lines(*arms: str, generator: str = "") -> list[str]:
    """`kvdlra.eval.runner`'s `[trial]` line, verbatim. `generator=""` is the v1 format,
    which every archived pod log is in."""
    gen = f" generator={generator}" if generator else ""
    return [
        f"[trial] task={task} ctx={ctx} arm={arm} seed={seed} trial={t} hit=1 frac=1.000{gen}"
        for arm in arms
        for ctx in CTXS
        for task in SUBTASKS
        for seed in (0, 1)
        for t in range(6)
    ]


def _ppl_lines(*arms: str) -> list[str]:
    """w18_g1's two `ppl` tasks: one `ppl=` line per (arm, ctx)."""
    return [
        f"  {arm:14s} [T={ctx}] ppl=5.403 tok_eq/layer=1398.0 ratio=0.163 sbits=0.163"
        for arm in arms
        for ctx in CTXS
    ]


def _pplw_lines(*arms: str) -> list[str]:
    """The per-window rows behind those sweeps: `n_samples` (4) windows per (arm, ctx).
    A sweep leaves both artifacts, and `check` requires both."""
    return [
        f"[pplw] T={ctx} {arm} ntok=511 nlls=1.000000,2.000000,3.000000,4.000000"
        for arm in arms
        for ctx in CTXS
    ]


def _harvested(dry_pod: Path, tmp_path: Path, lines: list[str]) -> Path:
    """A dry-run directory plus a harvest of the log those lines make -- the whole path a
    real pod takes, so anything the parser does not recover is missing here too."""
    d = _copy(dry_pod, tmp_path)
    m = json.loads((d / "manifest.json").read_text())
    m["dry_run"] = False
    (d / "manifest.json").write_text(json.dumps(m))
    log = d / "pod.log"
    log.write_text("\n".join(lines) + "\n")
    r = _run("harvest", "--pod", "w18_g1", "--log", str(log), "--out", str(d))
    assert r.returncode == 0, r.stdout + r.stderr
    return d


def test_check_fails_a_non_dry_run_harvest_with_no_records(dry_pod: Path, tmp_path: Path) -> None:
    """The rule that only counted the cells PRESENT passed a pod that produced nothing.
    Every configured cell is expected; a cell with no rows is the loudest failure there
    is."""
    r = _run("check", str(_harvested(dry_pod, tmp_path, [])))
    assert r.returncode == 1
    out = r.stdout + r.stderr
    assert out.count("CHECK FAIL cells") == len(ARMS) * len(SUBTASKS) * len(CTXS) == 24
    assert "CHECK FAIL cells: bugSseed-r64-h256 inhouse/vt ctx=32768 has 0 of 12 records" in out


def test_check_fails_when_only_one_arm_reported(dry_pod: Path, tmp_path: Path) -> None:
    """One arm of three at full n: 8 complete cells, 16 empty ones."""
    r = _run("check", str(_harvested(dry_pod, tmp_path, _trial_lines("bugSseed-r64-h256"))))
    assert r.returncode == 1
    out = r.stdout + r.stderr
    cells = [x for x in out.splitlines() if x.startswith("CHECK FAIL cells")]
    assert len(cells) == 16
    assert not [x for x in cells if "bugSseed-r64-h256" in x]  # the arm that ran is not flagged


def test_a_complete_harvest_of_a_v1_format_log_passes_check(dry_pod: Path, tmp_path: Path) -> None:
    """The `[trial]` line carries no `generator=` in any v1 log, and a cell's identity
    includes the generator -- so a harvest that left it null keyed every cell "None" and
    `check` rejected a complete, correct run (8 CHECK FAIL cells per task). `harvest`
    fills it from the pod's own task configs."""
    lines = [*_trial_lines(*ARMS), *_ppl_lines(*ARMS), *_pplw_lines(*ARMS)]
    d = _harvested(dry_pod, tmp_path, lines)
    assert {r["generator"] for r in _rows(d, "trials.jsonl")} == {"inhouse"}
    r = _run("check", str(d))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK (288 trials, 0 errors)" in r.stdout


def test_a_harvest_keeps_the_generator_the_row_itself_names(dry_pod: Path, tmp_path: Path) -> None:
    """The runner prints it, so no inference is needed -- and the pod-config map is not
    consulted at all for a row that says which generator built it."""
    lines = [*_trial_lines(*ARMS, generator="inhouse"), *_ppl_lines(*ARMS), *_pplw_lines(*ARMS)]
    d = _harvested(dry_pod, tmp_path, lines)
    assert {r["generator"] for r in _rows(d, "trials.jsonl")} == {"inhouse"}
    assert _run("check", str(d)).returncode == 0


def test_harvest_refuses_to_guess_a_generator_two_tasks_share(tmp_path: Path) -> None:
    """w19_fork runs the in-house AND the official 16K task, which give `niah_multivalue`
    and `vt` the same names at the same context. A v1 row could be either, and guessing
    would pool two benchmarks into one cell -- so the harvest stops instead."""
    log = tmp_path / "pod.log"
    row = "[trial] task=vt ctx=16384 arm=ea-k0.1-q2-kivi seed=0 trial=0 hit=1 frac=1.000"
    log.write_text(row + "\n")
    r = _run("harvest", "--pod", "w19_fork", "--log", str(log), "--out", str(tmp_path))
    assert r.returncode != 0
    assert "sub-task vt belongs to more than one generator" in r.stdout + r.stderr

    log.write_text(row + " generator=official_ruler\n")
    r = _run("harvest", "--pod", "w19_fork", "--log", str(log), "--out", str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert _rows(tmp_path, "trials.jsonl")[0]["generator"] == "official_ruler"


def test_harvest_counts_the_ppl_axis_error_lines(dry_pod: Path, tmp_path: Path) -> None:
    """A perplexity arm that raises produces no record at all, only an `[error]` line, so
    a harvest counting trial rows alone wrote `errors: 0` for a pod whose sweep died.
    The run path counts both axes; the log path has to agree."""
    d = _harvested(
        dry_pod,
        tmp_path,
        [
            *_trial_lines(*ARMS),
            *_ppl_lines("quant-2bit-kivi", "quant-4bit-kivi"),
            "[error] axis=ppl arm=bugSseed-r64-h256 ctx=16384 error=RuntimeError: boom",
        ],
    )
    assert json.loads((d / "manifest.json").read_text())["errors"] == 1
    r = _run("check", str(d))
    assert r.returncode == 1 and "CHECK FAIL errors" in r.stdout + r.stderr


def test_check_fails_any_pod_that_recorded_a_trial_error(dry_pod: Path, tmp_path: Path) -> None:
    """Ruling R29: a complete pod whose every trial RAISED used to pass -- the cells all
    held their full n, and the errors were only a number in the manifest. No tolerance
    knob: one recorded failure is a pod to fix or to re-run, not one to cite."""
    lines = [x.replace("hit=1 frac=1.000", "hit=0 frac=0.000 error=RuntimeError: boom")
             for x in _trial_lines(*ARMS)]  # fmt: skip
    d = _harvested(dry_pod, tmp_path, [*lines, *_ppl_lines(*ARMS)])
    assert json.loads((d / "manifest.json").read_text())["errors"] == 288
    r = _run("check", str(d))
    out = r.stdout + r.stderr
    assert r.returncode == 1
    assert "CHECK FAIL errors: 288 trial(s) raised (see the error field in trials.jsonl)" in out
    assert "CHECK FAIL cells" not in out  # every cell is full; the errors are the failure


def test_check_fails_a_ppl_task_that_produced_no_perplexity_record(
    dry_pod: Path, tmp_path: Path
) -> None:
    """A `ppl` task contributes no Bernoulli cells, so the cell rule never looked at it
    and a pod whose whole perplexity sweep died still passed. One record per (arm, ctx)
    is expected in ppl.jsonl -- w18_g1 names two ppl tasks, so 3 arms x 2 ctx = 6. The
    per-window rows are the sweep's other artifact and fail on their own count."""
    r = _run("check", str(_harvested(dry_pod, tmp_path, _trial_lines(*ARMS))))
    assert r.returncode == 1
    out = r.stdout + r.stderr
    assert out.count("CHECK FAIL ppl:") == len(ARMS) * len(CTXS) == 6
    assert out.count("CHECK FAIL pplw:") == 6
    assert "CHECK FAIL ppl: bugSseed-r64-h256 ctx=32768 has no perplexity record" in out
    assert "CHECK FAIL pplw: bugSseed-r64-h256 ctx=32768 has 0 of 4 window rows" in out


def test_check_fails_when_one_arm_is_missing_from_the_ppl_sweep(
    dry_pod: Path, tmp_path: Path
) -> None:
    lines = [
        *_trial_lines(*ARMS),
        *_ppl_lines("quant-2bit-kivi", "quant-4bit-kivi"),
        *_pplw_lines("quant-2bit-kivi", "quant-4bit-kivi"),
    ]
    r = _run("check", str(_harvested(dry_pod, tmp_path, lines)))
    assert r.returncode == 1
    assert r.stdout.count("CHECK FAIL ppl:") == 2  # the r64 arm's two contexts
    assert r.stdout.count("CHECK FAIL pplw:") == 2  # and its two window sets


def test_check_rejects_an_unresolvable_git_sha(dry_pod: Path, tmp_path: Path) -> None:
    d = _copy(dry_pod, tmp_path)
    m = json.loads((d / "manifest.json").read_text())
    m["git_sha"] = "0" * 40
    (d / "manifest.json").write_text(json.dumps(m))
    r = _run("check", str(d))
    assert r.returncode == 1 and "CHECK FAIL git_sha" in r.stdout + r.stderr


def test_check_rejects_an_env_that_drifted_from_the_pyproject_pins(
    dry_pod: Path, tmp_path: Path
) -> None:
    d = _copy(dry_pod, tmp_path)
    lines = (d / "env.txt").read_text().splitlines()
    (d / "env.txt").write_text(
        "\n".join("torch==0.0.0" if x.startswith("torch==") else x for x in lines) + "\n"
    )
    r = _run("check", str(d))
    assert r.returncode == 1 and "CHECK FAIL env: torch==0.0.0" in r.stdout + r.stderr


# --- L1.8: env.txt survives a log harvest -------------------------------------
#
# `run` writes results/<pod>/env.txt ON THE POD. A log harvest never brings that file
# back, so every harvested pod failed `check` on `env: env.txt is missing` -- the three
# Table-4 pods among them. boot.sh prints the same set into the log between
# `===ENV_BEGIN===` and `===ENV_END===`, and that is where the harvest reads it from.
ENV_BLOCK = [
    "===ENV_BEGIN===",
    "run_sha=deadbeef",
    "NVIDIA A100-SXM4-40GB, 40960 MiB, 595.84",
    "python=3.12.3",
    "torch=2.11.0+cu128 cuda_build=12.8 cuda_avail=True",
    "transformers=5.8.0 kvpress=0.5.1 optimum-quanto=0.2.7 hqq=0.2.8.post1",
    "device=NVIDIA A100-SXM4-40GB",
    "===ENV_END===",
]
# The set `env_lines()` writes on the pod, recovered from the block above. `cuda` is
# torch's build (`cuda_build=`), `gpu` is not a version at all, and a package the block
# does not print is `unrecorded` -- never guessed from the laptop running the harvest.
HARVESTED_ENV = [
    "torch==2.11.0+cu128",
    "cuda==12.8",
    "triton==unrecorded",
    "transformers==5.8.0",
    "kvpress==0.5.1",
    "optimum-quanto==0.2.7",
    "hqq==0.2.8.post1",
    "omegaconf==unrecorded",
    "gpu=NVIDIA A100-SXM4-40GB",
]


def test_harvest_writes_env_txt_from_the_logs_env_block(dry_pod: Path, tmp_path: Path) -> None:
    """A pod-written env.txt is left exactly as it is; with none there -- which is every
    harvested pod -- the log's own ENV block becomes one, and `check` passes on it."""
    lines = [*ENV_BLOCK, *_trial_lines(*ARMS), *_ppl_lines(*ARMS), *_pplw_lines(*ARMS)]
    d = _harvested(dry_pod, tmp_path, lines)
    on_pod = (d / "env.txt").read_text()
    assert on_pod != "\n".join(HARVESTED_ENV) + "\n"  # the run-side file, not the log's

    (d / "env.txt").unlink()  # what a harvest of a finished pod actually starts from
    r = _run("harvest", "--pod", "w18_g1", "--log", str(d / "pod.log"), "--out", str(d))
    assert r.returncode == 0, r.stdout + r.stderr
    assert (d / "env.txt").read_text().splitlines() == HARVESTED_ENV
    c = _run("check", str(d))
    assert c.returncode == 0, c.stdout + c.stderr


def test_harvest_writes_no_env_txt_when_the_log_has_no_env_block(
    dry_pod: Path, tmp_path: Path
) -> None:
    """No block, no file. A harvest that invented an env.txt from the laptop it runs on
    would record an environment no pod ever had; the missing file is the signal."""
    lines = [*_trial_lines(*ARMS), *_ppl_lines(*ARMS), *_pplw_lines(*ARMS)]
    d = _harvested(dry_pod, tmp_path, lines)
    (d / "env.txt").unlink()
    r = _run("harvest", "--pod", "w18_g1", "--log", str(d / "pod.log"), "--out", str(d))
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (d / "env.txt").exists()
    c = _run("check", str(d))
    assert c.returncode == 1 and "CHECK FAIL env: env.txt is missing" in c.stdout + c.stderr


def _env_txt(d: Path, torch_line: str) -> Path:
    (d / "env.txt").write_text("\n".join([torch_line, *HARVESTED_ENV[1:]]) + "\n")
    return d


def test_env_accepts_the_cuda_build_suffix_on_a_pinned_version(tmp_path: Path) -> None:
    """A pod reports torch as a PEP 440 LOCAL version (`2.11.0+cu128`); the pyproject pin
    is the public one, and the CUDA build is recorded separately as `cuda`. `unrecorded`
    is the same evidence gap as an absent line, and fails like one."""
    assert pod._env_fails(_env_txt(tmp_path, "torch==2.11.0+cu128")) == []
    (wrong,) = pod._env_fails(_env_txt(tmp_path, "torch==2.10.0+cu128"))
    assert wrong.startswith("env: torch==2.10.0+cu128 != the pyproject pin")
    assert pod._env_fails(_env_txt(tmp_path, "torch==unrecorded")) == [
        "env: torch is not recorded in env.txt"
    ]


def test_the_watchdog_keeps_the_env_block_rows() -> None:
    """The watchdog greps each log fetch through `ROWS` before appending it, so a row
    kind the filter drops never reaches the deduped `<label>.log` the harvest parses --
    which is why the ENV block came back empty. The pattern is read from the script:
    a copy of it here would pass while the script kept dropping the lines."""
    rows = next(
        x[len("ROWS='") : -1]
        for x in (REPO_ROOT / "scripts/pod/watchdog.sh").read_text().splitlines()
        if x.startswith("ROWS=")
    )
    kept = [
        *ENV_BLOCK,
        "triton=3.5.0 omegaconf=2.3.0 datasets=2.21.0 numpy=2.1.3 scipy=1.14.1",
        "===RUN_TIMEOUT_w18_g1_18.3h===",  # L2.1: boot.sh's budget and self-destruct
        "===SELF_DESTRUCT_FAILED_w18_g1===",  # markers reach the harvested log
        "[stage] load_model unsloth/Meta-Llama-3.1-8B-Instruct (61.3 s)",  # L2.3b timings
        f"[stage] dataset_sha256 haystack:pg19 {'a' * 64}",  # L2.9a: the digests' only way back
        "[stage] cell arm=full task=vt ctx=16384 elapsed_s=41.5 n=12",  # L3.1c: the per-cell clock
    ]
    r = subprocess.run(
        ["grep", "-aE", rows],
        input="\n".join([*kept, "+ echo run_sha=deadbeef", "some other log noise"]) + "\n",
        capture_output=True,
        text=True,
    )
    assert r.stdout.splitlines() == kept


def test_the_watchdog_harvests_under_the_venv_and_expires_at_the_pods_bar() -> None:
    """Three lines of the script, run as bash runs them (L2.9a): the harvest's interpreter
    is the repo's `.venv/bin/python` when there is one (the cycle pod's harvest died on a
    bare `python`), and the expiry is the pod's own `gpu_budget_h` plus boot.sh's 2 h
    grace in 150 s polls -- a flat 600 (25 h) would have destroyed the healthy 168 h smoke
    pod -- never under 600, and still whatever the environment says. The per-poll `sort -u`
    keeps `<label>.raw` from re-growing by the saturated 30,000-line tail every 150 s."""
    text = (REPO_ROOT / "scripts/pod/watchdog.sh").read_text()
    py, budget = (
        next(x for x in text.splitlines() if x.startswith(k)) for k in ("PY=", "BUDGET_ITERS=")
    )

    def sh(script: str, cwd: Path = REPO_ROOT) -> str:
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=cwd)
        assert r.returncode == 0, r.stderr
        return r.stdout.strip()

    def iters(pod: str, env: str = "") -> int:
        return int(sh(f'{env}POD={pod}; {budget}; echo "$BUDGET_ITERS"'))

    assert sh(f'{py}; echo "$PY"') == ".venv/bin/python"
    assert sh(f'{py}; echo "$PY"', cwd=REPO_ROOT / "tests") == "python3"
    assert iters("l2_smoke") == (168 * 3600 + 7200) // 150 + 1 == 4081
    assert iters("filler_realism") == iters("w18_g1") == iters("no_such_pod") == 600
    assert iters("l2_smoke", env="BUDGET_ITERS=7 ") == 7
    assert "$PY scripts/pod.py harvest" in text and "python scripts/pod.py" not in text
    assert 'sort -u "$H/${lab}.raw" -o "$H/${lab}.raw"' in text


def test_harvest_records_the_dataset_digests_the_run_printed(dry_pod: Path, tmp_path: Path) -> None:
    """`run` writes the haystack and corpus digests into the manifest ON THE POD, which
    dies with the instance -- every harvested manifest carried `dataset_sha256: {}`. They
    travel as `[stage] dataset_sha256 <key> <sha>` lines (a row kind the watchdog keeps)
    and the harvest writes them back; two tasks on one corpus print it twice, last wins."""
    d = _copy(dry_pod, tmp_path)
    log = d / "pod.log"
    log.write_text(
        LOG
        + f"[stage] dataset_sha256 haystack:pg19 {'a' * 64}\n"
        + f"[stage] dataset_sha256 pg19val {'b' * 64}\n"
        + f"[stage] dataset_sha256 pg19val {'c' * 64}\n"
    )
    assert json.loads((d / "manifest.json").read_text())["dataset_sha256"] == {}
    assert pod.harvest("w18_g1", log, d, force=False) == 0
    m = json.loads((d / "manifest.json").read_text())
    assert m["dataset_sha256"] == {"haystack:pg19": "a" * 64, "pg19val": "c" * 64}


def test_harvest_records_the_cell_timings_the_run_printed(dry_pod: Path, tmp_path: Path) -> None:
    """The per-arm min/sample a pre-flight pod is read for (`prereg/gate1_preflight.md`
    reading (iv)) has no other source: `[trial]` and cell rows carry no clock,
    `wall_clock_s` is null in every harvested manifest, and the watchdog `sort -u`s
    `<label>.raw` in place every poll, so arrival order is gone. `runner._cell` prints the
    seconds into the line itself; the last line for a key wins, and a log without them
    leaves the key absent rather than empty."""
    d = _copy(dry_pod, tmp_path)
    log = d / "pod.log"
    log.write_text(LOG)
    assert pod.harvest("w18_g1", log, d, force=False) == 0
    assert "cell_elapsed_s" not in json.loads((d / "manifest.json").read_text())
    log.write_text(
        LOG
        + "[stage] cell arm=full task=niah_single ctx=16384 elapsed_s=41.5 n=12\n"
        + "[stage] cell arm=bugSseed-r64-h256 task=vt ctx=16384 elapsed_s=180.0 n=12\n"
        + "[stage] cell arm=bugSseed-r64-h256 task=vt ctx=16384 elapsed_s=186.3 n=12\n"
    )
    assert pod.harvest("w18_g1", log, d, force=False) == 0
    m = json.loads((d / "manifest.json").read_text())
    assert m["cell_elapsed_s"] == {
        "full/niah_single/16384": 41.5,
        "bugSseed-r64-h256/vt/16384": 186.3,
    }


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT)
    return out.stdout.strip()


def _first_commit(path: str) -> str:
    return _git("log", "--reverse", "--format=%H", "--", path).split("\n")[0].strip()


def test_prereg_commit_order_against_real_history() -> None:
    """The three branches of the order rule, on commits this repo actually has.

    `Makefile` was committed in L0.3, well before HEAD -- a strict ancestor, which is
    the only shape that passes. `prereg/README.md` was created by L0.5c, so checking it
    against that very commit is the "written with the results in hand" case. And a
    commit `prereg/README.md`'s own commit does not descend to (L0.3, which predates it)
    is the not-an-ancestor case."""
    if _git("rev-parse", "--is-shallow-repository") == "true":
        pytest.skip("shallow clone: the commit-order rule needs the history to walk")
    makefile_first = _first_commit("Makefile")
    readme, readme_first = REPO_ROOT / "prereg" / "README.md", _first_commit("prereg/README.md")
    assert pod.prereg_error(REPO_ROOT / "Makefile", "HEAD") is None
    assert "itself" in (pod.prereg_error(readme, readme_first) or "")
    assert "not an ancestor" in (pod.prereg_error(readme, makefile_first) or "")


def test_check_fails_a_sweep_whose_per_window_rows_came_back_short(
    dry_pod: Path, tmp_path: Path
) -> None:
    """One sweep, two artifacts: the aggregate `ppl.jsonl` number can be there in full
    while `pplw.jsonl` -- the only thing that can re-pool it or carry an interval -- is a
    window short, which `_ppl_fails` alone never looked at."""
    lines = [*_trial_lines(*ARMS), *_ppl_lines(*ARMS), *_pplw_lines(*ARMS)]
    lines[-1] = lines[-1].replace(",4.000000", "")  # 3 windows for bugSseed... at 32K
    r = _run("check", str(_harvested(dry_pod, tmp_path, lines)))
    assert r.returncode == 1
    out = r.stdout + r.stderr
    assert "CHECK FAIL pplw: bugSseed-r64-h256 ctx=32768 has 3 of 4 window rows" in out
    assert [x for x in out.splitlines() if x.startswith("CHECK FAIL pplw")] == [
        "CHECK FAIL pplw: bugSseed-r64-h256 ctx=32768 has 3 of 4 window rows"
    ]


def test_harvest_counts_a_diag_line_the_fetch_cut_in_half(dry_pod: Path, tmp_path: Path) -> None:
    """A truncated `[diag]` payload cannot become a record. Skipping it silently made a
    half-fetched log look like a pod that printed no diagnostics, so the skip is counted
    into the manifest, printed by the harvest, and failed by `check`."""
    tmp_path = _copy(dry_pod, tmp_path)
    log = tmp_path / "pod.log"
    log.write_text(LOG.replace("[diag] {", '[diag] {"layer": 1, "ra\n[diag] {', 1))
    r = _run("harvest", "--pod", "w18_g1", "--log", str(log), "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "harvest: 1 [diag] line(s) skipped" in r.stdout
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["diag_skipped"] == 1 and m["records"]["diag.jsonl"] == 1  # the good one still lands
    c = _run("check", str(tmp_path))
    assert c.returncode == 1
    assert "CHECK FAIL diag: 1 [diag] line(s) were unparseable" in c.stdout + c.stderr


# --- L1.6: the Table-4 pods (prereg/hygiene_table4.md) ------------------------

TABLE4 = ("hygiene_table4_qwen_r128", "hygiene_table4_qwen_r256", "hygiene_table4_llama")


def test_the_table4_pods_resolve_end_to_end() -> None:
    """All three pods load, hash, name the shared prereg, and every arm and task they
    reference loads. In-process on purpose: `run --dry-run` spawns a torch-importing
    subprocess, and this asserts the same resolution for a tenth of the wall clock.

    The two Qwen halves share a model, so all that separates their hashes is the arm
    list Amendment 1 split them on. Each pod also carries `full` and exactly the one
    re-sized perplexity task: every paired statistic is computed inside a single pod,
    and a half that lost its uncompressed reference -- or drifted off the 16 windows the
    amendment pre-registered -- could not produce the numbers the decision rule reads."""
    hashes = set()
    for name in TABLE4:
        p = load_pod(name)
        assert p.prereg == "prereg/hygiene_table4.md"
        assert (REPO_ROOT / p.prereg).is_file(), "the prereg must be in the launch's ancestry"
        assert p.gpu_budget_h > 0, f"{name}: a pod to be launched needs a pre-registered budget"
        for a in p.arms:
            assert load_arm(a).name == a
        for t in p.tasks:
            assert load_task(t).name == t
        assert "full" in p.arms, f"{name}: no uncompressed reference to pair against"
        ppl = [t for t in p.tasks if load_task(t).generator == "ppl"]
        assert ppl == ["ppl_16k_pg19val_w16"], f"{name}: perplexity tasks {ppl}"
        assert load_task(ppl[0]).n_samples == 16, f"{name}: not the pre-registered n"
        hashes.add(config_hash(p))
    assert len(hashes) == len(TABLE4)  # three pods, three hashes


def test_the_table4_gist_arms_cap_the_diag_volume_the_log_can_carry() -> None:
    """`diag_every` is the one knob standing between this pod and a lost result.

    The log is the only channel back from a vast.ai instance. A 16K sample runs ~1024
    absorbs per layer; at the 64 default that is ~17 `[diag]` rows per layer per sample
    -- one Qwen half alone would print ~38,000 of them over its 5 gist arms x 16
    windows, none of them a clean summary. `diag_every: 4096` fixes the data shape, not
    a fetch budget: no diagnostic window ever completes, so the only row is
    `drain_diag`'s end-of-sample flush -- one per layer per sample, as the killed run
    measured (320 rows over 10 samples on 32 Llama layers,
    results/hygiene_table4_llama_killed_51394691/diag.jsonl). That is 33 rows/sample at
    most (32 diag + one `[pplw]` line) against the watchdog's 150s/30000-line poll
    (prereg/hygiene_table4.md §8 and Amendment 1)."""
    for name in TABLE4:
        for a in load_pod(name).arms:
            cfg = load_arm(a)
            if cfg.kind == "bug":
                assert cfg.cache.get("diag_every") == 4096, f"{a}: would flood the log"


def test_the_table4_control_arms_are_free_to_diverge() -> None:
    """The guard is the DEFAULT, so an arm that merely leaves `qr_every` unset is still
    guarded and the contrast would compare two guarded arms. The `_noguard` control has
    to switch BOTH tolerances off to be the v1 behaviour that produced the divergence --
    the tripwire still records its `orth_err` trace, which is the point."""
    arms = {a for name in TABLE4 for a in load_pod(name).arms}
    noguard = {a for a in arms if a.endswith("_noguard")}
    assert noguard, "the pods must carry the unguarded control"
    for a in noguard:
        c = load_arm(a).cache
        assert c["orth_fix_tol"] is None and c["orth_abort_tol"] is None
        assert "qr_every" not in c
    for a in {x for x in arms if x.endswith("_qr64")}:
        assert load_arm(a).cache["qr_every"] == 64
    for a in {x for x in arms if x.endswith("_tol")}:
        c = load_arm(a).cache
        assert not {"orth_fix_tol", "orth_abort_tol", "qr_every"} & set(c)  # the defaults


def test_the_table4_arms_equal_the_plain_cache_outside_the_named_knobs() -> None:
    """Every Table-4 cell's `cache:` is `isvd_r128` / `isvd_r256` / `isvd_r256_f0.01` at the
    shipped defaults, plus exactly the guard/floor/diag knobs this prereg varies -- a pin
    against a quiet drift in `rank`, `recent_window`, `absorb_block`, `n_sink` or `retention`
    that no other test here would catch."""
    source = {
        "isvd_r128_noguard": "isvd_r128",
        "isvd_r128_tol": "isvd_r128",
        "isvd_r128_qr64": "isvd_r128",
        "isvd_r128_f0.01_tol": "isvd_r128",
        "isvd_r128_f0.01_qr64": "isvd_r128",
        "isvd_r256_noguard": "isvd_r256",
        "isvd_r256_tol": "isvd_r256",
        "isvd_r256_qr64": "isvd_r256",
        "isvd_r256_f0.01_tol": "isvd_r256_f0.01",
        "isvd_r256_f0.01_qr64": "isvd_r256_f0.01",
    }
    guard_knobs = {"orth_fix_tol", "orth_abort_tol", "qr_every", "diag_every"}
    for arm, plain in source.items():
        cache = load_arm(arm).cache
        drop = guard_knobs | ({"min_sv_frac"} if "f0.01" in arm else set())
        got = {k: v for k, v in cache.items() if k not in drop}
        want = {k: v for k, v in load_arm(plain).cache.items() if k not in drop}
        assert got == want, f"{arm}: drifted from {plain} outside the guard knobs"
        # ...and the floor the NAME promises is the floor the arm carries. Dropped from
        # the comparison above (it is the knob the cell varies), so without this the four
        # `f0.01` cells could run any floor at all -- including the r256 arm's own, which
        # is where the prereg's branch-2 reference number comes from.
        if "f0.01" in arm:
            assert cache["min_sv_frac"] == 0.01, f"{arm}: not the floor its name claims"


# --- The L2 pods: prereg/filler_realism.md, prereg/ss2_families.md, prereg/l2_smoke.md,
# --- and L3's pre-flight pod: prereg/gate1_preflight.md ----------------------------------

FILLER_ARMS = [
    "full",
    "isvd_r64_h256_seed",
    "isvd_r64_h256_seed_q4",
    "kivi2_streaming",
    "kivi2_singleshot",
]
SS2_ARMS = ["isvd_r64_h256_seed", "kivi2_faithful", "kivi2_singleshot", "kivi4_faithful"]
INHOUSE = ["ruler_inhouse_16k", "ruler_inhouse_32k"]
INHOUSE_SUBTASKS = ["niah_single", "niah_multikey", "niah_multivalue", "vt"]
# Each pod as its prereg designs it: the prereg, the arm ORDER, the task list, and the
# arms the runner prefills in one shot (`chunkable: false`) -- what a YAML edit could
# drift from the prereg without any manifest noticing. The order is load-bearing: the
# cheap ceiling control (filler, pre-flight) or the paired r64 reference (cycle, ss2)
# comes first, so a pod that dies early still lands an interpretable result -- for the
# pre-flight pod, the ceiling plus both Gate-1 primary-contrast arms by arm 3
# (`prereg/gate1_preflight.md` §3, §7). The smoke pod's arm set is a rule, not a list
# (`test_the_smoke_pod_names_every_arm_but_the_table4_variants`), and its single-shot arms
# are each arm's own protocol. `gpu_budget_h` and the v2 design are not echoed here: the
# launch manifest's config_hash and the prereg pin those.
FILLER, SS2 = "prereg/filler_realism.md", "prereg/ss2_families.md"
PREFLIGHT_ARMS = ["full", "isvd_r64_h256_seed", "nogist_h2423", "frozen_r64_h256_seed"]
L2_PODS: dict[str, tuple[str, list[str] | None, list[str], list[str] | None]] = {
    "filler_realism": (FILLER, FILLER_ARMS, ["ruler_inhouse_16k_wikitext"], ["kivi2_singleshot"]),
    "filler_realism_cycle": (FILLER, ["isvd_r64_h256_seed", "full"], INHOUSE[:1], []),
    "ss2_families_mistral": (SS2, SS2_ARMS, INHOUSE, SS2_ARMS[1:]),
    "ss2_families_qwen": (SS2, SS2_ARMS, INHOUSE, SS2_ARMS[1:]),
    "ss2_families_llama": (SS2, SS2_ARMS, INHOUSE, SS2_ARMS[1:]),
    "l2_smoke": ("prereg/l2_smoke.md", None, ["ruler_v2_16k"], None),
    "gate1_preflight": ("prereg/gate1_preflight.md", PREFLIGHT_ARMS, ["ruler_v2_16k"], []),
}


@pytest.mark.parametrize("name", list(L2_PODS))
def test_the_l2_pods_resolve_end_to_end(name: str) -> None:
    """The pod loads, hashes, names its prereg (in the launch's ancestry), carries a budget
    to enforce, and every arm builds at the task's context exactly as the runner builds it
    before the first trial (`frontier.build_arm`: a config that cannot resolve fails the
    pod before a record exists). In-process and model-free: `build_arm` only captures the
    model, and `run --dry-run` would spawn a torch-importing subprocess per pod."""
    p = load_pod(name)
    prereg = L2_PODS[name][0]
    assert p.prereg == prereg and (REPO_ROOT / prereg).is_file()
    assert p.gpu_budget_h > 0, f"{name}: a pod to be launched needs a pre-registered budget"
    assert config_hash(p)
    ctx = load_task(p.tasks[0]).ctx
    for a in p.arms:
        cfg = load_arm(a)
        assert cfg.name == a
        assert build_arm(cfg, model=None, t=ctx)["name"] == (cfg.legacy_name or a)
    for t in p.tasks:
        assert load_task(t).name == t


def test_the_l2_pods_are_their_prereg_designs() -> None:
    """Row by row against `L2_PODS`: arm order, task list, the single-shot arms; every task
    at n = 12 (6 trials x 2 seeds, or the v2 design's 12 from one seed) and chunk 4096,
    the in-house pods on the four archived sub-tasks with generator-drawn depths and the
    cycled filler (`wikitext` on the real-text pod), the two v2 pods (smoke, pre-flight) on
    generator v2's five at 16K on the paper's model; bf16 on the -devel image (quanto
    JIT-builds its kernel); seven pods, seven hashes (one arm list against another, one
    filler or model against another keeps them apart)."""
    for name, (_, arms, tasks, single_shot) in L2_PODS.items():
        p = load_pod(name)
        assert p.tasks == tasks, f"{name}: tasks {p.tasks}"
        if arms is not None:
            assert p.arms == arms, f"{name}: not the pre-registered arm order"
        if single_shot is not None:
            assert [a for a in p.arms if not load_arm(a).chunkable] == single_shot, name
        assert p.dtype == "bfloat16" and "-devel" in p.image, f"{name}: {p.dtype} {p.image}"
        for tname in p.tasks:
            t = load_task(tname)
            assert t.n_trials * len(t.seeds) == 12 and t.chunk == 4096, tname
            if t.generator == "v2":
                assert isinstance(t, TaskV2Cfg) and t.ctx == 16384
                assert t.tasks == [*INHOUSE_SUBTASKS[:3], "niah_multiquery", "vt"]
            else:
                assert t.generator == "inhouse" and t.tasks == INHOUSE_SUBTASKS, tname
                filler = "wikitext" if name == "filler_realism" else "cycle"
                assert t.depths is None and t.filler == filler, tname
    for name in ("l2_smoke", "gate1_preflight"):
        assert load_pod(name).model == "unsloth/Meta-Llama-3.1-8B-Instruct", name
    assert len({config_hash(load_pod(n)) for n in L2_PODS}) == len(L2_PODS)


def test_the_filler_realism_pods_pair_with_the_archived_rows() -> None:
    """The real-text pod's records carry the prereg's row keys (the v1 arm strings), and the
    cycled control -- the harness-consistency control that separates a real-text drop from
    drift between `w10_ruler.py` and `pod.py run` (PR-L2-19), `full` joining it under
    Amendment 2 -- runs the same model, generator, context, sub-tasks and chunk: the filler
    is the only difference."""
    real, cycle = load_pod("filler_realism"), load_pod("filler_realism_cycle")
    assert [load_arm(a).legacy_name for a in real.arms] == [
        "full",
        "bugSseed-r64-h256",
        "bugSseed-r64-h256-q4",
        "quant-2bit-kivi",
        "quant-2bit-kivi#chunk0",
    ]
    rt, ct = load_task(real.tasks[0]), load_task(cycle.tasks[0])
    assert cycle.model == real.model
    assert (ct.generator, ct.ctx, ct.tasks, ct.chunk) == (rt.generator, rt.ctx, rt.tasks, rt.chunk)


def test_the_live_filler_manifests_still_hash_to_their_configs() -> None:
    """`doc:` is inside `config_hash`, so an arm a live manifest names carries a frozen
    docstring -- kivi2_streaming's label lives in a `#` comment block instead (L2.9b),
    and YAML comments are outside the hash: both filler pods name the arm, and the
    launched manifests still hash to the configs on disk."""
    for name in ("filler_realism", "filler_realism_cycle"):
        m = json.loads((REPO_ROOT / "results" / name / "manifest.json").read_text())
        assert config_hash(load_pod(name)) == m["config_hash"], name


# --- L3.2: Gate 1, Stage 1 (prereg/gate1_tracker_swap_v2.md) -----------------------------

# The arm ORDER is that file's section 3 and it is load-bearing: it buys an ordered loss --
# both primary contrasts have landed at 15.7 h of compute and the C branch is decidable at
# 22.3 h (section 9) -- so a pod killed at its bar still holds every cell the decision rule
# reads, and section 9's cut ladder drops the last two arms in the order they are listed.
# The no-gist twin is per KV width (1024 channels on Llama, 512 on Qwen), so one H cannot
# serve both pods and the two lists differ in exactly that arm.
GATE1_PREREG = "prereg/gate1_tracker_swap_v2.md"
GATE1_STAGE1_ARMS = [
    "full",
    "isvd_r64_h256_seed",
    "nogist_h2423",
    "frozen_r64_h256_seed",
    "fd_r64_h256_seed",
    "isvd_r64_h256_seed_bf16",
    "oja_r64_h256_seed_tuned",
    "random_r64_h256_seed",
]
GATE1_STAGE1_TASKS = ["ruler_v2_16k_g1", "ppl_16k_pg19val"]
GATE1_SUBTASKS = ["niah_single", "niah_multikey", "niah_multivalue", "vt"]
GATE1_PODS: dict[str, tuple[str, list[str]]] = {
    "gate1_v2_stage1_llama": ("unsloth/Meta-Llama-3.1-8B-Instruct", GATE1_STAGE1_ARMS),
    "gate1_v2_stage1_qwen": (
        "Qwen/Qwen2.5-7B-Instruct",
        ["nogist_h4460" if a == "nogist_h2423" else a for a in GATE1_STAGE1_ARMS],
    ),
}


@pytest.mark.parametrize("name", list(GATE1_PODS))
def test_the_gate1_stage1_pods_resolve_end_to_end(name: str) -> None:
    """Each pod loads, hashes, names the prereg that must precede its launch commit (and
    that file is in the tree, so `pod.py launch`'s ancestry check has something to check),
    carries a budget to enforce, and builds every arm at the task's context exactly as the
    runner builds it before the first trial (`frontier.build_arm`)."""
    p = load_pod(name)
    model, arms = GATE1_PODS[name]
    assert p.model == model and p.prereg == GATE1_PREREG
    assert (REPO_ROOT / GATE1_PREREG).is_file(), "the prereg must be in the launch's ancestry"
    assert p.gpu_budget_h > 0, f"{name}: a pod to be launched needs a pre-registered budget"
    assert config_hash(p)
    for a in arms:
        cfg = load_arm(a)
        assert cfg.name == a
        assert build_arm(cfg, model=None, t=16384)["name"] == (cfg.legacy_name or a)


def test_the_gate1_stage1_pods_are_their_prereg_design() -> None:
    """Section 3 row by row: the arm order, the two tasks, the four Gate-1 sub-tasks at
    n = 24 from one seed on the 2 x 3 x 4 design, the 32-window perplexity sweep on PG-19
    validation (16 would leave the +/-0.02 TOST undecidable at the spread section 2 (b)
    measured), bfloat16 on the -devel image, and a hash distinct from every other pinned
    pod's. `niah_multiquery` is not a Gate-1 task and no contrast in that file reads it.

    128 samples per arm is also what section 8 sizes the log for: at 32 windows a `[pplw]`
    line is ~447 characters and splits into FOUR `part=i/N` fragments per arm, which
    `records.parse_pplw_lines` reassembles -- so no reader here may assume one line per arm.
    """
    for name, (_, arms) in GATE1_PODS.items():
        p = load_pod(name)
        assert p.arms == arms, f"{name}: not the pre-registered arm order"
        assert p.tasks == GATE1_STAGE1_TASKS, f"{name}: tasks {p.tasks}"
        assert p.dtype == "bfloat16" and "-devel" in p.image, f"{name}: {p.dtype} {p.image}"
        ruler = load_task(p.tasks[0])
        assert isinstance(ruler, TaskV2Cfg) and ruler.ctx == 16384 and ruler.chunk == 4096
        assert ruler.tasks == GATE1_SUBTASKS and ruler.seeds == [0]
        assert ruler.n_trials * len(ruler.seeds) == 24, f"{name}: not the pre-registered n"
        assert ruler.design == {"haystacks": 2, "depths": 3, "codes": 4}
        ppl = load_task(p.tasks[1])
        assert (ppl.generator, ppl.corpus, ppl.window) == ("ppl", "pg19-val", 2048)
        assert ppl.n_samples == 32, f"{name}: not the pre-registered window count"
    every = [*L2_PODS, *GATE1_PODS]
    assert len({config_hash(load_pod(n)) for n in every}) == len(every)


# Gate G2 line 6: the k in {0.10, 0.15, 0.25} eviction grid + ThinK composed as its paper
# intends -- ticked by the smoke pod's harvest, so the pod has to carry all nine.
SMOKE_GATE_ARMS = [
    "snapkv_k0.10",
    "snapkv_k0.15",
    "snapkv_k0.25",
    "pyramidkv_k0.10",
    "pyramidkv_k0.15",
    "pyramidkv_k0.25",
    "ea_k0.10",
    "ea_k0.15",
    "think_c0.5_snapkv_k0.15",
]

# L3.1's Gate-1 controls (tests/test_gate1_arms.py): the learn-then-freeze and fixed-random
# tracker arms and the two byte-matched no-gist arms. Their pods are `gate1_preflight` (the
# 1024-wide no-gist arm and the frozen arm) and Stage 1's `gate1_tracker_swap_v2`, not the
# smoke pod -- see the exclusion note in the test below. L5.1's bf16-gist arm rides Stage 1
# beside the r64 arm (`prereg/gate1_tracker_swap_v2.md` arm 6, read by `prereg/bf16_gist.md`),
# so it is excluded for the same reason.
GATE1 = {
    "frozen_r64_h256_seed",
    "random_r64_h256_seed",
    "nogist_h2423",
    "nogist_h4460",
    "isvd_r64_h256_seed_bf16",
}


def test_the_smoke_pod_names_every_arm_but_the_table4_variants() -> None:
    """The arm set is a rule, not a list: every stem under configs/arms/ that is not a Table-4
    diagnostic variant -- the gist arms of the three Table-4 pods, which exist for one
    perplexity contrast and would add ten near-duplicate r128/r256 arms to a retrieval smoke.
    So a future arm cannot be left out silently, a future Table-4 variant is excluded by the
    same rule, and a change in the variants' count is a change to decide, not to inherit.
    Order is cheap -> expensive: `full` first, the twelve gist arms last (a pod that dies early
    still lands whole classes, and the pre-registered cheap first half is everything before the
    first gist arm). The nine arms gate G2 line 6 names are in; no OjaKV stem is (D-017).

    `GATE1` is the second exclusion, and it is named rather than derived because its pods are
    L3's: `prereg/l2_smoke.md` prices this pod at 40 arms x 5 tasks = 200 cells and 12 gist
    arms, while the Gate-1 controls belong to `gate1_preflight` (two of them, pre-registered)
    and to Stage 1's `gate1_tracker_swap_v2` pods (L5's prereg), so putting them here would
    amend a committed pre-registration rather than add a smoke reading. Listing them keeps the
    rule's point: nothing is left out silently."""
    p = load_pod("l2_smoke")
    table4 = {a for n in TABLE4 for a in load_pod(n).arms if load_arm(a).kind == "bug"}
    assert len(table4) == 10, sorted(table4)
    stems = {q.stem for q in (REPO_ROOT / "configs" / "arms").glob("*.yaml")}
    assert stems >= GATE1, sorted(GATE1 - stems)
    assert len(p.arms) == len(set(p.arms)), "an arm listed twice would double its cells"
    assert set(p.arms) == stems - table4 - GATE1
    assert p.arms[0] == "full"
    kinds = [load_arm(a).kind for a in p.arms]
    n_gist = kinds.count("bug")
    assert n_gist == 12, "prereg/l2_smoke.md sizes the budget and the log volume for 12 gist arms"
    assert kinds[-n_gist:] == ["bug"] * n_gist, kinds
    assert set(SMOKE_GATE_ARMS) <= set(p.arms)
    assert not [a for a in p.arms if a.startswith("ojakv")]
