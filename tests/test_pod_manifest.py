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

from kvdlra.eval.config import config_hash, load_pod

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
        "window_idx": 0, "ntok": 511, "nll_sum_nats": 1.573386 * 511, "source": f"{log}:7",
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
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "RUN_FAILED"


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


def test_prereg_refusal_reasons(tmp_path: Path) -> None:
    """Missing and uncommitted both refuse; the message names which."""
    missing = tmp_path / "nope.md"
    assert "missing" in (pod.prereg_error(missing, "HEAD") or "")
    uncommitted = tmp_path / "w99.md"
    uncommitted.write_text("# pre-registration\n")
    assert "uncommitted" in (pod.prereg_error(uncommitted, "HEAD") or "")


def test_launch_dry_run_prints_the_vastai_command(tmp_path: Path) -> None:
    """The command is the one the boot-script header documents: image from the pod
    YAML, --disk 80, the POD/SHA/MODEL/DTYPE env, boot.sh as --onstart."""
    cmd = pod.launch_command("w18_g1", "12345678", "deadbeef")
    assert cmd[:4] == ["vastai", "create", "instance", "12345678"]
    joined = " ".join(cmd)
    assert "--disk 80" in joined and "--onstart scripts/pod/boot.sh" in joined
    assert "--label kvdlra-w18_g1" in joined
    assert "-e POD=w18_g1 -e SHA=deadbeef" in joined
    assert load_pod("w18_g1").image in joined


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
    d = _harvested(dry_pod, tmp_path, [*_trial_lines(*ARMS), *_ppl_lines(*ARMS)])
    assert {r["generator"] for r in _rows(d, "trials.jsonl")} == {"inhouse"}
    r = _run("check", str(d))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK (288 trials, 0 errors)" in r.stdout


def test_a_harvest_keeps_the_generator_the_row_itself_names(dry_pod: Path, tmp_path: Path) -> None:
    """The runner prints it, so no inference is needed -- and the pod-config map is not
    consulted at all for a row that says which generator built it."""
    lines = [*_trial_lines(*ARMS, generator="inhouse"), *_ppl_lines(*ARMS)]
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
    is expected in ppl.jsonl -- w18_g1 names two ppl tasks, so 3 arms x 2 ctx = 6."""
    r = _run("check", str(_harvested(dry_pod, tmp_path, _trial_lines(*ARMS))))
    assert r.returncode == 1
    out = r.stdout + r.stderr
    assert out.count("CHECK FAIL ppl") == len(ARMS) * len(CTXS) == 6
    assert "CHECK FAIL ppl: bugSseed-r64-h256 ctx=32768 has no perplexity record" in out


def test_check_fails_when_one_arm_is_missing_from_the_ppl_sweep(
    dry_pod: Path, tmp_path: Path
) -> None:
    lines = [*_trial_lines(*ARMS), *_ppl_lines("quant-2bit-kivi", "quant-4bit-kivi")]
    r = _run("check", str(_harvested(dry_pod, tmp_path, lines)))
    assert r.returncode == 1
    assert r.stdout.count("CHECK FAIL ppl") == 2  # the r64 arm's two contexts


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
