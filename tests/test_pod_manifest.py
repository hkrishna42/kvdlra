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
                "hit": 0, "frac": 0.0, "haystack_id": None, "depth": None,
                "prompt_sha256": None, "error": "RuntimeError: boom", "source": "x",
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
            "seed": 0, "trial": t, "hit": 1, "frac": 1.0, "haystack_id": None, "depth": None,
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

    assert _rows(tmp_path, "diag.jsonl") == [{"layer": 0, "rank": 64, "source": f"{log}:12"}]
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
ARMS = ("quant-2bit-kivi", "quant-4bit-kivi", "bugSseed-r64-h256")
SUBTASKS = ("niah_single", "niah_multikey", "niah_multivalue", "vt")
CTXS = (16384, 32768)


def _trials(*arms: str) -> list[dict[str, object]]:
    return [
        {
            "model": "m", "arm": arm, "task": task, "ctx": ctx, "seed": seed, "trial": t,
            "hit": 1, "frac": 1.0, "haystack_id": None, "depth": None,
            "prompt_sha256": None, "error": None, "source": "x",
        }
        for arm in arms
        for ctx in CTXS
        for task in SUBTASKS
        for seed in (0, 1)
        for t in range(6)
    ]  # fmt: skip


def _harvested(dry_pod: Path, tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    """A dry-run directory turned into what a real (non-dry) harvest leaves behind."""
    d = _copy(dry_pod, tmp_path)
    m = json.loads((d / "manifest.json").read_text())
    m["dry_run"] = False
    (d / "manifest.json").write_text(json.dumps(m))
    (d / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_check_fails_a_non_dry_run_harvest_with_no_records(dry_pod: Path, tmp_path: Path) -> None:
    """The rule that only counted the cells PRESENT passed a pod that produced nothing.
    Every configured cell is expected; a cell with no rows is the loudest failure there
    is."""
    r = _run("check", str(_harvested(dry_pod, tmp_path, [])))
    assert r.returncode == 1
    out = r.stdout + r.stderr
    assert out.count("CHECK FAIL cells") == len(ARMS) * len(SUBTASKS) * len(CTXS) == 24
    assert "CHECK FAIL cells: bugSseed-r64-h256 vt ctx=32768 has 0 of 12 records" in out


def test_check_fails_when_only_one_arm_reported(dry_pod: Path, tmp_path: Path) -> None:
    """One arm of three at full n: 8 complete cells, 16 empty ones."""
    r = _run("check", str(_harvested(dry_pod, tmp_path, _trials("bugSseed-r64-h256"))))
    assert r.returncode == 1
    out = r.stdout + r.stderr
    assert out.count("CHECK FAIL cells") == 16
    assert "bugSseed-r64-h256" not in out  # the arm that did run is not flagged


def test_check_passes_a_complete_synthetic_harvest(dry_pod: Path, tmp_path: Path) -> None:
    r = _run("check", str(_harvested(dry_pod, tmp_path, _trials(*ARMS))))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK (288 trials, 0 errors)" in r.stdout


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


def _first_commit(path: str) -> str:
    out = subprocess.run(
        ["git", "log", "--reverse", "--format=%H", "--", path],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )  # fmt: skip
    return out.stdout.split("\n")[0].strip()


def test_prereg_commit_order_against_real_history() -> None:
    """The three branches of the order rule, on commits this repo actually has.

    `Makefile` was committed in L0.3, well before HEAD -- a strict ancestor, which is
    the only shape that passes. `prereg/README.md` was created by L0.5c, so checking it
    against that very commit is the "written with the results in hand" case. And a
    commit `prereg/README.md`'s own commit does not descend to (L0.3, which predates it)
    is the not-an-ancestor case."""
    makefile_first = _first_commit("Makefile")
    readme, readme_first = REPO_ROOT / "prereg" / "README.md", _first_commit("prereg/README.md")
    assert pod.prereg_error(REPO_ROOT / "Makefile", "HEAD") is None
    assert "itself" in (pod.prereg_error(readme, readme_first) or "")
    assert "not an ancestor" in (pod.prereg_error(readme, makefile_first) or "")
