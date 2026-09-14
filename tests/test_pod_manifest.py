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


LOG = """===RUN_SHA_deadbeef===
===ENV_BEGIN===
run_sha=deadbeef
===ENV_END===
[trial] task=niah_single ctx=16384 arm=bugSseed-r64-h256 seed=0 trial=0 hit=1 frac=1.000
[trial] task=niah_single ctx=16384 arm=bugSseed-r64-h256 seed=0 trial=1 hit=0 frac=0.000
[pplw] T=16384 bugSseed-r64-h256 ntok=511 nlls=1.573386,1.236791
  bugSseed-r64-h256 [T=16384] ppl=5.403 tok_eq/layer=1398.0 ratio=0.163 sbits=0.163
[diag] {"layer": 0, "rank": 64}
===ALL_DONE_w18_g1_deadbeef===
"""


def test_harvest_parses_a_log_into_records(dry_pod: Path, tmp_path: Path) -> None:
    tmp_path = _copy(dry_pod, tmp_path)
    log = tmp_path / "pod.log"
    log.write_text(LOG)
    r = _run("harvest", "--pod", "w18_g1", "--log", str(log), "--out", str(tmp_path))
    assert r.returncode == 0, r.stderr
    trials = [json.loads(x) for x in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert [t["hit"] for t in trials] == [1, 0]
    assert trials[0]["model"] == load_pod("w18_g1").model
    ppl = [json.loads(x) for x in (tmp_path / "ppl.jsonl").read_text().splitlines()]
    assert ppl == [
        {
            "arm": "bugSseed-r64-h256", "ctx": 16384, "ntok": 511,
            "nlls": [1.573386, 1.236791], "source": f"{log}:7",
        }
    ]  # fmt: skip  # [pplw] wins over the aggregate line when the log carries both
    assert json.loads((tmp_path / "diag.jsonl").read_text()) == {"layer": 0, "rank": 64}
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["records"] == {"trials.jsonl": 2, "ppl.jsonl": 1, "diag.jsonl": 1}
    assert m["errors"] == 0 and m["harvested_at"]


def test_harvest_falls_back_to_the_aggregate_ppl_lines(dry_pod: Path, tmp_path: Path) -> None:
    tmp_path = _copy(dry_pod, tmp_path)
    log = tmp_path / "pod.log"
    log.write_text("\n".join(x for x in LOG.splitlines() if not x.startswith("[pplw]")))
    _run("harvest", "--pod", "w18_g1", "--log", str(log), "--out", str(tmp_path))
    ppl = [json.loads(x) for x in (tmp_path / "ppl.jsonl").read_text().splitlines()]
    assert len(ppl) == 1 and ppl[0]["ppl"] == 5.403 and ppl[0]["sbits"] == 0.163


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
