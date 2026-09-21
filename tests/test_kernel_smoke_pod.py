"""A complete kernel-smoke log through harvest, check and the renderer: 16 `[kernel_check]`
lines + 9 `[latency]` lines pass `scripts/pod.py check`; a missing prompt or decode row fails
it by name; `scripts/tables.py latency` renders the pod's table with a PASS verdict from those
rows and REFUSED when the check is short.

In-process throughout, for the CPU budget: one module-scoped dry run (`pod.run`, the
`dry_pod` + `_copy` pattern of `tests/test_pod_manifest.py:66-76` -- `run` does not import
torch on the `dry_run=True` path), then `pod.harvest`, `pod.check` (`capsys` reads its
`CHECK FAIL ...` prints) and `tables.latency_table` as plain calls. The one exception is the
refusal path: a real subprocess is the simplest way to observe `launch`'s exit code and
combined stdout/stderr from a clean process, and it is the module's only one."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pod
import pytest
import tables

from kvdlra.eval.config import load_arm, load_pod, load_task
from tests.test_latency_table import KEYS, P50, PEAK

REPO_ROOT = Path(__file__).resolve().parents[1]
POD = "kernel_smoke"
ROLE = {key: role for role, key in KEYS.items()}  # the same numbers, keyed by record key


def _lines(n_prompts: int = 16) -> list[str]:
    cfg = load_pod(POD)
    names = [load_arm(a).legacy_name or a for a in cfg.arms]
    lat = load_task(cfg.tasks[1])
    out = [
        # The pre_run hook (L4.10): `check` refuses this pod without a recorded, zero
        # exit code -- boot.sh runs the measurement whatever the kernel's gpu tests said.
        "===PRE_RUN_BEGIN_kernel_smoke===",
        "[pre_run] 8 passed in 41.2s",
        "===PRE_RUN_END_kernel_smoke_rc=0===",
    ]
    out += [
        f"[kernel_check prompt={i} arm=isvd_r64_h256_seed_kernel ctx=4096 n_new=32 match=1"
        f" first_mismatch=- max_abs_diff=3.100e-03 worst_layer=17 sha={'a' * 64}"
        f" backend=triton"
        for i in range(n_prompts)
    ]
    for i, ctx in enumerate(lat.ctxs or []):
        for arm in names:
            out.append(
                f"[latency ctx{ctx}] {arm:22s} ms/tok={P50[ROLE[arm]][i]:.2f}"
                f" mean={P50[ROLE[arm]][i] * 1.05:.2f} max={P50[ROLE[arm]][i] * 2.5:.2f}"
                f" spikes={4 if ROLE[arm] == 'recon' else 0}"
                f" resident_gb=16.00 peak_gb={14.96 + PEAK[ROLE[arm]][i]:.2f} weights_gb=14.96"
                f" kv_peak_gb={PEAK[ROLE[arm]][i]:.2f} batch=1 kv_resident_gb=0.60"
                + (" backend=triton" if ROLE[arm] == "kernel" else "")
            )
    return out


@pytest.fixture(scope="module")
def dry_pod(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One dry run for the tests below to copy: `pod.run` writes the manifest and env.txt
    without touching torch (the `dry_run=True` path returns before the torch import)."""
    d = tmp_path_factory.mktemp(POD)
    assert pod.run(POD, d, dry_run=True) == 0
    return d


def _harvested(dry_pod: Path, dest: Path, lines: list[str]) -> Path:
    d = Path(shutil.copytree(dry_pod, dest))
    m = json.loads((d / "manifest.json").read_text())
    m["dry_run"] = False
    (d / "manifest.json").write_text(json.dumps(m))
    log = d / f"{POD}-1.log"
    log.write_text("\n".join(lines) + "\n")
    assert pod.harvest(POD, log, d, force=False) == 0
    return d


def test_a_complete_kernel_smoke_harvest_passes_check_and_renders(
    dry_pod: Path, tmp_path: Path
) -> None:
    d = _harvested(dry_pod, tmp_path / "complete", _lines())
    m = json.loads((d / "manifest.json").read_text())
    assert m["records"] == {"trials.jsonl": 0, "kernel_check.jsonl": 16, "latency.jsonl": 9}
    assert pod.check(d) == 0
    out = tmp_path / "kernel_smoke.md"
    tables.latency_table(load_pod(POD), d, out)
    md = out.read_text()
    assert "PRECONDITION: met (16/16 token-exact" in md
    assert "WEEK-3 GATE (batch 1): PASS" in md
    assert "archived 188.27 ms: agree (within 10%)" in md and "| 0.556 |" in md
    # The attested backend reaches the table, and only on the rows that ran one: every other
    # arm (and a kernel row logged before the field existed) renders `n/a` there.
    assert [ln.split(" | ")[0] for ln in md.splitlines() if ln.endswith("| triton |")] == [
        "| isvd_r64_h256_seed_kernel"
    ] * 3


def test_a_short_check_or_grid_fails_check_by_name(
    dry_pod: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = _harvested(dry_pod, tmp_path / "short", _lines(n_prompts=15))
    assert pod.check(d) == 1
    assert (
        "CHECK FAIL kernel_check: isvd_r64_h256_seed_kernel ctx=4096 has 15 of 16 prompt records"
        in capsys.readouterr().out
    )
    e = _harvested(
        dry_pod, tmp_path / "grid", [x for x in _lines() if "ctx65536] isvd_r64" not in x]
    )
    assert pod.check(e) == 1
    assert (
        "CHECK FAIL latency: isvd_r64_h256_seed_kernel ctx=65536 batch=1 has no decode record"
        in capsys.readouterr().out
    )


def test_launch_is_refused_from_a_dirty_or_unpushed_tree_and_needs_the_prereg() -> None:
    """The refusal path only: `--dry-run` reaches vastai never. A clean, pushed tree with the
    prereg as a strict ancestor is what the orchestrator's launch (D-011) must show. This
    module's one subprocess."""
    r = subprocess.run(
        [sys.executable, "scripts/pod.py", "launch", "--pod", POD,
         "--offer", "12345678", "--dry-run"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )  # fmt: skip
    joined = r.stdout + r.stderr
    assert "REFUSE" in joined or "vastai create instance 12345678" in joined
    assert "max-hours" not in joined  # the bar is pre-registered (5.0), never a refusal reason
