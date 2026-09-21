"""`scripts/tables.py latency` -- the kernel-smoke table and the Week-3 reading, from records
(prereg/kernel_smoke.md §7, §4, §2 (a)). Fixtures are `results/<pod>`-shaped directories; the
pod is an in-memory `PodCfg` over the committed arm and task files."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import tables

from kvdlra.accounting import bug_footprint
from kvdlra.eval.config import PodCfg, load_arm, role_of
from kvdlra.eval.records import write_jsonl

MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
ARMS = ["full", "isvd_r64_h256_seed", "isvd_r64_h256_seed_kernel"]
KEYS = {"full": "full", "recon": "bugSseed-r64-h256", "kernel": "isvd_r64_h256_seed_kernel"}
CTXS = (16384, 32768, 65536)
P50 = {"full": (25.9, 27.5, 37.0), "recon": (103.3, 188.3, 507.9), "kernel": (20.0, 30.0, 60.0)}
PEAK = {"full": (2.05, 4.08, 8.14), "recon": (3.25, 6.45, 12.83), "kernel": (0.35, 0.65, 1.20)}


def _pod() -> PodCfg:
    return PodCfg(name="fixture", model=MODEL, arms=ARMS, tasks=["latency_16k_32k_64k"],
                  prereg="prereg/kernel_smoke.md", gpu_budget_h=5.0)  # fmt: skip


def _row(role: str, i: int, **over: Any) -> dict[str, Any]:
    r = {
        "model": MODEL, "arm": KEYS[role], "ctx": CTXS[i], "batch": 1,
        "ms_per_token_p50": P50[role][i], "ms_mean": P50[role][i] * 1.05,
        "ms_max": P50[role][i] * 2.5, "spikes": 4 if role == "recon" else 0,
        "resident_gb": 16.0, "peak_gb": 14.96 + PEAK[role][i], "kv_peak_gb": PEAK[role][i],
        "kv_resident_gb": 0.6, "source": "fixture:run",
    }  # fmt: skip
    return {**r, **over}


def _kc(n_match: int = 16, worst: float = 3.1e-3) -> list[dict[str, Any]]:
    return [
        {"model": MODEL, "arm": KEYS["kernel"], "ctx": 4096, "prompt": i, "n_new": 32,
         "match": int(i < n_match), "first_mismatch": None if i < n_match else 7,
         "max_abs_diff": worst if i == 0 else 1e-3, "worst_layer": 17, "prompt_sha256": "a" * 64,
         "error": None, "source": "fixture:run"}
        for i in range(16)
    ]  # fmt: skip


def _results(tmp_path: Path, rows: list[dict[str, Any]], errors: int = 0,
             kc: list[dict[str, Any]] | None = None, log: str | None = None) -> Path:  # fmt: skip
    d = tmp_path / "fixture"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"pod": "fixture", "errors": errors}))
    write_jsonl(d / "latency.jsonl", rows)
    if kc is not None:
        write_jsonl(d / "kernel_check.jsonl", kc)
    if log is not None:
        (d / "fixture-1.log").write_text(log)
    return d


def _full_grid() -> list[dict[str, Any]]:
    return [_row(role, i) for i in range(3) for role in ("full", "recon", "kernel")]


def _render(tmp_path: Path, **kw: Any) -> str:
    d = _results(tmp_path, **kw)
    out = tmp_path / "kernel_smoke.md"
    tables.latency_table(_pod(), d, out)
    return out.read_text()


def test_roles_and_the_analytic_footprint() -> None:
    got_roles = {(c.legacy_name or c.name): role_of(c) for c in map(load_arm, ARMS)}
    assert got_roles == {KEYS["full"]: "full", KEYS["recon"]: "reconstruct",
                         KEYS["kernel"]: "kernel"}  # fmt: skip
    assert tables.analytic_stored_gib(load_arm("full"), 32768, MODEL) is None
    got = tables.analytic_stored_gib(load_arm("isvd_r64_h256_seed"), 32768, MODEL)
    want = bug_footprint(1024, rank=64, coord_count=32768 - 4 - 256 - 47, recent_len=47, n_sink=4,
                         retention="lowrank_surprise", hh_count=256, u_present=True)  # fmt: skip
    assert got is not None and abs(got - want.stored_bits() * 32 / 8 / 1024**3) < 1e-9
    assert 0.55 < got < 0.57  # ADR 0001 §1: "the stored state is 0.56 GB" at 32K
    assert got == tables.analytic_stored_gib(load_arm("isvd_r64_h256_seed_kernel"), 32768, MODEL)
    # M-6: pin all three R-L4-15 footprints, not just 32K -- the ring/coordinate arithmetic at
    # 16K and 64K is otherwise unpinned.
    for ctx, want_gib in ((16384, 0.302), (32768, 0.556), (65536, 1.064)):
        g = tables.analytic_stored_gib(load_arm("isvd_r64_h256_seed"), ctx, MODEL)
        assert g is not None and abs(g - want_gib) < 5e-4


def test_a_passing_pod_renders_the_table_and_the_verdict(tmp_path: Path) -> None:
    md = _render(tmp_path, rows=_full_grid(), kc=_kc())
    assert "| isvd_r64_h256_seed_kernel | kernel | 32768 | 1 | 30.00 |" in md
    assert "| bugSseed-r64-h256 | reconstruct | 32768 | 1 | 188.30 |" in md
    assert "0.556" in md and "not recorded" not in md and "| -- |" not in md
    assert "memory @32K b1: kernel kv_peak_gb 0.65 vs full 4.08 -> PASS (margin 84.1%)" in md
    assert (
        "speed @32K b1: reconstruct p50 188.30 ms / kernel p50 30.00 ms = 6.28x vs 3.0x -> PASS"
        in md
    )
    assert "archived 188.27 ms: agree (within 10%)" in md
    assert "PRECONDITION: met (16/16 token-exact; worst max|d| 3.100e-03 at layer 17 < 1e-2)" in md
    assert "WEEK-3 GATE (batch 1): PASS" in md


def test_fail_marginal_and_refusals(tmp_path: Path) -> None:
    slow = [
        r if r["arm"] != KEYS["kernel"] else {**r, "ms_per_token_p50": 70.0} for r in _full_grid()
    ]
    md = _render(tmp_path, rows=slow, kc=_kc())
    assert "= 2.69x vs 3.0x -> FAIL" in md and "WEEK-3 GATE (batch 1): FAIL" in md

    fat = [r if r["arm"] != KEYS["kernel"] else {**r, "kv_peak_gb": 3.80} for r in _full_grid()]
    md = _render(tmp_path, rows=fat, kc=_kc())
    assert "kernel kv_peak_gb 3.80 vs full 4.08 -> PASS (MARGINAL: margin 6.9% < 10%)" in md

    spiky = [r if not (r["arm"] == KEYS["kernel"] and r["ctx"] == 32768) else {**r, "spikes": 9}
             for r in _full_grid()]  # fmt: skip
    md = _render(tmp_path, rows=spiky, kc=_kc())
    assert "kernel spikes=9 > 8 of 56" in md and "WEEK-3 GATE (batch 1): REFUSED" in md

    md = _render(tmp_path, rows=_full_grid(), errors=1, kc=_kc())
    assert "1 error row(s) in the manifest" in md and "WEEK-3 GATE (batch 1): REFUSED" in md

    md = _render(tmp_path, rows=_full_grid(), kc=_kc(n_match=13))
    assert "PRECONDITION: NOT met (13/16 token-exact" in md and "REFUSED" in md
    md = _render(tmp_path, rows=_full_grid(), kc=_kc(worst=2e-2))
    assert "PRECONDITION: NOT met" in md and "2.000e-02" in md
    md = _render(tmp_path, rows=_full_grid())
    assert "PRECONDITION: not recorded on this pod" in md
    # M-2 / R-L4-25: a kernel-role arm with no kernel_check.jsonl refuses the verdict.
    assert (
        "WEEK-3 GATE (batch 1): REFUSED -- the correctness precondition was not recorded on"
        " this pod (no kernel_check.jsonl)" in md
    )


def test_missing_and_errored_cells_say_so(tmp_path: Path) -> None:
    rows = [r for r in _full_grid() if not (r["arm"] == KEYS["kernel"] and r["ctx"] == 65536)]
    log = (
        "[error] axis=latency arm=isvd_r64_h256_seed_kernel ctx=65536 batch=1"
        " error=RuntimeError: CUDA out of memory\n"
    )
    md = _render(tmp_path, rows=rows, errors=1, kc=_kc(), log=log)
    assert (
        "| isvd_r64_h256_seed_kernel | kernel | 65536 | 1 |"
        " not run (RuntimeError: CUDA out of memory) |" in md
    )
    rows = [r for r in rows if r["ctx"] != 32768]
    md = _render(tmp_path, rows=rows, kc=_kc())
    assert (
        "not run (no record)" in md and "no 32K record for ['full', 'reconstruct', 'kernel']" in md
    )
    assert "WEEK-3 GATE (batch 1): REFUSED" in md


def test_the_cli_renders_an_existing_pod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`w19_sysfix_latency` has no kernel arm: the table renders and the verdict is REFUSED
    with the reason, never a traceback and never `--`. In-process (no subprocess): patch
    `sys.argv` and call `tables.main()` directly -- this is the lane's only renderer
    subprocess if left as one, so it is kept in-process for the CPU budget."""
    import sys

    d = tmp_path / "w19_sysfix_latency"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"pod": "w19_sysfix_latency", "errors": 0}))
    write_jsonl(d / "latency.jsonl", [_row("full", 1), _row("recon", 1)])
    out = tmp_path / "t.md"
    monkeypatch.setattr(sys, "argv", ["tables.py", "latency", "--pods", str(d), "--out", str(out)])
    tables.main()
    md = out.read_text()
    assert "no kernel arm in the pod" in md and "WEEK-3 GATE (batch 1): REFUSED" in md
    assert "| quant-2bit-kivi | quant | 16384 | 1 | not run (no record) |" in md


def test_spikes_are_reported_on_every_row(tmp_path: Path) -> None:
    """R-L4-24: a `spikes > SPIKES_MAX` row is reported wherever it occurs, not only on the
    reconstruct/kernel arms at 32K -- that GATE_CTX-scoped refusal is unchanged (still exercised
    by test_fail_marginal_and_refusals' `spiky` case). 12 spikes on the kernel arm at 16K is
    outside the gate's scope, so the reading appears but the verdict still PASSes."""
    rows = [
        r if not (r["arm"] == KEYS["kernel"] and r["ctx"] == 16384) else {**r, "spikes": 12}
        for r in _full_grid()
    ]
    md = _render(tmp_path, rows=rows, kc=_kc())
    assert (
        "NOT STEADY STATE: isvd_r64_h256_seed_kernel ctx=16384 batch=1 spikes=12/56"
        " (ms_mean=21.00, ms_max=50.00) — p50 not read as a steady-state number" in md
    )
    assert "WEEK-3 GATE (batch 1): PASS" in md


def test_two_arms_playing_the_same_role_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M-3: `arm_of` inverts `roles`, so two arms mapped to the same gate role would otherwise
    silently collapse to whichever the dict comprehension visited last. No second real `kernel`
    arm config is committed, so fabricate one by wrapping `load_arm` -- same cache, new key."""
    second = "isvd_r64_h256_seed_kernel_2"

    def fake_load_arm(name: str) -> Any:
        if name != second:
            return load_arm(name)
        return dataclasses.replace(load_arm(KEYS["kernel"]), name=second, legacy_name=None)

    monkeypatch.setattr(tables, "load_arm", fake_load_arm)
    pod = PodCfg(name="fixture", model=MODEL, arms=[*ARMS, second],
                 tasks=["latency_16k_32k_64k"], prereg="prereg/kernel_smoke.md",
                 gpu_budget_h=5.0)  # fmt: skip
    out = tmp_path / "dup.md"
    tables.latency_table(pod, _results(tmp_path, rows=_full_grid(), kc=_kc()), out)
    md = out.read_text()
    assert (
        "WEEK-3 GATE (batch 1): REFUSED -- two arms play the kernel role:"
        f" ['{KEYS['kernel']}', '{second}']" in md
    )
