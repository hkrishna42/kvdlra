"""The `kernel_check` axis (prereg/kernel_smoke.md §4 by Amendment 1): on the pod's own model,
greedy decode of the 16 committed prompts under the kernel and under the reconstruct path,
token for token, plus the per-layer max|Δ| of the first kernel decode step -- one record per
(kernel arm, prompt) in `results/<pod>/kernel_check.jsonl`, printed as `[kernel_check ...]`
lines the harvest reads back and `scripts/pod.py check` counts. Here on the tiny model in
fp32 (token-exact by construction, Task 4) and through substituted checks for the wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pod
import pytest
import torch
from transformers import LlamaForCausalLM

from kvdlra.eval import kernel_check
from kvdlra.eval.config import PodCfg, TaskKernelCheckCfg, load_task
from kvdlra.eval.records import KERNEL_CHECK_RE, parse_kernel_check_lines, replayable
from kvdlra.eval.runner import run_pod
from kvdlra.kernel.prompts import TINY_N_NEW, tiny_prompts
from tests.conftest import tiny_cache

TINY_SDPA = True
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"


def _tiny_arm(model: LlamaForCausalLM, **extra: Any) -> dict[str, Any]:
    kw = {"rank": 8, "coord_budget": 64, "recent_window": 4, "absorb_block": 4, "n_sink": 1,
          "decode_attention": "kernel", "kernel_operand_dtype": "float32", **extra}  # fmt: skip
    return {
        "name": "tiny_kernel", "kind": "bug", "kwargs": kw,
        "make": lambda: tiny_cache(model, **kw),
    }  # fmt: skip


def test_is_kernel_arm() -> None:
    assert kernel_check.is_kernel_arm({"kind": "bug", "kwargs": {"decode_attention": "kernel"}})
    assert not kernel_check.is_kernel_arm({"kind": "bug", "kwargs": {}})
    assert not kernel_check.is_kernel_arm({"kind": "full"})


def test_check_prompt_on_the_tiny_model_is_token_exact_and_round_trips(
    tiny_model: LlamaForCausalLM,
) -> None:
    arm = _tiny_arm(tiny_model)
    lines = []
    for i, ids in enumerate(tiny_prompts()[:3]):
        row = kernel_check.check_prompt(tiny_model, arm, ids, index=i, n_new=TINY_N_NEW, chunk=32)
        want = ("tiny_kernel", 64, i, TINY_N_NEW)
        assert (row["arm"], row["ctx"], row["prompt"], row["n_new"]) == want
        assert row["match"] == 1 and row["first_mismatch"] is None and row["error"] is None
        assert row["worst_layer"] in (0, 1) and row["max_abs_diff"] is not None
        assert row["max_abs_diff"] < 1e-4  # fp32 operands: summation order only
        assert len(row["prompt_sha256"]) == 64
        lines.append(kernel_check.format_line(row))
    assert all(KERNEL_CHECK_RE.match(x) and replayable(x) for x in lines)
    back = parse_kernel_check_lines("\n".join(lines), model="M", source="log")
    assert [b["prompt"] for b in back] == [0, 1, 2]
    assert back[0]["max_abs_diff"] == float(f"{lines[0].split('max_abs_diff=')[1].split()[0]}")
    assert back[0]["source"] == "log:1" and back[0]["model"] == "M" and back[0]["error"] is None


def test_failed_row_and_its_line() -> None:
    arm = {"name": "k", "kind": "bug", "kwargs": {}}
    ids = torch.arange(10)
    row = kernel_check.failed_row(arm, ids, index=4, n_new=32, error="RuntimeError: boom")
    line = kernel_check.format_line(row)
    assert line.startswith("[kernel_check prompt=4 arm=k ctx=10 n_new=32 match=0 first_mismatch=- ")
    assert "max_abs_diff=- worst_layer=- sha=" in line
    assert line.endswith(" error=RuntimeError: boom")
    (back,) = parse_kernel_check_lines(line, model="M", source="s")
    assert back["match"] == 0 and back["max_abs_diff"] is None and back["worst_layer"] is None
    assert back["first_mismatch"] is None and back["error"] == "RuntimeError: boom"


def test_the_task_config_loads_and_is_validated() -> None:
    t = load_task("kernel_check_16")
    assert isinstance(t, TaskKernelCheckCfg) and t.generator == "kernel_check"
    assert (t.ctx, t.chunk, t.n_new, t.n_prompts) == (4096, 1024, 32, 16)
    assert t.tasks == []  # contributes no retrieval cells to `check`


def test_the_runner_writes_one_record_per_kernel_arm_and_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The loop through `run_pod` with the model absent: the corpus loader and the per-prompt
    check are substituted; only the kernel arm is checked; a raising prompt is recorded as an
    error row, printed as an `[error] axis=kernel_check` line and counted."""
    calls: list[tuple[str, int]] = []

    def fake_check(
        model: Any, arm: dict[str, Any], ids: torch.Tensor,
        *, index: int, n_new: int, chunk: int,
    ) -> dict[str, Any]:  # fmt: skip
        calls.append((arm["name"], index))
        if index == 5:
            raise RuntimeError("boom")
        return {
            "arm": arm["name"], "ctx": int(ids.shape[0]), "prompt": index, "n_new": n_new,
            "match": 1, "first_mismatch": None, "max_abs_diff": 0.001, "worst_layer": 3,
            "prompt_sha256": "a" * 64, "error": None,
        }  # fmt: skip

    monkeypatch.setattr(kernel_check, "check_prompt", fake_check)
    stream = torch.zeros(1_970_176, dtype=torch.long)
    monkeypatch.setattr("kvdlra.eval.runner.load_corpus_ids", lambda *a, **k: stream)
    cfg = PodCfg(name="kc_fixture", model=MODEL, tasks=["kernel_check_16"],
                 arms=["full", "isvd_r64_h256_seed", "isvd_r64_h256_seed_kernel"])  # fmt: skip
    run_pod(cfg, out=tmp_path, model=None, dry_model=True)
    out = capsys.readouterr().out
    assert calls == [("isvd_r64_h256_seed_kernel", i) for i in range(16)]
    rows = [json.loads(x) for x in (tmp_path / "kernel_check.jsonl").read_text().splitlines()]
    assert len(rows) == 16 and {r["arm"] for r in rows} == {"isvd_r64_h256_seed_kernel"}
    assert rows[5]["error"] == "RuntimeError: boom" and rows[5]["match"] == 0
    assert rows[0]["model"] == MODEL and rows[0]["source"] == "kc_fixture:run"
    assert out.count("[kernel_check prompt=") == 16
    assert ("[error] axis=kernel_check arm=isvd_r64_h256_seed_kernel ctx=4096"
            " error=RuntimeError: boom") in out  # fmt: skip
    assert "[stage] dataset_sha256 pg19-val " in out
    assert ("[stage] cell arm=isvd_r64_h256_seed_kernel task=kernel_check_16"
            " ctx=4096 elapsed_s=") in out  # fmt: skip
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["records"] == {"trials.jsonl": 0, "kernel_check.jsonl": 16} and m["errors"] == 1
    # the pod gate: 16 rows per kernel arm, or a named failure
    assert pod._kernel_check_fails(cfg, tmp_path) == []
    (tmp_path / "kernel_check.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:15]))
    assert pod._kernel_check_fails(cfg, tmp_path) == [
        "kernel_check: isvd_r64_h256_seed_kernel ctx=4096 has 15 of 16 prompt records"
    ]


def test_harvest_reads_kernel_check_lines(tmp_path: Path) -> None:
    line = ("[kernel_check prompt=0 arm=isvd_r64_h256_seed_kernel ctx=4096 n_new=32 match=1 "
            "first_mismatch=- max_abs_diff=3.100e-03 worst_layer=17 sha=" + "b" * 64)  # fmt: skip
    log = tmp_path / "pod.log"
    log.write_text(line + "\n")
    assert pod.harvest("w18_g1", log, tmp_path, force=False) == 0
    (row,) = [json.loads(x) for x in (tmp_path / "kernel_check.jsonl").read_text().splitlines()]
    got = (row["prompt"], row["match"], row["max_abs_diff"], row["worst_layer"])
    assert got == (0, 1, 3.1e-3, 17)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["records"]["kernel_check.jsonl"] == 1
