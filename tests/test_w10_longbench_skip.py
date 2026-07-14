"""Week-10/11 regression: one LongBench task's load failure is non-fatal.

``w10_longbench.run`` called ``load_dataset("THUDM/LongBench", task, ...)`` +
``build_example`` OUTSIDE the per-arm try/except, so ANY task's dataset load
failing aborted the whole axis before a single arm ran -- the parsed Week-10
logs showed ``lb:[]`` (nothing at all). The fix wraps the per-task load so a
failing task is skipped (logged) and the remaining tasks still run.

This drives ``run`` with everything heavy stubbed (no model/dataset download):
``load_dataset`` raises for the first task and succeeds for the second, and we
assert the second task still produced result rows.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import w10_longbench


def _args(tasks: list[str]) -> argparse.Namespace:
    return argparse.Namespace(
        model="stub-model",
        device="cpu",
        dtype="float32",
        tasks=tasks,
        max_len=2048,
        chunk=0,
        n_examples=2,
        max_new=4,
    )


def _stub_model() -> Any:
    cfg = SimpleNamespace(
        head_dim=16,
        num_key_value_heads=2,
        num_attention_heads=4,
        hidden_size=64,
    )
    return SimpleNamespace(config=cfg)


def test_run_skips_one_failing_task_keeps_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(w10_longbench, "install_kvpress_prefill_compat", lambda: None)
    monkeypatch.setattr(w10_longbench, "load_model", lambda *a, **k: (_stub_model(), object()))

    def fake_load_dataset(_name: str, task: str, **_kw: Any) -> list[dict[str, Any]]:
        if task == "bad_task":
            raise RuntimeError("simulated dataset load failure")
        return [{"_": i} for i in range(4)]

    monkeypatch.setattr(w10_longbench, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        w10_longbench,
        "build_example",
        lambda _tok, _ex, _max_len: (torch.ones(1, 1500, dtype=torch.long), ["gold"]),
    )
    monkeypatch.setattr(
        w10_longbench,
        "build_arms",
        lambda *_a, **_k: [{"name": "full", "kind": "full", "rank": None}],
    )
    monkeypatch.setattr(w10_longbench, "generate", lambda *_a, **_k: ("gold answer text", 0.5))

    blob = w10_longbench.run(_args(["bad_task", "good_task"]))
    tasks_seen = {r["task"] for r in blob["results"]}
    # The failing task is skipped; the healthy task still produced rows (not lb:[]).
    assert "bad_task" not in tasks_seen
    assert "good_task" in tasks_seen
    assert all(r["method"] == "full" for r in blob["results"])
