"""Week-16 cross-model decision table + Wilson 95% intervals  [DRAFT for orchestrator].

Builds ``results/w16-decision-table.json`` keyed ``model -> ctx -> method -> task`` from
the Week-16 line-files: RULER accuracies pooled as hits/n with a per-file n-per-ctx tag
(``hits = round(acc * n)``, exact for the all-or-nothing metric, same recovery as
``w11_merge.py``); ppl/mem from the ppl line-files. Then a Wilson 95% score interval for
every RULER cell. Kept separate from ``results/w11-decision-table.json`` (Llama-only,
Week-11, no model axis) so that table and its ``w15_intervals`` outputs stay byte-stable.

Emits:
- ``results/w16-decision-table.json``  -- model -> ctx -> method -> {task, n_task, ppl, mem}
- ``results/w16-ruler-intervals.json`` -- model -> ctx -> method -> task -> acc/n/hits/lo/hi
- ``results/w16-ruler-intervals.md``   -- per-model compact table

    uv run python scripts/w16_intervals.py

Read-only over the raw line-files; re-runnable (builds fresh, no pooling into own output).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from w11_merge import PPL_RE, ROW_RE, TASKS
from w15_intervals import Z95, wilson

# scratchpad test hook: point RESULTS_DIR at the line-files' location.
RESULTS = Path(os.environ.get("W16_RESULTS_DIR", "results"))
OUT_TABLE = RESULTS / "w16-decision-table.json"
OUT_JSON = RESULTS / "w16-ruler-intervals.json"
OUT_MD = RESULTS / "w16-ruler-intervals.md"

# (model, line-file, trials-per-row by ctx). Sweep cells are n=4 @16K; the tier runs are
# n=8 @16K (--n-trials 4 --seeds 0 1) and n=4 @32K. Each (ctx,method,task) appears in
# exactly ONE file, so no cross-file pooling occurs (no subset/superset hazard).
W16_RULER_LOGS: list[tuple[str, str, dict[str, int]]] = [
    ("qwen2.5-7b", "w16-qwen-sweep-lines.txt", {"16384": 4}),
    ("qwen2.5-7b", "w16-qwen-tier-lines.txt", {"16384": 8, "32768": 4}),
    ("mistral-7b-v0.3", "w16-mistral-sweep-lines.txt", {"16384": 4}),
    ("mistral-7b-v0.3", "w16-mistral-tier-lines.txt", {"16384": 8, "32768": 4}),
    ("llama-3.1-8b", "w16-llama8b-sweep-lines.txt", {"16384": 4}),
    ("llama-3.1-8b", "w16-llama8b-tier-lines.txt", {"16384": 8, "32768": 4}),
]
W16_PPL_LOGS: list[tuple[str, str]] = [
    ("qwen2.5-7b", "w16-qwen-ppl-lines.txt"),
    ("mistral-7b-v0.3", "w16-mistral-ppl-lines.txt"),
]

Method = dict[str, float]
Table = dict[str, dict[str, dict[str, Method]]]


def build_table() -> Table:
    table: Table = {}
    # RULER first, so RULER's mem (per-task exact-tier ratio) seeds "mem" before ppl.
    for model, fname, n_by_ctx in W16_RULER_LOGS:
        p = RESULTS / fname
        if not p.exists():
            print(f"  (skip missing {fname})")
            continue
        acc: dict[tuple[str, str, str], list[int]] = {}
        seen: set[tuple[str, str, str]] = set()
        for task, ctx, method, a, _rec, ratio in ROW_RE.findall(p.read_text(errors="replace")):
            key = (ctx, method, task)
            if key in seen or ctx not in n_by_ctx:
                continue
            seen.add(key)
            n = n_by_ctx[ctx]
            hits = round(float(a) * n)
            acc[key] = [hits, n]
            row = table.setdefault(model, {}).setdefault(ctx, {}).setdefault(method, {})
            row.setdefault("mem", float(ratio))
        for (ctx, method, task), (hits, total) in acc.items():
            row = table[model][ctx][method]
            row[task] = round(hits / total, 4)
            row[f"n_{task}"] = total  # int; assignable to dict[str, float]
    for model, fname in W16_PPL_LOGS:
        p = RESULTS / fname
        if not p.exists():
            print(f"  (skip missing {fname})")
            continue
        for method, t_len, ppl, ratio in PPL_RE.findall(p.read_text(errors="replace")):
            row = table.setdefault(model, {}).setdefault(t_len, {}).setdefault(method, {})
            row["ppl"] = float(ppl)
            row.setdefault("mem", float(ratio))
    return table


def build_intervals(table: Table) -> dict[str, object]:
    cells: dict[str, dict[str, dict[str, dict[str, dict[str, float | int]]]]] = {}
    for model in table:
        for ctx in sorted(table[model], key=int):
            for method, row in table[model][ctx].items():
                for task in TASKS:
                    if task not in row or f"n_{task}" not in row:
                        continue
                    a, n = float(row[task]), int(row[f"n_{task}"])
                    hits = round(a * n)
                    if abs(a * n - hits) > 0.01:
                        raise ValueError(f"non-integral hits at {model}/{ctx}/{method}/{task}")
                    lo, hi = wilson(hits, n)
                    (
                        cells.setdefault(model, {}).setdefault(ctx, {}).setdefault(method, {})[task]
                    ) = {"acc": a, "n": n, "hits": hits, "lo": round(lo, 4), "hi": round(hi, 4)}
    return {
        "meta": {
            "interval": "wilson",
            "z": Z95,
            "note": "hits = round(acc*n); lo/hi fractions in [0,1]",
        },
        "cells": cells,
    }


def fmt_md(blob: dict[str, object]) -> str:
    cells = blob["cells"]
    assert isinstance(cells, dict)
    out = [
        "# Week-16: Wilson 95% intervals (per model)",
        "",
        "Cell: `acc% [lo, hi] (hits/n)`. Source: Week-16 line-files.",
    ]
    for model in cells:
        out.append(f"\n## {model}\n")
        for ctx in cells[model]:
            out.append(f"### {int(ctx) // 1024}K context\n")
            out.append("| method | " + " | ".join(TASKS) + " |")
            out.append("|---|" + "---|" * len(TASKS))
            for method, row in cells[model][ctx].items():
                parts = []
                for t in TASKS:
                    c = row.get(t)
                    parts.append(
                        "—"
                        if c is None
                        else f"{100 * c['hits'] / c['n']:.0f} "
                        f"[{100 * c['lo']:.0f},{100 * c['hi']:.0f}] "
                        f"({c['hits']}/{c['n']})"
                    )
                out.append(f"| {method} | " + " | ".join(parts) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    table = build_table()
    OUT_TABLE.write_text(json.dumps(table, indent=2) + "\n")
    blob = build_intervals(table)
    OUT_JSON.write_text(json.dumps(blob, indent=2) + "\n")
    OUT_MD.write_text(fmt_md(blob))
    cells = blob["cells"]
    assert isinstance(cells, dict)
    n_cells = sum(len(r) for ms in cells.values() for cx in ms.values() for r in cx.values())
    print(f"[w16-intervals] {n_cells} RULER cells -> {OUT_JSON.name} + {OUT_MD.name}")


if __name__ == "__main__":
    main()
