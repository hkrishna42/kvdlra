"""Week-17 cross-model decision table + Wilson 95% intervals.

Reads the committed per-model line-files ``results/w17-<tag>-lines.txt`` (block markers
``===W17_<BLOCK>_BEGIN_<tag>=== `` + RULER rows + ppl rows, extracted verbatim from the
pod logs) and builds ``results/w17-decision-table.json`` (``model -> ctx -> method ->
{task, n_task, ppl, mem}``) plus a Wilson 95% score interval for every RULER cell.

The trial count ``n`` depends on the BLOCK, not just ctx:
  16K_CORE 16K n=12 | 32K_CORE 32K n=4 | VTFIX 16K(h512) n=12 / 32K(h1024) n=4 |
  MARQUEE 32K(s32,think,palu) n=16 | FLOOR 16K(floor-ruler) n=4.
When one (ctx, method, task) is measured in two blocks -- only think/palu {vt,mv}@32K, in
32K_CORE (n=4) AND MARQUEE (n=16), where the n=4 (seeds 0-1, trials 0-1) is a strict SUBSET
of the n=16 (trials 0-7) -- the LARGEST n (the superset) is kept; the two are never pooled
(that would double-count). Kept separate from ``w11``/``w16`` tables so those stay byte-stable.

    uv run python scripts/w17_intervals.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from w11_merge import PPL_RE, TASKS
from w15_intervals import Z95, wilson

RESULTS = Path("results")
MODELS = {"qwen": "qwen2.5-7b", "mistral": "mistral-7b-v0.3", "llama8b": "llama-3.1-8b"}

ROW = re.compile(
    r"^\[(niah_single|niah_multivalue|vt) ctx(\d+)\] (\S+)\s+"
    r"acc=([0-9.]+) recall=([0-9.]+) ratio=([0-9.]+)"
)
BEGIN = re.compile(r"^===W17_(16K_CORE|32K_CORE|VTFIX|MARQUEE|PPL|FLOOR)_")
# n_by_ctx per block; a ctx absent from a block -> that block's RULER rows are ignored.
NBLOCK: dict[str, dict[str, int]] = {
    "16K_CORE": {"16384": 12},
    "32K_CORE": {"32768": 4},
    "VTFIX": {"16384": 12, "32768": 4},
    "MARQUEE": {"32768": 16},
    "FLOOR": {"16384": 4},
    "PPL": {},
}

Cell = dict[str, float]
Table = dict[str, dict[str, dict[str, Cell]]]


def parse_model(tag: str) -> dict[str, dict[str, Cell]]:
    path = RESULTS / f"w17-{tag}-lines.txt"
    if not path.exists():
        print(f"  (skip missing {path.name})")
        return {}
    # (ctx, method, task) -> (n, hits, acc, ratio); keep the largest-n (superset-safe).
    cells: dict[tuple[str, str, str], tuple[int, int, float, float]] = {}
    block: str | None = None
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        mb = BEGIN.match(line)
        if mb:
            block = mb.group(1)
            continue
        mr = ROW.match(line)
        if not mr or block is None:
            continue
        task, ctx, method, acc, _rec, ratio = mr.groups()
        n = NBLOCK.get(block, {}).get(ctx)
        if n is None:
            continue
        key = (ctx, method, task)
        hits = round(float(acc) * n)
        prev = cells.get(key)
        if prev is None or n > prev[0]:
            cells[key] = (n, hits, float(acc), float(ratio))
    out: dict[str, dict[str, Cell]] = {}
    for (ctx, method, task), (n, _hits, acc, ratio) in sorted(cells.items()):
        row = out.setdefault(ctx, {}).setdefault(method, {})
        row.setdefault("mem", ratio)
        row[task] = acc
        row[f"n_{task}"] = float(n)
    # ppl: T is the ctx key; method-keyed (bugSseed-r64-h256 @ each T).
    for method, t_len, ppl, ratio in PPL_RE.findall(text):
        row = out.setdefault(t_len, {}).setdefault(method, {})
        row["ppl"] = float(ppl)
        row.setdefault("mem", float(ratio))
    return out


def build_intervals(table: Table) -> dict[str, object]:
    cells: dict[str, dict[str, dict[str, dict[str, dict[str, float | int]]]]] = {}
    for model in table:
        for ctx in sorted(table[model], key=int):
            for method, row in table[model][ctx].items():
                for task in TASKS:
                    if task not in row or f"n_{task}" not in row:
                        continue
                    acc, n = float(row[task]), int(row[f"n_{task}"])
                    hits = round(acc * n)
                    lo, hi = wilson(hits, n)
                    mcell = cells.setdefault(model, {}).setdefault(ctx, {}).setdefault(method, {})
                    mcell[task] = {
                        "acc": acc,
                        "n": n,
                        "hits": hits,
                        "lo": round(lo, 3),
                        "hi": round(hi, 3),
                    }
    return {"meta": {"interval": "wilson", "z": Z95, "note": "hits = round(acc*n)"}, "cells": cells}


def fmt_md(blob: dict[str, object], table: Table) -> str:
    cells = blob["cells"]
    assert isinstance(cells, dict)
    out = [
        "# Week-17 RULER Wilson 95% intervals (per model)",
        "",
        "Cell: `acc [lo,hi] (hits/n)`. Source: `results/w17-<model>-lines.txt`.",
        "",
    ]
    for model in cells:
        out.append(f"## {model}")
        for ctx in sorted(cells[model], key=int):
            out.append(f"\n### {int(ctx) // 1024}K context\n")
            heads = " | ".join(t.replace("niah_", "") for t in TASKS)
            out.append("| method | mem | " + heads + " |")
            out.append("|---|---|" + "---|" * len(TASKS))
            for method, row in cells[model][ctx].items():
                mem = float(table[model][ctx][method].get("mem", 0.0))
                parts = []
                for task in TASKS:
                    c = row.get(task)
                    if c is None:
                        parts.append("—")
                    else:
                        parts.append(
                            f"{c['acc']:.2f} [{c['lo']:.2f},{c['hi']:.2f}] ({c['hits']}/{c['n']})"
                        )
                out.append(f"| {method} | {mem:.3f}x | " + " | ".join(parts) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    table: Table = {}
    for tag, model in MODELS.items():
        t = parse_model(tag)
        if t:
            table[model] = t
    (RESULTS / "w17-decision-table.json").write_text(json.dumps(table, indent=2) + "\n")
    blob = build_intervals(table)
    (RESULTS / "w17-ruler-intervals.json").write_text(json.dumps(blob, indent=2) + "\n")
    (RESULTS / "w17-ruler-intervals.md").write_text(fmt_md(blob, table))
    cells = blob["cells"]
    assert isinstance(cells, dict)
    n_cells = sum(len(r) for ms in cells.values() for cx in ms.values() for r in cx.values())
    print(
        f"[w17-intervals] models={list(table)} · {n_cells} RULER cells "
        f"-> w17-decision-table.json + w17-ruler-intervals.{{json,md}}"
    )


if __name__ == "__main__":
    main()
