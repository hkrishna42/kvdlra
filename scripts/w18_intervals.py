"""Week-18 intervals + significance builder (marker-free, per-trial-aware).

Unlike w16/w17_intervals -- which recovered each cell's trial count from a hand-
maintained NBLOCK table and dropped every niah_multikey row via a narrow regex -- this
reads the real per-cell ``n=`` off the aggregate line (emitted by w10_ruler as of
Week-18) and parses ALL four RULER tasks including multikey. It also consumes the
per-trial ``[trial]`` lines to run exact McNemar tests on pre-registered arm contrasts
(every "beats" claim must clear McNemar p<0.05 on paired per-trial data).

Inputs: one or more line-files (harvested pod stdout). Outputs: a Wilson-interval table
(JSON+MD) and, for any --contrast A,B, an exact McNemar p-value per (task, ctx).

Usage:
    uv run python scripts/w18_intervals.py results/w18-*-lines.txt \
        --contrast bugSseed-r64-h256,think-c0.5 --out results/w18
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
from scipy.stats import binomtest
from w15_intervals import TASK_LABEL, wilson

# Aggregate row: n= and sbits= are Week-18 additions, appended after ratio= (so the
# older w17/w11 regexes still match); here we REQUIRE n= so the trial count is read off
# the line, never a hand table. sbits= is optional (older lines lack it).
ROW = re.compile(
    r"^\[([A-Za-z0-9_]+) ctx(\d+)\] (\S+)\s+"  # Week-19: official-RULER task names too
    r"acc=([0-9.]+) recall=([0-9.]+) ratio=([0-9.]+)"
    r"(?: sbits=([0-9.]+))? n=(\d+)",
    re.M,
)
# Per-trial line: one Bernoulli outcome per (task, ctx, arm, seed, trial).
TRIAL = re.compile(
    r"^\[trial\] task=(\S+) ctx=(\d+) arm=(\S+) seed=(\d+) trial=(\d+) hit=([01]) frac=([0-9.]+)",
    re.M,
)


def parse_cells(text: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    """(ctx, arm, task) -> {acc, n, hits, ratio, sbits}. Trial count from the line's n=."""
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for m in ROW.finditer(text):
        task, ctx, arm, acc, _recall, ratio, sbits, n = m.groups()
        n_i = int(n)
        hits = round(float(acc) * n_i)
        cells[(ctx, arm, task)] = {
            "acc": float(acc),
            "n": n_i,
            "hits": hits,
            "ratio": float(ratio),
            "sbits": float(sbits) if sbits is not None else None,
        }
    return cells


def parse_trials(text: str) -> dict[tuple[str, str, str], dict[tuple[int, int], int]]:
    """(ctx, arm, task) -> {(seed, trial): hit} for paired significance tests."""
    trials: dict[tuple[str, str, str], dict[tuple[int, int], int]] = defaultdict(dict)
    for m in TRIAL.finditer(text):
        task, ctx, arm, seed, trial, hit, _frac = m.groups()
        trials[(ctx, arm, task)][(int(seed), int(trial))] = int(hit)
    return trials


def mcnemar_exact(
    a: dict[tuple[int, int], int], b: dict[tuple[int, int], int]
) -> dict[str, Any] | None:
    """Exact McNemar on the trials shared by arms a and b (paired by seed,trial).
    b_disc = a-hit & b-miss, c_disc = a-miss & b-hit; exact two-sided binomial on the
    discordant pairs. Returns None if no shared trials."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    b_disc = sum(1 for k in shared if a[k] == 1 and b[k] == 0)
    c_disc = sum(1 for k in shared if a[k] == 0 and b[k] == 1)
    n_disc = b_disc + c_disc
    p = 1.0 if n_disc == 0 else float(binomtest(min(b_disc, c_disc), n_disc, 0.5).pvalue)
    return {
        "n_paired": len(shared),
        "a_favored": b_disc,
        "b_favored": c_disc,
        "p_value": p,
        "significant": p < 0.05,
    }


def build(paths: list[str], contrasts: list[tuple[str, str]]) -> dict[str, Any]:
    text = "\n".join(Path(p).read_text() for p in paths)
    cells = parse_cells(text)
    trials = parse_trials(text)

    interval_cells: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (ctx, arm, task), c in cells.items():
        lo, hi = wilson(c["hits"], c["n"])
        interval_cells[ctx][arm][task] = {
            "acc": c["acc"],
            "n": c["n"],
            "hits": c["hits"],
            "lo": round(lo, 3),
            "hi": round(hi, 3),
            "ratio": c["ratio"],
            "sbits": c["sbits"],
        }

    contrast_out: list[dict[str, Any]] = []
    ctxs = sorted({ctx for ctx, _, _ in cells})
    seen = {tk for _, _, tk in cells}
    # canonical in-repo tasks first, then any other task name (Week-19: the official
    # RULER cells niah_single_1..3 / multikey_1..3 / multiquery contrast too)
    tasks = [t for t in TASK_LABEL if t in seen] + sorted(seen - set(TASK_LABEL))
    for arm_a, arm_b in contrasts:
        for ctx in ctxs:
            for task in tasks:
                a = trials.get((ctx, arm_a, task))
                b = trials.get((ctx, arm_b, task))
                if not a or not b:
                    continue
                res = mcnemar_exact(a, b)
                if res is not None:
                    contrast_out.append(
                        {"arm_a": arm_a, "arm_b": arm_b, "ctx": ctx, "task": task, **res}
                    )
    return {
        "meta": {
            "interval": "wilson",
            "note": "n read from the line's n= field; hits=round(acc*n)",
            "mcnemar": "exact two-sided binomial on discordant paired trials",
        },
        "cells": interval_cells,
        "contrasts": contrast_out,
    }


def fmt_md(blob: dict[str, Any]) -> str:
    out = ["# Week-18 RULER intervals\n", "Cell: `acc [lo,hi] (hits/n)`.\n"]
    for ctx in sorted(blob["cells"], key=int):
        out.append(f"\n## ctx {ctx}\n")
        arms = blob["cells"][ctx]
        tasks = [t for t in TASK_LABEL if any(t in arms[a] for a in arms)]
        header = "| arm | mem | " + " | ".join(TASK_LABEL[t] for t in tasks) + " |"
        out.append(header)
        out.append("|" + "---|" * (len(tasks) + 2))
        for arm in sorted(arms):
            row = arms[arm]
            mem = next((row[t]["ratio"] for t in tasks if t in row), 0.0)
            cells = []
            for t in tasks:
                if t in row:
                    c = row[t]
                    cells.append(
                        f"{c['acc']:.2f} [{c['lo']:.2f},{c['hi']:.2f}] ({c['hits']}/{c['n']})"
                    )
                else:
                    cells.append("—")
            out.append(f"| {arm} | {mem:.3f}x | " + " | ".join(cells) + " |")
    if blob["contrasts"]:
        out.append("\n## McNemar contrasts (exact, paired per-trial)\n")
        out.append("| A | B | ctx | task | n | A>B | B>A | p | sig |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for c in blob["contrasts"]:
            out.append(
                f"| {c['arm_a']} | {c['arm_b']} | {c['ctx']} | {c['task']} | {c['n_paired']} | "
                f"{c['a_favored']} | {c['b_favored']} | {c['p_value']:.4f} | "
                f"{'YES' if c['significant'] else 'no'} |"
            )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="line-file(s) to parse")
    ap.add_argument(
        "--contrast",
        action="append",
        default=[],
        help="A,B arm pair for a McNemar test (repeatable)",
    )
    ap.add_argument("--out", default="results/w18", help="output prefix (.json + .md)")
    args = ap.parse_args()
    contrasts = [(a, b) for a, b in (c.split(",", 1) for c in args.contrast)]
    blob = build(args.paths, contrasts)
    Path(f"{args.out}-ruler-intervals.json").write_text(json.dumps(blob, indent=2))
    Path(f"{args.out}-ruler-intervals.md").write_text(fmt_md(blob))
    n_cells = sum(len(t) for ctx in blob["cells"].values() for t in ctx.values())
    print(
        f"[w18-intervals] {n_cells} RULER cells, {len(blob['contrasts'])} contrasts "
        f"-> {args.out}-ruler-intervals.{{json,md}}",
        flush=True,
    )


if __name__ == "__main__":
    main()
