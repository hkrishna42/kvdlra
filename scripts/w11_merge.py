"""Week-11 final merge: pool all RULER/ppl measurements into the decision table.

RULER accuracies are rebuilt from ALL measurement-line sources, pooled as
hits/total per cell; each source's trials-per-row n comes from its run config
(see ``RULER_LOGS``). hits are recovered as round(acc * n) -- exact for the
all-or-nothing RULER metric. The pre-firming ``results/w11-decision-table.json``
(snapshotted to ``-base.json`` on first run) contributes only ppl/mem for the
old arms; new-arm ppl comes from ``w11v2_new16.acc.log`` (16K) +
``w11v2_ppl32.acc.log`` (32K). Emits ``results/w11-decision-table.json`` (with per-cell ``n_<task>``
counts) and prints the two markdown tables for ``docs/week11-decision-table.md``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

TASKS = ("niah_single", "niah_multikey", "niah_multivalue", "vt")
ROW_RE = re.compile(
    r"^\[(niah_single|niah_multikey|niah_multivalue|vt) ctx(\d+)\] (\S+)\s+"
    r"acc=([0-9.]+) recall=([0-9.]+) ratio=([0-9.]+)",
    re.M,
)
PPL_RE = re.compile(r"^\s+(\S+)\s+\[T=(\d+)\] ppl=([0-9.]+) .*ratio=([0-9.]+)", re.M)

RULER_LOGS: list[tuple[str, dict[str, int]]] = [  # (path, trials-per-row by ctx)
    # -- prior sessions (aggregated lines; n from each run's --n-trials x --seeds)
    ("results/w11-table-ruler-lines.txt", {"16384": 3, "32768": 2}),
    ("results/w11-goalA-ruler-lines.txt", {"16384": 8}),
    ("results/w11-goalB-ruler-lines.txt", {"32768": 6}),
    ("results/w11-base-ruler-lines.txt", {"32768": 2}),
    # -- this session: salvaged v1 pods (3 trials x 2 seeds; lines extracted from
    #    the gitignored gpu_logs into committed files)
    ("results/w11-r128v1-ruler-lines.txt", {"16384": 6, "32768": 6}),
    # -- this session: v2 pods (2 trials x 2 seeds)
    ("results/w11-r128v2-ruler-lines.txt", {"16384": 4, "32768": 4}),
    # -- r256 retrieval follow-up (hard tasks only; needle unmeasured, 2 x 2 seeds)
    ("results/w11-r256-ruler-lines.txt", {"16384": 4, "32768": 4}),
    # -- Week-12 (pre-registered; the merge skip-logs files not yet harvested):
    #    T1 H1 ablation (bugSdrop, 2x2) + bug-r128 gap-fill (1x2); T2 r192 (2x2);
    #    T3 64K prediction test (mk 2x2, mv/vt 1x2).
    ("results/w12-drop-ruler-lines.txt", {"32768": 4}),
    ("results/w12-bugr128-ruler-lines.txt", {"32768": 2}),
    ("results/w12-r192-ruler-lines.txt", {"32768": 4}),
    ("results/w12-mk64-mk-lines.txt", {"65536": 4}),
    ("results/w12-mk64-mvvt-lines.txt", {"65536": 2}),
]
PPL_LOGS = [
    "results/w11-r128-ppl-lines.txt",
    "results/w12-r192-ppl-lines.txt",
]


def merge() -> dict[str, dict[str, dict[str, float]]]:
    # Idempotent: pool from a frozen pre-firming snapshot, never from our own
    # output (re-running against pooled accs would double-count the old trials).
    base = Path("results/w11-decision-table-base.json")
    if not base.exists():
        base.write_text(Path("results/w11-decision-table.json").read_text())
    table: dict[str, dict[str, dict[str, float]]] = json.loads(base.read_text())
    # RULER accuracies are rebuilt ENTIRELY from the measurement lines (the old
    # table's accs are themselves derived from the prior-session lines files, so
    # seeding from both would double-count). The old table contributes ppl/mem.
    acc: dict[tuple[str, str, str], list[int]] = {}
    for path, n_by_ctx in RULER_LOGS:
        p = Path(path)
        if not p.exists():
            print(f"  (skip missing {path})")
            continue
        seen: set[tuple[str, str, str]] = set()  # first occurrence only (logs accumulate)
        for task, ctx, method, a, _rec, ratio in ROW_RE.findall(p.read_text(errors="replace")):
            key = (ctx, method, task)
            if key in seen or ctx not in n_by_ctx:
                continue
            seen.add(key)
            n = n_by_ctx[ctx]
            hits = round(float(a) * n)
            if key in acc:
                acc[key][0] += hits
                acc[key][1] += n
            else:
                acc[key] = [hits, n]
            row = table.setdefault(ctx, {}).setdefault(method, {})
            row.setdefault("mem", float(ratio))

    for ctx, m, t in acc:
        hits, total = acc[(ctx, m, t)]
        table[ctx][m][t] = round(hits / total, 4)
        table[ctx][m][f"n_{t}"] = total

    for path in PPL_LOGS:
        p = Path(path)
        if not p.exists():
            print(f"  (skip missing {path})")
            continue
        for method, t_len, ppl, ratio in PPL_RE.findall(p.read_text(errors="replace")):
            row = table.setdefault(t_len, {}).setdefault(method, {})
            row["ppl"] = float(ppl)
            row.setdefault("mem", float(ratio))
    return table


ORDER = [
    "full",
    "ea-k0.1",
    "bugS-r128-h256",
    "bugS-r128-h1024",
    "bugSdrop-r128-h1024",
    "bugS-r192-h1024",
    "bugS-r256-h256",
    "bugS-r256-h1024",
    "bugS-r32-h256",
    "bugS-r32-h1024",
    "bugEVICT-h256",
    "bugEVICT-h1024",
    "bug-r32",
    "bug-r128",
    "morph-k0.25",
    "morph-k0.5",
    "snapkv-k0.1",
    "snapkv-k0.25",
    "think-c0.3",
    "think-c0.5",
    "palu-r0.5",
    "shadow-r64",
]


def fmt(table: dict[str, dict[str, dict[str, float]]]) -> str:
    out = []
    for ctx in ("16384", "32768", "65536"):
        if ctx not in table:  # 64K appears only once the T3 lines land
            continue
        out.append(f"\n## {int(ctx) // 1024}K context\n")
        out.append(
            "| method | memory | perplexity | needle | multi-key "
            "| multi-value | var-track | n/cell |"
        )
        out.append("|---|---|---|---|---|---|---|---|")
        methods = table[ctx]
        for m in ORDER:
            if m not in methods:
                continue
            r = methods[m]
            cells = []
            for t in TASKS:
                v = r.get(t)
                cells.append("—" if v is None else f"{round(100 * float(v))}")
            ns = sorted({int(r[f"n_{t}"]) for t in TASKS if f"n_{t}" in r})
            n_str = "-".join(str(x) for x in ([ns[0], ns[-1]] if len(ns) > 1 else ns)) or "—"
            ppl = r.get("ppl")
            out.append(
                f"| {m} | {r['mem']:.3f}\u00d7 | {'—' if ppl is None else f'{ppl:.2f}'} | "
                + " | ".join(cells)
                + f" | {n_str} |"
            )
    return "\n".join(out)


if __name__ == "__main__":
    table = merge()
    with open("results/w11-decision-table.json", "w") as f:
        json.dump(table, f, indent=2)
    print(fmt(table))
