"""Week-19 A2: per-record view of the official-RULER anchor -- which needles an arm missed,
at what depth, and what every other arm did on the same record.

The official records are regenerated locally with the pinned generator (RULER c3f5e3b,
``prepare.py --random_seed 42 --tokenizer_path <model>``), which is deterministic, so the
pod's ``[trial] ... trial=<index>`` lines map onto ``validation.jsonl`` records by index;
``token_position_answer / length`` is the needle depth the generator recorded.

    uv run python scripts/w19_a2_misses.py --data-dir <RULER save_dir> \
        --trials results/w19_pertrial/a2-llama-trials.txt --arm bugSseed-r64-h256
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

TRIAL = re.compile(r"task=(\S+) ctx=\d+ arm=(\S+) seed=\d+ trial=(\d+) hit=([01])")
ARMS = [
    "full",
    "ea-k0.1",
    "think-c0.5",
    "palu-r0.5",
    "quant-2bit-kivi",
    "quant-4bit-kivi",
    "bugSseed-r64-h256",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--trials", required=True, help="per-trial [trial] lines from the a2 pod")
    ap.add_argument("--arm", default="bugSseed-r64-h256")
    ap.add_argument("--out", default="results/w19-a2-flagship-misses.md")
    args = ap.parse_args()
    recs: dict[tuple[str, int], dict[str, object]] = {}
    for d in sorted(Path(args.data_dir).iterdir()):
        f = d / "validation.jsonl"
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                recs[(d.name, int(r["index"]))] = r
    hits: dict[tuple[str, int], dict[str, int]] = collections.defaultdict(dict)
    for line in Path(args.trials).read_text().splitlines():
        m = TRIAL.search(line)
        if m:
            hits[(m.group(1), int(m.group(3)))][m.group(2)] = int(m.group(4))
    out = [
        f"# Official RULER (Llama-3.1-8B, 16K): records missed by `{args.arm}`\n",
        "Depth = the generator's `token_position_answer / length`. Columns = hit (1) / miss (0) "
        "per arm on the SAME record.\n",
        "| task | record | depth | " + " | ".join(f"`{a}`" for a in ARMS) + " |",
        "|---|---|---|" + "---|" * len(ARMS),
    ]
    depths: dict[str, list[float]] = collections.defaultdict(list)
    for (task, idx), h in sorted(hits.items()):
        r = recs.get((task, idx), {})
        pos, ln = r.get("token_position_answer"), r.get("length")
        depth = pos / ln if isinstance(pos, int) and isinstance(ln, int) and ln else None
        for arm, hit in h.items():
            if hit == 0 and depth is not None:
                depths[arm].append(round(depth, 2))
        if h.get(args.arm) == 0:
            ds = f"{depth:.2f}" if depth is not None else "n/a"
            out.append(
                f"| {task} | {idx} | {ds} | " + " | ".join(str(h.get(a, "—")) for a in ARMS) + " |"
            )
    out.append("\n## Miss depths per arm (all official cells)\n")
    for arm in ARMS:
        out.append(f"- `{arm}`: {len(depths[arm])} misses at depths {sorted(depths[arm])}")
    Path(args.out).write_text("\n".join(out) + "\n")
    n_missed = sum(1 for row in out if row.startswith("| ") and args.arm not in row) - 1
    print(f"[wrote {args.out}] {n_missed} missed records")


if __name__ == "__main__":
    main()
