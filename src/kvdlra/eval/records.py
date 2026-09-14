"""Per-trial and per-cell records: the only unit `make tables` reads.

``[trial]`` lines are the Bernoulli outcomes the pods printed (one per task x ctx x arm
x seed x trial); ``[task ctxN] arm acc= ... n=`` lines are pooled cells. Both regexes are
the emitters' formats from Week 18/19 (previously duplicated in ten reader scripts).

``haystack_id``/``depth``/``prompt_sha256``/``error`` are carried in the schema but are
``None`` for the archived paper-v1 records: the v1 emitters never printed them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict

TRIAL_RE = re.compile(
    r"^\[trial\] task=(\S+) ctx=(\d+) arm=(\S+) seed=(\d+) trial=(\d+) hit=([01]) frac=([0-9.]+)"
)
CELL_RE = re.compile(
    r"^\[([A-Za-z0-9_]+) ctx(\d+)\] (\S+)\s+acc=([0-9.]+) recall=([0-9.]+) ratio=([0-9.]+)"
    r"(?: sbits=([0-9.]+))?(?: n=(\d+))?"
)


class TrialRecord(TypedDict):
    model: str
    arm: str
    task: str
    ctx: int
    seed: int
    trial: int
    hit: int
    frac: float
    haystack_id: str | None
    depth: float | None
    prompt_sha256: str | None
    error: str | None
    source: str


class CellRecord(TypedDict):
    model: str
    arm: str
    task: str
    ctx: int
    acc: float
    n: int | None
    hits: int | None
    ratio: float
    source: str


def parse_trial_lines(text: str, model: str, source: str) -> list[TrialRecord]:
    """Every ``[trial]`` line in ``text`` as a record citing ``<source>:<lineno>``."""
    out: list[TrialRecord] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = TRIAL_RE.match(line)
        if not m:
            continue
        task, ctx, arm, seed, trial, hit, frac = m.groups()
        out.append(
            {
                "model": model,
                "arm": arm,
                "task": task,
                "ctx": int(ctx),
                "seed": int(seed),
                "trial": int(trial),
                "hit": int(hit),
                "frac": float(frac),
                "haystack_id": None,
                "depth": None,
                "prompt_sha256": None,
                "error": None,
                "source": f"{source}:{i}",
            }
        )
    return out


def parse_cell_lines(text: str, model: str, source: str) -> list[CellRecord]:
    """Every pooled ``[task ctxN] arm acc=...`` line as a record. ``n``/``hits`` are
    ``None`` for pre-Week-18 rows, which printed no ``n=`` (no Bernoulli count to
    recover -- an interval cannot be computed from them)."""
    out: list[CellRecord] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = CELL_RE.match(line)
        if not m:
            continue
        task, ctx, arm, acc, _recall, ratio, _sbits, n = m.groups()
        n_i = int(n) if n is not None else None
        out.append(
            {
                "model": model,
                "arm": arm,
                "task": task,
                "ctx": int(ctx),
                "acc": float(acc),
                "n": n_i,
                "hits": round(float(acc) * n_i) if n_i is not None else None,
                "ratio": float(ratio),
                "source": f"{source}:{i}",
            }
        )
    return out


def write_jsonl(path: Path, rows: list[TrialRecord] | list[CellRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
