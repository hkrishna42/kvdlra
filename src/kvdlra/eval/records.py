"""Per-trial, per-cell and per-ppl records: the only unit `make tables` reads.

``[trial]`` lines are the Bernoulli outcomes the pods printed (one per task x ctx x arm
x seed x trial); ``[task ctxN] arm acc= ... n=`` lines are pooled cells; ``arm [T=ctx]
ppl=...`` lines are perplexity sweeps. All three regexes are the emitters' formats from
Week 11/17/18/19 (previously duplicated in six reader scripts: w10_parse_logs,
w11_merge, w17_intervals, w18_intervals, w19_a2_misses, w19_fork_report).

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
# Leading whitespace varies (0 or 2 spaces) across pods; tok_eq/layer and sbits are
# each sometimes absent. Verified against every `ppl=` line in results/*-lines.txt
# (175/175 match) -- see results/w11-table-ppl-lines.txt (no leading space, no sbits),
# results/w17-qwen-lines.txt (2-space, no sbits) and results/w18-*-ppl-lines.txt
# (2-space, with sbits).
PPL_RE = re.compile(
    r"^\s*(\S+)\s+\[T=(\d+)\] ppl=([0-9.]+)(?: tok_eq/layer=([0-9.]+))? .*?ratio=([0-9.]+)"
    r"(?: sbits=([0-9.]+))?"
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
    sbits: float | None
    source: str


class PplRecord(TypedDict):
    model: str
    arm: str
    ctx: int
    ppl: float
    ratio: float
    sbits: float | None
    tok_eq: float | None
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
    recover -- an interval cannot be computed from them); ``sbits`` (fp32-at-rest
    stored bits, the memory convention half the v1 tables print) is ``None`` for the
    rows that printed no ``sbits=``."""
    out: list[CellRecord] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = CELL_RE.match(line)
        if not m:
            continue
        task, ctx, arm, acc, _recall, ratio, sbits, n = m.groups()
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
                "sbits": float(sbits) if sbits is not None else None,
                "source": f"{source}:{i}",
            }
        )
    return out


def parse_ppl_lines(text: str, model: str, source: str) -> list[PplRecord]:
    """Every perplexity line (``arm [T=ctx] ppl=... ratio=...``) as a record.
    ``tok_eq`` and ``sbits`` are ``None`` when the source line did not print them
    (pre-Week-18 sweeps never printed ``sbits=``)."""
    out: list[PplRecord] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = PPL_RE.match(line)
        if not m:
            continue
        arm, ctx, ppl, tok_eq, ratio, sbits = m.groups()
        out.append(
            {
                "model": model,
                "arm": arm,
                "ctx": int(ctx),
                "ppl": float(ppl),
                "ratio": float(ratio),
                "sbits": float(sbits) if sbits is not None else None,
                "tok_eq": float(tok_eq) if tok_eq is not None else None,
                "source": f"{source}:{i}",
            }
        )
    return out


def write_jsonl(path: Path, rows: list[TrialRecord] | list[CellRecord] | list[PplRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
