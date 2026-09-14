"""Per-trial, per-cell and per-ppl records: the only unit `make tables` reads.

``[trial]`` lines are the Bernoulli outcomes the pods printed (one per task x ctx x arm
x seed x trial); ``[task ctxN] arm acc= ... n=`` lines are pooled cells; ``arm [T=ctx]
ppl=...`` lines are perplexity sweeps; ``[pplw]`` lines are the un-pooled per-window
NLLs behind one of those sweeps, and ``[diag]`` lines are JSON diagnostics. All the
regexes are the emitters' formats from Week 11/17/18/19 (previously duplicated in six
reader scripts: w10_parse_logs, w11_merge, w17_intervals, w18_intervals, w19_a2_misses,
w19_fork_report -- and, until the L0.5 fix round, in ``scripts/pod.py``).

A perplexity sweep leaves TWO artifacts, and they are different records in different
files: the aggregate ``PplRecord`` (``ppl.jsonl``) and the per-window ``PplwRecord``
(``pplw.jsonl``). Neither substitutes for the other.

``generator``/``haystack_id``/``depth``/``code_family``/``prompt_sha256``/``error`` are
carried in the schema but are ``None`` for the archived paper-v1 records: the v1
emitters never printed them, and the archive is not re-converted to invent them. Read
them with ``.get`` -- an archived row has the key absent, not null.

``generator`` is part of a cell's identity, not decoration: the in-house and official
RULER generators reuse the sub-task names ``niah_multivalue`` and ``vt`` at the same
context length, so a pod running both pools two different benchmarks into one key
unless the generator separates them (``scripts/pod.py``'s ``_expected_cells``).
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
# The emitter's own documented format (frontier._log_pplw, pinned by
# tests/test_w15_pplw.py): one line per (arm, T), except that a would-be >400-char line
# splits into ``part=i/N`` fragments of 8 values, because `vastai logs` truncates a line
# at ~500 chars. Dropping the fragments -- which is what the first harvest did -- loses
# the whole sweep silently.
PPLW_RE = re.compile(r"^\[pplw\] T=(\d+) (\S+) ntok=(\d+)(?: part=(\d+)/(\d+))? nlls=([0-9.,]+)$")
DIAG_RE = re.compile(r"^\[diag\] (\{.*\})\s*$")


class TrialRecord(TypedDict):
    model: str
    arm: str
    task: str
    ctx: int
    seed: int
    trial: int
    hit: int
    frac: float
    generator: str | None
    haystack_id: str | None
    depth: float | None
    code_family: str | None
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


class PplwRecord(TypedDict):
    """One scored window of a perplexity sweep.

    ``nll_sum_nats`` is the window's total NLL in nats: the emitter prints a per-token
    MEAN over ``ntok`` tokens, and the sum is the quantity that pools without carrying
    the weights around (``ppl == exp(sum(nll_sum_nats) / sum(ntok))``)."""

    model: str
    arm: str
    ctx: int
    window_idx: int
    ntok: int
    nll_sum_nats: float
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
                # A harvested log row carries only what the line printed; the generator
                # that produced it is not in the format, so it stays unknown here.
                "generator": None,
                "haystack_id": None,
                "depth": None,
                "code_family": None,
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


def parse_pplw_lines(text: str, model: str, source: str) -> list[PplwRecord]:
    """Every ``[pplw]`` window as a record, split lines reassembled in part order.

    Fragments are gathered per ``(arm, ctx)`` -- the emitter prints one group per
    ``(arm, T)`` -- and only emitted once every part of the group has arrived. An
    incomplete set raises ``SystemExit``: a truncated log is a harvest to redo, not a
    sweep to publish short. Every window of a group cites the line the group started on.
    """
    out: list[PplwRecord] = []
    pending: dict[tuple[str, int], tuple[int, int, int, dict[int, list[float]]]] = {}

    def emit(arm: str, ctx: int, ntok: int, line: int, vals: list[float]) -> None:
        out.extend(
            {
                "model": model,
                "arm": arm,
                "ctx": ctx,
                "window_idx": j,
                "ntok": ntok,
                "nll_sum_nats": v * ntok,
                "source": f"{source}:{line}",
            }
            for j, v in enumerate(vals)
        )

    for i, line in enumerate(text.splitlines(), 1):
        m = PPLW_RE.match(line)
        if not m:
            continue
        ctx, arm, ntok, part, n_parts, nlls = m.groups()
        vals = [float(x) for x in nlls.split(",") if x]
        if part is None:
            emit(arm, int(ctx), int(ntok), i, vals)
            continue
        key = (arm, int(ctx))
        n, tok, first, parts = pending.setdefault(key, (int(n_parts), int(ntok), i, {}))
        parts[int(part)] = vals
        if len(parts) == n:
            emit(arm, key[1], tok, first, [v for j in sorted(parts) for v in parts[j]])
            del pending[key]
    if pending:
        missing = {
            f"{arm} T={ctx}": sorted(set(range(1, n + 1)) - set(parts))
            for (arm, ctx), (n, _tok, _first, parts) in pending.items()
        }
        raise SystemExit(f"{source}: incomplete [pplw] part set, missing {missing}")
    return out


def parse_diag_lines(text: str, model: str, source: str) -> list[dict[str, object]]:
    """``[diag] {json}`` payloads, verbatim plus their ``model`` and ``source``. Nothing
    reads them yet -- the diagnostics land in L1 -- so they are carried through unparsed
    rather than dropped, but the model is stamped in like every other record type: a
    rank or an orthogonality number means nothing without the family it came from. A
    line whose payload is not JSON is skipped, not fatal: diagnostics are never evidence
    for a number."""
    out: list[dict[str, object]] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = DIAG_RE.match(line)
        if not m:
            continue
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        out.append({"model": model, **payload, "source": f"{source}:{i}"})
    return out


def write_jsonl(
    path: Path,
    rows: list[TrialRecord] | list[CellRecord] | list[PplRecord] | list[PplwRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
