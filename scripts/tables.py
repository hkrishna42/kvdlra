"""Tables entrypoint. `convert-v1` archives the paper-v1 line files as JSONL records;
`build` (Task 3) regenerates every paper-v1 table from results/paper-v1/.

The archive is the evidence behind the v1 tables: one directory per pod holding a
verbatim `raw/` copy of every source file that named it, the parsed records
(`trials.jsonl` / `cells.jsonl` / `ppl.jsonl`, whichever formats that pod's sources
contain) and a `manifest.json` self-describing every source's line/parsed/unconverted
counts. A pod directory exists for every source file the converter reads -- even one
whose lines match no known format -- because completeness lives in the raw/ copy, not
in whether a row was parsed. It is written once, from the line files at tag
`paper-v1-archive`, and read from then on.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Literal, TypedDict, cast

import _paths  # noqa: F401

from kvdlra.accounting import bug_footprint
from kvdlra.eval import gate1
from kvdlra.eval.config import (
    ArmCfg,
    PodCfg,
    arm_kwargs,
    load_arm,
    load_pod,
    load_task,
    role_of,
)
from kvdlra.eval.records import (
    CellRecord,
    KernelCheckRecord,
    LatencyRecord,
    PplRecord,
    PplwRecord,
    TrialRecord,
    paired_window_bits,
    parse_cell_lines,
    parse_error_lines,
    parse_latency_lines,
    parse_ppl_lines,
    parse_trial_lines,
    read_jsonl,
    window_bits,
    write_jsonl,
)
from kvdlra.eval.stats import Key, McNemar, mcnemar_exact, paired_bootstrap, tost, wilson

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_BY_TAG = {  # the exact HF ids the Week-18/19 pods ran
    "llama": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}
NOTE = "paper-v1 archive; haystack_id/depth/prompt_sha256 were not recorded and are null"


def _tag(stem: str) -> str:
    for t in MODEL_BY_TAG:
        if t in stem:
            return t
    raise SystemExit(f"no model tag in {stem}")


def convert_v1(out_root: Path) -> None:
    """Archive every paper-v1 line file under ``out_root/<pod>/``.

    Per-trial sources first, then the aggregate ones. Globs are resolved from
    ``REPO_ROOT``, not the process cwd, so this runs the same from any directory.
    24 pod names carry both kinds of source file and 20 of them end up with a
    trials.jsonl AND a cells.jsonl under one merged manifest (the other 4 per-trial
    files hold storage tables, not [trial] lines -- they still get an archived raw/
    copy and a manifest, just no trials.jsonl).
    """
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "paper-v1-archive^{commit}"], text=True
    ).strip()
    for src in sorted((REPO_ROOT / "results" / "w18_pertrial").glob("*-trials.txt")):
        stem = src.name.removesuffix("-trials.txt")
        pod = f"w18-g1-{stem}" if stem in MODEL_BY_TAG else f"w18-{stem}"
        _emit(out_root / pod, src, sha, per_trial=True)
    for src in sorted((REPO_ROOT / "results" / "w19_pertrial").glob("*-trials.txt")):
        _emit(out_root / f"w19-{src.name.removesuffix('-trials.txt')}", src, sha, per_trial=True)
    for src in sorted((REPO_ROOT / "results").glob("*-lines.txt")):
        _emit(out_root / src.name.removesuffix("-lines.txt"), src, sha, per_trial=False)


def _emit(pod_dir: Path, src: Path, sha: str, per_trial: bool) -> None:
    """Archive one source file into ``pod_dir``: a verbatim ``raw/`` copy always, plus
    whichever of trials/cells/ppl its lines parse as, and a manifest entry recording
    that source's line/parsed/unconverted counts (merged with any sibling source's).

    Every (pod, artifact) pair in the real archive has exactly one contributing
    source (checked across all 98 source files); if a source ever collided with an
    existing artifact this raises rather than silently dropping the earlier rows.
    """
    pod_dir.mkdir(parents=True, exist_ok=True)
    (pod_dir / "raw").mkdir(exist_ok=True)
    shutil.copyfile(src, pod_dir / "raw" / src.name)

    try:
        cite = str(src.relative_to(REPO_ROOT))
    except ValueError:
        cite = str(src)  # e.g. a test's tmp_path source, outside the repo
    text = src.read_text()
    model = _model(src, per_trial)
    rows_by_artifact: dict[str, list[TrialRecord] | list[CellRecord] | list[PplRecord]] = {
        "trials.jsonl": parse_trial_lines(text, model, cite),
        "cells.jsonl": parse_cell_lines(text, model, cite),
        "ppl.jsonl": parse_ppl_lines(text, model, cite),
    }

    manifest_path = pod_dir / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records: dict[str, int] = dict(old.get("records", {}))
    for name, rows in rows_by_artifact.items():
        if not rows:
            continue
        path = pod_dir / name
        if path.exists():
            raise SystemExit(f"{pod_dir.name}/{name}: {cite} would clobber existing rows")
        write_jsonl(path, rows)
        records[name] = len(rows)

    n_lines = sum(1 for line in text.splitlines() if line.strip())
    parsed = {name.removesuffix(".jsonl"): len(rows) for name, rows in rows_by_artifact.items()}
    source_files = [e for e in old.get("source_files", []) if e["path"] != cite]
    source_files.append(
        {
            "path": cite,
            "lines": n_lines,
            "parsed": parsed,
            "unconverted": n_lines - sum(parsed.values()),
            "raw": f"raw/{src.name}",
        }
    )
    manifest_path.write_text(
        json.dumps(
            {
                "pod": pod_dir.name,
                "git_sha": sha,
                "source_files": sorted(source_files, key=lambda e: e["path"]),
                "converted_by": "scripts/tables.py convert-v1",
                "records": records,
                "note": NOTE,
            },
            indent=2,
        )
        + "\n"
    )


def _model(src: Path, per_trial: bool) -> str:
    """The HF id the pod ran. Per-trial evidence must be attributable, so ``_tag``
    raises for it; aggregate rows may be "unknown" (the pre-Week-16 files pool
    several pods and never named a model)."""
    if not per_trial and not any(t in src.stem for t in MODEL_BY_TAG):
        return "unknown"
    return MODEL_BY_TAG[_tag(src.stem)]


# --- build: regenerate the paper-v1 tables ------------------------------------
#
# Every cell is counted from per-trial records, never from a pooled `acc=` line: some
# pods printed the same (task, ctx, arm) cell twice, and the same (model, arm, task,
# ctx) key exists in several pods under different generators (e.g. Llama 16K vt for
# `bugSseed-r64-h256` in both w18-g1-llama and w19-a2-llama). Selection is therefore
# pod-scoped -- a pod is the provenance unit and identifies the model.
#
# The memory columns are the exception: stored state is a property of the run, not of
# a needle, so they come from the pods' archived aggregate rows -- see `memory()`.

ARCHIVE = REPO_ROOT / "results" / "paper-v1"
DISPLAY = {"qwen": "Qwen2.5-7B", "mistral": "Mistral-7B-v0.3", "llama": "Llama-3.1-8B"}
N_FEATURES = {  # num_key_value_heads x head_dim -- an architecture constant, not a result
    "unsloth/Meta-Llama-3.1-8B-Instruct": 1024,  # 8 kv heads x 128
    "mistralai/Mistral-7B-Instruct-v0.3": 1024,  # 8 kv heads x 128
    "Qwen/Qwen2.5-7B-Instruct": 512,  # 4 kv heads x 128
}
MEMORY_SOURCE = {
    # The w18-g1-* pods printed [trial] lines only; the `[task ctxN] ... ratio= sbits=`
    # rows of the same run went to results/w18-<model>-lines.txt, which the archive
    # holds under a pod name of its own. Every other pod these tables read carries its
    # own cells.jsonl, so `memory()` falls back to the pod itself.
    "w18-g1-llama": "w18-llama",
    "w18-g1-mistral": "w18-mistral",
    "w18-g1-qwen": "w18-qwen",
}
TASKS = ("niah_single", "niah_multikey", "niah_multivalue", "vt")
OFFICIAL = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
)
K16, K32 = 16384, 32768
ALPHA = 0.05
R64, R128 = "bugSseed-r64-h256", "bugSseed-r128-h1024-s32"
KIVI = ("quant-2bit-kivi", "quant-4bit-kivi")
FP16_MEM = (
    "stored state = the arm's float-equivalent ratio (`ratio=`), the convention v1's"
    " caption states for this table"
)
BITS_MEM = (
    "stored = the arm's fp32-at-rest stored bits (`sbits=`), the convention v1's"
    " caption states for this table"
)
MEM_RULE = (
    "a memory value is the mean of the arm's archived cell rows, which agree to within"
    " one 0.001 print unit (stored state is a property of the run, not of a needle)"
)


def _read(pod: str, name: str) -> list[dict[str, object]]:
    path = ARCHIVE / pod / name
    if not path.is_file():
        raise SystemExit(f"no such archived artifact: {path}")
    return read_jsonl(path)


@cache
def _pod(pod: str) -> tuple[TrialRecord, ...]:
    """Every per-trial record of one pod (cached: each pod is read once per build)."""
    return tuple(cast(TrialRecord, r) for r in _read(pod, "trials.jsonl"))


@cache
def _cells(pod: str) -> tuple[CellRecord, ...]:
    """Every archived aggregate row of one pod -- the only source of memory columns."""
    return tuple(cast(CellRecord, r) for r in _read(pod, "cells.jsonl"))


def _keyed(pod: str, arm: str, task: str, ctx: int) -> dict[Key, int]:
    """One cell's Bernoulli outcomes, keyed by (seed, trial) so cells can be paired."""
    rows = [r for r in _pod(pod) if r["arm"] == arm and r["task"] == task and r["ctx"] == ctx]
    hits = {(r["seed"], r["trial"]): r["hit"] for r in rows}
    if not hits:
        raise SystemExit(f"no records: {pod} {arm} {task} ctx={ctx}")
    if len(hits) != len(rows):  # w19-fork-llama repeats needles; keying would silently drop them
        raise SystemExit(f"duplicate (seed,trial): {pod} {arm} {task} ctx={ctx}")
    return hits


def _count(pod: str, arm: str, task: str, ctx: int) -> tuple[int, int]:
    d = _keyed(pod, arm, task, ctx)
    return sum(d.values()), len(d)


def memory(pod: str, arm: str, ctx: int, kind: Literal["ratio", "sbits"]) -> float:
    """One arm's stored state, from the archived aggregate rows of ``MEMORY_SOURCE[pod]``.

    ``kind`` is the convention the table's v1 caption names: ``ratio`` is
    float-equivalent, ``sbits`` is fp32-at-rest stored bits. The pods printed one row
    per task, each a 3-decimal print of the same run-level quantity (they differ by at
    most one print unit, from prompt-length jitter), so the arm's value is their mean
    and a wider spread is a selection bug -- fail loud. Because the rows come from a
    *different* pod name for the w18-g1-* tables, every row is also checked against the
    per-trial records it claims to summarise.
    """
    src = MEMORY_SOURCE.get(pod, pod)
    rows = [r for r in _cells(src) if r["arm"] == arm and r["ctx"] == ctx]
    vals = [r["ratio"] if kind == "ratio" else r["sbits"] for r in rows]
    if not rows or None in vals:
        raise SystemExit(f"no {kind}= aggregate row: {src} {arm} ctx={ctx}")
    v = cast(list[float], vals)
    if max(v) - min(v) > 1e-3 + 1e-9:
        raise SystemExit(f"{kind} spans {min(v)}..{max(v)} across tasks: {src} {arm} ctx={ctx}")
    for r in rows:
        h, n = _count_or_none(pod, arm, r["task"], ctx)
        if n and abs(h / n - r["acc"]) > 0.005:
            raise SystemExit(
                f"{src} aggregate row is not {pod}'s run: {arm} {r['task']} ctx={ctx}"
                f" acc={r['acc']} but per-trial {h}/{n}"
            )
    return sum(v) / len(v)


def _count_or_none(pod: str, arm: str, task: str, ctx: int) -> tuple[int, int]:
    """``(hits, n)`` from the per-trial pod, or ``(0, 0)`` when it holds no such cell
    (the aggregate sources carry rows for arms and tasks the tables never print)."""
    rows = [r for r in _pod(pod) if r["arm"] == arm and r["task"] == task and r["ctx"] == ctx]
    return sum(r["hit"] for r in rows), len(rows)


def cell(pod: str, arm: str, task: str, ctx: int) -> str:
    """`acc [Wilson 95% lo,hi] (hits/n)`."""
    h, n = _count(pod, arm, task, ctx)
    lo, hi = wilson(h, n)
    return f"{h / n:.2f} [{lo:.2f},{hi:.2f}] ({h}/{n})"


def acc(pod: str, arm: str, task: str, ctx: int) -> str:
    """`acc (hits/n)` -- the tables v1 printed without intervals."""
    h, n = _count(pod, arm, task, ctx)
    return f"{h / n:.2f} ({h}/{n})"


def paired(pod_a: str, arm_a: str, pod_b: str, arm_b: str, task: str, ctx: int) -> McNemar:
    """Exact paired McNemar over the (seed, trial) keys the two arms share."""
    m = mcnemar_exact(_keyed(pod_a, arm_a, task, ctx), _keyed(pod_b, arm_b, task, ctx))
    if m is None:
        raise SystemExit(f"no shared needles: {arm_a} vs {arm_b} {task} ctx={ctx}")
    return m


def _favors_a(m: McNemar) -> bool:
    return m["a_favored"] > m["b_favored"] and m["p_value"] < ALPHA


def _favors_b(m: McNemar) -> bool:
    return m["b_favored"] > m["a_favored"] and m["p_value"] < ALPHA


# Tables 3 and 7 pair arms that ran on DIFFERENT pods. Pairing on (seed, trial) is only
# valid if both pods built the same prompt for a given key; the generator is
# deterministic and decoding is greedy, so they should have -- but the v1 records carry
# no prompt_sha256, so nothing in the archive proves it. Said once, cited twice.
CROSS_POD = (
    "pairing on (seed,trial) assumes both pods built the same prompt for a given key --"
    " which a deterministic generator under greedy decode does, but prompt_sha256 is null"
    " in the v1 records, so the archive cannot verify it"
)


def _tex(s: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s.replace("_", r"\_"))


def _md_rows(header: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    """A markdown table -- a blank line, the header, the rule, one line per row. The one
    emitter: every table this file writes (the numbered ones, the perplexity table and
    Gate 1's blocks) is this shape, and the paper-v1 golden pins it byte for byte."""
    return [
        "",
        "| " + " | ".join(header) + " |",
        "|" + " --- |" * len(header),
        *["| " + " | ".join(r) + " |" for r in body],
    ]


def _table(
    n: int | str,
    title: str,
    notes: Sequence[str],
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> tuple[str, str]:
    """One table as (markdown, latex). The markdown is the contract the golden pins;
    the .tex mirrors it. No blank line closes a file: the next table's `##` heading
    follows the last row directly, so `cat table*.md` is valid markdown."""
    md = [f"## Table {n} — {title}"]
    md += [f"<!-- {x} -->" for x in notes]
    md += _md_rows(header, rows)
    tex = [f"% Table {n} -- {title}"] + [f"% {x}" for x in notes]
    tex += [
        "\\begin{tabular}{" + "l" + "c" * (len(header) - 1) + "}",
        "\\toprule",
        " & ".join(_tex(h) for h in header) + " \\\\",
        "\\midrule",
    ]
    tex += [" & ".join(_tex(c) for c in r) + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


def _xmodel(
    n: int, ctx: int, label: str, pdf_n: int, mk: str, trunc: str, feat: bool = False
) -> tuple[str, str]:
    """Tables 1 and 2: the same three families under `bugSseed-r64-h256`, 16K and 32K."""
    notes = [
        f"paper-v1: paper/main.tex @ee8c0ab, table `{label}` (renders as Table {pdf_n} in the PDF)",
        "source: results/paper-v1/w18-g1-{qwen,mistral,llama}/trials.jsonl (cells),"
        " results/paper-v1/w18-{qwen,mistral,llama}/cells.jsonl (stored state)",
        "cell: acc [Wilson 95% lo,hi] (hits/n)",
        trunc,
        FP16_MEM,
        MEM_RULE,
    ]
    if feat:
        notes.append(
            "feat. n = num_key_value_heads x head_dim, a model constant, not a measurement"
        )
    rows = []
    for t in ("qwen", "mistral", "llama"):
        pod = f"w18-g1-{t}"
        row = [DISPLAY[t]]
        if feat:
            row.append(str(N_FEATURES[MODEL_BY_TAG[t]]))
        row += [cell(pod, R64, task, ctx) for task in TASKS]
        row.append(f"{memory(pod, R64, ctx, 'ratio'):.3f}x")
        rows.append(row)
    head = ["model"] + (["feat. n"] if feat else [])
    head += ["single", mk, "multi-value", "var-track", "stored state"]
    title = f"cross-model {ctx // 1024}K retrieval, config `{R64}`"
    return _table(n, title, notes, head, rows)


def table_1() -> tuple[str, str]:
    return _xmodel(
        1,
        K16,
        "tab:xmodel",
        2,
        "multi-key (not in v1)",
        "v1 printed the Llama var-track upper bound as 0.80 (truncated); correct rounding 0.81",
        feat=True,
    )


def table_2() -> tuple[str, str]:
    return _xmodel(
        2,
        K32,
        "tab:xmodel32",
        3,
        "multi-key",
        "v1 printed the Llama var-track upper bound as 0.98 (truncated); correct rounding 0.99",
    )


def table_3() -> tuple[str, str]:
    """32K variable-tracking on Llama: the r128 config against the three baselines it
    was contrasted with, plus the r/n=0.25 control column v1 discussed only in prose."""
    g4, a1, r256 = "w18-g4-llama", "w19-a1-llama", "bugSseed-r256-h1024"
    q4 = paired(g4, R128, a1, "quant-4bit-kivi", "vt", K32)
    notes = [
        "paper-v1: paper/main.tex @ee8c0ab, the 32K Llama variable-tracking table (its LaTeX"
        " label carries a word CLAUDE.md bans from new files, so it is not quoted here; it"
        " renders as Table 4 in the PDF)",
        f"source: results/paper-v1/{g4}/trials.jsonl (the n={_count(g4, R128, 'vt', K32)[1]} arms"
        f" and the n={_count(g4, r256, 'vt', K32)[1]} r256 control),"
        f" results/paper-v1/{a1}/trials.jsonl (quant-4bit-kivi,"
        f" n={_count(a1, 'quant-4bit-kivi', 'vt', K32)[1]}); stored state from each pod's"
        " own cells.jsonl",
        "cell: acc [Wilson 95% lo,hi] (hits/n); p = exact paired McNemar vs"
        f" {R128} on the shared (seed,trial) keys, 2 significant figures",
        f"discordant = pairs won by {R128} / pairs won by the row's arm",
        "v1 printed no McNemar p for the 4-bit row; it is computed here on the"
        f" {q4['n_paired']} shared keys",
        f"the quant-4bit-kivi row is a different pod ({a1}) from every other row ({g4}):"
        f" {CROSS_POD}",
        BITS_MEM.replace("stored =", "stored state ="),
        MEM_RULE,
        "v1 printed think-c0.5/palu-r0.5 to 2 decimals (0.75x/0.50x); the archived rows are"
        " printed here at the 3 decimals the other rows need",
        "the verdict column of v1 is omitted: it is an editorial reading, not a statistic",
    ]
    rows = [
        [
            R128,
            cell(g4, R128, "vt", K32),
            f"{memory(g4, R128, K32, 'sbits'):.3f}x",
            "---",
            "---",
            "---",
        ]
    ]
    for label, pod, arm in (
        ("think-c0.5", g4, "think-c0.5"),
        ("palu-r0.5", g4, "palu-r0.5"),
        ("quant-4bit-kivi", a1, "quant-4bit-kivi"),
        (f"{r256} (not in v1)", g4, r256),
    ):
        m = paired(g4, R128, pod, arm, "vt", K32)
        rows.append(
            [
                label,
                cell(pod, arm, "vt", K32),
                f"{memory(pod, arm, K32, 'sbits'):.3f}x",
                f"{m['p_value']:.1e}",
                f"{m['a_favored']}/{m['b_favored']}",
                str(m["n_paired"]),
            ]
        )
    return _table(
        3,
        f"32K variable-tracking on Llama-3.1-8B, config `{R128}` vs baselines",
        notes,
        ["config", "var-track", "stored state", "McNemar p", "discordant", "n paired"],
        rows,
    )


def table_6() -> tuple[str, str]:
    """Eviction (`ea-k0.1`) on the model x ctx cells v1 printed."""
    cells = (("llama", K16), ("llama", K32), ("qwen", K16), ("mistral", K16))
    budgets = {f"{memory(f'w18-g3-{t}', 'ea-k0.1', ctx, 'ratio'):.3f}x" for t, ctx in cells}
    if len(budgets) != 1:  # v1's caption states one budget for the whole table
        raise SystemExit(f"ea-k0.1 stored state differs across rows: {sorted(budgets)}")
    (budget,) = budgets
    notes = [
        "paper-v1: paper/main.tex @ee8c0ab, table `tab:evict` (renders as Table 7 in the PDF)",
        "source: results/paper-v1/w18-g3-{llama,qwen,mistral}/trials.jsonl (cells),"
        " and their cells.jsonl (the budget in the title)",
        "cell: acc [Wilson 95% lo,hi] (hits/n)",
        "rows are the model x ctx cells v1 showed; the pods also hold Qwen/Mistral 32K,"
        " which v1 did not print",
        "v1's table has no memory column: its budget is stated once in the caption, and is"
        f" regenerated in the title above from the `ratio=` rows of all {len(cells)} cells",
    ]
    rows = [
        [DISPLAY[t], str(ctx), *(cell(f"w18-g3-{t}", "ea-k0.1", task, ctx) for task in TASKS)]
        for t, ctx in cells
    ]
    return _table(
        6,
        f"eviction at {budget} stored state, arm `ea-k0.1`",
        notes,
        ["model", "ctx", "single", "multi-key", "multi-value", "var-track"],
        rows,
    )


def table_7() -> tuple[str, str]:
    """The r64 config against the 2-bit and 4-bit KIVI arms at matched stored bytes."""
    notes = [
        "paper-v1: paper/main.tex @ee8c0ab, table `tab:fairquant` (renders as Table 8 in the PDF)",
        "source: results/paper-v1/w18-g1-{llama,mistral,qwen}/trials.jsonl (r64 rows),"
        " results/paper-v1/w19-a1-{llama,mistral,qwen}/trials.jsonl (KIVI rows);"
        " stored bits from results/paper-v1/w18-{llama,mistral,qwen}/cells.jsonl (r64)"
        " and each w19-a1 pod's own cells.jsonl (KIVI)",
        "cell: acc (hits/n)",
        "bold = exact paired McNemar p<0.05 in the r64 arm's favour against a KIVI arm of the"
        " same model x ctx x task, paired on (seed,trial); no cell is significant in a KIVI"
        " arm's favour",
        f"the r64 rows and the KIVI rows are different pods (w18-g1-<model> vs"
        f" w19-a1-<model>): {CROSS_POD}",
        BITS_MEM,
        MEM_RULE,
    ]
    rows: list[list[str]] = []
    kivi_wins: list[str] = []  # the note above is a claim; this is what checks it
    for t in ("llama", "mistral", "qwen"):
        g1, a1 = f"w18-g1-{t}", f"w19-a1-{t}"
        for ctx in (K16, K32):
            base: list[str] = []
            for task in TASKS:
                c = acc(g1, R64, task, ctx)
                ms = [paired(g1, R64, a1, q, task, ctx) for q in KIVI]
                kivi_wins += [
                    f"{t} ctx={ctx} {task} {q}"
                    for q, m in zip(KIVI, ms, strict=True)
                    if _favors_b(m)
                ]
                base.append(f"**{c}**" if any(_favors_a(m) for m in ms) else c)
            rows.append([DISPLAY[t], str(ctx), R64, f"{memory(g1, R64, ctx, 'sbits'):.3f}x", *base])
            rows += [
                [
                    DISPLAY[t],
                    str(ctx),
                    q,
                    f"{memory(a1, q, ctx, 'sbits'):.3f}x",
                    *(acc(a1, q, task, ctx) for task in TASKS),
                ]
                for q in KIVI
            ]
    if kivi_wins:  # never let the note above become a claim the records stopped backing
        raise SystemExit(f"a KIVI arm significantly beats {R64} in: {kivi_wins}")
    return _table(
        7,
        "the 2-bit/4-bit KIVI baseline at matched stored bytes",
        notes,
        ["model", "ctx", "arm", "stored", "single", "multi-key", "multi-value", "var-track"],
        rows,
    )


def table_8() -> tuple[str, str]:
    """Official NVIDIA RULER at 16K on Llama: nine tasks, eight arms, 12 records each."""
    a2, q4 = "w19-a2-llama", "w19-q4off-llama"
    pooled = [_count(a2, R64, task, K16) for task in OFFICIAL]
    notes = [
        "paper-v1: paper/main.tex @ee8c0ab, table `tab:official` (renders as Table 9 in the PDF)",
        f"source: results/paper-v1/{a2}/trials.jsonl (7 arms),"
        f" results/paper-v1/{q4}/trials.jsonl (the q4 cell arm); stored bits from the same"
        " two pods' cells.jsonl",
        "cell: acc (hits/n); mean = mean of the nine printed 2-dp accuracies, as in v1"
        f" (pooling the records instead gives"
        f" {sum(h for h, _ in pooled) / sum(n for _, n in pooled):.2f} for {R64})",
        BITS_MEM,
        MEM_RULE,
    ]
    rows: list[list[str]] = []
    for arm, pod in (
        ("full", a2),
        ("think-c0.5", a2),
        ("palu-r0.5", a2),
        ("quant-4bit-kivi", a2),
        ("quant-2bit-kivi", a2),
        (R64, a2),
        (f"{R64}-q4", q4),
        ("ea-k0.1", a2),
    ):
        counts = [_count(pod, arm, task, K16) for task in OFFICIAL]
        mean = sum(round(h / n, 2) for h, n in counts) / len(counts)
        rows.append(
            [
                arm,
                f"{memory(pod, arm, K16, 'sbits'):.2f}x",
                *(f"{h / n:.2f} ({h}/{n})" for h, n in counts),
                f"{mean:.2f}",
            ]
        )
    head = ["arm", "stored", "s1", "s2", "s3", "mk1", "mk2", "mk3", "mv", "mq", "vt", "mean"]
    return _table(8, "official NVIDIA RULER at 16K on Llama-3.1-8B", notes, head, rows)


def table_baselines() -> tuple[str, str]:
    """The two v1 baseline rows the paper omitted (docs/plan/CODE_AUDIT.md Part B, Q4/Q5):
    ShadowKV after the Week-15 attach-scope fix, and plain eviction at 0.25x. Both are
    pre-Week-18 aggregate rows -- no per-trial records, so no hits/n and no interval: the
    archived point estimates are printed as they are, and a row that DID carry an n would
    belong in `cell()` with the others, so one here is refused."""
    src = (("w15-confirm", "shadow-r64"), ("w11-goalA-ruler", "ea-k0.25"))
    rows = []
    for pod, arm in src:
        found = [r for r in _cells(pod) if r["arm"] == arm and r["ctx"] == K16]
        cells = {r["task"]: r for r in found}
        if not cells:
            raise SystemExit(f"no aggregate rows: {pod} {arm} ctx={K16}")
        if len(cells) != len(found):  # keying by task would silently keep the last one
            raise SystemExit(f"duplicate (arm, task, ctx) row: {pod} {arm} ctx={K16}")
        if any(r["n"] is not None or r["hits"] is not None for r in cells.values()):
            raise SystemExit(f"{pod} {arm} carries hits/n: count it from records, not here")
        ratios = {r["ratio"] for r in cells.values()}
        if len(ratios) != 1 or None in ratios:
            raise SystemExit(f"no single ratio= across tasks: {pod} {arm} {ratios}")
        (ratio,) = cast(set[float], ratios)
        rows.append(
            [
                arm,
                f"{ratio:.3f}x",
                *(f"{cells[t]['acc']:.2f}" if t in cells else "---" for t in TASKS),
                "---",
                pod,
            ]
        )
    notes = [
        "not in paper-v1: the two competitive baseline rows the v1 tables omitted"
        " (docs/plan/CODE_AUDIT.md Part B, Q4 and Q5; PR-L2-13)",
        "source: results/paper-v1/w15-confirm/cells.jsonl (shadow-r64, the post-fix ShadowKV"
        " re-measure) and results/paper-v1/w11-goalA-ruler/cells.jsonl (ea-k0.25), the"
        " archived `[task ctx16384] arm acc= ratio=` rows; 16K, in-house generator",
        "cell: the archived acc, a point estimate -- these pre-Week-18 rows carry no per-trial"
        " records, so n is unknown (shown ---) and no Wilson interval is printed; a task the"
        " pod did not run is --- (shadow-r64 ran no multi-value)",
        "model: the archive rows record `unknown` (the pre-Week-16 line files named no"
        " model); CODE_AUDIT attributes both runs to Llama-3.1-8B",
        "stored state = the archived `ratio=` (float-equivalent); neither method holds"
        " fp32-at-rest state, so the stored-bits convention gives the same number; one value"
        " per row, shared by its task rows (checked)",
    ]
    return _table(
        "B",
        "the v1 baseline rows the paper omitted: ShadowKV post-fix and eviction at 0.25x",
        notes,
        ["arm", "stored state", "single", "multi-key", "multi-value", "var-track", "n", "source"],
        rows,
    )


# `_baselines` sorts after every digit in the `table*.md` glob the Makefile cats, so the
# appended block of docs/plan/paper-v1-tables.md is where the build puts it.
TABLES: dict[int | str, Callable[[], tuple[str, str]]] = {
    1: table_1,
    2: table_2,
    3: table_3,
    6: table_6,
    7: table_7,
    8: table_8,
    "_baselines": table_baselines,
}


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for n, fn in TABLES.items():
        md, tex = fn()
        (out / f"table{n}.md").write_text(md)
        (out / f"table{n}.tex").write_text(tex)


# ------------------------------------------------------------------ perplexity


class PplStat(TypedDict):
    """One arm's perplexity at one context length, on its STORED representation.

    ``bits`` is bits/token; ``d_bits`` its paired difference from the baseline arm with
    a bootstrap 95% CI, and ``p_tost`` the equivalence test against the margin. The four
    are ``None`` on the baseline row itself -- it has nothing to be paired against.
    """

    arm: str
    ctx: int
    corpus: str | None
    n_windows: int
    bits: float
    d_bits: float | None
    lo: float | None
    hi: float | None
    p_tost: float | None
    equivalent: bool | None


def ppl_stats(
    rows: Sequence[PplwRecord], baseline: str = "full", delta: float = 0.05
) -> list[PplStat]:
    """Per-window NLL rows -> bits/token per (arm, ctx, corpus), paired against
    ``baseline`` within the same corpus.

    Windows are paired by ``window_idx`` -- every arm scored the same slices of the same
    corpus, and the per-window spread across a corpus dwarfs the difference between two
    arms, so an unpaired comparison of pooled numbers hides the effect it is measuring.
    Both the interval and TOST run on those per-window differences, never on the pooled
    value: pooling first throws away the pairing and leaves one number with no spread.

    Keyed by ``(arm, ctx, corpus)``, not just ``(arm, ctx)``: two ppl tasks can share a
    ctx with different corpora (PG-19 validation + WikiText-103 test both ship a 16K
    task), and pairing across them would average two different texts' perplexity into
    one number instead of keeping each corpus's own comparison intact.
    """
    # `kvdlra.eval.gate1` starts from the same two functions, so both refuse the same
    # records. This entrypoint reports a bad file as a SystemExit, not a traceback.
    try:
        bits = window_bits(rows, lambda r: (r["arm"], r["ctx"], r.get("corpus")), "ppl")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    out: list[PplStat] = []
    groups = sorted({(c, corpus) for _, c, corpus in bits}, key=lambda x: (x[0], x[1] or ""))
    for ctx, corpus in groups:
        arms = sorted(a for a, c, cp in bits if c == ctx and cp == corpus)
        if baseline not in arms:
            raise SystemExit(
                f"ppl: no {baseline!r} arm at ctx={ctx} corpus={corpus} -- nothing to pair against"
            )
        base = bits[(baseline, ctx, corpus)]
        for arm in [baseline] + [a for a in arms if a != baseline]:
            w = bits[(arm, ctx, corpus)]
            # The baseline row has nothing to be paired against, so its five comparison
            # fields stay None; every other arm fills them in on the same row.
            stat: PplStat = {
                "arm": arm,
                "ctx": ctx,
                "corpus": corpus,
                "n_windows": len(w),
                "bits": sum(w.values()) / len(w),
                "d_bits": None,
                "lo": None,
                "hi": None,
                "p_tost": None,
                "equivalent": None,
            }
            if arm != baseline:
                try:
                    d = paired_window_bits(
                        w, base, f"ppl: {arm} ctx={ctx} corpus={corpus}", baseline
                    )
                except ValueError as exc:
                    raise SystemExit(str(exc)) from exc
                mean_d, lo, hi = paired_bootstrap(d)
                p_lo, p_hi, equivalent = tost(d, delta)
                stat |= {
                    "d_bits": mean_d,
                    "lo": lo,
                    "hi": hi,
                    "p_tost": max(p_lo, p_hi),
                    "equivalent": equivalent,
                }
            out.append(stat)
    return out


def ppl_table(results: Path, out: Path, baseline: str = "full", delta: float = 0.05) -> None:
    """The perplexity table for one pod's run directory, written as markdown.

    NOT a numbered `TABLES` entry and never named ``table*.md``: `make tables` concatenates
    those and diffs the result against the frozen paper-v1 golden, which this is not part
    of. It reads `pplw.jsonl` -- the per-window rows -- because the pooled `ppl.jsonl`
    number cannot produce an interval.
    """
    src = results / "pplw.jsonl"
    if not src.is_file():
        raise SystemExit(f"ppl: no per-window records at {src}")
    rows = [cast(PplwRecord, r) for r in read_jsonl(src)]
    if not rows:
        raise SystemExit(f"ppl: no per-window rows in {src}")
    corpora = sorted({str(r.get("corpus")) for r in rows})
    stats = ppl_stats(rows, baseline, delta)
    try:  # cite the repo-relative path, as `_emit` does -- a local absolute one cites nothing
        cite = str(src.relative_to(REPO_ROOT))
    except ValueError:
        cite = str(src)
    notes = [
        f"source: {cite} ({len(rows)} per-window rows; corpus: {', '.join(corpora)})",
        "cell: bits/token = mean over windows of nll_sum_nats / (ntok * ln 2)",
        f"delta: paired per window against `{baseline}`, mean [bootstrap 95% CI, 10k resamples]",
        f"TOST: two one-sided t-tests at +/-{delta} bits/token on the per-window differences;"
        " p is the larger one-sided p-value, equivalent at p < 0.05",
    ]
    header = [
        "arm",
        "ctx",
        "corpus",
        "windows",
        "bits/token",
        "delta bits [95% CI]",
        "TOST p",
        "equivalent",
    ]
    body = []
    for s in stats:
        d, lo, hi, p = s["d_bits"], s["lo"], s["hi"], s["p_tost"]
        body.append(
            [
                s["arm"],
                str(s["ctx"]),
                str(s["corpus"]),
                str(s["n_windows"]),
                f"{s['bits']:.4f}",
                "--" if d is None else f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]",
                "--" if p is None else f"{p:.3g}",
                "--" if s["equivalent"] is None else ("yes" if s["equivalent"] else "no"),
            ]
        )
    md = [f"## Perplexity — {results.name}"]
    md += [f"<!-- {x} -->" for x in notes]
    md += _md_rows(header, body)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")


# ---------------------------------------------------------------------- Gate 1

GATE1_PREREG = "prereg/gate1_tracker_swap_v2.md"


GATE1_LEGEND = (
    "Legend: `*` = Holm-significant primary contrast favouring isvd; `‡` = one favouring"
    " the control (rule 3 (i) reads a separation in either direction); a cell with error"
    " rows reads `FAILED (k errors)` and an arm that never ran that cell says so -- no"
    " arm is ever printed as `--`."
)


def _gate1_cell(data: gate1.Gate1Data, key: gate1.CellKey, mark: str) -> str:
    """One retrieval cell: `acc [Wilson 95%] (hits/n)`, or the failure that replaced it.

    "No arm is ever printed as `--`" (prereg section 4): a cell with error rows prints
    `FAILED (k errors)` and a cell that never ran prints `not run` WITH what is missing
    -- "an arm that never ran reads `not run` with the reason". `mark` is `*` or `‡`.
    """
    family, ctx, _, tracker = key
    if errs := data.errors.get(key):
        return f"FAILED ({len(errs)} errors)"
    outcomes = data.hits.get(key)
    if not outcomes:
        return f"not run (no records for {data.arms[(family, tracker)]} at ctx {ctx})"
    h, n = sum(outcomes.values()), len(outcomes)
    lo, hi = wilson(h, n)
    return f"{h / n:.2f} [{lo:.2f},{hi:.2f}] ({h}/{n})" + (f" {mark}" if mark else "")


def _gate1_md(title: str, header: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    """One Gate-1 block: its heading, then :func:`_md_rows`."""
    return ["", title, *_md_rows(header, body)]


def _gate1_ppl_cells(c: gate1.PplContrast) -> list[str]:
    """One perplexity contrast's five comparison cells.

    The TOST has three states, not two (prereg section 6): a member whose realised
    spread cannot fit inside the margin even at delta = 0 "is recorded as `not
    decidable` -- never as a pass, never as a quiet fail". The Holm p of an arm outside
    the 4-member primary family is not missing either: that family is the two contrasts
    the rule reads, and the rest are "uncorrected and descriptive" (section 7 (a)). A
    PRIMARY member without one is the other case: its family was refused, so it left
    the correction before Holm ran (ruling R-L3-16) and is printed as excluded rather
    than with an adjusted p it never had.
    """
    excluded = "refused (excluded from the Holm family)" if c.primary else "n/a (secondary)"
    return [
        f"{c.d_bits:+.4f}",
        f"[{c.lo:+.4f}, {c.hi:+.4f}]",
        "passes" if c.equivalent else ("fails" if c.decidable else "not decidable"),
        f"{c.p_holm:.3g}" if c.p_holm is not None else excluded,
        f"{c.p:.3g}",
    ]


BF16_LEGEND = (
    "Legend (bf16): cell = `delta [a_favored/b_favored of n_paired] Holm p`, with"
    " a = `isvd_r64_h256_seed` and b = `isvd_r64_h256_seed_bf16`, so `delta > 0` is the"
    " bf16 arm losing pairs; a task is non-inferior unless `delta` > 0.03 AND its Holm p <"
    " 0.05, both (§4 (1)). `sbits` is §7 (c)'s pin: outside 1 % of the expected ratio"
    " refuses the family."
)


def _bf16_cell(c: gate1.Contrast | None) -> str:
    """One bf16 member, or the state that replaced it. Nothing prints as `--` here
    either: a cell the pod never wrote says `not run` and one with error rows says so,
    because both are refusals of the family and the block is read alone."""
    if c is None:
        return "not run"
    if c.errors_a or c.errors_b:
        return f"FAILED ({c.errors_a + c.errors_b} errors)"
    if c.p is None or not c.n_paired:
        return f"no pairing ({len(c.dropped_keys)} keys dropped)"
    delta = (c.a_favored - c.b_favored) / c.n_paired
    # `bf16_contrasts`'s own eligibility is exactly the three conditions already ruled
    # out above (no error, a pairing, n_paired > 0), so a member reaching here always
    # carries an adjusted p -- there is no fourth state left to print.
    assert c.p_holm is not None, "an eligible bf16 member always carries an adjusted p"
    return f"{delta:+.3f} [{c.a_favored}/{c.b_favored} of {c.n_paired}] {c.p_holm:.3g}"


def _gate1_bf16_block(
    data: gate1.Gate1Data,
    contrasts: Sequence[gate1.Contrast],
    ppl: Sequence[gate1.PplContrast],
    verdict: gate1.Bf16Verdict,
) -> list[str]:
    """`prereg/bf16_gist.md` §4's retrieval reading, rendered by the code that renders
    the gate -- the reason it exists: §4 pre-registered it as a snippet run by hand, and
    a reading run by hand is a reading a harvest can skip.

    One row per (ctx, family): the four per-task members, the family's ±0.02 perplexity
    TOST (§4 (2)), §7 (c)'s stored-bits ratio and the family's verdict. Only the ctx
    §4 reads carries a verdict; any other context length is descriptive (§6), and the
    arm's own cells are already in the Gate-1 retrieval block above at every ctx.

    The guard below is "no bf16 records at all", not "the verdict is `not run`": a pod
    that carries ONLY a 32K bf16 reading (no 16K arm 6 at all, so the verdict reads
    `not run` -- §4 scopes it to 16K) still has rows to print, descriptively, and must
    not render nothing beside a `BF16: not run` line that says otherwise.
    """
    if not contrasts:
        return []
    idx = {(c.family, c.ctx, c.task): c for c in contrasts}
    seen = {c.task for c in contrasts}
    tasks = [t for t in gate1.TASK_ORDER if t in seen] + sorted(seen - set(gate1.TASK_ORDER))
    rows = []
    for ctx, family in sorted({(c.ctx, c.family) for c in contrasts}):
        tosts = [t for t in ppl if (t.family, t.ctx, t.b) == (family, ctx, gate1.BF16)]
        a_s, b_s = data.sbits.get((family, gate1.REFERENCE)), data.sbits.get((family, gate1.BF16))
        rows.append(
            [family, str(ctx)]
            + [_bf16_cell(idx.get((family, ctx, t))) for t in tasks]
            + [
                "; ".join(
                    ("passes" if t.equivalent else "fails" if t.decidable else "not decidable")
                    + f" ({t.d_bits:+.4f})"
                    for t in tosts
                )
                or "not run",
                f"{b_s / a_s:.4f}" if a_s and b_s else "not measured",
                verdict.families.get(family, "not run")
                if ctx == gate1.VERDICT_CTX
                else f"descriptive (§4 reads ctx {gate1.VERDICT_CTX})",
            ]
        )
    realised = sum(1 for c in contrasts if c.ctx == gate1.VERDICT_CTX and c.p_holm is not None)
    md = [
        "",
        "## bf16 non-inferiority — the gist stored at 16 bits",
        f"<!-- pre-registration: {gate1.BF16_PREREG}; the reading is its §4, and no member"
        " of it enters a Gate-1 family or moves a Gate-1 p-value (its §6) -->",
        f"<!-- Holm at alpha={gate1.ALPHA} inside this file's own retrieval family, at the"
        f" realised m={realised} of the {gate1.bf16_family_size()} members §6 fixes"
        " (2 model families x the tasks left after the pre-flight's exclusions) -->",
        f"<!-- delta: {gate1.REFERENCE} MINUS bf16, so delta > 0 is the bf16 arm losing;"
        f" the TOST is the same +/-{gate1.PPL_DELTA_BITS} bits/token one, per family -->",
    ]
    md += _md_rows(["family", "ctx", *tasks, "TOST", "sbits bf16/isvd", "verdict"], rows)
    md += ["", BF16_LEGEND, "", *[f"- {line}" for line in verdict.members]]
    return md


def gate1_table(pod_dirs: Sequence[Path], out: Path) -> None:
    """The Gate-1 table: one retrieval block per family x ctx, a perplexity block per
    family x ctx and corpus, then `gate1_verdict`'s branch and the members it was
    decided from.

    Written outside the `table*.md` glob `make tables` diffs against the paper-v1
    golden, exactly as `ppl_table` is: this is a new pod's reading, not a v1 table.
    `make gate1` is its entrypoint, and a number from it is citable only once
    `scripts/pod.py check` passes on the directories it read (prereg section 10).
    """
    data = gate1.load(list(pod_dirs))
    retr = gate1.retrieval_contrasts(data)
    ppl = gate1.ppl_contrasts(data)
    verdict = gate1.gate1_verdict(retr, ppl, data)
    # The bf16 arm rides these pods and is read by its own file (prereg/bf16_gist.md §4),
    # here rather than by hand so that a harvest cannot skip it.
    bf16 = gate1.bf16_contrasts(data)
    bf16_verdict = gate1.bf16_verdict(bf16, ppl, data)
    marks = {
        (c.family, c.ctx, c.task, c.b): ("*" if c.a_favored > c.b_favored else "‡")
        for c in retr
        if c.primary and c.p_holm is not None and c.p_holm < gate1.ALPHA
    }
    m = {  # the realised Holm family sizes section 6 asks the table to print
        "primary retrieval": sum(1 for c in retr if c.primary and c.p_holm is not None),
        "secondary retrieval": sum(1 for c in retr if not c.primary and c.p_holm is not None),
        "primary perplexity": sum(1 for c in ppl if c.p_holm is not None),
    }
    md = [
        "# Gate 1 — the tracker swap",
        "",
        f"<!-- pre-registration: {GATE1_PREREG}; the rule is its section 4 -->",
        "<!-- source: " + ", ".join(f"{f}={p}" for f, p in sorted(data.pods.items())) + " -->",
        "<!-- cell: acc [Wilson 95% lo,hi] (hits/n), marked per the legend below each block -->",
        "<!-- Holm at alpha=0.05 over each family's raw p-values, at the realised m: "
        + ", ".join(f"{k} m={v}" for k, v in m.items())
        + " -->",
        "<!-- delta: isvd MINUS the row's tracker, paired per window; delta < 0 is the"
        " r64 arm ahead -->",
        f"<!-- TOST: two one-sided t-tests at +/-{gate1.PPL_DELTA_BITS} bits/token on the"
        " per-window differences, alpha=0.05 uncorrected (intersection-union) -->",
        f"<!-- the verdict reads the ctx {gate1.VERDICT_CTX} contrasts only; any other"
        " context length is descriptive -->",
    ]
    for ctx, family in sorted({(k[1], k[0]) for k in data.hits}):
        present = [t for t in gate1.TRACKERS if (family, t) in data.arms]
        seen = {k[2] for k in data.hits if k[0] == family and k[1] == ctx}
        tasks = [t for t in gate1.TASK_ORDER if t in seen] + sorted(seen - set(gate1.TASK_ORDER))
        md += [
            "",
            "<!-- arms: " + ", ".join(f"{t}={data.arms[(family, t)]}" for t in present) + " -->",
        ]
        md += _gate1_md(
            f"## {family} — ctx {ctx}",
            ["tracker", *tasks],
            [
                [
                    tracker,
                    *[
                        _gate1_cell(
                            data,
                            (family, ctx, task, tracker),
                            marks.get((family, ctx, task, tracker), ""),
                        )
                        for task in tasks
                    ],
                ]
                for tracker in present
            ],
        )
        md += ["", GATE1_LEGEND]
        # "Every arm in the Gate-1 table carries either its cells or the exception text
        # that replaced them" (prereg section 4): the FAILED cell says how many, this says
        # what raised.
        failures = sorted(
            (k[3], k[2], errs) for k, errs in data.errors.items() if k[:2] == (family, ctx) and errs
        )
        # Section 7 (e): "A key that disagrees is dropped from every paired statistic in
        # that pod and the drop is reported with the key and the arms." One line per
        # Gate-1 or bf16 member that lost keys, with all of them -- a count alone is not
        # a report. One blank line ahead of the list, and nothing at all for a block
        # that has neither kind of line.
        bullets = [
            f"- `{data.arms[(family, tracker)]}` / {task}: {len(errs)} error records,"
            f" first `{errs[0]}`"
            for tracker, task, errs in failures
        ] + [
            f"- pairing: {len(c.dropped_keys)} key{'' if len(c.dropped_keys) == 1 else 's'}"
            f" dropped ({c.family}/{c.task}: {gate1.key_list(c.dropped_keys)})"
            f" -- {c.a} vs {c.b}, n_paired {c.n_paired}"
            for c in [*retr, *bf16]  # the bf16 members lose keys to the same invariant
            if c.dropped_keys and (c.family, c.ctx) == (family, ctx)
        ]
        md += ["", *bullets] if bullets else []
        for corpus in sorted(
            {k[2] for k in data.bits if (k[0], k[1]) == (family, ctx)}, key=lambda x: x or ""
        ):
            # Every contrast in the block is taken against the reference arm, so a
            # corpus it scored no window of has none. That is a labelled row, not a
            # dropped block: a block that silently vanishes hides a sweep that DID run
            # more thoroughly than the `--` prereg section 4 forbids.
            no_ref = f"reference arm has no windows for corpus {corpus}"
            sweeps = {c.b: c for c in ppl if (c.family, c.ctx, c.corpus) == (family, ctx, corpus)}
            rows = []
            for tracker in present:
                w = data.bits.get((family, ctx, corpus, tracker))
                if not w:
                    continue
                c = sweeps.get(tracker)
                # `c is None` is the reference arm itself, which has no contrast with
                # itself -- or, where the reference is the arm missing, every row in the
                # block. "No arm is ever printed as `--`" (prereg section 4), so the
                # cells say which it is instead of going blank.
                label = "reference" if tracker == gate1.REFERENCE else no_ref
                rows.append(
                    [tracker, f"{sum(w.values()) / len(w):.4f}"]
                    + ([label] * 5 if c is None else _gate1_ppl_cells(c))
                )
            md += _gate1_md(
                f"### {family} — perplexity, ctx {ctx}" + (f", {corpus}" if corpus else ""),
                [
                    "tracker",
                    "bits/token",
                    "delta (isvd - tracker)",
                    "95% CI",
                    f"TOST +/-{gate1.PPL_DELTA_BITS}",
                    "Holm p",
                    "paired t p",
                ],
                rows,
            )
    md += _gate1_bf16_block(data, bf16, ppl, bf16_verdict)
    md += ["", f"BF16: {bf16_verdict.overall} — {bf16_verdict.reason}"]
    # The members the branch was decided from, in full: the five-reviewer simulation
    # reads this table alone, so nothing the verdict weighed is left in the objects.
    md += ["", f"VERDICT: {verdict.branch} — {verdict.reason}", "", "members:"]
    md += [f"- {line}" for line in verdict.members]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")


# ------------------------------------------------------------- latency (kernel_smoke)

KERNEL_PREREG = "prereg/kernel_smoke.md"
W20_ARCHIVE = ARCHIVE / "w19-sysfix-llama" / "raw" / "w19-sysfix-llama-lines.txt"
N_LAYERS = {  # decoder layers -- an architecture constant, like N_FEATURES
    "unsloth/Meta-Llama-3.1-8B-Instruct": 32,
    "mistralai/Mistral-7B-Instruct-v0.3": 32,
    "Qwen/Qwen2.5-7B-Instruct": 28,
}
GIB = 1024**3  # the GiB `kvdlra.eval.latency.GB` reports kv_peak_gb in
GATE_CTX, SPEEDUP, MARGIN, SPIKES_MAX, AGREE = 32768, 3.0, 0.10, 8, 0.10  # prereg §4, §2 (a)
DIFF_MAX, MATCH_MIN = 1e-2, 14  # prereg §4: MATCH_MIN is the bar; DIFF_MAX is now the
#                                 REPORTED absolute number, no longer a bar (Amendment 2 A2.2)
REL_MAX, M_COEF = 2**-6, 2**-5  # Amendment 2: the relative per-layer bar (4 u, A2.2) and the
#                                 near-tie margin coefficient (8 u, A2.3); u = 2**-8, both
#                                 powers of two so they read as "4 / 8 bf16 ulp", not as fits
LATENCY_FIELDS = ("ms_per_token_p50", "ms_mean", "ms_max", "spikes", "resident_gb", "peak_gb",
                  "kv_peak_gb", "kv_resident_gb")  # fmt: skip


def analytic_stored_gib(cfg: ArmCfg, ctx: int, model: str) -> float | None:
    """`bug_footprint(...).stored_bits()` of a bug arm after a `ctx`-token prefill, summed
    over the model's layers, in the GiB `kv_peak_gb` uses -- the stored state ADR 0001 §3
    bills, printed BESIDE the measured peak (prereg §7). The ring is billed at its high water
    (`recent_window + absorb_block - 1`) and every non-sink, non-tier column as a coordinate
    (a `null` budget means the whole prefill). None for a non-bug arm; the quantized tier is
    not modelled (no kernel-smoke arm has one)."""
    if cfg.kind != "bug":
        return None
    kw = arm_kwargs(cfg, ctx)
    n_sink, hh = int(kw.get("n_sink", 4)), int(kw.get("hh_budget", 0))
    recent = int(kw.get("recent_window", 64)) + int(kw.get("absorb_block", 32)) - 1
    coord = min(int(kw.get("coord_budget", 1024)), ctx - n_sink - hh - recent)
    fp = bug_footprint(
        N_FEATURES[model], rank=float(kw["rank"]), coord_count=coord, recent_len=recent,
        n_sink=n_sink, retention=str(kw.get("retention", "fifo")), hh_count=hh, u_present=True,
        gist_bits=16 if kw.get("gist_dtype") == "bfloat16" else 32,
    )  # fmt: skip
    return fp.stored_bits() * N_LAYERS[model] / 8 / GIB


def _attributable(r: KernelCheckRecord) -> bool:
    """A greedy mismatch counts against the kernel only if the reconstruct path's top-2 gap
    exceeds the near-tie margin M = M_COEF * (the step's top-logit scale) (Amendment 2 A2.3).
    The per-record scale is `kernel_logit_for_ref_argmax` -- the kernel's logit at the token
    the reconstruct path chose (its top token), which equals the reconstruct top-1 logit to
    within the kernel's own rounding at exactly the near-tie steps this threshold separates;
    a large reversal (attributable) drives that logit down, only strengthening the call.
    Otherwise (small gap) the mismatch is a near-tie divergence, not counted."""
    gap, scale = r.get("gap_at_mismatch"), r.get("kernel_logit_for_ref_argmax")
    return gap is not None and scale is not None and gap > M_COEF * scale


def precondition_line(kc: Sequence[KernelCheckRecord]) -> str | None:
    """prereg §4's precondition from `kernel_check.jsonl`, as restated by Amendment 2: the
    relative per-layer bar `max_l(max|Δ_l|/max|ref_l|) <= 2^-6`, at least 14/16 prompts whose
    first mismatch is NOT attributable (near-tie divergences forgiven, A2.3), and no error
    rows. The absolute worst max|Δ| is printed beside `rel` as a number, no longer a bar
    (A2.2). None when the pod carries no check (the launch entry then names the evidence path);
    a row that predates Amendment 2 (no `rel_max_diff`) is not computable and REFUSES, never a
    silent pass."""
    if not kc:
        return None
    n = len(kc)
    errors = sum(1 for r in kc if r.get("error"))
    checks = [r for r in kc if r["max_abs_diff"] is not None]
    if not checks:
        n_match = sum(r["match"] for r in kc)
        return f"NOT met ({n_match}/{n} token-exact; no diff recorded; {errors} error rows)"
    if any(r.get("rel_max_diff") is None for r in checks):
        return (
            f"NOT met (not computable: record predates Amendment 2 -- no rel_max_diff/max|ref| "
            f"or logit fields; {n} rows, {errors} error rows)"
        )
    rel, rel_layer, _ = max(
        (r["rel_max_diff"], r["rel_worst_layer"], r["prompt"]) for r in checks
    )  # every checks row has rel_max_diff (guarded above); mypy sees `float | None`
    abs_worst, abs_layer, _ = max(
        (r["max_abs_diff"], r["worst_layer"], r["prompt"]) for r in checks
    )
    mism = [r for r in kc if r["match"] == 0 and not r.get("error")]
    n_attr = sum(1 for r in mism if _attributable(r))
    n_near = len(mism) - n_attr
    effective = n - n_attr
    assert rel is not None
    ok = rel <= REL_MAX and effective >= MATCH_MIN and errors == 0
    detail = (
        f"{effective}/{n} effective ({n_attr} attributable, {n_near} near-tie mismatches);"
        f" rel {rel:.3e} at layer {rel_layer} {'<=' if rel <= REL_MAX else '>'} 2^-6;"
        f" max|d| {abs_worst:.3e} at layer {abs_layer} (reported)"
    )
    if errors:
        detail += f"; {errors} error rows"
    return f"{'met' if ok else 'NOT met'} ({detail})"


def week3_gate(
    rows: Sequence[LatencyRecord], roles: Mapping[str, str], n_errors: int, precondition: str | None
) -> list[str]:
    """prereg §4's reading at batch 1 (cell list A), as lines: refusals, the two conditions
    with their numbers, the verdict. Other batches are reported the same way, descriptively.
    Two arms sharing a gate role make that role unusable (R-L4-25 -- `arm_of` would otherwise
    silently pick one), and every row with `spikes > SPIKES_MAX` gets its own steady-state
    reading regardless of arm/ctx/batch (R-L4-24); the GATE_CTX-scoped refusal further down
    (reconstruct/kernel at 32K only) is unchanged."""
    by = {(r["arm"], r["ctx"], r["batch"]): r for r in rows}
    gate_roles = ("full", "reconstruct", "kernel")
    dup = {role for role in gate_roles if sum(1 for r in roles.values() if r == role) > 1}
    arm_of = {role: key for key, role in roles.items() if role in gate_roles and role not in dup}
    lines: list[str] = []
    refusals: list[str] = []
    for role in sorted(dup):
        keys = [key for key, r in roles.items() if r == role]
        refusals.append(f"two arms play the {role} role: {keys}")
    if n_errors:
        refusals.append(
            f"{n_errors} error row(s) in the manifest -- any `[error]` refuses the verdict (§4)"
        )
    if precondition is None:
        lines.append(
            "PRECONDITION: not recorded on this pod -- the launch entry must name its"
            " evidence path (§4)"
        )
        if "kernel" in arm_of:
            refusals.append(
                "the correctness precondition was not recorded on this pod (no kernel_check.jsonl)"
            )
    else:
        lines.append(f"PRECONDITION: {precondition}")
        if precondition.startswith("NOT"):
            refusals.append(
                "the correctness precondition is not met: a fast wrong kernel is not a result (§4)"
            )
    for role in gate_roles:
        if role not in arm_of and role not in dup:
            refusals.append(f"no {role} arm in the pod")
    for r in sorted(rows, key=lambda r: (r["arm"], r["ctx"], r["batch"])):
        if r["spikes"] > SPIKES_MAX:
            lines.append(
                f"NOT STEADY STATE: {r['arm']} ctx={r['ctx']} batch={r['batch']} spikes="
                f"{r['spikes']}/56 (ms_mean={r['ms_mean']:.2f}, ms_max={r['ms_max']:.2f})"
                " — p50 not read as a steady-state number"
            )
    batches = sorted({r["batch"] for r in rows}) or [1]
    for batch in batches:
        if any(role not in arm_of for role in gate_roles):
            continue  # the refusal above already names the missing/ambiguous role
        tag = f"@32K b{batch}"
        cells = {role: by.get((arm_of[role], GATE_CTX, batch)) for role in arm_of}
        missing = [role for role in gate_roles if role in arm_of and cells.get(role) is None]
        if missing:
            (refusals if batch == 1 else lines).append(
                f"batch {batch}: no 32K record for {missing}"
            )
            continue
        full, rec, ker = cells["full"], cells["reconstruct"], cells["kernel"]
        assert full is not None and rec is not None and ker is not None
        spiky = []
        for role, r in (("reconstruct", rec), ("kernel", ker)):
            if r["spikes"] > SPIKES_MAX:
                spiky.append(role)
                lines.append(
                    f"{tag}: {role} spikes={r['spikes']} > {SPIKES_MAX} of 56 -- p50 refused"
                    f" as a steady state (ms_mean {r['ms_mean']:.2f}, ms_max {r['ms_max']:.2f})"
                )
        margin = (
            (full["kv_peak_gb"] - ker["kv_peak_gb"]) / full["kv_peak_gb"]
            if full["kv_peak_gb"]
            else float("nan")
        )
        mem = ker["kv_peak_gb"] < full["kv_peak_gb"]
        note = (
            f" (MARGINAL: margin {margin:.1%} < 10%)"
            if mem and margin < MARGIN
            else f" (margin {margin:.1%})"
        )
        lines.append(
            f"memory {tag}: kernel kv_peak_gb {ker['kv_peak_gb']:.2f} vs full"
            f" {full['kv_peak_gb']:.2f} -> {'PASS' if mem else 'FAIL'}{note}"
        )
        ratio = rec["ms_per_token_p50"] / ker["ms_per_token_p50"]
        speed = ratio >= SPEEDUP
        lines.append(
            f"speed {tag}: reconstruct p50 {rec['ms_per_token_p50']:.2f} ms / kernel p50"
            f" {ker['ms_per_token_p50']:.2f} ms = {ratio:.2f}x vs {SPEEDUP:.1f}x ->"
            f" {'PASS' if speed else 'FAIL'}"
            + (" (REFUSED: a refused p50 enters the ratio)" if spiky else "")
        )
        if batch != 1:
            lines.append(f"batch {batch}: descriptive -- the Week-3 verdict is batch 1 (§4 (3))")
            continue
        if spiky:
            refusals.append(f"a refused p50 ({', '.join(spiky)}) enters the 32K speed ratio")
        if refusals:
            verdict = "REFUSED -- " + "; ".join(refusals)
        else:
            verdict = (
                "PASS"
                if mem and speed
                else "FAIL -- "
                + "; ".join(c for c, ok in (("memory", mem), ("speed", speed)) if not ok)
            )
        lines.append(f"WEEK-3 GATE (batch 1): {verdict}")
    if not any(x.startswith("WEEK-3 GATE") for x in lines):
        lines.append(
            "WEEK-3 GATE (batch 1): REFUSED -- " + "; ".join(refusals or ["no batch-1 32K rows"])
        )
    return lines


def latency_table(pod: PodCfg, results: Path, out: Path) -> None:
    """The kernel-smoke table (prereg §7) and the Week-3 reading (§4) for one pod directory,
    written outside the `table*.md` set `build` pins. A number from it is citable only once
    `scripts/pod.py check` passes on that directory (§10)."""
    archive = W20_ARCHIVE
    lat = results / "latency.jsonl"
    rows = [cast(LatencyRecord, r) for r in read_jsonl(lat)] if lat.is_file() else []
    kcp = results / "kernel_check.jsonl"
    kc = [cast(KernelCheckRecord, r) for r in read_jsonl(kcp)] if kcp.is_file() else []
    mpath = results / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.is_file() else {}
    n_errors = int(manifest.get("errors") or 0)
    errors = {
        (e["arm"], e["ctx"], e["batch"]): str(e["error"])
        for log in sorted(results.glob("*.log"))
        for e in parse_error_lines(log.read_text(), log.name)
        if e["axis"] == "latency"
    }
    # Keyed like the records: one load per arm, not one per grid row -- and the roles come
    # off the same loaded configs rather than a second sweep of the arm files.
    cfgs = {(c.legacy_name or c.name): c for c in (load_arm(s) for s in pod.arms)}
    roles = {key: role_of(c) for key, c in cfgs.items()}
    w20 = (
        {
            (r["arm"], r["ctx"]): r
            for r in parse_latency_lines(archive.read_text(), pod.model, archive.name)
        }
        if archive.is_file()
        else {}
    )
    tasks = [t for t in (load_task(x) for x in pod.tasks) if t.generator == "latency"]
    grid = [
        (a, c, b) for t in tasks for c in (t.ctxs or [t.ctx]) for b in t.batch_sizes for a in roles
    ]
    by = {(r["arm"], r["ctx"], r["batch"]): r for r in rows}
    body = []
    for a, c, b in grid:
        r = by.get((a, c, b))
        if r is None:
            cells = [f"not run ({errors.get((a, c, b), 'no record')})"] + ["not run"] * (
                len(LATENCY_FIELDS) - 1
            )
        else:
            fields: Mapping[str, object] = r
            cells = []
            for f in LATENCY_FIELDS:
                v = fields.get(f)
                cells.append(
                    "not recorded"
                    if v is None
                    else (str(v) if f == "spikes" else f"{float(cast(float, v)):.2f}")
                )
        stored = analytic_stored_gib(cfgs[a], c, pod.model)
        # Which backend attended (`kvdlra.kernel.select_backend`, recorded per cache):
        # `n/a` wherever none was recorded -- a full/quant/reconstruct arm never runs one,
        # and the kernel rows of a pod logged before the field existed carry no value.
        backend = (r.get("backend") if r is not None else None) or "n/a"
        w = w20.get((a, c))
        if w is None:
            archived = ["none archived", "none archived", "n/a"]
        else:
            agree = (
                "not measured"
                if r is None
                else (
                    "agree (within 10%)"
                    if abs(r["ms_per_token_p50"] - w["ms_per_token_p50"])
                    <= AGREE * w["ms_per_token_p50"]
                    else f"differs by {(r['ms_per_token_p50'] / w['ms_per_token_p50'] - 1):+.0%}"
                )
            )
            archived = [
                f"{w['ms_per_token_p50']:.2f}",
                f"{w['kv_peak_gb']:.2f}",
                f"archived {w['ms_per_token_p50']:.2f} ms: {agree}",
            ]
        body.append(
            [
                a,
                roles[a],
                str(c),
                str(b),
                *cells,
                "n/a" if stored is None else f"{stored:.3f}",
                *archived,
                backend,
            ]
        )
    header = ["arm", "role", "ctx", "batch", "ms/tok p50", "ms mean", "ms max", "spikes",
              "resident_gb", "peak_gb", "kv_peak_gb", "kv_resident_gb", "stored GiB (analytic)",
              "W20 p50", "W20 kv_peak", "W20 agreement", "backend"]  # fmt: skip
    md = [
        f"# Kernel smoke — {results.name}",
        "",
        f"<!-- pre-registration: {KERNEL_PREREG}; the reading is its §4, the table its §7 -->",
        f"<!-- source: {results.name}/latency.jsonl ({len(rows)} rows), kernel_check.jsonl"
        f" ({len(kc)} rows), manifest errors {n_errors}; archived rows: {archive.name} -->",
        "<!-- cell: one measurement per (arm, ctx, batch): p50 of 56 timed decode steps, spikes"
        " = steps > 2x p50, kv_* = VRAM minus the weights; stored GiB ="
        " bug_footprint(...).stored_bits() summed over the layers -- beside the measured peak,"
        " never instead of it (§7) -->",
        "<!-- W20 = the archived Week-20 rows of §2 (a), re-measured by this pod's own full and"
        " reconstruct arms; a p50 within 10% of its archived value agrees, else the gap is"
        " reported -->",
        *_md_rows(header, body),
        "",
        *week3_gate(rows, roles, n_errors, precondition_line(kc)),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser(
        "convert-v1",
        help="archive the paper-v1 line files as JSONL; write-once -- re-running it over an"
        " existing archive raises rather than clobber rows, so a re-run means"
        " `rm -rf results/paper-v1` first",
    )
    c.add_argument("--out", default=None, help="default: <repo_root>/results/paper-v1")
    b = sub.add_parser("build", help="regenerate the paper-v1 tables from results/paper-v1")
    b.add_argument("--out", default="docs/paper/tables")
    p = sub.add_parser(
        "ppl",
        help="bits/token per arm with a paired 95% CI and TOST vs `full`, from one pod's"
        " pplw.jsonl; written outside the `table*.md` set `build` pins",
    )
    p.add_argument("--pod", required=True, help="a directory name under results/")
    p.add_argument("--out", default=None, help="default: docs/paper/tables/ppl_<pod>.md")
    p.add_argument("--baseline", default="full", help="the arm every other is paired against")
    p.add_argument("--delta", type=float, default=0.05, help="TOST margin, bits/token")
    g = sub.add_parser(
        "gate1",
        help=f"the Gate-1 table ({GATE1_PREREG}): Holm-corrected retrieval contrasts, the"
        " perplexity TOSTs, and the branch the rule selects; written outside the"
        " `table*.md` set `build` pins",
    )
    g.add_argument(
        "--pods", nargs="+", required=True, help="results/<pod> directories, one per model family"
    )
    g.add_argument("--out", default="docs/paper/tables/gate1.md")
    la = sub.add_parser(
        "latency",
        help=f"the kernel-smoke table and the Week-3 reading ({KERNEL_PREREG} §7, §4) from one"
        " pod's latency.jsonl + kernel_check.jsonl; written outside the `table*.md` set"
        " `build` pins",
    )
    la.add_argument("--pods", nargs=1, required=True, help="one results/<pod> directory")
    la.add_argument("--out", default=None, help="default: docs/paper/tables/<pod>.md")
    a = ap.parse_args()
    if a.cmd == "convert-v1":
        out = Path(a.out) if a.out else REPO_ROOT / "results" / "paper-v1"
        convert_v1(out)
    elif a.cmd == "ppl":
        out = Path(a.out) if a.out else REPO_ROOT / "docs/paper/tables" / f"ppl_{a.pod}.md"
        ppl_table(REPO_ROOT / "results" / a.pod, out, a.baseline, a.delta)
    elif a.cmd == "gate1":
        gate1_table([Path(p) for p in a.pods], Path(a.out))
    elif a.cmd == "latency":
        (pod_dir,) = (Path(p) for p in a.pods)
        name = str(json.loads((pod_dir / "manifest.json").read_text())["pod"])
        out = Path(a.out) if a.out else REPO_ROOT / "docs/paper/tables" / f"{name}.md"
        latency_table(load_pod(name), pod_dir, out)
    else:
        build(Path(a.out))


if __name__ == "__main__":
    main()
