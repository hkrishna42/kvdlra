"""Week-19 A1 report assembler: the fair-quant crux, at matched stored bytes.

Reads the per-family Wilson/McNemar builds (``w18_intervals.py`` over the W19 a1 quant
rows + the W18 flagship rows, paired on the shared ``seed*131+trial`` needles), the W19
perplexity rows, the sub-cliff compose rows (a1q) and the persistence rows (a3), and
writes ``results/w19-a1-report.md``. Numbers come from the files, never retyped.

    uv run python scripts/w19_a1_report.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
from w18_intervals import ROW

FAMILIES = [("llama", "Llama-3.1-8B"), ("mistral", "Mistral-7B-v0.3"), ("qwen", "Qwen2.5-7B")]
TASKS = ["niah_single", "niah_multikey", "niah_multivalue", "vt"]
SHORT = {"niah_single": "single", "niah_multikey": "mk", "niah_multivalue": "mv", "vt": "vt"}
FLAG = "bugSseed-r64-h256"
COMPOSE = "bugSseed-r64-h256-q4"
ARMS = [FLAG, "quant-2bit-kivi", "quant-4bit-kivi", "quant-8bit-kivi-hqq"]
CTXS = (16384, 32768)
PPL_RE = re.compile(r"^\s+(\S+)\s+\[T=(\d+)\] ppl=([0-9.]+) ", re.M)
PERSIST_RE = re.compile(
    r"^\[persist ctx(\d+)\] (\S+)\s+bytes=(\d+) ratio=([0-9.]+) save=([0-9.]+)s load=([0-9.]+)s "
    r"h2d=([0-9.]+)s ready=([0-9.]+)s cold=([0-9.]+)s",
    re.M,
)
INTRO = (
    "KIVI-faithful quantized KV (per-channel keys, per-token values; `--quant-scheme kivi`, "
    "quanto backend, g=64, residual 128; 8-bit control on the hqq backend) vs the flagship "
    "`bugSseed-r64-h256`, same harness, same needles (paired), n=12 per cell (8-bit control "
    "n=4). Stored ratios are the honest stored-bits billing (BUG's fp32-at-rest state "
    "included; quant aux at its measured bf16 dtype). Flagship rows: Week-18 g1 pods (same "
    "generator, seeds, trials); quant rows: Week-19 a1 pods."
)


def _cell(c: dict[str, Any] | None) -> str:
    if not c:
        return "—"
    return f"{c['acc']:.2f} [{c['lo']:.2f},{c['hi']:.2f}] n={c['n']}"


def _ppl_rows(path: Path) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    if path.exists():
        for m in PPL_RE.finditer(path.read_text()):
            out[(m.group(1), int(m.group(2)))] = float(m.group(3))
    return out


def _flag_ppl() -> dict[str, dict[int, float]]:
    """Flagship PPL4 from the committed Week-18 primaries (never retyped)."""
    res = Path("results")
    out: dict[str, dict[int, float]] = {}
    for tag, _ in FAMILIES:
        rows = _ppl_rows(res / f"w18-{tag}-ppl-lines.txt")
        out[tag] = {ctx: rows[(FLAG, ctx)] for ctx in CTXS if (FLAG, ctx) in rows}
    return out


FLAG_PPL = _flag_ppl()


def _family(res: Path, tag: str, name: str) -> list[str]:
    blob = json.loads((res / "w19_intervals" / f"a1-{tag}-ruler-intervals.json").read_text())
    cells = blob["cells"]
    out = [f"\n## {name}\n"]
    for ctx in CTXS:
        out.append(f"\n### {ctx // 1024}K retrieval (Wilson 95%)\n")
        out.append("| arm | stored | " + " | ".join(SHORT[t] for t in TASKS) + " |")
        out.append("|---|---|" + "---|" * len(TASKS))
        for arm in ARMS:
            a = cells.get(str(ctx), {}).get(arm)
            if not a:
                continue
            sb = next((v["sbits"] for v in a.values() if v), 0.0)
            out.append(
                f"| `{arm}` | {sb:.3f}x | " + " | ".join(_cell(a.get(t)) for t in TASKS) + " |"
            )
    out.append("\n### Paired McNemar (flagship vs quant; A>B = flagship hit where quant missed)\n")
    out.append("| vs | ctx | task | A>B | B>A | p | sig |")
    out.append("|---|---|---|---|---|---|---|")
    for c in blob["contrasts"]:
        if c["arm_a"] != FLAG or (c["a_favored"] == 0 and c["b_favored"] == 0):
            continue
        sig = "**YES**" if c["significant"] else "no"
        out.append(
            f"| `{c['arm_b']}` | {int(c['ctx']) // 1024}K | {SHORT[c['task']]} | "
            f"{c['a_favored']} | {c['b_favored']} | {c['p_value']:.4f} | {sig} |"
        )
    ppl = _ppl_rows(res / f"w19-a1-{tag}-lines.txt")
    out.append("\n### Perplexity (PPL4, WikiText-103, window 512)\n")
    out.append("| ctx | flagship (W18) | quant-2bit-kivi | quant-4bit-kivi |")
    out.append("|---|---|---|---|")
    for ctx in CTXS:
        q2, q4 = ppl.get(("quant-2bit-kivi", ctx)), ppl.get(("quant-4bit-kivi", ctx))
        cols = [f"{q:.2f}" if q else "—" for q in (q2, q4)]
        out.append(f"| {ctx // 1024}K | {FLAG_PPL[tag][ctx]:.2f} | {cols[0]} | {cols[1]} |")
    qf = res / f"w19-a1q-{tag}-lines.txt"
    if qf.exists():
        text = qf.read_text()
        rows = {
            (m.group(1), int(m.group(2))): (float(m.group(4)), int(m.group(8)), m.group(7))
            for m in ROW.finditer(text)
            if m.group(3) == COMPOSE
        }
        qppl = _ppl_rows(qf)
        if rows:
            out.append(
                f"\n### Sub-cliff compose `{COMPOSE}` "
                "(512 fp32 coords kept, the rest 4-bit PolarQuant, never dropped)\n"
            )
            out.append("| ctx | stored | single | mk | mv | vt | ppl |")
            out.append("|---|---|---|---|---|---|---|")
            for ctx in CTXS:
                got = [rows.get((t, ctx)) for t in TASKS]
                if not any(got):
                    continue
                sb = next((g[2] for g in got if g), "?")
                accs = " | ".join(f"{g[0]:.2f} (n={g[1]})" if g else "—" for g in got)
                p = qppl.get((COMPOSE, ctx))
                out.append(f"| {ctx // 1024}K | {sb}x | {accs} | {f'{p:.2f}' if p else '—'} |")
    return out


def main() -> None:
    res = Path("results")
    lines: list[str] = ["# Week-19 A1 — the fair 2-bit baseline, at matched stored bytes\n", INTRO]
    for tag, name in FAMILIES:
        lines.extend(_family(res, tag, name))
    pf = res / "w19-a3-llama2-lines.txt"
    if pf.exists():
        lines.append(
            "\n## Persistence cold start (Llama-3.1-8B, A100-40GB; a3-llama2, medians of 5)\n"
        )
        lines.append("| ctx | arm | bytes | ratio | load | h2d | attend-ready | cold total |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for m in PERSIST_RE.finditer(pf.read_text()):
            ctx, arm, b, ratio, _save, load, h2d, ready, cold = m.groups()
            lines.append(
                f"| {int(ctx) // 1024}K | `{arm}` | {int(b) / 1e6:.0f} MB | {float(ratio):.3f}x "
                f"| {load}s | {h2d}s | {ready}s | **{cold}s** |"
            )
    out = res / "w19-a1-report.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[wrote {out}] ({len(lines)} lines)")


if __name__ == "__main__":
    main()
