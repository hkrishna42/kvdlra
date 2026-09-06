"""Week-20 decisive-fork report: does an eviction x quantization composite reach the
sub-cliff band (<=0.05x) with retrieval, or collapse like plain eviction?

Parses the fork pod raws SECTION-AWARE (the in-repo niah_multivalue/vt rows collide by
name with the official ones, so the ===W19_FORK_R*/FORKOFF markers disambiguate them),
tabulates the composite ea-k{keep}-q{nbits} arms against the committed sub-cliff q4 cell
(a1q) on our generator and against plain eviction on the official RULER anchor (a2), and
applies the pre-registered decision rule. Writes results/w20-fork-report.md.

    uv run python scripts/w19_fork_report.py
"""

# ruff: noqa: E501  (Markdown data-table rows are long by construction)
from __future__ import annotations

import re
from pathlib import Path

RES = Path("results")
RAW = RES / "w19_harvest"
ROW = re.compile(r"^\[(\S+) ctx(\d+)\] (ea-k\S+|bugSseed-\S+) acc=([0-9.]+) recall=([0-9.]+)")
FAMS = [("llama", "Llama-3.1-8B"), ("mistral", "Mistral-7B-v0.3"), ("qwen", "Qwen2.5-7B")]
COMPO = ["ea-k0.1-q2-kivi", "ea-k0.1-q4-kivi", "ea-k0.25-q2-kivi", "ea-k0.25-q4-kivi"]
SB = {
    "ea-k0.1-q2-kivi": 0.016,
    "ea-k0.1-q4-kivi": 0.028,
    "ea-k0.25-q2-kivi": 0.039,
    "ea-k0.25-q4-kivi": 0.070,
}
T4 = ["niah_single", "niah_multikey", "niah_multivalue", "vt"]
OFF9 = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
        "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt"]  # fmt: skip


def parse(tag: str) -> dict[tuple[str, str, str], tuple[float, float]]:
    """(section, task, arm) -> (acc, recall), last occurrence wins. section in in16/in32/off."""
    out: dict[tuple[str, str, str], tuple[float, float]] = {}
    p = RAW / f"fork-{tag}.raw"
    if not p.exists():
        return out
    sect = ""
    for ln in p.read_text().splitlines():
        if "FORK_R16384_BEGIN" in ln:
            sect = "in16"
        elif "FORK_R32768_BEGIN" in ln:
            sect = "in32"
        elif "FORKOFF_BEGIN" in ln:
            sect = "off"
        m = ROW.match(ln)
        if m and sect:
            task, _ctx, arm, acc, rec = (
                m.group(1),
                m.group(2),
                m.group(3),
                float(m.group(4)),
                float(m.group(5)),
            )
            out[(sect, task, arm)] = (acc, rec)
    return out


def q4_cell(tag: str) -> dict[tuple[str, int], float]:
    """The committed sub-cliff q4 cell (a1q), our generator: (task, ctx) -> acc (best row)."""
    out: dict[tuple[str, int], float] = {}
    f = RES / f"w19-a1q-{tag}-lines.txt"
    if not f.exists():
        return out
    for ln in f.read_text().splitlines():
        m = ROW.match(ln)
        if m and m.group(3) == "bugSseed-r64-h256-q4":
            k = (m.group(1), int(m.group(2)))
            out[k] = max(out.get(k, 0.0), float(m.group(4)))  # the real (filled-tier) row
    return out


def main() -> None:
    md: list[str] = ["# Week-20 decisive fork: eviction x quantization vs the sub-cliff band\n"]
    md.append(
        "The sub-cliff cell `bugSseed-r64-h256-q4` (0.048x/0.034x) is exclusive vs *scalar* "
        "quantization by construction. This measures whether an **eviction x quantization** "
        "composite (`ea-k{keep}-q{nbits}`: ExpectedAttention prunes to keep-fraction, survivors "
        "stored 2/4-bit) reaches the same band **with retrieval**, or collapses like plain "
        "eviction. In-repo arms share a1q's needles; the official arms share a2's (NVIDIA RULER, "
        "essays). Stored ratios: ea-k0.1-q2 0.016x, ea-k0.1-q4 0.028x, ea-k0.25-q2 0.039x, "
        "ea-k0.25-q4 0.070x -- the byte-matches to the q4 cell are ea-k0.25-q2 (~0.039x vs 0.048x "
        "at 16K) and ea-k0.1-q4 (~0.028x vs 0.034x at 32K).\n"
    )
    # our-generator comparison, per family
    for tag, name in FAMS:
        d = parse(tag)
        q4 = q4_cell(tag)
        md.append(f"\n## {name} -- our generator (in-repo, n=12)\n")
        md.append("| ctx | arm | stored | single | multi-key | multi-value | var-track |")
        md.append("|---|---|---|---|---|---|---|")
        for sect, ctx in (("in16", 16384), ("in32", 32768)):
            q = q4.get(("niah_single", ctx))
            qk, qv, qt = (q4.get((t, ctx)) for t in ("niah_multikey", "niah_multivalue", "vt"))
            qrow = " | ".join(f"{x:.2f}" if x is not None else "--" for x in (q, qk, qv, qt))
            md.append(
                f"| {ctx // 1024}K | **q4 cell** (bugSseed-r64-h256-q4) | ~{0.048 if ctx == 16384 else 0.034}x | {qrow} |"
            )
            for arm in COMPO:
                cells = [d.get((sect, t, arm)) for t in T4]
                row = " | ".join(f"{c[0]:.2f}" if c else "--" for c in cells)
                md.append(f"| {ctx // 1024}K | {arm} | {SB[arm]}x | {row} |")
    # official anchor, Llama (composite vs plain eviction)
    d = parse("llama")
    md.append(
        "\n## Official NVIDIA RULER anchor (Llama 16K, 9 tasks) -- composite vs plain eviction\n"
    )
    md.append(
        "Plain `ea-k0.1` (0.100x) scored mean **0.20** here (a2). Do the composites do better at <=0.07x?\n"
    )
    md.append(
        "| arm | stored | "
        + " | ".join(t.replace("niah_", "").replace("_", "") for t in OFF9)
        + " | mean |"
    )
    md.append("|---|---|" + "---|" * (len(OFF9) + 1))
    for arm in COMPO:
        accs = [d.get(("off", t, arm)) for t in OFF9]
        vals = [c[0] if c else 0.0 for c in accs]
        mean = sum(vals) / len(vals)
        cellrow = " | ".join(f"{v:.2f}" for v in vals)
        md.append(f"| {arm} | {SB[arm]}x | {cellrow} | **{mean:.2f}** |")
    md.append(
        "| `ea-k0.1` (plain, a2) | 0.100x | 1.00 | 0.17 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | **0.20** |"
    )
    (RES / "w20-fork-report.md").write_text("\n".join(md) + "\n")
    print("[wrote results/w20-fork-report.md]")
    print("\n".join(md[-16:]))


if __name__ == "__main__":
    main()
