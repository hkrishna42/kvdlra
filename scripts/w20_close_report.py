"""Week-20 gate-closing report: the tracker-swap ablation, the measured decode
latency/peak, and the single-shot 2-bit prefill control -- each paired against its
committed reference. Writes results/w20-close-report.md.

* swap (prior-work 6->7): bugSseed-r64-h256 with the gist tracker swapped to Oja's rule
  / Frequent Directions, Llama 16K T4 n=12 + ppl, vs the flagship as-is -- which IS the
  fixed-rank incremental-SVD arm (theta=None, min_sv_frac=0), already measured (W18 g1).
* latency (systems 6->7): decode ms/token + resident/peak VRAM, full vs flagship vs
  KIVI-2bit at 16K/32K/64K, batch 1 (w20_latency.py rows).
* ss2 (rigor firming): quant-2bit-kivi with single-shot prefill (--chunk 0) at 16K vs the
  chunked-prefill rows of a1 (same needles) -- is the in-repo mv edge protocol-bound?

    uv run python scripts/w20_close_report.py
"""

# ruff: noqa: E501  (Markdown data-table rows are long by construction)
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import _paths  # noqa: F401
from w18_intervals import ROW

RES = Path("results")
T4 = ["niah_single", "niah_multikey", "niah_multivalue", "vt"]
PPL_RE = re.compile(r"^\s+(\S+)\s+\[T=(\d+)\] ppl=([0-9.]+)", re.M)
LAT_RE = re.compile(
    r"^\[latency ctx(\d+)\] (\S+)\s+ms/tok=([0-9.]+) mean=([0-9.]+) max=([0-9.]+) spikes=(\d+) "
    r"resident_gb=([0-9.]+) peak_gb=([0-9.]+) weights_gb=([0-9.]+) kv_peak_gb=([0-9.]+)",
    re.M,
)


def acc_rows(path: Path, ctx: int) -> dict[tuple[str, str], float]:
    """(arm, task) -> best acc at ``ctx`` (a re-emitted cell keeps its filled row)."""
    out: dict[tuple[str, str], float] = {}
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        m = ROW.match(ln)
        if m and int(m.group(2)) == ctx:
            k = (m.group(3), m.group(1))
            out[k] = max(out.get(k, 0.0), float(m.group(4)))
    return out


def ppl_rows(path: Path, ctx: int) -> dict[str, float]:
    if not path.exists():
        return {}
    return {
        m.group(1): float(m.group(3))
        for m in PPL_RE.finditer(path.read_text())
        if int(m.group(2)) == ctx
    }


def fmt4(d: dict[tuple[str, str], float], arm: str) -> str:
    return " | ".join(f"{d[(arm, t)]:.2f}" if (arm, t) in d else "--" for t in T4)


def main() -> None:
    md = ["# Week-20 gate-closing experiments\n"]
    # ---- 1. tracker swap
    swap = RES / "w19-swap-llama-lines.txt"
    ref = acc_rows(RES / "w18-llama-lines.txt", 16384)
    refp = ppl_rows(RES / "w18-llama-ppl-lines.txt", 16384)
    sw = acc_rows(swap, 16384)
    swp = ppl_rows(swap, 16384)
    md.append("## 1. Tracker-swap ablation (Llama-3.1-8B, 16K, n=12, same cache, same needles)\n")
    md.append(
        "Is the DLRA integrator load-bearing end-to-end? The flagship's defaults (theta=None, "
        "min_sv_frac=0) make its step **fixed-rank incremental SVD** (Brand 2006), so that arm is the "
        "flagship as already measured (W18 g1). Oja's rule = the OjaKV baseline (validated Week-2 "
        "schedule); Frequent Directions = shrinkage instead of truncation.\n"
    )
    md.append("| gist tracker | arm | single | multi-key | multi-value | var-track | ppl 16K |")
    md.append("|---|---|---|---|---|---|---|")
    f = "bugSseed-r64-h256"
    md.append(
        f"| **incremental SVD (BUG step, flagship)** | `{f}` | {fmt4(ref, f)} | {refp.get(f, float('nan')):.2f} |"
    )
    for trk in ("oja", "fd"):
        a = f"{f}-{trk}"
        name = "Oja's rule" if trk == "oja" else "Frequent Directions"
        p = swp.get(a)
        md.append(
            f"| {name} | `{a}` | {fmt4(sw, a)} | {p:.2f} |"
            if p is not None
            else f"| {name} | `{a}` | {fmt4(sw, a)} | -- |"
        )
    # ---- 2. latency
    lat = RES / "w19-sysfix-llama-lines.txt"
    md.append("\n## 2. Measured decode latency and VRAM (Llama-3.1-8B, A100-40GB, batch 1)\n")
    md.append(
        "One token per forward at true positions, CUDA-synced; p50 = steady state, max/spikes "
        "surface BUG's absorb-event middle rebuild. VRAM with model weights subtracted.\n"
    )
    md.append(
        "| ctx | arm | ms/tok p50 | mean | max | spikes | resident GB | peak GB | KV peak GB |"
    )
    md.append("|---|---|---|---|---|---|---|---|---|")
    if lat.exists():
        seen: dict[tuple[int, str], tuple[str, ...]] = {}
        for m in LAT_RE.finditer(lat.read_text()):
            seen[(int(m.group(1)), m.group(2))] = m.groups()
        for (ctx, arm), g in sorted(seen.items()):
            md.append(
                f"| {ctx // 1024}K | `{arm}` | {float(g[2]):.1f} | {float(g[3]):.1f} | {float(g[4]):.1f} | {g[5]} | {float(g[6]):.2f} | {float(g[7]):.2f} | {float(g[9]):.2f} |"
            )
    else:
        md.append("| -- | (pending) | | | | | | | |")
    # ---- 3. single-shot control
    ss = acc_rows(lat, 16384)
    ssp = ppl_rows(lat, 16384)
    a1 = acc_rows(RES / "w19-a1-llama-lines.txt", 16384)
    a1p = ppl_rows(RES / "w19-a1-llama-lines.txt", 16384)
    md.append("\n## 3. Single-shot 2-bit prefill control (Llama-3.1-8B, 16K, n=12, same needles)\n")
    md.append(
        "The a1 KIVI arm prefilled in 4096-token chunks (later chunks attend to 2-bit-dequantized "
        "history). Does the flagship's in-repo multi-value edge survive KIVI's single-shot "
        "(full-precision-prefill) operating point?\n"
    )
    md.append("| protocol | arm | single | multi-key | multi-value | var-track | ppl 16K |")
    md.append("|---|---|---|---|---|---|---|")
    q = "quant-2bit-kivi"
    md.append(f"| chunked (a1, 4096) | `{q}` | {fmt4(a1, q)} | {a1p.get(q, float('nan')):.2f} |")
    ssq = f"{ssp[q]:.2f}" if q in ssp else "--"
    md.append(f"| **single-shot (ss2, --chunk 0)** | `{q}` | {fmt4(ss, q)} | {ssq} |")
    md.append(
        f"| flagship (reference) | `{f}` | {fmt4(ref, f)} | {refp.get(f, float('nan')):.2f} |"
    )
    # ---- 4. the sub-cliff cell ON the official anchor (q4off) vs everything already there
    from w19_fork_report import COMPO, OFF9, SB
    from w19_fork_report import parse as fork_parse

    q4o = acc_rows(RES / "w19-q4off-llama-lines.txt", 16384)
    a2 = acc_rows(RES / "w19-a2-llama-lines.txt", 16384)
    offc = fork_parse("llama")  # section-aware: (sect, task, arm) -> (acc, recall)
    md.append("\n## 4. The sub-cliff cell on the official RULER anchor (Llama 16K, 9 tasks x 12)\n")
    md.append(
        "The fork's pre-registered rule compares the composite to the q4 cell *on the anchor*; "
        "the fork ran only the composites there, this runs the cell itself (same records, "
        "seed 42). Rule: q4 holds single/mk/mv where the composites collapsed -> the band is "
        "anchored and exclusive (significance 7); q4 also collapses -> the band is a "
        "our-generator property and the paper says so (stays 6).\n"
    )
    md.append(
        "| arm | stored | "
        + " | ".join(t.replace("niah_", "").replace("_", "") for t in OFF9)
        + " | mean |"
    )
    md.append("|---|---|" + "---|" * (len(OFF9) + 1))

    def offrow(label: str, stored: str, get: Callable[[str], float | None]) -> None:
        vals = [get(t) for t in OFF9]
        have = [v for v in vals if v is not None]
        cells = " | ".join(f"{v:.2f}" if v is not None else "--" for v in vals)
        if not have:
            mean = "--"
        elif len(have) == len(OFF9):
            mean = f"**{sum(have) / len(have):.2f}**"
        else:
            mean = f"({sum(have) / len(have):.2f}, {len(have)}/9)"
        md.append(f"| {label} | {stored} | {cells} | {mean} |")

    def _off(arm: str) -> Callable[[str], float | None]:
        return lambda t: (lambda v: v[0] if v else None)(offc.get(("off", t, arm)))

    q4 = "bugSseed-r64-h256-q4"
    offrow(f"**q4 cell** `{q4}`", "0.048x", lambda t: q4o.get((q4, t)))
    offrow(f"flagship `{f}` (a2)", "0.151x", lambda t: a2.get((f, t)))
    for arm in COMPO:
        offrow(f"`{arm}` (fork)", f"{SB[arm]}x", _off(arm))
    offrow("`ea-k0.1` plain (a2)", "0.100x", lambda t: a2.get(("ea-k0.1", t)))
    (RES / "w20-close-report.md").write_text("\n".join(md) + "\n")
    print("[wrote results/w20-close-report.md]")
    print("\n".join(md))


if __name__ == "__main__":
    main()
