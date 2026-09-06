"""Week-19 paper figures from the committed line-files (never retyped numbers).

* ``fairquant`` -- retrieval vs stored bytes, small multiples (rows = family, cols = task):
  flagship, the sub-cliff compose cell, KIVI 2-bit and 4-bit; 16K filled, 32K hollow,
  the two joined by a thin segment (the flagship's drifts left = its 1/T term).
* ``one_over_t`` -- stored ratio vs context (Llama): the flagship amortizes toward
  its 2r/n=0.125x asymptote, the 2-bit arm toward 0.156x.
  Picks up the 64K point automatically when ``results/w19-a4-llama-lines.txt`` exists.
* ``coldstart`` -- persisted-cache cold start (seconds) at 16K/32K: full vs flagship vs 2-bit.

Palette = the dataviz reference instance, first three categorical slots (validated
all-pairs); colour follows the entity across every figure (flagship blue, 2-bit orange,
4-bit aqua; the compose cell is the flagship's hue with a hollow diamond; full KV is ink).

    uv run python scripts/w19_figures.py   # -> figures/week19/*.{pdf,png}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import _paths  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from w18_intervals import ROW

RES = Path("results")
OUT = Path("figures/week19")
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
FAMILIES = [("llama", "Llama-3.1-8B"), ("mistral", "Mistral-7B-v0.3"), ("qwen", "Qwen2.5-7B")]
TASKS = [
    ("niah_single", "single"),
    ("niah_multikey", "multi-key"),
    ("niah_multivalue", "multi-value"),
    ("vt", "var-track"),
]
ARMS = {  # arm -> (label, colour, marker)
    "bugSseed-r64-h256": ("BUG flagship (r64, fp32 at rest)", BLUE, "o"),
    "bugSseed-r64-h256-q4": ("BUG + 4-bit coordinates", BLUE, "D"),
    "quant-2bit-kivi": ("KIVI 2-bit", ORANGE, "s"),
    "quant-4bit-kivi": ("KIVI 4-bit", AQUA, "^"),
}
PERSIST_RE = re.compile(
    r"^\[persist ctx(\d+)\] (\S+)\s+bytes=(\d+) ratio=([0-9.]+) save=[0-9.]+s load=([0-9.]+)s "
    r"h2d=([0-9.]+)s ready=([0-9.]+)s cold=([0-9.]+)s",
    re.M,
)
PPL_RE = re.compile(r"^\s+(\S+)\s+\[T=(\d+)\] ppl=[0-9.]+ .*?sbits=([0-9.]+)", re.M)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def _cells(tag: str) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """ctx -> arm -> task -> cell, merging the a1 intervals with the a1q compose rows."""
    blob = json.loads((RES / "w19_intervals" / f"a1-{tag}-ruler-intervals.json").read_text())
    cells: dict[str, dict[str, dict[str, dict[str, float]]]] = blob["cells"]
    qf = RES / f"w19-a1q-{tag}-lines.txt"
    if qf.exists():
        for m in ROW.finditer(qf.read_text()):
            task, ctx, arm = m.group(1), m.group(2), m.group(3)
            cells.setdefault(ctx, {}).setdefault(arm, {})[task] = {
                "acc": float(m.group(4)),
                "sbits": float(m.group(7) or m.group(6)),
                "n": float(m.group(8)),
            }
    return cells


def fig_fairquant() -> None:
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.4), sharex=True, sharey=True)
    for i, (tag, fam) in enumerate(FAMILIES):
        cells = _cells(tag)
        for j, (task, tlabel) in enumerate(TASKS):
            ax = axes[i][j]
            ax.grid(True, axis="y")
            ax.set_xscale("log")
            ax.set_xlim(0.035, 0.42)
            ax.set_ylim(-0.05, 1.08)
            ax.set_xticks([0.05, 0.1, 0.2, 0.4])
            ax.set_xticklabels(["0.05", "0.1", "0.2", "0.4"])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_yticks([0, 0.5, 1.0])
            for arm, (_label, colour, marker) in ARMS.items():
                pts = []
                for ctx in ("16384", "32768"):
                    c = cells.get(ctx, {}).get(arm, {}).get(task)
                    if c:
                        pts.append((ctx, c["sbits"], c["acc"]))
                if not pts:
                    continue
                if len(pts) == 2:
                    ax.plot(
                        [p[1] for p in pts],
                        [p[2] for p in pts],
                        color=colour,
                        lw=1,
                        alpha=0.5,
                        zorder=1,
                    )
                for ctx, x, y in pts:
                    hollow = ctx == "32768" or arm.endswith("-q4")
                    ax.plot(
                        x, y, marker=marker, ms=6.5, color=colour, mec=colour,
                        mfc="white" if hollow else colour, mew=1.6, ls="none", zorder=3,
                    )  # fmt: skip
            if i == 0:
                ax.set_title(tlabel, fontsize=8.5, color=INK)
            if j == 0:
                ax.set_ylabel(f"{fam}\naccuracy", fontsize=8)
            if i == 2:
                ax.set_xlabel("stored state / full KV", fontsize=8)

    def _h(marker: str, colour: str, label: str, hollow: bool) -> Line2D:
        fc = "white" if hollow else colour
        return Line2D(
            [], [], marker=marker, color=colour, mfc=fc, mec=colour, mew=1.6, ls="none",
            ms=6.5, label=label,
        )  # fmt: skip

    handles = [_h(m, c, lab, a.endswith("-q4")) for a, (lab, c, m) in ARMS.items()]
    handles += [_h("o", INK, "16K (filled)", False), _h("o", INK, "32K (hollow)", True)]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.02)
    )
    fig.suptitle(
        "Retrieval vs stored bytes: flagship, its 4-bit-coordinate compose cell, "
        "and the fair KIVI baseline (n=12)",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    _save(fig, "fairquant")


def fig_one_over_t() -> None:
    series: dict[str, dict[int, float]] = {
        a: {} for a in ("bugSseed-r64-h256", "quant-2bit-kivi", "quant-4bit-kivi")
    }
    for name in ("w19-a1-llama-lines.txt", "w18-llama-lines.txt", "w19-a4-llama-lines.txt"):
        f = RES / name
        if not f.exists():
            continue
        text = f.read_text()
        for m in ROW.finditer(text):
            arm, ctx, sb = m.group(3), int(m.group(2)), m.group(7)
            if arm in series and sb:
                series[arm].setdefault(ctx, float(sb))
        for m in PPL_RE.finditer(text):
            if m.group(1) in series:
                series[m.group(1)].setdefault(int(m.group(2)), float(m.group(3)))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.grid(True, axis="y")
    for arm, pts in series.items():
        if not pts:
            continue
        xs = sorted(pts)
        label, colour, marker = ARMS[arm]
        ax.plot(
            xs,
            [pts[x] for x in xs],
            color=colour,
            marker=marker,
            ms=6,
            lw=2,
            mec="white",
            mew=1,
            label=label,
        )
        ax.annotate(
            f"{pts[xs[-1]]:.3f}",
            (xs[-1], pts[xs[-1]]),
            textcoords="offset points",
            xytext=(6, -3),
            fontsize=7,
            color=INK,
        )
    ax.axhline(0.125, ls=":", lw=1, color=MUTED)
    ax.annotate(
        "2r/n = 0.125x (fp32-coord asymptote)",
        (16384, 0.125),
        textcoords="offset points",
        xytext=(2, 3),
        fontsize=6.5,
        color=MUTED,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([16384, 32768, 65536])
    ax.set_xticklabels(["16K", "32K", "64K"])
    ax.set_ylim(0, 0.32)
    ax.set_xlabel("context length T")
    ax.set_ylabel("stored state / full KV")
    ax.set_title(
        "Llama-3.1-8B: BUG amortizes toward 2r/n=0.125x; 2-bit toward 0.156x",
        fontsize=8,
        color=INK,
    )
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, "one_over_t")


def fig_coldstart() -> None:
    rows = PERSIST_RE.findall((RES / "w19-a3-llama2-lines.txt").read_text())
    order = ["full", "bugSseed-r64-h256", "quant-2bit-kivi"]
    labels = {
        "full": "full KV (fp16)",
        "bugSseed-r64-h256": "BUG flagship",
        "quant-2bit-kivi": "KIVI 2-bit",
    }
    colours = {"full": MUTED, "bugSseed-r64-h256": BLUE, "quant-2bit-kivi": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2), sharex=True)
    for ax, ctx in zip(axes, ("16384", "32768"), strict=True):
        ax.grid(True, axis="x")
        ys, vals = [], []
        for k, arm in enumerate(order):
            r = next(r for r in rows if r[0] == ctx and r[1] == arm)
            bytes_, cold = int(r[2]), float(r[7])
            ax.barh(k, cold, color=colours[arm], height=0.62)
            ax.text(
                cold + 0.03,
                k,
                f"{cold:.2f} s  ({bytes_ / 1e9:.2f} GB)",
                va="center",
                fontsize=7.5,
                color=INK,
            )
            ys.append(k)
            vals.append(cold)
        ax.set_yticks(ys)
        ax.set_yticklabels([labels[a] for a in order], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.55)
        ax.set_title(f"{int(ctx) // 1024}K context", fontsize=8.5, color=INK)
        ax.set_xlabel("seconds", fontsize=7.5)
    fig.suptitle(
        "Persisted-cache cold start to attend-ready (disk read + H2D + reconstruct), "
        "Llama-3.1-8B, A100-40GB, medians of 5",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "coldstart")


def _save(fig: Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote figures/week19/{name}.pdf|png]")


def main() -> None:
    _style()
    fig_fairquant()
    fig_one_over_t()
    fig_coldstart()


if __name__ == "__main__":
    main()
