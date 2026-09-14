"""The paper figures, built from the archived records (never retyped numbers).

Three figures, one per `\\includegraphics` in paper/main.tex @ee8c0ab:

* ``fairquant`` -- retrieval vs stored bytes, small multiples (rows = family, cols =
  task): the r64 arm, the sub-cliff compose cell, KIVI 2-bit and 4-bit; 16K filled, 32K
  hollow, the two joined by a thin segment (the r64 arm's drifts left = its 1/T term).
* ``one_over_t`` -- stored ratio vs context (Llama): r64 amortizes toward its
  2r/n=0.125x asymptote, the 2-bit arm toward 0.156x. Picks up the 64K point
  automatically when the w19-a4-llama archive directory exists.
* ``coldstart`` -- persisted-cache cold start (seconds) at 16K/32K: full vs r64 vs 2-bit.
  The protocol behind those rows -- warm page-cache read, H2D, reconstruct, median of 5
  repeats -- is `scripts/pod/w19.sh` @ `paper-v1-archive` (its persist block,
  ``--repeats 5``); the
  figure's subtitle states it, so this is where that claim comes from.

Every accuracy is counted from `results/paper-v1/<pod>/trials.jsonl`, the same per-trial
provenance `scripts/tables.py build` uses -- so a figure and a table can never disagree.
Stored state is a property of the run rather than of a needle, so it comes from the
archived aggregate rows (`cells.jsonl`, `ppl.jsonl`). The persisted-cache rows the
cold-start figure needs are a Week-19 storage-bench format that no record type covers;
they are read from that pod's verbatim `raw/` copy, which is archived for exactly this.

Palette = the dataviz reference instance, first three categorical slots (validated
all-pairs); colour follows the entity across every figure (r64 blue, 2-bit orange,
4-bit aqua; the compose cell is r64's hue with a hollow diamond; full KV is ink).

    python scripts/figures.py --out docs/paper/figures
"""

from __future__ import annotations

import argparse
import re
from functools import cache
from pathlib import Path
from typing import cast

import _paths  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from kvdlra.eval.records import CellRecord, PplRecord, TrialRecord, read_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = REPO_ROOT / "results" / "paper-v1"
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
FAMILIES = [("llama", "Llama-3.1-8B"), ("mistral", "Mistral-7B-v0.3"), ("qwen", "Qwen2.5-7B")]
TASKS = [
    ("niah_single", "single"),
    ("niah_multikey", "multi-key"),
    ("niah_multivalue", "multi-value"),
    ("vt", "var-track"),
]
ARMS = {  # arm -> (label, colour, marker)
    "bugSseed-r64-h256": ("BUG r64 (fp32 at rest)", BLUE, "o"),
    "bugSseed-r64-h256-q4": ("BUG r64 + 4-bit coordinates", BLUE, "D"),
    "quant-2bit-kivi": ("KIVI 2-bit", ORANGE, "s"),
    "quant-4bit-kivi": ("KIVI 4-bit", AQUA, "^"),
}
CTXS = (16384, 32768)
# The persisted-cache bench printed its own row format; no record type parses it, so the
# cold-start figure reads the archived raw copy of that pod's line file.
PERSIST_RE = re.compile(
    r"^\[persist ctx(\d+)\] (\S+)\s+bytes=(\d+) ratio=([0-9.]+) save=[0-9.]+s load=([0-9.]+)s "
    r"h2d=([0-9.]+)s ready=([0-9.]+)s cold=([0-9.]+)s",
    re.M,
)


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


# --- the archive ---------------------------------------------------------------


def _read(pod: str, name: str) -> list[dict[str, object]]:
    """One archived artifact, or nothing when that pod never produced it (w19-a4-llama
    is the optional 64K point; w18-llama carries no perplexity rows of its own)."""
    path = ARCHIVE / pod / name
    return read_jsonl(path) if path.is_file() else []


@cache
def _trials(pod: str) -> tuple[TrialRecord, ...]:
    return tuple(cast(TrialRecord, r) for r in _read(pod, "trials.jsonl"))


@cache
def _cells(pod: str) -> tuple[CellRecord, ...]:
    return tuple(cast(CellRecord, r) for r in _read(pod, "cells.jsonl"))


@cache
def _ppl(pod: str) -> tuple[PplRecord, ...]:
    return tuple(cast(PplRecord, r) for r in _read(pod, "ppl.jsonl"))


def _acc_pods(tag: str) -> tuple[str, ...]:
    """Where one family's per-trial records live: the KIVI arms and the r64 arm ran on
    different pods, and the compose cell on a third."""
    return (f"w19-a1-{tag}", f"w19-a1q-{tag}", f"w18-g1-{tag}")


def _mem_pods(tag: str) -> tuple[str, ...]:
    """Where the same family's aggregate rows live. The w18-g1-* pods printed `[trial]`
    lines only; their run-level `ratio=`/`sbits=` rows went to the line file the archive
    holds under `w18-<tag>` (the mapping `scripts/tables.py` calls MEMORY_SOURCE)."""
    return (f"w19-a1-{tag}", f"w19-a1q-{tag}", f"w18-{tag}")


def _acc(pods: tuple[str, ...], arm: str, task: str, ctx: int) -> float | None:
    """Pooled accuracy of one cell, counted from the per-trial records."""
    rows = [
        r
        for pod in pods
        for r in _trials(pod)
        if r["arm"] == arm and r["task"] == task and r["ctx"] == ctx
    ]
    return sum(r["hit"] for r in rows) / len(rows) if rows else None


def _stored(pods: tuple[str, ...], arm: str, task: str, ctx: int) -> float | None:
    """Stored state of one cell: fp32-at-rest bits when the pod printed them, else the
    float-equivalent ratio (the pre-Week-18 rows print only the latter)."""
    for pod in pods:
        for r in _cells(pod):
            if r["arm"] == arm and r["task"] == task and r["ctx"] == ctx:
                return r["sbits"] if r["sbits"] is not None else r["ratio"]
    return None


# --- the figures ---------------------------------------------------------------


def fig_fairquant(out: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.4), sharex=True, sharey=True)
    for i, (tag, fam) in enumerate(FAMILIES):
        acc_pods, mem_pods = _acc_pods(tag), _mem_pods(tag)
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
                pts = [
                    (ctx, _stored(mem_pods, arm, task, ctx), _acc(acc_pods, arm, task, ctx))
                    for ctx in CTXS
                ]
                pts = [p for p in pts if p[1] is not None and p[2] is not None]
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
                    hollow = ctx == 32768 or arm.endswith("-q4")
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
        "Retrieval vs stored bytes: the r64 arm, its 4-bit-coordinate compose cell, "
        "and the fair KIVI baseline (n=12)",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    _save(fig, "fairquant", out)


def fig_one_over_t(out: Path) -> None:
    series: dict[str, dict[int, float]] = {
        a: {} for a in ("bugSseed-r64-h256", "quant-2bit-kivi", "quant-4bit-kivi")
    }
    for pod in ("w19-a1-llama", "w18-llama", "w19-a4-llama"):
        for c in _cells(pod):
            sb = c["sbits"]
            if c["arm"] in series and sb is not None:
                series[c["arm"]].setdefault(c["ctx"], sb)
        for p in _ppl(pod):
            sb = p["sbits"]
            if p["arm"] in series and sb is not None:
                series[p["arm"]].setdefault(p["ctx"], sb)
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
        "Llama-3.1-8B: BUG r64 amortizes toward 2r/n=0.125x; 2-bit toward 0.156x",
        fontsize=8,
        color=INK,
    )
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, "one_over_t", out)


def fig_coldstart(out: Path) -> None:
    pod = "w19-a3-llama2"
    rows = PERSIST_RE.findall((ARCHIVE / pod / "raw" / f"{pod}-lines.txt").read_text())
    order = ["full", "bugSseed-r64-h256", "quant-2bit-kivi"]
    labels = {
        "full": "full KV (fp16)",
        "bugSseed-r64-h256": "BUG r64",
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
        "Persisted-cache cold start to attend-ready (warm page-cache read + H2D + reconstruct), "
        "Llama-3.1-8B, A100-40GB, medians of 5",
        fontsize=9,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "coldstart", out)


def _save(fig: Figure, name: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote {out}/{name}.pdf|png]")


def build(out: Path) -> None:
    _style()
    fig_fairquant(out)
    fig_one_over_t(out)
    fig_coldstart(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/paper/figures")
    build(Path(ap.parse_args().out))


if __name__ == "__main__":
    main()
