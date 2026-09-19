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

`rank-sweep` is not one of those three: it draws the stored-representation study
(`kvdlra.eval.recon`) from a `recon.jsonl`, which is a diagnostic, not a paper figure.

    python scripts/figures.py --out docs/paper/figures            # the three, `build`
    python scripts/figures.py rank-sweep --in results/recon_1b/recon.jsonl \
        --out docs/paper/figures/rank_sweep_1b.pdf
"""

from __future__ import annotations

import argparse
import re
from functools import cache
from math import sqrt
from pathlib import Path
from statistics import stdev
from typing import cast

import _paths  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
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


# --- the stored-representation study (`kvdlra.eval.recon`) ----------------------

RECON_STYLE = {  # row key -> (label, colour, dash); the isvd floor variant is dashed isvd
    "svd_oracle": ("per-sequence SVD oracle", INK, (0, (4, 2))),
    "isvd": ("incremental SVD", BLUE, "-"),
    "isvd_f0.01": ("incremental SVD, sv floor 0.01", BLUE, (0, (3, 1.5))),
    "fd": ("Frequent Directions", AQUA, "-"),
    "fd2": ("Frequent Directions, l=2r", AQUA, (0, (1, 1.5))),
    "oja": ("Oja's rule", ORANGE, "-"),
    "frozen_prefill_svd": ("frozen prefill basis", MUTED, "-"),
    "random_basis": ("random basis", MUTED, (0, (1, 2))),
}


def _recon_key(r: dict[str, object]) -> str:
    """One line per method, with the isvd singular-value floor as its own line."""
    floor = cast(float, r["min_sv_frac"])
    return f"{r['method']}_f{floor:g}" if floor else str(r["method"])


def fig_rank_sweep(src: Path, out: Path, block: int | None = None) -> None:
    """Reconstruction error on the representation each method STORES, vs stored width.

    One panel per kv, one line per method, mean +- SE over documents: the document is the
    sampling unit, so a document's layers -- and both block sizes, unless `--block` pins
    one -- are averaged before the spread over documents is taken. x is the stored width
    (`stored_rank`), not the nominal rank, because FD at l=2r stores twice as much. The
    `random_basis` control shares ONE Haar draw (`seed=0`) across every cell, so its band
    is document variation, not draw-to-draw noise.

    Exits non-zero when the file has nothing to draw, or when any series in `RECON_STYLE`
    has no rows at all: a study run without `min_sv_frac=(0.0, 0.01)` has no `isvd_f0.01`
    rows, and a silently absent line is an omission no reader can see.
    """
    rows = [r for r in read_jsonl(src) if block is None or r["block"] == block]
    if not rows:
        raise SystemExit(f"no rows for {src}" + ("" if block is None else f" at block={block}"))
    missing = [k for k in RECON_STYLE if not any(_recon_key(r) == k for r in rows)]
    if missing:
        raise SystemExit(
            f"{src}: no rows for {', '.join(missing)} -- rerun the study with every method "
            "(`isvd_f0.01` is `min_sv_frac=(0.0, 0.01)`), or drop the series from RECON_STYLE"
        )
    kvs = sorted({str(r["kv"]) for r in rows})
    fig, axes = plt.subplots(1, len(kvs), figsize=(3.7 * len(kvs), 2.9), sharey=True, squeeze=False)
    for ax, kv in zip(axes[0], kvs, strict=True):
        ax.grid(True, axis="y")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        for key, (label, colour, dash) in RECON_STYLE.items():
            sel = [r for r in rows if str(r["kv"]) == kv and _recon_key(r) == key]
            if not sel:
                continue
            xs: list[float] = []
            ys: list[float] = []
            es: list[float] = []
            for rank in sorted({cast(int, r["rank"]) for r in sel}):
                at = [r for r in sel if r["rank"] == rank]
                per_doc: dict[str, list[float]] = {}
                for r in at:
                    per_doc.setdefault(str(r["doc"]), []).append(cast(float, r["err"]))
                means = [sum(v) / len(v) for v in per_doc.values()]
                xs.append(sum(cast(int, r["stored_rank"]) for r in at) / len(at))
                ys.append(sum(means) / len(means))
                es.append(stdev(means) / sqrt(len(means)) if len(means) > 1 else 0.0)
            ax.errorbar(
                xs, ys, yerr=es, color=colour, ls=dash, marker="o", ms=3, lw=1.4,
                capsize=2, label=label,
            )  # fmt: skip
        ax.set_title(kv, fontsize=8.5, color=INK)
        ax.set_xlabel("stored width (columns of U)", fontsize=8)
    axes[0][0].set_ylabel("||M - UC||_F / ||M||_F", fontsize=8)
    # Below the panels, not inside one: the curves fall left-to-right and would sit under
    # an in-axes legend at any rank grid narrower than the study's. Merged over every
    # panel, not read off the first: a method whose rows cover only one kv is labelled too.
    legend: dict[str, Artist] = {}
    for ax in axes[0]:
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles, strict=True))
    fig.legend(
        list(legend.values()), list(legend.keys()),
        loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02),
    )  # fmt: skip
    ndocs = len({str(r["doc"]) for r in rows})
    fig.suptitle(
        f"Reconstruction error of the STORED representation, mean +- SE over {ndocs} documents"
        + ("" if block is None else f", block={block}"),
        fontsize=9,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote {out}]")


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
    sub = ap.add_subparsers(dest="cmd")  # optional: no subcommand means `build`
    # SUPPRESS so `--out` before the subcommand survives it (argparse would otherwise
    # overwrite the parsed value with the subparser's default).
    sub.add_parser("build", help="the three paper figures (the default)").add_argument(
        "--out", default=argparse.SUPPRESS
    )
    rs = sub.add_parser("rank-sweep", help="the stored-representation study (4.1)")
    rs.add_argument("--in", dest="src", required=True, help="a recon.jsonl from kvdlra.eval.recon")
    rs.add_argument("--out", required=True, help="the .pdf to write")
    rs.add_argument("--block", type=int, help="keep only this block size (default: all, averaged)")
    args = ap.parse_args()
    if args.cmd == "rank-sweep":
        _style()
        fig_rank_sweep(Path(args.src), Path(args.out), args.block)
    else:
        build(Path(args.out))


if __name__ == "__main__":
    main()
