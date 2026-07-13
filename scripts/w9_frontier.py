"""The classic comparison: streaming perplexity vs KV-cache memory budget.

Eviction (MorphKV) vs BUG (DLRA low-rank) vs the full O(T) cache, on matched
per-layer memory, at 1B (WikiText-2, doc0, prefill 512 + 1024 streamed decode
tokens -- one consistent protocol across all budgets). This is the load-bearing
"BUG vs eviction" frontier of the whole project: they cross. Eviction wins at
extreme compression (its `2n`/token beats a rank-squeezed low-rank summary); BUG
wins at moderate budgets (a near-oracle low-rank summary of *all* history beats a
handful of exact tokens), approaching the full-cache floor.

Data (both same protocol, ``docs/week9.md`` D2): ``results/w7-ranksweep-attn-doc0``
(tau=89) + ``results/w9-envelope-gate-1b`` (tau=128..360). Regenerate the figure:
``uv run python scripts/w9_frontier.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import _paths  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def load_frontier() -> dict[str, list[float]]:
    """Assemble (tau, full, morph, bugA-best) from the two same-protocol runs."""
    taus: list[float] = []
    full: list[float] = []
    morph: list[float] = []
    bug: list[float] = []

    w7 = json.loads(Path("results/w7-ranksweep-attn-doc0.json").read_text())["docs"]["0"]
    taus.append(89.0)
    full.append(w7["full"]["ppl"])
    morph.append(next(v["ppl"] for k, v in w7.items() if k.startswith("morph")))
    bug.append(min(v["ppl"] for k, v in w7.items() if k.startswith("bugA")))

    g = json.loads(Path("results/w9-envelope-gate-1b.json").read_text())
    full_ref = full[0]  # same doc0 full baseline (gate didn't run the full arm)
    for tau, b in g["budgets"].items():
        doc = b["docs"]["0"]
        taus.append(float(tau))
        full.append(full_ref)
        morph.append(next(v["ppl"] for k, v in doc.items() if k.startswith("morph")))
        bug.append(min(v["ppl"] for k, v in doc.items() if k.startswith("bugA")))
    order = sorted(range(len(taus)), key=lambda i: taus[i])
    return {
        "tau": [taus[i] for i in order],
        "full": [full[i] for i in order],
        "morph": [morph[i] for i in order],
        "bug": [bug[i] for i in order],
    }


def make_figure(d: dict[str, list[float]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    tau = d["tau"]
    # crossover ~ where morph and bug meet (between 89 and 128).
    xc = 122.0
    ax.axvspan(min(tau) * 0.9, xc, color="tab:green", alpha=0.06)
    ax.axvspan(xc, max(tau) * 1.1, color="tab:orange", alpha=0.06)
    ax.axvline(xc, color="0.5", ls=":", lw=1)

    ax.plot(tau, d["full"], ls="--", color="tab:blue", lw=1.6, marker="", label="full cache (O(T))")
    ax.plot(
        tau,
        d["morph"],
        "-s",
        color="tab:green",
        lw=2.2,
        ms=8,
        label="eviction (MorphKV)",
    )
    ax.plot(
        tau,
        d["bug"],
        "-o",
        color="tab:orange",
        lw=2.2,
        ms=8,
        label="BUG (DLRA low-rank)",
    )

    ax.text(96, 21.6, "eviction\nwins", color="tab:green", fontsize=10, ha="center", va="top")
    ax.text(300, 14.2, "BUG wins", color="tab:orange", fontsize=11, ha="center")
    ax.annotate(
        "crossover\n~130 tok-eq/layer",
        xy=(xc, 16.5),
        xytext=(150, 19.2),
        fontsize=9,
        color="0.35",
        arrowprops={"arrowstyle": "->", "color": "0.55", "lw": 1},
    )

    ax.set_xscale("log")
    ax.set_xticks(tau)
    ax.set_xticklabels([f"{t:.0f}" for t in tau])
    ax.xaxis.set_minor_formatter(NullFormatter())  # drop 2x10^2 etc. clutter
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel("KV-cache memory budget  (token-equivalents / layer)  →  more compression left")
    ax.set_ylabel("streaming perplexity  (lower is better)")
    ax.set_title(
        "BUG vs eviction: streaming ppl at matched memory\nLlama-3.2-1B, WikiText-2 "
        "(same protocol; they cross)",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.3)
    ax.margins(x=0.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=150)
    print(f"[wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}]")


def main() -> None:
    d = load_frontier()
    print("frontier points:")
    for i, t in enumerate(d["tau"]):
        print(
            f"  tau={t:.0f}  full={d['full'][i]:.2f}  "
            f"morph={d['morph'][i]:.2f}  bug={d['bug'][i]:.2f}"
        )
    make_figure(d, Path("figures/week9/comparison_frontier"))


if __name__ == "__main__":
    main()
