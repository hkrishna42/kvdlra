"""8B recall of dropped-then-queried content: premium rank recovers what eviction
forgets (the rank-relative D1 result).

Two panels from the committed 8B runs (Llama-3.1-8B, n=1024, bf16):
  (A) recall vs BUG rank -- rank 64 fails everywhere (n doubled vs 1B), rank 128
      recovers except the hardest case, rank 256 recovers all; the fidelity
      threshold. Annotated with the 1B->8B rank doubling (n: 512 -> 1024).
  (B) the airtight win -- in the cases where eviction (MorphKV) genuinely forgets,
      premium-rank BUG fully recovers, at a stated memory premium.

Regenerate: ``uv run python scripts/w9_recall_8b.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import _paths  # noqa: F401
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RS = "results/w9-recovery-8b-ranksweep.json"
WS = "results/w9-recovery-8b-win-single.json"
WM = "results/w9-recovery-8b-win-multikey.json"
MORPH_MEM = 16_515_072  # morph per-run stored floats (the 1.0x reference)


def get(path: str, ctx: int, prefix: str) -> tuple[float | None, int | None]:
    d = json.loads(Path(path).read_text())
    for rec in d["per_ctx"]:
        if rec["ctx"] == ctx:
            for row in rec["rows"]:
                if row["method"].startswith(prefix):
                    return row["accuracy"], row.get("mem_stored")
    return None, None


def rank_recall(path: str, ctx: int, rank: int) -> tuple[float, float | None]:
    """Recall + memory-multiple at a given fidelity rank (bugFID; rank-64 falls
    back to bugA-r64, both 0)."""
    a, m = get(path, ctx, f"bugFID-r{rank}-")
    if a is None and rank == 64:
        a, m = get(path, ctx, "bugA-r64")
    mult = (m / MORPH_MEM) if m else None
    return (a if a is not None else float("nan")), mult


def main() -> None:
    cases = [
        ("single ctx2048", RS, 2048, "tab:gray"),
        ("single ctx4096", WS, 4096, "tab:purple"),
        ("multi-key ctx2048", WM, 2048, "tab:orange"),
        ("multi-key ctx4096", WM, 4096, "tab:red"),
    ]
    ranks = [64, 128, 256]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.5, 5.3))

    # ---- Panel A: recall vs BUG rank ----
    ax_a.axvspan(56, 96, color="tab:red", alpha=0.06)
    ax_a.text(72, 0.5, "rank-64\nfails\n(n=1024)", color="tab:red", fontsize=9, ha="center")
    for label, path, ctx, color in cases:
        ys = [rank_recall(path, ctx, r)[0] for r in ranks]
        ax_a.plot(ranks, ys, "-o", color=color, lw=2, ms=8, label=label)
    ax_a.axhline(1.0, color="0.7", ls=":", lw=1)
    ax_a.set_xticks(ranks)
    ax_a.set_xlabel("BUG rank  (fidelity-isolated: needle provably retained)")
    ax_a.set_ylabel("recall accuracy")
    ax_a.set_ylim(-0.05, 1.08)
    ax_a.set_title("8B: recall vs rank — the fidelity threshold")
    ax_a.legend(fontsize=8, loc="center right")
    ax_a.grid(alpha=0.3)
    ax_a.annotate(
        "1B needed rank 64 (n=512);\n8B needs rank 128 (n=1024)\n→ rank scales with n",
        xy=(128, 0.62),
        xytext=(150, 0.30),
        fontsize=8.5,
        color="0.3",
        arrowprops={"arrowstyle": "->", "color": "0.55", "lw": 1},
    )

    # ---- Panel B: the airtight win (forgetting cases) ----
    win_cases = [
        ("single\nctx4096", WS, 4096),
        ("multi-key\nctx2048", WM, 2048),
        ("multi-key\nctx4096", WM, 4096),
    ]
    methods = [
        ("eviction (morph)", "morph", "tab:green"),
        ("BUG r128 (premium)", "bugFID-r128", "tab:orange"),
        ("BUG r256 (premium)", "bugFID-r256", "tab:brown"),
    ]
    x = np.arange(len(win_cases))
    w = 0.26
    for i, (mlabel, mpref, color) in enumerate(methods):
        vals, mults = [], []
        for _, path, ctx in win_cases:
            a, m = get(path, ctx, mpref)
            vals.append(a if a is not None else 0.0)
            mults.append((m / MORPH_MEM) if m else 1.0)
        bars = ax_b.bar(x + (i - 1) * w, vals, w, color=color, label=mlabel)
        for b, mm, v in zip(bars, mults, vals, strict=True):
            if mpref != "morph":
                ax_b.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.02,
                    f"{mm:.1f}x",
                    ha="center",
                    fontsize=7.5,
                    color=color,
                )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([c[0] for c in win_cases])
    ax_b.set_ylabel("recall accuracy")
    ax_b.set_ylim(0, 1.12)
    ax_b.set_title("8B: premium BUG recovers what eviction forgets\n(x = memory vs morph)")
    ax_b.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax_b.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "D1 at 8B (Llama-3.1-8B): the recovery-tier win is rank-relative — "
        "needs rank scaled to the feature dim, at a memory premium",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = Path("figures/week9/recall_8b")
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=150)
    print(f"[wrote {out.with_suffix('.png')}]")


if __name__ == "__main__":
    main()
