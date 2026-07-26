"""Week-13 Track-X (exploratory) CPU probe: effective rank of keys vs depth.

QUESTION. The mean-field / interacting-particle picture predicts that token
representations *cluster* as they flow up the stack, so the columns of the key
matrix should grow collinear and its **effective rank should collapse with
layer depth**. If true, that both explains "why low-rank works" and would fund a
follow-up: allocate BUG rank by depth (thin the deep layers, fatten the shallow).

This is a $0 CPU probe on captured 1B dumps -- no GPU, no model download, no
training, and it touches nothing under ``src/``. It only *describes* the geometry
of the captured keys.

METHOD. Per ``docs/notes/conventions.md`` the key feature matrix is

    m = k.transpose(0, 2, 1).reshape(H * D, T)     # rows = 8*64 = 512 features,
                                                   # cols = T tokens

We drop the first ``N_SINK`` token *columns* (StreamingLLM attention sinks are
low-rank outliers; PLAN pitfall #5). For each layer we take the singular values
``sigma_i`` of ``m`` (no centering -- this is the raw matrix BUG factors) and
report the effective rank TWO ways so the verifier can cross-check:

    (a) participation ratio       PR    = (sum_i sigma_i)^2 / sum_i sigma_i^2
    (b) entropy effective rank    ERANK = exp(-sum_i p_i log p_i),
                                          p_i = sigma_i^2 / sum_j sigma_j^2

Both are computed for ``K_pre`` (pre-RoPE -- the geometry BUG actually operates
on) and ``K`` (post-RoPE -- what is stored/attended), averaged over all docs.

VERDICT (pre-registered). ``passed_bar=true`` iff a clear, roughly-monotone
collapse with depth is seen on the **pre-RoPE** keys (the mean-field claim is
about the model's own token geometry, which RoPE only smears): for K_pre, BOTH
metrics must have Spearman rho(depth, rank) <= -0.60 AND a >= 15% relative
decline from layer 0 to layer 15. Otherwise ``passed_bar=false`` and we report
the trend honestly (flat / non-monotone / rising is a real, useful negative).
``kill=false`` always -- this is an explanatory track, not a fundable mechanism.

PROXY CAVEAT. Effective rank of the *captured* keys is a descriptive statistic,
not a direct perplexity/retrieval lever. A depth trend here would motivate, but
not by itself justify, depth-varying rank allocation; that must be confirmed
end-to-end (recon-error-at-fixed-rank, then ppl/retrieval).

Usage:

    uv run python scripts/w13_trackx_rank_probe.py \
        --dumps dumps/llama3.2-1b --out-json results/w13-trackx-rank.json
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
from pathlib import Path

import _paths  # noqa: F401  # bootstrap kvdlra import path (unused here, kept for parity)
import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import spearmanr

N_SINK = 4
N_LAYERS = 16
KEYS = ("K_pre", "K")  # pre-RoPE (BUG's domain) and post-RoPE (stored/attended)
Arr = npt.NDArray[np.float64]

# Pre-registered collapse bar (applied to the pre-RoPE curves).
RHO_BAR = -0.60
DECLINE_BAR = 0.15


def feature_matrix(k: torch.Tensor, n_sink: int) -> Arr:
    """(H, T, D) key tensor -> (H*D, T-n_sink) feature-by-token matrix, fp64.

    Follows ``docs/notes/conventions.md``: rows = head_dim * num_kv_heads = 512
    features, columns = tokens; the first ``n_sink`` token columns are dropped.
    """
    arr = k.double().numpy()
    h, _t, d = arr.shape
    m: Arr = arr.transpose(0, 2, 1).reshape(h * d, -1)  # (H*D, T)
    return m[:, n_sink:]


def effective_ranks(m: Arr) -> tuple[float, float]:
    """Return (participation ratio, entropy effective rank) of matrix ``m``."""
    s = np.linalg.svd(m, compute_uv=False)  # singular values, descending
    s = s[s > 0.0]
    s2 = s * s
    pr = float(s.sum() ** 2 / s2.sum())
    p = s2 / s2.sum()
    erank = float(np.exp(-(p * np.log(p)).sum()))
    return pr, erank


def trend_stats(mean_curve: list[float]) -> dict[str, float]:
    """Depth-trend descriptors for one 16-long effective-rank curve."""
    depth = np.arange(len(mean_curve), dtype=np.float64)
    y = np.asarray(mean_curve, dtype=np.float64)
    rho, pval = spearmanr(depth, y)
    slope = float(np.polyfit(depth, y, 1)[0])
    l0, l15 = float(y[0]), float(y[-1])
    rel_decline = (l0 - l15) / l0  # >0 means it fell from shallow to deep
    diffs = np.diff(y)
    return {
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "lin_slope_per_layer": slope,
        "layer0": l0,
        "layer15": l15,
        "peak_layer": int(np.argmax(y)),
        "peak_value": float(y.max()),
        "min_layer": int(np.argmin(y)),
        "rel_decline_l0_to_l15": float(rel_decline),
        "frac_steps_decreasing": float((diffs < 0).mean()),
    }


def collapse_confirmed(trends: dict[str, dict[str, float]]) -> bool:
    """Pre-registered gate on the pre-RoPE (K_pre) curves for BOTH metrics."""
    ok = True
    for metric in ("pr", "erank"):
        t = trends[f"K_pre::{metric}"]
        ok &= t["spearman_rho"] <= RHO_BAR
        ok &= t["rel_decline_l0_to_l15"] >= DECLINE_BAR
    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", type=Path, default=Path("dumps/llama3.2-1b"))
    ap.add_argument("--glob", default="*_len4096_rope-both", help="doc dir glob under --dumps")
    ap.add_argument("--n-sink", type=int, default=N_SINK)
    ap.add_argument("--out-json", type=Path, default=Path("results/w13-trackx-rank.json"))
    args = ap.parse_args()

    doc_dirs = sorted(glob.glob(str(args.dumps / args.glob)))
    if not doc_dirs:
        raise SystemExit(f"no doc dirs match {args.dumps / args.glob}")

    # per_doc[doc][key]["pr"/"erank"] = list over layers
    per_doc: dict[str, dict[str, dict[str, list[float]]]] = {}
    for dd in doc_dirs:
        doc = Path(dd).name
        per_doc[doc] = {k: {"pr": [], "erank": []} for k in KEYS}
        for layer in range(N_LAYERS):
            fp = Path(dd) / f"layer_{layer:02d}.pt"
            blob = torch.load(fp, map_location="cpu")
            for k in KEYS:
                m = feature_matrix(blob[k], args.n_sink)
                pr, erank = effective_ranks(m)
                per_doc[doc][k]["pr"].append(pr)
                per_doc[doc][k]["erank"].append(erank)
                del m
            del blob
            gc.collect()
        print(f"[done] {doc}")

    docs = list(per_doc)
    # Aggregate mean/std across docs, per layer, per key, per metric.
    mean: dict[str, dict[str, list[float]]] = {}
    std: dict[str, dict[str, list[float]]] = {}
    for k in KEYS:
        mean[k], std[k] = {}, {}
        for metric in ("pr", "erank"):
            stack = np.array([per_doc[d][k][metric] for d in docs])  # (n_doc, 16)
            mean[k][metric] = stack.mean(axis=0).tolist()
            std[k][metric] = stack.std(axis=0).tolist()

    trends: dict[str, dict[str, float]] = {}
    for k in KEYS:
        for metric in ("pr", "erank"):
            trends[f"{k}::{metric}"] = trend_stats(mean[k][metric])

    passed = collapse_confirmed(trends)

    out = {
        "probe": "w13-trackx-effective-rank-vs-depth",
        "meta": {
            "dumps": str(args.dumps),
            "docs": docs,
            "n_docs": len(docs),
            "n_layers": N_LAYERS,
            "n_sink_dropped": args.n_sink,
            "feature_dim": 512,
            "metrics": {
                "pr": "(sum sigma_i)^2 / sum sigma_i^2",
                "erank": "exp(-sum p_i log p_i), p_i = sigma_i^2 / sum sigma^2",
            },
            "centering": "none (raw feature-by-token matrix BUG factors)",
            "keys": {"K_pre": "pre-RoPE (BUG domain)", "K": "post-RoPE (stored)"},
            "bar": {
                "rule": "K_pre BOTH metrics: spearman_rho<=-0.60 AND rel_decline_l0_to_l15>=0.15",
                "rho_bar": RHO_BAR,
                "decline_bar": DECLINE_BAR,
            },
        },
        "per_doc": per_doc,
        "mean_over_docs": mean,
        "std_over_docs": std,
        "trends": trends,
        "verdict": {
            "collapse_confirmed": passed,
            "passed_bar": passed,
            "kill": False,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))

    # Console summary.
    print(f"\n=== effective rank vs depth ({len(docs)} docs) ===")
    for k in KEYS:
        print(f"\n[{k}]  layer:  PR (mean)   ERANK (mean)")
        for layer in range(N_LAYERS):
            print(f"  L{layer:02d}   {mean[k]['pr'][layer]:8.2f}   {mean[k]['erank'][layer]:8.3f}")
    print("\n--- trend stats (on doc-mean curve) ---")
    for name, t in trends.items():
        print(
            f"  {name:14s} rho={t['spearman_rho']:+.3f} "
            f"slope={t['lin_slope_per_layer']:+.3f}/layer "
            f"L0={t['layer0']:.2f} L15={t['layer15']:.2f} "
            f"reldecl={t['rel_decline_l0_to_l15']:+.3f} "
            f"peak@L{t['peak_layer']}"
        )
    print(f"\ncollapse_confirmed (pre-registered bar): {passed}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
