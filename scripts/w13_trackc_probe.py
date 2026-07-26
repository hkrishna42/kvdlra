"""Week-13 Track-C CPU probe: does LONG-context query-whitening calibration
beat a SHORT-prefix calibration for Q-BUG's key gist?

The shipped 8B Q-BUG whitened its low-rank key gist with a diagonal metric ``L``
whose query second moment ``Q = E[q q^T]`` was calibrated on ~160-token docs, and
it bought only ~0.4-1.2% perplexity. Track-C asks whether that meagre payoff is a
*calibration-length* artifact -- would an ``L`` fit on the full long context read
the directions long-range attention actually probes and pay off more? -- or an
*intrinsic* limit of diagonal whitening that a longer calibration cannot move.

The 1B dumps carry full 4096-token pre-RoPE queries (``Q_pre``), so both ``L``'s
can be compared on CPU, $0, no GPU, no new training.

Design (per doc, per layer, honoring GQA 4:1 + the pre/post-RoPE split, exactly
as ``scripts/w12_qbug_probe.py``):

* The **evaluation is always the full 4096-token context**: causal attention
  output error ``||A V - A_hat V||`` with the TRUE V, so the delta isolates the
  KEY reconstruction (identical metric + code path as w12's ``out_err``).
* Only the whitening ``L`` differs. ``L`` is the diagonal factor
  ``diag(sqrt(diag(Q_j)))`` per KV head, built two ways:
  - SHORT: ``Q_j`` from ``Q_pre[:, :160, :]`` (mimics the shipped short-doc fit),
  - LONG:  ``Q_j`` from the full ``Q_pre[:, :4096, :]`` (the recalibration).
  Both reconstruct the SAME rank-r key gist (one global SVD over the whitened,
  stacked GQA heads, exactly the deployed shared-rank basis), un-whiten, rotate to
  post-RoPE, and are scored by the full-context attention error above.

We reuse the w12 building blocks verbatim (``head_query_moments``, ``whiten_l``,
``whitened_recon``, ``plain_recon``, ``apply_rope``, ``attention_output``) -- that
file is imported, never edited. w12's ``out_err`` is a closure inside
``probe_layer`` (not importable), so its two lines are reconstructed here from the
imported ``apply_rope`` + ``attention_output`` (bit-for-bit the same computation).

Because w12 already fit ``Q`` on the FULL context, w12's ``diag_r{r}`` numbers are
the LONG-L payoff at the same docs/layers -- this probe's ``long_r{r}`` should
reproduce them (a built-in cross-check) and adds the SHORT-L arm to compare.

Bars (pre-registered):
* FUND (calibration length matters -> a cheap 8B long-doc recalibration is worth
  it): LONG-L beats SHORT-L by a clear margin -- relative attention-error gain
  ``100*(short_err - long_err)/plain_err`` >= ``--margin-bar`` (default 3%),
  layer-averaged, on >= 4/5 docs, at a deployed rank (r32 or r128).
* KILL (calibration length is irrelevant -> recalibration won't move the bounded
  result): LONG-L ~= SHORT-L (margin below the bar at both ranks).

We also report, per rank, whether EITHER L clears the w12 >=3% payoff-over-plain
bar, and the cosine between the SHORT and LONG diagonal ``L`` vectors -- a cheap
mechanistic explanation of the margin (high cosine => the per-dim query-energy
profile is stationary => short and long fits coincide).

CRITICAL PROXY CAVEAT: this is the SAME attention-output-error proxy w12 used,
which OVER-PREDICTED end-to-end perplexity by ~30-40x. Every number here predicts
the DIRECTION and RELATIVE ordering of SHORT vs LONG, never an absolute ppl.

Usage:

    uv run python scripts/w13_trackc_probe.py --dumps dumps/llama3.2-1b \
        --ranks 32 128 --out-json results/w13-trackc-probe.json
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
from pathlib import Path

import _paths  # noqa: F401  # bootstrap kvdlra import path
import numpy as np
import numpy.typing as npt
import torch
from w12_qbug_probe import (  # reuse verbatim; that file is never edited
    apply_rope,
    attention_output,
    head_query_moments,
    plain_recon,
    whiten_l,
    whitened_recon,
)

Arr = npt.NDArray[np.float64]
N_SHORT = 160  # tokens in the short-prefix calibration (the shipped 8B doc length)


def l_diag_cosine(moms_a: list[Arr], moms_b: list[Arr]) -> float:
    """Mean per-KV-head cosine between two diagonal ``L`` vectors ``sqrt(diag(Q))``.

    Near 1.0 means the per-dimension query-energy profile is the same whether fit
    on the short prefix or the full context -- so the diagonal whitening (which
    reads only that profile) cannot differ, mechanistically explaining a null
    payoff margin. Uses ``whiten_l``'s own diag construction so it matches exactly.
    """
    ls_a, _ = whiten_l(moms_a, "diag")
    ls_b, _ = whiten_l(moms_b, "diag")
    sims: list[float] = []
    for la, lb in zip(ls_a, ls_b, strict=True):
        a, b = np.diag(la), np.diag(lb)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        sims.append(float(a @ b) / denom if denom > 0 else 0.0)
    return float(np.mean(sims))


def probe_layer_trackc(
    dump_dir: Path,
    layer: int,
    rope: dict[str, torch.Tensor],
    ranks: list[int],
    n_kv: int,
    mu_frac: float,
    n_short: int,
) -> dict[str, float]:
    """One layer: plain vs SHORT-L vs LONG-L diag-whitened key-gist attention error.

    Evaluation is the full-context attention error against the true output (true V,
    isolating the KEY reconstruction) -- w12's metric. Only the calibration length
    of the diagonal whitening ``L`` differs between the short and long arms.
    """
    blob = torch.load(dump_dir / f"layer_{layer:02d}.pt", weights_only=False)
    k_pre = blob["K_pre"].float()  # (h, T, d)
    q_pre = blob["Q_pre"].float()  # (H_q, T, d)
    v = blob["V"].float()
    cos, sin = rope["cos"].float(), rope["sin"].float()

    # Full-context ground-truth attention (queries are the whole 4096, always).
    q_post = apply_rope(q_pre, cos, sin)
    o_true = attention_output(q_post, apply_rope(k_pre, cos, sin), v)
    o_norm = float(torch.linalg.norm(o_true))

    def out_err(k_pre_hat: torch.Tensor) -> float:
        # Identical to w12 probe_layer's out_err closure.
        o_hat = attention_output(q_post, apply_rope(k_pre_hat.float(), cos, sin), v)
        return float(torch.linalg.norm(o_hat - o_true)) / o_norm

    # The only difference between the arms: the calibration window for Q -> L.
    moms_short = head_query_moments(q_pre[:, :n_short, :], n_kv, mu_frac)
    moms_long = head_query_moments(q_pre, n_kv, mu_frac)  # full 4096

    res: dict[str, float] = {"l_diag_cos": l_diag_cosine(moms_short, moms_long)}
    for r in ranks:
        res[f"plain_r{r}"] = out_err(plain_recon(k_pre, r))
        res[f"short_r{r}"] = out_err(whitened_recon(k_pre, moms_short, r, "diag"))
        res[f"long_r{r}"] = out_err(whitened_recon(k_pre, moms_long, r, "diag"))
    # Big tensors (blob/k_pre/q_pre/v/q_post/o_true) are function-local and freed on
    # return; main() runs gc.collect() between layers so peak memory stays ~one layer.
    return res


def _rel_gain_pct(plain: float, err: float) -> float:
    """Relative attention-error reduction of ``err`` vs the plain gist, in percent."""
    return 100.0 * (plain - err) / plain if plain > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", default="dumps/llama3.2-1b")
    ap.add_argument("--ranks", type=int, nargs="+", default=[32, 128])
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="layer indices to probe (default: the w12 spread of 6, for cross-check)",
    )
    ap.add_argument("--n-short", type=int, default=N_SHORT, help="short-prefix calibration length")
    ap.add_argument(
        "--mu-frac", type=float, default=1e-3, help="Tikhonov floor for Q as a fraction of tr(Q)/d"
    )
    ap.add_argument(
        "--margin-bar",
        type=float,
        default=3.0,
        help="FUND if LONG beats SHORT by >= this %% (rel. to plain) on >= 4/5 docs",
    )
    ap.add_argument(
        "--payoff-bar",
        type=float,
        default=3.0,
        help="secondary: does EITHER L beat plain by >= this %%",
    )
    ap.add_argument("--out-json", default="results/w13-trackc-probe.json")
    args = ap.parse_args()

    dirs = sorted(
        p
        for p in glob.glob(f"{args.dumps}/*_rope-both")
        if (Path(p) / "rope.pt").exists()
        and json.loads((Path(p) / "meta.json").read_text()).get("with_q")
        and json.loads((Path(p) / "meta.json").read_text()).get("seq_len", 0) >= 4096
    )
    if not dirs:
        raise SystemExit(f"no with-q rope-both >=4096 dumps under {args.dumps}")

    aggs: list[dict[str, float]] = []  # per-doc layer-mean metrics (for the verdict)
    per_doc: list[dict[str, object]] = []  # richer records (for the JSON)
    for d in dirs:
        dp = Path(d)
        meta = json.loads((dp / "meta.json").read_text())
        n_kv = int(meta["num_key_value_heads"])
        n_layers = sum(1 for _ in dp.glob("layer_*.pt"))
        layers = args.layers or sorted({round(x) for x in np.linspace(1, n_layers - 1, 6)})
        rope = torch.load(dp / "rope.pt", weights_only=True)
        rows: list[dict[str, float]] = []
        for ell in layers:
            rows.append(
                probe_layer_trackc(dp, ell, rope, args.ranks, n_kv, args.mu_frac, args.n_short)
            )
            gc.collect()  # free the layer's tensors before loading the next (bound peak mem)
        agg = {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
        aggs.append(agg)
        per_doc.append({"doc": dp.name, "layers": layers, "agg": agg, "per_layer": rows})
        print(
            f"[{dp.name}] Lcos={agg['l_diag_cos']:.4f}  "
            + "  ".join(
                f"r{r}: plain={agg[f'plain_r{r}']:.4f} short={agg[f'short_r{r}']:.4f} "
                f"long={agg[f'long_r{r}']:.4f}"
                for r in args.ranks
            )
        )

    # Pre-registered verdict.
    n_docs = len(per_doc)
    need = max(4, int(0.8 * n_docs))
    verdict: dict[str, object] = {
        "n_docs": n_docs,
        "docs_needed": need,
        "n_short_tokens": args.n_short,
        "margin_bar_pct": args.margin_bar,
        "payoff_bar_pct": args.payoff_bar,
        "l_diag_cos_median": float(np.median([a["l_diag_cos"] for a in aggs])),
    }
    print(
        f"\n=== VERDICT (FUND: LONG beats SHORT by >= {args.margin_bar}% "
        f"on >= {need}/{n_docs} docs, at r32 or r128) ==="
    )
    any_fund = False
    for r in args.ranks:
        short_gains = [_rel_gain_pct(a[f"plain_r{r}"], a[f"short_r{r}"]) for a in aggs]
        long_gains = [_rel_gain_pct(a[f"plain_r{r}"], a[f"long_r{r}"]) for a in aggs]
        # margin = extra relative error reduction of LONG over SHORT (both vs plain)
        margins = [
            100.0 * (a[f"short_r{r}"] - a[f"long_r{r}"]) / a[f"plain_r{r}"]
            if a[f"plain_r{r}"] > 0
            else 0.0
            for a in aggs
        ]
        n_margin = sum(m >= args.margin_bar for m in margins)
        n_long_payoff = sum(g >= args.payoff_bar for g in long_gains)
        n_short_payoff = sum(g >= args.payoff_bar for g in short_gains)
        fund_r = n_margin >= need
        any_fund = any_fund or fund_r
        verdict[f"r{r}_short_gain_pct_median"] = float(np.median(short_gains))
        verdict[f"r{r}_long_gain_pct_median"] = float(np.median(long_gains))
        verdict[f"r{r}_margin_pct_median"] = float(np.median(margins))
        verdict[f"r{r}_margin_pct_min"] = float(np.min(margins))
        verdict[f"r{r}_margin_pct_max"] = float(np.max(margins))
        verdict[f"r{r}_margins_per_doc"] = [float(m) for m in margins]
        verdict[f"r{r}_n_docs_margin_ge_bar"] = int(n_margin)
        verdict[f"r{r}_n_docs_long_payoff_ge_bar"] = int(n_long_payoff)
        verdict[f"r{r}_n_docs_short_payoff_ge_bar"] = int(n_short_payoff)
        verdict[f"r{r}_fund_margin"] = bool(fund_r)
        print(
            f"  r{r}: short_payoff={np.median(short_gains):+.2f}%  "
            f"long_payoff={np.median(long_gains):+.2f}%  "
            f"margin(long-short)={np.median(margins):+.2f}% "
            f"[{np.min(margins):+.2f},{np.max(margins):+.2f}]  "
            f"docs margin>=bar {n_margin}/{n_docs}  -> {'FUND' if fund_r else 'kill'}"
        )
    verdict["overall_fund"] = bool(any_fund)
    verdict["overall_kill"] = bool(not any_fund)
    print(
        f"\nOVERALL: {'FUND' if any_fund else 'KILL'}  (median L-diag cosine "
        f"{verdict['l_diag_cos_median']:.4f})"
    )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"per_doc": per_doc, "verdict": verdict}, indent=2))
    print(f"\n[wrote {args.out_json}]")


if __name__ == "__main__":
    main()
