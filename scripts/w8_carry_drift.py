"""Week-8 CodeBUG gating experiment: does the coded-tier carry compound?

The load-bearing risk (variant D exploded 14.7x): storing coordinates as CODES
and re-expressing them every absorb re-injects quantization noise that compounds.
CodeBUG's fix -- freeze a reduced-rank anchor ``u_ref``, code each column ONCE at
graduation, never rotate -- makes the per-column error *structurally* fixed. This
script MEASURES that on real 1B KV before any perplexity number is believed, per
the unanimous Phase-0 panel recommendation.

Two things are measured, per coded column at end-of-decode (G >= 1024):
  * ``err``   = ||u_ref @ decode(codes) - x_true|| / ||x_true||  (total recon error)
  * ``floor`` = ||(I - u_ref u_ref^T) x_true|| / ||x_true||       (subspace residual;
                the irreducible cost of a rank-r_a anchor -- a perfect quantizer
                still pays it), where x_true is the token's rank-r BUG
                reconstruction at graduation (what the fp32 tier would store).

Binned by decode position, two arms:
  * ON  = frozen anchor (the real CodeBUG). Prediction: err flat in position
          (non-compounding) -- a column's error is set at graduation, forever.
  * OFF = ``_debug_recode_coded`` (variant-D analog: re-anchor + recode every
          absorb). Prediction: err RISES with age (older = more recodings).

PASS (both required, per the panel): ON tail/head err ratio <= 1.5x AND OFF
ratio >= 3x (the test can see compounding). A rising ON curve kills the design.
The absolute ON ``err`` is the accuracy the perplexity sweep then converts to ppl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from perplexity_sweep import load_corpus_ids, load_model
from w5_streamppl import N_SINK, stream_score

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer
from kvdlra.quant import ProductQuantizer
from kvdlra.utils.seed import seed_everything

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"


def _build_cache(model: Any, pq: ProductQuantizer, args: argparse.Namespace) -> BugStreamingCache:
    return BugStreamingCache(
        model,
        rank=args.rank,
        coord_budget=args.coord_budget,
        recent_window=args.recent_window,
        absorb_block=args.absorb_block,
        n_sink=N_SINK,
        retention="attn",
        score_decay=args.score_decay,
        coord_codebook=pq,
        anchor_rank=args.anchor_rank,
        anchor_seal_absorbs=args.anchor_seal_absorbs,
        code_budget=args.code_budget,
    )


def _measure(cache: BugStreamingCache) -> list[tuple[int, float, float]]:
    """(position, err, floor) for every currently-held coded column, all layers."""
    out: list[tuple[int, float, float]] = []
    for layer in cache.layers:
        assert isinstance(layer, BugStreamingLayer)
        if layer._q_len() == 0 or layer.u_ref_k is None or layer._drift_truth is None:
            continue
        assert layer.qk_codes is not None and layer.q_pos is not None
        recon = layer.u_ref_k @ layer._decode_coded(layer.qk_codes, layer.qk_norm)  # (n, W)
        recon = recon.to("cpu", torch.float32)
        u_ref = layer.u_ref_k.to("cpu", torch.float32)  # (n, r_a)
        positions = layer.q_pos.tolist()
        for i, p in enumerate(positions):
            truth = layer._drift_truth.get(int(p))
            if truth is None:
                continue
            tn = float(truth.norm()) + 1e-12
            err = float((recon[:, i] - truth).norm()) / tn
            resid = truth - u_ref @ (u_ref.mT @ truth)
            floor = float(resid.norm()) / tn
            out.append((int(p), err, floor))
    return out


def _bin_curve(rows: list[tuple[int, float, float]], prefill: int, bin_size: int) -> dict[str, Any]:
    if not rows:
        return {"bins": [], "n": 0}
    max_pos = max(p for p, _, _ in rows)
    nbins = (max_pos - prefill) // bin_size + 1
    bins: list[dict[str, float]] = []
    for b in range(max(1, nbins)):
        lo, hi = prefill + b * bin_size, prefill + (b + 1) * bin_size
        sel = [(e, f) for p, e, f in rows if lo <= p < hi]
        if sel:
            errs = [e for e, _ in sel]
            floors = [f for _, f in sel]
            bins.append(
                {
                    "pos": lo,
                    "n": len(sel),
                    "err": sum(errs) / len(errs),
                    "floor": sum(floors) / len(floors),
                }
            )
    return {"bins": bins, "n": len(rows), "mean_err": sum(e for _, e, _ in rows) / len(rows)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    corpus = load_corpus_ids(tokenizer, args.device, corpus=args.corpus)
    blob = torch.load(args.codebook, weights_only=False)
    pq = ProductQuantizer.from_state(blob["state"])
    span = args.prefill + args.g_tokens + 1
    ids = corpus[args.doc_stride * args.doc : args.doc_stride * args.doc + span]
    if ids.shape[0] < span:
        raise RuntimeError("corpus exhausted for this doc")

    results: dict[str, Any] = {
        "model": args.model,
        "corpus": args.corpus,
        "codebook": str(args.codebook),
        "prefill": args.prefill,
        "g_tokens": args.g_tokens,
        "bin_size": args.bin_size,
        "config": {
            k: getattr(args, k) for k in ("rank", "anchor_rank", "coord_budget", "code_budget")
        },
        "arms": {},
    }
    for arm, recode in (("ON_frozen", False), ("OFF_recode", True)):
        cache = _build_cache(model, pq, args)
        for layer in cache.layers:
            assert isinstance(layer, BugStreamingLayer)
            layer._drift_truth = {}
            layer._debug_recode_coded = recode
        stream_score(model, ids, cache, args.prefill, args.device)
        curve = _bin_curve(_measure(cache), args.prefill, args.bin_size)
        bins = curve["bins"]
        ratio = (bins[-1]["err"] / bins[0]["err"]) if len(bins) >= 2 and bins[0]["err"] > 0 else 1.0
        curve["tail_head_ratio"] = ratio
        results["arms"][arm] = curve
        print(
            f"[{arm}] cols={curve['n']} mean_err={curve.get('mean_err', 0):.4f} "
            f"tail/head={ratio:.2f} floor0={bins[0]['floor']:.4f}"
            if bins
            else f"[{arm}] no cols",
            flush=True,
        )
    on = results["arms"]["ON_frozen"].get("tail_head_ratio", 9.9)
    off = results["arms"]["OFF_recode"].get("tail_head_ratio", 0.0)
    results["verdict"] = {
        "on_flat": on <= 1.5,
        "off_compounds": off >= 3.0,
        "pass": (on <= 1.5) and (off >= 3.0),
    }
    print(
        f"[verdict] ON tail/head={on:.2f} (<=1.5?) OFF={off:.2f} (>=3?) -> {results['verdict']}",
        flush=True,
    )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--corpus", default="wikitext-2")
    p.add_argument("--codebook", default="results/w8-codebook.pt")
    p.add_argument("--doc", type=int, default=0)
    p.add_argument("--doc-stride", type=int, default=100_000)
    p.add_argument("--prefill", type=int, default=512)
    p.add_argument("--g-tokens", type=int, default=1024)
    p.add_argument("--bin-size", type=int, default=128)
    p.add_argument("--rank", type=int, default=24)
    p.add_argument("--anchor-rank", type=int, default=8)
    p.add_argument("--coord-budget", type=int, default=64)
    p.add_argument("--code-budget", type=int, default=1200)
    p.add_argument("--recent-window", type=int, default=32)
    p.add_argument("--absorb-block", type=int, default=16)
    p.add_argument("--anchor-seal-absorbs", type=int, default=4)
    p.add_argument("--score-decay", type=float, default=0.97)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default="results/w8-carry-drift.json")
    args = p.parse_args()

    results = run(args)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[wrote {out}]")


if __name__ == "__main__":
    main()
