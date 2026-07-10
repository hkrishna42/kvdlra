"""Week-8 CodeBUG: calibrate the shared product-quantization codebook.

Fits the ``ProductQuantizer`` on the TRUE deployment distribution -- the anchor-
frame coordinate vectors ``c_ref = u_ref^T (u_live @ c)`` that ``CodeBugCache``
codes at graduation -- captured by streaming the *actual* cache over a held-out
corpus (``BugStreamingLayer._calib_sink``). This is the honest, accepted
calibration cost (same class as PolarQuant's fixed rotation or KIVI's per-channel
stats): the codebook is model-specific, data-generic side information, calibrated
once and shipped, counted once per cache in the matched-memory audit.

**Leakage guardrail (Phase-3 skeptic iii).** Calibration uses WikiText-103
*train* (``--calib-corpus wikitext-103``); evaluation uses WikiText-2 *test*.
These are disjoint by the standard WikiText article split (train articles vs
held-out test articles). ``pg19`` is available as an even-stricter cross-corpus
check. The manifest records the split so disjointness is auditable.

The capture is codebook-independent: ``c_ref`` depends only on the BUG basis and
fp32 coordinates (both exact), never on the codes, so a throwaway bootstrap
codebook is used just to run the cache; the real codebook is then fit on the
collected ``c_ref`` and saved.

Usage::

    uv run python scripts/calibrate_codebook.py --anchor-rank 8 --pq-subspaces 4 \
        --pq-bits 8 --out results/w8-codebook-r8-m4b8.pt
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


def _bootstrap_codebook(
    dim: int, subspaces: int, bits: int, normalize: bool, seed: int
) -> ProductQuantizer:
    """A throwaway fitted PQ (on random data) just to construct the cache; the
    captured ``c_ref`` are independent of it."""
    g = torch.Generator().manual_seed(seed)
    return ProductQuantizer(
        dim=dim, bits=bits, subspaces=subspaces, normalize=normalize, seed=seed
    ).fit(torch.randn(2048, dim, generator=g), iters=3)


def run(args: argparse.Namespace) -> tuple[ProductQuantizer, dict[str, Any]]:
    seed_everything(args.seed)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    corpus = load_corpus_ids(tokenizer, args.device, corpus=args.calib_corpus)

    boot = _bootstrap_codebook(
        args.anchor_rank, args.pq_subspaces, args.pq_bits, not args.no_normalize, args.seed
    )
    cache = BugStreamingCache(
        model,
        rank=args.rank,
        coord_budget=args.coord_budget,
        recent_window=args.recent_window,
        absorb_block=args.absorb_block,
        n_sink=N_SINK,
        retention=args.retention,
        score_decay=args.score_decay,
        coord_codebook=boot,
        anchor_rank=args.anchor_rank,
        anchor_seal_absorbs=args.anchor_seal_absorbs,
        code_budget=args.code_budget,
    )
    sink: list[torch.Tensor] = []
    for layer in cache.layers:
        assert isinstance(layer, BugStreamingLayer)
        layer._calib_sink = sink

    span = args.prefill + args.g_tokens + 1
    n_coords = 0
    for d in range(args.n_calib_docs):
        start = args.calib_doc_stride * d
        ids = corpus[start : start + span]
        if ids.shape[0] < span:
            print(f"[calib] corpus exhausted at doc {d}", flush=True)
            break
        # Fresh cache state per doc; keep the SAME sink (pool across docs/layers).
        for layer in cache.layers:
            assert isinstance(layer, BugStreamingLayer)
            layer.reset()
            layer._calib_sink = sink
        stream_score(model, ids, cache, args.prefill, args.device)  # populates sink
        n_coords = sum(t.shape[0] for t in sink)
        print(f"[calib] doc {d}: pooled c_ref rows so far = {n_coords}", flush=True)
        if n_coords >= args.max_coords:
            break

    if not sink:
        raise RuntimeError(
            "no coordinates captured -- the coded tier never populated; "
            "lower coord_budget / raise g_tokens / lower anchor_seal_absorbs"
        )
    coords = torch.cat(sink, dim=0)  # (N, r_a), pooled over K+V, all layers, all docs
    if coords.shape[0] > args.max_coords:
        coords = coords[: args.max_coords]
    print(f"[calib] fitting PQ on {tuple(coords.shape)} c_ref vectors ...", flush=True)

    pq = ProductQuantizer(
        dim=args.anchor_rank,
        bits=args.pq_bits,
        subspaces=args.pq_subspaces,
        normalize=not args.no_normalize,
        seed=args.seed,
    ).fit(coords, iters=args.kmeans_iters)
    rel = pq.distortion(coords) / float((coords**2).sum(-1).mean())
    manifest = {
        "model": args.model,
        "calibration_corpus": args.calib_corpus,
        "calibration_split": "train" if args.calib_corpus == "wikitext-103" else args.calib_corpus,
        "eval_corpus": "wikitext-2 (test) -- DISJOINT by the WikiText article split",
        "n_calib_coords": int(coords.shape[0]),
        "anchor_rank": args.anchor_rank,
        "pq_bits": args.pq_bits,
        "pq_subspaces": args.pq_subspaces,
        "normalize": not args.no_normalize,
        "codebook_numel": pq.codebook_numel(),
        "calib_rel_distortion": rel,
        "config": {
            "rank": args.rank,
            "coord_budget": args.coord_budget,
            "code_budget": args.code_budget,
            "recent_window": args.recent_window,
            "absorb_block": args.absorb_block,
            "anchor_seal_absorbs": args.anchor_seal_absorbs,
            "retention": args.retention,
        },
    }
    print(
        f"[calib] PQ fit: rel_distortion={rel:.4f}, codebook_numel={pq.codebook_numel()}",
        flush=True,
    )
    return pq, manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument(
        "--calib-corpus", default="wikitext-103", choices=["wikitext-103", "pg19", "wikitext-2"]
    )
    p.add_argument("--n-calib-docs", type=int, default=3)
    p.add_argument("--calib-doc-stride", type=int, default=200_000)
    p.add_argument("--prefill", type=int, default=512)
    p.add_argument("--g-tokens", type=int, default=1024)
    p.add_argument("--max-coords", type=int, default=120_000)
    # cache config (MATCH the eval sweep's aggressive config)
    p.add_argument("--rank", type=int, default=24)
    p.add_argument("--anchor-rank", type=int, default=8)
    p.add_argument("--coord-budget", type=int, default=64)
    p.add_argument("--code-budget", type=int, default=1200)
    p.add_argument("--recent-window", type=int, default=32)
    p.add_argument("--absorb-block", type=int, default=16)
    p.add_argument("--anchor-seal-absorbs", type=int, default=4)
    p.add_argument("--retention", default="attn", choices=["attn", "fifo", "energy"])
    p.add_argument("--score-decay", type=float, default=0.97)
    # PQ config
    p.add_argument("--pq-bits", type=int, default=8)
    p.add_argument("--pq-subspaces", type=int, default=4)
    p.add_argument("--no-normalize", action="store_true", help="raw PQ (no per-col norm)")
    p.add_argument("--kmeans-iters", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/w8-codebook.pt")
    args = p.parse_args()

    pq, manifest = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": pq.state(), "manifest": manifest}, out)
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[wrote {out}] and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
