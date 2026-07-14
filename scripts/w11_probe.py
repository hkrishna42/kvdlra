"""Week-11 Phase-0 diagnosis probe: does surprise-selection land the RULER needle?

The falsifiable crux (``docs/week11-plan.md``): a sharp 5-digit needle is a
low-attention, high-residual outlier. SurpriseSLASH keeps the top-``hh_budget``
tokens by low-rank *surprise* verbatim. The go/no-go: at context ``ctx``, does the
needle's token(s) enter the exact ``hh`` tier at a SMALL ``hh_budget`` (=> a
frontier win vs ExpectedAttention's 0.10x), or only at a large one / never (=> the
fidelity wall holds even with a surprise-scored exact tier -- a bounded negative)?

This drives the *actual* mechanism (chunked ingest of a ``BugStreamingCache`` with
``hh_select="surprise"``) and reports, per ``hh_budget``, whether every needle
token is retained verbatim -- so the smallest capturing budget is (approximately)
the needle's surprise rank, which maps directly to a memory ratio.

Run cheap on 1B/CPU at a modest ctx for early signal; confirm at 8B/32K on a pod::

    uv run python scripts/w11_probe.py --model unsloth/Llama-3.2-1B-Instruct \
        --ctx 4096 --rank 32 --hh-budgets 32 64 128 256 512 1024 --chunk 1024
"""

from __future__ import annotations

import argparse
import json

import _paths  # noqa: F401  (prepends src/ to sys.path if needed)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from w10_ruler import build_task

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer

JSON_BEGIN = "===W11_PROBE_JSON_BEGIN==="
JSON_END = "===W11_PROBE_JSON_END==="


def _find_subseq(hay: list[int], needle: list[int]) -> list[int]:
    """Start indices of every contiguous occurrence of ``needle`` in ``hay``."""
    if not needle:
        return []
    hits = []
    for i in range(len(hay) - len(needle) + 1):
        if hay[i : i + len(needle)] == needle:
            hits.append(i)
    return hits


def _neighbor_covered(needle_pos: list[int], hh_set: set[int], w: int) -> bool:
    """Would keeping every exact token's +-w neighbours (SnapKV-style span
    expansion) place every needle token within reach of a retained one?"""
    return len(needle_pos) > 0 and all(any(abs(p - h) <= w for h in hh_set) for p in needle_pos)


def _needle_positions(tok: object, prefill: torch.Tensor, code: str) -> list[int]:
    """Token positions of the needle ``code`` in ``prefill`` (robust to digit
    tokenization: tries a few encodings, returns the tokens of the first match)."""
    ids = prefill[0].tolist()
    for variant in (f" {code}", code, f"{code}."):
        enc = tok(variant, add_special_tokens=False).input_ids  # type: ignore[operator]
        # trim a trailing "." token if we appended one
        starts = _find_subseq(ids, enc)
        if starts:
            s = starts[0]
            return list(range(s, s + len(enc)))
    return []


def probe(args: argparse.Namespace) -> dict[str, object]:
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        args.dtype
    ]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    model.config._attn_implementation = "sdpa"

    pre, _query, targets = build_task(
        tok,
        "niah_single",
        args.ctx,
        trial=args.trial,
        seed=args.seed,
        n_keys=1,
        n_values=1,
        n_hops=1,
    )
    pre = pre.to(args.device)
    code = targets[0]
    needle_pos = _needle_positions(tok, pre, code)
    t = int(pre.shape[1])
    rows: list[dict[str, object]] = []
    for hh in args.hh_budgets:
        cache = BugStreamingCache(
            model,
            rank=args.rank,
            coord_budget=t + args.recent_window + args.absorb_block,
            recent_window=args.recent_window,
            absorb_block=args.absorb_block,
            n_sink=4,
            retention="lowrank_surprise",
            hh_budget=hh,
            hh_select="surprise",
        )
        with torch.no_grad(), cache.ingesting():
            for s in range(0, t, args.chunk):
                e = min(t, s + args.chunk)
                pos = torch.arange(s, e, device=args.device).unsqueeze(0)
                model(pre[:, s:e], past_key_values=cache, use_cache=True, position_ids=pos)
                cache.consolidate()
        layer = next(la for la in cache.layers if isinstance(la, BugStreamingLayer))
        hh_set = set(layer.hh_pos.tolist()) if layer.hh_pos is not None else set()
        in_hh = [p for p in needle_pos if p in hh_set]
        n_in = len(in_hh)
        captured = len(needle_pos) > 0 and n_in == len(needle_pos)

        # Would neighbor/span expansion (keep each exact token's +-w neighbours,
        # like SnapKV's window) close the gap? The needle tokens are contiguous, so
        # if ANY is a strong outlier a small window should grab the rest verbatim.
        nbr = {w: _neighbor_covered(needle_pos, hh_set, w) for w in (1, 2, 4)}
        rows.append(
            {
                "hh_budget": hh,
                "captured": captured,
                "needle_in_hh": n_in,
                "in_hh_positions": in_hh,
                "neighbor_covered": nbr,
            }
        )
        print(
            f"[ctx{t} r{args.rank}] hh={hh:>5}  needle_in_hh={n_in}/{len(needle_pos)}  "
            f"captured={captured}  nbr(w1/2/4)={int(nbr[1])}/{int(nbr[2])}/{int(nbr[4])}  "
            f"ratio~={hh / t:.4f}"
        )
    smallest = next((r["hh_budget"] for r in rows if r["captured"]), None)
    result = {
        "model": args.model,
        "ctx": t,
        "rank": args.rank,
        "code": code,
        "needle_tokens": len(needle_pos),
        "needle_positions": needle_pos,
        "smallest_capturing_hh": smallest,
        "sweep": rows,
    }
    print(f"{JSON_BEGIN}{json.dumps(result)}{JSON_END}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--hh-budgets", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--recent-window", type=int, default=32)
    p.add_argument("--absorb-block", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--out-json", default="")
    args = p.parse_args()
    result = probe(args)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
