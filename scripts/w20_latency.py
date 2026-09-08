"""Week-20 systems fix: MEASURED decode latency and peak VRAM at the real operating point.

The exit-gate systems review (F2) rejected the paper's only latency datum -- 1B on CPU at
a 327-token context, +10% -- as unrepresentative, and its "decode residency ~1.06x" as an
analytic sum (stored + workspace) rather than a measured contrast, because the full-KV
arm's decode peak was never measured. This measures both, per arm x context, batch 1:

* decode ms/token -- one token per forward at TRUE positions (the apples-to-apples
  protocol every arm can run), N steps after a warm-up, CUDA-synchronized per step.
  p50 is the steady-state cost; max and the spike count (> 2x p50) surface BUG's
  absorb-event middle rebuild (the ``ready`` cost of the persistence benchmark,
  amortized over ``absorb_block`` tokens) and KIVI's per-step dequantize.
* peak VRAM during decode (``max_memory_allocated`` after a reset post-prefill) and the
  post-prefill resident allocation, with the model weights subtracted so the
  KV-attributable numbers are a direct contrast: full vs flagship vs KIVI-2bit.

Rows (harvested like ``[persist``):
    [latency ctx16384] full  ms/tok=.. mean=.. max=.. spikes=.. resident_gb=.. peak_gb=..

    python scripts/w20_latency.py --model M --device cuda --dtype bfloat16 --chunk 4096 \
        --context-lens 16384 32768 65536 --methods full bugslash quant --ranks 64 \
        --hh-budgets 256 --hh-neighbor 1 --warmup-seed --quant-nbits 2 --quant-scheme kivi
"""

from __future__ import annotations

import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from perplexity_sweep import load_model
from transformers.cache_utils import DynamicCache
from w10_frontier import _prefill_chunked, _prefill_plain, build_arms, build_parser

from kvdlra.press.compat import install_kvpress_prefill_compat

JSON_BEGIN = "===W20_LATENCY_JSON_BEGIN==="
JSON_END = "===W20_LATENCY_JSON_END==="
GB = 1024**3


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _mem(device: str) -> tuple[float, float]:
    """(allocated, peak) in GB; zeros off-CUDA."""
    if not device.startswith("cuda"):
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / GB, torch.cuda.max_memory_allocated() / GB


@torch.no_grad()
def run_latency(
    model: Any, args: Any, ctx: int, device: str, n_steps: int, warmup: int
) -> list[dict[str, Any]]:
    torch.manual_seed(0)
    hay = torch.randint(0, int(model.config.vocab_size), (1, ctx), device=device)
    weights_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / GB
    rows: list[dict[str, Any]] = []
    for arm in build_arms(args, model, ctx):
        kind = arm["kind"]
        if kind == "full":
            cache: Any = DynamicCache()
            model(hay, past_key_values=cache, use_cache=True, logits_to_keep=1)
        elif kind == "bug":
            cache = arm["make"]()
            with cache.attach(model):
                _prefill_chunked(model, cache, hay, args.chunk if args.chunk > 0 else ctx)
        elif kind == "quant":
            cache = arm["make"]()
            _prefill_plain(model, cache, hay, args.chunk)
        else:
            raise ValueError(f"w20_latency covers full/bug/quant arms, not {kind!r}")
        _sync(device)
        resident_gb, _ = _mem(device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        # Decode one token per forward at true positions; greedy next token.
        tok_id = torch.zeros((1, 1), dtype=torch.long, device=device)
        start = ctx
        times_ms: list[float] = []
        scope = cache.attach(model) if kind == "bug" else nullcontext()
        with scope:
            for _ in range(n_steps):
                pos = torch.arange(start, start + 1, device=device).unsqueeze(0)
                _sync(device)
                t0 = time.perf_counter()
                out = model(tok_id, past_key_values=cache, use_cache=True, position_ids=pos)
                _sync(device)
                times_ms.append((time.perf_counter() - t0) * 1000.0)
                tok_id = out.logits[0, -1].argmax().view(1, 1)
                start += 1
        _, peak_gb = _mem(device)
        steady = times_ms[warmup:] if len(times_ms) > warmup else times_ms
        p50 = statistics.median(steady)
        spikes = sum(1 for t in steady if t > 2.0 * p50)
        row: dict[str, Any] = {
            "method": arm["name"],
            "kind": kind,
            "ctx": ctx,
            "ms_per_tok_p50": p50,
            "ms_per_tok_mean": statistics.fmean(steady),
            "ms_per_tok_max": max(steady),
            "spikes_gt_2x": spikes,
            "n_steps": len(steady),
            "resident_gb": resident_gb,
            "peak_gb": peak_gb,
            "weights_gb": weights_gb,
            "kv_resident_gb": max(resident_gb - weights_gb, 0.0),
            "kv_peak_gb": max(peak_gb - weights_gb, 0.0),
            "per_step_ms": times_ms,
        }
        rows.append(row)
        print(
            f"[latency ctx{ctx}] {arm['name']:22s} ms/tok={p50:.2f} "
            f"mean={row['ms_per_tok_mean']:.2f} "
            f"max={row['ms_per_tok_max']:.2f} spikes={spikes} resident_gb={resident_gb:.2f} "
            f"peak_gb={peak_gb:.2f} weights_gb={weights_gb:.2f} kv_peak_gb={row['kv_peak_gb']:.2f}",
            flush=True,
        )
        del cache, out
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = build_parser()
    parser.add_argument("--context-lens", type=int, nargs="+", default=[2048])
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.set_defaults(out_json="results/w20-latency.json", methods=["full", "bugslash", "quant"])
    args = parser.parse_args()
    install_kvpress_prefill_compat()
    model, _tok = load_model(args.model, args.device, args.dtype)
    model.config._attn_implementation = "sdpa"
    rows = [
        r
        for ctx in args.context_lens
        for r in run_latency(model, args, ctx, args.device, args.n_steps, args.warmup)
    ]
    blob = {"model": args.model, "device": args.device, "dtype": args.dtype, "rows": rows}
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    print(JSON_BEGIN)
    print(json.dumps(blob))
    print(JSON_END)
    print(f"[wrote {out}]", flush=True)


if __name__ == "__main__":
    main()
