"""MEASURED decode latency and peak VRAM at the real operating point.

The exit-gate systems review rejected the paper's only latency datum -- 1B on CPU at a
327-token context, +10% -- as unrepresentative, and its "decode residency ~1.06x" as an
analytic sum (stored + workspace) rather than a measured contrast, because the full-KV
arm's decode peak was never measured. This measures both, per arm x context x batch:

* decode ms/token -- one token per forward at TRUE positions (the apples-to-apples
  protocol every arm can run), N steps after a warm-up, CUDA-synchronized per step.
  p50 is the steady-state cost; max and the spike count (> 2x p50) surface the
  absorb-event middle rebuild (the ``ready`` cost of the persistence benchmark,
  amortized over ``absorb_block`` tokens) and KIVI's per-step dequantize.
* peak VRAM during decode (``max_memory_allocated`` after a reset post-prefill) and the
  post-prefill resident allocation, with the model weights subtracted so the
  KV-attributable numbers are a direct contrast.

Rows (harvested by ``records.parse_latency_lines``)::

    [latency ctx16384] full  ms/tok=.. mean=.. max=.. spikes=.. resident_gb=.. peak_gb=..
        weights_gb=.. kv_peak_gb=.. batch=.. kv_resident_gb=..
"""

from __future__ import annotations

import statistics
import time
from contextlib import nullcontext
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from kvdlra.eval.frontier import _prefill_chunked, _prefill_plain

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
    model: Any,
    arms: list[dict[str, Any]],
    ctx: int,
    device: str,
    chunk: int,
    n_steps: int,
    warmup: int,
    batch: int = 1,
) -> list[dict[str, Any]]:
    torch.manual_seed(0)
    hay = torch.randint(0, int(model.config.vocab_size), (batch, ctx), device=device)
    weights_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / GB
    rows: list[dict[str, Any]] = []
    for arm in arms:
        kind = arm["kind"]
        if kind == "full":
            cache: Any = DynamicCache()
            model(hay, past_key_values=cache, use_cache=True, logits_to_keep=1)
        elif kind == "bug":
            cache = arm["make"]()
            with cache.attach(model):
                _prefill_chunked(model, cache, hay, chunk if chunk > 0 else ctx)
        elif kind == "quant":
            cache = arm["make"]()
            _prefill_plain(model, cache, hay, chunk)
        else:
            raise ValueError(f"latency covers full/bug/quant arms, not {kind!r}")
        _sync(device)
        resident_gb, _ = _mem(device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        # Decode one token per forward at true positions; greedy next token.
        tok_id = torch.zeros((batch, 1), dtype=torch.long, device=device)
        start = ctx
        times_ms: list[float] = []
        scope = cache.attach(model) if kind == "bug" else nullcontext()
        with scope:
            for _ in range(n_steps):
                pos = torch.arange(start, start + 1, device=device).unsqueeze(0).expand(batch, -1)
                _sync(device)
                t0 = time.perf_counter()
                out = model(tok_id, past_key_values=cache, use_cache=True, position_ids=pos)
                _sync(device)
                times_ms.append((time.perf_counter() - t0) * 1000.0)
                tok_id = out.logits[:, -1].argmax(-1).view(batch, 1)
                start += 1
        _, peak_gb = _mem(device)
        steady = times_ms[warmup:] if len(times_ms) > warmup else times_ms
        p50 = statistics.median(steady)
        spikes = sum(1 for t in steady if t > 2.0 * p50)
        row: dict[str, Any] = {
            "method": arm["name"],
            "kind": kind,
            "ctx": ctx,
            "batch": batch,
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
            f"peak_gb={peak_gb:.2f} weights_gb={weights_gb:.2f} kv_peak_gb={row['kv_peak_gb']:.2f}"
            # Appended last so the archived-line format keeps matching unchanged;
            # `kv_resident_gb` is the resident reading §3 of the kernel prereg asked the
            # record to carry.
            f" batch={batch} kv_resident_gb={row['kv_resident_gb']:.2f}",
            flush=True,
        )
        del cache, out
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return rows
