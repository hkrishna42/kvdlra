"""Week-19 A3: the realized systems win -- persisted-cache cold start, measured.

The byte ratio of the stored state is measured (``w16_storage.py``: cold-load 0.150x /
0.139x at 16K/32K). What turns it into a deployment existence proof is the *wall-clock*
of bringing a persisted cache back to attend-ready: serialize -> reload from disk ->
host-to-device -> reconstruct (BUG: ``_ensure_mid_cache``, the reconstruct-then-attend
middle; the quant baseline: one full dequantize; full KV: nothing). Arms: full KV, the
flagship ``bugSseed-r64-h256``, and the fair 2/4-bit KIVI baseline, same prefill.

What is persisted is exactly the state the accounting bills: for BUG the
``stored_state_numel`` tensor set (square-root cores as their diagonals); for the quant
baseline the packed codes + scales/zeros (+ the fp16 residual); for full KV its fp16
K/V. Timings are medians of ``--repeats`` runs after the file was just written (warm
page cache: the OS read is the floor, the H2D + reconstruct terms are the real cost).

Rows (``^\\[persist``) are the harvest record::

    [persist ctx16384] bugSseed-r64-h256  bytes=... ratio=0.150 save=..s load=..s \
        h2d=..s ready=..s cold=..s
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
from perplexity_sweep import load_model
from transformers.cache_utils import DynamicCache
from w10_frontier import _prefill_chunked, _prefill_plain, build_arms, build_parser

from kvdlra.press.compat import install_kvpress_prefill_compat
from kvdlra.quant.kivi_cache import _PerChannel

JSON_BEGIN = "===W19_PERSIST_JSON_BEGIN==="
JSON_END = "===W19_PERSIST_JSON_END==="

# The BUG layer's stored state (mirrors BugStreamingLayer.stored_state_numel): tiers,
# bases, coordinates, coded tier + norms, retention bookkeeping, whitening diagonal.
_BUG_ATTRS = (
    "sink_k", "sink_v", "recent_k", "recent_v", "u_k", "c_k", "u_v", "c_v", "hh_k", "hh_v",
    "u_ref_k", "u_ref_v", "qk_codes", "qv_codes", "qk_norm", "qv_norm", "mid_pos", "q_pos",
    "mid_score", "q_score", "ring_score", "hh_pos", "hh_score", "mid_weight", "w_key",
)  # fmt: skip


def state_tensors(kind: str, cache: Any) -> dict[str, torch.Tensor]:
    """The persisted state of an arm's post-prefill cache, as plain tensors."""
    out: dict[str, torch.Tensor] = {}
    if kind == "full":
        for i, layer in enumerate(cache.layers):
            out[f"{i}.keys"], out[f"{i}.values"] = layer.keys, layer.values
        return out
    if kind == "bug":
        for i, layer in enumerate(cache._bug_layers()):
            for name in _BUG_ATTRS:
                t = getattr(layer, name, None)
                if isinstance(t, torch.Tensor):
                    out[f"{i}.{name}"] = t
            for name in ("b_k", "b_v"):  # provably diagonal cores: r entries, not r^2
                core = getattr(layer, name, None)
                if isinstance(core, torch.Tensor):
                    out[f"{i}.{name}"] = torch.diagonal(core)
            if layer.track_surprise:
                for name in ("mid_surprise", "q_surprise"):
                    t = getattr(layer, name, None)
                    if isinstance(t, torch.Tensor):
                        out[f"{i}.{name}"] = t
        return out
    if kind == "quant":
        for i, layer in enumerate(cache.layers):
            for which in ("_quantized_keys", "_quantized_values"):
                q = getattr(layer, which)
                q = q.q if isinstance(q, _PerChannel) else q
                if isinstance(q, tuple):  # hqq: (W_q, meta)
                    out[f"{i}.{which}.wq"] = q[0]
                    out[f"{i}.{which}.scale"] = q[1]["scale"]
                    out[f"{i}.{which}.zero"] = q[1]["zero"]
                else:  # quanto WeightQBitsTensor: packed uint8 codes + scale + shift
                    raw = q._data
                    out[f"{i}.{which}.data"] = getattr(raw, "_data", raw)
                    out[f"{i}.{which}.scale"] = q._scale
                    out[f"{i}.{which}.shift"] = q._shift
            if layer.keys.dim() == 4:  # fp16 residual (empty after flush)
                out[f"{i}.keys"], out[f"{i}.values"] = layer.keys, layer.values
        return out
    raise ValueError(f"no persisted state defined for arm kind {kind!r}")


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def persist_roundtrip(
    tensors: dict[str, torch.Tensor], path: Path, device: str, repeats: int
) -> dict[str, float]:
    """torch.save -> bytes on disk; then median over ``repeats`` of reload (CPU) and H2D."""
    _sync(device)
    t0 = time.perf_counter()
    torch.save({k: v.detach() for k, v in tensors.items()}, path)
    t_save = time.perf_counter() - t0
    loads, h2ds = [], []
    for _ in range(repeats):
        t1 = time.perf_counter()
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        loads.append(time.perf_counter() - t1)
        t2 = time.perf_counter()
        moved = [v.to(device) for v in loaded.values()]
        _sync(device)
        h2ds.append(time.perf_counter() - t2)
        del moved, loaded
    return {
        "bytes": float(path.stat().st_size),
        "t_save": t_save,
        "t_load": statistics.median(loads),
        "t_h2d": statistics.median(h2ds) if device.startswith("cuda") else 0.0,
    }


@torch.no_grad()
def attend_ready_seconds(kind: str, cache: Any, device: str, repeats: int) -> float:
    """Median wall-clock to turn the resident persisted state into what attention reads:
    BUG rebuilds the middle (reconstruct-then-attend), quant dequantizes every layer,
    full KV needs nothing."""
    if kind == "full":
        return 0.0
    times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        if kind == "bug":
            for layer in cache._bug_layers():
                layer._mid_k_cache = None
                layer._mid_v_cache = None
                layer._ensure_mid_cache()
        else:
            for layer in cache.layers:
                layer._dequantize(layer._quantized_keys)
                layer._dequantize(layer._quantized_values)
        _sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


@torch.no_grad()
def run_persist(
    model: Any, args: Any, ctx: int, device: str, tmp: Path, repeats: int = 3
) -> list[dict[str, Any]]:
    torch.manual_seed(0)
    hay = torch.randint(0, int(model.config.vocab_size), (1, ctx), device=device)
    rows: list[dict[str, Any]] = []
    full_bytes: float | None = None
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
            raise ValueError(f"w19_persist covers full/bug/quant arms, not {kind!r}")
        state = state_tensors(kind, cache)
        m = persist_roundtrip(state, tmp / f"{arm['name']}.pt", device, repeats)
        m["t_ready"] = attend_ready_seconds(kind, cache, device, repeats)
        m["t_cold"] = m["t_load"] + m["t_h2d"] + m["t_ready"]
        if kind == "full":
            full_bytes = m["bytes"]
        if full_bytes is None:
            raise ValueError("run the 'full' arm first (it is the byte reference)")
        row: dict[str, Any] = {"method": arm["name"], "kind": kind, "ctx": ctx, **m}
        row["ratio_bytes"] = m["bytes"] / full_bytes
        rows.append(row)
        print(
            f"[persist ctx{ctx}] {arm['name']:22s} bytes={int(m['bytes'])} "
            f"ratio={row['ratio_bytes']:.4f} save={m['t_save']:.3f}s load={m['t_load']:.3f}s "
            f"h2d={m['t_h2d']:.3f}s ready={m['t_ready']:.3f}s cold={m['t_cold']:.3f}s",
            flush=True,
        )
        del cache, state
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = build_parser()
    parser.add_argument("--context-lens", type=int, nargs="+", default=[2048])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tmp", default="/tmp/w19_persist")
    parser.set_defaults(out_json="results/w19-persist.json", methods=["full", "bugslash", "quant"])
    args = parser.parse_args()
    install_kvpress_prefill_compat()
    model, _tok = load_model(args.model, args.device, args.dtype)
    model.config._attn_implementation = "sdpa"
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    rows = [
        r
        for ctx in args.context_lens
        for r in run_persist(model, args, ctx, args.device, tmp, args.repeats)
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
