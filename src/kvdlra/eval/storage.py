"""MEASURED storage footprint vs full KV (the systems tier).

The caches are *reconstruct-then-attend* (attention sees a full-length reconstruction of
the retained history), so a naive decode-time ``max_memory_allocated`` shows a HIGHER
peak than full KV -- the opposite of the paper's story, and a Mode-B low-rank attention
kernel (future work) is what a real throughput / peak-VRAM win needs. So this tier is
framed around what the method actually optimizes: the *stored* cache state.

After a real chunked prefill at several context lengths this measures the live cache's
``stored_state_numel()`` (summed ``.numel()`` over its real tensors -- the deployable
constant-memory state) against full KV (``2*t*n`` per layer), and reports:

* **measured storage ratio** (float-equivalent) vs context -- the empirical curve, not a
  bare formula;
* an **accounting-integrity** cross-check: measured floats vs the analytic
  ``Footprint.float_equiv()`` (pinned equal by ``tests/test_accounting.py`` on tiny
  configs -- here confirmed at real scale, bar = within +-5%);
* the **reconstruction workspace** (``workspace_numel()``) -- the transient the kernel
  would remove -- reported beside storage so the reconstruct-then-attend cost is explicit;
* on CUDA, ``measure_peak_gpu`` per arm, the measured resident contrast.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from kvdlra.accounting import measure_peak_gpu
from kvdlra.eval.frontier import _footprint, _prefill_chunked


@torch.no_grad()
def _measure(
    model: Any,
    arm: dict[str, Any],
    hay: torch.Tensor,
    ctx: int,
    n: int,
    h_kv: int,
    chunk: int,
    n_layers: int,
) -> dict[str, Any]:
    full_floats = float(2 * ctx * n * n_layers)
    if arm["kind"] == "full":
        return {
            "measured_floats": full_floats,
            "measured_ratio": 1.0,
            "ratio_fp16": 1.0,
            "ratio_stored_bits": 1.0,  # cold-load size ratio (full KV persists all of it)
            "workspace_floats": 0.0,
            "workspace_ratio": 0.0,
            "peak_bytes": None,
            "peak_ratio": None,
            "integrity_rel_err": 0.0,
        }
    cache = arm["make"]()
    with cache.attach(model), measure_peak_gpu(str(hay.device)) as peak_get:
        if 0 < chunk < ctx:
            _prefill_chunked(model, cache, hay, chunk)
        else:
            model(hay, past_key_values=cache, use_cache=True, logits_to_keep=1)
        measured = float(cache.stored_state_numel())  # the bounded deployable state
        fp = _footprint(arm, cache, ctx, n, h_kv)  # analytic, matches the post-prefill state
        # One decode step materializes the reconstruct-then-attend working set (u_k @ c_k
        # rebuilds the full-length middle). Measured AFTER stored/fp so the cache advance
        # does not perturb them; workspace ~ full KV is exactly why naive peak-VRAM shows
        # no win -- the real cost the future Mode-B kernel would remove. The peak-GPU
        # probe now spans EVERY arm (Week-18 M2), not just eviction -> the ~1.06x resident
        # is a disclosed measured number on CUDA (None on CPU, where it is unmeasured).
        pos = torch.arange(ctx, ctx + 1, device=hay.device).unsqueeze(0)
        model(hay[:, -1:], past_key_values=cache, use_cache=True, position_ids=pos)
        workspace = float(cache.workspace_numel()) if hasattr(cache, "workspace_numel") else 0.0
        peak = peak_get()
    del cache
    analytic = fp.float_equiv() * n_layers
    # Full-KV resident (fp16) is 2*ctx*n*n_layers*2 bytes; peak_ratio compares the arm's
    # measured peak against it (CUDA only). The cold-load SIZE win is ratio_stored_bits
    # itself (at-rest bits); the H2D/reconstruct TIMING is CUDA-only -> G5.
    full_kv_bytes = float(2 * ctx * n * n_layers * 2)
    return {
        "measured_floats": measured,
        "measured_ratio": measured / full_floats,
        "ratio_fp16": fp.ratio_fp16(ctx, n),
        "ratio_stored_bits": fp.ratio_stored_bits(ctx, n),
        "workspace_floats": workspace,
        "workspace_ratio": workspace / full_floats,
        "peak_bytes": peak,
        "peak_ratio": (peak / full_kv_bytes) if peak is not None else None,
        "integrity_rel_err": abs(measured - analytic) / analytic if analytic else 0.0,
    }


def run(
    model: Any,
    build: Callable[[int], list[dict[str, Any]]],
    context_lens: list[int],
    device: str,
    chunk: int,
) -> dict[str, Any]:
    """Measure every arm ``build(ctx)`` returns at each context length.

    ``build`` is a callable rather than a list because a cache arm's budgets resolve
    against the context length, so the arms have to be rebuilt per ``ctx``.
    """
    model.config._attn_implementation = "sdpa"
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    n_layers = int(cfg.num_hidden_layers)
    g = torch.Generator().manual_seed(0)
    rows: list[dict[str, Any]] = []
    worst_integrity = 0.0
    for ctx in context_lens:
        hay = torch.randint(0, int(cfg.vocab_size), (1, ctx), generator=g).to(device)
        for arm in build(ctx):
            m = _measure(model, arm, hay, ctx, n, h_kv, chunk, n_layers)
            worst_integrity = max(worst_integrity, float(m["integrity_rel_err"]))
            row = {"ctx": ctx, "method": arm["name"], "kind": arm["kind"], **m}
            rows.append(row)
            peak_s = f"{m['peak_ratio']:.3f}" if m["peak_ratio"] is not None else "cpu"
            print(
                f"[ctx{ctx:>6}] {arm['name']:22s} stored_ratio={m['measured_ratio']:.4f} "
                f"ratio_fp16={m['ratio_fp16']:.4f} coldload_ratio={m['ratio_stored_bits']:.4f} "
                f"workspace_ratio={m['workspace_ratio']:.3f} peak_ratio={peak_s} "
                f"integrity={m['integrity_rel_err']:.1e}",
                flush=True,
            )
    ok = worst_integrity <= 0.05
    print(
        f"[integrity] bar: measured within +-5% of analytic float_equiv | "
        f"verdict: {'PASS' if ok else 'FAIL'} (worst={worst_integrity:.2e})"
    )
    blob = {
        "model": str(getattr(cfg, "name_or_path", "?")),
        "n_features": n,
        "n_layers": n_layers,
        "context_lens": context_lens,
        "worst_integrity_rel_err": worst_integrity,
        "integrity_ok": ok,
        "results": rows,
    }
    return blob
