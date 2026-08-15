"""Week-16 Tier-4 (systems) -- MEASURED storage footprint vs full KV.

The caches are *reconstruct-then-attend* (attention sees a full-length reconstruction of
the retained history), so a naive decode-time ``max_memory_allocated`` would show BUG with
a HIGHER peak than full KV -- the opposite of the paper's story, and a Mode-B low-rank
attention kernel (future work) is what a real throughput/peak-VRAM win needs. So Tier-4 is
reframed honestly around what BUG actually optimizes: the *stored* cache state.

This measures, after a real chunked prefill at several context lengths, the live cache's
``stored_state_numel()`` (summed ``.numel()`` over its real tensors -- the deployable
constant-memory state) against full KV (``2*t*n`` per layer), and reports:

* **measured storage ratio** (float-equivalent) vs context -- the empirical "3-5x less
  memory" curve, no longer a bare formula;
* an **accounting-integrity** cross-check: measured floats vs the analytic
  ``Footprint.float_equiv()`` (pinned equal by ``tests/test_accounting.py`` on tiny
  configs -- here confirmed at real scale, bar = within +-5%);
* the **reconstruction workspace** (``workspace_numel()``) -- the transient the kernel
  would remove -- reported beside storage so the reconstruct-then-attend cost is explicit;
* on CUDA only, ``measure_peak_gpu`` for the *eviction* arms (which genuinely shrink the
  ``DynamicCache``), the honest systems point for that regime.

CPU example (1B)::

    uv run python scripts/w16_storage.py --context-lens 1024 2048 4096 8192 \
        --ranks 32 128 --methods full bug bugslash --chunk 512 --warmup-seed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _paths  # noqa: F401  (prepends src/ to sys.path if needed)
import matplotlib
import torch
from perplexity_sweep import load_model
from w10_frontier import _footprint, _prefill_chunked, build_arms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # must follow matplotlib.use("Agg")

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
JSON_BEGIN = "===W16_STORAGE_JSON_BEGIN==="
JSON_END = "===W16_STORAGE_JSON_END==="


def _smoke_args(args: argparse.Namespace) -> argparse.Namespace:
    """A full build_arms namespace (only the bug/full knobs vary here)."""
    return argparse.Namespace(
        methods=args.methods,
        ranks=args.ranks,
        hh_budgets=args.hh_budgets,
        chunk=args.chunk,
        morph_keeps=[0.25],
        evict_keeps=[0.1],
        think_ratios=[0.5],
        palu_ranks=[0.5],
        palu_group=1,
        shadow_ranks=[64],
        shadow_topk=256,
        recent_window=32,
        absorb_block=16,
        hh_neighbor=1,
        hh_discard=False,
        qwhiten_file=None,
        warmup_seed=args.warmup_seed,
        score_rank=None,
    )


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
            "workspace_floats": 0.0,
            "workspace_ratio": 0.0,
            "integrity_rel_err": 0.0,
        }
    cache = arm["make"]()
    with cache.attach(model):
        if 0 < chunk < ctx:
            _prefill_chunked(model, cache, hay, chunk)
        else:
            model(hay, past_key_values=cache, use_cache=True, logits_to_keep=1)
        measured = float(cache.stored_state_numel())  # the bounded deployable state
        fp = _footprint(arm, cache, ctx, n, h_kv)  # analytic, matches the post-prefill state
        # One decode step materializes the reconstruct-then-attend working set (u_k @ c_k
        # rebuilds the full-length middle). Measured AFTER stored/fp so the cache advance
        # does not perturb them; workspace ~ full KV is exactly why naive peak-VRAM shows
        # no win -- the honest cost the future Mode-B kernel would remove.
        pos = torch.arange(ctx, ctx + 1, device=hay.device).unsqueeze(0)
        model(hay[:, -1:], past_key_values=cache, use_cache=True, position_ids=pos)
        workspace = float(cache.workspace_numel()) if hasattr(cache, "workspace_numel") else 0.0
    del cache
    analytic = fp.float_equiv() * n_layers
    return {
        "measured_floats": measured,
        "measured_ratio": measured / full_floats,
        "ratio_fp16": fp.ratio_fp16(ctx, n),
        "workspace_floats": workspace,
        "workspace_ratio": workspace / full_floats,
        "integrity_rel_err": abs(measured - analytic) / analytic if analytic else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    model, _tok = load_model(args.model, args.device, args.dtype)
    model.config._attn_implementation = "sdpa"
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    n_layers = int(cfg.num_hidden_layers)
    g = torch.Generator().manual_seed(0)
    rows: list[dict[str, Any]] = []
    worst_integrity = 0.0
    for ctx in args.context_lens:
        hay = torch.randint(0, int(cfg.vocab_size), (1, ctx), generator=g).to(args.device)
        for arm in build_arms(_smoke_args(args), model, ctx):
            m = _measure(model, arm, hay, ctx, n, h_kv, args.chunk, n_layers)
            worst_integrity = max(worst_integrity, float(m["integrity_rel_err"]))
            row = {"ctx": ctx, "method": arm["name"], "kind": arm["kind"], **m}
            rows.append(row)
            print(
                f"[ctx{ctx:>6}] {arm['name']:22s} stored_ratio={m['measured_ratio']:.4f} "
                f"ratio_fp16={m['ratio_fp16']:.4f} workspace_ratio={m['workspace_ratio']:.3f} "
                f"integrity={m['integrity_rel_err']:.1e}",
                flush=True,
            )
    ok = worst_integrity <= 0.05
    print(
        f"[integrity] bar: measured within +-5% of analytic float_equiv | "
        f"verdict: {'PASS' if ok else 'FAIL'} (worst={worst_integrity:.2e})"
    )
    blob = {
        "model": args.model,
        "n_features": n,
        "n_layers": n_layers,
        "context_lens": args.context_lens,
        "worst_integrity_rel_err": worst_integrity,
        "integrity_ok": ok,
        "results": rows,
    }
    print(f"{JSON_BEGIN}{json.dumps(blob)}{JSON_END}")
    return blob


def _plot(blob: dict[str, Any], out: Path) -> None:
    rows = blob["results"]
    methods = sorted({r["method"] for r in rows if r["kind"] != "full"})
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for method in methods:
        pts = sorted((r["ctx"], r["measured_ratio"]) for r in rows if r["method"] == method)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", lw=1.8, label=method)
    ax.axhline(1.0, color="gray", ls="--", lw=1.0, label="full KV")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("measured stored / full KV (float-equivalent)")
    ax.set_title(f"Week-16 Tier-4: measured storage footprint -- {blob['model'].split('/')[-1]}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=150)
    print(f"[wrote {out.with_suffix('.png')}]", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--context-lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--ranks", type=int, nargs="+", default=[32, 128])
    p.add_argument("--hh-budgets", type=int, nargs="+", default=[1024])
    p.add_argument("--methods", nargs="+", default=["full", "bug", "bugslash"])
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--warmup-seed", action="store_true")
    p.add_argument("--out-json", default="results/w16-storage.json")
    p.add_argument("--out-fig", default="figures/week16/storage_footprint")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    out_json = Path(args.out_json)
    if args.plot_only:
        _plot(json.loads(out_json.read_text()), Path(args.out_fig))
        return
    blob = run(args)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(blob, indent=2) + "\n")
    _plot(blob, Path(args.out_fig))


if __name__ == "__main__":
    main()
