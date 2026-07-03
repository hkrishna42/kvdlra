"""Axis-B validation: BugStreamingCache on a real model -- coherence, memory, latency.

The three proofs the design note (``docs/notes/streaming-decode-design.md`` §6)
requires *before* any benchmark or pod time:

1. **Coherence / parity.** Greedy continuations under the streaming cache vs the
   full ``DynamicCache``. In *exact mode* (``rank = n_features``, budgets larger
   than anything generated -- nothing truncated or dropped) the continuation
   should match the baseline (the RoPE round-trip proof at the real-model level,
   the decode analog of ``results/w3-parity.md``); at aggressive rank it should
   drift gracefully, never to gibberish.
2. **Constant memory.** Stored cache floats vs generated length: flat for the
   streaming cache (bounded sawtooth), linear for the full cache. Measured, not
   asserted.
3. **Bounded per-token latency.** Wall-clock per decode step; the streaming
   cache's amortized absorb cost must not grow with generated length.

Runs on CPU with the 1B model in ~15 min; ``--device cuda --dtype bfloat16
--model unsloth/Meta-Llama-3.1-8B-Instruct`` for the pod. Results JSON is echoed
between ``===W5_DECODE_VALIDATE_JSON_BEGIN/END===`` markers (pod-log recipe,
``[[vastai-pod-flakiness-jul2026]]``). ``--plot-only`` rebuilds the figure.

Example (CPU, 1B):
    python scripts/w5_decode_validate.py --max-new-tokens 160
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import matplotlib
import torch
from generate_with_press import LONG_PROMPTS
from perplexity_sweep import load_model
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

from kvdlra.cache import BugStreamingCache, MorphKVCache
from kvdlra.utils.seed import seed_everything

matplotlib.use("Agg")  # headless / CPU-safe backend; set before pyplot import
import matplotlib.pyplot as plt

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
JSON_BEGIN = "===W5_DECODE_VALIDATE_JSON_BEGIN==="
JSON_END = "===W5_DECODE_VALIDATE_JSON_END==="


def build_caches(model: PreTrainedModel, n_features: int) -> dict[str, Cache | None]:
    """Named cache factories for one generation run (``None`` => DynamicCache).

    ``bug-exact`` is the parity configuration: rank == n_features and budgets no
    generation here can exhaust, so nothing is ever truncated or dropped and the
    only deviation from the full cache is the fp32 RoPE round trip on the middle
    block. The others are genuinely lossy operating points; ``sllm`` (rank 0) is
    the StreamingLLM degenerate mode of the same implementation.
    """
    return {
        "full": None,
        "bug-exact": BugStreamingCache(
            model, rank=n_features, coord_budget=65536, recent_window=64, absorb_block=32
        ),
        "bug-r128": BugStreamingCache(
            model, rank=128, coord_budget=512, recent_window=64, absorb_block=32
        ),
        "bug-r32": BugStreamingCache(
            model, rank=32, coord_budget=128, recent_window=32, absorb_block=16
        ),
        "sllm-w64": BugStreamingCache(
            model, rank=0, coord_budget=0, recent_window=64, absorb_block=32
        ),
        # Roughly bug-r128's stored budget at 1B (C+R whole tokens + score buffer).
        "morph-C256-R32": MorphKVCache(model, capacity=256, recent_window=32),
    }


def stored_floats(cache: Cache) -> int:
    """Stored cache float entries (BUG/MorphKV state floats, or full K/V numel)."""
    if isinstance(cache, BugStreamingCache | MorphKVCache):
        return int(cache.stored_state_numel())
    total = 0
    for layer in cache.layers:
        if isinstance(layer, DynamicLayer) and layer.keys is not None:
            assert layer.values is not None
            total += int(layer.keys.numel()) + int(layer.values.numel())
    return total


def greedy_decode(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    cache: Cache,
    max_new_tokens: int,
    device: str,
) -> dict[str, Any]:
    """Greedy decode with a manual step loop, recording per-step memory + latency.

    EOS is deliberately *not* honored: the memory/latency curves need a fixed
    number of steps per config to be comparable (noted in the writeup).
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    toks: list[int] = []
    mems: list[int] = []
    lats: list[float] = []
    attach = cache.attach(model) if isinstance(cache, MorphKVCache) else nullcontext()
    with torch.no_grad(), attach:
        out = model(ids, past_key_values=cache, use_cache=True)
        for _ in range(max_new_tokens):
            tok = out.logits[:, -1:].argmax(dim=-1)
            toks.append(int(tok.item()))
            t0 = time.perf_counter()
            out = model(tok, past_key_values=cache, use_cache=True)
            lats.append(time.perf_counter() - t0)
            mems.append(stored_floats(cache))
    text = tokenizer.decode(toks, skip_special_tokens=True)
    lat_sorted = sorted(lats[1:])  # drop the first step (warm-up noise)
    return {
        "prompt_tokens": int(ids.shape[1]),
        "tokens": toks,
        "text": text,
        "mem_floats": mems,
        "lat_ms_mean": 1e3 * sum(lat_sorted) / len(lat_sorted),
        "lat_ms_p50": 1e3 * lat_sorted[len(lat_sorted) // 2],
        "lat_ms_p90": 1e3 * lat_sorted[int(0.9 * len(lat_sorted))],
    }


def first_divergence(a: list[int], b: list[int]) -> int:
    """Index of the first differing token (== len if identical)."""
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x != y:
            return i
    return len(a)


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_features = int(head_dim * cfg.num_key_value_heads)

    results: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "n_features": n_features,
        "n_layers": int(cfg.num_hidden_layers),
        "max_new_tokens": args.max_new_tokens,
        "prompts": {},
    }
    for p_idx, prompt in enumerate(LONG_PROMPTS[: args.n_prompts]):
        per_prompt: dict[str, Any] = {}
        caches = build_caches(model, n_features)
        for name, cache in caches.items():
            cache_obj: Cache = cache if cache is not None else DynamicCache()
            print(f"[prompt {p_idx}] {name} ...", flush=True)
            per_prompt[name] = greedy_decode(
                model, tokenizer, prompt, cache_obj, args.max_new_tokens, args.device
            )
        base_toks = per_prompt["full"]["tokens"]
        for rec in per_prompt.values():
            rec["first_divergence"] = first_divergence(rec["tokens"], base_toks)
            rec["exact_match"] = rec["first_divergence"] == len(base_toks)
            del rec["tokens"]  # texts + divergence indices carry the story
        results["prompts"][str(p_idx)] = per_prompt
    return results


def make_figure(results: dict[str, Any], fig_path: Path) -> None:
    styles = {
        "full": ("tab:blue", "-"),
        "bug-exact": ("tab:purple", "-"),
        "bug-r128": ("tab:orange", "-"),
        "bug-r32": ("tab:red", "-"),
        "sllm-w64": ("tab:green", "--"),
        "morph-C256-R32": ("tab:brown", "-."),
    }
    prompt0 = results["prompts"]["0"]
    fig, (ax_mem, ax_lat) = plt.subplots(1, 2, figsize=(11, 4))
    for name, rec in prompt0.items():
        color, ls = styles.get(name, ("gray", "-"))
        mems_m = [m / 1e6 for m in rec["mem_floats"]]
        ax_mem.plot(range(1, len(mems_m) + 1), mems_m, color=color, ls=ls, label=name)
    ax_mem.set_xlabel("generated tokens")
    ax_mem.set_ylabel("stored cache floats (millions)")
    ax_mem.set_title(
        f"cache memory vs generated length\n({results['model'].split('/')[-1]}, "
        f"prompt of {prompt0['full']['prompt_tokens']} tokens)"
    )
    ax_mem.legend(fontsize=8)
    ax_mem.grid(alpha=0.3)

    names = list(prompt0)
    means = [prompt0[n]["lat_ms_mean"] for n in names]
    p90s = [prompt0[n]["lat_ms_p90"] for n in names]
    xs = range(len(names))
    ax_lat.bar(xs, means, color=[styles.get(n, ("gray",))[0] for n in names], alpha=0.8)
    ax_lat.plot(xs, p90s, "k_", markersize=18, label="p90")
    ax_lat.set_xticks(list(xs))
    ax_lat.set_xticklabels(names, rotation=20, fontsize=8)
    ax_lat.set_ylabel("per-token latency (ms)")
    ax_lat.set_title("decode step latency (mean bar, p90 tick)")
    ax_lat.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"[wrote {fig_path}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--n-prompts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default="results/w5-decode-validate-1b.json")
    parser.add_argument("--fig", default="figures/week5/decode_validate_1b.png")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    out_json = Path(args.out_json)
    if args.plot_only:
        results = json.loads(out_json.read_text())
    else:
        results = run(args)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"[wrote {out_json}]")
        print(JSON_BEGIN)
        print(json.dumps(results))
        print(JSON_END)
    make_figure(results, Path(args.fig))


if __name__ == "__main__":
    main()
