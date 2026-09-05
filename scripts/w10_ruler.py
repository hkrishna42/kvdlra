"""Week-10 PRIMARY long-context axis: RULER accuracy vs honestly-counted memory.

Perplexity (``w10_frontier.py``) is a weak long-context signal and structurally
favours BUG's global summary; the standard, fairer test is task **retrieval
accuracy** on RULER (and LongBench, ``w10_longbench.py``). This runs a *focused
RULER subset* -- the tasks most sensitive to KV compression -- as an
accuracy-vs-memory frontier, comparing the SAME arms as the ppl frontier (BUG
``BugStreamingCache`` ranks {32,64,128,256}, MorphKV, SnapKV / ExpectedAttention
presses) at honestly-matched memory (``kvdlra.accounting``).

Focused subset:

* ``niah_single``    -- one needle among filler (retrieve its code);
* ``niah_multikey``  -- ``n_keys`` distinct keys, retrieve ONE queried key's code
  (``w5_ruler.build_multikey_haystack``; eviction cannot know which key is asked);
* ``niah_multivalue``-- one key with ``n_values`` codes, retrieve ALL of them;
* ``vt``             -- variable tracking: a chain ``V0=<num>; V1=V0; ...`` then
  ask the value of the last variable (retrieve the root number through the chain).

Prefill uses OOM-safe chunked ingest for the streaming caches (BUG/MorphKV,
Phase 3) and single-shot full-``T`` prefill for the kvpress scorer presses
(``logits_to_keep=1`` + sdpa keeps it memory-safe, and SnapKV's ``q_len >
window_size`` assert forbids a chunked prefill); the short query + answer are
decoded one token per forward at TRUE positions (the eviction fairness fix,
``w9_recovery`` / ``w4_needle``). Memory is measured on the post-prefill
compressed cache. ``--plot-only`` rebuilds figures.

Example (CPU smoke, 1B)
-----------------------
    uv run python scripts/w10_ruler.py --device cpu --context-lens 2048 \
        --tasks niah_single niah_multikey --ranks 64 128 --n-trials 3 \
        --methods full bug morph snapkv
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import random
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import matplotlib
import torch
from perplexity_sweep import load_corpus_sentences, load_model
from transformers.cache_utils import Cache, DynamicCache
from w4_needle import _FILLER
from w5_ruler import _LABELS
from w10_frontier import _footprint, _prefill_chunked, _prefill_plain, build_arms

from kvdlra.press.compat import install_kvpress_prefill_compat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
JSON_BEGIN = "===W10_RULER_JSON_BEGIN==="
JSON_END = "===W10_RULER_JSON_END==="
_TAIL_K = 48  # FLOOR for the decoded query tail (question + assistant header, as in
# w4/w5); the actual tail is template-derived per family (see _templated) and never
# shorter than this, so Llama's validated 48-token slice is preserved bit-for-bit.
TASKS = ("niah_single", "niah_multikey", "niah_multivalue", "vt")


# ------------------------------------------------------------- task builders


def _filler_to(
    tok: Any,
    ctx: int,
    *,
    filler: str = "cycle",
    pool: list[str] | None = None,
    seed: int = 0,
    trial: int = 0,
) -> list[str]:
    """Filler sentences up to ``ctx`` tokens. ``filler="cycle"`` (default) is the
    bit-identical archived path: the fixed 10-sentence ``_FILLER`` cycled. Any other
    value draws from a realistic-corpus ``pool`` (loaded once by the caller and passed
    in, so tests need no network), seed-shuffled per (seed, trial) so every trial sees
    a different natural-text haystack -- the Week-18 external-validity fix.

    Week-19: memoized per (tokenizer, ctx, filler, pool, seed, trial). The builder grows
    the haystack one sentence at a time and re-tokenizes the whole text each step
    (O(n^2) tokenizer calls: minutes per call at 64K), and the SAME haystack is rebuilt
    for every arm and trial of a cell, so the cached tuple is returned as a fresh list."""
    key_pool = None if pool is None else tuple(pool)
    return list(_filler_cached(tok, ctx, filler, key_pool, seed, trial))


@functools.lru_cache(maxsize=32)
def _filler_cached(
    tok: Any, ctx: int, filler: str, pool: tuple[str, ...] | None, seed: int, trial: int
) -> tuple[str, ...]:
    base: Sequence[str] = _FILLER if filler == "cycle" else (pool or ())
    if not base:
        raise ValueError(f"filler={filler!r} requires a non-empty pool")
    order = list(range(len(base)))
    if filler != "cycle":
        random.Random(seed * 131 + trial).shuffle(order)
    sentences: list[str] = []
    i = 0
    while len(tok(" ".join(sentences)).input_ids) < ctx:
        sentences.append(base[order[i % len(order)]])
        i += 1
    return tuple(sentences)


def _first_divergence(a: torch.Tensor, b: torch.Tensor) -> int:
    """First index at which the 1-D id tensors ``a`` and ``b`` differ, or the shorter
    length if one is a prefix of the other."""
    m = min(int(a.shape[0]), int(b.shape[0]))
    if m == 0:
        return 0
    diff = torch.nonzero(a[:m] != b[:m], as_tuple=False)
    return int(diff[0].item()) if diff.numel() else m


def _tail_len(full_ids: torch.Tensor, body_ids: torch.Tensor, floor: int) -> int:
    """Number of trailing tokens of ``full_ids`` that are the decoded-at-true-position
    query = ``max(floor, question + generation-header)``. ``body_ids`` is the SAME chat
    template applied to the body alone (``add_generation_prompt=True``); the two share the
    body prefix and first diverge where the question begins, so ``len(full) - divergence``
    is exactly the question+header length -- template-derived and family-agnostic. The
    ``floor`` (Llama's fixed 48) keeps validated Llama slices unchanged and is a safe
    lower bound (a longer query only widens the uncompressed recent window; the mid-body
    needle is never inside the last 48 tokens). Clamped to leave >=1 token to compress."""
    div = _first_divergence(full_ids, body_ids)
    tail = max(floor, int(full_ids.shape[0]) - div)
    return min(tail, int(full_ids.shape[0]) - 1)


def _templated(tok: Any, body: str, question: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the templated (body+question) into (compressed-prefill, decoded-query).

    The tail is template-derived (``_tail_len``) so the whole question + generation
    header lands in the decoded query for ANY chat template, not just Llama's -- the
    fixed ``_TAIL_K=48`` mis-sliced Mistral/Qwen templates into silent needle failures.
    Fails loud if the question is not fully inside the decoded query (the per-family
    mis-slice tripwire)."""

    def _ids(text: str) -> torch.Tensor:
        out = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )["input_ids"]
        return cast(torch.Tensor, out)

    full = _ids(body + question)
    tail_k = _tail_len(full[0], _ids(body)[0], _TAIL_K)
    pre, query = full[:, :-tail_k], full[:, -tail_k:]
    q_norm = " ".join(question.split())
    if q_norm and q_norm not in " ".join(tok.decode(query[0]).split()):
        raise ValueError(
            f"RULER template mis-slice for model {getattr(tok, 'name_or_path', '?')!r}: "
            f"the question is not fully inside the decoded query (tail_k={tail_k}). This "
            f"chat template's question+header tail differs from the Llama assumption -- "
            f"inspect apply_chat_template before running (a wrong slice = silent 0s)."
        )
    return pre, query


def _codes(g: torch.Generator, k: int) -> list[int]:
    return (10000 + torch.randperm(89999, generator=g)[:k]).tolist()


def build_task(
    tok: Any,
    task: str,
    ctx: int,
    trial: int,
    seed: int,
    n_keys: int,
    n_values: int,
    n_hops: int,
    *,
    filler: str = "cycle",
    pool: list[str] | None = None,
    depths: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return (prefill_ids, query_ids, targets) for one RULER trial. ``targets`` is a
    list of strings that must ALL appear in the decoded answer for a hit. ``filler``
    selects the haystack corpus (``cycle`` = bit-identical archived path); ``depths``,
    when given, sweeps the niah_single needle depth across trials (Week-18 depth grid)."""
    g = torch.Generator().manual_seed(seed * 131 + trial)
    sents = _filler_to(tok, ctx, filler=filler, pool=pool, seed=seed, trial=trial)
    n = len(sents)

    if task == "niah_single":
        code = _codes(g, 1)[0]
        # Default placement is mid-depth with a small trial jitter (archived path). A
        # --depths grid instead lands the needle at depths[trial % len] * n.
        if depths:
            depth = depths[trial % len(depths)]
            insert_at = min(n, max(0, int(depth * n)))
        else:
            insert_at = n // 2 + (trial % 5)
        sents.insert(insert_at, f"The secret passcode is {code}.")
        body = " ".join(sents)
        q = "\n\nWhat is the secret passcode? Reply with only the number."
        pre, query = _templated(tok, body, q)
        return pre, query, [str(code)]

    if task == "niah_multikey":
        labels = _LABELS[:n_keys]
        codes = _codes(g, n_keys)
        for k, (lab, code) in enumerate(zip(labels, codes, strict=True)):
            pos = min(n, max(0, int((k + 1) / (n_keys + 1) * n) + (k % 3)))
            sents.insert(pos, f"The {lab} code is {code}.")
        qi = trial % n_keys
        body = " ".join(sents)
        q = f"\n\nWhat is the {labels[qi]} code? Reply with only the number."
        pre, query = _templated(tok, body, q)
        return pre, query, [str(codes[qi])]

    if task == "niah_multivalue":
        label = _LABELS[trial % len(_LABELS)]
        codes = _codes(g, n_values)
        for i, code in enumerate(codes):
            pos = min(n, max(0, int((i + 1) / (n_values + 1) * n) + (i % 3)))
            sents.insert(pos, f"The {label} code is {code}.")
        body = " ".join(sents)
        q = f"\n\nList every {label} code mentioned. Reply with only the numbers."
        pre, query = _templated(tok, body, q)
        return pre, query, [str(c) for c in codes]

    if task == "vt":
        root = _codes(g, 1)[0]
        var = [f"X{trial}{i}" for i in range(n_hops + 1)]
        chain = [f"VAR {var[0]} = {root}."] + [
            f"VAR {var[i]} = {var[i - 1]}." for i in range(1, n_hops + 1)
        ]
        for i, stmt in enumerate(chain):  # in causal order at increasing depth
            pos = min(len(sents), int((i + 1) / (len(chain) + 1) * len(sents)))
            sents.insert(pos, stmt)
        body = " ".join(sents)
        q = f"\n\nWhat is the value of {var[-1]}? Reply with only the number."
        pre, query = _templated(tok, body, q)
        return pre, query, [str(root)]

    raise ValueError(f"unknown task {task!r}")


# ------------------------------------------------------------- retrieval


@torch.no_grad()
def _decode(
    model: Any,
    tok: Any,
    cache: Cache,
    query: torch.Tensor,
    start: int,
    device: str,
    block: bool,
    max_new: int,
) -> str:
    """Greedy-decode the answer at TRUE positions. ``block`` feeds the query in one
    forward (DynamicCache presses); else one token per forward (streaming caches,
    which raise on a q_len>1 continuation)."""
    out = None
    if block:
        pos = torch.arange(start, start + query.shape[1], device=device).unsqueeze(0)
        out = model(query, past_key_values=cache, use_cache=True, position_ids=pos)
        start += int(query.shape[1])
    else:
        for j in range(int(query.shape[1])):
            pos = torch.arange(start, start + 1, device=device).unsqueeze(0)
            out = model(
                query[:, j : j + 1], past_key_values=cache, use_cache=True, position_ids=pos
            )
            start += 1
    assert out is not None
    generated: list[int] = []
    nxt = int(out.logits[0, -1].argmax())
    for _ in range(max_new):
        generated.append(nxt)
        pos = torch.arange(start, start + 1, device=device).unsqueeze(0)
        out = model(
            torch.tensor([[nxt]], device=device),
            past_key_values=cache,
            use_cache=True,
            position_ids=pos,
        )
        start += 1
        nxt = int(out.logits[0, -1].argmax())
    return str(tok.decode(generated))


@torch.no_grad()
def retrieve(
    model: Any,
    tok: Any,
    arm: dict[str, Any],
    hay: torch.Tensor,
    query: torch.Tensor,
    targets: list[str],
    device: str,
    chunk: int,
    n: int,
    h_kv: int,
    max_new: int,
) -> tuple[bool, float, float, float]:
    """Prefill the haystack (chunked, OOM-safe) then decode the query+answer. Returns
    (hit, fp16-memory-ratio, hits_fraction, stored-bits-ratio). A hit requires ALL
    ``targets`` in the output. Memory is the post-prefill compressed footprint
    (kvdlra.accounting); the stored-bits ratio bills fp32-at-rest state honestly."""
    hay = hay.to(device)
    ctx_len = int(hay.shape[1])
    streaming = arm["kind"] in ("bug", "morph", "shadow")
    if streaming:
        cache: Cache = arm["make"]()
        # Week-15 A1 fix: the attach() scope covers prefill AND decode (uniform for
        # all streaming arms). Previously only the prefill was attached, so
        # ShadowKV's pre-attention selection hook never ran at decode and
        # _selected_chunks fell back to the most-recent chunks -- excluding the
        # mid-context needle by construction (the published 0/0/0/0 rows are VOID).
        with cache.attach(model):  # type: ignore[attr-defined]
            if 0 < chunk < ctx_len:
                _prefill_chunked(model, cache, hay, chunk)
            else:
                model(hay, past_key_values=cache, use_cache=True, logits_to_keep=1)
            fp = _footprint(arm, cache, ctx_len, n, h_kv)
            text = _decode(
                model, tok, cache, query.to(device), ctx_len, device, block=False, max_new=max_new
            )
    elif arm["kind"] == "quant":
        # KIVI-style QuantizedCache baseline (Week-18/19): the arm supplies its OWN cache
        # object (not a press over a DynamicCache); prefill honors --chunk (Week-19: the
        # single-shot 16K/32K quant prefill OOM'd even on 80GB) and flushes the residual
        # so decode starts fully quantized, as after a single-shot prefill.
        cache = arm["make"]()
        _prefill_plain(model, cache, hay, chunk)
        fp = _footprint(arm, cache, ctx_len, n, h_kv)
        text = _decode(
            model, tok, cache, query.to(device), ctx_len, device, block=True, max_new=max_new
        )
    else:
        # Scorer presses run SINGLE-SHOT (no ChunkPress): kvpress compresses in
        # prefill and SnapKVPress asserts q_len > window_size (64), which the small
        # final ChunkPress chunk violates ("Query length ... should be greater than
        # the window size"). RULER prefill uses logits_to_keep=1 + sdpa, so full-T
        # prefill is memory-safe (proven by ExpectedAttention surviving at 32K).
        cache = DynamicCache()
        press = arm["make"]()
        with press(model) if press is not None else nullcontext():
            model(hay, past_key_values=cache, use_cache=True, logits_to_keep=1)
        fp = _footprint(arm, cache, ctx_len, n, h_kv)
        text = _decode(
            model, tok, cache, query.to(device), ctx_len, device, block=True, max_new=max_new
        )
    frac = sum(t in text for t in targets) / len(targets)
    hit = frac >= 1.0
    del cache
    gc.collect()
    return hit, fp.ratio_fp16(ctx_len, n), frac, fp.ratio_stored_bits(ctx_len, n)


# ------------------------------------------------------------- runner


def run(args: argparse.Namespace) -> dict[str, Any]:
    install_kvpress_prefill_compat()  # transformers 5.8: presses need the cache_position shim
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    model.config._attn_implementation = "sdpa"
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    max_new = 12 if args.tasks == ["niah_single"] else 40  # multi-value needs room
    # Realistic-filler pool loaded ONCE (not per trial); None for the cycle default.
    filler = getattr(args, "filler", "cycle")
    depths = getattr(args, "depths", None)
    pool = None if filler == "cycle" else load_corpus_sentences(filler)
    print(
        f"model={args.model} n={n} tasks={args.tasks} ctxs={args.context_lens} "
        f"filler={filler} depths={depths}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for ctx in args.context_lens:
        for task in args.tasks:
            for arm in build_arms(args, model, ctx):
                arm_chunk = args.chunk if arm.get("chunkable", True) else 0
                hits, ratios, fracs, sbits = 0, [], [], []
                for seed in args.seeds:
                    for trial in range(args.n_trials):
                        try:
                            hay, query, targets = build_task(
                                tokenizer,
                                task,
                                ctx,
                                trial,
                                seed,
                                args.n_keys,
                                args.n_values,
                                args.n_hops,
                                filler=filler,
                                pool=pool,
                                depths=depths,
                            )
                            hit, ratio, frac, sratio = retrieve(
                                model,
                                tokenizer,
                                arm,
                                hay,
                                query,
                                targets,
                                args.device,
                                arm_chunk,
                                n,
                                h_kv,
                                max_new,
                            )
                        except Exception as exc:
                            # One bad trial is skipped (logged); the arm survives, so a
                            # single OOM/edge trial no longer kills every seed of the arm.
                            print(
                                f"[{task} ctx{ctx}] {arm['name']:14s} "
                                f"SKIP {type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            if args.device.startswith("cuda"):
                                torch.cuda.empty_cache()
                            continue
                        hits += int(hit)
                        ratios.append(ratio)
                        fracs.append(frac)
                        sbits.append(sratio)
                        # Per-trial evidentiary line (Week-18): one row per trial so
                        # the exact per-cell n and hit/miss chain survives the pod log
                        # (recovered by the w18 intervals builder without a hand table).
                        print(
                            f"[trial] task={task} ctx={ctx} arm={arm['name']} "
                            f"seed={seed} trial={trial} hit={int(hit)} frac={frac:.3f}",
                            flush=True,
                        )
                if not ratios:  # every trial failed -> arm produced no measurement
                    continue
                total = len(ratios)
                row = {
                    "task": task,
                    "ctx": ctx,
                    "method": arm["name"],
                    "kind": arm["kind"],
                    "rank": arm["rank"],
                    "accuracy": hits / total,
                    "recall_frac": sum(fracs) / len(fracs),
                    "ratio_fp16": sum(ratios) / len(ratios),
                    "ratio_stored_bits": sum(sbits) / len(sbits),
                    "hits": hits,
                    "total": total,
                }
                results.append(row)
                # `n=` and `sbits=` are appended last so the archived-line regexes
                # (w17_intervals.ROW, w11_merge.ROW_RE) keep matching unchanged.
                print(
                    f"[{task} ctx{ctx}] {arm['name']:14s} acc={row['accuracy']:.2f} "
                    f"recall={row['recall_frac']:.2f} ratio={row['ratio_fp16']:.3f} "
                    f"sbits={row['ratio_stored_bits']:.3f} n={total}",
                    flush=True,
                )
    return {
        "model": args.model,
        "n_features": n,
        "context_lens": args.context_lens,
        "tasks": args.tasks,
        "n_trials": args.n_trials,
        "seeds": args.seeds,
        "results": results,
    }


# ------------------------------------------------------------- plot


def _plot(blob: dict[str, Any], out: Path) -> None:
    results = blob["results"]
    tasks = blob["tasks"]
    ctxs = sorted({int(r["ctx"]) for r in results})
    if not ctxs or not tasks:
        # Every arm SKIP'd (e.g. a quant run on a -runtime pod without the CUDA kernel):
        # nothing to plot. Skip cleanly instead of crashing plt.subplots on a 0-row grid.
        print(f"[plot skipped: no result rows for {out}]", flush=True)
        return
    fig, axes = plt.subplots(
        len(ctxs), len(tasks), figsize=(4.6 * len(tasks), 4.0 * len(ctxs)), squeeze=False
    )
    kind_style = {
        "bug": ("tab:orange", "o"),
        "morph": ("tab:green", "s"),
        "press": ("tab:red", "^"),
        "quant": ("tab:purple", "D"),
        "shadow": ("tab:brown", "v"),
        "full": ("tab:blue", "*"),
    }
    for i, ctx in enumerate(ctxs):
        for j, task in enumerate(tasks):
            ax = axes[i][j]
            rows = [r for r in results if r["task"] == task and int(r["ctx"]) == ctx]
            for kind, (c, m) in kind_style.items():
                pts = sorted((r["ratio_fp16"], r["accuracy"]) for r in rows if r["kind"] == kind)
                if pts:
                    ax.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        marker=m,
                        color=c,
                        lw=1.8,
                        ms=7,
                        label=kind,
                        alpha=0.9,
                    )
            ax.set_ylim(-0.05, 1.05)
            ax.set_xscale("log")
            ax.set_title(f"{task}  (T={ctx})", fontsize=10)
            ax.set_xlabel("KV memory / full fp16")
            ax.set_ylabel("accuracy")
            ax.grid(True, which="both", alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Week-10 RULER: retrieval accuracy vs memory -- {blob['model'].split('/')[-1]}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=150)
    print(f"[wrote {out.with_suffix('.png')}]", flush=True)


# ------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[2048])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--morph-keeps", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--evict-keeps", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--think-ratios", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--palu-ranks", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--palu-group", type=int, default=1)
    # Week-18 quant baseline + compose (mirror w10_frontier.build_parser so the quant/
    # q4 arms build here too -- build_arms reads --quant-nbits directly for the quant arm).
    parser.add_argument("--quant-nbits", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--quant-group", type=int, default=64)
    parser.add_argument("--quant-residual", type=int, default=128)
    parser.add_argument("--quant-scheme", default="token", choices=["token", "kivi"])
    parser.add_argument("--quant-backend", default="quanto", choices=["quanto", "hqq"])
    parser.add_argument("--bug-quant-bits", type=int, default=None)
    parser.add_argument("--bug-quant-budget", type=int, default=0)
    parser.add_argument("--shadow-ranks", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--shadow-topk", type=int, default=256)
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--absorb-block", type=int, default=16)
    parser.add_argument(
        "--hh-budgets",
        type=int,
        nargs="+",
        default=[256, 1024, 2048],
        help="Week-11 SurpriseSLASH exact-tier sizes (bugslash/bugevict)",
    )
    parser.add_argument(
        "--hh-neighbor", type=int, default=0, help="SurpriseSLASH span-expansion window (0=off)"
    )
    parser.add_argument(
        "--hh-discard",
        action="store_true",
        help="Week-12 H1 ablation: bugslash arms select-and-DISCARD (hh_retain=False; "
        "pool invisible to attention) -> bugSdrop-* arm names",
    )
    parser.add_argument(
        "--qwhiten-file",
        default=None,
        help="Week-12 Q-BUG: calibrated per-layer key-whitening diagonal "
        "(scripts/w12_calibrate_qkey.py) -> bugSQ-* arms (query-metric gist)",
    )
    parser.add_argument(
        "--warmup-seed",
        action="store_true",
        help="Week-13 T-B: seed the exact tier from the first ingest chunk's outliers "
        "-> bugSseed-* arms (fixes the warm-up window; requires --chunk>0)",
    )
    parser.add_argument(
        "--score-rank",
        type=int,
        default=None,
        help="Week-15 T2: cap the SLASH surprise-scoring basis at this many leading "
        "columns (selection-rank decoupled from storage-rank) -> '-s{k}' arm suffix; "
        "storage/footprint unchanged; requires 1 <= k <= rank",
    )
    parser.add_argument(
        "--min-sv-frac",
        type=float,
        default=0.0,
        help="Week-17: relative singular-value floor for the streaming integrator "
        "(0.0=off; e.g. 1e-2) -> '-f{v}' arm suffix; storage/footprint unchanged",
    )
    parser.add_argument("--chunk", type=int, default=0, help="chunked-prefill block size")
    parser.add_argument(
        "--filler",
        default="cycle",
        choices=["cycle", "wikitext", "wikitext-103", "pg19"],
        help="RULER haystack filler: 'cycle' (bit-identical 10-sentence loop) or a "
        "realistic corpus, seed-shuffled per trial (Week-18 external-validity fix)",
    )
    parser.add_argument(
        "--depths",
        type=float,
        nargs="+",
        default=None,
        help="niah_single needle-depth grid in [0,1] (swept across trials); "
        "default None keeps the archived mid-depth+jitter placement",
    )
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--n-keys", type=int, default=8)
    parser.add_argument("--n-values", type=int, default=4)
    parser.add_argument("--n-hops", type=int, default=3)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "full",
            "bug",
            "morph",
            "snapkv",
            "ea",
            "think",
            "palu",
            "shadow",
        ],
    )
    parser.add_argument("--out-json", default="results/w10-ruler-1b.json")
    # untracked scratch default so a local smoke run never clobbers a committed
    # figure; pass --out-fig figures/weekN/... explicitly to update a real one.
    parser.add_argument("--out-fig", default="figures/scratch/ruler_accuracy")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    out_json = Path(args.out_json)
    if args.plot_only:
        _plot(json.loads(out_json.read_text()), Path(args.out_fig))
        return

    blob = run(args)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(blob, indent=2) + "\n")
    _plot(blob, Path(args.out_fig))
    print(JSON_BEGIN)
    print(json.dumps(blob))
    print(JSON_END)
    print(f"[wrote {out_json}]", flush=True)


if __name__ == "__main__":
    main()
