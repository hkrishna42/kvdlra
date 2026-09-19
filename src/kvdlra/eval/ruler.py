"""The in-house RULER-style retrieval axis: needle generators + one-trial retrieval.

Four tasks, the ones most sensitive to KV compression:

* ``niah_single``    -- one needle among filler (retrieve its code);
* ``niah_multikey``  -- ``n_keys`` distinct keys, retrieve ONE queried key's code
  (eviction cannot know which key is asked);
* ``niah_multivalue``-- one key with ``n_values`` codes, retrieve ALL of them;
* ``vt``             -- variable tracking: a chain ``V0=<num>; V1=V0; ...`` then
  ask the value of the last variable (retrieve the root number through the chain).

Prefill uses OOM-safe chunked ingest for the streaming caches and single-shot full-``T``
prefill for the kvpress scorer presses (``logits_to_keep=1`` + sdpa keeps it memory-safe,
and SnapKV's ``q_len > window_size`` assert forbids a chunked prefill); the short query +
answer are decoded one token per forward at TRUE positions (the eviction fairness fix).
Memory is measured on the post-prefill compressed cache.

The generator (``build_task`` and the filler builder) is the v1 one, body for body: it
produced every archived cell, so a change here moves the numbers. Generator v2 is L2's.
"""

from __future__ import annotations

import functools
import gc
import hashlib
import random
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, cast

import torch
from transformers.cache_utils import Cache, DynamicCache

from kvdlra.eval.config import TaskCfg
from kvdlra.eval.data import FILLER, LABELS
from kvdlra.eval.frontier import _footprint, _prefill_chunked, _prefill_faithful, _prefill_plain
from kvdlra.eval.records import drained

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
    bit-identical archived path: the fixed 10-sentence ``FILLER`` cycled. Any other
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
    base: Sequence[str] = FILLER if filler == "cycle" else (pool or ())
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
        labels = LABELS[:n_keys]
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
        label = LABELS[trial % len(LABELS)]
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
    *,
    task: str | None = None,
    idx: int | None = None,
) -> tuple[bool, float, float, float]:
    """Prefill the haystack (chunked, OOM-safe) then decode the query+answer. Returns
    (hit, fp16-memory-ratio, hits_fraction, stored-bits-ratio). A hit requires ALL
    ``targets`` in the output. Memory is the post-prefill compressed footprint
    (kvdlra.accounting); the stored-bits ratio bills fp32-at-rest state as stored.
    ``task`` / ``idx`` only label the diagnostics this trial drains."""
    hay = hay.to(device)
    ctx_len = int(hay.shape[1])
    streaming = arm["kind"] in ("bug", "shadow")
    if streaming:
        cache: Cache = arm["make"]()
        # Week-15 A1 fix: the attach() scope covers prefill AND decode (uniform for
        # all streaming arms). Previously only the prefill was attached, so
        # ShadowKV's pre-attention selection hook never ran at decode and
        # _selected_chunks fell back to the most-recent chunks -- excluding the
        # mid-context needle by construction (the published 0/0/0/0 rows are VOID).
        with (
            # The tripwire's rows, before the cache goes -- `records.drained` says why in
            # a `finally`. `cache` is bound inside this branch, so the wrapper is here.
            drained(
                cache,
                model,
                source="ruler",
                arm=str(arm["name"]),
                ctx=ctx_len,
                task=task,
                idx=idx,
            ),
            cache.attach(model),  # type: ignore[attr-defined]
        ):
            if 0 < chunk < ctx_len:
                _prefill_chunked(model, cache, hay, chunk)
            else:
                model(hay, past_key_values=cache, use_cache=True, logits_to_keep=1)
            fp = _footprint(arm, cache, ctx_len, n, h_kv)
            text = _decode(
                model,
                tok,
                cache,
                query.to(device),
                ctx_len,
                device,
                block=False,
                max_new=max_new,
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
    elif arm["kind"] == "quant_faithful":
        # KIVI's own protocol (L2.2): full-precision single-shot prefill, the quantized
        # store built post hoc, then decode as the streaming quant arm above.
        cache = arm["make"]()
        _prefill_faithful(model, cache, hay)
        fp = _footprint(arm, cache, ctx_len, n, h_kv)
        text = _decode(
            model, tok, cache, query.to(device), ctx_len, device, block=True, max_new=max_new
        )
    elif arm["kind"] == "press_quant":
        # Week-20 composite (eviction x quantization): the eviction press prunes to the
        # keep-fraction during single-shot prefill, and the compat forward_hook's
        # QuantizedCache branch re-quantizes the pruned survivors -- so the cache holds
        # k*T tokens at nbits. Single-shot prefill (no ChunkPress), like the presses.
        cache = arm["make_cache"]()
        press = arm["make_press"]()
        with press(model):
            model(hay, past_key_values=cache, use_cache=True, logits_to_keep=1)
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
        # A per-layer-budget press (PyramidKV) decodes one token per forward like the
        # streaming caches: transformers' single causal mask is sized to layer 0's keys
        # and a q_len>1 block trips it on every other layer (frontier `_press`).
        cache = DynamicCache()
        press = arm["make"]()
        with press(model) if press is not None else nullcontext():
            model(hay, past_key_values=cache, use_cache=True, logits_to_keep=1)
        fp = _footprint(arm, cache, ctx_len, n, h_kv)
        text = _decode(
            model,
            tok,
            cache,
            query.to(device),
            ctx_len,
            device,
            block=not arm.get("per_layer_budget", False),
            max_new=max_new,
        )
    frac = sum(t in text for t in targets) / len(targets)
    hit = frac >= 1.0
    del cache
    gc.collect()
    return hit, fp.ratio_fp16(ctx_len, n), frac, fp.ratio_stored_bits(ctx_len, n)


# ------------------------------------------------------------- one trial

# The generator's shape parameters. Every v1 pod ran the defaults, and a task config
# has no field for them, so they are constants rather than a knob nobody turned.
N_KEYS, N_VALUES, N_HOPS = 8, 4, 3


def code_family(sub: str, trial: int) -> str | None:
    """The needle label ``build_task`` uses for this trial, or None for the tasks that
    have no label (``niah_single``, ``vt``). It mirrors the two index expressions in
    ``build_task``; ``tests/test_w10_ruler_filler.py`` pins it against the real prompt,
    so the two cannot drift apart silently."""
    if sub == "niah_multikey":
        return LABELS[:N_KEYS][trial % N_KEYS]
    if sub == "niah_multivalue":
        return LABELS[trial % len(LABELS)]
    return None


def run_trial(
    arm: dict[str, Any],
    model: Any,
    tok: Any,
    task: TaskCfg,
    sub: str,
    seed: int,
    trial: int,
    *,
    device: str,
    chunk: int,
    n: int,
    h_kv: int,
    pool: list[str] | None = None,
) -> tuple[int, float, dict[str, Any]]:
    """Build one prompt and retrieve through ``arm``. Returns (hit, frac, meta).

    ``meta`` is what the trial record carries beyond the outcome: the haystack the
    generator drew (its ``(filler, ctx, seed, trial)`` key -- those four determine it),
    the needle depth when the task pins a grid, the needle label, a sha256 over the
    exact token ids that were fed, and the two memory ratios the aggregate row averages.
    """
    hay, query, targets = build_task(
        tok,
        sub,
        task.ctx,
        trial,
        seed,
        N_KEYS,
        N_VALUES,
        N_HOPS,
        filler=task.filler,
        pool=pool,
        depths=task.depths,
    )
    # 12 tokens is enough for one number; every other task answers with several.
    max_new = 12 if task.tasks == ["niah_single"] else 40
    hit, ratio, frac, sbits = retrieve(
        model, tok, arm, hay, query, targets, device, chunk, n, h_kv, max_new, task=sub, idx=trial
    )
    depth = None
    if task.depths and sub == "niah_single":
        depth = float(task.depths[trial % len(task.depths)])
    return (
        int(hit),
        frac,
        {
            "haystack_id": f"{task.filler}:{task.ctx}:{seed}:{trial}",
            "depth": depth,
            "code_family": code_family(sub, trial),
            "prompt_sha256": prompt_sha256(hay, query),
            "ratio": ratio,
            "sbits": sbits,
        },
    )


def prompt_sha256(*ids: torch.Tensor) -> str:
    """sha256 over the exact token ids of a prompt, so two pods' cells can be compared
    without trusting that they built the same text."""
    h = hashlib.sha256()
    for t in ids:
        h.update(t.detach().cpu().to(torch.int64).numpy().tobytes())
    return h.hexdigest()
