"""The kernel correctness check on the pod's own model (prereg/kernel_smoke.md §4, Amendment 1).

Per prompt and kernel arm: greedy decode under the kernel and under the reconstruct twin,
token for token (GATES §G4 line 2: "full-model greedy decode token-exact on >= 14/16
prompts, mismatches logged"), and on the first kernel decode step the attention function's
compare mode records max|Δ| against reconstruct-then-attend for EVERY layer on the live K/V
and query -- the single-layer check on real 8B K/V that no 8B dump exists to run offline.
`kvdlra.eval.runner._kernel_check_rows` loops the prompts and arms and records one row each.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from kvdlra.cache import BugStreamingCache

__all__ = ["check_prompt", "failed_row", "format_line", "is_kernel_arm"]


def is_kernel_arm(arm: dict[str, Any]) -> bool:
    # `bool(...)`: an arm dict is `dict[str, Any]`, so the comparison is Any to mypy.
    return bool(arm["kind"] == "bug" and arm.get("kwargs", {}).get("decode_attention") == "kernel")


def _sha(ids: torch.Tensor) -> str:
    return hashlib.sha256(ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def _greedy(
    model: Any, cache: BugStreamingCache, ids: torch.Tensor, n_new: int, chunk: int,
    compare: bool, ref_toks: list[int] | None = None,
) -> tuple[
    list[int],
    dict[int, tuple[float, float]],
    dict[int, tuple[int, float, float]],
    tuple[int, float] | None,
]:  # fmt: skip
    """Chunked prefill of ``ids`` (1, T) keeping the last chunk's logits, then ``n_new``
    greedy steps at true positions, all under the cache's attach scope. ``compare`` turns on
    the attention function's reconstruct comparison for the first decode step.

    Returns the tokens fed; the compare pairs ``layer -> (max|Δ|, max|ref|)`` (Amendment 2
    A2.2); the per-step ``position -> (argmax, top1, top2)`` of every decode output (positions
    1..n_new); and -- only when ``ref_toks`` (the reconstruct twin's tokens, decoded FIRST) is
    given -- the ``(position, kernel logit at the twin's argmax)`` of the first position the
    two disagree on. That last is one scalar gathered in the loop (A2.3), so no full
    128k-wide logit row is kept for a single number 32 steps later."""
    t = int(ids.shape[1])
    toks: list[int] = []
    diffs: dict[int, tuple[float, float]] = {}
    tops: dict[int, tuple[int, float, float]] = {}
    gather: tuple[int, float] | None = None
    with cache.attach(model):
        with cache.ingesting():
            for start in range(0, t, chunk):
                stop = min(t, start + chunk)
                pos = torch.arange(start, stop, device=ids.device).unsqueeze(0)
                out = model(ids[:, start:stop], past_key_values=cache, use_cache=True,
                            position_ids=pos, logits_to_keep=1)  # fmt: skip
                cache.consolidate()
        tok = out.logits[:, -1].argmax(-1).view(1, 1)
        for s in range(n_new):
            toks.append(int(tok))
            if compare and s == 0:
                cache.kernel_compare = {}
            pos = torch.tensor([[t + s]], device=ids.device)
            out = model(tok, past_key_values=cache, use_cache=True, position_ids=pos)
            if compare and s == 0:
                diffs = dict(cache.kernel_compare or {})
                cache.kernel_compare = None
            logits = out.logits[:, -1]  # (1, vocab)
            top2 = logits.topk(2)
            nxt, produced = int(top2.indices[0, 0]), s + 1  # the token chosen for `produced`
            tops[produced] = (nxt, float(top2.values[0, 0]), float(top2.values[0, 1]))
            if (
                ref_toks is not None
                and gather is None
                and produced < len(ref_toks)
                and nxt != ref_toks[produced]
            ):
                gather = (produced, float(logits[0, ref_toks[produced]]))
            tok = top2.indices[:, :1].view(1, 1)
    return toks, diffs, tops, gather


def check_prompt(
    model: Any, arm: dict[str, Any], ids: torch.Tensor, *, index: int, n_new: int, chunk: int
) -> dict[str, Any]:
    """One prompt (a 1-D token tensor) through the kernel arm and its reconstruct twin."""
    x = ids.view(1, -1).to(next(model.parameters()).device)
    # The reconstruct twin decodes FIRST: its greedy tokens are what the kernel arm's single
    # gather (kernel logit at the twin's argmax, A2.3) reads at the mismatching step, so no
    # full logit row is kept. `del twin` before the kernel cache fills keeps one 4096-ctx cache
    # alive at a time (R-L4-35: the two otherwise coexist; hygiene, no recorded number reads it).
    twin = BugStreamingCache(model, **{**arm["kwargs"], "decode_attention": "reconstruct"})
    recon, _, recon_tops, _ = _greedy(model, twin, x, n_new, chunk, compare=False)
    del twin
    cache = arm["make"]()
    kern, diffs, _, gather = _greedy(model, cache, x, n_new, chunk, compare=True, ref_toks=recon)
    if not diffs:
        raise RuntimeError("the compare step recorded no layer: the kernel path was not taken")
    first = next((s for s, (a, b) in enumerate(zip(kern, recon, strict=True)) if a != b), None)
    # The absolute worst max|Δ| (the number A1.3 reported): still reported, no longer a bar (A2.2).
    worst = max(diffs, key=lambda k: diffs[k][0])
    # The relative per-layer bar max_l(max|Δ_l| / max|ref_l|); a layer with max|ref|==0 is
    # excluded and reported (A2.2), never divided by.
    rel = {la: d / m for la, (d, m) in diffs.items() if m != 0.0}
    zero_ref = sorted(la for la, (_d, m) in diffs.items() if m == 0.0)
    rel_worst_layer = max(rel, key=lambda la: rel[la]) if rel else None
    rel_max_diff = rel[rel_worst_layer] if rel_worst_layer is not None else None
    ref_max = diffs[rel_worst_layer][1] if rel_worst_layer is not None else None
    # The first mismatch's reconstruct top-2 gap and the kernel's logit at the twin's argmax
    # (A2.3): `-`/None on a prompt that matched.
    if first is not None and first in recon_tops:
        _argmax, top1, top2 = recon_tops[first]
        gap_at_mismatch: float | None = top1 - top2
        kernel_logit_for_ref_argmax = gather[1] if gather is not None else None
    else:
        gap_at_mismatch = kernel_logit_for_ref_argmax = None
    if first is not None:
        print(f"[kernel_check mismatch prompt={index} step={first} kernel={kern[first]} "
              f"reconstruct={recon[first]}", flush=True)  # fmt: skip
    if zero_ref:
        print(f"[kernel_check zero_ref prompt={index} arm={arm['name']} layers={zero_ref}"
              " (max|ref|==0, excluded from the relative bar)", flush=True)  # fmt: skip
    if index == 0:
        order = sorted(diffs)
        ds = ",".join(f"{diffs[k][0]:.3e}" for k in order)
        refs = ",".join(f"{diffs[k][1]:.3e}" for k in order)  # A2.5: max|ref| beside each max|Δ|
        print(f"[kernel_check layers prompt=0 arm={arm['name']} diffs={ds} refs={refs}", flush=True)
    return {
        "arm": arm["name"], "ctx": int(x.shape[1]), "prompt": index, "n_new": n_new,
        "match": int(first is None), "first_mismatch": first, "max_abs_diff": diffs[worst][0],
        "worst_layer": worst, "rel_max_diff": rel_max_diff, "rel_worst_layer": rel_worst_layer,
        "ref_max": ref_max, "gap_at_mismatch": gap_at_mismatch,
        "kernel_logit_for_ref_argmax": kernel_logit_for_ref_argmax,
        "prompt_sha256": _sha(x), "backend": cache.kernel_backend, "error": None,
    }  # fmt: skip


def failed_row(
    arm: dict[str, Any], ids: torch.Tensor, *, index: int, n_new: int, error: str
) -> dict[str, Any]:
    """The row of a prompt that raised: recorded, never dropped (CLAUDE.md)."""
    return {
        "arm": arm["name"], "ctx": int(ids.numel()), "prompt": index, "n_new": n_new,
        "match": 0, "first_mismatch": None, "max_abs_diff": None, "worst_layer": None,
        "rel_max_diff": None, "rel_worst_layer": None, "ref_max": None,
        "gap_at_mismatch": None, "kernel_logit_for_ref_argmax": None,
        "prompt_sha256": _sha(ids.view(-1)), "backend": None, "error": error,
    }  # fmt: skip


def format_line(row: dict[str, Any]) -> str:
    """The ``[kernel_check prompt=...]`` line `records.KERNEL_CHECK_RE` reads back."""
    first = "-" if row["first_mismatch"] is None else row["first_mismatch"]
    diff = "-" if row["max_abs_diff"] is None else f"{row['max_abs_diff']:.3e}"
    worst = "-" if row["worst_layer"] is None else row["worst_layer"]

    def _num(key: str) -> str:  # a float field, or `-` when absent
        v = row.get(key)
        return "-" if v is None else f"{v:.3e}"

    rel_layer = "-" if row.get("rel_worst_layer") is None else row["rel_worst_layer"]
    line = (
        f"[kernel_check prompt={row['prompt']} arm={row['arm']} ctx={row['ctx']}"
        f" n_new={row['n_new']} match={row['match']} first_mismatch={first}"
        f" max_abs_diff={diff} worst_layer={worst} sha={row['prompt_sha256']}"
        # L4.fw1: which backend attended (`-` where none was recorded -- an errored prompt,
        # or a log written before this field existed). Appended before the `error=` tail so
        # every field that came first keeps its place.
        f" backend={row.get('backend') or '-'}"
        # Amendment 2 (A2.5): the five relative-bar / near-tie fields, appended after
        # `backend=` and before the `error=` tail, `-` when absent, so every archived row
        # (which has none of them) still parses with None.
        f" rel_max_diff={_num('rel_max_diff')} rel_worst_layer={rel_layer}"
        f" ref_max={_num('ref_max')} gap_at_mismatch={_num('gap_at_mismatch')}"
        f" kernel_logit_for_ref_argmax={_num('kernel_logit_for_ref_argmax')}"
    )
    return line + (f" error={row['error']}" if row["error"] else "")
