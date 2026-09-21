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
    model: Any, cache: BugStreamingCache, ids: torch.Tensor, n_new: int, chunk: int, compare: bool
) -> tuple[list[int], dict[int, float]]:
    """Chunked prefill of ``ids`` (1, T) keeping the last chunk's logits, then ``n_new``
    greedy steps at true positions, all under the cache's attach scope. ``compare`` turns on
    the attention function's reconstruct comparison for the first decode step."""
    t = int(ids.shape[1])
    toks: list[int] = []
    diffs: dict[int, float] = {}
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
            tok = out.logits[:, -1].argmax(-1).view(1, 1)
    return toks, diffs


def check_prompt(
    model: Any, arm: dict[str, Any], ids: torch.Tensor, *, index: int, n_new: int, chunk: int
) -> dict[str, Any]:
    """One prompt (a 1-D token tensor) through the kernel arm and its reconstruct twin."""
    x = ids.view(1, -1).to(next(model.parameters()).device)
    cache = arm["make"]()
    kern, diffs = _greedy(model, cache, x, n_new, chunk, compare=True)
    twin = BugStreamingCache(model, **{**arm["kwargs"], "decode_attention": "reconstruct"})
    recon, _ = _greedy(model, twin, x, n_new, chunk, compare=False)
    first = next((s for s, (a, b) in enumerate(zip(kern, recon, strict=True)) if a != b), None)
    if not diffs:
        raise RuntimeError("the compare step recorded no layer: the kernel path was not taken")
    worst = max(diffs, key=lambda k: diffs[k])
    if first is not None:
        print(f"[kernel_check mismatch prompt={index} step={first} kernel={kern[first]} "
              f"reconstruct={recon[first]}", flush=True)  # fmt: skip
    if index == 0:
        print(f"[kernel_check layers prompt=0 arm={arm['name']} diffs="
              + ",".join(f"{diffs[k]:.3e}" for k in sorted(diffs)), flush=True)  # fmt: skip
    return {
        "arm": arm["name"], "ctx": int(x.shape[1]), "prompt": index, "n_new": n_new,
        "match": int(first is None), "first_mismatch": first, "max_abs_diff": diffs[worst],
        "worst_layer": worst, "prompt_sha256": _sha(x),
        "backend": cache.kernel_backend, "error": None,
    }  # fmt: skip


def failed_row(
    arm: dict[str, Any], ids: torch.Tensor, *, index: int, n_new: int, error: str
) -> dict[str, Any]:
    """The row of a prompt that raised: recorded, never dropped (CLAUDE.md)."""
    return {
        "arm": arm["name"], "ctx": int(ids.numel()), "prompt": index, "n_new": n_new,
        "match": 0, "first_mismatch": None, "max_abs_diff": None, "worst_layer": None,
        "prompt_sha256": _sha(ids.view(-1)), "backend": None, "error": error,
    }  # fmt: skip


def format_line(row: dict[str, Any]) -> str:
    """The ``[kernel_check prompt=...]`` line `records.KERNEL_CHECK_RE` reads back."""
    first = "-" if row["first_mismatch"] is None else row["first_mismatch"]
    diff = "-" if row["max_abs_diff"] is None else f"{row['max_abs_diff']:.3e}"
    worst = "-" if row["worst_layer"] is None else row["worst_layer"]
    line = (
        f"[kernel_check prompt={row['prompt']} arm={row['arm']} ctx={row['ctx']}"
        f" n_new={row['n_new']} match={row['match']} first_mismatch={first}"
        f" max_abs_diff={diff} worst_layer={worst} sha={row['prompt_sha256']}"
        # L4.fw1: which backend attended (`-` where none was recorded -- an errored prompt,
        # or a log written before this field existed). Appended before the `error=` tail so
        # every field that came first keeps its place.
        f" backend={row.get('backend') or '-'}"
    )
    return line + (f" error={row['error']}" if row["error"] else "")
