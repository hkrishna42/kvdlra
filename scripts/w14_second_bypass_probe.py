"""Week-14 W-5: second-structural-bypass probe.

QUESTION (pre-registered). Week-13 T-B fixed the *first-ingest-chunk* SLASH bypass
(``bugSseed``): the first chunk routes through ``_prefill`` -> ``_absorb_columns``
and never reaches ``_absorb_block_slash`` (the ONLY writer of the exact tier
``hh_k``/``hh_v``/``hh_pos``), so a first-chunk-middle needle can never enter the
exact tier. Is the first chunk the ONLY place a needle is *structurally* barred
from the exact tier, or is there a SECOND bypass -- a needle landing on a chunk
boundary mid-stream, or across the graduation/eviction transition?

PRE-REGISTERED BAR. FUND if a second real structural bypass exists AND has a cheap
fix (then specify the fix + a unit-test design, do NOT implement it). KILL if the
first chunk is the only structural gap (steady-state SLASH already captures
mid-stream / boundary needles). Honest prior: likely a KILL.

METHOD (leads with the $0 code-trace).

* Phase 1 -- static AST trace (no torch needed for the facts): enumerate EVERY
  writer of ``hh_k``/``hh_v``/``hh_pos`` and EVERY caller of the absorb paths, and
  map which ingest paths reach ``_absorb_block_slash`` and which bypass it. This is
  a machine-checkable proof of the control flow, extending
  ``scripts/w13_trackb_bypass.py`` to the *second-bypass* question.

* Phase 2 -- dynamic confirmation on a hermetic tiny Llama that drives the REAL
  routing end-to-end (``update`` -> ``_ingest_chunk`` -> ``consolidate`` ->
  ``_absorb_block_into_stream`` -> ``_absorb_block_slash``). Plant a needle at (a) a
  mid-stream chunk boundary and (b) deep inside a steady-state block, seed OFF, and
  check whether it CAN enter the exact tier. Contrast with the known first-chunk
  bypass (seed off = barred; seed on = captured). ``in_hh`` is itself proof the
  needle reached SLASH -- Phase 1 proves SLASH is the tier's only writer -- and an
  instrumentation wrapper independently records the positions SLASH was offered.

* Phase 3 -- real-dump supplement (``dumps/llama3.2-1b/*len4096*``): confirm the
  CAPTURE (selection) half on real, diverse (non-homogeneous) data -- real
  background columns read as low-surprise against a real streamed rank-r basis,
  while a planted out-of-subspace needle scores ~1.0 and ranks #1 in its block.
  Uses the REAL ``_surprise_scores`` + REAL ``augmented_bug_step`` basis builder.
  (The STRUCTURAL routing verdict rests on Phases 1-2; Phase 3 is robustness.)

PROXY CAVEAT. Phases 1-2 are exact for the structural question (control flow +
real routing on a real HF forward). Phase 3's needle is synthetic-outlier by
construction and its basis is a real streamed prefix, not a full 32K stream -- it
confirms the mechanism on diverse data, not an end-to-end retrieval number.

Usage::

    uv run python scripts/w14_second_bypass_probe.py \
        --dumps dumps/llama3.2-1b --out-json results/w14-second-bypass-facts.json
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

# The hatchling editable ``.pth`` is flaky on this Mac (see auto-memory /
# docs/*handover*), so ``kvdlra`` may not be importable without help; mirror
# pytest's ``pythonpath=["src"]`` by prepending src/ deterministically.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import Tensor
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer
from kvdlra.integrators.streaming_torch import augmented_bug_step

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "kvdlra" / "cache" / "bug_cache.py"
OUT_DEFAULT = REPO / "results" / "w14-second-bypass-facts.json"

HH_ATTRS = ("hh_k", "hh_v", "hh_pos", "hh_score")


# ------------------------------------------------------------- Phase 1: AST


@dataclass
class MethodInfo:
    """A ``BugStreamingLayer`` method's ``self.foo(...)`` targets, ``return`` lines,
    and ``self.hh_*`` assignment sites (line, is-``None``-constant)."""

    name: str
    lineno: int
    self_calls: dict[str, list[int]] = field(default_factory=dict)
    returns: list[int] = field(default_factory=list)
    hh_assigns: dict[str, list[tuple[int, bool]]] = field(default_factory=dict)


def _self_call_name(node: ast.Call) -> str | None:
    fn = node.func
    if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "self":
        return fn.attr
    return None


def _is_self_attr(node: ast.expr, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _is_none_const(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Constant) and value.value is None


def _collect_methods(tree: ast.Module) -> dict[str, MethodInfo]:
    """Index every ``BugStreamingLayer`` method (it holds the streaming control
    flow; ``BugStreamingCache`` only drives it)."""
    methods: dict[str, MethodInfo] = {}
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef) or cls.name != "BugStreamingLayer":
            continue
        for item in cls.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            info = MethodInfo(name=item.name, lineno=item.lineno)
            for sub in ast.walk(item):
                if isinstance(sub, ast.Call):
                    tgt = _self_call_name(sub)
                    if tgt is not None:
                        info.self_calls.setdefault(tgt, []).append(sub.lineno)
                elif isinstance(sub, ast.Return):
                    info.returns.append(sub.lineno)
                elif isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        for attr in HH_ATTRS:
                            if _is_self_attr(target, attr):
                                info.hh_assigns.setdefault(attr, []).append(
                                    (sub.lineno, _is_none_const(sub.value))
                                )
                elif isinstance(sub, ast.AnnAssign):
                    for attr in HH_ATTRS:
                        if _is_self_attr(sub.target, attr):
                            info.hh_assigns.setdefault(attr, []).append(
                                (sub.lineno, _is_none_const(sub.value))
                            )
            methods[item.name] = info
    return methods


def _branch_order_in_update(tree: ast.Module) -> dict[str, int]:
    """First ``return self._prefill/_ingest_chunk/_decode_step(...)`` line inside
    ``update`` -- proves the dispatch order (prefill checked before ingest)."""
    out: dict[str, int] = {}
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef) or cls.name != "BugStreamingLayer":
            continue
        for item in cls.body:
            if not (isinstance(item, ast.FunctionDef) and item.name == "update"):
                continue
            for sub in ast.walk(item):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call):
                    tgt = _self_call_name(sub.value)
                    if tgt in ("_prefill", "_ingest_chunk", "_decode_step") and tgt not in out:
                        out[tgt] = sub.lineno
    return out


def _callers_of(methods: dict[str, MethodInfo], target: str) -> dict[str, list[int]]:
    return {m.name: m.self_calls[target] for m in methods.values() if target in m.self_calls}


def _nonnull_hh_writers(methods: dict[str, MethodInfo], attr: str) -> dict[str, list[int]]:
    """Methods that assign a NON-``None`` value to ``self.<attr>`` (the real
    writers; the ``_reset_state`` ``= None`` init is excluded)."""
    out: dict[str, list[int]] = {}
    for m in methods.values():
        lines = [ln for (ln, is_none) in m.hh_assigns.get(attr, []) if not is_none]
        if lines:
            out[m.name] = lines
    return out


def phase1_static_trace() -> dict[str, Any]:
    tree = ast.parse(SRC.read_text(), filename=str(SRC))
    methods = _collect_methods(tree)
    facts: dict[str, Any] = {}

    # G1: the exact tier's true writer set. hh_k/hh_v/hh_pos are written non-None
    # ONLY by _absorb_block_slash -> reaching that method is NECESSARY to enter the
    # tier. (hh_score also gets an EMA update in observe_attention, but that only
    # re-weights an already-created attn-selected tier; it never creates hh_k/v/pos.)
    writers_kvp = {attr: _nonnull_hh_writers(methods, attr) for attr in ("hh_k", "hh_v", "hh_pos")}
    kvp_writer_names = {name for w in writers_kvp.values() for name in w}
    facts["G1_hh_kvpos_written_only_by_absorb_block_slash"] = {
        "value": kvp_writer_names == {"_absorb_block_slash"},
        "writers": writers_kvp,
    }
    hh_score_writers = _nonnull_hh_writers(methods, "hh_score")
    facts["G1b_hh_score_writers"] = {
        # hh_score is written by _absorb_block_slash (creation) and optionally the
        # observe_attention EMA (attn-select re-weight); neither creates hh_k/v/pos.
        "value": set(hh_score_writers) <= {"_absorb_block_slash", "observe_attention"},
        "writers": hh_score_writers,
        "note": "observe_attention only re-weights an existing attn-selected tier.",
    }

    # G2: the tier's only writer (_absorb_block_slash) is reached from exactly two
    # sites -- steady-state graduation (_absorb_block_into_stream) and the shipped
    # warm-up seed (_prefill, seed on). The seed is the SECOND, deliberate entrance
    # that already covers the first-chunk gap; there is no third, accidental one.
    slash_callers = _callers_of(methods, "_absorb_block_slash")
    facts["G2_slash_callers_are_graduation_and_seed"] = {
        "value": set(slash_callers) == {"_absorb_block_into_stream", "_prefill"},
        "callers": slash_callers,
        "note": (
            "_absorb_block_into_stream = steady-state graduation gateway; "
            "_prefill = Week-13 bugSseed warm-up seed (gated by seed_hh_warmup). "
            "In Week-13's pre-merge design-check the only caller was "
            "_absorb_block_into_stream; the seed added the _prefill route (the fix)."
        ),
    }

    # G3: _absorb_block_into_stream (the SLASH gateway) is reached from steady-state
    # graduation (consolidate = deferred ingest absorb) AND decode -- i.e. EVERY
    # mid-stream / decode graduating block passes through it.
    stream_callers = _callers_of(methods, "_absorb_block_into_stream")
    facts["G3_absorb_block_into_stream_callers"] = {
        "value": set(stream_callers) == {"consolidate", "_decode_step"},
        "callers": stream_callers,
    }

    # G4: the low-rank-tail writer (_absorb_columns, the BYPASS sink). Its callers
    # are the whole surface where a token can land in the tail; the ONLY one that is
    # not gated behind SLASH is _prefill.
    cols_callers = _callers_of(methods, "_absorb_columns")
    _tail_writers = {"_prefill", "_absorb_block_into_stream", "_absorb_block_slash"}
    facts["G4_absorb_columns_callers"] = {
        "value": set(cols_callers) == _tail_writers,
        "callers": cols_callers,
        "note": (
            "_absorb_block_slash's call is the post-SLASH demote path (not a bypass); "
            "_absorb_block_into_stream's call is unreachable when hh_enabled (see G6); "
            "_prefill's call is the FIRST-CHUNK bypass (seed off)."
        ),
    }

    # G5: update() dispatch order -- cumulative_length==0 -> _prefill BEFORE the
    # _mode=="ingest" check, so the FIRST chunk of a chunked ingest hits _prefill
    # (the bypass), and only chunks 2..N reach _ingest_chunk (which defers to
    # consolidate -> SLASH).
    order = _branch_order_in_update(tree)
    facts["G5_update_dispatch_prefill_before_ingest"] = {
        "value": 0 < order.get("_prefill", -1) < order.get("_ingest_chunk", -1),
        "return_lines": order,
    }

    # G6: within _absorb_block_into_stream, the _absorb_block_slash call is followed
    # by a `return` BEFORE the direct _absorb_columns call -> under hh_enabled the
    # direct tail-absorb is unreachable, so a graduating block cannot bypass SLASH.
    abis = methods["_absorb_block_into_stream"]
    slash_line = min(abis.self_calls.get("_absorb_block_slash", [10**9]))
    cols_line = min(abis.self_calls.get("_absorb_columns", [10**9]))
    return_between = any(slash_line < r < cols_line for r in abis.returns)
    facts["G6_slash_guarded_by_return_before_direct_absorb"] = {
        "value": slash_line < cols_line and return_between,
        "slash_call_line": slash_line,
        "absorb_columns_call_line": cols_line,
        "return_lines": abis.returns,
    }

    # G7: _prefill contains BOTH absorb paths -- the direct bypass (seed off) and
    # the SLASH route (seed on, the bugSseed fix). So the single structural gap has
    # a single, already-built structural fix.
    prefill = methods["_prefill"]
    facts["G7_prefill_has_both_bypass_and_seed_slash"] = {
        "value": (
            "_absorb_columns" in prefill.self_calls and "_absorb_block_slash" in prefill.self_calls
        ),
        "absorb_columns_lines": prefill.self_calls.get("_absorb_columns", []),
        "absorb_block_slash_lines": prefill.self_calls.get("_absorb_block_slash", []),
    }

    # G8: _ingest_chunk defers (no absorb of its own) -- so chunks 2..N are absorbed
    # only later by consolidate -> _absorb_block_into_stream -> SLASH. No chunk
    # boundary introduces a separate absorb site.
    ingest = methods["_ingest_chunk"]
    facts["G8_ingest_chunk_defers_no_absorb"] = {
        "value": (
            "_absorb_block_into_stream" not in ingest.self_calls
            and "_absorb_columns" not in ingest.self_calls
            and "_absorb_block_slash" not in ingest.self_calls
        ),
        "self_calls": sorted(ingest.self_calls),
    }

    all_pass = all(bool(v["value"]) for v in facts.values() if isinstance(v, dict) and "value" in v)
    facts["ALL_TRACE_FACTS_CONFIRMED"] = all_pass
    return facts


# --------------------------------------------------- Phase 2: dynamic (tiny model)

_H, _D = 2, 16


def _tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=_H,
        head_dim=_D,
        max_position_embeddings=4096,
    )
    model = LlamaForCausalLM(cfg)  # type: ignore[no-untyped-call]
    model.config._attn_implementation = "sdpa"
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _pos(start: int, n: int) -> Tensor:
    return torch.arange(start, start + n).unsqueeze(0)


def _bug_layer(cache: BugStreamingCache) -> BugStreamingLayer:
    return next(layer for layer in cache.layers if isinstance(layer, BugStreamingLayer))


def _needle_prompt(bg_id: int, needle_id: int, t: int, depth: int) -> Tensor:
    ids = torch.full((1, t), bg_id, dtype=torch.long)
    ids[0, depth] = needle_id
    return ids


def _ingest_chunked(
    model: LlamaForCausalLM, cache: BugStreamingCache, ids: Tensor, chunk: int
) -> None:
    """Attach-free chunked ingest (mirrors the deployed retrieval path): the FIRST
    chunk hits ``_prefill``, later chunks defer to ``consolidate`` -> SLASH."""
    t = int(ids.shape[1])
    with cache.ingesting():
        for start in range(0, t, chunk):
            stop = min(t, start + chunk)
            model(
                ids[:, start:stop],
                past_key_values=cache,
                use_cache=True,
                position_ids=_pos(start, stop - start),
            )
            cache.consolidate()


@contextmanager
def _record_slash_offers() -> Iterator[list[int]]:
    """Record every position offered to ``_absorb_block_slash`` (across all layers)
    -- an independent read of "did the needle reach the tier's writer?"."""
    seen: list[int] = []
    orig = BugStreamingLayer._absorb_block_slash

    def wrapper(
        self: BugStreamingLayer,
        grad_k: Tensor,
        grad_v: Tensor,
        grad_pos: Tensor,
        grad_score: Tensor | None,
    ) -> None:
        seen.extend(int(p) for p in grad_pos.tolist())
        orig(self, grad_k, grad_v, grad_pos, grad_score)

    BugStreamingLayer._absorb_block_slash = wrapper  # type: ignore[method-assign]
    try:
        yield seen
    finally:
        BugStreamingLayer._absorb_block_slash = orig  # type: ignore[method-assign]


def _run_case(
    model: LlamaForCausalLM,
    label: str,
    depth: int,
    *,
    seed_on: bool,
    chunk: int,
    t: int,
) -> dict[str, Any]:
    ids = _needle_prompt(bg_id=5, needle_id=200, t=t, depth=depth)
    cache = BugStreamingCache(
        model,
        rank=4,
        coord_budget=256,  # > total absorbed tokens -> no eviction blurs the contrast
        recent_window=8,
        absorb_block=4,
        prefill_block_size=8,
        retention="lowrank_surprise",
        hh_budget=1,  # the needle is THE most surprising token -> deterministic
        hh_select="surprise",
        seed_hh_warmup=seed_on,
    )
    with torch.no_grad(), _record_slash_offers() as offers:
        _ingest_chunked(model, cache, ids, chunk)
    layer = _bug_layer(cache)
    hh = layer.hh_pos.tolist() if layer.hh_pos is not None else []
    mid = layer.mid_pos.tolist() if layer.mid_pos is not None else []
    return {
        "label": label,
        "depth": depth,
        "seed_on": seed_on,
        "chunk": chunk,
        "t": t,
        "in_first_chunk": depth < chunk,
        "on_chunk_boundary": (depth % chunk) == 0,
        "offered_to_slash": depth in offers,
        "in_hh": depth in hh,
        "in_mid_tail": depth in mid,
    }


def phase2_dynamic() -> dict[str, Any]:
    model = _tiny_model()
    chunk, t = 32, 192
    # (label, depth, seed_on). (a) chunk-boundary mid-stream (depth % chunk == 0,
    # past the first chunk); (b) deep inside a steady-state block (mid-stream, off
    # boundary); the KNOWN first-chunk-middle bypass (depth in [n_sink, n_sink+mid))
    # seed off (barred) vs seed on (captured -- the shipped fix, a sanity control).
    specs: list[tuple[str, int, bool]] = [
        ("mid_chunk_boundary", 64, False),
        ("mid_chunk_boundary", 96, False),
        ("mid_deep_in_block", 50, False),
        ("mid_deep_in_block", 130, False),
        ("first_chunk_middle_seed_off", 15, False),
        ("first_chunk_middle_seed_on", 15, True),
    ]
    cases: list[dict[str, Any]] = [
        _run_case(model, label, depth, seed_on=seed, chunk=chunk, t=t)
        for (label, depth, seed) in specs
    ]

    def _find(label: str) -> list[dict[str, Any]]:
        return [c for c in cases if c["label"] == label]

    # Mid-stream (boundary + deep) needles CAN enter the exact tier under
    # steady-state SLASH (seed off) -> they are NOT structurally barred.
    midstream = _find("mid_chunk_boundary") + _find("mid_deep_in_block")
    checks = {
        "midstream_needles_enter_hh": all(
            c["in_hh"] and c["offered_to_slash"] and not c["in_first_chunk"] for c in midstream
        ),
        "first_chunk_needle_barred_seed_off": all(
            (not c["in_hh"]) and (not c["offered_to_slash"]) and c["in_mid_tail"]
            for c in _find("first_chunk_middle_seed_off")
        ),
        "first_chunk_needle_captured_seed_on": all(
            c["in_hh"] and c["offered_to_slash"] for c in _find("first_chunk_middle_seed_on")
        ),
    }
    return {
        "config": {"chunk": chunk, "t": t, "hh_budget": 1, "rank": 4, "coord_budget": 256},
        "cases": cases,
        "checks": checks,
        "ALL_DYNAMIC_CHECKS_PASS": all(checks.values()),
    }


# ------------------------------------------------ Phase 3: real-dump supplement


def _load_kpre(dump_dir: Path, layer: int, n_sink: int) -> Tensor:
    """Pre-RoPE feature-by-token matrix ``(h*d, t)`` for one layer, sinks dropped
    (mirrors ``scripts/w13_tracka_probe.load_key_matrix``)."""
    blob = torch.load(dump_dir / f"layer_{layer:02d}.pt", weights_only=False)
    kpre = cast(Tensor, blob["K_pre"]).float()  # (h, t, d)
    h, t, d = kpre.shape
    mat = kpre.permute(0, 2, 1).reshape(h * d, t).contiguous()  # (h*d, t)
    return mat[:, n_sink:] if mat.shape[1] > n_sink else mat


def _stream_basis(cols: Tensor, rank: int, block: int) -> Tensor:
    """Build a rank-``r`` basis over ``cols`` with the REAL deployed
    ``augmented_bug_step`` (the exact streaming-cache integrator)."""
    u: Tensor | None = None
    b: Tensor | None = None
    n = int(cols.shape[1])
    for start in range(0, n, block):
        u, b, _ = augmented_bug_step(u, b, cols[:, start : start + block], rank)
    assert u is not None
    return u


def phase3_real_dump(dumps_root: Path) -> dict[str, Any]:
    dirs = sorted(glob.glob(str(dumps_root / "*len4096*")))
    if not dirs:
        return {"skipped": f"no *len4096* dumps under {dumps_root}"}
    dump_dir = Path(dirs[0])
    n_sink, rank, block = 4, 32, 32
    cols = _load_kpre(dump_dir, layer=0, n_sink=n_sink)  # (512, ~4092)
    # A real, mature basis over a diverse real prefix.
    basis = _stream_basis(cols[:, :1024], rank=rank, block=block)

    # A real mid-stream candidate block (diverse background) + a planted needle:
    # an OUT-OF-SUBSPACE unit direction scaled to the median real column norm (this
    # is exactly what "a distinctive outlier" means to the surprise metric).
    cand = cols[:, 2000 : 2000 + block].clone()  # (512, 32) real background
    torch.manual_seed(0)
    raw = torch.randn(cols.shape[0])
    resid = raw - basis @ (basis.mT @ raw)  # component orthogonal to the basis
    needle = resid / resid.norm() * float(cand.norm(dim=0).median())
    needle_col = block // 2  # place mid-block (position within the block is irrelevant)
    cand[:, needle_col] = needle

    # Score with the REAL layer method (a real BugStreamingLayer as the vehicle; the
    # method uses only self.u_k + self._whiten_key, identity without w_key).
    vehicle = _bug_layer(
        BugStreamingCache(
            _tiny_model(),
            rank=rank,
            coord_budget=64,
            hh_budget=1,
            hh_select="surprise",
            retention="lowrank_surprise",
        )
    )
    vehicle.u_k = basis
    surprise = vehicle._surprise_scores(vehicle._whiten_key(cand))  # real code, real data
    bg_mask = torch.ones(block, dtype=torch.bool)
    bg_mask[needle_col] = False
    needle_surprise = float(surprise[needle_col])
    max_bg_surprise = float(surprise[bg_mask].max())
    median_bg_surprise = float(surprise[bg_mask].median())
    order = torch.argsort(surprise, descending=True)
    needle_rank = int((order == needle_col).nonzero().item())  # 0 == top-1

    return {
        "dump": dump_dir.name,
        "rank": rank,
        "basis_prefix_cols": 1024,
        "needle_surprise": round(needle_surprise, 4),
        "max_background_surprise": round(max_bg_surprise, 4),
        "median_background_surprise": round(median_bg_surprise, 4),
        "needle_rank_in_block": needle_rank,  # 0 => would take the top hh slot
        "checks": {
            "needle_is_top1": needle_rank == 0,
            "needle_separated_from_background": needle_surprise > max_bg_surprise,
        },
        "note": (
            "Real diverse background reads low-surprise vs a real streamed rank-32 "
            "basis; a routed out-of-subspace needle is top-1 -> the CAPTURE half "
            "holds on real data. STRUCTURAL routing verdict is from Phases 1-2."
        ),
    }


# --------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", type=Path, default=REPO / "dumps" / "llama3.2-1b")
    ap.add_argument("--out-json", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    p1 = phase1_static_trace()
    p2 = phase2_dynamic()
    p3 = phase3_real_dump(args.dumps)

    p3_ok = "skipped" in p3 or all(cast("dict[str, bool]", p3["checks"]).values())
    all_pass = (
        bool(p1["ALL_TRACE_FACTS_CONFIRMED"]) and bool(p2["ALL_DYNAMIC_CHECKS_PASS"]) and p3_ok
    )

    # The pre-registered decision: a second structural bypass would show up as a
    # mid-stream / boundary needle that CANNOT enter the tier (barred like the first
    # chunk). Phases 1-2 show the opposite -- every mid-stream / boundary needle
    # reaches SLASH and lands in hh; only the first chunk bypasses.
    second_bypass_found = not p2["checks"]["midstream_needles_enter_hh"]
    verdict = "FUND" if second_bypass_found else "KILL"

    result: dict[str, Any] = {
        "question": (
            "Is the first ingest chunk the ONLY place a needle is structurally barred "
            "from the exact hh tier, or is there a SECOND bypass (chunk boundary "
            "mid-stream / graduation-eviction transition)?"
        ),
        "pre_registered_bar": (
            "FUND if a second real structural bypass exists AND has a cheap fix; "
            "KILL if the first chunk is the only structural gap (steady-state SLASH "
            "captures mid-stream / boundary needles)."
        ),
        "verdict": verdict,
        "verdict_reason": (
            "No second structural bypass: every mid-stream and chunk-boundary needle "
            "reaches _absorb_block_slash (via consolidate/_decode_step -> "
            "_absorb_block_into_stream) and enters the exact tier under steady-state "
            "SLASH; the first-chunk middle (routed _prefill -> _absorb_columns, seed "
            "off) is the ONLY path that never reaches the tier's sole writer."
            if verdict == "KILL"
            else "A mid-stream / boundary needle failed to enter the exact tier -- a "
            "second structural bypass; see phase2 cases."
        ),
        "phase1_static_trace": p1,
        "phase2_dynamic_tiny_model": p2,
        "phase3_real_dump_selection": p3,
        "ALL_CHECKS_PASS": all_pass,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.out_json.relative_to(REPO)}")
    print(f"  phase1 static trace  : ALL_TRACE_FACTS_CONFIRMED = {p1['ALL_TRACE_FACTS_CONFIRMED']}")
    for k, v in p2["checks"].items():
        print(f"  phase2 dynamic check : [{'OK' if v else 'XX'}] {k}")
    if "skipped" in p3:
        print(f"  phase3 real dump     : skipped ({p3['skipped']})")
    else:
        for k, v in p3["checks"].items():
            print(f"  phase3 real check    : [{'OK' if v else 'XX'}] {k}")
    print(f"VERDICT = {verdict}  (ALL_CHECKS_PASS = {all_pass})")
    if not all_pass:
        raise SystemExit("a probe check FAILED -- re-read the trace before trusting the verdict")


if __name__ == "__main__":
    main()
