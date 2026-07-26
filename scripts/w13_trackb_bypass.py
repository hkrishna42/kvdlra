"""Week-13 Track-B (warm-up retrieval fix) DESIGN-CHECK: statically pin the
first-ingest-chunk SLASH bypass so the trace in ``results/w13-trackb-design.md``
is machine-checkable and reproducible.

This is a $0 CPU probe: it parses ``src/kvdlra/cache/bug_cache.py`` with the
``ast`` module -- NO torch, NO GPU, NO model, NO network -- and asserts the
load-bearing control-flow facts behind the bypass:

* ``_prefill`` absorbs the first block's middle via ``_absorb_columns`` directly
  and NEVER calls ``_absorb_block_slash`` (the SLASH select/demote path);
* ``_absorb_block_slash`` (the only site that writes ``hh_k``/``hh_v``/``hh_pos``)
  is reached ONLY through ``_absorb_block_into_stream``, which ``_prefill`` never
  calls;
* ``update`` routes ``cumulative_length == 0`` to ``_prefill`` BEFORE it checks
  ``_mode == "ingest"``, so the FIRST chunk of a chunked ingest hits ``_prefill``
  too (not just single-shot) -- i.e. the bypass survives chunked prefill;
* ``_reset_state`` initialises the exact tier empty; ``stored_state_numel`` counts
  ``hh_k``/``hh_v`` verbatim and ``hh_pos`` -- the accounting hooks the seed must
  leave unchanged.

Every asserted fact is emitted with the source line numbers that back it to
``results/w13-trackb-facts.json``. PROXY CAVEAT: this is a *static* control-flow
proof of the bypass, not a runtime measurement; whether seeding actually recovers
early-needle retrieval at 32K/64K (and preserves ppl) is a Phase-2 GPU question.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "kvdlra" / "cache" / "bug_cache.py"
OUT = REPO / "results" / "w13-trackb-facts.json"

# Observed deployment constants (NOT parsed here -- traceable to their sources).
# The deployed retrieval arms run chunked ingest at --chunk 4096 (120 occurrences
# in results/gpu_logs/*.log); docs/week12-next-session.md:133 states plainly that
# "bugslash/bugevict REQUIRE --chunk>0 (single-shot bypasses the exact tier)".
DEPLOYED_CHUNK = 4096
N_SINK = 4
RECENT_WINDOW_DEFAULT = 64  # BugStreamingCache default; a config knob, not a law


@dataclass
class MethodInfo:
    """A class method's self-call targets (name -> line numbers)."""

    name: str
    lineno: int
    self_calls: dict[str, list[int]] = field(default_factory=dict)
    assigns_none: set[str] = field(default_factory=set)
    name_hits: dict[str, list[int]] = field(default_factory=dict)


def _self_call_name(node: ast.Call) -> str | None:
    """Return ``foo`` for a ``self.foo(...)`` call, else ``None``."""
    fn = node.func
    if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "self":
        return fn.attr
    return None


def _collect_methods(tree: ast.Module) -> dict[str, MethodInfo]:
    """Index every ``BugStreamingLayer`` method by name (the layer holds the
    streaming control flow; ``BugStreamingCache`` only drives it)."""
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
                # Track `self.attr = None` (tier init) and any Attribute read of
                # self.attr (for stored_state_numel membership). The tier init in
                # _reset_state is annotated (`self.hh_k: Tensor | None = None`),
                # so both Assign and AnnAssign must be handled.
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if (
                            isinstance(t, ast.Attribute)
                            and isinstance(t.value, ast.Name)
                            and t.value.id == "self"
                            and isinstance(sub.value, ast.Constant)
                            and sub.value.value is None
                        ):
                            info.assigns_none.add(t.attr)
                if isinstance(sub, ast.AnnAssign):
                    tgt_ann = sub.target
                    if (
                        isinstance(tgt_ann, ast.Attribute)
                        and isinstance(tgt_ann.value, ast.Name)
                        and tgt_ann.value.id == "self"
                        and isinstance(sub.value, ast.Constant)
                        and sub.value.value is None
                    ):
                        info.assigns_none.add(tgt_ann.attr)
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"
                ):
                    info.name_hits.setdefault(sub.attr, []).append(sub.lineno)
            methods[item.name] = info
    return methods


def _branch_order_in_update(tree: ast.Module) -> tuple[int, int]:
    """Line numbers of the ``return self._prefill(...)`` under the
    ``cumulative_length == 0`` guard and the ``return self._ingest_chunk(...)``
    under the ``_mode == "ingest"`` guard, proving prefill is checked first."""
    prefill_line = -1
    ingest_line = -1
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef) or cls.name != "BugStreamingLayer":
            continue
        for item in cls.body:
            if not (isinstance(item, ast.FunctionDef) and item.name == "update"):
                continue
            for sub in ast.walk(item):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call):
                    tgt = _self_call_name(sub.value)
                    if tgt == "_prefill" and prefill_line < 0:
                        prefill_line = sub.lineno
                    elif tgt == "_ingest_chunk" and ingest_line < 0:
                        ingest_line = sub.lineno
    return prefill_line, ingest_line


def _callers_of(methods: dict[str, MethodInfo], target: str) -> dict[str, list[int]]:
    """Which methods call ``self.<target>(...)``, with the call line numbers."""
    return {m.name: m.self_calls[target] for m in methods.values() if target in m.self_calls}


def main() -> None:
    tree = ast.parse(SRC.read_text(), filename=str(SRC))
    methods = _collect_methods(tree)

    prefill = methods["_prefill"]
    slash_callers = _callers_of(methods, "_absorb_block_slash")
    absorb_stream_callers = _callers_of(methods, "_absorb_block_into_stream")
    prefill_line, ingest_line = _branch_order_in_update(tree)
    reset = methods["_reset_state"]
    numel = methods["stored_state_numel"]

    facts: dict[str, object] = {}

    # F1: the bypass site -- _prefill absorbs directly, never via SLASH select.
    facts["F1_prefill_calls_absorb_columns"] = {
        "value": "_absorb_columns" in prefill.self_calls,
        "lines": prefill.self_calls.get("_absorb_columns", []),
    }
    facts["F1_prefill_never_calls_absorb_block_slash"] = {
        "value": "_absorb_block_slash" not in prefill.self_calls,
        "lines": prefill.self_calls.get("_absorb_block_slash", []),
    }

    # F2: the SLASH select path (the ONLY writer of hh_k/hh_v/hh_pos) is reached
    # only via _absorb_block_into_stream, which _prefill never calls.
    facts["F2_slash_only_caller_is_absorb_block_into_stream"] = {
        "value": set(slash_callers) == {"_absorb_block_into_stream"},
        "callers": slash_callers,
    }
    facts["F2_prefill_never_calls_absorb_block_into_stream"] = {
        "value": "_prefill" not in absorb_stream_callers,
        "callers": absorb_stream_callers,
    }

    # F3: update checks cumulative_length==0 (-> _prefill) BEFORE _mode=="ingest"
    # (-> _ingest_chunk), so the first chunk of a chunked ingest hits _prefill.
    facts["F3_prefill_branch_precedes_ingest_branch"] = {
        "value": 0 < prefill_line < ingest_line,
        "prefill_return_line": prefill_line,
        "ingest_return_line": ingest_line,
    }

    # F4: _ingest_chunk defers (no absorb of its own); consolidate runs the absorb.
    ingest_chunk = methods["_ingest_chunk"]
    facts["F4_ingest_chunk_defers_absorb"] = {
        "value": (
            "_absorb_block_into_stream" not in ingest_chunk.self_calls
            and "_absorb_columns" not in ingest_chunk.self_calls
        ),
        "self_calls": sorted(ingest_chunk.self_calls),
    }

    # F5: the exact tier starts empty (retrieval must fill it later).
    facts["F5_reset_state_inits_tier_none"] = {
        "value": {"hh_k", "hh_v", "hh_pos"} <= reset.assigns_none,
        "assigned_none": sorted(reset.assigns_none & {"hh_k", "hh_v", "hh_pos", "hh_score"}),
    }

    # F6: accounting -- stored_state_numel counts hh_k/hh_v (verbatim) + hh_pos.
    facts["F6_stored_state_numel_counts_hh"] = {
        "value": all(a in numel.name_hits for a in ("hh_k", "hh_v", "hh_pos")),
        "lines": {a: numel.name_hits.get(a, []) for a in ("hh_k", "hh_v", "hh_pos", "hh_score")},
    }

    # Derived (traceable, not a runtime number): the warm-up window = the first
    # chunk's MIDDLE, which is exactly what _prefill sends past SLASH.
    warmup_mid = DEPLOYED_CHUNK - N_SINK - RECENT_WINDOW_DEFAULT
    facts["warmup_window_estimate"] = {
        "deployed_chunk": DEPLOYED_CHUNK,
        "n_sink": N_SINK,
        "recent_window_assumed": RECENT_WINDOW_DEFAULT,
        "first_chunk_mid_tokens_bypassing_slash": warmup_mid,
        "note": (
            "tokens 4..(chunk-recent) of the FIRST chunk go _prefill->_absorb_columns, "
            "never entering hh_k; config-dependent via recent_window. Matches the "
            "Week-12 ~4-5K absolute warm-up window at 64K."
        ),
    }

    all_pass = all(bool(v["value"]) for v in facts.values() if isinstance(v, dict) and "value" in v)
    facts["ALL_TRACE_FACTS_CONFIRMED"] = all_pass

    OUT.write_text(json.dumps(facts, indent=2, sort_keys=True))
    print(f"wrote {OUT.relative_to(REPO)}")
    for k, v in facts.items():
        if isinstance(v, dict) and "value" in v:
            print(f"  [{'OK' if v['value'] else 'XX'}] {k}")
    print(f"ALL_TRACE_FACTS_CONFIRMED = {all_pass}")
    if not all_pass:
        raise SystemExit("a trace fact FAILED -- the bypass model is stale, re-read the source")


if __name__ == "__main__":
    main()
