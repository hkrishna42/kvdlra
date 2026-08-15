"""Week-16 Tier-2 generality gate: a $0 CPU probe that must pass per family BEFORE any
GPU dollar is spent on Mistral-7B-v0.3 or Qwen2.5-7B.

The harness is already model-agnostic (geometry read dynamically, GQA inferred from
shapes, RoPE via each model's own ``rotary_emb``), so the risk is NOT the cache -- it is
the RULER *plumbing* around a new chat template. Five gates, each printing its
pre-registered bar + verdict:

* **G-GATE**  model + tokenizer load (ungated); a 401/403 => use an ungated mirror.
* **G-ROPE**  ``model.model.rotary_emb`` present; ``attention_scaling`` reported (expect
  ~1.0 for Mistral/Qwen vs Llama-3's long-RoPE).
* **G-BIAS**  q/k/v bias presence (Qwen2.5 carries it; it must flow through k_proj capture).
* **G-TAIL**  the template-derived query tail (``w10_ruler._templated``) puts the WHOLE
  question in the decoded query and keeps the needle in the compressed prefill -- the #1
  trap (a wrong Llama-tuned ``_TAIL_K`` = silent 0s).
* **G-SMOKE** a tiny CPU RULER (ctx 2048, niah_single + niah_multikey, arms ``full`` +
  ``bugslash-r32``): **``full`` MUST recover the needle (acc == 1.0) or the family is
  KILLED (no GPU)** -- an uncompressed model that cannot answer means the template /
  geometry is still wrong.

One family per invocation (like ``w11_probe``); run it twice::

    uv run python scripts/w16_tier2_probe.py --model mistralai/Mistral-7B-Instruct-v0.3
    uv run python scripts/w16_tier2_probe.py --model Qwen/Qwen2.5-7B-Instruct

The overall verdict is FUND (authorize GPU for this family) iff G-TAIL, G-ROPE and
G-SMOKE all pass.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import _paths  # noqa: F401  (prepends src/ to sys.path if needed)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from w10_frontier import build_arms
from w10_ruler import build_task, retrieve

JSON_BEGIN = "===W16_TIER2_PROBE_JSON_BEGIN==="
JSON_END = "===W16_TIER2_PROBE_JSON_END==="
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _load(model_id: str, device: str, dtype: str) -> tuple[Any, Any, dict[str, Any]]:
    """G-GATE: load tokenizer + model; a gating error is a clear, actionable verdict."""
    gate: dict[str, Any] = {"gate": "G-GATE", "model": model_id}
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model: Any = AutoModelForCausalLM.from_pretrained(model_id, dtype=_DTYPES[dtype])
        model = model.to(device).eval()
        model.config._attn_implementation = "sdpa"
        gate.update(passed=True, note="loaded ungated")
    except Exception as exc:  # report any load failure as the gate verdict, not a traceback
        gate.update(passed=False, note=f"{type(exc).__name__}: {exc}")
        print(
            f"[G-GATE] bar: loads ungated | verdict: FAIL ({gate['note']}) -> use an "
            f"ungated mirror (e.g. unsloth/*), as with Llama"
        )
        raise SystemExit(1) from exc
    print(f"[G-GATE] bar: loads ungated | verdict: PASS ({model_id})")
    return tok, model, gate


def _g_rope(model: Any) -> dict[str, Any]:
    base = getattr(model, "model", model)
    rotary = getattr(base, "rotary_emb", None)
    scaling = getattr(rotary, "attention_scaling", None)
    passed = rotary is not None
    verdict = "PASS" if passed else "FAIL"
    print(
        f"[G-ROPE] bar: model.rotary_emb present | verdict: {verdict} (attention_scaling={scaling})"
    )
    return {"gate": "G-ROPE", "passed": passed, "attention_scaling": _to_float(scaling)}


def _g_bias(model: Any) -> dict[str, Any]:
    base = getattr(model, "model", model)
    try:
        k_proj = base.layers[0].self_attn.k_proj
        has_bias = getattr(k_proj, "bias", None) is not None
    except (AttributeError, IndexError) as exc:  # unusual attention layout -> report, don't crash
        print(f"[G-BIAS] bar: q/k/v bias reported | verdict: INFO (introspection failed: {exc})")
        return {"gate": "G-BIAS", "passed": True, "k_proj_bias": None}
    # Informational: Qwen2.5 has bias (must flow through capture); Mistral has none.
    print(f"[G-BIAS] bar: q/k/v bias reported | verdict: INFO (k_proj.bias present={has_bias})")
    return {"gate": "G-BIAS", "passed": True, "k_proj_bias": has_bias}


def _g_tail(tok: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Build each smoke task through the real _templated: the needle must land in the
    compressed prefill and the whole question in the decoded query, for THIS template."""
    ok = True
    detail: dict[str, Any] = {}
    for task in ("niah_single", "niah_multikey"):
        try:
            pre, query, targets = build_task(
                tok, task, args.ctx, trial=0, seed=0, n_keys=args.n_keys, n_values=1, n_hops=1
            )
        except ValueError as exc:  # _templated's fail-loud mis-slice tripwire
            ok = False
            detail[task] = {"ok": False, "error": str(exc)}
            continue
        pre_txt = str(tok.decode(pre[0]))
        q_txt = str(tok.decode(query[0]))
        needle_in_pre = targets[0] in pre_txt
        q_in_query = "Reply with only the number" in q_txt
        needle_not_in_query = targets[0] not in q_txt
        task_ok = needle_in_pre and q_in_query and needle_not_in_query
        ok = ok and task_ok
        detail[task] = {
            "ok": task_ok,
            "tail_k": int(query.shape[1]),
            "needle_in_prefill": needle_in_pre,
            "question_in_query": q_in_query,
        }
    verdict = "PASS" if ok else "FAIL"
    tails = {t: d.get("tail_k") for t, d in detail.items()}
    print(
        f"[G-TAIL] bar: question fully in decoded query, needle in prefill | "
        f"verdict: {verdict} (tail_k={tails})"
    )
    return {"gate": "G-TAIL", "passed": ok, "detail": detail}


def _g_smoke(model: Any, tok: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Run the SAME arm machinery the pod uses (build_arms/retrieve): full must hit."""
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    smoke = argparse.Namespace(
        methods=["full", "bugslash"],
        ranks=[32],
        hh_budgets=[1024],
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
        warmup_seed=True,
        score_rank=None,
    )
    rows: list[dict[str, Any]] = []
    full_ok = True
    for task in ("niah_single", "niah_multikey"):
        hay, query, targets = build_task(
            tok, task, args.ctx, trial=0, seed=0, n_keys=args.n_keys, n_values=4, n_hops=3
        )
        for arm in build_arms(smoke, model, args.ctx):
            arm_chunk = args.chunk if arm.get("chunkable", True) else 0
            hit, ratio, frac = retrieve(
                model, tok, arm, hay, query, targets, args.device, arm_chunk, n, h_kv, 40
            )
            if arm["kind"] == "full" and not hit:
                full_ok = False
            rows.append(
                {
                    "task": task,
                    "arm": arm["name"],
                    "hit": hit,
                    "recall": round(frac, 3),
                    "ratio": round(ratio, 4),
                }
            )
            print(f"   [{task}] {arm['name']:22s} hit={hit} recall={frac:.2f} ratio={ratio:.4f}")
    verdict = "PASS" if full_ok else "FAIL (KILL -- no GPU)"
    print(f"[G-SMOKE] bar: 'full' recovers the needle (acc==1.0) | verdict: {verdict}")
    return {"gate": "G-SMOKE", "passed": full_ok, "rows": rows}


def _to_float(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def probe(args: argparse.Namespace) -> dict[str, Any]:
    tok, model, gate = _load(args.model, args.device, args.dtype)
    with torch.no_grad():
        rope = _g_rope(model)
        bias = _g_bias(model)
        tail = _g_tail(tok, args)
        smoke = _g_smoke(model, tok, args)
    fund = bool(tail["passed"] and rope["passed"] and smoke["passed"])
    result = {
        "model": args.model,
        "ctx": args.ctx,
        "gates": [gate, rope, bias, tail, smoke],
        "fund": fund,
    }
    print(f"{JSON_BEGIN}{json.dumps(result)}{JSON_END}")
    print(
        f"\n=== TIER-2 GATE [{args.model}]: "
        f"{'FUND -- authorize GPU' if fund else 'KILL -- fix before any pod'} ==="
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--n-keys", type=int, default=8)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--out-json", default="")
    args = p.parse_args()
    result = probe(args)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
