"""Week-9 D1: BUG as eviction's constant-memory recovery tier -- recall probe.

D1 asks whether a BUG low-rank sketch of the tokens an eviction cache *drops* can
recover content the eviction cache has *forgotten*, judged on the axis where the
weakness lives: **recall of evicted-then-queried content** (a needle placed in the
region eviction prunes), NOT the settled extreme-budget mean ppl.

This harness drives the *streaming* decode caches (``MorphKVCache`` and, once
built, the per-head ``HybridRecoveryCache``) -- unlike ``w4_needle``/``w5_needle``
which drive kvpress *prefill* presses on a plain ``DynamicCache``. The streaming
caches raise on ``q_len != 1`` after the single-shot prefill, so the question and
answer are fed **one token per forward** with explicit **true** ``position_ids``
(the eviction fairness fix, ``w4_needle.retrieve``).

GO/NO-GO gate (run FIRST, this file's default). Before building any hybrid, prove
the probe isolates *forgetting*: at an aggressive MorphKV capacity and a
mid-context needle depth, does the **full** cache retrieve it (evidence the model
*can*, so the task is well-posed) while **pure morph** *fails* (evidence it
dropped the needle)? Only a (full-high, morph-low) split means the recovery tier
has something real to recover. If pure morph already retrieves it, the probe is
not isolating forgetting -- deepen the needle / shrink capacity, or report null.

Infra mirrors ``w5_needle`` (each ctx in try/except, model loaded once, JSON
between markers for ``vastai logs``). ``--plot-only`` rebuilds the figure.

Example (CPU smoke, 1B -- the isolation gate)
---------------------------------------------
    uv run python scripts/w9_recovery.py --context-lens 2048 \
        --depths 0.3 0.5 --passcodes 48213 --morph-capacity 224
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import matplotlib
import torch
from perplexity_sweep import load_model
from transformers.cache_utils import Cache, DynamicCache
from w4_needle import build_haystack
from w5_ruler import build_multikey_haystack
from w5_streamppl import N_SINK
from w7_rank_sweep import coord_for_config

from kvdlra.cache import BugStreamingCache, MorphKVCache

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
JSON_BEGIN = "===W9_RECOVERY_JSON_BEGIN==="
JSON_END = "===W9_RECOVERY_JSON_END==="

_STREAMING = (MorphKVCache, BugStreamingCache)


@torch.no_grad()
def retrieve_streaming(
    model: Any,
    tokenizer: Any,
    haystack_ids: torch.Tensor,
    query_ids: torch.Tensor,
    target: str,
    cache: Cache,
    device: str,
    max_new: int = 12,
) -> bool:
    """Single-shot prefill (triggers eviction's prune) then one-token-per-forward
    decode of the query + answer at TRUE positions; True if ``target`` decoded.

    The streaming caches bound only what is *retained* for decode, so the query
    and answer tokens are fed individually (``q_len=1``) -- feeding the 48-token
    query as one block would raise (chunked continuation is unsupported)."""
    hay = haystack_ids.to(device)
    ctx_len = int(hay.shape[1])
    attach = cache.attach(model) if isinstance(cache, _STREAMING) else nullcontext()
    with attach:
        model(hay, past_key_values=cache, use_cache=True)  # prefill -> prune
        # Feed the (teacher-forced) query tokens one at a time at true positions.
        pos = ctx_len
        for j in range(int(query_ids.shape[1])):
            tok = query_ids[:, j : j + 1].to(device)
            position_ids = torch.arange(pos, pos + 1, device=device).unsqueeze(0)
            out = model(tok, past_key_values=cache, use_cache=True, position_ids=position_ids)
            pos += 1
        # Greedy-decode the answer, one token per forward at true positions.
        generated: list[int] = []
        nxt = int(out.logits[0, -1].argmax())
        for _ in range(max_new):
            generated.append(nxt)
            cur = torch.tensor([[nxt]], device=device)
            position_ids = torch.arange(pos, pos + 1, device=device).unsqueeze(0)
            out = model(cur, past_key_values=cache, use_cache=True, position_ids=position_ids)
            pos += 1
            nxt = int(out.logits[0, -1].argmax())
    return target in tokenizer.decode(generated)


def build_caches(model: Any, args: argparse.Namespace) -> dict[str, Cache | None]:
    """Recall-probe methods, all at (roughly) MATCHED memory to pure morph:

    * ``full`` -- the O(T) upper bound (proves the task is well-posed);
    * ``morph`` -- pure whole-token eviction (the *forgetting* baseline that drops
      the mid-context needle);
    * ``bugA`` -- a pure BUG low-rank summary of ALL history (retention="attn"):
      the constant-memory *recovery tier* in its purest form -- does its low-rank
      reconstruction of the dropped stream recover the needle?
    * ``slash`` -- SLASH (exact heavy-hitters + BUG tail): the shared-eviction
      realization of D1 -- eviction-like exact retention PLUS a BUG recovery
      summary of the rest.

    Every BUG arm's coordinate budget is solved to pure morph's per-layer stored
    floats (matched memory) via ``coord_for_config``; the per-head, MorphKV-
    *preserving* ``HybridRecoveryCache`` is the purest form and is noted as future
    work (the GQA per-head crux, ``docs/week9.md`` D1)."""
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    cap, mr = args.morph_capacity, args.morph_recent
    # Pure morph's honest per-layer footprint: kept K/V (2n each) + score buffer
    # (h_kv * mr rows over the kept length) -- the budget every BUG arm matches.
    budget = (cap + mr) * 2 * n + h_kv * mr * (cap + mr)
    rw, ab, r = args.bug_recent, args.bug_absorb, args.bug_rank
    w_bug = coord_for_config(budget, n, r, rw, ab, "attn")
    w_slash = coord_for_config(budget, n, r, rw, ab, "attn", hh_budget=args.slash_hh)
    w_surp = coord_for_config(budget, n, r, rw, ab, "lowrank_surprise")
    caches: dict[str, Cache | None] = {
        "full": None,
        f"morph-C{cap}-R{mr}": MorphKVCache(model, capacity=cap, recent_window=mr),
        f"bugA-r{r}-W{w_bug}": BugStreamingCache(
            model,
            rank=r,
            coord_budget=w_bug,
            recent_window=rw,
            absorb_block=ab,
            n_sink=N_SINK,
            retention="attn",
        ),
        # D3 cross-check: surprise retention keeps the outlier needle's coordinate;
        # tests whether "keep outliers" aids RECALL where attn retention does not.
        f"bugS-r{r}-W{w_surp}": BugStreamingCache(
            model,
            rank=r,
            coord_budget=w_surp,
            recent_window=rw,
            absorb_block=ab,
            n_sink=N_SINK,
            retention="lowrank_surprise",
        ),
        f"slash-r{r}-h{args.slash_hh}-W{w_slash}": BugStreamingCache(
            model,
            rank=r,
            coord_budget=w_slash,
            recent_window=rw,
            absorb_block=ab,
            n_sink=N_SINK,
            retention="attn",
            hh_budget=args.slash_hh,
        ),
    }
    # Optional +premium recovery arm (D1: a small memory premium for the net).
    if args.premium_rank:
        pr = args.premium_rank
        w_prem = coord_for_config(int(budget * args.premium_frac), n, pr, rw, ab, "attn")
        caches[f"bugA+prem-r{pr}-W{w_prem}"] = BugStreamingCache(
            model,
            rank=pr,
            coord_budget=w_prem,
            recent_window=rw,
            absorb_block=ab,
            n_sink=N_SINK,
            retention="attn",
        )
    # D1 fidelity-isolation (diagnostic, NOT matched-memory): coord_budget so large
    # NO coordinate is ever evicted, so the needle is *provably retained* as a
    # rank-r coordinate. Separates "low-rank can't reconstruct" (fidelity ceiling)
    # from "the needle was evicted before reconstruction" (the coordinate-eviction
    # confound the D1 skeptic flagged). Swept over ranks to test if higher rank
    # rescues fidelity.
    for fr in [int(x) for x in args.fidelity_ranks.split(",") if x]:
        caches[f"bugFID-r{fr}-Wall"] = BugStreamingCache(
            model,
            rank=fr,
            coord_budget=args.fidelity_coord,
            recent_window=rw,
            absorb_block=ab,
            n_sink=N_SINK,
            retention="fifo",
        )
    return caches


def run_one_ctx(model: Any, tokenizer: Any, ctx: int, args: argparse.Namespace) -> dict[str, Any]:
    hits: dict[str, int] = {}
    total: dict[str, int] = {}
    mem: dict[str, int] = {}
    order: list[str] = []
    trials = [(pc, d) for pc in args.passcodes for d in args.depths]
    for passcode, depth in trials:
        if args.multikey:
            hay, q, target = build_multikey_haystack(
                tokenizer, ctx, args.n_keys, int(depth), int(passcode)
            )
        else:
            hay, q, target = build_haystack(tokenizer, ctx, depth, passcode)
        for name, cache in build_caches(model, args).items():  # fresh per trial (stateful)
            if name not in order:
                order.append(name)
                hits[name] = total[name] = 0
            cache_obj: Cache = cache if cache is not None else DynamicCache()
            ok = retrieve_streaming(model, tokenizer, hay, q, target, cache_obj, args.device)
            hits[name] += int(ok)
            total[name] += 1
            if name not in mem and isinstance(cache_obj, _STREAMING):
                mem[name] = int(cache_obj.stored_state_numel())
    rows: list[dict[str, Any]] = []
    for name in order:
        acc = hits[name] / total[name] if total[name] else 0.0
        rows.append(
            {
                "method": name,
                "ctx": ctx,
                "accuracy": acc,
                "hits": hits[name],
                "total": total[name],
                "mem_stored": mem.get(name),
            }
        )
        memstr = f" mem={mem[name]}" if name in mem else ""
        print(
            f"[ctx {ctx}] {name:28s} recall acc={acc:.2f} ({hits[name]}/{total[name]}){memstr}",
            flush=True,
        )
    return {"ctx": ctx, "rows": rows}


def gate_verdict(per_ctx: list[dict[str, Any]]) -> dict[str, Any]:
    """Isolation gate + recall verdict. Isolation: full retrieves (well-posed) AND
    morph fails (forgot). Recall win: a BUG recovery arm (bugA / slash) beats pure
    morph -- i.e. its low-rank summary of the dropped stream recovers content
    whole-token eviction lost, at matched memory."""
    out = []
    for rec in per_ctx:
        rows = {r["method"]: r["accuracy"] for r in rec["rows"]}
        full = rows.get("full", 0.0)
        morph = max((v for k, v in rows.items() if k.startswith("morph")), default=0.0)
        bug = max((v for k, v in rows.items() if k.startswith("bugA")), default=0.0)
        slash = max((v for k, v in rows.items() if k.startswith("slash")), default=0.0)
        # every non-morph, non-full arm is a BUG recovery variant (bugA/bugS/slash/premium).
        recovery = [v for k, v in rows.items() if k not in ("full",) and not k.startswith("morph")]
        best_recovery = max(recovery, default=0.0)
        out.append(
            {
                "ctx": rec["ctx"],
                "full_acc": full,
                "morph_acc": morph,
                "bugA_acc": bug,
                "slash_acc": slash,
                "isolates_forgetting": bool(full >= 0.5 and morph <= 0.5 and full - morph >= 0.3),
                "recovery_beats_morph": bool(best_recovery - morph >= 0.3),
            }
        )
    return {
        "per_ctx": out,
        "any_isolates": any(r["isolates_forgetting"] for r in out),
        "any_recovery_win": any(r["recovery_beats_morph"] for r in out),
    }


def _plot(per_ctx: list[dict[str, Any]], out_path: Path) -> None:
    records = [r for rec in per_ctx for r in rec["rows"]]
    ctxs = sorted({int(r["ctx"]) for r in records})
    methods = sorted({r["method"] for r in records})
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in methods:
        xs, ys = [], []
        for c in ctxs:
            hit = [r for r in records if r["method"] == name and int(r["ctx"]) == c]
            if hit:
                xs.append(c)
                ys.append(float(hit[0]["accuracy"]))
        if xs:
            ax.plot(xs, ys, marker="o", lw=1.9, ms=7, label=name)
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("context length T")
    ax.set_ylabel("recall of evicted-then-queried needle")
    ax.set_title("D1 recovery gate: full vs pure-morph on dropped-content recall")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"[wrote {out_path}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--context-lens", type=int, nargs="+", default=[2048])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--passcodes", nargs="+", default=["48213", "70561"])
    parser.add_argument("--morph-capacity", type=int, default=192)
    parser.add_argument("--morph-recent", type=int, default=32)
    parser.add_argument("--bug-rank", type=int, default=64, help="BUG recovery-tier rank")
    parser.add_argument("--bug-recent", type=int, default=32)
    parser.add_argument("--bug-absorb", type=int, default=16)
    parser.add_argument("--slash-hh", type=int, default=32, help="SLASH exact heavy-hitter tier")
    parser.add_argument("--premium-rank", type=int, default=0, help="+premium arm BUG rank (0=off)")
    parser.add_argument("--premium-frac", type=float, default=1.2, help="+premium memory factor")
    parser.add_argument(
        "--fidelity-ranks", default="", help="diagnostic no-eviction bugA ranks, e.g. 64,256"
    )
    parser.add_argument("--fidelity-coord", type=int, default=4096, help="no-eviction coord budget")
    parser.add_argument("--multikey", action="store_true", help="RULER multi-key variant")
    parser.add_argument("--n-keys", type=int, default=8)
    parser.add_argument("--out-json", default="results/w9-recovery-gate-1b.json")
    parser.add_argument("--out-fig", default="figures/week9/recovery_gate_1b.png")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    out_fig = Path(args.out_fig)
    if args.plot_only:
        blob = json.loads(Path(args.out_json).read_text())
        _plot(blob["per_ctx"], out_fig)
        return

    model, tokenizer = load_model(args.model, args.device, args.dtype)
    print(f"model={args.model} ctxs={args.context_lens} morph_C={args.morph_capacity}", flush=True)

    per_ctx: list[dict[str, Any]] = []
    for ctx in args.context_lens:
        try:
            per_ctx.append(run_one_ctx(model, tokenizer, ctx, args))
        except RuntimeError as exc:
            print(f"[ctx {ctx}] FAILED ({type(exc).__name__}: {exc}); keeping prior", flush=True)
            break

    blob = {
        "model": args.model,
        "depths": args.depths,
        "passcodes": args.passcodes,
        "morph_capacity": args.morph_capacity,
        "multikey": bool(args.multikey),
        "per_ctx": per_ctx,
        "gate": gate_verdict(per_ctx),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(blob, indent=2) + "\n")
    if per_ctx:
        _plot(per_ctx, out_fig)
    print(JSON_BEGIN)
    print(json.dumps(blob))
    print(JSON_END)
    print(f"[wrote {out_json}] gate={blob['gate']['any_isolates']}", flush=True)


if __name__ == "__main__":
    main()
