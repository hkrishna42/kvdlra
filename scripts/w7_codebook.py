"""Week-8 CodeBUG benchmark: does an amortized PQ codebook beat eviction at the
extreme budget? (the single falsifiable number).

At the aggressive budget (~89 tok-eq/layer, 1B, WikiText-2, G=1024) eviction wins
(morph 18.71 vs best-BUG/SLASH 20.15; full 7.24). CodeBUG replaces the fp32
storage of the low-rank coordinate tail with codes into a shared, calibrated
product-quantization codebook expressed against a FROZEN reduced-rank anchor
(``docs/week8-codebook-plan.md``; carry design = anchor-basis freezing, the
unanimous Phase-0 panel choice). A coded token costs a few bits instead of ``r``
floats, so the same leftover budget buys far more (approximate) coverage.

The bet: many PQ-reconstructed tokens beat eviction's few exact tokens. This
script measures it at HONESTLY matched memory:

  * budget frozen at ``--tok-eq`` * 2n floats/layer (89 tok-eq = 91136);
  * morph / bugA / SLASH solved to that budget (reusing the w5/w7 solvers);
  * CodeBUG's coded coverage solved to that budget, counting the frozen anchor
    (``2n*r_a``), the codes (``2*M*bits/32``/token), norms, positions, scores,
    and the shared codebook ONCE (``codebook_numel``, ceil-shared per layer);
  * ``mem_max <= budget`` audited for every variant.

WIN = a CodeBUG variant's aggregate streaming ppl < morph (18.71) at matched
memory. Reported either way -- a clean "codebook doesn't close it" completes the
dominance map. Requires a calibrated codebook (``scripts/calibrate_codebook.py``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import matplotlib
import torch
from perplexity_sweep import load_corpus_ids, load_model
from transformers.cache_utils import Cache, DynamicCache
from w5_streamppl import N_SINK, morph_capacity_for_budget, stream_score, summarize
from w7_rank_sweep import coord_for_config

from kvdlra.cache import BugStreamingCache, MorphKVCache
from kvdlra.quant import ProductQuantizer
from kvdlra.utils.seed import seed_everything

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
JSON_BEGIN = "===W7_CODEBOOK_JSON_BEGIN==="
JSON_END = "===W7_CODEBOOK_JSON_END==="


def code_budget_for_config(
    budget: int,
    n: int,
    rank: int,
    anchor_rank: int,
    coord_budget: int,
    recent: int,
    absorb: int,
    pq_bits: int,
    pq_subspaces: int,
    normalize: bool,
    codebook_numel: int,
    n_layers: int,
) -> int:
    """Largest coded-tier coverage ``W_code`` at a FROZEN ``budget``, counting all
    CodeBUG memory honestly (mirrors ``BugStreamingLayer.stored_state_numel``):

    fixed = sinks + recent ring + live basis(2n*r) + diag core(2r) + frozen
    anchor(2n*r_a) + ring_score(ring, attn) + fp32 tier(coord_budget*(2r+2))
    + shared codebook (codebook_numel, ceil-shared over layers);
    per coded token = 2*M*bits/32 (K+V codes) + 2 norms + 1 pos + 1 score.
    """
    ring = recent + absorb - 1
    fixed = 2 * n * N_SINK + 2 * n * ring + 2 * n * rank + 2 * rank
    fixed += 2 * n * anchor_rank  # frozen anchor (K+V)
    fixed += ring  # ring_score buffer (attn)
    fixed += coord_budget * (2 * rank + 2)  # fp32 tier (attn: +pos +score)
    fixed += -(-codebook_numel // n_layers)  # shared codebook, charged once
    per_coded = 2.0 * pq_subspaces * pq_bits / 32.0  # K+V codes
    if normalize:
        per_coded += 2  # K+V norms
    per_coded += 2  # pos + score (attn)
    return max(1, int((budget - fixed) / per_coded))


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    n_layers = int(cfg.num_hidden_layers)
    corpus = load_corpus_ids(tokenizer, args.device, corpus=args.corpus)

    blob = torch.load(args.codebook, weights_only=False)
    pq = ProductQuantizer.from_state(blob["state"])
    anchor_rank = pq.dim
    normalize = pq.normalize

    budget = round(args.tok_eq * 2 * n)
    rw, ab = args.recent_window, args.absorb_block
    r = args.rank
    coord_budgets = [int(x) for x in args.coord_budgets.split(",")]

    # Baselines solved to the SAME frozen budget.
    morph_c = morph_capacity_for_budget(budget, n, h_kv, args.morph_recent)
    buga_w = coord_for_config(budget, n, r, rw, ab, "attn")
    slash_w = coord_for_config(budget, n, r, rw, ab, "attn", hh_budget=args.slash_hh)
    code_w = {
        cb: code_budget_for_config(
            budget,
            n,
            r,
            anchor_rank,
            cb,
            rw,
            ab,
            pq.bits,
            pq.subspaces,
            normalize,
            pq.codebook_numel(),
            n_layers,
        )
        for cb in coord_budgets
    }
    print(
        f"budget {budget} floats/layer (~{args.tok_eq:.0f} tok-eq); morph_C={morph_c}; "
        f"bugA W={buga_w}; slash-h{args.slash_hh} W={slash_w}; "
        f"CodeBUG r_a={anchor_rank} W_code={code_w}",
        flush=True,
    )

    results: dict[str, Any] = {
        "model": args.model,
        "corpus": args.corpus,
        "n_features": n,
        "n_layers": n_layers,
        "prefill": args.prefill,
        "g_tokens": args.g_tokens,
        "bin_size": args.bin_size,
        "budget_floats_per_layer": budget,
        "budget_token_equivalents": args.tok_eq,
        "rank": r,
        "codebook": str(args.codebook),
        "codebook_manifest": blob.get("manifest", {}),
        "anchor_rank": anchor_rank,
        "coord_budgets": coord_budgets,
        "docs": {},
    }

    def make_methods() -> dict[str, Cache | None]:
        m: dict[str, Cache | None] = {"full": None}
        m[f"morph-C{morph_c}"] = MorphKVCache(
            model, capacity=morph_c, recent_window=args.morph_recent
        )
        m[f"bugA-r{r}-W{buga_w}"] = BugStreamingCache(
            model,
            rank=r,
            coord_budget=buga_w,
            recent_window=rw,
            absorb_block=ab,
            n_sink=N_SINK,
            retention="attn",
            score_decay=args.score_decay,
        )
        if args.slash_hh > 0:
            m[f"slash-h{args.slash_hh}-W{slash_w}"] = BugStreamingCache(
                model,
                rank=r,
                coord_budget=slash_w,
                recent_window=rw,
                absorb_block=ab,
                n_sink=N_SINK,
                retention="attn",
                score_decay=args.score_decay,
                hh_budget=args.slash_hh,
            )
        for cb in coord_budgets:
            m[f"code-r{r}ra{anchor_rank}-cb{cb}-W{code_w[cb]}"] = BugStreamingCache(
                model,
                rank=r,
                coord_budget=cb,
                recent_window=rw,
                absorb_block=ab,
                n_sink=N_SINK,
                retention="attn",
                score_decay=args.score_decay,
                coord_codebook=pq,
                anchor_rank=anchor_rank,
                anchor_seal_absorbs=args.anchor_seal_absorbs,
                code_budget=code_w[cb],
            )
        return m

    span = args.prefill + args.g_tokens + 1
    for d in range(args.doc_start, args.doc_start + args.n_docs):
        start = args.doc_stride * d
        ids = corpus[start : start + span]
        if ids.shape[0] < span:
            print(f"[doc {d}] corpus exhausted", flush=True)
            break
        per_doc: dict[str, Any] = {}
        for name, cache in make_methods().items():
            cache_obj: Cache = cache if cache is not None else DynamicCache()
            rec = stream_score(model, ids, cache_obj, args.prefill, args.device)
            rec.update(summarize(rec.pop("nlls"), args.bin_size))
            over = rec["mem_max"] > budget * n_layers
            if name != "full" and over:
                print(
                    f"[doc {d}] WARN {name} OVER budget: {rec['mem_max']} > {budget * n_layers}",
                    flush=True,
                )
            rec["mem_over_budget"] = bool(name != "full" and over)
            per_doc[name] = rec
            print(f"[doc {d}] {name}: ppl={rec['ppl']:.3f} mem_max={rec['mem_max']}", flush=True)
        results["docs"][str(d)] = per_doc

    _verdict(results)
    return results


def _geo(vals: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _verdict(results: dict[str, Any]) -> None:
    docs = results["docs"]
    if not docs:
        return
    names = list(next(iter(docs.values())))
    aggs = {nm: _geo([docs[d][nm]["ppl"] for d in docs]) for nm in names}
    morph = next(nm for nm in names if nm.startswith("morph"))
    codes = [nm for nm in names if nm.startswith("code-")]
    best_code = min(codes, key=lambda nm: aggs[nm]) if codes else None
    results["aggregate_ppl"] = aggs
    beats = bool(best_code and aggs[best_code] < aggs[morph])
    results["verdict"] = {
        "morph": aggs[morph],
        "best_code": best_code,
        "best_code_ppl": aggs[best_code] if best_code else None,
        "beats_morph": beats,
    }
    bc_ppl = aggs[best_code] if best_code else float("nan")
    print(
        f"[VERDICT] morph={aggs[morph]:.3f}  best CodeBUG={best_code}={bc_ppl:.3f}  "
        f"beats_morph={beats}",
        flush=True,
    )


def make_figure(results: dict[str, Any], fig_path: Path) -> None:
    docs = results["docs"]
    if not docs:
        return
    names = list(next(iter(docs.values())))
    nb = min(len(docs[d][nm]["ppl_bins"]) for d in docs for nm in names)
    bs = results["bin_size"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for nm in names:
        if nm == "full":
            continue
        ys = [
            _geo([docs[d][nm]["ppl_bins"][i] / docs[d]["full"]["ppl_bins"][i] for d in docs])
            for i in range(nb)
        ]
        style = "k--" if nm.startswith("morph") else ("-" if nm.startswith("code-") else ":")
        ax.plot([(i + 1) * bs for i in range(nb)], ys, style, marker=".", label=nm)
    ax.set_xlabel("decode position")
    ax.set_ylabel("per-bin ppl ratio to full")
    ax.set_title(f"CodeBUG @ ~{results['budget_token_equivalents']:.0f} tok-eq (matched memory)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"[wrote {fig_path}]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--corpus", default="wikitext-2")
    p.add_argument("--codebook", default="results/w8-codebook.pt")
    p.add_argument("--prefill", type=int, default=512)
    p.add_argument("--g-tokens", type=int, default=1024)
    p.add_argument("--bin-size", type=int, default=128)
    p.add_argument("--n-docs", type=int, default=1)
    p.add_argument("--doc-start", type=int, default=0)
    p.add_argument("--doc-stride", type=int, default=100_000)
    p.add_argument("--tok-eq", type=float, default=89.0, help="frozen budget in token-equivalents")
    p.add_argument("--rank", type=int, default=24)
    p.add_argument("--coord-budgets", default="16,32,64", help="CodeBUG fp32-tier sizes to sweep")
    p.add_argument("--recent-window", type=int, default=32)
    p.add_argument("--absorb-block", type=int, default=16)
    p.add_argument("--morph-recent", type=int, default=32)
    p.add_argument("--slash-hh", type=int, default=4, help="SLASH exact-tier size for the baseline")
    p.add_argument("--anchor-seal-absorbs", type=int, default=4)
    p.add_argument("--score-decay", type=float, default=0.97)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default="results/w8-codebook-1b.json")
    p.add_argument("--fig", default="figures/week8/codebook_1b.png")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

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
