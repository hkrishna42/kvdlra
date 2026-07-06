"""Week-7 downstream eval: needle retention through long constant-memory DECODE.

The perplexity result (``docs/week7.md``) shows attention-scored coordinate
retention (``bugA``) beats FIFO retention (``bug``) and edges eviction at the
moderate budget. This asks whether that translates to a *capability*: can the
cache still **retrieve a planted fact** after a long generation has pushed it
far past the memory budget?

Protocol (decode-time, the streaming counterpart to ``w5_needle``'s prefill test)
---------------------------------------------------------------------------------
1. Pre-fill a prompt = instructions + a short haystack with one needle
   ("The secret passcode is NNNNN.") at ``depth``. The needle lands in the
   cache's compressed middle. During pre-fill the prompt's own attention scores
   the needle (``bugA`` seeds retention from those rows), so a distinctive fact
   the prompt attended starts with a *higher* retention score than the bland
   filler that follows -- the honest mechanism by which attention-scored
   retention can outlast FIFO **without** re-referencing the needle.
2. **Stream** ``g_tokens`` of fixed distractor text one token per forward through
   the decode cache (exactly the ``w5_streamppl`` regime). This is what ages the
   needle: under a bounded cache the needle's page is eventually evicted (FIFO)
   or retained (attention-scored) or approximated (BUG low-rank). Longer streams
   push the needle further past the budget -- the x-axis of the headline curve.
3. Append the query ("What is the secret passcode?") as decode steps and greedy-
   decode the answer. A hit = the passcode string appears in the answer.

Every method is solved to the **same** per-layer float budget (the
``w5_streamppl`` matched-memory solver): ``full`` (O(T) upper bound), ``bug``
(FIFO), ``bugA`` (attention retention), ``morph`` (eviction), ``sllm`` (naive
window). Accuracy vs stream length is the degradation curve; the honest question
is whether ``bugA`` sits above ``bug`` (retention helps) and how it compares to
eviction and to the un-bounded ``full`` ceiling.

Determinism: the distractor is fixed teacher-forced text (no sampling in the
stream), so every method sees the identical token trajectory; only the answer
generation is greedy. Results JSON between ``===W7_NEEDLE_JSON_BEGIN/END===``
markers ([[vastai-pod-flakiness-jul2026]]). ``--plot-only`` rebuilds the figure.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401  # bootstrap: make kvdlra importable when run as a script
import matplotlib
import torch
from perplexity_sweep import load_corpus_ids, load_model
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from w5_streamppl import bug_budget_floats, build_methods

from kvdlra.cache import BugStreamingCache, MorphKVCache
from kvdlra.utils.seed import seed_everything

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
JSON_BEGIN = "===W7_NEEDLE_JSON_BEGIN==="
JSON_END = "===W7_NEEDLE_JSON_END==="
N_SINK = 4


def build_prompt(
    tok: Any, haystack_len: int, depth: float, passcode: str, filler_ids: torch.Tensor
) -> torch.Tensor:
    """Chat-templated prefill = instructions + haystack (needle at ``depth``).

    The haystack is real corpus text (so the needle is the only distinctive
    fact); the needle sentence is spliced in at ``depth`` of the haystack.
    """
    needle = f" The secret passcode is {passcode}. "
    n_needle = int(filler_ids.shape[0])  # reuse filler as the haystack body
    body = filler_ids[:haystack_len]
    cut = int(depth * body.shape[0])
    needle_ids = tok(needle, return_tensors="pt", add_special_tokens=False).input_ids[0]
    hay = torch.cat([body[:cut], needle_ids, body[cut:]])
    hay_text = tok.decode(hay)
    msgs = [
        {
            "role": "user",
            "content": (
                "Read the following passage and remember the secret passcode "
                "stated in it.\n\n" + hay_text
            ),
        }
    ]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    assert n_needle >= haystack_len, "filler pool shorter than requested haystack"
    return cast(torch.Tensor, enc["input_ids"][0])


def _feed(model: PreTrainedModel, cache: Cache, ids: torch.Tensor, device: str) -> torch.Tensor:
    """Teacher-force ``ids`` through the cache one token per forward; return the
    final-step logits (for greedy continuation)."""
    out = None
    for t in range(ids.shape[0]):
        out = model(ids[t : t + 1].unsqueeze(0).to(device), past_key_values=cache, use_cache=True)
    assert out is not None
    return cast(torch.Tensor, out.logits[0, -1])


def retrieve_after_stream(
    model: PreTrainedModel,
    tok: Any,
    cache: Cache,
    prefill_ids: torch.Tensor,
    stream_ids: torch.Tensor,
    query_ids: torch.Tensor,
    passcode: str,
    device: str,
    max_new: int = 12,
) -> bool:
    """Pre-fill, stream the distractor, feed the query, greedy-decode, hit-test."""
    attach = (
        cache.attach(model)
        if isinstance(cache, MorphKVCache | BugStreamingCache)
        else nullcontext()
    )
    with torch.no_grad(), attach:
        model(prefill_ids.unsqueeze(0).to(device), past_key_values=cache, use_cache=True)
        if stream_ids.shape[0] > 0:
            _feed(model, cache, stream_ids, device)
        logits = _feed(model, cache, query_ids, device)
        out_ids: list[int] = []
        for _ in range(max_new):
            nxt = int(logits.argmax())
            out_ids.append(nxt)
            step = model(
                torch.tensor([[nxt]], device=device), past_key_values=cache, use_cache=True
            )
            logits = step.logits[0, -1]
    answer = tok.decode(out_ids)
    return passcode in answer


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = int(head_dim * cfg.num_key_value_heads)
    h_kv = int(cfg.num_key_value_heads)
    corpus = load_corpus_ids(tokenizer, args.device, corpus=args.corpus)

    tier = {
        "rank": args.rank,
        "coord_budget": args.coord_budget,
        "recent_window": args.recent_window,
        "absorb_block": args.absorb_block,
        "morph_recent": args.morph_recent,
    }
    budget = bug_budget_floats(
        n, args.rank, args.coord_budget, args.recent_window, args.absorb_block
    )
    query_ids = tokenizer(
        "\n\nWhat is the secret passcode? The secret passcode is",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]

    results: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "haystack_len": args.haystack_len,
        "depth": args.depth,
        "passcodes": args.passcodes,
        "stream_lens": args.stream_lens,
        "budget_token_equivalents": budget / (2 * n),
        "methods": args.methods.split(","),
        "rows": [],
    }
    # A fixed distractor pool + haystack filler, disjoint slices of the corpus.
    filler = corpus[200_000 : 200_000 + args.haystack_len + 8]
    distractor_pool = corpus[300_000 : 300_000 + max(args.stream_lens) + 8]

    for g in args.stream_lens:
        stream_ids = distractor_pool[:g]
        for name_tmpl in results["methods"]:
            hits = 0
            for pc in args.passcodes:
                prefill = build_prompt(tokenizer, args.haystack_len, args.depth, pc, filler)
                methods = build_methods(
                    model, n, h_kv, tier, methods=[name_tmpl], score_decay=args.score_decay
                )
                cache = next(iter(methods.values()))
                cache_obj: Cache = cache if cache is not None else DynamicCache()
                ok = retrieve_after_stream(
                    model, tokenizer, cache_obj, prefill, stream_ids, query_ids, pc, args.device
                )
                hits += int(ok)
            acc = hits / len(args.passcodes)
            results["rows"].append({"method": name_tmpl, "stream_len": g, "acc": acc, "hits": hits})
            n_pc = len(args.passcodes)
            print(f"[g={g}] {name_tmpl:8s} acc={acc:.2f} ({hits}/{n_pc})", flush=True)
    return results


def make_figure(results: dict[str, Any], fig_path: Path) -> None:
    rows = results["rows"]
    methods = results["methods"]
    gs = sorted({r["stream_len"] for r in rows})
    style = {
        "full": ("tab:blue", "o"),
        "bug": ("tab:brown", "s"),
        "bugA": ("tab:orange", "D"),
        "morph": ("tab:green", "^"),
        "snapkvD": ("tab:red", "v"),
        "sllm": ("tab:gray", "x"),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        ys = [
            next(
                (r["acc"] for r in rows if r["method"] == m and r["stream_len"] == g),
                float("nan"),
            )
            for g in gs
        ]
        c, mk = style.get(m, ("black", "o"))
        ax.plot(gs, ys, marker=mk, color=c, lw=1.9, ms=7, label=m)
    ax.set_xlabel("distractor stream length before query (decode steps)")
    ax.set_ylabel("needle retrieval accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.axvline(results["budget_token_equivalents"], color="k", ls=":", lw=1, label="~1x budget")
    ax.set_title(
        f"decode-time needle retention -- {results['model'].split('/')[-1]}, "
        f"budget ~{results['budget_token_equivalents']:.0f} tok-eq"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"[wrote {fig_path}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--corpus", default="wikitext-103")
    parser.add_argument("--haystack-len", type=int, default=256)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--passcodes", nargs="+", default=["48213", "70561", "91357", "26048"])
    parser.add_argument("--stream-lens", type=int, nargs="+", default=[0, 256, 512, 1024, 2048])
    parser.add_argument("--methods", default="full,bug,bugA,morph,sllm")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--coord-budget", type=int, default=1024)
    parser.add_argument("--recent-window", type=int, default=64)
    parser.add_argument("--absorb-block", type=int, default=32)
    parser.add_argument("--morph-recent", type=int, default=32)
    parser.add_argument("--score-decay", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default="results/w7-decode-needle-1b.json")
    parser.add_argument("--fig", default="figures/week7/decode_needle_1b.png")
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
