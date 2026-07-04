# Week 6 — Axis B: BUG as a decode-time streaming integrator (constant-memory generation)

> The capability Weeks 1–5 never measured: BUG used *as a streaming integrator
> during generation*, its core theoretical selling point. Design:
> `docs/notes/streaming-decode-design.md`. Plan context: `docs/week5-plan.md`
> §"Axis B"; the Week-5 scorecard (five honest negatives on eviction's turf) is
> in `docs/week5.md`. This file is the complete Axis-B record: what was built,
> the correctness proofs, the benchmark, and the verdict.

## The question (falsifiable)

Long autoregressive generation under a **constant-memory** KV cache: does a
rank-`r` *running low-rank summary* of the context preserve quality better
than a *constant-size set of adaptively retained whole tokens* (MorphKV,
ICML'25) at matched memory — and does BUG's σ_min-robust DLRA error bound make
it degrade more *gracefully* as generation runs far past the budget?

**Verdict up front (honest):** the capability works, is correct, and is
competitive — BUG **wins at 1B**, is **near-lossless before ~2× budget at both
scales** (better than every rival there), and always beats the naive window.
But the graceful-degradation hypothesis is **inverted**: past ~2–3× budget,
adaptive eviction holds a *flat* quality offset while BUG's penalty grows, so
MorphKV/SnapKV win the deep-horizon aggregate at 8B, decisively at tight
budgets. Mechanistically coherent, twice-reproduced, reported straight.

## What was built

**`BugStreamingCache`** (`src/kvdlra/cache/bug_cache.py`) — a transformers-5.8
`Cache` subclass. Per layer, all bounded: the 4 attention sinks + a recent
ring of `w` tokens **verbatim**; the middle as BUG state — orthonormal basis
`U` (n×r) tracked on exactly un-rotated **pre-RoPE** keys (the model's own
rotary module ⇒ bit-identical angles; the inverse divides by
`attention_scaling²`), square-root core `B`, and up to `W` per-token
**coordinate** columns (r floats/token instead of n). Every `absorb_block`
steps the oldest recents graduate through one augmented rank-adaptive BUG step
(`augmented_bug_step`, factored out of the validated blocked tracker — one
source of truth; the extraction also fixed the latent `rank+block>n`
degeneracy found in Week 4.5). Held coordinates are carried across basis
updates by `rot = U_newᵀU_old`; oldest coordinates are dropped at the cap.
That cap is the honest meaning of "constant memory": softmax attention needs
per-token state for whatever it attends to, so a bounded cache can only bound
the attended set — BUG's version holds a ~`n/r`× *longer but approximate*
history at the same budget. `rank=0` degenerates to StreamingLLM
(sinks+window): one implementation, two methods.

**`MorphKVCache`** (`src/kvdlra/cache/morph_cache.py`) — faithful-core MorphKV
(arXiv:2503.00979; kvpress 0.5.1 has none): R recent + top-C distant tokens,
re-ranked every decode step by sum/max fusion of the last R exactly-recomputed
attention rows, per KV head with GQA aggregation per the paper. Documented
deviations: the prompt is pruned once at prefill end; the score buffer's
memory is **counted** in its budget; `evict_interval>1` gives the paper's
coarse-grained variant, which doubles as the **SnapKV-style decode-eviction**
baseline. kvpress's own `DecodingPress` was **rejected** after a real bug:
under transformers 5.8 it rotates all buffered window queries with the current
single-token `position_embeddings` — a silent scoring bias.

**Correctness ladder** (25 new tests; suite 132 passed / 1 skipped, mypy-strict
+ ruff green): teacher-forced logit parity with `DynamicCache` when nothing is
truncated (MorphKV bitwise; BUG to fp tolerance — its middle transits an fp32
un-rotate/re-rotate round trip), mask-size == returned-length at every step
across absorb boundaries, stored memory constant over hundreds of steps, the
coordinate-carry invariant (`rot` really is `U_newᵀU_old`; last-absorbed
coords equal the direct projection), eviction mechanics on crafted scores, and
the `rank+block>n` clamp. 1B behavioral validation
(`scripts/w5_decode_validate.py`, `results/w5-decode-validate-1b.json`): exact
parity / graceful drift, flat measured memory, bounded latency — and a
qualitative preview: at ~equal memory StreamingLLM collapses into an
`assistantassistant…` loop at token 7 where BUG r=32 (with *less* memory)
stays coherent.

## Benchmark protocol (`scripts/w5_streamppl.py`)

**Streaming perplexity**: prefill P tokens, then feed the true continuation
**one token per forward through the decode path** for G tokens (exactly the
generation regime, deterministic, no judge), scoring each step against the
bounded cache. Per-position bins expose the degradation *slope* — the
graceful-degradation claim is a slope claim. Matched **worst-case stored
floats** per layer (BUG's budget solves MorphKV's capacity — score buffer
included — the SnapKV-decode capacity, and the StreamingLLM window; the solver
counts ring high-water marks). Full cache = O(T) upper bound. 1B: fp32 CPU,
WikiText-2 test, P=512, G=3072, 2 docs. 8B (`unsloth/Meta-Llama-3.1-8B
-Instruct`, bf16, RTX 6000 Ada): WikiText-103, P=1024, G=8192, 3 docs, two
budget tiers. ⚠️ Small-n caveat: 2–3 docs; doc-level variance is real (BUG won
doc 0 at 8B tier 1 but lost docs 1–2); read trends, not third decimals.

## Results

**Aggregate streaming perplexity** (geo-mean over docs; lower is better):

| scale / budget | full (O(T)) | BUG | MorphKV | SnapKV-dec | StreamingLLM |
|---|---:|---:|---:|---:|---:|
| 1B, ~515 tok-eq, G=3072 | 10.13 | **11.59** | 11.81 | 11.83 | 14.47 |
| 8B, ~499 tok-eq, G=8192 | 7.22 | 7.87 | **7.74** | **7.74** | 8.20 |
| 8B, ~183 tok-eq, G=8192 | 7.22 | 9.17 | **8.39** | 8.46 | 9.63 |

**The slope (the scientific content).** Ratio of each method's per-bin ppl to
the full cache's (1.0 = no penalty), geo-mean over docs:

- **1B**: BUG starts **1.03×** and drifts up to ~1.23× by G=3072 (6× budget);
  MorphKV/SnapKV start ~1.22× and stay **flat** (~1.10× late). BUG's early
  near-losslessness carries the aggregate → BUG wins at 1B.
- **8B tier 1**: same shape — BUG has the best early bins (**1.027** vs
  MorphKV 1.045), crosses over at ~2–3× budget (~1–1.5K generated), then sits
  ~1.08–1.13 while eviction holds ~1.05–1.07. Over G=16× budget the flat
  profile wins the aggregate by ~0.13 ppl.
- **8B tier 2 (aggressive)**: no early regime exists (bin 0 is already 8× past
  budget) and r=64's summary is too coarse — BUG behind everywhere
  (1.20–1.45× vs MorphKV's flat 1.11–1.21×), only ahead of the naive window.

**Memory and latency.** Stored floats measured flat for every bounded method
(BUG ring/coord sawtooth included; full cache linear, 604M floats by G=8192 at
8B vs BUG's 32.7M). Per-token p50 at 8B/GPU: full 14.8 ms, StreamingLLM 20.2,
**BUG 20.5**, SnapKV-decode 22.8, MorphKV 26.9 — BUG is the *fastest* adaptive
constant-memory method (MorphKV pays per-step scoring); per-token SVDs are
(r+b)² and hit no cusolver stall. BUG attends ~5× more positions per step than
eviction at the same memory (2116 vs 443) — the more-history-per-byte trade
made explicit.

**Reproduction.** The 8B tier-1 run executed twice on different pods/hosts
(the first pod's JSON was lost to vast.ai log truncation; method-level numbers
survived): max |Δppl| = **4×10⁻⁴** across all method×doc cells.

## Why BUG loses the deep horizon (mechanisms, for Week 7)

1. **FIFO coordinate eviction** — past `W`, oldest coords drop regardless of
   importance: a sliding window in coordinate space. BUG is adaptive in
   *subspace* but non-adaptive in *retention*; MorphKV is exactly the reverse,
   and retention is what the deep horizon rewards.
2. **Erosion by repeated projection** — each absorb applies `c ← rot·c`;
   surviving tokens' out-of-subspace components decay multiplicatively as the
   basis follows topic drift (`Π_j P_j k_s`).
3. **Fixed rank over growing history** — the plain information squeeze
   (dominant at tier 2).

The σ_min-robustness of the BUG step governs *tracking* error, not this
*retention* loss — the hypothesis conflated the two; the data separated them.

## Scorecard (Week 6)

| claim | verdict |
|---|---|
| decode-time streaming BUG correct + constant memory + bounded latency | ✅ proven (tests + measurement; twice-reproduced benchmark) |
| beats naive window (StreamingLLM) | ✅ everywhere tested |
| near-lossless before ~2× budget | ✅ best method in that regime at both scales |
| beats MorphKV at matched memory | ✅ 1B aggregate; ❌ 8B (−0.13 tier 1, −0.78 tier 2) |
| "degrades more gracefully than eviction" | ❌ **inverted** — eviction is flat, BUG's penalty grows past ~2–3× budget |
| fastest adaptive constant-memory method (per-token) | ✅ at 8B/GPU (20.5 vs 26.9 ms p50) |

**One sentence:** BUG's streaming decode is real, correct, cheap, and the best
constant-memory cache while generation stays within ~2× its budget — but its
FIFO coordinate retention and projection erosion hand the deep horizon to
adaptive eviction, and fixing exactly that (adaptive retention + quantized
aging in coordinate space) is Week 7 (`docs/week7-plan.md`).

## Infra notes (hard-won, this week)

- vast.ai CLI's "Balance" column reads a field that is 0 even when prepaid
  `credit` exists — read `credit` from `--raw` (a wrong $0 call briefly
  blocked this run).
- `pytorch/pytorch:2.11` images have PEP-668 externally-managed python: set
  `PIP_BREAK_SYSTEM_PACKAGES=1` and hard-fail the batch if imports are missing
  (`===DEPS_FAILED===`), else the job runs dep-less through every marker.
- vast.ai's log path truncates stdout lines at ~500 chars — emit results JSON
  as **base64 folded to short lines** between markers; never a one-line JSON.
- Total GPU spend for Week 6: ≈ $2.50 (one dud pod $0.03, two full runs).
  Detached auto-destroy watchers left nothing billing.
