# Week 10 — plan: the definitive long-context frontier

> The *spec* for Week 10 is `docs/week10-kickoff.md`; this file is the **build plan**
> produced by the Phase-0 design panel (harness / memory-accounting / ShadowKV
> go-no-go → judge). Read with `docs/week9.md` (prior week) and `docs/week4.md`
> (the closest prior-art fair frontier). State lives in `handover.md`.

## Goal

The definitive frontier — **quality vs honestly-counted KV-cache memory** — for
**BUG at ranks {32, 64, 128, 256}** against **SnapKV, MorphKV, ShadowKV**
(ExpectedAttention optional 5th) at long context. Deliverables: (a) a
quality-vs-memory frontier figure with every method's curve; (b) a headline memory
table ("at a given quality, how much more/less space does each BUG rank need vs
each competitor?" and the inverse); (c) a plain-English `docs/week10-explained.md`.
OOM-proof; 1B first, 8B confirm. Report the ranking straight — win or lose.

> **RE-SCOPE (2026-07-13, user steer).** The **primary** long-context evaluation is
> **task accuracy on RULER + LongBench**, not perplexity. Perplexity on WikiText is
> a weak long-context signal and structurally *favours* BUG's global summary while
> under-testing eviction (built for retrieval); RULER + LongBench are the standard
> and **fairer to eviction**. So:
> - **RULER** (`scripts/w10_ruler.py`) — synthetic, length-controllable to 32K/64K.
>   Focused subset: `niah_single`, `niah_multikey`, `niah_multivalue`, `vt`
>   (variable-tracking). **Compress-then-query** (query held out) — eviction's weak
>   spot. Accuracy vs memory.
> - **LongBench** (`scripts/w10_longbench.py`) — realistic QA (`qasper`,
>   `multifieldqa_en`, `hotpotqa`, `2wikimqa`), token-F1. **Query in-prompt**
>   (compress-whole-prompt-then-generate) — *fairer to eviction*. Most tasks < 32K,
>   so LongBench carries the realistic-tasks story, RULER carries 64K.
> - **Perplexity** (`scripts/w10_frontier.py`, DONE at 1B ≤4096) is now the
>   **supporting** axis. Both new harnesses reuse its arms + `accounting.py` memory
>   + Phase-3 chunked ingest. Both are generation/accuracy → more GPU-dependent at
>   32K/64K (validate small on CPU, run at scale on GPU).

## The single protocol (one path for every method)

For each `(method, T, sample)`:
1. **Prefill** `T` tokens with compression active (chunked at long T — see OOM).
2. **Frozen continuation score**: one forward of a held-out `W`-token window
   (default 512; sweep 1024), `past_key_values` held frozen, explicit
   `position_ids = arange(T, T+W)`, teacher-forced CE on `window[1:]` vs
   `logits[:-1]` — byte-for-byte `perplexity_sweep.window_nll`, fed a compressed
   cache. **Same tokens, same true positions, for every arm.** This is the
   documented "compress-then-score" deviation of `w4_fair`/`perplexity_sweep`,
   now uniform across streaming caches and kvpress presses.
3. Record four numbers/point: **perplexity** (primary y), **stored float-equiv/
   layer** (primary x), **`torch.cuda.max_memory_allocated`** (OOM-safety,
   separate), **RULER multi-key accuracy** (secondary axis — the Week-9 regime).

## Resolved crux decisions (binding)

### Chunked-prefill OOM → gated cache modes
Add `_mode ∈ {"normal","ingest","score"}` to `BugStreamingLayer` + `MorphKVLayer`,
**default `"normal"`**, flipped only by `BugStreamingCache.ingesting()` /
`.frozen_scoring()` context managers (try/finally restore). The `q_len != 1` raise
still fires in `"normal"`, so w5/w9 callers and the **202-test suite are
byte-unchanged**.
- **`score`** (non-mutating): return `[retained | window]`; needed so a `q_len=W`
  scoring forward on BUG/Morph doesn't hit the raise.
- **`ingest`** (chunked prefill): append chunk to recent, advance
  `cumulative_length`, defer absorb to a per-chunk `consolidate()` reusing the
  existing absorb loop / block schedule. `get_mask_sizes` in both modes uses
  `kv_length = attended_length() + q_len`, `kv_offset = cumulative_length -
  attended_length()` — algebraically identical to MorphKV's existing formula.

**Chosen mechanism:** chunked `ingest` (option a) is the production path for long
T; per-token streaming (option b, ~20 min/arm at 64K) is kept only as the small-T
**correctness oracle**. The *first* frontier is banked with single-shot prefill +
`score` mode at moderate T (CPU 1B) where single-shot fits; `ingest` is added and
then validated to match single-shot within fp tolerance at small T.

**OOM discipline:** bf16 weights, bs=1, `sdpa` (assert never eager at T≥32K),
`logits_to_keep=1` during prefill (kills the ~16.8 GB `T×vocab` logits at 64K),
lm_head only on the ≤1024-token window. Per-arm isolation: `reset_peak_memory_
stats()` before; `del cache; gc.collect(); torch.cuda.empty_cache()` after; wrap
each arm in `try/except OutOfMemoryError → status:"OOM"` and continue. SnapKV/EA
run the identical chunk loop via `kvpress.ChunkPress(press, chunk_length=P)` on a
`DynamicCache` (native `q_len=W` scoring — no cache surgery for press arms).

**Full-cache reference:** run via the same chunked `DynamicCache` prefill inside
try/except; on 64K OOM mark `full: skipped` and use **BUG rank256** (least-
compressed arm) as the reference line, stated in the caption.

### Memory accounting → one source of truth (the core deliverable)
Build **`src/kvdlra/accounting.py`**. One `Footprint` breakdown (verbatim fp
elements, quant code *bits*, aux fp32-words, + a ShadowKV-only GPU/CPU split)
rendered into three numbers:
- **(i) float-equiv/layer** (fp32-word unit) — `verbatim + ceil(quant_bits)/32 +
  aux`, identical to `stored_state_numel` (primary x-axis).
- **(ii) ratio to full fp16 cache** — `bits(16)/(2·T·n·16)`, identical to
  `kv_memory_ratio` (deployment headline).
- **(iii) measured peak GPU** — `torch.cuda.max_memory_allocated`, reported
  **alongside** (never instead of) the deployable footprint; `None` (not faked) on
  CPU 1B and for the unimplemented ShadowKV ppl.

Per-arm formulas mirror the repo constants (FP16=16, N_SINK=4, diagonal core = r
not r², quant codes at bits/32, positions/scores/norms at 1 each):
`bug_footprint` (mirrors `bug_budget_floats`/`coord_for_config`), `evict_footprint`
(SnapKV/EA; pure-fp16 ratio == keep_frac exactly), `morph_footprint` (mirrors
`MorphKVLayer.stored_state_numel`, incl. the `evict_interval-1` overshoot),
`shadow_footprint` (§ShadowKV).

**Anti-drift invariant (the crux):** a parametrized test asserts
`bug_footprint(cfg).float_equiv() == BugStreamingCache.stored_state_numel()` on a
live cache (frontier configs: fifo, attn, attn+hh), and the same for MorphKV — so
the formula path (SnapKV/ShadowKV) and the measured path (BUG/Morph) can't drift.

**Matched-memory gate:** `assert_all_within(reports, budget, n_layers)` over
high-water `mem_max` (measured arms) and closed-form peak-at-T (formula arms) —
no "matched" label printed for an over-budget arm.

**Headline table:** iso-ppl (interpolate BUG's ratio at the competitor's best ppl
on BUG's monotone frontier → "at ppl ≈ P, BUG rank R uses X% more/less memory than
{method}"; if below BUG's measured range, emit "no BUG rank in {32..256} reaches
ppl P" — never extrapolate a fabricated rank) and iso-memory (both ppls at a
matched budget).

**Delegator consolidation is deferred to Phase 7** (regression-pin the digit-exact
Week-4/5/7/9 outputs first, then replace the 6 duplicated formulas). The `evict_
quant_memory` +FP16-norm overcount for *pure-fp16* eviction (~0.2%, currently
favors BUG) is **fixed inside `accounting.py` from day one** (pure-fp16 ratio ==
keep_frac); the old scripts are untouched until the pinned Phase-7 refactor, so no
published number changes now.

### ShadowKV → DEFER, 8h local spike, option-2-only
No `ShadowKVPress` exists in pinned kvpress (`CURPress`/`LeverageScorePress` are
whole-token eviction, **not** faithful stand-ins). The **official ByteDance repo
(option 3) is forbidden**: it needs a CUDA-kernel build + old transformers, and its
headline win is a **GPU-capacity/throughput axis that vanishes under this repo's
float-equivalent accounting** (CPU-offloaded V counts at 1 float/elem, same as
GPU) → its numbers are **not comparable** = a misreport. If integrated it must be
**option 2** (in-repo faithful core: landmark chunk-means + low-rank pre-RoPE K +
V→CPU + top-k chunk retrieval as a `ShadowKVCache(CacheLayerMixin)`). Full faithful
port ≈ 25–40h; the one novel risk is a **pre-attention query-aware selection hook**
with consistent `get_mask_sizes` (transformers 5.8 passes no query to `update()`;
MorphKV's hook is post-attention/observe-only).

**Time-box: 8h local go/no-go spike** on that hook + mask consistency (tiny Llama
config, no GPU). If not working end-to-end at 8h → **STOP**; ship the 3-method
frontier with ShadowKV as an explicit "not yet integrated" follow-up (optionally a
**formula-only** point on the *memory* axis — GPU+CPU total counting offloaded V,
plus separate GPU-only/CPU/bandwidth lines — but **no perplexity point**). A
partial honest frontier beats a stalled or non-comparable one. ShadowKV's
`stored_state_numel` MUST count the offloaded V; reporting it "free" is forbidden.

## Build sequence (tests green — pytest 202/1 + ruff + mypy — at every gate)

| # | Phase | GPU? |
|---|---|---|
| 1a | `accounting.py` core + anti-drift pin (dependency-free core deliverable) | no |
| 1b | `frozen_scoring` (`score`) mode on BUG + MorphKV | no |
| 2 | `w10_frontier.py` **first real frontier** — 3 methods, moderate T, single-shot prefill, CPU 1B | no |
| 3 | chunked `ingest` mode + `consolidate()` (unlock 32K/64K OOM-safe) + single-shot-equivalence oracle | no |
| 4 | RULER secondary axis + `attn`-retention curve | no |
| 5 | **GPU 1B full frontier at 32K/64K** — escalation-gated on credit | **yes** |
| 6 | ShadowKV 8h go/no-go spike (local, tiny config) | no |
| 7 | **8B confirmation on pod** + delegator consolidation + `week10{,-explained}.md` | **yes** |

Ordering principle (from the kickoff): real frontier EARLY with the 3 available
methods before any ShadowKV time; 1B local BEFORE any 8B/GPU pod run.

## Defaults chosen (from the panel's escalations)
- **Corpus:** prefer **PG19** (contiguous books → genuine long-range 32K/64K
  windows) over WikiText-103 (article boundaries). WikiText-103 is confirmed
  reachable here; PG19 reachability is tested in Phase 2, WikiText-103 is the
  fallback. Absolute ppl is corpus-relative; only within-corpus ordering is the
  claim.
- **BUG retention:** headline ppl frontier uses **`retention="fifo"`** (no
  per-chunk-seeding confound); **`retention="attn"`** is the recall/RULER
  secondary curve. Report both.
- **`evict_quant_memory` +FP16 fix:** corrected in `accounting.py` only; published
  scripts untouched until the pinned Phase-7 consolidation.

## Escalations for the user (do not block CPU phases 1a–4, 6)
1. **GPU credit ~$3.2 is near the $2 floor** — needed only for Phase 5 (1B
   32K/64K) and Phase 7 (8B). Top up before Phase 5, or confirm we cap scope to
   the CPU-feasible frontier + an honest "GPU run pending" note.
2. **vast.ai + HF keys unrotated** — rotate/create the key BEFORE launching the
   instance; onstart batch; git-clone from `origin/week7` (SSH unusable). Confirm
   before any pod run.
3. **GPU tier (24 / 40 / 80 GB)** for Phase 5/7 — sets whether the full-cache
   reference is attemptable at 32K/64K (8B 64K full-ref needs A100 80GB / H100).
