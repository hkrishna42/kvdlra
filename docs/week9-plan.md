# Week 9 plan — BUG as an *aid* to eviction (three complementary hybrids)

> **Decision (post-Week-8).** The competition question is settled: at matched
> memory and **mean ppl at the extreme budget**, BUG does not beat eviction, and
> no BUG-flavored technique does — SLASH (+0.47), rank↔coverage water-filling,
> Frequent-Directions swap, merge, and now the CodeBUG codebook fork are all
> bounded by two *measured* walls (`docs/week7-dominance.md`, `docs/week8.md`).
> **Pivot: stop competing, start complementing.** BUG and eviction have
> **orthogonal failure modes** — eviction fails by *permanent forgetting* (a
> dropped token needed later is gone), BUG fails by *rank squeeze* at the deep
> tail. This week tests whether BUG can *aid* existing eviction methods by
> covering their weakness, and — non-negotiably — **evaluates on the axis where
> that weakness lives (retrieval / long-context recall / the frontier envelope),
> NOT the extreme-budget mean ppl that is already settled.**

## The evidence that motivates the pivot (measured, not assumed)

Per-bin ppl at the aggressive budget (1B, `results/w8-codebook-1b-doc{0,1}.json`):

- BUG (`bugA`) *edges* morph in the **middle** bins (doc0 bins 5–6: 14.5/14.9 vs
  15.5/16.5) but morph wins the **deep tail** (doc0 bin 7: morph 19.7 vs bugA
  28.1 — the rank squeeze) — a real, if narrow, **complementarity**.
- The naive "BUG smooths eviction's variance" pitch is **refuted**: BUG is
  *spikier* than eviction (stdev of ppl/full: bugA 1.34 vs morph 1.23 doc0; 0.48
  vs 0.20 doc1). So a hybrid must target *forgetting/recall*, not variance.
- SLASH (exact heavy-hitters + BUG residual) is the matched-memory "BUG+eviction"
  hybrid and is **bounded** (+0.47 aggressive, +0.06 moderate). So the mean-ppl
  matched-memory axis has no headroom — the win, if any, is on a different axis.

**BUG's proven edges (the Week-4 honest standing) are where a hybrid can live:**
near-oracle low-rank *summary* of all history, constant memory, **needle
retrieval ties ExpectedAttention (15/15) and beats SnapKV (3/15)**, extreme-
compression robustness, streaming. Eviction has none of these for *dropped*
content. That is the seam to exploit.

## The three directions (all "BUG aids eviction"; evaluate on the right axis)

### D1 — BUG-as-eviction's-long-term-memory (RECOMMENDED, highest upside)

**Mechanism.** Run any eviction cache (`MorphKVCache`, or its SnapKV-style
`evict_interval` variant) *unchanged*; instead of *dropping* its evicted tokens,
route them into a constant-memory `BugStreamingCache` sketch. Attention sees
`[eviction's exact/recent tokens] + [BUG low-rank summary of the evicted stream]`.
BUG becomes eviction's **recovery tier** — a cheap net against forgetting that
never touches eviction's recency behaviour.

**Falsifiable target (the RIGHT metric).** On **recall of evicted-then-queried
content** — a needle placed in the region eviction drops, plus RULER multi-hop /
variable-tracking, at long context (8K–32K): does `morph+BUG-net` beat pure
`morph` (i) at **matched** memory (net comes out of morph's budget) and (ii) at a
**small premium** (+10–20% for the net)? Report both. Success = a clear retrieval
/ recall win at ≤ small premium, with mean ppl not regressing.

**Honest risk (do not skip).** Week-5 found BUG does **not** beat eviction on
long-context *ppl* or RULER (mixed; EA wins). So D1 only lives if the win is on
**recall of dropped-then-needed content specifically** — a metric axis not tested
this way. Design the eval to isolate that (place the needle *in the evicted
region*), or it is another honest negative. Count the BUG net's memory.

**Reuse.** `src/kvdlra/cache/{morph_cache,bug_cache}.py`; `scripts/w5_needle.py`,
`scripts/w5_ruler.py`, `scripts/w5_longctx.py` (retrieval/long-ctx harnesses);
`attach()` for scoring. New: a `HybridRecoveryCache` (or a `recovery_bug=` mode on
`MorphKVCache`) that captures evicted K/V and feeds `augmented_bug_step`.

### D2 — Adaptive-SLASH as "the envelope" (SAFEST, real, incremental)

**Mechanism.** SLASH (`hh_budget` + `rank` + `coord_budget` on
`BugStreamingCache`) already *contains* pure-eviction (`rank`→small, `hh` large)
and pure-BUG (`hh`=0) as special cases. A **budget-adaptive allocation rule**
picks `(hh, r, W)` per total budget so one parameterized cache traces the Pareto
**envelope** of {eviction, BUG}.

**Falsifiable target.** Across budgets from **moderate → extreme** (a cross-budget
sweep), adaptive-SLASH ppl ≤ `min(pure-morph, pure-bugA)` + ε at *every* budget
(the no-regret envelope claim), and — the upside bet — **strictly** beats both in
the **crossover region** where the regimes meet (an interior optimum). BUG already
*wins* the moderate budget (10.62 vs 11.81), so the envelope genuinely extends
eviction into a regime it underperforms.

**Honest risk.** SLASH gains are small; the crossover may only *match* the
envelope, not strictly beat it. Then the contribution is "one cache, no regret,"
which is real but incremental — report it straight.

**Reuse.** `scripts/w7_rank_sweep.py` (`--hh-budgets`, `coord_for_config`) already
does the matched-memory SLASH sweep at one budget — extend it to sweep the budget
axis and fit/verify the allocation rule; `docs/week7-dominance.md` §SLASH.

### D3 — BUG-informed eviction *scoring* (NOVEL, speculative, orthogonal)

**Mechanism.** Use BUG's low-rank reconstruction residual `‖k − U Uᵀ k‖` as an
eviction/retention signal: tokens the low-rank basis already predicts well are
*redundant* (safe to evict); high-residual tokens are *outliers/surprising*
(keep exact). Compose with any eviction policy — replace or blend the
attention-mass score with the low-rank-surprise score.

**Falsifiable target.** BUG-informed eviction beats vanilla attention-scored
eviction (Morph/SnapKV) at matched memory on ppl and/or retrieval — i.e. the
surprise signal is *complementary* to attention mass, not redundant.

**Honest risk.** Untested; the residual may correlate with attention score
(redundant → no gain) or mislead. Higher variance; cheapest to prototype (the
residual is already computable from `BugStreamingCache.u_k`). Good as the
exploratory third track.

## Non-negotiable guardrails (same bar as every prior week)

- **Evaluate on the axis where BUG's edge is** — retrieval / long-context recall /
  the frontier envelope. **Do NOT re-run extreme-budget matched-memory mean ppl
  as the headline** (settled; it will reproduce the negative and waste compute).
- **Count ALL memory**, honest `stored_state_numel`; matched-memory audit
  (`mem_max ≤ budget`) for the matched arms; for premium arms report the *exact*
  premium %. The BUG recovery tier / envelope allocation / surprise buffers are
  all counted.
- **Report negatives straight.** Every prior overclaim was caught and retracted;
  keep that bar. A clean "BUG doesn't aid eviction on axis X either" is a valid,
  map-completing result.
- 1B locally; confirm any winner at 8B on one vast.ai pod (credit ~$4.69; recipe
  `docs/week6.md` §Infra). ROTATE the HF + vast.ai keys (still pending).

## Deliverables

- D1: `HybridRecoveryCache` (or `recovery_bug=` on `MorphKVCache`) + tests +
  `scripts/w9_recovery.py` (evict-then-recall / needle-in-dropped-region + RULER).
- D2: `scripts/w9_envelope.py` (cross-budget SLASH sweep + allocation rule +
  envelope/crossover verdict).
- D3: a `retention="lowrank_surprise"` (or blended) mode + `scripts/w9_surprise.py`.
- `docs/week9.md` with a per-direction verdict (win / bounded / negative), the
  honest accounting, and the axis each was judged on. Update `handover.md` +
  auto-memory. The overarching framing: **BUG as a complement that extends
  eviction, not a competitor that replaces it.**

## Multi-agent harness

The kickoff prompt (delivered in chat) runs this as a phased, adversarially-
verified multi-agent build — a design panel per direction, sequential verified
implementation, benchmarking on the right axes, independent skeptics attacking any
apparent win (right metric? honest memory? generalizes?), then 8B/writeup.
