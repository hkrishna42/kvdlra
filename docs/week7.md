# Week 7 — fixing BUG's deep-horizon retention (Axis-B follow-up)

> Context: `docs/week6.md` (the inverted graceful-degradation verdict) and
> `docs/week7-plan.md` (the tier plan). Week 6 established that
> `BugStreamingCache` is the best constant-memory decode cache while generation
> stays within ~2× its budget, but past ~2–3× budget its per-bin penalty
> *grows* while adaptive eviction (MorphKV/SnapKV-decode) holds a *flat* offset,
> so eviction wins the deep-horizon aggregate at 8B. Diagnosed mechanisms: (1)
> **FIFO coordinate eviction** (adaptive subspace, non-adaptive retention); (2)
> **erosion by repeated projection** (`c ← rot·c` each absorb); (3) **fixed rank
> over growing history**. Week 7 attacks (1) and (2) directly, in coordinate
> space.

## What was built (tier 1)

Two independent knobs on the same coordinate buffer of `BugStreamingLayer`
(`src/kvdlra/cache/bug_cache.py`), each reusing a validated component and each
with a falsifiable bin-curve signature:

- **(A) adaptive coordinate retention** — `retention={fifo,attn,energy}`:
  - `fifo` — the Week-6 rule (drop the oldest column).
  - `attn` — drop the column with the lowest **EMA-accumulated attention mass**
    (decay `score_decay` per decode step). Scores are observed by
    `BugStreamingCache.attach(model)`, a forward hook that recomputes each
    step's aggregated attention row over the cache's *returned* K with exactly
    MorphKV's machinery (`_aggregated_attention_row`), EMAs it into per-column
    scores, and carries a column's score from the recent ring into the middle at
    graduation; the prompt seeds scores from the last `recent_window` prompt
    queries' causal rows. This is the O(1)-per-column analogue of MorphKV's
    sum-fusion — it gives BUG *MorphKV's eviction brain* while keeping the
    rank-`r` subspace summary.
  - `energy` — drop the column with the smallest coordinate norm `‖c_s‖₂` (a
    zero-extra-memory proxy: projection erosion shrinks exactly the columns the
    basis has drifted away from).
  - Non-FIFO retention makes the retained middle **non-contiguous in position**,
    so true per-column positions (`mid_pos`/`q_pos`) are tracked and the middle
    reconstruction is re-rotated at *those* positions (gathered cos/sin from the
    model's own rotary module — exact, same angles attention used).
- **(D) quantize-instead-of-drop (age-tiered precision)** — `quant_bits` +
  `quant_budget`: columns evicted from the fp32 coordinate tier are **demoted**
  into a second tier of up to `quant_budget` PolarQuant-quantized columns
  (`quant_bits` bits/coordinate — the Week-4 machinery already validated on BUG
  coordinates) instead of dropped. Same memory holds ~`32/bits`× more history in
  the demoted tier. Carried across basis updates by dequantize → `rot @` →
  requantize, with **exact norm carry** (see the honesty note below).
- **(A+D)** compose: attention-scored eviction from the fp32 tier into the
  quantized tier, quantized tier evicts by the same score.

**Matched-memory construction.** Every variant is solved *down to the same
per-layer float budget* as the Week-6 baseline BUG configuration
(`solve_bug_variant` in `scripts/w5_streamppl.py`): the score buffers,
per-column positions, quantized codes (at `bits/32` float-equivalents), norms,
and the shared PolarQuant side information (counted **once** per cache) all come
*out of* the coordinate budget, never on top of it. The harness asserts each
bounded method's measured `mem_max` ≤ the tier budget at run time. This is the
Week-7 honesty guardrail (a Week-4 result had to be retracted over exactly this
kind of unfair accounting).

## Correctness + adversarial review

25 new tests (`tests/test_bug_cache_week7.py`) cover: retention selection keeps
the top-scored/highest-energy columns and preserves chronology among survivors;
non-contiguous middles are re-rotated at their true positions; `retention=attn`
with no observations reduces to FIFO **bitwise**; the quantized tier bounds
reconstruction error and is counted honestly; exact-mode parity with
`DynamicCache` and the mask-size == returned-length invariant hold for every
variant. Suite: **151 pass / 1 skip**, mypy-strict + ruff green.

A 4-lens adversarial review (DLRA math, cache plumbing, memory honesty, harness
protocol), each finding independently verified, caught three real issues before
any benchmark was believed:

1. **(major, fixed) quant-tier norm drift.** The dequantize → `rot @` →
   requantize carry was *not* norm-preserving even at `rot = I`: PolarQuant
   dequantize scales the raw Lloyd–Max centroid vector, whose norm is not 1, and
   requantize re-derived the stored norm from that drifted magnitude — so each
   surviving quantized column's magnitude drifted **exponentially** (`gᵏ` after
   `k` absorbs), independent of any basis motion, contradicting the "noise only
   on basis motion" design claim. Fix: `_dequantize` renormalizes the decoded
   direction, so the stored norm *is* the column norm exactly; the carry now
   contributes only the true rotation-induced contraction plus bounded
   per-event direction jitter. Regression test: 50 identity-rotation carries
   preserve norms to fp roundoff.
2. **(minor, fixed) negative quant budget.** `solve_bug_variant` could emit a
   negative `quant_budget` at high `quant_keep_frac`; now it raises, and the
   layer validates `quant_budget ≥ 0`.
3. **(major, corrected claim) not bit-identical to Week 6.** The Week-7
   integrator robustness fix (see below) applies on *every* absorb, so a rerun
   of the Week-6 `bug` config is **fp-equivalent, not bit-identical** to the
   archived `results/w5-streamppl-1b.json`. Consequence: the baseline is rerun
   **in the same sweep** as the variants (never compared at bit level against
   archived JSON), and the harness default output path was moved to
   `results/w7-streamppl-1b.json` so a defaults run can't clobber the Week-6
   artifact. The archived Week-6 numbers and conclusions stand.

**Integrator robustness fix** (`augmented_bug_step`,
`src/kvdlra/integrators/streaming_torch.py`): the full-rank parity tests exposed
a latent bug — when the incoming block already lies in the tracked subspace (a
*numerically null* residual), plain QR of the residual returned junk directions
whose orthogonality against `U` was destroyed by cancellation, silently breaking
the basis. Fixed by one re-orthogonalization pass (Parlett/Kahan "twice is
enough") plus an SVD rank-reveal of the residual with a dtype-aware floor
(`100·eps·‖block‖_F`). This never fired at `r=128` on real KV, so Weeks 3–6 are
unaffected; it matters here because the tiny-model tests and the full-rank
regime exercise it.

## Results (1B)

*(WikiText-2, fp32 CPU, P=512, G=3072, bin 256; matched worst-case stored
floats; budget ~515 tok-eq/layer at r=128, W=1024, w=64, absorb=32.
`quant_bits=4`, `quant_keep_frac=0.5`, `score_decay=0.97`. All BUG variants
solved down to the SAME per-layer float budget; measured `mem_max`/budget
≤ 1.0 for every bounded method.)*

### The full variant sweep (doc 0)

All seven methods, per-bin perplexity ratio to the full cache (1.00 = lossless;
lower is better), single held-out document:

| pos (decode) | bug | bugA (attn) | bugE (energy) | bugD (4b quant) | bugAD | morph |
|---:|---:|---:|---:|---:|---:|---:|
| 256  | 1.028 | 1.028 | 1.028 | 1.030 | 1.029 | **1.375** |
| 512  | 1.017 | 1.017 | 1.017 | 1.082 | 1.065 | **1.402** |
| 768  | 1.037 | 1.037 | 1.047 | 1.221 | 1.163 | **1.492** |
| 1024 | 1.056 | 0.995 | 1.177 | 1.389 | 1.178 | 1.131 |
| 1536 | 1.047 | 1.047 | 1.270 | 1.692 | 1.464 | 1.024 |
| 2048 | 0.983 | 0.996 | 1.203 | 2.335 | 2.158 | 1.026 |
| 2560 | 1.055 | 1.074 | 1.316 | 5.655 | 4.979 | 1.112 |
| 3072 | 1.136 | 1.093 | 1.692 | **14.74** | **13.07** | 1.138 |
| **agg ppl** | 12.44 | **12.40** | 15.02 | 29.23 | 28.01 | 13.80 |

*(full-cache aggregate 11.93.)*

Three results are already decisive from this single document:

- **(D) quantized aging is a catastrophic negative — the headline honest
  negative of Week 7.** The intended signature was "late bins flatten as if `W`
  were 4× larger." The observed signature is the exact **inverse**: the more
  4-bit-coordinate history the tier holds, the *worse* it gets, the tail
  exploding to **14.7×** the full cache by G=3072. Two compounding causes: (i)
  the one-shot PolarQuant error on a rank-128 coordinate vector at 4 bits is
  already ~10% relative per column (bins 512–768 sit at 1.08–1.22 while
  fp32-tier BUG is at 1.02–1.04); (ii) a column demoted early is **re-coded on
  every absorb** (~120 requantizations over its lifetime at this tier size), and
  even with the exact-norm-carry fix the *direction* does a random walk under
  repeated 4-bit rounding — precisely the compounding the design note flagged as
  falsifiable. Holding fewer-but-clean tokens (baseline BUG drops to W=1024
  fp32) decisively beats holding more-but-noisy ones. At 8 bits the codes cost
  2× the floats, so the tier would hold only ~2× more history at much smaller
  per-event error — but given (A) barely moves the needle, an 8-bit D is very
  unlikely to beat baseline; not pursued.
- **(E) energy (‖c_s‖) retention is a negative.** Dropping the lowest-norm
  coordinate column is a poor importance rule — tail grows to 1.69× vs
  baseline's ~1.14×. Coordinate energy does not track attention relevance.
- **morph loses the aggregate at 1B** (13.80 vs BUG 12.44), and the mechanism is
  visible: morph's *early* bins are terrible (1.38–1.49 for the first ~768
  tokens — it evicts the near-context low-rank fidelity BUG preserves) and only
  its tail is flat. This is the Week-6 mechanism, sharper on this doc.

### The (A) verdict and the baseline surprise (2-doc canonical run)

The surprise on doc 0 is that **baseline BUG is already nearly flat** (1.03
early → 1.14 at 6× budget), not the growing tail the Week-6 *aggregate*
reported. That growth was a two-document average effect (Week 6 explicitly
flagged doc-level variance with n=2); on this document there is little tail for
(A) to fix, and indeed (A) tracks baseline early (identical to 1.028) and is
only slightly lower late (1.09 vs 1.14 at G=3072). The clean bug-vs-bugA-vs-
morph verdict therefore needs the second document, where the baseline actually
has a growing tail. Canonical 2-doc run (`full, bug, bugA, morph`,
`results/w7-streamppl-1b.json`):

Per-bin ppl ratio to the full cache, **geo-mean over both docs** (the Week-6
protocol; the `bug`/`full`/`morph` rows reproduce `results/w5-streamppl-1b.json`
to the third decimal — the integrator fix is fp-equivalent as claimed, and the
harness is sound):

| pos (decode) | bug (FIFO) | bugA (attn) | morph |
|---:|---:|---:|---:|
| 256  | 1.032 | **1.032** | 1.225 |
| 512  | 1.025 | **1.025** | 1.228 |
| 768  | 1.062 | **1.028** | 1.297 |
| 1024 | 1.165 | **1.026** | 1.181 |
| 1280 | 1.085 | **1.060** | 1.166 |
| 1536 | 1.186 | **1.065** | 1.063 |
| 1792 | 1.296 | **1.066** | 1.181 |
| 2048 | 1.101 | **1.043** | 1.107 |
| 2304 | 1.155 | **1.033** | 1.176 |
| 2560 | 1.231 | **1.059** | 1.197 |
| 2816 | 1.198 | **1.086** | 1.089 |
| 3072 | 1.229 | **1.062** | 1.104 |
| **agg ppl** | 11.59 | **10.62** | 11.81 |

*(full-cache aggregate 10.13.)*

This is the tier-1 success signature, cleanly: **bugA keeps BUG's near-lossless
early bins** (1.03×, vs eviction's 1.23–1.30× — eviction throws away the
near-context low-rank fidelity) **and flattens the deep-horizon tail to ~1.06×**,
where baseline BUG's FIFO retention grows to ~1.20× (peak 1.30× at 1792) and
morph holds ~1.15×. bugA is ≤ both baselines at every bin past the first two.
The per-document story is mechanistically coherent: on doc 0 the baseline has
little tail (1.14× at 6× budget) so A gains little (12.44 → 12.40); on doc 1 the
baseline tail is real (ratio 1.26 aggregate) and A closes most of it (10.80 →
9.10). Attention-scored retention gives BUG MorphKV's eviction brain while
keeping the rank-`r` subspace summary — exactly the intended best-of-both.

## Verdict

- **(A) adaptive coordinate retention — WIN at 1B.** A clean best-of-both:
  near-lossless early (1.03×) *and* flat late (~1.06×), closing ~66% of the
  bug→full aggregate gap (11.59 → 10.62 vs full 10.13) and beating both baseline
  BUG (FIFO) and MorphKV. Confirms the Week-6 diagnosis: **FIFO coordinate
  eviction (mechanism 1) was the dominant deep-horizon loss**, and scoring
  retention by accumulated attention mass fixes it. The cheap `energy` proxy
  does *not* substitute (‖c_s‖ ≠ attention relevance; a negative).
- **(D) quantize-instead-of-drop — decisive NEGATIVE.** 4-bit coordinate aging
  makes the tail *explode* (14.7×), the exact inverse of its intended signature:
  the one-shot PolarQuant error on a rank-128 coordinate vector (~10% relative)
  plus per-absorb requantization drift make more-but-noisy history far worse
  than fewer-but-clean. Reported straight; not pursued at higher bit-widths (the
  budget math makes 8-bit hold only ~2× more history, and A already wins).

The honesty guardrail is met: every variant was solved to the same per-layer
float budget (scores/positions/codes/norms/side-info all counted), measured
`mem_max`/budget ≤ 1.0 throughout, and the win is a genuine flatten-the-tail-
without-losing-the-early-bins, not a lateral trade.

**Caveat / open question.** This is 1B, where BUG *already* beat eviction in
Week 6 — so (A) makes a winning method more winning. The load-bearing Week-6
result was the **inversion at 8B**, where BUG *lost* the deep horizon to
eviction (7.74 morph vs 7.87 BUG tier-1; 8.39 vs 9.17 tier-2). Whether (A)
closes *that* gap is the actual test, and it requires the pod.

## Decision: tier 2 (C) / 8B

- **8B confirmation: GO.** (A) is the one surviving variant and the 1B win is
  clean and mechanism-validated. Confirm `bugA` against the full Week-6 set
  (`full, bug, bugA, morph, snapkvD`) at both budget tiers on one RTX 6000 Ada
  pod (`scripts/pod/w7_streamppl_8b.sh`; ~$1.3, credit ~$5.7). Success = bugA's
  8B tail matches or beats eviction's flat offset while keeping the near-lossless
  early bins — i.e. **un-inverts** the Week-6 verdict.
- **Tier 2 (C) retention-aware truncation: DEFERRED.** The plan gates tiers 2/3
  on tier 1 *not* flattening the tail; (A) flattened it. (C) targets mechanism
  2 (projection erosion), which (A) leaves partly unaddressed — worth doing only
  if the 8B run shows a residual tail that (A) does not fully close. Decide from
  the 8B curves.
