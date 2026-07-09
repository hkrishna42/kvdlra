# Week-7 dominance program — can BUG *defeat* eviction, not just tie it?

> Follow-up to `docs/week7.md`. Week 7 showed attention-scored coordinate
> retention (`bugA`) turns BUG's Week-6 deep-horizon *loss* into a *win* at the
> moderate budget and un-inverts the 8B verdict — but bugA still **loses to
> eviction at the aggressive budget** (the rank squeeze, mechanism 3), and even
> its wins are thin. This program asks the sharper question: is there a
> technique that makes BUG *mathematically dominate* eviction and baseline BUG?
>
> A 6-lens multi-agent ideation panel (see the session transcript) generated and
> adversarially ranked candidates. It **rejected every "dominance" claim as a
> category error** (containing eviction and BUG as special cases does not imply
> the interior optimum beats them) and left two mechanism-targeted survivors,
> both aimed at the one un-closed regime (the aggressive-budget rank squeeze):
> **Technique 1 — rank↔coverage water-filling**, and **Technique 2 — SLASH**
> (exact heavy-hitters + low-rank residual). Both are implemented, tested, and
> run at 1B with honest matched-memory accounting.

## The target regime

At an aggressive per-layer budget (~89 tok-eq, 1B, WikiText-2, doc 0, G=1024)
eviction beats BUG. Baseline numbers (streaming ppl, matched memory, full cache
= 7.24):

| method | ppl | what it keeps |
|---|---:|---|
| morph (eviction) | **18.71** | 39 exact whole tokens |
| bug-r32 (Week-6 config) | 22.32 | rank-32 summary of 64 tokens |

The 3.6-ppl gap is what the two techniques try to close.

## Technique 1 — rank↔coverage water-filling (`w7_rank_sweep.py`)

At a **frozen** memory budget, split it differently between rank `r` (fidelity
per retained token) and coordinate coverage `W` (how many tokens the summary
spans). Lower `r` frees basis + core floats and cheapens each column, so `W`
grows. The augmented BUG integrator is already rank-agnostic — this is a pure
allocation question with **zero new state** (the cleanest possible matched
memory). FIFO sweep at ~89 tok-eq (all matched):

| rank r | coverage W | ppl |
|---:|---:|---:|
| 8 | 1912 | 25.50 |
| 12 | 1097 | 23.55 |
| 16 | 688 | 22.00 |
| **24** | **274** | **20.95** ← interior optimum |
| 32 | 64 | 22.32 |

**A clean U-shape** — not monotone. Trading rank 32→24 (coverage 64→274) buys
1.37 ppl; going coarser (r≤16) loses it again. So rank-for-coverage is a **real
lever** and the rank-adaptivity direction is alive. Adding attention retention
(`bugA`, matched accounting) improves the optimum to **20.62** at r=24. But that
is still **1.9 ppl short of morph (18.71)** — Technique 1 closes ~40% of the
gap, not all of it.

## Technique 2 — SLASH (`src/kvdlra/cache/bug_cache.py`, `hh_budget`)

On each absorb, pool the graduating block with the current exact tier, keep the
top-`hh_budget` tokens by recent-attention score **verbatim** (post-RoPE K, raw
V — like sinks), and feed only the rest to `augmented_bug_step`. Because the
exact peaks never enter the low-rank step, the rank-`r` basis summarizes the
**outlier-removed residual** spectrum — a robust-PCA decomposition matched to
attention's heavy-tailed structure. It contains pure eviction (`r`→small) and
pure BUG (`hh_budget`=0) as special cases. Demoted former-heavy-hitters re-enter
the tail at their true (non-contiguous) positions. Honest accounting: each exact
token costs a full `2n+2` floats (K+V+position+score), taken **out of** the
coordinate coverage. (6 tests: exact-mode parity, mask consistency across the
3-tier middle, promotion keeps the top-scored tokens verbatim, honest memory,
constant memory; `hh_budget=0` is bit-identical to prior behaviour.)

SLASH at ~89 tok-eq, r=24, matched memory:

| method | ppl | exact + low-rank |
|---|---:|---|
| morph | **18.71** | 39 exact |
| **slash h4** | **20.15** | 4 exact + 180 low-rank |
| slash h8 | 20.29 | 8 exact + 98 low-rank |
| bugA r24 (h0) | 20.62 | 262 low-rank |
| slash h12 | 22.52 | 12 exact + 16 low-rank |

SLASH improves bugA by **0.47 ppl** (best at h4). The cumulative chain of all
three levers — **22.32 → 20.95 (rank) → 20.62 (retention) → 20.15 (SLASH)** —
closes **60% of the gap to eviction**. But **morph still wins by 1.44 ppl.**

## Why eviction wins the aggressive budget (the honest mechanism)

At ~89 tok-eq (91136 floats/layer) BUG's **fixed overhead** — sinks + recent
ring + basis (`2nr`) + core (`2r²`) + the ring-score buffer — is **~78000
floats, 85% of the budget.** Only ~15% is left for actual token coverage.
Eviction pays *none* of this overhead: a MorphKV token costs `2n + h_kv·R/C`
floats and nothing else, so it fits 39 exact tokens where SLASH fits 4-8. At
extreme compression the low-rank machinery's **structural overhead is fatal**,
and no amount of retention/heavy-hitter cleverness overcomes it — the levers
narrow the gap (60%) but the overhead floor prevents closing it. This is a clean,
mechanistically-explained **honest negative**: BUG does *not* mathematically
dominate eviction; at extreme compression eviction is fundamentally more
memory-efficient, exactly because it has no basis to amortize.

The corollary — and the remaining shot at a decisive win — is that BUG's
advantage lives where its overhead is a *small* fraction: the **moderate
budget**, where bugA already beats morph (10.62 vs 11.81). Whether SLASH extends
that win toward the full-cache floor (10.13) is tested next.

## Moderate-budget SLASH

At ~515 tok-eq (1B, doc 0, G=1536), where BUG *already* beats eviction, does the
exact tier extend the lead toward the full-cache floor? (matched memory, full =
9.51):

| method | ppl | exact + low-rank |
|---|---:|---|
| **slash h32** | **9.71** | 32 exact + 888 low-rank |
| slash h64 | 9.73 | 64 exact + 761 low-rank |
| bugA (h0) | 9.77 | 1015 low-rank |
| slash h128 | 9.97 | 128 exact + 506 low-rank |
| morph | 11.68 | 380 exact |

SLASH (h32) improves bugA by **0.06 ppl**, closing ~22% of the bugA→full gap —
a real but **small** gain. Too many exact tokens (h128) *hurts* (the low-rank
coverage it displaces was worth more than the exactness bought). So at the
moderate budget the low-rank floor is already near-optimal and the exact tier
adds only a sliver.

## Verdict

**No dominance — an honest regime split, plus two incremental improvements.**

- **BUG does not mathematically defeat eviction.** The picture is a clean split:
  BUG/SLASH win the moderate budget decisively (9.71 vs morph 11.68), eviction
  wins the aggressive budget (18.71 vs best-BUG 20.15). Neither dominates; the
  ideation panel's rejection of "dominance-by-containment" was correct.
- **SLASH (exact heavy-hitters + low-rank residual) is a genuine but small
  gain**: +0.47 ppl at the aggressive budget (as part of a 60%-gap-closing
  chain), +0.06 at the moderate budget. Real, mechanism-backed, honestly
  accounted, tested — but not the "vast improvement" a dominance result would be.
- **The aggressive-budget loss is structural, not tunable.** BUG's fixed
  overhead (basis + core) is ~85% of a tiny budget; eviction pays none. No
  retention/heavy-hitter cleverness overcomes a structural overhead — only
  *removing* the overhead would, and low-rank cannot remove its own basis.

## DLRA without BUG — is the subspace tracker leaving accuracy on the table?

BUG is one DLRA integrator; the tracker could be swapped for another integrator
(PSI, projected-RK) or a non-DLRA streaming sketch. The strongest non-DLRA
competitor for our exact setting (sketching a streaming data matrix) is
**Frequent Directions** (Liberty 2013) — deterministic, with a provable
covariance bound. Implemented (`src/kvdlra/integrators/frequent_directions.py`,
5 tests) and compared to BUG / oracle / incremental-SVD on the same real KV
streams and metric as Week 2 (layer 8, pre-RoPE, 5 docs, mean relative-Frobenius
reconstruction error; lower is better):

| rank | oracle | BUG | FD (matched mem) | FD (2× mem) | incremental SVD |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.2996 | **0.3017** | 0.3223 | 0.3003 | 0.3058 |
| 32 | 0.2290 | **0.2313** | 0.2442 | 0.2292 | 0.2351 |
| 64 | 0.1567 | **0.1581** | 0.1629 | 0.1567 | 0.1608 |
| 128 | 0.0980 | **0.0994** | 0.0991 | 0.0980 | 0.1017 |

**The tracker is NOT leaving accuracy on the table.** BUG is within ~0.7-1.4% of
the oracle, and at **matched memory it beats Frequent Directions** at ranks
16/32/64 and ties at 128 — and beats incremental SVD everywhere. FD only reaches
oracle quality at **2× the memory** (`ell=2r`), so BUG is the more
*memory-efficient* tracker for the same accuracy. Conclusion: the DLRA/BUG
subspace tracker is a settled, well-justified choice; swapping integrators (FD,
PSI, projected-RK) cannot close the gap to the oracle because BUG is already
essentially at it. Any real gain must come from *changing what is stored*, not
*how the subspace is tracked* — i.e. the overhead floor, below.

**Where a decisive win actually lives (the shorthand/codebook direction).** The
one structural fix for the overhead floor is to make the per-token cost a shared,
amortized *shorthand* — a learned/product **vector-quantization codebook** whose
codebook is side-information shared across the whole cache, so a token costs a
few bits with ~zero marginal overhead. This is an active, strong line
(CommVQ arXiv:2506.18879; PQCache SIGMOD'25 arXiv:2407.12820; MILLION
arXiv:2504.03661 — the last is literally PQ-codes + outlier-exact, i.e. SLASH
from the other direction). The novel-to-us synthesis is a **vector codebook on
the BUG coordinates** (one short code per token instead of `r` floats), which
would collapse BUG's coverage cost the way VQ collapses eviction's. Caveat: the
*scalar* version of this (Week-7 variant D, PolarQuant per-coordinate) was
**catastrophic** under the streaming requantize carry, and a vector codebook
needs calibration data (a break from the training-free stance). This is a real
fork, flagged for a decision rather than pursued blindly.

## Hybrid brainstorm round (JPEG / lossless / merge)

A second round explored `(BUG + X)` hybrids inspired by classical compression.
The honest outcomes:

- **Hierarchical coordinate merging (BUG + token-merging / JPEG2000 wavelet
  analog) — NEGATIVE.** Instead of *dropping* the oldest coordinate columns on
  overflow, *merge* adjacent equal-weight pairs into count-weighted centroids (a
  dyadic pyramid: unbounded history at geometrically decaying resolution,
  constant memory). Result at the aggressive budget (1B, doc 0): `bugM ≈ bug`
  (r24 20.99 vs 20.95; r8 identical, r32 slightly worse). Mechanism: without the
  **log-count softmax key correction**, a merged super-column of `m` tokens is
  attended with a single token's weight, so it is under-weighted `m×` and
  effectively **ignored** — the merged history is present but invisible to
  attention. The documented approximation is falsified; making merging pay off
  needs attention-level surgery (a learned/log-count gate), the same invasive
  step the linear-attention tail needs. Built, tested (5 tests), reported
  straight. `merge` flag on `BugStreamingCache`.

- **JPEG pipeline mapped onto BUG.** DCT ≈ BUG's basis but *adaptive* (KLT is the
  optimal energy-compacting transform; JPEG's fixed DCT is a data-oblivious
  approximation whose only advantage is zero basis-storage — the overhead-floor
  fix, = the codebook direction). Quantization ≈ Week-7 variant D (catastrophic
  under streaming requant); JPEG's variable bit-allocation is the one untried
  twist but inherits D's failure mode. Entropy coding is genuinely new and
  lossless, but **breaks KV random access and the constant-memory guarantee**
  (variable-length codes), which is why no production KV-quant method uses it.
  Deeper point: JPEG's 10× comes from *image-signal redundancy*, not the
  pipeline; we measured KV states are far less compressible (~10% error even at
  r=128/512), so the pipeline transfers but the compressibility does not.

- **Lossless matrix compression (CSR / LZ4 / bit-packing) — wrong structure.**
  These exploit sparsity / repetition / limited range; BUG's `U`, `C` are dense,
  high-entropy, continuous floats (low-rank ≠ sparse). Lossless is
  entropy-bounded, and any such trick shrinks eviction's stored tokens equally,
  so it cannot change the matched-memory ranking (why we count dtype-agnostic
  floats).

- **Diagonal-core lossless win — REAL, implemented.** The one genuine lossless
  saving the round surfaced: BUG's square-root core `B` is *provably diagonal*
  (`augmented_bug_step` returns `diag(σ)` every step; nothing rotates it), yet we
  stored/counted it as a dense `r×r`. That over-counted BUG's memory by `r²−r`
  per stream — **~2% of budget at r=32, ~6% at r=128** — meaning we had been
  *under-provisioning BUG in every prior experiment*. Fixed: the core is now
  counted at its `r` diagonal entries (deployable footprint, same convention as
  quant codes at bits/32) and the budget solvers charge `2r` not `2r²`, freeing
  that memory for coordinate coverage. At the moderate tier this raises bugA's
  coverage from **W=1015 → W=1141 (+12.4%)** at identical 527360-float budget.

  **But the freed budget does NOT buy a ppl win** (1B, doc 0, moderate tier,
  budget held at 527360, morph unchanged at 11.682): bugA-r128 goes
  W=1015→1141 and ppl **9.772 → 9.793** — a slight *regression*. Reason: at the
  moderate tier bugA-r128 is already *at/past its coordinate-coverage sweet
  spot* (the same run family showed `slash-h32`, with *less* low-rank coverage +
  a few exact tokens, was best at 9.713), so more low-rank coverage adds
  approximate old tokens that don't help and marginally hurt. **Verdict:** the
  diagonal-core fix is a *correct, honest accounting improvement* (BUG's true
  footprint really is `2r` less per stream, and prior results had been charging
  BUG for phantom memory) — but it is *not* a free quality win, because coverage
  is not the binding constraint at the moderate tier. It would help where
  coverage *is* binding (reallocated to rank, or to SLASH exact tokens), not as
  extra low-rank width. Kept for correctness; reported straight.

## Bottom line of the whole dominance program

Across a 6-lens ideation panel and every hybrid built and measured — rank↔
coverage water-filling, SLASH (exact + low-rank), quantized aging (D),
Frequent-Directions tracker swap, hierarchical merge, the diagonal-core
accounting fix — the conclusion is consistent and honest: **BUG is a
near-oracle, correct, constant-memory streaming cache that decisively wins the
moderate-compression regime and loses the extreme-compression regime, and no
tuning-level change moves that line.** The two walls are measured, not
conjectured: the *near-oracle tracking ceiling* (FD/PSI can't beat BUG; BUG is
within ~1% of the oracle) and the *structural basis-overhead floor* (85% of a
tiny budget, which eviction doesn't pay). The only lever that could change the
frontier is amortizing the per-token cost with a **shared vector-quantization
codebook** (the shorthand/MILLION direction) — which needs calibration data and
solves the streaming-requant problem that killed variant D. That remains the one
open, higher-variance fork; everything reachable without it is now mapped.
