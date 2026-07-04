# Week 5 — placing streaming BUG in the low-rank field (in progress)

Plan: `docs/week5-plan.md`. This file records results as they land. Priority order
(plan §"Priority"): **4.5 integrator ablation → Axis C long-context (Exp A) → Axis
A (Palu/LoRC) → streaming-decode + Axis B → Exp B RULER.**

---

## Week 4.5 — DLRA integrator ablation (DONE)

**Question.** Before BUG enters the expensive Week-5 comparisons, is it the right
DLRA integrator for the streaming-KV subspace-tracking problem, or does a sibling
integrator track better / cost less? Offline bake-off — reconstruction error vs
the truncated-SVD **oracle**, per-sweep **cost**, and streaming
**rank-adaptivity**, on captured pre-RoPE KV. **No LLM, no pod.**

**Verdict — ship plain BUG (the incumbent).** In the fair operating range all
three integrators cluster near the oracle (a **near-tie**, as the plan expected —
not a competitive lever). BUG is chosen because it is the only candidate that is
*both* (1) at least as accurate as every rival and (2) **rank-adaptive at a
streaming block size** — fixed-rank projector-splitting (PSI) cannot exceed a rank
of one streaming block, and parallel-BUG is rank-adaptive but measurably less
accurate. This validates the project's use of the augmented rank-adaptive BUG step
on real KV — on the grounds of **rank-adaptivity and the robust square-root core**,
not (see the correction below) a numerical blow-up of the rivals.

> **Correction (honesty note).** An earlier version of this section claimed "PSI
> destabilizes at high rank (its backward step), err/oracle → 4.9× at r=384." That
> was **wrong** and is retracted. Our `psi_step` is *fixed-rank* (basic
> projector-splitting cannot grow rank), so at `block_size=128` PSI is capped at
> rank 128; the apparent "blow-up" at r>128 was an artifact of comparing a
> **rank-128 PSI subspace against a higher-rank oracle**, not the backward step. An
> independent critic caught this. When PSI is allowed to reach the target rank
> (block_size ≥ rank) it stays within ~1–3% of the oracle with **no instability**.
> The honest finding is a near-tie in accuracy; BUG's edge is rank-adaptivity, not
> rival instability. The backward step's known σ_min-sensitivity is a real
> theoretical concern that simply did not manifest on this well-conditioned KV.

### What was built
Three integrators run through **one** block-streaming harness (identical
first-block truncated-SVD seed, identical rank truncation via the shared
`_truncation_rank`, identical `‖M − U Uᵀ M‖_F / ‖M‖_F` metric — which depends only
on `span(U)`, so the comparison is exactly apples-to-apples):

- **BUG** — `integrators/streaming_torch.py::blocked_bug_subspace` (the incumbent):
  augmented Galerkin, **forward** S-step, **square-root** core `[[B, A],[0, R]]`
  (never forms `B Bᵀ`). Rank-adaptive. Ceruti–Kusch–Lubich 2022 (arXiv:2104.05247).
- **PSI** — `integrators/streaming_variants.py::psi_step` / `psi_subspace` (new):
  projector-splitting with a **backward** S-substep, in its natural
  **covariance-core** form. **Fixed-rank** (basic projector-splitting): the seed
  rank propagates, so a streaming block size `b` caps it at rank `≤ b`.
  Lubich–Oseledets 2014 (arXiv:1301.1058).
- **Parallel-BUG** — `integrators/streaming_variants.py::parallel_bug_step` /
  `parallel_bug_subspace` (new): the parallel rank-adaptive integrator
  (Ceruti–Kusch–Lubich 2024, arXiv:2304.05660) specialized to the streaming
  increment — the in-subspace rotation (an `r×(r+b)` SVD) and the new-direction
  admission (a `b×b` SVD) are computed **independently**, dropping the
  in-basis↔new-direction cross-coupling that BUG's joint `(r+b)×(r+b)` SVD keeps.
  Rank-adaptive.

Harness/figure/verdict: `scripts/w45_integrator_ablation.py` →
`figures/week5/integrator_ablation.{png,pdf,json}`. Tests:
`tests/test_streaming_variants.py` (15).

### The fair regime (why r ≤ 128 at block 128)
A fair accuracy comparison needs every method to actually reach the requested rank
`r`, and BUG's augmented `[U | Q]` basis must fit in `Rⁿ`. Both require:

- `r ≤ block_size` — else fixed-rank PSI is capped below `r` (apples-to-oranges);
- `r + block_size ≤ n_features` — else BUG's augmented basis is rank-deficient and
  degenerates (a latent constraint in `blocked_bug_subspace`; at r=256/block=384,
  `256+384 > 512`, BUG's error jumps to 5× — see the follow-up note).

Both hold across the **KV operating range r ≤ 128 at block_size = 128**
(`128+128 < 512`), which is the band that matters (Week-3/4 operate at r ≤ 128).
The ablation runs there and asserts each method reaches exactly rank `r`. A
separate rank-adaptivity probe pushes `rank_cap` past `block_size` to expose PSI's
fixed-rank cap.

### Results (5 docs, pre-RoPE Layer-8 K, block 128, fp64 core)

**Accuracy — near-tie** (relative reconstruction error, mean over docs;
err/oracle = gap to the Eckart–Young optimum, 1.000 = optimal):

| rank | oracle | BUG | BUG/orc | PSI | PSI/orc | parallel | par/orc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16  | 0.2996 | 0.3012 | **1.005** | 0.3021 | 1.008 | 0.3538 | 1.181 |
| 32  | 0.2290 | 0.2309 | **1.008** | 0.2316 | 1.011 | 0.2807 | 1.226 |
| 64  | 0.1567 | 0.1578 | **1.007** | 0.1583 | 1.010 | 0.2053 | 1.310 |
| 128 | 0.0980 | 0.0992 | **1.012** | 0.0997 | 1.017 | 0.1359 | 1.386 |

- **BUG and PSI both hug the oracle** (≤1.012× and ≤1.017×): the projector-splitting
  backward step costs essentially nothing on this well-conditioned KV. Near-tie,
  BUG a hair ahead.
- **Parallel-BUG trails (1.18–1.39× oracle).** The gap is the in-basis↔new-direction
  cross-coupling it decouples away; it shrinks monotonically as `block_size → T`
  (fewer decoupling steps: 1.39× at bs=64 → ~1.23× at bs=512 for r=64), confirming
  the decoupling is the cause. This is a property of *this* decoupling, not a
  general verdict on the parallel-integrator family.

**Streaming rank-adaptivity — the deciding axis** (achieved basis rank vs requested
`rank_cap`, at streaming `block_size = 128`):

| rank_cap | 64 | 128 | 192 | 256 | 384 |
|---|---:|---:|---:|---:|---:|
| BUG          | 64 | 128 | 192 | 256 | 384 |
| **PSI**      | 64 | 128 | **128** | **128** | **128** |
| parallel-BUG | 64 | 128 | 192 | 256 | 384 |

BUG and parallel-BUG grow the basis across blocks and reach any `rank_cap` (up to
`rank_cap + block_size ≤ n_features`); **fixed-rank PSI is stuck at rank ≤
block_size.** For genuine low-latency streaming (small blocks) where the useful
rank can exceed one block, that is a decisive practical advantage for the
rank-adaptive integrators.

### Cost (one streaming sweep, ms, dev CPU, largest doc, block 128)

| rank | BUG | PSI | parallel-BUG |
|---:|---:|---:|---:|
| 16  | 161 | **110** | 155 |
| 64  | 216 | **133** | 182 |
| 128 | 386 | **199** | 264 |

PSI is cheapest (fixed-rank → smaller QRs), but its cheapness comes bundled with
its fixed-rank ceiling; in the operating range every method's sweep is well under a
second. Cost is not the deciding factor.

### Decision
All three ≈ oracle in the operating range (rigor result, not a lever). **Winner =
BUG**: it is the most accurate method that is *also* rank-adaptive at a streaming
block size. PSI matches BUG's accuracy but is fixed-rank (a real streaming
limitation); parallel-BUG is rank-adaptive but 1.2–1.4× less accurate. Since the
winner is the incumbent, "wire the winner into the streaming path" (plan §4.5b) is
a no-op: the validated BUG step stays the default, and PSI/parallel-BUG remain as
documented, tested reference implementations in `integrators/streaming_variants.py`.
**Honest framing:** this confirms BUG's design (rank-adaptive + square-root core) is
the right choice for KV — a near-tie in accuracy decided on rank-adaptivity — and
rules out the "a cheaper/parallel integrator gets the streaming niche for free"
shortcut.

### Follow-up found during the ablation (out of scope)
`blocked_bug_subspace` (and the new variants) are only correct when
`rank_cap + block_size ≤ n_features`: the augmented `[U | Q]` basis otherwise
exceeds `Rⁿ` and degenerates (silent ~5× error at the boundary; a shape crash when
`block_size > n_features`). Never triggered in production (streaming uses
`block_size = 128 ≪ 512`), but a real latent bug — flagged for a separate fix
(guard/clamp the residual QR to at most `n − rank` new directions).

---

## Axis C / Experiment A — long-context scaling (DONE, 8B)

**Question.** BUG's fixed `U` basis is a per-layer overhead that **amortizes as
context grows** (its stored-memory ratio shrinks with `T`, while eviction keeps a
fixed token *fraction* → ctx-independent memory). Does BUG's fair
perplexity–memory frontier therefore **pass SnapKV's (and close on EA's) as `T`
grows**? Falsifiable either way.

**Setup.** Llama-3.1-8B (ungated `unsloth` mirror), bf16, WikiText-2,
prefill-then-score, **position-fair** (explicit `position_ids`), every mechanism
×TurboQuant-4bit. ctx ∈ {1K, 4K, 8K, 16K, 32K}; BUG ranks {64,128,256}; eviction
compression {0.5,0.7,0.85}. Ran on one RTX 6000 Ada (48 GB) via onstart-batch,
~50 min, ≈ $0.46; **all five context lengths completed, including 32K (no OOM)**.
`scripts/w5_longctx.py` (+`--plot-only`), `results/w5-longctx-8b.json`,
`figures/week5/longctx_crossover*.png`. ⚠️ **Noise caveat:** `n_windows=4`
(~1020 scored tokens/ctx), so fine gaps (±0.05 ppl) are within noise and absolute
ppl varies across ctx (different WikiText windows); read the *trends*, not the
third decimal. A confirmation run wants `n_windows ≥ 8`.

**Verdict — amortization is real, but the clean "BUG passes eviction as `T` grows"
thesis is NOT supported. Partial / non-monotonic result.**

**(1) Memory amortization: confirmed and strong** (structural, not noise). BUG's
stored-memory ratio at fixed rank drops ~4.5× as context grows, while eviction's is
ctx-independent:

| ctx | BUG r256 mem | BUG r128 mem | SnapKV cr0.5 mem |
|---:|---:|---:|---:|
| 1024  | 0.317 | 0.161 | 0.125 |
| 4096  | 0.127 | 0.064 | 0.125 |
| 8192  | 0.095 | 0.048 | 0.125 |
| 16384 | 0.079 | 0.040 | 0.125 |
| 32768 | 0.071 | 0.036 | 0.125 |

**(2) Quality crossover: BUG closes to parity by mid-context, then eviction
re-opens the gap.** Min perplexity gap between BUG's and each rival's Pareto
frontier over their overlapping memory range (negative = BUG's frontier passes the
rival somewhere):

| ctx | BUG − SnapKV | BUG − EA |
|---:|---:|---:|
| 1024  | +0.147 | +0.386 |
| 4096  | +0.056 | +0.004 |
| 8192  | **−0.028** | **−0.001** |
| 16384 | +0.056 | −0.021 |
| 32768 | +0.152 | n/a |

The gap is **U-shaped** (`figures/week5/longctx_crossover.png`): BUG starts well
behind at 1K, closes to **near-parity / a nominal crossover around 4–8K** (within
noise of zero), then the gap **re-widens** at 16–32K. Why the re-widening: at very
long context eviction becomes **near-lossless** — SnapKV/EA ×4-bit sit essentially
at baseline (e.g. ctx 32K: SnapKV cr0.5 +0.024, EA cr0.5 −0.036), leaving almost no
quality headroom for BUG to beat, while BUG's low-rank reconstruction keeps a
~+0.1–0.2 penalty at its (now very cheap) operating memory. So amortization shifts
BUG's frontier far to the left (cheaper) but does not let it *overtake* eviction on
perplexity at extreme context.

**Honest read.** The amortization mechanism is real and materially improves BUG's
standing with context — from clearly behind at 1K to competitive/near-parity at
4–8K. But it is **not** a monotonic "passes SnapKV and stays ahead": eviction's
near-losslessness at 16–32K reclaims the lead. This is consistent with the Week-4
finding (EA leads on WikiText perplexity; pure quant is a strong floor) and extends
it — context helps BUG most at *mid* range, not the extreme. The place BUG's
amortized, no-eviction frontier most plausibly *wins* is **retrieval**, where
eviction drops un-cued facts (already seen: BUG 15/15 vs SnapKV 3/15 at 1.6K) — that
is **Experiment B** (RULER / long-needle at 8–32K), the natural follow-up. A clean
negative on perplexity, an open (promising) question on retrieval.

---

## Hybrid press (exact tokens + low-rank residual) — DONE, honest negative on perplexity

**Idea (post-Axis-C).** Pure BUG spreads its budget across *all* tokens; a few
high-norm tokens carry most of the signal (what eviction keeps). The **hybrid**
(`BUGPress(n_exact=k)`) keeps the top-`k` high-norm tokens *exact* and low-ranks
the rest — matching eviction's strength on the important tokens while keeping a
cheap summary of everything else, and (bonus) leaving a more low-rank residual.

**Result — the mechanism works at fixed rank, but it does NOT help on the fair
perplexity–memory frontier, at 1B or 8B.**

- At *fixed aggressive rank* the hybrid clearly rescues BUG (1B ctx 1024, r128/4b:
  pure BUG **+2.21** ppl → hybrid **+0.28**). So keeping the outliers exact does
  what it should.
- But on the fair **memory** frontier it is **dominated by plain BUG**: the exact
  tokens cost memory that scales with `T`, so "spend the budget on higher rank"
  wins. 8B ctx 32768: hybrid-r256 gives the *identical* ppl to plain BUG-r256
  (7.124) at **more** memory (0.118 vs 0.071) — strictly worse, because BUG-r256 is
  already near-lossless so the extra exact tokens buy nothing. Min frontier-gap
  `hybrid − SnapKV` is **+0.13** at 8B ctx 8192 (behind), never negative.
- Neither BUG nor the hybrid passes SnapKV/EA on perplexity (eviction near-lossless
  at long ctx). Consistent with Axis C.

**Caveat / unfinished refinement.** The hybrid keeps its exact tokens at **fp16**,
while SnapKV×TurboQuant **4-bit-quantizes** the tokens *it* keeps — an unfair memory
handicap on the hybrid. A fair version 4-bit-quantizes the kept tokens too (≈4×
cheaper exact set), which would shift the hybrid frontier left and is the honest
next step before calling the hybrid dead. `scripts/w5_hybrid.py`,
`results/w5-hybrid-{1b,8b}.json`, `figures/week5/hybrid_frontiers*.png`.

## Experiment B — long-context needle retrieval — DONE, inconclusive (task saturates at 8B)

**Setup.** `scripts/w5_needle.py`: hide a 5-digit passcode at several depths in a
filler haystack, compress with each press, ask for it (true-position decode). 8B,
ctx {4K, 16K}, 6 trials/method. Methods at matched memory: BUG r128/r256 ×4b,
SnapKV/EA at keep-50%/keep-15%.

**Result — retrieval saturates: every method ≈ 100% at 8B.** BUG, SnapKV, and EA
all score **6/6** at both 4K and 16K; the *only* miss is **SnapKV keep-15%** (most
aggressive, mem 0.038) at ctx 4K → **5/6 (0.83)** — a whisper of the expected
"eviction drops the un-cued needle" effect, but at noise level. An 8B model is
strong enough to recover a single salient passcode even from a heavily compressed
cache, so this simple needle **does not differentiate** the methods.

**Honest read + next.** The dramatic 1B smoke (BUG 1/1 vs SnapKV 0/1) did not
reproduce at 8B — it was a weak-model + single-trial artifact. To actually expose
eviction's failure mode we need a **harder** retrieval task: **RULER**
(multi-needle / multi-hop / aggregation), *many distractor* keys, or a far more
aggressive memory budget where eviction must discard the evidence. That is the real
Experiment B; the single-passcode version is saturated and is a clean *inconclusive*.

## Week-5 honest scorecard
| Result | Verdict |
|---|---|
| 4.5 integrator ablation | BUG wins on rank-adaptivity (near-tie accuracy); PSI "blow-up" retracted |
| Axis C amortization (perplexity) | memory amortizes, but BUG does **not** pass eviction; U-shaped, closest at mid-ctx |
| Hybrid press (perplexity) | mechanism works at fixed rank, but **dominated by plain BUG** on the memory frontier; fp16-exact caveat unfixed |
| Experiment B (single-needle retrieval) | **inconclusive** — saturates at 8B; needs RULER/harder task |

**Bottom line unchanged from Week 4:** BUG is a competitive, well-validated low-rank
KV compressor; on WikiText perplexity it does not beat SnapKV/EA, and the two Week-5
attempts to find a decisive win (long-context amortization, hybrid) came back honest
negatives. The remaining credible win is a *harder retrieval* benchmark and/or the
4-bit-exact hybrid — both are concrete, scoped next steps.

---

## Follow-up 1: 4-bit-exact hybrid (fair kept-token quantization) — narrower negative

Fixed the unfair fp16 handicap: `BUGPress` now PolarQuant-4bit-quantizes the kept
`n_exact` tokens (like eviction ×TurboQuant quantizes ITS kept tokens); only the
`n_sink` sinks stay fp16. This helped materially:

| | 1B ctx 4096 | 8B ctx 8192 | 8B ctx 32768 |
|---|---|---|---|
| min gap hybrid − SnapKV (fp16 exact) | +0.396 | — | — |
| min gap hybrid − SnapKV (**4-bit exact**) | **+0.195** | **+0.041** | +0.181 |

The 4-bit-exact hybrid is now a legitimate frontier member — it posts the **lowest
absolute perplexity of any method** at moderate memory (1B ctx4096: 10.205 vs
SnapKV's best 10.328) and at 8B ctx 8192 it comes within **+0.04** of SnapKV. But it
still does not *pass* SnapKV at matched low memory (SnapKV owns the ≤0.04× regime
the U-basis floor can't reach), and at very long ctx eviction is near-lossless. **A
near-miss, not a win.** `results/w5-hybrid-{1b,8b}.json`.

## Follow-up 2: RULER-lite multi-key retrieval (the harder test) — MIXED, not a BUG win

`scripts/w5_ruler.py`: retrieve 1 of 12 distinct keys (distractors) at aggressive
memory. Unlike the saturated single-needle, this **discriminates** — but not in
BUG's favour. 8B, 6 trials/method:

| method | ctx 4096 (mem) | acc | ctx 16384 (mem) | acc |
|---|---|---|---|---|
| BUG r128/4b | 0.064 | **0.83** | 0.040 | 0.33 |
| BUG-hybrid r128 | 0.077 | 0.83 | 0.053 | 0.33 |
| SnapKV keep-15% | 0.038 | **0.00** | 0.038 | 0.67 |
| ExpectedAttn keep-15% | 0.038 | 0.50 | 0.038 | **1.00** |

- **SnapKV's failure mode is real**: at ctx 4K it drops *every* queried key (0/6) —
  it evicts the un-cued fact, exactly as predicted.
- **But ExpectedAttention is strong** (0.50 → **1.00**), and at **matched memory
  (~0.04×) at ctx 16K, both eviction methods beat BUG** (BUG **0.33** vs SnapKV 0.67
  vs EA 1.00). BUG's fixed-rank low-rank summary **loses precise facts as context
  grows** — r128 spread over 16K tokens can't hold exact 5-digit codes. The hybrid
  doesn't help: its high-*norm* exact tokens rarely include the queried key.
- The 1B smoke (BUG 2/2 vs SnapKV 0/2) overstated the case — it was unmatched memory
  (BUG pricier) + a weak model + 2 trials.

**Honest verdict:** BUG does **not** cleanly win retrieval either. It beats the
weaker eviction method (SnapKV) at shorter/pricier settings, but ExpectedAttention
retrieves better at matched memory, and BUG's precise-fact fidelity *degrades* with
context at aggressive rank — the opposite of the perplexity amortization benefit.

## Week-5 final scorecard (all honest)
| attempt | verdict |
|---|---|
| 4.5 integrator ablation | BUG best (rank-adaptivity); PSI "blow-up" retracted |
| Axis C amortization (perplexity) | memory amortizes; BUG does **not** pass eviction (U-shaped, closest mid-ctx) |
| Hybrid, fp16-exact (perplexity) | dominated by plain BUG |
| Hybrid, **4-bit-exact** (perplexity) | near-miss (+0.04 at 8B ctx 8192); still behind SnapKV at low mem |
| Single-needle retrieval | saturates at 8B (inconclusive) |
| RULER multi-key retrieval | **mixed** — SnapKV collapses, but EA wins and beats BUG at matched mem at long ctx |

**Bottom line (Week 5, unchanged from Week 4 and now stress-tested):** BUG is a
competitive, well-validated low-rank KV compressor. Across four distinct attempts to
find a decisive win — long-context amortization, a hybrid press, and two retrieval
tasks — none produced a clean BUG-beats-the-field result; the strongest is the
4-bit-exact hybrid's near-tie at mid-context. Its honest, defensible edges remain:
**streaming/online rank-adaptivity** (the untested-at-scale deployment niche),
graceful degradation, and near-oracle tracking — not a headline perplexity or
retrieval win over ExpectedAttention. Reported straight.

---

## Follow-up 3: clean fp16 (no TurboQuant) at long context — decisive negative

**Motivation.** Every prior comparison composed each mechanism with 4-bit TurboQuant;
that quant floor made eviction near-lossless and confounded low-rank-vs-eviction.
This run strips quant entirely (all fp16): fp16 eviction can only keep a *fraction*
of tokens (`keep_frac`), while fp16 BUG keeps a rank-`r` *summary of every* token at
the same memory (`r/n_features`). Hypothesis: without quant, BUG's "summarize
everything" finally beats eviction's "keep a fraction". 8B, ctx 32K (+64K attempted),
WikiText-103 (streamed, 2.7M tokens, 8 windows), pure BUG / BUG-hybrid / SnapKV /
ExpectedAttention, matched memory. `scripts/w5_fp16_longctx.py`.

**Result — the hypothesis is FALSIFIED; eviction wins decisively (a stronger negative
than the quantized runs).** 8B ctx 32768, ppl delta over baseline (7.632):

| memory | BUG (fp16) | SnapKV (fp16) | ExpectedAttn (fp16) |
|---:|---:|---:|---:|
| ~0.06x | r64 **+2.520** | keep-6% **+0.159** | +0.220 |
| ~0.125x | r128 **+0.953** | keep-12% **+0.083** | +0.083 |
| ~0.25x | r256 **+0.321** | keep-25% **+0.030** | +0.045 |

Eviction sits **flat near baseline across the entire frontier** (even keeping only
**6% of tokens** -- ~2K of 32K -- costs +0.16 ppl), while fp16 BUG climbs steeply as
memory drops. Min frontier-gap BUG-SnapKV = **+0.33**, hybrid-SnapKV = **+0.41** (both
positive, BUG behind everywhere); the gap widens to **+2.36** at aggressive memory.
Removing quant did **not** help BUG -- it made eviction look *better* (keeping a few
whole tokens preserves WikiText perplexity almost perfectly, whereas a rank-64
summary of all 32K tokens is very lossy). `figures/week5/fp16_longctx_8b.png`.

**Not captured (infra):** (1) **ctx 64K OOM'd** on the 48 GB card (8B fp16 + BUG's
on-GPU SVD needed ~15 GB more than free) -- no 64K perplexity. (2) **fp16 RULER
retrieval stalled** on the cusolver SVD slowdown (even at block_size=512) and did not
finish; the retrieval question stands at the earlier RULER verdict (mixed -- SnapKV
drops keys but ExpectedAttention is competitive/wins at matched memory). Both are GPU
limitations, not results; a bigger card (80 GB) + a CPU/gesvd SVD fallback would be
needed to complete them.

## Axis B — decode-time streaming BUG: the capability, built and validated

**What was built** (design note: `docs/notes/streaming-decode-design.md`). The
one un-measured structural edge after the negatives above is BUG's *streaming*
nature: constant-memory **decode**, where the tracked subspace advances per
generated token. Two new `transformers`-5.8 `Cache` subclasses:

- **`BugStreamingCache`** (`src/kvdlra/cache/bug_cache.py`) — per layer: the 4
  attention sinks + a recent ring of `w` tokens **verbatim**, and the middle as
  BUG state: basis `U` (n×r, tracked on exactly un-rotated **pre-RoPE** keys via
  the model's own rotary embedding), square-root core `B`, and up to `W`
  per-token **coordinate** columns (r floats/token instead of n). Absorption =
  one augmented rank-adaptive BUG step per graduating block (the factored
  `augmented_bug_step`, one source of truth with the validated blocked tracker
  — extraction also fixed the latent `rank+block>n` degeneracy found in 4.5);
  held coordinates are carried across basis updates by `rot = U_newᵀU_old`
  (each truncation projects old tokens onto the new subspace — the graceful-
  degradation mechanism the DLRA bound governs). Oldest coordinates are dropped
  at the cap: that is the honest bound — softmax attention needs per-token state
  for every attendable token, so "constant memory" can only mean bounding the
  attended set; BUG's version keeps a ~`n/r`× **longer but only approximately
  represented** history at the same budget. `rank=0` degenerates to StreamingLLM
  (sinks+window) — one implementation, two methods.
- **`MorphKVCache`** (`src/kvdlra/cache/morph_cache.py`) — a faithful-core
  reimplementation of MorphKV (ICML'25, arXiv:2503.00979; kvpress 0.5.1 has
  none): R recent + top-C distant tokens, re-ranked per decode step by sum/max
  fusion of the last R (exactly recomputed) attention rows, per-KV-head, GQA
  aggregation per the paper. Documented deviations: prompt pruned at prefill
  end; the score buffer's memory is **counted**; `evict_interval>1` gives the
  paper's coarse-grained variant, which doubles as the **SnapKV-style decode
  eviction** baseline (kvpress's own `DecodingPress`+SnapKV was rejected: under
  transformers 5.8 it rotates all buffered window queries with the current
  1-token `position_embeddings` — a silent scoring bias).

Both keep `get_seq_length()` cumulative (true positions keep advancing) and
report mask sizes equal to the returned K/V length; 25 new tests
(`test_bug_cache.py`, `test_morph_cache.py`) pin teacher-forced logit parity
with `DynamicCache` under sdpa **and** eager when nothing is truncated
(MorphKV bitwise; BUG to fp tolerance — its middle passes through an fp32
un-rotate/re-rotate round trip), mask consistency at every step, constant
stored memory over hundreds of steps, the coordinate-carry invariant, and the
eviction mechanics on crafted scores.

**1B validation** (`scripts/w5_decode_validate.py`,
`results/w5-decode-validate-1b.json`, `figures/week5/decode_validate_1b.png`;
CPU fp32, 2 long prompts, 160 greedy tokens): (1) *parity*: exact-config BUG is
byte-identical to the full cache on one prompt and flips a single near-tie at
token 71 on the other, continuing coherently (fp-tolerance, as designed);
MorphKV at full capacity is byte-identical on both. (2) *memory*: measured
stored floats are flat (bounded sawtooth) for the lossy BUG configs and
StreamingLLM, flatten at capacity for MorphKV, linear for the full cache.
(3) *latency*: bounded, +10–30 % over the full cache on CPU at these lengths.
Notable qualitative datum: at ~equal memory, **StreamingLLM (w=64) collapses
into a degenerate `assistantassistant…` loop at token 7** on the relativity
prompt while **BUG r=32 (less stored memory) stays coherent and factual** — the
rank-32 summary of the dropped middle carries real information. Not yet a
benchmark, but the first direct evidence the streaming low-rank middle does
useful work during decode.

**Benchmark** (`scripts/w5_streamppl.py`): teacher-forced *streaming
perplexity* — prefill P tokens, then feed G ≫ budget tokens one per forward
(exactly the decode regime), scoring each step against the bounded cache;
per-position-bin curves give the degradation *slope*, the falsifiable
graceful-degradation claim. Methods at matched worst-case stored floats (BUG
budget drives MorphKV capacity — score buffer included — and the StreamingLLM
window): full cache (upper bound), BUG, MorphKV, SnapKV-decode (periodic
variant), StreamingLLM.

**Results (complete — full analysis in `docs/week6.md`).** 1B (G=3072, ~515
tok-eq): full 10.13 | **BUG 11.59** | MorphKV 11.81 | SnapKV-dec 11.83 | sllm
14.47 — **BUG wins**. 8B (G=8192): tier 1 (~499 tok-eq) MorphKV/SnapKV 7.74 <
BUG 7.87 < sllm 8.20; tier 2 (~183 tok-eq) MorphKV 8.39 < BUG 9.17. The bin
curves explain the flip: BUG is near-lossless (1.03×) while generation is
within ~2× its budget — the best method there at both scales — then its
penalty grows, while adaptive eviction holds a flat offset; deep horizons
reward the flat profile. The graceful-degradation hypothesis is **inverted**;
verdict and mechanisms in `docs/week6.md`, fixes planned in
`docs/week7-plan.md`.

## Week-5 final scorecard (updated, all honest)
| attempt | verdict |
|---|---|
| 4.5 integrator ablation | BUG best (rank-adaptivity); PSI "blow-up" retracted |
| Axis C amortization (ppl, quant) | memory amortizes; BUG does not pass eviction (U-shaped) |
| Hybrid fp16-exact / 4-bit-exact (ppl) | dominated / near-miss; never passes SnapKV |
| Single-needle retrieval | saturates at 8B (inconclusive) |
| RULER multi-key retrieval (quant) | mixed -- SnapKV collapses, EA wins at matched mem |
| **Clean fp16 long-ctx (ppl)** | **decisive negative -- eviction wins the whole frontier at 32K** |

**Bottom line, now thoroughly stress-tested:** across five distinct attempts
(amortization, hybrid x2, two retrieval tasks, and the clean fp16 frontier), streaming
BUG **does not beat SnapKV/ExpectedAttention** on long-context perplexity or retrieval
at matched memory -- and the quant-free test makes the perplexity gap *clearer*, not
smaller. Eviction's "keep a few whole tokens" is simply a better fit for LM perplexity
than "a low-rank summary of everything". BUG's honest, defensible edges remain
narrow and specific: streaming/online **rank-adaptivity** (the constant-memory
decode niche, still the one un-measured place its online nature is a structural
advantage), graceful degradation, and near-oracle low-rank tracking. Reported straight.
