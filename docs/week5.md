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
