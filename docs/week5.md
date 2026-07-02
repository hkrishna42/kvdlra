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

## Axis C / Experiment A — long-context scaling (harness built; run pending pod)
The key open question (amortization test): does BUG's fair perplexity–memory
frontier pass SnapKV's as context grows (ctx {1K…32K}, 8B)? Harness:
`scripts/w5_longctx.py` — loops context lengths, redoes the fair
BUG/SnapKV/EA×TurboQuant comparison at each (position-fair; reuses
`perplexity_sweep`'s explicit `position_ids`), interpolates each frontier at fixed
memory budgets, and plots the **crossover vs T** + per-ctx frontier grid. Emits the
full results JSON to stdout (`===W5_LONGCTX_JSON_BEGIN/END===`) so it survives in
`vastai logs` (never post-hoc SSH — `[[vastai-pod-flakiness-jul2026]]`), and runs
each ctx in a try/except so a 32K OOM still yields shorter-ctx results. Smoke-tested
at 1B/CPU; the real run is 8B on a GPU pod (bf16, step ctx up gradually, watch OOM).
