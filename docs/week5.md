# Week 5 — placing streaming BUG in the low-rank field (in progress)

Plan: `docs/week5-plan.md`. This file records results as they land. Priority order
(plan §"Priority"): **4.5 integrator ablation → Axis C long-context (Exp A) → Axis
A (Palu/LoRC) → streaming-decode + Axis B → Exp B RULER.**

---

## Week 4.5 — DLRA integrator ablation (DONE)

**Question.** Before BUG enters the expensive Week-5 comparisons, is it the right
DLRA integrator for the streaming-KV subspace-tracking problem, or does a sibling
integrator track better / cost less? Offline bake-off — reconstruction error vs
the truncated-SVD **oracle** plus per-sweep **cost**, on captured pre-RoPE KV. **No
LLM, no pod.**

**Verdict — ship plain BUG (the incumbent).** BUG is the *only* candidate that
both hugs the oracle in the KV operating range **and** stays robust when pushed to
high rank. Projector-splitting (PSI) ties BUG cheaply at low rank but
**destabilizes at high rank** (its backward step); parallel-BUG is uniformly less
accurate (its decoupling drops a cross term that matters on heavy-tailed KV). This
empirically reproduces the numerical-analysis reason the project chose the
unconventional/BUG integrator in the first place, on real KV.

### What was built
Three integrators run through **one** block-streaming harness (identical
first-block seed, identical rank truncation, identical
`‖M − U Uᵀ M‖_F / ‖M‖_F` metric — which depends only on `span(U)`, so the
comparison is exactly apples-to-apples):

- **BUG** — `integrators/streaming_torch.py::blocked_bug_subspace` (the incumbent):
  augmented Galerkin, **forward** S-step, **square-root** core `[[B, A],[0, R]]`
  (never forms `B Bᵀ`). Ceruti–Kusch–Lubich 2022 (arXiv:2104.05247).
- **PSI** — `integrators/streaming_variants.py::psi_step` / `psi_subspace` (new):
  projector-splitting with a **backward** S-substep, in its natural
  **covariance-core** form (`S = B Bᵀ`, squaring the spectrum). Lubich–Oseledets
  2014 (arXiv:1301.1058).
- **Parallel-BUG** — `integrators/streaming_variants.py::parallel_bug_step` /
  `parallel_bug_subspace` (new): the parallel rank-adaptive integrator
  (Ceruti–Kusch–Lubich 2024, arXiv:2304.05660) specialized to the streaming
  increment — the in-subspace rotation and the new-direction admission are computed
  **independently** (an `r×(r+b)` SVD + a `b×b` SVD, dropping the
  in-basis↔new-direction cross-coupling that BUG's joint `(r+b)×(r+b)` SVD keeps).

Harness/figure/verdict: `scripts/w45_integrator_ablation.py` →
`figures/week5/integrator_ablation.{png,pdf,json}`. Tests:
`tests/test_streaming_variants.py` (13: block=T ⇒ oracle, exact rank-r recovery,
orthonormal + rank-capped basis, operating-range accuracy, guards).

### Results (5 docs, pre-RoPE Layer-8 K, block 128, fp64 core)

Relative reconstruction error (mean over docs); **oracle** = Eckart–Young rank-r
lower bound. "err/oracle" is the gap to optimal (1.000 = optimal).

| rank | oracle | BUG | BUG/orc | PSI | PSI/orc | parallel | par/orc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16  | 0.2996 | 0.3012 | **1.005** | 0.3021 | 1.008 | 0.3538 | 1.181 |
| 32  | 0.2290 | 0.2309 | **1.008** | 0.2316 | 1.011 | 0.2807 | 1.226 |
| 64  | 0.1567 | 0.1578 | **1.007** | 0.1583 | 1.010 | 0.2053 | 1.310 |
| 128 | 0.0980 | 0.0992 | **1.012** | 0.0997 | 1.017 | 0.1359 | 1.387 |
| 192 | 0.0677 | 0.0687 | **1.015** | 0.0997 | 1.473 | 0.0997 | 1.473 |
| 256 | 0.0472 | 0.0480 | **1.018** | 0.0997 | 2.113 | 0.0745 | 1.578 |
| 384 | 0.0203 | 0.0207 | **1.021** | 0.0997 | 4.918 | 0.0377 | 1.858 |

- **BUG hugs the oracle everywhere: ≤ 1.021× across the whole rank range.** Robust.
- **PSI ties BUG within ~2% through r=128** (the KV operating band) but its error
  **floors at ~0.0997 and never improves** past r≈160 — it cannot track the smaller
  singular directions. This is the projector-splitting **backward step** amplifying
  error on small singular values (compounded by the covariance-core squaring):
  err/oracle climbs 1.02 → 1.47 → 2.11 → **4.92**. Exactly the σ_min-sensitivity
  the BUG family was designed to avoid.
- **Parallel-BUG is uniformly worse (1.18–1.86× oracle).** The gap is the
  in-basis↔new-direction cross-coupling it decouples away; it shrinks monotonically
  as `block_size → T` (fewer decoupling steps: 1.39× at bs=64 → 1.18× at bs=512 for
  r=64), confirming the decoupling is the cause. Its per-block small SVDs are also
  less robust than BUG's single joint SVD (LAPACK non-convergence on large
  ill-conditioned residual blocks).

### Cost (one streaming sweep, ms, dev CPU, largest doc)

| rank | BUG | PSI | parallel-BUG |
|---:|---:|---:|---:|
| 64  | 211 | **128** | 179 |
| 128 | 383 | **190** | 265 |
| 256 | 753 | **188** | 566 |
| 384 | 1572 | **201** | 1033 |

PSI is cheapest and near-flat in rank (fixed-rank `(r+k)×r` QRs, no growing
augmented SVD); BUG's joint `(r+b)×(r+b)` SVD makes it the priciest at high rank.
But **the speed does not buy anything**: PSI's cheapness comes with its high-rank
blow-up, and parallel-BUG is cheaper than BUG only where it is also 1.4–1.9× less
accurate. There is **no free lunch** here — the plan's hoped-for "parallel-BUG =
BUG accuracy at lower cost" does not hold on real KV. In the operating range
(r ≤ 128) BUG's sweep is sub-second regardless.

### Decision (plan §4.5 rule)
A rival displaces BUG only if it **matches** BUG in the operating range (r ≤ 128)
**and** stays robust across the whole range (max err/oracle ≤ 1.05). PSI matches in
the operating range but is not robust (4.92× at r=384); parallel-BUG is neither.
**Winner = BUG.** Since the winner is the incumbent, the "wire the winner into the
streaming path behind a flag" step (plan §4.5b) is a no-op: the validated BUG step
remains the default, and PSI/parallel-BUG stay available in
`integrators/streaming_variants.py` as documented, tested reference
implementations. **Honest framing:** the ablation is a *rigor* result, not a
competitive lever (all three ≈ oracle in the useful band); its value is confirming,
on real KV, that BUG's forward square-root design is the right choice — and ruling
out the "a cheaper parallel integrator gets us the streaming-decode niche for free"
shortcut.

---

## Axis C / Experiment A — long-context scaling (TODO, needs GPU pod)
The key open question (amortization test): does BUG's fair perplexity–memory
frontier pass SnapKV's as context grows (ctx {1K…32K}, 8B)? Blocked on a GPU pod +
a **rotated** HF token / vast.ai key (the old ones were passed to rented pods — see
`[[vastai-pod-flakiness-jul2026]]`). Recipe: onstart-batch + HF token for fast
download + results via `vastai logs`, never post-hoc SSH.
