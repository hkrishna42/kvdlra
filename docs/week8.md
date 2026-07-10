# Week 8 — CodeBUG: amortized PQ-codebook coordinates vs eviction at extreme compression

> **STATUS: in progress.** Design locked (Phase-0 panel complete); implementation
> done and green; calibration + gating experiment + 1B ppl sweep running. Results
> sections marked _TBD_ are filled as runs land. Ethos unchanged: report numbers
> straight, count ALL memory, prefer honest negatives.

## The single falsifiable number

At the **aggressive** budget (~89 tok-eq/layer = 91,136 floats, 1B, WikiText-2,
G=1024), does an amortized codebook beat whole-token eviction at matched memory?

- Target: aggregate streaming ppl **< 18.71** (morph). Best prior BUG/SLASH 20.15.
  Full cache 7.24. Then confirm 8B tier-2 **< 8.39**.
- Reported either way. A clean "codebook doesn't close it" completes the
  dominance map (`docs/week7-dominance.md`).

## Phase-0 design panel — the carry solution (unanimous)

Three independent lenses each fully specified and stress-tested one requant-carry
solution against variant D's 14.7× compounding **and** the anchor-affordability
constraint, each running real experiments against the repo's own
`augmented_bug_step`/`PolarQuant`. All three converged on **anchor-basis
freezing**:

- **Lens C (rotation-robust) — rigorous NEGATIVE.** Measured `rot` from the
  integrator is a general dense SO(24) rotation (`‖rot−I‖∞≈2`, sometimes
  non-square as rank grows) — in no finite group, so no lattice/permutation code
  is closed under it (exact-update codes dead). A norm-only rotation-invariant
  code reconstructs at relerr 1.39 (worse than zero — direction is what attention
  needs). Fixed-frame coding **reduces to anchoring**. Contributes the rule A
  adopts: *code each column once against the basis live at graduation, never
  re-express*.
- **Lens B (residual/delta) — rigorous NEGATIVE.** Proved the algebraic identity
  `recon_B ≡ recon_A` (3.4e-14): delta coding's only bounded, constant-memory,
  non-compounding form **is** anchor-freezing (`U_k·rot_k…rot_2 = U_ref`). Every
  delta that does new work per absorb either needs the discarded raw token,
  collapses to the plain fp32 cache, or folds→recodes and **compounds** (measured
  3.3–4.6×). One keeper: a one-shot **residual-VQ base** (finer 2-stage code at
  graduation, rotated losslessly) — an enhancement to A, not a standalone carry.
- **Lens A (anchor-freezing) — the design.** One **frozen reduced-rank anchor**
  `U_ref` (rank `r_a`), sealed after a short decode warmup; the live adaptive
  basis (rank `r`) is kept for the recent/fp32 tier, so recent tokens retain
  near-oracle tracking and only the **old coded tail** (lowest attention mass)
  lives in the coarse frozen anchor. This sidesteps Lens C's "frozen frame is
  worst for recent tokens" failure by construction. Non-compounding confidence
  ~93–97%; **beats-morph confidence ~45%** (the frozen subspace may be "flat but
  mediocre").

**The load-bearing honesty finding (Lens B).** The anchor basis is **not free**:
`U_ref` is `n·r_a` per stream. At r=24 the fixed overhead is ~84% of the budget;
one full-rank extra anchor (24,576 fl) does **not** fit, so `r_a` must be small
(e.g. 8 → 8,192 fl). Counting the anchor honestly, coded coverage is **hundreds
to ~1,000** tokens, *not* the plan's optimistic ~3,250 (which omitted the
anchor). Still ≫ morph's 39 exact — the bet survives, but tighter than pitched.
The two-gate experiment (below) decides it on real data before any ppl number.

## The technique — CodeBUG (`coord_codebook`/`anchor_rank`/`code_budget` mode)

`BugStreamingCache` gains a CodeBUG mode reusing the existing second-tier
machinery. On each absorb the live basis advances as usual and the fp32 tier is
carried by `rot @ C`; the **coded tier is never rotated**. When an fp32 column is
evicted (by the retention rule) it is re-expressed **once** into the frozen anchor
(`c_ref = U_ref^T (U_live @ c)`), product-quantized, and appended; it is inert
thereafter. Attention reconstructs the coded tail as `U_ref @ decode(codes)` (the
structural inverse of variant D's `_rotate_quant_tier`). Honest accounting counts
the frozen anchor, codes at `bits/32`, per-column norms, positions/scores, and the
shared PQ codebook **once** per cache (`codebook_numel`).

Calibration (`scripts/calibrate_codebook.py`): the PQ is fit on the true
deployment distribution — the anchor-frame coords the cache actually codes,
captured by streaming the real cache over **WikiText-103 train** (disjoint from
the WikiText-2 **test** eval by the standard article split; `pg19` available as a
stricter cross-corpus check). Model-specific, data-generic side info, counted once.

## What is built (all green: 192 pass / 1 skip, ruff + mypy clean)

- `src/kvdlra/quant/product_quant.py` — `ProductQuantizer` (fit/encode/decode,
  PolarQuant-parity, `codebook_numel`); 13 tests.
- CodeBUG mode on `src/kvdlra/cache/bug_cache.py` (frozen anchor, code-once,
  never-rotate; honest accounting); 11 tests including the structural
  **non-compounding proof** (codes bit-stable across absorbs; `_rotate_quant_tier`
  provably never called in coded mode) and codebook-counted-once.
- `scripts/calibrate_codebook.py`, `scripts/w8_carry_drift.py` (the gating
  experiment), `scripts/w7_codebook.py` (the matched-memory ppl sweep).

## Results

### Calibration
71,424 anchor coords captured by streaming the real cache over WikiText-103 train;
PQ fit rel-distortion **0.0015** (near-lossless), codebook counted at 2,048 floats
(K·dim = 256·8) once per cache. So the PQ is *not* the bottleneck — the
frozen-anchor **subspace** projection is (confirmed below).

### Gate 1+2 — carry drift + accuracy (`w8_carry_drift.py`, 1B, doc0, G=1024)

Per coded column at readout, feature-space reconstruction error vs the token's
rank-r truth, binned by position. Two arms (17,856 coded columns each):

| arm | mean err | tail/head | err − floor (carry cost) |
|---|---:|---:|---:|
| **ON (frozen anchor)** | **0.272** | **1.38×** | **~0.003** (negligible) |
| OFF (recode every absorb) | 0.299 | 0.85× | ~0.07 |

- **Non-compounding CONFIRMED.** ON error rises only 1.38× over the full horizon,
  and `err ≈ floor` at every position (e.g. 0.246 vs 0.243 at pos 512; 0.304 vs
  0.301 at pos 1280) — the PQ carry adds **~0.003**. Freezing eliminates the carry
  noise exactly as designed; the mild rise is the subspace **floor** rising with
  position (staleness of the frozen anchor), not compounding.
- **The binding constraint is the subspace floor, not the quantizer.** Mean recon
  error is **27%**, essentially all the `(I − P_ref)` residual of the coarse r_a=8
  anchor. The PQ is near-lossless (calibration distortion 0.0015); **residual-VQ
  would not help** — the loss is subspace, not quantization. The only fidelity
  lever is `anchor_rank` (r_a↔coverage, the Week-7 water-filling trade reborn).
- **Freezing beats recoding** on total error (ON 0.272 < OFF 0.299): re-anchoring
  the OFF control lowers its floor (0.238) but the recode adds PQ noise, netting
  worse. Note the OFF control re-anchors, so it does not reproduce the scalar
  variant-D 14.7× blow-up (that is the Week-7 result, same machinery, coarser
  scalar carry) — the `pass=False` flag is a mis-specified OFF criterion, not a
  design failure. The load-bearing claim (frozen ON is non-compounding and
  floor-bound) holds.

### The single number — 1B matched memory (`w7_codebook.py`, doc0, ~89 tok-eq)

**CodeBUG does NOT beat eviction. Honest negative.** (mem_max ≤ 1,458,176 = budget
× 16 layers for every variant; morph 18.71 / bugA 20.60 / slash 20.04 reproduce
the Week-7 priors exactly.)

| method | ppl | coverage |
|---|---:|---|
| full | 7.24 | all |
| **morph (eviction)** | **18.71** | 39 exact |
| slash-h4 | 20.04 | 4 exact + 202 low-rank |
| bugA-r24 | 20.60 | 284 low-rank |
| **CodeBUG cb64** (best) | **22.43** | 64 fp32 + 453 coded |
| CodeBUG cb32 | 23.52 | 32 fp32 + 720 coded |
| CodeBUG cb16 | 24.26 | 16 fp32 + 853 coded |

**The kill mechanism (clean, bug-free).** CodeBUG is **monotone worse the more it
codes** (24.26 → 23.52 → 22.43 as coded coverage 853 → 453 shrinks), smoothly
converging toward bugA (20.60) as codes → 0. So the r_a=8 coded tokens are
*strictly worse* than rank-24 fp32 coords: replacing a sharp coord (~10% error)
with a coarse code (27% error, per the drift floor) trades quality for coverage,
and at the aggressive budget the trade loses. Adding coverage does not help
because the added tokens are too lossy.

**No `anchor_rank` rescues it (analytical + memory arithmetic).** The anchor costs
`2n·r_a`. At r_a=8 (the coverage-max) fidelity is too low (loses, above). At
r_a≥12 the anchor + fixed overhead leaves coverage *below bugA's 284* (r_a=12,
cb16: W_code≈160, total ≈176 < 284; r_a=16, cb16: fixed alone > budget), and bugA
already loses to morph — so no r_a both fits and out-covers bugA at adequate
fidelity. This is exactly the dominance-program's **structural basis-overhead
floor**: the codebook cannot escape it because it introduces *its own* basis (the
anchor). Eviction pays no such overhead and spends the whole budget on exact
tokens.

**Robustness (2-doc geo-mean aggregate, doc0+doc1) — negative holds:**

| method | doc0 | doc1 | geo-agg |
|---|---:|---:|---:|
| full | 7.24 | 9.70 | 8.38 |
| **morph** | 18.71 | 15.31 | **16.93** |
| slash-h4 | 20.04 | 18.15 | 19.07 |
| bugA-r24 | 20.60 | 20.91 | 20.76 |
| **CodeBUG cb64** (best) | 22.43 | 20.85 | **21.63** |
| CodeBUG cb32 | 23.52 | 21.61 | 22.54 |
| CodeBUG cb16 | 24.26 | 21.20 | 22.68 |

CodeBUG is the **worst** compression method on the aggregate (21.63), losing to
morph by 4.7 ppl, to slash by 2.6, and to bugA by 0.9. "Least coding wins" holds
on both docs (cb64 < cb32, cb16). The negative is doc-robust — as expected, since
it rests on the intrinsic rank-8 floor and the monotone-coverage mechanism,
neither of which is doc-specific.

### Adversarial verification (Phase 3) — 3/3 skeptics confirm the negative

Three independent skeptics each attacked a different axis, trying to *rescue*
CodeBUG. All three failed and confirmed the negative:

- **Accounting fairness.** Independently reproduced `stored_state_numel` **to the
  byte** (diff 0 vs measured `mem_max`) for every variant; morph is fairly (even
  conservatively) charged. CodeBUG's *ceiling* (W_code→0, all fp32) *is* bugA =
  20.60, which already exceeds morph while using *more* memory — so the gap is not
  memory starvation, and freeing memory for coverage moves the wrong way.
- **Implementation + configs.** Verified the re-expression `c_ref =
  u_ref^T(u_live @ c)` equals the exact orthogonal projection (to 1e-6) and the
  reconstruction is correct — no crippling bug. Ranked every untried config
  (end-of-prefill seal, raw-PQ, higher bits/subspaces, SLASH hedge, per-layer
  codebooks, lower rank r=16, higher r_a); all ≤6% to beat morph.
- **The 27% floor is intrinsic to rank-8.** It is `||(I−P_ref)x_true||` for
  `x_true` the rank-24 reconstruction and `P_ref` rank-8 — the rank-8-in-rank-24
  residual, grounded in the committed Week-1/2 spectra (pre-RoPE rank-8-in-rank-24
  ~0.23–0.31; measured 0.243 sits at the floor). Staleness is worth only ~0.03
  (a fresh anchor every absorb still floors at 0.244), PQ only ~0.003. Even a
  *perfect* rank-8 anchor loses.

**The convergent proof (all three, one sentence):** morph keeps 39 **exact
full-512-dim** tokens; CodeBUG's are two stacked lossy projections (rank-24 →
rank-8 anchor → PQ); bugA has strictly-sharper tokens *and* 7× the coverage and
still loses by 1.9 ppl — so "more, blurrier tokens" can never cross "39 exact."
Skeptic-proposed falsifiers (all predicted to lose at ~2–6%, commands preserved
for reproducibility, not run because the monotone-coverage trend + intrinsic floor
+ bugA ceiling already settle them):

```
# end-of-prefill seal (tests the staleness ceiling; predicted ppl ~22.0–22.4)
uv run python scripts/w7_codebook.py --anchor-seal-absorbs 30 ... (else identical)
# raw PQ, more coverage (predicted WORSE, coverage is the wrong lever)
uv run python scripts/calibrate_codebook.py --no-normalize ...; scripts/w7_codebook.py ...
# lower live rank r=16 to free basis budget (predicted ~22; r=16<r=24 per Week-7 sweep)
uv run python scripts/calibrate_codebook.py --rank 16 ...; scripts/w7_codebook.py --rank 16 ...
```

### 8B tier-2 confirmation — NOT RUN (gated, correctly)

The kickoff gates 8B on the 1B result *clearing* morph, which it did not. The kill
mechanism is structural (anchor overhead + intrinsic rank-`r_a` floor) and
scale-invariant; at 8B tier-2 eviction already beats BUG (morph 8.39 < bugA 8.89,
`docs/week7-dominance.md`), so CodeBUG would lose there for the same reason.
Spending a vast.ai pod (credit ~$4.69, keys still unrotated) to confirm a
predicted loss is unjustified. No 8B run.

## Verdict

**CodeBUG does not beat eviction at extreme compression. Honest negative — the
codebook fork, the last frontier candidate, does not change the frontier.** This
*confirms* (does not contradict) the dominance-program's two measured walls: the
near-oracle tracking ceiling and, decisively here, the **structural
basis-overhead floor**. The codebook successfully amortizes the per-token
*coordinate* cost (a coded token really does cost a few bits, non-compounding,
honestly counted) — but it cannot amortize the per-stream **basis** cost, because
coding coordinates requires a subspace to code them in (the anchor), and that
anchor is the very overhead eviction never pays. Forced small enough to fit
(r_a≤8), the anchor is too coarse (27% floor); made sharp enough to matter
(r_a→24), a code costs as much as an fp32 coord and coverage collapses to bugA —
which already loses. There is no interior point that wins.

The Week-8 map is complete: **BUG is a near-oracle, correct, constant-memory
streaming cache that decisively wins the moderate-compression regime and loses the
extreme-compression regime; no tuning-level change, and now no codebook, moves
that line.** The result is reported straight, either way, exactly as promised.
