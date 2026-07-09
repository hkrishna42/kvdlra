# Week 8 plan — CodeBUG: amortized codebook coordinates to beat eviction at extreme compression

> **Decision (post-dominance-program).** `docs/week7-dominance.md` mapped every
> tuning-level lever and found two *measured* walls: a **near-oracle tracking
> ceiling** (BUG beats Frequent Directions / is within ~1% of the oracle, so no
> integrator swap helps) and a **structural basis-overhead floor** (at extreme
> compression BUG's basis+core is ~85% of the budget, which whole-token eviction
> doesn't pay). The *only* lever that changes the frontier is **amortizing the
> per-token cost with a shared codebook**. This is that direction, committed.

## The goal (falsifiable, single number)

Beat whole-token eviction (MorphKV) at the **aggressive** memory budget, at
honestly matched memory:

- **1B, ~89 tok-eq/layer, WikiText-2, G=1024:** target aggregate streaming ppl
  **< 18.71** (morph). Best BUG so far there: 20.15 (SLASH). Full cache 7.24.
- **8B, ~183 tok-eq/layer, WikiText-103, G=8192 (tier-2):** target **< 8.39**
  (morph). Best BUG so far: 8.89 (bugA). Full 7.22.

If a CodeBUG variant clears these at matched memory, it is the first BUG cache
that *dominates* eviction across both regimes (it already wins moderate budgets:
bugA 10.62 vs morph 11.81 at 1B). If it does not, that is the honest ceiling of
low-rank + codebook, reported straight.

## Why a codebook can break the floor (the arithmetic)

At ~89 tok-eq (91 136 floats/layer) the aggressive budget spends ~78 000 floats
(85%) on the fixed basis+core+ring+sinks, leaving ~13 000 for token coverage. In
fp32 coordinates at r=24 that buys **W≈274** tokens (48 floats/token). Replace
each token's `r` fp32 coordinates with a **product-quantized code** — e.g. M=8
subspaces × 8 bits = 8 bytes = **2 float-equivalents per token** (K), 4 for K+V —
and the same ~13 000 floats buy **W≈3 250** tokens: **~12× more coverage**.
Whole-token eviction fits 39 exact tokens; if the PQ reconstruction of ~3 000
approximate tokens is good enough, that trade can win. That is the entire bet.

The codebook itself (K×r floats) is **shared side-information amortized over
every token and every layer**, so its per-token marginal cost → 0 — exactly the
property (zero un-amortized per-token overhead) that lets eviction win the
extreme regime and that BUG's per-layer basis lacks.

## The technique — `CodeBugCache` (BUG + shared PQ coordinates)

Keep BUG's near-oracle adaptive subspace; replace the *storage* of the per-token
coordinates with codes into a shared, calibrated **product-quantization (PQ)**
codebook.

1. **Subspace:** unchanged — BUG tracks the adaptive rank-`r` basis `U` (its
   validated, near-oracle role). `U` is still stored per layer (the diagonal
   core `B` at `r`, per the Week-7 fix).
2. **Coordinate coding:** the `r`-dim coordinate column `C[:,s]` is split into
   `M` subvectors; each is quantized to one of `K` centroids from a per-subspace
   codebook. Token `s` → `M` codes (`M·log₂K` bits). Codebook = `M·K·(r/M) = K·r`
   floats, **shared across all tokens/layers**, calibrated once.
3. **Calibration:** fit the PQ codebooks offline (k-means per subspace) on BUG
   coordinate columns captured from a held-out corpus (reuse `capture_kv.py` +
   `StreamingBUG`). This breaks the strict training-free stance — the honest,
   accepted cost (same as PolarQuant's Π, KIVI's per-channel stats). Codebooks
   are model-specific, data-generic; ship them as calibrated side info.

## The load-bearing research risk — the requant-carry problem (variant D's killer)

BUG's basis `U` drifts (topic drift), so held coordinates are re-expressed every
absorb (`C ← rot·C`). If coordinates are stored as **codes**, re-expressing means
decode → rotate → re-encode → **compounding quantization noise every absorb** —
this is exactly what made Week-7 variant D (scalar PolarQuant coordinates)
catastrophic (tail exploded 14.7×). Solving this is the crux; candidate designs
for the multi-agent build to evaluate:

- **(A) Anchor-basis freezing (recommended default).** Freeze `U` as a reference
  `U_ref` every `F` absorbs; express and code the graduating coordinates in
  `U_ref` **once**, and never rotate coded columns again — reconstruct them as
  `U_ref @ decode(codes)`. New tokens track in the live basis; at the freeze
  horizon they graduate into the current frozen-coded tier. Requantization is
  **once per token at graduation**, not per absorb. Cost: store a small set of
  frozen anchor bases `U_ref` (each `n·r`); count them. This is a "basis pyramid"
  of frozen anchors, each with a block of coded coordinates.
- **(B) Residual/delta coding** of the drift correction rather than re-coding
  from scratch.
- **(C) Rotation-robust coding** (code a rotation-invariant representation).

The non-negotiable verification, given variant D: **measure the reconstruction
drift of coded columns ON vs OFF the chosen carry solution over the full deep
horizon** (G=1024+), and confirm it does *not* compound. A design that
compounds is dead — report and move to the next.

## Honesty guardrails (non-negotiable — same bar as all prior weeks)

- **Count ALL memory.** Codes at `bits/32` float-equivalents; the shared
  codebook counted **once** (per cache if global across layers, else per layer)
  — it must be genuinely amortized or it recreates the Week-4 unfairness. Anchor
  bases counted. Norms/scales counted. The matched-memory audit
  (`mem_max ≤ budget`) must pass for every variant.
- **The codebook is side info, not free.** If a variant "wins" only by not
  counting the codebook, it is not a win.
- **Success is the single number above** (< morph at the aggressive budget),
  reported either way. A clean "codebook doesn't close it" is a fine result and
  completes the map.
- Same harness (`w5_streamppl.py` / `w7_rank_sweep.py`), same matched worst-case
  stored floats, geo-mean bins, 1B locally then confirm the winner at 8B on one
  vast.ai pod (credit ~$4.7).

## Deliverables

- `src/kvdlra/quant/product_quant.py` — `ProductQuantizer` (fit/encode/decode,
  API-parity with `PolarQuant`), + calibration script + captured codebooks.
- `CodeBugCache` (or a `coord_codebook=` mode on `BugStreamingCache`) with the
  chosen carry solution; honest `stored_state_numel`.
- Tests: PQ round-trip error bound, exact-mode parity, mask consistency,
  **carry-drift non-compounding** (the variant-D guard), honest accounting.
- `w7_codebook.py` experiment: CodeBUG vs bugA vs SLASH vs morph vs full at the
  aggressive budget, 1B, matched memory. Adversarial verification of the carry.
- `docs/week8.md` writeup with the single-number verdict.

## The multi-agent launch prompt

See `next-session-prompt.md` (the ready-to-run kickoff) and the orchestration
prompt embedded there. Escalate (don't silently retry) if: the carry solution
compounds despite fixes; the codebook can't be counted honestly; the 1B loop
exceeds ~a day; or the result contradicts the dominance-program walls (re-check
the harness before believing it).
