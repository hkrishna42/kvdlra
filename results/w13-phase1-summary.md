# Week-13 Phase-1 — five $0 CPU probes, all adversarially verified

One Workflow (`w13-phase1-probes`, run wf_6815a24d-f9d): 5 probes on the 1B dumps
(`dumps/llama3.2-1b/*_len4096_rope-both`, 5 docs × up to 16 layers), each re-derived
by an independent skeptic. **Outcome: 1 funded lever (T-B), 4 clean verified
negatives.** No GPU, no spend. Every number traces to a committed script + results
JSON; verifier verdicts are independent re-derivations, not re-runs.

| Track | Verdict | Verifier | Key number |
|---|---|---|---|
| **T-A** integrator surgery | **KILL** | CONFIRMED | deep-horizon erosion premise is false in-proxy |
| **T-B** warm-up retrieval fix | **FUND** (a lean) | CONFIRMED | first-block SLASH bypass real (8/8 static facts) |
| **T-C** long-doc-L | **KILL** | CONFIRMED | LONG≈SHORT L (cosine 0.996), 0/5 clear 3% bar |
| **T-X** layer angles | **KILL** | CONFIRMED | adjacent subspaces ~orthogonal (76° vs 78° null) |
| **T-X** rank-vs-depth | **negative** | CONFIRMED | effective rank does NOT collapse with depth |

## T-A — integrator surgery: KILL (`results/w13-tracka-probe.json`)
Replicated `augmented_bug_step` truncation with 6 variants (raw / Tikhonov ×2μ /
tik_ridge / energy-faithful / energy-rawdata / full-SVD oracle), streamed pre-RoPE
keys, reconstruction error binned by token position (bin_size 128 from
`w9-surprise-sweep-1b.json`), 5 docs × 5 layers. **The deep-horizon erosion premise
fails:** raw deep-bin0 error 0.221 (r32) / 0.099 (r128) is **FLAT** across 32→512
re-truncations (0.2211→0.2212), and raw already reconstructs the oldest bin **+2.8%
(r32) / +4.3% (r128) BETTER than the uniform oracle** — there is no deficit to fix.
No deployable fix passes: `energy_faithful` == raw exactly (proven no-op, since the
coord carry preserves the 2nd moment ⇒ ‖cᵢ‖==σᵢ); `tik_a` (literal spec) makes the
deep WORSE (+6%→+25%, growing with trunc count); `tik_ridge` improves deep only by
regressing the moderate/recent bands +17–31% and is unstable at ≤32-token blocks;
`energy_rawdata` (non-deployable upper bound) moves ≤0.06%. **Verifier CONFIRMED**
by calling the REAL deployed `blocked_bug_subspace` (bit-for-bit match), numpy fp64
SVD, and a Pythagorean-identity binning — all key numbers matched to rounding.
Caveat (both agents): pre-RoPE reconstruction-error proxy, not ppl; len-4096 dumps
emulate the 32K re-truncation COUNT but not the data volume/RoPE range of true 32K,
so long-range data-drift erosion at real 32K is unseen. For a NEGATIVE this runs
conservatively (can't fix a horizon that isn't broken in reconstruction).

## T-B — warm-up retrieval fix: FUND, a lean (`results/w13-trackb-design.md`)
The first ingest chunk hits `_prefill` (`update` routes `cumulative_length==0`→
`_prefill` at :1362 before the ingest check), which absorbs its middle via
`_absorb_columns` at :1442 and never reaches `_absorb_block_slash` (the sole writer
of `hh_k/hh_v/hh_pos`, reachable only via `consolidate`/`_decode_step`). So the first
~4028 tokens (chunk 4096) can never enter the exact tier — exactly Week-12's warm-up
window. **8/8 static trace facts confirmed** (`scripts/w13_trackb_bypass.py`, ast
only). A seed hook (`_seed_hh_from_prefill`) is disjointness-safe and
accounting-neutral (saturated model already assumes `hh` full). **Verifier
CONFIRMED** the trace + accounting, and flagged that a seed under the `hh_enabled`
guard would also fire on single-shot prefill and break the pinned
`test_single_shot_prefill_leaves_hh_empty` — so the implementation gates the seed on
`self._mode=="ingest"` (chunked only; single-shot stays empty by design). Payoff
(does seeding flip early-needle RULER cells at 16K/32K without regressing ppl/wins?)
is the Phase-2 GPU question.

## T-C — long-doc-L: KILL (`results/w13-trackc-probe.json`)
Compared L calibrated on a 160-token prefix vs the full 4096-token queries
(attention-output-error proxy, `w12_qbug_probe` machinery). SHORT↔LONG diagonal-L
cosine **0.9962** — the per-dim query-energy profile is stationary. LONG beats SHORT
by only **+0.22% (r32) / +0.80% (r128)**; **0/5 docs clear the 3% margin bar**. Both
L's clear the ≥3% payoff-vs-plain bar (~30%/44%) — that is the Week-12
"whitening beats plain" effect, independent of calibration length. So an 8B long-doc
recalibration will NOT move the bounded Q-BUG result. **Verifier CONFIRMED** (noted
the probe's free-text `raw_numbers` mis-transcribed short_r32 for 3/5 docs, but the
authoritative JSON, the margin, and the KILL are all correct).

## T-X — depth-continuous basis (angles): KILL (`results/w13-trackx-angles.json`)
Adjacent-layer pre-RoPE key subspaces (r=32): doc-averaged median leading principal
angle **57.7°** (FUND bar was <40°; barely under the 62.1° random null), all-angle
median **76.4° == 78.2° null**. Alignment does not decay with layer distance (gap2 ==
gap1). Only 1 of 15 pairs (L05→06) genuinely shares its top direction (23.7°). No
shared basis to fund a depth-continuous integrator. **Verifier CONFIRMED** (SVD-of-
UₐᵀU_b vs `scipy.subspace_angles`, <0.01°; a mean-cos² estimator shows a faint ~1.55×
excess over null driven by L05-06 — too weak for a shareable r-dim basis).

## T-X — mean-field rank-vs-depth: honest negative (`results/w13-trackx-rank.json`)
Effective rank does **not** collapse with depth. Pre-RoPE (BUG's domain, doc-mean):
participation ratio L0 26.2 → L15 33.2 (+27%, ρ≈+0.01), entropy erank 2.95 → 3.51
(+19%), both peaking early (L2). Post-RoPE net −4.6% only (driven by an L2 hump, not
a monotone deep collapse). The pre-registered collapse bar fails on 3/4 curves; the
mean-field "tokens cluster ⇒ rank drops" prediction is refuted, so it does not fund
rank-allocation-by-depth. RoPE inflates absolute rank ~3–4× (known smearing pitfall).
**Verifier CONFIRMED** (no discrepancy).

## Consequence
The two *perplexity* levers (T-A, T-C) are dead; the two exploratory PDE framings
(T-X) are refuted. **Phase-2 builds only T-B** — a retrieval lever. This session's
win, if any, is in retrieval, not ppl.
