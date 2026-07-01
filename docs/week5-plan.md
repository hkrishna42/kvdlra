# Week 5 plan — kvdlra vs. MorphKV and Palu

> A plan, not a writeup. Follows the project conventions (daily breakdown,
> concrete outputs, reuse notes, honest escalation). Read with `docs/PLAN.md`
> (weeks 1–4) and `docs/week4.md` (where we are).

## Goal

Position streaming BUG/DLRA against two *published, current* competitors on their
own terms — one that shares our mechanism and one that doesn't — and report
honestly where we win, tie, and lose. This is the "is it actually competitive
with the field?" week.

## The two competitors (and why)

1. **Palu** — ICLR 2025, [arXiv:2407.21118](https://arxiv.org/abs/2407.21118),
   [code](https://github.com/shadowpa0327/Palu). **Our closest rival: low-rank.**
   Palu SVD-decomposes the K/V *projection* layers offline (with calibration),
   caches the low-dim latent states, and reconstructs K/V on the fly — and
   composes with quantization (they report ~11× at ~91% compression). This is the
   sharpest scientific question of the whole project: **is a streaming DLRA
   subspace tracker a better low-rank KV compressor than offline SVD projection?**
   We already know streaming BUG ≈ the truncated-SVD oracle within 1–3% (Week 2);
   Palu is a *real, tuned* low-rank system to test that against.

2. **MorphKV** — ICML 2025, [arXiv:2503.00979](https://arxiv.org/abs/2503.00979).
   **A different mechanism, in its home setting: constant-size cache during long
   generation.** MorphKV keeps a *fixed-size* cache while generating extended
   responses by adaptively retaining tokens correlated with recent context. This
   pushes kvdlra into the **generation/decode** phase — which is BUG's natural
   home (it is a *streaming* tracker that updates per token), a strength we have
   not yet used (Weeks 3–4 only compressed the pre-fill). Comparing a constant-
   *rank* streaming BUG against a constant-*token* MorphKV is on-thesis and novel.

(Alternative to Palu if its code won't run cleanly: **ShadowKV** — low-rank
*pre-RoPE* keys, even closer to BUG, but heavier on systems/offloading. Keep as
fallback.)

## Two comparison axes

**Axis A — low-rank prefill compression (vs Palu).** Extend the Week-4 fair
figure: add Palu (and Palu×quant) to the perplexity-vs-memory plot alongside
BUG×TurboQuant / eviction×TurboQuant. Same prefill-then-score protocol, same
memory accounting. Also run the needle-retrieval test with Palu.

**Axis B — constant-size long generation (vs MorphKV).** New capability: a
**streaming-decode `BUGPress`** that updates the low-rank state per generated
token at a fixed rank cap (a *constant-size* cache), reusing the validated
`StreamingBUG.update()`. Compare against MorphKV on a long-generation task at
matched cache size.

## New capability to build (the interesting engineering)

`BUGPress` today compresses only the pre-fill and is then static during decode.
Week 5 builds **decode-time streaming**: each generated token's K/V updates the
tracked subspace/core (constant rank → constant memory), exactly what
`StreamingBUG.update()` / the blocked torch tracker already do. This is the first
time the project uses BUG *as a streaming integrator during generation* — its
core theoretical selling point. Wire it as a decode hook (or a custom cache) so
attention sees the running low-rank reconstruction.

## Benchmarks

- **Perplexity vs memory** (WikiText-2, ctx 1024, the existing harness) — Axis A.
- **Needle-in-a-haystack** (existing `w4_needle.py`) with Palu + MorphKV added.
- **Long-generation quality** — MorphKV's turf: a long-response task (e.g.
  LongBench subset: qasper/narrativeqa/gov_report, or long-form generation with a
  quality metric like ROUGE/LLM-judge). Measures whether a constant-size cache
  preserves coherence over long outputs. Needs the GPU pod (long generations);
  apply the Week-4 lesson (**`hf_transfer` for downloads, results emitted to
  `vastai logs`, never rely on post-hoc SSH** — see `[[vastai-pod-flakiness-jul2026]]`).

## Daily breakdown

| Day | Focus | Concrete output |
|---|---|---|
| **Mon** | Read Palu (§ method + calibration) and MorphKV (§ constant-size update rule). Decide integrate-their-code vs faithful reimplementation; check kvpress (neither is in it as of 0.5.1). | `docs/notes/palu-morphkv.md`: mechanisms + integration decision. |
| **Tue** | **Palu** in our harness: run their repo, or a faithful `PaluPress` (offline SVD of K/V projections, cache latents, reconstruct). Validate its reconstruction error vs our SVD oracle. | `PaluPress` (or their runner) producing perplexity-vs-memory points. |
| **Wed** | **Axis A**: extend `w4_fair.py` with Palu + Palu×TurboQuant; rerun the fair perplexity-vs-memory + needle. | `figures/week5/lowrank_fair.png`; verdict: BUG vs Palu at matched memory. |
| **Thu** | **Streaming-decode `BUGPress`** (constant-rank decode updates via `StreamingBUG`/blocked torch). Correctness: generation parity + constant-memory check. | Decode-time BUG; `docs`-level note on the constant-size mechanism. |
| **Fri** | **Axis B**: `MorphKVPress` (or their code) vs streaming-BUG on the long-generation task at matched cache size (GPU pod). Write `docs/week5.md` + critic pass. | `figures/week5/longgen.png`; `docs/week5.md` verdict. Tag `v0.4-w5-compare`. |

## Reuse
- `StreamingBUG.update()` / `blocked_bug_subspace` — the decode-time streaming update (already built + validated).
- `w4_fair.py` + `w4_head_to_head.py` + memory model — extend with Palu/MorphKV series.
- `perplexity_sweep.py` prefill-then-score + the **position-fairness fix** (eviction/constant-size methods shrink the cache — reuse the explicit-`position_ids` fix or they'll be scored unfairly).
- `press/compat.py` — if either competitor is wrapped as a kvpress press.
- `TurboQuantPress` — for the ×quant variants.

## Honest risks / escalation
- **Competitor code may not run** on transformers 5.8 / our stack (Palu/MorphKV target specific versions). If so, reimplement a faithful core and **document the deviation** — do not silently approximate. Escalate if a faithful reimplementation is infeasible in the week.
- **Fairness is the whole point.** Match memory accounting across mechanisms (Week-4 taught us how easy it is to be accidentally unfair — the eviction position bug). Every method scored by the same protocol.
- **MorphKV needs long generation → GPU pod.** vast.ai was flaky on 2026-07-01; budget for infra friction and use the batch/logs pattern.
- **Likely honest outcome (hypothesis, to be tested):** BUG competitive-to-better vs Palu in the low-rank regime (streaming ≈ oracle, and pre-RoPE helps); MorphKV strong on long generation where adaptive token retention matters — with streaming-BUG's constant-rank cache as a principled alternative. Report whatever the data says.

## Success criteria
A fair, single-axis figure per comparison (BUG vs Palu; streaming-BUG vs MorphKV),
each with the honest verdict, plus the decode-time streaming capability landed and
tested. Tag `v0.4-w5-compare`.
