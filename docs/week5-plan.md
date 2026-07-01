# Week 5 plan — kvdlra vs. MorphKV and Palu

> A plan, not a writeup. Follows the project conventions (daily breakdown,
> concrete outputs, reuse notes, honest escalation). Read with `docs/PLAN.md`
> (weeks 1–4) and `docs/week4.md` (where we are).

## Goal

Position streaming BUG/DLRA against two *published, current* competitors on their
own terms — one that shares our mechanism and one that doesn't — and report
honestly where we win, tie, and lose. This is the "is it actually competitive
with the field?" week.

## The competitor landscape (low-rank is a crowded field in 2025–26)

BUG/DLRA sits inside a fast-growing family of **low-rank** KV methods. The whole
point of Week 5 is to place it in that field and name its differentiators:
**streaming/online** (per-token updates with a σ_min-robust DLRA error bound),
**rank-adaptive** (Frobenius-tail θ), and **composes with quantization** (Week 4).
Most rivals are *offline SVD*; a few are online; one is a benchmark; one needs
pretraining. The map:

| Method | Venue | Family / key idea | vs. BUG | Plan |
|---|---|---|---|---|
| **Palu** | ICLR'25 ([2407.21118](https://arxiv.org/abs/2407.21118), [code](https://github.com/shadowpa0327/Palu)) | offline SVD of K/V **projections**; cache latents; GPU fusion kernels; +quant | offline vs streaming | **RUN** (primary low-rank rival) |
| **LoRC** | NeurIPS'24 ([2410.03111](https://arxiv.org/abs/2410.03111)) | low-rank of KV **weight matrices**, progressive per-layer rank, plug-and-play | offline, weight-space | **RUN** (simple reimpl) |
| **MorphKV** | ICML'25 ([2503.00979](https://arxiv.org/abs/2503.00979)) | **constant-size** cache during long **generation**, adaptive token retention | different axis (eviction/gen) | **RUN** (Axis B) |
| **STAR-KV** | ICML'26 ([2606.08382](https://arxiv.org/abs/2606.08382)) | **differentiable soft-threshold** adaptive rank (head+block) + hybrid decomp + MP quant; ≤75% (20× w/ quant) | *offline* adaptive-rank; BUG's θ is the *online* analog | cite; **RUN if code** (strongest adaptive rival) |
| **KV-CoRE** | 2026 ([2602.05929](https://arxiv.org/abs/2602.05929)) | **benchmark** of data-dependent low-rank compressibility (Frobenius-optimal SVD, layer-wise) | not a compressor — a measuring stick | **USE its methodology** to characterize our KV compressibility |
| **EchoKV** | 2026 ([2603.22910](https://arxiv.org/abs/2603.22910)) | reversible: lossless switch full↔compressed; lightweight net reconstructs residual KV | reversibility angle | cite |
| **LRKV** | 2026 ([2601.11471](https://arxiv.org/abs/2601.11471)) | head-dim low-rank residuals **learned during pretraining** | **not post-hoc** — needs training | cite only (can't run on a pretrained Llama without retraining) |
| **OjaKV** | 2025 ([2509.21623](https://arxiv.org/abs/2509.21623)) | **online** low-rank via Oja's rule | online rival — **already beaten in Week 2 (BUG 1.3–3×)** | cite our Week-2 result |
| ShadowKV / ReCalKV / KQ-SVD | 2024–25 | low-rank pre-RoPE keys / head-reorder+calib / provable attn fidelity | offline variants | cite; ShadowKV = Palu fallback |

**What we RUN head-to-head** (feasible in a week): **Palu** (token/projection SVD,
has code), **LoRC** (weight-matrix SVD, easy reimpl), **MorphKV** (constant-size
generation). These span the three sub-families that matter. **STAR-KV** is the
stretch target if its code is released. Everything else is cited related work with
an explicit "how BUG differs" line — and **KV-CoRE's compressibility metric is a
tool we adopt** to quantify, honestly, how low-rank our KV actually is (it caps
what *any* low-rank method — us included — can achieve).

**BUG's honest niche in this table:** the only member that is *both* streaming
(online per-token, deployable during decode) *and* rank-adaptive *and* carries a
DLRA robustness guarantee independent of σ_min. Palu/LoRC/STAR-KV are offline;
OjaKV is online but weaker; LRKV needs pretraining; KV-CoRE only measures. Whether
that niche translates into a win at matched memory is exactly what we measure.

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
| **Mon** | Read Palu, LoRC, MorphKV (+ skim STAR-KV/EchoKV/KV-CoRE). Decide code-vs-reimpl per method; check kvpress (none present in 0.5.1). Adopt KV-CoRE's Frobenius-optimal metric to report our KV's intrinsic compressibility ceiling. | `docs/notes/lowrank-landscape.md`: mechanisms, integration decision, KV-CoRE compressibility numbers for our layers. |
| **Tue** | **LoRC** (`LoRCPress`: low-rank of K/V weight matrices, progressive per-layer rank — quick reimpl) + **Palu** (their repo, else a faithful `PaluPress`). Validate each vs our SVD oracle. | `LoRCPress`, Palu runner; per-method reconstruction sanity. |
| **Wed** | **Axis A**: extend `w4_fair.py` with Palu, LoRC (+ their ×TurboQuant); rerun fair perplexity-vs-memory + needle. | `figures/week5/lowrank_fair.png`; verdict: BUG vs Palu vs LoRC at matched memory. |
| **Thu** | **Streaming-decode `BUGPress`** (constant-rank decode updates via `StreamingBUG`/blocked torch — reuse `.update()`). Correctness: generation parity + constant-memory check over long decode. | Decode-time BUG; note on the constant-size mechanism. |
| **Fri** | **Axis B**: `MorphKVPress` (or their code) vs streaming-BUG on a long-generation task at matched cache size (GPU pod). Write `docs/week5.md` + critic pass. STAR-KV as stretch if code exists. | `figures/week5/longgen.png`; `docs/week5.md` verdict. Tag `v0.4-w5-compare`. |

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
