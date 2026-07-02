# Week 5 plan — kvdlra vs. the low-rank field (Palu/MorphKV/…) + long-context scaling

> A plan, not a writeup. Follows the project conventions (daily breakdown,
> concrete outputs, reuse notes, honest escalation). Read with `docs/PLAN.md`
> (weeks 1–4) and `docs/week4.md` (where we are).

## Goal

Position streaming BUG/DLRA against two *published, current* competitors on their
own terms — one that shares our mechanism and one that doesn't — and report
honestly where we win, tie, and lose. This is the "is it actually competitive
with the field?" week.

## Week 4.5 (pre-step, ~1–2 days): DLRA integrator ablation

Before BUG enters the expensive Week-5 comparisons, pick the **best DLRA
integrator** for KV — an *internal* bake-off, cheap and offline (reconstruction
error vs the SVD oracle **+ per-token cost**, on the KV matrices we already
capture; **no LLM, no pod**).

| Integrator | What it is | Expectation |
|---|---|---|
| **BUG** (current) | unconventional robust integrator (Ceruti–Lubich '22); avoids the backward step | validated baseline |
| **Projector-Splitting (PSI)** | Lubich–Oseledets '14; K/S/L split with a **backward** S-step | ≈ BUG on well-conditioned KV; may be less robust (backward step is unstable for relaxation flows) — validates *why we chose BUG* |
| **Parallel BUG** | K- and L-steps computed independently (Ceruti–Kusch–Lubich parallel integrator) | ≈ BUG accuracy, **faster** → feeds the streaming/decode-latency niche |
| ~~Parallel-in-Time~~ | parareal across time steps | **skip** — fights the causal/sequential KV structure; high risk, unclear payoff (revisit only if the others show integrator choice matters a lot) |

**Decision rule:** run PSI + parallel-BUG against BUG on reconstruction error and
cost; **ship ONE winner into Week 5** (not all three — they'll cluster within
~1–3% since all ≈ oracle, so three curves add clutter not signal, and the real gap
is BUG-vs-eviction). Winner = parallel-BUG if it matches accuracy at lower cost
(speed → streaming story), else plain BUG. **Honest ceiling:** the oracle itself is
only *competitive* vs eviction+quant (Week 4), so a better integrator mainly buys
rigor ("we tried the DLRA family; here's the best for KV, and BUG's design was
right") and parallel-BUG's decode speed — not a competitive silver bullet.
Reuse: `lowrank.py` oracle, captured KV dumps, the streaming harness; new code is
a `psi_step` / `parallel_bug_step` in `integrators/`.

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

## Three comparison axes

**Axis A — low-rank prefill compression (vs Palu).** Extend the Week-4 fair
figure: add Palu (and Palu×quant) to the perplexity-vs-memory plot alongside
BUG×TurboQuant / eviction×TurboQuant. Same prefill-then-score protocol, same
memory accounting. Also run the needle-retrieval test with Palu.

**Axis B — constant-size long generation (vs MorphKV).** New capability: a
**streaming-decode `BUGPress`** that updates the low-rank state per generated
token at a fixed rank cap (a *constant-size* cache), reusing the validated
`StreamingBUG.update()`. Compare against MorphKV on a long-generation task at
matched cache size.

**Axis C — long-context scaling (the amortization test, vs SnapKV/EA).** The
headline follow-up from Weeks 3–4: BUG's one structural handicap is the *fixed*
`U` basis, which **amortizes as context grows** — while eviction memory scales
linearly with `T` and eviction must discard *more* at long context. So BUG's
standing vs SnapKV/EA should **improve monotonically with `T`**. Falsifiable and
cheap to test:

- **Experiment A (do first — decisive on the hypothesis).** Fix the model (8B),
  sweep **ctx ∈ {1K, 4K, 8K, 16K, 32K}**, and at each length redo the *fair*
  memory-vs-perplexity comparison (BUG×Turbo vs SnapKV×Turbo vs EA×Turbo, all
  quantized, position-fair). Plot the **crossover vs `T`**. *Win = BUG's Pareto
  frontier passes SnapKV's (and closes on EA's) as `T` grows; no movement falsifies
  the amortization thesis (a valuable negative result either way).* Reuses the
  existing harness — mostly a `--context-len` sweep on the torch backend + GPU.
- **Experiment B (the real-world test — sketch the harness).** **RULER**
  (standard long-context KV benchmark: multi-needle / multi-hop / aggregation at
  4K–128K) and/or `w4_needle` extended to 8K–32K. Retrieval accuracy vs memory;
  *win = higher accuracy at matched memory, margin growing with length* (eviction's
  drop-an-un-cued-fact failure worsens as `T` grows — already seen at 1.6K).

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

## Schedule (phased — honestly >5 days with 4.5 + long-context folded in)

| Phase | Focus | Concrete output |
|---|---|---|
| **4.5a** | **Integrator ablation** (offline, no LLM): `psi_step` + `parallel_bug_step` in `integrators/`; reconstruction-error + cost vs oracle on captured KV. | `figures/week5/integrator_ablation.png`; winner chosen. |
| **4.5b** | Wire the winning integrator into the streaming path (behind a flag; default stays the validated BUG). | winner usable by `BUGPress`/sweeps. |
| **5-Mon** | Read Palu/LoRC/MorphKV (+ skim STAR-KV/EchoKV/KV-CoRE). Integration decisions; adopt KV-CoRE's Frobenius-optimal compressibility metric. | `docs/notes/lowrank-landscape.md`. |
| **5-Tue** | **LoRC** (`LoRCPress`, quick reimpl) + **Palu** (their repo, else faithful `PaluPress`); validate vs SVD oracle. | presses + reconstruction sanity. |
| **5-Wed** | **Axis A** fair figure: `w4_fair.py` + Palu, LoRC (+ ×Turbo) + needle. | `figures/week5/lowrank_fair.png`; BUG vs Palu vs LoRC verdict. |
| **5-Thu** | **Axis C / Experiment A (highest value)**: long-context perplexity sweep, ctx {1K…32K} on 8B, fair (BUG/SnapKV/EA ×Turbo). GPU pod (token+logs recipe). | `figures/week5/longctx_crossover.png`; does BUG pass SnapKV as `T` grows? |
| **5-Fri** | **Streaming-decode `BUGPress`** (constant-rank via `StreamingBUG.update()`); correctness + constant-memory check. | decode-time BUG. |
| **5-Fri+** | **Axis B** (`MorphKVPress` vs streaming-BUG, long generation) + **Experiment B** (RULER / long-ctx needle) as capacity allows. Write `docs/week5.md` + critic. | `figures/week5/{longgen,ruler}.png`; `docs/week5.md`. Tag `v0.4-w5-compare`. |

Priority order if time is short: **4.5 ablation → Axis C Exp A (long-context, the
key open question) → Axis A (Palu/LoRC) → streaming-decode + Axis B → Exp B RULER.**
Experiment A is the single most decisive item — do it early.

## Reuse
- `lowrank.py` oracle + captured KV dumps + streaming harness — the **integrator ablation** (4.5) is pure offline numerics, no LLM.
- `StreamingBUG.update()` / `blocked_bug_subspace` — the decode-time streaming update (already built + validated).
- `w4_fair.py` / `w4_hybrid_sweep.py` (+ `--plot-only`) + the memory model — extend with Palu/LoRC series and reuse for the long-context sweep.
- `perplexity_sweep.py` prefill-then-score + the **position-fairness fix** (eviction shrinks the cache — reuse explicit `position_ids` or scoring is unfair; matters *more* at long context).
- `press/compat.py`, `TurboQuantPress` — competitor wrapping + the ×quant variants.

## Honest risks / escalation
- **Integrator ablation may be a near-tie** (all ≈ oracle within ~1–3%). That's fine — it's a rigor/speed result, not a competitive lever; don't over-invest, ship the winner, move on.
- **Long-context = GPU memory pressure.** KV grows with `T` (8B @ 32K ≈ 4 GB cache); use bf16 + torch/blocked backend, watch OOM (PLAN §8 #7), and step ctx up gradually. May need a bigger card or offload at 32K.
- **Competitor code may not run** on transformers 5.8 (Palu/MorphKV target specific versions). Reimplement a faithful core and **document the deviation**; escalate if infeasible.
- **Fairness is the whole point.** Same protocol, matched memory, quantize everyone (the Week-4 lessons). Every method scored identically.
- **Infra:** GPU pods flaky (2026-07-01); use onstart-batch + HF token (download) + results via `vastai logs`, never post-hoc SSH. See `[[vastai-pod-flakiness-jul2026]]`.
- **Likely honest outcome (to be tested):** BUG's standing vs SnapKV/EA **improves with context** (amortization) and may pass SnapKV at long ctx / on retrieval; EA is the harder frontier. Report whatever the data says — including a clean negative if amortization doesn't help.

## Success criteria
- **4.5:** integrator winner chosen with an honest ablation figure.
- **Axis C (the key one):** a long-context crossover figure showing whether/where BUG passes SnapKV as `T` grows — with the verdict either way.
- **Axis A/B:** fair single-axis figures (BUG vs Palu/LoRC; streaming-BUG vs MorphKV) + verdicts.
- `docs/week5.md` writeup + critic pass. Tag `v0.4-w5-compare`.
