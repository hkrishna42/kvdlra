# Week 11 — the decision table (all methods, 16K + 32K, 8B)

Llama-3.1-8B. RULER retrieval accuracy (%) on 4 tasks + perplexity (lower=better);
memory = share of full KV cache. `—` = not measured (baseline var-track @32K still
running at time of writing). Data: `results/w11-decision-table.json`.

## 16K context

| method | memory | perplexity | needle | multi-key | multi-value | var-track |
|---|---|---|---|---|---|---|
| full | 1.000× | 4.08 | 100 | 100 | 100 | 100 |
| ea-k0.1 | 0.100× | 4.29 | 100 | 88 | 100 | 12 |
| bugS-r32-h256 | 0.053× | 4.48 | 100 | 33 | 0 | 0 |
| bugS-r32-h1024 | 0.098× | 4.30 | 100 | 33 | 0 | 0 |
| bugEVICT-h256 | 0.018× | 4.44 | 100 | 33 | 0 | 0 |
| bugEVICT-h1024 | 0.065× | 4.36 | 100 | 33 | 0 | 0 |
| bug-r32 | 0.036× | 4.57 | 0 | 0 | 0 | 0 |
| bug-r128 | 0.135× | — | 25 | 38 | 38 | 0 |
| morph-k0.25 | 0.312× | 4.08 | 75 | 50 | 88 | 12 |
| morph-k0.5 | 0.624× | 4.06 | 100 | 100 | 100 | 75 |
| snapkv-k0.1 | 0.100× | 4.11 | 88 | 25 | 75 | 0 |
| snapkv-k0.25 | 0.250× | 4.04 | 100 | 50 | 100 | 0 |
| think-c0.3 | 0.852× | 4.09 | 100 | 100 | 100 | 100 |
| think-c0.5 | 0.750× | 4.19 | 100 | 100 | 100 | 38 |
| palu-r0.5 | 0.504× | 5.24 | 100 | 100 | 100 | 0 |
| shadow-r64 | 0.815× | 4.11 | 0 | 0 | 0 | 0 |

## 32K context

| method | memory | perplexity | needle | multi-key | multi-value | var-track |
|---|---|---|---|---|---|---|
| full | 1.000× | 7.62 | 100 | 100 | 100 | 100 |
| ea-k0.1 | 0.100× | 8.28 | 100 | 50 | 100 | 100 |
| bugS-r32-h256 | 0.043× | 9.16 | 100 | 83 | 100 | 100 |
| bugS-r32-h1024 | 0.066× | 8.88 | 100 | 50 | 100 | 100 |
| bugEVICT-h256 | 0.009× | 8.95 | 100 | 0 | 0 | 0 |
| bugEVICT-h1024 | 0.033× | 8.81 | 100 | 50 | 0 | 100 |
| bug-r32 | 0.034× | 9.31 | 0 | 0 | 0 | 0 |
| bug-r128 | 0.130× | 8.05 | 0 | — | — | — |
| morph-k0.25 | 0.312× | 7.54 | 100 | 100 | 100 | — |
| morph-k0.5 | 0.625× | 7.57 | 100 | 100 | 100 | — |
| snapkv-k0.1 | 0.100× | 7.87 | 100 | 0 | 50 | — |
| snapkv-k0.25 | 0.250× | 7.68 | 100 | 0 | 100 | — |
| think-c0.3 | 0.852× | 7.65 | 100 | 100 | 100 | — |
| think-c0.5 | 0.750× | 7.90 | 100 | 100 | 100 | — |
| palu-r0.5 | 0.502× | 9.24 | 100 | 100 | 100 | — |
| shadow-r64 | 0.814× | — | 0 | 0 | 0 | — |

## Recommendation (a lean, not a slam dunk)

**`bugS` (SurpriseSLASH) occupies a unique sweet spot: near-perfect retrieval at ~0.04× memory.**
At 32K it hits needle/multi-value/var-track = 100 and multi-key = 83 at **0.043×** — and every
method that matches that accuracy needs far more memory: MorphKV/ThinK/Palu all reach 100s but at
**0.3–0.85×** (7–20× more), and ExpectedAttention (0.10×) is *weaker* on multi-key (50 vs 83) while
SnapKV outright fails multi-key (0). So among **low-memory** methods, `bugS` is the strongest
multi-fact retriever.

- **Keep `bugS`:** best accuracy-per-byte for retrieval; the low-rank summary carries the hard tasks.
- **Drop `bugEVICT`:** cheapest (0.009×) and aces the single needle, but collapses on multi-key/
  value/var-track at a tight tier — a single-needle trap.
- **Retire plain BUG:** 0% on every retrieval task at 32K.
- Baselines are a *reference*: MorphKV is a strong retriever but memory-hungry; ThinK/Palu only win
  by barely compressing (0.5–0.85×); SnapKV is weak on multi-key; ShadowKV fails this axis.

Honest caveats: **small trial counts (2–6/cell)**, all-or-nothing metrics are noisy; the gist's edge
is **budget-dependent** (bigger exact tier narrows it) and **context-dependent** (at 16K both bugS and
bugEVICT are weak on hard tasks); this is a **memory-and-retrieval** win (EA/morph keep better ppl).
A confirming run with more trials would firm the `bugS`-vs-`bugEVICT` lean.
