# Week 11 — the decision table (BUG vs bugS vs bugEVICT)

Llama-3.1-8B. RULER retrieval accuracy (%) on 4 tasks + perplexity (ppl, lower=better);
memory = share of the full KV cache. Data: `results/w11-decision-table.json`.

## 16K context

| method | memory | needle | multi-key | multi-value | var-track | ppl |
|---|---|---|---|---|---|---|
| full | 1.000× | 100 | 100 | 100 | 100 | 4.08 |
| ea-k0.1 | 0.100× | 100 | 88 | 100 | 12 | 4.29 |
| bug-r32 | 0.036× | 0 | 0 | 0 | 0 | 4.57 |
| bug-r128 | 0.135× | 25 | 38 | 38 | 0 | — |
| bugS-r32-h256 | 0.053× | 100 | 33 | 0 | 0 | 4.48 |
| bugS-r32-h1024 | 0.098× | 100 | 33 | 0 | 0 | 4.30 |
| bugEVICT-h256 | 0.018× | 100 | 33 | 0 | 0 | 4.44 |
| bugEVICT-h1024 | 0.065× | 100 | 33 | 0 | 0 | 4.36 |
| morph-k0.25 | 0.312× | 75 | 50 | 88 | 12 | — |
| snapkv-k0.1 | 0.100× | 88 | 25 | 75 | 0 | — |
| think-c0.3 | 0.852× | 100 | 100 | 100 | 100 | — |
| palu-r0.5 | 0.504× | 100 | 100 | 100 | 0 | — |
| shadow-r64 | 0.815× | 0 | 0 | 0 | 0 | — |

## 32K context

| method | memory | needle | multi-key | multi-value | var-track | ppl |
|---|---|---|---|---|---|---|
| full | 1.000× | 100 | 100 | 100 | 100 | 7.62 |
| ea-k0.1 | 0.100× | 100 | 50 | 100 | 100 | 8.28 |
| bug-r32 | 0.034× | 0 | 0 | 0 | 0 | 9.31 |
| bug-r128 | 0.130× | 0 | — | — | — | 8.05 |
| bugS-r32-h256 | 0.043× | 100 | 83 | 100 | 100 | 9.16 |
| bugS-r32-h1024 | 0.066× | 100 | 50 | 100 | 100 | 8.88 |
| bugEVICT-h256 | 0.009× | 100 | 0 | 0 | 0 | 8.95 |
| bugEVICT-h1024 | 0.033× | 100 | 50 | 0 | 100 | 8.81 |
| morph-k0.25 | 0.312× | — | — | — | — | 7.54 |
| snapkv-k0.1 | 0.115× | — | — | — | — | 7.87 |
| think-c0.3 | 0.852× | 100 | — | — | — | 7.65 |
| palu-r0.5 | 0.502× | 100 | — | — | — | 9.24 |

## Recommendation (a lean, not a slam dunk)

- **Keep `bugS` (SurpriseSLASH):** at 32K it is never worse than `bugEVICT` and clearly
  better at the tight budget — h256: bugS 83/100/100 vs bugEVICT 0/0/0 on
  multi-key/value/var-track — and it beats ExpectedAttention on multi-key (83 vs 50) at
  <half the memory. The low-rank summary earns its (small) cost on multi-fact retrieval.
- **Drop `bugEVICT`:** cheapest and aces the single needle, but at a tight exact tier it
  collapses on the harder tasks — a single-needle trap. (With a *larger* exact tier, h1024,
  it partly catches up: var-track 100, multi-key 50 — so the gist's edge is budget-dependent.)
- **Retire plain BUG:** 0% on every retrieval task at 32K (the summary alone can't hold a
  sharp fact).

Honest caveats — read these, they matter: **small trial counts (2–6 per cell)**, so
individual cells (esp. the all-or-nothing multi-value) are noisy. The effect is
**context-dependent**: at 16K *both* bugS and bugEVICT are weak on the hard tasks
(multi-key 33, multi-value 0, var-track 0) — the gist's advantage only shows at 32K, and
mainly at the tight h256 budget. This is a **memory-and-retrieval** win; ExpectedAttention
keeps slightly better perplexity (8.28 vs ~9.0). Variable-tracking is hard for every
compressed method except at light compression. Net: the earlier single-needle+ppl read
("the summary is dead weight") was too narrow, but the correction is *"the summary helps on
hard retrieval, especially at tight budgets,"* not *"the summary is essential."* A confirming
run with more trials would firm the lean.
