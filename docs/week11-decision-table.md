# Week 11 — the decision table (POOLED: all sources, n per cell shown)

Llama-3.1-8B. RULER retrieval accuracy (%) on 4 tasks + perplexity (lower=better);
memory = share of full KV cache, ALL floats counted. Pooled across every run this week
(both seeds where run). Data: `results/w11-decision-table.json`, merged by
`scripts/w11_merge.py`. Probe evidence: `results/w11-probe8b-all.json`,
`results/w11-probe-1b-mk-*.json`.

## 16K context

| method | memory | perplexity | needle | multi-key | multi-value | var-track | n/cell |
|---|---|---|---|---|---|---|---|
| full | 1.000× | 4.08 | 100 | 100 | 100 | 100 | 8 |
| ea-k0.1 | 0.100× | 4.29 | 100 | 92 | 100 | 17 | 8-12 |
| bugS-r128-h256 | 0.150× | 4.17 | 100 | 25 | 25 | 0 | 4-10 |
| bugS-r128-h1024 | 0.191× | 4.16 | 100 | 25 | 25 | 0 | 4-10 |
| bugS-r256-h256 | 0.281× | 4.12 | — | 0 | 0 | 0 | 4 |
| bugS-r256-h1024 | 0.316× | 4.12 | — | 0 | 0 | 0 | 4 |
| bugS-r32-h256 | 0.053× | 4.48 | 100 | 14 | 0 | 0 | 3-7 |
| bugS-r32-h1024 | 0.098× | 4.30 | 100 | 33 | 0 | 0 | 3 |
| bugEVICT-h256 | 0.018× | 4.44 | 100 | 14 | 0 | 0 | 3-7 |
| bugEVICT-h1024 | 0.065× | 4.36 | 100 | 33 | 0 | 0 | 3 |
| bug-r32 | 0.036× | 4.57 | 0 | 0 | 0 | 0 | 8 |
| bug-r128 | 0.135× | — | 25 | 38 | 38 | 0 | 8 |
| morph-k0.25 | 0.312× | 4.08 | 75 | 50 | 88 | 12 | 8 |
| morph-k0.5 | 0.624× | 4.06 | 100 | 100 | 100 | 75 | 8 |
| snapkv-k0.1 | 0.100× | 4.11 | 88 | 25 | 75 | 0 | 8 |
| snapkv-k0.25 | 0.250× | 4.04 | 100 | 50 | 100 | 0 | 8 |
| think-c0.3 | 0.852× | 4.09 | 100 | 100 | 100 | 100 | 8 |
| think-c0.5 | 0.750× | 4.19 | 100 | 100 | 100 | 38 | 8 |
| palu-r0.5 | 0.504× | 5.24 | 100 | 100 | 100 | 0 | 8 |
| shadow-r64 | 0.815× | 4.11 | 0 | 0 | 0 | 0 | 8 |

## 32K context

| method | memory | perplexity | needle | multi-key | multi-value | var-track | n/cell |
|---|---|---|---|---|---|---|---|
| full | 1.000× | 7.62 | 100 | 100 | 100 | 100 | 2-8 |
| ea-k0.1 | 0.100× | 8.28 | 100 | 67 | 100 | 83 | 6-8 |
| bugS-r128-h256 | 0.139× | 8.15 | 100 | 75 | 100 | 75 | 4 |
| bugS-r128-h1024 | 0.159× | 8.12 | 100 | 75 | 100 | 75 | 4 |
| bugS-r256-h256 | 0.266× | 7.74 | — | 0 | 0 | 0 | 4 |
| bugS-r256-h1024 | 0.284× | 7.74 | — | 0 | 0 | 0 | 4 |
| bugS-r32-h256 | 0.043× | 9.16 | 100 | 67 | 100 | 100 | 6-14 |
| bugS-r32-h1024 | 0.066× | 8.88 | 100 | 50 | 100 | 100 | 2-8 |
| bugEVICT-h256 | 0.009× | 8.95 | 100 | 0 | 0 | 0 | 6-8 |
| bugEVICT-h1024 | 0.033× | 8.81 | 100 | 50 | 0 | 100 | 2-8 |
| bug-r32 | 0.034× | 9.31 | 0 | 0 | 0 | 0 | 2 |
| bug-r128 | 0.130× | 8.05 | 0 | — | — | — | — |
| morph-k0.25 | 0.312× | 7.54 | 100 | 100 | 100 | 50 | 2 |
| morph-k0.5 | 0.624× | 7.57 | 100 | 100 | 100 | 100 | 2 |
| snapkv-k0.1 | 0.100× | 7.87 | 100 | 0 | 50 | 0 | 2 |
| snapkv-k0.25 | 0.250× | 7.68 | 100 | 0 | 100 | 0 | 2 |
| think-c0.3 | 0.852× | 7.65 | 100 | 100 | 100 | 100 | 2 |
| think-c0.5 | 0.750× | 7.90 | 100 | 100 | 100 | 50 | 2 |
| palu-r0.5 | 0.502× | 9.24 | 100 | 100 | 100 | 100 | 2 |
| shadow-r64 | 0.814× | — | 0 | 0 | 0 | 0 | 2 |

## Recommendation (three operating points, honest)

- **Retrieval-per-byte at ≥32K: `bugS-r32-h256` (0.043×).** The cheapest point of the only
  sub-0.1×-capable family covering
  all four tasks: 100/67/100/100 pooled (n=6-14). EA at 0.100× is 100/67/100/83. The ppl cost
  is real (9.16 vs EA 8.28).
- **Balanced quality+retrieval at ≥32K: `bugS-r128-h1024` (~0.16×).** Beats EA on ppl (8.12 vs
  8.28) AND multi-key (75 vs 67), ties multi-value (100), loses var-track narrowly (75 vs 83;
  n=4). A quality-first point bought with 1.6× EA's memory — not a free win. Do NOT go
  higher: r256 buys more ppl (7.74) but loses retrieval entirely (0 on all hard tasks).
- **At 16K: EA (or SnapKV for pure ppl).** The BUG family's basis warm-up window (Q1) makes it
  weak on hard tasks below ~32K: bugS-r32-h256 pooled 14/0/0, r128 25/25/0, vs EA 92/100/17.
  bugS is a ≥32K method — state this plainly.

The gist-helps lean FIRMS at 32K: `bugEVICT-h256` collapses on the hard tasks (0/0/0, n=6-8)
where `bugS-r32-h256` scores 67/100/100 at 4.8× the memory but still 0.043×.

## Q1 answered: why bugS-r32 retrieves better at 32K than 16K

The 16K deficit is REAL and mechanistic — the prior "task-construction + small-n noise" lean
is refuted. Mechanism: a **basis warm-up window**. Surprise = residual against the streaming
low-rank basis; for roughly the first 4-5K tokens (8B, rank 32) the basis is young, filler is
as surprising as planted codes, and codes are NOT selected into the exact tier. The miss is at
selection time and budget-independent: 8B multikey capture is 6/8 codes at 16K flat from
hh=64 to hh=2048, and 7/8 at 32K, equally flat. Misses are exactly the earliest-planted items
(8B: keys {0,1} missed at 16K, {0} at 32K, both trials; 1B: {0,1}@4K, {0}@8K, none@16K).
RULER plants at relative positions, so longer contexts push items past the absolute window —
retro-predicted by multivalue@16K per-value recall = 0.75 (accuracy 0: the one value planted
at ~3.3K sits inside the window and is always lost, the other three retrieved) and vt@16K
= 0 vs vt@32K = 100 (chain root at ~3.3K vs ~6.6K). The old "EA jumps too" argument only ever
held for var-track (EA vt 17→83); EA scores 92/100 on multikey/multivalue at 16K, so it was
never evidence of a task artifact. The n≥5 rerun confirms: bugS-r32-h256 @16K pooled = mk 14
(1/7), mv 0 (0/7), vt 0 (0/7).

## Q2 answered: the balanced config bugS-r128

Rank is the ppl lever, confirmed: bugS-r128 ppl 4.17/4.16 (h256/h1024) at 16K vs EA 4.29;
8.15/8.12 at 32K vs EA 8.28 — at 0.14-0.19× memory. r256 reaches 4.12/7.74 at 0.27-0.32×
(diminishing returns; full = 4.08/7.62). 32K retrieval largely holds at r128 (n=4/cell):
needle 100, mk 75, mv 100, vt 75. At 16K the hard tasks stay weak at r128 (mk 25, mv 25,
vt 0) — same ≥32K caveat as r32.

**r256 follow-up (2026-07-18): retrieval COLLAPSES — r128 is a narrow sweet spot.** All 12
r256 hard-task cells (h256 and h1024, 16K and 32K, n=4 each) score **0.00 accuracy AND 0.00
recall** — not even partial values, where r128@16K still recalled 0.81 — while r256 ppl is
the best BUG-family sub-full ppl (7.74/4.12; MorphKV/SnapKV/ThinK still post lower ppl at
≥0.25× memory). A healthy-for-language cache that retrieves
nothing. This REFUTES the plain "cleaner basis / richer gist" candidate for r128's win: if
gist reconstruction carried retrieval, r256's better-fitting gist should be at least as
good, and it is strictly worse. The rank ladder now reads: r32 retrieves via the exact tier
(post-warm-up), r256 retrieves via nothing, and r128 retrieves via a mechanism we have NOT
attributed — a sweet spot between too-blurry-to-hold and too-well-fitted-to-select-or-
surface. The hh_budget=0 ablation at r128 is now the top-priority next experiment; an r192
point and a probe matched to the exact RULER trials are the follow-ups.

## Honest caveats

- 32K non-BUG baselines and all r128 retrieval cells are n=2-4 — all-or-nothing metrics are
  noisy at that n; the r128-vs-EA multi-key edge (75 vs 67) is inside the noise.
- bugS-r256 retrieval is now measured on the HARD tasks (0 across all 12 cells, n=4);
  its needle cells remain unmeasured (the family saturates needle at every measured arm,
  so budget went to discriminating cells).
- The r128 retrieval mechanism is unattributed and now bracketed from BOTH sides by the
  r256 collapse (see Q2) — flagging, not hiding, it.
- bug-r128 16K ppl and shadow-r64 32K ppl are unmeasured (—).
- EA's 32K var-track softened 100→83 with more trials; expect other n≤4 cells to move too.
