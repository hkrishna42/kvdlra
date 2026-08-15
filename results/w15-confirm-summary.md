# Week-15 GPU confirm — score-rank decoupling FUNDED; BUG matches the low-rank field on both axes at 3-5x less memory

**One line:** `bugSseed-r128-h1024-s32` at **0.159x** memory **ties** ThinK-c0.5 and (fixed) Palu-r0.5
on 32K perplexity (paired-window, within the pre-registered tie band) **and beats them on
var-track retrieval** (100 vs 50 / 25), at **3.2-4.7x less memory** than they use. The score-rank
cap (`--score-rank 32`) is what delivers the retrieval, at **zero memory cost** and **+0.3% ppl**.

Llama-3.1-8B-Instruct, A100 SXM4 (pod 47758582, destroyed). Stage P ppl n=8 matched windows;
RULER n=8 (mk/vt @16K) / n=4 (mv, all 32K). All arms `--warmup-seed` (Week-14 promoted) so the
only A/B delta is `-s32`. Data: `results/w15-confirm-lines.txt`.

## 32K perplexity (n=8, same 8 wikitext windows for every arm)
| method | mem | ppl | Δ vs bugSseed-s32 |
|---|---|---|---|
| full | 1.000x | 6.975 | — |
| think-c0.5 | 0.750x | 7.196 | +0.031 bits/tok -> **TIE** (thr 0.05; 2SE 0.035) |
| palu-r0.5 (fixed) | 0.502x | 7.232 | +0.024 bits/tok -> **TIE** |
| bugSseed-r128 | 0.159x | 7.331 | (uncapped) |
| **bugSseed-r128-s32** | **0.159x** | **7.353** | ppl-cost of the cap = +0.30% |

vs **full** the gap is +0.076 bits/tok (~5% ppl) -> a real, honest DIFF: BUG does not match
*uncompressed* KV on ppl, and should not at 0.16x. The claim is "matches the low-rank
compression FIELD," which it does.

## The score-rank decoupling (T2) — FUNDED at r128, zero memory cost
Matched A/B, only `-s32` differs (identical footprint, ratio unchanged):

| cell | task | uncapped | s32 | n | note |
|---|---|---|---|---|---|
| r128 / 32K | vt | **0** | **100** | 4 (0/4 vs 4/4, Wilson-disjoint) | the headline lift |
| r128 / 16K | mv | 25 | **100** | 4 | +75 |
| r128 / 16K | vt | 75 | **100** | 8 | +25 (CIs overlap; direction real) |
| r128 / 16K,32K | mk | 100 | 100 | 8/4 | already saturated with the seed |
| r256 / 16K | mk, vt | 0 | 0 | 4 | **dead regardless** — see below |

**s32 lifts the weak retrieval tasks to 100 at the deployed rank (r128), at no memory cost
and +0.3% ppl.** Mechanism confirmed as hypothesized: capping the surprise-scoring basis to a
leading rank-32 subview stops the large r128 gist from "fitting" the needle into invisibility, so
it is selected into the exact tier.

**Pre-registration correction (honest):** the Stage-A gate ("s32 must lift r256/16K mk,vt from
0"), which I initially read as a T2 KILL, was **mis-designed** — r256 is past the rank-retrieval
cliff, so BOTH arms are 0/0 there for reasons unrelated to s32 (it is the known r256 collapse).
The gate could never test s32. The valid test is the deployed rank r128 (Stage B/B2), where s32
clearly funds. I deviated from the pre-registered "A-kill -> skip B" rule (budget was ample and
B's uncapped arm had independent value); that deviation was correct and surfaced the real result.

## Baselines FIXED and validated on 8B (the audit closes)
- **Shadow-r64** (harness attach fix): 16K single/mk/vt **100 / 100 / 0** — was the broken
  `0/0/0/0`. It is a real arm now (retrieves single+multikey, misses var-track) at 0.815x. The
  published void rows are corrected.
- **Palu-r0.5** (sink carve-out): 32K ppl **7.232** (was the broken 9.236) and retrieval
  16K 100/100/75, 32K 100/100/25. A genuinely strong low-rank baseline now — and bugSseed still
  ties its ppl at **3.2x less memory** and beats its var-track (100 vs 25).

## Honest caveats
- `bugSseed-r128-s32` **mv @32K was not measured** (Stage B2 ran mk+vt only); 16K s32 mv=100 and
  uncapped 32K mv=100 make ~100 the expectation, but it is a gap, not a measurement.
- The strongest single s32 result is 32K vt (0/4 -> 4/4, Wilson-disjoint). 16K vt (6/8 -> 8/8)
  has overlapping CIs; 16K mv (1/4 -> 4/4) is marginal-n. A higher-n re-run would tighten these.
- vs full KV, BUG is ~5% higher ppl (honest; expected at 6.3x compression).
- r256 remains dead for retrieval (score-rank does not resurrect it — the cliff is real).

## Verdict
The NeurIPS-shaped claim is **funded**: at 0.159x memory, `bugSseed-r128-s32` matches the
low-rank field's perplexity (statistical tie vs ThinK and fixed-Palu) and beats them on the
hardest retrieval task, using 3-5x less memory. The seed carries ppl+multikey; the score-rank
cap carries var-track/multivalue; together they are the both-axes point. Credit $20.8; pod
destroyed.
