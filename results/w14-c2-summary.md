# Week-14 C2 — 32K r32 firm at n=4 (multikey win confirmed; a persistent var-track cost)

32K r32-h256, **n=4** (`--n-trials 2 --seeds 0 1`), matched A/B `bugS` vs `bugSseed`, pod
47657929 (A100 SXM4, destroyed). All values verified k/4. `results/w14-wc2n4-ruler-lines.txt`.

| task | `bugS` | `bugSseed` | Δ |
|---|---|---|---|
| niah_multikey | 50 | **100** | **+50** |
| niah_multivalue | 100 | 100 | flat |
| vt (var-track) | 100 | **75** | **−25** |

## Reading it straight
- **Multikey CONFIRMED (50 → 100).** The seed's 32K multikey win holds — same direction as
  the Week-13 n=2×2 (50→100). The warm-up window bites at 32K too, and the seed fixes it.
- **Multivalue flat** — both arms saturate at 100.
- **Var-track: a persistent −25.** `bugSseed` 75 (3/4) vs `bugS` 100 (4/4). This **replicates**
  the Week-13 vt 100→75 dip, so it is a real (if small, n=4) cost at 32K — not a one-flipped-
  trial artifact. A higher-n run would tighten the magnitude, but the *direction* has now shown
  up twice.

## The combined C1+C2 picture — the seed's effect is context-dependent
- **16K (C1):** the seed is a large rescue on every hard task, including var-track (0 → 88):
  at 16K the ~4–5K warm-up window kills `bugS`'s hard tasks, and the seed recovers them.
- **32K (C2):** where `bugS` already succeeds (mv 100, vt 100), the seed is neutral-to-slightly-
  negative — it firms multikey (50→100) but trades a small var-track cost (100→75).
- **Interpretation:** the seed helps most exactly where the warm-up window dominates (16K, and
  32K multikey); where `bugS` already retrieves, the first-block reshuffle is roughly a wash and
  can cost var-track a trial. Net across the retrieval frontier it is **strongly positive**.

## Verdict / promote
The retrieval win stands and the **16K `bugSseed` pick holds** (C1 firmed it). At 32K the seed is
**net-positive but not pure upside** (mk +50, mv flat, vt −25). Promotion stays **scoped docs**
(the pins showed a blanket default flip would also move `bugevict`); the 32K var-track cost is a
fair line to note in the decision table rather than bury. C3 (r128 @32K coverage) still unrun —
needs a top-up. Credit $3.70; keys unrotated.
