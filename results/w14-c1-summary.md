# Week-14 C1 — the 16K warm-up-seed headline, FIRMED at n=8 (a confirmed win, honestly refined)

**One line:** at n=8 the warm-up seed still delivers a large, clean 16K retrieval win at
0.05× memory — `bugSseed` clears the pre-registered ≥87.5 bar on **all three** hard RULER
tasks (multikey 100, multivalue 100, var-track 88) vs `bugS` 50 / 0 / 0 — but the n=8 sample
**refines** the Week-13 n=2×2 preview: `bugS` is not a *total* 0/0/0 collapse (multikey firms
to 50), and `bugSseed` var-track is 88 (7/8), not a perfect 100.

Model Llama-3.1-8B-Instruct, chunked ingest (chunk 4096), RULER **n=8** (`--n-trials 4
--seeds 0 1`, 12.5 pts/trial). Matched A/B, same pod/seeds/footprint. Pod 47643614 (A100
SXM4), destroyed. Data: `results/w14-wseed8-ruler-lines.txt` (all values verified k/8).

## The result (16K r32-h256, 0.05× memory; acc)

| task | `bugS` | `bugSseed` | Δ | Week-13 n=2×2 (for reference) |
|---|---|---|---|---|
| niah_single | 100 | 100 | flat (both saturate) | 100 → 100 |
| niah_multikey | **50** | **100** | **+50** | 0 → 100 |
| niah_multivalue | 0 (recall 75) | **100** | **+100** | 0 → 100 |
| vt (var-track) | 0 | **88** (7/8) | **+88** | 0 → 100 |

## Reading it straight
- **The seed win is CONFIRMED.** On every hard task `bugSseed` reaches the pre-registered
  bar (multikey/multivalue/var-track each **≥ 87.5**), while plain `bugS` gets 50 / 0 / 0.
  The direction and magnitude of the Week-13 result survive the higher-n re-run.
- **Two honest n=8 refinements** (why firming was worth doing):
  1. **`bugS` multikey is 50, not 0.** The n=2×2 "bugS 0/0/0" overstated the collapse: at
     n=8 plain `bugS` retrieves half the multikey needles. So the "bugS totally collapses at
     16K" framing holds for **multivalue and var-track (both 0)** but **not multikey**. The
     seed's cleanest, largest wins are mv (0→100) and vt (0→88); multikey is a solid +50 from
     a non-zero base.
  2. **`bugSseed` var-track is 88 (7/8), not 100.** One of eight trials misses. Still a large
     lift from 0, but not the perfect sweep the n=2 preview implied.
- **Pre-registered bar:** the "`bugSseed` ≥ 87.5 on all three hard tasks" half is **MET**
  (100 / 100 / 88). The "`bugS` ≤ 12.5" half is **not** met for multikey (bugS = 50) — a
  refinement of the premise, not a failure of the seed.

## Promote read
The firmed data **supports** the Week-13 decision to make `bugSseed` the 16K pick: at 0.05×
memory it is decisively better than `bugS` on the hard tasks (100 / 100 / 88 vs 50 / 0 / 0).
The honest headline is a **strong, clean lift on every hard task** (mk +50, mv +100, vt +88),
not a literal "0 → 100 on everything." Promotion stays **scoped docs** (the pins proved a
blanket default flip would also move `bugevict`).

## Scope / caveats (honest)
- **Only C1 ran.** C2 (32K r32 firm) and C3 (r128 @32K coverage) were **not** run: the
  `wseed8` sizing under-counted by the tasks-per-cell factor (~4× slower than estimated —
  ~25 min per RULER row at 16K), so the full 3-cell n=8 run did not fit $8.40. C1 is the
  load-bearing cell (the headline), so it was banked and the pod destroyed with credit
  preserved (**$5.84**). C2/C3 need either a reduced-n re-run that fits the remaining credit,
  or a top-up. See the cost note below.
- ppl was not re-measured (Week-13 already established no-regression; unchanged tok_eq/layer).
- var-track 88 = 7/8: one trial; a further-n run would tighten it, but it already clears the
  ≥87.5 bar.

## Corrected cost model (for the next run)
Measured rate: **~25 min per RULER row at 16K r32** (a row = the n=8 aggregate for one
task×arm). A cell's cost is `n_tasks × n_arms × (per-row) `— the earlier sheet omitted the
`n_tasks` factor. So: C1 (4 tasks × 2 arms) ≈ 200 min ≈ $2.5 ✓ (matches actual); C2 (3×2 at
32K, slower) and C3 (3×2 r128@32K, slower still) each materially exceed the leftover credit
at n=8. A **C2 at n=4** (`--n-trials 2 --seeds 0 1`) roughly halves it and fits the $5.84.
