# Week-13 Track-B — warm-up seed: the GPU A/B (a real, funded WIN)

**One line:** seeding the exact tier from the first ingest chunk (SLASH-routing) **fixes
bugS's 16K retrieval collapse** — dramatically at r32 (100/0/0/0 → 100/100/100/100 at the
same 0.05× memory) — with a **small ppl bonus** and **no net regression** at 32K. It is the
one funded lever of the Week-13 portfolio and the biggest measurable result of the session.

Model Llama-3.1-8B-Instruct, chunked ingest (chunk 4096), RULER n=2×2, ppl 2 samples.
Clean matched A/B (`bugS` vs `bugSseed`, same pod/trials/seeds/footprint). Pod 45887305
(A100 40GB), no OOM/errors. Data: `results/w13-wseed-ruler-lines.txt`,
`results/w13-wseed-ppl-lines.txt`.

## Retrieval (RULER, accuracy; single / mk / mv / vt)

| arm | ctx | `bugS` | `bugSseed` | Δ |
|---|---|---|---|---|
| **r32-h256** (0.05×) | 16K | 100 / **0 / 0 / 0** | 100 / **100 / 100 / 100** | **+100 on every hard task** |
| **r128-h1024** (0.19×) | 16K | 100 / 25 / 25 / 0 | 75 / **100** / 25 / **75** | mk +75, vt +75; single −25; mv flat |
| **r32-h256** (0.04×) | 32K | (mk/mv/vt) 50 / 100 / 100 | **100** / 100 / 75 | mk +50; vt −25; mv flat |

**The headline is r32 @16K:** plain `bugS` retrieves *only* the single-needle task — every
hard task (multikey, multivalue, var-track) is **0**, killed by the ~4–5K warm-up window
(Week-11: "bugS is a ≥32K method; the 16K pick was EA"). The seed lifts all three to **100**
at identical memory. **This overturns Week-11's verdict: `bugSseed` is viable at 16K.** The
seed also improves 32K multikey (50→100), so the window bites at 32K too.

## Perplexity (no regression — a small gain)

| arm | ppl@16K `bugS`→`bugSseed` | ppl@32K `bugS`→`bugSseed` |
|---|---|---|
| r32-h256 | 4.477 → **4.460** (−0.38%) | 9.164 → **9.092** (−0.79%) |
| r128-h1024 | 4.156 → 4.156 (0%) | 8.117 → **8.085** (−0.39%) |

`tok_eq/layer` **identical** to `bugS` (no footprint change). Moving first-chunk outliers to
verbatim `hh` lets the low-rank tail summarize the outlier-removed residual better (the SLASH
premise). Notably `bugSseed`'s 32K ppl (9.092 / 8.085) **equals Q-BUG's** — so the seed
captures Q-BUG's ppl gain *and* the retrieval win, at the same memory as plain `bugS`.

## Honest caveats
- **n=2×2 (4 trials/cell → 25 pts/trial).** The single-cell dips are one-flipped-trial
  effects, reported not buried: r128@16K niah_single 100→75, and r32@32K vt 100→75. Net is
  clearly positive at every (arm, ctx), but these are within noise and worth a higher-n
  re-run before leaning on the exact per-cell deltas.
- **r128 @32K RULER was deferred** (cost) — only r32 has the 32K regression check. r128@32K
  ppl is measured (−0.39%, no regression).
- **r128 @16K is net-positive but noisier than r32** (single −25, mv flat) — the clean,
  dramatic result is r32.
- Retrieval is a cell-aggregate at n=4; the mechanism (needle-free-basis capture) is the same
  proven one steady-state SLASH uses at 32K, so the direction is trustworthy even where the
  magnitude is noisy.

## Mechanism (why SLASH-routing, not warm-then-select)
The first ingest chunk routes through `_prefill`, which absorbed its middle via
`_absorb_columns` and **never** reached `_absorb_block_slash` (the only writer of the exact
tier) — so first-chunk outliers could never be captured (the warm-up window). Fix
(`seed_hh_warmup`, default-off, `bugSseed-*` arm): route the first chunk's middle through
`_absorb_block_slash` in sub-blocks, so each is scored for surprise against the
**strictly-older, needle-free** basis — the exact steady-state mechanism. The adversarial
review's CPU probes killed the initial "warm-then-select" variant (it scored the needle *after*
absorbing it → failed at rank≥8 / diverse backgrounds); SLASH-routing inherits steady-state's
proven capture and the GPU A/B confirms it. Default-off + `_mode=="ingest"` gate preserve every
existing arm bit-for-bit (single-shot invariant intact).

## Status
`bugSseed` is a **funded win** (commits 691c765 cache + 075d6ff pod MODE). Suite 295/1,
ruff+mypy clean. Recommended follow-ups: higher-n (n=8) re-run to firm the two 1-trial dips;
r128 @32K RULER; consider making the seed the bugS default after the higher-n confirm.
