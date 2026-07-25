# Week 12 — attributing the r128 sweet spot (T1/T2 landed, T3 pending)

Week 12 asked two questions left open by Week 11: **what carries bugS-r128's 32K
retrieval** (the mechanism was unattributed, bracketed by the starved-tier probe and the
r256 collapse), and **does the warm-up-window story predict 64K** (bugS-r32-h256 multikey
should rise above EA's 67 once all planted items exit the ~4–5K window). All new runs:
Llama-3.1-8B, chunked ingest, both seeds. Data: `results/w12-drop-ruler-lines.txt`,
`results/w12-bugr128-ruler-lines.txt`, `results/w12-r192-ruler-lines.txt`,
`results/w12-r192-ppl-lines.txt`, `results/w12-probe-r128-lines.txt`; pooling via
`scripts/w11_merge.py` (never hand-edit the JSON).

**Verdict.** H1 (withholding-from-absorption cleans the basis) is **refuted at r128**:
hiding the exact tier kills mv/vt outright (100→0, 75→0) and halves mk (75→50), leaving a
profile identical to plain bug-r128. The **visible exact tier is the mechanism** of
bugS-r128's edge (H2/H3 family), and the trial-matched probe shows capture is
**instance-dependent** — partial-to-full on the trials where wins plausibly came from —
which leans H2 (partial capture), a lean, not a slam dunk. Multi-key has a genuine
gist-only component (bug-r128 alone scores 50); mv/vt have none. r192 collapses
task-staggered, so r128 stays the largest rank covering all four tasks. The 64K
prediction test is running; nothing below depends on it.

## The design finding: the pre-registered ablation could not run

The planned `hh_budget=0` ablation is **degenerate**: hh=0 flips the `hh_enabled` gate,
turning the whole SLASH path off, and the retention rule is inert under arm-style
`coord_budget` — so "bugS with hh=0" is bit-for-bit **plain bug-r128**, and could not
discriminate H1 (now a pinned regression: `test_hh_zero_bugslash_equals_plain_bug`).
Built instead: **`hh_retain=False` select-and-discard ("bugSdrop")** — selection and
withholding IDENTICAL to bugS (the pool is kept and feeds re-selection/demotion; layer
state is bit-identical given the same K/V stream), but the tier is **invisible to
attention**. The pool is live state, so the arm honestly reports the same **0.159×** —
an ablation instrument, NOT a cheaper operating point. Pinned by
`tests/test_bug_cache_week12.py` (tier selected-but-invisible, stream-matches-retain
mechanics, decode mask sizes), the same-formula footprint pin in
`tests/test_accounting.py`, and the `--hh-discard` harness flag tests in
`tests/test_w12_harness.py` (arm name `bugSdrop-r128-h1024`).

## T1 — select-and-discard at r128 @32K (H1 test)

| arm | memory | multi-key | multi-value | var-track | n/cell |
|---|---|---|---|---|---|
| bugS-r128-h1024 (W11 baseline) | 0.159× | 75 | 100 | 75 | 4 |
| **bugSdrop-r128-h1024** (tier invisible) | 0.159× | **50** | **0** (recall 0.00) | **0** | 4 |
| bug-r128 (gap-fill, previously unmeasured) | 0.130× | 50 | 0 (recall 0.75) | 0 | 2 |

Reading: **H1 refuted at r128.** Same selection, same withholding, same footprint — only
visibility removed — and the accuracy profile drops to exactly plain bug-r128's 50/0/0.
Withholding-from-absorption contributes nothing detectable on these tasks at this n. The
split is clean: **mk has a gist-only component** (50 without any visible tier; the tier
adds 50→75), **mv/vt have none** (0 without it). bug-r128's mv recall 0.75 vs bugSdrop's
0.00 is a curiosity at n=2 — noted, not interpreted.

## T2a — r192: task-staggered collapse, not a single cliff

`bugS-r192-h1024` @32K (n=4): **mk 100 / mv 0 (recall 0.25) / vt 0** at 0.222×. So the
r128→r256 collapse is staggered by task: mk survives r192 (≥ r128's 75, n=4 vs 4), mv/vt
are already gone (r128: 100/75). Meanwhile ppl keeps improving with rank — r192
**4.124** @16K (0.254×) and **7.867** @32K (0.222×): below r128 (4.16/8.12) at both
lengths, already matching r256 at 16K (both 4.124) and between r128 and r256 (7.74) at
32K. The "balanced config" claim survives **only at r128**: it is the largest
rank that still covers all four tasks.

## T2b — trial-matched probe: capture is instance-dependent

`w11_probe.py` rerun at r128 on the exact RULER (trial,seed) combos t{0,1}×s{0,1}
(multikey, hh 256+1024). At 32K: t0s0 and t0s1 **starved** (codes 0/8 @256, 2/8 @1024,
needle_in_hh 0/3, neighbors 0/0/0); **t1s0 @1024: FULL capture** (needle_in_hh 3/3,
captured=True, nbr 1/1/1) though starved @256; **t1s1: PARTIAL at both budgets**
(needle_in_hh 2/3, nbr 1/1/1). At 16K all four combos are starved at both budgets
(needle_in_hh 0/3 everywhere) — consistent with the warm-up window and r128's weak 16K
hard tasks. So the W11 premise "the queried code is never in the tier" **does not hold
trial-matched**: it was an artifact of probe/RULER sampling mismatch. Partial-to-full
capture on the t1 combos is **consistent with H2 carrying the wins** — but per-trial
RULER outcomes are cell aggregates, so the capture↔win correlation is suggestive, not
airtight. H2 leads; H3 (interplay) is not excluded.

## Scoping correction applied to prior docs

Week-10/11 docs said "plain FIFO bug-r128 = 0" as if measured broadly. It was scoped to
**needle@32K only (n=2)**; the 32K hard-task cells were **unmeasured** until this week
(16K precedent: mk 38 / mv 38 / vt 0). Now measured: **50/0/0 (n=2)**. The correction is
applied wherever the claim appears; do not present mk=50 as contradicting a measured
wall — there was no wall, only a gap.

## T3 — 64K prediction test: PENDING

Pod `mk64` running: `bugS-r32-h256` + `ea` control @65536, mk n=4, mv/vt n=2.
Pre-registered prediction from the warm-up-window mechanism: **mk rises above 67** as
all planted items exit the ~4–5K absolute window. Results fold in when landed — this doc
makes no claim about them.

## Cost ledger and pod inventory

~**$5.1** of $25.94 spent through T1/T2 including the probe; credit **~$20.8** at T3
launch (well above the $3 escalation floor). Pods `drop32` (45755397), `r192_32`
(45755406), and `probe` (45755408) all completed their modes and were **destroyed**;
only `mk64` is live.

## Honest caveats

- Every new cell is **n=2–4**; one flipped trial moves a cell 25–50 points. The r192
  mk=100 vs r128 mk=75 gap is inside that noise.
- The bugSdrop == bug-r128 accuracy equality is **n=4 vs n=2** — profile-identical, not
  a matched-n equivalence test.
- H1 is refuted **at r128 on these three tasks**; a withholding effect below detection
  at this n, or at other ranks, is not excluded.
- The capture↔win link is trial-matched but aggregate-scored — suggestive, not airtight.
- r192 needle cells unmeasured (family saturates needle at every measured arm).
