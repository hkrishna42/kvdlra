# Week 14 — firm, extend, and promote the warm-up seed (the one funded lever)

> Continuation after Week 13. Repo `/Users/hari/Desktop/kv-dlra`, branch `week7`
> (**== `main`, pushed**, `740053c`). Suite green (**295 passed / 1 skipped**),
> ruff + mypy clean (tree unchanged since `740053c`). Credit **$8.40**
> (`uvx vastai show user --raw` → `credit`). Ethos: numbers straight, prefer honest
> negatives, no overclaim ("a lean, not a slam dunk"), all memory in one
> float-equivalent unit. Ultracode ON — Workflow fan-out with an adversarial
> number-verification stage before any claim enters a doc/dashboard. Keys still
> unrotated — flag, don't rotate mid-run.

## Context — why this week

Week 13 ran a four-track portfolio and got **exactly one funded win** — but it is a
good one. **`bugSseed`** (the warm-up seed) seeds the exact tier from the *first*
ingest chunk: it SLASH-routes the first block so its outliers can enter
`hh_k/hh_v/hh_pos` instead of bypassing the exact tier, and that **fixes bugS's 16K
retrieval collapse**:

- **16K r32 (0.05× memory) is the headline.** Plain `bugS` retrieves *only* the
  single-needle task — multikey / multivalue / var-track are all **0**, killed by the
  ~4–5K warm-up window. `bugSseed` lifts all three to **100** at identical memory.
  This **overturns Week-11's** "bugS is a ≥32K method; the 16K pick was EA."
- 32K r32: multikey 50→100 (the window bites at 32K too); mv flat; vt 100→75.
- ppl: small gain, **identical** `tok_eq/layer` — and `bugSseed`'s 32K ppl
  (9.164→**9.092** / 8.117→**8.085**) *equals Q-BUG's*, so the seed captures Q-BUG's
  ppl gain **and** the retrieval win, at plain-`bugS` memory.

Everything else was killed cheaply and honestly (T-A integrator surgery, T-C long-doc
calibration, T-X depth-PDE angles + rank). So Week 14 is **not** another exploration
sweep — it is the **consolidation week for the one real lever**: firm the n=2×2 result
to decision-grade statistics, close the coverage gap, decide whether the seed becomes
the default, and check the two places the win might reach further (64K, and a possible
*second* structural bypass). The bar is honesty, not hype — a 0→100 headline at n=2
must survive **n=8 with error bars** before it goes load-bearing on a dashboard or a
default. The README already names this as the next step ("higher-n (n≥8) confirmation
of the warm-up seed … with error bars").

## Current state (verified this session)
- Branch `week7` == `main` == `origin/{week7,main}`, all at **`740053c`**, tree clean.
  Suite **295 passed / 1 skipped**, ruff + mypy clean. Credit **$8.40**. **No pods
  running** (Week-13 pod `45887305` destroyed).
- The win ships as a **default-off** knob: flag `seed_hh_warmup`
  (`bug_cache.py:333`/`:418`, factory `:1744`/`:1795`). The seed fires only under
  chunked ingest via `_prefill` (`:1431`, condition `:1461`), routing the first block
  through `_absorb_block_slash` (`:1472`; writer at `:752`). A guard (`:452`) rejects
  seed + coded/quant/merge. Arm `--warmup-seed` / `bugSseed-*`. Identity pin
  (seed-off ⇒ bit-for-bit `bugS`) + accounting pin in `tests/test_bug_cache_*`.
- Pod MODE already exists: `wseed()` in `scripts/pod/w11_r128.sh:200` (matched A/B
  `bugS` vs `bugSseed`, currently `--n-trials 2 --seeds 0 1`). Week 14 = bump `n`,
  add the deferred cells.
- Data on `main`: `results/w13-trackb-summary.md`, `w13-wseed-{ruler,ppl}-lines.txt`,
  `w13-trackb-design.md`; Q-BUG in `results/w12-qbug-*`.

---

## Phase 0 — pre-flight ($0)
1. Re-verify green (`uv run pytest -q && uv run ruff check . && uv run mypy src tests
   scripts`), branch pushed, credit (`uvx vastai show user --raw` → `credit`).
2. Reproduce the `bugSseed` arm end-to-end on **CPU at a tiny size** (needle planted in
   the first block → lands in `hh_k`) — confirm the arm still works before spending a
   cent.
3. Size every planned cell from the cost table (below) and pin the pod MODE cell list.

## Phase 1 — $0 CPU / design work (no spend)
Two items, both free; run as a small Workflow with an adversarial verify stage.

**W-5 · second-structural-bypass probe** (the one new door, kept cheap).
- *Question.* The first-chunk bypass is fixed. Is it the **only** place a needle is
  structurally barred from the exact tier, or is there a second — a needle landing on a
  chunk boundary mid-stream, or across the graduation/eviction transition?
- *Method.* Same as T-B: trace every writer of `hh_k/hh_v/hh_pos` and every path that
  bypasses `_absorb_block_slash`; then a CPU probe on the 1B dumps
  (`dumps/llama3.2-1b/*len4096*`) that plants a needle at a chunk boundary and checks
  whether it can enter the exact tier under steady-state SLASH.
- *Fund* if a second real bypass with a cheap fix exists; *kill* if the first chunk is
  the only structural gap (steady-state SLASH already covers the rest — the belief we'd
  be confirming). **Honest prior: likely a kill, but $0 to know.**

**W-3-prep · promote scaffolding behind the flag** (build, do **not** flip).
- Write the "recommended-arm" plumbing and a **regression pin** proving that turning the
  seed on changes *only* the `bugS`/hh path and leaves every other arm (plain `bugS`,
  `bugevict`, coded, quant, merge) **bit-for-bit** — but leave `seed_hh_warmup`
  **default-off** until Phase-2 data lands.
- *Subtlety to bake in now:* a blanket code-default flip would trip the `:452` guard for
  coded/quant/merge, so promotion (Phase 3) is **docs + recommended arm + dashboard
  pick**, *not* a global default flip — unless a **scoped** default (on only when
  `hh_enabled and not (coded/quant/merge)`) is explicitly chosen, and then it needs its
  own pin.

## Phase 2 — GPU confirm the win to decision-grade stats (the core)
One matched-A/B pod (`bugS` vs `bugSseed`, same trials/seeds/footprint), MODE-driven off
`wseed`. **n=8/cell** unless noted (halves the 25-pt granularity to 12.5;
e.g. `--n-trials 4 --seeds 0 1`). Pre-registered bars:

- **W-1 · firm the r32 story (both contexts) — load-bearing, cheap.**
  - *16K r32 (the headline).* Bar: `bugSseed` multikey/multivalue/var-track each
    **≥ 87.5** (≥7/8) while `bugS` stays **≤ 12.5** ⇒ the 0→100 is real. Soften/kill:
    any hard task < 75.
  - *32K r32.* Bar: multikey **≥ 87.5** (firms 50→100); var-track **≥ 75** (resolves the
    100→75 one-trial dip — real regression or noise?); mv stays flat.
- **W-2 · close the coverage gap — r128 @32K RULER** (n=4 minimum, n=8 if budget). The
  one cell Week 13 deferred. Bar: no `bugSseed` cell regresses below `bugS` by > 12.5
  *that replicates*; ppl already known no-regression (−0.39%). Also firms the noisier
  16K r128 story (single −25, mv flat) if a cell is cheap to add.
- **rider · Q-BUG subsumption** — add a `bugSseedQ` arm at **one** 32K r32 cell. Since
  `bugSseed` ppl already *equals* `bugSQ`, test whether stacking buys anything beyond
  the seed alone. If not, Q-BUG is **subsumed** on the retrieval-first path (retire it
  from the recommended config; keep the knob).

**W-4 · 64K reach — STRETCH, gated.** Week 12 found 64K multikey confirmed (67→100) but
**multivalue regressed at n=2**, with the warm-up window named as the cause — so the seed
*should* recover it. Test `bugSseed` @64K on the mv (and mk) cell. Bar: seed 64K mv
≥ `bugS` mv (regression recovered). **Gate:** needs an **80GB card** (48GB OOMs) and only
runs if credit stays **> $3** after W-1/W-2; otherwise **defer to a top-up** — do not
grind the budget under the floor.

Pods launched in parallel where independent; a final `completeness-critic` asks "which
cell / wall did we not re-check?" Every surviving number is re-derived by the adversarial
verifier from the lines-file before it enters a doc.

## Phase 3 — the promote decision + close-out
- **Promote-or-not (data-gated).** If W-1 16K r32 holds at n=8 **and** no replicated
  regression anywhere: promote via **docs + recommended arm + dashboard pick** — flip the
  decision-table 16K pick to `bugSseed`, update overview + explainer **in place**. If the
  dips replicate: keep default-off and document the honest bounded win (clean at r32,
  noisier at r128) — **no promote**. Either outcome is honest and on the record.
- **Q-BUG:** record subsumed-or-not from the rider.
- Regenerate decision table + dashboards (update **in place**, pass `url:`), write
  `docs/week14.md`, update README current-focus, auto-memory (new Week-14 standing,
  supersede `[[kvdlra-week13-standing]]`), and `docs/week14-session-handover.md`.

## Harness engineering (NON-NEGOTIABLE — carry ALL forward)
- Green at every step: `uv run pytest -q && uv run ruff check . && uv run mypy src tests
  scripts`. Commit per verified increment; conventional commits; **never pipe
  `git commit` through `tail`** (a pre-commit hook failure was masked once) — commit
  bare, verify `git log -1`.
- New arms/knobs get a unit test + a `stored_state_numel` anti-drift pin **before any
  pod** (identity: seed-off ⇒ bit-for-bit `bugS`; retrieval-preservation;
  accounting) — mirror `tests/test_accounting.py`, `tests/test_bug_cache_*`.
- `bugslash`/`bugevict` REQUIRE `--chunk>0` (single-shot bypasses the exact tier).
- **Size runs from the measured cost table BEFORE launch:** r32@16K ~2 min/trial,
  r32@32K ~6.5, r128@16K ~5, r128@32K 10–15; 64K needs an 80GB card (48GB OOMs). A
  4-trial cell ≈ $0.6–0.8 → an n=8 cell ≈ 2×.
- **Harvest via printed row lines ONLY** (`^\[niah`/`^\[vt`, ppl `^  bugS…`/`^  bugSseed…`
  two leading spaces) — `vastai logs` truncates at 500 chars; base64/JSON unusable.
  `--tail 20000` (not 200000). ZSH: `${pair%%:*}` splitting, quote `===` markers.
- **DESTROY every pod** (`printf 'y\n' | uvx vastai destroy instance <id>`). Bounded
  waiters only. Pool RULER via `scripts/w11_merge.py` (extend `RULER_LOGS`); never
  hand-edit the table JSON.
- **ESCALATE (don't grind):** credit < $3, any 64K OOM, or a result contradicting a
  measured wall (a wall break gets re-verified before it's believed). Keys still
  unrotated — flag, don't rotate mid-run.

## Cost + budget
Credit **$8.40**; escalate < $3 → working budget ~$5. Phase-0/1 are **$0**. Estimates
(A100 40GB): W-1 r32 both ctx n=8 ≈ **$1.2–1.8**; `bugSseedQ` rider ≈ $0.5; W-2 r128@32K
n=4 ≈ **$1–1.5**. Core ≈ **$3–4**, leaving a buffer above the floor. **W-4 (64K) needs a
top-up** — an 80GB card at n=2×2 is ~$2–3 on its own; do NOT dip below $3 for it.

## Status cadence
`ScheduleWakeup` dynamic loop ~25–30 min — each check reports stage / %-done / ETA /
credit / errors; `stop:true` + PushNotification when done.

## Verification (end-to-end)
- Phase-1 probes print their pre-registered bar → fund/kill recorded; adversarial
  verifier re-derives by an independent path → CONFIRMED/PLAUSIBLE.
- Unit: seed-off ⇒ bit-for-bit `bugS` (identity pin); the Phase-1 regression pin proves
  every non-seed arm unchanged; accounting pin passes.
- Integration: each GPU cell beats its pre-registered bar **with n=8 error bars** AND the
  matched-A/B footprint is identical; regenerate table + dashboards; commit per verified
  increment; update docs + auto-memory + handover.
