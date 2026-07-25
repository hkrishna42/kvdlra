# Week 12 — session handover (T1 H1-refutation + T2 r192/probe DONE; T3 64K RUNNING)

> The single document a fresh session reads to resume work. Repo
> `/Users/hari/Desktop/kv-dlra`. Read this, then `docs/week11-decision-table.md`
> (current results doc) and `docs/week12-next-session.md` (task statuses).
> Ethos: numbers straight, prefer honest negatives, no overclaim ("a lean, not a
> slam dunk"), all memory in one float-equivalent unit.

## 1. State

- Branch `week7` == **`6a5a642`**, pushed to origin. `main` still at `69fe054`
  (4 week-12 commits behind); the ff-merge to `main` is part of T4, **after T3
  lands**. Working tree carries the UNCOMMITTED week-12 doc outputs (they go in
  with the T4 commit): regenerated `docs/week11-decision-table.md` +
  `results/w11-final-tables.md` (week-12 rows + update note), status-edited
  `docs/week12-next-session.md`, new `docs/week12.md` and this handover, plus a
  whitespace-only (trailing-newline) touch of `results/w11-decision-table.json`
  from a merge rerun.
- Suite **277 passed / 1 skipped** (278 collected), ruff + mypy clean. New week-12 anti-drift
  pins: the `hh_retain=False` select-and-discard mode is pinned **bit-identical
  in layer state to bugS given the same K/V stream** (unit test), and the r192
  memory-ratio formula is pinned in the accounting tests.
- vast.ai credit **~$20.8** at T3 launch (~$5.1 of $25.94 spent across the
  T1/T2/probe pods). **Keys still unrotated** — flag it, don't rotate mid-run.

## 2. What is DONE (Week-12 T1 + T2)

- **Design finding first**: the pre-registered `hh_budget=0` ablation could not
  discriminate H1 — hh=0 degenerates to plain `bug-r128` (hh_enabled False turns
  the whole SLASH path off). Built instead: `hh_retain=False` **"bugSdrop"
  (select-and-discard)** — selection + withholding IDENTICAL to bugS (pool kept,
  feeds re-selection/demotion), tier INVISIBLE to attention. Harness flag
  `--hh-discard`, arm name `bugSdrop-r128-h1024`. The pool is live state, so the
  arm honestly reports the same 0.159× — it is an **ablation instrument, NOT a
  cheaper operating point**.
- **T1 verdict — H1 REFUTED at r128.** Hiding the tier kills mv/vt outright
  (100→0, 75→0) and halves mk (75→50): `bugSdrop-r128-h1024` @32K = mk 50 /
  mv 0 (recall 0.00) / vt 0 at 0.159× (n=4). Its accuracy profile is IDENTICAL
  to plain `bug-r128` (50/0/0 at 0.130×, n=2 gap-fill).
  **Withholding-from-absorption contributes nothing detectable on these tasks at
  this n.** The visible exact tier IS the mechanism of bugS-r128's edge (H2/H3
  family).
- **mk has a genuine gist-only component**: plain `bug-r128` alone scores 50 on
  mk (the tier adds the rest: 50→75). mv/vt have NO gist-only component (0
  without a visible tier).
- **SCOPING FIX (apply everywhere)**: the Week-11 claim "plain FIFO bug-r128 =
  0" was scoped to **needle@32K (n=2)**; its 32K hard-task cells were UNMEASURED
  until this gap-fill (16K precedent: mk 38 / mv 38 / vt 0). Do not present
  mk=50 as contradicting a measured wall.
- **T2a — r192 is a task-staggered collapse, NOT a single cliff**:
  `bugS-r192-h1024` @32K = mk 100 (≥ r128's 75, n=4) / mv 0 (recall 0.25) /
  vt 0 at 0.222×, while ppl keeps improving with rank — 4.124 @16K (0.254×) /
  7.867 @32K (0.222×), vs r128 4.16/8.12 and r256 4.12/7.74. The "balanced
  config" claim survives ONLY at r128: it is the largest rank that still covers
  all four tasks.
- **T2b — trial-matched probe (r128, mk, hh 256+1024, combos t{0,1}×s{0,1})**:
  the W11 "queried code never in the tier" premise does NOT hold trial-matched —
  capture is **INSTANCE-DEPENDENT**. 32K: t0s0+t0s1 starved (codes 0/8 @256,
  2/8 @1024, needle_in_hh 0/3); t1s0 @1024 FULL capture (needle_in_hh 3/3,
  captured=True); t1s1 PARTIAL at both budgets (needle_in_hh 2/3). 16K: all four
  combos starved (see `results/w12-probe-r128-lines.txt`). CONSISTENT with
  partial capture carrying the wins (H2) — but per-trial RULER outcomes are cell
  aggregates, so the capture↔win correlation is **suggestive, not airtight**.
- **n-caveats**: every new cell is n=2 or n=4; one flipped trial moves a cell
  25–50 pts.

## 3. Data map (all under `results/` unless noted)

- `w11-decision-table.json` — AUTHORITATIVE pooled table (pooled accs +
  `n_<task>` per cell; now includes the week-12 arms).
  `w11-decision-table-base.json` — frozen pre-pool snapshot (idempotent merges).
  `w11-final-tables.md` — rendered tables. Regenerate ONLY via
  `uv run python scripts/w11_merge.py` — never hand-edit the JSON.
- **New week-12 lines-files** (the first four are wired into
  `RULER_LOGS`/`PPL_LOGS`; the probe file is standalone evidence, NOT merged):
  - `w12-drop-ruler-lines.txt` — T1 `bugSdrop-r128-h1024` @32K (n=4/row).
  - `w12-bugr128-ruler-lines.txt` — T1 plain `bug-r128` 32K hard-cell gap-fill
    (n=2/row).
  - `w12-r192-ruler-lines.txt` — T2a `bugS-r192-h1024` @32K (n=4/row).
  - `w12-r192-ppl-lines.txt` — T2a ppl rows @16K/32K.
  - `w12-probe-r128-lines.txt` — T2b probe evidence with per-combo markers
    (`===PROBE_c<ctx>_t<t>s<s>_BEGIN/DONE===` blocks).
- **Pending T3 lines-files, names already wired into `RULER_LOGS`**:
  `w12-mk64-mk-lines.txt` ({"65536": 4}) and `w12-mk64-mvvt-lines.txt`
  ({"65536": 2}) — harvest the mk64 pod rows into exactly these paths and rerun
  the merge.
- **Probe out-file naming convention**: the pod writes
  `results/w12-probe8b-mk-r128-c{16384,32768}-t{0,1}s{0,1}.json`
  (`--out-json` in `scripts/pod/w12_probe_r128.sh`), but those JSONs stay
  pod-side (`vastai logs` truncates; `vastai copy` is broken) — the committed
  lines-file marker blocks are the durable record. Same story for
  `w12-mk64.json`/`w12-mvvt64.json` (the mvvt out-json is named `w12-mvvt64`,
  not `w12-mk64-mvvt`).
- `gpu_logs/` is **gitignored** — committed lines-files in `results/` are the
  durable record. Week-11 lines-files and probes: see the data map in
  `docs/week11-session-handover.md` (unchanged).

## 4. Infra recipe that WORKS (vast.ai) — unchanged from Week 11, PLUS three gotchas

Everything in `docs/week11-session-handover.md` §4 still holds (offer search,
image `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`, onstart from the pushed
branch, `===DEPS_DONE===`/`===MODEL_OK===` validation, harvest = printed row
lines only, destroy every pod). Pod script now: `scripts/pod/w11_r128.sh` MODEs
`new16/new32/firm16/firm32/ppl32/r256_16/r256_32/drop32/r192_32/mk64` +
`scripts/pod/w12_probe_r128.sh`. mk64 **needs an 80GB card — 48GB OOMs**.
New this week:

- **zsh word-splitting gotcha (log-polling loops)**: zsh does NOT word-split
  unquoted `$VARS` and DOES glob bare `===` patterns — a
  `for id in $POD_IDS`-style poll loop sees one giant string, and inline `===`
  markers in compound commands can error out (`== not found`). Use explicit
  lists / `${=VAR}`, and quote marker strings.
- **`--tail 20000`, not 200000**: poll with
  `uvx vastai logs <id> --tail 20000` — the oversized tail is the thing that
  bit us, keep it at 20000.
- **Commit discipline: NEVER pipe `git commit` through `tail`** (or anything) —
  a pre-commit hook failure was masked once and the commit silently didn't
  land. Run `git commit` bare, then verify with `git log --oneline -1`.

## 5. Dashboards (update IN PLACE — pass `url:` from a new session)

**Updated in place 2026-07-24 with the Week-12 T1/T2 results** (bugSdrop +
r192 rows, the H1-refuted attribution verdict, the r256 cliff added to the
explainer; drafted via a 12-agent workflow with an adversarial
number-verification pass, then republished at the same URLs). **The 64K rows
are still pending** — fold them in after T3 lands.

- decision table (current): https://claude.ai/code/artifact/19e23647-d242-4310-896d-be2fb7e8ee0e
- overview (current): https://claude.ai/code/artifact/e811be6a-abb6-408a-89ec-d3fa8fd311d1
- explainer: https://claude.ai/code/artifact/c776074d-e7d4-475a-b325-1fb7eefe02d7

## 6. Open questions

- **T3 64K prediction test RUNNING** (pod MODE `mk64`: `bugS-r32-h256` + `ea`
  @65536, mk n=4, mv/vt n=2). Prediction from W11 Q1: **mk rises above 67** as
  all planted items exit the ~4–5K warm-up window. When it lands: harvest into
  the two pre-wired `w12-mk64-*` lines-files, rerun `w11_merge.py`, regenerate
  docs, update the three dashboards in place, update auto-memory, commit,
  ff-merge `main`, push. OOM = escalate, don't grind.
- **Capture↔win correlation is not closed**: the trial-matched probe gives
  per-instance capture, but RULER outcomes are cell aggregates — making the H2
  correlation airtight needs **per-trial RULER outcome rows, i.e. a harness
  change** to print per-trial results. Until then, "partial capture carries the
  wins" stays a lean, not a slam dunk.
- **Warm-up mitigation**: can the earliest-token miss be fixed by **seeding the
  tier from the first block**, making bugS viable at 16K?
- **Is mk's gist-only 50 itself a warm-up story?** Plain bug-r128 gets 50 on mk
  with no tier — is that the late-planted items surviving in the gist while the
  early ones die in the warm-up window (same mechanism, different tier)? Cheap
  to check against per-position outcomes if per-trial rows land.
