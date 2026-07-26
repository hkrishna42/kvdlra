# Week 13 — the parallel portfolio (four tracks, $0-probe-gated, adversarially verified)

> Continuation after the Week-12 Q-BUG close. Repo `/Users/hari/Desktop/kv-dlra`,
> branch `week7` (**== `main`, pushed**). Suite green (~291 passed / 1 skipped),
> ruff + mypy clean. Credit **~$14** (`uvx vastai show user --raw` → `credit`).
> Ethos: numbers straight, prefer honest negatives, no overclaim ("a lean, not a
> slam dunk"), all memory in one float-equivalent unit. Ultracode ON — Workflow
> fan-out with an adversarial number-verification stage before any claim enters a
> doc/dashboard.

## Context — why this week

Week 12 closed the two levers we understood:
- **Rank is spent.** r192/r256 *collapse* retrieval (mv/vt → 0) while ppl keeps
  falling; r128 is the largest rank covering all four RULER tasks. Cannot raise rank.
- **Q-BUG (query-metric whitened-key gist) shipped as an honest *bounded* result.**
  On 8B @32K it gives a real but small ppl gain and **misses both aggressive bars**:
  `bugSQ-r32` 9.164→**9.092** (target ≤8.90), `bugSQ-r128` 8.117→**8.085** (target
  <8.00) — a 0.4–1.2% gain. Retrieval preserved: multi-value exact, multi-key ~1
  trial soft. Key lesson: the CPU attention-error probe **over-predicted** the
  end-to-end ppl gain ~30–40× (a proxy-vs-downstream gap now on the record).

So the two obvious levers are a small win (Q-BUG) and a wall (rank). Week 13 opens a
**portfolio of four independent tracks** to find the next *real* lever, each gated by
a **$0 CPU probe** that funds or kills it before any GPU spend, and every surviving
number checked by an adversarial verifier that re-derives it from the dump/JSON.
Intended outcome: 1–2 funded tracks confirmed on GPU, the rest killed cheaply and
honestly, with the walls we did not break re-verified rather than assumed.

## Current state (verified this session)
- Pod **`45859737`** (`kvdlra-w12-qbug`, A100 40GB) is on its **final step** — the
  qbug MODE finished CALIB + all 4 ppl blocks + the r32 RULER gate and is completing
  the last `vt` row of the r128 RULER gate. **Bridge task below must harvest + verify
  + summarize + destroy it before Phase 1 launches.**
- Machinery for all $0 probes exists: `scripts/sigma_decay.py`,
  `scripts/w12_calibrate_qkey.py`, `scripts/w12_qbug_probe.py`, `scripts/w11_probe.py`;
  5 len-4096 dumps under `dumps/llama3.2-1b/doc{63,411,454,637,718}_*_rope-both`;
  per-bin ppl arrays in `results/w9-surprise-sweep-1b.json`.
- Source sites confirmed: truncation `streaming_torch.py:182-187` (`b_new =
  diag(sigma[:keep])` @:187); ingest hooks `bug_cache.py:781` (Q-BUG whiten),
  read `:1469`; first-block ingest `_ingest_chunk` @:1389 + SLASH select path
  `hh_k/hh_v/hh_pos` (:566-568, :764-810) for the warm-up fix.

---

## Bridge — close Week 12 (gates Phase 1)

Run **as soon as the pod's `vt` row lands** (`===RULER_bugSQ_r128_DONE===` /
`===ALL_DONE===`):
1. **Harvest** the printed rows only (`^  bugS`/`^  bugSQ` ppl, `^\[niah`/`^\[vt`
   RULER) via `uvx vastai logs 45859737 --tail 20000` → `results/w12-qbug-*-lines.txt`.
2. **Verify** every ppl/retrieval number against the baseline `bugS-*` rows (RULER
   gate = bugSQ within noise of bugS). Pull the bugS-r32/r128 32K baseline from the
   pooled table / prior lines-files for the gate comparison.
3. **Write `results/w12-qbug-summary.md`** (the honest bounded verdict + the
   proxy-vs-downstream gap) and update `docs/week12.md`.
4. **`printf 'y\n' | uvx vastai destroy instance 45859737`** — confirm gone.
5. Commit (`results(week12): Q-BUG GPU confirm — bounded ppl gain, retrieval
   preserved`); update auto-memory to a Week-13 standing.

---

## Phase 1 — five parallel $0 CPU probes (no spend; decides what gets built)

One Workflow: `pipeline([...])` of five `agent(...)` calls (phase `Probe`), each
returning `{metric, passed_bar, kill, evidence_path}`; then a **verify** stage per
result — an adversarial agent that re-derives the number by an *independent* method
(e.g. a separate SVD path) and returns `{confirmed, discrepancy}`. Barrier only to
collate the fund/kill decisions. **No silent caps** — `log()` anything dropped.

**T-A · Track-2 integrator surgery** — the other ppl lever.
- *Mechanism.* At 32K the basis is re-projected ~2000×; deep-horizon projection
  erosion / rank squeeze inflates long-context ppl — a stability property the
  Frobenius snapshot metric never sees (Axiom A does not cap it). Tikhonov-damped
  core (σ → σ²/(σ²+μ), μ tied to tracked σ_min) and/or energy-weighted truncation
  (σᵢ·‖cᵢ‖ instead of raw σ) at the single site `streaming_torch.py:187`.
- *Probe* (`probe:track2-truncation`). On `dumps/llama3.2-1b/*len4096*`: does
  damped / energy-weighted truncation lower long-horizon rank-r reconstruction error
  vs raw-σ? Reuse `sigma_decay.py` + the per-bin ppl arrays in
  `results/w9-surprise-sweep-1b.json`.
- *Bar:* deep-horizon (last-bin) error improves, **no moderate-band regression
  > 0.05 ppl**. *Kill:* damping bias regresses the moderate band.

**T-B · Warm-up retrieval fix** — the biggest *measurable* lever (pivots to retrieval).
- *Mechanism.* The 64K result confirmed the warm-up window is the dominant retrieval
  ceiling: the first ingest chunk bypasses SLASH, so the earliest-planted outliers
  structurally can't enter the exact tier, and the young basis scores everything
  ≈ equally surprising. Fix: seed the exact tier (`hh_k/hh_v/hh_pos`) from the first
  block in `_ingest_chunk` (`bug_cache.py:1389`) — defer the first block's graduation
  until the basis warms, or promote its top-surprise columns before absorption.
- *Probe* (`check:warmup-seed-hook`). **Design-check, not a metric probe:** confirm
  the first-block bypass structurally (trace `_ingest_chunk` → `_absorb_columns`),
  identify the exact seeding site, and specify the cheap unit test (a needle planted
  in the first block lands in `hh_k`). `passed_bar` = a concrete, test-backed hook
  design exists; `kill` = the bypass isn't real or seeding can't preserve the ≥32K
  wins. (This is a *retrieval* lever — deliberately re-scoped in vs. the Week-12
  pure-ppl plan, because the 64K test made it the largest measurable win.)

**T-C · Q-BUG follow-ups** — firm the soft result.
- *Probe* (`calib:wkey-longdocs`). Re-calibrate `w_key` on **long** docs (Week-12
  used ~580-tok C4 docs): `scripts/w12_calibrate_qkey.py --seq-len 8192 --n-docs 16`,
  then re-run `w12_qbug_probe.py` — does a cleaner L move the predicted ppl gain
  and/or remove the multi-key softening? *Bar:* cleaner-L attention-error gain
  exceeds the short-doc L by a margin that plausibly clears a bar downstream (noting
  the 30–40× proxy gap — treat as *relative* signal, not an absolute ppl promise).
  *Kill:* long-doc L ≈ short-doc L (no headroom). Higher-n RULER (n=8/cell) and
  per-head full-L are **Phase-2** follow-ups, not $0 probes.

**T-X · Exploratory PDE framings** — cheap kills only, build nothing yet.
- *Probe* (`probe:layer-angles`). **Depth-continuous basis** (lead): principal
  angles between adjacent-layer pre-RoPE key subspaces on `dumps/llama3.2-1b/*`.
  *Fund* if small (a shared basis + ∂U/∂ℓ could attack the overhead floor / Axiom B);
  *kill* if ≳ 60°.
- *Probe* (`probe:rank-vs-depth`). **Mean-field / interacting-particle view:**
  effective-rank-vs-depth plot on the dumps (tokens cluster with depth ⇒ effective
  rank should collapse; the needle is a Dirac mass resisting the drift). Explains
  "why," may sharpen rank allocation. RoPE multiple-scales is noted but out of scope
  for the $0 round.

---

## Phase 2 — build + GPU-confirm the funded track(s) (gated on Phase 1)

For each **funded** probe, `pipeline(item, implement → unit-test + accounting-pin →
size-run → launch-pod → harvest → verify)`. Pods launched **in parallel where
independent** (T-A ppl and T-B retrieval are separable). A final
`completeness-critic` agent asks "what's unverified / which wall did we not
re-check?"

- **T-A (if funded):** implement damped/energy truncation behind a flag with an
  `L=I`/`μ=0` identity unit test + `stored_state_numel` accounting pin; GPU confirm
  ppl @16K/32K at r32/r128 + a RULER retrieval gate. Composable with Q-BUG.
- **T-B (if funded):** implement the seed hook; unit test (needle-in-first-block →
  tier) + `stored_state_numel` pin **before any pod**; RULER gate 16K/32K (does mk/vt
  @16K rise from the current 14/0?); **regression-gate 32K** to keep the ≥32K wins.
- **T-C (if funded):** higher-n bugSQ RULER at n=8/cell (separate real multi-key
  softening from noise) and/or the long-doc-L ppl re-confirm.

Every finding passes the adversarial number-verification agent (trace to
lines-file / JSON / dump) before it enters a doc or dashboard.

## Harness engineering (NON-NEGOTIABLE — carry ALL forward)
- Green at every step: `uv run pytest -q && uv run ruff check . && uv run mypy src
  tests scripts`. Commit per verified increment; conventional commits; **never pipe
  `git commit` through `tail`** (a pre-commit hook failure was masked once) — commit
  bare, verify `git log -1`.
- New arms/knobs get a unit test + a `stored_state_numel` anti-drift pin **before any
  pod** (mirror `tests/test_accounting.py`, `tests/test_bug_cache_qbug.py`:
  `L=I`/`μ=0` identity, retrieval-preservation, accounting).
- `bugslash`/`bugevict` REQUIRE `--chunk>0` (single-shot bypasses the exact tier).
- **Size runs from the measured cost table BEFORE launch:** r32@32K ~6.5 min/trial,
  r128@32K 10–15, r256@32K 15–25, r32@16K ~2, r128@16K ~5; 64K needs an 80GB card
  (48GB OOMs). A 4-trial cell ≈ $0.6–0.8.
- **Harvest via printed row lines ONLY** (`^\[niah`/`^\[vt`, ppl `^  bugS…` two
  leading spaces) — `vastai logs` truncates at 500 chars; base64/JSON blocks
  unusable. `--tail 20000` (not 200000). ZSH: `${pair%%:*}` splitting, quote `===`.
- **DESTROY every pod** (`printf 'y\n' | uvx vastai destroy instance <id>`). Bounded
  waiters only. Pool RULER via `scripts/w11_merge.py` (extend `RULER_LOGS`); never
  hand-edit the table JSON.
- **ESCALATE (don't grind):** credit < $3, any 64K OOM, or a result contradicting a
  measured wall (a wall break gets re-verified before it's believed). Keys still
  unrotated — flag, don't rotate mid-run.

## Cost + budget
Credit ~$14. Phase-1 probes are **$0**. Phase-2 GPU: T-A confirm ≈ $2–3, T-B
retrieval gate ≈ $1.5–2, T-C higher-n RULER ≈ $1.5. **Escalate below $3.**

## Status cadence
`ScheduleWakeup` dynamic loop ~25–30 min — each check report stage / %-done / ETA /
credit / errors; `stop:true` + PushNotification when done.

## Verification (end-to-end)
- CPU probes each print their pre-registered bar (Δ error %, deep-horizon ppl Δ,
  principal angle °, effective-rank curve) → fund/kill recorded; adversarial verifier
  re-derives each by an independent path → CONFIRMED/PLAUSIBLE.
- Unit: identity pins (`L=I`, `μ=0`) reduce every new knob to baseline; accounting
  pin passes; retrieval-preservation gate for any tier-adjacent change.
- Integration: funded-track GPU confirm beats its pre-registered bar AND passes its
  retrieval gate; regenerate decision table + dashboards; commit; update docs +
  auto-memory; write `docs/week13-session-handover.md`.
