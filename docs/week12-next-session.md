# Week 12 — next-session plan (attribute the r128 sweet spot, test the 64K prediction)

> Continuation after the Week-11 pooled-table + r256 follow-up run. Repo
> `/Users/hari/Desktop/kv-dlra`, branch `week7` (**== `main` == `8df36d4`, pushed**).
> Suite **265 passed / 1 skipped**, ruff + mypy clean. Credit **~$25.94**
> (`uvx vastai show user --raw` → `credit`; the CLI "Balance" column is misleading).
> **Read first:** `docs/week11-session-handover.md` (ops + full context) and
> `docs/week11-decision-table.md` (the current pooled table). Auto-memory
> [[kvdlra-week11-standing]]. Ethos: numbers straight, prefer honest negatives, no
> overclaim ("a lean, not a slam dunk"), all memory in one float-equivalent unit.

## Where we are (DONE)
- **Pooled decision table complete** — every RULER cell rebuilt as hits/total across all
  lines-file sources via `scripts/w11_merge.py`, per-source n recorded:
  `results/w11-decision-table.json`, `results/w11-final-tables.md`,
  `docs/week11-decision-table.md`.
- **Q1 ANSWERED — the basis warm-up window.** Surprise is residual vs the streaming
  basis; items planted in the first **~4–5K absolute tokens** (8B r32) are never
  selected into the exact tier, and the miss is **budget-independent** (8B capture
  6/8 @16K flat across hh=64..2048 → 7/8 @32K flat; misses = the earliest keys
  {0,1}@16K, {0}@32K, both trials). RULER plants at relative positions, so longer
  contexts escape the window. Consequence: **bugS is a ≥32K method; the 16K pick is EA.**
- **Q2 ANSWERED — rank is the ppl lever.** `bugS-r128-h1024` is the balanced config:
  ppl 4.17/4.16 @16K (EA 4.29), 8.15/8.12 @32K (EA 8.28) at 0.14–0.19×; 32K retrieval
  (n=4) needle 100 / mk 75 (EA 67) / mv 100 / vt 75. Beats EA on ppl AND mk at ~1.6×
  EA's memory. 16K r128 retrieval is weak (mk 25, mv 25, vt 0) — consistent with Q1.
- **r256 follow-up — retrieval COLLAPSES.** All 12 hard-task cells (h256+h1024 ×
  16K+32K, n=4) = **0.00 accuracy AND 0.00 recall**, while r256 ppl is the best
  sub-full number (4.12/7.74 at 0.27–0.32×; full 4.08/7.62). This REFUTES the plain
  "cleaner basis" story: a richer gist should retrieve at least as well; it retrieves
  nothing. (r256 needle cells unmeasured — the family saturates needle.)
- **The rank ladder:** r32 = exact tier does the retrieving (post-warm-up); r256 =
  nothing; **r128 = an UNATTRIBUTED narrow sweet spot** between too-blurry-to-hold and
  too-well-fitted-to-select-or-surface. Probe detail that sharpens it: at r128 the
  exact tier is *starved* (0/8 codes captured at hh≤256, never >3/8 up to 2048; the
  queried code is never in the tier) — yet bugS-r128 retrieves at 32K where plain FIFO
  bug-r128 = 0.
- **Firmed 32K operating points (pooled, both seeds):** `bugS-r32-h256` =
  100/67/100/100 at **0.043×** (n=6–14), the cheapest point of the only sub-0.1×-capable
  family covering all 4 tasks (its h1024 sibling covers them too at 0.066×);
  `bugEVICT-h256` = 100/0/0/0 at 0.009×; EA = 100/67/100/83 at 0.100×. Recommendation
  stands: three operating points — bugS-r32-h256 (retrieval/byte, ≥32K),
  bugS-r128-h1024 (~0.16× balanced; don't go higher), EA at 16K.

## THE RESEARCH QUESTION: what carries bugS-r128's 32K retrieval?
Bracketed from both sides: it is NOT the exact tier holding the answer (probe: queried
code never captured at r128) and NOT a generically richer gist (r256 = richer gist,
retrieves nothing). Three hypotheses left standing:
1. **Withholding-from-absorption cleans the basis.** The surprise-aware retention path
   diverts outlier coordinates away from the low-rank absorption step, so the *gist
   itself* keeps the code recoverable — even if the diverted copies are later discarded.
2. **Partial/neighbor capture.** The tier holds neighbors or fragments of the planted
   span (the probe counts exact code-token capture only), and those partial exact
   coordinates are enough for the model to reconstruct the answer.
3. **Retention-ordering interplay.** Neither piece suffices alone; the interaction
   between what the tier holds and what the basis absorbed (ordering/timing of
   selection vs absorption) produces the retrieval.
T1 discriminates 1 vs {2,3}; the T2 trial-matched probe discriminates 2 vs 3.
Either way the attribution closes in — a clean negative is a result.

## The tasks (T1+T2 pods in parallel first; T3 after they land)
1. **T1 — `hh_budget=0` ablation at r128 @32K (~$1).** Does bugS-r128 retrieval survive
   with NO exact tier at all? Run `bugslash` r128, hh=0, 32K, hard tasks (mk/mv/vt),
   n = 2 trials × 2 seeds per cell.
   - **Survives** → the surprise-aware retention/withholding path is the mechanism (H1);
     narrow it further next.
   - **Dies** → the tier matters despite not holding the queried codes (H2/H3 —
     neighbor/partial capture or interplay).
   - **Harness check FIRST:** verify the harness accepts `--hh-budgets 0` for `bugslash`
     (the single-shot guard and arm naming both need checking). If a code change is
     needed, it gets a unit test + a `stored_state_numel` anti-drift pin BEFORE any pod
     launches.
2. **T2 — r192 point + trial-matched probe (~$1.5).** (a) `bugS-r192-h1024`: RULER hard
   tasks @32K n=4 + ppl @16K/32K — how narrow is the sweet spot between r128 (works)
   and r256 (collapse)? New rank ⇒ unit test + `stored_state_numel` pin. (b) Rerun
   `scripts/w11_probe.py --task niah_multikey` at r128 on the EXACT RULER (trial,seed)
   combos t{0,1} × s{0,1} to close the probe/RULER sampling gap (was the starved-tier
   probe measuring the same instances the RULER wins came from?).
3. **T3 — 64K prediction test (~$3–4, launch after T1/T2 land if credit allows).**
   `bugS-r32-h256` + **`ea` control** on multikey/mv/vt @65536, n=4. Prediction from Q1:
   mk rises above 67 (all planted items exit the ~4–5K warm-up window). **VRAM caution:**
   a 40GB A100 holds it on paper (weights 16GB + coords ~0.3GB + activations) but 64K
   sdpa chunk peaks are untested — prefer an 80GB host, or fall back to `--chunk 2048`;
   **OOM = escalate, don't grind.**
4. **T4 — merge, regenerate, publish.** Extend `RULER_LOGS` in `scripts/w11_merge.py`
   with the new lines-files → rerun (idempotent via
   `results/w11-decision-table-base.json`) → regenerate the docs tables from its stdout
   → update dashboards **IN PLACE** (full URLs — pass as `url` to the Artifact tool;
   if unavailable, find them via the artifact list):
   - decision table: https://claude.ai/code/artifact/19e23647-d242-4310-896d-be2fb7e8ee0e
   - overview: https://claude.ai/code/artifact/e811be6a-abb6-408a-89ec-d3fa8fd311d1
   - explainer: https://claude.ai/code/artifact/c776074d-e7d4-475a-b325-1fb7eefe02d7
   **update the explainer if the story shifts — and add the r256 cliff there either
   way** (its claims are true but it predates the r256 follow-up) → update docs +
   auto-memory → commit + push `origin/week7`.

## Pod/CLI recipe (adapt what works)
- Adapt `scripts/pod/w11_r128.sh` with new MODEs (existing: `new16/new32/firm16/firm32/
  ppl32/r256_16/r256_32`, `R256_HH` env) — add e.g. `hh0_32`, `r192_32`, `mk64`; probe
  runs go through `scripts/pod/w11_probe8b.sh` / `scripts/w11_probe.py`.
- vast.ai A100 (`gpu_name in [A100_PCIE,A100_SXM4]`, `reliability>0.99`,
  `disk_space>=60`, `inet_down>=800`), image
  `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`, `--disk 80`, `--onstart <script>`;
  scripts self-clone `origin/week7` with clone-retry, pin kvpress 0.5.1 / transformers
  5.8.0 / datasets 2.21.0, `HF_HUB_DISABLE_XET=1`, `HF_HUB_ENABLE_HF_TRANSFER=0`.
  **Create the SSH key BEFORE the instance; use onstart batch; SSH is unusable.**
- n = 2 trials × 2 seeds per cell; hard tasks only wherever the family saturates
  needle. Pool every new cell via `w11_merge.py` `RULER_LOGS` — never hand-edit the
  table JSON.

## Harness engineering (NON-NEGOTIABLE — carry ALL of these forward)
- Tests green at every step: `uv run pytest -q && uv run ruff check . && uv run mypy
  src tests scripts` (265/1). Commit per verified increment; conventional commits.
- **New arms/knobs (hh=0! r192!) get a unit test + a `stored_state_numel` anti-drift
  pin** (see the pins added in `tests/test_accounting.py`: high-rank+big-hh footprint
  case, bugS-r128-h1024@32K ~0.158× formula pin).
- `bugslash`/`bugevict` REQUIRE `--chunk>0` (single-shot bypasses the exact tier).
- **SIZE RUNS FROM THE MEASURED COST TABLE BEFORE LAUNCHING** (the v1 grid projected
  ~55 pod-hours and had to be replanned mid-flight). A100 per-trial costs:

  | run | ~time/trial |
  |---|---|
  | r32 @16K | ~2 min |
  | r128 @16K | ~5 min |
  | r256 @16K | ~4–8 min |
  | r32 @32K | ~6.5 min |
  | r128 @32K | ~10–15 min |
  | r256 @32K | ~15–25 min |

- **Harvest via the printed per-row result lines ONLY** (`^\[niah`/`^\[vt`, ppl
  `_log_row`): `vastai logs` TRUNCATES every line at 500 chars — base64 emit blocks and
  long JSON lines are unusable; `vastai copy` silently creates empty dirs; SSH unusable.
- `gpu_logs/` is **gitignored** — extract committed lines-files into `results/`.
- **DESTROY every pod** when its mode finishes: `printf 'y\n' | uvx vastai destroy
  instance <id>`.
- **BOUNDED waiters only** (`for i in $(seq 1 N); do ...; done`) — never infinite
  `until` loops. `TaskStop` leftovers + `ScheduleWakeup stop:true` when done.
- **ESCALATE (don't silently retry):** credit < $3; OOM at 64K; any result that
  contradicts a measured wall (e.g. r256 suddenly retrieving, bugEVICT winning a hard
  task). Keys still unrotated — flag it, don't rotate mid-run.

## Status cadence + multi-agent
- **`ScheduleWakeup` dynamic loop, ~25–30 min:** each check, accumulate logs
  (`uvx vastai logs <id> --tail N >> results/gpu_logs/<tag>.acc.log`), and report to
  the user: stage, %-done (trials landed / trials expected from the cost table), ETA,
  credit, any errors. Destroy pods on completion; stop the loop when all runs land.
- If multi-agent orchestration is available (ultracode/Workflow), use it for doc
  drafting with an adversarial
  number-verification pass before publishing anything (every number in a doc/dashboard
  traced back to a lines-file or the pooled JSON).

## First actions
```bash
cd /Users/hari/Desktop/kv-dlra && git checkout week7 && git pull
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest -q            # 265 passed, 1 skipped
uvx vastai show user --raw | python3 -c "import sys,json;print(json.load(sys.stdin)['credit'])"
```
Then: read `docs/week11-session-handover.md` + `docs/week11-decision-table.md`;
TodoWrite the 4 tasks; do the T1 harness check (`--hh-budgets 0`) locally with tests;
author the new pod MODEs in `scripts/pod/w11_r128.sh`; launch the **T1 and T2 pods in
parallel** (T3 after T1/T2 land if credit allows); arm the ~25–30 min status loop.

## Expected budget
T1 ~$1 · T2 ~$1.5 · T3 ~$3–4 → **total ~$6 of ~$26**. Escalate below $3 remaining.
