# Week 11 — next-session plan (firm the leans, measure the balanced config, finish the dashboard)

> Continuation after the Week-11 decision-table run. Repo `/Users/hari/Desktop/kv-dlra`,
> branch `week7`, **HEAD `c1f869f` (pushed `origin/week7`)**. Suite **263 passed / 1
> skipped**, ruff + mypy clean. Read with `docs/week11-decision-table.md` (the current
> complete table), `docs/week11.md` (technical), auto-memory [[kvdlra-week11-standing]].

## Where we are (DONE)
- **SurpriseSLASH** built + committed: BUG's exact tier selects by low-rank **surprise**
  (`hh_select="surprise"` + `hh_neighbor` span-boost on `BugStreamingCache`). Control
  `bugEVICT` = rank-1 degenerate BUG. Arms in `scripts/w10_frontier.py::build_arms`
  (`bugslash`, `bugevict`), CLI `--hh-budgets`/`--hh-neighbor`.
- **Decision table COMPLETE** (8B, all methods, every cell, 16K + 32K):
  `docs/week11-decision-table.md`, `results/w11-decision-table.json`. Headline:
  `bugS-r32-h256` retrieves all 4 tasks at **0.043×** (needle/mv/vt=100, mk=83) — the
  only sub-0.1× method that does; matching accuracy elsewhere costs 7–20× more memory.
- **Dashboards (update IN PLACE):** `19e23647` (decision table), `e811be6a` (Week-11
  overview), `c776074d` (beginner explainer). GOAL-A benchmark repair done (4-task RULER
  + LongBench for all methods). Honest framing everywhere: **"gist helps = a lean, not a
  slam dunk"** (small n; budget/context-dependent).

## The two open questions (the user asked — answer with experiments)

### Q1 — why does `bugS-r32-h256` retrieve *better* at 32K than 16K on the hard tasks?
(16K: mk=33 mv=0 vt=0; 32K: mk=83 mv=100 vt=100.) Current hypothesis, to confirm/refute:
- **Mostly a task-construction + small-n artifact, NOT a bugS property.** Tell: *EA shows
  the same 16K→32K jump* (vt 12→100), so a different method family swings the same way →
  the cause is RULER's fixed key/value counts sitting denser at 16K, plus 2–6 trials/cell
  on all-or-nothing metrics.
- **Secondary real effect:** surprise separates a rare code better against a *blander*
  (more-filler) long-context background → sharper outlier at 32K.
- **Diagnose:** (a) `scripts/w11_probe.py` surprise-rank of the needle tokens at 16K vs
  32K (does the code rank higher among all tokens at 32K?); (b) inspect
  `scripts/w10_ruler.py::build_task` placement of keys/chain vs the recent window at each
  length; (c) **rerun the 16K hard tasks with n≥5 trials, both seeds** to see if the gap
  shrinks (noise) or holds (real). Report straight.

### Q2 — the perplexity gap; measure `bugS-r128-h1024`
`bugS-r32` trades text quality (ppl 8.9–9.2 vs EA 8.28). **Rank is the lever:** plain
BUG at rank-128 already beats EA on ppl (8.05 @ 0.13×) and rank-256 nears full (7.71).
**Hypothesis:** `bugS-r128-h1024` (~0.15×) closes the ppl gap AND keeps strong retrieval
(rich gist + big exact tier) → the **balanced "production" config** (better than EA on
*both* axes, at ~1.5× EA's memory). **Measure it** (+ `bugS-r128-h256`, `bugS-r256-h1024`
as the frontier) at 16K+32K, 4 RULER tasks + ppl; add rows to the table + dashboard.
Honest: the ppl gain is bought with memory (the overhead-floor tradeoff), so report the
memory cost exactly — it is a *quality-first operating point*, not a free win.

## The tasks (parallelize aggressively)
1. **New-config experiments (the core):** run `bugslash` at `--ranks 128 256` × `--hh-budgets
   256 1024` (+ keep r32 for continuity) at **16K and 32K**, 4 RULER tasks + ppl, n-trials≥4.
   Launch **two pods in parallel** (one 16K, one 32K) to finish fast.
2. **Firm the leans:** rerun the decision arms (`bugS-r32/128`, `bugEVICT`, `ea`) at 32K with
   **n≥5 trials, both seeds** to tighten bugS-vs-bugEVICT and the gist-helps claim.
3. **Q1 diagnosis** (cheap, parallel): the surprise-rank probe + task-construction read +
   the n≥5 16K rerun (folds into task 2's harness).
4. **Finalize:** merge all results → regenerate `docs/week11-decision-table.md` +
   `results/w11-decision-table.json` (add the r128/r256 rows); update the decision-table
   dashboard `19e23647` (add the balanced configs; a memory-vs-accuracy AND memory-vs-ppl
   view); update `docs/week11.md` + `week11-explained.md` + memory with the Q1/Q2 answers;
   commit + push `origin/week7`.

## Infra recipe (WORKS — all robustness fixes committed)
- **Pod:** vast.ai A100 (`gpu_name in [A100_PCIE,A100_SXM4]`, `reliability>0.99`,
  `disk_space>=60`, `inet_down>=800`), image `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`,
  `--disk 80`, `--onstart <script>`. Scripts self-clone `origin/week7` with **clone-retry**,
  pin kvpress 0.5.1 / transformers 5.8.0 / datasets 2.21.0, `HF_HUB_DISABLE_XET=1` +
  `HF_HUB_ENABLE_HF_TRANSFER=0` (robust dl). Reuse/adapt `scripts/pod/w11_table.sh`
  (BUG-family arms) and `w11_baselines.sh` (baselines) — clone into a `w11_r128.sh`.
- **Validate** each pod's `===DEPS_DONE=== / torch 2.11.0+cu128 cuda True / ===MODEL_OK===`
  before trusting it; host flakiness (GPU-error / clone early-EOF / HF timeout) → relaunch
  on a fresh high-inet host.
- **Harvest:** accumulate `uvx vastai logs <id> --tail N >> results/gpu_logs/<tag>.acc.log`
  each poll (base64 blocks between `===<M>_RESULT_BEGIN/_END===` → `base64 -d`; result LINES
  `^\[niah/^\[vt`, ppl `_log_row`). **DESTROY every pod after** (`printf 'y\n' | uvx vastai
  destroy instance <id>`). Credit ~$22 (`uvx vastai show user --raw`); keys work, no rotation.

## Harness-engineering rules (NON-NEGOTIABLE)
- Tests green at every step: `uv run pytest -q && uv run ruff check . && uv run mypy src tests
  scripts` (263/1). Commit per verified increment; conventional commits, no trailer.
- Count ALL memory in one float-equiv unit; every arm pinned to `stored_state_numel`
  (anti-drift tests). New arms/knobs get a unit test + an anti-drift pin.
- Report numbers straight; **prefer honest negatives; no overclaim** (keep "a lean").
  `bugslash`/`bugevict` REQUIRE `--chunk>0` (single-shot bypasses the exact tier).
- **BOUNDED waiters only** (`for i in $(seq 1 N); do ...; done`) — NEVER infinite `until`
  loops (they become stuck "running tasks"). `TaskStop` any leftover waiters + `ScheduleWakeup
  stop:true` when fully done.
- ESCALATE (don't silently retry): credit < $3; OOM at 32K; a result contradicts a measured
  wall; keys/host fail repeatedly.

## Multi-agent + status (ultracode is on)
- Use the **Workflow tool** for design/verify fan-out; launch **multiple pods simultaneously**
  (16K + 32K in parallel) and **background agents** for independent work (Q1 diagnosis writeup,
  doc updates) while pods run. Give each a self-contained prompt + the honest framing.
- **Status cadence:** poll every ~30 min via a `ScheduleWakeup` dynamic loop; each check,
  accumulate logs and **tell the user progress** (stage, %-done, ETA, credit, any errors).
  Destroy pods on `ALL_DONE`; assemble + update dashboards; then stop the loop.

## First actions
```bash
cd /Users/hari/Desktop/kv-dlra && git checkout week7 && git pull
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest -q            # 263 passed, 1 skipped
uvx vastai show user --raw | python3 -c "import sys,json;print(json.load(sys.stdin)['credit'])"
```
Then: read this file + `docs/week11-decision-table.md`; TodoWrite the 4 tasks; author a
`w11_r128.sh` pod script (bugslash `--ranks 128 256 --hh-budgets 256 1024`, 16K+32K, 4 tasks
+ ppl, n-trials 5); launch the 16K and 32K pods in parallel; arm a 30-min status loop.
