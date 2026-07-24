# Week 11 — session handover (FINAL: pooled table + Q1/Q2 answered + r256 collapse)

> The single document a fresh session reads to resume work. Repo
> `/Users/hari/Desktop/kv-dlra`. Read this, then `docs/week11-decision-table.md`
> (current results doc) and `docs/week12-next-session.md` (the launch plan).
> Ethos: numbers straight, prefer honest negatives, no overclaim ("a lean, not a
> slam dunk"), all memory in one float-equivalent unit.

## 1. State

- Branch `week7` == `main` == **`8df36d4`**, pushed to origin.
- Suite **265 passed / 1 skipped**, ruff + mypy clean. Anti-drift pins in
  `tests/test_accounting.py` (high-rank+big-hh footprint case; bugS-r128-h1024@32K
  ~0.158x formula pin).
- vast.ai credit **~$25.94** (user topped up ~$24 mid-session; ~$20 GPU spent over
  10 pods). **Keys still unrotated.**

## 2. What is DONE

- **Pooled decision table** (all RULER sources merged as hits/total with per-source
  n): `results/w11-decision-table.json` + `results/w11-final-tables.md` +
  `docs/week11-decision-table.md`. Firmed 32K, both seeds:
  **bugS-r32-h256 = 100/67/100/100 (needle/mk/mv/vt) at 0.043×** (n=6–14) — the only
  sub-0.1× method covering all 4 tasks; bugEVICT-h256 = 100/0/0/0 at 0.009×;
  EA = 100/67/100/83 at 0.100×.
- **Q1 ANSWERED — basis warm-up window** (~4–5K absolute tokens, 8B r32): early
  items are never selected into the exact tier; the miss is budget-INDEPENDENT
  (capture 6/8@16K flat across hh=64..2048 → 7/8@32K flat; misses = earliest keys
  {0,1}@16K, {0}@32K, both trials). Retro-predicts mv@16K recall-0.75/accuracy-0 and
  vt 0@16K→100@32K. **Consequence: bugS is a ≥32K method; the 16K pick is EA.**
- **Q2 ANSWERED — rank is the ppl lever**: bugS-r128 ppl 4.17/4.16 @16K (EA 4.29),
  8.15/8.12 @32K (EA 8.28) at 0.14–0.19×. 32K retrieval r128 (n=4): needle 100,
  mk 75 (EA 67), mv 100, vt 75. 16K r128 weak (mk 25 / mv 25 / vt 0).
  **bugS-r128-h1024 is the balanced config** (~0.16×) — beats EA on ppl+mk @32K at
  ~1.6× EA memory. Attribution OPEN.
- **r256 follow-up — retrieval COLLAPSES**: all 12 hard-task cells (h256+h1024 ×
  16K+32K, n=4) = **0.00 accuracy AND 0.00 recall**, while r256 ppl is the best
  sub-full number (4.12/7.74 at 0.27–0.32×; full 4.08/7.62). Refutes the plain
  cleaner-basis story: a richer gist should retrieve at least as well; it retrieves
  nothing. Rank ladder: r32 = exact tier does the retrieving (post-warm-up),
  r256 = nothing, **r128 = unattributed narrow sweet spot**.
- **Probe detail behind that**: r128 exact tier is starved (0/8 codes at hh≤256 both
  ctx, never >3/8 up to 2048; the queried code is never in the tier) yet bugS-r128
  retrieves at 32K where plain FIFO bug-r128 = 0.
- **Recommendation (3 operating points)**: bugS-r32-h256 for retrieval/byte ≥32K;
  bugS-r128-h1024 for ~0.16× balanced (don't go higher in rank); EA at 16K.

## 3. Data map (all under `results/` unless noted)

- `w11-decision-table.json` — AUTHORITATIVE pooled table (pooled accs + `n_<task>`
  per cell). `w11-decision-table-base.json` — frozen pre-pool snapshot that makes
  merging idempotent. `w11-final-tables.md` — rendered tables.
- Lines-files (per-source RULER rows; each contributes its n to the pool):
  `w11-goalA-ruler-lines.txt` (16K 4-task, 7 baselines), `w11-goalB-ruler-lines.txt`
  (32K bugS/bugEVICT/ea/full), `w11-table-ruler-lines.txt`, `w11-base-ruler-lines.txt`,
  `w11-r128v1-ruler-lines.txt`, `w11-r128v2-ruler-lines.txt`,
  `w11-r256-ruler-lines.txt`. Ppl: `w11-goalB-ppl-lines.txt`,
  `w11-table-ppl-lines.txt`, `w11-base-ppl16-lines.txt`, `w11-r128-ppl-lines.txt`.
- Probes: `w11-probe8b-all.json` (8B per-code hh capture, Q1),
  `w11-probe-1b-mk-c{4096,8192,16384}-t*s*.json` + `w11-probe-1b-c*.json` (1B).
- **`scripts/w11_merge.py` contract**: to add new GPU results, save the printed row
  lines as a new `results/w11-*-lines.txt`, append it to `RULER_LOGS`, rerun —
  idempotent via the `-base.json` snapshot; regenerate the docs tables from its
  stdout. `gpu_logs/` is gitignored — committed lines-files in `results/` are the
  durable record.

## 4. Infra recipe that WORKS (vast.ai)

- Launch:
  ```bash
  OFFER=$(uvx vastai search offers 'num_gpus=1 gpu_name in [A100_PCIE,A100_SXM4] dph<0.8 reliability>0.99 rentable=true cuda_vers>=12.4 disk_space>=60 inet_down>=800' -o dph --raw | python3 -c "import sys,json;print(json.load(sys.stdin)[0]['id'])")
  uvx vastai create instance $OFFER --image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --disk 80 --onstart scripts/pod/<script>.sh --label kvdlra-w12
  ```
- Pod scripts: `scripts/pod/w11_r128.sh` (MODEs `new16/new32/firm16/firm32/ppl32/
  r256_16/r256_32`, `R256_HH` env) and `scripts/pod/w11_probe8b.sh`. Set MODE by
  `sed`-editing the `MODE=` line before `create instance` (onstart runs the committed
  file — push the branch first; pods git-clone from `origin/week7`).
- Validate before trusting: logs show `===DEPS_DONE===`, `torch 2.11.0+cu128 cuda
  True`, `===MODEL_OK===`.
- **Harvest = the printed per-row result lines ONLY.** `vastai logs` truncates every
  line at 500 chars (base64 emit blocks and long JSON lines are unusable); `vastai
  copy` silently creates empty dirs; SSH is unusable. Poll with
  `uvx vastai logs <id> --tail 200000`, grep the row lines into a
  `results/w11-*-lines.txt`, commit. **Destroy the pod after**:
  `printf 'y\n' | uvx vastai destroy instance <id>`.
- Sizing (measured per-trial, A100): r32@16K ~2min · r128@16K ~5min · r32@32K
  ~6.5min · r128@32K ~10–15min · r256@16K ~4–8min · r256@32K ~15–25min. (The v1
  full grid projected ~55 pod-hours and was replanned mid-flight — size first.)

## 5. Dashboards (update IN PLACE — pass `url:` from a new session)

- decision table (current): https://claude.ai/code/artifact/19e23647-d242-4310-896d-be2fb7e8ee0e
- overview (current): https://claude.ai/code/artifact/e811be6a-abb6-408a-89ec-d3fa8fd311d1
- explainer: https://claude.ai/code/artifact/c776074d-e7d4-475a-b325-1fb7eefe02d7
  — claims still true, but the r256 cliff is NOT
  yet added there — update if the story is touched.

## 6. Open questions → `docs/week12-next-session.md` for the launch plan

- **r128 attribution (bracketed, not closed)**: exact tier starved yet retrieval
  works — is it surprise-aware retention/withholding, or tier interplay? Next probe:
  hh_budget=0 ablation @32K (~$1); verify the harness accepts `--hh-budgets 0` for
  bugslash first (single-shot guard + arm naming). Then r192 (~$1.5) to measure how
  narrow the sweet spot is, + probe on the exact RULER (trial,seed) combos.
- **64K prediction test** (~$3–4): bugS-r32-h256 multikey should rise above 67 at
  65536 (all items exit the ~4–5K warm-up window). VRAM caution: prefer 80GB or
  chunk 2048; OOM = escalate, don't grind.
- **Warm-up mitigation**: can the earliest-token miss be fixed (e.g. seed the tier
  from the first block), making bugS viable at 16K?
- After any new run: merge lines-files via `w11_merge.py`, regenerate docs, update
  dashboards in place, update auto-memory, commit + push.
