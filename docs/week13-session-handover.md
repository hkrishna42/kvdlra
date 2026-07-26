# Week 13 — session handover (resume state)

> The single doc a fresh session reads to resume. Repo `/Users/hari/Desktop/kv-dlra`,
> branch `week7` (== `main`, pushed). Then read `docs/week13-plan.md` (the forward
> plan / next-session prompt), `results/w12-qbug-summary.md` +
> `docs/week12-qbug-explainer.md` (the Q-BUG close). Ethos: numbers straight, prefer
> honest negatives, no overclaim, all memory in one float-equivalent unit. Ultracode
> ON — Workflow fan-out with an adversarial number-verification stage before any
> claim enters a doc/dashboard.

## 1. State
- Branch `week7` == `main`, pushed to origin (verified `git log -1` both). Suite green
  (~291 passed / 1 skipped), ruff + mypy clean. Credit ~$13.9
  (`uvx vastai show user --raw` → `credit`). **Keys still unrotated** — flag, don't
  rotate mid-run.
- No pods running (the Q-BUG confirm pod 45859737 was destroyed).

## 2. What is DONE
- **Week-12 fully closed:** r128 attribution (visible exact tier carries retrieval;
  H1 refuted), r192 task-staggered collapse, 64K prediction (multi-key confirmed
  67→100, multi-value regressed at n=2). Dashboards updated in place. On `main`.
- **Q-BUG (bugS-ppl Track 1) — shipped as an honest bounded result.** Query-metric
  key whitening (`w_key`, a frozen per-feature diagonal) gives a real but small ppl
  gain (r32 9.164→9.092, r128 8.117→8.085; both aggressive bars **missed** by 3–4×)
  at ~zero memory, retrieval preserved within n=4 noise (multi-value exact, multi-key
  ~1 trial soft). Default-off knob with unit + accounting pins
  (`tests/test_bug_cache_qbug.py`). Key lesson: the CPU attention-error probe
  over-predicted end-to-end ppl ~30–40× (proxy-vs-downstream gap, on the record).
  See `results/w12-qbug-summary.md`, `docs/week12-qbug-explainer.md`, `docs/week12.md`.
- **Week-13 Phase-1 CPU probes ($0) — DONE, honest near-negatives:**
  - **Track-C** (Q-BUG long-doc calibration): **KILLED**. Long-doc L is 99.6% aligned
    with the short-doc L; attention-error gain margin +0.2% (bar 3%). Calibration is
    NOT the ppl bottleneck. (`results/w13-trackc-probe.json`.)
  - **Track-X depth-PDE / cross-layer basis:** **LEAN-KILL**. Adjacent-layer pre-RoPE
    key subspaces sit at 57.7° leading vs 62° random control (fund<40/kill>60), no
    shared basis → won't amortize the overhead floor. Effective-rank-vs-depth also
    recorded. (`results/w13-trackx-angles.json`, `w13-trackx-rank.json`.)
  - **Track-B** (warm-up retrieval fix): code-fact trace **CONFIRMED** the bypass
    (`_prefill` → `_absorb_columns`, never `_absorb_block_slash`; slash only runs via
    `_absorb_block_into_stream`), so the tier-seeding hook in `_prefill` is valid —
    **this track survives and is the lead build candidate.** (`results/w13-trackb-*`.)
  - **Track-A** (integrator surgery): probe recorded (`scripts/w13_tracka_probe.py`).

## 3. Data map (under `results/` unless noted)
- Q-BUG: `w12-qbug-ppl-lines.txt` (harvest, 2-leading-space ppl rows),
  `w12-qbug-ruler-lines.txt` (retrieval gate rows), `w12-qbug-summary.md`,
  `w12-qbug-probe.json` (the CPU probe). The 8B `w_key` is NOT committed (regenerable
  via `scripts/w12_calibrate_qkey.py --device cuda`; the arm loads it with
  `--qwhiten-file`).
- Week-13 Phase-1: `w13-trackc-probe.json`, `w13-trackx-angles.json`,
  `w13-trackx-rank.json`, `w13-trackb-facts.json`, `w13-trackb-design.md`; scripts
  `w13_tracka_probe.py`, `w13_trackb_bypass.py`, `w13_trackc_probe.py`,
  `w13_trackx_angles_probe.py`, `w13_trackx_rank_probe.py`.
- Real 1B K/V/Q dumps (CPU-probe fuel, gitignored): `dumps/llama3.2-1b/*_len4096_rope-both`
  (now with `Q_pre` + `rope.pt` via `capture_kv.py --with-q`).

## 4. Infra recipe (unchanged from Week 12) + the standing gotchas
vast.ai A100 (`dph<0.8 reliability>0.99 disk_space>=60 inet_down>=800`), image
`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`, `--disk 80`, `--onstart` the
sed-MODE'd `scripts/pod/w11_r128.sh`; pods clone `origin/week7` (push first). Gotchas:
**zsh** doesn't word-split unquoted vars and globs bare `===` — use `${pair%%:*}`,
quote markers; **`--tail 20000`** (not 200000); **never pipe `git commit` through
`tail`** (a hook failure was masked once) — commit bare, verify `git log -1`;
**DESTROY every pod**; harvest printed row lines ONLY.

## 5. Dashboards (update IN PLACE — pass `url:`)
- decision table: https://claude.ai/code/artifact/19e23647-d242-4310-896d-be2fb7e8ee0e
- overview: https://claude.ai/code/artifact/e811be6a-abb6-408a-89ec-d3fa8fd311d1
- explainer: https://claude.ai/code/artifact/c776074d-e7d4-475a-b325-1fb7eefe02d7 (Q-BUG section added Week-12 close)

## 6. Open questions → `docs/week13-plan.md`
The portfolio's fund/kill after Phase-1: **T-B (warm-up retrieval fix) is the lead**
(mechanism confirmed, biggest measurable lever per the 64K result — build the
tier-seeding hook in `_prefill` with a unit test + retrieval gate). **T-A (integrator
surgery)** pending its probe read. **T-C** killed (calibration not the bottleneck).
**T-X depth-PDE** lean-killed (layers near-orthogonal). Remaining Q-BUG follow-up: a
higher-n (n=8) matched retrieval re-run to firm the ~1-trial multi-key softness.
Phase-2 = build + GPU-confirm the survivor(s); size from the cost table; adversarial
number-verify before any claim.
