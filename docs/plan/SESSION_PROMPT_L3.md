# Next-session prompt — L3 (+ L5) (paste as the first message; cwd ~/src/kv-dlra, branch main)

You are the orchestrator for kvdlra Weeks 0–3 (docs/plan/ICML2027_PLAN.md §1–§2), continuing from the last
entries of docs/plan/STATE.md (2026-09-19: L2 merged into `main` at e116212; `main` is the only branch; D-005
CLOSED — the cycled-filler generator is retired, every retrieval pod runs generator v2). Read, in order, and do
not summarize back: CLAUDE.md; docs/plan/STATE.md (the last four entries); docs/plan/DECISIONS.md (D-005 CLOSED
and addenda 1–3, D-017, D-006, and the OPEN items D-001–D-004, D-007, D-010);
docs/plan/plans/2026-09-11-L3-L5-gate1-bf16-prereg.md (APPROVED 2026-09-17 — written before L0/L1/L2 landed);
docs/plan/lanes/L3_gate1_tracker_swap_v2.md, L5_bf16_gist_and_prereg.md and GATES.md §G3/§G5;
docs/plan/ICML2027_PLAN.md §2 Gate 1 (the branch rule, verbatim); prereg/hygiene_table4.md and
prereg/ss2_families.md (the house style a Gate-1 prereg must reach — both survived an opus "review as a
scientist" round); docs/plan/cleanup/l2-ledger.md (skim PR-L2-5/7/10/14/23 and R-L2-2…10 — the harness
conventions L3 inherits). Then state in one line each: what G3 needs, what Gate 1 costs at the RATES L2
MEASURED (not the plan's), and why the ss2 pods wait on Gate 1's prereg.

Mode: superpowers subagent-driven-development exactly as L1 and L2 ran it — fresh implementer per task, a task
review (spec + quality) and a scoped re-review per task, a whole-branch review + ponytail-review + the
clean-clone L6 gate before the finishing menu; /ponytail full; the ledger is the recovery map. ONE worktree,
ONE lane for L3 + L5: the plan says they are disjoint, but L3.1 and L5.1 both edit `bug_cache.py`'s
constructor/dispatch and `POST_V1` — run the tasks sequentially in this order: L3.1 (frozen/random trackers +
byte-matched no-gist arm) → L5.1 (bf16 gist storage, default-off, bit-identical when off) → L3.2 (Gate-1 pods,
`gate1` table, `gate1_verdict()`) → L5.2 (preregs gate1_tracker_swap_v2 / bf16_gist / kernel_smoke; the SHA-order
and tamper tests) → launch Gate-1 stage 1 ONLY after D-003's top-up lands.

Workspace: `git worktree add .claude/worktrees/L3-gate1 -b lane/L3-gate1-tracker-swap-v2 main`; symlink the
shared `.venv` and `dumps/llama3.2-1b` into it; `uv pip install -e ".[dev]"` from the worktree and
`chflags nohidden` on the venv's `.pth` files (iCloud re-hides them within minutes — re-run it before any bare
`.venv/bin/python` use; delete any `_editable_impl_kvdlra 2.pth`/`3.pth` twin); baseline `make test` = 538
passed (≈ 75–87 s on this Mac — the 90 s gate is tight; every new test stays light). The shell cwd resets to the
main checkout after every agent/monitor notification in this harness: prefix worktree commands with
`cd .claude/worktrees/L3-gate1 &&`, and delete any stray `.superpowers/` the SDD helpers write into the main
checkout.

Pre-flight before L3.1 (the plan predates L0/L1/L2): audit every path, symbol, YAML shape and test the plan
names against `main` and write the drift table + rulings to the ledger as PR-L3-* entries — in particular:
`kvdlra.tracker.TRACKERS` and the `step(u, b, block, rank_cap, **kw) -> (u, b, rot)` contract (L1; `oja_step`
grows to `rank_cap` and takes `eta0/decay`; `fd_step` keeps its own `k`); the existing arms
`oja_r64_h256_seed_tuned` (the tuned Oja) and `fd_r64_h256_seed`; `isvd_r64_h256_seed`'s `doc:` is FROZEN (a live
manifest pins its hash — `doc:` is inside `config_hash`) so new variants are new files; `TaskV2Cfg` + the four
`ruler_v2_*` tasks (`ruler_v2_16k_g1` = the n=24 design the plan names; there is NO `ruler_v2_32k_g1` yet —
create it); `ppl_16k_wt103test_g1` at 16 windows is IMPOSSIBLE (WikiText-103 test supplies 14 windows at
16K+2048 — `load_task` refuses it): Gate 1's perplexity runs `ppl_16k_pg19val_w16` (exists) and a
`ppl_32k_pg19val_w16` (create); `[trial]` lines carry `hay= depth= code= sha=` and `[stage] dataset_sha256`
lines reach the manifest through harvest; `pod.py launch --max-hours` (default = `gpu_budget_h`) and the
watchdog's derived `BUDGET_ITERS` (run it under `caffeinate -s -i`); `stats.mcnemar_exact/holm/tost/
paired_bootstrap` and `tables.py`'s subcommands (`build`, `ppl`, `table_baselines` — no `gate1` yet); the
`gist_dtype` boundary vs `_footprint`'s live-rank billing and `stored_state_numel` (bf16 must bill 16 bits);
`tests/test_golden_cache.py` (the r64 golden is never regenerated; `gist_dtype` off = bit-identical).
Re-size the pods from the rates L2 MEASURED on the real-text pod (Llama 16K, A100 40 GB: `full` ≈ 0.6,
r64 gist ≈ 3.1, q4 ≈ 3.8, KIVI ≈ 1.5 min per sample, task-independent — prefill dominates): the plan's
1.0/3.0/3.5/1.5 min per task and its "Stage 1 ≈ 41 GPU-h" are wrong; write the measured table (per arm × family
× ctx: 96 retrieval samples + 16 windows; 32K ≈ 2×; frozen/random ≈ 1.5× faster than isvd, no-gist ≈ 2×) and
pre-commit the cuts in the prereg (drop `random` first, then n=16 for oja/fd) — expect Stage 1 (Llama + Qwen
× 16K × 7 arms incl. bf16) near 60 GPU-h base / 120 at 2×, i.e. ≈ $27–90 at the measured $0.45–0.74/h.

Gate 1 is decided on generator v2, where D-005 showed the r64 configuration retrieving 0.25 / 0.08 / 0.00 at
16K on real text (q4 0/0/0) while KIVI-2 holds 1.00 / 0.50 / 0.50 — so the retrieval contrasts may land at the
floor for every gist tracker and the decision may rest on `nogist` (tier-only, h ≈ 1024) beating the gist arms
and on perplexity. Write the prereg's predictions from that evidence; the branch rule is applied by
`gate1_verdict()` verbatim, never by a reading. The generator-v2 `vt` task (official semantics, 4 hops) has never
run on hardware and the v1 `vt` collapsed uncompressed under WikiText: before the Stage-1 launch commit, run a
≈ 3 GPU-h pre-flight pod (`full` + `isvd_r64_h256_seed` + the new `frozen` + `nogist` arms on `ruler_v2_16k`,
n=12 — pre-registered as `prereg/gate1_preflight.md`, reading = completeness + `full` ≥ 0.9 on every task) so a
broken task cannot burn the 50-hour budget; its rows also give the prereg its real-text baseline.

Hard rules (unchanged): forbidden words in new code/configs/docs/paper/commits — DLRA (as a word), BUG
integrator, honest/honestly, marquee, flagship. Pod launches inside approved lanes are authorized (D-011) but
Gate 1 needs D-003's top-up first (credit ≈ $92.5 vs a ≈ 120 GPU-h Stage-1 bar); every launch and every kill
gets a DECISIONS line with SHAs, offer, cost. A trial that raises is recorded as `error`, never dropped; any
primary-contrast arm with errors makes `gate1_verdict()` refuse. The r64 golden is never regenerated. Commit by
explicit path (iCloud creates `* 2.py` duplicates — delete them). Never `git stash`, even pathspec'd (iCloud
resurrected one mid-fix). Never TaskOutput on a local agent; SendMessage is unavailable — follow-ups go to a
fresh agent with the report path; a watchdog for a running pod holds `scripts/pod/watchdog.sh` open — edit it
only via temp file + `mv`. Only you append STATE/DECISIONS. The merge into `main` stops at the finishing menu
unless I say "merge L3 when green". A committed launch manifest keeps CI red until its harvest: sequence the
merge after the harvest.

Harness follow-ups to take when a task touches the file anyway (else an L6 micro-task): `frontier.run_ppl`
bills press arms AFTER the scored window (≈ 3 % over-bill flattering ours — must be fixed before any Phase-2
ppl table; Gate 1's arms are all `bug` kind so it does not bias Gate 1); harvest's non-shrink guard for every
record file; harvested manifests carry `gpu`/`cuda: none`, `model_revision: null`, `wall_clock_s: null` (the
pod's own values die with the instance — print them as `[stage]` lines like the digests); `bug_cache.py`'s
"the fix never fires at the defaults" comment is false at bf16 (repairs every 64-absorb window — say so where
the `gist_dtype` work touches the file); the `.raw` growth and the 90 s suite gate (test durations: turbo_press,
pplw split, quant roundtrip are the slow pre-existing ones).

Decisions that are mine — surface them at the first report, with your recommendation, and continue on
everything not blocked: D-003 top-up amount for Gate 1 Stage 1 (+ the ss2/smoke bars: 121 + 168 GPU-h);
D-001 arXiv v1/v2; D-002 accept ADR 0001; D-004 real Palu; D-006 KVQuant; D-007 move the clone out of iCloud
(four incidents in L2); D-010 delete the parked litter; D-017 OjaKV (their harness on its own pod at their
ratios, Phase 2); the ss2 prereg's branch-3 amendment (`ruler_v2_*`) before any ss2 launch.

Milestones (kickoff pack Part B, Weeks 2–3): frozen/random/no-gist arms + bf16 gist merged with tests; Gate-1
v2 prereg committed, pre-flight pod harvested, Stage 1 launched under D-003; `make tables gate1` renders
Holm/TOST from Stage 1; DECISIONS names the branch from `gate1_verdict()`; kernel single-layer correctness
(L4, separate lane). Report at each in ≤ 10 lines: merged / measured / blocked / decide.

Begin: create the L3 worktree, run the pre-flight audit (drift table + PR-L3-* rulings, the re-sized pods),
dispatch L3.1.
