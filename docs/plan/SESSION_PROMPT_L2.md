# Next-session prompt — L2 (paste as the first message; cwd ~/Desktop/kv-dlra, branch main)

You are the orchestrator for kvdlra Weeks 0–3 (docs/plan/ICML2027_PLAN.md §1–§2), continuing from the last
entries of docs/plan/STATE.md (2026-09-19: L0, L1, L7, L4-ADR and L2prep are merged to `main`; `main` is the
only branch). Read, in order, and do not summarize back: CLAUDE.md; docs/plan/STATE.md (the last five entries);
docs/plan/DECISIONS.md (D-011 through D-016, and the OPEN items D-001–D-005, D-007, D-010);
docs/plan/plans/2026-09-11-L2-generator-v2-and-baselines.md (APPROVED 2026-09-17);
docs/plan/lanes/L2_generator_v2_and_baselines.md and GATES.md §G2; prereg/filler_realism.md;
docs/plan/cleanup/l1-ledger.md (skim the rulings — they are the harness conventions L2 inherits).
Then state in one line each: what G2 needs, what D-005 costs, and why L3 waits on L2.

Mode: superpowers subagent-driven-development exactly as L1 ran it — fresh implementer per task, a task review
(spec + quality) and a scoped re-review per task, a whole-branch review + ponytail-review + the clean-clone L6
gate before the finishing menu; one git worktree for the lane; /ponytail full; the ledger is the recovery map.

Workspace: `git worktree add .claude/worktrees/L2-generator-v2 -b lane/L2-generator-v2 main`; symlink the shared
`.venv` and `dumps/llama3.2-1b` into it; `uv pip install -e ".[dev]"` from the worktree and `chflags nohidden`
on the venv's `.pth` files (iCloud re-hides them); baseline `make test` = 469 passed.

Pre-flight before Task 1 (the L2 plan was written before L0 and L1 landed): audit every path, symbol, YAML
shape and test the plan names against `main` — `TaskCfg` (corpus, window, n_samples; `load_pod` refuses two
corpora at one ctx), `ArmCfg.cache` knobs (tracker `isvd`, guard knobs, `diag_every`), `arm_kwargs`,
`records.py` (`[trial]`/`[pplw]`/`ppl=`/`[diag]` contracts with `corpus`, arm/ctx/task/idx), `_footprint`
(live-rank billing), the parity allowlist `POST_V1` in tests/test_config_parity.py, `scripts/tables.py`
subcommands, `pod.py launch/harvest/check` (env.txt from the ENV block; the non-shrink guard is trials-only).
Write the drift table and its rulings to the ledger as PR-* entries, then dispatch.

L2 Task 1 first: the filler-realism diagnostic. Re-express prereg/filler_realism.md's design as a pod YAML
under the L0 launch path (configs/arms + configs/tasks + configs/pods, `pod.py launch`), amend the prereg
(append-only, dated) to that path with the measured-rate budget lesson from D-011 (≈ 5 min per 16K perplexity
sample on r256 arms; 2 min at r128), launch under D-005/D-011's standing authorization, and record the launch
in DECISIONS. Watchdog: run it under `caffeinate -s`; if a pod budget self-destruct (`--max-hours` in
`pod.py launch`, `timeout` in boot.sh) is cheap, add it in Task 1 — ≈ $12 of L1's spend was idle billing while
the launch machine slept. Then Tasks 2–8 per the plan (generator v2, faithful KIVI, svd_oracle docs, the
kvpress presses, the ss2 pods, the OjaKV adapter, the smoke pod).

Hard rules (unchanged): forbidden words in new code/configs/docs/paper/commits — DLRA (as a word), BUG
integrator, honest/honestly, marquee, flagship. Pod launches are authorized for approved lanes (D-011) but every
launch and every kill gets a DECISIONS line with SHAs, offer, cost. A trial that raises is recorded as `error`,
never dropped. The r64 golden is never regenerated. Commit by explicit path (iCloud creates `* 2.py`
duplicates — delete them). Never bare `git stash`. Never TaskOutput on a local agent; SendMessage is
unavailable — follow-ups go to a fresh agent with the report path. Only you append STATE/DECISIONS. The merge
into `main` stops at the finishing menu unless I say "merge L2 when green".

Harness follow-ups to take when a task touches the file anyway (else an L6 micro-task): harvest's non-shrink
guard for every record file; the watchdog's BUDGET_ITERS expiry must destroy the instance; `[stage]` prints
for model/corpus load in the runner; paper/main.tex still carries the v1 vocabulary (Phase-3 rewrite).

Decisions that are mine — surface them at the first report, with your recommendation, and continue on
everything not blocked: D-001 arXiv v1/v2; D-002 accept ADR 0001 (merged, OPEN); D-003 GPU split / top-up
(credit ≈ $97.75); D-004 real Palu vs svd_oracle; D-005 filler-diagnostic design and cost (now launchable);
D-007 move the clone out of iCloud; D-010 delete the parked litter.

Milestones (kickoff pack Part B, Week 2): generator v2 + faithful KIVI merged with tests; Gate-1 v2 pod
launched (L3, after L1 + L2); ss2 on Mistral/Qwen 16K + 32K; kernel single-layer correctness (L4). Report at
each in ≤ 10 lines: merged / measured / blocked / decide.

Begin: create the L2 worktree, run the pre-flight audit, dispatch Task 1.
