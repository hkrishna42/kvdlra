# Next-session prompt — Gate 1 Stage 1 → verdict, and the kernel lane L4 (paste as the first message; cwd ~/src/kv-dlra, branch main)

PROVISIONAL — written 2026-09-20 19:40 EDT at the L3 finishing menu; the orchestrator updates the "State at handover" block
when the re-run harvests, the lane merges and Stage 1 launches. If a line below says "already done", skip it.

You are the orchestrator for kvdlra Weeks 3–5 (docs/plan/ICML2027_PLAN.md §2 Gate 1 and Gate 3), continuing from the last
entries of docs/plan/STATE.md. Read, in order, and do not summarize back: CLAUDE.md; docs/plan/STATE.md (the 2026-09-20
entries); docs/plan/DECISIONS.md (D-011 addenda 9–11, D-003 CLOSED, D-018, and the delegated closures of 2026-09-20);
docs/plan/cleanup/l3-ledger.md (PR-L3-1–30, R-L3-1–22 — the harness conventions you inherit, especially R-L3-6/7/8/12/15/16/17);
prereg/gate1_tracker_swap_v2.md (§1–§11, Amendment 1a, and Amendment 1b if it exists); prereg/gate1_preflight.md (Amendment 1
and its correction); prereg/bf16_gist.md; prereg/kernel_smoke.md; docs/adr/0001-factored-attention-kernel.md (ACCEPTED);
docs/plan/lanes/L4_kernel.md and GATES.md §G3/§G4/§G5; docs/plan/reports/vt-template-comparison.md. Then state in one line
each: what Stage 1 decides and by which function; what the kernel's Week-3 gate is; why the ss2/smoke pods still wait.

## Concurrency with the L3 session (READ FIRST — the owner starts this session while the L3 session is still running)
The L3 session (Claude session name `kv-dlra-5e`) is ALIVE and OWNS, until it reports them done in docs/plan/STATE.md and
`git log main`: the pre-flight re-run's harvest and records commit, Amendment 1b, the L6 gate on the lane, the `--no-ff` merge of
`lane/L3-gate1-tracker-swap-v2` into main, and the Stage-1 launch (two pods, with their DECISIONS lines on main). DO NOT do any
of those in this session — a second launch would bill a second pair of pods. Your work in parallel: lane L4-kernel from the
start, and G1-verdict's PREPARATION only (the five-reviewer panel prompt). Branch `lane/L4-kernel` off the pushed lane head
`origin/lane/L3-gate1-tracker-swap-v2` (NOT off main — main is behind the lane until the merge, and L4's first task edits
`bug_cache.py`, which the lane changed); after the merge lands on main, `git rebase main` in the L4 worktree (identical
commits, a clean rebase). Never run git commands in the main checkout `~/src/kv-dlra` itself while the L3 session may be
merging there; work only in your worktrees. The shared `.venv`'s editable install points at whichever tree ran
`uv pip install -e ".[dev]"` last — run it from your L4 worktree before its first `make test`, and expect the L3 session to
re-point it when it merges (re-run yours if `import kvdlra` resolves to the wrong tree: `python -c "import kvdlra; print(kvdlra.__file__)"`).
Resume the G1-verdict items (Amendment 1b, merge, Stage 1) ONLY if STATE says the L3 session ended without them or the owner
says so in chat — then read D-011 addenda 9–11 and results/gate1_preflight_rerun/ first.

## State at handover (written 2026-09-20 21:20 EDT while the pre-flight re-run was still running; every line has an "already done" branch — check `git log main` and `docs/plan/STATE.md` first)
- Lane L3+L5 sits on `lane/L3-gate1-tracker-swap-v2` @ f3006fc (pushed; 47 commits over main @ 278bd01; L6 gate PASS at 6065e4e,
  Task 8 — the bf16 retrieval reading in the gate1 table — landed after it). NOT merged. The owner said "merge L3 when green after
  the harvest": once results/gate1_preflight_rerun/ is harvested and `scripts/pod.py check results/gate1_preflight_rerun` is OK,
  and Amendment 1b is committed, re-run the L6 gate on the final lane SHA (scratchpad script pattern: fresh clone of the pushed
  branch, make env/test/tables/figures/check, tree clean, forbidden-word grep), then `git merge --no-ff` into main from the main
  checkout, verify (make test / tables / check / ruff), push main, delete the lane branch and worktree. If `git log main` already
  shows the merge commit, skip all of this.
- Pre-flight re-run `gate1_preflight_rerun` (instance 51815080 @ d4ed366, offer 50895886, bar 14 h, ≈ $4): launched 17:53 EDT
  2026-09-20 with a watchdog (pid 9161, `caffeinate -s -i`, Mac on AC) whose log lives in the L3 session's scratchpad. Expected
  ALL_DONE ≈ 02:00 EDT 2026-09-21; the watchdog harvests and destroys the instance. If results/gate1_preflight_rerun/trials.jsonl
  is missing: `.venv/bin/python scripts/pod.py harvest --pod gate1_preflight_rerun` (the pod replays its records before ALL_DONE, so
  the log tail is complete); if the instance still exists (`vastai show instances`), destroy it after the harvest. Commit the
  records (manifest, trials, diag, env) by explicit path. Signals seen live: isvd 0.83 / 0.25 / 0.25 / 0.25 / vt 0.42; nogist_h2423
  1.00 / 1.00 on single/multikey at matched stored bytes (the exact tier carries retrieval on real documents — the Week-12 mechanism).
- Amendment 1b to prereg/gate1_tracker_swap_v2.md: NOT written. Write it FIRST (append-only, dated): §2's measured baseline from
  results/gate1_preflight/ (the complete `full` arm) + results/gate1_preflight_rerun/ (isvd, nogist, frozen; pair by `prompt_sha256`
  across the two pods — reading (iii)); reading (iv) from `manifest.cell_elapsed_s` (isvd ≈ 3.3, frozen ≈ 1.6, nogist = measured,
  `full` 3.2 s per sample) against the triggers → §9 re-sized if a trigger fires, else confirmed; the vt exclusion 16 → 12 (primary)
  and 24 → 18 (secondary) per D-018 with the three quotable sentences of docs/plan/reports/vt-template-comparison.md (Verdict B);
  `EXCLUDED_TASKS = {"vt"}` in src/kvdlra/eval/gate1.py landing in the same commit with the pin test's family sizes 12/18/4; the
  A1a.9 attribution fix (the 0.0010 minimum is Qwen `isvd_r256_f0.01_qr64 − isvd_r256_f0.01_tol`); the digest-drop scope and the
  verdict signature are already in Amendment 1a. Task-review it as a scientist (the L3 pattern) before the launch commit.
- Stage 1 (`gate1_v2_stage1_llama`, `gate1_v2_stage1_qwen`; bar 82 h each): NOT launched. Launch from MAIN after the merge
  (`pod.py launch --pod <pod> --offer <id>` per pod, two A100 SXM4 40 GB instances at once, reliability ≥ 0.99, ≈ $0.60–0.70/h,
  avoid the California PCIe host family 5116338x that stalled in L2), watchdogs under `caffeinate -s -i` with the Mac ON AC, a
  DECISIONS D-011 addendum per pod (SHAs, offer, rate, bar, credit), STATE line. Expected ≈ 41 h wall each, $37–61 total; the owner
  said "start it; let me know if you are running low, I'll top up" — ask when credit < $35 (credit ≈ $84 on 2026-09-20 20:30).
  The launch commit must descend from Amendment 1b's commit.
- Deferred to L6 / Phase 2 (do not do before Stage 1 launches): the vt generator repair as NEW task files + a 1 GPU-h `full` vt
  validation cell (D-018 addendum); the deferred minors in docs/plan/cleanup/l3-ledger.md and the whole-branch review's triage
  table (recon's private-name reach-in, the L2_PODS table name, the docstring items, the oja tuning-number mismatch with D-014).

## Mode
superpowers subagent-driven-development exactly as L1–L3 ran it: fresh implementer per task, task review (spec + quality)
and a scoped re-review per fix round, a whole-branch review + ponytail-review + the clean-clone L6 gate before the finishing
menu; /ponytail full; the ledger is the recovery map. ONE implementer per worktree at a time (R-L3-17 — two collided through
pre-commit's stash); reviewers and read-only investigators may run in parallel with it. Two worktrees this session:
`.claude/worktrees/L4-kernel` (branch `lane/L4-kernel` off main) for the kernel, and `.claude/worktrees/G1-verdict`
(branch `lane/G1-verdict`) for the Stage-1 harvest/table/verdict work — they touch disjoint files (src/kvdlra/kernel vs
results/, prereg/, docs/plan), so their implementers may run in parallel. Symlink the shared `.venv` and `dumps/llama3.2-1b`
into each; `uv pip install -e ".[dev]"` from whichever worktree runs the suite last (the editable install points at one tree —
run `make test` from the tree whose src you are testing). Baseline `make test` = 603 passed ≈ 61 s (gate 90 s).

## Lane G1-verdict (Weeks 3–4)
1. While Stage 1 runs: nothing reads the pods. Prepare the five-reviewer simulation prompt (KICKOFF Part E note: the table
   is read by five simulated reviewers BEFORE anyone on the team reads it) as docs/plan/reviews/gate1-panel-prompt.md.
2. At ALL_DONE: the watchdogs harvest; commit results/gate1_v2_stage1_{llama,qwen}/ (manifest, trials, pplw, ppl, diag,
   env); `make check` must pass on both (any error row = the pod is REFUSED for that arm per §4 — do not "fix" records);
   `make gate1` renders docs/paper/tables/gate1.md with the bf16 block; commit the table.
3. Run the five-reviewer simulation on gate1.md alone (five parallel opus agents with the panel prompt; each returns a
   one-page reading); then and only then read the verdict line; append the Gate-1 outcome to DECISIONS.md with the evidence
   path and the verdict's reason verbatim (`gate1_verdict()` decides; never a reading); tick GATES §G3 lines 3–4 and §G5 line 1
   (the bf16 reading) with evidence. If the verdict is REFUSED or UNDECIDED, the pre-registered next step is in prereg §4/§10
   (a refused arm's cause named and repaired by amendment; UNDECIDED → Stage 2's Mistral pod if D-003's credit allows).
4. Stage 2 (conditional on not-C): create the Mistral 16K pod + the 32K pods and no-gist twins (`nogist_h4501`,
   `nogist_h8612` — H re-solved on stored bits at t = 32768) by a dated amendment BEFORE reading the 32K design against Stage 1's
   numbers (R-L3-7 fixes the families already); launch only under a fresh D-003 line.
5. The ss2 pods' branch-3 amendment (`ruler_v2_16k`/`32k`, the v2 pairing, budget re-derived) and the smoke pod wait behind
   the Gate-1 reading (D-003).

## Lane L4-kernel (Weeks 3–5, parallel)
1. ADR 0001 is ACCEPTED: (iii) Triton tile-wise reconstruct-inside-attention primary; (i) post-RoPE tracking as the parallel
   accuracy row `isvd_postrope_r128` (Llama 16K, n = 24, generator v2, its own prereg section). First Week-2 task: a batched
   `BugStreamingCache` (`lazy_initialization` raises above batch 1) — ADR concern (1).
2. Single-layer correctness (Week 3 milestone): max|Δ| < 1e-2 bf16 vs reconstruct-then-attend on the 1B dumps; full-model
   greedy token-exact on ≥ 14/16 prompts — the 16 prompts committed in L4's test module BEFORE the kernel (prereg/kernel_smoke.md
   §4); mismatches logged. Tests CPU-only where possible (Triton needs a GPU: the correctness cell runs on the kernel-smoke pod).
3. `prereg/kernel_smoke.md` cell list A (batch 1) is the pod; its numbers-to-beat are within-pod; `scripts/tables.py` gains
   the latency renderer L4 owes; the Week-3 gate: kernel KV peak < full at 32K AND ms/token < the reconstruct path by ≥ 3×.
4. GATES §G4 lines 1–3 are the definition of done.

## Hard rules (unchanged)
Forbidden words in new code/configs/docs/paper/commits — DLRA (as a word), BUG integrator, honest/honestly, marquee, flagship.
Pod launches inside approved lanes are authorized (D-011); every launch and every kill gets a DECISIONS line with SHAs, offer,
cost; the Mac stays on AC while a watchdog runs (the first pre-flight harvest was lost to battery sleep); a launched pod's
prereg is never edited in place — dated amendments only; a trial that raises is `error`, never dropped; the r64 golden is never
regenerated; commit by explicit path (`git add <paths>` then `git commit -o -- <paths>`; `-o` refuses untracked files);
never `git stash`; a failed command in an `&&` chain skips what follows — verify every append with grep; SendMessage is
unavailable — follow-ups go to a fresh agent with the report path; never TaskOutput on a local agent; `git -C <dir> diff A..B
-- paths` inside a redirect block once produced nothing — use `git diff A B -- paths`; only you append STATE/DECISIONS; the
merge into main stops at the finishing menu unless the owner says "merge <lane> when green".

## Decisions that are the owner's — surface at the first report with your recommendation
D-003 top-up timing (credit ≈ $82 before Stage 1; bar $74–121); Stage 2's launch (a fresh D-003 line); the Gate-1 outcome
(the rule decides, the owner reads the DECISIONS entry); the kernel's H100-or-max-batch contrast for cell list B (ADR
concern 2); the arXiv posting timing (D-001: after Gate 1, Phase 3).

## Milestones (KICKOFF Part B, Weeks 3–5): Stage 1 harvested and `make gate1` rendered; the five-reviewer simulation run;
DECISIONS names the branch from `gate1_verdict()`; kernel single-layer correctness on the 1B dumps; the kernel-smoke pod
(cell list A) launched under prereg/kernel_smoke.md. Report at each in ≤ 10 lines: merged / measured / blocked / decide.

Begin: read the Concurrency block and the State-at-handover block; create the L4 worktree off the pushed lane head and the
G1-verdict worktree for the panel prompt only; dispatch L4's first task (the batched cache, TDD on the tiny model) and the panel-prompt
task in parallel (one implementer per worktree). Check `git log main` and STATE at each report for the L3 session's merge and launch.
