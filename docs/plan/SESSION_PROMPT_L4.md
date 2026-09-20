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

## State at handover (the orchestrator fills this in at the end of the L3 session)
- Lane L3+L5: merged into main at <SHA> / NOT yet merged (then: merge `lane/L3-gate1-tracker-swap-v2` `--no-ff` after
  `make check` passes on results/gate1_preflight_rerun; the owner said "merge L3 when green after the harvest").
- Pre-flight re-run `gate1_preflight_rerun` (instance 51815080 @ d4ed366): harvested at <SHA> with readings (i)/(iv) <pass/fail>
  / still running (watchdog pid 9161; log in the previous session's scratchpad — if lost, `pod.py harvest --pod
  gate1_preflight_rerun` fetches `vastai logs`; the pod self-destructs 2 h after ALL_DONE).
- Amendment 1b to prereg/gate1_tracker_swap_v2.md: committed at <SHA> / NOT yet (then write it FIRST: the pre-flight rows as
  §2's baseline; the vt exclusion 16 → 12 per D-018 with the template comparison's three sentences; the §9 re-size from
  `cell_elapsed_s` — isvd 3.3, frozen 1.6, nogist <measured> min/sample, `full` 3.2 s; the A1a.9 attribution fix (0.0010 is
  Qwen f0.01_qr64 − f0.01_tol); `EXCLUDED_TASKS = {"vt"}` lands with it, the pin test's family sizes 12/18/4).
- Stage 1 (`gate1_v2_stage1_llama`, `_qwen`; bar 82 h each): LAUNCHED at <SHA>, instances <ids>, offers <ids> / NOT yet (then
  launch under D-003 + D-011 with the Mac ON AC POWER, two instances at once, watchdogs under `caffeinate -s -i`, a
  DECISIONS line per pod; expected ≈ 41 h wall each, $37–61 total; ask the owner for a top-up when credit < $35).

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

Begin: read the State-at-handover block, create the two worktrees, dispatch G1-verdict's first task (Amendment 1b if it is
not committed; else the panel prompt) and L4's first task (the batched cache) in parallel.
