# Next-session prompt — Gate 1 Stage 1 → verdict, the kernel re-run under Amendment 2 (lane L4b), the L3 items if released, and the two merges (paste as the first message; cwd ~/src/kv-dlra, branch main)

Written 2026-09-21 10:40 EDT at the L4 finishing menu (lane/L4-kernel @ 503c754). Every line of "State at handover" has an
"already done" branch — check `git log --oneline -3 main`, `git log --oneline -1 origin/lane/{L3-gate1-tracker-swap-v2,L4-kernel,G1-verdict}`
and the last entries of docs/plan/STATE.md before believing any of it. This file supersedes docs/plan/SESSION_PROMPT_L4.md
(consumed by the L4 session and removed in the same commit; its L3-items block is carried verbatim in Appendix A).

You are the orchestrator for kvdlra Weeks 3–5 (docs/plan/ICML2027_PLAN.md §2 Gate 1 and Gate 3; the plan's schedule §4.3
puts today in Week 2–3, deliverables due Oct 6: "Gate 1 tracker swap running; faithful KIVI + generator v2 done; bf16-gist
result; kernel correctness"), continuing from the last entries of docs/plan/STATE.md. Read, in order, and do not summarize
back: CLAUDE.md; docs/plan/STATE.md (the 2026-09-20 and 2026-09-21 entries); docs/plan/DECISIONS.md (D-011 addenda 9–12,
D-003 CLOSED, D-018 and its addendum, the delegated closures of 2026-09-20, and the D-002 addendum "Gate-3 WEEK-3 OUTCOME");
docs/plan/cleanup/l3-ledger.md (R-L3-6/7/8/12/15/16/17) and docs/plan/cleanup/l4-ledger.md (R-L4-0..35 — especially
R-L4-17/21/26 the correctness bars, R-L4-20 the cross-lane ruling, R-L4-27/29/30 the pre_run hook, R-L4-33/34 why nothing on the
launched pod's rule moved, R-L4-35 the parked items that become Task 11 lines); prereg/kernel_smoke.md (§1–§11, Amendment 1 and
its three corrections); docs/plan/reports/kernel-smoke-amendment2-draft.md (all four parts — it is the pre-specification of the
re-run, committed at 503c754 before any calibration ran); docs/paper/tables/kernel_smoke.md; prereg/postrope_r128.md
(unlaunched); prereg/gate1_tracker_swap_v2.md (§1–§11, Amendment 1a, and 1b if it exists); prereg/gate1_preflight.md
(Amendment 1 + correction); docs/plan/reviews/gate1-panel-prompt.md; docs/plan/plans/2026-09-20-G1-verdict.md (T1 done,
T2–T5 gated); docs/adr/0001-factored-attention-kernel.md §5; docs/plan/lanes/GATES.md §G3/§G4/§G5/§G6;
docs/plan/reports/vt-template-comparison.md. Then state in one line each: what Stage 1 decides and by which function; why
kernel_smoke's Week-3 gate reads REFUSED although both systems bars were met on measurement; what Amendment 2 changes and what
it does not; why the ss2/smoke pods still wait.

## 0. Begin here — the owner's decisions (present this block with your recommendations at the first message; launch nothing before the answers)
(a) **The L3 items.** The L3 session (Claude session `kv-dlra-5e`, id local_334d7519-6763-405a-bd37-7e2db8f1eadb) stalled at
    02:24 EDT 2026-09-21 on its Monitor's ALL_DONE event (a stopped desktop session does not wake on Monitor events or on
    send_message; a status message is queued in it). Its watchdog harvested results/gate1_preflight_rerun/ to disk in
    `.claude/worktrees/L3-gate1` — UNCOMMITTED (manifest modified; diag/env/trials untracked). NOT done: the harvest commit,
    Amendment 1b, the L6 gate on the final lane SHA, the `--no-ff` merge of `lane/L3-gate1-tracker-swap-v2` into main, the
    Stage-1 launch. Options: (i) the owner reopens that session and it finishes its items — then this session never touches them;
    (ii) the owner says "take over the L3 items" — then Appendix A is your spec, the L3 worktree is yours, and Stage 1 launches
    from main after the merge. Recommendation: (ii) — the items are fully specified, the harvest is on disk, and every day
    without Stage 1 running slips the Oct 6 deliverable (Stage 1 is ≈ 41 h wall per pod).
(b) **The kernel re-run (D-002 addendum of 2026-09-21; the draft's §4).** (A) commit Amendment 2 + Task 11 + `kernel_smoke2`,
    re-run cell list A unchanged — ≈ $1.0–1.7 expected, ≈ $2.3–3.7 at the 5.0 h bar (kernel_smoke itself cost $1.27 / 1.19 h);
    (B) record REFUSED as the Week-3 outcome, defer to Phase 2 — $0 now, the systems paragraph stays unwritten and every later
    kernel pod inherits an unreadable bar; (C) amend and re-read instance 51903816 — impossible (the records store neither
    max|ref| nor any logit) and rejected on the record. What (A) asks the owner to approve, precisely: the constant 2⁻⁶ (4 bf16
    ulps of |ref|max) in A2.2; the near-tie margin M = 2⁻⁵ × the step's top reconstruct logit in A2.3 together with its κ > 8
    revision branch; the five record fields of A2.5; the label `kernel_smoke2`; ≈ $2. Recommendation: (A).
(c) **Cell list B's card (ADR 0001 concern 2; the Week-5 Gate-3 decision needs batch ≥ 4 at 64K).** H100 (a second GPU type,
    a new host family) or max-batch on the same A100-40 GB class ("fit batch 8 where full KV fits batch 2" is the plan's own
    fallback, §2 Gate 3.2). Recommendation: max-batch on the A100 class first, by a dated amendment to prereg/kernel_smoke.md,
    after the re-run's precondition reads; H100 deferred to Phase 2 with the "kernel tested on two GPU types" limitation line.
(d) **Credit (D-003 CLOSED: "let me know if you are running low, I'll top up").** Credit $79.48 at 10:24 EDT 2026-09-21. Stage 1
    expected $37–61 (≈ 41 h × 2 pods at $0.60–0.74/h), $74–121 at the 82 h bars; kernel_smoke2 ≈ $2 (≤ $4); the postrope row
    ≈ $10–20 (bar 24 h); cell list B later. vast.ai stops an instance when the balance reaches zero, and a stopped Stage-1 pod
    loses its unfinished cells. Recommendation: ask for ≈ $100 at Stage 1's launch (the L3 session's own recommendation), not at
    the $35 line.
(e) **The post-RoPE accuracy row** (`isvd_postrope_r128`, prereg/postrope_r128.md + configs/pods/postrope_r128.yaml, Llama 16K
    n = 24 + 16-window perplexity, bar 24 h; the ADR's option (i) test: does r = 128 post-RoPE match r = 64 pre-RoPE at equal
    bytes?). Inside the approved lane (D-011) but competes with Stage 1 for the same credit. Recommendation: launch after the
    top-up, on the same A100 class, in the same week as kernel_smoke2.
(f) **arXiv timing** — D-001: after Gate 1 reads, Phase 3. No action this session.
The merge order is not a decision: `lane/L3-gate1-tracker-swap-v2` first, then `lane/L4-kernel` (which descends from the L3
lane head), each only on the owner's "merge <lane> when green".

## 1. Concurrency (READ before any git command)
- Never run git commands in the main checkout `~/src/kv-dlra` while the L3 session may be alive there (its merge target). Work
  only in your worktrees: `.claude/worktrees/L4-kernel` (branch `lane/L4-kernel`, its own `.venv` via `make env`,
  `dumps/llama3.2-1b` symlinked) and `.claude/worktrees/G1-verdict` (branch `lane/G1-verdict`). `.claude/worktrees/L3-gate1`
  is the L3 session's until the owner releases it under 0(a)(ii) — then confirm that session is closed (ask the owner; do not
  message it) before touching the worktree.
- Every worktree runs its own `make env` and the suite from its own `.venv`; before the first `make test` confirm
  `.venv/bin/python -c "import kvdlra; print(kvdlra.__file__)"` prints that worktree's `src/kvdlra/__init__.py`.
- No pods are live (`vastai show instances` is empty at 10:24 EDT); no watchdog, Monitor or agent is running. A pod launched by
  this session is this session's to harvest: watchdog under `caffeinate -s -i` (holds only on AC), and poll its log in bounded
  waits — the harness's Monitor events do not wake a stopped desktop session (l4-ledger R-L4-20; memory gotcha 11).

## 2. State at handover (10:40 EDT 2026-09-21)
- main @ 278bd01 — unchanged since 2026-09-19; NOTHING of L3/L4 is merged.
- `lane/L3-gate1-tracker-swap-v2` @ 91d149d (pushed; 47 commits over main; L6 gate PASS at 6065e4e "conditional on the re-run's
  harvest"). results/gate1_preflight_rerun/ harvested to disk in the L3 worktree, UNCOMMITTED. results/gate1_preflight is the
  PARTIAL harvest (97/240 — the Mac slept on battery, D-011 addendum 10): `scripts/pod.py check results/gate1_preflight` reports
  `nogist_h2423 v2/vt ctx=16384 has 0 of 12 records` and stays that way; Amendment 1b reads the two pods together (reading (iii),
  paired by `prompt_sha256`; vt EXCLUDED per D-018). Amendment 1b NOT written; Stage 1 NOT launched.
- `lane/L4-kernel` @ 503c754 (pushed; 30 commits over 91d149d): Tasks 1–10 + the final fix wave; L6 clean-clone gate PASS at
  abc8af8 (680 passed / 8 gpu-skipped, 105 s cold — 76–79 s warm against the 90 s gate: ~12 % headroom, trim before adding
  Task 11's ≤ 25 s calibration test); `make tables` diff-clean; `make check` OK on every live manifest incl. results/kernel_smoke
  after the harvest (935f745). At the finishing menu; the `.superpowers/sdd/2026-09-20-L4-kernel/` workspace kept until the merge
  (its ledger is docs/plan/cleanup/l4-ledger.md).
- kernel_smoke (instance 51903816, launch SHA a691d44, $1.27, 1.19 h; D-011 addendum 12; D-002 addendum): pre_run gate PASSED on
  the A100 (Triton vs reference ≤ 2e-3 ∧ rms ≤ 1e-4 at n_splits == 1; 7 passed, 1 skipped); the 8B bf16 precondition NOT met —
  12/16 token-exact (bar ≥ 14/16; prompts 3/6/8/14 at steps 13/15/4/8) and worst per-layer max|Δ| 1.464e-2 (prompt 7, layer 26;
  bar < 1e-2; prompts 2/10/12 also over) → the Week-3 gate reads REFUSED under prereg §4. Measured, NOT cited: kv_peak kernel
  0.83 / 1.08 / 1.61 GB vs full 2.05 / 4.08 / 8.14 vs reconstruct 3.25 / 6.45 / 12.84 at 16K / 32K / 64K; ms/token p50 kernel
  39.2 / 56.1 / 88.0 vs full 24.7 / 30.1 / 42.2 vs reconstruct 126.0 / 248.3 / 605 — the gate's two conditions (kv_peak
  1.08 < 4.08 at 32K; 248.3 / 56.1 = 4.4× ≥ 3×) hold on measurement; at 64K the kernel is 2.1× slower than full KV (the Week-5
  throughput bar) and its resident KV is 0.26× of full's peak at 32K (the Week-5 memory bar is ≤ 0.25×). The two failing
  clauses are disjoint (every over-bar prompt token-exact, every diverging prompt under the bar) — the reading recorded in the
  D-002 addendum and made testable by Amendment 2's falsifiers. Kernel-arm spikes 5 / 4 / 4 (maxima 1,992 / 233 / 263 ms),
  below §4's > 8 refusal, cause unresolved (A2.4's log-only companion line decides first-touch JIT vs the 16-step absorb lattice).
- `lane/G1-verdict` @ 84c09ff (pushed, off 91d149d): T1 the five-reviewer panel prompt committed; T2–T5 gated on Stage 1.
- Unlaunched, committed preregs: prereg/postrope_r128.md (decision 0(e)); prereg/ss2_families.md (3 pods) and prereg/l2_smoke.md
  (wait behind the Gate-1 reading, D-003); prereg/bf16_gist.md (its accuracy reading rides Stage 1's bf16 block, GATES §G5 line 1).
- Memory: ~/.claude/projects/-Users-hari-src-kv-dlra/memory/kvdlra-l4-standing.md and kvdlra-harness-gotchas.md items (5)–(12)
  (the `-m`-before-`--` commit form; the base64 pre_run hand-off; vast.ai `success: False` launches; the Monitor stall).

## 3. Mode and the required skills
superpowers subagent-driven-development exactly as L1–L4 ran it: fresh implementer per task (`scripts/task-brief PLAN N` →
brief; the dispatch carries the brief path, interfaces, rulings, the report path), a task review (spec + quality) from
`scripts/review-package PLAN BASE HEAD`, a scoped re-review per fix round, the ledger in `scripts/sdd-workspace PLAN` as the
recovery map (`Ruling: … — why — cost if wrong`), a whole-branch review (fable) + `/ponytail-review` + the clean-clone L6 gate
(Appendix B) before the finishing menu; superpowers:finishing-a-development-branch stops at the menu. `/ponytail full` for all
src/ work; test-driven-development on every src/ change; systematic-debugging for any number that disagrees;
verification-before-completion before any STATE "done" (the verifier signature: commands + output paths);
using-git-worktrees (one lane per worktree — the three above exist; create none unless a new lane opens);
writing-plans for the L4b plan (short, self-contained: docs/plan/plans/2026-09-21-L4b-kernel-rerun.md, Tasks 11–13 below);
dispatching-parallel-agents for reviewers and read-only investigators only. ONE implementer per worktree at a time (R-L3-17).
Models: opus for implementers and reviewers touching prereg text, statistics or more than two files; sonnet for mechanical
fix rounds; fable for the plan scan and the whole-branch review. Harness: `make env` / `make test` (the suite; 90 s warm gate)
/ `make tables` (diff-clean) / `make figures` / `make check` / `make kernel_smoke` / `make gate1`; `scripts/pod.py launch|harvest|check`;
`scripts/pod/watchdog.sh`; `vastai show user --raw | grep credit`; `vastai show instances`. SendMessage is unavailable —
follow-ups go to a fresh agent with the report path; never TaskOutput on a local agent.

## 4. Lane L4b — Amendment 2 + Task 11 + `kernel_smoke2` (only under 0(b)(A); on `lane/L4-kernel`, new plan + new SDD workspace)
Order (the draft's A2.7 order is superseded for one reason, to be ledgered as a ruling: A2.3's κ calibration is a CPU test
that can revise M, and a launched pod's prereg takes appended text only — so the calibration runs first and Amendment 2 is
committed once, with ρ̂, L_tiny and κ printed in it; the pre-specification is the draft at 503c754, which precedes both):
1. **Task 11** (opus; the draft's §3 table, nine items, ≈ 175 CPU-tested lines): `kernel_compare[layer]` stores (max|Δ|, max|ref|);
   `_greedy` returns (argmax, top1, top2) per step with the reconstruct twin run FIRST; `check_prompt` computes `rel_max_diff` /
   `rel_worst_layer` / `ref_max` / `gap_at_mismatch` / `kernel_logit_for_ref_argmax`, `refs=` on the layers line; the five
   fields in `failed_row` and `format_line` appended BEFORE `error=` (archived rows still parse — a round-trip test); five
   optional groups on `KERNEL_CHECK_RE` + `KernelCheckRecord` + `parse_kernel_check_lines`; `scripts/tables.py` `REL_MAX = 2**-6`,
   `M_COEF = 2**-5`, `precondition_line` reads both bars, splits mismatches into attributable / near-tie, refuses `None` fields as
   "not computable (record predates Amendment 2)"; the `[latency spikes ctx=… arm=… steps=…]` companion line from `times_ms`;
   the bf16 tiny-model calibration test (16 prompts × 16 steps, κ = ρ̂·√(32/L_tiny) rounded up to a power of two, asserting
   κ ≤ 8 and printing ρ̂ / L_tiny / κ). Plus R-L4-35's lines: `del cache` before the reconstruct twin; `OSError` in the postrope
   snippet's except tuple. First measure `make test --durations=15` and trim so the warm suite stays under 85 s with the
   calibration in; never shrink the calibration's 16 × 16. Not in Task 11: `do_not_specialize`, any warm-up change, the
   post-basis early-return move (parked).
2. **Task 12** (opus; task-reviewed "as a scientist", the L3 pattern): append Amendment 2 to prereg/kernel_smoke.md verbatim from
   the draft's §2 with κ's measured branch filled in; `configs/pods/kernel_smoke2.yaml` byte-for-byte from `kernel_smoke.yaml`
   with `name: kernel_smoke2` (`prereg: prereg/kernel_smoke.md` kept; the `doc:` of kernel_smoke.yaml is hashed by the harvested
   manifest — never edit that file); a `make kernel_smoke2` target (the renderer takes `--pods results/<pod>`). The launch entry
   must name Amendment 2's SHA and paste `git merge-base --is-ancestor <amendment SHA> <launch SHA>` — `pod.py launch` checks
   only the prereg's FIRST commit (71c9be6).
3. **Task 13 — launch, harvest, outcome**: L6 gate on the launch SHA first (Appendix B); `pod.py launch --pod kernel_smoke2
   --offer <id>` on an A100 SXM4 40 GB, reliability ≥ 0.99, ≈ $0.8/h (avoid the California PCIe host family 5116338x; if
   `vastai create` answers `success: False` / intended state `stopped`, destroy within minutes and relaunch on another offer —
   D-011 addendum 12 attempt 1); watchdog under caffeinate, Mac on AC; DECISIONS D-011 addendum 13 (SHAs: prereg first commit
   71c9be6, Amendment 2, launch; offer, rate, bar 5.0 h, credit) and a STATE line. At ALL_DONE: `scripts/pod.py check
   results/kernel_smoke2` (the 16 kernel_check rows, the three pre_run refusals); commit the records by explicit path; `make
   kernel_smoke2`; commit the table; the D-002 addendum "Gate-3 WEEK-3 OUTCOME, re-run" — the rule decides (rel ≤ 2⁻⁶ ∧
   attributable-count ≥ 14 ∧ no errors, then the two gate conditions at batch 1) — and GATES §G4 lines 2–3 ticked or annotated
   with evidence; the spike companion line's reading (JIT vs lattice) recorded as a performance observation, no fix on this pod.
4. Then the postrope row under 0(e) (its own D-011 line, watchdog, harvest, `make tables` row, D-002 addendum: does r = 128
   post-RoPE match r = 64 pre-RoPE at equal bytes on retrieval and perplexity — prereg §11's refusals apply).
5. Whole-branch review (fable) + `/ponytail-review` + L6 → the finishing menu. Instance 51903816's records are never re-read,
   revised or deleted.

## 5. Lane G1-verdict (Weeks 3–4; `.claude/worktrees/G1-verdict`; docs/plan/plans/2026-09-20-G1-verdict.md T2–T5)
2. At Stage 1's ALL_DONE: the watchdogs harvest; commit results/gate1_v2_stage1_{llama,qwen}/ (manifest, trials, pplw, ppl, diag,
   env); `make check` must pass on both (any error row = the pod is REFUSED for that arm per §4 — do not "fix" records);
   `make gate1 > /dev/null` renders docs/paper/tables/gate1.md with the bf16 block; commit the table UNREAD.
3. Run the five-reviewer simulation on gate1.md alone (five parallel opus agents with docs/plan/reviews/gate1-panel-prompt.md,
   the preamble verbatim incl. the bf16-block sentence; outputs docs/plan/reviews/<date>-gate1/R<k>-<slug>.md + meta.md); then
   and only then read the verdict line; append the Gate-1 outcome to DECISIONS.md with the evidence path and the verdict's
   reason verbatim (`gate1_verdict()` decides; never a reading); tick GATES §G3 lines 3–4 and §G5 line 1 (the bf16 reading)
   with evidence. If REFUSED or UNDECIDED, the pre-registered next step is in prereg §4/§10 (a refused arm's cause named and
   repaired by amendment; UNDECIDED → Stage 2's Mistral pod if D-003's credit allows).
4. Stage 2 (conditional on not-C): the Mistral 16K pod + the 32K pods and no-gist twins (`nogist_h4501`, `nogist_h8612` — H
   re-solved on stored bits at t = 32768) by a dated amendment BEFORE reading the 32K design against Stage 1's numbers (R-L3-7
   fixes the families already); launch only under a fresh D-003 line.
5. The ss2 pods' branch-3 amendment (`ruler_v2_16k`/`32k`, the v2 pairing, budget re-derived) and the smoke pod wait behind the
   Gate-1 reading (D-003).

## 6. Merges (each only on the owner's "merge <lane> when green")
L3 first: the L6 gate on the final lane SHA (harvest commit + Amendment 1b included), `git merge --no-ff` into main from the
main checkout, verify (make test / tables / check / ruff), push main, delete the lane branch and worktree; Stage 1 launches
from main after it. L4 second: `git merge --no-ff lane/L4-kernel` — docs/plan/STATE.md and DECISIONS.md conflict at EOF (both
lanes append); keep both sides in date order, nothing dropped; then the same verification. Do not rebase L4 onto main
(identical trees either way; the merge keeps the lane's history and its L6 evidence SHAs valid).

## 7. Hard rules (unchanged, two forms corrected)
Forbidden words in new code/configs/docs/paper/commits — DLRA (as a word), BUG integrator, honest/honestly, marquee, flagship.
Pod launches inside approved lanes are authorized (D-011); every launch and every kill gets a DECISIONS line with SHAs, offer,
cost; the Mac stays on AC while a watchdog runs (the first pre-flight harvest was lost to battery sleep); a launched pod's
prereg is never edited in place — dated amendments only (an UNLAUNCHED prereg may be revised in place: D-011 addendum 9,
prereg/postrope_r128.md); a trial that raises is `error`, never dropped; the r64 golden is never regenerated; commit by explicit
path — `git add <paths>` then `git commit -o -m "<msg>" -- <paths>` (the `-m` MUST precede `--`: everything after `--` is a
pathspec, and `git commit -o -- <paths> -m` commits nothing); never `git stash`; a failed command in an `&&` chain skips what
follows — verify every append with grep; `task-brief` prints `wrote <path>: N lines`, use the path; ledger timestamps from
`$(date)`, never by hand; a pod's shell command goes to boot.sh base64-encoded (`PRE_RUN_B64`; vast.ai's `--env` parser splits
a quoted command); only you append STATE/DECISIONS; the merge into main stops at the finishing menu unless the owner says
"merge <lane> when green".

## 8. ICML plan (docs/plan/ICML2027_PLAN.md) — state vs tasks, 2026-09-21
| Plan item | Due (§4.3) | State | Next / blocker |
|---|---|---|---|
| Phase 0 hygiene, prereg discipline, kernel ADR, 1B dumps | Sep 22 | DONE — merged 2026-09-19 (L0/L1/L7/L4-ADR/L2prep, D-007 278bd01); eight preregs committed before their pods (§G5) | — |
| §2 Gate 1.1 tracker swap (6 trackers × 3 families × 2 ctx × 4 tasks, n = 24 + perplexity) | Oct 6 running, Oct 20 read | Machinery DONE on the L3 lane (frozen/random/no-gist twins, `gate1_verdict()`, `make gate1`, prereg v2 + 1a, two pre-flight pods: `full` ceiling 1.00/1.00/0.92/0.92/vt 0.75 → vt descriptive, D-018). Stage 1 (Llama + Qwen 16K, bar 82 h each) NOT launched; Stage 2 conditional on not-C | 0(a): Amendment 1b → L3 merge → Stage 1 → panel → DECISIONS names the branch (§5). Credit 0(d) |
| §2 Gate 1.2 §4.1 rank-sweep figure (5 ranks, all layers, K/V, 3 families, stored metric) | Oct 6 | 1B half DONE (§G1 line 6 [~]); the 8B dumps NOT collected (no dump pod ran) | a dump pod (≈ 10 GPU-h) after Stage 1 launches; Figure 2 in any branch |
| §2 Gate 2.1 generator v2 | Oct 6 | DONE (L2 merged e116212; cycled filler RETIRED, D-005) | — |
| §2 Gate 2.2 faithful KIVI + matched-bytes run (McNemar, Holm, family 12) | Oct 20 | Arm DONE (`kivi2_faithful`, G = 32, R = 128, full-precision prefill); KVQuant deferred (D-006); the ss2 pods (3 families) pre-registered, NOT launched | behind the Gate-1 reading (G1-verdict T5); a fresh D-003 line |
| §2 Gate 2.3 baseline pass | Oct 20 | `svd_oracle` rename DONE; SnapKV/PyramidKV/EA/ThinK+SnapKV arms built, smoke pod pre-registered NOT launched; real Palu → Phase 2 (D-004); OjaKV via the authors' harness in Phase 2 (D-017) | the smoke pod behind Gate 1 (T5) |
| §2 Gate 3.1 kernel ADR | Sep 22 | DONE — ADR 0001 ACCEPTED, option (iii) primary, (i) as the accuracy row (D-002) | — |
| §2 Gate 3.2 Week 2 single-layer correctness (max|Δ| < 1e-2 bf16) | Oct 6 | CPU halves MET (4.7e-3 / 5.2e-3 random 8B shapes, 2.75e-3 on the 1B layer; tiny model 16/16 fp32); Triton vs reference on the A100 MET (≤ 2e-3 / rms 1e-4); the 8B bf16 halves NOT met under the absolute bar (12/16; 1.464e-2) → REFUSED | 0(b): Amendment 2 + Task 11 + `kernel_smoke2` (§4) |
| §2 Gate 3.2 Week 3 full-model decode path (tier + ring + sinks as dense tiles) | Oct 6 | DONE — `decode_attention: kernel` via transformers' AttentionInterface, dense tokens as tiles, `kernel_check` axis, batched cache | — |
| §2 Gate 3.2 Week 4 measured peak VRAM + ms/token (16K–128K, b = 1 and 8, A100 + H100, KIVI kernels) | Oct 20 | b = 1 at 16K/32K/64K MEASURED (cell list A; not cited — §2 above); b = 4/8, 128K, H100, KIVI-with-kernels NOT run | cell list B by amendment — 0(c) |
| §2 Gate 3.2 Week 5 Gate-3 decision (resident ≤ 0.25× full at 32K; tokens/s ≥ full at 64K, b ≥ 4) | Oct 20 | Measured at b = 1: resident 0.26× of full's peak at 32K (kv_peak 1.08 vs 4.08 GB; analytic stored 0.556 GiB); 64K decode 2.1× SLOWER than full (88 vs 42 ms) — the plan's "memory axis alone → max-batch" fallback is the live path | after the re-run reads and cell list B runs |
| §2 Gate 3.3 bf16 gist storage | Oct 6 | `gist_dtype: bf16` BUILT (L5.1; billed 16 bits; drift 1e-2 … 5e-2 at 40 … 1000 absorbs, the guard repairs); prereg/bf16_gist.md committed; the accuracy/perplexity reading rides Stage 1's bf16 block (§G5 line 1) | reads with Gate 1 |
| ADR option (i): post-RoPE r = 128 accuracy row | Week 3–5 | prereg/postrope_r128.md + pod YAML committed, NOT launched | 0(e) |
| Gate review memo (G1 / G2 / G3 → branch A / A′ / A″ / B / C) | Oct 20 | not started; needs Stage 1's verdict, Gate 2's run, the kernel re-run + cell list B | Week 5 |
| Phase 2 build-out (Weeks 6–14) and Phase 3 paper (Weeks 13–20) | Oct 21 → | wait on the branch; nothing started (correct per §6: "nothing in the rewrite is worth doing until Gate 1 tells you which paper") | — |
Schedule risk in one line: the Oct 6 deliverable "Gate 1 tracker swap running" needs Stage 1 launched THIS week (≈ 41 h wall +
the panel); the Oct 20 memo needs Gate 2's ss2 pods launched by ≈ Oct 12, which needs the Gate-1 reading by then.

## Milestones (report at each in ≤ 10 lines: merged / measured / blocked / decide)
The owner's answers to §0 recorded (a DECISIONS line per decision); the L3 merge on main and Stage 1 launched (if released);
Task 11 + Amendment 2 committed with κ; `kernel_smoke2` launched; its outcome READ under Amendment 2 and GATES §G4 annotated;
Stage 1 harvested and `make gate1` rendered; the five-reviewer simulation run; DECISIONS names the branch from `gate1_verdict()`.

Begin: read the list above; present §0 with your recommendations and wait for the answers; then, per the answers, either
Appendix A (the L3 items) or §4 (the L4b plan via writing-plans, then subagent-driven-development, one implementer per
worktree). Check `git log main` and STATE at each report.

---

## Appendix A — the L3 items, verbatim from SESSION_PROMPT_L4.md's "State at handover" (2026-09-20 21:20 EDT); do these ONLY under 0(a)(ii)
- Lane L3+L5 sits on `lane/L3-gate1-tracker-swap-v2` @ f3006fc (pushed; 47 commits over main @ 278bd01; L6 gate PASS at 6065e4e,
  Task 8 — the bf16 retrieval reading in the gate1 table — landed after it). NOT merged. The owner said "merge L3 when green after
  the harvest": once results/gate1_preflight_rerun/ is harvested and `scripts/pod.py check results/gate1_preflight_rerun` is OK,
  and Amendment 1b is committed, re-run the L6 gate on the final lane SHA (scratchpad script pattern: fresh clone of the pushed
  branch, make env/test/tables/figures/check, tree clean, forbidden-word grep), then `git merge --no-ff` into main from the main
  checkout, verify (make test / tables / check / ruff), push main, delete the lane branch and worktree. If `git log main` already
  shows the merge commit, skip all of this. [2026-09-21 note: the lane head is now 91d149d (two prompt-file commits after
  f3006fc); the harvest is on disk in `.claude/worktrees/L3-gate1`, uncommitted — commit manifest, trials, diag, env by explicit
  path there after `check` is OK; the instance 51815080 is already destroyed.]
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
  The launch commit must descend from Amendment 1b's commit. [2026-09-21 note: credit $79.48; see 0(d).]
- Deferred to L6 / Phase 2 (do not do before Stage 1 launches): the vt generator repair as NEW task files + a 1 GPU-h `full` vt
  validation cell (D-018 addendum); the deferred minors in docs/plan/cleanup/l3-ledger.md and the whole-branch review's triage
  table (recon's private-name reach-in, the L2_PODS table name, the docstring items, the oja tuning-number mismatch with D-014).

## Appendix B — the L6 clean-clone gate (the L4 session's script; rewrite the two paths for your scratchpad and lane)
```bash
#!/bin/bash
# usage: l6-gate.sh <lane-branch> <base-sha> <sha>  -> writes $SP/l6-<sha>.txt (fresh clone of the PUSHED branch)
set -u
BR="${1:?lane branch}"; BASE="${2:?base sha (the lane's fork point)}"; SHA="${3:?sha}"
SP=<your scratchpad dir>; OUT="$SP/l6-${SHA:0:7}.txt"; D="$SP/l6-clone-${SHA:0:7}"
{
  echo "=== L6 gate $BR @ $SHA  $(date '+%Y-%m-%d %H:%M %Z') ==="
  rm -rf "$D"; git clone -q --branch "$BR" https://github.com/hkrishna42/kvdlra.git "$D" && cd "$D" || { echo "CLONE FAILED"; exit 1; }
  git checkout -q "$SHA" || { echo "CHECKOUT FAILED"; exit 1; }
  echo "HEAD: $(git rev-parse HEAD)"
  ln -s /Users/hari/src/kv-dlra/dumps/llama3.2-1b dumps/llama3.2-1b
  echo "--- make env ---"; /usr/bin/time -p make env 2>&1 | tail -3
  echo "--- make test ---"; /usr/bin/time -p make test 2>&1 | tail -4
  echo "--- make tables ---"; make tables 2>&1 | tail -2
  echo "--- make figures ---"; make figures 2>&1 | tail -2
  echo "--- make check (every live manifest OK; a launched-unharvested pod fails until harvested; results/gate1_preflight is the known partial) ---"
  for d in results/*/manifest.json; do [ -f "$d" ] || continue; p=$(dirname "$d"); r=$(.venv/bin/python scripts/pod.py check "$p" 2>&1 | tail -1); echo "$p: $r"; done
  echo "--- tree clean? ---"; git status --short | head -5; [ -z "$(git status --short)" ] && echo "TREE CLEAN" || echo "TREE DIRTY"
  echo "--- forbidden words over the lane's added lines ($BASE..$SHA), excluding the constraint lists ---"
  git diff "$BASE" "$SHA" | grep '^+' | grep -v '^+++' | grep -nE '\bDLRA\b|BUG integrator|\b[Hh]onest(ly)?\b|\b[Mm]arquee\b|\b[Ff]lagship\b' | grep -viE 'forbidden|constraint|never appear|do not appear|banned' | head -5  # forbidden-word list (DLRA case-sensitive: the repo path kv-dlra is not the word)
  echo "(above: 0 lines = clean)"
  echo "--- commit messages ---"
  git log --format=%B "$BASE".."$SHA" | grep -nE '\bDLRA\b|BUG integrator|\b[Hh]onest(ly)?\b|\b[Mm]arquee\b|\b[Ff]lagship\b' | head -3  # forbidden-word list
  echo "(above: 0 lines = clean)"
  echo "=== done $(date '+%H:%M %Z') ==="
} > "$OUT" 2>&1
echo "wrote $OUT"; tail -25 "$OUT"
```
The L4 gate at abc8af8 read: make env 2.9 s; make test 680 passed / 8 skipped / 105 s cold; tables diff-clean; figures OK; check
OK on every live manifest except the two pre-flight directories (the partial harvest; the re-run's harvest uncommitted) and
results/kernel_smoke (unharvested at that SHA; OK after 935f745); tree clean; forbidden-word grep 0 / 0.
