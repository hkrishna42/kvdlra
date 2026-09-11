# STATE — append-only, dated. Every lane appends on start / block / done; L6 signs "done" entries with commands + output paths. Never edit an earlier entry.

## 2026-09-11 · orchestrator · session 1 (kickoff)
- Base: week7 @ ee8c0ab == origin/week7. No local `origin/main` ref. Tree was clean apart from the four review docs under docs/plan/ (untracked; committed this session).
- Read in full: ICML2027_PLAN §1–2, CODE_AUDIT (summary + Parts A/B/C), PC_REVIEW (AC meta-review), KICKOFF_WEEKS0-3, results/w20-close-report.md.
- Gates confirmed OPEN after Week 20 (one line each):
  - G1 (does the tracker matter?): the W20 swap arm ran Oja at the untuned defaults eta0=1.0/decay=1e-3 (validated pre-RoPE schedule is 20.0/0.03; bug_cache.py:890-897 passes only n_seen); the FD arm crashed on every trial (linalg.svd non-convergence); frozen-prefill-SVD and no-gist controls never ran → no valid tracker ablation exists.
  - G2 (edge over faithful 2-bit?): single-shot-prefill KIVI-2 on Llama 16K scores mk 1.00 / mv 0.83 vs r64 1.00 / 1.00 (2 discordant of 12) → the chunked-prefill multi-value edge was a protocol artifact; Mistral/Qwen and 32K untested.
  - G3 (kernel win?): the only measured decode path is reconstruct-then-attend: 103/188/508 ms/tok vs full 26/28/37 (4–14× slower), KV peak 3.25/6.45/12.83 GB vs full 2.05/4.08/8.14 (≈1.6×). No kernel exists.
- Scaffolding written verbatim from KICKOFF_WEEKS0-3.md by awk extraction (diff-verified against the pack): CLAUDE.md (Part A, 75 lines), docs/plan/lanes/L0..L7 (Part C), docs/plan/lanes/GATES.md (Part D); the Part E §2 note is prepended to CODE_AUDIT.md; DECISIONS.md created (D-001..D-005 OPEN).
- Tag: `paper-v1-archive` → ee8c0ab (annotated, local until L0's branch is pushed). Nothing that fed a paper-v1 table may be deleted before this tag exists — it now does.
- ponytail-debt: `grep -rnE '(#|//) ?ponytail:'` → 0 markers. Clean ledger.
- Dispatched, read-only on the week7 checkout (reports land in docs/plan/cleanup/): (1) repo-wide ponytail-audit → ponytail-audit.md; (2) static reachability graph from the pod command lines + tests → reachability.md. Both are inputs to the L0 plan and deletions.md.
- Baseline `.venv/bin/python -m pytest -q` on ee8c0ab: running in the background; result appended in the next entry.
- vast.ai credit read 2026-09-11: $23.85. Enough for the filler-realism diagnostic (D-005), not for Gate-1 v2 (≤ 50 GPU-h) — top-up before L3.
- Worktree: .claude/worktrees/L7-paper-correction on branch lane/L7-paper-correction (off week7). L7 dispatch follows; L0 dispatch waits for the two cleanup reports.
