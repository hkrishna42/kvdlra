# DECISIONS — append-only, dated. Gate outcomes carry the evidence path. OPEN = the owner (Hari) decides; the orchestrator records options + a recommendation and continues on everything not blocked. Closing an item = a new dated entry, never an edit.

## 2026-09-11

### D-001 OPEN — Is arXiv v1 live?
- Evidence: commit 9baf571 (2026-09-06, "arXiv package delivered") changed only docs/week19-handover.md; paper/arxiv-v1.tar.gz exists locally (gitignored); a web search for the exact title on 2026-09-11 returns no arXiv listing. → v1 appears assembled but NOT posted.
- Options: (a) not posted → the L7 corrections land in paper/main.tex only, no v2 is needed, and v1 is not posted as-is (it carries the unsupported "tracker load-bearing" sentence and the framing CODE_AUDIT refutes). (b) posted → the L7 diff becomes an arXiv v2 within the week.
- Recommendation: (a). Until you confirm, L7 proceeds under (a) and notes the v2 path in its report.

### D-002 OPEN — Kernel option: (iii) Triton tile-wise reconstruct-inside-attention vs (i) post-RoPE tracking at r=128
- Recommendation (ICML2027_PLAN §2 Gate 3; L4 brief): (iii) primary — it preserves the pre-RoPE design exactly as shipped; (i) as a parallel one-row accuracy experiment (`isvd_postrope_r128`, Llama 16K, n=24). L4 formalises this in docs/adr/0001; the ADR stays OPEN until you accept it.

### D-003 OPEN — GPU split between L3 (Gate 1 v2) and L4 (kernel)
- Recommendation (KICKOFF Part E §5): Gate 1 v2 first — it decides which paper exists; kernel prototypes on a shared card at batch 1. Gate 1 v2 is budgeted ≤ 50 GPU-h; credit is $23.85, so a top-up (~$60–90 at A100 rates) is required before the G1 v2 launch commit.

### D-004 OPEN — `palu-r0.5`: real Palu port vs `svd_oracle` as an explicit static-low-rank upper bound
- Recommendation: rename to `svd_oracle_r0.5` now (a settled fact; L2 §5). Decide on a real-Palu port only if Gate 1 selects Branch A/B — it matters for the Phase-2 tables, not for the gates.

### D-005 OPEN — Launch the filler-realism diagnostic pod (Day-1 milestone; L2 §1)
- Spec: in-house four tasks, Llama-3.1-8B, 16K, n=12, real-text filler; arms = isvd r64-h256 (the paper's r64 configuration), its q4 cell, KIVI-2 streaming, KIVI-2 single-shot, full. Reading pre-registered in prereg/filler_realism.md (committed before the launch commit): a drop > 0.25 on any task for the r64 or q4 arm retires the cycled-filler generator from every headline claim.
- Cost: ~2–4 A100-hours ≈ $3–6 of the $23.85 credit. Spends money → needs your go; the orchestrator does not launch pods.
- Recommendation: go, once prereg/filler_realism.md is committed and L2 confirms `--filler` accepts a real-text source (today `--filler wikitext` draws WikiText-2 test sentences — real text, but not the PG-19 chapter the brief names).

### D-001 evidence addendum (2026-09-11, orchestrator; still OPEN for the owner)
- L7 searched paper/ for an arXiv ID, abs URL, report number or submission receipt: none; every `arXiv:` string is a citation in refs.bib; main.tex's "arXiv v1 preprint" header is intent, not a record. Combined with 9baf571 (tarball assembled) and the empty title search, the recommendation stands: (a) not posted, no v2.

### D-005 addendum (2026-09-11, orchestrator; OPEN) — cost revised, reduced-arm option
- L2prep's estimate from harvested per-trial timings: 5 arms × 4 tasks × n=12 = 9.1 h base, 18.3 h at the 2× safety factor → **$11–20** (credit $23.85). The earlier "$3–6" omitted tasks-per-cell.
- Option (a) full design, 5 arms: $11–20, most of the credit. Option (b) decisive arms only — r64 config, q4 cell, full (the rule is written on r64/q4; KIVI-2 chunked is already known to be worse under chunked prefill and its single-shot variant is a Gate-2 question, not a filler question): ≈ 5.3 h base / 10.6 h at 2× → **$6–12**.
- Recommendation: (b) now; the two KIVI arms join the ss2_families pod (L2 §4) where they belong. Prereg file to be amended to (b) before launch if you choose it; the pod-name/watchdog fixes L2prep listed are done at launch time by L2.

### D-002 evidence addendum (2026-09-11, orchestrator; OPEN — ADR ready for acceptance)
- ADR 0001 drafted (lane/L4-kernel @ e432df3): (iii) primary, (i) parallel accuracy row `isvd_postrope_r128`, (ii) rejected (GQA: 4× the key-side FLOPs of (iii); the pair factorization is per query head, reconstruction per KV head). Cost model regenerable from docs/adr/0001-cost-model.py; Week-3 gate reachable in principle (resident KV 0.60 GB vs full 4.29 GB at 32K; roofline 10.7 ms vs measured 188 ms reconstruct).
- Three concerns the owner should weigh before accepting: (1) `BugStreamingLayer.lazy_initialization` raises above batch 1 — Gate 3's batch ≥ 4 criterion needs a batched cache first (scope for L4 Week 2–3); (2) weights + full KV at 64K batch 4 = 50.4 GB does not fit an A100-40GB — that contrast moves to H100 or to "max batch that fits"; (3) the traffic model under-predicts the measured reconstruct path by 6.6–16× (full KV by 2.1–2.3×), so absolute kernel timings are not predicted, only the floors.
- Recommendation: ACCEPT (iii) primary + (i) row, with concern (1) added to the L4 milestones as the first Week-2 task.

## 2026-09-13

### D-007 OPEN — The repository lives inside an iCloud-synced folder (~/Desktop)
- Evidence: `brctl status` shows com.apple.CloudDocs syncing Desktop; during L0's test runs macOS "conflicted copy" duplicates appeared (`scripts/tables 2.py`, `tests/test_records 2.py`, plus `.mypy_cache`/`__pycache__` twins). They are untracked and were deleted, but a sync race can also resurrect stale file contents.
- Options: (a) move the clone out of iCloud (e.g. `~/src/kv-dlra`) and re-create the worktrees — the vast.ai pods clone from GitHub, so nothing else depends on the path; (b) exclude via a `.nosync` folder name (breaks every path in docs/scripts); (c) keep and mitigate (implementers commit by explicit path; orchestrator deletes `* 2.*` files at every checkpoint).
- Recommendation: (a) at the next quiet point (after L0 merges). Until then (c) is in force.

## 2026-09-14

### D-008 CLOSED (orchestrator ruling R13, recorded for the owner) — G0 "Dockerfile + uv.lock"
- uv.lock stays gitignored (the repo marks it platform-specific); reproducibility = pyproject exact pins + Dockerfile + `scripts/pod.py check` diffing env.txt against the pins. GATES.md G0 amended on the lane (96e1398, merged). Reopen if you want a committed lock for the pod image.

### D-009 CLOSED (orchestrator rulings during L0 Phase B, recorded for the owner — reopen any line)
- R34: `src/kvdlra/baselines/turbo_press.py` (+ its test) KEPT although it was a deletion candidate — Phase 2's composition Pareto (ICML plan §3.4) needs TurboQuant as a pure quantizer; L2 adds its arm YAML. Listed under "Kept despite being a candidate" in deletions.md.
- R35: `hydra-core` dropped; `omegaconf` is the direct pin (configs are plain structured configs, no hydra CLI). Two zero-caller accounting helpers and `seed_everything`'s never-passed parameter deleted.
- R29: `scripts/pod.py check` FAILS whenever a pod's manifest counts errors > 0 (`CHECK FAIL errors: N trials raised`); no tolerance knob. An errored pod becomes citable only by an L6 ruling per case. Cost if wrong: one flag to add.
- R36/R37: tests/ stays outside the forbidden-word scope (tests pin the words); skipped `[diag]` lines are counted into the manifest (`diag_skipped`) and fail `check` when > 0 — L1's diagnostics must emit rows under 400 chars or reuse the `part=` splitter.
- R38: `kvdlra.eval.persist` stays archive-only (its rows are parsed, not produced) until L4 wires it as a runner generator.
- R39: the watchdog's ROWS filter keeps `[error]` lines so a watchdog harvest counts the same errors a full-log harvest does.

### D-010 OPEN — the gitignored litter was parked, not deleted
- The L0 plan said `rm`. I moved the files instead to `~/Desktop/kv-dlra-litter-2026-09-14/` (2.7 MB): handover.md, explanation_week_{1_2,3_4,5_6}.md, next-session-prompt.md, compass_artifact_*.md, dashboards/ (two HTML dashboards), results/gpu_logs/ (30 raw `vastai logs` tails, Weeks 11–12 + the 1B/8B runs — paper-source-map.md cites none of them, so they exist nowhere else), results/scratch/ (9 smoke-run JSONs). Reason: irreversible deletion of files that live only on this disk is the owner's call; the repo outcome is identical (ignore lines dropped, tree clean, nothing under src/scripts/Makefile/configs references the paths).
- Decision: `rm -rf ~/Desktop/kv-dlra-litter-2026-09-14` when satisfied, or keep it outside the repo. Nothing depends on it.

### D-007 addendum (2026-09-14) — L0 has merged; this is the quiet point
- Only the main checkout remains as a worktree (the L0 one is removed); lane branches L2/L4/L7 have no worktrees. Moving the clone out of iCloud is now `mv` + re-creating worktrees on demand. Recommendation unchanged: (a).

### D-007 addendum 2 (2026-09-14) — iCloud also breaks the venv's `.pth` files
- Finding: every dot-prefixed path under the clone carries the macOS `hidden` flag (UF_HIDDEN; 67 in the repo, 65,660 under `.venv/`) — iCloud Desktop sync sets it. Python 3.12.13's `site` skips hidden `.pth` files ("Skipping hidden .pth file" under `python -v`), so the editable install `_editable_impl_kvdlra.pth` and `_virtualenv.pth` were silently ignored: `import kvdlra` failed from any directory outside the repo. Tests, `make tables` and the pod scripts never noticed because `scripts/_paths.py` and the test conftest prepend `src/` themselves.
- Fix applied locally (not a repo change): `chflags nohidden .venv/lib/python3.12/site-packages/*.pth` → import works. iCloud may re-flag files written later (a future `uv pip install -e` re-creates the `.pth`), so this is a symptom patch. The scratchpad clone used for the L6 gate (outside iCloud) never had the problem.
- Recommendation strengthened: (a) move the clone out of iCloud. No Makefile workaround is added — a macOS-only `chflags` in `make env` would paper over the sync hazard the owner is deciding on.
