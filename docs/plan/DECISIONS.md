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

## 2026-09-17

### D-011 OPENED and AUTHORIZED — Table-4 re-run pods (L1 Task 6: `hygiene_table4_qwen`, `hygiene_table4_llama`)
- Owner, 2026-09-17: "Credit is topped up so you can go ahead and launch pod if needed you have all the permissions to do so." Pod launches inside the approved lanes are authorized; each launch is still recorded here with pod name, prereg SHA, launch SHA, budget, and the harvest outcome.
- Plan for L1: launch the two Table-4 pods (≈ 6 GPU-h total, ≈ $5–8) only after Task 6 commits `prereg/hygiene_table4.md` AND the lane passes its whole-branch review + fix round + L6 clean-clone gate, from the pushed lane head (`scripts/pod.py launch` refuses an unpushed SHA and a prereg that is not a strict ancestor). The merge of L1 into week7 remains a separate decision (stops at the finishing menu unless the owner says "merge L1 when green").
- Not launched under this authorization without a further plan step: the 8B KV-dump pod for the Week-3 rank sweep (8B dumps cannot come back through the log channel; the study must run on the pod and emit rows — a design step for L1 Task 4b's report / L3), and the L2 filler diagnostic (D-005; L2's Task 1, after L1 merges and the lane rebases).

## 2026-09-18

### D-012 CLOSED (orchestrator ruling, recorded for the owner) — the orthonormality guard: what was measured, what ships
- The L1 plan's synthetic ratchet (n=512, cap 256, block 16, rank-40 signal + 4 outlier channels ×1e3, bf16-rounded) does NOT ratchet on the shipped incremental-SVD step: max ‖UᵀU−I‖_F = 6.6e-4–7.3e-4 over 1400 blocks (seeds 0–2; growth ∝ √t). The pre-Week-7 augmentation (plain residual QR, no re-orthogonalization) reaches 44–45 within 60 blocks. CODE_AUDIT.md:171's `audit_q4c.py` is not in the tree; CODE_AUDIT.md:172 — the ratchet through the production layer on Qwen r256, floor off (6e-4 → 1.5e-3 → 3.7e-3 → 35.1 at 8K/16K/32K/36K tokens) — is the finding that stands. Evidence: docs/plan/cleanup/l1-ledger.md (Task 1 Step 0), tests/test_orth_guard.py (pre-W7 vs shipped contrast; injection; cache tripwire).
- Shipped defaults: repair (thin QR + core re-diagonalization) above `orth_fix_tol = 1e-3`; abort (`OrthonormalityError`, recorded as `error`) above `orth_abort_tol = 1e-1`; periodic QR (`qr_every`) OFF. Spec §1 / ICML 0.1 asked for periodic QR every 64 absorbs by default — refused: it changes `u_k` on every stream and breaks the r64 golden; the periodic form is the Table-4 factor (`_qr64` arms). On the golden run the guard never fires (max 6.0e-5).
- Consequences: the oja/fd swap arms now raise instead of returning a divergent number; spec §4's "the exact block that crashed the swap pod" is not recoverable (the log holds the LAPACK message, no tensor) — the crash class is reproduced synthetically; spec §5's `isvd_r64_h256_seed_oja_tuned.yaml` became `oja_r64_h256_seed.yaml` (schedule plumbed) + `oja_r64_h256_seed_tuned.yaml` (D-014).

### D-013 CLOSED (recorded ruling, reopenable) — perplexity protocol
- `TaskCfg.corpus` (default `wikitext-103`, the archived v1 corpus: no archived number changed; every pod's config hash moved by the schema field only, no live manifest affected). New corpora: `wikitext-103-test` (288,937 Llama-3.2-1B tokens → 14 non-overlapping windows at 16K+2048, 7 at 32K) and `pg19-val` (2,968,224 tokens under the 3M loader cap → 160 / 84). PG-19 validation carries the 32-window claim (`ppl_16k_pg19val`, `ppl_32k_pg19val`); WikiText-103 test runs at 14/7 with the ceiling stated. No overlapping windows (paired-bootstrap independence). A load-time guard refuses more windows than a corpus supplies; `load_pod` refuses two corpora at one context length (a corpus axis in `check`/records is deferred until a two-corpus pod is wanted). `[pplw]` and `ppl=` lines carry ` corpus=` (optional regex groups; archived logs parse with None); `dataset_sha256` = sha256 of the token ids the windows were cut from. `scripts/tables.py ppl` reports bits/token, paired 95% bootstrap CI and TOST(±0.05 bits) vs `full` on per-window differences; Holm over a named family is applied by hand.

### D-014 CLOSED (recorded ruling) — Oja: rank pin fixed, schedule re-tuned, parity golden edited by hand
- `oja_step` pinned its rank at the seed block width (a rank-16 basis billed as 64 in the W20 swap) and its `n_seen` froze at tier saturation; both fixed (rank grows to `rank_cap` via the shared augmentation; `n_seen` = the monotone tokens-seen). Every Oja number changes; the W20 cell was already void (untuned).
- Re-tuned by `kvdlra.eval.recon.tune_oja` on the 1B dumps (doc63 + doc718 held out, r=16, layer 8): the surface falls AWAY from the Week-2 (20.0, 0.03) point; keys (5.0, 0.3), values (5.0, 0.1) (results/recon_1b/provenance.json, tuned at r=16 / layer 8 on doc63 + doc718, reused at every rank); (20, 0.03) scores 0.821 vs 0.405 on keys. `configs/arms/oja_r64_h256_seed_tuned.yaml` carries the key schedule; `oja_r64_h256_seed.yaml` stays the void-W20 arm at (20.0, 0.03); L3's Gate-1 v2 `oja_tuned` arm starts from the tuned file.
- `tests/golden/legacy_arm_kwargs.json`: the eight `tracker: bug` values rewritten to `isvd` and the oja arm's entry gained the schedule knobs (by hand; the generator was deleted in L0; documented in tests/test_config_parity.py).

### D-015 CLOSED (recorded ruling) — memory billing uses the live tracked rank
- `_footprint` bills the mean over all gist layers of the live basis widths (K and V averaged; exact by linearity of every rank term in `stored_state_numel`), not the configured cap. Byte-identical for every floor-off arm at the cap (all archived arms); a fresh floor-on ratio (e.g. `isvd_r256_f0.01`) is NOT comparable to the archived Week-17 row, which billed the cap. Evidence: tests/test_effective_rank_billing.py, tests/test_accounting.py (measured K/V asymmetry 21 vs 23 on a collapsed layer). The retrieval and longbench axes' `ratio=` move likewise for floor-on arms.

### D-011 addendum (2026-09-18, orchestrator) — Table-4 pods LAUNCHED under the owner's authorization
- `hygiene_table4_llama`: vast.ai instance 51394691 (offer 51163391, A100 PCIe 40 GB, $0.40/h), launch SHA ce48b29 (prereg first commit 35a8890, a strict ancestor), manifest commit 17d31a0; `hygiene_table4_qwen`: instance 51401786 (offer 50441077, A100 SXM4 40 GB, $0.40/h), launch SHA 17d31a0, manifest commit 54cf11f. Both from the pushed lane branch `lane/L1-harness-hygiene`; watchdogs running on the owner's machine (harvest + destroy on ALL_DONE; credit floor $6).
- Early signal from the first Llama sample (`isvd_r256_noguard`, 16K): per-layer ‖UᵀU−I‖ up to 1.01 with the guard off (layer 0: 0.15 / 0.21 for K/V; effective rank 128 of 256) — the production-layer ratchet CODE_AUDIT:172 reported on Qwen also shows on Llama at 16K. The tolerance and qr64 arms will show whether repair alone restores the perplexity; outcome recorded here after harvest (G1 line 8).
- Note for L6: `pod.py launch` echoes vast.ai's `instance_api_key` into the orchestrator's console (instance-scoped) — redact it in a later lane.
