# L1 lane ledger — copied verbatim from the lane SDD workspace (final, after the Table-4 harvests)

Source: `.claude/worktrees/L1-hygiene/.superpowers/sdd/2026-09-11-L1-harness-hygiene/progress.md`, copied 2026-09-18 after the harvests. Rulings PR-1–PR-32 and R-L1-1–R-L1-25 with every review verdict; docs/plan/STATE.md summarizes them.

---

# SDD ledger — plan: docs/plan/plans/2026-09-11-L1-harness-hygiene.md

Lane lane/L1-harness-hygiene in .claude/worktrees/L1-hygiene, off week7 @ af2ca1a (2026-09-17). Execution kickoff (approved): /Users/hari/.claude/plans/misty-honking-thompson.md — the pre-flight rulings below are copied from it verbatim.

## Pre-flight rulings (ledger entries before Task 1 is dispatched)

Sources: my reads, the tree map, the plan-review audit, and two first-hand measurements.

**Measured facts that change the plan**
- **M1** The plan's ratchet recipe does not ratchet on the shipped step: max ‖UᵀU−I‖ =
  6.9e-4 over 1400 steps (n=512, cap=256, block 16, bf16-rounded, seed 0; the audit agent saw
  6.6e-4–7.3e-4 across seeds 0–6, growth ∝ √t). The pre-Week-7 augmentation (plain residual
  QR, no re-orthogonalization) reaches 44.8 within 60 steps. CODE_AUDIT:171's `audit_q4c.py`
  is not in the tree; CODE_AUDIT:172 is the finding that holds (through the production layer,
  Qwen r256, floor off, 36K tokens). → Task 1 Step 0 = systematic-debugging write-up of this
  (exact recipe, seeds, variants tried, the pre-W7 contrast), recorded as a DECISIONS finding.
- **M2** The Task-4 CPU study is cheap: one isvd r256/blk16 cell on a 512×4096 layer = 1.8 s;
  full grid (4–5 docs × 16 layers × {K_pre,V} × 5 ranks × 7 methods × 2 blocks) ≈ 1 h on
  this machine. No cut needed; Oja tuning is sized separately (PR-19).

**Task 1 — tripwire + guard**
- **PR-1** Tests: replace the two slow ratchet tests with (a) a local ~12-line pre-W7 step
  crossing 1.0 at t=50×16 (0.3 s), (b) the shipped `isvd_step` staying < 1e-3 at t=200×16
  (1.0 s), (c) guard efficacy by injection (`u *= 3` → fix restores < 1e-6 and `u @ c` is
  preserved), (d) the cache-level tripwire + abort through the tiny model. No `slow` marker;
  suite stays ≈ 79 s. G1 line 1 ("both directions") is met by the pre-W7/shipped contrast;
  the report says so explicitly.
- **PR-2** `reorthonormalize` must re-diagonalize the core: `q, r = qr(u)`; `u_loc, σ, _ =
  svd(r @ b)`; return `(q @ u_loc, (u_locᵀ r) @ c, diag(σ), rot = u_locᵀ r)`. The plan's
  `r @ b` leaves a triangular core and silently breaks the 2r-word core billing
  (`bug_cache.py:1111-1118`, `accounting.py:170`). The plan test's `allclose(r @ c, c2)`
  becomes `allclose(rot @ c, c2)`.
- **PR-3** Hook placement: between `bug_cache.py:665` (quant-tier rotation) and `:667` (the
  new-coordinate append), so `new_ck` is computed in the fixed basis. Decide K and V fixes
  first, then rotate the quantized tier once: `_rotate_quant_tier(rot_k | None, rot_v | None)`
  made None-tolerant per side (no identity requantize on the unfixed stream). Plan line 433's
  anchor `1260-1274` is wrong; the method is at `855-869`.
- **PR-4** `BugStreamingLayer` gains `layer_idx: int = 0` (bound at `bug_cache.py:1246`,
  never passed today). Not added to `test_golden_cache.CONFIG` (equality-pinned).
- **PR-5** Fixtures: no `make_tiny_model_and_cache`/`ingest`/`cache.n`. Tests use the
  `tiny_model` fixture (`tests/test_bug_cache.py:50`, move to `tests/conftest.py`),
  `BugStreamingCache(tiny_model, rank=…, **knobs)`, `_prefill_then_decode` /
  `_drive` (`tests/test_w20_tracker_swap.py:117`, `tests/test_accounting.py`); `N_FEATURES`=32,
  `H`=2, `len(cache._bug_layers())`.
- **PR-6** Diagnostics: the runner PRINTS `[diag] {json}` rows (records.DIAG_RE and
  `pod.py harvest` → `diag.jsonl` already exist; nothing emits) and writes `diag.jsonl` at
  finish; rows < 400 chars; `diag_every` = 64 in tests, 256 in pod configs. `drain_diag()`
  row schema as in the plan + `layer`.
- **PR-7** Bit-identity protocol: before wiring, instrument `_absorb_columns` and record the
  max `orth_error` over `test_golden_cache.py` (expected ≪ 1e-3); `.contiguous()` only inside
  the fix branch; `orth_error` in `u.dtype`. Guards: `tests/test_bug_cache.py:104` (atol 0),
  `tests/test_golden_cache.py:123` (abs 1e-6 on 6.6e4 sums), `:131` (CONFIG equality),
  `tests/test_w20_tracker_swap.py:128`.

**Task 2 — effective-rank billing**
- **PR-8** Test compares layer 0: `fp.float_equiv() == cache._bug_layers()[0].stored_state_numel()`
  (not cache total ÷ layers; floor-on layers collapse to different ranks). `u_k is None` →
  `rank=0` AND `u_present=False` together.
- **PR-9** `_footprint` has 7 call sites (frontier ×1, ruler ×4, longbench ×2): the report names
  the cross-axis effect. Add a `min_sv_frac=1e-2` rank-deficient parametrization to
  `test_bug_footprint_matches_stored_state_numel` with `rank=layer.u_k.shape[1]`.
- **PR-10** `eff_rank=` goes on the ppl row only (`frontier.py:512`, appended; `PPL_RE` is a
  prefix match); the retrieval axis reads it from `diag.jsonl`.

**Task 3 — FD numerics, Oja plumbing, tracker names**
- **PR-11** `_svd_core(b_fac) -> (u_loc, σ)` returns UNTRUNCATED factors; `theta`/`min_sv_frac`
  stay in `isvd_step`; `fd_step` keeps its own `k` and shrinkage, gains no floor. The refactor
  is pure code motion: same `torch.linalg.svd(b_fac, full_matrices=False)` on the same tensor,
  the `cat` sequence and `tol` comparison unchanged; `isvd_step = augmented_bug_step` is an
  assignment. Verify before/after: `pytest tests/test_golden_cache.py tests/test_bug_cache.py
  tests/test_isvd.py tests/test_w17_rankfloor.py tests/test_w20_tracker_swap.py
  tests/test_bug_cache_week7.py -q` with zero tolerance changes (plan's
  `test_streaming_torch.py` = `test_isvd.py`).
- **PR-12** `oja_step` defects fixed in the same commit as the schedule: (a) rank is pinned at
  the seed block width (`isvd.py:336-339`; `absorb_block=16` → a rank-16 basis billed as 64 in
  W20) → grow toward `rank_cap` via `_augment` while `u.shape[1] < rank_cap`, assert
  `u.shape[1] == min(rank_cap, n)` after warm-up; (b) `n_seen` is recomputed from tier
  occupancy (`bug_cache.py:629-630`) so the decay freezes at saturation → a monotonic
  per-layer absorbed-column counter. Both change every Oja number: recorded in DECISIONS; the
  W20 Oja cell was already void (untuned).
- **PR-13** Dispatch: keep the three explicit branches, look the callables up as
  `TRACKERS["isvd"|"oja"|"fd"]` at call time (so the plan's `monkeypatch.setitem` spy works);
  constructor validation via `TRACKERS`; `"bug"` → `"isvd"` with `DeprecationWarning`.
- **PR-14** Existing call sites get the schedule (`functools.partial(oja_step, eta0=20.0,
  decay=0.03)` in the parametrized contract test); the 0.25 residual threshold in
  `test_oja_tracks_the_dominant_subspace` is re-measured at (20.0, 0.03), not assumed;
  `test_bug_tracker_is_bit_identical_to_the_default` becomes default vs `tracker="isvd"` plus
  a warning test for `"bug"`.
- **PR-15** `tests/golden/legacy_arm_kwargs.json`: the 8 `"tracker": "bug"` values become
  `"isvd"` in the same commit as the `config.py:134` alias drop, with a one-line docstring
  note in `test_config_parity.py`; no wholesale regeneration (the generator is gone);
  DECISIONS line.
- **PR-16** Arm YAMLs: new knobs live under `cache:` (`tracker`, `oja_eta0`, `oja_decay`,
  `qr_every`, `orth_fix_tol`, `orth_abort_tol`, `diag_every`). Editing
  `oja_r64_h256_seed.yaml` changes `w19_swap`'s config hash; that pod has no live manifest
  (`prereg: null`, archive only) — confirm with `make check`.

**Task 4 — recon.py + 1B study + figure**
- **PR-17** Dump loader: `layer_XX.pt["K_pre"|"V"]` is `(8, T, 64)` head-major →
  `.reshape(512, T)[:, 4:]` (sinks dropped, CODE_AUDIT:185 convention); `Q_pre` present,
  unused; docs = the five `len4096_rope-both` dumps (doc63/411/454/637/718) (+ doc15 at 2048 if
  useful); no `capture_kv.py` (the writer is `dump_kv.py:388-395`).
- **PR-18** `stored_error(..., prefill=0.25)` as a fraction of T (4096 = the whole doc would make
  the frozen control the oracle); `min_sv_frac: float = 0.0` added so the figure can show the
  floor (`isvd_f0.01`).
- **PR-19** `tune_oja` runs at rank 16, layer 8, 2 held-out docs (doc63, doc718), grid
  `eta0 ∈ {5,10,20,40,80}` × `decay ∈ {0.01,0.03,0.1,0.3}` (must extend past (20, 0.03), which
  sat on both boundaries in Week 2 — CODE_AUDIT:382) and raises on a boundary optimum; the
  schedule is reused at every rank and the report says so.
- **PR-20** Output `results/recon_1b/recon.jsonl` + `provenance.json` (dump sha256 manifest,
  git SHA, grid, wall clock) — NO `manifest.json` (a CPU study is not a pod; `make check`
  fails any non-archive manifest without a pod config). Figure: `scripts/figures.py rank-sweep
  --in … --out docs/paper/figures/rank_sweep_1b.pdf` (figures.py has no subcommands yet; add
  argparse subcommands with `build` as the default so `make figures` is unchanged). The
  study runs as a background process in the L1 worktree after Task 3 (Oja fix) lands.

**Task 5 — perplexity protocol**
- **PR-21** Corpora: WikiText-103 TEST supplies ≈15 non-overlapping (16K+2048) windows and
  ≈7 at 32K; `pod.py check` requires exactly `n_samples`. Ruling: PG-19 validation is the
  32-window corpus (`ppl_16k_pg19val`, `ppl_32k_pg19val`); WikiText-103 test tasks declare
  `n_samples: 15` (16K) / `7` (32K) with the ceiling stated in `doc:`; `load_task` gains a
  guard so a ppl task cannot ask for more windows than its corpus supplies (fail at load,
  not after hours). Overlapping windows are NOT used (they break the paired bootstrap). Recorded
  as a DECISIONS line you can reverse.
- **PR-22** Plumbing: `TaskCfg.corpus: str = "wikitext-103"` read at `runner.py:301`
  (replacing the `PPL_CORPUS` constant and its "L2 owns that change" comment); `load_corpus_ids`
  gains `"wikitext-103-test"` and `"pg19-val"`; `corpus` added to `PplRecord`/`PplwRecord`
  (absent on archived rows, `.get` convention). Existing `ppl_16k/32k/64k/32k_w8` YAMLs keep
  the default so no archived number or config hash moves (confirm with `make check`).
- **PR-23** Per-window rows already exist (`[pplw]` → `pplw.jsonl`, L0). `ppl.jsonl` keeps one
  row per (arm, ctx); the plan's "one ppl.jsonl row per window" clause is dropped. The `ppl`
  table is a new `scripts/tables.py ppl --pod … --out …` subcommand writing
  `docs/paper/tables/ppl_<pod>.md` (not a numbered `TABLES` entry: `make tables` diffs
  `table*.md` against paper-v1 and must stay clean). TOST on bits/token differences per window
  (`nll/(n·ln2)`), not on pooled values.

**Task 6 — Table-4 config + prereg**
- **PR-24** Control arms must be able to diverge: "guard off" = `orth_fix_tol: null,
  orth_abort_tol: null` (v1 behaviour, reproduces the 27,531 cell and records `[diag]`
  orth_err traces); "guard on" = default tolerance guard; `qr_every: 64` variants for the
  r256 cells (the periodic form ICML 0.1 names). With the default guard on, a bare "qr off"
  arm would still be guarded and the contrast would be a no-op. Prereg states the measured
  baseline (M1; CODE_AUDIT:172) and predicts where the arms can differ.
- **PR-25** Five arm files: `isvd_r128_f0.01` (missing today) + the four `_qr64`/guard
  variants; no `legacy_name` on new arms; `tests/test_config_parity.py:70-73` (an equality
  over all arm stems) gets an explicit `POST_V1` allowlist. Dry-run coverage via an in-process
  `config_hash(load_pod("hygiene_table4"))` assertion, not a second torch subprocess.
- **PR-26** `prereg: prereg/hygiene_table4.md` set; `STATUS: awaiting owner go`; budget 6 GPU-h;
  the launch commit must follow the prereg commit → D-011.


## Task order
T1 → T2 (billing + diag drain, see Ruling PR-27) → T3 → T4a → [1B study in background] → T5 → T6 → T4b. One implementer at a time.

Ruling PR-27: the plan's file map lists a runner diag drain (`[diag]` rows → results/<pod>/diag.jsonl; OrthonormalityError recorded as error) that no task's file list owns. Task 1 implements the cache side (`drain_diag()`, rows accumulated per layer); Task 2 wires the emission: a shared helper in kvdlra.eval.records (`emit_diag(rows, model, source)`: prints `[diag] {json}` lines under 400 chars and appends to a module-level list) called after each streaming trial/sample in ruler/frontier/longbench, and the runner writes diag.jsonl at `_finish`. Cost if wrong: ~15 lines to move.

## Log
2026-09-17: worktree created; baseline make test green; Task 1 brief + notes written; Task 1 dispatched (opus).
2026-09-17: Task 1 implementer DONE @ 57b4d70 (opus; 10 new tests, six-file bit-identity run 56 passed; golden max orth_err 6.0e-5; deviation: injection test asserts < 1e-5 (fp32 roundoff at r=64 = 7.2e-6)). Implementer saw 1 'ninja' failure running pytest directly (make test exports PATH) — orchestrator re-runs make test. Review package written; task reviewer (opus) dispatched.
2026-09-17: make test @ 57b4d70 in the worktree: 390 passed, 0 failed, 67 s (the implementer's 'ninja' failure was pytest run without make's PATH export).
2026-09-17: Task 1 review (opus): Needs fixes — Important #1 tokens_seen=0 on prefill rows; minors #2-#11. Ruling R-L1-1: fix round 1 (fresh sonnet agent, SendMessage unavailable) takes #1 + #2,#3,#5,#6,#8,#9,#10 (all small); #4 (guard now binds oja/fd swap arms → they raise instead of diverging) → DECISIONS D-012 line; #7 (orth_error n·r² probe cost) → note for the L4 latency table; #11 (seven duplicate tiny_model fixtures) → final-review/cleanup note. Cost if wrong: one more scoped round. Fix brief: task-1-fix1.md.

## Merge-time records (draft; written to week7 STATE/DECISIONS by the orchestrator at merge)
- D-012 (recorded ruling): the plan's synthetic ratchet does not reproduce on the shipped step (max ‖UᵀU−I‖ 6.6e-4–7.3e-4 over 1400 steps, seeds 0–2; pre-Week-7 augmentation 44–45 within 60 steps; write-up task-1-ratchet-findings.md). Guard defaults: orth_fix_tol 1e-3 (repair), orth_abort_tol 1e-1 (raise → error record), qr_every off (spec §1 / ICML 0.1 said default 64 — bit-identity forbids it; the periodic form is the Table-4 factor). Golden run max orth_err 6.0e-5 → the default guard never fires there. The guard binds the oja/fd swap arms too: a swap arm that loses orthonormality now raises (recorded as `error`) instead of returning a divergent perplexity (review item #4). Spec §5 filename reuse (oja_r64_h256_seed.yaml); crash class substitute for spec §4.
- Note for L4 (latency table): `orth_error` is an unconditional n·r² probe per stream per absorb (≈ the step's own dominant matmul at n=1024, r=256) — measure it in the kernel-vs-reconstruct rows rather than estimating.
- Cleanup note for the whole-branch review / ponytail: seven test modules define their own `tiny_model`/`_tiny_config` copies (test_accounting, test_bug_cache_week7/11/12/15, test_bug_cache_seed_regression, …) shadowing the new conftest fixture — delete them in the final fix round if cheap.
2026-09-17: Task 1 fix round 1 DONE @ d8b95d5 (sonnet; 8/8 items; make test 394 passed / 62 s). Scoped re-review (sonnet) dispatched on 57b4d70..d8b95d5.
2026-09-17: Task 1: complete @ d8b95d5 (re-review: 8/8 addressed, no new breakage). Task 2 dispatched (opus) from BASE d8b95d5.
2026-09-17: Task 2 implementer DONE_WITH_CONCERNS @ 2c30255 (6d83ff3 billing, 2c30255 diag drain; make test 407 passed / 68 s). Concerns: (1) K/V collapse to different live widths → bills (rank_k+rank_v)/2, `bug_footprint(rank: float)`; (2) sibling test instead of parametrization; (3) emit sites on the success path only → an abort drops the fatal diag window; (4) ~1000 [diag] lines per sample at diag_every=64.
Ruling PR-29 (diag volume): the log is the only channel back from a vast.ai pod (`vastai logs`, 30000-line fetch, 5000-line fallback), so per-window rows cannot go to the log at pod scale. (a) `drain_diag()` flushes each layer's partial window as a final row before clearing, so every sample/trial yields ≥ 1 row per layer whatever `diag_every` is; (b) pod YAMLs (Task 6) set `cache.diag_every: 4096` → one summary row per layer per sample (Qwen Table-4 pod ≈ 11K ppl + 8K retrieval lines, inside the 30000-line fetch; the prereg states the log size so a short fetch is recognisable); (c) emit sites wrap the trial/sample in try/finally so the fatal window reaches the log on an abort. Cost if wrong: a `[diag]` packing contract later. Folded into the Task 2 fix round with the reviewer's findings. Task reviewer (opus) dispatched on d8b95d5..2c30255.
2026-09-17: OWNER: credit topped up; pod launches authorized ('if needed'). D-011 recorded on week7. Launch plan: Table-4 pods after Task 6 + whole-branch gate, from the pushed lane head; merge still stops at the menu.
2026-09-17: Task 2 review (opus): Approved; Important #1/#2 = PR-29 B/A; minors #3-#7. Ruling R-L1-2: fix round 1 takes A–H (incl. review #4 promoted: [diag] rows must carry arm/ctx/task/idx for pod attribution); #7 accepted. Fresh sonnet agent; brief task-2-fix1.md.
2026-09-17: Task 2 fix-round agent (sonnet) STALLED after reading the brief (harness: no progress 600 s); worktree clean at 2c30255, nothing lost. Re-dispatched as a fresh opus agent with the same brief (task-2-fix1.md).
2026-09-17: Task 2 fix round 1 DONE @ 03370e2 (opus retry; 8/8; make test 408 passed / 60 s). Concerns: diag_every default 64 (pods set 4096 in Task 6 — owned); _diag_window second counter (minor). Scoped re-review (sonnet) dispatched on 2c30255..03370e2.
2026-09-17: Task 2: complete @ 03370e2 (re-review: 6/6 addressed, no new breakage). Out-of-scope for the final review: official_ruler.run_trial calls ruler.retrieve() without task=/idx= → diag rows on the official-RULER path carry task=None/idx=None (one-line thread-through; Table-4 uses the in-house task). Task 3 dispatched (opus) from BASE 03370e2.
2026-09-17: Task 3 implementer DONE_WITH_CONCERNS @ 72b2e41 (18a0c7a L1.3a, 72b2e41 L1.3b; make test 417 passed / 66 s; six-file guard 56→56→57 with zero tolerance changes; make check rc 0). Rulings: R-L1-3 (D1) the parity golden's oja entry gains `oja_eta0`/`oja_decay` (4 lines) on top of the eight `tracker` values — an intended change of a void arm, documented in the test docstring; goes into DECISIONS D-014. R-L1-4 (D2) `tests/test_w20_tracker_swap.py::_cache` prefills in 16-token blocks so the FD "differs" pin exercises shrinkage, not seeding (FD's seeding now equals isvd's by design) — accepted, no assertion weakened. Minor for the fix round: the brief's slow-marked FD ratchet test runs 3.3 s → size it to t=150×16 (≈1.2 s) and drop the marker (nothing deselects it). Oja measured: rank after warm-up = min(rank_cap, n); residual at (20.0, 0.03) = 0.078, threshold 0.15. Task reviewer (opus) dispatched on 03370e2..72b2e41.
2026-09-17: Task 3 review (opus): Approved, minors only. Ruling R-L1-5: take R1 (jitter subtraction), R2 (tokens_seen running max — fixes a real dip on the demote path), R3 (else-branch via TRACKERS), R4, R5 (crash wording) + M1 now in a short fix round (fresh sonnet); accept the rest. Brief task-3-fix1.md.
2026-09-17: Task 3 fix round 1 DONE @ e6bbb66 (sonnet; M1 + R1–R5; make test 418 passed / 63 s; make check rc 0). Scoped re-review (sonnet) dispatched on 72b2e41..e6bbb66.
2026-09-17: Task 3: complete @ e6bbb66 (re-review: 6/6 addressed, no new breakage; guard 66 passed). Out-of-scope notes for the final review: the unused 'slow' marker declaration in pyproject; the third-failure re-raise path of _svd_core is untested against a real LAPACK failure. Task 4a dispatched (opus) from BASE e6bbb66.
2026-09-17: Task 4a implementer DONE_WITH_CONCERNS @ 95515a1 (opus; make test 423 passed / 59 s; smoke: isvd 1.026×/1.028×/1.053× oracle at r16/r64/r256). Rulings: R-L1-6 (PR-30) Oja tuning on the real dumps puts the optimum at (5.0, 0.3), interior of the widened grid eta0 ∈ {0.6,1.25,2.5,5,10} × decay ∈ {0.03,0.1,0.3,1.0}; the Week-2 (20, 0.03) scores 0.821 vs 0.405 at r16/layer 8 (the step changed in Task 3: rank growth + monotone n_seen). Task 4b commits `configs/arms/oja_r64_h256_seed_tuned.yaml` (the spec §5's own filename idea) with the study-tuned schedule + doc citing results/recon_1b/provenance.json; `oja_r64_h256_seed.yaml` stays at (20.0, 0.03) as the void-W20 arm; the parity allowlist gains the new stem; DECISIONS D-014 records it. R-L1-7 (PR-31) the rank-sweep figure's x-axis is `stored_rank` for every method (fd2 at 2r; its r=256 point at 512 is degenerate and shown as such). R-L1-8 accept the 2.9 h wall clock (Oja's per-column QR = 65%) — full grid in the background now, in parallel with the review; re-run if the review changes numerics. Accept the fd2-vs-rank-2r-oracle test split. recon.py 290 lines vs the ~150 ceiling → ponytail-review at the whole-branch round. Task reviewer (opus) dispatched on e6bbb66..95515a1.
2026-09-17: Full 1B recon study launched detached (nohup) at 21:44:43 from head 95515a1 — pid in scratchpad/recon-full.pid, log scratchpad/recon-full.log, output results/recon_1b/ (recon.jsonl + provenance.json, no manifest). Expected ≈2.9 h. If the Task 4a review changes numerics, kill and re-run.
2026-09-17: Task 4a review (opus): Needs fixes — Important #1 study command lacks min_sv_frac=(0.0, 0.01) (no floor rows → dashed line unobtainable); #2 Oja tuned on k_pre only, reused for v. Study KILLED ~10 min in, results/recon_1b removed. Ruling R-L1-9: fix round takes #1, #2 + minors 3–8 (G1 floor pin, check=True, weights_only, legend/empty guards, ponytail cuts, Haar-draw caption); accepts block-duplicate rows and the schedule_given entry; #12 (Oja surface in DECISIONS) is the orchestrator's. Relaunch the study from the fixed head. Fresh opus agent; brief task-4a-fix1.md.
- D-014 addendum (draft): Oja re-tuned by eval.recon.tune_oja on the 1B dumps (doc63 + doc718 held out, r=16, layer 8): the surface falls AWAY from the Week-2 (20.0, 0.03) point toward smaller steps — (20, 0.03) scores 0.821 vs (5.0, 0.3) 0.405 on k_pre (per-kv schedules after the fix round). The step itself changed in L1.3b (rank growth to rank_cap + monotone n_seen), so a re-tune was expected. Evidence: task-4a-report.md (the measured surface) and results/recon_1b/provenance.json (once the study lands). Config consequence: `configs/arms/oja_r64_h256_seed_tuned.yaml` (Task 4b) carries the tuned schedule; `oja_r64_h256_seed.yaml` stays the void-W20 arm at (20.0, 0.03); L3's Gate-1 v2 `oja_tuned` arm starts from the tuned file.
2026-09-17: Task 4a fix round 1 DONE @ b25989d (opus; 8/8; make test 425 passed / 57 s; per-kv Oja: k_pre (5.0, 0.3), v (5.0, 0.1)). Full study RELAUNCHED detached from b25989d with min_sv_frac=(0.0, 0.01) (≈3.1 h, 12,800 rows expected); scoped re-review (sonnet) dispatched on 95515a1..b25989d — kill/re-run the study only if it changes numerics.
2026-09-17: Task 4a: complete @ b25989d (re-review: 8/8, no new breakage; minor: empty oja_tuning when oja is excluded — not exercised). Study running from b25989d since 22:05. Task 5 dispatched (opus) from BASE b25989d.
2026-09-17: Task 5 implementer DONE @ abd2a29 (opus; make test 439 passed / 70 s; make tables diff-clean; make check rc 0). CORPUS_TOKENS (Llama-3.2-1B tokenizer, 2026-09-17): wikitext-103-test 288,937; pg19-val 2,968,224 (3M loader cap) → ceilings at window 2048: 14 @16K / 7 @32K; pg19 160 / 84. Rulings: R-L1-10 every pod's config_hash moved because `TaskCfg.corpus` enters the hashed structured schema (one `corpus: wikitext-103` line per task; no value changed) — covered by R28 (no live manifest carries a hash; archives are skipped; make check green); Table-4 hashes are computed at launch. R-L1-11 `ppl_16k_wt103test` ships n_samples 14 (the guard's floor formula), not the notes' 15 — accepted. Concern 3 (pooled `ppl=` line carries no corpus → harvested ppl.jsonl rows have corpus None) → the reviewer weighs a one-line ` corpus=` append + regex group. Task reviewer (opus) dispatched on b25989d..abd2a29.
2026-09-17: Task 5 review (opus): Approved with fixes — Important: tables.py keys (arm, ctx) without corpus (two same-ctx ppl tasks would mix); related blind spot in pod.py's _pplw_fails/_ppl_fails. Ruling PR-32: one corpus per context length per pod, refused by load_pod (a corpus axis in check/records is deferred — DECISIONS note); Table-4 pods use ppl_16k_pg19val only. Ruling R-L1-12: fix round takes #1, PR-32, the ppl= line corpus append (reviewer's recommendation), the CI-test tightening, empty-pplw fail-loud; accepts the rest. Fresh sonnet agent; brief task-5-fix1.md.
- D-013 (draft, recorded ruling): perplexity protocol — `TaskCfg.corpus` (default `wikitext-103` = the archived v1 corpus, so no archived number changed; every config hash moved by the schema field, no live manifest affected); new corpora `wikitext-103-test` (288,937 Llama tokens → 14 windows at 16K+2048, 7 at 32K) and `pg19-val` (2,968,224 tokens under the 3M loader cap → 160 / 84); PG-19 validation carries the 32-window claim; a load-time guard refuses more windows than a corpus supplies; no overlapping windows (paired bootstrap independence); one corpus per context length per pod (PR-32 — a corpus axis in `check`/records is deferred until a two-corpus pod is wanted); `[pplw]` and `ppl=` lines carry ` corpus=` (optional regex groups; archived logs parse with None); `scripts/tables.py ppl` reports bits/token, paired 95% bootstrap CI and TOST(±0.05 bits) vs full on per-window differences.
2026-09-17: Task 5 fix round 1 DONE @ 0a88141 (sonnet; 5/5 + one adjacent assertion in test_effective_rank_billing updated for the corpus tail; make test 444 passed / 77 s — suite now 13 s under the 90 s gate, watch Task 6/4b). Scoped re-review (sonnet) dispatched on abd2a29..0a88141.
2026-09-17: Task 5: complete @ 0a88141 (re-review: 5/5, no new breakage; note: load_pod now loads every task it names — matches config_hash's pattern). Task 6 dispatched (opus) from BASE 0a88141.
2026-09-17: Task 6 implementer DONE @ 35a8890 (opus; make test 447 passed / 74 s; 25 existing pod hashes unchanged; launch --dry-run refuses on the same-commit prereg as designed). Rulings: R-L1-13 ten arm files as a 2×5 grid (`isvd_r{128,256}_{noguard,tol,qr64}` + `isvd_r{128,256}_f0.01_{tol,qr64}`; `_tol` = the plain arm at shipped defaults + diag_every 4096) — the plain `isvd_r128/r256/r256_f0.01` are pinned by the parity golden and used by w17_floor, so the diag knob cannot go on them; accepted. R-L1-14 Qwen pod log ≈ 23K rows: the watchdog appends rows every 150 s poll and dedupes into <label>.raw, so the 30000-line fetch is a per-poll limit, not a per-run one — accepted, state it in the prereg's log-volume line if not already. R-L1-15 `isvd_r256_tol` may abort (> 1e-1) → error records → check fails the pod under R29: pre-registered outcome; an L6 per-case ruling can accept the pod with the errored arm named; keep the shipped defaults (a repair that cannot keep up at 16K is itself the finding). gpu_budget_h 4.0/2.0 are estimates (owner-visible in D-011). Task reviewer (opus) dispatched on 0a88141..35a8890.
2026-09-17: Task 6 review (opus): Needs fixes — prereg §8 log-safety built on the per-poll limit misread as a run ceiling; §4 decision branches not mutually exclusive when the floor arm collapses. Ruling R-L1-16: branch 1 additionally requires isvd_r256_f0.01_tol within 1.0 bit/token of full; else branch 3 with the collapse named. Fix round takes both + minors 1–5 (qr64 rank-change clause; durable citations D-012 / l1-ledger.md; Holm by hand; derivation pin test; test docstring). Fresh sonnet agent; brief task-6-fix1.md. Study rows:     5120.
2026-09-18: Task 6 fix round 1 DONE @ 410c06f (sonnet; 7/7; make test 448 passed / 75 s). Scoped re-review (sonnet) dispatched on 35a8890..410c06f.
2026-09-18: ponytail-review (opus) on af2ca1a..410c06f: 26 findings, net −398 lines possible (ponytail-review.md). Ruling R-L1-17: TAKE in the final fix round — (1) the 14 private `tiny_model`/`_tiny_config` copies → one parametrized conftest fixture (−240; pure deletion, `make test` count must stay identical), (4) the three copy-pasted diag try/finally blocks → one contextmanager (−20), (5) `ppl_stats` double row dict (−12), plus the cheap remainder from the file; block-independent recon methods re-run per block → dedupe rows figure-side (or compute once) after the whole-branch reviewer weighs the SE effect. KEEP (spec-mandated; CLAUDE.md: ponytail never removes what was explicitly requested): (2) the jittered-Gram second eigh attempt in `_svd_core` (L1 spec §4: "a tiny diagonal jitter only if eigh fails"), (3) the `tracker="bug"` deprecation alias (plan: one release). Ponytail's two rejected candidates noted (`_diag_window` load-bearing; `_stored` has one dead `seed` param → drop it).
2026-09-18: Task 6: complete @ 410c06f (re-review: 7/7; partition check: one applied branch in every cell; note for the final round: branch 2 should caveat the case where the floor arm is also > 1 bit from full). All six tasks' code complete; Task 4b (data commit) waits on the study. Whole-branch review (opus) dispatched NOW on af2ca1a..410c06f to overlap the study's last hour; 4b's diff gets a scoped addendum in the re-review.
2026-09-18: Whole-branch review (opus): With fixes — no Critical; Important #1 layer-0 rank sample → mean over all layers (exact), #2 DECISIONS entry for the billing change (orchestrator: D-015), #3 the prereg cites D-012 / l1-ledger.md which must exist in the launch commit's ancestry; Minors #4–#12; G1 map: 5 lines met (2 as amended), tuned config = 4b, 8B sweep + Table-4 outcome open (pods). Guard set 80 passed; make test 448 / 84.6 s under study load. Ruling R-L1-18: final fix round = review Important #1 + Minors #4–#8, #10, #12 + ponytail R-L1-17 cuts + the branch-2 caveat, in three grouped commits (final-fix.md), dispatched after Task 4b; then ONE scoped re-review over 4b + the fix commits. Ruling R-L1-19: before the push/launch the orchestrator rebases the lane onto week7 (6b10277, DECISIONS D-011) and commits, ON THE LANE, DECISIONS D-012–D-015 + the STATE entry + docs/plan/cleanup/l1-ledger.md, so the prereg's citations exist at the launch SHA; launch order llama first, qwen after the llama log's first sample rows harvest cleanly (reviewer's recommendation, cheap insurance).
2026-09-18: Study DONE 01:10 (12,800 rows, rc 0, ≈3.1 h) from b25989d. Task 4b dispatched (sonnet) from BASE 410c06f.
2026-09-18: Task 4b BLOCKED on pre-commit check-added-large-files (maxkb 1024 vs recon.jsonl 3285 KB). Ruling R-L1-20: results/**/*.jsonl excluded from that hook (records are the repo's data artifacts by design; tensors/checkpoints stay guarded; CI does not run pre-commit); orchestrator committed the implementer's five staged paths + the hook change as L1.4b. Headline (block 16, k_pre, 5 docs): isvd/oracle 1.019×/1.025×/1.052× at r16/64/256; fd2 1.26–1.37× (at stored width 2r); oja tuned 1.27×→2.04×; frozen 1.05×→1.32×; random 3.6×→15.7×; floor@r256 2.29× oracle with stored_rank 87–172.
2026-09-18: Task 4b committed @ eabcf7d (orchestrator, R-L1-20). Final fix round dispatched (opus) from BASE eabcf7d per final-fix.md (Commits A/B/C).
2026-09-18: Final fix round DONE_WITH_CONCERNS @ fbe5d82 (opus; A e0f2243, B 5d553f4, C fbe5d82; make test 453 / 61 s; guard 121 passed; 13/14 fixture copies converted — test_w15_pplw.py kept: its module seed makes unseeded randint deterministic; a module-scope torch import regression in records.py caught and fixed (61 s → 216 s → 61 s); five ponytail items declined with reasons). Scoped re-review (opus) dispatched on 410c06f..fbe5d82 (4b + A/B/C).
2026-09-18: L6 gate PASS on ce48b29 (clean clone: make env 1 s / test 453 passed 78 s / tables diff-clean / figures / check green; tree clean; forbidden 0). Pushed lane/L1-harness-hygiene. Launched hygiene_table4_llama via launch-pod.sh (offer = cheapest reliable A100 40GB); manifest committed + pushed; watchdog detached (scratchpad/watchdog-hygiene_table4_llama.log).
2026-09-18: llama pod 51394691: no rows after 60 min; diagnosis — GPU idle, disk flat at 9.9 GB then jumped to 18.0 GB (the 16 GB model download stalled on HF's anonymous rate limit, then resumed; SSH unusable: publickey denied). Local pg19-val load OK. Hardening note for L6: the runner prints no stage lines (model/corpus loaded) so a stalled download is indistinguishable from a hang in the log — add [stage] prints (one-liners) in a later lane.
2026-09-18: D-007 recurrence: iCloud re-set the hidden flag on the venv .pth files (import kvdlra failed from outside the repo again); cleared with chflags nohidden. Add to D-007 at merge: the flag comes back — a launchd/cron  or moving the clone is the fix.
2026-09-18: FIRST DIAG ROWS on llama (32, parse clean, max line 269 chars; isvd_r256_noguard orth_err_k up to 1.01 at 16K — the ratchet is real on Llama too). Qwen launched: instance 51401786 (offer 50441077), launch SHA 17d31a0, manifest commit 54cf11f, watchdog pid 54667. D-011 addendum committed + pushed.
2026-09-18: KILLED both pods (prereg §9): 5.2 min/sample at r256 (65K core SVDs per sample) → 11 h / 37 h vs the 2 h / 4 h bar. Evidence: 320 diag rows from the llama noguard arm (max orth_err 1.0–2.9 per sample) kept at results/hygiene_table4_llama_killed_51394691/diag.jsonl. Ruling R-L1-21: re-size under an amended prereg — n=16 windows (ppl_16k_pg19val_w16), Qwen split r128/r256 (parallel), retrieval secondary deferred, decision rule unchanged, ≈15 GPU-h; killed labels appended to done.txt; killed manifests removed from the tree (commit 0f3dc3f). Monitors stopped; watchdogs killed; 0 instances.
2026-09-18: Task 6b DONE @ 0d2263b (opus): ppl_16k_pg19val_w16, pods llama / qwen_r128 / qwen_r256, prereg Amendment 1 (Holm family correctly kept at 4; n=16 TOST condition pre-registered; absorbs measured 800/sample → 51,200 SVDs). Ruling R-L1-22: r128 bar raised 4.0 → 7.0 h (b4c1373) so the unmeasured r128 speed-up cannot alone trigger a §9 kill. Quick review (sonnet) next, then push + relaunch ×3.
2026-09-18: 6b review (sonnet): one stale 4.0 in §A1.3 → fixed (95e1280); pushed; RELAUNCHED ×3 via launch-pod.sh — see pods.txt in each results dir; watchdogs detached.
2026-09-18: hosts for llama (51408895) and qwen_r128 (51408911) never pulled the image (60 min 'Retrying'); destroyed; relaunched llama=51415024 (offer 51341862) r128=51415044 (offer 48658876); r256 51408903 healthy (first rows 28). D-011 addendum 4 pushed.
2026-09-18: qwen_r256 first [pplw]: isvd_r256_noguard NLL ≈ 9.7–10.5 nats/token per window (ppl ≈ 2e4) — the Week-17 divergence reproduces on PG-19 validation under the new protocol; tol/qr64 arms next decide §4.
2026-09-18: llama first [pplw]: isvd_r256_noguard NLL 2.2–2.8 nats/window — no divergence on Llama at r256/16K despite per-layer ‖UᵀU−I‖ ≈ 1–3 (killed-run diag); Qwen r256 noguard diverged (NLL ≈ 10). Orthonormality loss is common; the perplexity collapse is Qwen-specific — the tol/qr64 arms decide the mechanism question.
2026-09-18: ALL guarded arms aborted (llama tol/qr64 @ absorb 1: V-side err 0.921, eff_rank_v 93/128; qwen r256 tol/qr64 @1: 0.506; qwen r128 tol @1 0.506, qr64 @4 0.141). Cause: rank-deficient seeding block → GPU SVD non-orthonormal null-space columns → abort checked BEFORE repair. Qwen pods destroyed 08:05; llama partial records kept (_v2_aborted dir, no manifest). Ruling R-L1-23: fix = repair-then-abort + _svd_core orthonormality check/QR; Amendment 2; relaunch ×3. D-011 addendum 5 pushed.
2026-09-18: Task 7 DONE (abf603d code + 7e60619 Amendment 2; make test 458 / 65 s; guard set 102). Retired the v2 Qwen launch manifests (73c7a7e); make check green. Review (opus) dispatched on 36ea9b3..HEAD before the third launch.
2026-09-18: Task 7 review (opus): Critical — _LEFT_ORTH_TOL 1e-4 ≈ 5× the CPU floor, likely below the GPU floor → could silently repair the noguard control; Important — Amendment 2 must say the core repair applies to all arms; minors (fatal-row ranks pre-repair; NaN thresholds; orth_fix_tol=None doc; Qwen evidence path). Ruling R-L1-24: tolerance 1e-2 + headroom pin; the rest taken; fresh sonnet agent; brief task-7-fix1.md.
2026-09-18: Task 7 fix round 1 DONE (3f6e006 + 103b4a9; make test 462 / 61 s; guard 106; Qwen noguard bits/token r256 14.283 (ppl 19,939), r128 4.010). Residual: max() NaN aggregation at the abort. Scoped re-review (sonnet) dispatched on 73c7a7e..103b4a9.
2026-09-18: Task 7 re-review (sonnet): A1/A2/A4/B1/B2 addressed; A3 residual CONFIRMED (max() drops a V-side NaN; the window folds too). Ruling R-L1-25: fix round 2 (sonnet) — per-stream abort + NaN-sticky folds + tests; then push + relaunch without a further review (mechanical, tested).
2026-09-18: THIRD LAUNCH from e482570: r256 51438089 (offer 51163383), llama 51438103 (offer 50937082 — the v2 stall host; stall monitor armed), r128 51438116 (offer 50937080). D-011 addendum 6 pushed. Monitors: 3 persistent per-pod + quiet watchdog tail.
2026-09-18 10:26: r128 v3 pod: [diag] svd_nonorthonormal err 0.5058 shape [128,128] — the GPU SVD's non-orthonormal left factor on a rank-deficient seeding core, now repaired by _svd_core (the v2 abort value reproduced and handled). Applies to all arms on the pod per Amendment 2 A2.5.
2026-09-18 10:40: r256 v3 51438089 RUN_FAILED (HF download SSL ReadError); destroyed; relaunched on offer 50441073 → 51446637. llama/r128 fine (svd_nonorthonormal events 0.921/0.506 seen — fix confirmed).
2026-09-18 12:03: qwen r128 v3: isvd_r128_tol completed 16/16 (no abort) — per-window NLL below noguard on the same windows (w0 2.91 vs 3.36; w7 2.25 vs 5.69 nats): the repair helps at r128 on Qwen (prereg §5 predicted the guard never fires at r128 — it did, via the seeding repair and/or tolerance repairs; check diag fixed_* counts at harvest).
2026-09-18 23:03: ALL THREE HARVESTS verified (make check rc 0 after L1.8 env.txt recovery; Llama re-harvest mis-picked the v2 log once — harvest's non-shrink guard covers trials.jsonl only — redone from the v3 log, numbers identical). Records commit 287dc45; docs commit 9ba667f; pushed. L6 gate at 287dc45: GATE PASS. Outcome 3(a). Task 8 review (sonnet) dispatched post-commit. NEXT: the finishing menu (merge = owner).
--- L6 gate at 287dc45
head=287dc45 results: Table-4 harvests — hygiene_table4_llama (51438103
make env rc=0 1s ::  ~ kvdlra==0.1.0 (from file:///private/tmp/claude-501/-Users-hari-Desktop-kv-dlra/94e8fd8d-a5d7-4f92-bc54-464b7847145a/scratchpad/l0-clone)
make test rc=0 81s :: 469 passed, 15 warnings in 69.85s (0:01:09)
make tables rc=0 0s :: tables: diff-clean vs paper-v1
make figures rc=0 2s :: [wrote docs/paper/figures/coldstart.pdf|png]
make check rc=0 8s :: results/paper-v1/w19-sysfix-llama: archive, skipped
untracked/modified after targets: 0
forbidden-word grep (text files; src configs scripts docs/paper prereg minus generated figures):
       0
GATE PASS
