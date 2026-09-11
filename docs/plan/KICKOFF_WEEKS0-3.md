# kvdlra — Claude Code kickoff pack, Weeks 0–3

Designed against the local repo at `~/Desktop/kv-dlra`, branch `week7` = `origin/main` @ `ee8c0ab` (2026-09-09), not the shallow GitHub clone used for the code audit. Read §0 first: the Week-20 pods you harvested on Sep 9 change what Weeks 0–3 must do.

The pack has five parts. **Part A** is the `CLAUDE.md` to commit at the repo root. **Part B** is the message you paste into Claude Code to start the session. **Part C** is the per-lane briefs (commit under `docs/plan/lanes/`). **Part D** is `GATES.md`. **Part E** is what to do before you paste.

---

## 0. State of the repo (what the audit could not see)

Corrections to the code-audit addendum, from the local tree:

- **History is intact.** 343 commits, tags `v0.1-w1`…`v0.3-w4-hybrid`, branches `week1-scaffold`…`week7`. The "one squashed commit" finding in the stats audit was an artifact of the `--depth 1` clone. Pre-registration *is* dateable from `git log` — but the marquee-comparator finding (named after the result) stands.
- **Reproducibility scaffolding exists.** `pyproject.toml` with pinned deps, `uv.lock`, `.pre-commit-config.yaml`, CI (`ci.yml`: ruff, mypy `--strict`, pytest CPU on Python 3.12 + torch 2.11 CPU wheel; `latex.yml`). Hydra `configs/` dir exists. No root `Makefile`. Pods run on vast.ai over SSH via `scripts/pod/w*.sh` MODE functions with a watchdog harvester. `dumps/llama3.2-1b` (4.8 GB, gitignored) holds the real 1B KV dumps — §4.1 can be redone locally.
- **Week-20 pods were harvested (`results/w20-close-report.md`, commit `cdf0395`) and the paper was updated to claim "tracker load-bearing" (`4d37043`). That claim is not supported by the harvest:**
  1. *Tracker swap.* The Oja arm was called with `oja_step` defaults `eta0=1.0, decay=1e-3` (`bug_cache.py:890-897` passes only `n_seen`), not the Week-2 validated schedule (pre-RoPE `(20.0, 0.03)`), despite the comment saying so. It collapsed (ppl 732, multi-key 0.08). The Frequent Directions arm crashed on every trial (`_LinAlgError: linalg.svd failed to converge`, `results/w19_harvest/swap-llama.raw:69-…`) and produced no rows. Frozen-prefill-SVD and no-gist controls were never run. **Gate 1 is open.** The sentence folded into the paper must be reverted or qualified before anything else — if the arXiv package (`9baf571`) was actually posted, a v2 correction goes in the queue this week.
  2. *Measured decode cost (`sysfix`).* Llama-3.1-8B, A100-40GB, batch 1: flagship 103 / 188 / 508 ms per token at 16K / 32K / 64K vs full KV 26 / 28 / 37 and KIVI-2 44 / 68 / 119; **KV peak 3.25 / 6.45 / 12.83 GB vs full 2.05 / 4.08 / 8.14 GB** (the reconstruct-then-attend path is 1.6× full KV at 64K, 4–14× slower). The kernel is not an enhancement; it is the paper.
  3. *Single-shot 2-bit control (`ss2`).* KIVI-2 with full-precision prefill on the same needles: multi-key 1.00, multi-value 0.83 vs flagship 1.00 / 1.00. The in-house multi-value edge is 2 discordant trials of 12 — the "2-bit loses multi-value" claim was a prefill-protocol artifact on Llama 16K. **Gate 2 fails on the one cell tested**; it must be rerun on Mistral/Qwen and at 32K before any claim survives.
  4. *Sub-cliff cell on the official anchor (`q4off`).* `bugSseed-r64-h256-q4` at 0.048× scores **0.00 on all nine official RULER tasks**, including single-needle, where the in-house generator gave 1.00. The in-house filler is a fixed 10-sentence cycle (`w10_ruler.py:73-93`) — a near-rank-deficient haystack that a rank-64 gist summarizes almost for free. Working hypothesis: every in-house band/retrieval headline is partly a filler artifact. First diagnostic in Week 0: flagship and q4 on the in-house tasks with `--filler` set to real text; if the band collapses there too, the in-house generator is retired from headline claims (kept only for depth-sweep diagnostics).
- The internal AC review of 2026-09-08 gave borderline-accept 6; the external five-reviewer simulation gave 3–4. The gap is entirely the four items above plus the tracker-identity and Palu findings. Do not average them; fix the items.

---

## Part A — `CLAUDE.md` (commit at repo root, branch `week7`)

```markdown
# kvdlra — agent feedforward

## What this is
Streaming low-rank KV-cache compression: a rank-r gist tracked online by a block
incremental-SVD step (`isvd`; the shipped "BUG step" at theta=None, min_sv_frac=0
IS Brand-2006 incremental SVD — streaming_torch.py:290-293), a surprise-selected
exact tier, sinks, a recent ring, a warm-up seed. From Week 1: a fused
factored-attention kernel. Target: ICML 2027. Working branch: week7.

## Read before touching anything
docs/plan/ICML2027_PLAN.md · docs/plan/CODE_AUDIT.md · docs/plan/PC_REVIEW.md ·
docs/plan/STATE.md (append-only log) · docs/plan/DECISIONS.md (append-only) ·
docs/plan/lanes/*.md · prereg/*.md · results/w20-close-report.md

## Facts that are settled (do not re-litigate; see CODE_AUDIT.md for file:line)
- The tracker is incremental SVD. "DLRA", "BUG integrator", "honest", "marquee",
  "flagship" do not appear in new code, configs, docs/paper, or commit messages.
  The CKL error bound does not apply to a column stream; no bound is claimed
  unless proven (a Frequent-Directions-type lemma is the applicable kind).
- U has no orthogonality guard; divergence is loss of orthonormality (audit A §Q4).
- §4.1 must score every method on its STORED representation.
- `palu-r0.5` is a per-sequence SVD oracle → rename `svd_oracle`.
- The KIVI arm is a transformers QuantizedCache mixin (G=64, chunked prefill).
  A faithful KIVI (G=32, R=128, full-precision prefill) is a separate arm.
- The Week-20 Oja swap arm used untuned defaults; the FD arm crashed. The
  "tracker load-bearing" sentence in paper/main.tex is unsupported.
- Measured decode: reconstruct path is 4–14× slower than full KV and its KV
  peak exceeds full KV. No systems claim is made until the kernel replaces it.
- The in-house filler (10 cycled sentences) is suspected of flattering the
  gist. No headline number comes from it after Week 0's realism diagnostic.
- Perplexity uses WikiText-103 TRAIN. Switch to TEST / PG-19 validation.
- A number is citable only if `make tables` regenerates it from a committed
  results/<pod>/trials.jsonl whose manifest passes `scripts/pod.py --check`
  and whose prereg/<pod>.md was committed before the launch commit.

## Skills
superpowers: brainstorming (only for genuinely open design questions: kernel ADR,
config schema, FD numerics), writing-plans (every lane, before code),
subagent-driven-development + dispatching-parallel-agents (execution; one lane
per git worktree via using-git-worktrees), test-driven-development (all src/),
systematic-debugging (any number that disagrees with expectation),
verification-before-completion (before any "done"), requesting-code-review /
receiving-code-review (before merge), finishing-a-development-branch (merge).
ponytail: `/ponytail full` for all src/ work; seven-rung ladder before every new
function; `/ponytail-audit` + `/ponytail-debt` in the cleanup lane;
`/ponytail-review` on every PR. Ponytail never removes: trust-boundary
validation, data-loss guards, the orthogonality tripwire, provenance, tests.

## Repo rules (CI-enforced after L0)
- Reuse what exists: uv.lock, pyproject pins, pre-commit, ci.yml, hydra configs,
  the vast.ai pod scripts + watchdog. Extend; do not rewrite.
- Target layout:
    src/kvdlra/{tracker,cache,tier,quant,baselines,kernel,eval,accounting,util}
    configs/{arms,tasks,pods}/*.yaml      # every experiment is a YAML
    prereg/*.md                           # committed BEFORE the pod launch commit
    scripts/{pod.py,tables.py,figures.py,dump_kv.py}  # the only entrypoints
    scripts/pod/                          # vast.ai launch/watchdog (kept, thinned)
    results/<pod>/{manifest.json,trials.jsonl,ppl.jsonl,diag.jsonl,env.txt}
    results/paper-v1/                     # archived per-trial logs behind v1 tables
    tests/                                # CPU-only, < 90 s, in CI
    docs/{plan,adr,paper}                 # nothing else under docs/
- Delete, don't archive — except files that produced a number in a paper-v1
  table, which are archived under git tag `paper-v1-archive` before removal.
- manifest.json records git SHA, config hash, HF model revision, dataset/
  haystack SHA256, torch/cuda/triton versions, GPU, wall-clock, command line.
- Seeds explicit; unseeded RNG raises. A trial that raises is recorded as
  `error` and counted; it never silently reduces n.
- One commit per logical change; commit message carries pod id + prereg path.

## State
docs/plan/STATE.md and docs/plan/DECISIONS.md are append-only, dated, never
edited. Every lane appends on start / block / done. Gate outcomes go to
DECISIONS.md with the evidence path.

## Definition of done: docs/plan/lanes/GATES.md
```

---

## Part B — Kickoff prompt (paste as the first message in Claude Code, cwd `~/Desktop/kv-dlra`, branch `week7`)

```
You are the orchestrator for kvdlra Weeks 0–3 (docs/plan/ICML2027_PLAN.md §1–§2).
Read, in order, and do not summarize back: CLAUDE.md; docs/plan/ICML2027_PLAN.md
§1–§2; docs/plan/CODE_AUDIT.md in full; results/w20-close-report.md;
docs/plan/lanes/*.md. Then confirm you can state, in one line each, why Gate 1,
Gate 2 and Gate 3 are all still open after Week 20. If you cannot, re-read.

Mode: superpowers subagent-driven-development with dispatching-parallel-agents;
one git worktree per lane (using-git-worktrees) off week7; /ponytail full for
the session. You plan, dispatch, review, gate, merge. You do not write src/ code.

Lanes (briefs in docs/plan/lanes/). Dispatch order: L0 and L7 immediately; L1,
L2, L4, L5 when L0's config/entrypoint skeleton lands (day 2, not day 4); L3
after L1+L2; L6 continuously.
  L0  repo-cleanup-and-reproducibility
  L1  harness-hygiene         (QR guard/tripwire, effective-rank billing, §4.1
                               stored-repr scoring, ppl split, FD numerics fix,
                               Oja schedule plumbing)
  L2  generator-v2-and-baselines (filler-realism diagnostic FIRST, then real
                               haystacks + depth sweep, faithful KIVI, svd_oracle
                               rename, SnapKV/PyramidKV/EA sweep, ThinK+SnapKV,
                               OjaKV e2e, ss2 across families)
  L3  gate1-tracker-swap-v2   (isvd / oja-tuned / fd-fixed / frozen-prefill-svd /
                               no-gist / random-basis; 3 families; 16K+32K)
  L4  kernel                  (ADR → Triton tile-wise prototype → correctness →
                               measured VRAM + ms/token vs the w20 sysfix rows)
  L5  bf16-gist-and-prereg
  L6  verifier                (GATES.md; STATE.md signatures; owns make tables)
  L7  paper-correction        (revert/qualify the "tracker load-bearing" and
                               "band" sentences now; arXiv v2 note if v1 is live)

Process per lane: writing-plans → my approval → executing-plans with TDD in the
worktree → spec-compliance review subagent → code-quality review subagent with
/ponytail-review → L6 runs the gate → finishing-a-development-branch → STATE.md.

Hard rules:
  - verification-before-completion: a lane is done when the gate output (test
    log, manifest, regenerated table) is attached. Claims without artifacts
    are rejected.
  - A number that disagrees with CODE_AUDIT.md or w20-close-report.md beyond
    noise stops the lane; systematic-debugging before any config change.
  - No pod launches without prereg/<pod>.md in an EARLIER commit. L6 checks
    SHA order at harvest.
  - Nothing that fed a paper-v1 table is deleted before `git tag
    paper-v1-archive` exists (L0 confirms in STATE.md).
  - Forbidden words in new code/configs/docs/commits: DLRA, BUG integrator,
    honest, marquee, flagship.
  - Decisions that are mine go to DECISIONS.md as OPEN with options and your
    recommendation; continue on everything not blocked. Known ones: kernel
    option (iii) vs (i); real Palu vs svd_oracle-as-upper-bound; GPU split
    between L3 and L4; whether arXiv v1 is live and needs a v2.

Milestones (report at each in ≤10 lines: merged / measured / blocked / decide):
  Day 1   L0 plan approved; /ponytail-audit + /ponytail-debt reports in
          docs/plan/cleanup/; deletion list with reachability evidence;
          L7 diff for the paper sentences ready for my review; L2 filler-realism
          diagnostic launched on the pod (flagship + q4, in-house tasks, real
          text filler, Llama 16K, n=12).
  Day 2   configs/ + scripts/{pod,tables,figures,dump_kv}.py skeleton merged so
          L1/L2/L4/L5 can start; root Makefile; paper-v1 logs converted to
          trials.jsonl and `make tables` reproduces Tables 1,2,3,6,7,8 of v1.
  Day 4   L0 merged; CI green on clean clone; unreachable-file check = 0.
  Wk 1    L1 merged incl. FD fix + Oja schedule via config; Table-4 cells
          re-run under the guard; kernel ADR accepted; bf16-gist result;
          prereg/gate1_tracker_swap_v2.md committed; filler diagnostic in
          DECISIONS.md (in-house generator retained or retired from headlines).
  Wk 2    Generator v2 + faithful KIVI merged with tests; Gate 1 v2 pod
          launched (3 families × 16K/32K × 6 trackers); kernel single-layer
          correctness passing; ss2 launched on Mistral/Qwen 16K+32K.
  Wk 3    Gate 1 v2 harvested; `make tables` renders it with Holm + TOST;
          DECISIONS.md names the branch (A/B, A″, C); kernel full-model decode
          with tier+ring+sinks; first measured row beating the w20 sysfix
          reconstruct path on KV peak and ms/token; L2 baseline smoke on Llama
          16K; §4.1 rank-sweep figure from the local dumps + 8B dumps.

Begin: run /ponytail-audit and /ponytail-debt on the repo yourself, attach both
to the L0 brief, and dispatch L0 and L7.
```

---

## Part C — Lane briefs (commit under `docs/plan/lanes/`)

### `L0_repo_cleanup_reproducibility.md`

**Goal.** A stranger with one GPU regenerates every table from a config and a log; nothing unreachable from the four entrypoints + tests exists; the existing scaffolding (uv, CI, pre-commit, hydra, vast pod scripts) is reused, not replaced.

1. **Reachability graph.** Roots: `scripts/{pod,tables,figures,dump_kv}.py` (to be created by consolidating the `w*_` scripts), `scripts/pod/*.sh` (actual command lines that produced results), `tests/`. Static imports + shell parsing. Unreachable → deletion candidate, listed with evidence in `docs/plan/cleanup/deletions.md`.
2. **Candidates to verify** (from the audit; confirm before deleting): `src/kvdlra/integrators/{bug,bug_adaptive,bug_class,bug_torch}.py` (ODE integrators never imported by tracker/press/cache), `streaming_variants.py`, `morph_cache.py` unless a v1 table uses it, `scripts/w5…w19_*.py` one-offs (fold survivors into the four entrypoints), `figures/week*/` (regenerate), `experiments/`, `docs/week*.md`, `docs/notes`, `docs/board`, `docs/week10_report`, `handover.md`, `next-session-prompt.md`, `explanation_week_*.md`, `compass_artifact_*.md`, `figs/`, `dashboards/`, `mkdocs.yml` + mkdocs deps (unless docs are actually published), `.mypy_cache`/`.pytest_cache`/`.ruff_cache`/`.DS_Store` (gitignore), `dumps/*.log` and `pod_watch_*` (keep `dumps/llama3.2-1b`, gitignored, with a `scripts/dump_kv.py --verify` SHA manifest committed). Keep `docs/reviews/` → move to `docs/plan/reviews/`. Keep `paper/` (arXiv assembly) but move its `Makefile` targets under the root Makefile.
3. **Tag first.** `git tag paper-v1-archive ee8c0ab`. Convert every per-trial line file behind v1 tables (`results/w18_pertrial`, `w19_pertrial`, the W15/W17 line files feeding Table 3 and the 16K think/palu comparators) into `results/paper-v1/<pod>/trials.jsonl` with schema `{model, arm, task, ctx, seed, trial, hit, frac, haystack_id: null, depth: null, prompt_sha256: null, error: null}` — nulls where unrecorded, never inferred. Then delete the originals.
4. **Environment.** Keep `uv.lock`; add a `Dockerfile` (CUDA 12.x, torch 2.11 + matching triton, transformers 5.8.0, kvpress 0.5.1, quanto, hqq) and `make env`; `scripts/pod.py` writes `env.txt` and diffs against the lock. The vast.ai launch path becomes `scripts/pod.py launch --pod <name>` wrapping the existing SSH + watchdog logic (reuse `w18_watchdog*.sh`, keep the 5000-line log-tail fallback from `847778f`).
5. **Config system.** Hydra is already a dependency — use it. `configs/arms/*.yaml` for every arm in v1 (`isvd_r64_h256_seed.yaml`, `svd_oracle_r0.5.yaml`, `think_c0.5.yaml`, `kivi2_streaming.yaml`, `kivi2_faithful.yaml`, `ea_k0.1.yaml`, …), `configs/tasks/*.yaml` (generator, ctx, n, depths, haystacks, seeds, filler), `configs/pods/*.yaml` (cross product + prereg path + GPU budget). CLI flag soup in `w10_ruler.py`/`w10_frontier.py` becomes config; the two scripts become `kvdlra/eval/{ruler,frontier}.py` called by `scripts/pod.py`.
6. **`make tables` / `make figures`.** `kvdlra/eval/stats.py` is the single statistics module: port the verified `wilson` and `mcnemar_exact`, add `holm`, `bh`, `paired_bootstrap`, `tost`; unit-test against the audit's recomputed values (12/12 → LB 0.758; 10/0 → 1.95e-3; 6/0 → 3.13e-2). Tables emit markdown + LaTeX to `docs/paper/tables/`; both targets run < 5 min on CPU.
7. **CI.** Extend `ci.yml`: `make test` (< 90 s), `make tables` on `results/paper-v1/` diffed against `docs/plan/paper-v1-tables.md`, the reachability check, and a grep that fails on `dlra|bug integrator|honest|marquee|flagship` under `src/ configs/ docs/paper/`. Keep mypy strict.
8. **README ≤ 120 lines.** What it is, install, reproduce one table in three commands, layout.

**Ponytail.** Audit + debt before and after; the after-report is evidence. Deletion is the default. Net LOC delta must be negative.

### `L1_harness_hygiene.md`

1. **Orthogonality guard + tripwire.** Thin QR of U every K absorbs (default 64) and after any rank change; per-layer `‖UᵀU−I‖_F` and effective rank to `diag.jsonl`; re-orthonormalize immediately above 1e-3; mark the trial `error` above 1e-1. Regression test: the audit's synthetic ratchet (n=512, cap=256, outlier channels, bf16-rounded) exceeds 1 without the guard and stays < 1e-5 with it.
2. **Effective-rank billing** in `accounting.py`; test that a collapsed floor-on config bills fewer bytes than its configured rank.
3. **§4.1 scoring.** `kvdlra/eval/recon.py`: every tracker exposes `stored()`; error computed only from it. Methods: `svd_oracle`, `isvd`, `fd` (ℓ=r, 2r), `oja` (tuned on 2 held-out docs; schedule saved in config), `frozen_prefill_svd`, `random_basis`. Run first on the local `dumps/llama3.2-1b` (CPU/MPS, no pod), then on 8B dumps from `scripts/dump_kv.py` (8 PG-19/GovReport docs × 3 families × {16K, 32K}, all layers, K and V; SHA manifest committed, data not).
4. **FD numerics fix.** `fd_step` crashed with `linalg.svd` non-convergence on real KV. Replace the dense SVD of the shrink step with an eigendecomposition of the small Gram (`torch.linalg.eigh` on `[B|A]`ᵀ-side, symmetric, well-conditioned) with a driver fallback (`gesvd`) and a tiny diagonal jitter only if `eigh` fails; test on the exact block that crashed (recover it from the swap pod's config + seed) and on the synthetic ratchet stream.
5. **Oja schedule plumbing.** `bug_cache.py:890-897` must pass `eta0`, `decay` from config; default them to the Week-2 validated pre-RoPE schedule `(20.0, 0.03)` and add a `configs/arms/isvd_r64_h256_seed_oja_tuned.yaml`. Test asserts the config values reach `oja_step`.
6. **Perplexity.** WikiText-103 TEST + PG-19 validation; window 2048; ≥ 32 windows; per-window NLL stored; fp32 accumulation; `stats.py` paired CI + TOST at ±0.05 bits.
7. **Re-run the Table-4 cells** (guard on/off × floor on/off, Qwen r=128/256, Llama r=256) under `prereg/hygiene_table4.md`; DECISIONS.md states whether the guard alone removes divergence.

### `L2_generator_v2_and_baselines.md`

1. **Filler-realism diagnostic (Day 1, before anything else).** In-house four tasks, Llama 16K, n=12, `--filler` = real text (PG-19 chapter), arms: flagship, q4 cell, KIVI-2 (streaming and single-shot), full. Compare to the cycled-filler rows. Written reading, pre-registered in `prereg/filler_realism.md`: if the flagship or q4 drops > 0.25 on any task, the cycled-filler generator is retired from every headline claim (DECISIONS.md), and all v1 in-house tables are marked "diagnostic only" in the paper.
2. **Generator v2** (`kvdlra/eval/gen.py`): ≥ 4 haystack sources (PG-19, arXiv text, Wikipedia, synthetic essays), balanced depth design over {0.05, 0.2, 0.4, 0.6, 0.8, 0.95}, ≥ 2 code families, official RULER task semantics. Records carry `haystack_id`, `depth`, `code_family`, `seed`, `prompt_sha256`. Pairing test: same (task, ctx, seed, trial) → byte-identical prompt across arms; golden file.
3. **Faithful KIVI** (`kvdlra/quant/kivi.py`): G=32, R=128, per-channel K / per-token V, fp16 full prefill, scales + zeros counted; reference kernels if shipped, else quanto with the deviation in the config. Keep `kivi2_streaming.yaml` labelled. KVQuant-style pre-RoPE 2/3-bit dense-and-sparse if it fits; else DECISIONS entry.
4. **ss2 across families.** Single-shot KIVI-2 and KIVI-4 on Mistral/Qwen 16K + all three at 32K, same needles as a1, n=12 → `prereg/ss2_families.md`. Reading: the multi-value claim survives only where flagship vs single-shot KIVI-2 is separated after Holm over 3 families × 4 tasks.
5. **Rename** `palu-r0.5` → `svd_oracle_r0.5`; docstring states what it is not. DECISIONS entry: real Palu (which variant) vs oracle-as-static-upper-bound.
6. **Eviction / structured** via kvpress: SnapKV, PyramidKV, ExpectedAttention at k ∈ {0.10, 0.15, 0.25}; ThinK(0.5)+SnapKV(0.15) as intended; standalone ThinK as ablation only. ShadowKV and `ea_k0.25` v1 rows must appear in `make tables`.
7. **OjaKV end-to-end** from their repo, matched bytes, Llama 16K/32K; commit + arXiv version recorded.
8. **Smoke pod**: all arms, Llama 16K, n=12, generator v2 — harness validation, not paper data.

### `L3_gate1_tracker_swap_v2.md`

Depends on L1 (guard, FD fix, Oja plumbing) and L2 (generator v2). `configs/pods/gate1_tracker_swap_v2.yaml`: same cache × tracker ∈ {isvd, oja_tuned, fd_fixed, frozen_prefill_svd, no_gist (bytes rebalanced to h ≈ 1024), random_basis} × {Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B} × {16K, 32K} × 4 tasks × n=24 (2 haystacks × 3 depths × 4 codes) + perplexity 16 windows paired. `prereg/gate1_tracker_swap_v2.md` (L5 writes; L3 cannot launch before it): primary contrasts isvd vs frozen_prefill_svd and isvd vs no_gist on retrieval (McNemar, Holm over 12) and perplexity (paired CI, TOST ±0.02 bits); decision rule verbatim from ICML2027_PLAN.md §2 Gate 1. ≤ 50 GPU-h. Any arm with > 0 `error` trials is reported as failed, not as `--`. Harvest → `results/gate1_v2/` → `make tables` → DECISIONS.md names the branch.

### `L4_kernel.md`

The w20 sysfix rows are the baseline to beat: reconstruct path 103 / 188 / 508 ms per token and KV peak 3.25 / 6.45 / 12.83 GB at 16K / 32K / 64K; full KV 26 / 28 / 37 ms and 2.05 / 4.08 / 8.14 GB.

1. **ADR** (`docs/adr/0001-factored-attention-kernel.md`) via `brainstorming` over the three options in ICML2027_PLAN.md §2 Gate 3, with a FLOP + HBM-traffic model per decode step at n=1024, r ∈ {64,128}, T ∈ {16K…128K}. Recommendation: (iii) Triton tile-wise reconstruct-inside-attention as primary; (i) post-RoPE r=128 as a parallel accuracy experiment (config `isvd_postrope_r128.yaml`, one Gate-1-style n=24 row on Llama 16K). Status OPEN until I accept.
2. **Prototype (iii).** Decode-only Triton kernel: each 64-token K/V tile materialized in SRAM from `U @ C_tile` (bf16 in, fp32 accumulate), RoPE at true positions, fused QKᵀ/softmax/PV; tier + ring + sinks appended as dense tiles. Tests: max |Δ| < 1e-2 bf16 vs reconstruct-then-attend on random and on dumped 8B KV for one layer; full-model greedy decode token-exact on ≥ 14/16 prompts with mismatches logged.
3. **First numbers** (Week 3): `torch.cuda.max_memory_allocated` and ms/token for full / reconstruct / kernel at 16K/32K/64K, batch 1 and 4, A100 (H100 if available), same script as w20 (`w20_latency.py` → `kvdlra/eval/latency.py`). Gate: kernel KV peak < full KV at 32K and ms/token < reconstruct path by ≥ 3×. Target for Week 5 (ICML plan Gate 3): resident KV ≤ 0.25× and tokens/s ≥ full KV at 64K, batch ≥ 4.

### `L5_bf16_gist_and_prereg.md`

1. **bf16 gist storage** (`isvd_r64_h256_seed_bf16.yaml`): C and U stored bf16, fp32 accumulate in the step and the guard; 3 families × 16K/32K × 4 tasks × n=24 + perplexity; reading: non-inferior to fp32 storage within δ = 0.03 retrieval and 0.02 bits. If it passes, every byte-matched comparison is re-planned at the new bytes (DECISIONS entry, not a silent rerun).
2. **Prereg files**: `filler_realism.md`, `hygiene_table4.md`, `gate1_tracker_swap_v2.md`, `ss2_families.md`, `bf16_gist.md`, `l2_smoke.md`, `kernel_smoke.md`. Each: arms, tasks, n, primary contrast(s), family size + correction, decision rule, GPU budget, the SHA it must precede.
3. **Provenance**: `scripts/pod.py` writes `manifest.json` at launch and harvest; `--check` verifies config hash, model revision, dataset SHA, env vs lock; `make tables` refuses unchecked pods; test that a tampered config hash is rejected.

### `L6_verifier.md`

Continuous. Owns `GATES.md`, `stats.py`, `make tables`, STATE.md signatures. For every "done": re-run the lane's tests from a clean worktree; re-run `make tables` and diff against claimed numbers; `/ponytail-review` on the merged diff; forbidden-word grep; prereg SHA precedes launch SHA; manifest error count equals `error` rows in `trials.jsonl`; no arm reported as `--` where trials errored. Sign STATE.md with commands + output paths. Failure → back to the lane with `systematic-debugging` required.

### `L7_paper_correction.md`

Same day, $0, no pod. In `paper/main.tex`: (a) remove or qualify every sentence derived from commit `4d37043` that says the tracker is load-bearing — the Oja arm was untuned and FD crashed; (b) mark the sub-0.05× band as "in-house generator only; 0.00 on the official anchor" until the filler diagnostic and a fixed-generator rerun say otherwise; (c) state the measured decode cost from `w20-close-report.md` §2 plainly (4–14× slower, KV peak above full KV) wherever the paper currently says "≈1.06× residency" or "+10%"; (d) replace "beats fixed-rank incremental SVD everywhere" with the stored-representation statement once L1 §3 lands (until then, delete it). Produce a diff for my review; do not push. DECISIONS entry: is arXiv v1 live? If yes, v2 with (a)–(c) within the week.

---

## Part D — `GATES.md`

```
G0  cleanup + reproducibility
    [ ] git tag paper-v1-archive == ee8c0ab
    [ ] docs/plan/cleanup/deletions.md: every removed path + reachability evidence
    [ ] reachability check: 0 unreachable files under src/; net LOC delta negative
    [ ] `make env && make test` green on clean clone; tests < 90 s
    [ ] `make tables` regenerates v1 Tables 1,2,3,6,7,8 from results/paper-v1/ and diffs
        clean against docs/plan/paper-v1-tables.md
    [ ] every v1 arm/task/pod is a YAML under configs/; w10_ruler/w10_frontier flag soup gone
    [ ] Dockerfile + uv.lock; scripts/pod.py --check passes on a synthetic manifest;
        pod launch/watchdog path reused, not rewritten
    [ ] forbidden-word grep clean; README ≤ 120 lines; ponytail after-report committed

G1  harness hygiene
    [ ] guard + tripwire merged; ratchet regression test passes in both directions
    [ ] effective-rank billing test passes
    [ ] recon.py scores from stored(); test: isvd stored error == cache rot-carried error (1e-6)
    [ ] FD step runs on the block that crashed the swap pod; no `--` arms
    [ ] Oja eta0/decay reach oja_step from config; tuned config committed
    [ ] rank-sweep figure regenerates from local 1B dumps (Week 1) and 8B dumps (Week 3)
    [ ] perplexity on TEST/PG-19 val, per-window NLL, paired CI + TOST
    [ ] Table-4 cells re-run under prereg; DECISIONS.md: guard alone removes divergence? y/n

G2  generator v2 + baselines
    [ ] filler-realism diagnostic harvested; DECISIONS.md: in-house generator retained/retired
    [ ] pairing test (identical prompt sha256 across arms); balanced-depth test
    [ ] kivi2_faithful: G=32, R=128, full-precision prefill; dequant test; bytes incl. scales
    [ ] ss2 on Mistral/Qwen 16K + 3 families 32K harvested; Holm result in DECISIONS.md
    [ ] svd_oracle rename complete; grep "palu" hits only the docstring + DECISIONS entry
    [ ] SnapKV/PyramidKV/EA k∈{0.10,0.15,0.25}, ThinK+SnapKV run in smoke pod
    [ ] OjaKV e2e runs on Llama 16K; version recorded
    [ ] ShadowKV and ea_k0.25 v1 rows appear in make tables

G3  gate 1 v2
    [ ] prereg SHA precedes launch SHA; ≤ 50 GPU-h in manifest
    [ ] 6 trackers × 3 families × 2 ctx × 4 tasks × n=24 harvested; 0 arms with error > budget
    [ ] make tables renders Holm-corrected retrieval + TOST perplexity
    [ ] DECISIONS.md names the branch with the rule from ICML2027_PLAN.md

G4  kernel
    [ ] ADR 0001 with FLOP/HBM model; OPEN or ACCEPTED by me
    [ ] single-layer max|Δ| < 1e-2 bf16; full-model greedy token-exact ≥ 14/16
    [ ] results/kernel_smoke/manifest.json: full / reconstruct / kernel at 16K/32K/64K, b=1,4;
        kernel KV peak < full at 32K; ms/token < reconstruct by ≥ 3×

G5  bf16 gist + prereg
    [ ] bf16 run harvested under prereg; non-inferiority result in DECISIONS.md
    [ ] all seven prereg files committed before their pods; SHA-order check passes
    [ ] tampered-manifest rejection test passes

G6  verifier
    [ ] every STATE.md "done" entry carries a verifier signature with commands + output paths
    [ ] no merge without spec-compliance review + /ponytail-review

G7  paper correction
    [ ] diff reviewed by me; unsupported sentences gone; measured decode cost stated
    [ ] DECISIONS.md: arXiv v1 status and v2 plan
```

---

## Part E — Before you paste

1. `cd ~/Desktop/kv-dlra && git checkout week7 && git status` — commit or stash the untracked `.claude/` (keep `settings.local.json` local; add `.claude/settings.local.json` to `.gitignore`).
2. Copy the three review-session documents to `docs/plan/` as `PC_REVIEW.md`, `CODE_AUDIT.md`, `ICML2027_PLAN.md`, and add a one-paragraph note at the top of `CODE_AUDIT.md`: "History is complete (343 commits); the 'squashed' remark in Part C §T4 was a shallow-clone artifact. Week-20 harvest (Sep 9) supersedes the 'no results' remarks in Parts B §Q6/Q8 and C §T8 — see results/w20-close-report.md and docs/plan/lanes/L7."
3. Create `docs/plan/STATE.md` and `docs/plan/DECISIONS.md` with a dated first entry each; create `docs/plan/lanes/` with Part C and `GATES.md` with Part D; commit `CLAUDE.md` from Part A.
4. Install: `/plugin install superpowers@claude-plugins-official`, `/plugin marketplace add DietrichGebert/ponytail`, `/plugin install ponytail@ponytail`.
5. Decide now, so the orchestrator is not blocked on day 1: is arXiv v1 live (commit `9baf571` says "package delivered")? Kernel option (iii) primary? GPU split (suggested: Gate 1 v2 first, kernel prototypes on a shared card at batch 1)?
6. Paste Part B.

Notes for you, not the agents: the orchestrator will want to keep the ODE integrators "for history" and to archive rather than delete — hold the line, the tag keeps them. It will also want to read the harvested w20 swap rows as Gate-1 evidence — they are not; the arm was untuned and FD crashed, and the brief says so. When Gate 1 v2 harvests, run the five-reviewer simulation on the table alone before reading it yourself.
