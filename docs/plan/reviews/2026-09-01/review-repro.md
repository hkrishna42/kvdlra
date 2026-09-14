# Reproducibility & Artifact Review — kvdlra (branch `week7` @ cd3d4e7)

Reviewer role: reproducibility & artifact reviewer. Read-only audit of tests, CI, determinism,
results provenance, pod scripts, dependency pinning, and the 17-week doc trail. No code executed
beyond `ls`/`grep`/`git log`/JSON key listing.

## Verdict

**7/10 (accept-level for the reproducibility dimension).** This is an unusually disciplined solo
research repo: exact dependency pins + a 9,285-line `uv.lock`, 307 test functions across 41 files
(8,278 lines), CI with strict mypy + ruff + CPU pytest, fp-equivalence caveats documented *in the
integrator source*, derived result tables that regenerate deterministically from committed verbatim
log extracts, and 17 weeks of candid docs (including negative results). The gaps are concentrated in
one place: the Week-13→17 headline numbers rest on **stdout harvests from destroyed vast.ai pods** —
per-trial records and the pods' raw `--out-json` files are gone, the pod environment (GPU model,
torch/CUDA version, code SHA) is not machine-recorded in the committed line-files, and the pod
bootstrap clones an **unpinned branch**, so a rerun next month runs different code by construction.
A camera-ready artifact also still needs: official-benchmark configs (the RULER here is custom,
in-repo), a one-command repro of the actual headline configs, a seeds/compute table, and a model
license/mirror statement (the Llama runs use the ungated `unsloth/` mirror of a gated Meta model).

---

## 1. Tests — strong (best-in-class for a research repo)

- **Volume/coverage**: 41 test files, 307 `def test` functions, 8,278 lines (`tests/`). Coverage maps
  onto the claims: integrator parity ladders (`tests/test_streaming_torch.py:42` block-1 matches the
  NumPy reference; `:54` full-block equals the oracle SVD; `:79` exact rank-r reconstructed exactly;
  `:96` bf16 storage runs an fp32 core), cross-family parity fixtures for the Tier-2 generality claim
  (`tests/test_bug_cache_families.py:93` Qwen2 QKV-bias pin, `:101`/`:105` exact-mode parity vs
  HF `DynamicCache` for Qwen2 and Mistral — these are precisely the pins that de-risk C1's
  cross-model claim), and the Week-17 floor (`tests/test_w17_rankfloor.py:56` cap-at-effective-rank,
  `:73` **`test_floor_off_is_bit_identical_regression_guard`** — directly backs C3's "default-off /
  bit-identical" claim, `:99` end-to-end threading through the cache).
- **Accounting is test-pinned**: `pyproject.toml:82-85` puts `scripts` on `pythonpath` explicitly so
  `tests/test_accounting.py` pins against the canonical `bug_budget_floats` / `kv_memory_ratio`
  formulas "they must not drift from" — the memory-ratio claim (C1's 0.075–0.149×) is an *analytic*
  number, and the repo at least guards the formula. (That it is analytic, not measured VRAM, is the
  evaluation reviewers' problem; as an artifact the accounting is consistent and tested.)
- **CI**: `.github/workflows/ci.yml` — ruff check + format (L37-40), **mypy `--strict` over
  `src tests scripts`** (L42-43, with `[tool.mypy] strict = true`, `pyproject.toml:64-66`), pytest
  (L45-46) on pinned CPU torch 2.11.0 (L30-32). Minor gap: CI triggers on `push: branches: [main]`
  and `pull_request` only (`ci.yml:3-6`) — all 17 weeks of work live on branch `week7`, so pushes
  there do **not** run CI unless a PR is open; the handovers' "green-gated" claims
  (`docs/week17-handover.md:55`) are local runs, unverifiable from CI history.
- Stale nit: `README.md:130` says "295 passed" while the tree now has 307 test defs.

## 2. Determinism & seeds — good design, incomplete environment control

- **Task generation is fully seeded**: `scripts/w10_ruler.py:144`
  `g = torch.Generator().manual_seed(seed * 131 + trial)`, with `--seeds` (default `[0, 1]`,
  `w10_ruler.py:505`) and `--n-trials` (`:504`) explicit CLI-recorded knobs; decode is **greedy
  argmax** (`w10_ruler.py:227`, `:238`), so no sampling nondeterminism.
- **fp-equivalence is documented at the source**, which is rare and commendable:
  `src/kvdlra/integrators/streaming_torch.py:139-151` states the Week-7 QR→SVD re-orthogonalization
  changed every step, so reruns of older configs are "**fp-equivalent, not bit-identical**...
  compare methods within one run, never fresh curves against archived JSON at bit level." The
  `min_sv_frac=0.0` default is documented as "bit-for-bit the archived path"
  (`streaming_torch.py:112-118`) and enforced by `tests/test_w17_rankfloor.py:73`.
- **Gaps**: (a) no `torch.use_deterministic_algorithms(True)` / cuBLAS workspace pinning anywhere —
  bf16 CUDA runs are reduction-order nondeterministic across GPU architectures; combined with (b):
  the GPU model for the Week-16/17 runs is **not recorded in any committed artifact**. `w16.sh:30`
  does `nvidia-smi --query-gpu=name` into the pod log, but those raw logs were not committed for
  w13–w17 (`results/gpu_logs/` holds 30 files ending at `w12-*`; nothing for w13+), and the
  harvested `w17-*-lines.txt` start directly at `===W17_16K_CORE_BEGIN...` with no environment
  header. A NeurIPS checklist "compute resources" answer currently cannot name the GPU for the
  headline runs. (c) The binomial cells are n=4–16, so exact per-trial replay matters more than
  usual — and per-trial records are gone (see §3).

## 3. Results provenance — the weakest area, split by era

- **Weeks 10–12 (good)**: raw JSON results were base64-folded into the pod log and scraped intact
  with integrity checks (`scripts/pod/scrape_w10.sh:15-24` — awk block extract, `base64 -d`,
  `json.load` validation), and 30 raw/acc pod logs are committed under `results/gpu_logs/`.
- **Weeks 13–17 (lossy)**: the pod scripts write `--out-json results/w17-...json` *on the pod*
  (`scripts/pod/w16.sh:148`, `:152`, `:168-175` etc.), but pods are destroyed after harvest
  (`docs/week16-handover.md:3` "all pods destroyed") and those JSONs were **never retrieved** — the
  repo has no `w16-*-16k-uncapped.json` / `w17-*-32k-s32-n16.json`, only the stdout **line-files**
  (`results/w17-{llama8b,qwen,mistral}-lines.txt`, 44/42/49 lines) harvested from `vastai logs`
  (`docs/week17-handover.md:72`). Each line is an *aggregate* over trials
  (`[vt ctx32768] bugSseed-r128-h1024-s32 acc=0.94 ... ` — `results/w17-llama8b-lines.txt`), so
  per-trial outcomes, model outputs, and needle placements are unrecoverable.
- **Reconstruction is lossy-but-currently-safe**: `scripts/w17_intervals.py:72`
  `hits = round(float(acc) * n)` reconstructs hit counts from a 2-decimal aggregate. For the n used
  (4, 12, 16) the round-trip is unambiguous (grid spacing 1/16 = 0.0625 > 0.01), but this is
  fragile-by-inspection, not by design — at n≥50 it would silently break.
- **Traceability of the derived tables is genuinely good**: `w17-decision-table.json` and
  `w17-ruler-intervals.md` regenerate deterministically from the committed line-files
  (`scripts/w17_intervals.py:1-17`), with the block→n mapping (`NBLOCK`, `:37-44`) exactly matching
  the pod script's trial arithmetic (`w16.sh:146-148` n-trials 6 × seeds {0,1} = 12 for 16K_CORE;
  `:167-171` 8×2=16 MARQUEE; verified consistent), and an explicit superset-not-pooled rule for the
  doubly-measured think/palu@32K cells (`w17_intervals.py:11-14`) — a real double-counting guard.
  The Wilson numbers in `results/w17-ruler-intervals.md` (e.g. marquee 0.94 [0.72,0.99] 15/16)
  match briefing claim C4.
- **Code-version provenance of the runs is by-narrative, not by-machine**: the pod bootstrap is
  `git clone --branch week7 https://github.com/hkrishna42/kvdlra.git` (`w16.sh:34`) — **branch, not
  SHA**. Which commit actually ran on 2026-08-16 is asserted only in handover prose
  (`docs/week17-handover.md:55-64`, week7@cd3d4e7/130cf19); nothing in the committed evidence
  records it, and a rerun of the same script today executes different code by construction.
- **Working-tree hygiene**: the review snapshot has *modified, uncommitted* headline figures
  (`figures/week10/ruler_accuracy.pdf/.png`, `git status`) and an **untracked orphan data file
  `results/w15b-complete-lines.txt`** containing headline-adjacent rows (32K niah_multivalue
  s32=1.00 vs non-s32=0.00) that is referenced by no script or doc (grep over `scripts/ docs/ src/`
  is empty). Uncommitted evidence lying around is exactly how provenance stories die in artifact
  review.

## 4. Pod scripts — reproducible-in-shape, not in-environment

`scripts/pod/w16.sh` (219 lines, 5 MODEs) is a real pre-registered protocol: cheap-first ordering
with comments, probe **gates that KILL the run** if `full` can't recover the needle
(`w16.sh:59-64`), env-gated fix arms (`VTFIX`/`FLOOR`/`MARQUEE`, `:141`, `:153-176`), and grep-able
`===BLOCK===` markers designed for the harvest. But the environment is only half-pinned: the pod
installs `kvpress==0.5.1`, `transformers==5.8.0`, `datasets==2.21.0` (`w16.sh:38-39`) **on top of
whatever torch/CUDA/python the vast.ai image ships** — `uv.lock` never touches the GPU path. The
version echo (`w16.sh:41`) went to logs that weren't kept for w16/17.

## 5. Dependency pinning — strong on CPU, asymmetric on GPU

`pyproject.toml:13-28`: `torch==2.11.0`, `transformers==5.8.0`, `kvpress==0.5.1`,
`datasets==2.21.0` (with an in-line comment explaining the kvpress cap, L17), `accelerate==1.13.0`,
plus dev pins (`pytest==9.0.3`, `ruff==0.15.12`, `mypy==2.0.0`, L31-36) and `uv.lock` (9,285 lines,
`torch` specifier `==2.11.0` at lock L3063). CI installs the CPU wheel of the same torch
(`ci.yml:30-32`). This is checklist-grade for the CPU/test story; the GPU asymmetry is §4's issue.

## 6. Doc trail — exemplary

44 files under `docs/` spanning week1→week17, including negative-result documents
(`week7-dominance.md`, the Week-8 codebook loss, Week-12 supersessions), per-week handovers with
commit tables (`docs/week17-handover.md:55-64`), sizing-miss postmortems (`docs/w14-sizing.md`), and
a README that leads with "**competitive compressor with distinct niches — not a universal SOTA**"
(`README.md:33-34`) and states the model-mirror choice (`docs/week17-handover.md:71`). Honesty
claims C5 are documented where a reviewer can find them. The `README.md:133-151` "Reproduce"
section exists but reproduces a *non-headline* config (r32-h256 at 16K), not C1's r64-h256 3-model
matrix nor C4's r128-h1024-s32 n=16 marquee.

## 7. NeurIPS reproducibility checklist scoring

| Checklist item | Status |
|---|---|
| Code released, license | **Yes** — Apache-2.0 (`LICENSE`, `pyproject.toml:11`) |
| Exact dependency versions | **Yes (CPU)** — pins + `uv.lock`; **partial (GPU)** — pod torch/CUDA/python unpinned |
| Training/eval commands | **Partial** — README examples + pod scripts, but no one-command headline repro |
| Random seeds reported | **Yes** — `--seeds 0 1` in every pod invocation; generator formula at `w10_ruler.py:144` |
| Error bars / statistical method | **Yes** — Wilson 95% CIs, n per cell, superset rule (`w17_intervals.py`) |
| Compute resources disclosed | **No** for w16/17 — GPU model unrecorded in committed artifacts; $ costs only in private memory/handovers |
| Raw results preserved | **Partial** — w10–12 raw JSON+logs committed; w13–17 aggregates only, per-trial lost |
| Benchmark provenance | **No** — RULER is custom in-repo (`scripts/w10_ruler.py` synthetic tasks); LongBench never run on flagship; no official-suite configs |
| Model licenses/cards | **No** — `unsloth/Meta-Llama-3.1-8B-Instruct` ungated mirror used to skip the HF gate (`docs/week17-handover.md:71`); Llama 3.1 Community License unaddressed |
| Deterministic-algorithms flags | **No** — greedy decode + seeded tasks, but no torch determinism flags; bf16 cross-GPU variance uncontrolled |

## 8. What a camera-ready artifact still needs (gap-fills, with cost)

1. **One-command repro scripts for the paper's tables** (`repro/table1.sh` → C1 matrix,
   `repro/marquee.sh` → C4), pinned to a SHA, writing JSON locally. Mostly refactoring of
   `w16.sh` MODE bodies into non-vast form. *CPU-only work, ~1 day; ~$30-50 GPU to re-verify.*
2. **Environment capture in every result artifact**: prepend `git rev-parse HEAD`, `nvidia-smi
   name`, `torch/transformers/kvpress` versions to the harvested block, and clone by SHA not
   branch in pod scripts (one-line change at `w16.sh:34`). *CPU-only, hours.*
3. **Preserve raw JSONs**: resurrect the `scrape_w10.sh` base64-fold pattern (already in-repo!)
   for the w16/17-style runs so `--out-json` files and per-trial records survive pod destruction;
   add per-trial rows to `w10_ruler.py` output. *CPU-only, hours; re-running w17 to regenerate the
   lost per-trial data ~$30-70 GPU.*
4. **Official benchmarks**: run official RULER (or at least publish the custom-task generator
   spec + a diff vs official templates) and one LongBench pass on the flagship config — this is
   both an eval and an artifact gap; reviewers *will* ask why "RULER" numbers come from an in-repo
   reimplementation. *~$50 GPU + days.*
5. **Seeds/compute table + model licenses section** in the paper appendix: seeds, n per cell,
   GPU type, $ cost, total GPU-hours; Llama 3.1 Community License statement and justification (or
   replacement) of the `unsloth/` mirror; HF model cards cited. *CPU-only, ~1 day.*
6. **Hygiene**: commit or delete the modified `figures/week10/ruler_accuracy.*` and the orphan
   `results/w15b-complete-lines.txt`; update the stale `README.md:130` test count; extend CI
   triggers to the working branch. *CPU-only, minutes.*
7. (Robustness, optional) Replace `round(acc*n)` reconstruction with emitting `hits/n` directly in
   `w10_ruler.py`'s summary line, so `w17_intervals.py:72` stops depending on 2-decimal printing.
   *CPU-only, minutes.*

## 9. Severity triage

- **Fatal**: none. Nothing here invalidates the numbers; the committed line-files + deterministic
  interval scripts form a coherent, auditable chain from pod stdout to paper table.
- **Major**: lost raw/per-trial results for w13–17; no machine-recorded environment (GPU, SHA) for
  the headline runs; custom-RULER-without-official-configs as an artifact gap; model-mirror
  licensing unaddressed.
- **Minor**: unpinned pod torch; CI branch triggers; dirty working tree + orphan data file; stale
  README count; `round(acc*n)` fragility; no torch determinism flags.
