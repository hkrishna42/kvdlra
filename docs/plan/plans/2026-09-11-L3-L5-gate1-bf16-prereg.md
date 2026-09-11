# L3 + L5 — Gate-1 v2 Tracker Swap, bf16 Gist Storage, Pre-registrations: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Part L3 runs in worktree `.claude/worktrees/L3-gate1` (branch `lane/L3-gate1-tracker-swap-v2`), Part L5 in `.claude/worktrees/L5-bf16-prereg` (branch `lane/L5-bf16-gist-and-prereg`); they touch disjoint files and can run in parallel once L1 and L2 have merged.

**Goal (L3):** Everything Gate 1 needs except the GPU hours: the two missing control trackers (frozen-after-prefill, random fixed basis) and the byte-matched no-gist arm; the pod configs (two stages, sized from measured per-trial times); the `gate1` table with Holm-corrected McNemar on the pre-registered contrasts and paired-CI + TOST perplexity; and a `gate1_verdict()` that applies the branch rule from `ICML2027_PLAN.md` §2 verbatim, so DECISIONS.md names the branch from a function, not a reading.

**Goal (L5):** bf16 storage of the gist (U, C, B) with fp32 accumulation in the step — the cheapest potential halving of the stored bytes — as a default-off, bit-identical-when-off knob, billed at 16 bits when on; and the pre-registration files that no pod may launch without (`gate1_tracker_swap_v2.md`, `bf16_gist.md`, `kernel_smoke.md`), plus verification that L0's provenance checks (tampered hash, prereg-before-launch SHA order) exist.

**Architecture:** New trackers use the same `step(u, b, block, rank_cap, **kw) -> (u, b, rot)` contract from L1 (`kvdlra.tracker.TRACKERS`), so the cache, the recon study and the accounting need no new branches. bf16 storage is a per-layer `gist_dtype` applied at the store boundary (after the step and the coordinate carry), never inside the step. Pod sizing uses the harvested per-trial wall-clock figures (L2prep report), with the Week-14 lesson (tasks-per-cell) written into the arithmetic.

**Tech Stack:** torch, omegaconf configs (L0), `kvdlra.eval.stats` (L0), `kvdlra.tracker` (L1), generator v2 task configs (L2).

**Spec:** `docs/plan/lanes/L3_gate1_tracker_swap_v2.md`, `docs/plan/lanes/L5_bf16_gist_and_prereg.md`, `docs/plan/ICML2027_PLAN.md` §2 Gate 1 (1.1) and Gate 3 (3.3), `docs/plan/lanes/GATES.md` §G3 and §G5.

## Global Constraints

- Executes after L1 (trackers, guard, FD fix, Oja plumbing) and L2 (generator v2 task configs) have merged into `week7`.
- Forbidden words in new code/configs/docs/commits: `DLRA` (word), `BUG integrator`, `honest(ly)`, `marquee`, `flagship`.
- `tests/test_golden_cache.py` stays green: `gist_dtype` defaults to `float32` and the default path is bit-identical.
- No pod launch inside these lanes. Every pod has a prereg file committed in an EARLIER commit than the launch commit (L0's `scripts/pod.py launch` refuses otherwise); the owner launches (money).
- Gate 1 budget: ≤ 50 GPU-h per stage in the manifest; the prereg states the arithmetic with a 2× safety factor.
- Any arm with > 0 `error` records in a harvested pod is reported as **failed**, never as `--` (the `gate1` table prints `FAILED (k errors)` in that cell and `gate1_verdict()` refuses to rule while a primary-contrast arm has errors).
- Commit messages carry `L3.<n>` / `L5.<n>`.

---

## Part L3 — Gate 1 v2

### Task L3.1: `frozen` and `random` trackers; byte-matched no-gist arm

**Files:**
- Modify: `src/kvdlra/tracker/isvd.py` (+ `frozen_step`, `random_step`), `src/kvdlra/tracker/__init__.py` (`TRACKERS["frozen"]`, `TRACKERS["random"]`), `src/kvdlra/cache/bug_cache.py` (constructor kwargs `freeze_after: int = 4096`, `basis_seed: int = 0`; dispatch passes `n_seen`/`freeze_after` to `frozen_step` and `seed`/`n` to `random_step`)
- Create: `configs/arms/frozen_r64_h256_seed.yaml` (`tracker: frozen`, `freeze_after: 4096`, `doc:` "xKV/ShadowKV-style: incremental SVD over the first 4096 tokens, then the basis is frozen; coordinates of later tokens are plain projections"), `configs/arms/random_r64_h256_seed.yaml` (`tracker: random`, `basis_seed: 0`, `doc:` "floor control: fixed Haar-random orthonormal basis"), `configs/arms/nogist_h<H>.yaml` (`kind: bug`, `rank: 1`, `hh_budget: H`, `hh_select: surprise`, `seed_hh_warmup: true`, `doc:` "tier + ring only; H chosen so stored bits at 16K match isvd_r64_h256_seed within 5% (computed by tests/test_gate1_arms.py::test_nogist_is_byte_matched)")
- Test: `tests/test_gate1_arms.py`

**Interfaces:**
- `frozen_step(u, b, block, rank_cap, *, n_seen: int, freeze_after: int)` → while `n_seen < freeze_after` behaves as `isvd_step`; afterwards returns `(u, b, I_r)` unchanged (rot = identity, so the coordinate carry is a no-op) — the caller appends `uᵀ block` as usual.
- `random_step(u, b, block, rank_cap, *, seed: int)` → on the seeding call returns a Haar-random orthonormal `u` (`torch.linalg.qr(torch.randn(n, rank_cap, generator=g))`) and `b = I`; afterwards identity like `frozen_step`.
- The cache passes `n_seen = (self.c_k.shape[1] if self.c_k is not None else 0) + self._q_len()` (the same expression the Oja branch uses).

- [ ] **Step 1: Tests**
```python
import torch

from kvdlra import accounting as acc
from kvdlra.eval.config import arm_kwargs, load_arm
from kvdlra.tracker import TRACKERS
from kvdlra.tracker.isvd import frozen_step, isvd_step, random_step


def test_frozen_matches_isvd_before_freeze_and_stops_after() -> None:
    g = torch.Generator().manual_seed(0)
    m = torch.randn(64, 20 * 16, generator=g)
    ua = ba = uf = bf = None
    seen = 0
    for s in range(0, m.shape[1], 16):
        blk = m[:, s : s + 16]
        ua, ba, _ = isvd_step(ua, ba, blk, 8)
        uf, bf, rot = frozen_step(uf, bf, blk, 8, n_seen=seen, freeze_after=160)
        seen += 16
        if seen <= 160:
            assert torch.equal(ua, uf)
        else:
            assert torch.equal(rot, torch.eye(8)) and torch.equal(uf, u_frozen)
        if seen == 160:
            u_frozen = uf.clone()


def test_random_basis_is_orthonormal_fixed_and_seeded() -> None:
    u1, b1, _ = random_step(None, None, torch.randn(64, 16), 8, seed=3)
    u2, _, rot = random_step(u1, b1, torch.randn(64, 16), 8, seed=3)
    u3, _, _ = random_step(None, None, torch.randn(64, 16), 8, seed=3)
    assert torch.allclose(u1.mT @ u1, torch.eye(8), atol=1e-5) and torch.equal(u1, u2) and torch.equal(u1, u3)
    assert torch.equal(rot, torch.eye(8))
    assert set(TRACKERS) >= {"isvd", "oja", "fd", "frozen", "random"}


def test_nogist_is_byte_matched_to_isvd_r64_at_16k() -> None:
    t, n = 16384, 1024
    ref = load_arm("isvd_r64_h256_seed"); nog = [p.stem for p in __import__("pathlib").Path("configs/arms").glob("nogist_h*.yaml")]
    assert len(nog) == 1
    kw_ref, kw_nog = arm_kwargs(ref, t), arm_kwargs(load_arm(nog[0]), t)
    fp_ref = acc.bug_footprint(n, rank=64, coord_count=t - 4 - 32 - 256, recent_len=32, retention="lowrank_surprise",
                               hh_count=256, hh_select="surprise")
    fp_nog = acc.bug_footprint(n, rank=1, coord_count=t - 4 - 32 - kw_nog["hh_budget"], recent_len=32,
                               retention="lowrank_surprise", hh_count=kw_nog["hh_budget"], hh_select="surprise")
    assert abs(fp_nog.stored_bits() / fp_ref.stored_bits() - 1.0) < 0.05
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement the two trackers (≈ 40 lines), the dispatch, the three YAMLs (compute H by solving `stored_bits(nogist, H) == stored_bits(isvd r64)` at 16K with `acc.bug_footprint` — record H and the arithmetic in the YAML `doc`). **Step 4:** `make test` → PASS (golden untouched). **Step 5:** Commit `L3.1: frozen + random trackers; byte-matched no-gist arm`.

### Task L3.2: Gate-1 pods (two stages), the `gate1` table, `gate1_verdict()`

**Files:**
- Create: `configs/pods/gate1_v2_stage1.yaml`, `configs/pods/gate1_v2_stage2.yaml`, `src/kvdlra/eval/gate1.py`, `tests/test_gate1_table.py`
- Modify: `scripts/tables.py` (`gate1 --pods results/gate1_v2_stage1 [results/gate1_v2_stage2]` → `docs/paper/tables/gate1.md`), `configs/tasks/ppl_16k_wt103test_g1.yaml` (window 2048, `n_samples: 16` — the Gate-1 sizing)

**Pod sizing (from the L2prep report's harvested per-trial wall-clock, r64 streaming arm, A100-40GB):** per 16K trial ≈ 1.0 / 3.0 / 3.5 / 1.5 min for single / multi-key / multi-value / vt → one arm × one family × 4 tasks × n=24 ≈ 24 × 9.0 min = 3.6 h; 32K ≈ 2× = 7.2 h; perplexity 16 windows × 2048 at 16K ≈ 0.5 h/arm. Six trackers (`isvd`, `oja`, `fd`, `frozen`, `random`, `nogist`) — the no-gist arm has no gist rebuild and runs ≈ 2× faster; frozen/random ≈ 1.5× faster (no per-absorb SVD).
- **Stage 1** = Llama-3.1-8B + Qwen2.5-7B × 16K × 6 arms × 4 tasks × n=24 + ppl(16): ≈ 2 × (3.6 + 3.6 + 3.6 + 2.4 + 2.4 + 1.8 + 6 × 0.5) ≈ 41 GPU-h; with the 2× safety factor 82 h. **This exceeds the ≤ 50 GPU-h line at 2×** — the prereg says so and pre-commits the cut: if Stage 1 runs past 50 h, the `random` arm is dropped first (it is a floor control), then `n` is reduced to 16 for `oja`/`fd` only (their contrast is secondary).
- **Stage 2** (conditional on Stage 1 not selecting Branch C) = Mistral-7B-v0.3 × 16K (6 arms) + all three families × 32K (`isvd`, `frozen`, `nogist` only — the primary contrasts): ≈ 17 + 3 × (7.2 + 4.8 + 3.6) ≈ 64 GPU-h.
- The bf16 gist arm (`isvd_r64_h256_seed_bf16`, Part L5) rides Stage 1 as a 7th arm (+3.6 h/family) so its fp32 reference is the same pod, same prompts.
- Money: at $1.0–1.5/h A100 (vast.ai, on-demand rates seen in W18–W20 harvests), Stage 1 ≈ $45–125 (2×), Stage 2 ≈ $65–100. Credit is $23.85 → **top-up required before Stage 1** (DECISIONS D-003).

**Interfaces:**
- `gate1.load(pod_dirs) -> Gate1Data` (trials + ppl rows from `results/<pod>/{trials,ppl}.jsonl`, arms mapped tracker ← arm name via `configs/arms/*.yaml`).
- `gate1.retrieval_contrasts(data) -> list[Contrast]` — for each family × ctx × task: `isvd` vs `frozen`, `isvd` vs `nogist` (primary; Holm over the primary family = families × tasks × 2) and `isvd` vs `oja`/`fd`/`random` (secondary; Holm over their own family); each `Contrast = {family, ctx, task, a, b, n_paired, a_favored, b_favored, p, p_holm, errors_a, errors_b}`.
- `gate1.ppl_contrasts(data) -> list[PplContrast]` — per family × ctx: Δ bits/token of `isvd` − each other tracker over paired windows, `paired_bootstrap` 95% CI, `tost(delta=0.02)`.
- `gate1.gate1_verdict(retrieval, ppl) -> Verdict` with `Verdict = {branch: "A/B" | "C" | "UNDECIDED", reason: str, families_separated: list[str]}` implementing `ICML2027_PLAN.md` §2 Gate 1 verbatim: **C** if `isvd` is not separated (Holm p ≥ 0.05) from `fd` AND from `frozen` on every task in every family AND every ppl TOST(±0.02) passes; **A/B** if `isvd` beats `frozen` AND `nogist` (Holm p < 0.05, a_favored > b_favored) on retrieval or beats them on perplexity (paired CI excludes 0 in isvd's favour) in ≥ 2 families; **UNDECIDED** otherwise (the memo then says which cells decide). Raises `ValueError("arm X has k error records")` if any primary-contrast arm has errors.

- [ ] **Step 1: Tests** (`tests/test_gate1_table.py`) — construct `trials.jsonl` fixtures in `tmp_path` for two families × 16K × 4 tasks with known outcomes: (a) all trackers identical hits → `gate1_verdict` = `C`; (b) `isvd` 22/24 vs `frozen` 10/24 and `nogist` 8/24 in both families (12 one-way discordant pairs each) → `A/B` with `families_separated == ["llama", "qwen"]`; (c) one family separated only → `UNDECIDED`; (d) an `fd` cell with 3 `error` records → the table prints `FAILED (3 errors)` and `gate1_verdict` raises. Perplexity fixture: 16 windows per arm with Δ = 0.005 bits → TOST passes; Δ = 0.05 → fails.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `gate1.py` (≈ 150 lines on top of `stats.py`), the `tables.py` subcommand (markdown: one block per family × ctx, cells `acc [lo,hi] (hits/n)` with `*` for Holm-significant primary contrasts, `FAILED (k errors)` where applicable; a perplexity block with Δ, CI, TOST verdict; a final line `VERDICT: <branch> — <reason>`), the two pod YAMLs, the ppl task variant. **Step 4:** `make test` → PASS; `scripts/pod.py run --pod gate1_v2_stage1 --dry-run` writes a manifest. **Step 5:** Commit `L3.2: gate1 pods (2 stages, sized), gate1 table with Holm/TOST, gate1_verdict() implementing the branch rule`.
- [ ] **Step 6 (after the owner launches and the watchdog harvests):** `make tables` renders `gate1.md`; run the five-reviewer simulation on the table alone (KICKOFF Part E note) before reading it; append the verdict + evidence path to DECISIONS.md as the Gate-1 outcome; STATE.md entry signed by L6.

---

## Part L5 — bf16 gist storage + pre-registrations

### Task L5.1: `gist_dtype` — bf16 storage of U, C, B with fp32 accumulation

**Files:**
- Modify: `src/kvdlra/cache/bug_cache.py` (`BugStreamingCache(..., gist_dtype: torch.dtype = torch.float32)` → layer attribute; in `_absorb_columns`: `u = self.u_k.to(torch.float32)`, `b = self.b_k.to(torch.float32)`, `c = self.c_k.to(torch.float32)` before the step/carry; after: `self.u_k = u_new.to(self.gist_dtype)`, `self.b_k = b_new.to(self.gist_dtype)`, `self.c_k = torch.cat([rot @ c, u_new.mT @ block_k], 1).to(self.gist_dtype)`; the same for V; `_ensure_mid_cache` upcasts to fp32 for `u @ c` (already fp32 math) — the surprise scores and the L1 guard read fp32 upcasts), `src/kvdlra/accounting.py` (`bug_footprint(..., gist_bits: int = 32)`: when 16, `fp32_verbatim` excludes U and C — they are billed in the fp16 part of `stored_bits()`), `src/kvdlra/eval/frontier.py:_footprint` (passes `gist_bits=16 if layer.gist_dtype == torch.bfloat16 else 32`), `configs/arms/isvd_r64_h256_seed_bf16.yaml` (`cache.gist_dtype: bfloat16`, `doc:` "U, C, B stored bf16; every step and the guard accumulate in fp32; coordinates are re-rounded to bf16 at every basis carry, so rounding compounds with the absorb count — this arm measures exactly that")
- Test: `tests/test_bf16_gist.py`

- [ ] **Step 1: Tests**
```python
import torch

from kvdlra import accounting as acc
from tests.test_bug_cache import make_tiny_model_and_cache


def test_default_gist_dtype_is_fp32_and_bit_identical() -> None:
    _, a = make_tiny_model_and_cache(rank=8)
    _, b = make_tiny_model_and_cache(rank=8, gist_dtype=torch.float32)
    x = torch.randn(3, 64, a.n, generator=torch.Generator().manual_seed(0))
    a.ingest(x.clone()); b.ingest(x.clone())
    assert torch.equal(a.layers[0].c_k, b.layers[0].c_k) and a.layers[0].u_k.dtype == torch.float32


def test_bf16_storage_halves_gist_bits_and_keeps_reconstruction_close() -> None:
    _, f32 = make_tiny_model_and_cache(rank=8)
    _, b16 = make_tiny_model_and_cache(rank=8, gist_dtype=torch.bfloat16)
    x = torch.randn(40, 16, f32.n, generator=torch.Generator().manual_seed(1))   # 40 absorbs: rounding compounds
    f32.ingest(x.clone()); b16.ingest(x.clone())
    l32, l16 = f32.layers[0], b16.layers[0]
    assert l16.u_k.dtype == torch.bfloat16 and l16.c_k.dtype == torch.bfloat16
    k32 = l32.u_k @ l32.c_k
    k16 = l16.u_k.float() @ l16.c_k.float()
    rel = float(torch.linalg.norm(k32 - k16) / torch.linalg.norm(k32))
    assert rel < 5e-2                                                       # the bf16 drift budget on this synthetic stream
    fp32 = acc.bug_footprint(f32.n, rank=8, coord_count=l32._f_len(), recent_len=l32._recent_len(), gist_bits=32)
    fp16 = acc.bug_footprint(f32.n, rank=8, coord_count=l16._f_len(), recent_len=l16._recent_len(), gist_bits=16)
    assert fp16.float_equiv() == fp32.float_equiv()                          # element counts identical (anti-drift pin)
    assert fp16.stored_bits() < fp32.stored_bits()
    gist_words = 2 * f32.n * 8 + 2 * 8 * l32._f_len()
    assert abs((fp32.stored_bits() - fp16.stored_bits()) - 16 * gist_words) < 1e-6


def test_guard_and_surprise_run_in_fp32_under_bf16_storage() -> None:
    _, c = make_tiny_model_and_cache(rank=8, gist_dtype=torch.bfloat16, orth_fix_tol=1e-3, diag_every=1)
    c.ingest(torch.randn(4, 64, c.n, generator=torch.Generator().manual_seed(2)))
    rows = c.drain_diag()
    assert rows and all(r["orth_err_k"] < 1e-2 for r in rows)                 # measured on the fp32 upcast of bf16 U
```
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (store-boundary casts only; nothing inside `kvdlra.tracker` changes). **Step 4:** `make test` → PASS incl. the golden and `tests/test_accounting.py`. **Step 5:** Commit `L5.1: gist_dtype (bf16 storage, fp32 accumulation), billed at 16 bits; default fp32 bit-identical`.
- [ ] **Step 6:** Record in the task report the measured drift curve on the synthetic stream (rel. error vs absorbs at 40/200/1000 absorbs) — the risk the prereg must name: coordinates are re-rounded at every carry.

### Task L5.2: Pre-registrations + provenance verification

**Files:**
- Create: `prereg/gate1_tracker_swap_v2.md`, `prereg/bf16_gist.md`, `prereg/kernel_smoke.md`
- Verify (no change expected): `tests/test_pod_manifest.py::test_check_rejects_tampered_config_hash` exists and passes (L0 Task 5); `scripts/pod.py launch` refuses when the prereg's first commit is not a strict ancestor of HEAD (L0 Task 5) — add `tests/test_prereg_order.py` if L0 did not test it:
```python
import subprocess, sys
from pathlib import Path


def test_launch_refuses_prereg_committed_in_head(tmp_path: Path) -> None:
    # a prereg file that exists only in the working tree (never committed) must block launch
    p = Path("prereg/_tmp_never_committed.md"); p.write_text("STATUS: test\n")
    try:
        r = subprocess.run([sys.executable, "scripts/pod.py", "launch", "--pod", "gate1_v2_stage1",
                            "--offer", "0", "--dry-run", "--prereg", str(p)], capture_output=True, text=True)
        assert r.returncode == 1 and "prereg" in (r.stdout + r.stderr).lower()
    finally:
        p.unlink()
```

Each prereg carries, in this order: purpose; arms; tasks; n and the design; primary contrast(s); family size and correction; decision rule verbatim from the plan; GPU budget with arithmetic and the 2× safety factor, in hours and dollars; provenance (pod name, `manifest.json` fields, the rule that this file's first commit precedes the launch commit and `scripts/pod.py check` verifies it); `STATUS: awaiting owner go (DECISIONS D-00x)`.

- `gate1_tracker_swap_v2.md`: Stage 1 / Stage 2 as sized in Task L3.2; primary contrasts isvd vs frozen and isvd vs nogist on retrieval (exact McNemar paired on `(seed, trial)`, Holm over families × tasks × 2) and perplexity (16 paired windows, bootstrap CI, TOST ±0.02 bits); secondary isvd vs oja/fd/random; the branch rule (`gate1_verdict()`); the pre-committed budget cuts (drop `random`, then n=16 for oja/fd); the five-reviewer simulation on the table before anyone reads it; "no arm reported as `--` where trials errored".
- `bf16_gist.md`: arm `isvd_r64_h256_seed_bf16` rides Stage 1 (Llama + Qwen 16K) and Stage 2 (Mistral 16K; three families 32K) alongside `isvd_r64_h256_seed`; reading: non-inferior within δ = 0.03 retrieval (per task, McNemar; the bf16 arm may not lose > 0.03 on point estimate with Holm-significant separation against it) and 0.02 bits/token perplexity (TOST); if it passes, DECISIONS records that every byte-matched comparison is re-planned at the new bytes (never a silent rerun); named risk: coordinate re-rounding at every carry (the CPU drift curve from Task L5.1 step 6 is quoted).
- `kernel_smoke.md` (for L4): arms full / reconstruct (`isvd_r64_h256_seed` today's path) / kernel (`isvd_r64_h256_seed` with `attention: kernel`); contexts 16K/32K/64K; batch 1 and 4; Llama-3.1-8B; A100-40GB (H100 if available); measured `torch.cuda.max_memory_allocated` KV peak and ms/token (p50 over 64 decode steps after warm-up; the w20 latency script's protocol, now `kvdlra/eval/latency.py`); the Week-3 gate verbatim: kernel KV peak < full KV at 32K AND ms/token < the reconstruct path by ≥ 3×; correctness precondition: single-layer max|Δ| < 1e-2 bf16 vs reconstruct-then-attend and full-model greedy token-exact on ≥ 14/16 prompts (mismatches logged); budget ≈ 4 GPU-h; the numbers to beat: reconstruct 103/188/508 ms and 3.25/6.45/12.83 GB, full 26/28/37 ms and 2.05/4.08/8.14 GB.

- [ ] Steps: write the three files → the SHA-order and tamper tests pass → commit `L5.2: preregs gate1_tracker_swap_v2 / bf16_gist / kernel_smoke; provenance checks verified`.

---

## Self-review

- **L3 brief coverage:** trackers isvd / oja_tuned / fd_fixed / frozen_prefill_svd / no_gist (bytes rebalanced) / random_basis ✓ (L1 + L3.1); 3 families × 16K/32K × 4 tasks × n=24 + ppl 16 windows ✓ (two stages, sized, cuts pre-committed); prereg before launch ✓ (L5.2 + the launch refusal test); ≤ 50 GPU-h ✓ per stage at 1× — at the 2× safety factor Stage 1 exceeds it and the prereg says so with the cut order; primary contrasts + Holm over 12 (families × tasks × 2 at Stage 2; 16 at Stage 1) + TOST ±0.02 ✓; `--` never printed for errored arms ✓; harvest → `make tables` → DECISIONS names the branch ✓ (`gate1_verdict()`).
- **L5 brief coverage:** bf16 gist (C and U — and B — stored bf16, fp32 accumulate in step and guard) on 3 families × 16K/32K × 4 tasks × n=24 + ppl ✓ (rides the Gate-1 pods; reading δ = 0.03 / 0.02 bits); prereg files ✓ (filler_realism, ss2_families, l2_smoke are on the L2 lane; hygiene_table4 on L1; the remaining three here); provenance (manifest at launch/harvest, `--check`, tampered-hash rejection) ✓ delivered by L0 Task 5 and verified here, plus the prereg-order launch refusal test.
- **GATES G3/G5 mapping:** prereg SHA precedes launch SHA; ≤ 50 GPU-h in the manifest (Stage 1 at 1×) ✓ · 6 trackers × families × ctx × tasks × n=24 harvested; 0 arms with error > budget → after launch · Holm + TOST rendered by `make tables` ✓ · DECISIONS names the branch by the rule ✓ · bf16 harvested under prereg → after launch · seven prereg files committed before their pods ✓ (three lanes) · tampered-manifest test ✓.
- **Placeholders:** `H` for the no-gist arm is computed, not guessed (test pins the 5% match); the per-trial minutes come from the L2prep report (`scratchpad/L2prep-report.md`, to be copied under `docs/plan/cleanup/` when L2 merges) — the prereg cites the source.
- **Risk stated, not hidden:** the bf16 arm's per-carry re-rounding may fail non-inferiority at 32K; that is the measurement, and the fp32 arm stays the default until it passes.
