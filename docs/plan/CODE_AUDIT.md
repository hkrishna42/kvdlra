# Code-Audit Addendum: `github.com/hkrishna42/kvdlra` vs. the submitted paper

> **Note (2026-09-11, KICKOFF_WEEKS0-3.md Part E §2):** History is complete (343 commits); the 'squashed' remark in Part C §T4 was a shallow-clone artifact. Week-20 harvest (Sep 9) supersedes the 'no results' remarks in Parts B §Q6/Q8 and C §T8 — see results/w20-close-report.md and docs/plan/lanes/L7_paper_correction.md.

**Scope and independence.** This is a separate pass from the blind PC simulation. Three auditor agents read the repository at commit `7229d04` (2026-09-08) against the paper text (dated 2026-09-06). `docs/reviews/` was deliberately not read. Reviewer scores from the PC simulation are unchanged by this pass; the addendum answers the questions the reviewers could not settle from the PDF. Every claim below carries a file:line citation in the attached reports, and the algebraic/numerical claims were reproduced on CPU with the repo's own classes.

## What the code settles — ranked by consequence

**1. The tracker is fixed-rank block incremental SVD, and the code says so.** `streaming_torch.py:290-293` and `bug_cache.py:425-427` state that `augmented_bug_step` with `theta=None, min_sv_frac=0` — the flagship defaults — "IS fixed-rank incremental SVD (Brand 2006)". The Brand identity was reproduced to 1e-15. No ODE, vector field, time step or K/L/S substep exists on any evaluated path; the actual DLRA integrators (`bug.py`, `bug_adaptive.py`, `bug_class.py`, `bug_torch.py`) are never imported by the tracker, press or cache. The one DLRA ingredient (Frobenius-tail tolerance θ) is off in every reported arm. **The paper's §2.1 concession ("rank-1 reduces to iSVD") understates it: the shipped method is iSVD at every block size.**

**2. §4.1 "beats fixed-rank iSVD everywhere" is a metric artifact.** BUG and the iSVD baseline produce the *same subspace* (‖P_bug − P_isvd‖ ≈ 1e-12) but are scored differently: BUG by re-projecting the full matrix onto the final basis (‖M − UUᵀM‖, using oracle access to M at eval time), iSVD by its stored factors. For identical U the former is ≤ the latter by construction. Worse, the iSVD factor error is exactly the fidelity of what the production cache actually stores (rot-carried coordinates), reproduced to 4 decimals. **So §4.1 reports the wrong number for the deployed gist — the deployed gist is ≈1.03–1.05× oracle, not 1.01–1.03×.** The Oja baseline's tuned step size sits on the grid boundary in both RoPE modes, so "1.3–3× worse" is an upper bound on Oja's gap; Frequent Directions at ℓ=2r beats BUG at every rank in the repo's own `w7-fd-ablation.json`.

**3. The Palu baseline is not Palu.** `palu_press.py` is a per-sequence, per-head truncated SVD of the actual K and V at prefill — no weight decomposition, no grouped heads, no Fisher rank allocation, no fine-tuning; the file's own docstring admits this (`palu_press.py:24-27`). kvpress has no Palu press. The paper's "run through kvpress" wording and the name `palu-r0.5` will be read as a faithful baseline; it must be renamed (e.g. "one-shot per-head SVD oracle") or replaced.

**4. The KIVI baseline is a `transformers.QuantizedCache` mixin, not KIVI.** G=64 (KIVI: 32), chunked-4096 prefill where later chunks attend to 2-bit dequantized context, whole-store re-quantization on residual overflow, residual flushed before decode. Multi-value needles at depths 0.2–0.8 land 3-of-4 in the degraded chunks — the exact cell where the paper claims its win. The single-shot-prefill 2-bit control (`w19.sh sysfix`, `--chunk 0`) that would test this was launched and never harvested.

**5. 8B GPU latency exists in the repo and contradicts the paper.** The paper says the 1B CPU datum (+10%) is the only latency measurement. `results/w7-streamppl-8b*.json` has Llama-3.1-8B GPU decode: full 5.8 ms/tok vs bug-r64 10.6 ms/tok vs bug-r128 12.6 ms/tok — **1.8–2.2× slower**. Attention is over Θ(T) reconstructed tokens; nothing is reduced. The `w20_latency.py` pod (16K–64K, full/flagship/KIVI, peak VRAM) was never harvested.

**6. Omitted competitive results.** (a) ShadowKV post-fix at Llama 16K (1.00/1.00/—/0.00) is in `w15-confirm-lines.txt` and unreported. (b) Plain `ea-k0.25` at 16K scores 1.00/0.88/1.00/0.50 (n=8) — on par with the flagship's 1.00/1.00/1.00/0.58 — and is omitted from the retrieval tables. (c) 4-bit KIVI at 64K = 12/10/12/7, contradicting "4-bit trails only on Mistral variable-tracking".

**7. The "pre-registered confirmatory" marquee is a deterministic re-execution.** The identical n=16 result (15/16 vs 5/16 vs 9/16) appears in Week-17 lines from `w16.sh MARQUEE=1` on the same prompts (deterministic generator + greedy decode). `week18-kickoff.md` already cites the result and asks to "name the marquee comparator" — comparator fixed after the outcome. The repo is one squashed commit, so nothing is dateable. The correct word is "replicated deterministically", not "confirmatory".

**8. The divergence mechanism is loss of orthonormality in U, not "rank siphoning".** Reproduced through the production `BugStreamingLayer`: near-null directions admitted just above the fp32 tolerance carry O(δ) overlap with span(U); re-orthogonalization is Gram–Schmidt against an already-non-orthonormal U; ‖UᵀU−I‖ ratchets 6e-4 → 3.7e-3 → **35** (σ_max(U)=5.6) by ~36K tokens. **There is no orthogonality check or periodic QR of U anywhere in the cache.** The 27,531-perplexity cell (`w17-qwen-lines.txt:33`) is `bug-r256`, a pure-gist arm with *no exact tier*, so the paper's tier-siphoning story cannot explain it. The relative SV floor works by discarding the tail that seeds the ratchet — but it is relative to σ_max, collapsed tracked rank 256 → 1 in a Qwen-like synthetic, and the accounting bills configured rank not effective rank (`w10_frontier.py:546-548`), so "footprint-neutral" is unverified.

**9. Statistics: every number matches; the design is thinner than described.** All of Tables 1, 2, 3, 6, 7, 8 and every headline McNemar p reproduce exactly from the per-trial records; pairing is genuine. But: one fixed 10-sentence haystack cycled for every trial (needle depth is a deterministic function of trial index — no depth sweep exists in any headline cell); design effect ≈2 in several cells, which pushes the 12/12 lower bound from 0.76 to ~0.61 and makes the marquee's "Wilson-disjoint" claim non-robust; perplexity windows are WikiText-103 **train** (paper: "held-out"); the "0.05 bits/token band" is a point-estimate tie whose paired CI reaches +0.072, and the flagship is significantly worse than full KV (+0.076 [0.031, 0.122]); bf16 NLL floor is 0.011–0.023 bits/token, not 0.006; no per-window NLLs exist for any §4.6 comparison; under Holm over the paper's own ~49 quoted contrasts, only one eviction blow-out survives; in the §4.7 WikiText-filler comparison the flagship ran single-needle only while baselines ran all four tasks.

**10. Memory accounting.** U, C fp32 and tiers fp16 are all counted (good), but the "measured 0.98× workspace" is the identity (T−292)/T, per-step `torch.cat` transients are excluded, and full-KV decode peak was never measured — the 1.06× residency is analytic. No test shows an fp16 gist fails; `test_bug_torch.py:111-125` calls bf16 storage + fp32 core "recommended".

## What this means for the resubmission

The good news: the per-trial data is real, the statistics are correctly computed, the pairing is honest, and the repo already contains the two experiments that decide the paper's fate — the **tracker swap** (`swap` pod: flagship cache with Oja / Frequent Directions in place of iSVD; the kickoff even pre-states "if Oja/FD match on retrieval and ppl, 'DLRA' is branding and the paper says so") and the **sysfix/ss2** pod (measured decode ms/token + peak VRAM, and single-shot-prefill KIVI-2 on the same needles). Neither has results. **Run those two pods before anything else**; they determine whether this is a method paper or an analysis paper, and whether the multi-value edge is real or a prefill-protocol artifact.

Then, in order: rename/replace the Palu baseline; report the 8B GPU latency; add a periodic QR / orthogonality tripwire on U and re-run the Table 4 cells with effective rank billed; fix §4.1 to score both methods on stored factors; report ShadowKV, ea-k0.25 and 4-bit-at-64K; add a depth sweep and ≥2 haystacks; call the marquee a replication; and fix the "held-out" WikiText wording.

Full auditor reports follow: A (tracker / numerics), B (baselines / accounting / protocol), C (statistics from per-trial records).

---



# Part A — Tracker and numerical audit

# CODE_A — Tracker / numerical-linear-algebra audit of `kvdlra`

Scope: `src/kvdlra/integrators/*`, `src/kvdlra/lowrank.py`, `src/kvdlra/cache/bug_cache.py`, the §4.1 scripts/results, and the flagship arm construction. Independent of `docs/reviews/` (not read). All paths are relative to the repo root `scratchpad/kvdlra/`. "Confirmed" = read in code and/or reproduced numerically here (CPU, torch 2.14, numpy 2.4); "inferred" = reasoned, not executed.

Repo tests run here: `tests/test_streaming_torch.py test_w17_rankfloor.py test_bug_cache.py test_bug_cache_week15.py test_streaming.py test_bug_cache_seed_regression.py test_bug_cache_week11.py` → 73 passed, 1 strict xfail (see Q7). `tests/test_w20_tracker_swap.py` fails only on a missing `datasets` module in this sandbox.

---

## Executive summary

1. **The "streaming BUG tracker" is fixed-rank block incremental SVD (Brand 2006), exactly.** Confirmed in code and numerically: the per-block update is `[U|Q] · svd([[B, A],[0, R]])`, truncated to `rank_cap`. There is no vector field, no time step, no K/L/S substep; the ODE-based BUG integrators in `bug.py`, `bug_adaptive.py`, `bug_class.py`, `bug_torch.py` are never imported by the tracker, press, or cache. The code itself says so: `streaming_torch.py:290-293` ("`augmented_bug_step` with `theta=None, min_sv_frac=0` (the flagship's defaults) IS fixed-rank incremental SVD (Brand 2006)") and `bug_cache.py:425-427`. The only "DLRA" ingredient — the Frobenius-tail tolerance θ — is **off** in every reported configuration (theta=None); the `--min-sv-frac` floor is a plain relative-σ cutoff.

2. **"Beats fixed-rank iSVD everywhere" (§4.1) is a metric artifact, not an algorithmic win.** The BUG tracker and the `lowrank.incremental_svd_recon` baseline produce the *same* subspace (‖P_bug − P_isvd‖₂ ≈ 1e-12 in fp64) but are scored differently: BUG by re-projecting the **full matrix** onto the final basis (‖M − UUᵀM‖, oracle access to M at eval time), iSVD by its **stored truncated factors** (‖M − USVᵀ‖). Same U ⇒ BUG's projection error equals iSVD's projection error to 1e-16. Worse for the paper: the iSVD factor error is *exactly* the error of what the production cache actually stores (rot-carried coordinates = chain of projections); reproduced to 4 decimals. So §4.1's BUG numbers overstate the cache's own gist fidelity by the same 1.7–4.5% that the paper credits as "beating iSVD".

3. **§4.4's divergence (ppl 27,531) has a concrete mechanism that the authors never localized** (their own test docstring: "NOT reproduced here… GPU-gated"). I reproduced it on CPU through the production `BugStreamingLayer` with a synthetic Qwen-like ill-conditioned stream: an **exponential loss-of-orthonormality ratchet in U**. Near-null residual directions admitted just above the fp32 tolerance carry O(δ) overlap with span(U); since the "re-orthogonalization" is Gram–Schmidt against a U that is itself no longer orthonormal, the defect compounds. In the layer: ‖UᵀU−I‖ grows 6e-4 → 3.7e-3 over 32K tokens, then jumps to **35** (σ_max(U)=5.6) at ~36K; reconstructed key norms then range 0.63–2.7× truth. There is **no orthogonality check or periodic QR of U anywhere in the cache**. The floor works because it discards the tail that seeds the ratchet — but it is *relative to σ_max*, so on a stream with a massive-activation channel it collapsed the tracked rank to **1** in my test while the accounting still bills the configured rank (`w10_frontier.py:546-548` uses `arm["rank"]`, not the live rank). Also: the 27,531 cell is `bug-r256` — a **pure-gist arm with no exact tier** (`results/w17-qwen-lines.txt:33`), so the paper's "rank siphoning by the exact tier" cannot be the cause of that specific number.

4. **§4.1 data exists** (`figures/week2/oja_vs_bug.json`, `figures/week2/pilot.json`, `results/w7-fd-ablation.json`; Llama-3.2-1B layer-8 K, 5 C4 docs, 4092 tokens after dropping 4 sinks, fp64, per-token b=1). Tables reproduced below. Oja's "fair" grid tuning landed on the **grid boundary** in both rope modes (eta0=1.0/min or 20/max; decay=0.03/max), contradicting the script's claim that the grid brackets the optimum. Frequent Directions at ℓ=2r **beats** BUG at every rank (0.986–0.995×) and ties the oracle to 3 decimals; at matched memory FD ties BUG at r=128. The repo has **no KV dumps** (`dumps/` empty), so no real-KV numbers could be re-run; synthetic runs are reported.

---

## Q1. Algebra of the streaming update

### What is stored
`streaming_torch.augmented_bug_step` (`src/kvdlra/integrators/streaming_torch.py:82-204`), called from the cache at `bug_cache.py:902-917`:

- `u` — (n, r) fp32 basis (`bug_cache.py:587`, "pre-RoPE key basis"), one for K and one for V.
- `b_core` — (r, r) fp32 **diagonal** `diag(sigma)` (line 202: `b_new = torch.diag(sigma[:keep])`); tests pin diagonality (`tests/test_bug_cache.py:343-363`). No V factor is stored anywhere.
- Per-token coordinates `c_k`, `c_v` — (r, T) fp32 (`bug_cache.py:589`).

### The per-block step (lines 165-204), verbatim

```python
r_old = u.shape[1]
a = u.mT @ block                    # (r, b)  coordinates
r_perp = block - u @ a              # (n, b)  residual
a2 = u.mT @ r_perp                  # one re-orthogonalization pass (Parlett/Kahan)
r_perp = r_perp - u @ a2
a = a + a2
q, s, vh = torch.linalg.svd(r_perp, full_matrices=False)     # rank-revealing residual
tol = 100.0 * torch.finfo(block.dtype).eps * torch.linalg.norm(block)
m = int((s > tol).sum().item()); m = min(m, max(0, n - r_old))
q = q[:, :m]
u_aug = torch.cat([u, q], dim=1)                              # (n, r+m)
top = torch.cat([b_core, a], dim=1)                           # (r, r+b)
bot = torch.cat([zeros(m, r_old), s[:m].unsqueeze(1) * vh[:m]], dim=1)   # (m, r+b)
b_fac = torch.cat([top, bot], dim=0)                          # (r+m, r+b)
u_loc, sigma, _ = torch.linalg.svd(b_fac, full_matrices=False)
keep = min(rank_cap, sigma.shape[0])            # theta (Frobenius tail) and min_sv_frac gates follow, both OFF by default
u_new = u_aug @ u_loc[:, :keep]
b_new = torch.diag(sigma[:keep])
rot = u_loc[:r_old, :keep].mT                    # = u_newᵀ u_old
```

Seeding (`u is None`, lines 159-164): reduced QR of the first block, then the same SVD/truncation — i.e. the exact truncated SVD of block 1.

### Is it Brand-2006 block incremental SVD on the column space?
**Yes, exactly** (confirmed). With `B_aug = [[B, A],[0, R]]` (here `R = s·vh`, the SVD form of the residual's R-factor), `B_aug B_augᵀ = [U|Q]ᵀ (U B Bᵀ Uᵀ + C Cᵀ) [U|Q]`, i.e. the Gram matrix of the *stored* model plus the new block in the augmented basis; the SVD + truncation is the Eckart–Young projection of that. Numerically (fp64, `audit_q1q2.py` §1): seed identity error 2.2e-15; step identity ‖U₂B₂B₂ᵀU₂ᵀ − P₂(UBBᵀUᵀ + CCᵀ)P₂‖ = 3.7e-15; `rot == u₂ᵀu` to 1.4e-15; ‖u₂ᵀu₂ − I‖ = 1.1e-14. Note the model is the projected Gram of the **stored** state, not of the full data — the full-data projected Gram already differs by 1.8% after one step (the truncation loss), which is standard for iSVD.

The numpy per-token version `streaming.StreamingBUG.update` (`streaming.py:201-263`) is the same thing with b=1: append `[a; ‖resid‖]` to `B`, SVD, keep top-r. It differs from the torch step only in an absolute residual gate `_RESID_EPS = 1e-12` (line 95; not relative to ‖k‖) and no re-orthogonalization.

### Is there ANY genuine BUG/DLRA step?
**No** (confirmed). In `streaming.py`/`streaming_torch.py` there is no vector field F, no step size h, no K-step, L-step or Galerkin S-step; the docstring (`streaming.py:29-51`) reinterprets the update as "the augmented BUG step specialized to the relaxation field F(Y)=Y_target−Y with the substep ODEs integrated exactly (their Galerkin steady state)" — a post-hoc relabelling of a projection. The real integrators exist (`bug.py:130-204` `bug_step` with RK2 K/L/S substeps; `bug_adaptive.rank_adaptive_bug_step`; `bug_torch.bug_step_torch`; `bug_class.BUG`) but are used **only** by synthetic tests (`tests/test_bug_synthetic.py`, `test_bug_class.py`, `test_bug_torch.py`) and `scripts/sigma_decay.py`. The only symbol the tracker imports from the DLRA modules is `truncation_rank` (`streaming.py:89`), the Frobenius-tail rule — and the torch/cache path re-implements it as `_truncation_rank` (`streaming_torch.py:67-79`), which only executes when `theta is not None`; every reported arm passes `theta=None` (`bug_cache.py:321,1810`; `w10_frontier.py` never sets it). The paper's "σ_min-independent error bound" (§2.1) is a property of the ODE integrators and has no referent in the executed code (inferred).

### Rank-1 vs block
The block case is not a different algorithm: `blocked_bug_subspace(block_size=1)` is validated for parity with the numpy per-token tracker (`tests/test_streaming_torch.py`). Block size changes the *truncation schedule* only (coarser blocks truncate less often, hence closer to the oracle): on my synthetic fp32 run, r=64: b=1 → 0.3415, b=16 → 0.3417, b=64 → 0.3408, b=256 → 0.3401, b=T → 0.3379 (= oracle). Note the flagship uses **two** block sizes in one run: 128 for the first chunk (`prefill_block_size` default, `bug_cache.py:324,1537`, not CLI-exposed) and `absorb_block=16` thereafter (`w10_ruler.py:579`).

---

## Q2. The iSVD baseline vs. BUG-with-adaptivity-off

Baseline: `lowrank.incremental_svd_recon` (`src/kvdlra/lowrank.py:31-66`); used by `scripts/week2_pilot.py:113` and `scripts/w7_fd_ablation.py`. Comparison (both fp64, b=1, truncate-after-augment, no re-orth, seeded from column 1):

| aspect | `StreamingBUG` (numpy) | `incremental_svd_recon` |
|---|---|---|
| precision | fp64 | fp64 |
| block | 1 | 1 |
| augmentation gate | skip if ‖resid‖ ≤ 1e-12 (abs) | always; `resid/(resid_norm+1e-12)` (non-unit junk column when residual ~0; harmless once truncated) |
| truncation | after SVD of augmented core, keep r | same (`if s.size > r`) |
| init | random rank-1 basis, overwritten by data | first column's SVD |
| re-orth of U | none | none |
| **metric** | `reconstruction_error`: ‖M − U Uᵀ M‖_F/‖M‖_F using the **full M at eval time** (`streaming.py:304-329`) | ‖M − (u·s)·vt‖_F/‖M‖_F from the **stored truncated factors** (`lowrank.py:65-66`) |

**Algebraically identical subspaces; the gap is 100% the metric** (confirmed, `audit_q1q2.py` §2, n=512, T=2048):

| r | oracle | BUG (proj, paper metric) | iSVD (factor USVᵀ) | iSVD subspace scored by projection | ‖P_bug − P_isvd‖₂ |
|---|---|---|---|---|---|
| 16 | 0.7603 | 0.7682 | 0.7761 | **0.7682** | 9.5e-13 |
| 32 | 0.5788 | 0.5856 | 0.5958 | **0.5856** | 9.7e-13 |
| 64 | 0.3379 | 0.3415 | 0.3474 | **0.3415** | 1.6e-11 |

Why the projection metric is always ≤ the factor metric: for any S,V, ‖M − USVᵀ‖ ≥ ‖M − UUᵀM‖ (UUᵀM is the best approximation with column space in span(U)). So "BUG beats iSVD everywhere" is guaranteed by construction for identical U.

**Which metric describes the cache?** The cache does *not* re-project the full history; old coordinates are carried by `c ← rot @ c` with `rot = U_newᵀU_old` (`bug_cache.py:921-922`), i.e. old token t is reconstructed as `P_T P_{T-1} ⋯ P_{t} k_t` — precisely the stored-factor model of incremental SVD. Reproduced (`audit_q2b.py`): cache-style rot-carried error = iSVD factor error to 4 decimals at r=16/32/64 (0.7713/0.5926/0.3428) vs the paper-metric BUG number 0.7616/0.5811/0.3341. The docs call this "erosion by repeated projection" (`bug_cache.py:50-52`). Hence the §4.1 "1.01–1.03× of oracle" figure is not the fidelity of the deployed gist; the deployed gist has the iSVD figure (≈1.03–1.05× oracle on the archived data, table in Q5).

---

## Q3. Coordinate consistency in `bug_cache.py`

**(a) Rotated.** Confirmed: `bug_cache.py:919-922`
```python
if self.c_k is not None:
    self.c_k = rot_k @ self.c_k
    self.c_v = rot_v @ self.c_v
```
with `rot = u_loc[:r_old, :keep].mT` = `u_newᵀ u_old` exactly because Q ⟂ U (`streaming_torch.py:203`; pinned by `tests/test_bug_cache.py:83-93`). Rank changes are handled by `rot` being (keep, r_old). New block coordinates are computed in the **post-step** basis: `new_ck = self.u_k.mT @ block_k` (line 933). Nothing is recomputed from retained full vectors (none are kept for the gist); no rotation chain is stored — the projection is applied eagerly each absorb. Consequence: `‖rot@c‖ ≤ ‖c‖` (rot is a sub-block of an orthogonal matrix) → coordinates can only shrink, *if U is orthonormal* (see Q4 for when it isn't). The quantized tier is carried by dequantize→rot→requantize (`_rotate_quant_tier`, lines 1260-1274); the CodeBUG tier is frozen in an anchor basis (lines 1293-1344; not in flagship).

**Reconstruction path at attention time** (`_decode_peek` → `_ensure_mid_cache`, lines 1569-1648): `k_pre_hat = u_k @ c_k` (fp32), `_unwhiten_key` (identity unless Q-BUG), re-apply RoPE at true positions (`_mat_rope_at`/`_mat_rope`, fp32), cast to `self.dtype` (**bf16** in the pod runs, `DTYPE=bfloat16` in `scripts/pod/w18_boot.sh:33`), concatenate `[sinks | hh exact | quant | fp32 gist | recent]`. Values: `u_v @ c_v` → bf16. The middle cache is invalidated on every absorb (`_absorb_block_into_stream`, lines 773-774) and rebuilt lazily.

**Per-step costs** (per layer; n=H·D, r, block b=16, f=#gist columns; flagship keeps f≈T since `coord_budget = t + rw + ab`, `w10_frontier.py:290-291`):
- every decode token: ring concat O(n); attention over sinks + f + hh + ring (the attended length is **not** reduced — Θ(T) tokens are attended);
- every b tokens (absorb): un-RoPE O(n b); two `augmented_bug_step`s: O(n r b) matmuls, SVD of r_perp O(n b²), SVD of b_fac O((r+b)³), `u_aug @ u_loc` O(n r (r+b)); rot-carry `rot @ C` O(r² f) for K and V; rebuild `U @ C` O(n r f) for K and V plus RoPE re-application O(n f); under SurpriseSLASH also un-RoPE + score the whole candidate pool (hh+b) O(n r (hh+b)).
- amortized per token with f≈T: O(n r T / b) — grows linearly with context (e.g. 512·64·32768/16 ≈ 67 MFLOP/token/layer for K alone). The docstring's "bounded, independent of generated length" (lines 44-46) holds only when `coord_budget` is finite, which the evaluated configs deliberately are not.

**Orthogonality checks / re-orthogonalization of U:** **none** in `bug_cache.py` (the word "orthonormal" appears only in docstrings; no `UᵀU` computation, no periodic QR). The only re-orth is of the *residual* against U inside the step (`streaming_torch.py:174-176`), which presumes U is orthonormal. Tests check orthonormality only for single steps in fp64 (`tests/test_bug_cache.py:90-93,110-117`).

**dtypes (confirmed):** bf16 (model dtype) — `sink_k/v`, `recent_k/v`, `hh_k/v`, the returned `_mid_k_cache/_mid_v_cache`; fp32 — `u_k/u_v`, `b_k/b_v`, `c_k/c_v`, `mid_surprise`, scores, RoPE cos/sin (`_RopeAngles._cos_sin_for`, line 214), all step math (`_mat_rope_with` upcasts, line 696; `block_v = grad_v.to(float32)`, line 786). Note the tracked "pre-RoPE keys" are obtained by un-rotating **bf16-stored post-RoPE** keys in fp32 (line 785), so the fp32 core ingests bf16 rounding (≈2⁻⁹ relative) of the post-RoPE key, mixed across the RoPE pairs; "all core linear algebra in fp32" is true, but the data entering it is bf16-quantized.

---

## Q4. Divergence mechanism (ppl 27,531)

### What the floor gates
`--min-sv-frac` → `BugStreamingCache(min_sv_frac)` → `BugStreamingLayer.min_sv_frac` → `augmented_bug_step(..., min_sv_frac=...)` for **both K and V** (`bug_cache.py:902-917`). Implementation (`streaming_torch.py:194-200`): after the `rank_cap`/θ truncation, `keep = max(1, min(keep, (sigma > min_sv_frac * sigma[0]).sum()))` — a relative cutoff on the singular values of the **augmented stored model** `b_fac` (accumulated history + new block; not "the block's numerical rank" as the docstring says, though the effect is similar). Default 0.0 = bit-identical off (tests pin this). It does not gate surprise scoring, coordinates, or reconstruction.

### Candidate causes, checked
- **Division by tiny singular values / whitening / B inverse:** none on the flagship path (confirmed). `_dequantize` and `_surprise_scores` divide by norms clamped at 1e-12; `w_key` inverse only for `bugSQ` arms; B is never inverted.
- **Stale coordinates + rotated basis:** coordinates are eagerly rotated (Q3); with orthonormal U this can only shrink norms — not a cause.
- **bf16 leakage:** the fp32 core ingests bf16-rounded keys, which pads the basis with genuine noise directions at ~2⁻⁹ relative energy; harmless for norms by itself (my bf16-only runs stayed at ‖UᵀU−I‖ ≤ 6e-3 over 16K tokens).
- **Rank-adaptive padding pulling in noise directions → loss of orthonormality of U (the actual mechanism, confirmed on synthetic data through the production layer).** With `theta=None` the step always fills to `rank_cap`. Residual directions with `s_j` just above `tol = 100·eps_fp32·‖block‖_F` (≈1.2e-5·‖block‖) are admitted; their overlap with span(U) is ≈ eps·‖r_perp‖/s_j (up to 1e-2). Once ‖UᵀU−I‖ = δ > 0, `block − U(Uᵀblock)` is no longer a projection, the newly admitted Q carries O(δ) overlap, `u_loc` mixes it into U, and δ grows geometrically. Instrumented (`audit_q4c.py`, raw `augmented_bug_step`, n=512, cap=256, b=16, rank-40 signal + 4 outlier channels ×1e3 + 1e-2 noise, bf16-rounded): ‖uᵀq‖ (overlap of admitted directions) 7.8e-5 (step 400) → 8.6e-4 (1200) → 0.41 (1394) → 0.88 (1396); ‖UᵀU−I‖ 5e-2 → 0.24 → 1.27 in three consecutive steps, then 36. `u_loc` itself stayed orthonormal (1.6e-5) throughout — the SVD is not the culprit; the augmented frame is.
  Through the **production** `BugStreamingLayer` (`audit_q4d.py`: bf16 storage, RoPE round trip, `_prefill` 4096 in 128-blocks + `consolidate()` in 16-blocks, rank=256, floor off): ‖UᵀU−I‖ = 6.0e-4 (8K) → 1.5e-3 (16K) → 3.7e-3 (32K) → **35.1, σ_max(U)=5.56 (36K)** → 16.0, σ_max=1.51 (40K). Final gist: reconstructed pre-RoPE key column norms 0.63× (median) to **2.73×** (max) of truth, relative reconstruction error 0.45 — attention then sees keys with wrong norms and directions, which is consistent with a four-digit perplexity. With `min_sv_frac=1e-2` the ratchet never starts (‖UᵀU−I‖ ≤ 4e-6). Onset time depends on the spectrum; real Qwen KV (massive key-bias channels, n=512, r=256 = n/2) evidently reached it before 16K. **Inferred**, not confirmed on real KV (the repo has no dumps).
- **Exact-tier "rank siphoning":** cannot explain the headline cell. `results/w17-qwen-lines.txt:33` — `bug-r256 [T=16384] ppl=27531.666` — is the `bug` arm (`w10_frontier.py:217-240`: `retention="fifo"`, no `hh_budget`). Only the `bugSseed-r128-h1024` "puzzle" cells involve the exact tier. Siphoning may lower the effective rank and thus increase padding, i.e. it can accelerate the mechanism above, but it is not a separate cause.

### Two caveats on the floor
1. **It is relative to σ_max of the accumulated model.** On a stream with one dominant channel (Qwen-style key bias / massive activations), everything below 1% of that channel is dropped: in my Qwen-like synthetic, `min_sv_frac=1e-2` collapsed the tracked rank from 256 to **1** (all 40 signal directions gone; `audit_q4b.py`, `audit_q4d.py`). Whether real Qwen/Mistral spectra sit in that regime is unknown from the repo.
2. **Reported memory is not measured for floor-on arms.** `w10_frontier._footprint` (`scripts/w10_frontier.py:546-548`) bills `rank=int(arm["rank"])` (configured), not `layer.u_k.shape[1]` (live). Hence `bug-r256` and `bug-r256-f0.01` show identical `tok_eq/layer=8466.5` (`results/w17-qwen-lines.txt:33,36`). The paper's "footprint-neutral" and the "monotone improvement with rank" under the floor are therefore stated at configured rank while the effective rank is unknown (could be far lower); the live `stored_state_numel()` would have shown it.

Other observations: the authors' own gate test says the blow-up "is NOT reproduced here" (`tests/test_w17_rankfloor.py:7-9`), i.e. the fix was validated empirically on GPU, not mechanistically. No divergence tripwire exists in the cache (no NaN/norm/orthogonality guard).

---

## Q5. §4.1 data

**Location and provenance.** `scripts/week2_oja_vs_bug.py` → `figures/week2/oja_vs_bug.{json,png,pdf}`; `scripts/week2_pilot.py` → `figures/week2/pilot.{json,png}` (iSVD); `scripts/w7_fd_ablation.py` → `results/w7-fd-ablation.json` (FD + iSVD). Data: `unsloth/Llama-3.2-1B-Instruct`, **layer 8 keys only** (no values), 5 C4 documents ≥4096 tokens (`figures/week2/pilot_docs.json`: doc idx 63/411/454/637/718), `len4096` dumps, first 4 sink tokens dropped → M is 512 × 4092 (8 KV heads × 64, **one shared basis per layer**, not per head), fp64, per-token streaming (b=1). Metric: ‖M − UUᵀM‖_F/‖M‖_F with the full M (Q2). Ranks 16/32/64/128 (Oja/FD) and budget ranks 26/51/102/205 (iSVD pilot). **`dumps/` is empty in the repo**, so none of this can be re-run on real KV here.

**Oja config** (`week2_oja_vs_bug.py:68-86,110-133`; `oja.py`): per-token L2 normalization (tracks the direction-only second moment, not MMᵀ), `eta_t = eta0/(1+decay·t)`, single pass, random Gaussian orthonormal init (seed 0), fixed rank from token 0, grid eta0∈{1,2,5,10,20} × decay∈{1e-3,3e-3,1e-2,3e-2} tuned **at r=16 on the evaluation data** and reused at all ranks. Chosen: post-RoPE (1.0, 0.03), pre-RoPE (20.0, 0.03) — **eta0 at the grid minimum/maximum and decay at the grid maximum in both modes**, i.e. the optimum was not bracketed, contrary to lines 72-76 ("the grid brackets the single-pass optimum").

**Reproduced tables from stored JSON (mean over 5 docs).**

Oja vs BUG (`figures/week2/oja_vs_bug.json`):

| rope | r | oracle | BUG | Oja | BUG/oracle | Oja/BUG |
|---|---|---|---|---|---|---|
| post | 16 | 0.4901 | 0.4979 | 0.7019 | 1.016 | 1.41 |
| post | 32 | 0.3849 | 0.3963 | 0.6712 | 1.030 | 1.69 |
| post | 64 | 0.2594 | 0.2626 | 0.6271 | 1.012 | 2.39 |
| post | 128 | 0.1791 | 0.1819 | 0.5502 | 1.015 | 3.03 |
| pre | 16 | 0.2996 | 0.3017 | 0.4010 | 1.007 | 1.33 |
| pre | 32 | 0.2290 | 0.2313 | 0.3451 | 1.010 | 1.49 |
| pre | 64 | 0.1567 | 0.1581 | 0.2945 | 1.009 | 1.86 |
| pre | 128 | 0.0980 | 0.0994 | 0.2377 | 1.014 | 2.39 |

→ matches "1.01–1.03× of truncated SVD" and "1.3–3.0× vs Oja"; "roughly halves the error pre-RoPE" is ~0.55–0.6×.

iSVD pilot (`figures/week2/pilot.json`, budgets → ranks):

| rope | r | oracle | iSVD | BUG | BUG/oracle | iSVD/BUG |
|---|---|---|---|---|---|---|
| post | 26 | 0.4205 | 0.4453 | 0.4310 | 1.025 | 1.033 |
| post | 51 | 0.2953 | 0.3135 | 0.3003 | 1.017 | 1.044 |
| post | 102 | 0.2035 | 0.2126 | 0.2060 | 1.012 | 1.032 |
| post | 205 | 0.1280 | 0.1371 | 0.1312 | 1.025 | 1.045 |
| pre | 26 | 0.2495 | 0.2565 | 0.2521 | 1.010 | 1.017 |
| pre | 51 | 0.1800 | 0.1848 | 0.1817 | 1.009 | 1.017 |
| pre | 102 | 0.1159 | 0.1196 | 0.1172 | 1.012 | 1.020 |
| pre | 205 | 0.0630 | 0.0660 | 0.0641 | 1.017 | 1.030 |

→ the entire 1.7–4.5% "win" is the metric difference of Q2 (same subspace). Per Q2, the iSVD column is the fidelity of what the cache stores.

FD ablation (`results/w7-fd-ablation.json`, pre-RoPE):

| r | ℓ_mm | oracle | BUG | FD (matched mem, ℓ=r+r²/n) | FD (ℓ=2r) | iSVD | FD(2r)/BUG |
|---|---|---|---|---|---|---|---|
| 16 | 17 | 0.2996 | 0.3017 | 0.3223 | **0.3003** | 0.3058 | 0.995 |
| 32 | 34 | 0.2290 | 0.2313 | 0.2442 | **0.2292** | 0.2351 | 0.991 |
| 64 | 72 | 0.1567 | 0.1581 | 0.1629 | **0.1567** | 0.1608 | 0.991 |
| 128 | 160 | 0.0980 | 0.0994 | **0.0991** | **0.0980** | 0.1017 | 0.986 |

→ FD at ℓ=2r is at the oracle and strictly better than BUG at every rank; at r=128 FD ties/beats BUG even at matched memory. Not mentioned in §4.1. `figures/week5/integrator_ablation.json` (fp64, b=128): PSI (projector-splitting) is within 0.1–0.5% of BUG; parallel-BUG ~20% worse.

**Synthetic re-run here** (`audit_q5.py`, repo classes, n=512, T=2048, fp64, decaying spectrum, week-2 Oja tuning protocol → eta0=5, decay=0.03):

| r | oracle | BUG | iSVD | Oja | FD(mm) | FD(2r) | BUG/orc | iSVD/BUG | Oja/BUG |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.7588 | 0.7668 | 0.7751 | 0.7666 | 0.8337 | 0.7648 | 1.010 | 1.011 | 1.000 |
| 32 | 0.5765 | 0.5852 | 0.5967 | 0.5877 | 0.6185 | 0.5766 | 1.015 | 1.020 | 1.004 |
| 64 | 0.3362 | 0.3394 | 0.3456 | 0.3935 | 0.3418 | 0.3362 | 1.009 | 1.018 | 1.160 |
| 128 | 0.1153 | 0.1166 | 0.1192 | 0.2801 | 0.1153 | 0.1153 | 1.012 | 1.022 | 2.402 |

→ Oja ties BUG at low rank and lags at high rank under a single pass with a schedule tuned at r=16 — the Oja gap is a convergence/tuning artifact that grows with r, consistent with the archived 1.3→3× trend.

---

## Q6. Surprise scoring, score-rank cap, warm-up seed

**Surprise** (`_surprise_scores`, `bug_cache.py:968-994`): `sqrt(‖k‖² − ‖uᵀk‖²)_+ / max(‖k‖,1e-12)` — the relative out-of-subspace residual (sin of the angle to span(u)), computed on **keys only** (pre-RoPE, un-rotated at true positions, whitened under Q-BUG), against the **pre-update** `self.u_k` (strictly older history; `_absorb_columns` computes it at line 889 before the step at 902). Returns 1.0 for every column when `u_k is None`. It relies on Pythagoras, i.e. on U being orthonormal (fails under the Q4 ratchet: with ‖UᵀU−I‖ ≫ 0 the clamp at 0 masks negative "residuals"). Values are never scored.

Two uses: (i) SurpriseSLASH selection (`_absorb_block_slash`, lines 835-865): the whole candidate pool = current exact tier + graduating block is **re-scored every absorb**; top-`hh_budget` stay verbatim (post-RoPE K, raw V), the rest are demoted into the gist. `hh_neighbor=1` (flagship) applies a ±1-position max ("span boost", lines 996-1023). (ii) `retention="lowrank_surprise"` stores a per-column snapshot (fp32 scalar, counted in memory) used only when `coord_budget` overflows — never in the evaluated configs.

**Tie order**: comment says "ties fall back to recency" (line 849) but `torch.argsort(sel, stable=True, descending=True)` keeps equal scores in **ascending index = oldest first** (verified: `[0,1,2,3,4,5]`). This matters for the seed (below), where all early scores are exactly 1.0.

**Score-rank `-s` (`score_rank`)**: `_surprise_scores(…, cap=self.score_rank)` (line 844) scores against `u_k[:, :min(cap, r)]` — the leading columns, which are energy-ordered because `b = diag(sigma)` descending (true for the BUG step; not for `tracker="oja"`, whose QR-updated columns are unordered — a latent mismatch if `-s` were combined with `--tracker oja`). Applied **only** at the SLASH selection site; the retention snapshot is uncapped (line 843). Storage rank and accounting unchanged. Validation requires `hh_select="surprise"`, `hh_budget ≥ 1`, `1 ≤ s ≤ rank` (lines 408-420). Flagship `bugSseed-r64-h256` does **not** use it; `s32` appears only in the r128-h1024 marquee (`scripts/pod/w18.sh:88,103`).

**Warm-up seed** (`seed_hh_warmup`, `_prefill` lines 1534-1552): fires only when `hh_enabled and _mode == "ingest"` (i.e. `--chunk > 0`; `w10_frontier.py:207-212` fails loud otherwise). The first chunk's middle (4096 − 4 sinks − 32 recent = 4060 tokens) is routed through `_absorb_block_slash` in **sub-blocks of `prefill_block_size`=128** with `grad_score=None`, so yes — the same selection/demotion path as steady state. Quirk (confirmed by reading; inferred behaviour): sub-block 1 has `u_k=None` → all 128 tokens score 1.0 → all kept (keep_n = min(256,128)); sub-block 2 likewise (256 kept, nothing absorbed, still no basis); sub-block 3: 384 candidates all at 1.0 → the **oldest 256** stay (stable tie order), the newest 128 are demoted and seed the basis. So the exact tier initially holds the first 256 middle tokens, and the first basis is built from tokens 256–383; these early tokens are only demoted once a later token out-scores their re-computed surprise. The paper's "scored against the strictly-older, needle-free basis" is accurate from sub-block 4 onward; the first three sub-blocks have no basis at all.

---

## Q7. Other things a reviewer should know

**Flagship config as actually run (`bugSseed-r64-h256`, `scripts/pod/w18.sh:29`, `w19.sh:67`, arm built at `w10_frontier.py:243-330`):** `rank=64`, `hh_budget=256`, `hh_select="surprise"`, `hh_neighbor=1`, `seed_hh_warmup=True`, `retention="lowrank_surprise"`, `coord_budget = T + recent_window + absorb_block` (**no coordinate eviction; full Θ(T) gist**), `n_sink=4` (`N_SINK`), `recent_window=32` and `absorb_block=16` (CLI defaults `w10_ruler.py:578-579`, `w10_frontier.py:884-885` — the class defaults are 64/32, `bug_cache.py:318-319,1807-1808`), `theta=None`, `min_sv_frac=0.0` (floor **off** in the flagship), `score_rank=None`, `prefill_block_size=128` (class default; not CLI-exposed), `--chunk 4096`, `--dtype bfloat16` on the pod (CLI default is float32), `tracker="bug"`. The paper's §2.3 does not state W=32, block sizes 128/16, or `hh_neighbor=1`.

**Strict xfail / known bug**: `tests/test_bug_cache_week15.py:304-336` — `seed_scores` indexes absolute positions into a chunk-length seed; under `retention="attn"` + `attach()` + chunked ingest the ring seed desyncs and `seed[self.mid_pos]` raises IndexError. Does not affect the flagship (`lowrank_surprise` retention, no `attach`).

**Comments admitting hazards**: `streaming_torch.py:139-151` (Week-7 fix: plain QR of a numerically null residual "silently breaks the basis"; reruns are "fp-equivalent, not bit-identical" to archived Week-3..6 results); `bug_cache.py:1212-1217` (exponential norm drift in the quant tier before renormalization was added); `tests/test_w17_rankfloor.py:7-9` (divergence not reproduced on CPU).

**Accounting nuances**: B is billed as r floats (diagonal) (`bug_cache.py:1703-1710`); the frontier footprint uses the configured rank (Q4 caveat); "constant memory" in the module docstring vs Θ(T) coordinate storage in the evaluated arms (paper §2.3 acknowledges the latter). Attention compute is over the full retained length (no FLOP savings).

**Numerics**: fp32 core with no orthogonality guard, no periodic re-orthonormalization of U, no divergence tripwire (Q4). `tol = 100·eps·‖block‖_F` uses the block's Frobenius norm, so a single outlier channel sets the admission threshold for every direction. `_RESID_EPS=1e-12` in the numpy tracker is absolute. The residual SVD `torch.linalg.svd(r_perp)` on (n, b) is rank-revealing but fp32; on GPU cuSOLVER's Jacobi path may return less orthonormal factors than LAPACK on ill-conditioned input (inferred, not tested).

**Terminology vs code**: module docstrings (`streaming.py:29-51`, `streaming_torch.py:16-19`) describe the update as "the same augmented BUG integrator" while other comments in the same files and in `bug_cache.py:425-427`, `w10_ruler.py:617-621` state it "IS fixed-rank incremental SVD". The paper's §2.1 uses the former framing; §2.1's "Relation to incremental SVD" concedes the rank-1 case only. The block case is also identical (Q1).

**Baseline framing**: `lowrank.py:31-36` describes iSVD as "a locally greedy streaming baseline — it never revisits a column once absorbed — i.e. the 'naive streaming' curve the BUG tracker should beat"; the BUG tracker never revisits a column either. The docstring also calls truncated SVD "the best any rank-r projector can do", which is the *projection* oracle — the one BUG is scored against but the cache does not achieve (Q2).

---

## Artifacts
Audit scripts (scratchpad): `audit_q1q2.py` (Gram identity, BUG-vs-iSVD subspace/metric), `audit_q2b.py` (rot-carry == iSVD factor error), `audit_q4.py`/`audit_q4b.py`/`audit_q4c.py` (orthogonality ratchet, raw step), `audit_q4d.py` (ratchet through the production `BugStreamingLayer`), `audit_q5.py` (BUG/iSVD/Oja/FD synthetic table).


---

# Part B — Baselines, accounting and protocol audit

# CODE audit B — baselines, accounting, latency, run protocol (kvdlra)

Repo root: `scratchpad/kvdlra` (paths below are relative to it). Paper: `scratchpad/paper.txt`.
Legend: **[C]** = confirmed by reading code / committed results; **[I]** = inferred (my reasoning, not directly asserted by code or data). Reference-implementation facts fetched from upstream sources are marked **[ref]**.

---

## Q1. KIVI baseline (`src/kvdlra/quant/kivi_cache.py`, `scripts/w10_frontier.py`)

**What it is [C].** A thin wrapper over `transformers==5.8.0` `QuantizedCache` (quanto or hqq backend), not a port of the KIVI repo.
- Factory `make_quant_cache(config, nbits, scheme, backend, group=64, residual=128)` — `kivi_cache.py:111-152`. Two schemes: `"token"` (transformers default axes, per-token groups for K **and** V — the Week-18 arms that scored 0.00) and `"kivi"` (per-channel keys, per-token values) — `kivi_cache.py:130-135`.
- Per-channel keys are realised by quanto `axis=-1`, which groups `g` consecutive *rows* of the flattened `(B*H*T)` axis per head-dim channel (verified against `optimum/quanto/tensor/grouped.py::group`: `base.reshape((axis_groups, group_size, axis_dim))` **[ref]**). `T` is edge-padded to a multiple of 64 so groups never straddle heads (`kivi_cache.py:70-78, 80-86`). hqq path transposes to `(B,H,D,T)` and groups along `T` (`kivi_cache.py:106-108`).
- Values: quanto `axis=0` → 64 consecutive channels of one token (2 groups per 128-d token) `kivi_cache.py:132`.
- **Group size G=64, residual R=128** (defaults in `w10_ruler.py:570-571`, `w10_frontier.py:849-852`; pod scripts never override them: `scripts/pod/w19.sh:28,51`).
- **Scale/zero precision**: `aux_words()` reads the actual `_scale`/`_shift` element sizes off layer 0 after prefill (`kivi_cache.py:161-172`). On the bf16 pod runs (`DTYPE=bfloat16`, `scripts/pod/w18_boot.sh:33`) that is 16+16 bits per group = 1 fp32 word → billed in `acc.quant_footprint(..., scale_words=aux_words(cache))` (`w10_frontier.py:581-589`, `accounting.py:420-447`). Asymptote (2 + 32/64)/16 = 0.156×, (4+0.5)/16 = 0.281× — matches the paper's numbers. Note the `quant_footprint` docstring still says "fp32 pairs → 0.1875×/0.3125×" (`accounting.py:433-436`), a stale comment; the measured `aux_words` path is what is used.
- Codes billed at `nbits` (no packing overhead), the 128-token residual billed as fp16 verbatim K+V (`accounting.py:441-447`).

**Prefill handling [C].** `_prefill_plain` (`w10_frontier.py:145-164`) runs the prompt in `--chunk` blocks (4096 on every pod: `w18_boot.sh:32`) with `logits_to_keep=1`, then `flush(cache)` folds the *entire* fp16 residual into the quantized store (`kivi_cache.py:88-98, 155-158`). The quant arm is marked `chunkable: True` (`w10_frontier.py:516`), so in `w10_ruler.retrieve` it gets `arm_chunk = 4096` (`w10_ruler.py:405, 334-344`). Consequences, traced through transformers 5.8 `QuantizedLayer.update` **[ref]**:
  1. chunk 1 (tokens 0–4095) is quantized wholesale on the first `update` (attention for that chunk uses the fp16 `key_states`);
  2. chunk 2 attends to **dequantized 2-bit** chunk 1 and is appended to the fp16 residual;
  3. chunk 3 triggers the residual-overflow branch (`keys.shape[-2]+1 >= 128`): transformers **dequantizes the whole store, concatenates, and re-quantizes everything** (chunk 1 is now double-quantized);
  4. chunk 4 → residual; `flush()` dequantizes+re-quantizes all again (chunk 1 three times, chunks 2–3 twice).
  Re-quantizing a dequantized affine grid with identical 64-token group boundaries (4096 % 64 == 0) is near-idempotent up to bf16 rounding of scale/shift **[I]**, so (3)/(4) are minor; (2) is not minor — see below.
- After `flush()` decode begins with **zero** fp16 residual; the query (~48 tokens) and ≤40 generated tokens then accumulate in fp16 (never reaching 128), so no further re-quantization during decode **[C]**.
- The footprint nevertheless bills a full 128-token fp16 residual (`_footprint` passes `residual_length=arm["quant_residual"]`, `w10_frontier.py:582-589`) that has just been flushed — a ~0.007× over-charge to the quantizer at 16K (0.163 vs 0.156), i.e. conservative *against* the baseline **[C]**.

**Deviations from KIVI reference [C/ref].**
| item | KIVI (jy-yuan/KIVI) | this repo |
|---|---|---|
| key grouping | per-channel, G=32 (`example.py`: `group_size = 32`) **[ref]** | per-channel, G=64 |
| value grouping | per-token, G=32 | per-token, G=64 |
| residual | `residual_length` 128 in paper/LongBench scripts (32 in `example.py`) **[ref]** | 128, but **flushed to 0 after prefill** |
| prefill | full-precision attention over the whole prompt, then quantize all but the last `T mod R` tokens **[ref, recalled from modeling_llama_kivi.py; not re-fetched]** | chunked 4096: later chunks attend to 2-bit dequantized earlier chunks; whole store re-quantized on overflow |
| kernel | fused 2-bit packed matmul, no dequant of the store | transformers dequantizes the whole store every step |
| zero-point | fp16 min/scale per group | bf16 `_shift`/`_scale` per group (float shift, no int zero-point) |

Paper §4.6 discloses G=64 and the streaming prefill (`paper.txt:515-522`), but says "a 128-token fp16 residual" (the residual is flushed) and does not mention the re-quantization cascade.

**Would this plausibly hurt 2-bit multi-value (the claimed win)? [I]** Yes, plausibly, and it is unmeasured:
- Multi-value plants 4 codes at depths 0.2/0.4/0.6/0.8 (`w10_ruler.py:223-232`). At 16K with 4096-token chunks these land in chunks 1,2,3,4; three of the four needles' hidden states (hence their own K/V) are computed from attention over already-2-bit context, an error KIVI's fp16 prefill never incurs. The error compounds across layers within a chunk. Single-needle (mid-depth, chunk 2/3) is also affected but is an easier task; multi-value is the task requiring all four to survive, so it is the most exposed cell.
- G=64 vs 32 halves the number of per-channel scales; the needle key shares its 64-token group's min/max with 63 near-identical filler tokens (the cyclic 10-sentence filler makes filler keys tightly clustered), so an outlier needle channel is coarsely quantized. Real KIVI (G=32) would be ~2× finer here.
- The flagship's own protocol also uses chunked ingest, but its exact tier stores needles verbatim, so it is structurally immune to the compounding that hits the quantizer.
- The authors planned the decisive control — `sysfix` mode runs `quant-2bit-kivi` with `--chunk 0` at 16K, n=12 (`scripts/pod/w19.sh:225-227`, "is the chunked-prefill mv edge protocol-bound?") — and a pod `sysfix-llama` was launched (`results/w19_harvest/pods.txt:17`) but it is **absent from `results/w19_harvest/done.txt` and no `ss2` rows exist anywhere under `results/`** **[C]**. The paper's §4.6 multi-value claim therefore rests on a 2-bit arm whose prefill protocol differs from KIVI's in a direction that specifically penalises it, with the single-shot control never harvested.

---

## Q2. Palu baseline (`src/kvdlra/press/palu_press.py`)

**What it implements [C].** `PaluPress(BUGPress)` with `rank_ratio=0.5, group=1` (`palu_press.py:48-63`). At the end of single-shot prefill (`BUGPress.forward_hook`/`compress`, `bug_press.py:306-389`) it takes the layer's **pre-RoPE keys recomputed from hidden states** (`get_prerope_key_states`, `bug_press.py:376-382`) and its values, and for each KV head builds the `(head_dim, T)` matrix, keeps the first `n_sink=4` columns exact, and replaces columns `4:` by their **truncated-SVD (Eckart–Young) rank-r reconstruction computed on the sequence's own K/V** in fp32 (`palu_press.py:40-45, 65-92`). Keys are re-rotated to post-RoPE and written back same-shape into the DynamicCache.
- `r = round(0.5 * 128) = 64` per head, applied to **both K and V** (`compress_values=True` inherited default, `bug_press.py:138, 386-387`).
- `group=1` → per-head decomposition (`--palu-group 1` default, `w10_frontier.py:840`; pod scripts never set it).
- No rank allocation (uniform r across layers/heads), no Fisher information, no fine-tuning, no calibration set, no weight decomposition, no quantization of latents **[C]**.
- Single-shot prefill only (`chunkable: False`, `w10_frontier.py:469`; `BUGPress.compress` raises on `q_len != cache_len`, `bug_press.py:367-374`). Decoded tokens are appended uncompressed (kvpress presses act only at prefill).

**Faithfulness [C].** The module docstring says so itself: "real Palu low-rank-decomposes the projection *weights* offline … This post-hoc *activation*-SVD is the Eckart–Young upper bound on that scheme" (`palu_press.py:24-27`). So this is **not Palu**: it is a per-sequence, per-head, data-dependent oracle SVD — an upper bound on any *fixed* projection's reconstruction, but also something that (a) cannot be computed online/causally (needs the full prefill K/V), (b) is only Eckart–Young-optimal in Frobenius norm on the sequence, not in attention output. Real Palu (Chang et al.): SVD of `W_k`/`W_v` (grouped-head G-LRD, groups of several heads sharing one latent), Fisher-information rank allocation across layers, latents computed online as `x·A`, optional fine-tuning **[ref, from the paper; not fetched]**. Palu's "r0.5" is a 50% KV-cache compression rate that in their grouped variants spans several heads — here it is strictly per-head.

**Attention-sink handling [C].** The sink carve-out (first 4 token columns kept exact for both K and V) is a repo-specific fix (`palu_press.py:68-77`, "Week-15 audit fix"). The paper's "Palu attention-sink low-ranking that had inflated its perplexity from 9.236 to 7.232" (`paper.txt:412-414`) is therefore a defect of the authors' *own oracle approximation* (per-sequence SVD dominated by high-norm sink columns), not of Palu — real Palu's weight-SVD basis is not fitted to any one sequence's sinks. The fix also gives this baseline an exemption the streaming flagship enjoys (BUG keeps `n_sink=4` exact too), which is fair.

**Bytes [C].** `acc.palu_footprint(t, n, head_dim, h_kv, 0.5, group=1)` (`accounting.py:313-343`) counts: 4 sink tokens K+V exact, per-token latent `2*(t-4)*r*n_groups`, and a per-group basis `2*r*head_dim*group*n_groups`. Ratio ≈ 0.50 + O(1/T) → 0.502/0.504 in the logs (`results/w15-confirm-lines.txt:18-23`). Real Palu stores no per-sequence basis (it lives in the weights), so the port is billed ~0.2% *more* than Palu would be — negligible. The same-shape DynamicCache is not measured; the footprint is analytic (`w10_frontier.py:609-614`).

Paper mismatch: §4 Setup says "think-c0.5, palu-r0.5 and Expected Attention are run through NVIDIA's kvpress" (`paper.txt:293-294`). kvpress has no Palu; `PaluPress` is in-repo and merely subclasses kvpress `BasePress` **[C]**.

---

## Q3. ThinK baseline (think-c0.5)

**[C]** `ThinKPress(key_channel_compression_ratio=cr)` straight from `kvpress==0.5.1` (`w10_frontier.py:441-455`; pin `pyproject.toml:21`). Standalone — **not** composed with SnapKV/H2O as in most ThinK-paper tables; no `ComposedPress` anywhere in the repo (grep). Keys only; values untouched; kvpress v0.5.1 `think_press.py` scores channels with the last `window_size=32` re-computed RoPE'd queries × key norms and **zeros** (does not remove) the lowest 50% key channels **[ref]**. Single-shot prefill (`chunkable: False`, `w10_frontier.py:452`); decode-time keys are not pruned (prefill-only hook, `compat.py:52-57`).
- Bytes: analytic `acc.think_footprint` = pruned K (64 of 128 channels) + full V + per-head kept-channel indices → 0.75× (`accounting.py:293-307`, `w10_frontier.py:605-608`). The zeroed DynamicCache is not measured (docstring notes kvpress realises no memory gain).
- Fair as "standalone ThinK λ=0.5"; the ThinK paper's headline operating points (ThinK+SnapKV at ≤0.5×) are not represented, so "beats think-c0.5 at 1.8–2.6× less stored state" compares to the weakest-compression ThinK configuration.

---

## Q4. ShadowKV

**Implementation [C].** `src/kvdlra/cache/shadow_cache.py` is a from-scratch port: sinks + recent verbatim, per-KV-head SVD of pre-RoPE middle keys at `rank_s`, mean-key landmarks per 8-token chunk, **full V offloaded to CPU**, top-k chunk selection from a pre-attention forward hook (`shadow_cache.py:1-95`). Documented deviations: no outlier-chunk tier, chunk-aligned middle, aggregated window selection, verbatim generated tokens, no chunked prefill (`shadow_cache.py:56-77`). Config used: `--shadow-ranks 64 128 --shadow-topk 256` (`w10_frontier.py:474-496`, `scripts/pod/w11_baselines.sh:22`), i.e. rank 64/128 (ShadowKV default 160), sparse budget 256×8 = 2048 tokens, no outliers.

**Results exist but are not in the paper [C].**
- Week-11 (pre-fix): `shadow-r64/r128` 0.00 on all four tasks at 16K and 32K (`results/w11-goalA-ruler-lines.txt:14-78`, `results/w11-base-ruler-lines.txt:6-46`), later voided (`results/w11-final-tables.md:26,53`; `results/w15-ruler-intervals.md:5`).
- The "decode-scope bug" fix: `attach()` originally covered only prefill, so the selection hook never ran at decode and the most-recent chunks were selected (`w10_ruler.py:320-325`).
- Post-fix re-measure (Llama-3.1-8B, 16K, n=8): `shadow-r64` single **1.00**, multi-key **1.00**, vt **0.00** at honest ratio **0.815×** (`results/w15-confirm-lines.txt:15-17`); 16K ppl 4.108 (r64) / 4.076 (r128) vs full 4.076 (`results/w11-base-ppl16-lines.txt:6-7`, `results/w11-table-ppl-lines.txt:13`) — i.e. ShadowKV is essentially lossless on fluency.
- Why excluded **[I]**: under the repo's device-agnostic accounting the CPU-offloaded V counts fully (`accounting.py:349-397`), so ShadowKV lands at 0.81–1.06× "stored state" — off the compression frontier the paper argues about; and no multi-value/32K re-measure was ever run after the fix. The paper mentions the bug only as a fixed defect (`paper.txt:412-413`) without reporting the arm. Note that ShadowKV's actual GPU-resident ratio is the `gpu_ratio_fp16` (~0.3×) which the code computes (`accounting.py:116-122`) but the paper never reports.

---

## Q5. Eviction baselines

**[C]** `ExpectedAttentionPress(compression_ratio=1-keep)` and `SnapKVPress` from kvpress 0.5.1 (`w10_frontier.py:375-399`), single-shot prefill (SnapKV asserts `q_len > window_size`, `w10_ruler.py:358-367`), footprint = measured kept fraction (`w10_frontier.py:615-618`). MorphKV (`src/kvdlra/cache/morph_cache.py`) also exists as a decode-time eviction arm (`morph-k{0.1,0.25,0.5}`) and StreamingLLM (`sllm-w*`) in Week-5/7 scripts.
- Budgets run at 7–8B: `ea-k0.1` everywhere (1475 result rows), `ea-k0.25` (887 rows, mostly the composite `ea-k0.25-q{2,4}` arms; plain `ea-k0.25` at 16K/32K on Llama in Week-11/18), `ea-k0.5`, `snapkv-k0.1/0.25/0.5`, `morph-k*` (Week-11, Llama, n=8/2).
- **No run at k≈0.15** (the flagship's honest bytes) anywhere in `results/` **[C]**.
- The plain `ea-k0.25` (0.250×) Llama 16K rows the paper omits: single 1.00 / mk 0.88 / mv 1.00 / vt 0.50 (n=8) (`results/w11-goalA-ruler-lines.txt:6,27,48,69`), vs flagship 1.00/1.00/1.00/0.58 at 0.151×. `ea-k0.5`: 1.00/1.00/1.00/0.88. The paper reports eviction only at 0.100× (Table 6) plus the LongBench single-document point, so the eviction curve between 0.1× and the flagship's 0.15× is unmeasured and the 0.25× point is unreported in the retrieval tables **[C]**.

---

## Q6. OjaKV

**[C]** No end-to-end OjaKV baseline. The Week-20 tracker-swap (`--tracker oja`, `w10_frontier.py:200-201, 616-623`; `bug_cache.py:890-897`; `streaming_torch.py:307-347`) would have run Oja's rule inside the BUG cache (same sinks/ring/surprise tier), i.e. a *tracker* ablation, not OjaKV's hybrid cache. Pod `swap-llama` was launched (`results/w19_harvest/pods.txt:16`) but is not in `done.txt` and no `-oja`/`-fd` rows exist in `results/` (grep empty). The cache-level `oja_step` uses defaults `eta0=1.0, decay=1e-3` (`streaming_torch.py:314-315`) while claiming "the validated Week-2 schedule" — the Week-2 tuned pre-RoPE optimum was `eta0=20, decay=0.03` (`figures/week2/oja_vs_bug.json`), so the swap arm as written is mis-tuned **[C]**.

**Reconstruction-only comparison (§4.1's "1.3–3.0×") [C].** `scripts/week2_oja_vs_bug.py` on Llama-3.2-1B, layer 8, 5 docs, `(512, T-4)` matrices, single pass, final basis projected retrospectively onto the whole matrix (`oja.py:284-299`, `streaming.py:304-329`). `OjaTracker`: per-token L2 normalisation (tracks the direction-only second moment, not the Frobenius objective the oracle optimises — acknowledged `oja.py:46-56`), Robbins–Monro `eta_t = eta0/(1+decay·t)`, tuned on `ETA0_GRID=[1,2,5,10,20] × DECAY_GRID=[1e-3..3e-2]` at rank 16 only, reused at all ranks (`week2_oja_vs_bug.py:77-86`). **The selected schedule sits on the grid boundary in both modes** (post-RoPE: eta0=1.0 = min, decay=0.03 = max; pre-RoPE: eta0=20 = max, decay=0.03 = max; `figures/week2/oja_vs_bug.json`), so the grid did not bracket Oja's optimum and the "1.3–3×" gap is an upper bound on Oja's true deficit **[I]**. Mean errors: post-RoPE oja/bug = 1.41 (r16) … 3.02 (r128); pre-RoPE 1.33 … 2.40.

---

## Q7. Memory accounting (`src/kvdlra/accounting.py`)

**Units [C].**
- `float_equiv()` = verbatim elems ×1 + ceil(code_bits/32) + aux words (`accounting.py:70-75`) — pinned equal to the live `stored_state_numel()` (`tests/test_accounting.py:92`, `bug_cache.py:1682-1748`).
- `ratio_fp16` ("float-equivalent", Tables 1–2): `bits(16)/(2·T·n·16)` — every verbatim element billed at 16 bits **including BUG's fp32 U and C**, codes native, aux at 32 (`accounting.py:77-80, 97-103`).
- `ratio_stored_bits` ("honest / fp32-at-rest", Tables 3, 5, 7, 8): `fp32_verbatim_elems` (U and C) at 32 bits, everything else as above (`accounting.py:82-95, 105-109`). For ThinK/Palu/eviction/quant `fp32_verbatim_elems=0` so the two ratios coincide (`accounting.py:286-300` test).

**What BUG counts (`bug_footprint`, `accounting.py:133-198`) [C].** Sinks `2n·4` (fp16), recent ring `2n·recent_len` (fp16), coordinates `2·r·coord_count` (fp32), exact tier `2n·hh_count` (fp16), basis U `2·n·r` (fp32), **core B billed as its `r` diagonal per stream (2r words), not r²** — justified by "the square-root core is provably diagonal" (`bug_cache.py:1703-1710`; live tensor is `(r, r)`, `bug_cache.py:588`), positions/surprise per column (1 word each; `mid_pos`/`hh_pos` are **int64** in the live cache but billed as 32-bit words, `bug_cache.py:607, 620`, `accounting.py:179-185` — a ~0.1% under-bill), `hh_pos` 1 word, quant tier codes at `nbits` + 2 fp32 norms/column. Not counted: the PolarQuant rotation matrix for the q4 arm (counted only at cache level via `_QuantBank`, `bug_cache.py:110-113`; `acc.bug_footprint` has no term) — negligible at r=64.
- Quant arms: scale+zero counted at measured width (Q1); codebooks: CodeBUG product-quantizer codebook counted once per cache (`bug_cache.py:1715`), not used in any paper arm.

**Workspace 0.98× — computed from live tensor numel, not memory [C].** `w16_storage._measure` runs one decode step then reads `cache.workspace_numel()` = numel of `_mid_k_cache`/`_mid_v_cache` (`w16_storage.py:108-110`, `bug_cache.py:1750-1754`) divided by `2·T·n·L`. Since the cached middle is exactly the non-sink/non-ring/non-hh tokens, this is `(T-4-32-256)/T` = 0.982 at 16K and 0.991 at 32K — an arithmetic identity, not a measurement of resident memory **[I]**. It also excludes the per-step `torch.cat` of `[sinks|hh|mid|recent]` and of `[k_ret, key_states]` (`bug_cache.py:1585-1591, 1473-1474`), i.e. at least two further full-length transients per step.

**Decode residency [C].** `torch.cuda.max_memory_allocated` is used (`accounting.measure_peak_gpu`, `accounting.py:453-466`; `w16_storage.py:95,111`) but the ratio printed divides by KV bytes and includes model weights (the 5–9× the paper footnotes). The full-KV arm's peak is never measured (`w16_storage.py:82-93` returns constants). The paper's "≈1.06× residency" is stored + workspace summed analytically (`paper.txt:776-778` admits this). `scripts/w20_latency.py` was written to measure the contrast (docstring lines 1-24) — never harvested (see Q8).

**Gist precision [C].** `u_k`, `b_k`, `c_k` are fp32 (`bug_cache.py:587-589`) and billed fp32 in `ratio_stored_bits`. There is **no test or comment showing fp16/bf16 *storage* fails**: `tests/test_bug_torch.py:111-125` `test_bug_torch_bfloat16_fp32core` calls bf16 storage + fp32 core "the recommended mixed-precision mode" and passes with ~6e-2 rel-Fro drift on a synthetic trajectory; only bf16 *core math* (QR) is shown to be worse (`test_bug_torch.py:128-155`). So fp32 storage is a conservative choice, not a demonstrated necessity; the paper's "fp16-storable gist not yet validated" (`paper.txt:718-719`) is accurate but the code offers no evidence either way on real KV.

---

## Q8. Latency / TTFT / throughput at 7–8B

**Paper** cites one datum: 1B CPU, `bug-r128` 224.9 vs full 204.5 ms/token, "our only end-to-end latency datum" (`paper.txt:778-780`). **This is not true of the repository [C].** 8B GPU per-token decode latency exists in committed results (Llama-3.1-8B, bf16, RTX 6000 Ada per `scripts/pod/w7_streamppl_8b.sh:12`; prefill 1024 + 8192 generated tokens, bounded-budget streaming setting):
- `results/w7-streamppl-8b.json`: full **5.8–5.9 ms/tok**, `bug-r128` **12.6–12.7** (2.2× slower), `bugA-r128` 19.0, morph 13.0, snapkvD 9.4.
- `results/w7-streamppl-8b-tier2.json`: full 5.8, `bug-r64` **10.6** (1.8×), `bugA-r64` 15.9.
- `results/w5-streamppl-8b.json` (earlier card): full 14.7–14.9, `bug-r128` 20.5 (1.4×).
- `docs/week7.md:322-335` tabulates these ("bugA … the slowest bounded method").
The CPU 1B figures in `results/w5-streamppl-1b.json`/`w7-streamppl-1b.json` show bug-r128 at 234–264 vs full 190–225 ms/tok (+11–39%), so the paper's "≈10%" is the most favourable reading **[I]**.
- No TTFT/prefill-time or throughput measurement exists. No latency at the 16K/32K prefill-compressor operating point: `scripts/w20_latency.py` (per-token ms + resident/peak VRAM for full/flagship/KIVI-2bit at 16K/32K/64K) was scheduled in `w19.sh:219-224` (`sysfix`), pod launched (`pods.txt:17`), **no results file or `[latency` rows committed**.
- The persistence cold-start timings (§4.9 Fig. 3) come from `scripts/w19_persist.py` (a3 pods, `results/w19_pertrial/a3-llama*-trials.txt`) — those are load/H2D/reconstruct timings, not decode latency.

---

## Q9. Run protocol

**Flagship command line [C]** (`scripts/pod/w18.sh:20-21, 29, 41-43`; identical wrappers in `w19.sh:22-23`):
```
PYTHONPATH=src python -u scripts/w10_ruler.py --model $MODEL --device cuda --dtype bfloat16 --chunk 4096 \
  --context-lens 16384 --tasks niah_single niah_multikey niah_multivalue vt \
  --methods bugslash --ranks 64 --hh-budgets 256 --hh-neighbor 1 --warmup-seed --n-trials 6 --seeds 0 1
```
→ arm `bugSseed-r64-h256` = `BugStreamingCache(rank=64, coord_budget=T+48, recent_window=32, absorb_block=16, n_sink=4, retention="lowrank_surprise", hh_budget=256, hh_select="surprise", hh_neighbor=1, seed_hh_warmup=True)` (`w10_frontier.py:242-324`). Marquee: `--ranks 128 --hh-budgets 1024 --hh-neighbor 1 --warmup-seed --score-rank 32 --n-trials 8 --seeds 0 1` at 32K (`w18.sh:87-90`). KIVI arm: `--methods quant --quant-scheme kivi --quant-nbits 2 4` (`w19.sh:28,51`). Baselines: `--methods think palu --think-ratios 0.5 --palu-ranks 0.5` (`w18.sh:91-93`), `--methods ea snapkv --evict-keeps 0.1` (`w18.sh:74-76`). Composite: `--methods composite --evict-keeps 0.25 0.1 --quant-nbits 2 4 --quant-scheme kivi` (`w19.sh:170`).

**Prefill protocol per arm [C]** (`w10_ruler.py:317-371`): streaming arms (bug/morph/shadow) chunked ingest 4096 under `attach()`; quant chunked 4096 + flush; presses (think/palu/ea/snapkv) and full **single-shot**; composite single-shot. Query (question + generation header, template-derived, ≥48 tokens) is fed after compression at true positions — block for presses/quant, one token per forward for streaming caches (`w10_ruler.py:254-294`); greedy decode `max_new = 12` if only `niah_single` else `40` (`w10_ruler.py:390`). Hit = every target string is a substring of the decoded text (`w10_ruler.py:372-373`).

**In-house RULER generator [C]** (`w10_ruler.py:73-248`):
- Filler: default `"cycle"` = the **10 fixed sentences** in `scripts/w4_needle.py:46-57` repeated until ≥ctx tokens (`w10_ruler.py:96-111`). `--filler wikitext` shuffles a WikiText-2 *test* sentence pool per (seed, trial) (`perplexity_sweep.py:147-187`).
- RNG: `torch.Generator().manual_seed(seed*131 + trial)`; codes are 5-digit ints from `10000 + randperm(89999)` (`w10_ruler.py:170-171, 192`).
- `niah_single`: "The secret passcode is {code}." at sentence index `n//2 + trial%5` (always ≈mid-depth) unless `--depths` (`w10_ruler.py:196-209`).
- `niah_multikey`: `n_keys=8` labels (`w5_ruler._LABELS`: amber, bronze, …), one code each at depths (k+1)/9 (+jitter k%3); queried key = `trial % 8` → only keys 0–5 are ever queried at n_trials=6 (`w10_ruler.py:211-221`).
- `niah_multivalue`: `n_values=4` codes of one label at depths 1/5…4/5; question "List every {label} code mentioned"; hit needs all 4 (`w10_ruler.py:223-232`).
- `vt`: `n_hops=3` → 4 statements `VAR X{trial}{i} = …` at depths 1/5…4/5; ask the last variable (`w10_ruler.py:234-246`). (Official RULER vt uses 4 hops / more chains.)
- n: `--n-trials 6 --seeds 0 1` → 12 trials per cell; marquee 8×2 = 16. Per-trial `[trial]` lines are printed (`w10_ruler.py:454-458`).

**Perplexity protocol [C]** (`scripts/w10_frontier.py`, pod wrappers `w18.sh:22-25`): corpus `--corpus wikitext-103` default = **WikiText-103 raw *train* split**, streamed and concatenated to ~3M tokens (`perplexity_sweep.py:97-103, 114-133`; docstring: "absolute ppl is not a held-out benchmark number"). Samples are the first `n_samples` consecutive non-overlapping blocks of `T+512` tokens (`w10_frontier.py:641-646`); `--window 512`, `--n-samples 4` (PPL4) or 8 (PPL). Each arm prefills `T` with compression active, then scores the 512-token continuation **in one block forward** attending to the compressed cache (511 scored tokens; `_score_window`, `w10_frontier.py:58-77`); ppl = exp(Σ nll / Σ tok). NLL summation was bf16 in archived runs; fixed to fp32 in `_nll_sum` (`w10_frontier.py:71-77`) after the runs (paper discloses, `paper.txt:395-397`).

**Mismatches with the paper's description [C unless noted].**
1. "perplexity on held-out windows" (`paper.txt:291`) — the windows are WikiText-103 **train**.
2. "a 128-token fp16 residual" for KIVI (`paper.txt:516-517`) — flushed to zero after prefill; only re-grows during the ≤90 decoded tokens.
3. "palu-r0.5 … run through NVIDIA's kvpress" (`paper.txt:293-294`) — in-repo oracle SVD press, not a kvpress or Palu implementation (Q2).
4. §4.7 realistic-filler comparison: the flagship ran `--filler wikitext --depths 0.1 0.3 0.5 0.7 0.9` on `niah_single` only (`w18.sh:61-63`), whereas the baselines ran `--filler wikitext` on all four tasks **without `--depths`** (mid-depth needle) (`w18.sh:64-67`). The flagship's 1.00 single-needle vs baselines' numbers are therefore not the same needle layout (depth grid vs mid-depth; harder for the flagship if anything) **[C]**, and the flagship has no wikitext-filler multi-key/multi-value/vt numbers at all, so "the flagship holds … while palu loses variable-tracking (0.08)" compares tasks the flagship did not run under that filler.
5. Table 3 pairs the marquee (n=16) with think/palu at n=16 but KIVI-4 at n=12 from a different pod/commit (a1 vs g4) — labelled "same bytes" without the n difference.
6. The 2-bit single-shot-prefill control (`ss2`) and the 8B latency/residency measurement (`w20_latency`) were pre-registered in `w19.sh` (sysfix) and never harvested.

---

## Q10. Other findings

- **"DLRA integrator" vs incremental SVD [C].** `streaming_torch.py:291-293`: "`augmented_bug_step` with `theta=None, min_sv_frac=0` (the flagship's defaults) IS fixed-rank incremental SVD (Brand 2006)". `w10_frontier.py:619-620` repeats it. §4.1 claims BUG "beats fixed-rank incremental SVD everywhere" — the flagship arm at its shipped defaults is, by the authors' own comment, that method; the §4.1 comparator must be a different incremental-SVD variant (Week-2 pilot), and the paper does not say so.
- **Hard-coded protocol asymmetries [C].** Streaming arms decode the query one token per forward while presses/quant decode it as a block (`w10_ruler.py:331-333, 342-344`) — same positions, but the streaming cache absorbs the query tokens into its ring/gist as it goes (they are within the 32+16 ring for ≤48 tokens, so likely neutral **[I]**).
- **Per-trial exception swallowing [C].** Any exception in a trial is logged as `SKIP` and the trial is dropped from both numerator and denominator (`w10_ruler.py:436-446`), so a cell's `n` can silently fall below 12 (the Week-18 quanto "group 64 must divide 65588" skips were exactly this). Reported `n=` in the row line makes it auditable (`w10_ruler.py:459-482`).
- **Strict xfail documenting a latent bug [C].** `tests/test_bug_cache_week15.py:304-340`: `retention="attn"` + `attach()` + chunked ingest desyncs `seed_scores` (absolute vs chunk-local positions). Not exercised by the flagship (`lowrank_surprise`), but the `attn`/`blend` retention modes advertised in `bug_cache.py:60-89` are broken under the chunked protocol every pod uses.
- **Stale accounting docstring [C].** `accounting.py:433-436` states quanto scale/shift are fp32 → 0.1875×/0.3125×; the run path bills the measured bf16 width (0.156/0.281). Harmless but misleading to a reader of the module.
- **Eviction at 0.25× omitted [C].** Plain `ea-k0.25` retrieval rows at 16K (Llama) exist and are competitive with the flagship (Q5) but appear in no paper table.
- **Palu/ThinK footprints are analytic; KIVI's residual is analytic; only eviction's kept fraction and BUG's state are measured from the live cache** (`w10_frontier.py:532-618`).
- **ShadowKV post-fix result unreported** (Q4); **Oja swap and sysfix pods unharvested** (Q6, Q8).
- **Defaults vs paper**: `--quant-scheme` default is `token` (the broken W18 config, `w10_frontier.py:853-860`); `--filler` default `cycle`; `--n-trials` default 4 (`w10_ruler.py:647`); the paper's n=12 comes only from the pod flags. `--palu-group 1` (per-head) is never varied.
- **Config-hazard framing [I].** §4.6 calls the W18 per-token 0.00 "a configuration hazard of the library default"; the repo's own default remains that configuration.


---

# Part C — Statistics audit from per-trial records

# CODE_C — Statistics audit of "Dynamical Low-Rank Approximation as an Online KV-Cache Compressor" (v. 2026-09-06)

Auditor: independent recomputation from the committed per-trial records and line-files in the repository
(`results/w18_pertrial/`, `results/w19_pertrial/`, `results/*-lines.txt`, `results/w15-*-lines.txt`),
using scipy 1.17 / statsmodels 0.15. Nothing under `docs/reviews/` was read. Docs consulted for T4 only:
`docs/week18-kickoff.md`, `docs/week19-kickoff.md`, `docs/week15-significance.md`, `docs/PLAN.md`
(plus `docs/week17-kickoff.md`, which the W18 kickoff points at as the marquee's origin) and the pod
drivers `scripts/pod/w16.sh`, `w18.sh`, `w19.sh`.

**One-paragraph verdict.** Every number I could recompute from the released per-trial records matches the
paper: Tables 1, 2, 3, 6, 7, 8, the Table-5 sub-cliff cell, all headline McNemar p-values (2.0e-3, 3.1e-2,
21 vs 0 pooled, 0.008/0.016/0.031 per family, 0.004/0.002 at 32K), the official-RULER contrasts, and the 64K
cells. The statistical machinery (Wilson score interval, exact two-sided binomial McNemar on
`(seed, trial)`-paired records) is implemented correctly. The problems are in *design and framing*, not
arithmetic: (i) the "pre-registered confirmatory" marquee contrast was already observed at the same
n=16 on the identical 16 prompts one program-week earlier (W17, bit-identical 15/16 vs 5/16 vs 9/16), so the
W18 run is a deterministic re-execution, not an independent confirmation; (ii) the n=12 cells are 12
needle-code draws on *one* fixed haystack with needle positions fixed by trial index, so trials are not
exchangeable samples of the task distribution and several cells show trial-index clustering (design effect up
to 2); (iii) the "0.05 bits/token band" perplexity tie is a point-estimate statement — the paired 95% CI
(n=8 windows) reaches +0.072 bits vs think-c0.5 and the flagship is significantly *worse* than full KV
(+0.076 [0.031, 0.122]); (iv) under any family-wise correction wider than "the one contrast" nothing but
the 12/0 eviction blow-outs survives Holm, and the flagship-vs-2-bit multi-value edge survives only BH at
16K/32K in-repo; (v) two Wilson upper bounds are truncated rather than rounded (0.80 vs 0.81; 0.98 vs 0.99);
(vi) task-name collisions (`vt`, `niah_multivalue`) between the in-repo and official generators produce
conflicting duplicate keys in `fork-llama-trials.txt` and across files, so Table 5's 16K var-track composite
cells cannot be unambiguously reconstructed from the committed per-trial records alone.

---

## T1. Per-trial record format and table reconstruction

### Format

Every record is one line:

```
[trial] task=<niah_single|niah_multikey|niah_multivalue|vt|niah_single_1..3|niah_multikey_1..3|niah_multiquery> ctx=<16384|32768|65536> arm=<arm> seed=<0|1> trial=<int> hit=<0|1> frac=<float>
```

Fields present: `task, ctx, arm, seed, trial, hit, frac`. **Not present:** model (inferred from the file
name), needle depth, filler id, generation text, timestamp, git SHA (the SHA is in a separate provenance
file per pod, not per record). `frac` is the fraction of targets found (all-or-nothing `hit` requires
`frac == 1`).

| file | records | content |
|---|---|---|
| `w18_pertrial/{llama,mistral,qwen}-trials.txt` | 192 each | G1: flagship `bugSseed-r64-h256` + the *invalid* `bugS-r64-h256-q4`; 4 tasks × 16K/32K × n=12 |
| `w18_pertrial/g2-qwen-trials.txt` | 156 | G2 **WikiText-filler** run (Qwen 16K): think/palu/ea + flagship single only |
| `w18_pertrial/g3-*-trials.txt` | 288 each | G3 eviction grid: `ea-k0.1`, `snapkv-k0.1`, `bugEVICT-h256` |
| `w18_pertrial/g4-llama-trials.txt` | 240 | marquee `bugSseed-r128-h1024-s32`, think, palu (n=16, trials 0–7) + `bugSseed-r256-h1024` (n=12) at 32K |
| `w18_pertrial/g5-llama-trials.txt` | 6 | storage rows, not trials |
| `w19_pertrial/a1-*-trials.txt` | 208 each | KIVI 2/4-bit n=12; 8-bit hqq n=4 (16K) |
| `w19_pertrial/a1q-*-trials.txt` | 96/96/48 | `bugSseed-r64-h256-q4` (sub-cliff cell); Qwen 16K only |
| `w19_pertrial/a2-llama-trials.txt` | 756 | official RULER, 7 arms × 9 tasks × 12 records (trial = official record index; `vt` uses seed 0 trials 0–11) |
| `w19_pertrial/a4-llama-trials.txt` | 176 | 64K: flagship n=8 (trials 0–3), comparators n=12 |
| `w19_pertrial/fork-*-trials.txt` | 808/384/384 | eviction×quant composites, in-repo 16K/32K + (Llama) official |
| `a3-*`, `forkdiag-qwen` | 0 | empty |

Trial layout: n=12 ⇔ seeds {0,1} × trials {0..5}; n=16 ⇔ seeds {0,1} × trials {0..7}; n=8 (64K) ⇔ seeds
{0,1} × trials {0..3}; n=4 (hqq) ⇔ seeds {0,1} × trials {0,1}.

**Data-hygiene finding.** 112 duplicate `(model, task, ctx, arm, seed, trial)` keys exist across files,
20 of them with *conflicting* `hit` values. All conflicts are task-name collisions between generators
(`vt` and `niah_multivalue` are used both by the in-repo generator and by the official-RULER runner; `g2`
is a WikiText-filler re-run of the same arm names). Inside `fork-llama-trials.txt` alone, the 16K `vt` cell
holds 21–23 records per arm (12 official + 12 in-repo, minus exact-duplicate lines), with 3–5 keys per arm
carrying both a 0 and a 1. The report script (`scripts/w19_fork_report.py`) disambiguates using section
markers in raw pod logs (`results/w19_harvest/fork-*.raw`) that are **not committed**.

### Table 1 (16K, `bugSseed-r64-h256`, n=12) — recomputed

| model | single | multi-key* | multi-value | var-track | paper |
|---|---|---|---|---|---|
| Qwen2.5-7B | 12/12 1.00 [0.76,1.00] | 12/12 | 12/12 1.00 [0.76,1.00] | 12/12 1.00 [0.76,1.00] | match |
| Mistral-7B | 12/12 | 12/12 | 12/12 | 6/12 0.50 [0.25,0.75] | match |
| Llama-3.1-8B | 12/12 | 12/12 | 12/12 | 7/12 0.58 [0.32,**0.81**] | paper prints 0.80 (Wilson upper = 0.8067; truncated, not rounded) |

*Multi-key at 16K is 12/12 on all three but is omitted from Table 1 (it appears in Table 7).

The think-c0.5 / palu-r0.5 comparators quoted in §4.2 for 16K ("Qwen 16K: 1.00 vs think-c0.5 0.83 /
palu-r0.5 0.92") come from **Week 17** (`results/w17-qwen-lines.txt`, n=12, no per-trial records exist).
The only 16K think/palu per-trial records (`g2-qwen`) are from the WikiText-filler run and give think mv
0.67, palu mv 0.83 — those are the §4.7 realistic-filler numbers, correctly attributed there. So the 16K
flagship-vs-think/palu statements are not reconstructable from per-trial records; they rest on W17
aggregate lines.

### Table 2 (32K, n=12) — recomputed

| model | single | multi-key | multi-value | var-track | paper |
|---|---|---|---|---|---|
| Qwen | 12/12 | 12/12 | 12/12 | 12/12 | match |
| Mistral | 12/12 | 12/12 | 10/12 0.83 [0.55,0.95] | 5/12 0.42 [0.19,0.68] | match |
| Llama | 12/12 | 12/12 | 12/12 | 11/12 0.92 [0.65,**0.99**] | paper prints 0.98 (Wilson upper = 0.9851) |

### Table 3 (marquee, Llama 32K) — recomputed

| arm | n | single | multi-key | multi-value | var-track | McNemar vs marquee (vt) | paper |
|---|---|---|---|---|---|---|---|
| bugSseed-r128-h1024-s32 | 16 | 16/16 | 15/16 0.94 | 16/16 1.00 [0.81,1.00] | 15/16 0.94 [0.72,0.99] | — | match |
| think-c0.5 | 16 | 16/16 | 16/16 | 16/16 | 5/16 0.31 [0.14,0.56] | 10/0, p = 0.00195 | match (2.0e-3) |
| palu-r0.5 | 16 | 16/16 | 16/16 | 16/16 | 9/16 0.56 [0.33,0.77] | 6/0, p = 0.03125 | match (3.1e-2) |
| KIVI 4-bit | 12 | 12/12 | 12/12 | 12/12 | 12/12 1.00 [0.76,1.00] | 0/1 on 12 shared, p = 1 | match |
| bugSseed-r256-h1024 (r/n=0.25 control) | 12 | 0/12 | 0/12 | 0/12 | 0/12 | — | match (§4.4) |

Note the marquee's own multi-key is 15/16 (not reported in Table 3, which shows only var-track). Wilson
"disjointness" 0.72 > 0.56 confirmed.

### Table 5 (sub-cliff cell and composites, Llama)

`bugSseed-r64-h256-q4` (a1q): Llama 16K 1.00/1.00/1.00/0.50, 32K 1.00/1.00/1.00/0.83; Mistral 16K
1.00/1.00/0.92/0.33, 32K 1.00/1.00/0.83/0.58; Qwen 16K 0/0/0/0 (no Qwen 32K records) — **all match**.
Composites from `fork-llama`: 32K rows and 16K single/multi-key/multi-value match exactly; 16K var-track
is **ambiguous in the per-trial file** (paper 0.00 / 0.42 / 0.50 lie within the reconstructable ranges
0.00 / 0.25–0.50 / 0.42–0.67). Official means 0.17/0.19/0.33 match `results/w20-fork-report.md`.

### Table 6 (`ea-k0.1`, n=12) — all 16 cells match. (Also reproduced: `snapkv-k0.1` 16K multi-key 0.17–0.42, vt 0 — matches §4.5 text.)

### Table 7 (KIVI, n=12) — all 72 accuracy cells match (hit counts below).

| model | ctx | arm | s | mk | mv | vt |
|---|---|---|---|---|---|---|
| Llama | 16K | flagship / 2-bit / 4-bit | 12/12 · 12/12 · 12/12 | 12 · 8 · 12 | 12 · 5 · 12 | 7 · 8 · 12 |
| Llama | 32K | " | 12 · 12 · 12 | 12 · 10 · 12 | 12 · 11 · 12 | 11 · 11 · 12 |
| Mistral | 16K | " | 12 · 11 · 12 | 12 · 7 · 12 | 12 · 6 · 12 | 6 · 4 · 3 |
| Mistral | 32K | " | 12 · 10 · 12 | 12 · 3 · 12 | 10 · 1 · 12 | 5 · 0 · 3 |
| Qwen | 16K | " | 12 · 12 · 12 | 12 · 10 · 12 | 12 · 4 · 11 | 12 · 11 · 12 |
| Qwen | 32K | " | 12 · 11 · 12 | 12 · 7 · 12 | 12 · 2 · 12 | 12 · 3 · 12 |

### Table 8 (official RULER, Llama 16K, 12 records/task) — all 63 cells and 7 means match.

Flagship misses = 22 (14 needle + 8 vt) ✓. Depth statements ("depths 0.15–0.95, six at ≤ 0.20") are
supported by `results/w19-a2-flagship-misses.md`, which joins record indices to the official generator's
`token_position_answer/length`; depth is **not** in the per-trial records. One nuance: record
`niah_multikey_3/4991` (depth 0.17) is missed by *every* arm including full KV, so "misses the
near-lossless arms do not make" holds for 13 of 14, not 14.

### Other text cells checked

- 8-bit hqq control (n=4): Llama 4/4 on all four; Qwen 4/4 on all four; **Mistral vt 1/4** — the paper
  correctly scopes the claim to "Llama and Qwen".
- 64K (a4): flagship 8/8 on all four; 2-bit 12/12, 7/12, 6/12, 12/12; ea-k0.1 12, 8, 10, 12 — match.
  **4-bit at 64K: 12/12, 10/12, 12/12, 7/12** — the paper does not report the 4-bit 64K retrieval; its
  statement that 4-bit "trails only on Mistral variable-tracking on point estimate" is true of Table 7 but
  not of the 64K cell (4-bit trails the flagship on multi-key 0.83 and var-track 0.58 there).
- Invalid W18 `bugS-r64-h256-q4` rows: 1.00/0.67/0.00/0.00 at 16K on all families — consistent with the
  footnote and `results/w18-g1-report.md`.

---

## T2. Wilson intervals, paired McNemar, and pairing

**Implementation check.** `scripts/w15_intervals.py::wilson` is the standard Wilson score interval
(z = 1.96). `scripts/w18_intervals.py::mcnemar_exact` pairs on `(seed, trial)` and computes the exact
two-sided binomial on discordant pairs (`binomtest(min(b,c), b+c, 0.5)`), p = 1 when there are no
discordants. Both agree with my independent implementation to all printed digits.

**Pairing.** For every n=12-vs-n=12 and n=16-vs-n=16 contrast the `(seed, trial)` key sets are identical
(`keys_equal=True` for all 120 in-repo contrasts and all 8 marquee-vs-think/palu contrasts). The generator
(`scripts/w10_ruler.py::build_task`) seeds `torch.Generator().manual_seed(seed*131 + trial)` for needle
codes, and with the default `--filler cycle` the haystack is `_FILLER` (a fixed 10-sentence pool) cycled to
length — identical for every seed, trial and arm. Decode is greedy (`argmax`). Hence same (seed, trial) ⇒
byte-identical prompt across arms: the McNemar pairing is valid. For n=16-vs-n=12 (marquee vs KIVI,
marquee vs r64 flagship) and 64K n=8-vs-n=12 the test silently restricts to the shared keys (12 and 8).

**Headline p-values (all confirmed):**

| contrast | discordant | McNemar p (exact) | paper |
|---|---|---|---|
| marquee vs think-c0.5, vt, n=16 | 10/0 | 1.95e-3 | 2.0e-3 ✓ |
| marquee vs palu-r0.5, vt, n=16 | 6/0 | 3.13e-2 | 3.1e-2 (0.03) ✓ |
| r64 vs 2-bit, mv 16K: Llama / Mistral / Qwen | 7/0, 6/0, 8/0 | 0.0156 / 0.0313 / 0.0078 | 0.016 / 0.031 / 0.008 ✓ |
| pooled mv 16K, three families | 21/0 | 9.5e-7 | "21 vs 0" ✓ |
| Mistral 32K mk / mv | 9/0, 9/0 | 0.0039 / 0.0039 | 0.004 ✓ |
| Qwen 32K mv / vt | 10/0, 9/0 | 0.0020 / 0.0039 | 0.002 / 0.004 ✓ |
| Llama 16K vt, r64 vs 2-bit | 3/4 | 1.0 | "3 vs 4, not separated" ✓ |
| Llama 16K vt, 4-bit vs r64 | 0/5 | 0.0625 | "5/0, p=0.06" ✓ |
| official: r64 vs ea-k0.1 | 6 of 9 tasks p ≤ 0.0078 | | "6 of 9, p ≤ 0.008" ✓ |
| official mv / mq: r64 vs 2-bit | 1/1, 0/4 | 1.0, 0.125 | ✓ |
| official vt: r64 vs 4-bit / palu / think | 0/8, 0/8, 0/7 | 0.0078, 0.0078, 0.0156 | ✓ |

Full per-model × task × ctx sweep (r64 flagship vs 2-bit, 4-bit, ea-k0.1, snapkv-k0.1, bugEVICT-h256,
q4 cell; marquee vs think/palu/4-bit/2-bit/r64; official vs six arms) is in the audit workspace
(`allcontrasts.json`, 202 contrasts). Contrasts favouring a **baseline** at p<0.05: Mistral 32K vt
`bugEVICT-h256` 0/6 (p=0.031); official vt vs full/palu/4-bit/think. No 2-bit or 4-bit cell favours the
quantizer at p<0.05 (confirmed).

A side-note on the "p≈0.0006" cited in the W18 kickoff for the same marquee cell: that is the unpaired
Fisher exact test on 15/16 vs 5/16 (I get 0.00064); the paper correctly switched to the paired McNemar.

---

## T3. Multiplicity

**Counting.** Accuracy cells tabulated in the paper: Table 1 = 9, Table 2 = 12, Table 3 = 4, Table 5 = 32
(+4 official means), Table 6 = 16, Table 7 = 72, Table 8 = 63 (+7 means) → **208 accuracy cells** in
tables (the paper's "∼40 descriptive Wilson cells" counts only those printed with intervals). Text-only
cells add ≈ 40 more (64K, hqq, r256, realistic filler, 4-bit leads etc.). Paired contrasts for which the
paper quotes a p-value or discordant count: ≈ 22. Paired contrasts *available* from the per-trial data: 202.

**Corrections (Holm FWER 0.05; Benjamini–Hochberg FDR 0.05).** Because the exact McNemar at n=12 has a
minimum attainable p of 2·0.5¹² = 4.9e-4 (n=16: 3.1e-5 only if all 16 are discordant), the power ceiling
matters as much as the family size.

| family | m | raw p<0.05 | Holm survivors | BH survivors |
|---|---|---|---|---|
| (a) r64 flagship vs KIVI-2-bit, all cells incl. 64K + official | 37 | 7 | **0** | 4 (Qwen 32K mv; Mistral 32K mk, mv; Qwen 32K vt) |
| (a′) same, in-repo 16K/32K only | 24 | 7 | 1 (Qwen 32K mv) | 5 (+ Qwen 16K mv) |
| (b) all flagship-vs-baseline contrasts | 202 | 68 | **0** | 51 |
| (b′) in-repo r64 vs any baseline, 16K/32K | 120 | 56 | 0 | 47 |
| (c) marquee vs think/palu/4-bit/2-bit, 4 tasks | 16 | 2 | 1 (think, vt) | 1 |
| (c′) marquee vt vs think & palu only | 2 | 2 | 2 | 2 |
| (d) the ≈49 contrasts the paper reports or calls "separated" | 49 | 19 | 1 (official mk2 vs ea) | 17 |

Survivors worth naming:
- The **think-c0.5 marquee contrast** survives Holm within the marquee family (16) and BH within any
  family tried; it does not survive Holm over all 202 or all 49 reported contrasts (threshold 0.05/49 =
  1.0e-3 < 1.95e-3). Its robustness therefore rests entirely on the "family = 1, pre-registered" argument (see T4).
- The **palu-r0.5** contrast survives nothing beyond the two-contrast family — consistent with the paper's
  "suggestive".
- The **flagship-vs-2-bit multi-value edge**: Llama 16K (p=0.016) and Mistral 16K (p=0.031) survive no
  correction in any family; Qwen 16K survives BH only within the 24-cell in-repo family; the 32K
  Mistral/Qwen cells (9/0, 10/0) survive BH but not Holm. The pooled 21/0 (p≈1e-6) is the robust
  statement — but it pools three families and is not itself pre-registered.
- Every 12/0 and 11/0 eviction blow-out (ea-k0.1, snapkv, bugEVICT) survives BH in all families.

---

## T4. Marquee pilot overlap and pre-registration

**Overlap.** The W17 pod driver (`scripts/pod/w16.sh`, `MARQUEE=1` block) ran `bugSseed-r128-h1024-s32`,
think-c0.5 and palu-r0.5 at 32K with `--n-trials 8 --seeds 0 1` (n=16). Its committed aggregate lines
(`results/w17-llama8b-lines.txt`, `W17_MARQUEE` block; `results/w17-ruler-intervals.md`) give **vt 15/16,
think 5/16, palu 9/16 and mv 16/16 for all three — identical to the W18 G4 cell** (`scripts/pod/w18.sh::g4`,
again `--n-trials 8 --seeds 0 1`). Because prompts are a deterministic function of (seed, trial) and decode
is greedy, the W18 "confirmatory" run re-executed the *same 16 prompts* and reproduced the same 16
outcomes (a computational replication, not an independent sample). The n=16 marquee trials are therefore
**not disjoint** from the earlier run; and the W17 32K "n=4" core cells (`--n-trials 2 --seeds 0 1` =
trials {0,1}) are a strict subset of the n=12 firming set (trials {0..5}) and of the n=16 set. "Firming what
were previously n=4 point estimates" is accurate as a description of n, but the added trials are new
needle codes on the *same* haystack and the same positions, not new pilot-independent draws.

**Pre-registration trail (no commit dating available).** The repository has a single squashed commit
(`7229d04`, 2026-09-08 19:13 -0400); every file carries that date, so `git log` cannot date any artifact.
From content:
- `docs/week15-significance.md` pre-registers the Wilson/n=8 separation logic and the perplexity tie rule,
  but no marquee contrast.
- `docs/week17-kickoff.md` pre-registers a matrix of `bugSseed-r64-h256` vs full/think/palu at 16K (n≥8)
  and 32K (n=4). The r128-h1024-s32 marquee is **not** in that matrix; it appears in W17 results as an
  added `MARQUEE=1` block.
- `docs/week18-kickoff.md` (written after the 2026-09-01 panel) already cites "marquee vt vs think-c0.5
  p≈0.0006, n=16" as a *surviving result* and lists G4 as "firming … marquee single/mk n=16 … marquee ppl
  same-pod". This is the earliest artifact that names the contrast and its comparator (think-c0.5, at the
  panel's request to "name the marquee comparator") — i.e. the comparator was fixed after the W17 result
  was known.
- `docs/week19-kickoff.md` (2026-09-05): "the multiple-comparisons defense is *pre-registration (family=1)*,
  not Bonferroni."
- `results/w18-g4-marquee-contrasts.json` holds the two McNemar results (think p=0.002, palu p=0.0312).
- No "pod matrix" file with a timestamp preceding the W17 marquee run exists in the repo.

**Assessment.** The paper's statement "registered in the pod matrix before the run" is defensible only if
"the run" means the W18 G4 re-execution. The contrast was selected after an identical n=16 result had been
observed in W17, and the G4 run could not have come out differently on the same prompts. The correct
description is "replicated deterministically" rather than "confirmatory"; the family-wise argument
("needs no inflation") should not rest on it.

---

## T5. Per-seed × per-trial structure, clustering, design effect

Depth is not recorded, but it is a deterministic function of the trial index in the in-repo generator:
single-needle at n/2 + (trial mod 5) sentences (≈ depth 0.50 always); multi-key: 8 keys at depths
(k+1)/9, queried key = trial mod 8 → depth 0.11, 0.22, … 0.67 for trials 0–5 (…0.89 for the n=16 marquee);
multi-value: 4 values at 0.2/0.4/0.6/0.8, all required; var-track: 4-statement chain at 0.2/0.4/0.6/0.8
(root at 0.2), variable names `X{trial}{i}`. Consequences: **no depth sweep exists in any headline cell**,
single-needle never probes early/late depth, and the two seeds differ only in the random codes.

Hit grids for the imperfect flagship cells (rows = seed, columns = trials 0..5/7):

| cell | seed 0 | seed 1 | hits by trial | ICC(seed) / DEFF | ICC(trial) / DEFF |
|---|---|---|---|---|---|
| Llama 16K vt (7/12) | 1 1 0 1 1 0 | 0 1 1 0 0 1 | 1 2 1 1 1 1 | 0.00 / 1.00 | 0.00 / 1.00 |
| Llama 32K vt (11/12) | 1 1 1 1 1 1 | 1 1 1 0 1 1 | 2 2 2 1 2 2 | 0 / 1 | 0 / 1 |
| Mistral 16K vt (6/12) | 0 1 1 0 0 1 | **0 1 1 0 0 1** | 0 2 2 0 0 2 | 0 / 1 | **1.00 / 2.00** |
| Mistral 32K vt (5/12) | 1 0 0 0 0 0 | 1 1 1 0 0 1 | 2 1 1 0 0 1 | 0.29 / 2.45 | 0.06 / 1.06 |
| Mistral 32K mv (10/12) | 1 1 1 1 1 1 | 0 1 1 1 1 0 | 1 2 2 2 2 1 | 0.20 / 2.00 | 0 / 1 |
| marquee 32K vt (15/16) | 1 1 1 1 1 1 1 1 | 1 1 1 0 1 1 1 1 | — | 0 / 1 | 0 / 1 |
| think-c0.5 32K vt (5/16) | 0 1 0 0 0 1 0 1 | 0 1 0 0 0 0 0 1 | 0 2 0 0 0 1 0 2 | 0 / 1 | **0.74 / 1.74** |
| palu-r0.5 32K vt (9/16) | 1 0 1 1 1 0 1 1 | 0 0 0 0 1 1 0 1 | 1 0 1 1 2 1 1 2 | 0.14 / 2.00 | 0 / 1 |
| Llama 16K 2-bit mk (8/12) | 0 1 1 0 1 1 | **0 1 1 0 1 1** | 0 2 2 0 2 2 | 0 / 1 | **1.00 / 2.00** |
| Mistral 32K 2-bit mk (3/12) | 0 0 0 0 1 1 | 0 0 0 0 1 0 | 0 0 0 0 2 1 | 0 / 1 | 0.62 / 1.62 |

Reading:
- **Within-seed clustering is weak** (ICC by seed ≤ 0.29). That is expected: the "seed" only changes the
  codes, and the 6 trials of a seed are *not* on a shared filler in any sense that distinguishes them
  from the other seed — every trial in the whole programme is on the same filler.
- **Clustering by trial index is strong in several cells** (Mistral 16K vt and Llama 16K 2-bit multi-key
  have identical patterns under both seeds; think-c0.5 vt hits only trials 1, 5, 7 in both seeds). For
  multi-key this is a *depth* effect (2-bit misses the queried key at depths 0.11 and 0.44 under both seeds;
  at 32K it hits only depths 0.56/0.67 — i.e. front-loaded misses in the baseline); for var-track it is
  a variable-name/position effect. Under trial-index clustering the design effect is 2.0 for those cells,
  and the effective n is 6, not 12: Mistral 16K vt 0.50 becomes [0.19, 0.81]; think vt 0.31 becomes
  [0.11, 0.63]; Llama 16K 2-bit mk 0.67 becomes [0.30, 0.90].
- For the flagship's perfect cells the Wilson lower bound with DEFF = 2 is 0.61 (12/12 → n_eff 6) instead of
  0.76; for the marquee 15/16 with DEFF = 2 it is 0.60 (vs 0.72), which would overlap think's
  DEFF-adjusted upper bound 0.64 — the "Wilson-disjoint" statement is not robust to a design effect of 2,
  though the paired McNemar (which conditions on the prompt) is unaffected by between-prompt clustering.
- **Front-loading of flagship misses (first ingest chunk = first 4096 tokens):** the in-repo cells cannot
  show it for single/multi-key (no early-depth single needles; multi-key misses by the flagship are 0 at
  16K/32K and 1 at the marquee's depth 0.78). The var-track root sits at depth 0.2, which *is* inside the
  first chunk at 16K (3.3K tokens) and *outside* it at 32K (6.6K); the flagship's vt is worse at 16K than
  32K on Llama (0.58 vs 0.92) and for the q4 cell (0.50 vs 0.83), consistent with a first-chunk effect,
  but Mistral goes the other way (0.50 vs 0.42). The official-RULER misses (6 of 14 at depth ≤ 0.20 where
  ≈ 20 % of the range lies in the first chunk) are the only direct evidence and are consistent with
  front-loading, but n=14 misses barely separate "front-loaded" from "uniform" (binomial P(≥6 of 14 at
  depth ≤ 0.25) = 0.11 under a uniform-depth null; P(≥6 at ≤ 0.20) = 0.044).

---

## T6. Perplexity uncertainty

Per-window NLLs are committed **only for the Week-15 runs** (`results/w15-confirm-lines.txt` 32K,
`results/w15-complete-lines.txt` 16K; 8 windows × 511 tokens; arms full, think-c0.5, palu-r0.5,
bugSseed-r128-h1024(-s32)). No `[pplw]` rows exist for the W18 "same-pod" marquee perplexity, the W19 KIVI
perplexities (n=4 windows) or the 64K point; those are single pooled numbers, so **no SE/CI can be computed
for any §4.6 fluency comparison**. The pooled W15 values (full 6.975, think 7.196, palu 7.232, marquee 7.353
at 32K) are digit-for-digit the values the paper attributes to the same-pod W18 run — consistent with a
deterministic re-execution, but it means the "same-pod" measurement added no new windows.

Paired per-window differences (bits/token, n=8, 95 % t-interval):

| T | contrast | Δ | SE | 95 % CI | windows +/− | tie rule \|Δ\| ≤ max(0.05, 2·SE) | CI inside ±0.05? |
|---|---|---|---|---|---|---|---|
| 32K | marquee − think | +0.031 | 0.017 | [−0.010, **+0.072**] | 5/2 | tie | **no** |
| 32K | marquee − palu | +0.024 | 0.016 | [−0.013, **+0.061**] | 6/1 | tie | **no** |
| 32K | marquee − full | **+0.076** | 0.019 | [+0.031, +0.122] | 7/0 | not a tie (p = 0.005) | no |
| 32K | think − full | +0.045 | 0.011 | [+0.020, +0.070] | 8/0 | tie by rule | no |
| 32K | palu − full | +0.052 | 0.011 | [+0.027, +0.078] | 8/0 | not a tie | no |
| 16K | marquee − think | +0.006 | 0.007 | [−0.012, +0.023] | 3/2 | tie | yes |
| 16K | marquee − palu | +0.004 | 0.008 | [−0.015, +0.023] | 4/3 | tie | yes |
| 16K | marquee − full | +0.044 | 0.011 | [+0.019, +0.069] | 7/0 | tie by rule | no |

So the "within the pre-registered 0.05-bits/token band of both compressors" claim at 32K is true of the
**point estimates** and satisfies the pre-registered tie rule, but a 95 % CI does **not** confine the
difference to the band (upper bounds 0.072 and 0.061); an equivalence (TOST) test at ±0.05 fails. The
"+0.076 over uncompressed full KV" is a statistically clear deficit. At 16K the band claim is CI-supported.

bf16 resolution: the committed per-window NLLs are quantized in steps of 0.007828 nats = **0.0113
bits/token** (the 32K window sums ≈ 990 nats fall in the bf16 bin of width 4); for windows summing above
1024 nats the step is 0.0226 bits. The paper's "≈ 0.006-bit/token floor" understates the resolution by
≈ 2×; the smallest Δ quoted (+0.024) is about two quantization steps.

§4.6 fluency deltas in bits/token (n=4 windows, no CI possible): Llama 16K flagship − 2-bit = −0.024 (the
flagship is *better* here, within noise), 32K +0.009; Mistral +0.140/+0.120; Qwen +0.288/+2.09. The
Llama ties and the Mistral/Qwen losses are consistent with the text, but "trails or ties on fluency" is not
supported by an interval for any cell.

---

## T7. The 64K cells

`a4-llama`: flagship `bugSseed-r64-h256` n=8 = seeds {0,1} × trials {0..3}; comparators n=12 = trials
{0..5}. "No contrast separates" was computed by the paired exact McNemar restricted to the **8 shared
keys** (the script silently intersects). Recomputed:

| task | flagship | 2-bit (12) / on shared 8 | McNemar | Fisher (unpaired, 8/8 vs k/12) |
|---|---|---|---|---|
| single | 8/8 | 12/12 · 8/8 | 0/0, p=1 | 1 |
| multi-key | 8/8 | 7/12 · 4/8 | 4/0, p=0.125 | 0.055 |
| multi-value | 8/8 | 6/12 · 4/8 | 4/0, p=0.125 | **0.042** |
| var-track | 8/8 | 12/12 · 8/8 | 0/0 | 1 |
| multi-key vs 4-bit | 8/8 | 10/12 · 6/8 | 2/0, p=0.5 | 0.50 |
| var-track vs 4-bit | 8/8 | 7/12 · 5/8 | 3/0, p=0.25 | 0.055 |
| multi-key / mv vs ea-k0.1 | 8/8 | 8/12 · 6/8, 10/12 · 7/8 | 2/0, 1/0 | 0.12, 0.50 |

With the baseline at 4/8 on the shared prompts the paired test cannot go below p = 0.125 (max 4
discordants), so "no contrast separates" is a power statement, not evidence of parity — the paper's own
caveat at n=12 ("≥ 6 one-way discordant pairs") applies a fortiori at n=8, where even 8/0 would give
p = 0.0078 and 4/0 is the observed ceiling. An unpaired Fisher test on the full n would call the
multi-value contrast significant (p = 0.042). The paper reports the comparators' n=12 accuracies next to
the flagship's n=8 without flagging that the paired test used only 8 pairs.

---

## T8. Post-paper material (`results/w20*`, commit "results(w20): register swap + sysfix pods")

- `results/w20-fork-report.md` (eviction×quantization composites) is already incorporated in the paper
  (Table 5) and reproduces from the `fork-*` per-trial files up to the `vt`-16K ambiguity noted in T1.
- The commit message refers to **registration of two new pods**, not results. `results/w19_harvest/pods.txt`
  lists `swap-llama` and `sysfix-llama` pod IDs; `scripts/pod/w19.sh` defines them:
  - `swap`: a **tracker-swap ablation** — the flagship cache (sinks, ring, warm-up seed, h256 surprise tier,
    r64) with the gist tracker replaced by Oja's rule or Frequent Directions, Llama 16K, 4 tasks, n=12 on
    the a1q needles plus same-pod perplexity. The comment states the pre-registered reading: "if Oja/FD
    match the flagship on retrieval AND ppl, 'DLRA' is branding and the paper says so; if they lose, the
    tracker is load-bearing." It also notes that the flagship as run *is* fixed-rank incremental SVD
    (θ off, `min_sv_frac` = 0).
  - `sysfix`: (1) measured decode ms/token and peak/resident VRAM for full / flagship / KIVI-2-bit at
    16K/32K/64K (replacing the 1B-CPU latency datum and the analytic 1.06× residency); (2) `ss2`, the
    KIVI-2-bit arm with **single-shot prefill** (`--chunk 0`) at 16K, n=12, same needles — to test whether
    the in-repo multi-value edge over 2-bit is an artefact of the chunked-prefill protocol.
- **No results from either pod are committed** (no `w19-*-swap*`, `w19-*-latency*`, `w19-*-ss2*`, no
  `w20-close-report.md`). Nothing in the repository yet changes the paper's numbers; but the two questions
  these pods ask (is the DLRA integrator load-bearing vs Oja/FD; is the 2-bit multi-value deficit
  protocol-bound) are exactly the two places where the paper's framing is most exposed, and the paper's
  §4.6 already concedes the streaming-prefill caveat.

---

## Summary of match / mismatch statements

| item | status |
|---|---|
| Tables 1, 2, 3, 6, 7, 8 accuracies and hit counts | **match** the per-trial records exactly |
| Wilson intervals | match, except two upper bounds printed truncated (0.80 for 0.807; 0.98 for 0.985) |
| Headline McNemar p-values and discordant counts (§4.3, §4.6, §4.7) | **match** exactly |
| Pairing on identical prompts across arms | **verified** (identical (seed, trial) keys; deterministic generator; greedy decode) |
| Table 5 sub-cliff cell and 32K composites | match; 16K composite var-track cells ambiguous in the committed per-trial file |
| §4.2 16K think/palu comparators | from W17 aggregate lines; no per-trial records |
| "Pre-registered confirmatory" marquee | **not supported as independent confirmation**: identical n=16 result existed in W17 on the same prompts; comparator named after the fact; no dated pre-registration artifact; repo has one commit |
| "Firming n=4 → n=12" | true of n; the n=4 prompts are a subset and all trials share one haystack |
| Family-wise robustness of think contrast | holds for family ≤ 16 (Holm) and under BH for any family tried; fails Holm over all reported (49) or all available (202) contrasts |
| Flagship-vs-2-bit multi-value "wins" | per-family p-values correct; none survive Holm; 16K Llama/Mistral survive no correction; pooled 21/0 is robust but pooled and post hoc |
| "0.05 bits/token band" at 32K | point estimate inside; 95 % CI not inside (upper 0.072 / 0.061); flagship significantly worse than full KV |
| "≈ 0.006 bit/token bf16 floor" | committed per-window rows show 0.011–0.023 bits/token |
| "No contrast separates at 64K" | true for paired McNemar on 8 shared pairs, whose attainable minimum p was 0.125; unpaired Fisher gives p = 0.042 on multi-value |
| 4-bit "trails only on Mistral var-track on point estimate" | true for Table 7; at 64K 4-bit trails on multi-key (0.83) and var-track (0.58) |
| Depth/front-loading claims (§4.7) | supported by a derived file (`w19-a2-flagship-misses.md`), not by the per-trial records; 1 of 14 flagship misses is universal |
| Effective sample size | all n=12 cells are 12 code draws on one fixed haystack; several cells cluster by trial index (DEFF ≈ 2), including think-c0.5's marquee vt |
| Post-paper W20 | swap/sysfix pods registered with pre-stated readings; no results committed |

Audit artefacts: `scratchpad/audit_load.py`, `scratchpad/contrasts.json`, `scratchpad/allcontrasts.json`.
