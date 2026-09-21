# kernel_smoke — Amendment 2 draft (for the owner's decision; nothing here is committed to the prereg or launched)

Drafted 2026-09-21 by the L4 session after the pod's harvest (results/kernel_smoke; D-002 addendum of 2026-09-21). The owner picks (A) amend + Task 11 + re-run, (B) record REFUSED and defer, or (C).

---

# Amendment 2 — draft for the owner's decision (2026-09-21, lane L4)

**Nothing in this file is committed, launched, or authoritative.** It is a draft of the
pre-registered follow-up to pod `kernel_smoke` / instance **51903816** (Llama-3.1-8B bf16,
A100-40 GB, launch SHA `a691d44`), whose correctness precondition
(`prereg/kernel_smoke.md:186-206`, Amendment 1 A1.2 at `:487-546`) was missed on both clauses.
Four parts: the reading of the pod, the amendment text, the code task it needs, the decision.

Line references are to `git show 99ea29f:<path>` where the working tree is mid-edit
(`scripts/tables.py`, `scripts/pod.py`, `src/kvdlra/eval/runner.py`, the tests) and to the
working tree otherwise (`src/kvdlra/kernel/attention.py`, `src/kvdlra/kernel/triton_kernel.py`,
`src/kvdlra/eval/kernel_check.py`, `src/kvdlra/eval/records.py`, `src/kvdlra/eval/latency.py`).

---

## 1. The reading of this pod, in pre-registration language

Three sentences the `docs/plan/DECISIONS.md` outcome entry can quote verbatim, plus the
falsifier.

> **(1)** The correctness precondition of `prereg/kernel_smoke.md` §4, as amended by A1.2, is
> **NOT met on both of its clauses**: 12 of 16 pg19-val prompts decoded token-exact against the
> reconstruct twin (bar: ≥ 14/16; the four that diverged are prompts 3, 6, 8 and 14, first
> mismatching at greedy steps 13, 15, 4 and 8), and the worst per-layer `max|Δ|` of the first
> kernel decode step was **1.464e-2** (prompt 7, layer 26) against the bar `< 1e-2`, with four
> prompts above it (2: 1.233e-2 · 7: 1.464e-2 · 10: 1.003e-2 · 12: 1.405e-2).
>
> **(2)** Under §4's rule — "a kernel whose correctness precondition has not passed is not
> measured for speed" (`prereg/kernel_smoke.md:203-205`) — the Week-3 gate reads **REFUSED**, and
> every latency row of this pod (16K: full 24.74 ms / 2.05 GB kv_peak; reconstruct 126.03 ms /
> 3.25 GB / 4 spikes; kernel 39.19 ms p50 / 0.83 GB kv_peak / 0.83 GB kv_resident / 88.0 ms mean
> / 1,992 ms max / 5 spikes. 32K: full 30.13 ms / 4.08 GB) is reported as **measured, not cited**,
> exactly as `scripts/tables.py` `week3_gate` renders it (`99ea29f:scripts/tables.py:1200-1207`).
>
> **(3)** The miss is a **bar-versus-bf16 question and not a demonstrated kernel error**: the
> `pre_run` gate passed on this instance (`7 passed, 1 skipped`) with the Triton kernel matching
> the torch reference on CUDA at `max ≤ 2e-3 ∧ rms ≤ 1e-4` at `n_splits == 1` and `≤ 4e-3` across
> splits on fp32 outputs (Amendment 1 A1.2, `prereg/kernel_smoke.md:535-546`); `kernel_compare`
> subtracts an **fp32** sdpa reference from the kernel's **bf16** output
> (`src/kvdlra/kernel/attention.py:90-99`, the cast at `src/kvdlra/kernel/triton_kernel.py:231`),
> so every recorded Δ carries the kernel's own output rounding, ≈ 1 bf16 unit roundoff
> (`u = 2⁻⁸ = 3.91e-3` relative) of a quantity whose scale the record never stored; and the two
> clauses failed on **disjoint** prompts — all four over-bar prompts are token-exact and all four
> mismatching prompts sit at 7.0e-3–9.4e-3, below the bar — which is the signature of two equally
> valid bf16 roundings diverging at near-ties, not of an arithmetic error, which would put the
> large-Δ prompts and the diverging prompts on the same rows.

**What would falsify that reading**, pre-registered here so the re-run can refute it: a layer
whose `max|Δ|` exceeds a few ulp of its own `max|ref|` — operationally, `max|Δ_l| / max|ref_l| >
2⁻⁶` (§2 (a)) — or a first mismatch at a step where the reconstruct path's top-2 logit gap
exceeds the margin `M` of §2 (b). Either one attributes the miss to the kernel rather than to the
bar. Neither is computable from instance 51903816's records, because `KernelCheckRecord`
(`src/kvdlra/eval/records.py:213-234`) stores neither `max|ref|` nor any logit — which is why
this pod stays REFUSED under its original bar and why the re-run, not a re-reading, is the
follow-up.

### Supporting evidence for (3), for the entry's evidence paragraph

- The CPU calibration the `< 1e-2` bar was set from measured the **reference** kernel against
  reconstruct-then-attend with bf16 operands **and a bf16 output** (`tests/test_kernel_reference.py`
  at `99ea29f`: `TOL` line 42, `random_case` lines 81-102, the assertion lines 110-123): **4.66e-3
  and 5.15e-3** on random 8B shapes whose outputs are deliberately scaled to `|out| ≲ 0.9` (the
  comment at lines 95-98 says why: unscaled, `|out| ≤ 0.11` makes the absolute bar a ~10 %
  relative bar), and **2.75e-3** on one 1B dump layer at `|out| ≤ 0.84`. In units of `u`: 1.33 u,
  1.47 u, 0.84 u. The bar was therefore calibrated at `|out| ≈ 0.9` and applied at an unmeasured
  `|out|`.
- ADR 0001 §5 "Numerics" (`docs/adr/0001-factored-attention-kernel.md:216-220`) predicts the
  kernel's error to be of the **same order** as the reconstruct path's, "~2⁻⁸ relative, roughly √2
  worse" — a *relative* statement, which the pre-registered bar then applied as an absolute one.
- The whole-branch review anticipated this before the rows existed
  (`.superpowers/sdd/2026-09-20-L4-kernel/final-review-report.md`, Important 3): "a correct kernel
  can therefore exceed 1e-2 at a layer whose outputs reach ≈ 2.5–3 even while every prompt is
  token-exact … the only sound follow-up is an amendment committed BEFORE any re-run that
  pre-registers a relative bar."
- Prompt 0's per-layer breakdown spans **2.5e-4 (layer 0) to 8.2e-3 (layer 8)**, a 33× range under
  one kernel, one rank and one set of shapes; the 16-prompt worst spans 5.564e-3 to 1.464e-2, a
  2.6× range under identical shapes. Both spreads are spreads in `|out|`, not in the kernel, if the
  relative error is near-constant — and that is the claim §2 (a) makes testable.
- The worst layer is **31** on 7 of 16 prompts and **26** on 3, i.e. the late layers whose outputs
  feed the lm_head — the layers whose `|out|` is largest.

- Task 4's tiny-model token-exact check was **fp32** (`tests/test_kernel_path.py` at `99ea29f`,
  `test_the_16_prompts_are_token_exact_in_fp32`: 16/16, logits to 1e-4): bf16 token-exactness
  between two equally valid roundings had never been measured anywhere before this pod. The
  `≥ 14/16` count was imported from `docs/plan/lanes/L4_kernel.md` / `GATES.md` §G4 line 2 with no
  bf16 calibration behind it.

---

## 2. Amendment 2 (draft text, to be appended after Amendment 1's corrections)

> ### Amendment 2 (2026-09-21, lane L4, before any re-run's launch commit)
>
> §1–§11 and Amendment 1 are left exactly as written; this file is append-only. This amendment
> **does not re-read instance 51903816**, whose precondition is REFUSED under the bar in force at
> its launch and whose latency rows stay measured-not-cited. It governs **only pods launched after
> its own commit**. It re-states §4's precondition in the units the quantity is actually measured
> in, names the record fields that make the re-statement computable, and says how the kernel arm's
> spike count is read. It moves no other threshold.
>
> #### A2.1 Why a re-statement is owed at all
>
> §4's precondition has two clauses, both written before any bf16 measurement of either existed.
> `kernel_compare` (`src/kvdlra/kernel/attention.py:90-99`) forms `Δ = kernel_output_bf16 −
> sdpa_fp32(bf16 store)`, so `Δ` is the sum of the kernel's operand roundings **and** the bf16
> rounding of its own output; both scale with `|out|`, which the record does not store. ADR 0001
> §5 states the numerics contract relatively ("~2⁻⁸ relative, roughly √2 worse"); the CPU
> calibration that fixed `1e-2` was taken at `|out| ≲ 0.9` (`tests/test_kernel_reference.py:95-98`
> says so in the fixture). An absolute bar read at an unmeasured scale is a bar on `|out|`. The
> `≥ 14/16` clause has the matching defect: two bf16 implementations that are each within a
> rounding of the same fp32 answer will disagree at a greedy near-tie with probability ≈ ½,
> independently of correctness, and no bf16 measurement of that rate existed before this pod.
>
> #### A2.2 (a) The per-layer criterion, re-stated as relative
>
> For each layer `l` of the first kernel decode step, `kernel_compare` records `d_l = max|Δ_l|`
> **and** `m_l = max|ref_l|` (the new field of A2.5). The criterion is
>
> ```
>     rel = max_l ( d_l / m_l )   ≤   2⁻⁶  =  1.5625e-2  =  4 u ,      u = 2⁻⁸ = 3.90625e-3
> ```
>
> where `u` is the bf16 unit roundoff (8 significand bits). A layer with `m_l = 0` is excluded and
> reported. **The absolute `max_l d_l` is still recorded, printed and reported** beside `rel`, with
> the layer index of each; it is a number, no longer a bar.
>
> **Where 4 u comes from — from the calibration, not from taste.** Every term below is a measured
> number already in this repository, expressed relative to the `|out|` it was measured at:
>
> | term | measured | relative | in `u` |
> |---|---|---|---|
> | reference kernel vs reconstruct, random 8B shapes, bf16 operands + bf16 output (`tests/test_kernel_reference.py:110-123`) | 4.66e-3, 5.15e-3 at `\|out\| ≤ 0.9` | 5.18e-3, 5.72e-3 | 1.33, **1.47** |
> | the same, on the 1B dump layer (`tests/test_kernel_reference.py:214+`) | 2.75e-3 at `\|out\| ≤ 0.84` | 3.27e-3 | 0.84 |
> | Triton split-merge reassociation on top of the reference (`pre_run` cross-split bar, A1.2) | ≤ 4e-3 at the same shapes | ≤ 4.44e-3 | ≤ 1.14 |
>
> The worst plausible sum for a **correct** kernel is `1.47 + 1.14 ≈ 2.6 u`; ADR §5's "roughly √2
> worse" is already inside the 1.47 u, which is a measurement of exactly that comparison. **4 u is
> 2.7× the measured worst single term and ~1.5× the worst plausible sum** — the same 2–10×
> discipline A1.2 states for the `pre_run` bars ("each bar sits ≈ 2–10× over the reassociation
> residual"), and it is chosen as a power of two so it reads as "four bf16 ulp" rather than as a
> fitted constant. It separates one extra rounding from an arithmetic error by two orders of
> magnitude: a wrong RoPE angle, a wrong GQA head map or a dropped tile perturbs the output by
> `O(1)` relative, i.e. `≳ 250 u`.
>
> **The prediction this bar makes, written before the re-run.** Applied to instance 51903816's
> absolute numbers, `rel ≤ 4 u` requires `m_l ≥ 0.94` at prompt 7 layer 26 (1.464e-2), `≥ 0.90` at
> prompt 12, `≥ 0.79` at prompt 2, `≥ 0.64` at prompt 10 and `≥ 0.36` at the smallest row
> (prompt 4, 5.564e-3). **It is predicted that every one of these passes** — that `max|ref|` at
> layers 26–31 of Llama-3.1-8B exceeds 1 — and a re-run in which it does not is the falsifier of
> §1 (3), reported as such and not re-argued.
>
> #### A2.3 (b) The token criterion, re-stated for bf16
>
> A greedy mismatch at step `s` **counts against the kernel** only if the reconstruct path's
> **top-2 logit gap at `s`** exceeds the pre-registered margin
>
> ```
>     M  =  2⁻⁵ · max_j L_recon[s, j]   ( = 8 u × the step's own top logit ; ≈ 0.6 for a
>                                         Llama-3.1-8B top logit of ≈ 20 )
> ```
>
> Otherwise the mismatch is a **near-tie divergence**, reported descriptively with its gap, its
> step and both logits, and not counted. The count is then read over attributable mismatches:
>
> ```
>     met  ⟺  ( n_prompts − #{prompts whose first mismatch is attributable} ) ≥ 14
>              AND  rel ≤ 2⁻⁶  AND  no error rows
> ```
>
> **Where M comes from.** The model runs bf16 end to end, so one bf16 rounding of a logit is
> `u·|L| = 2⁻⁸·|L|`. The kernel re-rounds the attention output of each of the 32 layers by ≲ 1 u
> of that layer's own output; through the residual stream those 32 perturbations are independent
> in sign and add in quadrature, `√32 = 5.66`, rounded **up** to the next power of two = 8. Hence
> `M = 8 u · max_j L[s, j] = 2⁻⁵ · max_j L[s, j]`. `M` is stated **relative to the step's own
> logit scale**, not as an absolute number of logit units, so it transfers across models and
> across the tiny model of the calibration below, whose vocabulary is 256
> (`src/kvdlra/kernel/prompts.py:30`) and whose logit scale is nothing like Llama's.
>
> **Only the first mismatch is read, and that is a property of the comparison, not a concession**:
> after step `s` the two paths have different contexts, so steps `> s` are not comparable
> measurements of the same quantity. The record therefore stores the first mismatch's gap, and the
> rule reads it.
>
> **The $0 calibration that fixes `M` before the re-run, and can only tighten it.** Before the
> launch commit, `tests/test_kernel_path.py` gains the bf16 twin of its fp32 check (Task 11): the
> tiny model cast to bf16, the same 16 committed prompts × 16 greedy steps, kernel arm vs
> reconstruct twin, recording per step `ρ_s = max_j |L_kernel[s,j] − L_recon[s,j]| / (u · max_j
> |L_recon[s,j]|)` over the top-8 tokens. Let `ρ̂` be the worst `ρ_s` observed. Because the tiny
> model has `L_tiny` layers and Llama has 32, the pre-registered extrapolation is `κ = ρ̂ ·
> √(32 / L_tiny)`, rounded **up** to the next power of two.
> - If `κ ≤ 8`, `M` stands at `2⁻⁵ · max_j L[s,j]` as written above.
> - If `κ > 8`, this amendment is revised to `M = κ · u · max_j L[s,j]` **before** the launch
>   commit, with the measured `ρ̂`, `L_tiny` and `κ` printed in the revision.
> Both branches are fixed here, in advance, and the evidence is a committed CPU test — the
> calibration never sees an 8B number.
>
> #### A2.4 (c) The kernel arm's spikes
>
> **No threshold moves.** The kernel arm's `spikes` and `ms_max` are read exactly as §4 reads them
> (`prereg/kernel_smoke.md:237-240`): `> 8` of 56 refuses that arm's p50 as a steady-state number,
> and the 5 spikes / 1,992 ms max of instance 51903816 are **below that refusal**, so its p50
> stands as a steady-state number under the original rule — while its *cause* is unresolved and
> §5's prediction of "0 on the kernel" is, as measured, wrong. That is reported as a missed
> prediction, not repaired.
>
> **The recompile hypothesis is already refuted by reading, and is recorded so it is not re-opened.**
> `_tiles_kernel` (`src/kvdlra/kernel/triton_kernel.py:39-41`) declares **nothing length-dependent
> as `tl.constexpr`**: `n_dense`, `n_mid`, `tiles_per_split` and `n_splits` are runtime scalars;
> the `tl.constexpr` set is `H_KV, G, D, HALF, R, M, GP`, all fixed for a model, a rank and the
> 64-token tile. **There is therefore no per-absorb-event recompile**, and "every 16 tokens the
> tile count moves, so Triton rebuilds the kernel" is not the explanation. What remains is
> bounded: Triton specializes integer runtime arguments on divisibility-by-16 and equality-to-1,
> so a handful of distinct variants (the parity of `n_dense` as the recent ring fills, one
> `tiles_per_split` step, `n_splits` fixed at 16 for `b=1, H_kv=8` by `n_splits_for`) compile on
> **first touch** and never again.
>
> **The check on the re-run is the cheapest one that separates the two remaining candidates, and
> it perturbs nothing measured**: `kvdlra.eval.latency.run_latency` already keeps every step's time
> (`src/kvdlra/eval/latency.py:106-108`), so it prints one **log-only companion line** per cell,
> `[latency spikes ctx=<T> arm=<key> steps=<i,j,k,…>]`, giving the indices of the spiking steps
> within the 56-step steady window. No record field, no regex, no renderer change, no change to
> the arm under measurement. The two candidates have disjoint signatures and the line decides
> between them with no further work:
> - **first-touch JIT compilation** → indices cluster in the first few steps of the window and
>   never recur;
> - **the block-16 absorb** → indices recur on a 16-step lattice across the whole window, which
>   would refute ADR 0001 §5's "block-16 absorbs invalidate nothing in the kernel path"
>   (`docs/adr/0001-factored-attention-kernel.md:213-215`) and is a finding about the design, not
>   about Triton.
>
> Either way it is a **performance** observation. `TRITON_PRINT_AUTOTUNING` is not used: there is
> no `@triton.autotune` on this kernel. Pinning the four scalars via `do_not_specialize`, or
> lengthening the warm-up, is a **fix**, pre-registered nowhere here, and belongs to a later commit
> and a later pod — after the line above says which cause it would fix. `num_warps=4` stays as
> A1.4 leaves it (`prereg/kernel_smoke.md:595-598`): untuned, visible only in the kernel arm's
> ms/token, and no threshold adjusted for it.
>
> #### A2.5 The record fields this amendment needs
>
> The criteria of A2.2 and A2.3 are not computable from `KernelCheckRecord` as it stands
> (`src/kvdlra/eval/records.py:213-234`). Five fields are added, appended **before** the `error=`
> tail of the `[kernel_check prompt=…]` line exactly as `backend=` was (L4.fw1,
> `src/kvdlra/eval/kernel_check.py:111-114`), so every archived row still parses and every field
> that came first keeps its place:
>
> | field | meaning |
> |---|---|
> | `rel_max_diff` | `max_l ( max\|Δ_l\| / max\|ref_l\| )` — the quantity A2.2 reads |
> | `rel_worst_layer` | the layer achieving it |
> | `ref_max` | `max\|ref\|` at `rel_worst_layer` — the denominator, so the ratio is auditable |
> | `gap_at_mismatch` | the reconstruct path's top-1 minus top-2 logit at the first mismatching step; `-` when the prompt matched |
> | `kernel_logit_for_ref_argmax` | the kernel's logit for the token the reconstruct path chose at that step, beside the kernel's own top logit — the reversal's magnitude, and the evidence that a small gap really was a near-tie |
>
> The per-layer `max|ref|` values join the existing log-only `[kernel_check layers prompt=0 …]`
> breakdown (`src/kvdlra/eval/kernel_check.py:80-82`) as a `refs=` list beside `diffs=`; they are
> not records. `scripts/tables.py` `precondition_line` (`99ea29f:scripts/tables.py:1181-1197`)
> reads **both** bars and prints both numbers with the attributable and non-attributable mismatch
> counts named separately; `DIFF_MAX` stays in the file as the reported absolute number.
> Task 11 (§3 below) is the code, and it is CPU-tested before the launch commit.
>
> #### A2.6 (d) What this amendment does not change
>
> The Week-3 gate's two conditions and their thresholds — `kv_peak_gb(kernel, 32K, b) <
> kv_peak_gb(full, 32K, b)` and `ms_per_token_p50(reconstruct, 32K, b) / ms_per_token_p50(kernel,
> 32K, b) ≥ 3.0`, pass = both at batch 1 (§4 (1)-(3)); the 10 % marginal-memory rule and the 10 %
> archived-agreement rule (§2 (a), §4 (1)); the error refusal, the `spikes > 8` refusal and the
> "never `--`" rule (§4); §6 (no statistic, no correction); the arm list (`full`,
> `isvd_r64_h256_seed`, `isvd_r64_h256_seed_kernel`) and their configs; cell list A (batch 1,
> 16K / 32K / 64K, 64 decode steps of which 56 are timed) and cell list B's conditions; the
> `pre_run` gate and **every bar it applies** — `max|Δ| ≤ 2e-3 ∧ rms(Δ) ≤ 1e-4` at `n_splits == 1`,
> `≤ 4e-3` across splits and for batch independence, `< 1e-2` against reconstruct-then-attend on
> the random 8B shapes and the 1B dump layer (A1.2, A1's correction) — which are comparisons of the
> Triton kernel against the torch **reference**, at a fixed calibrated `|out|`, and are unaffected
> by anything above; §7's KIVI-2 deviation; §11. The 16 prompts, their source and their pins are
> unchanged. Amendment 1 A1.3's record and renderer stand as written, extended only by A2.5.
>
> #### A2.7 (e) Budget and sequencing
>
> - **Budget unchanged**: the point estimate stays `1.8 h + 0.45 h = 2.25 h` and the bar stays
>   `gpu_budget_h: 5.0` (A1.4, `prereg/kernel_smoke.md:577-584`); the five new fields add no GPU
>   work — `max|ref|` is one `.abs().max()` on a tensor `kernel_compare` already materializes, and
>   the logits are already computed by the greedy loop. At A1.4's $0.45–0.74/h that is
>   **≈ $1.0–1.7 expected, ≈ $2.3–3.7 at the bar**. The check axis's own wall clock, still the
>   first number to read against A1.4's 0.45 h allowance, now has instance 51903816's actual to
>   read against as well.
> - **Order**: this amendment's commit, then Task 11's commit, then the pod YAML's commit, then the
>   launch commit. `scripts/pod.py launch` machine-enforces only that the prereg's **first** commit
>   is a strict ancestor of the launch SHA (`prereg_error`, `99ea29f:scripts/pod.py:373-388`), so
>   an appended amendment's ordering is **not** machine-checked: the launch entry in
>   `docs/plan/DECISIONS.md` names this amendment's SHA explicitly and
>   `git merge-base --is-ancestor <amendment SHA> <launch SHA>` is the evidence, pasted.
> - **The re-run is a new pod label**, `kernel_smoke2` (`configs/pods/kernel_smoke2.yaml`, the
>   `kernel_smoke` YAML byte-for-byte with `name:` changed and `prereg: prereg/kernel_smoke.md`
>   kept), so it writes `results/kernel_smoke2/` and cannot overwrite instance 51903816's harvest
>   (`scripts/pod.py` §4: one directory per pod, `results/<pod>/`). Instance 51903816's records are
>   committed, harvested and cited as **REFUSED under the original bar**; nothing about them is
>   revised, deleted or re-read.
> - **Citability** is unchanged (CLAUDE.md): a number of the re-run is citable only once
>   `scripts/pod.py check results/kernel_smoke2` passes — including `_kernel_check_fails`'s 16 rows
>   per kernel arm (`scripts/pod.py:877-891`) and the three `pre_run` refusals — and
>   `make kernel_smoke` regenerates its table from the committed records.

---

## 3. Task 11 — the code the amendment needs (one screen)

**Scope.** Five record fields, one log-line extension, one CPU calibration test, one log-only
latency companion line. No change to the Triton kernel, no change to the arm under measurement,
no new dependency, no new file.

| # | file | change | lines |
|---|---|---|---|
| 1 | `src/kvdlra/kernel/attention.py:97-99` | `kernel_compare[layer]` stores `(max\|Δ\|, max\|ref\|)` instead of `max\|Δ\|` — one extra `.abs().max()` on `ref`, already materialized | +3 |
| 2 | `src/kvdlra/eval/kernel_check.py:32-61` | `_greedy` returns per-step `(argmax, top1, top2)`; **run the reconstruct twin first**, then the kernel, so the kernel's logit at the twin's argmax is a single `gather` at the mismatching step (no full logit row kept: 128 k floats × 32 steps would be 33 MB for nothing) | +12 |
| 3 | `src/kvdlra/eval/kernel_check.py:64-88` | `check_prompt` computes `rel_max_diff` / `rel_worst_layer` / `ref_max` over the `(d_l, m_l)` pairs (skipping `m_l == 0`, reported), and `gap_at_mismatch` / `kernel_logit_for_ref_argmax` at `first`; `refs=` added to the `[kernel_check layers prompt=0 …]` line | +18 |
| 4 | `src/kvdlra/eval/kernel_check.py:91-116` | the five fields in `failed_row` (all `None`) and in `format_line`, appended **before** `error=`, `-` when absent | +10 |
| 5 | `src/kvdlra/eval/records.py:106-110, 213-234, 489+` | five optional groups on `KERNEL_CHECK_RE`, five keys on `KernelCheckRecord`, five `_field`/float conversions in `parse_kernel_check_lines`; archived rows (instance 51903816's among them) parse with `None` | +18 |
| 6 | `scripts/tables.py:1139, 1181-1197` | `REL_MAX = 2**-6`, `M_COEF = 2**-5`; `precondition_line` reads both bars, splits mismatches into attributable / near-tie by `gap > M_COEF · top1`, prints `rel` with its layer, the absolute `max\|Δ\|` with its layer, and both counts; a row whose new fields are `None` is reported as "not computable (record predates Amendment 2)" and refuses, never silently passes | +28 |
| 7 | `src/kvdlra/eval/latency.py:106-133` | the `[latency spikes ctx=… arm=… steps=…]` companion line from the `times_ms` already held; no record field, no regex | +4 |
| 8 | `tests/test_kernel_path.py` | **the $0 bf16 calibration of A2.3**: the tiny model in bf16, 16 prompts × 16 steps, kernel vs reconstruct, asserting the measured `κ = ρ̂ · √(32/L_tiny)` rounded up to a power of two is `≤ 8`; the failure message prints `ρ̂`, `L_tiny` and `κ`, which is the revision branch's input | +35 |
| 9 | `tests/test_records.py`, `tests/test_kernel_smoke_pod.py` | round-trip `format_line` → `KERNEL_CHECK_RE` with and without the new fields (an archived row still parses); `precondition_line` unit cases: met, relative-bar miss, attributable-mismatch miss, near-tie-only miss, `None`-fields refusal | +45 |

**≈ 175 lines, all CPU.** Runtime: items 1–7 and 9 are pure-Python and add **< 2 s**; item 8 is
the only measurable cost — the tiny model in bf16 on CPU over 16 × 16 × 2 forwards, **≤ 25 s**
measured against the existing fp32 twin, which keeps `tests/` inside CLAUDE.md's < 90 s CI bar.
`pytest -m gpu` is untouched, so the `pre_run` gate and its bars are bit-identical.

**Not in this task, deliberately.** `do_not_specialize` on `_tiles_kernel`'s four runtime scalars,
and any warm-up change: A2.4 establishes that no constexpr is length-dependent, so there is no
constexpr fix to make, and the specialization/absorb question is *diagnosed* by item 7 before
anything is changed on the measured arm. A CPU guard asserting the four names stay runtime
arguments would have to `ast`-parse `triton_kernel.py` (the module cannot be imported without
triton, and triton is `sys_platform == 'linux'` in `uv.lock`) — ~12 lines to defend a property
nothing is proposing to change. Skipped; add it with the fix, if the fix happens.

---

## 4. The owner's decision

**(A) Amend + Task 11 + re-run.** Commit Amendment 2, commit Task 11, commit
`configs/pods/kernel_smoke2.yaml`, relaunch cell list A unchanged. ≈ $1.0–1.7 expected
(≈ $2.3–3.7 at the 5.0 h bar), ≈ half a day of lane time, ~175 CPU-tested lines. Outcome: a
Week-3 gate that is READ — met or genuinely missed — with both bars and both mismatch classes on
the record, and the four falsifiers of §1 (3) live.

**(B) Report REFUSED as the Week-3 outcome, defer the re-run to Phase 2.** $0 now. The memory and
throughput rows stay informative descriptively (kernel `kv_peak` 0.83 GB vs full 2.05 GB at 16K; a
3.2× p50 ratio against the reconstruct path at 16K — not at the gate's 32K), and D-002's choice
between ADR options (i) and (iii) stays open on a REFUSED precondition. Cost: every later kernel
pod inherits the same unreadable bar, and the paper's systems paragraph stays unwritten.

**(C) Amend only, no re-run** — commit Amendment 2 and Task 11, re-read instance 51903816's
records under the new bars. **This does not work and should be rejected on the record**:
`KernelCheckRecord` stored neither `max|ref|` nor any logit, so neither new criterion is
computable from those rows. There is no re-reading; there is only a re-run.

**Recommended: (A).** (1) (C)'s impossibility is the argument: this pod's REFUSED verdict is
permanent, so deferring does not preserve an option, it postpones the only measurement that can
resolve it. (2) The evidence that the kernel is correct is strong but circumstantial — disjoint
failing prompts, a 2.6× spread under identical shapes, a passing `pre_run` gate at 2e-3/1e-4 — and
circumstantial is exactly what a pre-registered relative bar converts into a reading. (3) The cost
is ≈ $2 and one lane-day against the gate that blocks the only systems claim in the paper.
(4) (A) does not skip (B): the REFUSED outcome entry for instance 51903816 is written either way,
and (A) adds the re-run on top of it. (5) The risk in (A) is the 175 lines, not the GPU — and all
175 are CPU-tested, with the `pre_run` gate and every bar it applies untouched.

**What the owner is being asked to approve**, precisely: the constant `2⁻⁶` in A2.2, the margin
`M = 2⁻⁵ · max_j L[s,j]` in A2.3 together with the `κ > 8` revision branch, the five record
fields, the new pod label `kernel_smoke2`, and the ≈ $2. Everything else is unchanged.
