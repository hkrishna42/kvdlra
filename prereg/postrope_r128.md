# Pre-registration — `postrope_r128` (ADR 0001 option (i): post-RoPE tracking at r = 128)

**STATUS: written before the pod exists; launch is a later DECISIONS line under D-011, after the
Gate-1 Stage-1 Llama harvest is committed.** Lane L4 Task 9. This file is committed strictly
before the launch commit; `scripts/pod.py launch` refuses otherwise. It launches nothing.

## 1. Purpose

ADR 0001 §4: "(i) runs in parallel as an accuracy experiment only — config
`isvd_postrope_r128`, one Gate-1-style row on Llama 16K, n = 24. It is not a kernel fallback …
it is insurance if (iii) misses its FLOP target on some GPU." The pre-RoPE basis "roughly
halv[es] the reconstruction error at matched rank" (paper §2.2), and the surprise score that
admits tokens to the exact tier is computed against the tracked basis, so a post-RoPE basis
"changes what reads as surprising — retrieval, not just perplexity, must be re-measured" (ADR
§3). **The question:** at 16K on Llama-3.1-8B, is post-RoPE tracking at r = 128 non-inferior
to the shipped pre-RoPE r = 64 configuration on the four Gate-1 retrieval tasks and on
perplexity — at twice the coordinate bytes?

## 2. Measured baseline

- The comparison rows are the Gate-1 Stage-1 Llama pod's (`prereg/gate1_tracker_swap_v2.md`
  §3 arms 1–2: `full`, `isvd_r64_h256_seed`) on the same task files; their numbers are that
  pod's to report. The prior from the pre-flight (D-011 addenda 10–11): the r64 configuration
  retrieves 0.83 on `niah_single` at 16K on real documents; the `full` ceiling is 1.00 / 1.00
  / 0.92 / 0.92 on the four tasks with `vt` at 0.75 (D-018: `vt` is descriptive).
- Stored bits (`kvdlra.accounting.bug_footprint(...).stored_bits()`, ADR §3): r128 post-RoPE
  0.282× of the fp16 cache at 16K against r64's 0.151× — printed beside every row; this row
  buys accuracy with bytes and says so.
- The knob is exact at full rank (`tests/test_rope_basis.py`: DynamicCache parity, the middle's
  reconstruction is the stored post-RoPE key) and is not a no-op at low rank.

## 3. Arms, tasks, n

One arm, one pod: `isvd_postrope_r128` (`configs/arms/isvd_postrope_r128.yaml`) = arm 2 of
the Gate-1 design with `rank: 128` and `rope_basis: "post"`; everything else — the 256-token
surprise tier, 4 sinks, ring 32, absorb 16, the warm-up seed, the shipped guard — is arm 2's.
Tasks `ruler_v2_16k_g1` (the four Gate-1 tasks, n = 24 from one seed on the 2 × 3 × 4 design)
and `ppl_16k_pg19val` (32 non-overlapping 2048-token windows), chunk 4096, on
`unsloth/Meta-Llama-3.1-8B-Instruct` in bf16 — the same task files, seed and model as
`gate1_v2_stage1_llama`, so the generator builds byte-identical prompts (`prompt_sha256`) and
`frontier.windows` cuts identical windows (`window_idx`). **Pairing across the two pods** is
the pre-flight re-run's rule (D-011 addendum 10, reading (iii)): a retrieval key pairs only
when the two records' `prompt_sha256` agree; a mismatch drops the key from every paired
statistic and is reported with the key; a perplexity window pairs on `window_idx` within the
corpus `pg19-val`, and a window set that is not exactly Stage 1's refuses the TOST
(`records.paired_window_bits`).

## 4. Reading and decision rule

Per task (`niah_single`, `niah_multikey`, `niah_multivalue`; `vt` descriptive per D-018): exact
paired McNemar (`kvdlra.eval.stats.mcnemar_exact`) of `isvd_postrope_r128` (b) against
`isvd_r64_h256_seed` (a) on the paired keys, Holm over the m = 3 tasks (`stats.holm`); a task is
**lost** only when both the raw loss `(a_favored − b_favored) / n_paired > 0.03` AND Holm
p < 0.05 (the `prereg/bf16_gist.md` §4 rule). Perplexity: the paired-window TOST at ±0.02
bits/token (`stats.tost`, α = 0.05) of the post arm against the r64 arm, and descriptively
against `full`. **Non-inferior** = no task lost AND the TOST passes. **Refusals:** any error
row on either arm in a paired cell; a broken pairing; a missing Stage-1 record; a TOST that is
not decidable (`stats.tost_decidable`). No `--`: an arm that did not run reads `not run` with
the reason.

## 5. Predictions

Perplexity: within ±0.02 bits of r64 (the ADR's "roughly halves the error at matched rank"
predicts r128 post ≈ r64 pre on reconstruction). Retrieval: not predicted in either direction
— the surprise selection against a post-RoPE basis is the ADR's open risk; `niah_multivalue`
is where a change would show first (the Week-12 mechanism sits in the exact tier).

## 6. Family size

One Holm family of m = 3 (the three non-`vt` retrieval tasks); the TOST is an
intersection-union test at α = 0.05; nothing else is corrected. `vt` is descriptive.

## 7. Secondary outcomes

The stored-bits ratio of both arms; per-task raw accuracies with Wilson intervals; the
`[diag]` rows (guard repairs under the post basis); the Stage-1 `full` rows beside both.

## 8. Log volume

128 samples: 96 `[trial]` lines, 4 cell rows, one 32-window `[pplw]` group (4 `part=i/N`
fragments), one `ppl=` line, ≈ 416 `[diag]` rows per 16K sample (`prereg/gate1_tracker_swap_v2.md`
§8: 13 rows per layer × 32 layers) ≈ 53,000 rows over 128 samples, in per-sample bursts of 416,
plus ≈ 40 `[stage]` lines: well inside the watchdog's 30,000-line poll.

## 9. Budget

128 samples at ≤ 5.0 min/sample (the r128 rate bounded above by the measured r256 rate of 5.2
min, D-011 addendum 2, and below by r64's 3.3 min, addendum 10) = 640 min = 10.7 h, + 60 min
boot = **11.7 h point; `gpu_budget_h: 24.0`**, the 2× bar `pod.py launch --max-hours`
enforces. At $0.45–0.74/h: $5.3–8.7 expected, $10.8–17.8 at the bar. Overrun is a
stop-and-report. The three-arm alternative (repeating `full` and r64 in-pod) would bar at
≈ 39 h (18.6 h compute at §9's rates + 1 h boot, ×2; ≈ $18–29) for readings Stage 1 already
buys; rejected.

## 10. Provenance

Pod `configs/pods/postrope_r128.yaml` (this file as `prereg`, one arm, the two Gate-1 task
files, bar 24.0), arm `configs/arms/isvd_postrope_r128.yaml`, the knob in
`src/kvdlra/cache/bug_cache.py` (`rope_basis`), pins `tests/test_rope_basis.py` and
`tests/test_pod_manifest.py`. Outputs `results/postrope_r128/{manifest.json, trials.jsonl,
ppl.jsonl, pplw.jsonl, diag.jsonl, env.txt}`; the reading is rendered by `scripts/tables.py
gate1 --pods results/gate1_v2_stage1_llama results/postrope_r128` only if `gate1.load` can
take a single-arm pod — otherwise a `postrope` renderer is a later lane's task and is named
in the launch entry. Launch: `scripts/pod.py launch --pod postrope_r128 --offer <id>` from a
pushed clean SHA that descends from this file's first commit AND from the commit that adds
`results/gate1_v2_stage1_llama/trials.jsonl`; a DECISIONS line under D-011 with SHAs, offer,
rate, bar and credit. The launch entry also names the Stage-1 Llama harvest commit this pod
is paired against (the commit that added `results/gate1_v2_stage1_llama/trials.jsonl`).
Amendments only, never edits.

## 11. What this does not decide

Whether (iii) is the kernel (that is D-002 and `prereg/kernel_smoke.md`); anything at 32K, on
Qwen or Mistral, or at any rank but 128; the kernel's cost model. A pass makes (i) a viable
simplification at 2× the coordinate bytes for D-002 to weigh; a fail retires (i).
