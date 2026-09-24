# Gate 1 — the tracker swap

<!-- pre-registration: prereg/gate1_tracker_swap_v2.md; the rule is its section 4 -->
<!-- source: llama=gate1_v2_stage1_llama, qwen=gate1_v2_stage1_qwen -->
<!-- cell: acc [Wilson 95% lo,hi] (hits/n), marked per the legend below each block -->
<!-- Holm at alpha=0.05 over each family's raw p-values, at the realised m: primary retrieval m=12, secondary retrieval m=18, primary perplexity m=4 -->
<!-- delta: isvd MINUS the row's tracker, paired per window; delta < 0 is the r64 arm ahead -->
<!-- TOST: two one-sided t-tests at +/-0.02 bits/token on the per-window differences, alpha=0.05 uncorrected (intersection-union) -->
<!-- the verdict reads the ctx 16384 contrasts only; any other context length is descriptive -->

<!-- arms: full=full, isvd=bugSseed-r64-h256, nogist=nogist_h2423, frozen=frozen_r64_h256_seed, fd=bugSseed-r64-h256-fd, bf16=isvd_r64_h256_seed_bf16, oja=oja_r64_h256_seed_tuned, random=random_r64_h256_seed -->

## llama — ctx 16384

| tracker | niah_single | niah_multikey | niah_multivalue | vt |
| --- | --- | --- | --- | --- |
| full | 1.00 [0.86,1.00] (24/24) | 1.00 [0.86,1.00] (24/24) | 0.96 [0.80,0.99] (23/24) | 0.71 [0.51,0.85] (17/24) |
| isvd | 0.79 [0.60,0.91] (19/24) | 0.42 [0.24,0.61] (10/24) | 0.25 [0.12,0.45] (6/24) | 0.42 [0.24,0.61] (10/24) |
| nogist | 1.00 [0.86,1.00] (24/24) | 0.92 [0.74,0.98] (22/24) ‡ | 0.88 [0.69,0.96] (21/24) ‡ | 0.83 [0.64,0.93] (20/24) |
| frozen | 0.54 [0.35,0.72] (13/24) | 0.50 [0.31,0.69] (12/24) | 0.25 [0.12,0.45] (6/24) | 0.38 [0.21,0.57] (9/24) |
| fd | 0.75 [0.55,0.88] (18/24) | 0.38 [0.21,0.57] (9/24) | 0.25 [0.12,0.45] (6/24) | 0.50 [0.31,0.69] (12/24) |
| bf16 | 0.75 [0.55,0.88] (18/24) | 0.29 [0.15,0.49] (7/24) | 0.21 [0.09,0.40] (5/24) | 0.42 [0.24,0.61] (10/24) |
| oja | 0.71 [0.51,0.85] (17/24) | 0.46 [0.28,0.65] (11/24) | 0.25 [0.12,0.45] (6/24) | 0.54 [0.35,0.72] (13/24) |
| random | 0.00 [0.00,0.14] (0/24) | 0.00 [0.00,0.14] (0/24) | 0.00 [0.00,0.14] (0/24) | 0.00 [0.00,0.14] (0/24) |

Legend: `*` = Holm-significant primary contrast favouring isvd; `‡` = one favouring the control (rule 3 (i) reads a separation in either direction); a cell with error rows reads `FAILED (k errors)` and an arm that never ran that cell says so -- no arm is ever printed as `--`.

### llama — perplexity, ctx 16384, pg19-val

| tracker | bits/token | delta (isvd - tracker) | 95% CI | TOST +/-0.02 | Holm p | paired t p |
| --- | --- | --- | --- | --- | --- | --- |
| full | 3.5592 | +0.1452 | [+0.1162, +0.1784] | not decidable | n/a (secondary) | 3.45e-10 |
| isvd | 3.7043 | reference | reference | reference | reference | reference |
| nogist | 3.6511 | +0.0533 | [+0.0386, +0.0700] | fails | 9.54e-07 | 2.38e-07 |
| frozen | 3.7066 | -0.0023 | [-0.0071, +0.0026] | passes | 0.376 | 0.376 |
| fd | 3.7784 | -0.0741 | [-0.0912, -0.0570] | fails | n/a (secondary) | 2.12e-09 |
| bf16 | 3.7051 | -0.0007 | [-0.0044, +0.0030] | passes | n/a (secondary) | 0.702 |
| oja | 3.7489 | -0.0445 | [-0.0615, -0.0284] | fails | n/a (secondary) | 1.42e-05 |
| random | 10.3215 | -6.6172 | [-6.8647, -6.4195] | not decidable | n/a (secondary) | 9.01e-33 |

<!-- arms: full=full, isvd=bugSseed-r64-h256, nogist=nogist_h4460, frozen=frozen_r64_h256_seed, fd=bugSseed-r64-h256-fd, bf16=isvd_r64_h256_seed_bf16, oja=oja_r64_h256_seed_tuned, random=random_r64_h256_seed -->

## qwen — ctx 16384

| tracker | niah_single | niah_multikey | niah_multivalue | vt |
| --- | --- | --- | --- | --- |
| full | 0.96 [0.80,0.99] (23/24) | 1.00 [0.86,1.00] (24/24) | 0.50 [0.31,0.69] (12/24) | 0.67 [0.47,0.82] (16/24) |
| isvd | 0.83 [0.64,0.93] (20/24) | 0.79 [0.60,0.91] (19/24) | 0.21 [0.09,0.40] (5/24) | 0.29 [0.15,0.49] (7/24) |
| nogist | 0.96 [0.80,0.99] (23/24) | 1.00 [0.86,1.00] (24/24) | 0.42 [0.24,0.61] (10/24) | 0.54 [0.35,0.72] (13/24) |
| frozen | 0.75 [0.55,0.88] (18/24) | 0.67 [0.47,0.82] (16/24) | 0.17 [0.07,0.36] (4/24) | 0.25 [0.12,0.45] (6/24) |
| fd | 0.62 [0.43,0.79] (15/24) | 0.38 [0.21,0.57] (9/24) | 0.08 [0.02,0.26] (2/24) | 0.12 [0.04,0.31] (3/24) |
| bf16 | 0.83 [0.64,0.93] (20/24) | 0.71 [0.51,0.85] (17/24) | 0.17 [0.07,0.36] (4/24) | 0.25 [0.12,0.45] (6/24) |
| oja | 0.42 [0.24,0.61] (10/24) | 0.29 [0.15,0.49] (7/24) | 0.08 [0.02,0.26] (2/24) | 0.29 [0.15,0.49] (7/24) |
| random | 0.00 [0.00,0.14] (0/24) | 0.00 [0.00,0.14] (0/24) | 0.00 [0.00,0.14] (0/24) | 0.00 [0.00,0.14] (0/24) |

Legend: `*` = Holm-significant primary contrast favouring isvd; `‡` = one favouring the control (rule 3 (i) reads a separation in either direction); a cell with error rows reads `FAILED (k errors)` and an arm that never ran that cell says so -- no arm is ever printed as `--`.

### qwen — perplexity, ctx 16384, pg19-val

| tracker | bits/token | delta (isvd - tracker) | 95% CI | TOST +/-0.02 | Holm p | paired t p |
| --- | --- | --- | --- | --- | --- | --- |
| full | 3.7052 | +0.1015 | [+0.0905, +0.1131] | fails | n/a (secondary) | 2.71e-17 |
| isvd | 3.8068 | reference | reference | reference | reference | reference |
| nogist | 3.7878 | +0.0190 | [+0.0038, +0.0337] | fails | 0.0435 | 0.0217 |
| frozen | 3.8205 | -0.0137 | [-0.0224, -0.0047] | fails | 0.0167 | 0.00558 |
| fd | 3.9007 | -0.0939 | [-0.1098, -0.0772] | fails | n/a (secondary) | 1.69e-12 |
| bf16 | 3.7965 | +0.0102 | [+0.0031, +0.0176] | passes | n/a (secondary) | 0.0123 |
| oja | 3.8778 | -0.0710 | [-0.0867, -0.0548] | fails | n/a (secondary) | 1.22e-09 |
| random | 11.0329 | -7.2261 | [-7.5297, -6.9182] | not decidable | n/a (secondary) | 9.06e-30 |

## bf16 non-inferiority — the gist stored at 16 bits
<!-- pre-registration: prereg/bf16_gist.md; the reading is its §4, and no member of it enters a Gate-1 family or moves a Gate-1 p-value (its §6) -->
<!-- Holm at alpha=0.05 inside this file's own retrieval family, at the realised m=6 of the 6 members §6 fixes (2 model families x the tasks left after the pre-flight's exclusions) -->
<!-- delta: isvd MINUS bf16, so delta > 0 is the bf16 arm losing; the TOST is the same +/-0.02 bits/token one, per family -->

| family | ctx | niah_single | niah_multikey | niah_multivalue | TOST | sbits bf16/isvd | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama | 16384 | +0.042 [2/1 of 24] 1 | +0.125 [3/0 of 24] 1 | +0.042 [1/0 of 24] 1 | passes (-0.0007) | 0.5667 | PASS |
| qwen | 16384 | +0.000 [0/0 of 24] 1 | +0.083 [2/0 of 24] 1 | +0.042 [1/0 of 24] 1 | passes (+0.0102) | 0.5382 | PASS |

Legend (bf16): cell = `delta [a_favored/b_favored of n_paired] Holm p`, with a = `isvd_r64_h256_seed` and b = `isvd_r64_h256_seed_bf16`, so `delta > 0` is the bf16 arm losing pairs; a task is non-inferior unless `delta` > 0.03 AND its Holm p < 0.05, both (§4 (1)). `sbits` is §7 (c)'s pin: outside 1 % of the expected ratio refuses the family.

- llama: PASS -- 3 tasks non-inferior (worst point estimate +0.125, margin 0.03) and the +/-0.02 TOST passes (-0.0007 bits)
- qwen: PASS -- 3 tasks non-inferior (worst point estimate +0.083, margin 0.03) and the +/-0.02 TOST passes (+0.0102 bits)

BF16: pass — llama PASS, qwen PASS; Holm at the realised m=6 of the 6 members prereg/bf16_gist.md section 6 fixes; section 4's consequence fires: every byte-matched control is re-planned at the new bytes before it is re-run

VERDICT: UNDECIDED — nothing selected; C blocked by: llama: isvd vs fd TOST at +/-0.02 does not pass (d=-0.0741 bits, p=1); qwen: isvd vs frozen TOST at +/-0.02 does not pass (d=-0.0137 bits, p=0.09); qwen/niah_multikey: isvd vs fd separates (Holm p=0.027); qwen: isvd vs fd TOST at +/-0.02 does not pass (d=-0.0939 bits, p=1)

members:
- llama: isvd vs fd TOST at +/-0.02 does not pass (d=-0.0741 bits, p=1)
- qwen: isvd vs frozen TOST at +/-0.02 does not pass (d=-0.0137 bits, p=0.09)
- qwen/niah_multikey: isvd vs fd separates (Holm p=0.027)
- qwen: isvd vs fd TOST at +/-0.02 does not pass (d=-0.0939 bits, p=1)
