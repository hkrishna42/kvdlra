# Kernel smoke — kernel_smoke

<!-- pre-registration: prereg/kernel_smoke.md; the reading is its §4, the table its §7 -->
<!-- source: kernel_smoke/latency.jsonl (9 rows), kernel_check.jsonl (16 rows), manifest errors 0; archived rows: w19-sysfix-llama-lines.txt -->
<!-- cell: one measurement per (arm, ctx, batch): p50 of 56 timed decode steps, spikes = steps > 2x p50, kv_* = VRAM minus the weights; stored GiB = bug_footprint(...).stored_bits() summed over the layers -- beside the measured peak, never instead of it (§7) -->
<!-- W20 = the archived Week-20 rows of §2 (a), re-measured by this pod's own full and reconstruct arms; a p50 within 10% of its archived value agrees, else the gap is reported -->

| arm | role | ctx | batch | ms/tok p50 | ms mean | ms max | spikes | resident_gb | peak_gb | kv_peak_gb | kv_resident_gb | stored GiB (analytic) | W20 p50 | W20 kv_peak | W20 agreement | backend |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | full | 16384 | 1 | 24.74 | 24.72 | 24.88 | 0 | 16.97 | 17.01 | 2.05 | 2.01 | n/a | 25.89 | 2.05 | archived 25.89 ms: agree (within 10%) | n/a |
| bugSseed-r64-h256 | reconstruct | 16384 | 1 | 126.03 | 140.40 | 336.20 | 4 | 15.78 | 18.21 | 3.25 | 0.83 | 0.302 | 103.25 | 3.25 | archived 103.25 ms: differs by +22% | n/a |
| isvd_r64_h256_seed_kernel | kernel | 16384 | 1 | 39.19 | 88.04 | 1991.99 | 5 | 15.79 | 15.79 | 0.83 | 0.83 | 0.302 | none archived | none archived | n/a | n/a |
| full | full | 32768 | 1 | 30.13 | 30.18 | 30.61 | 0 | 18.97 | 19.04 | 4.08 | 4.01 | n/a | 27.49 | 4.08 | archived 27.49 ms: agree (within 10%) | n/a |
| bugSseed-r64-h256 | reconstruct | 32768 | 1 | 248.30 | 265.71 | 507.32 | 3 | 16.03 | 21.40 | 6.45 | 1.08 | 0.556 | 188.27 | 6.45 | archived 188.27 ms: differs by +32% | n/a |
| isvd_r64_h256_seed_kernel | kernel | 32768 | 1 | 56.12 | 68.56 | 232.60 | 4 | 16.03 | 16.04 | 1.08 | 1.08 | 0.556 | none archived | none archived | n/a | n/a |
| full | full | 65536 | 1 | 42.23 | 42.29 | 43.26 | 0 | 22.97 | 23.10 | 8.14 | 8.01 | n/a | 36.97 | 8.14 | archived 36.97 ms: differs by +14% | n/a |
| bugSseed-r64-h256 | reconstruct | 65536 | 1 | 605.27 | 627.32 | 981.02 | 0 | 16.55 | 27.79 | 12.84 | 1.59 | 1.064 | 507.94 | 12.83 | archived 507.94 ms: differs by +19% | n/a |
| isvd_r64_h256_seed_kernel | kernel | 65536 | 1 | 88.31 | 100.62 | 263.05 | 4 | 16.55 | 16.57 | 1.61 | 1.59 | 1.064 | none archived | none archived | n/a | n/a |

PRECONDITION: NOT met (12/16 token-exact; worst max|d| 1.464e-02 at layer 26 >= 1e-2)
memory @32K b1: kernel kv_peak_gb 1.08 vs full 4.08 -> PASS (margin 73.5%)
speed @32K b1: reconstruct p50 248.30 ms / kernel p50 56.12 ms = 4.42x vs 3.0x -> PASS
WEEK-3 GATE (batch 1): REFUSED -- the correctness precondition is not met: a fast wrong kernel is not a result (§4)
