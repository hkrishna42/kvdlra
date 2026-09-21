# Kernel smoke — kernel_smoke2

<!-- pre-registration: prereg/kernel_smoke.md; the reading is its §4, the table its §7 -->
<!-- source: kernel_smoke2/latency.jsonl (9 rows), kernel_check.jsonl (16 rows), manifest errors 0; archived rows: w19-sysfix-llama-lines.txt -->
<!-- cell: one measurement per (arm, ctx, batch): p50 of 56 timed decode steps, spikes = steps > 2x p50, kv_* = VRAM minus the weights; stored GiB = bug_footprint(...).stored_bits() summed over the layers -- beside the measured peak, never instead of it (§7) -->
<!-- W20 = the archived Week-20 rows of §2 (a), re-measured by this pod's own full and reconstruct arms; a p50 within 10% of its archived value agrees, else the gap is reported -->

| arm | role | ctx | batch | ms/tok p50 | ms mean | ms max | spikes | resident_gb | peak_gb | kv_peak_gb | kv_resident_gb | stored GiB (analytic) | W20 p50 | W20 kv_peak | W20 agreement | backend |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | full | 16384 | 1 | 36.98 | 37.00 | 37.49 | 0 | 16.97 | 17.01 | 2.05 | 2.01 | n/a | 25.89 | 2.05 | archived 25.89 ms: differs by +43% | n/a |
| bugSseed-r64-h256 | reconstruct | 16384 | 1 | 103.84 | 122.50 | 369.01 | 4 | 15.79 | 18.21 | 3.25 | 0.83 | 0.302 | 103.25 | 3.25 | archived 103.25 ms: agree (within 10%) | n/a |
| isvd_r64_h256_seed_kernel | kernel | 16384 | 1 | 56.85 | 141.85 | 3688.31 | 5 | 15.79 | 15.79 | 0.83 | 0.83 | 0.302 | none archived | none archived | n/a | triton |
| full | full | 32768 | 1 | 37.84 | 37.85 | 40.32 | 0 | 18.97 | 19.04 | 4.08 | 4.01 | n/a | 27.49 | 4.08 | archived 27.49 ms: differs by +38% | n/a |
| bugSseed-r64-h256 | reconstruct | 32768 | 1 | 188.90 | 210.17 | 492.48 | 4 | 16.03 | 21.40 | 6.45 | 1.08 | 0.556 | 188.27 | 6.45 | archived 188.27 ms: agree (within 10%) | n/a |
| isvd_r64_h256_seed_kernel | kernel | 32768 | 1 | 57.61 | 77.55 | 330.95 | 4 | 16.03 | 16.03 | 1.08 | 1.08 | 0.556 | none archived | none archived | n/a | triton |
| full | full | 65536 | 1 | 37.76 | 37.78 | 38.31 | 0 | 22.97 | 23.10 | 8.14 | 8.01 | n/a | 36.97 | 8.14 | archived 36.97 ms: agree (within 10%) | n/a |
| bugSseed-r64-h256 | reconstruct | 65536 | 1 | 510.95 | 534.95 | 911.01 | 0 | 16.55 | 27.79 | 12.83 | 1.59 | 1.064 | 507.94 | 12.83 | archived 507.94 ms: agree (within 10%) | n/a |
| isvd_r64_h256_seed_kernel | kernel | 65536 | 1 | 85.23 | 102.50 | 331.36 | 4 | 16.55 | 16.55 | 1.59 | 1.59 | 1.064 | none archived | none archived | n/a | triton |

PRECONDITION: NOT met (16/16 effective (0 attributable, 5 near-tie mismatches); rel 1.868e-02 at layer 8 > 2^-6; max|d| 1.464e-02 at layer 26 (reported))
memory @32K b1: kernel kv_peak_gb 1.08 vs full 4.08 -> PASS (margin 73.5%)
speed @32K b1: reconstruct p50 188.90 ms / kernel p50 57.61 ms = 3.28x vs 3.0x -> PASS
WEEK-3 GATE (batch 1): REFUSED -- the correctness precondition is not met: a fast wrong kernel is not a result (§4)
