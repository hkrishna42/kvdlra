# Week-18 RULER intervals

Cell: `acc [lo,hi] (hits/n)`.


## ctx 16384

| arm | mem | needle | multi-key | multi-value | var-track |
|---|---|---|---|---|---|
| bugSseed-r64-h256 | 0.085x | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.50 [0.25,0.75] (6/12) |
| quant-2bit-kivi | 0.163x | 0.92 [0.65,0.98] (11/12) | 0.58 [0.32,0.81] (7/12) | 0.50 [0.25,0.75] (6/12) | 0.33 [0.14,0.61] (4/12) |
| quant-4bit-kivi | 0.287x | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.25 [0.09,0.53] (3/12) |
| quant-8bit-kivi-hqq | 0.535x | 1.00 [0.51,1.00] (4/4) | 1.00 [0.51,1.00] (4/4) | 1.00 [0.51,1.00] (4/4) | 0.25 [0.05,0.70] (1/4) |

## ctx 32768

| arm | mem | needle | multi-key | multi-value | var-track |
|---|---|---|---|---|---|
| bugSseed-r64-h256 | 0.075x | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.83 [0.55,0.95] (10/12) | 0.42 [0.19,0.68] (5/12) |
| quant-2bit-kivi | 0.160x | 0.83 [0.55,0.95] (10/12) | 0.25 [0.09,0.53] (3/12) | 0.08 [0.01,0.35] (1/12) | 0.00 [0.00,0.24] (0/12) |
| quant-4bit-kivi | 0.284x | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.25 [0.09,0.53] (3/12) |

## McNemar contrasts (exact, paired per-trial)

| A | B | ctx | task | n | A>B | B>A | p | sig |
|---|---|---|---|---|---|---|---|---|
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_single | 12 | 1 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_multikey | 12 | 5 | 0 | 0.0625 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_multivalue | 12 | 6 | 0 | 0.0312 | YES |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | vt | 12 | 2 | 0 | 0.5000 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 32768 | niah_single | 12 | 2 | 0 | 0.5000 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 32768 | niah_multikey | 12 | 9 | 0 | 0.0039 | YES |
| bugSseed-r64-h256 | quant-2bit-kivi | 32768 | niah_multivalue | 12 | 9 | 0 | 0.0039 | YES |
| bugSseed-r64-h256 | quant-2bit-kivi | 32768 | vt | 12 | 5 | 0 | 0.0625 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_single | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_multikey | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_multivalue | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | vt | 12 | 3 | 0 | 0.2500 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 32768 | niah_single | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 32768 | niah_multikey | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 32768 | niah_multivalue | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 32768 | vt | 12 | 2 | 0 | 0.5000 | no |
