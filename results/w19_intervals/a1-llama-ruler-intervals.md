# Week-18 RULER intervals

Cell: `acc [lo,hi] (hits/n)`.


## ctx 16384

| arm | mem | needle | multi-key | multi-value | var-track |
|---|---|---|---|---|---|
| bugS-r64-h256-q4 | 0.085x | 1.00 [0.76,1.00] (12/12) | 0.67 [0.39,0.86] (8/12) | 0.00 [0.00,0.24] (0/12) | 0.00 [0.00,0.24] (0/12) |
| bugSseed-r64-h256 | 0.085x | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.58 [0.32,0.81] (7/12) |
| quant-2bit-kivi | 0.163x | 1.00 [0.76,1.00] (12/12) | 0.67 [0.39,0.86] (8/12) | 0.42 [0.19,0.68] (5/12) | 0.67 [0.39,0.86] (8/12) |
| quant-4bit-kivi | 0.287x | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) |
| quant-8bit-kivi-hqq | 0.535x | 1.00 [0.51,1.00] (4/4) | 1.00 [0.51,1.00] (4/4) | 1.00 [0.51,1.00] (4/4) | 1.00 [0.51,1.00] (4/4) |

## ctx 32768

| arm | mem | needle | multi-key | multi-value | var-track |
|---|---|---|---|---|---|
| bugS-r64-h256-q4 | 0.075x | 1.00 [0.76,1.00] (12/12) | 0.83 [0.55,0.95] (10/12) | 1.00 [0.76,1.00] (12/12) | 0.83 [0.55,0.95] (10/12) |
| bugSseed-r64-h256 | 0.075x | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.92 [0.65,0.98] (11/12) |
| quant-2bit-kivi | 0.160x | 1.00 [0.76,1.00] (12/12) | 0.83 [0.55,0.95] (10/12) | 0.92 [0.65,0.98] (11/12) | 0.92 [0.65,0.98] (11/12) |
| quant-4bit-kivi | 0.284x | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) |

## McNemar contrasts (exact, paired per-trial)

| A | B | ctx | task | n | A>B | B>A | p | sig |
|---|---|---|---|---|---|---|---|---|
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_single | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_multikey | 12 | 4 | 0 | 0.1250 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_multivalue | 12 | 7 | 0 | 0.0156 | YES |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | vt | 12 | 3 | 4 | 1.0000 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 32768 | niah_single | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 32768 | niah_multikey | 12 | 2 | 0 | 0.5000 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 32768 | niah_multivalue | 12 | 1 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 32768 | vt | 12 | 1 | 1 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_single | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_multikey | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_multivalue | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | vt | 12 | 0 | 5 | 0.0625 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 32768 | niah_single | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 32768 | niah_multikey | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 32768 | niah_multivalue | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 32768 | vt | 12 | 0 | 1 | 1.0000 | no |
