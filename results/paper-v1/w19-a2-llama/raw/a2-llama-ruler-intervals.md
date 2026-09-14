# Week-18 RULER intervals

Cell: `acc [lo,hi] (hits/n)`.


## ctx 16384

| arm | mem | multi-value | var-track |
|---|---|---|---|
| bugSseed-r64-h256 | 0.086x | 0.83 [0.55,0.95] (10/12) | 0.33 [0.14,0.61] (4/12) |
| ea-k0.1 | 0.100x | 0.00 [0.00,0.24] (0/12) | 0.33 [0.14,0.61] (4/12) |
| full | 1.000x | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) |
| palu-r0.5 | 0.504x | 0.42 [0.19,0.68] (5/12) | 1.00 [0.76,1.00] (12/12) |
| quant-2bit-kivi | 0.163x | 0.83 [0.55,0.95] (10/12) | 0.67 [0.39,0.86] (8/12) |
| quant-4bit-kivi | 0.287x | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) |
| think-c0.5 | 0.750x | 1.00 [0.76,1.00] (12/12) | 0.92 [0.65,0.98] (11/12) |

## McNemar contrasts (exact, paired per-trial)

| A | B | ctx | task | n | A>B | B>A | p | sig |
|---|---|---|---|---|---|---|---|---|
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_multivalue | 12 | 1 | 1 | 1.0000 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | vt | 12 | 1 | 5 | 0.2188 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_multikey_1 | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_multikey_2 | 12 | 1 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_multikey_3 | 12 | 4 | 1 | 0.3750 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_multiquery | 12 | 0 | 4 | 0.1250 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_single_1 | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_single_2 | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-2bit-kivi | 16384 | niah_single_3 | 12 | 1 | 1 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_multivalue | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | vt | 12 | 0 | 8 | 0.0078 | YES |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_multikey_1 | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_multikey_2 | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_multikey_3 | 12 | 0 | 1 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_multiquery | 12 | 0 | 4 | 0.1250 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_single_1 | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_single_2 | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | quant-4bit-kivi | 16384 | niah_single_3 | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | ea-k0.1 | 16384 | niah_multivalue | 12 | 10 | 0 | 0.0020 | YES |
| bugSseed-r64-h256 | ea-k0.1 | 16384 | vt | 12 | 2 | 2 | 1.0000 | no |
| bugSseed-r64-h256 | ea-k0.1 | 16384 | niah_multikey_1 | 12 | 7 | 1 | 0.0703 | no |
| bugSseed-r64-h256 | ea-k0.1 | 16384 | niah_multikey_2 | 12 | 12 | 0 | 0.0005 | YES |
| bugSseed-r64-h256 | ea-k0.1 | 16384 | niah_multikey_3 | 12 | 10 | 0 | 0.0020 | YES |
| bugSseed-r64-h256 | ea-k0.1 | 16384 | niah_multiquery | 12 | 8 | 0 | 0.0078 | YES |
| bugSseed-r64-h256 | ea-k0.1 | 16384 | niah_single_1 | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | ea-k0.1 | 16384 | niah_single_2 | 12 | 10 | 0 | 0.0020 | YES |
| bugSseed-r64-h256 | ea-k0.1 | 16384 | niah_single_3 | 12 | 10 | 0 | 0.0020 | YES |
| bugSseed-r64-h256 | palu-r0.5 | 16384 | niah_multivalue | 12 | 6 | 1 | 0.1250 | no |
| bugSseed-r64-h256 | palu-r0.5 | 16384 | vt | 12 | 0 | 8 | 0.0078 | YES |
| bugSseed-r64-h256 | palu-r0.5 | 16384 | niah_multikey_1 | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | palu-r0.5 | 16384 | niah_multikey_2 | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | palu-r0.5 | 16384 | niah_multikey_3 | 12 | 0 | 1 | 1.0000 | no |
| bugSseed-r64-h256 | palu-r0.5 | 16384 | niah_multiquery | 12 | 0 | 3 | 0.2500 | no |
| bugSseed-r64-h256 | palu-r0.5 | 16384 | niah_single_1 | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | palu-r0.5 | 16384 | niah_single_2 | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | palu-r0.5 | 16384 | niah_single_3 | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | think-c0.5 | 16384 | niah_multivalue | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | think-c0.5 | 16384 | vt | 12 | 0 | 7 | 0.0156 | YES |
| bugSseed-r64-h256 | think-c0.5 | 16384 | niah_multikey_1 | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | think-c0.5 | 16384 | niah_multikey_2 | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | think-c0.5 | 16384 | niah_multikey_3 | 12 | 0 | 1 | 1.0000 | no |
| bugSseed-r64-h256 | think-c0.5 | 16384 | niah_multiquery | 12 | 0 | 4 | 0.1250 | no |
| bugSseed-r64-h256 | think-c0.5 | 16384 | niah_single_1 | 12 | 0 | 2 | 0.5000 | no |
| bugSseed-r64-h256 | think-c0.5 | 16384 | niah_single_2 | 12 | 0 | 0 | 1.0000 | no |
| bugSseed-r64-h256 | think-c0.5 | 16384 | niah_single_3 | 12 | 0 | 2 | 0.5000 | no |
