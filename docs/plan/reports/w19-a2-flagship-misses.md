# Official RULER (Llama-3.1-8B, 16K): records missed by `bugSseed-r64-h256`

Depth = the generator's `token_position_answer / length`. Columns = hit (1) / miss (0) per arm on the SAME record.

| task | record | depth | `full` | `ea-k0.1` | `think-c0.5` | `palu-r0.5` | `quant-2bit-kivi` | `quant-4bit-kivi` | `bugSseed-r64-h256` |
|---|---|---|---|---|---|---|---|---|---|
| niah_multikey_1 | 11779 | 0.15 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| niah_multikey_1 | 15786 | 0.20 | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| niah_multikey_3 | 4991 | 0.17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| niah_multikey_3 | 24371 | 0.83 | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| niah_multiquery | 11675 | 0.16 | 1 | 0 | 1 | 0 | 1 | 1 | 0 |
| niah_multiquery | 15792 | 0.20 | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| niah_multiquery | 30401 | 0.41 | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| niah_multiquery | 68664 | 0.89 | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| niah_multivalue | 45862 | 0.59 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| niah_multivalue | 73062 | 0.95 | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| niah_single_1 | 29257 | 0.48 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| niah_single_1 | 35197 | 0.57 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| niah_single_3 | 13374 | 0.17 | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| niah_single_3 | 63447 | 0.82 | 1 | 0 | 1 | 1 | 0 | 1 | 0 |
| vt | 0 | n/a | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| vt | 1 | n/a | 1 | 0 | 1 | 1 | 0 | 1 | 0 |
| vt | 4 | n/a | 1 | 0 | 0 | 1 | 1 | 1 | 0 |
| vt | 5 | n/a | 1 | 1 | 1 | 1 | 0 | 1 | 0 |
| vt | 6 | n/a | 1 | 0 | 1 | 1 | 0 | 1 | 0 |
| vt | 7 | n/a | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| vt | 9 | n/a | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| vt | 11 | n/a | 1 | 0 | 1 | 1 | 1 | 1 | 0 |

## Miss depths per arm (all official cells)

- `full`: 1 misses at depths [0.17]
- `ea-k0.1`: 78 misses at depths [0.0, 0.03, 0.08, 0.08, 0.09, 0.13, 0.13, 0.13, 0.13, 0.16, 0.17, 0.17, 0.18, 0.2, 0.2, 0.2, 0.22, 0.22, 0.24, 0.25, 0.25, 0.25, 0.26, 0.28, 0.28, 0.3, 0.3, 0.33, 0.34, 0.34, 0.37, 0.37, 0.38, 0.41, 0.44, 0.46, 0.47, 0.52, 0.55, 0.55, 0.55, 0.56, 0.57, 0.58, 0.59, 0.59, 0.6, 0.64, 0.64, 0.64, 0.65, 0.68, 0.7, 0.72, 0.75, 0.75, 0.75, 0.75, 0.8, 0.81, 0.82, 0.82, 0.82, 0.82, 0.82, 0.83, 0.84, 0.84, 0.87, 0.89, 0.91, 0.93, 0.93, 0.93, 0.93, 0.95, 0.95, 0.99]
- `think-c0.5`: 1 misses at depths [0.17]
- `palu-r0.5`: 9 misses at depths [0.16, 0.17, 0.47, 0.59, 0.64, 0.7, 0.82, 0.84, 0.93]
- `quant-2bit-kivi`: 10 misses at depths [0.17, 0.22, 0.24, 0.26, 0.44, 0.47, 0.59, 0.64, 0.82, 0.93]
- `quant-4bit-kivi`: 1 misses at depths [0.17]
- `bugSseed-r64-h256`: 14 misses at depths [0.15, 0.16, 0.17, 0.17, 0.2, 0.2, 0.41, 0.48, 0.57, 0.59, 0.82, 0.83, 0.89, 0.95]
