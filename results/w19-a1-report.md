# Week-19 A1 — the fair 2-bit baseline, at matched stored bytes

KIVI-faithful quantized KV (per-channel keys, per-token values; `--quant-scheme kivi`, quanto backend, g=64, residual 128; 8-bit control on the hqq backend) vs the flagship `bugSseed-r64-h256`, same harness, same needles (paired), n=12 per cell (8-bit control n=4). Stored ratios are the honest stored-bits billing (BUG's fp32-at-rest state included; quant aux at its measured bf16 dtype). Flagship rows: Week-18 g1 pods (same generator, seeds, trials); quant rows: Week-19 a1 pods.

## Llama-3.1-8B


### 16K retrieval (Wilson 95%)

| arm | stored | single | mk | mv | vt |
|---|---|---|---|---|---|
| `bugSseed-r64-h256` | 0.151x | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 0.58 [0.32,0.81] n=12 |
| `quant-2bit-kivi` | 0.163x | 1.00 [0.76,1.00] n=12 | 0.67 [0.39,0.86] n=12 | 0.42 [0.19,0.68] n=12 | 0.67 [0.39,0.86] n=12 |
| `quant-4bit-kivi` | 0.287x | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 |
| `quant-8bit-kivi-hqq` | 0.535x | 1.00 [0.51,1.00] n=4 | 1.00 [0.51,1.00] n=4 | 1.00 [0.51,1.00] n=4 | 1.00 [0.51,1.00] n=4 |

### 32K retrieval (Wilson 95%)

| arm | stored | single | mk | mv | vt |
|---|---|---|---|---|---|
| `bugSseed-r64-h256` | 0.139x | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 0.92 [0.65,0.98] n=12 |
| `quant-2bit-kivi` | 0.160x | 1.00 [0.76,1.00] n=12 | 0.83 [0.55,0.95] n=12 | 0.92 [0.65,0.98] n=12 | 0.92 [0.65,0.98] n=12 |
| `quant-4bit-kivi` | 0.284x | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 |

### Paired McNemar (flagship vs quant; A>B = flagship hit where quant missed)

| vs | ctx | task | A>B | B>A | p | sig |
|---|---|---|---|---|---|---|
| `quant-2bit-kivi` | 16K | mk | 4 | 0 | 0.1250 | no |
| `quant-2bit-kivi` | 16K | mv | 7 | 0 | 0.0156 | **YES** |
| `quant-2bit-kivi` | 16K | vt | 3 | 4 | 1.0000 | no |
| `quant-2bit-kivi` | 32K | mk | 2 | 0 | 0.5000 | no |
| `quant-2bit-kivi` | 32K | mv | 1 | 0 | 1.0000 | no |
| `quant-2bit-kivi` | 32K | vt | 1 | 1 | 1.0000 | no |
| `quant-4bit-kivi` | 16K | vt | 0 | 5 | 0.0625 | no |
| `quant-4bit-kivi` | 32K | vt | 0 | 1 | 1.0000 | no |

### Perplexity (PPL4, WikiText-103, window 512)

| ctx | flagship (W18) | quant-2bit-kivi | quant-4bit-kivi |
|---|---|---|---|
| 16K | 5.31 | 5.40 | 4.90 |
| 32K | 8.33 | 8.28 | 7.54 |

### Sub-cliff compose `bugSseed-r64-h256-q4` (512 fp32 coords kept, the rest 4-bit PolarQuant, never dropped)

| ctx | stored | single | mk | mv | vt | ppl |
|---|---|---|---|---|---|---|
| 16K | 0.048x | 1.00 (n=12) | 1.00 (n=12) | 1.00 (n=12) | 0.50 (n=12) | 9.25 |

## Mistral-7B-v0.3


### 16K retrieval (Wilson 95%)

| arm | stored | single | mk | mv | vt |
|---|---|---|---|---|---|
| `bugSseed-r64-h256` | 0.150x | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 0.50 [0.25,0.75] n=12 |
| `quant-2bit-kivi` | 0.163x | 0.92 [0.65,0.98] n=12 | 0.58 [0.32,0.81] n=12 | 0.50 [0.25,0.75] n=12 | 0.33 [0.14,0.61] n=12 |
| `quant-4bit-kivi` | 0.287x | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 0.25 [0.09,0.53] n=12 |
| `quant-8bit-kivi-hqq` | 0.535x | 1.00 [0.51,1.00] n=4 | 1.00 [0.51,1.00] n=4 | 1.00 [0.51,1.00] n=4 | 0.25 [0.05,0.70] n=4 |

### 32K retrieval (Wilson 95%)

| arm | stored | single | mk | mv | vt |
|---|---|---|---|---|---|
| `bugSseed-r64-h256` | 0.139x | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 0.83 [0.55,0.95] n=12 | 0.42 [0.19,0.68] n=12 |
| `quant-2bit-kivi` | 0.160x | 0.83 [0.55,0.95] n=12 | 0.25 [0.09,0.53] n=12 | 0.08 [0.01,0.35] n=12 | 0.00 [0.00,0.24] n=12 |
| `quant-4bit-kivi` | 0.284x | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 0.25 [0.09,0.53] n=12 |

### Paired McNemar (flagship vs quant; A>B = flagship hit where quant missed)

| vs | ctx | task | A>B | B>A | p | sig |
|---|---|---|---|---|---|---|
| `quant-2bit-kivi` | 16K | single | 1 | 0 | 1.0000 | no |
| `quant-2bit-kivi` | 16K | mk | 5 | 0 | 0.0625 | no |
| `quant-2bit-kivi` | 16K | mv | 6 | 0 | 0.0312 | **YES** |
| `quant-2bit-kivi` | 16K | vt | 2 | 0 | 0.5000 | no |
| `quant-2bit-kivi` | 32K | single | 2 | 0 | 0.5000 | no |
| `quant-2bit-kivi` | 32K | mk | 9 | 0 | 0.0039 | **YES** |
| `quant-2bit-kivi` | 32K | mv | 9 | 0 | 0.0039 | **YES** |
| `quant-2bit-kivi` | 32K | vt | 5 | 0 | 0.0625 | no |
| `quant-4bit-kivi` | 16K | vt | 3 | 0 | 0.2500 | no |
| `quant-4bit-kivi` | 32K | mv | 0 | 2 | 0.5000 | no |
| `quant-4bit-kivi` | 32K | vt | 2 | 0 | 0.5000 | no |

### Perplexity (PPL4, WikiText-103, window 512)

| ctx | flagship (W18) | quant-2bit-kivi | quant-4bit-kivi |
|---|---|---|---|
| 16K | 5.50 | 4.99 | 4.83 |
| 32K | 3.76 | 3.46 | 3.31 |

### Sub-cliff compose `bugSseed-r64-h256-q4` (512 fp32 coords kept, the rest 4-bit PolarQuant, never dropped)

| ctx | stored | single | mk | mv | vt | ppl |
|---|---|---|---|---|---|---|
| 16K | 0.048x | 1.00 (n=12) | 1.00 (n=12) | 0.92 (n=12) | 0.33 (n=12) | — |

## Qwen2.5-7B


### 16K retrieval (Wilson 95%)

| arm | stored | single | mk | mv | vt |
|---|---|---|---|---|---|
| `bugSseed-r64-h256` | 0.275x | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 |
| `quant-2bit-kivi` | 0.163x | 1.00 [0.76,1.00] n=12 | 0.83 [0.55,0.95] n=12 | 0.33 [0.14,0.61] n=12 | 0.92 [0.65,0.98] n=12 |
| `quant-4bit-kivi` | 0.287x | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 0.92 [0.65,0.98] n=12 | 1.00 [0.76,1.00] n=12 |
| `quant-8bit-kivi-hqq` | 0.535x | 1.00 [0.51,1.00] n=4 | 1.00 [0.51,1.00] n=4 | 1.00 [0.51,1.00] n=4 | 1.00 [0.51,1.00] n=4 |

### 32K retrieval (Wilson 95%)

| arm | stored | single | mk | mv | vt |
|---|---|---|---|---|---|
| `bugSseed-r64-h256` | 0.265x | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 |
| `quant-2bit-kivi` | 0.160x | 0.92 [0.65,0.98] n=12 | 0.58 [0.32,0.81] n=12 | 0.17 [0.05,0.45] n=12 | 0.25 [0.09,0.53] n=12 |
| `quant-4bit-kivi` | 0.284x | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 | 1.00 [0.76,1.00] n=12 |

### Paired McNemar (flagship vs quant; A>B = flagship hit where quant missed)

| vs | ctx | task | A>B | B>A | p | sig |
|---|---|---|---|---|---|---|
| `quant-2bit-kivi` | 16K | mk | 2 | 0 | 0.5000 | no |
| `quant-2bit-kivi` | 16K | mv | 8 | 0 | 0.0078 | **YES** |
| `quant-2bit-kivi` | 16K | vt | 1 | 0 | 1.0000 | no |
| `quant-2bit-kivi` | 32K | single | 1 | 0 | 1.0000 | no |
| `quant-2bit-kivi` | 32K | mk | 5 | 0 | 0.0625 | no |
| `quant-2bit-kivi` | 32K | mv | 10 | 0 | 0.0020 | **YES** |
| `quant-2bit-kivi` | 32K | vt | 9 | 0 | 0.0039 | **YES** |
| `quant-4bit-kivi` | 16K | mv | 1 | 0 | 1.0000 | no |

### Perplexity (PPL4, WikiText-103, window 512)

| ctx | flagship (W18) | quant-2bit-kivi | quant-4bit-kivi |
|---|---|---|---|
| 16K | 8.18 | 6.70 | 6.20 |
| 32K | 35.08 | 8.23 | 7.71 |

### Sub-cliff compose `bugSseed-r64-h256-q4` (512 fp32 coords kept, the rest 4-bit PolarQuant, never dropped)

| ctx | stored | single | mk | mv | vt | ppl |
|---|---|---|---|---|---|---|
| 16K | 0.071x | 0.00 (n=12) | 0.00 (n=12) | 0.00 (n=12) | 0.00 (n=12) | 14489.99 |

## Persistence cold start (Llama-3.1-8B, A100-40GB; a3-llama2, medians of 5)

| ctx | arm | bytes | ratio | load | h2d | attend-ready | cold total |
|---|---|---|---|---|---|---|---|
| 16K | `bugSseed-r64-h256` | 325 MB | 0.151x | 0.067s | 0.028s | 0.039s | **0.134s** |
| 16K | `full` | 2148 MB | 1.000x | 1.114s | 0.112s | 0.000s | **1.226s** |
| 16K | `quant-2bit-kivi` | 336 MB | 0.156x | 0.092s | 0.025s | 0.020s | **0.137s** |
| 32K | `bugSseed-r64-h256` | 600 MB | 0.140x | 0.103s | 0.043s | 0.082s | **0.227s** |
| 32K | `full` | 4295 MB | 1.000x | 2.113s | 0.238s | 0.000s | **2.351s** |
| 32K | `quant-2bit-kivi` | 671 MB | 0.156x | 0.126s | 0.046s | 0.040s | **0.212s** |
