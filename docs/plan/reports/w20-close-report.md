# Week-20 gate-closing experiments

## 1. Tracker-swap ablation (Llama-3.1-8B, 16K, n=12, same cache, same needles)

Is the DLRA integrator load-bearing end-to-end? The flagship's defaults (theta=None, min_sv_frac=0) make its step **fixed-rank incremental SVD** (Brand 2006), so that arm is the flagship as already measured (W18 g1). Oja's rule = the OjaKV baseline (validated Week-2 schedule); Frequent Directions = shrinkage instead of truncation.

| gist tracker | arm | single | multi-key | multi-value | var-track | ppl 16K |
|---|---|---|---|---|---|---|
| **incremental SVD (BUG step, flagship)** | `bugSseed-r64-h256` | 1.00 | 1.00 | 1.00 | 0.58 | 5.31 |
| Oja's rule | `bugSseed-r64-h256-oja` | 0.92 | 0.08 | 0.08 | 0.00 | 732.88 |
| Frequent Directions | `bugSseed-r64-h256-fd` | -- | -- | -- | -- | -- |

## 2. Measured decode latency and VRAM (Llama-3.1-8B, A100-40GB, batch 1)

One token per forward at true positions, CUDA-synced; p50 = steady state, max/spikes surface BUG's absorb-event middle rebuild. VRAM with model weights subtracted.

| ctx | arm | ms/tok p50 | mean | max | spikes | resident GB | peak GB | KV peak GB |
|---|---|---|---|---|---|---|---|---|
| 16K | `bugSseed-r64-h256` | 103.2 | 117.6 | 309.7 | 4 | 15.79 | 18.20 | 3.25 |
| 16K | `full` | 25.9 | 26.2 | 33.4 | 0 | 16.97 | 17.01 | 2.05 |
| 16K | `quant-2bit-kivi` | 43.7 | 43.9 | 52.5 | 0 | 15.28 | 15.41 | 0.45 |
| 32K | `bugSseed-r64-h256` | 188.3 | 205.0 | 428.0 | 4 | 16.03 | 21.40 | 6.45 |
| 32K | `full` | 27.5 | 27.5 | 28.0 | 0 | 18.97 | 19.04 | 4.08 |
| 32K | `quant-2bit-kivi` | 67.9 | 67.9 | 67.9 | 0 | 15.59 | 15.85 | 0.89 |
| 64K | `bugSseed-r64-h256` | 507.9 | 530.2 | 850.4 | 0 | 16.55 | 27.79 | 12.83 |
| 64K | `full` | 37.0 | 37.0 | 37.0 | 0 | 22.97 | 23.10 | 8.14 |
| 64K | `quant-2bit-kivi` | 118.9 | 119.1 | 127.7 | 0 | 16.23 | 16.74 | 1.78 |

## 3. Single-shot 2-bit prefill control (Llama-3.1-8B, 16K, n=12, same needles)

The a1 KIVI arm prefilled in 4096-token chunks (later chunks attend to 2-bit-dequantized history). Does the flagship's in-repo multi-value edge survive KIVI's single-shot (full-precision-prefill) operating point?

| protocol | arm | single | multi-key | multi-value | var-track | ppl 16K |
|---|---|---|---|---|---|---|
| chunked (a1, 4096) | `quant-2bit-kivi` | 1.00 | 0.67 | 0.42 | 0.67 | 5.40 |
| **single-shot (ss2, --chunk 0)** | `quant-2bit-kivi` | 1.00 | 1.00 | 0.83 | 0.67 | -- |
| flagship (reference) | `bugSseed-r64-h256` | 1.00 | 1.00 | 1.00 | 0.58 | 5.31 |

## 4. The sub-cliff cell on the official RULER anchor (Llama 16K, 9 tasks x 12)

The fork's pre-registered rule compares the composite to the q4 cell *on the anchor*; the fork ran only the composites there, this runs the cell itself (same records, seed 42). Rule: q4 holds single/mk/mv where the composites collapsed -> the band is anchored and exclusive (significance 7); q4 also collapses -> the band is a our-generator property and the paper says so (stays 6).

| arm | stored | single1 | single2 | single3 | multikey1 | multikey2 | multikey3 | multivalue | multiquery | vt | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **q4 cell** `bugSseed-r64-h256-q4` | 0.048x | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |
| flagship `bugSseed-r64-h256` (a2) | 0.151x | 0.83 | 1.00 | 0.83 | 0.83 | 1.00 | 0.83 | 0.83 | 0.67 | 0.33 | **0.79** |
| `ea-k0.1-q2-kivi` (fork) | 0.016x | 0.75 | 0.08 | 0.00 | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.11** |
| `ea-k0.1-q4-kivi` (fork) | 0.028x | 1.00 | 0.25 | 0.00 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 | **0.19** |
| `ea-k0.25-q2-kivi` (fork) | 0.039x | 0.75 | 0.42 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.17** |
| `ea-k0.25-q4-kivi` (fork) | 0.07x | 1.00 | 0.58 | 0.00 | 0.42 | 0.25 | 0.00 | 0.00 | 0.00 | 0.75 | **0.33** |
| `ea-k0.1` plain (a2) | 0.100x | 1.00 | 0.17 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | **0.20** |
