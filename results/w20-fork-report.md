# Week-20 decisive fork: eviction x quantization vs the sub-cliff band

The sub-cliff cell `bugSseed-r64-h256-q4` (0.048x/0.034x) is exclusive vs *scalar* quantization by construction. This measures whether an **eviction x quantization** composite (`ea-k{keep}-q{nbits}`: ExpectedAttention prunes to keep-fraction, survivors stored 2/4-bit) reaches the same band **with retrieval**, or collapses like plain eviction. In-repo arms share a1q's needles; the official arms share a2's (NVIDIA RULER, essays). Stored ratios: ea-k0.1-q2 0.016x, ea-k0.1-q4 0.028x, ea-k0.25-q2 0.039x, ea-k0.25-q4 0.070x -- the byte-matches to the q4 cell are ea-k0.25-q2 (~0.039x vs 0.048x at 16K) and ea-k0.1-q4 (~0.028x vs 0.034x at 32K).


## Llama-3.1-8B -- our generator (in-repo, n=12)

| ctx | arm | stored | single | multi-key | multi-value | var-track |
|---|---|---|---|---|---|---|
| 16K | **q4 cell** (bugSseed-r64-h256-q4) | ~0.048x | 1.00 | 1.00 | 1.00 | 0.50 |
| 16K | ea-k0.1-q2-kivi | 0.016x | 0.75 | 0.08 | 0.00 | 0.00 |
| 16K | ea-k0.1-q4-kivi | 0.028x | 1.00 | 0.75 | 0.92 | 0.42 |
| 16K | ea-k0.25-q2-kivi | 0.039x | 0.92 | 0.08 | 0.08 | 0.00 |
| 16K | ea-k0.25-q4-kivi | 0.07x | 1.00 | 1.00 | 1.00 | 0.50 |
| 32K | **q4 cell** (bugSseed-r64-h256-q4) | ~0.034x | 1.00 | 1.00 | 1.00 | 0.83 |
| 32K | ea-k0.1-q2-kivi | 0.016x | 0.92 | 0.42 | 0.00 | 0.17 |
| 32K | ea-k0.1-q4-kivi | 0.028x | 1.00 | 0.83 | 0.83 | 0.58 |
| 32K | ea-k0.25-q2-kivi | 0.039x | 1.00 | 0.83 | 0.33 | 0.33 |
| 32K | ea-k0.25-q4-kivi | 0.07x | 1.00 | 1.00 | 1.00 | 0.33 |

## Mistral-7B-v0.3 -- our generator (in-repo, n=12)

| ctx | arm | stored | single | multi-key | multi-value | var-track |
|---|---|---|---|---|---|---|
| 16K | **q4 cell** (bugSseed-r64-h256-q4) | ~0.048x | 1.00 | 1.00 | 0.92 | 0.33 |
| 16K | ea-k0.1-q2-kivi | 0.016x | 0.00 | 0.00 | 0.00 | 0.00 |
| 16K | ea-k0.1-q4-kivi | 0.028x | 0.33 | 0.08 | 0.00 | 0.00 |
| 16K | ea-k0.25-q2-kivi | 0.039x | 0.00 | 0.00 | 0.00 | 0.08 |
| 16K | ea-k0.25-q4-kivi | 0.07x | 0.83 | 0.33 | 0.42 | 0.08 |
| 32K | **q4 cell** (bugSseed-r64-h256-q4) | ~0.034x | 1.00 | 1.00 | 0.83 | 0.58 |
| 32K | ea-k0.1-q2-kivi | 0.016x | 0.00 | 0.00 | 0.00 | 0.00 |
| 32K | ea-k0.1-q4-kivi | 0.028x | 0.58 | 0.08 | 0.00 | 0.00 |
| 32K | ea-k0.25-q2-kivi | 0.039x | 0.00 | 0.00 | 0.00 | 0.00 |
| 32K | ea-k0.25-q4-kivi | 0.07x | 0.58 | 0.08 | 0.42 | 0.00 |

## Qwen2.5-7B -- our generator (in-repo, n=12)

| ctx | arm | stored | single | multi-key | multi-value | var-track |
|---|---|---|---|---|---|---|
| 16K | **q4 cell** (bugSseed-r64-h256-q4) | ~0.048x | 0.00 | 0.00 | 0.00 | 0.00 |
| 16K | ea-k0.1-q2-kivi | 0.016x | 0.00 | 0.00 | 0.00 | 0.00 |
| 16K | ea-k0.1-q4-kivi | 0.028x | 0.08 | 0.00 | 0.00 | 0.00 |
| 16K | ea-k0.25-q2-kivi | 0.039x | 0.25 | 0.00 | 0.00 | 0.08 |
| 16K | ea-k0.25-q4-kivi | 0.07x | 0.33 | 0.08 | 0.00 | 0.25 |
| 32K | **q4 cell** (bugSseed-r64-h256-q4) | ~0.034x | -- | -- | -- | -- |
| 32K | ea-k0.1-q2-kivi | 0.016x | 0.00 | 0.00 | 0.00 | 0.00 |
| 32K | ea-k0.1-q4-kivi | 0.028x | 0.00 | 0.17 | 0.00 | 0.00 |
| 32K | ea-k0.25-q2-kivi | 0.039x | 0.00 | 0.00 | 0.00 | 0.00 |
| 32K | ea-k0.25-q4-kivi | 0.07x | 0.58 | 0.50 | 0.08 | 0.50 |

## Official NVIDIA RULER anchor (Llama 16K, 9 tasks) -- composite vs plain eviction

Plain `ea-k0.1` (0.100x) scored mean **0.20** here (a2). Do the composites do better at <=0.07x?

| arm | stored | single1 | single2 | single3 | multikey1 | multikey2 | multikey3 | multivalue | multiquery | vt | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ea-k0.1-q2-kivi | 0.016x | 0.75 | 0.08 | 0.00 | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.11** |
| ea-k0.1-q4-kivi | 0.028x | 1.00 | 0.25 | 0.00 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 | **0.19** |
| ea-k0.25-q2-kivi | 0.039x | 0.75 | 0.42 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.17** |
| ea-k0.25-q4-kivi | 0.07x | 1.00 | 0.58 | 0.00 | 0.42 | 0.25 | 0.00 | 0.00 | 0.00 | 0.75 | **0.33** |
| `ea-k0.1` (plain, a2) | 0.100x | 1.00 | 0.17 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | **0.20** |
