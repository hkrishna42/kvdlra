## Table 2 — cross-model 32K retrieval, config `bugSseed-r64-h256`
<!-- paper-v1: paper/main.tex @ee8c0ab, table `tab:xmodel32` (renders as Table 3 in the PDF) -->
<!-- source: results/paper-v1/w18-g1-{qwen,mistral,llama}/trials.jsonl (per-trial records only) -->
<!-- cell: acc [Wilson 95% lo,hi] (hits/n) -->
<!-- v1 printed the Llama var-track upper bound as 0.98 (truncated); correct rounding 0.99 -->
<!-- the stored-state column of v1 is omitted: memory ratios are not per-trial records -->

| model | single | multi-key | multi-value | var-track |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) |
| Mistral-7B-v0.3 | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.83 [0.55,0.95] (10/12) | 0.42 [0.19,0.68] (5/12) |
| Llama-3.1-8B | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.92 [0.65,0.99] (11/12) |
