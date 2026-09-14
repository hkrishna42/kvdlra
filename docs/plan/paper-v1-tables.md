## Table 1 — cross-model 16K retrieval, config `bugSseed-r64-h256`
<!-- paper-v1: paper/main.tex @ee8c0ab, table `tab:xmodel` (renders as Table 2 in the PDF) -->
<!-- source: results/paper-v1/w18-g1-{qwen,mistral,llama}/trials.jsonl (per-trial records only) -->
<!-- cell: acc [Wilson 95% lo,hi] (hits/n) -->
<!-- v1 printed the Llama var-track upper bound as 0.80 (truncated); correct rounding 0.81 -->
<!-- the stored-state column of v1 is omitted: memory ratios are not per-trial records -->

| model | single | multi-key (not in v1) | multi-value | var-track |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) |
| Mistral-7B-v0.3 | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.50 [0.25,0.75] (6/12) |
| Llama-3.1-8B | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.58 [0.32,0.81] (7/12) |
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
## Table 3 — 32K variable-tracking on Llama-3.1-8B, config `bugSseed-r128-h1024-s32` vs baselines
<!-- paper-v1: paper/main.tex @ee8c0ab, the 32K Llama variable-tracking table (its LaTeX label carries a word CLAUDE.md bans from new files, so it is not quoted here; it renders as Table 4 in the PDF) -->
<!-- source: results/paper-v1/w18-g4-llama/trials.jsonl (n=16 arms, and the n=12 r256 control), results/paper-v1/w19-a1-llama/trials.jsonl (quant-4bit-kivi, n=12) -->
<!-- cell: acc [Wilson 95% lo,hi] (hits/n); p = exact paired McNemar vs bugSseed-r128-h1024-s32 on the shared (seed,trial) keys, 2 significant figures -->
<!-- discordant = pairs won by bugSseed-r128-h1024-s32 / pairs won by the row's arm -->
<!-- v1 printed no McNemar p for the 4-bit row; it is computed here on the 12 shared keys -->
<!-- the stored-state and verdict columns of v1 are omitted: neither is a per-trial statistic -->

| config | var-track | McNemar p | discordant | n paired |
| --- | --- | --- | --- | --- |
| bugSseed-r128-h1024-s32 | 0.94 [0.72,0.99] (15/16) | --- | --- | --- |
| think-c0.5 | 0.31 [0.14,0.56] (5/16) | 2.0e-03 | 10/0 | 16 |
| palu-r0.5 | 0.56 [0.33,0.77] (9/16) | 3.1e-02 | 6/0 | 16 |
| quant-4bit-kivi | 1.00 [0.76,1.00] (12/12) | 1.0e+00 | 0/1 | 12 |
| bugSseed-r256-h1024 (not in v1) | 0.00 [0.00,0.24] (0/12) | 9.8e-04 | 11/0 | 12 |
## Table 6 — eviction at 0.100x stored state, arm `ea-k0.1`
<!-- paper-v1: paper/main.tex @ee8c0ab, table `tab:evict` (renders as Table 7 in the PDF) -->
<!-- source: results/paper-v1/w18-g3-{llama,qwen,mistral}/trials.jsonl (per-trial records only) -->
<!-- cell: acc [Wilson 95% lo,hi] (hits/n) -->
<!-- rows are the model x ctx cells v1 showed; the pods also hold Qwen/Mistral 32K, which v1 did not print -->
<!-- the stored-state column of v1 is omitted: memory ratios are not per-trial records -->

| model | ctx | single | multi-key | multi-value | var-track |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | 16384 | 1.00 [0.76,1.00] (12/12) | 0.92 [0.65,0.99] (11/12) | 1.00 [0.76,1.00] (12/12) | 0.08 [0.01,0.35] (1/12) |
| Llama-3.1-8B | 32768 | 1.00 [0.76,1.00] (12/12) | 0.92 [0.65,0.99] (11/12) | 0.92 [0.65,0.99] (11/12) | 0.58 [0.32,0.81] (7/12) |
| Qwen2.5-7B | 16384 | 0.17 [0.05,0.45] (2/12) | 0.00 [0.00,0.24] (0/12) | 0.00 [0.00,0.24] (0/12) | 0.00 [0.00,0.24] (0/12) |
| Mistral-7B-v0.3 | 16384 | 0.50 [0.25,0.75] (6/12) | 0.08 [0.01,0.35] (1/12) | 0.00 [0.00,0.24] (0/12) | 0.00 [0.00,0.24] (0/12) |
## Table 7 — the 2-bit/4-bit KIVI baseline at matched stored bytes
<!-- paper-v1: paper/main.tex @ee8c0ab, table `tab:fairquant` (renders as Table 8 in the PDF) -->
<!-- source: results/paper-v1/w18-g1-{llama,mistral,qwen}/trials.jsonl (r64 rows), results/paper-v1/w19-a1-{llama,mistral,qwen}/trials.jsonl (KIVI rows) -->
<!-- cell: acc (hits/n) -->
<!-- bold = exact paired McNemar p<0.05 in the r64 arm's favour against a KIVI arm of the same model x ctx x task, paired on (seed,trial); no cell is significant in a KIVI arm's favour -->
<!-- the stored-bits column of v1 is omitted: memory ratios are not per-trial records -->

| model | ctx | arm | single | multi-key | multi-value | var-track |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | 16384 | bugSseed-r64-h256 | 1.00 (12/12) | 1.00 (12/12) | **1.00 (12/12)** | 0.58 (7/12) |
| Llama-3.1-8B | 16384 | quant-2bit-kivi | 1.00 (12/12) | 0.67 (8/12) | 0.42 (5/12) | 0.67 (8/12) |
| Llama-3.1-8B | 16384 | quant-4bit-kivi | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) |
| Llama-3.1-8B | 32768 | bugSseed-r64-h256 | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) |
| Llama-3.1-8B | 32768 | quant-2bit-kivi | 1.00 (12/12) | 0.83 (10/12) | 0.92 (11/12) | 0.92 (11/12) |
| Llama-3.1-8B | 32768 | quant-4bit-kivi | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) |
| Mistral-7B-v0.3 | 16384 | bugSseed-r64-h256 | 1.00 (12/12) | 1.00 (12/12) | **1.00 (12/12)** | 0.50 (6/12) |
| Mistral-7B-v0.3 | 16384 | quant-2bit-kivi | 0.92 (11/12) | 0.58 (7/12) | 0.50 (6/12) | 0.33 (4/12) |
| Mistral-7B-v0.3 | 16384 | quant-4bit-kivi | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.25 (3/12) |
| Mistral-7B-v0.3 | 32768 | bugSseed-r64-h256 | 1.00 (12/12) | **1.00 (12/12)** | **0.83 (10/12)** | 0.42 (5/12) |
| Mistral-7B-v0.3 | 32768 | quant-2bit-kivi | 0.83 (10/12) | 0.25 (3/12) | 0.08 (1/12) | 0.00 (0/12) |
| Mistral-7B-v0.3 | 32768 | quant-4bit-kivi | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.25 (3/12) |
| Qwen2.5-7B | 16384 | bugSseed-r64-h256 | 1.00 (12/12) | 1.00 (12/12) | **1.00 (12/12)** | 1.00 (12/12) |
| Qwen2.5-7B | 16384 | quant-2bit-kivi | 1.00 (12/12) | 0.83 (10/12) | 0.33 (4/12) | 0.92 (11/12) |
| Qwen2.5-7B | 16384 | quant-4bit-kivi | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 1.00 (12/12) |
| Qwen2.5-7B | 32768 | bugSseed-r64-h256 | 1.00 (12/12) | 1.00 (12/12) | **1.00 (12/12)** | **1.00 (12/12)** |
| Qwen2.5-7B | 32768 | quant-2bit-kivi | 0.92 (11/12) | 0.58 (7/12) | 0.17 (2/12) | 0.25 (3/12) |
| Qwen2.5-7B | 32768 | quant-4bit-kivi | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) |
## Table 8 — official NVIDIA RULER at 16K on Llama-3.1-8B
<!-- paper-v1: paper/main.tex @ee8c0ab, table `tab:official` (renders as Table 9 in the PDF) -->
<!-- source: results/paper-v1/w19-a2-llama/trials.jsonl (7 arms), results/paper-v1/w19-q4off-llama/trials.jsonl (the q4 cell arm) -->
<!-- cell: acc (hits/n); mean = mean of the nine printed 2-dp accuracies, as in v1 (pooling the records instead gives 0.80 for bugSseed-r64-h256) -->
<!-- the stored-bits column of v1 is omitted: memory ratios are not per-trial records -->

| arm | s1 | s2 | s3 | mk1 | mk2 | mk3 | mv | mq | vt | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.99 |
| think-c0.5 | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 0.98 |
| palu-r0.5 | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 0.42 (5/12) | 0.92 (11/12) | 1.00 (12/12) | 0.92 |
| quant-4bit-kivi | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.99 |
| quant-2bit-kivi | 1.00 (12/12) | 1.00 (12/12) | 0.83 (10/12) | 1.00 (12/12) | 0.92 (11/12) | 0.58 (7/12) | 0.83 (10/12) | 1.00 (12/12) | 0.67 (8/12) | 0.87 |
| bugSseed-r64-h256 | 0.83 (10/12) | 1.00 (12/12) | 0.83 (10/12) | 0.83 (10/12) | 1.00 (12/12) | 0.83 (10/12) | 0.83 (10/12) | 0.67 (8/12) | 0.33 (4/12) | 0.79 |
| bugSseed-r64-h256-q4 | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 |
| ea-k0.1 | 1.00 (12/12) | 0.17 (2/12) | 0.00 (0/12) | 0.33 (4/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.33 (4/12) | 0.20 |
