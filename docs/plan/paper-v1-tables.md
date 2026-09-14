## Table 1 — cross-model 16K retrieval, config `bugSseed-r64-h256`
<!-- paper-v1: paper/main.tex @ee8c0ab, table `tab:xmodel` (renders as Table 2 in the PDF) -->
<!-- source: results/paper-v1/w18-g1-{qwen,mistral,llama}/trials.jsonl (cells), results/paper-v1/w18-{qwen,mistral,llama}/cells.jsonl (stored state) -->
<!-- cell: acc [Wilson 95% lo,hi] (hits/n) -->
<!-- v1 printed the Llama var-track upper bound as 0.80 (truncated); correct rounding 0.81 -->
<!-- stored state = the arm's float-equivalent ratio (`ratio=`), the convention v1's caption states for this table -->
<!-- a memory value is the mean of the arm's archived cell rows, which agree to within one 0.001 print unit (stored state is a property of the run, not of a needle) -->
<!-- feat. n = num_key_value_heads x head_dim, a model constant, not a measurement -->

| model | feat. n | single | multi-key (not in v1) | multi-value | var-track | stored state |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | 512 | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.149x |
| Mistral-7B-v0.3 | 1024 | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.50 [0.25,0.75] (6/12) | 0.085x |
| Llama-3.1-8B | 1024 | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.58 [0.32,0.81] (7/12) | 0.085x |
## Table 2 — cross-model 32K retrieval, config `bugSseed-r64-h256`
<!-- paper-v1: paper/main.tex @ee8c0ab, table `tab:xmodel32` (renders as Table 3 in the PDF) -->
<!-- source: results/paper-v1/w18-g1-{qwen,mistral,llama}/trials.jsonl (cells), results/paper-v1/w18-{qwen,mistral,llama}/cells.jsonl (stored state) -->
<!-- cell: acc [Wilson 95% lo,hi] (hits/n) -->
<!-- v1 printed the Llama var-track upper bound as 0.98 (truncated); correct rounding 0.99 -->
<!-- stored state = the arm's float-equivalent ratio (`ratio=`), the convention v1's caption states for this table -->
<!-- a memory value is the mean of the arm's archived cell rows, which agree to within one 0.001 print unit (stored state is a property of the run, not of a needle) -->

| model | single | multi-key | multi-value | var-track | stored state |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.139x |
| Mistral-7B-v0.3 | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.83 [0.55,0.95] (10/12) | 0.42 [0.19,0.68] (5/12) | 0.075x |
| Llama-3.1-8B | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 1.00 [0.76,1.00] (12/12) | 0.92 [0.65,0.99] (11/12) | 0.075x |
## Table 3 — 32K variable-tracking on Llama-3.1-8B, config `bugSseed-r128-h1024-s32` vs baselines
<!-- paper-v1: paper/main.tex @ee8c0ab, the 32K Llama variable-tracking table (its LaTeX label carries a word CLAUDE.md bans from new files, so it is not quoted here; it renders as Table 4 in the PDF) -->
<!-- source: results/paper-v1/w18-g4-llama/trials.jsonl (the n=16 arms and the n=12 r256 control), results/paper-v1/w19-a1-llama/trials.jsonl (quant-4bit-kivi, n=12); stored state from each pod's own cells.jsonl -->
<!-- cell: acc [Wilson 95% lo,hi] (hits/n); p = exact paired McNemar vs bugSseed-r128-h1024-s32 on the shared (seed,trial) keys, 2 significant figures -->
<!-- discordant = pairs won by bugSseed-r128-h1024-s32 / pairs won by the row's arm -->
<!-- v1 printed no McNemar p for the 4-bit row; it is computed here on the 12 shared keys -->
<!-- the quant-4bit-kivi row is a different pod (w19-a1-llama) from every other row (w18-g4-llama): pairing on (seed,trial) assumes both pods built the same prompt for a given key -- which a deterministic generator under greedy decode does, but prompt_sha256 is null in the v1 records, so the archive cannot verify it -->
<!-- stored state = the arm's fp32-at-rest stored bits (`sbits=`), the convention v1's caption states for this table -->
<!-- a memory value is the mean of the arm's archived cell rows, which agree to within one 0.001 print unit (stored state is a property of the run, not of a needle) -->
<!-- v1 printed think-c0.5/palu-r0.5 to 2 decimals (0.75x/0.50x); the archived rows are printed here at the 3 decimals the other rows need -->
<!-- the verdict column of v1 is omitted: it is an editorial reading, not a statistic -->

| config | var-track | stored state | McNemar p | discordant | n paired |
| --- | --- | --- | --- | --- | --- |
| bugSseed-r128-h1024-s32 | 0.94 [0.72,0.99] (15/16) | 0.284x | --- | --- | --- |
| think-c0.5 | 0.31 [0.14,0.56] (5/16) | 0.750x | 2.0e-03 | 10/0 | 16 |
| palu-r0.5 | 0.56 [0.33,0.77] (9/16) | 0.502x | 3.1e-02 | 6/0 | 16 |
| quant-4bit-kivi | 1.00 [0.76,1.00] (12/12) | 0.284x | 1.0e+00 | 0/1 | 12 |
| bugSseed-r256-h1024 (not in v1) | 0.00 [0.00,0.24] (0/12) | 0.534x | 9.8e-04 | 11/0 | 12 |
## Table 6 — eviction at 0.100x stored state, arm `ea-k0.1`
<!-- paper-v1: paper/main.tex @ee8c0ab, table `tab:evict` (renders as Table 7 in the PDF) -->
<!-- source: results/paper-v1/w18-g3-{llama,qwen,mistral}/trials.jsonl (cells), and their cells.jsonl (the budget in the title) -->
<!-- cell: acc [Wilson 95% lo,hi] (hits/n) -->
<!-- rows are the model x ctx cells v1 showed; the pods also hold Qwen/Mistral 32K, which v1 did not print -->
<!-- v1's table has no memory column: its budget is stated once in the caption, and is regenerated in the title above from the `ratio=` rows of all 4 cells -->

| model | ctx | single | multi-key | multi-value | var-track |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | 16384 | 1.00 [0.76,1.00] (12/12) | 0.92 [0.65,0.99] (11/12) | 1.00 [0.76,1.00] (12/12) | 0.08 [0.01,0.35] (1/12) |
| Llama-3.1-8B | 32768 | 1.00 [0.76,1.00] (12/12) | 0.92 [0.65,0.99] (11/12) | 0.92 [0.65,0.99] (11/12) | 0.58 [0.32,0.81] (7/12) |
| Qwen2.5-7B | 16384 | 0.17 [0.05,0.45] (2/12) | 0.00 [0.00,0.24] (0/12) | 0.00 [0.00,0.24] (0/12) | 0.00 [0.00,0.24] (0/12) |
| Mistral-7B-v0.3 | 16384 | 0.50 [0.25,0.75] (6/12) | 0.08 [0.01,0.35] (1/12) | 0.00 [0.00,0.24] (0/12) | 0.00 [0.00,0.24] (0/12) |
## Table 7 — the 2-bit/4-bit KIVI baseline at matched stored bytes
<!-- paper-v1: paper/main.tex @ee8c0ab, table `tab:fairquant` (renders as Table 8 in the PDF) -->
<!-- source: results/paper-v1/w18-g1-{llama,mistral,qwen}/trials.jsonl (r64 rows), results/paper-v1/w19-a1-{llama,mistral,qwen}/trials.jsonl (KIVI rows); stored bits from results/paper-v1/w18-{llama,mistral,qwen}/cells.jsonl (r64) and each w19-a1 pod's own cells.jsonl (KIVI) -->
<!-- cell: acc (hits/n) -->
<!-- bold = exact paired McNemar p<0.05 in the r64 arm's favour against a KIVI arm of the same model x ctx x task, paired on (seed,trial); no cell is significant in a KIVI arm's favour -->
<!-- the r64 rows and the KIVI rows are different pods (w18-g1-<model> vs w19-a1-<model>): pairing on (seed,trial) assumes both pods built the same prompt for a given key -- which a deterministic generator under greedy decode does, but prompt_sha256 is null in the v1 records, so the archive cannot verify it -->
<!-- stored = the arm's fp32-at-rest stored bits (`sbits=`), the convention v1's caption states for this table -->
<!-- a memory value is the mean of the arm's archived cell rows, which agree to within one 0.001 print unit (stored state is a property of the run, not of a needle) -->

| model | ctx | arm | stored | single | multi-key | multi-value | var-track |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B | 16384 | bugSseed-r64-h256 | 0.151x | 1.00 (12/12) | 1.00 (12/12) | **1.00 (12/12)** | 0.58 (7/12) |
| Llama-3.1-8B | 16384 | quant-2bit-kivi | 0.163x | 1.00 (12/12) | 0.67 (8/12) | 0.42 (5/12) | 0.67 (8/12) |
| Llama-3.1-8B | 16384 | quant-4bit-kivi | 0.287x | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) |
| Llama-3.1-8B | 32768 | bugSseed-r64-h256 | 0.139x | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) |
| Llama-3.1-8B | 32768 | quant-2bit-kivi | 0.160x | 1.00 (12/12) | 0.83 (10/12) | 0.92 (11/12) | 0.92 (11/12) |
| Llama-3.1-8B | 32768 | quant-4bit-kivi | 0.284x | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) |
| Mistral-7B-v0.3 | 16384 | bugSseed-r64-h256 | 0.150x | 1.00 (12/12) | 1.00 (12/12) | **1.00 (12/12)** | 0.50 (6/12) |
| Mistral-7B-v0.3 | 16384 | quant-2bit-kivi | 0.163x | 0.92 (11/12) | 0.58 (7/12) | 0.50 (6/12) | 0.33 (4/12) |
| Mistral-7B-v0.3 | 16384 | quant-4bit-kivi | 0.287x | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.25 (3/12) |
| Mistral-7B-v0.3 | 32768 | bugSseed-r64-h256 | 0.139x | 1.00 (12/12) | **1.00 (12/12)** | **0.83 (10/12)** | 0.42 (5/12) |
| Mistral-7B-v0.3 | 32768 | quant-2bit-kivi | 0.160x | 0.83 (10/12) | 0.25 (3/12) | 0.08 (1/12) | 0.00 (0/12) |
| Mistral-7B-v0.3 | 32768 | quant-4bit-kivi | 0.284x | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.25 (3/12) |
| Qwen2.5-7B | 16384 | bugSseed-r64-h256 | 0.275x | 1.00 (12/12) | 1.00 (12/12) | **1.00 (12/12)** | 1.00 (12/12) |
| Qwen2.5-7B | 16384 | quant-2bit-kivi | 0.163x | 1.00 (12/12) | 0.83 (10/12) | 0.33 (4/12) | 0.92 (11/12) |
| Qwen2.5-7B | 16384 | quant-4bit-kivi | 0.287x | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 1.00 (12/12) |
| Qwen2.5-7B | 32768 | bugSseed-r64-h256 | 0.265x | 1.00 (12/12) | 1.00 (12/12) | **1.00 (12/12)** | **1.00 (12/12)** |
| Qwen2.5-7B | 32768 | quant-2bit-kivi | 0.160x | 0.92 (11/12) | 0.58 (7/12) | 0.17 (2/12) | 0.25 (3/12) |
| Qwen2.5-7B | 32768 | quant-4bit-kivi | 0.284x | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) |
## Table 8 — official NVIDIA RULER at 16K on Llama-3.1-8B
<!-- paper-v1: paper/main.tex @ee8c0ab, table `tab:official` (renders as Table 9 in the PDF) -->
<!-- source: results/paper-v1/w19-a2-llama/trials.jsonl (7 arms), results/paper-v1/w19-q4off-llama/trials.jsonl (the q4 cell arm); stored bits from the same two pods' cells.jsonl -->
<!-- cell: acc (hits/n); mean = mean of the nine printed 2-dp accuracies, as in v1 (pooling the records instead gives 0.80 for bugSseed-r64-h256) -->
<!-- stored = the arm's fp32-at-rest stored bits (`sbits=`), the convention v1's caption states for this table -->
<!-- a memory value is the mean of the arm's archived cell rows, which agree to within one 0.001 print unit (stored state is a property of the run, not of a needle) -->

| arm | stored | s1 | s2 | s3 | mk1 | mk2 | mk3 | mv | mq | vt | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | 1.00x | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.99 |
| think-c0.5 | 0.75x | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 0.98 |
| palu-r0.5 | 0.50x | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 0.42 (5/12) | 0.92 (11/12) | 1.00 (12/12) | 0.92 |
| quant-4bit-kivi | 0.29x | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 1.00 (12/12) | 1.00 (12/12) | 1.00 (12/12) | 0.99 |
| quant-2bit-kivi | 0.16x | 1.00 (12/12) | 1.00 (12/12) | 0.83 (10/12) | 1.00 (12/12) | 0.92 (11/12) | 0.58 (7/12) | 0.83 (10/12) | 1.00 (12/12) | 0.67 (8/12) | 0.87 |
| bugSseed-r64-h256 | 0.15x | 0.83 (10/12) | 1.00 (12/12) | 0.83 (10/12) | 0.83 (10/12) | 1.00 (12/12) | 0.83 (10/12) | 0.83 (10/12) | 0.67 (8/12) | 0.33 (4/12) | 0.79 |
| bugSseed-r64-h256-q4 | 0.05x | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 |
| ea-k0.1 | 0.10x | 1.00 (12/12) | 0.17 (2/12) | 0.00 (0/12) | 0.33 (4/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.00 (0/12) | 0.33 (4/12) | 0.20 |
