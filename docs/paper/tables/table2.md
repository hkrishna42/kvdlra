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
