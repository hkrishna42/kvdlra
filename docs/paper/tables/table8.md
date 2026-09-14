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
