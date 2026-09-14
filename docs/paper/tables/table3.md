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
