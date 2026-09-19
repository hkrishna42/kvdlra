## Table B — the v1 baseline rows the paper omitted: ShadowKV post-fix and eviction at 0.25x
<!-- not in paper-v1: the two competitive baseline rows the v1 tables omitted (docs/plan/CODE_AUDIT.md Part B, Q4 and Q5; PR-L2-13) -->
<!-- source: results/paper-v1/w15-confirm/cells.jsonl (shadow-r64, the post-fix ShadowKV re-measure) and results/paper-v1/w11-goalA-ruler/cells.jsonl (ea-k0.25), the archived `[task ctx16384] arm acc= ratio=` rows; 16K, in-house generator -->
<!-- cell: the archived acc, a point estimate -- these pre-Week-18 rows carry no per-trial records, so n is unknown (shown ---) and no Wilson interval is printed; a task the pod did not run is --- (shadow-r64 ran no multi-value) -->
<!-- model: the archive rows record `unknown` (the pre-Week-16 line files named no model); CODE_AUDIT attributes both runs to Llama-3.1-8B -->
<!-- stored state = the archived `ratio=` (float-equivalent); neither method holds fp32-at-rest state, so the stored-bits convention gives the same number; one value per row, shared by its task rows (checked) -->

| arm | stored state | single | multi-key | multi-value | var-track | n | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shadow-r64 | 0.815x | 1.00 | 1.00 | --- | 0.00 | --- | w15-confirm |
| ea-k0.25 | 0.250x | 1.00 | 0.88 | 1.00 | 0.50 | --- | w11-goalA-ruler |
