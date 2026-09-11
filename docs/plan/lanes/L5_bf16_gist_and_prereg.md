# L5_bf16_gist_and_prereg


1. **bf16 gist storage** (`isvd_r64_h256_seed_bf16.yaml`): C and U stored bf16, fp32 accumulate in the step and the guard; 3 families × 16K/32K × 4 tasks × n=24 + perplexity; reading: non-inferior to fp32 storage within δ = 0.03 retrieval and 0.02 bits. If it passes, every byte-matched comparison is re-planned at the new bytes (DECISIONS entry, not a silent rerun).
2. **Prereg files**: `filler_realism.md`, `hygiene_table4.md`, `gate1_tracker_swap_v2.md`, `ss2_families.md`, `bf16_gist.md`, `l2_smoke.md`, `kernel_smoke.md`. Each: arms, tasks, n, primary contrast(s), family size + correction, decision rule, GPU budget, the SHA it must precede.
3. **Provenance**: `scripts/pod.py` writes `manifest.json` at launch and harvest; `--check` verifies config hash, model revision, dataset SHA, env vs lock; `make tables` refuses unchecked pods; test that a tampered config hash is rejected.
