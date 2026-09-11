# L2_generator_v2_and_baselines


1. **Filler-realism diagnostic (Day 1, before anything else).** In-house four tasks, Llama 16K, n=12, `--filler` = real text (PG-19 chapter), arms: flagship, q4 cell, KIVI-2 (streaming and single-shot), full. Compare to the cycled-filler rows. Written reading, pre-registered in `prereg/filler_realism.md`: if the flagship or q4 drops > 0.25 on any task, the cycled-filler generator is retired from every headline claim (DECISIONS.md), and all v1 in-house tables are marked "diagnostic only" in the paper.
2. **Generator v2** (`kvdlra/eval/gen.py`): ≥ 4 haystack sources (PG-19, arXiv text, Wikipedia, synthetic essays), balanced depth design over {0.05, 0.2, 0.4, 0.6, 0.8, 0.95}, ≥ 2 code families, official RULER task semantics. Records carry `haystack_id`, `depth`, `code_family`, `seed`, `prompt_sha256`. Pairing test: same (task, ctx, seed, trial) → byte-identical prompt across arms; golden file.
3. **Faithful KIVI** (`kvdlra/quant/kivi.py`): G=32, R=128, per-channel K / per-token V, fp16 full prefill, scales + zeros counted; reference kernels if shipped, else quanto with the deviation in the config. Keep `kivi2_streaming.yaml` labelled. KVQuant-style pre-RoPE 2/3-bit dense-and-sparse if it fits; else DECISIONS entry.
4. **ss2 across families.** Single-shot KIVI-2 and KIVI-4 on Mistral/Qwen 16K + all three at 32K, same needles as a1, n=12 → `prereg/ss2_families.md`. Reading: the multi-value claim survives only where flagship vs single-shot KIVI-2 is separated after Holm over 3 families × 4 tasks.
5. **Rename** `palu-r0.5` → `svd_oracle_r0.5`; docstring states what it is not. DECISIONS entry: real Palu (which variant) vs oracle-as-static-upper-bound.
6. **Eviction / structured** via kvpress: SnapKV, PyramidKV, ExpectedAttention at k ∈ {0.10, 0.15, 0.25}; ThinK(0.5)+SnapKV(0.15) as intended; standalone ThinK as ablation only. ShadowKV and `ea_k0.25` v1 rows must appear in `make tables`.
7. **OjaKV end-to-end** from their repo, matched bytes, Llama 16K/32K; commit + arXiv version recorded.
8. **Smoke pod**: all arms, Llama 16K, n=12, generator v2 — harness validation, not paper data.
