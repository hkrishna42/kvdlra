# 2026-06-22 — Post-RoPE KV-cache singular-value decay

## Hypothesis
Llama-3.2-1B's cached K is low-rank enough that truncated SVD / incremental SVD / a BUG sweep
reconstruct it at modest rank — but, because HuggingFace caches *post-RoPE* keys, the decay may
be shallow (the RoPE pitfall), making the keys harder to compress than pre-RoPE.

## Setup
Mac, CPU, fp32. `unsloth/Llama-3.2-1B-Instruct` (ungated verbatim mirror; the account lacks
Meta's gated allowlist). One C4 document (idx 15) at 2048 tokens. Layers 0/8/15. Matrix
`M = (512 features, 2044 tokens)` after excluding the first 4 sink-token columns. Ranks
4/8/16/32/64/128.

## Command
```bash
python scripts/capture_kv.py --device cpu --model unsloth/Llama-3.2-1B-Instruct --seq_len 2048 --doc_idx 15
python scripts/week1_sv_decay.py --dump dumps/llama3.2-1b/doc15_d1b3a6eb_len2048
```

## Wandb URL
n/a — no `WANDB_API_KEY` provided this session. Metrics recorded under Result; sync later with a key.

## Result
- Capture: 16 layers, per-layer K/V `(8, 2048, 64)`, ~20 s on CPU. `figures/week1/sv_decay.{pdf,png}`.
- Truncated-SVD oracle rel-Frobenius error (all 3 layers): r=32 ≈ 0.34–0.36, r=64 ≈ 0.22–0.25, r=128 ≈ 0.16–0.17.
- Incremental SVD ≈ oracle + ~0.01–0.02 (near-optimal streaming baseline).
- BUG (1 sweep, random init): ≈ 0.57–0.96 — the plan's placeholder, not the streaming integrator.
- SV spectrum: sharp knee at index ~30–40 (down to σ_i/σ_1 ~ 0.05), then a heavy tail to ~1e-2 by ~index 300.

## Interpretation
Post-RoPE K is **not cleanly low-rank**: the oracle needs rank ≈128/512 for ~17% Frobenius
error. This is the RoPE pitfall (post-RoPE keys resist low-rank compression). Incremental SVD is
near-oracle; the single-sweep BUG underperforms as expected.

## Decision
No pivot — the result is informative, not a go/no-go failure (that is the Week-2 pilot). It is
the expected motivation to move to **pre-RoPE** keys in Week 2/3. Next: streaming BUG on real KV
streams + the Week-2 go/no-go pilot.
