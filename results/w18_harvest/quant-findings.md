# W18 G1 quant baseline — GPU characterization (2026-09-03)

Harness: quant-{2,4}bit arms (transformers QuantizedCache, quanto backend), accounting
quant_footprint (verified on GPU: ratio 0.194 @2bit, 0.318 @4bit — matches prediction).

## Findings (Qwen-7B, 16K niah_single, GPU)
- Controls: `full` acc=1.00; BUG `bugSseed-r64-h256` acc=1.00 @0.149x/0.276x (headline holds).
- `quant-2bit` (axis0 per-channel): acc=0.00, ratio 0.194, n=12.
- `quant-4bit` (axis0 per-channel): acc=0.00, ratio 0.318, n=12.
- `quant-4bit-av-1` (per-token, KIVI-faithful): SKIP "Group size (64) must be a divisor of (65588)".
- `quant-8bit`: unavailable (quanto supports only 2/4-bit).
- quant ppl @16K: OOM on 40GB card (dequant + 16K attn over 7B).

## Interpretation
transformers' QuantizedCache/quanto with the DEFAULT per-channel value config gives ZERO
needle retrieval at 16K (2 AND 4-bit). That would strengthen BUG's exclusive-band claim,
BUT the per-token (KIVI-faithful) config won't run with group 64 (needs a divisor of the
data-dependent axis length), and 8-bit is unavailable — so a FAIR KIVI number is not yet
established. Do NOT report the 0.00 as "KIVI fails" without the per-token comparison.

## Follow-up (well-scoped, cheap)
1. Per-token value quant: find a group_size dividing the axis (or head_dim-aligned), OR
   use the reference KIVI codebase. Re-run quant-{2,4}bit RULER.
2. quant ppl: 80GB card (OOM on 40GB), or smaller n-samples / gradient-free scoring.
3. Then G1 quant is a fair baseline; until then the flagship exclusive-band claim stands on
   the eviction/channel-pruning baselines (which ARE measured), not quant.
