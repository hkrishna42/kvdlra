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

---

# W19 A1 follow-up — root cause of the 0.00 (2026-09-05, $0/CPU)

**The W18 zero was the wrong quantization scheme, not a decode bug.** In optimum-quanto's
grouping semantics (`optimum/quanto/tensor/grouped.py::group`), for a `(B, H, T, D)` KV tensor
`axis=0` reshapes to `(-1, g)` = groups of `g` *consecutive elements* = **per-token** groups, and
`axis=-1` groups `g` consecutive rows per head-dim channel = **per-channel** groups (needing
`B*H*T % g == 0` — the `65588 = 4 heads x 16397 tokens` SKIP). transformers' `QuantizedCache`
docstring calls its default (axis 0 for both) "per-channel"; it is per-token for BOTH keys and
values. KIVI is per-channel KEYS + per-token VALUES. The W18 arms therefore quantized keys
per-token — the scheme a single large key channel (the Qwen2.5 `k_proj` bias outlier) defeats:
one scale per token is set by the outlier, and every other channel collapses to the same
level. The g1diag attempt (`--quant-axis-value=-1`) moved the *values* to per-channel — the
wrong tensor — and still tripped the divisor.

Synthetic check (CPU, `(1,2,128,16)` unit-variance keys + one channel at +50, 4-bit, g=64):

| scheme | scale shape | mean abs err, non-outlier channels | rel err |
|---|---|---|---|
| per-token (axis 0, the W18 default) | (64,1) | **0.955** (unit-variance data → destroyed) | 0.247 |
| per-channel (axis -1, KIVI keys) | (1,64) | 0.081 | 0.020 |

**Fix (`kvdlra.quant.kivi_cache`, `--quant-scheme kivi`)**: per-channel keys via quanto
`axis=-1` with the token axis edge-padded to a multiple of the group (sliced off on
dequantize; groups never straddle heads), per-token values (`axis=0`); `--quant-backend hqq`
adds 1–8-bit (the 8-bit decode-path control; per-channel keys via a transpose). Arms are named
`quant-{n}bit-kivi[-hqq]`; the default `quant-{n}bit` (scheme `token`) stays bit-identical to
the W18 arms (pinned by test). The quant arm now honors `--chunk` (chunked prefill + a residual
`flush()` so decode starts fully quantized exactly as after single-shot) — the W18 ppl OOMs
failed on plain prefill-sized activations (`T x intermediate`), i.e. memory was already full
during the single-shot 16K/32K prefill; the OOM row now also prints allocated/peak GB.
Aux (scale+zero) is billed at the backend's *measured* dtype (16-bit pairs on a bf16 model,
not the fp32 the fp32 CPU probe suggested): 2-bit/g64 asymptote 0.156x (was 0.1875x).

GPU validation pending: `scripts/pod/w19.sh` MODE `a1diag` (Qwen 16K, n=4: full, token-4bit,
kivi-2/4bit, hqq-8/2bit-kivi, kivi ppl@16K), then `a1` per family.

## W19 a1diag — GPU result (Qwen2.5-7B, 16K niah_single, n=4, A100-40GB, SHA 24ac22a)

| arm | acc | stored ratio | note |
|---|---|---|---|
| full | 1.00 | 1.000 | control |
| quant-4bit (per-token, W18 default, chunked) | **0.00** | 0.287 | the W18 zero REPLICATES under chunked prefill → it is the scheme |
| quant-2bit-kivi (quanto) | **1.00** | 0.163 | the fair KIVI baseline retrieves the single needle |
| quant-4bit-kivi (quanto) | 1.00 | 0.287 | |
| quant-8bit-kivi-hqq | 1.00 | 0.535 | decode-path control: sane |
| quant-2bit-kivi-hqq | 1.00 | 0.163 | cross-implementation agrees with quanto |

The quant **ppl** cells still OOM'd on this pod (38.4 GB *allocated* during a 4K chunk):
root cause = `score_quant` (and my `_prefill_plain`) were not under `torch.no_grad()`, so the
prefill's autograd graph was retained (the RULER path is decorated, hence rows there). Fixed
+ test-pinned in 6734afa; the a1 fan-out (qwen/mistral/llama, 40GB, SHA 6734afa) carries
the ppl cells. Fund-bar note: single-needle retrieval is NOT where the exclusive band would be
decided (KIVI-2bit = 1.00 there); mk/mv/vt at 16K/32K decide it.
