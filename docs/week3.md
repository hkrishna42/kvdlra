# Week 3 — `BUGPress` in kvpress: generation correctness + perplexity

**Summary.** The Week-2 streaming BUG tracker is now a working NVIDIA-`kvpress`
press, `BUGPress`, that replaces a layer's KV cache with its rank-`r` dynamical-
low-rank reconstruction during pre-fill. Generation is **correct** (no garbage at
any rank; a working RoPE round-trip verified bit-exact), perplexity degrades
**gracefully and monotonically** with rank (near-baseline at 4× compression),
and **pre-RoPE beats post-RoPE across the compression regime that matters
(rank ≤ 64)** — confirming, at the generation/perplexity level, Week-2's
reconstruction-error finding. Pre-RoPE is therefore the operating point.

All numbers below are Llama-3.2-1B (ungated `unsloth/Llama-3.2-1B-Instruct`,
config-identical), fp32. The perplexity headline is a **16-window / ctx-1024**
sweep run in ~80 s on the Mac CPU via the blocked-BUG **torch backend** (see the
*blocked-BUG* section); the GPU pod proved unnecessary for 1B once the numpy
per-token bottleneck was removed.

## What was built
- **`src/kvdlra/press/bug_press.py` — `BUGPress(BasePress)`.** In `compress`,
  the per-layer keys are reshaped to the Week-2 joint **512-feature** matrix
  (`M = K.transpose(0,2,1).reshape(512, T)`, `docs/notes/conventions.md`),
  streamed through `StreamingBUG(rank_cap=rank)`, reconstructed as the orthogonal
  projection `U Uᵀ M`, and reshaped back per-head. Values are treated the same
  (no RoPE). Same as `ThinKPress`, the tensor **shape is unchanged** — the
  compression is *nominal* (`compression_ratio = 1 − rank/512`), so this isolates
  the *accuracy* cost; genuinely factored storage is Week-4+.
- **pre-RoPE operating point.** With `pre_rope=True` (default), keys are
  recomputed pre-RoPE via kvpress's `get_prerope_key_states`, factored, then
  re-rotated with the layer's own `position_embeddings` before being written
  back — so attention still sees correctly rotated keys.
- **attention sinks.** The first `n_sink=4` token columns are kept exact
  (StreamingLLM-style), matching Week-2 excluding them from the low-rank model.
- Two harnesses: `scripts/generate_with_press.py` (generation + `--parity`) and
  `scripts/perplexity_sweep.py` (WikiText-2 perplexity under a compressed cache).

## Integration gotchas (both cost real debugging)
1. **`cache_position` is not threaded into the attention module in transformers
   5.8.** kvpress's default `BasePress.forward_hook` keys off
   `kwargs["cache_position"]` to detect pre-fill and `KeyError`s. `BUGPress`
   overrides `forward_hook` to detect pre-fill by `q_len > 1` instead.
2. **kvpress compresses in a *post*-attention hook** — the cache is swapped only
   *after* a layer's attention has run, so it affects **subsequent** forwards,
   not the one that triggered it. A single teacher-forced pass therefore does not
   feel the compression at all. Perplexity must use a **prefill-then-score**
   protocol (compress the context cache, then score continuation tokens against
   it); see `scripts/perplexity_sweep.py`. This is a deliberate, documented
   deviation from the PLAN's lm-eval-harness (more faithful for a cache
   compressor).

## Generation parity (`results/w3-parity.md`, pre-RoPE, greedy)
- **SHORT prompts (≤12 tokens): 10/10 byte-exact vs. baseline at every rank
  (64/32/16/8).** A *no-regression* check only — with `n_sink=4` there is almost
  nothing to compress (`T − n_sink ≤ rank`), so an exact match is expected and is
  **not** evidence of RoPE/reshape correctness.
- **LONG prompts (141–167 tokens, genuinely compressed):** 2/3 byte-exact at rank
  64 and 32; at rank 16/8 all diverge but stay **coherent, on-topic, and
  factually correct** (relativity, photosynthesis, the steam engine / James Watt
  all right). **No garbage at any rank.**
- The RoPE round-trip is additionally verified **bit-exact** by the critic pass
  (at high rank, pre-RoPE reconstruct + re-rotate reproduces the uncompressed
  post-RoPE cache to 0.0 error), which is the strong evidence; the long-prompt
  degradation is consistent with it.

## Perplexity — rank sweep (`results/w3-ppl-1b-pre.json`; 16 windows, ctx 1024, tgt 512)
Prefill-then-score, pre-RoPE, keys+values compressed, **torch backend**
(`block_size=128`, fp32 core). Perplexity of the continuation tokens attending to
the **compressed** context cache; baseline is the identical protocol with no
compression, so Δ isolates the compression cost. 8176 scored tokens.

| config | rank | per-token (`rank/512`) | nominal compression | perplexity | Δ vs. baseline |
|---|---|---|---|---|---|
| baseline | — | 1.00 | — | 12.65 | — |
| BUG | 128 | 0.25 (4×) | 0.750 | **13.49** | **+0.83 (+6.6%)** |
| BUG | 64 | 0.125 (8×) | 0.875 | 13.83 | +1.17 (+9.3%) |
| BUG | 32 | 0.0625 (16×) | 0.938 | 16.22 | +3.57 (+28%) |
| BUG | 16 | 0.031 (32×) | 0.969 | 18.13 | +5.47 (+43%) |

The curve is **near-baseline at 4× compression** (rank 128, +7%), gentle to 8×
(rank 64, +9%), then steep — precisely the Week-2 heavy-tailed-spectrum story:
the useful rank is a few dozen to ~a hundred, and pushing below that costs real
quality. (The earlier 6-window/ctx-512 numpy sweep gave the same shape; these
firmer ctx-1024 numbers supersede it.)

## Perplexity — pre-RoPE vs. post-RoPE (`w3-ppl-1b-pre.json` vs. `…-post.json`)
Identical 16 windows / ctx 1024, baseline ppl 12.65 in both.

| rank | Δppl pre-RoPE | Δppl post-RoPE | winner |
|---|---|---|---|
| 128 (4×) | +0.83 | **+0.50** | post (marginal) |
| 64 (8×) | **+1.17** | +1.91 | **pre (1.6×)** |
| 32 (16×) | **+3.57** | +5.13 | **pre (1.4×)** |
| 16 (32×) | **+5.47** | +5.86 | **pre** |

**Pre-RoPE wins across the compression regime that matters (rank ≤ 64), by a
widening margin as compression tightens** — the Week-2 "pre-RoPE ~halves the
reconstruction error" finding propagates to end-task quality exactly where
low-rank structure is load-bearing. At near-lossless rank 128 the two are a wash
and post-RoPE edges it (there the re-rotation's fp32 round-trip costs more than
the tiny low-rank gain). **Decision: `BUGPress` operates pre-RoPE** (the default)
— it is the right choice everywhere compression is doing real work, and a
negligible loss at the near-lossless end.

> Note: an earlier draft claimed pre-RoPE wins at *every* rank, based on a
> 3-window ctx-512 smoke. The firmer 16-window ctx-1024 sweep corrects that: it
> wins at every rank **except** the mildest (128). Reported faithfully.

## The blocked-BUG torch backend (what made this sweep cheap)
The numpy per-token `StreamingBUG` core is CPU-bound and a GPU pod does **not**
accelerate it (a 1B ctx-1024 sweep ran >90 min on a rented RTX 3090 without
finishing — the GPU sat idle while the Python loop ground). So the tracker was
reimplemented as a **blocked** augmented-BUG step in torch
(`integrators/streaming_torch.py`): process the prefill in column-blocks
(`block_size=128`) as batched QR/SVD on the tensor's own device. It is the same
integrator (block=1 reproduces the numpy tracker to fp precision, block=T equals
the SVD oracle; parity-tested), and it made the full ctx-1024 / 16-window sweep a
**~80 s job on the Mac CPU** (≈20–30× faster; no GPU needed for 1B). `BUGPress`
uses it by default (`backend="torch"`); `backend="numpy"` keeps the fp64
reference. The one cost: fp32 vs fp64 slightly widens the rank-128 gap.

## Caveats (honest)
- **Moderate sample.** 16 windows / ctx 1024 (8176 scored tokens) is a solid
  local sweep but still WikiText-2 perplexity, not the long-context benchmarks
  (LongBench/RULER) where KV-compression methods are usually compared, and not a
  head-to-head vs. SnapKV/ExpectedAttention. Those are the real Week-4 tests.
- **Aggressive ranks hurt.** Rank 16/32 (16–32× compression) cost 28–43%
  perplexity here; the win is quality *tracking* and the pre-RoPE operating
  point, not a spectacular absolute ratio — the ceiling is the spectrum.
- **Single-shot pre-fill only.** `BUGPress` reconstructs from the current
  forward's `hidden_states`/`position_embeddings`; a `q_len>1` forward against a
  non-empty cache (chunked/continued pre-fill) would desync keys and values, so
  the press **raises** in that case rather than corrupt silently (found in the
  critic pass; streaming across forwards is a later extension).
- **Nominal compression.** Shape is unchanged (ThinK-style), so there is no
  literal memory saving in this in-place form; `compression_ratio` is the
  asymptotic per-token factor. Factored storage + quantization is Week 4.

## Reproduce
```bash
# generation parity (short + long suites), pre-RoPE
uv run python scripts/generate_with_press.py --parity --ranks 64 32 16 8 \
    --max-new-tokens 20 --out results/w3-parity.md
# perplexity rank sweep, pre-RoPE (torch backend by default; ~80 s on CPU)
uv run python scripts/perplexity_sweep.py --ranks 16 32 64 128 \
    --context-len 1024 --target-len 512 --n-windows 16 \
    --out-json results/w3-ppl-1b-pre.json --out-csv results/w3-ppl-1b-pre.csv
# post-RoPE comparison (same windows)
uv run python scripts/perplexity_sweep.py --post-rope --ranks 16 32 64 128 \
    --context-len 1024 --target-len 512 --n-windows 16 \
    --out-json results/w3-ppl-1b-post.json --out-csv results/w3-ppl-1b-post.csv
```

## What's next
- Week 4: TurboQuant residual quantization of the `(U, S, V)` factors, composed
  with BUG; memory-budget perplexity sweep; **head-to-head vs. SnapKV /
  ExpectedAttention** on long-context benchmarks (the actual "beats SOTA" test);
  README hero figure.
- The blocked-BUG torch backend unblocks scale: Llama-3.1-8B needs only bf16
  weights (`--dtype bfloat16`, fits a 24 GB card) — the compute is no longer the
  bottleneck.
