# Week 3 — `BUGPress` in kvpress: generation correctness + perplexity

**Summary.** The Week-2 streaming BUG tracker is now a working NVIDIA-`kvpress`
press, `BUGPress`, that replaces a layer's KV cache with its rank-`r` dynamical-
low-rank reconstruction during pre-fill. Generation is **correct** (no garbage at
any rank; a working RoPE round-trip verified bit-exact), perplexity degrades
**gracefully and monotonically** with rank (near-baseline at 4× compression),
and **pre-RoPE beats post-RoPE at every rank** — confirming, at the
generation/perplexity level, Week-2's reconstruction-error finding. Pre-RoPE is
therefore the operating point.

All numbers below are Llama-3.2-1B (ungated `unsloth/Llama-3.2-1B-Instruct`,
config-identical), fp32, CPU. They are a **preliminary CPU sweep** (small window
counts); the full/long-context and 8B sweeps are pod work (see *Caveats*).

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

## Perplexity — rank sweep (`results/w3-ppl.json`; 6 windows, ctx 512, tgt 256)
Prefill-then-score, pre-RoPE, keys+values compressed. Perplexity of the
continuation tokens attending to the **compressed** context cache; baseline is
the identical protocol with no compression, so Δ isolates the compression cost.

| config | rank | per-token (`rank/512`) | nominal compression | perplexity | Δ vs. baseline |
|---|---|---|---|---|---|
| baseline | — | 1.00 | — | 14.29 | — |
| BUG | 128 | 0.25 (4×) | 0.750 | **14.59** | **+0.31 (+2.1%)** |
| BUG | 64 | 0.125 (8×) | 0.875 | 15.55 | +1.26 (+8.8%) |
| BUG | 32 | 0.0625 (16×) | 0.938 | 20.20 | +5.91 (+41%) |
| BUG | 16 | 0.031 (32×) | 0.969 | 22.66 | +8.37 (+59%) |

The curve is **near-baseline at 4× compression** (rank 128, +2%), gentle to 8×
(rank 64, +9%), then steep — precisely the Week-2 heavy-tailed-spectrum story:
the useful rank is a few dozen to ~a hundred, and pushing below that costs real
quality.

## Perplexity — pre-RoPE vs. post-RoPE (matched: 3 windows, ctx 512, tgt 256)
The deciding experiment (`results/w3-ppl-smoke.json` vs. `…-smoke-post.json`;
identical windows, baseline ppl 10.32 in both).

| rank | Δppl pre-RoPE | Δppl post-RoPE | pre-RoPE advantage |
|---|---|---|---|
| 64 | **+1.35** | +3.86 | 2.9× smaller penalty |
| 32 | +7.68 | +9.76 | 1.3× |
| 16 | +10.14 | +10.80 | 1.1× |

**Pre-RoPE wins at every rank**, most decisively where the model is still usable
(rank 64: 2.9× smaller perplexity penalty). This confirms Week-2's "pre-RoPE
roughly halves the reconstruction error" now propagates to end-task quality.
**Decision: `BUGPress` operates pre-RoPE** (the default).

## Caveats (honest)
- **Small CPU sweep.** 3–6 windows / ctx 512 is noisy and short-context; the
  headline is the *shape* (monotone, pre>post, near-baseline at 4×), not the
  exact ppl values. The full sweep (many windows, ctx ≥1024, and the 8B model)
  is GPU-pod work — the numpy fp64 per-token BUG core makes long contexts slow on
  CPU.
- **Aggressive ranks hurt.** Rank 16/32 (16–32× compression) cost 40–60%
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
# perplexity rank sweep (pre-RoPE)
uv run python scripts/perplexity_sweep.py --ranks 16 32 64 128 \
    --context-len 512 --target-len 256 --n-windows 6
# pre- vs post-RoPE at matched windows
uv run python scripts/perplexity_sweep.py --post-rope --ranks 16 32 64 \
    --context-len 512 --target-len 256 --n-windows 3 \
    --out-json results/w3-ppl-smoke-post.json --out-csv results/w3-ppl-smoke-post.csv
```

## What's next
- Full perplexity sweep on the GPU pod (more windows, ctx ≥1024; optionally
  Llama-3.1-8B) to firm up the numbers.
- Week 4: TurboQuant residual quantization of the `(U, S, V)` factors, composed
  with BUG; memory-budget perplexity sweep; README hero figure.
