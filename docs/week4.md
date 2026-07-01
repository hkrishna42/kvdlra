# Week 4 — TurboQuant residual quantization composed with BUG

**Summary.** We add TurboQuant vector quantization (arXiv:2504.19874) to the
`kvdlra` stack and compose it with the streaming BUG low-rank compressor: BUG
reduces the *rank* (feature axis), TurboQuant reduces the *bits* (per-coordinate
axis), and the two **multiply**. Quantizing the BUG coordinate vectors at 4 bits
roughly **halves the stored cache for a negligible perplexity cost**. On a fair
head-to-head, **BUG+TurboQuant beats SnapKV and Expected Attention in the
aggressive-compression regime (~10×, memory ≈0.10×): 13.86 vs 15.57 vs 14.49
perplexity**; eviction methods win only at mild compression (≥0.5× memory).

All numbers: Llama-3.2-1B (ungated `unsloth` mirror), WikiText-2,
prefill-then-score, ctx 1024 / target 512 / 16 windows (8176 scored tokens),
baseline ppl **12.65**. Torch backend, fp32 core.

## What was built
- **`quant/polar.py` — PolarQuant** (TurboQuant §2): fixed data-oblivious
  rotation Π (QR of a Gaussian) makes a unit vector's coordinates near-independent
  and Beta-distributed, then a per-coordinate Lloyd–Max scalar quantizer. Measured
  distortion within the paper's **2.7× of the info-theoretic floor** (b=2/3/4 →
  1.83/2.14/2.37×). 9 tests.
- **`quant/qjl.py` — QJL** (TurboQuant §3): 1-bit sign hash + unbiased
  inner-product estimator (`√(π/2)/m · Sᵀz`). Unbiased to 0.25%; variance at the
  Lemma-4 bound `π/2m·‖y‖²` (ratio 0.98). 6 tests.
- **Composition in `BUGPress`** (`quant_bits`): take the BUG basis `U` and
  coordinates `C = UᵀM`, PolarQuant-quantize the per-token coordinate columns,
  reconstruct `UĈ`. Factors are **pre-RoPE**, so the fixed rotation Π and RoPE
  never need to commute (RoPE is applied last to the reconstruction) — see
  `docs/notes/turboquant-rope-interaction.md`.
  **Precision note:** "BUG+TurboQuant" here uses TurboQuant's *PolarQuant* scalar
  stage only. The QJL 1-bit residual de-biases *inner-product estimates* (Mode B,
  estimator-at-attention); in our reconstruct-then-attend press (Mode A) keys are
  rebuilt explicitly and QJL is not applied — it is implemented and tested
  standalone as a drop-in for a future Mode-B kernel.
- **`press/compat.py`** — shim so stock kvpress presses run on transformers ≥5.8
  (same `cache_position` issue fixed for BUGPress in Week 3).
- Scripts `w4_hybrid_sweep.py` (BUG vs BUG+TurboQuant on an honest memory axis)
  and `w4_head_to_head.py` (vs SnapKV / Expected Attention → hero figure).

## The composition works: 4-bit ≈ half the memory, ~free
Stored-memory model per compressed layer: `U` (fp16) + quantized coords (b bits)
+ per-token norms (fp16) + exact sinks, over the full fp16 cache. (`U` amortizes
as context grows.)

| rank | fp factors | 4-bit coords | Δppl cost of quant |
|---|---|---|---|
| 32 | 0.097× / +3.57 | **0.053× / +3.58** | +0.01 |
| 64 | 0.191× / +1.17 | **0.099× / +1.20** | +0.03 |
| 128 | 0.378× / +0.83 | **0.193× / +0.94** | +0.11 |

4-bit PolarQuant on the coordinates ≈ halves the stored cache for +0.01–0.11 ppl.
3-bit is a small step further; **2-bit is the cliff** (e.g. r64 0.084× / +5.7).
This is the Week-3 "compose to beat SOTA" argument made concrete: the low-rank and
quantization compression factors multiply.

## Head-to-head vs. SOTA (the "beats SOTA?" test)
Fair comparison on one perplexity-vs-memory axis (eviction methods scored at the
targets' true positions — see the fairness fix below). Hero figure:
`figures/week4/hero.png`.

| memory | BUG+TurboQuant | SnapKV | Expected Attention |
|---|---|---|---|
| ~0.10× | **13.86** (r64/4b) | 15.57 | 14.49 |
| ~0.19× | **13.59** (r128/4b) | — | — |
| ~0.25× | — | 14.18 | 13.58 |
| ~0.50× | — | 13.26 | **12.99** |

**Verdict — a real win in the high-compression regime, honest about the rest.**
At **~0.10× memory (10× compression)** BUG+TurboQuant (13.86) beats both
Expected Attention (14.49) and SnapKV (15.57) — the regime where you actually
need compression. Around 0.19–0.25× the methods are comparable (BUG+TQ 13.59 at
0.19× ≈ Expected Attention 13.58 at 0.25×, i.e. same quality at less memory). At
**mild compression (0.5×)** eviction wins — Expected Attention is near-lossless
(12.99) — which is unsurprising: with most tokens retained, exact keys beat any
low-rank reconstruction. So BUG+TurboQuant is the method of choice when the cache
must be small; eviction when it can be large.

**Composition > pure low-rank at a fixed budget.** At the same ~0.10× memory,
BUG+TurboQuant (r64, 4-bit → 13.86) massively beats BUG-alone (r32, fp16 → 16.22):
spending the bit budget on 4-bit quantization of a *higher-rank* factorization
beats an fp16 *lower-rank* one. Quantization and low-rank genuinely compose rather
than substitute — the core Week-4 thesis, confirmed.

## Needle-in-a-haystack retrieval (SnapKV's home turf)
Eviction methods are built for long-context *retrieval*, so we tested it directly:
hide a needle ("The secret passcode is NNNNN.") at depth `d` in a haystack,
compress it, then ask for it **after** compression (chat-templated, question fed
post-compression so the answer attends to the compressed cache). 1B baseline
retrieves 15/15. Accuracy over 5 depths × 3 codes (`figures/week4/needle.png`):

| method | memory | retrieval acc |
|---|---|---|
| baseline | 1.0× | 15/15 |
| **BUG+TurboQuant** r128/4b | **0.147×** | **15/15** |
| Expected Attention | 0.20× / 0.50× | 15/15 / 15/15 |
| BUG r64 | 0.167× | 9/15 |
| BUG+TurboQuant r64/4b | 0.076× | 7/15 |
| **SnapKV** | 0.20× / 0.50× | **3/15 / 3/15** |

**Surprising, honest result — it partly *reverses* the naïve expectation.** We
expected eviction to dominate retrieval and low-rank to "smear" the needle. In
fact: **Expected Attention and BUG+TurboQuant (r128) both retrieve perfectly**,
BUG+TQ at *less* memory (0.147× vs 0.20×). **SnapKV fails badly (3/15) and more
memory does not help** — in the compress-then-query setting it scores tokens by
the prompt's *own* recent-window attention (the question is not yet present), so a
non-salient needle is evicted regardless of budget. BUG's *uniform* low-rank
reconstruction instead preserves every token — but only at **sufficient rank**:
rank-128 retrieves 15/15, rank-64 drops to 9/15 and aggressive rank-64/4-bit to
7/15 (the reconstruction blurs a sharp fact when the rank is too low). So on
retrieval: Expected Attention and adequately-ranked BUG+TurboQuant win; SnapKV is
the weak baseline here; and BUG's rank is the knob that governs whether a sharp
fact survives.

### A fairness fix (honest note)
Our first head-to-head was **invalid**: the prefill-then-score harness let the
model derive target positions from the cache length, but eviction presses shrink
the cache, so targets got wrong RoPE positions (symptom: SnapKV perplexity *worse*
at keep-50% than keep-10% — physically impossible). BUG keeps full seq-length so
it was unaffected, which would have flattered BUG. We pass explicit `position_ids`
= true positions (a no-op for BUG/baseline, a correction for eviction); the
ordering is now sane. Numbers here are post-fix.

## Caveats (honest)
- **Nominal → real memory.** BUGPress reconstructs same-shape tensors in place, so
  the memory ratios are the *factored-storage* cost (what a real implementation
  would store), not a literal in-place saving. The fixed basis `U` amortizes only
  for long context.
- **WikiText-2 perplexity**, not long-context benchmarks (LongBench/RULER), and 1B
  scale. The eviction baselines are designed for long-context retrieval; a short
  fixed-window perplexity is not their home turf, so read the head-to-head as
  indicative, not a leaderboard result.
- **8B scale-up: deferred.** The plan calls for an 8B validation. Two vast.ai pods
  failed at the infrastructure level today (SSH key injection; container create) —
  not our code. The blocked-BUG torch backend + `--dtype bfloat16` make 8B ready
  to run on a working pod; recorded as a clean follow-up.
- **QJL Mode B (estimator-at-attention)** is implemented + tested but not wired
  into the reconstruct-then-attend press; it needs a custom attention kernel and is
  future work.

## Reproduce
```bash
uv run python scripts/w4_hybrid_sweep.py --ranks 32 64 128 --bits fp 4 3 2 \
    --context-len 1024 --target-len 512 --n-windows 16
uv run python scripts/w4_head_to_head.py --ratios 0.5 0.75 0.9 \
    --context-len 1024 --target-len 512 --n-windows 16
```
