# Q-BUG, in plain language: a good idea that measured small

*(Week-12 bugS-perplexity Track 1. Companion to `results/w12-qbug-summary.md`.)*

## The setup

`bugS` compresses an LLM's KV cache into a small **low-rank summary** (the "gist")
plus a tiny **exact tier** that keeps a few surprising tokens verbatim. The gist
carries text quality (perplexity); the exact tier carries retrieval (finding a
specific fact). Week 12 proved those two jobs live in **separate parts of the
state** — so we can try to improve quality *without* touching the part that does
retrieval.

The quality gap we wanted to close: at 32K tokens, `bugS`'s perplexity trails the
best eviction baseline (ExpectedAttention). The usual knob — raise the rank of the
gist — is spent: more rank *collapses* retrieval. So we needed a different lever.

## The idea (Q-BUG)

The gist minimizes reconstruction error on the keys in the plain (Frobenius) sense —
it treats every direction as equally important. But attention doesn't: it only
"reads" keys along the directions the **queries** actually point. Error in a
direction no query looks at is free; error along a query-heavy direction costs a
logit.

So: **weight the gist by where the queries look.** Estimate the query energy per
feature once (a frozen diagonal `w_key`), whiten the keys by it before compressing,
un-whiten when reading back. Same integrator, same rank, essentially zero extra
memory — but the rank now spends its fidelity on what attention reads. We call it
**Q-BUG**.

## Did it work? Yes — but small.

A cheap CPU probe on saved activations was very encouraging: query-whitening cut the
*attention-output error* of the rank-r reconstruction by **30–44%**. That funded the
build. We implemented it (with tests proving `w_key=1` is identical to plain bugS,
the round-trip is lossless at full rank, and a planted needle still lands in the
exact tier), then confirmed on the real 8B model at 16K and 32K:

| config | plain bugS ppl@32K | Q-BUG ppl@32K | gain |
|---|---|---|---|
| r32-h256 | 9.164 | 9.092 | −0.79% |
| r128-h1024 | 8.117 | 8.085 | −0.39% |

Real, consistent, at matched memory — but **~1%**, not the large gain the 30–44%
probe number implied. Both pre-registered bars (r32 ≤ 8.90, r128 < 8.00) were
**missed** by 3–4×.

## The lesson worth keeping

The probe measured **attention-output error at one layer**; perplexity is the
next-token loss after softmax and 31 more layers. Those absorb most of the
perturbation, so the probe **over-predicted the end-to-end gain by ~30–40×**. That's
not a bug in the probe — it's a reminder that *a proxy metric is a ceiling, not a
promise*. Every future $0 probe now states what proxy it measures, and any bar
derived from a proxy is treated as an upper bound.

## And retrieval?

Query-whitening slightly perturbs which tokens the surprise-selected exact tier
keeps. Measured effect at 32K (n=4): multi-value retrieval is **preserved exactly**
(100/100 at both ranks), var-track is preserved at r128 and one trial soft at r32,
and multi-key is **~1 trial soft at both ranks** (50 vs 67/75). Every cell is within
the ±25-point noise of a 4-trial all-or-nothing metric — but the multi-key softness
is *directional*, so it may be a small real cost rather than pure noise. (This is
Q-BUG's own n=4 vs the pooled baseline, not a matched-n test; Week-13 re-runs it at
higher n to separate the two.)

## Verdict

Q-BUG is an **honest bounded result**: a small (~1%) free-memory perplexity gain
with retrieval preserved within noise. It ships as a **default-off knob** with unit
and accounting tests — a real but minor lever, not a headline win. The mechanism
(change the objective to the metric attention uses) is sound; the *magnitude* at
these operating points is just small, and the walls we set out to move (the ppl gap
to eviction) turn out not to be mainly a "wrong-metric" problem. That itself is a
useful finding, and it points Week 13 at different mechanisms.
