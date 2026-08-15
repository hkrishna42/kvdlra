# Week 15, in plain words — one cache that wins on both axes at once

This is the readable version of the Week-15 result. It assumes you know what a
KV-cache and perplexity are, but nothing about this project. Authoritative
records, with every number: `results/w15-confirm-summary.md`,
`results/w15-complete-summary.md`, `docs/week15-significance.md`.

## The setup, in one breath

An LLM caches a key/value summary of every token it has read; that cache is the
memory bottleneck for long context. **BUG** is our compressor. It keeps two
things:

- a **low-rank "gist"** — a rank-*r* running summary of the whole history (cheap,
  blurry, carries general fluency), and
- a small **exact tier** — a handful of tokens kept verbatim, chosen by
  **surprise**: a token whose key the gist reconstructs *badly* (large residual)
  is a rare oddball worth keeping sharp. A "needle" — a random code buried in a
  haystack — is exactly such an oddball, so surprise-selection is how BUG answers
  needle-in-a-haystack questions.

For several weeks BUG owned one corner of the map: at **extreme** compression
(well under 0.1× the full cache) it could still retrieve a needle after every
eviction/low-rank method had already thrown it away — the best
**retrieval-per-byte** on the board. But it had a matching weakness: on plain
**perplexity** it was not competitive with the eviction and low-rank field
(ThinK, Palu, ShadowKV, ExpectedAttention) once those were handed a normal
memory budget.

**Week-15 question:** can a *single* BUG configuration beat that low-rank field
on **both** retrieval **and** perplexity — while using far less memory?
Historically you had to pick one.

## Discovery A — the wall between the two axes has a crack

Here is the wall. The obvious way to fix perplexity is to raise the gist rank
*r* (a sharper summary). It works for perplexity — and it **destroys
retrieval**. Push to rank 256 and every hard retrieval task drops to **0**
(measured: 16K multi-key and var-track both 0 at rank 256, capped or not). So
high rank = good perplexity, dead retrieval; low rank = good retrieval, poor
perplexity. Pick one.

Why does rank kill retrieval? Because retrieval rides on **surprise**, and
surprise *is* reconstruction error against the gist. A bigger gist reconstructs
**everything** better — including the needle. Its residual collapses toward
zero, so the "keep the surprising tokens" rule never selects it. **The summary
gets so good it hides the needle from its own selector.** Selection goes blind.

The fix is almost too simple, and it costs nothing. Keep storing and attending
with the full rank-128 gist (that is what buys perplexity), but compute the
**surprise score against only the leading rank-32 subview** of that same basis —
a deliberately *worse* reconstruction. Against a rank-32 view the needle is badly
reconstructed again, so its residual is large, so it gets selected. This is the
`--score-rank 32` knob (default-off). It adds **zero memory** — the rank-32 view
is just a truncation of bytes we already store — and costs about **+0.3%
perplexity**.

The intuition: **use a big summary to write fluently, but deliberately look with
a small summary so the needle still stands out.**

That one knob does *all* the retrieval lifting. Matched A/B, only `-s32` differs,
identical memory footprint:

| context | task | rank-128, no cap | rank-128 + s32 |
|---|---|---|---|
| 32K | multi-value | **0** | **100** |
| 32K | var-track | **0** | **100** |
| 16K | single needle | **0** | **100** |
| 16K | multi-value | 25 | **100** |
| 16K | var-track | 75 | **100** |

(Scores are % correct.) The uncapped rank-128 gist was literally hiding needles
from itself — visible even on the *easiest* single-needle test at 16K, where it
scored **0/8** until the cap un-blinded selection. Note the ceiling, though: this
is a rank-128 phenomenon. At **rank 256 you are past the cliff** — the needle is
reconstructed too well for even a rank-32 view to surface it, and `-s32` does not
resurrect it (still 0). The crack is real but narrow.

## Discovery B — two of our "baselines" were our own bugs

Setting up the comparison, we found that two baselines were losing for reasons
that were **our fault**, not theirs:

- **ShadowKV** had been posting a flat **0/0/0/0** on retrieval. That was a
  harness defect: its token-selection hook was not active during decoding, so the
  needle was excluded *by construction* — the method never got to vote. Fixed,
  ShadowKV-r64 retrieves: 16K single / multi-key / var-track = **100 / 100 / 0**
  at 0.815× memory. A real arm now.
- **Palu** had the worst perplexity in the table. That was our port
  **low-ranking the attention-sink tokens** — the first few tokens that every
  other method keeps exact. Carve the sinks out and Palu's 32K perplexity drops
  from **9.236 to 7.232**, turning it into a genuinely strong low-rank baseline.

We fixed both, even though each fix makes the baseline **stronger** and therefore
harder to beat. An honest paper has to beat the *fair* version of a competitor,
not our broken one. So the bar went up — and the headline result clears the
raised bar.

## The result — both axes, both contexts

The config `bugSseed-r128-h1024-s32` (rank-128 gist; the Week-14 warm-up seed;
exact-tier budget 1024; score-rank cap 32):

| context | memory | perplexity | single | multi-key | multi-value | var-track |
|---|---|---|---|---|---|---|
| 16K | 0.191× | 5.434 | 100 | 100 | 100 | 100 |
| 32K | 0.159× | 7.353 | 100 | 100 | 100 | 100 |

**A clean 100 on all four retrieval tasks, at both context lengths** — and the
perplexity **ties** the low-rank field:

- **32K:** 7.353 vs ThinK 7.196 / Palu 7.232 — a gap of **+0.031 / +0.024
  bits/token**, inside the pre-registered tie band.
- **16K:** 5.434 vs ThinK 5.413 / Palu 5.419 — **+0.0056 / +0.0042 bits/token**,
  and only +0.044 from *uncompressed full KV* (near-lossless).

It does this at **0.16–0.19× memory** — that is **3.2–4.7× less than the field at
32K, 2.6–3.9× less at 16K** (ThinK sits at 0.75×, Palu at 0.50×). On the *hardest*
task, var-track, the 100 is not even a tie: it beats the field outright (Palu 25
at 32K). Both axes, both contexts, at a large memory discount. That is the thing
we could not do before Week 15.

Two mechanisms combine to get here, and they split cleanly: the **warm-up seed**
carries perplexity and multi-key; the **score-rank cap** carries var-track and
multi-value. Neither alone is the both-axes point; together they are.

## What "ties on perplexity" actually means

Perplexity differences are **ratios**, not gaps, and the honest unit is
bits/token = log2(ppl_a / ppl_b) (`docs/week15-significance.md`). On that scale
the Week-15 gaps are **0.02–0.03 bits/token at 32K and 0.004–0.006 at 16K** —
statistically a tie at our eval size (paired on identical windows, n=8). For
scale: **9.0 vs 7.5 is 0.263 bits/token**, a decisive, model-generation-sized
difference. We are roughly 10× under that. This is a real tie, not a rounding
dodge.

One thing we will **not** claim: BUG does *not* match **uncompressed** full KV on
perplexity at 32K — it is **+0.076 bits/token (~5%)** behind full. That is
expected at ~6× compression, and we say it plainly. The claim is "matches the
compression **field**," which it does — not "matches an uncompressed cache,"
which it does not.

## Honest limits

- **The one airtight single retrieval result is 32K var-track** (0/4 → 4/4,
  Wilson intervals disjoint). Several 16K lifts are small-n: 16K var-track
  (6/8 → 8/8) has overlapping intervals, and 16K multi-value (1/4 → 4/4) is
  marginal. The *direction* is consistent everywhere; a higher-n rerun would
  tighten the weak cells.
- **Rank 256 stays dead.** The score-rank crack is a rank-128 trick; it does not
  scale up.
- **One genuine oddity:** the *uncapped* rank-128 arm at 16K gets multi-key 100
  but single-needle 0 — task-specific weirdness we do not fully explain. The
  `-s32` arm is uniformly clean (100 everywhere), and it is the one we ship.

## The one thing to remember

For years the rank knob forced a choice: **sharp summary = good text, blurry
summary = good retrieval, never both.** Week 15 breaks the trade with a free,
default-off trick — **store and attend at high rank, but *select* the exact tier
at low rank** — and lands a single config that ties the low-rank field's
perplexity, scores 100 on every retrieval task at 16K and 32K, and does it at
3–5× less memory. We also repaired two competitors into their stronger form
before beating them. Not a win over uncompressed KV — an honest win over the
compression field, on both axes at once. (Cost: about $21 of GPU credit; every
rented machine was destroyed afterward.)
