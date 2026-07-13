# Week 9, in plain words — does "BUG" help shrink an LLM's memory?

This is the no-jargon version of `docs/week9.md`. If you've never touched a
KV-cache, start here.

## The problem, simply

When an LLM reads a long prompt, it stores a little summary of every token it has
seen so far — the **KV-cache**. The longer the context, the bigger this cache, and
it can dominate memory. So people compress it. Two big families of tricks:

- **Eviction** (e.g., **MorphKV**, **SnapKV**): *throw away* the tokens that don't
  look important, keep the rest exactly. Simple and cheap. Weakness: if you throw
  away a token you later needed, it's **gone forever** ("forgetting").
- **BUG** (this project — a "Dynamical Low-Rank Approximation" method): instead of
  throwing tokens away, keep a **compressed low-rank summary of *everything***. Like
  keeping a blurry photo of the whole history instead of a few sharp photos.
  Weakness: the summary is **blurry** — fine detail (an exact 5-digit code) can be
  lost.

The whole project asks: **is BUG's "blurry summary of everything" a good way to
compress the KV-cache, compared to eviction's "sharp photos of a few things"?**

## What we already knew (Weeks 1–8)

- BUG is an *excellent* low-rank tracker — it summarizes the history about as well
  as the mathematically best possible low-rank summary.
- But the KV-cache **isn't very compressible** in the first place (its "spectrum"
  is heavy-tailed — you need a fairly big summary to capture it). So BUG is **not a
  magic compressor**: at very aggressive compression it loses to eviction; at
  moderate compression it wins. It's competitive, not a knockout.

## Week 9: can BUG *help* eviction instead of *replacing* it?

Since the two methods fail in *opposite* ways (eviction forgets; BUG blurs), we
tested three ways BUG could **complement** eviction. Verdicts:

### 1. BUG as a "recovery net" for what eviction forgets — ✅ **this works**

Idea: let eviction do its thing, but keep a cheap BUG summary of the tokens it
throws away. When you later ask about a forgotten token, the summary can recover
it.

**Test:** hide a secret code in a long document, at a spot eviction will discard,
then ask for it.
- **1B model:** eviction gets it **0 out of 4 times** (it forgot). BUG's summary
  gets it **4 out of 4 times** — using *less* memory. **Clear win.**
- **8B model (bigger):** at first BUG got 0 too — but that turned out to be a
  *fixable* fidelity problem (the bigger model has 2× more features, so BUG's
  summary needs to be 2× "sharper" — higher rank). With a bigger summary
  (**rank 128**), BUG recovers **6 out of 6** where eviction gets **0 out of 6**,
  using **1.8× the memory**. Still a win — you just pay some extra memory at scale.

See `figures/week9/recall_8b.png`. **Bottom line: BUG is genuinely good at
recovering content that eviction forgets — especially "find one fact among many,"
which is exactly where eviction struggles.**

### 2. Can a "best-of-both" cache beat both across all memory budgets? — ❌ bounded

We built a knob that blends eviction and BUG. It never does *better* than just
picking the winner at each budget — because BUG always pays a fixed "overhead" (it
needs to store the summary's basis) that eviction doesn't. So there's no free lunch
from blending. Honest negative.

### 3. Can BUG's math give eviction a smarter "what to keep" rule? — ❌ bounded

BUG can measure how "surprising" each token is (how badly its blurry summary
predicts it). We hoped surprising tokens are the ones worth keeping. They carry
*some* new information, but not enough — plain attention-importance is still the
better keep-rule. Honest negative (with one small exception: surprise *is* useful
for the recovery-net in idea #1).

## The one picture to remember

`figures/week9/comparison_frontier.png` — **perplexity (lower = better) vs how much
memory you allow.** Eviction (green) and BUG (orange) **cross**: eviction wins when
memory is extremely tight; BUG wins at moderate budgets, getting close to the
uncompressed "full" quality. Neither dominates — it's a **regime split**.

## Honest caveats (so nobody over-claims)

- The recovery-net winning version is **pure BUG** (a summary of everything), not
  literally "eviction + a BUG net." A cheaper combined version is untested.
- At 8B the recovery win **costs extra memory** (1.8×–5.3×), growing with context.
- Tests use small trial counts and one model family; the *direction* is solid, the
  exact numbers are indicative.

## What's next

A proper head-to-head at **long context (32K–64K)**: BUG at several ranks vs
**SnapKV**, **ShadowKV**, and **MorphKV**, measuring **both** perplexity **and**
exactly how much memory each uses. See `docs/week10-kickoff.md`.
