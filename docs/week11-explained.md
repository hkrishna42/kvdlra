# Week 11, in plain words — can "BUG" find a needle in a huge document?

This is the no-jargon version of `docs/week11.md`. If you've never touched a
KV-cache, start with `docs/week9-explained.md` first, then come back here.

## The one-minute recap

When an LLM reads a long prompt it stores a little summary of every token — the
**KV-cache** — and that cache eats memory. Two ways to shrink it:

- **Eviction** (MorphKV, SnapKV, ExpectedAttention): *throw away* the tokens that
  look unimportant, keep the rest **exactly**. Weakness: if you throw away something
  you later needed, it's gone.
- **BUG** (this project): don't throw anything away — keep one **blurry photo of the
  whole history**. Weakness: fine detail (an exact 5-digit code) blurs away.

## The problem this week: the "needle"

The classic hard test: bury a secret 5-digit code somewhere in a **32,000-token**
document (a "needle in a haystack"), then ask the model to read it back.

Here's the trap for BUG. A needle is a **rare, low-importance, oddball** token. BUG's
blurry summary is great at the *typical* stuff and worst at exactly this kind of rare
oddball. So last week (Week 10) we measured the wall bluntly: at the big 8B model,
32K context, **plain BUG found the needle 0% of the time — at every setting we
tried.** Meanwhile ExpectedAttention (an eviction method) found it **100% of the
time** using about a tenth of the memory.

**This week's question, put fairly:** can we make BUG find that needle using *no more
memory* than ExpectedAttention needs — or prove it can't?

## The idea: keep the *surprising* tokens exactly

BUG already has a side-pocket where it can keep a handful of tokens **perfectly
sharp** (not blurred) — normally reserved for the "loudest" tokens. But a needle is
the *opposite* of loud; it's **quiet and weird**. So instead of "keep the loudest,"
we told that pocket: **"keep the most *surprising* tokens"** — the ones BUG's blurry
summary predicts worst. A needle is the single most surprising thing in the
haystack, so it lands in the sharp pocket and survives.

One wrinkle we caught with a cheap pre-test: the 5-digit code is actually ~3
word-pieces, and one of those pieces isn't itself "surprising." Pure surprise kept 2
of the 3 and dropped one. Fix: when we keep a surprising token, **keep its immediate
neighbours too** (span expansion). With that, all 3 pieces survive — 3 out of 3, even
in a *tiny* sharp pocket.

## Does it work? Yes — a real win

At 8B / 32K, finding the needle (higher = better), and how much memory each uses
(smaller = better; "1.0×" is the full uncompressed cache):

| method | finds the needle | memory |
|---|---|---|
| full (no compression) | 100% | 1.0× |
| ExpectedAttention (the bar to beat) | 100% | 0.10× |
| plain BUG (last week) | **0%** | 0.03×–0.26× |
| **BUG + surprise-pocket** | **100%** | **as low as 0.009×** |

**BUG went from 0% to 100% — using about 11× *less* memory than ExpectedAttention.**
The wall isn't just matched, it's beaten.

## The catch we thought we had — and the correction

Here's the part we made ourselves check, because it's easy to over-claim. Was it
BUG's **blurry summary** that found the needle, or just the new **"keep the
surprising tokens"** rule?

To find out, we ran a control we call **bugEVICT**: same "keep-the-surprising-tokens"
rule, but with the blurry summary **shrunk to almost nothing**. On the single-needle
test *and* on general text quality (perplexity), **bugEVICT tied the full version** —
same 100% needle-finding, equal-or-better perplexity — at a fraction of the memory
(0.009× vs 0.043×). Our first read was: the blurry summary is **dead weight**; the win
is just the "keep the surprising tokens" rule working as smart eviction.

**Then we ran the harder tests, and that conclusion flipped.** The single needle is
the *easy* retrieval task — one secret code, no distractors. The full benchmark has
three harder ones: **multi-key** (several codes, fetch the right one), **multi-value**
(one key, several values), and **variable-tracking** (follow a chain of assignments).
On those:

| test | bugEVICT (almost no summary) | BUG + surprise (real summary) |
|---|---|---|
| single needle | 100% | 100% |
| multi-key | **0%** | **83%** |
| multi-value | **0%** | 100% |
| variable-tracking | **0%** | 100% |

(both at 32K / 8B; bugEVICT at its smallest 0.009× setting, BUG+surprise at 0.043×.)

**When the exact shelf is small, bugEVICT drops to zero the moment the task needs more
than one fact.** The full version — *with* the blurry summary — handles all four. So the
summary looks **not** dead weight after all: it helps the method answer questions that
need many soft facts at once, not just fish out one sharp code. The single-needle test
was too easy and flattered the stripped-down control.

The precise, honest statement: **the blurry summary barely helps for plain text quality
(perplexity) or for finding one needle — but it helps on the hard multi-fact
questions.** How strongly? A lean, not a slam dunk: the gap is biggest when the exact
shelf is small (with a bigger shelf, bugEVICT catches up on some tasks), it only shows
up at the longer 32K length, and each number rests on just a handful of trials. So the
surprise rule and the summary *both* look useful — worth one more run to be sure.

## So who wins, BUG-surprise or ExpectedAttention?

Use the full version (BUG + surprise, summary included — we call it **bugS**), not the
stripped-down control. Against ExpectedAttention it's an honest **trade**:

- **BUG+surprise (bugS):** matches or beats EA on every retrieval test — and **wins
  multi-key outright, 83% vs 50%** — at **less than half the memory**.
- **ExpectedAttention:** slightly **better general text quality** (perplexity).

So the win is **memory + hard retrieval**; the cost is a bit of perplexity. Neither
dominates on every axis, and we report both directions.

## Side quest: we also repaired last week's benchmark

While the big run was going, we fixed **three bugs** that had been silently breaking
Week 10's retrieval tests for several methods (each fix now has its own guard test):
MorphKV crashed on a short final chunk; SnapKV crashed on RULER's short queries; and
one bad dataset used to abort the *entire* benchmark instead of just skipping. Plus a
GPU fix for ShadowKV. With those in, the **full 16K retrieval scoreboard now runs for
every method.** Highlights: ExpectedAttention is the best low-memory all-rounder but
even it stumbles on "variable tracking"; plain BUG is weak across the board (the very
wall that motivated this week); and ShadowKV scores 0 on our memory-only yardstick
because its real strength is *speed*, which we deliberately don't reward here.

## Honest caveats (so nobody over-claims)

- These tests are **small** (2–6 samples per cell; the perplexity numbers are 2
  samples each). The big gaps in the table (0% vs 100%) are clear, but treat the exact
  percentages as rough.
- On plain text quality (perplexity) the blurry summary still barely helps — that part
  of the earlier "overhead" finding stands. The summary earns its keep on **multi-fact
  retrieval**, not on perplexity.
- **Variable-tracking is hard for every compressed method** except at light
  compression — even ExpectedAttention only clears it because it keeps more memory.
- Still to do: a run on the smaller **1B model**, and more samples to tighten the
  numbers.

## The one thing to remember

We beat a wall that plain BUG failed at 0% — BUG can now find a 32K needle at ~11×
less memory than the leading eviction method. We first thought BUG's blurry summary
was dead weight, but the harder multi-fact tests suggest it **helps**: strip it out and
the method drops toward 0% the moment a question needs more than one fact (at least when
the exact shelf is small). So the win looks like it comes from **both** pieces — the
"keep the surprising tokens" rule *and* the blurry summary — traded against slightly
worse text quality than ExpectedAttention. It's a lean worth one confirming run. A real
win, honestly attributed. (Cost: ~$17 of GPU credit, ~$9 left; the rented machines
were shut down after.)
