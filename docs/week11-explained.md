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

## The honest catch: it's smart eviction, not the summary

Here's the part we made ourselves check, because it's easy to over-claim. Was it
BUG's **blurry summary** that found the needle, or just the new **"keep the
surprising tokens"** rule?

To find out, we ran a control we call **bugEVICT**: same "keep-the-surprising-tokens"
rule, but with the blurry summary **shrunk to almost nothing**. If the summary
mattered, this stripped-down version should do worse.

It didn't. **bugEVICT tied or beat the full version on everything** — same 100%
needle-finding, equal-or-better text quality (perplexity) — at a *fraction* of the
memory (0.009× vs 0.043×). In other words:

> The needle was found by the **"keep the surprising tokens" rule**, working like a
> smart form of eviction. BUG's blurry summary added nothing here and cost ~5× the
> memory — it's **dead weight**.

That's not a disappointment; it's the honest result. And it's the *same lesson* the
whole project keeps finding: BUG's summary carries a fixed overhead that rarely pays
for itself. The genuinely useful thing BUG contributed is its **surprise signal** —
a good way to decide *what to keep* — not its compression.

## So who wins, BUG-surprise or ExpectedAttention?

It's an honest **trade**, not a clean sweep:

- **BUG-surprise (bugEVICT):** same 100% needle-finding at **3–12× less memory**.
- **ExpectedAttention:** slightly **better general text quality** (perplexity).

Pick your priority. Neither dominates — and we report both directions.

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

- The "the summary is dead weight" conclusion rests on **small tests** (2 samples per
  quality number, ~2% margins). It means "no measurable benefit here," not "provably
  harmful."
- The completed 32K result is the single-needle test; the harder multi-key and
  multi-value versions, and a run on the smaller 1B model, are still to do.
- **bugEVICT isn't really "BUG"** — that's the whole point of the control. The win
  belongs to BUG's *surprise idea used as eviction*, not to BUG's compression.

## The one thing to remember

We beat a wall that plain BUG failed at 0% — BUG can now find a 32K needle at ~11×
less memory than the leading eviction method. But when we checked *why*, it was BUG's
**"keep the surprising tokens" rule** doing the work, not its blurry summary. A real
win, honestly attributed. (Cost: ~$17 of GPU credit, ~$9 left; the rented machines
were shut down after.)
