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
| multi-key | **0%** | **67%** |
| multi-value | **0%** | 100% |
| variable-tracking | **0%** | 100% |

(both at 32K / 8B; bugEVICT at its smallest 0.009× setting, BUG+surprise at 0.043×.
Numbers are the final **pooled** counts across all runs — 6–14 trials per cell — which
supersede the earlier small-n snapshot; multi-key softened from an early 83% to 67%.)

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

- **BUG+surprise (bugS):** matches or ties EA on every retrieval test (with the pooled
  numbers multi-key is an honest **tie, 67% vs 67%** — an earlier small-n snapshot read
  83% vs 50%) — at **less than half the memory**, and it alone keeps variable-tracking
  at 100% (EA pooled: 83%).
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

- These tests are **small** (2–14 pooled trials per cell after the confirming reruns;
  the perplexity numbers are 2 samples each). The big gaps in the table (0% vs 100%) are clear, but treat the exact
  percentages as rough.
- On plain text quality (perplexity) the blurry summary still barely helps — that part
  of the earlier "overhead" finding stands. The summary earns its keep on **multi-fact
  retrieval**, not on perplexity.
- **At 16K, variable-tracking is hard for every compressed method** at low memory.
  At 32K the picture flips: bugS-r32-h256 holds it at 100 (0.043×) while pooled EA
  softens to 83 (0.100×).
- Still to do: a full decision-table run on the smaller **1B model** (1B probes exist),
  and more samples for the n=4 cells.

## The one thing to remember

We beat a wall that plain BUG failed at 0% — BUG can now find a 32K needle at ~11×
less memory than the leading eviction method. We first thought BUG's blurry summary
was dead weight, but the harder multi-fact tests suggest it **helps**: strip it out and
the method drops toward 0% the moment a question needs more than one fact (at least when
the exact shelf is small). So the win looks like it comes from **both** pieces — the
"keep the surprising tokens" rule *and* the blurry summary — traded against slightly
worse text quality than ExpectedAttention. It's a lean worth one confirming run. A real
win, honestly attributed. (Cost: ~$20 of GPU credit across the finalizing campaign incl. the rank-256 check;
~$26 left after a top-up; every rented machine was shut down after.)

---

## Update (2026-07-18): a mystery solved, and a "balanced" setting

Three things happened in the wrap-up session. We solved a genuinely weird mystery —
why our method was *worse* at finding things in a **shorter** document — we built a
mid-range setting that also competes on plain text quality, and we found a **cliff**
just past that setting. All come with honest fine print.

### The mystery: why was 16K *harder* than 32K?

Strange but true: BUG+surprise found almost everything in 32,000-token documents, but
kept missing things in 16,000-token ones. Shorter should be easier. What's going on?

**The answer is a warm-up window. Think of moving to a new city.** Your first few
days, *everything* is new — every street, every shop sign, every sound. Nothing
stands out, because it's all equally unfamiliar. After a couple of weeks the basics
are familiar, and now the genuinely odd thing — a llama in the park — instantly pops.

BUG's "surprise" score works the same way. A token is surprising if the blurry
summary predicts it badly. But at the *start* of a document, the summary is brand
new — it predicts *everything* badly, plain filler included. So a secret code planted
early streams past looking no more surprising than the sentence around it, and it
never gets picked for the sharp pocket. Only after roughly the first 4–5,000 tokens
does the summary know the "normal" of the document well enough for a weird code to
stand out.

Now the paradox dissolves. These benchmarks hide the codes at *proportional* spots —
say, 10%, 20%, 30% of the way in. In a 16K document, "10% in" is ~1,600 tokens: still
inside the warm-up window, so the code slips by. In a 32K document the same "10% in"
is ~3,200 tokens — further along, more often *past* the window. **Longer documents
push the hidden items out of the blind spot.** The document didn't get easier; the
items moved.

How sure are we? This one's actually well-nailed, because it made predictions that
came true:

- **The misses are exactly the earliest-planted items** — the first one or two codes,
  every time, on both the small 1B model and the big 8B model. Later codes are fine.
- **A bigger sharp pocket doesn't help at all.** We swept the pocket from tiny to
  huge and the miss count didn't move. That proves the code was never *selected* —
  it's a blind spot, not a space problem.
- **Two side-predictions checked out.** A "one key, four values" test at 16K missed
  exactly the *first* value (the one inside the window) and got the other three. And
  the chain-following test scored 0 at 16K (the chain's *root* sits in the window —
  lose the root, lose everything) but 100 at 32K (root pushed outside).

**The honest consequence:** BUG+surprise is a **long-document (32K-and-up) method**.
For 16K documents, ExpectedAttention is simply the better recommendation, and we say
so plainly. (We'd earlier leaned toward "the 16K dip is probably test noise" — that
lean was wrong, and the rerun with more samples confirmed the dip is real.)

### The balanced setting: a sharper photo

All season the story has been "BUG wins on memory, loses a bit on text quality." The
obvious knob: make the blurry photo **sharper** (rank 128 instead of 32). That costs
memory — about 0.16× instead of 0.043× — but here's what it buys at 32K:

- **Better text quality than ExpectedAttention** (perplexity 8.12 vs 8.28; same story
  at 16K, 4.16 vs 4.29) — the first time this season BUG beats EA on that axis.
- **Retrieval mostly holds**: beats EA on multi-key (75% vs 67%), ties on the needle
  and multi-value, narrowly loses chain-following (75% vs 83%).

So the sharper-photo setting beats EA on text quality *and* multi-key at once — but
it pays with memory (~1.6× EA's budget). A trade, not a free win. And it's still a
32K-and-up method: at 16K it stays far below EA — for the blurrier version we can
blame the warm-up window; for the sharper one the honest answer is we don't fully
know yet (its mechanism is the open question below).

**But the dial is not "sharper is better" — we tried the next notch, and there's a
cliff.** If rank 128 beats EA on text quality, why not rank 256? We ran it. Text
quality got even better (7.74 at 32K — the best of any BUG config, though heavier
methods like MorphKV still score lower at ~0.3×+ memory — at ~0.3×
memory) — but retrieval collapsed to **zero**. Every hard test — multi-key,
multi-value, chain-following — scored 0% at *both* lengths and *both* pocket sizes,
twelve cells of nothing; even the partial credit rank 128 still earned at 16K
vanished. A better photo alone retrieves nothing. So the rank ladder reads: 32 too
blurry to hold the codes, 256 apparently too well-fitted to ever surface them, and
128 a **narrow sweet spot that works for reasons we honestly cannot yet name**.

**One honest mystery remains.** With the sharper photo, we peeked inside the sharp
pocket — and the code we ask for is *never in it* (at most 1–3 stray codes appear, and only
at the larger pocket sizes). The sharper summary predicts the codes
well enough that they no longer look surprising, so they never get picked. Yet the
method still retrieves them — while plain BUG at the same sharpness scores 0%. So
the win isn't the pocket holding the codes, and it isn't the plain photo either. Our
best guess: keeping the weirdest tokens *out* of the photo keeps the photo cleaner
for everything else. We haven't proven that. The rank-256 cliff above makes the
mystery sharper, not simpler: it can't be "the photo just got good enough to answer
on its own," because an even better photo answers *nothing*. Whatever rank 128 is
doing, the pocket is somehow involved — even though it isn't holding the codes.
Next session's test: switch the pocket **off** entirely at this sharpness and see
if the retrieval survives — either answer closes in on the mechanism.

### Where this leaves the recommendations

- **Tightest memory, long documents (32K+):** BUG+surprise, blurry setting — all
  four retrieval tests covered at 0.043×, the only method under 0.1× that does it.
- **Balanced quality + retrieval, long documents:** the sharper rank-128 setting at
  ~0.16× — beats EA on text quality and multi-key. And stop there: rank 256 is past
  the cliff (better text, zero retrieval).
- **16K or shorter:** use ExpectedAttention. No hedging.

And one tidy prediction we get to test next time: at **64K**, *every* hidden item
should sit past the warm-up window, so retrieval should get even better. If it
doesn't, the warm-up story is in trouble — which is exactly what a good explanation
should risk.
