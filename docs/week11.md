# Week 11 — making BUG *retrieve* at 32K (SurpriseSLASH), and the honest attribution

> **Frame.** Week 10 left one clean wall: at 8B / 32K, plain BUG **retrieves a
> needle 0% of the time at every rank** (r32–r256, 0.033×–0.26× memory), while
> ExpectedAttention (`ea-k0.1`) gets **100% at 0.10×**. This is the
> *rank-vs-context fidelity wall* — a needle is a low-attention, high-residual
> outlier, which is exactly what a rank-`r` low-rank summary reproduces worst.
> Week 11 asks the falsifiable question: **can BUG retrieve the 32K needle at ≤ the
> memory ExpectedAttention needs — or prove it can't?** The answer is a genuine
> retrieval-frontier win, delivered by BUG's *surprise signal used to select an exact
> tier* on top of its low-rank gist. An initial single-needle-plus-perplexity read
> suggested the gist was dead weight (a rank-1 `bugEVICT` control matched it); a later
> **full 4-task RULER run overturns that attribution** — the low-rank gist *helps* on
> the hard multi-fact tasks (multi-key / multi-value / variable-tracking), where the
> gist-free control collapses to 0% *at a tight exact-tier budget*. A lean, not a slam
> dunk: small n (2–6/cell), and the edge is budget- and context-dependent (see the
> caveats). Read with `docs/week9.md` (D1 recovery / D3 surprise,
> the closest prior art), `docs/week10-handover.md` (the 7-method map and this
> wall), and `docs/week7-dominance.md` (the two measured walls).

Model: `unsloth/Llama-3.1-8B-Instruct`, GPU bf16 (RTX 3090 via vast.ai), `n =
head_dim · num_kv_heads = 128 · 8 = 1024` (1 token = `2n = 2048` floats/layer).
Every arm's memory is counted honestly in the same float-equivalent unit
(`stored_state_numel`), reported as `ratio` vs the full cache, and audited
`mem_max ≤ budget`. Suite: **263 pass / 1 skip**.

## Status table

| Goal | Question | Axis judged | Verdict |
|---|---|---|---|
| **B** | can BUG retrieve the 32K needle at ≤ EA's memory? | retrieval-vs-memory @ 32K/8B | **WIN — surprise-selected exact tier retrieves 100% at 0.009× (≈11× cheaper than EA's 0.10×) where plain BUG was 0%** |
| **B (attribution)** | is BUG's low-rank *gist* dead weight, or does it carry the win? | full 4-task RULER + ppl @ 32K/8B, `bugEVICT` control | **GIST HELPS (a lean) — single-needle + ppl alone flatter the gist-free `bugEVICT`; on the full suite `bugEVICT` collapses to 0% on multi-key/multi-value/var-track *at a tight tier (h256)* while `bugS` handles all four. Budget/context-dependent, small n; keep `bugS`, drop `bugEVICT`** |
| **A** | finish the broken Week-10 retrieval eval | 4-task RULER @ 16K + LongBench | **DONE — 3 harness bugs fixed (each with a regression test); complete matrix banked** |

---

## GOAL B — the mechanism: SurpriseSLASH (+ span expansion), and the `bugEVICT` control

**The idea.** BUG already carries an *exact heavy-hitter tier* (SLASH `hh_budget`):
a small set of tokens kept **verbatim** post-RoPE, bypassing the low-rank
absorption entirely — like sinks. In Week 7 that tier selected by
**attention mass** (keep the heavy hitters). But a needle is precisely the *opposite*
of a heavy hitter: it is a low-attention, high-**residual** outlier — the token the
rank-`r` basis reconstructs worst. So we made the exact tier select by
**low-rank surprise** instead:

> surprise(k) = ‖k − U Uᵀk‖ / ‖k‖  — the out-of-subspace residual fraction, the
> `sin` of the angle between the key and the current basis `span(U)`, scale-free
> across columns.

This is the *same* residual signal probed in Week-9 D3 (`retention="lowrank_surprise"`),
now repurposed from a low-rank **eviction/retention score** into an **exact-tier
selection rule**. A persistently un-representable outlier (a sharp needle) keeps
residual ≈ 1 and is never demoted; a token later captured by the grown basis is
demoted into the tail. Implementation (`src/kvdlra/cache/bug_cache.py`):

- New knob **`hh_select="surprise"`** on `BugStreamingCache`
  (`_absorb_block_slash` surprise branch, `_surprise_scores`). It is **attach-free**
  (no attention hook needed), **accounting-neutral** (stores *no* per-column score —
  it recomputes the whole candidate pool's residual against the current basis each
  absorb), and keeps the hard `hh_budget` cap. Crucially, single-shot prefill
  bypasses SLASH, so a SurpriseSLASH arm **must chunked-ingest** to exercise the
  tier — this is a real harness constraint, not a free switch.
- **Span expansion, `hh_neighbor`** (`_span_boost`). A Phase-0 probe found the
  5-digit code tokenizes to ~3 sub-tokens, and one sub-token is *not* itself an
  out-of-subspace outlier — so pure surprise selection can keep 2/3 and miss one.
  `hh_neighbor=w` applies a SnapKV-style position-windowed max: each token inherits
  the highest surprise within `±w` positions, so a non-outlier token adjacent to a
  sharp needle is pulled into the top-`hh_budget`, keeping the whole contiguous
  needle span verbatim. It **re-ranks within the cap; it never grows the tier** (so
  the memory accounting is untouched).

**The attribution control — `bugEVICT`.** To ask whether it is BUG's *low-rank
compression* or merely the *surprise selection rule* doing the retrieving, we run a
degenerate arm: **rank-1 BUG** (a negligible gist) with the same surprise-selected
verbatim tier. `bugEVICT` is therefore ≈ **pure surprise-selected verbatim
eviction** — it keeps the same exact tokens the gist arm keeps, but throws away
essentially all of the low-rank summary. If `bugEVICT` matches the real gist arm
**across the full task suite**, the gist is dead weight; if it matches on some tasks
but collapses on others, the gist is carrying those others. (Spoiler: it's the
latter — single-needle + ppl tie, but the hard multi-fact tasks separate them.) Two
eval arms carry the comparison:

- **`bugslash`** (`bugS-r32-h{hh}`): rank-32 low-rank gist **+** surprise exact tier.
- **`bugEVICT`** (`bugEVICT-h{hh}`): rank-1 degenerate gist **+** surprise exact tier
  = the attribution control.

Tests/scripts: `tests/test_bug_cache_week11.py`, `scripts/w11_probe.py`,
`scripts/pod/w11_gpu.sh`. Data lines on disk:
`results/w11-goalB-{probe,ruler,ppl}-lines.txt`.

## GOAL B — Phase-0 probe: does surprise catch the needle?

Before spending GPU budget on the frontier, a cheap probe checks that the surprise
signal actually places the needle sub-tokens in the exact tier
(`results/w11-goalB-probe-lines.txt`):

- **32K, hh=64 (0.002× memory), neighbor=1: needle captured 3/3 VERBATIM.** The
  needle survives even at a *tiny* exact tier.
- **16K, hh=64 (0.004×), neighbor=1: 3/3.** Same at half the context.
- **Span expansion validated.** At 32K / hh=512, `neighbor=0` captured only **2/3**
  (missed a sub-token that isn't itself an outlier); `neighbor=1` recovered **3/3**.
  This is the exact failure the probe was built to catch, and the fix works.

## GOAL B — retrieval frontier, `niah_single` @ 32K / 8B

Accuracy and honest memory ratio (`results/w11-goalB-ruler-lines.txt`):

| arm | acc | memory ratio |
|---|---|---|
| full | 1.00 | 1.000× |
| **ea-k0.1** (the bar) | **1.00** | **0.100×** |
| plain BUG (any rank, Week-10) | **0.00** | 0.033×–0.26× |
| bugS-r32-h256 | 1.00 | 0.043× |
| bugS-r32-h512 | 1.00 | 0.051× |
| bugS-r32-h1024 | 1.00 | 0.066× |
| bugS-r32-h2048 | 1.00 | 0.096× |
| **bugEVICT-h256** | **1.00** | **0.009×** |
| bugEVICT-h512 | 1.00 | 0.017× |
| bugEVICT-h1024 | 1.00 | 0.033× |
| bugEVICT-h2048 | 1.00 | 0.064× |

**The fidelity wall is beaten, not tied.** Where plain BUG retrieves 0% at every
rank, a surprise-selected exact tier retrieves the 32K needle **100% at 0.009×** —
about **11× cheaper** than ExpectedAttention's 0.10×. This is the headline of GOAL B.

**`niah_multikey` @ 32K.** The harder multi-key task degrades, but the gist arm holds
up best: `full` 1.00, `bugS-r32-h256` **0.83** — *above* ExpectedAttention's 0.50 at
<half its memory (see the full 4-task decision table below). This is the first sign
that the gist is *not* dead weight: the gist-free `bugEVICT` control drops to **0**
here, while the rank-32 gist keeps the arm above EA.

## GOAL B — joint quality (perplexity @ 32K / 8B): THE attribution

Retrieval alone can be gamed by a degenerate cache that only keeps a few verbatim
tokens, so we measure **streaming perplexity** at 32K as the joint-quality axis —
does the arm still *model text* well, not just fish out the needle?
(`results/w11-goalB-ppl-lines.txt`):

| arm | ppl | memory ratio |
|---|---|---|
| full | 7.624 | 1.000× |
| ea-k0.1 | 8.277 | 0.115× |
| **bugEVICT-h256** | **8.951** | **0.009×** |
| **bugEVICT-h1024** | **8.812** | **0.032×** |
| bugS-r32-h256 | 9.164 | 0.043× |
| bugS-r32-h1024 | 8.881 | 0.065× |

**On perplexity + single needle the control ties — but that view is too narrow.**
`bugEVICT` (rank-1, *no* gist) matches or slightly beats `bugslash` (rank-32 gist) on
perplexity: 8.951 vs 9.164 at hh=256, 8.812 vs 8.881 at hh=1024 (within ~2%), at ~5×
less memory (0.009× vs 0.043× at hh=256). On the *single-needle* retrieval task the
two are also tied (both 100%). Taken alone, these two axes suggest the gist is dead
weight — and that **was** the initial Week-11 conclusion. **The full 4-task RULER run
below overturns it:** single-needle + perplexity flatter the cheap gist-free control,
because neither task exercises what the gist is *for* — reconstructing the many soft
facts a hard multi-fact query needs. On perplexity the gist still adds little (the
Week-7/8 overhead-floor wall for *text modelling* stands); its payoff shows up only on
multi-fact retrieval.

## GOAL B — the full 4-task decision table (@ 32K / 8B): the attribution, corrected

Single-needle + perplexity are only two axes. Running the **full four RULER tasks**
(single needle, multi-key, multi-value, variable-tracking) at 32K / 8B is what
actually settles the attribution (`results/w11-goalB-ruler-lines.txt`), accuracy in %:

| arm | mem | needle | multi-key | multi-value | var-track | ppl |
|---|---|---|---|---|---|---|
| full | 1.0× | 100 | 100 | 100 | 100 | 7.62 |
| ea-k0.1 | 0.10× | 100 | 50 | 100 | 100 | 8.28 |
| **bugS-r32-h256** | **0.043×** | **100** | **83** | **100** | **100** | 9.16 |
| bugS-r32-h1024 | 0.066× | 100 | 50 | 100 | 100 | 8.88 |
| bugEVICT-h256 | 0.009× | 100 | **0** | **0** | **0** | 8.95 |
| bugEVICT-h1024 | 0.033× | 100 | 50 | **0** | 100 | 8.81 |
| plain BUG r32 | 0.03× | 0 | 0 | 0 | 0 | 9.31 |

**The gist helps on hard retrieval — not dead weight, but a lean.** The single-needle
-only view flattered `bugEVICT`: with essentially no gist, at a *tight* tier (h256) it
**collapses to 0** on multi-key, multi-value, and variable-tracking, and is still 0 on
multi-value at h1024. The rank-32 gist arm `bugS` **handles all four tasks** at 0.043×
— and **beats ExpectedAttention on multi-key, 83 vs 50, at <half EA's memory.** So the
low-rank summary does real work on the hard multi-fact retrieval the single needle
never probed. *But hold the strength honestly:* the gap is starkest at the tight h256
budget (at h1024 `bugEVICT` partly catches up — var-track 100, multi-key 50); at 16K
*both* arms are weak on the hard tasks; and n is 2–6/cell on all-or-nothing metrics. So
it is a **lean toward `bugS`**, worth a confirming run. Plain BUG (surprise tier off)
is **0% on all four** at 32K — the wall SurpriseSLASH was built to break.

## GOAL B — the honest decision (stated self-critically)

1. **WIN on retrieval-vs-memory.** A surprise-selected exact tier retrieves the 32K
   *single* needle at **0.009×** — ≈11× cheaper than ExpectedAttention's 0.10× —
   where plain BUG was 0% at every rank. The Week-10 rank-vs-context fidelity wall is
   **beaten, not tied.**
2. **The low-rank gist HELPS on hard retrieval (a lean) — the earlier "dead weight"
   attribution was a single-needle trap.** On single-needle + perplexity, the
   gist-free `bugEVICT` control ties the gist arm, which initially looked like the
   gist adding nothing. But on the **full 4-task suite**, `bugEVICT` collapses to 0 on
   multi-key/multi-value/variable-tracking *at a tight tier (h256)*, while the rank-32
   `bugS` handles all four and beats EA on multi-key. So the gist earns its `2nr` floats
   on multi-fact retrieval — **but as a lean, not a proof:** the edge is
   budget-dependent (h1024 narrows it), context-dependent (16K both weak), and n is
   2–6/cell. **On *perplexity* the gist still adds little** (~2% at n=2), so the
   Weeks-7/8 overhead-floor wall for *text modelling* is unchanged — the gist is
   ~neutral on ppl/single-needle, and *leans* useful on multi-fact retrieval.
3. **Recommendation: keep `bugS` (SurpriseSLASH); drop `bugEVICT`; retire plain BUG.**
   `bugEVICT` is a single-needle trap (cheap, but 0% on the hard tasks). Plain BUG is
   0% on all four at 32K. `bugS-r32-h256` is the arm to carry forward: all four tasks
   at 0.043× (~2.3× cheaper than EA), winning multi-key outright.
4. **vs ExpectedAttention: a memory-and-retrieval win with a perplexity trade, not a
   clean sweep.** `bugS` matches or beats EA on every RULER task (ties on
   needle/multi-value/var-track, wins multi-key 83 vs 50) at **<half the memory**
   (0.043× vs 0.10×); but EA keeps **better perplexity** (8.28 vs 9.16). Report both
   directions — the win is memory + hard-retrieval, the cost is ppl.

So Week 11 lands like Week 9's D1: **a genuine retrieval-frontier positive via BUG's
surprise-selected exact tier, and — once the full suite is run — a low-rank gist that
*earns its keep* on hard multi-fact retrieval** (even as it stays ~neutral on
perplexity, consistent with the overhead-floor finding for text modelling). A
publishable positive; the honest caveat is the perplexity trade and small `n`, not a
dead-weight gist.

---

## GOAL A — finishing Week 10 (harness repair + banked matrix)

Week 10's retrieval eval had been **silently broken** for several methods. Three
harness bugs were fixed (each with its own regression test), plus a device fix
already in the pod clone:

- **(a) MorphKV** — `_window_attention_rows` crashed when the final ingest chunk was
  shorter than `recent_window` (tensor-size mismatch) → bound the window to the
  `hidden_states` length.
- **(b) SnapKV** — crashed because ChunkPress fed a 16-token RULER query, shorter
  than its 64-token scoring window → run the scorer press single-shot in RULER.
- **(c) LongBench** — a single dataset-load failure aborted the whole axis →
  per-task `try/except` so one bad task no longer sinks the run.
- **(d) ShadowKV** — GPU device fix (commit `0bf3b7b`) now present in the clone.

**Result: the complete 4-task RULER @ 16K / 8B now runs for all methods**, including
the previously-absent MorphKV / SnapKV / ShadowKV. Accuracy
(`niah_single / multikey / multivalue / vt`; `results/w11-goalA-ruler-lines.txt`):

| method (mem) | single | multikey | multivalue | vt |
|---|---|---|---|---|
| full (1.00×) | 1.00 | 1.00 | 1.00 | 1.00 |
| ea-k0.1 (0.10×) | 1.00 | 0.88 | 1.00 | 0.12 |
| snapkv-k0.1 (0.10×) | 0.88 | 0.25 | 0.75 | 0.00 |
| morph-k0.1 (0.12×) | 0.00 | 0.00 | 0.00 | 0.00 |
| morph-k0.25 (0.31×) | 0.75 | 0.50 | 0.88 | 0.12 |
| plain bug-r128 (0.14×) | 0.25 | 0.38 | 0.38 | 0.00 |
| think-c0.3 (0.85×) | 1.00 | 1.00 | 1.00 | 1.00 |
| palu-r0.5 (0.50×) | 1.00 | 1.00 | 1.00 | 0.00 |
| shadow-r64 (0.81×) | 0.00 | 0.00 | 0.00 | 0.00 |

**Findings (consistent with Week 10).**
- **ExpectedAttention is strongest at low memory** — but even it collapses on
  variable-tracking (`vt` 0.12): under compression, `vt` is hard for *everyone*
  (only `full` and `think-c0.3` at 0.85× get it).
- **Plain BUG is weak throughout** (the wall — this is exactly what motivated GOAL
  B): 0.25/0.38/0.38/0 at r128.
- **ShadowKV fails on this float-equivalent axis** (0/0/0/0). Its niche is GPU
  *throughput*, deliberately not rewarded by an honest memory-only metric.
- **`vt` is the universal hard task under compression** — full-cache-only among the
  aggressive arms.

**LongBench `qasper`** (query-in-prompt QA, ~5.6K ctx, F1;
`results/w11-goalA-lb-lines.txt`): full 0.259, `think-c0.3` 0.267, `morph-k0.5`
0.267, `ea-k0.1` 0.149, `snapkv-k0.1` 0.136, `bug-r32` **0.076** (BUG weak at low
rank on in-prompt QA; eviction/ThinK competitive) — consistent with Week 10.

---

## Honest caveats (do not overstate)

- **Small `n`.** RULER cells are 2–6 trials each and the ppl attribution is `n=2`
  (gist-vs-evict ppl margins ~2%). The 4-task pattern is clear — the gist-vs-no-gist
  gaps on the hard tasks are 0-vs-100, not 2% — but exact per-task percentages carry
  trial-count noise; larger-`n` confirmation is a follow-up.
- **Full 4-task suite now run @ 32K.** single / multi-key / multi-value /
  variable-tracking are all measured for `bugS`, `bugEVICT`, EA, and plain BUG
  (decision table above). Variable-tracking stays hard for every compressed arm except
  at light compression.
- **8B only for GOAL B.** The 32K frontier is 8B; a **1B row** (`n=512`, where the
  Week-9 rank/n ratio was more favourable) remains to be run for scale-invariance.
- **`bugEVICT` is dropped, not headlined.** It was the attribution control; the full
  suite shows it is a single-needle trap (0% on the hard tasks). The carried-forward
  arm is `bugS`, gist included.
- **The trade vs EA is a trade, not a sweep.** `bugS` wins memory and hard retrieval
  (multi-key 83 vs 50 at <half the memory); EA keeps better perplexity. Both
  directions reported.

## Standing / follow-ups

- **Cost.** The 8B GPU run cost **≈$17 of vast.ai credit** (**≈$9 left**); pods
  destroyed after the run.
- **Open honest follow-ups:** `niah_multikey` / `niah_multivalue` @ 32K (complete
  the frontier), LongBench beyond `qasper`, and the **1B row** for GOAL B. All are
  extensions of a result that already stands, not fixes to it.

**Overarching read.** Week 11 breaks the 32K retrieval wall plain BUG failed at 0%.
BUG's *surprise signal*, used to select an exact tier, drives the single-needle
frontier win (~11× less memory than EA at the extreme). And — the correction that
matters — once the **full 4-task suite** is run, BUG's *low-rank gist* looks **not**
dead weight after all: it *helps* on the hard multi-fact tasks, where at a tight tier
the gist-free control collapses to 0% while the gist arm `bugS` handles all four and
beats EA on multi-key. The earlier "gist is dead weight" read came from single-needle +
perplexity alone, which don't exercise what the gist is for. Precise standing: the gist
stays ~neutral on perplexity (the Weeks-7/8 overhead-floor wall for *text modelling*
holds), and *leans* useful on multi-fact retrieval — a lean, not a proof (the edge is
budget/context-dependent and n is 2–6/cell; a confirming run would firm it). A real
positive — surprise-selection *and* the gist — with an honest perplexity trade vs EA.
