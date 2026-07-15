# Week 11 — making BUG *retrieve* at 32K (SurpriseSLASH), and the honest attribution

> **Frame.** Week 10 left one clean wall: at 8B / 32K, plain BUG **retrieves a
> needle 0% of the time at every rank** (r32–r256, 0.033×–0.26× memory), while
> ExpectedAttention (`ea-k0.1`) gets **100% at 0.10×**. This is the
> *rank-vs-context fidelity wall* — a needle is a low-attention, high-residual
> outlier, which is exactly what a rank-`r` low-rank summary reproduces worst.
> Week 11 asks the falsifiable question: **can BUG retrieve the 32K needle at ≤ the
> memory ExpectedAttention needs — or prove it can't?** The answer is a genuine
> retrieval-frontier win, delivered by BUG's *surprise signal used as an eviction
> rule*, together with a rigorous self-critical finding that BUG's low-rank gist
> itself is dead weight here. Read with `docs/week9.md` (D1 recovery / D3 surprise,
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
| **B (attribution)** | is it BUG's low-rank *gist* doing the work? | joint quality (ppl) + retrieval, `bugEVICT` control | **NO — the rank-1 `bugEVICT` control matches/beats the rank-32 gist arm on both; the gist is dead weight (extends the overhead-floor wall)** |
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
essentially all of the low-rank summary. If `bugEVICT` matches the real gist arm,
the gist is dead weight. Two eval arms carry the comparison:

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

**`niah_multikey` @ 32K (partial).** The harder multi-key task degrades: `full`
1.00, `bugS-r32-h256` and `bugS-r32-h512` both **0.83** (a needle among distractor
keys stresses selection). Honest caveat — this row is partial (single-needle is the
completed frontier; multikey/multivalue are follow-ups, below).

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

**The control wins.** `bugEVICT` (rank-1, *no* gist) matches or **beats** `bugslash`
(rank-32 gist) on **both** axes — retrieval-memory *and* perplexity: 8.951 vs 9.164
at hh=256, 8.812 vs 8.881 at hh=1024 (within ~2%). And it does so at **5× less
memory** (0.009× vs 0.043× at hh=256), because the rank-32 basis costs `2nr` floats
the rank-1 arm doesn't pay. So the honest reading is: **the low-rank gist does not
help and costs 5× the memory here — it is dead weight.** The retrieval win is the
*surprise selection rule* repurposed as eviction, **not** BUG's low-rank
compression.

## GOAL B — the honest decision (stated self-critically)

1. **WIN on retrieval-vs-memory.** A surprise-selected exact tier retrieves the 32K
   needle at **0.009×** — ≈11× cheaper than ExpectedAttention's 0.10× — where plain
   BUG was 0% at every rank. The Week-10 rank-vs-context fidelity wall is **beaten,
   not tied.**
2. **BUT it is the surprise SELECTION RULE, not BUG's low-rank compression.** The
   `bugEVICT` control (rank-1, no gist) matches/beats the rank-32 gist arm on both
   retrieval-memory *and* perplexity (8.95 vs 9.16 at h256; 8.81 vs 8.88 at h1024)
   at a fraction of the memory. So honestly: **the gist does not help and is dead
   weight** — this **extends the Week-7/8 overhead-floor wall**: BUG's `2nr` basis
   is dead weight *even when paired with a good exact tier*. Caveat the small
   margins (~2%) and small `n` (n=2 per ppl cell) — this is "no measurable benefit,"
   not "provably harmful."
3. **vs ExpectedAttention: an honest memory/quality TRADEOFF, not a clean sweep.**
   `bugEVICT` gets the same 100% retrieval **and** reasonable ppl (~8.8–9.0) at
   **3–12× less memory** than EA (0.009×–0.032× vs 0.115×); but EA keeps **better
   ppl** (8.28 vs ~8.8–9.0). Report both directions — neither dominates.

So Week 11 lands like Week 9's D1: **a genuine retrieval-frontier positive via BUG's
surprise signal (used as eviction), with a rigorous self-critical attribution that
the low-rank gist itself is dead weight here** — consistent with the whole project's
overhead-floor finding. A publishable positive-with-honest-caveat.

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

- **Small `n`.** The perplexity attribution is `n=2` per cell and the gist-vs-evict
  margins are ~2%. This supports "the gist gives no measurable benefit here," **not**
  "the gist is provably harmful." Larger-`n` confirmation is a follow-up.
- **Multikey partial.** The completed 32K frontier is `niah_single`;
  `niah_multikey` is partial (bugS 0.83) and `niah_multivalue` is not yet run at
  32K.
- **8B only for GOAL B.** The 32K frontier is 8B; a **1B row** (`n=512`, where the
  Week-9 rank/n ratio was more favourable) remains to be run for scale-invariance.
- **`bugEVICT` is by construction not "BUG."** Its win is the honest *result*, not a
  claim that BUG's compression retrieves — the whole point of the control.
- **The trade vs EA is a trade, not a sweep.** EA keeps better ppl; `bugEVICT` keeps
  the same retrieval at much less memory. Both directions reported.

## Standing / follow-ups

- **Cost.** The 8B GPU run cost **≈$17 of vast.ai credit** (**≈$9 left**); pods
  destroyed after the run.
- **Open honest follow-ups:** `niah_multikey` / `niah_multivalue` @ 32K (complete
  the frontier), LongBench beyond `qasper`, and the **1B row** for GOAL B. All are
  extensions of a result that already stands, not fixes to it.

**Overarching read.** Week 11 is the retrieval mirror of the whole project's
compression finding. BUG's *surprise signal* is genuinely valuable — as an eviction
rule it beats the 32K fidelity wall at ~11× less memory than ExpectedAttention. But
BUG's *low-rank gist*, the thing under study, is once again dead weight: the rank-1
control matches it on every axis at a fraction of the memory. A real positive
(surprise-as-eviction) plus a rigorous negative (the gist), reported straight — the
same shape as Week 9's D1, and the same overhead-floor wall as Weeks 7–8.
