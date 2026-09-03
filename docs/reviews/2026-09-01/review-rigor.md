# Experimental-rigor & statistics review — kvdlra NeurIPS-readiness panel

Reviewer: experimental design / statistics. Repo `/Users/hari/Desktop/kv-dlra` @ branch `week7`.
Score: **5/10 (borderline reject)** — internal statistical hygiene is unusually good (pre-registered
thresholds, Wilson CIs, paired-window SEs, honest downgrades), but the evidence base under C1/C4 is a
custom synthetic task family whose construction structurally favors the method, evaluated at small n,
with the strongest cheap baseline absent from the flagship grid and zero realistic-task numbers on the
flagship config. A NeurIPS reviewer's one-line kill: *"every retrieval number in the paper comes from a
needle generator you wrote yourself, with 10 cycled filler sentences, and your one LongBench run shows
the extreme-compression story collapsing on real text."*

---

## Credit where due (things most ML papers do NOT have)

1. **Pre-registered significance machinery.** `docs/week15-significance.md:1-4` pre-registers ppl tie
   thresholds (|Δbits/tok| ≤ max(0.05, 2×SE), n=8 paired windows, `:66-77`) and RULER bars (n=8 to
   separate 7/8 vs 1/8, `:92-97`) *before* the GPU spend. `scripts/pod/w16.sh:135-151` (MODE=w17) is a
   written confirm protocol (commit 7f49091 "pre-registered Week-17 cross-model confirm").
2. **Wilson 95% intervals on every cell** — `scripts/w15_intervals.py:55-64` (correct formula, z=1.96),
   `results/w15-ruler-intervals.json` (228 cells) + w16 (79) + w17 (74).
3. **Honest downgrades on new data**: the Week-16 marquee claim was *narrowed* at n=16 ("beats ThinK
   Wilson-separated; leads Palu not separated", `docs/week17-explained.md:103-107`); Llama vt corrected
   0.75→0.58 (`:36-38`); invalidated baseline rows are flagged in-table, not deleted
   (`docs/week11-decision-table.md:32-33,59-60`).
4. **Paired design exists implicitly**: all arms score the same frozen ppl windows
   (`scripts/w10_frontier.py:474-481`) and the same RULER layouts (same `seeds × trials`,
   `scripts/w10_ruler.py:319-331`).

These earn the 5 instead of a 3. Now the attack.

---

## (a) The retrieval tasks are custom synthetics — how much does that undermine C1?

**What the code actually does.** `scripts/w10_ruler.py:139-193` (`build_task`) builds all four tasks
from scratch. The haystack filler is **10 bland sentences cycled verbatim** (`_filler_to`,
`w10_ruler.py:70-76`, importing `_FILLER` from `scripts/w4_needle.py:46-57` — "The weather in the
valley was mild…", ×10, repeated to 16K/32K tokens). Needle = "The secret passcode is {5-digit}."
inserted at `n//2 + (trial % 5)` (`w10_ruler.py:150`) — i.e. **always within ~5 sentences of the
exact middle**; no depth sweep. Scoring = substring match of the code(s) in ≤12/40 greedy tokens
(`:292-293,310`). This is not official RULER (arXiv:2402.13718); it borrows the task *names*.

**Why this is worse than "merely nonstandard" for THIS method.** BUG's two mechanisms are (1) a
low-rank gist and (2) an exact tier selected by out-of-subspace surprise `||k − UUᵀk||/||k||`
(briefing; `src/kvdlra/cache/bug_cache.py`). A 10-sentence cyclic filler produces a nearly **periodic
key stream of tiny intrinsic rank** — the pathological best case for both mechanisms simultaneously:
rank-16 gists can fit the filler almost exactly (which is precisely what `w16-ruler-intervals.md:11-13`
shows — r16 at 0.04× already retrieves 4/4), and the single injected needle is maximally
out-of-subspace, so surprise selection is close to trivially perfect. The claim "full retrieval at
0.075–0.149×" (C1) is therefore partially a property of the *filler*, not only of the method. The repo
itself concedes the mirror-image bias for ppl ("structurally favours BUG's global summary",
`w10_ruler.py:3-5`) but never audits the retrieval-side bias.

**In-repo evidence the concern is real.** The only realistic-text retrieval numbers in the repo
(Week-11 LongBench, Llama-3.1-8B per `scripts/pod/w11_gpu.sh:22`, `results/w11-goalA-lb-lines.txt`)
show the extreme-compression story NOT transferring: qasper F1 `bug-r32 = 0.076` and `bug-r64 = 0.099`
vs `full = 0.259`, `snapkv-k0.5 = 0.253`, `think-c0.5 = 0.264`; even `bug-r128 (0.16×) = 0.221` loses
to eviction at matched-or-less memory (`ea-k0.25 = 0.216` at 0.25×). Pre-seed configs, yes — but that
is the point: nobody has shown the flagship on anything except the in-house generator. Similarly, ppl
on real text at the flagship rank is clearly worse than baselines (Qwen 16K: bugSseed-r64 8.18 vs full
6.20/think 6.41/palu 6.36, `results/w17-decision-table.json:12,31,41,44`), so the *only* axis where BUG
wins is the axis measured by the custom generator.

**Reviewer phrasing.** "All retrieval claims (C1, C4) are measured on a bespoke needle generator with
degenerate near-periodic filler that minimizes gist rank and maximizes needle surprise — the two
quantities the method is built on. No official RULER/NIAH/∞Bench number appears anywhere. I cannot
distinguish 'BUG retrieves at 0.08×' from 'BUG retrieves at 0.08× on rank-deficient haystacks'."

**Cheapest defusing experiment (the single highest-value $ in this review).** One-line change to
`_filler_to`: draw filler sentences from wikitext-103/PG19 (both already wired into
`perplexity_sweep.load_corpus_ids`, `scripts/perplexity_sweep.py:83-96`) instead of `_FILLER`. Rerun
flagship + think/palu (+EA, see (h)) on Llama+Qwen 16K, n=12: CPU probe at 2K first ($0), then one pod
(~$10–15). If 12/12 holds on high-rank filler, C1 survives and the paper gets a robustness table; if it
drops, better to know now. Second cheapest: a depth sweep {0.1…0.9} for `niah_single` — the harness
already has a depth-parameterized builder (`w4_needle.build_haystack`, `w4_needle.py:66-79`); ~$10.
Gold-standard fix: run the *official* RULER pipeline on the flagship (weekend + ~$50 GPU).

---

## (b) Task breadth: 3 synthetic needle tasks + wikitext ppl; LongBench never run on flagship

- The Week-16/17 cross-model grid runs exactly **three** tasks — `niah_single niah_multivalue vt`
  (`scripts/pod/w16.sh:145-151`); `niah_multikey` (n_keys=8 default, `w10_ruler.py:507`) was dropped
  from the flagship grid (every mk column in `results/w17-ruler-intervals.md` is "—") even though
  multikey was historically BUG's *best* task (Week-9 wins; `docs/week11-decision-table.md:84`). Dropping
  a task the method wins is conservative, but it also means the flagship generality claim rests on 2
  strong tasks + 1 honest failure (vt).
- `scripts/w10_longbench.py` exists, is wired to the same arms, was run once in Week-11 on 8B
  (see (a)), and **was never run on `bugSseed-r64-h256` or the marquee config** (briefing confirms:
  Tier-3 deferred). QA-style tasks (question in prompt) are explicitly acknowledged to be *fairer to
  eviction* (`w10_longbench.py:4-9`) — i.e. the protocol most favorable to the baselines is the one
  omitted.
- No summarization, no aggregation, no multi-hop beyond the 3-hop `vt` chain (n_hops=3,
  `w10_ruler.py:508`).

**Reviewer phrasing.** "Table 1 generalizes over models but not over tasks. Four needle variants from
one generator is one task family, and the harness you built for realistic tasks was never pointed at
the headline configuration."

**Cheapest defuse.** `w10_longbench.py` on flagship + think/palu, Llama+Qwen, 4 QA tasks ×
`--n-examples 25`, 16K budget: one pod, ~$15–25. Even a *loss* reported honestly ("BUG is a
retrieval-per-byte method, F1 degrades at 0.08×") is far stronger than absence — and the Week-11 data
suggests that is the likely outcome, so budget the framing accordingly.

---

## (c) n sizes and seeds

**Facts.** Seeds `{0,1}` × `n-trials` (`w10_ruler.py:502-503`); RNG = `seed*131 + trial`
(`:144`). 16K flagship: n-trials 6 × 2 seeds = **n=12/cell** (`w16.sh:147`). 32K core: n-trials 2 × 2 =
**n=4/cell** (`w16.sh:151`). Marquee: n=16 (`w16.sh:165-170`). 64K: the ONLY 64K rows in the program
are `ea-k0.1` and pre-seed `bugS-r32-h256` at **n=2–4** (`docs/week11-decision-table.md:62-67`); the
flagship has **zero** 64K measurements.

**Three problems beyond "n is small":**
1. **n=4 Wilson intervals are vacuous**: 4/4 → [0.51, 1.0] (`results/w17-ruler-intervals.md:20-22`).
   Every 32K cross-model cell in C1 ("Also 1.00 at 32K (n=4)") is consistent with true accuracy 55%.
   Worse, C1's 32K sentence is **factually contradicted** for Mistral: 32K mv = 0.75 (3/4) and vt = 0.75
   (`w17-ruler-intervals.md:40-41`) — the 1.00-at-32K claim holds only for Qwen and Llama. (The
   explainer scopes it correctly to Qwen, `week17-explained.md:30`; the paper must too.)
2. **Effective n < nominal n.** The filler text is deterministic and identical across all trials, seeds,
   arms, and models at a given ctx (`_filler_to` has no RNG). Randomization spans only the 5-digit codes
   and a ≤5-sentence placement jitter: `niah_single` position = `n//2 + (trial%5)` (`:150`), so the
   n=12 cell contains at most **5 distinct needle positions** of one fixed haystack. The Wilson CI is a
   valid CI over *code draws at ~5 mid-depth positions of one document*, not over any task
   distribution. A reviewer will (correctly) say the 12 trials are near-replicates.
3. **Selection-then-confirm shares the generator.** The flagship config (r64-h256, sweet spot) was
   selected on the Week-16 sweep and confirmed at higher n in Week-17 on *fresh trials of the same
   generator*. That guards against trial noise, not against overfitting the task family — (a) and (c)
   compound.

**Cheapest defuse.** (i) Raise 32K to n≥12 for flagship+baselines (3 models × 3 tasks ×
4 arms × 8 more trials — one pod each, ~$20 total at Week-17 rates). (ii) Randomize filler
composition per trial (shuffle sentence order + random offset: 2-line change) so trials are
independent draws; re-run 16K n=12 (~$10). (iii) Add flagship 64K rows or delete 64K from the paper's
scope (the briefing already lists 2K–64K; as measured, the honest range is 16K–32K).

---

## (d) Single-sequence, batch=1

All prefill/decode paths take `[1, T]` inputs (`retrieve`, `w10_ruler.py:242-296`; `_decode` one token
per forward `:219-237`). No batching, no multi-turn, no measured VRAM/latency — and the repo's own
Tier-4 measurement shows decode **workspace ≈ 0.98× full** because attention runs on reconstructed K/V
(`scripts/w16_storage.py`, briefing). Statistically this is a claim-scope issue, not a validity issue:
C1's "memory" is analytic float-equivalent storage (`ratio_fp16`, `w10_ruler.py:296`), which is fine
*as stated*, but a reviewer will ask why compression at batch=1 with no wall-clock or peak-VRAM win is
useful. **Defuse:** (i) report `ratio_fp16` alongside measured peak VRAM for one flagship/baseline pair
(hooks already exist: `acc.measure_peak_gpu`, `w10_frontier.py:487`) — $0 on the next scheduled run;
(ii) state the fused-kernel-future-work limitation in the main text, not an appendix.

## (e) Perplexity protocol

- Corpus: wikitext-103 only (`w10_frontier.py:706`), windows sliced **sequentially from the corpus
  start** (`:477-479` — `range(0, len-window, window)` then `[:n_samples]`), not sampled; so "ppl at
  32K" = the first 4–8 disjoint 33K-token spans of wikitext-103, scoring 512 tokens each
  (`w16.sh:44-47`).
- Published Week-17 ppl cells use **n-samples 4** ("leaner ppl", `w16.sh:46-47,199-203`) ≈ **2,044
  scored tokens**, while the pre-registered tie threshold was defined *at n=8 windows*
  (`week15-significance.md:66`). A tie/SE computed from 4 paired windows has df=3; the program
  quietly halved its own pre-registered sample size. The C4 "ties ThinK/Palu ppl (≤0.03 bits/tok)"
  numbers are Week-15 n=8 — fine — but any Week-17 ppl claim (e.g. floor results C3: 27531→6.995,
  138→5.574) is n=4. For C3 the effect sizes are 3–4 orders of magnitude, so n=4 is genuinely
  sufficient *there*; say so explicitly rather than hoping nobody checks.
- No second corpus. The Week-13 lesson that a CPU proxy over-predicted ppl effects ~30–40×
  (memory: Q-BUG) shows this program already knows single-metric ppl extrapolation is fragile.

**Defuse:** one PG19 ppl column at 16K/32K for flagship + full + think/palu (pg19 already a supported
`--corpus` choice, `w10_frontier.py:706`): ~$10. Restore n=8 for any cell a claim cites.

## (f) Model scale

1B–8B, three families at 7–8B. No ≥13B, no MoE, no long-context-native model (e.g. Qwen2.5-1M-class).
The rank-vs-retrieval wall onset is already **model-dependent** (C2: Qwen r128 collapse
`w16-ruler-intervals.md:14-15` — 0/4, 0/8 — vs Llama r128 fine), so scale-dependence of the sweet spot
is a live risk, not a hypothetical. **Defuse:** one config (flagship + think/palu, 16K, n=8, single+mv)
on a 32B-class model rented for a day (~$30–50). Not blocking for an honest "7–8B scale" scoping
sentence, but the claim "generalizes" should then read "generalizes across 7–8B families".

## (g) Wilson-per-cell vs multiple comparisons

381 uncorrected 95% intervals across the program (228 + 79 + 74; `w15_intervals.py` note "all 228
cells"). At 95%, ~19 cells are expected to mislabel. Mitigations in place: the headline comparisons
were pre-registered (w16.sh MODE=w17 comments; `week15-significance.md`), which legitimately shrinks
the family to a handful. My checks (Fisher exact, computed from the published counts):

| comparison | counts | two-sided p | verdict |
|---|---|---|---|
| Marquee vt: BUG 15/16 vs ThinK 5/16 (`w17-ruler-intervals.md:60,64`) | 15/16 vs 5/16 | **0.00064** | survives even Bonferroni×74 (p≈0.047); the one airtight separation |
| Marquee vt: BUG 15/16 vs Palu 9/16 (`:60,63`) | 15/16 vs 9/16 | 0.037 | nominal only; repo honestly claims "leads, not separated" — keep that wording |
| "Beats both on mv": Qwen 16K BUG 12/12 vs ThinK 10/12 (`week17-explained.md:28-29`) | 12/12 vs 10/12 | **0.48** | NOT significant — the word "beats" is unsupported |
| same vs Palu 11/12 | 12/12 vs 11/12 | 1.0 | NOT significant |
| Qwen 32K mv BUG 4/4 vs Palu 1/4 | 4/4 vs 1/4 | 0.14 | not significant |

Two concrete fixes, both $0: (1) delete/soften every "beats" that a Fisher test can't back
(`week17-explained.md:28-29` is the offender that must not reach the paper); (2) since all arms share
identical trial layouts (same `seed*131+trial` streams), report **paired McNemar** on discordant
trials for the pre-registered contrasts instead of unpaired Wilson-overlap — strictly more power at
zero GPU cost, and it turns "CIs overlap" hand-waving into a real test. State in the paper that
per-cell CIs are descriptive and only the pre-registered contrasts are confirmatory.

## (h) Hyperparameter fairness to baselines

**The asymmetry is real and is the second-biggest problem after (a).**
- On the cross-model grid, baselines run at exactly one config each — `--think-ratios 0.5
  --palu-ranks 0.5` (`w16.sh:70-71,146,151,169-170`) — while the BUG family was swept per-model over
  rank {16,32,64,128,(256)} × hh {256,512,1024} × {seed, s32, f0.01} (`w16.sh` sweep modes;
  `w16-ruler-intervals.md` lists 8 BUG arms vs 2 baseline arms), and the flagship/sweet-spot was
  chosen *after* seeing those sweeps.
- The baseline configs were swept only once, on **Llama** in Week-10/11 (`think 0.3/0.5/0.7`,
  `palu 0.25/0.5`, `w10_ruler.py:453-454`), then transplanted to Qwen/Mistral. And that Llama sweep
  shows the choice *matters against the marquee*: **think-c0.3 (0.852×) scores vt=100 at both 16K and
  32K** (`docs/week11-decision-table.md:30,57`, n=8/n=2) where the marquee's think-c0.5 comparator
  scores 0.31 (5/16). The Wilson-separated marquee win (C4) is a win over ThinK's *half-channel*
  operating point, with an in-repo config that plausibly erases it left unrun at n=16. Same story for
  morph-k0.5 (vt=100@32K, `:54`).
- **The strongest cheap baseline is missing from the flagship grid entirely.** ExpectedAttention at
  keep=0.1 costs **0.100×** — inside C1's claimed 0.075–0.149× band — and on Llama scored
  100/92/100/17 @16K and 100/67/100/83 @32K (`week11-decision-table.md:40`; the repo's own Week-11
  recommendation was literally "At 16K: EA", `:90-91`). EA never ran on Qwen/Mistral and never ran
  against the seed-fixed flagship. The claim "a regime eviction and channel-pruning cannot enter at
  all — you cannot evict your way to 0.05× and still answer a needle" (`week16-explained.md:69-70`) is
  refuted at 0.1× by the program's own EA rows (and its own `bugEVICT-h256` answers the single needle
  at **0.009–0.018×**, `week11-decision-table.md:49`). As written, C1's "5–13× less memory" is
  measured against the only two baselines *structurally unable* to compress below ~0.5×.

**Reviewer phrasing.** "You tuned your method per-model over ≥12 configs and froze both baselines at
one transplanted config each; the baseline family that reaches your memory regime (eviction: EA,
SnapKV at keep 0.05–0.1) is absent from every headline table; and your own Week-11 data contains a
ThinK config that scores 100 on the marquee task."

**Cheapest defuse (~$25 total, one pod per model):**
1. Add `ea-k0.1` (+ `snapkv-k0.1`) to the 16K n=12 and 32K grids on all three models — the arms
   already exist in `build_arms` (`w10_ruler.py` methods list `:509-520`). BUG's honest edge vs EA is
   vt and sub-0.05× memory; measure it instead of asserting "cannot enter".
2. Run `think-c0.3` and `palu-r0.25` at the marquee cell (Llama 32K vt/mv, n=16). If think-c0.3
   repeats its vt=100, the marquee claim becomes "beats ThinK at matched memory band" and must name
   the band; if it doesn't, the separation is armored.
3. Rewrite "cannot enter at all" → "channel-pruning (ThinK) and offline low-rank (Palu) are
   structurally floored at ~0.5–0.75×; eviction can reach 0.1× but loses vt there" — supportable today
   from `week11-decision-table.md:40` at $0, pending (1) for cross-model.

---

## Ranked findings

**Fatal (must fix before submission):**
- F1 (a): all retrieval evidence from an in-house generator with degenerate cyclic filler that
  structurally favors surprise-selection + low-rank gist; no official benchmark anywhere. Defuse:
  realistic-filler rerun + official RULER on flagship (~$25–50 + weekend).
- F2 (h): eviction baselines (EA/SnapKV @0.1×) absent from all flagship/cross-model tables while the
  paper claims a "regime eviction cannot enter"; own Week-11 data contradicts the phrasing. Defuse:
  ~$25 grid add + rewrite.

**Major:**
- M1 (b): LongBench harness never run on flagship; only realistic-task data in repo shows
  extreme-compression F1 collapse (qasper bug-r32 0.076 vs full 0.259). ~$15–25.
- M2 (c): 32K cells n=4 (Wilson [0.51,1.0]); C1's "1.00 at 32K" false for Mistral mv (3/4); 64K
  flagship absent. ~$20.
- M3 (h): marquee ThinK comparator at c0.5 only; in-repo think-c0.3 scored vt=100@32K (n=2). ~$8.
- M4 (c): effective-n inflation — deterministic shared filler, ≤5 needle positions per cell; Wilson CI
  quantifies code-draw noise only. 2-line randomization + rerun, ~$10.
- M5 (g): "beats both baselines on multi-value" (week17-explained.md:28-29) has Fisher p=0.48/1.0 —
  must be deleted. $0.

**Minor:**
- m1 (e): Week-17 ppl at n=4 windows vs pre-registered n=8; sequential-from-start windows; single
  corpus. ~$10 + $0 wording.
- m2 (f): 1B–8B only; wall onset already model-dependent → scale-dependence risk. ~$30–50 or scope the
  claim.
- m3 (g): 381 uncorrected 95% CIs; adopt paired McNemar for pre-registered contrasts ($0).
- m4 (d): batch=1, no VRAM/latency; workspace ≈0.98× makes "memory" storage-only — keep the honest
  framing in the main text ($0).
- m5 (a): no depth sweep for `niah_single` (mid-depth ±5 sentences only); `w4_needle.build_haystack`
  already parameterizes depth (~$10).

**Bottom line.** The statistics ON the collected data are close to exemplary for this genre; the data
GENERATING process is the weak joint. Two pods (~$50) and one honest rewrite convert F2, M2, M3, M5
into non-issues within a week. F1 and M1 are the open scientific risk: if flagship retrieval survives
realistic filler and shows a defensible LongBench story, this dimension moves to a 7; if not, the
paper's C1 is a property of the benchmark, and no amount of Wilson machinery saves it.
