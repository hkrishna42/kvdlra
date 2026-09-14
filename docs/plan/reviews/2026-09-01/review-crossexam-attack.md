# Cross-examination — attacking the acceptance case (steelmanned Reviewer 2)

Role: adversarial examiner. Target: not the paper's weaknesses (six reviews covered those) but the
panel's **strongest acceptance arguments** — the things the reviews credited as reasons this could
become a 6–7. Each is attacked with repo evidence. Repo `/Users/hari/Desktop/kv-dlra` @ `week7`.

**Verdict in one line:** under cross-examination the acceptance case collapses to "an interesting
idea with honest bookkeeping": the celebrated 16K generality table contains **zero Wilson-separated
flagship-vs-baseline differences in either direction** (ceiling-effect tasks, not matched quality),
the proposed "constant-state asymptotic" rescue is refuted by the repo's own accounting formula
(coords are linear in T; the honest asymptote equals 2-bit KIVI's ratio), `min_sv_frac` is the
published CKL truncation step switched back on and shipped default-off, and there is no measurable
axis — resident memory, latency, storage-vs-real-competitors, ppl, or discriminative retrieval —
on which the flagship beats the best available alternative. Score: **3/10** as it stands.

---

## A1. "The generality claim is Wilson-firm (12/12 × 3 families)" — it is a ceiling-effect artifact

The panel's S1 (significance review) and the claims review both credit C1's 16K matrix as
"statistically firm." Look at what the table actually discriminates
(`results/w17-ruler-intervals.md:11-14,30-34,51-55`):

| 16K cell | bugSseed-r64 (0.085–0.149×) | palu-r0.5 (0.504×) | think-c0.5 (0.750×) |
|---|---|---|---|
| Llama single / mv | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| Qwen single / mv | 1.00 / 1.00 | 1.00 / 0.92 | 1.00 / 0.83 |
| Mistral single / mv | 1.00 / 1.00 | 1.00 / 0.92 | 1.00 / 1.00 |
| Llama vt | **0.58** | 0.83 | 0.42 |
| Qwen vt | 1.00 | 1.00 | 1.00 |
| Mistral vt | **0.50** | 0.33 | **0.83** |

1. **Not one of these comparisons is Wilson-separated, in either direction.** Every single/mv cell
   overlaps ([0.76,1.0] vs [0.55,0.95] at worst); every vt cell overlaps (Mistral think 0.83
   [0.55,0.95] vs bug 0.50 [0.25,0.75]). "Matched retrieval at 5–13× less memory" is therefore
   equally readable as: *the tasks cannot detect any quality difference across an 8.8× memory
   range*. A benchmark on which 0.085× and 0.75× are indistinguishable at n=12 has no power to
   certify "matched" — it certifies insensitivity. This is exactly what the degenerate haystack
   predicts (10 sentences cycled ~130×, `scripts/w4_needle.py:46-57`, `scripts/w10_ruler.py:70-76`;
   r16 at 0.04× already retrieves 4/4, `results/w16-ruler-intervals.md:11`).
2. On the **only discriminative task** (vt), the flagship *loses on point estimate* to Palu on
   Llama (0.58 vs 0.83) and to ThinK on Mistral (0.50 vs 0.83). The generality table's honest
   summary: "matches saturated baselines on tasks with no discriminative power; trails on the task
   that has some."
3. **Effective n ≪ 12.** Trials share one byte-identical filler document; randomization = a 5-digit
   code + position `n//2 + (trial % 5)` (`scripts/w10_ruler.py:150`) → at most 5 distinct needle
   placements, mid-depth only. The Wilson interval quantifies code-draw determinism, not task
   generalization — and the flagship (r64, sweet spot) was *selected* on this generator (Week-16
   sweep) and *confirmed* on fresh draws of the same generator.
4. **"Generality" is a patchwork, not a config.** What generalizes is one cell type (single+mv,
   16K, cyclic filler). Everything else is model-specific: s32 rescues Llama only (0/8 with s32 on
   Qwen and Mistral, `w16-ruler-intervals.md:18,40`); the floor is needed on Qwen/Mistral, untested
   on Llama; vt is 1.00/0.58/0.50 across families with no fix (h512/h1024 refuted); Mistral 32K mv
   = 3/4; and the Qwen 32K cell that holds 1.00/1.00/1.00 retrieval has **ppl 35.08 vs full 7.76**
   — a 4.5× fluency collapse *in the flagship's own headline cell*
   (`results/w17-decision-table.json` qwen/32768). A method whose per-model recipe differs in rank,
   tier size, scoring rank, and floor — chosen post-hoc per family — has demonstrated tuning
   transfer, not method generality.
5. Multikey — the one task the program historically *won* (Week-9) and the one where eviction
   structurally fails — is absent from the entire Week-16/17 matrix (every mk cell "—";
   `scripts/w17_intervals.py:32-34` parses only single|multivalue|vt). The matrix keeps the tasks
   everyone saturates and drops the task that could have separated.

**Surviving core after attack:** "on a low-rank synthetic haystack, an r64 gist + 256-token
surprise tier does not lose the needle on three families." That is a sanity check, not a frontier.

## A2. "min_sv_frac is a genuine numerical-analysis contribution" (C3) — it is a bugfix for a self-inflicted failure, shipped off

The prior-work review calls it "the strongest purely original artifact"; the claims review calls C3
"the strongest claim of the five." Cross-examination:

1. The rank-adaptive BUG integrator **as published already contains the truncation tolerance** —
   the Frobenius-tail `theta` is in the repo's own implementation
   (`src/kvdlra/integrators/streaming_torch.py:40-41,67,191-193`). The Week-17 divergences arose in
   runs configured with that tolerance effectively off, padding to `rank_cap`
   (`docs/week17-explained.md:64-69`). `min_sv_frac` (`streaming_torch.py:194-200`) is a *relative*
   restatement of the published step. Claim C3 as headlined = "we re-enabled Ceruti–Kusch–Lubich's
   safety mechanism and it stopped diverging." That is an ablation footnote at any venue with a
   DLRA-literate reviewer.
2. **The default-off paradox.** C3 claims the floor "removes the whole high-rank divergence class"
   — yet it is default-off and *no headline config uses it*. Either the fix is load-bearing (then
   the method shipped without its stabilizer and the flagship numbers are one ill-conditioned
   stream away from ppl 27531), or it is not (then it is not a contribution). The paper cannot have
   both.
3. **The rescued cells are dominated.** The showcase (Qwen r256 floored: ppl 27531→6.995 at 0.517×)
   is strictly worse than palu-r0.5 (6.355 at 0.504×, `results/w17-decision-table.json`); Mistral
   r128 floored (5.574) likewise sits above baselines at similar memory. "Extends the safe rank"
   extends it into a region where BUG loses on both axes. Retrieval-neutrality of the floor is n=4.
4. The genuinely interesting kernel — the tier siphons rank-carrying tokens out of the gist,
   collapsing its numerical rank — is a *diagnosis*, worth half a page in a stability subsection
   with CKL's ϑ cited. As a headline claim it will be read as inflating an engineering fix.

## A3. "The memory frontier is real; the storage-axis reframe saves it" — no axis survives, and the asymptotic rescue is arithmetically false

The systems review already reduced the memory claim to storage-only (resident ≈1.06× full,
`docs/week16-handover.md:20`; only latency datum: BUG 224.9ms vs full 204.5ms per token,
`results/w5-decode-validate-1b.json`). The significance review then offered the acceptance-side
rescue: "BUG's state is O(r·n + hh·n), **independent of context length** … a claim no quantizer can
match asymptotically." Cross-examination kills the rescue with the repo's own formula:

1. **The state is linear in T.** `src/kvdlra/accounting.py:145`:
   `verbatim = … + 2 * rank * coord_count + …` — the coordinate matrix is r×T (K and V), growing
   one column per absorbed token (confirmed by the claims review's own footprint decomposition:
   coords ≈80% of the r64/16K footprint). The ratio falls from 16K→32K (0.085→0.075) only because
   the fixed basis/tier amortize; it **asymptotes at r/n**, not 0.
2. **The honest asymptote equals 2-bit quantization.** For Llama/Mistral (n=1024, r=64): r/n =
   0.0625 as billed at 16 bits — but the coords/basis are stored **fp32** (`bug_cache.py:575-577`)
   and billed at 16 (`accounting.py:76-85`), so the honest at-rest asymptote is **0.125× — exactly
   KIVI's 2-bit ratio** — at ppl 1.08–4.5× of full, where KIVI is near-lossless and was never run.
   For Qwen (n=512): 0.25× honest — *above* published 2-bit and 3-bit KV work. The "regime others
   cannot enter" does not exist at any T, on any billing, against the right competitor class.
3. **The no-axis table.** Best measured/known alternative per axis vs the flagship:
   ppl — loses everywhere (e.g. Qwen 16K 8.18 vs palu 6.36); resident memory — loses (≈1.06× vs
   ThinK 0.75×, eviction ≤0.1×); latency — loses (only datum, 10% slower); throughput — no data;
   storage — ~parity-to-losing vs 2-bit KV at honest bits, uncompared; discriminative retrieval at
   16K — trails on vt (A1); realistic-text retrieval — the repo's only datum is a collapse (qasper
   F1 bug-r64 0.099 vs full 0.259, `results/w11-goalA-lb-lines.txt`); eviction-inaccessible tasks
   (multikey) — dropped from the matrix. There is currently **no measured axis of victory**, which
   is the definition of a reject for a method paper, however honest the bookkeeping.

## A4. "The statistics are exemplary" — precision without validity, and the pre-registration was itself violated

Pre-registration, Wilson CIs, and honest downgrades earned every reviewer's respect. But: (i) the
machinery is pointed at a generator with ~5 effective items and ceiling-effect tasks — careful
measurement of the wrong quantity; (ii) the pre-registered ppl protocol (n=8 windows,
`docs/week15-significance.md:66`) was quietly halved to n=4 in Week-17 (`scripts/pod/w16.sh:46-47`)
for the very cells now cited; (iii) the sole Wilson-separated retrieval result at n≥12 in the whole
17-week program is the marquee vt vs **think-c0.5**, while the repo's own sweep contains
**think-c0.3 with vt=100 at both 16K and 32K** (`docs/week11-decision-table.md:30,57`) — unrun at
the marquee cell; and (iv) the marquee config (r128-h1024-s32) is not the flagship config
(r64-h256), whose Llama vt is 0.58 — **no single configuration delivers both headlines**, and
nothing in the stats machinery protects against per-cell config shopping. Exemplary hygiene on a
biased instrument does not average out to sound inference.

## A5. "Reproducibility 7/10, accept-level artifact" — it reproduces the bookkeeping, not the results

What regenerates deterministically is *table generation from stdout aggregates*. The results
themselves cannot be reproduced by anyone, authors included: per-trial records and raw JSONs died
with the pods (w13–17), the GPU model/torch/CUDA/commit for the headline runs are recorded nowhere
machine-readable, the pod bootstrap clones a moving branch (`w16.sh:34`), the GPU env is unpinned
atop an arbitrary vast.ai image, and bf16 reductions are architecture-dependent with no determinism
flags. A 7 for this dimension conflates auditability of arithmetic with reproducibility of
evidence; an artifact-track evaluator who tries to regenerate Table 1 gets different code, unknown
hardware, and no per-trial ground truth to compare against. Add the unaddressed Llama license
(ungated `unsloth/` mirror used expressly to skip the Meta gate, `docs/week17-handover.md:71`) and
this is a 5 with a compliance flag, not a 7.

## A6. "The rank-vs-retrieval wall is the paper's best scientific asset" — plausibly a benchmark artifact with an untested confound

1. **Filler dependence.** The wall ("a better gist absorbs the needle") is measured only on the
   10-sentence cyclic haystack, whose intrinsic rank is tiny — at r128 the gist has enormous spare
   capacity, so absorbing the one out-of-subspace item is *maximally* easy. On natural text
   (intrinsic rank ≫ r), the gist is saturated by filler and the needle plausibly stays
   out-of-subspace at every rank — i.e., the wall may not exist off-benchmark. No realistic-filler
   replication exists.
2. **The r/n confound.** The "model-dependent onset" (Qwen collapses at r128, Llama at ~r256)
   coincides exactly with matched *relative* rank: Qwen's KV width is n=512 (4 KV heads × 128) vs
   Llama/Mistral n=1024 — r128/512 = r256/1024 = 0.25. The claimed model-dependence may be a single
   dimensionless ratio, never controlled: the matched-r/n retrieval cell (Llama r256) was never
   run. Same confound shadows the "1024-dim vt weakness" (C5). One ~$10 column settles it; until
   then "model-dependent onset" is uninterpretable.
3. **The fix does not transfer.** If needle-absorption were the mechanism, score-rank decoupling
   should help wherever the wall bites; it rescues Llama only and does nothing on Qwen/Mistral
   (0/8, `w16-ruler-intervals.md:18,40`). A mechanistic story whose intervention works on 1 of 3
   models is a correlation with a name.

## A7. "The honest-limits record raises trust" — there is no manuscript for the honesty to live in

The honesty is real and it lives in `docs/` — a private lab notebook. The only paper artifact is
`paper/main.tex`: 157 lines, **3 citations**, Week-4-era content with a retraction note; no
related work, no Week-9–17 result, no limitation section. Every panel score, including this one, is
a score for a *hypothetical* document. And honesty is symmetric: honestly reported, the current
evidence says "we beat nobody on any measured axis, on a benchmark we wrote, at n whose
effective size is ~5." NeurIPS does not accept candor in lieu of a contribution; it accepts
contributions stated candidly.

---

## What survives cross-examination (the honest residue)

1. **The idea is real and unclaimed**: no prior work applies a rank-adaptive DLRA integrator to
   the streaming KV cache; per-sequence, training-free, online subspace adaptation remains a
   genuine open axis (prior-work review's search stands).
2. **The engineering substrate is trustworthy**: accounting formulas test-pinned to live caches,
   parity ladders, strict-mypy CI, a doubling-guarded interval script. Whatever is eventually
   claimed can be audited.
3. **The tier–gist rank-siphoning diagnosis** (Week-17) is a real observation about coupling a
   selection mechanism to a low-rank tracker, worth a stability subsection.
4. **The program knows how to kill its own claims** (Q-BUG, codebook, h512/h1024, Palu-separation
   downgrade) — the *process* is publishable-grade even where the results are not.

## Score: 3/10 (reject as it stands)

Calibration: this is the harshest-realistic-reviewer score for the would-be paper, not for the
program. The acceptance case rests on four pillars — firm generality, a novel floor, a real memory
frontier, exemplary statistics — and each fails under examination: ceiling-effect tables with zero
separated cells, a re-enabled published truncation step shipped default-off, a storage ratio whose
honest asymptote equals the uncompared 2-bit baseline, and violated pre-registration against a
strawman comparator. The path back up is the same one the other reviews price out (~$100–150 GPU +
weeks of writing): discriminative benchmarks (official RULER, realistic filler, depth sweep,
multikey restored), the missing baseline families (EA/SnapKV at 0.1×, KIVI-2bit), one matched-r/n
control, honest fp32/fp16-probe accounting, a measured persistence number, and a manuscript that
frames the surviving contribution — a training-free online low-rank KV tracker with a surprise
tier, its failure modes, and its stability requirements — at the size it actually is.
