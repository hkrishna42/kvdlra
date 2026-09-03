# Prior-work & novelty review — kvdlra NeurIPS-readiness panel

Reviewer dimension: **prior work, novelty, positioning, missing baselines/citations**
Repo: `/Users/hari/Desktop/kv-dlra` @ branch `week7` (HEAD cd3d4e7). Date: 2026-09-01.

## 0. What I read / searched

Repo: `paper/main.tex` (the only draft — 157 lines, 3 citations, Week-4-era content);
`docs/week17-explained.md`, `docs/week16-explained.md:69`, `docs/week16-handover.md:42`,
`docs/week11-decision-table.md:26-33`, `docs/week5.md:201-299`, `docs/week7-dominance.md`,
`results/w17-decision-table.json`, `src/kvdlra/integrators/streaming_torch.py:1-56`,
`src/kvdlra/cache/bug_cache.py:74-560`.
Literature (8 searches, 2023–2026): Palu, Eigen Attention, LESS, LoLA, xKV, LoRC, KV-CoRE,
KQ-SVD, EliteKV, KIVI, KVQuant, MomentKV, ResKV, CaM, KVMerger, KVSlimmer, SelKV, DLRT
(Schotthöfer et al.), DLRA compression of NNs (NeurIPS 2025), ShadowKV, surveys 2412.19442 / 2407.18003.

## 1. Originality verdict

**The core mechanism is genuinely novel; the positioning is dangerously under-built.**

### 1a. What is new (and defensibly so)

1. **DLRA/BUG integrator applied to the KV stream.** I found *no* prior or concurrent work
   applying the Ceruti–Kusch–Lubich rank-adaptive BUG integrator (arXiv:2104.05247) — or any
   DLRA integrator — to online KV-cache compression. The implementation
   (`src/kvdlra/integrators/streaming_torch.py:28-45`: range-augment → Galerkin core →
   Frobenius-tail truncate, blocked variant interpolating between per-token tracker and SVD
   oracle) is a faithful, non-trivial import from numerical analysis, with the near-oracle
   validation (1.01–1.03× of truncated SVD, `paper/main.tex:95-98`) giving it a principled
   story no heuristic tracker in the KV literature has. The Week-17 `min_sv_frac` relative
   singular-value floor (`bug_cache.py:420-429`; Qwen bug-r256 ppl 27531.7→6.995,
   `results/w17-decision-table.json` qwen/16384) is a real numerical-analysis contribution —
   *if* framed as such; there is no analogue in Palu/xKV-style offline SVD work because they
   never integrate a stream.

2. **Per-sequence, training-free, online adaptivity.** The closest low-rank competitors are
   all *offline/calibration-time* projections: Palu (arXiv:2407.21118, ICLR 2025) decomposes
   the K/V *weight* matrices with a rank search on calibration data; Eigen Attention
   (arXiv:2408.05646, EMNLP-Findings 2024) builds fixed low-rank bases from calibration;
   LoRC (arXiv:2410.03111) progressively compresses weight matrices; xKV (arXiv:2503.18893)
   runs SVD at prefill across layers; KQ-SVD (arXiv:2512.05916) is closed-form offline;
   EliteKV joint low-rank projection; MatryoshkaKV trains nested projections. None adapts the
   subspace *during* a single sequence as tokens stream. That axis (data-of-this-sequence,
   zero calibration, rank-adaptive online) is BUG's honest differentiator and survives my
   search.

3. **The mechanistic decomposition** (gist=fluency vs exact-tier=retrieval, C2) and the
   rank-vs-retrieval "needle absorption" wall with model-dependent onset is, as far as I can
   find, not articulated anywhere in the literature; it is a genuinely interesting analysis
   contribution.

### 1b. What is NOT as new as the docs imply

1. **The hybrid design pattern (constant-size low-rank state + small exact cache) is
   anticipated by LESS** (Dong et al., ICML 2024, arXiv:2402.09398): a constant-size low-rank
   recurrent state accumulating information from evicted tokens + a sparse-policy exact
   cache. Functionally this is the same architecture as gist+exact-tier. Differences that must
   be argued explicitly: LESS *learns* row-wise kernel functions per model (training
   required), approximates the softmax numerator/denominator rather than reconstructing K/V,
   and its low-rank state is tiny (~4 KV pairs equivalent). BUG is training-free and
   reconstructs. LoLA (arXiv:2505.23666, 2025) extends the same low-rank-linear + sparse-cache
   pattern. Neither is cited anywhere in the repo (grep for "LESS"/"LoLA" in docs: no
   design-pattern discussion; `paper/main.tex` cites only Ceruti×2 + TurboQuant). A NeurIPS
   reviewer WILL frame the paper as "training-free LESS with a principled tracker" — the paper
   must own that framing first.

2. **Surprise = out-of-subspace residual selection has concurrent near-duplicates.**
   MomentKV (arXiv:2606.01563, June 2026) scores tokens by attention weight × *residual norm
   w.r.t. a summary subspace* — "a large residual signals directional content that would be
   lost upon eviction" — which is the same signal as `bug_cache.py`'s
   `||k − UUᵀk||/||k||` surprise (`bug_cache.py:74-88`, Week-9 D3). ResKV
   (arXiv:2607.29591) splits an exact main cache + compact residual cache. Both are 2026
   concurrent work (kvdlra's Week-9 surprise selection predates them in-repo), so this is
   citable-as-concurrent, not novelty-destroying — but only if the paper is written and
   timestamped soon. Every month of delay converts these from "concurrent" to "prior."

3. **DLRA-in-ML is established.** DLRT (Schotthöfer, Zangrando, Kusch, Ceruti, Tudisco,
   NeurIPS 2022 "Low-rank lottery tickets") uses rank-adaptive DLRA integrators for NN
   *training*, and a NeurIPS 2025 follow-up (arXiv:2505.08022) uses it for NN *compression*.
   The novelty claim must be scoped to "first application to KV-cache/streaming inference
   state," not "DLRA meets ML." Currently `paper/main.tex` cites neither.

## 2. The missing-baseline problem (most damaging finding)

### 2a. KV quantization is a direct competitor inside the claimed exclusive regime — and it is absent

C1's flagship framing: 0.075–0.149× is "a regime eviction/channel-pruning cannot enter"
(`docs/week16-handover.md:42`, `docs/week16-explained.md:69`; briefing C1). That sentence is
*literally* true for eviction and channel-pruning, and the decision tables honestly show it
(ThinK 0.75×, Palu 0.50×, SnapKV floors at 0.10× with retrieval collapse,
`docs/week11-decision-table.md:26-33`). But the sentence is **false for the compressor class
the comparison omits**:

- **KIVI** (ICML 2024, arXiv:2402.02750): tuning-free 2-bit KV → 0.125× payload
  (~0.14–0.19× with group/residual overhead), "almost the same quality" on Llama/Mistral/
  Falcon, streaming-friendly (per-channel K, per-token V). This sits **inside** 0.149× and at
  the top of the claimed band, with near-lossless quality — kvdlra's own r64 config pays
  1.08–1.32× full ppl at 16K and 4.5× full ppl on Qwen at 32K (35.1 vs 7.76,
  `w17-decision-table.json` qwen/32768; `docs/week17-explained.md:113-118`).
- **KVQuant** (NeurIPS 2024, arXiv:2401.18079): 3-bit <0.1 ppl degradation (≈0.19×);
  2-bit with Q-Norm viable. Also *pre-RoPE per-channel keys* — the same operating point
  kvdlra rediscovers (`paper/main.tex:71-74`) with no citation.
- **xKV** (arXiv:2503.18893): cross-layer SVD, "up to 8× native compression" = 0.125× with
  accuracy maintained on long-context tasks — another published entrant *inside* the band.

The repo itself knows quantization is the elephant: `paper/main.tex:119` admits "pure 4-bit
TurboQuant is near-lossless at 0.25×," and Week-5 ran fair ×TurboQuant compositions
(`docs/week5.md:201`, "SnapKV/EA ×4-bit sit essentially near-lossless"). The Week-15→17
headline program then **dropped quantized baselines entirely** (briefing: "no quantization
baselines … ABSENT") while claiming an exclusive memory regime measured in
float-equivalent `ratio_fp16` — an accounting unit in which bits are exactly what
quantization buys. This is the single most attackable sentence in the would-be paper: one
reviewer sentence — "KIVI is tuning-free, 2-bit, near-lossless, and lives at 0.13–0.19×;
Table X omits it" — puts C1 at risk without a single new experiment by the authors.

**Severity: fatal to C1's current wording; fixable.** Two repairs, both needed:
1. Run KIVI-2bit (HF-integrated, tuning-free) and/or KVQuant-3bit on the three models at
   16K/32K, ppl + the same RULER tasks (~$50 GPU at Week-17 harness scale).
2. Re-aim the uniqueness claim *below* the 2-bit floor: pure scalar quantization cliffs below
   ~0.125× payload (repo's own Week-4 finding: "2 bits is the cliff," `paper/main.tex:110`),
   while BUG composes — the repo already built BUG×TurboQuant (rank-64/4-bit = 0.099×,
   `paper/main.tex:109-113`) and Week-10 showed BUG "uniquely reaches 0.033×". The defensible
   exclusive band is ≲0.06×, or "matched-quality at 0.075× *without* touching bit-width, and
   multiplicative with it." That is still a real claim — but it must be measured at 16K/32K
   retrieval, not inherited from ctx-1024 Week-4 numbers.

### 2b. Other absent comparisons, ranked by danger

| Absent | Why it matters | Danger |
|---|---|---|
| **KIVI / KVQuant / GEAR / ZipCache (quantization)** | occupies the claimed exclusive regime at near-lossless quality | **fatal-if-unaddressed (C1)** |
| **xKV (cross-layer SVD)** | 0.125× published, low-rank family, accuracy held | major (C1) |
| **LESS / LoLA (hybrid gist+exact)** | same architecture, learned; novelty framing | major (novelty of C2, citation only may suffice) |
| **Eigen Attention** | canonical low-rank-KV citation; NOT the repo's "EA" (ExpectedAttention) — a reviewer confusion trap | major (positioning; cheap to cite, name-collision must be dispelled) |
| **Official RULER / LongBench** | all retrieval numbers are custom in-repo tasks (`scripts/w10_ruler.py`, briefing note); zero cross-paper comparability with the very baselines' published numbers | major (shared with eval reviewer) |
| MatryoshkaKV, LoRC, KQ-SVD, EliteKV, KV-CoRE | low-rank KV neighborhood, cite+discuss | minor |
| CaM / KVMerger / KVSlimmer (merging) | "retain info from evicted tokens" family; repo explored merge (`docs/week7-dominance.md:250`) | minor |
| MLA (DeepSeek) | low-rank KV by architecture — shows the regime is reachable by training; scoping citation | minor |
| Quest | selection w/o memory saving (bandwidth); one-line scoping | nitpick |
| H2O / ScissorHands / PyramidKV / NACL | standard eviction citations; SnapKV/MorphKV are compared, these are cite-only | nitpick |

### 2c. Citation debt is essentially total

`paper/main.tex:145-155` has **three** bibliography entries (Ceruti×2, TurboQuant) and its
experimental content stops at Week 4 (ctx 1024, WikiText-2, 1B, including a retraction note at
line 121). There is no related-work section anywhere in the repo. The 17 weeks of docs
name-check baselines (SnapKV, MorphKV, ThinK, Palu, ShadowKV, ExpectedAttention) but never
place BUG against LESS, KIVI, Eigen Attention, xKV, or DLRT. For NeurIPS this is not a
polish item; the paper's entire novelty case rests on a related-work section that does not
exist yet. Estimated ~20–25 required citations; writing cost is CPU-only but non-trivial
(the LESS-vs-BUG and KIVI-vs-BUG paragraphs are load-bearing arguments, not lists).

## 3. Claim-by-claim novelty assessment

- **C1** (extreme-compression frontier, exclusive regime): numbers real
  (`w17-decision-table.json`: 1.00 retrieval @0.085–0.149×, n=12 Wilson-firm), but
  "regime others cannot enter" is falsified by uncompared KIVI/KVQuant/xKV. **Reword +
  1 quantization baseline required.**
- **C2** (gist/exact decomposition, absorption wall): novel as analysis; architecture
  anticipated by LESS/LoLA (cite), surprise signal concurrent with MomentKV/ResKV (cite as
  concurrent). **Survives with honest framing.**
- **C3** (`min_sv_frac` floor): novel; no analogue in offline-SVD works. Strongest purely
  original artifact. Frame as a DLRA-numerics contribution (cite parallel rank-adaptive
  integrator arXiv:2304.05660 as context).
- **C4** (marquee vs ThinK/Palu): comparison set is fine but narrow — two baselines at fixed
  operating points (think-c0.5 at 0.75×, palu-r0.5 at 0.50×) vs a tuned BUG config; no
  quantized or hybrid competitor at matched memory. Honesty corrections in
  `docs/week17-explained.md:103-108` are commendable.
- **C5** (honest limits): unusually good for a preprint program; the vt-refutation
  (`week17-explained.md:40-56`) and Week-15 baseline-bug disclosure raise credibility.

## 4. Score and rationale

**Score: 5/10 (borderline reject) for the prior-work/novelty dimension as the repo stands.**

The mechanism core (streaming rank-adaptive BUG + surprise-selected exact tier +
`min_sv_frac`) is genuinely new — on that alone this would be a 7. Two things pull it down:
(1) the flagship uniqueness claim C1 is contradicted by a well-known, uncompared competitor
class (2-bit KV quantization) that a median reviewer will raise in their first read; (2) the
related-work apparatus is absent (3 citations, Week-4-era draft), so the paper currently
cannot even *state* its novelty correctly (no LESS, no DLRT, no Eigen-Attention
disambiguation). Both are fixable: one ~$50 GPU baseline run + a below-the-2-bit-floor
composition experiment + roughly a week of related-work writing moves this dimension to 7.

## 5. Concrete gap-fills (ranked)

1. **KIVI-2bit baseline** on Llama-3.1-8B/Qwen2.5-7B/Mistral-7B @16K+32K, same ppl+RULER
   harness, `ratio_fp16` accounting incl. group overhead. (~$50 GPU.) Without it C1 cannot
   ship.
2. **BUG×quant composition at the extreme edge** (r64-h256 coords at 4-bit → ~0.04×; the
   TurboQuant path exists, `paper/main.tex:76-86`) to re-establish an *actually* exclusive
   band below the scalar-quant cliff, with retrieval measured. (~$50 GPU.)
3. **Related-work section** (~20 citations; LESS/LoLA paragraph, quantization paragraph,
   offline-low-rank paragraph incl. Palu/EigenAttention/xKV/LoRC/MatryoshkaKV, eviction
   paragraph, merging paragraph, DLRA/DLRT paragraph, concurrent MomentKV/ResKV note; fix the
   EA name collision explicitly). (CPU-only, ~3–5 days of writing.)
4. **Reword C1** everywhere ("eviction/channel-pruning cannot enter" → scoped claim +
   quantization discussion). (CPU-only, hours.)
5. **Official RULER subset** run for the flagship config on one model to anchor the custom
   tasks to published numbers. (~$10–50 GPU; shared with eval reviewer.)
6. Cite-and-scope: DLRT (NeurIPS 2022) + arXiv:2505.08022 so "first DLRA in inference-state
   compression" is precise. (CPU-only, minutes.)

## Sources (key literature located)

- Palu: https://arxiv.org/abs/2407.21118 (ICLR 2025)
- LESS: https://arxiv.org/pdf/2402.09398 (ICML 2024)
- LoLA: https://arxiv.org/pdf/2505.23666
- Eigen Attention: https://arxiv.org/abs/2408.05646 (EMNLP-Findings 2024)
- xKV: https://arxiv.org/pdf/2503.18893
- LoRC: https://arxiv.org/abs/2410.03111
- KIVI: https://arxiv.org/abs/2402.02750 (ICML 2024)
- KVQuant: https://arxiv.org/abs/2401.18079 (NeurIPS 2024)
- MomentKV: https://arxiv.org/pdf/2606.01563 (concurrent, 2026)
- ResKV: https://arxiv.org/html/2607.29591 (concurrent, 2026)
- KVMerger: https://arxiv.org/abs/2407.08454; KV-CoRE: https://arxiv.org/html/2602.05929v2
- KQ-SVD: https://arxiv.org/abs/2512.05916; EliteKV: https://arxiv.org/pdf/2503.01586
- DLRT: https://proceedings.neurips.cc/paper_files/paper/2022/hash/7e98b00eeafcdaeb0c5661fb9355be3a-Abstract-Conference.html
- DLRA NN compression: https://arxiv.org/abs/2505.08022 (NeurIPS 2025)
- Parallel rank-adaptive DLRA integrator: https://arxiv.org/pdf/2304.05660
- Surveys: https://arxiv.org/pdf/2412.19442, https://arxiv.org/html/2407.18003v1
