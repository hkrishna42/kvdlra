# Experimental rigor / statistics review — kvdlra exit-gate re-review (2026-09-06)

Reviewer dimension: statistics, pairing, confounds, the official anchor, the q4 invalidation.
Repo `/Users/hari/Desktop/kv-dlra` @ `week7` (HEAD 5bc772f). $0, read-only; every McNemar/Wilson
number below was recomputed from the committed per-trial files with an exact binomial.

**Score: 6/10 (borderline accept / poster).** No FATAL flaw. The previous panel's fatal set for this
dimension (no 2-bit baseline; custom benchmark only) is closed, and closed *against interest*: the
fair KIVI arm is built correctly, paired needle-for-needle, and the official RULER anchor is reported
even though it takes the multi-value headline away. What keeps this at 6 rather than 7: the abstract
promotes uncorrected exploratory per-cell p-values after the pre-registered A1 fund bar *failed*; two
underpowered nulls are worded as equivalences; the perplexity scorer accumulates NLL in bf16 (every
per-window sum in the logs is an integer multiple of 4/8 nats), which puts several quoted fluency
deltas at the rounding floor; and the "KIVI-faithful" arm silently runs a chunked, lossy prefill with
no single-shot control. All four are $0–$10 fixes.

---

## 0. Credit where due (verified, not asserted)

- **Right tools.** Wilson score intervals (`scripts/w15_intervals.py:55-64`, correct formula, z=1.96)
  on every cell; exact two-sided McNemar on discordant pairs (`scripts/w18_intervals.py:73-92`,
  `binomtest(min(b,c), n_disc, 0.5)` = 2·P(X≤min), the right test for paired all-or-nothing trials at
  n=12). `n` is read off the emitted `n=` field, hits = `round(acc·n)` (`:47-61`) — unambiguous at
  n≤16 with 2-dp accuracies.
- **Pairing is real, verified two ways.** (a) Code: the generator path (`build_task`,
  `_filler_to`, `_codes`, `_LABELS`, `_FILLER`) is byte-identical between the Week-18 flagship pods
  (SHA 15678a7 → b157acd: only `_plot` guard + two argparse flags, `git diff 15678a7 b157acd --
  scripts/w10_ruler.py`) and the Week-19 quant pods (b157acd → 6734afa: only the quant branch of
  `retrieve()` + flag rename, lines 317-327/539-546 of the diff); `w4_needle.py`/`w5_ruler.py` have no
  diff at all. The later memoization (673331f, pods a1q/a2/a4) is pinned equal-output by
  `tests/test_w10_ruler_filler.py:112-128` and the `cycle` path has no seed dependence
  (`scripts/w10_ruler.py:100-105`). Same `transformers=5.8.0`, `CHUNK=4096`, `DTYPE=bfloat16`
  (`results/w18-env-provenance.txt`; `results/w19_harvest/a1-llama.raw` header;
  `scripts/pod/w18_boot.sh:32-33`). (b) Records: the (seed,trial) key sets are identical across arms,
  zero duplicate keys in any line-file, and my recomputation of all 48 A1 contrasts from
  `results/w18_pertrial/{llama,mistral,qwen}-trials.txt` + `results/w19_pertrial/a1-*-trials.txt`
  reproduces `results/w19_intervals/a1-*-ruler-intervals.md` exactly (all 7 significant cells:
  Llama 16K mv 7/0 p=.0156; Mistral 16K mv 6/0 p=.0312, 32K mk 9/0 & mv 9/0 p=.0039; Qwen 16K mv 8/0
  p=.0078, 32K mv 10/0 p=.0020, vt 9/0 p=.0039). The official-anchor record-index sets are identical
  across all 7 arms for all 9 tasks (`results/w19_pertrial/a2-llama-trials.txt`).
- **Reported against interest.** Official anchor mean 0.79 vs 0.87 (`paper/main.tex:762-765`);
  4-bit dominates on Qwen (`:663-666`); the Week-18 q4 table annotated INVALID rather than deleted
  (`results/w18-g1-report.md:18-23`); the first a3 persist run dropped with its cause named
  (commit 69f7ea9, 5e8b275; `results/w19_harvest/a3-llama.raw.superseded` flagship 0.40x/0.26x
  view-inflated vs the fixed 0.151x/0.140x in `results/w19-a3-llama2-lines.txt:1,4`).
- **Marquee statistics hold up.** `results/w18-g4-marquee-contrasts.json`: think 10/0 discordant
  (p=0.0020), palu 6/0 (p=0.0312), n=16, consistent with 15/16 vs 5/16 and 9/16; the paper's
  "confirmatory vs suggestive" split (`paper/main.tex:466-484`) is the correct reading and the
  pre-registration is traceable (`scripts/pod/w18.sh:78-96` names the three arms and the cell).

---

## 1. Statistics

### 1.1 Multiple comparisons: the "family = 1" stance does not cover the Week-19 contrasts — FIXABLE ($0)

- The paper's only multiplicity statement (`paper/main.tex:479-484`) declares the marquee
  vt-vs-think contrast the single pre-registered confirmatory comparison and everything else
  exploratory. That is credible for the marquee (`scripts/pod/w18.sh:78-96`). It is **not** invoked
  for the fair-quant table, whose bolding rule is "paired McNemar p<0.05" uncorrected over 48 tests
  (`:687-688`), nor for the abstract, which states "wins multi-value retrieval on all three families
  (paired McNemar p≤0.03) and multi-key/variable-tracking at 32K on two" (`:75-78`) as a headline.
- What was actually pre-registered for A1 was a *different* bar: "if the flagship holds single+mv
  ≥0.7 Wilson-lo where 2-bit KIVI ≤0.3, the exclusive band is claimed; otherwise the band claim is
  retired" (`scripts/pod/w19.sh:15-18`; `docs/week19-kickoff.md:77-79`). The 2-bit mv point
  estimates are 0.42/0.50/0.33 (Wilson-hi 0.68/0.75/0.61) — **the bar failed**. The paper honours
  the retirement (`:672-680`, "narrower than an exclusive band") but then elevates the post-hoc
  per-cell p-values to the abstract. That is the textbook move pre-registration is meant to prevent.
- Numbers (recomputed): over the 24 flagship-vs-2-bit contrasts, 7 are nominal p<0.05;
  **Bonferroni: 1 survives (Qwen 32K mv, p=0.0020 < 0.0021); Holm: 1; BH(0.05): 5**. Over the full
  48-contrast table (2-bit + 4-bit): Bonferroni 0, BH 4. Even within the narrowest family the
  abstract names — the three 16K mv cells — Mistral (p=0.031) fails Bonferroni(3)=0.0167.
- The honest and *stronger* framing is the pooled paired sign test the authors already have the
  data for: 16K multi-value pooled over families = **21 vs 0 discordant, p≈1e-6**; all 24 cells
  pooled = **93 vs 5, p≈5e-22**; vs 4-bit pooled = 6 vs 8, p=0.79 (genuinely a tie). Seven of seven
  nominal cells one-directional is decisive as a pooled contrast and needs no per-cell p-values.
  Fix: state the pooled contrast as the (post-hoc but transparent) test, keep per-cell counts as
  descriptive, drop "p≤0.03" from the abstract, and either un-bold tab:fairquant or bold under BH.
- Stale count: "~40 descriptive Wilson cells" (`:479`) — the tables now carry ≈175 (xmodel 9,
  xmodel32 12, marquee 3, evict 16, fairquant 72, official 63).

### 1.2 Underpowered nulls worded as equivalence — FIXABLE ($0)

- **Minimum detectable discordance at n=12** (exact two-sided McNemar, α=0.05): 6 discordant pairs
  *all one way* (p=0.031) is the smallest significant pattern; 5/0 is p=0.0625; 7 vs 1 is p=0.070;
  8 vs 1 is p=0.039. I.e. the test cannot detect an accuracy gap below **0.50 with zero reversals**
  (≈0.58 with one reversal). Any per-task gap ≤0.33 is invisible by construction.
- The abstract says the flagship and the 2-bit arm "are indistinguishable" on the official suite
  (`paper/main.tex:80-82`; also `:678`, `:963` "no task separated"). Per task, the gaps are ≤0.33
  (vt 0.33 vs 0.67 = 1 vs 5 discordant, p=0.22) — exactly the undetectable band. The appropriate
  test is the pooled paired contrast over the 108 shared records: **flagship 86/108 vs 2-bit 94/108,
  8 vs 16 discordant, p=0.15**, with the direction 2:1 against the flagship (Wilson [0.71,0.86] vs
  [0.79,0.92]). "Not separated at this n (pooled p=0.15; point estimate trails)" is supportable;
  "indistinguishable" is not. Same fix for `:963`.
- Pooled, the anchor *does* separate the flagship from every larger arm: vs 4-bit 0 vs 21
  (p≈1e-6), vs think 0 vs 20, vs palu 6 vs 19 (p=0.015), vs full 0 vs 21 — all in the flagship's
  disfavour; and vs ea-k0.1 69 vs 5 in its favour. The paper reports only the per-task vt
  separations (`:765-767`); the pooled picture (flagship strictly below every ≥0.29x arm on the
  official prompts) belongs in the text.

### 1.3 n=8 at 64K — FIXABLE ($0)

- The abstract claims "full four-task retrieval at 64K" with no n (`paper/main.tex:91-93`);
  §memory gives n=8 (`:840`); Limitations gives n=8 (`:983-984`). Wilson-lo for 8/8 is **0.676**, so
  the abstract's sentence is "no detected loss at ≥0.68", weaker than every other headline (0.76).
- Worse, the side-by-side "1.00 on all four (n=8) where the 2-bit arm scores 1.00/0.58/0.50/1.00
  (n=12)" (`:840-842`) mixes sample sets: the flagship ran trials 0–3 × 2 seeds
  (`results/w19_pertrial/a4-llama-trials.txt`; `scripts/pod/w19.sh:64`), the comparators 0–5. On the
  **shared 8 needles** the 2-bit arm scores mk 4/8 and mv 4/8 (its n=12 figures are 7/12 and 6/12),
  so the paired contrast is 4/0 and 4/0, **p=0.125 each** — not separated. State n=8 in the abstract,
  and report the paired-8 numbers, not the n=12 comparator column, beside the flagship.

### 1.4 Descriptive cells are still near-replicates (carried from the 2026-09-01 review) — NOTE

- Unchanged: `niah_single` needle at `n//2 + trial%5` (`scripts/w10_ruler.py:204`), cyclic filler
  identical across trials/arms (`:100-111`), so a 12-trial cell is 2 seeds × 6 code draws at ≤5
  mid-depth positions of one haystack. The paired McNemar is the right defence for *contrasts*; the
  per-cell Wilson intervals remain intervals over code draws, not over a task distribution. The
  paper's setup paragraph says "small fixed filler pool" (`paper/main.tex:374-376`) but never says
  "single-needle is mid-depth only on the in-repo generator". One sentence.

### 1.5 The perplexity scorer accumulates NLL in bf16 — FIXABLE (1-line fix; ~$15 to re-score)

- Evidence: every per-window summed NLL in the logs is an integer multiple of 4 nats for sums in
  [512,1024) and of 8 nats for sums in [1024,2048) — the bf16 (8-bit mantissa) quantization steps.
  `[pplw]` lines × ntok=511: a1-llama 16K quant-2bit `864,684,920,980`; g4-llama 32K full
  `980,1096,984,1064,920,724,1176,996`, marquee `1012,1120,996,1080,944,724,1232,1048`; a4-llama
  64K ea-k0.1 `1208,1168,800,1112`, quant-2bit `1216,1176,768,1128` (`results/w19_harvest/a4-llama.raw:
  37428,37639`; `results/w18_harvest/g4-llama.raw`). Cause: `nll = cross_entropy(logits[:-1], …,
  reduction="sum")` on the model's bf16 logits (`scripts/w10_frontier.py:69`; transformers 5.8
  `modeling_llama.py:487` does not upcast) returns a bf16 scalar.
- Consequence: each window's sum carries ±2–4 nats of rounding on ≈1000, i.e. **±0.006–0.011
  bits/token per window** (worst case). The paper quotes fluency differences at that scale: "at 16K
  the gap to the baselines is ≤0.006 bits/token" (`paper/main.tex:515`); marquee "+0.024/+0.031
  bits/token" (`:513-514`; also note the order is swapped — think is +0.031, palu +0.024 from
  7.196/7.232 vs 7.353); Llama 16K flagship 5.31 vs 2-bit 5.40 (0.024 bits, `:669`); 32K 8.33 vs 8.28
  (0.009 bits); and the 64K "8.15 (2-bit and eviction)" (`:841-842`) is literally the same
  bf16-rounded sum 4288 for two different arms (ea `1208+1168+800+1112` = quant `1216+1176+768+1128`),
  which the paper presents as a coincidence of nature. The pre-registered 0.05-bit band is safe; sub-
  0.02-bit statements are not. Fix: `logits.float()` before `cross_entropy` (and mean in fp32), then
  re-score the cells the paper quotes at <0.03 bits (marquee n=8, fair-quant PPL4, 64K): ~$15.
  Also carried: the W19 fluency table is PPL4 (n=4 windows, `scripts/pod/w19.sh:26`) while the
  pre-registered tie rule was defined at n=8 (prior review §e); no SE is reported for any W19 ppl.

---

## 2. Pairing — VERIFIED (code + records), one residual, one provenance slip

- Verified as in §0. Residual risk: the model/tokenizer revision is not pinned
  (`scripts/perplexity_sweep.py:76-77`, plain `from_pretrained(model_id)`; no `revision`); the
  haystack sentence count `n` depends on the tokenizer (`scripts/w10_ruler.py:108`). If the
  `unsloth` mirror's tokenizer changed between 2026-09-03 and 09-05 the needles would differ
  silently. Cheap tripwire: print a hash of `hay` per `[trial]` line ($0, future runs).
- Provenance slip in the paper: "the quantization arm at 15678a7 and the flagship/eviction/firming
  grids at b157acd" (`paper/main.tex:975-976`). Per `results/w18-env-provenance.txt` the g1 pods
  (= the **flagship 16K/32K grids + the W18 quant + q4 arms**) ran at 15678a7; g2–g5 (filler,
  eviction, marquee firming, storage) at b157acd. Tables xmodel/xmodel32 and the flagship rows of
  tab:fairquant are 15678a7. Harmless (generator identical) but wrong as written.
- The W19 per-trial files (`results/w19_pertrial/`) are never named in the paper; `:372-374`
  points only at `results/w18_pertrial/`. Nit.

---

## 3. Confounds

### 3.1 Chunked, lossy prefill for the "KIVI-faithful" arm; no single-shot control — FIXABLE (~$10)

- Week-18 quant arm: single-shot prefill (`git show b157acd:scripts/w10_ruler.py:319-327`). Week-19:
  `_prefill_plain(model, cache, hay, chunk)` with CHUNK=4096 + `flush()`
  (`scripts/w10_ruler.py:334-344`; `scripts/w10_frontier.py:139-157`). The stated reason — "the
  single-shot 16K/32K quant prefill OOM'd even on 80GB" (`w10_ruler.py:336-337`) — was the missing
  `no_grad` on the *ppl* path (`results/w18_harvest/quant-findings.md:77-81`, fixed in 6734afa); the
  W18 RULER quant path was decorated and its single-shot 16K/32K rows ran (Qwen g1quant). So the
  protocol change was not forced, and it is undisclosed in the paper (§quantbaseline `:630-646`
  describes the scheme only; only `docs/week19-official-ruler.md:31-33` mentions it).
- What chunking does to a `QuantizedLayer` (transformers 5.8 `cache_utils.py:542-578`): after
  chunk 1 is quantized, chunk 2 (4096 tokens) sits entirely in the fp16 residual (`:574`, the
  `else` branch, because the residual is empty), chunk 3 triggers re-quantization of
  `[dequant(1), 2, 3]` (`:568-571`), chunk 4 sits in the residual until `flush()`. So (i) the
  prefill attention of chunks 2–4 reads 2-bit-dequantized earlier keys/values — errors compound
  through 4 chunks and 32 layers, unlike KIVI's full-precision prefill; (ii) chunk 1 is quantized →
  dequantized → re-quantized twice (near-idempotent for min/max affine groups on aligned
  boundaries, so second-order). The flagship is chunk-streamed too, so flagship-vs-quant is
  protocol-*matched*, but the arm is then "KIVI-scheme under a streaming protocol", not the KIVI
  operating point, and the presses/full run exact single-shot prefill (`w10_ruler.py:345-354`).
- Bounding evidence that this is not the whole story: under the same chunked protocol the 2-bit arm
  scores 0.83 on official multi-value (= flagship) and 4-bit/8-bit chunked arms are perfect
  (`results/w19-a1-report.md:14-15,61-62,111-112`). But 2-bit noise is 4× 4-bit noise and the
  compounding is 2-bit-specific by construction, so the mv edge on the in-repo generator is not
  cleared of it. **Control**: one family (Llama), 16K, `quant-2bit-kivi` with `--chunk 0`, all four
  tasks, n=12 (fits: the W18 single-shot rows ran at 16K on a 40GB card) — ~$10. Disclose either way.
- Residual flush semantics are otherwise correct (`src/kvdlra/quant/kivi_cache.py:88-98`; pinned by
  `tests/test_w19_quant_kivi.py:109-125`): post-flush state equals the single-shot state; the decoded
  tail (≤88 tokens) accumulates in the residual below `residual_length`, as KIVI would.

### 3.2 Aux billing at bf16 — CORRECT; residual over-billed vs measured — NIT

- `aux_words()` reads the backend's stored scale/shift dtype (`kivi_cache.py:161-172`): 16-bit
  pairs on a bf16 model → 2-bit/g64 asymptote (2 + 32/64)/16 = 0.156x. Honest and measured.
- But `quant_footprint` charges a 128-token fp16 residual from the arm *config*
  (`scripts/w10_frontier.py:531-538` → `src/kvdlra/accounting.py:443-449`) that `flush()` has
  emptied. Hence tab:fairquant bills the 2-bit arm at 0.163x (16K) / 0.160x (32K)
  (`paper/main.tex:650,696`) while the serialized cache measures 0.1563x
  (`results/w19-a3-llama2-lines.txt:3,6`) and §subcliff/§memory quote 0.156x (`:576,899`). The
  over-bill is against the quantizer (conservative for the matched-bytes claim: 0.151 vs 0.156 is
  still matched) but the paper carries two numbers for one arm without reconciling them.

### 3.3 fp32-at-rest for BUG — CORRECT and conservative

- `bug_footprint` bills `U` and the coordinate columns at 32 bits, sinks/ring/exact tier at 16,
  positions/surprise as 32-bit aux (`src/kvdlra/accounting.py:168-199`); `Footprint.stored_bits`
  (`:84-99`). Sanity: the 0.048x compose cell reconstructs analytically
  (U 4.2M + fp32 coords 2.1M + 4-bit codes 8.0M + norms 1.0M + hh-256 verbatim 8.4M + sink/ring 1.7M
  + aux 1.0M ≈ 26.4M bits / 537M = 0.049). Note for the reader: the 256-token exact tier is 32% of
  that cell and `U` 16% — the "1/T" term is only the coordinate part.

### 3.4 Compose-arm budget semantics — CORRECT now; disclosed in results, not in the paper text

- `--bug-quant-budget` = fp32 columns kept (512, fixed in T), quant tier = the whole middle
  (`scripts/w10_frontier.py:268-282`, `:279-280`); pinned by
  `tests/test_w10_ruler_quant.py:158-178` (`_q_len() > 0`). Ratio falls with T as claimed
  (0.048 → 0.034, `results/w19-a1q-llama-lines.txt`).
- The compose cell is compared to the 2-bit arm in prose ("that the 2-bit arm at three times the
  bytes loses (0.67/0.42 …)", `paper/main.tex:582-583`) with no test, although the needles are the
  same (a1q pods post-memoization, bit-identical generator): Llama 16K mk 12/12 vs 8/12, mv 12/12 vs
  5/12 — a paired contrast is free and would be ≥ 4/0 and ≥ 7/0. Add it or drop the comparison.

### 3.5 The "earlier draft 0.00" story — VERIFIED on Qwen, asserted elsewhere — NIT

- Root cause (quanto `axis=0` = per-token groups; KIVI = per-channel keys) is documented with a CPU
  synthetic (`results/w18_harvest/quant-findings.md:30-50`), replicated on GPU under the token scheme
  (Qwen 16K single, 0.00, n=4, `:66-76`) and cross-checked with an hqq 2-bit KIVI arm (1.00). On
  Llama/Mistral the W18 quant arms never ran (all rows `SKIP ImportError: quanto_cuda.so`,
  `results/w18_harvest/{llama,mistral}.raw`), so the "0.00" was a Qwen-only observation. The paper's
  explanation (`paper/main.tex:640-645`) is consistent but should say "measured on Qwen".

### 3.6 Wikitext-filler paragraph juxtaposes unpaired, unequal task sets — FIXABLE (~$10 or $0)

- `paper/main.tex:736-745`: "the flagship holds single-needle retrieval (1.00) while ea-k0.1
  collapses to 0.00 on all four tasks". The flagship's wikitext run was `niah_single` only, with a
  `--depths 0.1…0.9` grid (`scripts/pod/w18.sh:55-57`); the baselines ran all four tasks *without*
  the depth grid (`:58-61`), so the prompts differ and the flagship has **no** mk/mv/vt measurement
  on realistic filler anywhere (`results/w18_pertrial/g2-qwen-trials.txt`: 12 flagship lines, all
  `niah_single`). Literally true, but the sentence invites the four-task inference. Either run the
  three missing flagship cells (Qwen 16K, n=12, ~$10) or say "single-needle only".

---

## 4. The official anchor

- **Protocol**: sound and honestly described (`docs/week19-official-ruler.md:12-44`;
  `scripts/w19_official_ruler.py:52-95`); records shared across arms (verified §0); RULER's
  `string_match_all` semantics (`:9`). Twelve records per task is 1/40 of RULER's default 500 and is
  the power problem of §1.2.
- **Depth/miss analysis** (`results/w19-a2-flagship-misses.md`; `paper/main.tex:767-769`, "not a
  warm-up-window or recency effect but scattered misses"): of the 22 flagship misses, 8 are vt (no
  depth) and 14 have depths `[0.15,0.16,0.17,0.17,0.20,0.20,0.41,0.48,0.57,0.59,0.82,0.83,0.89,0.95]`.
  Six of 14 are at depth ≤0.20 (uniform expectation 2.8; binomial P(X≥6)=0.044), then **none** in
  (0.20, 0.41). Depth 0.15–0.20 at 16K is tokens 2.5K–3.3K — inside the *first 4096-token ingest
  chunk*, the exact window the warm-up seed exists to repair (`paper/main.tex:239-244`). With 14
  points this is suggestive, not conclusive (against a 0.25 boundary P(X≥6)=0.11), but the categorical
  "not a warm-up-window effect" is an over-read; "front-loaded (6/14 in the first chunk), with a
  mid/late tail" is what the table shows. The 2-bit arm's misses are also front-loaded
  (4/10 at ≤0.26). A depth-stratified single-needle run on Llama (the in-repo `--depths` grid was
  Qwen-only, `scripts/pod/w18.sh:55`) would settle it: ~$10.
- **"No cell separated"**: see §1.2 — at n=12 that is guaranteed for any gap <0.5; the pooled test
  (8 vs 16, p=0.15) is the informative one and trails.
- **Template**: prompts generated with `--model_template_type base` and re-wrapped in the
  tokenizer's chat template + answer prefix (`docs/week19-official-ruler.md:40-42`;
  `w19_official_ruler.py:63-95`) — a documented deviation from RULER's `meta-llama3` template; fine,
  but it is *not* "their template" as `paper/main.tex:753-755` says ("templates").

---

## 5. The Week-18 q4 invalidation

- **Disclosure**: adequate in results (`results/w18-g1-report.md:18-23` INVALID banner with the
  mechanism; `results/w18_harvest/quant-findings.md:84-94`), and the paper never printed those
  numbers (v1.1 cited the W11 `bugS-r32-h256` 0.043x cell). In the manuscript the fact survives only
  as a LaTeX source comment (`paper/main.tex:608`) — invisible in the PDF. A one-line footnote in
  §subcliff ("an earlier compose arm's quant tier never filled; those rows were the unseeded
  flagship") costs nothing and matches the paper's own practice for the retracted 1B claim (`:812-813`).
- **Root cause and blast radius**: `build_arms` passed `coord_budget = t+rw+ab` (whole context
  fp32) and `quant_budget = 512`, the inverse of the cache's documented semantics
  (`src/kvdlra/cache/bug_cache.py:99-100`), so nothing was ever demoted; the rows bill
  byte-identically to the unseeded flagship (`results/w18-llama-lines.txt`: q4 and bugSseed both
  ratio 0.085 / sbits 0.151). **Other arms**: none share it — `qbits` is set only by the `-q` arm;
  `bug`/`bugslash` intend `cb`, `bugevict` intends `coord_budget=1` (`w10_frontier.py:207-221,
  322-347`); `w5_streamppl.py:111-132` passes explicit Wf/Wq budgets (Week-5, not in the paper); the
  1B composition frontier (§frontier, `paper/main.tex:800-817`) uses the BUGPress+TurboQuant path
  (`scripts/w4_fair.py:1-15`, `accounting.bug_prefill_footprint`), a different code path. The
  `--bug-quant-budget` flag in `scripts/w10_longbench.py:324` inherits the fix via `build_arms`; no
  q4 LongBench row was ever published.
- **Residual hazard**: the fix was validated by a tier-fills test, not by an equivalence test
  against an independent quantizer; and the Qwen compose divergence (ppl 14,490, 0/0/0/0,
  `results/w19-a1q-qwen-lines.txt`) is reported as a family property without a diagnostic. Fine
  for v1 if labelled as such.

---

## Verdict lists

**FATAL:** none.

**Fixable (must-do before the exit gate):**
1. Multiplicity/pre-registration mismatch for the Week-19 contrasts (§1.1) — $0.
2. "Indistinguishable"/"no task separated" from an n=12 design with a 0.5 minimum detectable gap;
   report the pooled paired contrasts (§1.2) — $0.
3. bf16 NLL accumulation in the ppl scorer; sub-0.03-bit fluency statements are at the rounding
   floor; the 64K "8.15 = 8.15" is a rounding collision (§1.5) — 1-line fix + ~$15 re-score.
4. Chunked lossy prefill for the "KIVI-faithful" arm, undisclosed, no single-shot control (§3.1) —
   $0 disclosure + ~$10 control.
5. 64K: n=8 missing from the abstract; comparator numbers quoted on 12 trials the flagship did not
   run; paired-8 is 4/0, 4/0, p=0.125 (§1.3) — $0.
6. Official-anchor depth claim over-read; first-chunk cluster (§4) — $0 wording, ~$10 to test.
7. Wikitext-filler sentence implies four-task flagship coverage that was never run (§3.6) — $0/$10.

**Nitpicks:** provenance SHAs swapped (`:975-976`); "zero Wilson-separated flagship-vs-baseline
cells at 16K" (`:397`) is now false against the 2-bit arm (mv 1.00 [0.76,1] vs 0.42 [0.19,0.68] /
0.33 [0.14,0.61] / 0.50 [0.25,0.75] are Wilson-separated) — scope it to think/palu; "+0.024/+0.031"
order swapped (`:513`); "~40 descriptive Wilson cells" stale (`:479`); dangling "Three" fragment
(`:585-586`) renders in the PDF; 0.163x vs 0.156x for one arm (§3.2); "measured on Qwen" for the
earlier-draft zero (§3.5); `results/w19_pertrial/` unnamed in the paper; official-anchor "templates"
wording (§4).

---

## Top 5 findings (for the summary)

1. The abstract's C1 p-values are post-hoc per-cell tests promoted after the pre-registered A1 bar
   failed; only 1 of 24 survives Bonferroni/Holm — while the pooled paired contrast (93 vs 5,
   p≈5e-22; 16K mv 21 vs 0) is decisive and should replace them.
2. Two nulls are worded as equivalence ("indistinguishable", "no cell separated") where n=12 cannot
   detect a gap <0.5; pooled over 108 official records the flagship trails 2:1 (8 vs 16, p=0.15).
3. The perplexity scorer sums NLL in bf16 (every window sum is a multiple of 4/8 nats): ±0.006–0.011
   bits/token per window, so "≤0.006 bits/token", "+0.024/+0.031", and the identical 64K "8.15/8.15"
   are at or below the rounding floor.
4. The 2-bit arm runs a chunked, lossy prefill that compounds 2-bit error through 4 chunks; the
   OOM that motivated it was the (fixed) no_grad bug; undisclosed in the paper; no single-shot control.
5. Pairing is genuinely verified (generator byte-identical across the four SHAs; record keys
   identical; all 48 A1 contrasts reproduced), which is why none of the above is fatal.

## Three highest-leverage fixes (cost)

1. **$0 — Rewrite the inferential framing**: pooled paired sign tests (in-repo: 93 vs 5; official:
   8 vs 16) as the reported tests; per-cell counts descriptive; drop p-values from the abstract;
   "not separated at n=12 (power floor 0.5)" instead of "indistinguishable"; n=8 + Wilson-lo 0.68 at
   64K with the paired-8 comparison; scope `:397` to think/palu; fix the SHA sentence and the
   "Three" fragment.
2. **~$10 GPU — Single-shot 2-bit control** (Llama 16K, `--chunk 0`, four tasks, n=12): either
   armours the in-repo mv edge or reclassifies it as a streaming-protocol result. Add one sentence
   disclosing chunked prefill for the quant arm either way.
3. **1 line + ~$15 GPU — `logits.float()` in `_score_window`** and re-score the marquee n=8, the
   fair-quant PPL4 cells, and the 64K row; quote fluency deltas with a resolution statement (or
   only at the 0.05-bit pre-registered band).

With (1)–(3) this dimension is a 7; the science does not change, the wording and one numerical
protocol do.
