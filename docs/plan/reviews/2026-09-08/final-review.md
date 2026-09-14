# AC final content review — kvdlra manuscript (2026-09-08)

Area Chair, final pass. Manuscript `paper/main.tex` (1167 lines) and `paper/refs.bib` (42 keys, all
resolve both ways — `comm` of used-vs-defined is empty in both directions) read in full at HEAD
`7a5e434` (branch `week7`). Inputs: `docs/reviews/2026-09-06/review-meta-verdict.md` (prior AC),
`results/w20-fork-report.md` (the decisive fork), the six dimension reviews, and the committed
line-files / per-trial records. $0, read-only; the only file written is this one. Every claim below
carries a `file:line` or a quoted paper line. Calibration: 4 reject / 5 borderline reject /
6 borderline accept (poster) / 7 accept / 8 strong.

**In-flight status (not guessed):** `results/w19_harvest/pods.txt:16-17` lists `swap-llama:50320261`
and `sysfix-llama:50320262` (uncommitted edit); neither is in `done.txt`; no `swap`, `latency`, or
`ss2` rows exist under `results/`; the untracked `results/w20-close-report.md` holds only the
committed reference rows and `--`/`(pending)` placeholders (`:9-11`, `:19`, `:27-29`). Nothing has
landed. Section 6 gives the finalization rule to apply the moment they do.

---

## 1. The direct answer

**Borderline-accept (6), not yet accept, at ICML 2027 calibration — but a high 6, and the
distance to 7 is now one un-launched ~$10 run plus the two pods already in flight plus a
$0 wording pass.** Zero fatal flaws; the fold of the decisive fork is present and, with two
sentence-level exceptions below, accurately stated. Three dimensions still sit below 7:

- **Significance 6.** The fork's *pre-registered* favourable branch (`scripts/pod/w19.sh:162-169`)
  is "composite COLLAPSES on essays ... **while q4 holds**" — both on the official anchor. The
  composites were run on the anchor and collapsed (`results/w20-fork-report.md:57-61`, means
  0.11–0.33); the q4 cell was **not** run on the anchor (`w19.sh:179-189` runs `--methods composite`
  only; the paper says so honestly, `main.tex:1111-1112`). So the rule's second conjunct is
  untested, on a generator the paper itself says flatters the parent config (`main.tex:899-902`,
  the flagship's mv edge does not transfer). The in-repo half of the fork is a clean, paired win
  (§3.3 below); the anchored half is missing. 7 needs the ~$10 cell-on-anchor run — which is **not**
  among the three in-flight experiments.
- **Prior-work 6.** Tracker-swap in flight. Independently of it, the manuscript contains a
  novelty contradiction a reviewer will find in ten minutes (§3.4): §oracle says the tracker
  "beats fixed-rank incremental SVD everywhere" (`main.tex:439-440`), while the repo's own
  commit for the swap (`7117a32`) and `scripts/pod/w19.sh:195-196` state that the flagship's
  tracker step, at its defaults `theta=None, min_sv_frac=0`, **is** fixed-rank incremental SVD.
  Both are true — the Week-2 "incremental SVD" baseline differs only in *reconstruction
  convention* (§3.4) — but the paper never says so, and the deployed cache uses the baseline's
  convention, not the near-oracle one. $0 to fix; must be fixed before any tracker-swap result
  is interpretable.
- **Systems 6.** Latency/peak in flight; the abstract still carries the analytic "$\approx1.06\times$"
  (`main.tex:105`) and the 327-token 1B-CPU "+10%" datum (`:1011-1012`, `:1063`).

Claims 7, rigor 7, repro 7 hold (repro is now firmer than at the prior verdict: CI green at
HEAD, W18 ppl line-files and `results/w19-env-provenance.txt` committed).

---

## 2. Per-dimension table

| dimension | post-$0 (prior AC) | NOW (fork folded, verified in text) | what the in-flight / missing experiment changes |
|---|---|---|---|
| claims | 7 | **7** — every headline direction traces (§4); fold introduces two overstatements (`:695`/`:709` "at or below 0.20" for a 0.33 cell; `:690` "none reaches" at Llama 32K where the q4-composite is 2/0 behind, p=0.5) and the 16K "3.4–10×/6.7–13×" arithmetic residue (`:73`, `:458-459`, `:1146`) persists — all $0 | **ss2 control**: if single-shot 2-bit mv stays ≲0.6 on Llama 16K → 7 firm. If single-shot 2-bit mv ≈ 0.9–1.0 → the C1 headline "wins multi-value on all three families" is protocol-bound and must be re-scoped to "under chunked streaming prefill" in abstract `:87-89`, intro `:177-179`, §quantbaseline `:774-776`, §limits `:1090-1091`, conclusion `:1153-1154` → 6 until re-worded, 7 after ($0) |
| prior-work | 6 | **6** — OjaKV/Brand/FD identity stated (`:232-245`); new: MiniKV named uncited (`:689`); the iSVD contradiction (§3.4) | **tracker-swap**: → **7** if Oja *or* FD loses to the flagship on retrieval or ppl at matched cache (any task ≥6/0 paired, or ppl ≥0.05 bits/tok) *and* the paper states the iSVD identity/convention (§3.4). Stays **6** if both match (then "DLRA" is branding: retitle, re-frame the contribution as the cache design + fair comparison + band; novelty vs OjaKV's hybrid store and MomentKV's residual score is then thin) |
| rigor | 7 | **7** — inferential framing honest; fold reports composite contrasts *without* the paired test the paper uses everywhere else, though the per-trial records exist (`results/w19_pertrial/fork-llama-trials.txt`, `a1q-llama-trials.txt`; recomputed in §3.3) | **ss2**: either outcome keeps 7 provided it is disclosed; the favourable outcome (edge survives) makes it an 8-class rigor story with the pooled tests. Also $0: add the pooled official contrast (8 vs 16 discordant over 108 records, p=0.15) beside "not separated on any task" (`:894`, `:1107`) |
| significance | 6 | **6** (high) — in-repo exclusivity now *measured*: byte-matched `ea-k0.25-q2` loses mk/mv 11/0 at 16K (p=0.001), mv 8/0 at 32K (p=0.008); Mistral loses at every composite byte level incl. 2× bytes (§3.3). Un-anchored: the cell has no official number | **cell on the official anchor (~$10, not in flight)**: → **7** if `bugSseed-r64-h256-q4` holds single/mk/mv on official RULER at 16K (mean over s1–s3/mk1–mk3/mv/mq ≥ ~0.7 where composites score 0.11–0.33). Stays **6** if it collapses like the composites — then delete "exclusive"/"neither" and write the band as an in-repo observation |
| systems | 6 | **6** — storage-vs-resident framing clean everywhere; 1/T rewritten correctly; residual = analytic 1.06× and the 327-token datum; batch-1 disclosed (`:1065-1067`) | **latency/peak**: → **7** when the measured ms/token (p50 + absorb-rebuild spikes) and KV-attributable peak at 16/32/64K replace `:105-106`, `:1001-1004`, `:1010-1012`, `:1061-1063` — *any* value, honestly stated. If decode ≥1.5× full at 64K, the abstract's "$\approx1.06\times$" sentence must become "decode is X× slower and peak Y× without a kernel"; systems still 7, but expect a significance reviewer to dock the persistence framing |
| repro | 7 | **7** (firmer) — CI green at HEAD (`gh run list`: 34287237088, 34286925102 success); W18 ppl line-files committed (`results/w18-*-ppl-lines.txt`); `results/w19-env-provenance.txt` committed. New nits: fork pods `@5d3239f` absent from the provenance paragraph (`:1127-1131`) and from `w19-env-provenance.txt`; `tab:composite` is the only table whose generator reads gitignored raws (`scripts/w19_fork_report.py:4-5,38`), and the committed `results/w19-fork-llama-lines.txt` carries unlabeled in-repo/official duplicates (`:25-32`, `:57-64`); the per-trial file has 16 colliding official/in-repo vt keys (§3.3 note) | none in flight; $0 to close the nits |

**Gate (≥7 on all six, zero fatal): not met — 3 of 6 at 6; zero fatal.** Prior verdict had the same
three below 7; significance moved from "asserted" to "measured in-repo, un-anchored", which is
progress but not the pre-registered 7.

---

## 3. Content review of the manuscript as written

### 3.1 (a) Is the abstract the best honest statement of the contribution? One notch off — in length and in one clause, not in honesty.

- **Length.** 642 words / 4,812 characters (`main.tex:53-113`). arXiv's metadata abstract limit is
  1,920 characters, so a short abstract has to be written anyway; ICML reviewers will read 640
  words with ~30 numbers as "the authors could not decide what the contribution is". The
  honesty content is right; its *placement* is wrong. Keep in the abstract: the mechanism (one
  sentence), the fair-comparison verdict (one sentence: ties 2-bit at ≈2 bits/element on
  single-needle and on official RULER; multi-value edge on our generator only; dominated by
  4-bit where its bytes reach 4 bits), the band (one sentence, with "neither scalar quantization
  nor eviction×quantization at matched bytes on our generator; the cell has no official number"),
  the map (one sentence), the systems scope (one sentence). Move the per-family p-values, the
  Mistral-vt-0.25-vs-0.50 parenthetical (`:92-94`), the Qwen 35.1 clause (`:91`), and the 64K
  arithmetic (`:109-113`) to the body — they are already there.
- **The new composite clause is garbled.** `:98-101`: "reaches $0.048\times$ at 16K and
  $0.034\times$ at 32K, below any fixed-bit scalar-quantizer floor and below the retrieval any
  eviction$\times$quantization composite reaches at matched bytes" — "below the retrieval" hangs
  off "reaches 0.048×". Rewrite: "...below any fixed-bit scalar-quantizer floor, with
  single/multi-key/multi-value retrieval that no eviction×quantization composite matches at
  ≤ its bytes on our generator (byte-matched composites collapse on official RULER; the cell
  itself is unmeasured there)".
- **"near-oracle" (`:62-64`) describes the basis, not the deployed reconstruction** — see §3.4.
  Add "as a subspace tracker".
- **"$3.4$--$10\times$ float-equivalent" (`:73`)** is 32K arithmetic in a 16K sentence:
  0.75/0.085 = 8.8×, 0.50/0.149 = 3.4×; the 10× needs the 32K 0.075× (`results/w18-llama-lines.txt:4`).
  Same at `:458-459` ("$6.7$--$13\times$": 1/0.085 = 11.8×) and `:1146`. Say "3.4–8.8×" and
  "6.7–12×", or cite 32K explicitly. (Flagged by the claims review as #3; not fixed.)
- Nits: "$19\%$ under" (`:111`, `:969`, `:977`): 1 − 0.125/0.156 = 19.9% → "20%". "4-bit ... trails
  only on Mistral variable-tracking" (`:93`): it also trails on Qwen 16K multi-value on point
  estimate (0.92 vs 1.00, `results/w19-a1-report.md:109,111`, not separated); say "only on point
  estimate, on Mistral vt (0.25 vs 0.50) and Qwen 16K mv (0.92 vs 1.00)".

### 3.2 (b) Structure and flow — stale text is gone; four linear-reader snags remain.

Verified gone (grep count 0): "v2"/"primary v2 experiment", "v1's main scoping decision", the
dangling "Three" fragment, "indistinguishable", "not floored", "keeps falling", "independent of
$T$", "constant in context length", "concedes nothing", "3.2–4.7×". The header comment
(`:1-25`) is current. Related work no longer contradicts §quantbaseline.

Remaining snags a linear reader hits:
1. **§oracle vs Method vs the swap pre-registration** — `:439-440` "beats fixed-rank incremental
   SVD everywhere" after `:232-235` "reduces to the classical incremental-SVD update" and the
   flagship defaults that make it exactly that. Contradiction until the reconstruction
   convention is named (§3.4).
2. **Intro bullet one notch behind the abstract.** `:176-183` still says the cell is "below any
   fixed-bit scalar-quantizer floor" only; abstract (`:100-101`), §subcliff (`:697-698`), §limits
   (`:1112-1114`) and conclusion (`:1156-1159`) say "neither scalar quantization nor
   eviction×quantization". Add the clause to the bullet.
3. **Self-contradicting sentence in the fold.** `:694-695` "every composite at $\le0.07\times$
   collapses to mean $0.11$--$0.33$, at or below plain eviction's $0.20$" — 0.33 > 0.20; the
   caption repeats it (`:708-709`). Correct: the byte-matched composites (≤0.039×) score
   0.11–0.17, below plain eviction; `ea-k0.25-q4` at 0.070× (1.5–2× the cell's bytes) reaches
   0.33 — still collapsed, but above eviction (`results/w20-fork-report.md:57-61`).
4. **Setup's lower-$n$ sentence is stale.** `:425-427` "some 32K composition cells and the 1B study
   are lower-$n$" — the 32K compose cells are now $n{=}12$ (`results/w19-a1q-llama-lines.txt:4,6,8,10`);
   the lower-$n$ cells are the 64K flagship ($n{=}8$, `w19-a4-llama-lines.txt:6`) and the 8-bit
   control ($n{=}4$, `w19-a1-llama-lines.txt:7`). (Claims review 3.10; not fixed.)
5. "Two lines of evidence" (`:864`) introduces three numbered items (`:865`, `:874`, `:880`).
6. "$\sim$40 descriptive Wilson cells" (`:543`) — the tables now carry ≈180 (rigor review 1.1).

### 3.3 (c) Overclaims and under-claims, with the paired numbers the paper should print

The fork paragraph (`:685-700`) reports point estimates only. The paired contrasts from the
committed records (cell = `results/w19_pertrial/a1q-{llama,mistral}-trials.txt`; composites =
`results/w19_pertrial/fork-{llama,mistral}-trials.txt`; keys `(task,ctx,seed,trial)`, zero
duplicates; exact two-sided McNemar) are:

| family | ctx | composite (stored) | single | multi-key | multi-value | var-track |
|---|---|---|---|---|---|---|
| Llama | 16K | `ea-k0.25-q2` (0.039×, byte-match) | 12 vs 11, 1/0 | 12 vs 1, **11/0 p=0.001** | 12 vs 1, **11/0 p=0.001** | 6 vs 0, **6/0 p=0.031** |
| Llama | 16K | `ea-k0.1-q4` (0.028×, 58% of bytes) | tie | 12 vs 9, 3/0 p=0.25 | 12 vs 11, 1/0 p=1 | 6 vs 5 (pairing not recoverable — see note) |
| Llama | 32K | `ea-k0.25-q2` (0.039× vs cell 0.034×) | tie | 12 vs 10, 2/0 p=0.5 | 12 vs 4, **8/0 p=0.008** | 10 vs 4, **6/0 p=0.031** |
| Llama | 32K | `ea-k0.1-q4` (0.028×, 82% of bytes) | tie | 12 vs 10, 2/0 p=0.5 | 12 vs 10, 2/0 p=0.5 | 10 vs 7, 3/0 p=0.25 |
| Llama | 32K | `ea-k0.25-q4` (0.070×, 2× bytes) | tie | tie | tie | 10 vs 4, 7/1 p=0.07 |
| Mistral | 16K | `ea-k0.25-q4` (0.070×, 1.5× bytes) | 12 vs 10, 2/0 | 12 vs 4, **8/0 p=0.008** | 11 vs 5, 7/1 p=0.07 | 4 vs 1 |
| Mistral | 16K/32K | every composite ≤0.039× | **≥8/0 p≤0.008** (12/0 for the q2 arms) | **≥11/0 p≤0.001** | **≥10/0 p≤0.002** | 32K: **7/0 p=0.016** |
| Mistral | 32K | `ea-k0.25-q4` (0.070×) | 12 vs 7, 5/0 p=0.06 | 12 vs 1, **11/0 p=0.001** | 10 vs 5, 6/1 p=0.125 | 7 vs 0, **7/0 p=0.016** |

Note on the 16K vt column: the committed `results/w19_pertrial/fork-llama-trials.txt` keys the
official vt records as `seed=0 trial=0..11`, which collide with the in-repo keys `seed=0
trial=0..5` (16 duplicate `(task,ctx,arm,seed,trial)` keys, all `vt ctx=16384`; e.g. the two
`trial=0` rows for `ea-k0.25-q4-kivi` carry `hit=0` and `hit=1`). The 16K vt contrasts against
the two 4-bit composites, and the official vt cells of `tab:composite`, therefore cannot be
recomputed from the committed records (the generator reads the gitignored raws' section
markers instead, `scripts/w19_fork_report.py:4-5`). The contrasts against the two 2-bit
composites are safe (0/12 in-repo hits either way). Repro fix: prefix official task names
(e.g. `off:vt`) in the emitted `[trial]` rows.

**Overclaims (all $0):**
- `:690` "At matched bytes none reaches the cell's retrieval" — true and decisive at Llama 16K
  and on Mistral; at Llama 32K the byte-matched q2 composite ties multi-key (2/0) and the
  cheaper `ea-k0.1-q4` (0.028×, fewer bytes than the cell's 0.034×) is within 2/0 on *every*
  task. By the paper's own rule (`:780-782`, "nulls bound rather than establish equivalence")
  that is a point-estimate lead, not exclusivity. Write: "on Llama the 2-bit byte-matched
  composite loses multi-key/multi-value decisively at 16K (11/0) and multi-value at 32K (8/0);
  a 4-bit composite at 0.028× trails the cell by 1–3 paired misses per task, not separated at
  $n{=}12$; on Mistral every composite up to 2× the cell's bytes loses single/multi-key/
  multi-value (p ≤ 0.008)".
- `:694-695`, `:708-709` — the 0.33 "at or below 0.20" error (§3.2.3).
- `:439-440` "beats fixed-rank incremental SVD everywhere" — a reconstruction-convention
  artifact (§3.4).
- `:889` "The $0.29$--$0.75\times$ arms (4-bit KIVI, ThinK, Palu) are near-perfect" and `:899`
  "misses the near-lossless arms (4-bit KIVI, ThinK, Palu) do not make" — Palu misses 9
  official records incl. multi-value 0.42 (`results/w19-a2-flagship-misses.md:35`;
  `results/w19-a2-llama-lines.txt:32`; `tab:official` row `:920`). Drop Palu from both
  parentheticals (4-bit KIVI and ThinK each miss exactly the one record every arm misses,
  `misses.md:9,32,34,37`).
- `:683` "SnapKV ... multi-key $0.17$--$0.42$" is Llama-only (`results/w18-g3-llama-lines.txt:3,6`);
  Qwen/Mistral are 0.00 (`w18-g3-qwen-lines.txt:3,6`, `w18-g3-mistral-lines.txt:3,6`) → "0.00–0.42".

**Under-claims (the paper is entitled to more):**
- **Mistral is the composite's worst case and is absent from `tab:composite` and the fork
  paragraph.** On Mistral no composite matches the cell at *any* byte level, including
  `ea-k0.25-q4` at 2× the bytes (mk 0.33/0.08 vs 1.00/1.00; `results/w19-fork-mistral-lines.txt`
  rows above; `w20-fork-report.md:25-34`). That is the cleanest exclusivity evidence the paper
  has, and it is on the family where the cell's fluency cost is smallest (6.56/4.28 vs 5.50/3.76,
  `:670-671`). Add a Mistral block to the table and lead the band with it.
- 16K multi-key is 12/12 on all three families (`results/w18-{llama,qwen,mistral}-lines.txt:2`)
  but `tab:xmodel` (`:476`) has no multi-key column.
- `tab:composite` prints "---" for `ea-k0.1` in-repo although `tab:evict` has those cells
  (1.00/0.92/1.00/0.08 and 1.00/0.92/0.92/0.58); print them, they show plain eviction beats
  its own quantized composites on mk/mv at 16K (0.92/1.00 vs 0.08/0.08 for `ea-k0.25-q2`), which
  is itself a finding: quantizing survivors to 2 bits *destroys* eviction's Llama retrieval.

### 3.4 (d) Novelty statement vs OjaKV / incremental SVD — one sentence is missing and one is wrong

What is now right: OjaKV credited as opening the axis (`:57-58`, `:191-192`, `:364-368`); the
Brand/FD identity paragraph (`:232-245`); "measurement, not a theorem" (`:240-242`); MomentKV's
residual-norm score acknowledged (`:404-406`).

What is missing — and it is load-bearing for interpreting the tracker swap:
1. **The flagship runs with no rank adaptivity.** `scripts/w10_frontier.py:198` (`min_sv_frac`
   default 0.0), `src/kvdlra/integrators/streaming_torch.py:88-89` (`theta=None`,
   `min_sv_frac=0.0`), `:192-200` (both criteria skipped at defaults) → the flagship's step is
   range-augment, re-SVD, truncate to $r$: blocked fixed-rank incremental SVD, exactly as
   `7117a32` and `w19.sh:195-196` say. The Method paragraph says DLRA "adds the rank-adaptive
   truncation criterion ... and the $\sigma_{\min}$-independent error bound" (`:237-240`) without
   saying that neither is exercised by the flagship; the rank-adaptive machinery appears only in
   the floored cells of `tab:floor`. Add: "The flagship uses $\theta{=}\text{None}$ and
   `min_sv_frac`$=0$, so its tracker step is the blocked fixed-rank incremental-SVD update; the
   rank-adaptive criterion is exercised only in the floored cells (Table~\ref{tab:floor})."
2. **"beats fixed-rank incremental SVD everywhere" (`:439-440`) is a reconstruction-convention
   difference, and the cache uses the baseline's convention.** The Week-2 baseline
   (`src/kvdlra/lowrank.py:30-63`) performs the *same* append/re-SVD/truncate update and differs
   only in reconstructing from the tracked right factors (`(u*s)@vt`, `:62`) — old columns carry
   every truncation they lived through — whereas the BUG number re-projects the raw matrix onto
   the *final* basis (`src/kvdlra/integrators/streaming.py:22-23`, `:285-328`,
   `reconstruction_error(m)` in `scripts/week2_pilot.py:76-80`). The 4–8% gap
   (`docs/week2-pilot.md:36`) is that convention. The deployed cache cannot re-project (the raw
   middle is gone): it stores coordinates and rotates them through every truncation,
   `C <- rot @ C` (`src/kvdlra/cache/bug_cache.py:39`, `:917-920`), and reconstructs `u @ c`
   (`:1315`, `:1603-1618`) — the incremental-SVD convention. So "near-oracle" is a property of
   the basis $U$ (true, and the 1.3–3× gap to Oja is a basis gap that survives); the cache's
   per-token reconstruction sits on the "naive" curve the paper says it beats. Fix ($0): scope
   "near-oracle" to the subspace, delete or re-scope "beats fixed-rank incremental SVD", and
   state the convention. This also means the swap's `isvd` arm *is* the flagship by construction
   — the paper must not later present "BUG vs iSVD" as an end-to-end contrast.
3. **MiniKV** (`:689`, "the MiniKV pattern") is named without a citation and is not in
   `refs.bib` — it is precisely the eviction×2-bit composite the fork instantiates (Sharma et
   al., 2024, arXiv:2411.18077 — verify id). Must cite.
4. Concurrency (`:186-199`, `:396-407`): "independent contemporaneous work, not evaluated
   baselines" is the right wording for arXiv v1; for ICML 2027 (deadline ≈ Jan 2027) June/July
   2026 papers are not concurrent, and a reviewer will ask for a MomentKV/ResKV comparison if
   code exists (§3.7).

### 3.5 (e) Table / figure hygiene

- **`tab:composite` (`:703-723`)**: (i) Llama-only — add Mistral (§3.3); (ii) no Wilson or
  McNemar although every other retrieval table has one — add the discordant counts above;
  (iii) "off.\ mean" is not defined in the caption as the official-RULER 9-task mean; (iv) caption
  "at or below plain eviction" is false for the 0.33 row; (v) the cell's "off." entry "---" should
  read "not run" to match `:1111-1112`; (vi) plain `ea-k0.1` in-repo cells left blank though
  measured (`tab:evict`).
- **`fig:one_over_t`** (`figures/week19/one_over_t.png`): title and dotted asymptote agree with
  the caption and with §memory (`:967-970`); but the caption cites "fit $0.127+380/T$" and no fit
  curve is drawn, and the "2r/n = 0.125x (fp32-coord asymptote)" annotation is drawn over the
  BUG series at 16K–32K (partly illegible). Draw the fit as a thin dashed line; move the label
  below the asymptote.
- **`fig:coldstart`**: the figure's suptitle still says "disk read + H2D + reconstruct"
  (`scripts/w19_figures.py:259`; visible in `coldstart.png`) while the caption (`:1046-1047`) and
  text (`:1028`) now say "warm page-cache read". Regenerate the figure.
- **`tab:xmodel` (`:467-483`)**: stored-state column is fp16-equivalent while the header
  invariant (`:14-15`) says honest bytes lead every headline; add an honest column (0.150/0.150/
  0.275) or state both in the caption; add multi-key.
- **`tab:marquee`**, **`tab:official`**, **`tab:fairquant`**, **`tab:floor`**, **`tab:evict`**: verified
  against their sources (§4); no issues. `tab:official` Palu mean 0.92 with mv 0.42 is what makes
  `:889`/`:899` wrong.
- Label hygiene: `tab:composite` is referenced from §quantbaseline, §limits and the conclusion;
  all resolve. `\S2` in "TurboQuant~\S2" (`:646`) is a bare section reference into the cited
  paper — write "TurboQuant's rotated Lloyd–Max stage (their §2)".

### 3.6 (f) Prose

- The abstract's fair-comparison sentence (`:85-97`) is one 110-word sentence with five
  semicolon-clauses; split at "ties single-needle everywhere" and at "on the official NVIDIA
  RULER suite".
- `:685-700` reads well except the garbled matched-bytes clause (§3.2.3).
- Sentences a reviewer will read as evasive: `:889` "near-perfect" and `:899` "near-lossless
  arms ... do not make" (Palu, §3.3); `:976-978` "a $19\%$ margin, not an unbounded one" is fine
  but "$19\%$" should be 20%; `:1033-1034` "(the small artifact loads at a higher per-byte rate
  than the multi-GB full cache)" explains a 9–10× reload from a 6.6–7.2× byte ratio — say the
  per-byte rates (1.9–2.0 vs 4.9–5.8 GB/s, from `results/w19-a3-llama2-lines.txt:1-5`) so it is
  a measurement, not an excuse.
- Hedging that has become unreadable: `:92-97` (three concessions and one win in one clause);
  move to the body.

### 3.7 (g) What a reviewer will raise that the three in-flight experiments do NOT cover

| # | objection | cost | notes |
|---|---|---|---|
| 1 | **The q4 cell has no official-anchor number** (`:1111-1112`); the band's external validity is the significance gate | **~$10** (one a2-style pod, one arm, 9 tasks × 12 records) | the single item that decides 6 vs 7 |
| 2 | KIVI arm at $G{=}64$ vs paper-KIVI $G{=}32$ (`:755-757`); no reproduced published KIVI number | ~$10 | prior-work X5, still open; disclosed |
| 3 | The $r/n\approx0.25$ wall's control (`:598-601`) scores 0.00 on single-needle too and has no ppl line — a diverged model also scores 0.00; the abstract asserts the wall as mechanism (`:81-82`) | ~$10 (floored `bugSseed-r256-h1024` Llama + Qwen r128 with ppl) | claims review 3.6, still open |
| 4 | Every quantized arm clusters at 0.25–0.33 on Mistral vt incl. the 8-bit control (`results/w19-a1-report.md:60-62`) — task degeneracy or harness defect? | ~$5 (full-KV Mistral vt 16K) | affects the Mistral vt column of `tab:fairquant` |
| 5 | MomentKV / ResKV comparison for an ICML submission (§3.4.4) | ~$20–50 + harness days, if code exists | not needed for arXiv v1 |
| 6 | Official RULER at 32K / Qwen / Mistral (`:1109-1110`) | ~$30–50 | disclosed as open |
| 7 | fp16-storable gist (`:1098-1099`) — the paper's own "most valuable follow-up"; would halve honest bytes and change the Qwen verdict | ~$50 + days | |
| 8 | fp32 re-score of sub-0.03-bit fluency deltas (`:581-584`) | ~$15 | disclosed; optional |
| 9 | Depth-stratified official single-needle (misses front-loaded, `:896-898`) | ~$10 | |
| 10 | Persistence claim not positioned against the persistence literature (CacheGen as codec, AttentionStore, CacheBlend, Prompt Cache) — `grep` finds CacheGen only at `:128`, `:348` | $0 | one sentence in `:1017-1041` |
| 11 | Compute appendix (`:1134-1136` admits none); fork pods' $3.40 and SHA `5d3239f` absent from provenance | $0 | |
| 12 | Short arXiv abstract (≤1,920 chars) | $0 | forced anyway |
| 13 | Fused kernel; batch >1 (`:1060-1067`) | weeks | disclosed; not required |

---

## 4. Verification ledger (abstract / intro / conclusion → result file:line)

✓ = traces and agrees · ~ = traces, imprecise · ✗ = does not trace or arithmetic wrong.

| # | paper line | claim | evidence | status |
|---|---|---|---|---|
| 1 | abstract `:62-64`, §oracle `:437-441` | near-oracle 1.01–1.03× of truncated SVD; Oja 1.3–3× worse | `docs/week2-pilot.md:26-28` (ratios 1.009–1.025), `:66-67` (Oja 1.3–3.0×) — 1B, layer 8, 5 docs; no `results/` file | ~ carried; abstract omits "1B" and "as a subspace tracker" (§3.4) |
| 2 | abstract `:67-70`, intro `:162-164` | 12/12 single+mv on three families, Wilson ≥0.758, 0.15×/0.28× honest (0.085–0.149× fp16-eq) | `results/w18-llama-lines.txt:6,10` (1.00, sbits 0.151/0.150, ratio 0.085); `w18-mistral-lines.txt:6,10`; `w18-qwen-lines.txt:6,10` (sbits 0.275/0.276, ratio 0.149) | ✓ |
| 3 | abstract `:71-74`, §xmodel `:458-459`, concl `:1146` | 1.8–5× less than think/palu at honest bytes; **3.4–10×** fp16-eq; 3.6–6.7× less than full | honest: 0.75/0.28 = 2.7, 0.75/0.15 = 5.0, 0.50/0.28 = 1.8 ✓; full 1/0.28 = 3.6, 1/0.15 = 6.7 ✓; fp16-eq at 16K: 0.50/0.149 = 3.4, **0.75/0.085 = 8.8**, full **1/0.085 = 11.8**; the 10×/13× need 32K 0.075× (`w18-llama-lines.txt:4`) | ✗ upper bounds (3 places) |
| 4 | abstract `:74-77`, §marquee `:526-528` | marquee vt 0.94 vs think 0.31, 10/0, p=2.0e-3, n=16; palu 0.56, 6/0, p=0.03 | `results/w18-g4-llama-lines.txt:13,15,16`; `results/w18-g4-marquee-contrasts.json:6-7` | ✓ |
| 5 | abstract `:78-80`, `tab:marquee` `:566,569` | marquee 0.284× honest; 4-bit KIVI vt 1.00 at 0.284× | `w18-g4-llama-lines.txt:13` (sbits 0.284); `results/w19-a1-llama-lines.txt:4,24` (0.284×, vt 1.00) | ✓ |
| 6 | abstract `:85-89` | flagship 0.15 vs 2-bit 0.16×; Qwen 0.275× (1.7×); mv pooled 21 vs 0; per-family p=0.008–0.031 | `results/w19-a1-report.md:12-13` (0.151/0.163), `:109` (0.275), `:30` (7/0 p=.0156), `:78` (6/0 p=.0312), `:127` (8/0 p=.0078) → 21 vs 0 | ✓ |
| 7 | abstract `:89-90`, concl `:1154` | 32K multi-key on Mistral, vt on Qwen | `w19-a1-report.md:81` (mk 9/0 p=.0039), `:132` (vt 9/0 p=.0039) | ✓ |
| 8 | abstract `:91` | Qwen 32K ppl 35.1 vs 8.2 | `results/w18-qwen-ppl-lines.txt:2` (35.083); `w19-a1-report.md:140` (8.23) | ✓ |
| 9 | abstract `:92-94` | 4-bit trails only on Mistral vt 0.25 vs 0.50 | `w19-a1-report.md:59,61`; also Qwen 16K mv 0.92 vs 1.00 (`:109,111`) | ~ |
| 10 | abstract `:95-97`, `tab:official` `:922-924` | official: flagship 0.79 vs 2-bit 0.87, no task separated; eviction collapses | `results/w19-a2-llama-lines.txt` flagship rows 1,8,15,22,29,36,43,50,57 → 7.15/9 = 0.794; 2-bit rows 5,12,19,26,33,40,47,54,61 → 0.870; ea rows → 0.203; `results/w19_intervals/a2-llama-ruler-intervals.md:22-30` (no sig) | ✓ |
| 11 | abstract `:99`, §subcliff `:648-649`, `:662` | cell 0.048×/0.034× | `results/w19-a1q-llama-lines.txt:1-2` (sbits 0.048/0.034) | ✓ |
| 12 | abstract `:101-103`, §subcliff `:655-656`, `:662-663` | Llama 1/1/1/0.50 & 1/1/1/0.83; Mistral 1/1/0.92/0.33 & 1/1/0.83/0.58; ppl 9.25/17.1 vs 5.31/8.33 | `w19-a1q-llama-lines.txt:3-10`, `:1-2` (9.254/17.074); `w19-a1q-mistral-lines.txt:3-10`; `results/w18-llama-ppl-lines.txt:1-2` (5.308/8.326) | ✓ |
| 13 | abstract `:100-101`, §subcliff `:690-693`, `tab:composite` | byte-matched `ea-k0.25-q2` mk/mv 0.08/0.08 at 16K; `ea-k0.25-q4` matches at 0.070× | `results/w19-fork-llama-lines.txt:3,30` (0.08/0.08), `:4,32` (1.00/1.00 @0.070); `results/w20-fork-report.md:13-14` | ✓ point estimates; strength at 32K ✗ (§3.3) |
| 14 | §subcliff `:694-695`, `tab:composite` | official composites mean 0.11–0.33, "at or below 0.20" | `w20-fork-report.md:57-61` (0.11/0.19/0.17/0.33; plain 0.20); recomputed 0.33 from `w19-fork-llama-lines.txt` official rows | ✗ "at or below" for 0.33 |
| 15 | abstract `:104-106`, §memory `:1000-1004`, limits `:1061-1062` | decode residency ≈1.06× | analytic: `results/w18-g5-llama-lines.txt:2` workspace 0.982 + stored 0.084 = 1.066; labelled analytic at `:1001-1004` but not in the abstract | ~ in flight |
| 16 | abstract `:107-108`, §memory `:1030-1033` | ≈7× smaller persisted cache; 9–10× faster warm reload; 0.13 s vs 1.2 s | `results/w19-a3-llama2-lines.txt:1-2` (325.1 MB vs 2147.5 MB = 6.6×; 0.134 vs 1.226 s = 9.15×), `:4-5` (7.2×; 10.4×) | ✓ |
| 17 | abstract `:109-113`, §memory `:967-975` | 0.151→0.140→0.133; 64K all four 1.00 (n=8); 2-bit 1.00/0.58/0.50/1.00; ppl 8.59/8.15/7.27/7.26 | `results/w19-a4-llama-lines.txt:1` (sbits 0.133, ppl 8.591), `:6,10,14,18` (1.00 n=8), `:8,12,16,20`, `:2-5`; `w19-a3-llama2-lines.txt:1,4` (0.1514/0.1397) | ✓ (19% → 19.9%) |
| 18 | abstract `:80-84`, §siphon `:598-601`, `tab:floor` | r256 control 0.00 on all four at n=12; floor table | `w18-g4-llama-lines.txt:2,6,10,14`; `docs/week17-explained.md:79-84` | ✓ (control has no ppl line — §3.7 #3) |
| 19 | §marquee `:577-580` | same-pod ppl 6.975/7.196/7.232/7.353; +0.031/+0.024 bits | `results/w18-g4-llama-ppl-lines.txt:1-4`; log2(7.353/7.196)=0.031, log2(7.353/7.232)=0.024 | ✓ (now committed; order fixed) |
| 20 | §realistic `:866-871` | WikiText filler: flagship single 1.00; ea 0/0/0/0; palu 1/1/0.83/0.08; think 1/1/0.67/0.33 | `results/w18-g2-qwen-lines.txt:7` (flagship single only), `:1,4,8,11`, `:2,5,9,12`, `:3,6,10,13` | ✓ (single-needle wording now literal) |
| 21 | §realistic `:875-879` | LongBench Qasper F1 0.221@0.160; ea-k0.25 0.216; ea-k0.1 0.149; snapkv 0.136; palu-r0.25 0.073; full 0.259 | `results/w11-goalA-lb-lines.txt:3,8,7,18,14,10` | ✓ |
| 22 | `tab:evict` `:738-741`, §subcliff `:683` | eviction grid; SnapKV mk 0.17–0.42 | `results/w18-g3-llama-lines.txt:2,8,14,20,5,11,17,23`; `w18-g3-qwen-lines.txt:14,2,8,20`; `w18-g3-mistral-lines.txt:14,2,8,20`; SnapKV `w18-g3-llama-lines.txt:3,6` (Qwen/Mistral 0.00) | ✓ grid; ~ SnapKV range Llama-only |
| 23 | §realistic `:896-898` | 22 misses; 14 needle at 0.15–0.95, six ≤0.20; 8 vt | `results/w19-a2-flagship-misses.md:7-28` (22 rows), `:38` | ✓ |
| 24 | limits `:1127-1131` | W19 SHAs 24ac22a/6734afa/1cbc31f/5e8b275/c331ebd/d15769d | `results/w19-env-provenance.txt` (RUN_SHA blocks) | ✓; fork `5d3239f` missing from both |

Untraceable to any committed result file: none of the headline numbers; #1 is carried from a
docs file (acceptable if labelled "1B, layer 8"). Arithmetic errors: #3 (three places), #14.

---

## 5. The sentence the author can say today

> An online, training-free low-rank KV cache — a blocked incremental-SVD/BUG gist tracker
> plus a surprise-selected exact tier and warm-up seed — matches a fair KIVI-scheme 2-bit
> quantizer at ≈2 bits per element on single-needle retrieval and on the official RULER suite,
> keeps multi-value/multi-key retrieval that 2-bit quantization loses on our generator, and,
> composed with 4-bit coordinates, is the only method we measured — against scalar quantization
> and eviction×quantization composites at matched bytes — that retains single/multi-key/
> multi-value retrieval at 0.03–0.05× stored state on Llama and Mistral (at ~2× perplexity, on
> our generator; unmeasured on the official suite), together with a mechanistic map of where
> the representation breaks.

If the tracker swap shows Oja/FD interchangeable, replace "incremental-SVD/BUG gist tracker"
with "a low-rank gist (any incremental-SVD-class tracker)" and drop "DLRA" from the title. If the
cell holds on the official anchor, delete "on our generator; unmeasured on the official suite"
and the sentence is an accept-level claim.

---

## 6. Finalization rule for the in-flight experiments (apply on landing; no re-review needed)

Committed references the reports pair against: flagship Llama 16K
`results/w18-llama-lines.txt:2,6,10,14` (1.00/1.00/1.00/0.58) and `w18-llama-ppl-lines.txt:1`
(5.308); chunked 2-bit `results/w19-a1-llama-lines.txt:5,10,15,20` (0.67/0.42/1.00/0.67) and `:1`
(5.403); analytic residency 1.066 (`w18-g5-llama-lines.txt:2`).

1. **Tracker swap (`swap-llama`, MODE `swap`, `w19.sh:202-211`; rule `:199-201`).** Pair
   `bugSseed-r64-h256-{oja,fd}` against the flagship on the a1q needles.
   - Oja *or* FD loses on any task at ≥6/0 one-way discordant (p ≤ 0.031) **or** trails by
     ≥0.05 bits/token (ppl ≥ 5.49 vs 5.31) → tracker load-bearing → **prior-work 7**, provided
     §3.4 items 1–2 are written (the swap's "iSVD" arm is the flagship; say so).
   - Both match within the n=12 detection floor (≤5 discordant, ppl within 0.05 bits) →
     "DLRA" is branding → keep **prior-work 6**; retitle; re-frame the contribution as cache
     design + fair comparison + band; the paper stays an overall 6 regardless of the other two.
   - Mixed (Oja loses, FD matches) → the SVD-class tracker is load-bearing vs Oja's rule, DLRA
     specifically is not → **prior-work 7 only with** the title/novelty re-scoped to
     "incremental-SVD-class tracker with rank-adaptive truncation", else 6.
2. **Latency + peak (`sysfix-llama`, `w20_latency.py` rows `[latency ctx..]`).** Fold p50 ms/token,
   max/spike count (absorb rebuild), resident and KV-attributable peak for full/flagship/KIVI-2bit
   at 16/32/64K into `:105-106`, `:1001-1004`, `:1010-1012`, `:1061-1063`, replacing the analytic
   1.06× and the 1B-CPU datum → **systems 7** at any value. If flagship decode > 1.5× full at 64K,
   the abstract's residency clause becomes "decode is X× slower without a kernel" and the
   persistence paragraph keeps its framing; if the KV-attributable peak > 1.2× full, §memory's
   "workspace ≈ full KV" derivation (`:994-1000`) must be re-derived before folding.
3. **Single-shot 2-bit control (`ss2`, `--chunk 0`, Llama 16K).** Compare to the chunked row.
   - single-shot mv ≤ 0.6 (edge survives) → **rigor 7 firm**; add one sentence at `:757-760`.
   - single-shot mv ≥ 0.9 (edge is protocol-bound) → re-scope C1 to "under chunked streaming
     prefill (protocol-matched to the flagship; KIVI's single-shot operating point retains
     multi-value)" at the five places listed in §2 → claims 6 until re-worded, 7 after;
     significance unchanged (the edge was already non-transferring).
4. **Not in flight — the gate item.** `bugSseed-r64-h256-q4` on the official anchor (a2 protocol,
   `w19.sh:86-116` with the compose flags of MODE `a1q`), ~$10: holds single/mk/mv (mean ≥ ~0.7
   where composites score 0.11–0.33) → **significance 7**; collapses → significance 6, delete
   "exclusive"/"neither" everywhere (abstract `:100-101`, `:660`, `:697-698`, `:807-808`,
   `:1112-1114`, `:1156-1159`).

**Overall = 7 (accept) iff** prior-work 7 (rule 1, favourable) **and** systems 7 (rule 2, folded)
**and** significance 7 (rule 4, favourable) **and** the $0 pass in §3 (the two fold overstatements,
the iSVD convention sentence, the 3.4–10× arithmetic, MiniKV, Palu "near-lossless", figure
regeneration). Any one unfavourable → an honest, well-bounded **6 / poster**, which is where the
paper is today.

---

## 7. Top three content fixes (all $0, in order)

1. **State the incremental-SVD identity *and* the reconstruction convention** (§3.4, items 1–2):
   the flagship's tracker step is fixed-rank blocked incremental SVD; "near-oracle" is a basis
   property; the cache reconstructs from rotated coordinates (the baseline's convention); delete
   "beats fixed-rank incremental SVD everywhere". Without this the tracker swap cannot be
   written up coherently, and a reviewer who knows Brand (2006) will call the title branding
   before reading the ablation.
2. **Repair the fold** (§3.2.3, §3.3): fix "at or below plain eviction's 0.20" (0.33 is above);
   re-scope "none reaches the cell's retrieval" to the paired counts (decisive at Llama 16K and
   on Mistral; point-estimate lead at Llama 32K); add Mistral and the discordant counts to
   `tab:composite`; cite MiniKV; un-garble the abstract clause; update the intro bullet.
3. **Arithmetic and the Palu parentheticals**: 3.4–8.8× / 6.7–12× at 16K (`:73`, `:458-459`,
   `:1146`); drop Palu from "near-perfect"/"near-lossless" (`:889`, `:899`); 19% → 20%; "Two
   lines" → three; stale lower-$n$ sentence (`:425-427`); "∼40 cells" (`:543`); regenerate
   `coldstart` ("disk read") and `one_over_t` (fit line, label overlap).

*Read-only. No paper edits, no pods launched, no pushes. Verdict written to this file only.*

---

## Finalization (2026-09-09): the three in-flight experiments landed

All three pods completed (harvested from the logs after a watchdog fetch bug; results
committed at `cdf0395`, folded into the paper at `4d37043`, paper CI green). Applying the
finalization rule above to what they showed:

| dimension | as-worded | post-$0 | **final** | what landed |
|---|---|---|---|---|
| claims | 6 | 7 | **7** | C1 re-scoped: the matched-bytes multi-value edge is a chunked-prefill property (single-shot 2-bit on Llama 16K: 1.00/1.00/0.83/0.67, edge n.s.) |
| prior-work | 5 | 6 | **7** | tracker is load-bearing: Oja's rule 0.92/0.08/0.08/0.00 ppl 733; Frequent Directions no valid cell (SVD fails after shrinkage); iSVD identity stated in §oracle |
| rigor | 6 | 7 | **7** | the single-shot control was run and disclosed |
| significance | 5 | 6 | **6** | **the sub-cliff cell scores 0.00 on all nine official tasks** (flagship 0.79 on the same records): the band is a our-generator property with no external validity; "exclusive" scoped to our generator |
| systems | 6 | 6 | **7** | measured: KV-attributable decode peak 1.6x full KV; decode 4/7/14x slower at 16K/32K/64K; folded honestly (tab:latency) |
| repro | 7 | 7 | **7** | all four runs committed with per-trial records + provenance |

**Final: borderline-accept (6), zero fatal — five dimensions at 7, significance at 6.** The
rule ("7 iff prior-work favourable AND latency folded AND the cell holds on the anchor")
fails on its third clause. This is no longer a wording gap: the 4-bit coordinate codes that
retrieve on the fixed-pool filler do not survive real essay text. The paper now says so in
the abstract, §subcliff, tab:composite, tab:official, §limits and the conclusion.

**What would move significance to 7 (research, not wording):** make the composed cell
survive real text — diagnose the collapse (heavy-tailed coordinates on essays vs the
fixed pool; try 8-bit coordinates, or the singular-value floor on the compose arm, or a
per-family codebook) and re-run q4off (~$10–20 per attempt). Until then the honest venue
statement is "poster: a new online axis with a measured, bounded advantage, whose one
exclusive band is generator-specific."
