# Claims-vs-evidence audit — kvdlra NeurIPS-readiness panel

Reviewer role: claims-vs-evidence auditor. Repo `/Users/hari/Desktop/kv-dlra` @ branch `week7`.
Everything below is grounded in committed result files, scripts, and docs; approximate p-values marked
"reviewer-computed" were derived by hand from the published hit counts.

**Verdict in one line:** the measurement/accounting infrastructure is unusually honest and the interval
arithmetic checks out, but three of the five claims are worded stronger than the files support — one
headline multiplier ("5–13×") does not reproduce from the cited cells, the "regime eviction cannot enter"
framing is contradicted by the repo's own Week-11 table, and the retrieval harness (10 cyclic filler
sentences, one needle depth) makes "Wilson-firm 1.00" far weaker externally than it sounds.

**Score: 5/10 (borderline reject as currently worded; most fixes are $0 rewording + one ~$25–50 GPU
baseline/benchmark pass).**

---

## C1 — extreme-compression generality (r64-h256, three families)

### What the files actually show
- 16K, n=12/cell: single = 1.00 (12/12) and multivalue = 1.00 (12/12) on all three families —
  `results/w17-ruler-intervals.md:11,30,51`; Wilson [0.758, 1.0] verified by hand (lower = n/(n+z²) =
  12/15.84 = 0.758 ✓). Memory 0.085–0.149× (`results/w17-decision-table.json:5,124,261`).
- Baselines matched at n=12, same tasks, same items (seeds×trials deterministic,
  `scripts/w10_ruler.py:144`): palu-r0.5 = 0.504×, think-c0.5 = 0.750× — matched-n comparison is clean.

### Puncture 1 (MAJOR): "5–13× less memory than ThinK/Palu" does not reproduce
`docs/week17-explained.md:27` and `docs/week17-handover.md:14` claim "5–13× under ThinK/Palu".
Per-cell factors from the committed tables:
- vs ThinK 0.75×: 0.75/0.149 = **5.0×** (Qwen 16K) … 0.75/0.075 = **10.0×** (Llama/Mistral 32K).
- vs Palu 0.504×: 0.504/0.149 = **3.4×** (Qwen 16K) … 0.502/0.075 = **6.7×** (32K).

The honest range vs the stated baselines is **3.4–10×**. "13×" only appears as 1/0.075 = 13.3 — i.e. vs
**full KV**, not vs a baseline (or vs Palu ÷ Week-16's r16 mem ≈ 0.039, a different config than the r64
claim). Either way the headline multiplier is not derivable from the r64 cells vs ThinK/Palu.
**Fix ($0):** recompute per-cell, state "3.4–10× less than ThinK/Palu (6.7–13× less than full KV)".

### Puncture 2 (MAJOR): "Also 1.00 at 32K (n=4)" is false for Mistral
`results/w17-ruler-intervals.md:40-41`: Mistral 32K `bugSseed-r64-h256` multivalue = **0.75 (3/4)**, vt =
0.75 (3/4) (h1024 identical). Only Qwen (:20) and Llama (:62) hold 1.00/1.00/1.00 at 32K.
`docs/week17-explained.md:31` is honest ("Qwen additionally holds…"), but the paper-claim wording in the
briefing generalizes it. **Fix ($0):** scope the 32K sentence to Llama+Qwen, state Mistral 3/4; or firm
Mistral 32K to n=12 (~$5 GPU).

### Puncture 3 (FATAL if unaddressed): "a regime eviction/channel-pruning cannot enter" — the repo's own Week-11 table refutes the eviction half
`docs/week16-explained.md:69` ("you cannot evict your way to 0.05× and still answer a needle") and
`docs/week16-handover.md:42` ("the regime eviction/channel-pruning cannot enter"). The repo's own
Llama-8B table, `docs/week11-decision-table.md`:
- **line 14** — `ea-k0.1` at **0.100×**: needle 100, multikey 92, multivalue 100 (vt 17) @16K.
- **line 40** — `ea-k0.1` at 0.100× @32K: needle 100, mv 100, vt 83.
- **line 66** — `ea-k0.1` @64K: mv 100, vt 100.
- **line 22** — the repo's own `bugEVICT-h256` (pure exact tier, i.e. eviction) at **0.018×**: needle 100.

So ExpectedAttention eviction at 0.10× matches BUG's headline task pair (single+mv = 1.00) on the same
harness and model, at memory only 1.2–1.3× above BUG's 0.075–0.085×; and "answering a needle" at 0.02× by
eviction is *their own measured row*. The claim is defensible only for the two baselines actually carried
into Weeks 16–17 — ThinK (architecturally ≥0.5×: values untouched, `src/kvdlra/accounting.py:255`) and
Palu (measured collapse at r0.25 = 0.25×: 0/8 on all four tasks, `results/w15-ruler-intervals.md:36`) —
and **the strongest cheap baseline (EA) was dropped from the entire cross-model program** (no eviction
arm appears in `results/w16-ruler-intervals.md` or `w17-ruler-intervals.md`; EA was never run on
Qwen/Mistral at all). Reviewer 2 finds this in one afternoon.
**Fix:** reword to "a regime channel-pruning and weight-space low-rank cannot enter (measured collapse
points: think-c0.7, palu-r0.25)"; separately run `ea-k0.1` (and ea at ~0.075×) n=12 on all three families
(~$15–25 GPU). If EA holds single+mv at 0.075×, the eviction comparison must move to the vt column and
the memory delta shrinks to ~1.2×.

### Puncture 4 (MAJOR): external validity of "retrieval = 1.00, Wilson-firm"
- The haystack filler is **ten fixed sentences cycled** to fill 16–32K tokens
  (`scripts/w4_needle.py:46-57`, cycled at `scripts/w10_ruler.py:70-76`). A 16K context is ~130 verbatim
  repeats of the same ~120-token paragraph — a *maximally low-rank* KV stream in which the needle is the
  only high-surprise content. This is the friendliest possible setting for a low-rank gist + surprise-
  selected exact tier; official RULER uses diverse essay/noise filler. The 0.075× retrieval result may
  not survive natural-text haystacks, and no official RULER/LongBench/NIAH number exists for the flagship
  config (`scripts/w10_longbench.py` never run on it — briefing confirms Tier-3 deferred).
- **No depth sweep**: single-needle position is fixed at mid-context ± ≤4 sentences
  (`scripts/w10_ruler.py:150`, `n // 2 + (trial % 5)`); official RULER sweeps depth 0–100%.
- **Trial independence**: the 12 items per cell differ only in the random 5-digit code, tiny position
  offsets, and vt variable names (`w10_ruler.py:144-191`); the filler text is byte-identical across
  trials. Wilson intervals assume independent Bernoulli draws; these are near-replicates of one prompt
  family, so "n=12 needed to clear lower bound 0.70" (`docs/week17-explained.md:18`) measures per-config
  determinism more than generalization. The interval math itself is correct; the *interpretation*
  ("statistically firm generality") is overstated.
**Fix:** one official-RULER (or at minimum PG-essay filler + depth-swept in-repo variant) pass on the two
flagship configs × 3 families, ~$50 GPU + a few days; this is the single highest-value gap-fill for a
NeurIPS submission.

### Puncture 5 (MAJOR): the memory unit bills fp32-at-rest state at 16 bits
`ratio_fp16` computes stored bits with verbatim elements at 16 bits (`src/kvdlra/accounting.py:76-85`),
but BUG's dominant footprint components are **fp32 at rest**: basis `u_k` (n,r) fp32, coords `c_k` fp32
(`src/kvdlra/cache/bug_cache.py:575-577`; blocks upcast at :684). At r64-h256/16K/Llama, coords+U are
~80% of the footprint (coords 2·64·~16.2K = 2.08M of ~2.74M elems). Billed at their actual 32 bits the
ratio is ≈ **0.15×, not 0.085×** (Qwen: ≈0.27×, not 0.149×) — reviewer-computed from
`accounting.py:bug_footprint` components. The 16-bit billing assumes an fp16-stored-state variant that
was **never run**, and Week-17's own finding is that this integrator is numerically fragile
(near-null-tail explosions, C3) — precisely the setting where a fp16 cast is not obviously safe.
Baselines (ThinK/Palu, kvpress DynamicCache in model dtype) are billed at their true 16 bits.
The docstring is honest that this is a "deployment headline" (`accounting.py:26-31`), and the briefing
flags "not measured VRAM" — but the cross-method claim silently mixes a hypothetical BUG with actual
baselines. **Fix:** either validate fp16-stored state (one GPU probe, ~$10: cast U/C to fp16 between
steps, re-run r64 RULER+ppl) or report the fp32-at-rest ratio alongside; at fp32-at-rest the marquee
multiplier drops to ~3–5×.

### Also (MINOR)
- "BUG beats both baselines on multi-value (Qwen)" (`docs/week17-handover.md:14`,
  `week17-explained.md:29`): 12/12 vs think 10/12 is not separated (Fisher two-sided ≈ 0.48,
  reviewer-computed); Qwen 32K 4/4 vs palu 1/4 ≈ 0.14. "Leads on point estimate" is the honest verb.
- 0.075× is a 32K-only endpoint; the 16K range is 0.085–0.149× (wording).
- **Multikey was dropped from the whole Week-16/17 cross-model matrix** — the parser regex only knows
  single|multivalue|vt (`scripts/w17_intervals.py:32-34`); every mk cell in
  `results/w16-ruler-intervals.md`/`w17-ruler-intervals.md` is "—". Multikey was historically the hard
  NIAH variant for bugS (25% at 16K pre-seed, `results/w15-ruler-intervals.md:13`) and has **never been
  measured on Qwen or Mistral**. A reviewer will read the column of dashes as selective reporting.
  Fix: one mk block × 3 families, n=12 (~$5–10 GPU).

---

## C2 — mechanism (gist=fluency / tier=retrieval; rank wall; s32 on Llama)

Adequately evidenced at the stated strength, with small n but 0-vs-100 effect sizes:
- Tier-only vs gist-only dissociation: `bugEVICT-h256` needle 100/ppl 4.44 vs `bug-r128` needle 25/ppl
  4.17-ish (`docs/week11-decision-table.md:22,25`); Week-12 `bugSdrop == bug-r128` (memory pin).
- Rank wall with model-dependent onset: r64→r128 collapse 100→0 on Qwen (12 cells,
  `results/w16-ruler-intervals.md:13-15`), Mistral (:35-37); Llama onset later (:57-59).
- s32 A/B is a genuinely matched pair (only `-s32` differs): 0→100 flips at 16K single and 32K mv/vt
  (`results/w15-complete-summary.md:29-33`, raw lines in the **untracked** `results/w15b-complete-lines.txt`
  — commit it; git status shows it dangling).
- s32 does NOT rescue Qwen/Mistral (0/8 with s32, `w16-ruler-intervals.md:18,40`) — claim correctly
  scoped to Llama.
Caveat: all mechanism cells are n=4–8 and inherit the degenerate-filler concern above (the "surprise
isolates the needle" story is easiest exactly when the filler is 10 repeated sentences).

---

## C3 — `min_sv_frac` floor

Strongest claim of the five; numbers check out exactly:
- Qwen bug-r256 27531.666 → 6.995 (`results/w17-decision-table.json:56-70`); the h1024 "puzzle"
  714.39 → 6.941 (:75-82); Mistral bug-r128 138.317 → 5.574, r256 37.205 → 5.226 (:179-198).
- Monotone-in-rank with floor: Mistral 6.22→5.574→5.226, Qwen 7.535→7.218→6.995 ✓.
- Default-off/bit-identical is test-pinned (`tests/test_w17_rankfloor.py:75-110`).
Weak spots (MINOR): retrieval-neutrality is n=4 only (`w17-ruler-intervals.md:12,31`); "extends the safe
rank" is within-method — floored Qwen r256 (0.517×, ppl 6.995) is dominated by palu-r0.5 (0.504×, 6.355,
`w17-decision-table.json:23-31`), so higher rank is *safe*, not *competitive*; say so. Llama untested
(floor "not needed", onset >r128 — fine, but state it as untested, not unnecessary).

---

## C4 — marquee (Llama r128-h1024-s32 @32K)

- The interval logic is exactly right and I verified it: bug vt 15/16 → [0.717, 0.99]; think 5/16 upper
  0.556 → **separated** ✓; palu 9/16 upper 0.769 > 0.717 → **not separated** ✓
  (`results/w17-ruler-intervals.md:60,63-64`). The Week-16→17 downgrade of "beats Palu AND ThinK" to
  "beats ThinK, leads Palu not separated" (`docs/week17-explained.md:103-108`) is a model honesty
  correction. This *is* the strongest honest version under interval-overlap — but interval overlap is a
  conservative test: the trials are **paired by construction** (same deterministic items per seed/trial,
  `w10_ruler.py:144`; marquee block = seeds 0–1 × trials 0–7, `scripts/w17_intervals.py:12-14`), so
  McNemar on paired outcomes (or Fisher: 15/16 vs 9/16 ≈ p 0.04 two-sided, reviewer-computed) would
  likely make the Palu lead significant **for free** — if per-trial outcomes were retained from the pod
  logs; the committed line-files only carry aggregated acc, so this may need a re-run (~$5).
- (MAJOR) **Cross-week splicing**: the n=16 retrieval is Week-17 pods; the "ties ThinK/Palu ppl at 3–5×
  less memory" is Week-15 pods (`results/w15-complete-summary.md:17-24`, paired-window n=8, pre-registered
  tie threshold |Δ| ≤ max(0.05, 2SE) bits/tok, `docs/week15-significance.md:66-76`). The marquee row in
  `results/w17-decision-table.json:305-311` has **no ppl field**. Legitimate if the paper states the
  provenance; a reviewer who notices w15 ppl 7.353 < w17 full 7.52 across runs will ask. Also 32K deltas
  (+0.031, +0.024 bits) are ties only by the 0.05 floor — report the deltas, not just "TIE".
- (MINOR) "3–5×" should be 3.2–4.7× (0.502/0.159, 0.75/0.159); the explained doc's "3–4.7×" is right,
  the briefing's "3–5" rounds up.
- (MINOR) "beats ThinK" = beats **think-c0.5**; think-c0.3 (0.852×) had vt=100 (n=2) at 32K
  (`docs/week11-decision-table.md:59` region). The memory story survives (0.16 vs 0.85) but name the
  config. Marquee single/multikey at n=16 were never run (w17 row has only mv/vt); n=8 support is w15
  (`w15-complete-summary.md:12-13`) from a different pod — fill for ~$3 or footnote.

---

## C5 — honest limits

Verified present and correct: BUG loses ppl to baselines everywhere (e.g. Llama 16K 5.308 vs palu 5.006,
`w17-decision-table.json:268,287`); vt weakness stated with the n=12 downgrade 0.75→0.58
(`week17-explained.md:36`); h512/h1024 refutation shown (Mistral 0.50→0.25, `w17-ruler-intervals.md:32`);
32K r64 dip Qwen-only with cross-model ppl ratios that check out (35.083/7.76 vs 1.11–1.14× elsewhere).
One co-location demand: the Qwen 32K "1.00/1.00/1.00" retrieval cell and its ppl-35 fluency collapse are
the same cell — the paper must print them together, or a reviewer will do it for you. Note the ppl
behind that dip is n=4 windows (w17 "leaner ppl", commit 3e27089) — say so.

---

## Ranked gap-fills

1. **$0** — Fix "5–13×" → "3.4–10× vs ThinK/Palu"; scope 32K generality to Llama+Qwen (Mistral mv 3/4);
   "beats on mv" → "leads"; "3–5×" → "3.2–4.7×"; name think-c0.5; state ppl/retrieval provenance split.
2. **$0** — Reword "eviction cannot enter" to the measured collapse points (think-c0.7, palu-r0.25) and
   acknowledge ea-k0.1@0.100× from Week-11; commit `results/w15b-complete-lines.txt`.
3. **~$10 GPU** — fp16-stored-state probe (cast U/C between steps, r64 RULER+ppl ×3 families) OR publish
   fp32-at-rest ratios alongside; this defends the memory unit.
4. **~$15–25 GPU** — EA (and SnapKV) at 0.075–0.10×, n=12, all three families, all four tasks; plus the
   missing multikey column and marquee single/mk at n=16.
5. **~$50 GPU + days** — official RULER (essay filler, depth sweep) on the two flagship configs; without
   this, C1's "1.00 retrieval" rests on a 10-sentence cyclic haystack with one needle depth and will not
   survive review at a venue where RULER is a named benchmark.

## What's genuinely strong (keep)

- `src/kvdlra/accounting.py` is a model of comparative honesty: ShadowKV's offloaded values counted
  (:303-354), ThinK counted analytically because kvpress only zeros channels (:250-264), formula pinned
  to live caches by tests, matched-budget assertion gate (:453-464).
- `scripts/w17_intervals.py:8-15` explicitly refuses to pool subset/superset runs (no double-count).
- Pre-registered ppl tie threshold (`docs/week15-significance.md`), pre-registered w17 confirm matrix
  (commit 7f49091), and a documented record of claims corrected *downward* when n grew (vt 0.75→0.58;
  "beats Palu" retracted at n=16; two of their own baselines' bugs found and fixed in Week-15).
- Wilson arithmetic in every cited cell reproduces by hand (n=4/8/12/16 endpoints all verified).
