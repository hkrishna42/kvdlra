# Significance & framing review — kvdlra exit-gate re-review (2026-09-06)

Reviewer dimension: **significance & framing**. Prior scores for this dimension: 5 (2026-09-01 panel,
`docs/reviews/2026-09-01/review-significance.md:4`), 6 (Week-18 exit gate,
`docs/week19-kickoff.md:30-31`). Evidence base for this review: `paper/main.tex` (1016 lines, read in full),
`results/w19-a1-report.md`, `docs/week19-official-ruler.md`, `docs/board/week19-board.html`,
`results/w19-a4-llama-lines.txt`, `results/w19-a3-llama2-lines.txt`, `results/w19-a1q-*-lines.txt`,
`results/w19-a2-flagship-misses.md`, `results/w18-g4-llama-lines.txt`, `results/w18_harvest/quant-findings.md`,
`scripts/w10_frontier.py:187-345`, `src/kvdlra/accounting.py:150-250`.

**Score: 5/10 as worded (borderline reject); 6/10 after the $0 reframe below; 7 reachable with one ~$50
experiment if it lands.**

**FATAL: yes, as worded — but $0 to fix.** The abstract's closing frontier statement ("what it does not
share is the 1/T term", `main.tex:90-93`) and §memory's "all O(rn+hn), independent of T ... a bounded-rank
tracker is not floored at all" (`main.tex:831-845`) are refuted by the paper's own three-point series and
by the harness: the flagship's coordinate buffer is the whole context (`scripts/w10_frontier.py:209-210,
239-240`: `cb = t + rw + ab`), so its state is Θ(rT), and 0.151 → 0.140 → 0.133 (`main.tex:838`,
`results/w19-a4-llama-lines.txt:1`) fits a + b/T with a = 0.129 (predicts 0.1345 at 64K; measured 0.133):
it converges to 2r/n = 0.125×, not to zero. This is the same class of error the 2026-09-01 AC called
fatal for "eviction cannot enter" (`review-meta-verdict.md:51-55`): a headline sentence contradicted by
the repo's own numbers. It is fixable in an afternoon, and the corrected statement is still favourable
(see §2.1), which is why it is fatal-as-worded rather than fatal-to-the-program.

One-line verdict: *after the fair baseline and the official anchor, the paper's quantitative edge at
≥2 bits per element is gone (tie on official RULER, dominated by 4-bit at equal bytes on Qwen, marquee
tied by 4-bit at identical bytes); what survives is a genuinely new axis, a good failure map, and one
exclusive band below 2 bits/element that is measured on the in-repo generator only and has not yet met
its real competitor (eviction × quantization). That is a credible poster/borderline-accept mechanism
paper, not yet a main-track result paper — and the manuscript's own pre-registered fork said so
(`docs/week19-kickoff.md:44-49`: "2-bit also retrieves → BUG is a sub-0.05× storage-tier method →
honest poster on mechanism+rigor").*

---

## 0. What the narrowing left — the ledger

| claim (as worded) | evidence | status after W19 |
|---|---|---|
| C1 flagship vs 2-bit at matched bytes | `w19-a1-report.md:29-36,74-86,124-132` | wins mv at 16K on 3/3 (p=0.016/0.031/0.008); at 32K Mistral mk+mv, Qwen mv+vt; Llama 32K not separated; single tie everywhere; fluency to the quantizer on Mistral/Qwen (`:88-93,135-140`) |
| C1 vs 4-bit | `w19-a1-report.md:35-36,84-86,133` | never significant in BUG's favour; Qwen flagship 0.275× = 4-bit's bytes, 4-bit perfect (`:109-120`) |
| C2 official RULER | `week19-official-ruler.md:50-58,62-73` | mean 0.79 vs 0.87; mv 1 vs 1 discordant; pooled discordants over 9 tasks 8 (BUG) vs 16 (2-bit), pooled exact p ≈ 0.15; 4-bit/ThinK/Palu beat flagship on vt p=0.008/0.016/0.008 |
| C3 sub-cliff q4 cell | `w19-a1q-llama-lines.txt`, `-mistral-`, `-qwen-` | Llama 1/1/1/0.50 @0.048× (ppl 9.25 vs 5.31), 1/1/1/0.83 @0.034× (ppl 17.1 vs 8.33); Mistral 1/1/0.92/0.33, 1/1/0.83/0.58 (ppl 6.56/4.28 vs 5.50/3.76); Qwen 0/0/0/0, ppl 14490 |
| C4 persistence | `w19-a3-llama2-lines.txt` | 0.134 s vs full 1.226 s vs 2-bit 0.137 s (16K); 0.227 / 2.351 / 0.212 s (32K) — shared with 2-bit, 2-bit slightly faster at 32K |
| C5 marquee | `w18-g4-llama-lines.txt` | vt 0.94 n=16, **sbits = 0.284×**, billed in the paper as 0.16× (`main.tex:461,501`); KIVI-4bit at 0.284× scores vt 1.00 (`main.tex:700`) |
| C6 64K | `w19-a4-llama-lines.txt:6-21` | flagship 1/1/1/1 n=8 @0.133×; 2-bit 1/0.58/0.50/1 n=12 @0.158×; ea-k0.1 1/0.67/0.83/1 @0.100×; unpaired, no McNemar |

The Week-18 exit gate scored significance 6 with C1's outcome unknown and the sub-cliff band
"conditional on a fair 2-bit baseline never run" (`week19-kickoff.md:36-41`). C1 landed on the narrow
branch on the external anchor, C5 is now tied at equal bytes, and C6 is mis-described. Net of the new
honest cells (C3, C4, C6), significance is *lower* than at the exit gate as worded, and equal to it once
reframed.

---

## 1. The surviving contribution set — item by item

**1.1 The online-DLRA mechanism + surprise tier (novel).** Unoccupied axis; both prior panels and the
prior-work reviewer agreed (`review-meta-verdict.md:116-119`). The differentiation from LESS/LoLA
(`main.tex:319-332`) and from MomentKV/ResKV (`:345-352`) is well argued. Near-oracle tracking
(1.01–1.03× of truncated SVD, `:380-386`) is a real property. **Verdict: publishable idea; on its own a
workshop/poster contribution, because novelty without a quantitative win is what it currently is.**

**1.2 The mechanistic map** — r/n≈0.25 wall with the seeded Llama r256 control at n=12
(`main.tex:524-532`; `w18-g4-llama-lines.txt`: r256 0.00 on all four @0.534× honest), rank siphoning +
relative floor (`:536-570`), vt chains resist surprise selection with the obvious fix refuted
(`:926-935`). **Verdict: the paper's best scientific asset, unchanged from the last review. It is an
analysis of the paper's own representation, so it carries a method paper but cannot carry a main-track
paper alone.**

**1.3 The sub-cliff composed cell** (`:572-608`). This is the one band where the claim "no scalar
quantizer reaches" is literally true — and the paper states it with the right qualifications
(family-dependent, fluency cost, vt unchanged). Two things limit its significance today:
(a) it is in-repo-generator only; the official anchor (`:751-772`) was run on the flagship, not on this
cell, and the paper's Limitations do not say so; (b) its real competitor is not "a scalar quantizer" but
*eviction composed with quantization* — KIVI-4bit×ea-k0.1 ≈ 0.03×, KIVI-2bit×ea-k0.25 ≈ 0.04× — which
is unrun at 7B/8B. The paper's own 1B study already found EA×TurboQuant and SnapKV×TurboQuant "on or
ahead of BUG's Pareto frontier through 0.08–0.18×" with BUG best only below 0.07× (`:806-812`); at 8B on
retrieval, the Llama in-repo eviction cell at 0.1× holds single/mk/mv 1.00/0.92/1.00 (`:622`), so a 4-bit
composite would plausibly hold them at 0.03×. On official RULER ea-k0.1 collapses (mean 0.20,
`week19-official-ruler.md:58`), which is exactly why the composite must be run there. **Verdict: the
paper's only exclusive quantitative claim; currently under-anchored and under-contested. Fixable with one
pod (§5).**

**1.4 The 1/T term to 64K.** As measured it is real and honest: 0.151 → 0.140 → 0.133 with full four-task
retrieval at n=8 where 2-bit loses mk/mv (7/12, 6/12). As *framed* it is wrong (§2.1): the ratio is
converging to 2r/n ≈ 0.125×, a 20% margin under 2-bit's 0.156× floor, not "not floored at all". Also
note eviction at 0.100× scores 1/0.67/0.83/1 at 64K (`w19-a4-llama-lines.txt:7,11,15,19`) — the 64K
point is not exclusive against eviction either, and it is unpaired (flagship n=8 vs n=12; no McNemar).
**Verdict: a modest, honest asymptotic point once restated; not a frontier statement.**

**1.5 Persistence, measured.** The 9–10× cold-start win over full KV is shared with 2-bit
(`w19-a3-llama2-lines.txt:3,6`: 0.137 s / 0.212 s — 2-bit is *faster* at 32K). The paper says so
(`:893-900`). **Verdict: closes the last panel's "no deployment number" gap (`review-significance.md
(09-01):65-70`) but adds no differential significance. It is a systems sanity check, not a contribution.**

**1.6 The honest negatives** (Qwen q4 divergence, Qwen r64 32K ppl 35.1, vt ceiling, official-anchor
non-transfer, q4-bug disclosure `results/w18-g1-report.md:18-23`). **Verdict: they protect the surviving
numbers; they do not add significance (§4).**

**Is the set an ICML main-track contribution?** Not as a result paper: there is no cell at ≥2 bits per
element where BUG is separated from the best fixed-bit method at equal honest bytes on an external
benchmark. As a mechanism paper with one exclusive band it is a 6 if the band is anchored (§5) and a
5 if it is not.

---

## 2. Framing — is the abstract the best honest statement? No. Three concrete defects and one lens.

### 2.1 The lens the paper is missing: rank is a bit axis, and the fp32 coordinates make the flagship a 2-bit code

`src/kvdlra/accounting.py:170-177` bills the coordinates as `2·rank·coord_count` fp32 elements
(K and V). Per token that is 2r·32 bits against 2n·16 bits for full KV, i.e. **b_eff = 32·r/n bits per
element**:

| config | n | r | coord dtype | b_eff | measured asymptote |
|---|---|---|---|---|---|
| flagship Llama/Mistral | 1024 | 64 | fp32 | **2.0** | 0.129× (fit) ≈ 2/16 |
| flagship Qwen | 512 | 64 | fp32 | **4.0** | 0.265× @32K → 0.25 |
| marquee Llama r128 | 1024 | 128 | fp32 | **4.0** | 0.284× (`w18-g4-llama-lines.txt`) |
| q4 cell | 1024 | 64 | 4-bit | **0.25** | 0.020× (fit from 0.048/0.034) |
| KIVI-2bit incl. aux | — | — | — | 2.5 | 0.156× |
| KIVI-4bit incl. aux | — | — | — | 4.6 | 0.284× |

This one table explains every outcome of the fair comparison: the flagship ties 2-bit on Llama/Mistral
because it *is* a 2-bit code; it is dominated by 4-bit on Qwen because there it *is* a 4-bit code with
worse fluency; the marquee is tied by 4-bit KIVI at literally identical bytes (0.284× vs 0.284×; vt
0.94 vs 1.00, mv 1.00 vs 1.00, ppl 7.353 vs 7.54); and the q4 cell is exclusive because 0.25 bits/element
cannot be configured on a scalar quantizer. The paper never states this equivalence, and so its narrative
reads as a sequence of surprising retreats instead of one predictable law. Adopting the lens turns the
narrowing into the paper's organizing result: **"at equal bits per element, the rank axis retains
retrieval at least as well as the scalar axis (in-repo: better on mv; official: tie) and is the only axis
that continues below 2 bits."** It also makes the fp16-gist follow-up (`:954-955`) legible as "a 1-bit
code", and an 8-bit-coordinate cell as "0.5 bits".

### 2.2 Defect A (fatal-as-worded): the 1/T / "not floored" claim

`main.tex:831-833` ("all O(rn+hn), independent of T"), `:843-845` ("not floored at all"), abstract
`:90-93`. Refuted by `scripts/w10_frontier.py:209-210,239-240` (coordinate buffer = whole context) and
by the paper's own series (a + b/T fit: fp32 a = 0.129; fp16-eq a = 0.065 ≈ r/n = 0.0625; 64K tok_eq
4497.7 ≈ 65536/16 + U 64 + tier 256 + ring, `w19-a4-llama-lines.txt:1`). The bounded-buffer variant the
sentence describes exists in the code (`:233-237`, coordinate eviction) but is not the measured
flagship. **Fix ($0):** "the flagship's stored ratio converges to 2r/n (0.125× on 1024-wide KV,
20% under the 2-bit quantizer's 0.156× floor); the O(1) basis/tier/ring overhead amortizes as 1/T
(0.151 → 0.140 → 0.133); the composed cell converges to ≈0.02× (0.25 bits/element), where its 1/T term
is the one that matters (0.048 → 0.034)." That is still a favourable, true sentence.

### 2.3 Defect B: the marquee is billed at 0.16× and never meets 4-bit KIVI

`main.tex:461,470,501` bill `bugSseed-r128-h1024-s32` at 0.16× and claim "3.2–4.7× less stored state"
than think/palu; `results/w18-g4-llama-lines.txt` gives sbits = 0.284×. The honest margins are 2.6×
(think) and 1.8× (palu), and KIVI-4bit at the same 0.284× scores vt 1.00 / mv 1.00 at 32K
(`main.tex:700`). §marquee (`:457-519`) does not mention the 4-bit arm at all. The abstract's "beats
think-c0.5 ... at 3.2–4.7× less stored state" (`:66-70`) is therefore the flattering billing of a cell
that is tied at equal honest bytes by a method the paper itself ran. **Fix ($0):** bill 0.284×, add the
4-bit row to Table tab:marquee, and state the marquee as "ties 4-bit KIVI at identical bytes (vt 0.94 vs
1.00 n.s.; ppl 7.35 vs 7.54) while beating ThinK/Palu at 1.8–2.6× less stored state". It stays a
positive result; it stops being a frontier one.

### 2.4 Defect C: two billings of one config inside one abstract, plus stale v1 text

The abstract quotes the flagship at 0.085–0.149× (`:62`) and at 0.15× (`:76`); the intro at 0.085–0.149×
(`:144`), §xmodel at 0.085–0.149× with "3.4–10×" (`:398-400`, honest: 1.8–5×), §quantbaseline at 0.151×
(`:649`). A reviewer who notices asks which one the paper believes. The dual-billing footnote
(`:822-827`) does not license leading with the flattering one. Separately, the manuscript still says the
2-bit baseline is absent: header `:5-8,20`, §quant `:277-281` ("primary v2 experiment"), §related
`:302-305` ("their absence as a baseline is v1's main scoping decision") — contradicting §quantbaseline.
Also the broken sentence at `:585-586` ("Three\nAt 32K"). **Fix ($0):** honest billing everywhere;
delete the v1 sentences.

### 2.5 Genre call and the strongest honest statement

- *Storage-tier paper*: weaker than last time — the persistence win is shared with 2-bit (`:893-900`),
  and the 1/T separator is a 20% asymptotic margin. Do not lead with it.
- *Negative-result paper*: ICML main track does not reward "we ran the fair baseline and it tied"
  unless the negative is about a widely held belief; this one is about the authors' own prior claim.
- *Mechanism + bits-per-element + sub-cliff paper*: the right genre. The mechanism is novel, the lens
  organizes every table, the q4 cell is the exclusive quantitative claim, the map explains the limits,
  the official anchor is the external-validity ceiling.

**Strongest honest contribution statement (3 sentences):**

> We introduce the first online, training-free KV-cache compressor that tracks a per-sequence low-rank
> subspace with a dynamical low-rank (BUG) integrator and keeps only surprise-selected tokens exact;
> because rank-r fp32 coordinates are a (32r/n)-bit-per-element code, the fair comparison is against
> scalar quantization at equal bits per element, and at 2 bits (r=64 on 1024-wide KV) the rank axis ties
> a KIVI-faithful 2-bit quantizer on single-needle retrieval and on the official RULER suite, wins
> multi-value/multi-key retrieval on our generator at a fluency cost, and is dominated by 4-bit
> quantization wherever its own bytes reach 4 bits per element (Qwen, the r=128 marquee). Below 2 bits
> only the rank axis exists: the composed rank-64/4-bit-coordinate cell (0.25 bits/element; 0.048× at
> 16K, 0.034× at 32K, converging to ≈0.02×) retains full single/multi-key/multi-value retrieval on Llama
> and Mistral at a measured fluency cost, fails on Qwen, and is the one band no fixed-bit quantizer can be
> configured to enter. We map where the representation breaks — a dimensionless r/n≈0.25 retrieval wall,
> a rank-siphoning tier–gist interaction stabilized by a relative truncation floor, and variable-tracking
> chains that surprise selection cannot protect — and show that its stored-state advantage is today a
> persistence/transfer property shared with quantization rather than a resident-memory one.

---

## 3. Audience — who reads this and who changes practice

- **KV-compression researchers (the ICML audience):** would read it for the new axis, the bits/element
  equivalence, and the failure map. They would *not* change what they benchmark against, because the
  paper shows fixed-bit quantization is at least as good at ≥2 bits/element on the external suite.
- **Serving/persistence engineers (CacheGen/Mooncake/LMCache):** no practice change. The measured
  cold-start win is matched by 2-bit KIVI (`w19-a3-llama2-lines.txt`), which is simpler, already in
  transformers, and wins resident memory; a 20% asymptotic byte margin at a fluency cost does not move
  them. The q4 cell (5× smaller than 2-bit) could interest an *archival retrieval tier* for
  agent memories, but only on Mistral is its fluency cost small (4.28 vs 3.76 at 32K); the paper leads
  with Llama (17.1 vs 8.33), which undersells the one family where the cell is deployable.
- **DLRA / numerical-analysis community:** genuinely interested (first activation-cache use of BUG;
  the siphoning/floor finding), but they are not ICML's reviewers.

Net: practice changes for nobody today. That is acceptable for a mechanism paper, provided the abstract
stops implying a frontier win.

---

## 4. Does the self-scrutiny help or hurt at a top venue?

**Helps trust, hurts legibility, and the balance is currently wrong.** Retiring the "beats SOTA at 10×"
claim (`:812-813`), disclosing the q4 budget inversion (`:608` comment; `results/w18-g1-report.md:18-23`),
fixing two baselines in their own favour (`:515-518`), and stating that the anchor cuts both ways
(`:957-966`) are exactly what reviewers reward — *once they can find the contribution*. Today the
abstract is 47 lines, states five results with roughly a dozen qualifiers, and its two strongest-sounding
sentences (marquee 3.2–4.7×; 1/T "no scalar quantizer can match") are the two that do not survive
scrutiny. A reviewer skimming it concludes "unclear contribution" before reaching the honesty. The
disclosure content is right; its placement is not: the q4-bug note belongs in provenance, the
official-anchor non-transfer belongs in the results paragraph where it already is, and the abstract
should carry one claim (the 3-sentence statement above) plus one limitation.

One specific over-concession: "4-bit concedes nothing" (`:78,158,944-945`) is generous to the
quantizer — on Mistral the flagship leads 4-bit on vt at both contexts on point estimate (0.50 vs 0.25,
0.42 vs 0.25; `w19-a1-report.md:59-70`) at half the bytes, not separated. "No significant contrast in
either direction" is the accurate phrase.

---

## 5. The single ≤$50 experiment that most raises significance

**Run the sub-cliff cell against its real competitor, on the official anchor.** One pod
(~$40–50, Llama + Mistral; ~$30 Llama only), pre-registered:

- arms: `bugSseed-r64-h256-q4` (0.048×/0.034×), plus the composites KIVI-4bit×ea-k0.1 (≈0.03×) and
  KIVI-2bit×ea-k0.25 (≈0.04×) — kvpress presses compose with a quantized cache, and the repo already
  owns both halves (`src/kvdlra/quant/kivi_cache.py`, `build_arms` press path);
- in-repo four tasks at 16K/32K, n=12, paired on the Week-18/19 needles; official RULER Llama 16K, 9
  tasks, 12 records (`scripts/w19_official_ruler.py`).

Why this and not the fp16/8-bit gist: the q4 cell is the paper's *only* exclusive quantitative claim,
and it currently rests on (a) a generator the official suite has already shown to overstate the
flagship's mv edge (`week19-official-ruler.md:109`) and (b) the absence of the composite competitor the
paper's own 1B frontier says is the one to beat in the 0.03–0.08× band (`main.tex:806-812`). If the
composite collapses on the essay haystack the way ea-k0.1 does (mean 0.20, `week19-official-ruler.md:58`)
while the q4 cell holds single/mk/mv, the band is anchored and exclusive against *every* family — a
clean 6–7. If the composite holds, the paper has no exclusive quantitative claim and must be the
mechanism paper it partly already is — better to learn that now than from Reviewer 2. This is the
same decisive-fork logic the program applied to A1 (`week19-kickoff.md:42-49`).

**Runner-up (~$30 GPU + days of harness): 8-bit / fp16 coordinate demotion** (`bugSseed-r64-h256-q8`,
0.5 bits/element, ≈0.08× at 16K). The q4 cell already proves retrieval survives coordinate quantization;
q8 would very likely keep the flagship's fluency (5.31) at half the 2-bit floor — a headline cell "below
2-bit at 2-bit's fluency". Higher upside, but it still needs the composite competitor above to be
conclusive, so it is second.

---

## 6. FATAL / fixable / nitpick

**FATAL as worded ($0 to fix; must fix before any submission):**
- F1. "O(rn+hn), independent of T" / "not floored at all" / abstract 1/T sentence (`main.tex:90-93,
  829-845`) — refuted by `scripts/w10_frontier.py:209-210,239-240` and by the paper's own 0.151/0.140/
  0.133 series (converges to 0.129 ≈ 2r/n). Restate per §2.2.
- F2. Marquee billed at 0.16× (`:461,501`) against sbits 0.284× (`results/w18-g4-llama-lines.txt`);
  "3.2–4.7× less stored state" (`:66-70,469,477`) is 1.8–2.6× honestly, and 4-bit KIVI ties it at
  identical bytes (`:700`) without appearing in §marquee. Re-bill and add the row.

**Fixable (each priced):**
- X1. Adopt the bits-per-element lens (§2.1) as the organizing frame of §quantbaseline/§subcliff/§memory
  and the abstract; rewrite the abstract to the 3-sentence statement. $0, a day.
- X2. One billing in the abstract/intro/§xmodel (0.151/0.275 honest; "1.8–5× under ThinK/Palu"); delete
  the stale "baseline absent / v2 experiment" sentences (`:5-8,20,277-281,302-305`). $0.
- X3. The sub-cliff cell vs eviction×quantization composites on official RULER + in-repo (§5). ~$40–50.
- X4. State explicitly in Limitations that the q4 cell has no official-anchor number (`:957-966` covers
  only the flagship). $0 until X3 lands.
- X5. 64K: pair the flagship to the comparators' needles (n=12) and report McNemar; say that ea-k0.1 at
  0.100× also holds single/vt there (`w19-a4-llama-lines.txt:7,19`). ~$10.
- X6. Lead the q4 showcase with Mistral (ppl 6.56/4.28 vs 5.50/3.76) rather than Llama (9.25/17.1);
  the deployable instance of the exclusive band is Mistral. $0.
- X7. q8/fp16 coordinate cell (§5 runner-up). ~$30 + days.

**Nitpick:**
- N1. "indistinguishable" on official RULER (`:80-82`) — n.s. at 12 records/task but the pooled
  discordants are 8 vs 16 (p≈0.15) and the flagship trails on 4 of 9 tasks; "not separated; trails on
  point estimate" (which §realistic already says at `:762-765`) is the abstract-safe wording.
- N2. "multi-key/variable-tracking at 32K on two" (`:77`) — it is mk on Mistral and vt on Qwen, one
  each; say so.
- N3. "4-bit concedes nothing" → "no significant contrast either way" (Mistral vt leads 4-bit on point
  estimate at half the bytes, §4).
- N4. Broken sentence `:585-586` ("Three / At 32K ...").
- N5. Table tab:official mean for the flagship is 0.79 in the paper, 0.80 in `week19-official-ruler.md:105`
  — reconcile.

---

## 7. Score rationale and trajectory

- **5 as worded.** The two abstract-level frontier sentences that a reviewer would test first (1/T
  unfloored; marquee 3.2–4.7×) fail against the paper's own artifacts, and at ≥2 bits/element the
  method is tied or dominated by fixed-bit quantization at equal honest bytes on the external suite.
  Under ICML calibration a novel mechanism with a good failure map and no external-benchmark win is
  borderline reject when the abstract overclaims.
- **6 after F1/F2/X1/X2 ($0).** Same evidence, honestly organized: a new axis, a bits/element law that
  predicts every fair-comparison outcome, an exclusive sub-2-bit band with stated caveats, and the map.
  That is a defensible poster / borderline accept — the "honest poster on mechanism+rigor" branch the
  program pre-registered.
- **7 if X3 lands favourably** (q4 cell holds single/mk/mv on official RULER while the composites
  collapse): the exclusive band becomes anchored and contested-and-won, and the paper has one clean
  quantitative claim no other family can make. **6 stays if X3 goes the other way**, and the paper should
  then be written as the mechanism paper without the word "exclusive".
- Persistence (C4) and the 64K point (C6) do not move the score in either direction once restated; they
  are hygiene, not significance.
