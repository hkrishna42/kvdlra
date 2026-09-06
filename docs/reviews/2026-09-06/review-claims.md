# CLAIMS review — kvdlra manuscript (paper/main.tex @ 5bc772f), exit-gate panel 2026-09-06

Dimension: are the paper's claims supported by the cited evidence **at the strength stated**?
Method: every headline number in the abstract, intro bullets, §4 and the conclusion was traced to a
result file line; paired tests were recomputed from the per-trial records where the paper does not
report them; arithmetic was re-derived. $0, read-only. Paths are relative to `/Users/hari/Desktop/kv-dlra`.

**Score: 6 / 10** (borderline accept / poster). **FATAL: no.** The evidence discipline is unusually
good (per-trial records, paired McNemar, SHA-pinned pods, dual memory billing disclosed) and the
*direction* of every headline survives. What does not survive is the *strength* of several headline
statements: the paper bills the same configuration three different ways across sections, and the
abstract is consistently one notch firmer than the body. All findings below are fixable; the three
must-fixes are CPU-only wording plus about $20 of GPU. With them the dimension is a 7.

---

## 1. Verification ledger (headline numbers -> evidence)

| # | Claim (main.tex line) | Evidence (file:line) | Status |
|---|---|---|---|
| 1 | Abstract:56-57 near-oracle 1.01-1.03x of truncated SVD | `README.md:117-118` only; no result file cited | carried, unverifiable from named evidence |
| 2 | Abstract:60-62 / §4.2:393-396 single+mv 12/12 on 3 families at 0.085-0.149x, Wilson >=0.758 | `results/w18-g1-report.md:7-9`; `results/w18-{llama,mistral,qwen}-lines.txt` | **verified** (fp16-equivalent billing) |
| 3 | Abstract:63-66, Intro:144, §4.2:399-400, Concl:997 "3.4-10x less than think/palu, 6.7-13x less than full KV" **for 16K** | 0.75/0.085 = 8.8x; 1/0.085 = 11.8x. The 10x and 13x need the **32K** 0.075x (`w18-g1-report.md:16`) | **overstated upper bounds** in a sentence scoped to 16K (4 places) |
| 4 | Abstract:66-68 / §4.4:466-468 marquee beats think-c0.5, 0.94 vs 0.31, n=16, p=2.0e-3, Wilson-disjoint | `results/w18-g4-marquee-contrasts.json:6`; `results/w18-g4-llama-lines.txt:13,16` | **verified** |
| 5 | Abstract:69 / §4.4:469-471 palu-r0.5 0.56, p=0.03, suggestive | `w18-g4-marquee-contrasts.json:7` (p=0.0312); `w18-g4-llama-lines.txt:15` | **verified** |
| 6 | Abstract:70, Intro:147, §4.4:461/469, Table 3:501, Concl:1000 marquee at **0.16x**, "3.2-4.7x less stored state" | `w18-g4-llama-lines.txt:13`: `ratio=0.160 sbits=0.284`. Honest (fp32-at-rest) bytes are **0.284x** -> 1.8-2.6x less, and byte-matched with KIVI-4bit (0.284x, `results/w19-a1-report.md:23`) which scores vt **1.00 [0.76,1.0]** at 32K Llama vs marquee 0.94 [0.72,0.99] | **overclaim / billing inconsistency** — the paper never prints 0.284 for the marquee (grep: 0.284 appears only in KIVI-4 rows and a source comment) |
| 7 | Abstract:75-77, Intro:156-159 "at matched stored bytes (0.15 vs 0.16x) ... wins multi-value on all three families (p<=0.03)" | `w19-a1-report.md:30,78,127`: p = 0.0156 / **0.0312** / 0.0078. Qwen flagship is **0.275x** vs 2-bit 0.163x (`w19-a1-report.md:109-110`) | p<=0.03 false for Mistral (nit); **Qwen is not matched bytes** (1.69x the bytes) — overclaim |
| 8 | Abstract:77, Limits:948-949, Concl:1005 "multi-key/variable-tracking at 32K on two [families]" | mk significant on **Mistral only** (`w19-a1-report.md:81`, 9/0 p=0.004; Qwen mk 5/0 p=0.0625 `:130`); vt significant on **Qwen only** (`:132`; Mistral vt 5/0 p=0.0625 `:83`) | **reads stronger than the file**: one task per family, not both on two. Body §4.7:654-656 states it correctly |
| 9 | Abstract:78 "ties single-needle everywhere" | all single contrasts p>=0.5 (`w19_intervals/a1-*-ruler-intervals.md:29,33`) | verified |
| 10 | Abstract:78 "matched or beaten on fluency"; §4.7:668 "goes to the quantizer or ties" | Llama 16K flagship 5.31 **<** 2-bit 5.40 (`w19-a1-report.md:42`); Qwen 32K **35.1 vs 8.23** (`:140`) | Llama contrast is inside the instrument's resolution (see §3.5); Qwen 32K is a 4x ppl collapse that the abstract never mentions — abstract underdiscloses |
| 11 | Abstract:78-79, Intro:159, §4.7:660-661, Limits:949-950, Concl:1006 "4-bit ... concedes nothing / at least as good in **every** cell / matches or beats in every cell" | Mistral vt 16K: 4-bit **0.25** vs flagship 0.50 (`w19-a1-report.md:59,61`); 32K: 0.25 vs 0.42 (`:68,70`); Qwen 16K mv 0.92 vs 1.00 (`:109,111`). None significant (`:84,86,133`) | **false on point estimate in 3 cells**, stated 5 times; the paper's own Table 6 (lines 702-707) contradicts the sentence above it |
| 12 | Abstract:80-82 official RULER: flagship and 2-bit "indistinguishable", eviction collapses | `results/w19_intervals/a2-llama-ruler-intervals.md:22-30` no sig contrast; means 0.79 vs 0.87 (`docs/week19-official-ruler.md:56-57`; recomputed 86/108 vs 94/108); pooled paired 8 vs 16 discordant, p=0.15 (recomputed) | defensible; but the abstract does not say the multi-value edge *failed to transfer* — body (760-772) and Limits (957-966) do |
| 13 | Abstract:83-86 / §4.6:576-597 composed cell 0.048x/0.034x, Llama 1/1/1/0.50 & 1/1/1/0.83, Mistral 1/1/0.92/0.33 & 1/1/0.83/0.58, ppl 9.25/17.1 vs 5.31/8.33, Mistral 6.56/4.28, Qwen 0/0/0/0 diverged | `results/w19-a1q-llama-lines.txt:1-10`, `w19-a1q-mistral-lines.txt:1-10`, `w19-a1q-qwen-lines.txt:1-5` | **verified**; "at a fluency cost" (abstract) understates a 2.05x ppl (17.1 vs 8.33) — see §3.4 |
| 14 | Abstract:87 / §4.10:864-865 decode residency ~1.06x | `main.tex:912` source comment: "from review-systems.md"; = workspace 0.982 (`results/w18-g5-llama-lines.txt:2`) + stored 0.084 | analytic estimate presented as a measured figure — say "estimated" |
| 15 | Abstract:89-90 / §4.10:889-893 cold start 0.13 s vs 1.2 s (16K), 0.23 vs 2.35 (32K), 9-10x, shared with 2-bit (0.14/0.21 s) | `results/w19-a3-llama2-lines.txt:1-6` (0.134/1.226/0.137; 0.227/2.351/0.212); 1.226/0.134 = 9.15, 2.351/0.227 = 10.4 | numbers verified; **label wrong**: `scripts/w19_persist.py:13-14` says timings are taken *after the file was just written (warm page cache)* — this is a warm-cache reload, not a "disk read"/"cold start" (main.tex:890) |
| 16 | Abstract:91-93 / §4.10:837-843 1/T: 0.151 -> 0.140 -> 0.133 at 16K/32K/64K; 64K flagship 1.00 on all four; 2-bit 1.00/0.58/0.50/1.00; ea 1.00/0.67/0.83/1.00; ppl 8.59/8.15/8.15/7.27/7.26 | `results/w19-a4-llama-lines.txt:1-21` (flagship **n=8**, comparators n=12); `w19-a3-llama2-lines.txt:1,4` | numbers verified; abstract omits n=8; **no contrast separates at n=8** (recomputed from `results/w19_pertrial/a4-llama-trials.txt`: mk 4/0 p=0.125, mv 4/0 p=0.125 vs 2-bit; vt 3/0 p=0.25 vs 4-bit). 2-bit is not perfectly "flat": 0.163 -> 0.160 -> 0.158 (`w19-a1-llama-lines.txt:1-2`, `w19-a4-llama-lines.txt:4`) |
| 17 | §4.3:426-437 / Table 2 32K n=12 cells and Qwen ppl 35.1 | `results/w18-g1-report.md:14-16` | verified |
| 18 | §4.4:511-514 same-pod ppl full 6.975 / think 7.196 / palu 7.232 / marquee 7.353; "+0.024/+0.031 bits/token"; "+0.076 vs full" | `results/w18_harvest/g4-llama.raw` (full/think/palu lines; `bugSseed-r128-h1024-s32 ppl=7.353 sbits=0.284`) | numbers verified; bits/token are **swapped**: log2(7.353/7.196)=0.031 (think), log2(7.353/7.232)=0.024 (palu); ppl arm is named `bugSseed-r128-s32` in the text (512) but is the h1024 arm in the raw — naming nit |
| 19 | §4.5:529-532 r256 control 0.00 on all four 32K tasks, n=12 | `w18-g4-llama-lines.txt:2,6,10,14` | verified; but the control has **no ppl line** in `g4-llama.raw` — see §3.6 |
| 20 | §4.5 Table 5 floor numbers | `docs/week17-explained.md:80-84` | verified (carried) |
| 21 | §4.6:598-599 "ThinK and Palu are floored at 0.50-0.75x **by construction**" | the paper's own `palu-r0.25` at 0.257x (`results/w11-goalA-lb-lines.txt:14`, cited at main.tex:749) | **self-contradiction** — those are chosen operating points, not floors |
| 22 | §4.6:606-607 SnapKV "multi-key 0.17-0.42, vt 0" | Llama 0.17/0.42 (`results/w18-g3-llama-lines.txt:3,6`); Qwen and Mistral mk = **0.00** (`w18-g3-qwen-lines.txt:3,6`, `w18-g3-mistral-lines.txt:3,6`) | range is Llama-only; three-family range is 0.00-0.42 |
| 23 | §4.6 Table 4 eviction grid | `w18-g3-llama-lines.txt:2,8,14,20,5,11,17,23`; `w18-g3-qwen-lines.txt:14,2,8,20`; `w18-g3-mistral-lines.txt:14,2,8,20` | verified |
| 24 | §4.7 Table 6 fair-quant cells and stored ratios | `w19-a1-report.md:12-23,59-70,109-120` | verified; Mistral 16K flagship printed 0.151x, file says 0.150x (`:59`; `w18-mistral-lines.txt` mixes 0.150/0.151) — nit |
| 25 | §4.7:639-640 8-bit hqq control 1.00 on all four on Llama and Qwen (n=4) | `w19-a1-report.md:15,112` | verified; on **Mistral** the 8-bit control scores vt **0.25** (`:62`), same as 4-bit (0.25) and 2-bit (0.33) — see §3.7 |
| 26 | §4.8:736-744 WikiText filler: flagship single 1.00; ea 0/0/0/0; palu 1/1/0.83/0.08; think 1/1/0.67/0.33 | `results/w18-g2-qwen-lines.txt:1-13` | numbers verified; **the flagship was run only on single-needle** (`:7`), baselines on all four — see §3.8 |
| 27 | §4.8:745-750 LongBench Qasper F1 | `results/w11-goalA-lb-lines.txt:3,7,8,10,14,18` | verified (one document, disclosed) |
| 28 | §4.8:751-772 / Table 7 official RULER cells, means, 6/9 separations, 1v1 & 0v4 discordant, vt p=0.008 | `docs/week19-official-ruler.md:50-58,62-101`; `w19_intervals/a2-llama-ruler-intervals.md:22-66`; `results/w19-a2-llama-lines.txt` | verified |
| 29 | §4.8:767-769 "Its 14 misses sit at needle depths 0.15-0.95, so this is not a warm-up-window or recency effect but scattered misses" | `results/w19-a2-flagship-misses.md:7-20,38`: **6 of 14 at depth <= 0.20** (uniform expectation 2.8; binomial P(>=6)=0.044; 5/13 excluding record 4991 that every arm misses, p=0.10). 2-bit: 1/10 at <=0.20; Palu 2/9 | **interpretive overclaim**: the misses lean *early*; also 22 misses in total (8 vt misses have no depth, `:21-28`) |
| 30 | §4.9:802-804 1B composition 0.099x vs 0.097x | `docs/week4.md:56-57` | verified (carried) |
| 31 | §4.10:860-862, 881-882 workspace 0.982/0.991, coldload 0.150/0.139, gist 0.135/0.130 | `results/w18-g5-llama-lines.txt:1-5` | verified |
| 32 | Setup:372-373 per-trial records for every headline cell | `results/w18_pertrial/*`, `results/w19_pertrial/a1*,a2*,a4*` present (a4: 176 rows = 8x4 + 3x12x4) | verified; W19 dir not cited in the text (only `w18_pertrial` at 373, 973) |

---

## 2. FATAL flaw

**None.** The closest candidate is #6 (marquee billed at 0.16x where the paper's own honest metric
gives 0.284x). It is not fatal because (a) dual billing is defined and disclosed (main.tex:821-827),
(b) the "beats think-c0.5" contrast is on retrieval, which is billing-independent, and (c) under honest
billing the marquee is still 2.6x under ThinK. But as printed, the abstract's "3.2-4.7x less stored
state" is 1.8x too strong on the paper's own preferred metric, and the marquee sits — unremarked —
at exactly the bytes of the 4-bit quantizer that the paper says "concedes nothing".

---

## 3. Fixable gaps (must-fix, in priority order)

### 3.1 One config, three billings, no reconciliation in the abstract (CPU-only)
The abstract bills `bugSseed-r64-h256` as 0.085-0.149x (line 62, fp16-equivalent), 0.15x (line 76,
fp32-at-rest), and 0.151x (line 92, persisted bytes) without saying these are the same configuration.
The marquee is billed only fp16-equivalent (0.16x, lines 70/461/501) while its comparison class in
§4.7 is billed honest. Fix: pick the honest number for every headline (0.151/0.275x flagship; 0.284x
marquee), keep fp16-equivalent as the secondary figure, add an "honest stored" column to Table 3, and
restate the multiples (marquee 1.8-2.6x under Palu/ThinK; flagship 16K 2.7-5.0x under ThinK, 1.8-3.3x
under Palu, 3.6-6.6x under full KV). Add one sentence: "at honest bytes the marquee is byte-matched with
KIVI-4bit (0.284x), which scores vt 1.00 (12/12) at 32K; the marquee's separation is from ThinK/Palu,
not from quantization."

### 3.2 Abstract firmer than body in five places (CPU-only)
(i) line 76-77 "at matched stored bytes ... all three families" -> "Llama/Mistral at matched bytes;
Qwen at 1.7x the 2-bit arm's bytes". (ii) line 77 "multi-key/variable-tracking at 32K on two" ->
"multi-key on Mistral and variable-tracking on Qwen at 32K" (also 948-949, 1005). (iii) line 78-79
"4-bit ... concedes nothing" -> "4-bit ... is not separated from the flagship in any cell (it trails on
point estimate only on Mistral variable tracking, 0.25 vs 0.50)"; same at 159, 660-661, 949-950,
1006. (iv) line 78 "matched or beaten on fluency" -> name the Qwen 32K collapse (35.1 vs 8.2).
(v) line 92-93 "full four-task retrieval at 64K" -> "(n=8; no contrast separates)". Also fix
63-66/144/399-400/997: "3.4-8.8x" and "6.7-12x" at 16K, or cite 32K for the 10x/13x.

### 3.3 The "honestly exclusive band" has no eviction comparator at its bytes (~$10 GPU)
Lines 584-585 and 1007 call the 0.048x cell "the honestly exclusive band". It is exclusive vs
b>=2-bit quantization by construction, but eviction is not floored: `ea-k0.05` at 0.05x was never run,
and at 0.1x on Llama eviction already retains 1.00/0.92/1.00 (16K) and 1.00/0.92/0.92/0.58 (32K)
(Table 4). Given the previous panel's fatal item was precisely "eviction cannot enter, refuted", the
wording should be "exclusive of scalar quantization" until `ea-k0.05`/`snapkv-k0.05` are measured on
Llama/Mistral at 16K/32K, n=12. Also drop "by construction" at 598-599 (contradicted by palu-r0.25).

### 3.4 The composed cell's fluency cost is understated in the abstract and absent from the intro/conclusion (CPU-only)
Llama ppl doubles (9.25/17.1 vs 5.31/8.33, `w19-a1q-llama-lines.txt:1-2`); the abstract says "at a
fluency cost" (86), the intro bullet (160-161) and conclusion (1007) say nothing, and vt (0.50/0.83
Llama, 0.33/0.58 Mistral) is omitted from the abstract. State "roughly 2x perplexity on Llama" and the
vt numbers wherever the 0.048x figure appears. The Qwen negative is disclosed (86, 590-593); keep it.

### 3.5 Perplexity instrument resolution is ~1.6 % per window; several "tie/band" statements are below it (code one-liner + ~$10-20 GPU)
`scripts/w10_frontier.py:69` computes `cross_entropy(..., reduction="sum")` on model-dtype logits; the
harvested per-window NLLs are all integer multiples of 8/511 above 1024 and 4/511 below
(`results/w19_harvest/a4-llama.raw` `[pplw]` rows: 1208, 1168, 800, 1112 ... 14/14 values), i.e. the
window sum is bf16-quantized (ulp 8 at ~1200 nats). Resolution per window = exp(8/511)-1 = 1.6 %; the
byte-identical 8.149 for `ea-k0.1` and `quant-2bit-kivi` at 64K is this quantization, not a copy (the
window NLLs differ, the sums coincide). Consequences: "+0.3 % perplexity" for score-rank (249), "gap
<= 0.006 bits/token" at 16K (514), Llama 16K/32K flagship-vs-2-bit (1.7 %/0.6 %, 669-670), and
"7.27 vs 7.26" at 64K (842) are at or below the instrument. Fix: `.float()` before the sum, rerun the
ppl cells (4 windows x 512 tokens is thin anyway; 16 windows would cost little), and state the
resolution. Mistral/Qwen fluency gaps (10-22 %) are real and unaffected.

### 3.6 The r/n = 0.25 "wall" is confounded with the divergence class the floor fixes (~$10 GPU)
The Llama r256 control (529-532) scores 0.00 on *all four* tasks, including single-needle with a
1024-token exact tier, and has no perplexity line (`g4-llama.raw`). Table 5 shows the same rank class
diverging without the floor (Qwen `bugSseed-r128-h1024` ppl 714 -> 6.94 with floor). A diverged model
also scores 0.00 everywhere, so the control cannot distinguish "the needle stops looking surprising"
(527-528) from numerical blow-up. Run `bugSseed-r256-h1024` (Llama) and the Qwen r128 seeded config
*with* `--min-sv-frac 1e-2`, report ppl and retrieval. If retrieval stays 0 at healthy ppl the wall is
real; if it recovers, the mechanistic claim (abstract 71-72, intro 148, conclusion 1001) must change.

### 3.7 Quant decode path on Mistral variable tracking is unvalidated (~$5 GPU or a grep)
Every quantized arm clusters at 0.25-0.33 on Mistral vt (2-bit 0.33, 4-bit 0.25, **8-bit 0.25**,
`w19-a1-report.md:60-62,69-70`). An 8-bit control should be near full KV. The paper's "decode path is
sound" (640) is validated on Llama and Qwen only. Either report full-KV Mistral vt at 16K (if it is
also ~0.3 the task is degenerate on Mistral and the flagship's 0.50 is not a "weakness"; if it is ~1.0
the quant harness has a Mistral vt defect and Table 6's Mistral vt column is unreliable).

### 3.8 WikiText-filler paragraph compares the flagship's one task against the baselines' four (~$10 GPU or reword)
`w18-g2-qwen-lines.txt:7` is the only flagship row; lines 742-744 conclude "the flagship does not
[collapse]". Either run mk/mv/vt with `--filler wikitext` (Qwen 16K, n=12) or reword to
"holds single-needle; the other tasks were not run under this filler" and lean on the official anchor.

### 3.9 Official-anchor miss-depth interpretation (CPU-only)
Replace 767-769 with the data: "14 needle misses at depths 0.15-0.95, six of them at depth <= 0.20
(uniform expectation 2.8), plus 8 of 12 variable-tracking chains" — a mild early skew is exactly what a
residual warm-up effect would look like, so do not assert its absence on n=14.

### 3.10 Stale text that now contradicts §4.7 (CPU-only)
main.tex:276-281 ("treat a head-to-head against dedicated 2-bit KV quantizers ... as the primary v2
experiment"), 303-304 ("their absence as a baseline is v1's main scoping decision"), 369-371 ("some 32K
composition cells ... are lower-n" — the lower-n cells are now the 64K flagship n=8 and the 8-bit
control n=4), and the header comment 5-8/20. The dangling "Three" at 585 renders in the PDF.

### 3.11 Disclosure: Week-18 q4 invalidation and the first persistence run (CPU-only)
The q4 invalidation lives only in a LaTeX comment (608) and `results/w18-g1-report.md:18-23`. The
files the paper cites at 681 (`results/w19_intervals/a1-*-ruler-intervals.md:10,20`) and the base
line-files (`results/w18-*-lines.txt`, odd rows) still carry `bugS-r64-h256-q4` rows at 0.085x with
1.00/0.67/0.00/0.00 — unannotated, and contradicting the paper's 0.048x cell. Annotate or drop those
rows and add one footnote in §4.6. Likewise the first persistence run measured BUG at **0.4019x**
(`results/w19_harvest/a3-llama.raw.superseded:65`) before a view-serialization fix
(`scripts/w19_persist.py:105-106`, commit 69f7ea9); the corrected 0.1514x matches the independent
accounting (0.151 sbits, 0.150 coldload), so the fix is legitimate, but say so in a footnote. Note
the 2-bit cold start moved 0.088 s -> 0.137 s between the two runs (`superseded:80` vs
`w19-a3-llama2-lines.txt:3`): the flagship-vs-2-bit "tie" is inside run-to-run noise, and the raw
per-repeat timings are not logged (only medians; default `repeats=3` at `w19_persist.py:153`, text
says five).

---

## 4. Underclaims (the paper is entitled to more than it takes)
- 16K multi-key is 12/12 on all three families (`w18-*-lines.txt:2`) but absent from Table 1.
- Qwen 32K flagship vs 2-bit: mv 10/0 (p=0.002) and vt 9/0 (p=0.004) are the strongest paired
  results in the paper (`w19-a1-report.md:131-132`) and get half a sentence (655).
- The Llama 64K point at 0.133x with 8/8 on every task is real evidence for the 1/T story; report
  Wilson [0.68, 1.0] and the paired counts (4/0, 4/0) rather than leaving n=8 to the provenance section.
- `bugEVICT-h256` at 0.009x scores vt 1.00/0.92/0.92 at 32K on Llama/Qwen/Mistral
  (`w18-g3-*-lines.txt:22`) — unreported, mixed (mk/mv collapse), but it is the paper's cheapest
  variable-tracking result and bears on the "wide-KV vt is unsolved" limitation.

## 5. The specific items the panel asked about
- **"No detected loss" non-inferiority phrasing (393-396, abstract 61-62):** correctly hedged —
  Wilson lower bound 0.758 stated, "not a match or a win" stated. Adequate. It would be cleaner to name
  the comparator (full KV, also 12/12) and say the design has power only against losses to <=0.76.
- **Matched-bytes framing (0.151 vs 0.163):** fair on Llama/Mistral (flagship 7 % fewer bytes; the
  quantizer's g=64 grouping is generous to its byte count). Not matched on Qwen (0.275 vs 0.163), and
  the abstract hides that. The marquee is the bigger problem (3.1).
- **Official anchor honesty:** the body (760-772) and Limitations (957-966) state plainly that the
  multi-value edge does not reproduce and that the flagship trails on average; the abstract (80-82) says
  only "indistinguishable". Body honest, abstract soft. One clause fixes it.
- **Sub-cliff fluency cost and Qwen negative:** both disclosed in §4.6 with the right numbers; the
  fluency cost is understated in the abstract and missing from the intro bullet and conclusion (3.4).
- **1/T at 64K (n=8):** numbers verified; n=8 disclosed at 840 and 984 but not in the abstract; no
  contrast separates at n=8 and the paper does not claim one — but "full four-task retrieval" in the
  abstract reads as a result, not a point estimate (3.2 v).
- **Week-18 q4 invalidation:** not disclosed in rendered text; the cited intervals files still carry
  the invalid rows unannotated (3.11).

## 6. Nitpicks
- p<=0.03 (76) is p<=0.032 (Mistral 0.0312).
- bits/token swapped at 513 (think +0.031, palu +0.024).
- 2-bit ratio series is 0.163/0.160/0.158 in the accounting (Table 6, a4) but "0.156-0.158" at 839
  and "0.156" at 899 (persisted bytes) — use one series.
- Mistral 16K flagship 0.151x (Table 6:702) vs 0.150x (`w19-a1-report.md:59`).
- "14 misses" (767) — 22 including vt (`w19-a2-flagship-misses.md:21-28`).
- SnapKV mk range (607) is Llama-only.
- 368 promises per-window NLL; none appears (the `[pplw]` rows exist in the harvest — put them in an
  appendix, it also documents 3.5).
- ThinK also separates from the flagship on official vt (p=0.016, `week19-official-ruler.md:96`);
  766-767 names only 4-bit and Palu.
- Intervals files are titled "Week-18 RULER intervals" for the Week-19 runs.
- "medians of five" (888) vs `repeats: int = 3` default (`w19_persist.py:153`); the pod invocation
  is not in the line-file, so the "five" is unverifiable.
- `results/w19-a3-llama-lines.txt` is an empty tracked file; delete it.

## 7. Highest-leverage fixes
1. **Unify the billing** (3.1 + 3.2): honest bytes in every headline, marquee honest column + the
   KIVI-4bit sentence, abstract strength aligned with the body. CPU-only, ~half a day, and it is the
   difference between "selective billing" and "dual billing" in a reviewer's eyes.
2. **Fix the ppl instrument and re-run the ppl cells** (3.5): one `.float()`, ~$10-20 GPU, then restate
   every sub-2 % fluency claim with the resolution in hand. Cheap, and it pre-empts the one finding a
   numerics-literate reviewer will make from the harvest alone.
3. **Close the two mechanistic holes with one pod** (3.3 + 3.6 + 3.7): `ea-k0.05`/`snapkv-k0.05` on
   Llama/Mistral, floored r256-Llama/r128-Qwen with ppl, full-KV Mistral vt. ~$20 GPU. This decides
   whether "honestly exclusive band" and "r/n = 0.25 wall" stay in the abstract.

---
*Read-only review; no repo edits, no pods launched.*
