# Simulated Program-Committee Review: *Dynamical Low-Rank Approximation as an Online KV-Cache Compressor*

**Simulation setup.** Isolated blind review of the submitted PDF only (no prior context consulted). Five independent PhD-level reviewer agents, each with a distinct lens, followed by an Area Chair agent that read all five reviews plus the paper and independently re-verified the most consequential factual claims. A final verification pass recomputed the statistics and text counts cited below directly from the PDF.

| Role | Lens | Soundness | Presentation | Contribution | Overall (1–10) | Confidence |
|---|---|---|---|---|---|---|
| R1 | Numerical linear algebra / DLRA theory | 2 | 2 | 2 | 4 | 4 |
| R2 | KV-cache systems & baselines | 2 | 2 | 2 | 3 | 4 |
| R3 | Statistics & methodology | 2 | 2 | 2 | 4 | 4 |
| R4 | Writing & presentation | 3 | 2 | 3 | 4 | 4 |
| R5 | Senior reviewer — novelty & significance | 2 | 2 | 2 | 3 | 4 |
| **AC decision** | | | | | **Reject (not yet)** | |

## Verdict in one paragraph

The paper is **not ready** for NeurIPS/ICML/ICLR main track. Every reviewer scored it 3–4/10 with high confidence and the AC concurs. The consensus is unusually coherent: (1) the algorithm in §2.1 is algebraically block incremental SVD (Brand 2006) with V discarded — the DLRA/BUG framing and the σ_min-independent CKL bound in the title, abstract and §1 do not apply to a column stream and no bound is stated; (2) the central method claim (§4.1 — near-oracle, beats iSVD and Oja 1.3–3×) has **no table or figure anywhere in the 21 pages** and is internally inconsistent with §2.1; (3) there is no end-to-end ablation showing the tracker matters — the paper's own evidence (Qwen 32K: gist diverged to ppl 35.1 yet retrieval 1.00) suggests retrieval is carried by the exact tier, which OjaKV's current version already selects by the same residual statistic the paper calls "surprise" and claims as its own; (4) the method saves no resident memory today (≈1.06–1.13× full KV), is slower, and trails a self-implemented, weakened 2-bit KIVI on the only external benchmark (official RULER 0.79 vs 0.87); (5) n=12 cells cannot support "perfect retrieval" (Wilson LB 0.757; a true accuracy of 0.80 passes 12/12 with p=0.07), "not separated" at n=12 is an absence-of-power statement, ~200 cells are reported with one contrast surviving family-wise correction, and the "marquee" is a different, post-pilot configuration beating unmatched baselines that 4-bit KIVI also beats at the same bytes; (6) presentation is far below venue standard (685-word abstract, ~17 pages with no appendix, no algorithm box, "honest" ×31, three memory conventions, double-blind violations, arithmetic slips).

**What is good and must be kept:** the candor and reporting hygiene (paired McNemar, per-trial records, retractions, negative results kept), the rank-siphoning diagnosis and singular-value floor, the r/n≈0.25 wall, the low-rank × quantization composition argument (0.048×), and the stored-vs-resident memory accounting. Reviewers R1, R3 and R5 each said an honestly reframed *analysis* paper built on these would be viewed favorably.

**Realistic odds:** main track ~0–5% now; ~20–35% after the MUST list (hinging on whether the tracker ablation shows the tracker matters); workshop ~50–60% now after a format rewrite, ~85% after MUST. MLSys only with a fused kernel and a measured VRAM/throughput win.

The rest of this document contains the AC meta-review (with the prioritized MUST / SHOULD / NICE fix list) followed by the five full reviews.

---

# Area Chair Meta-Review — NeurIPS 2026

**Paper:** *Dynamical Low-Rank Approximation as an Online KV-Cache Compressor*
**Reviewers:** R1 (numerical linear algebra / DLRA), R2 (systems / KV compression), R3 (experimental methodology & statistics), R4 (writing / presentation), R5 (senior; novelty & significance)

The AC read all five reviews and the submission (abstract, §1–§3, §4.1–§4.9, §5–§6, plus the PDF for §2.1 and the tables) and independently checked the reviewers' most consequential factual claims before writing this. Where the AC re-verified a claim it is marked **[AC-verified]**; where the AC could not verify it under the isolation rule (e.g. contents of the current OjaKV arXiv version, the kvpress press list) it is marked **[not independently verified]** and weighted accordingly.

---

## (a) Summary of the paper

The paper treats the per-layer, pre-RoPE KV tensor of a single sequence as a column-streamed matrix M ∈ ℝ^{n×T} and maintains an orthonormal rank-r basis U (plus a small square-root core B) that is updated as blocks of new tokens arrive, via a step the authors describe as the rank-adaptive basis-update & Galerkin (BUG) integrator of Ceruti–Kusch–Lubich. Tokens are stored as r-dimensional coordinates ("gist") instead of n-dimensional vectors. The gist is combined with (i) attention-sink tokens, (ii) a verbatim recent ring, (iii) an "exact tier" of up to h tokens selected by *surprise* — the relative residual ‖k − UUᵀk‖/‖k‖ — and (iv) a "warm-up seed" that routes the first ingest chunk through the surprise path. A relative singular-value floor is proposed to stop high-rank divergence.

Claimed results: near-oracle reconstruction (1.01–1.03× of truncated SVD; Oja's rule 1.3–3× worse) on Llama-3.2-1B KV (§4.1); 12/12 single- and multi-value retrieval at 16K on Llama-3.1-8B, Qwen2.5-7B and Mistral-7B at 0.15–0.28× fp32-at-rest (0.085–0.149× fp16-equivalent) stored state (Table 1); a paired-McNemar win over ThinK-c0.5 on 32K Llama variable tracking using a different, larger configuration (Table 3); a matched-bytes comparison against a self-implemented KIVI-scheme 2-bit/4-bit quantizer that is mixed (Table 7); an official-RULER anchor where the flagship trails the 2-bit arm (0.79 vs 0.87, Table 8); a "rank-siphoning" diagnosis and an r/n ≈ 0.25 retrieval wall (§4.4); a composed gist+4-bit-coordinates cell at 0.048× (§4.5); and an explicit accounting that decode-time resident memory is ≈1.06× full KV because the method reconstructs then attends (§4.9), with the only latency datum (1B, CPU) ≈10% slower. The concrete systems payoff is a ~7× smaller persisted cache and 9–10× faster warm reload, which the paper concedes is shared with 2-bit quantization.

---

## (b) Consensus strengths

All five reviewers agree on the following, and the AC concurs:

1. **Exceptional candor and reporting hygiene.** The paper retracts an earlier "beats SOTA at 10×" claim (§4.8), fixes baseline port bugs in the baselines' favor (§4.3), keeps the official-RULER loss (Table 8), reports the 1.06× decode residency and the 10% slowdown, and releases per-trial records with SHA-pinned provenance. Every reviewer names this as a model for the field.
2. **Correct paired statistics.** Wilson intervals and exact McNemar on shared needles are the right tools; R1 and R3 independently recomputed every interval and p-value and found them correct. **[AC-verified]** — 12/12 → [0.758, 1]; 15/16 → [0.717, 0.989]; 5/16 → [0.142, 0.556]; 10/0 discordant → p = 1.95×10⁻³; 6/0 → p = 0.031.
3. **The right competitors were eventually run.** A matched-bytes 2-bit quantizer under the same streaming protocol (§4.6), the eviction×quantization composite (Table 5), and an official RULER anchor (§4.7) are the comparisons a skeptical reader would demand, and they were included even though they cut against the paper.
4. **The rank-siphoning diagnosis (§4.4, Table 4)** is a genuinely interesting mechanistic observation — content-selected exact tiers pull rank-carrying columns out of the tracked subspace — and the singular-value floor convincingly removes the divergence class within-method (27,531 → 6.995 perplexity).
5. **Stored-state vs. resident-memory separation (§4.9) and the measured cold-start path (Figure 3)** are honest systems accounting that most KV-compression papers omit. The 1/T amortization argument is correct and clearly presented.
6. **The composition observation (§4.5)** — low-rank compresses the feature axis, scalar quantization the bit axis, and they multiply to reach 0.048× — identifies an operating point a fixed-bit scalar quantizer cannot be configured to reach, even if the current cell costs ~2× perplexity and fails on Qwen.

---

## (c) Consensus critical weaknesses, ranked by decisiveness

### C1. The paper's title claim — "DLRA / BUG integrator" — is not what the algorithm is, and the central method evidence (§4.1) does not exist in the paper. (R1, R5; R2 and R4 partially) — **decisive**

**[AC-verified]** The update in §2.1 (A = UᵀC; R⊥ = C − UA = QR; B_aug = [B A; 0 R]; SVD-truncate) satisfies B_aug B_augᵀ = [U|Q]ᵀ [M C][M C]ᵀ [U|Q]: it is exactly the block incremental SVD of Brand (2006) applied to the column space with V discarded, followed by rank truncation. The paper concedes the rank-one case but the identity holds for every block size b. There is no matrix ODE, no vector field F, no time step, and hence no counterpart to the CKL "σ_min-independent" bound that the abstract and §1 headline; the paper itself admits the near-oracle quality "is a measurement on real KV, not a theorem for this column-stream setting" (§2.1). The relative singular-value floor (§2.4) is the classic rank-revealing relative tolerance, not a CKL-θ variant (CKL's criterion is an absolute Frobenius-tail bound). No error bound of any kind is stated or proved.

**[AC-verified]** §4.1 is a single paragraph. The claims "1.01–1.03× of truncated SVD at every rank," "beats fixed-rank incremental SVD everywhere," and "beats Oja's rule by 1.3–3.0×" appear in no table or figure anywhere in the 21 pages; there is no error metric definition, no rank sweep, no context length, no Oja configuration (step size, initialization, per-head vs. shared basis, passes), and no definition of the iSVD baseline. Worse, "beats fixed-rank incremental SVD everywhere" is internally inconsistent with §2.1 + §2.4, which together state that with θ off the rank-adaptive step pads to the rank cap — i.e. the tracker *is* fixed-rank block iSVD. Either the two implementations differ in an undisclosed way, or the statement is wrong.

Consequence: the single distinctive component the paper claims over OjaKV (the tracker) has no reported evidence, and the paper's identity is a renaming of a known streaming-PCA primitive. This alone is disqualifying for a main-track method paper.

### C2. No end-to-end evidence that the tracker matters, and the retrieval results appear to be carried by the exact tier. (R2 Q9, R3 W11, R5 W3; R1 W7) — **decisive**

The paper states "perplexity rides mostly on the gist; needle retrieval rides mostly on the exact tier" (§2.3). On Qwen at 32K the r=64 gist has *diverged* (perplexity 35.1 vs 8.2) yet retrieval is 1.00 (Table 2, §4.6) — retrieval is evidently carried by the verbatim tier + ring, i.e. a surprise-scored eviction policy. There is no ablation in which the same cache (tier + sinks + ring + seed) is run with (a) an Oja tracker, (b) a plain truncated-iSVD tracker, (c) a frozen one-shot prefill SVD (xKV/ShadowKV-style), or (d) no gist at all, on the 7B/8B retrieval grid. Without this, nothing in Tables 1–3 or 7–8 can be attributed to the paper's distinctive component.

### C3. The method is not, by its own measurement, a KV-cache compressor in the deployed sense, and it does not beat a plain 2-bit quantizer at matched bytes on the only external benchmark. (R2 W1, R5 W4–W5; R1, R3 agree) — **decisive for significance**

Decode residency ≈1.06× full KV (§4.9; ≈1.13× under the paper's own preferred fp32-at-rest basis, as R4 notes **[AC-verified]**: 0.982 + 0.151 = 1.133); batch size 1; no GPU decode latency or TTFT at 8B; 1B-CPU +10% slower. Every competitor in Tables 7–8 reduces resident VRAM today. On official RULER (Table 8) the flagship is 0.79 vs 0.87 for a *self-implemented, coarser (G=64, chunked-prefill)* 2-bit arm at the same bytes, and 0.33 vs 1.00 on variable tracking against 4-bit KIVI, ThinK and Palu (p ≤ 0.016). The persistence win over full KV (0.13 s vs 1.23 s) is shared with the 2-bit cache (0.14 s) to within 0.01 s. What remains distinctive is a 19% lower storage asymptote (0.125× vs 0.156×) realized only with an fp32 gist, and the 0.048× composed cell at ~2× perplexity on two of three families.

### C4. Statistical power and framing. (R3 primarily; R1, R2, R5 concur) — **strongly negative but fixable**

n=12 per cell means 12/12 admits a true accuracy of 0.76; "perfect … retrieval" (abstract, §1, §6) describes the sample, not the population. "Not separated" at n=12 requires ≥6 one-way discordant pairs to reach p<0.05, so the headline "4-bit at twice the bytes is not separated in any cell" is an absence-of-power statement presented as parity — and is literally contradicted by Table 8 (4-bit vs flagship on vt: 1.00 vs 0.33, p=0.008) **[AC-verified]**. Roughly 200 accuracy cells are reported (R3's count is credible from the tables) vs. the paper's "∼40"; under Holm over the 24 flagship-vs-2-bit contrasts in Table 7 only Qwen-32K multi-value survives; under BH, 5 of 7 **[AC-verified arithmetic]**. Perplexity numbers (n=4–8 windows) carry no uncertainty. The 64K cells (n=8) are compared against n=12 comparators without a paired test.

### C5. The "marquee" is a different configuration from the flagship, selected in a way that reads as post hoc, and it beats only weak, unmatched baselines. (R1 W13, R2 W7, R3 W4, R4 W7, R5 W6) — **[AC-verified]**

Table 3 uses bugSseed-r128-h1024-s32 at 0.284× (1.9× the flagship's bytes) with the -ss knob that §2.3 says was introduced precisely because it "recovers Llama's r=128 retrieval." §4.2 says the 32K cells "firm what were previously n=4 point estimates," so the arm and hypothesis followed a pilot on the same task and model. The pre-registration is "in the pod matrix," not externally timestamped. The win is over ThinK-c0.5 (0.75×) and Palu-r0.5 (0.50×) at 1.8–2.6× *more* bytes — not a Pareto comparison — and 4-bit KIVI at the same 0.284× scores 1.00. The "KIVI 4-bit 1.00 [0.76, 1.0]" row in Table 3 is an n=12 arm from Table 7, not the paired n=16 run, and the cross-reference to "Table 8" for the 32K 4-bit number is wrong (Table 8 is 16K only). The marquee therefore establishes that ThinK-alone is bad at variable tracking, not that the method is good.

### C6. The OjaKV differentiation is stale and OjaKV is not run as a baseline. (R1 W4, R2 W4, R5 W1) — **[not independently verified; three reviewers with cited sources agree]**

The paper claims to differ from OjaKV "in selecting the exact tier by content (surprise) rather than by position" (§3, §1). R1, R2 and R5 each independently report that the current OjaKV version (arXiv v2 / Findings of ACL 2026) retains full-rank tokens by the reconstruction residual r_t = k_t − U Uᵀ k_t — the same statistic. If so, the paper's claim that "the content-selected exact tier … [is] ours" (§1) is false and the only remaining delta is the tracker (see C1, C2). OjaKV is compared only as an un-tabulated reconstruction curve at 1B and never on a downstream task. The paper cites [42] without a version; the authors must state which version they compared against.

### C7. Presentation is far below venue standard. (all five; R4 in detail)

**[AC-verified]** The abstract is 681 words with ~60 numbers and three memory-billing conventions; "honest" appears 31 times, "flagship" 59, "marquee" 10; main text runs ~17 pages with no appendix; commit SHAs, repo paths, pod-hours and dollar cost sit in the body; there is no algorithm box and no mechanism figure; the same configuration is billed 0.085× (Table 1) and 0.151× (Table 7) with no cross-reference; author names, a GitHub URL, a date and "arXiv v1" versioning language appear on the title page (a double-blind violation). Arithmetic: "6.7–13× less than full KV" in a sentence about 16K mixes the 16K (0.149×) and 32K (0.075×) values **[AC-verified]**. Three different verdicts on the palu contrast (separates / suggestive / not separated) appear in §4.3, abstract and §6.

---

## (d) Points of disagreement and AC adjudication

**D1. Is the DLRA framing legitimate at all?** R1 and R5 say the framing is "borrowed prestige" / "rebranding"; R2 and R4 credit it as "a genuinely new tool imported into the KV-compression space" / "crisp, memorable framing." **Adjudication: R1 and R5 are correct on substance.** The AC verified the Gram-update identity. The DLRA lens did motivate a rank-adaptive tolerance and the relative floor, which is a legitimate (small) contribution, but the title, abstract and §1 sell a theorem that does not apply and a distinction from iSVD that does not exist. R4's Strength 1 ("σ_min-independence argument … stated well") and Strength 2 ("positioning against OjaKV is explicit and fair") are outside R4's declared expertise and are contradicted by the three domain reviewers; the AC discounts them.

**D2. Is rewriting alone enough?** R4 believes a restructuring "would yield a paper in the 5–6 range on the same evidence" and gives Soundness 3 / Contribution 3 (the only reviewer above 2 on either). **Adjudication: R4 is too optimistic.** The missing §4.1 table, the missing tracker ablation (C2), and the stale OjaKV differentiation (C6) are evidence gaps, not writing problems. R4's presentation diagnosis and rewrite plan are, however, the single most useful practical document the authors will receive and should be followed nearly verbatim.

**D3. "Pareto-dominated by 4-bit KIVI on every model tested" (R2 Contribution justification).** **Adjudication: overstated.** 4-bit sits at ~2× the flagship's bytes on Llama and Mistral; "dominated" requires ≤ bytes. Only on Qwen (where the n=512 gist costs 0.275× ≈ 4-bit's 0.284×) is the flagship dominated at matched bytes — which the paper says itself. R2's substantive point (no model on which the method is Pareto-competitive with 4-bit across all tasks + perplexity) survives; the wording does not.

**D4. "A baseline was run and its results withheld" (R5 W7, ShadowKV).** **Adjudication: legitimate question, harsh wording.** §4.3 mentions fixing "a ShadowKV decode-scope bug" and ShadowKV appears in no table **[AC-verified]**. The authors owe an explanation, but "withheld" presumes intent the record does not establish.

**D5. Size of the abstract.** R5 says ~900 words; R3/R4 say ≈680. **[AC-verified: 681.]** Immaterial to the decision but noted for accuracy.

**D6. Palu port.** R2 asserts kvpress has no Palu press, so the Palu baseline is the authors' own port. **[not independently verified]**, but consistent with the paper's own admission of "a Palu attention-sink low-ranking" port defect (§4.3). Palu scoring 0.42 on official multi-value while ThinK at fewer compressed bytes scores 1.00 (Table 8) is suspicious and supports R2's concern. The authors should state exactly which Palu variant is implemented.

**D7. Test-set tuning (R3 W5).** R3 argues every fix (seed, -ss, floor) was developed and evaluated on the same cells. **Adjudication: largely fair, with one mitigation.** The official RULER anchor functions as a partial held-out test, and on it the in-house edge did *not* reproduce — which is exactly the pattern R3 predicts and thus strengthens rather than weakens the concern. The seed and floor are stability/architecture fixes rather than hyperparameter sweeps, so "test-set tuning" is slightly strong, but the absence of any held-out generator/filler set is real.

**D8. The coordinate-consistency question (R1 W5–W6).** Only R1 raises it: stored coordinates c_t = U_tᵀ m_t are expressed in the basis current at ingest, but U changes every block; reconstructing with the final U is wrong unless coordinates are rotated (O(r²T) per step, compounding loss) or old bases are kept. The paper is silent **[AC-verified]**. R1 further observes that a projection with orthonormal U satisfies ‖UUᵀM‖ ≤ ‖M‖ and therefore *cannot* produce perplexity 27,531 by itself — so the divergence in Table 4 must live in orthogonality loss, coordinate staleness, or bf16 leakage rather than "ODE-style stiffness." **Adjudication: the AC finds this the sharpest unresolved technical point in the reviews.** It is not proven to be a bug, but the authors must answer it (report max‖UᵀU − I‖ over the stream with the floor off, and state how stored coordinates are handled across basis updates). It also bears on what §4.1's "1.01–1.03×" measures — the final basis, or the actually-stored cache.

---

## (e) Decision and score table

| Reviewer | Expertise | Soundness | Presentation | Contribution | Overall | Confidence |
|---|---|---|---|---|---|---|
| R1 | Numerical LA / DLRA | 2 | 2 | 2 | 4 (borderline reject) | 4 |
| R2 | Systems / KV compression | 2 | 2 | 2 | 3 (reject) | 4 |
| R3 | Methodology / statistics | 2 | 2 | 2 | 4 (borderline reject) | 4 |
| R4 | Writing / presentation | 3 | 2 | 3 | 4 (reject in current form) | 4 |
| R5 | Senior / novelty | 2 | 2 | 2 | 3 (reject) | 4 |
| **Mean** | | **2.2** | **2.0** | **2.2** | **3.6** | **4.0** |

**Decision: Reject.**

No reviewer recommends acceptance; the two most positive (R1, R3) are at borderline-reject and both state explicitly that they cannot recommend acceptance as submitted. The consensus is unusually coherent across five very different lenses: the paper is a carefully executed, candidly reported empirical study whose (i) mathematical identity is misdescribed, (ii) central method claim has no reported evidence, (iii) distinctive component is never shown to matter end-to-end, (iv) deployed benefit is absent today, and (v) matched-bytes edge over a plain 2-bit quantizer does not survive the one external benchmark. Every reviewer also states that a version with the MUST items below could be a solid contribution, and the AC agrees. This is a "not yet," not a "never."

The AC wants to be explicit that the authors' honesty was *not* held against them. Several reviewers noted that the candor made the gaps easier to see; that is the correct outcome of candor, and the field is better for it.

---

## (f) Path to acceptance

### MUST (blocking — the paper cannot be accepted anywhere main-track without these)

1. **Publish §4.1 as a table/figure and reconcile it with §2.1.** Reconstruction error vs. rank (r ∈ {16, 32, 64, 128, 256}) for: truncated-SVD oracle, this tracker (θ on / θ off), block iSVD at the same block size and precision, and Oja with a stated step-size schedule, initialization, per-head vs. shared basis, and number of passes. Report on at least one 7B/8B model at 16K–32K, not only 1B at ctx 1024. Define the metric (‖M − M̂‖_F / ‖M‖_F, and whether M̂ uses the *stored* coordinates or the final basis). State in linear-algebra terms what differs between "the tracker with θ off" and "fixed-rank iSVD"; if nothing, remove "beats fixed-rank incremental SVD everywhere."

2. **Reframe the method honestly.** Retitle (e.g. "Online rank-adaptive subspace tracking with a surprise-selected exact tier for KV-cache compression"). In §2.1 say plainly that on a column stream the update is block incremental SVD / a Frequent-Directions-style deterministic sketch, that the CKL bound does not apply, and either (a) state and prove an applicable bound (Frequent Directions: ‖MMᵀ − BBᵀ‖₂ ≤ ‖M − M_k‖_F²/(ℓ − k), adapted to the rank-adaptive truncation), or (b) claim no bound. Rename the "CKL-θ variant" as what it is: a relative numerical-rank tolerance. Delete "provably" from §5 or supply the proof.

3. **Run the tracker/gist ablation on the 7B/8B grid and on official RULER.** Same cache (sinks + ring + h=256 surprise tier + seed) with: (a) this tracker, (b) Oja tracker, (c) truncated-iSVD tracker, (d) frozen one-shot prefill SVD at the same r, (e) *no gist* (tier-only + ring). Report Tables 1, 2, 7-equivalents and Table 8. If (a)–(e) tie on retrieval, the paper must be reframed as an analysis of surprise-selected exact tiers, with the low-rank gist justified by perplexity/bytes only.

4. **Resolve the coordinate-consistency and divergence mechanism question (R1 W5–W6).** State how stored coordinates are handled when U changes (rotated per step, recomputed, or stale). Report max_t ‖U_tᵀU_t − I‖_F over the stream with the floor off on the Table 4 cells, and identify mechanically where perplexity 27,531 originates. Report the error of the *stored cache* (not the final U) against the SVD oracle.

5. **Fix the OjaKV positioning and run it as a baseline.** Cite the specific OjaKV version compared against; if the current version selects the exact tier by reconstruction residual, delete the "content vs. position" differentiation and the "the content-selected exact tier … [is] ours" claim. Run OjaKV end-to-end at matched bytes on at least Llama-3.1-8B 16K/32K (in-house grid and official RULER). Report the ShadowKV numbers that §4.3 implies were run, or explain their absence.

6. **One memory convention, and one that corresponds to bytes actually stored.** Use fp32-at-rest everywhere in tables and headline claims (0.15×/0.28×), mention fp16-equivalent once in §4.9 with the explicit statement that an fp16 gist is unvalidated. Recompute all "N× less than" ratios from one convention and one context (fix "6.7–13×"). Report decode residency in the same convention (≈1.13×, not 1.06×). Say in the abstract that "stored state" is not peak decode VRAM.

7. **Rewrite to venue format.** ≤250-word abstract stating one contribution and the experiment that establishes it (R4's 160-word draft is a good start once C1/C6 are reflected); ≤9 pages main text + appendix; Algorithm 1 for the block update, surprise scoring, tier admission/eviction, seed and score-rank cap; a mechanism figure (sinks | ring | gist | tier); a config-name → (r, h, seed, s, floor, n_sink, W) table with values for the flagship; remove author names, GitHub URL, date, "arXiv v1," "earlier draft," week labels, SHAs, pod-hours from the main text; delete every "honest(ly)," "marquee," "crux"; fix broken cross-references (§4.3 → Table 7 not Table 8; §4.9 self-reference; "Two lines" → three).

8. **Statistical language must match power.** Remove "perfect" everywhere; write "12/12, Wilson LB 0.76." Never present "not separated" at n=12 as parity in a headline; state the minimum detectable effect (≥6 one-way discordant pairs). Reconcile "4-bit not separated in any cell" with Table 8's p=0.008 vt separation. Report the true number of cells (~200) and apply Holm/BH across all flagship-vs-baseline paired contrasts in Tables 7–8, claiming only the survivors as wins. Add SE/CI or a paired test for every perplexity comparison. Use one verdict for the palu contrast everywhere.

### SHOULD (strongly raises odds)

9. **Faithful quantization baselines.** KIVI at G=32 with full-precision prefill (its published operating point) and KVQuant (pre-RoPE, dense-and-sparse) at 2–3 bits; re-run Table 7 and report whether any multi-value separation survives.
10. **Matched-budget eviction and structured baselines.** SnapKV / PyramidKV / H2O at k ≈ 0.15 (the flagship's bytes) on the official anchor; ThinK and Palu swept down to 0.15–0.28× so Table 3 becomes a Pareto comparison. State exactly which Palu variant the port implements and verify it reproduces Palu's published LongBench numbers at 50%.
11. **Official benchmarks at the scale the field expects.** Full RULER (all 13 tasks) at 16K and 32K on all three families with ≥100 samples/task (official uses 500); full LongBench (all tasks) for the flagship on at least Llama-3.1-8B; drop the single-document "probe."
12. **Increase n or model the variance.** n ≥ 50 per cell for headline contrasts, or a hierarchical model over seeds × depths; report the per-seed × per-depth breakdown (depth is first-order for a method with a warm-up window and a recent ring). Use a held-out filler pool and a held-out model family for every stabilizer (seed, -ss, floor).
13. **Externally timestamp any pre-registration** (OSF/AsPredicted or a signed, dated commit) and state whether marquee trials exclude the pilot. Make the flagship and the confirmatory configuration the same configuration, or explain why a practitioner should deploy two.
14. **Compute accounting at 8B.** Per-block update cost (QR, core SVD, basis rotation, coordinate handling) and per-decode-step reconstruction cost in ms and FLOPs; peak decode VRAM and tokens/s at 16K/32K on the A100 for full KV, flagship, KIVI-2, KIVI-4, ThinK head-to-head. Estimate the FLOP cost of a fused kernel that must re-apply RoPE after reconstruction vs. a post-RoPE r-dim kernel (R1 W8).
15. **Report the missing cells.** Flagship on WikiText filler for all four tasks and all three families (§4.7(1) reports only Qwen single-needle); Qwen 32K with the floor on at r=64 and at the "natural higher rank"; the r/n wall as a curve (retrieval vs. r/n, three families) with and without -ss, resolving R1 W11.

### NICE (polish / strengthens the story)

16. Validate an fp16-storable gist (or fp16 coordinates with fp32 U and B) — this would halve the honest bytes and change the quantizer comparison materially; the paper already identifies it as the most valuable follow-up.
17. A fused factored-attention kernel with a measured resident-memory win at batch > 1. This would turn the paper into a systems contribution and is the natural MLSys path (see (g)).
18. Test the composed 0.048× cell against TurboQuant/KVQuant as pure 2–3-bit quantizers and against SnapKV×KIVI composites, and give it an official-anchor number.
19. Cite and position against StreamingKV (OpenReview ZFGtDo7RKI) if it is prior art for the online per-sequence low-rank axis (R5 W8), and against streaming-PCA finite-sample theory (Oja/Krasulina analyses, Frequent Directions).
20. Reference-list cleanup (venue fields, "et al." consistency); Figure 1 legend/color mismatch, Figure 2 annotation overlap, Table 5 as a grid; footnote placement.

---

## (g) Venue recommendation

| Venue | Odds, current form | Odds after MUST items (1–8) | Odds after MUST + SHOULD |
|---|---|---|---|
| NeurIPS / ICML / ICLR main track | ~0–5%. Five reviewers at 3–4 with confidence 4; no champion. | ~20–35%. Depends heavily on what item 3 shows. If the tracker ablation shows the BUG/iSVD tracker is *necessary* for task-level results and the §4.1 table holds at 8B, the paper has a real method contribution and lands borderline. If the trackers tie, the method contribution collapses to the exact tier (OjaKV's) and main-track odds stay low regardless of polish. | ~45–60% if a separated win over faithful 2-bit at matched bytes appears on official RULER, or a resident-memory win exists. Otherwise ~25–35% as a well-executed analysis/negative-result paper, which main tracks accept rarely. |
| MLSys | ~5%. No kernel, batch 1, 1.06–1.13× residency, no GPU latency. | ~10–15%. Format fixes do not supply the systems result MLSys needs. | ~50%+ *only* with item 17 (fused kernel, measured VRAM/throughput win, batch > 1) plus a serving-context persistence study (prefix-cache hit rates, multi-tenant). Without a kernel, not a fit. |
| Workshop (efficient inference / long-context / ES-FoMo-style) | ~50–60% even now, after the format rewrite (item 7) and honest reframing (item 2). The rank-siphoning diagnosis, the r/n wall, the composition argument and the stored-vs-resident accounting are exactly what a workshop audience values. | ~85%+. | — |
| arXiv-only | Appropriate today; the paper describes itself as "an arXiv v1 timestamp." Do items 2, 6, 7 and 8 before the next version so the arXiv record does not carry the misdescription. | | |

**AC recommendation:** Do the MUST list, submit the mechanistic core (rank-siphoning, r/n wall, composition, stored-vs-resident) to a workshop now to establish priority with a corrected framing, and decide between a main-track resubmission and MLSys based on the outcome of item 3 (does the tracker matter?) and whether the authors intend to build the kernel (item 17). ICLR is marginally friendlier than NeurIPS/ICML to careful analysis papers with narrow empirical edges; if the tracker ablation ties, target ICLR with an explicit "analysis" framing rather than a "method" framing.

---

## (h) What is actually good and worth keeping

The authors should not read this decision as "start over." The following are real and should survive intact into any resubmission:

- **The surprise-selected exact tier coupled to a tracked subspace, and the diagnosis of what that coupling does.** The rank-siphoning mechanism (§4.4) — content selection removing exactly the columns that carry rank, starving the tracker until it pads with near-null directions — is a genuine, non-obvious observation about *any* residual-scored hybrid cache, including OjaKV-style ones. The relative singular-value floor is the right pragmatic fix even though it should be named for what it is. The r/n ≈ 0.25 wall, once shown as a curve with and without -ss, is a useful design rule for the whole hybrid low-rank family.
- **The composition argument and the 0.048× cell.** That a rank-r coordinate stream can be scalar-quantized independently of the basis, pre-RoPE, and reach a byte band a fixed-bit quantizer cannot, is a correct and useful structural point. It needs an official-anchor number and a fluency story, but it is the one place the method occupies unique territory.
- **The stored-state vs. resident-memory distinction and the persistence measurement.** Separating the two, measuring the 0.98× reconstruction workspace, and reporting the cold-start path on the same card are the honest systems accounting the field should adopt. The 1/T amortization analysis is correct and cleanly argued. This part of the paper needs no new experiments, only consistent units.
- **The experimental hygiene.** Paired McNemar on shared needles, Wilson intervals on every cell, per-trial records, SHA-pinned provenance, retractions and baseline fixes in the baselines' favor, and an official anchor kept even though it lost. Every reviewer singled this out. Keep every bit of it; move the provenance to an appendix, and let the intervals — not the adjective "honest" — carry the credibility.
- **Pre-RoPE tracking as the operating point.** Not new (KVQuant, ShadowKV), but correctly chosen and correctly justified; R1's FLOP concern about the deferred kernel is a design trade-off to state, not a reason to abandon it.
- **The negative results themselves.** "A plain 2-bit quantizer under a streaming protocol matches or beats an online low-rank tracker at matched bytes on official RULER, and eviction collapses below 0.1× on essay haystacks" is information the community needs. If item 3 shows the tracker does not matter for retrieval, an honestly framed analysis paper saying so — with the mechanistic map above — is more valuable than most incremental method papers, and reviewers R1, R3 and R5 each said they would look favorably on that reframing.


---

# Appendix: Full Reviews

# Review — Reviewer 1 (numerical linear algebra / DLRA)

**Paper:** *Dynamical Low-Rank Approximation as an Online KV-Cache Compressor*

## Summary

The paper proposes tracking the per-layer, per-sequence KV matrix M ∈ ℝ^{n×T} (features × tokens) with a rank-r orthonormal basis U updated online as tokens stream in, storing r-dimensional coordinates per token instead of n, and pairing this "gist" with (i) attention-sink tokens, (ii) a recent ring, (iii) an exact tier of up to h tokens selected by residual norm ‖k − UUᵀk‖/‖k‖ ("surprise"), and (iv) a "warm-up seed" that routes the first ingest chunk through the surprise path. The tracker is described as a "streaming rank-adaptive BUG integrator" for DLRA and operates on pre-RoPE keys. The authors report near-oracle reconstruction versus truncated SVD on Llama-3.2-1B KV (§4.1, text only), 12/12 retrieval on single/multi-value needles at 16K on three 7–8B models at 0.085–0.149× fp16-equivalent (0.15–0.28× fp32-at-rest) stored state (Table 1), a McNemar-separated win over ThinK-c0.5 on 32K variable tracking on Llama (Table 3), a matched-bytes comparison against a KIVI-scheme 2-bit quantizer (Table 7) that is mixed, an official RULER anchor where the method trails the 2-bit arm (0.79 vs 0.87, Table 8), a "rank-siphoning" diagnosis fixed by a relative singular-value floor (Table 4), and an explicitly stated systems limitation: decode residency ≈ 1.06× full KV because the method reconstructs then attends.

The paper is unusually candid about its negative results and the empirical program is large. My concerns are with the mathematical framing, the absence of any theory that applies to the stated setting, the incompleteness of the central §4.1 claim, and the fairness/attribution of the OjaKV comparison.

## Strengths

1. **Candor.** The paper reports retractions of earlier claims (§4.8), fixes made "in the baselines' favor" (§4.3), the official RULER result where it loses (§4.7, Table 8), the 1.06× residency (§4.9), and the Qwen divergence. This is rare and valuable.
2. **Paired statistics.** Wilson intervals and exact McNemar on shared needles are the right tools; I verified the reported intervals ([0.758,1] for 12/12; [0.72,0.99] for 15/16; [0.14,0.56] for 5/16) and p-values (2·0.5¹⁰ = 1.95×10⁻³; 2·0.5⁶ = 3.1×10⁻²). The pre-registration of a single confirmatory contrast is good practice.
3. **The matched-bytes 2-bit baseline** (§4.6) and the eviction×quantization composite (Table 5) are the right competitors, and running them under the same streaming protocol is fair.
4. **Separating stored state from resident memory** (§4.9) and measuring the cold-start path (Figure 3) is honest systems accounting that many KV-compression papers omit.
5. The observation that low-rank coordinates compose multiplicatively with scalar quantization to reach 0.048× (§4.5) is a genuinely interesting operating point, even if it costs fluency.

## Weaknesses

**W1. The "streaming BUG integrator" is incremental SVD; the DLRA framing is borrowed prestige.** The algorithm in §2.1 — A = UᵀC, R⊥ = C − UA = QR, B_aug = [B A; 0 R], SVD-truncate — is exactly the block incremental SVD of Brand [2] / the sequential Karhunen–Loève update, with the right factor V discarded. One can check directly that B_aug B_augᵀ = [U | Q]ᵀ [M C][M C]ᵀ [U | Q], so the step is the exact Gram update followed by rank-r truncation; nothing specific to BUG (the K-, L-, and Galerkin S-steps applied to a vector field F(t,Y)) survives, because there is no F. The paper concedes the rank-one case ("reduces to the classical incremental-SVD update," §2.1) but the reduction holds for every block size b, and Brand's algorithm is a block algorithm. What the paper says the DLRA framing "adds" — the rank-adaptive truncation criterion and "the σ_min-independent error bound" — is either a one-line tolerance on a small SVD that incremental-SVD implementations already have, or a theorem that does not apply (W2). Renaming incremental SVD as a DLRA integrator, titling the paper accordingly, and claiming "we appear to be the first to use [DLRA] as an inference-time, streaming compressor" (§3) overstates the connection.

**W2. No theory is stated or proved, and the theory that is invoked does not apply.** The abstract and §1 motivate the method by the Ceruti–Lubich / CKL error bound "independent of the smallest tracked singular value." That bound (CKL Theorem 2) is ‖Y_n − A(t_n)‖ ≤ c₀δ + c₁ε + c₂h + c₃nϑ for a matrix ODE Ȧ = F(t,A) with F Lipschitz and bounded, ε the normal component of F, and h the step size. In a discrete column stream there is no F, no h, no tangent-space normal component; "stiffness" arises in the ODE setting from inverting S in the projector-splitting/K-S-L equations, and no inverse is ever formed in the algorithm of §2.1. A projection UUᵀM with orthonormal U cannot "explode" regardless of how small the discarded singular values are. The paper half-acknowledges this ("the near-oracle quality we report (§4.1) is a measurement on real KV, not a theorem for this column-stream setting," §2.1) yet still headlines the bound in the abstract and introduction. Meanwhile, theory that *does* apply to exactly this setting exists and is not used: the Frequent Directions guarantee (Liberty [20]; Ghashami, Liberty, Phillips, Woodruff 2016) gives ‖MMᵀ − BBᵀ‖₂ ≤ ‖M − M_k‖_F²/(ℓ−k) for a deterministic column-stream sketch of this shape, and incremental-SVD error analyses exist. As written, the paper has zero stated bounds; "near-oracle" is purely empirical (and see W3).

**W3. The central §4.1 claim has no table, figure, or protocol.** The 1.01–1.03× near-oracle ratio, the "beats fixed-rank incremental SVD everywhere," and "beats Oja's rule by 1.3–3.0×" are stated in one paragraph with no numbers per rank, no context length, no definition of the error metric, no description of how the Oja baseline was configured (step size, initialization, orthonormalization frequency, per-head vs shared basis, number of passes), and no code pointer. I also cannot reconcile "beats fixed-rank incremental SVD everywhere" with §2.1 + §2.4, which together say that with θ off "the rank-adaptive step always pads the basis to the rank cap," i.e. the tracker *is* fixed-rank block incremental SVD. Either the two implementations differ in some undisclosed way (numerics? truncation before vs after augmentation?) or the statement is inconsistent. Further, the near-oracle measurement is on a 1B model (n = 512) and, if it uses the §4.8 setup, at ctx 1024 — not the 16K–64K, n = 1024 regime the paper's claims live in. Finally, it is unclear *what* is being compared to the oracle: the final basis U_T's projection error ‖M − U_T U_Tᵀ M‖_F, or the error of the actually-stored cache, where each token's coordinates were computed against the basis current at ingestion time (see W5).

**W4. The OjaKV comparison is unfair as described and the differentiation is misattributed.** (a) §3 claims the paper differs from OjaKV "in selecting the exact tier by content (surprise) rather than by position." This is true of OjaKV arXiv v1 (first n_start and last n_recent tokens), but the current OjaKV version (arXiv v2; ACL Findings 2026) selects full-rank tokens by the residual r_t = k_t − U_kU_kᵀk_t weighted by recent queries — i.e. essentially the same surprise statistic this paper calls "ours" ("the content-selected exact tier … are ours," §1). The paper must either cite the version it compares against and acknowledge the later one, or drop the differentiation claim. (b) OjaKV uses per-head bases (rank relative to d_h = 128, chosen by a 0.99 energy criterion, at 0.6–0.8× compression), prefill initialization on salient tokens, and step sizes 0.10/0.05. This paper stacks all KV heads into n = HD = 1024 and uses a shared r = 64 basis. "1.3–3× worse at matched memory" (§3) is meaningless without stating whether Oja was run in the same shared-basis parameterization, how its step size was tuned, and how many passes it got. Oja's rule is a stochastic-gradient method whose accuracy is step-size limited; incremental SVD is an exact rank-r Gram update. That the latter has lower reconstruction error at equal rank is expected and not a contribution.

**W5. Coordinate consistency across basis updates is never addressed.** The gist stores "per-token coordinate columns C = UᵀM" (§2.3), but U changes at every block. A coordinate c_t = U_tᵀ m_t computed at ingestion is expressed in U_t; after the basis becomes U_{t+1}, reconstructing with U_{t+1} c_t is wrong unless the whole coordinate buffer is rotated by U_{t+1}ᵀU_t at each step (O(r²T) per update, compounding projection loss), or unless old bases are retained. The paper is silent. This matters for (i) the correctness of "M̂ = UUᵀM," (ii) the per-token complexity, and (iii) whether the 1.01–1.03× number describes the stored cache or only the final basis. I would also not be surprised if the "numerical explosion" of §4.4 lives here or in loss of orthogonality of U (which is the only way a projection-based scheme can produce perplexity 27,531), rather than in the ODE-style stiffness the text invokes.

**W6. The relative singular-value floor is a standard numerical-rank tolerance, not a "CKL-θ variant," and its role is under-examined.** CKL's criterion is a Frobenius-norm bound on the *discarded tail*, (Σ_{j>r₁} σ_j²)^{1/2} ≤ ϑ, absolute in scale. Dropping σ_i ≤ f·σ₁ is the classic rank-revealing relative tolerance (as in `rank`/`pinv`). Branding it CKL-θ is not accurate. More importantly: if a projection with orthonormal U cannot diverge, then the floor is masking an implementation defect (orthogonality loss, bf16 leakage, coordinate-rotation drift). The right diagnostic is to report ‖UᵀU − I‖ over the stream with the floor off, which the paper does not do. It is also odd that the fix is default-off while the flagship's own Qwen 32K perplexity diverges to 35.1 (Table 2, §4.6); the paper says "the natural 32K operating point for Qwen is a higher rank with the stability floor" but never reports that cell. "Retrieval- and footprint-neutral at the operating point" (Table 4 caption) is asserted without a table.

**W7. The online-tracking motivation is never tested against the obvious static control.** The whole premise (§1) is that the KV subspace is non-stationary and must be tracked. The one ablation that would demonstrate this — a one-shot SVD of the first chunk (xKV/ShadowKV-style), frozen for the rest of the sequence, at the same r — is absent. Without it, the paper cannot distinguish "online tracking helps" from "any rank-64 pre-RoPE basis plus a surprise tier and a recent ring works." Given that §2.1 says "on real KV all block sizes agree to 1–3%" (with b = T being the offline oracle), the KV matrix may be close to stationary and a frozen basis may do nearly as well.

**W8. Pre-RoPE operating point vs. the promised memory win.** Reconstruct-then-RoPE is correct (RoPE is a per-token orthogonal rotation, so Frobenius error is preserved, and pre-RoPE keys are lower-rank). But it is in tension with the paper's escape hatch for the 1.06× residency. For post-RoPE keys or for values, attention can be run in r dimensions: qᵀUUᵀk = (Uᵀq)ᵀ(Uᵀk). For pre-RoPE keys the effective basis is RoPE_pos·U, different for every token, so no r-dimensional inner product exists; a "fused factored-attention kernel" must materialize n-dimensional keys tile-by-tile at O(nrT) FLOPs per layer per decode step. Palu had to handle exactly this. The paper should say plainly that the pre-RoPE choice makes the deferred kernel harder and more compute-bound, and estimate the FLOP cost. Relatedly, the only latency datum (1B, CPU, +10%) does not inform GPU behavior at 32K.

**W9. Complexity is not stated.** The paper gives O(rn + hn) for the *memory* overhead but never the *time* per token: QR of the n×b residual (O(nb²)), SVD of the (r+b)×(r+b) core (O((r+b)³)), basis rotation U_aug P (O(n(r+b)r)), possible coordinate-buffer rotation (O(r²T)), reconstruction (O(nrT) per decode step unless cached — and it is cached in a ≈0.98×-full-KV workspace, hence 1.06× residency). "2r/n asymptote" (0.125× at fp32 = 4r fp16-units / 2n) is correct as stated, but the fit constant 380/T should be given its units (tokens) and the 0.127 vs 0.125 gap explained (the r×r core B and sink/ring pieces, presumably).

**W10. Selective reporting in §4.7(1).** With WikiText filler on Qwen 16K, only the flagship's *single-needle* score is given (1.00) while the baselines' failures on all four tasks are itemized. The flagship's multi-key, multi-value and variable-tracking on realistic filler are exactly the numbers a reader needs and are not reported. Likewise the LongBench "probe" is a single ~5.6K-token document.

**W11. The r/n ≈ 0.25 wall is asserted from two points and conflicts with the score-rank fix.** The "dimensionless onset" rests on Llama r=256 and Qwen r=128, plus a "matched control" that is the same Llama configuration; Mistral at r=256 (n=1024) is measured only for perplexity (Table 4). The mechanism offered (a large U makes needles look unsurprising) is exactly what the -ss score-rank decoupling (§2.3) is designed to remove, yet the wall is presented as a property of the method. Either -ss dissolves the wall (then it is not a wall), or it does not (then the mechanism is not the stated one).

**W12. Unsupported words and small arithmetic slips.** "surprise selection *provably* keeps sharp, isolated needles" (§5): no proof or statement is given. "3.4–10× less than think-c0.5 … and 6.7–13× less than full KV" (§4.2): with 0.085–0.149× the ratios are 0.75/0.085 = 8.8× and 1/0.085 = 11.8×, not 10× and 13×. "0.284× honest, 0.16× float-equivalent" (§4.3): 0.284/2 = 0.142; the discrepancy needs a one-line explanation of which parts are fp32. Mixing fp16-equivalent (Tables 1, 2) and fp32-at-rest (Tables 3, 7, 8) ratios across tables forces the reader to keep two conventions; pick one for tables and put the other in a column.

**W13. The KIVI marquee tie is not paired.** Table 3 reports the marquee at n=16 against "KIVI 4-bit 1.00 [0.76,1.0]" — that interval is a 12/12 result, i.e. the n=12 arm from Table 7, not a paired n=16 run. Note this.

## Questions for the authors

1. Please write out the exact difference, in linear-algebra terms, between your "streaming BUG tracker" with θ off and rank cap r and the "fixed-rank incremental SVD" you beat "everywhere" in §4.1. If none, please reconcile.
2. How are stored coordinates handled when U changes? Rotated (cost?), left in stale bases, or recomputed from retained bases? What error does the *stored cache* (not the final U) have relative to the truncated SVD?
3. For §4.1: model, context length, error metric, per-rank numbers, and the Oja configuration (step size schedule, init, per-head vs shared, passes, orthonormalization). Which OjaKV arXiv version is your baseline and your "position vs content" differentiation based on?
4. With θ off and floor off, please report max_t ‖U_tᵀU_t − I‖_F and whether any bf16 op touches U, B, or the coordinates. Where exactly does perplexity 27,531 come from mechanically?
5. Add the frozen one-shot-SVD control (same r, same tiers) at 16K/32K. How much of Table 1 survives with no online update at all?
6. Flagship on WikiText filler (§4.7(1)): all four tasks, all three families.
7. Does -ss move the r/n = 0.25 wall on Llama? If yes, in what sense is it a wall?
8. Estimated FLOPs/token for a fused kernel that must apply RoPE after reconstruction, versus a post-RoPE r-dim kernel. Would you consider tracking post-RoPE keys at a higher rank as the deployable variant?
9. Which parts of the marquee config are fp32 vs fp16 (to explain 0.284 vs 0.16)?
10. Why is the floor default-off when the flagship's own Qwen 32K cell diverges? Report Qwen 32K with the floor on at r=64 and at the "natural" higher rank.

## Limitations assessment

The Limitations section is thorough on the systems side (no kernel, 1.06× residency, batch size 1), on external validity (in-house generator; official RULER trails), and on the wide-KV variable-tracking failure. It does not acknowledge (a) that the algorithm is incremental SVD and that no applicable error bound is given, (b) the absence of a frozen-basis control, (c) the coordinate-rotation question, or (d) the changed OjaKV selection mechanism. No societal-impact concerns beyond the usual for inference efficiency.

## Scores

- **Soundness: 2** (fair). The empirical protocol is careful, but the mathematical framing misattributes the method, the invoked theory does not apply, the flagship §4.1 claim is unsupported by any reported data, and a possible correctness issue (W5) is unaddressed.
- **Presentation: 2** (fair). Dense, self-referential abstract (a full page); two memory conventions across tables; the key near-oracle result has no table; several arithmetic slips.
- **Contribution: 2** (fair). Applying incremental SVD with a residual-selected exact tier to KV is a reasonable engineering idea, but the tracker is classical, the residual-selected exact tier now also appears in OjaKV, and the method does not reduce resident memory; the genuinely new pieces are the diagnostic observations and the 0.048× composed cell.
- **Overall: 4** (Borderline reject).
- **Confidence: 4.**

## Justification

I want to be clear that I respect the honesty of this paper; it reports its losses as plainly as its wins and its statistics are correct. But the paper's identity — "Dynamical Low-Rank Approximation as an Online KV-Cache Compressor," a "BUG integrator" with a "σ_min-independent error bound" — does not survive inspection: the update in §2.1 is block incremental SVD, the CKL bound is for matrix ODEs and has no counterpart here, no bound of any kind is stated, and the relative singular-value floor is a standard numerical-rank tolerance rather than a CKL variant. The single paragraph carrying the "near-oracle, 1.3–3× better than Oja" claim has no numbers, no protocol, and an internal inconsistency with §2.1, while the OjaKV differentiation ("content vs position") is out of date relative to that paper's current version. The one ablation that would justify online tracking at all (a frozen prefill SVD) is missing, and the coordinate-basis consistency question could affect correctness. What remains is a well-instrumented empirical study of incremental-SVD KV compression that trails a 2-bit quantizer on the official benchmark and does not reduce resident memory. Reframed honestly, with §4.1 tabulated, the frozen-basis control, the Oja protocol specified, and either an applicable (Frequent-Directions-style) bound or no bound claimed, this could be a solid contribution; as submitted, I cannot recommend acceptance.

Sources consulted for citation checks: [Ceruti–Kusch–Lubich rank-adaptive integrator (arXiv:2104.05247)](https://arxiv.org/pdf/2104.05247), [OjaKV v1 (arXiv:2509.21623v1)](https://arxiv.org/html/2509.21623v1), [OjaKV v2 (arXiv:2509.21623v2)](https://arxiv.org/html/2509.21623v2), [OjaKV ACL Findings 2026](https://aclanthology.org/2026.findings-acl.494.pdf), [ShadowKV (arXiv:2410.21465)](https://arxiv.org/abs/2410.21465).


---

# Review — Reviewer 2 (NeurIPS 2026 Main Track)

**Paper:** *Dynamical Low-Rank Approximation as an Online KV-Cache Compressor*

---

## Summary

The paper adapts the rank-adaptive basis-update & Galerkin (BUG) integrator from dynamical low-rank approximation (Ceruti–Kusch–Lubich) into a streaming, training-free, per-sequence low-rank tracker for the pre-RoPE KV matrix. The cache consists of (i) attention sinks, (ii) a recent ring, (iii) a rank-r "gist" (basis U plus r-dim coordinates per token), and (iv) an exact tier of h tokens selected by "surprise" (out-of-subspace residual), plus a "warm-up seed" that routes the first ingest chunk through the surprise path. A relative singular-value floor is proposed to stabilize high-rank divergence. Experiments on Llama-3.1-8B, Qwen2.5-7B, and Mistral-7B-v0.3 use an in-house RULER-style generator (n=12–16 trials/cell) at 16K/32K, one official RULER run (Llama, 16K, 12 records/task), WikiText perplexity (n=4 windows), and a single-document LongBench probe. Baselines are think-c0.5 and palu-r0.5 (via kvpress ports), Expected Attention / SnapKV eviction at 0.1×, and a self-implemented "KIVI-scheme" 2-/4-bit quantizer. The paper's own summary is that at matched stored bytes the method wins multi-value retrieval over 2-bit quantization on its own generator, does not reproduce that edge on official RULER (0.79 vs 0.87 mean), is not separated from 4-bit at 2× the bytes, has decode residency ≈1.06× full KV, and is ≈10% slower per token in the one latency datum.

## Strengths

1. **A genuinely new tool imported into the KV-compression space.** Using a robust DLRA integrator (σ_min-independent error bound, rank-adaptive truncation) as an online subspace tracker is a principled alternative to Oja's rule, and the measurement that BUG lands within 1.01–1.03× of truncated SVD where Oja is 1.3–3× worse (§4.1) is a nice, if small-scale (1B), result.
2. **Unusually candid reporting.** The paper explicitly separates stored state from resident memory (§4.9), reports the 1.06× decode residency, the ≈10% slowdown, the Qwen divergence, the r/n≈0.25 wall, the failure of the multi-value edge to transfer to official RULER, and retracts an earlier "beats SOTA at 10×" claim (§4.8). Paired McNemar tests on shared needles and released per-trial records are good practice.
3. **The rank-siphoning diagnosis (§4.4, Table 4)** is a real and interesting mechanistic observation: content-based exact-tier selection removes exactly the high-residual columns that carry rank, starving the tracker. The relative SV floor fixing 27,531 → 6.995 perplexity is convincing as a within-method stability result.
4. **Composition argument.** The observation that low-rank (feature axis) and quantization (bit axis) compose, and the 0.048× cell (§4.5), is the one place the paper occupies a byte band a fixed-bit scalar quantizer cannot reach.

## Weaknesses

**W1. The paper is not, by its own measurements, a KV-cache compressor in the deployed sense (§4.9, §5).** Decode residency is "≈1.06× full KV—slightly worse than no compression"; the reconstruction workspace is 0.98–0.99× full KV; the only end-to-end latency datum (1B, CPU) is 10% *slower*; batch size is 1 only; no fused kernel exists. Every competing family in Table 7/8 (KIVI, ThinK, Palu, eviction) *does* reduce resident memory today. The measured systems payoff is a persisted-cache reload of 0.13 s vs 1.23 s at 16K — but the paper concedes the 2-bit quantized cache reloads in 0.14 s (Figure 3), so the persistence win over quantization is ~0. What remains is a 19% lower storage asymptote (0.125× vs 0.156×) that is only realized at 64K and only if the gist stays fp32. For a venue paper titled "KV-Cache Compressor," this is a framing problem, not a footnote.

**W2. The memory metric is non-standard and used inconsistently in the paper's own favor.** Tables 1–2 and the contribution bullets report "float-equivalent" ratios (0.085×, 0.075×) that bill the fp32 gist as if it were fp16, while a "fp16-storable gist variant is not yet validated (§5)." The "honest" fp32-at-rest ratio for the same config is 0.151×/0.139× (Table 7) — a 1.8× discrepancy for the identical run. On Qwen the gap is 0.149× vs 0.275×. The abstract leads with "0.15× ... to 0.28×" but the first contribution bullet and §4.2 title say "0.085–0.149×," and the "3.4–10× less than think/palu" multiplier uses the hypothetical number. Only one convention should appear in headline claims, and it must be the one that corresponds to bytes actually stored. Even then, "stored bytes" is not the metric the field uses (peak KV VRAM at decode), and the paper should say so plainly in the abstract.

**W3. think-c0.5 and palu-r0.5 are not matched-budget baselines, and the Palu port is questionable.**
- think-c0.5 (0.75×) and palu-r0.5 (0.50×) sit at 3–5× more bytes than the flagship; the paper then reports "beats think-c0.5 ... at 1.8–2.6× less honest stored state." That is a comparison across different points on the Pareto curve, not a fair contrast. ThinK is designed and evaluated in its paper composed with SnapKV/H2O; standalone key-channel pruning at λ=0.5 is a weak configuration.
- kvpress has no Palu press (I checked the current press list: ThinKPress exists, no PaluPress). The Palu baseline is therefore the authors' own port, and the paper admits "a Palu attention-sink low-ranking ... had inflated its perplexity from 7.232 to 9.236" was a port defect. Palu's actual method uses Fisher-information rank search across grouped heads and (optionally) fine-tuning; the paper does not say which of these the port implements. Palu scoring 0.42 on official RULER multi-value while ThinK at fewer compressed bytes scores 1.00 (Table 8) is inconsistent with Palu's published near-lossless 50% results and suggests the port is still off.
- No eviction baseline at the flagship's bytes: SnapKV/PyramidKV/H2O at k≈0.15 are absent; only ea-k0.1 (Table 6) and one SnapKV-k0.1 sentence appear. H2O and PyramidKV are cited in the intro but never run.

**W4. OjaKV is the direct prior on the exact axis claimed, but is neither run end-to-end nor characterized correctly.** OjaKV is compared only as a reconstruction-error curve on Llama-3.2-1B (§4.1). The paper's stated differentiator — "selecting the exact tier by content (surprise) rather than by position" (§3) — describes OjaKV v1; the current arXiv v2 (April 2026) explicitly "retain[s] the top-k tokens with highest error scores at full rank" using query-weighted reconstruction residuals, i.e., a content-based exact tier. That is essentially this paper's mechanism. OjaKV also reports LongBench and RULER end-to-end on Llama-3.1-8B, which this paper never matches. The remaining delta (BUG vs Oja, rank-adaptivity, seed) may be real but must be shown on downstream tasks, not reconstruction error at 1B.

**W5. The "KIVI-scheme" baseline is a weakened reimplementation, and the flagship still loses to it on the official suite.** G=64 (KIVI default 32), chunked 4096-token prefill with residual flushed (not KIVI's full-precision-prefill operating point), quanto backend, no KIVI kernels. The paper acknowledges "ours is a slightly coarser but cheaper 2-bit arm." The multi-value wins in Table 7 (KIVI-2 at 0.42/0.50/0.33 at 16K) are well below what KIVI reports on Llama-family models at 2-bit, and are exactly the cells where a coarser group size hurts. On official RULER the flagship trails even this weakened 2-bit arm (0.79 vs 0.87; vt 0.33 vs 0.67; mq 0.67 vs 1.00). Against a faithful KIVI (G=32, fp16 prefill) or KVQuant (pre-RoPE, dense-and-sparse, ~3-bit near-lossless), the Table 7 edge would likely vanish.

**W6. Benchmark coverage is thin and the statistics cannot support the language used.** n=12 trials/cell gives a Wilson lower bound of 0.758 at 12/12; calling this "perfect ... retrieval with no detected loss" in the abstract is misleading when the CI admits 24% failure. Official RULER: one model, one context (16K), 12 records/task (the official protocol uses 500), and the flagship is the worst arm above eviction on 7 of 9 tasks. LongBench: "a single Qasper document, ∼5.6K tokens" — a single data point with a *different* config (bug-r128, 0.160×), F1 0.221 vs full 0.259 (−15%). Perplexity: n=4 windows of 512 tokens, NLL summed in bf16. There is no summarization, code, QA, or few-shot task at any scale. The paper itself says "a three-family, 32K official grid and a flagship LongBench sweep remain open." Those are the experiments a NeurIPS paper needs, not the camera-ready.

**W7. The marquee result (Table 3) is a different configuration from the flagship, and its selection is opaque.** The flagship is bugSseed-r64-h256; the only separated result uses bugSseed-r128-h1024-s32 at 0.284× honest bytes — 1.9× the flagship's bytes — with the score-rank knob that "recovers Llama's r=128 retrieval." §2.3 describes -ss as chosen because it fixed Llama's r=128 retrieval, i.e., tuned on the evaluation task. The "pre-registered in the pod matrix" claim is unverifiable without a timestamped registration. And at those bytes, 4-bit KIVI scores 1.00 on the same task with better perplexity (7.54 vs 8.33 at 32K); the marquee therefore establishes only that ThinK-alone at 0.75× is bad at variable tracking.

**W8. Generality across models is not demonstrated; the three families each break a different way.** Llama/Mistral: variable tracking 0.50–0.58 at 16K, "unsolved" (§5). Qwen: r=64 gist diverges at 32K (perplexity 35.1 vs 8.2), the honest cost is 0.275× (= 4-bit bytes) and the paper concedes it "is dominated outright"; the composed cell scores 0/0/0/0 on Qwen. There is no model on which the method is Pareto-competitive with 4-bit quantization across all four tasks plus perplexity. Nothing is shown beyond 7–8B, and nothing at all with batching.

**W9. The 0.048× composed cell (§4.5) is not a usable operating point and its competitor set is incomplete.** Llama perplexity roughly doubles (9.25 vs 5.31 at 16K; 17.1 vs 8.33 at 32K); variable tracking ≤0.83; Qwen fails. The "fixed-bit floor" argument is only against scalar per-element KIVI; the paper does not test TurboQuant at 2–3 bits as a *pure* quantizer (it uses TurboQuant only for its own coordinates), KVQuant's 2-bit dense-and-sparse, vector quantization, or SnapKV×KIVI at 0.05× — the eviction×quant composite uses only Expected Attention. The claim "below the retrieval any eviction×quantization composite reaches" is therefore over-generalized from one eviction scorer.

**W10. No compute-overhead accounting at 8B.** Each block step requires a QR of the n×b residual, an SVD of the augmented core, and re-orthogonalization; each decode step requires reconstructing U·C for the entire history (O(n·r·T) per layer) before attention. The paper reports no TTFT, no GPU decode ms/token, no FLOPs, and no throughput on the 8B models — only 1B CPU (+10%). For a systems-flavored contribution this is a serious gap; OjaKV, for comparison, reports TTFT and memory at 32K.

**W11. Novelty of the tracker is modest and partly conceded.** §2.1 states that for rank-1 increments BUG "reduces to the classical incremental-SVD update" (Brand 2006) and the blocked truncation "is a deterministic sketch in the spirit of Frequent Directions." Pre-RoPE factorization is prior art (ShadowKV, KVQuant). The exact-tier-plus-low-rank decomposition is in LESS, LoLA, and (now) OjaKV v2. The near-oracle result is "a measurement on real KV, not a theorem." What is left that is new is the rank-adaptive truncation tolerance and the relative SV floor — the paper itself calls the latter "a small variant of published DLRA truncation, not a new mechanism."

**W12. Presentation.** The abstract is ~650 words of hedged sub-claims; "honest" appears dozens of times as a modifier on numbers; internal provenance (Week-18/Week-19, pod, commit SHAs, "results/w18 pertrial/") is lab-notebook material that should go to an appendix; the same config is reported at 0.085× (Table 1) and 0.151× (Table 7) without a cross-reference; §4.7 is titled "Moderate compression on realistic text" but its main content is the official RULER anchor where the method loses; the "per-family seed/window/depth appendix table is deferred to the camera-ready." The paper reads as a work-in-progress report rather than a finished submission.

## Questions for Authors

1. Please report peak decode VRAM and ms/token on Llama-3.1-8B at 16K/32K on the A100 for full KV, flagship, KIVI-2, and think-c0.5. What is the per-block BUG update cost (ms) and the per-step reconstruction cost?
2. Which Palu variant is implemented in your kvpress port — per-head SVD of W_K/W_V, grouped-head decomposition, Fisher-based rank allocation, with or without fine-tuning? Can you reproduce Palu's published LongBench numbers at 50% with your port?
3. Why is OjaKV not run as an end-to-end baseline on Llama-3.1-8B at matched bytes, given it addresses the identical online per-sequence axis? Please also update the characterization of its hybrid storage in light of arXiv v2.
4. Please run the KIVI baseline at G=32 with full-precision prefill (the paper's stated operating point) and re-run Table 7. Does any multi-value separation survive?
5. Was bugSseed-r128-h1024-s32 selected before or after observing 32K Llama variable-tracking results? Is there a timestamped record of the pre-registration? Why is the marquee config not also the flagship?
6. What does the flagship score on full LongBench (all tasks) and on the official RULER suite at 32K on all three families, at the official 500 samples/task?
7. What are the failure modes of the fp16-stored gist? Which quantity overflows/cancels — the core B, the coordinates C, or the QR of R⊥?
8. Have you tried SnapKV or PyramidKV at k=0.15 (matched to the flagship's honest bytes) on the official RULER anchor?
9. At r/n≈0.25 retrieval collapses to 0.00 on all four tasks (§4.4). Doesn't this imply the exact tier, not the gist, carries essentially all retrieval — in which case what does the low-rank gist contribute over a "surprise-selected sparse cache + recent window" with no gist at all? Please include that ablation.

## Limitations Assessment

The Limitations section is thorough and, to the authors' credit, leads with the two most damaging facts (no resident-memory saving; variable tracking unsolved on wide-KV models). However, thoroughness of disclosure does not substitute for the missing evidence. The limitations enumerated — no kernel, batch=1, 1B-only frontier, single-model/single-context official anchor, one-document LongBench, unvalidated fp16 gist, ~40 exploratory Wilson cells with one pre-registered contrast — collectively describe a paper whose central claims (a deployable compressor, cross-family generality, an edge over quantization) are not yet established. Potential negative societal impact is not discussed, which is acceptable for this topic.

## Scores

- **Soundness:** 2 (fair) — Methodology is careful in places (paired tests, released records), but the memory metric is self-defined and inconsistently applied, key baselines are self-ported/weakened or absent (Palu port, KIVI G=64/chunked, no OjaKV/SnapKV/PyramidKV at matched bytes), n=12 cannot support "perfect retrieval," and the marquee config appears tuned on the test task.
- **Presentation:** 2 (fair) — Clear prose sentence by sentence, but the abstract and contribution list over-claim relative to the body, two memory conventions are mixed in headline numbers, and lab-notebook provenance dominates the experimental section.
- **Contribution:** 2 (fair) — The BUG-as-tracker idea and the rank-siphoning diagnosis are worthwhile, but the demonstrated delta over OjaKV/incremental SVD/Frequent Directions is small, the system saves no resident memory, and the method is Pareto-dominated by 4-bit KIVI on every model tested.
- **Overall:** 3 (Reject)
- **Confidence:** 4

## Justification

The paper introduces an interesting numerical-analysis tool into KV compression and is admirably transparent, but transparency about the gaps does not close them. On the paper's own measurements, the method uses ≈1.06× the resident memory of an uncompressed cache, decodes slower, is not separated from (and on Qwen is dominated by) 4-bit quantization, trails a deliberately weakened 2-bit KIVI reimplementation on the only external benchmark (official RULER, 0.79 vs 0.87), has one LongBench data point on a single document, and reports headline compression ratios in a "float-equivalent" convention that undercounts stored bytes by 1.8× relative to what was actually run. The closest prior work (OjaKV) is not run end-to-end and its current version already uses the content-selected exact tier the paper claims as its differentiator; the Palu baseline is a self-written port with acknowledged defects and no kvpress upstream. The one separated result is against ThinK-alone at 0.75× — a weak standalone configuration — using a different, apparently post-hoc-selected configuration at bytes where 4-bit quantization matches it. I would encourage the authors to (a) build the fused factored-attention kernel or otherwise demonstrate a resident-memory win, (b) validate the fp16 gist, (c) run OjaKV, faithful KIVI (G=32), KVQuant, and matched-budget SnapKV/PyramidKV on full official RULER (32K, three families, 500 samples) and full LongBench, and (d) adopt a single, standard memory metric. With those, this could be a solid contribution; as submitted it is a well-documented negative-to-neutral result.

---

Sources consulted for baseline checks: [OjaKV arXiv abs](https://arxiv.org/abs/2509.21623), [OjaKV v1 HTML](https://arxiv.org/html/2509.21623v1), [OjaKV v2 HTML](https://arxiv.org/html/2509.21623v2), [OjaKV OpenReview](https://openreview.net/pdf?id=XVjgvJhLTY), [NVIDIA kvpress](https://github.com/NVIDIA/kvpress), [MomentKV abs](https://arxiv.org/abs/2606.01563), [MomentKV HTML](https://arxiv.org/html/2606.01563v1), [ResKV abs](https://arxiv.org/abs/2607.29591).


---

# Review — Reviewer 3 (experimental methodology & statistics)

**Paper:** *Dynamical Low-Rank Approximation as an Online KV-Cache Compressor*

## Summary

The paper adapts the rank-adaptive BUG integrator for dynamical low-rank approximation into a streaming, per-sequence KV-cache compressor. The stored cache is a low-rank "gist" (basis U, core B, per-token coordinates) plus a verbatim "exact tier" of up to h tokens chosen by out-of-subspace residual ("surprise"), plus sinks, a recent ring, and a warm-up seed. The paper reports (i) near-oracle reconstruction vs. truncated SVD on 1B KV; (ii) a cross-model 16K/32K retrieval grid on a RULER-style in-house generator (n=12 per cell) for Llama-3.1-8B, Qwen2.5-7B and Mistral-7B; (iii) a "marquee" 32K Llama variable-tracking contrast (n=16) with paired McNemar vs. ThinK and Palu; (iv) a matched-bytes 2-bit/4-bit KIVI-scheme baseline; (v) an official NVIDIA RULER anchor on Llama 16K; (vi) a rank-siphoning / singular-value-floor diagnosis; and (vii) a stored-state vs. resident-memory accounting showing the method does not yet reduce decode VRAM.

## Strengths

1. **Unusually transparent reporting.** Per-trial hit records, git SHAs, environment provenance, retracted earlier claims (§4.8), bugs fixed in the baselines' favour (§4.3), and an explicit statement that decode residency is ≈1.06× full KV are all commendable. The Limitations section leads with the two most damaging facts.
2. **Correct choice of paired test.** All arms are scored on identical needles, and the authors use exact McNemar rather than comparing independent Wilson intervals. I recomputed every reported p-value and they are correct (see below).
3. **The fair 2-bit baseline and the official RULER anchor were added, and the negative results were kept.** The official-suite result (flagship 0.79 vs 2-bit 0.87, no task separated; multi-value edge does not reproduce) is reported as plainly as the positive in-house result.
4. **Wilson intervals are computed correctly** (12/12 → [0.758, 1]; 15/16 → [0.717, 0.989]; 5/16 → [0.142, 0.556]; 9/16 → [0.332, 0.769]; all match my recomputation).

## Weaknesses

**W1. Sample sizes are too small to support most of the language, and "no detected loss" is much weaker than the abstract's "perfect".**
With n=12, an observed 12/12 gives Wilson LB 0.758 (Clopper-Pearson: 0.735). The probability of observing 12/12 when the true accuracy is 0.90 is 0.28; when it is 0.80 it is 0.07. So the 16K cross-model result (Table 1, §4.2) is consistent with a 10–20 percentage-point retrieval loss, yet the abstract, contributions list, and conclusion all say "perfect single- and multi-value 16K retrieval". "Perfect" describes the sample, not the population; it should be removed. Similarly, the 64K claim ("retrieves all four tasks, n=8; no contrast separates", §4.9) is 8/8, Wilson LB 0.676 — a 30-point loss is not excluded.

**W2. Null results are systematically read as ties/parity despite having essentially no power.**
At n=12, the smallest attainable two-sided exact McNemar p-value with one-way discordance is 2·0.5^6 = 0.031 at ≥6 discordant pairs — i.e., a 50-percentage-point paired gap is required to reach p<0.05. The paper acknowledges this once (§4.6, "nulls bound rather than establish equivalence") but then repeatedly uses non-separation as a positive: "4-bit quantization at twice the bytes is not separated from the flagship in any cell" (abstract, §1, §4.6, §5, §6); "ties single-needle everywhere"; "on the official NVIDIA RULER suite the flagship and the 2-bit arm are not separated on any task". The abstract's "not separated" against 4-bit hides that 4-bit KIVI leads on Llama 16K var-track 1.00 vs 0.58 (5/0 discordant, p=0.0625 — one trial short of "significant"), and on the official RULER var-track 1.00 vs 0.33 (p=0.008). "Not separated" at n=12 is not a tie; it should not appear as a headline claim.

**W3. Multiple comparisons: the count is understated by ~5× and the correction is applied only to the one contrast that survives it.**
§4.3 says the tables report "∼40 descriptive Wilson cells". I count roughly 9 (T1) + 12 (T2) + 4 (T3) + ~30 (T5) + 16 (T6) + 72 (T7) + 63 (T8) ≈ 200 accuracy cells, and Table 7 alone implies 48 flagship-vs-quantizer paired contrasts. Seven flagship-vs-2-bit wins are reported (p = 0.016, 0.031, 0.008, 0.004, 0.004, 0.002, 0.004). Treating just the 24 flagship-vs-2-bit tests in Table 7 as the family: Bonferroni α = 0.0021 → only Qwen-32K multi-value (p=0.00195) survives; Holm → the same single contrast (0.002·24 = 0.047; next 0.004·23 = 0.092 fails); Benjamini–Hochberg at q=0.05 → 5 of 7 survive, 2 do not. The paper's own phrase "BUG buys multi-value retrieval that 2-bit quantization loses" is therefore a claim that rests on 1 FWER-robust cell (or 5 FDR-robust cells), on a generator the official suite does not reproduce. The "pooled 21 vs 0" (abstract) is a post-hoc pooling of the one task chosen because it won across families; its nominal p (2·0.5^21 ≈ 10⁻⁶) is not a valid confirmatory statistic.

**W4. The "pre-registered confirmatory comparison" is not verifiable and is confounded with configuration selection.**
The marquee config bugSseed-r128-h1024-s32 is not the "single configuration" of the abstract (r64-h256); it is a different rank, tier size, and an extra knob (-ss) that §2.3 says was introduced precisely because it "recovers Llama's r=128 retrieval". §4.2 says the 32K cells "firm what were previously n=4 point estimates", so the hypothesis and the arm were chosen after a pilot on the same task/model. "Registered in the pod matrix before the run" is an internal artifact with no external timestamp (OSF/AsPredicted or a signed commit). Whether the n=16 excludes the n=4 pilot trials is not stated; if included, the p-value is anti-conservative. The recomputed p-values themselves are correct (10/0 → 2·0.5^10 = 0.00195; 6/0 → 0.03125; Wilson 0.717 vs 0.556 disjoint), but the confirmatory framing is over-claimed. Also note the confirmatory win is against think-c0.5 (0.31), a baseline at 2.6× more bytes that is trivially beaten by the paper's own 4-bit KIVI arm (1.00 at the same 0.284×); statistically the marquee establishes that ThinK is bad at this task, not that BUG is good.

**W5. Every "fix" was discovered and evaluated on the same evaluation cells — test-set tuning.**
(a) The warm-up seed was added because 16K recall failed on this generator and is evaluated on the same 16K generator (§2.3, §4.2). (b) Score-rank decoupling (-ss) was added to recover Llama r=128 retrieval and is evaluated on Llama r=128 retrieval. (c) The singular-value floor (--min-sv-frac 1e-2) was tuned to the perplexity blow-ups in Table 4 and its success is demonstrated on exactly those five perplexity cells; no held-out prompt set or held-out model verifies it. (d) The q4 cell's "budget inversion" was found because results looked wrong and rerun (footnote 1). None of this is dishonest — the paper is candid about it — but there is no held-out split anywhere, and the one external test (official RULER) fails to reproduce the key in-house edge, which is exactly what one expects after iterating on the in-house generator with a "small fixed filler pool" (§4, §5).

**W6. Variance structure is unreported and the effective n is likely below 12.**
"n=12 (6 trials × 2 seeds)" with a fixed filler pool means trials within a seed share most of the context; the per-seed, per-depth, per-window breakdown is "deferred to the camera-ready". Needle depth is a first-order variable for a method with a warm-up window and a recent ring (the official-RULER misses are "front-loaded — six of fourteen fall in the first ingest chunk"), yet no depth-stratified results are given for the in-house grid. Wilson/McNemar treat the 12 trials as i.i.d.; with clustering by seed, the intervals are too narrow.

**W7. Perplexity comparisons have no uncertainty at all and use tiny, mismatched samples.**
§4.6 fluency uses n=4 windows of 512 tokens; §4.3 uses n=8 windows; no SE, CI, or paired test is given, yet the text declares "within the pre-registered 0.05-bits/token band" (an equivalence claim requiring a CI inside the band) and "fluency goes to the quantizer" (a superiority claim). A bf16 summation floor of ≈0.006 bits/token is disclosed after the fact. Also, "perplexity at 16K/32K" over 512-token windows needs definition: is the compressed cache 16K long with a 512-token scored window?

**W8. Mixed and unpaired n within tables.** Table 3 mixes n=16 arms with a 12/12 4-bit KIVI arm; the 64K flagship cells are n=8 vs comparators n=12 (§4.9) so "no contrast separates" cannot be a paired McNemar; the 8-bit hqq control is n=4. Table 3's "tie (same bytes)" for 4-bit KIVI is asserted without a test.

**W9. The sub-0.05× "band exclusivity" claim is not statistically supported.** Table 5: ea-k0.1-q4 at 0.028× scores 1.00/0.75/0.92/0.42 vs the cell's 1.00/1.00/1.00/0.50 at 0.048× — 3 and 1 discordant trials out of 12 (p≈0.25 and 1.0). The abstract's "below the retrieval any eviction×quantization composite reaches at matched bytes" is a point-estimate statement dressed as a finding.

**W10. Arithmetic/consistency slips in the headline ratios.** §4.2 states 16K stored state 0.085–0.149× is "3.4–10× less than think-c0.5 (0.75×) ... and 6.7–13× less than full KV". 0.75/0.085 = 8.8 and 1/0.085 = 11.8; the 10× and 13× figures use the 32K value 0.075×. Two accounting units (float-equivalent 0.085× vs. "honest" fp32-at-rest 0.150×) alternate between tables and within the abstract, which makes the byte-matching arguments hard to audit.

**W11. Attribution of the retrieval effect to DLRA is untested at 7B/8B.** Qwen at 32K has r=64 perplexity 35.1 (gist has diverged) yet retrieval is 1.00 — retrieval is evidently carried by the verbatim exact tier + ring, i.e., a surprise-scored eviction policy, not by the low-rank tracker. No tier-only / gist-only ablation on the 7B retrieval grid isolates the DLRA contribution the title advertises.

**W12. Presentation.** The abstract is ≈680 words, contains ~40 numbers, three memory-accounting conventions, and hedges in both directions in the same sentence ("perfect ... with no detected loss (12/12, Wilson ≥ 0.758)"). An abstract this long is itself a methodology signal: the results do not compress to a claim. Config identifiers (bugSseed-r128-h1024-s32, ea-k0.25-q2, think-c0.5) and week/commit provenance (w16, w18, w19, "Week-19 a1q run") belong in an appendix.

## Questions for authors

1. Are the n=16 marquee trials disjoint from the n=4 pilot trials that motivated the comparison? Can you provide an externally timestamped pre-registration?
2. Provide the per-seed × per-depth breakdown for Tables 1, 2, 7. How many distinct filler documents exist in the "small fixed filler pool"? What is the intra-seed correlation of hits?
3. Report a tier-only ablation (surprise-selected h=256 + sinks + ring, no gist) on the 7B retrieval grid. How much of the multi-value edge survives without DLRA?
4. Apply Holm or BH across all flagship-vs-baseline paired contrasts in Tables 7 and 8 and state which cells survive.
5. Give SE/CI for every perplexity number; how many tokens are scored per "16K" perplexity figure?
6. Was the singular-value-floor threshold 1e-2 chosen on Table 4's cells? Does it hold on a held-out set of prompts and on Llama?
7. For the 64K point (n=8 vs n=12), what test underlies "no contrast separates"?

## Limitations assessment

The Limitations section is thorough and honest about systems limitations (no kernel, 1.06× residency), the wide-KV var-track failure, the fluency cost, and external validity. What it does not acknowledge is the statistical fragility: that every "fix" was developed on the evaluation itself, that "not separated" at n=12 is uninformative, that the multiple-comparison count is ~200 rather than ~40, and that perplexity claims carry no uncertainty. The paper is transparent, but transparency about provenance is not the same as adequate power or held-out validation.

## Scores

- **Soundness: 2** (fair) — statistics are computed correctly but samples are too small for the claims, corrections are applied selectively, and the development/evaluation split is absent.
- **Presentation: 2** (fair) — exemplary provenance, but the abstract and prose are overloaded, unit conventions shift, and configuration-switching between headline cells obscures what "the method" is.
- **Contribution: 2** (fair) — a plausible and novel mechanism (DLRA tracker + surprise tier) whose measurable advantage, after honest statistical accounting, reduces to one FWER-robust cell on an in-house generator that the official suite does not reproduce, at no resident-memory saving.
- **Overall: 4** (borderline reject)
- **Confidence: 4** — I recomputed all Wilson intervals, McNemar p-values, and correction thresholds reported here; I did not run the code.

## Justification

The authors have done the reporting parts of good methodology — paired tests, per-trial records, retractions, negative results kept — and the individual numbers check out. But the study design cannot bear the abstract's weight: n=12 cells admit 20-point losses under "perfect", non-separation at n=12 is treated as parity in the headline against 4-bit quantization, roughly 200 cells yield a handful of nominal wins of which one survives family-wise control, the "confirmatory" contrast is on a config engineered for that model/task after a pilot on the same task and beats a baseline that the paper's own 4-bit arm also trivially beats, every stabilizer was tuned on the evaluation cells, and the one external test does not reproduce the in-house edge. A resubmission needs (i) a held-out generator/filler set and model family for all fixes, (ii) n≥50 per cell or a Bayesian/hierarchical treatment of seeds and depths, (iii) global FWER/FDR control with the survivors as the only claimed wins, (iv) perplexity CIs, (v) a tier-only ablation isolating DLRA, and (vi) an abstract of ≤250 words that states the one or two claims the data support.


---

# NeurIPS 2026 Review — Reviewer 4 (Writing / Presentation Specialist)

**Paper:** *Dynamical Low-Rank Approximation as an Online KV-Cache Compressor*

---

## Summary

The paper repurposes the rank-adaptive basis-update & Galerkin (BUG) integrator from dynamical low-rank approximation (Ceruti–Kusch–Lubich) as an online, training-free, per-sequence tracker of the pre-RoPE KV feature subspace. The tracked low-rank "gist" is paired with (i) attention sinks, (ii) a recent-token ring, (iii) a small exact tier selected by *surprise* (residual norm to the tracked subspace), and (iv) a warm-up seed that routes the first chunk through the surprise path. The paper reports, on Llama-3.1-8B, Qwen2.5-7B and Mistral-7B-v0.3: near-oracle reconstruction vs. truncated SVD (1.01–1.03×, vs. Oja 1.3–3×), 12/12 single/multi-value retrieval at 16K at 0.085–0.149× float-equivalent stored state, a paired-McNemar win over ThinK on 32K variable tracking, a matched-bytes comparison against a KIVI-scheme 2-bit/4-bit quantizer (mixed: wins multi-value on the in-house generator, not separated on official RULER, trails 0.79 vs 0.87 on average), a composed gist+4-bit cell at 0.048×, and an honest accounting that decode residency is ≈1.06× full KV because no fused kernel exists. Two limitations are foregrounded: storage ≠ resident memory, and variable tracking on n=1024 models is unsolved.

The scientific content is careful and unusually candid. The *communication* of that content is, in its current form, far below what NeurIPS expects and is the main obstacle to acceptance.

---

## Strengths

1. **The core idea is clearly motivated in §1.** The observation that every prior low-rank KV method fixes its subspace (offline, train-time, or one-shot prefill) and that the KV stream is a non-stationary low-rank matrix — "precisely the object DLRA was invented to track" — is a crisp, memorable framing. The σ_min-independence argument for why BUG (rather than a naive (U,S,V) ODE or Oja) is the right tool is stated well.
2. **Positioning against OjaKV is explicit and fair.** §1 "Scope and concurrency", §2.1 and §3 all state exactly what is shared (the online per-sequence axis) and what differs (integrator, content-selected tier, mechanistic map). Reviewers will not have to hunt for this.
3. **Statistical reporting is exemplary in spirit.** Wilson intervals on every retrieval cell, paired McNemar on shared needles, a single pre-registered confirmatory contrast with everything else flagged exploratory, and per-trial hit records released. The "12/12 → no detected loss at ≥0.76, not a win" framing is exactly right.
4. **The Limitations section (§5) is genuinely informative** — it leads with the two most damaging facts (no resident-memory win; variable-tracking unsolved on wide-KV models), refutes the obvious fix, and scopes the external validity honestly. This is a model of what §5 should contain, even if the prose needs tightening.
5. **Figures 2 and 3 are the right figures**: the 1/T amortization plot and the cold-start bar chart each make a single argument cleanly.
6. **Corrections to earlier drafts are disclosed** (retracted "beats SOTA at 10×", fixed Palu/ShadowKV port bugs, invalid Week-18 q4 rows). Scientifically admirable.

---

## Weaknesses

### W1. The abstract is 682 words and unreadable as an abstract (p. 1–2).
It spans two pages, contains ~60 numeric quantities, three different memory-billing bases (fp16-equivalent, fp32-at-rest "honest", and bytes), five model/task-specific p-values, and a parenthetical about Qwen perplexity divergence. After reading it, an AC cannot state the contribution in two sentences; they can only state that the authors ran many experiments and are anxious about overclaiming. NeurIPS abstracts are ~150–250 words. This alone would cause many reviewers to stop reading charitably.

### W2. Length: ~17 pages of main text vs. the 9-page NeurIPS limit; no appendix exists.
Main text runs pages 1–18 (Conclusion ends mid-p.18), then 3.5 pages of references. There is no appendix at all — everything is in the main body, including commit SHAs (p. 18: `15678a7`, `b157acd`, `24ac22a`, `6734afa`, `1cbc31f`, `5e8b275`, `c331ebd`, `d15769d`), repo paths (`results/w18_pertrial/`, `scripts/w19_figures.py`), pod-hours and dollar cost ("≈ 42 A100-40GB pod-hours (≈ $39)"), "Week-18/Week-19" lab-notebook labels (6 occurrences), and footnote 1 explaining a budget-inversion bug in a discarded arm. This is a research log, not a paper. Roughly half the text must move to an appendix (see "Concrete suggestions").

### W3. Double-blind violation and arXiv-versioning language (p. 1, p. 3).
Author names, a GitHub URL (`github.com/kvdlra`) and a date appear on the title page. §1 says "This is an arXiv v1 timestamp of the mechanism… This version adds the two baselines the first draft lacked." §4.3/§4.8 refer to "an earlier draft". For a NeurIPS submission this is a format violation and the versioning narrative must be removed.

### W4. No algorithm box; the method is prose only (§2.1, p. 3).
The BUG step is described in a single paragraph with roman numerals (i)–(iv) and a hand-typeset block matrix that is broken in the text extraction and cramped in the PDF ($B_{\mathrm{aug}} = [\,^{B}_{0}\,^{A}_{R}\,]$). There is no pseudo-code for the per-block update, the surprise scoring, the tier admission/eviction rule, the seed path, or the score-rank cap. A reader cannot re-implement the cache from §2. A single `Algorithm 1: Streaming BUG-KV ingest of block C` (inputs U, B, tier, ring; outputs updated state) plus a one-line reconstruction rule would fix this.

### W5. Notation is defined inconsistently and several symbols are overloaded.
- §1 introduces the factorization as $USV^\top$; §2.1 immediately switches to $(U,B)$ with $BB^\top = U^\top MM^\top U$. $S$ and $V$ never appear again. Say once that you track only the column space and drop $V$.
- $b$ is the block size in §2.1 and the bit-width in §4.5 ("a $b$-bit quantizer… $k\,b/16$").
- $h$ is the exact-tier size; $H$ is the number of heads (§2 Convention). Adjacent and confusable.
- $n_{\text{sink}}$, $W$ (ring size), $\theta$ (Frobenius tolerance), $s$ (score rank), $\varepsilon$ are introduced but $n_{\text{sink}}$, $W$, $\theta$ are never given values anywhere. What are the sink and ring sizes in `bugSseed-r64-h256`?
- Config names (`bugSseed-r64-h256`, `-s32`, `-ss`, `--min-sv-frac 1e-2`, `ea-kk-qb`) are CLI flags, not notation. A table mapping config name → (r, h, seed, s, floor) is needed, and the text should say "the flagship" or "r=64, h=256" consistently rather than the flag string.

### W6. Three memory-billing bases are used interchangeably without a consistent flag.
Float-equivalent (fp16) ratios: 0.085/0.149× (Tables 1, 2). "Honest" fp32-at-rest ratios: 0.150/0.275× (Tables 3, 7, 8, Fig. 1, 2). Serialized bytes (GB). The abstract switches between them mid-sentence ("0.15× … as honest fp32-at-rest stored state (0.085–0.149× float-equivalent)—1.8–5× less … (3.4–10× float-equivalent)"). Table 1 says 0.085× for Llama, Table 7 says 0.151× for the same config/context/model. Pick one primary basis (fp32-at-rest, since that is what is actually stored), report it everywhere, and mention the fp16-equivalent once in §4.9.

### W7. The marquee result is not findable in under a minute.
"Marquee" appears 10 times, "flagship" 60 times, "crux" twice — three different words for two different configurations, and the paper never states plainly which single result the reader should take away. §4.3 is titled "Marquee" but concerns a *different* configuration (`r128-h1024-s32`) from the "flagship" (`r64-h256`) used in every other table. The reader has to infer this. Moreover, §4.3's own conclusion is that at matched bytes 4-bit KIVI ties the marquee, so the "marquee" is a win only over ThinK/Palu — which §4.6 then shows are not the relevant competitors. This sequencing buries the actual headline (the matched-bytes quantizer comparison in §4.6/Fig. 1) on pages 11–13.

### W8. Tone: disclaimers as a substitute for concision.
Counts: "honest/honestly" 33, "flagship" 60, "marquee" 10, "suggestive" 6, "not separated" 13, "crux" 2. Phrases like "characterize it honestly", "the honest headline is therefore", "Two honest notes travel with it", "Three qualifications travel with it", "with the honest caveat that" read as anxiety rather than rigor, and repeated 30+ times they undermine the paper: a reader begins to wonder what is *not* honest. Scientific honesty is demonstrated by the intervals and the paired tests, not by the adjective. Remove essentially every instance; "stored (fp32-at-rest)" is the technical term.

### W9. Figures: too few carrying the argument; the ones present have layout defects.
- **No figure for §4.1**, which is the central *method* claim (near-oracle vs. Oja/incremental SVD). The 1.01–1.03× and 1.3–3× numbers appear nowhere in a table or plot. This is the one place a reconstruction-error-vs-rank curve is mandatory.
- **No architecture/mechanism figure** for §2.3 (sinks | ring | gist | surprise tier) or for the rank-siphoning / r/n≈0.25 wall (§4.4), which is described as "a mechanism worth naming" but never plotted (retrieval vs. r/n would be one panel).
- **Figure 1** (p. 13) is a full float page with ~35% blank space above it; the legend says "aqua triangles" but they render green; joined 16K→32K segments overlap markers at identical x; the composed cell (hollow diamonds) is invisible on most panels. Twelve small panels with 2–4 points each is a lot of ink for little data — a 3×1 or a single "accuracy vs bytes" panel per family would suffice.
- **Figure 2** (p. 15): the annotation "2r/n = 0.125x (fp32-coord asymptote)" is drawn directly over the blue and orange lines and is illegible.
- **Table 5** encodes eight numbers per cell as "1.00/1.00/1.00/0.50 / 1.00/1.00/1.00/0.83". This needs to be a proper grid or a figure.
- **Table 7** is good but its "Bold = McNemar p<0.05" convention is not applied to Tables 1–3, 8.
- Captions are mostly self-contained (good), but Table 4's caption lists ten perplexity numbers in the body text again two paragraphs later (§4.4) — duplicate content.

### W10. Broken / misleading cross-references and internal inconsistencies.
1. §4.3 (p. 8): "4-bit KIVI arm, which scores variable-tracking 1.00 (12/12) at 32K (**Table 8**)". Table 8 is the official RULER run at *16K only*; the 32K 4-bit number is in Table 7.
2. §4.9 (p. 15): "the persistence win (§4.9 below)" — a section referencing itself.
3. §4.7 (p. 12): "**Two** lines of evidence say…" followed by items (1), (2), (3).
4. §4.2 (p. 7): "0.085–0.149×… 6.7–13× less than full KV". 1/0.085 = 11.8×, not 13×; 13× corresponds to the 32K value 0.075× from Table 2.
5. Abstract "0.28× (Qwen)" vs. body 0.275×; abstract "0.15×" vs. body 0.151×/0.150×; fine individually but the abstract rounds while claiming "honest" precision.
6. Conclusion says the marquee "leads palu-r0.5 (**not separated**)"; §4.3 says it "**separates** under the paired test (p=3.1×10⁻²)" but is "suggestive"; abstract says "suggestive separation". Three different verdicts on the same cell.
7. §4.9: decode residency "≈1.06× full KV" is 0.98 (workspace) + 0.085 (fp16-equiv stored). Under the paper's own preferred "honest" fp32-at-rest basis it would be ≈1.13×. The paper switches basis in the one place where the less favorable number would apply.
8. Perplexity for full-KV Llama at 32K is 6.975 in §4.3 (n=8 windows) but the flagship's 8.33 in §4.5/§4.6 is on WikiText-103 n=4 windows with no full-KV figure given; §4.9 gives full KV 7.26 at 64K. Three perplexity protocols, none labeled in a table.
9. Footnote 2 (p. 16) is typeset *between* the §5 heading and its first paragraph.
10. Reference formatting: [31] and [36] include "Oral"/"Spotlight" in the venue field; [7] lists "ICML; arXiv" as venue; [9] and [12] use "et al." in author lists while others list 14 authors. Inconsistent.

### W11. Related work is fair but misplaced and partly redundant with §1.
§1 already lists every eviction/quantization/low-rank method with citations; §3 lists them again with one clause each. Two "Contemporaneous 2026 work" paragraphs (§1 and §3) say the same thing. Merge: keep the taxonomy in §1 (one sentence per family), and make §3 only "Closest priors, and how we differ" (OjaKV, LESS, LoLA, ShadowKV, DLRA lineage). The OjaKV comparison — the single most important positioning — is numerically supported only by §4.1's un-tabulated 1.3–3× and OjaKV is never run as a baseline in the retrieval grids; the text should say so plainly in §3 rather than only in §2.1.

### W12. Conclusion (p. 18) is a second abstract.
It restates ~15 numbers and re-litigates the quantizer comparison. A conclusion should state the contribution, the one caveat, and the next step in ~120 words. The final sentence ("a fp16-storable gist and a fused factored-attention kernel are the program…") is the right ending.

---

## Questions for the authors

1. What are $n_{\text{sink}}$, $W$, and $\theta$ for the flagship? Are they included in the stored-state ratios?
2. Which configuration is "the" contribution: `r64-h256` (flagship) or `r128-h1024-s32` (marquee)? If a practitioner deploys one config, which one, and at which context?
3. §4.1 is on Llama-3.2-1B only. Does the 1.01–1.03× near-oracle ratio and the 1.3–3× gap to Oja hold on the 7B/8B models used elsewhere? Can you show the curve?
4. Why is OjaKV not run as a retrieval baseline given it is the closest prior? Even a single 16K Llama row would settle the positioning.
5. In the "1.06× decode residency" figure, why is stored state billed fp16-equivalent when everywhere else you insist on fp32-at-rest?
6. Can the "r/n ≈ 0.25 wall" be shown as a curve (retrieval vs. r/n for the three families) rather than two points?
7. Was the pre-registration of the 32K-Llama variable-tracking contrast recorded anywhere inspectable (OSF, a committed file with a timestamp)?

---

## Concrete rewriting suggestions

### Proposed abstract (≈160 words)
> The KV cache dominates long-context LLM memory. Existing low-rank compressors fix their subspace offline or in a one-shot prefill pass; we instead track it online with the rank-adaptive basis-update & Galerkin (BUG) integrator from dynamical low-rank approximation, whose error bound is independent of the smallest tracked singular value. On real pre-RoPE KV the streaming tracker is within 1.01–1.03× of the truncated-SVD optimum, where Oja's rule (OjaKV) is 1.3–3× worse. We combine the low-rank gist with a small exact tier selected by *surprise*—the residual to the tracked subspace—and a warm-up seed. On Llama-3.1-8B, Qwen2.5-7B and Mistral-7B, one configuration retains full single- and multi-value retrieval at 16K and 32K at 0.15–0.28× of full-KV stored state, and at matched bytes recovers multi-value retrieval that 2-bit KIVI-style quantization loses, at a fluency cost; 4-bit quantization at twice the bytes is not separated from it, and on official RULER the two are tied. Composing the gist with 4-bit coordinates reaches 0.048×. We characterize the method's limits—a retrieval wall at r/n≈0.25, a tier–gist rank-siphoning interaction and its fix, and an open weakness on variable tracking—and show that stored-state savings do not yet translate to decode residency without a fused kernel.

### Restructuring to fit 9 pages
**Main text (9 pp):** §1 Intro (1 p, with taxonomy; drop the "Scope and concurrency" versioning). §2 Method (1.5 pp) with **Algorithm 1** and a **mechanism figure** (sinks | ring | gist | surprise tier); one notation table. §3 Related work (0.5 p, closest priors only). §4 Experiments (4.5 pp): 4.1 near-oracle **with a figure**; 4.2 matched-bytes comparison (Table 7 + a slimmed Fig. 1) as the *first* and headline result; 4.3 cross-model tables 1–2 merged into one; 4.4 mechanisms (r/n wall figure + Table 4); 4.5 memory accounting (Fig. 2, one paragraph). §5 Limitations (0.75 p). §6 Conclusion (0.25 p).
**Appendix:** marquee §4.3 and Table 3 (or fold the one contrast into §4.2); Table 5/6 and the composite-competitor paragraph; §4.7 realistic-filler + LongBench + Table 8; §4.8 1B study; Fig. 3 cold-start; all provenance (SHAs, paths, pod-hours), all "earlier draft" corrections, footnotes 1–2, the port-defect disclosure, quantization-backend details (quanto vs hqq, G=64 vs G=32).

### Global edits
- Delete every instance of "honest(ly)", "marquee", "crux"; replace "flagship" with "the r=64 configuration" or define it once in a notation table and then use it sparingly.
- Report one memory basis (fp32-at-rest) in every table; state the fp16-equivalent once.
- Fix cross-refs W10.1–W10.3, the arithmetic in W10.4, and reconcile the three verdicts on the palu contrast (W10.6) to one phrase used everywhere ("leads on point estimate; paired p=0.03, not robust to correction").
- Remove author names, GitHub URL, date, and arXiv versioning language for the blind submission.
- Add the Bold-if-significant convention to all retrieval tables, or none.

---

## Scores

- **Soundness: 3** — The statistics are done properly and the negative results are disclosed; the small n and single-generator dependence are acknowledged. Some basis-switching (W10.7) shades borderline.
- **Presentation: 2** — Candidly, borderline 1. A 682-word abstract, ~17 pages of main text with no appendix, no algorithm box, no figure for the central method claim, overloaded notation, lab-notebook content in the body, and a tone that undercuts its own results. The figures that exist are the right ones and the captions are self-contained, which keeps this from a 1.
- **Contribution: 3** — Repurposing rank-adaptive BUG as a streaming KV tracker with surprise-selected escape hatch is a real and clearly articulated idea, and the mechanistic map (r/n wall, rank siphoning) is useful. The empirical edge over the fair quantization baseline is narrow, as the paper itself concludes.
- **Overall: 4 (Reject in current form; would revisit a rewritten version)**
- **Confidence: 4**

### Justification
The science here is more careful than most submissions I see, and the paper's willingness to report that 4-bit quantization is not separated from it and that decode residency is worse than full KV is to its credit. But NeurIPS reviewers and ACs evaluate the manuscript, and this manuscript does not yet function as a paper: it is roughly twice the page limit with no appendix, its abstract cannot be read, its method has no algorithm, its key method claim (§4.1) has no data displayed, and its prose is saturated with disclaimers and internal versioning that a conference reader should never see. Every one of these is fixable without new experiments, and the concrete restructuring above would, I believe, yield a paper in the 5–6 range on the same evidence. I recommend the authors do that rewrite rather than add more cells.


---

# Review — Reviewer 5 (Senior / AC-experienced; novelty & significance)

**Paper:** Dynamical Low-Rank Approximation as an Online KV-Cache Compressor

**Prior-work check performed before writing (paper judged on its own; web used only for closest priors):** OjaKV (arXiv 2509.21623, now published at Findings of ACL 2026) already (a) tracks a per-sequence subspace online, (b) keeps a hybrid full-rank tier selected by the reconstruction residual r_t = k_t − U U^T k_t (attention-weighted), (c) evaluates on RULER, LongBench, AIME and lm-eval-harness across four models, and (d) reports a real, measured decode-time memory reduction (16 GB → 11.6 GB at 32K) with FlashAttention compatibility and 11–13% latency overhead. An OpenReview submission "StreamingKV: Adaptive Low-Rank KV Caching with Test-time Updates" (ZFGtDo7RKI) is not cited; its content was behind OpenReview's challenge wall so I flag it rather than characterize it.

---

## Summary

The paper applies the rank-adaptive basis-update & Galerkin (BUG) integrator from dynamical low-rank approximation to the column-streamed, pre-RoPE key/value matrix of a transformer, producing an online per-sequence low-rank "gist." It pairs this with a small exact tier of tokens selected by the tracker's own out-of-subspace residual ("surprise"), attention sinks, a recent ring, and a warm-up seed. On a home-built RULER-style generator (n = 12–16 trials per cell) at 16K/32K on Llama-3.1-8B, Qwen2.5-7B and Mistral-7B, one configuration retrieves single/multi-value needles perfectly at 0.15× (Llama/Mistral) or 0.28× (Qwen) of full KV in "honest" fp32-at-rest stored bytes. The paper then reports, with commendable candor, that (i) a KIVI-scheme 2-bit quantizer at matched bytes ties on single-needle, wins fluency, and is not separated on the official RULER suite (where the method trails 0.79 vs 0.87 and full KV is 0.99); (ii) 4-bit KIVI at ~2× the bytes is not separated anywhere and dominates on Qwen at the same bytes; (iii) decode-time resident memory is ≈1.06× full KV because the method reconstructs then attends, and the only latency datum is ~10% slower; and (iv) variable tracking on the 1024-wide models is unsolved. The concrete systems payoff is a ~7× smaller persisted cache and 9–10× faster warm reload — a win the authors acknowledge is shared with quantization.

## Strengths

1. **Unusual candor.** The limitations section, the retraction of an earlier "beats SOTA at 10×" claim, the correction of baseline port bugs in the baselines' favor, the paired McNemar analysis on shared needles, and the separation of "stored" from "resident" memory are all exemplary reviewing hygiene.
2. **The rank-siphoning diagnosis (§4.4) is a genuinely interesting mechanistic observation**: content-selected exact tiers pull rank-carrying tokens out of the tracked subspace, which then pads with near-null directions and diverges. The relative singular-value floor is a sensible fix, and Table 4 is convincing within-method.
3. **The r/n ≈ 0.25 retrieval wall** is a clean, dimensionless failure mode with a matched control (Llama r=256 vs Qwen r=128). It is a useful warning for anyone coupling residual-based selection to a low-rank summary.
4. **The persistence measurement (§4.9, Fig. 3)** is a real, end-to-end, same-card number, and the 1/T amortization argument is correct and clearly presented.
5. Pre-RoPE factorization is the right operating point (though not new — ShadowKV, KVQuant).

## Weaknesses

1. **The novelty claim does not survive contact with OjaKV (§1, §3).** The paper positions itself against OjaKV on two axes: the tracker (BUG vs Oja) and "selecting the exact tier by content (surprise) rather than by position." The second claim is false. OjaKV's hybrid storage policy retains the top-k tokens by a reconstruction-residual score r_t = k_t − U_kU_k^⊤k_t (attention-weighted), which is the paper's "surprise" statistic in all but name. OjaKV also already provides an attention-error bound in terms of residual norms. That leaves only the first axis, and OjaKV is now a Findings-of-ACL-2026 paper that additionally evaluated on RULER, LongBench, AIME and lm-eval-harness with real decode memory savings. The remaining delta is therefore: a different online PCA update rule. See W2 for whether even that is a delta.

2. **"BUG on a column stream" is truncated incremental SVD; the paper concedes this in §2.1 and then reports beating incremental SVD "everywhere" (§4.1).** For a stream of appended columns there is no ODE, no time step, and no stiffness; the augmented BUG step is exactly: project the new block, QR the residual, re-factor the small augmented core, truncate. That is Brand (2006) with blocked updates and a Frobenius-tail tolerance — i.e., a deterministic sketch in the Frequent-Directions family. The σ_min-independent error bound of Ceruti–Lubich is a statement about integrating a matrix ODE and the authors admit it "is not a theorem for this column-stream setting." So the DLRA framing is a rebranding of a known streaming-PCA primitive, and the claim of beating "fixed-rank incremental SVD" must be an artifact of the comparison configuration (fixed rank without rank-adaptive truncation? single-column updates in bf16?). No details are given in §4.1: no table, no figure, no ranks, no definition of the iSVD baseline, and the study is on Llama-3.2-1B only. The single most-repeated quantitative claim in the paper — "near-oracle, 1.01–1.03× of truncated SVD; Oja 1.3–3× worse" — has no supporting table anywhere in the 21 pages.

3. **The tracker difference is never demonstrated end-to-end.** The whole argument for BUG over Oja is reconstruction error on 1B. There is no ablation in which the same cache architecture (same exact tier, sinks, ring, seed) is run with an Oja tracker or a plain truncated-iSVD tracker on the 7B/8B retrieval grid. Without that, a reader cannot tell whether any of the task-level results depend on the paper's one distinctive component. Given that §2.3 says "perplexity rides mostly on the gist; needle retrieval rides mostly on the exact tier," the a-priori expectation is that the retrieval results are insensitive to the tracker — which would mean the contribution is the exact tier, which is OjaKV's.

4. **There is no result in which the method is better than a trivial baseline at matched bytes on an external benchmark.**
   - Official RULER (Table 8, the only third-party benchmark): flagship 0.79 vs KIVI-2 0.87 at the same bytes vs full KV 0.99; on variable tracking 0.33 vs 1.00 for KIVI-4/Palu/ThinK.
   - "Marquee" (Table 3): the win is over think-c0.5 and palu-r0.5 at *different* (2–3× larger) byte budgets, and 4-bit KIVI at the *same* bytes scores 1.00 vs 0.94. The marquee is thus tied-or-beaten by a scalar quantizer at matched memory. Neither ThinK nor Palu is swept to the flagship's byte budget, so this is not a Pareto comparison.
   - Table 7 (own generator): wins are multi-value over 2-bit only, lost on official RULER; Qwen flagship is dominated outright by 4-bit.
   - §4.8 (1B frontier): "Expected Attention × TurboQuant and SnapKV × TurboQuant sit on or ahead of BUG's Pareto frontier through 0.08–0.18×."
   The paper's own §5 summary is accurate: the edge is confined to one composed cell at <0.05× on two of three models, at ~2× perplexity (9.25/17.1 vs 5.31/8.33), with Qwen diverging. That is a niche within a niche, with no official-anchor number.

5. **The headline memory number is not a memory number.** Decode residency ≈1.06× full KV (§4.9), batch size 1 only, ~10% slower per token on the one CPU datum, no GPU decode latency, and pre-RoPE reconstruction implies re-applying RoPE to all T reconstructed keys every step. OjaKV, by contrast, reports a real 27.5% VRAM saving at 32K with FlashAttention. A 2024–2025 accepted NeurIPS/ICLR KV paper (KIVI, KVQuant, SnapKV, Palu, ThinK, ShadowKV) delivered a wall-clock or VRAM win on a real serving path plus LongBench/RULER breadth across 4–6 models; this paper delivers neither. The persisted-cache win is real but is (a) shared with 2-bit quantization at 16–32K to within 0.01 s, and (b) a serving-systems result whose natural venue is MLSys, where it would need multi-tenant / prefix-cache-hit-rate context.

6. **Statistical power is too low for the number of cells.** n = 12 gives a Wilson lower bound of 0.758 for 12/12; a true accuracy of 0.80 passes 12/12 with probability 7%, yet the abstract headlines "perfect... retrieval with no detected loss." Perplexity is on n = 4 windows of 512 tokens (§4.6). LongBench is one document. The 64K point is n = 8. ~40 Wilson cells, one pre-registered contrast, and the pre-registered config (r128-h1024-s32) differs from the flagship (r64-h256), which raises the question of how the marquee config was chosen.

7. **A baseline was run and its results withheld.** §4.3 mentions "a ShadowKV decode-scope bug" among port defects fixed before comparison, but ShadowKV — the closest pre-RoPE per-sequence low-rank-key method, with a shipped kernel and real throughput gains — appears in no table. Likewise xKV and OjaKV itself are not run as task-level baselines.

8. **Missing/uncited related work.** StreamingKV (OpenReview ZFGtDo7RKI) appears to address exactly the online per-sequence low-rank axis and is not cited. Streaming PCA with finite-sample bounds (Oja/Krasulina analyses, Frequent Directions guarantees) is the right theoretical comparison point for a column stream, not ODE-based DLRA.

9. **Presentation.** The abstract is ~900 words and reads as a results dump with hedges. The word "honest" appears dozens of times — candor is a virtue, not a contribution. Internal week/commit identifiers ("Week-18 arm", "w19 pertrial", six git SHAs) belong in an appendix. Two memory conventions (fp16-equivalent 0.085× vs fp32-at-rest 0.15×) are used interchangeably in abstract, tables and text. §4.1, the foundation of the method claim, has no table or figure.

## Questions for authors

1. Reconcile the claim that you differ from OjaKV by "selecting the exact tier by content rather than by position" with OjaKV's hybrid storage, which retains top-k tokens by reconstruction residual r_t = k_t − U U^⊤ k_t. What is the actual difference in the selection rule?
2. What exactly is the "fixed-rank incremental SVD" baseline in §4.1 that BUG beats "everywhere," given that §2.1 states the augmented BUG step *reduces to* incremental SVD for column increments? Provide the table/figure with ranks, block sizes, and precision.
3. Run the full cache (tier + sinks + ring + seed) with (a) an Oja tracker and (b) a truncated-iSVD tracker on the 7B/8B 16K/32K retrieval grid and on official RULER. Do any task-level numbers change?
4. Where are the ShadowKV results referenced in §4.3?
5. Sweep ThinK and Palu (and OjaKV) down to the flagship's byte budget so Table 3 is a Pareto comparison.
6. What is GPU decode latency and peak VRAM for flagship vs full KV vs KIVI-2 at 32K on the A100, head to head?
7. Why is the "pre-registered confirmatory" configuration (r128-h1024-s32) different from the flagship evaluated everywhere else?
8. Is StreamingKV (OpenReview ZFGtDo7RKI) prior art for online per-sequence low-rank KV, and how does it differ?

## Limitations assessment

The limitations section is thorough and undersells nothing. The problem is not that limitations are hidden; it is that once read together — no resident memory saving, slower decode, no separation from 2-bit at matched bytes on the external benchmark, dominated by 4-bit on one of three models, variable tracking unsolved on two of three, 2× perplexity in the one exclusive band, batch size 1 — very little of a NeurIPS-level contribution remains. The paper describes itself as "an arXiv v1 timestamp of the mechanism" (§1); I agree with that self-assessment.

## Scores

- **Soundness: 2** (fair). Experiments that exist are carefully executed and correctly analyzed, but the central method claim (§4.1) has no reported evidence in the paper, the key end-to-end tracker ablation is missing, a run baseline is unreported, and n = 12 cells are underpowered for the "no detected loss" framing.
- **Presentation: 2** (fair). Meticulous but undisciplined: 900-word abstract, no thesis, dual memory conventions, internal provenance in main text, missing table for the core claim.
- **Contribution: 2** (fair). The distinctive component (BUG vs Oja/iSVD) is a reframing of truncated incremental SVD with no demonstrated end-to-end effect; the exact-tier idea is OjaKV's; the mechanistic findings are diagnoses of the method's own failure modes.
- **Overall: 3** (reject).
- **Confidence: 4.** I know the KV-compression and DLRA literatures well and verified the OjaKV overlap directly; I could not access StreamingKV's content.

## Justification

This is a carefully executed and admirably candid negative result dressed as a method paper. Stripped of hedges, it shows that an online truncated-SVD tracker plus a residual-selected exact tier — an architecture OjaKV already published with a different PCA update — can match, but not beat, a plain 2-bit KIVI-style quantizer at equal stored bytes on the only external benchmark, is dominated by 4-bit quantization at equal bytes on one of three models, and does not reduce decode memory or latency today. The one thing it does that quantization cannot — reach <0.05× stored state — costs a doubled perplexity, fails on Qwen, and has no external validation. The novelty claim ("first to track the subspace online with DLRA") is technically true and materially thin: on a column stream the BUG step is incremental SVD with a tolerance, and the paper says so. The rank-siphoning and r/n-wall analyses are the most durable content and would make a good workshop paper; the persisted-cache reload result would need multi-tenant serving context to be an MLSys paper. I would not fight for this paper in discussion; I would argue against acceptance while praising its honesty as a model for the field.

## What would flip me to accept

1. **An end-to-end tracker ablation on 7B/8B**: the identical cache with BUG vs Oja vs truncated-iSVD trackers on the 16K/32K grid *and* official RULER, showing the BUG tracker is necessary for task-level results. If the trackers tie, reframe as an analysis paper.
2. **A published §4.1 table** (per-rank reconstruction error, iSVD baseline definition, block sizes, precision) and an explanation of why blocked truncated iSVD is worse than blocked BUG when §2.1 says they coincide.
3. **A fused factored-attention kernel or an fp16-storable gist**, with head-to-head GPU peak VRAM and decode tokens/s vs full KV and KIVI-2/4 at 32K and 64K, batch >1. A real resident-memory win at ≤0.15× that quantization cannot reach at matched bytes would be a main-track result.
4. **Official RULER (all 13 tasks) and LongBench on all three families at 16K and 32K**, with OjaKV, ShadowKV, xKV, and KIVI (G=32 default) as baselines at matched bytes, plus ThinK/Palu swept to the same budget. A separated win over 2-bit at matched bytes on official RULER — or over 4-bit on Qwen — would change my assessment of significance.
5. **A solution to wide-KV variable tracking**, the paper's own hardest task and only current "marquee."
6. **A rewritten abstract and introduction (≤250 words)** stating one contribution and the one experiment that establishes it.

**Venue fit:** Not main-track NeurIPS/ICML/ICLR in current form. Best fit: an efficient-inference workshop (mechanism + failure-mode analysis), or MLSys after a kernel and a serving-context persistence study. As-is, arXiv is the appropriate home, which the authors themselves say.

Sources consulted: [OjaKV arXiv](https://arxiv.org/html/2509.21623); [OjaKV, Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.494.pdf); [StreamingKV OpenReview (content inaccessible, flagged)](https://openreview.net/forum?id=ZFGtDo7RKI); [PuzzleKV](https://arxiv.org/html/2608.23843); [Palu, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/7da6e0e00702c60607a6ae05c802ef85-Paper-Conference.pdf).
