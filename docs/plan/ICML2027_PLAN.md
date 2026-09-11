# kvdlra → ICML 2027: Experiment and Paper Plan

**Target:** ICML 2027 (abstract/paper deadline assumed ~Jan 28, 2027; confirm on the CFP when posted — ICML has shifted between late Jan and early Feb).
**Runway:** 20 weeks from Sept 9, 2026.
**Compute:** several hundred A100/H100 hours (plan budgets ~450 GPU-h with ~100 h reserve).
**Commitment:** build a fused factored-attention kernel so the storage ratio becomes a resident-memory and throughput win.

This plan continues from the PC simulation and code audit. It is organized around three **gates** that decide what kind of paper you are writing, then a build-out for whichever branch the gates select, then a hard-stop rewrite. Every experiment carries a spec (what, on what, how many, what counts as a pass) so nothing is re-litigated at the end.

---

## 0. The thesis you are aiming for

A NeurIPS/ICML-level version of this paper needs one sentence a reviewer can repeat. The candidates, in order of strength:

**A (method + systems, strongest).** *An online per-sequence low-rank KV cache with a residual-selected exact tier, executed by a factored-attention kernel, reduces resident KV memory to ~0.15× and increases decode throughput at 32K–128K while matching full-KV accuracy on RULER and LongBench across four model families — a regime scalar quantization cannot reach at matched bytes.*

**B (method, no kernel yet).** Same as A minus throughput, with the stored-state story and an honest residency figure. Weaker; that is essentially the paper you have, and it was rejected on evidence gaps, not framing alone.

**C (analysis).** *We show that residual-selected exact tiers, not the low-rank tracker, carry retrieval in hybrid low-rank KV caches; characterize the r/n retrieval wall and a loss-of-orthonormality failure mode that all incremental-SVD trackers share; and give design rules.* ICLR-friendly, honest, and publishable if Gate 1 fails.

Gate 1 below decides between A/B and C. Gate 3 decides between A and B. Do not write a word of the new paper before Gate 1 closes.

---

## 1. Phase 0 — Hygiene that must precede every new measurement (Weeks 0–1, ~10 GPU-h)

These fix things the audit found in the harness itself. Any result produced before they land is not citable.

**0.1 Orthogonality guard on U.** Add a periodic re-orthonormalization (thin QR of U every K absorbs, K≈64) and a tripwire that logs ‖UᵀU−I‖_F per layer and aborts/flags above 1e-3. Log effective rank (count of σ above floor) per layer per absorb. *Pass:* Table-4 Qwen r=256 cells re-run with the guard; report whether the 27,531 divergence disappears without the floor. This determines whether "rank siphoning" survives as a mechanism or is replaced by "orthonormality ratchet" in the paper.

**0.2 Bill effective rank, not configured rank.** `_footprint` in `w10_frontier.py:546-548` must read the tracked rank from the layer state. Re-emit every floor-on row.

**0.3 Score both trackers the same way in §4.1.** Reconstruction error for every method must be computed from the *stored representation* (rot-carried coordinates for BUG, stored factors for iSVD, sketch for FD, U for Oja) — never by re-projecting the full matrix. Re-generate `oja_vs_bug.json` and `pilot.json` on the 8B models (see 1.2).

**0.4 Harness fixes.** (a) Rename `palu-r0.5` → `svdK V-oracle-r0.5` (or drop it; see 2.3). (b) Per-trial exceptions must fail the trial, not silently reduce n. (c) Fix task-name collisions between in-house and official generators. (d) Commit raw logs (`w19_harvest/*.raw`) that disambiguate Table 5. (e) Un-squash history going forward: one commit per pod launch with the pod matrix file, so pre-registration is dateable. (f) Perplexity: switch to WikiText-103 **test** (or PG-19 validation), store per-window NLL, always emit paired CIs.

**0.5 Pre-registration infrastructure.** For each phase below, commit `prereg/<phase>.md` naming: arms, tasks, n, primary contrast(s), family size for correction, and the decision rule — *before* the pod launches. Use OSF or a signed tag if you want it externally dateable. This is what lets you write "pre-registered" later without a reviewer disputing it.

---

## 2. Phase 1 — The three gates (Weeks 1–5, ~120 GPU-h)

### Gate 1: Does the tracker matter? (Weeks 1–3)

The `swap` pod exists; extend it.

**1.1 Tracker-swap ablation at task level.** Same cache (sinks 4, ring 32, h=256 surprise tier, seed on, floor as decided in 0.1), swapping only the gist tracker:
(a) incremental SVD (current), (b) Oja with the Week-2 tuned schedule *and* a re-tuned schedule on held-out 8B KV, (c) Frequent Directions ℓ=2r, (d) frozen one-shot prefill SVD (xKV/ShadowKV-style: SVD of first 4K chunk, fixed thereafter), (e) **no gist** — tier + ring only, with the freed bytes given to the tier (h≈1024) so bytes match, (f) random orthonormal fixed basis (a floor control).
Models: Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B. Contexts 16K, 32K. Tasks: single, multi-key, multi-value, variable-tracking on the in-house generator **with the Phase-2 generator fixes (2.1)**, n=24 per cell (2 haystacks × 3 depths × 4 codes) plus perplexity on 16 windows with paired CIs.
Pre-registered readings: if (a) is not separated from (c)/(d) on any task and perplexity is within 0.02 bits/token, the tracker is not the contribution → **Branch C**. If (a) beats (d) and (e) on retrieval or perplexity in ≥2 families with Holm-corrected p<0.05 → the online-tracked gist is doing work → **Branch A/B**. Budget ~40 GPU-h.

**1.2 §4.1 done properly.** Reconstruction error vs rank r∈{16,32,64,128,256}, all layers (not layer 8 only), K and V separately, on 8B Llama/Mistral/Qwen at 16K and 32K real KV (dump 8 documents from PG-19/GovReport, not C4 snippets), fp32, block sizes 16 and 128, stored-representation metric (0.3). Methods: truncated-SVD oracle, iSVD/BUG, FD(ℓ=r and 2r), Oja (grid tuned on 2 held-out docs), frozen prefill SVD. Report mean ± SE over docs. Budget ~10 GPU-h. *This becomes Figure 2 in any branch.*

### Gate 2: Is the multi-value edge over 2-bit real or a prefill-protocol artifact? (Weeks 2–4)

**2.1 Fix the generator first.** Current: one fixed 10-sentence haystack, depth a deterministic function of trial index, single needle always mid-depth. New: ≥4 distinct haystacks (PG-19 chapters, arXiv text, Wikipedia, synthetic essays), needle depth sampled from {0.05, 0.2, 0.4, 0.6, 0.8, 0.95} with balanced design, ≥2 code families. Record depth and haystack id in the per-trial log. Verify pairing still holds (same seed → same prompt across arms).

**2.2 Faithful KIVI.** Implement or vendor KIVI at its published operating point: G=32, R=128 residual, per-channel keys / per-token values, **full-precision prefill** (no chunked dequant), and ALSO the paper's streaming-protocol variant, clearly labelled. Add KVQuant-style pre-RoPE 2/3-bit dense-and-sparse as a second quantizer. Count scales/zeros. Run flagship vs KIVI-2 vs KIVI-4 vs KVQuant-3 at matched stored bytes on the 2.1 generator, 3 families × 16K/32K, n=48 per cell (4 haystacks × 6 depths × 2 codes). Pre-registered primary contrast: flagship vs faithful KIVI-2 on multi-value, pooled over families (McNemar), family size = 12 (3 families × 4 tasks), Holm. *Pass for Branch A/B:* multi-value separation survives Holm against the faithful (full-prefill) KIVI-2. *If it does not:* the paper loses "beats 2-bit" and keeps only the sub-0.05× composition band; that is survivable in Branch A (throughput carries the paper) but fatal to Branch B. Budget ~45 GPU-h.

**2.3 Baseline honesty pass.** Replace the SVD-oracle "Palu" with either real Palu (their repo, G-rank allocation, no fine-tune — report which) or drop the name entirely and keep the oracle as an explicit *upper bound for static low-rank*. Run ThinK composed with SnapKV as in the ThinK paper (that is its intended use), not standalone. Add SnapKV, PyramidKV, and ExpectedAttention swept to the flagship's bytes (k∈{0.10, 0.15, 0.25}). Report ShadowKV (already run) and `ea-k0.25` (already run, competitive). Run OjaKV end-to-end from their code at matched bytes on Llama-3.1-8B 16K/32K. Budget ~25 GPU-h.

### Gate 3: Can the kernel deliver a resident-memory and throughput win? (Weeks 1–5, parallel workstream)

**3.1 Kernel design decision (Week 1).** The pre-RoPE gist means K_t = R_t U c_t with a per-token rotation R_t, so q·K_t ≠ (Uᵀq)·c_t. Three options; pick one in Week 1 with a written ADR:
- **(i) Post-RoPE tracking at higher rank.** Track post-RoPE keys; then scores = (Uᵀq)ᵀ C is an r×T GEMV — trivially FlashAttention-compatible with r-dim keys. Costs rank (pre-RoPE roughly halves error at matched budget per your §2.2). Cheapest kernel; test whether r=128 post-RoPE matches r=64 pre-RoPE on accuracy at equal bytes.
- **(ii) RoPE-factored pre-RoPE kernel.** Exploit that RoPE acts on 2-D pairs: q·R_t U c_t = Σ_pairs [cos(θ_j t), sin(θ_j t)] · (Q_jᵀ U_j c_t). Precompute Q_jᵀU_j (2×r per pair, n/2 pairs → n×r work per step, shared across all T tokens), then per token an r-dim dot per pair — O(nr + Tr·(n/2)/... ) — needs care; write out the FLOPs before committing. This is what Palu/ShadowKV had to solve; read their kernels.
- **(iii) Tile-wise reconstruction inside a Triton attention kernel.** Materialize U·C_tile → RoPE → QKᵀ per tile in SRAM; never write n-dim keys to HBM. Memory win is complete; compute is O(nrT) extra per step. Feasible at r=64 (n×r GEMM per 64-token tile is small). This is the most general and matches your current pre-RoPE design exactly.
Recommendation: prototype (iii) in Triton first (it preserves everything you have), and run (i) as an accuracy experiment in parallel because if post-RoPE r=128 is accuracy-equivalent, the kernel becomes near-trivial.

**3.2 Kernel milestones.** Week 2: correctness (max abs diff vs reconstruct-then-attend < 1e-2 in bf16) on single layer. Week 3: full model decode path with tier + ring + sinks as an extra dense block appended to the tile loop. Week 4: measured peak VRAM (`torch.cuda.max_memory_allocated`) and ms/token for full KV, flagship-kernel, flagship-reconstruct (current), KIVI-2/4 (with their kernels or quanto), at 16K/32K/64K/128K, batch 1 and 8, Llama-3.1-8B on one A100 and one H100. Week 5: **Gate 3 decision.** *Pass for Branch A:* flagship-kernel resident KV ≤ 0.25× full at 32K and decode tokens/s ≥ full KV at 64K, batch ≥4. If it lands at ≥ full-KV throughput but not below, Branch A still works on the memory axis alone (that lets you fit batch 8 where full KV fits batch 2 — measure that). Budget ~30 GPU-h (mostly engineer-time).

**3.3 fp16/bf16 gist storage.** There is no evidence fp16 storage fails; the test file calls bf16 storage + fp32 core "recommended". Run the flagship with coordinates and U stored bf16 (fp32 accumulate in the step) on all three families 16K/32K; if accuracy and perplexity hold within CI, the honest ratio halves (0.15× → ~0.08×) and the whole quantizer comparison shifts. 8 GPU-h. Do this in Week 2 — it is the cheapest large win in the plan.

### Gate review (end of Week 5)

Write a one-page decision memo: Gate 1 result (tracker matters? yes/no), Gate 2 (edge over faithful 2-bit? yes/no), Gate 3 (kernel VRAM/throughput? numbers). Branch:

| G1 | G2 | G3 | Paper |
|---|---|---|---|
| yes | yes | yes | **A** — method+systems, ICML main track, strong |
| yes | no | yes | **A′** — systems-led; the pitch is memory/throughput at full accuracy, composition to <0.05×; no "beats 2-bit" claim |
| yes | any | no | **B** — method; ICML plausible but borderline; consider ICLR 2028 after kernel |
| no | any | yes | **A″** — the kernel + exact-tier cache is the contribution; drop "DLRA" from title; tracker becomes an implementation detail with FD/iSVD interchangeable |
| no | any | no | **C** — analysis paper; target ICLR 2028 / a strong workshop now |

---

## 3. Phase 2 — Build-out for Branch A (Weeks 6–14, ~250 GPU-h)

If Branch C is selected, skip to §3-C.

### 3.1 Benchmark breadth (Weeks 6–10)

**Models (4 families, 2 sizes).** Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B, plus one wide-KV and one non-GQA or larger model to answer "does it generalize": Llama-3.1-70B (or 3.3-70B) at 16K on 2×A100 (batch 1) for a single RULER row, and Gemma-3-12B or Phi-4-14B. The wide-KV variable-tracking weakness must be addressed or clearly scoped — see 3.3.

**Official RULER.** All 13 tasks, 16K and 32K, 3 core families, ≥200 samples/task (official is 500; 200 gives Wilson half-widths ≈±0.06, enough to separate 0.05 gaps at pooled level). Arms: full, flagship (kernel path), flagship-bf16-gist, faithful KIVI-2, KIVI-4, KVQuant-3, SnapKV@0.15, ThinK+SnapKV, OjaKV, composed gist+4-bit (0.048×). ~120 GPU-h — the single biggest line item; cut to 2 families × 32K if behind schedule.

**LongBench (v1 English subset + LongBench-v2 if time).** Full task set for flagship, full, KIVI-2, KIVI-4, SnapKV on Llama-3.1-8B and Qwen2.5-7B. ~40 GPU-h. Report per-task and macro-average; this kills the "single Qasper document" criticism.

**Realistic-filler NIAH.** Retire the in-house generator as a headline source; keep it as an appendix diagnostic only (depth sweeps, r/n wall curve).

**Statistics protocol (fixed now, in `prereg/phase2.md`).** Primary: flagship vs full-KV non-inferiority on RULER macro-average, margin 0.03, paired bootstrap over samples. Secondary (Holm, family = number of arms × 2 contexts): flagship vs each baseline at matched bytes. Perplexity: PG-19 test, 32 windows × 2048 tokens, paired CI. Every table shows n and CI. The words "perfect", "no detected loss" and "not separated" do not appear; "non-inferior within δ at 95%" replaces them.

### 3.2 Systems results (Weeks 6–12)

- Peak resident KV VRAM and end-to-end decode tokens/s at 16K/32K/64K/128K, batch {1, 4, 8, 16}, A100-40GB and H100-80GB, for full/flagship-kernel/KIVI-2 (with KIVI's Triton kernels)/SnapKV. Report "max batch that fits" per method per context — this is the number practitioners care about.
- TTFT / prefill overhead with the streaming ingest (block 128/16) vs plain prefill.
- Persisted-cache reload (you have this; keep it, one figure).
- Ablate block size (16/64/128) and QR period K on throughput; show the 1/T amortization curve to 128K.
- Compute a FLOP model for the kernel and validate against measured.

### 3.3 Fix or scope the wide-KV variable-tracking weakness (Weeks 6–11)

Variable tracking is a chain: a summary must preserve *relational* structure, not just point facts. Three cheap interventions to test on Llama/Mistral 16K/32K, n=48: (a) query-aware tier admission — re-score the tier by max attention from the current query block (OjaKV-style attention weighting) rather than pure residual; (b) a small "recency-weighted surprise" so recent chain hops stay verbatim longer; (c) per-head-group bases instead of one shared basis over all heads (n=1024 shared may be the real problem; 8 groups of 128 with r=8 each costs the same bytes). If none reaches ≥0.85, scope it in Limitations with the mechanism (chain hops land in the gist), not as an open mystery.

### 3.4 Mechanism section, made rigorous (Weeks 8–12)

- **r/n wall:** retrieval vs r/n curve for 3 families, with and without score-rank cap, n=48, 8 rank points. One figure.
- **Orthonormality ratchet (or rank siphoning, whichever 0.1 shows):** ‖UᵀU−I‖ and effective-rank traces over the stream for r∈{64,128,256}, floor on/off, guard on/off; plus perplexity. One figure. State the fix (QR guard and/or floor) and its cost.
- **Composition:** gist bytes × coordinate bits Pareto on Llama and Mistral with KIVI/KVQuant/TurboQuant as pure quantizers at 2, 3, 4 bits and SnapKV×KIVI composites — one Pareto figure per family, official RULER macro-average on the y-axis.
- **Theory (small but real).** Replace the CKL bound with what applies: a Frequent-Directions-style deterministic guarantee for the truncated column-stream sketch (‖MMᵀ − UBBᵀUᵀ‖₂ ≤ ‖M−M_r‖_F²/(ℓ−r) with your rank-adaptive ℓ), and a one-paragraph statement of what the QR guard buys (bounded orthonormality error per step ⇒ no exponential ratchet). Two lemmas with proofs in the appendix; no theorem you cannot prove. If you want the DLRA connection, keep it to one honest paragraph in Related Work: "the augmented step coincides with the BUG basis-update step when the flow is an exact projection; we do not use its ODE error bound."

### 3.5 Reproducibility package (Weeks 12–14)

Public repo tag with: one-command RULER/LongBench runners, the kernel with tests, `prereg/*.md`, per-sample logs for every table, a `make tables` that regenerates every number in the paper from logs (the audit showed your numbers already reproduce — make that a feature reviewers can run).

---

## 3-C. Build-out for Branch C (analysis paper) — only if Gate 1 fails

Weeks 6–12, ~120 GPU-h. Keep 3.4 in full (it *is* the paper), plus: (a) the tracker-swap grid extended to 4 families and official RULER; (b) a tier-only vs tier+gist bytes-matched sweep showing exactly where the gist earns its bytes (perplexity, long-range coherence) and where it does not (retrieval); (c) a study of residual-selection failure modes shared by OjaKV/LESS/LoLA (rank siphoning, chain-task failure) reproduced in OjaKV's own code; (d) the r/n wall as a general design rule with a simple model. Title along the lines of "What the low-rank gist actually carries: an analysis of hybrid low-rank KV caches". Target ICLR 2028 or an ICML 2027 submission with an explicit analysis framing (ICML does accept these when the analysis is decisive and reproducible).

---

## 4. Phase 3 — The paper (Weeks 13–20)

Hard rule: **no new experiments after Week 16** except reviewer-anticipated single cells.

### 4.1 Structure (9 pages main + appendix)

1. **Introduction (1 p).** Problem, the three families, the online-per-sequence gap, one-paragraph contribution list (≤4 bullets), Figure 1 = the headline Pareto (RULER macro-avg vs resident KV bytes, all methods, with throughput as marker size or a twin panel).
2. **Method (1.75 p).** Notation table. Algorithm 1 (ingest block: project, residual QR, augmented SVD, truncate, QR guard every K, rotate coordinates, surprise-score, tier admit/evict, seed). Algorithm 2 (kernel: tile loop). Mechanism figure: sinks | ring | tier | gist, and the kernel data flow. State plainly: "the update is block incremental SVD with rank-adaptive truncation; we give a Frequent-Directions-type guarantee (Lemma 1)."
3. **Related work (0.5 p).** Closest priors and exact deltas: OjaKV (same residual tier, Oja tracker, no kernel/no 4-family RULER), ShadowKV (pre-RoPE static SVD + offload), Palu/xKV/Eigen (static), KIVI/KVQuant (bit axis), StreamingKV (check and cite), LESS/LoLA.
4. **Experiments (4.5 p).** 4.1 Setup and statistics protocol (½ p). 4.2 Main results: official RULER + LongBench table, 4 families (1 p). 4.3 Systems: VRAM, throughput, max-batch, reload (¾ p). 4.4 Matched-bytes quantizer comparison and composition Pareto (¾ p). 4.5 Ablations: tracker swap, tier-only, seed, floor/guard, block size (¾ p). 4.6 Mechanisms: r/n wall, ratchet (¾ p).
5. **Limitations (0.5 p).** Honest but not self-flagellating: wide-KV chain tasks (if unsolved), Qwen fluency at r=64, kernel tested on two GPU types, batch ≤16.
6. **Conclusion (0.25 p).**
Appendix: full per-task tables, prereg documents, proofs, kernel details, generator specs, all provenance.

### 4.2 Writing rules (from R4, adopted verbatim)

- Abstract ≤ 200 words, ≤ 8 numbers, one memory convention (resident VRAM measured; stored bytes in appendix).
- Delete "honest(ly)", "marquee", "flagship", "crux". Name the method once (pick a name — "kvdlra" or something that does not promise DLRA if Gate 1 says the tracker is interchangeable).
- One configuration is *the* method. If the paper needs two, explain in one sentence why a practitioner picks each.
- Bold = significant after correction, in every table or none.
- No week labels, SHAs, pod-hours, "earlier draft" language in the main text.
- Every claim of superiority carries a corrected p or a CI in the same sentence; every null is stated as "non-inferior within δ" or "underpowered to detect < X", never as a tie.
- Double-blind: no names, no repo URL (anonymous link), no arXiv-version narrative.

### 4.3 Schedule

| Week | Dates | Deliverable |
|---|---|---|
| 0–1 | Sep 9–22 | Phase 0 hygiene merged; prereg/phase1.md committed; kernel ADR written; §4.1 KV dumps collected |
| 2–3 | Sep 23–Oct 6 | Gate 1 tracker swap running; faithful KIVI + generator v2 done; bf16-gist result; kernel correctness |
| 4–5 | Oct 7–20 | Gate 2 matched-bytes run; baseline honesty pass; kernel measured; **Gate review memo Oct 20** |
| 6–8 | Oct 21–Nov 10 | Official RULER 3 families launched; LongBench launched; wide-KV interventions; §4.1 figure final |
| 9–11 | Nov 11–Dec 1 | Systems table complete; mechanism figures; 70B/12B rows; theory lemmas drafted |
| 12–14 | Dec 2–22 | Composition Pareto; repro package; **all experiments frozen Dec 22 except single cells** |
| 15–16 | Dec 23–Jan 5 | Full draft v1 (9 pp + appendix); internal blind review by 2 people not on the paper |
| 17–18 | Jan 6–19 | Revise; run an adversarial review pass (repeat the 5-agent simulation on the new PDF); fix |
| 19–20 | Jan 20–28 | Final polish; supplementary zip; anonymous repo; submit ≥48 h before deadline |

### 4.4 GPU-hour budget (A100-equivalent)

| Item | Hours |
|---|---|
| Phase 0 re-runs | 10 |
| Gate 1 tracker swap + §4.1 dumps | 50 |
| Gate 2 faithful quantizers + generator v2 | 45 |
| Baseline honesty pass (Palu/ThinK+SnapKV/SnapKV sweep/OjaKV/ShadowKV) | 25 |
| Kernel benchmarking | 30 |
| bf16 gist | 8 |
| Official RULER, 3 families × 2 ctx × 13 tasks × 200 × ~10 arms | 120 |
| LongBench, 2 families × 5 arms | 40 |
| 70B + 12B rows | 30 |
| Wide-KV interventions | 25 |
| Mechanism figures (r/n wall, ratchet traces) | 20 |
| Composition Pareto | 25 |
| Reviewer-anticipated cells / reserve | 100 |
| **Total** | **~530** |

If you are closer to 300 h: drop the 70B row, run RULER at 2 families × 32K only, and LongBench on Llama only. Never cut Gates 1–3.

---

## 5. Risks and pre-decided responses

| Risk | Signal | Response |
|---|---|---|
| Gate 1 fails (tracker interchangeable) | FD/iSVD/frozen tie the tracker on all tasks | Branch A″ if kernel works (cache + kernel is the contribution; tracker is a plug-in), else Branch C. Do not try to rescue "DLRA" by finding a cell where it wins. |
| Gate 2 fails (faithful KIVI-2 matches multi-value) | Holm p > 0.05 pooled | Drop the claim; lead with memory/throughput and the <0.05× band. Say plainly that at ≥0.15× a good 2-bit quantizer is competitive on accuracy. |
| Kernel throughput below full KV | tokens/s < full at 64K, batch 8 | Report memory win and max-batch; frame throughput as future work honestly; still Branch A′ if VRAM ≤ 0.25×. |
| Qwen r=64 diverges at 32K even with guard | ppl > 2× full | Use r=128 + guard for Qwen as the stated per-family rank rule (bytes rise to ~0.28×); explain via n=1024 wide KV. |
| Wide-KV variable tracking stays < 0.7 | none of 3.3 works | Scope it as the one task class where hybrid low-rank caches fail, with the mechanism; cite it as shared with OjaKV if reproduced there. |
| OjaKV end-to-end matches you at matched bytes | RULER macro within CI | Then the delta is the kernel + guard + composition. Say so; co-position rather than differentiate on the tier. |
| Time overrun | RULER not done by Week 11 | Cut to 2 families × 32K; drop LongBench-v2; keep 70B out. |
| StreamingKV (OpenReview) is prior art for online per-sequence low-rank | on inspection | Cite, position, and make sure your kernel/guard/composition are the differentiators. |

---

## 6. What to do this week

1. Merge the QR guard + orthonormality tripwire + effective-rank billing; re-run the five Table-4 cells. (Decides the mechanism story.)
2. Fix §4.1 scoring and dump 8B KV for the rank-sweep figure.
3. Write `prereg/phase1.md` and the kernel ADR (option (iii) as primary, (i) as parallel accuracy test).
4. Launch the tracker-swap pod with the six trackers on Llama 16K as a smoke run while generator v2 is built.
5. Run bf16-gist storage on all three families — cheapest potential halving of your bytes.
6. Rename the Palu arm in the codebase today so no new result inherits the label.

The ordering matters: nothing in the rewrite is worth doing until Gate 1 tells you which paper you are writing, and nothing in the benchmark grid is worth running until the harness hygiene in Phase 0 is merged.
