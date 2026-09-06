# Prior-work / novelty review — kvdlra exit-gate re-review (2026-09-06)

Reviewer dimension: **novelty, positioning, fairness of the 2-bit baseline, related-work accuracy, missing citations**.
Repo `/Users/hari/Desktop/kv-dlra` @ `week7` (paper HEAD `5bc772f`). $0, read-only.

## 0. What I read

`docs/reviews/2026-09-06/review-briefing.md`; `paper/main.tex` (1016 lines, in full); `paper/refs.bib` (35 entries);
`docs/reviews/2026-09-01/review-prior-work.md` (previous panel); `src/kvdlra/quant/kivi_cache.py`,
`src/kvdlra/quant/polar.py`, `src/kvdlra/accounting.py:420-447`, `results/w18_harvest/quant-findings.md`,
`results/w19-a1-report.md`, `scripts/pod/w19.sh`, `scripts/w10_frontier.py` (quant arm build);
`src/kvdlra/integrators/{streaming.py,streaming_torch.py,frequent_directions.py,oja.py}` headers;
`docs/week1.md`, `docs/week2-pilot.md`, `docs/week5-plan.md:57-71`, `docs/PLAN.md:23-35`, `docs/week19-kickoff.md:112-121`;
`tests/test_w19_quant_kivi.py` (test names). External papers from my own knowledge (flagged "verify" where I am not certain).

## 1. Verdict

**Score: 5/10 as worded (borderline reject on this dimension); 7/10 after the CPU-only fixes in §6, with the tracker-swap
ablation making it firm.**

**FATAL (as worded, fixable in hours): the paper's central novelty sentence is contradicted by a prior work the authors
implement as a baseline and do not cite.** `paper/main.tex:118-119` ("None tracks the subspace *online, incrementally*, as
tokens arrive"), `:169` ("the online-DLRA axis remains, to our knowledge, unoccupied"), abstract `:52-53` ("Existing low-rank
methods fix the subspace at calibration time"), and `:314-317` ("Every one of these fixes its subspace statically ... none
updates it incrementally as the sequence streams"). **OjaKV** (arXiv 2509.21623, Sept 2025) is an online, training-free,
per-sequence low-rank KV compressor that adapts its projection basis with Oja's rule during prefill and decode, with a hybrid
store keeping the first and most-recent tokens verbatim (from my knowledge of its abstract; verify). The repo knows it:
`src/kvdlra/integrators/oja.py:1-12` ("Oja's-rule online subspace tracker ... (OjaKV baseline) ... OjaKV, arXiv:2509.21623"),
`docs/week5-plan.md:57` ("OjaKV ... **online** low-rank via Oja's rule ... online rival — already beaten in Week 2"),
`docs/PLAN.md:35,702`. The paper then reports "beats Oja's rule by 1.3–3.0×" (`main.tex:383`) with no citation at all — not
to Oja (1982), not to OjaKV. A reviewer who knows OjaKV reads this as either careless or evasive, which is far worse than the
actual novelty gap (the BUG integrator, the surprise tier, the seed, the mechanistic map, and the fair-quant comparison all
survive OjaKV). Fix = cite, reframe the axis as "opened by OjaKV, occupied here with a near-oracle rank-adaptive integrator
and a content-selected exact tier", and (ideally) show the tracker matters end-to-end (§6, fix B).

The rest of the dimension is in good shape relative to 2026-09-01: 35 references, a real related-work section, the EA
name-collision note (`:294-297`), LESS/LoLA owned (`:319-332`), DLRT scoped (`:334-343`), the fair 2-bit arm built and
disclosed. What the closing revealed is that the *exclusive-band* rhetoric has migrated from "eviction cannot enter" to
"quantization/Palu/ThinK cannot reach" (`:584, :598-599, :679, :843-845`), and that second formulation is also wrong in places
(§3.5).

## 2. Previous panel's fatal set — status in my dimension

| 2026-09-01 item | Status | Evidence |
|---|---|---|
| No 2-bit quant baseline (fatal to C1) | **Closed**, honestly | `main.tex:630-681`, Table `tab:fairquant`; `results/w19-a1-report.md`; the multi-value edge does not transfer to official RULER and the paper says so (`:760-772`, `:957-966`) |
| "regime eviction cannot enter" wording | Closed; **re-emerged** as "floored by construction" for Palu/ThinK (`:598-599`) and "no scalar quantizer can" (`:160-161, :843-845`) | §3.5 below |
| Citation debt "essentially total" (3 refs) | Mostly closed (35 refs); **OjaKV, ShadowKV, Brand/FD/Oja, GEAR, PolarQuant, merging family, persistence line still absent** | §3.4 |
| LESS/LoLA framing not owned | Closed | `:319-332` accurate |
| Eigen-Attention / Expected-Attention collision | Closed | `:294-297`; `refs.bib:109,182` |
| DLRT scoping | Closed | `:334-343` ("first ... inference-time, streaming ... activation KV") — still true after OjaKV only if scoped to *DLRA*; OjaKV is online-low-rank but not DLRA |
| MomentKV/ResKV surprise-signal overlap (flagged 09-01 §1b.2; instructed in `docs/week19-kickoff.md:113`) | **Not done** | `:345-352` describes both as "eviction-plus-correction" and never says MomentKV's token score is a residual norm w.r.t. a summary subspace, i.e. the same signal as "surprise" (`:227-229`) |

## 3. Findings by question

### 3.1 Is the online-DLRA compressor genuinely unoccupied? (Q1)

**Against the named set — Palu, Eigen Attention, xKV, MatryoshkaKV, LoRC, LESS, LoLA, ShadowKV, MLA:** yes, the *DLRA
integrator + rank-adaptive online basis + surprise-selected exact tier* combination is not in any of them, and the paper's
one-line characterizations are accurate (§3.3). Two of them are mis-shelved by the abstract's "calibration time" sentence:
xKV and ShadowKV both compute a per-sequence SVD at prefill (the intro handles xKV correctly at `:117-118`; ShadowKV is not
in related work at all, see §3.4).

**Against OjaKV: no, the *online per-sequence adaptive-subspace* axis is occupied** (see §1). What remains genuinely
BUG's, and what the paper should claim instead:
1. the integrator: rank-adaptive augmented BUG with Frobenius-tail truncation, near-oracle (1.01–1.03× of truncated SVD)
   where Oja is 1.3–3.0× worse at matched memory (`docs/week2-pilot.md:66-69`, `figures/week2/oja_vs_bug.png`) — Oja's rule
   is step-size-sensitive and never revisits a token (`src/kvdlra/integrators/oja.py:28-59`); this is a real, quantified
   improvement on the same axis;
2. the exact tier selected by *content* (out-of-subspace residual) rather than by *position* (OjaKV keeps first/recent
   tokens verbatim — the same sink+ring pattern as `main.tex:221-226` — but, to my knowledge, has no content-selected
   exact tier; verify);
3. the warm-up seed, score-rank decoupling, the r/n ≈ 0.25 wall, rank siphoning + `min_sv_frac`, and the paired fair-quant
   and official-RULER evidence — none of which OjaKV has.

**A second, deeper positioning risk the paper does not address: the augmented-BUG column-append step is algebraically Brand's
incremental SVD.** The repo says so itself: `src/kvdlra/integrators/streaming.py:29-52` ("for a *rank-one* data increment
[the augmented BUG step] collapses to: append the incoming column's U-coordinates as a new column of the square-root core B,
re-SVD the small augmented factor, and keep the leading directions"; the ODE is the artificial relaxation field
`F(Y) = Y_target − Y` with substeps "integrated exactly"), and `src/kvdlra/integrators/frequent_directions.py:14-16` ("Our
problem is not integrating an ODE -- it is *sketching a streaming data matrix* -- and FD is the canonical deterministic
algorithm for exactly that"). Steps (i)–(iv) at `main.tex:188-195` are the standard append-update of Brand (2002, 2006) /
Baker–Gallivan–Van Dooren (2012, incremental dominant subspace) without tracking V. The paper's framing — DLRA is "a
numerical-analysis tool built for exactly this problem" (`:54-55`), the σ_min-independent error bound "is what a streaming KV
compressor needs" (`:126-130`) — will be attacked by any NLA-literate reviewer: there is no matrix flow, the bound is about
integrating flows on the rank-r manifold, and the algorithm that results is a known incremental SVD with a rank-adaptive
truncation rule. Worse, `:382` claims BUG "beats incremental SVD everywhere" without saying what that baseline was
(`docs/PLAN.md:488` "Brand-style incremental SVD: process columns one by one, keep top-r" — i.e. the *same* update with a
fixed-rank policy) or why they differ; `docs/week1.md:36-39` calls incremental SVD "a near-optimal streaming baseline" that
"tracks the oracle closely". The README (`README.md:39`) also claims BUG beats Frequent Directions; the paper omits FD. None of
Brand, Baker et al., Liberty/Ghashami et al. (FD), or Oja is cited. **This does not kill the contribution** (a rank-adaptive,
blocked, fp32-core, σ-floored incremental tracker with a principled truncation criterion is a fine engineering object, and the
downstream system is the paper), but the *story* must state the identity and locate the delta precisely (rank-adaptive
θ-truncation; block/Galerkin form; `min_sv_frac`; the near-oracle *measurement* rather than a theorem). Otherwise the
"DLRA" in the title reads as branding. CPU-only fix; one paragraph plus four citations.

**Concurrency (MomentKV 2606.01563, ResKV 2607.29591):** handled *politely* but not *fairly*:
- `:345-352` omits the one overlap a reviewer will care about — MomentKV scores tokens with a residual norm w.r.t. a summary
  subspace of the evicted set (per the 09-01 review's reading; I cannot verify a June-2026 paper). That is the "surprise"
  signal of `:227-229`, and the paper's own novelty sentence at `:327-329` ("select the exact tier by low-rank *surprise* ...
  rather than by attention mass. The adaptivity is the novel axis; the surprise coupling is what makes ...") is exposed to it.
  Say it explicitly: "MomentKV's selection statistic is also an out-of-subspace residual; ours is measured against a
  rank-adaptive tracked basis and the selected tokens are kept exact rather than corrected."
- "We cite them to mark the timeline" (`:352`) and "arXiv v1 timestamp" (`:164`) claim priority the paper cannot document
  (git history is not citable). For an ICML 2027 submission (deadline ~late Jan 2027) work from June/July 2026 will not be
  "concurrent" under the usual ~2–3-month norm; reviewers will expect a discussion of their *results*, and possibly a
  comparison if code exists. Reword to "independent contemporaneous work" now; budget a comparison (~$20–50) for the ICML
  version.
- Also missing from the "summarize the discarded remainder" family the paper contrasts itself with at `:349-351`: the
  merging methods CaM (Zhang et al., ICML 2024), KVMerger (Wang et al., 2024, arXiv 2407.08454), D2O — one sentence.

### 3.2 Is the 2-bit baseline a fair KIVI? (Q2)

**Scheme fidelity: yes, verified in code.** `src/kvdlra/quant/kivi_cache.py:14-24, 130-135`: keys per-channel (quanto
`axis=-1`, groups of `g` consecutive tokens per head-dim channel, `T` edge-padded to a group multiple and sliced on
dequantize), values per-token (`axis=0`, `g` consecutive channels), asymmetric with scale+zero (`:171` reads `_scale`,
`_shift`), fp16 residual `residual=128` (`:118`), residual flushed after chunked prefill so decode starts from the same state as
single-shot (`:88-98`), test-pinned (`tests/test_w19_quant_kivi.py:63,80,94,109,127`). Cross-backend agreement at 2-bit (quanto
vs hqq both 1.00 on Qwen single-needle, `results/w18_harvest/quant-findings.md:72-75`) is a partial cross-implementation
check the paper could mention alongside the 8-bit control (`main.tex:639-640`).

**Accounting: honest and slightly generous to the quantizer.** Aux billed at the measured 16-bit dtype on a bf16 model
(`kivi_cache.py:161-172`; `quant-findings.md:60-61`), asymptote (2 + 2·16/64)/16 = 0.156× (`main.tex:638-639`) — correct
arithmetic; residual billed as 128 fp16 tokens even after flush (`accounting.py:441-443`), which over-bills the arm by ~0.007
(0.163 vs 0.156) — immaterial and against BUG.

**Two places where "KIVI-faithful" (`:75, :156, :170, :633`) overstates:**
1. **Group size.** The arm uses `g=64` (`scripts/pod/w19.sh:28` + `w10_frontier.py:784` default; `results/w19-a1-report.md:3`).
   KIVI's paper/README default is, to my recollection, **G=32** with R=128 (verify against Liu et al. 2024 §5 and the
   official repo); the ablation in KIVI shows G=64/128 costs accuracy. G=64 is the transformers `QuantizedCacheConfig`
   default. Effect: the arm stores fewer bytes (0.156 vs 0.1875×) but with coarser groups, i.e. it is *slightly weaker per
   element* than paper-default KIVI. At the paper's "matched bytes" framing (0.151 vs 0.163) this is the right comparison to
   make, but it must be stated, and a G=32 row (0.19×) is a ~$10 robustness check that would make the fairness claim airtight.
   No repo document notes the G=32 default (grep for `group.*32` in docs/results: none).
2. **No validation against a published KIVI number.** The 8-bit hqq control (`:639-640`) validates the decode path, not
   2-bit fidelity. One ppl or LongBench point reproduced from the KIVI paper (Llama-2-7B or Mistral-7B, ~$5–10) would close the
   "your re-implementation is weaker than KIVI" attack, which is the obvious response to Table `tab:fairquant`'s 0.42/0.50
   multi-value cells for a method advertised as near-lossless.
   Rename to "KIVI-scheme" or "KIVI-style (per-channel-K / per-token-V, asymmetric, fp16 residual; HF/quanto grouping G=64)".

**The "earlier draft reported 0.00" disclosure (`:640-645`) is adequate and accurate.** It matches the root cause in
`quant-findings.md:32-42` (optimum-quanto `axis=0` = per-token groups for *both* K and V while the transformers docstring calls
it "per-channel"; one outlier key channel — Qwen2.5's `k_proj` bias — sets the per-token scale). One improvement: name the
trap (the library docstring mislabels the default) so readers do not repeat it; and note that the zero *also* appeared at 4-bit
(`quant-findings.md:9,71`), which strengthens "configuration hazard, not quantization".

**Post-RoPE keys.** The arm quantizes post-RoPE keys (HF cache), as KIVI does; KVQuant quantizes pre-RoPE — fine, since the
paper says KIVI-faithful, not KVQuant-faithful. But `:299-302` credits KVQuant's "pre-RoPE, non-uniform, dense-and-sparse" as
reaching 0.125–0.19× near-losslessly and then runs only the KIVI scheme; a sentence saying why KVQuant was not the arm
(calibration/non-uniform codebooks, no HF cache path) would pre-empt "you picked the weaker 2-bit method".

### 3.3 Related-work accuracy — spot checks (Q3)

| Citation | Paper's description | Verdict |
|---|---|---|
| H2O `:286-288` | rank tokens by (proxies for) attention mass | accurate (accumulated attention + recent) |
| SnapKV `:287` | same family | accurate (observation-window attention, pooled) |
| PyramidKV `:288-289` | varies budget by layer | accurate |
| Quest `:289, :102-105` | "selects KV pages per query"; listed under **eviction** that "discards the rest" (`:101-105`) | **inaccurate shelving**: Quest discards nothing (query-aware sparse attention over full KV, a bandwidth method); say so or move it to a "selection without memory saving" clause |
| Expected Attention `:289-291` | expected attention of future queries | accurate; bib `refs.bib:111` "and others" — list all authors (it is a 3–4-author paper) |
| KIVI `:299-300` | per-channel keys, per-token values, 2-bit; 0.125–0.19× | accurate |
| KVQuant `:300-301` | pre-RoPE, non-uniform, dense-and-sparse | accurate |
| CacheGen `:302` | targets network/serving path | accurate, but it is a *persistence/transfer codec* and the paper's own persistence claim (`:878-900`) never positions against it |
| Palu `:307-308` | decomposes KV projections, caches low-rank latents | accurate — **but** `:598-599` "floored at 0.50–0.75× by construction" is false: Palu supports higher rank reduction and composes with 3-bit latent quantization, reporting ~91% (11.4×, ≈0.088×) KV reduction in its own paper (verify exact figure) |
| ThinK `:312` | prunes key channels | accurate — **but** ThinK evaluates 40–60% pruning and composes with KIVI-2/4-bit in its own paper, so "floored at 0.75× by construction" is false |
| Eigen Attention `:308-309` | attention in a fixed low-rank space | accurate; bib `refs.bib:183-188` is arXiv-only — it is EMNLP 2024 Findings (verify) |
| xKV `:309-310, :117-118` | shares singular vectors across layers, one-shot at prefill | accurate to my knowledge; bib venue "ICML 2026" (`refs.bib:193`) — verify |
| MatryoshkaKV `:310-311` | trains nested orthogonal projections | accurate |
| LoRC `:311-312` | progressive weight-matrix low-rank | accurate |
| MLA `:313-314` | train-time latent KV | accurate |
| LESS `:320-323` | learned constant-size low-rank state (tiny MLPs) + eviction policy | accurate |
| LoLA `:323-325` | linear attention + sparse global cache | accurate |
| Ceruti–Lubich / Ceruti–Kusch–Lubich `:124-127, :188-195` | BUG, rank-adaptive augmented step, σ_min-independent bound | accurate as description of the integrators; the *applicability* of the bound to a column-append stream is the issue in §3.1 |
| Koch–Lubich 2007, DLRT 2022, Schotthöfer 2025 `:334-341` | lineage | accurate |
| TurboQuant `:275` "we test TurboQuant" | — | **overstated**: the composition uses only the rotated Lloyd–Max ("PolarQuant") stage at integer bits, no QJL residual (`src/kvdlra/quant/polar.py:1-24`); and `:577` uses the name "PolarQuant" with no citation — PolarQuant is also a separate paper (Han et al., arXiv 2502.02617) |

Internal contradictions a reviewer will catch on first read (related-work accuracy about *this* paper):
- `:278-281` "treat a head-to-head against dedicated 2-bit KV quantizers (KIVI, KVQuant) as the primary v2 experiment" and
  `:302-305` "their absence as a baseline is v1's main scoping decision ... composition, not competition" — **stale**, both
  contradicted by §6.7 (`:630-681`) and the abstract. Header comments `:5-8, :20` are stale too (non-rendering).
- `:585` dangling word "Three" (a half-deleted sentence; renders in the PDF).
- `:52-53` (abstract) vs `:117-118` (intro): the abstract says all low-rank methods fix the subspace at calibration; the
  intro correctly says xKV is per-sequence prefill.

### 3.4 Missing citations a reviewer would demand (Q4)

Ranked. "Must" = a reviewer will raise it; "should" = strengthens; "nice" = optional.

| # | Work | Why | Where in paper | Priority |
|---|---|---|---|---|
| 1 | **OjaKV** (arXiv 2509.21623) | online Oja-rule low-rank KV with hybrid store; occupies the claimed axis; implemented in repo | abstract `:52-53`, intro `:115-119, :169`, related `:314-317`, `:383` | **must (fatal as worded)** |
| 2 | **Brand 2002/2006** (incremental SVD), **Baker–Gallivan–Van Dooren 2012**, **Frequent Directions** (Liberty 2013; Ghashami–Liberty–Phillips–Woodruff 2016), **Oja 1982/1992** | the baselines at `:380-383` are uncited; FD compared in README but absent from paper; the BUG step's algebraic identity with Brand's update (§3.1) | `:184-204, :378-386` | **must** |
| 3 | **ShadowKV** (Sun et al., arXiv 2410.21465) | per-sequence pre-RoPE low-rank keys + CPU-offloaded values; the pre-RoPE-is-low-rank observation the paper presents at `:206-213` crediting only KVQuant — the repo's own docs credit ShadowKV §3.1 (`docs/week1.md:40`, `docs/PLAN.md:23,714`); named at `:517` with no citation | `:206-213, :307-317, :517` | **must** |
| 4 | **GEAR** (Kang et al. 2024, arXiv 2403.05527) and Palu's own low-rank×quant result | prior "low-rank + quantization compose" for KV; `:272-281` and `:572-580` read as if composition were new | `:270-281, :572-608` | **must** |
| 5 | **PolarQuant** (Han et al., arXiv 2502.02617) | name used at `:577, :592` uncited; disambiguate from TurboQuant §2 | `:275, :577` | must (cheap) |
| 6 | **CaM** (ICML 2024), **KVMerger** (2407.08454), **D2O** | the "summarize the discarded remainder" family the paper contrasts itself with | `:345-352` | should |
| 7 | **AttentionStore/CachedAttention** (Gao et al., USENIX ATC 2024), **CacheBlend** (EuroSys 2025), **Prompt Cache** (MLSys 2024), CacheGen as codec | the persisted-cache cold-start claim (`:878-900`, Fig. coldstart) is positioned only vs full KV and vs 2-bit; the persistence literature owns this scenario | `:878-900, :919-924` | should |
| 8 | **Deshpande–Vempala 2006** (adaptive sampling ∝ squared residual), **online leverage-score sampling** (Cohen–Musco–Pachocki 2016) | "surprise = out-of-subspace residual" has a 20-year NLA lineage; citing it pre-empts "MomentKV does this too" and makes the tier principled | `:227-229, :319-332` | should |
| 9 | **CommVQ** (ICML 2025, 1-bit codebook KV), **SKVQ** (2-bit K / 1.5-bit V), **CacheGen** entropy coding, **MiniKV** (eviction×2-bit) | bound the "below the quantizer floor" claim against sub-2-bit / composite methods (§3.5) | `:572-608, :843-845` | should |
| 10 | Ceruti–Kusch–Lubich 2023 parallel rank-adaptive BUG (arXiv 2304.05660); Kieri–Lubich–Walach 2016 (the robust bound) | DLRA context for the "σ_min-independent" claim; the 09-01 review asked for the former | `:124-130, :334-343` | nice |
| 11 | Loki (NeurIPS 2024, PCA keys), SubGen (2402.06082, streaming sublinear KV via online clustering), Compressive Transformer / Infini-attention (trained "compressed far past + exact recent") | neighbourhood; a theory reviewer will ask about SubGen's streaming guarantees | related work | nice |
| 12 | KQ-SVD, EliteKV, KV-CoRE, ReCalKV (2025 low-rank KV) | completeness of the low-rank paragraph | `:307-317` | nice |

Bib hygiene: `refs.bib:111` "and others" for a short author list; `refs.bib:52,178,193` put venues in `journal=` fields
(fine for arXiv but will misrender under ICML style); Eigen Attention venue missing; xKV venue to verify.

### 3.5 Is "below the quantizer floor" fair? (Q5)

The paper makes the claim in four strengths; only two are defensible:
- `:574-575` "A b-bit quantizer cannot store fewer than b/16 of full KV plus its scale overhead" and `:829-836, :843-845`
  "any fixed-bit method ... holds a constant memory ratio ... BUG's ratio falls as 1/T" — **fair and true** (a scalar
  quantizer at fixed b is Θ(T) with constant ratio; the gist is O(rn + hn)). The 64K point (`:837-843`) is the right evidence.
- abstract `:84` "below any 2-bit floor" — **true, narrowly**.
- `:160-161` "where no scalar quantizer can", `:584` "a compression depth scalar quantization cannot reach", `:679` "What
  quantization cannot do is reach below its own floor" — **tautological or misleading**: (i) 1-bit scalar schemes exist (QJL
  keys; SKVQ 1.5-bit values; KVQuant 2-bit) and a 1-bit/g64 arm with 16-bit aux would sit at ≈0.09×; (ii) codebook/vector
  quantizers (CommVQ 1-bit ≈0.07× from my knowledge; verify) and entropy-coded caches (CacheGen) are "quantization" in any
  reviewer's taxonomy and go below 0.156×; (iii) **composites reach the band without BUG**: eviction × 2-bit (ea-k0.3 × KIVI-2
  ≈ 0.05×; MiniKV is this pattern), Palu-r0.5 × 3-bit (≈0.09× in Palu's own paper), ThinK × KIVI-2 (≈0.12×). The paper's own
  1B study (`:806-811`) says EA×TurboQuant and SnapKV×TurboQuant sit on or ahead of BUG's frontier through 0.08–0.18× and BUG
  wins only <0.07× — so at 7B/8B the "honestly exclusive band" (`:584, :1007`) is asserted, not measured: no composite
  competitor was run at 0.03–0.05×. The nearest published composites are ~1.5–2× above the 0.048×/0.034× cells, not
  "unreachable".
- `:598-599` "Channel-pruning (ThinK) and low-rank factorization (Palu) are floored at 0.50–0.75× by construction" — **false**
  (§3.3); it is a floor of the operating points *run here* (`think-c0.5`, `palu-r0.5`).

Fair statement: "below the floor of any fixed-bit scalar quantizer applied to every token (0.156× at 2 bits; ≈0.09× at 1
bit), and, at 16–32K, 1.5–2× below the lowest published composite low-rank×quant / eviction×quant operating points we know of
[Palu+quant, MiniKV, CommVQ]; whether those composites retrieve at 0.05× is not measured here." Plus the 1/T argument, which is
the actually exclusive property.

## 4. FATAL / fixable / nitpick

**FATAL (as worded; fixable CPU-only in hours)**
- F1. OjaKV uncited while the paper claims the online axis is unoccupied and reports beating "Oja's rule" (§1, §3.1).
  `main.tex:52-53, 115-119, 169, 314-317, 383`; `src/kvdlra/integrators/oja.py:1-12`; `docs/week5-plan.md:57`.

**Fixable (material to the novelty/positioning score)**
- X1. State the algebraic identity between the augmented-BUG column-append step and Brand's incremental SVD / FD, define the
  incremental-SVD baseline of `:382`, cite Brand, Baker et al., Liberty/Ghashami et al., Oja; scope the σ_min-robustness
  claim (`:126-130`) to what is actually shown (a measurement, not a theorem for this setting). CPU, one paragraph.
- X2. Delete/rewrite the stale "quant baseline is v2" sentences `:278-281, :302-305` (and header `:5-8, :20`); they contradict
  §6.7. CPU, minutes.
- X3. Fix `:598-599` ("by construction") and `:160-161, :584, :679` per §3.5; cite GEAR, Palu-composed, CommVQ/SKVQ/MiniKV.
  CPU, hour. Optional ~$10–20 GPU: one composite competitor at matched ≈0.05× bytes (palu-r0.5×KIVI-2 or ea-k0.3×KIVI-2) on
  Llama+Mistral 16K/32K, n=12 — turns "honestly exclusive" from assertion into measurement.
- X4. Cite ShadowKV at `:206-213` and `:517`; move it into the low-rank paragraph (`:307-317`) as the per-sequence pre-RoPE
  low-rank-key precedent. CPU, minutes.
- X5. KIVI arm: rename "KIVI-faithful" → "KIVI-scheme"; state G=64 (HF default) vs KIVI's G=32; ~$10 GPU for a G=32 row and
  ~$5–10 for one published-number reproduction (§3.2).
- X6. Concurrency paragraph `:345-352`: acknowledge MomentKV's residual-norm score as the same signal as surprise; replace
  "mark the timeline" with "independent contemporaneous work"; plan a comparison for the ICML version (~$20–50 if code exists).
- X7. Cite PolarQuant at `:577` and say the composition uses TurboQuant's Lloyd–Max stage without QJL (`:275`); cite the
  merging family at `:349-351`; cite the persistence line (AttentionStore/CacheBlend/Prompt Cache, CacheGen as codec) at
  `:878-900`; cite Deshpande–Vempala / online leverage sampling at `:227-229`. CPU, hour.

**Nitpick**
- N1. `:585` dangling "Three".
- N2. Quest mis-shelved under eviction (`:101-105, :289`).
- N3. `refs.bib:111` "and others"; Eigen Attention venue; xKV "ICML 2026" verify; venue-in-`journal` fields `refs.bib:52,178,193`.
- N4. "gist" collides with Gist Tokens (Mu et al., NeurIPS 2023) — a one-line disambiguation or a different word.
- N5. `:639-640` could add the 2-bit quanto-vs-hqq agreement (`quant-findings.md:72-75`) as a cross-implementation check.
- N6. `:380` near-oracle evidence is 1B/layer-8/5-docs (`docs/week2-pilot.md:55-57`); say "on 1B" in the abstract claim `:56-57`.

## 5. What the closing of the previous fatal set revealed (for the AC)

Closing the quant gap was done properly and it *narrowed* the paper: the abstract now spends five sentences (`:75-93`)
conceding what 2-bit/4-bit quantization takes away. That is honest and reviewers will credit it. But the paper still needs a
one-sentence, defensible novelty statement, and today it does not have one that survives OjaKV + the incremental-SVD identity.
The candidate that does: **"a near-oracle rank-adaptive online subspace tracker (vs. Oja's rule 1.3–3× worse, vs. fixed-rank
incremental SVD [state delta]) paired with a residual-selected exact tier, whose stored state is O(rn+hn) and therefore falls as
1/T — the one property no fixed-bit method has — with a paired fair-quant comparison that shows exactly where that buys
retrieval (multi-value, sub-0.16×) and where it does not (official RULER, fluency)."** Everything for that sentence is already
in the repo.

## 6. Highest-leverage fixes with cost

- **A (CPU, ~1 day):** the citation/reframing pass — F1, X1, X2, X3-wording, X4, X6, X7, N1–N3. This alone moves the
  dimension 5 → 6.5.
- **B (~$15 GPU, Llama 16K, n=12, 4 tasks + ppl):** *tracker-swap ablation* — same cache (sinks, ring, seed, h256 tier), swap
  the gist tracker among BUG / Oja (`oja.py`) / FD (`frequent_directions.py`) / fixed-rank incremental SVD. The only current
  evidence that the DLRA integrator matters is Week-2 reconstruction error on one layer of a 1B model; an end-to-end delta (or
  its absence) is what makes "DLRA" in the title load-bearing rather than branding. `docs/week7-dominance.md` already framed
  this ablation. Moves 6.5 → 7.
- **C (~$20–30 GPU):** fairness closure — KIVI G=32 row (X5) and one composite competitor at ≈0.05× (X3), both on
  Llama+Mistral. Removes the two remaining "your baseline is weak / your band is not exclusive" attacks.

## 7. Score rationale

The mechanism (rank-adaptive BUG tracker + residual-selected tier + seed + `min_sv_frac`) and the mechanistic map are
genuinely new and, after this program, unusually well-bounded. The positioning, however, currently (a) asserts an unoccupied
axis that the authors' own baseline code shows is occupied, (b) presents a known incremental-SVD update as a DLRA tool "built
for exactly this problem" without stating the identity, and (c) has swapped an over-claim about eviction for over-claims about
Palu/ThinK/quantization floors while leaving stale "quant is v2" sentences in the text. Each is a wording/citation fix, none
needs new science, but together they are what a Reviewer 2 uses to write "the novelty claim is not accurately stated" —
which at ICML is a 4–5. With fix A the honest novelty sentence in §5 stands and the dimension is a 6–7; with B it is a firm 7.
