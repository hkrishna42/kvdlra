# Meta-review (Area Chair) — kvdlra NeurIPS-readiness panel

Date: 2026-09-01. Panel: 6 dimension reviews + attack/defense cross-examination.
Scores received: claims 5, prior-work 5, rigor 5, significance 5, systems 3, repro 7;
cross-exam attack 3, cross-exam defense 6.

**Overall: 5/10 as it stands (borderline reject). Decision: close-needs-targeted-work.**
Post-gap-fill trajectory: 6–7 (poster-to-accept). The program's science largely survives
cross-examination; the paper's *framing*, two missing baseline families, one external
benchmark anchor, and the manuscript itself do not yet exist in submittable form.

---

## 1. How the cross-examination resolved (AC spot-verified)

The attack's 3/10 rested on seven pillars. Four fail on checkable facts, verified by me
directly against the repo this session:

**Discounted (defense refutations verified):**
- *"The repo's only realistic-text datum shows collapse / bug-r128 loses to eviction."*
  REFUTED. `results/w11-goalA-lb-lines.txt` (verified): bug-r128 qasper F1 **0.221 @0.160×**
  beats ea-k0.25 (0.216 @0.250×), ea-k0.1 (0.149 @0.100×), snapkv-k0.1 (0.136 @0.100×);
  the *baselines'* extreme configs collapse on real text (palu-r0.25 = 0.073). Residual kept:
  flagship-on-LongBench never run, and the pre-seed r64 cell (0.099 vs full 0.259) is honest
  motivation to run it.
- *"The matched-r/n control (Llama r256) was never run."* REFUTED.
  `docs/week11-decision-table.md:18-19` (verified): bugS-r256-h256/h1024 = **0 on all hard
  tasks**. Llama r256/1024 = Qwen r128/512 = 0.25 — the r/n≈0.25 dimensionless onset is
  *supported*, and adopting it strengthens C2. Residual: pre-seed, n=4; a seeded r256 column
  is a ~$10 nicety.
- *"'Also 1.00 at 32K' is false for Mistral."* REFUTED as a repo indictment (verified):
  `docs/week17-explained.md:30` scopes 32K-perfection to Qwen; the false generalization lives
  in the panel briefing's paraphrase. Residual kept: all 32K cells are n=4 with vacuous CIs.
- *"Ceiling-effect benchmark with no power; effective n≈5."* REFUTED as stated: the same
  instrument recorded collapse cells for every method pushed into the extreme band, including
  BUG's own (r256=0, pre-fix 0% @32K, palu-r0.25 0/8, Qwen r128=0), and interior cells
  (7/12, 15/16, 9/16) show real trial variance; mk/mv/vt span depths ~11–89%. What survives
  is the external-validity form (below) and the correct statistical framing: the 16K matrix
  is a **non-inferiority** result (12/12 ⇒ acc ≥0.76 at 95%), with zero Wilson-separated
  flagship-vs-baseline cells in either direction — the paper must say "no detected loss,"
  not "matched," until the discriminative reruns land.

**Standing (conceded by defense; these gate the decision):**
1. **No quantization/codec baseline in 17 weeks** (KIVI 2-bit ~0.125–0.19×, KVQuant, CacheGen)
   while the headline band is 0.075–0.149× — the one family that *can* contest the band is
   absent. Mitigations on record (main.tex concedes 4-bit near-lossless at 0.25×; measured
   BUG×TurboQuant 0.099×; an in-repo sub-cliff cell bugS-r32 @**0.043×** = 100/67/100/100 @32K).
2. **Storage-vs-resident memory**: decode workspace ≈0.98× full, resident ~1.06×, persistent
   mid-cache verified; the "5–13× less memory" sentence reads as memory but is storage
   accounting; caveat absent from both paper-facing narratives; only latency datum unfavorable.
3. **"A regime eviction/channel-pruning cannot enter" is false as worded** (verified:
   `docs/week16-explained.md:69-70` vs ea-k0.1 @0.100× = 100/92/100/17 in the repo's own
   Week-11 table; bugEVICT-h256 answers the needle at 0.018×). The honest reword — pruning
   floored at 0.5–0.75×; eviction reaches 0.1× but loses vt (17) there and its mk decays with
   length (92→67→50) — retains most of the frontier and is $0.
4. **External validity**: all retrieval evidence from a custom in-repo generator whose filler
   is 10 fixed sentences cycled ~130×; no official RULER/LongBench number on any flagship
   config; selection and confirmation share the generator.
5. **fp32-at-rest state billed at fp16** in `ratio_fp16` (~0.15×/0.27× honest vs 0.085×/0.149×
   billed); the fp16-storable variant never run; Week-17 itself proved the integrator's
   numerics are delicate.
6. **Evidentiary chain**: per-trial records, raw JSONs, GPU/torch/CUDA identity and run SHA
   for the w13–17 headline pods are unrecorded (pods destroyed; clone-by-branch); recovery
   via round(acc·n) is exact but the compute-disclosure checklist is currently unanswerable.
7. **There is no manuscript**: paper/main.tex = 157 lines, 3 bibitems, Week-4 content; no
   related work (LESS/LoLA/KIVI/DLRT/CKL-θ uncited; the ExpectedAttention/Eigen-Attention
   name collision unaddressed); MomentKV/ResKV concurrency clock is running.
8. **Wording debts** ($0, mechanical): 5–13× → 3.4–10× vs ThinK/Palu (6.7–13× vs full KV);
   "beats both on mv" → "leads" (Fisher p≈0.48); name the comparator (think-c0.5); state C4's
   cross-week ppl/retrieval provenance; demote C3 to a stability subsection citing CKL's θ
   (min_sv_frac = a self-scaling *relative* variant of the published truncation — real but
   modest delta; headline the rank-siphoning diagnosis instead); keep "extends the safe rank"
   within-method (floored cells are Palu-dominated at matched memory).

**Attack points I rejected as overstated** (defense arguments adopted): think-c0.3 does not
threaten the marquee (it costs 0.852× — a near-full-memory config scoring vt=100 no more
contests a 0.16× frontier point than full KV does); the multikey drop is not selective
reporting (mk is BUG's historically *best* task; the documented Week-14 sizing miss is the
proximate cause — but the Qwen/Mistral mk gap must still be filled); pre-registration was not
"quietly violated" (every cited ppl tie is Week-15 n=8; the n=4 cells back only 3–4
order-of-magnitude floor effects, and the halving is in a titled commit); "no single config
delivers both headlines" is false on Llama (the marquee delivers 100 on all four tasks at
both contexts + pre-registered ppl ties at 0.16–0.19×); and C1 is one uniform config string
across all three families, not a per-model patchwork.

## 2. Verdict logic

The surviving fatal set is: (a) two missing baseline families (quantization; eviction in the
headline tables), (b) one framing error repeated in paper-facing text (storage billed as
memory; "cannot enter"), (c) an unanchored benchmark, (d) no manuscript. None of these
contradicts a measured in-repo result; all are enumerated, priced (~$120–170 GPU + ~2–3
weeks work), and two already have their honest framing written elsewhere in the repo. That
is the definition of *close-needs-targeted-work*: a 5 today with a mechanical path to 6–7,
and a program whose self-scrutiny (Q-BUG kill, Palu-separation downgrade, baseline bugs
fixed in the baselines' favor) suggests the fills will be executed honestly.

## 3. Ranked gap-fill plan

| # | Gap | Why reviewers care | Exact fill | Cost | Blocking |
|---|-----|--------------------|-----------|------|----------|
| 1 | No quantization baseline; band contested by 2-bit KV | KIVI/KVQuant occupy 0.125–0.19× near-lossless; one reviewer sentence kills C1 as worded | Run KIVI-2bit, 3 families, 16K/32K, same ppl+RULER harness; add fp16-stored-state probe (cast U/C between steps, rerun r64 RULER+ppl) to settle billing; reframe as compose-not-compete and re-anchor the exclusive band below the scalar-quant cliff using the measured bugS-r32 0.043× cell (+ optional BUG×4-bit retrieval at ~0.04×) | ~$60 GPU + hours CPU | YES |
| 2 | Custom cyclic-filler generator; no official-benchmark anchor | External validity of every retrieval claim; "self-authored benchmark" is a standing reject reason | Swap filler to wikitext/PG19 sentences (one-line) + depth sweep; rerun flagship+ThinK/Palu+EA at 16K n=12 on Llama+Qwen; official RULER subset on the flagship, one model | ~$15 + ~$50 GPU, days | YES |
| 3 | Storage-vs-resident memory framing | Systems reviewer finds workspace ≈0.98× and the rebuttal collapses; only latency datum is 10% slower | $0 rewrite to "stored/persisted cache state" everywhere; promote the w16_storage.py caveat into main text; re-run w16_storage on CUDA and commit JSON+figure; one measured persistence/cold-load number (serialize @32K: bytes, reload, H2D vs full-KV load — full KV 4.29GB vs ~0.7GB is a real ~3–6× measured win) | $0 + ~$10 GPU | YES |
| 4 | Eviction absent from headline tables; "cannot enter" false as worded | Refuted by the repo's own Week-11 table; EA reaches 0.1× | $0 reword to measured collapse points + EA's vt=17/mk-decay profile; run ea-k0.1 + snapkv-k0.1, n=12, all 3 families, **multikey included**, vs the seeded flagship | $0 + ~$25 GPU | YES (reword); grid supports it |
| 5 | Manuscript + related work | Every panel score is for a hypothetical document; MomentKV/ResKV clock running | Full draft; ~20–25-citation related work (LESS/LoLA differentiation, KIVI/KVQuant/CacheGen, DLRT, CKL θ, H2O/SnapKV/PyramidKV, MLA, EA name collision, MomentKV/ResKV as concurrent); non-inferiority phrasing for the 16K matrix; arXiv preprint ASAP for the timestamp | 2–3 weeks writing, $0 | YES |
| 6 | $0 wording pass | Each slip (5–13×, "beats", unnamed comparator, unscoped floor claim) is a credibility hit that invites the audit that finds the rest | Mechanical pass over week16/17 explained+handover before any paper text: 3.4–10×; "leads"; think-c0.5 named; C4 provenance stated; C3 → stability subsection | $0, hours | YES |
| 7 | Evidentiary chain (per-trial, env, SHA) | Camera-ready compute disclosure currently unanswerable; artifact track fails | SHA-pinned clone + nvidia-smi/version/SHA header inside harvested blocks + per-trial hit emission in w10_ruler.py; regenerate headline cells' per-trial data by riding along with runs 1–4 | hours CPU + $0 marginal GPU | YES for camera-ready; do now free |
| 8 | Statistical firming | n=4 32K cells vacuous; mk unmeasured on Qwen/Mistral; marquee single/mk n=16 unfilled | 32K to n≥12 (flagship+baselines, 3 families); mk block Qwen/Mistral; marquee single/mk n=16; seeded Llama-r256 column (r/n law) | ~$30–40 GPU, rides along | no (but cheap and pod-shareable) |
| 9 | Hygiene + licensing | Desk-reject-adjacent details | Llama license/model cards (drop or justify unsloth mirror); CI trigger for week7; commit-or-delete w15b-complete-lines.txt + stale figures; README test count | ~1 day CPU | no |

GPU total ≈ $130–185 (pod-shareable down to ~$120). Current credit $70.4 → needs a ~$50–100
top-up. Items 3+6+7 and the reword half of 4 are $0 and should happen this week regardless.

## 4. What survives as-is

- **Novelty of the core mechanism**: training-free, per-sequence, online rank-adaptive DLRA
  tracking of the KV stream + surprise-selected exact tier. Both the prior-work reviewer and
  the attacking examiner agree no prior work occupies this. Unclaimed axis, real idea.
- **The mechanistic decomposition** (gist=fluency, tier=retrieval) and the **tier–gist
  rank-siphoning diagnosis** — publishable observations about coupling selection to a
  low-rank tracker.
- **The rank-vs-retrieval wall**, now with the r/n≈0.25 constant-onset framing the Week-11
  r256 control supports.
- **The Week-11 realistic-text ordering at moderate-extreme compression**: bug-r128
  0.221 @0.16× > every sub-0.25× eviction arm; baseline extreme configs collapse on real text.
- **The marquee vt separation vs think-c0.5** (p≈0.0006, n=16) — the program's one genuinely
  separated retrieval result; keep, with comparator named.
- **The warmup-seed fix** (16K 100/50/0/0 → 100/100/100/88, n=8) and the **floor rescues**
  (27531→6.995 etc. — effect sizes for which n=4 suffices).
- **The sub-cliff cell** bugS-r32 @0.043× = 100/67/100/100 @32K — the seed of the honestly
  exclusive band below scalar quantization.
- **The honest-limits record and infrastructure**: pre-registration, Wilson machinery,
  parity-ladder tests, strict-mypy CI, self-inflicted claim kills. The process is
  publishable-grade; the camera-ready will survive an adversarial audit once the chain
  (item 7) is repaired.

## 5. Venue advice (as of 2026-09-01)

NeurIPS 2026 has passed. **ICLR 2027 (~late Sept, ≈3–4 weeks out) is too tight**: the
blocking set includes a manuscript that does not exist plus ~$130–185 of GPU runs; shipping
in 3 weeks means shipping the current wording, which draws exactly the attack review this
panel already wrote. Recommended sequence:

1. **This week ($0)**: wording pass (items 3/4/6), evidentiary-chain process fixes (7),
   hygiene (9).
2. **By ~Sept 20**: an **arXiv preprint** of the honest current story (frontier scoped to
   measured collapse points, storage framing, non-inferiority language) — this timestamps
   the mechanism against MomentKV/ResKV, which is the real cost of waiting, and costs no
   venue eligibility.
3. **September GPU program**: items 1, 2, 4, 8 on shared pods.
4. **Primary target: ICML 2027 (~Jan)** — four months is right-sized for the runs + a real
   related-work section + a draft that leads with the sub-cliff/compose-with-quantization
   framing. Escape hatch: if by mid-September the KIVI and realistic-filler results land
   favorably and full-time writing is available, an ICLR 2027 submission becomes defensible;
   do not force it otherwise. NeurIPS 2027 (May) is the fallback, but eight more months of
   concurrency risk makes it strictly worse than ICML given the preprint timestamp.

— AC
