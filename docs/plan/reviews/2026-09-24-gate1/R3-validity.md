# R3 — Experimental validity — Gate 1 tracker swap (blind)

## 1. Numbers I read
- Table header realised sizes: primary retrieval **m=12**, secondary retrieval **m=18**, perplexity **m=4** → one task left the families. Amendment 1b (A1b.4) names it **vt** (pre-flight `full`/vt 9/12=0.75<0.9); Stage-1 corroborates the low ceiling — `llama full / vt` **0.71 (17/24)**, `qwen full / vt` **0.67 (16/24)**. vt is descriptive; blocks nothing.
- Degraded operative ceiling: `qwen full / niah_multivalue` **0.50 [0.31,0.69] (12/24)** — <0.9 on a task the (Llama-only) pre-flight passed (pre-flight `full`/multivalue 0.92). §6(ii) territory.
- The only two primary marks, both **‡** (favour the control): `llama nogist / niah_multikey` **0.92 (22/24) ‡** vs `llama isvd / niah_multikey` **0.42 (10/24)**; `llama nogist / niah_multivalue` **0.88 (21/24) ‡** vs `llama isvd / niah_multivalue` **0.25 (6/24)**. No `*` anywhere in either block.
- Floor-against-floor: `llama isvd / niah_multivalue` **0.25 (6/24)** = `llama frozen / niah_multivalue` **(6/24)** = `llama fd / niah_multivalue` **(6/24)**, all unmarked.
- Unresolved marginal gaps at m=12: `llama isvd / niah_single` **0.79 (19/24)** vs `frozen` **0.54 (13/24)** and vs `nogist` **1.00 (24/24)** — both unmarked (a 5–6 pair marginal gap is under the first Holm slot).
- Floor control alive: `random` **0/24** on every task both families; `random` ppl delta `llama` **-6.6172** (10.3215), `qwen` **-7.2261** (11.0329); `full` ppl advantage `llama isvd` **+0.1452**. The axis has dynamic range.
- members: `llama isvd-vs-fd` TOST fail **d=-0.0741 p=1**; `qwen isvd-vs-frozen` TOST fail **d=-0.0137 p=0.09**; `qwen/niah_multikey isvd-vs-fd separates Holm p=0.027` (`qwen isvd / niah_multikey` **0.79 (19/24)** vs `fd` **0.38 (9/24)**); `qwen isvd-vs-fd` TOST fail **d=-0.0939 p=1**. Perplexity nogist: `llama nogist` **+0.0533 Holm 9.54e-7**, `qwen nogist` **+0.0190 Holm 0.0435** (delta = isvd − tracker; +ve = isvd worse).
- bf16 block printed PASS/PASS; `sbits bf16/isvd` 0.5667 / 0.5382 — rule not in the two files.

## 2. Rule compliance as I read it
- §4 rule 2 (A/B needs ≥2 families): a family separates only if isvd beats **both** frozen and nogist. isvd beats nogist on **no** axis in either family — retrieval nogist ties/‡ ahead of isvd, perplexity nogist Holm-sig with **isvd worse** (+0.0533 / +0.0190). So **0 families separated** → not A/B. Agree.
- §4 rule 3 C(i): blocked by `qwen/niah_multikey isvd-vs-fd` Holm p=0.027. C(ii): blocked by three failed TOSTs (llama fd, qwen frozen, qwen fd). C blocked. Agree.
- Precedence → **UNDECIDED**, not REFUSED (no error/FAILED/not-run cell, no refusal text). Agree with the VERDICT line — but "not REFUSED" I can only *infer* from the missing refusal, not verify.
- vt carries no marks and sits at m=12/18 — descriptive, consistent with A1b.4. Agree.

## 3. Three strongest objections
1. **The letter understates a clean negative.** Every C-blocker is tracker-vs-tracker — isvd beating a deliberately-hobbled ℓ=r FD arm (qwen multikey; both fd TOSTs) or edging frozen. The *mechanism* control says the opposite: byte-matched **nogist ties or beats isvd on both axes in both families** (‡ on llama retrieval; +0.0533 / +0.0190 perplexity, both Holm-sig). The rule's asymmetry (C ignores nogist) produces UNDECIDED where the evidence points to "the tier, not the tracker." Correct under the rule, but reads more neutral than the data is.
2. **The §6(ii) degraded-ceiling flag is not rendered.** `qwen full / niah_multivalue` = **0.50** makes its isvd-vs-frozen and isvd-vs-nogist members uninformative, yet they sit inside the m=12/18 families (multivalue is not excluded — only vt is). §6(ii) requires each such member be *flagged* as resting on a degraded ceiling; the table prints the 0.50 cell but no flag, and the VERDICT line does not name it. A reader must catch it unaided.
3. **The load-bearing pins are absent from the table.** The ‡ "nogist beats isvd" reading is a *mechanism* claim only if nogist held ≈1× isvd's stored bits; §4 prints that ratio **only on refusal**, so the table never shows it. No "96 keys; 0 disagree" pairing line and no per-member realised n_paired (retrieval McNemar keys, or the 32 ppl windows). Byte-match, pairing, and frozen-dispatch are taken on faith from the missing refusal.

## 4. What I would need before believing the branch
- The measured `median(sbits nogist)/median(sbits isvd)` (within 1±0.05) and the 96-key pairing result, **printed**, not inferred from silence.
- The §6(ii) flag on every qwen-multivalue member, the 0.50 ceiling beside it, in the verdict entry.
- Per-member n_paired (McNemar keys; the 32 perplexity windows) confirming none ran short, plus `pod.py check` = OK / zero error rows.
- The frozen-dispatch (no repair past freeze) and random-flat `diag` checks (§7 c) confirmed clean.

## 5. One-line verdict
UNDECIDED is the correct letter and n=24 was adequate to answer the real question — the design resolves the ≥10-pair swings and outside-±0.02 effects that matter, and they run *against* the tracker (nogist ≥ isvd on both axes; isvd only ever beats a worse tracker) — so the sound reading is tier-over-tracker, held out of Branch C by fd, pending the byte-match, pairing, and degraded-ceiling disclosures the table omits.
