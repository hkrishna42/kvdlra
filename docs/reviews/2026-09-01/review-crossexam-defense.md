# Cross-examination — defense of the record against the panel's rejection case

Role: adversarial examiner, defense side. Task: test every fatal/major rejection argument against
the actual repo (`/Users/hari/Desktop/kv-dlra` @ `week7`). Each finding is classified
**REFUTED** (the panel's factual premise is wrong), **OVERSTATED** (real defect, wrong severity or
wrong inference drawn from it), or **STANDS** (conceded, with the honest mitigation on record).
All citations file:line, verified this session.

**Verdict in one line:** four of the panel's rejection pillars rest on checkable factual errors —
the "realistic-text collapse" datum actually shows BUG *beating* every sub-0.25× eviction arm in the
same file, the "missing matched-r/n control" was run in Week-11 (Llama r256 = 0 everywhere), the
"false 32K claim" lives in the panel's own briefing paraphrase and not in the repo docs, and the
"no-power ceiling benchmark" recorded collapse cells for every method (BUG included) pushed into the
extreme band — while the genuinely fatal items (quantization baselines, storage-vs-resident
disclosure, custom-generator external validity) are conceded, disclosed-in-repo, and priced at
~$85–150 GPU + $0 rewording. Post-defense calibration: 6 (borderline accept/poster after the $0
rewrites; accept-trajectory with the priced runs), not the attack's 3.

---

## 1. REFUTED — "the repo's only realistic-text datum shows the story collapsing" (rigor F1/M1, attack A3.3)

The rigor review and attack A3 cite `results/w11-goalA-lb-lines.txt` (qasper: bug-r32 F1 0.076,
bug-r64 0.099 vs full 0.259) as the in-repo proof that extreme compression fails on real text, and
the rigor review adds: *"even bug-r128 (0.16×) = 0.221 loses to eviction at matched-or-less memory
(ea-k0.25 = 0.216 at 0.25×)."*

**That sentence is arithmetically backwards.** The full committed file reads:

| arm | F1 | memory |
|---|---|---|
| bug-r128 | **0.221** | **0.160×** |
| ea-k0.25 | 0.216 | 0.250× |
| ea-k0.1 | 0.149 | 0.100× |
| snapkv-k0.1 | 0.136 | 0.100× |
| snapkv-k0.25 | 0.223 | 0.250× |
| palu-r0.25 | 0.073 | 0.257× |
| think-c0.7 | 0.138 | 0.649× |
| full | 0.259 | 1.000× |

bug-r128 has *higher* F1 than ea-k0.25 at *36% less* memory, and beats both 0.100× eviction arms by
+0.07–0.09 F1. On the panel's own chosen realistic-text datum, the sub-0.25× frontier ordering is
BUG > eviction, and the *baselines'* extreme configs collapse on real text exactly as the synthetic
tables predict (palu-r0.25 = 0.073, think-c0.7 = 0.138). Additionally: (a) these are **pre-seed
plain-`bug`/`bugS` configs** at qasper's ~5.6K context — inside the Week-11-diagnosed warm-up window
(`docs/week11-decision-table.md:97-112`) that the warmup-seed fix (Weeks 13–14) was built to remove,
so the r32/r64 cells cannot indict the flagship, whose defining feature is that fix; (b) the honest
residual — flagship-on-LongBench never run — stands (see §12), but "the only realistic datum shows
collapse" is a misreading of a file that mostly shows the opposite.

## 2. REFUTED — "the matched-r/n control (Llama r256) was never run" (attack A6.2)

A6 claims the rank-wall's "model-dependent onset" is an uncontrolled r/n confound because "the
matched-r/n retrieval cell (Llama r256) was never run." It was run, in Week-11:
`docs/week11-decision-table.md:18-19` (16K) and `:45-46` (32K) — `bugS-r256-h256` and
`bugS-r256-h1024` score **0 on all hard tasks** at both contexts (n=4/cell), with the follow-up note
(`:122-128`): "All 12 r256 hard-task cells … 0.00 accuracy AND 0.00 recall." Llama r256/1024 = 0.25
= Qwen r128/512: the in-repo data are *consistent with* a constant dimensionless onset r/n ≈ 0.25,
and the Week-16 handover already places Llama's onset at r256 (`docs/week16-handover.md:38`).
Caveats owed: the Week-11 cells are pre-seed `bugS` at n=4, so a clean bugSseed-r256 column is worth
~$10 — but A6's premise ("never controlled") is false, and the correct move is to *adopt* the r/n
framing as the unifying law (it strengthens C2, not weakens it). A6.3's "the fix doesn't transfer"
remains fair (s32 rescues Llama only) and the paper already scopes it that way
(`docs/week17-explained.md` §; `results/w16-ruler-intervals.md:18,40`).

## 3. REFUTED — "ceiling-effect benchmark with no discriminative power; n=12 measures determinism" (attack A1.1/A1.3)

A1's central inference — *"a benchmark on which 0.085× and 0.75× are indistinguishable certifies
insensitivity"* — is contradicted by the same tables' collapse cells. The identical instrument, same
tasks, same n, returns: palu-r0.25 **0/8 on all four tasks** (`results/w15-ruler-intervals.md`, Week-15
audit), think-c0.7 collapse, Qwen bug-r128 **0** (`results/w16-ruler-intervals.md:14-15`), Llama
bugS-r256 **0 everywhere** (§2), snapkv-k0.1 mk/mv/vt failures (`week11-decision-table.md:28,55`),
and pre-fix BUG **0% at 32K where EA scored 100%** (Week-10 — recorded *against* the method).
An instrument that returns zeros for every method pushed into the extreme band — including the
authors' own arms, repeatedly, across 17 weeks — is not saturated; the flagship's *non-collapse* at
0.085× is precisely the measured claim, and its correct statistical form is non-inferiority
(12/12 ⇒ true acc ≥ 0.76 at 95%), which the paper should state.

A1.3/A4's "effective n ≈ 5, the CI quantifies per-config determinism" is refuted empirically by the
interior cells: vt 7/12, think mv 10/12, palu mv 11/12, marquee 15/16, palu vt 9/16
(`results/w17-ruler-intervals.md:30,13-14,51-53,60-64`). Near-replicate deterministic trials would
produce only 0/n or n/n cells; observed within-cell variance shows the code/position/label draws do
change outcomes. What survives is the *external-validity* form of the objection (one fixed filler
document; single-needle depth jitter of ≤5 sentences, `scripts/w10_ruler.py:150`) — see §11 — and a
partial correction: multikey, multivalue and vt place items at depths spanning ~11–89% by
construction (`w10_ruler.py:160,172,186`), so "no depth variation" is true only of niah_single.

## 4. REFUTED as a repo indictment — "'Also 1.00 at 32K (n=4)' is false for Mistral" (claims P2, rigor M2)

Verified: Mistral 32K mv = vt = 0.75 (3/4) (`results/w17-ruler-intervals.md:40-41`). But the false
generalization exists **only in the panel briefing's paraphrase of C1**. The repo's paper-facing
narrative scopes it correctly — "Qwen additionally holds 1.00/1.00/1.00 at 32K"
(`docs/week17-explained.md:30`) — and in fact *under*-claims, since Llama also holds 4/4/4 at 32K
(`w17-ruler-intervals.md:62`). The committed intervals file prints the Mistral 3/4 cells with their
n. The required "$0 fix" is already done in the docs; the paper must simply copy the docs, not the
briefing. (The n=4 vacuity of all 32K cells is conceded — §12.)

## 5. OVERSTATED — "the 5–13× headline does not reproduce" (claims P1, major)

Verified real: `docs/week17-explained.md:27-28` and `docs/week17-handover.md:14` say "5–13× less
memory than ThinK (0.75×) or Palu (0.50×)", and the per-cell honest range vs those baselines is
3.4–10× (13.3 = 1/0.075 is vs *full KV*). But severity is wrong. The constituent numbers are printed
adjacent to the multiplier in the same sentence and table (0.75×, 0.50×, per-model mem column), the
Week-16 narrative got the same arithmetic right ("3–8× memory advantage",
`docs/week16-explained.md:68`), and the corrected range 3.4–10× (6.7–13× vs full) preserves the
claim's entire force. This is a two-sentence range slip with the raw data co-published — a $0
mechanical fix, minor, not evidence of an inflated evidentiary record. Same class:
"beats both baselines on mv" → "leads" (Fisher p=0.48; $0), "3–5×" → the docs' own "3–4.7×".

## 6. OVERSTATED — "'a regime eviction cannot enter' is refuted; the eviction delta shrinks to ~1.2×" (claims P3 fatal, rigor F2)

**Conceded core:** the sentence at `docs/week16-explained.md:69-70` / `week16-handover.md:42` is
wrong as universally worded — `ea-k0.1` at 0.100× scores 100/92/100/17 @16K and 100/67/100/83 @32K
(`docs/week11-decision-table.md:14,40`), and the repo's own `bugEVICT-h256` answers the single
needle at 0.018× (`:22`). The reword is mandatory and $0.

**But the panel's replacement narrative overstates the other way.** The same Week-11 table shows the
honest eviction picture the reworded claim keeps: (a) **EA's var-track = 17 [4.7,44.8] (2/12) at
16K** (`results/w15-ruler-intervals.md:12`) vs the flagship's 0.58/0.50/1.00 — the discriminative
task favors BUG against the one eviction arm that reaches the band; (b) **EA's multikey decays with
context 92→67→50** (16K→32K→64K, `week11-decision-table.md:14,40,66` — the 64K note says outright
"EA degrades as the context it must score over grows") while the seeded BUG family *rises* 14→67→100
and the marquee config holds mk=100 at both contexts (n=8, `results/w15-complete-summary.md:12-13`);
(c) the other 0.1× evictor, snapkv-k0.1, fails mk/mv/vt at both contexts (`:28,55`). So the
$0-supportable rewrite — "channel-pruning and offline low-rank are structurally floored at
0.5–0.75×; eviction reaches 0.1× but loses var-track there and its multikey decays with length" —
retains most of the frontier. What stands: EA/SnapKV were never run on Qwen/Mistral or vs the
seed-fixed flagship (~$15–25 GPU), and the Week-11 "At 16K: EA" recommendation (`:90-91`) — which
the panel wields — was a *pre-seed* verdict about a weakness (the warm-up window) the seed fix was
then shown to remove (Week-14: 16K 100/50/0/0 → 100/100/100/88). Fatal → major-wording + one cheap
grid add.

## 7. OVERSTATED — "pre-registration was quietly violated for the very cells now cited" (attack A4.ii, rigor m1)

The pre-registered n=8 protocol (`docs/week15-significance.md:66`) governs **ppl tie claims**. Every
cited tie is Week-15 n=8 with the paired deltas printed in the committed summary
(`results/w15-complete-summary.md:17-24`: +0.0056/+0.0042 @16K, +0.0311/+0.0240 @32K). The Week-17
n=4 "leaner ppl" cells (`scripts/pod/w16.sh:47`, disclosed in commit 3e27089's own title) back only
(i) floor rescues with 1.5–4 order-of-magnitude effects (27531.7→6.995) and (ii) *concessions*
(r64 ppl worse than baselines; the Qwen 35.1 dip). No tie-threshold claim rests on an n=4 cell —
the rigor review itself concedes n=4 suffices for C3's effect sizes. Residual ($0–10): restore n=8
for any ppl cell the paper prints, and state window counts. "Quietly violated" additionally ignores
that the halving is in a titled, pushed commit and a commented pod script — the opposite of quiet.

## 8. OVERSTATED — "think-c0.3 plausibly erases the marquee separation" (rigor M3)

think-c0.3 costs **0.852×** (`docs/week11-decision-table.md:30,57`) — more memory than think-c0.5
(0.750×) and 5.4× the marquee's 0.159×. A config at 85% of full memory scoring vt=100 (n=2 at 32K)
no more contests a 0.16× frontier point than full KV (1.0×, vt=100) does; the marquee claim is
quality-*at-memory*, and every point on the table with ≥0.6× memory already scores vt ≥75
(morph-k0.5 100 @0.624×). The $0 fix — name the comparator ("beats think-c0.5") — fully protects
the claim; the ~$8 c0.3 rerun is optics, not protection. The genuine residuals: marquee single/mk at
n=16 unfilled (n=8=100 exists from Week-15), and the C4 ppl/retrieval cross-week provenance must be
stated (marquee row has no ppl field, `results/w17-decision-table.json:305-311` — verified).

## 9. OVERSTATED — "multikey was dropped: selective reporting" (claims minor+, attack A1.5)

The omission is real (w17 runs used `--tasks niah_single niah_multivalue vt`, `w16.sh:145-151`;
every mk cell "—"). But the *direction* of the insinuation is backwards: multikey is the task BUG
historically **wins** — Week-9's airtight 6/6 vs morph 0/6; marquee mk=100 at 16K *and* 32K (n=8,
`results/w15-complete-summary.md:12-13`); Week-14 seed pins mk 100 (n=8); EA's mk decays with
context (§6) — while vt, the task BUG *loses* on 1024-dim models, was kept and headlined as an
honest limitation (`docs/week17-explained.md:33-36`). A team cherry-picking drops its worst task,
not its best. The proximate cause is on record: the Week-14 documented ~4× sizing miss ("omitted
tasks/cell") that forced matrix cuts. Residual stands: mk has never been measured on Qwen/Mistral
(~$5–10), and one sentence of rationale belongs in the paper.

## 10. OVERSTATED in part — "min_sv_frac = re-enabling CKL's published truncation, default-off paradox" (attack A2, significance W5)

Half-right, and the panel's reframe (stability subsection, cite CKL's ϑ, lead with the
rank-siphoning diagnosis) should be adopted. But two corrections: (a) the in-repo `theta` is an
**absolute** Frobenius-tail tolerance (`streaming_torch.py:67-77,88`), which cannot be set once
across layers/streams whose singular-value scales differ by orders of magnitude; `min_sv_frac` is a
**self-scaling relative** floor ("no per-stream tuning", `:194-200`) — a real, if modest, delta from
the published step, and the honest claim is "KV streams make CKL-style truncation mandatory and the
robust form is relative." (b) The "default-off paradox" dissolves on the record: the floor is a
measured **strict no-op below the rank-wall onset**, the flagship r64 sits below onset on all three
families (no divergence anywhere in 17 weeks of r64 cells), default-off preserves bit-identical
reproducibility of archived results (`tests/test_w17_rankfloor.py`), and the handover *already*
proposes promoting it to default (`docs/week17-handover.md:85-86`). Conceded: the rescued high-rank
cells are dominated by palu at matched memory (Qwen r256-floored 6.995@0.517× vs palu 6.355@0.504×),
so "extends the *safe* rank" must not become "extends the *competitive* rank"; retrieval-neutrality
is n=4.

## 11. STANDS with corrections — custom generator / external validity (rigor F1, claims P4)

Conceded as the program's largest open scientific risk: 10 fixed sentences cycled (~136× at 16K,
`scripts/w4_needle.py:46-57`, `w10_ruler.py:70-76`), no official RULER/LongBench number on any
flagship config, selection and confirmation on the same generator. The realistic-filler +
depth-sweep + official-RULER pass (~$25–50) is the right top priority. Three corrections to the
severity narrative: (a) the harness is query-agnostic — compression happens *before* the question is
revealed (`w10_ruler.py:20-26`), the honest streaming protocol, and the "fairer-to-eviction"
question-in-prompt harness is disclosed as such in-repo (`w10_longbench.py:4-9`,
`week17-handover.md:82-84`: "realistic-QA is eviction's home turf"); (b) "maximally friendly to both
mechanisms" must explain why the same generator produced BUG zeros for years of cells (§3) and a
rank wall on a low-rank stream — the benchmark demonstrably has teeth against this method; (c)
mk/mv/vt items span depths 11–89% (§3), so the depth criticism is single-task. The panel's cyclic-
filler *description* is accurate; the claim that the results are therefore "a sanity check, not a
frontier" is a hypothesis the ~$25 filler swap will test, not a demonstrated fact.

## 12. STANDS — conceded fatal/major items (the honest reject-risk core)

1. **Quantization baselines absent (prior-work fatal, significance W1).** No KIVI/KVQuant/CacheGen
   arm anywhere; conceded, ~$50. Mitigations on record: the offending sentence names only
   eviction/channel-pruning; the repo never hid quantization (main.tex:119 concedes 4-bit
   near-lossless at 0.25×; Week-4 "2-bit cliff"; Week-5 measured ×TurboQuant compositions incl.
   rank-64/4-bit = 0.099×); and an in-repo sub-cliff cell already exists — bugS-r32-h256 at
   **0.043×** scoring 100/67/100/100 @32K (n=6–14) and 0.038× @64K (`week11-decision-table.md:47,67`)
   — i.e. the *actually* exclusive band below the scalar-quant cliff is occupied by measured cells;
   the paper should claim that band and compose with quantization rather than compete with it.
   Also contra attack A3.2 ("the regime does not exist on any billing"): at the architected fp16
   storage billing the r64 asymptote is r/n = 0.0625 for the 1024-dim families — 2× *below* the
   2-bit slope; parity holds only under fp32-at-rest billing, which is what the ~$10 fp16-store
   probe decides. (Qwen, n=512, is above 2-bit on any billing — say so.)
2. **Storage-vs-resident memory (systems fatal ×2).** Verified end to end: persistent mid-cache
   (`bug_cache.py:1578-1581` early-return), workspace ≈0.98× (`week16-handover.md:20`), no
   `w16-storage.json`, no `figures/week16/`, caveat absent from both explained docs (grep verified
   empty), only latency datum unfavorable (1B/CPU/fp32, Week-5). Fully conceded: the $0 "stored
   state" rewrite, promotion of the `w16_storage.py:1-27` paragraph, and the ~$10 M1/M2 load-path +
   peak-VRAM measurements are mandatory before submission. Noted for the record: the caveat is the
   *program's own* Tier-4 finding, in a committed script and handover — a disclosure-location
   failure, not a concealment.
3. **fp32-at-rest vs fp16 billing (claims P5).** Real: `u_k/c_k` fp32 at rest (`bug_cache.py`
   `_reset_state` comments) billed at 16 bits (`accounting.py:76-85`) while baselines store 16-bit.
   Mitigations: the convention is documented ("the deployment headline", `accounting.py:26-31`) with
   the fp32-word `float_equiv` published alongside; the store-vs-compute dtype split is an
   architected design ("bf16 storage is safe, bf16 *core* math is not", `streaming_torch.py`
   docstring); and even at fp32-at-rest (~0.15×/0.27×) the frontier survives at 3.4–5×. The ~$10
   fp16-store probe (or dual-billing table) is required.
4. **32K at n=4, no flagship 64K; EA/quant/mk gaps on Qwen/Mistral; LongBench on flagship.**
   All conceded, ~$40–60 total.
5. **Reproducibility chain** (repro majors): per-trial records and env metadata died with pods;
   clone-by-branch; CI excludes week7; unsloth-mirror licensing; stray files (`w15b-complete-lines.txt`
   untracked — verified in git status). Conceded with corrections: `hits = round(acc·n)` recovery is
   *exact* at n ≤ 16 (`week15-significance.md`: "recoverable exactly"), the generator is
   deterministic given (seed, trial) so every prompt byte-reconstructs, python deps ARE pinned in
   the pod script (kvpress==0.5.1, transformers==5.8.0, datasets==2.21.0 — `w16.sh:38-39`, contra
   "unpinned atop the image"; torch/CUDA/image are not), and cross-pod bit-identical ppl
   reproduction was empirically observed (`w15-complete-summary.md` process note). Hours of $0 fixes
   + ~$30–70 regeneration.
6. **Manuscript debt (attack A7, prior-work 2c).** paper/main.tex = 3 bibitems, Week-4 content —
   verified. Out of the panel's charter (readiness-to-write-up, briefing:3-4), but the ~20-citation
   related-work section incl. LESS/LoLA/KIVI/DLRT and the EA name-collision fix is real, load-bearing
   writing (~3–5 days) and the concurrent-work clock (MomentKV/ResKV) is running.

## 13. Two structural defenses the panel under-weighted

- **C1 is one uniform config, not a per-model patchwork.** Attack A1.4 claims "a method whose
  per-model recipe differs in rank, tier size, scoring rank, and floor — chosen post-hoc per
  family". The C1 arm is a single fixed string on all three models
  (`w16.sh:142`: `--ranks 64 --hh-budgets 256 --hh-neighbor 1 --warmup-seed`; no s32, no floor,
  identical n, identical tasks). The per-model variation belongs to *other* claims (marquee = a
  second disclosed operating point on the same rank knob; floor = the C3 fix, off in C1). And on
  Llama, one config (the marquee) does deliver both axes at once: 100 on all four tasks at both
  contexts + pre-registered ppl ties at 0.16–0.19× (`w15-complete-summary.md:9-24`), so A4.iv's
  "no single configuration delivers both headlines" is false on the model where both headlines are
  claimed.
- **The claims got *stronger under scrutiny the program itself paid for*.** Every downgrade the
  panel cites (vt 0.75→0.58, "beats Palu" retracted at n=16, Week-15 discovery of two of its own
  baselines' bugs *in the baselines' favor*, the h512/h1024 refutation, the Q-BUG/codebook kills)
  was self-inflicted before any external reviewer existed. The rejection case's factual errors
  (§1, §2, §4) were caught by reading the same files the program committed. That is evidence the
  record can survive an adversarial camera-ready process once the conceded runs land.

## Scorecard of the panel's fatal/major findings

| finding | panel severity | defense verdict |
|---|---|---|
| LongBench "collapse" datum (rigor M1/A3.3) | major | **REFUTED** (misread file; BUG leads sub-0.25× eviction in it) |
| matched-r/n control missing (A6.2) | major | **REFUTED** (Llama r256 = 0 everywhere, Week-11) |
| ceiling-effect / effective-n (A1.1, A1.3, rigor M4) | fatal-adjacent | **REFUTED** as stated; survives as external-validity scope |
| 32K "false for Mistral" (claims P2) | major | **REFUTED** as repo indictment (briefing artifact; docs scoped) |
| 5–13× not reproducible (claims P1) | major | **OVERSTATED** → minor, $0 |
| "eviction cannot enter" (claims P3, rigor F2) | fatal | **OVERSTATED** → major wording + ~$20 grid; EA vt=17, mk decays |
| pre-registration violated (A4.ii) | major | **OVERSTATED** (no tie claim on n=4 cells; disclosed commit) |
| think-c0.3 erases marquee (rigor M3) | major | **OVERSTATED** (0.852× memory; full-KV argument) |
| multikey drop = selective reporting (A1.5) | major | **OVERSTATED** (dropped its *best* task; documented sizing miss) |
| min_sv_frac = re-enabled CKL / default-off paradox (A2) | fatal-adjacent | **OVERSTATED** (relative vs absolute ϑ; no-op below onset) |
| custom generator, no official benchmark (rigor F1) | fatal | **STANDS** (with §11 corrections), ~$25–50 |
| quantization baselines absent (prior-work/significance) | fatal | **STANDS**, ~$50; sub-cliff band already occupied in-repo |
| storage vs resident memory (systems ×2) | fatal | **STANDS**, $0 rewrite + ~$10 measurements mandatory |
| fp32-at-rest billing (claims P5) | major | **STANDS**, ~$10 probe or dual billing |
| repro chain / env / licensing | major | **STANDS** with corrections (deps pinned; recovery exact) |

**Post-defense calibration: 6/10.** The rejection case's three genuinely fatal residues
(quantization arm, storage reframe, external benchmark) are all priced, none contradicts a measured
in-repo result, and two of them already have their honest framing written inside the repo. The
attack's 3/10 required its factual errors (§1–§4) to hold; they do not.
