# `vt` template comparison — generator v2 vs official RULER `variable_tracking`

## 1. Purpose

2026-09-20. `prereg/gate1_preflight.md` §4 (ii) (l. 231–245) fixed the uncompressed ceiling bar
at ≥ 0.9 per task and pre-committed that a task below it is "a **generator finding, not a
compression one**". It fired on `vt` and nothing else: `full` = 9/12 = 0.75 (A1.2 (ii),
l. 652–657). This is the "comparison of the generator's template against official RULER's" that
A1.2 (ii) (l. 668–670) names as the first step of the repair; it feeds
`prereg/gate1_tracker_swap_v2.md` Amendment 1b and decides nothing (D-018, DECISIONS.md:198–199).
Question: is v2's `vt` faithful, and is 0.75 the model or an artefact?

## 2. Sources

| what | where |
| --- | --- |
| v2 `vt` builder | `src/kvdlra/eval/gen.py`:61–78 (templates, `N_HOPS`, `MAX_NEW`, `DEPTH_GRID`, `STEP`), 201–212 (`_place`, `_wrap`), 222–245 (`_values`, `_variables`), 274–282 (the `vt` branch) |
| hit rule | `src/kvdlra/eval/ruler.py`:381–382 — `frac = sum(t in text for t in targets)/len(targets)`; `hit = frac >= 1.0` |
| task config | `configs/tasks/ruler_v2_16k.yaml`:11–12, 16; Gate-1 twin `configs/tasks/ruler_v2_16k_g1.yaml`:15, 17 |
| what tests pin | `tests/test_gen_v2.py`:226–230 — the question string, 5 targets matching `[A-Z]{5}`, and `VAR t1 = VAR t0` present in the prefill. Nothing pins chain ORDER, value type, or the answer prefix. |
| official generator | `NVIDIA/RULER@main scripts/data/synthetic/variable_tracking.py` (fetched 2026-09-20): 86 (noise string), 91 (`DEPTHS`), 93–114 (`generate_chains`), 130–157 (`generate_input_output`), 191–296 + 299–314 (the one-shot wiring) |
| official template | `.../synthetic/constants.py`:31–35 |
| official task args | `NVIDIA/RULER@main scripts/synthetic.yaml`:95–100 — `type_haystack: noise`, `num_chains: 1`, `num_hops: 4` |
| official metric | `.../scripts/eval/synthetic/constants.py`:28–29 (`string_match_all`) and 34 (`variable_tracking` uses it) |
| the typo | RULER commit `4a216f47` (2025-05-13, "fix typo") changed `assgined` → `assigned` in `constants.py`. Our pin `c3f5e3b` (2026-07-22, `src/kvdlra/eval/official_ruler.py`:6) is **after** that fix. |
| pre-flight records | `results/gate1_preflight/trials.jsonl`; `[trial]` lines at `results/gate1_preflight/gate1_preflight-51722149.log`:15488–15513 |
| prior art | DECISIONS.md:145 (D-005 addendum 2, the v1 `= = Title = =` collision hypothesis), :177–186 (D-011 addendum 10), :198–199 (D-018) |

All fetches succeeded. Not found anywhere: an NVIDIA-published per-task `vt` cell for
Llama-3.1-8B-Instruct (see §5).

## 3. Side-by-side

| property | official RULER `vt` | generator v2 `vt` | same? |
| --- | --- | --- | --- |
| chains | 1 (`synthetic.yaml`:99) | 1 (`gen.py`:276–278) | yes |
| hops | 4 (`synthetic.yaml`:100) | `N_HOPS = 4` (`gen.py`:71) | yes |
| statements | 5 (`VAR A = <v>`, then 4 × `VAR B = VAR A`) | 5, same shape (`gen.py`:276–278) | yes |
| distractor variables | none | none | yes |
| answer | all 5 names (`vars[0]`, l. 177) | `targets = names`, 5 (`gen.py`:282) | yes |
| variable names | 5 upper-case letters, `random.choices` **with** replacement (l. 98) | 5 letters from one 26-permutation, no letter reused (`gen.py`:238–245) | v2 easier |
| value | `np.random.randint(10000, 99999)` — 5-digit integer, always (l. 110) | 7-digit integer **or** `adjective-noun` from the cell's `words` family (`gen.py`:222–235; ruling R-L2-4, `ruler_v2_16k.yaml`:11–12) | **no** |
| instruction + question | `constants.py`:33 | `VT_TEMPLATE`, `gen.py`:61–65 — character-identical | yes |
| answer prefix | `…variables are **assigned** the value…, they are: ` (`constants.py`:34, post-`4a216f47`) | `…are **assgined**…` (`gen.py`:66–69) | **no** |
| one-shot example | **always present** — `main()` builds a 500-token ICL sample and `sys_vartrack_w_noise_random` splices it in (`add_fewshot=True` by default, l. 194, 262–265; the `--add_fewshot` argparse flag is never passed) | **absent** | **no** |
| haystack | repeated "The grass is green. The sky is blue. …" (l. 86), one per line | real documents from pg19 / arxiv / wikipedia / essays, sentence-split, space-joined (`gen.py`:122–127, 172–198) | **no** (deliberate, D-005) |
| statement terminator | noise mode: own line, no period (l. 157); essay mode: `.strip() + "."` (l. 148) | none — runs into the next real sentence under `" ".join(sents)` (`gen.py`:276–280) | **no** |
| chain positions | random, then **`sorted`** and zipped with chain order (l. 154–156; essay mode l. 138–148 likewise) → definition always first | design depth + 0.2·i **modulo 1** (`_wrap`, `gen.py`:210–212) → wraps | **no** |
| decode budget | 30 (`constants.py`:32) | 64 (`gen.py`:75) | v2 more generous |
| scoring | `string_match_all` = **partial credit**, case-insensitive, averaged (`eval/…/constants.py`:28–29) | `frac` is exactly that but case-**sensitive**; the reported `acc` additionally thresholds at `frac ≥ 1.0` (`ruler.py`:381–382) | **no** — v2's headline is stricter |

**The wrap, reproduced** — `gen._wrap`/`gen._place` on a 100-sentence stand-in:

| design depth | needle depths | textual order | forward references |
| --- | --- | --- | --- |
| 0.05 | 0.05 0.25 0.45 0.65 0.85 | `A=<v>`, `B=A`, `C=B`, `D=C`, `E=D` | **0** |
| 0.40 | 0.40 0.60 0.80 0.00 0.20 | `D=C`, `E=D`, `A=<v>`, `B=A`, `C=B` | **2** |
| 0.95 | 0.95 0.15 0.35 0.55 0.75 | `B=A`, `C=B`, `D=C`, `E=D`, `A=<v>` | **4** |

Official RULER never produces a forward reference, in either haystack mode. This is the one
deviation that is *invisible* on the four `niah_*` tasks, whose needles are order-independent —
which is why they sit at 1.00/1.00/0.92/0.92 on the same 12 haystacks.

## 4. The 12 pre-flight rows

`full`, `unsloth/Meta-Llama-3.1-8B-Instruct`, `ruler_v2_16k`, seed 0, 16 384 ctx, 0 errors.
The records carry `hit` and `frac`; **no completion text is stored anywhere** — not in
`trials.jsonl`, not on the `[trial]` lines, not in `diag.jsonl` (whose keys are tracker
diagnostics only). Which variable each miss dropped is therefore not recoverable.

| trial | depth | fwd refs | family | haystack | hit | frac |
| --- | --- | --- | --- | --- | --- | --- |
| 0, 6 | 0.05 | 0 | numbers, words | `pg19:d0` | 1, 1 | 1.0, 1.0 |
| 1, 7 | 0.05 | 0 | numbers, words | `arxiv:d1..d5:5` | 1, 1 | 1.0, 1.0 |
| 2 | 0.40 | 2 | numbers | `arxiv:d0..d2:3` | 1 | 1.0 |
| **8** | 0.40 | 2 | **words** | `arxiv:d0..d2:3` | **0** | **0.4** |
| 3, 9 | 0.40 | 2 | numbers, words | `wikipedia:d1..d3:3` | 1, 1 | 1.0, 1.0 |
| 4, 10 | 0.95 | 4 | numbers, words | `wikipedia:d0..d3:4` | 1, 1 | 1.0, 1.0 |
| **5** | 0.95 | 4 | numbers | `essays:d1..d9:9` | **0** | **0.8** |
| **11** | 0.95 | 4 | **words** | `essays:d1..d9:9` | **0** | **0.2** |

- **Every miss is partial, none is a zero** (4, 2 and 1 of 5 names): the model located the chain
  in all twelve and failed to close it three times.
- The all-or-nothing rate is **9/12 = 0.750, Wilson 95 % [0.468, 0.911]**. The RULER-comparable
  number — `string_match_all` = the mean `frac`, printed by the runner as `recall` on log
  l. 15513 — is **0.867 (52/60 slots), Wilson 95 % [0.758, 0.931]**.
- **Monotone in forward references**: 0.05 → 4/4 (mean frac 1.00), 0.40 → 3/4 (0.85), 0.95 → 2/4
  (0.75). Suggestive, **not significant**: in-order 4/4 vs wrapped 5/8 is Fisher p = 0.49.
- **Confounded.** `design_source` (`gen.py`:159–169) gives `essays` only at d = 2 in a 2 × 3 × 2
  design, so `essays` and depth 0.95 cannot be separated: within 0.95, `wikipedia` is 2/2 and
  `essays` 0/2. Family is 5/6 numbers vs 4/6 words. `essays` is `sgoel9/paul_graham_essays`
  (`haystacks.py`:78–80) — the corpus RULER itself ships as the alternative vt haystack, so "real
  documents" is not exotic. The Gate-1 design (`codes: 4`) crosses every (depth, family) with all
  four sources and de-confounds this at n = 24.
- A1.2 (ii) (l. 674–676) records "the three misses are spread, not stacked" — true of family and
  source; the depth axis is the one that is *not* flat.

## 5. The published reference

- **NVIDIA does not publish this cell.** The RULER README leaderboard lists "Llama3.1 (8B)" only
  as aggregates — 4K 95.5, 8K 93.8, **16K 91.6**, 32K 87.4, 64K 84.7, 128K 77.0, Avg 88.3 — with
  no per-task column; the paper (arXiv:2404.06654, latest v3 2024-08-06) predates Llama-3.1.
- **Third-party runs of the official generator do.** Double-P (arXiv:2602.05191) Table 4, "Ruler
  results across tasks for LLaMA-3.1-8B and Qwen3-8B": full attention **vt = 99.60 at 16K**
  (Table 5: 99.80 @32K; Table 6: 98.70 @64K). vAttention (arXiv:2510.05688) Table 4, "RULER @ 32K
  (Meta-llama/Llama-3.1-8B-Instruct)": dense **vt = 97.4**. Both are `string_match_all`, so they
  compare to our 0.867, not our 0.750. Two independent reproductions near 99 is the best
  reference available; neither is NVIDIA's own.
- **The gap is not sampling noise.** Our 0.867 has an upper 95 % bound of 0.931. Treating 99.6 as
  a per-trial all-correct rate, P(≤ 9 of 12) = 1.4 × 10⁻⁵; at a charitable 0.98, P = 1.5 × 10⁻³.

## 6. Verdict — **(B)**: v2 deviates, and at least one deviation plausibly lowers the ceiling

**For (A).** Chain count, hop count, statement shape, target set and both prose templates are
RULER's character for character; the decode budget is larger, the variable names are *less*
confusable, and the metric family is the same. The misses span both code families, two of three
depths and two of four sources — not what a single-cell defect looks like. Wilson [0.47, 0.91]
overlaps 0.9 and the depth gradient is Fisher p = 0.49, so n = 12 alone cannot reject "model".

**For (B).** Five differences, in descending order of how plausibly they cost accuracy:

1. **Chain order.** `_wrap`'s mod-1 rotation makes the definition the *last* of the five
   statements at 0.95 and the third at 0.40; official RULER sorts positions before zipping them
   with the chain, so the value always precedes every reference. The 0 → 1 → 2 miss count tracks
   0 → 2 → 4 forward references, and this is the only deviation confined to `vt` — the task that
   fired the bar — while the four order-independent `niah_*` tasks sit at ceiling on the same
   twelve haystacks.
2. **No one-shot example.** Official `vt` prompts always carry a randomized ICL demonstration
   (l. 194, 262–265); v2's are zero-shot. Every reported ≈ 99 number is a one-shot number.
3. **The `words` family.** Official values are 5-digit integers only; R-L2-4 lets v2 draw
   `adjective-noun`. 2 of 3 misses are `words` — but so are 4 of 9 hits, and the 0.8 miss is
   `numbers`.
4. **No terminal period.** v2's statements splice into a space-joined run of real sentences with
   no terminator, so `VAR ABCDE = 4812993` fuses with the next sentence; RULER appends "." in
   essay mode and gives each its own line in noise mode.
5. **`assgined`.** Fixed upstream in `4a216f47` (2025-05-13); our pin `c3f5e3b` is a year later,
   so v2's priming string misspells the key verb relative to official RULER at our own pin.

**Smallest fix** — three edits in `kvdlra/eval/gen.py`, all pure RULER fidelity, no new
machinery: (a) one line in the `vt` branch — pair the five wrapped depths with the chain **after
sorting them ascending**, so the definition leads and depth coverage is unchanged; (b) append
`"."` to each chain statement; (c) `assgined` → `assigned`. Defer the one-shot example (largest
change, consumes context budget) and leave R-L2-4 alone until (a)–(c) are measured.

**Cost to validate** — one cell: `full`, `vt` only, `ruler_v2_16k`, n = 12. At the measured
`full` rate of 0.055 min/sample (A1.2 (iv)) compute is ≈ 40 s; the bill is boot + weights,
≈ 1.0–1.2 GPU-h ≈ **$0.6–0.9** at the $0.60–0.67/h A100 offers in use. Ordering hazard: any of
(a)–(c) changes `vt`'s `prompt_sha256`, so repaired rows are not poolable with the pre-flight's
or with Stage 1's.

## 7. What Amendment 1b should say (quotable)

> `vt` leaves the Gate-1 primary retrieval family (16 → 12) and the secondary family (24 → 18)
> under `prereg/gate1_preflight.md` §4 (ii), because the pre-flight `full` ceiling was 9/12 = 0.75
> (RULER-comparable `string_match_all` 0.867, Wilson 95 % [0.758, 0.931]) against third-party
> official-RULER runs that put full-attention Llama-3.1-8B at vt ≈ 99.6 at 16K
> (arXiv:2602.05191 Table 4; arXiv:2510.05688 Table 4 gives 97.4 at 32K).
>
> `docs/plan/reports/vt-template-comparison.md` attributes the shortfall to `kvdlra.eval.gen` and
> not to the checkpoint: v2's mod-1 depth wrap presents the assignment chain out of order at
> depths 0.40 and 0.95 — where official RULER is always definition-first — and v2 additionally
> omits RULER's mandatory one-shot example, may draw the value from the `words` family, and ends
> each chain statement without a period; the repair required by §4 (ii) is scoped to those items
> and validated by one `full`-arm `vt` cell at ≈ 1 GPU-h.
>
> Until that repair lands and is measured, Stage 1 runs `vt` on the unrepaired generator and
> reports it descriptively (D-018), its rows are read only against the pre-flight ceiling this
> amendment records, and no `vt` row from before the repair is pooled with one from after it.
