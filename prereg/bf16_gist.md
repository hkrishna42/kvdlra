# Pre-registration — `isvd_r64_h256_seed_bf16` on the Gate-1 Stage-1 pods (the bf16 gist)

**STATUS: rides the Gate-1 Stage-1 pods (DECISIONS D-003; launch under D-011).** This file
creates no pod and launches nothing. It reads **one arm** — arm 6 of the eight
`prereg/gate1_tracker_swap_v2.md` §3 puts on `gate1_v2_stage1_llama` and
`gate1_v2_stage1_qwen` — and it is committed strictly before those pods' launch commit, in the
commit that adds it and nothing else. Lane L3 Task 4b (plan item L5.2); gate G5 line 1
(`docs/plan/lanes/GATES.md`: "bf16 run harvested under prereg; non-inferiority result in
DECISIONS.md").

**Everything about the run is the Gate-1 design, quoted, never restated differently.** The two
pods, their arm order, the four tasks at n = 24, the 32 paired perplexity windows, the
byte-identical prompts, the α = 0.05 level, the ±0.02 bits/token margin, the refusal rules, the
budget and the amendment discipline are all fixed by `prereg/gate1_tracker_swap_v2.md`; where a
fact of the run appears below it is a quotation of that file with its section named. What is
**new here** is the reading: §4's non-inferiority rule, §6's two families, and the consequence
§4 pre-registers for whichever way it goes. `prereg/gate1_tracker_swap_v2.md` §3 and §7 (b) point
at this file for exactly that, and its §6 puts the bf16 arm outside all three Gate-1 families:
"Everything not in the three families is **descriptive** and carries no corrected p-value: …
the bf16 arm (its own file)." **No member of this file enters a Gate-1 family, and no Gate-1
p-value is recomputed when this reading lands.**

---

## 1. Purpose

`configs/arms/isvd_r64_h256_seed_bf16.yaml` (lane L3 Task L5.1, commit 63dfa80, amended a27058d)
is `isvd_r64_h256_seed` with one knob moved: `gist_dtype: "bfloat16"`. The basis `U`, the
diagonal core `B` and the coordinate tier `C` are held at 16 bits **at the store boundary only**;
every absorb upcasts them to fp32 working copies for the incremental-SVD step, the coordinate
carry `rot @ C`, the quantized-tier rotation, the orthonormality guard and the surprise scores,
then rounds the store back down once (`BugStreamingLayer._cast_gist`, `src/kvdlra/cache/bug_cache.py`
`_absorb_columns` entry and exit). `U` and `C` are billed at 16 bits
(`kvdlra.accounting.bug_footprint(gist_bits=16)`); the diagonal core stays in `aux_words` at 32,
which is conservative and is 2r words per layer against the basis's 2nr.

**The question this file answers:** at 16K, on two model families, is the bf16 arm **non-inferior**
to the fp32 arm — on retrieval, per task, and on perplexity, within ±0.02 bits/token — when it is
fed byte-identical prompts and the same windows in the same pod?

It matters for one reason beyond the arm itself. §2 (a) computes that the halving is real and
large at rest: **0.150× → 0.085× stored bits on Llama-3.1-8B and Mistral-7B (n = 1024) and
0.275× → 0.148× on Qwen2.5-7B (n = 512)** at t = 16384. Every byte-matched control in this
project is solved against the fp32 arm's stored bits, so a pass does not merely license a
cheaper default: it **re-plans** those controls at new bytes (§4's consequence), and a plan that
is not re-planned is an unmatched comparison. Nothing else is decided here (§11) — in particular
this file makes no resident-memory, throughput or latency claim (§5, §7 e; Gate 3 is
`prereg/kernel_smoke.md`), and it does not ask whether bf16 is the right default at any other
rank, context or tier budget.

## 2. Measured baseline — what is known, and what it predicts

Every number below was computed at the commit that adds this file by the snippet beside it, or
is quoted with its source. **No row of this arm exists on a GPU anywhere** — the arm has CPU
tests (`tests/test_bf16_gist.py`, 9 tests) and nothing else, and per
`prereg/gate1_tracker_swap_v2.md` §2 (f) no arm of any kind has been scored on generator v2 on a
GPU beyond the pre-flight's four. Stage 1 is the first measurement of this arm.

### (a) The accounting, exactly — what "halving the gist" is worth at 16K

At t = 16384 the r64 arm holds `16384 − 256 (exact tier) − 4 (sinks) − 32 (ring)` = **16,092**
coordinate columns — the convention every 16K arm `doc:` and
`prereg/gate1_tracker_swap_v2.md` §3 solve their byte match in:

```python
from kvdlra.accounting import bug_footprint
T, HH, SINK, RING = 16384, 256, 4, 32
cc = T - HH - SINK - RING                                   # 16092
for n in (1024, 512):
    for gb in (32, 16):
        f = bug_footprint(n, 64, cc, RING, n_sink=SINK, retention="lowrank_surprise",
                          hh_count=HH, gist_bits=gb)
        print(n, gb, f"{f.stored_bits():.0f}", f"{f.ratio_stored_bits(T, n):.6f}",
              f"bits16={f.bits(16):.0f}")
```

| layer width `n` | arm | `stored_bits()` / layer | `ratio_stored_bits(16384, n)` |
| --- | --- | --- | --- |
| **1024** (Llama-3.1-8B, Mistral-7B-v0.3) | `isvd_r64_h256_seed` (`gist_bits=32`) | 80,717,568 | **0.150348** |
| | `isvd_r64_h256_seed_bf16` (`gist_bits=16`) | **45,664,000** | **0.085056** |
| **512** (Qwen2.5-7B) | `isvd_r64_h256_seed` | 73,836,288 | **0.275062** |
| | `isvd_r64_h256_seed_bf16` | **39,831,296** | **0.148383** |

The denominators are the full fp16 cache, `2·t·n·16` bits/layer = 536,870,912 (n = 1024) and
268,435,456 (n = 512). The 80,717,568 and 73,836,288 are the same two numbers
`prereg/gate1_tracker_swap_v2.md` §3 solves `nogist_h2423` and `nogist_h4460` against, which is
the check that this arithmetic is that arithmetic.

Two consequences worth stating as identities rather than as approximations:

- **The bf16 arm's at-rest bill is exactly the fp32 arm's fp16-equivalent bill.**
  `Footprint.stored_bits()` differs from `bits(16)` only by the `fp32_verbatim_elems` subset, and
  `gist_bits=16` empties that subset (`accounting.py`: "A `gist_bits=16` arm stores C and U in
  bf16 too, so the subset is empty there"), so `stored_bits()_bf16 == bits(16)_fp32 = 45,664,000`
  / `39,831,296` — verified `True` at both widths by the snippet above. The bf16 arm is what the
  fp16-equivalent `ratio` column always claimed the fp32 arm was; the dual-billing gap
  (`ratio` 0.085 vs `sbits` 0.150) is what it closes.
- **The saving is 0.566× (n = 1024) and 0.540× (n = 512)**, not 0.5: the sinks, the ring, the
  256-token exact tier and the `aux_words` (32,568 per layer: 2r core + 2 per coordinate column
  + one position per tier token) do not move, and only the gist's `2nr + 2·r·coord_count`
  elements are re-billed.

### (b) The bf16-rounded basis, and why the guard is load-bearing here

A bf16-rounded orthonormal basis is not orthonormal. The committed generator
`kvdlra.eval.recon.bf16_gist_drift(n, rank, block, absorbs, seed)` (added by L5.1's fix round,
a27058d) builds two `BugStreamingLayer`s on one seeded synthetic stream — one at
`gist_dtype=torch.float32`, one at `torch.bfloat16` — drives `absorbs` blocks through
`_absorb_columns` (the cache-level path, guard included) and returns each layer's **final stored**
`‖UᵀU − I‖_F` beside their relative reconstruction difference. Regenerated at the commit that adds
this file:

```
$ .venv/bin/python -c "
from kvdlra.eval.recon import bf16_gist_drift
for absorbs in (40, 200, 1000):
    print(absorbs, bf16_gist_drift(1024, 64, 16, absorbs, 0))
print('n=512', bf16_gist_drift(512, 64, 16, 40, 0))
"
40   {'rel_error': 0.010126033797860146, 'orth_error_bf16': 0.004608154296875, 'orth_error_fp32': 3.7591493310173973e-05}
200  {'rel_error': 0.0226544588804245,  'orth_error_bf16': 0.004547119140625, 'orth_error_fp32': 9.618153126211837e-05}
1000 {'rel_error': 0.04991922155022621, 'orth_error_bf16': 0.00469970703125,  'orth_error_fp32': 0.00034641928505152464}
n=512 {'rel_error': 0.010289540514349937, 'orth_error_bf16': 0.0067138671875, 'orth_error_fp32': 3.8297112041618675e-05}
```

So the stored basis sits at **4.6e-3 at n = 1024, r = 64** and **6.7e-3 at n = 512** — above
`orth_fix_tol` = 1e-3 and two decades below `orth_abort_tol` = 1e-1, which is the regime the
shipped guard was written for. Two other measurements of the same quantity, quoted with their
construction because they differ and the difference is not a defect: random orthonormal draws
(`qr(randn)`, rounded once, 5-seed mean) give **4.9e-3** at n = 1024 (range 4.77e-3–5.02e-3),
**7.0e-3** at n = 512 and **3.8e-3** at the tiny model's n = 32, r = 8; and the live tiny-model
stored basis after 44 absorbs read **3.95e-3** — all three in the task report
(`.superpowers/sdd/2026-09-11-L3-L5-gate1-bf16-prereg/task-2-report.md` §2 (a), and its fix round
for the reconciliation). They land in one band, a few × 1e-3, which is the only property §4 and §7
read; the generator's own number is the one this file quotes because it is the committed,
regenerable one.

**The guard therefore repairs on every absorb but the first** (the seeding absorb builds its basis
from nothing, so the step already returns an orthonormal factor). Measured on the same synthetic
stream at n = 1024, r = 64, 1000 absorbs: `fixed_k` **999/1000** under bf16 storage against
**0/1000** at fp32, with a maximum pre-repair `orth_err_k` of 5.26e-3 (bf16) against 3.50e-4
(fp32); on the tiny model, 43/44 against 0/44 (task report §2 (c)). The 999/1000 count is a
**report-only** number from a scratch script — no committed test asserts it; the committed test
asserts the direction and shape at a tiny size
(`tests/test_bf16_gist.py::test_bf16_gist_drift_shape_and_direction`), and the tiny-model 43/44 is
covered by that file's guard test. §7 (a) turns the claim into a pass/fail check on the pod's own
`diag.jsonl` rather than resting on either.

**The guard is load-bearing for this arm, and this is stated before the run rather than
discovered after it.** With `orth_fix_tol` disabled (or set above ~1e-2) a bf16-gist run would
accumulate basis error unchecked. **No such variant is run**: every Stage-1 arm carries the
shipped guard at `orth_fix_tol` 1e-3 / `orth_abort_tol` 1e-1
(`prereg/gate1_tracker_swap_v2.md` §3), and an `OrthonormalityError` on this arm is an `error` row
handled by §4's refusal, never a re-run with the tripwire off.

### (c) The drift curve — the basis holds flat, the coordinates compound

From the same output: the relative reconstruction difference of the bf16 arm's stored gist against
the fp32 arm's, `‖U₁₆C₁₆ − U₃₂C₃₂‖_F / ‖U₃₂C₃₂‖_F`, is **1.0e-2 / 2.3e-2 / 5.0e-2 at 40 / 200 /
1000 absorbs**, while `orth_error_bf16` stays at 4.5–4.7e-3 across the whole range. ×5 absorbs
multiplies the drift by 2.24 and 2.20 against √5 = 2.236, i.e. **√absorbs**, a random-walk
accumulation: the basis rounding is re-imposed and then repaired every absorb, so it never
accumulates, and the coordinates are re-rounded at every `rot @ C` carry with nothing repairing
them.

**Absorbs per prefill.** A 16K prefill is **≈ 800–810 absorbs per layer** — the first 4096-token
chunk in 128-column sub-blocks (32 absorbs) plus the remaining ≈ 12.3 K tokens at
`absorb_block` = 16 — measured, not estimated: `results/filler_realism/diag.jsonl` holds 13
`[diag]` rows per layer per 16K sample at the cache default `diag_every` = 64 and a maximum
`absorbs` of 810 (`prereg/gate1_preflight.md` §6, with its snippet;
`prereg/gate1_tracker_swap_v2.md` §8 reuses the same anchor). A 32K prefill is ≈ 1,600. Extrapolated
on the √absorbs law from the 1000-absorb point, the stored-gist drift at a 16K prefill is
**≈ 4.5e-2** (0.04992 × √(800/1000)) and at 32K **≈ 6.3e-2** (× √(1600/1000)).

### (d) What these numbers are not

A **CPU synthetic stream** — a rank-40 signal with a decaying spectrum plus 1e-2 noise, one seed
(0), `coord_budget` large enough that nothing is evicted, no exact tier, no sinks, no ring, no
real model. The 16K deployment keeps its whole tail too (`coord_budget: null`), so the geometry
matches, but a real key stream's spectrum and the exact tier's removal of outliers move the
constant. **It is a compounding-*rate* claim (≈ √absorbs), not a transferable magnitude**, and §5
predicts accordingly. Nothing in §2 is a Gate-1 or a bf16 reading; it is the prior §5 writes from.

## 3. Arms, tasks, n — quoted from the Gate-1 design

- **The arm runs in position 6 of eight on both Stage-1 pods**, after `fd_r64_h256_seed` and
  before `oja_r64_h256_seed_tuned` (`prereg/gate1_tracker_swap_v2.md` §3's arm table and its
  ordered-loss argument; `configs/pods/gate1_v2_stage1_llama.yaml` and `…_qwen.yaml` carry that
  order and `tests/test_pod_manifest.py` pins it arm for arm). Its **record key is
  `isvd_r64_h256_seed_bf16`** — the arm file sets no `legacy_name`, so `load_arm(stem).legacy_name
  or stem` resolves to the stem, and `kvdlra.eval.gate1.ARM_TRACKER` maps it to the tracker label
  **`bf16`**.
- **The paired reference is arm 2, `isvd_r64_h256_seed`** (record key `bugSseed-r64-h256`). The two
  arms differ in `gist_dtype` and in nothing else: rank 64, `coord_budget: null`, ring 32,
  `absorb_block` 16, 4 sinks, `retention: lowrank_surprise`, `hh_budget: 256`, `hh_select:
  surprise`, `hh_neighbor: 1`, `hh_retain: true`, `seed_hh_warmup: true`, `score_rank: null`,
  `min_sv_frac: 0.0`, `tracker: isvd`, no quantization — identical lines in the two YAMLs.
- **Tasks and n, unchanged:** `configs/tasks/ruler_v2_16k_g1.yaml` — generator v2, ctx 16384,
  `seeds: [0]`, `design: {haystacks: 2, depths: 3, codes: 4}` → **n = 24** per (arm, task) cell on
  the four Gate-1 tasks `niah_single`, `niah_multikey`, `niah_multivalue`, `vt`; and
  `configs/tasks/ppl_16k_pg19val.yaml` — PG-19 validation, 16K prefill under the arm's
  compression, a frozen 2048-token scoring window, **32 non-overlapping** windows. **128 samples
  for this arm per pod**, 96 retrieval + 32 perplexity, exactly as every other arm.
- **Pairing.** `gen.make_trial` is deterministic in `(task, seed, trial)` and never sees the arm,
  so the bf16 arm and the fp32 arm are fed the same token ids for a given key, and
  `prompt_sha256` on every record makes that **verified rather than assumed**
  (`prereg/gate1_tracker_swap_v2.md` §3 and §7 (e); 96 keys per pod, each holding 8 records with
  one digest). The perplexity windows are cut identically from one corpus at one context length
  inside a single pod. **Every contrast in this file is within-pod**; none crosses a pod boundary
  and none crosses a model family.
- **Stage 2 and 32K.** `prereg/gate1_tracker_swap_v2.md` §9 sizes the conditional Mistral-7B-v0.3
  16K pod at **all 8 arms**, this one included, and excludes this arm from the three 32K pods —
  "`isvd_r64_h256_seed_bf16`, `oja_r64_h256_seed_tuned` and `random_r64_h256_seed` are not in the
  32K pods, which is the cut ladder applied in advance — the bf16 arm joins by
  `prereg/bf16_gist.md`'s own amendment if its Stage-1 reading passes". So: **a Mistral 16K run
  carries this arm by that file's Stage-2 amendment and is read by a dated amendment here** (§6
  fixes its family sizes now), and **32K is reachable only by an amendment here**, only if the
  Stage-1 reading passes — which is where §5's named 32K risk would be tested. Neither changes
  §4's Stage-1 reading, which stands as read.

## 4. Primary contrast and decision rule

**The reading is NON-INFERIORITY of the bf16 arm to the fp32 arm, per model family, on both
axes.** Non-inferiority is a claim that has to be *earned* by measurements that could have refuted
it, so the rule below says what refutes it, in the records, before they exist.

### The statistic, as code

Every statistic is a shipped `kvdlra.eval.stats` function, already tested in `tests/test_stats.py`.
`kvdlra.eval.gate1` computes the perplexity member itself; the retrieval member it does **not**
compute (its `retrieval_contrasts` loops over `PRIMARY_CONTROLS + SECONDARY_CONTROLS` =
frozen / nogist / oja / fd / random, and `bf16` is in neither), so this file states the retrieval
member as the snippet that produces it — the pattern of `prereg/hygiene_table4.md` §6, where the
Holm step is likewise applied by hand over shipped functions. **No code change is pre-registered
here and none is needed.**

```python
import json, collections
from pathlib import Path
from kvdlra.eval.stats import mcnemar_exact, holm
from kvdlra.eval import gate1

# Retrieval: one member per (family, task). `a` = the fp32 arm, `b` = the bf16 arm, so
# `a_favored` counts the pairs the bf16 arm LOST.
cells = collections.defaultdict(dict)
for pod, family in (("gate1_v2_stage1_llama", "llama"), ("gate1_v2_stage1_qwen", "qwen")):
    for r in map(json.loads, open(f"results/{pod}/trials.jsonl")):
        cells[(family, r["task"], r["arm"])][(r["seed"], r["trial"])] = r["hit"]
members = {
    (f, t): mcnemar_exact(cells[(f, t, "bugSseed-r64-h256")],
                          cells[(f, t, "isvd_r64_h256_seed_bf16")])
    for f, t, a in cells if a == "bugSseed-r64-h256"
}                                    # 8 members at Stage 1: 2 families x 4 tasks
p_holm = dict(zip(members, holm([m["p_value"] for m in members.values()])))

# Perplexity: `gate1.ppl_contrasts` already emits the bf16 member -- it loops over every
# tracker -- with d = isvd - bf16 per window, the paired bootstrap CI, the two-sided paired
# t-test and `stats.tost(d, 0.02)`'s boolean. `make gate1` prints it in the per-family
# perplexity table (columns: bits/token, delta, 95% CI, TOST +/-0.02, Holm p, paired t p).
data = gate1.load([Path("results/gate1_v2_stage1_llama"), Path("results/gate1_v2_stage1_qwen")])
ppl = {(c.family, c.b): c for c in gate1.ppl_contrasts(data) if c.b == "bf16" and c.ctx == 16384}
```

Sign conventions, fixed now: `mcnemar_exact(a=fp32, b=bf16)` → `a_favored` is the number of
`(seed, trial)` keys the fp32 arm hit and the bf16 arm missed; `gate1.ppl_contrasts` computes
`d = bits(fp32) − bits(bf16)` per window, so **`d_bits < 0` means the bf16 arm is worse** (higher
bits/token). The `p_holm` field `gate1.ppl_contrasts` returns is `None` for this member — Holm
there runs over the Gate-1 primary members only — which is correct and intended: this file's
perplexity members are read uncorrected (§6), and its retrieval members carry the Holm of the
snippet above.

### The rule

1. **Retrieval, per (family, task).** The bf16 arm is **non-inferior on that task** unless **both**
   hold: (i) it loses by more than **0.03** on the point estimate (`(a_favored − b_favored)/24`),
   and (ii) that member's **Holm-adjusted p < 0.05** in this file's 8-member retrieval family
   (§6). Both conditions, together, are what "the bf16 arm is worse on this task" means here.
   - **The 0.03 clause does not bind at n = 24, and that is stated rather than left to be
     discovered.** One flipped pair moves the point estimate by 1/24 = **0.0417**, so *any*
     net loss of one or more pairs already exceeds 0.03 and condition (i) is satisfied whenever
     (ii) can be. The decision at Stage 1's n is therefore carried by the Holm test; the margin
     binds only at an n where 0.03 is resolvable (≥ 34 paired keys), which **no pod in this
     design runs** — Stage 2's conditional pods carry the same n = 24 cells
     (`prereg/gate1_tracker_swap_v2.md` §9). It is kept in the rule because it is the rule, and
     because a later design at a larger n would read it.
   - **What the Holm test can resolve.** The exact two-sided McNemar p at *a* discordant pairs one
     way and *b* the other is `2·P(Bin(a+b, ½) ≤ min(a, b))`; at b = 0 that is `2^(1−a)`:
     (8,0) = 0.0078, (9,0) = 0.0039, (10,0) = 0.00195. Holm's first slot in an 8-member family is
     0.05/8 = **0.00625**. So **a retrieval member separates on its own only at (≥ 9, 0)** — nine
     of twenty-four pairs lost with none won; (8,0) at 0.0078 clears the **third** slot
     (0.05/6 = 0.00833) at best — it misses the second, 0.05/7 = 0.00714 — i.e. only behind two
     other separations. This is the same arithmetic
     `prereg/gate1_tracker_swap_v2.md` §6 does at m = 16 and reaches (≥ 10, 0), and it is stated
     here for the same reason: **a table of non-separations is the resolution this design bought,
     not evidence of equivalence.** Non-inferiority on the retrieval axis is therefore a weak
     claim by construction, and §5 says why it is expected to be uninformative in fact.
2. **Perplexity, per family.** The **TOST at ±0.02 bits/token** on the 32 paired per-window
   bits/token values (`nll_sum_nats / (ntok · ln 2)` from `pplw.jsonl`, never the pooled `ppl=`
   number), read at **α = 0.05, uncorrected**, exactly as `prereg/gate1_tracker_swap_v2.md` §4 and
   §6 read theirs: the claim is a conjunction over its component tests, so it is an
   **intersection–union test** whose size is at most α however many components it has, and a
   multiplicity correction would buy nothing and cost power. The bf16 arm is **non-inferior on
   perplexity in that family iff `stats.tost(d, 0.02)[2]` is `True`.**
   - The shipped `tost` is **two-sided equivalence**, so passing it is strictly stronger than
     one-sided non-inferiority: it also rules out the bf16 arm being *better* by more than 0.02.
     The stronger reading is the one registered, because it is the one the shipped tool and
     `make gate1`'s table compute. The one-sided non-inferiority p-value is the `p_lo` of
     `stats.tost` (H₀: mean d ≤ −0.02, tested `greater`) and is **reported beside the boolean**;
     no rule below reads it.
3. **Overall.** The arm **passes** iff, at 16K, it is non-inferior **on all four tasks in both
   families** (8 retrieval members) **and** both families' perplexity TOSTs pass (2 members). Any
   other outcome is a **fail**, and the report names which member failed and on which axis. There
   is no partial pass and no per-family pass: the consequence below moves bytes for the
   whole project, so it is bought on both families or not at all.
4. **Refusals, read before the rule above is applied.**
   - **An `error` row on the bf16 arm or on `isvd_r64_h256_seed` in a cell refuses the reading for
     that family** — the member is listed with its exception text and no adjusted p-value, and the
     verdict for that family is `REFUSED (error)`. An error on the fp32 arm additionally refuses
     the whole Gate-1 verdict for that family under `prereg/gate1_tracker_swap_v2.md` §4, which is
     not this file's business but is why the two files cannot disagree about it. Nothing is re-run
     on the pod with a knob changed; a fix is a later commit and a later pod.
   - **`mcnemar_exact` returning `None`** (no shared `(seed, trial)` key) is a broken pairing, not
     a result: the member is listed with its keys and no adjusted p-value, and the reading is
     refused for that family. A `prompt_sha256` mismatch does not remove a member; it shrinks that
     member's paired n, the drop is reported with the key and the arms, and §4 (1)'s resolution
     note is re-read at the smaller n. With no discordance at all (a = b = 0 — the likely shape of
     a floor-against-floor cell) `mcnemar_exact` returns **p = 1.0**, which enters Holm as an
     ordinary p-value and is **never** read as evidence of equivalence.
   - **A missing or partial cell refuses it too.** The reading needs all four retrieval cells at
     n = 24 and the 32-window sweep on both arms in that pod; a pod that stopped inside arm 6
     leaves `not run`, never a partial reading (§9). `scripts/pod.py check` fails such a pod
     independently.
   - **An `OrthonormalityError` on this arm** — the guard could not restore the basis *after* a
     repair (the post-repair meaning fixed by `prereg/hygiene_table4.md` A2.3) — is an `error` row
     and is handled by the first refusal. Given §2 (b) it is a possible outcome, not an accident,
     and it is the single most informative failure this arm can produce: it would mean the bf16
     store rounds the basis further out than a thin QR can pull it back, which is the mechanism
     this arm rests on.

### The consequence, pre-registered

- **If it passes.** `docs/plan/DECISIONS.md` records the reading and, with it, that **every
  byte-matched comparison in this project is re-planned at the new bytes — never silently re-run
  at the old ones.** The no-gist twin matched to the bf16 arm's stored bits at 16K is
  **H = 1355** on a 1024-wide layer and **H = 2389** on a 512-wide one, re-solved on
  `stored_bits()` at the commit that adds this file:

  ```python
  from kvdlra.accounting import bug_footprint
  ref = bug_footprint(1024, 64, 16092, 32, n_sink=4, retention="lowrank_surprise",
                      hh_count=256, gist_bits=16).stored_bits()          # 45,664,000
  base = bug_footprint(1024, 1, 1, 32, n_sink=4, retention="lowrank_surprise").stored_bits()
  per  = bug_footprint(1024, 1, 1, 32, n_sink=4, retention="lowrank_surprise",
                       hh_count=1).stored_bits() - base                  # 1,245,376 + 32,800*H
  H = (ref - base) / per     # 1354.2263 -> 1355, a 1.000556x match
  ```

  and, at n = 512, `(39,831,296 − 622,784) / 16,416 = 2388.43 → 2389`, a 1.000234× match. These are
  the same two numbers `prereg/gate1_tracker_swap_v2.md` §3 names as "the fp16-equivalent solve on
  the same inputs gives 1355 and 2389", and §2 (a) says why they coincide exactly rather than
  approximately. Both solve the twin with its own rank-1 gist still fp32 at rest, which is how
  `nogist_h2423` / `nogist_h4460` are billed; a twin that also stored bf16 would need H = 1356 /
  2390, and which convention a future arm file takes is that file's to state.
  **The fp32 arm stays the default** — `isvd_r64_h256_seed.yaml` is not edited by this reading and
  `gist_dtype` stays default-off — until a re-planned comparison at the new bytes is actually run.
  A pass licenses the re-plan; it does not retroactively re-label any existing table's memory
  column.
- **If it fails.** The fp32 arm stays the paper's arm, and the bf16 row is reported as **the
  measured cost of halving the gist's at-rest bytes** — the failing member, its effect size, its
  interval and the axis it failed on, printed beside the 0.150× → 0.085× / 0.275× → 0.148× the
  halving buys. A failure is a result, not a defect: it is the first measurement of what the
  storage dtype costs end to end, and §2 (c) predicts where it would come from.

## 5. Predictions, written before the run

Written from §2, with **no row of this arm on any hardware** (§2 opening). Retrieval is hits/24 per
cell; perplexity is bits/token, and Δ is `fp32 − bf16`.

| axis | prediction | how it was reached |
| --- | --- | --- |
| retrieval, both families | **non-inferior on all four tasks, and uninformative** | `prereg/gate1_tracker_swap_v2.md` §2 (a) and §5 put the fp32 r64 arm at or below **0.25 / 0.08 / 0.00** on single / multikey / multivalue on generator v2 (≈ 6 / 2 / 0 of 24), `vt` unknown; D-005's closing rows are the basis. Two arms at a floor produce few discordant pairs, and §4 (1) needs (≥ 9, 0) to separate. **A non-inferior retrieval table here is the resolution of a floor, and the report says so rather than claiming equivalence.** |
| perplexity, both families | **TOST at ±0.02 passes** (|Δ| < 0.02 bits/token) | three reasons, none of them a transfer function (below) |
| the direction, if there is one | **Δ < 0** — the bf16 arm at or fractionally above the fp32 arm's bits/token | the coordinates are a strictly coarser store of the same subspace; nothing in the construction makes bf16 better |
| 32K (not run in Stage 1) | **the named risk** | ≈ 1,600 absorbs ⇒ drift ≈ 6.3e-2 (§2 c–d), 1.4× the 16K figure, and the accumulation is √absorbs with nothing repairing it. If the arm ever joins a 32K pod (§3), this is the cell that could fail. |

**How the perplexity prediction was reached, and what it is not.** It is **not** derived from a
measured reconstruction-to-perplexity transfer: no such transfer exists in this repository, and
the drift number is a CPU synthetic-stream rate (§2 d). The three things it does rest on:
(i) at a 16K prefill the bf16 store perturbs the stored gist by ≈ 4.5e-2 *relative*, while the
r64 gist's own stored-representation error against the true KV is already **0.1524 on keys and
0.5448 on values** (`prereg/gate1_tracker_swap_v2.md` §2 (c), `results/recon_1b/recon.jsonl` at
rank 64) — a small perturbation of an already-coarse approximation, not a new failure mode;
(ii) the basis error does not compound (§2 c), so nothing here has the shape of the divergences
that moved perplexity by whole bits (the Table-4 unguarded Qwen arm at **+10.77** bits/token with
paired SD 0.61 — `prereg/gate1_tracker_swap_v2.md` §2 b); (iii) the closest *measured* analogue on
this exact protocol — one cache, one substantive knob, no divergence — is Llama
`isvd_r256_noguard` − `isvd_r256_tol` at **+0.0016** bits/token (SD 0.0019) and
`isvd_r256_tol` − `isvd_r256_qr64` at **−0.0006** (SD 0.0016), i.e. thousandths of a bit at n = 16
windows. **The magnitude is therefore not predicted; the direction and the decidability are.** If
the realised |Δ| lands between 0.005 and 0.02 the TOST still passes and the prediction stands; a
|Δ| above 0.02 refutes it and is the interesting result.

**The resident-vs-stored expectation, stated before the pod reports it.** `_cast_gist(torch.float32)`
materializes a **full fp32 working copy of U, B and C on every absorb**, so the bf16 arm's peak GPU
allocation will **not** fall by the `16 bits × (2nr + 2r·coord_count)` per layer that
`stored_bits()` does. The saving this file measures is **at-rest bytes only** — precisely the
storage-vs-resident objection the Week-18 panel raised — and a *resident* saving needs the kernel
(lane L4, `prereg/kernel_smoke.md`, ADR 0001, D-002), which is the only thing that removes the
reconstruction from the decode path. **And the expectation is not testable on these pods at all**,
which is named here rather than left as an implied promise: `frontier.run_ppl` does wrap each arm
in `accounting.measure_peak_gpu` (which resets `torch.cuda.max_memory_allocated` per arm, so the
quantity is a genuine per-arm peak), but `runner._ppl_record` does not carry `peak_gpu_bytes` into
`ppl.jsonl` and `frontier._log_row` does not print it, so it dies with the process and **Stage 1's
records and log contain no resident number for any arm**. Measuring it is a record change, and a
record change is lane L4's or a later lane's, not this file's. **No memory claim of any kind is
made from these pods beyond the analytic at-rest bill of §2 (a) and its verification in §7 (c).**

## 6. Family size and correction

Two families, fixed now, both **this file's own** — no member enters a Gate-1 family and no Gate-1
p-value moves (`prereg/gate1_tracker_swap_v2.md` §6).

- **Retrieval — 8 members.** `isvd` vs `bf16` on each of `niah_single`, `niah_multikey`,
  `niah_multivalue`, `vt`, on Llama and on Qwen. 4 × 2 = 8. Holm at α = 0.05
  (`kvdlra.eval.stats.holm`) over the 8 raw p-values.
- **Perplexity — 2 TOSTs.** One per family, at ±0.02 bits/token, **α = 0.05 uncorrected**, for the
  intersection–union reason §4 (2) gives.

**Realised m.** A member whose cell holds an `error` row **leaves the family** (the rest are
corrected together at the smaller m, so they stay decidable) and is listed with its exception —
but under §4 (4) an error on either arm refuses that family's reading outright, so this path only
ever shrinks the family across the *other* family's members. **A task excluded by the pre-flight's
ceiling rule** (`full` < 0.9 — `prereg/gate1_preflight.md` §4 (ii), `vt` being the task it is
written for) leaves this family too, by the same amendment that shrinks the Gate-1 families:
**8 − 2 per excluded task** (1 contrast × 2 families), so one exclusion gives 6 and two give 4.
The perplexity family has no task axis and is unaffected. The realised m is printed beside the
family, and the resolution note of §4 (1) is re-read at it: at m = 6 the first slot is 0.00833 and
**(8, 0) becomes self-sufficient**; at m = 4 it is 0.0125 and (8,0) clears with room.

**Stage 2's families are fixed now, so that running it later decides nothing this file has not
already set** (the discipline of `prereg/gate1_tracker_swap_v2.md` §6). If the conditional Mistral
16K pod runs, it carries this arm (§3) and brings **one model family's worth: retrieval 4**
(4 tasks × 1 contrast) **and perplexity 1**, Holm inside its own retrieval family and its TOST
uncorrected like the others. **No Stage-2 member is pooled with a Stage-1 member and no Stage-1
p-value is recomputed when Stage 2 lands**: §4 (3)'s pass is read over Stage 1's two families and
stands as read, and Mistral is a third family's reading recorded beside it by the amendment that
runs it. A 32K pod, if an amendment ever puts this arm in one, carries the same 4 / 1 sizes and is
**descriptive** — it reports whether the 16K reading generalizes to twice the context and to
≈ 1,600 absorbs, and it does not retract a Stage-1 pass.

**The 32-window TOST, and its decidability condition.** `stats.tost` is two one-sided *t*-tests on
the paired per-window differences, so equivalence at ±δ fires when `t(1−α, n−1)·s/√n < δ − |d̄|`.
At n = 32, α = 0.05, `t(0.95, 31) = 1.6955`, the half-width is `0.2997·s` and **a ±0.02 TOST is
decidable at d̄ ≈ 0 only for s < 0.0667** — the table and the arithmetic are
`prereg/gate1_tracker_swap_v2.md` §6, and the 32 windows exist because of it. The measured
arm-vs-arm paired SD on this protocol spans **0.0010–0.0563** bits/token over 15 contrasts on the
three Table-4 pods (that file's §2 b), and the two Llama contrasts closest in kind to this one —
two arms differing only in a repair schedule, no divergence — sit at **0.0016–0.0019**, a 35×
margin. The residual is pre-registered with its reporting obligation, in the same words:

- `s` is read back from the run — the paired bootstrap CI's half-width is ≈ 1.96·s/√n, so
  `s ≈ (hi − lo)·√n / 3.92`, and both bounds are printed in `make gate1`'s perplexity table.
- **If a member's realised `s` exceeds its bound, its TOST cannot fire whatever the point estimate
  is**, and the member is recorded as **`not decidable`** — never as a pass, never as a quiet fail.
  The reading is then `REFUSED (not decidable)` for that family and the report says which it is:
  |d̄| genuinely above the margin, or the interval too wide at the realised spread. Only the second
  is a reason to buy more windows; neither is a reason to widen the margin. **A wide interval can
  never manufacture a non-inferiority pass** — which is the asymmetry that makes ±0.02 the
  conservative choice for this reading as well as for Gate 1's.

Everything else in this file is **descriptive** and carries no corrected p-value: §7's
diagnostics, the Wilson intervals (`kvdlra.eval.stats.wilson`) printed beside each cell, the
`ratio` / `sbits` rows, and the bf16 arm's own cells as rows in the Gate-1 table.

## 7. Secondary outcomes

- **(a) The guard did its job — a pass/fail check, from `diag.jsonl`.** Per layer, over this arm's
  rows: the share of 64-absorb windows with `fixed_k` / `fixed_v` **true**, the maximum
  **pre-repair** `orth_err_k` / `orth_err_v`, and the abort count. **Pass = `fixed_k` and
  `fixed_v` true in ≥ 99 % of this arm's windows on every layer, and zero aborts.** §2 (b)
  predicts essentially 100 %: a `[diag]` row ORs the window's repairs, the guard fires on every
  absorb but the first, so every window of 64 absorbs carries at least one. **Below that share is
  the finding** — it would mean the store is not being rounded where this file says it is.
  - **The share alone cannot distinguish this arm from the fp32 one, and that is stated now.** At
    the pods' bf16 *model* dtype the fp32 arm already repairs in essentially every window:
    `results/filler_realism/diag.jsonl`, 19,968 rows of `bugSseed-r64-h256`, reads `fixed_k`
    **100.0 %**, `fixed_v` **99.2 %**, with pre-repair `orth_err_k` min 1.00e-3 / median 1.02e-3 /
    max 1.41e-3 and `orth_err_v` max 8.14e-3 (`prereg/gate1_tracker_swap_v2.md` §7 c). The
    **discriminating** quantity is therefore the pre-repair *magnitude*: this arm's per-window
    maximum should sit at the bf16 rounding scale of §2 (b) (a few × 1e-3 on **both** streams, the
    error the previous absorb's rounding propagated through the step) rather than just above the
    1e-3 tolerance that gates the fp32 arm's repairs — and both remain two decades below the 1e-1
    abort. Reported as the two arms' distributions side by side, from one pod; **descriptive, no
    p-value.**
- **(b) `eff_rank`.** `eff_rank_k` / `eff_rank_v` per layer from `diag.jsonl` (and the `eff_rank=`
  field `frontier._log_row` appends to the `ppl=` line). Both arms run `min_sv_frac: 0.0`, so both
  should sit at the cap 64 on every layer and every window; a bf16 arm that tracks a lower
  effective rank than the fp32 arm on the same stream is an accounting finding (it would bill a
  rank it does not carry — D-015, audit finding 0.2) and is reported as one.
- **(c) The measured stored-bits ratio against the analytic one.** `ppl.jsonl` carries one row per
  (arm, ctx) whose **`sbits` field is `Footprint.ratio_stored_bits(t, n)`** as
  `frontier.run_ppl` computed it from the **live** layer — the live tracked rank, the live
  coordinate count, the live ring, and `gist_bits=16` read off `layer.gist_dtype`
  (`src/kvdlra/eval/frontier.py`, `runner._ppl_record`). **This is where the stored-bits ratio is
  read; `trials.jsonl` rows carry no such field** (`kvdlra.eval.records.TrialRecord`), and
  `kvdlra.eval.gate1.load` reads `sbits` from `ppl.jsonl` for exactly this reason.
  - **The check, with its tolerance and its two conventions.** The analytic value at the arm-doc
    convention (ring 32) is **0.0851** (Llama) / **0.1484** (Qwen); at the ring's high water
    (`recent_window + absorb_block − 1` = 47, coordinate columns 16,077) it is **0.0859** /
    **0.1492** — the 0.5–1.0 % spread between them is the ring's occupancy at the moment the
    footprint is taken, not an accounting defect, and the higher pair is the one ADR 0001's cost
    model pins as "0.151× / 0.086× at 16K r64". Because of that spread the load-bearing pin is the
    **ratio of the two arms on the same pod**: `median(sbits of isvd_r64_h256_seed_bf16) /
    median(sbits of isvd_r64_h256_seed)` must lie within **1 %** of **0.566** (Llama, n = 1024) /
    **0.540** (Qwen, n = 512) — the occupancy nearly cancels there (0.5657 at ring 32 against
    0.5684 at ring 47). Outside it, the arm did not store what this file says it stores, and the
    reading is **refused** pending a dated amendment. Read **before** §4's rule, never after.
  - `prereg/gate1_tracker_swap_v2.md` §7 (f) expects arm 6 to print `sbits` "below them by its
    coordinate dtype"; this is that expectation with the number in it.
- **(d) The arm's measured rate.** `manifest.cell_elapsed_s` carries this arm's four retrieval
  cells (`[stage] cell arm=… task=… ctx=… elapsed_s=… n=…`); its min/sample is
  `Σ(four values) ÷ 5,760`, against the 3.1 min/sample §9 budgets it at. Descriptive, and it is
  what re-sizes the arm if it ever joins a Stage-2 pod.
- **(e) No resident-memory secondary.** For the reason §5 gives: the per-arm peak is measured
  in-process but is carried into no record and printed on no line these pods write, so there is
  nothing to read. The resident question belongs to `prereg/kernel_smoke.md`.

## 8. Log volume — this arm's share

Identical to one gist arm of `prereg/gate1_tracker_swap_v2.md` §8, whose measured anchor is **416
`[diag]` rows per 16K sample** on Llama-3.1-8B (13 rows per layer × 32 layers at the cache default
`diag_every` = 64, 804–810 absorbs per layer; `results/filler_realism/diag.jsonl`, 39,936 rows over
48 samples × 2 gist arms) and **364** on Qwen2.5-7B's 28 layers. Per pod, for this arm alone:

| rows | Llama | Qwen |
| --- | --- | --- |
| `[diag]`: 128 samples × 416 / 364 | **53,248** (≤ 69,632 at the 17-rows-per-layer bound) | **46,592** (≤ 60,928) |
| `[trial]`: 4 tasks × 24 | 96 | 96 |
| cell rows + `[stage] cell` (§7 d): 4 + 4 | 8 | 8 |
| `[pplw]` parts + `ppl=`: 4 + 1 | 5 | 5 |
| **this arm's share of `<label>.log`** | **≈ 53,360 lines** (≈ 14.7 MB at 275 B/row) | **≈ 46,700 lines** (≈ 12.8 MB) |

Nothing here changes the pod-level arithmetic or the watchdog's clock: §8 of the Gate-1 file
already counts seven gist arms including this one (372,736 / 326,144 `[diag]` rows per pod), and
its poll-gap argument is bounded by the *fastest* gist arm (`nogist_*` at 93 s/sample), not by this
one at 3.1 min. **Completeness for this arm, exactly:** 96 `[trial]` lines, four `[pplw]` parts
(`_log_pplw` splits a 32-value line into ⌈32/8⌉ = 4 `part=i/n` groups, and
`records.parse_pplw_lines` raises on a gap), 32 rows in `pplw.jsonl`, one row in `ppl.jsonl`
carrying `sbits`, and `manifest.diag_skipped` = 0.

## 9. Budget — no pod of its own

This arm is **arm 6 of Stage 1**, billed inside `prereg/gate1_tracker_swap_v2.md` §9 at its
measured-rate table: **3.1 min/sample × 128 samples = 396.8 min = 6.6 h of compute per pod**,
**13.2 GPU-h across the two pods** at the point estimate — 26.5 GPU-h as this arm's share of the
pods' 2× bar, which is set per pod and not per arm — at the
$0.45–0.74/h of `prereg/gate1_preflight.md` §7, **$6–10 expected and $12–20 at the bar**. That is
already inside Stage 1's 82 GPU-h point / 164 GPU-h bar ($37–61 / $74–121); **this file asks for no
additional GPU time and creates no pod.** The 3.1 rate is `isvd`'s own measured rate, applied here
because the arm does the same one core factorization per absorb per stream; at the observed
bracket's top (3.7 min/sample) the arm costs 7.9 h per pod instead of 6.6.

**Where it sits in the ordered loss.** §9's cumulative column puts the arm at **28.9 h** of
compute — after the two primary contrasts have landed (15.7 h) and after the C branch is decidable
(22.3 h), and before the two secondaries. So a pod that dies anywhere before 28.9 h of compute has
already returned everything Gate 1's rule reads and **nothing of this file's** — which is the
point of the order, and the reason the reading below can be void without Gate 1's being.

**The cut ladder does not reach it.** §9's two pre-committed rungs drop `random_r64_h256_seed`
(rung 1) and then `oja_tuned` (rung 2) — arms 8 and 7, both after this one — so the bf16 arm
survives both cuts, and no third rung is pre-registered. If a deeper cut is ever ordered, arm 6 is
the next in reverse order and **this file's entire reading is void**: it is recorded as `not run`
with the rung that cut it, never as a partial reading. The same applies to a pod that stops inside
arm 6: a partial arm is `not run` for the missing cells and the family is refused (§4 (4)), and
`scripts/pod.py check` fails such a pod anyway.

## 10. Provenance

- **Pod and arm files, none of them created or edited by this commit.** Pods:
  `configs/pods/gate1_v2_stage1_llama.yaml`, `configs/pods/gate1_v2_stage1_qwen.yaml` (committed by
  lane L3 Task 3 at 4bcfdb5, both listing `isvd_r64_h256_seed_bf16` sixth).
  Arms: `configs/arms/isvd_r64_h256_seed_bf16.yaml` (L5.1, 63dfa80; its `doc:` amended at
  a27058d) and `configs/arms/isvd_r64_h256_seed.yaml`. Tasks:
  `configs/tasks/ruler_v2_16k_g1.yaml`, `configs/tasks/ppl_16k_pg19val.yaml`. The arm stem is
  pinned by `tests/test_config_parity.py`'s `POST_V1` set, its **position 6** in both Stage-1 pods
  by `tests/test_pod_manifest.py`'s `GATE1_STAGE1_ARMS` (and its exclusion from the smoke pod by
  that file's `GATE1` set); the drift generator by
  `tests/test_bf16_gist.py::test_bf16_gist_drift_shape_and_direction`; the default-off fp32 path by
  `tests/test_golden_cache.py`. **The commit that adds this file adds nothing else and launches
  nothing.**
- **The commit order, and the two mechanisms that enforce it — stated as
  `prereg/gate1_tracker_swap_v2.md` §10 states them, because the situation is that file's second
  case.** For the Stage-1 pods the *launching* pod's prereg is
  `prereg/gate1_tracker_swap_v2.md`, and `scripts/pod.py launch` refuses outright unless **that**
  file's first commit is a strict ancestor of the launch commit (`prereg_error` covers missing,
  uncommitted, not-a-strict-ancestor and committed-by-the-launch-itself, plus a dirty tree and an
  unpushed SHA). `pod.py launch` checks **only** the launching pod's own prereg, so **this file's
  precedence over the Stage-1 launch commit is a lane rule, not a machine refusal** — evidenced
  after the fact with

  ```
  git merge-base --is-ancestor <this file's first commit> <Stage-1 launch SHA>
  ```

  and recorded, with both SHAs, in the Stage-1 launch entry in `docs/plan/DECISIONS.md`.
  `scripts/pod.py check` re-checks the launching pod's order against the manifest's `git_sha` at
  harvest. **The refusals themselves are tested, not assumed:**
  `tests/test_pod_manifest.py::test_prereg_refusal_reasons` (missing, uncommitted),
  `::test_prereg_commit_order_against_real_history` (strict ancestor, not-an-ancestor, and
  committed-by-the-commit-itself, on commits this repository actually has),
  `::test_launch_refuses_a_pod_with_no_prereg` (a `prereg: null` pod is refused before `vastai` is
  reached) and `::test_check_rejects_tampered_config_hash` (the manifest side).
- **Amendments only, never edits.** §1–§11 are not edited after the Stage-1 launch commit. Every
  later change is a dated Amendment appended below, in the pattern of
  `prereg/hygiene_table4.md` — including the Stage-2 / 32K amendment §3 names and any re-statement
  of §5 that the pre-flight's rows justify — each committed before the launch commit it governs.
- **Outputs.** `results/gate1_v2_stage1_llama/` and `results/gate1_v2_stage1_qwen/`: this arm's
  96 rows in `trials.jsonl` (each with `prompt_sha256`, `haystack_id`, `depth`, `code_family`,
  `ratio`, `sbits`), its 32 rows in `pplw.jsonl`, its one row in `ppl.jsonl` (the `sbits` §7 (c)
  reads), its `[diag]` rows in `diag.jsonl`, and its four `cell_elapsed_s` entries in
  `manifest.json`.
- **Rendering, and citability.** The bf16 arm appears as the `bf16` row of `make gate1`'s
  per-(family, ctx) retrieval table and of its perplexity table
  (`scripts/tables.py gate1 --pods results/gate1_v2_stage1_llama results/gate1_v2_stage1_qwen
  --out docs/paper/tables/gate1.md`), where the perplexity row already carries this file's
  primary statistic: bits/token, Δ = isvd − bf16, the 95 % paired bootstrap CI, and the ±0.02
  **TOST passing/failing**. The 8 retrieval members of §4 are produced by §4's snippet and are
  reported in the DECISIONS entry beside that table; they are not rendered by `make gate1`, which
  computes no bf16 retrieval contrast. A number here is citable only once
  `scripts/pod.py check` passes on the pod directory (config hash, commit order, every retrieval
  cell at n = 24, `env.txt` at the pyproject pins) and the table regenerates from the committed
  records (CLAUDE.md).
- **The verdict** — pass or fail, the failing member if any, the two families' adjusted p-values
  and TOST booleans, the refusal if one fired, the `sbits` check of §7 (c), and (on a pass) the
  re-planning consequence of §4 with H = 1355 / 2389 — goes to `docs/plan/DECISIONS.md` with the
  evidence path `results/gate1_v2_stage1_{llama,qwen}/` and the table path. `GATES.md` §G5 line 1
  is ticked against that entry; line 2 ("all seven prereg files committed before their pods;
  SHA-order check passes") and line 3 ("tampered-manifest rejection test passes") are ticked
  against the four tests named above. **The "seven" in line 2 predates this file**: with it and
  `prereg/kernel_smoke.md` the repository holds **eight** pre-registrations beside
  `prereg/README.md` (`filler_realism`, `gate1_preflight`, `gate1_tracker_swap_v2`,
  `hygiene_table4`, `l2_smoke`, `ss2_families`, `bf16_gist`, `kernel_smoke`). The line's substance
  is the order check, which is what is ticked; the count is for whoever updates `GATES.md`.

## 11. What this does not decide

Whether fp16 storage would do the same (`GIST_DTYPES` deliberately excludes it: it would store at
16 bits and bill at 32); quantized coordinates, which is a different mechanism with its own arm
and its own gate (`prereg/ss2_families.md`, D-005's branch-3 amendment); anything at 32K or on
Mistral-7B-v0.3 before the amendment §3 names, and in particular whether the √absorbs drift of
§2 (c) bites at 1,600 absorbs; the default of `gist_dtype`, which stays off until a re-planned
byte-matched comparison has actually run (§4); any resident-memory, throughput or latency claim,
which needs the kernel (`prereg/kernel_smoke.md`, ADR 0001, D-002) and which §5 explains cannot
come from these pods at all; the guard's tolerances or `min_sv_frac`, whose defaults are Table-4's
(D-011 addendum 8); and the Gate-1 branch itself, which this arm is outside of by
`prereg/gate1_tracker_swap_v2.md` §4–§6 and which no member of this file can move. This file asks
one question — does storing the gist at 16 bits cost anything measurable at 16K on two model
families — and one arm of two pods answers that one.

**STATUS: rides the Gate-1 Stage-1 pods (DECISIONS D-003; launch under D-011).**
