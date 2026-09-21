# Pre-registration — `gate1_v2_stage1_llama` + `gate1_v2_stage1_qwen` (Gate 1, Stage 1)

**STATUS: awaiting owner go (DECISIONS D-003 top-up; launch under D-011).** Written before
either pod is launched, and before the Gate-1 pre-flight (`prereg/gate1_preflight.md`) has run —
so no row of generator v2 on hardware exists anywhere when §1–§11 below are committed. Launch is
a separate, later commit that spends money and is recorded in `docs/plan/DECISIONS.md`:
`scripts/pod.py launch` refuses unless this file's first commit is a *strict* ancestor of the
launch commit, so the commit that adds this file launches nothing. The authorization for the
launch itself is D-011's standing one (2026-09-17); the money is not there yet — Stage 1's bar is
$74–121 against a credit of $92.52 with two other pre-registered pods already queued (§9), so
**D-003 precedes the Stage-1 launch commit**.

Two pods, one pre-registration: `PodCfg` carries a single `model`, so the Llama and Qwen halves
of one experiment are two configs (the pattern of `prereg/ss2_families.md`). Lane L3; gate G3.

**Amendments this file has already promised, before anything is measured.**
`prereg/gate1_preflight.md` §1, §4 (ii) and §4 (iv) commit the pre-flight's three returns to this
file as *dated amendments*, each committed before the **Stage-1** launch commit and each visible
as a reaction to a body that was fixed first:

1. **Amendment 1** quotes the pre-flight's harvested per-task rows — `full`,
   `isvd_r64_h256_seed`, `nogist_h2423`, `frozen_r64_h256_seed` on `ruler_v2_16k`, n = 12 — into
   §2 as this file's measured generator-v2 baseline, restating (never replacing) the predictions
   of §5.
2. If a task's `full` ceiling on the pre-flight is **< 0.9**, that task is **excluded from the
   primary Holm family** by the same amendment, which re-states the family size and the per-cell
   resolution at the smaller family (§6 gives both arithmetics now: 16 − 4 per excluded task on
   retrieval, 24 − 6 on the secondary family, the perplexity family untouched).
3. If a pre-flight **budget trigger** fires (§9), the §9 table is **re-derived from the measured
   `manifest.cell_elapsed_s` rates** by the same amendment, and each pod YAML's `gpu_budget_h`
   moves in the commit that precedes the launch. The pods are never launched over their
   pre-registered bar.
4. And the precondition those three rest on (§10): Stage 1 launches only if the pre-flight's
   readings **(i) completeness** and **(iii) pairing** passed — otherwise a dated amendment names
   the failure, its repair and the commit carrying it, before the Stage-1 launch commit.

---

## 1. Purpose

Gate 1 (`docs/plan/ICML2027_PLAN.md` §2 Gate 1 item 1.1, `docs/plan/lanes/GATES.md` §G3,
`docs/plan/lanes/L3_gate1_tracker_swap_v2.md`) asks one question: **does the online-tracked gist
do work that a frozen basis, a random basis, or no gist at all does not?** It is the gate that
decides which paper is written — `ICML2027_PLAN.md` §0: "Gate 1 below decides between A/B and C.
Do not write a word of the new paper before Gate 1 closes."

Everything in the cache is held fixed — sinks 4, recent ring 32, `absorb_block` 16, a
surprise-selected exact tier, the warm-up seed, the shipped orthonormality guard, the
singular-value floor off — and only the gist's tracker is swapped:

| plan label | this design's arm | what it removes from the r64 configuration |
| --- | --- | --- |
| (a) | `isvd_r64_h256_seed` | nothing — the reference |
| (b) | `oja_r64_h256_seed_tuned` | the incremental-SVD step, for Oja's rule at the tuned schedule |
| (c) | `fd_r64_h256_seed` | the incremental-SVD truncation, for Frequent-Directions shrinkage |
| (d) | `frozen_r64_h256_seed` | the **online tracking** (learn on the first 4096 tokens, then freeze) |
| (e) | `nogist_h2423` / `nogist_h4460` | the **gist**, its bytes given to the exact tier |
| (f) | `random_r64_h256_seed` | any data in the basis at all (the floor control) |

**Stage 1 is the two 16K pods this file sizes and reads: Llama-3.1-8B and Qwen2.5-7B, eight arms
each, the four Gate-1 tasks at n = 24, plus 32 paired perplexity windows.** Stage 2 — Mistral
16K and all three families at 32K — is sized here as conditional (§9) and its pods are created by
amendment only if Stage 1 does not select Branch C.

Nothing else is decided here (§11). In particular this file makes no memory, throughput or
baseline-comparison claim, says nothing about 2-bit quantization (that is Gate 2,
`prereg/ss2_families.md`) or the kernel (Gate 3), and does not ask whether the r64 configuration
has a real-text retrieval niche at some *other* rank or tier budget — D-005's closing entry left
that question open and it stays open.

## 2. Measured baseline — what is known, and what it predicts

Every number in this section was computed from the committed records by the snippet beside it, at
the commit that adds this file — none is copied from a table or from memory. **No `frozen`, no
`random` and no `nogist` row exists on hardware anywhere in this repository**, and no arm of any
kind has been scored on generator v2 on a GPU; (f) below says so with the check that establishes
it, and the pre-flight pod (`prereg/gate1_preflight.md`, `results/gate1_preflight/` once
harvested) supplies the first generator-v2 rows — quoted here by **Amendment 1** before the
Stage-1 launch commit, never as a result of this experiment.

### (a) The retrieval rows that closed D-005 — the r64 configuration on real text

`results/filler_realism/` (instance 51553610, launch SHA 56e89ee, `ALL_DONE`, 240 records,
0 errors, harvested 97fd76e; D-005 CLOSED 2026-09-19). The **v1 in-house** four tasks at 16K on
WikiText-2 sentences — not generator v2 — hits/12:

```python
import json, collections, pathlib
rows = [json.loads(l) for l in pathlib.Path("results/filler_realism/trials.jsonl").read_text().splitlines()]
cells = collections.defaultdict(list)
for r in rows:
    cells[(r["arm"], r["task"])].append(r)
for (arm, task), rs in sorted(cells.items()):
    print(arm, task, f"{sum(r['hit'] for r in rs)}/{len(rs)}", "errors", sum(1 for r in rs if r["error"]))
```

| arm (record key) | single | multikey | multivalue | vt | errors |
| --- | --- | --- | --- | --- | --- |
| `full` | 12/12 | 12/12 | 11/12 (0.92) | **1/12 (0.08)** | 0 |
| `bugSseed-r64-h256` (= `isvd_r64_h256_seed`) | 3/12 (**0.25**) | 1/12 (**0.08**) | **0/12** | 0/12 | 0 |
| `bugSseed-r64-h256-q4` | 0/12 | 0/12 | 0/12 | 0/12 | 0 |
| `quant-2bit-kivi` | 12/12 | 6/12 (0.50) | 6/12 (0.50) | 1/12 (0.08) | 0 |
| `quant-2bit-kivi#chunk0` | 12/12 | 10/12 (0.83) | 5/12 (0.42) | 2/12 (0.17) | 0 |

The cycled control, `results/filler_realism_cycle/` (51559661, a5cd89c, 96 records, 0 errors,
harvested 4ded444; D-005 addendum 3), same snippet: `bugSseed-r64-h256` **12/12 / 12/12 / 12/12 /
8/12 (0.67)** and `full` 12/12 on all four. Every task within 0.25 of the archived `w18-g1-llama`
row (1.00 / 1.00 / 1.00 / 0.58) → `prereg/filler_realism.md` A1.3: the L0 runner reproduces
`w10_ruler.py`, and harness drift is excluded as an explanation for anything below.

**What it predicts for Stage 1.** Generator v2's haystacks are real documents from four sources
under official RULER templates — harder filler than WikiText-2 sentences by construction — so the
r64 arm is expected **at or below 0.25 / 0.08 / 0.00** on single / multikey / multivalue
(`prereg/l2_smoke.md` §2 and §4 reading 4 pre-register exactly this for every gist arm on this
task file; "a 0/12 cell is a *pass*" there). At n = 24 that is roughly 6 / 2 / 0 of 24. §5 writes
the consequence: the *retrieval* axis of Gate 1 is being run near a floor, and a floor is where
paired contrasts have the least to separate.

### (b) The perplexity protocol Stage 1 reuses, and what an incremental-SVD gist costs on it

`ppl_16k_pg19val` — PG-19 validation, 16K prefill under the arm's compression, a frozen
2048-token scoring window, **32 non-overlapping** windows of the 160 PG-19 validation supplies at
this span (D-013; `configs/tasks/ppl_16k_pg19val.yaml`, unchanged and unedited). The only rows on
this protocol are the Table-4 harvests (`results/hygiene_table4_{llama,qwen_r128,qwen_r256}/`,
harvested 287dc45, D-011 addendum 8), which ran its **16**-window variant `ppl_16k_pg19val_w16` —
identical corpus, context, window and non-overlap, the cut a budget decision of that lane
(`prereg/hygiene_table4.md` Amendment 1) — on arms that differ from the Gate-1 configuration:
rank 128/256, coordinate tier over the whole prefill, **no exact tier**. They are an anchor for the
*protocol* and for the per-window spread, not for the r64 arm, and every SD below is measured at
n = 16:

```python
import json, collections, pathlib, math, statistics
from kvdlra.eval.stats import paired_bootstrap, tost
rows = [json.loads(l) for l in pathlib.Path("results/hygiene_table4_llama/pplw.jsonl").read_text().splitlines()]
per = collections.defaultdict(dict)
for r in rows:
    per[r["arm"]][r["window_idx"]] = r["nll_sum_nats"] / r["ntok"] / math.log(2)
for a, v in sorted(per.items()):
    d = [v[k] - per["full"][k] for k in sorted(v)]
    print(a, round(statistics.fmean(v.values()), 4), paired_bootstrap(d), round(statistics.stdev(d), 4))
```

| pod (model) | arm | bits/token | Δ vs `full`, paired [95 % bootstrap] | SD of the paired per-window Δ |
| --- | --- | --- | --- | --- |
| `hygiene_table4_llama` (Llama-3.1-8B) | `full` | 3.4003 | — | — |
| | `isvd_r256_tol` | 3.4200 | +0.0196 [+0.0153, +0.0239] | 0.0091 |
| | `isvd_r256_qr64` | 3.4206 | +0.0203 [+0.0158, +0.0246] | 0.0093 |
| | `isvd_r256_noguard` | 3.4216 | +0.0212 [+0.0169, +0.0254] | 0.0090 |
| `hygiene_table4_qwen_r256` (Qwen2.5-7B) | `full` | 3.3238 | — | — |
| | `isvd_r256_tol` | 3.4943 | +0.1705 [+0.1511, +0.1966] | 0.0488 |
| | `isvd_r256_f0.01_tol` | 3.3429 | +0.0191 [+0.0147, +0.0237] | 0.0096 |
| `hygiene_table4_qwen_r128` (Qwen2.5-7B; its own `full` is 3.3238, the same corpus and windows) | `isvd_r128_tol` | 3.4075 | +0.0838 [+0.0667, +0.1074] | 0.0435 |
| | `isvd_r128_f0.01_tol` | 3.3617 | +0.0379 [+0.0315, +0.0448] | 0.0140 |

**Two things this fixes for §4 and §6.** First, a guarded incremental-SVD gist at 16K sits
+0.02 bits/token from `full` on Llama and +0.08…+0.17 on Qwen: the effect sizes Gate 1's
perplexity axis works in are hundredths to tenths of a bit, which is the scale the ±0.02
equivalence margin was written at. Second — and this is the number that decides whether the C
branch is *reachable* — the **spread** of the paired per-window difference, the quantity a TOST is
bounded by. Two arms differing in one substantive knob, same pod, same windows:

```python
import itertools
from kvdlra.eval.stats import paired_bootstrap          # `per` as above, one pod at a time
for a, b in itertools.combinations(sorted(per), 2):     # the arm-vs-arm paired difference
    d = [per[a][k] - per[b][k] for k in sorted(set(per[a]) & set(per[b]))]
    print(a, b, paired_bootstrap(d), round(statistics.stdev(d), 4))
```

| arm-vs-arm contrast (same pod, 16 paired windows) | mean Δ | SD of the paired Δ |
| --- | --- | --- |
| Qwen `isvd_r256_tol` − `isvd_r256_f0.01_tol` (the floor: a real change of subspace) | +0.1514 | **0.0490** |
| Qwen `isvd_r128_tol` − `isvd_r128_f0.01_tol` | +0.0458 | **0.0464** |
| Qwen `isvd_r256_tol` − `isvd_r256_qr64` (the repair schedule only) | −0.0029 | 0.0165 |
| Llama `isvd_r256_tol` − `isvd_r256_qr64` | −0.0006 | 0.0016 |
| Llama `isvd_r256_noguard` − `isvd_r256_tol` | +0.0016 | 0.0019 |
| Qwen `isvd_r256_f0.01_qr64` − `isvd_r256_qr64` (population maximum, `hygiene_table4_qwen_r256`) | −0.1546 | **0.0563** |

So the measured range of arm-vs-arm paired SD on this protocol — enumerated over every
compression-arm-vs-compression-arm contrast within that population (every guard-on arm, plus
Llama's `isvd_r256_noguard`) across the three Table-4 pods, 15 contrasts in all (3 on
`hygiene_table4_llama`, 6 each on `hygiene_table4_qwen_r128` and `hygiene_table4_qwen_r256`) — is
**0.0010 to 0.0563 bits/token**; the table above keeps the five contrasts worth naming for their
substantive meaning and adds the population maximum. The two unguarded arms that **did** diverge
sit two to three orders of magnitude above it, from the same snippet on the same pods: Qwen
`isvd_r256_noguard` − `isvd_r256_tol` has SD **0.6132** at a mean of +10.77 bits/token, and Qwen
`isvd_r128_noguard` − `isvd_r128_tol` SD **1.1934** at +0.6043. Neither bounds anything here —
**no Stage-1 arm runs unguarded** (§3 fixes the shipped guard at `orth_fix_tol` 1e-3 /
`orth_abort_tol` 1e-1 on every arm), and a basis that diverged past the abort tolerance would raise
`OrthonormalityError`, which is an `error` row and §4's refusal rule, not a wide TOST. It is the
*Qwen guard-on* end that bounds the C branch, and it is the reason §9 buys **32** windows: §6
computes that a ±0.02 TOST at n = 32 is decidable at d̄ = 0 for SD < **0.0667**, so the population
maximum, **0.0563** (`isvd_r256_f0.01_qr64` − `isvd_r256_qr64` on `hygiene_table4_qwen_r256`), sits
inside the bound with a ≈ 1.2× margin (0.0667 / 0.0563 = 1.18), where at 16 windows (bound 0.0456)
it would have been outside and undecidable. No branch flips: the C branch was, and remains,
reachable on this population; only the stated range and margin move. §6 pre-registers the residual
as a condition with a stated failure mode, not as an assumption.

### (c) The tracker ordering, measured — reconstruction on the 1B dumps at exactly r = 64

`results/recon_1b/recon.jsonl` (provenance `results/recon_1b/provenance.json`, git SHA
410c06f, 12,800 rows, dump manifest `dumps/llama3.2-1b.sha256`
`b53487ba…`): the stored-representation reconstruction error (`kvdlra.eval.recon`, the §4.1 metric
CLAUDE.md requires) of every tracker this pod swaps, on Llama-3.2-1B KV dumps — 5 documents of
4096 tokens × 16 layers, pre-RoPE keys and values separately, **rank 64, block 16, floor off,
25 % prefill** (the same fraction as `freeze_after: 4096` at 16K). `stored_rank` is 64 for every
method here except `fd2`, which stores 128:

```python
import json, collections, statistics, pathlib
rows = [json.loads(l) for l in pathlib.Path("results/recon_1b/recon.jsonl").read_text().splitlines()]
sel = [r for r in rows if r["rank"] == 64 and r["block"] == 16 and r["min_sv_frac"] == 0.0]
idx = collections.defaultdict(dict)
for r in sel:
    idx[r["method"]][(r["doc"], r["layer"], r["kv"])] = r["err"]
for m in sorted(idx):
    for kv in ("k_pre", "v"):
        ks = [k for k in idx[m] if k[2] == kv]
        print(m, kv, round(statistics.fmean(idx[m][k] for k in ks), 4),
              sum(1 for k in ks if idx[m][k] > idx["isvd"][k]), "/", len(ks))
```

| method (`stored_rank`) | keys: mean err | worse than `isvd` on | mean err ratio | values: mean err | worse on | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| `svd_oracle` (64) | 0.1487 | 0/80 | 0.976 | 0.5295 | 0/80 | 0.972 |
| **`isvd`** (64) | **0.1524** | — | 1.000 | **0.5448** | — | 1.000 |
| `frozen_prefill_svd` (64) | 0.1640 | **80/80** | **1.076** | 0.5716 | **80/80** | **1.050** |
| `fd`, ℓ = r (64) | 0.1948 | 80/80 | 1.277 | 0.6926 | 80/80 | 1.273 |
| `fd2`, ℓ = 2r (**128**) | 0.1312 | 0/80 | 0.859 | 0.5229 | 13/80 | 0.954 |
| `oja`, tuned (64) | 0.2381 | 80/80 | 1.572 | 0.6150 | 80/80 | 1.131 |
| `random_basis` (64) | 0.9349 | 80/80 | 6.288 | 0.9356 | 80/80 | 1.734 |

**What this fixes.** (i) The ordering is unanimous and it is not close between groups: incremental
SVD is within **2.5–2.9 %** of the Eckart–Young floor (the table's ratios are *other ÷ isvd*, so
the distance above the floor is 1/0.976 and 1/0.972, not 1 − 0.976); freezing costs
5.0–7.6 %; FD at ℓ = r costs 27 %; Oja at its tuned schedule costs 13–57 %; a random basis costs
73–529 %. (ii) The
`isvd`-vs-`frozen` gap — the primary contrast — is **uniform in sign but small in size**: frozen
loses on all 80 (document, layer) cells on both streams, by 5–8 %. A task-level separation
between them requires a task sensitive to a 5–8 % reconstruction gap, which is exactly why the
Gate-1 rule reads retrieval **or** perplexity and not retrieval alone. (iii) `fd2` beats `isvd`
only because it stores twice the columns; at matched bytes (ℓ = r) FD is 27 % worse. That is the
whole reason §3's FD arm is ℓ = r and not the plan's ℓ = 2r.
(iv) The Oja rows here run the **per-stream** tuned schedules from `provenance.json` — keys
(η₀ 5.0, decay 0.3), values (5.0, 0.1) — while `configs/arms/oja_r64_h256_seed_tuned.yaml`
carries a single pair (5.0, 0.3) applied to both streams (§3); the 1.131 value ratio is therefore
a *lower bound* on the Gate-1 Oja arm's value-side error. Restricting to the three documents that
were not used for tuning changes almost nothing (keys 1.540, values 1.124), so the schedule is
not overfit to its two tuning documents.

**What it does not fix.** This is a **1B** model on 4096-token documents, the tracker in
isolation — no exact tier, no sinks, no ring, no seed — and reconstruction error is not a task
metric. The 8B half of this sweep is open (`GATES.md` §G1 line 6, "1B half (8B open — dump
pod)"). Nothing here is a Gate-1 reading; it is the prior §5 writes its ordering from.

### (d) The exact tier, not the tracker, is what retrieved in v1 — and it is cited as a mechanism, not a result

Under D-005 every v1 in-house retrieval table is **diagnostic only** (the cycled generator is
retired from every headline claim), so the rows below are cited for the mechanism they identify
and for nothing else; the n per cell is 2–4. `docs/plan/reports/w11-final-tables.md` at 32K:
`bugS-r128-h256` (rank-128 gist **with** a 256-token exact tier) 100 / 75 / 100 / 75 against
`bug-r128` (the same rank, **no** exact tier) **0 / 50 / 0 / 0**, measured on all four; and
`bugSdrop-r128-h1024` — the arm that *stores* the tier but withholds it from attention — reads
— / 50 / 0 / 0, where the dash is a cell that table never scored. So the two tier-less conditions
agree on the three tasks both were scored on (50 / 0 / 0) and the withholding arm simply has no
needle row; nothing here rests on a dash. The **visible** exact tier is the retrieval mechanism;
the gist's rank is not.

**What it predicts, and it is the sharpest prediction in this file.** The no-gist control is
byte-matched by *enlarging the tier*: `nogist_h2423` holds **2423** verbatim tokens against the
r64 arm's 256 — **9.5×** as many — and `nogist_h4460` holds 4460, **17.4×** as many (§3). If
retrieval is carried by the visible exact tier, then at matched stored bits the no-gist control is
expected to **beat** `isvd_r64_h256_seed` on retrieval, not to lose to it. §5 writes that as the
prediction and §4 states what it would and would not license.

### (e) The one existing perplexity comparison between a gist and a tier-only arm, and why it settles nothing

`configs/arms/evict_surprise_h256.yaml` (record key `bugEVICT-h256`) is the same construction as
the no-gist arms — `rank: 1`, `coord_budget: 1`, surprise tier — at `hh_budget: 256` instead of
2423. Its v1 rows (`docs/plan/reports/w11-final-tables.md`, 16K; the column labelled *perplexity*
is bits/token, `full` 4.08 there against 3.4003 on PG-19 validation above; the corpus is
WikiText-103 **train**, which CLAUDE.md retires, and the memory column is the fp16-equivalent
`ratio`, not stored bits):

| arm | ratio | bits/token |
| --- | --- | --- |
| `bugEVICT-h256` (tier + ring, no gist) | 0.018× | 4.44 |
| `bugS-r32-h256` (rank-32 gist, same tier) | 0.053× | 4.48 |
| `bugEVICT-h1024` | 0.065× | 4.36 |
| `bugS-r32-h1024` (rank-32 gist, same tier) | 0.098× | 4.30 |

The comparison **splits both ways and is byte-matched in neither direction**: at h = 256 the
tier-only arm is 0.04 bits *better* at 0.34× the bytes; at h = 1024 it is 0.06 bits *worse* at
0.66× the bytes. So "the gist carries fluency" is a mechanism claim with a mixed, non-byte-matched
prior and no measurement at the Gate-1 configuration. **A byte-matched gist-vs-no-gist perplexity
comparison has never been made in this repository. It is the single most informative cell Stage 1
buys**, and it is why a perplexity-only A/B is a live outcome (§5).

### (f) No `frozen`, no `random`, no `nogist` row exists — the check

Both control trackers were created on this branch at 7f3e546 (L3 Task 1) and have only CPU tests
behind them (`tests/test_gate1_arms.py`); no tracker other than `isvd`, `oja` and (aborted) `fd`
has ever run on a GPU here:

```python
import json, pathlib
arms = {json.loads(l)["arm"] for p in pathlib.Path("results").rglob("trials.jsonl")
        for l in p.read_text().splitlines()}
print(len(arms), sorted(a for a in arms if "fro" in a or "nog" in a or "oja" in a or "rand" in a))
# 22 ['bugSseed-r64-h256-oja']   (the Week-20 swap pod's void Oja cell; CLAUDE.md settled facts)
```

The Week-20 swap pod is not evidence for anything here: its Oja arm ran untuned defaults and its
FD arm crashed (CLAUDE.md settled facts; D-012, D-014), and the "tracker load-bearing" sentence in
`paper/main.tex` that rested on it is unsupported. **Stage 1 is the first measurement of the
frozen, no-gist and random arms anywhere in this project, and the first measurement of any arm on
generator v2 beyond the pre-flight's four.**

## 3. Arms, tasks, n

Two pods, one arm list, in this order — the uncompressed ceiling first (cheapest), then the
paired reference, then the two **primary-contrast** controls, then the arm the C branch needs,
then the bf16 arm, then the two secondaries the cut ladder of §9 drops in this order. The order
buys an *ordered loss*: a pod that dies early still lands the readings in the order they matter.

**This order supersedes the lane brief's, and the reason is the ordered loss.** The brief lists the
trackers as `isvd`, `oja`, `fd`, `frozen`, `random`, `nogist` with the bf16 arm as "a 7th arm"
(`docs/plan/plans/2026-09-11-L3-L5-gate1-bf16-prereg.md`, Task L3.2 "Pod sizing") — an inventory,
not a run order: it would spend the two arms with no prior anywhere (`oja`, `random`) before the two
the decision rule reads (`frozen`, `nogist`), so a pod that died at its bar could return neither
primary contrast nor a decidable C branch. Under the order above both primary contrasts have landed
at 15.7 h of compute and the C branch is decidable at 22.3 h (§9), and the bf16 arm — the one arm
whose reading is not in this file — is paid for only after that. **The Stage-1 pod YAMLs L3 Task 3
commits carry this order**, and `tests/test_pod_manifest.py` pins it arm for arm.

| # | arm config | `kind` | tracker | record key | the cache, in one line |
| --- | --- | --- | --- | --- | --- |
| 1 | `full` | `full` | — | `full` | uncompressed; single-shot prefill |
| 2 | `isvd_r64_h256_seed` | `bug` | `isvd` | `bugSseed-r64-h256` | rank 64, 256-token surprise tier, 4 sinks, ring 32, warm-up seed |
| 3 | `nogist_h2423` (Llama) / `nogist_h4460` (Qwen) | `bug` | `isvd` | `nogist_h2423` / `nogist_h4460` | rank **1**, `coord_budget` **1**, tier **2423** / **4460**, same sinks / ring / seed |
| 4 | `frozen_r64_h256_seed` | `bug` | `frozen` | `frozen_r64_h256_seed` | arm 2 verbatim with `freeze_after: 4096` |
| 5 | `fd_r64_h256_seed` | `bug` | `fd` | `bugSseed-r64-h256-fd` | arm 2 verbatim, Frequent-Directions shrinkage at **ℓ = r = 64** |
| 6 | `isvd_r64_h256_seed_bf16` | `bug` | `isvd` | fixed by L5.1's arm file | arm 2 with bf16 gist storage; **its reading is pre-registered separately** |
| 7 | `oja_r64_h256_seed_tuned` | `bug` | `oja` | `oja_r64_h256_seed_tuned` | arm 2 verbatim, Oja's rule at η₀ = 5.0, decay = 0.3 |
| 8 | `random_r64_h256_seed` | `bug` | `random` | `random_r64_h256_seed` | arm 2 verbatim, a fixed Haar-random basis |

Arms 3–8 are each arm 2 with **one mechanism removed** — arm 3 moves three knobs (`rank` 64 → 1,
`coord_budget` null → 1, `hh_budget` 256 → 2423 / 4460), which is what removing the gist while
keeping its bytes costs, and arm 4 moves two (`tracker`, `freeze_after`). Everything else — rank,
tier size, sinks, ring, seed, `absorb_block`, `min_sv_frac: 0.0`, the shipped guard at
`orth_fix_tol` 1e-3 / `orth_abort_tol` 1e-1 — is identical across arms 2, 4–8, which is what makes
these cells *mechanism* comparisons rather than budget comparisons.

- **`frozen_r64_h256_seed`** (`configs/arms/frozen_r64_h256_seed.yaml`) — incremental SVD over the
  first **4096 tokens** of the stream, measured on the cache's *monotone* tokens-seen frontier
  (`BugStreamingLayer._tokens_seen`; a frontier read off tier occupancy would stop at the budget
  and a long context would never freeze); past it `frozen_step` returns `U` and the core untouched
  behind `rot = I`, so every later token is a plain projection `Uᵀ·block` onto the warm-up basis.
  That is xKV / ShadowKV-style static low rank. At 16K the basis is learned on the first **25 %**
  of the context; at 32K (Stage 2) the same 4096 is **12.5 %**, so the frozen control is weaker at
  32K *by construction* — a stated property of the design, not a defect, and the reason a 32K
  separation from frozen is weaker evidence than a 16K one.
- **`nogist_h2423` / `nogist_h4460`** — tier + ring only, with the gist's bytes given to the tier.
  `H` is solved on **stored** bits (`kvdlra.accounting.Footprint.stored_bits`, the at-rest billing
  CLAUDE.md §4.1 requires) at t = 16384 against arm 2. Llama-3.1-8B and Mistral-7B store 8 KV
  heads × 128 = **1024** channels per layer: arm 2 bills **80,717,568** bits/layer and this arm
  bills `1,245,376 + 32,800·H`, so **H = 79,472,192 / 32,800 = 2422.93 → 2423**, a **1.00003×**
  match. Qwen2.5-7B stores 4 × 128 = **512**: arm 2 bills 73,836,288 and the twin bills
  `622,784 + 16,416·H` → **H = 4459.89 → 4460**, a **1.00003×** match. One H cannot serve both
  widths, which is why there are two files. The arithmetic is each arm file's `doc:` and is pinned
  within 5 % by `tests/test_gate1_arms.py::test_the_nogist_arms_are_byte_matched_to_isvd_r64_at_16k`.
  The tier still selects by residual, but against a rank-1 basis — approximately norm-ordering,
  which is what "no gist" means for a surprise tier. Both H values are far above the plan's
  "h ≈ 1024", which was an fp16-equivalent estimate; the fp16-equivalent solve on the same inputs
  gives 1355 and 2389.
- **`random_r64_h256_seed`** — one seeded Haar draw per **(layer, stream)**: a CPU fp32 Gaussian's
  reduced QR at `basis_seed + 2·layer_idx` (`+1` for V), cast to the block dtype, never updated,
  core = identity. Reproducible from the arm config alone, and K and V never share a basis.
- **`fd_r64_h256_seed`** — the shared rank-revealing augmentation and core factorization of the
  incremental-SVD step, with FD's shrinkage as the only algorithmic difference, at **ℓ = r = 64**.
  `ICML2027_PLAN.md` §2 item 1.1 names ℓ = 2r; **this design deviates** and says why: ℓ = 2r
  stores a 128-column sketch, twice the bytes, and §2 (c) measures what that buys (`fd2` beats
  `isvd` at ℓ = 2r and loses by 27 % at ℓ = r). A byte-matched swap is the only swap this gate can
  read. ℓ = 2r stays in the reconstruction study, where the plan also puts it (item 1.2).
- **`oja_r64_h256_seed_tuned`** — the schedule tuned by `kvdlra.eval.recon.tune_oja` on the **1B**
  dumps (**tuned on** doc63 + doc718 at r = 16, layer 8 — `results/recon_1b/provenance.json`
  `oja_tuning.docs`, the same two documents for both streams; the three documents §2 (c)'s
  sensitivity check restricts to are the held-out ones), where the
  surface falls away from the Week-2 (20.0, 0.03) point: the arm's `doc:` records 0.8115 for
  (20, 0.03) against 0.3991 tuned on `k_pre` r16 doc63, and D-014 records 0.821 against 0.405 for
  the same comparison (two write-ups of one measurement; the arm file is what the config hash
  covers). **The plan's second Oja arm — "a re-tuned schedule on held-out 8B KV" — is not in this
  design**: no 8B dump exists (`GATES.md` §G1 line 6), and the arm carries one (η₀, decay) pair
  for both streams while the tuning found (5.0, 0.3) for keys and (5.0, 0.1) for values. Both
  deviations are named in the verdict entry; neither is repaired on the pod.
- **`isvd_r64_h256_seed_bf16`** rides Stage 1 beside arm 2 on identical prompts and windows. **Its
  non-inferiority reading is pre-registered separately** in `prereg/bf16_gist.md` (lane L3
  Task 4b), committed before the Stage-1 launch commit; no reading of it appears in §4–§6 here,
  and its bytes differ from arms 2–5, 7, 8, so it is not a member of any byte-matched contrast in
  this file.

| pod | model | dtype / image | tasks | cells |
| --- | --- | --- | --- | --- |
| `gate1_v2_stage1_llama` | `unsloth/Meta-Llama-3.1-8B-Instruct` | bfloat16, `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` | `ruler_v2_16k_g1`, `ppl_16k_pg19val` | 8 × 4 = **32** retrieval + 8 perplexity sweeps |
| `gate1_v2_stage1_qwen` | `Qwen/Qwen2.5-7B-Instruct` | same | same | same |

**Tasks and n.** `configs/tasks/ruler_v2_16k_g1.yaml`: `generator: v2`, `ctx: 16384`, `seeds: [0]`,
`design: {haystacks: 2, depths: 3, codes: 4}` → **`n_trials: 24`**, real-document haystacks from
`pg19`, `arxiv`, `wikipedia`, `essays` (materialized by `pod.py run` before the model loads,
digested into `manifest.dataset_sha256`), official RULER needle / question / answer-prefix
templates, `vt`'s value taking the cell's code family (ruling R-L2-4), chunked prefill at 4096.
**The shipped file still lists five sub-tasks**; the Gate-1 family is the four that
`GATES.md` §G3 and `ICML2027_PLAN.md` §2 item 1.1 name — `niah_single`, `niah_multikey`,
`niah_multivalue`, `vt` — and **lane L3 Task 3 edits the task file down to those four to match
this section**, in the commit that adds the pod YAMLs and before the launch commit;
`scripts/pod.py check` and `tests/test_pod_manifest.py` pin the match (the task list, the arm
order, n = 24, the prereg path and a distinct `config_hash`, every arm through
`frontier.build_arm` at t = 16384 as the runner builds it). `niah_multiquery` is not a Gate-1
task and no contrast in §4–§6 reads it. **24 records per (arm, task) cell, 96 retrieval samples +
32 perplexity windows = 128 samples per arm, 1,024 per pod**; a trial that raises is a record with
`error` set and `hit = 0`, counted in n (ruling R29).

`configs/tasks/ppl_16k_pg19val.yaml` is unchanged and unedited: **32** non-overlapping windows, of
the 160 PG-19 validation supplies at this span (D-013). Non-overlap is what keeps the paired
bootstrap's independence; 32 rather than the Table-4 pods' 16 is what makes a ±0.02 TOST decidable
at the spread those pods measured (§2 b, §6), and the windows are budgeted as samples in §9 rather
than absorbed into overhead.

**The 32K task files Stage 2 would run** — `configs/tasks/ruler_v2_32k_g1.yaml`, committed by L3
Task 3 before the Stage-1 launch commit and written to match this section, and
`configs/tasks/ppl_32k_pg19val.yaml`, which exists unchanged (32 non-overlapping windows of the 84
PG-19 validation supplies at that span, D-013) — **run nothing in Stage 1** (§9).

**Pairing.** `gen.make_trial` is deterministic in `(task, seed, trial)` and never sees the arm, so
all eight arms are fed the same token ids for a given key; the runner's `[trial]` line and record
carry `prompt_sha256` over exactly those ids (post-L2.3b), so the pairing is **verified** from the
records (§7 (e)), not assumed. Whether an arm prefilled the ids in one shot (`full`) or in
4096-token chunks (arms 2–8) does not enter the digest. The perplexity windows are cut identically
from one corpus at one context length inside a single pod, so every paired statistic here is
within-pod; **no contrast in this file crosses a pod boundary.**

## 4. Primary contrasts and decision rule

### The rule, verbatim from the plan

> **Pre-registered readings: if (a) is not separated from (c)/(d) on any task and perplexity is
> within 0.02 bits/token, the tracker is not the contribution → Branch C. If (a) beats (d) and
> (e) on retrieval or perplexity in ≥2 families with Holm-corrected p<0.05 → the online-tracked
> gist is doing work → Branch A/B. Budget ~40 GPU-h.**
>
> — `docs/plan/ICML2027_PLAN.md` §2, Gate 1, item 1.1

with (a) = `isvd_r64_h256_seed`, (c) = `fd_r64_h256_seed`, (d) = `frozen_r64_h256_seed`,
(e) = `nogist_h2423` / `nogist_h4460` (§1). The budget sentence is superseded by §9, which is
built on rates the plan did not have.

### Primary contrasts, written exactly

**Retrieval.** Per family and per task, `isvd_r64_h256_seed` vs `frozen_r64_h256_seed` and
`isvd_r64_h256_seed` vs the family's `nogist_*` arm: the **exact paired McNemar** over the 24
shared `(seed, trial)` keys of that cell — `kvdlra.eval.stats.mcnemar_exact`, the r64 arm as *a*,
the two cells' Bernoulli outcomes keyed by `(seed, trial)`, the two-sided exact binomial on the
discordant pairs — on **byte-identical prompts** (§3's pairing invariant, verified from
`prompt_sha256` per §7 (e); a key whose digests disagree is dropped from that member and the drop
is reported with the key). **16 members**: 2 contrasts × 2 families × 4 tasks (§6).

**Perplexity.** Per family, the same two contrasts on the **32 paired per-window** bits/token
values (`nll_sum_nats / (ntok · ln 2)` from `pplw.jsonl`, never the pooled `ppl=` number): a
two-sided paired *t*-test on the per-window differences as the Holm member, the 95 % paired
bootstrap CI reported beside it, and a **TOST at ±0.02 bits/token** whose boolean is read by
**both** branches — it gates the A/B perplexity route as well as the C branch (the rule below).
**4 members**: 2 contrasts × 2 families (§6).

### The statistic, as code

`kvdlra.eval.gate1` is committed by lane L3 Task 3 to match this section, before the Stage-1
launch commit; it adds no statistics of its own — every statistic below is a shipped
`kvdlra.eval.stats` function, already tested in `tests/test_stats.py`, plus the
`scipy.stats.ttest_1samp` that `stats.tost` itself calls:

```python
from kvdlra.eval import gate1

retr = gate1.retrieval_contrasts(trials)  # {(family, task, a, b): Contrast}
#   Contrast: n_paired, a_favored, b_favored, p_value (= stats.mcnemar_exact), p_holm,
#             errors_a, errors_b  -- the per-arm error counts the refusal rule reads
ppl = gate1.ppl_contrasts(pplw)           # {(family, a, b): PplContrast}
#   PplContrast: mean_d, lo, hi = stats.paired_bootstrap(d)
#                p_paired       = scipy.stats.ttest_1samp(d, 0).pvalue  -> Holm inside its family
#                tost_pass      = stats.tost(d, 0.02)[2]   -- the BOOLEAN at alpha = 0.05,
#                                 uncorrected (§6). The two one-sided p-values are reported
#                                 beside it; no rule below reads them.
verdict, members = gate1.gate1_verdict(retr, ppl, diag=diag, sbits=sbits)
#   "A/B" | "C" | "UNDECIDED", + the members that decided it. `diag` is diag.jsonl and `sbits`
#   the records' sbits column: the two refusals below are decided from the records, not by eye.
```

Holm is `kvdlra.eval.stats.holm` at α = 0.05, applied over each family's raw p-values as §6
defines them; `scripts/tables.py` applies no correction of its own.

### The operationalization, exactly as `gate1_verdict()` implements it

**Scope: the verdict reads the 16K contrasts only.** Stage 1's two 16K pods — and, if Stage 2 runs,
its Mistral 16K pod, which enters this same rule with the Holm families §6 fixes now — are the
inputs to rules 1–4. The three 32K pods are **descriptive**: they report whether whatever 16K found
generalizes to twice the context, with Holm applied inside each of their own families (§6), and
they **never change the branch**. A 32K family that separates where its 16K twin did not, or fails
to where it did, is written into the verdict entry as that — a generalization reading — and moves
no letter.

1. A **family is separated** iff, for **each** of `frozen` and `nogist` separately, the r64 arm
   beats it — **on retrieval on at least one task** (that member's Holm-adjusted p < 0.05 in the
   primary retrieval family **and** `a_favored > b_favored`) **or on perplexity**, where the
   perplexity route requires **all three** of: that member's Holm-adjusted p < 0.05 in the primary
   perplexity family, **Δ < 0** (the r64 arm's mean bits/token lower), and that contrast's ±0.02
   **TOST failing** (`stats.tost(d, 0.02)[2]` is `False`). The third condition is the ±0.02 margin
   doing its work in both directions: an advantage that is statistically real *and* demonstrably
   inside 0.02 bits/token is, by the practical-significance criterion this gate adopted, not the
   tracker doing work — it is equivalence measured tightly, and it counts for neither branch. Both
   controls must be beaten; beating one is not a separated family.
2. **Branch A/B** iff **≥ 2 families are separated**, counting **model families separated at
   16K**: Llama and Qwen from Stage 1, Mistral from Stage 2 if it runs. Stage 1 has exactly two, so
   at Stage 1 "≥ 2" means **both**. A single separated family cannot reach A/B on Stage 1 alone —
   that is precisely the outcome Stage 2's Mistral pod exists to resolve (§9): it enters rule 1
   unchanged, at its own families (§6), and a separation there is the second. One family separated
   is recorded as `UNDECIDED (one family separated)`, never rounded up, and no 32K pod can supply
   the missing one.
3. **Branch C** iff **both** of its conditions hold. **(i) Retrieval:** no Holm-significant
   separation of the r64 arm from `fd` and none from `frozen`, on **any task in any 16K family** —
   no Holm-adjusted p < 0.05 in either direction on any of those members, the `fd` members read
   from the secondary family of §6 at that family's **realised** m. **(ii) Perplexity:** **every**
   `isvd`-vs-`fd` and `isvd`-vs-`frozen` TOST at ±0.02 bits/token **passes** (4 TOSTs at Stage 1:
   2 contrasts × 2 families; the two `frozen` TOSTs come from the primary perplexity family, the
   two `fd` TOSTs from the same `gate1.ppl_contrasts` call on the same 32 windows, reported
   descriptively in §7 (a) and read here). A Holm-significant perplexity *t*-test against `frozen`
   or `fd` does not by itself block C: what blocks C is a TOST **failing**, and (ii) is that
   condition stated positively. Because C requires *all* of its TOSTs to pass, its perplexity
   condition is an **intersection–union test**, so each TOST is read at α = 0.05 **uncorrected**
   (§6 states why that controls the error rate with no multiplicity correction, while the paired
   *t*-tests keep Holm).
4. Otherwise **UNDECIDED**, with the members that blocked each branch listed.

**The plan's asymmetry is kept, and named.** Branch C reads (c) and (d) — FD and frozen — and not
(e); so a no-gist control that beats the r64 arm everywhere does **not** by itself block C. It is
nonetheless the strongest single piece of evidence for "the exact tier, not the tracker", and the
DECISIONS verdict entry records it explicitly whenever it happens, beside the branch the rule
selected. The rule is not widened here to absorb it: widening a pre-registered rule after writing
down what the evidence probably is, is the thing pre-registration exists to prevent.

**A/B and C are mutually exclusive by construction, and the ±0.02 margin is what makes them so.**
Every route by which a family can be separated from `frozen` contradicts one of C's `frozen`
conditions: a retrieval win is a Holm-significant separation on some task, which C (i) forbids; a
perplexity win now additionally requires its ±0.02 TOST to **fail**, which C (ii) forbids. No
outcome satisfies both. The third possibility — a Holm-significant perplexity difference whose TOST
*passes* — satisfies neither route and leaves the letter to the other contrasts, and it is not
hypothetical: Llama `isvd_r256_noguard` − `isvd_r256_tol` over the Table-4 windows has
d̄ = **+0.0016** bits/token at s = **0.0019**, a two-sided paired *t*-test **p = 3.8e-3**, and a
±0.02 TOST that **passes** (`stats.tost(d, 0.02)` → p_lo ≈ 6.3e-18, p_hi ≈ 6.8e-17, `True`; §2 b's
snippet, n = 16). Under rule 1 that pair is **not** an A/B perplexity win — two arms that differ
significantly *and* equivalently — so it cannot collide with C. 4 is everything else. Only one
branch is applied; the consistency check when the table is written is **one branch, named, with
its blocking members listed**.

### Refusal, and the `--` rule

- **An `error` row on `isvd_r64_h256_seed`, `frozen_r64_h256_seed`, the family's `nogist_*` arm or
  `fd_r64_h256_seed` refuses a verdict for that family**: `gate1_verdict` returns `UNDECIDED` with
  the arm and the exception text named, and the pod is not cited (a raised trial is a record with
  `error` set and `hit = 0`, counted in n; `scripts/pod.py check` fails the pod — ruling R29, no
  tolerance knob). An error on arms 6–8 removes that arm's secondary members and nothing else.
  Nothing is re-run on the pod with a knob changed; a fix is a later commit and a later pod.
- **No arm is ever printed as `--`** (`docs/plan/lanes/L3_gate1_tracker_swap_v2.md`: "Any arm with
  `> 0` `error` trials is reported as failed, not as `--`"). Every arm in the Gate-1 table carries
  either its cells or the exception text that replaced them; an arm that never ran reads `not run`
  with the reason. A dash in a Gate-1 table is what the Week-20 swap pod's FD cell looked like, and
  it is the reason that pod decides nothing.
- **An `OrthonormalityError` on any gist arm** (the guard could not restore the basis *after* a
  repair — the post-repair meaning fixed by `prereg/hygiene_table4.md` A2.3) is an `error` row and
  is handled by the rule above. It is a possible outcome, not an accident, and the arm is never
  re-run with the tripwire disabled.
- **A frozen arm still repairing after its freeze voids the `isvd`-vs-`frozen` contrast.** Any
  `diag.jsonl` row of `frozen_r64_h256_seed` with `fixed_k` or `fixed_v` **true** at
  `tokens_seen > freeze_after` (4096) — other than the one window per (sample, layer) that
  straddles the freeze, which §7 (c) exempts and says why — is a dispatch defect (§7 c): the arm
  did not run the mechanism this file says it runs, so its cells are not the contrast that was
  pre-registered. The verdict is **refused** for that family — `UNDECIDED (frozen dispatch)`, with
  the offending rows named — **pending a dated amendment that names the defect and its repair**.
  The frozen cells are not cited, and nothing is re-read around them.
- **A no-gist arm off its byte match voids the `isvd`-vs-`nogist` contrast.** The **measured**
  stored-bits ratio of each `nogist_*` arm to `isvd_r64_h256_seed`, read from the records' `sbits`
  exactly as §7 (f) describes, must lie within **1 ± 0.05** — the tolerance
  `tests/test_gate1_arms.py::test_the_nogist_arms_are_byte_matched_to_isvd_r64_at_16k` pins the
  *design* at, here required of the *run*. Outside it the arm is not byte-matched on the pod
  whatever its file solves for, the `isvd`-vs-`nogist` members of that family are void, and the
  verdict is refused the same way, pending a dated amendment. It is read **before** the rule above
  is applied, never after.
- **The two `mcnemar_exact` boundaries, fixed now.** With no discordance at all (a = b = 0 — two
  cells agreeing on all 24 keys, the likely shape of a floor-against-floor cell) it returns
  **p = 1.0**, and that 1.0 enters Holm as an ordinary p-value; it is never read as missing and
  never as evidence of equivalence (§6). With **no shared key** it returns **`None`**, which is not
  a result but a broken pairing: that is §6's `prompt_sha256` mismatch rule at its limit — paired
  n = 0, the member listed with its keys and no adjusted p-value, and a primary member at `None`
  refuses the verdict for its family exactly as an `error` row does.

Stage 2, if it runs, carries **its own Holm families** (§6) and its own amendment; no Stage-2
member is pooled with a Stage-1 member and no Stage-1 p-value is recomputed when Stage 2 lands.

## 5. Prediction per arm, written before the run

Written from §2 — the D-005 real-text rows, the reconstruction ordering at r = 64, the Table-4
perplexity anchors — with **no generator-v2 row in hand on any arm** (§2 (f)), and restated, never
replaced, by Amendment 1 when the pre-flight harvests. Retrieval is hits/24 per cell.

| arm | retrieval prediction | perplexity prediction |
| --- | --- | --- |
| `full` | the ceiling. ≥ 0.9 on every task — but a Stage-1 `full` cell below 0.9 **excludes nothing**: only the pre-flight's amendment can shrink a family, and a degraded Stage-1 ceiling is flagged beside its members instead (§6 ii) | the reference every Δ is taken against |
| `isvd_r64_h256_seed` | **at or near the floor**: ≲ 0.25 / 0.08 / 0.00 on single / multikey / multivalue (§2 a, harder filler by construction) → ≈ 6 / 2 / 0 of 24; `vt` unknown on v2 | +0.02…+0.17 bits from `full` (§2 b, at a different rank and with an exact tier — an anchor, not a prediction of the value) |
| `nogist_*` | **plausibly above `isvd`** — the visible exact tier is the retrieval mechanism (§2 d) and this arm holds 9.5× (Llama) / 17.4× (Qwen) as many verbatim tokens at the same stored bits | **worse than `isvd`** — higher bits/token — if the gist carries fluency: the mechanism claim, with a mixed and non-byte-matched prior (§2 e) |
| `frozen_r64_h256_seed` | **≈ `isvd`**: same rank, same tier, same seed; reconstruction says frozen loses 5.0–7.6 % uniformly (§2 c), and a retrieval task at the floor cannot resolve 5–8 % | slightly worse than `isvd`, by the same 5–8 % of a reconstruction gap; this is the cell the perplexity axis exists for |
| `fd_r64_h256_seed` | ≈ `isvd` or worse: 27 % worse reconstruction at matched bytes (§2 c) | worse than `isvd` (higher bits/token). The C branch needs *both* non-separation and a passing TOST here |
| `oja_r64_h256_seed_tuned` | worse than `isvd`: 57 % worse reconstruction on keys, ≥ 13 % on values, and the arm's single schedule is key-tuned (§2 c, §3) | worse than `isvd` (higher bits/token) |
| `random_r64_h256_seed` | the **floor control**. Reconstruction 6.3× worse on keys (§2 c); at 16K this arm should be at or near 0/24 on every task. A random basis that matches `isvd` on a task means the task does not read the gist at all — which is a Gate-1 finding, not a bug | the worst gist arm on the axis — the highest bits/token of the eight |
| `isvd_r64_h256_seed_bf16` | read only by `prereg/bf16_gist.md` | read only by `prereg/bf16_gist.md` |

**The live outcome that must be named now: a perplexity-only A/B.** If the predictions above hold
— `nogist` at or above `isvd` on retrieval because the tier retrieves, `isvd` above `nogist` and
`frozen` on perplexity because the gist reconstructs — **and each of those perplexity advantages is
larger than the ±0.02 margin, i.e. Holm-significant with its own TOST failing** (§4 rule 1; an
advantage inside the margin licenses nothing and selects nothing) — then **both families separate
on the perplexity axis alone and the rule selects A/B while every retrieval cell says the tracker
changes nothing about what is retrieved.** That outcome is legitimate under the rule as the plan
wrote it ("on retrieval **or** perplexity") and it is pre-registered as such, but what it licenses
is narrow and is fixed here, before it happens:

- **It licenses**: "the online-tracked gist reconstructs the context better than a frozen basis
  and better than no gist at matched stored bits, **by more than 0.02 bits/token**, measured as
  teacher-forced perplexity on PG-19 validation at 16K over 32 paired windows, on two model
  families" — with the paired CI, the adjusted p-value and the failed TOST printed beside them.
- **It does not license**: any claim that the tracker improves retrieval, any headline built on
  needle accuracy, or the sentence in `paper/main.tex` that this gate was run to test. A
  perplexity-only A/B is recorded in DECISIONS **as** perplexity-only, with the retrieval table
  printed beside it, and the paper's contribution sentence is written from the axis that
  separated and no other.
- **It changes the paper's shape**, and the verdict entry says so: an A/B whose whole support is a
  fluency axis is Branch A′/B territory (systems-led or method-led), not the retrieval story the
  v1 draft told.

**The alternative outcomes, and what each means for the paper.**

- **Branch C** (`isvd` not separated from `fd` or `frozen` anywhere, every TOST passing) — *the
  tier, not the tracker*: the contribution is the residual-selected exact tier and the cache
  around it, the tracker is an interchangeable component, and `ICML2027_PLAN.md` §3-C is the
  build-out. This is the outcome §2 (c)'s 5–8 % `isvd`-vs-`frozen` gap makes plausible: a gap that
  small may simply not move either axis past what §4 asks of it — ten one-directional discordant
  pairs of 24 on retrieval (§6), or an advantage outside ±0.02 bits/token on 32 paired windows.
- **A/B on both axes**, retrieval included — the strongest outcome, and the least likely given
  §2 (a)'s floor.
- **`nogist` beats `isvd` on retrieval *and* on perplexity** — the sharpest negative available:
  the byte-matched control dominates the method on both axes, the rule returns UNDECIDED or C
  depending on `fd` and `frozen`, and the paper is the analysis paper whatever the branch letter
  says. It is reported in full, with the discordant pairs listed.
- **UNDECIDED** — including the one-family-separated case (rule 2), the two refusals of §4
  (frozen dispatch, byte match), and the case where a TOST is **not decidable** at the realised
  spread even on 32 windows (§6). UNDECIDED is a real outcome with a real cost (Stage 2, §9), not a
  failure to be argued away.

**Abort and error are possible outcomes, not accidents.** An `OrthonormalityError` on a gist arm,
an OOM in `full`'s single-shot 16K prefill, a `LinAlgError` the `_svd_core` fallback does not
catch — each is an `error` row, counted in n, and §4's refusal rule applies. The FD arm is the
named risk: the Week-20 pod crashed in exactly this step, the class of failure is reproduced in
`tests/test_fd_numerics.py`, and the shared rank-revealing augmentation plus the `eigh` fallback
are the repair. If FD raises again, the C branch cannot be read and the verdict is UNDECIDED with
the exception quoted — it is not worked around on the pod.

## 6. Family size and correction

Three families, fixed now (`kvdlra.eval.stats.holm`, α = 0.05, applied over raw p-values):

- **Primary retrieval — 16 members.** `isvd` vs `frozen` and `isvd` vs `nogist`, on each of
  `niah_single`, `niah_multikey`, `niah_multivalue`, `vt`, on Llama and on Qwen. 2 × 4 × 2 = 16.
- **Primary perplexity — 4 members.** The same two contrasts, per family. 2 × 2 = 4.
- **Secondary — 24 members.** `isvd` vs `oja_tuned`, `isvd` vs `fd`, `isvd` vs `random`, on the
  four tasks, on both families. 3 × 4 × 2 = 24. **The C branch reads `fd` from this family**, at
  this family's correction — which is the conservative direction: a larger family makes a
  separation from `fd` harder to reach, and C requires non-separation.

**The secondary family is corrected at its *realised* m.** Its 24 members assume all three
secondary arms run. An arm the §9 cut ladder drops before launch, or one the ordered loss never
reaches, takes **8 members with it** (4 tasks × 2 families): rung 1 (drop `random`) leaves
**m = 16**, rungs 1 + 2 (also `oja_tuned`) leave **m = 8**, which is exactly the `fd` members. The
C branch reads its `fd` members at whatever m actually ran (§4 rule 3 (i)) — and that is the
direction that matters: a smaller family widens Holm's slots, so a separation from `fd` becomes
*easier* to reach and C becomes **harder**, never easier, as the budget shrinks. The realised m is
printed beside the family in the table, and §9's ladder names it at each rung.

**Stage 2's families are fixed now, so that running it later decides nothing this file has not
already set.** If the Mistral 16K pod runs it carries one model family's worth of the Stage-1
sizes — **retrieval 8** (4 tasks × 2 contrasts), **perplexity 2** (2 contrasts), **secondary 12**
(4 tasks × 3 contrasts) — with Holm inside each, and its separation enters §4 rule 2 as the third
model family. Each 32K pod carries the same three sizes, **8 / 2 / 12** per family, Holm inside
each family and **descriptive only** (§4's scope note). No Stage-2 member is pooled with a Stage-1
member and no Stage-1 p-value is recomputed when Stage 2 lands.

A member is defined by its pairing-key set — the 24 `(seed, trial)` keys of one (family, task)
cell, both arms, or the 32 windows of one (family) perplexity cell. **A member whose cell holds an
`error` row leaves its family** (the remaining members are corrected together at the smaller m, so
the others stay decidable) and is listed beside the family with its exception and no adjusted
p-value — but on the four arms §4 names, an error refuses the verdict outright, so this path
applies to arms 6–8 only. A `prompt_sha256` mismatch does not remove a member; it shrinks that
member's paired n, and the resolution note below is re-read at the smaller n.

**A task excluded by the pre-flight's ceiling rule** (`full` < 0.9 —
`prereg/gate1_preflight.md` §4 (ii); `vt` is the task the rule is written for) leaves the primary
retrieval family and the secondary family by Amendment: **16 − 4 per excluded task** (2 contrasts ×
2 families) and **24 − 6** (3 contrasts × 2 families). One exclusion gives 12 and 18; two give 8
and 12. The perplexity family is unaffected — it has no task axis. The excluded task is still run,
still reported descriptively, and is fixed in `kvdlra.eval.gen` before any pod runs it again.

**The arm-drop and task-exclusion shrinks to the secondary family compose by members, not by
subtracting counts.** Both remove **members** from the 24-member secondary family (3 contrasts ×
4 tasks × 2 families), not fixed counts from 24: the realised size is **m = (contrasts running) ×
(tasks in the family) × 2 families**. If, e.g., one secondary arm is dropped by the §9 ladder
(contrasts running 3 → 2) *and* one task is excluded by the ceiling rule (tasks 4 → 3) at the same
time, the family is **not** 24 − 8 − 6 = 10; it is 2 × 3 × 2 = **12**. Each shrink above is correct
in isolation, holding the other axis at its full value — they do not sum when both fire.

**Two things that rule does not cover, fixed here rather than later.** (i) **The pre-flight runs
Llama only** (`prereg/gate1_preflight.md` §9 decides nothing about Qwen), so an exclusion is
decided on a Llama ceiling and applies to the task in **both** families — which is the right
scope, because a ceiling that low is a generator/template defect (the A2.6 hypothesis for v1 `vt`)
and not a property of one checkpoint. The same argument carries the *design* gap between the two
pods: the pre-flight measures that ceiling at **n = 12 over two code families**
(`ruler_v2_16k`, `design: {haystacks: 2, depths: 3, codes: 2}`) and Gate 1 runs **n = 24 over
four** (`ruler_v2_16k_g1`, `codes: 4`), and the rule transfers because the ceiling is a property of
the generator and the template rather than of a code family. The Stage-1 `full` cells then report
that ceiling at four code families — which is where a task the pre-flight passed can still be seen
to sag, and (ii) below says what happens then. (ii) **Qwen's own ceiling is unmeasured until
Stage 1 runs it.** If a Stage-1 `full` cell falls below 0.9 on a task the pre-flight passed, the task is **not**
excluded — choosing members after seeing the data is what the pre-flight exists to avoid — but
every member on that task is flagged in the table and in the verdict entry as resting on a
degraded ceiling, with the `full` cell printed beside it, and a `full` cell at or near 0 makes
its members uninformative in fact whatever their adjusted p-values say. A post-hoc exclusion is
available only as a *later, separately pre-registered* pod.

**The 24-key resolution, stated once.** One flipped pair moves an accuracy by 1/24 = 0.042. The
exact two-sided McNemar p at *a* discordant pairs in the r64 arm's favour and *b* against is
`2·P(Bin(a+b, ½) ≤ min(a, b))`; at b = 0 that is `2^(1−a)`: (8,0) = 0.0078, (9,0) = 0.0039,
(10,0) = 0.00195, (11,0) = 0.00098, (12,0) = 0.00049; off the axis, (11,1) = 0.0063,
(12,1) = 0.0034, (13,1) = 0.0018. Holm's slots in a 16-member family, in order, are 0.05/16 =
0.00313, 0.00333, 0.00357, 0.00385, 0.00417, 0.00455, 0.00500, 0.00556, 0.00625, 0.00714, 0.00833,
0.01000, 0.01250, 0.01667, 0.02500, 0.05000. So:

> **A retrieval member separates on its own — needing no other member to clear a slot first — only
> at (≥ 10, 0): ten of twenty-four pairs won with none lost.** (9,0) at 0.0039 clears the fifth
> slot at best, i.e. only behind four other separations; (8,0) at 0.0078 clears the eleventh.
> In the 24-member secondary family the first slot is 0.00208 and (10,0) at 0.00195 still clears
> it; at that family's realised m under §9's ladder the slot widens to 0.05/16 = 0.00313 (where
> (10,0) still clears and (9,0) at 0.0039 does not) and to 0.05/8 = 0.00625 (where (9,0) does).
> If a task is excluded and the primary family falls to 12, the first slot widens to 0.00417
> and **(9,0) becomes self-sufficient** — an exclusion costs a task and buys one pair of
> resolution on the rest.

Against §5's floor prediction this is the arithmetic that matters: if `isvd` scores 6/24 on
`niah_single` and `frozen` scores 5/24, the discordance cannot plausibly reach (10, 0), and that
cell will not separate whatever the point estimates look like. The retrieval axis at n = 24 can
detect, **in a member that has to clear the first Holm slot on its own**, a ten-pair
one-directional swing and nothing smaller — a member standing behind four other separations
resolves (9, 0), and behind ten, (8, 0). That is stated now so that a table of non-separations is
read as the resolution the design bought, not as evidence of
equivalence — non-separation is not equivalence, which is exactly why the C branch additionally
requires a TOST to **pass**.

**The 32-window resolution, the α the TOSTs are read at, and the condition the C branch rests
on.** `kvdlra.eval.stats.tost` is two one-sided *t*-tests at α on the paired per-window
differences, so equivalence at ±δ fires when `t(1−α, n−1)·s/√n < δ − |d̄|`, where `s` is the SD of
the paired differences. **Every TOST is read at α = 0.05, uncorrected, and that is a decision, not
an omission**: the C branch requires *all* of its TOSTs to pass, which is an **intersection–union
test** — its null ("at least one of these contrasts is non-equivalent") is rejected only when every
component test rejects at α, and such a test has size at most α however many components it has, so
a multiplicity correction would buy nothing and cost power. The paired *t*-tests are the opposite
shape — a union, where any single member can separate a family — and they keep Holm (§4). At
n = 32 and d̄ ≈ 0, with t(0.95, 31) = 1.6955:

| | **α = 0.05, n = 32** (Stage 1) | α = 0.05, n = 16 (what 16 windows would have bought) |
| --- | --- | --- |
| t, half-width | 1.6955, 0.2997·s | 1.7531, 0.4383·s |
| decidable at **δ = 0.02** for | **s < 0.0667** | s < 0.0456 |
| decidable at δ = 0.05 for | s < 0.1668 | s < 0.1141 |

The measured arm-vs-arm paired SD on this protocol, over every contrast between arms whose basis
stayed orthonormal (§2 b: 15 contrasts across the three Table-4 pods), spans **0.0010 to 0.0563**
bits/token: the Llama contrasts sit at 0.0016–0.0019, a 35× margin, and the **Qwen** contrasts
between arms that really differ reach as high as **0.0563** — *inside* the 0.0667 bound by
≈ 1.2× (0.0667 / 0.0563 = 1.18), where on 16 windows they would have been outside 0.0456 and
Branch C would have been unreachable on that family for a reason having nothing to do with the
tracker.
That decidability is what §9's 32 windows are bought for. **The residual is pre-registered, with
its reporting obligation** (the pattern of `prereg/hygiene_table4.md` A1.2):

- `s` is read back from the run — `tables.py ppl` / `gate1.ppl_contrasts` print the paired
  bootstrap CI, whose half-width is ≈ 1.96·s/√n, so `s ≈ (hi − lo)·√n / 3.92`.
- If a member's realised `s` still exceeds its bound, **its TOST cannot fire whatever the point
  estimate is**, and the member is recorded as **`not decidable`** — never as a pass, never as a
  quiet fail. The C branch requires every one of its four TOSTs to pass, so the verdict is then
  **UNDECIDED**, and the report must say **which** it is: |d̄| genuinely above the margin
  (non-equivalence) or the interval too wide at the realised spread (not decidable). Only the
  second is a reason to spend a wider perplexity run; neither is a reason to widen the margin.
- **A wide interval cannot manufacture a Branch C**; it can only withhold one. On the A/B side the
  rule reads the boolean, which is `False` in two different situations — a difference outside the
  margin, and an interval too wide to place against it — so the conjunction §4 rule 1 requires
  (Holm-significant **and** TOST `False`) reads *"real, and not demonstrably inside ±0.02"*, which
  is weaker than *"demonstrably outside it"*. Where a separating member's TOST is `False` because
  it is `not decidable`, the verdict entry says so beside the branch, with `s` and the CI. That
  asymmetry is the reason ±0.02 is kept rather than widened to the tool's ±0.05 default: the plan
  fixed 0.02, a wider margin would make C *easier* to reach on noise, and the conservative
  direction for a gate that can retire the method's central claim is to refuse rather than to
  conclude.

Everything not in the three families is **descriptive** and carries no corrected p-value: §7's
diagnostics, the Wilson intervals, the `ratio` / `sbits` rows, the bf16 arm (its own file), the
pre-flight's rows, and `niah_multiquery` if it is ever run on these pods.

## 7. Secondary outcomes

- **(a) The other three trackers.** `isvd` vs `oja_tuned`, `fd`, `random`, per family and task,
  exact paired McNemar, **the secondary family** of §6 — 24 members if all three arms run, at its
  realised m otherwise (16 after the ladder's rung 1, 8 after rung 2), the C branch reading its
  `fd` members at that m; the `oja` and `random` members are reported with adjusted p-values and
  feed no branch. Wilson 95 % intervals (`kvdlra.eval.stats.wilson`) per cell beside the point
  estimates. Perplexity for the same three contrasts is reported with paired bootstrap CIs,
  **uncorrected and descriptive**, except the `fd` TOSTs the C branch names in §4.
- **(b) The bf16 arm.** Reported here as rows only; every reading is
  `prereg/bf16_gist.md`'s, committed before the Stage-1 launch commit.
- **(c) The guard, per arm, from `diag.jsonl`.** Per layer: maximum **pre-repair** `orth_err_k` /
  `orth_err_v`, the share of windows with `fixed_k` / `fixed_v` true, and the abort count. Expected
  from the same arm's measured behaviour (`results/filler_realism/diag.jsonl`, 19,968 rows of
  `bugSseed-r64-h256`: `fixed_k` 100.0 %, `fixed_v` 99.2 %, pre-repair `orth_err_k` min 1.00e-3 /
  median 1.02e-3 / max 1.41e-3, `orth_err_v` max 8.14e-3, 0 rows within an order of magnitude of
  the 1e-1 abort): **the repair fires in essentially every 64-absorb window at bf16 and a firing is
  not a finding.** Two arm-specific checks are:
  - **`frozen_r64_h256_seed` should show no repairs after the freeze.** `_guard_orthonormality`
    runs on every absorb for every tracker, and past `freeze_after` = 4096 `frozen_step` returns
    `U` unchanged with `rot = I`, so the measured error stays where the last repair left it and
    `fixed_k` / `fixed_v` should read **false in every window past the freeze** — with one
    exemption, stated now: a `[diag]` row covers a 64-absorb window and its `tokens_seen` is the
    last absorb in it, so the **first row per (sample, layer) with `tokens_seen > 4096`** is the
    window that straddles the freeze and can legitimately carry a repair from an absorb before it.
    Every row after that one must read false. A frozen arm still repairing there is a **dispatch
    defect**: §4's refusal rule voids the `isvd`-vs-`frozen` contrast for that family and refuses
    the verdict pending a dated amendment, and the defect is reported to lane L3.
  - **`random_r64_h256_seed`'s error should be flat.** Its basis is one CPU fp32 QR cast to the
    block dtype and never touched again, so `‖UᵀU − I‖_F` is that cast's rounding error and
    constant over the run. A rising trace on that arm is a dispatch defect, reported to lane L3 the
    same way — it removes that arm's secondary members (§6's realised m) and refuses nothing, since
    no branch reads `random`.
- **(d) The measured min/sample per arm, read from `manifest.cell_elapsed_s`.** `runner._cell`
  prints `[stage] cell arm=<arm> task=<task> ctx=<ctx> elapsed_s=<s> n=<records>` after every
  completed cell, timed with `time.perf_counter()` around that cell's trials; `scripts/pod.py
  harvest` (`CELL_S_RE`) folds them into `manifest["cell_elapsed_s"]` as
  `{"<arm>/<task>/<ctx>": seconds}` — **32 entries per Stage-1 pod** (8 arms × 4 tasks). Shipped on
  this branch at 03fba42 and pinned by `tests/test_pod_run_records_errors.py` and
  `tests/test_pod_manifest.py::test_harvest_records_the_cell_timings_the_run_printed`. The
  measurement, exactly:

  > **retrieval min/sample for an arm = (Σ of its four `cell_elapsed_s` values, seconds) ÷ 60 s/min
  > ÷ 96 samples = Σ ÷ 5,760.**

  **The perplexity axis emits no cell line** — `frontier.run_ppl` is not `_cell` — so the measured
  rate covers 96 of an arm's 128 samples. The other 32 are the perplexity windows, and they are
  **billed in §9 as samples at the arm's rate**, not absorbed into overhead; what the run does not
  return is a separate *measurement* of their rate, which is why §9's safety factor and not the
  cell timings is what covers a window slower than a retrieval trial. Each arm's measured rate is
  reported against its §9 budgeted rate; the reading is descriptive for Stage 1 and is what
  re-sizes Stage 2 (§9). Because each line carries
  its own seconds, the watchdog's per-poll `sort -u` cannot damage it and nobody has to be watching
  the run.
- **(e) The pairing invariant, verified.** For every `(task, seed, trial)` in a pod the eight arms'
  `prompt_sha256` must be identical: 4 tasks × 1 seed × 24 trials = **96 keys per pod**, each
  holding 8 records with one digest.

  ```python
  import json, collections
  rows = [json.loads(l) for l in open("results/gate1_v2_stage1_llama/trials.jsonl")]
  sha = collections.defaultdict(set)
  for r in rows: sha[(r["task"], r["seed"], r["trial"])].add(r["prompt_sha256"])
  bad = {k: v for k, v in sha.items() if len(v) != 1 or None in v}
  print(len(sha), "keys;", len(bad), "disagree;", sorted(bad)[:5])
  ```

  Expected `96 keys; 0 disagree`. A key whose set contains `None` is a failure as well as a
  mismatch: these pods run post-L2.3b code where the digest is written for every record. A key that
  disagrees is dropped from every paired statistic in that pod and the drop is reported with the
  key and the arms.
- **(f) Stored bits, on the pod.** `ratio` and `sbits` are recorded on every row. Arms 2, 4, 5, 7
  and 8 should print identical `sbits` (all bill the **live tracked rank** — D-015 — which is the
  cap at these settings), arm 3 within ≈ 0.1 % of them (§3's 1.00003× / 1.00003× match), and arm 6
  below them by its coordinate dtype. The pin §4's refusal rule reads is the **measured** ratio
  `median(sbits of nogist_*) / median(sbits of isvd_r64_h256_seed)` over that pod's records, which
  must lie within **1 ± 0.05**; outside it the byte match that makes the `nogist` contrast a
  mechanism contrast did not hold on the pod, those members are void and the verdict is refused.
  Any other gap is an accounting finding, reported. Both are read **before** §4's rule is applied,
  not after.

## 8. Log volume, and what counts as a complete `<label>.log`

The pod log is the only channel back from a vast.ai instance (`results/<pod>/` dies with it). Rows
per sample: one `[trial]` line per retrieval record, one cell row per (arm, task), one
`[stage] cell` timing row per (arm, task), **four** `[pplw]` lines (the 32-window split, below)
and one `ppl=` line per arm, and for each of the **seven** `bug`-kind arms the `[diag]` rows
`records.drained` prints when a sample's cache is drained — `full` emits none.

**No arm sets `diag_every`, so the cache default (64) applies.** The per-sample count is
**measured**: `results/filler_realism/diag.jsonl` holds 39,936 rows over 48 samples × 2 gist arms
at 16K on Llama-3.1-8B, i.e. **416 `[diag]` rows per 16K sample** = 13 rows per layer × 32 layers
(twelve completed 64-absorb windows plus `drain_diag`'s flush of the open one, at 804–810 absorbs
per layer) — `prereg/gate1_preflight.md` §6, the same anchor, with its snippet. Qwen2.5-7B has
**28** layers, so the same 13 rows per layer give **364**. A perplexity window prefills the same
16K and therefore sets the same absorb schedule; the Table-4 pods confirm that a perplexity sample
drains and emits (`diag_every: 4096` there gave exactly one row per layer per sample). The naive
1,025-absorb count — 17 rows per layer, **544** on Llama and **476** on Qwen — is carried as the
upper bound; `nogist_*` plausibly emits *fewer* than 416, since its 2423-token exact tier holds
columns the gist never absorbs.

| per pod | rows (Llama) | rows (Qwen) |
| --- | --- | --- |
| `[diag]`: 7 gist arms × 128 samples × 416 / 364 | **372,736** (≤ 487,424) | **326,144** (≤ 426,496) |
| `[trial]`: 8 arms × 4 tasks × 24 | 768 | 768 |
| cell rows: 8 × 4 | 32 | 32 |
| `[stage] cell` (§7 d): 8 × 4 | 32 | 32 |
| `[pplw]` + `ppl=`: 8 × 4 + 8 | 40 | 40 |
| `[stage]` other + ENV block + markers | ≈ 50 | ≈ 50 |
| **expected deduped `<label>.log`** | **≈ 374,000 lines** (≈ 103 MB at 275 B/row) | **≈ 327,000 lines** (≈ 90 MB) |

(275 B/row is measured, not assumed: the live filler-realism pod's `.raw` read 78.5 MB over
286,000 lines — `prereg/ss2_families.md` §8.)

**No `[diag]` row can be lost, by the arithmetic that decides it.** The only way the watchdog loses
a row for good is a **poll-to-poll gap**: more matched lines printed between two 150 s polls than
`vastai logs --tail 30000` holds (5,000 when the empty-fetch fallback is taken —
`scripts/pod/watchdog.sh`). A sample's `[diag]` rows are printed in one burst when its trial
drains, so the burst per poll is bounded by how many gist samples can *complete* in 150 s. The
fastest gist arm here is `nogist_*` at 1.55 min = 93 s per sample (§9), so **at most two** samples
close inside one poll: **≤ 2 × 416 + 2 = 834 rows per poll** (≤ 1,090 at the bound) — 36× under the
30,000-line window and 6× under the 5,000-line fallback. The unfiltered log would have to add
more than 4,000 lines in 150 s on top of the worst burst to open a gap. The watchdog dedupes
`<label>.raw` in place after every append (the L2 fix wave c437e75), so the raw stays at the size
of its distinct rows (≈ 90–103 MB) instead of growing by the re-fetched tail each poll; the launch
entry names the watchdog SHA.

**The watchdog's own clock needs no override.** `BUDGET_ITERS` defaults to
`(gpu_budget_h · 3600 + 7200) / 150 + 1` polls (floor 600); at the §9 bar of 82.0 h that is
**2,017 polls = 84.0 h** — the bar plus `boot.sh`'s 2 h `GRACE_S` self-destruct. Nothing goes on
the launch line.

**The completeness test is on the deduped `<label>.log`.** Exact: `grep -c '^\[trial\]'` = **768**,
`grep -c '^\[pplw\]'` = **32**, *four* per arm: `_log_pplw` prints one line while it fits in 400
characters and otherwise splits into `part=i/n` groups of **8** values, and at 32 windows the line
is ≈ **447** characters (32 values of 11 characters at `%.6f` for a 2048-token window's nll sum,
31 commas, the `[pplw] T=16384 <arm> ntok=2048 nlls=` head and the ` corpus=pg19-val` tail), so
every arm prints ⌈32/8⌉ = 4 parts — where the Table-4 pods at 16 windows printed one line of
≈ 255 characters. `records.parse_pplw_lines` reassembles the parts and raises on a gap, so a lost
fragment fails the harvest rather than shortening a sweep silently. And
`scripts/pod.py check results/gate1_v2_stage1_<family>` returns 0 (config hash, commit order, all
32 retrieval cells at n = 24, `env.txt` against the pyproject pins). Approximate:
`grep -c '^\[diag'` ≈ 374,000 (Llama) / 327,000 (Qwen) — **± 32 rows per Llama sample and ± 28 per
Qwen sample** (one per layer) is expected noise, since the per-layer window count moves by one when
a sample's absorb count crosses a multiple of 64 and v2 prompts vary by a few tokens across
haystacks, and `nogist_*`'s own count is one of the things the pre-flight measures — and
`manifest.diag_skipped` = 0 (a `[diag]` line a fetch cut in half is counted, not dropped, and fails
`check`). A `.log` whose `[trial]` count is short is short by construction: read `trials.jsonl` and
the `error` lines before concluding truncation, and do not harvest it as final.

## 9. Budget

**Unit and anchors.** "min/sample" is the wall-clock of one `(arm, task, seed, trial)` or one
perplexity window; a retrieval cell is 24 samples, the four-task context is 96 samples, and with
the **32** windows each arm runs **128 samples**. All rates on an A100 40 GB, and all of them are
the rates `prereg/gate1_preflight.md` §1 and §7 already fixed for this design. That file's §1 table
sizes these same eight arms at 112 samples — i.e. at 16 windows; the 32 windows §2 (b) and §6 buy
are the only change to its arithmetic, and they move the per-pod compute from 35.0 h to **40.0 h**:

- **`full` 0.6 — measured.** The D-005 real-text pod's `full` arm ran its 48 samples in ≈ 30 min
  (`docs/plan/cleanup/l2-ledger.md`, the 05:45 pod-signal line; `prereg/filler_realism.md` A2.5,
  instance 51553610).
- **`isvd_r64_h256_seed` 3.1 — measured, inside its observed bracket.** The same pod's r64 arm ran
  48 samples in ≈ 2.3–3.0 h → **2.9–3.7 min/sample** (`docs/plan/cleanup/l2-ledger.md`, Task 8's
  concern line "live-pod rates above the prereg's (r64 2.9–3.7, q4 3.8–4.5 min/sample)";
  `prereg/l2_smoke.md` §7), against the 2.1 min/sample measured on the W19 `a1q` pod that every pod
  since has budgeted at 3.0. **3.1 sits in the lower half of that bracket, and the sensitivity is
  stated rather than hidden**: at the bracket's top (3.7, the derived rates scaling with it to
  1.85 and 2.47) the eight arms cost 76.8 + 4×473.6 + 2×316.2 + 236.8 = **2,840 min = 47.3 h** and
  the point estimate per pod is 48.3 h — still 1.7× inside the 82 h bar.
- **`frozen` and `random` 2.1 — derived, = isvd / 1.5** (no per-absorb SVD after the freeze; a
  random basis never updates). **`nogist_*` 1.55 — derived, = isvd / 2** (no gist rebuild). Both
  factors are the lane plan's (`docs/plan/plans/2026-09-11-L3-L5-gate1-bf16-prereg.md`, "Pod
  sizing"). **Caveat, and it is the pre-flight's first suspect:** the 2423 / 4460-token exact tier
  is re-scored every absorb, which the r64 arm's 256-token tier is not.
- **`fd`, `oja_tuned`, `bf16` 3.1 — derived, = isvd**: each does one core factorization per absorb
  per stream, as the incremental-SVD step does.
- **Overhead 60 min**: boot, clone, `pip`, the weight download that stalled ≈ 1 h on a Table-4 pod
  (D-011 addendum 2), plus generator v2's ≈ 10 min of haystack materialization and ≈ 2 min of
  per-trial prompt construction (`prereg/l2_smoke.md` §7), which sit **inside** the safety factor
  rather than moving the bar.

| # | arm | min/sample | × 128 | minutes | cumulative compute | what has landed |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `full` | 0.6 | | 76.8 | 1.3 h | the ceiling |
| 2 | `isvd_r64_h256_seed` | 3.1 | | 396.8 | 7.9 h | the paired reference |
| 3 | `nogist_h2423` / `nogist_h4460` | 1.55 | | 198.4 | 11.2 h | one primary contrast |
| 4 | `frozen_r64_h256_seed` | 2.1 | | 268.8 | **15.7 h** | **both primary contrasts, both axes** |
| 5 | `fd_r64_h256_seed` | 3.1 | | 396.8 | **22.3 h** | **the C branch decidable** |
| 6 | `isvd_r64_h256_seed_bf16` | 3.1 | | 396.8 | 28.9 h | the bf16 reading |
| 7 | `oja_r64_h256_seed_tuned` | 3.1 | | 396.8 | 35.5 h | secondary |
| 8 | `random_r64_h256_seed` | 2.1 | | 268.8 | 40.0 h | the floor control |
| | **compute per pod** | | | **2,400 min = 40.0 h** | | |

The total is exact, not rounded: 128 × (0.6 + 4 × 3.1 + 2 × 2.1 + 1.55) = 128 × 18.75 = 2,400 min.

| pod | compute | + overhead | point estimate | **`gpu_budget_h` (2× bar)** |
| --- | --- | --- | --- | --- |
| `gate1_v2_stage1_llama` | 40.0 h | + 60 min | **41.0 h** | **82.0** |
| `gate1_v2_stage1_qwen` | 40.0 h | + 60 min | **41.0 h** | **82.0** |
| **Stage 1 total** | | | **82 GPU-h** | **164 GPU-h** |

At the **$0.45–0.74/h** of `prereg/gate1_preflight.md` §7 (the floor rounded *up* from the observed
$0.40, so the low end of every figure here is the conservative one):

| | GPU-h | × $0.45 | × $0.74 |
| --- | --- | --- | --- |
| Stage 1, point | 82 | $37 | $61 |
| **Stage 1, bar** | **164** | **$74** | **$121** |
| after arms 1–4 on both pods (both primary contrasts), + overhead | 33.4 | $15 | $25 |
| after arms 1–5 on both pods (the C branch decidable), + overhead | 46.6 | $21 | $34 |

**This experiment asks for ≈ 82 GPU-hours expected, 164 at the bar — $37–61 expected, $74–121 at
the bar.** The two pods run on two instances at once, so the wall clock is the longer of them
(≈ 41 h expected, 82 h at its bar), not the sum.

**This supersedes the plan's "Budget ~40 GPU-h" and `GATES.md` §G3's "≤ 50 GPU-h in manifest"**,
exactly as `prereg/gate1_preflight.md` §1 already records: both numbers predate the L2 pods' rate
measurements and the tasks-per-cell correction (`prereg/filler_realism.md` §GPU budget), and
neither was ever derived from a measured min/sample. The gate line is read against this table, the
re-sizing is recorded in the DECISIONS launch entry, and the owner sees the cut ladder below before
a dollar is spent.

**The pre-committed ways to spend less, in this order** — the arm order of §3 is what makes them a
design and not a salvage:

| cut | compute/pod | point/pod | bar/pod | Stage-1 point | Stage-1 bar | $ at the bar | secondary m (§6) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 40.0 h | 41.0 | 82.0 | 82 GPU-h | 164 GPU-h | $74–121 | 24 |
| **1. drop `random`** | 35.5 h | **36.5** | **73.0** | 73 GPU-h | 146 GPU-h | $66–108 | **16** |
| **2. also drop `oja_tuned`** | 28.9 h | **29.9** | **59.8** | 60 GPU-h | 120 GPU-h | $54–89 | **8** |

Rung 1 costs the floor control: the row that says what the rank-64 storage, the tier, the sinks and
the ring are worth with no tracking at all. Rung 2 costs plan arm (b) entirely. **Neither rung
touches §4's rule**: every member of the primary families and the C branch's `fd` members survives
both cuts, and the secondary family is corrected at the realised m each rung leaves (last column) —
which widens Holm's slots, makes a separation from `fd` easier to reach, and so makes Branch C
harder, never easier, to select.

**Why the ladder is not "n = 16 for `oja`/`fd`".** It saves *less* (n = 16 on those two arms drops
2 × 32 samples per pod = 198 min = **3.3 h**, against **4.5 h** for dropping `random` outright), it
breaks the balanced 2 haystacks × 3 depths × 4 codes cell of `ruler_v2_16k_g1` that every other
arm's rows are comparable through, and at n = 16 a self-sufficient Holm separation needs (≥ 10, 0)
of *sixteen* pairs — a 0.63 swing instead of 0.42 — on exactly the two arms with no prior on
hardware.

**Overrun is a stop-and-report, not a silent extension.** `gpu_budget_h` is the pre-registered bar,
enforced on the pod itself by `pod.py launch --max-hours` (`boot.sh` runs the entrypoint under
`timeout`; a run that reaches the bar prints `===RUN_TIMEOUT_…===` and is harvested as
`RUN_FAILED` with `timeout: true`). A pod still running past its bar is a pod to kill and diagnose
(`prereg/hygiene_table4.md` §9), and the **first arm-2 cell's `cell_elapsed_s`** (§7 d) is the
first thing to read against the 3.1 assumption — at 24 samples a cell that is 74 budgeted minutes
of compute, so the reading lands ≈ **3.5 h** into the run (60 min boot + `full`'s 77 min + 74),
against an 82 h bar.

**Re-sizing by amendment, before the launch commit.** If any pre-flight trigger fires —
`isvd_r64_h256_seed` **> 4.7** min/sample, `nogist_h2423` **> 3.1**, `frozen_r64_h256_seed`
**> 3.1** (`prereg/gate1_preflight.md` §4 (iv)) — this table is re-derived from the measured
`manifest.cell_elapsed_s` rates in a dated amendment appended below, and each pod YAML's
`gpu_budget_h` moves in the commit that precedes the Stage-1 launch. If the pre-flight's harvested
log carries no `[stage] cell` lines at all, its §4 (iv) fallback applies: each trigger is recorded
**`not measured`**, never "passed", and this table is scaled by that pod's aggregate κ. **The pods
are never launched over their pre-registered bar.**

**Stage 2 — sized now, conditional, launched only by amendment.** Stage 2 runs only if Stage 1
does **not** select Branch C, and its pods are created by a dated amendment to this file that
states the arm list, the Holm families and the re-derived budget before any launch:

| Stage-2 pod | arms | rate basis | point | bar |
| --- | --- | --- | --- | --- |
| Mistral-7B-v0.3, 16K (`nogist_h2423`, the 1024-wide twin) | all 8 | the table above, at 128 samples | **41.0** | **82.0** |
| each of Llama / Qwen / Mistral at 32K | the **five** arms the rule reads at 16K — `full`, `isvd`, the 32K `nogist` twin (below), `frozen`, `fd` | **2× the 16K rates** (twice the absorbs): 1.2 + 6.2 + 3.1 + 4.2 + 6.2 = 20.9 min/sample × 128 = 2,675 min = 44.6 h, + 60 min | **45.6** | **91.2** |
| **Stage 2 total** | | | **178 GPU-h** | **356 GPU-h** |

At $0.45–0.74/h: **$80–132 point, $160–263 at the bar** (41.0 + 3 × 45.6 = 177.8 → 178;
82.0 + 3 × 91.2 = 355.6 → 356). Stage 2 at 32K uses `ruler_v2_32k_g1` and
`configs/tasks/ppl_32k_pg19val.yaml` — 32 non-overlapping windows of the 84 PG-19 validation
supplies at that span (D-013), the same 128 samples per arm;
`isvd_r64_h256_seed_bf16`, `oja_r64_h256_seed_tuned` and `random_r64_h256_seed` are not in
the 32K pods, which is the cut ladder applied in advance — the bf16 arm joins by
`prereg/bf16_gist.md`'s own amendment if its Stage-1 reading passes. The Mistral 16K pod enters
§4's rule as a third model family; the three 32K pods are **descriptive** (§4's scope note, §6's
families) and change no letter.

**The 32K no-gist arms are new arm files, and here is their arithmetic.** The byte match is solved
at the context the pod runs (§3), so `nogist_h2423` / `nogist_h4460` — solved at t = 16384 — are
**not** byte-matched at 32K and are not the arms those pods run. Re-solving with
`kvdlra.accounting.bug_footprint` exactly as the 16K arm docs did, at **t = 32768**, where the r64
arm holds 32768 − 256 − 4 − 32 = **32,476** coordinate columns: for a **1024**-wide layer
(Llama-3.1-8B, Mistral-7B-v0.3) the reference bills **148,875,008** stored bits/layer and the twin
bills `1,245,376 + 32,800·H`, so **H = 147,629,632 / 32,800 = 4500.90 → 4501**, a **1.00002×**
match; for a **512**-wide layer (Qwen2.5-7B) the reference bills **141,993,728** and the twin
`622,784 + 16,416·H`, so **H = 141,370,944 / 16,416 = 8611.78 → 8612**, a **1.00003×** match.
`nogist_h4501` and `nogist_h8612` are **committed by the Stage-2 amendment** — each with this
arithmetic in its `doc:`, the same 5 % test pin as the 16K twins, and a **re-derived rate**: the
2× rule above bills them at 3.1 min/sample, but a 4501 / 8612-token exact tier is re-scored every
absorb (the caveat above, with 1.9× more tier than at 16K), so the amendment re-derives that rate
from Stage 1's measured `cell_elapsed_s` and not from the factor alone.

**Credit.** **$92.52** (`vastai show user --raw`, 2026-09-19 13:50, as `prereg/gate1_preflight.md`
§7 records it). Stage 1's point estimate of $37–61 fits inside it; **its bar of $74–121 does
not**, at any rate above $0.56/h — and it is not the only pod queued: `prereg/ss2_families.md` §9
carries a 121 GPU-h bar and `prereg/l2_smoke.md` §7 a 168 GPU-h bar, both also waiting. **D-003
(the top-up) is open and precedes the Stage-1 launch commit.** The pre-flight's own $4–6 is what
keeps this $74–121 from being spent on a design that does not run
(`prereg/gate1_preflight.md` §7).

## 10. Provenance

- **Pods**: `configs/pods/gate1_v2_stage1_llama.yaml`, `configs/pods/gate1_v2_stage1_qwen.yaml` —
  **committed by lane L3 Task 3 to match §3 and §9**, before the launch commit, together with
  `configs/tasks/ruler_v2_16k_g1.yaml` reduced to the four Gate-1 tasks, `kvdlra.eval.gate1`
  (§4's three functions), the `make gate1` target, and the `tests/test_pod_manifest.py` rows that
  pin each pod's prereg path, arm order, task list, n = 24, a positive `gpu_budget_h`, a
  `config_hash` distinct from every other pinned pod's, and every arm through
  `frontier.build_arm` at t = 16384 as the runner builds it before its first trial.
  `tests/test_pod_manifest.py::test_check_rejects_tampered_config_hash` covers the manifest side.
- **Arms**: `configs/arms/full.yaml`, `isvd_r64_h256_seed.yaml`, `nogist_h2423.yaml`,
  `nogist_h4460.yaml`, `frozen_r64_h256_seed.yaml`, `fd_r64_h256_seed.yaml`,
  `oja_r64_h256_seed_tuned.yaml`, `random_r64_h256_seed.yaml` — all on this branch and **none
  edited by this commit or by the launch commit** — plus `isvd_r64_h256_seed_bf16.yaml`, committed
  by lane L3 Task L5.1 before the launch commit and read by `prereg/bf16_gist.md`.
- **This file must be committed strictly before the launch commit**, and before
  `prereg/gate1_preflight.md`'s own launch commit as well (`prereg/gate1_preflight.md` §1 and §8:
  the Gate-1 body is fixed before the pre-flight runs, so every row the pre-flight returns enters
  here as a dated amendment and never as an authored premise). **The two orderings are enforced
  differently, and the difference is stated rather than blurred.** For the **Stage-1** pods this
  file *is* the launching pod's prereg, so `scripts/pod.py launch` refuses outright —
  `prereg_error` covers missing, uncommitted, not-a-strict-ancestor and
  committed-by-the-launch-itself, plus a dirty tree and an unpushed SHA — and `scripts/pod.py
  check` re-checks the order against the manifest's `git_sha` at harvest. For the **pre-flight**
  launch there is no machine refusal: `pod.py launch` checks only the launching pod's own prereg
  (`prereg/gate1_preflight.md` §8 says the same), so this file's precedence over *that* commit is a
  **lane rule**, evidenced after the fact with
  `git merge-base --is-ancestor <this file's first commit> <pre-flight launch SHA>` and recorded in
  the pre-flight's DECISIONS launch entry, which names both SHAs. The same command checks the
  Stage-1 ordering that `launch` already refused on.
  **The commit that adds this file launches nothing and adds nothing else.**
- **The Stage-1 launch precondition the pre-flight sets.** Stage 1 may be launched only if the
  pre-flight's readings **(i) completeness** and **(iii) pairing** passed
  (`prereg/gate1_preflight.md` §4) — an arm that does not run on the pod path, or a broken
  `prompt_sha256` pairing, is a defect in the machinery every paired statistic in §4 rests on. If
  either failed, a **dated amendment here names the failure, its repair and the commit that
  carries it, before the Stage-1 launch commit**; reading (ii) fires the task exclusion of §6 and
  reading (iv) the §9 re-sizing, both by the same route. No reading is waived by being
  inconvenient, and none of the four is read as "passed" when it was `not measured`.
- **Amendments only, never edits.** §1–§11 are not edited after the launch commit. Every later
  change is a dated Amendment appended below, in the pattern of `prereg/hygiene_table4.md`:
  Amendment 1 (§2's generator-v2 rows), the §4 (ii) task exclusion, the §9 re-sizing, and Stage 2's
  own amendment — each committed before the launch commit it governs.
- **Launch**: `scripts/pod.py launch --pod gate1_v2_stage1_<family> --offer <id>` (`--max-hours`
  defaulting to `gpu_budget_h` = 82.0), from a pushed SHA on a clean tree; the watchdog under
  `caffeinate -s -i scripts/pod/watchdog.sh gate1_v2_stage1_<family>` (no `BUDGET_ITERS` override,
  §8); the pod self-destructs `GRACE_S` = 2 h after its final marker. Each launch is a
  `docs/plan/DECISIONS.md` entry under D-011's standing authorization naming the pod, this file's
  first-commit SHA, the launch SHA, the offer id, the hourly rate, the bar and the credit before
  launch — the format of the D-005 addendum entries — and naming **D-003's resolution**, since the
  bar does not fit the credit at writing.
- **Outputs**: `results/gate1_v2_stage1_llama/` and `results/gate1_v2_stage1_qwen/`, each with
  `manifest.json` (git SHA, config hash, HF model revision, `dataset_sha256` for the four haystack
  sources and for `pg19-val`, `cell_elapsed_s` for the 32 retrieval cells, torch / CUDA / triton /
  transformers versions, GPU, wall clock, command line, `errors`, `records`, `diag_skipped`,
  `timeout`), `trials.jsonl` (32 cells × 24, every row with `prompt_sha256`, `haystack_id`,
  `depth`, `code_family`, `ratio`, `sbits`), `ppl.jsonl`, `pplw.jsonl` (8 arms × 32 windows = 256
  rows, reassembled from the four `[pplw]` parts per arm — §8),
  `diag.jsonl` (the seven gist arms' rows), `env.txt` (rebuilt from the log's ENV block),
  `pods.txt`.
- **Citability.** A number from these pods is citable only once `scripts/pod.py check` passes on
  its directory (config hash, commit order, every retrieval cell at n = 24, `env.txt` at the
  pyproject pins) and `make gate1` regenerates the table from the committed records into
  `docs/paper/tables/gate1.md` (CLAUDE.md's rule, and `make tables` stays diff-clean beside it).
- **The five-reviewer simulation runs on the table before anyone reads it.**
  `docs/plan/KICKOFF_WEEKS0-3.md` Part E: "When Gate 1 v2 harvests, run the five-reviewer
  simulation on the table alone before reading it yourself." The simulation sees
  `docs/paper/tables/gate1.md` and this file, and nothing else — no lane report, no summary, no
  branch recommendation — and its output is committed beside the table. Only then is the verdict
  read.
- **The verdict.** `gate1.gate1_verdict`'s branch, the members that decided it, the three adjusted
  p-value families of §6, the refusal rule's outcome if it fired, the deviations §3 names (FD at
  ℓ = r, no 8B-tuned Oja arm, one Oja schedule for both streams), and — if it happened — the
  perplexity-only reading of §5 with what it does and does not license, go to
  `docs/plan/DECISIONS.md` as the **Gate-1 outcome**, with the evidence path
  `results/gate1_v2_stage1_{llama,qwen}/` and the table path. `GATES.md` §G3's four lines are
  ticked against that entry — **and Stage 1 alone can tick only three of them**: line 1 (prereg SHA
  precedes launch SHA; its "≤ 50 GPU-h in manifest" superseded by §9, the supersession recorded in
  the launch entry), line 3 (`make tables` renders Holm-corrected retrieval + TOST perplexity) and
  line 4 (DECISIONS names the branch with the rule from `ICML2027_PLAN.md`). **Line 2 — "6 trackers
  × 3 families × 2 ctx × 4 tasks × n = 24 harvested"** — needs the third model family and the 32K
  context, so it is tickable **only with Stage 2**, and it stays open while Stage 1 stands alone.

## 11. What this does not decide

Gate 2 — whether the multi-value edge over 2-bit KIVI at its published protocol survives
(`prereg/ss2_families.md`, D-005's branch-3 amendment) — and Gate 3, the kernel's resident-memory
and throughput question (`prereg/kernel_smoke.md`, ADR 0001, D-002); the real-Palu port (D-004);
the KVQuant arm (D-006); the OjaKV comparison (D-017 — the `oja_*` arms here are Oja subspace
tracking inside this harness at matched bytes and are **not** OjaKV, whose published operating
range is ≈ 0.47× stored state); the bf16 arm's non-inferiority reading (`prereg/bf16_gist.md`);
anything at 32K, and anything on Mistral-7B-v0.3, before Stage 2's amendment; whether the r64
configuration has a real-text retrieval niche at some other rank or tier budget (a question D-005's
closing entry left open, and one this file's fixed r = 64 and h = 256 cannot answer); the default
value of `min_sv_frac`, `qr_every` or the guard's tolerances (Table-4's, D-011 addendum 8); the
exact tier's selection rule or size; `niah_multiquery`; official RULER's own harness, LongBench,
and any memory, throughput or latency claim. Gate 1 asks one question — does the online-tracked
gist do work a frozen basis, a random basis or no gist does not — and these two pods answer that
one, on two families at one context.

**STATUS: awaiting owner go (DECISIONS D-003 top-up; launch under D-011).**

---

## Amendment 1a (2026-09-20, before the Stage-1 launch commit; first committed at 5f2cdcf)

§1–§11 above are the design as it was written before either pod was launched and before any
generator-v2 row existed on hardware. They are left untouched. This section records only what
**does not depend on pre-flight data**: the places where the shipped `kvdlra.eval.gate1` and
`scripts/pod.py` are narrower, or differently worded, than the body describes, and the lane
rulings that fixed them. No Stage-1 pod has launched, so this is committed before the launch
commit exactly as §10 and the head of this file require.

**Order, stated exactly, because this amendment's heading first got it wrong.** It was drafted
against the code alone, to land before the pre-flight harvested — and the pre-flight did not
wait: instance 51722149 was destroyed at 14:11 EDT and its **PARTIAL** harvest was committed at
**8d10483** (14:25:53 EDT, 97 of 240 trials; the capture loss and its repair are D-011 addendum
10), three minutes before this section was first committed at **5f2cdcf** (14:29:04 EDT). The
original heading said "before the pre-flight harvest", which is false in commit order; that
line is corrected here rather than left standing. **What it claimed of the content is
unchanged and is checkable**: every item below is read off the code at HEAD and names the
`file:line` it matches, not one is read off `results/gate1_preflight/`, and the pre-flight's
three promised returns — the §2 baseline rows, the `vt` ceiling decision and the §9 re-size —
are **not** here. They are Amendment 1b's, and that amendment must also say what the partial
capture leaves readable.

**Amendment 1b, after that harvest, carries the rest** — the pre-flight's per-task rows into §2
(the promise at the head of this file, item 1), the `vt` ceiling decision (item 2), and the §9
re-size against the measured rates (item 3). Nothing below anticipates any of them.

**No rule changes here.** §4's decision rule stands branch for branch and threshold for
threshold; §6's three families keep their composition; §9's bars do not move. What changes is
the *description* of machinery that §1–§11 were written before.

Line numbers are as of the commit that corrected this heading (the cleanup commits after
5f2cdcf moved `gate1.py`); the symbol names beside them are the durable anchor.

### A1a.1 The `prompt_sha256` drop is scoped to the member — §4 governs, §7 (e) is withdrawn on this point

§4 ("Primary contrasts, written exactly") drops a key whose two digests disagree "from that
member and the drop is reported with the key". §7 (e) says instead that such a key "is dropped
from **every paired statistic in that pod**". Those are two different rules, and the code
implements §4's: `_mismatched` (`src/kvdlra/eval/gate1.py:441`) is computed per **cell pair**
inside `_draft_retrieval` (`:401`), so only the keys *those two arms* disagree on leave
*that one* McNemar; every other member of the pod keeps its 24 keys.

**§7 (e)'s "from every paired statistic in that pod" is withdrawn.** A digest disagreement is a
property of the pair of arms that wrote them: arms X and Y disagreeing on key *k* is no evidence
about arms P and Q on key *k*, and dropping *k* pod-wide would shrink members that are still
byte-identical. Nothing else in §7 (e) changes — the drop is still reported with the key **and**
the arms, its check is still "96 keys; 0 disagree", and a pod with any mismatch at all is a
defect to diagnose, not a pod to quietly re-pair. (Lane ledger, Task-3 fix round 1, 2026-09-20.)

### A1a.2 `REFUSED` is a fourth verdict value, and the signature the module ships

§4's code sketch reads `verdict, members = gate1.gate1_verdict(retr, ppl, diag=diag, sbits=sbits)`
returning `"A/B" | "C" | "UNDECIDED"`. Two corrections.

**(i) Ruling R-L3-12 — a refusal is its own branch value.** "The rule ran and neither branch
held" (`UNDECIDED`) and "the rule never ran on that family" (`REFUSED`) are different findings,
and a table that prints one for the other misreports the pod. `Verdict.branch`
(`gate1.py:184-190`) therefore takes **four** values, and the precedence the module applies is
**A/B, then C, then REFUSED, then UNDECIDED** (`gate1_verdict`, `:689-722`). Every sentence of
§4's "Refusal, and the `--` rule" that says a refusal "returns `UNDECIDED`" reads **`REFUSED`**.
Nothing else about those five refusals moves: they are still read before the rules, still
*return* rather than raise (a raise would leave `make gate1` with no table in which to print the
failure), and still name the arm and the exception text.

**(ii) Ruling R-L3-13 — the shipped signature.** The §4 sketch is restated to it:

```python
from pathlib import Path
from kvdlra.eval import gate1

data      = gate1.load([Path("results/gate1_v2_stage1_llama"),
                        Path("results/gate1_v2_stage1_qwen")])   # Gate1Data
retrieval = gate1.retrieval_contrasts(data)                      # list[Contrast]
ppl       = gate1.ppl_contrasts(data)                            # list[PplContrast]
verdict   = gate1.gate1_verdict(retrieval, ppl, data)            # -> Verdict
#   Verdict: branch ("A/B" | "C" | "REFUSED" | "UNDECIDED"), reason,
#            families_separated, members
```

`diag` and `sbits` are not parameters. `Gate1Data` carries both — `frozen_defects` read from
`diag.jsonl` and `sbits` from `ppl.jsonl` (`gate1.load`, `:274-342`) — so refusals 1, 3 and 4 are
still decided "from the records, not by eye", through one object instead of three arguments. The
readings are identical and no shim is added. Two smaller restatements of the same sketch: the
contrasts come back as **lists** of `Contrast` / `PplContrast`, not as dicts keyed by tuple, and
`Contrast` carries `dropped_keys` (A1a.1) beside the fields §4 names.

### A1a.3 Refusals compose per family, and a refused family leaves every Holm family before the correction

**Ruling R-L3-15 — per family, not per verdict.** §4 reads as though any refusal refuses the
whole verdict. The shipped rule scopes each refusal to the family it fires on: the per-family
loops in `gate1_verdict` skip a refused family (`:613-615` for rule 1's separations, `:626-628`
for rule 3's blockers), so **A/B is read on the families that remain** and a refusal elsewhere is
printed beside the branch, never instead of it (`:689-700`). **C is not** read that way: a
refused family blocks C outright (`is_c = not blockers and not refusals`, `:675`), because C is a
positive claim of non-separation over *every* family in the input and a family whose members were
never read is not evidence for it. `REFUSED` is what is left once neither branch composes
(`:707-712`).

**Ruling R-L3-16 — before the correction, not only before the rules.** "A refused family
contributes no evidence, in either direction" has to hold of Holm's realised *m* as well, so
`_refusals` (`gate1.py:495`) runs first, in `load` (`:341`), and a refused family's members leave **every** Holm family
— primary retrieval, secondary, perplexity — before `_holm_by_group` (`:363`) corrects it
(`:435`, `:489`, `:609`). The realised *m* is then the members the non-refused families
contribute, which is the *m* those families would have had on their own. Without it, a third
family refused on its byte match would widen Stage 1's 16-member family to 24 and withdraw an A/B
the two clean families earned. The excluded members are still computed and still printed, with
`refused (excluded from the Holm family)` where their adjusted p would be.

**The Holm family is the pod set of ONE `make gate1` invocation** — one stage per call. §6's
primary retrieval family is 16 *pooled* across Llama and Qwen, not 8 per model, so the grouping
key is the invocation, not the model family; the context length splits the verdict's 16K family
from a descriptive 32K one inside one invocation, which §6 also fixes (`_holm_by_group`'s docstring,
`:372-374`). One stage per call is the caller's contract — `make gate1` names Stage 1's two pods —
and not a check inside the function.

**Neither ruling can change a Stage-1 verdict, and neither is a post-hoc widening.** Stage 1 has
exactly two families and `MIN_FAMILIES_FOR_AB` is 2 (`gate1.py:76`). Refuse one and at most one
remains, so "≥ 2 families separated" is unreachable; C is unreachable too, since any refusal
blocks it. Every Stage-1 input carrying a refusal therefore lands on `REFUSED`, with or without
R-L3-15 and R-L3-16. Both rulings bite only where a *third* family is in the same invocation —
Stage 2's Mistral pod, which §4 rule 2 and §6 already admit as the third model family — and both
move in the conservative direction there: a refused family is excluded from the count that can
reach A/B, and it blocks C outright.

### A1a.4 `sbits` lives in `ppl.jsonl`, as `ratio_stored_bits`

§7 (f) opens "`ratio` and `sbits` are recorded on every row", and §10's `trials.jsonl` field list
names "`ratio`, `sbits`". Neither is true of `trials.jsonl`: `TrialRecord`
(`src/kvdlra/eval/records.py:102-118`) carries no such field, and none is written. The `sbits`
that §4's byte-match refusal reads is `PplRecord["sbits"]` in **`ppl.jsonl`**
(`records.py:155-164`), written from the frontier row's **`ratio_stored_bits`**
(`runner._ppl_record`, `src/kvdlra/eval/runner.py:385`) — an arm's stored bits relative to
`full`, which is why the common denominator cancels in the `nogist`/`isvd` ratio the refusal
takes. `gate1.load` reads it there and nowhere else (`gate1.py:322-325`) and takes the **median**
over that arm's perplexity rows (`:335`).

So §7 (f) and §10 read: **`sbits` is a `ppl.jsonl` field, `ratio_stored_bits` at the source; a
`trials.jsonl` row carries `prompt_sha256`, `haystack_id`, `depth` and `code_family`, and no
footprint column.** The retrieval cell *line* does print `ratio=`/`sbits=` (`runner.py:314`) and
`records.parse_cell_lines` (`:228`) reads them, but `pod.py harvest` writes no `cells.jsonl` for
these pods (`scripts/pod.py:594-615`), so `ppl.jsonl` is the only committed carrier.

One consequence, stated plainly: **the byte-match refusal needs the perplexity axis to have run.**
A pod whose `ppl.jsonl` is absent, or whose rows carry no `sbits`, reads `byte match not measured
(no sbits on the records)` and is **refused** (`gate1.py:536-537`) — an unmeasured reading is
never a pass (§9's rule). §7 (f)'s descriptive comparison across arms is unaffected.

### A1a.5 Two blockers `gate1_verdict` adds that §4 does not state

Both are refusals to conclude, both withhold a C and neither can manufacture one or create an
A/B, so both move only in the conservative direction.

**(a) C requires the complete task set** (ruling R-L3-16 (b)). §4 rule 3 (i) claims
non-separation "on **any task in any 16K family**", and those tasks are §3's four, not whichever
ones the records happen to carry: quantified over the tasks merely *present*, a pod that ran two
of them and separated on neither would print the strongest claim this gate can make off half the
evidence. So C is blocked unless every task of `TASK_ORDER` — `niah_single`, `niah_multikey`,
`niah_multivalue`, `vt` (`gate1.py:105`) — is present with an `isvd` row set in **every** 16K
family: `incomplete task set: <family> lacks <tasks>` (`:629-635`). This guards **presence**
only. Per-cell n-completeness (every retrieval cell at n = 24) is `scripts/pod.py check`'s job,
exactly as §10's citability rule already has it, and the verdict does not duplicate it.

**(b) An input with no 16K family is UNDECIDED, never C.** C is selected as "nothing blocks it",
so an empty (dry-run) pod directory or a 32K-only record set would otherwise satisfy it
vacuously — a Branch C read from no members at all. `NO_MEMBERS` (`gate1.py:121`) blocks C when
no family carries the reference arm at 16K, and the branch reads `UNDECIDED` with that reason
(`:670-671`, `:713-714`).

### A1a.6 `manifest.cell_elapsed_s` carries 40 entries per Stage-1 pod, and the perplexity rate is measured

§7 (d) states "**The perplexity axis emits no cell line** — `frontier.run_ppl` is not `_cell` —
so the measured rate covers 96 of an arm's 128 samples", and §8's log-volume table and §10's
manifest bullet both count **32** entries. Lane task L3.3a made the perplexity axis print the
same line: one per (arm, ctx) sweep, keyed by the perplexity **task** name (`ppl_16k_pg19val`,
which no retrieval sub-task is called) so that `harvest` folds both axes into one
`cell_elapsed_s` without pooling two cells (`src/kvdlra/eval/runner.py:368-375`; `CELL_S_RE`,
`scripts/pod.py:430`; the fold, `pod.py:620-622`).

Per Stage-1 pod, therefore, **40 entries**: 8 arms × 4 retrieval sub-tasks = 32, plus 8 arms × 1
perplexity sweep = 8. §8's row "`[stage] cell` (§7 d): 8 × 4 | 32 | 32" reads **8 × 4 + 8 = 40**,
and §10's "`cell_elapsed_s` for the 32 retrieval cells" reads "for the 32 retrieval cells and the
8 perplexity sweeps". §8's expected line counts move by 8 lines per pod, which changes no
conclusion it draws.

§7 (d)'s formula is unchanged and still reads the four **retrieval** cells only:

> retrieval min/sample for an arm = (Σ of its four retrieval `cell_elapsed_s` values, seconds)
> ÷ 60 s/min ÷ 96 samples = Σ ÷ 5,760.

What changes is the sentence after it. The perplexity windows are still **billed** in §9 at the
arm's retrieval rate — §9 is not re-derived here and no bar moves — but their rate is now
**measured at harvest** from that arm's own perplexity entry (its seconds over the 32 windows of
the sweep), instead of being covered only by §9's safety factor. §7 (d)'s "the measured rate
covers 96 of an arm's 128 samples" is withdrawn: the harvest carries a clock for all 128, and the
perplexity rate is reported beside the retrieval rate as a measurement, not an assumption.

### A1a.7 Harvesting a relaunched instance of a Stage-1 pod

§10 names one output directory per pod and does not say what happens when an instance dies and
the pod is relaunched. Lane task L3.3b widened `pod.py harvest`'s non-shrink guard from
`trials.jsonl` to **every** record file the harvest writes (`_shrink_refusal`,
`scripts/pod.py:399-413`), and that guard is **all-or-nothing**: if any one file would shrink,
the whole harvest is refused and nothing is written (`pod.py:602-606`). A truncated fallback
fetch — `vastai logs --tail` capped at 5,000 lines when the full fetch comes back empty — is
exactly the case it exists for, and it would otherwise refuse a harvest that is good on the other
four files.

The recipe, pre-committed here: **a relaunched instance harvests into its own directory**,
`pod.py harvest --out results/<pod>_<instance>` (`pod.py:908`), never over the first one. The two
directories are then compared and the complete one is what `make gate1` reads — never both at
once, which `gate1.load` refuses anyway ("a second `<family>` pod — one pod per family",
`gate1.py:302`), and never a merge of the two. `--force` (`pod.py:909-911`) overwrites in place
and is used only with its reason written into the DECISIONS entry beside the instance id, because
it discards rows already on disk. This is a harvest recipe and not a change to what is measured:
no statistic in §4 or §6 crosses a pod boundary, and two instances of one pod are two attempts at
one pod, never two families.

### A1a.8 The task-exclusion knob exists; which task it names is Amendment 1b's business

§6 states the arithmetic for a task the pre-flight's ceiling rule excludes (16 − 4 on the primary
retrieval family, 24 − 6 on the secondary, the perplexity family untouched) but named no
mechanism. The mechanism lands on this branch in the commit after this one: **`EXCLUDED_TASKS`**,
a named module constant in `src/kvdlra/eval/gate1.py`, an empty `frozenset()` that an amendment
sets. A task it names leaves the primary retrieval family, the secondary family, and both of
C's per-task readings (the blocker loop and the `TASK_ORDER` completeness check of A1a.5 (a)).
Its cells are still run, still scored and still **rendered** in the Gate-1 table as descriptive
rows, exactly as §6 requires ("The excluded task is still run, still reported descriptively"),
and a separation on an excluded task blocks nothing.

**Which task, if any, it names is Amendment 1b's business.** The ceiling rule reads the
pre-flight's harvested `full` accuracy, that harvest has not happened, and nothing here
anticipates it. The knob is empty on this branch, and the Holm families are §6's 16 / 4 / 24
unless Amendment 1b sets it.

### A1a.9 One wording correction in §2 (b)

§2 (b)'s arm-vs-arm table labels the Qwen `isvd_r256_tol` − `isvd_r256_f0.01_tol` row (paired SD
**0.0490**) "the floor: a real change of subspace". "Floor" is the wrong word for that number:
the paragraph below the table enumerates the population as **0.0010 to 0.0563** bits/token over
15 contrasts, so 0.0490 sits in the middle-to-upper part of it and the minimum is **0.0010**
(Llama `isvd_r256_tol` − `isvd_r256_qr64`). The row keeps its substantive point — it is the
smallest paired SD among the contrasts where a knob that *changes the subspace* is the difference
— and nothing downstream reads "floor" as a quantity: §6's bounds are computed from the stated
range, not from that label. The body is not edited; this note is the correction.

## Amendment 1b (2026-09-21, before the Stage-1 launch commit; after `gate1_preflight_rerun` harvested)

The return Amendment 1a and `prereg/gate1_preflight.md` A1.5 promised: the pre-flight's three
returns to this file — the §2 measured baseline rows, the `vt` exclusion, and the §9 re-size — plus
the `EXCLUDED_TASKS` value A1a.8 left to this amendment and the A1a.9 attribution correction. It is
read off the two committed pre-flight pods and **changes no rule, threshold or branch**: §4 stands
branch for branch, §6's three families keep their composition (now at the smaller size the `vt`
exclusion sets), §9's bars do not move. No number here is cited as a result; `make tables` reads
neither directory; the Gate-1 verdict is Stage 1's, at n = 24 over two families.

Evidence, both `scripts/pod.py check` = OK: `results/gate1_preflight/` (instance 51722149, the
**complete** `full` arm; the partial capture is D-011 addendum 10) and `results/gate1_preflight_rerun/`
(instance 51815080, the complete `isvd`/`nogist`/`frozen` arms; harvested and committed at **07c4f8d**).

### A1b.1 The measured real-text baseline (the §2 (a) rows Amendment 1 promised)

`ruler_v2_16k` (generator v2, real documents), Llama-3.1-8B, 16 384 ctx, seed 0, n = 12, 0 errors.
`full` from `results/gate1_preflight/`; the three compressed arms from `results/gate1_preflight_rerun/`,
which **supersede cell by cell** the two partial `isvd` cells and the one `frozen` cell that survived
the first pod. `niah_multiquery` is run by the pod but is **not** in `TASK_ORDER` (`gate1.py:158`:
`niah_single`, `niah_multikey`, `niah_multivalue`, `vt`); it is descriptive, listed for completeness:

| arm (record key) | niah_single | niah_multikey | niah_multivalue | (niah_multiquery) | vt |
| --- | --- | --- | --- | --- | --- |
| `full` (ceiling) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 0.92 (11/12) | 0.75 (9/12) |
| `nogist_h2423` (byte-matched no-gist twin, (e)) | 1.00 (12/12) | 1.00 (12/12) | 0.92 (11/12) | 0.92 (11/12) | 0.83 (10/12) |
| `bugSseed-r64-h256` (`isvd_r64_h256_seed`, the reference (a)) | 0.83 (10/12) | 0.25 (3/12) | 0.25 (3/12) | 0.25 (3/12) | 0.42 (5/12) |
| `frozen_r64_h256_seed` (learn-then-freeze, (d)) | 0.42 (5/12) | 0.33 (4/12) | 0.25 (3/12) | 0.08 (1/12) | 0.33 (4/12) |

**What it predicts, stated as §2's header asks (and it is a prediction, not the verdict).** On the
three operative tasks that survive the `vt` exclusion (A1b.4): (i) **`nogist` ≈ `full`** — the
byte-matched no-gist twin retrieves at the ceiling (1.00 / 1.00 / 0.92), so on real text the **exact
tier, not the gist, is what retrieves** (the Week-12 mechanism, `docs/plan/DECISIONS.md`; consistent
with D-005's retirement of the cycled filler, under which `isvd` had read 1.00). (ii) **`isvd` ≪
`nogist` at matched stored bytes** on every task (0.83 / 0.25 / 0.25 vs 1.00 / 1.00 / 0.92): the
online-tracked gist adds nothing over the tier-only twin. (iii) `isvd` beats `frozen` on
`niah_single` only (0.83 vs 0.42), ties `niah_multivalue` (0.25) and loses `niah_multikey`
(0.25 vs 0.33) — so **no task separates `isvd` from *both* controls**, which is what §4 rule 1
requires for A/B. This pre-flight — one family, one seed, n = 12, and no perplexity or `fd` axis —
therefore points toward **Branch C or the perplexity route, not A/B on retrieval**. It is a baseline
and a power check, not a reading of §4: Stage 1 decides the branch at n = 24 over Llama **and** Qwen,
with the perplexity TOST and the `fd` control that this pod did not run (`gate1_verdict`, §4).

### A1b.2 Reading (i) completeness and (iii) the pairing invariant, across the two pods

**(i) Complete.** `gate1_preflight_rerun` carries all 15 cells × 12 = **180** records, 0 errors,
`scripts/pod.py check results/gate1_preflight_rerun` = OK; with the complete `full` arm surviving in
`results/gate1_preflight/`, the four-arm baseline above is complete on every operative task. (The
first pod's reading (i) **failed** on capture, D-011 addendum 10; this is the repair A1.4 pod's job,
done.)

**(iii) The pairing invariant holds across the two pods.** By `prereg/gate1_preflight.md` A1.4's
snippet, over the 60 `(task, seed, trial)` keys each arm shares with the first pod's `full`:
`bugSseed-r64-h256` **60/60**, `nogist_h2423` **60/60**, `frozen_r64_h256_seed` **60/60** identical
`prompt_sha256`, **0 disagree**, and **0 null** digests across both pods. The seeded generator
produced byte-identical prompts on the two hosts; the pods are **one experiment** and poolable, so
the surviving `full` arm is the ceiling the re-run's arms are read against (A1.4).

### A1b.3 Reading (iv): the measured rates, no trigger fires, §9 is not re-sized

From `results/gate1_preflight_rerun/manifest.json` `cell_elapsed_s` by `prereg/gate1_preflight.md`
§4 (iv)'s formula (Σ of an arm's five cell seconds ÷ 60 s/min ÷ 60 samples):

| arm | Σ cell_elapsed_s | min/sample | trigger | outcome |
| --- | --- | --- | --- | --- |
| `bugSseed-r64-h256` (`isvd`) | 14 189.1 s | **3.94** | > 4.7 | **not fired** |
| `nogist_h2423` | 10 101.1 s | **2.81** | > 3.1 | **not fired** |
| `frozen_r64_h256_seed` | 5 735.7 s | **1.59** | > 3.1 | **not fired** |

No trigger fires, so **§9's per-pod bar stays 82 h and Stage 1 is not re-sized.** Two caveats.
The `isvd` anchor, 3.94, is above §9's stated 3.7 sensitivity top (the D-005 bracket was 2.9–3.7):
the compute per pod rises to ≈ 50 h and the point estimate (compute + the 60 min overhead) to ≈ 51 h
(the `isvd`-proportional arms scale by 3.94 / 3.1), still ≈ 1.6× inside the 82 h bar and well under
the 4.7 trigger (1.5 × 3.1) that would re-size it. And
`nogist` at 2.81 — the rate §7 named reading (iv)'s "first suspect" because its 2 423-token tier is
re-scored every absorb — sits comfortably under its 3.1 trigger; it is *not* the anomaly the caveat
anticipated. `frozen` at 1.59 is faster than its derived 2.1.

### A1b.4 §6 task exclusion: `vt` leaves the primary and secondary families (D-018)

The pre-flight's `full` ceiling on generator v2's `vt` was **9/12 = 0.75 < 0.9**
(`prereg/gate1_preflight.md` §4 (ii); reading (ii) is **decided** and not re-read on the re-run), and
§4 (ii) pre-committed that a task below the ceiling is a generator finding, not a compression one.
`docs/plan/reports/vt-template-comparison.md` returns Verdict B. Quoting its §7 verbatim:

> `vt` leaves the Gate-1 primary retrieval family (16 → 12) and the secondary family (24 → 18) under
> `prereg/gate1_preflight.md` §4 (ii), because the pre-flight `full` ceiling was 9/12 = 0.75
> (RULER-comparable `string_match_all` 0.867, Wilson 95 % [0.758, 0.931]) against third-party
> official-RULER runs that put full-attention Llama-3.1-8B at vt ≈ 99.6 at 16K (arXiv:2602.05191
> Table 4; arXiv:2510.05688 Table 4 gives 97.4 at 32K).
>
> `docs/plan/reports/vt-template-comparison.md` attributes the shortfall to `kvdlra.eval.gen` and
> not to the checkpoint: v2's mod-1 depth wrap presents the assignment chain out of order at depths
> 0.40 and 0.95 — where official RULER is always definition-first — and v2 additionally omits
> RULER's mandatory one-shot example, may draw the value from the `words` family, and ends each
> chain statement without a period; the repair required by §4 (ii) is scoped to those items and
> validated by one `full`-arm `vt` cell at ≈ 1 GPU-h.
>
> Until that repair lands and is measured, Stage 1 runs `vt` on the unrepaired generator and reports
> it descriptively (D-018), its rows are read only against the pre-flight ceiling this amendment
> records, and no `vt` row from before the repair is pooled with one from after it.

**The mechanism, in code.** `EXCLUDED_TASKS` — the `frozenset()` A1a.8 shipped empty — is set to
`frozenset({"vt"})` in `src/kvdlra/eval/gate1.py`, in the commit that lands this amendment. `vt`
then leaves the primary retrieval family, the secondary family and both of C's per-task readings,
and the realised Holm sizes are **primary retrieval 12, secondary 18, perplexity 4** (the perplexity
family has no task axis and is untouched; `bf16` non-inferiority, `prereg/bf16_gist.md`, drops from 8
to 6 the same way). `vt`'s cells are still run, still scored and still **rendered descriptively** in
the Gate-1 table, and a separation on `vt` blocks nothing (`test_an_excluded_task_leaves_every_family_and_blocks_nothing`,
`test_vt_is_excluded_from_the_gate1_families_by_default`).

### A1b.5 The A1a.9 attribution correction

A1a.9 named the §2 (b) population minimum (paired arm-vs-arm per-window SD, 0.0010 bits/token) as
**Llama `isvd_r256_tol` − `isvd_r256_qr64`**. That contrast's SD is **0.0016**, not 0.0010 — it is
§2 (b)'s own named row (l. 176). Recomputing every guard-on compression-arm-vs-compression-arm
contrast from the three Table-4 pods' `pplw.jsonl` by §2 (b)'s snippet, the true minimum **0.0010**
is **Qwen `isvd_r256_f0.01_qr64` − `isvd_r256_f0.01_tol`** on `hygiene_table4_qwen_r256` (mean
−0.0003, 16 windows; both arms guard-on, differing only in the repair schedule). The population
**range 0.0010 – 0.0563** and the §6 bound it feeds (a ±0.02 TOST at n = 32 decidable for SD <
0.0667, margin 1.18 at the 0.0563 maximum) are **unchanged**: only A1a.9's parenthetical attribution
of the minimum moves, from a Llama row that is 0.0016 to the Qwen row that is 0.0010.

### A1b.6 What does not change

§4's rule and its precedence (A/B, C, REFUSED, UNDECIDED), the ±0.02 margin and the exclusive
A/B-vs-C construction; §6's composition (now realised at 12 / 18 / 4 with `vt` excluded, and the
per-family refusal rulings R-L3-15/16); §9's 82 h bar and its cut ladder; the verdict signature
(A1a.2), the per-member `prompt_sha256` drop (A1a.1) and the completeness/`NO_MEMBERS` blockers
(A1a.5). The two pre-flight pods enter Stage 1's reading as the **measured baseline only** — the
ceiling the descriptive rows and the byte match are read against — and no statistic in §4 or §6
crosses from them into a Stage-1 pod.

**STATUS: committed before the Stage-1 launch commit. Stage 1 (`gate1_v2_stage1_llama`,
`gate1_v2_stage1_qwen`) launches from `main` after this branch merges, under D-003.**
