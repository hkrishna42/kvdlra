# Pre-registration — `gate1_preflight`

**STATUS: awaiting launch (DECISIONS, under D-011).** Written before the pod is launched. Launch
is a separate, later commit that spends money and is recorded in `docs/plan/DECISIONS.md`:
`scripts/pod.py launch` refuses unless this file's first commit is a *strict* ancestor of the
launch commit, so the commit that adds this file launches nothing. The authorization is D-011's
standing one (2026-09-17: "Pod launches inside the approved lanes are authorized; each launch is
still recorded here with pod name, prereg SHA, launch SHA, budget, and the harvest outcome") —
lane L3, Task 1b. This pod is **not** a Gate-1 pod: it protects one.

---

## 1. Purpose

Gate 1 Stage 1 (`docs/plan/lanes/L3_gate1_tracker_swap_v2.md`, `docs/plan/ICML2027_PLAN.md` §2
Gate 1, `docs/plan/lanes/GATES.md` §G3) is two pods — Llama and Qwen, eight arms each, the Gate-1
tasks at n = 24 plus 32 paired perplexity windows — whose readings are paired contrasts on
generator v2. Its size, at §7's rates — each arm **128 samples** (4 × 24 + 32 windows):

| §7 rate (min/sample) | arms at it | minutes |
| --- | --- | --- |
| 0.6 | `full` | 128 × 0.6 = **76.8** |
| 3.1 | `isvd_r64_h256_seed`, `isvd_r64_h256_seed_bf16`, `fd_r64_h256_seed`, `oja_r64_h256_seed` | 4 × (128 × 3.1 = 396.8) = **1,587.2** |
| 2.1 | `frozen_r64_h256_seed`, `random_r64_h256_seed` | 2 × (128 × 2.1 = 268.8) = **537.6** |
| 1.55 | `nogist_h2423` (Llama) / `nogist_h4460` (Qwen) | 128 × 1.55 = **198.4** |
| | **compute per pod** (8 arms) | 76.8 + 1,587.2 + 537.6 + 198.4 = **2,400 min = 40.0 h** |

Plus §7's 60 min boot → **≈ 41 GPU-h per pod**; two families (Llama, Qwen) → **82 GPU-h at the
point estimate, 164 at the 2× bar**, and at the $0.45–0.74/h of §7 **$37–61 point, $74–121 at the
bar**. The perplexity task is `ppl_16k_pg19val` (32 windows, not 16): at 16 windows the ±0.02
TOST is undecidable at the measured Qwen spread (`prereg/gate1_tracker_swap_v2.md` §6). This is
the sizing L3 Task 3 commits with the Stage-1 pod YAMLs and
`prereg/gate1_tracker_swap_v2.md` §9 carries, and it **supersedes** the lane plan's "≈ 41 GPU-h;
with the 2× safety factor 82 h" (`docs/plan/plans/2026-09-11-L3-L5-gate1-bf16-prereg.md` "Pod
sizing"), which predates the L2 pods' rate measurements (§7) — as it supersedes the lane file's
"≤ 50 GPU-h", written against those same pre-L2 rates; the owner sees the cut ladder in that
prereg. The four tasks are the Gate-1 family of `GATES.md` §G3, which L3 Task 3 edits
`configs/tasks/ruler_v2_16k_g1.yaml` down to (the shipped file still lists five — §3); the table
above is over four, and Task 3 re-derives it with the YAMLs if that changes. **This pod's own
task keeps all five.** Two facts make launching that bar without a pre-flight a bad bet:

1. **Generator v2 has never run on a GPU.** The only pods on the L0 runner are the three Table-4
   pods (perplexity, D-011 addendum 8) and the two filler-realism pods (the *in-house* generator,
   D-005) — `prereg/l2_smoke.md` §1, still true: the smoke pod that would have shown generator v2
   running end to end is written and **not launched** (its STATUS is "awaiting owner go"; its bar
   of 168 GPU-h waits on D-003). Generator v2's `vt` — official RULER's template and its 4-hop
   chain — and its fifth task `niah_multiquery` have never been scored at all, on any hardware.
2. **The `vt` task has already collapsed once, uncompressed.** On real-text filler the *v1* `vt`
   ceiling fell to **0.08 (1/12) with the cache intact** (`full`; DECISIONS D-005 addendum 2,
   harvested row in §2 below). The hypothesis recorded for it (`prereg/filler_realism.md` A2.6) is
   a generator/template interaction, not a model limit — the cycled control put the same cell at
   12/12. A task whose ceiling is 0.08 measures nothing about compression: a Gate-1 family that
   contains one spends a quarter of its retrieval cells on a member that cannot separate anything.

**This pod is the pre-flight and nothing else.** It runs the Gate-1 primary-contrast arms — the
r64 configuration, its byte-matched no-gist control and its learn-then-freeze control — under the
uncompressed ceiling, on `ruler_v2_16k` (five tasks, n = 12, one seed) on Llama-3.1-8B at 16K, and
is read for five things, fixed in §4: (a) completeness, (b) the `full` ceiling **per task**,
(c) the pairing invariant, (d) the measured min/sample per arm that decides whether Stage 1 is
re-sized before its launch commit, and (e) descriptive per-task rows that
`prereg/gate1_tracker_swap_v2.md` quotes in its §2 **by a dated Amendment 1** as its measured
real-text baseline.

**The order the lane runs in, which is what makes (d) and (e) pre-registrations and not
hindsight.** `prereg/gate1_tracker_swap_v2.md` §1–§11 — its arms, its family, its decision rule,
its predictions and its §9 sizing — are **written and committed BEFORE this pod's launch
commit**, with no row of this pod in hand; its predictions are written from the evidence of §2
below, which is on disk and committed already (D-005). Everything this pod sends that file is a
**dated amendment** committed before the *Stage-1* launch commit: Amendment 1 for the descriptive
rows (§5), the §4 (ii) task exclusion if the ceiling fires, the §9 re-sizing if a §4 (iv) trigger
fires. So no part of the Gate-1 body can be authored around what this pod returned, and every
part that reacts to it is dated and visible as a reaction.

**No number from this pod is cited as a result** — not in the paper, not in a table, not as a
Gate-1 outcome. It decides nothing about the tracker (§9). Its accuracy rows exist so that the
Gate-1 prereg's already-written predictions can be **restated by amendment against measured rows
on the arms Stage 1 actually runs** — its controls included, which the D-005 pod never ran —
rather than left resting only on the four-arm real-text rows of §2a.

## 2. The evidence on disk that sets the expectations

Every number in this section was computed from the committed records by the snippet beside it, at
the commit that adds this file — none is copied from a table or from memory.

**(a) The real-text rows that closed D-005** — `results/filler_realism/` (instance 51553610, launch
SHA 56e89ee, `ALL_DONE`, 240 records, 0 errors, harvested 97fd76e; D-005 CLOSED 2026-09-19). The
*v1 in-house* four tasks at 16K on WikiText-2 sentences — not generator v2 — hits/12:

```python
import json, collections, pathlib
rows = [json.loads(l) for l in pathlib.Path("results/filler_realism/trials.jsonl").read_text().splitlines()]
cells = collections.defaultdict(list)
for r in rows:
    cells[(r["arm"], r["task"])].append(r)
for (arm, task), rs in sorted(cells.items()):
    print(arm, task, f"{sum(r['hit'] for r in rs)}/{len(rs)}", "errors", sum(1 for r in rs if r["error"]),
          "shas", len({r["prompt_sha256"] for r in rs}))
```

| arm (record key) | single | multikey | multivalue | vt | errors |
| --- | --- | --- | --- | --- | --- |
| `full` | 12/12 | 12/12 | 11/12 (0.92) | **1/12 (0.08)** | 0 |
| `bugSseed-r64-h256` (= `isvd_r64_h256_seed`) | 3/12 (**0.25**) | 1/12 (**0.08**) | **0/12** | 0/12 | 0 |
| `bugSseed-r64-h256-q4` | 0/12 | 0/12 | 0/12 | 0/12 | 0 |
| `quant-2bit-kivi` | 12/12 (1.00) | 6/12 (0.50) | 6/12 (0.50) | 1/12 (0.08) | 0 |
| `quant-2bit-kivi#chunk0` | 12/12 (1.00) | 10/12 (0.83) | 5/12 (0.42) | 2/12 (0.17) | 0 |

**(b) The cycle control** — `results/filler_realism_cycle/` (51559661, launch SHA a5cd89c, 96
records, 0 errors; D-005 addendum 3), the same snippet: `bugSseed-r64-h256` 12/12 / 12/12 / 12/12 /
8/12 (0.67) and `full` 12/12 on all four, against the archived `w18-g1-llama` row 1.00 / 1.00 /
1.00 / 0.58. Every task within 0.25 → `prereg/filler_realism.md` A1.3: **the L0 runner reproduces
`w10_ruler.py`**, and harness drift is excluded as an explanation for anything below. That pod also
carries the only hardware verification of the pairing invariant: 48 `(task, seed, trial)` keys, 12
distinct `prompt_sha256` per cell, **0 keys where the two arms disagree**.

**The real-text pod cannot make that check**: it ran a pre-L2.3b SHA and every one of its 240
records carries `prompt_sha256: null` (`len({r["prompt_sha256"] for r in rows}) == 1`, the single
value being `None` — the `shas 1` column above). The pairing invariant has therefore been verified
on hardware **once**, on two arms, on the cycled filler. Reading (iii) of §4 is where it is
verified on generator v2, on four arms, with the Gate-1 controls among them.

**(c) What is expected of the gist arms on generator v2 — low, and not read as a result.**
`prereg/l2_smoke.md` §2 and §4 reading 4 pre-register exactly this for every gist arm on this same
task file: generator v2's haystacks are real documents from four sources under official RULER
templates, i.e. harder filler than WikiText-2 sentences by construction, so the r64 arm is expected
at or below its 0.25 / 0.08 / 0.00, and "a 0/12 cell is a *pass*". The same holds here.

**(d) No `nogist` or `frozen` row exists anywhere.** Both arms were created on this branch at
7f3e546 (L3 Task 1) and have only CPU tests behind them (`tests/test_gate1_arms.py`): no tracker
other than `isvd`, `oja` and (aborted) `fd` has ever run on a GPU in this repository.

```python
import json, pathlib
arms = {json.loads(l)["arm"] for p in pathlib.Path("results").rglob("trials.jsonl")
        for l in p.read_text().splitlines()}
print(len(arms), sorted(a for a in arms if "fro" in a or "nog" in a or "oja" in a))
# 22 ['bugSseed-r64-h256-oja']   (the Week-20 swap pod's void Oja cell; CLAUDE.md settled facts)
```

So §5 states **no accuracy prediction** for those two arms. They have no prior anywhere, which is
one more reason this pod exists before the pods whose decision rule reads them.

## 3. Arms, tasks, n

Four arms, in this order (`configs/pods/gate1_preflight.yaml`, unedited): the cheap uncompressed
ceiling first, then the two arms of the Gate-1 **primary contrasts** (`isvd` vs `nogist`,
`isvd` vs `frozen`), then `frozen` — so a pod that dies early still lands the ceiling and one
whole primary contrast (§7 prices the cut points). The name, kind, prefill and record-key columns
were printed from the configs (`load_arm` + `frontier.build_arm` at t = 16384 over
`load_pod("gate1_preflight").arms`), not typed:

| # | arm config | `kind` | prefill | record key | the cache, in one line |
| --- | --- | --- | --- | --- | --- |
| 1 | `full` | `full` | single-shot (one forward, `DynamicCache`) | `full` | uncompressed |
| 2 | `isvd_r64_h256_seed` | `bug` | chunked 4096 | `bugSseed-r64-h256` | rank 64, `tracker: isvd`, 256-token surprise tier, 4 sinks, ring 32, warm-up seed |
| 3 | `nogist_h2423` | `bug` | chunked 4096 | `nogist_h2423` | rank **1**, `coord_budget` **1**, **2423**-token surprise tier, same sinks / ring / seed |
| 4 | `frozen_r64_h256_seed` | `bug` | chunked 4096 | `frozen_r64_h256_seed` | arm 2 verbatim with `tracker: frozen`, `freeze_after: 4096` |

Arms 3 and 4 are each arm 2 with **one mechanism removed** — not one knob: arm 4 moves two
(`tracker`, `freeze_after`) and arm 3 moves three (`rank` 64 → 1, `coord_budget` null → 1,
`hh_budget` 256 → 2423), which is what removing the gist while keeping its bytes costs. One
mechanism at a time is what makes the Gate-1 cells mechanism comparisons rather than budget
comparisons:

- **`frozen_r64_h256_seed`** — same rank, same 256-token tier, same seed, same bytes; incremental
  SVD over the first 4096 tokens, then the basis is frozen and later tokens are plain projections
  (xKV / ShadowKV-style). A cell against arm 2 measures the **online tracking** and nothing else.
- **`nogist_h2423`** — tier + ring only, with the gist's bytes given to the tier. `H` is solved on
  **stored** bits (`kvdlra.accounting.Footprint.stored_bits`, the at-rest billing CLAUDE.md §4.1
  requires) at t = 16384 for a 1024-wide layer (Llama-3.1-8B: 8 KV heads × 128), against arm 2:
  arm 2 (rank 64, 16,092 coordinate columns, ring 32, 4 sinks, 256 exact) bills **80,717,568
  bits/layer**; this arm bills `1,245,376 + 32,800·H` (2·n·16 = 32,768 fp16 K+V bits per exact
  token, + 32 for its position word), so **H = 79,472,192 / 32,800 = 2422.93 → 2423**, a
  **1.00003×** match (80,719,776 / 80,717,568). The arithmetic is the arm file's `doc:`
  (`configs/arms/nogist_h2423.yaml`) and is pinned within 5 % by
  `tests/test_gate1_arms.py::test_the_nogist_arms_are_byte_matched_to_isvd_r64_at_16k`. That H is
  2.4× the ICML plan's "h ≈ 1024", which was an fp16-equivalent estimate; the fp16-equivalent
  solve on the same inputs gives 1355. The tier still selects by residual, but against a rank-1
  basis — approximately norm-ordering, which is what "no gist" means for a surprise tier. The
  512-wide twin `nogist_h4460` (Qwen) is **not** in this pod: this pod runs one family.

| pod | model | dtype / image | task | cells |
| --- | --- | --- | --- | --- |
| `gate1_preflight` | `unsloth/Meta-Llama-3.1-8B-Instruct` | bfloat16, `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` | `ruler_v2_16k` | 4 arms × 5 tasks = **20** |

**Task and n.** `configs/tasks/ruler_v2_16k.yaml`, unchanged and unedited — the file the ss2 pods
(under their filler condition's branch 3) and the smoke pod will run, so its cells are validated
for them too, `niah_multiquery` included: `generator: v2`, `ctx: 16384`, the five sub-tasks
`niah_single`, `niah_multikey`, `niah_multivalue`, `niah_multiquery`, `vt` (official RULER needle /
question / answer-prefix templates; `vt`'s value takes the cell's code family — ruling R-L2-4),
haystacks from `pg19`, `arxiv`, `wikipedia`, `essays` (64 documents each, materialized by `pod.py
run` before the model loads, digests into `manifest.dataset_sha256`), `design: {haystacks: 2,
depths: 3, codes: 2}` → `n_trials: 12` at depths 0.05 / 0.4 / 0.95, `seeds: [0]`, `chunk: 4096`.
**12 records per (arm, task) cell, 60 samples per arm, 240 in the pod**; `pod.py check` requires
all 20 cells at exactly 12, errors counted (a trial that raises is a record with `error` set and
`hit = 0`, counted in n — ruling R29). `MAX_NEW` = 48 for the four `niah_*` tasks and 64 for `vt`
(`gen.MAX_NEW`).

**Gate-1's own family is four of these five.** `docs/plan/lanes/GATES.md` §G3 and
`ICML2027_PLAN.md` §1.1 name `niah_single`, `niah_multikey`, `niah_multivalue` and `vt`; the
shipped n = 24 Stage-1 task file `configs/tasks/ruler_v2_16k_g1.yaml` still lists all five, and
which of them Stage 1 runs is settled by L3 Task 3 with the Stage-1 pod YAMLs and their
pre-registration — not here. `niah_multiquery` rides this pod because `ruler_v2_16k` carries it
and the ss2 and smoke pods need its cells validated; a multiquery finding here is reported to
those pods and to that decision, not to Gate 1.

**Pairing.** `gen.make_trial` is deterministic in `(task, seed, trial)` and never sees the arm, so
all four arms are fed the same token ids for a given key; the runner's `[trial]` line and record
carry `prompt_sha256` over exactly those ids (post-L2.3b), so the pairing is *verified* from the
records (§4 reading (iii)), not assumed. Whether an arm prefilled the ids in one shot (`full`) or
in 4096-token chunks (arms 2–4) does not enter the digest.

## 4. Readings, fixed now

Four readings, each a **pass / fail** on the harvested records — readings (i)–(iii) on
`trials.jsonl`, reading (iv) on the manifest's `cell_elapsed_s`; they are independent and the pod
is "validated" only when all four pass. Nothing here is a hypothesis test and no correction
applies. "The records" are `results/gate1_preflight/{trials.jsonl, manifest.json}` as
`pod.py harvest` writes them from the deduped `<label>.log`, with `pod.py check` run on the
directory. Every reading is decidable **after the fact, from the harvested directory alone**: no
reading depends on anyone watching the run.

**(i) Completeness.** All 20 `(arm, task)` cells hold exactly 12 records, `sum(1 for r in rows if
r["error"])` = 0, `grep -c '^\[trial\]' <label>.log` = 240, and `scripts/pod.py check
results/gate1_preflight` returns 0 (config hash, commit order, cell counts, `env.txt` against the
pyproject pins). *Fail* names the arm(s) and the exception text(s) verbatim (the row carries
`"<ExcType>: <message>"`). An arm whose trials raise is an arm that does not run on the pod path —
the finding this pod exists to produce, reported to lane L3, **never worked around on the pod**:
nothing is re-run with a knob changed, and a fix is a later commit and a later pod.

**(ii) The `full` ceiling, per task: ≥ 0.9 (≥ 11/12) on every one of the five tasks.** A task below
that is a **generator finding, not a compression one**, and two things follow, both
pre-registered here:

- the task is **excluded from the Gate-1 primary Holm family** by an amendment to
  `prereg/gate1_tracker_swap_v2.md` — dated, appended to a body already committed before **this**
  pod's launch commit (§1), and itself committed before the **Stage-1** launch commit — which
  re-states the family size and the per-cell resolution at the smaller family, in
  the pattern of `prereg/filler_realism.md` Amendment 2 (which took `vt` out of the D-005 rule for
  exactly this reason at exactly this threshold); and
- it is **fixed in `kvdlra.eval.gen` before any pod runs it again** — the excluded task does not
  simply drop out of the paper; it is a defect with an owner.

**`vt` is the task this rule is written for** (§1, §2a), and `niah_multiquery` is the task nobody
has ever scored. A ceiling at 12/12 on all five is the outcome that leaves the Gate-1 design as
`prereg/gate1_tracker_swap_v2.md` will write it.

**(iii) Pairing: `prompt_sha256` identical across the four arms for every `(task, seed, trial)`.**
5 tasks × 1 seed × 12 trials = **60 keys**, each holding 4 records with one digest:

```python
import json, collections
rows = [json.loads(l) for l in open("results/gate1_preflight/trials.jsonl")]
sha = collections.defaultdict(set)
for r in rows: sha[(r["task"], r["seed"], r["trial"])].add(r["prompt_sha256"])
bad = {k: v for k, v in sha.items() if len(v) != 1 or None in v}
print(len(sha), "keys;", len(bad), "disagree;", sorted(bad)[:5])
```

Expected `60 keys; 0 disagree`. A key whose set is `{None}` or contains `None` is a *fail* as well
as a mismatch: this pod runs post-L2.3b code, where the digest is written for every record, and a
missing digest is a runner defect (the D-005 real-text pod's records, §2b, are what that looks
like). *Fail* names the keys and the arms whose digest differs — every paired statistic in Gate 1
rests on this invariant, and Stage 1 cannot launch on a path where it is broken.

**(iv) Budget trigger: the measured min/sample per arm, read from `manifest.cell_elapsed_s`.**
Stage 1 is sized from *derived* rates for three of its arms (§7; the no-gist and frozen rates are
the plan's factors over the measured r64 rate, not measurements). The measurement, exactly:

> **min/sample for an arm = (Σ of its five `cell_elapsed_s` values, seconds) ÷ 60 s/min ÷ 60
> samples = Σ ÷ 3,600.**

`_cell` (`src/kvdlra/eval/runner.py`) prints `[stage] cell arm=<arm> task=<task> ctx=<ctx>
elapsed_s=<s> n=<records>` after every completed cell, timed with `time.perf_counter()` around
that cell's trials; the watchdog's `ROWS` filter keeps `^\[stage` rows; `scripts/pod.py harvest`
(`CELL_S_RE`) folds them into `manifest["cell_elapsed_s"]` as `{"<arm>/<task>/<ctx>": seconds}`,
20 entries for this pod's 20 cells. Because each line carries its own seconds, the watchdog's
per-poll `sort -u` — which destroys arrival order — cannot damage the reading, and **nobody has
to be watching the run**. Shipped on this branch at **03fba42** (`L3.1c`, an ancestor of the
launch commit) and pinned by `tests/test_pod_run_records_errors.py` (the runner prints one line
per cell) and `tests/test_pod_manifest.py::test_harvest_records_the_cell_timings_the_run_printed`
(the harvest carries them into the manifest).

**The triggers.** If any of

| arm | budgeted | **trigger** | what the trigger means |
| --- | --- | --- | --- |
| `isvd_r64_h256_seed` | 3.1 | **> 4.7** (1.5 × 3.1) | the one *measured* anchor is wrong on generator v2 — every Stage-1 arm's rate moves with it |
| `nogist_h2423` | 1.55 | **> 3.1** (2 × 1.55) | the "2× faster, no gist rebuild" factor does not hold — re-scoring a 2423-token tier every absorb is the suspected reason (§7) |
| `frozen_r64_h256_seed` | 2.1 | **> 3.1** | the 1.5× factor does not hold (3.1 is `isvd`'s own rate: frozen would be no cheaper than the arm it controls) |

fires, **the Stage-1 pods are re-sized from the measured rates before their launch commit** — a
dated amendment to `prereg/gate1_tracker_swap_v2.md` §9 (a body committed before *this* pod's
launch commit, §1) re-deriving the table and the `gpu_budget_h` of each pod YAML in the commit
that precedes the Stage-1 launch — and they are **never launched over their pre-registered bar**.
A trigger is not a failure of this pod; it is the pod doing its job.

**Fallback, pre-committed now, if the harvested log carries no `[stage] cell` lines** (the pod
ran an older SHA, or the fetch lost them — the manifest then has no `cell_elapsed_s` key at all):
every per-arm trigger above is recorded **`not measured`** in the DECISIONS entry, never as
"passed", and Stage 1 is re-sized from the pod's **aggregate** rate instead. That rate is
`manifest.launched_at` → the harvest time of the `===ALL_DONE_…===` marker
(`manifest.harvested_at`) **minus §7's 60 min boot**, apportioned by §7's per-arm shares:
κ = (that compute, in minutes) ÷ 441, and each arm's rate = κ × its §7 rate (0.6 / 3.1 / 1.55 /
2.1 min/sample). κ > 1 re-sizes Stage 1 upward by the same rule a trigger would; κ ≤ 1 leaves
§7's table standing. Resolution: on the D-005 real-text pod `launched_at` → `harvested_at` reads
6.99 h against the ≈ 6.8 h billed — a ≈ 0.2 h harvest lag on a ≈ 7 h pod. What the fallback
cannot do is tell one arm's rate from another's: it can only scale all four together, which is
why it is the fallback and the `[stage] cell` line is the source.

**Consistency check when the table is written:** four lines, each `pass` or `fail`, none blank; a
*fail* on (i) makes (ii) and (iv) unreadable for the affected arm and is written as
`fail (not reached)` there, never as pass. The one non-`pass`/`fail` entry the table allows is
reading (iv) under its fallback above — `not measured`, with κ and the re-sized table beside it —
and it is never abbreviated to `pass`.

## 5. Expectations, descriptive

No accuracy threshold is pre-registered for arms 2–4, and **nothing here decides Gate 1**. The
Gate-1 prereg's own predictions are **not** written from these rows: its body is committed before
this pod's launch commit (§1) and predicts from the D-005 real-text evidence of §2a. What follows
is expected here, so that a surprise is recognisable, and is what Amendment 1 restates against
the arms Stage 1 runs:

- **`full`**: the ceiling, reading (ii). 12/12 on the four `niah_*` tasks is what the D-005
  real-text pod produced on three of its four (1.00 / 1.00 / 0.92) with the cache intact; `vt` and
  `niah_multiquery` are unknown.
- **`isvd_r64_h256_seed`**: **near the floor on single / multikey / multivalue** — the D-005
  reading (§2a, §2c) puts it at or below 0.25 / 0.08 / 0.00 on harder filler under official RULER
  templates. A 0/12 cell is not a finding of this pod; it is the pre-registered expectation, and it
  is what Gate 1 is designed to interrogate at n = 24 against controls.
- **`nogist_h2423`, `frozen_r64_h256_seed`**: **no prediction** (§2d — no row exists anywhere).
  Whichever way they land relative to arm 2, the contrast is Stage 1's to make at n = 24 with a
  Holm correction, on two families, not this pod's at n = 12 on one.
- **The `frozen` arm's `[diag]` rows are expected to show no repairs after the freeze.**
  `_guard_orthonormality` runs on every absorb for every tracker (`src/kvdlra/cache/bug_cache.py`,
  the call after the tracker dispatch), and past `freeze_after` = 4096 `frozen_step` returns `U`
  unchanged with `rot = I`, so the measured ‖UᵀU − I‖_F stays at the value the last repair left and
  `fixed_k` / `fixed_v` should read **false in every window past the freeze** — against the r64
  arm's measured `fixed_k` 100.0 % / `fixed_v` 99.2 % (19,968 rows, `results/filler_realism/
  diag.jsonl`; pre-repair `orth_err_k` min 1.00e-3 / median 1.02e-3 / max 1.41e-3, `orth_err_v` max
  8.14e-3, 0 rows within an order of magnitude of the 1e-1 abort). A frozen arm still repairing
  after the freeze is a **dispatch defect**, reported as such to lane L3.
- **Ratio / stored bits**: `ratio` and `sbits` are recorded on every row (the accounting ran) and
  are the on-pod confirmation of §3's byte match — arms 2 and 3 should print `sbits` within ≈ 0.1 %
  of each other, and arm 4 within the same margin of arm 2 (all three bill the *live tracked* rank,
  D-015, which is the rank cap at these settings). A wider gap is an accounting finding, reported.

**Abort and error are possible outcomes, not accidents.** An `OrthonormalityError` on any gist arm
(the guard could not restore the basis after a repair), an OOM in `full`'s single-shot 16K prefill,
a `LinAlgError` the fallback does not catch: each is an `error` row, counted in n, failing reading
(i) and naming its arm. Nothing is re-run on the pod with a knob changed.

## 6. Log volume, and what counts as a complete `<label>.log`

The pod log is the only channel back from a vast.ai instance (`results/<pod>/` dies with it). Rows
per sample: one `[trial]` line per record, one cell row per `(arm, task)`, and for each of the
three `bug`-kind arms the `[diag]` rows `records.drained` prints when the trial's cache is drained
— `full` emits none.

**None of the three gist arms sets `diag_every`, so the cache default (64) applies**, and no
`_diag4096` variant is created (the question `prereg/ss2_families.md` §8 and `prereg/l2_smoke.md`
§6 worked through; the arithmetic below reaches the same conclusion with a wider margin). The
per-sample count is **measured**, not estimated — and now from harvested records rather than a live
capture: `results/filler_realism/diag.jsonl` holds 39,936 rows over 48 samples × 2 gist arms at
this model and context, i.e. **416 `[diag]` rows per 16K sample** (= 13 rows per layer × 32 layers:
twelve completed 64-absorb windows plus `drain_diag`'s flush of the open one, at 804–810 absorbs
per layer), identically for both arms:

```python
import json, collections, pathlib
d = [json.loads(l) for l in pathlib.Path("results/filler_realism/diag.jsonl").read_text().splitlines()]
per = collections.Counter((r["arm"], r["task"], r["idx"]) for r in d)   # idx = trial; two seeds per idx
print(len(d), collections.Counter(r["arm"] for r in d), sorted(set(per.values())), max(r["absorbs"] for r in d))
# 39936 Counter({'bugSseed-r64-h256': 19968, 'bugSseed-r64-h256-q4': 19968}) [832] 810   (832 = 2 seeds x 416)
```

The absorb schedule is set by the prompt length and the cache's block sizes — the first 4096-token
chunk in 128-column sub-blocks (32 absorbs) and the remaining ≈ 12.3 K tokens in `absorb_block` = 16
column blocks — and `_prefill` / `consolidate` are shared by every `retention`, `tracker` and rank,
so **416 is the anchor for all three gist arms here**. Arm 3 plausibly emits *fewer* (its
2423-token exact tier holds columns the gist never absorbs); **544 = 17 rows per layer, the naive
1,025-absorb count, is the bound** used for the margin below. v2 prompts are the same length as the
in-house ones (`gen` stops at ≥ 16,384 haystack tokens plus needles and template).

| the whole pod | rows |
| --- | --- |
| `[diag]`: 3 gist arms × 60 samples × 416 | **74,880** (≤ 97,920 at the bound) |
| `[trial]`: 4 × 60 | 240 |
| cell rows: 4 × 5 | 20 |
| `[stage]`: the 20 cell timings reading (iv) reads + four haystack digests, corpora, model load; ENV block, markers | ≈ 50 |
| **expected deduped `<label>.log`** | **≈ 75,200 lines** (≈ 98,200 at the bound; ≈ 21–27 MB at 275 B/row) |

(275 B/row is measured, not assumed: the live filler-realism pod's `.raw` read 78.5 MB over
286,000 lines on 2026-09-19 07:13 — `prereg/ss2_families.md` §8, "the cost of that choice,
stated". 75,200 × 275 B = 20.7 MB; 98,200 × 275 B = 27.0 MB. The 20 `[stage] cell` lines reading
(iv) needs are ≈ 90 B each — 1.8 kB of the total, and the reading fails only if the log is lost
entirely, which §4 (iv)'s fallback covers.)

**No `[diag]` row can be lost, by the arithmetic that decides it.** The only way the watchdog loses
a row for good is a **poll-to-poll gap**: more matched lines printed between two 150 s polls than
the `vastai logs --tail 30000` window holds (5,000 when the empty-fetch fallback is taken —
`scripts/pod/watchdog.sh`). A sample's `[diag]` rows are printed in one burst when its trial
drains, so the burst per poll is bounded by how many gist samples can *complete* in 150 s. The
fastest gist arm here is `nogist_h2423` at 1.55 min = 93 s per sample (§7), so **at most two**
samples close inside one poll: **≤ 2 × 416 + 2 = 834 rows per poll** (≤ 1,090 at the bound) — 36×
under the 30,000-line window and 6× under the 5,000-line fallback. The unfiltered log would have to
add > 4,000 lines in 150 s on top of the worst burst to open a gap.

**The raw-log growth `prereg/l2_smoke.md` §6 made a launch precondition is already shipped.**
`scripts/pod/watchdog.sh` now dedupes `<label>.raw` in place after every append (`sort -u
"$H/${lab}.raw" -o "$H/${lab}.raw"`, the L2 fix wave c437e75), so the raw stays at the size of its
distinct rows (≈ 21–27 MB here) instead of growing by the re-fetched tail each poll; the same fix
made the watchdog's harvest step use the repo venv's python. Neither is a waiver: both are on
`main` and in this branch's ancestry, and the launch entry names the watchdog SHA.

**The watchdog's own clock needs no override for this pod.** `BUDGET_ITERS` defaults to
`(gpu_budget_h · 3600 + 7200) / 150 + 1` polls with a floor of 600 — here 457 → **600 polls = 25 h**,
which covers the 17 h bar plus `boot.sh`'s 2 h `GRACE_S` self-destruct with 6 h to spare. (The
l2_smoke pod needs `BUDGET_ITERS=4100`; this one needs nothing on the launch line.)

**The completeness test is on the deduped `<label>.log`.** Exact: `grep -c '^\[trial\]'` = 240 and
`pod.py check` finds all 20 cells at n = 12 — that is the gate. Approximate: `grep -c '^\[diag'`
≈ 74,900 (± 32 rows per sample is expected noise: the per-layer window count moves by one when a
sample's absorb count crosses a multiple of 64, and v2 prompts vary by a few tokens across
haystacks; arm 3's own count is one of the things this pod measures), and `manifest.diag_skipped`
= 0 (a `[diag]` line a fetch cut in half is counted, not dropped, and fails `check`). A `.log`
whose `[trial]` count is short is short by construction — read `trials.jsonl` and the `error` lines
before concluding truncation.

## 7. Budget

**Unit and anchors.** "min/sample" is the wall-clock of one `(arm, task, seed, trial)`; a cell is
12 samples and the five-task context is **60 samples per arm** (the tasks-per-cell correction of
`prereg/filler_realism.md` §GPU budget). All rates on an A100 40 GB:

- **`full` 0.6 — measured.** The D-005 real-text pod's `full` arm ran its 48 samples in ≈ 30 min
  (`prereg/filler_realism.md` A2.5, on instance 51553610). The smoke and filler preregs budget
  `full` at 1.0; this pod uses the measurement, and the 2× bar absorbs the difference.
- **`isvd_r64_h256_seed` 3.1 — measured, inside its observed bracket.** The same pod's r64 arm ran
  48 samples in ≈ 2.3–3.0 h → **2.9–3.7 min/sample** (`prereg/l2_smoke.md` §7, from the D-005
  addendum 2 / 3 cell timestamps), against the 2.1 min/sample measured on the W19 `a1q` pod
  (`prereg/filler_realism.md` A1.4) that every pod since has budgeted at 3.0. 3.1 sits in the lower
  half of that bracket, so the sensitivity is stated rather than hidden: at the bracket's top (3.7,
  the two derived rates scaling with it to 1.85 and 2.47) the four arms cost 36 + 222 + 111 + 148 =
  517 min and the point estimate is 9.6 h — still 1.8× inside the 17 h bar. 3.1 is the rate this
  lane sizes Stage 1 with, and reading (iv) is what re-sizes it if generator v2 moves it.
- **`frozen_r64_h256_seed` 2.1 — derived, = isvd / 1.5.** No per-absorb SVD after the 4096-token
  freeze (the factor is the lane plan's, `docs/plan/plans/2026-09-11-L3-L5-gate1-bf16-prereg.md`
  "Pod sizing": "frozen/random ≈ 1.5× faster (no per-absorb SVD)").
- **`nogist_h2423` 1.55 — derived, = isvd / 2** (same source: "the no-gist arm has no gist rebuild
  and runs ≈ 2× faster"). **Caveat, and it is reading (iv)'s first suspect:** the 2423-token exact
  tier is re-scored every absorb, which the r64 arm's 256-token tier is not.
- **Overhead 60 min**: boot, clone, `pip`, the weight download that stalled ≈ 1 h on a Table-4 pod
  (D-011 addendum 2; `prereg/filler_realism.md` A1.4). Generator v2 adds two more items this pod
  does not bill separately, both taken from `prereg/l2_smoke.md` §7's overhead bullet: **≈ 10 min
  of haystack materialization** (four sources × 64 documents streamed from Hugging Face at the
  pinned revisions — its "+10 for haystack materialization", measured at 1.1–5.1 s per 2 documents
  on the L2.3b proof run, and the same four sources at the same 64 documents here) and **≈ 2 min
  of per-trial prompt construction** (its ≈ 0.5 s per `make_trial` — the sentence split plus ≈ 800
  tokenizer calls, not memoized across arms — which gave that pod 20 min over 2,400 samples;
  0.5 s × 240 samples = 120 s here). Those 12 min sit **inside the safety factor** rather than
  moving the bar, exactly as `prereg/filler_realism.md` A1.4 placed its added 35 min.

| # | arm | min/sample | × 60 | minutes | cumulative compute |
| --- | --- | --- | --- | --- | --- |
| 1 | `full` | 0.6 | | 36 | 0.6 h |
| 2 | `isvd_r64_h256_seed` | 3.1 | | 186 | 3.7 h |
| 3 | `nogist_h2423` | 1.55 | | 93 | **5.3 h** ← the ceiling + both primary-contrast arms |
| 4 | `frozen_r64_h256_seed` | 2.1 | | 126 | 7.35 h |
| | **compute** | | | **441 min = 7.35 h** | |

| pod | compute | + overhead | point estimate | **`gpu_budget_h` (2× bar)** |
| --- | --- | --- | --- | --- |
| `gate1_preflight` | 7.35 h | + 60 min | **8.4 h** | **17.0** |

17.0 is 2.0× the point estimate (and 1.99× the 8.55 h that also bills the two v2 extras above).
The **observed** band for an A100 40 GB is **$0.40–0.74/h**: D-005 addenda $0.449 on 51553610,
$0.471 and $0.67 on the two cycle hosts; D-011's first addendum — the Table-4 launch entry,
2026-09-18 — **$0.40 on both pods**, its addendum 3 $0.54 / $0.60 / $0.54, addendum 4 $0.67 and
$0.74, addendum 6 $0.44 (the attribution corrected by `prereg/filler_realism.md` A1.7). The cells
below cost at **$0.45–0.74/h** — the floor rounded *up* from the observed $0.40, so the low end of
every dollar figure here and in §1 is the conservative one:

| | GPU-h | × $0.45 | × $0.74 |
| --- | --- | --- | --- |
| point | 8.4 | $3.8 | $6.2 |
| **bar** | **17.0** | **$7.7** | **$12.6** |
| after arms 1–3 (both primary-contrast arms landed), + overhead | 6.3 | $2.8 | $4.7 |

**This experiment asks for ≈ 8.4 GPU-hours expected, 17 at the bar — $4–6 expected, $8–13 at the
bar.** The arm order buys an ordered loss, not a stopping rule: at ≈ 5.3 h of compute (≈ 6.3 h
from instance creation) the log already holds the ceiling, the r64 arm and the no-gist control,
which is reading (ii) in full, reading (iii) on three arms, and two of the three rates reading
(iv) needs — so a pod that *dies* there still returns most of what it was launched for.

**The pod runs all four arms.** Arm 4 is dropped only by the watchdog's credit floor or by the
`gpu_budget_h` overrun stop below — never on what arms 1–3 showed. There is no early stop keyed
to an accuracy, a rate or a surprise in the first three arms, and a run stopped by hand for any
such reason is a **fail** of reading (i), reported as one: `frozen_r64_h256_seed` has no prior
anywhere (§2d), which is precisely the kind of cell a mid-run judgement would quietly delete.

**Credit.** $92.52 (`vastai show user --raw`, 2026-09-19 13:50). This pod's bar fits inside it with
room; **Stage 1 does not** — 164 GPU-h at the bar, $74–121 (the rates above over Stage 1's design,
§1), against pre-registered bars of 121 GPU-h (`prereg/ss2_families.md` §9) and 168 GPU-h
(`prereg/l2_smoke.md` §7) already queued. D-003 (the top-up) is open and **precedes the
Stage-1 launch commit**, not this one: the pre-flight is exactly the $4–6 that keeps the $74–121
from being spent on a broken task.

`gpu_budget_h` is the pre-registered bar, enforced on the pod itself by `pod.py launch
--max-hours` (`boot.sh` runs the entrypoint under `timeout`; a run that reaches the bar prints
`===RUN_TIMEOUT_…===` and is harvested as `RUN_FAILED` with `timeout: true`). Overrun is a
stop-and-report, not a silent extension: a pod still running past its bar is a pod to kill and
diagnose (`prereg/hygiene_table4.md` §9), and the first arm-2 cell's measured rate is the first
thing to read against the 3.1 assumption.

## 8. Provenance

- Pod: `configs/pods/gate1_preflight.yaml`. Arms: `configs/arms/full.yaml`,
  `isvd_r64_h256_seed.yaml`, `nogist_h2423.yaml`, `frozen_r64_h256_seed.yaml` — the last two new on
  this branch (7f3e546), **none edited by this commit**. Task:
  `configs/tasks/ruler_v2_16k.yaml`, unchanged. Test: `tests/test_pod_manifest.py` (the launch-pod
  table — prereg path, arm order, task list, n = 12, a positive `gpu_budget_h`, every arm through
  `frontier.build_arm` at t = 16384 as the runner builds it before its first trial, and a
  `config_hash` distinct from every other pinned pod's).
- **`prereg/gate1_tracker_swap_v2.md` §1–§11 must be committed strictly before this pod's launch
  commit** (§1) — the lane runs in that order, and the DECISIONS launch entry below names that
  file's first-commit SHA beside this one's, so both orderings are checkable with
  `git merge-base --is-ancestor <prereg first commit> <launch SHA>`. Nothing this pod returns may
  enter that file except as a **dated amendment**, itself committed before the Stage-1 launch
  commit: Amendment 1 (§5's rows), the §4 (ii) exclusion, the §4 (iv) §9 re-sizing. `pod.py
  launch` enforces the order for *this* pod's prereg only; the Gate-1 body's order is this lane's
  commitment and the harvest entry is where it is evidenced.
- **This file must be committed strictly before the launch commit.** `scripts/pod.py launch`
  refuses otherwise (`prereg_error`: missing, uncommitted, not a strict ancestor, or committed by
  the launch commit itself — plus a dirty tree and an unpushed SHA), and `scripts/pod.py check`
  re-checks the order against the manifest's `git_sha` at harvest. The commit that adds this file
  adds the pod config and the test row and **launches nothing**.
- Launch: `scripts/pod.py launch --pod gate1_preflight --offer <id>` (`--max-hours` defaulting to
  `gpu_budget_h` = 17.0), from a pushed SHA on a clean tree; the watchdog under
  `caffeinate -s -i scripts/pod/watchdog.sh gate1_preflight` (no `BUDGET_ITERS` override — §6); the
  pod self-destructs `GRACE_S` = 2 h after its final marker.
- The launch is a `docs/plan/DECISIONS.md` entry under **D-011's standing authorization** naming:
  the pod, this file's first-commit SHA, the launch SHA, the offer id, the hourly rate, the bar
  (17.0) and the credit before launch — the format of the D-005 addendum entries.
- Outputs: `results/gate1_preflight/` with `manifest.json` (git SHA, config hash, model revision,
  `dataset_sha256` for the four haystack sources, `cell_elapsed_s` for the 20 cells — reading
  (iv)'s source — torch / CUDA / transformers versions, GPU, wall
  clock, command line, `errors`, `records`, `diag_skipped`, `timeout`), `trials.jsonl` (20 cells ×
  12, every row with `prompt_sha256`, `haystack_id`, `depth`, `code_family`, `ratio`, `sbits`),
  `diag.jsonl` (the three gist arms' rows), `env.txt` (rebuilt from the log's ENV block),
  `pods.txt`.
- The harvest's DECISIONS entry records the four readings of §4 as pass / fail with the evidence
  path, the four per-arm rates from `manifest.cell_elapsed_s` (or `not measured` under §4 (iv)'s
  fallback, with κ), and — if reading (ii) or (iv) fires — the amendment to
  `prereg/gate1_tracker_swap_v2.md` it obliges, by SHA, before the Stage-1 launch commit. **No
  number from this pod is cited as a result** (§1); `make tables` reads nothing from
  `results/gate1_preflight/`. The descriptive rows enter `prereg/gate1_tracker_swap_v2.md` §2 only
  as its measured baseline, by that file's dated Amendment 1, with this pod's evidence path.

## 9. What this pod does not decide

The branch (that is Gate 1: `gate1_verdict()` over Stage 1's paired contrasts at n = 24 on two
families with a Holm correction — n = 12 on one family is not that test, and no p-value is computed
here); any paper number (nothing from this pod is cited); anything at 32K; anything on Qwen2.5-7B
or Mistral-7B-v0.3 (the 512-wide twin `nogist_h4460` is not in this pod); the bf16 gist arm (L5.1,
not yet built — it is a Stage-1 arm, not a pre-flight one); the ss2 question
(`prereg/ss2_families.md`, whose branch-3 amendment is its own commit); the perplexity axis (no
`ppl` task here); whether the r64 configuration has a real-text retrieval niche (a Gate-1
question, explicitly left open by D-005's closing entry); the exact tier, the tracker, the guard's
tolerances, the kernel, or which arms belong in the paper. Task 1b asks one question — does the
Gate-1 design run, and at what rate, before 164 GPU-h are committed to it — and this pod answers
that one.

**STATUS: awaiting launch (DECISIONS, under D-011).**

---

## Amendment 1 (2026-09-20, after the partial harvest; committed strictly before the re-run's launch commit)

**STATUS: launched 2026-09-20 under D-011's standing authorization (DECISIONS D-011 addendum 9),
harvested PARTIAL at 8d10483** — this supersedes the "awaiting launch" line at the top of the
file and the one that closes it. The instance is destroyed. This amendment records what the pod
returned and what it lost, reads the readings that survived, and pre-registers the two repairs:
the harness change that makes a harvest robust to a polling gap, and a re-run pod
`gate1_preflight_rerun` — the three compressed arms, under **this** file, whose first commit
(**8db2db2**) precedes every commit on this branch and so is a strict ancestor of any launch
commit `scripts/pod.py launch` will accept.

§1–§9 above are the design as it was written before the launch and are left untouched. Nothing
below changes a reading, a threshold or a trigger: §4's four readings are read here exactly as
they are written there, and the two that the capture loss leaves undecided are re-read on the
re-run under the same rules. The incident is the harvest commit **8d10483** and
`docs/plan/STATE.md`'s 2026-09-20 14:30 EDT addendum; in `docs/plan/DECISIONS.md` it is **D-011
addendum 10** (the orchestrator's entry — that file is append-only and not this lane's).

### A1.1 What happened: the pod finished, the log did not come back

Instance **51722149** (offer 31632919, A100 SXM4 40 GB at $0.668/h, launch SHA 9e77314, bar 17 h
— D-011 addendum 9) was created at **08:44:33 UTC** (`manifest.launched_at`) and ran to the end:
the harvested log carries `===ALL_DONE_gate1_preflight_9e77314…===`, the manifest records
`status: ALL_DONE`, `errors: 0`, `diag_skipped: 0`, and the log holds the `[stage] cell` line for
`frozen_r64_h256_seed / vt` — the **last cell of the last arm** — at `n=12`. The pod did what §3
asked of it.

What failed is the capture, on this laptop:

- **The Mac slept on battery.** `pmset -g log`: `Entering Sleep state due to 'Low Power Sleep'
  … Using Batt (Charge:1%)` at **06:23:23 EDT**, and a wake from hibernate at **14:09:35 EDT**
  when it was put back on AC. The watchdog was running under `caffeinate -s -i` as §8 requires —
  but `caffeinate -s` asserts only while the machine is on **AC power** (macOS `caffeinate(1)`:
  the flag "is valid only when system is running on AC power"), so on battery it held nothing.
- **The watchdog therefore polled nothing for 7h 49m.** `results/gate1_preflight/watchdog.out`:
  `06:22 iter=39 done=0/1`, then the next line is `14:11 gate1_preflight-51722149 ALL_DONE ->
  destroy`. Its credit column brackets the bill: $92.5218 at 04:44 → $86.1195 at 14:11, i.e.
  **$6.40 for a 9.44 h instance** (`launched_at` → `harvested_at`), of which the run itself was
  ≈ 6.5 h (at A1.2's rates, `nogist` at its budgeted 1.55) — the rest is the boot and the idle
  hours the sleeping watchdog could not end.
- **The log endpoint returns a tail of about 4 MB.** The 14:11 fetch asked for 30,000 lines
  (`--tail 30000`) and returned **15,519** of them, **4,265,336 bytes** — so the cap is bytes,
  not lines. `[diag]` filled it: **15,381 of the 15,519 lines** are diagnostic rows (the gist
  arms emit ≈ 416 per sample), leaving **97 of the 240 `[trial]` rows**, **8 of the 20
  `[stage] cell` lines** and 8 cell-summary lines. The other 143 trial rows were printed on the
  pod, never fetched, and died with the instance at 14:11.
- **§6 sized the log in rows, not in bytes**, and that is the miss: it counted ≈ 40k diagnostic
  rows against a 30,000-line fetch and concluded the `<label>.log` would hold the records, when
  what the endpoint enforces is a byte ceiling the diagnostics reach first. A1.3 is the repair.

The instance was still alive at 14:11 only because it was inside `GRACE_S` = 2 h from its final
marker; a gap longer than that returns **nothing at all**. So the repair cannot be the replay
alone (A1.3).

### A1.2 What is void, what is kept — §4's four readings, read now

`results/gate1_preflight/` is kept as the evidence it is: `trials.jsonl` (97 rows),
`manifest.json` (8 `cell_elapsed_s` entries), the deduped `<label>.log` and `diag.jsonl`
(15,381 rows). `scripts/pod.py check results/gate1_preflight` returns **1** — 12 of the 20 cells
are short or absent — and that is the correct recorded state of a partial harvest, not a
finding about the pod. **No number from this pod is cited as a result** (§1, §8): `make tables`
reads nothing from that directory, and nothing below enters the paper.

**(i) Completeness — `fail (capture)`, and never written as `pass`.** 97 of 240 records reached
the laptop; of the 20 cells, 8 hold their 12 records and 12 are short or absent. The pod's own
evidence says the loss is the fetch and not the run (ALL_DONE, `errors: 0`, the last cell present
at `n=12`), but §4 (i) is a reading on the harvested records, and on those it fails. It is
**re-read on the re-run**.

**(ii) The `full` ceiling per task — READ NOW, and decided.** The `full` arm is the one arm whose
five cells came back whole (60 records, 0 errors), so this reading does not depend on a lost row:

| task | `full` | ≥ 0.9? |
| --- | --- | --- |
| `niah_single` | 12/12 = **1.00** | pass |
| `niah_multikey` | 12/12 = **1.00** | pass |
| `niah_multivalue` | 11/12 = **0.92** | pass |
| `niah_multiquery` | 11/12 = **0.92** | pass |
| `vt` | 9/12 = **0.75** | **FIRES** |

The rule fires on `vt` and on nothing else. Both consequences §4 (ii) pre-registers follow, and
neither is re-opened by the capture loss:

- **`vt` leaves the Gate-1 primary Holm family**, by a dated amendment to
  `prereg/gate1_tracker_swap_v2.md` committed before the Stage-1 launch commit — that file's
  **Amendment 1b**, which sets the `EXCLUDED_TASKS` knob its Amendment 1a (A1a.8) shipped empty
  and re-states §6's family sizes at the smaller family. Which task the knob names was left to
  1b precisely so it could not be written before this row existed; this is the row.
- **`vt` is repaired in `kvdlra.eval.gen` before any pod runs it again**, and the repair starts
  from **a comparison of the generator's template against official RULER's**, which precedes any
  pod: the v1 `vt` collapsed the same way on real text (0.08 uncompressed, D-005 addendum 2) and
  §1's recorded hypothesis is a generator/template interaction.

**The three misses are spread, not stacked.** They are trials 5, 8 and 11 of the 12-point design:
`words`×2 and `numbers`×1 (both code families), depths **0.95**×2 and **0.40**×1 (two of the
three depths the 12-trial grid draws), haystacks `essays`×2 and `arxiv`×1. So the ceiling is not
one bad haystack, one depth or one code family — which is what a single-cell defect would look
like, and is not what this is.

**(iii) Pairing — `pass` on every key that came back.** The 97 rows fall into **60 keys**
(5 tasks × 1 seed × 12 trials), **0 disagree**, and no digest is `None`; **37 of the 60** keys
hold two or more arms, so the cross-arm invariant is exercised on 37 keys and is unbroken
wherever it could be tested. The keys with one row are not evidence either way. The reading is
**re-read on the re-run, and across the two pods** (A1.4).

**(iv) Rates — two arms measured, one not; no trigger fires.** From the 8 surviving
`cell_elapsed_s` entries (seconds ÷ 12 records ÷ 60):

| arm | cells measured | min/sample | §7 budget | trigger | outcome |
| --- | --- | --- | --- | --- | --- |
| `full` | 5 of 5 (197.3 s total) | **0.055** | 0.6 | — | 11× under budget |
| `isvd_r64_h256_seed` | 2 of 5 (`niah_single` 2351.6 s, `niah_multikey` 2464.4 s) | **3.27 / 3.42** | 3.1 | > 4.7 | **not fired** |
| `frozen_r64_h256_seed` | 1 of 5 (`vt` 1137.8 s) | **1.58** | 2.1 | > 3.1 | **not fired** |
| `nogist_h2423` | 0 of 5 | **not measured** | 1.55 | > 3.1 | read on the re-run |

§4 (iv)'s fallback (the aggregate κ) is **not** used: `cell_elapsed_s` exists, for 8 cells. The
measured `isvd` rate is 5–10% above the 3.1 §7 sizes Stage 1 with and well inside its trigger, so
**§9 of `prereg/gate1_tracker_swap_v2.md` is not re-sized by this pod**; Amendment 1b records
3.27–3.42 as the measured anchor Stage 1 is read against. `frozen` at 1.58 is *faster* than its
derived 2.1. The one rate §7 named as reading (iv)'s "first suspect" — `nogist_h2423`, whose
2423-token tier is re-scored every absorb — is exactly the one the gap swallowed, which is a
reason the re-run keeps all three arms rather than the two that are undecided.

**§5's descriptive rows, as far as they go.** Two complete compressed cells survive:
`isvd_r64_h256_seed` `niah_single` **10/12 = 0.83** and `niah_multikey` **3/12 = 0.25**, and
`frozen_r64_h256_seed` `vt` **4/12 = 0.33** (a task now outside the primary family). A cell short
of 12 records is **not** quoted as a rate anywhere — the single `niah_multivalue` row is one
trial, not a cell. These are the rows `prereg/gate1_tracker_swap_v2.md` Amendment 1b may carry
into its §2 as the measured real-text baseline, superseded cell by cell by the re-run's.

### A1.3 The repair: the pod replays its records, and the laptop stays on AC

**(a) The records replay** (commit **049c66b**, `kvdlra.eval.records.replayed`, wired into
`scripts/pod.py run`). At the end of the run — after `run_pod` returns, before the final
`[stage] wall_clock_s` line, and before boot.sh prints `===ALL_DONE_…===` — the pod prints every
compact record line a second time, between `===RECORDS_REPLAY_BEGIN===` and
`===RECORDS_REPLAY_END===`: the `[trial]` rows, the cell summaries, the `[pplw]` and `ppl=`
rows, the `[error]` lines and every `[stage]` line (the four `dataset_sha256` digests included —
the manifest that holds them dies with the instance). **`[diag]` rows are not replayed**: they
are the volume, not the reading, and replaying them would reproduce the failure.

**Why it makes the harvest robust to a polling gap.** For the re-run the block is ≈ 180 `[trial]`
+ 15 cell summaries + 15 `[stage] cell` + ≈ 10 other `[stage]` lines ≈ **220 lines, under 30 KB**
— 0.7% of a 4 MB tail; a Stage-1 pod's block is ≈ 1,000 lines. Since it is printed last, any tail
that reaches back past it carries **every reading**, and a polling gap of any length costs
`[diag]` rows and nothing else. Two limits, stated rather than left to be discovered: the block
does not help a pod whose **instance is destroyed before the run ends** (nothing is printed yet),
and it does not help a tail smaller than the block.

Two mechanical consequences, both already shipped and tested:

- `scripts/pod.py harvest` **dedupes exact-duplicate lines** (order-preserving) before parsing,
  so a log fetched by hand — which carries both copies — yields byte-identical records to one
  without the replay. The watchdog's per-poll `sort -u` already did this for `<label>.log`.
- §4 (i)'s `grep -c '^\[trial\]' <label>.log` = 240 is read on the **deduped** `<label>.log` the
  watchdog writes, where the replay collapses into the rows it repeats. On a raw dump the count
  is doubled; the harvest's own dedupe is what the records are parsed from either way.

**(b) The launch procedure gains one line, and §8's launch bullet is amended to it:** the Mac
runs the watchdog under `caffeinate -s -i` **and stays on AC power** — `caffeinate -s` holds
only on AC, so a watchdog started on battery is a launch to postpone, not a launch to watch. The
pod's own `GRACE_S` = 2 h self-destruct bounds the idle billing a gap can cost (it did here:
$6.40 against a $5.6 estimate) but also bounds the window in which the log can still be fetched,
which is why (a) and (b) are both required and neither replaces the other.

### A1.4 The re-run: `gate1_preflight_rerun`

`configs/pods/gate1_preflight_rerun.yaml` — the same model, dtype, image and task as
`configs/pods/gate1_preflight.yaml` (`ruler_v2_16k`: generator v2, five tasks, the 2 × 3 × 2
design, n = 12, one seed, chunk 4096, Llama-3.1-8B at 16K), under **this** pre-registration
(`prereg: prereg/gate1_preflight.md`), with the arms:

| # | arm | why |
| --- | --- | --- |
| 1 | `isvd_r64_h256_seed` | reading (iii)'s and (iv)'s primary arm; 3 of its 5 cells were lost |
| 2 | `nogist_h2423` | the byte-matched control; **all 5** cells lost, and the unmeasured rate |
| 3 | `frozen_r64_h256_seed` | the learn-then-freeze control; 4 of its 5 cells lost |

**`full` is not re-run.** Its five cells are complete and its reading (ii) is decided above; the
arm would add 60 samples of ceiling for a row that already exists, and the re-run's arms are
compared against **the ceiling this pod measured**. The arm order is §3's with that arm removed,
so the pod still lands the primary contrast first if it dies early.

**How the two pods are shown to be one experiment.** The generator is seeded and its inputs are
pinned — both manifests must carry the same four `dataset_sha256` haystack digests
(`4c2b7e03…` arxiv, `12a16c81…` essays, `7a9b488c…` pg19, `6e37fe2f…` wikipedia) — so the
re-run's prompts must be byte-identical to the first pod's, key by key:

```python
import json, collections
rows = [json.loads(l) for d in ("gate1_preflight", "gate1_preflight_rerun")
        for l in open(f"results/{d}/trials.jsonl")]
sha = collections.defaultdict(set)
for r in rows: sha[(r["task"], r["seed"], r["trial"])].add(r["prompt_sha256"])
bad = {k: v for k, v in sha.items() if len(v) != 1 or None in v}
print(len(sha), "keys;", len(bad), "disagree")
```

Expected `60 keys; 0 disagree`. A key that disagrees is a **fail of reading (iii) on the re-run**
and the two pods are **not pooled**: the surviving `full` rows would then be a ceiling for other
prompts, and the re-run would be read on its own with the ceiling recorded as not re-measured.

**The readings on the re-run are §4's, unchanged.** (i) completeness at **15 cells × 12 = 180**
records, 0 errors, `scripts/pod.py check results/gate1_preflight_rerun` = 0; (iii) the pairing
invariant, within the pod (60 keys × 3 arms) and across the two pods as above; (iv) the same
three triggers — `isvd` > 4.7, `nogist` > 3.1, `frozen` > 3.1 — now read off 15
`cell_elapsed_s` entries, five per arm, with §4 (iv)'s κ fallback standing if the lines are
absent. Reading (ii) is **not** re-read: it is decided in A1.2, and no arm in this pod can
measure it.

**Budget, at the rates this pod measured.** 60 samples per arm; `isvd` at the lower of its two
measured rates, `frozen` at its measured one, `nogist` at §7's derived 1.55 (it is the rate that
is unmeasured, so it is budgeted, not assumed away):

| arm | min/sample | source | × 60 | minutes |
| --- | --- | --- | --- | --- |
| `isvd_r64_h256_seed` | 3.27 | measured (A1.2; 3.42 on the other cell) | | 196 |
| `nogist_h2423` | 1.55 | §7, derived — not measured | | 93 |
| `frozen_r64_h256_seed` | 1.58 | measured (A1.2) | | 95 |
| | **compute** | | | **384 min = 6.4 h** |

Plus **20 min boot**: `prereg/l2_smoke.md` §7's measured 15–20 min bracket for this image and
host family, at its top; this pod's own in-run setup is in its log and sits inside it
(`load_model` 194.7 s + the four `materialize` lines 46.1 s + `load_corpora` 0.1 s = **4.0 min**).

| pod | compute | + boot | point estimate | **`gpu_budget_h` (2× bar)** |
| --- | --- | --- | --- | --- |
| `gate1_preflight_rerun` | 6.4 h | + 20 min | **6.8 h** | **14.0** |

14.0 is 2.06× the point. The sensitivity is stated rather than hidden: at `isvd`'s upper measured
rate (3.42) the point is 6.9 h, and at §7's *unamended* 60 min overhead it is 7.4 h — 14.0 is
still 1.9× the worse of the two. At **$0.68–0.74/h** (the first instance billed $0.668/h; the top
is §7's observed band) that is **$4.6–5.0 expected, $9.5–10.4 at the bar**, against credit
$86.12 (`watchdog.out`, 14:11). The bar is enforced on the pod by `pod.py launch --max-hours`
(default `gpu_budget_h`), and §7's overrun rule — a pod past its bar is killed and diagnosed —
stands unchanged.

**Provenance, as §8 requires of the original.** Pod config `configs/pods/gate1_preflight_rerun.yaml`,
no arm or task file edited; pinned in `tests/test_pod_manifest.py` (the launch-pod table: prereg
path, arm order, task list, n = 12, the 14.0 bar, every arm through `frontier.build_arm` at
t = 16384, and a `config_hash` distinct from every other pinned pod's — the dropped `full` arm is
what separates it from the pod it repeats). This file's first commit **8db2db2** is a strict
ancestor of every commit on this branch, and this amendment is committed before the re-run's
launch commit. Launch: `scripts/pod.py launch --pod gate1_preflight_rerun --offer <id>` from a
pushed SHA on a clean tree, the watchdog under `caffeinate -s -i scripts/pod/watchdog.sh
gate1_preflight_rerun` **on AC power** (A1.3 (b)), recorded in `docs/plan/DECISIONS.md` under
D-011's standing authorization with the offer id, the hourly rate, the bar and the credit before
launch. Outputs: `results/gate1_preflight_rerun/` — `manifest.json` (15 `cell_elapsed_s`
entries), `trials.jsonl` (180 rows), `diag.jsonl`, `env.txt`, `pods.txt`.

### A1.5 What does not change

§1–§9 stand as written, and this amendment adds no reading and moves no threshold: §3's design,
§4's four readings with their triggers and their fallback, §5's descriptive expectations, §6's
log-volume rule (its byte miss is recorded in A1.1 and repaired in A1.3, not re-derived here),
§7's rates, its 2× bar rule and its overrun rule, §8's provenance and ordering requirements, and
§9's list of what this pod does not decide — which the re-run does not decide either. The
`full` arm's absence from the re-run changes no reading's definition: reading (ii) is **decided**,
not dropped. No number from either instance is cited as a result, and `make tables` reads neither
directory. The pre-flight's three returns to `prereg/gate1_tracker_swap_v2.md` — the §2 rows, the
`vt` exclusion and the §9 re-size (not triggered) — remain that file's **Amendment 1b**, to be
committed before the Stage-1 launch commit.

**STATUS: harvested PARTIAL (8d10483); the re-run `gate1_preflight_rerun` is pre-registered here
and awaits launch (DECISIONS, under D-011).**

### Amendment 1 — correction (2026-09-20 17:42 EDT, before the re-run's launch commit)

Nothing above is edited — §1–§9 and Amendment 1 itself stand as committed. This note narrows how
two of Amendment 1's own readings are worded and names two of its evidence paths as local-only.
No reading, threshold, trigger or number changes.

**A1.2 (iv): `isvd_r64_h256_seed` and `frozen_r64_h256_seed` read `fail (not reached)`, not "not
fired."** §4's consistency check (l. 311): a fail on reading (i) makes (ii) and (iv)
**unreadable** for the affected arm, written `fail (not reached)`, never as a pass or a decision.
Reading (i) fails for `isvd_r64_h256_seed` (2 of its 5 cells captured, 3 short/absent) and for
`frozen_r64_h256_seed` (1 of 5, 4 short/absent) — the same capture loss A1.2 (i) already names.
"Not fired" (l. 690–691) reads as a trigger legitimately evaluated and cleared; on an arm (i)
fails for, it is not that, and those two outcome cells are corrected here to `fail (not reached)`.
(`nogist_h2423`'s existing `not measured` / "read on the re-run" wording already carries that
meaning and needs no correction.) The measured 3.27–3.42 min/sample (`isvd`) and 1.58 (`frozen`)
are unchanged and still stand — but as A1.2 already called them, **descriptively**, from the
cells that survived: no re-size followed from them there, and none follows from this correction
either. Amendment 1b's anchor for Stage 1 is the re-run's own five-cell-per-arm rates (A1.4), not
these partial ones. Reading (ii) is **not** touched by this note: it is decided on the `full` arm
alone, whose 5 of 5 cells are complete — (i) does not fail for `full`, so (ii) was never
"unreadable" under the rule this note applies to (iv).

**Two evidence paths A1.1/A1.2/A1.4 cite are not in the repository.**
`results/gate1_preflight/watchdog.out` — A1.1's sleep/wake timeline and dollar figures, and
A1.4's "against credit $86.12 (`watchdog.out`, 14:11)" — was never committed and is now removed
from the working tree. The deduped `<label>.log` A1.2 lists among the kept evidence
(`results/gate1_preflight/gate1_preflight-51722149.log`) is gitignored (`.gitignore:54`,
`results/*/*.log`) and present only on this laptop. Both are **local-only**: reproducible by
whoever ran the pod, not from `git log` alone. This does not weaken A1.1's finding — the poll gap
it is evidence for is independently readable from what committed evidence and system logs give on
their own: `pmset -g log`'s sleep-at-06:23:23-EDT / wake-at-14:09:35-EDT entries (cited in A1.1
itself, and not sourced from `watchdog.out`), and the committed `manifest.json`'s `launched_at`
(08:44:33Z) vs `harvested_at` (18:11:09Z) — a 9.44 h span that alone brackets the same gap. The
one place the two evidence sources disagree, at the second decimal, is the dollar figure: A1.1's
**$6.40** was read off `watchdog.out`'s last two polls before the destroy, a file that no longer
exists to re-check; `docs/plan/DECISIONS.md`'s D-011 addendum 10 states the settled **$6.44**
(credit $92.52 at launch, addendum 9, → $86.08). DECISIONS is the append-only ledger CLAUDE.md
names for gate outcomes and their evidence path — its figure is the one to cite going forward;
A1.1's $6.40 stands as what the amendment read at the time, not as a second, competing cost.

**The "16 → 12" aside.** A1.2 (ii)'s consequence — `vt` leaving the primary Holm family — is that
family (§6: 2 contrasts × 4 tasks × 2 models = **16**) losing `vt`'s share (2 contrasts × 1 task
× 2 models = **4**), **16 → 12**. `prereg/gate1_tracker_swap_v2.md`'s own preamble already states
the rule ("16 − 4 per excluded task on retrieval") and D-011 addendum 10 already uses the same
"16 → 12" shorthand; that file's own **Amendment 1b — not yet committed —** is where the shrink is
formally landed, not here. This sentence only closes the loop this file's own cross-reference left
open.

**A third limit on the repair, beside A1.3's two (l. 724–726).** The replay does not survive a run
killed at the `MAX_HOURS` bar: `scripts/pod/boot.sh:140` runs `pod.py run` under `timeout
--signal=TERM --kill-after=60`, and Python's default disposition for `SIGTERM` ends the process
without running any `finally` — including `replayed()`'s — so a pod that hit its bar mid-run would
have printed no replay block at all, the same loss A1.3 exists to repair. Fixed at commit
**4a5f71e** (`L3.4c`, this branch): `scripts/pod.py run` now installs
`signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))` before `run_pod` starts, turning
`SIGTERM` into `SystemExit` so `replayed()`'s `finally` runs inside the 60 s kill grace.
