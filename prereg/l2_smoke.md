# Pre-registration — `l2_smoke`

**STATUS: awaiting owner go (DECISIONS).** Written before the pod is launched. Launch is a
separate, later commit that spends money and is recorded in `docs/plan/DECISIONS.md`:
`scripts/pod.py launch` refuses unless this file's first commit is a *strict* ancestor of the
launch commit, so the commit that adds this file launches nothing. Lane L2 item 8 ("Smoke pod:
all arms, Llama 16K, n=12, generator v2 — harness validation, not paper data"); gate G2 line 6
is ticked by this pod's harvest (§4 reading 5). Gate G2 line 7 (OjaKV) is **not** claimed by this
pod: there is no OjaKV arm (DECISIONS D-017, OPEN — the authors' code cannot be run through this
harness), and the `oja_*` arms here are Oja subspace tracking in our own cache, not OjaKV.

---

## 1. Purpose

Every arm this lane built — the faithful KIVI arms (L2.2), the k ∈ {0.10, 0.15, 0.25} eviction
grid with PyramidKV decoding token-by-token (L2.4/L2.4b, ruling R-L2-5), the composed
ThinK + SnapKV arm — and every arm it inherited (the gist variants, ShadowKV, the streaming /
single-shot / hqq quant arms, the MiniKV-style composites, the Oja and FD tracker swaps, the SVD
oracle, standalone ThinK) has run on a GPU only through the harness it was written for: the
inherited arms through `w10_ruler.py` on the cycled filler, the new ones on CPU tests with a tiny
model. Generator v2 (`kvdlra.eval.gen`, L2.3a–c) has run on a GPU **never**: the only pods on the
L0 runner (`scripts/pod.py run` → `kvdlra.eval.runner`) are the three Table-4 pods (D-011, the
perplexity axis) and the two filler-realism pods (D-005, the in-house generator). Before any gate
pod is trusted — L3's Gate-1 tracker-swap pods run generator v2 at n = 24 on three families —
every arm has to be shown to complete end to end on generator v2, on the pod path, at the
paper's context and family.

**This pod is that demonstration and nothing else.** It is harness validation: **no number from
it is cited** — not in the paper, not in a table, not as a baseline row, not as a prediction's
basis. Its readings (§4) are all completeness readings: did every arm produce its 12 records per
cell without raising, were all 40 arms fed byte-identical prompts, did the arms the gate names
complete. Accuracy is recorded, because the harness records it, and is read only as a
harness-sanity signal (§5): a `full` cell at 0 is a generator finding, not a compression one.

What "runs end to end" means, arm by arm, is fixed by the harness and not by this file:
`runner.run_pod` builds every arm of the pod before the first trial (`frontier.build_arm` — a
config that cannot resolve fails the pod before a record exists), then for each arm in order runs
the five sub-tasks × 12 trials, each trial `gen.run_trial` → `ruler.retrieve` under the arm's own
prefill protocol (chunked only for the streaming and quant kinds; see §3), a greedy decode of
`MAX_NEW` tokens, the `string_match_all` hit rule, and — for a `bug`-kind arm — the `[diag]`
drain. A trial that raises is a record with `error` set and `hit = 0`, counted in n (ruling R29).

## 2. The evidence on disk that sets the expectations

Every number in this section was computed from committed records or from the live capture by
the snippet beside it, at the commit that adds this file. The plan's reading for the r64
configuration ("within 0.15 of its v1 cycled-filler accuracy on single-needle, else the
filler-realism reading applies") is **superseded** by these numbers (task brief, ruling recorded
there): the filler-realism diagnostic has already measured what real-text filler does to the gist
arms, and the smoke's expectation is written from that, not re-tested here.

**(a) The harness replicates the archive on the cycled filler** — `results/filler_realism_cycle/`
(instance 51559661, `ALL_DONE`, harvested 4ded444; DECISIONS D-005 addendum 3), hits/12 per
cell, errors, distinct `prompt_sha256` per cell, and the pairing invariant over the 48
`(task, seed, trial)` keys:

```python
import json, collections, pathlib
rows = [json.loads(l) for l in pathlib.Path("results/filler_realism_cycle/trials.jsonl").read_text().splitlines()]
cells = collections.defaultdict(list)
for r in rows:
    cells[(r["arm"], r["task"])].append(r)
for (arm, task), rs in sorted(cells.items()):
    print(arm, task, f"{sum(r['hit'] for r in rs)}/{len(rs)}", "errors", sum(1 for r in rs if r["error"]),
          "shas", len({r["prompt_sha256"] for r in rs}))
keys = collections.defaultdict(set)
for r in rows:
    keys[(r["task"], r["seed"], r["trial"])].add(r["prompt_sha256"])
print(len(keys), "keys;", sum(1 for v in keys.values() if len(v) != 1), "disagree")
```

| arm (record key) | single | multikey | multivalue | vt | errors |
| --- | --- | --- | --- | --- | --- |
| `bugSseed-r64-h256` (= `isvd_r64_h256_seed`) | 12/12 | 12/12 | 12/12 | 8/12 (0.67) | 0 |
| `full` | 12/12 | 12/12 | 12/12 | 12/12 | 0 |

48 keys, 0 disagree: the two arms were fed byte-identical prompts on every key — the pairing
invariant, verified on hardware once already (D-005 addendum 3). The archived `w18-g1-llama` row
for the r64 arm is 1.00 / 1.00 / 1.00 / 0.58; every task is within 0.25, so under
`prereg/filler_realism.md` A1.3 the L0 runner reproduces `w10_ruler.py` and harness drift is
excluded as an explanation for anything below.

**(b) The same arm on real-text filler** — `results/filler_realism/filler_realism-51553610.raw`
(instance 51553610, launch SHA 56e89ee, **still running and unharvested** at the time of
writing; the watchdog's raw capture, deduped by the cell rows; read 2026-09-19 08:03 EDT):

```sh
grep -a '^\[niah\|^\[vt' results/filler_realism/filler_realism-51553610.raw | sort -u   # the cell rows
```

| arm (record key) | single | multikey | multivalue | vt |
| --- | --- | --- | --- | --- |
| `full` | 1.00 | 1.00 | 0.92 | 0.08 |
| `bugSseed-r64-h256` | **0.25** | **0.08** | **0.00** | 0.00 |
| `bugSseed-r64-h256-q4` | **0.00** (cell closed 08:03 EDT) | running | — | — |

These are not harvested numbers and are not cited as numbers (the cell rows are the runner's
own aggregate lines; the harvest that closes D-005 is what makes them records). They are the
reason §4 reading 4 and §5 read the way they do: on WikiText sentences instead of the ten cycled
ones, with the model, harness, tasks, n and arm config held fixed, the r64 configuration fell
from 1.00 / 1.00 / 1.00 to 0.25 / 0.08 / 0.00 on the three tasks whose ceiling (`full`) held at
≥ 0.92 — and its q4 cell is at 0.00 on single-needle. The `vt` column is descriptive there
(the ceiling itself fell to 0.08; `prereg/filler_realism.md` Amendment 2) and is not part of any
reading here. Generator v2's haystacks are real documents from four sources (PG-19, arXiv,
Wikipedia, essays), i.e. harder filler than WikiText sentences by construction, under official
RULER templates.

**Consequence:** the gist arms are **expected to score low** on this pod — on every task, at
every rank — and nothing about their accuracy here is a finding. A gist arm at 0.00 on
`niah_single` with 12 clean records is a *pass* of this pod. The one thing the gist cells decide
is whether the arm runs.

## 3. Arms, tasks, n

**The arm set is a rule, not a list:** every stem under `configs/arms/` that is not a Table-4
diagnostic variant. The ten `isvd_r{128,256}_{noguard,tol,qr64,f0.01_tol,f0.01_qr64}` files
exist for one perplexity contrast (`prereg/hygiene_table4.md`; they are `isvd_r128` /
`isvd_r256` / `isvd_r256_f0.01` under guard knobs and a wider diagnostic window) and would add
ten near-duplicate r128/r256 arms (5 × 60 × 4.0 + 5 × 60 × 6.0 min = 50 GPU-h at the §7
rates) to a retrieval smoke that already runs their three plain sources.
`tests/test_pod_manifest.py` (`# --- L2.5b`) pins `set(arms) == {all stems} − {the gist arms of
the three Table-4 pods}`, so an arm added later cannot be left out of the smoke silently and a
Table-4 variant added later is excluded by the same rule. 50 stems − 10 = **40 arms**, in this order (cheap → expensive, by class, so a pod
that dies early still lands whole classes; the runner runs one arm's five cells before the next
arm):

| # | arm config | `kind` | prefill | record key (`legacy_name` or config name) |
| --- | --- | --- | --- | --- |
| 1 | `full` | `full` | single-shot (one forward, `DynamicCache`) | `full` |
| 2 | `snapkv_k0.1` | `press` | single-shot (`chunkable: true` in the archived config; the press branch ignores `chunk`) | `snapkv-k0.1` |
| 3 | `snapkv_k0.10` | `press` | single-shot | `snapkv_k0.10` |
| 4 | `snapkv_k0.15` | `press` | single-shot | `snapkv_k0.15` |
| 5 | `snapkv_k0.25` | `press` | single-shot | `snapkv_k0.25` |
| 6 | `pyramidkv_k0.10` | `press` | single-shot, decodes token-by-token | `pyramidkv_k0.10` |
| 7 | `pyramidkv_k0.15` | `press` | single-shot, decodes token-by-token | `pyramidkv_k0.15` |
| 8 | `pyramidkv_k0.25` | `press` | single-shot, decodes token-by-token | `pyramidkv_k0.25` |
| 9 | `ea_k0.1` | `press` | single-shot (as arm 2) | `ea-k0.1` |
| 10 | `ea_k0.10` | `press` | single-shot | `ea_k0.10` |
| 11 | `ea_k0.15` | `press` | single-shot | `ea_k0.15` |
| 12 | `ea_k0.25` | `press` | single-shot (as arm 2) | `ea-k0.25` |
| 13 | `ea_k0.5` | `press` | single-shot (as arm 2) | `ea-k0.5` |
| 14 | `think_c0.5` | `press` | single-shot | `think-c0.5` |
| 15 | `think_c0.5_snapkv_k0.15` | `press` | single-shot | `think_c0.5_snapkv_k0.15` |
| 16 | `svd_oracle_r0.5` | `press` | single-shot | `palu-r0.5` (the archived string; D-004) |
| 17 | `kivi2_streaming` | `quant` | chunked 4096 | `quant-2bit-kivi` |
| 18 | `kivi4_streaming` | `quant` | chunked 4096 | `quant-4bit-kivi` |
| 19 | `kivi2_singleshot` | `quant` | single-shot | `quant-2bit-kivi#chunk0` |
| 20 | `kivi2_faithful` | `quant_faithful` | fp16 single-shot, quantized post hoc | `kivi2_faithful` |
| 21 | `kivi4_faithful` | `quant_faithful` | fp16 single-shot, quantized post hoc | `kivi4_faithful` |
| 22 | `kivi8_hqq` | `quant` | chunked 4096 | `quant-8bit-kivi-hqq` |
| 23 | `ea_k0.1_kivi2` | `composite` | single-shot | `ea-k0.1-q2-kivi` |
| 24 | `ea_k0.1_kivi4` | `composite` | single-shot | `ea-k0.1-q4-kivi` |
| 25 | `ea_k0.25_kivi2` | `composite` | single-shot | `ea-k0.25-q2-kivi` |
| 26 | `ea_k0.25_kivi4` | `composite` | single-shot | `ea-k0.25-q4-kivi` |
| 27 | `shadowkv_r64` | `shadow` | single-shot | `shadow-r64` |
| 28 | `shadowkv_r128` | `shadow` | single-shot | `shadow-r128` |
| 29 | `evict_surprise_h256` | `bug` | chunked 4096 | `bugEVICT-h256` |
| 30 | `isvd_r64` | `bug` | chunked 4096 | `bug-r64` |
| 31 | `isvd_r64_h256_seed` | `bug` | chunked 4096 | `bugSseed-r64-h256` |
| 32 | `isvd_r64_h256_seed_q4` | `bug` | chunked 4096 | `bugSseed-r64-h256-q4` |
| 33 | `oja_r64_h256_seed` | `bug` | chunked 4096 | `bugSseed-r64-h256-oja` |
| 34 | `oja_r64_h256_seed_tuned` | `bug` | chunked 4096 | `oja_r64_h256_seed_tuned` |
| 35 | `fd_r64_h256_seed` | `bug` | chunked 4096 | `bugSseed-r64-h256-fd` |
| 36 | `isvd_r128` | `bug` | chunked 4096 | `bug-r128` |
| 37 | `isvd_r128_h1024_seed_s32` | `bug` | chunked 4096 | `bugSseed-r128-h1024-s32` |
| 38 | `isvd_r256` | `bug` | chunked 4096 | `bug-r256` |
| 39 | `isvd_r256_f0.01` | `bug` | chunked 4096 | `bug-r256-f0.01` |
| 40 | `isvd_r256_h1024_seed` | `bug` | chunked 4096 | `bugSseed-r256-h1024` |

The name, kind and record-key columns were printed from the configs (`load_arm` +
`frontier.build_arm` at t = 16384 over `load_pod("l2_smoke").arms`), not typed. The prefill
column is `ruler.retrieve`'s, not the config's: the runner passes `chunk = task.chunk if
arm.chunkable else 0`, but `chunk` reaches only the streaming (`bug`, `shadow`) and `quant`
branches; the `full`, press and composite branches prefill the whole prompt in one forward
whatever `chunkable` says (the archived `snapkv_k0.1` / `ea_k0.*` configs carry `chunkable:
true` from the legacy dict and ran single-shot then too — CODE_AUDIT's protocol note). So
**fifteen arms prefill chunked** (the twelve gist arms and the three chunked quant arms 17, 18,
22) and **twenty-five single-shot**. **The cheap first half** (§7) is arms 1–28 — everything
before the first `bug`-kind arm; **the gist half** is arms 29–40.

**The four press families the gate names** (G2 line 6): SnapKV = arms 3–5, PyramidKV = 6–8,
ExpectedAttention = 10–11 plus the archived `ea_k0.25` (arm 12: the k = 0.25 point of the grid
under its archived stem — the pods that pin its hash keep it; ruling PR-L2-12), ThinK + SnapKV
= arm 15. The three PyramidKV arms decode one token per forward (ruling R-L2-5: transformers
builds one causal mask from layer 0's key count, and PyramidKV's deeper layers hold fewer keys);
their perplexity path is refused and no `ppl` task is here.

**Not an arm:** the ten Table-4 variants (above); any `ojakv*` (D-017; the test pins that no
such stem is in the pod); the `isvd_r64_h256_seed_diag4096` variant `prereg/ss2_families.md` §8
considered and did not create (§6 reaches the same conclusion here).

| pod | model | dtype / image | task | cells |
| --- | --- | --- | --- | --- |
| `l2_smoke` | `unsloth/Meta-Llama-3.1-8B-Instruct` | bfloat16, `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` (quanto JIT-builds its kernel) | `ruler_v2_16k` | 40 arms × 5 tasks = 200 |

**Task and n.** `configs/tasks/ruler_v2_16k.yaml`, unchanged: `generator: v2`, `ctx: 16384`,
the five sub-tasks `niah_single`, `niah_multikey`, `niah_multivalue`, `niah_multiquery`, `vt`
(official RULER templates; `vt`'s value takes the cell's code family — ruling R-L2-4), haystacks
from `pg19`, `arxiv`, `wikipedia`, `essays` (64 documents each, materialized by `pod.py run`
before the model loads, digests in `manifest.dataset_sha256`), `design: {haystacks: 2, depths: 3,
codes: 2}` → `n_trials: 12` at depths 0.05 / 0.4 / 0.95, `seeds: [0]`, `chunk: 4096` (the
schema default). **12 records per (arm, task) cell**, 60 per arm, **2,400 in the pod**; `pod.py
check` requires all 200 cells at exactly 12, errors counted.

**Pairing.** `make_trial` is deterministic in `(task, seed, trial)` and never sees the arm, so
all 40 arms are fed the same token ids for a given key; the runner's `[trial]` line and record
carry `prompt_sha256` over exactly those ids, so the pairing is verified from the records (§4
reading 3), not assumed. Whether an arm prefilled the ids in one shot or in 4096-token chunks
does not enter the digest (it is over the ids, not the forward), so the single-shot and chunked
arms must agree too — as `full` (single-shot) and `bugSseed-r64-h256` (chunked 4096) do on the
cycle pod (instance 51559661, launch SHA a5cd89c, post-L2.3b): 48 paired keys, 0 disagreeing
`prompt_sha256` (§2a).

**Decode budgets.** `MAX_NEW` = 48 for the four `niah_*` tasks and 64 for `vt` (`gen.MAX_NEW`),
not the in-house 40; the query is fed in one forward for the DynamicCache arms and one token per
forward for the streaming caches and the PyramidKV arms.

## 4. Readings, fixed now

Each reading is a **pass / fail** on the harvested records; the six are independent, and the
pod is "validated" only when all six pass. Nothing here is a hypothesis test and no correction
applies. "The records" are `results/l2_smoke/trials.jsonl` as `pod.py harvest` writes them from
the deduped `<label>.log`, with `pod.py check` run on the directory.

1. **No arm with `error` records.** `sum(1 for r in rows if r["error"])` = 0. `pod.py check`
   fails the pod on the first error row (R29). *Fail* names the arm(s) and the exception text(s)
   verbatim (the row carries `"<ExcType>: <message>"`); an arm whose every trial errored is an arm
   whose `make()` or `retrieve` path does not run on the pod — the finding the smoke exists to
   produce, reported to the lane that owns the arm, never worked around on the pod.
2. **Every arm's n = 12 per cell.** All 200 `(arm, task)` cells hold exactly 12 records —
   `pod.py check`'s cell rule, and `grep -c '^\[trial\]' <label>.log` = 2,400. *Fail* is a
   short or missing cell: the pod died (a `RUN_FAILED` marker, or the `--max-hours` bar —
   `timeout: true` in the manifest) before reaching it, and the arms after the last complete one
   are unvalidated; the harvest is kept as partial evidence and not as a validated smoke.
3. **`prompt_sha256` identical across arms per `(task, seed, trial)`.** The check, over
   `trials.jsonl`:

   ```python
   import json, collections
   rows = [json.loads(l) for l in open("results/l2_smoke/trials.jsonl")]
   sha = collections.defaultdict(set)
   for r in rows: sha[(r["task"], r["seed"], r["trial"])].add(r["prompt_sha256"])
   bad = {k: v for k, v in sha.items() if len(v) != 1}
   print(len(sha), "keys;", len(bad), "disagree;", sorted(bad)[:5])
   ```

   Expected `60 keys; 0 disagree`, every key holding 40 records with one digest (a key whose
   set is `{None}` or contains `None` is a *fail* too: a record without a digest is a runner
   defect). *Fail* names the keys and the arms whose digest differs; the pairing invariant that
   every paired statistic of the Gate-1 pods rests on is then broken on the pod path and L3
   cannot launch on it.
4. **The gist arms: completeness, never accuracy** (supersedes the plan's "within 0.15 of the
   v1 cycled-filler accuracy" reading — §2). Arms 29–40 pass this reading when readings 1–3
   hold for their 60 cells; **their accuracy is expected to be low on every task and is not
   read**. In particular the r64 configuration at 0.25 / 0.08 / 0.00 on WikiText filler (§2b)
   is expected at or below that on generator v2's documents, and a 0/12 cell is a pass. What
   *is* recorded from these arms, descriptively: `ratio` / `sbits` on every row (the accounting
   ran), and the `[diag]` rows of §6 — the guard's pre-repair maxima and the abort count, read
   as `prereg/ss2_families.md` §7(f) reads them (the repair firing every window at bf16 is not a
   finding; an abort is an `error` row and fails reading 1).
5. **The four press families' cells complete → G2 line 6.** Arms 3–8, 10–12 and 15 — the
   eight k ∈ {0.10, 0.15, 0.25} arms `snapkv_k0.10/0.15/0.25`, `pyramidkv_k0.10/0.15/0.25`,
   `ea_k0.10/0.15`, the archived `ea_k0.25`, and `think_c0.5_snapkv_k0.15` — hold 12 clean
   records in all five cells each, **including the three PyramidKV arms decoding token-by-token**
   (R-L2-5): 60 rows each, no `error`. On pass, `docs/plan/lanes/GATES.md` G2 line 6 is ticked in
   the harvest's DECISIONS entry with the evidence path. *Fail* on any of them leaves the line
   open and names the arm and exception.
6. **The `oja_*` and `fd_*` tracker-swap arms complete without `error` rows** (arms 33–35).
   The Week-20 swap pod's FD arm crashed on every trial (`linalg.svd` non-convergence) and its
   Oja arm ran untuned defaults (CLAUDE.md settled facts); L1 rewrote `fd_step`'s augmentation
   and gave `_svd_core` an eigendecomposition fallback (`tests/test_fd_numerics.py`), fixed the
   Oja rank pin and re-tuned the schedule (D-014), and put the orthonormality guard on every
   `bug`-kind arm: a basis the guard cannot repair now **raises** `OrthonormalityError` instead
   of diverging silently. So the reading counts aborts: **expected 0** `error` rows on the three
   arms (180 records); any abort is an `error` row naming the layer, the post-repair error and
   the absorb, and is reported per arm with its count — a tracker that cannot hold
   orthonormality on real text at bf16 is a finding for L3's Gate-1 design (which runs these
   trackers at n = 24 on three families), not a pod to re-run with the guard off.

**Consistency check when the table is written:** six lines, each `pass` or `fail`, none
blank; a `fail` on 1 or 2 makes 5 and 6 unreadable for the affected arms and is written as
`fail (not reached)` there, never as pass.

## 5. Expectations per arm class, descriptive

No accuracy prediction is pre-registered — no arm has a generator-v2 number anywhere, and this
pod does not make one. What is expected, so that a surprise is recognisable:

- **`full`**: the generator ceiling. Real-text `full` on the in-house tasks read 1.00 / 1.00 /
  0.92 / 0.08 (§2b), the `vt` collapse being the WikiText × v1-`vt` heading interaction of
  `prereg/filler_realism.md` A2.6; v2's `vt` uses RULER's own template and 4-hop chain, and v2's
  `niah_*` use RULER's needle and question strings. A `full` cell far below 1.00 on any of the
  five tasks — above all `niah_single` — is a **generator or template finding** (a prompt the
  model cannot answer with the KV cache intact), reported to L3's Gate-1 pre-registration before
  it launches, and not a compression result. No threshold is fixed for it; it is reported.
- **The presses and the quant arms** (2–28): descriptive rows, expected in the same order as
  their archived cycled-filler rows (4-bit and 8-bit near the ceiling, single-shot 2-bit above
  streaming 2-bit, k = 0.25 above k = 0.10) — but the archived rows are on the cycled filler and
  do not transfer; an inversion is noted, not read.
- **The gist arms** (29–40): low everywhere (§2, §4 reading 4).
- **`kivi8_hqq`** is the near-lossless decode-path control its config describes: a zero on
  `niah_single` there means the hqq decode path is broken, not the bit width — a harness
  finding, reported as such.

**Abort and error are possible outcomes, not accidents.** An OOM in a single-shot prefill (25
arms prefill the whole ≈ 16.4K-token prompt in one forward; on an A100 40 GB the expected peak
is the 16 GB bf16 weights + a ≈ 2 GB 16K cache + the arm's temporaries — the SVD oracle's
per-head SVD of a 16K × 128 slab, the faithful arms' per-layer quantization — well inside the
card), a `LinAlgError` the fallback does not catch, an `OrthonormalityError`, a kvpress
incompatibility with transformers 5.8 that the compat shim misses, a ShadowKV host-offload
path that does not survive a 16K decode: each is an `error` row, counted, failing reading 1 and
naming its arm. **Nothing is re-run on the pod with a knob changed**; the pod runs the configs
as committed, and a fix is a later commit and a later pod.

## 6. Log volume, and what counts as a complete `<label>.log`

The pod log is the only channel back from a vast.ai instance (`results/<pod>/` never leaves it).
Rows per sample: one `[trial]` line per record, one cell row per (arm, task), and for each of the
twelve `bug`-kind arms the `[diag]` rows `records.drained` prints when the trial's cache is
drained — every other arm emits none.

**None of the twelve gist arms sets `diag_every`, so the cache default (64) applies — the same
`_diag4096` question `prereg/ss2_families.md` §8 worked through.** Task 6 created no
`isvd_r64_h256_seed_diag4096` variant, and none is created here. The per-sample count is
measured, not estimated, on the live real-text pod, which runs two of these arms at the same
default on the same model and context (read 2026-09-19 08:06 EDT; every `[diag]` line in the
watchdog's raw capture parsed, none truncated):

```python
import json, collections, pathlib
raw = pathlib.Path("results/filler_realism/filler_realism-51553610.raw")
d = [json.loads(l[7:]) for l in {x for x in raw.read_text(errors="replace").splitlines() if x.startswith("[diag] ")}]
per_sample = collections.Counter((r["arm"], r["task"], r["idx"]) for r in d)  # idx = trial; two seeds per idx
print(len(d), collections.Counter(r["arm"] for r in d), sorted(set(per_sample.values())))
print(sorted(collections.Counter((r["arm"], r["absorbs"]) for r in d if r["absorbs"] % 64).items())[:8])
# 25376 Counter({'bugSseed-r64-h256': 19968, 'bugSseed-r64-h256-q4': 5408}) [416, 832]
```

**416 `[diag]` rows per 16K sample** = 13 rows per layer × 32 layers — twelve completed
64-absorb windows plus `drain_diag`'s flush of the open one — at **804–810 absorbs per layer**,
identically for the seeded r64 arm and its q4 cell (832 per `(task, idx)` = two seeds × 416; the
q4 cells still open at the time of reading hold 416 = one seed). The absorb count is set by the
prompt length and the cache's absorb schedule, not by the rank or the tier: every `bug`-kind arm
takes the first 4096-token chunk in `prefill_block_size` = 128-column sub-blocks (32 absorbs)
and the remaining ≈ 12.3K tokens in `absorb_block` = 16-column blocks (≈ 770) — the 806 measured
against the naive 1,025 — and the `_prefill` / `consolidate` loops are shared by every
`retention`, `tracker` and rank. So **each of the twelve gist arms emits ≈ 416 rows per sample,
and 544 (17 per layer, the naive count) is the bound** used below; v2 prompts are the same
length as the in-house ones (`_window` stops at ≥ 16,384 haystack tokens plus needles and
template, as v1's builder did).

| per sample | `[diag]` | `[trial]` |
| --- | --- | --- |
| any gist arm (29–40), 16K | 416 (≤ 544) | 1 |
| any other arm (1–28) | 0 | 1 |

| the whole pod | rows |
| --- | --- |
| `[diag]`: 12 arms × 60 samples × 416 | **299,520** (≤ 391,680 at the bound) |
| `[trial]`: 40 × 60 | 2,400 |
| cell rows: 40 × 5 | 200 |
| `[stage]` (model, corpora, four `materialize`), ENV block, markers | ≈ 40 |
| **expected deduped `<label>.log`** | **≈ 302,000 lines** (≈ 394,000 at the bound; ≈ 83–108 MB at the measured 275 B/row) |

**Does the pod need `_diag4096` variants of the gist arms? No — by the arithmetic that decides
it.** The only way the watchdog loses a row for good is a **poll-to-poll gap**: more raw log lines
printed between two 150 s polls than the `vastai logs --tail 30000` window holds (5,000 when the
empty-fetch fallback is taken). A sample's `[diag]` rows are printed in one burst when its trial
drains, so the burst per poll is bounded by how many gist samples can *complete* in 150 s. The
fastest gist arm is the rank-1 `evict_surprise_h256` (its augmented SVD core is 17 × 17 per
absorb against the r64 arms' 80 × 80; billed at the r64 rate in §7 but plausibly nearer
`full`'s 0.6 min): at the ≥ 0.6 min (36 s) floor, 150 s / 36 s = 4.2, so **up to five**
samples complete inside one poll (both endpoints) — **≤ 5 × 544 + 5 = 2,725 rows per poll at the
bound (2,085 at the measured count)**, 11× under the 30,000-line window and 1.8× under the
5,000-line fallback; the r64-class arms at ≥ 2.1 min per sample give ≤ 2 samples (≤ 1,090), the
r128/r256 arms at ≥ 4 min ≤ 1 sample (≤ 545). The unfiltered raw log (download progress,
warnings) would have to add > 2,275 lines in 150 s on top of the worst burst to open a gap.
Nothing in the readings needs one summary row per layer (the data shape the Table-4 pods set
`diag_every: 4096` for); the
union of every poll's matched rows, deduped at the terminal marker, is the record, and the run's
≈ 300,000 rows never have to fit inside any one fetch. Keeping the shipped arms also keeps every
record key the archive and `scripts/tables.py` know. **If the r128/r256 arms had needed
variants, ten more arm files would have been the cost; they do not, and none is made.**

**The cost of the default, stated — and a launch precondition.** `scripts/pod/watchdog.sh`
appends *every* matched row of each fetch to `<label>.raw` and dedupes only at the terminal
marker (`sort -u` → `<label>.log`). Once the instance log passes 30,000 matched lines — after
≈ 70 gist samples, i.e. early in arm 30 — the tail is saturated and every poll re-appends
≈ 30,000 rows ≈ 8.3 MB (275 B/row measured on the live pod). Naively — saturated from t = 0 of
the whole pod — `<label>.raw` would reach **≈ 33 GB** at this pod's bar (§7, 168 h = 4,032
polls) and **≈ 17 GB** at the point estimate (84 h); both are upper bounds, since the cheap
first half emits no `[diag]` row and never saturates the window itself (below). Billing only the
gist phase as saturated (post arm 28, 35.0 h of compute) tightens that to **≈ 26 GB at the bar,
≈ 10 GB at the point**; the gist half alone (its own separate launch) is unchanged at ≈ 19 GB at
its bar, ≈ 10 GB at its point. That is disk and a slow final dedupe inside
the iCloud-synced tree the launch machine runs from (D-007), not loss — but it is not a size the
`filler_realism` pods (≈ 95 MB raw after four hours) have exercised. The first half alone never
saturates the window (≈ 1,900 matched rows in total; its raw stays under 1 GB at its bar).
**Pre-registered: any launch that includes a gist arm waits for the one-line per-poll dedupe
`prereg/ss2_families.md` §8 queued for L6** (`sort -u -o "$H/${lab}.raw" "$H/${lab}.raw"` after
the append; the deduped raw then stays at the ≈ 100 MB of distinct rows) — a commit to the
watchdog made only when no watchdog is running it (the `filler_realism` watchdog is running it
now), strictly before the launch commit, named in the launch's DECISIONS entry. The owner may
waive this by naming the disk plan instead.

**The watchdog's own clock.** `BUDGET_ITERS` defaults to 600 polls × 150 s = 25 h, after which
the watchdog **destroys every instance in `pods.txt`** (D-011 addendum 2's fix). Every bar in §7
exceeds 25 h, so the watchdog must be started with `BUDGET_ITERS` ≥ bar × 24 + margin —
**`BUDGET_ITERS=4100` for the whole pod (168 h), 1800 for the first half (72.8 h), 2400 for the
gist half (97.5 h)** — on the launch line in DECISIONS. `pod.py launch --max-hours` (default:
`gpu_budget_h`) is the bar the pod enforces on itself; the two clocks must agree.

**The completeness test is on the deduped `<label>.log`.** Exact: `grep -c '^\[trial\]'` = 2,400
(1,680 for the first half, 720 for the gist half) and `pod.py check` finds all 200 (140 / 60)
cells at n = 12 — that is the gate. Approximate: `grep -c '^\[diag'` ≈ 299,500 for any pod that
includes all twelve gist arms (the per-layer window count moves by one if a sample's absorb count
crosses a multiple of 64; v2 prompts vary by a few tokens across haystacks, so ± 32 rows per
sample is expected noise), and `manifest.diag_skipped` = 0 (a `[diag]` line the fetch cut in half
is counted, not dropped, and fails `check`). A `.log` whose `[trial]` count is short is short by
construction — read `trials.jsonl` and the `error` lines before concluding truncation, and do not
harvest it as a validated smoke.

## 7. Budget

**Unit and anchors.** "min/sample" is the wall-clock of one `(arm, task, seed, trial)`; a cell
is 12 samples and the five-task context is **60 samples per arm** (the tasks-per-cell
correction of `prereg/filler_realism.md` §GPU budget). Rates are per class, from the task brief,
with the anchor behind each — all on an A100 40 GB:

- **`full` 1.0**: measured ≈ 0.6 min/sample on the real-text pod (48 samples ≈ 30 min;
  `prereg/filler_realism.md` A2.5); budgeted at 1.0 as every pod since has.
- **presses 1.0** (arms 2–16): no rate on record for a kvpress press under the L0 runner. A
  single-shot 16K prefill is one forward (≈ 1–2 s), the press scoring one pass over the
  prefill's attention, the decode 48–64 tokens at `full`'s ≈ 26 ms/token; PyramidKV adds one
  forward per query token (≈ 50), the SVD oracle 512 thin SVDs of a 16K × 128 slab. Budgeted at
  1.0 = 1.7× the `full` measurement.
- **quant 1.5** (arms 17–22): the streaming arm's 43.7 vs 25.9 ms/token decode ≈ 1.7× `full`
  (`prereg/filler_realism.md`); the faithful arms add one post-hoc quantization pass per layer,
  inside the estimate.
- **composites 1.5** (arms 23–26): a single-shot press plus the quantized store — the quant
  class's rate.
- **shadow 2.0** (arms 27–28): **an estimate, no rate on record** (the archived ShadowKV rows
  came from `w11_baselines.sh` and left no timing). The port keeps the whole value cache on the
  host and, per decode step and layer, concatenates the new value on the CPU and gathers the
  selected chunks' values host → device — a host round trip per layer per step over ≈ 100 steps.
  Priced above the quant class for that reason.
- **gist r64 3.0** (arms 29–35, incl. the rank-1 `evict_surprise_h256`, the two Oja arms and
  the FD arm): **2.1 min/sample measured** (W19 `a1q`: "≈ 25 min per 16K cell" = 25/12;
  `prereg/filler_realism.md` A1.4), budgeted at 3.0 since. Two whole-run cross-checks on the L0
  runner + guard, both estimates bracketed from timestamps and not measurements: the cycle pod
  ran 48 r64 + 48 `full` samples in 2.93 h from instance creation to `ALL_DONE` (manifest
  `launched_at` 08:39:09 UTC → D-005 addendum 3's 07:35 EDT), which after ≈ 29 min of `full` and
  ≈ 15–20 min of boot puts r64 at **≈ 2.6–2.8**; on the real-text pod the r64 arm ran from
  `full`'s last cell (by 04:30 EDT, D-005 addendum 2) to its own last cell (between the 06:47 and
  07:13 EDT reads recorded in `prereg/ss2_families.md`), **≈ 2.3–3.0 h for 48 samples ≈ 2.9–3.7**
  — at or a little above 3.0 on real text, inside the 2× bar. The Oja arms replace the SVD core
  with a rank-64 Oja update and the FD arm with a sketch of the same size; all three are billed
  at the r64 rate, no rate of their own being on record (the Week-20 swap pod's Oja cell is
  void and its FD cell never completed a trial).
- **q4 3.5** (arm 32): the r64 arm plus the coordinate tier's quant/dequant
  (`prereg/filler_realism.md`). The live pod's q4 arm closed its `niah_single` cell and was at
  13 samples at 08:03 EDT, having started between 07:04 and 07:13 (the r64 arm's last cell) —
  **≈ 3.8–4.5 min/sample so far**, above the 3.5 budget and inside the 2× bar; at 4.5 its 60
  samples here cost 270 min against the 210 budgeted, + 1.2 % of the pod's compute, which does
  not move the bar.
- **gist r128 4.0** (arms 36–37): between the r64 measurement and the r256 one; the Week-11
  handover's r128 anchor (≈ 5 min/trial at 16K on the older harness, `prereg/filler_realism.md`
  §GPU budget) sits inside the 2× bar.
- **gist r256 6.0** (arms 38–40): **5.2 min per 16K sample measured** on the Table-4 r256 arms
  (perplexity axis, 16K + 512 windows; D-011 addendum 2), budgeted at 6.0 — the retrieval sample
  has ≈ 4 % fewer absorbs and ≈ 100 decode forwards through the reconstruct path (4–14× `full`'s
  per-token cost, CLAUDE.md), which is seconds against the prefill's minutes.
- **Overhead 90 min**: 60 (boot, clone, `pip`, the quanto JIT build, the weight download that
  stalled ≈ 1 h on a Table-4 pod — D-011 addendum 2, `prereg/filler_realism.md` A1.4) + **10 for
  haystack materialization** (four sources × 64 documents, streamed from Hugging Face at the
  pinned revisions; the L2.3b proof run took 1.1–5.1 s per 2 documents, PG-19's 4-MB books the
  slowest) + **20 for per-trial prompt construction** (`make_trial` is called per trial per arm,
  not memoized across arms: the sentence split of the document and ≈ 800 tokenizer calls,
  ≈ 0.5 s × 2,400 samples).

The table, from the pod YAML and the rates above (the snippet is the derivation; every figure
below is its output):

```python
from kvdlra.eval.config import load_arm, load_pod, load_task
pod, task = load_pod("l2_smoke"), load_task("ruler_v2_16k")
per_arm = len(task.tasks) * task.n_trials * len(task.seeds)  # 5 x 12 x 1 = 60
RATE = {"full": 1.0, "press": 1.0, "quant": 1.5, "quant_faithful": 1.5, "composite": 1.5, "shadow": 2.0}
GIST = {64: 3.0, 128: 4.0, 256: 6.0}  # by storage rank; rank 1 billed at the r64 rate
def rate(name):
    a = load_arm(name)
    if a.kind != "bug":
        return RATE[a.kind]
    return 3.5 if a.cache.get("quant_bits") else GIST[max(64, int(a.cache["rank"]))]
def hours(arms):
    compute = sum(per_arm * rate(n) for n in arms)
    overhead = 60 + 10 + 0.5 * per_arm * len(arms) / 60
    return compute / 60, (compute + overhead) / 60
gist = [n for n in pod.arms if load_arm(n).kind == "bug"]
for label, arms in (("whole", pod.arms), ("first half", [n for n in pod.arms if n not in gist]), ("gist half", gist)):
    c, p = hours(arms)
    print(label, len(arms), f"compute {c:.1f} h point {p:.1f} h bar {2*p:.1f} h ${0.40*p:.1f}-{0.74*p:.1f} / ${0.80*p:.1f}-{1.48*p:.1f}")
```

| class | arms | min/sample | × 60 × arms | minutes |
| --- | --- | --- | --- | --- |
| `full` | 1 | 1.0 | | 60 |
| presses | 15 | 1.0 | | 900 |
| quant | 6 | 1.5 | | 540 |
| composites | 4 | 1.5 | | 360 |
| shadow | 2 | 2.0 | | 240 |
| gist r64-class (rank 1, r64, oja ×2, fd) | 6 | 3.0 | | 1,080 |
| gist r64 q4 | 1 | 3.5 | | 210 |
| gist r128 | 2 | 4.0 | | 480 |
| gist r256 | 3 | 6.0 | | 1,080 |
| **compute** | **40** | | | **4,950 min = 82.5 h** |

| pod | compute | + overhead | point estimate | **`gpu_budget_h` (2× bar)** |
| --- | --- | --- | --- | --- |
| **`l2_smoke`, whole** (as committed) | 82.5 h | + 90 min | **84.0 h** | **168.0** |
| the cheap first half — arms 1–28, everything but the gist arms | 35.0 h | + 84 min | 36.4 h | 72.8 |
| the gist half — arms 29–40 | 47.5 h | + 76 min | 48.8 h | 97.5 |

At the **$0.40–0.74/h** the Table-4 and filler pods paid for an A100 40 GB (D-011 and its
addenda; `prereg/filler_realism.md` A1.4, A1.7):

| | GPU-h | × $0.40 | × $0.74 |
| --- | --- | --- | --- |
| whole, point | 84.0 | $33.6 | $62.2 |
| **whole, bar** | **168.0** | **$67.2** | **$124.3** |
| first half, point / bar | 36.4 / 72.8 | $14.6 / $29.1 | $26.9 / $53.9 |
| gist half, point / bar | 48.8 / 97.5 | $19.5 / $39.0 | $36.1 / $72.2 |
| the three r256 arms alone (18.0 h of compute, 22 % of the pod) | 18.0 | $7.2 | $13.3 |

**This experiment asks for ≈ 84 GPU-hours expected, 168 at the bar — $34–62 expected, $67–124
at the bar. Said plainly: that is twice what the task brief's "~30 arms" estimate anticipated,
because the rule of §3 yields 40 arms and the three r256 arms cost 6 min a sample.** It is the
pod that validates every arm end to end, and it is the owner's call whether to run it whole or
in halves:

- **Whole:** one instance, ≈ 3.5 days expected, 7 at the bar; the watchdog and the raw-log
  precondition of §6 sized for it. One launch, one DECISIONS entry, one harvest.
- **In halves:** the cheap first half first (28 arms, ≈ 36 h, $15–27 expected) — it validates
  generator v2, the pairing invariant, every press (G2 line 6), every quant arm, the composites
  and ShadowKV, and it emits no `[diag]` row, so §6's raw-log precondition does not bind it;
  then the gist half (12 arms, ≈ 49 h, $20–36 expected), which is where the r256 cost and the
  log volume sit. The split is made in the commit that precedes the launch: this YAML's `arms:`
  becomes arms 1–28 and a new `configs/pods/l2_smoke_gist.yaml` (same model, dtype, image, task,
  prereg; arms 29–40; `gpu_budget_h: 97.5`) carries the rest; this YAML's `gpu_budget_h` becomes
  72.8; the `# --- L2.5b` test's set pin then reads over the union of the two pods' arms, in the
  same commit. Both halves can also run on two instances at once, in which case the wall clock
  is the longer of them. An instance death does not sink what it already ran: the surviving
  arms' records are kept at harvest, and only the missing arms relaunch — as a new pod, a new
  manifest under the same prereg — so the prices above are the worst case.
- **Cutting single arms** (the third pre-registered way to spend less): the class table above
  prices any subset; the three r256 arms are the largest single saving (18 GPU-h expected).
  An arm cut from the smoke is an arm **not validated** and is listed as such in the harvest's
  DECISIONS entry; it cannot enter a gate pod until a later smoke covers it.

**Credit.** $93.75 at 07:50 EDT today (the running watchdog's own reading), with the real-text
filler pod still billing at $0.449/h against its 18.3 h bar (≤ $6.4 more at the bar, ≈ $2.5
more at its point). The whole smoke's bar at the top rate ($124) exceeds the credit on its own;
the `ss2_families` pods pre-registered beside it ask for another $49–90 at their bars. **Both
wait on the D-003 top-up, and the top-up precedes the launch commit.** The first half alone at
its bar ($29–54) fits the remaining credit at either rate.

`gpu_budget_h` in the pod YAML is the pre-registered bar, enforced on the pod itself by `pod.py
launch --max-hours` (`boot.sh` runs the entrypoint under `timeout`; a run that reaches the bar
prints `===RUN_TIMEOUT_…===` and is harvested as `RUN_FAILED` with `timeout: true`). Overrun is a
stop-and-report, not a silent extension: a pod still running past its bar is a pod to kill and
diagnose (`prereg/hygiene_table4.md` §9), and the first r256 samples' measured rate is the first
thing to read against the 6.0 assumption — the arm order puts them last precisely so that
everything before them is already on the log.

## 8. Provenance

- Pod: `configs/pods/l2_smoke.yaml`. Arms: the 40 files of §3 — **none new, none edited**.
  Task: `configs/tasks/ruler_v2_16k.yaml`, unchanged. Test: `tests/test_pod_manifest.py`
  `# --- L2.5b` (the pod resolves in-process — every arm through `frontier.build_arm` at
  t = 16384, as the runner does before its first trial; the arm-set rule; the task at n = 12;
  the nine gate arms present; no `ojakv*` stem).
- **This file must be committed strictly before the launch commit.** `scripts/pod.py launch`
  refuses otherwise (a pushed SHA, the prereg's first commit a strict ancestor of the launch
  SHA, a clean tree), and `scripts/pod.py check` re-checks the order against the manifest's
  `git_sha` at harvest. The commit that adds this file adds the config and the test and launches
  nothing.
- Launch: `scripts/pod.py launch --pod l2_smoke --offer <id>` (`--max-hours` defaulting to
  `gpu_budget_h`); the watchdog under `caffeinate -s -i` with `BUDGET_ITERS` per §6; the pod
  self-destructs `GRACE_S` after its final marker. The launch is a `DECISIONS.md` entry with the
  pod name(s), this file's first-commit SHA, the launch SHA, offer, rate, bar, `BUDGET_ITERS`,
  and — if the pod is split or an arm is cut — the split commit's SHA and the arms not validated.
  The raw-log precondition of §6 (or the owner's waiver) is named there.
- Outputs: `results/l2_smoke/` (and `results/l2_smoke_gist/` if split) with `manifest.json`
  (git SHA, config hash, model revision, `dataset_sha256` for the four haystack sources,
  torch/CUDA/transformers versions, GPU, wall clock, command line, `errors`, `records`,
  `diag_skipped`, `timeout`), `trials.jsonl` (200 cells × 12; every row with `prompt_sha256`,
  `haystack_id`, `depth`, `code_family`), `diag.jsonl` (the twelve gist arms' rows), `env.txt`
  (rebuilt from the log's ENV block), `pods.txt`.
- The harvest's DECISIONS entry records the six readings of §4 as pass / fail with the evidence
  path, ticks G2 line 6 on a pass of reading 5, and lists every arm that did not validate with
  its exception. **No number from this pod is cited** (§1); `make tables` reads nothing from
  `results/l2_smoke/`.

## 9. What this pod does not decide

Whether the cycled generator is retained (D-005, closing at the real-text pod's harvest);
anything about the r64 configuration's standing on generator v2 (its accuracy here is expected
low and is not read — L3's Gate-1 pods at n = 24 are where generator-v2 accuracy is measured and
compared); the ss2 question (`prereg/ss2_families.md`); the memory or stored-bits comparison
between any two arms (billed elsewhere; `ratio` / `sbits` are recorded here only to show the
accounting ran); the perplexity axis (no `ppl` task, and the PyramidKV arms have none);
official RULER or LongBench; OjaKV (D-017); anything about the tracker, the guard's tolerances,
the exact tier, the kernel, or which arms belong in the paper. Lane item 8 asks one question —
does every arm run end to end on generator v2 on the pod path — and this pod answers that one.

**STATUS: awaiting owner go (DECISIONS).**
