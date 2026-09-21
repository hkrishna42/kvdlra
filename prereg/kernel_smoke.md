# Pre-registration — `kernel_smoke` (the Week-3 kernel measurement; Gate 3)

**STATUS: awaiting L4's kernel (DECISIONS D-002).** The kernel does not exist: there is no
`src/kvdlra/kernel/` package, no kernel arm YAML, and no kernel test in this repository at the
commit that adds this file. What is pre-registered here is **the measurement** — the grid, the
protocol, the numbers it is read against and what would count as passing — written before the
thing it measures is built, so the Week-3 gate cannot be re-shaped around whatever the kernel
turns out to do. ADR 0001 (`docs/adr/0001-factored-attention-kernel.md`) is **OPEN**; D-002 is the
decision that accepts it. Lane L4; gate G4.

The pod itself (`configs/pods/kernel_smoke.yaml`), the kernel arm's config and any task-file
change §3 names are committed **by lane L4**, in the commit that precedes the launch commit and
after this file's first commit. `scripts/pod.py launch` refuses to create the instance unless this
file's first commit is a *strict* ancestor of the launch commit, so the commit that adds this file
launches nothing.

---

## 1. Purpose

The gist is factored **pre-RoPE**, so `q·K_t ≠ (Uᵀq)·c_t` and today's decode path reconstructs:
`_ensure_mid_cache` rebuilds the whole n-dimensional middle from `U·C`, re-rotates it at true
positions and casts it to bf16, and `_decode_peek` concatenates `[sinks | exact tier | gist |
recent]` into a dense `n × T` array that attention reads every step (ADR 0001 §1). The storage
ratio therefore never reaches memory or throughput — which is why CLAUDE.md's settled facts say
"Measured decode: reconstruct path is 4–14× slower than full KV and its KV peak exceeds full KV.
No systems claim is made until the kernel replaces it."

**The question this pod answers, verbatim from `docs/plan/lanes/L4_kernel.md` item 3:**

> Gate: kernel KV peak < full KV at 32K and ms/token < reconstruct path by ≥ 3×.

and, verbatim from `docs/plan/lanes/GATES.md` §G4 line 3:

> `results/kernel_smoke/manifest.json`: full / reconstruct / kernel at 16K/32K/64K, b=1,4;
> kernel KV peak < full at 32K; ms/token < reconstruct by ≥ 3×

Both are read from `latency.jsonl` by §4. **This pod as pre-registered is cell list A — batch 1
only (§3) — so line 3's "b=1,4" is ticked in two halves, not one: §4 (3) and §10 name which.**
**This pod measures decode cost and peak VRAM and nothing else** (§11): no accuracy claim beyond
the correctness precondition §4 requires *before* the pod is launched, no Gate-1 or Gate-2
reading, and not the Week-5 Gate-3 target, which §4 states as the later bar it is not.

## 2. Measured baseline — the numbers to beat, and exactly where they live

### (a) The Week-20 rows

| ctx | arm | ms/token p50 | KV peak GB |
| --- | --- | --- | --- |
| 16K | reconstruct (`bugSseed-r64-h256`) | **103.25** | **3.25** |
| 16K | `full` | **25.89** | **2.05** |
| 32K | reconstruct | **188.27** | **6.45** |
| 32K | `full` | **27.49** | **4.08** |
| 64K | reconstruct | **507.94** | **12.83** |
| 64K | `full` | **36.97** | **8.14** |

Llama-3.1-8B, A100-40GB, batch 1; `weights_gb` 14.96 on every row; `quant-2bit-kivi` ran beside
them at 43.70 / 67.86 / 118.92 ms and 0.45 / 0.89 / 1.78 GB.

**Where they live, and what they are.** Two places, and **neither is a harvested record**:

- **`results/paper-v1/w19-sysfix-llama/raw/w19-sysfix-llama-lines.txt`, lines 1–9** — the nine
  `[latency ctx…]` lines the Week-20 sysfix pod printed, archived verbatim.
- **`docs/plan/reports/w20-close-report.md` §2** — the same nine rows as a table (rounded to
  103 / 188 / 508 ms), which is what ADR 0001 §1 and `docs/plan/lanes/L4_kernel.md` quote.

There is **no `latency.jsonl` behind them**: `results/paper-v1/w19-sysfix-llama/` holds
`trials.jsonl` and `cells.jsonl` only, and its own `manifest.json` records
`"unconverted": 9` against that source file — the nine latency lines. They are also **not
re-parseable by the shipped harvester today**: `records.LATENCY_RE` requires the `batch=` field
that `latency.run_latency` appends last, and the archived lines predate it, so
`parse_latency_lines` returns **0 rows of 9** on that file (checked at the commit that adds this
file). `configs/pods/w19_sysfix_latency.yaml` is the re-run contract for them and carries
`prereg: null`, which is v1's missing pre-registration recorded as a field.

**Consequence, pre-registered:** the rows above are a **prior**, not a comparison base.
Every reading in §4 is taken **within this pod**, against this pod's own `full` and `reconstruct`
rows measured on the same card in the same session, and the Week-20 rows are quoted beside them as
"to be re-measured by the `full` / `reconstruct` arms of this pod" — which is what the pod does
anyway. A disagreement between them is itself reportable (a different card, a different torch, a
different transformers) and is reported; it changes no reading.

### (b) What the cost model predicts, and how far it can be trusted

ADR 0001 §3's cost model (regenerable by `docs/adr/0001-cost-model.py`, stdlib only) puts option
(iii) — the primary decision, Triton tile-wise reconstruct inside attention — at **resident KV
0.60 GB against full KV's 4.29 GB at 32K** and a **10.7 ms model floor at 32K** against the
reconstruct path's measured 188 ms, so the Week-3 gate is reachable in principle with margin; at
the Week-5 point (64K, batch 8) the model gives 16.2 ms and 9.14 GB against full KV's 54.5 ms and
68.72 GB. **The third of D-002's three concerns bounds what that is worth:** "the traffic model
under-predicts the measured reconstruct path by 6.6–16× (full KV by 2.1–2.3×), so absolute kernel
timings are not predicted, only the floors." §5 predicts accordingly — floors and orderings, not
timings.

### (c) The two constraints that shape the grid, before it is run

Both are D-002's other concerns, and both are design facts here rather than things to discover:

1. **`BugStreamingLayer.lazy_initialization` raises above batch 1** (D-002 addendum concern 1;
   ADR 0001 §5 "Batch > 1 is a prerequisite"). Every batch-4 cell of every `bug`-kind arm — the
   reconstruct arm included — is therefore **conditional on L4 landing a batched cache**, and
   `scripts/pod.py check` admits no partial grid (§3).
2. **Weights + full KV does not fit an A100-40GB at 64K batch 4**: 50.4 GB by the ADR's model
   (concern 2), 47.5 GB from the archived row's own numbers (14.96 weights + 4 × 8.14 KV peak).
   The reconstruct arm busts the same card earlier, at **32K batch 4** (14.96 + 4 × 6.45 = 40.8 GB).
   The replacement contrast is named in §3 rather than improvised on the pod.

## 3. Arms, tasks, the grid — and why the grid is all-or-nothing

**Three arms, one model, one card.** Llama-3.1-8B (`unsloth/Meta-Llama-3.1-8B-Instruct`), bfloat16,
the same image the other pods run:

| # | arm | what it is |
| --- | --- | --- |
| 1 | `full` | the uncompressed cache: the memory and throughput both other arms are read against |
| 2 | reconstruct = `isvd_r64_h256_seed` | today's reconstruct-then-attend path, unchanged — the r64 configuration exactly as Gate 1 runs it |
| 3 | kernel | arm 2 with the fused factored-attention kernel of ADR 0001 option (iii), enabled by **a config knob to be named by L4's ADR 0001 implementation** |

**The kernel arm's knob is deliberately not invented here.** No YAML key is pre-registered for it;
L4 names it in the arm file it commits, and the amendment that commits the pod YAML records the
name. One implementation fact does constrain L4 and is stated now: `kvdlra.eval.latency.run_latency`
dispatches on `kind` and raises `ValueError("latency covers full/bug/quant arms, not …")` for
anything else, so the kernel arm must be a **`bug`-kind arm carrying the knob** (which is also
ADR 0001's shape — the kernel replaces `_ensure_mid_cache` / `_decode_peek` inside the same cache,
and `_absorb_columns` is untouched, so the prefill path is identical to arm 2's) or `run_latency`
needs an edit in the same commit. Either way the pod YAML and the runner are committed before the
launch commit and `tests/test_pod_manifest.py` pins the pod.

**Protocol — `configs/tasks/latency_16k_32k_64k.yaml`, cited field by field.**
`generator: latency`; `ctx: 16384` with `ctxs: [16384, 32768, 65536]` as the sweep
(`runner._latency_rows` loops `for ctx in task.ctxs or [task.ctx]`, rebuilding every arm per
context, because a cache arm's budgets resolve against the context length); `batch_sizes: [1]` as
shipped (see the cell lists below); `n_steps: 64`; `warmup: 8`; `chunk` at its `TaskCfg` default
**4096**, so the `bug`-kind arms prefill chunked (`_prefill_chunked`) and `full` prefills in one
shot. What is timed, exactly (`src/kvdlra/eval/latency.py`): after the prefill and a
`torch.cuda.synchronize()`, `resident_gb = torch.cuda.memory_allocated()` is read and
`torch.cuda.reset_peak_memory_stats()` is called, then **64 decode forwards** of one token at true
positions, each CUDA-synchronized around `time.perf_counter()`, greedy next token; `p50` is the
median of the **56** steps left after the first 8 are discarded, `spikes` counts steps above
2 × p50, and `peak_gb = torch.cuda.max_memory_allocated()` is read after the loop.
`kv_peak_gb = max(peak_gb − weights_gb, 0)` and `kv_resident_gb = max(resident_gb − weights_gb, 0)`,
where `weights_gb` is summed from the model's parameters. **`n_steps` is the total, not an
addition to the warm-up: 64 forwards per cell, 56 of them counted.**

**Records.** One `LatencyRecord` per **(arm, ctx, batch)** in `results/kernel_smoke/latency.jsonl`
(`kvdlra.eval.records.LatencyRecord`: `model, arm, ctx, batch, ms_per_token_p50, ms_mean, ms_max,
spikes, resident_gb, peak_gb, kv_peak_gb, source`), harvested from the `[latency ctx…]` lines by
`records.parse_latency_lines`. **No trials, no seeds, no cells**, and — unlike every other axis —
**no `[diag]` rows**: `records.drained` wraps the retrieval and perplexity samples, not
`run_latency`, so the guard's behaviour under the kernel is not observed on this pod. That is a
stated limit, not an omission (§11).

**One field the record does not carry, and the reading that needs it.** `kv_resident_gb` is
computed by `run_latency` but is **not** in `LatencyRecord` (`runner._latency_rows` writes
`resident_gb`, `peak_gb`, `kv_peak_gb`), and `weights_gb` is printed on the log line but is in no
record. So the Week-3 gate's "KV peak" is directly readable (`kv_peak_gb`), while the Week-5
target's *resident* ratio must be recovered as `resident_gb − weights_gb` from the log line, or by
a record change L4 makes. §4 reads only what the record carries.

**The grid is all-or-nothing, and that is what fixes the cell lists.**
`scripts/pod.py check`'s `_latency_fails` requires one record for **every** (arm, ctx, batch) in
the pod's arms × the task's `ctxs` × its `batch_sizes`; a point that OOMs is an
`[error] axis=latency` line and a counted error, which fails the pod twice over (the missing
record and ruling R29's zero-error rule). A grid containing a cell that cannot fit is therefore a
pod that cannot pass, so the cell list is pre-registered rather than discovered:

- **Cell list A — batch 1, all three arms, 16K / 32K / 64K: 9 cells.** This is the Week-3 gate
  reading, it runs on the shipped task file unchanged (`batch_sizes: [1]`), and it needs no
  batched cache. **This is the pod as pre-registered.**
- **Cell list B — batch 4: 9 more cells, conditional on both of §2 (c)'s constraints being
  cleared**, and launched only by a dated amendment to this file that states which: (i) L4's
  batched cache has landed and `lazy_initialization` no longer raises above batch 1, **and**
  (ii) the card holds every arm's largest cell. By §2 (c)'s arithmetic an **A100-40GB admits
  batch 4 only at 16K** (reconstruct 28.0 GB; at 32K it needs 40.8 GB), while an **H100-80GB
  admits 16K / 32K / 64K** (the largest cell is the reconstruct arm at 64K, 66.3 GB). The
  amendment names the card and commits a task file whose `ctxs` × `batch_sizes` is exactly the
  grid that fits — **a second task file, not an edit of `latency_16k_32k_64k.yaml`**, which is the
  re-run contract of the archived Week-20 measurement (`configs/pods/w19_sysfix_latency.yaml`,
  driven in CI by `tests/test_latency_runner.py`): editing it would silently re-define what those
  rows meant and move that pod's `config_hash`. No live manifest pins that hash today, which is
  what makes the edit tempting and wrong.
- **"Max batch that fits" is not this pod.** ADR 0001 §5 and the plan's §3.2 ask for it where full
  KV OOMs; the (arm, ctx, batch) grid cannot express a per-arm batch, and `check` would fail the
  asymmetric grid. It is a separate, separately pre-registered measurement.

## 4. Readings and decision rule

### The precondition, which is not measured by this pod

**Correctness first, and it gates the launch.** From `docs/plan/lanes/L4_kernel.md` item 2 and
`GATES.md` §G4 line 2, verbatim: single-layer **`max |Δ| < 1e-2` in bf16 vs reconstruct-then-attend**
on random tensors **and** on dumped 8B KV, and full-model greedy decode **token-exact on ≥ 14/16
prompts, mismatches logged**. **The dumped 8B KV this needs names a dependency this file does not
supply**: `GATES.md` §G1 line 6 records the rank-sweep figure's 8B half as still open (1B dumps
landed 2026-09-18; the 8B dump pod has not run), so the single-layer check is blocked on that pod
as well as on the kernel, and the launch entry names whichever one unblocked it. **The 16 prompts
are fixed before the kernel is written, not chosen after seeing its output**: they are committed in
L4's test module, and the launch entry in `docs/plan/DECISIONS.md` names their source and
selection and the path of the mismatch log the < 14/16 case would read. Neither check is a
decode-latency measurement and neither is produced by the `latency` axis, so both must have
**passed and been recorded before the launch commit**, with their evidence path (L4's test module
and, for the full-model check, the GPU run that produced it) named in the launch entry in
`docs/plan/DECISIONS.md`. **A kernel whose correctness precondition has not passed is not measured
for speed**: a fast wrong kernel is not a result, and a pod launched without the precondition is
reported as launched without it. If L4 instead runs the full-model check on this pod, that needs a
runner axis the repository does not have and is an **amendment to this file**, committed before
the launch commit, which also states how its output is recorded.

### The Week-3 gate, read from `latency.jsonl`

The two conditions below are read **within this pod**, per batch, with the arms' rows matched on
`(ctx, batch)`:

1. **Memory.** `kv_peak_gb(kernel, 32768, b) < kv_peak_gb(full, 32768, b)`. The gate names 32K;
   16K and 64K are reported beside it and do not enter the pass. **A pass whose margin is under
   10 %** — `(kv_peak_gb(full, 32768, b) − kv_peak_gb(kernel, 32768, b)) / kv_peak_gb(full, 32768,
   b) < 0.10` — **is reported as marginal, with both raw peaks printed beside it**: a single
   measurement per cell carries no error bar to fall back on (§6), so a thin margin is named
   rather than shown with the same confidence as a wide one.
2. **Speed.** `ms_per_token_p50(reconstruct, 32768, b) / ms_per_token_p50(kernel, 32768, b) ≥ 3.0`.
   Against this pod's own reconstruct row; if that row is within 10 % of the archived 188.27 ms the
   two agree and both are printed, and if it is not, the pod's own row is the one the ratio uses
   and the discrepancy is reported (§2 a).
3. **Pass = both, at batch 1** (cell list A). **This ticks `GATES.md` §G4 line 3 AS AMENDED: the
   b = 1 half of "b=1,4"; the b = 4 half stays open until cell list B runs**, under the card §3's
   amendment names. Cell list B's batch-4 cells, if they run, are reported the same way and are
   **descriptive**: they do not change the Week-3 verdict, because the batch-4 grid is conditional
   on a cache that does not exist at the time this file is committed and its card may differ.

**Refusals, read before the rule.**

- **Any `[error] axis=latency` row, on any arm, at any point of the grid, refuses the verdict** and
  the pod is reported as failed, not as a partial reading: the error is counted into the manifest,
  `scripts/pod.py check` fails the pod under ruling R29 (no tolerance knob), and a missing
  (arm, ctx, batch) record fails `_latency_fails` independently. An OOM is an error like any other.
  Nothing is re-run on the pod with a knob changed; a fix is a later commit and a later pod.
- **A `spikes` count above 8 of 56 steps on any arm refuses that arm's p50 as a steady-state
  number** — **8 is twice the archived handful of 4** (below), a margin rather than a fitted
  threshold: no distributional model of `spikes` is claimed here, only that the archived rate
  doubled is still an anomaly. It is reported with `ms_mean` and `ms_max` beside it and the cell
  is read as "not a steady state", never as a faster median. **A refused p50 on the reconstruct
  arm or the kernel arm at 32K refuses the Week-3 gate's speed condition** — §4 (2) above reads
  exactly those two p50s — **and therefore refuses the pass** (§4 (3)): a numerator or denominator
  that is not a steady-state number cannot be compared to the 3.0× threshold, whatever the ratio
  would print. The reconstruct arm's archived rows carry **4 spikes at 16K and 32K and 0 at 64K**
  (its absorb-event rebuild), so a handful is expected there and **zero** is expected on `full`; a
  spiking *kernel* arm would mean the absorb-event rebuild did not actually leave the decode path,
  which is the claim the kernel rests on.
- **No arm is ever printed as `--`.** An arm that did not run reads `not run` with the reason, and
  an arm that errored carries its exception text — the rule
  `prereg/gate1_tracker_swap_v2.md` §4 fixes, applied here for the same reason.

### What the pod does not read

**The Week-5 Gate-3 target is the later bar, and is not this pod's:** "resident KV ≤ 0.25× full at
32K and tokens/s ≥ full KV at 64K, batch ≥ 4" (`docs/plan/lanes/L4_kernel.md` item 3;
`ICML2027_PLAN.md` §2 Gate 3, whose ADR-quoted form allows Branch A on the memory axis alone). Its
resident ratio is not in `LatencyRecord` (§3) and its batch ≥ 4 cells are cell list B. It is
recorded here so that a pod that meets the Week-3 gate is not reported as meeting Gate 3.

## 5. Predictions, written before the kernel exists

| quantity | prediction | basis |
| --- | --- | --- |
| kernel `kv_peak_gb` at 32K, b1 | **below full KV** — the model says 0.60 GB against 4.29 GB, i.e. a 7× margin on a gate that asks only for "<" | ADR 0001 §3's traffic model; the resident term is the stored footprint, which is measured, not modelled |
| kernel `ms_per_token_p50` at 32K, b1 | **≥ 3× faster than the reconstruct path** — the threshold is 188.27 / 3 = **62.8 ms**, so the kernel may miss its own 10.7 ms model floor by **5.9×** and still pass | ADR 0001 §4 and §6's Week-3 milestone ("ms/token ≤ 62.8 at 32K … model floor 10.7 ms"); **the absolute time is not predicted** (D-002 concern 3: the model under-predicts the measured reconstruct path 6.6–16×) |
| kernel vs `full` on ms/token | **not predicted, either direction.** The Week-3 gate does not ask for it, and at 32K the model's 10.7 ms against full's measured 27.5 ms sits inside the model's own error bar | D-002 concern 3 |
| reconstruct arm, all cells | **reproduces the Week-20 rows within ~10 %** on the same card | same protocol, same arm, same model; a larger gap is a finding about the harness and is reported (§2 a) |
| `spikes` | 0 on `full`, a handful on reconstruct (4/4/0 archived), **0 on the kernel** | the absorb-event rebuild is what spikes, and ADR 0001 §5 argues it leaves the decode path: `_absorb_columns` rotates retained coordinates eagerly, so exactly one current (U, C) pair exists at every step — no basis versioning and no memo to drop |
| 64K, batch 1 | the widest gap: the reconstruct path is 508 ms there against full's 37 | the reconstruct cost is Θ(T) in n dimensions and the archived rows already show it |

**Failure is a result.** A kernel that misses the 3× at 32K, or whose KV peak sits above full KV,
is reported as measured with the cost model's prediction printed beside it — that combination is
the ADR's own escape hatch ("(i) runs in parallel as an accuracy experiment only … it is insurance
if (iii) misses its FLOP target on some GPU"), and choosing between (i) and (iii) on that evidence
is D-002's, not this file's.

## 6. Family size and correction — none, and why

There is **no inferential statistic on this pod and no p-value anywhere in it.** A cell is one
measurement, not a sample of trials: one prefill, 64 timed forwards, a median over 56 of them, and
two VRAM reads. The dispersion that exists is reported directly — `ms_mean`, `ms_max` and `spikes`
beside every `ms_per_token_p50`, and a thin memory margin flagged the same way (§4's 10 % rule) —
and the gate's two conditions are ratios of medians against fixed thresholds (3.0×, and "<"), not
tests. Inventing a family here would be the cheapest way to
dress a systems measurement as an inference. The pod's discipline is elsewhere: the grid is
complete or the pod fails (§3), the arms are matched on `(ctx, batch)` within one pod and one
session (§4), and the precondition is met before the launch (§4).

## 7. Secondary outcomes

- **The full table, all 9 (or 18) cells** — `ms_per_token_p50`, `ms_mean`, `ms_max`, `spikes`,
  `resident_gb`, `peak_gb`, `kv_peak_gb` per (arm, ctx, batch) — printed whether or not the gate
  passes, beside the Week-20 rows of §2 (a).
- **The analytic footprint beside the measured peak, never instead of it** (ADR 0001 §5's
  measurement plan, and `kvdlra.accounting`'s own convention): `bug_footprint(...).stored_bits()`
  for the two `bug`-kind arms at each context, against their measured `kv_peak_gb`. The gap **is**
  the workspace, and quantifying it is the point: ADR 0001 §1 records that at 32K the stored state
  is 0.56 GB against a 6.45 GB measured peak, i.e. **~91 % of the reconstruct path's peak is
  workspace** — the number the kernel is supposed to remove, and the Week-18 panel's
  storage-vs-resident objection in one line.
- **The archived-vs-remeasured comparison** of §2 (a), reported as a harness reading.
- **Not in this pod, and named as a deviation from ADR 0001 §5's measurement plan**: the KIVI-2
  row (that plan lists "full / reconstruct / kernel / KIVI-2"). A quantizer is not part of the
  Week-3 gate's two conditions, it adds three cells per batch to an all-or-nothing grid, and its
  archived rows (43.70 / 67.86 / 118.92 ms, 0.45 / 0.89 / 1.78 GB) already sit in §2 (a) for
  context. It joins by amendment if the owner wants the systems table complete.

## 8. Log volume, and what counts as a complete `<label>.log`

Small enough that nothing about the watchdog needs saying. The latency axis emits **one
`[latency ctx…]` line per (arm, ctx, batch)** and no `[trial]`, no cell row, no `[pplw]`, no
`ppl=` and **no `[diag]`** (§3):

| the whole pod | rows (cell list A) | + cell list B |
| --- | --- | --- |
| `[latency ctx…]`: arms × ctxs × batches | 3 × 3 × 1 = **9** | 3 × 3 × 2 = **18** |
| `[stage]` lines, model load, ENV block, markers | ≈ 40 | ≈ 40 |
| **expected deduped `<label>.log`** | **≈ 50 lines** (≈ 14 kB at 275 B/row) | ≈ 60 lines |

**The completeness test.** Exact: `grep -c '^\[latency'` = **9** (18 with cell list B) and
`scripts/pod.py check results/kernel_smoke` returns 0 — which for this pod means the config hash,
the commit order against `manifest.git_sha`, `env.txt` against the pyproject pins, `errors` = 0,
and `_latency_fails` finding a record for every (arm, ctx, batch). A `.log` with fewer
`[latency]` lines than the grid is a pod whose missing points must be read off the
`[error] axis=latency` lines before anything else is concluded. `BUDGET_ITERS` needs no override
(its default is `(gpu_budget_h · 3600 + 7200)/150 + 1` polls with a floor of 600, and both of §9's
bars land under the floor — 145 polls at 4.0 h and 241 at 8.0 h — so the watchdog runs its
600 polls = **25 h**, covering either bar plus `boot.sh`'s 2 h `GRACE_S` self-destruct with room),
and no `[diag]` burst can open a poll-to-poll gap because there are no `[diag]` rows.

## 9. Budget

**Unit.** A cell is one prefill plus 64 decode forwards. There is no "min/sample" here: the cost
is prefill-dominated and the decode loop is bounded by the measured ms/token of §2 (a).

- **Prefill, per arm per context**, billed at the measured retrieval-sample rates
  `prereg/gate1_tracker_swap_v2.md` §9 fixes — **`full` 0.6 and the r64 gist 3.1 min/sample at
  16K** — scaled **1× / 2× / 4×** for 16K / 32K / 64K (that file's Stage-2 rule, "2× the 16K rates
  (twice the absorbs)", extended one step). Those rates include a needle generation the latency
  axis does not run, so each is an over-bill, which is the conservative direction. The kernel arm's
  prefill is the reconstruct arm's: ADR 0001 §5 leaves `_absorb_columns` untouched.
  `full` 0.6 + 1.2 + 2.4 = **4.2 min**; reconstruct **21.7 min**; kernel **21.7 min**.
- **Decode, per arm**, 64 forwards × the measured p50s of §2 (a): `full`
  64 × (25.9 + 27.5 + 37.0) ms = **0.10 min**; reconstruct 64 × (103.3 + 188.3 + 507.9) ms =
  **0.85 min**; kernel ≤ the reconstruct path's by the gate itself, **≤ 0.85 min** (≈ 0.04 at the
  model floor).
- **Overhead 60 min**: boot, clone, `pip`, and the weight download that stalled ≈ 1 h on a Table-4
  pod (D-011 addendum 2). No haystack materialization — the latency generator builds a random
  token tensor (`torch.randint`, seeded) and loads no corpus.

| | compute | + overhead | point estimate | **`gpu_budget_h`** |
| --- | --- | --- | --- | --- |
| **cell list A** (batch 1; 9 cells) | 4.3 + 22.6 + 22.6 = **49.5 min = 0.83 h** | + 60 min | **1.8 h** | **4.0** |
| cell list A + B (batch 4 billed 4×, if the amendment runs it) | 0.83 + 3.30 = **4.1 h** | + 60 min | **5.1 h** | **8.0** |

At the **$0.45–0.74/h** of `prereg/gate1_preflight.md` §7 (the A100-40GB band, its floor rounded
*up* from the observed $0.40, so the low end is the conservative one): **cell list A costs $0.8–1.3
expected and $1.8–3.0 at its 4.0 h bar; A + B costs $2.3–3.8 expected and $3.6–5.9 at its 8.0 h
bar.** **This pod is cheap and the pre-registration is most of its cost.** Billing batch 4 at 4×
the batch-1 cost is deliberate over-billing: the decode step is weight-bandwidth-bound at these
shapes (ADR 0001 §4's crossover table), so batch 4 costs well under 4× in practice.

**The H100 rate is unmeasured in this repository.** Every rate in `docs/plan/DECISIONS.md` and
`docs/plan/STATE.md` is an A100-40GB rate ($0.40–0.74/h observed, D-005 addenda and D-011
addenda 1–6). If cell list B moves to an H100-80GB (§3), its hourly rate is **not predicted here**;
the amendment that commits that grid records the observed rate with the offer, and the dollar
figure above does not transfer.

**The bar, and the overrun rule.** `gpu_budget_h` is **4.0 for cell list A** — 2.2× the point
estimate, the 2× discipline of the sibling pods, the extra margin covering a 64K prefill slower
than 4× the 16K one. The **8.0 for A + B is 1.6×, not 2×, and the deviation is named rather than
rounded away**: the batch-4 half is already billed at a deliberate 4× over-bill (above), so the
point estimate it is 1.6× of is itself conservative; an amendment that prefers the plain 2× rule
sets 10.0 and says so. `pod.py launch --max-hours` defaults to whichever is in the pod YAML and
`boot.sh` runs the entrypoint under `timeout`; a run that reaches the bar prints
`===RUN_TIMEOUT_…===` and is harvested as `RUN_FAILED` with `timeout: true`. **Overrun is a
stop-and-report, not a silent extension**, and the first thing to read against these assumptions
is the wall clock at which the first `[latency ctx16384]` line appears — ≈ 61 min in (boot plus
`full`'s 16K prefill and its 64 decode steps) against a 4 h bar.

**Credit.** $92.52 (`vastai show user --raw`, 2026-09-19 13:50, as `prereg/gate1_preflight.md` §7
records it). This pod's bar fits inside it many times over; it is not what D-003 is about, and it
does not queue behind Stage 1 on money — only on the kernel existing.

## 10. Provenance

- **Pod**: `configs/pods/kernel_smoke.yaml` — **committed by lane L4 to match §3 and §9**
  (`prereg: prereg/kernel_smoke.md`, the three arms in the order of §3, `tasks:` naming the task
  file of the cell list being run, `gpu_budget_h` from §9's table), together with the kernel arm's
  config, the task file if cell list B runs, any `run_latency` dispatch change §3 names, and the
  `tests/test_pod_manifest.py` row pinning the pod's prereg path, arm order, task list, a positive
  `gpu_budget_h` and a `config_hash` distinct from every other pinned pod's. **Arms**:
  `configs/arms/full.yaml` and `configs/arms/isvd_r64_h256_seed.yaml`, neither edited by this
  commit or by the launch commit, plus the kernel arm's new file.
- **This file must be committed strictly before the launch commit**, and — unlike the bf16
  reading, which rides another pod — here that order is **machine-enforced**: this file is the
  launching pod's own prereg, so `scripts/pod.py launch` refuses outright unless its first commit
  is a strict ancestor of the launch commit. `prereg_error` covers missing, uncommitted,
  not-a-strict-ancestor and committed-by-the-launch-itself, plus a dirty tree and an unpushed SHA;
  `scripts/pod.py check` re-checks the order against `manifest.git_sha` at harvest. The same
  command that evidences it by hand is
  `git merge-base --is-ancestor <this file's first commit> <launch SHA>`. Those refusals are
  themselves tested: `tests/test_pod_manifest.py::test_prereg_refusal_reasons`,
  `::test_prereg_commit_order_against_real_history`, `::test_launch_refuses_a_pod_with_no_prereg`
  and `::test_check_rejects_tampered_config_hash`. **The commit that adds this file adds nothing
  else and launches nothing.**
- **Amendments only, never edits.** §1–§11 are not edited after the launch commit. Every later
  change is a dated Amendment appended below, in the pattern of `prereg/hygiene_table4.md`: the
  kernel arm's knob name and the pod YAML, cell list B's card and grid, a runner axis for the
  full-model correctness check, and the KIVI-2 row if it is ordered — each committed before the
  launch commit it governs.
- **Launch**: `scripts/pod.py launch --pod kernel_smoke --offer <id>` (`--max-hours` defaulting to
  `gpu_budget_h`), from a pushed SHA on a clean tree; the watchdog under
  `caffeinate -s -i scripts/pod/watchdog.sh kernel_smoke` with no `BUDGET_ITERS` override (§8); the
  pod self-destructs `GRACE_S` = 2 h after its final marker. The launch is a
  `docs/plan/DECISIONS.md` entry naming the pod, this file's first-commit SHA, the launch SHA, the
  offer id, the GPU model, the hourly rate, the bar, the credit before launch, **and the
  correctness precondition's evidence path** (§4).
- **Outputs**: `results/kernel_smoke/` with `manifest.json` (git SHA, config hash, HF model
  revision, torch / CUDA / triton / transformers versions, GPU, wall clock, command line,
  `errors`, `records`, `timeout`), `latency.jsonl` (one `LatencyRecord` per (arm, ctx, batch)),
  `env.txt` (rebuilt from the log's ENV block) and `pods.txt`. No `trials.jsonl`, no `ppl.jsonl`,
  no `diag.jsonl`: this axis writes none of them.
- **Citability, and the renderer that does not exist yet.** A number from this pod is citable only
  once `scripts/pod.py check results/kernel_smoke` passes (§8) **and** a committed entrypoint
  regenerates its table from the committed records (CLAUDE.md's rule). **`scripts/tables.py` has
  no latency subcommand today** — it carries `convert-v1`, `build`, `ppl` and `gate1` and the
  string "latency" appears in it zero times — so **L4 commits the renderer with the pod**, in the
  same commit, writing outside the `table*.md` set that `make tables` diffs against paper-v1
  exactly as `make gate1` does (`docs/paper/tables/gate1.md`), so `make tables` stays diff-clean
  beside it. Until that renderer exists, a row of this pod is a measurement and not a citable
  number, and the DECISIONS entry says which it is.
- **The verdict** — the two gate conditions with their measured numbers, the spike counts, the
  refusal if one fired, the archived-vs-remeasured comparison, and the deviations §3 and §7 name —
  goes to `docs/plan/DECISIONS.md` as the **Gate-3 Week-3 outcome**, under D-002, with the
  evidence path `results/kernel_smoke/` and the table path. **`GATES.md` §G4 line 3 is this pod's
  to tick, and only that one** — **cell list A ticks it AS AMENDED: the b = 1 half of "b=1,4"
  (§1, §4 (3)); the b = 4 half is ticked only once cell list B's amendment runs and reports it** —
  line 2 is ticked by the correctness precondition's own evidence (§4) and line 1 by D-002
  accepting or rejecting ADR 0001 — neither is a reading of these cells.

## 11. What this does not decide

Gate 1 (`prereg/gate1_tracker_swap_v2.md`) and Gate 2 (`prereg/ss2_families.md`): no retrieval,
no perplexity and no accuracy of any kind is measured here beyond §4's correctness precondition,
which is a *precondition* and not a reading. Whether ADR 0001's option (iii) is the right choice
(that is D-002, and this pod is evidence for it, not the decision); option (i)
(`isvd_postrope_r128`), which is a Gate-1-style accuracy row on another pod; the Week-5 Gate-3
targets (§4), which need a batched cache, the resident field the record lacks, and batch ≥ 4;
"max batch that fits" (§3); the KIVI-2 systems row (§7); anything on Qwen2.5-7B or
Mistral-7B-v0.3, on any rank but 64, or at any context but 16K / 32K / 64K; the guard's behaviour
under the kernel, which this axis emits no `[diag]` rows to observe; and the paper's systems
paragraph, which CLAUDE.md's settled facts keep unwritten until a kernel replaces the reconstruct
path. This pod asks one question — does the kernel beat full KV on peak VRAM at 32K and the
reconstruct path by ≥ 3× on ms/token — and nine cells answer that one.

**STATUS: awaiting L4's kernel (DECISIONS D-002).**
