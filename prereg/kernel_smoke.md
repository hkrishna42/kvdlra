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

## Amendment 1 (2026-09-21, lane L4, before the launch commit)

§1–§11 are left exactly as written. This amendment names what §10 said L4 would name, takes §4's
own escape hatch for the full-model check, and re-states the budget and the log volume it
changes. Read it before reading §3, §4, §7, §8 and §9 below it.

### A1.1 The kernel arm, its knob, and the pod YAML (§3, §10)

- The knob is `decode_attention` on the `cache:` block of a `bug`-kind arm: `"reconstruct"`
  (the default, today's path, bit-identical — `tests/test_golden_cache.py`) or `"kernel"` (ADR
  0001 option (iii): a decode step returns only `[sinks | exact tier | recent | new token]` and
  the attention function registered for the attach scope attends the low-rank middle from
  `(U, C)` tile by tile, RoPE in-kernel in fp32 at the true positions, the dense tokens as
  further tiles of the same online softmax). A second knob, `kernel_operand_dtype`
  (`"bfloat16"` default), is the tensor-core input width; the pod runs the default.
  `run_latency` needed no dispatch change: the arm is `bug`-kind.
- Arm 3 is `configs/arms/isvd_r64_h256_seed_kernel.yaml` (commit `99842f2`): arm 2's
  `cache:` block verbatim plus `decode_attention: "kernel"`, record key
  `isvd_r64_h256_seed_kernel`. Prefill, chunked ingest, the absorb step, the exact tier and
  the stored state are arm 2's, so §9's prefill billing for arm 3 stands.
- The pod is `configs/pods/kernel_smoke.yaml`: `prereg: prereg/kernel_smoke.md`; arms
  `full`, `isvd_r64_h256_seed`, `isvd_r64_h256_seed_kernel` in §3's order; tasks
  `kernel_check_16` (A1.2) then `latency_16k_32k_64k` (cell list A, the shipped task file
  unchanged: batch 1, 16K / 32K / 64K); bfloat16 on the -devel image; `gpu_budget_h: 5.0`
  (A1.4). `tests/test_pod_manifest.py::test_the_kernel_smoke_pod_is_its_prereg_design` pins
  the prereg path, the arm order, the task list, the bar and a distinct `config_hash`.

### A1.2 The correctness precondition runs on this pod, first (§4's amendment clause)

§4 says: "If L4 instead runs the full-model check on this pod, that needs a runner axis the
repository does not have and is an amendment to this file, committed before the launch
commit, which also states how its output is recorded." This is that amendment.

- **Why on the pod.** No GPU exists on the lane machine and no 8B dumps exist (`GATES.md`
  §G1 line 6, still open); the check needs Llama-3.1-8B in bf16.
- **The axis.** `generator: kernel_check` (`kvdlra.eval.kernel_check`, commit `27d3593`),
  driven by `configs/tasks/kernel_check_16.yaml` (`ctx: 4096`, `chunk: 1024`, `n_new: 32`,
  `n_prompts: 16`), the first task of the pod. For every `bug`-kind arm with
  `decode_attention: kernel` — arm 3 only — and every prompt: a 1024-token chunked prefill and
  32 greedy decode steps under the kernel, the same under a reconstruct twin built from the
  same kwargs with `decode_attention: "reconstruct"`; a prompt **matches** when the 32 tokens
  agree, and the first mismatching step is recorded. On the first kernel decode step the
  attention function also computes reconstruct-then-attend **in fp32 over the bf16 stored
  representation** (`_decode_peek` + sdpa in fp32 — the reference of
  `tests/test_kernel_reference.py`) on the same live K/V and query for **every layer** and
  records `max|Δ|` per layer; the row keeps
  the worst layer's value and index. This is §4's single-layer check on real 8B K/V, with
  the model's own queries, on all 32 layers, in place of one dumped layer — the 1B-dump half
  runs offline in `tests/test_kernel_reference.py` (layer 8 of `doc411`, skipped in CI).
- **The 16 prompts, source and selection** (§4: "committed in L4's test module"):
  `kvdlra.kernel.prompts` and its pin `tests/test_kernel_prompts.py` — the 16 non-overlapping
  4096-token windows of the PG-19 validation token stream `load_corpus_ids(tok, device,
  corpus="pg19-val")` returns (the stream `ppl_16k_pg19val` cuts its windows from; its sha256
  lands in the manifest's `dataset_sha256["pg19-val"]` from the `[stage] dataset_sha256`
  line), at offsets `i × 131072` for `i = 0..15`; committed at `e22b661`, before any
  kernel code.
- **How the output is recorded.** One `[kernel_check prompt=<i> arm=<key> ctx=4096 n_new=32
  match=<0|1> first_mismatch=<step|-> max_abs_diff=<x|-> worst_layer=<l|-> sha=<sha256>[
  error=<text>]` line per prompt, kept by the watchdog and replayed at the end of the run,
  harvested by `records.parse_kernel_check_lines` into `results/kernel_smoke/kernel_check.jsonl`
  (`KernelCheckRecord`); the mismatch log is the `[kernel_check mismatch prompt= step= kernel=
  reconstruct=` line per non-matching prompt and the `[kernel_check layers prompt=0 ...]`
  per-layer breakdown, both in the harvested `.log`. A prompt that raises is an error row, an
  `[error] axis=kernel_check` line and a counted error (R29 fails the pod). `scripts/pod.py
  check` requires 16 rows for arm 3 (`_kernel_check_fails`); the launch entry's evidence path
  for §4 is `results/kernel_smoke/kernel_check.jsonl` plus `tests/test_kernel_reference.py`
  and `tests/test_kernel_path.py` (the CPU checks: random 8B shapes, the 1B layer, the tiny
  model token-exact in fp32).
- **Reading order.** The check precedes the measurement in the task list, and `make
  kernel_smoke` prints its verdict first: **met** when ≥ 14 of 16 prompts match AND the worst
  `max_abs_diff` < 1e-2; otherwise the Week-3 gate reads **REFUSED** ("a kernel whose
  correctness precondition has not passed is not measured for speed") and the latency rows
  are reported as measured without a passed precondition, not cited. The Triton kernel's own
  tests (`pytest -m gpu`, `tests/test_kernel_triton.py` and the gpu params of
  `tests/test_kernel_reference.py`) are run on the pod before `pod.py run`, from the same
  clone; their output is pasted into the launch entry. **The bars those tests apply**, as
  they stand at this amendment: the pre-registered `max|Δ| < 1e-2` against
  reconstruct-then-attend is unchanged and runs on CUDA through the `BACKENDS` parametrization
  of `tests/test_kernel_reference.py`; `tests/test_kernel_triton.py` adds DIRECT
  Triton-vs-reference comparisons on fp32 outputs — at `n_splits == 1` (B = 16 rows, the whole
  tile loop in one split) `max|Δ| ≤ 2e-3` **and** `rms(Δ) ≤ 1e-4`, and across 16 splits and
  for batch independence `max|Δ| ≤ 4e-3` (rulings R-L4-21 / R-L4-26, commits `6330274` and
  `c3071ce`; each bar sits ≈ 2–10× over the reassociation residual a CPU emulation of the
  blocking measures, and a kernel that passes every other check but the rms bar goes back to
  review, not to a loosened bar). Those comparisons are a **correctness gate on the kernel's
  implementation of the numerics contract, not a reading of this pod**: no number from them
  enters §4 or §7.

### A1.3 The record, the archived rows, and the renderer (§2 (a), §3, §7, §10)

- `LatencyRecord` now carries `kv_resident_gb` (the field §3 said it lacked): `run_latency`
  prints it last, `records.LATENCY_RE` takes it as optional, so the archived lines still
  parse — and `batch=` is optional too (absent → 1), so §2 (a)'s "returns 0 rows of 9" is
  superseded: the nine Week-20 lines parse and `make kernel_smoke` prints them beside the
  re-measured rows with the 10 % agreement note. The Week-5 resident ratio is therefore
  readable from the record when cell list B runs; nothing in §4 reads it now.
- The renderer §10 owed is `scripts/tables.py latency --pods results/kernel_smoke --out
  docs/paper/tables/kernel_smoke.md` (`make kernel_smoke`), commit `b44ad40` — before this
  pod YAML's commit, which satisfies §10's "the renderer with the pod" at least as strictly
  as one commit would. It prints §7's table (every record field per (arm, ctx, batch), the
  analytic `bug_footprint(...).stored_bits()` in GiB beside `kv_peak_gb` — 0.302 / 0.556 /
  1.064 GiB for the r64 configuration at 16K / 32K / 64K, ADR 0001's 0.151× / 0.139× /
  0.133× — the archived row and its agreement), then §4's reading: refusals (manifest
  errors, a missing 32K record, a `spikes` count above 8 of 56 on the reconstruct or kernel
  arm at 32K, the precondition), the memory condition with both peaks and the 10 % marginal
  rule, the speed condition with both p50s against 3.0×, and the verdict; `not run
  (<reason>)` and the `[error]` text for a missing cell, never `--`.
- **Three sentences above are superseded by this section, and named here rather than edited**
  (§1–§11 stay as written, this file being append-only): §3's "`kv_resident_gb` is computed
  by `run_latency` but is **not** in `LatencyRecord`" and its "must be recovered as
  `resident_gb − weights_gb` from the log line", §4's "Its resident ratio is not in
  `LatencyRecord` (§3)" and §11's "the resident field the record lacks" describe the record
  as it stood when they were written; the record carries `kv_resident_gb` and the renderer
  prints it. Only that clause of each sentence is superseded — the Week-5 target's batch ≥ 4
  half is cell list B's and is untouched, so §4's and §11's conclusions stand on their other
  leg.

### A1.4 Budget and log volume (§8, §9)

- The check adds 16 prompts × 2 caches × (a 4096-token chunked prefill of the r64
  configuration ≈ 0.78 min -- ¼ of the r64 gist's 3.1 min/sample at §9's rate (3.3 measured,
  D-011 addendum 10), plus 32 decode steps ≤ 0.1 s each) ≈ **27 min ≈ 0.45 h** of compute.
  Point estimate **1.8 + 0.45 = 2.25 h**; the bar is **`gpu_budget_h: 5.0`** — 2.2× the point,
  the sibling pods' 2× discipline kept (§9's 4.0 would be 1.8×). Cell list A's dollar figure
  moves to $1.0–1.7 expected and $2.3–3.7 at the bar at §9's $0.45–0.74/h.
- Log volume: +16 `[kernel_check prompt=…]` lines, ≤ 16 `[kernel_check mismatch …]` lines,
  1 `[kernel_check layers …]` line, and 3 `[stage]` lines (corpus load, digest, cell clock):
  the expected deduped `<label>.log` is ≈ 90 lines. The completeness test gains
  `grep -c '^\[kernel_check prompt='` = **16** beside `grep -c '^\[latency'` = 9.
- **Two things this pod is the first to put a clock on, and neither is a threshold.** The
  check axis's own wall clock — two full prefill-and-decode runs, the kernel and its
  reconstruct twin, × 16 prompts on arm 3 — is **unmeasured until this pod**: there is no GPU
  on the lane machine, and the ≈ 0.45 h above is an allowance built from §9's prefill rate,
  not a measurement. It is the first number to read against this amendment, the way §9 reads
  the wall clock of the first `[latency ctx16384]` line against its own assumptions. And the
  Triton launch runs at a fixed `num_warps=4` (`src/kvdlra/kernel/triton_kernel.py`), untuned
  for the 8B shape: register spilling there would show in the kernel arm's ms/token and
  nowhere else — a codegen detail the latency reading may expose, which A1.2's correctness
  gate does not depend on and for which no threshold in §4 is adjusted.

### A1.5 What this amendment does not change

The grid (cell list A), the two conditions and their thresholds, the refusals, §6 (no
statistic), the KIVI-2 deviation of §7, cell list B's conditions (§3; the batched cache has
landed — L4.1 — so its condition (i) is met; condition (ii), the card, is still the owner's
D-002/H100 decision), and §11. The 8B rank-sweep dumps (`GATES.md` §G1 line 6) stay open:
this pod does not produce them.

### Amendment 1 — correction (2026-09-21 06:07 EDT, before the launch commit)

A1.2's last paragraph says the Triton kernel's own tests "are run on the pod before `pod.py run`,
from the same clone; their output is pasted into the launch entry". No script performed that step:
`scripts/pod/boot.sh` cloned at `$SHA`, installed the evaluation stack (not the `[dev]` extras — the
pod had no pytest) and handed off to the entrypoint. As worded, the sentence described an intention,
not a mechanism, and a pod launched under it would have run the check axis and the nine latency
cells without ever executing the kernel's correctness gate. This section narrows the sentence to the
mechanism now in the repository. It changes no reading, no threshold and no bar.

- **The command.** `configs/pods/kernel_smoke.yaml` gains one key, `pre_run`, whose value is
  verbatim:

  ```
  python -m pytest -m gpu -q -rA -p no:cacheprovider tests/test_kernel_triton.py tests/test_kernel_reference.py 2>&1 | sed 's/^/[pre_run] /'
  ```

  `-m gpu` selects exactly the items A1.2 names (`tests/test_kernel_triton.py`, whose module is
  gpu-marked whole, and the `triton` parametrization of `tests/test_kernel_reference.py`), and
  `-rA` prints the per-item summary so each one's measured max|Δ| and rms(Δ) — the values those
  tests print into their assert messages — ride the pod log. `import kvdlra` resolves from the
  clone with no install: `pyproject.toml`'s `[tool.pytest.ini_options] pythonpath = ["src",
  "scripts", "."]`. The 1B-dump item stays skipped on the pod as it is in CI (the dump tree is
  gitignored and absent from the clone); `-rA` names it and its reason.
- **Where it runs.** `scripts/pod/boot.sh` executes it after the environment block
  (`===ENV_END===`) and before the `timeout`-bounded entrypoint, from the same clone at the same
  `$SHA`, as `bash -o pipefail -c "$PRE_RUN"` — so the recorded code is pytest's and not the
  `sed`'s — bounded by `===PRE_RUN_BEGIN_kernel_smoke===` and
  `===PRE_RUN_END_kernel_smoke_rc=<rc>===`. It is **not** under the `timeout`: that bar is the
  measurement's (A1.4). A non-zero code does **not** abort the pod — the check axis and the
  latency cells still run and are still recorded, which is deliberate: a kernel that fails this
  gate still produces rows worth reading, and they are simply not citable.
- **How it is recorded.** `scripts/pod.py harvest` writes every `[pre_run] `-prefixed row,
  prefix stripped, to `results/kernel_smoke/pre_run.txt`, and the exit code to
  `manifest.pre_run_rc` (null when the markers are absent). The rows are a kind
  `scripts/pod/watchdog.sh` keeps, so they reach the deduped `<label>.log` the harvest parses;
  that per-poll `sort -u` orders them lexically, so `pre_run.txt` is the set of lines the command
  printed, not their sequence.
- **What refuses.** `scripts/pod.py check` fails the pod with `CHECK FAIL pre_run: rc=<n>` on a
  non-zero code and `CHECK FAIL pre_run: not recorded` when none was harvested (a pod that never
  ran the command, or a log fetch that lost the marker). Since a number of this pod is citable
  only through a manifest that passes `check`, a failed or unrecorded gate blocks §4's reading
  exactly as the pre-registration intended.
- **The launch entry** cites `results/kernel_smoke/pre_run.txt` — its pytest summary line and the
  per-item max/rms prints — instead of pasting the output. The bars themselves are unchanged and
  are the ones A1.2 states (`max|Δ| ≤ 2e-3` **and** `rms(Δ) ≤ 1e-4` at `n_splits == 1`,
  `max|Δ| ≤ 4e-3` across splits and for batch independence, the pre-registered `max|Δ| < 1e-2`
  against reconstruct-then-attend); no number from them enters §4 or §7.
- **`config_hash` moves** because the pod YAML gained a key (and its `doc:`, which is inside the
  hash, gained a sentence naming the hook). No manifest for this pod exists, so nothing that has
  already been recorded is invalidated; `make check` over the committed manifests is unaffected,
  because a pod that declares no `pre_run` does not carry the key into its hash.

**2026-09-21 06:40 EDT, before the launch commit — the hand-off, the recorded version, and a
third refusal.** The command above is superseded, verbatim, by:

```
python -m pytest --version 2>&1 | sed 's/^/[pre_run] /'; python -m pytest -m gpu -rA -p no:cacheprovider tests/test_kernel_triton.py tests/test_kernel_reference.py 2>&1 | sed 's/^/[pre_run] /'
```

— the same gate over the same items, with the tool's own version recorded ahead of it (under
`bash -o pipefail -c` the exit status is the LAST pipeline's, so the recorded code is still the
gate's pytest and not the `--version` call's), and with the `-q` of the previous wording
**dropped**: `pyproject.toml`'s `addopts` already carries one, and a second puts pytest's
terminal reporter at verbosity -2, where it prints no counts row at all
(`_pytest/terminal.py::summary_stats` returns first) — measured on this clone, where the gate
command with both flags produced rc 0 and zero counts rows, and with one produced
`8 skipped, 8 deselected, 14 warnings in 0.07s` on a CUDA-less box and
`8 passed, 3 skipped, 14 warnings in 1.04s` on the items that can run here. `-ra` in `addopts`
is still overridden to `-rA` on the command line, so the per-item rows A1.2 asks for are
unaffected. It reaches the instance **base64-encoded**, as `-e PRE_RUN_B64=`, and
`scripts/pod/boot.sh` decodes it into `$PRE_RUN` before the hook: vast.ai's own `--env` parser
(`vastai/utils.py`, read on the installed CLI) splits that string on spaces outside quotes and
toggles the quote state on every `'`, so the command's inner `'s/^/[pre_run] /'` closes the outer
quote and the value arrives truncated and unterminated, however it was quoted — verified on the
installed parser, which returns the whole base64 token byte-intact and the quoted form cut at
`python -m pytest --version 2>&1 | sed 's/^/[pre_run]`. The decoded command is echoed as the
first `[pre_run] ` row (`$ <command>`), so `results/kernel_smoke/pre_run.txt` records what
actually ran rather than what was meant to. **A third refusal** joins the two above:
`scripts/pod.py check` also fails the pod with `CHECK FAIL pre_run: no passed row` unless
`pre_run.txt` carries a pytest counts row with an `N passed` count, and with `CHECK FAIL
pre_run: failures in the summary` when that row names a failure or an error — because
`conftest` skips every `gpu` item on a card without CUDA and pytest exits 0 on an all-skipped
run, so the exit code alone would have passed this precondition on a run of nothing. The check
reads only a row of pytest's counts shape; `-rA`'s per-item `FAILED`/`ERROR` rows and the echoed
command row are evidence, not verdicts. No reading, no threshold and no bar moves. `config_hash`
moves again, because the pod YAML's `pre_run` value and `doc:` changed; no manifest for this pod
exists, so nothing recorded is invalidated.

### Amendment 1 — correction 3 (2026-09-21 08:20 EDT, the pod running; no reading, threshold or bar moves)

Instance 51903816 has been running from `a691d44` since the launch commit. Nothing here moves a
reading, a threshold or a bar, and no number is re-read; these are five corrections of record, so
that the harvest reads this file as it was meant and the outcome entry can be written against it.

(i) **"below it" reads "above it".** Amendment 1's preamble (§ "Amendment 1", third sentence)
says "Read it before reading §3, §4, §7, §8 and §9 below it". The amendment sits at the END of
this file; those sections are **above** it.

(ii) **A1.2's "met when" carries a third conjunct.** A1.2 states the precondition as "**met**
when ≥ 14 of 16 prompts match AND the worst `max_abs_diff` < 1e-2". The renderer
(`scripts/tables.py::precondition_line`) applies a third: **and `errors == 0`**, i.e. no prompt
recorded as an error row. This is not a new condition and does not change any outcome: §10's R29
already fails the whole pod on any counted error, and A1.2 already says a prompt that raises is a
counted error. The sentence simply did not spell out what the code reads.

(iii) **The record carries the backend that attended, from commit `32b9a0a`.** `backend="auto"`
can degrade from the Triton kernel to the torch reference on a live rank the kernel refuses
(R-L4-22), which a ms/token alone cannot show. From `32b9a0a` the choice is made once per cache
(`kvdlra.kernel.select_backend`, `BugStreamingCache.kernel_backend`) and recorded: a ` backend=`
field on the `[kernel_check prompt=…]` and `[latency …]` lines, a `backend` key on
`KernelCheckRecord` / `LatencyRecord`, and a `backend` column in the §7 table. **The running
pod's records predate all of it** and carry no such field; both regexes make it optional, so
those rows harvest with `backend: null` and every count, threshold and bar is untouched. The
launch entry names the backend from the pre_run evidence (`results/kernel_smoke/pre_run.txt`,
whose Triton-vs-reference items ran the kernel at exactly this pod's shapes) instead.

(iv) **The running pod's kernel arm carries a one-row copy of the factored middle, removed from
commit `32b9a0a`.** At batch 1 the attention glue built `FactoredMiddle.cat([m])` from the single
row, and `torch.cat` of one tensor allocates: ≈ 17 MB of U/C/positions copied per layer per
decode step at 32K, r64 (≈ 0.55 GB of writes and ≈ 160 extra launches per token over 32 layers),
estimated ≈ 1–1.5 ms/token at 32K and ≈ 2–3 ms/token at 64K. It is **conservative** — it burdens
the kernel arm only, ≈ 2 % of the 62.8 ms/token bar, and cannot manufacture a pass — and its
transient is invisible in `kv_peak_gb`. It is therefore reported as part of this pod's kernel-arm
p50, not corrected out of it; `32b9a0a` hands the row's own middle to the kernel instead
(numerically identical), so a later pod's p50 is not comparable to this one at that precision.

(v) **The 1e-2 precondition bar is absolute and is read on a bf16 output.** `kernel_compare`
subtracts an fp32 reference from the kernel's **bf16** output, so the difference includes that
output's own quantization (≤ 2⁻⁹·|out|) on top of the operand roundings, and therefore scales
with |out|. Prompt 0's reading on this pod — worst layer 8.2e-3 — is ≈ one bf16 ulp at |out| ≈ 2,
consistent with the CPU calibration (2.75e-3 at |out| ≤ 0.84). A **correct** kernel can thus
exceed 1e-2 at a layer whose outputs reach ≈ 2.5–3 while every prompt is still token-exact. The
rule is unchanged: if the 16-prompt worst crosses 1e-2 the precondition reads **NOT met**, the
Week-3 gate reads **REFUSED** under §4, and the number is reported as measured — never repaired,
re-run or re-scaled on this pod. Only a **relative** bar (max|Δ| / max|out| per layer, or
ulp-normalised), pre-registered by a further amendment committed **before** any re-run and still
reporting the absolute number beside it, could read such a case differently.

### Amendment 2 (2026-09-21, lane L4, before any re-run's launch commit)

§1–§11 and Amendment 1 are left exactly as written; this file is append-only. This amendment
**does not re-read instance 51903816**, whose precondition is REFUSED under the bar in force at
its launch and whose latency rows stay measured-not-cited. It governs **only pods launched after
its own commit**. It re-states §4's precondition in the units the quantity is actually measured
in, names the record fields that make the re-statement computable, and says how the kernel arm's
spike count is read. It moves no other threshold.

#### A2.1 Why a re-statement is owed at all

§4's precondition has two clauses, both written before any bf16 measurement of either existed.
`kernel_compare` (`src/kvdlra/kernel/attention.py:90-99`) forms `Δ = kernel_output_bf16 −
sdpa_fp32(bf16 store)`, so `Δ` is the sum of the kernel's operand roundings **and** the bf16
rounding of its own output; both scale with `|out|`, which the record does not store. ADR 0001
§5 states the numerics contract relatively ("~2⁻⁸ relative, roughly √2 worse"); the CPU
calibration that fixed `1e-2` was taken at `|out| ≲ 0.9` (`tests/test_kernel_reference.py:95-98`
says so in the fixture). An absolute bar read at an unmeasured scale is a bar on `|out|`. The
`≥ 14/16` clause has the matching defect: two bf16 implementations that are each within a
rounding of the same fp32 answer will disagree at a greedy near-tie with probability ≈ ½,
independently of correctness, and no bf16 measurement of that rate existed before this pod.

#### A2.2 (a) The per-layer criterion, re-stated as relative

For each layer `l` of the first kernel decode step, `kernel_compare` records `d_l = max|Δ_l|`
**and** `m_l = max|ref_l|` (the new field of A2.5). The criterion is

```
    rel = max_l ( d_l / m_l )   ≤   2⁻⁶  =  1.5625e-2  =  4 u ,      u = 2⁻⁸ = 3.90625e-3
```

where `u` is the bf16 unit roundoff (8 significand bits). A layer with `m_l = 0` is excluded and
reported. **The absolute `max_l d_l` is still recorded, printed and reported** beside `rel`, with
the layer index of each; it is a number, no longer a bar.

**Where 4 u comes from — from the calibration, not from taste.** Every term below is a measured
number already in this repository, expressed relative to the `|out|` it was measured at:

| term | measured | relative | in `u` |
|---|---|---|---|
| reference kernel vs reconstruct, random 8B shapes, bf16 operands + bf16 output (`tests/test_kernel_reference.py:110-123`) | 4.66e-3, 5.15e-3 at `\|out\| ≤ 0.9` | 5.18e-3, 5.72e-3 | 1.33, **1.47** |
| the same, on the 1B dump layer (`tests/test_kernel_reference.py:214+`) | 2.75e-3 at `\|out\| ≤ 0.84` | 3.27e-3 | 0.84 |
| Triton split-merge reassociation on top of the reference (`pre_run` cross-split bar, A1.2) | ≤ 4e-3 at the same shapes | ≤ 4.44e-3 | ≤ 1.14 |

The worst plausible sum for a **correct** kernel is `1.47 + 1.14 ≈ 2.6 u`; ADR §5's "roughly √2
worse" is already inside the 1.47 u, which is a measurement of exactly that comparison. **4 u is
2.7× the measured worst single term and ~1.5× the worst plausible sum** — the same 2–10×
discipline A1.2 states for the `pre_run` bars ("each bar sits ≈ 2–10× over the reassociation
residual"), and it is chosen as a power of two so it reads as "four bf16 ulp" rather than as a
fitted constant. It separates one extra rounding from an arithmetic error by two orders of
magnitude: a wrong RoPE angle, a wrong GQA head map or a dropped tile perturbs the output by
`O(1)` relative, i.e. `≳ 250 u`.

**The prediction this bar makes, written before the re-run.** Applied to instance 51903816's
absolute numbers, `rel ≤ 4 u` requires `m_l ≥ 0.94` at prompt 7 layer 26 (1.464e-2), `≥ 0.90` at
prompt 12, `≥ 0.79` at prompt 2, `≥ 0.64` at prompt 10 and `≥ 0.36` at the smallest row
(prompt 4, 5.564e-3). **It is predicted that every one of these passes** — that `max|ref|` at
layers 26–31 of Llama-3.1-8B exceeds 1 — and a re-run in which it does not is the falsifier of
§1 (3), reported as such and not re-argued.

#### A2.3 (b) The token criterion, re-stated for bf16

A greedy mismatch at step `s` **counts against the kernel** only if the reconstruct path's
**top-2 logit gap at `s`** exceeds the pre-registered margin

```
    M  =  2⁻⁵ · max_j L_recon[s, j]   ( = 8 u × the step's own top logit ; ≈ 0.6 for a
                                        Llama-3.1-8B top logit of ≈ 20 )
```

Otherwise the mismatch is a **near-tie divergence**, reported descriptively with its gap, its
step and both logits, and not counted. The count is then read over attributable mismatches:

```
    met  ⟺  ( n_prompts − #{prompts whose first mismatch is attributable} ) ≥ 14
             AND  rel ≤ 2⁻⁶  AND  no error rows
```

**Where M comes from.** The model runs bf16 end to end, so one bf16 rounding of a logit is
`u·|L| = 2⁻⁸·|L|`. The kernel re-rounds the attention output of each of the 32 layers by ≲ 1 u
of that layer's own output; through the residual stream those 32 perturbations are independent
in sign and add in quadrature, `√32 = 5.66`, rounded **up** to the next power of two = 8. Hence
`M = 8 u · max_j L[s, j] = 2⁻⁵ · max_j L[s, j]`. `M` is stated **relative to the step's own
logit scale**, not as an absolute number of logit units, so it transfers across models and
across the tiny model of the calibration below, whose vocabulary is 256
(`src/kvdlra/kernel/prompts.py:30`) and whose logit scale is nothing like Llama's.

**The computable top-logit scale, named precisely (a controller ruling).** A2.3 above writes
`M = 2⁻⁵ · max_j L_recon[s, j]`, but the five fixed fields of A2.5 store no bare `max_j L_recon`.
The shipped reader (`scripts/tables.py` `precondition_line` / `_attributable`) uses the stored
**`kernel_logit_for_ref_argmax`** — the kernel's logit at the token the reconstruct path chose —
as that top-logit scale. It equals `max_j L_recon[s, j]` to within the kernel's own rounding
(≈ 2⁻⁸ relative) at exactly the near-tie steps this threshold separates: at a near-tie the
reconstruct path's chosen token is its own top token, and the kernel reproduces that logit to a
bf16 rounding. And it fails **safe**: a large reversal — an attributable mismatch — depresses the
kernel's logit at the reconstruct's token *below* `max_j L_recon`, so `M` shrinks and attribution
becomes *more* likely, never less — the one direction that cannot manufacture a pass. No threshold
and no rule moves; only the description is made precise about the quantity the record actually
carries, and no sixth field is added.

**Only the first mismatch is read, and that is a property of the comparison, not a concession**:
after step `s` the two paths have different contexts, so steps `> s` are not comparable
measurements of the same quantity. The record therefore stores the first mismatch's gap, and the
rule reads it.

**The $0 calibration that fixes `M` before the re-run, and can only tighten it.** Before the
launch commit, `tests/test_kernel_path.py` gains the bf16 twin of its fp32 check (Task 11): the
tiny model cast to bf16, the same 16 committed prompts × 16 greedy steps, kernel arm vs
reconstruct twin, recording per step `ρ_s = max_j |L_kernel[s,j] − L_recon[s,j]| / (u · max_j
|L_recon[s,j]|)` over the top-8 tokens. Let `ρ̂` be the worst `ρ_s` observed. Because the tiny
model has `L_tiny` layers and Llama has 32, the pre-registered extrapolation is `κ = ρ̂ ·
√(32 / L_tiny)`, rounded **up** to the next power of two.
- If `κ ≤ 8`, `M` stands at `2⁻⁵ · max_j L[s,j]` as written above.
- If `κ > 8`, this amendment is revised to `M = κ · u · max_j L[s,j]` **before** the launch
  commit, with the measured `ρ̂`, `L_tiny` and `κ` printed in the revision.
Both branches are fixed here, in advance, and the evidence is a committed CPU test — the
calibration never sees an 8B number.

**Measured (Task 11, `tests/test_kernel_path.py::test_bf16_calibration_of_the_near_tie_margin`).**
ρ̂ = 1.9845 and L_tiny = 2, so κ_raw = ρ̂ · √(32 / L_tiny) = 1.9845 · √(32 / 2) = 1.9845 · 4 =
7.938, rounded **up** to the next power of two = **κ = 8**. Since **κ ≤ 8**, the first branch
holds: `M = 2⁻⁵ · max_j L[s, j]` stands exactly as written, and the `κ > 8` revision branch is
*not* triggered. κ = 8 is boundary-passing — κ_raw 7.938 sits just under 8, and the calibration is
deterministic under the uv.lock pins — so κ must be re-read from `tests/test_kernel_path.py`'s
calibration before the launch commit if the CPU numeric stack (torch / BLAS) changes.

#### A2.4 (c) The kernel arm's spikes

**No threshold moves.** The kernel arm's `spikes` and `ms_max` are read exactly as §4 reads them
(`prereg/kernel_smoke.md:237-240`): `> 8` of 56 refuses that arm's p50 as a steady-state number,
and the 5 spikes / 1,992 ms max of instance 51903816 are **below that refusal**, so its p50
stands as a steady-state number under the original rule — while its *cause* is unresolved and
§5's prediction of "0 on the kernel" is, as measured, wrong. That is reported as a missed
prediction, not repaired.

**The recompile hypothesis is already refuted by reading, and is recorded so it is not re-opened.**
`_tiles_kernel` (`src/kvdlra/kernel/triton_kernel.py:39-41`) declares **nothing length-dependent
as `tl.constexpr`**: `n_dense`, `n_mid`, `tiles_per_split` and `n_splits` are runtime scalars;
the `tl.constexpr` set is `H_KV, G, D, HALF, R, M, GP`, all fixed for a model, a rank and the
64-token tile. **There is therefore no per-absorb-event recompile**, and "every 16 tokens the
tile count moves, so Triton rebuilds the kernel" is not the explanation. What remains is
bounded: Triton specializes integer runtime arguments on divisibility-by-16 and equality-to-1,
so a handful of distinct variants (the parity of `n_dense` as the recent ring fills, one
`tiles_per_split` step, `n_splits` fixed at 16 for `b=1, H_kv=8` by `n_splits_for`) compile on
**first touch** and never again.

**The check on the re-run is the cheapest one that separates the two remaining candidates, and
it perturbs nothing measured**: `kvdlra.eval.latency.run_latency` already keeps every step's time
(`src/kvdlra/eval/latency.py:106-108`), so it prints one **log-only companion line** per cell,
`[latency spikes ctx=<T> arm=<key> steps=<i,j,k,…>]`, giving the indices of the spiking steps
within the 56-step steady window. No record field, no regex, no renderer change, no change to
the arm under measurement. The two candidates have disjoint signatures and the line decides
between them with no further work:
- **first-touch JIT compilation** → indices cluster in the first few steps of the window and
  never recur;
- **the block-16 absorb** → indices recur on a 16-step lattice across the whole window, which
  would refute ADR 0001 §5's "block-16 absorbs invalidate nothing in the kernel path"
  (`docs/adr/0001-factored-attention-kernel.md:213-215`) and is a finding about the design, not
  about Triton.

Either way it is a **performance** observation. `TRITON_PRINT_AUTOTUNING` is not used: there is
no `@triton.autotune` on this kernel. Pinning the four scalars via `do_not_specialize`, or
lengthening the warm-up, is a **fix**, pre-registered nowhere here, and belongs to a later commit
and a later pod — after the line above says which cause it would fix. `num_warps=4` stays as
A1.4 leaves it (`prereg/kernel_smoke.md:595-598`): untuned, visible only in the kernel arm's
ms/token, and no threshold adjusted for it.

#### A2.5 The record fields this amendment needs

The criteria of A2.2 and A2.3 are not computable from `KernelCheckRecord` as it stands
(`src/kvdlra/eval/records.py:213-234`). Five fields are added, appended **before** the `error=`
tail of the `[kernel_check prompt=…]` line exactly as `backend=` was (L4.fw1,
`src/kvdlra/eval/kernel_check.py:111-114`), so every archived row still parses and every field
that came first keeps its place:

| field | meaning |
|---|---|
| `rel_max_diff` | `max_l ( max\|Δ_l\| / max\|ref_l\| )` — the quantity A2.2 reads |
| `rel_worst_layer` | the layer achieving it |
| `ref_max` | `max\|ref\|` at `rel_worst_layer` — the denominator, so the ratio is auditable |
| `gap_at_mismatch` | the reconstruct path's top-1 minus top-2 logit at the first mismatching step; `-` when the prompt matched |
| `kernel_logit_for_ref_argmax` | the kernel's logit for the token the reconstruct path chose at that step, beside the kernel's own top logit — the reversal's magnitude, and the evidence that a small gap really was a near-tie |

The per-layer `max|ref|` values join the existing log-only `[kernel_check layers prompt=0 …]`
breakdown (`src/kvdlra/eval/kernel_check.py:80-82`) as a `refs=` list beside `diffs=`; they are
not records. `scripts/tables.py` `precondition_line` (`99ea29f:scripts/tables.py:1181-1197`)
reads **both** bars and prints both numbers with the attributable and non-attributable mismatch
counts named separately; `DIFF_MAX` stays in the file as the reported absolute number.
Task 11 (§3 below) is the code, and it is CPU-tested before the launch commit.

#### A2.6 (d) What this amendment does not change

The Week-3 gate's two conditions and their thresholds — `kv_peak_gb(kernel, 32K, b) <
kv_peak_gb(full, 32K, b)` and `ms_per_token_p50(reconstruct, 32K, b) / ms_per_token_p50(kernel,
32K, b) ≥ 3.0`, pass = both at batch 1 (§4 (1)-(3)); the 10 % marginal-memory rule and the 10 %
archived-agreement rule (§2 (a), §4 (1)); the error refusal, the `spikes > 8` refusal and the
"never `--`" rule (§4); §6 (no statistic, no correction); the arm list (`full`,
`isvd_r64_h256_seed`, `isvd_r64_h256_seed_kernel`) and their configs; cell list A (batch 1,
16K / 32K / 64K, 64 decode steps of which 56 are timed) and cell list B's conditions; the
`pre_run` gate and **every bar it applies** — `max|Δ| ≤ 2e-3 ∧ rms(Δ) ≤ 1e-4` at `n_splits == 1`,
`≤ 4e-3` across splits and for batch independence, `< 1e-2` against reconstruct-then-attend on
the random 8B shapes and the 1B dump layer (A1.2, A1's correction) — which are comparisons of the
Triton kernel against the torch **reference**, at a fixed calibrated `|out|`, and are unaffected
by anything above; §7's KIVI-2 deviation; §11. The 16 prompts, their source and their pins are
unchanged. Amendment 1 A1.3's record and renderer stand as written, extended only by A2.5.

#### A2.7 (e) Budget and sequencing

- **Budget unchanged**: the point estimate stays `1.8 h + 0.45 h = 2.25 h` and the bar stays
  `gpu_budget_h: 5.0` (A1.4, `prereg/kernel_smoke.md:577-584`); the five new fields add no GPU
  work — `max|ref|` is one `.abs().max()` on a tensor `kernel_compare` already materializes, and
  the logits are already computed by the greedy loop. At A1.4's $0.45–0.74/h that is
  **≈ $1.0–1.7 expected, ≈ $2.3–3.7 at the bar**. The check axis's own wall clock, still the
  first number to read against A1.4's 0.45 h allowance, now has instance 51903816's actual to
  read against as well.
- **Order**: this amendment's commit, then Task 11's commit, then the pod YAML's commit, then the
  launch commit. `scripts/pod.py launch` machine-enforces only that the prereg's **first** commit
  is a strict ancestor of the launch SHA (`prereg_error`, `99ea29f:scripts/pod.py:373-388`), so
  an appended amendment's ordering is **not** machine-checked: the launch entry in
  `docs/plan/DECISIONS.md` names this amendment's SHA explicitly and
  `git merge-base --is-ancestor <amendment SHA> <launch SHA>` is the evidence, pasted.
- **The re-run is a new pod label**, `kernel_smoke2` (`configs/pods/kernel_smoke2.yaml`, the
  `kernel_smoke` YAML byte-for-byte with `name:` changed and `prereg: prereg/kernel_smoke.md`
  kept), so it writes `results/kernel_smoke2/` and cannot overwrite instance 51903816's harvest
  (`scripts/pod.py` §4: one directory per pod, `results/<pod>/`). Instance 51903816's records are
  committed, harvested and cited as **REFUSED under the original bar**; nothing about them is
  revised, deleted or re-read.
- **Citability** is unchanged (CLAUDE.md): a number of the re-run is citable only once
  `scripts/pod.py check results/kernel_smoke2` passes — including `_kernel_check_fails`'s 16 rows
  per kernel arm (`scripts/pod.py:877-891`) and the three `pre_run` refusals — and
  `make kernel_smoke` regenerates its table from the committed records.
