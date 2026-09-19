# Pre-registration — `ss2_families_mistral` + `ss2_families_qwen` + `ss2_families_llama`

**STATUS: awaiting owner go (DECISIONS; after D-005 closes).** Written before any of the three
pods is launched. Launch is a separate, later commit that spends money and is recorded in
`docs/plan/DECISIONS.md`: `scripts/pod.py launch` refuses unless this file's first commit is a
*strict* ancestor of the launch commit, so the commit that adds this file launches nothing. The
launch further waits on the filler-realism reading (D-005) — see *The filler condition* below,
which may amend the task list before anything runs.

Three pods, one pre-registration: `PodCfg` carries a single `model`, so the Mistral, Qwen and
Llama cells of one experiment are three configs (ruling PR-L2-15). Lane L2 item 4; gate G2 line 4.

---

## The filler condition (ruling PR-L2-16) — read before §1

These pods run the **cycled** in-house filler (`ruler_inhouse_16k` / `ruler_inhouse_32k`,
`filler: cycle`) **by design**: it is the filler behind every archived cell in §2, and running
it is what lets this pod's records pair with those cells on `(seed, trial)` — the archived
`w19-a1-*` KIVI rows, the `w18-g1-*` r64 rows and the `w19-sysfix-llama` single-shot row. The
same filler is, as of this writing, the subject of the filler-realism diagnostic
(`prereg/filler_realism.md`, DECISIONS D-005, running on instances 51553610 / 51559661): its
decision rule retires the cycled generator from every headline claim if the r64 configuration or
its q4 cell drops by more than 0.25 on any task on real-text filler. The unharvested rows of that
pod visible in the watchdog's raw capture at the time of writing (`results/filler_realism/
filler_realism-51553610.raw`, 2026-09-19 06:47) read `bugSseed-r64-h256` **0.25 / 0.08 / 0.00**
on single / multikey / multivalue on WikiText filler against the archived cycled **1.00 / 1.00 /
1.00**, with the cycle-control pod replicating the archived row at 1.00 / 1.00 on the two cells it
had reached. That is not a harvested number and is not cited as one; it is the reason this
section is load-bearing.

**Pre-registered, before D-005 closes:**

1. **Launch waits for D-005.** None of the three pods is launched until D-005 is closed in
   `DECISIONS.md` with the harvested evidence path.
2. **If D-005 retains the cycled generator** (the r64 configuration and the q4 cell within 0.25 of
   their archived rows on every task, and the cycle control within 0.25 of its archived row), the
   pods run exactly as §3 describes, and every within-pod and cross-pod comparison in §4–§7 is
   read as written.
3. **If D-005 retires the cycled generator from headline claims**, this pre-registration is
   **amended before launch** — as a dated *Amendment 1* appended below, §1–§11 left untouched, in
   the pattern of `prereg/hygiene_table4.md` — to run **`ruler_v2_16k` / `ruler_v2_32k`**
   (generator v2, `kvdlra.eval.gen`: real-text haystacks from four sources, official RULER
   templates, the balanced 2 × 3 × 2 design of 12 trials per seed) in place of
   `ruler_inhouse_16k` / `ruler_inhouse_32k`, on the same three pods, the same four arms in the
   same order, the same n = 12 per cell. Under that amendment:
   - **the cross-pod pairing with the archive is dropped** — no record of these pods is paired
     with, or read against, any `results/paper-v1/` row (§2's archived cells then serve only as
     the *predictions'* basis, not as a comparison base); the harness-consistency replication of
     §7(c), the chunked-vs-single-shot recovery of §7(d) and the archived Llama 16K contrast of
     §6 are all withdrawn;
   - **only the within-pod pairing remains**: the primary contrast of §4, the Holm families of §6
     (the 32K family of 12 and the 16K family of 8, both entirely within-pod) and the
     within-pod secondaries §7(a), (b), (e) are unchanged in form and are read exactly as written,
     on the v2 records — every arm in a pod sees byte-identical prompts per `(task, ctx, seed,
     trial)` (the v2 pairing invariant, `prompt_sha256` in every record), which is all the
     paired test needs;
   - `niah_multiquery`, which v2 adds as a fifth task, is reported descriptively and does **not**
     enter either Holm family (the families are fixed at 4 tasks here; a fifth member would be
     admitted only by the amendment, which must say so before launch);
   - the predictions of §5 are re-stated in the amendment on the v2 semantics (they are written
     here on the cycled filler and are not expected to transfer — the diagnostic's own reading is
     that they do not);
   - the budget of §9 is re-derived there (v2 adds haystack materialisation and per-trial
     prompt construction that the cycled path does not pay; the arm rates stand).
4. Whichever branch applies, the outcome of D-005 and the branch taken are named in the launch
   entry in `DECISIONS.md`, with this file's first-commit SHA and (if any) the amendment's.

---

## 1. Purpose

Lane L2 item 4 ("ss2 across families"). The v1 paper's "ss2" observation rests on **one cell**:
on Llama-3.1-8B at 16K, the r64 configuration's multi-value edge over 2-bit KIVI — 1.00 against
0.42 under the *streaming* KIVI arm (chunked 4096 prefill, later chunks attending to
already-quantized context) — shrinks to 1.00 against **0.83** when the same 2-bit arm prefills
in one shot (`results/paper-v1/w19-sysfix-llama`), a 2-of-12 discordance the exact paired
McNemar cannot separate (p = 0.50). The v1 reading was that the multi-value edge is at least
partly **protocol-bound**: an artefact of how the streaming arm was prefilled, not of 2-bit
storage. It was measured on one family at one context, against a single-shot arm that still
used the streaming mixin's grouping (G = 64, the fp16 residual flushed at the end of prefill).

L2 Task 3 shipped **KIVI at its published operating point** (Liu et al. 2024): per-channel keys,
per-token values, **G = 32, R = 128, full-precision single-shot prefill** with the quantized
store built post hoc (`configs/arms/kivi2_faithful.yaml`, `kivi4_faithful.yaml`;
`kvdlra.quant.kivi`). That arm, not the streaming mixin, is the baseline a reviewer means by
"KIVI-2".

**The question these pods answer:** on Mistral-7B-v0.3 and Qwen2.5-7B at 16K and 32K, and on
Llama-3.1-8B at 32K, does the r64 configuration remain **separated** from 2-bit KIVI on the
in-house multi-value task once KIVI runs its own protocol — or does the faithful arm close the
gap everywhere, as the one Llama 16K cell suggests?

- **Separated after Holm in a cell** → the multi-value claim survives *in that cell* and the
  paper may state it there, against the faithful arm, with the corrected p-value.
- **Not separated in a cell** → the multi-value claim does **not** survive there; the paper's
  §4 says the edge is against the streaming arm only, and says so per family.

Reading, fixed now and verbatim from the lane: **"the multi-value claim survives only where the
r64 configuration vs single-shot KIVI-2 is separated after Holm over 3 families × 4 tasks."**

Nothing else is decided here. In particular these pods produce no memory or throughput claim,
do not touch the exact tier or the tracker, and say nothing about official RULER or generator v2
(unless the filler condition's branch 3 applies, in which case they say what the amendment says).

## 2. Measured baseline — the archived cells, and what they predict

Every number in this section was computed from the committed per-trial records by the snippet
below, run at the commit that adds this file — none is copied from a table or from memory.
Model ids in the records: `unsloth/Meta-Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`,
`Qwen/Qwen2.5-7B-Instruct` — the ids the three pod YAMLs name.

```python
import json, collections, pathlib
for pod in ["w19-a1-llama", "w19-a1-mistral", "w19-a1-qwen", "w19-sysfix-llama",
            "w18-g1-llama", "w18-g1-mistral", "w18-g1-qwen"]:
    rows = [json.loads(l) for l in pathlib.Path(f"results/paper-v1/{pod}/trials.jsonl").read_text().splitlines()]
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["arm"], r["ctx"], r["task"])].append(r)
    for (arm, ctx, task), rs in sorted(cells.items()):
        print(pod, arm, ctx, task, f"{sum(r['hit'] for r in rs)}/{len(rs)}",
              "errors", sum(1 for r in rs if r["error"]))
```

Every cell below holds exactly 12 records, keys `(seed, trial)` = {0, 1} × {0..5}, zero errors
(the 8-bit HQQ control in the `w19-a1-*` pods holds 4 and is not used here).

**The r64 configuration** — `bugSseed-r64-h256` (= `configs/arms/isvd_r64_h256_seed.yaml`),
`results/paper-v1/w18-g1-{llama,mistral,qwen}/trials.jsonl`, hits/12 on single / multikey /
multivalue / vt:

| family | 16K | 32K |
| --- | --- | --- |
| Llama | 12 / 12 / 12 / 7 (0.58) | 12 / 12 / 12 / 11 (0.92) |
| Mistral | 12 / 12 / 12 / 6 (0.50) | 12 / 12 / **10 (0.83)** / 5 (0.42) |
| Qwen | 12 / 12 / 12 / 12 | 12 / 12 / 12 / 12 |

**2-bit KIVI, streaming arm** — `quant-2bit-kivi` (= `kivi2_streaming`: G = 64, R = 128,
chunked 4096 prefill, residual flushed), `results/paper-v1/w19-a1-{llama,mistral,qwen}/trials.jsonl`:

| family | 16K | 32K |
| --- | --- | --- |
| Llama | 12 / 8 (0.67) / **5 (0.42)** / 8 (0.67) | 12 / 10 (0.83) / 11 (0.92) / 11 (0.92) |
| Mistral | 11 (0.92) / 7 (0.58) / **6 (0.50)** / 4 (0.33) | 10 (0.83) / 3 (0.25) / **1 (0.08)** / 0 (0.00) |
| Qwen | 12 / 10 (0.83) / **4 (0.33)** / 11 (0.92) | 11 (0.92) / 7 (0.58) / **2 (0.17)** / 3 (0.25) |

**4-bit KIVI, streaming arm** — `quant-4bit-kivi` (= `kivi4_streaming`), same files:

| family | 16K | 32K |
| --- | --- | --- |
| Llama | 12 / 12 / 12 / 12 | 12 / 12 / 12 / 12 |
| Mistral | 12 / 12 / 12 / 3 (0.25) | 12 / 12 / 12 / 3 (0.25) |
| Qwen | 12 / 12 / 11 (0.92) / 12 | 12 / 12 / 12 / 12 |

**2-bit KIVI, single-shot (the v1 ss2 cell)** — `quant-2bit-kivi` in
`results/paper-v1/w19-sysfix-llama/trials.jsonl` (= `kivi2_singleshot`: G = 64, R = 128, chunk 0,
residual flushed), Llama 16K only: **12 / 12 / 10 (0.83) / 8 (0.67)**.

**What the archive already says, paired.** `scripts/tables.py`'s `paired` (the exact paired
McNemar of `kvdlra.eval.stats.mcnemar_exact` over the 12 shared `(seed, trial)` keys, the r64
arm as *a*), on `niah_multivalue`, r64 vs the *streaming* 2-bit arm — `a_favored / b_favored`, p:

| | Llama | Mistral | Qwen |
| --- | --- | --- | --- |
| 16K | 7 / 0, p = 0.0156 | 6 / 0, p = 0.0312 | 8 / 0, p = 0.0078 |
| 32K | 1 / 0, p = 1.0 | 9 / 0, p = 0.0039 | 10 / 0, p = 0.0020 |

Holm over the 12 members (3 families × 4 tasks) at each context, on those archived cells: at
**32K** the separated members are Mistral multikey (adj. 0.043), Mistral multivalue (0.043), Qwen
multivalue (0.023) and Qwen vt (0.043); at **16K nothing separates** — the smallest adjusted
p-value is Qwen multivalue at 0.094. So even against the streaming arm, the archive's multi-value
edge is Holm-separated at n = 12 only at 32K on Mistral and Qwen; at 16K it never was.

The v1 ss2 cell itself, paired the same way: r64 vs single-shot 2-bit on Llama 16K — single
0 / 0, multikey 0 / 0, **multivalue 2 / 0 (p = 0.50)**, vt 2 / 3 (p = 1.0). And single-shot vs
streaming 2-bit on Llama 16K: multikey 4 / 0 (p = 0.125), multivalue 6 / 1 (p = 0.125), vt 3 / 3
— the "recovery" the v1 reading rests on is itself a 6-of-12 swing that n = 12 cannot separate.

**Consequence for this run.** The faithful arm has finer groups (32 against 64), keeps its
trailing `T mod 128` prompt tokens in fp16 into decode (folded only once the residual reaches
128) where the streaming mixin flushes them before the first decode step, and it never attends
to a dequantized token during prefill. It is therefore expected to score **at least
as well as** the G = 64 single-shot arm on every task, and the Llama 16K precedent puts single-shot
2-bit at 0.83 on multi-value against 0.42 chunked. Against an r64 arm that is at 12/12 on
multi-value in five of the six archived (family, ctx) cells — and at 10/12 on Mistral 32K —
separation after Holm needs the faithful arm to lose **at least nine of twelve pairs with none
won** (§6). The prediction written in §5 is that this does not happen in any cell, with Qwen 32K
named as the one cell where it plausibly could.

## 3. Arms, tasks, n

Three pods, one arm list, in this order (ruling PR-L2-15):

| # | arm config | `kind` | prefill | grouping | record key (`legacy_name` or config name) |
| --- | --- | --- | --- | --- | --- |
| 1 | `isvd_r64_h256_seed` | `bug` | chunked 4096 | rank 64 gist, 256-token surprise tier, warm-up seed | `bugSseed-r64-h256` |
| 2 | `kivi2_faithful` | `quant_faithful` | **fp16 single-shot, quantized post hoc** | G = 32, R = 128 kept fp16 | `kivi2_faithful` |
| 3 | `kivi2_singleshot` | `quant` | single-shot into the streaming mixin (`chunk: 0`) | G = 64, R = 128 flushed | `quant-2bit-kivi#chunk0` |
| 4 | `kivi4_faithful` | `quant_faithful` | fp16 single-shot, quantized post hoc | 4-bit, G = 32, R = 128 kept | `kivi4_faithful` |

**The r64 arm goes first** because it is the paired reference every contrast needs: a pod that
dies after its second arm still lands the primary contrast; one that dies after its first lands
the harness-consistency replication of §7(c). **No `full` arm**: every contrast here is paired on
identical prompts within the pod, and an uncompressed ceiling is not a member of any of them
(the archived `w19-a1-*` pods carry none either). Arms 2–4 are `chunkable: false`, so the runner
prefills them in one shot whatever the task's `chunk` says (`chunk = task.chunk if
arm.chunkable else 0`); arm 1 takes the task's `chunk: 4096`, exactly as its archived rows did.

`isvd_r64_h256_seed.yaml` is **not edited** (the parity golden and the w18 configs pin it), and
no `_diag4096` variant of it is created: §8 works the log volume through and finds the default
`diag_every` (64) inside the watchdog's margin. The arm runs under the shipped default
orthonormality guard (repair above `‖UᵀU − I‖_F` = 1e-3, abort above 1e-1), which its archived
rows — produced before L1 — did not have; at rank 64 (n/16 on Llama and Mistral, n/8 on Qwen)
the trace is not expected to reach the repair threshold, and `fixed_k` / `fixed_v` in
`diag.jsonl` is the read (§7(f)).

| pod | model | tasks | cells |
| --- | --- | --- | --- |
| `ss2_families_mistral` | `mistralai/Mistral-7B-Instruct-v0.3` | `ruler_inhouse_16k`, `ruler_inhouse_32k` | 4 arms × 4 tasks × 2 ctx = 32 |
| `ss2_families_qwen` | `Qwen/Qwen2.5-7B-Instruct` | `ruler_inhouse_16k`, `ruler_inhouse_32k` | 32 |
| `ss2_families_llama` | `unsloth/Meta-Llama-3.1-8B-Instruct` | `ruler_inhouse_32k` | 16 |

Llama runs **32K only**: its 16K single-shot cell is the archived v1 observation of §1
(`w19-sysfix-llama`) and is not re-bought. All three: `dtype: bfloat16`, the `-devel` image
(quanto JIT-builds its kernel).

**Tasks and n.** Both tasks are the shipped in-house configs, unchanged: `generator: inhouse`,
the four sub-tasks `niah_single`, `niah_multikey`, `niah_multivalue`, `vt` in one call per arm
(so `max_new` resolves to 40 as it did for every archived row), `n_trials: 6` × `seeds: [0, 1]` =
**12 records per (arm, task, ctx)**, `filler: cycle`, `chunk: 4096`, depths drawn by the
generator. `pod.py check` requires every one of the 32 / 32 / 16 cells to hold exactly 12
records, errors counted (a trial that raises is a row with `hit = 0` and the exception in
`error`; one error row fails the pod's gate, ruling R29 — the cell is then excluded from the
decision rule and the pod is citable only by an L6 per-case ruling naming the cell and the
error, the rule `prereg/filler_realism.md` A1.2 pre-registered).

**Pairing.** Within a pod, all four arms see the same prompt for a given `(task, ctx, seed,
trial)`: the cycled haystack and the needle are deterministic in `(seed, trial)`, and the
runner's `[trial]` line and record carry `prompt_sha256`, so the pairing is *verified*, not
assumed (§7(e)). Across pods — this pod's rows against the archived `w19-a1-*` / `w18-g1-*` /
`w19-sysfix-llama` rows — the pairing rests on the same determinism but the v1 records carry
`prompt_sha256: null`, so it cannot be verified (`scripts/tables.py`'s `CROSS_POD` note); every
cross-pod comparison below is descriptive for that reason as well as by design.

**Single-shot prefill at 32K is a stated risk.** `_prefill_faithful` runs one forward over the
whole 32K prompt into a `DynamicCache` (`logits_to_keep=1`, sdpa), then quantizes layer by layer;
`kivi2_singleshot` runs one forward into the streaming mixin's `QuantizedCache`. Neither has run
at 32K in this repository (the v1 single-shot cell was 16K; the archived 32K KIVI rows are
chunked). On an A100 40 GB the expected peak is weights + a 32K bf16 cache (≈ 4 GB on Llama /
Mistral, ≈ 1.8 GB on Qwen) + one layer's quantization temporaries — well inside the card — but
an OOM in either arm is recorded as an `error` row and handled as above; it is not retried with
a chunked prefill, which would make the arm the streaming one.

## 4. Primary contrast and decision rule

**Primary contrast, written exactly:** per family and context, on **`niah_multivalue`**,
**`isvd_r64_h256_seed` vs `kivi2_faithful`**, the **exact paired McNemar** over the 12 shared
`(seed, trial)` keys — `kvdlra.eval.stats.mcnemar_exact`, as `scripts/tables.py paired` computes
it (the two cells' Bernoulli outcomes keyed by `(seed, trial)`, the two-sided exact binomial on
the discordant pairs) — with the r64 arm as *a*. Five cells: Mistral 16K, Mistral 32K, Qwen 16K,
Qwen 32K, Llama 32K. Each p-value is a member of its context's Holm family (§6); the reading is
on the **Holm-adjusted** p-value.

**Decision rule, fixed now.** For each of the five cells:

1. **Separated** — the adjusted p-value is < 0.05 **and** the discordance is in the r64 arm's
   favour (`a_favored > b_favored`) → *the multi-value claim survives in that cell*: the paper
   states the r64 configuration's multi-value edge over KIVI-2 at its published protocol for
   that family and context, with the adjusted p.
2. **Not separated** — the adjusted p-value is ≥ 0.05, or the discordance favours the faithful
   arm → *the claim does not survive in that cell*: the paper's §4 states that the multi-value
   edge in that cell is against the streaming arm only, and the v1 ss2 reading (protocol-bound)
   is confirmed for that family and context.
3. **The reverse** — adjusted p < 0.05 with the discordance in the faithful arm's favour → a
   finding, not a failure: reported as a KIVI-2 win on multi-value in that cell, and the
   paper's multi-value sentence for that family is withdrawn rather than softened.

The reading is **per cell**; it is not pooled. A claim that "survives" survives where it
survives, and the paper's §4 lists the cells. The Llama 16K cell is not re-run and its v1 reading
— not separated (2 / 0, p = 0.50, §2) — stands as the archived observation.

The statistic, once the records are harvested (`scripts/tables.py`'s `paired()` reads the
archive under `results/paper-v1/`; pointing it at a live `results/<pod>/` directory is Task 8's
table work and is not pre-registered here — the statistic is the function, applied to the pod's
`trials.jsonl`):

```python
from kvdlra.eval.stats import mcnemar_exact, holm
# a, b: {(seed, trial): hit} for bugSseed-r64-h256 / kivi2_faithful, one pod, one task, one ctx
m = mcnemar_exact(a, b)          # n_paired, a_favored, b_favored, p_value
adjusted = holm([...])           # the 12 (32K) or 8 (16K) raw p-values of §6, together
```

## 5. Prediction per arm and family, written before the run

Accuracy = hits/12 on the cycled filler. "Separated" refers to the §4/§6 reading.

| arm | family / ctx | prediction |
| --- | --- | --- |
| `isvd_r64_h256_seed` | all | **replicates its archived row within 0.25 on every task** (§2 r64 table; the cycle-control pod of `prereg/filler_realism.md` A1.3 is at 1.00 / 1.00 on the cells it has reached). Multi-value 12/12 everywhere except Mistral 32K (10/12 archived). A cell more than 0.25 from its archived row is harness drift (§7(c)). |
| `kivi2_faithful` | Mistral 16K | multi-value **≥ 0.75** (streaming 0.50; the Llama precedent adds +0.42 for single-shot alone, and G = 32 with the residual kept adds, not subtracts); multikey ≥ 0.83 (from 0.58); vt ≥ 0.33. → **not separated**: ≤ 3 discordant pairs in the r64 arm's favour. |
| `kivi2_faithful` | Qwen 16K | multi-value **≥ 0.67** (streaming 0.33); multikey ≥ 0.92; vt ≥ 0.92. → **not separated** (≤ 4 / 0). |
| `kivi2_faithful` | Llama 32K | multi-value ≥ 0.92 (streaming already 0.92); nothing to separate. → **not separated**. |
| `kivi2_faithful` | Mistral 32K | multi-value **0.50–0.75** (streaming 0.08 — the weakest archived cell; single-shot recovers most of it); multikey ≥ 0.58 (from 0.25); vt ≥ 0.17 (from 0.00). → **not separated**: the r64 arm is itself at 10/12 here, so a (≥ 9, 0) pattern would need the faithful arm at ≤ 1/12 with both r64 misses coinciding with faithful misses. |
| `kivi2_faithful` | Qwen 32K | multi-value **0.50–0.75** (streaming 0.17); multikey ≥ 0.75; vt ≥ 0.50 (from 0.25). → **not separated** — but this is the **one live cell**: the r64 arm is 12/12 on all four tasks, streaming 2-bit is at 0.17 / 0.25 on multivalue / vt, and if the faithful arm recovers to ≤ 0.25 on either task the (≥ 9, 0) pattern §6 needs is reached. A separation here on multi-value is the one outcome that keeps a multi-value sentence in the paper against the faithful arm. |
| `kivi2_singleshot` | all | between the streaming row and the faithful arm on every task; **never above the faithful arm by more than 0.25** (finer groups and a kept residual cannot make the faithful arm worse). Its Llama 16K archived row (12 / 12 / 10 / 8) is the shape expected on Mistral/Qwen 16K. A cell where it beats the faithful arm by ≥ 3 pairs is a finding about the residual mechanics, §7(a). |
| `kivi4_faithful` | all | **12/12 on single / multikey / multivalue** in every cell (the streaming 4-bit arm already is, bar Qwen 16K multivalue 11/12); vt 12/12 on Llama and Qwen; **Mistral vt stays low (≈ 0.25)** — the archived 4-bit and 8-bit arms both sit at 0.25 there and the r64 arm at 0.50 / 0.42, so Mistral vt is hard for every method under this template. Descriptive. |

**Summary prediction:** the multi-value claim **does not survive against the faithful arm in any
of the five cells**; the v1 ss2 reading generalizes. The pre-registered alternative is a
separation on Qwen 32K. A separation anywhere else would contradict the Llama 16K precedent and
would be reported as such, with its discordant pairs listed.

**Abort and error are possible outcomes, not accidents.** An `OrthonormalityError` on the r64
arm (the guard could not restore the basis after a repair), an OOM in a single-shot prefill at
32K (§3), or any other exception, is an `error` row: counted in n, fails `pod.py check`, excludes
the cell from §4, and is reported with the exception. No arm is re-run with the guard off or with
a chunked prefill.

## 6. Family size and correction

**Two Holm families, one per context, fixed now** (`kvdlra.eval.stats.holm`, α = 0.05, applied by
hand over the raw `mcnemar_exact` p-values; `scripts/tables.py paired` applies no correction of
its own):

- **The 32K family: 3 families × 4 tasks = 12 members.** `isvd_r64_h256_seed` vs
  `kivi2_faithful` on each of `niah_single`, `niah_multikey`, `niah_multivalue`, `vt`, on Mistral
  32K, Qwen 32K and Llama 32K — every member within its pod. This is the family the lane's
  reading names ("Holm over 3 families × 4 tasks"), and the only context at which all three
  families run the faithful arm.
- **The 16K family: 2 families × 4 tasks = 8 members.** The same contrast on the four tasks on
  Mistral 16K and Qwen 16K, within-pod. The third family's 16K cell is the archived v1
  observation — r64 (`w18-g1-llama`) vs single-shot G = 64 2-bit (`w19-sysfix-llama`), cross-pod,
  a different arm from `kivi2_faithful` — and is the observation under replication, not a test
  this pod runs; it is reported beside the family (§2: 2 / 0, p = 0.50) and does not enter it.

The three primary multi-value p-values at 32K are three of the twelve members of the 32K family;
the two at 16K are two of the eight members of the 16K family; the twelve (and eight) raw
p-values are corrected together, and every member's adjusted p-value is reported, not only the multi-value ones — a
separation on `vt` or `niah_multikey` in a family is a finding about that task and is reported as
such, but only the multi-value members feed the decision rule of §4.

**The 12-key resolution, stated once.** One flipped pair moves an accuracy by 1/12 = 0.083. The
exact two-sided McNemar p-value at *a* discordant pairs in the r64 arm's favour and *b* against
is `2 · P(Bin(a + b, ½) ≤ min(a, b))`: (9, 0) gives 0.0039, (8, 0) 0.0078, (12, 0) 0.00049, (9, 1)
0.0215, (11, 1) 0.0064. Holm's first step in the 12-member family needs p ≤ 0.05/12 = 0.0042, and
in the 8-member family p ≤ 0.00625: **a member separates after Holm essentially only when the r64
arm wins at least nine of the twelve pairs and loses none.** (8, 0) can pass only as the seventh
or later member of the 32K family, behind six members at p ≤ 0.0071 (which means (≥ 9, 0) or
(11, 1)); a single pair in the KIVI arm's favour takes (9, 1) out of reach entirely. That is the resolution n = 12 buys, and it is
why §5 predicts "not separated" as the default and names the one cell where (≥ 9, 0) is reachable.

Everything not in the two families is **descriptive** and carries no corrected p-value: §7's
secondaries, the `kivi4_faithful` rows, the harness-consistency replication, the cross-pod
recovery comparison and the `diag.jsonl` traces.

## 7. Secondary outcomes

- **(a) Protocol-bound or grouping-bound.** `kivi2_singleshot` vs `kivi2_faithful`, within-pod,
  per family / context / task, exact paired McNemar, **uncorrected and descriptive**. Both arms
  prefill in one shot; they differ in grouping (G = 64 against G = 32) and in what happens to the
  fp16 residual (flushed into the quantized store against kept through decode). If the two are
  within 2 pairs of each other on every task, the v1 ss2 effect is the **prefill protocol** and
  the grouping is immaterial at 2 bits on these tasks; if the faithful arm leads by ≥ 3 pairs
  on multi-value in a family, the residual/grouping mechanics carry part of the edge and the
  paper's description of "single-shot KIVI-2" must name which arm it means.
- **(b) `kivi4_faithful`**, descriptively: Wilson 95% intervals per cell (`kvdlra.eval.stats.wilson`),
  beside the streaming 4-bit rows of §2. Its role is the 4-bit reference the paper's fair-quant
  table needs at the published operating point; no contrast is pre-registered on it.
- **(c) Harness-consistency replication of the r64 arm.** This pod's `bugSseed-r64-h256` cells
  against the archived `w18-g1-<family>` cells at the same context, per task, on point
  estimates at the 0.25 resolution (three flipped trials) — the rule of `prereg/filler_realism.md`
  A1.3: a cell more than 0.25 from its archived row is **harness drift** (the archived rows came
  from `w10_ruler.py` on an older `transformers`; the L0 runner's parity with it rests on CPU
  bit-identity tests and on the cycle-control pod of D-005), and every cross-pod comparison in
  this file — (d) and the archived Llama 16K contrast — is suspended for that family until the
  drift is explained. The within-pod primary contrast is unaffected: it pairs two arms that ran on
  the same harness. The flipped `(seed, trial)` keys are listed as descriptive evidence of where
  any drift sits.
- **(d) Single-shot recovery over the streaming arm, per family.** `kivi2_faithful` (this pod)
  against the archived streaming `quant-2bit-kivi` (`w19-a1-<family>`), cross-pod on
  `(seed, trial)`, per task and context, descriptive (`CROSS_POD` caveat). This is the replication
  of the v1 recovery (+0.42 on Llama 16K multi-value, 6 / 1 pairs) on two more families and a
  second context, and it is what the paper's fair-quant table gains: a row for KIVI-2 at its
  published protocol beside the streaming row it printed.
- **(e) The pairing invariant, verified.** For every `(task, ctx, seed, trial)` in a pod, the
  four arms' `prompt_sha256` must be identical. A key where they differ is dropped from every
  paired statistic in that pod and the drop is reported with the key; the paired n is then below
  12 for that cell and §6's resolution note applies with the smaller n.
- **(f) The guard on the r64 arm.** `fixed_k` / `fixed_v` and the maximum `orth_err_k` /
  `orth_err_v` per layer from `diag.jsonl`: whether the shipped default guard ever fired at rank 64
  on any family or context. Expected never (§3); a firing is reported beside §7(c), since it is
  the one knob that differs between this arm and its archived rows.

## 8. Log volume, and what counts as a complete `<label>.log`

The pod log is the only channel back from a vast.ai instance. Rows per sample, from the runner:
one `[trial]` line per record, one cell row (`[<task> ctx<ctx>] <arm> acc=…`) per (arm, task,
ctx), and for the r64 arm the `[diag]` rows `records.drained` prints when the trial's cache is
drained — the three KIVI arms emit none.

**`isvd_r64_h256_seed.yaml` sets no `diag_every`, so the cache default (64) applies.** The
per-sample count is measured, not estimated: the filler-realism pod running this very arm at 16K
(`results/filler_realism/filler_realism-51553610.raw`, read 2026-09-19 06:47, 41 of its r64
samples drained) records **804–810 absorbs per layer per 16K sample** (`absorbs` in the `[diag]`
payload; the Table-4 note's 800) and **416 `[diag]` rows per sample** = 13 rows per layer × 32
layers (twelve completed 64-absorb windows plus `drain_diag`'s end-of-sample flush of the open
one). At 32K each further 16 tokens is one absorb: 806 + 1024 ≈ **1830 absorbs → 29 rows per
layer** (28 completed windows + the flush).

| sample | Llama / Mistral (32 layers) | Qwen (28 layers) |
| --- | --- | --- |
| r64, 16K | 416 `[diag]` + 1 `[trial]` | 364 + 1 |
| r64, 32K | 928 + 1 | 812 + 1 |
| any KIVI arm | 0 + 1 | 0 + 1 |

Per pod (48 r64 samples per context):

| pod | `[diag]` | `[trial]` | cell rows | `[stage]` + banners | expected `<label>.log` |
| --- | --- | --- | --- | --- | --- |
| `ss2_families_mistral` | 48 × 416 + 48 × 928 = 64,512 | 384 | 32 | ~30 | **≈ 65,000** |
| `ss2_families_qwen` | 48 × 364 + 48 × 812 = 56,448 | 384 | 32 | ~30 | **≈ 56,900** |
| `ss2_families_llama` | 48 × 928 = 44,544 | 192 | 16 | ~30 | **≈ 44,800** |

**Does the pod need a `_diag4096` variant of the r64 arm? No — by the arithmetic that decides
it.** The only way the watchdog loses a row for good is a **poll-to-poll gap**: more raw log lines
printed between two 150 s polls than the `vastai logs --tail 30000` window holds (5,000 when the
watchdog's empty-fetch fallback is taken). The `[diag]` rows of a sample are printed in one burst
when the trial drains. At the r64 arm's rates (§9: ≈ 2.1 min per 16K sample, ≈ 4.2 min per 32K
sample) at most **two** 16K samples or **one** 32K sample complete inside a poll: **≤ 832 + 2 rows
(16K) or ≤ 929 rows (32K) per poll** — 32× under the 30,000-line window and 5× under the 5,000-line
fallback. The unfiltered raw log (download progress, warnings) would have to add > 4,000 lines in
150 s on top of that to open a gap. The Table-4 pods set `diag_every: 4096` for the **data
shape** of their §3.4 figure (one summary row per layer per sample), which no reading here needs;
the union of every poll's matched rows, deduped at the terminal marker, is the record, and the
whole run's ≈ 65,000 rows never have to fit inside any one fetch. Keeping the shipped arm also
keeps the record key `bugSseed-r64-h256` — the key the archive, the D-005 pods and `scripts/tables.py`
pair by — instead of a new name the pairing would have to map.

**The cost of that choice, stated.** `scripts/pod/watchdog.sh` appends *every* matched row of
each fetch to `<label>.raw` and dedupes only at the end (`sort -u` → `<label>.log`). Once the
instance log passes 30,000 lines the tail is saturated and each poll re-appends up to ~30,000
matched rows: over a ≈ 20 h run (§9, ≈ 480 polls) `<label>.raw` can reach several million lines
(≈ 2–3 GB) before the terminal-marker `sort -u` — the live filler-realism pod's `.raw` is at
238,000 lines for 17,000 distinct `[diag]` rows after three hours. That is disk and a slow
dedupe, not loss; `pod.py harvest` reads the deduped `.log`. The launch machine needs ≈ 3× that
headroom under `results/<pod>/` (gitignored, but inside the iCloud-synced tree — D-007). A
one-line per-poll dedupe in the watchdog (`sort -u -o "$H/${lab}.raw" "$H/${lab}.raw"` after the
append) removes the growth for every pod; it is an L6 item, not made here, and **not** made to a
watchdog script that is running. **Contingency, pre-registered:** the D-005 harvest — the same
arm, the same default, ≈ 40,000 `[diag]` rows across its two gist arms — exercises exactly this
path before these pods launch; if that harvest is short, or the fetch fails on volume, the
amendment that precedes launch adds `configs/arms/isvd_r64_h256_seed_diag4096.yaml` (the same
`cache:` block plus `diag_every: 4096`, no `legacy_name`, its stem in `POST_V1`) to the three
pods, the records then key by that name, and the pairing is on `(seed, trial)`, never on the arm
string.

**The completeness test is on the deduped `<label>.log`.** Exact: `grep -c '^\[trial\]'` must be
**384** (Mistral, Qwen) or **192** (Llama), and `pod.py check` must find all 32 / 32 / 16 cells at
n = 12 — that is the gate. Approximate: `grep -c '^\[diag'` ≈ 64,500 / 56,400 / 44,500 (the
per-layer window count moves by one if a sample's absorb count crosses a multiple of 64; the
cycled haystack is a fixed token count per context, so it should not). A `.log` whose `[trial]`
count is short is short by construction — read `trials.jsonl` and the `error` lines before
concluding truncation, and do not harvest it as final.

## 9. Budget

**Unit and anchors.** "min/sample" is the wall-clock of one `(arm, task, seed, trial)`; a cell is
12 samples and a four-task context is **48 samples per arm** (the tasks-per-cell correction of
`prereg/filler_realism.md` §GPU budget). Anchors, all on an A100 40 GB:

- **r64 at 16K: 2.1 min/sample measured** (W19 `a1q`: "Streaming r64: ~25 min per 16K cell" =
  25/12; `prereg/filler_realism.md` A1.4), **budgeted at 3.0**; the running cycle-control pod is
  on pace with it (34 samples in under an hour of run time). **32K: twice the absorbs → 6.0**
  (derived, not measured: no r64 32K rate is on record; the tracker's cost is one core SVD per
  absorb per stream, and §8's absorb count doubles).
- **KIVI single-shot at 16K: 1.5 min/sample** (the streaming arm's 43.7 vs 25.9 ms/token decode ≈
  1.7× `full`'s 1.0, `prereg/filler_realism.md`; the v1 single-shot pod ran at 16K and left no
  rate of its own on record, so the streaming arm's stands for it). **32K: 3.5 — an estimate**: the fp16 single-shot prefill is quadratic in T on the attention
  term and the decode dequantizes a store twice the size at every one of the 40 steps; 3.5 is
  2.3× the 16K rate. The faithful arms add one post-hoc quantization pass per layer, inside the
  estimate.
- **Pod overhead: 60 min** — boot, clone, `pip`, the quanto JIT build, the weight download that
  stalled for ≈ 1 h on a Table-4 pod (D-011 addendum 2; `prereg/filler_realism.md` A1.4).

| arm | 16K min/sample | × 48 | 32K min/sample | × 48 |
| --- | --- | --- | --- | --- |
| `isvd_r64_h256_seed` | 3.0 | 144 | 6.0 | 288 |
| `kivi2_faithful` | 1.5 | 72 | 3.5 | 168 |
| `kivi2_singleshot` | 1.5 | 72 | 3.5 | 168 |
| `kivi4_faithful` | 1.5 | 72 | 3.5 | 168 |
| **per context** | | **360 min** | | **792 min** |

| pod | compute | + overhead | point estimate | **`gpu_budget_h` (2× bar)** |
| --- | --- | --- | --- | --- |
| `ss2_families_mistral` | 360 + 792 = 1152 min | + 60 | 1212 min = **20.2 h** | **40.4** |
| `ss2_families_qwen` | 1152 min | + 60 | **20.2 h** | **40.4** |
| `ss2_families_llama` | 792 min | + 60 | 852 min = **14.2 h** | **28.4** |
| **total** | | | **54.6 GPU-h** | **109.2 GPU-h** |

At the **$0.40–0.74/h** the Table-4 and filler pods paid for an A100 40 GB (D-011 and its addenda;
`prereg/filler_realism.md` A1.4, A1.7):

| | GPU-h | × $0.40 | × $0.74 |
| --- | --- | --- | --- |
| three pods, point | 54.6 | $21.8 | $40.4 |
| **three pods, bar** | **109.2** | **$43.7** | **$80.8** |
| after the first two arms only (the primary contrast landed; §3 order), with overhead | 12.2 + 12.2 + 8.6 = 33.0 | $13.2 | $24.4 |
| without `kivi4_faithful` (descriptive, §7(b)) | 54.6 − 10.8 = 43.8 | $17.5 | $32.4 |

**This experiment asks for ≈ 55 GPU-hours expected, ≈ 109 GPU-hours at the bar — $22–40 expected,
$44–81 at the bar — against a credit of $97.2 before the filler-realism pods' own spend (D-005
addendum 2; their bars come to ≈ $13 at most).** The three pods run on three instances at once,
so the wall clock is the longest of them (≈ 20 h expected, 40 h at its bar), not the sum. The
owner decides; the two rows below the total are the pre-registered ways to spend less, and the arm
order of §3 is what makes the first of them a design and not a salvage: after `isvd_r64_h256_seed`
and `kivi2_faithful`, every pod holds its primary contrast and its Holm members. Dropping
`kivi4_faithful` costs the 4-bit reference row and nothing in §4–§6.

`gpu_budget_h` in each pod YAML is the pre-registered bar, enforced on the pod itself by
`pod.py launch --max-hours` (`boot.sh` runs the entrypoint under `timeout`; a run that reaches the
bar prints `===RUN_TIMEOUT_…===` and is harvested as `RUN_FAILED` with `timeout: true`). Overrun is
a stop-and-report, not a silent extension: a pod still running past its bar is a pod to kill and
diagnose (`prereg/hygiene_table4.md` §9), and the first r64 32K samples' measured rate is the
first thing to read against the 6.0 assumption.

## 10. Provenance

- Pods: `configs/pods/ss2_families_mistral.yaml`, `configs/pods/ss2_families_qwen.yaml`,
  `configs/pods/ss2_families_llama.yaml`. Arms: `configs/arms/isvd_r64_h256_seed.yaml`,
  `configs/arms/kivi2_faithful.yaml`, `configs/arms/kivi2_singleshot.yaml`,
  `configs/arms/kivi4_faithful.yaml` — none of them new, none edited. Tasks:
  `configs/tasks/ruler_inhouse_16k.yaml`, `configs/tasks/ruler_inhouse_32k.yaml` (or, under the
  filler condition's branch 3, `ruler_v2_16k.yaml` / `ruler_v2_32k.yaml`, named by the amendment).
- **This file must be committed strictly before the launch commit** — and any amendment under
  *The filler condition* strictly before it as well. `scripts/pod.py launch` refuses otherwise
  (a pushed SHA, the prereg's first commit a strict ancestor of the launch SHA, a clean tree), and
  `scripts/pod.py check` re-checks the order against the manifest's `git_sha` at harvest. The
  commit that adds this file adds the three configs and the tests and launches nothing.
- Launch: `scripts/pod.py launch --pod ss2_families_<family> --offer <id>` per pod (the r64 arm
  runs first on each), `--max-hours` defaulting to the pod's `gpu_budget_h`; the watchdog under
  `caffeinate -s -i scripts/pod/watchdog.sh ss2_families_<family>`; the pod self-destructs
  `GRACE_S` after its final marker. Each launch is a `DECISIONS.md` entry with pod name, this
  file's first-commit SHA, the launch SHA, offer, rate, and bar.
- Outputs: `results/ss2_families_mistral/`, `results/ss2_families_qwen/`,
  `results/ss2_families_llama/`, each with `manifest.json` (git SHA, config hash, model revision,
  torch/CUDA/transformers versions, GPU, wall clock, command line, `errors`, `records`),
  `trials.jsonl` (32 / 32 / 16 cells × 12 records; every row carries `prompt_sha256`),
  `diag.jsonl` (the r64 arm's rows), `env.txt` (rebuilt from the log's ENV block), `pods.txt`.
- A number from these pods is citable only once `scripts/pod.py check` passes on its directory
  (config hash, commit order, every cell at n = 12 with zero errors, `env.txt` at the pyproject
  pins) and the table regenerates from the committed records (CLAUDE.md). The Holm result of §6
  goes to `DECISIONS.md` with the evidence path (gate G2 line 4).

## 11. What these pods do not decide

Whether the cycled generator is retained (that is D-005, and this file waits on it); anything
on generator v2 unless the filler condition's amendment applies; the memory or stored-bits
comparison between the r64 configuration and KIVI (billed elsewhere; no arm here changes it);
the perplexity axis (no `ppl` task); the 4-bit KIVI baseline's standing beyond a descriptive row;
official RULER or LongBench; anything about the tracker, the guard's defaults, the exact tier or
the kernel; and the Llama 16K cell, which is not re-run. Lane item 4 asks one question — does the
multi-value claim survive KIVI-2 at its published protocol, per family and context — and these
pods answer that one.
