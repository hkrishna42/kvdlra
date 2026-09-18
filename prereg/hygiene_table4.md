# Pre-registration — `hygiene_table4_qwen` + `hygiene_table4_llama`

**STATUS: awaiting owner go.** Written before either pod is launched. Launch is authorized in
principle (`docs/plan/DECISIONS.md` D-011) but is a separate, later commit: `scripts/pod.py
launch` refuses unless this file's first commit is a *strict* ancestor of the launch commit, so
the commit that adds this file launches nothing.

Two pods, one pre-registration: `PodCfg` carries a single `model`, so the Qwen matrix and its
Llama replicate are separate configs describing one experiment.

---

## 1. Purpose

ICML plan item 0.1. The Week-17 Qwen cell `bug-r256 [T=16384] ppl=27531.666`
(`results/w17-qwen-lines.txt:33`, commit `ee8c0ab`) was repaired by turning on the relative
singular-value floor (`min_sv_frac=1e-2`). The audit attributes the divergence to a *loss of
orthonormality* in the tracked basis U rather than to the exact tier's "rank siphoning"
(`docs/plan/CODE_AUDIT.md` Q4). L1.1 shipped a repair for exactly that: a tolerance guard that
re-orthonormalizes U when `‖UᵀU − I‖_F` crosses a threshold, plus an optional periodic thin QR.

**The question this pod answers:** does the orthonormality guard *alone* remove the divergence,
with the singular-value floor off?

- **Yes** → the paper's mechanism section says **orthonormality ratchet**, the floor becomes an
  optional belt-and-braces knob, and the rank a run reports is the rank it actually carries.
- **No** → the floor stays **required**, the paper keeps both, and names the ratchet as the
  substrate the floor happens to suppress.

Nothing else is being decided here. In particular this pod does not compare the gist against any
baseline method, does not touch the exact tier, and produces no memory or throughput claim.

## 2. Measured baseline — what is already known, and what it predicts

Two measurements bracket the expected behaviour, and they disagree, which is why the arms below
include an unguarded control rather than assuming the divergence reproduces.

- **The tracker alone does not ratchet on the synthetic recipe.** The shipped step
  (`kvdlra.tracker.isvd.isvd_step`, `theta=None`, `min_sv_frac=0`) run on the plan's recipe
  (n=512, rank cap 256, block 16, bf16-rounded, 1400 blocks) reaches a maximum
  `‖UᵀU − I‖_F` of **6.6e-4 – 7.3e-4** across seeds 0–2, growing smoothly as √t and never
  crossing the 1e-3 repair threshold. The pre-Week-7 augmentation (plain residual QR, no
  re-orthogonalization) reaches 44–45 within 60 blocks on the same stream, so the probe is
  sensitive; the shipped step simply already defends against the mechanism *inside* the step.
  Write-up: L1 Task 1's ratchet-tracking findings, recorded as **DECISIONS D-012** and copied
  into **`docs/plan/cleanup/l1-ledger.md`**, both written at merge (the pattern
  `docs/plan/cleanup/l0-ledger.md` already follows for L0); the measured numbers above stand
  regardless of when either lands.
- **The production layer does ratchet.** Through `BugStreamingLayer` (bf16 storage, RoPE round
  trip, chunked prefill, rank 256, floor off), `‖UᵀU − I‖_F` measured
  **6.0e-4 (8K) → 1.5e-3 (16K) → 3.7e-3 (32K) → 35.1 (36K)**
  (`docs/plan/CODE_AUDIT.md:172`). With `min_sv_frac=1e-2` it never starts (≤ 4e-6).
- **Rank relative to width matters.** Qwen2.5-7B stores n = 4 KV heads x 128 = **512** channels
  per layer, so r=256 is n/2 — the regime the audit names as reaching onset before 16K on real
  Qwen KV. Llama-3.1-8B stores n = 8 x 128 = **1024**, so the same r=256 is n/4. That is the
  stated reason the Llama pod is a replicate and not a second chance at the divergence.

**Consequence for this run:** at 16K the production trace sits at ~1.5e-3, *above* the
`orth_fix_tol = 1e-3` repair threshold and far below the `orth_abort_tol = 1e-1` raise
threshold. The tolerance guard is therefore expected to FIRE on the r=256 arms and to fire
rarely or never on the r=128 arms. That expectation is written per arm in §5 before the run.

## 3. Arms, tasks, n

**Factors.** rank ∈ {128, 256} x guard ∈ {off, tolerance, qr64} x floor ∈ {off, 1e-2}. The
floor-on cells get no "off" variant: with `min_sv_frac=1e-2` the error stays ≤ 4e-6, so an
unguarded floor-on arm is the guarded one. That leaves 5 cells per rank.

The tolerance guard is the **default** in the shipped cache, so the control has to switch it off
explicitly (`orth_fix_tol: null` and `orth_abort_tol: null`); an arm that merely leaves
`qr_every` unset is still guarded. The tripwire still records the `[diag]` `orth_err` trace on
the unguarded arms, so the ratchet is *measured* where it is not *corrected*.

`hygiene_table4_qwen` — `Qwen/Qwen2.5-7B-Instruct`, bf16, 11 arms:

| arm | rank | guard | floor |
| --- | --- | --- | --- |
| `isvd_r128_noguard` | 128 | off | off |
| `isvd_r128_tol` | 128 | tolerance (1e-3 / 1e-1) | off |
| `isvd_r128_qr64` | 128 | tolerance + QR every 64 absorbs | off |
| `isvd_r128_f0.01_tol` | 128 | tolerance | 1e-2 |
| `isvd_r128_f0.01_qr64` | 128 | tolerance + QR/64 | 1e-2 |
| `isvd_r256_noguard` | 256 | off | off |
| `isvd_r256_tol` | 256 | tolerance | off |
| `isvd_r256_qr64` | 256 | tolerance + QR/64 | off |
| `isvd_r256_f0.01_tol` | 256 | tolerance | 1e-2 |
| `isvd_r256_f0.01_qr64` | 256 | tolerance + QR/64 | 1e-2 |
| `full` | — | — | — |

`hygiene_table4_llama` — `unsloth/Meta-Llama-3.1-8B-Instruct`, bf16, 4 arms:
`isvd_r256_noguard`, `isvd_r256_tol`, `isvd_r256_qr64`, `full`.

Every other cache knob is the plain r128/r256 arm's, unchanged: `recent_window` 32,
`absorb_block` 16, `n_sink` 4, `retention` fifo, coordinate tier sized to the whole prefill, no
exact tier, no quantization. The three guard/floor knobs and `diag_every` are the only
differences between the cells, which is what makes the contrast a contrast. `qr_every: 64` fires
on two triggers, not one: every 64 absorbs unconditionally, and immediately after that stream's
own tracked rank changes (`bug_cache.py:844-848`) — the `_qr64` arms are "periodic + rank-change
repair", not periodic alone.

The three `_tol` cells are the shipped arms `isvd_r128`, `isvd_r256` and `isvd_r256_f0.01` at
their default knobs; they are separate files only because those three arms are frozen against
the v1 parity golden (`tests/golden/legacy_arm_kwargs.json` pins their keyword set exactly) and
are named by the archived `w17_floor` pods, so the `diag_every` §8 requires cannot be added to
them without changing what an archived config claims. Same cache, same numbers, wider
diagnostic window.

**Tasks and n.**

- `ppl_16k_pg19val` — teacher-forced perplexity at 16K on **PG-19 validation**, 2048-token
  scoring window, **n = 32 non-overlapping windows** (the corpus supplies 160 at this span).
  Both pods. Per-window NLLs land in `pplw.jsonl`; the paired statistics are computed on them,
  never on the pooled number.
- `ruler_inhouse_16k` — the in-house needle generator at 16K, 4 sub-tasks x 6 trials x 2 seeds
  = **12 records per (arm, sub-task)**. Qwen pod only. Every arm runs it (a pod's tasks apply to
  every arm); only the r128 arms are *read* for the neutrality outcome in §7.

One corpus per context length per pod, which is what `load_pod` now enforces: each pod names
exactly one `ppl` task. The in-house filler is ten cycled sentences and is suspected of
flattering a low-rank gist — which is why retrieval here is a **neutrality check between two
gist arms at the same rank**, not a headline number.

## 4. Primary contrast and decision rule

**Primary contrast:** `isvd_r256_tol` vs `isvd_r256_noguard`, Qwen, 16K perplexity, **paired
over the 32 PG-19 windows**, in bits/token (`nll_sum_nats / (ntok · ln 2)`), with a 95% paired
bootstrap CI. Command:

```
scripts/tables.py ppl --pod hygiene_table4_qwen --baseline isvd_r256_noguard
scripts/tables.py ppl --pod hygiene_table4_qwen --baseline isvd_r256_f0.01_tol
scripts/tables.py ppl --pod hygiene_table4_qwen --baseline full
```

**Decision rule, fixed now:**

1. **Ratchet.** Requires **both**: (a) `isvd_r256_tol` is equivalent to `isvd_r256_f0.01_tol`
   within **±0.05 bits/token** (TOST on the paired per-window differences, p < 0.05 after the
   Holm correction of §6 — `tables.py ppl` itself applies no family correction; §6 says how the
   four members are combined by hand); **and** (b) `isvd_r256_f0.01_tol` itself is within **1.0
   bit/token of `full`**. (a) alone is not enough: if the floor has collapsed the tracked rank
   (§5's `isvd_r128_f0.01_tol` caveat applies just as well at r256; §7 "Effective rank under the
   floor"), `isvd_r256_tol` could match a *bad* reference within 0.05 bits for the wrong reason.
   Both true → the mechanism the paper reports is the **orthonormality ratchet**; the
   singular-value floor becomes optional and stays a default-off knob; §3.4 states the guard as
   the fix.
2. **Floor required.** If `isvd_r256_tol` still exceeds `full` by **more than 1.0 bit/token**
   (equivalently, more than 2x the full-KV perplexity), lower CI bound above that threshold →
   the floor is **required**; the paper keeps both and names the ratchet as the substrate.
   If `isvd_r256_f0.01_tol` is **also** more than 1.0 bit/token from `full`, the floor did not
   close the gap either: this branch still fires (it is the guard arm's distance to `full` that
   defines it), the paper still keeps both, **and it says so** — "required" then means required
   and not sufficient at r256, and neither knob is reported as the fix.

3. **Neither** → reported as measured, both kept, no third run pre-authorized, for either of two
   readings: (a) fails on its own — the guard narrows the gap by orders of magnitude but the
   residual gap to `isvd_r256_f0.01_tol` is above the ±0.05 margin while still under 1 bit/token
   — the ratchet is the substrate, the floor is the fix, and the guard is the diagnostic that
   proves it; **or** (b) fails — `isvd_r256_f0.01_tol` itself sits more than 1.0 bit/token from
   `full` — **the floor arm collapsed** (`eff_rank` in `diag.jsonl` is the read), branch 1 cannot
   fire on a collapsed reference regardless of (a), and the paper reports neither mechanism as
   settled from this pod.

These three outcomes partition every result. 1 and 2 are mutually exclusive by construction: 1
needs `isvd_r256_f0.01_tol` within 1.0 bit of `full` and `isvd_r256_tol` within 0.05 bits of
that (so `isvd_r256_tol` is itself close to `full`); 2 needs `isvd_r256_tol` to be *more* than
1.0 bit from `full`. 3 is everything else, including both the case where `isvd_r256_noguard`
does **not** reproduce the divergence at 16K and the case where the reference itself collapsed
(condition (b)). The first of those is a live possibility (see §5) and is not a failure of the
pod: it would mean the 27,531 cell depends on something outside the guard/floor factor, and
§3.4 would then say so and cite this pod.

## 5. Prediction per arm, written before the run

Perplexity in bits/token relative to `full` on Qwen 16K unless stated. "Fires" = at least one
`fixed_k`/`fixed_v` true in that arm's `diag.jsonl` rows.

| arm | guard fires? | prediction |
| --- | --- | --- |
| `isvd_r128_noguard` | n/a (off) | no divergence; the Week-17 r128 cells were well-behaved. The r128 reference row. |
| `isvd_r128_tol` | **no** (trace stays under 1e-3 at this rank) | within 0.01 bits of `isvd_r128_noguard`; plausibly bit-identical. |
| `isvd_r128_qr64` | yes, unconditionally every 64 absorbs | within 0.05 bits of `isvd_r128_tol`. A larger gap means the repair itself costs accuracy and that is a finding. |
| `isvd_r128_f0.01_tol` | no (floor keeps the error ≤ 4e-6) | **may be worse than floor-off.** The floor is relative to σ_max, and Qwen's massive key-bias channels can collapse the tracked rank (the audit saw 256 → 1 on a Qwen-like synthetic). `eff_rank` in `diag.jsonl` is the diagnostic. |
| `isvd_r128_f0.01_qr64` | forced, but with nothing to repair | indistinguishable from `isvd_r128_f0.01_tol`. |
| `isvd_r256_noguard` | n/a (off) | **the divergence cell.** Predicted three digits or more of perplexity at 16K. This is the least certain prediction on the page: the layer probe only reached 35.1 at 36K, while the archived cell diverged at 16K on real Qwen KV, so the onset on real weights is earlier than the synthetic probe showed. If it does not diverge, outcome 3 of §4 applies. |
| `isvd_r256_tol` | **yes** (the trace crosses 1e-3 before 16K) | orders of magnitude better than `isvd_r256_noguard`. Whether it reaches `isvd_r256_f0.01_tol` is the question of §4 and is deliberately not predicted. |
| `isvd_r256_qr64` | yes, unconditionally | at least as good as `isvd_r256_tol`; the periodic form repairs before the tolerance is crossed. |
| `isvd_r256_f0.01_tol` | no | recovers the cell (Week-17, measured). The reference for outcome 1. |
| `isvd_r256_f0.01_qr64` | forced, nothing to repair | indistinguishable from `isvd_r256_f0.01_tol`. |
| `full` | n/a | the reference every Δ is taken against. |

Llama: all three r256 arms predicted indistinguishable from each other within 0.05 bits, because
r=256 is n/4 there and the trace is not expected to approach 1e-3. A guard that costs accuracy
where there is nothing to repair would show up here and nowhere else.

**Abort is a possible outcome, not an accident.** If `‖UᵀU − I‖_F` exceeds `orth_abort_tol =
1e-1` on a guarded arm, the cache raises `OrthonormalityError`, the trial is recorded as an
`error`, and `scripts/pod.py check` FAILS the pod (Ruling R29 — no tolerance knob). That is the
designed behaviour and it is pre-registered as such: the cell is reported as "the guard could
not hold the basis", counted in n, and the pod's other cells are not cited until the run is
repeated with the abort understood. It is *not* to be re-run with the tripwire disabled.

## 6. Family size and correction

**Family size 4**, Holm-corrected at α = 0.05, fixed now:

1. `isvd_r256_tol` vs `isvd_r256_noguard` — Qwen 16K, paired (the primary contrast).
2. `isvd_r256_tol` vs `isvd_r256_f0.01_tol` — Qwen 16K, TOST at ±0.05 bits (outcome 1 above).
3. `isvd_r256_qr64` vs `isvd_r256_tol` — Qwen 16K, paired (periodic + rank-change repair vs
   tolerance-only).
4. `isvd_r256_tol` vs `isvd_r256_noguard` — Llama 16K, paired (the replicate).

**How the correction is actually computed.** `scripts/tables.py ppl --pod P --baseline B`
computes one TOST p-value per call — the `p_tost` field on the non-baseline arm's row, at the
tool's default **±0.05 bits/token** margin — and applies **no family correction of its own**;
each invocation is independent of the others. The Holm step is applied **by hand**, once all
four numbers exist, over the four raw p-values via `kvdlra.eval.stats.holm` (shipped, tested in
`tests/test_stats.py`, not otherwise called by this pod). Every member's p-value is that same
`p_tost`, whatever the member's own reading is — for members 1 and 4 it is bookkeeping for the
family, not the basis of their own decision (the paired CI is, per §4 and the replicate check in
§7): member 1 is `isvd_r256_tol`'s `p_tost` from `--pod hygiene_table4_qwen --baseline
isvd_r256_noguard`; member 2 is the same row's `p_tost` from `--baseline isvd_r256_f0.01_tol`
(the call §4 outcome 1 already needs); member 3 is `isvd_r256_qr64`'s `p_tost` from `--baseline
isvd_r256_tol`; member 4 is `isvd_r256_tol`'s `p_tost` from `--pod hygiene_table4_llama --baseline
isvd_r256_noguard`. Four calls, four `p_tost` values, one `holm([p1, p2, p3, p4])`.

Everything else on this page is **descriptive** and carries no p-value: the r128 rows, the
`diag.jsonl` traces, and the retrieval neutrality check. 12 records per retrieval cell cannot
support a fifth member of a corrected family, and pretending otherwise would be the cheapest
way to turn a hygiene pod into an overclaim.

## 7. Secondary outcomes

- **The §3.4 figure.** Per arm and per layer, the maximum `orth_err_k`/`orth_err_v` and the
  minimum `eff_rank_k`/`eff_rank_v` over the stream, from `results/<pod>/diag.jsonl`. This is
  the trace that shows the ratchet directly rather than through its effect on perplexity, and
  it is the reason the unguarded arms are in the matrix at all.
- **Effective rank under the floor.** `eff_rank` on the `f0.01` arms answers the audit's open
  caveat: whether the floor collapses the tracked rank on a Qwen spectrum, i.e. whether a
  floor-on arm bills a rank it does not carry.
- **Retrieval neutrality at r128.** `isvd_r128_tol` vs `isvd_r128_qr64` on the four in-house
  sub-tasks, 12 records per cell, reported as Wilson intervals. **Pass = the intervals
  overlap.** A non-overlap at r128 would mean the repair moves the basis enough to change which
  needles survive, which would need its own investigation before the guard's default changes.
- **The Llama replicate** — whether the guard costs anything where it has nothing to repair.

## 8. Log volume, and what counts as a complete `<label>.log`

The pod log is the only channel back from a vast.ai instance. At 16K a sample runs ~1022 absorbs
per layer, so the 64-absorb `diag_every` default would print ~17 `[diag]` rows per layer per
sample. Every gist arm here therefore sets **`diag_every: 4096`**, larger than any sample's
absorb count: no diagnostic window ever completes, so `drain_diag`'s end-of-sample flush is the
only row emitted, exactly one per layer per sample. That choice is about the **data shape** —
guaranteeing one summary row instead of several thousand — not about the watchdog's fetch size;
what that volume costs against the watchdog's log-carrying capacity is a separate, downstream
question, worked out below.

Expected `[diag]` volume (28 layers on Qwen2.5-7B, 32 on Llama-3.1-8B; `full` emits none):

| pod | perplexity | retrieval | `[diag]` total | other rows | expected total |
| --- | --- | --- | --- | --- | --- |
| `hygiene_table4_qwen` | 10 arms x 32 windows x 28 = 8,960 | 10 x 48 x 28 = 13,440 | 22,400 | ~600 (`[trial]`, `[pplw]`, `ppl=`, banners) | **≈ 23,000** |
| `hygiene_table4_llama` | 3 arms x 32 x 32 = 3,072 | — | 3,072 | ~70 | **≈ 3,150** |

**What the watchdog can and cannot lose.** `scripts/pod/watchdog.sh` polls every 150 s
(`:64`); each poll runs `vastai logs --tail 30000` (`:38`) — the *current* tail of the whole
instance log at that moment, not a whole-run ceiling — greps it for the `ROWS` pattern (`:28`),
and **appends** the matches to `<label>.raw` (`:43`). Because every poll appends rather than
overwrites, `<label>.raw` accumulates the *union* of every fetch across the run's ~hundreds of
polls; only when a poll's fetch contains the terminal marker (`===ALL_DONE_<pod>_<sha>===`, or a
failure marker) does the watchdog dedupe it with `sort -u` into `<label>.log` (`:49-50`), which
`pod.py harvest` reads. So **30000 is a per-poll window, not a whole-run cap**: the run's ≈23,000
expected Qwen rows never have to fit inside any one fetch, only inside the deduped union of
~hundreds of them.

The only way a row is lost for good is a **poll-to-poll gap**: if the raw log — matched rows plus
every banner, download-progress and warning line the harness never filters — grows past 30000
lines between two consecutive 150 s polls, the earliest of those lines scroll out of the tail
before either poll's fetch captures them. The worst-case *rate* that has to stay under that: each
sample contributes at most 32 `[diag]` rows (Llama's per-layer count, the larger of the two
pods) plus one `[pplw]` line per arm per sample — **33 rows/sample**. The Qwen budget (§9: 4.0
GPU-h over 11 arms x 80 samples) implies ~16 s/sample on average, so even a generous twenty
samples finishing inside one 150 s poll is ~660 rows — two orders of magnitude under the
30000-line window.

**The completeness test is therefore on the deduped `<label>.log`, not on any one fetch.** A
`.log` file is written only once a poll's fetch has captured a terminal marker, so its mere
existence already implies `===ALL_DONE_<pod>_<sha>===` (or a failure marker) was seen —
independently re-checking for the `===RUN_SHA_<sha>===` boot header *inside a fetch* adds
nothing on top of that: the header is emitted once, near the start of the run
(`scripts/pod/boot.sh:53`), gets captured on the *first* poll while the whole log is still far
under 30000 lines, and then lives in `.raw`/`.log` permanently (append-only, deduped, never
truncated) — a *later* fetch that does not happen to contain it is normal, not a sign of loss.
What a poll-to-poll gap actually leaves behind is a **short `.log`**: expect ≈23,000 rows for
Qwen and ≈3,150 for Llama (table above); a completed Qwen `.log` under ~20,000 rows is short by
construction and must not be harvested as final. Harvest refusing to shrink an existing
`trials.jsonl` is the second line of defence, not the first.

If the layer counts above turn out wrong for these checkpoints, the multiplier is the only thing
that changes; recompute before concluding a `.log` was truncated.

## 9. Budget

| pod | GPU-h | note |
| --- | --- | --- |
| `hygiene_table4_qwen` | 4.0 | 11 arms x (32 perplexity windows + 48 retrieval records) at 16K |
| `hygiene_table4_llama` | 2.0 | 4 arms x 32 perplexity windows at 16K |
| **total** | **6.0** | ≈ $5–8 at the rates the Week-18/19 pods paid |

`gpu_budget_h` in each pod YAML is the pre-registered bar. Overrun is a stop-and-report, not a
silent extension: the watchdog destroys the instance on the completion marker, and a pod still
running past its budget is a pod to kill and diagnose.

## 10. Provenance

- Pods: `configs/pods/hygiene_table4_qwen.yaml`, `configs/pods/hygiene_table4_llama.yaml`.
  Arms: `configs/arms/isvd_r{128,256}_{noguard,tol,qr64,f0.01_tol,f0.01_qr64}.yaml` and
  `configs/arms/full.yaml`. Tasks: `configs/tasks/ppl_16k_pg19val.yaml`,
  `configs/tasks/ruler_inhouse_16k.yaml`.
- **This file must be committed strictly before the launch commit.** `scripts/pod.py launch`
  refuses otherwise, and `scripts/pod.py check` re-checks the order against the manifest's
  `git_sha` at harvest. The commit that adds this file adds the configs too and launches
  nothing.
- Outputs: `results/hygiene_table4_qwen/` and `results/hygiene_table4_llama/`, each with
  `manifest.json` (git SHA, config hash, model revision, corpus SHA256, torch/CUDA versions,
  GPU, wall clock, command line), `trials.jsonl`, `ppl.jsonl`, `pplw.jsonl`, `diag.jsonl`,
  `env.txt`.
- A number from this pod is citable only once `scripts/pod.py check` passes on its directory and
  the table regenerates from the committed records.

## 11. What this pod does not decide

The default value of `qr_every` (a repair schedule is a knob, not a result); whether the floor
ships on by default for any arm other than the r256 cells measured here; anything about the
exact tier, the seed, surprise scoring, or the kernel; and any memory, throughput or
baseline-comparison claim. Item 0.1 asks one question and this pod answers one question.
