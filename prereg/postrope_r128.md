# Pre-registration — `postrope_r128` (ADR 0001 option (i): post-RoPE tracking at r = 128)

**STATUS: written before the pod exists; launch is a later DECISIONS line under D-011, after the
Gate-1 Stage-1 Llama harvest is committed.** Lane L4 Task 9. This file is committed strictly
before the launch commit; `scripts/pod.py launch` refuses otherwise. It launches nothing.

## 1. Purpose

ADR 0001 §4: "(i) runs in parallel as an accuracy experiment only — config
`isvd_postrope_r128`, one Gate-1-style row on Llama 16K, n = 24. It is not a kernel fallback …
it is insurance if (iii) misses its FLOP target on some GPU." The pre-RoPE basis "roughly
halv[es] the reconstruction error at matched rank" (paper §2.2), and the surprise score that
admits tokens to the exact tier is computed against the tracked basis, so a post-RoPE basis
"changes what reads as surprising — retrieval, not just perplexity, must be re-measured" (ADR
§3). **The question:** at 16K on Llama-3.1-8B, is post-RoPE tracking at r = 128 non-inferior
to the shipped pre-RoPE r = 64 configuration on the **three** Gate-1 retrieval tasks the rule
reads (`niah_single`, `niah_multikey`, `niah_multivalue` — `vt` is measured and reported but is
**descriptive**, not part of the decision, under D-018; §4 and §6) and on perplexity — at twice
the coordinate bytes?

## 2. Measured baseline

- The comparison rows are the Gate-1 Stage-1 Llama pod's (`prereg/gate1_tracker_swap_v2.md`
  §3 arms 1–2: `full`, `isvd_r64_h256_seed`) on the same task files; their numbers are that
  pod's to report. The prior from the pre-flight (D-011 addenda 10–11): the r64 configuration
  retrieves 0.83 on `niah_single` at 16K on real documents, and the `full` ceiling **on the four
  Gate-1 tasks** is **1.00 / 1.00 / 0.92 / 0.75** — `niah_single` / `niah_multikey` /
  `niah_multivalue` / `vt`, the 0.75 being what D-018 made `vt` descriptive for. The other 0.92
  the pre-flight measured is `niah_multiquery`'s, which is **not** a Gate-1 task: it lives in
  `configs/tasks/ruler_v2_16k.yaml`, not in this pod's `ruler_v2_16k_g1`.
- **That ceiling is a prior about the model, not a row this pod's cells are read against.** It
  was measured at **n = 12 on `ruler_v2_16k`** (`design: {haystacks: 2, depths: 3, codes: 2}`),
  a different design from this pod's **n = 24 on `ruler_v2_16k_g1`** (`codes: 4`) — different
  prompts, a different code draw per cell, twice the trials. The rows §4 pairs against are
  Stage 1's own, on this pod's task file (D-011 addendum 10).
- Stored bits (`kvdlra.accounting.bug_footprint(...).stored_bits()`, ADR §3): r128 post-RoPE
  0.282× of the fp16 cache at 16K against r64's 0.151× — printed beside every row; this row
  buys accuracy with bytes and says so.
- The knob is exact at full rank (`tests/test_rope_basis.py`: DynamicCache parity, the middle's
  reconstruction is the stored post-RoPE key) and is not a no-op at low rank.

## 3. Arms, tasks, n

One arm, one pod: `isvd_postrope_r128` (`configs/arms/isvd_postrope_r128.yaml`) = arm 2 of
the Gate-1 design with `rank: 128` and `rope_basis: "post"`; everything else — the 256-token
surprise tier, 4 sinks, ring 32, absorb 16, the warm-up seed, the shipped guard — is arm 2's.
Tasks `ruler_v2_16k_g1` (the four Gate-1 tasks, n = 24 from one seed on the 2 × 3 × 4 design)
and `ppl_16k_pg19val` (32 non-overlapping 2048-token windows), chunk 4096, on
`unsloth/Meta-Llama-3.1-8B-Instruct` in bf16 — the same task files, seed and model as
`gate1_v2_stage1_llama`, so the generator builds byte-identical prompts (`prompt_sha256`) and
`frontier.windows` cuts identical windows (`window_idx`). **Pairing across the two pods** is
the pre-flight re-run's rule (D-011 addendum 10, reading (iii)): a retrieval key `(task, seed,
trial)` pairs only when the two records' `prompt_sha256` agree; a mismatch drops the key from
every paired statistic and is reported with the key; a perplexity window pairs on `window_idx`
within the corpus `pg19-val`, and a window set that is not exactly Stage 1's refuses the TOST
(`records.paired_window_bits`). §10 is the code that does it.

## 4. Reading and decision rule

Per task (`niah_single`, `niah_multikey`, `niah_multivalue`; `vt` descriptive per D-018): exact
paired McNemar (`kvdlra.eval.stats.mcnemar_exact`) of `isvd_postrope_r128` (b) against
`isvd_r64_h256_seed` (a) on the paired keys, Holm over the m = 3 tasks (`stats.holm`); a task is
**lost** only when both the raw loss `(a_favored − b_favored) / n_paired > 0.03` AND Holm
p < 0.05 (the `prereg/bf16_gist.md` §4 rule).

**The 0.03 clause does not bind at this design's n = 24, and that is stated here rather than
left to be discovered.** One flipped pair moves the point estimate by 1/24 = **0.0417**, so any
net loss of one or more pairs already clears 0.03 and the first condition is satisfied whenever
the second can be; a `prompt_sha256` drop only shrinks `n_paired` below 24, which raises
1/`n_paired` above 0.0417 and binds the clause less still. **The retrieval decision at this n is
therefore carried by the Holm-adjusted McNemar alone.** The margin would bind only at ≥ 34
paired keys, which no pod in this design runs; it is kept because it is the rule, and because a
later design at a larger n would read it.

**What Holm can resolve at m = 3, n = 24.** The exact two-sided McNemar p at *a* discordant
pairs one way and *b* the other is `2·P(Bin(a + b, ½) ≤ min(a, b))`; at b = 0 that is
`2^(1−a)`. Holm's first slot in a 3-member family is 0.05/3 = 0.0167, so **a task is lost only
at (≥ 7, 0)** — seven of twenty-four keys lost with none won: (7, 0) is p = 0.0156, Holm-adjusted
**0.047** < 0.05. **(6, 0) is not**: p = 0.0313, Holm **0.094**. This is the same arithmetic
`prereg/gate1_tracker_swap_v2.md` §6 does at m = 16 and `prereg/bf16_gist.md` §4 at m = 8, and
it is stated for the same reason: **a table of non-separations is the resolution this row
bought, not evidence of equivalence.**

Perplexity: the paired-window TOST at ±0.02 bits/token (`stats.tost`, α = 0.05) of the post arm
against the r64 arm — on the 32 paired per-window bits/token values
(`nll_sum_nats / (ntok · ln 2)` from `pplw.jsonl`, never the pooled `ppl=` number), with
**d = r64 − post**, so `d < 0` means the post arm is worse — and descriptively against `full`.
**The TOST is reported with its effect size, never alone:** `stats.paired_bootstrap(d)`'s **mean
difference and 95 % CI** over those same 32 windows are printed beside it, as
`prereg/gate1_tracker_swap_v2.md` §4 and `prereg/bf16_gist.md` §4 print theirs. A pass at mean
d = −0.019 bits and a pass at mean d = −0.001 bits are not the same result, and only the
interval says which one happened.

**Non-inferior** = no task lost AND the TOST passes. **Refusals:** any error row on either arm
in a paired cell; a broken pairing; a missing Stage-1 record; a TOST that is not decidable
(`stats.tost_decidable`). **The error-row refusal is deliberately wider than "a paired cell."**
§10's reading scans every row of an arm, `vt` and any key with no partner included, because
over-refusal is the safe direction. Their consequence is §11's `REFUSED`, which is neither a
pass nor a fail. No `--`: an arm that did not run reads `not run` with the reason.

## 5. Predictions

Perplexity: within ±0.02 bits of r64 (the ADR's "roughly halves the error at matched rank"
predicts r128 post ≈ r64 pre on reconstruction). Retrieval: not predicted in either direction
— the surprise selection against a post-RoPE basis is the ADR's open risk; `niah_multivalue`
is where a change would show first (the Week-12 mechanism sits in the exact tier).

## 6. Family size

One Holm family of m = 3 (the three non-`vt` retrieval tasks); the TOST is an
intersection-union test at α = 0.05; nothing else is corrected. `vt` is descriptive.

## 7. Secondary outcomes

The stored-bits ratio of both arms; per-task raw accuracies with Wilson intervals
(`stats.wilson`), `vt` among them; the perplexity axis's **mean difference and 95 % bootstrap
CI** from `stats.paired_bootstrap(d)` (§4), reported whether or not the TOST passes; the
`[diag]` rows (guard repairs under the post basis); the Stage-1 `full` rows beside both.

## 8. Log volume

128 samples: 96 `[trial]` lines, 4 cell rows, one 32-window `[pplw]` group (4 `part=i/N`
fragments), one `ppl=` line, ≈ 416 `[diag]` rows per 16K sample
(`prereg/gate1_tracker_swap_v2.md` §8: 13 rows per layer × 32 layers) — 416 × 128 = **53,248**
rows over the run, printed in per-sample bursts of 416 — plus ≈ 40 `[stage]` lines.

**The binding constraint is the 5,000-line fallback, not the 30,000-line poll.**
`scripts/pod/watchdog.sh:65-69` asks for `vastai logs --tail 30000` and falls back to
`--tail 5000` when that fetch comes back **empty** (the Week-20 failure mode), so 5,000 is the
window a poll can actually get. A row is lost only to a poll-to-poll gap — more matched lines
printed between two 150 s polls than the window holds — and at this arm's 5.0 min = 300 s per
sample **at most one** sample can close inside one poll: **≤ 416 + 1 rows**, which clears the
5,000-line fallback **12×** and the 30,000-line window 72×. The unfiltered log would have to add
> 4,500 lines in 150 s on top of that burst to open a gap.

## 9. Budget

128 samples at ≤ 5.0 min/sample (the r128 rate bounded above by the measured r256 rate of 5.2
min, D-011 addendum 2, and below by r64's 3.3 min, addendum 10) = 640 min = 10.7 h, + 60 min
boot = **11.7 h point; `gpu_budget_h: 24.0`**, the **2.05× bar** `pod.py launch --max-hours`
enforces (24.0 / 11.67 = 2.057). At $0.45–0.74/h: $5.3–8.7 expected, $10.8–17.8 at the bar.
Overrun is a stop-and-report.

**That bracket mixes two kinds of rate, and the mixing is stated rather than hidden.** 5.2 min
is an r256 *perplexity* rate (D-011 addendum 2) and 3.3 min an r64 *retrieval* rate (addendum
10), so the bracket is not an interpolation between two comparable measurements. Its direction
is the conservative one — this pod's 96 retrieval samples are billed at a rate whose upper
anchor came from a heavier per-sample workload at twice the rank — and the number that re-sizes
any later r128 pod is this pod's own first `cell_elapsed_s`, not this bracket.

The three-arm alternative (repeating `full` and `isvd_r64_h256_seed` in-pod) is rejected on
cost: **at sizing rates (`full` 0.6 and r64 3.1 from Gate-1 §9; r128 5.0 is this file's
bracket), 128 × 8.7 = 1,114 min = 18.6 h of compute, + 1 h boot, ×2 ≈ 39 h at the bar,
≈ $18–29** — for readings Stage 1 already buys. At the *measured* rates instead (`full`
3.2–3.6 s/sample and r64 3.3 min, D-011 addendum 10; r128 still the unmeasured 5.0) the same
alternative is 128 × 8.36 = 1,070 min = 17.8 h + 1 h boot, ×2 ≈ **38 h**: the conclusion is
unchanged at either rate set.

## 10. Provenance

Pod `configs/pods/postrope_r128.yaml` (this file as `prereg`, one arm, the two Gate-1 task
files, bar 24.0), arm `configs/arms/isvd_postrope_r128.yaml`, the knob in
`src/kvdlra/cache/bug_cache.py` (`rope_basis`), pins `tests/test_rope_basis.py` and
`tests/test_pod_manifest.py`. Outputs `results/postrope_r128/{manifest.json, trials.jsonl,
ppl.jsonl, pplw.jsonl, diag.jsonl, env.txt}`.

### The reading, as code

**It is not `scripts/tables.py gate1`.** `gate1.load` refuses a second pod of the same model
family (`src/kvdlra/eval/gate1.py:372`) and its `_tracker` raises on an arm absent from
`ARM_TRACKER` (`:429`), which `isvd_postrope_r128` is — and neither is a defect to repair here:
this is a one-row cross-pod reading, not a Gate-1 family, so nothing is gained by teaching the
Gate-1 renderer about it. §4 is therefore pre-registered as the snippet that produces it over
shipped, tested functions (`kvdlra.eval.records`, `kvdlra.eval.stats`; `tests/test_records.py`,
`tests/test_stats.py`) — the pattern of `prereg/bf16_gist.md` §4, with the cross-pod
`prompt_sha256` check of `prereg/gate1_preflight.md` A1.4. **No code change is pre-registered
here and none is needed.** A `postrope` renderer, if a later lane wants one, is that lane's and
changes no number below.

```python
from pathlib import Path

from kvdlra.eval.config import load_arm
from kvdlra.eval.records import paired_window_bits, read_jsonl, window_bits
from kvdlra.eval.stats import holm, mcnemar_exact, paired_bootstrap, tost, tost_decidable

# record keys are `legacy_name or name` (kvdlra.eval.frontier.build_arm), not the config name
A = load_arm("isvd_r64_h256_seed").legacy_name or "isvd_r64_h256_seed"
B = "isvd_postrope_r128"   # a = the shipped arm, b = this pod's (no legacy_name)
CTX, CORPUS = 16384, "pg19-val"
TASKS = ("niah_single", "niah_multikey", "niah_multivalue")  # the Holm family; vt is descriptive
DIRS = (Path("results/gate1_v2_stage1_llama"), Path("results/postrope_r128"))

# --- Retrieval: one row per (arm, task, seed, trial) across the two pods --------------------
rows: dict[tuple[str, str], dict[tuple[int, int], dict]] = {}
for d in DIRS:
    for r in read_jsonl(d / "trials.jsonl"):
        if r["arm"] in (A, B) and r["ctx"] == CTX:
            rows.setdefault((r["arm"], r["task"]), {})[r["seed"], r["trial"]] = r

# Refusals are read BEFORE the rule (section 4): an error row on either arm in a paired cell.
# Deliberately wider than "a paired cell": every row of an arm, `vt` and unpaired keys included
# -- over-refusal is the safe direction.
errors = {(arm, t): [k for k, r in rows.get((arm, t), {}).items() if r.get("error")]
          for arm in (A, B) for t in (*TASKS, "vt")}

refused: list[str] = []
if any(errors.values()):
    refused.append("error row")

members, dropped, missing_stage1, missing_postrope = {}, {}, {}, {}
for t in TASKS:
    a_rows, b_rows = rows.get((A, t), {}), rows.get((B, t), {})
    only_b = sorted(set(b_rows) - set(a_rows))    # post keys with no Stage-1 partner
    only_a = sorted(set(a_rows) - set(b_rows))    # Stage-1 keys with no post partner
    if only_b:
        missing_stage1[t] = only_b     # reported with the key; never silently (section 3)
    if only_a:
        missing_postrope[t] = only_a
    shared = set(a_rows) & set(b_rows)
    bad = sorted(k for k in shared                     # section 3's cross-pod pairing rule
                 if a_rows[k].get("prompt_sha256") is None
                 or a_rows[k]["prompt_sha256"] != b_rows[k].get("prompt_sha256"))
    if bad:
        dropped[t] = bad      # reported with the key; it shrinks n_paired, never silently
    keep = shared - set(bad)
    # a = the r64 arm, b = the post arm, so `a_favored` counts the keys the POST arm LOST.
    members[t] = mcnemar_exact({k: int(a_rows[k]["hit"]) for k in keep},
                               {k: int(b_rows[k]["hit"]) for k in keep})

if any(m is None for m in members.values()):
    refused.append("no shared key")
if missing_stage1:
    refused.append("missing Stage-1 record")
if missing_postrope:
    refused.append("missing postrope record")

# REFUSED (section 11) is a state collected as reasons, not a bool: an error row, a broken
# pairing (`mcnemar_exact` -> None, no shared key), a missing Stage-1/postrope record, or
# (below, once the perplexity axis runs) a TOST that is not decidable.
p_holm = {} if refused else dict(zip(TASKS, holm([members[t]["p_value"] for t in TASKS])))
lost = {t for t in p_holm      # section 4's two conditions; at n = 24 the Holm term is binding
        if (members[t]["a_favored"] - members[t]["b_favored"]) / members[t]["n_paired"] > 0.03
        and p_holm[t] < 0.05}

print("dropped (prompt_sha256 disagreed):", dropped or "none")
print("missing Stage-1 record:", missing_stage1 or "none")
print("missing postrope record:", missing_postrope or "none")
print("error rows:", {k: v for k, v in errors.items() if v} or "none")
for t in TASKS:
    m = members[t]
    print(t, "REFUSED (no shared key)" if m is None else
          f"n_paired={m['n_paired']} a_favored={m['a_favored']} b_favored={m['b_favored']}"
          f" p={m['p_value']:.4f}" + ("" if refused else f" holm={p_holm[t]:.4f}"))
print("lost:", sorted(lost) or "none")

# --- Perplexity: the 32 paired windows' bits/token, never the pooled `ppl=` number ----------
# `window_bits` refuses a window scored twice; `paired_window_bits` refuses a window set that
# is not exactly Stage 1's.
bits = window_bits([r for d in DIRS for r in read_jsonl(d / "pplw.jsonl")],
                   key=lambda r: (r["arm"], r["ctx"], r["corpus"]),
                   where="postrope_r128 x gate1_v2_stage1_llama")
d_bits = paired_window_bits(bits[A, CTX, CORPUS], bits[B, CTX, CORPUS], f"{A} @ {CTX}", B)
p_lo, p_hi, equivalent = tost(d_bits, 0.02)      # d = r64 - post; d < 0 means post is worse
decidable = tost_decidable(d_bits, 0.02)  # not decidable is REFUSED (section 11), never fail
if not decidable:
    refused.append("TOST not decidable")
mean_d, ci_lo, ci_hi = paired_bootstrap(d_bits)  # the effect size, printed beside the TOST
print(f"ppl: n={len(d_bits)} mean_d={mean_d:+.4f} bits/token, 95% CI"
      f" [{ci_lo:+.4f}, {ci_hi:+.4f}], TOST p_lo={p_lo:.3g} p_hi={p_hi:.3g}"
      f" equivalent={equivalent} decidable={decidable}")
print("VERDICT:", f"REFUSED — {', '.join(refused)}" if refused
      else ("non-inferior" if not lost and equivalent else "fail"))
```

Sign conventions, fixed now: `mcnemar_exact(a=r64, b=post)` → `a_favored` is the number of
`(seed, trial)` keys the r64 arm hit and the post arm missed, so `a_favored` is what §4's raw
loss counts; `d = bits(r64) − bits(post)` per window, so **`d < 0` means the post arm is worse**.

**The snippet is exercised before the launch, not after the harvest.** Run against **two
dry-run-shaped directories** — the shape `scripts/pod.py run --pod <name> --dry-run` leaves
(`manifest.json`, `env.txt`, an empty `trials.jsonl`), filled with hand-written rows, one row
per arm per key and per window — it must print a reading for a clean pair; drop and report a
`prompt_sha256` disagreement with its key; suppress Holm when an `error` row is present; and
raise on a window set that is not exactly Stage 1's. A snippet that does not run is not a
pre-registration, and this one is run before the pod is.

### Launch

`scripts/pod.py launch --pod postrope_r128 --offer <id>` from a pushed clean SHA that descends
from this file's first commit AND from the commit that adds
`results/gate1_v2_stage1_llama/trials.jsonl`; a DECISIONS line under D-011 with SHAs, offer,
rate, bar and credit. The launch entry also names the Stage-1 Llama harvest commit this pod
is paired against (the commit that added `results/gate1_v2_stage1_llama/trials.jsonl`).
Amendments only, never edits.

## 11. What this does not decide

Whether (iii) is the kernel (that is D-002 and `prereg/kernel_smoke.md`); anything at 32K, on
Qwen or Mistral, or at any rank but 128; the kernel's cost model. A pass makes (i) a viable
simplification at 2× the coordinate bytes for D-002 to weigh; a fail retires (i).

**A pass does not move the operating point, and the ADR says so before the run.** At r = 128 the
post-RoPE row stores **0.282× of the fp16 cache at 16K and 0.267× at 32K** (ADR §3) against
r64's 0.151× / 0.139×, so **a pass still fails Gate 3's 0.25× resident-KV criterion at 32K,
where r = 64 passes** — ADR §3's "even a pass costs 2× the coordinate bytes … failing Gate 3's
0.25× criterion where r=64 passes", against `docs/plan/ICML2027_PLAN.md` §2 Gate 3's "resident
KV ≤ 0.25× full at 32K". That criterion is read on the **stored** ratio, exactly as ADR §3's own
convention does — `resident KV = stored gist + tiers`, billed via
`accounting.bug_footprint(...).stored_bits()` — because no kernel exists yet to measure a
runtime residency. In ADR §4's wording, winning "buys a simpler kernel, not a better
operating point": (i) is insurance if (iii) misses its FLOP target on some GPU, and nothing this
row can show makes it the shipped configuration.

**`REFUSED` is a third overall state, not a kind of fail** (`prereg/bf16_gist.md` §4 (3)'s
pattern). The reading is `REFUSED` whenever any of §4's refusals fires — an `error` row on
either arm in a paired cell, a broken pairing (`mcnemar_exact` returns `None`: no shared key), a
member with nothing left to pair after the `prompt_sha256` drops, a missing Stage-1 record, or a
TOST that is `not decidable` — whatever the members that did read show. The three states compose
as: **pass** iff no task is lost and the TOST passes; **fail** iff that conjunction is broken and
nothing is refused; **`REFUSED`** otherwise. Its consequence is narrower than a fail's: **no
reading and no cost figure.** A fail reports what the post basis costs — the failing member, its
effect size, its interval, the axis — beside the 0.151× → 0.282× the rank buys; a refusal has
measured nothing to report a cost from, so it reports none. (i) stays exactly where ADR §4 left
it, and the cause is named in `docs/plan/DECISIONS.md` and repaired **by an amendment to this
file, committed before any relaunch** — never by a knob changed on a re-run of the same pod.
