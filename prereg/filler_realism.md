# prereg — filler-realism diagnostic (`filler-llama`)

> **Amended 2026-09-19 — read [Amendment 1](#amendment-1-2026-09-19-before-the-launch-commit) at the end of this file before reading anything below it: the run moves from the deleted `w18_boot.sh`/`w21.sh` driver to `configs/pods/filler_realism.yaml` + `scripts/pod.py launch`, a trial that raises is recorded as an `error` row rather than dropped, the budget is re-derived at the rates the Table-4 launches measured, a harness-consistency control pod is added, and the `STATUS:` line at the bottom of the original text is superseded by the dated status line that closes Amendment 1. Everything between this banner and that original `STATUS:` line is left exactly as it was written on 2026-09-11.**

Lane L2 item 1 · gate G2 first checkbox · DECISIONS `D-005`.
Committed **before** the launch commit. Nothing here has been run.

## Purpose

The in-house RULER generator builds its haystack by cycling **ten fixed sentences**
(`scripts/w4_needle.py:46-57`, selected by `scripts/w10_ruler.py:100`) until the context
is full — at 16 384 tokens that is roughly 1 400 repetitions of the same ten strings. A
haystack with ten distinct rows is close to rank-deficient by construction, which is
precisely the structure a rank-64 gist is best at absorbing, so every in-house retrieval
number this project has published is open to the reading that the generator, not the
method, produced it. Two facts sharpen the suspicion into a testable one. First, the
same cache configuration that scores 1.00 / 1.00 / 1.00 on the in-house needles scores
**0.00 on all nine official RULER tasks** in its q4 cell (`results/w20-close-report.md`
§4), on Paul-Graham-essay haystacks — real prose. Second, the r64 configuration itself
drops from 1.00 in-house to a mean of 0.79 on the same official anchor. That gap has two
candidate explanations — template/task-semantics differences, or haystack realism — and
nothing in the record separates them, because the one previous real-text run
(`w18.sh` MODE `g2`) was Qwen-only and ran the r64 arm on `niah_single` alone while its
baselines ran all four tasks. This diagnostic changes exactly one variable — the filler —
and holds model, context, tasks, needle placement, n, seeds and arm flags fixed at the
values that produced the committed cycled-filler rows.

## Design

| field | value |
|---|---|
| model | `meta-llama/Llama-3.1-8B-Instruct` (`TAG=llama`, `DTYPE=bfloat16`) |
| context | 16 384 |
| tasks | `niah_single niah_multikey niah_multivalue vt` — **all arms on all four, in one call per arm** |
| n | 12 per (arm, task) = `--n-trials 6 --seeds 0 1` |
| filler | `--filler wikitext` (see *Filler source* below) |
| needle depths | generator defaults — **`--depths` is not passed** |
| prefill | `--chunk 4096` (pod default), except the single-shot arm |
| perplexity | **not run** (see *No perplexity line* below) |

All four tasks go through a **single** `w10_ruler.py` call per arm. This is not
cosmetic: `scripts/w10_ruler.py:390` sets `max_new = 12 if args.tasks == ["niah_single"]
else 40`, so a one-task call decodes a different number of tokens than a four-task call.
Every cycled-filler reference row below came from a four-task call. `w18.sh` MODE `g2`
did not — it ran the r64 arm as `--tasks niah_single --depths 0.1 0.3 0.5 0.7 0.9` while
its baselines ran all four tasks with no depth grid, so its two halves are comparable
neither to each other nor to the archive. That asymmetry is not repeated here.

### Filler source

`--filler` accepts `cycle | wikitext | wikitext-103 | pg19` (`scripts/w10_ruler.py:632-638`),
resolved by `perplexity_sweep.load_corpus_sentences` (`scripts/perplexity_sweep.py:147-187`).
This diagnostic uses **`wikitext` = the WikiText-2 *test* split**, split on terminal
punctuation into a pool of up to 20 000 sentences (≥ 20 chars, ≥ 2 words), shuffled per
`(seed, trial)` by `random.Random(seed*131 + trial)` and laid down until the context is
full (`scripts/w10_ruler.py:96-111`). At 16 K roughly 650 distinct sentences are drawn
from a pool of thousands, so no sentence repeats inside one haystack and every trial sees
a different one.

**Deviation from the L2 brief, stated.** The brief names a *PG-19 chapter*. Two reasons
that is not what runs today, and why the substitution is acceptable:

1. **No code path produces a chapter.** `_filler_cached` shuffles a sentence pool and
   cycles it. `--filler pg19` exists, but it too yields a shuffled bag of PG-19 sentences,
   not contiguous prose. Contiguous-document filler is Generator v2 (L2 item 2), not a
   flag that exists now; pre-registering "PG-19" today would describe a run the harness
   cannot perform.
2. **`pg19` carries avoidable launch risk.** It streams `deepmind/pg19` through the
   legacy dataset-script loader with `trust_remote_code=True`, which its own docstring
   calls "fragile on a fresh pod" (`scripts/perplexity_sweep.py:104-106`). `wikitext` is
   a plain parquet download and is the one real-text path already exercised on GPU
   (`w18.sh` `g2`, `results/w18-g2-qwen-lines.txt`).

The hypothesis under test is *rank deficiency of the haystack*, and a shuffled WikiText-2
pool falsifies it as sharply as a chapter would: it removes the ten-row structure. What
it does **not** test is long-range discourse coherence, which only a contiguous document
supplies. That limit is the reason this run is labelled a diagnostic and Generator v2
still has to ship; it is not a reason to delay the diagnostic.

### Arms (5)

| # | arm name in the rows | flags added to the common line |
|---|---|---|
| 1 | `full` | `--methods full` |
| 2 | `bugSseed-r64-h256` | `--methods bugslash --ranks 64 --hh-budgets 256 --hh-neighbor 1 --warmup-seed` |
| 3 | `bugSseed-r64-h256-q4` | arm 2 + `--bug-quant-bits 4 --bug-quant-budget 512` |
| 4 | `quant-2bit-kivi` (chunked) | `--methods quant --quant-scheme kivi --quant-nbits 2` (inherits `--chunk 4096`) |
| 5 | `quant-2bit-kivi` (single-shot) | arm 4 + `--chunk 0` |

Arms 2–5 are flag-identical to the runs that produced their cycled-filler references
(`w18.sh` `g1`, `w19.sh` `a1q`, `w19.sh` `a1`, `w19.sh` `sysfix`/ss2 respectively). Arm 1
is the ceiling control: an uncompressed model on the new haystack. If `full` also falls,
the tasks got harder for every method and no drop is attributable to compression.

Ordering inside the MODE is cheap-control-first, then the two arms the decision rule
names, then the baselines, so a pod that dies early still lands an interpretable result:
`full` → `bugSseed-r64-h256` → `bugSseed-r64-h256-q4` → KIVI-2 chunked → KIVI-2 single-shot.

### No perplexity line

`w18.sh` `g1` and `w19.sh` `a1q`/`swap` each ran a same-pod `PPL4` line. This one does
not, deliberately: `scripts/w10_frontier.py` has no `--filler` argument at all —
perplexity is scored over a separate corpus sweep and never touches the RULER haystack —
so a perplexity line here would re-measure the committed 16 K value (5.31 / 9.25) at
about an hour of extra pod time and tell us nothing about filler realism.

### Control considered and skipped

A sixth cell — `full` on the **cycled** filler — was considered, because no committed
in-house four-task `full` row exists for Llama-3.1-8B at 16 K (`results/w11-goalA-ruler-lines.txt`
is a different model and predates `sbits=`). It is skipped: arm 1 measures the ceiling
directly on the new filler, which is the quantity the reading needs. If arm 1 comes back
below 1.00 the cycled `full` cell becomes worth about $1 to buy, and this prereg is
amended before it is run rather than after.

## Reference rows (cycled filler, Llama-3.1-8B, 16 K, n=12)

| arm | single | multikey | multivalue | vt | source (aggregate row / per-trial) |
|---|---|---|---|---|---|
| `bugSseed-r64-h256` | 1.00 | 1.00 | 1.00 | 0.58 | `results/w18-llama-lines.txt` · `results/w18_pertrial/llama-trials.txt` (W18 `g1`) |
| `bugSseed-r64-h256-q4` | 1.00 | 1.00 | 1.00 | 0.50 | `results/w19-a1q-llama-lines.txt` · `results/w19_pertrial/a1q-llama-trials.txt` (W19 `a1q`) |
| `quant-2bit-kivi` chunked | 1.00 | 0.67 | 0.42 | 0.67 | `results/w19-a1-llama-lines.txt` · `results/w19_pertrial/a1-llama-trials.txt` (W19 `a1`) |
| `quant-2bit-kivi` single-shot | 1.00 | 1.00 | 0.83 | 0.67 | `results/w19-sysfix-llama-lines.txt` · `results/w19_pertrial/sysfix-llama-trials.txt` (W19 `sysfix`) |
| `full` | — | — | — | — | no committed in-house Llama-8B row (see *Control considered and skipped*) |

The first, third and fourth rows are also tabulated in `results/w20-close-report.md` §1
and §3.

## Primary contrast

For each arm and task, the **drop** `Δ = p_cycled − p_real`, where `p_cycled` is the
reference row above (12 samples) and `p_real` is this run's row (12 samples).

**Pairing on `(seed, trial)` is not available across fillers, and the analysis does not
claim it.** The two runs share the *question item*: `build_task` derives the code,
the queried key index (`qi = trial % n_keys`), the multivalue label and the vt variable
chain from `(seed, trial)` alone (`scripts/w10_ruler.py:174-246`). But the stimulus is a
different text, and needle placement is by **sentence index** (`int((k+1)/(n_keys+1) * n)`),
so with longer real sentences the same trial index puts the needle at a different token
offset inside a different document. A McNemar or paired-permutation test assumes the same
item under two treatments; here only half the item is the same. The two rows are therefore
treated as **independent samples** and the test is the **unpaired difference of two
proportions, n=12 vs n=12** — Wilson intervals per cell and a Newcombe interval on the
difference, Fisher's exact for a p-value. Resolution is coarse and stated up front:
1/12 = 0.083 per flipped trial, so the 0.25 threshold is exactly **three flipped trials**.

Both rows come from the same pod family, model revision and SHA-pinned harness, so the
independence assumption is about the haystack draw, not about drifting infrastructure.

## Decision rule

> "if the r64 config or the q4 cell drops by more than 0.25 on any task, the cycled-filler
> generator is retired from every headline claim (DECISIONS.md entry) and all v1 in-house
> tables are marked 'diagnostic only' in the paper"

The rule fires on **point estimates**, exactly as the L2 brief writes it — no significance
test gates it. The multiplicity correction below is reported alongside as supporting
evidence and does not modify the rule.

**Family size = 2 arms × 4 tasks = 8** (`bugSseed-r64-h256` and `bugSseed-r64-h256-q4`,
four tasks each). Holm–Bonferroni over those 8 Fisher p-values, reported as a secondary
column. The two KIVI arms and `full` are reported descriptively and are not in the family;
they exist to say whether any drop is specific to the gist or general to the benchmark.

## GPU budget

**Unit, and the assumption behind it.** "min/trial" = wall-clock for one
`(arm, task, seed, trial)` sample. A **cell** is one `(arm, task, ctx)` at n=12, so a
four-task call is **four** cells. This is the correction for the Week-14 sizing miss,
which was about 4× low precisely because it costed a four-task call as one cell
(`docs/w14-sizing.md`; memory note "SIZING MISS (~4×, omitted tasks/cell)").

Measured anchors: `docs/week11-session-handover.md:87-89` — A100, r32@16K ≈ 2 min/trial,
r128@16K ≈ 5 min/trial. `docs/week19-kickoff.md:153` — "16K streaming RULER cell ≈ 20–30
min"; `scripts/pod/w19.sh` (a1q header) — "Streaming r64: ~25 min per 16K cell", i.e.
25/12 ≈ 2.1 min/trial, which sits between the r32 and r128 anchors as it should. Rounded
**up** to 3.0 for the r64 arm, per the Week-14 lesson.

48 samples per arm (4 tasks × 12):

| # | arm | min/sample | × 48 | minutes |
|---|---|---|---|---|
| 1 | `full` | 1.0 (uncompressed; fastest path, 25.9 ms/tok decode) | | 48 |
| 2 | `bugSseed-r64-h256` | 3.0 (2.1 measured, rounded up) | | 144 |
| 3 | `bugSseed-r64-h256-q4` | 3.5 (arm 2 + coordinate-tier quant/dequant) | | 168 |
| 4 | `quant-2bit-kivi` chunked | 1.5 (43.7 vs 25.9 ms/tok decode ≈ 1.7× full) | | 72 |
| 5 | `quant-2bit-kivi` single-shot | 1.5 | | 72 |
| | **compute subtotal** | | | **504 min = 8.40 h** |

Two additions the cycled path does not pay:

- **Haystack construction.** With `cycle` there is one haystack per context and
  `_filler_cached` builds it once. With a real corpus there are 12 — one per
  `(seed, trial)` — memoized per process and shared across tasks and arms inside that
  process, rebuilt in each of the 5 processes. The builder is O(n²) in tokenizer calls
  (`docs/week19-handover.md:59`: ≈ 5 min/trial at 64 K, pre-memoization), so at 16 K
  ≈ 5 × (16/64)² ≈ 0.31 min per haystack: 5 × 12 × 0.31 ≈ **19 min**.
- **Pod overhead** — boot, clone, `pip`, 8 B weight download ≈ **25 min**
  (`docs/w14-sizing.md` used ~20).

**Total wall = 504 + 19 + 25 = 548 min = 9.13 h.**
**With the 2× safety factor: 18.3 h.**

Rate: A100 PCIe 40 GB ≈ **$0.60–1.10/hr** (`docs/week19-kickoff.md:132`).

| | hours | × $0.60 | × $1.10 |
|---|---|---|---|
| point estimate | 9.13 | $5.48 | $10.04 |
| **2× ceiling** | 18.3 | **$10.98** | **$20.13** |

**Budget requested: $20 ceiling; expected $6–10.** Credit is $23.99
(`docs/week19-handover.md:202`), so this consumes most of it at the ceiling. Note for the
owner: D-005 currently scopes this as "~2–4 A100-hours ≈ $3–6"; that figure omits
tasks/cell and is low by roughly 3–4× — the same arithmetic error as Week 14. The
ordering above is the mitigation: arms 1–3 (360 min ≈ 6 h, ≈ $4–7) answer the decision
rule on their own, and arms 4–5 are the descriptive baselines that can be dropped if the
credit floor is hit.

Watchdog credit floor: `FLOOR=6.0` as in `scripts/pod/w19_watchdog.sh`.

## Provenance

- **Pod label `filler-llama`**, `pods.txt` line `filler-llama:<instance-id>:filler:llama`
  (`<mode>-<tag>`, the Week-19 convention; `results/w19_harvest/pods.txt`).
- Launched by `scripts/pod/w18_boot.sh` with
  `MODE=filler MODEL=meta-llama/Llama-3.1-8B-Instruct TAG=llama SHA=<launch-sha> DRIVER=scripts/pod/w21.sh`,
  image `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` (the quant arms need `nvcc` for
  optimum-quanto's kernel).
- **This file's commit SHA must be an ancestor of the launch commit SHA.** The launch
  commit is the SHA baked into the `--onstart` copy of `w18_boot.sh`; the pod echoes it
  back as `===RUN_SHA_<sha>===`, and that value must match.
- Harvest into **`results/filler_realism/`**: aggregate rows, `[trial]` per-trial lines,
  and the `===ENV_BEGIN===` block (run SHA, `nvidia-smi`, torch/CUDA/transformers). The
  existing watchdog writes `results/w19-filler-llama-lines.txt` and
  `results/w19_pertrial/filler-llama-trials.txt`; those are moved into
  `results/filler_realism/` in the harvest commit.
- A trial that raises is logged `SKIP <ExcType>` and **dropped**, so `n=` on the
  aggregate row can be below 12 (`scripts/w10_ruler.py:436-443`). Any cell whose
  harvested `n` is not 12 is reported with its actual n and is excluded from the
  decision rule, because a 0.25 threshold on a reduced denominator is a different test.
- Outcome recorded as a dated `DECISIONS.md` entry closing `D-005`, with the evidence
  path, whatever it shows.

STATUS: awaiting owner go (DECISIONS D-005)

## Amendment 1 (2026-09-19, before the launch commit)

Everything above this heading is the design as it was written on 2026-09-11 and is left untouched.
This amendment re-expresses the *run* of that design on the launch path the L0 lane built after the
prereg was written — the driver the original names (`scripts/pod/w18_boot.sh` + `scripts/pod/w21.sh`)
was deleted on `main` (DECISIONS D-016) — and records what the Table-4 launches of 2026-09-18 taught
about budgets and idle billing (DECISIONS D-011 addenda 2 and 8). It changes no arm, no task, no n and
no reading; A1.6 lists what it leaves alone. It is committed **before** the launch commit, as the
original's provenance section requires, and `scripts/pod.py launch` enforces that order mechanically.

### A1.1 Launch path: `scripts/pod.py launch` → `scripts/pod/boot.sh` → `scripts/pod.py run`

**Launch.** `scripts/pod.py launch --pod filler_realism --offer <id>` and, beside it,
`scripts/pod.py launch --pod filler_realism_cycle --offer <id>` (the control pod of A1.3). Each
builds one `vastai create instance <offer> --image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel
--disk 80 --env "-e POD=<pod> -e SHA=<launch-sha> -e MODEL=unsloth/Meta-Llama-3.1-8B-Instruct
-e DTYPE=bfloat16 -e MAX_HOURS=<gpu_budget_h>" --onstart scripts/pod/boot.sh --label kvdlra-<pod>`
and records it as the manifest's `command_line`. `launch` refuses a dirty tree, a SHA on no remote
branch, a pod naming no prereg, and a prereg whose first commit is not a *strict* ancestor of the
launch SHA. `boot.sh` (the only file uploaded with `--onstart`) clones the repository, checks out
exactly `$SHA` — failing loud with `===CHECKOUT_FAILED_…===` otherwise — and echoes
`===RUN_SHA_<sha>===`; that value must equal the launch commit SHA, and this file's first commit
(2026-09-11) must be a strict ancestor of it. Then it hands off to the committed entrypoint,
`python scripts/pod.py run --pod "$POD"`, which — being cloned at `$SHA` — is itself SHA-pinned; every
knob is in the pod YAML, none on the command line.

**`configs/pods/filler_realism.yaml` ↔ the design table, row by row.**

| design-table row | as written above | where it lives on the pod path |
|---|---|---|
| model | `meta-llama/Llama-3.1-8B-Instruct`, `DTYPE=bfloat16` | `model: "unsloth/Meta-Llama-3.1-8B-Instruct"`, `dtype: bfloat16` — the mirror; see *Model id* below |
| context | 16 384 | `configs/tasks/ruler_inhouse_16k_wikitext.yaml`: `generator: inhouse`, `ctx: 16384` |
| tasks | all four, in one call per arm | `tasks: ["niah_single", "niah_multikey", "niah_multivalue", "vt"]` in that ONE task config, so the runner's `max_new = 12 if task.tasks == ["niah_single"] else 40` (`src/kvdlra/eval/ruler.py`) resolves to 40, exactly as the four-task `w10_ruler.py` call behind every reference row did |
| n | 12 = `--n-trials 6 --seeds 0 1` | `TaskCfg` defaults `n_trials: 6`, `seeds: [0, 1]`; `pod.py check` requires every cell to hold exactly 12 records |
| filler | `--filler wikitext` | `filler: "wikitext"` → `kvdlra.eval.data.load_corpus_sentences("wikitext")`: the WikiText-2 raw **test** split (`Salesforce/wikitext`, `wikitext-2-raw-v1`), up to 20 000 sentences — the function the original cites under its pre-L0 name `perplexity_sweep.load_corpus_sentences`; the per-`(seed, trial)` shuffle and lay-down are unchanged |
| needle depths | `--depths` not passed | `depths: null` (the default) — drawn by the generator |
| prefill | `--chunk 4096`, except the single-shot arm | `chunk: 4096` (the default); the runner takes `chunk = task.chunk if arm.chunkable else 0`, and `kivi2_singleshot.yaml` is `chunkable: false` |
| perplexity | not run | no `ppl` task in `tasks:` |

**The five arms, in the pre-registered order**, and the strings the records carry (`legacy_name` in
each `configs/arms/*.yaml`, which is the `arm` field of every `trials.jsonl` row and the key
`pod.py check` counts cells by):

| # | row key in the original's tables | arm config | `legacy_name` |
|---|---|---|---|
| 1 | `full` | `full` | `full` |
| 2 | `bugSseed-r64-h256` | `isvd_r64_h256_seed` | `bugSseed-r64-h256` |
| 3 | `bugSseed-r64-h256-q4` | `isvd_r64_h256_seed_q4` | `bugSseed-r64-h256-q4` |
| 4 | `quant-2bit-kivi` (chunked) | `kivi2_streaming` | `quant-2bit-kivi` |
| 5 | `quant-2bit-kivi` (single-shot) | `kivi2_singleshot` | `quant-2bit-kivi#chunk0` |

The `#chunk0` suffix is how the L0 configs keep the two KIVI-2 rows apart inside one pod (v1 keyed
them by pod, never by arm name): a record whose `arm` is `quant-2bit-kivi#chunk0` **is the original's
arm 5**, and its reference row is the `quant-2bit-kivi` *single-shot* row (W19 `sysfix`) in the table
above; `quant-2bit-kivi` without the suffix is arm 4 against the *chunked* row (W19 `a1`). The flags:
each config's `cache:`/`quant:` block is the legacy flag set of its row, pinned key for key against
the frozen legacy builder by `tests/test_config_parity.py` (arms 1–4); `kivi2_singleshot` is arm 4's
`quant:` block with `chunk: 0`, pinned by the same file. `tests/test_pod_manifest.py` pins the pod
to this table (arm order, legacy names, task, filler, n, depths, chunk, the one non-chunkable arm).

**Provenance, superseded.** The original's *Provenance* section — pod label `filler-llama`, the
`pods.txt` line `filler-llama:<id>:filler:llama`, `w18_boot.sh` with `MODE=filler … DRIVER=scripts/pod/w21.sh`,
the harvest into `results/w19-filler-llama-lines.txt` and `results/w19_pertrial/filler-llama-trials.txt`
and their move into `results/filler_realism/` — describes a driver that no longer exists and is
superseded as follows. `pod.py launch` writes `results/filler_realism/manifest.json` (git SHA, config
hash, model, library versions, the command line above) and appends
`filler_realism-<instance>:<instance>:filler_realism:Meta-Llama-3.1-8B-Instruct` to
`results/filler_realism/pods.txt`. The watchdog, `scripts/pod/watchdog.sh filler_realism` (credit
floor `FLOOR=6.0` as before), polls the instance log every 150 s, keeps the `[trial]`, `[diag]`,
`[error]`, ENV-block and marker rows, and on the pod's final marker destroys the instance and runs
`scripts/pod.py harvest`, which writes `results/filler_realism/{trials.jsonl,diag.jsonl,env.txt}`
and completes the manifest (`status`, `errors`, `records`, `wall_clock_s`, `harvested_at`). The
same, under `results/filler_realism_cycle/`, for the control pod. **`scripts/pod.py check
results/filler_realism` is the gate**: the config hash re-derived from the YAML must match the
manifest, `git_sha` must resolve and descend strictly from this file's first commit, every one of
the 5 × 4 cells must hold exactly 12 records (errors counted, A1.2), and `env.txt` must sit at the
pyproject pins. A number from this pod is citable only through that gate (CLAUDE.md).

**Model id.** The design table names `meta-llama/Llama-3.1-8B-Instruct`. The repo runs the
`unsloth/Meta-Llama-3.1-8B-Instruct` mirror everywhere — including the pods that produced the
reference rows: `results/paper-v1/w18-llama/cells.jsonl`, `w18-g1-llama/trials.jsonl`,
`w19-a1q-llama/`, `w19-a1-llama/` and `w19-sysfix-llama/` all carry
`"model": "unsloth/Meta-Llama-3.1-8B-Instruct"`. Same weights, no gated download; the pod YAML names
the mirror, and this run and its references are on the same model id.

### A1.2 Error handling, superseded: a raised trial is recorded, never dropped

The original's provenance section describes `w10_ruler.py`'s behaviour — a trial that raises is
logged `SKIP <ExcType>` and dropped, so `n=` on an aggregate row could fall below 12 — and reads any
cell with `n ≠ 12` as excluded from the decision rule. The L0 runner (`kvdlra.eval.runner`) instead
**records** a raised trial as a row with `hit = 0`, `frac = 0.0` and the exception in its `error`
field, so every cell holds exactly 12 rows whatever happened inside it, and `pod.py check` **fails
any pod with one or more error rows** (ruling R29, no tolerance knob). The pre-registered reading
under the new mechanics is the one the `n ≠ 12` clause intended: a cell with error rows is reported
with its error count (`errors` in the manifest; the `error` field per row) and is **excluded from the
decision rule**, because a 0.25 threshold on a denominator that includes rows the arm never answered
is a different test. Such a pod is then citable only by an L6 per-case ruling that names the cell
and the error (the same route the Table-4 prereg pre-registered for an aborting arm); it is not
re-run silently, and the error rows are not filtered out of `trials.jsonl`.

### A1.3 The harness-consistency control pod: `configs/pods/filler_realism_cycle.yaml`

Ruling PR-L2-19 adds a second, cheap pod beside the diagnostic: `isvd_r64_h256_seed` alone on
`ruler_inhouse_16k` — the **cycled** filler, the same four tasks, n = 12, chunk 4096, no depths —
which replicates the archived w18-g1 row on the L0 runner: `bugSseed-r64-h256`, 16K, cycled filler,
**1.00 / 1.00 / 1.00 / 0.58** (single / multikey / multivalue / vt),
`results/paper-v1/w18-llama/cells.jsonl` (the aggregate row, from `results/w18-llama-lines.txt`) and
`results/paper-v1/w18-g1-llama/trials.jsonl` (its 48 per-trial records). Why it is needed: the
reference rows in the table above came from `w10_ruler.py` on an older `transformers`, and the L0
runner's parity with that path rests on CPU bit-identity tests, never on a GPU re-run. Without this
pod, a drop on the real-text pod would have two candidate causes — the filler, or drift between the
harness that produced the references and `pod.py run` — and the original's "same pod family … and
SHA-pinned harness" sentence would no longer hold on its own.

**Pre-registered reading.** Per task, the replication is compared with the archived row as two
independent n = 12 samples, on point estimates, at the same 0.25 resolution (three flipped trials):

- a replication that differs from the archived row by **> 0.25 on any task is harness drift**, and
  the real-text reading of the decision rule is **inconclusive until the drift is explained** —
  the diagnostic is not read against references its own harness cannot reproduce;
- **≤ 0.25 on every task**, and the archived rows stand as the comparison base for the decision rule
  exactly as written.

This pod is a control on the *harness*, not a member of the contrast: it does **not** enter the
Holm family (which stays 2 arms × 4 tasks = 8), and its own row is reported descriptively.
Per-trial pairing on `(seed, trial)` *is* available here — the cycled haystack and the question are
deterministic in `(seed, trial)` — and the flipped trials are listed alongside as descriptive evidence
of where any drift sits; the rule above is still stated on the point estimates.

### A1.4 Budget, re-derived at the measured rates

The per-sample rates in the original's *GPU budget* section stand, with one clarification the
Table-4 launches make necessary: D-011 addendum 2's **5.2 min per 16K sample is the r = 256
perplexity rate** (1024-token windows, one core SVD per absorb per stream at rank 256) and **no r256
arm runs here**. The r64 retrieval rate is the one the original measured and rounded — **2.1 min per
sample on the W19 `a1q` pod, budgeted at 3.0** — and it is kept; the two KIVI rows keep 1.5 and
`full` keeps 1.0 (the compute subtotal is unchanged at 504 min). Haystack construction stays at
≈ 19 min. **Pod overhead is re-budgeted from 25 to 60 min**: the Table-4 Llama pod's anonymous
Hugging Face weight download stalled for ≈ 1 h before it was killed (D-011 addendum 2), and a
budget that assumes a 25-min boot is one that a routine stall alone overruns.

- **Point estimate: 504 + 19 + 60 = 583 min ≈ 9.7 h** (the original: 548 min = 9.13 h).
- **Bar: `gpu_budget_h: 18.3`** — the original's 2× ceiling, kept: it is 2× the original point and
  1.9× the re-derived one; the 35 min of added overhead sits inside the safety factor rather than
  moving the bar. This is the bar in the sense of `prereg/hygiene_table4.md` §9 — a pod still running
  past it is a pod to kill and diagnose — and, new since A1.5, the bar the pod enforces on itself.
- **The control pod (A1.3): 48 samples × 3.0 min = 144 min = 2.4 h of compute** (no haystack
  construction: the cycled filler builds one haystack per context), ≈ 3.4 h with the 60-min pod
  overhead; **bar `gpu_budget_h: 6.0`**.

**Rates.** The original assumed $0.60–1.10/h. The Table-4 launches paid **$0.40–0.74/h for an
A100 40 GB** (D-011 addenda 3–4: $0.40, $0.44, $0.54, $0.60, $0.67, $0.74). At those rates:

| | hours | × $0.40 | × $0.74 |
|---|---|---|---|
| diagnostic, point | 9.7 | $3.9 | $7.2 |
| **diagnostic, bar** | **18.3** | **$7.3** | **$13.5** |
| control, point | 2.4–3.4 | $1.0 | $2.5 |
| **control, bar** | **6.0** | **$2.4** | **$4.4** |

**Ceiling ≈ $7–14 + $2–4; expected ≈ $4–7 + $1–2. Credit: $97.75** (D-011 addendum 8). The
original's "Budget requested: $20 ceiling; expected $6–10 … Credit is $23.99" paragraph is
superseded by this one; its arm ordering (arms 1–3 answer the decision rule on their own) and its
note on D-005's original "$3–6" (low by the tasks-per-cell factor) stand. D-005's addendum option
(b) — the decisive arms only — saves ≈ 2.4 GPU-h ≈ $1.5 at these rates and is not taken (ruling
PR-L2-18): the full five-arm design as pre-registered runs.

### A1.5 Idle billing: the bar is now enforced on the pod itself

D-011 addendum 8: of the Table-4 v3 launch's $20.4, **≈ $12 was idle billing** — the three pods
finished by 13:15 / 14:30 / 17:15 and sat until 22:42, because the launch machine slept and its
watchdogs (under `caffeinate -i`, which prevents idle sleep only) stopped polling. Three changes,
all in the launch path this pod uses (`scripts/pod.py`, `scripts/pod/boot.sh`, `scripts/pod/watchdog.sh`):

1. The watchdog runs under **`caffeinate -s -i scripts/pod/watchdog.sh <pod>`** (`-s` prevents
   system sleep on AC power), detached.
2. **`pod.py launch --max-hours H`** (default: the pod's `gpu_budget_h`, so 18.3 and 6.0 here;
   ≤ 0 is refused — `timeout 0` would disable the bar) passes `-e MAX_HOURS=H` to the instance, and
   `boot.sh` runs the entrypoint under `timeout --signal=TERM --kill-after=60 "${MAX_HOURS}h"`. A run
   that reaches the bar exits 124; the pod prints `===RUN_TIMEOUT_<pod>_<H>h===` and then the same
   `===RUN_FAILED_<pod>_<sha>===` as any failed run — never `ALL_DONE` — so the watchdog destroys it,
   and the harvest records `status: RUN_FAILED` with `timeout: true` in the manifest. The bar of A1.4
   is thus the same number in three places: the prereg, the pod YAML, and the pod's own clock.
3. After its final marker (`ALL_DONE` or `RUN_FAILED`) the pod waits a grace period
   (`GRACE_S`, default 7200 s) and then **destroys itself** with the instance-scoped credentials
   vast.ai injects into every container (`CONTAINER_ID`, `CONTAINER_API_KEY`); if either is unset or
   the call fails it prints `===SELF_DESTRUCT_FAILED_<pod>===` and the watchdog stays the destroyer.
   The watchdog's `BUDGET_ITERS` expiry (25 h) now destroys every instance in `pods.txt` instead of
   merely ending its loop (D-011 addendum 2's open item).

The trade-off of the grace period, stated once: **a watchdog asleep for longer than the grace loses
the rows printed since its last poll — the log dies with the instance — so `caffeinate -s` is the
primary fix and the self-destruct is the cap on idle billing (≈ 2 h ≈ $1 at these rates).**

### A1.6 What this amendment does not change

The design table (model weights, 16 384 context, the four tasks in one call, n = 12 = 6 × [0, 1],
`wikitext` filler, generator-drawn depths, chunk 4096 with the single-shot exception, no
perplexity); the five arms and their order; the filler source and the stated deviation from the
brief's PG-19 chapter; the reference rows; the primary contrast (unpaired, n = 12 vs n = 12, Wilson
and Newcombe intervals, Fisher's exact); the decision rule on point estimates at 0.25; the Holm
family of 8; the *No perplexity line* section; and the skipped cycled-`full` control (the control pod
of A1.3 runs the r64 arm on the cycled filler, not `full`; the cycled-`full` cell is still bought only
if arm 1 comes back below 1.00, by a further amendment before it runs).

STATUS 2026-09-19: launch authorized under D-005/D-011 (owner 2026-09-17); launched by the orchestrator after this amendment's commit — see DECISIONS.md D-005 addendum for SHAs, offers and cost.

### A1.7 Correction (2026-09-19)

A1.4 misattributes all six measured rates to "D-011 addenda 3–4": only $0.54, $0.60, $0.67 and $0.74
are there — $0.40 is D-011's original launch entry (2026-09-18, under the owner's 2026-09-17
authorization) and $0.44 is D-011 addendum 6. The $0.40–0.74/h range and every number A1.4 derives
from it are unchanged.

## Amendment 2 (2026-09-19, before the cycle pod's relaunch; after the real-text pod's `full` arm, before any other arm's cells)

### A2.1 Evidence: the real-text pod's `full` arm (2026-09-19 04:30, instance 51553610, launch SHA 56e89ee)

The `full` arm (uncompressed) on the WikiText filler finished all four cells:
`[niah_single ctx16384] full acc=1.00 n=12` · `[niah_multikey ctx16384] full acc=1.00 n=12` ·
`[niah_multivalue ctx16384] full acc=0.92 recall=0.98 n=12` (seed 0 trial 1: frac 0.75) ·
`[vt ctx16384] full acc=0.08 recall=0.08 n=12` (1/12: seed 1 trial 0 only). The cycle-control pod
(`filler_realism_cycle`, instance 51553637) never left vast.ai's "loading" state in 47 min and was
destroyed without running anything; it is relaunched after this amendment.

### A2.2 Consequence: the cycled-`full` control is bought

The prereg's *Control considered and skipped* paragraph pre-committed to exactly this case: "If arm 1
comes back below 1.00 the cycled `full` cell becomes worth about $1 to buy, and this prereg is amended
before it is run rather than after." `full` fell on `vt` (0.08) and slightly on `niah_multivalue`
(0.92), so the cycled-`full` control is bought — `full` joins `filler_realism_cycle` (arms
`["isvd_r64_h256_seed", "full"]`, cycled filler, same 4 tasks, n=12).

### A2.3 How the reading changes

For a task where `full` on real text is ≥ 0.92 (single, multikey, multivalue) the decision rule of
§"Decision rule" applies unchanged. For `vt`, where `full` itself scores 0.08, no drop of the r64 or
q4 arm is attributable to compression ("If `full` also falls, the tasks got harder for every method
and no drop is attributable to compression" — §Arms), so `vt` is EXCLUDED from the decision rule on
real text and reported descriptively, and the Holm family shrinks from 8 to 6 (2 arms × 3 tasks). The
cycled-`full` cell measures whether the ceiling itself was already below 1.00 on the archived filler —
the archived rows have no `full`.

### A2.4 What this amendment does not change

The arms, order and n of the real-text pod; the ≥ 0.25 point-estimate threshold; the two KIVI arms
reported descriptively.

### A2.5 Budget

+48 `full` samples ≈ 30 min (measured ≈ 0.6 min/sample on 51553610: the full arm's 48 samples took
≈ 30 min wall) → cycle pod bar 6.0 → 7.0 h.

### A2.6 A note for the generator, not yet checked

The `vt` failure mode on real text is itself a finding for the generator: v1 `vt` inserts
`VAR X.. = ..` sentences into WikiText-2 prose whose headings are `= = Title = =` lines that the
sentence splitter keeps. Hypothesis only, to be checked when the records are harvested — the `frac`
per trial is in the log; the decoded answers are not.

STATUS 2026-09-19 (2): cycle pod relaunched with `full` under Amendment 2; real-text pod unchanged.
