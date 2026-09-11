# prereg — filler-realism diagnostic (`filler-llama`)

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
