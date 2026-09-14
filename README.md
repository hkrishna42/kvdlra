# kvdlra

An online low-rank KV-cache compressor for long-context LLM inference. As the prompt
streams in, each layer keeps a rank-`r` **gist** of the past — a basis tracked by a block
incremental-SVD step, one block of key/value columns at a time — alongside a small
**exact tier** of the tokens the gist reproduces worst (selected by their out-of-subspace
residual, i.e. by surprise), the usual **attention sinks**, and a **recent ring**. A
**warm-up seed** routes the very first chunk through the same surprise selection, scored
against a strictly-older basis, so facts planted at the start of the context are not
structurally lost. Decode reconstructs the middle from the factors and attends over the
result. The repository also carries the baselines it is measured against — token
eviction, KV quantization, low-rank key projection, a per-sequence SVD oracle — so every
method is scored on the state it actually stores, in one unit.

## Install

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/). CI runs 3.12.

```
make env      # uv venv + CPU torch + `uv pip install -e ".[dev]"`
make test     # the CPU suite, < 90 s
```

## Reproduce a table

Every number in the paper's tables is regenerated from committed per-trial records —
never retyped:

```
git clone https://github.com/hkrishna42/kvdlra.git && cd kvdlra
make env
make tables
```

`make tables` writes `docs/paper/tables/` from `results/paper-v1/*/trials.jsonl` and
diffs the result against `docs/plan/paper-v1-tables.md`; it prints `tables: diff-clean`
or fails. `make figures` rebuilds the paper's three figures from the same records, and
`make check` re-verifies every committed results directory.

## Layout

```
src/kvdlra/
  tracker/      the incremental-SVD step (and the Oja / frequent-directions alternatives)
  cache/        the streaming caches: the low-rank cache, ShadowKV
  baselines/    kvpress presses: low-rank, SVD oracle, TurboQuant, the prefill-hook shim
  quant/        PolarQuant and the KIVI-style QuantizedCache mixin
  eval/         arms, tasks, RULER, LongBench, perplexity, persistence, records, stats
  accounting.py stored-state accounting: one float-equivalent unit for every method
configs/
  arms/ tasks/ pods/   one YAML per arm, task and pod — every experiment is a config
prereg/         one `<pod>.md` per pod, committed BEFORE the pod's launch commit
scripts/
  pod.py        launch / run / harvest / check
  tables.py     regenerate the tables from the archived records
  figures.py    regenerate the figures from the same records
  dump_kv.py    dump and sha256-verify a KV tree for offline work
  pod/          the vast.ai bootstrap (`boot.sh`) and watchdog (`watchdog.sh`)
results/
  paper-v1/     archived records + verbatim raw logs behind the published tables
docs/plan/      plan of record, decisions, reviews, reports, cleanup ledgers
paper/          the manuscript
tests/          CPU-only, in CI
```

## Running a pod

A "pod" is one GPU experiment: a `configs/pods/<pod>.yaml` naming its arms, tasks, model
and image, plus a `prereg/<pod>.md` saying what it will run and which outcome would mean
what. The order is enforced, not suggested.

```
# 1. pre-register, and commit it on its own
$EDITOR prereg/mypod.md configs/pods/mypod.yaml
git add prereg/mypod.md configs/pods/mypod.yaml && git commit -m 'prereg: mypod'

# 2. launch (refuses unless the prereg commit is a strict ancestor of HEAD)
git push                                    # the pod clones the exact SHA
scripts/pod.py launch --pod mypod --offer <vast-offer-id> [--dry-run]

# 3. harvest the log into records, then gate it
scripts/pod.py harvest --pod mypod
scripts/pod.py check results/mypod
```

`launch` builds a `vastai create instance` call whose `--onstart` payload is
`scripts/pod/boot.sh`; that script pins the run to the pushed SHA, stamps the
environment into the harvested log, and hands off to `scripts/pod.py run --pod <pod>`.
`scripts/pod/watchdog.sh` polls, harvests and destroys the instance unattended.

On a GPU you already have, skip steps 2 and 3 and run the same loop locally:
`scripts/pod.py run --pod mypod` writes `results/mypod/` directly, and `check` gates it.

`check` is the gate a citable number has to pass. It re-derives the config hash from the
pod's YAML, resolves the SHA, enforces the pre-registration commit order, and requires
every cell the config calls for — arm × generator × sub-task × context — to hold exactly
`n_trials × len(seeds)` records. A trial that raised is recorded as an `error` and still
counted, so a cell can never silently shrink, and any recorded error fails the pod. A pod
that produced nothing cannot be passed off as a clean run.

A run writes `results/<pod>/`: `manifest.json` (git SHA, config hash, model revision,
dataset and haystack digests, library and CUDA versions, GPU, wall clock, command line),
`env.txt`, and the records — `trials.jsonl`, `ppl.jsonl`, `pplw.jsonl`,
`latency.jsonl`, `diag.jsonl`.

## License

Apache-2.0 — see [LICENSE](LICENSE).
