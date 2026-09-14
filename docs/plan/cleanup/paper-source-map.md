# Paper source map — where every path `paper/main.tex` cites lives now

`paper/main.tex` cites evidence in two ways: `% source:` comments above a table
or a claim, and `\texttt{results/...}` paths in the body (the reproducibility and
limits paragraphs). Task 9 deleted or moved most of those paths. Task 9 does not
edit `paper/**` — that lane owns it — so this table is the input Phase 3 needs to
rewrite the citations.

Nothing cited was lost. Bytes went one of three ways:

* **archived** — byte-identical copy under `results/paper-v1/<pod>/raw/`
  (verified with `cmp` before the original was removed). `<pod>` is the archive
  naming `scripts/tables.py convert-v1` established;
* **moved** — a narrative report relocated to `docs/plan/reports/` (R17);
* **tag only** — prose with no surviving in-repo equivalent; recoverable at the
  git tag `paper-v1-archive` (`ee8c0ab`), which is what that tag is for.

## `% source:` comments

| line | old path (as cited) | where it is now | how |
|---|---|---|---|
| 436 | `results/w19-swap-llama-lines.txt` | `results/paper-v1/w19-swap-llama/raw/w19-swap-llama-lines.txt` | archived |
| 455 | docs (Week-2), `README` "near-oracle" | — | tag only (`README.md` rewritten in L0.9c; the Week-2 note is at the tag) |
| 475 | `docs/week17-explained.md`, `docs/week16-handover.md` | — | tag only |
| 507 | `results/w18-g1-report.md` | `docs/plan/reports/w18-g1-report.md` | moved |
| 559 | `results/w18-g4-marquee-contrasts.json` | `results/paper-v1/w18-g4-llama/raw/w18-g4-marquee-contrasts.json` | archived |
| 559 | `results/w18_pertrial/g4-llama-trials.txt` | `results/paper-v1/w18-g4-llama/raw/g4-llama-trials.txt` | archived (by `convert-v1`) |
| 559 | `scripts/pod/w18.sh` MODE=g4 (pre-registration) | `configs/pods/w18_g4.yaml` (the arms/tasks the MODE encoded) | folded (L0.9a); the launcher itself is at the tag |
| 598, 612 | `results/w18-g4-llama` (pod) | `results/paper-v1/w18-g4-llama/` | already the archive path |
| 613 | `docs/week11-decision-table.md`, `docs/week16-explained.md` | — | tag only |
| 629 | `docs/week17-explained.md` §3; `src/kvdlra/tracker/isvd.py` `min_sv_frac` | source path unchanged | tag only (docs half) |
| 722 | `results/w19-q4off-llama-lines.txt` | `results/paper-v1/w19-q4off-llama/raw/w19-q4off-llama-lines.txt` | archived |
| 722 | `results/w19_pertrial/q4off-llama-trials.txt` | `results/paper-v1/w19-q4off-llama/raw/q4off-llama-trials.txt` | archived |
| 723 | `results/w20-fork-report.md` | `docs/plan/reports/w20-fork-report.md` | moved |
| 723 | `scripts/w19_fork_report.py` | — | deleted (L0.9a, bucket `scripts-only`); its output is the report above |
| 753 | `results/w19-a1q-{llama,mistral,qwen}-lines.txt` | `results/paper-v1/w19-a1q-{llama,mistral,qwen}/raw/<same basename>` | archived |
| 753 | `results/w18-g1-report.md` | `docs/plan/reports/w18-g1-report.md` | moved |
| 753 | `results/w18-g3-{qwen,mistral,llama}-lines.txt` | `results/paper-v1/w18-g3-{qwen,mistral,llama}/raw/<same basename>` | archived |
| 845 | `results/w19-a1-report.md` | `docs/plan/reports/w19-a1-report.md` | moved |
| 845 | `scripts/w19_a1_report.py` | — | deleted (L0.9a, bucket `none`); its output is the report above |
| 845 | `results/w19_intervals/*.json` | `results/paper-v1/w19-a1-{llama,mistral,qwen}/raw/a1-<model>-ruler-intervals.json`, `results/paper-v1/w19-a2-llama/raw/a2-llama-ruler-intervals.json` | archived |
| 845 | `results/w18_harvest/quant-findings.md` | `docs/plan/reports/quant-findings.md` | moved |
| 938 | `results/w19-a2-llama-lines.txt` | `results/paper-v1/w19-a2-llama/raw/w19-a2-llama-lines.txt` | archived |
| 938 | `results/w19_intervals/a2-llama-ruler-intervals.md` | `results/paper-v1/w19-a2-llama/raw/a2-llama-ruler-intervals.md` | archived |
| 938 | `results/w19-a2-flagship-misses.md` | `docs/plan/reports/w19-a2-flagship-misses.md` | moved |
| 938 | `docs/week19-official-ruler.md` | — | tag only |
| 964 | `results/w18-g2-qwen-lines.txt` | `results/paper-v1/w18-g2-qwen/raw/w18-g2-qwen-lines.txt` | archived |
| 964 | `results/w11-goalA-lb-lines.txt` | `results/paper-v1/w11-goalA-lb/raw/w11-goalA-lb-lines.txt` | archived |
| 983 | `docs/week4.md`, `main.tex(old)`, `README` Week-4 | — | tag only |
| 1018 | `results/w19-a4-llama-lines.txt` | `results/paper-v1/w19-a4-llama/raw/w19-a4-llama-lines.txt` | archived |
| 1056 | `results/w19-sysfix-llama-lines.txt` | `results/paper-v1/w19-sysfix-llama/raw/w19-sysfix-llama-lines.txt` | archived |
| 1117 | `results/w19-a3-llama2-lines.txt` | `results/paper-v1/w19-a3-llama2/raw/w19-a3-llama2-lines.txt` | archived |
| 1118 | `results/w18-g5` (pod) | `results/paper-v1/w18-g5-llama/` | already the archive path |
| 1144 | `docs/week17-explained.md` §2, §1 | — | tag only |

15 of the comments name a `results/` path; every one of those resolves to an
archive path above. The remaining six name prose (`docs/`, `README`) and resolve
to the tag.

## `\texttt{results/...}` in the body

| line | old path (as cited) | where it is now | how |
|---|---|---|---|
| 403, 1187 | `results/w18_pertrial/` | `results/paper-v1/w18-{llama,mistral,qwen,g1-llama,g1-mistral,g1-qwen,g2-qwen,g3-llama,g3-mistral,g3-qwen,g4-llama,g5-llama}/raw/*-trials.txt` | archived per pod |
| 403 | `results/w19_pertrial/` | `results/paper-v1/w19-*/raw/*-trials.txt` | archived per pod |
| 405, 1198 | `results/w18-env-provenance.txt` | `results/paper-v1/_cited-extras/raw/w18-env-provenance.txt` | archived |
| 406, 1198 | `results/w19-env-provenance.txt` | `results/paper-v1/_cited-extras/raw/w19-env-provenance.txt` | archived |
| 663 | `results/w18-g1-report.md` | `docs/plan/reports/w18-g1-report.md` | moved |
| 1188 | `results/w18_harvest/` | `docs/plan/reports/quant-findings.md` (its only narrative); the `DONE.txt` / `G4DONE.txt` / `SUMMARY.txt` markers and the mid-run `results/w18_checkpoint/` snapshots are at the tag | moved + tag only |

## Naming note

`results/paper-v1/_cited-extras/` is the only archive directory without a
`manifest.json`: it holds files `scripts/tables.py convert-v1` never converted
(no records come out of them), so no pod owns them. Its `README.md` says what
each one is. Adding a raw copy to an existing pod directory does not change that
pod's `manifest.json`, which describes the conversion, not a directory listing —
`scripts/pod.py check` skips archive directories (`converted_by`), so `make
check` stays green either way.
