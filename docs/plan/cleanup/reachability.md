# Static reachability analysis -- src/kvdlra/ and scripts/

Read-only static analysis, repo `kv-dlra` @ branch `week7`. Goal: for the
repo cleanup to four entrypoints, classify every file under `src/kvdlra/`
and `scripts/*.py` by whether it is reachable from (R1) a Python invocation
inside `scripts/pod/*.sh`, or (R2) an import in `tests/*.py`, or neither.

**Method.** R1 was built by reading all 22 `scripts/pod/*.sh` files in full
(1951 lines) and cross-checked with `grep -nE 'scripts/[A-Za-z0-9_]+\.py'`
plus a scan for `uv run` / `python -m` / bare `python3` across every file
(none found beyond the R1 set below and inline `python -c`/heredoc
diagnostics that name no repo script). The import closure was built with
Python's `ast` module: every `Import`/`ImportFrom` node reachable via
`ast.walk` (so nested, function-body imports are caught, not just
module-level ones -- this mattered, see §6) over all 92 `.py` files under
`src/kvdlra/` (30) and `scripts/*.py` (62, excluding `scripts/pod/`).
Absolute (`kvdlra.x.y`), submodule-of-package (`from kvdlra import x` where
`x` is itself a file), and relative (`from .foo import bar`) imports are
all resolved; a resolved import also pulls in every ancestor package's
`__init__.py` (real Python import semantics -- `from kvdlra.integrators.bug
import rk2_step` executes `kvdlra/integrators/__init__.py` first even
though nothing names it explicitly). `tests/*.py` (46 files) get `src` and
`scripts` on their import path per `pyproject.toml`'s
`pythonpath = ["src", "scripts"]`, so `import w10_ruler`-style test imports
resolve into the same graph. Grep confirmed **zero relative imports**
anywhere in the repo (`src/kvdlra`, `scripts/`, `tests/`) -- the codebase
uses absolute `from kvdlra.x.y import z` throughout, so the relative-import
resolver path is implemented (per the brief) but empirically unexercised.

Helper script: `scratchpad/reach.py` (ast-based closure) run with
`.venv/bin/python`; output `scratchpad/reach_out.json`. Nothing in the repo
was modified; this file is the only write.

## 1. Roots table (R1 -- every Python invocation in `scripts/pod/*.sh`)

16 distinct `scripts/*.py` files are invoked from `scripts/pod/*.sh`,
always as `PYTHONPATH=src python -u scripts/<file>.py ...` (never
`uv run`, never `python -m`). All 22 pod files were read line-by-line;
inline `python -c "..."` / heredoc diagnostics (dep checks, model-tokenizer
sanity, credit-JSON parsing) name no repo script and are listed under §6.
6 pod files (`scrape_w10.sh`, `w18_finalizer.sh`, `w18_g4fallback.sh`,
`w18_watchdog.sh`, `w18_watchdog2.sh`, `w19_watchdog.sh`) invoke **no**
`scripts/*.py` at all -- they are pure shell (git/vastai/grep/sort) log
harvesters. `w18_boot.sh` is the actual `--onstart` payload: it has no
`scripts/*.py` call of its own but `source`s the MODE driver named by
`$DRIVER` (default `scripts/pod/w18.sh`; `scripts/pod/w19.sh` sets
`DRIVER=scripts/pod/w19.sh` at launch time per its own header comment), so
w18.sh/w19.sh's invocations below run *through* w18_boot.sh, not directly.
Two lines in `w11_r128.sh` (309, 312) are commented out (`#`-prefixed, an
"OPTIONAL RIDER" never wired into any MODE) -- excluded as non-live.

| invoked script | `scripts/pod/*.sh:line` (deduped) | live flags seen (compact) |
|---|---|---|
| `scripts/w10_frontier.py` | w10_gpu.sh:57; w11_baselines.sh:43; w11_gpu.sh:96; w11_r128.sh:59,83,136,177,181,225,229; w11_table.sh:49; w16.sh:44,46,129; w18.sh:22,24; w19.sh:24 | --model --device --dtype --T &lt;ctx...&gt; --chunk --window 512 --n-samples {2,3,4,8} --methods {full,bug,bugslash,bugevict,ea,morph,snapkv,think,palu,shadow,quant} --ranks --hh-budgets --hh-neighbor --evict-keeps --morph-keeps --think-ratios --palu-ranks --shadow-ranks --warmup-seed --score-rank --min-sv-frac --qwhiten-file --quant-nbits --tracker {oja,fd} --no-ruler --out-json |
| `scripts/w10_ruler.py` | w10_gpu.sh:67; w11_baselines.sh:52; w11_gpu.sh:80; w11_r128.sh:51,70,96,111,118,129,148,155,187,193,210,215,236,241,267,272,279,284,291,296; w11_table.sh:40,59; w16.sh:43; w18.sh:20; w19.sh:22 | --model --device --dtype --context-lens --tasks {niah_single,niah_multikey,niah_multivalue,vt} --methods {full,bug,bugslash,bugevict,ea,morph,snapkv,think,palu,shadow,quant,composite} --ranks --hh-budgets --hh-neighbor --hh-discard --evict-keeps --think-ratios --palu-ranks --warmup-seed --score-rank --min-sv-frac --qwhiten-file --quant-nbits --quant-scheme {token,kivi} --quant-backend hqq --quant-axis-value --bug-quant-bits --bug-quant-budget --tracker {oja,fd} --filler wikitext --depths --chunk --n-trials --seeds --out-json |
| `scripts/w10_longbench.py` | w10_gpu.sh:77 | --model --device --dtype --tasks {qasper,multifieldqa_en,hotpotqa,2wikimqa} --max-len 16384 --chunk --n-examples 20 --out-json |
| `scripts/w11_probe.py` | w11_gpu.sh:66; w11_probe8b.sh:37,43; w12_probe_r128.sh:41 | --model --device --dtype --task niah_multikey --ctx --rank {32,128} --hh-budgets {64..2048 or 256 1024} --hh-neighbor {0,1} --chunk --trial --seed --out-json |
| `scripts/w12_calibrate_qkey.py` | w11_r128.sh:170 (MODE=qbug) | --model --n-docs 8 --seq-len 4096 --device cuda --out results/w12-wkey-8b.pt |
| `scripts/w16_tier2_probe.py` | w16.sh:61 (MODE=tier2) | --model --device --dtype --ctx 2048 (gate: stdout must contain "FUND -- authorize GPU") |
| `scripts/w16_storage.py` | w18.sh:112 (MODE=g5) | --model --device --dtype --context-lens 16384 32768 --chunk --methods full bug bugslash --ranks 64 --hh-budgets 256 --warmup-seed --out-json |
| `scripts/w19_official_ruler.py` | w19.sh:92 (OFF(), MODE=a2/fork/q4off) | --model --device --dtype --chunk --data-dir /root/ruler_data --context-len 16384 --tasks &lt;9 official RULER tasks&gt; --methods {full,think,palu,ea,quant,bugslash,composite} + matching sub-flags --out-json |
| `scripts/w19_persist.py` | w19.sh:129 (MODE=a3) | --model --device --dtype --chunk --context-lens 16384 32768 --repeats 5 --tmp /root/persist --methods full bugslash quant --ranks 64 --hh-budgets 256 --hh-neighbor 1 --warmup-seed --quant-nbits 2 --quant-scheme kivi --out-json |
| `scripts/w20_latency.py` | w19.sh:221 (MODE=sysfix) | --model --device --dtype --chunk --context-lens 16384 32768 65536 --methods full bugslash quant --ranks 64 --hh-budgets 256 --hh-neighbor 1 --warmup-seed --quant-nbits 2 --quant-scheme kivi --out-json |
| `scripts/w5_hybrid.py` | w5_confirm_8b.sh:25; w5_hybrid_needle_8b.sh:26 | [legacy, clones branch `week3`] --model --device cuda --dtype bfloat16 --context-lens 1024 8192 32768 --ranks 128 256 --exact-fracs {0.03 0.10 or 0.05} --evict-ratios 0.5 0.7 0.85 --bits 4 --n-windows 3 --out-json |
| `scripts/w5_ruler.py` | w5_confirm_8b.sh:36; w5_fp16_8b.sh:34 | [legacy, `week3`] --model --device cuda --dtype bfloat16 --context-lens {4096 16384 or 32768 65536} --n-keys 12 --n-queries 3 --seeds 0 1 --bits 4 (or --no-quant) --out-json |
| `scripts/w5_fp16_longctx.py` | w5_fp16_8b.sh:25 | [legacy, `week3`] --model --device cuda --dtype bfloat16 --context-lens 32768 65536 --corpus wikitext-103 --max-tokens 3000000 --ranks 64 128 256 --evict-ratios 0.75 0.875 0.94 --exact-fracs 0.03 --n-windows 8 --out-json |
| `scripts/w5_needle.py` | w5_hybrid_needle_8b.sh:35 | [legacy, `week3`] --model --device cuda --dtype bfloat16 --context-lens 4096 16384 --depths 0.25 0.5 0.75 --passcodes 48213 70561 --out-json |
| `scripts/w5_longctx.py` | w5_longctx_8b.sh:27 | [legacy, `week3`] --model --device cuda --dtype bfloat16 --context-lens 1024 4096 8192 16384 32768 --ranks 64 128 256 --evict-ratios 0.5 0.7 0.85 --bits 4 --n-windows 4 --out-json |
| `scripts/w5_streamppl.py` | w5_streamppl_8b.sh:38,54 (`week3`); w7_streamppl_8b.sh:39,53 (`week7`) | --model --device cuda --dtype bfloat16 --corpus wikitext-103 --prefill 1024 --g-tokens 8192 --bin-size 512 --n-docs 3 --doc-stride 60000 --rank {128/64} --coord-budget {2048/1024} --recent-window {64/32} --absorb-block {32/16} --morph-recent 32 [--methods full,bug,bugA,morph,snapkvD -- w7 pod only] --out-json --fig |

**Transitive pod-sh dependency not obvious from the table above:**
`scripts/w12_calibrate_qkey.py` (root, via `w11_r128.sh` MODE=qbug) imports
`scripts/capture_kv.py`, which is therefore pod-sh-reachable despite never
appearing on a `scripts/pod/*.sh` command line itself.

## 2. Full table -- every file under `src/kvdlra/` and `scripts/*.py`

92 files total (30 under `src/kvdlra/`, 62 `scripts/*.py`; `scripts/pod/`
excluded as shell, not Python). `reachable-from` per the brief's exact
definitions: **pod-sh** = in the R1 import closure; **tests** = imported
by `tests/*.py` (directly or transitively) but not pod-sh; **pod-sh+tests**
= both; **scripts-only** = imported by ≥ 1 file, but every importer chain
dead-ends in a script nothing invokes/imports; **none** = imported by
nothing at all (and, for the 16 R1 roots' own bucket, n/a -- they are
reachable by direct shell invocation, not import).

| path | LOC | reachable-from |
|---|---|---|
| `scripts/capture_kv.py` | 381 | pod-sh |
| `scripts/w11_probe.py` | 203 | pod-sh |
| `scripts/w12_calibrate_qkey.py` | 98 | pod-sh |
| `scripts/w16_storage.py` | 229 | pod-sh |
| `scripts/w16_tier2_probe.py` | 228 | pod-sh |
| `scripts/w20_latency.py` | 160 | pod-sh |
| `scripts/w5_fp16_longctx.py` | 292 | pod-sh |
| `scripts/w5_hybrid.py` | 309 | pod-sh |
| `scripts/w5_longctx.py` | 355 | pod-sh |
| `scripts/w5_needle.py` | 264 | pod-sh |
| `scripts/w11_merge.py` | 165 | tests |
| `scripts/w15_intervals.py` | 161 | tests |
| `scripts/w18_intervals.py` | 212 | tests |
| `src/kvdlra/integrators/bug_class.py` | 240 | tests |
| `src/kvdlra/integrators/bug_torch.py` | 177 | tests |
| `src/kvdlra/integrators/frequent_directions.py` | 132 | tests |
| `src/kvdlra/integrators/oja.py` | 310 | tests |
| `src/kvdlra/integrators/streaming_variants.py` | 281 | tests |
| `src/kvdlra/lowrank.py` | 66 | tests |
| `scripts/_paths.py` | 30 | pod-sh+tests |
| `scripts/perplexity_sweep.py` | 322 | pod-sh+tests |
| `scripts/w10_frontier.py` | 985 | pod-sh+tests |
| `scripts/w10_longbench.py` | 362 | pod-sh+tests |
| `scripts/w10_ruler.py` | 689 | pod-sh+tests |
| `scripts/w19_official_ruler.py` | 195 | pod-sh+tests |
| `scripts/w19_persist.py` | 224 | pod-sh+tests |
| `scripts/w4_fair.py` | 197 | pod-sh+tests |
| `scripts/w4_hybrid_sweep.py` | 180 | pod-sh+tests |
| `scripts/w4_needle.py` | 212 | pod-sh+tests |
| `scripts/w5_ruler.py` | 280 | pod-sh+tests |
| `scripts/w5_streamppl.py` | 562 | pod-sh+tests |
| `src/kvdlra/__init__.py` | 1 | pod-sh+tests |
| `src/kvdlra/accounting.py` | 540 | pod-sh+tests |
| `src/kvdlra/cache/__init__.py` | 14 | pod-sh+tests |
| `src/kvdlra/cache/bug_cache.py` | 2022 | pod-sh+tests |
| `src/kvdlra/cache/morph_cache.py` | 449 | pod-sh+tests |
| `src/kvdlra/cache/shadow_cache.py` | 582 | pod-sh+tests |
| `src/kvdlra/integrators/__init__.py` | 0 | pod-sh+tests |
| `src/kvdlra/integrators/bug.py` | 204 | pod-sh+tests |
| `src/kvdlra/integrators/bug_adaptive.py` | 198 | pod-sh+tests |
| `src/kvdlra/integrators/streaming.py` | 329 | pod-sh+tests |
| `src/kvdlra/integrators/streaming_torch.py` | 390 | pod-sh+tests |
| `src/kvdlra/press/__init__.py` | 9 | pod-sh+tests |
| `src/kvdlra/press/bug_press.py` | 389 | pod-sh+tests |
| `src/kvdlra/press/compat.py` | 75 | pod-sh+tests |
| `src/kvdlra/press/palu_press.py` | 92 | pod-sh+tests |
| `src/kvdlra/press/turbo_press.py` | 107 | pod-sh+tests |
| `src/kvdlra/quant/__init__.py` | 9 | pod-sh+tests |
| `src/kvdlra/quant/kivi_cache.py` | 172 | pod-sh+tests |
| `src/kvdlra/quant/polar.py` | 146 | pod-sh+tests |
| `src/kvdlra/quant/product_quant.py` | 300 | pod-sh+tests |
| `src/kvdlra/quant/qjl.py` | 109 | pod-sh+tests |
| `src/kvdlra/utils/__init__.py` | 0 | pod-sh+tests |
| `src/kvdlra/utils/seed.py` | 35 | pod-sh+tests |
| `scripts/generate_with_press.py` | 269 | scripts-only |
| `scripts/w12_qbug_probe.py` | 299 | scripts-only |
| `scripts/w19_fork_report.py` | 135 | scripts-only |
| `scripts/w7_rank_sweep.py` | 326 | scripts-only |
| `scripts/calibrate_codebook.py` | 193 | none |
| `scripts/sigma_decay.py` | 207 | none |
| `scripts/w10_parse_logs.py` | 80 | none |
| `scripts/w13_tracka_probe.py` | 557 | none |
| `scripts/w13_trackb_bypass.py` | 236 | none |
| `scripts/w13_trackc_probe.py` | 282 | none |
| `scripts/w13_trackx_angles_probe.py` | 298 | none |
| `scripts/w13_trackx_rank_probe.py` | 231 | none |
| `scripts/w14_second_bypass_probe.py` | 629 | none |
| `scripts/w15_scorerank_probe.py` | 435 | none |
| `scripts/w16_intervals.py` | 159 | none |
| `scripts/w17_intervals.py` | 164 | none |
| `scripts/w19_a1_report.py` | 157 | none |
| `scripts/w19_a2_misses.py` | 81 | none |
| `scripts/w19_dashboard.py` | 488 | none |
| `scripts/w19_figures.py` | 284 | none |
| `scripts/w20_close_report.py` | 183 | none |
| `scripts/w45_integrator_ablation.py` | 368 | none |
| `scripts/w4_head_to_head.py` | 181 | none |
| `scripts/w5_decode_validate.py` | 246 | none |
| `scripts/w7_codebook.py` | 327 | none |
| `scripts/w7_decode_needle.py` | 281 | none |
| `scripts/w7_fd_ablation.py` | 114 | none |
| `scripts/w8_carry_drift.py` | 200 | none |
| `scripts/w9_envelope.py` | 471 | none |
| `scripts/w9_frontier.py` | 133 | none |
| `scripts/w9_recall_8b.py` | 138 | none |
| `scripts/w9_recovery.py` | 368 | none |
| `scripts/w9_surprise.py` | 355 | none |
| `scripts/week1_sv_decay.py` | 39 | none |
| `scripts/week2_oja_vs_bug.py` | 311 | none |
| `scripts/week2_pilot.py` | 203 | none |
| `scripts/week2_select_docs.py` | 62 | none |
| `src/kvdlra/utils/hydra_resolvers.py` | 20 | none |

**Counts:** pod-sh 10 (2519 LOC) &middot; tests 9 (1744 LOC) &middot; pod-sh+tests 35 (10410 LOC) &middot; scripts-only 4 (1029 LOC) &middot; none 34 (8481 LOC). Total 92 files, 24183 LOC.

## 3. Deletion-candidate lists

### 3a. `none` -- imported by nothing, not an R1 root
34 files, 8481 LOC subtotal.

| path | LOC |
|---|---|
| `scripts/w14_second_bypass_probe.py` | 629 |
| `scripts/w13_tracka_probe.py` | 557 |
| `scripts/w19_dashboard.py` | 488 |
| `scripts/w9_envelope.py` | 471 |
| `scripts/w15_scorerank_probe.py` | 435 |
| `scripts/w45_integrator_ablation.py` | 368 |
| `scripts/w9_recovery.py` | 368 |
| `scripts/w9_surprise.py` | 355 |
| `scripts/w7_codebook.py` | 327 |
| `scripts/week2_oja_vs_bug.py` | 311 |
| `scripts/w13_trackx_angles_probe.py` | 298 |
| `scripts/w19_figures.py` | 284 |
| `scripts/w13_trackc_probe.py` | 282 |
| `scripts/w7_decode_needle.py` | 281 |
| `scripts/w5_decode_validate.py` | 246 |
| `scripts/w13_trackb_bypass.py` | 236 |
| `scripts/w13_trackx_rank_probe.py` | 231 |
| `scripts/sigma_decay.py` | 207 |
| `scripts/week2_pilot.py` | 203 |
| `scripts/w8_carry_drift.py` | 200 |
| `scripts/calibrate_codebook.py` | 193 |
| `scripts/w20_close_report.py` | 183 |
| `scripts/w4_head_to_head.py` | 181 |
| `scripts/w17_intervals.py` | 164 |
| `scripts/w16_intervals.py` | 159 |
| `scripts/w19_a1_report.py` | 157 |
| `scripts/w9_recall_8b.py` | 138 |
| `scripts/w9_frontier.py` | 133 |
| `scripts/w7_fd_ablation.py` | 114 |
| `scripts/w19_a2_misses.py` | 81 |
| `scripts/w10_parse_logs.py` | 80 |
| `scripts/week2_select_docs.py` | 62 |
| `scripts/week1_sv_decay.py` | 39 |
| `src/kvdlra/utils/hydra_resolvers.py` | 20 |

**Caveat (see §4 and §6): 6 of these are the paper's own
report/figure/dashboard generators**, run manually against harvested
`results/` data (never from a pod script, never imported by a test, so
they are correctly `none` by this task's R1/R2 definitions) but they are
the confirmed producers of paper-cited artifacts: `w19_a1_report.py`
(`results/w19-a1-report.md`), `w19_a2_misses.py`
(`results/w19-a2-flagship-misses.md`), `w19_figures.py` (all three
`figures/week19/*.pdf`), `w20_close_report.py` (`results/w20-close-report.md`,
and it nested-imports `w19_fork_report.py`, see below), `w16_intervals.py`
/ `w17_intervals.py` (feed the superseded Week-16/17 dashboards). Deleting
them would not break anything R1/R2 can see, but it would remove the
paper's reproduction path for those tables/figures. Flag for the cleanup
lane rather than auto-delete.

### 3b. `tests` (a.k.a. tests-only) -- kept alive solely by `tests/*.py`
9 files, 1744 LOC subtotal.
These files themselves are not deletion candidates while their owning
test(s) are kept, but **the test files are candidates too** if the
cleanup drops the module under test; named per module below.

| module | LOC | kept alive by |
|---|---|---|
| `scripts/w11_merge.py` | 165 | `tests/test_w15_pplw.py`, `tests/test_w18_intervals.py`, `tests/test_w19_official_ruler.py` |
| `scripts/w15_intervals.py` | 161 | `tests/test_w18_intervals.py`, `tests/test_w19_official_ruler.py` |
| `scripts/w18_intervals.py` | 212 | `tests/test_w18_intervals.py`, `tests/test_w19_official_ruler.py` |
| `src/kvdlra/integrators/bug_class.py` | 240 | `tests/test_bug_class.py` |
| `src/kvdlra/integrators/bug_torch.py` | 177 | `tests/test_bug_torch.py` |
| `src/kvdlra/integrators/frequent_directions.py` | 132 | `tests/test_frequent_directions.py` |
| `src/kvdlra/integrators/oja.py` | 310 | `tests/test_oja.py` |
| `src/kvdlra/integrators/streaming_variants.py` | 281 | `tests/test_streaming_variants.py` |
| `src/kvdlra/lowrank.py` | 66 | `tests/test_frequent_directions.py` |

### 3c. `scripts-only` -- imported only by a script nothing invokes
4 files, 1029 LOC subtotal.

| module | LOC | sole importer(s) (themselves `none`) |
|---|---|---|
| `scripts/generate_with_press.py` | 269 | `scripts/w5_decode_validate.py` |
| `scripts/w12_qbug_probe.py` | 299 | `scripts/w13_trackc_probe.py` |
| `scripts/w19_fork_report.py` | 135 | `scripts/w20_close_report.py` |
| `scripts/w7_rank_sweep.py` | 326 | `scripts/w7_codebook.py`, `scripts/w9_envelope.py`, `scripts/w9_recovery.py`, `scripts/w9_surprise.py` |

`scripts/w19_fork_report.py` is the sharpest example of why `scripts-only`
≠ safe-to-delete: it is imported only by `w20_close_report.py` (itself
`none`) via a **function-body** import (`w20_close_report.py:135-136`,
`from w19_fork_report import COMPO, OFF9, SB` / `... import parse as
fork_parse`) that a plain top-of-file grep would miss but `ast.walk` does
not -- and the paper's own `% source:` comment at `paper/main.tex:723`
names `scripts/w19_fork_report.py` as the direct producer of
`results/w20-fork-report.md`.

## 4. Paper provenance (`paper/main.tex`)

21 `% source:` comments (`grep -c '% source:' paper/main.tex`); plus a
supplementary grep for any `results/`/`figures/` path literal anywhere in
`main.tex` (catches the 3 `\includegraphics` figure paths and the
`results/w18\_pertrial/` / `results/w19\_pertrial/` prose mentions the
`% source:`-only grep would miss). Brace-expansion citations
(`results/w19-a1q-{llama,mistral,qwen}-lines.txt`,
`results/w18-g3-{qwen,mistral,llama}-lines.txt`) expanded by hand; two
citations are prose shorthand for a prefix (`results/w18-g4-llama`,
`results/w18-g5`) rather than one literal filename, resolved against the
actual files below. **All 32 resolved paths exist on disk and none are
gitignored** (`git check-ignore -v` on every path, and on the containing
dirs `results/`, `results/w18_pertrial/`, `results/w19_intervals/`,
`figures/week19/`, all exit 1 / no match).

`.gitignore` only excludes narrower patterns inside `results/`/`figures/`:
`results/*.csv`, `results/gpu_logs/`, `results/scratch/`,
`results/w18_harvest/*.raw`, `results/w19_harvest/*.raw`,
`results/w19_harvest/{watchdog.log,watchdog.pid,status.txt}`,
`figures/scratch/` -- none of the paper-cited paths fall in these.

| cited path | exists | producer / consumer among `scripts/*.py` |
|---|---|---|
| `results/w18_pertrial/` | 9 files, all git-tracked | written by shell watchdog harvest (`w18_watchdog.sh`/`w18_watchdog2.sh` `${H}/${tag}.raw` -> `[trial]`-row extraction); read by no `scripts/*.py` |
| `results/w19_pertrial/` | 18 files, all git-tracked | written by `w19_watchdog.sh`'s `extract()` (`grep -aE '^\[trial\]' > results/w19_pertrial/${l}-trials.txt`); `q4off-llama-trials.txt` specifically -> no `scripts/*.py` reference; the directory is read generically by `scripts/w19_a2_misses.py` |
| `results/w18-env-provenance.txt` | yes | assembled from the `===ENV_BEGIN===`/`===ENV_END===` block `w18_boot.sh` prints (SHA, nvidia-smi, python/torch/transformers versions via an inline heredoc, not a repo script) + watchdog raw harvest; no `scripts/*.py` writes/reads this merged filename |
| `results/w19-env-provenance.txt` | yes | same mechanism as above (w19 side) |
| `results/w18-g1-report.md` | yes | no `scripts/*.py` reference found -- reads as a hand-authored analysis doc over the g1 raws |
| `results/w18-g2-qwen-lines.txt` | yes | written by `w18_watchdog2.sh` `extract()`; no `scripts/*.py` reader |
| `results/w18-g3-{qwen,mistral,llama}-lines.txt` | yes (3 files) | written by `w18_watchdog2.sh` `extract()`; no `scripts/*.py` reader |
| `results/w18-g4-llama-lines.txt + -ppl-lines.txt` | yes (2 files) | written by `w18_watchdog2.sh` `extract()` and/or the dedicated `w18_g4fallback.sh`; no `scripts/*.py` reader |
| `results/w18-g4-marquee-contrasts.json` | yes | no `scripts/*.py` reference -- hand-derived |
| `results/w18-g5-llama-lines.txt` | yes | written by `w18_watchdog2.sh` `extract()`; no `scripts/*.py` reader |
| `results/w18_harvest/quant-findings.md` | yes | no `scripts/*.py` reference -- hand-authored analysis doc |
| `results/w19-a1-report.md` | yes | **written by `scripts/w19_a1_report.py`** (literal match) |
| `results/w19-a1q-{llama,mistral,qwen}-lines.txt` | yes (3 files) | written by `w19_watchdog.sh` `extract()`; no `scripts/*.py` reader |
| `results/w19-a2-flagship-misses.md` | yes | **written by `scripts/w19_a2_misses.py`** (literal match) |
| `results/w19-a2-llama-lines.txt` | yes | written by `w19_watchdog.sh`; **read by `scripts/w20_close_report.py`** (`RES / "w19-a2-llama-lines.txt"`) |
| `results/w19-a3-llama2-lines.txt` | yes | written by `w19_watchdog.sh`; **read by `w19_a1_report.py`, `w19_figures.py`, `w19_dashboard.py`** |
| `results/w19-a4-llama-lines.txt` | yes | written by `w19_watchdog.sh`; **read by `w19_figures.py`, `w19_dashboard.py`** |
| `results/w19-q4off-llama-lines.txt` | yes | written by `w19_watchdog.sh`; **read by `w20_close_report.py`** |
| `results/w19-swap-llama-lines.txt` | yes | written by `w19_watchdog.sh`; **read by `w20_close_report.py`** |
| `results/w19-sysfix-llama-lines.txt` | yes | written by `w19_watchdog.sh`; **read by `w20_close_report.py`** (also the source for `results/w19-a3-llama2-lines.txt`'s latency companion) |
| `results/w19_intervals/a2-llama-ruler-intervals.md` | yes | no `scripts/*.py` reference to this exact basename; the `results/w19_intervals/` dir is read generically by `w19_a1_report.py`, `w19_figures.py`, `w19_dashboard.py` |
| `results/w20-fork-report.md` | yes | **written by `scripts/w19_fork_report.py`** (literal match; see §3c -- imported by `w20_close_report.py` via a function-body import) |
| `results/w11-goalA-lb-lines.txt` | yes | no `scripts/*.py` reference (predates the merge-script family; sibling `w11-goalA-ruler-lines.txt`/`w11-goalB-*` files *are* read by `scripts/w11_merge.py`, this LongBench one is not) |
| `figures/week19/{coldstart,fairquant,one_over_t}.pdf` | yes (3 files) | **written by `scripts/w19_figures.py`** (`fig_coldstart`/`fig_fairquant`/`fig_one_over_t`, `OUT = Path("figures/week19")`; basename built from an f-string so a literal-string grep for `coldstart.pdf` alone finds nothing -- confirmed by reading the function bodies) |

**Pattern:** every `-lines.txt` evidence file is produced by a *shell*
watchdog (`grep`/`sort` over `vastai logs`, not a `scripts/*.py`
only the second-stage report/figure generators
(`w19_a1_report.py`, `w19_a2_misses.py`, `w19_fork_report.py`,
`w20_close_report.py`, `w19_figures.py`, `w19_dashboard.py`) are Python and
read a subset of those files back in -- and every one of those generators
is itself `none`-reachable (§3a), so **this whole provenance chain is
invisible to the R1/R2 reachability graph** the cleanup lane is using.

## 5. `results/` inventory

`du -sh results` total: **99M**, 186 top-level entries (8 directories +
178 files). Full top-level listing, sorted by size:

```
 74M	results/w18_harvest
 20M	results/w19_harvest
2.5M	results/gpu_logs
352K	results/w19_pertrial
204K	results/w18_checkpoint
176K	results/w18_pertrial
 96K	results/w13-tracka-probe.json
 80K	results/w19_intervals
 72K	results/w15-scorerank-probe.json
 52K	results/w5-decode-validate-1b.json
 36K	results/w15-ruler-intervals.json
 36K	results/scratch
 20K	results/w9-envelope-gate-1b.json
 20K	results/w13-trackx-rank.json
 20K	results/w13-trackb-design.md
 20K	results/w10-gpu-parsed.json
 16K	results/w7-streamppl-8b.json
 16K	results/w7-streamppl-8b-tier2.json
 16K	results/w17-ruler-intervals.json
 16K	results/w16-ruler-intervals.json
 16K	results/w13-trackc-probe.json
 16K	results/w11-decision-table.json
 12K	results/w9-surprise-blend-1b.json
 12K	results/w8-codebook.pt
 12K	results/w5-streamppl-8b.json
 12K	results/w5-streamppl-8b-tier2.json
 12K	results/w10-frontier-1b.json
8.0K	results/w9-surprise-sweep-1b.json
8.0K	results/w8-codebook-1b-doc1.json
8.0K	results/w8-codebook-1b-doc0.json
8.0K	results/w7-streamppl-1b.json
8.0K	results/w7-streamppl-1b-doc0.json
8.0K	results/w7-merge-doc0.json
8.0K	results/w7-fd-ablation.json
8.0K	results/w5-streamppl-1b.json
8.0K	results/w5-longctx-8b.json
8.0K	results/w5-hybrid-1b.json
8.0K	results/w3-parity.md
8.0K	results/w19-fork-llama-lines.txt
8.0K	results/w19-env-provenance.txt
8.0K	results/w19-a2-llama-lines.txt
8.0K	results/w19-a1-report.md
8.0K	results/w17-decision-table.json
8.0K	results/w16-decision-table.json
8.0K	results/w15-ruler-intervals.md
8.0K	results/w15-e2e-probe.json
8.0K	results/w15-confirm-summary.md
8.0K	results/w14-second-bypass-summary.md
8.0K	results/w14-second-bypass-facts.json
8.0K	results/w13-trackx-angles.json
8.0K	results/w13-trackb-summary.md
8.0K	results/w13-phase1-summary.md
8.0K	results/w11-probe8b-all.json
8.0K	results/w11-goalA-ruler-lines.txt
8.0K	results/w11-decision-table-base.json
4.0K	results/w9-surprise-probe-1b.json
4.0K	results/w9-recovery-gate-1b.json
4.0K	results/w9-recovery-firm-1b.json
4.0K	results/w9-recovery-ctxsweep-1b.json
4.0K	results/w9-recovery-8b-win-single.json
4.0K	results/w9-recovery-8b-win-multikey.json
4.0K	results/w9-recovery-8b-single.json
4.0K	results/w9-recovery-8b-ranksweep.json
4.0K	results/w9-recovery-8b-multikey.json
4.0K	results/w9-recovery-1b.json
4.0K	results/w8-codebook.json
4.0K	results/w8-carry-drift.json
4.0K	results/w7-slash-moderate-doc0.json
4.0K	results/w7-slash-doc0.json
4.0K	results/w7-ranksweep-fifo-doc0.json
4.0K	results/w7-ranksweep-attn-doc0.json
4.0K	results/w7-diagcore-moderate-doc0.json
4.0K	results/w5-streamppl-1b-smoke.json
4.0K	results/w5-ruler-8b.json
4.0K	results/w5-needle-8b.json
4.0K	results/w5-hybrid-8b.json
4.0K	results/w5-fp16-longctx-8b.json
4.0K	results/w4-needle.json
4.0K	results/w4-hybrid.json
4.0K	results/w4-hybrid-8b.json
4.0K	results/w4-head-to-head.json
4.0K	results/w4-fair.json
4.0K	results/w4-fair-8b.json
4.0K	results/w3-ppl-1b-pre.json
4.0K	results/w3-ppl-1b-pre.csv
4.0K	results/w3-ppl-1b-post.json
4.0K	results/w3-ppl-1b-post.csv
4.0K	results/w20-fork-report.md
4.0K	results/w20-close-report.md
4.0K	results/w19-sysfix-llama-lines.txt
4.0K	results/w19-swap-llama-lines.txt
4.0K	results/w19-q4off-llama-lines.txt
4.0K	results/w19-fork-qwen-lines.txt
4.0K	results/w19-fork-mistral-lines.txt
4.0K	results/w19-a4-llama-lines.txt
4.0K	results/w19-a3-llama2-lines.txt
4.0K	results/w19-a2-flagship-misses.md
4.0K	results/w19-a1q-qwen-lines.txt
4.0K	results/w19-a1q-mistral-lines.txt
4.0K	results/w19-a1q-llama-lines.txt
4.0K	results/w19-a1diag-qwen-lines.txt
4.0K	results/w19-a1-qwen-lines.txt
4.0K	results/w19-a1-mistral-lines.txt
4.0K	results/w19-a1-llama-lines.txt
4.0K	results/w18-qwen-ppl-lines.txt
4.0K	results/w18-qwen-lines.txt
4.0K	results/w18-mistral-ppl-lines.txt
4.0K	results/w18-mistral-lines.txt
4.0K	results/w18-llama-ppl-lines.txt
4.0K	results/w18-llama-lines.txt
4.0K	results/w18-g5-llama-lines.txt
4.0K	results/w18-g4-marquee-contrasts.json
4.0K	results/w18-g4-llama-ppl-lines.txt
4.0K	results/w18-g4-llama-lines.txt
4.0K	results/w18-g3-qwen-lines.txt
4.0K	results/w18-g3-mistral-lines.txt
4.0K	results/w18-g3-llama-lines.txt
4.0K	results/w18-g2-qwen-lines.txt
4.0K	results/w18-g1-ruler-intervals.md
4.0K	results/w18-g1-ruler-intervals.json
4.0K	results/w18-g1-report.md
4.0K	results/w18-g1-qwen-ruler-intervals.md
4.0K	results/w18-g1-qwen-ruler-intervals.json
4.0K	results/w18-g1-mistral-ruler-intervals.md
4.0K	results/w18-g1-mistral-ruler-intervals.json
4.0K	results/w18-g1-llama-ruler-intervals.md
4.0K	results/w18-g1-llama-ruler-intervals.json
4.0K	results/w18-env-provenance.txt
4.0K	results/w17-ruler-intervals.md
4.0K	results/w17-qwen-lines.txt
4.0K	results/w17-mistral-lines.txt
4.0K	results/w17-llama8b-lines.txt
4.0K	results/w16-ruler-intervals.md
4.0K	results/w16-qwen-tier-lines.txt
4.0K	results/w16-qwen-sweep-lines.txt
4.0K	results/w16-qwen-ppl-lines.txt
4.0K	results/w16-mistral-tier-lines.txt
4.0K	results/w16-mistral-sweep-lines.txt
4.0K	results/w16-mistral-ppl-lines.txt
4.0K	results/w16-llama8b-tier-lines.txt
4.0K	results/w16-llama8b-sweep-lines.txt
4.0K	results/w15b-complete-lines.txt
4.0K	results/w15-confirm-lines.txt
4.0K	results/w15-complete-summary.md
4.0K	results/w15-complete-lines.txt
4.0K	results/w14-wseed8-ruler-lines.txt
4.0K	results/w14-wc2n4-ruler-lines.txt
4.0K	results/w14-c2-summary.md
4.0K	results/w14-c1-summary.md
4.0K	results/w13-wseed-ruler-lines.txt
4.0K	results/w13-wseed-ppl-lines.txt
4.0K	results/w13-trackb-facts.json
4.0K	results/w12-r192-ruler-lines.txt
4.0K	results/w12-r192-ppl-lines.txt
4.0K	results/w12-qbug-summary.md
4.0K	results/w12-qbug-ruler-lines.txt
4.0K	results/w12-qbug-probe.json
4.0K	results/w12-qbug-ppl-lines.txt
4.0K	results/w12-probe-r128-lines.txt
4.0K	results/w12-mk64-mvvt-lines.txt
4.0K	results/w12-mk64-mk-lines.txt
4.0K	results/w12-drop-ruler-lines.txt
4.0K	results/w12-bugr128-ruler-lines.txt
4.0K	results/w11-table-ruler-lines.txt
4.0K	results/w11-table-ppl-lines.txt
4.0K	results/w11-r256-ruler-lines.txt
4.0K	results/w11-r128v2-ruler-lines.txt
4.0K	results/w11-r128v1-ruler-lines.txt
4.0K	results/w11-r128-ppl-lines.txt
4.0K	results/w11-probe-1b-mk-c8192-t1s1.json
4.0K	results/w11-probe-1b-mk-c8192-t0s0.json
4.0K	results/w11-probe-1b-mk-c4096-t1s1.json
4.0K	results/w11-probe-1b-mk-c4096-t0s0.json
4.0K	results/w11-probe-1b-mk-c16384-t0s0.json
4.0K	results/w11-probe-1b-c8192-t0s0.json
4.0K	results/w11-probe-1b-c4096-t1s1.json
4.0K	results/w11-probe-1b-c4096-t0s0.json
4.0K	results/w11-goalB-ruler-lines.txt
4.0K	results/w11-goalB-probe-lines.txt
4.0K	results/w11-goalB-ppl-lines.txt
4.0K	results/w11-goalA-lb-lines.txt
4.0K	results/w11-final-tables.md
4.0K	results/w11-base-ruler-lines.txt
4.0K	results/w11-base-ppl16-lines.txt
  0B	results/w19-forkdiag-qwen-lines.txt
  0B	results/w19-a3-llama-lines.txt
```

Directories recursed one level (git-tracked count via `git ls-files`):

| dir | size | entries | git-tracked | note |
|---|---|---|---|---|
| `results/w18_harvest/` | 74M | 13 | 4 | 9 `*.raw` files (~73M) are gitignored (`results/w18_harvest/*.raw`); tracked: `quant-findings.md`, `SUMMARY.txt`, `G4DONE.txt`, `DONE.txt` |
| `results/w19_harvest/` | 20M | 22 | 3 | 11 `*.raw` files (~19M, incl. one `a3-llama.raw.superseded`) are gitignored; tracked: `status.txt` and 2 others |
| `results/gpu_logs/` | 2.5M | 30 | 0 | entire directory gitignored (`results/gpu_logs/`) -- pre-Week-10 `.acc.log` files, historical only |
| `results/w19_pertrial/` | 352K | 18 | 18 | fully tracked |
| `results/w18_checkpoint/` | 204K | 6 | 6 | fully tracked (not gitignored despite the name) |
| `results/w18_pertrial/` | 176K | 9 | 9 | fully tracked |
| `results/w19_intervals/` | 80K | 8 | 8 | fully tracked |
| `results/scratch/` | 36K | 9 | 0 | entire directory gitignored (`results/scratch/`) -- CPU probe scratch JSON (w19/w20 diagnostics) |

### `results/*.json` / `results/*-lines.txt` (top-level only) -> referencing `scripts/*.py`

150 basenames checked (79 `.json` + 71 `-lines.txt`, top-level only, glob non-recursive) against the literal text of every `scripts/*.py` file.
**66 basenames appear literally in a script's source** (below);
**84 do not** -- the general reason (verified against
several `--out-json`/`--out` argparse defaults, e.g. `w10_frontier.py`
defaults to `results/w10-frontier-1b.json`, which matches and explains why
that one *is* in the referenced list) is that the R1 harness scripts
(`w5_hybrid.py`, `w5_ruler.py`, `w5_needle.py`, `w5_longctx.py`,
`w5_fp16_longctx.py`, `w5_streamppl.py`, `w10_frontier.py`, `w10_ruler.py`,
`w10_longbench.py`, `w11_probe.py`, `w12_calibrate_qkey.py`, `w16_storage.py`,
`w16_tier2_probe.py`, `w19_official_ruler.py`, `w19_persist.py`,
`w20_latency.py`) all take `--out-json PATH` / `--out PATH` as a CLI flag;
the concrete `wNN-...json` basename on disk is almost always supplied by
the *shell caller* (`scripts/pod/*.sh`, §1) or is a scraped/merged product
of a shell watchdog (§4), so it never appears as a string literal inside
the `.py` source itself even though that script is the true writer.
Matching a `none`-referenced basename's `wNN-<verb>` prefix against §1's
roots table (or §4's shell-watchdog note) recovers the producer in almost
every case.

| basename | referencing `scripts/*.py` |
|---|---|
| `w10-frontier-1b.json` | w10_frontier.py |
| `w10-gpu-parsed.json` | w10_parse_logs.py |
| `w11-base-ruler-lines.txt` | w11_merge.py |
| `w11-decision-table-base.json` | w11_merge.py |
| `w11-decision-table.json` | w11_merge.py, w15_intervals.py, w16_intervals.py |
| `w11-goalA-ruler-lines.txt` | w11_merge.py |
| `w11-goalB-ruler-lines.txt` | w11_merge.py |
| `w11-r128-ppl-lines.txt` | w11_merge.py |
| `w11-r128v1-ruler-lines.txt` | w11_merge.py |
| `w11-r128v2-ruler-lines.txt` | w11_merge.py |
| `w11-r256-ruler-lines.txt` | w11_merge.py |
| `w11-table-ruler-lines.txt` | w11_merge.py |
| `w12-bugr128-ruler-lines.txt` | w11_merge.py |
| `w12-drop-ruler-lines.txt` | w11_merge.py |
| `w12-mk64-mk-lines.txt` | w11_merge.py |
| `w12-mk64-mvvt-lines.txt` | w11_merge.py |
| `w12-qbug-probe.json` | w12_qbug_probe.py |
| `w12-r192-ppl-lines.txt` | w11_merge.py |
| `w12-r192-ruler-lines.txt` | w11_merge.py |
| `w13-tracka-probe.json` | w13_tracka_probe.py |
| `w13-trackb-facts.json` | w13_trackb_bypass.py |
| `w13-trackc-probe.json` | w13_trackc_probe.py |
| `w13-trackx-angles.json` | w13_trackx_angles_probe.py |
| `w13-trackx-rank.json` | w13_trackx_rank_probe.py |
| `w14-second-bypass-facts.json` | w14_second_bypass_probe.py |
| `w14-wseed8-ruler-lines.txt` | w15_intervals.py |
| `w15-ruler-intervals.json` | w15_intervals.py |
| `w15-scorerank-probe.json` | w15_scorerank_probe.py |
| `w16-decision-table.json` | w16_intervals.py |
| `w16-llama8b-sweep-lines.txt` | w16_intervals.py |
| `w16-llama8b-tier-lines.txt` | w16_intervals.py |
| `w16-mistral-ppl-lines.txt` | w16_intervals.py |
| `w16-mistral-sweep-lines.txt` | w16_intervals.py |
| `w16-mistral-tier-lines.txt` | w16_intervals.py |
| `w16-qwen-ppl-lines.txt` | w16_intervals.py |
| `w16-qwen-sweep-lines.txt` | w16_intervals.py |
| `w16-qwen-tier-lines.txt` | w16_intervals.py |
| `w16-ruler-intervals.json` | w16_intervals.py |
| `w17-decision-table.json` | w17_intervals.py |
| `w17-ruler-intervals.json` | w17_intervals.py |
| `w18-llama-lines.txt` | w19_figures.py, w20_close_report.py |
| `w18-llama-ppl-lines.txt` | w20_close_report.py |
| `w19-a1-llama-lines.txt` | w19_figures.py, w20_close_report.py |
| `w19-a2-llama-lines.txt` | w20_close_report.py |
| `w19-a3-llama2-lines.txt` | w19_a1_report.py, w19_dashboard.py, w19_figures.py |
| `w19-a4-llama-lines.txt` | w19_dashboard.py, w19_figures.py |
| `w19-q4off-llama-lines.txt` | w20_close_report.py |
| `w19-swap-llama-lines.txt` | w20_close_report.py |
| `w19-sysfix-llama-lines.txt` | w20_close_report.py |
| `w4-fair.json` | w4_fair.py |
| `w4-head-to-head.json` | w4_head_to_head.py |
| `w4-hybrid.json` | w4_fair.py, w4_head_to_head.py, w4_hybrid_sweep.py |
| `w4-needle.json` | w4_needle.py |
| `w5-decode-validate-1b.json` | w5_decode_validate.py |
| `w5-streamppl-1b.json` | w5_streamppl.py |
| `w7-fd-ablation.json` | w7_fd_ablation.py |
| `w7-ranksweep-attn-doc0.json` | w9_frontier.py |
| `w7-slash-doc0.json` | w9_envelope.py |
| `w7-streamppl-1b.json` | w5_streamppl.py |
| `w8-carry-drift.json` | w8_carry_drift.py |
| `w9-envelope-gate-1b.json` | w9_frontier.py |
| `w9-recovery-8b-ranksweep.json` | w9_recall_8b.py |
| `w9-recovery-8b-win-multikey.json` | w9_recall_8b.py |
| `w9-recovery-8b-win-single.json` | w9_recall_8b.py |
| `w9-recovery-gate-1b.json` | w9_recovery.py |
| `w9-surprise-sweep-1b.json` | w13_tracka_probe.py |

Basenames with **no** literal `scripts/*.py` reference (producer is a
shell watchdog and/or the file is a raw `--out-json` target whose path was
supplied by the calling `.sh`, per the pattern above):

```
w11-base-ppl16-lines.txt
w11-goalA-lb-lines.txt
w11-goalB-ppl-lines.txt
w11-goalB-probe-lines.txt
w11-probe-1b-c4096-t0s0.json
w11-probe-1b-c4096-t1s1.json
w11-probe-1b-c8192-t0s0.json
w11-probe-1b-mk-c16384-t0s0.json
w11-probe-1b-mk-c4096-t0s0.json
w11-probe-1b-mk-c4096-t1s1.json
w11-probe-1b-mk-c8192-t0s0.json
w11-probe-1b-mk-c8192-t1s1.json
w11-probe8b-all.json
w11-table-ppl-lines.txt
w12-probe-r128-lines.txt
w12-qbug-ppl-lines.txt
w12-qbug-ruler-lines.txt
w13-wseed-ppl-lines.txt
w13-wseed-ruler-lines.txt
w14-wc2n4-ruler-lines.txt
w15-complete-lines.txt
w15-confirm-lines.txt
w15-e2e-probe.json
w15b-complete-lines.txt
w17-llama8b-lines.txt
w17-mistral-lines.txt
w17-qwen-lines.txt
w18-g1-llama-ruler-intervals.json
w18-g1-mistral-ruler-intervals.json
w18-g1-qwen-ruler-intervals.json
w18-g1-ruler-intervals.json
w18-g2-qwen-lines.txt
w18-g3-llama-lines.txt
w18-g3-mistral-lines.txt
w18-g3-qwen-lines.txt
w18-g4-llama-lines.txt
w18-g4-llama-ppl-lines.txt
w18-g4-marquee-contrasts.json
w18-g5-llama-lines.txt
w18-mistral-lines.txt
w18-mistral-ppl-lines.txt
w18-qwen-lines.txt
w18-qwen-ppl-lines.txt
w19-a1-mistral-lines.txt
w19-a1-qwen-lines.txt
w19-a1diag-qwen-lines.txt
w19-a1q-llama-lines.txt
w19-a1q-mistral-lines.txt
w19-a1q-qwen-lines.txt
w19-a3-llama-lines.txt
w19-fork-llama-lines.txt
w19-fork-mistral-lines.txt
w19-fork-qwen-lines.txt
w19-forkdiag-qwen-lines.txt
w3-ppl-1b-post.json
w3-ppl-1b-pre.json
w4-fair-8b.json
w4-hybrid-8b.json
w5-fp16-longctx-8b.json
w5-hybrid-1b.json
w5-hybrid-8b.json
w5-longctx-8b.json
w5-needle-8b.json
w5-ruler-8b.json
w5-streamppl-1b-smoke.json
w5-streamppl-8b-tier2.json
w5-streamppl-8b.json
w7-diagcore-moderate-doc0.json
w7-merge-doc0.json
w7-ranksweep-fifo-doc0.json
w7-slash-moderate-doc0.json
w7-streamppl-1b-doc0.json
w7-streamppl-8b-tier2.json
w7-streamppl-8b.json
w8-codebook-1b-doc0.json
w8-codebook-1b-doc1.json
w8-codebook.json
w9-recovery-1b.json
w9-recovery-8b-multikey.json
w9-recovery-8b-single.json
w9-recovery-ctxsweep-1b.json
w9-recovery-firm-1b.json
w9-surprise-blend-1b.json
w9-surprise-probe-1b.json
```

## 6. Dynamic-import / uncertainty

**AST-detected dynamic imports: zero.** Every `.py` file under
`src/kvdlra/`, `scripts/`, and `tests/` was scanned (as part of the same
`ast.walk` pass used for the closure) for `importlib.import_module(...)`
and `__import__(...)` calls -- none found anywhere in the repo.

**Supplementary greps (belt-and-suspenders, since string-based/getattr
dispatch wouldn't show up as an `ast.Call` to `import_module`):**
- `importlib\.import_module|__import__\(|getattr\(\s*sys\.modules|globals\(\)\[|locals\(\)\[` -- zero matches repo-wide.
- Any mention of `importlib` at all -- exactly one file, `scripts/_paths.py`,
  which calls `importlib.util.find_spec("kvdlra")` to *check* whether the
  package is already importable before deciding whether to prepend `src/`
  to `sys.path`. This is a static existence probe, not a dynamic import of
  a variable module name -- no closure implication.
- Zero relative imports anywhere (`from \.+`), confirmed above in the
  Method note.

**Hydra: configured but currently dead.** `src/kvdlra/utils/hydra_resolvers.py`
(20 LOC) is `none`-reachable and has **zero** references anywhere in the
repo, including `configs/{eval,model,press}/` -- all three config
subdirectories are empty (`find configs -type f` returns nothing). A Hydra
resolver is normally wired up by a YAML `_target_:`/config default, which
is precisely the kind of string-based reference static `ast` analysis
cannot see -- but there is no YAML anywhere in this repo to carry such a
reference, so the "maybe it's config-driven" escape hatch is closed:
this module is genuinely unreached, not just statically-invisible-reached.

**Shell-side indirection (not a Python dynamic import, but a form of
indirection worth flagging):** `w11_r128.sh`, `w16.sh`, `w18.sh`, `w19.sh`
all dispatch on `case "$MODE" in ...)` to decide which shell function (and
therefore which `python` invocation) runs. Every arm is a static,
enumerable string in the file (`new16|new32|firm16|...`, `tier2|tier1|...`,
`g1|g1quant|g1diag|...`, `a1diag|a1|a1q|...`) -- all were read and are
reflected in §1 -- so this doesn't hide any script invocation from the
static scan; it only means a given pod run executes a strict subset of
the flags shown per script (whichever MODE was set via `-e MODE=...` at
`vastai create instance` time, outside this repo).

**Nested/function-body imports mattered.** `ast.walk` (rather than only
scanning module-level statements) is what caught `w10_frontier.py`'s
function-body `from kvdlra.press import PaluPress` /
`from kvdlra.cache import ShadowKVCache, ShadowKVLayer`, and
`w20_close_report.py`'s function-body `from w19_fork_report import ...`
(§3c). A plain top-of-file-only grep for `^(import|from)` would have
missed all four and mis-classified `w19_fork_report.py` as fully dead.

**External, out-of-repo Python invocations (not part of this closure):**
`w19.sh`'s `a2_prep()` clones `NVIDIA/RULER` at a pinned SHA and runs *its*
`download_paulgraham_essay.py` and `prepare.py` (`w19.sh:103,106`) -- these
are scripts inside the externally-cloned RULER repo, not
`kv-dlra/scripts/*.py`, so they are out of scope for the src/kvdlra +
scripts/ closure but are real `python <file>.py` invocations inside a
`scripts/pod/*.sh` file if a fully literal reading of R1 is wanted.

**Argparse-default ambiguity (not dynamic, but a source of false
negatives if someone re-derives §5's table differently):** several `-1b`
suffixed `results/*.json` files (e.g. `results/w5-hybrid-1b.json`) match a
harness script's argparse *default* `--out-json` value rather than any
`scripts/pod/*.sh` line (none of which requests the `-1b` suffix) --
consistent with a local/CPU smoke run outside the pod-sh pipeline, not a
gap in the static analysis.
