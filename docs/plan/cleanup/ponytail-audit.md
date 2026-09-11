# ponytail-audit — kv-dlra, repo-wide

Read-only scan, 2026-09-11, branch `week7`. Ranked biggest cut first.
Reachability evidence = which `scripts/pod/*.sh` MODE function builds a command line
that runs the file, and what that file imports.

## LOC by directory

| Location | Files | Lines |
|---|---|---|
| `src/` | 30 py | 7,398 |
| `scripts/` (top-level .py) | 62 py | 16,785 |
| `scripts/pod/` | 22 sh | 1,951 |
| `tests/` | 46 py | 9,555 |
| `docs/` (.md) | 76 md | 15,452 (of which `docs/plan/` 2,253) |
| `paper/` (.tex/.bib/Makefile) | 5 | 3,213 |

Also present and unaccounted for by the target layout: `experiments/` 825, `configs/` **0
bytes (three empty dirs, nothing tracked)**, `figures/` 89 tracked files, `dashboards/`
+ `handover.md` + `explanation_week_*.md` + `next-session-prompt.md` + `compass_artifact_*.md`
(all already gitignored — delete from disk, no commit needed).

---

## Findings

### scripts/ — the biggest pile

`delete:` 31 dead one-off probes, never on any pod command line, results already frozen in `results/` + `docs/`: `w12_qbug_probe`, `w13_{tracka_probe,trackb_bypass,trackc_probe,trackx_angles_probe,trackx_rank_probe}`, `w14_second_bypass_probe`, `w15_scorerank_probe`, `w45_integrator_ablation`, `w4_{fair,head_to_head,hybrid_sweep,needle}`, `w5_decode_validate`, `w7_{codebook,decode_needle,fd_ablation,rank_sweep}`, `w8_carry_drift`, `w9_{envelope,recall_8b,recovery,surprise}`, `week1_sv_decay`, `week2_{oja_vs_bug,pilot,select_docs}`, `perplexity_sweep`, `generate_with_press`, `calibrate_codebook`, `sigma_decay`. Replacement: nothing — `comm -23` of all-scripts vs pod-invoked shows zero .sh references, and no doc/paper cites them as repro steps. **-8,537 lines.** [scripts/]

`yagni:` the `w5_*` evaluation family (`w5_ruler`, `w5_needle`, `w5_hybrid`, `w5_longctx`, `w5_fp16_longctx`, `w5_streamppl`) is a whole second eval harness superseded by `w10_ruler.py` + `w10_frontier.py`; only `scripts/pod/w5_*_8b.sh` + `w7_streamppl_8b.sh` still reference it and those pods are superseded by `w18.sh`/`w19.sh`. Replacement: the w10 pair. **-1,700 lines** (plus the 6 pod .sh below). [scripts/w5_*.py]

`shrink:` four `w*_intervals.py` are **not** independent — `w16` and `w17` both do `from w15_intervals import Z95, wilson` + `from w11_merge import PPL_RE, ROW_RE, TASKS`, and `w18` does `from w15_intervals import TASK_LABEL, wilson`. So scripts/ is already an undeclared library. **None is a superset**: w18 is the most capable (adds paired trials + exact McNemar + marker-free parsing) but depends on w15; w15 is the only one reading a pre-built JSON decision table, w16/w17 parse logs, w18 parses `[trial]` lines. Shorter form: one `stats.py` (wilson, the row regexes, TASK_LABEL, `fmt_md`) + one `build()` per cell-source in `tables.py`. 696 lines → ~250. [scripts/w15_intervals.py:55, w16_intervals.py:26-27, w17_intervals.py:25-26, w18_intervals.py:29]

`shrink:` ten files each re-declare their own regex for the *same* three log row formats (`ppl=`, `acc=/recall=`, `ratio=`): `w10_parse_logs`, `w11_merge`, `w15/w17/w18_intervals`, `w19_{a1_report,a2_misses,dashboard,figures,fork_report}`, `w20_close_report`. Replacement: the three compiled patterns live once next to the emitter (`w10_frontier._log_row`, scripts/w10_frontier.py:758); every reader imports them. [scripts/]

`stdlib:` hand-rolled Wilson score interval. Replacement: `scipy.stats.binomtest(hits, n).proportion_ci(method="wilson")` — scipy is already a runtime dependency and `w18_intervals.py:28` already imports `binomtest` from it. -13 lines and one fewer thing to get wrong. [scripts/w15_intervals.py:55-65]

`yagni:` `w9_frontier.py` (133 l) is a plot-only precursor of `w10_frontier.py` (985 l, which owns `_plot` at line 775); it reads a results file that the w10 arm-set replaced. Replacement: `figures.py`. [scripts/w9_frontier.py]

`shrink:` 29 scripts re-declare `--model` and 33 re-declare `--out-json` with their own defaults; `w10_ruler.py` alone has 42 `add_argument` calls. Replacement: one shared `build_common_parser()` used as an argparse `parents=[...]`. [scripts/]

**Fold map** (for the four-entrypoint target): `pod.py` ← w10_frontier, w10_ruler, w10_longbench, w11_probe, w12_calibrate_qkey, w16_storage, w16_tier2_probe, w19_official_ruler, w19_persist, w20_latency, _paths · `tables.py` ← w10_parse_logs, w11_merge, w15/16/17/18_intervals, w19_a1_report, w19_a2_misses, w19_fork_report, w20_close_report · `figures.py` ← w19_figures, w19_dashboard, w9_frontier · `dump_kv.py` ← capture_kv · `delete` ← the 31 above + the w5_* family.

### src/kvdlra — 13 modules that nothing executable reaches

`delete:` the four ODE integrators `bug.py`, `bug_adaptive.py`, `bug_class.py`, `bug_torch.py`. Confirmed dead: the only callers of `rk2_step`/`bug_step`/`build_operator` outside `integrators/` are `scripts/sigma_decay.py` (never on a pod line) and tests; `bug_class`/`bug_torch` are imported by nothing but their own test. The shipped tracker is `augmented_bug_step`, called at src/kvdlra/cache/bug_cache.py:902,910. Replacement: nothing. **-819 lines.** [src/kvdlra/integrators/]

`delete:` `streaming.py`, the numpy per-token twin of `streaming_torch.py`. Its only runtime entry is `BUGPress(backend="numpy")` (src/kvdlra/press/bug_press.py:224-226) and the sole caller passing that string is `tests/test_bug_press.py:201`. Replacement: nothing; delete the `backend` knob too (src/kvdlra/press/bug_press.py:141,161-162,220) — one implementation, one branch. **-329 lines + ~40 in bug_press.** [src/kvdlra/integrators/streaming.py]

`delete:` `integrators/oja.py` (310 l) and `integrators/frequent_directions.py` (132 l). **Surprise check**: the Week-20 tracker-swap ablation that `scripts/pod/w19.sh:202` actually runs does *not* use these — `w10_frontier.py:923` `--tracker {bug,oja,fd}` routes to `oja_step`/`fd_step`, which are torch ports living inside `streaming_torch.py:307+` (src/kvdlra/cache/bug_cache.py:159). The numpy originals are reachable only from `week2_oja_vs_bug.py`/`w7_fd_ablation.py` (both dead) + tests. Replacement: nothing. **-442 lines.** [src/kvdlra/integrators/]

`delete:` `integrators/streaming_variants.py` (281 l) — one importer, `scripts/w45_integrator_ablation.py:47`, which is dead. [src/kvdlra/integrators/streaming_variants.py]

`delete:` `lowrank.py` (66 l) — importers are `week2_pilot`, `week2_oja_vs_bug`, `w7_fd_ablation`, `sigma_decay`, all dead. [src/kvdlra/lowrank.py]

`delete:` `quant/product_quant.py` (300 l) + the CodeBUG path it feeds (`coord_codebook`, `anchor_rank`, `anchor_seal_absorbs`, `code_budget` — ~55 referencing lines in bug_cache). The only setters are `calibrate_codebook.py:76`, `w7_codebook.py:193`, `w8_carry_drift.py:58`, all dead; Week-8 recorded CodeBUG as a loss. Replacement: nothing. **-355 lines.** [src/kvdlra/quant/product_quant.py, src/kvdlra/cache/bug_cache.py:339,518,594,1828]

`delete:` `quant/qjl.py` (109 l) — exported by `quant/__init__.py:7` and imported by nothing but `tests/test_qjl.py`. [src/kvdlra/quant/qjl.py]

`delete:` `press/turbo_press.py` (107 l) — `TurboQuantPress` setters are `w4_fair`, `w4_hybrid_sweep`, `w5_hybrid`, `w5_longctx`, all in the superseded set. [src/kvdlra/press/turbo_press.py]

`delete:` `utils/hydra_resolvers.py` (20 l) — **zero importers anywhere**, and `hydra` is imported by nothing in the repo. [src/kvdlra/utils/hydra_resolvers.py]

`yagni:` `RETENTION_MODES = ("fifo", "attn", "energy", "lowrank_surprise", "blend")` — five modes, two ever used. No pod .sh passes `--retention`; `w10_frontier.build_arms` hard-codes `"fifo"` and `"lowrank_surprise"`. Replacement: keep those two, delete the `attn`/`energy`/`blend` branches (bug_cache.py:1033-1050 + the `track_surprise`/`score_decay`/`surprise_blend` plumbing they justify). [src/kvdlra/cache/bug_cache.py:166]

`yagni:` `merge=` on `BugStreamingCache` — one setter, `scripts/w7_rank_sweep.py:186` (dead), yet it forces `track_positions` and a guard at bug_cache.py:498. Replacement: drop the parameter. ~20 lines. [src/kvdlra/cache/bug_cache.py:357]

`yagni:` the Q-BUG `w_key` whitening path (~42 lines in bug_cache + `_load_wkey` + `--qwhiten-file` + the `bugSQ` naming branch at w10_frontier.py:255-269). Week-13 closed it as bounded and default-off; no surviving pod passes `--qwhiten-file`. Keep only if the paper still reports the ablation. [src/kvdlra/cache/bug_cache.py, scripts/w10_frontier.py:182]

`yagni:` `cache/morph_cache.py` (449) + `cache/shadow_cache.py` (582) are reachable only from `scripts/pod/w11_baselines.sh` (`--methods morph shadow`); `w18.sh`/`w19.sh` never ask for them. Not a delete — they are the paper's MorphKV/ShadowKV baselines — but if the paper's baseline table no longer cites them, that is 1,031 lines. Decide before folding pods. [src/kvdlra/cache/]

`native:` `press/compat.py` is a live monkeypatch of kvpress's prefill hook (imported by 14 scripts incl. `w10_ruler.py:56`). Keep, but it is a shim for a pinned `kvpress==0.5.1` — recheck whether 0.5.1 still needs it before resubmission. [src/kvdlra/press/compat.py]

### scripts/pod

`delete:` 12 superseded launcher/scrape scripts — `w5_*.sh` (5), `w7_streamppl_8b.sh`, `w10_gpu.sh`, `scrape_w10.sh`, `w11_*.sh` (4), `w12_probe_r128.sh`. Their arms are all re-expressed in `w16.sh`/`w18.sh`/`w19.sh`. **-1,044 lines.** [scripts/pod/]

`shrink:` three watchdogs — `w18_watchdog.sh` (53), `w18_watchdog2.sh` (27), `w19_watchdog.sh` (58) — differ by ~40 changed lines each, all the same poll-log-then-destroy loop. Replacement: one `watchdog.sh` taking the pod id and the log-tail size (the "empty-fetch → 5000-line fallback" fix from commit 847778f lives in only one of them). **-78 lines.** [scripts/pod/]

`yagni:` `w18_finalizer.sh` (14) and `w18_g4fallback.sh` (22) are one-shot recovery hacks for a specific 2026-08 pod failure. Replacement: nothing. [scripts/pod/]

### Top-level litter and docs

`delete:` `mkdocs.yml` (tracked) + the four mkdocs dev deps. **Nothing publishes it**: `git branch -r` lists only `main`, `week1-scaffold`, `week2`, `week3`, `week7` — no `gh-pages`; `.github/workflows/{ci,latex}.yml` never mention mkdocs; `site_url` points at a GitHub Pages URL that has no build. [mkdocs.yml]

`delete:` already-gitignored files still sitting on disk — `handover.md` (37 KB), `explanation_week_{1_2,3_4,5_6}.md`, `next-session-prompt.md`, `compass_artifact_wf-*.md` (44 KB), `dashboards/`, `.DS_Store` (+ the ones in `.github/`, `paper/`, `experiments/`). `git ls-files` returns nothing for all of them. Replacement: `rm`. Zero repo impact, ~100 KB of confusion. [/]

`delete:` `experiments/` — 6 tracked files, two `run.py` from Week 2, superseded by `results/<pod>/` + `docs/`. Nothing imports or references them. **-825 lines.** [experiments/]

`delete:` `figs/` — tracked only via `.gitkeep`, and `.gitignore` already excludes `figs/*.{pdf,png}`. `figures/` is the real one (89 tracked files across `week{1,2,4,5,7,8,9,10,19}/`); the target layout keeps none of it outside `paper/`. Keep only figures `paper/main.tex` actually `\includegraphics`. [figs/, figures/]

`delete:` `docs/` outside `{plan,adr,paper}` — 51 week/handover/kickoff/explainer files + `docs/{notes,board,week10_report}` (997 l) + `docs/reviews/` (4,458 l) + `docs/PLAN.md`. **-13,199 lines** of the 15,452. The reviews are the panel record and probably belong under `docs/adr/` or `docs/paper/`, not deleted — decide per file. [docs/]

`yagni:` `configs/{press,model,eval}/` — three empty directories, zero tracked files, and no `hydra`/`omegaconf` import survives the cuts above. The target layout wants `configs/{arms,tasks,pods}/*.yaml`; that is new construction, not a rename. [configs/]

### tests (reported, not proposed for cutting)

11 test files exist solely to test deletion candidates and would move with their module, not be lost: `test_bug_class` (312), `test_bug_torch` (207), `test_bug_synthetic` (152), `test_bug_adaptive` (142), `test_oja` (277), `test_frequent_directions` (80), `test_streaming` (235), `test_streaming_variants` (151), `test_qjl` (94), `test_product_quant` (163), `test_turbo_press` (53) = **1,866 lines**. Every other test covers a surviving module. No test loads an HF model or hits the network — the only `from_pretrained` reference is in `test_w10_longbench_skip.py`, a skip guard — so the suite is genuinely CPU-only; `test_bug_cache_week7.py` (838 l) and `test_bug_cache_week11.py` (523 l) are the slowest by inspection (dense prefill+decode loops), worth timing but not cutting.

---

## Dependency table

| Dep | Used by |
|---|---|
| torch, transformers, numpy, matplotlib | live, 28-89 files each |
| kvpress==0.5.1 | 15 files (`press/*`, all eval scripts) |
| datasets==2.21.0 | 5 files (`w10_longbench`, `w12_calibrate_qkey`, `capture_kv`, …) |
| scipy>=1.13 | `w18_intervals.py:28` (`binomtest`); should also absorb the hand-rolled `wilson` |
| optimum-quanto, hqq | **no direct import** but load-bearing: backend strings routed via `transformers` QuantizedCache (`src/kvdlra/quant/kivi_cache.py:36`). **Keep.** |
| accelerate==1.13.0 | **UNUSED** — 0 imports, and no `device_map=`/`low_cpu_mem_usage` anywhere |
| huggingface_hub==1.14.0 | **UNUSED** — 0 imports; transitive via transformers |
| wandb==0.26.1 | **UNUSED** — 0 imports (only a stale mypy override + `.gitignore` entry) |
| python-dotenv | **UNUSED** — 0 imports |
| tqdm | **UNUSED** — 0 imports |
| hydra-core==1.3.2 | **UNUSED** — 0 imports |
| omegaconf>=2.3 | only `src/kvdlra/utils/hydra_resolvers.py:12`, itself dead |
| mkdocs-material, mkdocstrings[python], mkdocs-jupyter, pymdown-extensions (dev) | **UNUSED** — nothing builds or publishes docs |
| lm-eval[vllm]==0.4.11 (extra `eval`) | **UNUSED** — 0 imports |
| flash-attn (extra `flash`) | **UNUSED** — 0 imports |

**13 removable.** Also drop the now-dangling `wandb.*` and `flash_attn.*` mypy overrides in `pyproject.toml`.

## Forbidden-word inventory (case-insensitive occurrences)

| Location | `dlra` | `bug integrator` | `honest` | `marquee` | `flagship` |
|---|---|---|---|---|---|
| `src/` | 83 | 7 | 30 | 0 | 1 |
| `scripts/` (.py) | 0 | 0 | 0 | 0 | 0 |
| `scripts/pod/` (.sh) | 58 | 0 | 1 | 11 | 15 |
| `configs/` | 0 | 0 | 0 | 0 | 0 |
| `tests/` | 99 | 6 | 27 | 0 | 4 |
| `paper/main.tex` | 17 | 1 | 32 | 19 | 62 |
| `README.md` | 14 | 0 | 7 | 1 | 0 |
| `docs/` (excl. `docs/plan/`) | 286 | 16 | 400 | 136 | 273 |

Must reach zero in `src/`, `configs/`, `paper/`: **121 src + 131 paper = 252 sites**, nearly all in docstrings and prose — a mechanical pass, except that `dlra` is also the *package name* (`src/kvdlra/`, `pyproject.toml:name`, `mypy_path`, `pythonpath`, 100+ import lines). A rename is a separate, much larger job than a word pass; scope it explicitly before starting. `configs/` is already zero (it is empty). `scripts/` .py is already zero — the 84 hits in `scripts/pod/*.sh` are in the prose comments the pods carry.

---

net: -26,600 lines, -13 deps possible.
(src ~2,850 · scripts ~11,100 · scripts/pod ~1,160 · docs ~13,200 outside `docs/{plan,adr,paper}` · experiments 825 · plus 1,866 test lines that retire *with* their modules, not independently.)
