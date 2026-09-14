# ponytail-after — kv-dlra, repo-wide

Read-only scan, 2026-09-14, branch `lane/L0-repo-cleanup` @ `b5009db` (50 commits on
`ee8c0ab` = tag `paper-v1-archive`). Same format and same scope as the Day-1 BEFORE
report (`ponytail-audit.md`): over-engineering only. Trust-boundary validation (the
launch refusals and `pod.py check` rules), data-loss guards, provenance (manifests,
archives, anti-drift pins) and tests are out of scope and are never proposed for
cutting. Findings are proposals: nothing here is applied, the orchestrator decides.

## LOC, before and after

BEFORE = `2719a50` (the Day-1 measurement point: `ee8c0ab` plus the plan pack, which
added `docs/plan/` and changed no code). AFTER = `b5009db`.
Commands: `find <dir> -name '*.py' | xargs wc -l`, `.sh` for `scripts/pod`, `.md` for
`docs`, `git ls-files | wc -l` for the file count.

| Location | files before | lines before | files after | lines after | Δ lines |
|---|---|---|---|---|---|
| `src/` (.py) | 30 | 7,398 | 29 | 6,568 | **-830** |
| `scripts/` (.py) | 62 | 16,785 | 5 | 2,183 | **-14,602** |
| `scripts/pod/` (.sh) | 23 | 1,951 | 2 | 167 | **-1,784** |
| `tests/` (.py) | 46 | 9,555 | 43 | 8,090 | **-1,465** |
| **`src`+`scripts`+`tests` (.py)** | **138** | **33,738** | **77** | **16,841** | **-16,897** |
| `docs/` (.md) | 88 | 15,452 | 74 | 12,413 | -3,039 |
| — of which `docs/plan/` | 15 | 2,253 | 68 | 12,308 | +10,055 |
| — of which outside `docs/plan/` | 73 | 13,199 | 6 | 105 | **-13,094** |
| `configs/` (.yaml) | 0 | 0 | 64 | 930 | +930 |
| tracked files (`git ls-files`) | 589 | | 533 | | -56 |

Two numbers that look smaller than the work behind them:

* `src/` sheds only 830 net because Task 8 moved the eval harnesses *into* it: about
  3,570 lines of dead modules left (integrators, numpy twins, product quant, qjl, morph)
  and about 2,740 lines of harness arrived (`eval/{frontier,ruler,runner,records,
  longbench,persist,data,official_ruler,latency,config,stats}.py`).
* `docs/` sheds only 3,039 net because the lane's own plan and evidence live under
  `docs/plan/`. Outside it, `docs/` went from 13,199 lines to 105 (the six generated
  `docs/paper/tables/*.md`).

The brief's BEFORE file count of 931 does not reproduce: `git ls-files | wc -l` at
`ee8c0ab` is 573 and at `2719a50` is 589. The measured pair is used above.

## Dependencies

`pyproject.toml`, counting every named requirement.

| group | before | after |
|---|---|---|
| runtime `dependencies` | 16 | 11 |
| `dev` extra | 9 | 6 |
| `eval` / `flash` extras | 2 | 0 |
| **total** | **27** | **17** |

Removed (11): `accelerate`, `huggingface_hub`, `wandb`, `python-dotenv`, `tqdm`,
`mkdocs-material`, `mkdocstrings[python]`, `mkdocs-jupyter`, `pymdown-extensions`,
`lm-eval[vllm]`, `flash-attn`. Added (1): `ninja` (dev — torch's C++ extension loader
shells out to the executable; without it the quant tests cannot JIT on a clean clone).
`omegaconf` was on the Day-1 removable list and is now load-bearing
(`src/kvdlra/eval/config.py` is the whole config system). `hydra-core` was on that list
and is still installed with zero importers — see the findings.

## ponytail-debt ledger

`grep -rnE '(#|//) ?ponytail:' . --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.superpowers`

One hit, and it is the convention sentence inside `docs/plan/cleanup/ponytail-debt.md`
itself. **0 markers in code, 0 with no trigger.** The ledger is still clean — which is
also why the four "written ahead of its caller" findings below have nowhere to be
recorded today.

---

## Findings

`delete:` `TurboQuantPress`, the Week-4 quantization fairness control. No arm YAML
builds it (no `kind: press` file names it, `grep -rn turbo configs/` is empty), no
entrypoint imports it, and no arm string in `results/paper-v1/**/{trials,cells}.jsonl`
contains `turbo` — the composed baselines that survived are the `*_kivi*` arms. Its only
reacher is its own test, so `tests/test_reachability.py` passes while nothing can run it.
Task 9 kept it under the "a `pod-sh` bucket is not deleted" rule; every `w5_*.sh` that
carried that edge was deleted in the same task, so the rule no longer holds it.
Replacement: nothing — `quant/polar.py` (`PolarQuant`) is the live quantizer, used by
`cache/bug_cache.py` and `baselines/lowrank_press.py`. Caveat for the decision: the
paper's prose cites "SnapKV x TurboQuant" arms, and those records are not in the v1
archive, so the module is the only in-repo trace of them; it survives byte-for-byte at
`paper-v1-archive`. **-107 lines** (+53 test lines that retire with it).
[src/kvdlra/baselines/turbo_press.py]

`delete:` `accounting.shadow_bandwidth_per_token` and `accounting.audit_matched` — zero
callers anywhere, not even a test (`grep -rn` over `src scripts tests` returns only the
`def` lines). Replacement: nothing; `shadow_footprint`'s docstring already states the
offload bandwidth rule the first was meant to print, and `assert_all_within` is the
matched-memory gate the second duplicated in dict form. **-16 lines** with their banner
comments. [src/kvdlra/accounting.py:377, src/kvdlra/accounting.py:494-503]

`stdlib:` the hand-rolled 8-entry LRU in `_RopeAngles` — an `OrderedDict`, a
`move_to_end` on hit, and a `while len(...) > 8: popitem(last=False)` on miss.
Replacement: `@lru_cache(maxsize=8)`; the key `(start, length, device)` is already
hashable and already built by hand. **-12 lines** and one eviction rule fewer to get
wrong. [src/kvdlra/cache/bug_cache.py:163,168-178]

`shrink:` `scripts/tables.py` and `scripts/figures.py` each declare their own
`_read(pod, name)` + `@cache`-wrapped `_trials`/`_cells` (and `_ppl`) over the same
`results/paper-v1/<pod>/*.jsonl` layout; they differ only in whether a missing file
raises or returns `[]`. Replacement: one loader next to `read_jsonl` in
`kvdlra.eval.records` taking that strictness as its one argument. **-12 lines**.
[scripts/tables.py:210-227, scripts/figures.py:97-118]

`yagni:` `seed_everything(..., deterministic=False)` — the only caller
(`scripts/dump_kv.py:282`) never passes it, so the deterministic-algorithms branch, the
cuBLAS workspace variable and the two cudnn flags have never run in this tree.
Replacement: drop the parameter; a lane that needs bit-reproducibility adds it back with
the run that proves it. **-8 lines**. [src/kvdlra/util/seed.py:17-35]

`delete:` `hydra-core==1.3.2` from `dependencies` — zero importers (`grep -rn 'import
hydra'` over `src scripts tests configs` is empty). The config system is `omegaconf`
alone: three flat directories, merge-onto-schema, `to_object`. Replacement: nothing.
**-1 dep.** [pyproject.toml]

### Written ahead of its caller — mark, do not cut

Four places carry working, tested code that no live path reaches, each one a named
deliverable of a *later* gate. Deleting them is churn and re-writing them is waste; the
lazy move is the `ponytail:` marker the ledger above has no entries for, so they read as
deferred rather than live. Counted as 0 in the net.

`yagni:` `eval/persist.py` (192 l) has no `generator:` in `eval.runner.GENERATORS` and no
`configs/tasks/*.yaml`, so the cold-start bench cannot be re-run from a config; the
figure it feeds (`figures.fig_coldstart`) parses archived line-file rows through a regex
that lives in `figures.py` because no record type matches. This is exactly the state
Task 9 flagged for `latency.py`/`storage.py` and Task 10 resolved (R31) — and the same
split applies: the paper cites its numbers, so wire it (a `"persist": persist` entry plus
a task YAML, mirroring `latency`) rather than delete it. **+3 lines, not -192.**
[src/kvdlra/eval/persist.py, src/kvdlra/eval/runner.py:45-49]

`yagni:` `eval/stats.py`'s `holm`, `bh`, `paired_bootstrap`, `tost` (~25 l) — no table or
figure calls any of them; only `tests/test_stats.py` does. G3 ("`make tables` renders
Holm-corrected retrieval + TOST perplexity") is the caller they are waiting for.
[src/kvdlra/eval/stats.py:42-72]

`yagni:` `PodCfg.gpu_budget_h` — declared, set to `0.0` in all 25 pod YAMLs, and read by
no code at all. G3's "≤ 50 GPU-h in manifest" is the reader it is waiting for; until
then a field nothing reads is a comment with a type annotation.
[src/kvdlra/eval/config.py:77]

`yagni:` `TaskCfg.depths` — set in 0 of 11 task YAMLs, so `ruler.py`'s
`if task.depths and sub == "niah_single"` branch has never fired. G2's balanced-depth
test is its caller. [src/kvdlra/eval/config.py:52, src/kvdlra/eval/ruler.py:414-415]

### Considered and not proposed

* **The repeated `cache:` keys in `configs/arms/*.yaml`.** Six keys are constant across
  all 13 arms that carry a cache block (`absorb_block: 16`, `n_sink: 4`,
  `recent_window: 32`, `hh_select: surprise`, `hh_retain: true`, `seed_hh_warmup: true`).
  Under the usual rule that is "config for a value that never changes" — but here the
  YAML restating a default is what stops a future default change from silently moving an
  archived arm, and `tests/test_config_parity.py` pins each file to the frozen legacy
  kwargs. Provenance, kept.
* **The eight `tests/test_bug_cache*.py` files** (2,666 lines over one module). Tests are
  out of scope, and each file pins a different week's behaviour.
* **`omegaconf`'s merge-onto-schema load.** Replacing it with `yaml.safe_load` plus
  hand-written type checks is more code, not less, and the load is where a wrong key or
  type fails.
* **`baselines/compat.py`.** The Day-1 report asked whether the pinned `kvpress==0.5.1`
  still needs the prefill-hook monkeypatch. It is still live (`eval/runner.py:29` plus
  four tests); re-checking it means bumping kvpress, which is a dependency change, not a
  cleanup.

---

net: -209 lines, -1 dep possible.
(turbo_press 107 + its 53 test lines · accounting 16 · rope LRU 12 · tables/figures
readers 12 · seed flag 8 · one dependency line. The four "written ahead of its caller"
findings are 0 — they take a `ponytail:` marker, not a deletion.)

Against the Day-1 report's `net: -26,600 lines, -13 deps possible`, the tree is now
within 209 lines of lean. Ship.
