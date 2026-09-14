# Deletions ledger — L0 repo cleanup

One row per removed path or precisely-scoped group, with the evidence that
nothing executable reached it, and one row per path that *looked* like a
candidate but was kept (with the reason). Buckets and LOC are quoted from
`docs/plan/cleanup/reachability.md` (§2 full table, §3 candidate lists); `audit`
cites `docs/plan/cleanup/ponytail-audit.md` §"src/kvdlra — 13 modules that
nothing executable reaches".

A *group* row (`docs/week*.md`, the 86 unreferenced `figures/` files, the two
`results/` buckets) stands for a set whose members share one rule and one piece
of evidence; **§"Complete listing"** at the end of this file is the verbatim
`git diff --name-status ee8c0ab..HEAD` output, so every individual path is
greppable and every group row can be expanded.

Started in Task 6; Tasks 7, 8, 9 and 10 append below; Task 11 closes with the
coverage count and the complete listing.

## Task 6 — source-tree moves

No code was deleted by the move commit; every importer was re-pointed
(`grep -rn 'kvdlra\.(integrators|press|utils|lowrank)' src scripts tests` is
empty afterwards).

| from | to | note |
|---|---|---|
| `src/kvdlra/integrators/streaming_torch.py` | `src/kvdlra/tracker/isvd.py` | the shipped tracker (`augmented_bug_step` / `oja_step` / `fd_step`, called from `cache/bug_cache.py`) |
| `src/kvdlra/press/palu_press.py` | `src/kvdlra/baselines/svd_oracle.py` | class `PaluPress` → `SVDOraclePress`; module docstring rewritten to state what it is (per-sequence, per-head truncated SVD of the prefill K/V, sinks exact — an upper bound for static low-rank methods) and what it is not (no weight decomposition, no grouped-head Fisher rank search, no fine-tuning, not computable online) |
| `src/kvdlra/press/bug_press.py` | `src/kvdlra/baselines/lowrank_press.py` | class name `BUGPress` unchanged (L0 renames no class but `PaluPress`); the `backend=` knob and its `StreamingBUG` numpy branch removed |
| `src/kvdlra/press/compat.py` | `src/kvdlra/baselines/compat.py` | live kvpress prefill-hook shim, 14 importers |
| `src/kvdlra/press/turbo_press.py` | `src/kvdlra/baselines/turbo_press.py` | **not deleted** — see "kept despite being a candidate" below |
| `src/kvdlra/utils/seed.py` | `src/kvdlra/util/seed.py` | |
| `tests/test_palu_press.py` | `tests/test_svd_oracle.py` | follows the class rename |
| `tests/test_streaming_torch.py` | `tests/test_isvd.py` | follows the module rename |

New packages: `src/kvdlra/{tracker,baselines,util}/__init__.py` (one-line
docstring each, no re-exports — importers name the module). Deleted with the
moves: `src/kvdlra/press/__init__.py` (9 LOC; the package is empty afterwards).
Its three re-exports were the only reason three modules looked reachable — see
the severance notes below.

Test edits forced by the numpy-backend removal:

- `tests/test_bug_press.py` — the two `("numpy", …)/("torch", …)` parametrizations
  collapse to the torch path (the numpy tolerances went with the backend), and
  `test_quant_requires_torch_backend` becomes `test_quant_bits_guard` (the
  `quant_bits requires backend='torch'` raise no longer exists).
- `tests/test_isvd.py` — `test_block1_matches_numpy_streaming` deleted: its
  reference implementation *was* `StreamingBUG`. The band test now takes its
  upper bound from `blocked_bug_subspace(..., block_size=1)`, the same quantity
  the deleted test pinned the numpy tracker against.

## Task 6 — deletions

| path | LOC | bucket | evidence |
|---|---|---|---|
| `src/kvdlra/integrators/bug.py` | 204 | pod-sh+tests → none after the move commit | reachability §2 L127. Its only non-test importers are `integrators/bug_adaptive.py` (deleted here) and `scripts/sigma_decay.py` (bucket `none`, §3a L152). The pod-sh edge ran `w10_frontier.py:459 → kvdlra/press/__init__.py → bug_press.py:66 → integrators.streaming → bug_adaptive → bug`; the move commit deletes `press/__init__.py` and the `StreamingBUG` import, severing it. audit: "the four ODE integrators … Confirmed dead". |
| `src/kvdlra/integrators/bug_adaptive.py` | 198 | pod-sh+tests → none | reachability §2 L128; same severed chain. Sole importers `integrators/streaming.py` (deleted) + `tests/test_bug_adaptive.py` (deleted). |
| `src/kvdlra/integrators/bug_class.py` | 240 | tests | reachability §3b: kept alive by `tests/test_bug_class.py` only. |
| `src/kvdlra/integrators/bug_torch.py` | 177 | tests | reachability §3b: kept alive by `tests/test_bug_torch.py` only. |
| `src/kvdlra/integrators/streaming.py` | 329 | pod-sh+tests → none | reachability §2 L129. audit: "its only runtime entry is `BUGPress(backend="numpy")` … and the sole caller passing that string is `tests/test_bug_press.py:201`". That backend is removed in the move commit; remaining importers are `tests/test_{streaming,oja}.py` (deleted) and `scripts/{w7_fd_ablation,week2_pilot,week2_oja_vs_bug}.py` (all bucket `none`). |
| `src/kvdlra/integrators/oja.py` | 310 | tests | reachability §3b. The Week-20 tracker swap runs the **torch** `oja_step` inside `tracker/isvd.py` (audit "Surprise check"; `cache/bug_cache.py:160` — the move commit re-sorted the import block, so `:159` is the quant import and `:160` the tracker one), not this numpy original. |
| `src/kvdlra/integrators/frequent_directions.py` | 132 | tests | reachability §3b; same "Surprise check" — the live `fd_step` is the torch port in `tracker/isvd.py`. |
| `src/kvdlra/integrators/streaming_variants.py` | 281 | tests | reachability §3b; audit: "one importer, `scripts/w45_integrator_ablation.py:47`, which is dead". |
| `src/kvdlra/integrators/__init__.py` | 0 | pod-sh+tests | empty file; the package is empty once the eight modules above are gone. Its bucket was ancestor-package semantics only (reachability §"Method"). |
| `src/kvdlra/lowrank.py` | 66 | tests | reachability §3b (kept alive by `tests/test_frequent_directions.py`); audit: "importers are `week2_pilot`, `week2_oja_vs_bug`, `w7_fd_ablation`, `sigma_decay`, all dead". |
| `src/kvdlra/quant/qjl.py` | 109 | pod-sh+tests → none | reachability §2 L140. audit: "exported by `quant/__init__.py:7` and imported by nothing but `tests/test_qjl.py`". The re-export is the whole pod-sh edge (`bug_cache.py:159`/`lowrank_press.py` import `kvdlra.quant`, which executes `__init__`); removing `QJL` from `quant/__init__.py` severs it. `grep -rn QJL src scripts tests` finds no other user. |
| `src/kvdlra/utils/hydra_resolvers.py` | 20 | none | reachability §3a (the only `src/` file in the `none` bucket); audit: "zero importers anywhere, and `hydra` is imported by nothing in the repo". |
| `src/kvdlra/utils/__init__.py` | 0 | pod-sh+tests | empty file; the package is empty after `seed.py` moves out and `hydra_resolvers.py` is deleted. |
| `src/kvdlra/press/__init__.py` | 9 | pod-sh+tests | reachability §2 L131. Deleted **in the move commit** (Task 6a), which is why the move prose above mentions it; it belongs in this table too. Its bucket was entirely its own three re-exports (`BUGPress`, `PaluPress`, `StreamingBUG`): every importer was re-pointed at the module it actually wants (`kvdlra.baselines.lowrank_press` / `kvdlra.baselines.svd_oracle`), so the package is empty afterwards and the `w10_frontier.py:459 -> press/__init__.py -> bug_press.py -> integrators.streaming` pod-sh edge that kept four ODE modules alive is severed at this file. |
| `tests/test_bug_class.py` | 312 | — | tests a deleted module. |
| `tests/test_bug_torch.py` | 207 | — | tests a deleted module. |
| `tests/test_bug_synthetic.py` | 152 | — | tests `integrators/bug.py` (deleted). |
| `tests/test_bug_adaptive.py` | 142 | — | tests a deleted module. |
| `tests/test_oja.py` | 277 | — | tests a deleted module. |
| `tests/test_frequent_directions.py` | 80 | — | tests deleted modules (`frequent_directions.py` + `lowrank.py`). |
| `tests/test_streaming.py` | 235 | — | tests a deleted module. |
| `tests/test_streaming_variants.py` | 151 | — | tests a deleted module. |
| `tests/test_qjl.py` | 94 | — | tests a deleted module. |

### Six dead scripts brought forward from Task 9

Deleting the modules above left these importing code that no longer exists.
Every one is bucket `none` in reachability §3a (imported by nothing, invoked by
no `scripts/pod/*.sh`), and every one exists *only* to exercise a module deleted
above, so there is nothing left for Task 9 to weigh: `mypy` already refuses
`week2_oja_vs_bug.py` once `OjaTracker` resolves to `Any`
(`no-any-return` ×2), and no type-ignore in dead code is worth keeping.

| path | LOC | bucket | evidence |
|---|---|---|---|
| `scripts/week2_oja_vs_bug.py` | 311 | none | §3a; compares `integrators.oja` / `integrators.streaming` / `lowrank`, all deleted. |
| `scripts/w45_integrator_ablation.py` | 368 | none | §3a; its PSI / parallel-BUG arms are `integrators.streaming_variants`, deleted. |
| `scripts/sigma_decay.py` | 207 | none | §3a; plots `integrators.bug.rk2_step` + `lowrank`, both deleted. |
| `scripts/week2_pilot.py` | 203 | none | §3a; drives `integrators.streaming` + `lowrank`, both deleted. |
| `scripts/w7_fd_ablation.py` | 114 | none | §3a; drives `integrators.frequent_directions` + `integrators.streaming` + `lowrank`, all deleted. |
| `scripts/week1_sv_decay.py` | 39 | none | §3a; a 39-line argv shim that `runpy`-executes `sigma_decay` (`week1_sv_decay.py:27,34`) and nothing else. Its output `figures/week1/sv_decay.pdf` is not cited by `paper/main.tex` (the only `\includegraphics` paths there are `figures/week19/{fairquant,one_over_t,coldstart}.pdf`). |

Task 9's script list shrinks by these six. The remaining dead callers of moved
modules (`w4_fair`, `w4_hybrid_sweep`, `w5_*`, `w7_codebook`, `w8_carry_drift`,
`w9_*`, `calibrate_codebook`, …) still import modules that *exist*, so they were
left alone.

`paper/main.tex:629` — the `% source:` comment naming `streaming_torch.py` was
re-pointed at `src/kvdlra/tracker/isvd.py` (same file, new path). No other
reference to a moved module exists outside `src/`, `scripts/`, `tests/`.

## Task 7 — dead paths inside `bug_cache.py` / `w10_frontier.py`

Three features, one commit each, `make test` green after every one (423 -> 397
-> 387 -> 369 passing). `tests/test_golden_cache.py` is unchanged and green at
every step: it is the r64 arm's bit-identity proof and none of these paths reach
it. `tests/test_accounting.py`'s anti-drift pin (`float_equiv() ==
stored_state_numel()`) is green on every surviving config after the R2 prune.

### 7a — the Week-8 CodeBUG codebook path (`9bed011`)

| path | LOC | bucket | evidence |
|---|---|---|---|
| `src/kvdlra/quant/product_quant.py` | 300 | pod-sh+tests -> none | reachability §2 L139. The pod-sh edge was `bug_cache.py:159`'s module-level `ProductQuantizer` import (the `coord_codebook` argument); removing the CodeBUG path in the same commit severs it. Its other importers are the three CodeBUG-only scripts below. audit: `coord_codebook` / `anchor_rank` / `anchor_seal_absorbs` / `code_budget`, ~55 lines, setters only in dead scripts; Week 8 recorded CodeBUG as a **loss** at extreme compression (22.43 vs morph 18.71). |
| `tests/test_product_quant.py` | 163 | — | tests a deleted module. |
| `tests/test_bug_cache_week8.py` | 365 | — | CodeBUG end to end (every test is named `test_codebug_*`). Checked for a salvageable PolarQuant-only assertion, as the task required: there is none. `test_codebug_never_rotates_codes` asserts CodeBUG does **not** enter `_rotate_quant_tier` — a CodeBUG claim — and the variant-D path keeps its own pins in `tests/test_bug_cache_week7.py`. |
| `scripts/w7_codebook.py` | 327 | none | §3a; the Week-7 CodeBUG rank/budget sweep — every arm is `BugStreamingCache(coord_codebook=...)`. |
| `scripts/calibrate_codebook.py` | 193 | none | §3a; fits the PQ off the `_calib_sink` hook, which goes with the path. |
| `scripts/w8_carry_drift.py` | 200 | none | §3a; drives the `_debug_recode_coded` / `_drift_truth` ablation hooks, which go with the path. |

The three scripts are brought forward from Task 9 on the Task-6 precedent
("deleting the modules above left these importing code that no longer exists …
every one exists *only* to exercise a module deleted above"): all are bucket
`none`, and `mypy --strict` rejects them the moment `ProductQuantizer` stops
existing.

### 7b — Q-BUG whitened-key gist (`f64ced4`)

Week 13 closed Q-BUG as **bounded** (both pre-registered bars missed,
default-off) and no arm ever set it: the six bug arm YAMLs pinned `w_key: null`
only so config parity matched the legacy builder's keyword set, and the key is
now simply absent from both sides.

| removed | where |
|---|---|
| the `w_key` constructor argument + per-layer list broadcast; `_whiten_key` / `_unwhiten_key` and their three call sites; the `stored_state_numel` term | `src/kvdlra/cache/bug_cache.py` |
| `bug_footprint(..., w_key=)` and its `aux += n` term | `src/kvdlra/accounting.py` |
| `_load_wkey`, the `bugSQ` naming branch, `--qwhiten-file` | `scripts/w10_frontier.py` (same flag in `w10_ruler.py`, `w10_longbench.py`) |
| dead `qwhiten_file` namespace fields | `scripts/w16_tier2_probe.py`, `tests/test_w10_ruler_quant.py` |
| `w_key: null` | `configs/arms/{isvd_r64_h256_seed,isvd_r64_h256_seed_q4,isvd_r128_h1024_seed_s32,isvd_r256_h1024_seed,oja_r64_h256_seed,fd_r64_h256_seed}.yaml` |
| the `w_key` state entry; the `_whiten_key` wrappers | `scripts/w19_persist.py`, `scripts/w14_second_bypass_probe.py`, `scripts/w15_scorerank_probe.py` |

| path | LOC | bucket | evidence |
|---|---|---|---|
| `tests/test_bug_cache_qbug.py` | 257 | — | the Q-BUG contract in full; nothing in it survives the argument. |

(`tests/test_w12_harness.py` loses its two `qwhiten` cases, ~20 LOC.)

Every `_whiten_key` / `_unwhiten_key` call was an identity when `w_key` was
unset, which it always was, so the surviving path is bit-for-bit unchanged — the
golden test proves it.

`scripts/w12_calibrate_qkey.py` (98 LOC, bucket **pod-sh**, `w11_r128.sh:170`)
is left for Task 9: it is now an orphan (nothing consumes the `.pt` it writes)
but it imports nothing that was removed, and the L0 rule keeps `pod-sh` paths.
`scripts/pod/w11_r128.sh`'s `MODE=qbug` block still passes `--qwhiten-file`; it
is a Week-11 launcher for a run whose results are already archived, and pod
launchers are R1 roots, so it was left untouched — flagged here rather than
edited.

### 7c — `merge=` and the `attn` / `energy` / `blend` retention modes (`e9fa697`)

`RETENTION_MODES` is now `("fifo", "lowrank_surprise")` — the only two any pod
arm ever built. Removed with the three modes: `score_decay`, `surprise_blend`,
the whole attention-score plumbing (`mid_score` / `q_score` / `ring_score` /
`hh_score`, `observe_attention`, `seed_scores`, `_prompt_seed_scores`,
`_rank_normalize`, `_seen_observation` / `_warned_unattached`), the Week-9
correlation probe (`_probe_surprise` / `_probe_sink` / `_probe_record` /
`enable_surprise_probe` — its whole purpose is correlating surprise *against
attention mass*), and `merge=` with `_merge_down` / `mid_weight`.

Ruling R2 (`src/kvdlra/accounting.py`): `_TRACK_SCORE` deleted, `_TRACK_SURPRISE`
narrowed to `("lowrank_surprise",)`, and the `merge`, ring-score, `hh_score` and
`hh_select` terms dropped. The `lowrank_surprise` surprise-buffer accounting is
kept, as required.

Four judgement calls, recorded because a reviewer will ask:

- **`BugStreamingCache.attach` is kept as a no-op context manager.** Nine
  scripts and `tests/test_golden_cache.py:82` wrap a cache in
  `with cache.attach(model):` to stay uniform across `BugStreamingCache`,
  `MorphKVCache`, `ShadowKVCache` and the presses, and that golden test must not
  change. Keeping it is what severs the `morph_cache` import without touching a
  single call site.
- **`hh_select` keeps both values and its validation, but an enabled tier now
  requires `'surprise'`.** `hh_select="attn"` picked heavy hitters by the retired
  mode's EMA scores, so after 7c the old guard
  (`hh_budget > 0 and hh_select == 'attn' and retention != 'attn'`) could only
  ever have let the tier degenerate to an all-zero FIFO. It fails loud instead;
  `hh_select="attn"` survives only as the default on arms with no exact tier,
  where it is a no-op.
- **`seed_hh_warmup`'s fence goes because both fenced-off features do.** It
  rejected `coord_codebook` (removed in 7a) and `merge` (removed here); the
  PolarQuant tier was already explicitly allowed (Week-19). Nothing reachable is
  left to guard, so `test_seed_warmup_rejects_coded_and_merge` (week11) and the
  fence parametrization in `test_bug_cache_seed_regression.py` go with it.
- **The strict xfail
  `tests/test_bug_cache_week15.py::test_seed_scores_chunked_ingest_latent_bug` is
  deleted, not converted.** It characterized a desync in `seed_scores` under
  `retention="attn"` + `attach()` + chunked ingest: **the latent bug leaves with
  the mode** — there is no `seed_scores`, no `ring_score` and no `mid_score` left
  to desync.

Kept untouched, as required: the SLASH `--chunk` invariant in
`w10_frontier.build_arms`, `_rotate_quant_tier`'s exact norm renormalization, the
`hh_select` and `score_rank` validation, and every quant / hh / rank bound.

| path | LOC | bucket | evidence |
|---|---|---|---|
| `scripts/w7_rank_sweep.py` | 326 | scripts-only | reachability §2 L146. Its three importers (`w9_envelope`, `w9_recovery`, `w9_surprise`) are all bucket `none` and are deleted here too, so the bucket empties. The file *is* the matched-memory solver for the removed modes: `coord_for_config` bills `track_score` / `merge` per column and `make_methods` builds `bugA` / `bugMA` / attn-selected `slash` arms. |
| `scripts/w9_envelope.py` | 471 | none | §3a; the Week-9 envelope sweep — `retention="attn"` arms + `coord_for_config`. |
| `scripts/w9_surprise.py` | 355 | none | §3a; the Week-9 GO/NO-GO correlation probe — its only reason to exist is `enable_surprise_probe` on a `retention="attn"` cache, and it also sweeps `surprise_blend`. |
| `scripts/w9_recovery.py` | 368 | none | §3a; the third `w7_rank_sweep` importer. Deleted for the import, not for its own arms. |

Edited rather than deleted (bucket `pod-sh`, which the L0 rule keeps):

- `scripts/w5_streamppl.py` — the `bugA` (attn), `bugE` (energy) and `bugAD`
  variants leave `build_methods` / `solve_bug_variant`, along with
  `--score-decay`. An unknown method name already raises, so
  `scripts/pod/w7_streamppl_8b.sh:37` (`METHODS=full,bug,bugA,morph,snapkvD`)
  now fails loud at arm construction rather than silently measuring something
  else. **Not edited**: that launcher and `w5_streamppl_8b.sh` are R1 roots and
  both `git clone --branch week3/week7` from origin, so this tree's copy is not
  what they execute — the same nominal-edge argument that kept `turbo_press`.
- `scripts/w7_decode_needle.py`, `scripts/w14_second_bypass_probe.py`,
  `scripts/w15_scorerank_probe.py` — one-line signature/keyword follow-ons.

### Task 7 — not done: `morph_cache.py`

The task assigned `src/kvdlra/cache/morph_cache.py` (+ `tests/test_morph_cache.py`)
to 7c. **It was not deleted; the owner moves to Task 9.** Task 6's note counted
only the `src/` importers. The actual blast radius:

| holder | what it needs |
|---|---|
| `src/kvdlra/cache/__init__.py:4` | re-exports `MorphKVCache` / `MorphKVLayer` |
| `src/kvdlra/accounting.py` | `morph_footprint` |
| `scripts/w10_frontier.py`, `w10_ruler.py`, `w10_longbench.py` | a live `morph` arm, `--morph-keeps`, and `morph` in `--methods` choices (three pod-sh scripts) |
| `scripts/w16_tier2_probe.py` | a `morph_keeps` namespace field |
| **`scripts/w5_streamppl.py`** | bucket **pod-sh**; MorphKV is its *comparison arm* (`morph`, `snapkvD`) — the whole point of the Week-5/6/7 Axis-B measurement |
| `scripts/w5_decode_validate.py`, `scripts/w7_decode_needle.py` | bucket `none`; both build a `MorphKVCache` |
| `tests/{test_accounting,test_w10_ingest,test_w10_score_mode,test_morph_cache}.py` | all build one |

Task 7's mandate is dead paths *inside* `bug_cache.py` and `w10_frontier.py`, and
it discharged the part that actually blocks the deletion: after 7c nothing under
`src/` imports `morph_cache`, and `_aggregated_attention_row` /
`_window_attention_rows` are defined **in** `morph_cache.py` and used by
`MorphKVCache` itself, so the module is self-contained and fully working. What
is left is a scripts-and-baseline decision — gutting a `pod-sh` script's
comparison arm — which is Task 9's call, not a cleanup side effect. Per the L0
rule ("any path the report marks `pod-sh` is NOT deleted — report the
discrepancy"), it is reported here rather than guessed at.

`scripts/w10_frontier.py`'s `morph` arm branch and `--morph-keeps` were therefore
left alone: the task made that removal conditional ("once `morph_cache.py` is
deleted"), and it is not.

## Kept despite being a candidate

| path | LOC | why it stayed | owner |
|---|---|---|---|
| `src/kvdlra/cache/morph_cache.py` (+ `tests/test_morph_cache.py`) | 449 + 260 | The paper-row rule is satisfied (`git grep -n -i morph paper/main.tex` prints nothing). Task 7 severed the `src/` import — `bug_cache.py:158` no longer borrows `_aggregated_attention_row` / `_window_attention_rows` — **but the module is still not free-standing**; see "Task 7 — not done: `morph_cache.py`" for the blast radius Task 6's note did not count. Bucket `pod-sh+tests` (reachability §2 L124). | Task 9 (was Task 7) |
| `src/kvdlra/press/turbo_press.py` → `src/kvdlra/baselines/turbo_press.py` (+ `tests/test_turbo_press.py`) | 107 + 53 | **Reachability/brief discrepancy.** The brief lists it for deletion; the report buckets it `pod-sh+tests` (§2 L135) and the task rule is "any path the report marks `pod-sh` is NOT deleted — report the discrepancy". The pod-sh edge is real but nominal: `scripts/w5_hybrid.py:54` and `scripts/w5_longctx.py:58` import `TurboQuantPress` and are R1 roots (reachability §1), yet all three pod files that invoke them (`scripts/pod/w5_{confirm_8b,hybrid_needle_8b,longctx_8b}.sh:13`) run `git clone --depth 1 --branch week3` — they execute *week3's* copy of those scripts, never this tree's. So the delete is safe but out of this task's rule; the module was moved with the rest of `press/` instead, which is what lets `src/kvdlra/press/` be removed. | Task 9 (deletes `w4_fair`, `w4_hybrid_sweep`, `w5_hybrid`, `w5_longctx` and the `w5_*.sh` launchers) |
| `src/kvdlra/cache/shadow_cache.py` | 582 | ShadowKV baseline; its v1 rows are in `make tables`. | kept |
| `src/kvdlra/quant/{polar,kivi_cache}.py` | 146 + 172 | live: `PolarQuant` in `bug_cache.py`/`lowrank_press.py`, `kivi_cache` in `w10_frontier.py`/`w19_persist.py`. | kept |

## LOC

`find src scripts tests -name '*.py' | xargs wc -l | tail -1`

| point | total |
|---|---|
| before Task 6 (`0815997`) | 36,907 |
| after Task 6 (`0e32bf6`) | 31,896 |
| after Task 7a (`9bed011`) | 30,125 |
| after Task 7b (`f64ced4`) | 29,734 |
| after Task 7c (`e9fa697`) | 27,226 |
| after the 7 self-review pass | 27,224 |
| **delta, Task 6** | **-5,011** |
| **delta, Task 7** | **-4,672** |
| **delta, cumulative** | **-9,683** |

# --- Task 8 ---------------------------------------------------------------

## Task 8 --- the CLI shells the eval port replaced

Task 8 moved the eval harnesses into `kvdlra.eval` and replaced the argparse
builders with `build_arm(ArmCfg, model, t)`. Three of those moves changed enough
of the file to fall under git's rename-similarity threshold, so the cumulative
`ee8c0ab..HEAD` diff shows them as a delete plus an add rather than an `R`; a
fourth path is a test whose subject no longer exists. Recorded here so no `D` row
in §"Complete listing" is unaccounted for.

| path | LOC | commit | evidence |
|---|---|---|---|
| `scripts/perplexity_sweep.py` | 322 | `4de82c5` | **split.** The WikiText loader, the natural-text sentence pool and the two frozen word lists moved to `src/kvdlra/eval/data.py`, bodies unchanged (they produced every archived number). What is gone is the CLI sweep around them, which `configs/tasks/ppl_*.yaml` + `kvdlra.eval.runner.run_pod` replace. |
| `scripts/w16_storage.py` | 229 | `4de82c5` | moved to `src/kvdlra/eval/storage.py`, then deleted in Task 10 (R31) --- see the Task-10 section for why the workspace ratio was superseded by `latency.py`'s measured `kv_peak_gb`. |
| `scripts/w19_official_ruler.py` | 195 | `4de82c5` | moved to `src/kvdlra/eval/official_ruler.py`; the 30-odd `add_argument` calls and the `main()` around the harness are what did not come with it. The module is live: `generator: official_ruler` in `kvdlra.eval.runner.GENERATORS`, driven by `configs/tasks/ruler_official_16k.yaml`. |
| `tests/test_w10_longbench_skip.py` | 75 | `9615e5d` | its subject is gone. It drove `w10_longbench.run(argparse.Namespace)` with a stubbed `load_dataset` that raised for the first task, asserting the axis loop continued. There is no axis loop now: `longbench.run_trial(arm, model, tok, task, sub, ...)` runs one trial, and the try/except moved up to `kvdlra.eval.runner:223-229`. The behaviour changed deliberately --- a task that cannot load is no longer skipped quietly; it lands in `trials.jsonl` with `error=` and `scripts/pod.py check` fails the pod (R29, covered by `tests/test_pod_run_records_errors.py`). |

# --- Task 9 ---------------------------------------------------------------

## Task 9a --- scripts, pod launchers, morph, figures, stale docs

Rules applied: R21 (morph), R22 (figures), R23 (docs), R24 (scripts). The
buckets and LOC are the `docs/plan/cleanup/reachability.md` §2 values; the
`pod-sh` edges all die in the same commit, because the pod launchers that
carried them are deleted here too (R24) and, as Tasks 6--7 established, those
launchers `git clone --branch week3/week7` from origin and never executed this
tree's copy anyway.

### `scripts/*.py` --- every file outside the five entrypoints

Surviving entrypoints: `_paths.py`, `pod.py`, `tables.py`, `figures.py`,
`dump_kv.py` (+ `make_arxiv.sh`, kept --- see "kept" below).

| path | LOC | bucket | evidence |
|---|---|---|---|
| `scripts/generate_with_press.py` | 269 | scripts-only | reachability §2: reached only from other `scripts/*.py`, all deleted here |
| `scripts/w10_parse_logs.py` | 80 | none | reachability §2: no pod launcher invokes it and no test imports it |
| `scripts/w11_merge.py` | 165 | tests | decision-table merger; superseded by `scripts/tables.py build` |
| `scripts/w11_probe.py` | 203 | pod-sh | reachability §2 `pod-sh`: its only live edge was a `scripts/pod/*.sh` launcher, deleted in this commit (R24) |
| `scripts/w12_calibrate_qkey.py` | 98 | pod-sh | reachability §2 `pod-sh`: its only live edge was a `scripts/pod/*.sh` launcher, deleted in this commit (R24) |
| `scripts/w12_qbug_probe.py` | 299 | scripts-only | reachability §2: reached only from other `scripts/*.py`, all deleted here |
| `scripts/w13_tracka_probe.py` | 557 | none | reachability §2: no pod launcher invokes it and no test imports it |
| `scripts/w13_trackb_bypass.py` | 236 | none | reachability §2: no pod launcher invokes it and no test imports it |
| `scripts/w13_trackc_probe.py` | 282 | none | reachability §2: no pod launcher invokes it and no test imports it |
| `scripts/w13_trackx_angles_probe.py` | 298 | none | reachability §2: no pod launcher invokes it and no test imports it |
| `scripts/w13_trackx_rank_probe.py` | 231 | none | reachability §2: no pod launcher invokes it and no test imports it |
| `scripts/w14_second_bypass_probe.py` | 628 | none | reachability §2: no pod launcher invokes it and no test imports it |
| `scripts/w15_intervals.py` | 161 | tests | folded into `kvdlra.eval.stats` (Task 8); no surviving test imports it |
| `scripts/w15_scorerank_probe.py` | 435 | none | reachability §2: no pod launcher invokes it and no test imports it |
| `scripts/w16_intervals.py` | 159 | none | folded into `kvdlra.eval.stats` (Task 8) |
| `scripts/w16_tier2_probe.py` | 228 | pod-sh | reachability §2 `pod-sh`: its only live edge was a `scripts/pod/*.sh` launcher, deleted in this commit (R24) |
| `scripts/w17_intervals.py` | 164 | none | folded into `kvdlra.eval.stats` (Task 8) |
| `scripts/w18_intervals.py` | 212 | tests | folded into `kvdlra.eval.stats` (Task 8); `tests/test_w18_intervals.py` deleted with it |
| `scripts/w19_a1_report.py` | 157 | none | report generator; its output is kept as a narrative report (R17) |
| `scripts/w19_a2_misses.py` | 81 | none | report generator; its output is kept as a narrative report (R17) |
| `scripts/w19_dashboard.py` | 488 | none | report generator; superseded by `scripts/tables.py build` |
| `scripts/w19_fork_report.py` | 135 | scripts-only | report generator; its output is kept as a narrative report (R17) |
| `scripts/w20_close_report.py` | 183 | none | report generator; its output is kept as a narrative report (R17) |
| `scripts/w4_fair.py` | 201 | pod-sh+tests | reachability §2 `pod-sh+tests`: its only live edge was a `scripts/pod/*.sh` launcher, deleted in this commit (R24) |
| `scripts/w4_head_to_head.py` | 185 | none | reachability §2: no pod launcher invokes it and no test imports it |
| `scripts/w4_hybrid_sweep.py` | 182 | pod-sh+tests | reachability §2 `pod-sh+tests`: its only live edge was a `scripts/pod/*.sh` launcher, deleted in this commit (R24) |
| `scripts/w4_needle.py` | 197 | pod-sh+tests | reachability §2 `pod-sh+tests`: its only live edge was a `scripts/pod/*.sh` launcher, deleted in this commit (R24) |
| `scripts/w5_decode_validate.py` | 246 | none | reachability §2: no pod launcher invokes it and no test imports it |
| `scripts/w5_fp16_longctx.py` | 296 | pod-sh | reachability §2 `pod-sh`: its only live edge was a `scripts/pod/*.sh` launcher, deleted in this commit (R24) |
| `scripts/w5_hybrid.py` | 314 | pod-sh | reachability §2 `pod-sh`: its only live edge was a `scripts/pod/*.sh` launcher, deleted in this commit (R24) |
| `scripts/w5_longctx.py` | 360 | pod-sh | reachability §2 `pod-sh`: its only live edge was a `scripts/pod/*.sh` launcher, deleted in this commit (R24) |
| `scripts/w5_needle.py` | 264 | pod-sh | reachability §2 `pod-sh`: its only live edge was a `scripts/pod/*.sh` launcher, deleted in this commit (R24) |
| `scripts/w5_ruler.py` | 263 | pod-sh+tests | reachability §2 `pod-sh+tests`: its only live edge was a `scripts/pod/*.sh` launcher, deleted in this commit (R24) |
| `scripts/w5_streamppl.py` | 517 | pod-sh+tests | reachability §2 `pod-sh+tests`: its only live edge was a `scripts/pod/*.sh` launcher, deleted in this commit (R24) |
| `scripts/w7_decode_needle.py` | 278 | none | reachability §2: no pod launcher invokes it and no test imports it |
| `scripts/w9_frontier.py` | 133 | none | reachability §2: no pod launcher invokes it and no test imports it |
| `scripts/w9_recall_8b.py` | 138 | none | reachability §2: no pod launcher invokes it and no test imports it |
| `scripts/week2_select_docs.py` | 62 | none | reachability §2: no pod launcher invokes it and no test imports it |

Subtotal: 38 files, 9385 LOC.

### `scripts/pod/*.sh` --- every launcher except `boot.sh` and `watchdog.sh`

R24: the per-week launchers are folded into the two SHA-pinned survivors ---
`boot.sh` (the `--onstart` payload, which now hands off to
`scripts/pod.py run --pod "$POD"` with the arms/tasks read from
`configs/pods/<pod>.yaml`) and `watchdog.sh` (harvest + destroy). Everything a
week-launcher encoded as flags is now a committed pod config, so the launcher
is duplicate, not evidence.

| path | LOC | bucket | evidence |
|---|---|---|---|
| `scripts/pod/scrape_w10.sh` | 24 | pod launcher | pure-shell log harvester (reachability §1) |
| `scripts/pod/w10_gpu.sh` | 84 | pod launcher | week launcher for pods whose arms/tasks are now `configs/pods/*.yaml`; every `scripts/*.py` it invoked is deleted in this commit |
| `scripts/pod/w11_baselines.sh` | 58 | pod launcher | week launcher for pods whose arms/tasks are now `configs/pods/*.yaml`; every `scripts/*.py` it invoked is deleted in this commit |
| `scripts/pod/w11_gpu.sh` | 105 | pod launcher | week launcher for pods whose arms/tasks are now `configs/pods/*.yaml`; every `scripts/*.py` it invoked is deleted in this commit |
| `scripts/pod/w11_probe8b.sh` | 50 | pod launcher | week launcher for pods whose arms/tasks are now `configs/pods/*.yaml`; every `scripts/*.py` it invoked is deleted in this commit |
| `scripts/pod/w11_r128.sh` | 333 | pod launcher | week launcher for pods whose arms/tasks are now `configs/pods/*.yaml`; every `scripts/*.py` it invoked is deleted in this commit |
| `scripts/pod/w11_table.sh` | 65 | pod launcher | week launcher for pods whose arms/tasks are now `configs/pods/*.yaml`; every `scripts/*.py` it invoked is deleted in this commit |
| `scripts/pod/w12_probe_r128.sh` | 50 | pod launcher | week launcher for pods whose arms/tasks are now `configs/pods/*.yaml`; every `scripts/*.py` it invoked is deleted in this commit |
| `scripts/pod/w16.sh` | 219 | pod launcher | folded into `scripts/pod/boot.sh` + `configs/pods/w17_floor*.yaml` (Task 8) |
| `scripts/pod/w18.sh` | 161 | pod launcher | folded into `scripts/pod/boot.sh` + `configs/pods/w18_*.yaml` (Task 8) |
| `scripts/pod/w18_boot.sh` | 94 | pod launcher | folded into `scripts/pod/boot.sh` (Task 8) --- it was the same `--onstart` payload with a `$DRIVER` indirection |
| `scripts/pod/w18_finalizer.sh` | 14 | pod launcher | pure-shell log harvester (reachability §1: invokes no `scripts/*.py`); `scripts/pod.py harvest` replaces it |
| `scripts/pod/w18_g4fallback.sh` | 22 | pod launcher | pure-shell log harvester (reachability §1) |
| `scripts/pod/w18_watchdog.sh` | 53 | pod launcher | folded into `scripts/pod/watchdog.sh` (Task 8) |
| `scripts/pod/w18_watchdog2.sh` | 27 | pod launcher | folded into `scripts/pod/watchdog.sh` (Task 8) |
| `scripts/pod/w19.sh` | 259 | pod launcher | folded into `scripts/pod/boot.sh` + `configs/pods/w19_*.yaml` (Task 8) |
| `scripts/pod/w19_watchdog.sh` | 58 | pod launcher | folded into `scripts/pod/watchdog.sh` (Task 8) |
| `scripts/pod/w5_confirm_8b.sh` | 40 | pod launcher | week launcher for pods whose arms/tasks are now `configs/pods/*.yaml`; every `scripts/*.py` it invoked is deleted in this commit |
| `scripts/pod/w5_fp16_8b.sh` | 38 | pod launcher | week launcher for pods whose arms/tasks are now `configs/pods/*.yaml`; every `scripts/*.py` it invoked is deleted in this commit |
| `scripts/pod/w5_hybrid_needle_8b.sh` | 39 | pod launcher | week launcher for pods whose arms/tasks are now `configs/pods/*.yaml`; every `scripts/*.py` it invoked is deleted in this commit |
| `scripts/pod/w5_longctx_8b.sh` | 32 | pod launcher | week launcher for pods whose arms/tasks are now `configs/pods/*.yaml`; every `scripts/*.py` it invoked is deleted in this commit |
| `scripts/pod/w5_streamppl_8b.sh` | 63 | pod launcher | week launcher for pods whose arms/tasks are now `configs/pods/*.yaml`; every `scripts/*.py` it invoked is deleted in this commit |
| `scripts/pod/w7_streamppl_8b.sh` | 63 | pod launcher | week launcher for pods whose arms/tasks are now `configs/pods/*.yaml`; every `scripts/*.py` it invoked is deleted in this commit |

Subtotal: 23 files, 1951 LOC.
### morph (R21)

`MorphKV` is an eviction baseline whose numbers do not appear anywhere in
`paper/main.tex` (`git grep -n -i morph paper/main.tex` is empty) and no
`configs/arms/*.yaml` declares `kind: morph`, so nothing `make tables` or
`make figures` regenerates depends on it. Task 7 severed the `src/` import that
made it look load-bearing; this commit removes the module and everything that
mirrored it.

| path | LOC | bucket | evidence |
|---|---|---|---|
| `src/kvdlra/cache/morph_cache.py` | 449 | pod-sh+tests | reachability §2 L124. Every `pod-sh` edge ran through `scripts/w10_{frontier,ruler,longbench}.py` (deleted Task 8) and `scripts/w5_streamppl.py` / `w5_decode_validate.py` / `w7_decode_needle.py` (deleted above), plus the `w5_*.sh` / `w7_*.sh` launchers (deleted above). Nothing is left. |
| `tests/test_morph_cache.py` | 260 | test of the deleted module | the only thing it tests is `morph_cache` |

Edited, not deleted (the morph references that survived Task 8):

- `src/kvdlra/cache/__init__.py` — the `MorphKVCache` / `MorphKVLayer` import and
  the two `__all__` entries.
- `src/kvdlra/eval/frontier.py` — the `MorphKVCache` import, the
  `_footprint` `kind == "morph"` branch, and the two type unions
  (`score_streaming`'s `cache:` annotation, the `ingesting` noqa comment).
- `src/kvdlra/eval/ruler.py`, `src/kvdlra/eval/longbench.py` — `"morph"` dropped
  from the `arm["kind"] in (...)` streaming tuple (no config declares it).
- `src/kvdlra/accounting.py` — `morph_footprint` (the formula mirror of
  `MorphKVLayer.stored_state_numel`, which no longer exists) and the two
  docstring mentions.
- `src/kvdlra/cache/shadow_cache.py` — the `:meth:`MorphKVCache.attach`` Sphinx
  cross-reference becomes prose. The two *prose* references to MorphKV as prior
  work (§4 of the module docstring) stay: they describe the published method,
  not our class.
- `tests/test_accounting.py` — `test_morph_footprint_matches_stored_state_numel`
  (the anti-drift pin for a formula that is gone).
- `tests/test_w10_ingest.py` —
  `test_morph_chunked_ingest_lossless_matches_dynamic_cache`; the BUG lossless
  oracle, the bounded-state pin and the mode-restore pin all stay.
- `tests/test_w10_score_mode.py` —
  `test_morph_frozen_scoring_matches_dynamic_cache_when_no_eviction`, the
  `_build(model, kind)` dispatcher and the two `["bug", "morph"]`
  parametrizations (which collapse to the BUG case).

### figures (R22)

Kept — exactly the three files `git show ee8c0ab:paper/main.tex | grep
includegraphics` names:

- `figures/week19/coldstart.pdf`
- `figures/week19/fairquant.pdf`
- `figures/week19/one_over_t.pdf`

Deleted: the other 86 tracked files under `figures/` (`week1`, `week2`,
`week4`, `week5`, `week7`, `week8`, `week9`, `week10`, the three `week19`
`.png` companions, and the `.json` data sidecars) — no `\includegraphics`
names them, and `make figures` regenerates the paper's three from
`results/paper-v1/` into the gitignored `docs/paper/figures/`, so a committed
raster of a superseded figure is a stale copy of nothing the build reads.
Also deleted: `figs/.gitkeep` (the directory is an unused scratch output root;
its `.gitignore` lines go in 9c) and `experiments/` (6 files — four dated
Week-1/2 scratch `README.md`s and two `run.py` prototypes, neither imported by
anything nor invoked by any launcher).

### docs (R23)

`git mv docs/reviews docs/plan/reviews` — the 2026-09-06 / 2026-09-08 review
packets are planning records and belong under `docs/plan/` with the rest
(CLAUDE.md's target layout is `docs/{plan,adr,paper}` and nothing else).

Deleted, 57 files: `docs/week*.md` (49 weekly writeups, kickoffs, handovers and
explainers), `docs/notes/` (5), `docs/board/` (2), `docs/week10_report/` (1),
`docs/index.md`, `docs/reference.md`, `docs/PLAN.md`, `docs/w14-sizing.md`, and
`mkdocs.yml` (the site that published them; its four dev deps go in 9c).
Evidence: cited by nothing executable — no test, no entrypoint, no config and
no `Makefile` target reads any of them; the live plan of record is
`docs/plan/` and the paper is `paper/`. They are recoverable at the tag
`paper-v1-archive` (ee8c0ab).

**Hand-off to Task 10 (R16 wording pass): dangling docstring citations.**
Deleting `docs/week*.md` and `docs/notes/` leaves these *prose* references in
surviving source with no target. Each needs the citation dropped or re-pointed;
none of them affects behaviour, and none has a surviving in-repo equivalent
(the content exists only at the tag), so re-pointing is an editorial call this
task does not make:

| file | reference |
|---|---|
| `src/kvdlra/accounting.py:257` | `docs/week10-plan.md` |
| `src/kvdlra/cache/bug_cache.py:3,4,48,241` | `docs/week5-plan.md`, `docs/notes/streaming-decode-design.md`, `docs/week7-plan.md`, `docs/notes/conventions.md` |
| `src/kvdlra/tracker/isvd.py:6,44,116,133,196` | `docs/notes/conventions.md`, `docs/PLAN.md`, `docs/week17`, `docs/week5.md` |
| `src/kvdlra/eval/frontier.py:434` | `docs/week15-significance.md` |
| `src/kvdlra/baselines/turbo_press.py:19` | `docs/notes/turboquant-rope-interaction.md` |
| `src/kvdlra/baselines/lowrank_press.py:24,31,41,135,273` | `docs/notes/{conventions,rope-pitfall,turboquant-rope-interaction}.md`, `docs/PLAN.md` |
| `scripts/dump_kv.py:77` | `docs/notes/rope-pitfall.md` |
| `scripts/tables.py:41` | `docs/week18-kickoff.md` |
| `tests/test_bug_cache.py:16,110` · `test_bug_cache_week7.py:4` · `test_w17_rankfloor.py:5` · `test_bug_press.py:22` · `test_accounting.py:130` | `docs/week5.md`, `docs/week7-plan.md`, `docs/week11*`, `docs/week17`, `docs/notes/conventions.md` |

Repaired here instead (they cite a *script* deleted in this commit, and the
successor is an in-repo path, so the re-point is mechanical, not editorial):
`src/kvdlra/eval/official_ruler.py` (`scripts/pod/w19.sh` → "the pod
bootstrap"), `src/kvdlra/baselines/turbo_press.py` and
`src/kvdlra/cache/bug_cache.py` (`scripts/w4_fair.py` → dropped),
`tests/test_ruler_template_tail.py` (`scripts/ruler.py` →
`kvdlra.eval.ruler._tail_len`; the `scripts/w16_tier2_probe.py` line dropped),
`tests/test_bug_press.py` (`scripts/generate_with_press.py` → dropped),
`tests/test_golden_cache.py` (`scripts/pod/w18.sh` flags →
`configs/arms/isvd_r64_h256_seed`).

Two anti-drift pins in `tests/test_accounting.py` imported their reference
formula *from* a deleted script (`w5_streamppl.bug_budget_floats`,
`w4_hybrid_sweep.kv_memory_ratio`). They test `kvdlra.accounting`, a surviving
module, so they are kept: each reference formula is transcribed into the test
as a private helper naming the script and the tag it came from. The assertion
is unchanged, so the pin still fails if `accounting` drifts.
`pyproject.toml`'s `pythonpath` comment, which justified `scripts` on the path
by exactly those two imports, is re-worded to the real remaining reason
(`import pod` / `import tables` in the entrypoint tests).

## Task 9b --- `results/` legacy files

Rules applied: R17 (narrative reports move), R18 (JSON dumps and harvest scratch
are deleted unless a `% source:` comment cites them), R19 (the source map).

Everything the paper cites survives: `docs/plan/cleanup/paper-source-map.md` maps
every cited path to where its bytes are now, and nothing was removed until a
`cmp` against its archive copy returned equal. `results/` afterwards holds
exactly `.gitkeep` and `paper-v1/`.

### Moved --- narrative reports (R17)

`git mv results/*.md docs/plan/reports/` --- 23 files, names kept:
`w11-final-tables`, `w12-qbug-summary`, `w13-phase1-summary`, `w13-trackb-design`,
`w13-trackb-summary`, `w14-c1-summary`, `w14-c2-summary`,
`w14-second-bypass-summary`, `w15-complete-summary`, `w15-confirm-summary`,
`w15-ruler-intervals`, `w16-ruler-intervals`, `w17-ruler-intervals`,
`w18-g1-llama-ruler-intervals`, `w18-g1-mistral-ruler-intervals`,
`w18-g1-qwen-ruler-intervals`, `w18-g1-report`, `w18-g1-ruler-intervals`,
`w19-a1-report`, `w19-a2-flagship-misses`, `w20-close-report`, `w20-fork-report`,
`w3-parity`. Plus `results/w18_harvest/quant-findings.md` →
`docs/plan/reports/quant-findings.md` (same reason: it is the Week-18 quant
narrative, and `paper/main.tex:845` cites it as the mechanism behind a table).

These are prose about runs, not run records; `results/<pod>/` is for records
(`manifest.json`, `trials.jsonl`, …) and everything under `docs/plan/` is the
planning trail. Four of them are cited by `paper/main.tex` (`w18-g1-report`,
`w19-a1-report`, `w19-a2-flagship-misses`, `w20-fork-report`) --- see the source
map.

### Archived, then deleted --- the 11 cited files with no archive copy yet

Byte-copied into the archive (each verified with `cmp` before removal):

| from | to | cited at |
|---|---|---|
| `results/w18-g4-marquee-contrasts.json` | `results/paper-v1/w18-g4-llama/raw/` | `main.tex:559` (the McNemar contrasts behind `tab:vt`) |
| `results/w19_intervals/a1-{llama,mistral,qwen}-ruler-intervals.{json,md}` | `results/paper-v1/w19-a1-{llama,mistral,qwen}/raw/` | `main.tex:845` (`results/w19_intervals/*.json`) |
| `results/w19_intervals/a2-llama-ruler-intervals.{json,md}` | `results/paper-v1/w19-a2-llama/raw/` | `main.tex:845`, `:938` |
| `results/w18-env-provenance.txt`, `results/w19-env-provenance.txt` | `results/paper-v1/_cited-extras/raw/` | `main.tex:405`, `:406`, `:1198` --- the body's reproducibility paragraph, not a `% source:` comment |

`results/paper-v1/_cited-extras/` is new: the two provenance files are evidence
`main.tex` points at but no `convert-v1` conversion produced, so no pod owns
them. It carries a `README.md` instead of a `manifest.json`, and
`scripts/pod.py check` skips archive directories, so `make check` is unaffected.

### Deleted --- already byte-archived (109 files)

Every `results/*-lines.txt`, `results/w18_pertrial/*-trials.txt` and
`results/w19_pertrial/*-trials.txt`: `scripts/tables.py convert-v1` already
copied each one verbatim to `results/paper-v1/<pod>/raw/<basename>` and recorded
it in that pod's `manifest.json` (`source_files[].raw`). Re-verified file by file
with `cmp` immediately before `git rm`; 109/109 identical, 0 differing, 0
missing. The tables and figures read the converted records
(`trials.jsonl` / `cells.jsonl` / `ppl.jsonl`), never the line files, so
`make tables` stays diff-clean.

### Deleted --- cited by nothing (91 files)

R18: no `% source:` comment in `paper/main.tex` names them, neither
`scripts/tables.py` nor `scripts/figures.py` reads them (both open only
`results/paper-v1/`), and no `\texttt{results/...}` in the body names them
either. Recoverable at the tag `paper-v1-archive` (ee8c0ab).

| group | count | what |
|---|---|---|
| `results/w{3,4,5,7,8,9}-*.json` + `w8-codebook.pt` | 47 | Week-3 to Week-9 probe/sweep dumps, written by `scripts/w{4,5,7,8,9}_*.py` --- all deleted in 9a |
| `results/w1{0,1,2}-*.json` | 15 | Week-10/11/12 frontier and probe dumps, and the `w11-decision-table*.json` / `w16-decision-table.json` / `w17-decision-table.json` merges that `scripts/tables.py build` replaces |
| `results/w1{3,4,5}-*.json` | 9 | the Week-13 track probes, the Week-14 second-bypass facts, the Week-15 e2e/score-rank probes |
| `results/w1{5,6,7,8}-*-ruler-intervals.json` | 7 | the interval JSONs the deleted `w1*_intervals.py` emitted; `kvdlra.eval.stats` recomputes them from the archived records, and the `.md` renderings moved to `docs/plan/reports/` |
| `results/w18_checkpoint/` | 6 | mid-run harvest snapshots (3 partial `*-ruler-lines.txt` + 3 `*.raw.log`), superseded by the completed `results/w18-{llama,mistral,qwen}-lines.txt` that **are** archived |
| `results/w18_harvest/{DONE,G4DONE,SUMMARY}.txt` | 3 | watchdog completion markers and a derived extract of the g1 cells |
| `results/w19_harvest/` | 3 | `done.txt` / `pods.txt` (the watchdog's instance list, already gitignored by `results/*/pods.txt`) and `a3-llama.raw.superseded`, a raw log the a3 re-run replaced |
| `results/w8-codebook.pt` | (in the first row) | an 11 KB torch tensor from the Week-8 codebook probe; the CodeBUG fork was closed as bounded and nothing loads it |

Kept: `results/.gitkeep` (the directory is where a new pod writes) and all of
`results/paper-v1/`.

## Task 9c --- dependencies, `.gitignore`, `README.md`

### `pyproject.toml` (R20, R25)

`requires-python = ">=3.10"` → `">=3.11"`: the code already needs it
(`tomllib` in `scripts/pod.py`, `hashlib.file_digest` in `scripts/dump_kv.py`
are both 3.11 stdlib). CI and `make env` keep pinning 3.12.

Removed --- each verified unimported by
`grep -rn 'import <m>\|from <m>' src/ scripts/ tests/ configs/ Makefile .github/`,
which returns 0 hits for every one:

| dep | where it lived | why it goes |
|---|---|---|
| `accelerate==1.13.0` | runtime | 0 imports; the pods load models with `transformers` directly (`device_map` is never used) |
| `huggingface_hub==1.14.0` | runtime | 0 imports; `transformers` pulls its own |
| `wandb==0.26.1` | runtime | 0 imports; no run was ever logged to it. Its `.gitignore` entry and its `[[tool.mypy.overrides]]` module go too |
| `python-dotenv` | runtime | 0 imports; the pod reads plain `-e` env vars |
| `tqdm` | runtime | 0 imports; the pods print marker lines the watchdog greps, not bars |
| `eval = ["lm-eval[vllm]==0.4.11"]` | extra | 0 imports; RULER/LongBench are `kvdlra.eval`, and the official RULER path shells out to NVIDIA's generator |
| `flash = ["flash-attn"]` | extra | 0 imports; `attn_implementation` is never set to flash. Its mypy override goes too |
| `mkdocs-material==9.7.6`, `mkdocstrings[python]==1.0.4`, `mkdocs-jupyter`, `pymdown-extensions` | dev | the site they built is `mkdocs.yml` + `docs/index.md` + `docs/reference.md`, all deleted in 9a (R23) |

Added: `ninja` to `dev`. `torch`'s C++ extension loader shells out to the `ninja`
**executable**, so without it `optimum-quanto` cannot JIT `quanto_cuda.so` and the
quant tests fail on a clean clone. The `Makefile` already puts the venv's `bin/`
on `PATH` for exactly this reason; the dependency is what puts `ninja` there.

Kept, deliberately: `hydra-core` + `omegaconf` (the config layer under
`kvdlra.eval.config`), `optimum-quanto` and `hqq` (the quant backends the KIVI
arm names as strings, so grep finds no import), `datasets`, `scipy`,
`matplotlib`, `kvpress`, `transformers`, `torch`, `numpy`.

Two follow-ons from the 3.11 floor, both `ruff --fix` output: `scripts/pod.py`
sorts `tomllib` into the stdlib block (it *is* stdlib at 3.11) and uses
`datetime.UTC` instead of `datetime.timezone.utc`. `[tool.ruff] target-version`
moves `py310` → `py311` to match. The `[project] description` is reworded to
match the README's first paragraph.

### `.pre-commit-config.yaml`

`trailing-whitespace` and `end-of-file-fixer` now `exclude: ^results/paper-v1/`.
Caught in 9b: the end-of-file hook silently stripped a trailing blank line from a
verbatim archive copy, which is precisely the byte-identity the archive exists to
guarantee. The file was restored from the pre-deletion blob and re-verified with
`cmp`.

### `.gitignore` (R25)

Removed: `wandb/` (no longer a dependency), `figs/*.pdf` / `figs/*.png` /
`!figs/.gitkeep` (the directory is deleted). The `compass_artifact_*.md` comment
loses its `docs/PLAN.md` reference (that file is deleted; the on-disk artifact
is not).

Kept: `dumps/**` and its `.sha256` / `.gitkeep` negations; the per-pod runtime
patterns `results/*/{pods.txt,done.txt,*.raw,watchdog.pid,status.txt,*.log}`;
`results/{gpu_logs,scratch}/`, `figures/scratch/`; `docs/paper/figures/`;
`uv.lock`; `paper/{arxiv/,arxiv-v1.tar.gz,main.bbl,main.pdf}`; `.venv/` and the
tool caches. Kept too: `handover.md`, `explanation_week_*.md`,
`next-session-prompt.md`, `compass_artifact_*.md`, `dashboards/` --- those files
still exist on disk in the main checkout (see the next section), so their
entries are still load-bearing.

### `README.md` (R26)

Rewritten, 236 → 104 lines: what it is (one paragraph), install, reproduce a
table in three commands, the layout tree, the pod loop
(pre-register → launch → harvest → check), license. No forbidden words, no week
labels, no SHAs; every path it names was checked to exist.

### Untracked on-disk litter --- for the orchestrator, post-merge (R27)

A worktree cannot see these: they are untracked files in the **main** checkout
(`/Users/hari/Desktop/kv-dlra`), all matched by `.gitignore`, so no commit can
remove them. Delete them there after the merge:

| path | what it is |
|---|---|
| `handover.md` | rolling session handover note |
| `explanation_week_1_2.md`, `explanation_week_3_4.md`, `explanation_week_5_6.md` | narrative explainers |
| `next-session-prompt.md` | scratch prompt |
| `compass_artifact_*.md` | the raw planning artifact `docs/PLAN.md` was derived from |
| `dashboards/` | session-generated research dashboards |
| `results/gpu_logs/` | raw `vastai logs` tails; the committed evidence for a pod is its records + manifest |
| `results/scratch/` | local smoke-run output |
| `**/.DS_Store` | macOS metadata |

Once those are gone, the `.gitignore` lines that existed only to hide them go
with them --- there is nothing left to ignore, and a rule for a file that cannot
come back is a rule nobody can check: line 23 (`compass_artifact_*.md`), lines
34--38 (`handover.md`, the three `explanation_week_*.md`,
`next-session-prompt.md`), line 39 (`results/gpu_logs/`), line 42
(`dashboards/`) and line 49 (`results/scratch/`). **Keep** `.DS_Store` (line 20:
macOS writes it again on every folder open) and every build-output rule ---
`figures/scratch/`, `docs/paper/figures/`, `dumps/**`, `.venv/`,
`results/*/{pods.txt,done.txt,*.raw,watchdog.pid,status.txt,*.log}`,
`paper/arxiv*`, `uv.lock`.

Also untracked and to be left alone: `figures/scratch/`, `docs/paper/figures/`
(regenerated by `make figures`), `dumps/` (4.7 GB, identified by its committed
`.sha256`), `.venv/`.

The orchestrator also re-syncs the venv after the merge
(`uv pip install -e ".[dev]"`) --- this worktree shares the main checkout's
`.venv`, so 9c could not install `ninja` or uninstall the eight pruned
dependencies. The removals were verified statically instead (0 imports each),
and `make test` runs green against the pre-prune venv.

### Kept and flagged --- two orphaned `src/` modules (NOT a Task-9 ruling)

The closing reachability sweep ("every `src/kvdlra/**/*.py` is imported by some
entrypoint or test") finds two modules with **no importer at all**:

| path | LOC | why it is still here |
|---|---|---|
| `src/kvdlra/eval/latency.py` | 127 | the measured decode p50 / KV-peak bench --- `paper/main.tex:1056` cites its numbers |
| `src/kvdlra/eval/storage.py` | 145 | the measured stored-state / cold-load / workspace bench --- `paper/main.tex:1118` cites its numbers |

Both were ported out of `scripts/` in Task 8, but nothing was wired to call
them: `kvdlra.eval.runner.GENERATORS` maps only `inhouse` / `official_ruler` /
`longbench`, no `configs/tasks/*.yaml` names them, and no test imports them.

Task 9 does not delete them. No ruling covers them (R24 is about `scripts/`),
and they are the only in-repo way to regenerate two numbers the paper reports,
so deleting them would break the rule that a citable number must be
regenerable. What they need is a caller --- a `generator:` (or a `pod.py`
sub-command) plus a task config --- or an explicit decision to drop the systems
tier. Owner: whoever owns the Task-8 port. Until then the sweep has exactly two
known exceptions, listed here so it is not mistaken for a clean pass.

## LOC (Task 9)

`find src scripts tests -name '*.py' | xargs wc -l | tail -1`

| point | total |
|---|---|
| after the Task-7 self-review pass | 27,224 |
| after Task 8 (start of Task 9) | 26,681 |
| after Task 9a | 16,465 |
| **delta, Task 9** | **-10,216** |

## Task 10 --- the two orphaned `src/` modules, resolved (R31)

Task 9 flagged `eval/latency.py` and `eval/storage.py` as having no importer at
all and handed the decision on. R31 splits them: the decode bench gets a caller,
the storage bench goes.

| path | LOC | resolution |
|---|---|---|
| `src/kvdlra/eval/latency.py` | 127 | **wired**, not deleted --- `generator: latency` in `kvdlra.eval.runner.run_pod`, driven by `configs/tasks/latency_16k_32k_64k.yaml` + `configs/pods/w19_sysfix_latency.yaml`; rows land in `results/<pod>/latency.jsonl` and `scripts/pod.py check` requires the full (arm, ctx, batch) grid |
| `src/kvdlra/eval/storage.py` | 145 | **deleted** |

Why `storage.py` goes: its headline output is the reconstruction-workspace ratio
(`0.982x` at 16K, `0.991x` at 32K), and a workspace that is ~1.0x full KV is the
identity the code audit flagged --- reconstruct-then-attend materializes the
full-length middle, so the measurement says only that a full-length tensor is
full length. The residency question it was built to answer is now answered by a
*measured contrast* instead: the decode-time KV peak per arm, which is
`latency.py`'s `kv_peak_gb` (1.6x full KV --- the number the paper reports, and
the number the kernel lane has to move). Nothing imports it, no task config
names it, no test covers it. `paper/main.tex:1118` cites its numbers; the file
is preserved byte-for-byte at the tag `paper-v1-archive` (ee8c0ab), which is
what the "a citable number must stay regenerable" rule asks for once the number
is superseded rather than re-run.

With this, the reachability sweep has **no** exceptions: every `.py` under
`src/` and `scripts/` is in the static import closure of the four entrypoints,
`tests/test_*.py`, and what `scripts/pod/*.sh` invokes --- pinned by
`tests/test_reachability.py`.

# --- Task 11 --------------------------------------------------------------

## Coverage --- every removed path is accounted for

`git diff --name-status ee8c0ab..HEAD | grep -E '^(D|R)'` returns
**517 rows: 352 `D` (deleted) and 165 `R` (renamed/moved)**. Checked path by
path against this ledger:

| how the `D` path is covered | count |
|---|---|
| a row that names the exact path | 112 |
| a row that names the file (stem) in prose or a table | 6 |
| the `figures/` group row (R22: "the other 86 tracked files under `figures/`") | 86 |
| the `results/` group rows (R18: 109 byte-archived, 91 cited by nothing, 11 archived-then-deleted) | 91 |
| the `docs/` group row (R23: "Deleted, 57 files: `docs/week*.md` (49), `docs/notes/` (5), `docs/board/` (2), `docs/week10_report/` (1), ...") | 57 |
| **not covered** | **0** |

Computed mechanically: a throwaway script pulls every `D` path from the
complete listing below and classifies each by first match — a directory-group
row first (`figures/`, `docs/`, `results/`, each a blanket per-prefix ruling
that accounts for every path under it), then a row naming the exact path, then
a row naming the file's stem or a brace/glob sub-pattern elsewhere in this file.

The four that had no row until this task --- `scripts/perplexity_sweep.py`,
`scripts/w16_storage.py`, `scripts/w19_official_ruler.py`,
`tests/test_w10_longbench_skip.py` --- are the Task-8 section above.

The `R` rows need no ruling: a rename removes nothing. 165 of them, and every
one is either a `git mv` recorded in the Task-6 move table, a narrative report
moved to `docs/plan/reports/` (R17), a review packet moved to
`docs/plan/reviews/`, or a v1 line file moved under `results/paper-v1/<pod>/raw/`
by the archive conversion (R18) --- all `R100`, byte-identical, which is what
`docs/plan/cleanup/paper-source-map.md` maps.

**Figures kept** (R22, restated here because the complete listing shows 86
`figures/` deletions and only these three survive) --- exactly the files
`git show ee8c0ab:paper/main.tex | grep includegraphics` names:
`figures/week19/coldstart.pdf`, `figures/week19/fairquant.pdf`,
`figures/week19/one_over_t.pdf`. `make figures` regenerates all three from
`results/paper-v1/` into the gitignored `docs/paper/figures/`.

## Complete listing

Verbatim `git diff --name-status ee8c0ab..HEAD | grep -E '^(D|R)'`. Two archived
filenames carry words this repo does not use in prose; they are reproduced as
written because the point of this section is that every path is greppable.

```
D	docs/PLAN.md
D	docs/board/week17-archive.html
D	docs/board/week19-board.html
D	docs/index.md
D	docs/notes/conventions.md
D	docs/notes/dependency-substitutions.md
D	docs/notes/rope-pitfall.md
D	docs/notes/streaming-decode-design.md
D	docs/notes/turboquant-rope-interaction.md
R100	results/w18_harvest/quant-findings.md	docs/plan/reports/quant-findings.md
R100	results/w11-final-tables.md	docs/plan/reports/w11-final-tables.md
R100	results/w12-qbug-summary.md	docs/plan/reports/w12-qbug-summary.md
R100	results/w13-phase1-summary.md	docs/plan/reports/w13-phase1-summary.md
R100	results/w13-trackb-design.md	docs/plan/reports/w13-trackb-design.md
R100	results/w13-trackb-summary.md	docs/plan/reports/w13-trackb-summary.md
R100	results/w14-c1-summary.md	docs/plan/reports/w14-c1-summary.md
R100	results/w14-c2-summary.md	docs/plan/reports/w14-c2-summary.md
R100	results/w14-second-bypass-summary.md	docs/plan/reports/w14-second-bypass-summary.md
R100	results/w15-complete-summary.md	docs/plan/reports/w15-complete-summary.md
R100	results/w15-confirm-summary.md	docs/plan/reports/w15-confirm-summary.md
R100	results/w15-ruler-intervals.md	docs/plan/reports/w15-ruler-intervals.md
R100	results/w16-ruler-intervals.md	docs/plan/reports/w16-ruler-intervals.md
R100	results/w17-ruler-intervals.md	docs/plan/reports/w17-ruler-intervals.md
R100	results/w18-g1-llama-ruler-intervals.md	docs/plan/reports/w18-g1-llama-ruler-intervals.md
R100	results/w18-g1-mistral-ruler-intervals.md	docs/plan/reports/w18-g1-mistral-ruler-intervals.md
R100	results/w18-g1-qwen-ruler-intervals.md	docs/plan/reports/w18-g1-qwen-ruler-intervals.md
R100	results/w18-g1-report.md	docs/plan/reports/w18-g1-report.md
R100	results/w18-g1-ruler-intervals.md	docs/plan/reports/w18-g1-ruler-intervals.md
R100	results/w19-a1-report.md	docs/plan/reports/w19-a1-report.md
R100	results/w19-a2-flagship-misses.md	docs/plan/reports/w19-a2-flagship-misses.md
R100	results/w20-close-report.md	docs/plan/reports/w20-close-report.md
R100	results/w20-fork-report.md	docs/plan/reports/w20-fork-report.md
R100	results/w3-parity.md	docs/plan/reports/w3-parity.md
R100	docs/reviews/2026-09-01/review-briefing.md	docs/plan/reviews/2026-09-01/review-briefing.md
R100	docs/reviews/2026-09-01/review-claims.md	docs/plan/reviews/2026-09-01/review-claims.md
R100	docs/reviews/2026-09-01/review-crossexam-attack.md	docs/plan/reviews/2026-09-01/review-crossexam-attack.md
R100	docs/reviews/2026-09-01/review-crossexam-defense.md	docs/plan/reviews/2026-09-01/review-crossexam-defense.md
R100	docs/reviews/2026-09-01/review-meta-verdict.md	docs/plan/reviews/2026-09-01/review-meta-verdict.md
R100	docs/reviews/2026-09-01/review-prior-work.md	docs/plan/reviews/2026-09-01/review-prior-work.md
R100	docs/reviews/2026-09-01/review-repro.md	docs/plan/reviews/2026-09-01/review-repro.md
R100	docs/reviews/2026-09-01/review-rigor.md	docs/plan/reviews/2026-09-01/review-rigor.md
R100	docs/reviews/2026-09-01/review-significance.md	docs/plan/reviews/2026-09-01/review-significance.md
R100	docs/reviews/2026-09-01/review-systems.md	docs/plan/reviews/2026-09-01/review-systems.md
R100	docs/reviews/2026-09-06/review-briefing.md	docs/plan/reviews/2026-09-06/review-briefing.md
R100	docs/reviews/2026-09-06/review-claims.md	docs/plan/reviews/2026-09-06/review-claims.md
R100	docs/reviews/2026-09-06/review-meta-verdict.md	docs/plan/reviews/2026-09-06/review-meta-verdict.md
R100	docs/reviews/2026-09-06/review-prior-work.md	docs/plan/reviews/2026-09-06/review-prior-work.md
R100	docs/reviews/2026-09-06/review-repro.md	docs/plan/reviews/2026-09-06/review-repro.md
R100	docs/reviews/2026-09-06/review-rigor.md	docs/plan/reviews/2026-09-06/review-rigor.md
R100	docs/reviews/2026-09-06/review-significance.md	docs/plan/reviews/2026-09-06/review-significance.md
R100	docs/reviews/2026-09-06/review-systems.md	docs/plan/reviews/2026-09-06/review-systems.md
R100	docs/reviews/2026-09-08/final-review.md	docs/plan/reviews/2026-09-08/final-review.md
D	docs/reference.md
D	docs/w14-sizing.md
D	docs/week1.md
D	docs/week10-handover.md
D	docs/week10-kickoff.md
D	docs/week10-plan.md
D	docs/week10_report/index.html
D	docs/week11-decision-table.md
D	docs/week11-explained.md
D	docs/week11-kickoff.md
D	docs/week11-next-session.md
D	docs/week11-session-handover.md
D	docs/week11.md
D	docs/week12-next-session.md
D	docs/week12-qbug-explainer.md
D	docs/week12-session-handover.md
D	docs/week12.md
D	docs/week13-plan.md
D	docs/week13-session-handover.md
D	docs/week14-plan.md
D	docs/week15-explained.md
D	docs/week15-session-handover.md
D	docs/week15-significance.md
D	docs/week15-t3-note.md
D	docs/week16-explained.md
D	docs/week16-handover.md
D	docs/week17-explained.md
D	docs/week17-handover.md
D	docs/week17-kickoff.md
D	docs/week18-kickoff.md
D	docs/week18-phase01-handover.md
D	docs/week19-handover.md
D	docs/week19-kickoff.md
D	docs/week19-official-ruler.md
D	docs/week2-pilot.md
D	docs/week3.md
D	docs/week4.md
D	docs/week5-plan.md
D	docs/week5.md
D	docs/week6.md
D	docs/week7-dominance.md
D	docs/week7-plan.md
D	docs/week7.md
D	docs/week8-codebook-plan.md
D	docs/week8.md
D	docs/week9-explained.md
D	docs/week9-plan.md
D	docs/week9.md
D	experiments/2026-06-20-scaffold-and-synthetic-bug/README.md
D	experiments/2026-06-22-sv-decay-figure/README.md
D	experiments/2026-w2-rank-policy/README.md
D	experiments/2026-w2-rank-policy/run.py
D	experiments/2026-w2-streaming-bug/README.md
D	experiments/2026-w2-streaming-bug/run.py
D	figs/.gitkeep
D	figures/week1/sv_decay.json
D	figures/week1/sv_decay.pdf
D	figures/week1/sv_decay.png
D	figures/week10/frontier_longctx.pdf
D	figures/week10/frontier_longctx.png
D	figures/week10/ruler_accuracy.pdf
D	figures/week10/ruler_accuracy.png
D	figures/week19/coldstart.png
D	figures/week19/fairquant.png
D	figures/week19/one_over_t.png
D	figures/week2/oja_vs_bug.json
D	figures/week2/oja_vs_bug.pdf
D	figures/week2/oja_vs_bug.png
D	figures/week2/pilot.json
D	figures/week2/pilot.pdf
D	figures/week2/pilot.png
D	figures/week2/pilot_docs.json
D	figures/week2/rank_policy.pdf
D	figures/week2/rank_policy.png
D	figures/week2/rank_policy_metrics.json
D	figures/week2/streaming_bug_metrics.json
D	figures/week2/streaming_bug_rank.pdf
D	figures/week2/streaming_bug_rank.png
D	figures/week4/fair-8b.pdf
D	figures/week4/fair-8b.png
D	figures/week4/fair.pdf
D	figures/week4/fair.png
D	figures/week4/hero.pdf
D	figures/week4/hero.png
D	figures/week4/hybrid-8b.pdf
D	figures/week4/hybrid-8b.png
D	figures/week4/hybrid.pdf
D	figures/week4/hybrid.png
D	figures/week4/needle.pdf
D	figures/week4/needle.png
D	figures/week5/decode_validate_1b.png
D	figures/week5/fp16_longctx_8b.pdf
D	figures/week5/fp16_longctx_8b.png
D	figures/week5/fp16_longctx_8b_gap.pdf
D	figures/week5/fp16_longctx_8b_gap.png
D	figures/week5/hybrid_frontiers.pdf
D	figures/week5/hybrid_frontiers.png
D	figures/week5/hybrid_frontiers_8b.pdf
D	figures/week5/hybrid_frontiers_8b.png
D	figures/week5/hybrid_frontiers_8b_gap.pdf
D	figures/week5/hybrid_frontiers_8b_gap.png
D	figures/week5/hybrid_frontiers_gap.pdf
D	figures/week5/hybrid_frontiers_gap.png
D	figures/week5/integrator_ablation.json
D	figures/week5/integrator_ablation.pdf
D	figures/week5/integrator_ablation.png
D	figures/week5/longctx_crossover.pdf
D	figures/week5/longctx_crossover.png
D	figures/week5/longctx_crossover_frontiers.pdf
D	figures/week5/longctx_crossover_frontiers.png
D	figures/week5/needle_longctx_8b.pdf
D	figures/week5/needle_longctx_8b.png
D	figures/week5/ruler_longctx_8b.pdf
D	figures/week5/ruler_longctx_8b.png
D	figures/week5/streamppl_1b.png
D	figures/week5/streamppl_1b_smoke.png
D	figures/week5/streamppl_8b.png
D	figures/week5/streamppl_8b_tier2.png
D	figures/week7/diagcore_moderate.png
D	figures/week7/merge_doc0.png
D	figures/week7/ranksweep_attn_doc0.png
D	figures/week7/ranksweep_fifo_doc0.png
D	figures/week7/slash_doc0.png
D	figures/week7/slash_moderate_doc0.png
D	figures/week7/streamppl_1b.png
D	figures/week7/streamppl_1b_doc0.png
D	figures/week7/streamppl_8b.png
D	figures/week7/streamppl_8b_tier2.png
D	figures/week8/codebook_1b_doc0.png
D	figures/week8/codebook_1b_doc1.png
D	figures/week9/comparison_frontier.pdf
D	figures/week9/comparison_frontier.png
D	figures/week9/envelope_gate_1b.png
D	figures/week9/recall_8b.pdf
D	figures/week9/recall_8b.png
D	figures/week9/recovery_1b.png
D	figures/week9/recovery_ctxsweep_1b.png
D	figures/week9/recovery_firm_1b.png
D	figures/week9/recovery_gate_1b.png
D	figures/week9/surprise_blend_1b.png
D	figures/week9/surprise_sweep_1b.png
D	mkdocs.yml
R100	results/w18-env-provenance.txt	results/paper-v1/_cited-extras/raw/w18-env-provenance.txt
R100	results/w19-env-provenance.txt	results/paper-v1/_cited-extras/raw/w19-env-provenance.txt
R100	results/w11-base-ppl16-lines.txt	results/paper-v1/w11-base-ppl16/raw/w11-base-ppl16-lines.txt
R100	results/w11-base-ruler-lines.txt	results/paper-v1/w11-base-ruler/raw/w11-base-ruler-lines.txt
R100	results/w11-goalA-lb-lines.txt	results/paper-v1/w11-goalA-lb/raw/w11-goalA-lb-lines.txt
R100	results/w11-goalA-ruler-lines.txt	results/paper-v1/w11-goalA-ruler/raw/w11-goalA-ruler-lines.txt
R100	results/w11-goalB-ppl-lines.txt	results/paper-v1/w11-goalB-ppl/raw/w11-goalB-ppl-lines.txt
R100	results/w11-goalB-probe-lines.txt	results/paper-v1/w11-goalB-probe/raw/w11-goalB-probe-lines.txt
R100	results/w11-goalB-ruler-lines.txt	results/paper-v1/w11-goalB-ruler/raw/w11-goalB-ruler-lines.txt
R100	results/w11-r128-ppl-lines.txt	results/paper-v1/w11-r128-ppl/raw/w11-r128-ppl-lines.txt
R100	results/w11-r128v1-ruler-lines.txt	results/paper-v1/w11-r128v1-ruler/raw/w11-r128v1-ruler-lines.txt
R100	results/w11-r128v2-ruler-lines.txt	results/paper-v1/w11-r128v2-ruler/raw/w11-r128v2-ruler-lines.txt
R100	results/w11-r256-ruler-lines.txt	results/paper-v1/w11-r256-ruler/raw/w11-r256-ruler-lines.txt
R100	results/w11-table-ppl-lines.txt	results/paper-v1/w11-table-ppl/raw/w11-table-ppl-lines.txt
R100	results/w11-table-ruler-lines.txt	results/paper-v1/w11-table-ruler/raw/w11-table-ruler-lines.txt
R100	results/w12-bugr128-ruler-lines.txt	results/paper-v1/w12-bugr128-ruler/raw/w12-bugr128-ruler-lines.txt
R100	results/w12-drop-ruler-lines.txt	results/paper-v1/w12-drop-ruler/raw/w12-drop-ruler-lines.txt
R100	results/w12-mk64-mk-lines.txt	results/paper-v1/w12-mk64-mk/raw/w12-mk64-mk-lines.txt
R100	results/w12-mk64-mvvt-lines.txt	results/paper-v1/w12-mk64-mvvt/raw/w12-mk64-mvvt-lines.txt
R100	results/w12-probe-r128-lines.txt	results/paper-v1/w12-probe-r128/raw/w12-probe-r128-lines.txt
R100	results/w12-qbug-ppl-lines.txt	results/paper-v1/w12-qbug-ppl/raw/w12-qbug-ppl-lines.txt
R100	results/w12-qbug-ruler-lines.txt	results/paper-v1/w12-qbug-ruler/raw/w12-qbug-ruler-lines.txt
R100	results/w12-r192-ppl-lines.txt	results/paper-v1/w12-r192-ppl/raw/w12-r192-ppl-lines.txt
R100	results/w12-r192-ruler-lines.txt	results/paper-v1/w12-r192-ruler/raw/w12-r192-ruler-lines.txt
R100	results/w13-wseed-ppl-lines.txt	results/paper-v1/w13-wseed-ppl/raw/w13-wseed-ppl-lines.txt
R100	results/w13-wseed-ruler-lines.txt	results/paper-v1/w13-wseed-ruler/raw/w13-wseed-ruler-lines.txt
R100	results/w14-wc2n4-ruler-lines.txt	results/paper-v1/w14-wc2n4-ruler/raw/w14-wc2n4-ruler-lines.txt
R100	results/w14-wseed8-ruler-lines.txt	results/paper-v1/w14-wseed8-ruler/raw/w14-wseed8-ruler-lines.txt
R100	results/w15-complete-lines.txt	results/paper-v1/w15-complete/raw/w15-complete-lines.txt
R100	results/w15-confirm-lines.txt	results/paper-v1/w15-confirm/raw/w15-confirm-lines.txt
R100	results/w15b-complete-lines.txt	results/paper-v1/w15b-complete/raw/w15b-complete-lines.txt
R100	results/w16-llama8b-sweep-lines.txt	results/paper-v1/w16-llama8b-sweep/raw/w16-llama8b-sweep-lines.txt
R100	results/w16-llama8b-tier-lines.txt	results/paper-v1/w16-llama8b-tier/raw/w16-llama8b-tier-lines.txt
R100	results/w16-mistral-ppl-lines.txt	results/paper-v1/w16-mistral-ppl/raw/w16-mistral-ppl-lines.txt
R100	results/w16-mistral-sweep-lines.txt	results/paper-v1/w16-mistral-sweep/raw/w16-mistral-sweep-lines.txt
R100	results/w16-mistral-tier-lines.txt	results/paper-v1/w16-mistral-tier/raw/w16-mistral-tier-lines.txt
R100	results/w16-qwen-ppl-lines.txt	results/paper-v1/w16-qwen-ppl/raw/w16-qwen-ppl-lines.txt
R100	results/w16-qwen-sweep-lines.txt	results/paper-v1/w16-qwen-sweep/raw/w16-qwen-sweep-lines.txt
R100	results/w16-qwen-tier-lines.txt	results/paper-v1/w16-qwen-tier/raw/w16-qwen-tier-lines.txt
R100	results/w17-llama8b-lines.txt	results/paper-v1/w17-llama8b/raw/w17-llama8b-lines.txt
R100	results/w17-mistral-lines.txt	results/paper-v1/w17-mistral/raw/w17-mistral-lines.txt
R100	results/w17-qwen-lines.txt	results/paper-v1/w17-qwen/raw/w17-qwen-lines.txt
R100	results/w18_pertrial/llama-trials.txt	results/paper-v1/w18-g1-llama/raw/llama-trials.txt
R100	results/w18_pertrial/mistral-trials.txt	results/paper-v1/w18-g1-mistral/raw/mistral-trials.txt
R100	results/w18_pertrial/qwen-trials.txt	results/paper-v1/w18-g1-qwen/raw/qwen-trials.txt
R100	results/w18_pertrial/g2-qwen-trials.txt	results/paper-v1/w18-g2-qwen/raw/g2-qwen-trials.txt
R100	results/w18-g2-qwen-lines.txt	results/paper-v1/w18-g2-qwen/raw/w18-g2-qwen-lines.txt
R100	results/w18_pertrial/g3-llama-trials.txt	results/paper-v1/w18-g3-llama/raw/g3-llama-trials.txt
R100	results/w18-g3-llama-lines.txt	results/paper-v1/w18-g3-llama/raw/w18-g3-llama-lines.txt
R100	results/w18_pertrial/g3-mistral-trials.txt	results/paper-v1/w18-g3-mistral/raw/g3-mistral-trials.txt
R100	results/w18-g3-mistral-lines.txt	results/paper-v1/w18-g3-mistral/raw/w18-g3-mistral-lines.txt
R100	results/w18_pertrial/g3-qwen-trials.txt	results/paper-v1/w18-g3-qwen/raw/g3-qwen-trials.txt
R100	results/w18-g3-qwen-lines.txt	results/paper-v1/w18-g3-qwen/raw/w18-g3-qwen-lines.txt
R100	results/w18-g4-llama-ppl-lines.txt	results/paper-v1/w18-g4-llama-ppl/raw/w18-g4-llama-ppl-lines.txt
R100	results/w18_pertrial/g4-llama-trials.txt	results/paper-v1/w18-g4-llama/raw/g4-llama-trials.txt
R100	results/w18-g4-llama-lines.txt	results/paper-v1/w18-g4-llama/raw/w18-g4-llama-lines.txt
R100	results/w18-g4-marquee-contrasts.json	results/paper-v1/w18-g4-llama/raw/w18-g4-marquee-contrasts.json
R100	results/w18_pertrial/g5-llama-trials.txt	results/paper-v1/w18-g5-llama/raw/g5-llama-trials.txt
R100	results/w18-g5-llama-lines.txt	results/paper-v1/w18-g5-llama/raw/w18-g5-llama-lines.txt
R100	results/w18-llama-ppl-lines.txt	results/paper-v1/w18-llama-ppl/raw/w18-llama-ppl-lines.txt
R100	results/w18-llama-lines.txt	results/paper-v1/w18-llama/raw/w18-llama-lines.txt
R100	results/w18-mistral-ppl-lines.txt	results/paper-v1/w18-mistral-ppl/raw/w18-mistral-ppl-lines.txt
R100	results/w18-mistral-lines.txt	results/paper-v1/w18-mistral/raw/w18-mistral-lines.txt
R100	results/w18-qwen-ppl-lines.txt	results/paper-v1/w18-qwen-ppl/raw/w18-qwen-ppl-lines.txt
R100	results/w18-qwen-lines.txt	results/paper-v1/w18-qwen/raw/w18-qwen-lines.txt
R100	results/w19_intervals/a1-llama-ruler-intervals.json	results/paper-v1/w19-a1-llama/raw/a1-llama-ruler-intervals.json
R100	results/w19_intervals/a1-llama-ruler-intervals.md	results/paper-v1/w19-a1-llama/raw/a1-llama-ruler-intervals.md
R100	results/w19_pertrial/a1-llama-trials.txt	results/paper-v1/w19-a1-llama/raw/a1-llama-trials.txt
R100	results/w19-a1-llama-lines.txt	results/paper-v1/w19-a1-llama/raw/w19-a1-llama-lines.txt
R100	results/w19_intervals/a1-mistral-ruler-intervals.json	results/paper-v1/w19-a1-mistral/raw/a1-mistral-ruler-intervals.json
R100	results/w19_intervals/a1-mistral-ruler-intervals.md	results/paper-v1/w19-a1-mistral/raw/a1-mistral-ruler-intervals.md
R100	results/w19_pertrial/a1-mistral-trials.txt	results/paper-v1/w19-a1-mistral/raw/a1-mistral-trials.txt
R100	results/w19-a1-mistral-lines.txt	results/paper-v1/w19-a1-mistral/raw/w19-a1-mistral-lines.txt
R100	results/w19_intervals/a1-qwen-ruler-intervals.json	results/paper-v1/w19-a1-qwen/raw/a1-qwen-ruler-intervals.json
R100	results/w19_intervals/a1-qwen-ruler-intervals.md	results/paper-v1/w19-a1-qwen/raw/a1-qwen-ruler-intervals.md
R100	results/w19_pertrial/a1-qwen-trials.txt	results/paper-v1/w19-a1-qwen/raw/a1-qwen-trials.txt
R100	results/w19-a1-qwen-lines.txt	results/paper-v1/w19-a1-qwen/raw/w19-a1-qwen-lines.txt
R100	results/w19_pertrial/a1diag-qwen-trials.txt	results/paper-v1/w19-a1diag-qwen/raw/a1diag-qwen-trials.txt
R100	results/w19-a1diag-qwen-lines.txt	results/paper-v1/w19-a1diag-qwen/raw/w19-a1diag-qwen-lines.txt
R100	results/w19_pertrial/a1q-llama-trials.txt	results/paper-v1/w19-a1q-llama/raw/a1q-llama-trials.txt
R100	results/w19-a1q-llama-lines.txt	results/paper-v1/w19-a1q-llama/raw/w19-a1q-llama-lines.txt
R100	results/w19_pertrial/a1q-mistral-trials.txt	results/paper-v1/w19-a1q-mistral/raw/a1q-mistral-trials.txt
R100	results/w19-a1q-mistral-lines.txt	results/paper-v1/w19-a1q-mistral/raw/w19-a1q-mistral-lines.txt
R100	results/w19_pertrial/a1q-qwen-trials.txt	results/paper-v1/w19-a1q-qwen/raw/a1q-qwen-trials.txt
R100	results/w19-a1q-qwen-lines.txt	results/paper-v1/w19-a1q-qwen/raw/w19-a1q-qwen-lines.txt
R100	results/w19_intervals/a2-llama-ruler-intervals.json	results/paper-v1/w19-a2-llama/raw/a2-llama-ruler-intervals.json
R100	results/w19_intervals/a2-llama-ruler-intervals.md	results/paper-v1/w19-a2-llama/raw/a2-llama-ruler-intervals.md
R100	results/w19_pertrial/a2-llama-trials.txt	results/paper-v1/w19-a2-llama/raw/a2-llama-trials.txt
R100	results/w19-a2-llama-lines.txt	results/paper-v1/w19-a2-llama/raw/w19-a2-llama-lines.txt
R100	results/w19_pertrial/a3-llama-trials.txt	results/paper-v1/w19-a3-llama/raw/a3-llama-trials.txt
R100	results/w19-a3-llama-lines.txt	results/paper-v1/w19-a3-llama/raw/w19-a3-llama-lines.txt
R100	results/w19_pertrial/a3-llama2-trials.txt	results/paper-v1/w19-a3-llama2/raw/a3-llama2-trials.txt
R100	results/w19-a3-llama2-lines.txt	results/paper-v1/w19-a3-llama2/raw/w19-a3-llama2-lines.txt
R100	results/w19_pertrial/a4-llama-trials.txt	results/paper-v1/w19-a4-llama/raw/a4-llama-trials.txt
R100	results/w19-a4-llama-lines.txt	results/paper-v1/w19-a4-llama/raw/w19-a4-llama-lines.txt
R100	results/w19_pertrial/fork-llama-trials.txt	results/paper-v1/w19-fork-llama/raw/fork-llama-trials.txt
R100	results/w19-fork-llama-lines.txt	results/paper-v1/w19-fork-llama/raw/w19-fork-llama-lines.txt
R100	results/w19_pertrial/fork-mistral-trials.txt	results/paper-v1/w19-fork-mistral/raw/fork-mistral-trials.txt
R100	results/w19-fork-mistral-lines.txt	results/paper-v1/w19-fork-mistral/raw/w19-fork-mistral-lines.txt
R100	results/w19_pertrial/fork-qwen-trials.txt	results/paper-v1/w19-fork-qwen/raw/fork-qwen-trials.txt
R100	results/w19-fork-qwen-lines.txt	results/paper-v1/w19-fork-qwen/raw/w19-fork-qwen-lines.txt
R100	results/w19_pertrial/forkdiag-qwen-trials.txt	results/paper-v1/w19-forkdiag-qwen/raw/forkdiag-qwen-trials.txt
R100	results/w19-forkdiag-qwen-lines.txt	results/paper-v1/w19-forkdiag-qwen/raw/w19-forkdiag-qwen-lines.txt
R100	results/w19_pertrial/q4off-llama-trials.txt	results/paper-v1/w19-q4off-llama/raw/q4off-llama-trials.txt
R100	results/w19-q4off-llama-lines.txt	results/paper-v1/w19-q4off-llama/raw/w19-q4off-llama-lines.txt
R100	results/w19_pertrial/swap-llama-trials.txt	results/paper-v1/w19-swap-llama/raw/swap-llama-trials.txt
R100	results/w19-swap-llama-lines.txt	results/paper-v1/w19-swap-llama/raw/w19-swap-llama-lines.txt
R100	results/w19_pertrial/sysfix-llama-trials.txt	results/paper-v1/w19-sysfix-llama/raw/sysfix-llama-trials.txt
R100	results/w19-sysfix-llama-lines.txt	results/paper-v1/w19-sysfix-llama/raw/w19-sysfix-llama-lines.txt
D	results/w10-frontier-1b.json
D	results/w10-gpu-parsed.json
D	results/w11-decision-table-base.json
D	results/w11-decision-table.json
D	results/w11-probe-1b-c4096-t0s0.json
D	results/w11-probe-1b-c4096-t1s1.json
D	results/w11-probe-1b-c8192-t0s0.json
D	results/w11-probe-1b-mk-c16384-t0s0.json
D	results/w11-probe-1b-mk-c4096-t0s0.json
D	results/w11-probe-1b-mk-c4096-t1s1.json
D	results/w11-probe-1b-mk-c8192-t0s0.json
D	results/w11-probe-1b-mk-c8192-t1s1.json
D	results/w11-probe8b-all.json
D	results/w12-qbug-probe.json
D	results/w13-tracka-probe.json
D	results/w13-trackb-facts.json
D	results/w13-trackc-probe.json
D	results/w13-trackx-angles.json
D	results/w13-trackx-rank.json
D	results/w14-second-bypass-facts.json
D	results/w15-e2e-probe.json
D	results/w15-ruler-intervals.json
D	results/w15-scorerank-probe.json
D	results/w16-decision-table.json
D	results/w16-ruler-intervals.json
D	results/w17-decision-table.json
D	results/w17-ruler-intervals.json
D	results/w18-g1-llama-ruler-intervals.json
D	results/w18-g1-mistral-ruler-intervals.json
D	results/w18-g1-qwen-ruler-intervals.json
D	results/w18-g1-ruler-intervals.json
D	results/w18_checkpoint/llama-ruler-lines.txt
D	results/w18_checkpoint/llama.raw.log
D	results/w18_checkpoint/mistral-ruler-lines.txt
D	results/w18_checkpoint/mistral.raw.log
D	results/w18_checkpoint/qwen-ruler-lines.txt
D	results/w18_checkpoint/qwen.raw.log
D	results/w18_harvest/DONE.txt
D	results/w18_harvest/G4DONE.txt
D	results/w18_harvest/SUMMARY.txt
D	results/w19_harvest/a3-llama.raw.superseded
D	results/w19_harvest/done.txt
D	results/w19_harvest/pods.txt
D	results/w3-ppl-1b-post.json
D	results/w3-ppl-1b-pre.json
D	results/w4-fair-8b.json
D	results/w4-fair.json
D	results/w4-head-to-head.json
D	results/w4-hybrid-8b.json
D	results/w4-hybrid.json
D	results/w4-needle.json
D	results/w5-decode-validate-1b.json
D	results/w5-fp16-longctx-8b.json
D	results/w5-hybrid-1b.json
D	results/w5-hybrid-8b.json
D	results/w5-longctx-8b.json
D	results/w5-needle-8b.json
D	results/w5-ruler-8b.json
D	results/w5-streamppl-1b-smoke.json
D	results/w5-streamppl-1b.json
D	results/w5-streamppl-8b-tier2.json
D	results/w5-streamppl-8b.json
D	results/w7-diagcore-moderate-doc0.json
D	results/w7-fd-ablation.json
D	results/w7-merge-doc0.json
D	results/w7-ranksweep-attn-doc0.json
D	results/w7-ranksweep-fifo-doc0.json
D	results/w7-slash-doc0.json
D	results/w7-slash-moderate-doc0.json
D	results/w7-streamppl-1b-doc0.json
D	results/w7-streamppl-1b.json
D	results/w7-streamppl-8b-tier2.json
D	results/w7-streamppl-8b.json
D	results/w8-carry-drift.json
D	results/w8-codebook-1b-doc0.json
D	results/w8-codebook-1b-doc1.json
D	results/w8-codebook.json
D	results/w8-codebook.pt
D	results/w9-envelope-gate-1b.json
D	results/w9-recovery-1b.json
D	results/w9-recovery-8b-multikey.json
D	results/w9-recovery-8b-ranksweep.json
D	results/w9-recovery-8b-single.json
D	results/w9-recovery-8b-win-multikey.json
D	results/w9-recovery-8b-win-single.json
D	results/w9-recovery-ctxsweep-1b.json
D	results/w9-recovery-firm-1b.json
D	results/w9-recovery-gate-1b.json
D	results/w9-surprise-blend-1b.json
D	results/w9-surprise-probe-1b.json
D	results/w9-surprise-sweep-1b.json
D	scripts/calibrate_codebook.py
R067	scripts/capture_kv.py	scripts/dump_kv.py
R052	scripts/w19_figures.py	scripts/figures.py
D	scripts/generate_with_press.py
D	scripts/perplexity_sweep.py
D	scripts/pod/scrape_w10.sh
D	scripts/pod/w10_gpu.sh
D	scripts/pod/w11_baselines.sh
D	scripts/pod/w11_gpu.sh
D	scripts/pod/w11_probe8b.sh
D	scripts/pod/w11_r128.sh
D	scripts/pod/w11_table.sh
D	scripts/pod/w12_probe_r128.sh
D	scripts/pod/w16.sh
D	scripts/pod/w18.sh
D	scripts/pod/w18_boot.sh
D	scripts/pod/w18_finalizer.sh
D	scripts/pod/w18_g4fallback.sh
D	scripts/pod/w18_watchdog.sh
D	scripts/pod/w18_watchdog2.sh
D	scripts/pod/w19.sh
D	scripts/pod/w19_watchdog.sh
D	scripts/pod/w5_confirm_8b.sh
D	scripts/pod/w5_fp16_8b.sh
D	scripts/pod/w5_hybrid_needle_8b.sh
D	scripts/pod/w5_longctx_8b.sh
D	scripts/pod/w5_streamppl_8b.sh
D	scripts/pod/w7_streamppl_8b.sh
D	scripts/sigma_decay.py
D	scripts/w10_frontier.py
D	scripts/w10_longbench.py
D	scripts/w10_parse_logs.py
D	scripts/w10_ruler.py
D	scripts/w11_merge.py
D	scripts/w11_probe.py
D	scripts/w12_calibrate_qkey.py
D	scripts/w12_qbug_probe.py
D	scripts/w13_tracka_probe.py
D	scripts/w13_trackb_bypass.py
D	scripts/w13_trackc_probe.py
D	scripts/w13_trackx_angles_probe.py
D	scripts/w13_trackx_rank_probe.py
D	scripts/w14_second_bypass_probe.py
D	scripts/w15_intervals.py
D	scripts/w15_scorerank_probe.py
D	scripts/w16_intervals.py
D	scripts/w16_storage.py
D	scripts/w16_tier2_probe.py
D	scripts/w17_intervals.py
D	scripts/w18_intervals.py
D	scripts/w19_a1_report.py
D	scripts/w19_a2_misses.py
D	scripts/w19_dashboard.py
D	scripts/w19_fork_report.py
D	scripts/w19_official_ruler.py
D	scripts/w20_close_report.py
D	scripts/w45_integrator_ablation.py
D	scripts/w4_fair.py
D	scripts/w4_head_to_head.py
D	scripts/w4_hybrid_sweep.py
D	scripts/w4_needle.py
D	scripts/w5_decode_validate.py
D	scripts/w5_fp16_longctx.py
D	scripts/w5_hybrid.py
D	scripts/w5_longctx.py
D	scripts/w5_needle.py
D	scripts/w5_ruler.py
D	scripts/w5_streamppl.py
D	scripts/w7_codebook.py
D	scripts/w7_decode_needle.py
D	scripts/w7_fd_ablation.py
D	scripts/w7_rank_sweep.py
D	scripts/w8_carry_drift.py
D	scripts/w9_envelope.py
D	scripts/w9_frontier.py
D	scripts/w9_recall_8b.py
D	scripts/w9_recovery.py
D	scripts/w9_surprise.py
D	scripts/week1_sv_decay.py
D	scripts/week2_oja_vs_bug.py
D	scripts/week2_pilot.py
D	scripts/week2_select_docs.py
R096	src/kvdlra/press/compat.py	src/kvdlra/baselines/compat.py
R085	src/kvdlra/press/bug_press.py	src/kvdlra/baselines/lowrank_press.py
R062	src/kvdlra/press/palu_press.py	src/kvdlra/baselines/svd_oracle.py
R094	src/kvdlra/press/turbo_press.py	src/kvdlra/baselines/turbo_press.py
D	src/kvdlra/cache/morph_cache.py
R100	src/kvdlra/integrators/__init__.py	src/kvdlra/eval/__init__.py
R059	scripts/w20_latency.py	src/kvdlra/eval/latency.py
R070	scripts/w19_persist.py	src/kvdlra/eval/persist.py
D	src/kvdlra/integrators/bug.py
D	src/kvdlra/integrators/bug_adaptive.py
D	src/kvdlra/integrators/bug_class.py
D	src/kvdlra/integrators/bug_torch.py
D	src/kvdlra/integrators/frequent_directions.py
D	src/kvdlra/integrators/oja.py
D	src/kvdlra/integrators/streaming.py
D	src/kvdlra/integrators/streaming_variants.py
D	src/kvdlra/lowrank.py
D	src/kvdlra/press/__init__.py
D	src/kvdlra/quant/product_quant.py
D	src/kvdlra/quant/qjl.py
R087	src/kvdlra/integrators/streaming_torch.py	src/kvdlra/tracker/isvd.py
R100	src/kvdlra/utils/seed.py	src/kvdlra/util/seed.py
D	src/kvdlra/utils/__init__.py
D	src/kvdlra/utils/hydra_resolvers.py
D	tests/test_bug_adaptive.py
D	tests/test_bug_cache_qbug.py
D	tests/test_bug_cache_week8.py
D	tests/test_bug_class.py
D	tests/test_bug_synthetic.py
D	tests/test_bug_torch.py
D	tests/test_frequent_directions.py
R068	tests/test_streaming_torch.py	tests/test_isvd.py
D	tests/test_morph_cache.py
D	tests/test_oja.py
D	tests/test_product_quant.py
D	tests/test_qjl.py
D	tests/test_streaming.py
D	tests/test_streaming_variants.py
R083	tests/test_palu_press.py	tests/test_svd_oracle.py
D	tests/test_w10_longbench_skip.py
D	tests/test_w18_intervals.py
```
