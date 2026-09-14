# Deletions ledger — L0 repo cleanup

One line per removed path, with the evidence that nothing executable reached it,
and one line per path that *looked* like a candidate but was kept (with the
reason). Buckets and LOC are quoted from `docs/plan/cleanup/reachability.md`
(§2 full table, §3 candidate lists); `audit` cites
`docs/plan/cleanup/ponytail-audit.md` §"src/kvdlra — 13 modules that nothing
executable reaches".

Started in Task 6; Task 7 appended below; Task 9 appends after it.

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
| **delta, Task 6** | **-5,011** |
| **delta, Task 7** | **-4,670** |
| **delta, cumulative** | **-9,681** |
