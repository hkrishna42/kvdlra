# Week-13 Track-B: warm-up retrieval fix (design-check)

**Verdict: the first-ingest-chunk SLASH bypass is REAL, and a clean, test-backed,
accounting-neutral seed hook exists. Design FUNDS (a lean, not a slam dunk): the
control-flow proof is airtight and static-checked, but "seeding recovers
early-needle retrieval at 32K/64K without regressing the wins/ppl" is a Phase-2
GPU claim, not a measurement — this is a design-check.**

Proxy caveat: **design-check, not a measurement.** Every control-flow fact below
is statically pinned by `scripts/w13_trackb_bypass.py` (CPU/`ast` only, no torch)
→ `results/w13-trackb-facts.json` (`ALL_TRACE_FACTS_CONFIRMED = true`). The
retrieval/ppl payoff is argued, not run.

---

## 1. The bypass, proven (line-quoted trace)

Data path for a chunked-ingest run (the deployed retrieval arms; `--chunk 4096`,
120 occurrences in `results/gpu_logs/*.log`; `docs/week12-next-session.md:133`:
"bugslash/bugevict REQUIRE --chunk>0 (single-shot bypasses the exact tier)").

### 1a. The FIRST chunk hits `_prefill`, even in ingest mode

`update` (`src/kvdlra/cache/bug_cache.py`) checks `cumulative_length == 0` and
returns `_prefill` **before** it ever looks at `_mode == "ingest"`:

```
1361        if self.cumulative_length == 0:
1362            return self._prefill(key_states, value_states)
1363        if self._mode == "score":
1364            return self._score_forward(key_states, value_states)
1365        if self._mode == "ingest":
1366            return self._ingest_chunk(key_states, value_states)
```

So the first chunk (`cumulative_length == 0`) routes to `_prefill` at :1362; only
chunks 2..N reach `_ingest_chunk` at :1366. The `ingesting()` docstring says so
outright (:1848-1851): *"the FIRST chunk hits the normal single-shot `_prefill`
(cumulative_length == 0), each later chunk is appended to the recent ring with its
absorb deferred."* (Static fact `F3`: prefill-return line 1362 < ingest-return
line 1366.)

### 1b. `_prefill` absorbs the middle DIRECTLY, never through SLASH

`_prefill`'s middle-compression loop calls `_absorb_columns` directly:

```
1434        if mid > 0 and self.lowrank_enabled:
1435            k_pre = self._mat_rope(k_mat[:, n_sink : n_sink + mid], n_sink, inverse=True)
1436            v_mid = v_mat[:, n_sink : n_sink + mid].to(torch.float32)
1437            for start in range(0, mid, self.prefill_block_size):
...
1442                self._absorb_columns(k_pre[:, start:stop], v_mid[:, start:stop], pos, None)
```

There is **no** `hh_enabled` branch here and **no** call to `_absorb_block_slash`.
(Static facts `F1`: `_prefill` calls `_absorb_columns` at :1442 and calls
`_absorb_block_slash` at *no* line.)

### 1c. The SLASH select path is the ONLY writer of `hh_k`/`hh_v`/`hh_pos`, and `_prefill` never reaches it

`hh_k`/`hh_v`/`hh_pos`/`hh_score` are written only inside `_absorb_block_slash`
(:798-802). Its sole caller is `_absorb_block_into_stream`:

```
730         if self.hh_enabled:
731             self._absorb_block_slash(grad_k, grad_v, grad_pos, grad_score)
732             return
```

and `_absorb_block_into_stream` is called only by `consolidate` (:1414, the
deferred-absorb driver) and `_decode_step` (:1456). `_prefill` calls neither.
(Static facts `F2`: `_absorb_block_slash` callers = `{_absorb_block_into_stream}`;
`_absorb_block_into_stream` callers = `{consolidate, _decode_step}`, no
`_prefill`.) `_ingest_chunk` itself defers — it only appends to the recent ring
and returns `_decode_peek` (:1400-1408); its absorb runs later in `consolidate`
(static fact `F4`: `_ingest_chunk` self-calls = `{_to_mat, _decode_peek}` only).

### 1d. Consequence — the warm-up window

The exact tier starts empty (`_reset_state` :566-568 sets `hh_k=hh_v=hh_pos=None`;
static fact `F5`). Every non-sink, non-recent token of the **first chunk**
(≈ `chunk − n_sink − recent_window` = `4096 − 4 − 64 ≈ 4028` tokens for the
deployed config) is absorbed via :1442 straight into the low-rank tail and can
**never** enter `hh_k`. This is exactly the Week-12 "~4-5K absolute warm-up
window": an early-planted needle in the first chunk is only ever a low-rank
coordinate, so retrieval that depends on the verbatim exact tier misses it.

This bypass is already codified as an *intended* invariant in
`tests/test_bug_cache_week11.py:212-229`
(`test_single_shot_prefill_leaves_hh_empty`, asserting `layer._hh_len() == 0`
after single-shot `_prefill`) — i.e. the test we must invert is the statement of
the bug.

### 1e. Why "just route it through SLASH as-is" is not enough (the young-basis half)

Even if `_prefill` fed the middle to `_absorb_block_slash`, selection is by
`_surprise_scores`, which returns **all-1.0 when `self.u_k is None`** (:902-903)
and near-uniform-high for a young rank-r basis. Selecting *during* the first block
(before the basis has seen enough history) is uninformative — every column looks
equally surprising, so top-`hh_budget` degenerates to FIFO/recency and the real
outliers are not preferentially kept. The fix must **warm the basis first, then
select** (task's option 1), not select on a cold basis.

---

## 2. Seed design (exact site + mechanism)

### Primary: warm-then-seed inside `_prefill` (new method `_seed_hh_from_prefill`)

**Site.** In `_prefill`, immediately **after** the middle-absorb loop (after
:1442) and **before** `self.cumulative_length = t` (:1443). Hoist `k_pre` (:1435)
out of the `if` so it is in scope. Guard:
`if mid > 0 and self.lowrank_enabled and self.hh_enabled:`.

**Mechanism.** The loop has already absorbed the whole first-chunk middle, so
`self.u_k` is now a **mature** basis over all `mid` columns — the warm-up
precondition is satisfied by construction. Then promote the true outliers into the
exact tier and remove them from the tail:

```python
def _seed_hh_from_prefill(self, k_mat, v_mat, k_pre, n_sink, mid):
    # Score against the NOW-MATURE basis, in the whitened metric (mirrors
    # _absorb_block_slash:789). k_pre = pre-RoPE middle (n, mid), positions
    # contiguous n_sink .. n_sink+mid-1.
    sel = self._surprise_scores(self._whiten_key(k_pre))
    pos_all = torch.arange(n_sink, n_sink + mid, dtype=torch.int64, device=k_pre.device)
    if self.hh_neighbor > 0:                      # span-boost, as in slash (:790-791)
        sel = self._span_boost(sel, pos_all, self.hh_neighbor)
    keep_n = min(self.hh_budget, mid)
    order = torch.argsort(sel, stable=True, descending=True)
    promote = order[:keep_n].sort().values        # chronological local indices
    # Promote VERBATIM from the raw first-chunk tensors (exact post-RoPE K / raw V,
    # exactly what decode's hh stores). hh_score = None: surprise stores no score (:802).
    self.hh_k = k_mat[:, n_sink : n_sink + mid][:, promote].clone()
    self.hh_v = v_mat[:, n_sink : n_sink + mid][:, promote].clone()
    self.hh_pos = pos_all[promote].clone()
    self.hh_score = None
    # Drop promoted columns from the low-rank tail by POSITION set-difference (a
    # promoted token may or may not have survived prefill eviction -> isin is exact,
    # no fragile index remap). Requires track_positions, which hh configs set
    # (retention != "fifo" -> :505).
    if self.track_positions and self.mid_pos is not None:
        keep = ~torch.isin(self.mid_pos, self.hh_pos)
        self.c_k, self.c_v = self.c_k[:, keep], self.c_v[:, keep]
        self.mid_pos = self.mid_pos[keep]
        if self.track_surprise and self.mid_surprise is not None:
            self.mid_surprise = self.mid_surprise[keep]
        # mid_score / mid_weight analogously iff tracked (they are not, for the
        # deployed retention="lowrank_surprise" bugS configs).
    self._mid_k_cache = self._mid_v_cache = None   # tail changed -> invalidate cache
```

Why this shape:

- **Warms the basis first** (task option 1): scoring uses the post-loop `u_k`, not
  a cold one — fixes §1e.
- **Verbatim from `k_mat`/`v_mat`** (task option 2's promote): the tier is exact,
  matching decode's hh (post-RoPE K, raw V). It can even recover an early outlier
  that prefill eviction discarded, because `k_mat` still holds the whole first
  chunk in scope.
- **Position set-difference eviction** keeps `hh` and the tail **disjoint** (the
  SLASH invariant), so no double-count in `_decode_peek` (:1469-1477).
- **Idempotent with steady state.** From chunk 2 on, the unchanged
  `_absorb_block_slash` includes `hh_k` in its candidate pool every absorb
  (:764-768) and re-scores it against the maturing basis, so a seeded column the
  basis can now reconstruct is demoted (:803-810) — self-correcting — while a
  persistent needle (residual ≈ 1) is never demoted (:753-757).

### Alternative (smaller diff, less robust): route the prefill middle through `_absorb_block_slash`

Replace :1437-1442 with, when `hh_enabled`, a loop feeding the **post-RoPE**
middle sub-blocks to `_absorb_block_slash` (positions `arange`, `grad_score=None`).
Pros: reuses the tested path verbatim, disjoint by construction, basis built only
from non-hh (matches the steady-state invariant). Cons: the young-basis (§1e) is
**unmitigated** for a first block absorbed as few blocks — in the limit of one
giant block (single-shot, or `prefill_block_size ≥ mid`) it selects on an empty
basis (FIFO) and misses the needle; and it cannot recover a token once evicted.
It works in the real regime (4096-token first chunk / 128-token sub-blocks → 32
sub-blocks → basis matures) but is fragile to config. Prefer the primary.

---

## 3. Unit-test spec (mirror `tests/test_bug_cache_week11.py`)

Reuse the file's fixtures/helpers: `tiny_model`, `_bug_layer`, `_needle_prompt`,
`_chunked_prefill`, `_pos` (:37-109).

**T1 — `test_prefill_seeds_needle_into_hh` (the core fix; single-shot).** Mirror
`test_surprise_slash_retains_needle_verbatim` (:172) but plant the needle in the
first block and drive **single-shot** `_prefill`:
```python
ids = _needle_prompt(bg_id=5, needle_id=200, t=64, depth=30)
bug = BugStreamingCache(tiny_model, rank=4, coord_budget=128, recent_window=8,
                        absorb_block=4, retention="lowrank_surprise",
                        hh_budget=4, hh_select="surprise")
with torch.no_grad():
    tiny_model(ids, past_key_values=bug, use_cache=True)   # single-shot prefill
layer = _bug_layer(bug)
assert layer.hh_pos is not None and 30 in layer.hh_pos.tolist()   # seeded, verbatim
assert layer.mid_pos is None or 30 not in layer.mid_pos.tolist()  # not double-counted
```
This is the exact inverse of the current `test_single_shot_prefill_leaves_hh_empty`
(:212-229), which **must be updated** (the "invariant" it pins is the bug). Rename
it `test_prefill_seeds_hh_from_first_block` and assert `layer._hh_len() > 0` and
`30 in layer.hh_pos.tolist()` after single-shot prefill.

**T2 — `test_first_chunk_needle_captured` (chunked ingest).** Plant the needle at
`depth < chunk` (inside the first chunk) and ingest via `_chunked_prefill(...,
chunk=16, attach=False)`; assert the needle is in `hh_pos`. Guards the deployed
warm-up path, not just single-shot.

**T3 — `test_seed_preserves_disjointness_and_caps`.** After a seeded prefill with
`mid > coord_budget` (force in-prefill eviction) and `hh_budget` small: assert
`set(hh_pos) ∩ set(mid_pos) == ∅`, `len(hh_pos) <= hh_budget`, and
`_f_len() <= coord_budget`.

**T4 — losslessness unchanged.** `test_surprise_slash_lossless_matches_dynamic_cache`
(:115) must still pass (full rank + full budget → seed moves outliers to verbatim
hh, tail exact → reconstruction unchanged). Keep as a regression guard.

---

## 4. Accounting: seeding within an unchanged `hh_budget` does NOT change `stored_state_numel`

`stored_state_numel` counts `hh_k`/`hh_v` as full verbatim tensors (:1587-1588)
and `hh_pos` as bookkeeping (:1619); `hh_score` is `None` under surprise select so
it costs nothing (:1620) — the seed preserves that (`self.hh_score = None`).
(Static fact `F6`.) The seed introduces **no new kind of stored tensor**; it only
(i) fills `hh` from `k_mat`/`v_mat`, capped at `keep_n = min(hh_budget, mid)`, and
(ii) removes the promoted columns from `c_k`/`c_v`/`mid_pos`/`mid_surprise`.

Both tiers stay independently capped: `hh ≤ hh_budget` (by `keep_n`) and
`c_k ≤ coord_budget` (removal-only, then `_enforce_budgets`). So the total
`coord_budget + second_tier + hh_budget` bound (`_post_update_lengths` mid_cap,
:689-691) is untouched.

The saturated accounting model **already assumes `hh` is full**:
`bug_footprint_saturated` passes `hh_count=hh_budget`
(`src/kvdlra/accounting.py:190`), and `bug_footprint` charges
`hh_count·(1 if surprise else 2)` aux words + `2n·hh_count` verbatim (:145,:156).
In the current code `hh` fills to `hh_budget` within the first ~`hh_budget`
graduated tokens of decode anyway; the seed just fills it at end-of-prefill
instead. **At the 32K/64K measurement checkpoint (saturated), `stored_state_numel`
is identical with or without the seed.** The anti-drift pin
`test_bug_footprint_matches_stored_state_numel`
(`tests/test_accounting.py:92-128`) computes the footprint from the **live**
counts (`hh_count=layer._hh_len()`, `coord_count=_f_len()+_q_len()`) and asserts
equality with `stored_state_numel()` — it is seed-agnostic and still passes.

Honest transient note: at *end-of-prefill* the seeded cache holds `hh=keep_n`
that the current code has not filled yet, so its end-of-prefill footprint is
larger by `keep_n·(2n − 2r − 1)` — but this is bounded by the same saturated
high-water (both tiers capped), and it vanishes by saturation. No peak-memory
increase.

---

## 5. Regression argument: the ≥32K wins are preserved

1. **Scoped.** The change lives entirely in `_prefill` under `self.hh_enabled`
   (`hh_budget ≥ 1`). Plain BUG and every non-SLASH arm never enter it → their
   32K numbers cannot move.
2. **Decode/ingest path untouched.** `_absorb_block_slash` / `_decode_step` /
   `consolidate` are unchanged, so mid- and late-context needle capture (the bulk
   of the existing wins) is bit-identical.
3. **Monotone for early needles.** The seed can only *add* first-chunk outliers to
   `hh` that were previously impossible to capture; it removes nothing from the
   mid/late tiers.
4. **Self-correcting.** Steady-state re-selection (:764-810) still runs and can
   demote a mis-seeded column, so `hh` cannot be permanently polluted.
5. **Perplexity (predicted, not measured).** Moving first-chunk outliers to
   verbatim `hh` leaves the low-rank tail basis to summarise the *outlier-removed*
   residual — the SLASH premise — so tail reconstruction should be equal-or-better,
   not worse. Minor caveat: the warm loop transiently absorbed the outliers before
   promoting them, so the first-chunk basis "saw" them; the effect is small because
   a rank-`r` basis cannot represent sharp outliers anyway (that is *why* their
   surprise stays high), and the basis re-bases every decode absorb.

**Kill conditions that do NOT hold:** the bypass is real (not "the first block
already enters the tier"); seeding does not change saturated `stored_state_numel`;
and there is a clean disjointness-preserving hook — so none of the KILL clauses
fire.

---

## 6. What remains a Phase-2 (funded, GPU) question

- Does the seed actually flip early-needle RULER cells at 32K/64K (the payoff)?
- Do the two operating points' ppl (bugS-r32-h256, bugS-r128-h1024) hold within
  noise (the §5.5 prediction)?
- Confirm the anti-drift + week11 suites pass with the updated test.

These need the model; this track only establishes that the design is concrete,
disjointness-safe, accounting-neutral, and test-backed. Evidence:
`results/w13-trackb-facts.json` (from `scripts/w13_trackb_bypass.py`).
