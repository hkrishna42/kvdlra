# Week-15 T3 scoping note — hybrid attention tier (design only, no build)

The secondary bet: select the exact SLASH tier by **attention mass** (or an
attn+surprise blend) instead of pure low-rank surprise, so a needle the basis has
absorbed (invisible to surprise — the measured rank-retrieval coupling, and B2's
KILL on capped-surprise recovery) is still kept verbatim because the model *reads*
it. Scoped $0; builds only per the plan's B2-kill trigger.

## The three blockers (all confirmed)

1. **attn-select forces attn-retention** (`src/kvdlra/cache/bug_cache.py:394-399`):
   `hh_budget > 0` + `hh_select='attn'` raises
   unless `retention='attn'`. A hybrid tier wants an attn-selected exact tier OVER
   the deployed `lowrank_surprise` tail — the guard structurally forbids the
   combination and must be relaxed (attn-select requires `attach()`, not
   attn-retention).
2. **The harness never attached decode** — attn scores only accumulate inside
   `attach()`; the retrieval harness wrapped just the prefill, so decode-step
   `observe_attention` never fired (same defect class that voided ShadowKV's
   rows). W-A's A1 fix widens the scope (`scripts/w10_ruler.py` now wraps prefill
   AND decode; `w10_longbench.py` in the same change set).
3. **The latent `seed_scores` bug** (`bug_cache.py:1382-1401`; the IndexError
   line is :1397): the attach() hook
   seeds prompt scores per *chunk*, but `seed_scores` indexes **absolute**
   positions into the **chunk-length** seed from `_prompt_seed_scores` — the ring
   slice `seed[cumulative_length - rlen : cumulative_length]` silently desyncs
   from chunk 2 (out-of-range slice → wrong length) and `seed[self.mid_pos]`
   raises IndexError once any retained position ≥ chunk length (chunk 3). Chunked
   ingest is the only deployed long-context path, so an attn tier is broken at
   16K+ until this is fixed. **Pinned** by the strict-xfail characterization test
   `tests/test_bug_cache_week15.py::test_seed_scores_chunked_ingest_latent_bug`
   (fails today at the exact line; a half-fix XPASSes and trips).

## Reuse map (nothing new to invent)

- `src/kvdlra/cache/morph_cache.py:265-320` — `_aggregated_attention_row`
  (decode-step GQA-aggregated row) and `_window_attention_rows` (prompt-window
  causal rows, already chunk-aware via `w = min(window, hidden, T)`). Both are
  already what `BugStreamingCache.attach` calls; the hybrid tier reuses them
  unchanged.
- `src/kvdlra/cache/bug_cache.py:275-298` — `_prompt_seed_scores` (decay-collapsed
  window seed). Correct per-chunk; only the *mapping* in `seed_scores` is wrong.

## Minimal viable design

`hh_select="attn"` (or `"blend"`-style rank-mix with surprise) over the deployed
surprise tail, three surgical changes: (a) relax the :394 guard to allow
attn-select with any tail retention (keep the fail-loud unattached warning);
(b) fix `seed_scores` to take the chunk's absolute offset (`cumulative_length -
T_chunk`) and map: ring ← the seed's overlap with `[cum-rlen, cum)`, mid/q ←
`seed[pos - off]` masked to `pos >= off`, EMA-merged (not overwritten) across
chunks — the xfail test then XPASSes and is replaced by real pins; (c) run under
W-A's widened attach scope. Cost: `hh_score` (+1 fp32/column, already the counted
convention) + the attach-hook compute; storage rank and tiers unchanged, so
accounting is neutral. Risk to pre-register: attention mass is *observed after
retention decisions* on earlier tokens — a first-chunk needle still needs the
Week-13 seed (compose, don't replace).
