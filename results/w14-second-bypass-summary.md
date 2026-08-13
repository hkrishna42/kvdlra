# Week-14 W-5 — second-structural-bypass probe: **KILL**

**One line:** the first ingest chunk's *middle* is the **only** place a needle is
*structurally* barred from the exact tier. Every mid-stream needle — on a chunk
boundary or deep inside a steady-state block — reaches `_absorb_block_slash` (the
tier's sole writer) and enters the exact tier under steady-state SLASH. No second
structural bypass exists, so there is nothing new to fund.

- **Verdict: KILL** (honest prior confirmed).
- **Pre-registered bar:** *FUND* if a second real **structural** bypass with a cheap
  fix exists (then specify fix + unit-test design, do **not** implement); *KILL* if the
  first chunk is the only structural gap (steady-state SLASH captures mid-stream /
  boundary needles). → **KILL clause fires.**
- Evidence (all `$0`, CPU): `scripts/w14_second_bypass_probe.py` →
  `results/w14-second-bypass-facts.json` (`ALL_CHECKS_PASS = true`,
  8 static facts + 3 dynamic checks + 2 real-dump checks).

---

## 1. The trace (why there is only one structural gap)

The exact tier `hh_k`/`hh_v`/`hh_pos` is a low-entry-count set with a **single
writer**, and a token can reach the low-rank tail through a **closed** set of sites.
Enumerated statically (`phase1`, AST over `src/kvdlra/cache/bug_cache.py`):

| fact | claim | evidence |
|---|---|---|
| **G1** | `hh_k`/`hh_v`/`hh_pos` written non-`None` **only** by `_absorb_block_slash` | writer set `= {_absorb_block_slash}` (`:813-815`) |
| **G2** | `_absorb_block_slash` reached from exactly two sites | callers `= {_absorb_block_into_stream (:746), _prefill (:1472)}` — graduation gateway + the shipped `bugSseed` seed |
| **G3** | the graduation gateway runs for **every** mid-stream / decode block | `_absorb_block_into_stream` callers `= {consolidate (:1429), _decode_step (:1493)}` |
| **G4** | the low-rank tail (`_absorb_columns`) has a closed caller set | `= {_prefill, _absorb_block_into_stream, _absorb_block_slash}` |
| **G6** | under `hh_enabled`, the graduation gateway's **direct** tail-absorb is unreachable | `_absorb_block_slash(:746)` then `return (:747)` **before** `_absorb_columns(:750)` |
| **G5/G7/G8** | dispatch + seed + defer | `update` sends `cumulative_length==0 → _prefill` before the `ingest` check (`:1377<:1381`); `_prefill` holds **both** the bypass (`_absorb_columns :1479`) and the seed route (`_absorb_block_slash :1472`); `_ingest_chunk` defers (no absorb of its own) |

**Reading G1–G8 together:** the only way a token becomes a tail-only coordinate
(never offered to the exact tier) is `_prefill → _absorb_columns` on the **first
chunk's middle** with the seed off. Chunks 2..N defer (G8) to `consolidate →
_absorb_block_into_stream → _absorb_block_slash` (G3, G6); decode does the same; the
seed (G2/G7) is the deliberate *second* entrance that already closes the first-chunk
gap. A chunk boundary introduces **no** new absorb site (G8), and the graduating-block
schedule is driven by `recent_window + absorb_block`, not by chunk edges.

## 2. Independent re-derivation (a second, disjoint path to the same verdict)

Not trusting the AST alone, re-derived by the **tail-append closure** (grep, not the
probe's logic): *new* columns are appended to the low-rank tail only at
`_absorb_columns:873-877` (`self.c_k = torch.cat(...)`, `self.mid_pos = torch.cat(...)`).
Every other write to `c_k`/`mid_pos` is:
- `_absorb_columns:859-860` — the carry/rotation of **existing** columns (no new token);
- `_enforce_budgets:1023-1031` — **removal only** (`self.c_k = self.c_k[:, keep]`);
- `_merge_down:1113-1120` — **recombine only** (merge mode; disabled for `bugS`, and
  the seed guard `:452` rejects `seed + merge`).

So eviction and merge are strictly *downstream* of SLASH — they never route a token
*around* it. This kills the "graduation/eviction transition" hypothesis specifically:
a token evicted from the tail already faced SLASH (or was a first-chunk-middle bypass);
eviction cannot retroactively bar a token from a tier it already passed through. This
closure argument reaches the same conclusion as G1–G8 by a disjoint route. **CONFIRMED.**

## 3. Dynamic confirmation on the real routing (tiny hermetic Llama, seed OFF)

Driving the deployed control flow end-to-end (`update → _ingest_chunk → consolidate →
_absorb_block_into_stream → _absorb_block_slash`), `chunk=32`, `t=192`, `hh_budget=1`,
`hh_select="surprise"`. A needle in a homogeneous background is the cleanest isolated
outlier, so selection is deterministic and the test isolates the **structural** signal.
`in_hh` is itself proof the needle reached SLASH (G1: SLASH is the tier's only writer);
an instrumentation wrapper independently logs the positions offered to SLASH.

| case | depth | first chunk? | boundary? | seed | offered→SLASH | in `hh` | in tail |
|---|---|---|---|---|---|---|---|
| mid, chunk boundary | 64 | no | **yes** | off | **yes** | **yes** | no |
| mid, chunk boundary | 96 | no | **yes** | off | **yes** | **yes** | no |
| mid, deep in block | 50 | no | no | off | **yes** | **yes** | no |
| mid, deep in block | 130 | no | no | off | **yes** | **yes** | no |
| first-chunk middle (contrast) | 15 | **yes** | no | off | **no** | **no** | yes |
| first-chunk middle (control) | 15 | yes | no | **on** | yes | yes | no |

Mid-stream and boundary needles are **captured** at seed-off (not barred); only the
first-chunk-middle needle bypasses SLASH (offered = no, lands in the tail), and the
shipped seed captures it. This is the KILL picture: one structural gap, already fixed.

## 4. Real-data robustness (the capture/selection half, `dumps/llama3.2-1b/*len4096*`)

The structural verdict is control-flow (data-independent). To confirm the **selection**
half generalizes off the homogeneous background, on real `doc411 len4096` layer-0
pre-RoPE keys with a real streamed rank-32 basis (`augmented_bug_step`, the deployed
integrator) and the real `_surprise_scores`:

- real diverse background surprise: median **0.203**, max **0.277** (basis captures it);
- a planted out-of-subspace needle: surprise **1.0**, **rank #0** (top-1) in its block.

So once routed, a real mid-stream outlier is selected — the capture mechanism is not an
artifact of the tiny homogeneous background. (Honest scope: synthetic-outlier needle +
prefix basis, not an end-to-end 32K retrieval number; that is the Phase-2 GPU domain.)

## 5. Honest boundaries of this KILL

- **Selection *capacity* is not a structural bypass and is out of scope.** On real data a
  mid-stream needle competes with real outliers for `hh_budget` slots and could be
  crowded out / demoted — but that is the ordinary SLASH capacity trade-off, it applies
  **uniformly at every position**, not at a special second site, and the token still
  *reaches* the tier. It does not meet the "structural bypass" bar and is already the
  known `rank/hh_budget`-vs-context wall.
- **Sinks and the first-chunk recent tail are not barred.** The first `n_sink` tokens are
  kept verbatim (trivially retrievable), and the first chunk's *recent* tail graduates
  through SLASH when chunk 2 ingests — so the residual first-chunk gap is only the
  **middle**, exactly what `bugSseed` targets (window shrinks to one `prefill_block_size`
  sub-block, per `results/w13-trackb-design.md §1`).
- The dynamic test uses a homogeneous background by design (isolates structure); real-data
  selection is covered separately in §4.

## 6. Decision

**KILL — no new lever.** The first-chunk middle is the sole structural bypass and it is
already closed by the funded `bugSseed`. Week-14 effort stays where the plan puts it:
firming the seed win to decision-grade stats (Phase 2), not chasing a second bypass. No
fix is proposed and none is implemented (per the bar). Evidence:
`results/w14-second-bypass-facts.json`; reproduce with
`uv run python scripts/w14_second_bypass_probe.py`.
