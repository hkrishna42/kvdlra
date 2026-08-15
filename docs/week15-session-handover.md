# Week 15 — session handover (resume state)

> The single doc a fresh session reads to resume. Repo `/Users/hari/Desktop/kv-dlra`,
> branch `week7` (== `main`, pushed, `127b4b4`). Then read `results/w15-complete-summary.md`
> + `results/w15-confirm-summary.md` (the funded result) and `docs/week15-significance.md`
> (the pre-registered D0 significance framework). Forward plan: memory
> `kvdlra-week16-plan` + `~/.claude/plans/based-on-the-current-smooth-stearns.md`. Ethos:
> numbers straight, prefer honest negatives, no overclaim ("matches the FIELD, not full"),
> all memory in one float-equivalent unit. Ultracode ON — Workflow fan-out with an
> adversarial number-verification stage before any claim enters a doc/dashboard.

## 1. State
- Branch `week7` == `main` == `origin/week7`, all at **`127b4b4`**, tracked tree clean
  (`git rev-parse HEAD` == `origin/week7`). Suite **322 passed / 1 skipped / 1 xfailed**;
  ruff + mypy clean. The **1 xfailed is deliberate**: the T3 latent `seed_scores` bug,
  pinned by the strict-xfail `tests/test_bug_cache_week15.py::test_seed_scores_chunked_ingest_latent_bug`
  (fails today at the exact IndexError line; a half-fix XPASSes and trips — see §2 T3).
- **No pods running** — the three Week-15 pods `47758582` (main confirm) / `47794970`
  (completion) / `47811756` (16K-ppl recovery) are all **destroyed**.
- Credit **~$117** (topped up). **Keys still unrotated** — flag, don't rotate mid-run.

## 2. What is DONE — the both-axes point is FUNDED on Llama-3.1-8B

**HEADLINE.** `bugSseed-r128-h1024-s32` (Llama-3.1-8B-Instruct) at **0.159×** (32K) /
**0.191×** (16K) memory **matches the low-rank field (ThinK, Palu) on perplexity AND beats
it on retrieval, at 3–5× less memory.** The complete row (`w15-complete-summary.md`):

| ctx | mem | ppl (n=8 matched) | single | multi-key | multi-value | var-track |
|---|---|---|---|---|---|---|
| 16K | 0.191× | 5.434 | 100 | 100 | 100 | 100 |
| 32K | 0.159× | 7.353 | 100 | 100 | 100 | 100 |

A clean **100/100/100/100** on all four retrieval tasks at **both** contexts.

- **Perplexity ties** (paired-window bits/token, pre-registered rule `|Δbits/tok| ≤
  max(0.05, 2·SE)`, n=8; `docs/week15-significance.md §3`):
  - **32K** (pooled ppl: full 6.975, think-c0.5 7.196, palu-r0.5 7.232, s32 7.353):
    s32 vs think **+0.031** (2SE=0.035, thr 0.05 → **TIE**); vs fixed-palu **+0.024** →
    **TIE**; vs **full +0.076** (~5%, an honest **DIFF** — BUG does not match *uncompressed*
    KV at 6.3× compression, and should not). At **3.2–4.7× less memory** than think/palu.
  - **16K** (pooled: full 5.272, think 5.413, palu 5.419, s32 5.434): s32 vs think
    **+0.0056** → TIE; vs palu **+0.0042** → TIE; vs **full +0.0438** → below the 0.05
    floor = **near-full-KV quality**. At **2.6–3.9× less memory**.
  - The claim is "matches the low-rank **field**," which it does — **not** full KV.

- **MECHANISM — the score-rank decoupling (T2), the funded lever.** `--score-rank 32`
  caps the SLASH surprise-scoring basis at a leading **rank-32 subview**. **Default-off,
  ZERO memory cost** (ratio unchanged, matched A/B), **+0.30% ppl** (7.331→7.353 @32K).
  It un-blinds exact-tier selection: the large r128 gist otherwise "fits" the needle into
  low surprise so it is never selected into the exact tier. Matched A/B (only `-s32`
  differs) does **all** the retrieval lifting:

  | cell | task | uncapped | s32 |
  |---|---|---|---|
  | 32K r128 | var-track | **0** (0/4) | **100** (4/4) — Wilson-**disjoint** [0,49] vs [51,100] |
  | 32K r128 | multi-value | 0 | 100 |
  | 16K r128 | single | 0 | 100 |
  | 16K r128 | multi-value | 25 | 100 |
  | 16K r128 | var-track | 75 | 100 |
  | 16K/32K r128 | multi-key | 100 | 100 (seed-saturated) |

  Honest oddity: uncapped 16K gets mk=100 but single=0 (task-specific); the s32 arm is
  uniformly clean. All arms carry `--warmup-seed` (Week-14 promoted), so the **only** A/B
  delta is `-s32`.

- **BASELINES FIXED + validated on 8B** (they were **our** bugs; fixing them makes the
  baselines STRONGER — mandatory for honesty, and BUG still wins):
  - **ShadowKV**: `0/0/0/0` was a harness `attach()`-scope defect (decode was never
    attached — same defect class that broke the T3 attn tier). Fixed → `shadow-r64` @16K
    single/mk/vt = **100/100/0** at 0.815×. Week-11 shadow rows **voided**.
  - **Palu**: worst-ppl was the port **low-ranking the attention sinks**. Sink carve-out →
    32K ppl **7.232** (was 9.236; CPU-1B 28.39→16.13), retrieval 16K 100/100/75, 32K
    100/100/25. A genuinely strong low-rank baseline now — and BUG still **ties its ppl at
    3.2× less memory** and beats its var-track (**100 vs 25**).

- **RIGOR shipped.** Per-window NLL emission (`[pplw]` lines, `scripts/w10_frontier.py`,
  pooled ppl ≡ `exp(Σnll·tok/Σtok)` pinned by `tests/test_w15_pplw.py`); paired-window SE
  cancels shared document difficulty. Wilson 95% intervals for **all 228** RULER cells
  (`scripts/w15_intervals.py` → `results/w15-ruler-intervals.{json,md}` — NB: that
  instrument is over the **pre-confirm** decision table, so it still shows the old
  void/broken shadow+palu rows; the fixed 8B numbers live in the lines-files).

- **KNOB/CODE.** `score_rank` in `src/kvdlra/cache/bug_cache.py` (`_surprise_scores` capped
  **at the SLASH site only**); `--score-rank` flag in `scripts/w10_{frontier,ruler}.py`;
  arm-name suffix `-s{k}`. Identity (default-off ⇒ bit-for-bit) + accounting-neutrality
  pins in `tests/test_bug_cache_week15.py`.

- **T3 (hybrid attention tier) — DESIGN ONLY, scoped $0** (`docs/week15-t3-note.md`).
  Three blockers all confirmed: the `hh_select='attn'` retention guard
  (`bug_cache.py:394-399`), decode-never-attached (the ShadowKV defect class), and the
  latent `seed_scores` IndexError (`:1382-1401`, raise at `:1397`) — pinned by the
  strict-xfail above. Builds only per a **B2-kill** trigger; B2 did **not** kill (s32
  funds), so **not built**. Compose with the seed, don't replace it.

## 3. Data map (under `results/` unless noted)
- `w15-confirm-lines.txt` — main harvest: the 32K `[pplw]`/ppl block (full/think/palu/
  bug-uncapped/bug-s32) + retrieval rows (matched uncapped-vs-s32, the fixed shadow+palu
  rows, and the r256 **dead** rows).
- `w15-complete-lines.txt` — the fill cells: 32K single + multi-value, 16K single, and
  the 16K `[pplw]` block.
- `w15-confirm-summary.md` (32K verdict) / `w15-complete-summary.md` (completed both-
  contexts row) — the authoritative writeups. Every ppl Δ in them is a **paired-window**
  bits/tok from the `[pplw]` NLLs, not a rounded pooled-ppl gap.
- `docs/week15-significance.md` — D0 framework (ratio bands, the pre-registered tie rule,
  Wilson). `docs/week15-t3-note.md` — T3 scoping. `results/w15-ruler-intervals.{json,md}`
  — the 228-cell Wilson table.
- Scripts/tests: `scripts/w15_intervals.py`, `scripts/w10_frontier.py` (pplw at
  `:516-525`), `scripts/w10_ruler.py` (attach scope widened to prefill+decode);
  `tests/test_bug_cache_week15.py`, `tests/test_w15_pplw.py`.
- `gpu_logs/` is gitignored; committed lines-files are the durable record. Pod out-JSONs
  stayed pod-side (`vastai copy` broken) and died with the destroyed pods — the printed
  rows are the record.
- Commits: `0e65662` (score_rank knob) · `206906a` (baseline fixes) · `5064c94`
  (per-window NLL) · `8cde7dc` (T2 probe verdict) · `06eac36` (GPU confirm) · `aacf288`
  (README) · `127b4b4` (completion).

## 4. Infra recipe + the standing gotchas
vast.ai **A100 SXM4** (~$0.56–0.82/hr), image
`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`; pods clone `origin/week7` — **push
first**. `--onstart` must be **≤16KB** (trim the MODE list to fit). Gotchas, all carried:
- **STREAMING bug ppl arms are ~30 min EACH at 16K** — far slower than the press
  baselines. Size cells from this (the Week-14 tasks-per-cell sizing miss). Rough card
  costs: r32@16K ~2 min/trial, r32@32K ~6.5, r128@16K ~5, r128@32K 10–15; **64K needs an
  80GB card (48GB OOMs).**
- **Harvest PRINTED ROWS only**: `^\[niah`, `^\[vt`, the two-leading-space ppl lines
  (`^  bugS…`/`^  full…`), and `^\[pplw`. `vastai logs` truncates; `--tail 20000` (not
  200000).
- **zsh**: `${pair%%:*}` splitting, **quote** `===` markers (bare `===` globs / errors).
- **NEVER pipe `git commit` through `tail`** (a hook failure was masked once) — commit
  bare, verify `git log -1`.
- **DESTROY every pod** (`printf 'y\n' | uvx vastai destroy instance <id>`).
- **WATCHERS record the terminal reason (ALL_DONE vs TIMEOUT) and it MUST be checked
  before any destroy** — the completion pod `47794970` was destroyed on a *timeout*
  before its last ppl arm finished, losing the two bug-arm 16K ppls; a **$0.4** recovery
  pod (`47811756`) re-measured them and the base ppls **reproduced bit-identically**
  (deterministic windows). Retrieval + 32K ppl were never at risk. Lesson is now the rule.

## 5. Dashboards (update IN PLACE — pass `url:` from a new session)
- decision table (Week-15 completed row): https://claude.ai/code/artifact/19e23647-d242-4310-896d-be2fb7e8ee0e
- Week-14 / understanding board: https://claude.ai/code/artifact/ada55e23-81b6-4731-b8d7-6760c1495524
- older, still live: overview https://claude.ai/code/artifact/e811be6a-abb6-408a-89ec-d3fa8fd311d1 ·
  explainer https://claude.ai/code/artifact/c776074d-e7d4-475a-b325-1fb7eefe02d7

## 6. Open questions / honest caveats → Week-16
**Approved: Week-16 NeurIPS-readiness program** (memory `kvdlra-week16-plan`; plan file
`~/.claude/plans/based-on-the-current-smooth-stearns.md`). Three pillars:
1. **Generality** — does the both-axes result hold off Llama? Run the arm on
   **Mistral-7B-v0.3** + **Qwen2.5-7B**.
2. A **storage-reframe** tier.
3. **Firming** the marginal-n lifts.

Carried caveats (the firming targets):
- **32K var-track (0/4→4/4, Wilson-disjoint) is the airtight single retrieval result.**
  **16K var-track** (6/8→8/8) has **overlapping CIs** (direction real); **16K
  multi-value** (1/4→4/4) is **marginal-n**. A higher-n re-run tightens these.
- **r256 stays dead for retrieval** — score-rank does **not** resurrect it (the rank
  cliff is real; both arms 0/0). Not a Week-16 target absent a new mechanism.
- vs **full** KV, BUG is **~5% higher ppl @32K** (honest, expected at 6.3× compression) —
  the both-axes claim is against the low-rank **field**, never full.
- **Pre-registration wobble (on the record):** the Stage-A gate ("s32 must lift r256/16K
  mk,vt from 0") was **mis-designed** — r256 is past the cliff, so both arms are 0/0 there
  for reasons unrelated to s32; I briefly misread that as a T2 KILL. The valid test is the
  deployed rank r128 (Stage B/B2), where s32 clearly funds. Deviating from "A-kill → skip
  B" (budget ample; uncapped-B had independent value) was correct and surfaced the result.
- **T3 hybrid attn tier** remains design-only (three blockers mapped, latent bug pinned) —
  build only if a firming/generality result calls for it.
