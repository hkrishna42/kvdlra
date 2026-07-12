# Week 9 — BUG as an *aid* to eviction (three complementary hybrids)

> **Frame.** The competition question is closed (Weeks 5–8: BUG does not beat
> eviction at matched-memory extreme-budget mean ppl; every hybrid is bounded by
> two measured walls). This week pivots to *complement, not compete*: BUG and
> eviction have **orthogonal failure modes** — eviction *forgets* (a dropped token
> needed later is gone), BUG *rank-squeezes* the deep tail. Each direction is
> judged **on the axis where the weakness lives** (envelope / retrieval / recall),
> NOT the settled extreme-budget mean ppl. Read with `docs/week9-plan.md`
> (the plan) and `docs/week7-dominance.md` / `docs/week8.md` (why competition is
> closed).

Model: `unsloth/Llama-3.2-1B-Instruct`, CPU/fp32, WikiText-2, `n = head_dim ·
num_kv_heads = 512` (1 token = `2n = 1024` floats/layer). Every arm's memory is
counted honestly (`stored_state_numel`) and audited `mem_max ≤ budget`.

## Status table

| Dir | Mechanism | Axis judged | Verdict |
|---|---|---|---|
| **D1** | BUG recovery tier for eviction's dropped tokens | recall-of-dropped-content | **WIN** (bounded by buffer saturation) |
| **D2** | adaptive-SLASH as the {eviction, BUG} envelope | cross-budget Pareto | **bounded** (no extension) |
| **D3** | low-rank surprise as an eviction score | matched-mem ppl / deep tail | **bounded** (complementary, not exploitable) |

---

## D2 — adaptive-SLASH as "the envelope" (BOUNDED)

**Mechanism.** SLASH (`hh_budget` exact heavy-hitters + rank-`r` BUG tail +
`coord_budget` coverage `W` on `BugStreamingCache`) spans a family from
eviction-like (large `hh`, small `r`) to BUG-like (`hh=0`). A budget-adaptive
allocation `(hh, r, W)` should trace the Pareto envelope of {eviction, BUG} across
budgets. `scripts/w9_envelope.py` (a-priori schedule + optional per-budget-best
oracle + no-regret/crossover verdict). Honest caveat: SLASH does **not** contain
MorphKV (`rank=0` degenerates to StreamingLLM, not attention-scored whole-token
eviction), so morph is an **external** baseline, not a special case.

**Gate result (crossover region, doc0, `results/w9-envelope-gate-1b.json`).**

bugA-best swept over ranks {32,48,64,96,128}; slash a-priori = the committed
closed-form `τ→(r,hh)` rule.

| τ (tok-eq) | morph | bugA-best (rank) | slash a-priori (r,hh) | envelope winner | s − min(pure) |
|---|---|---|---|---|---|
| 128 | 16.63 | **16.38** (r48) | 18.70 (r64,h4) | bugA | +2.33 (a-priori over-ranks) |
| 180 | 15.25 | **10.38** (r64) | 11.07 (r64,h8) | bugA | +0.69 (hh costs coverage) |
| 256 | 13.67 | 8.48 (r96) | **8.45** (r64,h8) | slash (within ε) | −0.03 |
| 360 | 12.04 | 7.59 (r96) | **7.57** (r96,h16) | slash (within ε) | −0.02 |

**Findings.**
1. **The crossover is below τ ≈ 128.** With the finer rank grid, bugA-r48
   (16.38) already *beats* morph (16.63) at τ=128 — BUG wins the whole swept
   range moderate→up. (The earlier coarse grid {32,64,96} missed r48 and
   mislabelled τ=128 "morph wins" — an adversarial-review correction.) The
   overhead-floor wall still governs the *extreme* end (τ<~100, where r≥64's `2nr`
   basis alone exceeds budget and only rank-squeezed r16/r32 are feasible → morph
   wins, `docs/week7-dominance.md`).
2. **No strict envelope extension anywhere.** At τ=256/360 adaptive-SLASH
   *numerically edges* the best pure BUG (8.45<8.48, 7.57<7.59) but by ≤0.03 —
   **within ε** (a within-noise tie), the `hh` exact tier buying a whiff and no
   more (consistent with the Week-7 SLASH wall, +0.06/+0.47 bounded).
3. **The a-priori allocation is genuinely fiddly** (this IS the bounded story):
   the optimal rank tracks *coverage-adequacy*, not a clean `√τ`, so the
   closed-form rule over-ranks at τ=128 (picks r64 with `W`=69 → 18.70, when r48
   was best) and pays a coverage tax for `hh` at τ=180. Even the per-budget-best
   `hh`-sweep (Week-7 SLASH) only ties the envelope. **Rank-squeeze floor:**
   r16/r32 plateau at ppl ~17.4 regardless of coverage — viable minimum r64.

**Verdict: BOUNDED.** The *achievable* SLASH family only ties the {eviction, BUG}
envelope (numerically edges pure BUG by ≤0.03 at τ=256/360, within ε; never a
strict win), and a deployable closed-form allocation cannot even reliably realize
that (it over-ranks at low budget, taxes coverage for `hh`). It does **not** extend
the envelope, and cannot reach morph at the extreme end because the `2 n r` basis
is dead weight morph never pays. A clean, map-completing negative on the envelope
axis — consistent with the Week-7 SLASH wall. 8B not warranted (bounded).

---

## D3 — low-rank surprise as an eviction score (GATE: GO; sweep pending)

**Mechanism.** Evict the coordinate columns the low-rank basis reconstructs best
(`||k − U Uᵀk|| / ||k||` small = *redundant* with the summary) and keep the
high-residual *outliers* the summary cannot reproduce. By Pythagoras this residual
is the out-of-subspace half of `||k||²`, so it is the **orthogonal complement** of
`retention="energy"` (the in-subspace half, a Week-7 negative), and a candidate
*better* signal than attention mass. Stored per column as one fp32 scalar (a
graduation snapshot — not recomputable later, `U C` has zero residual), the same
per-column cost as an `attn` score. `retention="lowrank_surprise"` on
`BugStreamingCache`; `scripts/w9_surprise.py`.

**GO/NO-GO gate — the correlation probe** (`results/w9-surprise-probe-1b.json`,
318k samples over 2144 eviction events):

- **Spearman(surprise, attention-mass) = −0.11** → |ρ| ≪ 0.4 → **not redundant**
  with attention (redundancy would need a large *positive* ρ). The residual is at
  most *weakly* informative relative to attention, not "richly orthogonal" — the
  probe pools full buffers repeatedly (range-restriction that attenuates |ρ|
  toward 0), so the true per-column ρ is likely more negative, which only makes
  "not redundant" safer; it does not make the signal strong. **GO** (gate is only
  a green light to run the sweep; the verdict rests on the sweep, not this ρ).
- Spearman(surprise, energy) = −0.09 (weak, not the naive near-−1: the normalized
  surprise `resid/||k||` is scale-free and thus decoupled from unnormalized energy
  `||c||` because `||k||` varies per token).

**Sweep (`results/w9-surprise-sweep-1b.json`, `w9-surprise-blend-1b.json`, r64,
τ≈243, 2 docs).** Streaming ppl (geo over docs):

| retention | agg ppl | deep-tail (last bin) |
|---|---|---|
| bugA (attn) | 9.743 | 11.54 |
| **bugB α=0.75** (blend) | **9.740** | 11.54 |
| bugS (pure surprise) | 9.827 | 11.83 |
| bug (fifo) | 10.121 | — |
| bugE (energy) | 10.695 | 15.53 |
| morph | 13.115 | — |

- **Ranking: attn > surprise > fifo > energy.** Pure surprise is a *reasonable*
  eviction signal (clearly beats fifo and energy, so the residual is informative)
  but as a standalone replacement it **trails attention** (+0.9% agg, and it loses
  on the deep tail — the very axis it should win by keeping outliers).
- **The blend (α=0.75) beats attention by 0.003 ppl (0.03%)** — within the
  doc-to-doc variance, and it *reverses* on the deep tail. Not a meaningful win.

**Verdict: BOUNDED on its own axis, but with ONE real payoff via D1.** As a
general-purpose eviction score, surprise is not usefully exploitable — attention
mass already captures what matters for streaming ppl, so neither replacing nor
blending it beats attention beyond noise. **But** surprise retention is precisely
"keep the outliers," and in D1's recall setting the dropped needle *is* a
low-attention high-residual outlier: there `bugS` keeps the needle's coordinate
where attention retention evicts it (D1 ctx-2048: bugS 0.25 = the no-eviction
diagnostic, vs bugA 0.00). So surprise's honest home is **outlier-retention for
recall, not mean-ppl eviction** — a bounded-but-not-worthless result. 8B not
warranted for the ppl axis.

---

## D1 — BUG as eviction's recovery tier (gate pending)

**Mechanism.** Run MorphKV unchanged; route its **evicted** tokens into a
constant-memory BUG sketch so attention sees `[morph kept] + [BUG summary of the
dropped stream]`. GQA crux: MorphKV evicts *per KV head*, so the recovery sketch
must be **per-head, disjoint by construction** (a shared stacked sketch is
provably wrong — cannot rebuild a token's full column / double-counts).

**Realization tested.** The *shared-eviction* form: `BugStreamingCache` — its
constant-memory low-rank middle IS a recovery summary of everything eviction
would drop; `bugA` (retention="attn") summarizes all history, `slash` adds an
exact heavy-hitter tier. The per-head MorphKV-*preserving* `HybridRecoveryCache`
is the purest form, noted as future work (the GQA per-head crux); the per-head
sketch uses lower rank `r_h≤8` so it would not reconstruct *better*.

**Isolation gate (`results/w9-recovery-gate-1b.json`).** full retrieves a
mid-context needle **4/4**; pure morph **0/4** — the probe cleanly isolates
forgetting (task well-posed, eviction dropped the needle).

**Recall result** (`results/w9-recovery-{ctxsweep,firm}-1b.json`; needle at depth
0.3/0.5, matched budget = morph's per-layer floats; `mem` = measured total stored
floats). bugA/bugS/slash rank 64, coordinate buffer W≈1298:

| ctx | morph | bugA (attn) | bugS (surprise) | bugFID r64 (no-evict) | bugFID r256 | full |
|---|---|---|---|---|---|---|
| 512 | 0.25 | **1.00** | — | — | — | 1.00 |
| 1024 | 0.00 (4.59M) | **1.00 (4.02M)** | 1.00 | 1.00 | 1.00 | 1.00 |
| 2048 | 0.00 | 0.00 (4.54M) | 0.25 | 0.25 (6.1M) | **1.00 (21M)** | 1.00 |

**Findings (the honest decomposition — an adversarial-review catch turned a
premature "negative" into this).**
1. **WIN up to ~1024 tokens, at LOWER memory.** When the needle's coordinate is
   retained, BUG's rank-64 constant-memory summary recovers it **4/4** where
   whole-token eviction forgets it (0/4) — and at *less* memory (bugA 4.02M <
   morph 4.59M at ctx 1024). The falsifiable D1 target (recovery beats morph at
   matched memory on dropped-content recall) is **met**. Consistent with BUG's
   Week-4 needle edge (ties ExpectedAttention 15/15), now shown in the
   streaming-decode *recovery-tier* framing.
2. **The ctx-2048 failure decomposes into TWO real limits, not a single
   "fidelity ceiling" (my first writeup overclaimed):**
   - *Coordinate eviction.* bugA's own buffer saturates (W=1298 < mid 2012) and
     its FIFO-at-prefill eviction drops the low-attention needle → 0/4. **Surprise
     retention fixes this half:** `bugS` (keep high-residual outliers) → 0.25 =
     the no-eviction `bugFID-r64` → 0.25, i.e. surprise *successfully kept the
     outlier needle's coordinate* where attention retention evicted it. **A real
     D3×D1 synergy** (the one place D3's signal pays off).
   - *Rank-vs-context fidelity.* Even with **zero** eviction, rank-64 over 2012
     tokens recovers only 1/4 (`bugFID-r64`); **rank-256 recovers 4/4** (`bugFID
     -r256`, 21M mem). So a genuine rank-fidelity limit bites at long context,
     fixed by more rank = more memory. A +19% premium (r96) → 0.50.

**Verdict: WIN in the matched-memory moderate-context regime (≤ ~1024 tokens),
transitioning to "needs a memory premium" at longer context** as the rank-vs-
context fidelity tradeoff (Week-4/5) reasserts. Surprise retention (D3×D1) removes
the eviction half of the long-context failure but not the rank half.

**Honest caveats (do not overstate):** small n (4 trials/cell = 2 depths × 2
passcodes; the 4/4-vs-0/4 split is decisive but coarse); single sharp-fact needle
(RULER multi-key / multi-hop untested at 1B here — Week-4 showed BUG ties
ExpectedAttention on needle retrieval, so the direction is consistent, but the
harder multi-key case is a follow-up); morph capacity 192 is the aggressive
operating point that forces forgetting — at bugA's *lower* 4.02M memory morph
(cap≈160) still drops the needle, so the win survives matching memory in the
adverse direction. The per-head MorphKV-preserving `HybridRecoveryCache` and an 8B
confirmation are the natural follow-ups. 8B deferred (1B result is decisive on its
axis; pod credit conserved; keys still unrotated).

---

## Overarching framing

BUG as a **complement that extends eviction where eviction is weak**, not a
competitor that replaces it. The three directions, judged each on the axis where
eviction's weakness lives:

- **D1 (WIN):** on **recall of dropped-then-queried content** — the axis where
  eviction's *forgetting* lives — BUG's constant-memory low-rank summary recovers
  needles whole-token eviction loses (4/4 vs 0/4, at ≤ matched memory, up to
  ~1024 tokens). This is the load-bearing positive result of the week and the
  clearest demonstration of BUG *aiding* eviction. Bounded at long context by the
  rank-vs-context fidelity tradeoff (needs a memory premium).
- **D2 (bounded):** on the cross-budget **envelope**, adaptive-SLASH does not
  extend the {eviction, BUG} Pareto frontier — the `2nr` basis overhead is dead
  weight eviction never pays, so no allocation reaches morph at extreme budgets.
- **D3 (bounded):** BUG's reconstruction residual is complementary to attention
  mass but not a better *general* eviction score; its one real use is
  outlier-retention feeding D1's recall win.

**The map is complete and consistent with Weeks 5–8:** BUG does not beat eviction
on mean ppl (settled), and does not *extend* the envelope (D2) or improve the
eviction *score* (D3) — but it genuinely **aids** eviction by *recovering
forgotten content* (D1), exactly the orthogonal-failure-mode seam the week set out
to test. Every verdict reported straight; the D1 headline was itself corrected
from a premature negative by the adversarial-verification pass (the ctx-2048
`bugA=0/4` was a coordinate-eviction artifact, not a fidelity ceiling).
