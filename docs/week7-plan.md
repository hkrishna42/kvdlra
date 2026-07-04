# Week 7 plan — fixing BUG's deep-horizon retention (Axis-B follow-up)

> Context: `docs/week6.md`. The decode-time streaming BUG cache is correct,
> constant-memory, and the best method while generation stays within ~2× its
> budget, but past that its penalty *grows* while adaptive eviction stays
> *flat* (hypothesis inverted). The measured mechanisms: (1) FIFO coordinate
> eviction — adaptive subspace but non-adaptive retention; (2) erosion by
> repeated projection (`c ← rot·c` per absorb); (3) fixed rank over growing
> history. Week 7 attacks retention directly, in coordinate space.

## Priority order

**Tier 1 — run first, together, at 1B locally (independent knobs, same buffer):**

- **(A) Adaptive coordinate retention.** Drop the coordinate column with the
  lowest recent-attention mass (MorphKV-style scoring — the hook machinery in
  `MorphKVCache` already recomputes recent attention rows; cheaper proxies:
  ‖c_s‖ energy or accumulated attention) instead of the oldest. Gives BUG
  MorphKV's eviction brain while keeping the rank-r summary. *Falsifiable
  signature: reproduces MorphKV's flat late-bin ratio while keeping BUG's
  ~1.03× early bins — strictly the best of both curves if mechanism (1)
  dominates.*
- **(D) Quantize-instead-of-drop (age-tiered precision).** On coordinate
  overflow, PolarQuant the oldest coords to 4 bits (Week-4 machinery,
  `BUGPress.quant_bits` already validated BUG×PolarQuant on coordinates) —
  same memory holds ~4× more history; optional 2-bit second tier. *Signature:
  late bins flatten roughly as if `W` were 4× larger → by the ~2×-budget
  crossover rule the winning regime should extend to ~8× budget.*

**Tier 2 — the numerically interesting ablation:**

- **(C) Retention-aware truncation.** The BUG step currently truncates by the
  singular values of the accumulated-stream core `B` — optimizing fidelity to
  history including tokens already dropped. Truncate instead by the spectrum
  of `[C | A_new]` (the *retained* coordinates + incoming block), optionally
  attention-weighted: DLRA on the **windowed process**, not the stream. Cost
  O((r+b)²W) per absorb. The forgetting-factor knob on `B` is the poor-man's
  version — ablate both. This is the item that engages the DLRA theory most
  directly (it targets erosion, mechanism 2).

**Tier 3 — paper-worthy novelty, ONLY if tier 1 closes the gap but does not
flatten the tail:**

- **(B) Merge with log-count softmax correction.** Overflowing coords merge
  into super-columns (weighted centroids); keys attend via centroid + `log m`
  logit bias (first-order cluster-softmax); values merge exactly (attention
  output is a weighted sum). History becomes a constant-memory pyramid.
- **(F) Linear-attention tail.** Fold would-be-dropped tokens into a fixed
  associative state `S = Σ φ(k_s)v_sᵀ` (random-feature softmax, Infini-
  attention-like): exact recent + BUG middle + kernel tail. Statistical error
  instead of structural deletion; riskiest (RF noise at low dim).

Cheap orthogonal ablations if time permits: per-layer budget water-filling on
measured spectra; positional squashing for the middle (cache-relative
positions à la StreamingLLM).

## Honesty guardrails (non-negotiable)

- **Every variant counts ALL its memory** — quant scales, score buffers,
  cluster counts, kernel state — in the matched-memory budget, or we recreate
  the Week-4 unfairness.
- **Success criterion is specific:** flatten the late-bin ratio **without**
  losing the near-lossless early bins. A fix that trades one for the other is
  a lateral move, not a win.
- Same harness (`scripts/w5_streamppl.py`), same protocol, geo-mean bins,
  matched worst-case stored floats; validate at 1B locally (CPU is enough —
  Week-6 1B ran in ~2.5 h), confirm at 8B on one pod (recipe + gotchas in
  `docs/week6.md` §Infra); n_docs ≥ 3 at 8B, note doc-level variance.
- Report the verdict either way; a clean "retention fixes don't close it"
  negative is a fine result.
