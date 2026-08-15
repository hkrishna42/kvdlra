# Week-15 D0 — when is a perplexity (or RULER) difference significant?

Every number below is cited to its source; the thresholds and the Week-15 tie
rule are **pre-registered here before any Phase-2 GPU dollar is spent**.

## 1. Perplexity differences are ratios, not gaps

Perplexity is the exponential of a mean NLL: the harness pools
`ppl = exp(sum nll / sum tok)` (`scripts/w10_frontier.py:506`). So the natural
difference measure between two arms is the **ratio**, best read in bits:

```
Δbits/token = log2(ppl_a / ppl_b) = (mean NLL_a − mean NLL_b) / ln 2
```

An absolute gap ("0.22 ppl") is meaningless without the base — 0.22 off ppl 4
is 2.6× more bits than 0.22 off ppl 9.

Worked examples (ppl cells: `results/w11-decision-table.json`, key `"32768"`):

| a vs b | Δbits/tok | reading |
|---|---|---|
| 9.0 vs 7.5 (the motivating hypothetical) | **0.263** | 20% ratio — a regime difference, model-generation-sized, decisively significant |
| bugS-r32-h256 9.164 vs think-c0.5 7.897 | 0.215 | same regime-sized gap, from real cells |
| bugS-r32-h256 9.164 vs ea-k0.1 8.277 | 0.147 | 10.7% — clearly significant (but see §5) |
| **bugS-r128-h1024 8.117 vs think-c0.5 7.897** | **0.040** | 2.8% — inside the Week-15 tie threshold (§3); this is the Phase-2 Stage-P working contrast |

## 2. Rules of thumb (ratio bands)

| ppl ratio | Δbits/tok | verdict at our eval size |
|---|---|---|
| < 1% | < 0.0144 | tie — indistinguishable from window noise |
| 1–3% | 0.014–0.043 | real but small (Week-13 Q-BUG's −0.79%/−0.39% lived here: 9.164→9.092, 8.117→8.085, `docs/week12-qbug-explainer.md:42-43` — reported as honest-bounded, not a win) |
| 3–10% | 0.043–0.14 | clearly significant, visible in generation |
| > 10% | > 0.14 | regime difference |

"Our eval size" is small: the published 8B table ppls used the defaults `W=512`
(`w10_frontier.py:635`) × `n_samples=2` (`:677`), i.e. 2 windows × 511 scored
tokens (`_score_window` scores `win_ids[1:]`, `:68-69`) = **1,022 tokens per
published ppl, with no distribution information retained** — until Week-15.

## 3. Per-window NLLs and the pre-registered Week-15 tie threshold

The harness now keeps what it used to discard: each row carries `window_nlls`
(per-window mean NLL, nats/token) and `window_toks`
(`w10_frontier.py:494-495`, `:532-533`), and prints a harvestable
`^\[pplw` line per (arm, T) (`:516-525`), pinned by
`tests/test_w15_pplw.py` (pooled ppl ≡ `exp(sum(nll_i·tok_i)/sum(tok_i))`
recomputed from the fields; `w11_merge.PPL_RE` at `scripts/w11_merge.py:25`
still matches the pooled line unchanged).

**Paired-window SE.** All arms at a given T score the *same* frozen windows
(samples are sliced once per T, `w10_frontier.py:465-468`, before the arm loop
`:474`). For arms a, b with per-window mean NLLs over n shared windows, form
per-window differences in bits:

```
d_i = (nll_a,i − nll_b,i) / ln 2        (i = 1..n)
Δ   = mean(d_i)   (= Δbits/tok, since windows are uniform-length)
SE  = sd(d_i, ddof=1) / sqrt(n)
```

Pairing on identical windows cancels the dominant variance source — shared
document difficulty — leaving only the between-arm difference to fluctuate.

**Pre-registered tie threshold (Week-15, n=8 windows):** arms a and b tie on
perplexity iff

```
|Δbits/tok| ≤ max(0.05, 2 × SE)
```

The 0.05 floor keeps a lucky tiny SE from manufacturing significance out of a
sub-3% ratio; the 2×SE arm keeps a noisy run from declaring ties it cannot
support. The current headline contrast, think-c0.5 vs bugS-r128-h1024 at 32K =
0.040 bits/tok (§1), sits **inside** this threshold — Stage P re-measures it at
n=8 with the SE computed from the `[pplw]` rows.

## 4. RULER cells: Wilson 95% intervals

RULER is all-or-nothing per trial, and the decision table stores each cell's
exact trial count, so hit counts are recoverable exactly:
`hits = round(acc · n)` (the recovery `w11_merge.py:77-83` itself uses). The
Wilson score interval (`scripts/w15_intervals.py:55-64`) is

```
(p + z²/2n ± z·sqrt(p(1−p)/n + z²/4n²)) / (1 + z²/n),   z = 1.96
```

computed for all 228 cells in `results/w15-ruler-intervals.json` / `.md`.

**The n=8 separation example** (why Phase-2 bars demand ≥87.5 at n=8):
7/8 = 87.5% → CI **[52.9, 97.8]** vs 1/8 = 12.5% → CI **[2.2, 47.1]** —
disjoint. At n=2 nothing separates: 0/2 → [0, 65.8] (e.g. shadow-r64@32K,
`results/w15-ruler-intervals.json`, key `"32768"/"shadow-r64"` — void anyway
pending the W-A attach-scope re-measure). A 100% cell needs n=8 just to clear
two-thirds: 8/8 → [67.6, 100].

## 5. Significance is per-objective — ppl ≠ task ability

bugS-r32-h256 is 0.147 bits/tok *worse* than ea-k0.1 at 32K ppl (§1), yet
out-retrieves it on the hard cells at 0.43× of EA's memory: 64K multikey 4/4 vs
2/4, 32K var-track 6/6 vs 5/6 (`results/w11-decision-table.json`, keys
`"65536"`/`"32768"`). Honesty note: at these n the retrieval Wilson intervals
still overlap (4/4 [51.0, 100] vs 2/4 [15.0, 85.0]) — the *direction* is
consistent across cells, but this is exactly why Phase-2 retrieval bars are set
at n=8, where 7/8-vs-1/8 separates (§4). A significance claim must therefore
always name its objective: ppl significance and retrieval significance are
separate tests, each with its own error bar.
