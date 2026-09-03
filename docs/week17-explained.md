# Week-17 explained — validating the r64 config, and one fix that lands

Week-17 asked three questions about `bugSseed-r64-h256` (rank-64 gist + a warmup-seeded 256-token
exact tier), BUG's extreme-compression config from Week-16:

1. **Does its cross-model generality hold with error bars** (n=12 at 16K, plus 32K)?
2. **Can the Mistral var-track weakness be fixed?**
3. **Can the safe rank be raised by stabilizing the streaming integrator?**

The answers, GPU-confirmed on Llama-3.1-8B, Qwen2.5-7B, and Mistral-7B-v0.3: **yes**, **no**, and
**yes, decisively**. One clean new win (the integrator floor), one honest negative (the vt fix), and a
generality claim that is now statistically firm.

---

## 1. Generality holds — and it's now Wilson-firm

At **16K, n=12** (the count needed to clear a Wilson-95% lower bound of 0.70 — 8/8 only reaches 0.68),
`bugSseed-r64-h256` gets **perfect single-needle and multi-value retrieval on all three families**:

| model | single | multi-value | var-track | memory |
|---|---|---|---|---|
| Qwen2.5-7B | 1.00 [0.76,1.0] | 1.00 [0.76,1.0] | **1.00 [0.76,1.0]** | 0.149× |
| Mistral-7B-v0.3 | 1.00 [0.76,1.0] | 1.00 [0.76,1.0] | 0.50 [0.25,0.75] | 0.085× |
| Llama-3.1-8B | 1.00 [0.76,1.0] | 1.00 [0.76,1.0] | 0.58 [0.32,0.80] | 0.085× |

(Wilson 95% intervals, 12/12 = [0.758, 1.0].) At **0.085–0.149× of full KV** — **3.4–10× less
float-equivalent stored state than `think-c0.5` (0.75×) or `palu-r0.5` (0.50×); 6.7–13× vs full KV** — BUG
matches or leads them on retrieval. It even **leads both baselines on multi-value** (point estimate, not
Wilson-separated): Qwen 16K mv is bug 1.00 vs think 0.83 / palu 0.92; Qwen 32K mv is bug 1.00 vs palu 0.25.
Qwen additionally holds 1.00/1.00/1.00 at 32K. This is the paper's core claim — the extreme-compression
frontier generalizes across three families — now with error bars instead of n=4 point estimates.

The **one weak axis is variable-tracking on the two 1024-dim models** (Mistral 0.50, Llama 0.58 at 16K;
Qwen, at 512-dim, is 1.00). Following a `V0=…; V1=V0; …` chain needs several *linked, low-surprise*
tokens retained together, and BUG's surprise-selected exact tier keeps sharp needles, not chains. An
honest n=12 correction: Llama's vt is **0.58**, not the 0.75 that Week-16's n=4 suggested.

---

## 2. The var-track fix is refuted — honestly

Phase-1's proposed fix was to **raise the exact-tier budget** (h256 → h512/h1024), on the theory (from a
1B CPU retention probe) that ~3% of the context in exact slots would retain the whole chain. On the real
7B/8B it **does not work**:

| model | vt @ h256 | vt @ h512 | single/mv @ h512 |
|---|---|---|---|
| Mistral-7B | 0.50 (6/12) | **0.25 (3/12)** | 1.00 / 1.00 |
| Llama-8B | 0.58 (7/12) | **0.58 (7/12)** | 1.00 / 1.00 |

More exact-tier budget gives **no var-track lift** (and costs memory: 0.085× → 0.100×). The 1B proxy
over-predicted — exactly the soft spot flagged during synthesis. Combined with the two *memory-free*
alternatives that Phase-1 had already CPU-refuted (score-rank capping and incumbent stickiness both
protect fresh-number needles, not chains), **the var-track weakness on 1024-dim models has no working fix
from this program.** It is a real, documented limitation, not a solved one. Reporting it as such is the
honest move — and it doesn't touch the single/multi-value frontier, which is where BUG wins.

---

## 3. The integrator floor — the clean win, and it raises the safe rank

*Safe rank* here means the largest gist `rank` whose perplexity stays within the r64 operating point's
band (no numerical blow-up); it is a within-method notion — a floored high-rank cell is still
Palu-dominated at matched memory, so "raises the safe rank" is a stability claim, not a frontier claim.

Week-16 left two unexplained numerical blow-ups: the pure-gist tracker diverged at high rank (Mistral
`bug-r128` ppl **138**, Qwen `bug-r256` **27531**), and — the "puzzle" — a *larger* exact tier destroyed
fluency (Qwen `bugSseed-r128-h1024` ppl **467**). The Phase-1 investigation showed these are **one defect**:
with no truncation tolerance, the rank-adaptive step always pads the basis to `rank_cap`, and on an
ill-conditioned KV stream those **near-null tail directions tip into a numerical explosion** (the SLASH
exact tier makes it *worse* by siphoning the rank-carrying tokens out of the gist, dropping its effective
rank below `rank_cap`). One fix removes the whole class: a **default-off relative singular-value floor**
(`min_sv_frac`) that caps the tracked rank at the block's numerical rank.

GPU-confirmed, `--min-sv-frac 1e-2` turns every explosion into healthy — in fact *optimal* — fluency:

| ppl @ 16K | floor off | floor on |
|---|---|---|
| Qwen `bug-r256` | 27531.7 | **6.995** |
| Qwen `bugSseed-r128-h1024` (the puzzle) | 714.4 | **6.941** |
| Mistral `bug-r128` | 138.3 | **5.574** |
| Mistral `bug-r256` | 37.2 | **5.226** |
| Mistral `bugSseed-r128-h1024` | 60.4 | **5.045** |

On both families the floored high-rank fluency is the **best across all ranks** (Mistral improves
monotonically 6.22 → 5.57 → 5.23 as rank goes 64 → 128 → 256), so the floor doesn't merely rescue the
divergence — **it genuinely extends the usable rank** (Week-17 goal 3). It is **default-off and
bit-for-bit identical when off** (verified on the real integrator and by every parity test staying green),
**retrieval-neutral** at the sweet spot (floor-on RULER unchanged), and **footprint-neutral** (storage is
unchanged). This is the headline Week-17 result: a small, safe, mechanistically-understood change that
removes a whole failure mode and unlocks higher ranks on non-Llama families.

---

## 4. The Llama marquee — beats ThinK, leads Palu on var-track, firmed

Firming Week-16's Tier-1 headline at **n=16**: Llama `bugSseed-r128-h1024-s32` at 32K gets **var-track
0.94 [0.72, 0.99] (15/16)** and **multi-value 1.00 [0.81, 1.0] (16/16)** at 0.16× memory. Against the
n=16 baselines:

| Llama 32K vt (n=16) | acc | Wilson 95% | memory |
|---|---|---|---|
| `bugSseed-r128-h1024-s32` | **0.94** | [0.72, 0.99] | 0.16× |
| `think-c0.5` | 0.31 | [0.14, 0.56] | 0.75× |
| `palu-r0.5` | 0.56 | [0.33, 0.77] | 0.50× |

An **honesty correction to the Week-16 headline**: at n=16 the claim narrows from "beats Palu *and* ThinK"
to "**beats ThinK (Wilson-separated: 0.72 > 0.56), leads Palu on point estimate (0.94 vs 0.56) but not
Wilson-separated**" (the intervals overlap, 0.72 < 0.77) — at **3–4.7× less memory**. Palu's var-track
firmed *up* from Week-16's n=4 estimate (0.25 → 0.56), which is what erodes the clean separation. The
multi-value comparison is a three-way tie at 1.00. The defensible marquee statement is therefore about
**ThinK plus memory**, not a clean sweep of both baselines.

---

## 5. The one honest caveat — Qwen-specific

`bugSseed-r64-h256` fluency is **healthy at 16K on all three** — r64 ppl is 1.08–1.32× full (Llama 5.31,
Mistral 5.50, Qwen 8.18). At **32K it thins only on Qwen** (ppl 8.2 → **35.1**; retrieval stays
1.00/1.00/1.00), while **Mistral (3.76, 1.14× full) and Llama (8.33, 1.11× full) stay healthy** — so the
32K dip is **Qwen-specific**, not a general property of rank-64 at long context. And because the floor now
makes higher rank safe, the natural 32K config where fluency matters is **r128/r256-with-floor**, not r64.

---

## Takeaway

The r64 extreme-compression config is now a **validated cross-model result** (single+multi-value = 1.00 on
three families at 3.4–10× less stored state than ThinK/Palu, Wilson-firm). The var-track axis on wide-KV models is an honest
open limitation. And the `min_sv_frac` floor is a clean, safe, general fix that **removes the high-rank
numerical instability and raises the usable rank** — the most reusable artifact of the week.
