# Week-16 explained — does BUG generalize beyond Llama?

The Tier-2 question was the single biggest reviewer ask: **the whole story so far is on Llama. Does the
method work on other model families?** We tested Mistral-7B-v0.3 and Qwen2.5-7B. The answer is subtle,
honest, and — once you see *why* — actually strengthens the paper.

---

## The short version

- The **Llama-tuned r128 config does not transfer.** On Qwen and Mistral it collapses (retrieval → 0).
- But this is **not** because "their KV is less compressible." It's a **rank-dependent failure**, and it
  spares exactly the regime BUG is built for.
- **BUG's extreme-compression niche (rank 16–64, ~0.04–0.15× of full KV) transfers cleanly** — near-perfect
  retrieval with healthy fluency on both new families. The **sweet spot is rank 64.**

So the honest headline is not "BUG is competitive at r128 everywhere" (false) but **"BUG's distinctive
extreme-compression frontier generalizes across three model families; the mid-compression config is
Llama-specific."** The second claim is both true and more interesting.

---

## What BUG stores, in one paragraph

At each step BUG keeps two things per layer: a **low-rank "gist"** of the whole retained history (a rank-`r`
subspace tracked by the Ceruti–Lubich streaming integrator — this carries *fluency*), plus a small
**exact tier** of `hh` verbatim tokens picked by "surprise" (this carries *sharp facts / needles* — this
carries *retrieval*). Perplexity mostly rides on the gist; needle retrieval mostly rides on the exact tier.
The two can fail independently, and this week they did.

---

## Why r128 breaks — two different mechanisms

We swept rank {16, 32, 64, 128} and measured memory, perplexity, and RULER retrieval on each family.

**Qwen — the needle gets absorbed.** At r128 Qwen's perplexity is *fine* (7.81, ~1.3× full) — the gist
reconstructs the text beautifully. But retrieval is **0**. The reason is the same "rank-vs-retrieval wall"
we found on Llama, only *earlier*: a bigger gist is a *smoother, more complete* summary, so a planted needle
stops looking surprising, never enters the exact tier, and is never retrieved. On Llama this wall arrives at
r256; on Qwen it's already at r128. (Week-15's `-s32` score-rank fix, which un-blinds selection on Llama,
does **not** recover it on Qwen — the wall is steeper here.)

**Mistral — the integrator diverges.** At r128 Mistral's perplexity *itself* blows up (43.9). This is a
**numerical** failure of the streaming low-rank integrator, not a selection failure: tracking more rank on
Mistral's KV stream makes the reconstruction unstable. (The pure-gist `bug` run shows the clean signature:
perplexity *improves* rank 16→64 as Eckart–Young says it should, then *diverges* — Qwen only at r256,
Mistral by r128. A batch SVD can't do that; a streaming DLRA integrator can, when the KV is ill-conditioned.)

**What it is NOT.** We ruled out the obvious culprits. Mistral has Llama's *exact* head geometry (n=1024)
and *no* QKV bias, yet still fails — so it isn't head count, and isn't Qwen's bias. It's a property of how
each model's KV cache interacts with a high-rank streaming low-rank tracker, and the safe-rank threshold is
**model-dependent**: Llama tolerates r256, Qwen r128 (fluency) but loses retrieval by r128, Mistral r64.

---

## Why the niche survives — and why that's the point

Below the threshold (**rank 16–64**) everything works on both families:

| | memory | perplexity | single | multivalue | var-track |
|---|---|---|---|---|---|
| Qwen r64 | **0.148×** | 8.18 (full 6.22) | 1.00 | 1.00 | **1.00** |
| Mistral r64 | **0.085×** | 5.50 (full 4.87) | 1.00 | 1.00 | **0.50** |

BUG's perplexity here is *above* ThinK/Palu (whose channel-pruning / low-rank factorization keep ppl very
close to full). **BUG does not win on perplexity.** What it does is reach **full single+multivalue retrieval
at 0.04–0.15× of full KV**, where ThinK and Palu sit at **0.50–0.75×**. That is a **3.4–10× advantage in
float-equivalent stored state at matched retrieval**. Channel-pruning / low-rank factorization are
structurally floored at 0.50–0.75×; eviction can reach 0.1× but loses var-track there (ea-k0.1 vt 0.17)
and its multikey decays with length (92→67→50) — all Week-11 measured. This is exactly the frontier the
paper is about, and it now holds on three model families.

The one honest blemish: **Mistral variable-tracking** tops out at ~0.50 even at low rank. Following a
`V0=…; V1=V0; …` chain needs *several* linked facts retained together; Mistral's exact tier doesn't keep the
whole chain. That's a real limitation to state, and a Week-17 target.

---

## The one open puzzle

`bugSseed-r128` perplexity was **467** with the big exact tier (`h1024`, the tier-2 config) but **7.81** with
the small one (`h256`, the sweep config) on Qwen. A *larger* verbatim tier should only *help* fluency, not
destroy it — so something about the large exact tier interacting with the r128 gist (surprise-scoring? the
warm-up seed? position handling?) is destabilizing at high rank. Unexplained; flagged for Week-17.

---

## Takeaway for the paper

Lead with the frontier, not the config: **"a dynamical-low-rank KV compressor whose extreme-compression
regime (≤0.15×) transfers across Llama, Qwen, and Mistral,"** report the model-dependent rank threshold as
an honest characterization (not a failure), name the Mistral-vt limit, and keep the mechanism (gist =
fluency, exact tier = retrieval; the two fail independently) as the through-line.
