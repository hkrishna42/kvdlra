# kvdlra paper-readiness review — shared briefing (read first)

You are one reviewer on a NeurIPS/ICML-style panel evaluating whether the **kvdlra** research program
(repo `/Users/hari/Desktop/kv-dlra`, branch `week7`) is ready to be written up as a **NeurIPS main-track
paper**. Your job: rigorous, evidence-grounded review in your assigned dimension. Be adversarial the way a
real Reviewer 2 is — but ground every criticism in a file, a number, or a cited paper. Separate
"fatal flaw" from "fixable gap" from "nitpick".

## What the method is (one paragraph)
**BUG** is a streaming KV-cache compressor for LLM inference. Per layer it maintains (1) a rank-`r`
low-rank "gist" of the retained K/V history, tracked online by the rank-adaptive **BUG integrator**
(Ceruti–Kusch–Lubich 2022, dynamical low-rank approximation, `src/kvdlra/integrators/streaming_torch.py`),
and (2) a small **exact tier** of `hh` verbatim tokens selected by **surprise** = out-of-subspace residual
`||k − UUᵀk||/||k||` (`src/kvdlra/cache/bug_cache.py`). Mechanistic decomposition: gist carries fluency
(ppl), exact tier carries retrieval (needles). Knobs: warmup-seed (seed the tier from the first chunk),
score-rank decoupling (`-s32`: score surprise against leading columns only), and the Week-17
`min_sv_frac` relative singular-value floor (caps tracked rank at the stream's numerical rank;
default-off). Prefill is chunked ingest; decode re-applies RoPE at true positions;
attention runs on **reconstructed** K/V (reconstruct-then-attend).

## The would-be paper claims (evaluate THESE against evidence)
C1. **Extreme-compression frontier generalizes**: `bugSseed-r64-h256` gets RULER single+multivalue
    retrieval = 1.00 (12/12, Wilson [0.76,1.0]) on Llama-3.1-8B, Qwen2.5-7B, Mistral-7B-v0.3 at 16K, at
    **0.075–0.149× of full-KV float-equivalent memory** — 5–13× less than ThinK(0.75×)/Palu(0.50×) at
    matched retrieval, "a regime eviction/channel-pruning cannot enter". Also 1.00 at 32K (n=4).
C2. **Mechanism**: gist=fluency vs exact-tier=retrieval decomposition; the rank-vs-retrieval wall
    (bigger gist absorbs the needle) with model-dependent onset; score-rank decoupling fixes it on Llama.
C3. **`min_sv_frac` floor**: one default-off fix removes the whole high-rank divergence class
    (Qwen bug-r256 ppl 27531→6.995; Mistral bug-r128 138→5.574; the h1024 "puzzle" 714→6.941) and
    extends the safe rank (floored high rank = best fluency, monotone in rank).
C4. **Marquee**: Llama r128-h1024-s32 @32K var-track 0.94 [0.72,0.99] (n=16) beats ThinK (0.31)
    Wilson-separated; leads Palu (0.56) not separated; ties ThinK/Palu ppl at 3–5× less memory (Week-15).
C5. Honest limits stated: BUG does NOT beat baselines on ppl; var-track weak on 1024-dim models
    (Llama 0.58, Mistral 0.50 vs Qwen 1.00), no working fix; 32K r64 fluency dip Qwen-only.

## Evidence map (ground truth — READ what your dimension needs)
- Results w/ CIs: `results/w17-decision-table.json`, `results/w17-ruler-intervals.md`,
  `results/w16-ruler-intervals.md`, `results/w11-decision-table.json`, `docs/week11-decision-table.md`
  (the full method×task Llama table: full/EA/SnapKV/MorphKV/ThinK/Palu/Shadow/bugS variants @16/32/64K).
- Narratives: `docs/week17-explained.md`, `docs/week17-handover.md`, `docs/week16-explained.md`,
  `docs/week16-handover.md`, `docs/week10_report/index.html` (7-method comparison + regime map).
- Code: `scripts/w10_ruler.py` (**NOTE: the RULER tasks are CUSTOM-BUILT in-repo** — synthetic
  needle/multikey/multivalue/vt with our own filler + templates, NOT the official RULER suite),
  `scripts/w10_frontier.py` (ppl on wikitext-103 sliding windows), `scripts/w10_longbench.py`
  (LongBench harness — **never run on the flagship config**; Tier-3 deferred),
  `scripts/w16_storage.py` (**Tier-4: measured decode WORKSPACE ≈ 0.98× full** — reconstruct-then-attend
  means NO naive VRAM/latency win; a fused kernel is explicitly future work),
  `src/kvdlra/` (the library), `tests/` (parity ladders, family fixtures, floor tests).
- Program history (17 weeks, honest walls): Week-4 "competitive, NOT a SOTA winner"; Week-7/8 dominance
  map = two measured walls (near-oracle ceiling at moderate compression; overhead floor at extreme);
  Week-9 "BUG aids eviction" pivot; Week-10 7-method comparison (no single winner; BUG uniquely reaches
  0.033×; BUG=0% on RULER@32K pre-fix vs EA=100%); Weeks 13–15 warmup-seed + score-rank fixes;
  Week-16 generality + two failure modes; Week-17 (above).
- Eval scope caveats you must weigh: models 1B–8B only; contexts 2K–64K (64K sparse); n=12 @16K,
  n=4 @32K (n=16 marquee); single-sequence batch=1; custom synthetic retrieval tasks; ppl corpus =
  wikitext-103; memory = float-equivalent `ratio_fp16` (storage accounting), not measured VRAM;
  no latency/throughput numbers; no quantization baselines (KIVI/2-bit KV etc. ABSENT); no
  official-benchmark numbers (official RULER/LongBench/∞Bench/NIAH-suite absent).

## Output discipline
Return the structured object AND write your full review to
`/private/tmp/claude-501/-Users-hari-Desktop-kv-dlra/244f8cf1-60eb-421a-93d6-bc76db5fcb25/scratchpad/review-<your-slug>.md`.
Every claim: file:line / number / citation. Rate your dimension 1–10 (NeurIPS calibration: 4=reject,
5=borderline reject, 6=borderline accept/poster, 7=accept, 8=strong). List concrete gap-fills with rough
cost (CPU-only / ~$10 GPU / ~$50 GPU / weeks of work). $0 — read-only review; do NOT edit the repo,
do NOT launch pods.
