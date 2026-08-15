# kvdlra

**Streaming KV-cache compression for LLMs via Dynamical Low-Rank Approximation** — the
Ceruti–Lubich *BUG* integrator from numerical analysis
([arXiv:2010.02022](https://arxiv.org/abs/2010.02022),
[arXiv:2104.05247](https://arxiv.org/abs/2104.05247)) turned into an *online* KV-cache
compressor: a rank-`r` low-rank **gist** of the past, a **surprise-selected exact tier** for
the sharp facts a summary can't hold, and a **warm-up seed** that repairs early-context
retrieval — a principled alternative to greedy eviction (H2O, SnapKV, ExpectedAttention).

DLRA tracks a moving low-rank matrix without the σ_min stiffness that breaks naive (U, S, V)
ODE schemes, with an error bound *independent of the smallest singular value*. kvdlra uses it
as a near-oracle streaming tracker and builds a bounded-memory decode cache on top. It is also
wired into NVIDIA's [kvpress](https://github.com/NVIDIA/kvpress) as `BUGPress`.

## What it is

`BugStreamingCache` ([`cache/bug_cache.py`](src/kvdlra/cache/bug_cache.py)) is a drop-in
HuggingFace cache that compresses the KV stream online into parts, all counted in **one
honest float-equivalent unit** (`accounting.py`, every buffer charged):

- a **rank-`r` BUG gist** of the bulk — the low-rank summary;
- a small **exact "SLASH" tier** — the highest-*surprise* (out-of-subspace residual) tokens
  kept verbatim, exactly the sharp facts a low-rank gist reproduces worst (a planted needle, a
  rare code), selected attach-free;
- **attention sinks + a recent ring** (StreamingLLM-style).

**Week 13 adds a warm-up seed** (`--warmup-seed`, default off). The first ingest chunk used to
bypass the exact tier, so the earliest-planted facts were structurally lost — the *warm-up
window* that made the family a ≥32K-only method. Routing that first chunk through the same
surprise path (scored against the strictly-older, needle-free basis) fixes it.

## Where it stands (honestly)

BUG is a **competitive compressor with distinct niches — not a universal SOTA**. At moderate
compression, token eviction (ExpectedAttention, MorphKV) is near-lossless and often ahead. A
*dominance program* ([`docs/week7-dominance.md`](docs/week7-dominance.md)) proved BUG cannot
beat eviction everywhere: a regime split bounded by two **measured walls** — a near-oracle
tracking ceiling (~1% off the Eckart–Young optimum; BUG beats Frequent Directions, incremental
SVD, and Oja) and a structural basis-overhead floor that whole-token eviction doesn't pay.

Where BUG **wins**:

- **Extreme compression (<0.05× memory).** Eviction turns catastrophic; the low-rank gist
  degrades gracefully. `bugEVICT` finds the single needle at **0.009×**, ~11× under
  ExpectedAttention.
- **Long-context retrieval-per-byte** (RULER, Llama-3.1-8B @ 32K). `bugS-r32-h256` covers all
  four tasks — needle / multi-key / multi-value / var-track = **100 / 67 / 100 / 100 at
  0.043×** — the cheapest point of the only sub-0.1× family that does.
- **Constant-memory streaming decode** (Weeks 5–6).

**Week-13 headline — the warm-up seed fixes 16K.** At 16K the family used to collapse on the
hard tasks. With the seed, a matched A/B at *identical* memory (Llama-3.1-8B, RULER n=2×2):

| 16K RULER — single / multi-key / multi-value / var-track | `bugS` | `bugSseed` |
|---|---|---|
| **r32-h256** (0.05×) | 100 / 0 / 0 / 0 | **100 / 100 / 100 / 100** |
| **r128-h1024** (0.19×) | 100 / 25 / 25 / 0 | 75 / 100 / 25 / 75 |

Perplexity is unchanged-to-slightly-better at the same footprint (r32 @32K 9.16 → 9.09). Full
accounting: [`results/w13-trackb-summary.md`](results/w13-trackb-summary.md).

### Current focus (Week 15 — both axes at 3–5× less memory)

**BUG now matches the low-rank KV-compression field on perplexity *and* beats it on retrieval,
at a paper-grade memory advantage.** The headline arm `bugSseed-r128-h1024-s32` (0.159× memory)
adds a **score-rank decoupling** knob to the Week-14 warm-up seed: `--score-rank 32` caps the
surprise-scoring basis at a leading rank-32 subview (default-off, **zero** memory cost, +0.3%
ppl), un-blinding the exact-tier selection that a large r128 gist otherwise "fits" into
invisibility. Confirmed on Llama-3.1-8B (n=8 ppl / n=4–8 RULER, GPU):

- **Perplexity (32K, same 8 windows for every arm):** full 6.975 · think-c0.5 7.196 ·
  palu-r0.5 *(fixed)* 7.232 · **bugSseed-r128-s32 7.353** — a statistical **tie** with ThinK
  (+0.031 bits/tok) and Palu (+0.024), inside the pre-registered 0.05 band, at **3.2–4.7× less
  memory**. (Versus uncompressed full it is +0.076 bits/tok — BUG matches the compression
  *field*, not full KV.)
- **Retrieval:** the s32 cap lifts the hardest tasks at r128 — **32K var-track 0→100** (Wilson-
  disjoint), 16K multi-value 25→100, 16K var-track 75→100 — so it **beats ThinK (vt 50) and
  Palu (vt 25) on var-track** at 3–5× less memory.

**Two "baselines" turned out to be our own bugs, now fixed and validated on 8B** (a prerequisite
of any honest claim): ShadowKV's published 0/0/0/0 was a harness defect (decode ran outside
`attach()`), now retrieving 100/100 single/multikey; Palu's worst-in-table perplexity was our
port low-ranking the attention sinks, now a strong 7.232 (was 9.236). We also ship **per-window
NLL error bars**, **Wilson intervals on every RULER cell**, and a **perplexity-significance
framework** ([`docs/week15-significance.md`](docs/week15-significance.md)) — e.g. a ppl gap of
9.0 vs 7.5 is 0.263 bits/token, a large, decisively-significant difference; 0.04 bits/token is a
tie.

**Completed (n=8):** the `s32` arm scores a clean **100/100/100/100 at both 16K and 32K**, and
its 16K perplexity (5.434) ties ThinK/Palu within 0.006 bits/token — near full-KV — at 2.6–3.9×
less memory. **Honest notes:** the decoupling is dead at r256 (past the rank-retrieval cliff);
the airtight single retrieval result is 32K var-track (0/4→4/4, Wilson-disjoint); some 16K lifts
are marginal-n; vs uncompressed full KV, BUG is ~5% higher perplexity at 32K. Record:
[`results/w15-confirm-summary.md`](results/w15-confirm-summary.md),
[`results/w15-complete-summary.md`](results/w15-complete-summary.md); explainer
[`docs/week15-explained.md`](docs/week15-explained.md).

### The compression tradeoff (Week 4, the fair control)

![Fair comparison: every mechanism × TurboQuant](figures/week4/fair.png)

Perplexity vs. stored KV memory (Llama-3.2-1B, WikiText-2, ctx 1024), **every mechanism
quantized equally**. Low-rank (feature axis) and quantization (bit axis) genuinely *compose*,
but under a fair comparison **BUG×TurboQuant is competitive, not the winner** — Expected
Attention is on/ahead of BUG's frontier through the mid-aggressive band. BUG's edge is the
extreme edge and the retrieval/streaming niches above. Full accounting (incl. a retracted
unfair-comparison claim) in [`docs/week4.md`](docs/week4.md).

## How it works

1. **Streaming BUG tracker** ([`integrators/streaming.py`](src/kvdlra/integrators/streaming.py),
   [`streaming_torch.py`](src/kvdlra/integrators/streaming_torch.py)) — a rank-adaptive
   augmented-BUG subspace tracker for the column-streamed KV matrix; near-oracle (within
   ~1.01–1.03× of truncated SVD). Operates **pre-RoPE** (roughly halves the error); a blocked
   torch variant runs on GPU.
2. **`BugStreamingCache`** ([`cache/bug_cache.py`](src/kvdlra/cache/bug_cache.py)) — the decode
   cache: BUG gist + surprise-selected exact tier (SLASH) + sinks/recent ring + the warm-up
   seed; chunked ingest for OOM-safe 32K/64K pre-fill. Footprint pinned by `accounting.py`.
3. **`BUGPress`** ([`press/bug_press.py`](src/kvdlra/press/bug_press.py)) — a `kvpress` press
   that reconstructs the KV cache at rank `r` during pre-fill (pre-RoPE), for the
   perplexity/compression track ([`docs/week3.md`](docs/week3.md)).
4. **TurboQuant** ([`quant/`](src/kvdlra/quant/)) — PolarQuant + QJL + product quantization,
   composable with the coordinate factors ([`docs/week4.md`](docs/week4.md)).

## Install

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest -q          # 295 passed, 1 skipped (bf16 QR skips on CPU LAPACK)
```

## Reproduce

Long-context retrieval + perplexity (the current headline; one A100):

```bash
# RULER retrieval, bugS vs bugSseed A/B at 16K (drop --warmup-seed for the bugS baseline)
PYTHONPATH=src uv run python scripts/w10_ruler.py --model unsloth/Meta-Llama-3.1-8B-Instruct \
    --device cuda --context-lens 16384 --tasks niah_single niah_multikey niah_multivalue vt \
    --methods bugslash --ranks 32 --hh-budgets 256 --hh-neighbor 1 --chunk 4096 --warmup-seed \
    --n-trials 2 --seeds 0 1

# perplexity frontier
PYTHONPATH=src uv run python scripts/w10_frontier.py --model unsloth/Meta-Llama-3.1-8B-Instruct \
    --device cuda --T 16384 32768 --chunk 4096 --methods bugslash --ranks 32 128 --no-ruler
```

GPU runs are orchestrated as vast.ai pods via MODE-dispatched `scripts/pod/*.sh`. The Week-4
compression figure:

```bash
uv run python scripts/w4_hybrid_sweep.py --ranks 32 64 128 --bits fp 4 3 2 \
    --context-len 1024 --target-len 512 --n-windows 16
```

## Layout

- `src/kvdlra/cache/` — `BugStreamingCache` (`bug_cache.py`: BUG gist + SLASH exact tier +
  warm-up seed; chunked ingest), MorphKV / ShadowKV baselines
- `src/kvdlra/integrators/` — BUG (`bug.py`, `bug_adaptive.py`, `bug_class.py`), torch port
  (`bug_torch.py`), streaming trackers (`streaming.py`, `streaming_torch.py`), baselines
  (`oja.py`, `frequent_directions.py`)
- `src/kvdlra/press/` — `BUGPress` (`bug_press.py`), Palu/Turbo presses, transformers≥5.8 shim
- `src/kvdlra/quant/` — TurboQuant: `polar.py` (PolarQuant), `qjl.py` (QJL), `product_quant.py`
- `src/kvdlra/accounting.py` — the one-unit float-equivalent memory accounting (all buffers)
- `scripts/` — KV capture, RULER (`w10_ruler.py`), perplexity/frontier (`w10_frontier.py`),
  Q-BUG calibration (`w12_calibrate_qkey.py`), Week-13 CPU probes (`w13_*`), pod launchers (`pod/`)
- `docs/` — weekly writeups `week1.md … week13-*`; key ones: [week4](docs/week4.md) (fair
  comparison), [week7-dominance](docs/week7-dominance.md) (the walls),
  [week9](docs/week9.md) (recovery-tier recall), [week11-decision-table](docs/week11-decision-table.md),
  [week12](docs/week12.md) (attribution + Q-BUG), [week13-plan](docs/week13-plan.md) (the portfolio)
- `paper/` — arXiv-style preprint draft (`main.tex`)
- `dashboards/` — self-contained HTML visual writeups (see below)

## Visual writeups (the story, without the code)

Two self-contained HTML dashboards in [`dashboards/`](dashboards/) — open either file in a
browser, no server needed:

- **[The research story, explained simply](dashboards/research_story_dashboard.html)** — the
  plain-language arc (weeks 1–11): the problem (a KV cache costs more than the weights), the
  idea (a physics-simulation low-rank integrator as an online compressor), the week-by-week
  journey, the *map of who-wins-where*, the two clearest wins (extreme-compression needle at
  0.009×; long-context retrieval-per-byte), and how we keep ourselves honest.
- **[Beyond the Two Walls: a mathematical attack plan](dashboards/research_brainstorm_dashboard.html)**
  — the research brainstorm treating the two measured walls (near-oracle tracking ceiling,
  basis-overhead floor) and the retrieval fidelity wall as *axioms*, then eight
  mechanism-targeted proposals — each naming the wall it attacks, the math, an adversarial
  pre-mortem, and a **falsifiable bar with a kill criterion**. It seeded the Week-11→13 work
  (the SLASH exact tier, Q-BUG's query-metric geometry, the integrator-surgery track).

Live interactive companions (rendered from the results JSON) are also published as Claude
Artifacts — the pooled **decision table** (every method side-by-side at 16K / 32K / 64K), the
**overview** (BUG finds a needle at 32K, the warm-up window, the Week-12 attribution + Q-BUG),
and the **plain-language explainer** (*"Finding a needle you've already thrown away"*, 11 steps
through to the Q-BUG quality knob). The committed `dashboards/*.html` above are the openable,
version-controlled record; the source data is under [`results/`](results/).

## Honest caveats

- **Competitive, not universal SOTA.** Eviction is near-lossless at moderate compression; BUG's
  edges are extreme compression, streaming decode, and long-context retrieval-per-byte.
- **Small-n on retrieval.** RULER cells are n=2–4 (25–50 pts/trial) — a single flipped trial
  moves a cell. Headline claims need a higher-n (n≥8) re-run with error bars; the two Week-13
  single-cell dips (r128@16K single, r32@32K var-track) are one-trial effects.
- **Single model family** (Llama-3.1-8B + 3.2-1B) and **no systems metrics yet** — reported
  memory is the honest float-equivalent ratio, not measured throughput / latency / peak-VRAM.
  LongBench coverage is partial.
- New arms (`bugSseed` warm-up seed, `w_key` Q-BUG query-whitening) ship **default-off** pending
  higher-n confirmation.

## License

Apache-2.0.
