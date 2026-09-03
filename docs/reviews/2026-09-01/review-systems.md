# Systems-claims review — kvdlra NeurIPS readiness

Reviewer dimension: **systems claims** (memory / latency / throughput / deployability).
Repo: `/Users/hari/Desktop/kv-dlra`, branch `week7`. All citations are file:line in that tree.

## Verdict in one paragraph

The paper's memory headline (C1: "0.075–0.149× of full-KV memory, 5–13× less than ThinK/Palu")
is a **storage-accounting number, not a resident-memory number**, and the repo's own Tier-4
measurement says so: BUG is reconstruct-then-attend, its decode-time reconstruction workspace is
**≈0.98× full KV at ctx 2048** (`docs/week16-handover.md:20`), so actual decode residency is
**≈1.06× full KV — slightly worse than no compression at all**. There are **zero** measured
VRAM, latency, or throughput numbers at any flagship config in `results/`, and the one honest
caveat lives in a script docstring (`scripts/w16_storage.py:1–27`) and one handover table row —
it appears **nowhere** in the public narratives (`grep kernel|workspace|resident
docs/week16-explained.md docs/week17-explained.md` → zero hits). As worded, a systems-literate
NeurIPS reviewer reads C1 as a deployment claim and rejects. The repo, to its credit, has already
built the honest reframe (`w16_storage.py`) and the fix is cheap — but it has not been run to a
committed result, and the paper claims have not been rewritten around it.

---

## 1. The evidence, pinned

### 1.1 Reconstruct-then-attend is structural, and the workspace is *persistent*, not transient

- `src/kvdlra/cache/bug_cache.py:9` — "attention keeps seeing a running low-rank reconstruction";
  `:42` — attention input is `[sinks | RoPE(U C, true positions) | recent]`.
- The reconstruction is **cached across decode steps**: `_mid_k_cache` (`bug_cache.py:616`,
  `(n, mid_len)` in storage dtype) is rebuilt only on absorb events
  (`_ensure_mid_cache`, `bug_cache.py:1578–1581`: early-returns if present). So at every decode
  step the accelerator holds **stored state + a ≈full-length K/V reconstruction**. The
  `workspace_numel()` docstring (`bug_cache.py:1716–1720`) calls it "avoidable by recomputing
  each step" — true, but recomputing per step trades the memory for a per-step
  `U@C` + re-RoPE over the whole middle, i.e. exactly the cost a fused kernel exists to hide.
- `scripts/w16_storage.py:3–7` states the consequence outright: "a naive decode-time
  `max_memory_allocated` would show BUG with a HIGHER peak than full KV — the opposite of the
  paper's story, and a Mode-B low-rank attention kernel (future work) is what a real
  throughput/peak-VRAM win needs."

### 1.2 The measured numbers

- `docs/week16-handover.md:20`: "workspace ≈ 0.98× full → why naive VRAM backfires" (ctx 2048).
  Decode-resident ≈ stored (0.08–0.16×) + workspace (0.98×) ≈ **1.06–1.14× full KV**. The
  workspace ratio only approaches 1 from below as ctx grows (`1 − (sinks+recent+tier)/ctx`), so
  at the headline 16K/32K contexts residency stays ≈1.0×+stored.
- **The Tier-4 result was never committed.** `results/w16-storage.json` does not exist,
  `figures/week16/` does not exist (default outputs at `scripts/w16_storage.py:197–198`);
  the 0.98× figure survives only as prose in the handover. For a paper, this measurement
  currently has no artifact.

### 1.3 Zero systems measurements at scale (verified by search)

- `grep -rE 'latency|throughput|vram|max_memory_allocated|perf_counter' scripts/ results/ src/`
  finds only:
  - `results/w5-decode-validate-1b.json` — Week-5, **Llama-3.2-1B, CPU, float32**, pre-bugSseed
    config; and it shows BUG **slower**: per-token mean 224.9 ms (`bug-r128`) vs 204.5 ms (`full`).
    This is the repo's *only* end-to-end latency data, and it is unfavorable.
  - `scripts/w45_integrator_ablation.py:144–158` — integrator-only microbenchmark, not end-to-end.
  - `scripts/w10_frontier.py:486,550` records `peak_gpu_bytes` — but `results/w10-frontier-1b.json`
    has `"peak_gpu_bytes": null` (CPU run), and the GPU log-parse (`results/w10-gpu-parsed.json`)
    **dropped the field entirely** (zero `peak` keys). So the GPU campaigns produced no surviving
    VRAM numbers.
- All memory numbers in C1/C4 are `ratio_fp16` = stored bits / full-fp16-cache bits
  (`src/kvdlra/accounting.py:81–85`), a *storage* metric byte-identical to
  `stored_state_numel` — which explicitly excludes the workspace (`bug_cache.py:1716–1720`
  reports it "separately").

### 1.4 The baseline comparison is asymmetric

- In-repo **Palu is also reconstruct-then-attend** (`scripts/w10_frontier.py:353`), so
  BUG-vs-Palu at 0.50× is accounting-consistent *within the repo* — but the real Palu paper
  ships fused reconstruct-GEMM kernels, so its published 0.50× is deployment-real.
- In-repo **ThinK** (kvpress drop-in, `w10_frontier.py:337–347`) genuinely prunes the live key
  channels: its 0.75× **is** resident memory. "5–13× less memory than ThinK" therefore compares
  BUG's storage against ThinK's residency. On the resident axis today, ThinK (0.75×) and every
  eviction arm **beat BUG (≈1.06×)**.
- The repo knows this class of issue well — the ShadowKV port refuses to count CPU-offloaded
  values as free (`src/kvdlra/cache/shadow_cache.py:8–14`, "the forbidden misreport"). The same
  discipline has simply not been applied to BUG's own workspace in the paper-facing claims.

---

## 2. Question 1 — how fatal at NeurIPS?

**As currently worded: near-fatal for any reviewer who reads the memory claim as deployment.**
Every recent accepted KV-compression paper this would be compared to ships at least one realized
systems axis: H2O (throughput), SnapKV (measured decode memory+latency), KIVI (kernels +
throughput), Palu (kernels), ShadowKV (throughput; arXiv:2410.21465, ported in-repo). Papers
whose headline is a cache-size ratio get away with it because eviction/quantization ratios
*translate directly to residency*. BUG is the unusual case where the headline ratio does **not**
translate — residency is ≥1.0× — and no measurement in the repo says otherwise.

My estimate: with C1 as written ("5–13× less memory", unqualified — mirrored verbatim at
`docs/week17-explained.md:27–28`), **~3 of 4 reviewers flag it; for roughly half of those it is
the primary rejection reason** ("the paper's central number is not the quantity practitioners
mean by KV-cache memory"). Expect a confidence-4 systems reviewer to write "misleading" —
worse than "incomplete". If a reviewer additionally finds the only latency datum in the repo
(1B/CPU, BUG 10% slower), the rebuttal is very hard. **Fatality: fatal as worded; fixable by
reframing, because the quality results (C1 retrieval, C2–C3 mechanism) do not depend on it.**

## 3. Question 2 — can honest reframing save it? Yes, and the repo half-built it already.

### The reframe

Retitle the memory axis as **stored/persisted KV state**, and stand up the use cases where
storage *is* the binding constraint:

1. **KV offload & persistence** — prefix-cache stores, conversation resumption, multi-tenant
   cold caches, KV migration (vLLM prefix caching, LMCache/Mooncake-style tiers). Store 0.08–0.15×,
   reconstruct on load with one GEMM per layer. **Closest prior work is CacheGen (SIGCOMM'23,
   KV compression for network/disk loading) — it must be cited and positioned against, or a
   reviewer will do it for you.**
2. **Bandwidth-bound decode (projection, clearly labeled)** — the factored middle reads
   `(n·r + r·m)` floats/step vs `n·m` materialized: at r64, n=1024, 32K that is ≈0.28 GB/step
   vs ≈4.3 GB/step across 32 layers, ~7–12× less KV traffic — the quantitative case for the
   Mode-B kernel, presented as analysis, not result.
3. **Disclose the residency number in the main table**: decode-resident ≈1.06× today; kernel
   future work. The w16_storage docstring text is already the right paragraph — promote it from
   `scripts/w16_storage.py:1–27` into the paper.

### The minimal measurements that make it credible (~$10 GPU, ~1 day)

- **M1 (load-path win, the new headline systems number):** measured host→device transfer +
  reconstruct wall-clock vs full-KV transfer, Llama-3.1-8B @16K/32K. Full KV @32K fp16 =
  2·32768·1024·32·2B = **4.29 GB**; BUG r128-s32 state 0.16× ≈ 0.69 GB; r64-h256 ≈ 0.37 GB.
  At ~25 GB/s PCIe: ~170 ms vs ~27 ms + reconstruct GEMM (≈0.55 TFLOP total, ~5–20 ms on A100)
  → **a ~3–4× measured cold-load speedup**, growing with context and with slower tiers
  (NVMe/network). ~30 lines on top of existing cache code.
- **M2 (the disclosed elephant):** run `scripts/w16_storage.py` on CUDA at 16K/32K, commit
  `results/w16-storage.json` + figure, and add `measure_peak_gpu`
  (`src/kvdlra/accounting.py:378–388`) rows for bug/full/eviction arms so the ≈1.06× residency
  is a measured, disclosed number rather than a reviewer's discovery. (Currently the script's
  CUDA peak path only covers eviction arms — extend to all arms.)
- **M3 (optional, strengthens):** end-to-end resume latency — persist state, reload, first-token
  time vs recomputing prefill vs reloading full KV. This is the use case in one number.

With M1+M2 in the paper and C1 rewritten (below), the systems objection converts from
"misleading" to "scoped": I estimate the reject-on-systems fraction drops to ~15–25%
(some reviewers will still discount novelty absent a kernel, but that is a significance
argument, not an integrity one).

## 4. Question 3 — the fused-kernel path (Mode-B): scope and cost

What it is: a flash-decoding-style kernel where each middle KV tile is rebuilt **in SRAM**:
load `U` (n×r, shared per layer), a tile of `C` (r×T_tile) + positions; compute `K_tile = U@C_tile`
in-register; apply RoPE from gathered cos/sin; score/softmax-accumulate; merge partial softmax
with the verbatim segments (sinks, exact tier, recent) — flash-decoding already merges multiple
KV sources. Two simplifications specific to BUG:

- **The V side needs no kernel at all.** Values carry no RoPE, so
  `O = A·(U_v C_v)ᵀ = (A·C_vᵀ)·U_vᵀ` — exact, two plain GEMMs, implementable in PyTorch today.
  Only the K score path is blocked by per-position RoPE.
- **Direct precedent exists in-repo**: the ShadowKV port already reconstructs selected chunks
  on demand (`coeff[chunk] @ basis`, re-rotated at true positions —
  `src/kvdlra/cache/shadow_cache.py:26–28`); ShadowKV and Palu both shipped exactly this kernel
  class, so feasibility is not in question.

Complications: rank-adaptivity (bounded by r_max — pad), absorb events rewriting C (kernel
agnostic), GQA head mapping, and numerics parity tests against the current materialized path.

**Cost estimate:** a batch-1, decode-only Triton prototype with parity tests and one
throughput/peak-VRAM curve: **2–4 weeks** for someone fluent in Triton (the V-side factorization
is days). A paper-grade kernel section: ~1 month. Production (prefill chunking, batching, CUDA
graphs, vLLM integration): 3+ months. **Recommendation: do NOT gate the paper on it** — ship
M1/M2 + the bandwidth analysis, cite Palu/ShadowKV as the realization path, keep the kernel as
explicit future work.

## 5. The sentence of C1 that must be rewritten

Current (briefing C1; verbatim source `docs/week17-explained.md:27–28`):

> "At **0.085–0.149× of full KV** — **5–13× less memory than ThinK (0.75×) or Palu (0.50×)** —
> BUG matches or beats them on retrieval."

Required rewrite (or equivalent):

> "at 0.075–0.149× of full-KV **stored state** (float-equivalent accounting, Sec. X) — 5–13×
> less **persisted cache** than ThinK/Palu at matched retrieval. Under the current
> reconstruct-then-attend implementation, decode-time **working memory remains ≈1.0× full KV**
> (measured, Sec. X); realizing the storage ratio as resident memory requires a fused factored-
> attention kernel (as in Palu/ShadowKV), which we leave to future work. The storage ratio is
> realized today on the cache load/persistence path (Sec. X: N× faster KV load)."

Likewise "a regime eviction/channel-pruning cannot enter" must be scoped to the storage axis —
on the resident axis, eviction currently occupies a regime BUG cannot enter.

## 6. Score (systems-claims dimension)

**3/10** as it stands: headline memory claim not realized in residency, zero systems
measurements at scale, the one honest measurement uncommitted, the caveat undisclosed in every
paper-facing narrative, and the only latency datum unfavorable. The underlying accounting
library is genuinely honest (`accounting.py`, `shadow_cache.py:8–14`) and the repair is cheap
(~$10 GPU + rewording + one cited paper), which is why this is 3 and not 1–2 — but the repair
has to actually happen before submission.

### Gap-fill list (cost-ordered)
1. Rewrite C1/C4 memory sentences to "stored state"; scope "cannot enter" claim — **$0, hours**.
2. Promote the `w16_storage.py:1–27` paragraph into the paper's limitations/systems section — **$0**.
3. Run + commit `w16_storage` on CUDA @16K/32K incl. peak-VRAM for all arms (M2) — **~$5 GPU**.
4. Measured H2D-load + reconstruct vs full-KV load, 8B @16K/32K (M1) — **~$10 GPU, ~1 day**.
5. Bandwidth-bound decode analysis table (bytes/step, labeled projection) — **$0, CPU-only**.
6. Cite + position against CacheGen (SIGCOMM'23) and vLLM prefix-cache/LMCache offload context — **$0**.
7. (Optional) V-side factored attention in PyTorch (exact, no kernel) to show partial realization — **days**.
8. (Deferred) Mode-B Triton decode kernel prototype — **2–4 weeks**; not needed for an honest submission.
