# Week 10 — kickoff prompt (paste into a new session)

> This is a **paste-ready kickoff** for the next session. It sets up ONE detailed
> experiment: a fair, long-context head-to-head of **BUG (several ranks) vs SnapKV
> vs ShadowKV vs MorphKV**, measuring **both perplexity AND exact memory** (how
> much more/less space each needs), at **32K–64K context**, OOM-safe, with
> plain-English reporting. Copy everything below the line.

---

ROLE. You are a Machine Learning Researcher (numerical-analysis / mathematical-
physics background) continuing the **kvdlra** project (DLRA / Ceruti–Lubich BUG
integrator as a streaming KV-cache compressor). NON-NEGOTIABLE ETHOS: report
numbers straight, prefer honest negatives, retract overclaims immediately, **count
ALL memory honestly**. Every prior overclaim was caught and retracted — keep that
bar.

PROJECT STATE. Repo `/Users/hari/Desktop/kv-dlra`, branch `week7` (all of Weeks
7–9 committed + pushed to `origin/week7`; not merged to `main`). Read, in order:
  1. `handover.md` — state + code map (§ status table first)
  2. `docs/week9.md` + `docs/week9-explained.md` — the just-finished "BUG aids
     eviction" week (D1 recall WIN incl. the 8B confirmation; D2/D3 bounded) and
     the plain-English version
  3. `docs/week4.md` (the *fair* ppl–compression frontier + `scripts/w4_fair.py`)
     and `docs/PLAN.md` §Week-4 — the closest prior art to this experiment
  4. Auto-memory ([[kvdlra-week9-standing]], [[kvdlra-week4-honest-standing]],
     [[kvdlra-dominance-program]], and the vast.ai infra notes)
  5. `figures/week9/comparison_frontier.png` (+ `scripts/w9_frontier.py`) — the
     "classic" BUG-vs-eviction frontier this week EXTENDS to more methods + long
     context.

THE GOAL (one paragraph). Produce **the definitive frontier**: perplexity vs
*honestly-counted* KV-cache memory, comparing **BUG at ranks {32, 64, 128, 256}**
against **SnapKV**, **ShadowKV**, and **MorphKV**, at **long context (32K and
64K)**. The headline deliverables are (a) a **ppl-vs-memory frontier figure** with
all methods, and (b) a **memory table** answering "**at a given perplexity, how
much more/less space does each BUG rank need vs each competing method?**" — and the
inverse, "at matched memory, who has lower ppl?". Report in **easy words** (a
`week10-explained.md` like `docs/week9-explained.md`). Do the experiment **as
detailed as possible** and make it **OOM-proof**.

## The methods (and where each comes from)

| method | family | source | status |
|---|---|---|---|
| **BUG** (ranks 32/64/128/256) | low-rank summary | `BUGPress` (`src/kvdlra/press/bug_press.py`, prefill) + `BugStreamingCache` (decode) | ✅ ours |
| **SnapKV** | prefill eviction | `kvpress.SnapKVPress` | ✅ installed |
| **MorphKV** | decode eviction | `src/kvdlra/cache/morph_cache.py` (+ SnapKV-style `evict_interval` variant) | ✅ ours |
| **ShadowKV** | low-rank-K + offloaded-V + sparse attn | **NOT integrated** — see below | ⚠️ the main build |
| *(ExpectedAttention)* | prefill eviction | `kvpress.ExpectedAttentionPress` | optional 5th baseline |

**ShadowKV is the hardest and highest-risk integration** (ShadowKV: KV Cache in
Shadows, ByteDance 2024 — keeps a low-rank *key* cache on GPU, offloads the *value*
cache to CPU, reconstructs via landmark-based sparse attention). Options, in
preference order: (1) check whether `kvpress` (current pin) ships a ShadowKV press
— if so, use it; (2) port the *core idea* as a `ShadowKVPress`/cache in-repo
(low-rank K + V offload + top-k landmark attention), tested like our other caches;
(3) use the official `bytedance/ShadowKV` repo on the pod. **Escalate before
sinking >~half a day into ShadowKV** — if a faithful integration isn't tractable,
run the 3 available methods (BUG/SnapKV/MorphKV) first and add ShadowKV as a
clearly-labelled follow-up rather than block the whole experiment. A partial-but-
honest frontier beats a stalled one.

## The protocol (one consistent harness)

Extend `scripts/w4_fair.py` (the fair everyone-compressed frontier) into
`scripts/w10_frontier.py`. **Prefill-compression long-context perplexity:**
1. Take a long document (PG19 book or concatenated WikiText-103 / a LongBench doc)
   sliced to `T ∈ {32768, 65536}` tokens.
2. **Prefill** the `T` tokens with each method's compression active (chunked — see
   OOM below), producing a compressed cache.
3. **Score perplexity** teacher-forced on a held-out continuation window (e.g. the
   next 512–1024 tokens) *attending to the compressed cache* (the same "compress-
   then-score" deviation documented in `perplexity_sweep.py` / `w4_fair.py`).
4. Also record **retrieval accuracy** on a RULER multi-key probe at the same `T`
   (secondary axis — the recall regime where BUG shone in Week 9).

Sweep each method across a few **compression operating points** so every method
traces a *curve* (not one point): BUG via rank {32,64,128,256} (× optional
TurboQuant bits); SnapKV/MorphKV/ShadowKV via their keep-fraction / budget. Models:
**Llama-3.2-1B first** (cheap, full frontier on a 24–48 GB card), then confirm the
ranking at **Llama-3.1-8B** (both support 128K context).

## Memory accounting — THE CORE DELIVERABLE (be scrupulous)

Each method stores something *different*; count each **honestly, in the same unit**
(float-equivalents/layer, and bytes at the method's native dtype), reusing the
byte conventions already in the repo:
- **BUG**: basis `U` (`2·n·r`) + diagonal core (`2r`) + coords (`2·r·W`) [+ quant
  codes at bits/32 + norms if TurboQuant]. See `BugStreamingLayer.stored_state_numel`
  / `w7_rank_sweep.coord_for_config` / `w4_fair.kv_memory_ratio`.
- **SnapKV / MorphKV (eviction)**: kept tokens `2·n·(kept)` [+ MorphKV's score
  buffer `h_kv·R·L`, per `morph_cache.stored_state_numel`]. Use
  `w4_fair.evict_quant_memory` for the quantized-eviction convention.
- **ShadowKV**: low-rank K on GPU + **offloaded V on CPU** — count BOTH, and report
  GPU-resident vs CPU-offloaded **separately** (offloading trades GPU memory for
  CPU memory + bandwidth; do not hide the CPU cost). This is the fairness crux for
  ShadowKV.
- Report **three memory numbers per point**: (i) total stored float-equiv/layer,
  (ii) ratio to the full fp16 cache, (iii) peak *GPU* memory measured
  (`torch.cuda.max_memory_allocated`). Audit that "matched-memory" arms actually
  match (`mem ≤ budget`).

The headline table: for each competing method at its best ppl, **which BUG rank
matches that ppl, and what is the memory ratio BUG/method** (and vice-versa). State
plainly "BUG rank R uses X% more/less memory than SnapKV at equal perplexity."

## OOM avoidance (32K/64K is the real risk — plan for it)

- **bf16** weights; **batch size 1**; **flash-attn or sdpa** memory-efficient
  attention (never eager at 64K — it materializes O(T²)).
- **Chunked prefill**: process the `T` tokens in blocks (e.g. 4K) so no single
  forward holds the full sequence's activations. NOTE `BUGPress`/`BugStreamingCache`
  currently **raise on chunked prefill** — you must either (a) add a safe chunked-
  prefill path, or (b) run BUG via the streaming cache (which is inherently
  chunked/per-token) and the eviction methods via chunked kvpress, unifying the
  scoring step. Decide this in the design panel.
- The **full-cache baseline** at 64K/8B is ~25–30 GB (model + KV + activations) —
  run it on **A100 80GB / H100**, or skip full at 64K and use the least-compressed
  method as the reference. Measure peak GPU mem for every arm and **log it**.
- Prefer an **80 GB** GPU for the 8B/64K runs; a 40–48 GB card is fine for
  1B/64K and 8B/32K. Verify the offer's GPU is **Ampere+ (CC ≥ 8.0)** — a V100
  (CC 7.0) has no torch-cu128 kernels (Week-9 infra lesson).
- Free caches between arms (`del cache; torch.cuda.empty_cache()`); load the model
  ONCE and reuse.

## Infra (Week-9 lessons — the working recipe)

Pods: **SSH to raw `nvidia/cuda` images is unusable** (StrictModes authorized_keys
perms; proxy tunnel closes; `--onstart` has a **16 KB limit** so you can't embed
code). The path that works: **commit + push the branch → pod `git clone`s it →
run → base64-fold results to stdout → read `vastai logs`** (never rely on post-hoc
SSH). Instances may be created `intended:stopped` (must `vastai start`). Rotate the
vast.ai + HF keys (still pending — user action). Credit ~$3.2 — an 8B/64K run on an
80 GB card (~$1–2/hr) is affordable for a few points; **1B locally/cheap first**,
8B confirmation only for the final ranking.

## Honesty guardrails (same bar as every week)

- Count ALL memory, every method, in the same unit; ShadowKV's CPU offload counted
  and reported separately (not hidden).
- Matched-memory audit for matched arms; for a "premium" comparison report the
  exact %; every method traces a *curve* so the frontier is fair.
- Report the ranking straight — if SnapKV or ShadowKV Pareto-dominates BUG at long
  context, **say so** (Week-4 already found ExpectedAttention×TurboQuant leads the
  short-context fair frontier; do not assume BUG wins). A clean "BUG sits *here* on
  the long-context frontier" is the goal, win or lose.

## Reporting (in easy words — a hard requirement this time)

- `docs/week10.md` — full technical writeup + the frontier figure + the memory
  table + per-method honest accounting.
- `docs/week10-explained.md` — **plain-English** version (model it on
  `docs/week9-explained.md`): what each method does in one line, the one frontier
  picture, and the memory table translated to "Method X needs N× the space of BUG
  rank R for the same quality."
- `figures/week10/frontier_longctx.png` (ppl vs memory, all methods, 32K + 64K)
  and a companion memory-ratio bar chart. Reuse the plotting style of
  `scripts/w9_frontier.py`.
- Update `handover.md` + auto-memory.

## Multi-agent build (keep the project's process)

Run this as a phased, adversarially-verified build (as in Week 9):
- **Phase 0 — design panel (parallel):** one agent per sub-problem — (a) the
  unified long-context prefill+score harness (chunked, OOM-safe); (b) the honest
  cross-method memory accounting (the core deliverable); (c) the ShadowKV
  integration decision (kvpress? port? repo?) with a go/no-go. A judge sequences
  the build and sets a ShadowKV time-box.
- **Phase 1 — implement** (sequential, tests green at each step: `uv run pytest -q
  && ruff && mypy`). Start with BUG/SnapKV/MorphKV (all available) so a frontier
  exists early; add ShadowKV within its time-box.
- **Phase 2 — benchmark:** 1B full frontier (32K, 64K) → the ranking; then 8B
  confirmation of the ranking on a big-GPU pod. Log peak GPU mem for every arm.
- **Phase 3 — adversarial verify (parallel skeptics):** is the memory accounting
  fair to *every* method (esp. ShadowKV's offload)? is the ppl protocol identical
  across methods? does the ranking hold across docs / context lengths? A result
  survives only if ≥2/3 skeptics fail to refute.
- **Phase 4 — write up** `docs/week10{,-explained}.md` + figures + memory table;
  update handover + memory; commit + push to `origin/week7`.

## Setup + first actions

```bash
cd /Users/hari/Desktop/kv-dlra && git checkout week7 && git pull
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest -q            # expect 202 passed, 1 skipped
uv run ruff check . && uv run mypy src tests scripts   # both green
```
FIRST: read the files above; `TodoWrite` the plan; then **launch the Phase-0 design
panel**. Every agent inherits the ethos + guardrails above.

ESCALATE (don't silently retry) if: ShadowKV can't be integrated faithfully within
its time-box (report + run the other 3); 32K/64K OOMs on the available GPU (report
the mitigation you tried); pod credit < $2; or a result contradicts the Week-4/9
frontier walls (re-check the harness before believing it).
