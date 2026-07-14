# Week 10 — session handover (read this first, then `docs/week10-plan.md`)

> Written at a context-window handoff. Week 10 is **substantially complete**: all 7
> methods built + committed + pushed, GPU results collected at 1B+8B/32K+64K, and a
> published visual report. This file is the state + the loose ends for the next
> session. Repo `/Users/hari/Desktop/kv-dlra`, branch `week7`, HEAD `005a653`
> (pushed to `origin/week7`). Suite **248 passed / 1 skipped**, ruff + mypy clean.

## What Week 10 delivered

**The definitive long-context comparison of 7 KV-cache compressors**, all on one
honest memory axis (float-equiv/layer; a CPU-offloaded float counts like a GPU one):
- ours: **BUG** (ranks 32/64/128/256), **MorphKV**
- eviction: **SnapKV**, **ExpectedAttention**
- the 3 requested low-rank baselines: **ThinK** (channel), **Palu** (low-rank
  projection — ported), **ShadowKV** (low-rank K + CPU-offloaded V — faithfully
  ported incl. the novel pre-attention query-selection hook)

Three axes: **perplexity** (supporting), **RULER retrieval** + **LongBench QA**
(primary, per the user's re-scope). User's directive: RULER + LongBench are the
PRIMARY eval, not ppl.

### The published report (the deliverable)
- **Artifact URL:** https://claude.ai/code/artifact/204eb116-c821-4af2-91ba-d3e6dd884fce
- Source: `docs/week10_report/index.html` (and scratchpad `w10_report.html`).
- **To UPDATE it from a fresh session:** re-publish the HTML via the Artifact tool
  **passing `url: "https://claude.ai/code/artifact/204eb116-c821-4af2-91ba-d3e6dd884fce"`**
  (a session that didn't publish it otherwise mints a NEW url). Data is inlined as a
  JS literal (`const DATA = {...}`); regenerate it from `results/w10-gpu-parsed.json`.

## The honest findings (report these straight — nuanced, not a BUG-wins story)

**Full cache sizes (the `1.0×` reference):** 1B = 512 feat/layer×16 layers → **1.0
GiB @32K, 2.0 GiB @64K** (32 KB/token). 8B = 1024 feat/layer×32 layers → **4.0 GiB
@32K, 8.0 GiB @64K** (128 KB/token). Batch 1. Both GQA (8 KV heads).

1. **Perplexity (COMPLETE: 1B+8B × 32K+64K).** Eviction (MorphKV/SnapKV) is
   near-lossless in the mid band (8B/32K morph-k0.25 **7.54** vs full 7.62); **BUG
   uniquely reaches extreme compression** — bug-r32 @ **0.033×** (135 MB vs 4 GiB) —
   but pays a ppl premium there (9.31). Palu (per-head) is weak (13.9–79 ppl);
   ThinK only lives in the high-memory band (compresses keys only, ratio 1−cr/2);
   ShadowKV can't beat **0.5×** (keeps all V on CPU — counted honestly).
   → matches the project's honest Week-4 finding (eviction leads on ppl).
2. **RULER needle retrieval.** At **32K, BUG = 0% at EVERY rank** (the rank-vs-
   context fidelity wall) while **ExpectedAttention = 100% at 0.10×** (8B). The
   wall is crossed **between 16K and 32K**: at 16K, **bug-r128 = 0.50** (partial
   recovery). So BUG's Week-9 recall win holds at moderate context but NOT at 32K.
3. **Who wins where:** extreme compression → BUG (only option); moderate/long-ctx
   ppl → eviction; retrieval@32K → ExpectedAttention. **No single winner.**

## Data + code map (Week 10)
- **Parsed GPU results:** `results/w10-gpu-parsed.json` (ppl 76 pts = 1B+8B×32K+64K
  ×~19 arms; ruler 35 pts incl. 32K niah_single + partial 16K). Raw pod logs +
  accumulated result lines in `results/gpu_logs/` (gitignored).
- **Parser:** `scripts/w10_parse_logs.py` (harvests per-arm lines from raw logs;
  NOTE it tags by filename, so `8b_supp.raw.log` → model `8b_supp` — merge those).
- **Harnesses:** `scripts/w10_frontier.py` (ppl + `build_arms`/`_footprint` shared
  by all), `w10_ruler.py` (RULER subset), `w10_longbench.py` (QA F1). All 7 methods
  wired; `--methods`/`--ranks`/`--morph-keeps`/`--evict-keeps`/`--think-ratios`/
  `--palu-ranks`/`--shadow-ranks`. Per-arm try/except (one bad arm won't kill a run).
- **Accounting:** `src/kvdlra/accounting.py` (bug/morph/evict/think/palu/shadow/full
  footprints + anti-drift pins). **Caches:** gated `score`/`ingest` modes on
  BugStreamingLayer/MorphKVLayer (`bug_cache.py`/`morph_cache.py`); `shadow_cache.py`.
- **Presses:** `palu_press.py`. **Pod:** `scripts/pod/w10_gpu.sh` (RUN_PPL/RUN_RULER/
  RUN_LB + RULER_TASKS/LB_TASKS/SEEDS knobs), `scripts/pod/scrape_w10.sh`.
- Plan: `docs/week10-plan.md`. Kickoff: `docs/week10-kickoff.md`.

## LOOSE ENDS for the next session (priority order)
1. **Complete the primary axes (credit now ample ~$26):** re-run the **16K RULER (4
   tasks) + LongBench QA** at 1B+8B — the main GPU run got 32K ppl+RULER but the
   generation axes at 16K only partially landed (8B niah_single only; 1B pod
   stalled; LongBench never ran). ShadowKV GPU also pending (device bug fixed in
   `0bf3b7b`, but shadow arms errored in the main run — the fix wasn't in the pod's
   cloned code). Use `scripts/pod/w10_gpu.sh` with `RUN_PPL=0`; **harvest robustly**
   (accumulate per-arm lines each poll — the base64 blocks scrape unreliably;
   `vastai logs` S3 fetch is flaky). Then update the report (pass `url`, above).
2. **Write the docs (Phase 7):** `docs/week10.md` (technical) + `docs/week10-
   explained.md` (plain-English, like `week9-explained.md`) + the memory-ratio
   table. The HTML report has the narrative; port it. Regenerate the matplotlib
   figures or point to the report.
3. **Delegator consolidation (Phase 7, deferred):** fold the 6 duplicated accounting
   formulas (kv_memory_ratio/evict_quant_memory/coord_for_config/…) into
   `accounting.py` delegators — REGRESSION-PIN the digit-exact current outputs first
   (the `evict_quant_memory` +FP16 fix would shift Week-4/5 figures ~0.2%).
4. **Update `handover.md`** (the top-level state file) + auto-memory
   ([[kvdlra-week10-standing]]) with the final numbers.
5. **Merge to `main`** — user's call (nothing merged yet; all on `week7`).

## Infra notes
- **Credit ~$26** (topped up; `credit` field lagged ~hours before posting — read
  `vastai show user --raw`). **Keys still unrotated** (vast.ai + HF — user action).
- **Pod recipe that WORKS:** git-clone `origin/week7` → pip install pinned
  (kvpress==0.5.1, transformers==5.8.0, datasets==2.21.0) → **`HF_HUB_DISABLE_XET=1`
  + hf_xet** (the Xet CAS returns 401 on pods → model/dataset download fails without
  this) → run → **accumulate per-arm log lines** (don't rely on base64 blocks or a
  single final fetch). SSH unusable. 32K generation is SLOW (~5 hrs/pod full scope)
  → prefer 16K + scoped tasks. Verify GPU is Ampere+ (cc≥8.0). **Always DESTROY pods
  after** (`printf 'y\n' | uvx vastai destroy instance <id>`). No pods currently
  running (all destroyed at handoff).
