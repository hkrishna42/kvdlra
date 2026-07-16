# Week 11 — session handover (decision-table run + 2 pending deliverables)

> **STATUS 2026-07-16: the decision-table run COMPLETED and both deliverables shipped.**
> Authoritative result: `docs/week11-decision-table.md` (the unified 16K+32K table +
> recommendation). Key correction since this doc was written: the "gist is dead weight"
> line below is **superseded** — the full 4-task suite shows the low-rank gist *helps*
> on hard multi-fact retrieval (a lean; keep `bugS`, drop `bugEVICT`, retire plain BUG).
> Explainer artifact `c776074d`. The rest of this doc is kept as the historical run recipe.

> Written mid-session so a fresh session can finish cleanly if this one runs out
> of context. Repo `/Users/hari/Desktop/kv-dlra`, branch `week7`, **HEAD `113f31e`
> (pushed to `origin/week7`)**. Suite **263 passed / 1 skipped**, ruff + mypy clean.
> **Read this first, then `docs/week11.md`** (the technical writeup) and auto-memory
> [[kvdlra-week11-standing]].

## TL;DR — where we are

Week-11 **core research is DONE, committed, and pushed**. What remains is **one
GPU run in flight** (a "decision table") and **two deliverables** built from it:
1. assemble the single **unified decision table**, 2. build a **beginner-friendly
explainer dashboard**. Credit is ample (**~$34**, user topped up — do NOT cut on
credit; the user said run to completion). vast.ai keys work; no rotation needed.

## The result already banked (the honest decision)

Goal was: make BUG **retrieve** a 32K needle at ≤ ExpectedAttention's memory, or
prove it can't. **Answer: a WIN with a self-critical attribution.**
- **Mechanism SurpriseSLASH:** BUG's exact heavy-hitter tier (`hh_budget`) selects
  by low-rank **surprise** (out-of-subspace residual) not attention — a needle is a
  low-attention high-residual outlier. `hh_select="surprise"` + `hh_neighbor` span
  boost on `BugStreamingCache`. Control **`bugEVICT`** = rank-1 degenerate BUG
  (≈ pure surprise-eviction, no gist).
- **8B/32K:** `bugEVICT-h256` = **100% retrieval @ 0.009×** (~11× cheaper than
  `ea-k0.1`'s 0.10×), where **plain BUG = 0%**. Wall beaten.
- **Honest catch:** the win is BUG's surprise *signal used as eviction*, **not** its
  low-rank gist. `bugEVICT` (no gist) ties/beats `bugslash` (rank-32 gist) on BOTH
  retrieval-memory AND perplexity (ppl 8.95 vs 9.16 @h256; n=2, small margins) → the
  **gist is dead weight**, extending the Week-7/8 overhead-floor wall. vs EA it's a
  memory/quality trade, not a sweep.
- **Goal A (finish Week-10):** fixed 3 harness bugs (MorphKV short-final-chunk /
  SnapKV single-shot presses / LongBench per-task try/except; + ShadowKV device
  `0bf3b7b`) → full **4-task RULER @16K/8B** + LongBench `qasper` now run all methods.

## THE IN-FLIGHT POD — resume this FIRST

- **Contract `45023383`** (vast.ai A100, label `kvdlra-w11-table`), running
  `scripts/pod/w11_table.sh`. Purpose: fill the **decision table** —
  BUG / bugS / bugEVICT / ea / full × **4 RULER tasks + perplexity** × **16K & 32K**.
- **Order (cheap→expensive):** `R16` (16K RULER, new arms only) → `PPL` (16K+32K) →
  `R32` (32K 4-task). Watch for `===R16_DONE=== ===PPL_DONE=== ===R32_DONE===
  ===ALL_DONE===`.
- **Monitor:** `uvx vastai logs 45023383 --tail 200000` (ignore `port forwarding`
  SSH noise — that is NOT an error). Accumulate each poll:
  `uvx vastai logs 45023383 --tail 200000 >> results/gpu_logs/w11_table.acc.log`.
- **Result lines:** `^\[niah…`, `^\[vt…` (RULER acc); ppl `  <method>  [T=<t>] ppl=…
  ratio=…`.
- **Harvest on `ALL_DONE`** (decode the base64 blocks):
  ```bash
  for m in R16 PPL R32; do
    awk "/===${m}_RESULT_BEGIN===/{f=1;next} /===${m}_RESULT_END===/{f=0} f" \
      results/gpu_logs/w11_table.acc.log | tr -d ' \r\n' | base64 -d \
      > results/w11-table-$(echo $m|tr A-Z a-z).json
  done
  ```
- **DESTROY after:** `printf 'y\n' | uvx vastai destroy instance 45023383`.
- **If the pod died** (host flakiness — we hit GPU-error host / clone early-EOF / HF
  timeout, all patched): relaunch —
  ```bash
  OFFER=$(uvx vastai search offers 'num_gpus=1 gpu_name in [A100_PCIE,A100_SXM4] dph<0.8 reliability>0.99 rentable=true cuda_vers>=12.4 disk_space>=60 inet_down>=800' -o dph --raw | python3 -c "import sys,json;print(json.load(sys.stdin)[0]['id'])")
  uvx vastai create instance $OFFER --image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime --disk 80 --onstart scripts/pod/w11_table.sh --label kvdlra-w11-table
  ```
  Validate deps: logs should show `===DEPS_DONE===`, `torch 2.11.0+cu128 cuda True`,
  `===MODEL_OK===` before trusting it.

## Deliverable 1 — the unified decision table

One table to decide **continue with BUG vs bugS vs bugEVICT**. Rows: plain BUG
(r32, r128), **bugS-r32** (h256, h1024), **bugEVICT** (h256, h1024), `ea-k0.1`,
`full`, + reference baselines (`morph-k0.25`, `snapkv-k0.1`, `think-c0.3`,
`palu-r0.5`, `shadow-r64`). Columns: **Memory · Needle · Multi-key · Multi-value ·
Var-track · Perplexity**. Two blocks: **16K** and **32K**.
Data sources to merge:
- NEW (pod): `results/w11-table-{r16,ppl,r32}.json`.
- Existing on disk: `results/w11-goalA-ruler-lines.txt` (16K 4-task, 7 baselines),
  `results/w11-goalB-{ruler,ppl}-lines.txt` (32K bugS/bugEVICT/ea/full + ppl),
  `results/w10-gpu-parsed.json` (8B 32K ppl + niah_single for the 7 baselines).
Commit the assembled table (as a `.md` or JSON under `results/` or `docs/`) + push.

## Deliverable 2 — beginner-friendly explainer dashboard

A **NEW** Artifact (call the `artifact-design` skill first) for a reader with only a
basic idea of a KV cache. Teach from basics → the finding → the table + a plain
recommendation. Arc: what a KV cache is + why it grows + why compress → the two
families (eviction vs low-rank/BUG) → the needle-retrieval problem → why BUG's blur
fails at 32K → the SurpriseSLASH idea (keep *surprising* tokens exact) → the honest
finding (100% @ 0.009×, ~11× < EA, **but** it's surprise-eviction not the gist —
`bugEVICT` dead-weight test) → **end with the full decision table + "continue with
BUG, bugS, or bugEVICT?" recommendation**. Editorial teaching treatment, light+dark
themes, cool-slate palette, mono for metrics. **NEW file path**
`scratchpad/kvdlra_explainer.html` — keep the existing dashboard (artifact
`e811be6a`) separate. Report the new artifact URL to the user.

## Already-published artifacts (update IN PLACE only if needed)
- **`e811be6a`** — the Week-11 dashboard (both goals). To update: republish
  `scratchpad/goalA_dashboard.html` same path (same URL) in the SAME session, OR
  pass `url:` that URL from a new session.
- **`204eb116`** — the Week-10 report (now has the Week-11 retrieval-flip banner).

## Code map (the Week-11 mechanism)
- `src/kvdlra/cache/bug_cache.py`: `hh_select` knob (`_absorb_block_slash` surprise
  branch), `_surprise_scores`, `_span_boost` (hh_neighbor); single-shot `_prefill`
  bypasses SLASH → arm MUST chunked-ingest.
- `src/kvdlra/accounting.py`: `bug_footprint(..., hh_select=)` (anti-drift pin).
- `scripts/w10_frontier.py`: `build_arms` (bugslash/bugevict arms), `_footprint`
  (retention thread); `--hh-budgets`/`--hh-neighbor` CLI.
- `scripts/w10_ruler.py` / `w10_longbench.py`: GOAL-A harness fixes.
- `scripts/w11_probe.py`: needle-surprise probe. `scripts/pod/w11_{gpu,table}.sh`.
- Tests: `tests/test_bug_cache_week11.py`, `tests/test_accounting.py`.
- Docs: `docs/week11.md`, `docs/week11-explained.md`. Data: `results/w11-*-lines.txt`.

## Guardrails / gotchas
- Suite must stay **263 pass / 1 skip**, ruff + mypy clean. Commit per verified step.
- Pre-commit `ruff-format` reformats then FAILS the commit → re-`git add -A` and
  re-commit (normal).
- `handover.md` (top-level) is **gitignored** — a committed handover goes in `docs/`.
- Bare `import kvdlra` is flaky on this Mac (editable install) — pytest is robust via
  `pythonpath=["src"]`; if imports fail, `uv pip install -e ".[dev]"` again.
- Credit ~$34; keys work (no rotation). vast host flakiness → relaunch on fresh host.
