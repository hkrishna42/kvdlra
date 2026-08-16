# Week-17 kickoff prompt (multi-agent)

*Paste the block below into a fresh Claude Code session in this repo to launch the next phase. It is
self-contained; it points the session at the Week-16 artifacts and asks it to run a multi-agent workflow.*

---

> **Week-17: turn `r64 + warmup-seed` into BUG's validated, improved cross-model config. Use a multi-agent
> workflow (ultracode) for the parallel investigation/design/code, then orchestrate the GPU confirmation
> yourself.**
>
> **Start by reading** `docs/week16-handover.md`, `docs/week16-explained.md`, and the `kvdlra-week16-plan`
> memory. One-paragraph summary so you have it: Week-16 tested generality on Mistral-7B-v0.3 and
> Qwen2.5-7B. The Llama-tuned **r128** config does **not** transfer, but BUG's **extreme-compression niche
> (rank 16–64, 0.04–0.15× memory)** does: near-perfect RULER retrieval + healthy perplexity on both, **sweet
> spot r64**. Two model-dependent r128 failure modes — Qwen *absorbs the needle* (ppl fine, retrieval 0, the
> rank-vs-retrieval wall at r128 vs Llama's r256; `-s32` doesn't fix it), Mistral's *streaming integrator
> diverges* (ppl 43.9). BUG doesn't beat baselines on ppl; its edge is **memory at matched retrieval (3–8×
> less than ThinK/Palu)**, in a regime eviction can't enter. Mistral **variable-tracking** is the one weak
> axis (~0.50 even at low rank). Dashboard artifact `757d6777`.
>
> **Goal.** Make `bugSseed-r64-h256` (r64 + warmup-seed, exact tier 256) the *validated, improved* config:
> (a) firm it with error bars across all three families at 16K **and** 32K; (b) fix the Mistral-vt limit;
> (c) try to raise the rank threshold by stabilizing the integrator; (d) close the rigor loose ends.
>
> **Run a workflow** (`ultracode`) that fans out these workstreams. Most are $0 CPU investigation/design and
> code — parallelize them; the GPU confirmation is the one sequential step you orchestrate after the designs
> land. Suggested phases:
>
> **Phase 1 — parallel $0 investigation + design (fan out ~4–5 agents, each returns a structured finding):**
> 1. **Mistral var-track fix.** Why does vt cap at ~0.50 even at r16–r64 while single/mv are 1.00? Read the
>    vt task builder (`scripts/w10_ruler.py:build_task`, the `V0=…;V1=V0;…` chain) and the surprise/exact-tier
>    selection (`src/kvdlra/cache/bug_cache.py`, `_surprise_scores` ~:906, the hh tier, `hh_neighbor` span
>    expansion). Hypotheses to weigh: the *chain* needs several linked tokens retained together (raise
>    `hh_budget` or `hh_neighbor`?); the root value is early (a warm-up-window residual?). Propose a concrete,
>    default-off, $0-CPU-probe-gated fix.
> 2. **Integrator high-rank stability.** Why does the streaming low-rank integrator diverge at high rank on
>    Mistral (r128) / Qwen (r256)? Read `src/kvdlra/integrators/streaming_torch.py` (`augmented_bug_step`,
>    `blocked_bug_subspace`) and the fp32-core/bf16-storage split. Propose stabilizations (re-orthogonalization
>    frequency, conditioning/rank-truncation guard, a divergence tripwire) that could extend the safe rank —
>    each with a $0 CPU test on the tiny Qwen2/Mistral fixtures in `tests/test_bug_cache_families.py`.
> 3. **The `h1024`-vs-`h256` r128 puzzle.** On Qwen, `bugSseed-r128-h1024` ppl = 467 but `-h256` ppl = 7.81.
>    A *larger* verbatim exact tier should only help fluency. Find why the big tier destabilizes at r128
>    (surprise-scoring against a larger basis? the warm-up seed? position handling of the hh tokens?). Write a
>    strict-xfail characterization test if it's a real bug.
> 4. **Rigor / decision-table merge.** Design the merge of the Week-16 Tier-1/2 + sweep line-files into
>    `results/w11-decision-table.json` via `scripts/w11_merge.py` (respect the one-n-per-(file,ctx) rule and
>    the `seed*131+trial` pooling discipline — see `docs/week16-handover.md`), then Wilson CIs via
>    `scripts/w15_intervals.py`. Note: raw pod logs are gone; the Week-16 numbers are in
>    `docs/week16-handover.md` §2 and the scratchpad `w16-*-sweep-raw.log` of the *previous* session (may be
>    unavailable — re-derive from the doc, or re-measure).
> 5. *(optional)* **Tier-3 breadth.** Scope LongBench (all `QA_TASKS` in `scripts/w10_longbench.py`) +
>    ∞Bench at `bugSseed-r64-h256`.
>
> **Phase 2 — synthesize** the findings into one concrete plan: the exact GPU confirmation matrix +
> the code fixes to implement. Then **implement the $0 fixes** (green-gate each: `uv run pytest -q && ruff
> check . && mypy src tests scripts`; commit per increment; push `origin/week7`).
>
> **Phase 3 — GPU confirmation (you orchestrate; do NOT let sub-agents launch pods).** Pre-registered matrix:
> `bugSseed-r64-h256` **±any Phase-2 fix** across **Llama-3.1-8B / Qwen2.5-7B / Mistral-7B-v0.3**, RULER
> {single, mv, vt} at **16K (n≥8)** and **32K (n=4)**, plus a matched ppl block; baselines `full`, `think-c0.5`,
> `palu-r0.5`. **Fund bars:** retrieval single+mv ≥ Wilson-lower 0.7 on all three at r64; a Mistral-vt fix is
> funded iff vt rises to Wilson-separated > 0.50; a stability fix is funded iff it raises the safe rank
> (r128 ppl within 2× full) without regressing r64. Use the proven pod driver `scripts/pod/w16.sh`
> (`MODE=sweep`, or add a focused `MODE`); one pod per model, staged cheap-first; **harvest from a live
> monitor** (`vastai logs` truncates); **destroy every pod when done.**
>
> **Facts & gotchas.** Both 7B families are **ungated** (canonical HF ids, no mirror). `w10_ruler.py`
> **must** get `--chunk` (the `RULER()` helper in `w16.sh` does — a dropped `--chunk` voided a whole pod
> round in Week-16). A100 ≈ $0.57–0.75/hr; credit ~**$97**; escalate if < $3; keys unrotated (flag only).
> Memory ratios are honest `ratio_fp16`. Update the dashboard artifact `757d6777` (pass its `url`) and write
> `docs/week17-*.md`.
>
> **Deliverable:** the consolidated r64-seed cross-model result (all three, with Wilson CIs, 16K+32K), any
> funded fix for Mistral-vt / the rank threshold, and an updated dashboard + writeup.

---

## Notes for the human (not part of the paste)

- The workflow's sub-agents are for **reading, diagnosing, designing, and writing code/tests** — the parts
  that parallelize. **GPU pods stay with the orchestrator** (stateful vast.ai creds, sequential, real money).
- If you don't want the full multi-agent run, the single highest-value next step is just **Phase 3** for
  `r64-seed` across the three models with Wilson intervals — that alone converts the Week-16 point estimates
  into a defensible generality claim.
- Budget: a 3-model r64 confirm at 16K(n≥8)+32K(n=4) is ~$10–18 on three parallel pods.
