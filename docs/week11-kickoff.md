# Week 11 — kickoff prompt (paste into a fresh session)

> **Paste-ready.** Everything below the line is the prompt. It (1) picks up the
> Week-10 handoff, (2) finishes the two Week-10 loose ends, and (3) opens a real
> research thrust — **brainstorm + iteratively improve BUG so it can retrieve at
> 32K**, run as a multi-agent, harness-engineered, adversarially-verified build.

---

ROLE. You are a Machine Learning Researcher (numerical-analysis / mathematical-
physics background) continuing the **kvdlra** project (the Ceruti–Lubich **BUG**
integrator as a streaming KV-cache compressor). NON-NEGOTIABLE ETHOS: report numbers
straight, **prefer honest negatives**, retract overclaims immediately, **count ALL
memory honestly in the same float-equivalent unit regardless of device**. Every
prior overclaim was caught and retracted — keep that bar. Ultracode is on: use the
**Workflow tool** for substantive orchestration (design panels, fan-out, adversarial
verify); token cost is not a constraint, correctness is.

PROJECT STATE. Repo `/Users/hari/Desktop/kv-dlra`, branch `week7` (Weeks 7–10
committed + pushed to `origin/week7`, HEAD `4502ad5`, **not merged to main**). Read,
in order:
  1. **`docs/week10-handover.md`** — the state + loose ends (READ FIRST).
  2. `handover.md` §status table, and auto-memory ([[kvdlra-week10-standing]],
     [[kvdlra-week9-standing]], [[kvdlra-dominance-program]], [[vastai-pod-flakiness-jul2026]]).
  3. `docs/week10-plan.md` (the build plan) + the published report
     `docs/week10_report/index.html` (the findings, in charts + plain English).
  4. `docs/week9.md` (D1 recovery-tier + D3 surprise — the closest prior art to the
     retrieval thrust) and `docs/week7-dominance.md` (the two measured walls).

WHERE WE LEFT OFF (the one-paragraph truth). All 7 methods are built + honest-memory-
accounted (BUG, MorphKV, SnapKV, ExpectedAttention, and the low-rank baselines
ThinK / Palu / ShadowKV — the last faithfully ported incl. a pre-attention query-
selection hook). The **perplexity** frontier is complete at 1B+8B × 32K+64K. The
headline is honest and nuanced — **no single winner**: eviction is near-lossless in
the mid memory band at long context; **BUG uniquely reaches extreme compression
(0.033×, ~135 MB vs a 4 GiB full cache at 8B/32K)**; and on **retrieval, BUG fails
at 32K (0% at EVERY rank) while ExpectedAttention hits 100% at 0.10×** — the rank-
vs-context fidelity wall, measured to cross **between 16K (bug-r128 = 0.50) and 32K
(0.00)**. Full report: https://claude.ai/code/artifact/204eb116-c821-4af2-91ba-d3e6dd884fce

## GOAL A (finish Week-10 first — bank it, ~1 pod-session)
Complete the two generation axes that only partially ran (32K generation was too
slow; a pod stalled; ShadowKV's GPU device-fix `0bf3b7b` wasn't in the run's code):
- **16K RULER (all 4 tasks) + LongBench QA at 1B+8B, including ShadowKV.**
- Update the published report **in place** (re-publish `docs/week10_report/index.html`
  via the Artifact tool **passing `url:` the artifact URL above** — a fresh session
  otherwise mints a new URL). Regenerate the inlined `DATA` from
  `results/w10-gpu-parsed.json` + the new runs (`scripts/w10_parse_logs.py`).
- Pod recipe that WORKS (see handover): git-clone `origin/week7` → pinned deps →
  **`HF_HUB_DISABLE_XET=1` + hf_xet** (Xet CAS 401s on pods) → run
  `scripts/pod/w10_gpu.sh RUN_PPL=0` → **ACCUMULATE per-arm log lines each poll**
  (base64 blocks + single `vastai logs` fetches are flaky — append+`sort -u`) →
  **DESTROY pods after**. Credit ~$26 (topped up); keys still unrotated (user action).

## GOAL B (the research thrust — the real work)
**Can we make BUG retrieve a needle at 32K, at less memory than ExpectedAttention
needs — or prove it can't?** This is falsifiable and may well be a bounded/negative
result; that is an acceptable, publishable outcome. Report it straight either way.

**Why BUG fails today (the diagnosis to confirm first).** A "needle" (a rare 5-digit
code, low attention mass, ~1/32000 of the sequence) is a *sharp, high-residual
outlier*. A rank-`r` summary captures the dominant `r` directions of the whole
context; the needle is not a dominant direction, so its exact value is smeared and
un-readable. Higher rank doesn't fix it (rank-256 still fails at 32K) — it's the
rank-vs-context wall, not a tuning knob. ExpectedAttention wins because it keeps
needle-relevant tokens *verbatim*.

**The falsifiable bar (the honesty crux).** ExpectedAttention gets **100% at 0.10×
memory** at 8B/32K. Any BUG variant only "wins retrieval" if it reaches comparable
accuracy at **≤ that memory** — otherwise it's "eviction with extra low-rank
overhead," which the Week-7/8 **overhead-floor wall** predicts is dominated (BUG's
`2nr` basis is dead weight eviction never pays). State the memory premium exactly.
The clean outcomes: (i) BUG+X retrieves at ≤0.10× → a genuine win; (ii) it needs a
premium → quantify it (like the Week-9 8B recall win at 1.8×); (iii) it can't → a
map-completing negative confirming the wall at extreme context.

**Brainstorm seeds (expand these in a design panel — don't just implement them).**
The most promising is a **hybrid: BUG's low-rank gist + a tiny EXACT outlier tier**,
because the needle is precisely what the low-rank basis fits worst. The machinery
already exists — reuse, don't rebuild:
  1. **Surprise/outlier-exact tier.** `retention="lowrank_surprise"` (Week-9 D3)
     already keeps high-residual columns; SLASH `hh_budget` keeps top tokens
     *verbatim* (post-RoPE). Combine: low-rank the redundant bulk, keep the top-k
     highest-residual tokens EXACT. Does a small exact tier catch the needle at 32K?
     (D3's one proven payoff was exactly outlier-retention-for-recall.)
  2. **Query-aware reconstruction on BUG (ShadowKV-style).** Add per-chunk landmarks
     to BUG's basis; at decode, reconstruct only query-selected chunks at higher
     effective fidelity. `shadow_cache.py`'s landmark + pre-attention-hook machinery
     is a template.
  3. **Two-tier recovery at scale (Week-9 D1).** BUG gist + a MorphKV-preserving
     per-head recovery tier for the dropped stream (the `HybridRecoveryCache` future
     work noted in `docs/week9.md` D1).
  4. **Rank-adaptive / residual-boosted** allocation of rank to the high-residual
     tail — but Week-7 flagged this as the bounded "rank-squeeze"; treat as a
     control, not a favourite.
Let the panel also propose mechanisms you haven't listed, and pick the one with the
best chance of clearing the 0.10× bar. Adversarially pre-mortem each: what makes it
collapse into "just eviction"?

## Multi-agent build (harness engineering — the project's process)
Run each phase as a Workflow; keep tests green (`uv run pytest -q && ruff check .
&& mypy src tests scripts`, currently 248/1) at every step; commit per verified
increment; 1B/CPU or a cheap 1B pod before any 8B pod.

- **Phase 0 — design panel (parallel agents, 1 per sub-problem):** (a) confirm the
  diagnosis (is the needle a clean high-residual outlier at 32K? probe it); (b)
  design the top hybrid mechanism (exact signatures, reusing surprise/SLASH/shadow
  machinery, honest accounting for the exact tier); (c) design the retrieval eval
  harness (extend `scripts/w10_ruler.py`; matched-memory audit vs the EA bar) + the
  ShadowKV/16K completion. A **judge** sequences the build, sets a time-box, and
  writes `docs/week11-plan.md`. Escalate before sinking >½ day into any one mechanism.
- **Phase 1 — implement** the chosen mechanism (new cache mode / press), tests green,
  anti-drift memory pin, lossless-oracle + a synthetic needle unit test at each step.
- **Phase 2 — iterate on 1B** (cheap): sweep the exact-tier budget / rank at ctx
  4K→16K→32K; find where retrieval re-emerges and at what memory. Then **confirm the
  ranking at 8B** on a pod. Log peak GPU; matched-memory audit vs EA's 0.10×.
- **Phase 3 — adversarial verify (parallel skeptics, ≥2/3 must fail to refute):** is
  the exact tier's memory counted honestly? is this beating EA at matched memory or
  just re-deriving eviction? does it hold across needle depths / multi-key / 8B?
- **Phase 4 — write up** `docs/week11.md` + update the report (pass the artifact
  `url`) + `docs/week11-explained.md`; update `handover.md` + auto-memory; commit +
  push `origin/week7`.

## Honesty guardrails + escalation
- Every arm's memory pinned to its live `stored_state_numel` (anti-drift tests);
  the exact-outlier tier is real memory — count it, never hide it.
- Report the retrieval bar honestly: if BUG+X can't beat ExpectedAttention at matched
  memory at 32K, **say so** — a clean bounded negative confirming the fidelity wall
  is a real result (like Weeks 7/8).
- ESCALATE (don't silently retry): a mechanism blows its time-box; credit < $3;
  32K/64K OOMs on the card; the vast.ai/HF keys need rotating before a pod; or a
  result contradicts the Week-10 walls (re-check the harness before believing it).

## First actions
```bash
cd /Users/hari/Desktop/kv-dlra && git checkout week7 && git pull
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest -q            # expect 248 passed, 1 skipped
uv run ruff check . && uv run mypy src tests scripts   # both green
```
Then: read `docs/week10-handover.md`; `TodoWrite` the plan; launch the **Phase-0
design panel** (diagnosis + top-hybrid design + eval/completion harness → judge).
Every agent inherits the ethos + guardrails above.
