# Week-18 kickoff prompt (multi-agent): paper program — verdict 5/10 → strong-accept

*Paste the block below into a fresh Claude Code session in this repo. It is self-contained. It encodes
the 2026-09-01 NeurIPS-readiness panel verdict (6 dimension reviewers + adversarial cross-exam + AC
meta-review; full reviews were in that session's scratchpad `review-*.md`, verdict summarized in
this doc's §Verdict) and converts the AC's ranked gap-fill plan into an executable multi-agent program.*

**Honesty bar:** no process guarantees a "10". The exit bar is: *all fatal/major panel findings
resolved; an internal adversarial re-review panel scores ≥7 on every dimension with zero fatal
findings; arXiv preprint live; ICML-2027-ready draft.* That is strong-accept territory (7–8).

---

> **Week-18/19: execute the paper program for kvdlra. Use multi-agent workflows (ultracode) for the
> parallel harness/test/writing work; orchestrate all GPU pods yourself (never sub-agents). Target:
> ICML 2027 (~Jan deadline); arXiv preprint by ~Sept 20 to timestamp against concurrent
> MomentKV (arXiv 2606.01563) / ResKV (arXiv 2607.29591).**
>
> **Start by reading** `docs/week18-kickoff.md` §Verdict (below the paste block), `docs/week17-handover.md`,
> `docs/week17-explained.md`, and the `kvdlra-week17-standing` memory. The panel verdict in one line:
> **5/10 borderline-reject as worded — the science survives cross-examination, but two baseline families
> are missing (2-bit KV quantization; eviction in the headline grids), the memory framing conflates
> storage with residency, the benchmark is self-authored, and the manuscript doesn't exist.** Every gap
> is enumerated and priced. The core mechanism (online rank-adaptive DLRA tracking + surprise-selected
> exact tier), the rank-siphoning diagnosis, the r/n≈0.25 wall, and the marquee vt separation
> (p≈0.0006, n=16) are verified-novel and survive.
>
> ## Phase 0 — $0 wording + chain-of-evidence pass (do FIRST, sequential, ~half a day)
> Green-gate each increment (`uv run pytest -q && ruff check . && mypy src tests scripts`), commit per
> increment, push `origin/week7`.
> 1. **Wording pass** over `docs/week16-explained.md`, `docs/week16-handover.md`,
>    `docs/week17-explained.md`, `docs/week17-handover.md`, README:
>    "5–13× less memory" → **"3.4–10× less float-equivalent stored state than ThinK/Palu (6.7–13× vs
>    full KV)"**; every non-separated "beats" → "leads" (Qwen mv 12/12 vs 10/12 is Fisher p≈0.48);
>    name the marquee comparator (**think-c0.5**); state C4's cross-week provenance (retrieval=W17 pods,
>    ppl ties=W15 pods); scope "a regime eviction/channel-pruning cannot enter" → **"channel-pruning/
>    low-rank-factorization structurally floored at 0.50–0.75×; eviction reaches 0.1× but loses
>    var-track (0.17) and multikey decays with length (92→67→50) — Week-11 measured"**; scope
>    "extends the safe rank" as within-method (floored r256 is Palu-dominated at matched memory);
>    16K matrix language = **non-inferiority** ("no detected loss", 12/12 ⇒ acc≥0.76) until
>    discriminative reruns land.
> 2. **Dual memory billing**: `ratio_fp16` bills fp32-at-rest gist state (u_k/c_k) at 16 bits.
>    Add an honest `ratio_stored_bits` (actual at-rest bits) alongside — report BOTH everywhere
>    (r64 ≈0.15×/0.27× honest vs 0.085×/0.149× fp16-equiv). Test-pin both formulas.
> 3. **Evidentiary chain (free forever after)**: pod driver clones by **SHA not branch**; prepend
>    `git SHA + nvidia-smi + python/torch/CUDA versions` INSIDE the harvested log block; add
>    per-trial hit emission to `w10_ruler.py` (one `[trial]`-prefixed line per trial: task, ctx, arm,
>    seed, trial, hit, frac) + base64-fold the out-JSONs through the log (`scripts/pod/scrape_w10.sh`
>    pattern). Also: recover per-trial marquee hits from the preserved Week-17 pod logs if the old
>    session scratchpad still exists (episodic-memory search for its path) — else regenerate in Phase 3.
> 4. **Hygiene**: model license/cards section (Llama 3.1 Community License; justify-or-replace the
>    `unsloth/Meta-Llama-3.1-8B-Instruct` mirror); add `week7` to CI push triggers; commit-or-delete
>    `results/w15b-complete-lines.txt` + stale `figures/week10/*`; fix README test count.
>
> ## Phase 1 — parallel $0 harness engineering + tests + writing (ultracode workflow, ~8 agents)
> Every code workstream: **default-off flag, bit-identical-off regression test, fail-loud tripwire,
> CPU probe proving the path end-to-end on `unsloth/Llama-3.2-1B-Instruct` before any GPU.** Each
> agent returns a structured finding + probe output; sub-agents NEVER launch pods or edit outside
> their workstream. Suggested workstreams:
> 1. **H1 realistic filler + depth grid** (`w10_ruler.py`): `--filler {cycle,wikitext,pg19}` (default
>    `cycle` = bit-identical archived path; corpus sentences via `perplexity_sweep` loaders, seeded
>    shuffle per trial) + `--depths 0.1..0.9` grid reusing the depth-parameterized `w4_needle`
>    machinery. Tests: default bit-identical; wikitext filler tokenizes to ≥ctx; needle survives
>    template tripwire at every depth.
> 2. **H2 quantized-KV baseline arm**: 2-bit KV via transformers' `QuantizedCache` (quanto/HQQ
>    backend, KIVI-style residual window) as arm `quant-2bit` (+`quant-4bit`), honest accounting
>    (quantized bits + scales/zeros + fp16 residual window counted), parity smoke vs DynamicCache at
>    8-bit, CPU probe. This is the panel's #1 blocking gap.
> 3. **H3 BUG×quant compose + fp16-state probe**: (a) revive the coordinate-quant path
>    (`quant_bits=4`) on the flagship → `bugSseed-r64-h256-q4` ≈0.04–0.05× — the honestly-exclusive
>    sub-cliff arm; (b) `--state-dtype fp16` probe that casts U/C between integrator steps (decides
>    the dual-billing question empirically). Tests: default-off bit-identity; q4 accounting pinned.
> 4. **H4 eviction-in-grid**: wire `ea-k0.1` + `snapkv-k0.1` + `bugEVICT` into the w18 cross-model
>    pod MODE (arms already exist in `build_arms`); include **multikey** everywhere (its absence
>    reads as selective reporting).
> 5. **H5 persistence/cold-load bench** (the systems reviewer's M1/M2): serialize flagship cache
>    @32K → bytes on disk, reload + H2D transfer + reconstruct-to-attend-ready timing vs full-KV
>    load (~4.29GB vs ~0.7GB @8B-32K); rerun `w16_storage.py` on CUDA committing JSON+figure.
>    Report decode-workspace ≈0.98× **in the same table** (no burying); fused kernel stays future work.
> 6. **H6 official-benchmark anchor**: official RULER (NVIDIA repo) niah subset runner for the
>    flagship on one model, OR the LongBench flagship thread-through (4-arg argparse addition scoped
>    in W17 WS5) — prefer BOTH if cheap. Publish the generator-vs-official template diff.
> 7. **H7 stats upgrades**: per-trial McNemar/paired-permutation for pre-registered contrasts
>    (consumes H2's per-trial lines); Wilson stays for cells; multiple-comparisons statement.
> 8. **H8 paper**: LaTeX skeleton in `paper/` (contributions, method with CKL θ lineage —
>    `min_sv_frac` framed as self-scaling relative variant + the rank-siphoning diagnosis, NOT as a
>    new invention), **related work ~20–25 citations** (LESS/LoLA differentiation paragraph;
>    KIVI/KVQuant/CacheGen; DLRT NeurIPS-22; H2O/SnapKV/ScissorHands/PyramidKV/Quest; Palu/Eigen-
>    Attention/xKV/MatryoshkaKV; MLA; EA-name-collision note; MomentKV/ResKV as concurrent),
>    limitations section leading with workspace/kernel + vt-on-1024-dim. A **citation-verifier agent**
>    WebSearch-checks every bibitem (title/venue/year/arXiv id) — no hallucinated refs.
> Synthesis agent: merge findings → pre-registered `MODE=w18` matrix + fund bars (below) → implement
> order. Then implement (TDD; green-gate; commit per increment; push before pods).
>
> ## Phase 2 — pre-registered GPU matrix (YOU orchestrate; pods clone by SHA)
> One A100 pod per model (`unsloth/Meta-Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`,
> `mistralai/Mistral-7B-Instruct-v0.3`), staged cheap-first, RULER before ppl, live Monitor per pod
> (stage markers + result rows + failure signatures: Traceback|OOM|illegal memory|SKIP), harvest
> full log BEFORE destroy (`vastai logs --tail 20000` + the base64 JSON folds), destroy every pod;
> on a mid-run CUDA fault: harvest completed blocks immediately, re-run only the dead block on a
> fresh card (the W17 `w17ppl` lesson). Matrix (≈$130–185 total; **credit $70.4 → flag the user for
> a ~$50–100 top-up BEFORE launching; escalate below $3**):
> - **G1 (quant, blocking)**: `quant-2bit`/`quant-4bit` + flagship + `bugSseed-r64-h256-q4`,
>   16K n=12 + 32K n=12, RULER{single,mk,mv,vt} + ppl, 3 families. (~$60)
> - **G2 (realistic filler, blocking)**: H1 filler+depth grid: flagship + think/palu + ea-k0.1,
>   16K n=12, Llama+Qwen. (~$15) · Official-RULER/LongBench anchor, one model. (~$50)
> - **G3 (eviction grid)**: ea-k0.1 + snapkv-k0.1 (+bugEVICT-h256) all 3 families, mk included,
>   16K n=12 / 32K n=12. (~$25)
> - **G4 (firming, rides along)**: 32K n=4→12 everywhere; mk on Qwen/Mistral; marquee single/mk
>   n=16 + think-c0.3/palu-r0.25 at the marquee cell; seeded Llama r256 (r/n law); fp16-state probe;
>   marquee ppl same-pod (kills the cross-week splice). (~$30–40)
> - **G5 (persistence)**: H5 measurements on one pod. (~$10)
> **Fund bars (pre-registered, decide wording not spin):** C1 survives iff flagship single+mv hold
> Wilson-lo ≥0.7 under realistic filler; the exclusive-band claim moves to the sub-cliff arm iff
> `q4`-compose holds retrieval at ≤0.05×; quantization results are reported WHATEVER they show
> (compose-not-compete framing); every "beats" requires McNemar p<0.05 on per-trial data.
> - Write per-model MODEs into `scripts/pod/w16.sh` successor (`w18.sh`), `bash -n` + CPU smoke of
>   every new flag through the REAL harness before any pod. **Never drop `--chunk`.**
>
> ## Phase 3 — analysis, paper, dashboard, gate
> 1. Marker-aware intervals builder for w18 (extend the `w17_intervals.py` pattern; per-trial files
>    committed; env headers verified present). Regenerate ALL paper tables from committed line-files.
> 2. Update `docs/` + the results dashboard artifact `757d6777` (pass `url:`) with the new grids.
> 3. Finish the draft; run figures through the dataviz/graphing skill; prose through
>    elements-of-style; `/code-review` on all harness diffs.
> 4. **Exit gate: re-run the adversarial reviewer panel** (6 dimensions + attack/defense cross-exam +
>    AC, same protocol as 2026-09-01) against the DRAFT + new results. Bar: ≥7 every dimension, zero
>    fatal. Iterate until passed. Then arXiv preprint (user approves submission), ICML draft frozen.
>
> ## Skills / plugins / agents to leverage (available in this environment)
> - **Workflow (ultracode)** for every parallel phase; `Monitor` for pods; `TaskCreate/Update` for
>   phase tracking; worktree isolation for parallel code agents touching the same files.
> - **superpowers**: `writing-plans` (before Phase 1 implementation), `test-driven-development`
>   (every harness change), `verification-before-completion` + `requesting-code-review` (before every
>   commit claim), `systematic-debugging` (any pod failure), `dispatching-parallel-agents`.
> - **feature-dev agents**: `code-explorer` (trace harness paths before editing), `code-architect`
>   (H2 quant-arm design), `code-reviewer` (post-implementation).
> - **episodic-memory** search: recover prior-session scratchpad paths (preserved W16/W17 raw pod
>   logs; per-trial recovery) before re-buying data.
> - **elements-of-style:writing-clearly-and-concisely** (paper prose), **claude-tag-data-viz:graphing**
>   + **dataviz** (paper figures), **artifact-design** (dashboard), **WebSearch/WebFetch** (citation
>   verification; check MomentKV/ResKV status), **code-review** skill (harness diffs).
> - GitHub plugin is unauthenticated — use the `gh` CLI (already authed) for anything GitHub.
>
> ## Facts & gotchas (carried)
> - vast.ai: read `credit` from `vastai show user --raw` (Balance column lies); A100 ≈$0.60–0.80/hr;
>   create-retry across offers (some machines demand identity verification); one pod per model;
>   onstart ≤16KB; `vastai logs` truncates — harvest via monitor + `--tail 20000`; DESTROY every pod;
>   keys unrotated (flag only). Streaming ppl arms ≈30min each at n-samples 8 — size cells first.
> - Both 7B families ungated; Llama via the unsloth mirror (license statement now required — Phase 0).
> - Bit-identical-off is the house rule for every new flag; archived results must reproduce.
> - The panel's full reviews: 8 files `review-*.md` + `review-meta-verdict.md` in the 2026-09-01
>   session scratchpad (recover via episodic-memory if needed; verdict summarized below).

---

## §Verdict (2026-09-01 panel — condensed record for the next session)

**5/10 borderline reject → close-needs-targeted-work; post-fill trajectory 6–7+.** Scores: claims 5,
prior-work 5, rigor 5, significance 5, systems 3, repro 7 (attack 3 / defense 6; AC spot-verified).

**Survives:** DLRA-tracker + surprise-tier novelty (attacker conceded); gist/tier decomposition +
rank-siphoning diagnosis; r/n≈0.25 wall (W11 r256 control supports); marquee vt vs think-c0.5
p≈0.0006 n=16; floor rescues (27531→6.995 etc.); warmup-seed fix; bugS-r32 @0.043× = 100/67/100/100
@32K (sub-cliff seed); W11 realistic-text ordering (bug-r128 0.221 F1 @0.16× beats all sub-0.25×
eviction); the honesty/pre-registration infrastructure.

**Fatal set (all priced, none contradict measured results):** (1) no quantization baseline while
KIVI-2bit/KVQuant occupy 0.125–0.19×; (2) storage-vs-resident conflation (workspace ≈0.98×, resident
~1.06×, only latency datum 10% slower); (3) "eviction cannot enter" refuted by own W11 table
(ea-k0.1 @0.100× = 100/92/100/17); (4) self-authored cyclic-filler benchmark, no official anchor;
(5) fp32-at-rest billed as fp16 (honest ≈0.15×/0.27×); (6) per-trial/env/SHA chain lost with pods;
(7) no manuscript (main.tex = 157 lines, 3 cites); (8) wording debts (5–13×, "beats", unnamed
comparator, C3-as-invention).

**AC-rejected attacks (do not re-litigate):** think-c0.3 doesn't threaten the marquee (0.852×
memory); mk drop ≠ selective reporting (documented W14 sizing miss — but fill it); pre-registration
not violated; "no single both-headline config" false (Llama marquee); C1 is one uniform config.
