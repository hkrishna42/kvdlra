# DECISIONS — append-only, dated. Gate outcomes carry the evidence path. OPEN = the owner (Hari) decides; the orchestrator records options + a recommendation and continues on everything not blocked. Closing an item = a new dated entry, never an edit.

## 2026-09-11

### D-001 OPEN — Is arXiv v1 live?
- Evidence: commit 9baf571 (2026-09-06, "arXiv package delivered") changed only docs/week19-handover.md; paper/arxiv-v1.tar.gz exists locally (gitignored); a web search for the exact title on 2026-09-11 returns no arXiv listing. → v1 appears assembled but NOT posted.
- Options: (a) not posted → the L7 corrections land in paper/main.tex only, no v2 is needed, and v1 is not posted as-is (it carries the unsupported "tracker load-bearing" sentence and the framing CODE_AUDIT refutes). (b) posted → the L7 diff becomes an arXiv v2 within the week.
- Recommendation: (a). Until you confirm, L7 proceeds under (a) and notes the v2 path in its report.

### D-002 OPEN — Kernel option: (iii) Triton tile-wise reconstruct-inside-attention vs (i) post-RoPE tracking at r=128
- Recommendation (ICML2027_PLAN §2 Gate 3; L4 brief): (iii) primary — it preserves the pre-RoPE design exactly as shipped; (i) as a parallel one-row accuracy experiment (`isvd_postrope_r128`, Llama 16K, n=24). L4 formalises this in docs/adr/0001; the ADR stays OPEN until you accept it.

### D-003 OPEN — GPU split between L3 (Gate 1 v2) and L4 (kernel)
- Recommendation (KICKOFF Part E §5): Gate 1 v2 first — it decides which paper exists; kernel prototypes on a shared card at batch 1. Gate 1 v2 is budgeted ≤ 50 GPU-h; credit is $23.85, so a top-up (~$60–90 at A100 rates) is required before the G1 v2 launch commit.

### D-004 OPEN — `palu-r0.5`: real Palu port vs `svd_oracle` as an explicit static-low-rank upper bound
- Recommendation: rename to `svd_oracle_r0.5` now (a settled fact; L2 §5). Decide on a real-Palu port only if Gate 1 selects Branch A/B — it matters for the Phase-2 tables, not for the gates.

### D-005 OPEN — Launch the filler-realism diagnostic pod (Day-1 milestone; L2 §1)
- Spec: in-house four tasks, Llama-3.1-8B, 16K, n=12, real-text filler; arms = isvd r64-h256 (the paper's r64 configuration), its q4 cell, KIVI-2 streaming, KIVI-2 single-shot, full. Reading pre-registered in prereg/filler_realism.md (committed before the launch commit): a drop > 0.25 on any task for the r64 or q4 arm retires the cycled-filler generator from every headline claim.
- Cost: ~2–4 A100-hours ≈ $3–6 of the $23.85 credit. Spends money → needs your go; the orchestrator does not launch pods.
- Recommendation: go, once prereg/filler_realism.md is committed and L2 confirms `--filler` accepts a real-text source (today `--filler wikitext` draws WikiText-2 test sentences — real text, but not the PG-19 chapter the brief names).

### D-001 evidence addendum (2026-09-11, orchestrator; still OPEN for the owner)
- L7 searched paper/ for an arXiv ID, abs URL, report number or submission receipt: none; every `arXiv:` string is a citation in refs.bib; main.tex's "arXiv v1 preprint" header is intent, not a record. Combined with 9baf571 (tarball assembled) and the empty title search, the recommendation stands: (a) not posted, no v2.

### D-005 addendum (2026-09-11, orchestrator; OPEN) — cost revised, reduced-arm option
- L2prep's estimate from harvested per-trial timings: 5 arms × 4 tasks × n=12 = 9.1 h base, 18.3 h at the 2× safety factor → **$11–20** (credit $23.85). The earlier "$3–6" omitted tasks-per-cell.
- Option (a) full design, 5 arms: $11–20, most of the credit. Option (b) decisive arms only — r64 config, q4 cell, full (the rule is written on r64/q4; KIVI-2 chunked is already known to be worse under chunked prefill and its single-shot variant is a Gate-2 question, not a filler question): ≈ 5.3 h base / 10.6 h at 2× → **$6–12**.
- Recommendation: (b) now; the two KIVI arms join the ss2_families pod (L2 §4) where they belong. Prereg file to be amended to (b) before launch if you choose it; the pod-name/watchdog fixes L2prep listed are done at launch time by L2.

### D-002 evidence addendum (2026-09-11, orchestrator; OPEN — ADR ready for acceptance)
- ADR 0001 drafted (lane/L4-kernel @ e432df3): (iii) primary, (i) parallel accuracy row `isvd_postrope_r128`, (ii) rejected (GQA: 4× the key-side FLOPs of (iii); the pair factorization is per query head, reconstruction per KV head). Cost model regenerable from docs/adr/0001-cost-model.py; Week-3 gate reachable in principle (resident KV 0.60 GB vs full 4.29 GB at 32K; roofline 10.7 ms vs measured 188 ms reconstruct).
- Three concerns the owner should weigh before accepting: (1) `BugStreamingLayer.lazy_initialization` raises above batch 1 — Gate 3's batch ≥ 4 criterion needs a batched cache first (scope for L4 Week 2–3); (2) weights + full KV at 64K batch 4 = 50.4 GB does not fit an A100-40GB — that contrast moves to H100 or to "max batch that fits"; (3) the traffic model under-predicts the measured reconstruct path by 6.6–16× (full KV by 2.1–2.3×), so absolute kernel timings are not predicted, only the floors.
- Recommendation: ACCEPT (iii) primary + (i) row, with concern (1) added to the L4 milestones as the first Week-2 task.
