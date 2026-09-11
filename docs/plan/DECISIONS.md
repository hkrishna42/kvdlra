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
