# GATES — definition of done per lane (from KICKOFF_WEEKS0-3.md Part D)

G0  cleanup + reproducibility
    [x] git tag paper-v1-archive == ee8c0ab
    [x] docs/plan/cleanup/deletions.md: every removed path + reachability evidence
    [x] reachability check: 0 unreachable files under src/; net LOC delta negative
    [x] `make env && make test` green on clean clone; tests < 90 s
    [x] `make tables` regenerates v1 Tables 1,2,3,6,7,8 from results/paper-v1/ and diffs
        clean against docs/plan/paper-v1-tables.md
    [x] every v1 arm/task/pod is a YAML under configs/; w10_ruler/w10_frontier flag soup gone
    [x] Dockerfile + pyproject exact pins (R13 2026-09-14: uv.lock stays gitignored — platform-specific; `pod.py check` diffs env.txt against the pins); scripts/pod.py --check passes on a synthetic manifest;
        pod launch/watchdog path reused, not rewritten
    [x] forbidden-word grep clean; README ≤ 120 lines; ponytail after-report committed
    ✔ G0 PASSED 2026-09-14 at week7 10c19ac — evidence per line in docs/plan/STATE.md (L0 Phase B entry); ledger docs/plan/cleanup/l0-ledger.md.

G1  harness hygiene
    [x] guard + tripwire merged; ratchet regression test passes in both directions
    [x] effective-rank billing test passes
    [x] recon.py scores from stored(); test: isvd stored error == cache rot-carried error (1e-6)
    [x] FD step runs on the block that crashed the swap pod; no `--` arms
    [x] Oja eta0/decay reach oja_step from config; tuned config committed
    [~] rank-sweep figure regenerates from local 1B dumps (Week 1) and 8B dumps (Week 3)
    [x] perplexity on TEST/PG-19 val, per-window NLL, paired CI + TOST
    [x] Table-4 cells re-run under prereg; DECISIONS.md: guard alone removes divergence? y/n  — outcome 3(a), D-011 addendum 8: the guard alone removes 98.4% of the Qwen r256 divergence (+10.94 → +0.17 bits); the floor closes the remaining 0.15
    ✔ G1 2026-09-18: lines 1–5, 7, 8 met (4 as amended); 6 = 1B half (8B open — dump pod). Evidence: docs/plan/STATE.md, docs/plan/DECISIONS.md D-011/D-012–D-015, docs/plan/cleanup/l1-ledger.md.

G2  generator v2 + baselines
    [x] filler-realism diagnostic harvested; DECISIONS.md: in-house generator retained/retired  — RETIRED (D-005 CLOSED 2026-09-19: r64 0.25/0.08/0.00 and q4 0/0/0 on real text vs 1.00 cycled; ceiling 1.00/1.00/0.92; harness replicates the archive; results/filler_realism{,_cycle})
    [x] pairing test (identical prompt sha256 across arms); balanced-depth test  — tests/test_gen_v2.py; on hardware: 48/48 paired keys identical in results/filler_realism_cycle/trials.jsonl
    [x] kivi2_faithful: G=32, R=128, full-precision prefill; dequant test; bytes incl. scales  — src/kvdlra/quant/kivi.py, tests/test_kivi_faithful.py
    [ ] ss2 on Mistral/Qwen 16K + 3 families 32K harvested; Holm result in DECISIONS.md  — pre-registered (prereg/ss2_families.md, 3 pods); launch after the D-003 top-up; amended to ruler_v2_* per D-005
    [x] svd_oracle rename complete; grep "palu" hits only the docstring + DECISIONS entry  — met as amended (PR-L2-14): hits = disclaimer + archive keys + DECISIONS; tests/test_palu_rename.py
    [ ] SnapKV/PyramidKV/EA k∈{0.10,0.15,0.25}, ThinK+SnapKV run in smoke pod  — arms built (src/kvdlra/baselines/presses.py); pod pre-registered (prereg/l2_smoke.md); launch = DECISIONS
    [ ] OjaKV e2e runs on Llama 16K; version recorded  — BLOCKED: D-017 (repo @182b034, arXiv 2509.21623v2 recorded; their code cannot run through this harness)
    [x] ShadowKV and ea_k0.25 v1 rows appear in make tables  — docs/paper/tables/table_baselines.md; make tables diff-clean
    ✔ G2 2026-09-19: lines 1, 2, 3, 8 met; 5 met as amended; 4 and 6 pre-registered, open until launched; 7 open (D-017). Evidence: docs/plan/STATE.md, docs/plan/DECISIONS.md D-004/D-005/D-006/D-017, docs/plan/cleanup/l2-ledger.md.

G3  gate 1 v2
    [ ] prereg SHA precedes launch SHA; the PRE-REGISTERED bar (prereg/gate1_tracker_swap_v2.md §9: 41.0 h point, 82.0 h bar per Stage-1 pod at the L2-measured rates) in the manifest — the plan's "≤ 50 GPU-h" was a sizing estimate written before those rates (2× low) and is retired by D-003 (2026-09-20)
    [ ] 6 trackers × 3 families × 2 ctx × 4 tasks × n=24 harvested; 0 arms with error > budget
    [ ] make tables renders Holm-corrected retrieval + TOST perplexity
    [ ] DECISIONS.md names the branch with the rule from ICML2027_PLAN.md

G4  kernel
    [ ] ADR 0001 with FLOP/HBM model; OPEN or ACCEPTED by me
    [ ] single-layer max|Δ| < 1e-2 bf16; full-model greedy token-exact ≥ 14/16
    [ ] results/kernel_smoke/manifest.json: full / reconstruct / kernel at 16K/32K/64K, b=1,4;
        kernel KV peak < full at 32K; ms/token < reconstruct by ≥ 3×

G5  bf16 gist + prereg
    [ ] bf16 run harvested under prereg; non-inferiority result in DECISIONS.md
    [x] all eight prereg files committed before their pods; SHA-order check passes — prereg/{bf16_gist,filler_realism,gate1_preflight,gate1_tracker_swap_v2,hygiene_table4,kernel_smoke,l2_smoke,ss2_families}.md; `pod.py launch` refuses a non-ancestor (scripts/pod.py `prereg_error`), tests/test_pod_manifest.py::test_prereg_commit_order_against_real_history + ::test_prereg_refusal_reasons
    [x] tampered-manifest rejection test passes — tests/test_pod_manifest.py::test_check_rejects_tampered_config_hash (lane/L3-gate1-tracker-swap-v2)

G6  verifier
    [ ] every STATE.md "done" entry carries a verifier signature with commands + output paths
    [ ] no merge without spec-compliance review + /ponytail-review

G7  paper correction
    [ ] diff reviewed by me; unsupported sentences gone; measured decode cost stated
    [ ] DECISIONS.md: arXiv v1 status and v2 plan
