# GATES — definition of done per lane (from KICKOFF_WEEKS0-3.md Part D)

G0  cleanup + reproducibility
    [ ] git tag paper-v1-archive == ee8c0ab
    [ ] docs/plan/cleanup/deletions.md: every removed path + reachability evidence
    [ ] reachability check: 0 unreachable files under src/; net LOC delta negative
    [ ] `make env && make test` green on clean clone; tests < 90 s
    [ ] `make tables` regenerates v1 Tables 1,2,3,6,7,8 from results/paper-v1/ and diffs
        clean against docs/plan/paper-v1-tables.md
    [ ] every v1 arm/task/pod is a YAML under configs/; w10_ruler/w10_frontier flag soup gone
    [ ] Dockerfile + uv.lock; scripts/pod.py --check passes on a synthetic manifest;
        pod launch/watchdog path reused, not rewritten
    [ ] forbidden-word grep clean; README ≤ 120 lines; ponytail after-report committed

G1  harness hygiene
    [ ] guard + tripwire merged; ratchet regression test passes in both directions
    [ ] effective-rank billing test passes
    [ ] recon.py scores from stored(); test: isvd stored error == cache rot-carried error (1e-6)
    [ ] FD step runs on the block that crashed the swap pod; no `--` arms
    [ ] Oja eta0/decay reach oja_step from config; tuned config committed
    [ ] rank-sweep figure regenerates from local 1B dumps (Week 1) and 8B dumps (Week 3)
    [ ] perplexity on TEST/PG-19 val, per-window NLL, paired CI + TOST
    [ ] Table-4 cells re-run under prereg; DECISIONS.md: guard alone removes divergence? y/n

G2  generator v2 + baselines
    [ ] filler-realism diagnostic harvested; DECISIONS.md: in-house generator retained/retired
    [ ] pairing test (identical prompt sha256 across arms); balanced-depth test
    [ ] kivi2_faithful: G=32, R=128, full-precision prefill; dequant test; bytes incl. scales
    [ ] ss2 on Mistral/Qwen 16K + 3 families 32K harvested; Holm result in DECISIONS.md
    [ ] svd_oracle rename complete; grep "palu" hits only the docstring + DECISIONS entry
    [ ] SnapKV/PyramidKV/EA k∈{0.10,0.15,0.25}, ThinK+SnapKV run in smoke pod
    [ ] OjaKV e2e runs on Llama 16K; version recorded
    [ ] ShadowKV and ea_k0.25 v1 rows appear in make tables

G3  gate 1 v2
    [ ] prereg SHA precedes launch SHA; ≤ 50 GPU-h in manifest
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
    [ ] all seven prereg files committed before their pods; SHA-order check passes
    [ ] tampered-manifest rejection test passes

G6  verifier
    [ ] every STATE.md "done" entry carries a verifier signature with commands + output paths
    [ ] no merge without spec-compliance review + /ponytail-review

G7  paper correction
    [ ] diff reviewed by me; unsupported sentences gone; measured decode cost stated
    [ ] DECISIONS.md: arXiv v1 status and v2 plan
