# kvdlra — agent feedforward

## What this is
Streaming low-rank KV-cache compression: a rank-r gist tracked online by a block
incremental-SVD step (`isvd`; the shipped "BUG step" at theta=None, min_sv_frac=0
IS Brand-2006 incremental SVD — src/kvdlra/tracker/isvd.py:296-298), a surprise-selected
exact tier, sinks, a recent ring, a warm-up seed. From Week 1: a fused
factored-attention kernel. Target: ICML 2027. Working branch: week7.

## Read before touching anything
docs/plan/ICML2027_PLAN.md · docs/plan/CODE_AUDIT.md · docs/plan/PC_REVIEW.md ·
docs/plan/STATE.md (append-only log) · docs/plan/DECISIONS.md (append-only) ·
docs/plan/lanes/*.md · prereg/*.md · docs/plan/reports/w20-close-report.md

## Facts that are settled (do not re-litigate; see CODE_AUDIT.md for file:line)
- The tracker is incremental SVD. "DLRA", "BUG integrator", "honest", "marquee",
  "flagship" do not appear in new code, configs, docs/paper, or commit messages.
  The CKL error bound does not apply to a column stream; no bound is claimed
  unless proven (a Frequent-Directions-type lemma is the applicable kind).
- U has no orthogonality guard; divergence is loss of orthonormality (audit A §Q4).
- §4.1 must score every method on its STORED representation.
- `palu-r0.5` is a per-sequence SVD oracle → rename `svd_oracle`.
- The KIVI arm is a transformers QuantizedCache mixin (G=64, chunked prefill).
  A faithful KIVI (G=32, R=128, full-precision prefill) is a separate arm.
- The Week-20 Oja swap arm used untuned defaults; the FD arm crashed. The
  "tracker load-bearing" sentence in paper/main.tex is unsupported.
- Measured decode: reconstruct path is 4–14× slower than full KV and its KV
  peak exceeds full KV. No systems claim is made until the kernel replaces it.
- The in-house filler (10 cycled sentences) is suspected of flattering the
  gist. No headline number comes from it after Week 0's realism diagnostic.
- Perplexity uses WikiText-103 TRAIN. Switch to TEST / PG-19 validation.
- A number is citable only if `make tables` regenerates it from a committed
  results/<pod>/trials.jsonl whose manifest passes `scripts/pod.py --check`
  and whose prereg/<pod>.md was committed before the launch commit.

## Skills
superpowers: brainstorming (only for genuinely open design questions: kernel ADR,
config schema, FD numerics), writing-plans (every lane, before code),
subagent-driven-development + dispatching-parallel-agents (execution; one lane
per git worktree via using-git-worktrees), test-driven-development (all src/),
systematic-debugging (any number that disagrees with expectation),
verification-before-completion (before any "done"), requesting-code-review /
receiving-code-review (before merge), finishing-a-development-branch (merge).
ponytail: `/ponytail full` for all src/ work; seven-rung ladder before every new
function; `/ponytail-audit` + `/ponytail-debt` in the cleanup lane;
`/ponytail-review` on every PR. Ponytail never removes: trust-boundary
validation, data-loss guards, the orthogonality tripwire, provenance, tests.

## Repo rules (CI-enforced after L0)
- Reuse what exists: uv.lock, pyproject pins, pre-commit, ci.yml, omegaconf configs,
  the vast.ai pod scripts + watchdog. Extend; do not rewrite.
- Target layout:
    src/kvdlra/{tracker,cache,tier,quant,baselines,kernel,eval,accounting,util}
    configs/{arms,tasks,pods}/*.yaml      # every experiment is a YAML
    prereg/*.md                           # committed BEFORE the pod launch commit
    scripts/{pod.py,tables.py,figures.py,dump_kv.py}  # the only entrypoints
    scripts/pod/                          # vast.ai launch/watchdog (kept, thinned)
    results/<pod>/{manifest.json,trials.jsonl,ppl.jsonl,diag.jsonl,env.txt}
    results/paper-v1/                     # archived per-trial logs behind v1 tables
    tests/                                # CPU-only, < 90 s, in CI
    docs/{plan,adr,paper}                 # nothing else under docs/
- Delete, don't archive — except files that produced a number in a paper-v1
  table, which are archived under git tag `paper-v1-archive` before removal.
- manifest.json records git SHA, config hash, HF model revision, dataset/
  haystack SHA256, torch/cuda/triton versions, GPU, wall-clock, command line.
- Seeds explicit; unseeded RNG raises. A trial that raises is recorded as
  `error` and counted; it never silently reduces n.
- One commit per logical change; commit message carries pod id + prereg path.

## State
docs/plan/STATE.md and docs/plan/DECISIONS.md are append-only, dated, never
edited. Every lane appends on start / block / done. Gate outcomes go to
DECISIONS.md with the evidence path.

## Definition of done: docs/plan/lanes/GATES.md
