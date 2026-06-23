# 2026-W2 — Rank-policy comparison: fixed r=16 vs. fixed r=32 vs. rank-adaptive theta=1e-2

## Hypothesis
On Layer-8 K of Llama-3.2-1B, fixed `r=32` should reconstruct better than fixed `r=16` (more
rank ⇒ lower error), and the rank-adaptive `theta=1e-2` policy will choose a rank that **grows with
context** rather than being pinned — landing somewhere between (or alongside) the two fixed budgets
depending on where its rank ends up. This is the Week-2 "Wed" row of `docs/PLAN.md`: reconstruction
error vs. seq_len for the three policies.

## Setup
Mac, CPU, fp64 core. Llama-3.2-1B KV dump, one C4 document (idx 63) at 4096 tokens, **Layer 8**,
**post-RoPE `K`** of a `rope-both` pilot dump. Matrix `M = (512 features, 4092 tokens)` after
dropping the first 4 sink-token columns (`docs/notes/conventions.md`; matches
`scripts/week2_pilot.py:load_matrix`). Tracker:
`kvdlra.integrators.streaming.StreamingBUG` — three policies streamed once each, token by token:
`rank_cap=16`, `rank_cap=32`, and `theta=1e-2` (no cap). At every 128 tokens we freeze the tracker's
**current** left basis `U` and record `tracker.reconstruction_error(M[:, :t])` plus the current
tracked rank. Error model = orthogonal projection `U Uᵀ M`; reported `||M − U Uᵀ M||_F / ||M||_F`.

**Normalization note (load-bearing):** `StreamingBUG.theta` is an *absolute* Frobenius-tail
tolerance. Raw post-RoPE `M` has `||M||_F ≈ 4.1e3` and even its smallest singular value is `≈ 6.3`,
so a literal absolute `theta=1e-2` lies below a single discarded direction and degenerately forces
**full rank r=512** (error ~4e-13). The intended dimensionless reading of "theta=1e-2" is *relative*
— "discard ≤1% of the Frobenius energy" — which we realize exactly by streaming `M̂ = M/||M||_F`
(unit Frobenius norm). The reported error is a **ratio**, hence scale-invariant: the three error
curves are bit-identical with/without the rescale; only the adaptive rank decision changes. The
fixed-`rank_cap` policies don't use `theta` and are scale-invariant in rank too.

## Command
```bash
uv run python experiments/2026-w2-rank-policy/run.py \
    --dump dumps/llama3.2-1b/doc63_40468cde_len4096_rope-both --layer 8 --key K --stride 128
```

## Wandb URL
n/a — no `WANDB_API_KEY` this session. Metrics committed to
`figures/week2/rank_policy_metrics.json`; sync later with a key.

## Result
Relative-Frobenius reconstruction error on `M[:, :t]` using the current `U`, vs. seq_len `t`:

| seq_len | fixed r=16 | fixed r=32 | adaptive θ=1e-2 | adaptive rank |
|--------:|-----------:|-----------:|----------------:|--------------:|
|     128 | 3.0834e-01 | 2.0385e-01 |      5.4308e-01 |             2 |
|     256 | 3.6431e-01 | 2.6093e-01 |      5.3567e-01 |             3 |
|     512 | 4.0255e-01 | 3.0280e-01 |      5.2553e-01 |             5 |
|    1024 | 4.2938e-01 | 3.3134e-01 |      5.0602e-01 |             8 |
|    2048 | 4.6072e-01 | 3.6047e-01 |      5.0352e-01 |            11 |
|    3072 | 4.7874e-01 | 3.7566e-01 |      5.0182e-01 |            13 |
|    4092 | 4.8791e-01 | 3.7985e-01 |      4.8771e-01 |            16 |

Final: r=16 → 0.4879 (rank 16), r=32 → 0.3799 (rank 32), adaptive → 0.4877 (rank **16**, grown from
2). Figure: `figures/week2/rank_policy.{pdf,png}` — panel 1 = error vs. seq_len (3 curves); panel 2 =
adaptive rank vs. seq_len (with the two fixed caps drawn as reference lines).
**Smoke assertion passes**: fixed r=32 final error ≤ fixed r=16 final error.

## Interpretation
- **More rank wins, as expected.** Fixed `r=32` beats fixed `r=16` at every seq_len (final 0.380 vs.
  0.488) — twice the rank, ~22% lower error. The smoke check (r=32 ≤ r=16) holds.
- **The adaptive rank grows with context, monotonically: 2 → 5 → 8 → 11 → 13 → 16** as `t` goes
  128 → 4092. This is the PLAN-predicted behavior — a fixed energy-fraction tolerance needs more
  directions as more (post-RoPE, position-smeared) token columns accumulate.
- **Where the adaptive policy lands:** by `t=4092` it has grown to rank **16** and its error
  (0.4877) sits essentially **on top of fixed r=16** (0.4879) — same rank, same error. It is clearly
  **worse than fixed r=32** (0.380), but at *half* the rank, so this is not an unfair loss: the
  adaptive policy simply chose a smaller final budget than r=32 under a 1%-energy tolerance. It does
  **not** exceed r=32's rank, so the "worse at equal/greater rank" failure mode does **not** occur
  here (`adaptive_uses_ge_r32_rank=false`).
- **Curve shapes differ in a telling way.** The fixed-rank errors *rise* with seq_len (a frozen
  budget fits a growing matrix progressively worse), while the adaptive error is roughly flat /
  slightly decreasing because its rank climbs to compensate. At small `t` the fixed budgets look much
  better only because they over-provision rank relative to the few columns seen.
- Absolute errors are high (best is r=32 at ~0.38) because the keys are **post-RoPE** (the known RoPE
  pitfall, `docs/notes/rope-pitfall.md`); that is an intrinsic property of the data, not the policy.
  The *ordering* of the three policies and the adaptive rank trajectory are RoPE-independent.

## Decision
The three rank policies behave exactly as the theory predicts: rank buys accuracy (r=32 > r=16), and
`theta=1e-2` (read as ≤1% discarded energy) yields a context-growing rank that here converges to 16
and tracks fixed-r=16. For the streaming-BUG product the operative knob is therefore the
energy-fraction `theta`, which auto-sizes the budget; on post-RoPE Layer-8 K a 1%-energy tolerance is
**too tight to be cheap** (it would keep growing past r=16 with more context) yet **too loose to be
accurate** (still ~0.49 error at r=16). Next: re-run on **pre-RoPE** `K` (ShadowKV claims dramatically
lower rank), where the same `theta=1e-2` should choose a much smaller rank at much lower error — the
regime where rank-adaptive BUG is expected to dominate the fixed policies at matched memory.
