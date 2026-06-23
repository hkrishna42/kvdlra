# 2026-W2 — Streaming rank-adaptive BUG vs. the truncated-SVD oracle

## Hypothesis
A *real* streaming BUG tracker that processes the Layer-8 K-cache one token (column) at a time,
tracking the left feature subspace, reaches a final reconstruction error competitive with the
truncated-SVD oracle (within ~1.5×) and no worse than incremental SVD — i.e. it is a sound
"rank-adaptive BUG" curve for the Week-2 go/no-go pilot (PLAN §5), unlike the crude single-sweep
placeholder in `scripts/sigma_decay.py`.

## Setup
Mac, CPU, fp64 core. `unsloth/Llama-3.2-1B-Instruct` KV dump, one C4 document (idx 15) at 2048
tokens, **Layer 8**. Matrix `M = (512 features, 2044 tokens)` after dropping the first 4 sink-token
columns (`docs/notes/conventions.md`; StreamingLLM, PLAN §8 #5). Tracker:
`kvdlra.integrators.streaming.StreamingBUG` — augmented (rank-adaptive) BUG step
(Ceruti–Kusch–Lubich 2022, arXiv:2104.05247 §2) specialized to the streaming relaxation field
`F(Y)=Y_target−Y` with the linear substeps solved exactly; per-token cost `O(n r + r^3)`, kept in
square-root form (no covariance squaring, PLAN §8 #4). Reuses `truncation_rank` from
`bug_adaptive.py`. Reconstruction model = orthogonal projection `U Uᵀ M`; reported error
`||M − U Uᵀ M||_F / ||M||_F`. Ranks 8/16/32/64/128; adaptive `theta=20`, cap 128 for the
rank-trajectory plot.

## Command
```bash
python experiments/2026-w2-streaming-bug/run.py \
    --dump dumps/llama3.2-1b/doc15_d1b3a6eb_len2048 --layer 8
```

## Wandb URL
n/a — no `WANDB_API_KEY` this session. Metrics committed to
`figures/week2/streaming_bug_metrics.json`; sync later with a key.

## Result
Final rel-Frobenius reconstruction error vs. rank (Layer 8):

| rank | StreamingBUG | oracle SVD | incremental SVD | BUG / oracle |
|-----:|-------------:|-----------:|----------------:|-------------:|
|    8 |   5.4597e-01 | 5.4167e-01 |      5.5451e-01 |        1.008 |
|   16 |   4.7897e-01 | 4.6881e-01 |      4.9230e-01 |        1.022 |
|   32 |   3.7461e-01 | 3.6234e-01 |      3.9065e-01 |        1.034 |
|   64 |   2.5470e-01 | 2.5181e-01 |      2.6590e-01 |        1.012 |
|  128 |   1.7461e-01 | 1.7134e-01 |      1.8182e-01 |        1.019 |

Adaptive `theta=20` (cap 128): final_rank=73, mean_rank=58.0, rel_err=2.40e-01.
Figure: `figures/week2/streaming_bug_rank.{pdf,png}` (rank vs. token index).
**VERDICT: competitive = True** (BUG within 1.5× oracle and ≤ incremental SVD at every rank).

## Interpretation
The streaming BUG tracker is essentially **on top of the oracle** — BUG/oracle ∈ [1.008, 1.034],
well inside the loose 1.5× bar and even inside the strict 1.05× pilot criterion — and it **beats
incremental SVD at every rank**. The earlier single-sweep placeholder (rel-err 0.55–0.96) is
replaced by a genuine token-at-a-time integrator. Absolute errors stay high (oracle needs r≈128 for
~17%) because the keys are **post-RoPE** (the known RoPE pitfall), but that is an intrinsic property
of the data, not the tracker; the *relative* BUG-vs-oracle gap is what the pilot tests and it passes
decisively. The exactness property is also verified on synthetic exactly-rank-r inputs
(`tests/test_streaming.py`, reconstructed to ~1e-15).

## Decision
Streaming BUG is a viable, near-oracle, streaming-friendly low-rank tracker — the algorithmic
prerequisite for the §5 go/no-go is met. The remaining go/no-go risk is **compressibility, not the
integrator**: post-RoPE Layer-8 K needs a large rank for low error. Next steps for the full §5
verdict: (1) sweep memory budget across multiple C4 docs with 1σ bands; (2) re-run on **pre-RoPE**
keys (ShadowKV claims dramatically lower rank), which is where the absolute numbers — not the
relative gap — should improve.
