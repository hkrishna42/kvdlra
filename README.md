# kvdlra

**Streaming KV-cache compression for LLMs via Dynamical Low-Rank Approximation** — the
Ceruti–Lubich *BUG* integrator from numerical analysis
([arXiv:2010.02022](https://arxiv.org/abs/2010.02022),
[arXiv:2104.05247](https://arxiv.org/abs/2104.05247)) — with optional
[TurboQuant](https://arxiv.org/abs/2504.19874) residual quantization.

DLRA tracks a moving low-rank matrix without the σ_min stiffness that breaks naive (U, S, V)
ODE schemes, with a robust error bound *independent of the smallest singular value*. kvdlra
explores it as a principled, streaming-friendly alternative to greedy KV-cache eviction
heuristics (H2O, SnapKV).

> **Status — Week 1.** From-scratch BUG integrators (fixed-rank + rank-adaptive) validated
> against analytic references, and a reproducible singular-value-decay figure for the
> Llama-3.2-1B K-cache. Writeup: [`docs/week1.md`](docs/week1.md).

## Week 1 figure

![KV-cache singular-value decay](figures/week1/sv_decay.png)

Post-RoPE K-cache of Llama-3.2-1B (layers 0/8/15, one C4 doc @ 2048 tokens, 4 sink tokens
excluded). **Finding:** post-RoPE keys are *not* cleanly low-rank — even the truncated-SVD
oracle needs rank ≈128 of 512 to reach ~17% Frobenius error. This is the expected
[RoPE pitfall](docs/notes/rope-pitfall.md) and motivates compressing **pre-RoPE** keys
(Week 2/3). Details + reproduction: [`docs/week1.md`](docs/week1.md).

## Install

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest -q          # 8 passed, 1 skipped (bf16 QR skips on CPU LAPACK)
```

## Layout

- `src/kvdlra/integrators/` — BUG (`bug.py`), torch mixed-precision port (`bug_torch.py`),
  rank-adaptive BUG (`bug_adaptive.py`)
- `scripts/` — `capture_kv.py` (KV capture), `sigma_decay.py` / `week1_sv_decay.py` (the figure)
- `docs/` — [plan](docs/PLAN.md), [conventions](docs/notes/conventions.md),
  [RoPE pitfall](docs/notes/rope-pitfall.md), [Week 1 writeup](docs/week1.md)
- `experiments/` — dated lab-notebook entries (hypothesis → command → result → decision)

## License

Apache-2.0.
