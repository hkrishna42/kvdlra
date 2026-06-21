# 2026-06-20 — Scaffold + synthetic BUG integrator

## Hypothesis
A from-scratch implementation of the Ceruti–Lubich BUG integrator (arXiv:2010.02022 §3.1)
reproduces the analytic solution of a stiff matrix Lyapunov flow to the accuracy predicted by
its (first-order) time convergence, and the numpy implementation ports faithfully to a
mixed-precision torch version (fp32 core / bf16 storage) per plan §8 pitfall #4.

## Setup
- Hardware: MacBook (Apple Silicon, arm64), CPU only — Day 1–2 needs no GPU.
- Env: uv-managed CPython 3.12.13; torch 2.11.0 (CPU), numpy 2.3.5, scipy 1.18.0, pytest 9.0.3.
- Problem: `n=100`, `r=8` Lyapunov flow `Y' = -(B Y + Y B^T)`, `B = V − 0.5 D`,
  geometric singular values `10^{-i}`.
- References: dense RK45 `solve_ivp` (rtol 1e-10) and the analytic `e^{-tB} Y0 e^{-tB}` (B symmetric);
  the two agree to ~1e-11.

## Command
```
uv pip install -e ".[dev]"
uv run pytest -v
uv run mypy src tests && uv run ruff check . && uv run pre-commit run --all-files
```

## Wandb URL
n/a — Day 1–2 is local/CPU with no training run. The first wandb run is the Day-5 SV-decay sweep.

## Result
- `pytest`: **5 passed, 1 skipped** (~3 s). The skip is the pure-bf16 QR test — CPU LAPACK has no
  bf16 `geqrf`; it runs on CUDA.
- BUG error vs. RK45 reference at `h=1e-3`: **8.39e-5**.
- Convergence: error halves exactly as `h` halves — **ratio 2.00** across
  `h ∈ {2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4}` ⇒ first order.
- BUG error vs. exact at `h=6.25e-5`: **< 1e-5** (the plan's headline target, met at the finer `h`
  that first-order BUG requires).
- Mixed precision: fp32-storage/fp32-core rel-err **< 1e-3**; bf16-storage/fp32-core rel-err **~6.4e-2**.
- Install clean; mypy strict clean (11 files); ruff clean; all pre-commit hooks green.

## Interpretation
The integrator is correct: clean O(h) convergence to the analytic solution validates the substep
ordering (K & L from the old factors in parallel, then S forward). The plan's "`< 1e-5` at `h=1e-3`"
is optimistic for the *first-order* basic BUG — 8.4e-5 at `h=1e-3` is exactly the expected error, and
`1e-5` is reached near `h≈6e-5`. Mixed precision behaves as pitfall #4 predicts: bf16 is usable only
with an fp32 linear-algebra core; the pure-bf16 QR path is materially worse (and CPU LAPACK cannot run
it at all).

## Decision
Accept the integrator as Week-1 Track B complete. The brittle absolute threshold from the plan's
Script #1 is replaced by a first-order convergence test plus a fine-`h` test that meets `1e-5` (a
documented, evidence-backed deviation). Next: rank-adaptive BUG (Day 3–4) and — once `VAST_API_KEY`
and an HF token are provided — the GPU KV-capture + SV-decay figure (Day 3–7).
