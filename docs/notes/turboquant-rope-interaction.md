# TurboQuant × BUG × RoPE — method summary and the Π/RoPE interaction

Week-4 note (PLAN Mon). Covers (a) the TurboQuant method we will implement, (b)
the QJL residual estimator + its guarantees, and (c) the load-bearing question:
**how the fixed quantization rotation Π composes with RoPE** — and why our
pipeline sidesteps the non-commutation.

Source: Zandieh, Daliri, Hadian, Mirrokni, *"TurboQuant: Online Vector
Quantization with Near-optimal Distortion Rate"*, arXiv:2504.19874.

## 1. TurboQuant, in the form we implement

TurboQuant quantizes a **vector** ``x`` (a KV row/coordinate vector) in two stages.

**Stage 1 — rotated scalar quantization ("PolarQuant").**
- Normalize ``u = x / ‖x‖`` (store the scalar ``‖x‖`` separately).
- Apply a **fixed, data-oblivious random rotation** ``Π ∈ ℝ^{d×d}`` — generated
  once as ``Q`` from the QR decomposition of a Gaussian matrix (seeded, shared,
  never stored per-vector). Then ``Πu`` is uniform on the unit sphere.
- Each coordinate of ``Πu`` is then **Beta-distributed**,
  ``f(t) = Γ(d/2)/(√π Γ((d−1)/2)) · (1−t²)^{(d−3)/2}`` on ``[−1,1]``, which
  concentrates to ``N(0, 1/d)`` in high ``d``. Because the coordinates are
  near-independent, an **optimal 1-D scalar quantizer applied per coordinate** is
  near-optimal for the whole vector.
- Quantize each coordinate to ``b`` bits with **Lloyd–Max** levels computed for
  that marginal (2^b centroids on ``[−1,1]``, Voronoi cell boundaries at centroid
  midpoints; codebook precomputed once per ``(d, b)``).
- **Distortion (Thm 1):** ``D_mse ≤ (√3 π / 2) · 4^{−b}`` for unit vectors; the
  info-theoretic lower bound is ``4^{−b}``, so TurboQuant is within the constant
  ``√3π/2 ≈ 2.7`` of optimal (→ ~1.45 at ``b=1``). **This is the Tue unit-test
  target:** measured per-coordinate MSE ``≲ 2.7 · 4^{−b}``.

**Stage 2 — 1-bit QJL residual (unbiased inner products).**
MSE-optimal quantizers *bias* inner-product estimates. TurboQuant corrects this by
QJL-quantizing the residual ``r = x − Q_mse^{-1}(Q_mse(x))``:
- ``Q_qjl(x) = sign(S x) ∈ {±1}^d``, ``S ∈ ℝ^{d×d}`` i.i.d. ``N(0,1)`` (seeded).
- dequant ``Q_qjl^{-1}(z) = (√(π/2) / d) · Sᵀ z``.
- **Unbiased inner-product estimator** of ``⟨y, x⟩``:
  ``⟨y, Q_mse^{-1}(Q_mse(x))⟩ + ‖r‖ · ⟨y, Q_qjl^{-1}(Q_qjl(r))⟩``.
- **Variance (Lemma 4):** ``Var(⟨y, Q_qjl^{-1}(Q_qjl(x))⟩) ≤ (π / 2d) ‖y‖²``.
  **Wed unit-test target:** estimator unbiased; empirical variance within ~10% of
  ``(π/2d)‖y‖²``.

## 2. The Π / RoPE interaction (the actual question)

RoPE applies a **position-dependent** block-diagonal rotation ``R(pos)`` (2×2
rotations per frequency pair) to each key. TurboQuant applies a **fixed** rotation
``Π``. Two rotations in different planes **do not commute**: in general
``Π R(pos) ≠ R(pos) Π``. So *if we quantized post-RoPE keys directly*, the
quantizer's ``Π`` would entangle with the per-position ``R(pos)`` and we could not
cleanly factor position out (each position would see a different effective
codebook geometry).

**Why our pipeline avoids this entirely.** `BUGPress` already operates **pre-RoPE**
(Weeks 2–3): it factors the *pre-RoPE* keys ``M_pre ≈ U C``, and RoPE is
re-applied **after** reconstruction, as the last step before attention:

```
M_pre ──BUG──▶ (U, C=UᵀM_pre) ──TurboQuant──▶ quantize factors
                                   │
        dequantize ──▶ M̂_pre ──apply RoPE R(pos)──▶ K̂ (post-RoPE) ──▶ attention
```

TurboQuant quantizes the **pre-RoPE factors** (the coordinate columns of ``C``,
and optionally ``U``); ``Π`` lives in the pre-RoPE feature/coordinate space, and
``R(pos)`` is applied afterwards to the reconstructed keys. The two rotations are
**composed in sequence, never required to commute**. RoPE-free **values** are a
non-issue by construction. So the "Π and RoPE don't commute" caveat is real but
**does not bite our design** — a direct consequence of the pre-RoPE operating
point we already committed to.

## 3. What we quantize, and the two integration modes

**What.** The low-rank reconstruction is ``M̂ = U C`` with ``C = UᵀM`` (``r×T``).
For long ``T`` the **coordinates ``C`` dominate** storage (``r×T`` vs the fixed
``512×r`` of ``U``). So we TurboQuant-quantize the **per-token coordinate vectors**
``c_t ∈ ℝ^r`` (unit-normalize, rotate by ``Π_r``, Lloyd–Max at ``b`` bits, store
the norm), and keep ``U`` at fp16 (small, amortized) — a knob we can revisit.
Combined per-token cost ``≈ (r/512)·(b/16)`` of the full 16-bit cache, i.e. the
low-rank and quantization factors **multiply** (the Week-3 "compose to beat SOTA"
argument made concrete). Memory-budget sweep tunes ``(r, b)`` to hit
``{0.25×, 0.5×, 1×}``.

**Mode A — reconstruct-then-attend (Week-4 press, kvpress-native).** Dequantize
``ĉ_t``, form ``M̂_pre = U Ĉ``, apply RoPE, run standard attention. Simple, drops
into `BUGPress`. Here QJL is used as a **reconstructable residual** (add
``Q_qjl^{-1}(Q_qjl(r))`` back to ``ĉ_t``), improving reconstruction MSE.

**Mode B — estimator-at-attention (TurboQuant-native, future).** Never reconstruct
``k``; store 1-bit QJL codes and estimate ``⟨q, k⟩`` directly via the unbiased
estimator. Smaller/faster but needs a **custom attention kernel** (incompatible
with kvpress's reconstruct-then-standard-attention), so it is out of Week-4 scope.
We implement + unit-test the QJL estimator standalone (Wed) so Mode B is a later
drop-in.

## 4. Implementation plan (fp32 core, PLAN §8 #4)
- `src/kvdlra/quant/polar.py` — `PolarQuant(dim, bits)`: seeded ``Π`` via QR of a
  Gaussian; precomputed Lloyd–Max codebook for the Beta(``d``) marginal;
  `quantize(x) -> (codes, norm)`, `dequantize(...) -> x̂`. MSE test vs ``2.7·4^{-b}``.
- `src/kvdlra/quant/qjl.py` — `QJL(dim)`: seeded ``S``; `sign_hash(x)`,
  `estimate_inner(y, codes, norm)`. Unbiasedness + variance tests vs ``π/2d``.
- Compose in the press / a `quantize_factors` helper; memory-budget perplexity
  sweep; head-to-head vs SnapKV / ExpectedAttention (kvpress) for the hero figure.
