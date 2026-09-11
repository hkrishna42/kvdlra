# L1 — Harness Hygiene: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the harness defects the code audit found before any new measurement: an orthonormality tripwire + guard on the tracked basis, effective-rank billing, §4.1 scored on the stored representation, a Frequent-Directions step that cannot crash, the Oja schedule plumbed from config, and a held-out perplexity protocol with per-window records. Then the Table-4 re-run config + pre-registration (launch is an owner decision).

**Architecture:** All numerics live in `src/kvdlra/tracker/isvd.py` (one file: shared augmentation, the incremental-SVD truncation, FD shrinkage, Oja) and are consumed by `src/kvdlra/cache/bug_cache.py:_absorb_columns`. New knobs default to the safe behaviour and are bit-identical on every stream that never trips the guard (the L0 golden test proves it); the periodic unconditional QR is an experimental factor (`qr_every`, default off). Diagnostics are accumulated in-memory per layer (library code does no I/O) and drained by the runner into `results/<pod>/diag.jsonl`. §4.1 is re-done as a small `eval/recon.py` that drives every tracker through the same `step(u, b, block) -> (u, b, rot)` contract and scores `U @ C` with rot-carried coordinates — the cache's own model.

**Tech Stack:** torch 2.11 (CPU in tests), numpy, scipy (stats), omegaconf configs from L0, `datasets` for WikiText-103 test / PG-19 validation.

**Spec:** `docs/plan/lanes/L1_harness_hygiene.md`; `docs/plan/CODE_AUDIT.md` Part A §Q4 (ratchet mechanism, synthetic recipe), §Q2 (metric artifact), Part B §Q6 (Oja defaults), §Q9 (perplexity protocol); `docs/plan/ICML2027_PLAN.md` §1 (0.1–0.4); `docs/plan/lanes/GATES.md` §G1.

## Global Constraints

- Worktree `.claude/worktrees/L1-hygiene`, branch `lane/L1-harness-hygiene` off `week7` **after L0 Phase A has merged** (the plan uses the post-L0 paths: `src/kvdlra/tracker/isvd.py`, `src/kvdlra/eval/{config,frontier,records,stats}.py`, `scripts/pod.py`). If Phase B has not landed yet, `tracker/isvd.py` is still `src/kvdlra/integrators/streaming_torch.py` and `eval/frontier.py` is still `scripts/w10_frontier.py` — same edits, old paths; say so in the task report.
- Forbidden words in new code/configs/docs/commits: `DLRA` (word), `BUG integrator`, `honest(ly)`, `marquee`, `flagship`. Call the shipped step "incremental SVD"; the r64 configuration is `isvd_r64_h256_seed`.
- `tests/test_golden_cache.py` (L0) must stay green after every task: the defaults are bit-identical on any stream whose orthonormality error never exceeds `orth_fix_tol`.
- Never remove the tripwire once added (CLAUDE.md: ponytail never removes the orthogonality tripwire).
- A trial that raises (including `OrthonormalityError`) is recorded as `error` and counted — never dropped.
- No pod launch inside this lane. Task 6 produces the config + prereg; the launch commit must come AFTER the prereg commit and is an owner decision (DECISIONS.md).
- All new tests CPU-only; the whole suite stays < 90 s (mark a > 5 s test `@pytest.mark.slow` and keep it in CI unless the budget is exceeded).
- Commit messages carry `L1.<n>`.

---

## File map

```
src/kvdlra/tracker/isvd.py          + _augment(), _svd_core() (eigh fallback), orth_error(), reorthonormalize(),
                                      isvd_step() (= augmented_bug_step; old name kept as alias), fd_step() (rewritten
                                      on the shared augmentation), oja_step(eta0, decay REQUIRED), eff_rank()
src/kvdlra/tracker/__init__.py      TRACKERS = {"isvd": isvd_step, "fd": fd_step, "oja": oja_step}
src/kvdlra/cache/bug_cache.py       BugStreamingCache(..., orth_fix_tol=1e-3, orth_abort_tol=1e-1, qr_every=None,
                                      diag_every=64, oja_eta0=20.0, oja_decay=0.03); tracker "isvd" ("bug" accepted
                                      as a deprecated alias for one release); per-layer diag rows; drain_diag()
src/kvdlra/eval/frontier.py         _footprint bills the live rank
src/kvdlra/accounting.py            bug_footprint docstring: rank = live tracked rank
src/kvdlra/eval/recon.py            stored-representation reconstruction study (§4.1), trackers via the step contract
src/kvdlra/eval/data.py             + wikitext103_test(), pg19_validation() corpora
src/kvdlra/eval/frontier.py         ppl protocol: window 2048, n_samples 32, per-window rows, fp32 accumulation
src/kvdlra/eval/runner.py           drains diag rows -> results/<pod>/diag.jsonl; records OrthonormalityError as error
scripts/tables.py                   + `ppl` table: bits/token vs full, paired bootstrap CI + TOST(±0.05)
scripts/figures.py                  + `rank-sweep` figure from results/recon_*/recon.jsonl
configs/arms/oja_r64_h256_seed.yaml (eta0 20.0, decay 0.03), fd_r64_h256_seed.yaml, isvd_*_qr64.yaml variants,
configs/tasks/ppl_16k_wt103test.yaml, ppl_16k_pg19val.yaml (window 2048, n_samples 32)
configs/pods/hygiene_table4.yaml    Qwen r128/r256 × floor {off,on} × qr {off,on}; Llama r256 × same
prereg/hygiene_table4.md
tests/test_orth_guard.py, test_effective_rank_billing.py, test_fd_numerics.py, test_oja_plumbing.py,
tests/test_recon.py, test_ppl_protocol.py
```

---

### Task 1: Orthonormality tripwire + guard

**Files:**
- Modify: `src/kvdlra/tracker/isvd.py` (add `orth_error`, `reorthonormalize`, `eff_rank`)
- Modify: `src/kvdlra/cache/bug_cache.py` (`BugStreamingCache.__init__` / `BugStreamingLayer`: knobs, `_absorb_columns` hook, `diag` rows, `drain_diag()`), `src/kvdlra/cache/__init__.py` (export `OrthonormalityError`)
- Test: `tests/test_orth_guard.py`

**Interfaces:**
- Produces (tracker): `orth_error(u: Tensor) -> float` = `‖UᵀU − I‖_F`; `reorthonormalize(u, c, b) -> tuple[Tensor, Tensor, Tensor, Tensor]` returning `(q, r @ c, r @ b, r)` from the thin QR `u = q r` (so `u c == q (r c)` exactly; `r` is returned so the caller can rotate the quantized tier with it); `eff_rank(b: Tensor, rel: float = 1e-6) -> int` = count of diagonal entries of `b` above `rel · max`.
- Produces (cache): constructor kwargs `orth_fix_tol: float | None = 1e-3` (re-orthonormalize immediately when `orth_error > tol`; `None` disables), `orth_abort_tol: float | None = 1e-1` (raise `OrthonormalityError` above it; `None` disables), `qr_every: int | None = None` (unconditional re-orthonormalization every `qr_every` absorbs and after every rank change; the experimental factor), `diag_every: int = 64` (one diag row per layer per `diag_every` absorbs carrying the window max of `orth_error` and min of `eff_rank`). `BugStreamingCache.drain_diag() -> list[dict]` returns and clears rows `{layer, absorbs, tokens_seen, orth_err_k, orth_err_v, eff_rank_k, eff_rank_v, rank_k, rank_v, fixed_k, fixed_v}`.
- Exception: `class OrthonormalityError(RuntimeError)` with message `layer=<i> orth_err=<x> > abort_tol=<y> at absorb <n>`.

- [ ] **Step 1: Write the failing tests**

`tests/test_orth_guard.py`:
```python
"""The audit's synthetic ratchet (CODE_AUDIT Part A §Q4): n=512, cap=256, block 16,
rank-40 signal + 4 outlier channels x1e3 + 1e-2 noise, bf16-rounded input. Without the
guard ‖UᵀU−I‖ ratchets past 1; with it the error stays < 1e-5 and the reconstruction of
the signal survives."""

import pytest
import torch

from kvdlra.cache import BugStreamingCache, OrthonormalityError
from kvdlra.tracker.isvd import eff_rank, isvd_step, orth_error, reorthonormalize


def ratchet_stream(n: int = 512, t: int = 1400 * 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(n, 40, generator=g))[0]
    sig = q @ (torch.randn(40, t, generator=g) * torch.linspace(3.0, 0.5, 40).unsqueeze(1))
    m = sig + 1e-2 * torch.randn(n, t, generator=g)
    m[:4] *= 1e3                      # four massive-activation channels
    return m.to(torch.bfloat16).to(torch.float32)   # bf16-rounded, fp32 core (the pod path)


def track(m: torch.Tensor, cap: int, fix_tol: float | None, block: int = 16) -> tuple[float, int]:
    u = b = None
    worst, fixes = 0.0, 0
    for s in range(0, m.shape[1], block):
        u, b, _ = isvd_step(u, b, m[:, s : s + block], cap)
        e = orth_error(u)
        worst = max(worst, e)
        if fix_tol is not None and e > fix_tol:
            u, _, b, _ = reorthonormalize(u, u.new_zeros(u.shape[1], 0), b)
            fixes += 1
    return worst, fixes


@pytest.mark.slow
def test_ratchet_without_guard_exceeds_one() -> None:
    worst, _ = track(ratchet_stream(), cap=256, fix_tol=None)
    assert worst > 1.0


@pytest.mark.slow
def test_ratchet_with_guard_stays_tiny() -> None:
    worst_seen, fixes = track(ratchet_stream(), cap=256, fix_tol=1e-3)
    assert fixes >= 1
    # after every fix the error is at roundoff; the running max of the *post-fix* state:
    u = b = None
    m = ratchet_stream()
    post = 0.0
    for s in range(0, m.shape[1], 16):
        u, b, _ = isvd_step(u, b, m[:, s : s + 16], 256)
        if orth_error(u) > 1e-3:
            u, _, b, _ = reorthonormalize(u, u.new_zeros(u.shape[1], 0), b)
        post = max(post, orth_error(u))
    assert post < 1e-5


def test_reorthonormalize_preserves_reconstruction() -> None:
    g = torch.Generator().manual_seed(1)
    u = torch.linalg.qr(torch.randn(64, 8, generator=g))[0]
    u = u + 1e-3 * torch.randn(64, 8, generator=g)          # perturbed off orthonormality
    c = torch.randn(8, 50, generator=g); b = torch.diag(torch.rand(8, generator=g))
    q, c2, b2, r = reorthonormalize(u, c, b)
    assert orth_error(q) < 1e-6
    assert torch.allclose(q @ c2, u @ c, atol=1e-6) and torch.allclose(q @ b2, u @ b, atol=1e-6)
    assert torch.allclose(r @ c, c2)


def test_eff_rank_counts_above_floor() -> None:
    b = torch.diag(torch.tensor([3.0, 1.0, 1e-9, 0.0]))
    assert eff_rank(b) == 2


def test_cache_tripwire_records_and_aborts() -> None:
    from tests.test_bug_cache import make_tiny_model_and_cache   # the suite's fixture builder (real name per L0 Task 1)

    _, cache = make_tiny_model_and_cache(rank=8, orth_fix_tol=1e-3, orth_abort_tol=1e-1, diag_every=1)
    cache.ingest(torch.randn(2, 64, cache.n))                   # two blocks, real method names per the fixture
    rows = cache.drain_diag()
    assert rows and {"layer", "absorbs", "orth_err_k", "eff_rank_k", "rank_k", "fixed_k"} <= rows[0].keys()
    assert cache.drain_diag() == []                              # drained
    # force an abort: corrupt U and absorb once more
    layer = cache.layers[0]
    layer.u_k = layer.u_k * 3.0
    with pytest.raises(OrthonormalityError):
        cache.ingest(torch.randn(1, 16, cache.n))


def test_defaults_are_bit_identical_on_benign_stream() -> None:
    # the L0 golden covers the r64 configuration end to end; here the tracker alone:
    g = torch.Generator().manual_seed(2)
    m = torch.randn(128, 512, generator=g)
    u1 = b1 = u2 = b2 = None
    for s in range(0, 512, 16):
        u1, b1, _ = isvd_step(u1, b1, m[:, s : s + 16], 16)
        u2, b2, _ = isvd_step(u2, b2, m[:, s : s + 16], 16)
        if orth_error(u2) > 1e-3:      # never fires on a benign stream
            u2, _, b2, _ = reorthonormalize(u2, u2.new_zeros(16, 0), b2)
    assert torch.equal(u1, u2) and torch.equal(b1, b2)
```

- [ ] **Step 2: Run → FAIL** (`ImportError` on `orth_error`, `OrthonormalityError`).

- [ ] **Step 3: Implement the tracker helpers** (`src/kvdlra/tracker/isvd.py`):
```python
def orth_error(u: Tensor) -> float:
    r = u.shape[1]
    eye = torch.eye(r, dtype=u.dtype, device=u.device)
    return float(torch.linalg.norm(u.mT @ u - eye))


def reorthonormalize(u: Tensor, c: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Thin QR u = q r; re-express stored coordinates and the core in q so u @ c == q @ (r @ c)."""
    q, r = torch.linalg.qr(u, mode="reduced")
    return q.contiguous(), r @ c, r @ b, r


def eff_rank(b: Tensor, rel: float = 1e-6) -> int:
    d = torch.diagonal(b).abs()
    if d.numel() == 0 or float(d.max()) == 0.0:
        return 0
    return int((d > rel * d.max()).sum().item())
```
`isvd_step = augmented_bug_step` (add the new name; keep the old as an alias for one release with a one-line deprecation comment).

- [ ] **Step 4: Wire the cache.** In `BugStreamingLayer._absorb_columns`, after the tracker step and the coordinate carry (`self.c_k = rot_k @ self.c_k`), and after the quant-tier rotation, add for each of K and V:
```python
e = orth_error(self.u_k)
fixed = False
if self.orth_abort_tol is not None and e > self.orth_abort_tol:
    raise OrthonormalityError(f"layer={self.layer_idx} orth_err={e:.3e} > abort_tol={self.orth_abort_tol} at absorb {self._absorbs}")
if (self.orth_fix_tol is not None and e > self.orth_fix_tol) or (self.qr_every and self._absorbs % self.qr_every == 0):
    self.u_k, self.c_k, self.b_k, r = reorthonormalize(self.u_k, self.c_k, self.b_k)
    if self._q_len() > 0:
        self._rotate_quant_tier(r, torch.eye(...))   # K side only; V handled in its own branch with (I, r_v)
    fixed = True
```
(the V branch mirrors it; keep the two branches explicit rather than a loop — the existing code is explicit per stream). Increment `self._absorbs`; keep window maxima `self._diag_max_err_k/v`, minima `self._diag_min_rank_k/v`, and every `diag_every` absorbs append the row to `self.diag` and reset the window. `BugStreamingCache.drain_diag()` concatenates every layer's rows (with `layer` index) and clears them. Also re-orthonormalize after any rank change (`keep != r_old`) when `qr_every` is set (the brief).

- [ ] **Step 5: Run** `tests/test_orth_guard.py tests/test_golden_cache.py tests/test_bug_cache*.py` → PASS; time the two slow tests (`pytest --durations=5`); if either exceeds 10 s, reduce `t` to `1000*16` and confirm the ratchet still crosses 1.0 (the audit saw the jump at step ~1394; adjust the outlier scale to 3e3 if needed and record the recipe in the test docstring).

- [ ] **Step 6: Commit** — `L1.1: orthonormality tripwire + guard (orth_fix_tol/abort_tol/qr_every), diag rows, eff_rank; ratchet regression test`

### Task 2: Effective-rank billing

**Files:**
- Modify: `src/kvdlra/eval/frontier.py:_footprint` (`rank=int(arm["rank"])` → `rank=int(layer.u_k.shape[1]) if layer.u_k is not None else 0`), `src/kvdlra/accounting.py:bug_footprint` docstring (rank = live tracked rank), the `[arm ctx] … ratio=` log row to also print `eff_rank=<min over layers>`
- Test: `tests/test_effective_rank_billing.py`

- [ ] **Step 1: Test**
```python
import torch

from kvdlra import accounting as acc
from kvdlra.eval.frontier import _footprint
from tests.test_bug_cache import make_tiny_model_and_cache


def test_collapsed_floor_on_layer_bills_fewer_floats() -> None:
    # a rank-deficient stream with the floor on collapses the tracked rank below the cap
    _, cache = make_tiny_model_and_cache(rank=32, min_sv_frac=1e-2)
    g = torch.Generator().manual_seed(0)
    q = torch.linalg.qr(torch.randn(cache.n, 4, generator=g))[0]
    stream = (q @ torch.randn(4, 512, generator=g)).T.reshape(4, 128, cache.n)
    cache.ingest(stream)
    layer = cache.layers[0]
    assert layer.u_k.shape[1] < 32
    arm = {"kind": "bug", "rank": 32, "retention": "fifo", "hh_select": "attn"}
    fp = _footprint(arm, cache, t=512, n=cache.n, h_kv=cache.h_kv)
    configured = acc.bug_footprint(cache.n, rank=32, coord_count=layer._f_len(), recent_len=layer._recent_len())
    assert fp.float_equiv() < configured.float_equiv()
    assert fp.float_equiv() == cache.stored_state_numel() / cache.n_layers   # the anti-drift pin still holds
```
- [ ] **Step 2: Run → FAIL** (bills configured rank). **Step 3:** one-line fix + docstring + log field. **Step 4:** `make test` → PASS (including `tests/test_accounting.py`'s anti-drift pin). **Step 5:** Commit — `L1.2: bill the live tracked rank, not the configured one; eff_rank in the log row`.

### Task 3: FD numerics fix + Oja schedule plumbing + tracker names

**Files:**
- Modify: `src/kvdlra/tracker/isvd.py`: factor `_augment(u, b_core, block) -> (u_aug, b_fac, r_old)` out of `isvd_step` (the rank-revealing residual SVD with `tol = 100·eps·‖block‖_F`, the Parlett/Kahan re-orthogonalization, the `n − r` clamp); `_svd_core(b_fac) -> (u_loc, sigma)`: `torch.linalg.svd(b_fac, full_matrices=False)`; on `torch.linalg.LinAlgError` → `evals, evecs = torch.linalg.eigh(b_fac @ b_fac.mT)` (ascending → flip), `sigma = sqrt(clamp(evals, 0))`, `u_loc = evecs`; on a second failure add `1e-7 · trace/dim` jitter to the Gram and retry once; else re-raise. `fd_step` rewritten on `_augment` + `_svd_core` + shrinkage; `oja_step(..., eta0: float, decay: float)` — REQUIRED keyword arguments (no defaults). `TRACKERS = {"isvd": isvd_step, "fd": fd_step, "oja": oja_step}` in `tracker/__init__.py`.
- Modify: `src/kvdlra/cache/bug_cache.py`: `tracker: str = "isvd"` (accept `"bug"` as a deprecated alias that maps to `"isvd"` with a `DeprecationWarning`); `oja_eta0: float = 20.0`, `oja_decay: float = 0.03` constructor kwargs passed to `oja_step`; dispatch through `TRACKERS`.
- Modify: `src/kvdlra/eval/config.py:arm_kwargs` — drop the `{"isvd": "bug"}` alias map (the cache now speaks `isvd`); `configs/arms/oja_r64_h256_seed.yaml` (`tracker: oja`, `oja_eta0: 20.0`, `oja_decay: 0.03`, `legacy_name: bugSseed-r64-h256-oja`, `doc:` "Week-2 validated pre-RoPE schedule; the W20 swap ran the (1.0, 1e-3) defaults and is void"); `configs/arms/fd_r64_h256_seed.yaml` (`tracker: fd`, ℓ = r = 64 — bytes-matched to isvd).
- Tests: `tests/test_fd_numerics.py`, `tests/test_oja_plumbing.py`; update `tests/test_w20_tracker_swap.py` to the new names.

- [ ] **Step 1: Tests**

`tests/test_fd_numerics.py`:
```python
import pytest
import torch

from kvdlra.tracker import isvd
from kvdlra.tracker.isvd import _svd_core, fd_step, isvd_step, orth_error


def test_svd_core_falls_back_to_eigh_on_linalg_error(monkeypatch: pytest.MonkeyPatch) -> None:
    g = torch.Generator().manual_seed(0)
    b = torch.randn(40, 56, generator=g)
    u_ref, s_ref, _ = torch.linalg.svd(b, full_matrices=False)
    real_svd = torch.linalg.svd

    def boom(*a, **k):
        raise torch.linalg.LinAlgError("linalg.svd failed to converge")

    monkeypatch.setattr(torch.linalg, "svd", boom)
    u, s = _svd_core(b)
    monkeypatch.setattr(torch.linalg, "svd", real_svd)
    assert torch.allclose(s, s_ref, atol=1e-5)
    assert torch.allclose((u * s) @ (u * s).mT, (u_ref * s_ref) @ (u_ref * s_ref).mT, atol=1e-4)


def test_fd_shrinkage_repeated_zero_spectrum_does_not_raise() -> None:
    # a stream whose augmented core carries many exact zeros after shrinkage (the swap-pod crash class)
    g = torch.Generator().manual_seed(1)
    q = torch.linalg.qr(torch.randn(256, 6, generator=g))[0]
    m = q @ torch.randn(6, 16 * 300, generator=g)
    u = b = None
    for s in range(0, m.shape[1], 16):
        u, b, _ = fd_step(u, b, m[:, s : s + 16], 64)
    assert orth_error(u) < 1e-4 and torch.isfinite(b).all()


def test_fd_matches_isvd_subspace_when_no_tail() -> None:
    g = torch.Generator().manual_seed(2)
    m = torch.randn(64, 320, generator=g)
    ua = ba = ub = bb = None
    for s in range(0, 320, 16):
        ua, ba, _ = isvd_step(ua, ba, m[:, s : s + 16], 64)   # cap == n: nothing discarded
        ub, bb, _ = fd_step(ub, bb, m[:, s : s + 16], 64)
    assert torch.allclose(ua @ ua.mT, ub @ ub.mT, atol=1e-5)


@pytest.mark.slow
def test_fd_survives_the_ratchet_stream() -> None:
    from tests.test_orth_guard import ratchet_stream

    m = ratchet_stream(t=600 * 16)
    u = b = None
    for s in range(0, m.shape[1], 16):
        u, b, _ = fd_step(u, b, m[:, s : s + 16], 256)
    assert torch.isfinite(u).all()
```
`tests/test_oja_plumbing.py`:
```python
import pytest
import torch

import kvdlra.cache.bug_cache as bc
from kvdlra.eval.config import arm_kwargs, load_arm
from tests.test_bug_cache import make_tiny_model_and_cache


def test_config_schedule_reaches_oja_step(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[float, float]] = []
    real = bc.TRACKERS["oja"]

    def spy(u, b, block, cap, *, n_seen, eta0, decay):
        seen.append((eta0, decay))
        return real(u, b, block, cap, n_seen=n_seen, eta0=eta0, decay=decay)

    monkeypatch.setitem(bc.TRACKERS, "oja", spy)
    kw = arm_kwargs(load_arm("oja_r64_h256_seed"), t=1024)
    assert kw["tracker"] == "oja" and kw["oja_eta0"] == 20.0 and kw["oja_decay"] == 0.03
    _, cache = make_tiny_model_and_cache(rank=8, tracker="oja", oja_eta0=20.0, oja_decay=0.03)
    cache.ingest(torch.randn(2, 64, cache.n))
    assert seen and all(s == (20.0, 0.03) for s in seen)


def test_oja_step_has_no_default_schedule() -> None:
    from kvdlra.tracker.isvd import oja_step

    with pytest.raises(TypeError):
        oja_step(None, None, torch.randn(8, 4), 4)     # eta0/decay are required


def test_bug_alias_warns_and_maps_to_isvd() -> None:
    with pytest.warns(DeprecationWarning):
        _, cache = make_tiny_model_and_cache(rank=8, tracker="bug")
    assert cache.layers[0].tracker == "isvd"
```
- [ ] **Step 2: Run → FAIL. Step 3: implement** (the refactor of `isvd_step` onto `_augment`/`_svd_core` must keep `tests/test_golden_cache.py` and `tests/test_streaming_torch.py` green — the SVD path is unchanged when it converges). **Step 4:** `make test` → PASS; `mypy` clean. **Step 5:** Commits — `L1.3a: shared augmentation + _svd_core with eigh fallback; fd_step on it`, `L1.3b: oja_step requires eta0/decay; cache plumbs oja_eta0/oja_decay from config; tracker 'isvd' (bug = deprecated alias)`.

### Task 4: §4.1 on the stored representation — `eval/recon.py` + rank-sweep figure

**Files:**
- Create: `src/kvdlra/eval/recon.py`, `tests/test_recon.py`
- Modify: `scripts/figures.py` (`rank-sweep` subcommand), `scripts/dump_kv.py` (document the dump layout the study reads: per layer `k_pre` / `v` `(n, T)` fp32 tensors — confirm against `capture_kv.py`'s writer)

**Interfaces:**
- Produces: `Method = Literal["svd_oracle","isvd","fd","fd2","oja","frozen_prefill_svd","random_basis"]`; `stored_error(m: Tensor, method: Method, r: int, block: int = 16, *, oja: tuple[float,float] | None = None, prefill: int = 4096, seed: int = 0) -> float` — runs the tracker through blocks of `m` (`(n, T)`), carries coordinates `c ← rot @ c; c = cat(c, uᵀ block)`, and returns `‖m − u @ c‖_F / ‖m‖_F` (`svd_oracle`: `‖m − u_r s_r vh_r‖_F/‖m‖_F`; `frozen_prefill_svd`: `u` = left singular vectors of the first `prefill` columns, then `c = uᵀ block`; `random_basis`: fixed Haar-random `u`; `fd2` = FD with ℓ = 2r); `tune_oja(docs: list[Tensor], r: int, grid: dict[str, list[float]]) -> tuple[float, float]` (argmin of mean stored error over the held-out docs; RAISES if the argmin lies on the grid boundary — widen the grid, the audit's complaint); `run_study(dump_dir: Path, ranks, methods, layers="all", kv=("k","v"), blocks=(16,128), out: Path)` writing `recon.jsonl` rows `{model, doc, layer, kv, method, rank, block, err, oja_eta0, oja_decay}`.

- [ ] **Step 1: Tests**
```python
import torch

from kvdlra.eval.recon import stored_error, tune_oja


def _stream(n: int = 96, t: int = 640, k: int = 12, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(n, k, generator=g))[0]
    return q @ (torch.randn(k, t, generator=g) * torch.linspace(4.0, 0.5, k).unsqueeze(1)) + 0.05 * torch.randn(n, t, generator=g)


def test_isvd_stored_error_equals_cache_rot_carried_error() -> None:
    """GATES G1: the study's isvd number IS the cache's own gist fidelity (to 1e-6)."""
    from kvdlra.tracker.isvd import isvd_step

    m = _stream()
    u = b = c = None
    for s in range(0, m.shape[1], 16):
        blk = m[:, s : s + 16]
        u, b, rot = isvd_step(u, b, blk, 8)
        c = u.mT @ blk if c is None else torch.cat([rot @ c, u.mT @ blk], dim=1)
    cache_err = float(torch.linalg.norm(m - u @ c) / torch.linalg.norm(m))
    assert abs(stored_error(m, "isvd", 8, block=16) - cache_err) < 1e-6


def test_oracle_is_a_lower_bound() -> None:
    m = _stream()
    o = stored_error(m, "svd_oracle", 8)
    for meth in ("isvd", "fd", "fd2", "frozen_prefill_svd", "random_basis"):
        assert stored_error(m, meth, 8, prefill=128) >= o - 1e-9, meth


def test_frozen_basis_is_worse_on_a_drifting_stream() -> None:
    a, b = _stream(seed=1), _stream(seed=2)
    m = torch.cat([a, b], dim=1)        # subspace changes half-way
    assert stored_error(m, "frozen_prefill_svd", 8, prefill=640) > stored_error(m, "isvd", 8) + 0.05


def test_tune_oja_rejects_boundary_optimum() -> None:
    import pytest

    docs = [_stream(seed=s) for s in (3, 4)]
    with pytest.raises(ValueError):
        tune_oja(docs, r=8, grid={"eta0": [20.0], "decay": [0.03]})   # a 1-point grid is all boundary
```
- [ ] **Step 2: Run → FAIL. Step 3: implement `recon.py`** (~120 lines; trackers imported from `kvdlra.tracker`; no I/O except `run_study`'s jsonl writer). **Step 4:** run the study on the local 1B dumps on CPU: `python -c "from kvdlra.eval.recon import run_study; ..."` for ranks {16,32,64,128,256}, layers all, K and V, blocks {16,128}, methods all → `results/recon_1b/recon.jsonl` + `manifest.json` (dump SHA manifest from `scripts/dump_kv.py verify`, git SHA, wall clock); `scripts/figures.py rank-sweep --in results/recon_1b/recon.jsonl --out docs/paper/figures/rank_sweep_1b.pdf` (mean ± SE over docs, one panel per K/V, log-y). Expected from the audit: isvd ≈ 1.03–1.05× oracle at the stored representation; fd2 at the oracle; frozen and random clearly above. **Step 5:** `make test` → PASS. **Step 6:** Commits — `L1.4a: eval.recon — stored-representation reconstruction study (all trackers through the step contract)`, `L1.4b: results/recon_1b + rank-sweep figure (1B, CPU)`.
- [ ] **Step 7:** 8B dumps (Week 3): the command is `scripts/dump_kv.py dump --model <id> --docs pg19:8 --ctx 16384 32768 --layers all --out dumps/<model>` on a pod; the SHA manifest is committed, the data is not. Launch = owner decision (DECISIONS); record the command in the task report.

### Task 5: Perplexity protocol — held-out corpora, window 2048, ≥ 32 windows, per-window rows, paired CI + TOST

**Files:**
- Modify: `src/kvdlra/eval/data.py` (+ `wikitext103_test()`, `pg19_validation()` loaders via `datasets`, streamed and concatenated, with the corpus SHA256 recorded), `src/kvdlra/eval/frontier.py` (window/n_samples from `TaskCfg`; fp32 NLL accumulation — already `_nll_sum`; emit one `ppl.jsonl` row per window `{model, arm, ctx, corpus, window_idx, nll_sum_nats, n_tok}`), `configs/tasks/ppl_16k_wt103test.yaml`, `ppl_32k_wt103test.yaml`, `ppl_16k_pg19val.yaml` (`window: 2048`, `n_samples: 32`), `scripts/tables.py` (`ppl` table: per arm bits/token, Δ vs full with `paired_bootstrap` 95% CI and `tost(delta=0.05)`)
- Test: `tests/test_ppl_protocol.py` — with a fake scorer returning known per-window NLLs, the `ppl` table prints bits/token = nll/(n·ln 2), the paired CI brackets the true Δ, and TOST reports equivalence for a 0.01-bit shift and non-equivalence for 0.10.

- [ ] Steps: test → fail → implement → `make test` → commit `L1.5: perplexity on WikiText-103 TEST / PG-19 validation, window 2048 x 32, per-window NLL rows, paired CI + TOST`.

### Task 6: Table-4 re-run config + pre-registration (no launch)

**Files:**
- Create: `configs/arms/isvd_r128_qr64.yaml`, `isvd_r256_qr64.yaml`, `isvd_r128_f0.01_qr64.yaml`, `isvd_r256_f0.01_qr64.yaml` (each = the plain arm + `qr_every: 64`), `configs/pods/hygiene_table4.yaml` (Qwen2.5-7B: `isvd_r128`, `isvd_r256`, `isvd_r128_f0.01`, `isvd_r256_f0.01` × qr {off, on}; Llama-3.1-8B: `isvd_r256`, `isvd_r256_qr64`; tasks `ppl_16k_wt103test` (+ `ruler_inhouse_16k` for the r128 Qwen arms only, to confirm retrieval-neutrality); `gpu_budget_h: 6`), `prereg/hygiene_table4.md`
- Test: `tests/test_pod_manifest.py::test_dry_run_writes_manifest_and_env` parametrized over `hygiene_table4` (dry-run loads the config).

`prereg/hygiene_table4.md` states: purpose (does the orthonormality guard alone remove the Qwen r=256 divergence — ppl 27,531 at ee8c0ab — without the singular-value floor? Which of "orthonormality ratchet" / "rank siphoning" is the mechanism the paper reports?); arms/tasks/n (32 windows); primary contrast = `isvd_r256_qr64` vs `isvd_r256` on Qwen 16K ppl (paired over windows; decision rule: guard-alone ppl within 0.05 bits of `isvd_r256_f0.01` ⇒ "orthonormality ratchet" is the mechanism and the floor becomes optional; guard-alone still > 2× full ⇒ the floor is required and the paper keeps both, naming the ratchet as the substrate); secondary: `diag.jsonl` max `orth_err` and min `eff_rank` traces per arm (the figure for §3.4); family size 4 (Holm); GPU budget 6 h (≈ $5–8); provenance (pod `hygiene-table4`, prereg SHA must precede the launch SHA); `STATUS: awaiting owner go`.

- [ ] Steps: write configs + prereg → dry-run manifest test → commit `L1.6: hygiene_table4 pod config + prereg (guard on/off x floor on/off; launch = DECISIONS)`.

---

## Self-review

- **Spec coverage (L1 brief §1–§7):** §1 guard + tripwire + diag + ratchet test both directions → Task 1. §2 effective-rank billing + test → Task 2. §3 recon.py on `stored()` (every tracker through one contract), methods list, local 1B run now, 8B dumps later via `dump_kv.py` → Task 4. §4 FD numerics (eigh of the Gram, driver fallback, jitter only on second failure; tests on the crash class + the ratchet stream) → Task 3. §5 Oja schedule from config, default (20.0, 0.03), config file, spy test → Task 3. §6 perplexity TEST/PG-19, window 2048, ≥32 windows, per-window NLL, fp32, paired CI + TOST → Task 5. §7 Table-4 cells under prereg → Task 6 (launch withheld — money).
- **GATES G1 mapping:** guard + tripwire merged; ratchet regression both directions ✓ T1 · effective-rank billing test ✓ T2 · recon from stored(); isvd == cache rot-carried to 1e-6 ✓ T4 · FD runs on the crash class; no `--` arms (errors are recorded) ✓ T3 + runner · Oja eta0/decay reach oja_step from config; tuned config committed ✓ T3 · rank-sweep figure from 1B dumps (Week 1) ✓ T4; 8B (Week 3) pending the dump pod · perplexity on TEST/PG-19 val, per-window, paired CI + TOST ✓ T5 · Table-4 under prereg + DECISIONS ✓ T6 (after the owner's go).
- **Placeholders:** the fixture builder name in the cache tests is the real helper L0 Task 1 identified (`tests/test_bug_cache.py`); the `_rotate_quant_tier(r, I)` call shape must match the existing signature (`rot_k, rot_v`) — the implementer reads `bug_cache.py:1260-1274`.
- **Type consistency:** `reorthonormalize` returns `(q, c, b, r)` in every use; `TRACKERS` keys are the `tracker` strings the config and the cache agree on (`isvd`, `fd`, `oja`); `stored_error`'s `Method` names are the `configs/arms` tracker names plus the two controls.
- **Not in L1 (deliberately):** any change to surprise scoring, the seed, or the tier — those are L2/L3 territory; the kernel (L4); bf16 gist storage (L5).
