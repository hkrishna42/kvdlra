"""Week-20 tracker-swap ablation: the gist tracker is pluggable behind
``BugStreamingCache(tracker=...)`` with the BUG step's exact ``(u, b, rot)`` contract.

Pins: (1) ``oja_step``/``fd_step`` honor the contract (orthonormal basis, ``rot =
u_new^T u_old``, square core) and Oja grows to the rank it is billed for; (2) FD's
shrinkage removes energy relative to the plain augmented (incremental-SVD) step; (3)
``tracker="isvd"`` is bit-identical to the default, and the deprecated ``"bug"`` alias
reaches the same step; (4) the swapped trackers run end-to-end through prefill + decode;
(5) ``build_arms`` names the swapped arms and threads the knob.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.tracker.isvd import augmented_bug_step, fd_step, frozen_step, oja_step

# The Week-2 validated pre-RoPE schedule, which the arm config names. ``oja_step`` has no
# defaults to fall back on, so every call site states the schedule it is testing.
OJA = partial(oja_step, eta0=20.0, decay=0.03)
# L3.1's Gate-1 control, in its TRACKING phase (a freeze the stream never reaches), which
# is where it has a contract to honor at all -- past the freeze it returns its input. Its
# sibling ``random_step`` is not here: its basis is rank_cap-wide from the seeding call, so
# it cannot honor the seeding shape below by design (pinned in tests/test_gate1_arms.py).
FROZEN = partial(frozen_step, n_seen=0, freeze_after=1 << 30)

N, R, B = 32, 6, 5  # features, rank cap, block columns


def _orthonormal(u: torch.Tensor, tol: float = 1e-5) -> bool:
    eye = torch.eye(u.shape[1], dtype=u.dtype)
    return bool(torch.allclose(u.mT @ u, eye, atol=tol))


def _stream(seed: int, t: int = 40, true_rank: int = 4) -> torch.Tensor:
    """A low-rank column stream (n x t) plus small noise, fp32."""
    g = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(N, true_rank, generator=g))[0]
    coef = torch.randn(true_rank, t, generator=g)
    out: torch.Tensor = basis @ coef + 0.01 * torch.randn(N, t, generator=g)
    return out


@pytest.mark.parametrize("step", [OJA, fd_step, FROZEN], ids=["oja", "fd", "frozen"])
def test_swapped_trackers_honor_the_step_contract(step: Any) -> None:
    m = _stream(0)
    u, b, rot = step(None, None, m[:, :B], R)  # seeding = reduced QR, like the isvd step
    assert u.shape == (N, min(R, B)) and _orthonormal(u)
    assert b.shape == (u.shape[1], u.shape[1]) and rot.shape == (u.shape[1], 0)
    for start in range(B, m.shape[1] - B, B):
        u_old = u
        u, b, rot = step(u, b, m[:, start : start + B], R)
        assert _orthonormal(u), "swapped tracker must keep the basis orthonormal"
        assert b.shape == (u.shape[1], u.shape[1]), "core must stay square"
        assert torch.allclose(rot, u.mT @ u_old, atol=1e-5), "rot must be u_new^T u_old"
        assert u.shape[1] <= R
    # The rank the arm is BILLED for is the rank it tracks: a basis pinned at the seeding
    # block's width (min(R, B) = 5 here) was the Week-20 Oja arm's silent mis-billing.
    assert u.shape[1] == min(R, N), "a warmed-up tracker must fill its rank cap"


def test_fd_shrinkage_removes_energy_vs_incremental_svd() -> None:
    """On a stream whose rank exceeds the cap, FD's core carries strictly less
    Frobenius energy than the plain augmented step's (the shrinkage is the one
    algorithmic difference from fixed-rank incremental SVD)."""
    m = _stream(1, t=60, true_rank=12)  # true rank > cap -> a tail to shrink
    u_i, b_i, _ = augmented_bug_step(None, None, m[:, :B], R)
    u_f, b_f, _ = fd_step(None, None, m[:, :B], R)
    for start in range(B, m.shape[1] - B, B):
        blk = m[:, start : start + B]
        u_i, b_i, _ = augmented_bug_step(u_i, b_i, blk, R)
        u_f, b_f, _ = fd_step(u_f, b_f, blk, R)
    assert torch.linalg.norm(b_f) < torch.linalg.norm(b_i)


def test_oja_tracks_the_dominant_subspace() -> None:
    """After streaming a rank-4 signal, Oja's basis should capture it: the projection
    residual of a fresh sample from the same subspace is small (not a junk basis)."""
    m = _stream(2, t=200, true_rank=4)
    u, b, _ = OJA(None, None, m[:, :B], R)
    seen = B
    for start in range(B, m.shape[1] - B, B):
        u, b, _ = OJA(u, b, m[:, start : start + B], R, n_seen=seen)
        seen += B
    probe = m[:, -1:]
    resid = torch.linalg.norm(probe - u @ (u.mT @ probe)) / torch.linalg.norm(probe)
    # Re-measured at (eta0, decay) = (20.0, 0.03) on the grown basis: 7.7965e-02. The pin is
    # ~2x that, and a random basis of the same rank scores 9.5e-01 here, so it separates a
    # tracked subspace from a junk one by an order of magnitude. (The previous 0.25 was set
    # against the untuned defaults and a rank-pinned basis.)
    assert float(resid) < 0.15


# --------------------------------------------------------------- the cache


def _model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=N,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    m = LlamaForCausalLM(cfg)  # type: ignore[no-untyped-call]
    m.config._attn_implementation = "sdpa"
    m.eval()  # type: ignore[no-untyped-call]
    return m


def _cache(model: LlamaForCausalLM, **kw: Any) -> BugStreamingCache:
    return BugStreamingCache(
        model,
        rank=R,
        coord_budget=64,
        recent_window=4,
        absorb_block=4,
        n_sink=1,
        retention="lowrank_surprise",
        hh_budget=2,
        # The 48-token prefill must take MORE THAN ONE augmented step: every tracker keeps
        # the same ``u_aug @ u_loc[:, :k]`` on a seeding block, so what distinguishes FD --
        # the shrinkage, which only touches the core -- reaches the basis on the next step
        # or not at all. At the default 128 the prefill is one block and the "must change
        # the gist" pin below passes only while FD's seeding differs, which it no longer
        # does (it shares ``_augment``/``_svd_core`` with the incremental-SVD step).
        prefill_block_size=16,
        hh_select="surprise",
        **kw,
    )


def _prefill_then_decode(model: LlamaForCausalLM, cache: BugStreamingCache) -> torch.Tensor:
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (1, 48))
    with cache.attach(model):
        model(ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        pos = torch.tensor([[48]])
        out = model(ids[:, :1], past_key_values=cache, use_cache=True, position_ids=pos)
    logits: torch.Tensor = out.logits
    return logits


def test_isvd_tracker_is_bit_identical_to_the_default() -> None:
    m = _model()
    a = _prefill_then_decode(m, _cache(m))  # no knob = the archived path
    b = _prefill_then_decode(m, _cache(m, tracker="isvd"))
    assert torch.equal(a, b)


def test_the_bug_alias_is_deprecated_and_reaches_the_same_step() -> None:
    """The archived string still builds -- loudly, and on the same code path."""
    m = _model()
    a = _prefill_then_decode(m, _cache(m))
    with pytest.warns(DeprecationWarning, match="isvd"):
        cache = _cache(m, tracker="bug")
    assert cache._bug_layers()[0].tracker == "isvd"
    assert torch.equal(a, _prefill_then_decode(m, cache))


@pytest.mark.parametrize("trk", ["oja", "fd"])
def test_swapped_tracker_runs_end_to_end_and_differs(trk: str) -> None:
    m = _model()
    base = _prefill_then_decode(m, _cache(m))
    swapped = _prefill_then_decode(m, _cache(m, tracker=trk))
    assert swapped.shape == base.shape and torch.isfinite(swapped).all()
    assert not torch.equal(swapped, base), f"tracker={trk} must change the gist"


def test_invalid_tracker_fails_loud() -> None:
    m = _model()
    with pytest.raises(ValueError, match="tracker"):
        _cache(m, tracker="svd")


def test_the_swap_arm_configs_name_and_thread_the_tracker() -> None:
    """The three tracker arms are three configs, not three CLI flags: each names its own
    swapped arm and its ``tracker`` reaches the constructed layer."""
    from kvdlra.eval.config import load_arm
    from kvdlra.eval.frontier import build_arm

    m = _model()
    for cfg_name, legacy, trk in (
        ("isvd_r64_h256_seed", "bugSseed-r64-h256", "isvd"),
        ("oja_r64_h256_seed", "bugSseed-r64-h256-oja", "oja"),
        ("fd_r64_h256_seed", "bugSseed-r64-h256-fd", "fd"),
    ):
        arm = build_arm(load_arm(cfg_name), m, 64)
        assert arm["name"] == legacy and arm["kwargs"]["tracker"] == trk
        assert arm["make"]()._bug_layers()[0].tracker == trk
