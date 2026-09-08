"""Week-20 tracker-swap ablation: the gist tracker is pluggable behind
``BugStreamingCache(tracker=...)`` with the BUG step's exact ``(u, b, rot)`` contract.

Pins: (1) ``oja_step``/``fd_step`` honor the contract (orthonormal basis, ``rot =
u_new^T u_old``, square core); (2) FD's shrinkage removes energy relative to the
plain augmented (incremental-SVD) step; (3) ``tracker="bug"`` is bit-identical to the
pre-knob cache; (4) the swapped trackers run end-to-end through prefill + decode; (5)
``build_arms`` names the swapped arms and threads the knob.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.integrators.streaming_torch import augmented_bug_step, fd_step, oja_step

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


@pytest.mark.parametrize("step", [oja_step, fd_step])
def test_swapped_trackers_honor_the_step_contract(step: Any) -> None:
    m = _stream(0)
    u, b, rot = step(None, None, m[:, :B], R)  # seeding = reduced QR, like BUG
    assert u.shape == (N, min(R, B)) and _orthonormal(u)
    assert b.shape == (u.shape[1], u.shape[1]) and rot.shape == (u.shape[1], 0)
    for start in range(B, m.shape[1] - B, B):
        u_old = u
        u, b, rot = step(u, b, m[:, start : start + B], R)
        assert _orthonormal(u), "swapped tracker must keep the basis orthonormal"
        assert b.shape == (u.shape[1], u.shape[1]), "core must stay square"
        assert torch.allclose(rot, u.mT @ u_old, atol=1e-5), "rot must be u_new^T u_old"
        assert u.shape[1] <= R


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
    u, b, _ = oja_step(None, None, m[:, :B], R)
    seen = B
    for start in range(B, m.shape[1] - B, B):
        u, b, _ = oja_step(u, b, m[:, start : start + B], R, n_seen=seen)
        seen += B
    probe = m[:, -1:]
    resid = torch.linalg.norm(probe - u @ (u.mT @ probe)) / torch.linalg.norm(probe)
    assert float(resid) < 0.25


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


def test_bug_tracker_is_bit_identical_to_the_default() -> None:
    m = _model()
    a = _prefill_then_decode(m, _cache(m))  # no knob = the archived path
    b = _prefill_then_decode(m, _cache(m, tracker="bug"))
    assert torch.equal(a, b)


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


def test_build_arms_names_and_threads_the_tracker() -> None:
    from w10_frontier import build_arms, build_parser

    m = _model()
    ns = build_parser().parse_args([])
    ns.methods, ns.ranks, ns.hh_budgets, ns.chunk = ["bugslash"], [R], [2], 16
    ns.warmup_seed = True
    for trk, suf in (("bug", ""), ("oja", "-oja"), ("fd", "-fd")):
        ns.tracker = trk
        arms = [a for a in build_arms(ns, m, 64) if a["kind"] == "bug"]
        assert arms and all(a["name"].endswith(suf) or trk == "bug" for a in arms)
        layer = arms[0]["make"]()._bug_layers()[0]
        assert layer.tracker == trk


def test_parser_exposes_tracker_flag() -> None:
    from w10_frontier import build_parser

    ns = build_parser().parse_args(["--tracker", "fd"])
    assert ns.tracker == "fd"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--tracker", "svd"])


def _ns(**kw: Any) -> argparse.Namespace:  # kept for symmetry with sibling tests
    return argparse.Namespace(**kw)
