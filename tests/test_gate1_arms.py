"""Gate-1 controls: is the ONLINE tracker load-bearing, or would a fixed basis do?

Three controls hold everything but the gist fixed at the r64 operating point.
``tracker="frozen"`` runs the incremental-SVD step over the first ``freeze_after``
tokens and then stops updating (the xKV/ShadowKV-style learn-then-freeze arm);
``tracker="random"`` never tracks at all (one seeded orthonormal draw -- the floor); the
``nogist_h*`` arms delete the gist and spend its bytes on the exact tier instead. There
are two of those because the tier's bytes scale with the layer width ``n`` while the
coordinate bytes do not, so no single ``H`` matches both a 1024-wide layer
(Llama-3.1-8B, Mistral-7B) and a 512-wide one (Qwen2.5-7B).

Pins: the two steps' contracts, both cache dispatch branches (the frozen arm's monotone
frontier and the random arm's per-layer/per-stream seed), the byte match itself, and
that a nogist arm really runs as tier + ring.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor
from transformers import LlamaForCausalLM

from kvdlra import accounting as acc
from kvdlra.cache import BugStreamingCache
from kvdlra.eval.config import ROOT, arm_kwargs, load_arm
from kvdlra.eval.frontier import build_arm
from kvdlra.tracker import TRACKERS
from kvdlra.tracker.isvd import frozen_step, isvd_step, random_step
from tests.conftest import N_FEATURES, tiny_cache

T16K = 16384
# Each nogist arm and the layer width ``n = num_kv_heads * head_dim`` it is sized for.
NOGIST = {"nogist_h2423": 1024, "nogist_h4460": 512}


# --------------------------------------------------------------- the two steps


def test_frozen_matches_isvd_before_freeze_and_stops_after() -> None:
    """Under ``freeze_after`` the step IS the incremental-SVD step; past it the basis and
    core come back untouched behind an identity rotation, so the cache's coordinate carry
    is a no-op and later tokens are plain projections onto the warm-up basis."""
    g = torch.Generator().manual_seed(0)
    m = torch.randn(64, 20 * 16, generator=g)
    ua: Tensor | None = None
    ba: Tensor | None = None
    uf: Tensor | None = None
    bf: Tensor | None = None
    frozen: Tensor | None = None
    seen = 0
    for s in range(0, m.shape[1], 16):
        blk = m[:, s : s + 16]
        ua, ba, _ = isvd_step(ua, ba, blk, 8)
        uf, bf, rot = frozen_step(uf, bf, blk, 8, n_seen=seen, freeze_after=160)
        seen += 16
        if frozen is None:  # set below at seen == 160, i.e. on the last tracking block
            assert torch.equal(ua, uf), "before the freeze the two are the same step"
        else:
            assert torch.equal(rot, torch.eye(8)) and torch.equal(uf, frozen)
        if seen == 160:
            frozen = uf.clone()
    assert ua is not None and frozen is not None
    assert not torch.equal(ua, frozen), "...and the incremental-SVD basis kept moving"


def test_random_basis_is_orthonormal_fixed_and_seeded() -> None:
    u1, b1, _ = random_step(None, None, torch.randn(64, 16), 8, seed=3)
    u2, _, rot = random_step(u1, b1, torch.randn(64, 16), 8, seed=3)
    u3, _, _ = random_step(None, None, torch.randn(64, 16), 8, seed=3)
    u4, _, _ = random_step(None, None, torch.randn(64, 16), 8, seed=4)
    assert torch.allclose(u1.mT @ u1, torch.eye(8), atol=1e-5) and u1.shape == (64, 8)
    assert b1.shape == (8, 8), "the core stays square, as every other step's does"
    assert torch.equal(u1, u2) and torch.equal(u1, u3), "fixed for the life of the stream"
    assert not torch.equal(u1, u4), "and a different seed is a different draw"
    assert torch.equal(rot, torch.eye(8))
    assert set(TRACKERS) >= {"isvd", "oja", "fd", "frozen", "random"}


# --------------------------------------------------------------- the dispatch


def _prefill(model: LlamaForCausalLM, cache: BugStreamingCache, t: int = 160) -> None:
    g = torch.Generator().manual_seed(0)
    with torch.no_grad(), cache.attach(model):
        model(torch.randint(0, 256, (1, t), generator=g), past_key_values=cache)


def _cache(model: LlamaForCausalLM, **kw: Any) -> BugStreamingCache:
    return tiny_cache(model, rank=4, coord_budget=32, prefill_block_size=32, **kw)


def test_the_frozen_arm_freezes_at_the_configured_frontier(
    monkeypatch: pytest.MonkeyPatch, tiny_model: LlamaForCausalLM
) -> None:
    """``freeze_after`` reaches the step from the constructor, and the frontier it is
    compared against is ``_tokens_seen`` -- the monotone one, which keeps growing after
    the tiers saturate (a frontier read off tier occupancy would stop and never freeze).
    ``out is u`` is the freeze itself: the step hands the same basis object back."""
    calls: list[tuple[int, int, bool]] = []
    real = TRACKERS["frozen"]

    def spy(u: Any, b: Any, blk: Any, cap: int, *, n_seen: int, **kw: Any) -> Any:
        out = real(u, b, blk, cap, n_seen=n_seen, **kw)
        freeze_after = int(kw["freeze_after"])
        calls.append((n_seen, freeze_after, out[0] is u))
        return out

    monkeypatch.setitem(TRACKERS, "frozen", spy)
    cache = _cache(tiny_model, tracker="frozen", freeze_after=100)
    _prefill(tiny_model, cache)
    assert {f for _, f, _ in calls} == {100}, "the arm's freeze_after, not the default"
    # A prefill runs each layer's whole block loop before the next layer's, so the record
    # is one layer's sequence after another's; the frontier is per layer.
    layer0 = [n for n, _, _ in calls][: len(calls) // len(cache._bug_layers())]
    assert layer0 == sorted(layer0), "a layer's frontier only moves forward"
    assert any(n < 100 and not held for n, _, held in calls), "it tracks before the freeze"
    assert all(held for n, _, held in calls if n >= 100), "and holds the basis after it"
    assert max(n for n, _, _ in calls) > 100, "the stream really crossed the frontier"


def test_the_random_arm_draws_one_basis_per_layer_and_stream(
    tiny_model: LlamaForCausalLM,
) -> None:
    """``basis_seed + 2*layer_idx (+1 for V)``: reproducible from the arm config alone, a
    different draw per layer and per stream, and -- because each stored basis still equals
    its seeding draw after the whole prefill -- never updated."""
    cache = _cache(tiny_model, tracker="random", basis_seed=7)
    _prefill(tiny_model, cache)
    probe = torch.zeros(N_FEATURES, 1)  # only its shape/dtype/device reach the draw
    seen = []
    for i, layer in enumerate(cache._bug_layers()):
        for stored, off in ((layer.u_k, 0), (layer.u_v, 1)):
            want, _, _ = random_step(None, None, probe, 4, seed=7 + 2 * i + off)
            assert stored is not None and torch.equal(stored, want)
            seen.append(want)
    assert not any(torch.equal(a, b) for i, a in enumerate(seen) for b in seen[i + 1 :])


def test_the_gate1_arm_configs_name_and_thread_their_knobs(tiny_model: LlamaForCausalLM) -> None:
    """A ``cache:`` block IS the constructor call, so every key in one has to be a keyword
    the cache takes and has to land on the layer -- a knob misspelled in a YAML passes
    every other test in this file and fails on the pod."""
    for stem, trk in (
        ("frozen_r64_h256_seed", "frozen"),
        ("random_r64_h256_seed", "random"),
        ("nogist_h2423", "isvd"),
        ("nogist_h4460", "isvd"),
    ):
        arm = build_arm(load_arm(stem), tiny_model, T16K)
        layer = arm["make"]()._bug_layers()[0]
        assert arm["name"] == stem and arm["kwargs"]["tracker"] == trk
        assert layer.tracker == trk and layer.rank == arm["kwargs"]["rank"]
        if trk == "frozen":
            assert layer.freeze_after == 4096
        if trk == "random":
            assert layer.basis_seed == 0


# --------------------------------------------------------------- the nogist arms


def test_the_nogist_arms_are_byte_matched_to_isvd_r64_at_16k() -> None:
    """Each arm's ``H`` is what makes its STORED bits (the at-rest billing) equal the r64
    arm's at 16K, so a Gate-1 cell compares mechanisms and not budgets. Two arms, one per
    layer width; the file name carries the ``H`` its ``doc:`` derives."""
    assert {p.stem for p in (ROOT / "arms").glob("nogist_h*.yaml")} == set(NOGIST)
    for stem, n in NOGIST.items():
        kw = arm_kwargs(load_arm(stem), T16K)
        assert kw["rank"] == 1 and kw["coord_budget"] == 1, f"{stem}: tier + ring only"
        ref = acc.bug_footprint(
            n, rank=64, coord_count=T16K - 4 - 32 - 256, recent_len=32,
            retention="lowrank_surprise", hh_count=256,
        )  # fmt: skip
        nogist = acc.bug_footprint(
            n, rank=1, coord_count=1, recent_len=32,
            retention="lowrank_surprise", hh_count=kw["hh_budget"],
        )  # fmt: skip
        assert abs(nogist.stored_bits() / ref.stored_bits() - 1.0) < 0.05, stem


def test_a_nogist_arm_runs_as_tier_plus_ring(tiny_model: LlamaForCausalLM) -> None:
    """The arm as configured, on a stream longer than its tier: the exact tier fills to
    ``H`` and the coordinate tier holds its one column -- a rank-1 gist is the residual the
    tier selects against (so selection is ~norm-ordered), not a reconstruction of the
    middle. ``H`` is 64 here because the stream is 512 tokens, not 16K."""
    from kvdlra.eval.frontier import _prefill_chunked  # the prefill the pods run

    kw = {**arm_kwargs(load_arm("nogist_h2423"), T16K), "hh_budget": 64}
    cache = BugStreamingCache(tiny_model, **kw)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad(), cache.attach(tiny_model):
        _prefill_chunked(tiny_model, cache, torch.randint(0, 256, (1, 512), generator=g), 128)
    for layer in cache._bug_layers():
        assert layer.hh_pos is not None and int(layer.hh_pos.shape[0]) == 64
        assert layer.c_k is not None and layer.c_k.shape == (1, 1)


def test_vt_is_excluded_from_the_gate1_families_by_default() -> None:
    """Amendment 1b (D-018) sets the shipped ``EXCLUDED_TASKS`` default to ``{"vt"}``: the
    pre-flight ``full`` ceiling on generator v2's ``vt`` was 9/12 = 0.75 < 0.9, a generator
    finding attributed to ``kvdlra.eval.gen`` (docs/plan/reports/vt-template-comparison.md),
    not the checkpoint. The behaviour -- primary retrieval 12, secondary 18, perplexity 4,
    bf16 6 -- is tested in ``test_gate1_table``/``test_gate1_bf16``, both of which pin the
    constant empty for their full-family mechanics; this is the only test that reads the
    shipped default, so a revert of the module constant fails here."""
    from kvdlra.eval import gate1

    assert frozenset({"vt"}) == gate1.EXCLUDED_TASKS
