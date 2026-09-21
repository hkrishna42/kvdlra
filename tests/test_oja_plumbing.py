"""The Oja arm's schedule is configuration, not a library default, and the cache's tracker
string is ``isvd``.

The Week-20 swap ran ``oja_step`` at its own ``(1.0, 1e-3)`` defaults -- a schedule nobody
chose -- so the arm's cell is void. ``eta0``/``decay`` are now required keyword arguments
with no defaults to fall back on, the arm YAML carries the Week-2 validated ``(20.0,
0.03)``, and the cache threads them through. ``tracker="bug"`` still builds, with a
``DeprecationWarning``, so an archived call site keeps working while it is renamed.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import LlamaForCausalLM

from kvdlra.eval.config import arm_kwargs, load_arm
from kvdlra.tracker import TRACKERS  # the same dict object the cache dispatches through
from tests.conftest import tiny_cache
from tests.test_w20_tracker_swap import _prefill_then_decode


def test_config_schedule_reaches_oja_step(
    monkeypatch: pytest.MonkeyPatch, tiny_model: LlamaForCausalLM
) -> None:
    seen: list[tuple[float, float]] = []
    real = TRACKERS["oja"]

    def spy(u: Any, b: Any, block: Any, cap: int, *, n_seen: int, eta0: float, decay: float) -> Any:
        seen.append((eta0, decay))
        return real(u, b, block, cap, n_seen=n_seen, eta0=eta0, decay=decay)

    monkeypatch.setitem(TRACKERS, "oja", spy)
    kw = arm_kwargs(load_arm("oja_r64_h256_seed"), t=1024)
    assert kw["tracker"] == "oja" and kw["oja_eta0"] == 20.0 and kw["oja_decay"] == 0.03
    cache = tiny_cache(tiny_model, tracker="oja", oja_eta0=20.0, oja_decay=0.03)
    _prefill_then_decode(tiny_model, cache)
    assert seen and all(s == (20.0, 0.03) for s in seen)


def test_tuned_arm_carries_the_k_pre_schedule_from_the_recon_study() -> None:
    """``oja_r64_h256_seed_tuned`` (L1.4b) replaces the void Week-2 (20.0, 0.03) pair with
    the schedule ``kvdlra.eval.recon.tune_oja`` found on the 1B stored-representation study
    (``results/recon_1b/provenance.json``, ``oja_tuning.k_pre``)."""
    kw = arm_kwargs(load_arm("oja_r64_h256_seed_tuned"), t=1024)
    assert kw["tracker"] == "oja" and (kw["oja_eta0"], kw["oja_decay"]) == (5.0, 0.3)


def test_oja_n_seen_keeps_growing_after_the_tiers_saturate(
    monkeypatch: pytest.MonkeyPatch, tiny_model: LlamaForCausalLM
) -> None:
    """``t`` in ``eta0 / (1 + decay * t)`` must be TOKENS SEEN.

    It was derived from the coordinate tiers' current occupancy, which stops at the budget
    -- so on any context longer than the budget the decay froze and the arm ran the rest of
    the stream at a constant learning rate. Here the budget (8) saturates long before the
    200-token prefill ends.

    ``hh_budget`` + ``seed_hh_warmup`` (the shipped Oja arm's own heavy-hitter config) also
    puts the WHOLE single-shot prefill through the SLASH path (``cache.ingesting()`` makes
    ``_prefill``'s ``seed`` flag true for this one call): each ``prefill_block_size``
    sub-block re-scores the exact-tier candidate pool by surprise, and once the tracked
    basis is no longer empty a low-surprise OLD resident can be demoted after a more recent
    sub-block was demoted instead -- a real dip, measured at this rank/budget/seed, in the
    positions a demote batch carries. ``cumulative_length`` stays 0 for every one of these
    absorbs (it is only set once, after the whole prefill loop returns). That is the fix1 R2
    scenario: ``_tokens_seen`` must be a running maximum across absorbs, not just this call's
    own ``max(cumulative_length, positions.max() + 1)``.
    """
    budget, seen = 8, []
    real = TRACKERS["oja"]

    def spy(u: Any, b: Any, block: Any, cap: int, *, n_seen: int, **kw: float) -> Any:
        seen.append(n_seen)
        return real(u, b, block, cap, n_seen=n_seen, **kw)

    monkeypatch.setitem(TRACKERS, "oja", spy)
    cache = tiny_cache(
        tiny_model,
        coord_budget=budget,
        prefill_block_size=2,
        tracker="oja",
        rank=2,
        hh_budget=2,
        hh_select="surprise",
        seed_hh_warmup=True,
    )
    g = torch.Generator().manual_seed(0)
    with torch.no_grad(), cache.attach(tiny_model), cache.ingesting():
        tiny_model(torch.randint(0, 256, (1, 200), generator=g), past_key_values=cache)
    # A single-shot prefill runs each layer's whole chunk loop before the next layer's, so
    # the first `1 / len(_bug_layers())` share of the record is layer 0's (K and V per
    # absorb) -- not necessarily half (R4). Within a layer the frontier only moves forward
    # -- past what the saturated tiers can report, and past the demote path's older ones.
    layer0 = seen[: len(seen) // len(cache._bug_layers())]
    assert layer0 == sorted(layer0) and len(set(layer0)) > 4
    assert max(seen) > budget + cache._bug_layers()[0]._q_len()


def test_oja_step_has_no_default_schedule() -> None:
    from kvdlra.tracker.isvd import oja_step

    with pytest.raises(TypeError):
        oja_step(None, None, torch.randn(8, 4), 4)  # type: ignore[call-arg]


def test_bug_alias_warns_and_maps_to_isvd(tiny_model: LlamaForCausalLM) -> None:
    with pytest.warns(DeprecationWarning):
        cache = tiny_cache(tiny_model, tracker="bug")
    assert cache._bug_layers()[0].tracker == "isvd"
