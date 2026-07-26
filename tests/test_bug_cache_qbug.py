"""Week-12 Q-BUG: query-metric (whitened-key) low-rank gist (``w_key``).

The CPU probe (``scripts/w12_qbug_probe.py``) funded this: whitening the tracked
keys by a frozen per-feature diagonal ``L`` (= sqrt of calibrated query energy)
cuts post-RoPE attention error 30-44% at matched rank. These pins nail the
implementation contract:

* ``w_key = 1`` is **bit-identical** to plain bugS (the identity guard);
* whiten -> track -> un-whiten is **lossless at full rank** (a non-trivial ``L``
  with full rank + full budget still matches a ``DynamicCache``): the round trip
  does not corrupt reconstruction, only reallocates where a *truncated* rank
  spends its fidelity;
* the exact tier is **untouched**: a planted needle still lands verbatim in
  ``hh_pos`` under whitening (retrieval preserved -- the whole thesis);
* the frozen diagonal is **counted** (``stored_state_numel`` / ``bug_footprint``);
* the validation guards.

Hermetic tiny Llama, mirroring ``tests/test_bug_cache_week11.py``.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from kvdlra import accounting as acc
from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer

H, D = 2, 16
N_FEATURES = H * D


def _tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=H,
        head_dim=D,
        max_position_embeddings=4096,
    )


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    model = LlamaForCausalLM(_tiny_config())  # type: ignore[no-untyped-call]
    model.config._attn_implementation = "sdpa"
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _pos(start: int, n: int) -> torch.Tensor:
    return torch.arange(start, start + n).unsqueeze(0)


def _bug_layer(cache: BugStreamingCache) -> BugStreamingLayer:
    return next(layer for layer in cache.layers if isinstance(layer, BugStreamingLayer))


def _chunked_prefill(
    model: LlamaForCausalLM, cache: BugStreamingCache, ids: torch.Tensor, chunk: int
) -> None:
    t = int(ids.shape[1])
    with cache.ingesting():
        for start in range(0, t, chunk):
            stop = min(t, start + chunk)
            model(
                ids[:, start:stop],
                past_key_values=cache,
                use_cache=True,
                position_ids=_pos(start, stop - start),
            )
            cache.consolidate()


def _needle_prompt(bg_id: int, needle_id: int, t: int, depth: int) -> torch.Tensor:
    ids = torch.full((1, t), bg_id, dtype=torch.long)
    ids[0, depth] = needle_id
    return ids


def _wkey(seed: int, lo: float = 0.5, hi: float = 2.0) -> torch.Tensor:
    """A non-trivial, bounded, strictly-positive per-feature diagonal."""
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(N_FEATURES, generator=g)


# ----------------------------------------------------- identity guard


def test_wkey_ones_is_bit_identical_to_plain(tiny_model: LlamaForCausalLM) -> None:
    """``w_key = 1`` must reproduce plain bugS exactly (basis, coords, tier)."""
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 256, (1, 96), generator=g)
    kw = {
        "rank": 4,
        "coord_budget": 4096,
        "recent_window": 8,
        "absorb_block": 4,
        "n_sink": 4,
        "retention": "lowrank_surprise",
        "hh_budget": 6,
        "hh_select": "surprise",
    }
    plain = BugStreamingCache(tiny_model, **kw)  # type: ignore[arg-type]
    ones = BugStreamingCache(tiny_model, w_key=torch.ones(N_FEATURES), **kw)  # type: ignore[arg-type]
    with torch.no_grad():
        _chunked_prefill(tiny_model, plain, ids, chunk=16)
        _chunked_prefill(tiny_model, ones, ids, chunk=16)
    pl, ol = _bug_layer(plain), _bug_layer(ones)
    for name in ("u_k", "c_k", "u_v", "c_v", "mid_pos", "hh_pos"):
        a, b = getattr(pl, name), getattr(ol, name)
        assert (a is None) == (b is None), name
        if a is not None:
            assert torch.equal(a, b), name
    kp, _ = pl._decode_peek()
    ko, _ = ol._decode_peek()
    assert torch.equal(kp, ko)


# ----------------------------------------------------- lossless at full rank


def test_wkey_lossless_at_full_rank(tiny_model: LlamaForCausalLM) -> None:
    """Whiten -> track -> un-whiten is exact at full rank + full budget: a
    non-trivial ``w_key`` still reconstructs the true past (matches DynamicCache).
    The round trip cannot corrupt reconstruction; it only reallocates a
    *truncated* rank's fidelity."""
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 256, (1, 64), generator=g)
    win = torch.randint(0, 256, (1, 16), generator=torch.Generator().manual_seed(9))
    ref = DynamicCache()
    with torch.no_grad():
        tiny_model(ids, past_key_values=ref, use_cache=True)
        ref_out = tiny_model(win, past_key_values=ref, use_cache=True, position_ids=_pos(64, 16))
    bug = BugStreamingCache(
        tiny_model,
        rank=N_FEATURES,
        coord_budget=4096,
        recent_window=8,
        absorb_block=4,
        retention="lowrank_surprise",
        hh_budget=8,
        hh_select="surprise",
        w_key=_wkey(3),
    )
    with torch.no_grad():
        _chunked_prefill(tiny_model, bug, ids, chunk=16)
        with bug.frozen_scoring():
            out = tiny_model(win, past_key_values=bug, use_cache=True, position_ids=_pos(64, 16))
    assert torch.allclose(out.logits, cast(torch.Tensor, ref_out.logits), atol=2e-3, rtol=2e-3)


# ----------------------------------------------------- retrieval preserved


def test_wkey_preserves_needle_in_exact_tier(tiny_model: LlamaForCausalLM) -> None:
    """A planted needle still lands verbatim in the exact tier under whitening --
    the tier is stored raw (post-RoPE), and a bounded diagonal L does not knock a
    genuine high-residual outlier out of the surprise selection."""
    depth, t = 40, 96
    ids = _needle_prompt(bg_id=5, needle_id=200, t=t, depth=depth)
    cache = BugStreamingCache(
        tiny_model,
        rank=4,
        coord_budget=128,
        recent_window=8,
        absorb_block=4,
        retention="lowrank_surprise",
        hh_budget=1,
        hh_select="surprise",
        w_key=_wkey(7),
    )
    with torch.no_grad():
        _chunked_prefill(tiny_model, cache, ids, chunk=16)
    layer = _bug_layer(cache)
    assert layer.hh_pos is not None and depth in layer.hh_pos.tolist()  # verbatim
    mid = layer.mid_pos.tolist() if layer.mid_pos is not None else []
    assert depth not in mid  # withheld from the low-rank tail


# ----------------------------------------------------- honest accounting


def test_wkey_counted_in_footprint(tiny_model: LlamaForCausalLM) -> None:
    """``bug_footprint(..., w_key=True)`` equals the measured ``stored_state_numel``
    and is exactly ``n`` above the un-whitened footprint."""
    wk = _wkey(5)
    cache = BugStreamingCache(
        tiny_model,
        rank=8,
        coord_budget=24,
        recent_window=8,
        absorb_block=4,
        n_sink=4,
        retention="lowrank_surprise",
        hh_budget=6,
        hh_select="surprise",
        w_key=wk,
    )
    # Drive to a saturated steady state.
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 256, (1, 120), generator=g)
    with torch.no_grad():
        _chunked_prefill(tiny_model, cache, ids, chunk=16)
    layer = _bug_layer(cache)
    coord_count = layer._f_len() + layer._q_len()
    common = {
        "rank": 8,
        "coord_count": coord_count,
        "recent_len": layer._recent_len(),
        "n_sink": 4,
        "retention": "lowrank_surprise",
        "hh_count": layer._hh_len(),
        "hh_select": "surprise",
        "u_present": layer.u_k is not None,
    }
    fp = acc.bug_footprint(N_FEATURES, w_key=True, **common)  # type: ignore[arg-type]
    fp_no = acc.bug_footprint(N_FEATURES, w_key=False, **common)  # type: ignore[arg-type]
    assert fp.float_equiv() == layer.stored_state_numel()
    assert fp.float_equiv() - fp_no.float_equiv() == N_FEATURES


# ----------------------------------------------------- validation guards


def test_wkey_must_be_1d(tiny_model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError, match="w_key must be 1-D"):
        BugStreamingCache(tiny_model, rank=4, coord_budget=32, w_key=torch.ones(N_FEATURES, 1))


def test_wkey_must_be_positive(tiny_model: LlamaForCausalLM) -> None:
    bad = torch.ones(N_FEATURES)
    bad[0] = 0.0
    with pytest.raises(ValueError, match="w_key must be strictly positive"):
        BugStreamingCache(tiny_model, rank=4, coord_budget=32, w_key=bad)


def test_wkey_wrong_length_raises_at_prefill(tiny_model: LlamaForCausalLM) -> None:
    cache = BugStreamingCache(tiny_model, rank=4, coord_budget=32, w_key=torch.ones(N_FEATURES + 3))
    ids = torch.randint(0, 256, (1, 32))
    with pytest.raises(ValueError, match=r"w_key shape .* != \(n_features"), torch.no_grad():
        _chunked_prefill(tiny_model, cache, ids, chunk=16)


def test_wkey_list_length_must_match_layers(tiny_model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError, match=r"w_key list has .* != .* layers"):
        BugStreamingCache(tiny_model, rank=4, coord_budget=32, w_key=[torch.ones(N_FEATURES)])
