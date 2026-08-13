"""Week-13/14 Track-B warm-up seed (``seed_hh_warmup``): the regression pins.

The funded lever seeds the exact heavy-hitter tier from the *first* ingest chunk:
when ``seed_hh_warmup=True`` the first chunk's middle is SLASH-routed through
``_absorb_block_slash`` in sub-blocks (scored against a strictly-older, needle-free
basis), so an early-planted outlier enters ``hh_k``/``hh_v``/``hh_pos`` instead of
being bypassed into the low-rank tail (the ~4-5K warm-up window). It ships
**default-off** and is guarded against the coded/quant/merge tiers.

``tests/test_bug_cache_week11.py`` already pins the *mechanism* (capture at rank 4/8,
disjointness, span, losslessness). This file is the **regression contract** for
turning the knob on: a self-contained on/off proof-of-life, the bit-for-bit identity
of the off path, the full coded/quant/merge guard, honest non-interference with the
other arm families, and the accounting neutrality (identical ``stored_state_numel``).

Hermetic tiny Llama, mirroring ``tests/test_bug_cache_qbug.py`` /
``tests/test_accounting.py``.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra import accounting as acc
from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer
from kvdlra.quant import ProductQuantizer

H, D = 2, 16
N_FEATURES = H * D

# Every stored per-layer tensor the seed could touch (compared bit-for-bit).
_STORED = (
    "sink_k",
    "sink_v",
    "recent_k",
    "recent_v",
    "u_k",
    "c_k",
    "u_v",
    "c_v",
    "b_k",
    "b_v",
    "hh_k",
    "hh_v",
    "hh_pos",
    "hh_score",
    "mid_pos",
    "mid_surprise",
    "ring_score",
)


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
    """A homogeneous background with one distinct high-residual needle at ``depth``."""
    ids = torch.full((1, t), bg_id, dtype=torch.long)
    ids[0, depth] = needle_id
    return ids


def _slash(**over: object) -> dict[str, object]:
    """The deployed ``bugS`` (bugslash) shape: low-rank gist + a surprise-selected
    exact tier, with a small ``prefill_block_size`` so the first-chunk basis warms
    across sub-blocks before the needle (the seed's operating regime)."""
    kw: dict[str, object] = {
        "rank": 4,
        "coord_budget": 128,
        "recent_window": 8,
        "absorb_block": 4,
        "n_sink": 4,
        "prefill_block_size": 8,
        "retention": "lowrank_surprise",
        "hh_budget": 1,
        "hh_select": "surprise",
    }
    kw.update(over)
    return kw


def _assert_state_identical(a: BugStreamingLayer, b: BugStreamingLayer) -> None:
    for name in _STORED:
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            assert torch.equal(x, y), name


# ------------------------------------------- 1. positive capture (proof-of-life)


def test_seed_captures_first_chunk_needle_off_bypasses(tiny_model: LlamaForCausalLM) -> None:
    """Phase-0 reproduce: a distinctive outlier needle in the FIRST ingest chunk lands
    verbatim in the exact tier with ``seed_hh_warmup=True`` and is bypassed to the
    low-rank tail with it off -- the warm-up-window fix as a clean matched A/B."""
    depth, t = 20, 96  # needle in the first (chunk=32) block, past sub-block 1
    ids = _needle_prompt(bg_id=5, needle_id=200, t=t, depth=depth)
    on = BugStreamingCache(tiny_model, seed_hh_warmup=True, **_slash())  # type: ignore[arg-type]
    off = BugStreamingCache(tiny_model, seed_hh_warmup=False, **_slash())  # type: ignore[arg-type]
    with torch.no_grad():
        _chunked_prefill(tiny_model, on, ids, chunk=32)
        _chunked_prefill(tiny_model, off, ids, chunk=32)
    lon, loff = _bug_layer(on), _bug_layer(off)
    # seed ON: the needle is promoted verbatim into the exact tier, disjoint from tail.
    assert lon.hh_pos is not None and depth in lon.hh_pos.tolist()
    assert lon.mid_pos is None or depth not in lon.mid_pos.tolist()
    # seed OFF: the first-chunk needle bypasses the exact tier (absorbed straight into
    # the tail during _prefill); later chunks' steady-state SLASH fills hh with OTHER
    # tokens but can never reach back to promote the tail-resident needle.
    assert loff.hh_pos is None or depth not in loff.hh_pos.tolist()
    assert loff.mid_pos is not None and depth in loff.mid_pos.tolist()


# ------------------------------------------- 2. identity: off is a true no-op


def test_seed_off_is_bit_identical_to_baseline(tiny_model: LlamaForCausalLM) -> None:
    """``seed_hh_warmup=False`` is bit-for-bit identical -- every stored tensor AND the
    forward logits on a fixed window -- to the deployed ``bugS`` arm built with the flag
    omitted. Also an anti-drift guard on the default: flipping it to True fails here."""
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 256, (1, 96), generator=g)
    win = torch.randint(0, 256, (1, 12), generator=torch.Generator().manual_seed(9))
    base = BugStreamingCache(tiny_model, **_slash())  # type: ignore[arg-type]  # flag omitted
    off = BugStreamingCache(tiny_model, seed_hh_warmup=False, **_slash())  # type: ignore[arg-type]
    with torch.no_grad():
        _chunked_prefill(tiny_model, base, ids, chunk=16)
        _chunked_prefill(tiny_model, off, ids, chunk=16)
    _assert_state_identical(_bug_layer(base), _bug_layer(off))
    with torch.no_grad():
        with base.frozen_scoring():
            ob = tiny_model(win, past_key_values=base, use_cache=True, position_ids=_pos(96, 12))
        with off.frozen_scoring():
            oo = tiny_model(win, past_key_values=off, use_cache=True, position_ids=_pos(96, 12))
    assert torch.equal(cast(torch.Tensor, ob.logits), cast(torch.Tensor, oo.logits))


# ------------------------------------------- 3. guard: coded / quant / merge


@pytest.mark.parametrize(
    "extra",
    [
        {
            "coord_codebook": ProductQuantizer(dim=4, bits=4, subspaces=2, seed=1),
            "anchor_rank": 4,
            "code_budget": 8,
        },
        {"quant_bits": 4, "quant_budget": 16},
        {"merge": True},
    ],
    ids=["coord_codebook", "quant_budget", "merge"],
)
def test_seed_rejects_coded_quant_merge(
    tiny_model: LlamaForCausalLM, extra: dict[str, object]
) -> None:
    """The seed reasons over the fp32 low-rank tail only, so it is rejected (documented
    message) with a coded/quant second tier or a merged (non-unique-position) tail --
    the combinations that could double-count or mis-evict a promoted column."""
    with pytest.raises(ValueError, match="fp32 low-rank tail only"):
        BugStreamingCache(
            tiny_model,
            rank=4,
            coord_budget=64,
            recent_window=8,
            absorb_block=4,
            n_sink=4,
            retention="lowrank_surprise",
            hh_budget=2,
            hh_select="surprise",
            seed_hh_warmup=True,
            **extra,  # type: ignore[arg-type]
        )


# ------------------------------------------- 4. non-interference (honest)


def test_seed_is_noop_on_non_hh_arm(tiny_model: LlamaForCausalLM) -> None:
    """Strongest bit-for-bit non-interference: on an arm with NO exact tier
    (``hh_budget=0`` -> ``hh_enabled`` False, e.g. plain BUG), the ``hh_enabled`` gate
    in ``_prefill`` makes ``seed_hh_warmup=True`` a true no-op -- identical stored
    state to seed off."""
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 256, (1, 96), generator=g)
    kw: dict[str, object] = {
        "rank": 4,
        "coord_budget": 128,
        "recent_window": 8,
        "absorb_block": 4,
        "n_sink": 4,
        "prefill_block_size": 8,
        "retention": "fifo",
        "hh_budget": 0,
    }
    on = BugStreamingCache(tiny_model, seed_hh_warmup=True, **kw)  # type: ignore[arg-type]
    off = BugStreamingCache(tiny_model, seed_hh_warmup=False, **kw)  # type: ignore[arg-type]
    assert not _bug_layer(on).hh_enabled  # no exact tier -> the seed gate can never fire
    with torch.no_grad():
        _chunked_prefill(tiny_model, on, ids, chunk=16)
        _chunked_prefill(tiny_model, off, ids, chunk=16)
    _assert_state_identical(_bug_layer(on), _bug_layer(off))


def test_seed_is_active_on_bugevict_not_gate_inert(tiny_model: LlamaForCausalLM) -> None:
    """Honest scoping: ``bugevict`` (rank-1 gist + a surprise exact tier) is itself a
    SLASH-family arm (``hh_enabled`` True), so -- unlike a non-hh arm -- the seed is NOT
    gate-inert on it. Turning it on DOES seed the first-chunk needle that the off arm
    bypasses. bugevict is thus protected by the default-off policy, not by the gate,
    which is exactly why a blanket default flip would need its own scoping."""
    depth, t = 20, 96
    ids = _needle_prompt(bg_id=5, needle_id=200, t=t, depth=depth)
    evict: dict[str, object] = {
        "rank": 1,
        "coord_budget": 1,
        "recent_window": 8,
        "absorb_block": 4,
        "n_sink": 4,
        "prefill_block_size": 8,
        "retention": "lowrank_surprise",
        "hh_budget": 2,
        "hh_select": "surprise",
    }
    on = BugStreamingCache(tiny_model, seed_hh_warmup=True, **evict)  # type: ignore[arg-type]
    off = BugStreamingCache(tiny_model, seed_hh_warmup=False, **evict)  # type: ignore[arg-type]
    assert _bug_layer(on).hh_enabled  # bugevict IS a SLASH-family arm (not gate-inert)
    with torch.no_grad():
        _chunked_prefill(tiny_model, on, ids, chunk=32)
        _chunked_prefill(tiny_model, off, ids, chunk=32)
    lon, loff = _bug_layer(on), _bug_layer(off)
    assert loff.hh_pos is None or depth not in loff.hh_pos.tolist()  # off: needle bypassed
    assert lon.hh_pos is not None and depth in lon.hh_pos.tolist()  # on: needle seeded


# ------------------------------------------- 5. accounting neutrality


def test_seed_footprint_identical_on_off(tiny_model: LlamaForCausalLM) -> None:
    """At a saturated steady state the seed only reallocates WHICH tokens are verbatim
    (hh) vs low-rank (tail); both tiers stay capped, so ``stored_state_numel`` is
    identical seed-on vs seed-off (Week-13's unchanged tok_eq/layer), and the
    formula-path ``bug_footprint`` still matches (no new uncounted stored tensor)."""
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 256, (1, 400), generator=g)  # well past saturation
    kw = _slash(coord_budget=24, hh_budget=6)
    on = BugStreamingCache(tiny_model, seed_hh_warmup=True, **kw)  # type: ignore[arg-type]
    off = BugStreamingCache(tiny_model, seed_hh_warmup=False, **kw)  # type: ignore[arg-type]
    with torch.no_grad():
        _chunked_prefill(tiny_model, on, ids, chunk=16)
        _chunked_prefill(tiny_model, off, ids, chunk=16)
    lon, loff = _bug_layer(on), _bug_layer(off)
    # Both tiers saturate to their budgets in BOTH arms -> identical counts.
    assert lon._hh_len() == loff._hh_len() == 6
    assert lon._f_len() == loff._f_len()
    assert lon._recent_len() == loff._recent_len()
    assert lon.stored_state_numel() == loff.stored_state_numel()
    assert on.stored_state_numel() == off.stored_state_numel()  # cache-level total
    # Anti-drift: the formula path reproduces the seeded cache's measured footprint.
    coord_count = lon._f_len() + lon._q_len()
    fp = acc.bug_footprint(
        N_FEATURES,
        rank=4,
        coord_count=coord_count,
        recent_len=lon._recent_len(),
        n_sink=4,
        retention="lowrank_surprise",
        hh_count=lon._hh_len(),
        hh_select="surprise",
        u_present=lon.u_k is not None,
    )
    assert fp.float_equiv() == lon.stored_state_numel()
