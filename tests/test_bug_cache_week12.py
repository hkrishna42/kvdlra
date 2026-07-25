"""Week-12 H1 ablation: ``hh_retain=False`` select-and-discard SurpriseSLASH.

The r128 attribution question: bugS-r128 retrieves at 32K while the exact tier
never holds the queried codes (probe) and both plain BUG (r128) and a richer
gist (r256) retrieve nothing. Hypothesis H1 says the *withholding* -- diverting
high-surprise outlier columns away from the low-rank absorption step -- is what
keeps the gist recoverable, independent of the tier's contents being visible.

``hh_retain=False`` is the discriminating instrument: selection + withholding
stay bit-identical to the deployed bugS arm (the pool still lives in
``hh_k``/``hh_v``, feeds re-selection, and demotes into the tail), but the pool
is invisible to attention. It is an attribution ablation, NOT a cheaper
operating point: the pool is live stored state and ``stored_state_numel``
keeps counting it (see ``tests/test_accounting.py``).

Also pinned here: the ``hh_budget=0`` degeneration -- a bugslash config with no
tier never enters the SLASH path at all and is numerically plain BUG (the
reason the pre-registered hh=0 ablation could not discriminate H1).

Hermetic tiny Llama, mirroring ``tests/test_bug_cache_week11.py``.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer, _RopeAngles

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
    """Attach-free chunked ingest (the only path that populates the hh pool)."""
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


def _slash_cache(
    model: LlamaForCausalLM, *, hh_retain: bool, hh_budget: int = 8
) -> BugStreamingCache:
    return BugStreamingCache(
        model,
        rank=4,
        coord_budget=4096,
        recent_window=8,
        absorb_block=4,
        n_sink=4,
        retention="lowrank_surprise",
        hh_budget=hh_budget,
        hh_select="surprise",
        hh_retain=hh_retain,
    )


# ----------------------------------------------- discard: selected but invisible


def test_discard_tier_selected_but_invisible(tiny_model: LlamaForCausalLM) -> None:
    """Under ``hh_retain=False`` the needle is still *selected* into the pool and
    still *withheld* from absorption, but attention sees exactly the retain-mode
    K minus the tier."""
    depth, t = 40, 96
    ids = _needle_prompt(bg_id=5, needle_id=200, t=t, depth=depth)
    retain = _slash_cache(tiny_model, hh_retain=True)
    discard = _slash_cache(tiny_model, hh_retain=False)
    with torch.no_grad():
        _chunked_prefill(tiny_model, retain, ids, chunk=16)
        _chunked_prefill(tiny_model, discard, ids, chunk=16)
    dl, rl = _bug_layer(discard), _bug_layer(retain)
    assert dl.hh_pos is not None and depth in dl.hh_pos.tolist()  # selected
    mid = dl.mid_pos.tolist() if dl.mid_pos is not None else []
    assert depth not in mid  # withheld from absorption
    assert dl._hh_len() == rl._hh_len() > 0
    k_discard = dl._decode_peek()[0]
    k_retain = rl._decode_peek()[0]
    assert k_retain.shape[2] - k_discard.shape[2] == dl._hh_len()  # invisible


def test_discard_absorption_stream_matches_retain(tiny_model: LlamaForCausalLM) -> None:
    """The ablation contract, pinned at the LAYER level: given the SAME K/V
    stream, discard and retain produce bit-identical pool contents, basis and
    coordinates -- ``hh_retain`` changes attention visibility and nothing else.

    Deliberately not an end-to-end model comparison: through the model, hiding
    the tier changes what later layers/chunks attend to, so their K/V
    projections (and hence the two caches' states) legitimately diverge -- that
    feedback IS the ablation. The mechanics invariant only holds stream-in."""

    def make(hh_retain: bool) -> BugStreamingLayer:
        return BugStreamingLayer(
            rope=_RopeAngles(cast(nn.Module, tiny_model.model.rotary_emb)),
            rank=4,
            coord_budget=4096,
            recent_window=8,
            absorb_block=4,
            n_sink=2,
            retention="lowrank_surprise",
            hh_budget=8,
            hh_select="surprise",
            hh_retain=hh_retain,
        )

    retain, discard = make(True), make(False)
    g = torch.Generator().manual_seed(7)

    def step(t: int) -> None:
        k = torch.randn(1, H, t, D, generator=g)
        v = torch.randn(1, H, t, D, generator=g)
        retain.update(k.clone(), v.clone())
        discard.update(k.clone(), v.clone())

    step(18)  # prefill
    for _ in range(48):  # 12 absorbs: pool fills (2 absorbs) then demotes (10)
        step(1)
    assert retain._hh_len() == discard._hh_len() == 8  # pool at cap
    assert discard._f_len() > 0  # demoted columns were absorbed
    for name in ("u_k", "c_k", "u_v", "c_v", "mid_pos", "hh_k", "hh_v", "hh_pos"):
        r_t, d_t = getattr(retain, name), getattr(discard, name)
        assert r_t is not None and d_t is not None, name
        assert torch.equal(r_t, d_t), name
    # Same state, different visibility: the returned K differs by exactly the pool.
    k_r = retain._decode_peek()[0]
    k_d = discard._decode_peek()[0]
    assert k_r.shape[2] - k_d.shape[2] == retain._hh_len()


def test_discard_mask_sizes_track_decode(tiny_model: LlamaForCausalLM) -> None:
    """Decode-time mask consistency under discard: ``get_mask_sizes`` must equal
    the actually returned K length at every step, through BOTH pool phases --
    filling (visible mid grows by 0 per absorb) and full (grows by the demoted
    count). This is the shape-desync class the RULER decode loop would hit."""
    hh_budget = 24
    cache = _slash_cache(tiny_model, hh_retain=False, hh_budget=hh_budget)
    ids = torch.full((1, 32), 5, dtype=torch.long)
    with torch.no_grad():
        _chunked_prefill(tiny_model, cache, ids, chunk=16)
    layer = _bug_layer(cache)
    assert 0 < layer._hh_len() < hh_budget  # pool still filling at decode start
    saw_full = False
    pos = 32
    with torch.no_grad():
        for _ in range(24):
            pred_len, pred_off = layer.get_mask_sizes(1)
            tiny_model(
                torch.tensor([[7]]),
                past_key_values=cache,
                use_cache=True,
                position_ids=_pos(pos, 1),
            )
            pos += 1
            actual = int(layer._decode_peek()[0].shape[2])
            assert pred_len == actual
            assert pred_off == layer.cumulative_length - actual
            saw_full = saw_full or layer._hh_len() == hh_budget
    assert saw_full  # the demotion phase was actually exercised


# ----------------------------------------------- the hh=0 degeneration, pinned


def test_hh_zero_bugslash_equals_plain_bug(tiny_model: LlamaForCausalLM) -> None:
    """Regression documenting the Week-12 planning finding: ``hh_budget=0`` under
    ``retention='lowrank_surprise'`` never enters the SLASH path (``hh_enabled``
    is False) and, with the arm-style non-binding ``coord_budget``, is
    numerically plain BUG -- which is why the pre-registered hh=0 ablation could
    not discriminate H1. Bit-equal basis/coordinates; reconstructed K/V agree to
    fp tolerance (tracked-position vs contiguous RoPE paths, same math)."""
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, 256, (1, 96), generator=g)
    hh0 = _slash_cache(tiny_model, hh_retain=True, hh_budget=0)
    plain = BugStreamingCache(
        tiny_model,
        rank=4,
        coord_budget=4096,
        recent_window=8,
        absorb_block=4,
        n_sink=4,
        retention="fifo",
    )
    with torch.no_grad():
        _chunked_prefill(tiny_model, hh0, ids, chunk=16)
        _chunked_prefill(tiny_model, plain, ids, chunk=16)
    for h_layer, p_layer in zip(hh0.layers, plain.layers, strict=True):
        assert isinstance(h_layer, BugStreamingLayer)
        assert isinstance(p_layer, BugStreamingLayer)
        assert not h_layer.hh_enabled
        assert h_layer.u_k is not None and p_layer.u_k is not None
        assert torch.equal(h_layer.u_k, p_layer.u_k)
        assert h_layer.c_k is not None and p_layer.c_k is not None
        assert torch.equal(h_layer.c_k, p_layer.c_k)
        hk, hv = h_layer._decode_peek()
        pk, pv = p_layer._decode_peek()
        assert torch.allclose(hk, pk, atol=1e-5, rtol=1e-5)
        assert torch.allclose(hv, pv, atol=1e-5, rtol=1e-5)


# ----------------------------------------------- validation guards


def test_discard_requires_nonzero_hh_budget(tiny_model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError, match=r"hh_retain=False .* requires hh_budget >= 1"):
        BugStreamingCache(
            tiny_model,
            rank=4,
            coord_budget=32,
            retention="lowrank_surprise",
            hh_budget=0,
            hh_select="surprise",
            hh_retain=False,
        )


def test_discard_requires_surprise_select(tiny_model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError, match="hh_retain=False requires hh_select='surprise'"):
        BugStreamingCache(
            tiny_model,
            rank=4,
            coord_budget=32,
            retention="attn",
            hh_budget=4,
            hh_select="attn",
            hh_retain=False,
        )
