"""Week-11 SurpriseSLASH: a surprise-selected exact-outlier tier on BUG.

The Week-10 wall: at long context a rank-``r`` summary cannot reproduce a *sharp*
needle (a rare token, a low-attention high-residual outlier), so BUG retrieves 0%.
SurpriseSLASH keeps the highest-*surprise* (out-of-subspace residual) tokens
**verbatim** in the exact ``hh`` tier -- exactly the columns the low-rank basis fits
worst -- selected without any attention hook (attach-free). These pins:

* lossless config (full rank + full budget + an exact tier) still matches a full
  ``DynamicCache`` -- the surprise path does not corrupt reconstruction;
* the surprise signal ranks an orthogonal outlier highest (the needle detector);
* a planted needle lands in the ``hh`` tier **verbatim** under ``hh_select=
  "surprise"`` where plain BUG only smears it into the low-rank coordinates;
* the exact tier is populated by chunked ingest, NOT single-shot prefill (the
  load-bearing invariant: the needle lives in the prompt, so the arm must ingest);
* the coupling / validation guards.

Hermetic tiny Llama, mirroring ``tests/test_w10_ingest.py``.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

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
    model: LlamaForCausalLM,
    cache: BugStreamingCache,
    ids: torch.Tensor,
    chunk: int,
    *,
    attach: bool,
) -> None:
    """OOM-safe chunked prefill (mirrors ``w10_frontier._prefill_chunked``); the
    exact ``hh`` tier only populates through this ingest path. ``attach=False``
    exercises the attach-free surprise selection."""
    t = int(ids.shape[1])
    ctx = cache.attach(model) if attach else cache.ingesting()
    if attach:
        with ctx, cache.ingesting():
            for start in range(0, t, chunk):
                stop = min(t, start + chunk)
                model(
                    ids[:, start:stop],
                    past_key_values=cache,
                    use_cache=True,
                    position_ids=_pos(start, stop - start),
                )
                cache.consolidate()
    else:
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
    """A homogeneous background (identical pre-RoPE keys => rank-1, surprise ~0)
    with one distinct needle token at ``depth`` (a clean high-residual outlier)."""
    ids = torch.full((1, t), bg_id, dtype=torch.long)
    ids[0, depth] = needle_id
    return ids


# --------------------------------------------------------- lossless oracle


def test_surprise_slash_lossless_matches_dynamic_cache(tiny_model: LlamaForCausalLM) -> None:
    """Full rank + full budget + a surprise-selected exact tier still reconstructs
    the exact past: chunked-ingest frozen scoring matches a full DynamicCache."""
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
    )
    with torch.no_grad():
        _chunked_prefill(tiny_model, bug, ids, chunk=16, attach=False)
        with bug.frozen_scoring():
            out = tiny_model(win, past_key_values=bug, use_cache=True, position_ids=_pos(64, 16))
    assert torch.allclose(out.logits, cast(torch.Tensor, ref_out.logits), atol=2e-3, rtol=2e-3)


# --------------------------------------------------------- the surprise signal


def test_surprise_scores_ranks_orthogonal_outlier_highest(tiny_model: LlamaForCausalLM) -> None:
    """``_surprise_scores`` gives ~0 for columns inside ``span(u_k)`` and ~1 for an
    orthogonal outlier -- the needle detector, independent of the model."""
    bug = BugStreamingCache(tiny_model, rank=4, coord_budget=32)
    layer = _bug_layer(bug)
    # Basis spans e0..e2; columns 0-2 lie in it (surprise ~0), column 3 = e3 is
    # orthogonal (surprise ~1).
    eye = torch.eye(N_FEATURES)
    layer.u_k = eye[:, :3].clone()
    block = torch.stack([eye[:, 0], eye[:, 0] + eye[:, 1], eye[:, 2], eye[:, 5]], dim=1)  # (n, 4)
    surp = layer._surprise_scores(block)
    assert int(torch.argmax(surp)) == 3
    assert surp[3] > 0.99
    assert torch.all(surp[:3] < 1e-5)


def test_surprise_scores_all_one_without_basis(tiny_model: LlamaForCausalLM) -> None:
    """With no basis yet every column is maximally surprising (1.0)."""
    layer = _bug_layer(BugStreamingCache(tiny_model, rank=4, coord_budget=32))
    layer.u_k = None
    surp = layer._surprise_scores(torch.randn(N_FEATURES, 5))
    assert torch.allclose(surp, torch.ones(5))


# --------------------------------------------------------- the needle end-to-end


def test_surprise_slash_retains_needle_verbatim(tiny_model: LlamaForCausalLM) -> None:
    """A planted needle (a distinct token in a homogeneous background) lands in the
    verbatim ``hh`` tier under ``hh_select='surprise'`` and is NOT smeared into the
    low-rank coordinates. Plain BUG (no exact tier) only holds it as a coordinate."""
    depth, t = 40, 96
    ids = _needle_prompt(bg_id=5, needle_id=200, t=t, depth=depth)

    slash = BugStreamingCache(
        tiny_model,
        rank=4,
        coord_budget=128,
        recent_window=8,
        absorb_block=4,
        retention="lowrank_surprise",
        hh_budget=1,
        hh_select="surprise",
    )
    with torch.no_grad():
        _chunked_prefill(tiny_model, slash, ids, chunk=16, attach=False)
    layer = _bug_layer(slash)
    assert layer.hh_pos is not None and depth in layer.hh_pos.tolist()  # verbatim
    mid = layer.mid_pos.tolist() if layer.mid_pos is not None else []
    assert depth not in mid  # not smeared into the low-rank tail

    # Plain BUG (hh_budget=0) has no exact tier -> the needle is only a coordinate.
    plain = BugStreamingCache(
        tiny_model,
        rank=4,
        coord_budget=128,
        recent_window=8,
        absorb_block=4,
        retention="lowrank_surprise",
    )
    with torch.no_grad():
        _chunked_prefill(tiny_model, plain, ids, chunk=16, attach=False)
    player = _bug_layer(plain)
    assert player.hh_pos is None or len(player.hh_pos) == 0
    assert player.mid_pos is not None and depth in player.mid_pos.tolist()


def test_single_shot_prefill_leaves_hh_empty(tiny_model: LlamaForCausalLM) -> None:
    """The load-bearing invariant: single-shot ``_prefill`` bypasses the SLASH path
    entirely, so the exact tier is empty -- the retrieval arm MUST ingest chunked."""
    ids = _needle_prompt(bg_id=5, needle_id=200, t=64, depth=30)
    bug = BugStreamingCache(
        tiny_model,
        rank=4,
        coord_budget=128,
        recent_window=8,
        absorb_block=4,
        retention="lowrank_surprise",
        hh_budget=4,
        hh_select="surprise",
    )
    with torch.no_grad():  # single-shot prefill: one forward, no ingesting() context
        tiny_model(ids, past_key_values=bug, use_cache=True)
    layer = _bug_layer(bug)
    assert layer._hh_len() == 0


# --------------------------------------------------------- span expansion


def test_span_boost_is_position_windowed_max(tiny_model: LlamaForCausalLM) -> None:
    """``_span_boost`` gives each token the max surprise within +-w positions, and
    a gap wider than w does not leak across clusters."""
    layer = _bug_layer(BugStreamingCache(tiny_model, rank=4, coord_budget=32))
    sel = torch.tensor([0.1, 0.9, 0.2, 0.0, 0.8, 0.05])
    pos = torch.tensor([10, 11, 12, 50, 51, 52])  # two position clusters
    out = layer._span_boost(sel, pos, 1)
    assert torch.allclose(out, torch.tensor([0.9, 0.9, 0.9, 0.8, 0.8, 0.8]))
    # a lone outlier does not pull a far-away token (gap 100 > w)
    lone = layer._span_boost(torch.tensor([1.0, 0.0]), torch.tensor([0, 100]), 1)
    assert torch.allclose(lone, torch.tensor([1.0, 0.0]))
    assert torch.allclose(layer._span_boost(sel, pos, 0), sel)  # w=0 is a no-op


def test_span_expansion_captures_low_surprise_needle_member(
    tiny_model: LlamaForCausalLM,
) -> None:
    """A contiguous needle span whose MIDDLE token is an ordinary (low-surprise)
    background token is only fully captured verbatim with span expansion: the
    middle is pulled in by its outlier neighbours, keeping the hh_budget cap."""
    depth, t = 40, 96
    ids = torch.full((1, t), 5, dtype=torch.long)
    ids[0, depth], ids[0, depth + 2] = 200, 201  # two outliers flanking a bg token
    # ids[0, depth+1] stays 5 (a low-surprise background token between them)

    def run(neighbor: int) -> set[int]:
        cache = BugStreamingCache(
            tiny_model,
            rank=4,
            coord_budget=128,
            recent_window=8,
            absorb_block=4,
            n_sink=4,
            retention="lowrank_surprise",
            hh_budget=8,
            hh_select="surprise",
            hh_neighbor=neighbor,
        )
        with torch.no_grad():
            _chunked_prefill(tiny_model, cache, ids, chunk=16, attach=False)
        layer = _bug_layer(cache)
        return set(layer.hh_pos.tolist()) if layer.hh_pos is not None else set()

    no_exp = run(0)
    assert depth in no_exp and depth + 2 in no_exp  # both outliers kept
    assert depth + 1 not in no_exp  # the bg middle is NOT verbatim
    with_exp = run(1)
    assert {depth, depth + 1, depth + 2} <= with_exp  # span expansion grabs it


# --------------------------------------------------------- validation guards


def test_hh_select_surprise_allows_lowrank_surprise_retention(tiny_model: LlamaForCausalLM) -> None:
    """``hh_select='surprise'`` is attach-free and imposes no retention constraint."""
    BugStreamingCache(
        tiny_model,
        rank=4,
        coord_budget=32,
        retention="lowrank_surprise",
        hh_budget=4,
        hh_select="surprise",
    )
    BugStreamingCache(
        tiny_model, rank=4, coord_budget=32, retention="fifo", hh_budget=4, hh_select="surprise"
    )


def test_hh_select_attn_still_requires_attn_retention(tiny_model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError, match="requires retention='attn'"):
        BugStreamingCache(
            tiny_model,
            rank=4,
            coord_budget=32,
            retention="lowrank_surprise",
            hh_budget=4,
            hh_select="attn",
        )


def test_hh_select_invalid_raises(tiny_model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError, match="hh_select must be"):
        BugStreamingCache(tiny_model, rank=4, coord_budget=32, hh_budget=4, hh_select="bogus")


def test_hh_neighbor_requires_surprise(tiny_model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError, match=r"hh_neighbor.*requires hh_select='surprise'"):
        BugStreamingCache(
            tiny_model,
            rank=4,
            coord_budget=32,
            retention="attn",
            hh_budget=4,
            hh_select="attn",
            hh_neighbor=1,
        )
