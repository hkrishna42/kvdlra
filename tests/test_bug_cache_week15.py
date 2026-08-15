"""Week-15 T2 score-rank decoupling (``score_rank``): the regression pins.

The lead-bet knob caps SurpriseSLASH *selection* scoring to the leading
``score_rank`` columns of the tracked basis (``u_k[:, :s]`` -- energy-ordered by
the integrator's per-step SVD, orthonormal for free), decoupling selection-rank
from storage-rank: a big basis fits needles (residual -> 0 -> never selected, the
measured rank-retrieval coupling), while the leading-``s`` subview keeps them
surprising. It ships **default-off** (``None``), applies ONLY at the SLASH
selection site (``_absorb_block_slash``), and leaves the tail-retention surprise
snapshot (``_absorb_columns``), storage rank and accounting untouched.

This file is the regression contract: off-path bit-for-bit identity (omitted vs
``None``; ``score_rank=rank`` -- the plumbing pin), the validation surface,
composition with ``seed_hh_warmup``, the caps-only-SLASH site discipline, and
accounting neutrality. Plus the Week-15 T3 scoping deliverable: a strict-xfail
characterization test pinning the latent ``seed_scores`` chunked-ingest bug.

Hermetic tiny Llama, mirroring ``tests/test_bug_cache_seed_regression.py``.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer

H, D = 2, 16
N_FEATURES = H * D
RANK = 4

# Every stored per-layer tensor score_rank could touch (compared bit-for-bit).
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
    """The deployed ``bugS`` shape: low-rank gist + a surprise-selected exact tier
    (mirrors ``tests/test_bug_cache_seed_regression.py``)."""
    kw: dict[str, object] = {
        "rank": RANK,
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


def _run_pair_identity(
    tiny_model: LlamaForCausalLM, kw_a: dict[str, object], kw_b: dict[str, object]
) -> None:
    """Chunked-prefill two configs on the same stream; assert bit-for-bit stored
    state AND frozen-scoring logits on a fixed continuation window."""
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 256, (1, 96), generator=g)
    win = torch.randint(0, 256, (1, 12), generator=torch.Generator().manual_seed(9))
    a = BugStreamingCache(tiny_model, **kw_a)  # type: ignore[arg-type]
    b = BugStreamingCache(tiny_model, **kw_b)  # type: ignore[arg-type]
    with torch.no_grad():
        _chunked_prefill(tiny_model, a, ids, chunk=16)
        _chunked_prefill(tiny_model, b, ids, chunk=16)
    _assert_state_identical(_bug_layer(a), _bug_layer(b))
    with torch.no_grad():
        with a.frozen_scoring():
            oa = tiny_model(win, past_key_values=a, use_cache=True, position_ids=_pos(96, 12))
        with b.frozen_scoring():
            ob = tiny_model(win, past_key_values=b, use_cache=True, position_ids=_pos(96, 12))
    assert torch.equal(cast(torch.Tensor, oa.logits), cast(torch.Tensor, ob.logits))


# --------------------------------------------- 1. identity: off is a true no-op


def test_score_rank_none_identity(tiny_model: LlamaForCausalLM) -> None:
    """``score_rank=None`` is bit-for-bit identical -- every stored tensor AND the
    frozen-scoring logits -- to the deployed arm built with the flag omitted."""
    _run_pair_identity(tiny_model, _slash(), _slash(score_rank=None))


def test_score_rank_full_identity(tiny_model: LlamaForCausalLM) -> None:
    """The plumbing pin: ``score_rank=rank`` caps at ``min(rank, u_k.shape[1])`` --
    never fewer columns than the full basis -- so the capped path must be
    bit-for-bit the baseline. Any accidental off-by-one / mis-ordering in the
    subview trips this before a pod ever runs."""
    _run_pair_identity(tiny_model, _slash(), _slash(score_rank=RANK))


# ------------------------------------------------------------- 2. validation


@pytest.mark.parametrize(
    ("score_rank", "over"),
    [
        (0, {}),
        (RANK + 1, {}),
        (2, {"retention": "attn", "hh_select": "attn", "hh_budget": 2}),
        (2, {"hh_budget": 0}),
    ],
    ids=["zero", "above_rank", "attn_select", "no_hh_tier"],
)
def test_score_rank_validation(
    tiny_model: LlamaForCausalLM, score_rank: int, over: dict[str, object]
) -> None:
    """``score_rank`` requires ``1 <= score_rank <= rank``, ``hh_select='surprise'``
    and an enabled SLASH tier (``hh_budget >= 1``); each violation fails loud."""
    with pytest.raises(ValueError, match="score_rank"):
        BugStreamingCache(tiny_model, score_rank=score_rank, **_slash(**over))  # type: ignore[arg-type]


# --------------------------------------------- 3. composes with seed_hh_warmup


def test_score_rank_seed_composes(tiny_model: LlamaForCausalLM) -> None:
    """``seed_hh_warmup=True`` + ``score_rank`` construct together (the guard is
    not extended) and the seeded first-chunk needle is still captured verbatim
    into the exact tier -- the Week-13 warm-up fix survives the Week-15 cap."""
    depth, t = 20, 96
    ids = _needle_prompt(bg_id=5, needle_id=200, t=t, depth=depth)
    cache = BugStreamingCache(
        tiny_model,
        seed_hh_warmup=True,
        score_rank=2,
        **_slash(),  # type: ignore[arg-type]
    )
    with torch.no_grad():
        _chunked_prefill(tiny_model, cache, ids, chunk=32)
    layer = _bug_layer(cache)
    assert layer.score_rank == 2 and layer.seed_hh_warmup
    assert layer.hh_pos is not None and depth in layer.hh_pos.tolist()
    assert layer.mid_pos is None or depth not in layer.mid_pos.tolist()


# --------------------------------------------- 4. the cap applies ONLY at SLASH


def test_score_rank_caps_only_slash(
    tiny_model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Site discipline: the cap is passed at the ``_absorb_block_slash`` selection
    site ONLY; the ``_absorb_columns`` tail-retention snapshot stays uncapped.

    Two independent reads: (i) record every ``_surprise_scores`` call's caller +
    ``cap`` -- SLASH calls must carry the cap, snapshot calls must not; (ii) on a
    stream where capped and uncapped SELECT the same tokens (one dominant needle,
    ``hh_budget=1``), the stored ``mid_surprise`` snapshots must be bit-identical
    -- a wrongly-capped snapshot would be measured against a truncated basis and
    diverge numerically."""
    import sys

    calls: list[tuple[str, int | None]] = []
    orig = BugStreamingLayer._surprise_scores

    def recording(
        self: BugStreamingLayer, block_k: torch.Tensor, cap: int | None = None
    ) -> torch.Tensor:
        calls.append((sys._getframe(1).f_code.co_name, cap))
        return orig(self, block_k, cap)

    monkeypatch.setattr(BugStreamingLayer, "_surprise_scores", recording)

    depth, t = 40, 128  # mid-stream needle: captured by steady-state SLASH
    ids = _needle_prompt(bg_id=5, needle_id=200, t=t, depth=depth)
    capped = BugStreamingCache(tiny_model, score_rank=2, **_slash())  # type: ignore[arg-type]
    with torch.no_grad():
        _chunked_prefill(tiny_model, capped, ids, chunk=32)
    slash_caps = {c for (caller, c) in calls if caller == "_absorb_block_slash"}
    snapshot_caps = {c for (caller, c) in calls if caller == "_absorb_columns"}
    assert slash_caps == {2}  # every SLASH selection scored against u_k[:, :2]
    assert snapshot_caps == {None}  # every retention snapshot scored full-basis
    assert {caller for (caller, _c) in calls} == {"_absorb_block_slash", "_absorb_columns"}

    calls.clear()
    uncapped = BugStreamingCache(tiny_model, **_slash())  # type: ignore[arg-type]
    with torch.no_grad():
        _chunked_prefill(tiny_model, uncapped, ids, chunk=32)
    lc, lu = _bug_layer(capped), _bug_layer(uncapped)
    # The dominant needle is selected at either scoring rank -> same exact tier...
    assert lc.hh_pos is not None and lu.hh_pos is not None
    assert torch.equal(lc.hh_pos, lu.hh_pos)
    assert depth in lc.hh_pos.tolist()
    # ...so identical demotions reach the tail, and the (uncapped) snapshots agree
    # bit-for-bit. A cap leak into _absorb_columns would break this equality.
    assert lc.mid_surprise is not None and lu.mid_surprise is not None
    assert torch.equal(lc.mid_surprise, lu.mid_surprise)


# ------------------------------------------------------ 5. accounting identity


def test_score_rank_accounting_identity(tiny_model: LlamaForCausalLM) -> None:
    """``score_rank`` changes WHICH tokens the exact tier keeps, never how many
    floats are stored: at a saturated steady state, capped vs uncapped have
    identical tier lengths and identical ``stored_state_numel`` (layer and cache
    level) at the same storage rank -- the zero-accounting-change pin."""
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 256, (1, 400), generator=g)  # well past saturation
    kw = _slash(coord_budget=24, hh_budget=6)
    capped = BugStreamingCache(tiny_model, score_rank=2, **kw)  # type: ignore[arg-type]
    uncapped = BugStreamingCache(tiny_model, **kw)  # type: ignore[arg-type]
    with torch.no_grad():
        _chunked_prefill(tiny_model, capped, ids, chunk=16)
        _chunked_prefill(tiny_model, uncapped, ids, chunk=16)
    lc, lu = _bug_layer(capped), _bug_layer(uncapped)
    assert lc._hh_len() == lu._hh_len() == 6
    assert lc._f_len() == lu._f_len()
    assert lc._recent_len() == lu._recent_len()
    assert lc.u_k is not None and lu.u_k is not None
    assert lc.u_k.shape == lu.u_k.shape  # storage rank untouched by the cap
    assert lc.stored_state_numel() == lu.stored_state_numel()
    assert capped.stored_state_numel() == uncapped.stored_state_numel()


# ---------------------- 6. Week-15 T3 scoping: the latent seed_scores bug (xfail)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "LATENT BUG (Week-15 T3 characterization): seed_scores indexes ABSOLUTE "
        "positions (cumulative_length, mid_pos) into a CHUNK-LENGTH seed from "
        "_prompt_seed_scores, so under chunked ingest with attach() the ring seed "
        "silently desyncs (out-of-range slice -> wrong length) and mid_pos >= "
        "chunk raises IndexError. A fix must map chunk-local rows to absolute "
        "positions; when it lands, this strict xfail trips and must be replaced "
        "by real correctness pins."
    ),
)
def test_seed_scores_chunked_ingest_latent_bug(tiny_model: LlamaForCausalLM) -> None:
    """Pin the failing condition minimally: retention='attn' (score seeding
    active) + attach() + chunked ingest. By chunk 3 the retained middle holds
    absolute positions >= chunk length, so the hook's ``seed[self.mid_pos]``
    is out of bounds; the ring slice ``seed[cumulative-rlen:cumulative]`` had
    already desynced at chunk 2. Asserts the CORRECT behaviour (in-sync score
    buffers after a crash-free ingest) so a silent half-fix XPASSes and trips."""
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, 256, (1, 96), generator=g)
    cache = BugStreamingCache(
        tiny_model,
        rank=RANK,
        coord_budget=128,
        recent_window=8,
        absorb_block=4,
        n_sink=4,
        prefill_block_size=8,
        retention="attn",
    )
    with torch.no_grad(), cache.attach(tiny_model):
        _chunked_prefill(tiny_model, cache, ids, chunk=32)
    layer = _bug_layer(cache)
    assert layer.ring_score is not None
    assert int(layer.ring_score.shape[0]) == layer._recent_len()
    assert layer.mid_score is not None
    assert int(layer.mid_score.shape[0]) == layer._f_len()
