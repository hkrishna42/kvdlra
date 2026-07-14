"""Tests for :mod:`kvdlra.cache.morph_cache` (the MorphKV faithful core).

Hermetic where possible (synthetic score rows drive the eviction logic
directly); end-to-end behavior runs on the same tiny random-weight Llama as
``test_bug_cache.py``. The ladder:

1. full-capacity mode is *bitwise* identical to ``DynamicCache`` (kept tokens
   are stored verbatim -- no reconstruction anywhere);
2. eviction keeps exactly ``R`` recent + top-``C`` fused distant slots, per KV
   head, in chronological order (sum and max fusion);
3. constant memory over long decode with true positions still advancing;
4. the GQA-aggregated attention row is a proper distribution per query head
   (rows sum to the group size).
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer

from kvdlra.cache import MorphKVCache, MorphKVLayer

H, D = 2, 16  # KV heads x head_dim => n_features 32
N_QUERY_HEADS = 4


def _tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=N_QUERY_HEADS,
        num_key_value_heads=H,
        head_dim=D,
        max_position_embeddings=2048,
    )


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    model = LlamaForCausalLM(_tiny_config())  # type: ignore[no-untyped-call]
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _prompt(t: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, t), generator=g)


# --------------------------------------------------------------------------
# Eviction mechanics (hermetic, synthetic scores)
# --------------------------------------------------------------------------


def _kv(t: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(1, H, t, D, generator=g),
        torch.randn(1, H, t, D, generator=g),
    )


def _positional_kv(t: int) -> tuple[torch.Tensor, torch.Tensor]:
    """K/V whose value at slot ``i`` is the constant ``i`` -- makes kept slots
    identifiable after per-head gathering."""
    k = torch.arange(t, dtype=torch.float32).view(1, 1, t, 1).expand(1, H, t, D).clone()
    return k, k.clone()


@pytest.mark.parametrize("fusion", ["sum", "max"])
def test_evict_keeps_top_c_distant_plus_recent(fusion: str) -> None:
    # 8 tokens, capacity C=2, window R=2 => keep 2 distant + 2 recent = 4.
    layer = MorphKVLayer(capacity=2, recent_window=2, fusion=fusion)
    k, v = _positional_kv(8)
    layer.update(k, v)  # pre-fill stores everything

    # Crafted rows: distant slots 1 and 4 are the clear winners for head 0,
    # slots 0 and 5 for head 1 (per-KV-head selection must differ).
    rows = torch.zeros(H, 2, 8)
    rows[0, :, 1] = 5.0
    rows[0, :, 4] = 3.0
    rows[1, :, 0] = 4.0
    rows[1, :, 5] = 2.0
    rows[:, :, 6:] = 0.1  # recent slots score too, but are kept regardless
    layer.seed_scores(rows)

    assert layer.keys is not None
    assert layer.keys.shape[2] == 4  # C + R
    kept_head0 = layer.keys[0, 0, :, 0].tolist()
    kept_head1 = layer.keys[0, 1, :, 0].tolist()
    assert kept_head0 == [1.0, 4.0, 6.0, 7.0]  # chronological, per-head top-C
    assert kept_head1 == [0.0, 5.0, 6.0, 7.0]
    # Score buffer stays column-aligned with the kept slots.
    assert layer.score_rows is not None
    assert layer.score_rows.shape == (H, 2, 4)
    assert layer.score_rows[0, 0, 0] == 5.0  # slot for token 1, head 0


def test_observe_pads_older_rows_and_slides_window() -> None:
    layer = MorphKVLayer(capacity=8, recent_window=2)
    k, v = _kv(3)
    layer.update(k, v)
    k1, v1 = _kv(1, seed=1)
    layer.update(k1, v1)
    layer.observe(torch.full((H, 4), 0.25))
    k2, v2 = _kv(1, seed=2)
    layer.update(k2, v2)
    layer.observe(torch.full((H, 5), 0.2))
    assert layer.score_rows is not None
    # Window slides at R=2 rows; the older row got a zero pad for the new token.
    assert layer.score_rows.shape == (H, 2, 5)
    assert float(layer.score_rows[0, 0, 4]) == 0.0
    assert float(layer.score_rows[0, 1, 4]) == pytest.approx(0.2)


def test_observe_validates_row_shape() -> None:
    layer = MorphKVLayer(capacity=4, recent_window=2)
    layer.update(*_kv(3))
    with pytest.raises(ValueError, match="attn_row"):
        layer.observe(torch.zeros(H, 7))


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="capacity"):
        MorphKVLayer(capacity=0, recent_window=2)
    with pytest.raises(ValueError, match="recent_window"):
        MorphKVLayer(capacity=2, recent_window=0)
    with pytest.raises(ValueError, match="fusion"):
        MorphKVLayer(capacity=2, recent_window=2, fusion="mean")


# --------------------------------------------------------------------------
# End-to-end on the tiny model
# --------------------------------------------------------------------------


def test_full_capacity_is_bitwise_dynamic_cache(tiny_model: LlamaForCausalLM) -> None:
    # capacity + window > everything generated => nothing is ever evicted and the
    # stored tokens are verbatim, so logits must be *bitwise* equal to the stock
    # cache (stronger than the BUG cache's fp-tolerance parity: no reconstruction).
    cache = MorphKVCache(tiny_model, capacity=512, recent_window=8)
    ref = DynamicCache()
    stream = _prompt(70, seed=7)
    with torch.no_grad(), cache.attach(tiny_model):
        out_a = tiny_model(stream[:, :30], past_key_values=cache, use_cache=True)
        out_b = tiny_model(stream[:, :30], past_key_values=ref, use_cache=True)
        assert torch.equal(out_a.logits, out_b.logits)
        for t in range(30, 70):
            tok = stream[:, t : t + 1]
            out_a = tiny_model(tok, past_key_values=cache, use_cache=True)
            out_b = tiny_model(tok, past_key_values=ref, use_cache=True)
            assert torch.equal(out_a.logits, out_b.logits)
    layer = cache.layers[0]
    ref_layer = ref.layers[0]
    assert isinstance(layer, MorphKVLayer)
    assert isinstance(ref_layer, DynamicLayer)
    assert layer.keys is not None and ref_layer.keys is not None
    assert torch.equal(layer.keys, ref_layer.keys)


def test_constant_memory_and_positions_advance(tiny_model: LlamaForCausalLM) -> None:
    cap, win = 16, 8
    cache = MorphKVCache(tiny_model, capacity=cap, recent_window=win)
    ids = _prompt(60)
    mems = []
    with torch.no_grad(), cache.attach(tiny_model):
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        for _ in range(150):
            tok = out.logits[:, -1:].argmax(dim=-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
            mems.append(cache.stored_state_numel())
    assert max(mems[20:]) <= max(mems[:20])  # bound reached early, never exceeded
    layer = cache.layers[0]
    assert isinstance(layer, MorphKVLayer)
    assert layer.cumulative_length == 210  # true positions kept advancing
    assert layer.keys is not None
    # Steady state: kept tokens oscillate in [C+R, C+R+1) around the hook timing.
    assert cap + win <= layer.keys.shape[2] <= cap + win + 1


def test_aggregated_row_is_group_normalized(tiny_model: LlamaForCausalLM) -> None:
    # Each query head's softmax row sums to 1; summing over the group means each
    # KV head's aggregated row must sum to (num query heads / num kv heads).
    # Capacity is generous so no eviction gathers columns out of the row first.
    cache = MorphKVCache(tiny_model, capacity=64, recent_window=4)
    ids = _prompt(20)
    with torch.no_grad(), cache.attach(tiny_model):
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        tok = out.logits[:, -1:].argmax(dim=-1)
        tiny_model(tok, past_key_values=cache, use_cache=True)
    layer = cache.layers[0]
    assert isinstance(layer, MorphKVLayer)
    assert layer.score_rows is not None
    last_row = layer.score_rows[:, -1, :]
    groups = N_QUERY_HEADS / H
    assert torch.allclose(last_row.sum(dim=1), torch.full((H,), groups), atol=1e-4)


def test_evict_interval_bounds_overshoot(tiny_model: LlamaForCausalLM) -> None:
    # The periodic ("SnapKV-decode") variant prunes every k steps: kept tokens may
    # overshoot capacity+window by at most k-1 between prunes, never more.
    cap, win, k = 12, 4, 8
    cache = MorphKVCache(tiny_model, capacity=cap, recent_window=win, evict_interval=k)
    ids = _prompt(40)
    peaks = []
    with torch.no_grad(), cache.attach(tiny_model):
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        for _ in range(60):
            tok = out.logits[:, -1:].argmax(dim=-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
            layer = cache.layers[0]
            assert isinstance(layer, MorphKVLayer)
            assert layer.keys is not None
            peaks.append(int(layer.keys.shape[2]))
    assert max(peaks) <= cap + win + k
    assert min(peaks[10:]) <= cap + win + 1  # it does get pruned back down


def test_chunked_prefill_raises(tiny_model: LlamaForCausalLM) -> None:
    cache = MorphKVCache(tiny_model, capacity=16, recent_window=4)
    with torch.no_grad(), cache.attach(tiny_model):
        tiny_model(_prompt(16), past_key_values=cache, use_cache=True)
        with pytest.raises(NotImplementedError, match="single-shot"):
            tiny_model(_prompt(8, seed=2), past_key_values=cache, use_cache=True)


def test_chunked_ingest_short_final_chunk(tiny_model: LlamaForCausalLM) -> None:
    # Week-10 regression: the OOM-safe ingesting() path feeds the prompt in chunks
    # under attach(). When the FINAL chunk carries fewer tokens than recent_window
    # (here 4 < 8), _window_attention_rows must bound its window by the chunk length
    # -- otherwise the q/cos rows (w) mismatch the hidden-states rows and the einsum
    # raises "size of tensor a (w) must match tensor b (chunk) at dim 2".
    cap, win, chunk = 16, 8, 8
    prompt = _prompt(20, seed=3)  # chunks: [0:8], [8:16], [16:20] -> final = 4 < win
    cache = MorphKVCache(tiny_model, capacity=cap, recent_window=win)
    with torch.no_grad(), cache.attach(tiny_model), cache.ingesting():
        for start in range(0, prompt.shape[1], chunk):
            stop = min(prompt.shape[1], start + chunk)
            pos = torch.arange(start, stop).unsqueeze(0)
            tiny_model(
                prompt[:, start:stop],
                past_key_values=cache,
                use_cache=True,
                position_ids=pos,
                logits_to_keep=1,
            )
            cache.consolidate()
    layer = cache.layers[0]
    assert isinstance(layer, MorphKVLayer)
    assert layer.keys is not None and layer.score_rows is not None
    length = int(layer.keys.shape[2])
    # Score window never exceeds R rows and stays column-aligned with the kept keys.
    assert layer.score_rows.shape[1] <= win
    assert int(layer.score_rows.shape[2]) == length
    assert length <= cap + win  # kept set stayed bounded through the ragged ingest
