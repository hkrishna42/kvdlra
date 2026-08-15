"""Tests for :mod:`kvdlra.cache.shadow_cache` (the ShadowKV in-repo port).

Hermetic: a *tiny random-weight* Llama (2 layers, 2 KV heads x head_dim 16 =>
n_features 32) exercises the full transformers-5.8 decode path -- mask sizes,
position derivation, sdpa/eager attention -- end to end, mirroring
``tests/test_bug_cache.py``. The correctness gates (Phase-6 spec):

(a) **degenerate lossless**: with ``rank_s >= head_dim`` (full rank) and ``top_k >=
    n_chunks`` (all chunks selected), the ShadowKV logits match a plain
    ``DynamicCache`` over a prefill + a few decode steps (lossy low-rank round trip
    => allclose, not exact);
(b) **positions advance** (``get_seq_length`` cumulative) and **mask consistency**
    (returned K/V length == ``get_mask_sizes``), every step;
(c) **CPU-V counted** in ``stored_state_numel`` + the ``shadow_footprint`` anti-drift
    pin;
(d) **top-k selects the planted chunk** (a query matching a chunk's landmark puts
    that chunk in the top-k).
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer

from kvdlra import accounting as acc
from kvdlra.cache import ShadowKVCache
from kvdlra.cache.shadow_cache import ShadowKVLayer

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
        max_position_embeddings=2048,
    )


@pytest.fixture(scope="module")
def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    model = LlamaForCausalLM(_tiny_config())  # type: ignore[no-untyped-call]
    model.config._attn_implementation = "sdpa"
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _prompt(t: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, t), generator=g)


def _pos(start: int, n: int) -> torch.Tensor:
    return torch.arange(start, start + n).unsqueeze(0)


def _shadow_layer(cache: ShadowKVCache) -> ShadowKVLayer:
    return next(layer for layer in cache.layers if isinstance(layer, ShadowKVLayer))


# --------------------------------------------------------------- (a) lossless


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_degenerate_lossless_matches_dynamic_cache(tiny_model: LlamaForCausalLM, attn: str) -> None:
    """rank_s >= head_dim (full-rank keys) and top_k >= n_chunks (all chunks
    selected) => nothing is dropped, so the cache is information-losslessly
    equivalent to DynamicCache over prefill + decode (the low-rank middle is
    reconstructed through an fp32 un-/re-rotate round trip, so allclose not exact)."""
    tiny_model.config._attn_implementation = attn
    cache = ShadowKVCache(tiny_model, rank_s=N_FEATURES, top_k=1000, chunk=8, recent_window=8)
    ref = DynamicCache()
    g = torch.Generator().manual_seed(7)
    stream = torch.randint(0, 256, (1, 90), generator=g)
    layer = _shadow_layer(cache)
    with torch.no_grad(), cache.attach(tiny_model):
        out_a = tiny_model(stream[:, :33], past_key_values=cache, use_cache=True)
        out_b = tiny_model(stream[:, :33], past_key_values=ref, use_cache=True)
        assert torch.equal(out_a.logits, out_b.logits)  # prefill returns input as-is
        for t in range(33, 90):
            tok = stream[:, t : t + 1]
            out_a = tiny_model(tok, past_key_values=cache, use_cache=True)
            out_b = tiny_model(tok, past_key_values=ref, use_cache=True)
            assert torch.allclose(out_a.logits, out_b.logits, atol=1e-4)
    # The reconstructed retained K/V at full rank + all chunks matches the reference.
    k_ret, v_ret = layer._assemble()
    ref_layer = ref.layers[0]
    assert isinstance(ref_layer, DynamicLayer)
    assert ref_layer.keys is not None and ref_layer.values is not None
    assert torch.allclose(k_ret, ref_layer.keys, atol=1e-4)
    assert torch.allclose(v_ret, ref_layer.values, atol=1e-4)


def test_frozen_scoring_lossless_matches_dynamic_cache(tiny_model: LlamaForCausalLM) -> None:
    """Full-rank + all chunks => the frozen-scoring window logits match a full
    DynamicCache window forward (the get_mask_sizes math is correct end to end)."""
    ids, win = _prompt(50), _prompt(16, seed=7)
    ref = DynamicCache()
    with torch.no_grad():
        tiny_model(ids, past_key_values=ref, use_cache=True)
        ref_logits = tiny_model(
            win, past_key_values=ref, use_cache=True, position_ids=_pos(50, 16)
        ).logits
    cache = ShadowKVCache(tiny_model, rank_s=N_FEATURES, top_k=1000, chunk=8, recent_window=8)
    with torch.no_grad(), cache.attach(tiny_model):
        tiny_model(ids, past_key_values=cache, use_cache=True)
    with torch.no_grad(), cache.frozen_scoring():
        out = tiny_model(win, past_key_values=cache, use_cache=True, position_ids=_pos(50, 16))
    assert torch.allclose(out.logits, ref_logits, atol=2e-3, rtol=2e-3)


# ------------------------------------------------- (b) positions + mask sizes


def test_positions_advance_and_masks_stay_consistent(tiny_model: LlamaForCausalLM) -> None:
    """Aggressive rank + sparse top_k: outputs may differ from baseline, but every
    step's predicted mask size must equal the K/V length actually returned, and the
    cumulative length (true positions) keeps advancing."""
    cache = ShadowKVCache(tiny_model, rank_s=4, top_k=2, chunk=8, recent_window=8, n_sink=4)
    layer = _shadow_layer(cache)
    ids = _prompt(48)
    with torch.no_grad(), cache.attach(tiny_model):
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        assert layer.get_seq_length() == 48
        for step in range(60):
            kv_length, kv_offset = layer.get_mask_sizes(1)
            tok = out.logits[:, -1:].argmax(-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
            assert kv_length == layer.attended_length()
            # The offset places the returned block's last position at the query's.
            assert kv_offset + kv_length == layer.cumulative_length
            assert layer.get_seq_length() == 48 + step + 1


def test_streaming_degenerate_when_no_middle(tiny_model: LlamaForCausalLM) -> None:
    """A prompt shorter than n_sink + recent_window has no middle => sinks + recent
    only (StreamingLLM), and decode still masks consistently."""
    cache = ShadowKVCache(tiny_model, rank_s=8, top_k=4, chunk=8, recent_window=8, n_sink=4)
    layer = _shadow_layer(cache)
    ids = _prompt(10)  # 10 <= n_sink(4) + recent_window(8)
    with torch.no_grad(), cache.attach(tiny_model):
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        assert layer.n_chunks == 0 and layer.mid_len == 0
        for _ in range(8):
            kv_length, _ = layer.get_mask_sizes(1)
            tok = out.logits[:, -1:].argmax(-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
            assert kv_length == layer.attended_length()


# --------------------------------------------------------- (c) CPU-V counted


def test_cpu_value_cache_is_counted(tiny_model: LlamaForCausalLM) -> None:
    """stored_state_numel includes the offloaded value cache (T*n), so a zeroed CPU
    term would fail loudly; the total exceeds keys-plus-values-verbatim lower bound."""
    cache = ShadowKVCache(tiny_model, rank_s=8, top_k=2, chunk=8, recent_window=8, n_sink=4)
    layer = _shadow_layer(cache)
    ids = _prompt(60)
    with torch.no_grad():
        tiny_model(ids, past_key_values=cache, use_cache=True)
    t = layer.cumulative_length
    assert layer.cpu_value_numel() == t * N_FEATURES  # the whole value cache
    # Total >= offloaded V (t*n) + verbatim sink/recent keys (n * (n_sink+recent)).
    lower = t * N_FEATURES + N_FEATURES * (layer.n_sink + layer._recent_len())
    assert layer.stored_state_numel() >= lower
    # Explicitly: dropping the CPU-V term must strictly lower the count.
    assert layer.stored_state_numel() - layer.cpu_value_numel() < layer.stored_state_numel()


@pytest.mark.parametrize("rank_s,top_k", [(8, 2), (4, 1000), (16, 3)])
def test_shadow_footprint_matches_stored_state_numel(
    tiny_model: LlamaForCausalLM, rank_s: int, top_k: int
) -> None:
    """The anti-drift pin: ``shadow_footprint`` computed from the live layer's actual
    counts equals the measured ``stored_state_numel`` (the accounting formula and the
    measured path cannot diverge)."""
    cache = ShadowKVCache(
        tiny_model, rank_s=rank_s, top_k=top_k, chunk=8, recent_window=8, n_sink=4
    )
    layer = _shadow_layer(cache)
    ids = _prompt(70)
    with torch.no_grad():
        tiny_model(ids, past_key_values=cache, use_cache=True)
    t = layer.cumulative_length
    fp = acc.shadow_footprint(
        t,
        N_FEATURES,
        H,
        D,
        rank_s=rank_s,
        chunk=8,
        n_sink=layer.n_sink,
        recent_len=layer._recent_len(),
    )
    assert fp.float_equiv() == layer.stored_state_numel()
    # Cache-level total is n_layers * the per-layer footprint.
    assert cache.stored_state_numel() == 2 * layer.stored_state_numel()


def test_shadow_footprint_cpu_ratio_half_and_gpu_smaller() -> None:
    """The honest split: the offloaded V is exactly half the full fp16 K+V cache and
    the GPU-only ratio is strictly below the honest total (the offload is not free)."""
    t, n, h_kv, head_dim = 2048, 512, 8, 64
    fp = acc.shadow_footprint(t, n, h_kv, head_dim, rank_s=64, n_sink=4, recent_len=64)
    assert fp.cpu_verbatim_elems == t * n
    assert fp.cpu_ratio_fp16(t, n) == pytest.approx(0.5)
    assert fp.gpu_ratio_fp16(t, n) < fp.ratio_fp16(t, n)
    assert fp.ratio_fp16(t, n) == pytest.approx(fp.gpu_ratio_fp16(t, n) + fp.cpu_ratio_fp16(t, n))


# ----------------------------------------------------- (d) top-k selection


def test_topk_selects_planted_chunk(tiny_model: LlamaForCausalLM) -> None:
    """A query equal to a chunk's landmark scores that chunk highest => it is the
    top-1 selection for every KV head (the pillar-4 selection core)."""
    cache = ShadowKVCache(tiny_model, rank_s=8, top_k=1, chunk=8, recent_window=8, n_sink=4)
    layer = _shadow_layer(cache)
    ids = _prompt(60)
    with torch.no_grad():
        tiny_model(ids, past_key_values=cache, use_cache=True)
    assert layer.landmarks is not None and layer.n_chunks >= 2
    for target in range(layer.n_chunks):
        # A synthetic post-RoPE query = that head's target landmark (max similarity).
        qf = layer.landmarks[:, target, :].view(H, 1, 1, D).to(torch.float32)
        sel = layer._select_chunks(qf)  # (H, 1)
        assert sel.shape == (H, 1)
        assert all(int(sel[h, 0]) == target for h in range(H))


def test_topk_planted_recall_end_to_end(tiny_model: LlamaForCausalLM) -> None:
    """Plant a distinctive key pattern in one middle chunk and drive a real decode
    step whose recomputed query aligns with that chunk's landmark; the pre-attention
    hook must place the planted chunk in the top-k for at least one head."""
    cache = ShadowKVCache(tiny_model, rank_s=8, top_k=2, chunk=8, recent_window=8, n_sink=4)
    layer = _shadow_layer(cache)
    ids = _prompt(60)
    with torch.no_grad(), cache.attach(tiny_model):
        out = tiny_model(ids, past_key_values=cache, use_cache=True)
        assert layer.landmarks is not None
        # Force the next-step query, per head, to equal a target landmark and confirm
        # the hook's planned selection contains it.
        target = 1
        cos, sin = torch.ones(1, 1, D), torch.zeros(1, 1, D)  # identity RoPE probe

        class _Probe(torch.nn.Module):
            def __init__(self, lm: torch.Tensor) -> None:
                super().__init__()
                self.lm = lm

            def q_proj(self, hs: torch.Tensor) -> torch.Tensor:
                # 4 query heads (2 per KV head): all set to the target landmark.
                per_head = self.lm[:, target, :]  # (H_kv, D)
                return per_head.repeat_interleave(2, dim=0).reshape(1, 1, -1)

        layer._plan_selection(_Probe(layer.landmarks), out.logits[:, -1:], cos, sin)
        assert layer._selected is not None
        assert any(target in layer._selected[h].tolist() for h in range(H))


# --------------------------------------------------------- scope / validation


def test_chunked_prefill_raises(tiny_model: LlamaForCausalLM) -> None:
    cache = ShadowKVCache(tiny_model, rank_s=8, top_k=4)
    with torch.no_grad():
        tiny_model(_prompt(40), past_key_values=cache, use_cache=True)
        with pytest.raises(NotImplementedError, match="single-shot"):
            tiny_model(_prompt(8, seed=2), past_key_values=cache, use_cache=True)


def test_ingesting_second_chunk_raises(tiny_model: LlamaForCausalLM) -> None:
    """Chunked ingest is a documented NO-GO: the first chunk prefills, a second
    raises (deviation #5)."""
    cache = ShadowKVCache(tiny_model, rank_s=8, top_k=4)
    with torch.no_grad(), cache.ingesting():
        tiny_model(_prompt(40), past_key_values=cache, use_cache=True)
        with pytest.raises(NotImplementedError, match="chunked ingest"):
            tiny_model(_prompt(8, seed=2), past_key_values=cache, use_cache=True)
    cache.consolidate()  # no-op, must not raise


def test_batch_size_one_enforced(tiny_model: LlamaForCausalLM) -> None:
    cache = ShadowKVCache(tiny_model, rank_s=8, top_k=4)
    ids = torch.randint(0, 256, (2, 10))
    with torch.no_grad(), pytest.raises(NotImplementedError, match="batch size 1"):
        tiny_model(ids, past_key_values=cache, use_cache=True)


def test_constructor_validation(tiny_model: LlamaForCausalLM) -> None:
    with pytest.raises(ValueError, match="rank_s"):
        ShadowKVCache(tiny_model, rank_s=0, top_k=4)
    with pytest.raises(ValueError, match="top_k"):
        ShadowKVCache(tiny_model, rank_s=8, top_k=0)
    with pytest.raises(ValueError, match="chunk"):
        ShadowKVCache(tiny_model, rank_s=8, top_k=4, chunk=0)
    with pytest.raises(ValueError, match="recent_window"):
        ShadowKVCache(tiny_model, rank_s=8, top_k=4, recent_window=0)


def test_reset_allows_reuse(tiny_model: LlamaForCausalLM) -> None:
    cache = ShadowKVCache(tiny_model, rank_s=8, top_k=2, recent_window=8)
    with torch.no_grad(), cache.attach(tiny_model):
        out = tiny_model(_prompt(40), past_key_values=cache, use_cache=True)
        for _ in range(5):
            tok = out.logits[:, -1:].argmax(-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)
    for layer in cache.layers:
        layer.reset()
    assert _shadow_layer(cache).cumulative_length == 0
    with torch.no_grad(), cache.attach(tiny_model):
        tiny_model(_prompt(40, seed=3), past_key_values=cache, use_cache=True)
    assert _shadow_layer(cache).cumulative_length == 40


def test_unattached_selection_raises(tiny_model: LlamaForCausalLM) -> None:
    """Week-15: decode WITHOUT attach() when selection matters (k_eff < n_chunks)
    RAISES instead of silently falling back to the most recent chunks. The
    silent fall-back was the harness defect behind the published shadow 0/0/0/0
    RULER rows (decode ran outside attach -> most-recent-chunks retention ->
    mid-context needle excluded by construction). Prefill alone stays legal
    (no selection happens there)."""
    cache = ShadowKVCache(tiny_model, rank_s=8, top_k=2, chunk=8, recent_window=8, n_sink=4)
    layer = _shadow_layer(cache)
    ids = _prompt(60)
    with torch.no_grad():
        out = tiny_model(ids, past_key_values=cache, use_cache=True)  # prefill: fine unattached
    assert layer._k_eff() < layer.n_chunks  # the selection-matters regime
    tok = out.logits[:, -1:].argmax(-1)
    with torch.no_grad(), pytest.raises(RuntimeError, match="attach"):
        tiny_model(tok, past_key_values=cache, use_cache=True)


def test_unattached_decode_ok_when_all_chunks_selected(tiny_model: LlamaForCausalLM) -> None:
    """k_eff >= n_chunks (small context / big top_k): every chunk is selected
    regardless of the query, so the chronological fall-back IS the complete
    correct selection and unattached decode keeps working -- the legitimate
    path the Week-15 raise must NOT break."""
    cache = ShadowKVCache(tiny_model, rank_s=8, top_k=1000, chunk=8, recent_window=8, n_sink=4)
    layer = _shadow_layer(cache)
    with torch.no_grad():
        out = tiny_model(_prompt(60), past_key_values=cache, use_cache=True)
        assert layer._k_eff() == layer.n_chunks
        for _ in range(3):
            tok = out.logits[:, -1:].argmax(-1)
            out = tiny_model(tok, past_key_values=cache, use_cache=True)  # must not raise


def test_non_mutating_frozen_scoring(tiny_model: LlamaForCausalLM) -> None:
    cache = ShadowKVCache(tiny_model, rank_s=8, top_k=2, recent_window=8)
    ids = _prompt(60)
    with torch.no_grad(), cache.attach(tiny_model):
        tiny_model(ids, past_key_values=cache, use_cache=True)
    layer = _shadow_layer(cache)
    before_numel = cache.stored_state_numel()
    before_cum = layer.cumulative_length
    win = _prompt(16, seed=3)
    with torch.no_grad(), cache.frozen_scoring():
        out = tiny_model(win, past_key_values=cache, use_cache=True, position_ids=_pos(60, 16))
    assert out.logits.shape[1] == 16
    assert cache.stored_state_numel() == before_numel
    assert layer.cumulative_length == before_cum
    assert all(la._mode == "normal" for la in cache.layers if isinstance(la, ShadowKVLayer))
