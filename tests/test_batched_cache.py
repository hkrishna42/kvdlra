"""Batch > 1 on the streaming cache (ADR 0001 §5 "Batch > 1 is a prerequisite"; D-002 concern 1).

One ``BugStreamingLayer`` per model layer still, but at batch B > 1 the layer owns B
independent row layers, built from its own constructor kwargs at the first ``update`` and
each driven through the unchanged batch-1 path on its ``(1, H, T, D)`` slice. So the pins
are: a batch-2 run IS two batch-1 runs (logits, greedy choices and per-row state), every
cache-level reader fans out (lengths, mask sizes, footprints, diagnostics, modes, reset),
and the latency axis runs at batch 2. Batch 1 never enters the row path --
``tests/test_golden_cache.py`` is the bit-identity tripwire for that.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer
from kvdlra.eval.frontier import _prefill_chunked
from kvdlra.eval.latency import run_latency
from tests.conftest import tiny_cache

TINY_SDPA = True
T, CHUNK, N_NEW = 48, 16, 12


def _prompt(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, T), generator=g)


def _stream(seed: int, b: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (b, N_NEW), generator=g)


@torch.no_grad()
def _teacher_forced(
    model: LlamaForCausalLM, cache: BugStreamingCache, ids: torch.Tensor, stream: torch.Tensor
) -> list[torch.Tensor]:
    """Chunked prefill of ``ids`` (B, T), then N_NEW teacher-forced decode steps on
    ``stream`` (B, N_NEW) at explicit true positions -- how `latency.run_latency` drives a
    cache. Returns the per-step last-token logits, fp32 (B, vocab)."""
    logits = []
    with cache.attach(model):
        _prefill_chunked(model, cache, ids, CHUNK)
        for s in range(stream.shape[1]):
            pos = torch.full((ids.shape[0], 1), T + s, dtype=torch.long)
            out = model(
                stream[:, s : s + 1], past_key_values=cache, use_cache=True, position_ids=pos
            )
            logits.append(out.logits[:, -1].float())
    return logits


def _rows(layer: BugStreamingLayer) -> list[BugStreamingLayer]:
    assert layer._rows is not None
    return layer._rows


def test_batch2_equals_two_batch1_runs(tiny_model: LlamaForCausalLM) -> None:
    ids = torch.cat([_prompt(1), _prompt(2)])
    stream = _stream(9, 2)
    both = tiny_cache(tiny_model)
    got = _teacher_forced(tiny_model, both, ids, stream)
    singles = [tiny_cache(tiny_model) for _ in range(2)]
    want = [
        _teacher_forced(tiny_model, c, ids[b : b + 1], stream[b : b + 1])
        for b, c in enumerate(singles)
    ]
    for s in range(N_NEW):
        for b in range(2):
            assert torch.allclose(got[s][b], want[b][s][0], atol=1e-4), (s, b)
            top = want[b][s][0].topk(2).values
            if top[0] - top[1] > 1e-3:  # the greedy choice, wherever it is not a near-tie
                assert int(got[s][b].argmax()) == int(want[b][s][0].argmax()), (s, b)
    # The rows' STATE is the batch-1 state: the same reconstruction (sign-invariant, unlike
    # U alone), the same ring, the same lengths, the same footprint.
    for li, layer in enumerate(both._bug_layers()):
        for b, single in enumerate(singles):
            row, ref = _rows(layer)[b], single._bug_layers()[li]
            assert row.cumulative_length == ref.cumulative_length == T + N_NEW
            assert row.u_k is not None and row.c_k is not None
            assert ref.u_k is not None and ref.c_k is not None
            assert torch.allclose(row.u_k @ row.c_k, ref.u_k @ ref.c_k, atol=1e-4)
            assert row.recent_k is not None and ref.recent_k is not None
            assert torch.allclose(row.recent_k, ref.recent_k, atol=1e-5)
            assert row.stored_state_numel() == ref.stored_state_numel()
    assert both.stored_state_numel() == sum(c.stored_state_numel() for c in singles)
    assert both.workspace_numel() == sum(c.workspace_numel() for c in singles)


def test_readers_fan_out_over_rows(tiny_model: LlamaForCausalLM) -> None:
    cache = tiny_cache(tiny_model, diag_every=1)
    ids = torch.cat([_prompt(3), _prompt(4)])
    with cache.ingesting():  # before the rows exist the layer itself carries the mode
        assert all(la._mode == "ingest" for la in cache._bug_layers())
    _teacher_forced(tiny_model, cache, ids, _stream(10, 2))
    layers = cache._bug_layers()
    for layer in layers:
        rows = _rows(layer)
        assert len(rows) == 2 and all(r._mode == "normal" for r in rows)
        assert layer.get_seq_length() == rows[0].cumulative_length == T + N_NEW
        assert layer.get_mask_sizes(1) == rows[0].get_mask_sizes(1)
        assert layer.attended_length() == rows[0].attended_length()
        assert layer.stored_state_numel() == sum(r.stored_state_numel() for r in rows)
        assert layer.workspace_numel() == sum(r.workspace_numel() for r in rows)
    with cache.frozen_scoring():
        assert all(r._mode == "score" for la in layers for r in _rows(la))
    assert all(r._mode == "normal" for la in layers for r in _rows(la))
    diag = cache.drain_diag()
    assert len(diag) == sum(r._absorbs for la in layers for r in _rows(la)) > 0
    assert all(not r.diag for la in layers for r in _rows(la))
    assert cache.drain_diag() == []  # drained once
    with pytest.raises(NotImplementedError, match="built for batch 2"), torch.no_grad():
        tiny_model(ids[:1, -1:], past_key_values=cache, use_cache=True)
    # Cache.reset -> every layer.reset(): rows dropped, reusable at any batch.
    cache.reset()  # type: ignore[no-untyped-call]
    assert all(la._rows is None and la.cumulative_length == 0 for la in layers)
    _teacher_forced(tiny_model, cache, ids[:1], _stream(11, 1))
    assert all(la._rows is None and la.cumulative_length == T + N_NEW for la in layers)


def test_reset_keeps_the_rows_undrained_diagnostics(tiny_model: LlamaForCausalLM) -> None:
    """The drained-once contract: rows' rows survive a reset and come out of the next drain."""
    cache = tiny_cache(tiny_model, diag_every=1)
    _teacher_forced(tiny_model, cache, torch.cat([_prompt(5), _prompt(6)]), _stream(12, 2))
    n = sum(r._absorbs for la in cache._bug_layers() for r in _rows(la))
    cache.reset()  # type: ignore[no-untyped-call]
    assert len(cache.drain_diag()) == n > 0


def test_ragged_rows_raise(tiny_model: LlamaForCausalLM, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tiny_cache(tiny_model)
    with torch.no_grad():
        tiny_model(torch.cat([_prompt(7), _prompt(8)]), past_key_values=cache, use_cache=True)
    layer = cache._bug_layers()[0]
    row1 = _rows(layer)[1]
    real = row1.update

    def short(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kk, vv = real(k, v)
        return kk[:, :, :-1], vv[:, :, :-1]

    monkeypatch.setattr(row1, "update", short)
    with pytest.raises(NotImplementedError, match="ragged"):
        layer.update(torch.randn(2, 2, 1, 16), torch.randn(2, 2, 1, 16))


def test_run_latency_runs_at_batch_2(tiny_model: LlamaForCausalLM) -> None:
    arms = [
        {"name": "full", "kind": "full", "make": lambda: None},
        {"name": "tiny_bug", "kind": "bug", "make": lambda: tiny_cache(tiny_model)},
    ]
    rows = run_latency(tiny_model, arms, 96, "cpu", chunk=32, n_steps=4, warmup=1, batch=2)
    assert [r["method"] for r in rows] == ["full", "tiny_bug"]
    assert all(r["batch"] == 2 and r["n_steps"] == 3 and r["ms_per_tok_p50"] > 0 for r in rows)
