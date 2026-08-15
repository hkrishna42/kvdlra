"""Palu low-rank-projection press: per-head-group truncated-SVD compression.

Unit-tests the ``_compress_tensor`` step directly on random KV blocks (no model),
mirroring ``tests/test_bug_press.py``: shape preserved; ``rank_ratio = 1`` is
lossless (rank = head_dim = full for a T > D block); ``rank_ratio < 1`` is a lossy
but reasonable low-rank approximation; grouped low-rank works; validation.
"""

from __future__ import annotations

import pytest
import torch

from kvdlra.press import PaluPress

H, D = 2, 16


def _kv(t: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, H, t, D, generator=g)


def test_palu_shape_preserved() -> None:
    out = PaluPress(rank_ratio=0.5)._compress_tensor(_kv(40))
    assert out.shape == (1, H, 40, D)


def test_palu_full_rank_is_lossless() -> None:
    """rank_ratio=1 => r=head_dim => truncated SVD of a (D, T>D) block is exact."""
    x = _kv(48)
    out = PaluPress(rank_ratio=1.0)._compress_tensor(x)
    assert torch.allclose(out, x, atol=1e-4)


def test_palu_low_rank_is_lossy_but_reasonable() -> None:
    x = _kv(64)
    out = PaluPress(rank_ratio=0.25)._compress_tensor(x)
    assert not torch.allclose(out, x, atol=1e-2)
    assert (out - x).norm() / x.norm() < 0.95  # a real approximation, not garbage


def test_sink_columns_exact() -> None:
    """Week-15 audit fix: the ``n_sink`` leading token columns survive
    ``_compress_tensor`` BIT-EXACTLY (K and V take the same path at this level;
    through the full ``compress()`` the K sinks additionally ride BUGPress's
    pre-RoPE un/re-rotate round trip, so they are allclose rather than
    bit-equal there). Non-sink columns are still genuinely compressed -- the
    exemption does not leak. Pre-fix, Palu was the ONLY frontier arm that
    low-ranked the sinks (the incoherent 1B ppl signature)."""
    press = PaluPress(rank_ratio=0.25)
    x = _kv(64)
    out = press._compress_tensor(x)
    s = press.n_sink
    assert s == 4  # inherits BUGPress's default sink exemption
    assert torch.equal(out[:, :, :s], x[:, :, :s])
    assert not torch.allclose(out[:, :, s:], x[:, :, s:], atol=1e-2)


def test_nonsink_block_is_lowrank() -> None:
    """The compressed non-sink block has numerical rank <= r = round(rank_ratio
    * D) per head (group=1): the sink carve-out keeps Eckart--Young truncation
    on exactly the columns ``n_sink:`` (rank spent there, not on the sinks)."""
    rr = 0.25
    press = PaluPress(rank_ratio=rr)
    x = _kv(64)
    out = press._compress_tensor(x)
    r = round(rr * D)
    for h in range(H):
        blk = out[0, h, press.n_sink :, :].T.to(torch.float32)  # (D, T - n_sink)
        sv = torch.linalg.svdvals(blk)
        assert int((sv > 1e-5 * float(sv[0])).sum()) <= r


def test_palu_grouped_low_rank_runs() -> None:
    out = PaluPress(rank_ratio=0.5, group=2)._compress_tensor(_kv(40))  # both heads share
    assert out.shape == (1, H, 40, D)


def test_palu_validation() -> None:
    with pytest.raises(ValueError):
        PaluPress(rank_ratio=0.0)
    with pytest.raises(ValueError):
        PaluPress(rank_ratio=1.5)
    with pytest.raises(ValueError):
        PaluPress(rank_ratio=0.5, group=0)
