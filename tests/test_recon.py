"""§4.1 scored on what a cache actually STORES (CODE_AUDIT Q2/Q5).

The published §4.1 re-projected the whole matrix onto each tracker's final basis, which
credits a method with coordinates it never kept. Here every method is driven through the
same ``step`` contract and scored on its stored pair ``(u, c)``, with the coordinate carry
the streaming cache itself runs. The first test is GATES G1 line 3: the study's ``isvd``
number IS the cache's own rot-carried error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from kvdlra.eval.recon import load_stream, stored_error, tune_oja


def _stream(n: int = 96, t: int = 640, k: int = 12, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(n, k, generator=g))[0]
    scale = torch.linspace(4.0, 0.5, k).unsqueeze(1)
    m: Tensor = q @ (torch.randn(k, t, generator=g) * scale) + 0.05 * torch.randn(n, t, generator=g)
    return m


def test_isvd_stored_error_equals_cache_rot_carried_error() -> None:
    """GATES G1: the study's isvd number IS the cache's own gist fidelity (to 1e-6)."""
    from kvdlra.tracker.isvd import isvd_step

    m = _stream()
    u: Tensor | None = None
    b: Tensor | None = None
    c: Tensor | None = None
    for s in range(0, m.shape[1], 16):
        blk = m[:, s : s + 16]
        u, b, rot = isvd_step(u, b, blk, 8)
        c = u.mT @ blk if c is None else torch.cat([rot @ c, u.mT @ blk], dim=1)
    assert u is not None and c is not None
    cache_err = float(torch.linalg.norm(m - u @ c) / torch.linalg.norm(m))
    assert abs(stored_error(m, "isvd", 8, block=16) - cache_err) < 1e-6


def test_oracle_is_a_lower_bound() -> None:
    m = _stream()
    o = stored_error(m, "svd_oracle", 8)
    for meth in ("isvd", "fd", "oja", "frozen_prefill_svd", "random_basis"):
        assert stored_error(m, meth, 8, prefill=0.25, oja=(20.0, 0.03)) >= o - 1e-9, meth
    # fd2 stores 2r columns, so the rank-r oracle is not its bound -- the rank-2r one is.
    assert stored_error(m, "fd2", 8) >= stored_error(m, "svd_oracle", 16) - 1e-9


def test_frozen_basis_is_worse_on_a_drifting_stream() -> None:
    a, b = _stream(seed=1), _stream(seed=2)
    m = torch.cat([a, b], dim=1)  # subspace changes half-way
    frozen = stored_error(m, "frozen_prefill_svd", 8, prefill=0.5)
    assert frozen > stored_error(m, "isvd", 8) + 0.05


def test_tune_oja_rejects_boundary_optimum() -> None:
    docs = [_stream(seed=s) for s in (3, 4)]
    with pytest.raises(ValueError):
        tune_oja(docs, r=8, grid={"eta0": [20.0], "decay": [0.03]})  # 1 point = all boundary


def test_dump_loader_reshapes_head_major_and_drops_sinks(tmp_path: Path) -> None:
    """(heads, T, head_dim) -> (heads*head_dim, T-4): one shared basis per layer over
    head-major features, the 4 sink columns dropped (CODE_AUDIT.md:185)."""
    x = torch.randn(2, 40, 8)
    torch.save({"K_pre": x, "V": 2.0 * x}, tmp_path / "layer_00.pt")

    m, kv = load_stream(tmp_path / "layer_00.pt", "k_pre")
    assert (tuple(m.shape), kv) == ((16, 36), "k_pre")
    assert m[0 * 8 + 3, 0] == x[0, 4, 3]  # head 0, dim 3, first kept column == token 4
    assert m[1 * 8 + 2, 5] == x[1, 9, 2]  # head 1, dim 2, column 5 == token 9
    assert load_stream(tmp_path / "layer_00.pt", "v")[0][0, 0] == 2.0 * x[0, 4, 0]

    torch.save({"K": x}, tmp_path / "layer_01.pt")  # a post-RoPE-only dump
    assert load_stream(tmp_path / "layer_01.pt", "k_pre")[1] == "k_post"
