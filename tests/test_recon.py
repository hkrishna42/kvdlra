"""§4.1 scored on what a cache actually STORES (CODE_AUDIT Q2/Q5).

The published §4.1 re-projected the whole matrix onto each tracker's final basis, which
credits a method with coordinates it never kept. Here every method is driven through the
same ``step`` contract and scored on its stored pair ``(u, c)``, with the coordinate carry
the streaming cache itself runs. The first test is GATES G1 line 3: the study's ``isvd``
number IS the cache's own rot-carried error.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch
from torch import Tensor

from kvdlra.eval.recon import METHODS, load_stream, stored_error, tune_oja
from kvdlra.eval.records import read_jsonl


def _stream(n: int = 96, t: int = 640, k: int = 12, seed: int = 0, noise: float = 0.05) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(n, k, generator=g))[0]
    scale = torch.linspace(4.0, 0.5, k).unsqueeze(1)
    m: Tensor = q @ (torch.randn(k, t, generator=g) * scale)
    return m + noise * torch.randn(n, t, generator=g)  # noise=0 -> exactly rank k


def _cache_carry(m: Tensor, r: int, block: int = 16, min_sv_frac: float = 0.0) -> tuple[float, int]:
    """The streaming cache's own gist error and stored width, from `bug_cache.py`'s carry."""
    from kvdlra.tracker.isvd import isvd_step

    u: Tensor | None = None
    b: Tensor | None = None
    c: Tensor | None = None
    for s in range(0, m.shape[1], block):
        blk = m[:, s : s + block]
        u, b, rot = isvd_step(u, b, blk, r, theta=None, min_sv_frac=min_sv_frac)
        c = u.mT @ blk if c is None else torch.cat([rot @ c, u.mT @ blk], dim=1)
    assert u is not None and c is not None
    return float(torch.linalg.norm(m - u @ c) / torch.linalg.norm(m)), int(u.shape[1])


def test_isvd_stored_error_equals_cache_rot_carried_error() -> None:
    """GATES G1: the study's isvd number IS the cache's own gist fidelity (to 1e-6)."""
    m = _stream()
    cache_err, cols = _cache_carry(m, 8)
    assert cols == 8
    assert abs(stored_error(m, "isvd", 8, block=16) - cache_err) < 1e-6

    # And with the Week-17 singular-value floor, which the study sweeps: on a rank-4
    # stream it stops the basis short of the cap, and the two paths still agree exactly.
    flat = _stream(k=4, noise=0.0)
    floor_err, cols = _cache_carry(flat, 8, min_sv_frac=0.01)
    assert cols < 8
    assert stored_error(flat, "isvd", 8, block=16, min_sv_frac=0.01) - floor_err == 0.0


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


def test_run_study_tunes_oja_once_per_kv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Oja's optimum moves between matrix families (CODE_AUDIT Q5), so keys and values get
    their own schedule: one tuning call each, stamped on that kv's rows and in provenance."""
    from kvdlra.eval import recon

    doc = tmp_path / "model" / "doc0_len40"
    doc.mkdir(parents=True)
    x = torch.randn(2, 40, 8)
    torch.save({"K_pre": x, "V": 2.0 * x}, doc / f"layer_{recon.TUNE_LAYER:02d}.pt")

    calls: list[int] = []

    def fake_tune(docs: list[Tensor], r: int, grid: dict[str, list[float]]) -> tuple[float, float]:
        calls.append(len(docs))
        return 1.0 + len(calls), 0.1 * len(calls)

    monkeypatch.setattr(recon, "tune_oja", fake_tune)
    recon.run_study(
        [doc],
        ranks=(4,),
        methods=("oja",),
        layers=[recon.TUNE_LAYER],
        kv=("k_pre", "v"),
        blocks=(16,),
        out=tmp_path / "out",
        tune_docs=[doc],
    )

    assert calls == [1, 1]  # once per kv, not once for kv[0] reused on both
    rows = read_jsonl(tmp_path / "out" / "recon.jsonl")
    assert {str(r["kv"]): (r["oja_eta0"], r["oja_decay"]) for r in rows} == {
        "k_pre": (2.0, 0.1),
        "v": (3.0, 0.2),
    }
    tuning = json.loads((tmp_path / "out" / "provenance.json").read_text())["oja_tuning"]
    assert [tuning[k]["eta0"] for k in ("k_pre", "v")] == [2.0, 3.0]


def test_rank_sweep_fails_loud_on_a_missing_series(tmp_path: Path) -> None:
    """A study run without the floor sweep has no `isvd_f0.01` rows, and the figure would
    silently draw that dashed line as nothing at all. Say so, and exit non-zero."""
    import figures

    src = tmp_path / "recon.jsonl"
    src.write_text(
        "".join(
            json.dumps(
                {
                    "doc": "doc0", "kv": "k_pre", "method": meth, "rank": 8, "stored_rank": 8,
                    "block": 16, "err": 0.5, "min_sv_frac": 0.0,
                }
            )
            + "\n"
            for meth in METHODS
        )
    )  # fmt: skip
    with pytest.raises(SystemExit, match=re.escape("isvd_f0.01")):
        figures.fig_rank_sweep(src, tmp_path / "rank_sweep.pdf")
    # and an empty selection, rather than a figure with zero panels
    with pytest.raises(SystemExit, match="no rows"):
        figures.fig_rank_sweep(src, tmp_path / "rank_sweep.pdf", block=128)
