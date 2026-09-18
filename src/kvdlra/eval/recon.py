"""§4.1's reconstruction study, scored on the representation a method actually STORES.

The published §4.1 projected the *whole* matrix onto each tracker's final basis
(``U U^T M``), which credits every method with coordinates it never kept and flatters the
ones whose basis moves most (CODE_AUDIT.md Q2/Q5). Here each method is driven through the
one ``step(u, b, block) -> (u, b, rot)`` contract and scored on its stored pair ``(u, c)``
with the carry the streaming cache itself runs (``bug_cache.py:_absorb_columns``: rotate
the held coordinates by ``rot``, then append the new block's): ``c <- rot @ c``,
``c = cat(c, u^T block)``. So ``stored_error(m, "isvd", r)`` IS the gist fidelity of a
cache with no exact tier, no sinks and no ring -- GATES G1 line 3 pins the two together.

The controls answer "does the tracking do anything?": ``frozen_prefill_svd`` is the best
fixed basis a prefill window can give (no tracking), ``random_basis`` is no information at
all, and ``svd_oracle`` is the per-sequence offline bound. Every cell of the study shares
one Haar draw for ``random_basis`` (``seed=0``), so the spread around that line is document
variation and not draw-to-draw noise. ``fd2`` is Frequent Directions at ``ell = 2r``: its
stored representation is ``2r`` wide, so its bound is the rank-``2r`` oracle, not the
rank-``r`` one -- every row carries ``stored_rank = u.shape[1]`` for exactly that reason
and the figure plots by stored width.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, Literal, get_args

import torch
from torch import Tensor

from kvdlra.eval.records import write_jsonl
from kvdlra.tracker import TRACKERS

__all__ = ["METHODS", "OJA_GRID", "Method", "load_stream", "run_study", "stored_error", "tune_oja"]

Method = Literal["svd_oracle", "isvd", "fd", "fd2", "oja", "frozen_prefill_svd", "random_basis"]
METHODS: tuple[Method, ...] = get_args(Method)
SINKS = 4  # the first 4 tokens are attention sinks, kept verbatim by every arm
PREFILL = 0.25  # fraction of T the frozen control fits its basis on
# Week-2 tuned Oja's schedule to eta0=20, decay=0.03, on BOTH grid boundaries (CODE_AUDIT
# Q5/Q6), so its deficit was an upper bound rather than a measurement. Measured here on the
# tuning cell (1B layer 8, pre-RoPE keys, r=16, doc63+doc718, stored-representation error):
# the surface falls monotonically AWAY from that corner toward smaller steps -- 0.821 at
# (20, 0.03), 0.405 at (5, 0.3) -- so a grid extending past it, as Week 2's boundary
# suggested, brackets nothing. This one brackets the measured minimum on all four sides
# (eta0=5 between 2.5 and 10, decay=0.3 between 0.1 and 1.0). `tune_oja` raises when a
# dataset's optimum still lands on an edge; widen it again, and record the surface.
OJA_GRID: dict[str, list[float]] = {
    "eta0": [0.6, 1.25, 2.5, 5.0, 10.0],
    "decay": [0.03, 0.1, 0.3, 1.0],
}
TUNE_RANK, TUNE_LAYER = 16, 8  # where Oja's schedule is tuned: once per kv, reused at every rank


def load_stream(layer_pt: Path, kv: str) -> tuple[Tensor, str]:
    """One layer dump as the ``(n, T - SINKS)`` fp32 matrix the study tracks, plus the kv
    it turned out to be: ``"k_pre"`` unless the dump predates pre-RoPE capture, in which
    case its post-RoPE keys come back labelled ``"k_post"``.

    ``(heads, T, head_dim) -> (heads * head_dim, T)``, head-major, so one basis is shared
    across heads (the Week-2 convention, CODE_AUDIT.md:185); the sink columns are dropped
    because no arm ever compresses them.
    """
    blob: dict[str, Tensor] = torch.load(layer_pt, map_location="cpu", weights_only=True)
    if kv == "v":
        key = "V"
    else:
        key = "K_pre" if "K_pre" in blob else "K"
        kv = "k_pre" if key == "K_pre" else "k_post"
    x = blob[key]
    heads, t, dim = x.shape
    return x.permute(0, 2, 1).reshape(heads * dim, t)[:, SINKS:].float().contiguous(), kv


def _stored(
    m: Tensor,
    method: Method,
    r: int,
    block: int = 16,
    *,
    oja: tuple[float, float] | None = None,
    prefill: float = PREFILL,
    min_sv_frac: float = 0.0,
) -> tuple[float, Tensor]:
    """``(||m - u c||_F / ||m||_F, u)`` -- the relative error of the pair ``(u, c)`` a
    method ends the stream with, and the basis it stored it in (``u.shape[1]`` is the
    stored width, which ``fd2`` doubles and the ``min_sv_frac`` floor can shrink).

    ``prefill`` is a FRACTION of ``T`` (the frozen control's window); ``min_sv_frac`` is
    the ``isvd`` singular-value floor; ``oja`` is that tracker's ``(eta0, decay)``, which
    it has no default for. ``random_basis`` is ONE Haar draw at seed 0, shared by every
    rank and document -- that is the control's design (module docstring), so the draw is
    not a knob.
    """
    n, t = m.shape
    u: Tensor | None = None
    c: Tensor | None = None
    if method == "svd_oracle":
        left, s, vh = torch.linalg.svd(m, full_matrices=False)
        k = min(r, int(s.shape[0]))
        u, c = left[:, :k], s[:k].unsqueeze(1) * vh[:k]
    elif method == "random_basis":
        g = torch.Generator().manual_seed(0)
        u = torch.linalg.qr(torch.randn(n, min(r, n), generator=g))[0].to(m)
        c = u.mT @ m
    elif method == "frozen_prefill_svd":
        p = max(1, min(t, round(prefill * t)))
        u = torch.linalg.svd(m[:, :p], full_matrices=False)[0][:, :r]
        c = u.mT @ m
    else:
        if method == "oja" and oja is None:
            raise ValueError("method 'oja' has no default schedule: pass oja=(eta0, decay)")
        step = TRACKERS["fd" if method == "fd2" else method]
        cap = 2 * r if method == "fd2" else r
        b = None
        seen = 0
        for start in range(0, t, block):
            blk = m[:, start : start + block]
            if method == "oja":
                assert oja is not None
                u, b, rot = step(u, b, blk, cap, n_seen=seen, eta0=oja[0], decay=oja[1])
            elif method == "isvd":
                u, b, rot = step(u, b, blk, cap, theta=None, min_sv_frac=min_sv_frac)
            else:
                u, b, rot = step(u, b, blk, cap)
            c = u.mT @ blk if c is None else torch.cat([rot @ c, u.mT @ blk], dim=1)
            seen += int(blk.shape[1])
    assert u is not None and c is not None  # every branch assigns; t >= 1, so the loop ran
    return float(torch.linalg.norm(m - u @ c) / torch.linalg.norm(m)), u


def stored_error(m: Tensor, method: Method, r: int, block: int = 16, **kw: Any) -> float:
    """``||m - u c||_F / ||m||_F`` for one method on one ``(n, T)`` stream.

    The scalar half of :func:`_stored`, whose keyword options (``oja``, ``prefill``,
    ``min_sv_frac``) this forwards.
    """
    return _stored(m, method, r, block, **kw)[0]


def tune_oja(docs: list[Tensor], r: int, grid: dict[str, list[float]]) -> tuple[float, float]:
    """Oja's ``(eta0, decay)`` minimizing the mean stored error over held-out ``docs``.

    Raises if the argmin sits on a grid edge: an unbracketed optimum makes the arm's
    deficit an upper bound rather than a measurement (CODE_AUDIT.md Q5/Q6). Widen the grid.
    """
    etas, decays = grid["eta0"], grid["decay"]
    err = {
        (i, j): sum(stored_error(d, "oja", r, oja=(e, dec)) for d in docs) / len(docs)
        for i, e in enumerate(etas)
        for j, dec in enumerate(decays)
    }
    i, j = min(err, key=lambda k: err[k])
    if i in (0, len(etas) - 1) or j in (0, len(decays) - 1):
        raise ValueError(
            f"oja optimum on the grid boundary (eta0={etas[i]}, decay={decays[j]}): widen the grid"
        )
    return etas[i], decays[j]


def _doc_dirs(d: Path) -> list[Path]:
    """One doc dump, or every doc dump under a model directory."""
    if (d / "layer_00.pt").is_file():
        return [d]
    return sorted(p.parent for p in d.glob("*/layer_00.pt"))


def run_study(
    dump_dir: str | Path | Sequence[Path],
    ranks: Sequence[int],
    methods: Sequence[Method] = METHODS,
    layers: Sequence[int] | str = "all",
    kv: Sequence[str] = ("k_pre", "v"),
    blocks: Sequence[int] = (16, 128),
    *,
    out: Path,
    oja: tuple[float, float] | None = None,
    tune_docs: Sequence[Path] = (),
    min_sv_frac: Sequence[float] = (0.0,),
) -> None:
    """Score every method on every cell of the grid into ``out/recon.jsonl`` (+ provenance).

    ``dump_dir`` is one doc dump, a directory of them, or an explicit list. ``oja`` pins
    one schedule for every kv; without it, one is tuned PER KV on ``tune_docs`` at rank
    ``TUNE_RANK`` / layer ``TUNE_LAYER`` and reused at every rank -- keys and values are
    different matrix families and Oja's optimum moves between them (CODE_AUDIT.md Q5).
    ``min_sv_frac`` sweeps the ``isvd`` floor only.
    """
    t0, started = time.perf_counter(), datetime.now(UTC).isoformat(timespec="seconds")
    # str included: iterating one as a Sequence[Path] would silently walk its characters.
    docs = _doc_dirs(Path(dump_dir)) if isinstance(dump_dir, str | Path) else [*map(Path, dump_dir)]
    schedules: dict[str, tuple[float, float]] = {}
    for want in kv if "oja" in methods else ():
        if oja is not None:
            schedules[want] = oja
            continue
        if not tune_docs:
            raise ValueError("method 'oja' needs oja=(eta0, decay) or tune_docs to tune it on")
        mats = [load_stream(p / f"layer_{TUNE_LAYER:02d}.pt", want)[0] for p in tune_docs]
        schedules[want] = tune_oja(mats, TUNE_RANK, OJA_GRID)
    tuning = {
        w: {
            "eta0": sched[0],
            "decay": sched[1],
            "schedule_given": oja is not None,
            "docs": [] if oja is not None else [p.name for p in tune_docs],
            "rank": TUNE_RANK,
            "layer": TUNE_LAYER,
            "reused_at_every_rank": True,
        }
        for w, sched in schedules.items()
    }

    specs = [(me, f) for me in methods for f in (min_sv_frac if me == "isvd" else (0.0,))]
    rows: list[dict[str, object]] = []
    for doc in docs:
        meta = doc / "meta.json"
        model = str(json.loads(meta.read_text())["model"]) if meta.is_file() else doc.name
        idx = (
            sorted(int(p.stem.split("_")[1]) for p in doc.glob("layer_*.pt"))
            if isinstance(layers, str)
            else list(layers)
        )
        for li in idx:
            for want in kv:
                m, got = load_stream(doc / f"layer_{li:02d}.pt", want)
                sched = schedules.get(want)
                for block, r, (method, msf) in product(blocks, ranks, specs):
                    err, u = _stored(m, method, r, block, oja=sched, min_sv_frac=msf)
                    rows.append(
                        {
                            "model": model,
                            "doc": doc.name,
                            "layer": li,
                            "kv": got,
                            "method": method,
                            "rank": r,
                            "stored_rank": int(u.shape[1]),
                            "block": block,
                            "err": err,
                            "min_sv_frac": msf,
                            "oja_eta0": sched[0] if method == "oja" and sched else None,
                            "oja_decay": sched[1] if method == "oja" and sched else None,
                        }
                    )
        # Rewritten after every document, not once at the end: the full grid is hours of
        # CPU and a crash in the last one would otherwise discard every finished document.
        # provenance.json is written only on a clean finish, so a partial file is visibly
        # partial.
        write_jsonl(out / "recon.jsonl", rows)
        print(f"[{doc.name}: {len(rows)} rows, {time.perf_counter() - t0:.0f}s]", flush=True)
    manifest = docs[0].parent.parent / f"{docs[0].parent.name}.sha256"
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "started_at": started,
                "wall_clock_s": round(time.perf_counter() - t0, 1),
                "git_sha": subprocess.run(
                    ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,  # a study whose provenance says `git_sha: ""` is not provenance
                ).stdout.strip(),
                "dump_sha256_manifest": {
                    "path": str(manifest),
                    "sha256": sha256(manifest.read_bytes()).hexdigest()
                    if manifest.is_file()
                    else None,
                },
                "grid": {
                    "docs": [d.name for d in docs],
                    "ranks": list(ranks),
                    "methods": list(methods),
                    "layers": "all" if isinstance(layers, str) else list(layers),
                    "kv": list(kv),
                    "blocks": list(blocks),
                    "min_sv_frac": list(min_sv_frac),
                    "prefill": PREFILL,
                    "oja_grid": OJA_GRID,
                },
                "oja_tuning": tuning,
                "rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"[wrote {out}/recon.jsonl: {len(rows)} rows in {time.perf_counter() - t0:.0f}s]")
