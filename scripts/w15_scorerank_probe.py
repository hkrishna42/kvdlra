"""Week-15 W-B B2: dump-level score-rank decoupling probe.

QUESTION (pre-registered, plan §Phase 1 W-B). ``_surprise_scores`` scores SLASH
candidates against the FULL tracked basis ``u_k`` -- THE rank-retrieval coupling
(a big basis fits needles -> residual -> 0 -> never selected; measured ladder
r32=92 hard-mean, r128=83, r256=0). Does capping *selection* scoring to the
leading ``score_rank`` basis columns (``u_k[:, :s]``, energy-ordered by the
integrator's per-step SVD) restore a needle's selectability at high STORAGE rank?

CRITICAL DESIGN (locked; the fix vs the w14 phase-3 probe). The w14 needle was
orthogonal-by-construction -- top-1 at any rank, structurally blind to the
coupling. Here the needle must be ABSORBED: stream ~2000 real key columns through
the REAL ``augmented_bug_step`` WITH a distinctive needle column planted
mid-history (~col 1000), THEN score a candidate block containing that same needle
column against the evolved basis -- exactly what SLASH re-scoring does to a
previously-absorbed (warm-up-bypassed or demoted) needle. The needle is a REAL
key column from a DIFFERENT dump (same layer), scaled -- absorbable by
construction, NOT a synthetic orthogonal vector. Surprise is scale-free
(``resid/||k||``), so scaling changes only how strongly the basis absorbs it.

CONTROL (validity gate, evaluated FIRST): at (store 256, score full) the needle
percentile must be <= 50th on most dumps (>= 3/5, both probed layers) -- the
probe must REPRODUCE the measured r256 failure or it measures nothing. If the
control fails at the default needle, an escalation ladder makes the needle MORE
absorbable (scale up / plant repeats) and the change is reported.

PRE-REGISTERED BARS (dump-level = worst case over store in {128,256} x layers
{0,8} at score 32; strict on purpose):
* FUND: needle rank-in-block <= 3 (>= p90) on >= 4/5 dumps, control passing.
* KILL: capped percentile < 75th on >= 3/5 dumps.
(The thresholds are mutually exclusive: a fund-dump's every cell has rank <= 3
of 32 => percentile >= 93.5 > 75.)

Usage::

    uv run python scripts/w15_scorerank_probe.py \
        --dumps dumps/llama3.2-1b --out-json results/w15-scorerank-probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

# The hatchling editable ``.pth`` is flaky on this Mac (see auto-memory /
# docs/*handover*); mirror pytest's ``pythonpath=["src"]`` deterministically.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import Tensor
from transformers import LlamaConfig, LlamaForCausalLM

from kvdlra.cache import BugStreamingCache
from kvdlra.cache.bug_cache import BugStreamingLayer
from kvdlra.integrators.streaming_torch import augmented_bug_step

REPO = Path(__file__).resolve().parents[1]
OUT_DEFAULT = REPO / "results" / "w15-scorerank-probe.json"

DOCS = ("doc63", "doc411", "doc454", "doc637", "doc718")
LAYERS = (0, 8)
STORE_RANKS = (32, 64, 128, 192, 256)
SCORE_RANKS: tuple[int | None, ...] = (1, 8, 16, 32, 64, None)  # None = full basis

N_SINK = 4
BLOCK = 32  # streaming block == candidate block size (31 background cols)
HISTORY = 2000  # real columns streamed into the basis
PLANT_AT = 1000  # needle planted mid-history
CAND_AT = 2000  # candidate background block start (unseen by the basis)
NEEDLE_SRC_COL = 3000  # column taken from the OTHER dump as the needle

# Escalation ladder for the control gate: (scale x median host col norm, repeats).
NEEDLE_LADDER = ((2.0, 1), (2.0, 4), (4.0, 4), (4.0, 8))

# Bars (see module docstring).
FUND_STORES = (128, 256)
FUND_SCORE = 32
FUND_RANK_MAX = 3
FUND_MIN_DUMPS = 4
KILL_PCT = 75.0
KILL_MIN_DUMPS = 3
CONTROL_PCT = 50.0
CONTROL_MIN_DUMPS = 3


# ------------------------------------------------------------------ loading


def _load_kpre(dump_dir: Path, layer: int, n_sink: int) -> Tensor:
    """Pre-RoPE feature-by-token matrix ``(h*d, t)`` for one layer, sinks dropped
    (the w14 phase-3 loader, kept byte-compatible)."""
    blob = torch.load(dump_dir / f"layer_{layer:02d}.pt", weights_only=False)
    kpre = cast(Tensor, blob["K_pre"]).float()  # (h, t, d)
    h, t, d = kpre.shape
    mat = kpre.permute(0, 2, 1).reshape(h * d, t).contiguous()  # (h*d, t)
    return mat[:, n_sink:] if mat.shape[1] > n_sink else mat


def _find_dump_dirs(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for doc in DOCS:
        hits = sorted(root.glob(f"{doc}_*_len4096_rope-both"))
        if not hits:
            raise SystemExit(f"missing dump dir {doc}_*_len4096_rope-both under {root}")
        out[doc] = hits[0]
    return out


# ------------------------------------------------------------------ vehicle

_H, _D = 2, 16


def _tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=_H,
        head_dim=_D,
        max_position_embeddings=4096,
    )
    model = LlamaForCausalLM(cfg)  # type: ignore[no-untyped-call]
    model.config._attn_implementation = "sdpa"
    model.eval()  # type: ignore[no-untyped-call]
    return model


def _make_vehicle() -> BugStreamingLayer:
    """A real ``BugStreamingLayer`` as the scoring vehicle (the w14 pattern):
    ``_surprise_scores`` reads only ``self.u_k`` (+ identity whitening), so the
    REAL deployed method -- including the new ``cap`` path -- scores the cells."""
    cache = BugStreamingCache(
        _tiny_model(),
        rank=32,
        coord_budget=64,
        hh_budget=1,
        hh_select="surprise",
        retention="lowrank_surprise",
    )
    return next(la for la in cache.layers if isinstance(la, BugStreamingLayer))


# ------------------------------------------------------------------ streaming


_FP64_FALLBACKS = 0


def _robust_step(
    u: Tensor | None, b: Tensor | None, block: Tensor, rank: int
) -> tuple[Tensor, Tensor]:
    """One deployed ``augmented_bug_step``; on the (CPU-LAPACK ``gesdd``) SVD
    convergence failure that fp32 sometimes hits at high rank, redo the SAME step
    in float64 and cast back -- data unchanged, occurrences counted."""
    global _FP64_FALLBACKS
    try:
        un, bn, _ = augmented_bug_step(u, b, block, rank)
        return un, bn
    except torch.linalg.LinAlgError:  # type: ignore[attr-defined]
        _FP64_FALLBACKS += 1
        un, bn, _ = augmented_bug_step(
            None if u is None else u.double(),
            None if b is None else b.double(),
            block.double(),
            rank,
        )
        return un.float(), bn.float()


def _stream_basis_with_needle(
    cols: Tensor, needle: Tensor, rank: int, block: int, repeats: int
) -> Tensor:
    """Build a rank-``rank`` basis over ``HISTORY`` real columns with the REAL
    deployed ``augmented_bug_step``, with ``needle`` PLANTED (replacing real
    columns) at ``PLANT_AT``, ``PLANT_AT``+50, ... -- so the evolved basis has
    absorbed the needle exactly as a streamed cache would have."""
    hist = cols[:, :HISTORY].clone()
    for r in range(repeats):
        col = PLANT_AT + 50 * r
        assert col < HISTORY
        hist[:, col] = needle
    u: Tensor | None = None
    b: Tensor | None = None
    for start in range(0, HISTORY, block):
        u, b = _robust_step(u, b, hist[:, start : start + block], rank)
    assert u is not None
    return u


def _make_needle(host: Tensor, donor: Tensor, scale: float) -> Tensor:
    """A REAL key column from the donor dump (same layer), rescaled to ``scale`` x
    the host's median column norm -- distinctive but absorbable by construction."""
    src = donor[:, NEEDLE_SRC_COL].clone()
    host_norm = float(host[:, :HISTORY].norm(dim=0).median())
    return cast(Tensor, src / src.norm() * (scale * host_norm))


def _score_cell(
    vehicle: BugStreamingLayer, basis: Tensor, cand: Tensor, needle_col: int, cap: int | None
) -> dict[str, Any]:
    """Score the candidate block with the REAL ``_surprise_scores`` (capped or
    full) and report the needle's standing among the 31 real background columns."""
    vehicle.u_k = basis
    with torch.no_grad():
        s = vehicle._surprise_scores(vehicle._whiten_key(cand), cap=cap)
    bg = torch.ones(int(cand.shape[1]), dtype=torch.bool)
    bg[needle_col] = False
    ns = float(s[needle_col])
    bg_s = s[bg]
    pct = 100.0 * float((bg_s < ns).float().mean())
    rank_in_block = 1 + int((bg_s > ns).sum())
    return {
        "needle_surprise": round(ns, 4),
        "percentile": round(pct, 1),
        "rank_in_block": rank_in_block,
        "bg_max": round(float(bg_s.max()), 4),
        "bg_median": round(float(bg_s.median()), 4),
        "margin": round(ns - float(bg_s.max()), 4),
    }


# ------------------------------------------------------------------ the grid


def _run_grid(
    dump_dirs: dict[str, Path], scale: float, repeats: int
) -> dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]:
    """grid[doc][layer][store][score] -> cell metrics. Basis built once per
    (doc, layer, store); every score cap reuses it."""
    docs = list(DOCS)
    grid: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {}
    vehicle = _make_vehicle()
    for i, doc in enumerate(docs):
        donor_doc = docs[(i + 1) % len(docs)]
        grid[doc] = {}
        for layer in LAYERS:
            host = _load_kpre(dump_dirs[doc], layer, N_SINK)
            donor = _load_kpre(dump_dirs[donor_doc], layer, N_SINK)
            needle = _make_needle(host, donor, scale)
            cand = host[:, CAND_AT : CAND_AT + BLOCK].clone()
            needle_col = BLOCK // 2
            cand[:, needle_col] = needle
            per_store: dict[str, dict[str, dict[str, Any]]] = {}
            for store in STORE_RANKS:
                basis = _stream_basis_with_needle(host, needle, store, BLOCK, repeats)
                per_store[str(store)] = {
                    ("full" if cap is None else str(cap)): _score_cell(
                        vehicle, basis, cand, needle_col, cap
                    )
                    for cap in SCORE_RANKS
                }
            grid[doc][f"layer{layer}"] = per_store
    return grid


def _control_cells(grid: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Needle percentile at (store 256, score full) per (doc, layer)."""
    return {
        doc: {layer: float(cells["256"]["full"]["percentile"]) for layer, cells in by_layer.items()}
        for doc, by_layer in grid.items()
    }


def _control_pass(grid: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """A dump reproduces the measured r256 failure iff percentile <= 50 on BOTH
    probed layers at (store 256, score full); gate = >= 3/5 dumps."""
    cells = _control_cells(grid)
    per_dump = {doc: max(v.values()) <= CONTROL_PCT for doc, v in cells.items()}
    n = sum(per_dump.values())
    return n >= CONTROL_MIN_DUMPS, {
        "cells_pct_at_store256_full": cells,
        "dump_reproduces_failure": per_dump,
        "n_reproducing": n,
        "bar": f"percentile <= {CONTROL_PCT} on both layers, on >= {CONTROL_MIN_DUMPS}/5 dumps",
    }


def _evaluate_bars(grid: dict[str, Any]) -> dict[str, Any]:
    """FUND/KILL on the dump-level WORST capped cell (store in {128,256} x both
    layers, score 32): max rank-in-block / min percentile."""
    per_dump: dict[str, dict[str, float | int]] = {}
    for doc, by_layer in grid.items():
        ranks: list[int] = []
        pcts: list[float] = []
        for cells in by_layer.values():
            for store in FUND_STORES:
                cell = cells[str(store)][str(FUND_SCORE)]
                ranks.append(int(cell["rank_in_block"]))
                pcts.append(float(cell["percentile"]))
        per_dump[doc] = {"worst_rank": max(ranks), "worst_pct": min(pcts)}
    fund_dumps = [d for d, v in per_dump.items() if int(v["worst_rank"]) <= FUND_RANK_MAX]
    kill_dumps = [d for d, v in per_dump.items() if float(v["worst_pct"]) < KILL_PCT]
    return {
        "aggregation": (
            f"dump-level worst case over store in {list(FUND_STORES)} x layers "
            f"{list(LAYERS)} at score {FUND_SCORE}"
        ),
        "per_dump_worst_capped": per_dump,
        "fund_bar": (
            f"rank-in-block <= {FUND_RANK_MAX} (>= p90) on >= {FUND_MIN_DUMPS}/5 dumps, "
            "control passing"
        ),
        "kill_bar": f"capped percentile < {KILL_PCT} on >= {KILL_MIN_DUMPS}/5 dumps",
        "fund_dumps": fund_dumps,
        "kill_dumps": kill_dumps,
    }


# ------------------------------------------------------------------ printing


def _print_grid(grid: dict[str, Any]) -> None:
    caps = ["full" if c is None else str(c) for c in SCORE_RANKS]
    for layer in LAYERS:
        print(f"\n  needle percentile (rank-in-block) -- layer {layer}")
        header = "  doc      store " + "".join(f"{('s' + c):>12}" for c in caps)
        print(header)
        for doc in DOCS:
            for store in STORE_RANKS:
                cells = grid[doc][f"layer{layer}"][str(store)]
                row = "".join(
                    f"{cells[c]['percentile']:>8.1f} ({cells[c]['rank_in_block']:>2})" for c in caps
                )
                print(f"  {doc:<8} r{store:<4} {row}")


# ---------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", type=Path, default=REPO / "dumps" / "llama3.2-1b")
    ap.add_argument("--out-json", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    dump_dirs = _find_dump_dirs(args.dumps)

    # CONTROL FIRST: walk the needle ladder until the probe reproduces the
    # measured (store 256, score full) failure; record every attempt honestly.
    ladder_log: list[dict[str, Any]] = []
    grid: dict[str, Any] | None = None
    chosen: tuple[float, int] | None = None
    control: dict[str, Any] = {}
    for scale, repeats in NEEDLE_LADDER:
        print(f"[control] needle scale={scale} repeats={repeats} ...")
        g = _run_grid(dump_dirs, scale, repeats)
        ok, detail = _control_pass(g)
        ladder_log.append(
            {"scale": scale, "repeats": repeats, "control_pass": ok, "detail": detail}
        )
        print(
            f"[control] reproduces r256 failure on {detail['n_reproducing']}/5 dumps "
            f"(need >= {CONTROL_MIN_DUMPS}) -> {'PASS' if ok else 'FAIL'}"
        )
        if ok:
            grid, chosen, control = g, (scale, repeats), detail
            break
    if grid is None or chosen is None:
        # Report the strongest attempt; the probe is VOID without a passing control.
        result: dict[str, Any] = {
            "verdict": "VOID",
            "verdict_reason": (
                "control never passed: the probe could not reproduce the measured "
                "(store 256, score full) selection failure at any needle-ladder "
                "config -- the cells measure nothing; redesign before reading them"
            ),
            "needle_ladder": ladder_log,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True))
        print("VERDICT = VOID (control failed at every needle config)")
        raise SystemExit(1)

    scale, repeats = chosen
    if (scale, repeats) != NEEDLE_LADDER[0]:
        print(
            f"[control] NOTE: needle redesigned from default {NEEDLE_LADDER[0]} to "
            f"scale={scale}, repeats={repeats} to make it absorbable enough to "
            "reproduce the failure (reported in JSON)."
        )

    _print_grid(grid)
    bars = _evaluate_bars(grid)

    n_fund, n_kill = len(bars["fund_dumps"]), len(bars["kill_dumps"])
    if n_fund >= FUND_MIN_DUMPS:
        verdict = "FUND"
    elif n_kill >= KILL_MIN_DUMPS:
        verdict = "KILL"
    else:
        verdict = "MIXED"

    print(f"\n  control: {control['n_reproducing']}/5 dumps reproduce ({control['bar']})")
    print(f"  FUND bar: {bars['fund_bar']}")
    print(f"  KILL bar: {bars['kill_bar']}")
    for doc, v in bars["per_dump_worst_capped"].items():
        print(f"    {doc}: worst capped rank={v['worst_rank']}  worst pct={v['worst_pct']}")
    print(f"  fund dumps: {n_fund}/5 {bars['fund_dumps']}  kill dumps: {n_kill}/5")
    print(f"VERDICT = {verdict}")

    result = {
        "design": (
            "needle = real key column from the NEXT dump (same layer), scaled to "
            f"{scale}x median host col norm, planted {repeats}x from col {PLANT_AT} "
            f"(step 50) inside {HISTORY} streamed real cols (augmented_bug_step, "
            f"block {BLOCK}); candidate = {BLOCK} unseen real cols from col "
            f"{CAND_AT} with the needle at {BLOCK // 2}; scored with the REAL "
            "_surprise_scores(cap=score_rank)"
        ),
        "needle": {"scale": scale, "repeats": repeats, "ladder": ladder_log},
        "grid_axes": {
            "store_ranks": list(STORE_RANKS),
            "score_ranks": ["full" if c is None else c for c in SCORE_RANKS],
            "dumps": list(DOCS),
            "layers": list(LAYERS),
        },
        "control": control,
        "bars": bars,
        "verdict": verdict,
        "fp64_step_fallbacks": _FP64_FALLBACKS,
        "grid": grid,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.out_json.relative_to(REPO)}")


if __name__ == "__main__":
    main()
