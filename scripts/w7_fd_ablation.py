"""Week-7 dominance follow-up: is the SUBSPACE TRACKER leaving accuracy on the table?

Swaps the streaming subspace tracker (BUG, a DLRA integrator) for **Frequent
Directions** (a deterministic non-DLRA streaming sketch) and compares both to the
truncated-SVD oracle and incremental SVD, on the same real KV streams and the
same relative-Frobenius reconstruction metric BUG was validated with in Week 2.

The honest question (``docs/week7-dominance.md`` §"DLRA without BUG"): BUG tracks
to within ~1-3% of the oracle -- does a different, arguably better-matched
streaming-sketch algorithm close that last gap, or is BUG already at the ceiling?
If FD ties BUG at matched memory, the tracker choice is settled and the real
levers are elsewhere (the overhead floor / the codebook direction).

Matched memory (the fair axis). BUG at rank ``r`` stores ``U`` (n x r) + core
(r x r) ~= ``n*r + r^2`` floats. FD stores a sketch of ``ell`` rows = ``ell*n``.
We report FD at two settings: **matched memory** (``ell = r + round(r^2/n)`` so
``ell*n ~= n*r + r^2``, projecting at rank ``r``) -- the apples-to-apples number
-- and the textbook ``ell = 2r`` (~2x BUG's memory) as an upper reference for
FD's capability. Oracle and incremental SVD are the r-dim lower/near bounds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import numpy.typing as npt
import torch

from kvdlra.integrators.frequent_directions import FrequentDirections
from kvdlra.integrators.streaming import StreamingBUG
from kvdlra.lowrank import incremental_svd_recon, truncated_svd_recon

N_SINK = 4
RANKS = [16, 32, 64, 128]


def load_matrix(layer_path: Path, key: str) -> npt.NDArray[np.float64]:
    blob = torch.load(layer_path, weights_only=False)
    k = blob[key].float().numpy()
    h, t, d = k.shape
    m = k.transpose(0, 2, 1).reshape(h * d, t)[:, N_SINK:]
    return np.ascontiguousarray(m, dtype=np.float64)


def bug_error(m: npt.NDArray[np.float64], r: int) -> float:
    tracker = StreamingBUG(m.shape[0], rank_cap=r)
    tracker.update_many(m)
    return float(tracker.reconstruction_error(m))


def fd_error(m: npt.NDArray[np.float64], r: int, ell: int) -> float:
    tracker = FrequentDirections(m.shape[0], rank=r, ell=ell)
    tracker.update_many(m)
    return float(tracker.reconstruction_error(m))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--key", default="K_pre", choices=["K", "K_pre"])
    parser.add_argument("--out-json", default="results/w7-fd-ablation.json")
    args = parser.parse_args()

    dump_root = Path("dumps/llama3.2-1b")
    dirs = sorted(d for d in dump_root.glob("doc*_len4096_rope-both") if d.is_dir())
    paths = [d / f"layer_{args.layer:02d}.pt" for d in dirs]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit(f"no layer-{args.layer} dumps under {dump_root}")
    mats = [load_matrix(p, args.key) for p in paths]
    n = mats[0].shape[0]
    print(f"{len(mats)} docs, n_features={n}, T~{mats[0].shape[1]}, key={args.key}\n")

    results: dict[str, object] = {
        "layer": args.layer,
        "key": args.key,
        "n_docs": len(mats),
        "ranks": {},
    }
    ranks_out: dict[str, object] = {}
    print(f"{'rank':>5} {'oracle':>8} {'BUG':>8} {'FD-mm':>8} {'FD-2r':>8} {'iSVD':>8}   verdict")
    for r in RANKS:
        ell_mm = min(n, r + round(r * r / n)) if r + round(r * r / n) > r else r + 1
        keys = ("oracle", "bug", "fd_mm", "fd_2r", "isvd")
        rows: dict[str, list[float]] = {k: [] for k in keys}
        for m in mats:
            rows["oracle"].append(truncated_svd_recon(m, r))
            rows["bug"].append(bug_error(m, r))
            rows["fd_mm"].append(fd_error(m, r, ell_mm))
            rows["fd_2r"].append(fd_error(m, r, 2 * r))
            rows["isvd"].append(incremental_svd_recon(m, r))
        means = {k: float(np.mean(v)) for k, v in rows.items()}
        # Honest verdict: does FD (matched memory) beat/tie BUG?
        diff = means["fd_mm"] - means["bug"]
        verdict = "FD ties/wins" if diff <= 1e-3 else f"BUG wins by {diff:.4f}"
        print(
            f"{r:>5} {means['oracle']:>8.4f} {means['bug']:>8.4f} {means['fd_mm']:>8.4f} "
            f"{means['fd_2r']:>8.4f} {means['isvd']:>8.4f}   {verdict}  (FD-mm ell={ell_mm})"
        )
        ranks_out[str(r)] = {"ell_mm": ell_mm, "means": means, "per_doc": rows}
    results["ranks"] = ranks_out

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n[wrote {out}]")


if __name__ == "__main__":
    main()
