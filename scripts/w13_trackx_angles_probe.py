"""Week-13 Track-X CPU probe: are adjacent-layer PRE-RoPE key subspaces aligned?

MOTIVE (attacks the overhead floor / Axiom B). BUG pays an independent basis per
layer: for a rank-r factoring of the 512-feature key matrix, the left factor U is
512 x r floats *per layer*, and that ``2nr``-style footprint is the overhead-floor
wall (Week-8/dominance map). IF adjacent layers span nearly the same key
subspace, a **depth-continuous basis** becomes possible -- one shared U plus a
cheap increment ``dU/d(layer)`` instead of 16 independent bases -- amortizing the
overhead across depth. Small adjacent-layer principal angles FUND that idea;
large angles KILL it (nothing to share).

WHAT THIS MEASURES. Per layer, build the pre-RoPE feature-by-token key matrix per
``docs/notes/conventions.md``::

    M = K_pre.transpose(0, 2, 1).reshape(H*D, T)     # (512, T), rows=features

drop the first ``N_SINK`` sink columns (StreamingLLM outliers), take the top-r
LEFT-singular subspace ``U in R^{512 x r}`` (the feature basis BUG tracks), then
compute PRINCIPAL ANGLES between ADJACENT layers L and L+1. Angles come from the
SVD of ``U_a^T U_b`` (singular values = cos of the principal angles), clipped to
[-1, 1]; cross-checked against ``scipy.linalg.subspace_angles`` at startup.

Reported per adjacent pair, averaged over all docs:
  * leading angle  = theta_1, the SMALLEST principal angle (best-aligned direction)
  * median angle   = median over the r principal angles (bulk overlap)
A random-subspace control (two Haar-random r-dim subspaces in R^512) calibrates
"how small is small" -- adjacent-layer angles are only meaningful relative to it.
Gap-2 pairs (L vs L+2) are also reported as a depth-continuity texture: if
alignment decays smoothly with depth, dU/d(layer) is well-behaved.

BAR (pre-registered, on the deployed-ish rank r=32; r=64 also reported):
  FUND  if median-over-pairs of the leading angle < ~40 deg AND clearly below the
        random control (a shared top direction exists across depth).
  KILL  if median-over-pairs of the leading angle > ~60 deg (too different).
Between is a lean. The median-of-all-angles is reported alongside and gates the
honesty of any FUND: a small leading angle with a ~90 deg median means only ONE
direction is shared, NOT a shared basis -- that is a KILL for the full-basis idea.

PROXY CAVEAT. Subspace overlap is NECESSARY-not-SUFFICIENT for a shared-basis
integrator: small angles say a depth-continuous basis is *representable*, they do
NOT prove a cheap online update can track it or that end-to-end ppl/retrieval
survives. This is a geometry ceiling, not an end-to-end promise. Pre-RoPE only
(RoPE smears low-rank structure across positions; see docs/notes/rope-pitfall.md).

Usage::

    uv run python scripts/w13_trackx_angles_probe.py --dumps dumps/llama3.2-1b \
        --ranks 32 64 --out-json results/w13-trackx-angles.json
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
from pathlib import Path

import _paths  # noqa: F401  # bootstrap kvdlra import path
import numpy as np
import numpy.typing as npt
import torch

N_SINK = 4  # leading attention-sink columns dropped before factoring (StreamingLLM)
Arr = npt.NDArray[np.float64]


def key_subspace(k_pre: torch.Tensor, r: int, n_sink: int) -> Arr:
    """Top-r LEFT-singular subspace of the pre-RoPE feature-by-token key matrix.

    ``k_pre`` is (H, T, D); the factored matrix is ``M = (H*D, T)`` with rows =
    features per ``docs/notes/conventions.md``, first ``n_sink`` token columns
    dropped. Returns U with orthonormal columns, shape (H*D, r).
    """
    h, _t, d = k_pre.shape
    m = k_pre.double().numpy().transpose(0, 2, 1).reshape(h * d, -1)  # (512, T)
    m = m[:, n_sink:]  # drop sink columns
    u, _s, _vt = np.linalg.svd(m, full_matrices=False)
    r = min(r, u.shape[1])
    return np.ascontiguousarray(u[:, :r])


def principal_angles_deg(ua: Arr, ub: Arr) -> Arr:
    """Principal angles (degrees, ASCENDING) between two orthonormal-column bases.

    Via the SVD of ``U_a^T U_b``: its singular values are the cosines of the
    principal angles (Bjorck-Golub). Clipped to [-1, 1] for arccos safety.
    angles[0] is the leading (smallest) angle.
    """
    s = np.linalg.svd(ua.T @ ub, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    deg: npt.NDArray[np.float64] = np.degrees(np.arccos(s))  # descending cos -> ascending angle
    return deg


def random_control_angles(n: int, r: int, trials: int, seed: int) -> tuple[float, float]:
    """Mean (leading, median) principal angle for two Haar-random r-dim subspaces in R^n.

    Calibrates the null: what adjacent-layer angles would look like with no shared
    structure. Orthonormal bases via QR of Gaussian matrices.
    """
    rng = np.random.default_rng(seed)
    leads, meds = [], []
    for _ in range(trials):
        qa, _ = np.linalg.qr(rng.standard_normal((n, r)))
        qb, _ = np.linalg.qr(rng.standard_normal((n, r)))
        ang = principal_angles_deg(qa[:, :r], qb[:, :r])
        leads.append(float(ang[0]))
        meds.append(float(np.median(ang)))
    return float(np.mean(leads)), float(np.mean(meds))


def cross_check_scipy(ua: Arr, ub: Arr) -> float:
    """Max abs deg difference between the SVD path and scipy.linalg.subspace_angles.

    scipy returns angles in DESCENDING order; ours ascending -- reverse to compare.
    Returns the largest per-angle disagreement (should be ~1e-9..1e-6 deg).
    """
    from scipy.linalg import subspace_angles  # local import: not a hard dep of the probe

    ours = principal_angles_deg(ua, ub)
    theirs = np.degrees(subspace_angles(ua, ub))[::-1]  # scipy descending -> ascending
    return float(np.max(np.abs(ours - theirs)))


def probe_doc(
    dump_dir: Path, layers: list[int], ranks: list[int], n_sink: int
) -> dict[int, dict[int, Arr]]:
    """Per-rank, per-adjacent/gap pair principal-angle arrays for one document.

    Returns ``{r: {gap: stacked (n_pairs_at_gap, r) angle arrays}}``. Loads each
    layer lazily and frees it (del + gc) so concurrent probes stay memory-bounded.
    """
    r_max = max(ranks)
    subs: dict[int, dict[int, Arr]] = {r: {} for r in ranks}
    for ell in layers:
        blob = torch.load(dump_dir / f"layer_{ell:02d}.pt", weights_only=False)
        k_pre = blob["K_pre"].float()  # (H, T, D)
        u_full = key_subspace(k_pre, r_max, n_sink)  # (512, r_max)
        for r in ranks:
            subs[r][ell] = np.ascontiguousarray(u_full[:, :r])
        del blob, k_pre, u_full
        gc.collect()
    return subs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", default="dumps/llama3.2-1b")
    ap.add_argument("--ranks", type=int, nargs="+", default=[32, 64])
    ap.add_argument(
        "--gaps", type=int, nargs="+", default=[1, 2], help="layer distances (1=adjacent)"
    )
    ap.add_argument("--n-sink", type=int, default=N_SINK)
    ap.add_argument("--ctrl-trials", type=int, default=20, help="random-subspace control trials")
    ap.add_argument("--fund-deg", type=float, default=40.0, help="bar: median leading angle FUND <")
    ap.add_argument("--kill-deg", type=float, default=60.0, help="bar: median leading angle KILL >")
    ap.add_argument("--out-json", default="results/w13-trackx-angles.json")
    args = ap.parse_args()

    dirs = sorted(
        p
        for p in glob.glob(f"{args.dumps}/*_len4096_rope-both")
        if (Path(p) / "layer_00.pt").exists()
    )
    if not dirs:
        raise SystemExit(f"no len4096 rope-both dumps under {args.dumps}")

    # Layer set + shape constants from the first doc's meta.
    meta0 = json.loads((Path(dirs[0]) / "meta.json").read_text())
    n_feat = int(meta0["num_key_value_heads"]) * int(meta0["head_dim"])  # 512
    n_layers = sum(1 for _ in Path(dirs[0]).glob("layer_*.pt"))
    layers = list(range(n_layers))

    # Collect per-doc subspaces, then per-doc/per-pair angle arrays.
    # per_pair[r][gap][(a,b)] -> list over docs of (leading, median) in degrees.
    per_pair: dict[int, dict[int, dict[str, list[list[float]]]]] = {
        r: {g: {} for g in args.gaps} for r in args.ranks
    }
    checked = False
    for d in dirs:
        dp = Path(d)
        subs = probe_doc(dp, layers, args.ranks, args.n_sink)
        for r in args.ranks:
            for g in args.gaps:
                for a in layers[:-g]:
                    b = a + g
                    ang = principal_angles_deg(subs[r][a], subs[r][b])
                    if not checked:  # one-time numeric cross-check vs scipy
                        diff = cross_check_scipy(subs[r][a], subs[r][b])
                        print(f"[cross-check] SVD vs scipy.subspace_angles max diff {diff:.2e} deg")
                        checked = True
                    key = f"{a:02d}-{b:02d}"
                    rec = [float(ang[0]), float(np.median(ang))]
                    per_pair[r][g].setdefault(key, []).append(rec)
        del subs
        gc.collect()
        print(f"[{dp.name}] subspaces + angles done")

    # Random-subspace control per rank (the null: no shared structure).
    controls: dict[int, dict[str, float]] = {}
    for r in args.ranks:
        lead, med = random_control_angles(n_feat, r, args.ctrl_trials, 0)
        controls[r] = {"leading": lead, "median": med}

    # Aggregate: mean-over-docs per pair, then median-over-pairs (the verdict stat).
    out: dict[str, object] = {
        "meta": {
            "docs": [Path(d).name for d in dirs],
            "n_docs": len(dirs),
            "n_layers": n_layers,
            "n_feat": n_feat,
            "n_sink": args.n_sink,
            "ranks": args.ranks,
            "gaps": args.gaps,
            "space": "pre-RoPE keys (RoPE smears low-rank structure)",
        },
        "random_control_deg": controls,
    }
    summary: dict[str, dict[str, float]] = {}
    pair_detail: dict[str, dict[str, dict[str, float]]] = {}
    for r in args.ranks:
        for g in args.gaps:
            pairs = per_pair[r][g]
            # mean over docs for each pair
            pair_lead = {k: float(np.mean([v[0] for v in vs])) for k, vs in pairs.items()}
            pair_med = {k: float(np.mean([v[1] for v in vs])) for k, vs in pairs.items()}
            lead_vals = np.array(sorted(pair_lead.values()))
            med_vals = np.array(sorted(pair_med.values()))
            tag = f"r{r}_gap{g}"
            summary[tag] = {
                "median_leading_deg": float(np.median(lead_vals)),
                "mean_leading_deg": float(np.mean(lead_vals)),
                "min_leading_deg": float(lead_vals.min()),
                "max_leading_deg": float(lead_vals.max()),
                "median_of_median_deg": float(np.median(med_vals)),
                "mean_of_median_deg": float(np.mean(med_vals)),
                "min_of_median_deg": float(med_vals.min()),
                "max_of_median_deg": float(med_vals.max()),
            }
            pair_detail[tag] = {"per_pair_leading_deg": pair_lead, "per_pair_median_deg": pair_med}
    out["pair_detail"] = pair_detail
    out["summary"] = summary

    # Verdict on the deployed-ish rank (smallest requested), adjacent pairs (gap=1).
    r_dep = min(args.ranks)
    s = summary[f"r{r_dep}_gap1"]
    ml = float(s["median_leading_deg"])
    mm = float(s["median_of_median_deg"])
    ctrl_lead = controls[r_dep]["leading"]
    below_ctrl = ml < ctrl_lead - 5.0  # clearly below the null
    if ml < args.fund_deg and below_ctrl:
        verdict = "FUND"
    elif ml > args.kill_deg:
        verdict = "KILL"
    else:
        verdict = "LEAN"
    # Honesty gate: a small leading angle with a near-90 median is NOT a shared basis.
    shared_basis = mm < 60.0
    out["verdict"] = {
        "rank": r_dep,
        "gap": 1,
        "median_leading_deg": ml,
        "median_of_median_deg": mm,
        "control_leading_deg": ctrl_lead,
        "fund_deg": args.fund_deg,
        "kill_deg": args.kill_deg,
        "leading_below_control": below_ctrl,
        "bulk_shared_basis": shared_basis,
        "verdict": verdict,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))

    print(f"\n=== Track-X: adjacent-layer PRE-RoPE key principal angles (n={len(dirs)} docs) ===")
    for r in args.ranks:
        c = controls[r]
        print(f"  random control r{r}: leading {c['leading']:.1f}  median {c['median']:.1f} (deg)")
    for r in args.ranks:
        for g in args.gaps:
            s = summary[f"r{r}_gap{g}"]
            print(
                f"  r{r} gap{g}: leading median {s['median_leading_deg']:.1f} deg "
                f"(range {s['min_leading_deg']:.1f}-{s['max_leading_deg']:.1f})  |  "
                f"all-angle median {s['median_of_median_deg']:.1f} deg "
                f"(range {s['min_of_median_deg']:.1f}-{s['max_of_median_deg']:.1f})"
            )
    print(
        f"\nVERDICT (r{r_dep}, adjacent): leading median {ml:.1f} deg "
        f"[FUND<{args.fund_deg} / KILL>{args.kill_deg}], control {ctrl_lead:.1f} deg, "
        f"bulk-median {mm:.1f} deg (shared-basis {'yes' if shared_basis else 'NO'}) -> {verdict}"
    )
    print(f"[wrote {args.out_json}]")


if __name__ == "__main__":
    main()
