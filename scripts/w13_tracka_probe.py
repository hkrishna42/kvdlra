"""Week-13 Track-A CPU probe: integrator-surgery at the BUG truncation site.

QUESTION (pre-registered). At 32K the low-rank basis is re-truncated ~2000x (once
per absorb block). Repeated re-projection can erode the OLDEST directions
("deep-horizon erosion") -- a stability property the single Frobenius snapshot
never sees. Does a *damped* or *energy-weighted* truncation at the single BUG
basis-update site lower long-horizon rank-r reconstruction error vs raw-sigma,
WITHOUT regressing the moderate band?

WHAT THIS REPLICATES (no src/ edit). ``kvdlra.integrators.streaming_torch.
augmented_bug_step`` builds, per block, the augmented basis ``u_aug`` and square-
root core ``b_fac``, then at lines 182-187 truncates::

    u_loc, sigma, _ = svd(b_fac); keep = min(rank_cap, len(sigma))
    u_new = u_aug @ u_loc[:, :keep]; b_new = diag(sigma[:keep])

This probe copies that block-update math VERBATIM (incl. the Week-7 re-orth +
rank-revealing residual SVD and the ``n - r_old`` admission clamp) and swaps ONLY
the truncation site for these variants:

  * ``raw``            -- baseline: keep top-``keep`` by sigma, core=diag(sigma).
  * ``tik_a``          -- Tikhonov filter-factor on the *stored core* (the literal
                          spec "sigma -> sigma^2/(sigma^2+mu)" read dimensionally
                          as core_i = sigma_i * sigma_i^2/(sigma_i^2+mu); this
                          SHRINKS the weakest kept directions). mu = mu_c*sigma_min^2,
                          sigma_min = smallest kept sigma this step. Selection is
                          unchanged (top-keep by sigma); only the core fed forward
                          changes, so the downstream basis trajectory changes.
  * ``tik_ridge``      -- the sensible-SIGN alternative damping (exploratory):
                          core_i = sqrt(sigma_i^2 + mu). LIFTS the weakest kept
                          directions (ridge on the second-moment eigenvalues) so
                          old directions survive re-truncation longer.
  * ``energy_faithful``-- rank directions by sigma_i * ||c_i|| where c_i is the
                          i-th row of the *carried per-token coordinate matrix*
                          (exactly what the streaming cache carries via ``rot``).
                          NOTE (proven in the returned method): with the deployment-
                          faithful carry, ||c_i|| == sigma_i, so this reduces to
                          raw-sigma; included as an empirical sanity that it is a
                          NO-OP (guards Phase-2 against a dead idea).
  * ``energy_rawdata`` -- the *informative* energy weighting: rank by
                          sigma_i * sqrt(d_i^T G d_i) where G = sum of the RAW past
                          key columns' outer products and d_i the candidate feature-
                          space direction -- i.e. true data energy each candidate
                          direction explains (this diverges from sigma because the
                          square-root core has already dropped truncated energy).
                          NON-DEPLOYABLE (uses all past raw columns); it is an
                          UPPER-BOUND "does a better selection even exist" probe.
  * ``oracle``         -- reference floor: top-r SVD subspace of the whole matrix
                          (best uniform rank-r; its per-bin error is the intrinsic
                          low-rank difficulty, so raw-vs-oracle IS the erosion).

METRIC. After streaming the whole pre-RoPE key matrix (rows = 512 features =
head_dim*num_kv_heads, cols = tokens; first ``n_sink``=4 sink columns dropped per
docs/notes/conventions.md), reconstruct EVERY column from the FINAL rank-r basis
(``u_final @ u_final^T @ m`` -- pure projection, matching
``blocked_bug_project``) and report the relative Frobenius reconstruction error
BINNED BY TOKEN POSITION with ``bin_size`` loaded from
results/w9-surprise-sweep-1b.json. Bin 0 = earliest-entering = the deep-horizon
(most re-truncated) bin.

PROXY CAVEAT. The number is *pre-RoPE key reconstruction error*, NOT perplexity.
Week-12 saw a CPU attention-output proxy over-predict end-to-end ppl ~30-40x; the
error->ppl mapping here is UNCERTAIN and the ">0.05 ppl" target is only nominal.
A bar met here is a CEILING, not a promise. Reconstruction is also on pre-RoPE
keys; attention reads post-RoPE, a second proxy gap.

Usage::

    uv run python scripts/w13_tracka_probe.py \
        --dumps dumps/llama3.2-1b --ranks 32 128 \
        --out-json results/w13-tracka-probe.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import _paths  # noqa: F401  # bootstrap kvdlra import path (mirrors repo scripts)
import numpy as np
import torch

N_SINK = 4  # StreamingLLM sink columns dropped before factoring (conventions.md)
W9_JSON = "results/w9-surprise-sweep-1b.json"
_SVD_FALLBACKS = [0]  # count of fp32 SVDs that needed an fp64 retry (small blocks)


def _safe_svd(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``torch.linalg.svd`` with an fp64 fallback for ill-conditioned tiny blocks.

    The deployed integrator runs fp32; at very small block sizes (the erosion-count
    stress runs) fp32 gedd can fail to converge (repeated singular values). Retry in
    fp64 and cast back so the sweep completes; fallbacks are counted and reported.
    """
    try:
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
    except RuntimeError:
        _SVD_FALLBACKS[0] += 1
        u64, s64, vh64 = torch.linalg.svd(x.double(), full_matrices=False)
        u, s, vh = u64.to(x.dtype), s64.to(x.dtype), vh64.to(x.dtype)
    return u, s, vh


def load_key_matrix(dump_dir: Path, layer: int, n_sink: int) -> torch.Tensor:
    """Pre-RoPE feature-by-token matrix ``(h*d, t)`` for one layer, sinks dropped.

    BUG operates PRE-RoPE, so this uses ``K_pre`` (not the post-RoPE ``K``). The
    reshape matches docs/notes/conventions.md: rows = head_dim*num_kv_heads = 512.
    """
    blob = torch.load(dump_dir / f"layer_{layer:02d}.pt", weights_only=False)
    k: torch.Tensor = blob["K_pre"].float()  # (h, t, d)
    h, t, d = k.shape
    m = k.permute(0, 2, 1).reshape(h * d, t).contiguous()  # (512, t)
    if m.shape[1] > n_sink:
        m = m[:, n_sink:]
    return m


def _augment(
    u: torch.Tensor | None, b: torch.Tensor | None, block: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Verbatim copy of the augmented block-update in ``augmented_bug_step`` (up to
    the truncating SVD). Returns ``(u_aug, b_fac, r_old)``."""
    n = block.shape[0]
    if u is None:
        q, r = torch.linalg.qr(block, mode="reduced")
        return q, r, 0
    assert b is not None
    r_old = u.shape[1]
    a = u.mT @ block  # (r, b)
    r_perp = block - u @ a  # (n, b)
    # Re-orthogonalize once ("twice is enough") -- Week-7 fix, replicated.
    a2 = u.mT @ r_perp
    r_perp = r_perp - u @ a2
    a = a + a2
    q, s, vh = _safe_svd(r_perp)
    tol = 100.0 * torch.finfo(block.dtype).eps * torch.linalg.norm(block)
    m_adm = int((s > tol).sum().item())
    m_adm = min(m_adm, max(0, n - r_old))
    q = q[:, :m_adm]
    u_aug = torch.cat([u, q], dim=1)  # (n, r+m)
    top = torch.cat([b, a], dim=1)  # (r, r+b)
    zeros = torch.zeros(m_adm, r_old, dtype=block.dtype, device=block.device)
    bot = torch.cat([zeros, s[:m_adm].unsqueeze(1) * vh[:m_adm]], dim=1)  # (m, r+b)
    b_fac = torch.cat([top, bot], dim=0)  # (r+m, r+b)
    return u_aug, b_fac, r_old


def _select_core(
    sigma: torch.Tensor,
    keep: int,
    variant: str,
    mu_c: float,
    weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(sel, core)`` for the truncation site given a variant.

    ``sel`` are the kept direction indices into ``u_loc`` columns; ``core`` the
    stored square-root core diagonal (fed forward to the next block).
    """
    if variant in ("raw", "tik_a", "tik_ridge", "energy_faithful", "energy_rawdata"):
        if variant.startswith("energy") and weight is not None:
            sel = torch.topk(weight, keep).indices
        else:
            sel = torch.arange(keep, device=sigma.device)
        s_sel = sigma[sel]
        if variant == "tik_a":
            s_min = sigma[keep - 1]
            mu = mu_c * s_min * s_min
            core = s_sel * (s_sel * s_sel) / (s_sel * s_sel + mu)
        elif variant == "tik_ridge":
            s_min = sigma[keep - 1]
            mu = mu_c * s_min * s_min
            core = torch.sqrt(s_sel * s_sel + mu)
        else:  # raw, energy_* keep true singular values
            core = s_sel
        return sel, core
    raise ValueError(f"unknown variant {variant!r}")


def stream_subspace(
    m: torch.Tensor,
    rank_cap: int,
    block_size: int,
    variant: str,
    mu_c: float,
) -> torch.Tensor:
    """Stream a blocked augmented-BUG basis over ``m``'s columns with ``variant`` at
    the single truncation site; return the FINAL orthonormal basis ``(n, r)``."""
    n, t = m.shape
    u: torch.Tensor | None = None
    b: torch.Tensor | None = None
    coords: torch.Tensor | None = None  # (r, N) carried per-token coords (energy_faithful)
    g_data: torch.Tensor | None = None  # (n, n) raw-data gram (energy_rawdata)

    for start in range(0, t, block_size):
        block = m[:, start : start + block_size]  # (n, b)
        u_aug, b_fac, r_old = _augment(u, b, block)
        u_loc, sigma, _ = _safe_svd(b_fac)
        keep = min(rank_cap, int(sigma.shape[0]))

        weight: torch.Tensor | None = None
        c_cand: torch.Tensor | None = None
        if variant == "energy_faithful":
            # candidate per-direction coordinate rows in the u_loc directions.
            cnew = u_loc.mT @ (u_aug.mT @ block)  # (K, b) new-block token coords
            if coords is None:
                c_cand = cnew
            else:
                cold = u_loc[:r_old, :].mT @ coords  # (K, N_old) carried old coords
                c_cand = torch.cat([cold, cnew], dim=1)  # (K, N_total)
            weight = sigma * torch.linalg.norm(c_cand, dim=1)
        elif variant == "energy_rawdata":
            g_data = block @ block.mT if g_data is None else g_data + block @ block.mT
            d = u_aug @ u_loc  # (n, K) candidate feature-space directions
            e = (d * (g_data @ d)).sum(dim=0)  # d_i^T G d_i per direction
            weight = sigma * torch.sqrt(torch.clamp(e, min=0.0))

        sel, core = _select_core(sigma, keep, variant, mu_c, weight)
        u = u_aug @ u_loc[:, sel]  # (n, keep)
        b = torch.diag(core)
        if variant == "energy_faithful" and c_cand is not None:
            coords = c_cand[sel, :]

    if u is None:  # empty matrix guard
        u = torch.zeros(n, 1, dtype=m.dtype)
    return u


def oracle_subspace(m: torch.Tensor, rank_cap: int) -> torch.Tensor:
    """Top-``rank_cap`` left-singular subspace of the whole matrix (uniform best)."""
    u, _s, _vh = torch.linalg.svd(m, full_matrices=False)
    r = min(rank_cap, u.shape[1])
    u_r: torch.Tensor = u[:, :r]
    return u_r


def binned_rel_err(m: torch.Tensor, u: torch.Tensor, bin_size: int) -> list[float]:
    """Per-bin relative Frobenius error of the projection reconstruction ``u u^T m``.

    Bins are contiguous ``bin_size``-column blocks (bin 0 = earliest = deep horizon).
    Computed in float64 for a clean metric regardless of the fp32 streaming basis.
    """
    md = m.double()
    ud = u.double()
    recon = ud @ (ud.mT @ md)
    diff = md - recon
    t = md.shape[1]
    out: list[float] = []
    for start in range(0, t, bin_size):
        num = torch.linalg.norm(diff[:, start : start + bin_size])
        den = torch.linalg.norm(md[:, start : start + bin_size])
        out.append(float(num / den) if float(den) > 0 else 0.0)
    return out


def probe_layer(
    m: torch.Tensor,
    rank_cap: int,
    block_size: int,
    bin_size: int,
    variants: list[tuple[str, float]],
) -> dict[str, list[float]]:
    """Per-bin relative error for every (variant, mu) at one rank on one matrix."""
    res: dict[str, list[float]] = {}
    res["oracle"] = binned_rel_err(m, oracle_subspace(m, rank_cap), bin_size)
    for name, mu_c in variants:
        fixed = ("raw", "energy_faithful", "energy_rawdata")
        label = name if name in fixed else f"{name}_mu{mu_c:g}"
        u = stream_subspace(m, rank_cap, block_size, name, mu_c)
        res[label] = binned_rel_err(m, u, bin_size)
    return res


def summarize(
    per_cell: list[dict[str, list[float]]],
    labels: list[str],
    n_bins: int,
) -> dict[str, object]:
    """Aggregate per-(doc,layer) per-bin errors into means and raw-relative deltas.

    Deep bin = 0; recent band = last 2 bins; moderate band = bins [2, n_bins-2).
    Deltas are (variant - raw) / raw, averaged across cells, per bin.
    """
    mean_bins: dict[str, list[float]] = {}
    for lab in labels:
        stack = np.array([c[lab] for c in per_cell if lab in c])  # (cells, n_bins)
        mean_bins[lab] = stack.mean(axis=0).tolist()

    raw = np.array([c["raw"] for c in per_cell])  # (cells, n_bins)
    mod_lo, mod_hi = 2, max(3, n_bins - 2)
    rel_delta: dict[str, dict[str, float]] = {}
    for lab in labels:
        if lab == "raw":
            continue
        var = np.array([c[lab] for c in per_cell])
        rd = (var - raw) / np.clip(raw, 1e-12, None)  # (cells, n_bins) relative delta
        rd_mean = rd.mean(axis=0)  # per-bin mean relative delta
        deep = float(rd_mean[0])
        moderate_max = float(rd_mean[mod_lo:mod_hi].max())
        moderate_mean = float(rd_mean[mod_lo:mod_hi].mean())
        recent = float(rd_mean[-2:].mean())
        deep_improved_frac = float((rd[:, 0] < 0).mean())
        rel_delta[lab] = {
            "deep_bin0_rel_delta": deep,
            "moderate_band_max_rel_delta": moderate_max,
            "moderate_band_mean_rel_delta": moderate_mean,
            "recent_band_rel_delta": recent,
            "deep_improved_frac_cells": deep_improved_frac,
            "per_bin_rel_delta": rd_mean.tolist(),
        }
    return {
        "moderate_band_bins": [mod_lo, mod_hi],
        "mean_rel_err_by_bin": mean_bins,
        "rel_delta_vs_raw": rel_delta,
    }


def verdict_for_rank(summary: dict[str, object], deployable: list[str]) -> dict[str, object]:
    """Pre-registered bar per variant: deep bin improves AND moderate band does not
    regress. Tolerances: improve < -0.1% rel; regress > +0.5% rel on any moderate bin.
    """
    tol_improve = -1e-3
    tol_regress = 5e-3
    rd = summary["rel_delta_vs_raw"]
    assert isinstance(rd, dict)
    out: dict[str, object] = {"tol_improve": tol_improve, "tol_regress": tol_regress}
    any_deployable_pass = False
    for lab, d in rd.items():
        assert isinstance(d, dict)
        deep_ok = d["deep_bin0_rel_delta"] < tol_improve
        mod_ok = d["moderate_band_max_rel_delta"] <= tol_regress
        passed = bool(deep_ok and mod_ok)
        out[lab] = {
            "deep_improves": bool(deep_ok),
            "moderate_no_regress": bool(mod_ok),
            "passed": passed,
            "deployable": lab in deployable,
        }
        if passed and lab in deployable:
            any_deployable_pass = True
    out["any_deployable_pass"] = any_deployable_pass
    return out


def main() -> None:
    """Run the Track-A probe and write results/w13-tracka-probe.json."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", default="dumps/llama3.2-1b")
    ap.add_argument("--ranks", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12, 15])
    ap.add_argument(
        "--block-list",
        type=int,
        nargs="+",
        default=[128, 32, 16, 8],
        help="absorb-block sizes to sweep. 128=deployed@4K; smaller emulates the 32K/64K "
        "re-truncation COUNT on 4096 data (4092/block truncations): 128->32, 32->128, "
        "16->256(==32K@block128), 8->512(==64K) -- erosion is about truncation count.",
    )
    ap.add_argument("--mu-list", type=float, nargs="+", default=[0.25, 1.0])
    ap.add_argument("--w9-json", default=W9_JSON)
    ap.add_argument("--out-json", default="results/w13-tracka-probe.json")
    args = ap.parse_args()

    bin_size = int(json.loads(Path(args.w9_json).read_text())["bin_size"])

    dirs = sorted(glob.glob(f"{args.dumps}/*_len4096_rope-both"))
    if not dirs:
        raise SystemExit(f"no len4096 rope-both dumps under {args.dumps}")

    torch.manual_seed(0)
    # Variant plan: raw baseline, Tikhonov filter-factor at each mu (the pre-
    # registered "sweep ~2 mu"), one alt-sign ridge, and the two energy readings.
    variants: list[tuple[str, float]] = [("raw", 0.0)]
    for mu in args.mu_list:
        variants.append(("tik_a", mu))
    variants.append(("tik_ridge", args.mu_list[-1]))
    variants += [("energy_faithful", 0.0), ("energy_rawdata", 0.0)]
    deployable = ["tik_a", "tik_ridge", "energy_faithful"]  # energy_rawdata is not

    out: dict[str, object] = {
        "probe": "w13-tracka-integrator-surgery",
        "model": "unsloth/Llama-3.2-1B-Instruct",
        "compute_dtype": "float32 (matches streaming_torch compute_dtype)",
        "recon": "u_final @ u_final^T @ m (projection, pre-RoPE keys, sinks dropped)",
        "n_sink": N_SINK,
        "block_list": args.block_list,
        "T_note": "dumps are len4096 -> block 128 = only 32 re-truncations; the "
        "deployed 32K regime is ~256+ re-truncations (block sweep emulates the COUNT)",
        "bin_size": bin_size,
        "bin_axis": "KEY/context column position (deep=bin0=oldest); NOT the w9 ppl "
        "query-position axis -- ppl_bins there index generation tokens, a different axis",
        "layers": args.layers,
        "docs": [Path(d).name for d in dirs],
        "mu_list": args.mu_list,
        "variant_notes": {
            "tik_a": "core_i = sigma_i * sigma_i^2/(sigma_i^2+mu*sigma_min^2) (literal "
            "filter-factor; shrinks weakest kept)",
            "tik_ridge": "core_i = sqrt(sigma_i^2 + mu*sigma_min^2) (alt sign; lifts weakest)",
            "energy_faithful": "rank by sigma_i*||carried coord row_i||; provably == raw "
            "(carry preserves 2nd moment) -- NO-OP sanity",
            "energy_rawdata": "rank by sigma_i*sqrt(d_i^T G d_i), G=sum raw col outer prods; "
            "NON-DEPLOYABLE upper bound (uses all past raw columns)",
        },
        "ranks": {},
    }

    ranks_out = out["ranks"]
    assert isinstance(ranks_out, dict)
    for rank_cap in args.ranks:
        blocks_out: dict[str, object] = {}
        for block_size in args.block_list:
            per_cell: list[dict[str, list[float]]] = []
            labels: list[str] = []
            for d in dirs:
                dp = Path(d)
                for layer in args.layers:
                    m = load_key_matrix(dp, layer, N_SINK)
                    cell = probe_layer(m, rank_cap, block_size, bin_size, variants)
                    if not labels:
                        labels = list(cell.keys())
                    per_cell.append(cell)
                    del m
            n_bins = len(per_cell[0]["raw"])
            n_trunc = -(-(4096 - N_SINK) // block_size)  # ceil, replicated truncations
            summary = summarize(per_cell, labels, n_bins)
            verdict = verdict_for_rank(summary, deployable)
            blocks_out[str(block_size)] = {
                "n_cells": len(per_cell),
                "n_bins": n_bins,
                "n_retruncations": n_trunc,
                "labels": labels,
                "summary": summary,
                "verdict": verdict,
            }
            # Console: deep-bin and moderate-band deltas per variant.
            print(
                f"\n=== rank {rank_cap}  block {block_size}  "
                f"({len(per_cell)} cells, {n_trunc} re-truncations, {n_bins} bins) ==="
            )
            rd = summary["rel_delta_vs_raw"]
            assert isinstance(rd, dict)
            mre = summary["mean_rel_err_by_bin"]
            assert isinstance(mre, dict)
            print(
                f"  {'raw abs err':20s} deep(bin0) {mre['raw'][0]:.4f}  "
                f"oracle deep {mre['oracle'][0]:.4f}  (raw-vs-oracle = the erosion)"
            )
            for lab, dd in rd.items():
                assert isinstance(dd, dict)
                print(
                    f"  {lab:20s} deep(bin0) {dd['deep_bin0_rel_delta']:+.4f}  "
                    f"mod_max {dd['moderate_band_max_rel_delta']:+.4f}  "
                    f"mod_mean {dd['moderate_band_mean_rel_delta']:+.4f}  "
                    f"recent {dd['recent_band_rel_delta']:+.4f}  "
                    f"deep_impr_frac {dd['deep_improved_frac_cells']:.2f}"
                )
            print(f"  any deployable variant passes bar: {verdict['any_deployable_pass']}")
        ranks_out[str(rank_cap)] = blocks_out

    out["svd_fp64_fallbacks"] = _SVD_FALLBACKS[0]
    if _SVD_FALLBACKS[0]:
        print(f"\n[note] {_SVD_FALLBACKS[0]} fp32 SVDs fell back to fp64 (small blocks)")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[wrote {args.out_json}]")


if __name__ == "__main__":
    main()
