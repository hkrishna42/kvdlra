"""Week-12 Track-1 (Q-BUG) CPU probe: is a query-metric key gist worth building?

The bugS-r32-h256 / bugS-r128-h1024 configs pay a perplexity gap (9.16 vs EA 8.28
at 32K) that rank cannot close (r192/r256 collapse retrieval; Axiom A caps the
same-metric tracker at ~1%). P1's thesis: BUG minimizes ``||M - M_hat||_F`` on
keys, but attention only cares about ``q . (k - k_hat)`` -- error along the
directions queries probe costs logits, orthogonal error is free. If query energy
concentrates in directions a plain rank-r key summary drops, running BUG on
**query-whitened** keys reallocates that rank to what attention reads, for free.

This probe decides (kills or funds) that idea on captured 1B dumps -- **no GPU,
no new training** -- with a pre-registered bar.

Two measurements per (doc, layer), both honoring GQA + the pre/post-RoPE split
(BUG operates pre-RoPE; attention is post-RoPE):

1. PRECONDITION (pre-RoPE geometry). Per KV head, how much of the query second
   moment ``Q = E[q q^T]`` lives OUTSIDE the plain rank-r key subspace. If ~0
   (queries already live where the key summary keeps energy), whitening cannot
   help -> kill. Reported as the "query energy the plain gist drops".

2. PAYOFF (post-RoPE attention output). Reconstruct the key gist at rank r two
   ways -- plain truncated SVD vs whitened (block-diagonal L from Q, per KV head)
   -- rotate both to post-RoPE, and compare the attention-output error
   ``||A V - A_hat V||`` against the true attention (true V, so the delta isolates
   the KEY reconstruction). Whitening tested at two honest footprints:
   ``diag`` (512 floats/layer, ~+1.6% of the r32 basis) and ``full`` (per-head
   64x64 Cholesky, ~half the r32 basis -- only if diag underperforms).

Bar (pre-registered): whitened attention-output error < plain by >= 3% (relative),
layer-averaged, on >= 4/5 docs, at the deployed rank. Kill: gain < 3%, or the
whitening is numerically unstable (cond(Q) explodes even with the trace-tied
Tikhonov floor).

Usage (after ``scripts/capture_kv.py --rope both --with-q`` has written dumps):

    uv run python scripts/w12_qbug_probe.py --dumps dumps/llama3.2-1b \
        --ranks 32 128 --out-json results/w12-qbug-probe.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import _paths  # noqa: F401  # bootstrap kvdlra import path
import numpy as np
import numpy.typing as npt
import torch

N_SINK = 4
Arr = npt.NDArray[np.float64]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE's rotate_half: split the last dim in two, negate-swap."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` (heads, T, d) by the captured angles ``cos``/``sin`` (T, d)."""
    c, s = cos.unsqueeze(0), sin.unsqueeze(0)
    return x * c + rotate_half(x) * s


def truncated_recon(m: Arr, r: int) -> Arr:
    """Rank-r truncated-SVD reconstruction of ``m`` (features x tokens)."""
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    r = min(r, s.shape[0])
    recon: Arr = (u[:, :r] * s[:r]) @ vt[:r]
    return recon


def head_query_moments(q_pre: torch.Tensor, n_kv: int, mu_frac: float) -> list[Arr]:
    """Per-KV-head query second moment ``Q_j = E[q q^T]`` (d x d), Tikhonov-floored.

    ``q_pre`` is (H_q, T, d); GQA maps query head h -> KV head h // (H_q // n_kv).
    Averaging the group's queries gives the moment that weights KV head j's keys.
    A trace-tied floor ``mu = mu_frac * tr(Q)/d`` bounds cond(Q) for the Cholesky.
    """
    hq, _t, d = q_pre.shape
    group = hq // n_kv
    q = q_pre.double().numpy()
    out: list[Arr] = []
    for j in range(n_kv):
        qs = q[j * group : (j + 1) * group]  # (group, T, d)
        qs = qs.reshape(-1, d)  # (group*T, d)
        mom = (qs.T @ qs) / qs.shape[0]  # (d, d)
        mom += (mu_frac * np.trace(mom) / d) * np.eye(d)
        out.append(mom)
    return out


def whiten_l(moms: list[Arr], mode: str) -> tuple[list[Arr], list[Arr]]:
    """Block-diagonal whitening factors ``L_j`` and their inverses-transpose.

    ``diag``: L_j = diag(sqrt(diag(Q_j))) -- axis-aligned, 512 floats/layer total.
    ``full``: L_j = chol(Q_j) -- captures rotated query energy, d*d per head.
    Returns (L_j) and (L_j^{-T}) per KV head; whitening applies L^T, un-whitening L^{-T}.
    """
    ls_fac: list[Arr] = []
    linv_t: list[Arr] = []
    for q in moms:
        if mode == "diag":
            lj = np.diag(np.sqrt(np.clip(np.diag(q), 1e-12, None)))
        else:  # full
            lj = np.linalg.cholesky(q)  # lower-tri, Q = L L^T
        ls_fac.append(lj)
        linv_t.append(np.linalg.inv(lj).T)
    return ls_fac, linv_t


def whitened_recon(k_pre: torch.Tensor, moms: list[Arr], r: int, mode: str) -> torch.Tensor:
    """Rank-r key gist reconstructed in the query metric, back in pre-RoPE space.

    Per KV head j: whiten K_j by L_j^T, stack all heads into the (n, T) feature
    matrix BUG actually tracks, take ONE global rank-r SVD (rank is shared across
    heads, exactly as the deployed basis), reconstruct, un-whiten by L_j^{-T}.
    Returns (heads, T, d) pre-RoPE.
    """
    n_kv, t, d = k_pre.shape
    ls_fac, linv_t = whiten_l(moms, mode)
    k = k_pre.double().numpy()  # (h, T, d)
    # Whiten each head's keys (columns are tokens): Lt @ K_j^T  -> (d, T)
    blocks = [ls_fac[j].T @ k[j].T for j in range(n_kv)]  # each (d, T)
    m_w = np.concatenate(blocks, axis=0)  # (h*d, T) -- the whitened stacked features
    m_w_hat = truncated_recon(m_w, r)  # global rank-r
    # Un-whiten per head and reshape back to (h, T, d)
    out = np.empty((n_kv, t, d), dtype=np.float64)
    for j in range(n_kv):
        blk = m_w_hat[j * d : (j + 1) * d]  # (d, T)
        out[j] = (linv_t[j] @ blk).T  # un-whiten -> (T, d)
    return torch.from_numpy(out)


def plain_recon(k_pre: torch.Tensor, r: int) -> torch.Tensor:
    """Plain rank-r key gist (global SVD over stacked heads), pre-RoPE."""
    n_kv, t, d = k_pre.shape
    m = k_pre.double().numpy().transpose(0, 2, 1).reshape(n_kv * d, t)
    m_hat = truncated_recon(m, r)
    return torch.from_numpy(m_hat.reshape(n_kv, d, t).transpose(0, 2, 1))


def attention_output(q_post: torch.Tensor, k_post: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Causal softmax attention output (H_q, T, d), GQA-expanding K/V to query heads.

    Loops over query heads so peak memory is one T x T score matrix, not H_q of
    them -- at T=4096 that is the difference between ~67 MB and ~2 GB on CPU.
    """
    hq, t, d = q_post.shape
    n_kv = k_post.shape[0]
    rep = hq // n_kv
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    out = torch.empty_like(q_post)
    for h in range(hq):
        j = h // rep  # this query head's KV head
        scores = (q_post[h] @ k_post[j].transpose(0, 1)) / (d**0.5)  # (T, T)
        scores = scores.masked_fill(mask, float("-inf"))
        out[h] = torch.softmax(scores, dim=-1) @ v[j]
    return out


def precondition_dropped_energy(k_pre: torch.Tensor, moms: list[Arr], r: int) -> float:
    """Fraction of query energy the PLAIN rank-r key gist drops, layer-mean over heads.

    Per KV head: project Q_j onto the orthogonal complement of the plain rank-r
    key subspace for that head's rows, and report tr(Q_perp)/tr(Q). ~0 means the
    key summary already keeps what queries read (whitening can't help); large
    means queries probe dropped directions (whitening should help).
    """
    n_kv, t, d = k_pre.shape
    m = k_pre.double().numpy().transpose(0, 2, 1).reshape(n_kv * d, t)
    u, _s, _vt = np.linalg.svd(m, full_matrices=False)
    ur = u[:, : min(r, u.shape[1])]  # (n, r) global key subspace
    dropped = []
    for j in range(n_kv):
        uj = ur[j * d : (j + 1) * d]  # (d, r) -- this head's rows of the basis
        # Energy of Q_j inside span(uj) vs total; dropped = 1 - inside/total.
        pj = uj @ np.linalg.pinv(uj)  # (d,d) projector onto the retained row-space
        inside = np.trace(pj @ moms[j] @ pj.T)
        total = np.trace(moms[j])
        dropped.append(max(0.0, 1.0 - inside / total))
    return float(np.mean(dropped))


def probe_layer(
    dump_dir: Path,
    layer: int,
    rope: dict[str, torch.Tensor],
    ranks: list[int],
    n_kv: int,
    mu_frac: float,
) -> dict[str, float]:
    blob = torch.load(dump_dir / f"layer_{layer:02d}.pt", weights_only=False)
    k_pre = blob["K_pre"].float()  # (h, T, d)
    q_pre = blob["Q_pre"].float()  # (H_q, T, d)
    v = blob["V"].float()
    cos, sin = rope["cos"].float(), rope["sin"].float()

    moms = head_query_moments(q_pre, n_kv, mu_frac)
    q_post = apply_rope(q_pre, cos, sin)
    k_post_true = apply_rope(k_pre, cos, sin)
    o_true = attention_output(q_post, k_post_true, v)
    o_norm = float(torch.linalg.norm(o_true))

    def out_err(k_pre_hat: torch.Tensor) -> float:
        o_hat = attention_output(q_post, apply_rope(k_pre_hat.float(), cos, sin), v)
        return float(torch.linalg.norm(o_hat - o_true)) / o_norm

    res: dict[str, float] = {}
    for r in ranks:
        res[f"drop_r{r}"] = precondition_dropped_energy(k_pre, moms, r)
        res[f"plain_r{r}"] = out_err(plain_recon(k_pre, r))
        res[f"diag_r{r}"] = out_err(whitened_recon(k_pre, moms, r, "diag"))
        res[f"full_r{r}"] = out_err(whitened_recon(k_pre, moms, r, "full"))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", default="dumps/llama3.2-1b")
    ap.add_argument("--ranks", type=int, nargs="+", default=[32, 128])
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="layer indices to probe (default: a spread of 6)",
    )
    ap.add_argument(
        "--mu-frac", type=float, default=1e-3, help="Tikhonov floor for Q as a fraction of tr(Q)/d"
    )
    ap.add_argument("--min-gain", type=float, default=3.0, help="bar: %% improvement")
    ap.add_argument("--out-json", default="results/w12-qbug-probe.json")
    args = ap.parse_args()

    dirs = sorted(
        p
        for p in glob.glob(f"{args.dumps}/*_rope-both")
        if (Path(p) / "rope.pt").exists()
        and json.loads((Path(p) / "meta.json").read_text()).get("with_q")
    )
    if not dirs:
        raise SystemExit(f"no with-q rope-both dumps under {args.dumps} (run capture_kv --with-q)")

    aggs: list[dict[str, float]] = []  # per-doc layer-mean metrics (for the verdict)
    per_doc: list[dict[str, object]] = []  # richer records (for the JSON)
    for d in dirs:
        dp = Path(d)
        meta = json.loads((dp / "meta.json").read_text())
        n_kv = int(meta["num_key_value_heads"])
        n_layers = sum(1 for _ in dp.glob("layer_*.pt"))
        layers = args.layers or sorted({round(x) for x in np.linspace(1, n_layers - 1, 6)})
        rope = torch.load(dp / "rope.pt", weights_only=True)
        rows = [probe_layer(dp, ell, rope, args.ranks, n_kv, args.mu_frac) for ell in layers]
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        aggs.append(agg)
        per_doc.append({"doc": dp.name, "layers": layers, "agg": agg})
        print(
            f"[{dp.name}] "
            + "  ".join(
                f"r{r}: drop={agg[f'drop_r{r}']:.3f} plain={agg[f'plain_r{r}']:.4f} "
                f"diag={agg[f'diag_r{r}']:.4f} full={agg[f'full_r{r}']:.4f}"
                for r in args.ranks
            )
        )

    # Pre-registered verdict: per rank, per whitening, count docs with >= min-gain %.
    verdict: dict[str, object] = {"min_gain_pct": args.min_gain, "n_docs": len(per_doc)}
    print(
        "\n=== VERDICT (bar: whitened < plain by >= "
        f"{args.min_gain}% on >= {max(4, int(0.8 * len(per_doc)))}/{len(per_doc)} docs) ==="
    )
    for r in args.ranks:
        for mode in ("diag", "full"):
            gains = [
                100.0 * (a[f"plain_r{r}"] - a[f"{mode}_r{r}"]) / a[f"plain_r{r}"] for a in aggs
            ]
            n_pass = sum(g >= args.min_gain for g in gains)
            passed = n_pass >= max(4, int(0.8 * len(per_doc)))
            verdict[f"r{r}_{mode}_median_gain_pct"] = float(np.median(gains))
            verdict[f"r{r}_{mode}_n_pass"] = int(n_pass)
            verdict[f"r{r}_{mode}_passed"] = bool(passed)
            print(
                f"  r{r} {mode:4s}: median gain {np.median(gains):+.2f}%  "
                f"docs passing {n_pass}/{len(per_doc)}  -> {'FUND' if passed else 'kill'}"
            )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"per_doc": per_doc, "verdict": verdict}, indent=2))
    print(f"\n[wrote {args.out_json}]")


if __name__ == "__main__":
    main()
