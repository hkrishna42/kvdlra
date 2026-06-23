"""Reference low-rank reconstruction-error baselines: truncated + incremental SVD.

Shared by ``scripts/sigma_decay.py`` (the Week-1 figure) and
``scripts/week2_pilot.py`` (the Week-2 go/no-go pilot). Each takes a
feature-by-token matrix ``m`` (rows = features, columns = tokens; see
``docs/notes/conventions.md``) and a target rank ``r``, and returns the relative
Frobenius reconstruction error ``||m - m_r||_F / ||m||_F`` of the rank-``r``
approximation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["incremental_svd_recon", "truncated_svd_recon"]


def truncated_svd_recon(m: npt.NDArray[np.float64], r: int) -> float:
    """Relative Frobenius error of the best rank-``r`` (truncated-SVD) recon.

    The Eckart--Young optimal rank-``r`` approximation of ``m`` -- the oracle /
    irreducible lower bound at rank ``r``.
    """
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    mr = (u[:, :r] * s[:r]) @ vt[:r]
    return float(np.linalg.norm(m - mr, ord="fro") / np.linalg.norm(m, ord="fro"))


def incremental_svd_recon(m: npt.NDArray[np.float64], r: int) -> float:
    """Brand-style incremental SVD: stream columns of ``m``, keep top-``r``.

    A locally greedy streaming baseline -- it never revisits a column once
    absorbed -- i.e. the "naive streaming" curve the BUG tracker should beat at
    matched rank.
    """
    u: npt.NDArray[np.float64] = np.zeros((m.shape[0], 0))
    s: npt.NDArray[np.float64] = np.zeros(0)
    vt: npt.NDArray[np.float64] = np.zeros((0, 0))
    for j in range(m.shape[1]):
        c = m[:, j : j + 1]
        if u.size == 0:
            u, sj, vtj = np.linalg.svd(c, full_matrices=False)
            s = sj
            vt = np.zeros((sj.shape[0], 1))
            vt[:, -1] = vtj[:, 0]
            continue
        # Project the new column onto the current basis and take the residual.
        proj = u.T @ c
        resid = c - u @ proj
        resid_norm = float(np.linalg.norm(resid))
        resid_unit = resid / (resid_norm + 1e-12)
        # Build and re-decompose the small (k+1)x(k+1) core.
        core = np.block([[np.diag(s), proj], [np.zeros((1, s.size)), np.array([[resid_norm]])]])
        up, sp, vtp = np.linalg.svd(core, full_matrices=False)
        # Absorb the rotation back into the bases.
        u = np.hstack([u, resid_unit]) @ up
        s = sp
        vt_ext = np.block(
            [[vt, np.zeros((vt.shape[0], 1))], [np.zeros((1, vt.shape[1])), np.array([[1.0]])]]
        )
        vt = vtp @ vt_ext
        if s.size > r:
            u, s, vt = u[:, :r], s[:r], vt[:r, :]
    mr = (u * s) @ vt
    return float(np.linalg.norm(m - mr, ord="fro") / np.linalg.norm(m, ord="fro"))
