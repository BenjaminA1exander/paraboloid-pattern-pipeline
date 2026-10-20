r"""vares_pattern_measures.py -- measures of a converged VaReS cutting pattern, as used by Fig. 8 and Table 4.

    load_rungs     the converged rungs of a nested chain (pattern.npz of each rung): h_eff of the spatial gore mesh,
                   the compensation field measure and the shape quantities below
    field_rms      area-weighted rms of the compensation field u = pattern - developed leaf after an area-weighted
                   2-D rigid fit, over (whole gore, away from the tip, tip zone = last TIP_FRAC of the gore length)
    quantities     gore length, rim width, seam-to-seam width at mid-gore and at 5 % of the length from the tip
    lumped_area, mls1, rigid_fit_area
                   the lumped nodal areas, the local-linear Gaussian smoother (exact for a linear field) and the
                   area-weighted rigid fit used to draw the compensation field of Fig. 8(d, e)
"""
import glob
import os

import numpy as np
from scipy.spatial import cKDTree

import MeshStudy_paraboloid_vares_ckpt as CK
import vares_study as MS

LINE_TOL = 1.0                                  # mm: nodes within this of the rim line count as rim nodes
STATIONS = {"mid": 0.50, "tip": 0.05}           # width stations, fraction of the gore length from the tip
TIP_FRAC = 0.10                                 # tip zone: last 10 % of the gore length
SMOOTH_SIGMA = 1.5                              # smoothing length of mls1, in units of h_eff


def seam_point(Xl, P, ids, x):
    o = np.argsort(Xl[ids, 0])
    xs, pts = Xl[ids[o], 0], P[ids[o]]
    return np.array([np.interp(x, xs, pts[:, 0]), np.interp(x, xs, pts[:, 1])])


def quantities(Xl, P, seam):
    """(length, rim width, width mid, width near tip) of the 2-D point set P (nodes in leaf order Xl)."""
    xmin, xmax = Xl[:, 0].min(), Xl[:, 0].max()
    rim = np.where(np.abs(Xl[:, 0] - xmin) < LINE_TOL)[0]
    corner_p, corner_m = rim[np.argmax(Xl[rim, 1])], rim[np.argmin(Xl[rim, 1])]
    tip = int(np.argmax(Xl[:, 0]))
    sp = np.array([i for i in seam if Xl[i, 1] > 1e-9])
    sm = np.array([i for i in seam if Xl[i, 1] < -1e-9])
    L_leaf = xmax - xmin
    out = [float(np.linalg.norm(P[tip] - 0.5 * (P[corner_p] + P[corner_m]))),
           float(np.linalg.norm(P[corner_p] - P[corner_m]))]
    for f in STATIONS.values():
        x = xmax - f * L_leaf
        out.append(float(np.linalg.norm(seam_point(Xl, P, sp, x) - seam_point(Xl, P, sm, x))))
    return np.array(out)


def field_rms(Xl, P, T):
    e = Xl[T]
    A = 0.5 * np.abs((e[:, 1, 0] - e[:, 0, 0]) * (e[:, 2, 1] - e[:, 0, 1]) - (e[:, 2, 0] - e[:, 0, 0]) * (e[:, 1, 1] - e[:, 0, 1]))
    w = np.zeros(len(Xl))
    np.add.at(w, T.ravel(), np.repeat(A / 3.0, 3))           # lumped nodal area
    ca, cb = (w[:, None] * P).sum(0) / w.sum(), (w[:, None] * Xl).sum(0) / w.sum()
    U, _, Vt = np.linalg.svd(((P - ca) * w[:, None]).T @ (Xl - cb))
    Rm = (U @ Vt).T
    if np.linalg.det(Rm) < 0:
        Vt[-1] *= -1
        Rm = (U @ Vt).T
    u = (P - ca) @ Rm.T + cb - Xl                             # compensation field, rigid motion removed
    ue = u[T]                                                  # (ne, 3, 2)
    m2 = ((ue ** 2).sum((1, 2)) + sum((ue[:, i] * ue[:, (i + 1) % 3]).sum(1) for i in range(3))) / 6.0
    d = Xl[:, 0].max() - e[:, :, 0].mean(1)
    L = Xl[:, 0].max() - Xl[:, 0].min()
    tip = d < TIP_FRAC * L
    out = []
    for m in (np.ones(len(T), bool), ~tip, tip):
        out.append(float(np.sqrt((A[m] * m2[m]).sum() / A[m].sum())))
    return np.array(out)


def load_rungs(grading, rungs):
    """One record per rung of the chain `grading` that exists on disk.  A rung whose pattern.npz is not converged
    is read from its newest pattern_snapshot_uncertified_*.npz instead and flagged ok = False."""
    CK.configure("cstraw", grading)
    rows = []
    for h in rungs:
        p = MS.rung_npz(h)
        if not os.path.exists(p) or not bool(np.load(p)["converged"]):
            snap = sorted(glob.glob(os.path.join(os.path.dirname(p), "pattern_snapshot_uncertified_*.npz")))
            if not os.path.exists(p) and not snap:
                continue
            p = snap[-1] if snap else p
        d = np.load(p, allow_pickle=True)
        Xl, P = d["X_leaf"][:, :2], d["X_converged"][:, :2]
        seam, T = np.asarray(d["seam_ids"], int), np.asarray(d["elements"], int)
        crit = str(d["converged_criterion"]) if "converged_criterion" in d.files else ("tol" if bool(d["converged"]) else "none")
        e = d["X_spatial"][T]
        he = float(np.sqrt(4.0 * 0.5 * np.linalg.norm(np.cross(e[:, 1] - e[:, 0], e[:, 2] - e[:, 0]), axis=1).sum() /
                           (np.sqrt(3.0) * len(T))))          # h_eff of the spatial gore mesh
        rows.append(dict(tag=MS.tag(h), h=he, ok=bool(d["converged"]), crit=crit,
                         R=float(d["R_history"][-1]), field=field_rms(Xl, P, T),
                         shape=quantities(Xl, P, seam) - quantities(Xl, Xl, seam), src=os.path.basename(p)))
    return rows


def lumped_area(P, T):
    a = 0.5 * np.abs(np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]))
    w = np.zeros(len(P))
    for k in range(3):
        np.add.at(w, T[:, k], a / 3)
    return w, a


def mls1(P, F, w, sig):
    """Local-linear Gaussian smoother at every node (exact for linear F)."""
    tree = cKDTree(P)
    out = np.empty_like(F)
    for i, nb in enumerate(tree.query_ball_point(P, 3 * sig)):
        d = P[nb] - P[i]
        W = w[nb] * np.exp(-0.5 * (d ** 2).sum(1) / sig ** 2)
        B = np.c_[np.ones(len(nb)), d]
        c = np.linalg.lstsq(B * np.sqrt(W)[:, None], F[nb] * np.sqrt(W)[:, None], rcond=None)[0]
        out[i] = c[0]
    return out


def rigid_fit_area(A, B, w):
    """Rigid fit Q A ~ B in least squares with nodal weights w (lumped area); returns Q A.  An unweighted fit counts
    nodes, so on a tip-graded mesh it pins the tip and moves the residual to the rim."""
    ca, cb = (w[:, None] * A).sum(0) / w.sum(), (w[:, None] * B).sum(0) / w.sum()
    U, _, Vt = np.linalg.svd(((A - ca) * w[:, None]).T @ (B - cb))
    if np.linalg.det(U @ Vt) < 0:
        Vt[-1] *= -1
    return (A - ca) @ (U @ Vt) + cb
