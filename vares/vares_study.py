r"""vares_study.py -- the shared conventions of the VaReS mesh studies: where a rung's result lives, how a rung is
named, and the leaf-sampling helpers the worker and the figures use.

    ROOT, LADDER, MAX_IT     the default study root, the sqrt(2) ladder of nominal edge lengths (mm) and the
                             Newton iteration budget (MESHSTUDY_MAX_IT; a budget, never a tolerance).
                             MeshStudy_paraboloid_vares_ckpt.configure re-points ROOT at the target's study.
    tag, rung_dir, rung_npz  "h100" for 100 mm; <ROOT>/h100; <ROOT>/h100/pattern.npz
    leaf_half_width          half-width of the developed leaf at a meridian station, r(s) theta
    p1_sample                a nodal field of the gore mesh evaluated at points of the leaf
    rigid_fit                the best 2-D rigid fit of one point set onto another
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paraboloid_dish_config as CFG            # noqa: E402

ROOT = os.path.join(CFG.TEMP_DIR, "meshstudy_vares")
LADDER = [200.0, 141.4, 100.0, 70.7, 50.0, 35.4, 25.0]       # sqrt(2) ladder in nominal edge length
MAX_IT = int(os.environ.get("MESHSTUDY_MAX_IT", "300"))


def tag(h):
    return "h%03d" % int(round(h))


def rung_dir(h):
    return os.path.join(ROOT, tag(h))


def rung_npz(h):
    return os.path.join(rung_dir(h), "pattern.npz")


_PROF = None                                     # the gore profile (set by MeshStudy_paraboloid_vares_ckpt.configure)


def _prof():
    global _PROF
    if _PROF is None:
        import paraboloid_profile as pp
        _PROF = pp.extract_paraboloid()[0]
    return _PROF


def leaf_half_width(x_mm, d):
    """h(x) exactly as build_gore_gmsh draws the seam: r(s) * theta."""
    prof = _prof()
    s = x_mm / d["geom_scale"] + d["s_mid_mm"] / d["geom_scale"]
    return prof.spl_r(s) * d["geom_scale"] * np.pi / CFG.N_GORES


def p1_sample(leaf_xy, els, values, pts):
    """Evaluate the P1 field `values` (per node) at leaf points `pts`.  Points just outside the
    chordal boundary (the seam is a spline, the mesh edge a chord) are linearly extrapolated
    from the nearest element, an O(h^2) gap."""
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(leaf_xy[:, 0], leaf_xy[:, 1], els)
    ti = tri.get_trifinder()(pts[:, 0], pts[:, 1])
    cen = leaf_xy[els].mean(axis=1)
    out = np.empty((len(pts), values.shape[1]))
    for k, (p, t) in enumerate(zip(pts, ti)):
        if t < 0:
            t = int(np.argmin(((cen - p) ** 2).sum(1)))
        a, b, c = leaf_xy[els[t]]
        M = np.array([b - a, c - a]).T
        l1, l2 = np.linalg.solve(M, p - a)
        out[k] = (1 - l1 - l2) * values[els[t, 0]] + l1 * values[els[t, 1]] \
            + l2 * values[els[t, 2]]
    return out


def rigid_fit(A, B):
    """Q = (R, t) minimising |R A + t - B| in 2-D; returns R A + t."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    Rm = (U @ Vt).T
    if np.linalg.det(Rm) < 0:
        Vt[-1] *= -1
        Rm = (U @ Vt).T
    return (A - ca) @ Rm.T + cb
