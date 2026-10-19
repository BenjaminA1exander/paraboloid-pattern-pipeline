r"""build_seamtip_ladder.py -- SEAM- and TIP-resolved nested gore meshes for the one-gore Abaqus sector ladder.  Meshes, patterns for the quality check, checks and figures only: no deck is written, nothing is solved.

we get convergence.  Seam elements and tip elements must also undergo subdivision at the same rate or it is not a fair
comparison ... the optimum mesh such that it and its subdivisions are of the highest quality possible.  This will be
confirmed before running."  Coarsest seam element chosen by the user: 6.25 mm (h/16 at the 100 mm rung; weft rms, seam
force and crease all have shrinking successive changes from a 6.1 mm seam element down, seam-resolution study).

Base mesh (level 0), meshed in the developed leaf (x = meridian arc length, |y| <= hb(x) = r(s) theta) like
formfound_profile.build_gore_gmsh, but
  * size field  min( h,  h (0.02 + k d_tip),  S0 + G d_seam )   h = 100 mm, k = 2 theta / 210 mm (the tip-polar C=210
    grading of the current ladder, unchanged), S0 = 6.25 mm, G = 0.4 (the smallest growth rate tested that keeps every
    non-apex angle >= 30 deg; 0.5 left two at 28.7 / 29.6 deg);
  * EXACT apex: the profile closes the gore exactly (r(s_hi) = 0, hb = theta d), so there is no TIP_H taper.  The
    wedge (opening 2 atan(theta) = 14.9 deg) is narrower than the size field for d < ~15 mm, which is where every bad
    element of the current meshes sits (12.4 deg at 2.9 mm from the apex); there a STRUCTURED STRIP replaces gmsh:
    apex triangle to the row at d1 = 0.02 h, then rows at geometric d (ratio ~1 + 2 theta, square cells) up to
    DT = 20 mm, each cell split into two triangles (diagonals alternating).  The apex triangle's 14.9 deg angle is set
    by the gore geometry; red refinement keeps its descendants similar to it (they stay inside the level-0 apex
    triangle, 2 mm long);
  * outside the strip: gmsh Frontal-Delaunay (Algorithm 6, Smoothing 10), the -seam curve PERIODIC (reflection) with
    the +seam, then the -seam nodes set to the exact mirror of the +seam nodes and both snapped onto y = +-hb(x)
    (moves <= 3e-5 mm), so the cyclic seam pairs are exact;
  * node 0 = the tip (the strip is numbered first).
Levels 1..4: red refinement (every triangle -> 4 at its edge midpoints) in the leaf, the scheme of gore_subdivision.py,
with a seam-edge midpoint put onto the EXACT seam curve (hb evaluated from the profile spline, not gore_subdivision's
tapered table); rim midpoints stay on the straight rim; coarse nodes keep leaf AND spatial coordinates bit for bit.
Spatial map: gore_subdivision.leaf_to_spatial (build_gore_gmsh's node map).  Targets T: element-centre values of
prof.sigma_field at the spatial centroid (= stage3.element_centre_targets for S3; checked to 1e-13 on the old store).

Patterns, here ONLY for the flat-pattern quality check (the decks take them in sector_seamtip_ladder.py):
  vares    u(x) = X_converged - X_leaf of the converged 6.25 mm VaReS run (meshstudy_vares_cstraw_tippolar210nestgrade/
           h006/pattern.npz): cubic spline per seam side through the h006 seam nodes (the pattern is not mirror
           symmetric), cubic spline along the rim, P1 on the h006 mesh inside;
  uniform  X_leaf * (1 + c_m, 1 + c_w), c of the current h100 rung (-0.7693 / -0.6963 %) -- quality is insensitive to it.
Quality per level and configuration (leaf, VaReS flat, uniform flat, target surface): min / max angle, radius ratio
q = 2 r_in / R_circ, grading jump (largest area ratio across an edge), excluding the apex-triangle descendants, which
are reported separately.  Targets set before the meshes were seen: min angle >= 30 deg, max <= 120 deg, q >= 0.5,
jump <= 2 (except at the apex triangle).

Runs in the BASE anaconda env (gmsh; no jax needed).  Laptop: ~1.5 M elements at level 4, a few minutes.
OUT  temp/gore_seamtip_s625/<tag>.npz (h100 h050 h025 h012 h006) + checks.json   (never overwritten: stops if present)
     results/abaqus_ladder_seamtip/mesh/fig_seamtip_mesh_design.{png,pdf}
USAGE  python build_seamtip_ladder.py [--check | --figure-only]   (--check: build and report, write nothing;
       --figure-only: rebuild in memory and redraw the figure, store untouched)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paraboloid_dish_config as CFG              # noqa: E402
import paraboloid_profile_cst as PC               # noqa: E402
import gore_subdivision as GSUB                   # noqa: E402

N_GORES, GORE = 24, 0
TH = np.pi / N_GORES
H0, LEVELS = 100.0, 4
S0, G = 6.25, 0.4                                 # seam element at level 0, growth rate (quality sweep)
TIP_G0, TIP_C = 0.02, 210.0                       # tip-polar grading of the current ladder
TIP_K = 2.0 * TH / TIP_C
DT, D1 = 20.0, TIP_G0 * H0                        # structured tip strip: extent, first row
NB = 801                                          # gmsh seam spline samples
ALG, SMOOTH = 6, 10                               # gmsh 2D algorithm (Frontal-Delaunay), smoothing passes
UNIFORM_C = (-0.007693, -0.006963)                # h100 uniform_compensation, quality check only
STORE = os.path.join(CFG.TEMP_DIR, "gore_seamtip_s625")
FIG_DIR = os.path.join(HERE, "results", "abaqus_ladder_seamtip", "mesh")
VARES_H006 = os.path.join(CFG.TEMP_DIR, "meshstudy_vares_cstraw_tippolar210nestgrade", "h006", "pattern.npz")
GRADING = "seamtip_s625_g04_tippolar210"


def tag(h):
    return "h%03d" % int(round(h))              # = MeshStudy_paraboloid_vares.tag


PROF = PC.extract_paraboloid_cst(gore=GORE)[0]
GS_, SMID = PROF.geom_scale, 0.5 * (PROF.s_lo + PROF.s_hi)
XMIN, XMAX = (PROF.s_lo - SMID) * GS_, (PROF.s_hi - SMID) * GS_


def hb(x):
    """exact developed half-width r(s(x)) theta (mm)."""
    s = np.clip(np.asarray(x, float) / GS_ + SMID, PROF.s_lo, PROF.s_hi)
    return np.maximum(PROF.spl_r(s) * GS_ * TH, 0.0)


# ------------------------------------------------------------------------------------------------ base mesh
def tip_strip():
    K = max(1, int(round(np.log(DT / D1) / np.log(1.0 + 2.0 * TH))))
    d = D1 * (DT / D1) ** (np.arange(K + 1) / K)
    P = [(XMAX, 0.0)]
    for x in XMAX - d:
        y = float(hb(x))
        P += [(x, y), (x, -y)]
    T = [(0, 2, 1)]
    for k in range(K):
        a, b, c, e = 1 + 2 * k, 2 + 2 * k, 3 + 2 * k, 4 + 2 * k     # row k top/bottom, row k+1 top/bottom
        T += [(a, b, e), (a, e, c)] if k % 2 == 0 else [(a, b, c), (b, e, c)]
    return np.array(P), np.array(T), XMAX - d[-1]


def outer_gmsh(xt):
    import gmsh
    s = np.linspace(PROF.s_lo, PROF.s_hi, NB)
    xb = (s - SMID) * GS_
    m = xb < xt - 1e-9
    xb = np.r_[xb[m], xt]
    yb = hb(xb)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        geo = gmsh.model.geo
        pt = [geo.addPoint(x, y, 0) for x, y in zip(xb, yb)]
        pb = [geo.addPoint(x, -y, 0) for x, y in zip(xb, yb)]
        ct, cb = geo.addSpline(pt), geo.addSpline(pb)
        cr, cx = geo.addLine(pb[0], pt[0]), geo.addLine(pt[-1], pb[-1])
        geo.addPlaneSurface([geo.addCurveLoop([ct, cx, -cb, cr])])
        geo.mesh.setTransfiniteCurve(cx, 2)                     # one edge: the strip's last row
        geo.synchronize()
        gmsh.model.mesh.setPeriodic(1, [cb], [ct], [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        F = gmsh.model.mesh.field
        ds = F.add("Distance")
        F.setNumbers(ds, "CurvesList", [ct, cb])
        F.setNumber(ds, "Sampling", 20000)
        fm = F.add("MathEval")
        F.setString(fm, "F", f"Min({H0},Min({H0 * TIP_G0}+{H0 * TIP_K}*Sqrt((x-{XMAX})^2+y^2),{S0}+{G}*F{ds}))")
        F.setAsBackgroundMesh(fm)
        for o in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
            gmsh.option.setNumber("Mesh." + o, 0)
        gmsh.option.setNumber("Mesh.Algorithm", ALG)
        gmsh.option.setNumber("Mesh.Smoothing", SMOOTH)
        gmsh.model.mesh.generate(2)
        tags, co, _ = gmsh.model.mesh.getNodes()
        co = np.asarray(co).reshape(-1, 3)[:, :2]
        _, en = gmsh.model.mesh.getElementsByType(2)
        idx = np.zeros(int(tags.max()) + 1, int)
        idx[tags.astype(int)] = np.arange(len(tags))
        return co, idx[np.asarray(en, int).reshape(-1, 3)], gmsh.__version__
    finally:
        gmsh.finalize()


def boundary_sets(X, T):
    E = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    uE, cnt = np.unique(E, axis=0, return_counts=True)
    bnd = np.unique(uE[cnt == 1])
    on_rim = np.abs(X[bnd, 0] - XMIN) < 1e-9
    corner = on_rim & (np.abs(np.abs(X[bnd, 1]) - float(hb(XMIN))) < 1e-6)
    seam = bnd[~on_rim | corner]                                # seams incl. tip and the two rim corners
    return seam, bnd


def base_mesh():
    P, Tp, xt = tip_strip()
    co, To, gv = outer_gmsh(xt)
    X = np.r_[P, co]
    T = np.r_[Tp, To + len(P)]
    _, first, inv = np.unique(np.round(X, 7), axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(first)                                   # strip first -> tip = node 0
    new = np.empty(len(first), int)
    new[order] = np.arange(len(first))
    X, T = X[first[order]], new[inv[T]]
    used = np.unique(T)                                          # drop gmsh's unused spline control points
    m = -np.ones(len(X), int)
    m[used] = np.arange(len(used))
    X, T = X[used].copy(), m[T]
    a = X[T]
    ar = (a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1])
    T[ar < 0] = T[ar < 0][:, [0, 2, 1]]                          # all counter-clockwise
    seam, bnd = boundary_sets(X, T)
    top = seam[X[seam, 1] > 0]
    bot = seam[X[seam, 1] < 0]
    top, bot = top[np.argsort(X[top, 0])], bot[np.argsort(X[bot, 0])]
    assert len(top) == len(bot), "seam sides differ in node count"
    mirror_fix = float(np.abs(X[bot] - X[top] * [1, -1]).max())
    X[bot] = X[top] * [1, -1]                                    # exact mirror pairs
    snap = float(np.abs(np.abs(X[top, 1]) - hb(X[top, 0])).max())
    X[top, 1] = hb(X[top, 0])
    X[bot, 1] = -hb(X[bot, 0])
    assert int(np.argmax(X[:, 0])) == 0 and X[0, 1] == 0.0
    return X, T, seam, bnd, dict(gmsh=gv, mirror_fix_mm=mirror_fix, seam_snap_mm=snap, strip_elements=len(Tp),
                                  strip_rows=int((len(P) - 1) // 2), strip_extent_mm=DT)


def subdivide(X, T, seam, bnd):
    """red refinement; seam midpoints onto the exact seam; coarse nodes 0..N-1 unchanged."""
    N = len(X)
    E = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    uE, inv, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    on_b = cnt == 1
    iss, isb = np.zeros(N, bool), np.zeros(N, bool)
    iss[seam], isb[bnd] = True, True
    mid = 0.5 * (X[uE[:, 0]] + X[uE[:, 1]])
    se = on_b & iss[uE[:, 0]] & iss[uE[:, 1]]
    be = on_b & isb[uE[:, 0]] & isb[uE[:, 1]]
    mid[se, 1] = np.sign(mid[se, 1]) * hb(mid[se, 0])
    m_id = N + np.arange(len(uE))
    ne = len(T)
    e01, e12, e20 = m_id[inv[:ne]], m_id[inv[ne:2 * ne]], m_id[inv[2 * ne:]]
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    Tn = np.vstack([np.c_[a, e01, e20], np.c_[e01, b, e12], np.c_[e20, e12, c], np.c_[e01, e12, e20]])
    return np.r_[X, mid], Tn, np.sort(np.r_[seam, m_id[se]]), np.sort(np.r_[bnd, m_id[be]]), uE


# ------------------------------------------------------------------------------------------------ patterns
class VaresMap:
    """leaf -> flat VaReS pattern from the converged h006 run."""

    def __init__(self):
        import matplotlib.tri as mtri
        from scipy.interpolate import CubicSpline
        p = np.load(VARES_H006, allow_pickle=True)
        XL, XC = p["X_leaf"][:, :2], p["X_converged"][:, :2]
        T6, S6 = np.asarray(p["elements"]), np.asarray(p["seam_ids"])
        self.U = XC - XL
        tri = mtri.Triangulation(XL[:, 0], XL[:, 1], T6)
        self.fx = mtri.LinearTriInterpolator(tri, self.U[:, 0])
        self.fy = mtri.LinearTriInterpolator(tri, self.U[:, 1])
        tip = int(np.argmax(XL[:, 0]))
        self.spl, self.ids = {}, {}
        for sg in (1, -1):
            ids = S6[sg * XL[S6, 1] > 1e-9]
            ids = np.r_[ids, [tip]]
            ids = ids[np.argsort(XL[ids, 0])]
            self.spl[sg], self.ids[sg] = CubicSpline(XL[ids, 0], self.U[ids], axis=0), ids
        rim = np.nonzero(np.abs(XL[:, 0] - XL[:, 0].min()) < 1e-6)[0]
        self.rim = rim[np.argsort(XL[rim, 1])]
        self.splr = CubicSpline(XL[self.rim, 1], self.U[self.rim], axis=0)
        self.XL, self.XC = XL, XC
        self.check = max(max(float(np.abs(self.spl[sg](XL[self.ids[sg], 0]) - self.U[self.ids[sg]]).max()) for sg in (1, -1)),
                         float(np.abs(self.splr(XL[self.rim, 1]) - self.U[self.rim]).max()))

    def __call__(self, X, seam, rim):
        U = np.c_[np.asarray(self.fx(X[:, 0], X[:, 1]).filled(np.nan)), np.asarray(self.fy(X[:, 0], X[:, 1]).filled(np.nan))]
        out = np.isnan(U[:, 0])
        iss = np.zeros(len(X), bool)
        iss[seam], iss[rim] = True, True
        stray = out & ~iss                                       # interior nodes between an h006 seam chord and the curve
        for sg in (1, -1):
            k = (out | iss) & (np.sign(X[:, 1]) == sg)
            U[k] = self.spl[sg](X[k, 0])
        U[rim] = self.splr(X[rim, 1])
        U[0] = self.spl[1](XMAX)
        dstray = float((hb(X[stray, 0]) - np.abs(X[stray, 1])).max()) if stray.any() else 0.0
        return X + U, int(stray.sum()), dstray


# ------------------------------------------------------------------------------------------------ quality
def angles(Y, T):
    P = Y[T]
    out = []
    for i in range(3):
        u, v = P[:, (i + 1) % 3] - P[:, i], P[:, (i + 2) % 3] - P[:, i]
        out.append(np.degrees(np.arccos(np.clip((u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1), -1, 1))))
    return np.array(out).T


def areas(Y, T):
    P = Y[T]
    c = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    return 0.5 * (np.abs(c) if c.ndim == 1 else np.linalg.norm(c, axis=1)), c


def quality(Y, T, apexd):
    A = angles(Y, T)
    mn, mx = A.min(1), A.max(1)
    P = Y[T]
    L = np.stack([np.linalg.norm(P[:, (i + 1) % 3] - P[:, i], axis=1) for i in range(3)], 1)
    ar, c = areas(Y, T)
    q = 2.0 * (2.0 * ar / L.sum(1)) / (L.prod(1) / (4.0 * ar))
    # grading jump across interior edges
    E = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1).astype(np.int64)
    el = np.tile(np.arange(len(T)), 3)
    key = E[:, 0] * (int(E.max()) + 1) + E[:, 1]
    o = np.argsort(key, kind="stable")
    k, e = key[o], el[o]
    j = np.nonzero(k[1:] == k[:-1])[0]
    a1, a2 = e[j], e[j + 1]
    r = np.maximum(ar[a1] / ar[a2], ar[a2] / ar[a1])
    ok = ~(apexd[a1] | apexd[a2])
    na = ~apexd
    inv = int((c <= 0).sum()) if c.ndim == 1 else 0
    return dict(min_angle=float(mn[na].min()), p1_min_angle=float(np.percentile(mn[na], 1)),
                median_min_angle=float(np.median(mn[na])), n_below_30=int((mn[na] < 30).sum()),
                max_angle=float(mx[na].max()), n_above_120=int((mx[na] > 120).sum()), q_min=float(q[na].min()),
                q_p1=float(np.percentile(q[na], 1)), jump_max=float(r[ok].max()), jump_p99=float(np.percentile(r[ok], 99)),
                jump_max_incl_apex=float(r.max()), inverted=inv,
                apex=dict(n=int(apexd.sum()), min_angle=float(mn[apexd].min()), max_angle=float(mx[apexd].max()))), mn


def spatial(X, seam):
    iss = np.zeros(len(X), bool)
    iss[seam] = True
    return GSUB.leaf_to_spatial(PROF, X[:, 0], X[:, 1], iss, N_GORES, GORE)


def targets(Xs, T):
    c = Xs[T].mean(1)
    return np.array([PROF.sigma_field(p) for p in c])


def sizes(X, T, seam):
    """seam edge (mm, p5 / median / p95 on the seam away from the tip strip), tip edges, interior h_eff."""
    E = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    uE, cnt = np.unique(E, axis=0, return_counts=True)
    iss = np.zeros(len(X), bool)
    iss[seam] = True
    b = uE[(cnt == 1) & iss[uE[:, 0]] & iss[uE[:, 1]]]
    Lb = np.linalg.norm(X[b[:, 0]] - X[b[:, 1]], axis=1)
    far = (XMAX - X[b].mean(1)[:, 0]) > 100.0
    tipel = (T == 0).any(1)
    Lt = np.linalg.norm(X[T[tipel]] - X[T[tipel]][:, [1, 2, 0]], axis=2)
    cen = X[T].mean(1)
    inner = (hb(cen[:, 0]) - np.abs(cen[:, 1]) > 200.0) & ((XMAX - cen[:, 0]) > 1000.0)
    ai = areas(X, T[inner])[0]
    return dict(seam_edge_mm=np.percentile(Lb[far], [5, 50, 95]).tolist(), tip_edge_mm=[float(Lt.min()), float(Lt.max())],
                interior_h_eff_mm=float(np.sqrt(4.0 * ai.mean() / np.sqrt(3.0))))


# ------------------------------------------------------------------------------------------------ figure
def figure(levels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
    X0, T0 = levels[0]["X"], levels[0]["T"]
    X1, T1 = levels[1]["X"], levels[1]["T"]

    def edges(X, T, box=None):
        E = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
        seg = X[E]
        if box is not None:
            x0, x1, y0, y1 = box
            c = seg.mean(1)
            seg = seg[(c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 1] > y0) & (c[:, 1] < y1)]
        return seg
    W, H = 180.0, 150.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    F = lambda x, y, w, hh: [x / W, y / H, w / W, hh / H]     # noqa: E731
    ax = fig.add_axes(F(12, 100, 163, 45))
    ax.add_collection(LineCollection(edges(X0, T0), colors="0.25", linewidths=0.2))
    ax.set_xlim(XMIN - 20, XMAX + 20)
    ax.set_ylim(-350, 350)
    ax.set_aspect("equal")
    ax.set_xlabel("Leaf $x$ (mm)")
    ax.set_ylabel("Leaf $y$ (mm)")
    fig.text(2 / W, 147 / H, "(a)", fontsize=10, fontweight="bold")
    # (b) tip, level 0 and level 1
    ax = fig.add_axes(F(12, 57, 75, 30))
    box = (XMAX - 60, XMAX + 2, -10, 10)
    ax.add_collection(LineCollection(edges(X1, T1, box), colors="#E69F00", linewidths=0.35, label="Level 1"))
    ax.add_collection(LineCollection(edges(X0, T0, box), colors="#0072B2", linewidths=0.7, label="Level 0"))
    ax.set_xlim(XMAX - 60, XMAX + 1)
    ax.set_ylim(-9, 9)
    ax.set_aspect("equal")
    ax.set_xlabel("Leaf $x$ (mm)")
    ax.set_ylabel("Leaf $y$ (mm)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7, handlelength=1.5)
    fig.text(2 / W, 92 / H, "(b)", fontsize=10, fontweight="bold")
    # (c) seam band at mid-gore, level 0
    xm = 0.0
    ax = fig.add_axes(F(100, 57, 75, 30))
    yh = float(hb(xm))
    ytop = float(hb(xm - 100)) + 1.0                                # the seam rises towards the rim (left)
    box = (xm - 120, xm + 120, yh - 100, ytop + 2)
    ax.add_collection(LineCollection(edges(X1, T1, box), colors="#E69F00", linewidths=0.35))
    ax.add_collection(LineCollection(edges(X0, T0, box), colors="#0072B2", linewidths=0.7))
    ax.set_xlim(xm - 100, xm + 100)
    ax.set_ylim(yh - 70, ytop)
    ax.set_aspect("equal")
    ax.set_xlabel("Leaf $x$ (mm)")
    ax.set_ylabel("Leaf $y$ (mm)")
    fig.text(90 / W, 92 / H, "(c)", fontsize=10, fontweight="bold")
    # (d) min-angle distribution per level (target surface), (e) per configuration at level 4
    ax = fig.add_axes(F(12, 12, 75, 32))
    cols = ["#000000", "#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    ls = ["-", (0, (5, 2)), (0, (2, 1.5)), "-.", (0, (1, 1))]
    bins = np.linspace(0, 60, 121)
    for L, lv in enumerate(levels):
        mn = lv["minang"]["spatial"]
        hgm, _ = np.histogram(mn, bins)
        ax.plot(0.5 * (bins[1:] + bins[:-1]), hgm / hgm.sum(), color=cols[L], ls=ls[L], lw=1.0, label=f"Level {L}")
    ax.axvline(30, color="0.5", lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("Smallest element angle (deg)")
    ax.set_ylabel("Fraction of elements")
    ax.set_xlim(10, 60)
    ax.legend(fontsize=7, handlelength=2.2)
    fig.text(2 / W, 48 / H, "(d)", fontsize=10, fontweight="bold")
    ax = fig.add_axes(F(100, 12, 75, 32))
    lv = levels[-1]
    for i, (k, lab) in enumerate((("leaf", "Leaf"), ("vares", "VaReS pattern"), ("uniform", "Uniform pattern"), ("spatial", "Target surface"))):
        hgm, _ = np.histogram(lv["minang"][k], bins)
        ax.plot(0.5 * (bins[1:] + bins[:-1]), hgm / hgm.sum(), color=cols[i], ls=ls[i], lw=1.0, label=lab)
    ax.axvline(30, color="0.5", lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("Smallest element angle (deg)")
    ax.set_ylabel("Fraction of elements")
    ax.set_xlim(10, 60)
    ax.legend(fontsize=7, handlelength=2.2)
    fig.text(90 / W, 48 / H, "(e)", fontsize=10, fontweight="bold")
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(f"{path}.{ext}", **kw)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def main():
    fig_only = "--figure-only" in sys.argv
    write = "--check" not in sys.argv and not fig_only
    if write and os.path.exists(STORE) and os.listdir(STORE):
        sys.exit(f"[seamtip] {STORE} exists and is not empty -- not overwriting")
    t0 = time.time()
    X, T, seam, bnd, binfo = base_mesh()
    vm = VaresMap()
    print(f"[seamtip] base: {len(T)} elements, {len(X)} nodes, strip {binfo['strip_elements']} elements "
          f"({binfo['strip_rows']} rows to {DT} mm); mirror fix {binfo['mirror_fix_mm']:.1e} mm, seam snap "
          f"{binfo['seam_snap_mm']:.1e} mm; VaReS map reproduces h006 seam/rim nodes to {vm.check:.1e} mm")
    checks = dict(base=binfo, parameters=dict(h0=H0, seam_size0=S0, growth=G, tip_g0=TIP_G0, tip_C_mm=TIP_C,
                                              strip_extent_mm=DT, strip_first_row_mm=D1, uniform_c=UNIFORM_C),
                  vares_map_check_mm=vm.check, levels=[])
    levels = []
    Xs_prev = None
    XL_prev = None
    for L in range(LEVELS + 1):
        if L:
            Xn, Tn, seam, bnd, _ = subdivide(X, T, seam, bnd)
            assert np.array_equal(Xn[:len(X)], X)
            X, T = Xn, Tn
        h = H0 / 2 ** L
        rim = bnd[np.abs(X[bnd, 0] - XMIN) < 1e-9]
        Xs = spatial(X, seam)
        if Xs_prev is not None:
            Xs[:len(Xs_prev)] = Xs_prev                         # nested: coarse spatial nodes bit for bit
        top, bot = seam[X[seam, 1] > 0], seam[X[seam, 1] < 0]
        top, bot = top[np.argsort(X[top, 0])], bot[np.argsort(X[bot, 0])]
        pairs_ok = bool(len(top) == len(bot) and np.array_equal(X[top, 0], X[bot, 0]) and np.array_equal(X[top, 1], -X[bot, 1]))
        on_curve = float(np.abs(np.abs(X[seam, 1]) - hb(X[seam, 0])).max())
        Xv, nstray, dstray = vm(X, seam, rim)
        Xu = X * (1.0 + np.asarray(UNIFORM_C))
        cen = X[T].mean(1)
        apexd = cen[:, 0] > XMAX - D1 * (1 - 1e-9)              # descendants of the level-0 apex triangle
        rec = dict(level=L, tag=tag(h), h_nominal=h, seam_size_nominal=S0 / 2 ** L, n_elements=len(T),
                   n_elements_full_dish=len(T) * N_GORES, n_nodes=len(X), n_seam_pairs=len(top), seam_pairs_exact=pairs_ok,
                   seam_on_curve_mm=on_curve, tip_node0=bool(int(np.argmax(X[:, 0])) == 0),
                   nested_leaf_exact=bool(XL_prev is None or np.array_equal(X[:len(XL_prev)], XL_prev)),
                   vares_stray_interior_nodes=nstray, vares_stray_max_dist_from_seam_mm=dstray, **sizes(X, T, seam),
                   quality={})
        mins = {}
        for k, Y in (("leaf", X), ("vares", Xv), ("uniform", Xu), ("spatial", Xs)):
            rec["quality"][k], mins[k] = quality(Y, T, apexd)
        q = rec["quality"]
        print(f"[seamtip] L{L} {tag(h)}: {len(T):8d} el ({len(T) * N_GORES / 1e6:.2f} M full dish), seam edge p5/50/95 "
              f"{np.round(rec['seam_edge_mm'], 3)} mm, tip edges {np.round(rec['tip_edge_mm'], 3)} mm, interior h_eff "
              f"{rec['interior_h_eff_mm']:.1f} mm; pairs exact {pairs_ok}, on curve {on_curve:.1e}, stray {nstray} "
              f"({time.time() - t0:.0f} s)")
        for k in ("leaf", "vares", "uniform", "spatial"):
            r = q[k]
            print(f"           {k:8s} angle {r['min_angle']:5.1f} (p1 {r['p1_min_angle']:5.1f}, <30: {r['n_below_30']}) .. "
                  f"{r['max_angle']:6.1f} (>120: {r['n_above_120']}), q_min {r['q_min']:.3f}, jump max {r['jump_max']:.2f} "
                  f"p99 {r['jump_p99']:.2f}, inverted {r['inverted']} | apex triangle x{r['apex']['n']}: "
                  f"{r['apex']['min_angle']:.1f}/{r['apex']['max_angle']:.1f}")
        lv = dict(X=X, T=T, minang=mins)
        if write:
            os.makedirs(STORE, exist_ok=True)
            Tt = targets(Xs, T)
            np.savez(os.path.join(STORE, tag(h) + ".npz"), X_spatial=Xs, X_leaf=np.c_[X, np.zeros(len(X))],
                     seam_ids=np.asarray(seam, int), boundary_ids=np.asarray(bnd, int), elements=T.astype(np.int32), T=Tt,
                     X_vares_flat=np.c_[Xv, np.zeros(len(X))], family="seamtip", level=L, h_nominal=h, base_h=H0,
                     n_parent_nodes=0 if XL_prev is None else len(XL_prev), target="cstraw", grading=GRADING,
                     tip_affine=np.array([TIP_G0, TIP_K]), seam_size=S0 / 2 ** L, seam_growth=G, n_gores=N_GORES,
                     gore_index=GORE, vares_source=VARES_H006,
                     note="seam+tip-resolved nested gore mesh (build_seamtip_ladder.py): structured tip strip + gmsh base, "
                          "red refinement, seam midpoints on the exact seam; nodes 0..n_parent_nodes-1 = the parent level")
            print(f"           -> {os.path.join(STORE, tag(h) + '.npz')} (targets {time.time() - t0:.0f} s)")
        levels.append(lv)
        checks["levels"].append(rec)
        Xs_prev, XL_prev = Xs, X
    if write:
        with open(os.path.join(STORE, "checks.json"), "w") as f:
            json.dump(checks, f, indent=1, default=float)
    if write or fig_only:
        os.makedirs(FIG_DIR, exist_ok=True)
        figure(levels, os.path.join(FIG_DIR, "fig_seamtip_mesh_design"))
        print("[seamtip] figure ->", FIG_DIR, "" if fig_only else "; checks -> " + os.path.join(STORE, "checks.json"))


if __name__ == "__main__":
    main()
