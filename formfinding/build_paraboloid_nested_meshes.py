r"""build_paraboloid_nested_meshes.py -- NESTED mesh ladder for the paraboloid form-finding study, by uniform 1->4
subdivision of one base mesh (replaces independent gmsh + ODT remeshing per rung, whose irregular nodes move
between rungs and make the stress-error constant jump).

Scheme, per level (all in the paraboloid's CONFORMAL plane, where the base mesh was built isotropic):
  * every edge gets one new node at its midpoint:
        interior edge: Cartesian midpoint of the two end nodes in the conformal plane (rho, theta),
                       mapped back to r(rho) and lifted onto z = DEPTH (1 - r^2/R^2);
        rim edge     : the angular midpoint ON the rim circle r = R (z = 0), so the rim stays a circle;
  * every triangle (a, b, c) becomes (a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca) -- same orientation;
  * old nodes never move; every new interior node has valence 6, so the base mesh's irregular nodes
    (valence != 6) are the ONLY irregular nodes on every level, at fixed positions;
  * the apex node and the rim nodes of the base are kept; new rim midpoints join the rim (pinned) set.
Every level therefore contains the previous one exactly (nested), and h halves per level.

Base: the 289-node ODT option-B mesh of the existing 289 rung (paraboloid_mesh_odt/mesh_u0289_odt_B_ladder.npz).
Outputs: temp/paraboloid_mesh_nested/mesh_L{0..3}.npz (solver mesh format + parent bookkeeping) and the
scheme figure temp/paraboloid_mesh_nested/fig_nested_scheme.{png,pdf}.

Run (base env): python build_paraboloid_nested_meshes.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paraboloid_mesh_odt as PO                       # noqa: E402  (quality(), conformal tables)

import paraboloid_dish_config as CFG
TEMP = CFG.TEMP_DIR
BASE = f"{TEMP}/paraboloid_mesh_odt/mesh_u0289_odt_B_ladder.npz"
OUT_DIR = f"{TEMP}/paraboloid_mesh_nested"
N_LEVELS = 3
R, D = PO.R, PO.D
MM = 2500.0 / R
r_t, rho_t = PO._r_t, PO._rho_t


def to_conformal(P):
    return PO.conformal_xy(P)


def from_conformal(Q):
    rho = np.hypot(Q[:, 0], Q[:, 1])
    r = np.interp(rho, rho_t, r_t)
    sc = np.where(rho > 1e-14, r / np.where(rho > 1e-14, rho, 1.0), 1.0)
    return np.column_stack([Q[:, 0] * sc, Q[:, 1] * sc, D * (1.0 - r ** 2 / R ** 2)])


def subdivide(P, T, rim):
    """One 1->4 level. Returns (P', T', rim', edge midpoint map)."""
    E = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    is_rim = np.zeros(len(P), bool); is_rim[rim] = True
    # an edge is a RIM edge if it has only one adjacent triangle
    cnt = {}
    for tri in T:
        for i, j in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            k = (min(i, j), max(i, j)); cnt[k] = cnt.get(k, 0) + 1
    Q = to_conformal(P)
    newP, mid = [], {}
    for n_, (a, b) in enumerate(E):
        if cnt[(a, b)] == 1:                                   # boundary edge -> on the rim circle
            th_a, th_b = np.arctan2(P[a, 1], P[a, 0]), np.arctan2(P[b, 1], P[b, 0])
            dth = np.angle(np.exp(1j * (th_b - th_a)))
            th = th_a + 0.5 * dth
            newP.append([R * np.cos(th), R * np.sin(th), 0.0])
        else:
            newP.append(from_conformal(0.5 * (Q[a] + Q[b])[None])[0])
        mid[(a, b)] = len(P) + n_
    P2 = np.vstack([P, np.asarray(newP)])
    m = lambda i, j: mid[(min(i, j), max(i, j))]          # noqa: E731
    T2 = []
    for a, b, c in T:
        ab, bc, ca = m(a, b), m(b, c), m(c, a)
        T2 += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
    rim2 = np.concatenate([rim, [mid[k] for k in mid if cnt[k] == 1]]).astype(int)
    return P2, np.asarray(T2, int), rim2


def valence(P, T):
    E = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    return np.bincount(E.ravel(), minlength=len(P)), E


def min_angles(P, T):
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    out = []
    for p_, q_, r_ in ((a, b, c), (b, c, a), (c, a, b)):
        u, v = q_ - p_, r_ - p_
        out.append(np.degrees(np.arccos(np.clip(np.sum(u * v, 1) / np.linalg.norm(u, axis=1)
                                                / np.linalg.norm(v, axis=1), -1, 1))))
    return np.min(out, 0), np.max(out, 0)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    b = np.load(BASE, allow_pickle=True)
    P, T, rim, apex = np.asarray(b["vertices"]), np.asarray(b["tris"]), np.asarray(b["rim"]), int(b["apex"])
    assert np.max(np.abs(P[:, 2] - D * (1 - (P[:, 0] ** 2 + P[:, 1] ** 2) / R ** 2))) < 1e-9, "base not on surface"
    levels = [(P, T, rim)]
    for _ in range(N_LEVELS):
        levels.append(subdivide(*levels[-1]))
    base_deg, _ = valence(P, T)
    interior0 = np.setdiff1d(np.arange(len(P)), rim)
    defects0 = interior0[base_deg[interior0] != 6]
    rows = []
    for L, (P_, T_, rim_) in enumerate(levels):
        q = PO.quality(P_, T_, rim_)
        deg, E = valence(P_, T_)
        inn = np.setdiff1d(np.arange(len(P_)), rim_)
        dfx = inn[deg[inn] != 6]
        nested = np.array_equal(P_[:len(levels[L - 1][0])], levels[L - 1][0]) if L else True
        same_defects = np.array_equal(np.sort(dfx), np.sort(defects0))
        on_surf = np.max(np.abs(P_[:, 2] - D * (1 - (P_[:, 0] ** 2 + P_[:, 1] ** 2) / R ** 2)))
        rim_r = np.abs(np.hypot(P_[rim_, 0], P_[rim_, 1]) - R).max()
        Lh = np.linalg.norm(P_[E[:, 0]] - P_[E[:, 1]], axis=1)
        rows.append(dict(L=L, n=len(P_), e=len(E), t=len(T_), rim=len(rim_), h=np.median(Lh) * MM,
                         q=q, dfx=len(dfx), dfrac=len(dfx) / len(inn)))
        print(f"[L{L}] {len(P_):6d} nodes {len(E):6d} edges {len(T_):6d} tris {len(rim_):4d} rim | median h "
              f"{np.median(Lh) * MM:6.1f} mm | " + PO.fmt(q) +
              f" | defects {len(dfx)} ({100 * len(dfx) / len(inn):.2f}% of interior), same nodes as base: "
              f"{same_defects} | nested {nested} | on surface {on_surf:.1e}, rim |r-R| {rim_r:.1e}", flush=True)
        np.savez(f"{OUT_DIR}/mesh_L{L}.npz", vertices=P_, tris=T_, rim=rim_, apex=apex, R=R, DEPTH=D,
                 h_nominal=np.median(Lh), h_eff=np.median(Lh), n_target=len(P_), odt_iters=0, odt_omega=0.0,
                 odt_move_hist=np.zeros(1), source=f"{BASE} subdivided 1->4 x{L} (build_paraboloid_nested_meshes.py)",
                 level=L, n_parent_nodes=len(levels[L - 1][0]) if L else 0)

    # ---- scheme figure ----------------------------------------------------------------------------------------
    C_5, C_7 = "#2456c4", "#d97706"
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7.5,
                         "mathtext.fontset": "stixsans", "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(190 / 25.4, 4.3))
    S = 2500.0 / R / 1000.0

    # (a) the 1 -> 4 rule on one interior and one rim triangle, in the conformal plane
    ax = fig.add_axes([0.01, 0.55, 0.23, 0.36])
    A_, B_, C_ = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.45, 0.8])
    for tri, off, lab in (((A_, B_, C_), np.array([0, 0]), "interior"),):
        a_, b_, c_ = (v + off for v in tri)
        ab, bc, ca = (a_ + b_) / 2, (b_ + c_) / 2, (c_ + a_) / 2
        ax.fill(*np.array([a_, b_, c_]).T, color="#f2f2f2", lw=0)
        ax.plot(*np.array([a_, b_, c_, a_]).T, "-", color="0.1", lw=1.0)
        ax.plot(*np.array([ab, bc, ca, ab]).T, "--", color=C_7, lw=0.9)
        ax.plot(*np.array([a_, b_, c_]).T, "o", ms=4, color="0.1")
        ax.plot(*np.array([ab, bc, ca]).T, "o", ms=4, mfc="white", mec=C_7, mew=1.0)
        for p_, t_ in ((a_, "a"), (b_, "b"), (c_, "c")):
            ax.annotate(t_, p_, xytext=(-7 if t_ == "a" else 4, -8 if t_ != "c" else 3), textcoords="offset points")
    # rim triangle: two rim nodes on an arc, midpoint on the arc
    th = np.linspace(np.radians(-25), np.radians(25), 50); Rr = 1.0; cx = 2.45
    arc = np.column_stack([cx + Rr * np.cos(th) - 1.0, 0.4 + Rr * np.sin(th)])
    p1, p2 = arc[0], arc[-1]; p3 = np.array([cx - 0.75, 0.4])
    pm = np.array([cx + Rr - 1.0, 0.4]); m13, m23 = (p1 + p3) / 2, (p2 + p3) / 2
    ax.plot(*arc.T, "-", color="0.4", lw=0.8)
    ax.fill(*np.array([p1, pm, p2, p3]).T, color="#f2f2f2", lw=0)
    ax.plot(*np.array([p1, p2, p3, p1]).T, ":", color="0.5", lw=0.7)
    ax.plot(*np.array([p1, pm, p2]).T, "-", color="0.1", lw=1.0)
    ax.plot(*np.array([p2, p3, p1]).T, "-", color="0.1", lw=1.0)
    ax.plot(*np.array([pm, m23, m13, pm]).T, "--", color=C_7, lw=0.9)
    ax.plot(*np.array([p1, p2, p3]).T, "o", ms=4, color="0.1")
    ax.plot(*np.array([pm, m13, m23]).T, "o", ms=4, mfc="white", mec=C_7, mew=1.0)
    ax.annotate("rim midpoint\non the circle", pm, xytext=(-26, -34), textcoords="offset points", fontsize=6.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) 1 → 4 rule, conformal plane", fontsize=7.5, loc="left")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", color="0.1", ms=4, label="kept node"),
                       Line2D([], [], ls="none", marker="o", mfc="white", mec=C_7, ms=4, label="new midpoint"),
                       Line2D([], [], ls="--", color=C_7, lw=0.9, label="new edges")],
              loc="upper left", fontsize=6.3, frameon=False, ncol=1, handletextpad=0.3, bbox_to_anchor=(0.0, 0.0))

    # (b-e) the four levels, plan view, irregular nodes marked; inset box shows the zoom of row 2
    zoom = (0.55 * R, 0.95 * R, -0.2 * R, 0.2 * R)
    for L, (P_, T_, rim_) in enumerate(levels):
        ax = fig.add_axes([0.25 + 0.187 * L, 0.52, 0.177, 0.40])
        _, E = valence(P_, T_)
        ax.add_collection(LineCollection(np.stack([P_[E[:, 0], :2], P_[E[:, 1], :2]], 1) * S, colors="0.25",
                                         linewidths=0.35 if L < 2 else (0.15 if L == 2 else 0.06)))
        d = base_deg[defects0]
        ax.plot(P[defects0[d < 6], 0] * S, P[defects0[d < 6], 1] * S, "o", ms=2.6, color=C_5, zorder=3)
        ax.plot(P[defects0[d > 6], 0] * S, P[defects0[d > 6], 1] * S, "s", ms=2.4, color=C_7, zorder=3)
        x0, x1, y0, y1 = (v * S for v in zoom)
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "-", color="0.0", lw=0.6)
        ax.set_aspect("equal"); ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.axis("off")
        r_ = rows[L]
        ax.set_title(f"({'bcde'[L]}) L{L}\n{r_['n']} nodes, h = {r_['h']:.0f} mm", fontsize=7.0, loc="left")
        # zoom row
        az = fig.add_axes([0.25 + 0.187 * L, 0.10, 0.177, 0.33])
        az.add_collection(LineCollection(np.stack([P_[E[:, 0], :2], P_[E[:, 1], :2]], 1) * S, colors="0.25",
                                         linewidths=[0.8, 0.6, 0.4, 0.25][L]))
        az.plot(P[defects0[d < 6], 0] * S, P[defects0[d < 6], 1] * S, "o", ms=4, color=C_5, zorder=3)
        az.plot(P[defects0[d > 6], 0] * S, P[defects0[d > 6], 1] * S, "s", ms=3.6, color=C_7, zorder=3)
        if L:
            Pp = levels[L - 1][0]
            az.plot(Pp[:, 0] * S, Pp[:, 1] * S, ".", ms=1.6 if L < 3 else 0.8, color="0.0", zorder=2)
        az.set_xlim(x0, x1); az.set_ylim(y0, y1); az.set_aspect("equal")
        az.set_xticks([]); az.set_yticks([])
        az.set_title(f"irregular: {r_['dfx']} = {100 * r_['dfrac']:.2f}% of interior", fontsize=6.3, loc="left")
    # (f) quality vs level
    ax = fig.add_axes([0.065, 0.10, 0.15, 0.27])
    for L, (P_, T_, rim_) in enumerate(levels):
        amin, _ = min_angles(P_, T_)
        ax.hist(amin, bins=np.arange(30, 61, 1.0), histtype="step", density=True, lw=1.0,
                color=plt.cm.viridis(L / 3.2), label=f"L{L}")
    ax.set_xlabel("smallest angle per triangle (deg)"); ax.set_ylabel("density")
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    ax.set_title("(f) element shape per level", fontsize=7.5, loc="left")
    fig.text(0.25, 0.015, "Blue circles: valence-5, orange squares: valence-7 nodes of the base -- the only irregular "
             "nodes on every level.\nBlack dots (zooms): nodes of the previous level, all kept.", fontsize=6.3,
             color="0.25")
    for ext, kw in (("png", {"dpi": 400}), ("pdf", {})):
        fig.savefig(f"{OUT_DIR}/fig_nested_scheme.{ext}", **kw)
    print("saved", f"{OUT_DIR}/fig_nested_scheme.png")
