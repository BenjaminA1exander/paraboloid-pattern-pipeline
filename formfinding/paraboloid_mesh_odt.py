r"""paraboloid_mesh_odt.py -- make an unstructured paraboloid mesh as uniform as possible (equal-size,
near-equilateral triangles) by Optimal Delaunay Triangulation (ODT) smoothing on the surface.

Input: a paraboloid_unstructured_mesh.py npz (gmsh frontal-Delaunay in the conformal disc).
Each iteration:
  1. ODT node update (Chen & Xu 2004; Alliez et al. 2005), in 3-D: every free node moves to the
     area-weighted mean of the circumcentres of the triangles in its star,
         x_i <- sum_T |T| c_T / sum_T |T|,
     which minimises the interpolation error of |x|^2 -- the update that drives a Delaunay mesh to
     equal-size, near-equilateral cells.  (Under-relaxed by OMEGA; a circumcentre that falls outside
     its own triangle is replaced by the centroid, which keeps the update bounded near the rim.)
  2. Closest-point projection back onto z = D(1 - r^2/R^2).
  3. Re-triangulation: Delaunay of the nodes in the paraboloid's CONFORMAL plane (angle-preserving,
     so Delaunay there is Delaunay on the surface to first order).  This is where the edge flips
     that remove valence defects happen.
Held fixed: the rim nodes (re-spaced to exactly equal arcs first) and the apex node at r = 0, which
the form-finder's apex term needs.  The node count is unchanged.

A perfect valence-6 equilateral mesh does not exist here: the cap carries int K dA = 2 pi (1 - cos 45)
= 1.84 rad ~ 1.8 five-fold disclinations, so some defects / distortion must remain.

Run (base env):
    python paraboloid_mesh_odt.py --in mesh.npz --out mesh_odt.npz [--iters 300]
"""
import argparse
import os

import numpy as np
from scipy.spatial import Delaunay

import paraboloid_unstructured_mesh as PM

R, D = PM.R_FORM, PM.DEPTH_FORM
K = 2.0 * D / R ** 2
_r_t, _s_t, _rho_t, _lam_t = PM.meridian_tables()
APEX_R0 = float(os.environ.get("APEX_R0", "0.01"))   # 0.01 = apex node only (option B, 2026-10-03)


def project(P):
    """Closest point on the paraboloid (Newton on the meridian parameter t in the point's plane)."""
    r = np.hypot(P[:, 0], P[:, 1])
    z = P[:, 2]
    t = r.copy()
    for _ in range(30):
        zt = D * (1 - t ** 2 / R ** 2)
        dz = -K * t
        f = (t - r) + (zt - z) * dz             # d/dt of half squared distance
        fp = 1.0 + dz * dz + (zt - z) * (-K)
        t = np.clip(t - f / fp, 0.0, R)
    sc = np.where(r > 1e-14, t / np.where(r > 1e-14, r, 1.0), 0.0)
    return np.column_stack([P[:, 0] * sc, P[:, 1] * sc, D * (1 - t ** 2 / R ** 2)])


def conformal_xy(P):
    r = np.hypot(P[:, 0], P[:, 1])
    rho = np.interp(r, _r_t, _rho_t)
    sc = np.where(r > 1e-14, rho / np.where(r > 1e-14, r, 1.0), 1.0)
    return P[:, :2] * sc[:, None]


def triangulate(P):
    T = Delaunay(conformal_xy(P)).simplices.astype(int)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    flip = np.cross(b - a, c - a)[:, 2] < 0
    T[flip] = T[flip][:, ::-1]
    return T


def circumcentres(P, T):
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    n2 = np.sum(n * n, 1)
    cc = a + (np.cross(n, ab) * np.sum(ac * ac, 1)[:, None]
              + np.cross(ac, n) * np.sum(ab * ab, 1)[:, None]) / (2.0 * n2[:, None])
    # barycentric test: circumcentre outside its triangle (obtuse) -> use the centroid
    cen = (a + b + c) / 3.0

    def side(p, q, x):
        return np.sum(np.cross(q - p, x - p) * n, 1)
    inside = (side(a, b, cc) >= 0) & (side(b, c, cc) >= 0) & (side(c, a, cc) >= 0)
    return np.where(inside[:, None], cc, cen), 0.5 * np.sqrt(n2)


def quality(P, T, fixed):
    E = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    la, lb, lc = (np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1),
                  np.linalg.norm(a - b, axis=1))
    SR = (la ** 2 + lb ** 2 + lc ** 2) / (4 * np.sqrt(3) * A)
    ang = []
    for p_, q_, r_ in ((a, b, c), (b, c, a), (c, a, b)):
        u, v = q_ - p_, r_ - p_
        ang.append(np.degrees(np.arccos(np.clip(np.sum(u * v, 1) / np.linalg.norm(u, axis=1)
                                                / np.linalg.norm(v, axis=1), -1, 1))))
    ang = np.array(ang)
    deg = np.bincount(E.ravel(), minlength=len(P))
    interior = np.setdiff1d(np.arange(len(P)), fixed)
    vals, cnt = np.unique(deg[interior], return_counts=True)
    all_fixed = int(np.isin(T, fixed).all(1).sum())
    return dict(edge_cv=L.std() / L.mean(), area_cv=A.std() / A.mean(), sr_mean=SR.mean(),
                sr_max=SR.max(), ang_min=ang.min(), ang_max=ang.max(),
                ang_5070=np.mean((ang > 50) & (ang < 70)), valence=dict(zip(vals.tolist(), cnt.tolist())),
                defects=int((deg[interior] != 6).sum()), all_rim_tris=all_fixed, n_tri=len(T),
                h_mean=L.mean())


def fmt(q):
    return (f"edge CV {100 * q['edge_cv']:.2f}%  area CV {100 * q['area_cv']:.2f}%  SR mean "
            f"{q['sr_mean']:.4f} max {q['sr_max']:.3f}  angles {q['ang_min']:.1f}-{q['ang_max']:.1f} "
            f"({100 * q['ang_5070']:.1f}% in 50-70)  defects {q['defects']} {q['valence']}  "
            f"all-rim tris {q['all_rim_tris']}")


def valence_flips(P, T, fixed, min_angle=25.0, max_sweeps=50, rim_weight=1.0, apex_weight=1.0):
    """Valence-improving edge flips (Botsch & Kobbelt 2004): flip interior edge (a,b) with opposite
    nodes (c,d) when it lowers sum (deg - target)^2 over {a,b,c,d} (target 6 inside, 4 on the rim)
    and both new triangles are positively oriented in the conformal plane with every angle above
    min_angle (3-D).  Returns the new triangle list and the number of flips."""
    T = [list(t) for t in T]
    is_fixed_rim = np.zeros(len(P), bool)
    is_fixed_rim[fixed[:-1]] = True                     # rim (the apex is interior: target 6)
    tgt = np.where(is_fixed_rim, 4, 6)
    wv = np.where(is_fixed_rim, rim_weight, 1.0)   # rim_weight 0: rim valence free (absorbs misfit)
    # apex_weight >> 1: insist on a defect-free CENTRE -- the weight decays from the apex over
    # r0 = APEX_R0 (~2.5 edges), so the defects are pushed out of the central patch, not one ring out
    _r = np.hypot(P[:, 0], P[:, 1])
    wv = np.where(is_fixed_rim, wv, 1.0 + (apex_weight - 1.0) * np.exp(-(_r / APEX_R0) ** 2))
    Q = conformal_xy(P)

    def orient(i, j, k):
        return (Q[j, 0] - Q[i, 0]) * (Q[k, 1] - Q[i, 1]) - (Q[k, 0] - Q[i, 0]) * (Q[j, 1] - Q[i, 1])

    def min_ang(i, j, k):
        out = 180.0
        for p_, q_, r_ in ((i, j, k), (j, k, i), (k, i, j)):
            u, v = P[q_] - P[p_], P[r_] - P[p_]
            out = min(out, np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
        return out

    n_flip = 0
    for _ in range(max_sweeps):
        deg = np.zeros(len(P), int)
        emap = {}
        for ti, t in enumerate(T):
            for k in range(3):
                a, b = t[k], t[(k + 1) % 3]
                emap.setdefault((min(a, b), max(a, b)), []).append(ti)
        for (a, b) in emap:
            deg[a] += 1
            deg[b] += 1
        flipped = 0
        done = set()
        for (a, b), tl in list(emap.items()):
            if len(tl) != 2 or tl[0] in done or tl[1] in done:
                continue
            t0, t1 = T[tl[0]], T[tl[1]]
            c = [v for v in t0 if v not in (a, b)][0]
            d = [v for v in t1 if v not in (a, b)][0]
            before = sum(wv[v] * (deg[v] - tgt[v]) ** 2 for v in (a, b, c, d))
            after = (wv[a] * (deg[a] - 1 - tgt[a]) ** 2 + wv[b] * (deg[b] - 1 - tgt[b]) ** 2
                     + wv[c] * (deg[c] + 1 - tgt[c]) ** 2 + wv[d] * (deg[d] + 1 - tgt[d]) ** 2)
            if after >= before:
                continue
            # new triangles (c, d, a) and (d, c, b), oriented counter-clockwise in the plane
            n0 = [c, d, a] if orient(c, d, a) > 0 else [d, c, a]
            n1 = [d, c, b] if orient(d, c, b) > 0 else [c, d, b]
            if orient(*n0) <= 0 or orient(*n1) <= 0:
                continue
            # convexity: a and b on opposite sides of cd
            if np.sign(orient(c, d, a)) == np.sign(orient(c, d, b)):
                continue
            if min(min_ang(*n0), min_ang(*n1)) < min_angle:
                continue
            T[tl[0]], T[tl[1]] = n0, n1
            done.update(tl)
            for v, dv in ((a, -1), (b, -1), (c, 1), (d, 1)):
                deg[v] += dv
            flipped += 1
        n_flip += flipped
        if flipped == 0:
            break
    return np.asarray(T, int), n_flip


def relax(P, T, free, apex, omega, iters):
    """ODT-style tangential relaxation with the connectivity HELD (no re-triangulation)."""
    for _ in range(iters):
        cc, A = circumcentres(P, T)
        num = np.zeros_like(P)
        den = np.zeros(len(P))
        for k in range(3):
            np.add.at(num, T[:, k], A[:, None] * cc)
            np.add.at(den, T[:, k], A)
        P[free] = P[free] + omega * (num[free] / den[free, None] - P[free])
        P = project(P)
        P[apex] = [0.0, 0.0, D]
    return P


def maxmin_angle_polish(P, T, free_mask, passes=15, ang_thresh=48.0, n_grid=9):
    """Local polish of the worst elements, connectivity held: every free node in a triangle with an
    angle below ang_thresh (or above 180 - 2*ang_thresh) tries positions on an n_grid x n_grid grid
    in its tangent plane (span 0.3 x local edge length) and keeps the one that maximises the
    smallest angle of its star (re-projected onto the surface)."""
    star = [[] for _ in range(len(P))]
    for ti, t in enumerate(T):
        for v in t:
            star[v].append(ti)

    def star_min_angle(v, pv):
        Pl = P.copy()
        Pl[v] = pv
        out = 180.0
        for ti in star[v]:
            i, j, k = T[ti]
            for p_, q_, r_ in ((i, j, k), (j, k, i), (k, i, j)):
                u, w = Pl[q_] - Pl[p_], Pl[r_] - Pl[p_]
                out = min(out, np.degrees(np.arccos(np.clip(
                    u @ w / np.linalg.norm(u) / np.linalg.norm(w), -1, 1))))
        return out

    for _ in range(passes):
        a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        angs = []
        for p_, q_, r_ in ((a, b, c), (b, c, a), (c, a, b)):
            u, w = q_ - p_, r_ - p_
            angs.append(np.degrees(np.arccos(np.clip(np.sum(u * w, 1) / np.linalg.norm(u, axis=1)
                                                     / np.linalg.norm(w, axis=1), -1, 1))))
        angs = np.array(angs)
        bad = (angs.min(0) < ang_thresh) | (angs.max(0) > 180.0 - 2.0 * ang_thresh)
        nodes = [v for v in np.unique(T[bad]) if free_mask[v]]
        moved = 0
        for v in nodes:
            nb = np.unique(T[star[v]])
            h = np.mean(np.linalg.norm(P[nb] - P[v], axis=1)) * 1.5
            n = np.array([K * P[v, 0], K * P[v, 1], 1.0])
            n /= np.linalg.norm(n)
            e1 = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            best_p, best_a = P[v].copy(), star_min_angle(v, P[v])
            for s1 in np.linspace(-0.15, 0.15, n_grid):
                for s2 in np.linspace(-0.15, 0.15, n_grid):
                    cand = project((P[v] + h * (s1 * e1 + s2 * e2))[None])[0]
                    a_ = star_min_angle(v, cand)
                    if a_ > best_a + 1e-9:
                        best_p, best_a = cand, a_
            if not np.allclose(best_p, P[v]):
                P[v] = best_p
                moved += 1
        if moved == 0:
            break
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--omega", type=float, default=0.8)
    ap.add_argument("--rounds", type=int, default=60, help="flip + relax rounds (stage 2)")
    ap.add_argument("--relax-iters", type=int, default=20)
    ap.add_argument("--flip-min-angle", type=float, default=25.0)
    ap.add_argument("--rim-weight", type=float, default=1.0, help="0 = rim valence free")
    ap.add_argument("--polish", type=int, default=1, help="max-min-angle polish of the worst elements")
    ap.add_argument("--polish-angle", type=float, default=42.0)
    ap.add_argument("--apex-weight", type=float, default=20.0,
                    help="flip-energy weight of the apex valence (keeps the centre defect-free)")
    args = ap.parse_args()

    m = np.load(args.inp)
    P = np.asarray(m["vertices"], float).copy()
    rim = np.asarray(m["rim"], int)
    apex = int(m["apex"])
    # re-space the rim to exactly equal arcs, keeping its angular order and phase
    th = np.arctan2(P[rim, 1], P[rim, 0])
    o = np.argsort(th)
    th_eq = th[o][0] + 2 * np.pi * np.arange(len(rim)) / len(rim)
    P[rim[o]] = np.column_stack([R * np.cos(th_eq), R * np.sin(th_eq), np.zeros(len(rim))])
    fixed = np.concatenate([rim, [apex]])
    free = np.setdiff1d(np.arange(len(P)), fixed)

    T = triangulate(P)
    q0 = quality(np.asarray(m["vertices"], float), np.asarray(m["tris"], int), fixed)
    print("[gmsh seed] " + fmt(q0))
    hist = []
    for it in range(1, args.iters + 1):
        cc, A = circumcentres(P, T)
        num = np.zeros_like(P)
        den = np.zeros(len(P))
        for k in range(3):
            np.add.at(num, T[:, k], A[:, None] * cc)
            np.add.at(den, T[:, k], A)
        target = num[free] / den[free, None]
        move = args.omega * (target - P[free])
        P[free] = P[free] + move
        P = project(P)
        P[apex] = [0.0, 0.0, D]
        T = triangulate(P)
        hist.append(float(np.abs(move).max()))
        if it % 50 == 0 or it == 1:
            print(f"[odt {it:4d}] max move {hist[-1]:.2e}  " + fmt(quality(P, T, fixed)), flush=True)
    print("[odt final] " + fmt(quality(P, T, fixed)))
    # ---- stage 2: valence flips + held-connectivity relaxation (5-7 pairs glide and annihilate) ----
    for rnd in range(1, args.rounds + 1):
        T, nf = valence_flips(P, T, fixed, min_angle=args.flip_min_angle, rim_weight=args.rim_weight,
                              apex_weight=args.apex_weight)
        P = relax(P, T, free, apex, args.omega, args.relax_iters)
        if rnd % 5 == 0 or rnd == 1 or nf == 0:
            print(f"[remesh {rnd:3d}] flips {nf:4d}  " + fmt(quality(P, T, fixed)), flush=True)
        if nf == 0:
            break
    P = relax(P, T, free, apex, args.omega, 200)              # final relaxation, connectivity frozen
    print("[relaxed] " + fmt(quality(P, T, fixed)))
    if args.polish:
        fm = np.ones(len(P), bool)
        fm[fixed] = False
        P = maxmin_angle_polish(P, T, fm, ang_thresh=args.polish_angle)
    q1 = quality(P, T, fixed)
    print("[final] " + fmt(q1))
    E = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, vertices=P, tris=T, rim=rim, apex=apex, R=R, DEPTH=D,
             h_nominal=float(m["h_nominal"]), h_eff=float(np.sqrt(4 * 0 + 1) * 0 + q1["h_mean"]),
             n_target=int(m["n_target"]), odt_iters=args.iters, odt_omega=args.omega,
             odt_move_hist=np.asarray(hist), source=args.inp)
    print(f"[mesh] saved {args.out}: {len(P)} nodes, {len(E)} edges, {len(T)} tris")


if __name__ == "__main__":
    main()
