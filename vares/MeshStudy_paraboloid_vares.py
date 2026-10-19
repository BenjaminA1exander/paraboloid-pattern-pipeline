r"""MeshStudy_paraboloid_vares.py -- mesh convergence of the VaReS gore OPTIMISATION (stage 3 of
the paraboloid-dish pipeline, optimise_cutting_pattern_paraboloid_gore_chen.py, full Chen law).

QUESTION
    The published cutting pattern (cutting_pattern_paraboloid_chen.npz) was optimised on a gmsh
    leaf of nominal edge GMSH_SIZE = 100 mm (203 nodes, 323 tris).  How much of that pattern --
    its outline, its compensation strains, and the stress it induces -- is the membrane and how
    much is the mesh?

WHAT IS HELD FIXED
    The target sigma*(x): same form-finding npz (CFG.FORM_NPZ, the 48x24 production net), same
    linear closest-fit, same P_PHYS, t, N_GORES, TRIM_DISH.  The stage-2 stress field is a
    SEPARATE study (MeshStudy_paraboloid_stress.py) and is deliberately not varied here, so the
    two discretisations are not confounded.
    The solver: the stage-3 recipe verbatim (lsq_edge seed, geometric 3-dof pin, current-frame
    target, chen.run_inner_loop).  NEWTON_TOL_R = 1e-7 is NOT touched.
    NEWTON_MAX_IT is raised 100 -> MESHSTUDY_MAX_IT (default 300).  It is a budget, not a
    tolerance: the dof count grows with the mesh, and a rung that stops on the budget is still
    descending.  Every rung is classified by HOW it exited; only rungs with ||R|| < NEWTON_TOL_R
    are called converged, and the report says so loudly otherwise (the stall and step-floor
    exits of run_inner_loop leave ||R|| above tolerance and are NOT convergence).

    NOT uniform, and cannot be without changing the production mesher: build_gore_gmsh grades
    the narrow apex end to the local gore width, Max(lc_floor, Min(h, theta*(r_ref + d))), with
    lc_floor ~ 11 mm.  Near the apex the element size is set by the gore width, not by h, so
    that region is only refined once h drops below theta*(r_ref + d).  The ladder therefore
    refines the body of the gore; the tip is close to fixed.

HOW RUNGS ARE COMPARED (mesh-free, rigid-motion-free)
    Every rung is meshed in the same developed leaf (x = meridian arc, |y| <= h(x) = r(s) theta)
    and its P1 pattern map  phi_h : leaf -> flat plane  is sampled at the SAME leaf points on
    every rung.  The pin set is picked geometrically and so differs between meshes, i.e. each
    pattern carries a different rigid motion; the measures below remove it:
      e_map    rms |phi_h - Q phi_ref| over interior samples, Q the best 2-D rigid fit (mm)
      W(x)     flat seam-to-seam width |phi(x,+h) - phi(x,-h)| at each station (rigid-invariant)
      eps_w    W(x)/2h(x) - 1        hoop compensation (the cut is narrower than the leaf)
      eps_m    d|phi(x,0)|/dx - 1    meridional compensation along the gore centreline
      L        centreline length of the flat pattern
    plus the stress the pattern induces (area-weighted over the spatial gore):
      mis      rms of the weighted mismatch |sigma - sigma*|_w (MPa) -- converges to the
               CONTINUUM incompatibility of the target, not to zero
      sx, sy   area-weighted s-bin profiles of the induced warp / weft stress
    Errors are taken against the finest converged rung, and orders from successive rungs.

USAGE (base anaconda env)
    python MeshStudy_paraboloid_vares.py            # analyse whatever rungs exist
    python MeshStudy_paraboloid_vares.py --run      # run missing rungs first (parallel, slow)
    python MeshStudy_paraboloid_vares.py --rung 50  # worker: optimise one rung (used by --run)

Outputs:
    temp/meshstudy_vares/hNNN/pattern.npz + run.log     one optimisation per rung
    fig_meshstudy_paraboloid_vares.{pdf,png}
"""
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paraboloid_dish_config as CFG            # noqa: E402

ROOT = os.path.join(CFG.TEMP_DIR, "meshstudy_vares")
PY = sys.executable

# sqrt(2) ladder in nominal edge length -> ~2x elements per rung.  100 mm is production.
LADDER = [200.0, 141.4, 100.0, 70.7, 50.0, 35.4, 25.0]
PRODUCTION = 100.0
MAX_IT = int(os.environ.get("MESHSTUDY_MAX_IT", "300"))
MAX_JOBS = int(os.environ.get("MESHSTUDY_JOBS", "7"))

# leaf sampling: stations along the gore, fractions of the local half-width across it
N_STATIONS = 80
STATION_FRAC = (0.005, 0.995)
ETA_INTERIOR = np.linspace(-0.9, 0.9, 7)
N_SBINS = 28

WIDTH_IN = float(os.environ.get("FIG_W_MM", "190")) / 25.4
FS_LABEL, FS_TICK, FS_ANN, FS_LEG, FS_PANEL = 8.0, 7.0, 7.0, 7.0, 9.0


def tag(h):
    return "h%03d" % int(round(h))


def rung_dir(h):
    return os.path.join(ROOT, tag(h))


def rung_npz(h):
    return os.path.join(rung_dir(h), "pattern.npz")


# =============================================================================
# Worker: one optimisation
# =============================================================================

def run_one(h):
    import paraboloid_profile as pp
    sys.path.insert(0, os.path.join(HERE, "legacy"))     # the dense Newton loop of the earlier driver
    import optimise_cutting_pattern_npl_gore_chen as chen
    import vares_setup as stage3
    from vares_membrane import build_phys_targets_field
    from vares_flatten import least_squares_edge_flatten

    chen.NEWTON_MAX_IT = MAX_IT          # budget only; NEWTON_TOL_R untouched
    prof, D, meta = pp.extract_paraboloid()
    t0 = time.time()
    X_spatial, X_leaf, seam_ids, boundary_ids, elements = prof.build_gore_gmsh(
        h, CFG.N_GORES, gore_index=0)
    print(f"[rung {h}] {len(elements)} tris, {X_spatial.shape[0]} nodes; "
          f"tol {chen.NEWTON_TOL_R:g}, max_it {chen.NEWTON_MAX_IT}", flush=True)

    nodes_2d = least_squares_edge_flatten(
        X_spatial, elements, init_2d=X_leaf[:, :2], anchor_to=X_leaf[:, :2])
    X_initial = np.zeros_like(X_leaf)
    X_initial[:, :2] = nodes_2d
    pinned = stage3.build_pin_dof_set_geom(X_initial, boundary_ids)
    free_dofs = [a for a in range(2 * X_initial.shape[0]) if a not in pinned]
    sigma_star_all, detJ_ref_all = build_phys_targets_field(
        X_initial, X_spatial, elements, prof.sigma_field)

    X, R_hist, n_indef = chen.run_inner_loop(
        X_initial.copy(), X_spatial, elements, free_dofs, None, detJ_ref_all,
        f"mesh study h={h}", linear_material=False,
        sigma_star_all=sigma_star_all, verbose=True)
    wall = time.time() - t0

    res_init, S_init, T = stage3.phys_residual_field(X_initial, X_spatial, elements,
                                                     prof.sigma_field)
    res_conv, S_conv, _ = stage3.phys_residual_field(X, X_spatial, elements,
                                                     prof.sigma_field)
    els = np.asarray(elements)
    s_cen = np.array([prof.s_of_pos(p) for p in X_spatial[els].mean(axis=1)]) \
        * prof.geom_scale
    converged = bool(R_hist[-1] < chen.NEWTON_TOL_R)
    capped = (not converged) and len(R_hist) >= chen.NEWTON_MAX_IT
    np.savez(
        rung_npz(h),
        X_initial=X_initial, X_converged=X, X_spatial=X_spatial, X_leaf=X_leaf,
        elements=els, seam_ids=np.asarray(seam_ids), boundary_ids=np.asarray(boundary_ids),
        R_history=np.asarray(R_hist, float), n_indef=n_indef,
        res_init=res_init, res_conv=res_conv, S_init=S_init, S_conv=S_conv, T=T,
        s_cen=s_cen, gmsh_mesh_size=h, converged=converged, capped=capped,
        newton_tol=chen.NEWTON_TOL_R, newton_max_it=chen.NEWTON_MAX_IT, wall_s=wall,
        geom_scale=prof.geom_scale, s_mid_mm=0.5 * (prof.s_lo + prof.s_hi) * prof.geom_scale,
        s_lo_mm=prof.s_lo * prof.geom_scale, s_hi_mm=prof.s_hi * prof.geom_scale,
        n_gores=CFG.N_GORES, trim_dish=CFG.TRIM_DISH, p_fdm=meta["p_fdm"],
        form_npz=CFG.FORM_NPZ)
    print(f"[rung {h}] {len(R_hist)} its, ||R|| {R_hist[-1]:.3e} -> "
          f"{'CONVERGED' if converged else ('CAPPED' if capped else 'STOPPED ABOVE TOL')}"
          f"  ({wall:.0f} s)", flush=True)


def run_missing():
    os.makedirs(ROOT, exist_ok=True)
    todo = [h for h in LADDER if not os.path.exists(rung_npz(h))]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MPLBACKEND="Agg")
    procs = []
    # finest first so the long rungs start immediately
    for h in sorted(todo):
        while sum(p.poll() is None for _, p, _ in procs) >= MAX_JOBS:
            time.sleep(5)
        os.makedirs(rung_dir(h), exist_ok=True)
        log = open(os.path.join(rung_dir(h), "run.log"), "w")
        print(f"[run] {tag(h)} ...", flush=True)
        procs.append((h, subprocess.Popen([PY, "-u", os.path.abspath(__file__), "--rung",
                                           repr(h)], cwd=HERE, env=env, stdout=log,
                                          stderr=subprocess.STDOUT), log))
    for h, p, log in procs:
        rc = p.wait()
        log.close()
        print(f"[run] {tag(h)} exit {rc}", flush=True)


# =============================================================================
# Sampling one rung in the shared leaf
# =============================================================================

def leaf_half_width(x_mm, d):
    """h(x) exactly as build_gore_gmsh draws the seam: r(s) * theta."""
    prof = _prof()
    s = x_mm / d["geom_scale"] + d["s_mid_mm"] / d["geom_scale"]
    return prof.spl_r(s) * d["geom_scale"] * np.pi / CFG.N_GORES


_PROF = None


def _prof():
    global _PROF
    if _PROF is None:
        import paraboloid_profile as pp
        _PROF = pp.extract_paraboloid()[0]
    return _PROF


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


def one_rung(h):
    d = dict(np.load(rung_npz(h), allow_pickle=True))
    for k in ("geom_scale", "s_mid_mm", "s_lo_mm", "s_hi_mm"):
        d[k] = float(d[k])
    leaf = d["X_leaf"][:, :2]
    els = d["elements"]
    X = d["X_converged"][:, :2]
    Xs = d["X_spatial"]

    # shared stations: fixed fractions of the leaf length (identical on every rung)
    L_leaf = d["s_hi_mm"] - d["s_lo_mm"]
    x0 = d["s_lo_mm"] - d["s_mid_mm"]
    fr = np.linspace(*STATION_FRAC, N_STATIONS)
    xs = x0 + fr * L_leaf
    hw = leaf_half_width(xs, d)

    interior = np.array([(x, e * w) for x, w in zip(xs, hw) for e in ETA_INTERIOR])
    phi_int = p1_sample(leaf, els, X, interior)
    top = p1_sample(leaf, els, X, np.c_[xs, hw])
    bot = p1_sample(leaf, els, X, np.c_[xs, -hw])
    cl = p1_sample(leaf, els, X, np.c_[xs, np.zeros_like(xs)])
    W = np.linalg.norm(top - bot, axis=1)
    eps_w = W / (2 * hw) - 1.0
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    eps_m = seg / np.diff(xs) - 1.0
    L = float(seg.sum())

    # stress on the spatial gore, area weighted
    e = Xs[els]
    area = 0.5 * np.linalg.norm(np.cross(e[:, 1] - e[:, 0], e[:, 2] - e[:, 0]), axis=1)
    ef = np.c_[X, np.zeros(len(X))][els]
    area_flat = 0.5 * np.linalg.norm(np.cross(ef[:, 1] - ef[:, 0], ef[:, 2] - ef[:, 0]),
                                     axis=1)
    mis = float(np.sqrt(np.sum(area * d["res_conv"] ** 2) / area.sum()))
    mis_init = float(np.sqrt(np.sum(area * d["res_init"] ** 2) / area.sum()))
    edges = np.r_[els[:, [0, 1]], els[:, [1, 2]], els[:, [2, 0]]]
    h_mean = float(np.linalg.norm(Xs[edges[:, 0]] - Xs[edges[:, 1]], axis=1).mean())
    # element-count size: the edge of an equilateral tri with the mean area.  The mean edge
    # length is useless as an abscissa here -- it is dominated by the many small apex-graded
    # elements (nominal 200 and 141 mm both give ~95 mm).
    h_eff = float(np.sqrt(4.0 * area.sum() / (np.sqrt(3.0) * len(els))))

    sb = np.linspace(d["s_lo_mm"], d["s_hi_mm"], N_SBINS + 1)
    ib = np.clip(np.digitize(d["s_cen"], sb) - 1, 0, N_SBINS - 1)
    wsum = np.bincount(ib, area, N_SBINS)
    prof_sx = np.bincount(ib, area * d["S_conv"][:, 0], N_SBINS) / wsum
    prof_sy = np.bincount(ib, area * d["S_conv"][:, 1], N_SBINS) / wsum
    prof_tx = np.bincount(ib, area * d["T"][:, 0], N_SBINS) / wsum
    prof_ty = np.bincount(ib, area * d["T"][:, 1], N_SBINS) / wsum
    prof_mis = np.sqrt(np.bincount(ib, area * d["res_conv"] ** 2, N_SBINS) / wsum)

    R = d["R_history"]
    return dict(h=float(d["gmsh_mesh_size"]), h_mean=h_mean, h_eff=h_eff, n_nodes=len(X), n_tri=len(els),
                its=len(R), R_end=float(R[-1]), converged=bool(d["converged"]),
                capped=bool(d["capped"]), n_indef=int(d["n_indef"]), wall=float(d["wall_s"]),
                xs=xs, s_st=xs - x0, hw=hw, phi_int=phi_int, W=W, eps_w=eps_w, eps_m=eps_m,
                s_mid_seg=0.5 * (xs[1:] + xs[:-1]) - x0, L=L,
                A_ratio=float(area_flat.sum() / area.sum()),
                mis=mis, mis_init=mis_init, mis_max=float(d["res_conv"].max()),
                s_bin=0.5 * (sb[1:] + sb[:-1]), prof_sx=prof_sx, prof_sy=prof_sy,
                prof_tx=prof_tx, prof_ty=prof_ty, prof_mis=prof_mis,
                X=X, els=els, X0=d["X_initial"][:, :2])


def compare(rows):
    ref = rows[-1]
    for r in rows:
        r["e_map"] = float(np.sqrt(np.mean(np.sum(
            (rigid_fit(r["phi_int"], ref["phi_int"]) - ref["phi_int"]) ** 2, axis=1))))
        r["e_W"] = float(np.sqrt(np.mean((r["W"] - ref["W"]) ** 2)))
        r["e_W_max"] = float(np.abs(r["W"] - ref["W"]).max())
        r["e_epsw"] = float(np.sqrt(np.mean((r["eps_w"] - ref["eps_w"]) ** 2)))
        r["e_epsm"] = float(np.sqrt(np.mean((r["eps_m"] - ref["eps_m"]) ** 2)))
        r["e_L"] = abs(r["L"] - ref["L"])
        r["e_sig"] = float(np.sqrt(np.mean((r["prof_sx"] - ref["prof_sx"]) ** 2
                                           + (r["prof_sy"] - ref["prof_sy"]) ** 2)))
        r["e_mis"] = abs(r["mis"] - ref["mis"])
    for i, r in enumerate(rows):
        r["d_map"] = np.nan
        if i + 1 < len(rows):
            q = rows[i + 1]
            r["d_map"] = float(np.sqrt(np.mean(np.sum(
                (rigid_fit(r["phi_int"], q["phi_int"]) - q["phi_int"]) ** 2, axis=1))))


def order(e1, e2, h1, h2):
    if not (np.isfinite(e1) and np.isfinite(e2)) or e1 <= 0 or e2 <= 0:
        return np.nan
    return float(np.log(e1 / e2) / np.log(h1 / h2))


def fit_order(rows, key):
    h = np.array([r["h_eff"] for r in rows])
    e = np.array([r[key] for r in rows])
    m = np.isfinite(e) & (e > 0)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(h[m]), np.log(e[m]), 1)[0])


# =============================================================================
# Report
# =============================================================================

def production_check():
    """The 100 mm rung is the production recipe; it must reproduce the published npz."""
    if not (os.path.exists(rung_npz(PRODUCTION)) and os.path.exists(CFG.PAT_NPZ)):
        return
    a = np.load(rung_npz(PRODUCTION))
    b = np.load(CFG.PAT_NPZ)
    if a["X_converged"].shape != b["X_converged"].shape:
        print("  !! production rung mesh differs from the published pattern "
              f"({a['X_converged'].shape} vs {b['X_converged'].shape})")
        return
    dx = np.abs(a["X_converged"] - b["X_converged"]).max()
    print(f"  production check: {tag(PRODUCTION)} vs {os.path.basename(CFG.PAT_NPZ)}  "
          f"max |dX| = {dx:.2e} mm  ({'reproduced' if dx < 1e-6 else 'DIFFERS'})")


def report(rows):
    def t(r):
        return tag(r["h"]) + (" *" if r["h"] == PRODUCTION else "")

    print("\n" + "=" * 100)
    print("RUNGS  (* = production mesh of the published pattern)  "
          f"converged means ||R|| < {1e-7:g}; the tolerance is never relaxed")
    print("%-7s %7s %7s %8s %8s %6s %11s %6s %8s  %s"
          % ("mesh", "nodes", "tris", "h_eff", "h_mean", "its", "||R||_end", "indef", "wall s", "exit"))
    for r in rows:
        ex = "converged" if r["converged"] else ("CAPPED" if r["capped"]
                                                  else "STOPPED ABOVE TOL")
        print("%-7s %7d %7d %8.1f %8.1f %6d %11.2e %6d %8.0f  %s"
              % (t(r), r["n_nodes"], r["n_tri"], r["h_eff"], r["h_mean"], r["its"], r["R_end"],
                 r["n_indef"], r["wall"], ex))
    production_check()
    bad = [r for r in rows if not r["converged"]]
    if bad:
        print("  !! %s did not reach tolerance: their patterns are not what those meshes give."
              % ", ".join(tag(r["h"]) for r in bad))

    print("\n" + "=" * 100)
    print("PATTERN vs finest rung (%s)   orders vs h_eff = sqrt(4A/(sqrt3 N_e))" % tag(rows[-1]["h"]))
    print("%-7s %9s %6s %10s %6s %10s %11s %11s %9s" %
          ("mesh", "e_map mm", "order", "e_W mm", "order", "maxW mm", "eps_w rms", "eps_m rms",
           "|dL| mm"))
    for i, r in enumerate(rows):
        om = ow = np.nan
        if i and i < len(rows) - 1:
            q = rows[i - 1]
            om = order(q["e_map"], r["e_map"], q["h_eff"], r["h_eff"])
            ow = order(q["e_W"], r["e_W"], q["h_eff"], r["h_eff"])
        f = lambda v: "--" if np.isnan(v) else "%.2f" % v   # noqa: E731
        print("%-7s %9.3f %6s %10.3f %6s %10.3f %11.2e %11.2e %9.3f"
              % (t(r), r["e_map"], f(om), r["e_W"], f(ow), r["e_W_max"], r["e_epsw"],
                 r["e_epsm"], r["e_L"]))
    sub = rows[:-1]
    if len(sub) >= 2:
        print("%-7s %9.2f %6s %10.2f   (least-squares slope, reference rung excluded)"
              % ("slope", fit_order(sub, "e_map"), "", fit_order(sub, "e_W")))
    print("  successive-rung map change (no reference needed):")
    for i in range(len(rows) - 1):
        r, q = rows[i], rows[i + 1]
        o = order(rows[i - 1]["d_map"], r["d_map"], rows[i - 1]["h_eff"], r["h_eff"]) if i else np.nan
        print("    %s -> %s : %.3f mm%s" % (tag(r["h"]), tag(q["h"]), r["d_map"],
                                           "" if np.isnan(o) else "   order %.2f" % o))

    print("\n" + "=" * 100)
    print("COMPENSATION and STRESS")
    print("%-7s %10s %10s %10s %10s %11s %11s %10s %10s"
          % ("mesh", "L mm", "A_fl/A_sp", "eps_w mid", "eps_m mid", "mis MPa", "mis max",
             "e_mis", "e_sig MPa"))
    for r in rows:
        k = len(r["eps_w"]) // 2
        print("%-7s %10.2f %10.5f %10.5f %10.5f %11.4f %11.3f %10.4f %10.4f"
              % (t(r), r["L"], r["A_ratio"], r["eps_w"][k], r["eps_m"][k - 1], r["mis"],
                 r["mis_max"], r["e_mis"], r["e_sig"]))
    print("  mis = area-weighted rms weighted mismatch at convergence (seed: %.1f MPa); it tends"
          % rows[-1]["mis_init"])
    print("  to the target's continuum incompatibility, so e_mis, not mis, is the error.")

    if len(rows) >= 3:
        print("\n" + "=" * 100)
        print("RICHARDSON (last three rungs) -- errors against the finest rung understate the truth;")
        print("this estimates how far the finest rung itself is from the h -> 0 limit")
        h1, h2 = rows[-2]["h_eff"], rows[-1]["h_eff"]
        m = np.array([r["mis"] for r in rows[-3:]])
        d1, d2 = m[1] - m[0], m[2] - m[1]
        if d1 * d2 > 0:
            p = np.log(d1 / d2) / np.log(rows[-3]["h_eff"] / h1)
            lim = m[2] + d2 / ((h1 / h2) ** p - 1.0)
            print("  mismatch rms: observed order %.2f, limit %.3f MPa" % (p, lim))
            for r in rows:
                print("    %-7s %.3f MPa  (+%.0f%% over the limit)"
                      % (t(r), r["mis"], 100 * (r["mis"] / lim - 1.0)))
        dm1, dm2 = rows[-3]["d_map"], rows[-2]["d_map"]
        if dm1 > 0 and dm2 > 0:
            p = np.log(dm1 / dm2) / np.log(rows[-3]["h_eff"] / h1)
            own = dm2 / ((h1 / h2) ** p - 1.0)
            print("  pattern map: observed order %.2f; finest rung's own error ~%.3f mm, so the"
                  % (p, own))
            print("  production error is ~%.3f mm (e_map + that)"
                  % (next(r for r in rows if r["h"] == PRODUCTION)["e_map"] + own))


# =============================================================================
# Figure
# =============================================================================

def figure(rows, stem="fig_meshstudy_paraboloid_vares"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEG, "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.linewidth": 0.4, "legend.frameon": False,
        "legend.handletextpad": 0.6, "legend.borderpad": 0.15,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    cmap = plt.get_cmap("viridis")
    col = {i: cmap(0.08 + 0.78 * i / max(len(rows) - 1, 1)) for i in range(len(rows))}
    is_prod = [r["h"] == PRODUCTION for r in rows]
    lab = ["$h=%g$ mm" % round(r["h"]) for r in rows]

    fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * 0.60))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    def panel(ax, letter, x=0.015, y=1.04):
        ax.text(x, y, "(%s)" % letter, transform=ax.transAxes, fontsize=FS_PANEL,
                va="bottom")
        ax.grid(alpha=0.22)

    def style(i):
        return dict(color=col[i], lw=1.4 if is_prod[i] else 0.9,
                    alpha=1.0 if is_prod[i] else 0.75)

    # (a) hoop compensation (solid) and meridional compensation (dashed) along the gore
    for i, r in enumerate(rows):
        ax_a.plot(r["s_st"] / 1000.0, 100 * r["eps_w"], "-", label=lab[i], **style(i))
        ax_a.plot(r["s_mid_seg"] / 1000.0, 100 * r["eps_m"], "--", **style(i))
    ax_a.set_xlabel("Meridian arc from rim, $s$ (m)")
    ax_a.set_ylabel("Pattern compensation (%)")
    ax_a.annotate("solid hoop $W/2h-1$,  dashed meridional", xy=(0.03, 0.04),
                  xycoords="axes fraction", fontsize=FS_ANN, color="0.35")
    ax_a.set_ylim(top=max(ax_a.get_ylim()[1], 2.9))
    ax_a.legend(loc="upper left", ncol=4, fontsize=FS_LEG, columnspacing=0.8,
                handlelength=1.4)
    panel(ax_a, "a")

    # (b) width deviation from the finest rung
    ref = rows[-1]
    ax_b.axhline(0.0, color="0.3", lw=0.7)
    for i, r in enumerate(rows[:-1]):
        ax_b.plot(r["s_st"] / 1000.0, r["W"] - ref["W"], "-", **style(i))
    ax_b.set_xlabel("Meridian arc from rim, $s$ (m)")
    ax_b.set_ylabel("$W_h - W_{\\rm ref}$ (mm)")
    ax_b.annotate("ref. $h=%g$ mm" % round(ref["h"]), xy=(0.97, 0.96),
                  xycoords="axes fraction", ha="right", va="top", fontsize=FS_ANN,
                  color="0.35")
    panel(ax_b, "b")

    # (c) convergence
    sub = rows[:-1]
    h = np.array([r["h_eff"] for r in sub])
    for key, mk, c, l, sc in (("e_map", "o", "#c0261f", "pattern map (mm)", 1.0),
                              ("e_W", "s", "#d97706", "seam width (mm)", 1.0),
                              ("e_sig", "^", "#2456c4", "induced stress (MPa)", 1.0),
                              ("e_mis", "D", "#6b21a8", "mismatch rms (MPa)", 1.0)):
        e = np.array([r[key] for r in sub]) * sc
        m = e > 0
        ax_c.loglog(h[m], e[m], marker=mk, ls="-", color=c, lw=1.0, ms=3.2, mew=0, label=l)
    if len(h):
        # guides sit a factor 0.4 under the series they are compared with, so neither
        # the dotted line nor its label lands on the data
        hg = np.array([h.min(), h.max()])
        for p, key in ((1, "e_map"), (2, "e_W")):
            e0 = 0.4 * sub[-1][key]
            ax_c.loglog(hg, e0 * (hg / h[-1]) ** p, ls=":", color="0.45", lw=0.9)
            ax_c.annotate("slope %d" % p, xy=(hg[1], e0 * (hg[1] / h[-1]) ** p),
                          xytext=(0, -3), textcoords="offset points",
                          fontsize=FS_ANN, color="0.45", ha="right", va="top")
    ax_c.set_xlabel(r"Effective element size, $h_{\rm eff}$ (mm)")
    ax_c.set_ylabel("Difference from finest rung")
    ax_c.set_ylim(top=ax_c.get_ylim()[1] * 12)   # headroom for the legend
    ax_c.legend(loc="upper left", fontsize=FS_LEG)
    panel(ax_c, "c")
    ax_c.grid(alpha=0.22, which="both")

    # (d) induced vs target stress
    ax_d.plot(ref["s_bin"] / 1000.0, ref["prof_tx"], "-", color="0.15", lw=1.2, zorder=5,
              label="target $\\sigma^\\star$")
    ax_d.plot(ref["s_bin"] / 1000.0, ref["prof_ty"], "--", color="0.15", lw=1.2, zorder=5)
    for i, r in enumerate(rows):
        ax_d.plot(r["s_bin"] / 1000.0, r["prof_sx"], "-", **style(i))
        ax_d.plot(r["s_bin"] / 1000.0, r["prof_sy"], "--", **style(i))
    ax_d.set_xlabel("Meridian arc from rim, $s$ (m)")
    ax_d.set_ylabel("Induced stress (MPa)")
    ax_d.annotate("solid warp,  dashed weft", xy=(0.03, 0.04), xycoords="axes fraction",
                  fontsize=FS_ANN, color="0.35")
    ax_d.legend(loc="upper right", fontsize=FS_LEG)
    panel(ax_d, "d")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.945, bottom=0.095)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        p = os.path.join(HERE, "%s.%s" % (stem, ext))
        fig.savefig(p, **kw)
        print("[figure] saved %s" % p)
    plt.close(fig)


# =============================================================================

if __name__ == "__main__":
    if "--rung" in sys.argv:
        run_one(float(sys.argv[sys.argv.index("--rung") + 1]))
        sys.exit(0)
    if "--run" in sys.argv:
        run_missing()
    rows = []
    for h in LADDER:
        if not os.path.exists(rung_npz(h)):
            print("[skip] %s not run yet (--run)" % tag(h))
            continue
        rows.append(one_rung(h))
    if len(rows) < 2:
        raise SystemExit("need at least two rungs")
    # the reference must be a converged rung
    while rows and not rows[-1]["converged"]:
        print("[ref] %s not converged -- not usable as reference" % tag(rows[-1]["h"]))
        rows = rows[:-1]
    compare(rows)
    report(rows)
    figure(rows)
