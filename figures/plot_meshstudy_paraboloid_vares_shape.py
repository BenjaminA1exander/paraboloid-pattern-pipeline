r"""plot_meshstudy_paraboloid_vares_shape.py -- what the gore mesh does to the converged VaReS
cutting pattern (companion to fig_meshstudy_paraboloid_vares from MeshStudy_paraboloid_vares.py).

Reads the converged rungs in temp/meshstudy_vares/hNNN/pattern.npz; nothing is re-solved.
Sampling and the rigid fit are MeshStudy_paraboloid_vares's own (p1_sample, rigid_fit), so the
numbers agree with its report (e_map is printed as a cross-check).

    (a) the seven gore meshes, drawn on their converged flat patterns
    (b) |phi_h - Q phi_ref| over the gore: how far each converged pattern sits from the
        finest converged rung after the best 2-D rigid fit Q
    (c) seam outline offset from the finest rung, outward normal, along the meridian

USAGE (base anaconda env)
    python plot_meshstudy_paraboloid_vares_shape.py
Outputs:
    fig_meshstudy_paraboloid_vares_shape.{pdf,png}
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import MeshStudy_paraboloid_vares_ckpt as CK  # noqa: E402
import MeshStudy_paraboloid_vares as MS          # noqa: E402

N_ST = 160                                       # stations for the deviation field
ETA = np.linspace(-0.98, 0.98, 21)               # fractions of the local half-width


def rung(h):
    d = dict(np.load(MS.rung_npz(h), allow_pickle=True))
    for k in ("geom_scale", "s_mid_mm", "s_lo_mm", "s_hi_mm"):
        d[k] = float(d[k])
    if not bool(d["converged"]):
        raise SystemExit("%s is not converged" % MS.tag(h))
    leaf = d["X_leaf"][:, :2]
    els = d["elements"]
    X = d["X_converged"][:, :2]
    x0 = d["s_lo_mm"] - d["s_mid_mm"]
    xs = x0 + np.linspace(*MS.STATION_FRAC, N_ST) * (d["s_hi_mm"] - d["s_lo_mm"])
    hw = MS.leaf_half_width(xs, d)
    grid = np.array([(x, e * w) for x, w in zip(xs, hw) for e in ETA])
    seam_p = np.c_[xs, hw]
    seam_m = np.c_[xs, -hw]
    pts = np.r_[grid, seam_p, seam_m]
    phi = MS.p1_sample(leaf, els, X, pts)
    # converged pattern placed on its own leaf (rigid only) for drawing the mesh
    ca, cb = X.mean(0), leaf.mean(0)
    U, _, Vt = np.linalg.svd((X - ca).T @ (leaf - cb))
    Rm = (U @ Vt).T
    X_draw = (X - ca) @ Rm.T + cb
    return dict(h=float(d["gmsh_mesh_size"]), n_nodes=len(X), n_tri=len(els), els=els,
                X_draw=X_draw, x0=x0, xs=xs, hw=hw, grid=grid, phi=phi,
                ng=len(grid), ns=len(xs))


def deviations(rows):
    ref = rows[-1]
    for r in rows:
        n = r["ng"]
        # rigid fit on the interior grid (as e_map), applied to the seams too
        A, B = r["phi"][:n], ref["phi"][:n]
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        Rm = (U @ Vt).T
        if np.linalg.det(Rm) < 0:
            Vt[-1] *= -1
            Rm = (U @ Vt).T
        P = (r["phi"] - ca) @ Rm.T + cb
        D = P - ref["phi"]
        r["dev"] = np.linalg.norm(D[:n], axis=1)
        r["e_map_grid"] = float(np.sqrt(np.mean(r["dev"] ** 2)))
        ns = r["ns"]
        for key, sl, sgn in (("off_p", slice(n, n + ns), 1.0),
                             ("off_m", slice(n + ns, n + 2 * ns), -1.0)):
            S = ref["phi"][sl]
            t = np.gradient(S, axis=0)
            t /= np.linalg.norm(t, axis=1)[:, None]
            nrm = np.c_[-t[:, 1], t[:, 0]]
            # outward = away from the centreline
            cen = ref["phi"][:n].reshape(ns, len(ETA), 2)[:, len(ETA) // 2]
            flip = np.sign(np.sum(nrm * (S - cen), axis=1))
            r[key] = np.sum(D[sl] * nrm, axis=1) * flip


def figure(rows, stem="fig_meshstudy_paraboloid_vares_shape"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import Normalize

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": MS.FS_LABEL, "axes.labelsize": MS.FS_LABEL,
        "xtick.labelsize": MS.FS_TICK, "ytick.labelsize": MS.FS_TICK,
        "legend.fontsize": MS.FS_LEG, "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.linewidth": 0.4, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    nr = len(rows)
    vir = plt.get_cmap("viridis")
    col = [vir(0.08 + 0.78 * i / max(nr - 1, 1)) for i in range(nr)]
    is_prod = [r["h"] == MS.PRODUCTION for r in rows]
    ref = rows[-1]

    W = MS.WIDTH_IN
    fig = plt.figure(figsize=(W, W * 1.12))
    gs = fig.add_gridspec(3, nr, height_ratios=[1.0, 1.0, 0.62], hspace=0.22, wspace=0.08)

    def to_plot(xy, x0):
        # rim at the bottom, apex at the top; metres
        return xy[:, 1] / 1000.0, (xy[:, 0] - x0) / 1000.0

    vmax = max(r["dev"].max() for r in rows[:-1])
    norm = Normalize(0.0, vmax)
    cdev = plt.get_cmap("cividis")
    xl = 1.05 * max(r["hw"].max() for r in rows) / 1000.0
    yl = (ref["xs"][-1] - ref["x0"]) / 1000.0

    axes_a, axes_b = [], []
    for i, r in enumerate(rows):
        a = fig.add_subplot(gs[0, i])
        u, v = to_plot(r["X_draw"], r["x0"])
        a.triplot(mtri.Triangulation(u, v, r["els"]), color="0.15",
                  lw=0.45 if r["h"] >= 70 else (0.3 if r["h"] >= 35 else 0.2))
        a.set_title("$h=%g$ mm%s\n%d tris"
                    % (round(r["h"], 1), "\n(prod.)" if is_prod[i] else "", r["n_tri"]),
                    fontsize=MS.FS_ANN, fontweight="bold" if is_prod[i] else "normal",
                    pad=3)
        axes_a.append(a)

        b = fig.add_subplot(gs[1, i])
        if i < nr - 1:
            u, v = to_plot(ref["grid"], ref["x0"])
            tri = mtri.Triangulation(u, v)
            b.tricontourf(tri, r["dev"], levels=np.linspace(0, vmax, 15), cmap=cdev,
                          norm=norm)
            b.set_title("rms %.2f mm\nmax %.2f mm" % (r["e_map_grid"], r["dev"].max()),
                        fontsize=MS.FS_ANN, pad=3,
                        fontweight="bold" if is_prod[i] else "normal")
        else:
            b.set_title("reference\n$h=%g$ mm" % round(r["h"], 1), fontsize=MS.FS_ANN, pad=3)
        axes_b.append(b)

    for row in (axes_a, axes_b):
        for j, a in enumerate(row):
            a.set_aspect("equal")
            a.set_xlim(-xl, xl)
            a.set_ylim(-0.03, yl + 0.03)
            a.set_xticks([-0.3, 0, 0.3])
            if j:
                a.tick_params(labelleft=False)
            else:
                a.set_ylabel("Meridian arc from rim, $s$ (m)")
            a.tick_params(labelbottom=False)
    for a in axes_b[:-1]:
        a.tick_params(labelbottom=True)
    axes_b[-1].axis("off")
    axes_b[nr // 2].set_xlabel("Across gore (m)")
    axes_a[0].text(0.0, 1.13, "(a)", transform=axes_a[0].transAxes, fontsize=MS.FS_PANEL,
                   va="bottom", ha="right")
    axes_b[0].text(0.0, 1.13, "(b)", transform=axes_b[0].transAxes, fontsize=MS.FS_PANEL,
                   va="bottom", ha="right")

    # colour bar inside the reference slot of row (b)
    pos = axes_b[-1].get_position()
    cax = fig.add_axes([pos.x0 + 0.62 * pos.width, pos.y0 + 0.08 * pos.height,
                        0.09 * pos.width, 0.70 * pos.height])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cdev)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$|\phi_h - Q\,\phi_{\rm ref}|$ (mm)", fontsize=MS.FS_ANN)
    cb.ax.tick_params(labelsize=MS.FS_TICK, width=0.5)
    cb.outline.set_linewidth(0.5)

    # (c) seam outline offset
    c = fig.add_subplot(gs[2, :])
    c.axhline(0.0, color="0.3", lw=0.6)
    for i, r in enumerate(rows[:-1]):
        lw = 1.6 if is_prod[i] else 0.9
        s = (r["xs"] - r["x0"]) / 1000.0
        c.plot(s, r["off_p"], "-", color=col[i], lw=lw, label="$h=%g$ mm" % round(r["h"], 1))
        c.plot(s, r["off_m"], "--", color=col[i], lw=lw)
    c.set_xlabel("Meridian arc from rim, $s$ (m)")
    c.set_ylabel("Seam offset from\n$h=%g$ mm (mm)" % round(ref["h"], 1))
    c.annotate("outward normal;  solid $+y$ seam,  dashed $-y$ seam", xy=(0.40, 0.95),
               xycoords="axes fraction", ha="center", va="top", fontsize=MS.FS_ANN,
               color="0.35")
    c.set_ylim(-0.27, 0.12)
    c.legend(loc="lower left", ncol=6, fontsize=MS.FS_LEG, handlelength=1.6,
             columnspacing=1.0)
    c.grid(alpha=0.22)
    c.text(0.0, 1.04, "(c)", transform=c.transAxes, fontsize=MS.FS_PANEL, va="bottom",
           ha="right")

    fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.05)
    # the colour bar was placed before subplots_adjust; re-place it
    pos = axes_b[-1].get_position()
    cax.set_position([pos.x0 + 0.25 * pos.width, pos.y0 + 0.05 * pos.height,
                      0.10 * pos.width, 0.80 * pos.height])
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        p = os.path.join(HERE, "%s%s.%s" % (stem, CK.SUFFIX, ext))
        fig.savefig(p, **kw)
        print("[figure] saved %s" % p)
    plt.close(fig)


if __name__ == "__main__":
    CK.configure()                     # $MESHSTUDY_TARGET: linearfit (default) or cst
    rows = [rung(h) for h in CK.ladder()]
    deviations(rows)
    print("mesh    nodes   tris   rms dev mm   max dev mm   max |seam off| mm")
    for r in rows[:-1]:
        print("h%03d  %6d %6d   %9.3f   %9.3f   %9.3f"
              % (round(r["h"]), r["n_nodes"], r["n_tri"], r["e_map_grid"], r["dev"].max(),
                 max(np.abs(r["off_p"]).max(), np.abs(r["off_m"]).max())))
    figure(rows)
