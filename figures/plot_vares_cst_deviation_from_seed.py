r"""plot_vares_cst_deviation_from_seed.py -- how far VaReS moves the flat gore away from its seed, on the
finest converged CST rung (h = 12.5 mm, temp/meshstudy_vares_cst/h012).

Seed = the least-squares edge flattening of the target surface (least_squares_edge_flatten), i.e. the
pattern with NO stress compensation.  The converged h012 run itself was seeded from the 17.7 mm pattern
(SEED_FROM, only 0.009 mm away), so the lsq seed is read from h012_stall_lsqseed/pattern.npz, which has
the identical leaf mesh (elements checked equal) and the identical frozen seed recipe.

Deviation u = Q X_converged - X_seed after the best 2-D rigid fit Q (pins are arbitrary), split into
along-meridian u_s and across-gore u_w.  Smoothed = local-linear (MLS1) Gaussian smoother on the nodes,
sigma = SMOOTH_SIGMA x h_eff, lumped-area weights: it reproduces any linear field exactly, so only the
element-scale part is removed; its rms is printed and reported in the panel titles.

    (a) u_s smoothed   (b) u_w smoothed   (c) seam-normal offset of the cut line along the meridian

USAGE (base anaconda env)
    python plot_vares_cst_deviation_from_seed.py
Outputs:
    fig_vares_cst_h012_deviation_from_seed.{pdf,png}
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MESHSTUDY_TARGET", "cst")

import MeshStudy_paraboloid_vares as MS                # noqa: E402
import MeshStudy_paraboloid_vares_ckpt as CK           # noqa: E402

ROOT = CK.ROOTS["cst"]
RUN = os.path.join(ROOT, "h012", "pattern.npz")
SEED = os.path.join(ROOT, "h012_stall_lsqseed", "pattern.npz")
SMOOTH_SIGMA = 1.5                                     # x h_eff
N_ST = 400                                             # stations along the seams


def rigid_fit(A, B):
    """Q A ~ B in least squares (rotation + translation); returns Q A."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    if np.linalg.det(U @ Vt) < 0:
        Vt[-1] *= -1
    return (A - ca) @ (U @ Vt) + cb


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


def main():
    a = dict(np.load(RUN, allow_pickle=True))
    b = dict(np.load(SEED, allow_pickle=True))
    assert bool(a["converged"]) and np.array_equal(a["elements"], b["elements"])
    assert np.allclose(a["X_leaf"], b["X_leaf"])
    T = a["elements"]
    leaf = a["X_leaf"][:, :2]
    Xs = b["X_initial"][:, :2]                         # lsq seed
    Xc = rigid_fit(a["X_converged"][:, :2], Xs)
    u = Xc - Xs
    w, ae = lumped_area(Xs, T)
    heff = float(np.sqrt(4 * ae.mean() / np.sqrt(3)))
    sig = SMOOTH_SIGMA * heff
    us = mls1(Xs, u, w, sig)
    res = u - us
    rms = lambda f: float(np.sqrt((w * f ** 2).sum() / w.sum()))
    print(f"h_eff {heff:.2f} mm, sigma {sig:.1f} mm, {len(Xs)} nodes")
    for k, nm in enumerate(("along u_s", "across u_w")):
        print(f"  {nm}: raw rms {rms(u[:, k]):.3f} range [{u[:, k].min():.2f}, {u[:, k].max():.2f}] mm | "
              f"smoothed rms {rms(us[:, k]):.3f} | element-scale residual rms {rms(res[:, k]):.4f} "
              f"max {np.abs(res[:, k]).max():.4f} mm")

    # seam-normal offset of the cut line (smoothed displacement, sampled along both seams of the seed)
    s_lo, s_mid, s_hi = (float(a[k]) for k in ("s_lo_mm", "s_mid_mm", "s_hi_mm"))
    x0 = s_lo - s_mid
    xs = x0 + np.linspace(*MS.STATION_FRAC, N_ST) * (s_hi - s_lo)
    hw = MS.leaf_half_width(xs, {k: (float(v) if np.ndim(v) == 0 and k.endswith("_mm") else v)
                                 for k, v in a.items()})
    seams = {}
    for sgn in (1.0, -1.0):
        pts_leaf = np.c_[xs, sgn * hw]
        S = MS.p1_sample(leaf, T, Xs, pts_leaf)        # seam of the seed
        U = MS.p1_sample(leaf, T, us, pts_leaf)
        t = np.gradient(S, axis=0)
        t /= np.linalg.norm(t, axis=1)[:, None]
        nrm = np.c_[-t[:, 1], t[:, 0]]
        nrm *= np.sign(nrm[:, 1] * sgn)[:, None]       # outward = away from the centreline
        seams[sgn] = np.sum(U * nrm, axis=1)
    print(f"  seam offset +y [{seams[1].min():.3f}, {seams[1].max():.3f}], "
          f"-y [{seams[-1].min():.3f}, {seams[-1].max():.3f}] mm")
    figure(Xs, T, us, res, rms, heff, sig, x0, xs, seams)


def figure(Xs, T, us, res, rms, heff, sig, x0, xs, seams, stem="fig_vares_cst_h012_deviation_from_seed"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans", "font.size": MS.FS_LABEL, "axes.labelsize": MS.FS_LABEL,
        "xtick.labelsize": MS.FS_TICK, "ytick.labelsize": MS.FS_TICK, "legend.fontsize": MS.FS_LEG,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    W = MS.WIDTH_IN
    fig = plt.figure(figsize=(W, W * 0.72))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.95)
    tri = mtri.Triangulation(Xs[:, 1] / 1000.0, (Xs[:, 0] - x0) / 1000.0, T)   # rim bottom, apex top
    yl = (xs[-1] - x0) / 1000.0
    xl = 1.08 * np.abs(Xs[:, 1]).max() / 1000.0
    for j, (k, lab, sym) in enumerate(((0, "along meridian", "u_s"), (1, "across gore", "u_w"))):
        ax = fig.add_subplot(gs[0, j])
        f = us[:, k]
        lim = float(np.abs(f).max())
        lv = np.linspace(-lim, lim, 23)
        cf = ax.tricontourf(tri, f, levels=lv, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -lim, lim))
        step = 2.0 if lim > 6 else 0.5                 # labelled lines: values readable in greyscale
        cl = np.arange(-np.floor(lim / step) * step, lim, step)
        cs = ax.tricontour(tri, f, levels=cl, colors="k", linewidths=0.35)
        ax.clabel(cs, fmt="%g", fontsize=5.5, inline_spacing=1)
        ax.set_aspect("equal")
        ax.set_xlim(-xl, xl)
        ax.set_ylim(-0.03, yl + 0.03)
        ax.set_xticks([-0.3, 0, 0.3])
        ax.set_xlabel("Across gore (m)")
        if j == 0:
            ax.set_ylabel("Meridian arc from rim, $s$ (m)")
        ax.set_title("$%s$, %s\nelement-scale part removed: rms %.4f mm" % (sym, lab, rms(res[:, k])),
                     fontsize=MS.FS_ANN, pad=3)
        cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.04, ticks=np.linspace(-lim, lim, 5).round(1))
        cb.set_label("$%s$ (mm)" % sym, fontsize=MS.FS_ANN)
        cb.ax.tick_params(labelsize=MS.FS_TICK, width=0.5)
        cb.outline.set_linewidth(0.5)
        ax.text(0.0, 1.1, "(%s)" % "ab"[j], transform=ax.transAxes, fontsize=MS.FS_PANEL, ha="right")
    c = fig.add_subplot(gs[0, 2])
    s = (xs - x0) / 1000.0
    c.axhline(0.0, color="0.3", lw=0.6)
    c.plot(s, seams[1], "-", color="#0072B2", lw=1.2, label="$+y$ seam")
    c.plot(s, seams[-1], "--", color="#D55E00", lw=1.2, label="$-y$ seam")
    c.set_xlabel("Meridian arc from rim, $s$ (m)")
    c.set_ylabel("Cut-line offset from seed,\noutward normal (mm)")
    c.legend(loc="best")
    c.text(0.0, 1.03, "(c)", transform=c.transAxes, fontsize=MS.FS_PANEL, ha="right")
    fig.suptitle("Converged VaReS gore (CST target, $h$ = 12.5 mm, $h_{\\rm eff}$ = %.1f mm) minus its "
                 "least-squares seed, best rigid fit removed;\nlocal-linear Gaussian smoothing, "
                 "$\\sigma$ = %.0f mm" % (heff, sig), fontsize=MS.FS_ANN, y=0.99)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(HERE, "%s.%s" % (stem, ext)), bbox_inches="tight", **kw)
    fig.savefig(os.path.join(HERE, "%s_preview.png" % stem), dpi=150, bbox_inches="tight")
    print("[out]", os.path.join(HERE, stem + ".{pdf,png}"))


if __name__ == "__main__":
    main()
