r"""plot_vares_shape_asymptote.py -- convergence of the VaReS cutting pattern to an ASYMPTOTE.

Ladder: the last complete one, tip-graded raw-CST (MESHSTUDY_TARGET=cstraw, MESHSTUDY_GRADING=tiptest, 70.7 -> 12.5 mm,
independent gmsh meshes, all converged).  Scalar shape quantities of the converged flat pattern X_converged, each as a
change from the developed leaf (the uncompensated gore), so the plotted value is the compensation itself:
  (a) gore length      tip node -> mid-point of the rim                                        (mm)
  (b) rim width        chord between the two rim corners                                       (mm)
  (c) width mid-gore   seam to seam at 50 % of the length from the tip                          (mm)
  (d) width near tip   seam to seam at 5 % of the length from the tip                           (mm)
Seam points at a station are interpolated along each seam in leaf x (the seam nodes keep their leaf order).
Asymptote: least-squares fit f(h) = f_inf + C h^p over all rungs (p free, 0.5..4); f_inf drawn dashed.  Band:
sqrt(se(f_inf)^2 + rms(fit residual)^2) -- the fit covariance alone understates it on independent (non-nested)
meshes, whose rung-to-rung scatter the residual carries.  If the trend is not resolved (se(p) > p: the quantity is
flat within its scatter, e.g. the rim width) the limit is the mean and the band the standard deviation.  Journal style: letters only, legends and axis labels the only
text, no fitted order on the panels (printed instead).
OUT  fig_vares_shape_asymptote.{pdf,png}, vares_shape_asymptote.json
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MESHSTUDY_TARGET", "cstraw")
os.environ.setdefault("MESHSTUDY_GRADING", "tiptest")
import MeshStudy_paraboloid_vares_ckpt as CK                      # noqa: E402
import MeshStudy_paraboloid_vares as MS                           # noqa: E402
import plot_meshstudy_paraboloid_vares_convergence as CV          # noqa: E402

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402

OUT = os.path.join(HERE, "fig_vares_shape_asymptote")
LINE_TOL = 1.0
STATIONS = {"mid": 0.50, "tip": 0.05}


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


def fit(h, y):
    def res(v):
        return v[0] + v[1] * h ** v[2] - y
    s = least_squares(res, x0=[y[-1], (y[0] - y[-1]) / h[0] ** 2, 2.0], bounds=([-np.inf, -np.inf, 0.5], [np.inf, np.inf, 4.0]))
    J = s.jac
    dof = max(len(y) - 3, 1)
    cov = np.linalg.pinv(J.T @ J) * (s.fun @ s.fun) / dof
    return s.x, np.sqrt(np.diag(cov))


def main():
    CK.configure("cstraw", "tiptest")
    hs = CK.ladder()
    rows = []
    for h in hs:
        d = np.load(MS.rung_npz(h))
        if not bool(d["converged"]):
            continue
        Xl, P, seam = d["X_leaf"][:, :2], d["X_converged"][:, :2], np.asarray(d["seam_ids"], int)
        rows.append((CV.h_eff(h), quantities(Xl, P, seam) - quantities(Xl, Xl, seam), MS.tag(h)))
    h = np.array([r[0] for r in rows])
    Q = np.array([r[1] for r in rows])
    names = ["Gore length", "Rim width", "Width at mid-gore", "Width near the tip"]
    keys = ["length", "rim_width", "width_mid", "width_tip"]
    out = {"rungs": [r[2] for r in rows], "h_eff": h.tolist()}

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True, "legend.frameon": False})
    W, H = 170.0, 142.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    F = lambda x, y, w, hh: [x / W, y / H, w / W, hh / H]          # noqa: E731
    pos = [F(24, 88, 58, 44), F(106, 88, 58, 44), F(24, 26, 58, 44), F(106, 26, 58, 44)]
    hh = np.linspace(0, 75, 200)
    for i, (k, nm) in enumerate(zip(keys, names)):
        ax = fig.add_axes(pos[i])
        y = Q[:, i]
        v, se = fit(h, y)
        flat = not (se[2] < v[2])
        if flat:
            lim, band = float(y.mean()), float(y.std(ddof=1))
        else:
            r = v[0] + v[1] * h ** v[2] - y
            lim, band = float(v[0]), float(np.hypot(se[0], np.sqrt(np.mean(r ** 2))))
        out[k] = dict(values_mm=y.tolist(), limit=lim, band=band, trend_resolved=not flat, f_inf=float(v[0]),
                      C=float(v[1]), p=float(v[2]), p_se=float(se[2]))
        print("%-18s values %s  ->  limit %.4f +- %.4f mm   (%s)" % (
            nm, np.round(y, 4), lim, band, "flat: mean +- sd" if flat else "p %.2f +- %.2f" % (v[2], se[2])))
        ax.axhspan(lim - band, lim + band, color="0.85", lw=0, zorder=0)
        ax.axhline(lim, color="0.35", lw=1.0, ls=(0, (4, 2)), zorder=1)
        if not flat:
            ax.plot(hh, v[0] + v[1] * hh ** v[2], color="#0072B2", lw=1.1, zorder=2)
        ax.plot(h, y, "o", color="#0072B2", mec="k", mew=0.6, ms=6, zorder=3)
        ax.set_xlim(0, 75)
        ax.set_xlabel(r"Element size $h_{\mathrm{eff}}$ (mm)")
        ax.set_ylabel("%s change (mm)" % nm, fontsize=9)
        fig.text(pos[i][0], pos[i][1] + pos[i][3] + 2.5 / H, "(%s)" % "abcd"[i], fontsize=11, fontweight="bold")
    fig.legend(handles=[Line2D([], [], color="#0072B2", marker="o", mec="k", mew=0.6, ms=6, lw=0, label="VaReS pattern"),
                        Line2D([], [], color="#0072B2", lw=1.1, label=r"fit $f_\infty + C\,h^p$"),
                        Line2D([], [], color="0.35", lw=1.0, ls=(0, (4, 2)), label=r"extrapolated limit $f_\infty$"),
                        Patch(color="0.85", label="uncertainty of the limit")],
               loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 0.0), handlelength=2.0)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig("%s.%s" % (OUT, ext), **kw)
    with open(os.path.join(HERE, "vares_shape_asymptote.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
