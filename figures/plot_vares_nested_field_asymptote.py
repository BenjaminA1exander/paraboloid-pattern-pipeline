r"""plot_vares_nested_field_asymptote.py -- convergence of the WHOLE cutting-pattern shape change to an asymptote,
tip-graded nested meshes only.

Per converged rung, the shape change is the compensation FIELD u(X) = pattern - developed leaf, after the pattern is
rigidly aligned onto the leaf (area-weighted 2-D Procrustes, so the dense tip nodes do not dominate).  Its size is the
area-weighted rms of |u| over the leaf (P1 field, exact element integral of |u|^2), in three regions:
  (a) whole gore     (b) away from the tip     (c) tip zone = last 10 % of the gore length
(region by element centroid).  Series and asymptote fit as plot_vares_nested_shape_asymptote.py: f_inf + C h^p fitted
to the tip-polar C = 105 rungs; the C = 210 rungs (other mesh design) are an independent check.  Journal style.
OUT  fig_vares_nested_field_asymptote.{pdf,png}, vares_nested_field_asymptote.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_vares_shape_asymptote import fit, CK, MS, CV, plt, Line2D, Patch      # noqa: E402
from plot_vares_nested_shape_asymptote import SERIES, FAMILY                   # noqa: E402

OUT = os.path.join(HERE, "fig_vares_nested_field_asymptote")
TIP_FRAC = 0.10
NAMES = ["Whole gore", "Away from the tip", "Tip zone"]


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


def load(grading, fam):
    CK.configure("cstraw", grading)
    out = []
    for h in FAMILY[grading][fam]:
        p = MS.rung_npz(h)
        if not os.path.exists(p):
            continue
        d = np.load(p)
        if not bool(d["converged"]):
            continue
        out.append((CV.h_eff(h), field_rms(d["X_leaf"][:, :2], d["X_converged"][:, :2], np.asarray(d["elements"], int)),
                    MS.tag(h)))
    return out


def main():
    data = {(g, f): load(g, f) for g, f, *_ in SERIES}
    pts = data[("tippolarnest", "A")] + data[("tippolarnest", "B")]
    hF = np.array([r[0] for r in pts])
    QF = np.array([r[1] for r in pts])
    o = np.argsort(-hF)
    hF, QF = hF[o], QF[o]
    out = {"%s %s" % k: [(r[2], r[0], r[1].tolist()) for r in v] for k, v in data.items()}

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True, "legend.frameon": False})
    W, H = 190.0, 90.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    F = lambda x, y, w, hh: [x / W, y / H, w / W, hh / H]          # noqa: E731
    hh = np.linspace(0, 85, 200)
    for i, nm in enumerate(NAMES):
        pos = F(20 + 60 * i, 32, 46, 46)
        ax = fig.add_axes(pos)
        y = QF[:, i]
        v, se = fit(hF, y)
        flat = not (se[2] < v[2]) or v[2] <= 0.51
        if flat:
            lim, band = float(y.mean()), float(y.std(ddof=1))
        else:
            r = v[0] + v[1] * hF ** v[2] - y
            lim, band = float(v[0]), float(np.hypot(se[0], np.sqrt(np.mean(r ** 2))))
        out[nm] = dict(limit=lim, band=band, trend_resolved=not flat, p=float(v[2]), p_se=float(se[2]))
        c210 = [r[1][i] for r in data[("tippolar210nest", "C")]]
        print("%-18s C105 %s -> limit %.4f +- %.4f mm (%s);  C210 %s" % (
            nm, np.round(y, 4), lim, band, "flat" if flat else "p %.2f +- %.2f" % (v[2], se[2]), np.round(c210, 4)))
        ax.axhspan(lim - band, lim + band, color="0.85", lw=0, zorder=0)
        ax.axhline(lim, color="0.35", lw=1.0, ls=(0, (4, 2)), zorder=1)
        if not flat:
            ax.plot(hh, v[0] + v[1] * hh ** v[2], color="#0072B2", lw=1.1, zorder=2)
        for g, f, lab, col, mk in SERIES:
            pp = data[(g, f)]
            ax.plot([r[0] for r in pp], [r[1][i] for r in pp], mk, color=col, mec="k", mew=0.6, ms=6, zorder=3)
        ax.set_xlim(0, 85)
        ax.set_xlabel(r"Element size $h_{\mathrm{eff}}$ (mm)")
        if i == 0:
            ax.set_ylabel("rms shape change (mm)")
        fig.text(pos[0], pos[1] + pos[3] + 2.5 / H, "(%s)" % "abc"[i], fontsize=11, fontweight="bold")
    fig.legend(handles=[Line2D([], [], color=c, marker=m, mec="k", mew=0.6, ms=6, lw=0, label=l) for _, _, l, c, m in SERIES] +
               [Line2D([], [], color="#0072B2", lw=1.1, label=r"fit $f_\infty + a\,h^p$ (C = 105)"),
                Line2D([], [], color="0.35", lw=1.0, ls=(0, (4, 2)), label=r"extrapolated limit $f_\infty$"),
                Patch(color="0.85", label="uncertainty of the limit")],
               loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0), handlelength=2.0)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig("%s.%s" % (OUT, ext), **kw)
    with open(os.path.join(HERE, "vares_nested_field_asymptote.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
