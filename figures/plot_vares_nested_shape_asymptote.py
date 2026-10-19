r"""plot_vares_nested_shape_asymptote.py -- convergence of the VaReS cutting pattern to an asymptote, TIP-GRADED NESTED
meshes only.

Series (converged rungs only, read from the study roots):
  tip-polar C = 105, two nested families: A 50 -> 25, B 70.7 -> 35.35 -> 17.7     (grading tippolarnest)
  tip-polar C = 210, one nested family from 100 mm: 100 -> 50 -> 25 (12.5, 6.25 pending)   (tippolar210nest)
Quantities, change from the developed leaf (= the compensation), as plot_vares_shape_asymptote.py: gore length, rim
width, width at mid-gore, width at 5 % of the length from the tip.  Asymptote: f_inf + C h^p fitted to the C = 105
rungs (5 points); band sqrt(se^2 + rms residual^2), or mean +- sd if the trend is not resolved.  The C = 210 rungs are
an independent mesh design: they should approach the same limit (no fit while < 4 rungs).  Journal style.
OUT  fig_vares_nested_shape_asymptote.{pdf,png}, vares_nested_shape_asymptote.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_vares_shape_asymptote import quantities, fit, CK, MS, CV, plt, Line2D, Patch   # noqa: E402

OUT = os.path.join(HERE, "fig_vares_nested_shape_asymptote")
SERIES = [("tippolarnest", "A", "tip-polar C = 105, family A", "#0072B2", "o"),
          ("tippolarnest", "B", "tip-polar C = 105, family B", "#0072B2", "s"),
          ("tippolar210nest", "C", "tip-polar C = 210, from 100 mm", "#D55E00", "D")]
FAMILY = {"tippolarnest": {"A": (50.0, 25.0, 12.5), "B": (70.7, 35.4, 17.7)},
          "tippolar210nest": {"C": (100.0, 50.0, 25.0, 12.5, 6.25)}}
NAMES = ["Gore length", "Rim width", "Width at mid-gore", "Width near the tip"]
KEYS = ["length", "rim_width", "width_mid", "width_tip"]


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
        Xl, P, seam = d["X_leaf"][:, :2], d["X_converged"][:, :2], np.asarray(d["seam_ids"], int)
        out.append((CV.h_eff(h), quantities(Xl, P, seam) - quantities(Xl, Xl, seam), MS.tag(h)))
    return out


def main():
    data = {(g, f): load(g, f) for g, f, *_ in SERIES}
    fitpts = data[("tippolarnest", "A")] + data[("tippolarnest", "B")]
    hF = np.array([r[0] for r in fitpts])
    QF = np.array([r[1] for r in fitpts])
    o = np.argsort(-hF)
    hF, QF = hF[o], QF[o]
    out = {"%s %s" % k: [(r[2], r[0], r[1].tolist()) for r in v] for k, v in data.items()}

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True, "legend.frameon": False})
    W, H = 170.0, 150.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    F = lambda x, y, w, hh: [x / W, y / H, w / W, hh / H]          # noqa: E731
    pos = [F(24, 96, 58, 44), F(106, 96, 58, 44), F(24, 34, 58, 44), F(106, 34, 58, 44)]
    hh = np.linspace(0, 85, 200)
    for i, (k, nm) in enumerate(zip(KEYS, NAMES)):
        ax = fig.add_axes(pos[i])
        y = QF[:, i]
        v, se = fit(hF, y)
        flat = not (se[2] < v[2]) or v[2] <= 0.51
        if flat:
            lim, band = float(y.mean()), float(y.std(ddof=1))
        else:
            r = v[0] + v[1] * hF ** v[2] - y
            lim, band = float(v[0]), float(np.hypot(se[0], np.sqrt(np.mean(r ** 2))))
        out[k] = dict(limit=lim, band=band, trend_resolved=not flat, p=float(v[2]), p_se=float(se[2]))
        c210 = [r[1][i] for r in data[("tippolar210nest", "C")]]
        print("%-18s C105 %s -> limit %.4f +- %.4f mm (%s);  C210 %s" % (
            nm, np.round(y, 4), lim, band, "flat" if flat else "p %.2f +- %.2f" % (v[2], se[2]), np.round(c210, 4)))
        ax.axhspan(lim - band, lim + band, color="0.85", lw=0, zorder=0)
        ax.axhline(lim, color="0.35", lw=1.0, ls=(0, (4, 2)), zorder=1)
        if not flat:
            ax.plot(hh, v[0] + v[1] * hh ** v[2], color="#0072B2", lw=1.1, zorder=2)
        for g, f, lab, col, mk in SERIES:
            pts = data[(g, f)]
            ax.plot([r[0] for r in pts], [r[1][i] for r in pts], mk, color=col, mec="k", mew=0.6, ms=6, zorder=3)
        ax.set_xlim(0, 85)
        ax.set_xlabel(r"Element size $h_{\mathrm{eff}}$ (mm)")
        ax.set_ylabel("%s change (mm)" % nm, fontsize=9)
        fig.text(pos[i][0], pos[i][1] + pos[i][3] + 2.5 / H, "(%s)" % "abcd"[i], fontsize=11, fontweight="bold")
    fig.legend(handles=[Line2D([], [], color=c, marker=m, mec="k", mew=0.6, ms=6, lw=0, label=l) for _, _, l, c, m in SERIES] +
               [Line2D([], [], color="#0072B2", lw=1.1, label=r"fit $f_\infty + C\,h^p$ (C = 105)"),
                Line2D([], [], color="0.35", lw=1.0, ls=(0, (4, 2)), label=r"extrapolated limit $f_\infty$"),
                Patch(color="0.85", label="uncertainty of the limit")],
               loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0), handlelength=2.0)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig("%s.%s" % (OUT, ext), **kw)
    with open(os.path.join(HERE, "vares_nested_shape_asymptote.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
