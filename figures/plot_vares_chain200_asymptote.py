r"""plot_vares_chain200_asymptote.py -- shape convergence of the VaReS cutting pattern on the nested chain from a 200 mm
base (grading tippolar210c200nest: 200 -> 100 -> 50 -> 25 -> 12.5 mm far field, tip = the 100 mm chain's tip), with the
100 mm chain's converged 6.25 mm rung (tippolar210nest h006, h_eff 4.9 mm) added as an independent point.
Same measures, fit and layout as plot_vares_chain100_asymptote.py: f_inf + a h^p fitted to the 200 mm chain's converged
rungs only; the old h006 is a different (non-nested) mesh family, so it is plotted as a separate marker and NOT fitted --
its deviation from the fit is in the json.
OUT  fig_vares_chain200_field_asymptote.{pdf,png}, fig_vares_chain200_shape_asymptote.{pdf,png}, vares_chain200_asymptote.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_vares_chain100_asymptote as B           # noqa: E402
from plot_vares_shape_asymptote import plt, Line2D, Patch   # noqa: E402

COL, COL_X = "#D55E00", "#0072B2"


def load_grading(grading, rungs):
    B.GRADING = grading
    return B.load(rungs)


def main():
    rows = load_grading("tippolar210c200nest", (200.0, 100.0, 50.0, 25.0, 12.5))
    xtra = load_grading("tippolar210nest", (6.25,))
    for r in rows + xtra:
        print("%s h_eff %6.2f  %-14s ||R|| %.3e  src %s" % (r["tag"], r["h"], r["crit"], r["R"], r["src"]))
    fitr = [r for r in rows if r["ok"]]
    hF = np.array([r["h"] for r in fitr])
    out = {"rungs": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in rows],
           "old_h006_100mm_chain": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in xtra]}
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True, "legend.frameon": False})
    for key, names, ylab in B.FIGS:
        n = len(names)
        W, H = (190.0, 90.0) if n == 3 else (170.0, 150.0)
        fig = plt.figure(figsize=(W / 25.4, H / 25.4))
        F = lambda x, y, w, hh: [x / W, y / H, w / W, hh / H]          # noqa: E731
        pos = ([F(20 + 60 * i, 32, 46, 46) for i in range(3)] if n == 3 else
               [F(24, 92, 58, 46), F(106, 92, 58, 46), F(24, 28, 58, 46), F(106, 28, 58, 46)])
        XMAX = 105
        hh = np.linspace(0, XMAX, 200)
        out[key] = {}
        for i, nm in enumerate(names):
            ax = fig.add_axes(pos[i])
            y = np.array([r[key][i] for r in fitr])
            v, se, lim, band, flat = B.limit(hF, y)
            fit_at = (lambda h_: lim) if flat else (lambda h_: v[0] + v[1] * h_ ** v[2])
            res = dict(limit=lim, band=band, trend_resolved=not flat, p=float(v[2]), p_se=float(se[2]),
                       fit_rungs=[r["tag"] for r in fitr],
                       old_h006={r["tag"]: dict(h_eff=r["h"], value=float(r[key][i]),
                                                dev_from_limit=float(r[key][i] - lim),
                                                dev_from_fit=float(r[key][i] - fit_at(r["h"]))) for r in xtra})
            out[key][nm] = res
            print("%-18s %s -> limit %.4f +- %.4f mm (%s)  old h006 %s" % (
                nm, np.round(y, 4), lim, band, "flat" if flat else "p %.2f +- %.2f" % (v[2], se[2]),
                ", ".join("%.4f (fit %+.4f)" % (r[key][i], r[key][i] - fit_at(r["h"])) for r in xtra)))
            ax.axhspan(lim - band, lim + band, color="0.85", lw=0, zorder=0)
            ax.axhline(lim, color="0.35", lw=1.0, ls=(0, (4, 2)), zorder=1)
            if not flat:
                ax.plot(hh, v[0] + v[1] * hh ** v[2], color=COL, lw=1.1, zorder=2)
            ax.plot(hF, y, "D", color=COL, mec="k", mew=0.6, ms=6, zorder=3)
            if xtra:
                ax.plot([r["h"] for r in xtra], [r[key][i] for r in xtra], "D", color=COL, mec="k", mew=0.6, ms=6,
                        zorder=3)        # 2026-10-14 (user): same orange series, no separate legend
            ax.set_xlim(0, XMAX)
            ax.ticklabel_format(axis="y", useOffset=False)
            ax.set_xlabel(r"Element size $h_{\mathrm{eff}}$ (mm)")
            if i % (3 if n == 3 else 2) == 0:
                ax.set_ylabel(ylab)
            fig.text(pos[i][0], pos[i][1] + pos[i][3] + 2.5 / H, "(%s)" % "abcd"[i], fontsize=11, fontweight="bold")
        hd = [Line2D([], [], color=COL, marker="D", mec="k", mew=0.6, ms=6, lw=0, label="converged rung"),
              Line2D([], [], color=COL, lw=1.1, label=r"fit $f_\infty + a\,h^p$"),
              Line2D([], [], color="0.35", lw=1.0, ls=(0, (4, 2)), label=r"extrapolated limit $f_\infty$"),
              Patch(color="0.85", label="uncertainty of the limit")]
        fig.legend(handles=hd, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0), handlelength=2.0)
        o = os.path.join(HERE, "fig_vares_chain200_%s_asymptote" % key)
        for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
            fig.savefig("%s.%s" % (o, ext), **kw)
        print("[out]", o)
    with open(os.path.join(HERE, "vares_chain200_asymptote.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
