r"""plot_vares_chain100_asymptote.py -- shape convergence of the VaReS cutting pattern to an asymptote on the SINGLE
nested tip-polar chain from 100 mm (grading tippolar210nest: 100 -> 50 -> 25 -> 12.5 -> 6.25 mm, each a 1->4
subdivision of the previous one).

Two figures, both from plot_vares_nested_*_asymptote.py measures:
  field   rms of the compensation field u = pattern - developed leaf (rigid motion removed, area-weighted):
          (a) whole gore  (b) away from the tip  (c) tip zone (last 10 % of the length)
  lengths change of gore length, rim width, width at mid-gore, width near the tip
Fit f_inf + a h^p to the CONVERGED rungs (tolerance or criterion (a) certificate).  A rung that is not converged is
shown as an OPEN marker and kept out of the fit: h006 is read from h006/pattern.npz when it is converged, otherwise
from the snapshot h006/pattern_snapshot_uncertified_*.npz (||R|| 2.47e-7 = 2.5 tol, certificate pending).
Family D (128 -> 64 mm, second nested family, h_eff ~98 / ~49): converged rungs plotted as blue circles, not fitted;
their deviation from the chain's fit is in the json.
OUT  fig_vares_chain100_field_asymptote.{pdf,png}, fig_vares_chain100_shape_asymptote.{pdf,png},
     vares_chain100_asymptote.json
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_vares_shape_asymptote import quantities, fit, CK, MS, CV, plt, Line2D, Patch   # noqa: E402
from plot_vares_nested_field_asymptote import field_rms                                  # noqa: E402

GRADING = "tippolar210nest"
RUNGS = (100.0, 50.0, 25.0, 12.5, 6.25)
# family D: second nested family 128 -> 64 mm, h_eff ~98 / ~49 -- coarser points, an independent mesh
# family, shown but NOT fitted (the fit stays the 100 mm chain's)
# OFF by default -- the figure is the 100 mm chain only;
# --family-d adds the D points back (results kept in the study root)
RUNGS_D = (128.0, 64.0) if "--family-d" in sys.argv else ()
COL, COL_D = "#D55E00", "#0072B2"
FIGS = [("field", ["Whole gore", "Away from the tip", "Tip zone"], "rms shape change (mm)"),
        ("shape", ["Gore length", "Rim width", "Width at mid-gore", "Width near the tip"], "Change from the leaf (mm)")]


def load(rungs=RUNGS):
    CK.configure("cstraw", GRADING)
    rows = []
    for h in rungs:
        p = MS.rung_npz(h)
        if not os.path.exists(p) or not bool(np.load(p)["converged"]):
            snap = sorted(glob.glob(os.path.join(os.path.dirname(p), "pattern_snapshot_uncertified_*.npz")))
            if not os.path.exists(p) and not snap:
                continue
            p = snap[-1] if snap else p
        d = np.load(p, allow_pickle=True)
        Xl, P = d["X_leaf"][:, :2], d["X_converged"][:, :2]
        seam, T = np.asarray(d["seam_ids"], int), np.asarray(d["elements"], int)
        crit = str(d["converged_criterion"]) if "converged_criterion" in d.files else ("tol" if bool(d["converged"]) else "none")
        e = d["X_spatial"][T]
        he = float(np.sqrt(4.0 * 0.5 * np.linalg.norm(np.cross(e[:, 1] - e[:, 0], e[:, 2] - e[:, 0]), axis=1).sum() /
                           (np.sqrt(3.0) * len(T))))          # = CV.h_eff, from the file actually read
        rows.append(dict(tag=MS.tag(h), h=he, ok=bool(d["converged"]), crit=crit,
                         R=float(d["R_history"][-1]), field=field_rms(Xl, P, T),
                         shape=quantities(Xl, P, seam) - quantities(Xl, Xl, seam), src=os.path.basename(p)))
    return rows


def limit(h, y):
    v, se = fit(h, y)
    flat = not (se[2] < v[2]) or v[2] <= 0.51
    if flat:
        return v, se, float(y.mean()), float(y.std(ddof=1)), True
    r = v[0] + v[1] * h ** v[2] - y
    return v, se, float(v[0]), float(np.hypot(se[0], np.sqrt(np.mean(r ** 2)))), False


def main():
    rows = load()
    rowsD = [r for r in load(RUNGS_D) if r["ok"]]
    for r in rows + rowsD:
        print("%s h_eff %6.2f  %-14s ||R|| %.3e  field %s  shape %s" % (r["tag"], r["h"], r["crit"], r["R"],
              np.round(r["field"], 4), np.round(r["shape"], 4)))
    out = {"rungs": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in rows],
           "family_D": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in rowsD]}
    fitr = [r for r in rows if r["ok"]]
    opn = [r for r in rows if not r["ok"]]
    hF = np.array([r["h"] for r in fitr])
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True, "legend.frameon": False})
    for key, names, ylab in FIGS:
        n = len(names)
        W, H = (190.0, 90.0) if n == 3 else (170.0, 150.0)
        fig = plt.figure(figsize=(W / 25.4, H / 25.4))
        F = lambda x, y, w, hh: [x / W, y / H, w / W, hh / H]          # noqa: E731
        pos = ([F(20 + 60 * i, 32, 46, 46) for i in range(3)] if n == 3 else
               [F(24, 92, 58, 46), F(106, 92, 58, 46), F(24, 28, 58, 46), F(106, 28, 58, 46)])
        XMAX = 110 if rowsD else 85
        hh = np.linspace(0, XMAX, 200)
        out[key] = {}
        for i, nm in enumerate(names):
            ax = fig.add_axes(pos[i])
            y = np.array([r[key][i] for r in fitr])
            v, se, lim, band, flat = limit(hF, y)
            res = dict(limit=lim, band=band, trend_resolved=not flat, p=float(v[2]), p_se=float(se[2]),
                       fit_rungs=[r["tag"] for r in fitr])
            if opn:
                res["open_rungs"] = {r["tag"]: dict(value=float(r[key][i]), dev_from_limit=float(r[key][i] - lim),
                                                    R=r["R"]) for r in opn}
            out[key][nm] = res
            print("%-18s %s -> limit %.4f +- %.4f mm (%s)%s" % (nm, np.round(y, 4), lim, band,
                  "flat" if flat else "p %.2f +- %.2f" % (v[2], se[2]),
                  "".join("  [%s open %.4f]" % (r["tag"], r[key][i]) for r in opn)))
            ax.axhspan(lim - band, lim + band, color="0.85", lw=0, zorder=0)
            ax.axhline(lim, color="0.35", lw=1.0, ls=(0, (4, 2)), zorder=1)
            if not flat:
                ax.plot(hh, v[0] + v[1] * hh ** v[2], color=COL, lw=1.1, zorder=2)
            ax.plot(hF, y, "D", color=COL, mec="k", mew=0.6, ms=6, zorder=3)
            if opn:
                ax.plot([r["h"] for r in opn], [r[key][i] for r in opn], "D", mfc="none", mec="k", mew=0.9, ms=6,
                        zorder=4)
            if rowsD:
                ax.plot([r["h"] for r in rowsD], [r[key][i] for r in rowsD], "o", color=COL_D, mec="k", mew=0.6,
                        ms=6, zorder=3)
                res["family_D"] = {r["tag"]: dict(value=float(r[key][i]), dev_from_limit=float(r[key][i] - lim),
                                                  dev_from_fit=float(r[key][i] - (lim if flat else v[0] + v[1] * r["h"] ** v[2])))
                                   for r in rowsD}
            ax.set_xlim(0, XMAX)
            ax.ticklabel_format(axis="y", useOffset=False)
            ax.set_xlabel(r"Element size $h_{\mathrm{eff}}$ (mm)")
            if i % (3 if n == 3 else 2) == 0:
                ax.set_ylabel(ylab)
            fig.text(pos[i][0], pos[i][1] + pos[i][3] + 2.5 / H, "(%s)" % "abcd"[i], fontsize=11, fontweight="bold")
        hd = [Line2D([], [], color=COL, marker="D", mec="k", mew=0.6, ms=6, lw=0, label="converged rung")]
        if opn:
            hd.append(Line2D([], [], color="k", marker="D", mfc="none", mew=0.9, ms=6, lw=0,
                             label="not yet converged (not fitted)"))
        if rowsD:
            hd.append(Line2D([], [], color=COL_D, marker="o", mec="k", mew=0.6, ms=6, lw=0,
                             label="second nested family (not fitted)"))
        hd += [Line2D([], [], color=COL, lw=1.1, label=r"fit $f_\infty + a\,h^p$"),
               Line2D([], [], color="0.35", lw=1.0, ls=(0, (4, 2)), label=r"extrapolated limit $f_\infty$"),
               Patch(color="0.85", label="uncertainty of the limit")]
        fig.legend(handles=hd, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0), handlelength=2.0)
        o = os.path.join(HERE, "fig_vares_chain100_%s_asymptote" % key)
        for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
            fig.savefig("%s.%s" % (o, ext), **kw)
        print("[out]", o)
    with open(os.path.join(HERE, "vares_chain100_asymptote.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
