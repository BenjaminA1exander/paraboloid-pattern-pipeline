r"""plot_seamtip_asymptote_combined.py -- normalised, dimensionless asymptote plot of the seam- and tip-resolved Abaqus
sector ladder (build_seamtip_ladder.py / sector_seamtip_ladder.py), VaReS and uniform shrinkage.

Panels (a) VaReS pattern, L0-L4 (5 rungs) ; (b) uniform shrinkage, L0-L3 (4 rungs; L4 fails by the seam-rim corner
crush, left out).  In each panel shape, warp and weft are divided by their own extrapolated limit f_inf, so all
approach 1 on one dimensionless axis; the fit f_inf + a h^p and its band are divided by the same f_inf.
Measures: abaqus_ladder_seamband.load / measures, unchanged, over the WHOLE gore (band 0: the seams are resolved and in
scope on this ladder) -- shape rms (mm), warp and weft rms(sigma / sigma* - 1).  Apex error is reported, not drawn
(a signed point value, near zero: f / f_inf is not meaningful).
Run used per rung = the import the settle chain declared settled (asserted against SETTLED below, from queue.log).
Fit in NOMINAL h (100 ... 6.25 mm, exact ratio 2: every element is subdivided 1 -> 4); abscissa = elements in the
full 24-gore dish, N = N_L0 (100 / h)^2 exactly.
Asymptote rule: plot_abaqus_sector_vares_asymptote.classify, unchanged; a measure the rule does not call "converging"
has no f_inf and is NOT drawn (reported, and listed in the json); "flat" (no mesh trend) is drawn
as open markers / its mean, band +- sd, no curve.  With 4 rungs (uniform) the 3-parameter fit has one
residual degree of freedom -- stated in the json.
Journal style: letters, legends and axis labels only; Okabe-Ito colours + distinct markers (greyscale safe).
OUT results/abaqus_ladder_seamtip/paper/fig_seamtip_asymptote_combined.{pdf,png,json}
Per-rung measures are cached in <OUT>_measures.json (the L4 npz is 0.4 GB); --reload recomputes them.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abaqus_ladder_seamband as SB                                   # noqa: E402
from plot_abaqus_sector_vares_asymptote import classify              # noqa: E402
from plot_vares_shape_asymptote import plt, Line2D                   # noqa: E402

TEMP = SB.TEMP
EULER = os.path.join(TEMP, "abaqus_sector_seamtip_mirror")
DB = os.path.join(TEMP, "abaqus_sector_seamtip_db_mirror")
TAGS = ("h100", "h050", "h025", "h012", "h006")
HNOM = {"h100": 100.0, "h050": 50.0, "h025": 25.0, "h012": 12.5, "h006": 6.25}
ROOT = {"vares": {"h100": EULER, "h050": EULER, "h025": EULER, "h012": EULER, "h006": DB},
        "uniform": {"h100": EULER, "h050": EULER, "h025": EULER, "h012": EULER}}       # L4 uniform: excluded
SETTLED = {"vares": {"h100": 1, "h050": 1, "h025": 1, "h012": 2, "h006": 2},
           "uniform": {"h100": 2, "h050": 4, "h025": 6, "h012": 7}}
SER = [("shape_rms", "Shape", "#0072B2", "o"), ("warp_rms", "Warp stress", "#E69F00", "s"),
       ("weft_rms", "Weft stress", "#009E73", "D")]
PAT = [("vares", "VaReS pattern"), ("uniform", "Uniform shrinkage")]
OUTD = os.path.join(HERE, "results", "abaqus_ladder_seamtip", "paper")
# --area (2026-10-16, user: "replot it by area"): each measure by AREA QUANTILE instead of rms -- P90 = the error level
# not exceeded on 90 % of the gore area (P50, the area median, in the cache / json too); per element |e| for warp and
# weft, sqrt(element mean square normal deviation) for shape.  The rms weights each element by its error squared, so a
# 1 % seam strip with 10x the error carries half of it; the quantile counts area.
AREA = "--area" in sys.argv
# --mae (2026-10-16, user: "error should be weighted by the area of the error; a tiny element with an error shouldn't
# overwhelm its neighbours"): area-weighted MEAN ABSOLUTE error, sum(a_i |e_i|) / sum(a_i) -- area-weighted like the
# rms but not squared, so a small region with a large error counts by (area x error), not (area x error^2)
MAE = "--mae" in sys.argv
QS = (50, 90)
# MEASURE_VERSION 2 (2026-10-16 audit): warp / weft from the membrane resultant SF (SB.load force=True) instead of the
# section-point mean stress; shape L1 (mae) = exact integral of |d| per element.  Cache entries of another version are
# recomputed.
MEASURE_VERSION = 2
CACHE = os.path.join(OUTD, "fig_seamtip_asymptote_combined_measures.json")
OUT = os.path.join(OUTD, "fig_seamtip_asymptote_combined" + ("_area_p90" if AREA else "_mae" if MAE else ""))
if AREA or MAE:
    SER = [(k.replace("_rms", "_p90" if AREA else "_mae"), lab, c, m) for k, lab, c, m in SER]


def wquantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, q / 100.0 * c[-1])])


def area_quantiles(e):
    f = {"shape": np.sqrt(e["m2"]), "warp": np.abs(e["warp"]), "weft": np.abs(e["weft"])}
    out = {f"{k}_p{q}": wquantile(x, e["area"], q) for k, x in f.items() for q in QS}
    out.update({f"{k}_mae": float((e["area"] * x).sum() / e["area"].sum()) for k, x in f.items()})
    out["shape_mae"] = float(e["absd"].sum() / e["area"].sum())          # exact integral of |d|
    return out


def rung_measures(v, t):
    jd = os.path.join(ROOT[v][t], t, f"{v}_dp")
    npz = SB.last_job(jd, v)
    if npz is None:
        return None
    k = int(os.path.basename(npz).split("_")[1][3:])
    assert k == SETTLED[v][t], f"{jd}: last import stl{k}, settled one is stl{SETTLED[v][t]}"
    e = SB.load(jd, v, force=True)
    m = SB.measures(e, 0.0)
    m.update(area_quantiles(e), version=MEASURE_VERSION)
    fl = np.load(os.path.join(jd, "flower.npz"))
    m.update(job=e["job"], frames=e["frames"], n_el_sector=int(len(e["area"])), sector_gores=int(fl["sector_gores"]),
             heff=e["heff"])
    return m


def main():
    os.makedirs(OUTD, exist_ok=True)
    cache = CACHE
    M = json.load(open(cache)) if os.path.exists(cache) and "--reload" not in sys.argv else {}
    missing = []
    for v, _ in PAT:
        M.setdefault(v, {})
        for t in ROOT[v]:
            if t not in M[v] or M[v][t].get("version") != MEASURE_VERSION:
                m = rung_measures(v, t)
                if m is None:
                    missing.append(f"{v} {t}")
                    continue
                M[v][t] = m
                with open(cache, "w") as fh:
                    json.dump(M, fh, indent=1, default=float)
            m = M[v][t]
            print(f"{v:8s} {t}: {m['job']:16s} {m['n_el_sector']:8d} el/sector  apex {m['apex_err']:+.4f} mm  shape "
                  f"{m['shape_rms']:.5f} mm  warp {100 * m['warp_rms']:.4f} %  weft {100 * m['weft_rms']:.4f} %  | area P90: shape "
                  f"{m['shape_p90']:.5f} mm  warp {100 * m['warp_p90']:.4f} %  weft {100 * m['weft_p90']:.4f} %")
    if missing:
        raise SystemExit("[seamtip] not mirrored yet: " + ", ".join(missing))

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True, "legend.frameon": False})
    W, H = 180.0, 78.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    pos = [[16 / W, 24 / H, 72 / W, 48 / H], [104 / W, 24 / H, 72 / W, 48 / H]]
    out = {"note": "whole gore (no seam band); fit in nominal h; abscissa elements in the full dish", "pattern": {}}
    hh = np.geomspace(5.0, 120.0, 300)
    axes = []
    kinds = set()
    for i, (v, title) in enumerate(PAT):
        tags = list(ROOT[v])
        h = np.array([HNOM[t] for t in tags])
        ng = M[v][tags[0]]["sector_gores"]
        N = np.array([M[v][t]["n_el_sector"] * ng for t in tags], float)
        cN = N * h ** 2
        assert np.ptp(cN) == 0, f"{v}: N h^2 not constant ({cN})"
        cN = float(cN[0])
        ax = fig.add_axes(pos[i])
        axes.append(ax)
        o = out["pattern"][v] = {"rungs": tags, "h_nominal": h.tolist(), "n_elements_full": N.tolist(),
                                 "fit_residual_dof": len(h) - 3, "qoi": {}}
        for k, lab, c, mk in SER:
            f = np.array([M[v][t][k] for t in tags])
            kind, p, se, lim, band, d = classify(h, f)
            o["qoi"][k] = dict(kind=kind, values=f.tolist(), successive_changes=d.tolist(), limit=lim, band=band,
                               p=float(p[2]), p_se=float(se[2]), f_inf_fit=float(p[0]), f_inf_se=float(se[0]),
                               normalised=(f / lim).tolist() if lim else None)
            print(f"  {v:8s} {k:10s} {kind:14s} {np.round(f, 6)} changes {np.round(d, 6)} fit p {p[2]:.2f} +- "
                  f"{se[2]:.2f}" + (f" -> limit {lim:.6f} +- {band:.6f}, f/f_inf {np.round(f / lim, 4)}" if lim else ""))
            if kind == "flat":                                   # no mesh trend: markers / mean, band +- sd, no curve
                ax.axhspan(1 - band / lim, 1 + band / lim, color=c, alpha=0.18, lw=0, zorder=1)
                ax.plot(N, f / lim, mk, mfc="white", mec=c, mew=1.2, ms=6, ls="none", zorder=3)
                kinds.add("flat")
            if kind != "converging":
                continue
            fh = p[0] + p[1] * hh ** p[2]
            ax.fill_between(cN / hh ** 2, (fh - band) / lim, (fh + band) / lim, color=c, alpha=0.18, lw=0, zorder=1)
            ax.plot(cN / hh ** 2, fh / lim, color=c, lw=1.1, zorder=2)
            ax.plot(N, f / lim, mk, color=c, mec="k", mew=0.5, ms=6, ls="none", zorder=3)
        o["apex_err_mm"] = [M[v][t]["apex_err"] for t in tags]
        ax.axhline(1.0, color="0.35", lw=0.9, ls=(0, (4, 2)), zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(cN / 120.0 ** 2, cN / 5.0 ** 2)
        ax.set_xlabel("Number of elements")
        ax.text(0.97, 0.94, title, transform=ax.transAxes, ha="right", va="top", fontsize=9)
        fig.text(pos[i][0] - 12 / W, pos[i][1] + pos[i][3] + 1.5 / H, "(%s)" % "ab"[i], fontsize=11, fontweight="bold")
    axes[0].set_ylabel(r"Normalised measure $f/f_\infty$")
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    hd = [Line2D([], [], color=c, marker=mk, mec="k", mew=0.5, ms=6, lw=1.1, label=lab) for _, lab, c, mk in SER]
    hd += [Line2D([], [], color="0.35", lw=0.9, ls=(0, (4, 2)), label="extrapolated limit"),
           plt.matplotlib.patches.Patch(color="0.6", alpha=0.35, label="uncertainty")]
    if "flat" in kinds:
        hd.insert(3, Line2D([], [], color="0.35", marker="o", mfc="white", mew=1.2, ms=6, lw=0,
                            label=r"no mesh trend ($f_\infty$ = mean)"))
    fig.legend(handles=hd, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.0), handlelength=2.0)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig("%s.%s" % (OUT, ext), **kw)
    with open(OUT + ".json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
