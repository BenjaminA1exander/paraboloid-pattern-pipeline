r"""plot_meshstudy_paraboloid_vares_convergence_journal.py -- journal version of
fig_meshstudy_paraboloid_vares_convergence_cst: same four series, same metrics, presentation only.

Metrics (plot_meshstudy_paraboloid_vares_convergence.metrics, unchanged), after the best 2-D rigid fit:
    seam offset   mean |outward seam offset| over both seams (mm)
    pattern map   rms |phi_h - Q phi| over the gore interior (mm)
Filled, solid: against the finest rung (h = 12.5 mm).  Open, dashed: against the next finer rung.
Orders: least-squares slope over all points, as in the original.

Changes: y range fitted to the data (was 10x empty above it); plain-number x ticks; Okabe-Ito colours
plus distinct markers and line styles (greyscale safe); slope-2 triangle instead of a free dotted
line; production line labelled clear of the data; compact legend.  Metrics are cached in
temp/meshstudy_vares_cst/convergence_metrics.npz (delete it to recompute).

USAGE (base anaconda env)
    python plot_meshstudy_paraboloid_vares_convergence_journal.py
Outputs:
    fig_meshstudy_paraboloid_vares_convergence_cst_journal.{pdf,png}
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MESHSTUDY_TARGET", "cst")

import MeshStudy_paraboloid_vares_ckpt as CK                     # noqa: E402
import MeshStudy_paraboloid_vares as MS                           # noqa: E402
import plot_meshstudy_paraboloid_vares_shape as SH                # noqa: E402
import plot_meshstudy_paraboloid_vares_convergence as CV          # noqa: E402


def width_change(r, q):
    """rms seam WIDTH change of rung r against rung q: the mean of the two outward seam offsets.  Every
    rigid motion moves the two seams of the mirror-symmetric gore by equal and opposite amounts, so this
    is independent of the rigid fit (which is chosen on the interior map and leaves 2-3 um of seam
    misalignment -- the cause of the non-monotone mean|offset| at the fine end)."""
    SH.deviations([r, q])
    sym = 0.5 * (r["off_p"] + r["off_m"])
    return float(np.sqrt(np.mean(sym ** 2)))


def compute():
    CK.configure()
    cache = os.path.join(MS.ROOT, "convergence_metrics.npz")
    hs = CK.ladder()
    if os.path.exists(cache):
        c = dict(np.load(cache))
        if list(c["h_nom"]) == hs[:-1] and "ref_width" in c:
            return c
    rows = [SH.rung(h) for h in hs]
    for r in rows:
        r["h_eff"] = CV.h_eff(r["h"])
    out = {k: [] for k in ("ref_edge", "ref_map", "suc_edge", "suc_map", "ref_width", "suc_width")}
    for i, r in enumerate(rows[:-1]):
        a, b = CV.metrics(r, rows[-1]), CV.metrics(r, rows[i + 1])
        out["ref_edge"].append(a[0]); out["ref_map"].append(a[1])
        out["suc_edge"].append(b[0]); out["suc_map"].append(b[1])
        out["ref_width"].append(width_change(r, rows[-1]))
        out["suc_width"].append(width_change(r, rows[i + 1]))
    c = {k: np.array(v) for k, v in out.items()}
    c["h"] = np.array([r["h_eff"] for r in rows[:-1]])
    c["h_nom"] = np.array(hs[:-1])
    c["h_ref"] = np.array(rows[-1]["h"])
    np.savez(cache, **c)
    return c


def main():
    c = compute()
    h = c["h"]
    od = {k: CV.order(c[k], h) for k in ("ref_edge", "ref_map", "suc_edge", "suc_map")}
    print("h_nom  h_eff   edge/ref  map/ref  edge/next  map/next (mm)")
    for i in range(len(h)):
        print("%5.1f %6.1f  %8.4f %8.4f %9.4f %9.4f" % (c["h_nom"][i], h[i], c["ref_edge"][i],
                                                       c["ref_map"][i], c["suc_edge"][i], c["suc_map"][i]))
    print("orders", {k: round(v, 2) for k, v in od.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter, FixedFormatter
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.minor.top": True, "ytick.minor.right": True,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    C_EDGE, C_MAP = "#0072B2", "#D55E00"             # Okabe-Ito blue / vermillion
    fig, ax = plt.subplots(figsize=(90 / 25.4, 66 / 25.4))
    kw_ref = dict(lw=1.1, ms=4.2, mew=0.0, zorder=4)
    kw_suc = dict(lw=0.8, ls=(0, (4, 2)), ms=4.2, mfc="white", mew=0.9, zorder=5)
    ax.loglog(h, c["ref_map"], "s-", color=C_MAP, **kw_ref)
    ax.loglog(h, c["suc_map"], "s", color=C_MAP, **kw_suc)
    ax.loglog(h, c["ref_edge"], "o-", color=C_EDGE, **kw_ref)
    ax.loglog(h, c["suc_edge"], "o", color=C_EDGE, **kw_suc)

    # slope-2 triangle, in the empty band below the seam series
    x0, x1 = 38.0, 38.0 * 1.6
    y0 = 1.3e-3
    y1 = y0 * (x1 / x0) ** 2
    ax.plot([x0, x1, x1, x0], [y0, y0, y1, y0], color="0.25", lw=0.6)
    ax.text(x1 * 1.03, np.sqrt(y0 * y1), "2", ha="left", va="center", fontsize=7, color="0.25")
    ax.text(np.sqrt(x0 * x1), y0 / 1.12, "1", ha="center", va="top", fontsize=7, color="0.25")

    i_p = list(c["h_nom"]).index(MS.PRODUCTION)
    ax.axvline(h[i_p], color="0.55", lw=0.6, ls=(0, (1, 1.5)), zorder=0)
    ax.text(h[i_p] / 1.03, 7e-4, "production", rotation=90, ha="right", va="bottom", fontsize=6.5,
            color="0.35")

    ax.set_xlim(15, 115)
    ax.set_ylim(5e-4, 1.5)
    xt = [20, 30, 50, 70, 100]
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in xt]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Effective element size $h_{\mathrm{eff}}$ (mm)")
    ax.set_ylabel("Difference in converged pattern (mm)")

    hd = [Line2D([], [], color=C_MAP, marker="s", ls="-", lw=1.1, ms=4.2, mew=0,
                 label=r"pattern map, rms   $p = %.1f$" % od["ref_map"]),
          Line2D([], [], color=C_EDGE, marker="o", ls="-", lw=1.1, ms=4.2, mew=0,
                 label=r"seam offset, mean   $p = %.1f$" % od["ref_edge"]),
          Line2D([], [], color="0.3", marker="o", ls="-", lw=1.1, ms=4.2, mew=0,
                 label="vs finest mesh ($h$ = %g mm)" % float(c["h_ref"])),
          Line2D([], [], color="0.3", marker="o", ls=(0, (4, 2)), lw=0.8, ms=4.2, mfc="white", mew=0.9,
                 label=r"vs next finer mesh   ($p$ = %.1f, %.1f)" % (od["suc_map"], od["suc_edge"]))]
    ax.legend(handles=hd, loc="upper left", handlelength=2.4, borderaxespad=0.4, labelspacing=0.35)
    fig.tight_layout(pad=0.3)
    stem = "fig_meshstudy_paraboloid_vares_convergence_cst_journal"
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(HERE, "%s.%s" % (stem, ext)), **kw)
    fig.savefig(os.path.join(HERE, stem + "_preview.png"), dpi=200)
    print("[out]", os.path.join(HERE, stem + ".{pdf,png}"))


if __name__ == "__main__":
    main()
