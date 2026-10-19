r"""plot_vares_convergence_tip_split.py -- mesh convergence of the converged VaReS cutting pattern, with the
gore split into the region away from the tip and the tip zone.

near-tip area separately so their errors can be examined separately.

Ladder: the tip-graded raw-CST rungs (MESHSTUDY_TARGET=cstraw, MESHSTUDY_GRADING=tiptest, 70.7 -> 12.5 mm).
Sampling: plot_meshstudy_paraboloid_vares_shape.rung (unchanged) -- 160 stations along the gore x 21
fractions of the local half-width on the common developed leaf, plus both seams at every station.

Metric: SUCCESSIVE differences d_k = |pattern_k - pattern_{k+1}|, reference-free (with e = C h^p,
d_k = C (1 - r^-p) h_k^p has the same slope p; differences against the finest rung bend down near it).
Each pair is rigidly aligned on the WHOLE interior grid (SH.deviations), then the norms are taken per region:
  pattern map  rms |difference| over the interior grid, area-weighted (weight = local half-width)
  seam width   rms of the symmetric seam offset (mean of the two outward offsets; rigid-fit independent)
Regions by distance from the tip along the gore, as a fraction of the gore length L:
  away from tip  >= TIP_FRAC * L          tip zone  < TIP_FRAC * L
Plotted against h_eff of the coarser mesh of each pair; p = least-squares slope, +- 1 standard error.
OUT  fig_vares_convergence_tip_split.{pdf,png}, _preview.png, and the numbers printed.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MESHSTUDY_TARGET", "cstraw")
os.environ.setdefault("MESHSTUDY_GRADING", "tiptest")

import MeshStudy_paraboloid_vares_ckpt as CK                      # noqa: E402
import plot_meshstudy_paraboloid_vares_shape as SH                # noqa: E402
import plot_meshstudy_paraboloid_vares_convergence as CV          # noqa: E402
# plot_vares_journal_combined_flat (journal rcParams, sizes) is imported inside main() only: importing it here
# would re-apply its FONT_SCALE to the rcParams when that script itself calls split_metrics()

TIP_FRAC = 0.10
OUT = os.path.join(HERE, "fig_vares_convergence_tip_split")


def fit(h, y):
    X = np.c_[np.ones(len(h)), np.log(h)]
    b = np.linalg.lstsq(X, np.log(y), rcond=None)[0]
    r = np.log(y) - X @ b
    se = np.sqrt((r @ r) / max(len(y) - 2, 1) * np.linalg.inv(X.T @ X)[1, 1])
    return b[1], se


def split_metrics():
    """(h_eff of the coarser mesh of each pair, {region: {"map": [...], "width": [...]}}, tip-zone length mm)
    -- the computation of main(), reusable by plot_vares_journal_combined_flat.py panel (c)."""
    CK.configure()
    rows = [SH.rung(h) for h in CK.ladder()]
    for r in rows:
        r["h_eff"] = CV.h_eff(r["h"])
    ne = len(SH.ETA)
    xs = rows[0]["xs"]
    L = xs.max() - xs.min()
    tip = (xs.max() - xs) < TIP_FRAC * L
    w = np.repeat(rows[0]["hw"], ne)
    reg = {"away from tip": ~tip, "tip zone": tip}
    out = {k: {"map": [], "width": []} for k in reg}
    for a, b in zip(rows[:-1], rows[1:]):
        SH.deviations([a, b])
        sym = 0.5 * (a["off_p"] + a["off_m"])
        for k, m in reg.items():
            mg = np.repeat(m, ne)
            out[k]["map"].append(float(np.sqrt((w[mg] * a["dev"][mg] ** 2).sum() / w[mg].sum())))
            out[k]["width"].append(float(np.sqrt(np.mean(sym[m] ** 2))))
    return np.array([r["h_eff"] for r in rows[:-1]]), out, TIP_FRAC * L


def main():
    import plot_vares_journal_combined_flat as FL
    from plot_vares_journal_combined_flat import plt, Line2D
    CK.configure()
    hs = CK.ladder()
    rows = [SH.rung(h) for h in hs]
    for r in rows:
        r["h_eff"] = CV.h_eff(r["h"])
    ns, ne = rows[0]["ns"], len(SH.ETA)
    xs = rows[0]["xs"]
    L = xs.max() - xs.min()
    dist_tip = xs.max() - xs                                   # station distance from the tip (mm)
    tip = dist_tip < TIP_FRAC * L
    w = np.repeat(rows[0]["hw"], ne)                           # area weight of the grid points
    reg = {"away from tip": ~tip, "tip zone": tip}
    out = {k: {"map": [], "width": []} for k in reg}
    for a, b in zip(rows[:-1], rows[1:]):
        SH.deviations([a, b])                                  # a against b (b is the "reference" here)
        dev = a["dev"]                                         # per interior grid point
        sym = 0.5 * (a["off_p"] + a["off_m"])                  # per station
        for k, m in reg.items():
            mg = np.repeat(m, ne)
            out[k]["map"].append(float(np.sqrt((w[mg] * dev[mg] ** 2).sum() / w[mg].sum())))
            out[k]["width"].append(float(np.sqrt(np.mean(sym[m] ** 2))))
    h = np.array([r["h_eff"] for r in rows[:-1]])

    print("pairs (coarser h_nom):", [r["h"] for r in rows[:-1]], " h_eff:", np.round(h, 1))
    print("tip zone = last %.0f %% of L = %.0f mm" % (100 * TIP_FRAC, TIP_FRAC * L))
    fits = {}
    for k in reg:
        for q in ("map", "width"):
            y = np.array(out[k][q])
            p, se = fit(h, y)
            fits[(k, q)] = (p, se)
            loc = np.log(y[:-1] / y[1:]) / np.log(h[:-1] / h[1:])
            print("%-14s %-6s d = %s mm   p = %.2f +- %.2f   local %s" % (
                k, q, np.array2string(y, precision=5), p, se, np.round(loc, 2)))

    st = {"map": dict(color="0.05", marker="s", ls="-", mfc="0.05", lab="pattern map"),
          "width": dict(color="0.45", marker="o", ls=(0, (4, 1.6)), mfc="white", lab="seam width")}
    W_MM, H_MM = 140.0, 66.0
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    F = lambda x, y, w_, h_: [x / W_MM, y / H_MM, w_ / W_MM, h_ / H_MM]
    axes = [fig.add_axes(F(18, 12, 48, 44)), fig.add_axes(F(88, 12, 48, 44))]
    for ax, k, letter in zip(axes, reg, ("(a)", "(b)")):
        handles = []
        for q in ("map", "width"):
            s = st[q]
            ax.loglog(h, out[k][q], color=s["color"], marker=s["marker"], ls=s["ls"], mfc=s["mfc"],
                      lw=1.1, ms=4.2, mew=0.9)
            p, se = fits[(k, q)]
            handles.append(Line2D([], [], color=s["color"], marker=s["marker"], ls=s["ls"], mfc=s["mfc"], lw=1.1,
                                  ms=4.2, mew=0.9, label="%s, $p = %.1f \\pm %.1f$" % (s["lab"], p, se)))
        ax.set_xlim(14, 75)
        ax.set_ylim(3e-4, 6.0)          # headroom: the legend sits above the data
        xt = [20, 30, 40, 60]
        ax.xaxis.set_major_locator(FL.CB.FixedLocator(xt))
        ax.xaxis.set_major_formatter(FL.CB.FixedFormatter([str(t) for t in xt]))
        ax.xaxis.set_minor_formatter(FL.CB.NullFormatter())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(True)
        ax.set_xlabel(r"Element size $h_{\mathrm{eff}}$ (mm)")
        ax.legend(handles=handles, loc="upper left", fontsize=FL.FS_TICK, handlelength=2.0, handletextpad=0.5,
                  labelspacing=0.35, borderaxespad=0.4)
        fig.text((ax.get_position().x0 * W_MM - 3) / W_MM, 58.5 / H_MM, letter, fontsize=FL.FS_PANEL, ha="right")
    axes[0].set_ylabel("Difference to next finer mesh (mm)")
    FL.report_clearance(fig)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig("%s.%s" % (OUT, ext), **kw)
    fig.savefig(OUT + "_preview.png", dpi=150)
    print("[out] %s.{pdf,png}" % OUT)


if __name__ == "__main__":
    main()
