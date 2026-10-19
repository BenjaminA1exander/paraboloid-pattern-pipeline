r"""plot_meshstudy_paraboloid_vares_convergence.py -- one-panel convergence plot of the converged
VaReS gore pattern against mesh size (companion to plot_meshstudy_paraboloid_vares_shape.py).

Metrics, both after the best 2-D rigid fit to the comparison pattern:
    edge  mean |outward seam offset| over both seams (mm)
    map   rms |phi_h - Q phi| over the gore interior (mm)
Solid: against the finest converged rung.  Dashed, open: against the next finer rung
(needs no reference, so it cannot converge just because the reference is the finest mesh).
Abscissa h_eff = sqrt(4A/(sqrt3 N_e)) as in MeshStudy_paraboloid_vares.

USAGE (base anaconda env)
    python plot_meshstudy_paraboloid_vares_convergence.py
Outputs:
    fig_meshstudy_paraboloid_vares_convergence.{pdf,png}
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import MeshStudy_paraboloid_vares_ckpt as CK  # noqa: E402
import MeshStudy_paraboloid_vares as MS                        # noqa: E402
import plot_meshstudy_paraboloid_vares_shape as SH            # noqa: E402


def h_eff(h):
    d = np.load(MS.rung_npz(h))
    e = d["X_spatial"][d["elements"]]
    A = 0.5 * np.linalg.norm(np.cross(e[:, 1] - e[:, 0], e[:, 2] - e[:, 0]), axis=1).sum()
    return float(np.sqrt(4.0 * A / (np.sqrt(3.0) * len(e))))


def metrics(r, q):
    """edge and map difference of rung r from rung q (same sample points on both)."""
    SH.deviations([r, q])
    edge = float(np.mean(np.abs(np.r_[r["off_p"], r["off_m"]])))
    return edge, r["e_map_grid"]


def order(e, h):
    return float(np.polyfit(np.log(h), np.log(e), 1)[0])


if __name__ == "__main__":
    CK.configure()                     # $MESHSTUDY_TARGET: linearfit (default) or cst
    rows = [SH.rung(h) for h in CK.ladder()]
    for r in rows:
        r["h_eff"] = h_eff(r["h"])
    ref = rows[-1]
    ref_edge, ref_map, suc_edge, suc_map = [], [], [], []
    for i, r in enumerate(rows[:-1]):
        a = metrics(r, ref)
        b = metrics(r, rows[i + 1])
        ref_edge.append(a[0]); ref_map.append(a[1])
        suc_edge.append(b[0]); suc_map.append(b[1])
    h = np.array([r["h_eff"] for r in rows[:-1]])

    print("mesh   h_eff   edge vs ref   map vs ref   edge vs next   map vs next   (mm)")
    for i, r in enumerate(rows[:-1]):
        print("h%03d  %6.1f   %10.4f   %10.4f   %11.4f   %11.4f"
              % (round(r["h"]), h[i], ref_edge[i], ref_map[i], suc_edge[i], suc_map[i]))
    for name, e in (("edge vs ref", ref_edge), ("map vs ref", ref_map),
                    ("edge vs next", suc_edge), ("map vs next", suc_map)):
        print("  order %-13s %.2f" % (name, order(np.array(e), h)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stixsans", "font.size": MS.FS_LABEL,
        "axes.labelsize": MS.FS_LABEL, "xtick.labelsize": MS.FS_TICK,
        "ytick.labelsize": MS.FS_TICK, "legend.fontsize": MS.FS_LEG, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "grid.linewidth": 0.4,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(90 / 25.4, 70 / 25.4))
    c_edge, c_map = "#1f4e79", "#b35806"
    ax.loglog(h, ref_edge, "o-", color=c_edge, lw=1.2, ms=4, mew=0,
              label="seam edge, vs $h=%g$ (order %.1f)" % (round(ref["h"], 1), order(np.array(ref_edge), h)))
    ax.loglog(h, ref_map, "s-", color=c_map, lw=1.2, ms=4, mew=0,
              label="pattern map, vs $h=%g$ (order %.1f)" % (round(ref["h"], 1), order(np.array(ref_map), h)))
    ax.loglog(h, suc_edge, "o--", color=c_edge, lw=0.9, ms=4, mfc="white",
              label="seam edge, vs next finer (order %.1f)" % order(np.array(suc_edge), h))
    ax.loglog(h, suc_map, "s--", color=c_map, lw=0.9, ms=4, mfc="white",
              label="pattern map, vs next finer (order %.1f)" % order(np.array(suc_map), h))
    # slope-2 guide under the edge series
    hg = np.array([h.min(), h.max()])
    e0 = 0.35 * min(ref_edge[-1], suc_edge[-1])
    ax.loglog(hg, e0 * (hg / h[-1]) ** 2, ":", color="0.45", lw=0.9)
    ax.annotate("slope 2", xy=(hg[1], e0 * (hg[1] / h[-1]) ** 2), xytext=(0, -3),
                textcoords="offset points", ha="right", va="top", fontsize=MS.FS_ANN,
                color="0.45")
    i_p = [r["h"] for r in rows[:-1]].index(MS.PRODUCTION)
    ax.axvline(h[i_p], color="0.6", lw=0.6, ls="-", zorder=0)
    ax.annotate("production", xy=(h[i_p], 0.0), xycoords=("data", "axes fraction"),
                xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom",
                fontsize=MS.FS_ANN, color="0.4")
    ax.set_xlabel(r"Effective element size, $h_{\rm eff}$ (mm)")
    ax.set_ylabel("Pattern difference (mm)")
    ax.set_ylim(top=ax.get_ylim()[1] * 40)
    ax.legend(loc="upper left", fontsize=MS.FS_LEG - 0.5)
    ax.grid(alpha=0.22, which="both")
    fig.tight_layout()
    stem = "fig_meshstudy_paraboloid_vares_convergence"
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        p = os.path.join(HERE, "%s%s.%s" % (stem, CK.SUFFIX, ext))
        fig.savefig(p, **kw)
        print("[figure] saved %s" % p)
