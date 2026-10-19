r"""plot_vares_journal_combined.py -- one double-column journal figure for the VaReS gore study.

    (a) sigma_warp(eps_warp, eps_weft)   (b) sigma_weft(eps_warp, eps_weft)    Chen (2022) law as the VUMAT
    (c) u_s, (d) u_w: converged 12.5 mm VaReS gore minus its least-squares seed (rigid fit removed,
        element-scale part smoothed out), drawn on the cut form
    (e) mesh convergence of the converged pattern (CST target ladder)

Data and helpers come unchanged from plot_chen_response_surface_journal.py,
plot_vares_cst_deviation_from_seed.py and plot_meshstudy_paraboloid_vares_convergence_journal.py.

USAGE (base anaconda env)
    python plot_vares_journal_combined.py
Output: fig_vares_journal_combined.{pdf,png}  (190 mm wide)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MESHSTUDY_TARGET", "cst")

import plot_chen_response_surface_journal as RS                    # noqa: E402  (sets rcParams)
import plot_vares_cst_deviation_from_seed as DV                     # noqa: E402
import plot_meshstudy_paraboloid_vares_convergence_journal as CJ    # noqa: E402
import plot_meshstudy_paraboloid_vares_convergence as CV            # noqa: E402
import MeshStudy_paraboloid_vares as MS                             # noqa: E402

import matplotlib                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import matplotlib.tri as mtri                                       # noqa: E402
from matplotlib.colors import TwoSlopeNorm                          # noqa: E402
from matplotlib.lines import Line2D                                 # noqa: E402
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter  # noqa: E402

OUT = os.path.join(HERE, "fig_vares_journal_combined")
W_MM, H_MM = 190.0, 168.0
FS, FS_TICK, FS_PANEL = 8.0, 7.0, 9.0
C_EDGE, C_MAP = "#0072B2", "#D55E00"


def deviation():
    a = dict(np.load(DV.RUN, allow_pickle=True))
    b = dict(np.load(DV.SEED, allow_pickle=True))
    assert bool(a["converged"]) and np.array_equal(a["elements"], b["elements"])
    T = a["elements"]
    Xs = b["X_initial"][:, :2]
    u = DV.rigid_fit(a["X_converged"][:, :2], Xs) - Xs
    w, ae = DV.lumped_area(Xs, T)
    heff = float(np.sqrt(4 * ae.mean() / np.sqrt(3)))
    us = DV.mls1(Xs, u, w, DV.SMOOTH_SIGMA * heff)
    x0 = float(a["s_lo_mm"]) - float(a["s_mid_mm"])
    return Xs, T, us, x0


def gore_panel(ax, cax, Xs, T, f, x0, step, cticks, lab, ylab):
    tri = mtri.Triangulation(Xs[:, 1] / 1000.0, (Xs[:, 0] - x0) / 1000.0, T)  # rim bottom, apex top
    lim = float(np.abs(f).max())
    cf = ax.tricontourf(tri, f, levels=np.linspace(-lim, lim, 21), cmap="RdBu_r",
                        norm=TwoSlopeNorm(0.0, -lim, lim))
    lv = np.arange(-np.floor(lim / step) * step, lim, step)
    cs = ax.tricontour(tri, f, levels=lv, colors="k", linewidths=0.3)
    ax.clabel(cs, levels=lv[::2] if len(lv) > 9 else lv, fmt="%g", fontsize=5.5, inline_spacing=1)
    ax.set_aspect("equal")
    ax.set_xlim(-0.36, 0.36)
    ax.set_ylim(-0.02, 2.90)
    ax.set_xticks([-0.3, 0, 0.3])
    ax.set_xticklabels(["−0.3", "0", "0.3"])
    ax.set_xlabel("$y$ (m)")
    if ylab:
        ax.set_ylabel("$s$ (m)")
    else:
        ax.tick_params(labelleft=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    cb = plt.colorbar(cf, cax=cax, ticks=cticks)
    cb.ax.set_title(lab, fontsize=FS_TICK, pad=4)
    cb.ax.tick_params(labelsize=FS_TICK - 0.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)


UNIFORM_MAX = 71.0     # nominal h: from 70.7 mm down the seam edge equals h; 200/141/100 are apex-graded


def convergence_panel(ax, c):
    k = c["h_nom"] <= UNIFORM_MAX                                 # uniformly refined rungs only
    h = c["h"][k]
    ax.loglog(h, c["ref_map"][k], "s-", color=C_MAP, lw=1.1, ms=3.8, mew=0, zorder=4)
    ax.loglog(h, c["ref_width"][k], "o-", color=C_EDGE, lw=1.1, ms=3.8, mew=0, zorder=4)
    ax.loglog(h, c["suc_map"][k], ":", color=C_MAP, lw=1.2, zorder=3)
    ax.loglog(h, c["suc_width"][k], ":", color=C_EDGE, lw=1.2, zorder=3)
    x0, x1, y0 = 33.0, 33.0 * 1.6, 1.2e-3                        # slope-2 triangle
    y1 = y0 * (x1 / x0) ** 2
    ax.plot([x0, x1, x1, x0], [y0, y0, y1, y0], color="0.3", lw=0.6)
    ax.text(x1 * 1.04, np.sqrt(y0 * y1), "2", ha="left", va="center", fontsize=FS_TICK, color="0.3")
    ax.text(np.sqrt(x0 * x1), y0 / 1.12, "1", ha="center", va="top", fontsize=FS_TICK, color="0.3")
    ax.set_xlim(15, 70)
    ax.set_ylim(5e-4, 4.0)
    xt = [20, 30, 40, 60]
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in xt]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.set_xlabel(r"$h_{\mathrm{eff}}$ (mm)")
    ax.set_ylabel("Pattern difference (mm)")
    ax.legend(handles=[
        Line2D([], [], color=C_MAP, marker="s", lw=1.1, ms=3.8, mew=0, label="pattern map (rms)"),
        Line2D([], [], color=C_EDGE, marker="o", lw=1.1, ms=3.8, mew=0, label="seam width change (rms)"),
        Line2D([], [], color="0.45", lw=1.1, label="vs finest mesh"),
        Line2D([], [], color="0.45", lw=1.2, ls=":", label="vs next finer mesh")],
        loc="upper left", fontsize=FS_TICK, handlelength=2.2, labelspacing=0.3, borderaxespad=0.3)


def main():
    e = np.linspace(RS.EPS_MIN, RS.EPS_MAX, RS.N_FINE)
    EX, EY, SX, SY, BX, BY = RS.evaluate(e)
    Xs, T, us, x0 = deviation()
    c = CJ.compute()

    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    F = lambda x, y, w, h: [x / W_MM, y / H_MM, w / W_MM, h / H_MM]   # positions in mm from bottom-left

    axa = fig.add_axes(F(9, 84, 89, 84), projection="3d")
    axb = fig.add_axes(F(101, 84, 89, 84), projection="3d")
    RS.panel(axa, EX, EY, SX, BX, RS.CMAP_W, RS.C_W, r"$\sigma_{\mathrm{warp}}$ (MPa)")
    RS.panel(axb, EX, EY, SY, BY, RS.CMAP_F, RS.C_F, r"$\sigma_{\mathrm{weft}}$ (MPa)")
    fig.legend(handles=[
        Line2D([], [], color="0.05", lw=0.9, ls=(0, (1, 1.3)), label=r"$\varepsilon = 3$ % (calibration limit)"),
        Line2D([], [], color="0.05", lw=1.0, ls=(0, (3.5, 1.8)), label="branch boundary")],
        loc="center", bbox_to_anchor=(0.5, 85.5 / H_MM), ncol=2, fontsize=FS_TICK,
        handlelength=2.2, columnspacing=2.5)

    axc = fig.add_axes(F(13, 12, 16, 64))                         # 16 mm = the gore's own width at this height
    caxc = fig.add_axes(F(31, 18, 2.2, 46))
    axd = fig.add_axes(F(53, 12, 16, 64))
    caxd = fig.add_axes(F(71, 18, 2.2, 46))
    gore_panel(axc, caxc, Xs, T, us[:, 0], x0, 2.0, [-12, -8, -4, 0, 4, 8, 12], "$u_s$ (mm)", True)
    gore_panel(axd, caxd, Xs, T, us[:, 1], x0, 1.0, [-3, -2, -1, 0, 1, 2, 3], "$u_y$ (mm)", False)

    axe = fig.add_axes(F(104, 12, 83, 62))
    convergence_panel(axe, c)

    for s, (x, y) in zip("abcde", ((4, 161), (99, 161), (2, 74), (42, 74), (90, 74))):
        fig.text(x / W_MM, y / H_MM, "(%s)" % s, fontsize=FS_PANEL)

    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig("%s.%s" % (OUT, ext), **kw)
    fig.savefig(OUT + "_preview.png", dpi=150)
    print("[out] %s.{pdf,png}" % OUT)


if __name__ == "__main__":
    main()
