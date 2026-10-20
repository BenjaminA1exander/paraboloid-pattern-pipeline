r"""plot_vares_journal_combined_flat.py -- Fig. 8 of the paper (Sec. 3.4.2): the VaReS cutting pattern of the
paraboloid gore, at full text width.

    top row     (a) sigma_warp(eps_warp, eps_weft)   (b) sigma_weft(eps_warp, eps_weft)   the Chen (2022) law
                (c) convergence of the pattern's compensation field on the nested 200 mm chain against the number of
                    degrees of freedom, with the extrapolated limit f_inf (asymptote_fit.limit on x = N^-1/2)
    bottom row  (d) u_s   (e) u_y   the compensation field of the 12.5 mm rung of that chain (converged pattern minus
                    the least-squares seed of the same mesh, rigid motion removed, element-scale part smoothed out),
                    drawn on the cut gore lying flat (rim left, tip right), in units of the aperture diameter D

Inputs (data root paraboloid_dish_config.TEMP_DIR): the converged patterns of the 200 mm chain
(meshstudy_vares_cstraw_tippolar210c200nestgrade/h{200,...,012}/pattern.npz) and the 100 mm chain's 6.25 mm rung
(meshstudy_vares_cstraw_tippolar210nestgrade/h006/pattern.npz).  The least-squares seed of the 12.5 mm rung is
computed once with the solver's own flattening call and cached next to the pattern (h012/lsq_seed.npz).
Measures: vares_pattern_measures (field_rms, lumped_area, mls1, rigid_fit_area); fit: asymptote_fit.limit;
panels (a, b): plot_chen_response_surface_journal.panel.

USAGE (base env)   python plot_vares_journal_combined_flat.py
OUT   <data>/results/figures/fig_vares_journal_combined_flat.{pdf,png} (+ _preview.png)
ENV   FIG_FONT_SCALE (1.3), FIG_H_MM (99), FIG_H3D (56), FIG_TOP (43.5), FIG_GH (15.7): font scale and layout in mm
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("MESHSTUDY_TARGET", "cstraw")

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import plot_chen_response_surface_journal as RS                    # noqa: E402  (sets the journal rcParams)
import matplotlib.pyplot as plt                                    # noqa: E402
import matplotlib.tri as mtri                                      # noqa: E402
from matplotlib.colors import TwoSlopeNorm                         # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.ticker import FixedFormatter                       # noqa: E402

import paraboloid_dish_config as CFG                               # noqa: E402
import MeshStudy_paraboloid_vares_ckpt as CK                       # noqa: E402
import vares_study as MS                            # noqa: E402
import vares_pattern_measures as VM                                # noqa: E402
import asymptote_fit as AF                                         # noqa: E402

OUTD = os.path.join(CFG.TEMP_DIR, "results", "figures")
OUT = os.path.join(OUTD, "fig_vares_journal_combined_flat")
W_MM, H_MM = 190.0, float(os.environ.get("FIG_H_MM", "99"))
# one factor on every font size (rcParams set by RS + the constants here); FONT_SCALE = 1 gives 7 pt ticks at 190 mm
FONT_SCALE = float(os.environ.get("FIG_FONT_SCALE", "1.3"))
FS_TICK, FS_PANEL, FS_CLABEL = 7.0 * FONT_SCALE, 9.0 * FONT_SCALE, 6.0 * FONT_SCALE
plt.rcParams.update({k: plt.rcParams[k] * FONT_SCALE for k in
                     ("font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")
                     if isinstance(plt.rcParams[k], (int, float))})
D_MM = float(CFG.DIAMETER_MM)       # lengths in (c), (d), (e) over the dish aperture diameter; (a), (b) stay in MPa
C_COL = "k"                         # panel (c) in black: blue and orange already mean warp and weft in (a), (b)
CHAIN, CHAIN_RUNGS = "tippolar210c200nest", (200.0, 100.0, 50.0, 25.0, 12.5)      # the 200 mm chain (Table 4)
EXTRA, EXTRA_RUNGS = "tippolar210nest", (6.25,)                                   # the 100 mm chain's finest rung


def gore_panel_h(ax, cax, Xs, T, f, x0, step, cticks, lab, show_ylab, scale=None):
    tri = mtri.Triangulation((Xs[:, 0] - x0) / D_MM, Xs[:, 1] / D_MM, T)   # s/D along x, rim left
    lim = float(np.abs(f).max())
    cf = ax.tricontourf(tri, f, levels=np.linspace(-lim, lim, 21), cmap="RdBu_r",
                        norm=TwoSlopeNorm(0.0, -lim, lim))
    for pc in cf.collections:                  # no hairline seams between filled levels in PDF viewers
        pc.set_edgecolor("face")
        pc.set_linewidth(0.15)
    lv = np.arange(-np.floor(lim / step) * step, lim, step)
    cs = ax.tricontour(tri, f, levels=lv, colors="k", linewidths=0.4)
    lab_lv = lv[::2] if len(lv) > 9 else lv
    tl = ax.clabel(cs, levels=lab_lv, fmt=lambda v: ("%g" % v).replace("-", "−"),      # true minus sign
                   fontsize=FS_CLABEL, inline_spacing=1)
    for t in tl:                               # horizontal labels: on near-vertical contours the minus sign vanished
        t.set_rotation(0)
    ax.set_aspect("equal")
    ax.set_xlim(-20.0 / D_MM, 2900.0 / D_MM)
    ax.set_ylim(-360.0 / D_MM, 360.0 / D_MM)
    ax.set_yticks([-0.06, 0, 0.06])
    ax.set_yticklabels(["−0.06", "0", "0.06"])
    if show_ylab:
        ax.set_ylabel("$y/D$", labelpad=1)
    ax.set_xlabel("$s/D$", labelpad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    cb = plt.colorbar(cf, cax=cax, ticks=cticks)
    cb.set_label(lab, fontsize=FS_TICK, labelpad=2)        # beside the bar: a title wider than the bar hit the gore
    cb.ax.yaxis.set_major_formatter(FixedFormatter([("%g" % t).replace("-", "−") for t in cticks]))
    cb.ax.tick_params(labelsize=FS_TICK - 0.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    if scale:                                  # scale factor above the bar, as matplotlib's offset text
        cb.ax.text(0.0, 1.0, scale, transform=cb.ax.transAxes, ha="left", va="bottom", fontsize=FS_TICK - 0.5)


def chain_dofs(grading, rungs):
    """N = 2 n_nodes - 3 of each rung's pattern.npz: two in-plane coordinates per node less the solver's 3-dof
    rigid pin (build_pin_dof_set_geom; equals len(free) in every setup.npz)."""
    CK.configure("cstraw", grading)
    return np.array([2 * len(np.load(MS.rung_npz(hr))["X_leaf"]) - 3 for hr in rungs], float)


def asymptote_panel(ax):
    """Panel (c): whole-gore rms of the compensation field on the 200 mm chain plus the 100 mm chain's 6.25 mm
    rung, against the degrees of freedom N (log scale); fit f_inf + a N^-q (asymptote_fit.limit on x = N^-1/2,
    q = p/2) to the chain's rungs only."""
    rows = [r for r in VM.load_rungs(CHAIN, CHAIN_RUNGS) if r["ok"]]
    xtra = [r for r in VM.load_rungs(EXTRA, EXTRA_RUNGS) if r["ok"]]
    if len(rows) != len(CHAIN_RUNGS) or len(xtra) != len(EXTRA_RUNGS):
        raise SystemExit("not every rung is converged")
    y = np.array([r["field"][0] for r in rows])                   # whole gore
    yx = np.array([r["field"][0] for r in xtra])
    N = chain_dofs(CHAIN, CHAIN_RUNGS)
    Nx = chain_dofs(EXTRA, EXTRA_RUNGS)
    v, se, lim, band, flat = AF.limit(N ** -0.5, y)
    fitN = lambda n: v[0] + v[1] * n ** (-0.5 * v[2])               # noqa: E731
    print("[c] DOF axis: N %s + %d ; limit %.4f +- %.4f mm, q %.3f +- %.3f; 100 mm chain h006 %.4f (fit %+.5f)" % (
        N.astype(int), Nx[0], lim, band, v[2] / 2, se[2] / 2, yx[0], yx[0] - fitN(Nx[0])))
    k = 1e3 / D_MM                                                  # mm -> (length / D) x 10^3
    NN = np.logspace(2.4, 5.3, 200)
    ax.axhline(lim * k, color="0.35", lw=1.0, ls=(0, (4, 2)), zorder=1)
    if not flat:
        ax.semilogx(NN, fitN(NN) * k, color=C_COL, lw=1.1, zorder=2)
    ax.semilogx(np.r_[N, Nx], np.r_[y, yx] * k, "D", color=C_COL, mec="k", mew=0.6, ms=4.5, zorder=3)
    ax.set_xlim(NN[0], NN[-1])
    ax.set_ylim(1.501, 1.541)                                     # data range, not the fit's steep N -> 0 end
    ax.set_yticks([1.505, 1.515, 1.525, 1.535])
    ax.yaxis.set_major_formatter(FixedFormatter(["1.505", "1.515", "1.525", "1.535"]))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    ax.set_xlabel(r"Degrees of freedom $N$")
    ax.set_ylabel(r"$\Delta_{\mathrm{rms}}/D$ ($\times 10^{-3}$)")        # rms shape change over D
    handles = [Line2D([], [], color=C_COL, marker="D", mec="k", mew=0.6, ms=4.5, lw=0, label="nested chain"),
               Line2D([], [], color="0.35", lw=1.0, ls=(0, (4, 2)), label=r"limit $f_\infty$")]
    ax.legend(handles=handles, loc="upper right", fontsize=FS_TICK, handlelength=1.4, handletextpad=0.4,
              labelspacing=0.35, borderaxespad=0.2)


def deviation_field():
    """Panels (d, e): the converged 12.5 mm pattern of the chain minus the least-squares seed of the same mesh,
    area-weighted rigid fit removed, element-scale part smoothed out (mls1, sigma = SMOOTH_SIGMA h_eff)."""
    d = os.path.join(CK.study_root(os.environ["MESHSTUDY_TARGET"], CHAIN), "h012")
    print("[dev] (d, e) from %s" % d)
    a = dict(np.load(os.path.join(d, "pattern.npz"), allow_pickle=True))
    assert bool(a["converged"])
    T = a["elements"]
    sp = os.path.join(d, "lsq_seed.npz")
    if os.path.exists(sp):
        s = np.load(sp)
        assert np.array_equal(s["elements"], T)
        Xs = s["X_lsq"]
    else:
        from vares_flatten import least_squares_edge_flatten   # the solver's own seed call
        Xs = least_squares_edge_flatten(a["X_spatial"], T, init_2d=a["X_leaf"][:, :2],
                                        anchor_to=a["X_leaf"][:, :2])[:, :2]
        np.savez(sp, X_lsq=Xs, elements=T, note="lsq_edge seed of the 12.5 mm mesh "
                 "(plot_vares_journal_combined_flat.py); the run itself was warm-started")
    w, ae = VM.lumped_area(Xs, T)
    u = VM.rigid_fit_area(a["X_converged"][:, :2], Xs, w) - Xs
    heff = float(np.sqrt(4 * ae.mean() / np.sqrt(3)))
    us = VM.mls1(Xs, u, w, VM.SMOOTH_SIGMA * heff)
    x0 = float(a["s_lo_mm"]) - float(a["s_mid_mm"])
    return Xs, T, us, x0


MIN_GAP_MM = 1.5          # required clearance between any two figure elements


def report_clearance(fig):
    """Spacing check: the rendered bounding boxes of every element (each axes with its tick labels, axis labels and
    legend; 3-D axes as a whole; colour bars; figure texts; the figure legend), every pair closer than MIN_GAP_MM
    printed, and anything outside the page flagged."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    mm = 25.4 / fig.dpi
    boxes = []
    for i, ax in enumerate(fig.axes):
        if ax.name == "3d":
            boxes.append(("3d axes %d" % i, ax.get_tightbbox(r)))
        else:
            boxes.append(("axes %d (%s)" % (i, ax.get_ylabel() or ax.get_xlabel() or "cbar"), ax.get_tightbbox(r)))
    for t in fig.texts:
        boxes.append(("text %r" % t.get_text()[:18], t.get_window_extent(r)))
    for lg in fig.legends:
        boxes.append(("figure legend", lg.get_window_extent(r)))
    W, H = fig.bbox.width, fig.bbox.height
    bad = 0
    for n, b in boxes:
        if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H:
            print("[space] OUTSIDE PAGE: %s" % n); bad += 1
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (na, a), (nb, b) = boxes[i], boxes[j]
            gx = max(b.x0 - a.x1, a.x0 - b.x1) * mm
            gy = max(b.y0 - a.y1, a.y0 - b.y1) * mm
            gap = max(gx, gy)                                  # < 0 on both axes = overlap
            if gap < MIN_GAP_MM:
                print("[space] %-28s <-> %-28s gap %.1f mm" % (na, nb, gap)); bad += 1
    print("[space] %d clearance problem(s)" % bad)
    return bad


def main():
    os.makedirs(OUTD, exist_ok=True)
    e = np.linspace(RS.EPS_MIN, RS.EPS_MAX, RS.N_FINE)
    EX, EY, SX, SY, BX, BY = RS.evaluate(e)
    Xs, T, us, x0 = deviation_field()
    print("[dev] |u_s| max %.2f mm, |u_y| max %.2f mm" % (np.abs(us[:, 0]).max(), np.abs(us[:, 1]).max()))

    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    F = lambda x, y, w, h: [x / W_MM, y / H_MM, w / W_MM, h / H_MM]   # mm from bottom-left  # noqa: E731

    # panel letters only; what each panel shows goes in the caption, so the stresses are z-axis labels
    H3D = float(os.environ.get("FIG_H3D", "56"))
    TOP = float(os.environ.get("FIG_TOP", "43.5"))              # bottom edge of the top-row axes (mm)
    axa = fig.add_axes(F(13, TOP, 53, H3D), projection="3d")
    axb = fig.add_axes(F(79.5, TOP, 53, H3D), projection="3d")
    RS.panel(axa, EX, EY, SX, BX, RS.CMAP_W, RS.C_W, "")
    RS.panel(axb, EX, EY, SY, BY, RS.CMAP_F, RS.C_F, "")
    # z-axis labels as rotated text next to the z ticks: mplot3d puts its own z label far outside the ticks
    for x_mm, lab in ((4.0, r"$\sigma_{\mathrm{warp}}$ (MPa)"), (70.8, r"$\sigma_{\mathrm{weft}}$ (MPa)")):
        fig.text(x_mm / W_MM, (TOP + 34) / H_MM, lab, rotation=90, ha="center", va="center",
                 fontsize=plt.rcParams["axes.labelsize"])
    fig.legend(handles=[
        Line2D([], [], color="0.05", lw=0.9, ls=(0, (1, 1.3)), label=r"calibration limit, $\varepsilon = 3$ %"),
        Line2D([], [], color="0.05", lw=1.0, ls=(0, (3.5, 1.8)), label="branch boundary of the law")],
        loc="center", bbox_to_anchor=(69 / W_MM, (TOP - 3.6) / H_MM), ncol=2, fontsize=FS_TICK,
        handlelength=2.2, columnspacing=2.0)
    axc = fig.add_axes(F(149.5, TOP + 15.5, 37.5, 30))      # level with the material surfaces of (a), (b)
    asymptote_panel(axc)

    # bottom row: gores lying flat, side by side; gh sets the gore height, the width follows the aspect
    gh = float(os.environ.get("FIG_GH", "15.7"))
    gw = gh * 2.92 / 0.72
    xe = 106 + (gw - 15.0 * 2.92 / 0.72)
    yb = 11.3                                                  # bottom edge of the gore axes
    axd = fig.add_axes(F(16.5, yb, gw, gh))
    caxd = fig.add_axes(F(16.5 + gw + 2.0, yb, 1.8, gh))        # bar as tall as the gore
    axe = fig.add_axes(F(xe, yb, gw, gh))
    caxe = fig.add_axes(F(xe + gw + 2.0, yb, 1.8, gh))
    k = 1e3 / D_MM                                             # u / D in units of 10^-3, factor printed above each bar
    sc = r"$\times 10^{-3}$"
    gore_panel_h(axd, caxd, Xs, T, us[:, 0] * k, x0, 0.4, [-2, 0, 2], r"$u_s/D$", True, sc)
    gore_panel_h(axe, caxe, Xs, T, us[:, 1] * k, x0, 0.2, [-0.6, 0, 0.6], r"$u_y/D$", False, sc)
    y_top, y_bot = TOP + 50, yb + gh + 3.5                    # letters just above the surfaces
    heads = (("(a)", 1.2, y_top), ("(b)", 67.5, y_top), ("(c)", 134.5, y_top), ("(d)", 1.2, y_bot), ("(e)", xe - 12.0, y_bot))
    for s, x, y in heads:
        fig.text(x / W_MM, y / H_MM, s, fontsize=FS_PANEL)

    report_clearance(fig)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig("%s.%s" % (OUT, ext), **kw)
    fig.savefig(OUT + "_preview.png", dpi=150)
    print("[out] %s.{pdf,png}" % OUT)


if __name__ == "__main__":
    main()
