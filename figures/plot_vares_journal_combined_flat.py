r"""plot_vares_journal_combined_flat.py -- landscape (flat) layout of fig_vares_journal_combined, for a
figure placed automatically at full text width.

    top row     (a) sigma_warp   (b) sigma_weft   (c) mesh convergence
    bottom row  (d) u_s          (e) u_y          on the cut gore, lying horizontally (rim left, tip right)

Same data and helpers as plot_vares_journal_combined.py; only the arrangement differs.  Fonts are
>= 7 pt at 190 mm so they stay >= 6 pt if the page scales the figure down to a ~160 mm text width.

: updated to the NEW results -- the tip-graded VaReS ladder, raw CST target of gore 0
(--target cstraw --grading tiptest, size = h*min(1, 0.08+0.006 d), 70.7 -> 12.5 mm, all converged to 1e-7).
Panel (c) is the same metric code on that ladder (CK.configure via MESHSTUDY_TARGET / MESHSTUDY_GRADING;
cache convergence_metrics.npz in its root).  Panels (d, e) are its converged 12.5 mm pattern minus the
least-squares seed ON THE SAME GRADED MESH (the graded run itself was warm-started from the uniform rung, so
the lsq seed is computed here with the solver's own least_squares_edge_flatten call and cached as
h012/lsq_seed.npz).  The previous figure (full-tip default mesher, averaged CST) was moved aside to
fig_vares_journal_combined_flat_fulltip_cst_superseded.*; reproduce it with
MESHSTUDY_TARGET=cst MESHSTUDY_GRADING=none.

USAGE (base anaconda env)
    python plot_vares_journal_combined_flat.py
Output: fig_vares_journal_combined_flat.{pdf,png}
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MESHSTUDY_TARGET", "cstraw")      # before CB, whose setdefault would pick "cst"
os.environ.setdefault("MESHSTUDY_GRADING", "tiptest")
GRADED = os.environ["MESHSTUDY_GRADING"] != "none"

import plot_vares_journal_combined as CB                           # noqa: E402
from plot_vares_journal_combined import RS, CJ, plt, mtri, TwoSlopeNorm, Line2D  # noqa: E402

OUT = os.path.join(HERE, "fig_vares_journal_combined_flat")
W_MM, H_MM = 190.0, float(os.environ.get("FIG_H_MM", "99"))   # 122 -> 99 mm
# all text bigger.  One factor on every font size (rcParams set by RS + the constants here);
# FONT_SCALE = 1 reproduces the previous sizes.
FONT_SCALE = float(os.environ.get("FIG_FONT_SCALE", "1.3"))
FS_TICK, FS_PANEL, FS_CLABEL = 7.0 * FONT_SCALE, 9.0 * FONT_SCALE, 6.0 * FONT_SCALE
plt.rcParams.update({k: plt.rcParams[k] * FONT_SCALE for k in
                     ("font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")
                     if isinstance(plt.rcParams[k], (int, float))})
# everything dimensionless -- lengths in (c), (d), (e) over the dish aperture diameter D
# (paraboloid_dish_config.DIAMETER_MM = 5000 mm); (a), (b) stay in MPa
import paraboloid_dish_config as CFG                               # noqa: E402
D_MM = float(CFG.DIAMETER_MM)
# panel (c) in black -- orange read as belonging to the weft surface (b), blue to the warp (a).
# Previous orange version: fig_vares_journal_combined_flat_orangec_superseded.*
C_COL = "k"


def gore_panel_h(ax, cax, Xs, T, f, x0, step, cticks, lab, show_ylab, scale=None, one_sided=False):
    tri = mtri.Triangulation((Xs[:, 0] - x0) / D_MM, Xs[:, 1] / D_MM, T)   # s/D along x, rim left
    lim = float(np.abs(f).max())
    if one_sided:
        # inward displacement, >= 0 except round-off-sized values on the centreline (where sign(y)
        # flips).  Red half of the same RdBu_r (red = positive, as in (d)); values below 0 take the white end.
        from matplotlib.colors import ListedColormap, Normalize
        cmap = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0.5, 1.0, 256)))
        cmap.set_under(cmap(0.0))
        cf = ax.tricontourf(tri, f, levels=np.r_[min(float(f.min()), 0.0) - 1e-9, np.linspace(0.0, lim, 11)[1:]],
                            cmap=cmap, norm=Normalize(0.0, lim))
    else:
        cf = ax.tricontourf(tri, f, levels=np.linspace(-lim, lim, 21), cmap="RdBu_r",
                            norm=TwoSlopeNorm(0.0, -lim, lim))
    for pc in cf.collections:                  # no hairline seams between filled levels in PDF viewers
        pc.set_edgecolor("face")
        pc.set_linewidth(0.15)
    lv = np.arange(step, lim, step) if one_sided else np.arange(-np.floor(lim / step) * step, lim, step)
    cs = ax.tricontour(tri, f, levels=lv, colors="k", linewidths=0.4)
    # true minus sign in the contour labels (as on the colour bar), not a hyphen
    # one-sided (e): every other level, and not the top ones, which crowd the rim corners
    lab_lv = lv[::2][:-1] if one_sided else (lv[::2] if len(lv) > 9 else lv)
    tl = ax.clabel(cs, levels=lab_lv, fmt=lambda v: ("%g" % v).replace("-", "−"),
                   fontsize=FS_CLABEL, inline_spacing=1)
    # horizontal labels: rotated onto the near-vertical u_s contours the minus sign lay along the
    # contour line and vanished ("−8" read "8")
    for t in tl:
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
    if one_sided:
        from matplotlib.cm import ScalarMappable
        cb = plt.colorbar(ScalarMappable(norm=cf.norm, cmap=cf.cmap), cax=cax, ticks=cticks)
    else:
        cb = plt.colorbar(cf, cax=cax, ticks=cticks)
    # label beside the bar, not above it: a title wider than the bar ran back into the gore panel
    cb.set_label(lab, fontsize=FS_TICK, labelpad=2)
    cb.ax.yaxis.set_major_formatter(CB.FixedFormatter([("%g" % t).replace("-", "−") for t in cticks]))
    cb.ax.tick_params(labelsize=FS_TICK - 0.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    if scale:                                  # scale factor above the bar, as matplotlib's offset text
        cb.ax.text(0.0, 1.0, scale, transform=cb.ax.transAxes, ha="left", va="bottom", fontsize=FS_TICK - 0.5)


def convergence_panel_tip_split(ax):
    """Panel (c), pattern map only, split into the region away from the tip and the tip
    zone (last 10 % of the gore), as SUCCESSIVE differences (each mesh against the next finer one; reference-
    free, the convention of Richardson / GCI studies).  Computation: plot_vares_convergence_tip_split.
    split_metrics (one rigid fit per pair over the whole gore, area-weighted rms per region).  Plotted at h_eff
    of the coarser mesh of each pair.  No reference slope and no fitted order on the panel."""
    import plot_vares_convergence_tip_split as TS
    h, out, _ = TS.split_metrics()
    series = (("away from tip", dict(color="0.05", marker="s", ls="-", mfc="0.05")),
              ("tip zone", dict(color="0.45", marker="o", ls=(0, (4, 1.6)), mfc="white")))
    handles = []
    for key, st in series:
        ax.loglog(h, out[key]["map"], lw=1.1, ms=4.2, mew=0.9, zorder=4, **st)
        handles.append(Line2D([], [], lw=1.1, ms=4.2, mew=0.9, label=key, **st))
    ax.set_xlim(14, 75)
    ax.set_ylim(3e-3, 3.0)
    xt = [20, 30, 40, 60]
    ax.xaxis.set_major_locator(CB.FixedLocator(xt))
    ax.xaxis.set_major_formatter(CB.FixedFormatter([str(t) for t in xt]))
    ax.xaxis.set_minor_formatter(CB.NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    ax.set_xlabel(r"Element size $h_{\mathrm{eff}}$ (mm)")
    ax.set_ylabel("Difference to next finer mesh (mm)")
    ax.legend(handles=handles, loc="upper left", fontsize=FS_TICK, handlelength=2.0, handletextpad=0.5,
              labelspacing=0.35, borderaxespad=0.4)


def asymptote_panel_chain100(ax):
    """Panel (c), panel (b) of fig_vares_chain100_wholegore -- whole-gore rms of the compensation
    field on the nested 100 mm chain vs h_eff, fit f_inf + a h^p, extrapolated limit.
    Same data and styles as plot_vares_chain100_wholegore.py; the legend sits inside the axes (lower left).  No
    uncertainty band."""
    import plot_vares_chain100_asymptote as AS
    rows = [r for r in AS.load() if r["ok"]]
    if len(rows) != len(AS.RUNGS):
        raise SystemExit("not every chain rung is converged")
    h = np.array([r["h"] for r in rows])
    y = np.array([r["field"][0] for r in rows])                   # whole gore
    v, se, lim, band, flat = AS.limit(h, y)
    print("[c] asymptote: limit %.4f +- %.4f mm, p %.2f +- %.2f" % (lim, band, v[2], se[2]))
    col = AS.COL
    hh = np.linspace(0, 85, 200)
    k = 1e3 / D_MM                                                  # mm -> (length / D) x 10^3
    ax.axhline(lim * k, color="0.35", lw=1.0, ls=(0, (4, 2)), zorder=1)
    if not flat:
        ax.plot(hh * k, (v[0] + v[1] * hh ** v[2]) * k, color=col, lw=1.1, zorder=2)
    ax.plot(h * k, y * k, "D", color=col, mec="k", mew=0.6, ms=4.5, zorder=3)
    ax.set_xlim(0, 85 * k)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([1.496, 1.500, 1.504])
    ax.yaxis.set_major_formatter(CB.FixedFormatter(["1.496", "1.500", "1.504"]))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    ax.set_xlabel(r"$h_{\mathrm{eff}}/D$ ($\times 10^{-3}$)")
    ax.set_ylabel(r"$\Delta_{\mathrm{rms}}/D$ ($\times 10^{-3}$)")        # rms shape change over D
    handles = [Line2D([], [], color=col, marker="D", mec="k", mew=0.6, ms=4.5, lw=0, label="nested chain"),
               Line2D([], [], color=col, lw=1.1, label=r"fit $f_\infty + a\,h^p$"),
               Line2D([], [], color="0.35", lw=1.0, ls=(0, (4, 2)), label=r"limit $f_\infty$")]
    ax.legend(handles=handles, loc="lower left", fontsize=FS_TICK, handlelength=1.4, handletextpad=0.4,
              labelspacing=0.35, borderaxespad=0.2)


def asymptote_panel_chain200(ax):
    """Panel (c), specifically, field"): panel (a)
    of fig_vares_chain200_field_asymptote -- whole-gore rms of the compensation field on the nested chain from a 200 mm
    base (200 -> 12.5 mm far field, h_eff 99 -> 6.2 mm) plus the 100 mm chain's converged 6.25 mm rung (h_eff 4.9 mm,
    0.3 um off the fit), one orange series; fit f_inf + a h^p to the 200 mm chain rungs.  Same format as
    asymptote_panel_chain100 (kept: FIG_PANEL_C=chain100 reproduces the previous figure).
    default x axis is the number of
    degrees of freedom N = 2 n_nodes - 3 (two in-plane coordinates per node less the solver's 3-dof rigid pin,
    build_pin_dof_set_geom; equals len(free) in every setup.npz on disk), log scale.  Fit in N directly,
    f_inf + a N^-q (AS.limit on x = N^-1/2, q = p/2).  FIG_C_X=h reproduces the h_eff/D axis."""
    import plot_vares_chain100_asymptote as AS
    import plot_vares_chain200_asymptote as A2
    rows = [r for r in A2.load_grading("tippolar210c200nest", (200.0, 100.0, 50.0, 25.0, 12.5)) if r["ok"]]
    xtra = [r for r in A2.load_grading("tippolar210nest", (6.25,)) if r["ok"]]
    if len(rows) != 5 or len(xtra) != 1:
        raise SystemExit("not every rung is converged")
    h = np.array([r["h"] for r in rows])
    y = np.array([r["field"][0] for r in rows])                   # whole gore
    hx = np.array([r["h"] for r in xtra]); yx = np.array([r["field"][0] for r in xtra])
    if os.environ.get("FIG_C_X", "dof") != "h":
        return _asymptote_panel_chain200_dof(ax, AS, y, yx)
    v, se, lim, band, flat = AS.limit(h, y)
    print("[c] asymptote (200 mm chain): limit %.4f +- %.4f mm, p %.2f +- %.2f; 100 mm chain h006 %.4f (fit %+.5f)" % (
        lim, band, v[2], se[2], yx[0], yx[0] - (v[0] + v[1] * hx[0] ** v[2])))
    col = C_COL
    hmax = 105.0
    hh = np.linspace(0, hmax, 200)
    k = 1e3 / D_MM                                                  # mm -> (length / D) x 10^3
    ax.axhline(lim * k, color="0.35", lw=1.0, ls=(0, (4, 2)), zorder=1)
    if not flat:
        ax.plot(hh * k, (v[0] + v[1] * hh ** v[2]) * k, color=col, lw=1.1, zorder=2)
    ax.plot(np.r_[h, hx] * k, np.r_[y, yx] * k, "D", color=col, mec="k", mew=0.6, ms=4.5, zorder=3)
    ax.set_xlim(0, hmax * k)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([1.505, 1.515, 1.525, 1.535])
    ax.yaxis.set_major_formatter(CB.FixedFormatter(["1.505", "1.515", "1.525", "1.535"]))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    ax.set_xlabel(r"$h_{\mathrm{eff}}/D$ ($\times 10^{-3}$)")
    ax.set_ylabel(r"$\Delta_{\mathrm{rms}}/D$ ($\times 10^{-3}$)")        # rms shape change over D
    handles = [Line2D([], [], color=col, marker="D", mec="k", mew=0.6, ms=4.5, lw=0, label="nested chain"),
               Line2D([], [], color="0.35", lw=1.0, ls=(0, (4, 2)), label=r"limit $f_\infty$")]
    ax.legend(handles=handles, loc="upper left", fontsize=FS_TICK, handlelength=1.4, handletextpad=0.4,
              labelspacing=0.35, borderaxespad=0.2)


def _chain_dofs(AS, grading, rungs):
    """N = 2 n_nodes - 3 of each rung's pattern.npz (see asymptote_panel_chain200)."""
    AS.CK.configure("cstraw", grading)
    return np.array([2 * len(np.load(AS.MS.rung_npz(hr))["X_leaf"]) - 3 for hr in rungs], float)


def _asymptote_panel_chain200_dof(ax, AS, y, yx):
    N = _chain_dofs(AS, "tippolar210c200nest", (200.0, 100.0, 50.0, 25.0, 12.5))
    Nx = _chain_dofs(AS, "tippolar210nest", (6.25,))
    v, se, lim, band, flat = AS.limit(N ** -0.5, y)
    fitN = lambda n: v[0] + v[1] * n ** (-0.5 * v[2])               # noqa: E731
    print("[c] DOF axis: N %s + %d ; limit %.4f +- %.4f mm, q %.3f +- %.3f; 100 mm chain h006 %.4f (fit %+.5f)" % (
        N.astype(int), Nx[0], lim, band, v[2] / 2, se[2] / 2, yx[0], yx[0] - fitN(Nx[0])))
    col = C_COL
    k = 1e3 / D_MM                                                  # mm -> (length / D) x 10^3
    NN = np.logspace(2.4, 5.3, 200)
    ax.axhline(lim * k, color="0.35", lw=1.0, ls=(0, (4, 2)), zorder=1)
    if not flat:
        ax.semilogx(NN, fitN(NN) * k, color=col, lw=1.1, zorder=2)
    ax.semilogx(np.r_[N, Nx], np.r_[y, yx] * k, "D", color=col, mec="k", mew=0.6, ms=4.5, zorder=3)
    ax.set_xlim(NN[0], NN[-1])
    ax.set_ylim(1.501, 1.541)                                     # data range, not the fit's steep N -> 0 end
    ax.set_yticks([1.505, 1.515, 1.525, 1.535])
    ax.yaxis.set_major_formatter(CB.FixedFormatter(["1.505", "1.515", "1.525", "1.535"]))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    ax.set_xlabel(r"Degrees of freedom $N$")
    ax.set_ylabel(r"$\Delta_{\mathrm{rms}}/D$ ($\times 10^{-3}$)")        # rms shape change over D
    handles = [Line2D([], [], color=col, marker="D", mec="k", mew=0.6, ms=4.5, lw=0, label="nested chain"),
               Line2D([], [], color="0.35", lw=1.0, ls=(0, (4, 2)), label=r"limit $f_\infty$")]
    ax.legend(handles=handles, loc="upper right", fontsize=FS_TICK, handlelength=1.4, handletextpad=0.4,
              labelspacing=0.35, borderaxespad=0.2)


def convergence_panel_ref_only(ax, c):
    """CB.convergence_panel without the 'vs next finer mesh' series: only the differences
    against the finest rung.  Same data, styles, axes and slope triangle."""
    # 2026-10-11 presentation pass: neutral black / grey (blue and orange already mean warp / weft in (a), (b));
    # series told apart by marker fill and dash too (greyscale safe); y range fitted to the data; the observed
    # order p (least-squares slope over the points) printed at each line instead of a generic slope triangle.
    k = c["h_nom"] <= CB.UNIFORM_MAX
    h = c["h"][k]
    series = (("ref_map", "pattern map", dict(color="0.05", marker="s", ls="-", mfc="0.05")),
              ("ref_width", "seam width", dict(color="0.45", marker="o", ls=(0, (4, 1.6)), mfc="white")))
    handles = []
    for key, lab, st in series:
        y = c[key][k]
        ax.loglog(h, y, lw=1.1, ms=4.2, mew=0.9, zorder=4, **st)
        handles.append(Line2D([], [], lw=1.1, ms=4.2, mew=0.9, label=lab, **st))
    # no slope triangle: differences are taken against the finest mesh, which steepens
    # the curve near h_ref, and the observed order (~1.7-1.8 from successive rungs) is not second order --
    # a slope-2 reference would suggest a rate that is not present
    ax.set_xlim(14, 75)
    ax.set_ylim(6e-4, 2.0)
    xt = [20, 30, 40, 60]
    ax.xaxis.set_major_locator(CB.FixedLocator(xt))
    ax.xaxis.set_major_formatter(CB.FixedFormatter([str(t) for t in xt]))
    ax.xaxis.set_minor_formatter(CB.NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    ax.set_xlabel(r"Element size $h_{\mathrm{eff}}$ (mm)")
    ax.set_ylabel("Difference to finest mesh (mm)")
    ax.legend(handles=handles, loc="upper left", fontsize=FS_TICK, handlelength=2.0, handletextpad=0.5,
              labelspacing=0.35, borderaxespad=0.4)


def rigid_fit_area(A, B, w):
    """DV.rigid_fit with nodal weights w (lumped area): Q A ~ B, returns Q A.  2026-10-16: the unweighted fit counts
    nodes, so on the tip-polar mesh (nodes crowded at the tip) it pinned the tip and moved the residual to the rim --
    (d) changed with the mesh.  Area weighting is mesh independent and matches field_rms of panel (c)."""
    ca, cb = (w[:, None] * A).sum(0) / w.sum(), (w[:, None] * B).sum(0) / w.sum()
    U, _, Vt = np.linalg.svd(((A - ca) * w[:, None]).T @ (B - cb))
    if np.linalg.det(U @ Vt) < 0:
        Vt[-1] *= -1
    return (A - ca) @ (U @ Vt) + cb


def deviation_graded():
    """CB.deviation for the graded 12.5 mm rung: converged pattern minus the lsq seed of the same mesh
    (rigid fit removed, element-scale part smoothed out exactly as CB.deviation).
    the 12.5 mm rung of the 200 mm nested chain
    (grading tippolar210c200nest, 30 689 nodes, converged on the round-off floor) -- the chain of panel (c).  Was the
    tiptest ladder's 12.5 mm rung (7 879 nodes); FIG_DE_GRADING=tiptest reproduces that
    (fig_vares_journal_combined_flat_de_tiptest_superseded.*)."""
    import MeshStudy_paraboloid_vares_ckpt as CK
    DV = CB.DV
    grading = os.environ.get("FIG_DE_GRADING", "tippolar210c200nest")
    d = os.path.join(CK.study_root(os.environ["MESHSTUDY_TARGET"], grading), "h012")
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
        np.savez(sp, X_lsq=Xs, elements=T, note="lsq_edge seed of the graded 12.5 mm mesh "
                 "(plot_vares_journal_combined_flat.py); the run itself was warm-started")
    w, ae = DV.lumped_area(Xs, T)
    u = rigid_fit_area(a["X_converged"][:, :2], Xs, w) - Xs       # was DV.rigid_fit (node-weighted)
    heff = float(np.sqrt(4 * ae.mean() / np.sqrt(3)))
    us = DV.mls1(Xs, u, w, DV.SMOOTH_SIGMA * heff)
    x0 = float(a["s_lo_mm"]) - float(a["s_mid_mm"])
    return Xs, T, us, x0


MIN_GAP_MM = 1.5          # required clearance between any two figure elements


def report_clearance(fig):
    """Spacing check.  Rendered bounding boxes of every element --
    each axes with its tick labels, axis labels and legend (3-D axes: their tick labels and axis labels only,
    not the empty panes), colour bars, figure texts, figure legend -- and every pair closer than
    MIN_GAP_MM is printed.  Also flags anything outside the page."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    mm = 25.4 / fig.dpi
    boxes = []
    for i, ax in enumerate(fig.axes):
        if ax.name == "3d":
            # mplot3d tick-label extents are stale (placed only at draw time), so use the whole rendered panel
            boxes.append(("3d axes %d" % i, ax.get_tightbbox(r)))
        else:
            boxes.append(("axes %d (%s)" % (i, ax.get_ylabel() or ax.get_xlabel() or "cbar"), ax.get_tightbbox(r)))
    for t in fig.texts:
        boxes.append(("text %r" % t.get_text()[:18], t.get_window_extent(r)))
    for lg in fig.legends:
        boxes.append(("figure legend", lg.get_window_extent(r)))
    W, H = fig.bbox.width, fig.bbox.height
    if os.environ.get("FIG_SPACE_DEBUG"):
        for n, b in boxes:
            print("[box] %-40s x %6.1f..%6.1f  y %6.1f..%6.1f mm" % (n, b.x0 * mm, b.x1 * mm, b.y0 * mm, b.y1 * mm))
        for i, ax in enumerate(fig.axes):
            if ax.name == "3d":
                for a in (ax.xaxis, ax.yaxis, ax.zaxis):
                    for t in a.get_ticklabels() + [a.label]:
                        if t.get_visible() and t.get_text():
                            b = t.get_window_extent(r)
                            print("   3d%d %-14r x %6.1f..%6.1f y %6.1f..%6.1f" % (i, t.get_text(), b.x0 * mm,
                                  b.x1 * mm, b.y0 * mm, b.y1 * mm))
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
    e = np.linspace(RS.EPS_MIN, RS.EPS_MAX, RS.N_FINE)
    EX, EY, SX, SY, BX, BY = RS.evaluate(e)
    Xs, T, us, x0 = deviation_graded() if GRADED else CB.deviation()
    print("[dev] |u_s| max %.2f mm, |u_y| max %.2f mm" % (np.abs(us[:, 0]).max(), np.abs(us[:, 1]).max()))
    c = CJ.compute()

    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    F = lambda x, y, w, h: [x / W_MM, y / H_MM, w / W_MM, h / H_MM]   # mm from bottom-left

    # 2026-10-11 presentation pass: panel letters only -- what each panel shows goes in the caption;
    # inside the figure the only text is axis labels and legends.  The stresses are therefore z-axis labels.
    # The empty band between the rows is closed up and the gores are drawn larger.
    H3D = float(os.environ.get("FIG_H3D", "56"))
    TOP = float(os.environ.get("FIG_TOP", "43.5"))                                                  # bottom edge of the top-row axes (mm)
    axa = fig.add_axes(F(13, TOP, 53, H3D), projection="3d")
    axb = fig.add_axes(F(79.5, TOP, 53, H3D), projection="3d")
    RS.panel(axa, EX, EY, SX, BX, RS.CMAP_W, RS.C_W, "")
    RS.panel(axb, EX, EY, SY, BY, RS.CMAP_F, RS.C_F, "")
    # z-axis labels set as rotated text next to the z ticks: mplot3d puts its own z label far outside the
    # ticks, where it is clipped at the page edge in (a) and runs into the surface of (a) in (b)
    for x_mm, lab in ((4.0, r"$\sigma_{\mathrm{warp}}$ (MPa)"), (70.8, r"$\sigma_{\mathrm{weft}}$ (MPa)")):
        fig.text(x_mm / W_MM, (TOP + 34) / H_MM, lab, rotation=90, ha="center", va="center",
                 fontsize=plt.rcParams["axes.labelsize"])
    fig.legend(handles=[
        Line2D([], [], color="0.05", lw=0.9, ls=(0, (1, 1.3)), label=r"calibration limit, $\varepsilon = 3$ %"),
        Line2D([], [], color="0.05", lw=1.0, ls=(0, (3.5, 1.8)), label="branch boundary of the law")],
        loc="center", bbox_to_anchor=(69 / W_MM, (TOP - 3.6) / H_MM), ncol=2, fontsize=FS_TICK,
        handlelength=2.2, columnspacing=2.0)
    # flatter (48 -> 30 mm tall) and level with the material surfaces of (a), (b): axes plus
    # its x label centred on their rendered extent (~55-103 mm from the bottom)
    axc = fig.add_axes(F(149.5, TOP + 15.5, 37.5, 30))      # 1 mm right: "1.504" ticks are wider than "7.52"
    # the 200 mm chain + the 100 mm chain's 6.25 mm rung; previous figure kept as
    # fig_vares_journal_combined_flat_chain100_superseded.*, reproduce with FIG_PANEL_C=chain100
    if os.environ.get("FIG_PANEL_C", "chain200") == "chain100":
        asymptote_panel_chain100(axc)          # was convergence_panel_tip_split(axc)
    else:
        asymptote_panel_chain200(axc)

    # bottom row: gores lying flat, side by side
    # width budget of the row (190 mm): y label + ticks 14.5 | gore gw | bar + ticks + label 13 | gap |
    # e ticks 9 | gore gw | bar 13  ->  gw = 65 mm (>= 1.5 mm clearance, checked)
    # 2026-10-13: 16 -> 15 mm and (d) moved right 2 mm -- the y/D tick labels ("−0.06") are wider than "−0.3"
    # bottom row edge to edge with the top row -- outer extents x = 1.2 .. 187.0 mm; gh sized so the
    # (e) colour-bar label ends at the right edge of (c); (e) shifts right with the wider (d)
    gh = float(os.environ.get("FIG_GH", "15.7"))               # gore height (mm); width follows the aspect
    gw = gh * 2.92 / 0.72
    xe = 106 + (gw - 15.0 * 2.92 / 0.72)
    yb = 11.3                                                  # bottom edge of the gore axes (12 -> 11.3: letter room)
    axd = fig.add_axes(F(16.5, yb, gw, gh))
    caxd = fig.add_axes(F(16.5 + gw + 2.0, yb, 1.8, gh))        # bar as tall as the gore
    axe = fig.add_axes(F(xe, yb, gw, gh))
    caxe = fig.add_axes(F(xe + gw + 2.0, yb, 1.8, gh))
    # u / D in units of 10^-3, factor printed above each bar
    k = 1e3 / D_MM
    sc = r"$\times 10^{-3}$"
    gore_panel_h(axd, caxd, Xs, T, us[:, 0] * k, x0, 0.4, [-2, 0, 2], r"$u_s/D$", True, sc)
    # (e) u_y.  2026-10-16: tried u_in = -sign(y) u_y (inward shrink, one-sided red map; gore_panel_h one_sided=True,
    # fig_vares_journal_combined_flat_inward_superseded.*) -- user: "return to the old plotting method"
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
