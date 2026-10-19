r"""plot_chen_response_surface_journal.py -- journal version of fig_chen_response_surface (same law, same
domain, same two surfaces), cleaned up for print.

    (a) sigma_warp(eps_warp, eps_weft)      (b) sigma_weft(eps_warp, eps_weft)      gamma_xy = 0

Every point goes through chen2022_constitutive.chen_stress() (the line-by-line port of the VUMAT),
exactly as plot_chen_response_surface.py.  Changes against that figure are presentation only:
  * iso-strain lines every 1 % drawn ON the surface: the shape reads without relying on colour
    (greyscale safe) and stress can be read off at whole-percent strains;
  * branch boundary and calibration limit are drawn ON the sheet, not projected on a floor 42 % of the
    range below it (empty band, floating "0" tick, lines hidden under the sheet); floor contours dropped;
  * the branch boundary is traced on a 241 x 241 grid (was the 81 x 81 staircase);
  * tick labels at every percent, regular-weight panel letters, shorter legend.

Run (BASE anaconda env):
    python plot_chen_response_surface_journal.py
Output: fig_chen_response_surface_journal.{pdf,png}  (190 mm, vector PDF, Type-42 fonts)
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from chen2022_constitutive import chen_stress, UMAT_PROPS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_chen_response_surface_journal")

EPS_MIN, EPS_MAX = -0.010, 0.040
N_FINE, STRIDE = 241, 3                  # branch map on the fine grid; surface on every 3rd point (81)
EPS_CAL = 0.030
ISO = np.arange(-0.01, 0.0401, 0.01)     # iso-strain lines on the surface

C_W, C_F = "#2456c4", "#d97706"          # pipeline palette: warp blue, weft orange
CMAP_W, CMAP_F = "Blues", "Oranges"
CMAP_LO = 0.22

WIDTH_IN = 190 / 25.4
FIGSIZE = (WIDTH_IN, WIDTH_IN * 0.40)
FS_LABEL, FS_TICK, FS_LEG, FS_PANEL = 8.0, 7.0, 7.0, 9.0

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans", "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEG,
    "axes.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
BRANCH_ID = {"biaxial": 0, "uniaxial": 1, "fallback": 2}


def evaluate(e):
    EX, EY = np.meshgrid(e, e, indexing="ij")
    SX, SY = np.zeros_like(EX), np.zeros_like(EX)
    BX, BY = np.zeros(EX.shape, int), np.zeros(EX.shape, int)
    for i in range(len(e)):
        for j in range(len(e)):
            S, _, info = chen_stress(np.array([[EX[i, j], 0.0], [0.0, EY[i, j]]]), UMAT_PROPS)
            SX[i, j], SY[i, j] = S[0, 0], S[1, 1]
            BX[i, j], BY[i, j] = BRANCH_ID[info["branch_x"]], BRANCH_ID[info["branch_y"]]
    return EX, EY, SX, SY, BX, BY


def panel(ax, EXf, EYf, Zf, Bf, cmap_name, col, zlabel):
    EX, EY, Z = EXf[::STRIDE, ::STRIDE], EYf[::STRIDE, ::STRIDE], Zf[::STRIDE, ::STRIDE]
    cmap = plt.get_cmap(cmap_name)
    z_lo, z_hi = float(Z.min()), float(Z.max())
    norm = Normalize(z_lo, z_hi)
    Zc = 0.25 * (Z[:-1, :-1] + Z[1:, :-1] + Z[:-1, 1:] + Z[1:, 1:])
    fc = cmap(CMAP_LO + (1.0 - CMAP_LO) * norm(Zc))
    n = EX.shape[0]
    surf = ax.plot_surface(EX, EY, Z, facecolors=fc, rcount=n, ccount=n, shade=False,
                           linewidth=0.15, antialiased=True)
    surf.set_edgecolors(fc.reshape(-1, 4))       # same-colour edges: no hairline moire in the PDF

    # iso-strain lines on the sheet (from the fine grid, so they follow the kinks exactly)
    e = EXf[:, 0]
    for v in ISO:
        k = int(np.argmin(np.abs(e - v)))
        for xs, ys, zs in ((EXf[k, :], EYf[k, :], Zf[k, :]), (EXf[:, k], EYf[:, k], Zf[:, k])):
            ax.plot(xs, ys, zs, color="0.15", lw=0.35, alpha=0.75, zorder=6)
    for xs, ys, zs in ((EXf[0, :], EYf[0, :], Zf[0, :]), (EXf[-1, :], EYf[-1, :], Zf[-1, :]),
                       (EXf[:, 0], EYf[:, 0], Zf[:, 0]), (EXf[:, -1], EYf[:, -1], Zf[:, -1])):
        ax.plot(xs, ys, zs, color=col, lw=0.7, zorder=7)

    # branch boundary and calibration box drawn ON the sheet (a floor projection is hidden under it)
    from scipy.interpolate import RegularGridInterpolator
    zi = RegularGridInterpolator((e, e), Zf)
    if Bf.min() != Bf.max():
        cs = plt.figure().add_subplot().contour(EXf, EYf, Bf.astype(float), levels=[0.5, 1.5])
        for path in (p for seg in cs.allsegs for p in seg):
            if len(path) > 1:
                ax.plot(path[:, 0], path[:, 1], zi(path) + 0.004 * (z_hi - z_lo), color="0.05",
                        lw=1.0, ls=(0, (3.5, 1.8)), zorder=8)
        plt.close(cs.axes.figure)
    t = np.linspace(0, EPS_CAL, 200)
    for path in (np.c_[t, 0 * t + EPS_CAL], np.c_[0 * t + EPS_CAL, t]):   # the two edges not on iso-0 lines
        ax.plot(path[:, 0], path[:, 1], zi(path) + 0.004 * (z_hi - z_lo), color="w", lw=1.6, zorder=8)
        ax.plot(path[:, 0], path[:, 1], zi(path) + 0.004 * (z_hi - z_lo), color="0.05", lw=0.9,
                ls=(0, (1, 1.3)), zorder=9)
    z_floor = z_lo - 0.04 * (z_hi - z_lo)

    ax.set_xlim(EPS_MIN, EPS_MAX)
    ax.set_ylim(EPS_MIN, EPS_MAX)
    ax.set_zlim(z_floor, z_hi)
    ax.view_init(elev=24, azim=-128)
    ax.set_box_aspect((1.0, 1.0, 0.72))
    ticks = np.arange(-0.01, 0.0401, 0.01)
    labs = [""] + ["%d" % round(100 * t) for t in ticks[1:]]   # -1 would collide at the corner
    ax.set_xticks(ticks); ax.set_xticklabels(labs)
    ax.set_yticks(ticks); ax.set_yticklabels(labs)
    ax.set_zticks(np.arange(0.0, z_hi + 1, 200.0))
    ax.set_xlabel(r"$\varepsilon_{\mathrm{warp}}$ (%)", labelpad=-5)
    ax.set_ylabel(r"$\varepsilon_{\mathrm{weft}}$ (%)", labelpad=-5)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(zlabel, labelpad=1, rotation=90)
    ax.tick_params(axis="x", pad=-3.5)
    ax.tick_params(axis="y", pad=-3.5)
    ax.tick_params(axis="z", pad=0.5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("0.75")
        axis._axinfo["grid"].update({"color": "0.88", "linewidth": 0.35})
        axis.line.set_linewidth(0.6)
        axis.line.set_color("0.4")
    return z_lo, z_hi


def main():
    e = np.linspace(EPS_MIN, EPS_MAX, N_FINE)
    assert (N_FINE - 1) % STRIDE == 0
    EX, EY, SX, SY, BX, BY = evaluate(e)
    fig = plt.figure(figsize=FIGSIZE)
    axa = fig.add_axes([0.015, 0.075, 0.48, 0.93], projection="3d")
    axb = fig.add_axes([0.505, 0.075, 0.48, 0.93], projection="3d")
    ra = panel(axa, EX, EY, SX, BX, CMAP_W, C_W, r"$\sigma_{\mathrm{warp}}$ (MPa)")
    rb = panel(axb, EX, EY, SY, BY, CMAP_F, C_F, r"$\sigma_{\mathrm{weft}}$ (MPa)")
    fig.text(0.06, 0.90, "(a)", fontsize=FS_PANEL)
    fig.text(0.55, 0.90, "(b)", fontsize=FS_PANEL)
    fig.legend(handles=[
        Line2D([], [], color="0.15", lw=0.6, label=r"iso-strain lines, 1 % spacing"),
        Line2D([], [], color="0.05", lw=0.9, ls=(0, (1, 1.3)), label=r"calibration limit, $\varepsilon = 3$ %"),
        Line2D([], [], color="0.05", lw=1.0, ls=(0, (3.5, 1.8)), label="boundary between branches of the law")],
        loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, handlelength=2.0, columnspacing=2.2)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig("%s.%s" % (OUT, ext), **kw)
    fig.savefig(OUT + "_preview.png", dpi=150)
    print("[out] %s.{pdf,png}" % OUT)
    print("sigma_warp %.1f .. %.1f MPa, sigma_weft %.1f .. %.1f MPa" % (ra + rb))
    inv = {v: k for k, v in BRANCH_ID.items()}
    for nm, B in (("warp", BX), ("weft", BY)):
        print(nm, ", ".join("%s %.1f%%" % (inv[k], 100 * (B == k).mean()) for k in np.unique(B)))


if __name__ == "__main__":
    main()
