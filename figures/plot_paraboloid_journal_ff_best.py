r"""plot_paraboloid_journal_ff_best.py -- the four-panel journal figure of plot_paraboloid_journal_ff.py (1 x 4, 190 mm),
rebuilt on the BEST, most up-to-date optimised form-finding state and made fully DIMENSIONLESS:

  state : nested L3 (17089 nodes), optimiser with the curvature-sensitive shape term (W_QUAL = 0, W_CURV = 1.237e-7,
          L-BFGS m = 50; Euler checkpoint step 19250), CST projection (project_self_stress_cst.py)
          -> temp/journal_ff_best/curv_17089_step19250_projected_cst.npz
  (a) signed normal deviation from the target paraboloid / aperture D = 2R, mesh overlaid; plan axes x/R, y/R
  (b) principal tension trajectories (plot_paraboloid_tension_trajectories_fit.py on the same state, fitted field,
      bootstrap stop), colour = principal tension / (p f)  (p f = exact apex tension)
  (c) every interior node's recovered N_phi, N_theta / (p f) vs r/R, with exact membrane statics
      (quadratic MLS of the nodal virial, rho = 8 h_ref sqrt(h/h_ref), h_ref = 159 mm -- the paper recovery)
  (d) mesh convergence of the best optimised state of each nested rung: full-mesh area-weighted rms of N/N_exact - 1
      (%) vs degrees of freedom (3 x free nodes); numbers = observed order in h per refinement
      (L0-L2 stage 2; L3 as above; values from table3_formfinding.score, cached in rim_onsurface_verify/table3_scores.json)

Run (base env): python plot_paraboloid_journal_ff_best.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paraboloid_profile as pp                       # noqa: E402
import local_stress_recovery as LR                     # noqa: E402

import paraboloid_dish_config as CFG
TEMP = CFG.TEMP_DIR
BD = f"{TEMP}/journal_ff_best"
STATE = f"{BD}/curv_17089_step19250_projected_cst.npz"
TRAJ = f"{BD}/fig_paraboloid_tension_trajectories_fit_cst_lines.npz"
BEST = [f"{TEMP}/paraboloid_edge_constq_nest{n}_odt_qual1e-4_warm/jaxfdm_paraboloid_dish.npz" for n in ("289", "1105", "4321")]
BEST += [f"{TEMP}/rim_onsurface_verify/curv_17089_step19250.npz"]
PANEL_D = os.environ.get("PANEL_D", "rms")             # rms: rms + orders (log-log); conv: mean of N/N_exact - 1; l1: L1 norm
SMOOTH = os.environ.get("SMOOTH", "0") == "1"          # monotone cubic (PCHIP) curve through the points, in log DOF
OUT = os.environ.get("OUT", f"{TEMP}/results/figures/fig_paraboloid_journal_ff_best" + ({"conv": "_conv", "l1": "_l1"}.get(PANEL_D, ""))
                     + ("_smooth" if SMOOTH else ""))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
R, DEP = LR.R, LR.D
F_FOC = R * R / (4 * DEP)
MM = 2500.0 / R
D_AP = 2.0 * R
H_REF = 159.0 / MM                                    # MLS reference edge (form units), as the paper recovery
K_SUPP = 8.0
C_X, C_Y = "#2456c4", "#d97706"
WIDTH_IN = 190.0 / 25.4
FS_LABEL, FS_TICK, FS_LEG, FS_PANEL = 8.0, 7.5, 7.0, 9.0
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans", "font.size": FS_LABEL, "legend.fontsize": FS_LEG,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "axes.labelsize": FS_LABEL,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "grid.linewidth": 0.5, "grid.alpha": 0.25, "legend.frameon": False, "legend.handletextpad": 0.5,
    "legend.borderpad": 0.2, "lines.solid_capstyle": "round", "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "savefig.facecolor": "white"})


def plan_axes(ax, L, ylabel=True):
    ax.set_aspect("equal"); ax.set_xlim(-L, L); ax.set_ylim(-L, L)
    t = [-1, -0.5, 0, 0.5, 1]; ax.set_xticks(t); ax.set_yticks(t)
    ax.set_xticklabels(["-1", "", "0", "", "1"]); ax.set_yticklabels(["-1", "", "0", "", "1"])
    ax.set_xlabel("$x/R$", labelpad=1)
    if ylabel:
        ax.set_ylabel("$y/R$", labelpad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)


st = np.load(STATE, allow_pickle=True)
V, T, E, fixed = (np.asarray(st[k]) for k in ("vertices", "faces_tri", "edges", "fixed"))
q = np.asarray(st["q_categories"], float)[np.asarray(st["cat_ids"], int)]
p = abs(float(st["GAS_K"]) / pp._enclosed_volume(V, T))
PF = p * F_FOC                                        # exact apex tension: the tension scale
x, y, z = V[:, 0], V[:, 1], V[:, 2]
r = np.hypot(x, y)
inn = np.ones(len(V), bool); inn[fixed] = False
k_ = 2.0 * DEP / R ** 2
dev_n = (z - DEP * (1.0 - r ** 2 / R ** 2)) / np.sqrt(1.0 + (k_ * r) ** 2) / D_AP
S, Ai, nrm, h = LR.nodal_virial(V, T, E, q, fixed)
rho = K_SUPP * H_REF * np.sqrt(h / H_REF)
print(f"[mls] support rho = {rho * MM:.0f} mm = {rho / h:.2f} median edges")
Sr = LR.recover(V, S, Ai, nrm, inn, rho, "mls2")
Nm, Nh = LR.mer_hoop(V, Sr)
u = np.sqrt(1 + (r / (2 * F_FOC)) ** 2)
em, eh = 100 * (Nm / (PF * u) - 1), 100 * (Nh / (2 * PF * u * (1 - 1 / (2 * u ** 2))) - 1)
ok = inn & np.isfinite(em) & np.isfinite(eh)
print(f"[extract] full mesh rms: N_phi {np.sqrt(np.mean(em[ok] ** 2)):.3f}%, N_theta {np.sqrt(np.mean(eh[ok] ** 2)):.3f}% (unweighted)")

Xn, Yn = x / R, y / R
L = 1.06
rim_o = fixed[np.argsort(np.arctan2(y[fixed], x[fixed]))]; rim_o = np.append(rim_o, rim_o[0])

SQ_MAP, SQ_PLOT, BOT, CBW = 1.36, 1.24, 0.36, 0.07
FIG_H = BOT + SQ_MAP + 0.30
X_A, X_CBA, X_B, X_CBB, X_C, X_D = 0.40, 1.81, 2.33, 3.74, 4.43, 6.17
fig = plt.figure(figsize=(WIDTH_IN, FIG_H))


def box(x0, w, h_=None, y0=BOT):
    h_ = w if h_ is None else h_
    return fig.add_axes([x0 / WIDTH_IN, y0 / FIG_H, w / WIDTH_IN, h_ / FIG_H])


def cbar_box(x0):
    hh = 0.92 * SQ_MAP
    return box(x0, CBW, hh, BOT + 0.5 * (SQ_MAP - hh))


# ---- (a) deviation / D ---------------------------------------------------------------------------------------------
ax = box(X_A, SQ_MAP)
e10 = int(np.floor(np.log10(np.percentile(np.abs(dev_n[inn]), 99.5))))
dv = dev_n * 10.0 ** (-e10)
vm = float(np.percentile(np.abs(dv[inn]), 99.5))
g_ = 0.5
nrm_c = mcolors.FuncNorm((lambda v: np.sign(v) * np.abs(v) ** g_, lambda v: np.sign(v) * np.abs(v) ** (1 / g_)),
                         vmin=-vm, vmax=vm)
im = ax.tripcolor(mtri.Triangulation(Xn, Yn, T), np.clip(dv, -vm, vm), shading="gouraud", cmap="RdBu_r",
                  norm=nrm_c, rasterized=True)
Ee = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
P2 = np.column_stack([Xn, Yn])
ax.add_collection(LineCollection(np.stack([P2[Ee[:, 0]], P2[Ee[:, 1]]], axis=1), colors="0.2",
                                 linewidths=0.04, alpha=0.3, zorder=3, rasterized=True))
ax.plot(Xn[rim_o], Yn[rim_o], "-", color="0.15", lw=0.7, zorder=4)
plan_axes(ax, L)
cb = fig.colorbar(im, cax=cbar_box(X_CBA), extend="both")
cb.ax.tick_params(labelsize=FS_TICK, width=0.6); cb.outline.set_linewidth(0.6)
cb.ax.set_title(r"$\delta/D\;(\times10^{%d})$" % e10, fontsize=5.5, pad=13, loc="center")   # one line, above the frame of (a)
base = [v_ for v_ in (0.25, 0.5, 1, 2, 4, 8) if 0.15 * vm <= v_ < 0.75 * vm]
tk = sorted(set([-vm] + [-v_ for v_ in base] + [0.0] + base + [vm]))
cb.set_ticks(tk); cb.set_ticklabels([f"{t:.1f}" if abs(abs(t) - vm) < 1e-12 else f"{t:g}" for t in tk])
print(f"[dev] delta/D rms {np.sqrt(np.mean(dev_n[inn] ** 2)):.2e}, max |.| {np.abs(dev_n[inn]).max():.2e}")

# ---- (b) trajectories, colour N / (p f) ----------------------------------------------------------------------------
ax = box(X_B, SQ_MAP)
tz = np.load(TRAJ)
S2 = 1000.0 / MM / R                                   # trajectory points are in m -> x/R
cmap_t = mcolors.LinearSegmentedColormap.from_list("plasma_trunc", plt.cm.plasma(np.linspace(0.0, 0.88, 256)))
norm_t = mcolors.Normalize(float(tz["vlo"]) / PF, float(tz["vhi"]) / PF)
ax.fill(Xn[rim_o], Yn[rim_o], color="#f5f5f5", lw=0, zorder=0)
off, fam = tz["offsets"], tz["family"]
for z_, f_want, lw_ in ((3, 2, 0.45), (4, 1, 0.95)):
    for kk in np.flatnonzero(fam == f_want):
        Pk = tz["pts"][off[kk]:off[kk + 1]] * S2; vk = tz["vals"][off[kk]:off[kk + 1]] / PF
        lc = LineCollection(np.stack([Pk[:-1], Pk[1:]], axis=1), cmap=cmap_t, norm=norm_t, linewidths=lw_,
                            capstyle="round", joinstyle="round", zorder=z_)
        lc.set_array(0.5 * (vk[:-1] + vk[1:])); ax.add_collection(lc)
i0 = int(np.argmin(r))
wS = np.linalg.eigvalsh(Sr[i0]); n_apex = 0.5 * (wS[1] + wS[2]) / PF
ax.plot(Xn[i0], Yn[i0], "o", ms=3.0, mfc=cmap_t(norm_t(n_apex)), mec=cmap_t(norm_t(n_apex)), mew=0, zorder=6)
ax.plot(Xn[rim_o], Yn[rim_o], "-", color="0.15", lw=0.7, zorder=5)
plan_axes(ax, L, ylabel=False)
cb = fig.colorbar(ScalarMappable(norm=norm_t, cmap=cmap_t), cax=cbar_box(X_CBB))
cb.ax.tick_params(labelsize=FS_TICK, width=0.6); cb.outline.set_linewidth(0.6)
cb.ax.set_title(r"$N/pf$", fontsize=5.5, pad=3, loc="center")

# ---- (c) every node vs exact statics, N / (p f) vs r / R -------------------------------------------------------------
ax = box(X_C, SQ_PLOT)
rr = np.linspace(0, R, 300); uu = np.sqrt(1 + (rr / (2 * F_FOC)) ** 2)
ax.plot(r[ok] / R, Nh[ok] / PF, "^", ms=1.4, mfc="none", mec=C_Y, mew=0.3, alpha=0.6, rasterized=True)
ax.plot(r[ok] / R, Nm[ok] / PF, "o", ms=1.4, mfc="none", mec=C_X, mew=0.3, alpha=0.6, rasterized=True)
ax.plot(rr / R, 2 * uu * (1 - 1 / (2 * uu ** 2)), "-", color="0.05", lw=1.0)
ax.plot(rr / R, uu, "--", color="0.05", lw=1.0)
ax.set_xlim(0, 1); ax.set_xlabel("$r/R$", labelpad=1); ax.set_ylabel(r"$N/(p\,f)$", labelpad=2)
ax.grid(True)
ax.legend(handles=[Line2D([], [], ls="none", marker="^", mfc="none", mec=C_Y, ms=3.5, label=r"$N_\theta$"),
                   Line2D([], [], ls="none", marker="o", mfc="none", mec=C_X, ms=3.5, label=r"$N_\phi$"),
                   Line2D([], [], color="0.05", lw=1.0, label=r"exact $N_\theta$"),
                   Line2D([], [], color="0.05", lw=1.0, ls="--", label=r"exact $N_\phi$")],
          loc="upper left", ncol=2, fontsize=FS_LEG, handlelength=1.5, columnspacing=0.8, labelspacing=0.3)

# ---- (d) mesh convergence of the best optimised state of each rung -------------------------------------------------
ax = box(X_D, SQ_PLOT)
import table3_formfinding as T3                                   # noqa: E402
rows = [T3.score_cached(path) for path in BEST]                    # the Table 3 measures of each state (cached)
dof = np.array([c["ndof"] for c in rows], float)


def series(ax, x, v, col, mk, logy):
    """Markers at the measured points; joined by straight segments, or (SMOOTH) by a monotone cubic through them."""
    if not SMOOTH:
        (ax.loglog if logy else ax.semilogx)(x, v, "-", color=col, lw=1.2, marker=mk, ms=4.0, mfc=col, mec=col,
                                              zorder=3)
        return
    from scipy.interpolate import PchipInterpolator
    xs = np.logspace(np.log10(x[0]), np.log10(x[-1]), 200)
    yv = np.log10(v) if logy else v
    ys = PchipInterpolator(np.log10(x), yv)(np.log10(xs))
    (ax.loglog if logy else ax.semilogx)(xs, 10 ** ys if logy else ys, "-", color=col, lw=1.2, zorder=2)
    ax.plot(x, v, mk, color=col, ms=4.0, mfc=col, mec=col, zorder=3)


for key, col, mk, dy in ((("phi", C_X, "o", -10), ("th", C_Y, "^", 5)) if PANEL_D == "rms" else ()):
    v = 100 * np.array([c[f"r{key}"] for c in rows])
    series(ax, dof, v, col, mk, True)
    o = np.log(v[:-1] / v[1:]) / np.log(np.sqrt(dof[1:] / dof[:-1]))
    for i, oi in enumerate(o):
        dyi = dy if i < len(o) - 1 else (-10 if dy > 0 else 5)   # lines cross before the last step: swap sides
        ax.annotate(f"{oi:.1f}", (np.sqrt(dof[i] * dof[i + 1]), np.sqrt(v[i] * v[i + 1])), xytext=(0, dyi),
                    textcoords="offset points", fontsize=6, color=col, ha="center")
    print(f"[conv] {key}: rms % {np.round(v, 4)}  orders {np.round(o, 2)}")
if PANEL_D == "conv":
    for key, col, mk in (("phi", C_X, "o"), ("th", C_Y, "^")):
        v = 100 * np.array([c[f"m{key}"] for c in rows])
        series(ax, dof, v, col, mk, False)
        print(f"[conv] {key}: mean % {np.round(v, 4)}")
    ax.axhline(0.0, color="0.55", lw=0.8, zorder=1)
    ax.set_ylabel(r"$\langle N/N_{\mathrm{exact}}\rangle - 1$ (%)", labelpad=1)   # <.> = area-weighted mean
elif PANEL_D == "l1":                                  # area-weighted L1 norm, convergence style (linear y to 0)
    for key, col, mk in (("phi", C_X, "o"), ("th", C_Y, "^")):
        v = 100 * np.array([c[f"a{key}"] for c in rows])
        series(ax, dof, v, col, mk, False)
        print(f"[conv] {key}: L1 % {np.round(v, 4)}  orders "
              f"{np.round(np.log(v[:-1] / v[1:]) / np.log(np.sqrt(dof[1:] / dof[:-1])), 2)}")
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r"$\Vert N/N_{\mathrm{exact}} - 1\Vert_{1}$ (%)", labelpad=1)   # sum a|e| / sum a
else:
    ax.set_ylabel(r"rms of $N/N_{\mathrm{exact}}-1$ (%)", labelpad=2)
ax.set_xlim(500, 8e4); ax.grid(True, which="major")
ax.set_xlabel("degrees of freedom", labelpad=1)
ax.legend(handles=[Line2D([], [], color=C_Y, marker="^", ms=4, lw=1.2, label=r"$N_\theta$"),   # same order as (c)
                   Line2D([], [], color=C_X, marker="o", ms=4, lw=1.2, label=r"$N_\phi$")],
          loc={"rms": "lower left", "l1": "upper right"}.get(PANEL_D, "lower right"), fontsize=FS_LEG)

fig.canvas.draw()
y_let = (BOT + SQ_MAP + 0.10) / FIG_H
for kk, x_ in enumerate((X_A - 0.38, X_B - 0.22, X_C - 0.40, X_D - 0.42)):
    fig.text(x_ / WIDTH_IN, y_let, f"({'abcd'[kk]})", fontsize=FS_PANEL, fontweight="bold", va="bottom")
for ext, kw in (("png", {"dpi": 600}), ("pdf", {})):
    fig.savefig(f"{OUT}.{ext}", **kw)
    print("saved", f"{OUT}.{ext}")
