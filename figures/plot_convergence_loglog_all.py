r"""plot_convergence_loglog_all.py -- every mesh-refinement study of the CMAME paper on log-log axes.

One panel per study, error against the number of degrees of freedom / elements, with slope guides.  In 2-D,
h ~ N^(-1/2), so O(h^2) is a slope of -1 and O(h) a slope of -1/2 per decade of N.  The observed order between
the two finest rungs, p = -2 x slope, is printed for every series.

  (a) Sec. 3.1, Fig. 3(e): limit point of the cap, p_f, z_f and the gas-law lambda_min; uniform against the analytic
      cap (p_f, z_f) and the 1/192 run (lambda), zoned against the 1/192 run.  Source: the fold_h*.npz files of
      cap_verify FOLD_ONLY (results/fdm_limit_point/bowling_pin, incl. the Euler 1/96 and 1/192 runs).
  (b) Sec. 3.4.1, Table 3: paraboloid form finding, surface error e_s/D and enclosed-volume error against the exact
      paraboloid.  Source: history.csv of the nested runs (L0-L2), the published L3 value, and the volume errors of
      paraboloid_ff_subsection.tex (provenance block).
  (c) Sec. 3.4.1, Table 3 / Fig. 7(d): recovered resultants N_phi, N_theta, L1 and L2 area-weighted norms against the
      exact statics.  Source: temp/rim_onsurface_verify/optimised_vs_solved_mean_rms.json (L0-L2 optimised states,
      L3 = curv_17089_step19250).
  (d) Sec. 3.4.2, Table 4 / Fig. 8(c): VaReS cutting pattern on the nested gore chain: |Delta/Delta_inf - 1| (Delta_inf
      the Celik extrapolation of the three finest), the successive pattern change delta X (L1, L2, % of Delta_inf),
      and the stress mismatch e_sigma minus its extrapolated limit.  Source: results/convergence_loglog/conv_table.json
      (the Table 4 script) and Table 4 (e_sigma).
  (e) Sec. 3.4.3, Table 5 / Fig. 9(d): Abaqus/Explicit inflation, warp and weft resultant deviations (L1, L2) of the
      VaReS and the uniform-shrinkage pattern, |f - f_inf| / f_inf with f_inf the Celik limit of the three finest.
      Source: results/abaqus_ladder_seamtip/vares_vs_uniform/vu_summary.json.

Run (jaxfdm env or base): python plot_convergence_loglog_all.py
Out: results/convergence_loglog/fig_convergence_loglog_all.{png,pdf} and orders.txt
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
TEMP = os.path.join(HERE, "..", "..", "temp")
OUT = os.path.join(HERE, "results", "convergence_loglog")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, GREY, BLACK = "#2456c4", "#d97706", "#6b7280", "#111827"
lines = []


def order(N, e):
    """Observed order between the two finest rungs, p = -2 d ln e / d ln N (2-D meshes)."""
    N, e = np.asarray(N, float), np.asarray(e, float)
    ok = np.isfinite(e) & (e > 0)
    N, e = N[ok], e[ok]
    if len(N) < 2:
        return np.nan, np.nan
    p_last = -2.0 * np.log(e[-1] / e[-2]) / np.log(N[-1] / N[-2])
    p_fit = -2.0 * np.polyfit(np.log(N), np.log(e), 1)[0]
    return p_last, p_fit


def series(ax, N, e, label, **kw):
    N, e = np.asarray(N, float), np.asarray(e, float)
    ok = np.isfinite(e) & (e > 0)
    ax.loglog(N[ok], e[ok], label=label, **kw)
    pl, pf = order(N, e)
    lines.append(f"{ax.get_title()[:4]:5s} {label:48s} last-pair order {pl:5.2f}   least-squares order {pf:5.2f}")


def guides(ax, x0, y0, decades=0.6):
    for slope, lab, dy in ((-0.5, r"$O(h)$", 1.0), (-1.0, r"$O(h^2)$", 1.0)):
        x1 = x0 * 10 ** decades
        y1 = y0 * dy * 10 ** (slope * decades)
        ax.plot([x0, x1], [y0 * dy, y1], color=GREY, lw=0.9, ls="-", zorder=1)
        ax.text(x1 * 1.15, y1, lab, color=GREY, fontsize=7, va="center")


def tidy(ax, xlabel, legend_loc="lower left"):
    ax.grid(True, which="major", lw=0.4, alpha=0.3)
    ax.set_xlabel(xlabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=5.8, frameon=False, loc=legend_loc, handlelength=2.2, labelspacing=0.25)


plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
fig, axs = plt.subplots(2, 3, figsize=(230 / 25.4, 0.62 * 230 / 25.4), constrained_layout=True)
axs = axs.ravel()

# ---------------------------------------------------------------- (a) Sec. 3.1 limit point
ax = axs[0]
ax.set_title("(a) Sec. 3.1, Fig. 3e: cap limit point", loc="left")
D = os.path.join(HERE, "results", "fdm_limit_point", "bowling_pin")
P_STAR = 2.0 * np.sqrt(3.0)


def fold(level, zoned):
    pat = (f"fold_h{level}_g0.25_b10_rb0.5_w0.125_blend_2*.npz" if zoned else f"fold_h{level}_g1_b1_rb0.5_2*.npz")
    f = sorted(glob.glob(os.path.join(D, pat)))
    return np.load(f[-1]) if f else None


U = [(L, fold(L, False)) for L in (6, 12, 24, 48, 96, 192)]
U = [(L, z) for L, z in U if z is not None]
Z = [(L, fold(L, True)) for L in (12, 24, 48, 96, 192)]
Z = [(L, z) for L, z in Z if z is not None]
dof_u = np.array([int(z["n_dof"]) for _, z in U], float)
p_u = np.array([float(z["pf"]) / P_STAR for _, z in U])
z_u = np.array([float(z["z_apex"]) for _, z in U])
l_u = np.array([float(np.mean(z["lamG_fold"][:2])) for _, z in U])
dof_z = np.array([int(z["n_dof"]) for _, z in Z], float)
p_z = np.array([float(z["pf"]) for _, z in Z])
z_z = np.array([float(z["z_apex"]) for _, z in Z])
l_z = np.array([float(np.ravel(z["lamG_fold"])[0]) for _, z in Z])
series(ax, dof_u, np.abs(p_u - 1), r"uniform $p_f$ vs analytic", color=BLACK, marker="o", ms=3.5, lw=1.1)
series(ax, dof_u, np.abs(z_u - 1), r"uniform $z_f$ vs analytic", color=BLACK, marker="^", ms=3.5, lw=1.1, ls=":")
series(ax, dof_u[:-1], np.abs(l_u[:-1] / l_u[-1] - 1), r"uniform $\tilde\lambda_{\min}$ vs 1/192", color=BLACK,
       marker="s", ms=3.5, lw=1.1, ls="-.")
series(ax, dof_z[:-1], np.abs(p_z[:-1] / p_z[-1] - 1), r"zoned $p_f$ vs 1/192", color=GREY, marker="o", ms=3.5,
       lw=1.1, mfc="white")
series(ax, dof_z[:-1], np.abs(z_z[:-1] / z_z[-1] - 1), r"zoned $z_f$ vs 1/192", color=GREY, marker="^", ms=3.5,
       lw=1.1, ls=":", mfc="white")
series(ax, dof_z[:-1], np.abs(l_z[:-1] / l_z[-1] - 1), r"zoned $\tilde\lambda_{\min}$ vs 1/192", color=GREY,
       marker="s", ms=3.5, lw=1.1, ls="-.", mfc="white")
guides(ax, 2e3, 3e-1)
ax.set_ylabel(r"$|f_h/f_{\rm ref} - 1|$")
tidy(ax, "degrees of freedom", "upper right")

# ---------------------------------------------------------------- (b) Sec. 3.4.1 shape and volume
ax = axs[1]
ax.set_title("(b) Sec. 3.4.1, Table 3: shape and volume", loc="left")
dof3 = np.array([3 * (289 - 48), 3 * (1105 - 96), 3 * (4321 - 192), 3 * (17089 - 384)], float)
es = []
for d in ("paraboloid_edge_constq_nest289_odt_qual1e-4_warm", "paraboloid_edge_constq_nest1105_odt_qual1e-4_warm",
          "paraboloid_edge_constq_nest4321_odt_qual1e-4_warm"):
    h = np.genfromtxt(os.path.join(TEMP, d, "history.csv"), delimiter=",", names=True)
    es.append(float(h["surf_err"][-1]) / 6.0)          # e_s / D, D = 2R = 6 (solver units)
es.append(0.0012e-2)                                    # L3: Table 3 (0.0012 %), stage-3 state
vol = [7.17e-3, 1.80e-3, 4.50e-4, 1.12e-4]              # relative volume error, provenance block of the tex
series(ax, dof3, es, r"surface error $e_s/D$", color=BLACK, marker="o", ms=3.5, lw=1.1)
series(ax, dof3, vol, r"enclosed-volume error", color=GREY, marker="s", ms=3.5, lw=1.1, ls="--")
guides(ax, 3e3, 3e-3)
ax.set_ylabel("relative error")
tidy(ax, "degrees of freedom", "upper right")

# ---------------------------------------------------------------- (c) Sec. 3.4.1 resultants
ax = axs[2]
ax.set_title("(c) Sec. 3.4.1, Table 3 / Fig. 7d: resultants", loc="left")
J = json.load(open(os.path.join(TEMP, "rim_onsurface_verify", "optimised_vs_solved_mean_rms.json")))
rows = {}
for k, v in J.items():
    name = os.path.basename(k.split("|")[0])
    rows[(name, v["ne"])] = v
tab = [rows[("jaxfdm_paraboloid_dish.npz", 528)], rows[("jaxfdm_paraboloid_dish.npz", 2112)],
       rows[("jaxfdm_paraboloid_dish.npz", 8448)], rows[("curv_17089_step19250.npz", 33792)]]
for key, lab, col, mk, ls in (("aphi", r"$N_\phi$, $\|e\|_1$", BLUE, "o", "-"), ("rphi", r"$N_\phi$, $\|e\|_2$", BLUE, "o", "--"),
                              ("ath", r"$N_\theta$, $\|e\|_1$", ORANGE, "s", "-"), ("rth", r"$N_\theta$, $\|e\|_2$", ORANGE, "s", "--")):
    series(ax, dof3, [100 * r[key] for r in tab], lab, color=col, marker=mk, ms=3.5, lw=1.1, ls=ls,
           mfc=(col if ls == "-" else "white"))
guides(ax, 3e3, 1.0)
ax.set_ylabel("error of the resultant (%)")
tidy(ax, "degrees of freedom", "upper right")

# ---------------------------------------------------------------- (d) Sec. 3.4.2 cutting pattern
ax = axs[3]
ax.set_title("(d) Sec. 3.4.2, Table 4 / Fig. 8c: VaReS pattern", loc="left")
C = json.load(open(os.path.join(OUT, "conv_table.json")))
N4 = np.array([c["N"] for c in C], float)
D2 = np.array([c["Delta_L2"] for c in C])
f1, f2, f3 = D2[-1], D2[-2], D2[-3]
pR = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(2.0)
Dinf = (2 ** pR * f1 - f2) / (2 ** pR - 1)
series(ax, N4, np.abs(D2 / Dinf - 1) * 100, r"$|\Delta/\Delta_\infty - 1|$ (%)", color=BLACK, marker="o", ms=3.5, lw=1.1)
chg1 = np.array([c.get("chg_L1", np.nan) for c in C[:-1]]) / Dinf * 100
chg2 = np.array([c.get("chg_L2", np.nan) for c in C[:-1]]) / Dinf * 100
series(ax, N4[1:], chg1, r"successive change $\|\delta X\|_1$ (% of $\Delta_\infty$)", color=GREY, marker="^",
       ms=3.5, lw=1.1, ls=":")
series(ax, N4[1:], chg2, r"successive change $\|\delta X\|_2$ (% of $\Delta_\infty$)", color=GREY, marker="v",
       ms=3.5, lw=1.1, ls="--")
esig = np.array([3.36, 2.39, 1.95, 1.78, 1.73])          # Table 4, e_sigma (%)
g1, g2, g3 = esig[-1], esig[-2], esig[-3]
pe = np.log(abs((g3 - g2) / (g2 - g1))) / np.log(2.0)
einf = (2 ** pe * g1 - g2) / (2 ** pe - 1)
series(ax, N4, np.abs(esig - einf), rf"$|e_\sigma - e_{{\sigma,\infty}}|$ (%), $e_{{\sigma,\infty}}$ = {einf:.2f} %",
       color=ORANGE, marker="s", ms=3.5, lw=1.1, ls="-.")
guides(ax, 3e3, 1.0)
ax.set_ylabel("(%)")
tidy(ax, r"degrees of freedom $N$", "upper right")

# ---------------------------------------------------------------- (e) Sec. 3.4.3 Abaqus
ax = axs[4]
ax.set_title("(e) Sec. 3.4.3, Table 5 / Fig. 9d: Abaqus inflation", loc="left")
S = json.load(open(os.path.join(HERE, "results", "abaqus_ladder_seamtip", "vares_vs_uniform", "vu_summary.json")))
nel = np.array([9354240 / 4 ** k for k in (3, 2, 1, 0)], float)       # h100, h050, h025, h012
for pat, col in (("vares", BLUE), ("uniform", ORANGE)):
    for meas, mk, ls in (("warp_L1", "o", "-"), ("weft_L1", "s", "-"), ("warp_L2", "o", "--"), ("weft_L2", "s", "--")):
        v = S["convergence"][pat][meas]
        vals = np.array(v["values_pct"])
        f1, f2, f3 = vals[-1], vals[-2], vals[-3]                 # Celik limit of the three finest (r = 2)
        pC = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(2.0)
        lim = (2 ** pC * f1 - f2) / (2 ** pC - 1)
        series(ax, nel, np.abs(vals - lim) / lim, f"{pat} {meas.replace('_', ' ')} (p = {pC:.2f})", color=col,
               marker=mk, ms=3.3, lw=1.0, ls=ls, mfc=(col if ls == "-" else "white"))
guides(ax, 3e5, 3e-1)
ax.set_ylabel(r"$|f - f_\infty| / f_\infty$")
tidy(ax, "number of elements", "lower left")

# ---------------------------------------------------------------- (f) legend of the slope convention
ax = axs[5]
ax.axis("off")
ax.text(0.0, 0.95, "Reading the slopes", fontsize=8, fontweight="bold", va="top")
ax.text(0.0, 0.85, "2-D meshes: h ~ N^(-1/2), so\n  slope -1   = O(h^2)\n  slope -1/2 = O(h)\n"
        "observed order p = -2 x slope\n\nReferences:\n  (a) analytic cap (uniform p_f, z_f) or the 1/192 run\n"
        "  (b), (c) exact paraboloid and its membrane statics\n  (d) Celik extrapolation of the three finest rungs\n"
        "  (e) Celik extrapolation of the three finest rungs\n\nOrders per series: orders.txt",
        fontsize=7, va="top", family="monospace")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig_convergence_loglog_all.{ext}"), dpi=220)
with open(os.path.join(OUT, "orders.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
print("saved", os.path.join(OUT, "fig_convergence_loglog_all.png"))
