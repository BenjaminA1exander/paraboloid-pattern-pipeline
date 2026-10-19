r"""plot_paraboloid_optimised_vs_solved_nested.py -- mesh convergence of the nested ladder for two equilibrium
states per rung, against the number of elements:

  optimised : the L-BFGS form-finding result (stage 2, normal objective + quality; L3 = stage 2b, step 12251)
  solved    : the same mesh with every free node moved along the normal ONTO the target paraboloid and the force
              densities solved exactly from B(x) q = P(x) (constant-mass pressure p = GAS_K / V(x), min-norm change
              from the optimised q; all q > 0; residual <= 1e-11 |P|; L2 verified by the solver: 0 Newton its,
              drift 1.5e-15).  temp/rim_onsurface_verify/exact_<n>.npz.

Scored over the FULL mesh (every free node), area-weighted, against exact statics; stress from the paper recovery
(CST projection, quadratic MLS of the nodal virial, rho = 8 h_ref sqrt(h/h_ref), h_ref = 159 mm).
Panels: (a) shape |V - V_ex| / V_ex, (b) N_phi (warp) rms, (c) N_theta (weft) rms; numbers = observed order in h.

Run (base env): python plot_paraboloid_optimised_vs_solved_nested.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recover_stress_l2 as RL                         # noqa: E402
import paraboloid_profile as pp                        # noqa: E402
import local_stress_recovery as LR                     # noqa: E402

import paraboloid_dish_config as CFG
TEMP = CFG.TEMP_DIR
OUT = f"{TEMP}/fig_paraboloid_optimised_vs_solved_nested"
CACHE = f"{TEMP}/rim_onsurface_verify/optimised_vs_solved_scores.json"
LEVELS = ("289", "1105", "4321", "17089")
SRC = {"optimised": lambda n: f"{TEMP}/paraboloid_edge_constq_nest{n}_odt_qual1e-4_warm/jaxfdm_paraboloid_dish.npz",
       "solved": lambda n: f"{TEMP}/rim_onsurface_verify/exact_{n}.npz"}
R, DEP = LR.R, LR.D
F = R * R / (4 * DEP)
MM = 2500.0 / R
V_EX = np.pi * R ** 2 * DEP / 2


def score(path):
    st = np.load(path, allow_pickle=True)
    V, T, E, fx = (np.asarray(st[k]) for k in ("vertices", "faces_tri", "edges", "fixed"))
    q = np.asarray(st["q_categories"], float)[np.asarray(st["cat_ids"], int)]
    _, qc = RL.cst_sigma(V, T, E, fx, q)
    Vol = abs(pp._enclosed_volume(V, T)); p = abs(float(st["GAS_K"]) / Vol)
    S, Ai, nrm, h = LR.nodal_virial(V, T, E, qc, fx)
    inn = np.ones(len(V), bool); inn[fx] = False
    Sr = LR.recover(V, S, Ai, nrm, inn, 8 * (159 / MM) * np.sqrt(h / (159 / MM)), "mls2")
    m, hh = LR.mer_hoop(V, Sr)
    r = np.hypot(V[:, 0], V[:, 1]); u = np.sqrt(1 + (r / (2 * F)) ** 2)
    em = m / (p * F * u) - 1; eh = hh / (2 * p * F * u * (1 - 1 / (2 * u ** 2))) - 1
    P_ = V[T]; ta = 0.5 * np.linalg.norm(np.cross(P_[:, 1] - P_[:, 0], P_[:, 2] - P_[:, 0]), axis=1)
    A = np.zeros(len(V)); np.add.at(A, T.ravel(), np.repeat(ta / 3, 3))
    k = inn & np.isfinite(em) & np.isfinite(eh); w = A[k] / A[k].sum()
    return dict(ne=int(len(T)), h=float(h * MM), vol=float(abs(Vol / V_EX - 1)),
                phi=float(np.sqrt(w @ em[k] ** 2)), th=float(np.sqrt(w @ eh[k] ** 2)))


cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
res = {}
for kind, f in SRC.items():
    for n in LEVELS:
        path = f(n); key = f"{os.path.abspath(path)}|{os.path.getmtime(path):.0f}"
        if key not in cache:
            cache[key] = score(path); json.dump(cache, open(CACHE, "w"), indent=1)
        res[(kind, n)] = cache[key]
        c = cache[key]
        print(f"{kind:9s} [{n:>5s}] {c['ne']:6d} tri  dV/V {c['vol']:.3e}  N_phi rms {c['phi']:.3e}  N_theta rms {c['th']:.3e}")

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7.5,
                     "mathtext.fontset": "stixsans", "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "legend.frameon": False})
fig, axs = plt.subplots(1, 3, figsize=(190 / 25.4, 2.7), sharey=True)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.89, wspace=0.08)
STY = {"optimised": ("o", "-", None), "solved": ("s", "--", "white")}
for ax, (key, lab, col) in zip(axs, (("vol", r"(a) shape: $|V-V_{ex}|/V_{ex}$", "#000000"),
                                     ("phi", r"(b) warp: $N_\phi$ rms", "#2456c4"),
                                     ("th", r"(c) weft: $N_\theta$ rms", "#d97706"))):
    for kind, (mk, ls, mfc) in STY.items():
        ne = np.array([res[(kind, n)]["ne"] for n in LEVELS], float)
        hm = np.array([res[(kind, n)]["h"] for n in LEVELS])
        v = np.array([res[(kind, n)][key] for n in LEVELS])
        ax.loglog(ne, v, ls=ls, marker=mk, color=col, ms=4.5, lw=1.1, mfc=mfc or col)
        o = np.log(v[:-1] / v[1:]) / np.log(hm[:-1] / hm[1:])
        for i, oi in enumerate(o):
            ax.annotate(f"{oi:.1f}", (np.sqrt(ne[i] * ne[i + 1]), np.sqrt(v[i] * v[i + 1])),
                        xytext=(0, 5 if kind == "optimised" else -10), textcoords="offset points", fontsize=6,
                        color=col, ha="center")
        print(f"order {kind:9s} {key:4s} {np.round(o, 2)}")
    ax.set_title(lab, fontsize=8, loc="left"); ax.set_xlabel("number of elements (triangles)")
    ax.grid(True, which="major", alpha=0.3)
axs[0].set_ylabel("relative error vs exact (–)")
axs[0].legend(handles=[Line2D([], [], color="0.2", marker="o", ls="-", ms=4, label="optimised (L-BFGS form-finding)"),
                       Line2D([], [], color="0.2", marker="s", ls="--", ms=4, mfc="white",
                              label="solved (nodes on target, q from exact equilibrium)")],
              fontsize=6.2, loc="upper right")
fig.text(0.99, 0.01, "nested ladder (289, 1105, 4321, 17089 nodes); full mesh, area-weighted; paper MLS recovery; "
         "numbers: observed order in $h$ per step", ha="right", fontsize=5.8, color="0.35")
for ext, kw in (("png", {"dpi": 400}), ("pdf", {})):
    fig.savefig(f"{OUT}.{ext}", **kw)
print("saved", f"{OUT}.png")
