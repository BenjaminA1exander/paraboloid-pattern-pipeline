r"""local_stress_recovery.py -- LOCAL (symmetry-free) recovery of the membrane tension tensor from the
nodal virial of a force-density net, usable on any surface.

Input field: the nodal virial tensor N_i (3x3, tangent-projected, GLOBAL xyz components),
    N_i = 1/(2 A_i) sum_{e at i} q_e L_e^2 t_e (x) t_e ,   A_i = 1/3 star area.
Recovery at node i from the nodes j within plan radius rho (edges) -- nothing assumes a symmetry axis:
  mean      A_j-weighted mean over the disc                               (top-hat patch virial)
  gauss     A_j * exp(-(d/sigma)^2 / 2) weighted mean, sigma = rho / 2
  mls1      moving least squares: per component, A_j-Gaussian-weighted fit of a + b u + c v in local
            tangent coordinates (u, v) at i, value a                       (patch recovery, linear)
  mls2      the same with a full quadratic (6 terms)                       (patch recovery, quadratic)
The recovered tensor is re-projected on node i's tangent plane.  Averaging 3x3 global components is valid
while the patch is small against the radius of curvature (here rho <= 8 h ~ 0.75 vs R_c >= 3).

Run (base env): STATE_NPZ=<jaxfdm_paraboloid_dish.npz> python local_stress_recovery.py
  -> prints rms error vs exact membrane statics for every method x radius, saves a summary figure.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paraboloid_profile as pp                       # noqa: E402
import cut_extraction as CX                            # noqa: E402

R, D = CX.R_FORM, CX.D_FORM
K = CX.K
F_FOC = R * R / (4 * D)


def nodal_virial(V, T, E, q, fixed):
    N = len(V)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    fn = np.cross(b - a, c - a)
    Ai = np.zeros(N)
    nrm = np.zeros((N, 3))
    for k in range(3):
        np.add.at(Ai, T[:, k], 0.5 * np.linalg.norm(fn, axis=1) / 3)
        np.add.at(nrm, T[:, k], 0.5 * fn)
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    dd = V[E[:, 1]] - V[E[:, 0]]
    L = np.linalg.norm(dd, axis=1)
    t = dd / L[:, None]
    S = np.zeros((N, 3, 3))
    o = (q * L ** 2)[:, None, None] * t[:, :, None] * t[:, None, :]
    np.add.at(S, E[:, 0], o)
    np.add.at(S, E[:, 1], o)
    S /= (2 * Ai)[:, None, None]
    Pm = np.eye(3)[None] - nrm[:, :, None] * nrm[:, None, :]
    return Pm @ S @ Pm, Ai, nrm, float(np.median(L))


def recover(V, S, Ai, nrm, interior, rho, method):
    """Recovered tensor at every interior node (3x3 global, re-projected)."""
    tree = cKDTree(V[:, :2])
    out = np.full_like(S, np.nan)
    sig = 0.5 * rho
    comp = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for i in np.flatnonzero(interior):
        nb = np.asarray([j for j in tree.query_ball_point(V[i, :2], rho) if interior[j]])
        d = V[nb] - V[i]
        dist = np.linalg.norm(d, axis=1)
        if method == "mean":
            w = Ai[nb]
        else:
            w = Ai[nb] * np.exp(-0.5 * (dist / sig) ** 2)
        if method in ("mean", "gauss"):
            M = np.tensordot(w, S[nb], axes=1) / w.sum()
        else:
            n = nrm[i]
            e1 = np.cross(n, [0.0, 1.0, 0.0] if abs(n[1]) < 0.9 else [1.0, 0.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            u, v = d @ e1, d @ e2
            cols = [np.ones_like(u), u, v] if method == "mls1" else \
                   [np.ones_like(u), u, v, u * u, u * v, v * v]
            A = np.column_stack(cols)
            if len(nb) < 2 * A.shape[1]:
                continue
            sw = np.sqrt(w)
            M = np.zeros((3, 3))
            Y = np.column_stack([S[nb, a_, b_] for a_, b_ in comp])
            coef, *_ = np.linalg.lstsq(A * sw[:, None], Y * sw[:, None], rcond=None)
            for k, (a_, b_) in enumerate(comp):
                M[a_, b_] = M[b_, a_] = coef[0, k]
        P = np.eye(3) - np.outer(nrm[i], nrm[i])
        out[i] = P @ M @ P
    return out


def mer_hoop(V, S):
    x, y = V[:, 0], V[:, 1]
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    m = np.column_stack([np.cos(th), np.sin(th), -K * r]) / np.sqrt(1 + (K * r) ** 2)[:, None]
    h = np.column_stack([-np.sin(th), np.cos(th), np.zeros(len(V))])
    return np.einsum("ni,nij,nj->n", m, S, m), np.einsum("ni,nij,nj->n", h, S, h)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    STATE = os.environ["STATE_NPZ"]
    OUT_DIR = os.environ.get("OUT_DIR", os.path.dirname(STATE))
    st = np.load(STATE, allow_pickle=True)
    V, T, E, fixed = (np.asarray(st[k]) for k in ("vertices", "faces_tri", "edges", "fixed"))
    q = np.asarray(st["q_categories"], float)[np.asarray(st["cat_ids"], int)]
    p = abs(float(st["GAS_K"]) / pp._enclosed_volume(V, T))
    S, Ai, nrm, h = nodal_virial(V, T, E, q, fixed)
    interior = np.ones(len(V), bool)
    interior[fixed] = False
    r = np.hypot(V[:, 0], V[:, 1])
    u = np.sqrt(1 + (r / (2 * F_FOC)) ** 2)
    exm, exh = p * F_FOC * u, 2 * p * F_FOC * u * (1 - 1 / (2 * u ** 2))
    band = interior & (r > 0.2 * R) & (r < 0.96 * R)
    rim = interior & (r >= 0.9 * R)
    rms = lambda e: 100 * float(np.sqrt(np.nanmean(e ** 2)))        # noqa: E731
    Nm0, Nh0 = mer_hoop(V, S)
    print(f"[raw nodal virial] N_phi {rms((Nm0 / exm - 1)[band]):.2f}%  N_theta {rms((Nh0 / exh - 1)[band]):.2f}%")
    radii = [2, 3, 4, 6, 8]
    methods = ["mean", "gauss", "mls1", "mls2"]
    res = {}
    for meth in methods:
        for rho in radii:
            Sr = recover(V, S, Ai, nrm, interior, rho * h, meth)
            Nm, Nh = mer_hoop(V, Sr)
            res[(meth, rho)] = (rms((Nm / exm - 1)[band]), rms((Nh / exh - 1)[band]),
                                rms((Nm / exm - 1)[rim]), rms((Nh / exh - 1)[rim]))
            print(f"[{meth:5s} rho={rho}h] interior N_phi {res[(meth, rho)][0]:.2f}%  N_theta "
                  f"{res[(meth, rho)][1]:.2f}%   | outer 0.9R+: N_phi {res[(meth, rho)][2]:.2f}%  "
                  f"N_theta {res[(meth, rho)][3]:.2f}%", flush=True)
    np.savez(os.path.join(OUT_DIR, "local_stress_recovery_results.npz"),
             keys=np.array([f"{m}|{r_}" for m, r_ in res]), vals=np.array(list(res.values())))
    # summary figure
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "xtick.direction": "in", "ytick.direction": "in",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(160 / 25.4, 62 / 25.4), sharey=True)
    sty = {"mean": ("#888888", "o", "-"), "gauss": ("#d97706", "s", "--"),
           "mls1": ("#2456c4", "^", "-."), "mls2": ("#c0261f", "D", ":")}
    for k, (ax, lab) in enumerate(zip(axs, (r"meridional $N_\phi$", r"hoop $N_\theta$"))):
        for meth in methods:
            c, mk, ls = sty[meth]
            ax.plot(radii, [res[(meth, rr)][k] for rr in radii], ls, color=c, marker=mk, ms=4, mfc="white",
                    mew=1.0, lw=1.1, label=meth)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(radii); ax.set_xticklabels([str(v) for v in radii])
        ax.set_xlabel("patch radius (edges)"); ax.set_title(lab, fontsize=8)
        ax.grid(alpha=0.25, which="both")
    axs[0].set_ylabel("rms error vs exact, 0.2R-0.96R (%)")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_local_stress_recovery.png"), dpi=300)
    print("[figure] fig_local_stress_recovery.png")
