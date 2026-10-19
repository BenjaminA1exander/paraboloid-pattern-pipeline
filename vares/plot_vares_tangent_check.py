r"""plot_vares_tangent_check.py -- finite-difference check of the VaReS tangent K = dr/dX.

The solver's own assembly (vares_sparse.assemble_residual_tangent_sparse, same kwargs as MeshStudy_paraboloid_vares_ckpt:
full Chen law, current-frame target, vary_detJ) on G0 of the 200 mm nested chain (164 nodes).  For random unit
directions v on the free coordinates, the relative error
    e(eps) = || K v - [r(X + eps v) - r(X - eps v)] / (2 eps) || / || K v ||
is plotted against eps (mm) at two states: the initial guess (large mismatch, every term of K active) and the converged
pattern.  An exact K gives e ~ eps^2 until round-off takes over at small eps.
OUT  fig_vares_tangent_check.{pdf,png}
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
import MeshStudy_paraboloid_vares_ckpt as CK                      # noqa: E402
import optimise_cutting_pattern_npl_gore_chen as chen             # noqa: E402
import vares_sparse as VS                                         # noqa: E402

TAG = os.environ.get("TANGENT_RUNG", "h200")
N_DIR = 3
EPS = np.logspace(0, -7, 8)
OUT = os.path.join(HERE, "fig_vares_tangent_check")


def main():
    root = CK.study_root("cstraw", "tippolar210c200nest")
    d = np.load(os.path.join(root, TAG, "pattern.npz"), allow_pickle=True)
    s = np.load(os.path.join(root, TAG, "setup.npz"), allow_pickle=True)
    Xs, els = d["X_spatial"], [t for t in d["elements"]]
    free, dJ = s["free"], list(s["detJ_ref_all"])
    kw = dict(linear_material=False, vary_detJ=chen.VARY_DETJ, sigma_star_all=list(s["sigma_star_all"]))
    rng = np.random.default_rng(0)
    res = {}
    for state, X in (("initial guess", d["X_initial"].copy()), ("converged", d["X_converged"].copy())):
        R, K = VS.assemble_residual_tangent_sparse(X, Xs, els, None, dJ, **kw)
        K = VS.restrict(K, free)
        rows = []
        for _ in range(N_DIR):
            v = np.zeros(2 * len(Xs))
            v[free] = rng.standard_normal(len(free))
            v /= np.linalg.norm(v)
            Kv = K @ v[free]
            e = []
            for eps in EPS:
                Xp, Xm = X.copy(), X.copy()
                Xp[:, :2] += eps * v.reshape(-1, 2)
                Xm[:, :2] -= eps * v.reshape(-1, 2)
                fd = (chen.assemble_residual(Xp, Xs, els, None, dJ, **kw)[free]
                      - chen.assemble_residual(Xm, Xs, els, None, dJ, **kw)[free]) / (2 * eps)
                e.append(np.linalg.norm(Kv - fd) / np.linalg.norm(Kv))
            rows.append(e)
            print("%-14s ||r|| %.2e  e(eps) %s  min %.1e" % (state, np.linalg.norm(R[free]), np.array2string(np.array(e), precision=1), min(e)))
        res[state] = np.array(rows)
    np.savez(OUT + "_data.npz", eps=EPS, **{k.replace(" ", "_"): v for k, v in res.items()})

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True, "pdf.fonttype": 42, "legend.frameon": False})
    fig = plt.figure(figsize=(90 / 25.4, 70 / 25.4))
    ax = fig.add_axes([0.2, 0.18, 0.75, 0.75])
    for (state, E), col, mk in zip(res.items(), ("k", "0.55"), ("o", "s")):
        for e in E:
            ax.loglog(EPS, e, "-", color=col, marker=mk, ms=3.2, lw=0.9, mfc="white" if col != "k" else col)
    ax.set_xlabel(r"Step $\epsilon$ (mm)")
    ax.set_ylabel(r"Relative error $\|\mathbf{K}\mathbf{v} - \delta_\epsilon\mathbf{r}\| \,/\, \|\mathbf{K}\mathbf{v}\|$")
    ax.legend(handles=[Line2D([], [], color="k", marker="o", ms=3.2, lw=0.9, label="initial guess"),
                       Line2D([], [], color="0.55", marker="s", mfc="white", ms=3.2, lw=0.9, label="converged")],
              loc="upper left", fontsize=8)
    for ext, kw_ in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig("%s.%s" % (OUT, ext), **kw_)
    fig.savefig(OUT + "_preview.png", dpi=200)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
