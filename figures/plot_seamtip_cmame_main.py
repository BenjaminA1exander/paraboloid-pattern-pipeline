r"""plot_seamtip_cmame_main.py -- CMAME main-text convergence material and the THESIS convergence figure for the seam-
and tip-resolved Abaqus sector ladder.

Data: the per-rung cache of plot_seamtip_asymptote_combined.py (whole gore; settled import per rung; VaReS L0-L4,
uniform L0-L3).  Norms of the relative deviation e = sigma / sigma* - 1 (stress) and of the normal deviation from the
target surface (shape), area-weighted:  L1 = sum a_i |e_i| / sum a_i ,  L2 = sqrt(sum a_i e_i^2 / sum a_i).
GCI (Celik et al. 2008, J. Fluids Eng. 130 078001; Roache): three finest rungs phi1 (finest), phi2, phi3, constant
refinement ratio r = 2; eps21 = phi2 - phi1, eps32 = phi3 - phi2; observed order p = |ln|eps32 / eps21|| / ln r;
convergence type monotonic (eps32 / eps21 > 0) or oscillatory (< 0); phi_ext = (r^p phi1 - phi2) / (r^p - 1);
e_a = |phi1 - phi2| / |phi1|; GCI_fine = 1.25 e_a / (r^p - 1).
OUT results/abaqus_ladder_seamtip/paper/
  fig_seamtip_order_L1L2.{pdf,png}     CMAME: relative change between successive meshes, log-log, rows L1 / L2
  table_seamtip_gci.{csv,tex,json}     CMAME: observed order and GCI on the finest mesh
  ../thesis/fig_seamtip_convergence_thesis.{pdf,png}   THESIS: normalised asymptote f / f_inf, rows L1 / L2; the
      fit f_inf + a h^p and the pre-set rule (plot_abaqus_sector_vares_asymptote.classify) as before; error bar on
      the finest mesh = GCI_fine (replaces the earlier ad-hoc fit band)
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_seamtip_asymptote_combined as AC                         # noqa: E402
from plot_abaqus_sector_vares_asymptote import classify              # noqa: E402
from plot_vares_shape_asymptote import plt, Line2D                   # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter   # noqa: E402

OUTD = AC.OUTD
THD = os.path.join(os.path.dirname(OUTD), "thesis")
PAT = [("vares", "VaReS pattern", "#0072B2", "o"), ("uniform", "Uniform shrinkage", "#D55E00", "s")]
QOI = [("shape", "Shape"), ("warp", "Warp stress"), ("weft", "Weft stress")]
# order plot: stresses only -- the last shape changes
# (<= 1.1 um, < 0.1 %) are below the settle tolerance, so no observed order is reported for shape
QOI_ORDER = [("warp", "Warp stress"), ("weft", "Weft stress")]
NORM = [("mae", r"$L_1$"), ("rms", r"$L_2$")]
QCOL = {"shape": ("#0072B2", "o"), "warp": ("#E69F00", "s"), "weft": ("#009E73", "D")}
R = 2.0
FS = 1.25


def gci(f):
    """Celik et al. 2008 on the three finest values (f ordered coarse -> fine)."""
    p3, p2, p1 = f[-3], f[-2], f[-1]
    e21, e32 = p2 - p1, p3 - p2
    ratio = e32 / e21
    p = abs(np.log(abs(ratio))) / np.log(R)
    ext = (R ** p * p1 - p2) / (R ** p - 1)
    ea = abs(p1 - p2) / abs(p1)
    return dict(phi1=p1, phi2=p2, phi3=p3, ratio=ratio, kind="monotonic" if ratio > 0 else "oscillatory", p=p,
                phi_ext=ext, e_a=ea, e_ext=abs(ext - p1) / abs(ext), gci_fine=FS * ea / (R ** p - 1))


def series(M, v, k):
    tags = list(AC.ROOT[v])
    h = np.array([AC.HNOM[t] for t in tags])
    N = np.array([M[v][t]["n_el_sector"] * M[v][t]["sector_gores"] for t in tags], float)
    return tags, h, N, np.array([M[v][t][k] for t in tags])


def style(fs=9):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": fs,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True, "legend.frameon": False})


def order_plot(M):
    style()
    W, H = 150.0, 118.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    out = {}
    for r_, (nk, nlab) in enumerate(NORM):
        for c_, (q, qlab) in enumerate(QOI_ORDER):
            ax = fig.add_axes([(20 + 68 * c_) / W, (68 - 50 * r_) / H, 50 / W, 38 / H])
            for v, name, col, m in PAT:
                tags, h, N, f = series(M, v, f"{q}_{nk}")
                e = np.abs(np.diff(f)) / np.abs(f[1:])
                ax.plot(N[1:], e, color=col, lw=1.0, zorder=2)
                ax.plot(N[1:], e, m, color=col, mec="k", mew=0.5, ms=5, ls="none", zorder=3)
                out.setdefault(v, {})[f"{q}_{nk}"] = dict(n_elements_finer=N[1:].tolist(), rel_change=e.tolist(),
                                                          observed_p=np.log2(e[:-1] / e[1:]).tolist())
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(2.5e5, 6e7)
            ax.xaxis.set_major_locator(FixedLocator([1e6, 1e7]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: r"$10^{%d}$" % round(np.log10(x))))
            ax.xaxis.set_minor_formatter(NullFormatter())
            lo, hi = ax.get_ylim()
            ax.set_ylim(10 ** np.floor(np.log10(lo)), 10 ** np.ceil(np.log10(hi)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            if r_ == 1:
                ax.set_xlabel("Number of elements")
            if c_ == 0:
                ax.set_ylabel(r"$|f_{h} - f_{h/2}|\,/\,|f_{h/2}|$")
            ax.set_title(f"{qlab}, {nlab}", fontsize=9, pad=3)
            fig.text((20 + 68 * c_ - 15) / W, (68 - 50 * r_ + 41) / H, f"({'abcd'[2 * r_ + c_]})", fontsize=10,
                     fontweight="bold")
    hd = [Line2D([], [], color=col, marker=m, mec="k", mew=0.5, ms=5, lw=1.0, label=name) for _, name, col, m in PAT]
    fig.legend(handles=hd, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    o = os.path.join(OUTD, "fig_seamtip_order_L1L2")
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(f"{o}.{ext}", **kw)
    json.dump(out, open(o + ".json", "w"), indent=1)
    print("[out]", o)


def gci_table(M):
    rows = []
    for v, name, *_ in PAT:
        for q, qlab in QOI:
            for nk, nlab in NORM:
                tags, h, N, f = series(M, v, f"{q}_{nk}")
                g = gci(f)
                sc = 1.0 if q == "shape" else 100.0
                rows.append(dict(pattern=name, measure=qlab, norm=nlab.replace("$", "").replace("_", ""),
                                 unit="mm" if q == "shape" else "%", n_elements_finest=int(N[-1]),
                                 phi_finest=g["phi1"] * sc, phi_ext=g["phi_ext"] * sc, observed_p=g["p"],
                                 convergence=g["kind"], e_a_pct=100 * g["e_a"], gci_fine_pct=100 * g["gci_fine"]))
                print(f"{v:8s} {q:5s} {nk}: {g['phi1'] * sc:.5f} -> ext {g['phi_ext'] * sc:.5f}  p {g['p']:.2f} "
                      f"({g['kind']})  e_a {100 * g['e_a']:.3f} %  GCI_fine {100 * g['gci_fine']:.3f} %")
    o = os.path.join(OUTD, "table_seamtip_gci")
    with open(o + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(method="Celik et al. 2008, three finest rungs, r = 2, Fs = 1.25", rows=rows), open(o + ".json", "w"),
              indent=1)
    with open(o + ".tex", "w") as fh:
        fh.write("\\begin{tabular}{lllrrrlrr}\n\\hline\nPattern & Measure & Norm & $N_\\mathrm{el}$ & $f_1$ & "
                 "$f_\\mathrm{ext}$ & $p$ & type & $\\mathrm{GCI}_\\mathrm{fine}$ (\\%)\\\\\n\\hline\n")
        for r in rows:
            u = " mm" if r["unit"] == "mm" else "\\%"
            fh.write(f"{r['pattern']} & {r['measure']} & $L_{r['norm'][1]}$ & {r['n_elements_finest']:,} & "
                     f"{r['phi_finest']:.4f}{u} & {r['phi_ext']:.4f}{u} & {r['observed_p']:.2f} & {r['convergence']} & "
                     f"{r['gci_fine_pct']:.2f}\\\\\n".replace(",", "{,}"))
        fh.write("\\hline\n\\end{tabular}\n")
    print("[out]", o)
    return rows


def thesis_plot(M):
    os.makedirs(THD, exist_ok=True)
    style(10)
    W, H = 180.0, 150.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    hh = np.geomspace(5.0, 120.0, 300)
    axes, kinds, out = [], set(), {}
    for r_, (nk, nlab) in enumerate(NORM):
        for c_, (v, title, *_) in enumerate(PAT):
            ax = fig.add_axes([(18 + 86 * c_) / W, (88 - 60 * r_) / H, 72 / W, 48 / H])
            axes.append(ax)
            for q, qlab in QOI:
                col, m = QCOL[q]
                tags, h, N, f = series(M, v, f"{q}_{nk}")
                cN = float(N[0] * h[0] ** 2)
                kind, p, se, lim, band, d = classify(h, f)
                g = gci(f)
                out.setdefault(v, {})[f"{q}_{nk}"] = dict(kind=kind, limit=lim, p_fit=float(p[2]), gci_fine=g["gci_fine"])
                if kind == "not converged":
                    continue
                kinds.add(kind)
                if kind == "converging":
                    ax.plot(cN / hh ** 2, (p[0] + p[1] * hh ** p[2]) / lim, color=col, lw=1.1, zorder=2)
                    ax.plot(N, f / lim, m, color=col, mec="k", mew=0.5, ms=6, ls="none", zorder=3)
                else:                                              # flat: markers / mean, no curve
                    ax.plot(N, f / lim, m, mfc="white", mec=col, mew=1.2, ms=6, ls="none", zorder=3)
                ax.errorbar(N[-1], f[-1] / lim, yerr=g["gci_fine"] * f[-1] / lim, fmt="none", ecolor=col, elinewidth=1.0,
                            capsize=2.5, zorder=4)
            ax.axhline(1.0, color="0.35", lw=0.9, ls=(0, (4, 2)), zorder=0)
            ax.set_xscale("log")
            ax.set_xlim(cN / 120.0 ** 2, cN / 5.0 ** 2)
            if r_ == 1:
                ax.set_xlabel("Number of elements")
            if c_ == 0:
                ax.set_ylabel(f"Normalised {nlab} error " + r"$f/f_\infty$")
            ax.text(0.97, 0.94, title, transform=ax.transAxes, ha="right", va="top", fontsize=9)
            fig.text((18 + 86 * c_ - 13) / W, (88 - 60 * r_ + 51) / H, f"({'abcd'[2 * r_ + c_]})", fontsize=11,
                     fontweight="bold")
    lo, hi = min(a.get_ylim()[0] for a in axes), max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(lo, hi)
    hd = [Line2D([], [], color=QCOL[q][0], marker=QCOL[q][1], mec="k", mew=0.5, ms=6, lw=1.1, label=lab) for q, lab in QOI]
    hd += [Line2D([], [], color="0.35", lw=0.9, ls=(0, (4, 2)), label="extrapolated limit"),
           Line2D([], [], color="0.35", marker="|", ms=8, lw=0, label=r"GCI$_\mathrm{fine}$")]
    if "flat" in kinds:
        hd.append(Line2D([], [], color="0.35", marker="o", mfc="white", mew=1.2, ms=6, lw=0, label="no mesh trend"))
    fig.legend(handles=hd, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    o = os.path.join(THD, "fig_seamtip_convergence_thesis")
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(f"{o}.{ext}", **kw)
    json.dump(out, open(o + ".json", "w"), indent=1, default=float)
    print("[out]", o)
    for v in out:
        for k, x in out[v].items():
            print(f"  thesis {v:8s} {k:10s} {x['kind']}")


def main():
    M = json.load(open(AC.CACHE))
    order_plot(M)
    gci_table(M)
    thesis_plot(M)


if __name__ == "__main__":
    main()
