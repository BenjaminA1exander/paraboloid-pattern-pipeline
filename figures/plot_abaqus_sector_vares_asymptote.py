r"""plot_abaqus_sector_vares_asymptote.py -- mesh convergence of the Abaqus/Explicit inflation of the VaReS pattern on
the nested 100 -> 6.25 mm ladder (one-gore cyclic sector, sector_abaqus_ladder.py), asymptote style.

Data: temp/abaqus_sector_mirror/<tag>/vares_dp (copies of the Euler sector root): every rung settled (30 s-window gate
<= 0.01 mm); positions = mean of the last 30 s of field frames (the gate's window), stress = last frame.  Cross-check:
the 24-gore full model at 100 / 50 mm (temp/abaqus_prod_tippolar210nest_mirror), open markers.
Quantities -- the definitions of the earlier Abaqus ladders (temp/abaqus_vares_cst/plot_dp_convergence.py):
  (a) apex error      apex height minus the target apex height (mm)
  (b) shape rms       area-weighted rms normal deviation from the target surface (even polynomial in r/R through the
                      form-found nodes; exact P1 integral) (mm)
  (c) warp, (d) weft  area-weighted rms of sigma / sigma* - 1 (%), sigma* = element-centre target
Abscissa: h_eff of the spatial gore mesh (as the VaReS chain figures).
Asymptote rule, FIXED BEFORE LOOKING AT THE DATA (no forcing):
  converging      fit f_inf + a h^p (3 params, 5 rungs) has p > 0.5 and se(p) < p, AND the successive changes between
                  rungs shrink in magnitude over the three finest steps -> fit curve, limit, band sqrt(se^2 + rms res^2)
  flat            otherwise, if the successive changes ALTERNATE in sign (scatter about a level) -> mean +- sd band
                  (tightened 2026-10-14 after seeing the weft data, in the conservative direction: the first version
                  called any mixed-sign sequence "flat"; weft falls then rises -- a reversal, not scatter)
  not converged   otherwise -> data only, no line, no band; reported as such
Journal style: letters only, legends and axis labels the only text.  OUT fig_abaqus_sector_vares_asymptote.{pdf,png},
abaqus_sector_vares_asymptote.json
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_vares_shape_asymptote import fit, plt, Line2D, Patch      # noqa: E402

TEMP = os.path.normpath(os.path.join(HERE, "..", "..", "temp"))
SECTOR = os.path.join(TEMP, "abaqus_sector_mirror")
FULL = os.path.join(TEMP, "abaqus_prod_tippolar210nest_mirror")
TAGS = ("h100", "h050", "h025", "h012", "h006")
OUT = os.path.join(HERE, "fig_abaqus_sector_vares_asymptote")
QOIS = [("apex_err", "Apex error (mm)", 1.0), ("shape_rms", "Shape rms deviation (mm)", 1.0),
        ("warp_rms", "Warp stress rms deviation (%)", 100.0), ("weft_rms", "Weft stress rms deviation (%)", 100.0)]
COL = "#0072B2"


def last_job(jd):
    fs = glob.glob(os.path.join(jd, "cst_stl*_vares_final.npz"))
    if not fs:
        return None
    return max(fs, key=lambda p: int(re.search(r"stl(\d+)_", os.path.basename(p)).group(1)))


def metrics(jd):
    npz = last_job(jd)
    if npz is None:
        return None
    r = np.load(npz, allow_pickle=True)
    fl = np.load(os.path.join(jd, "flower.npz"))
    N = int(fl["n_gore_nodes"]) if "n_gore_nodes" in fl.files else len(fl["nodes"])
    ng_model = 1 if "n_gore_nodes" in fl.files else int(fl["n_gores"])
    T = fl["tris"]
    R = float(fl["ring_r"])
    ft, fc = np.asarray(r["frame_t"], float), r["frame_coord"].astype(float)
    w = ft >= ft[-1] - 30.0 - 1e-6
    x = fc[w].mean(0)[:N * ng_model]                          # gore nodes only (the sector's dummy nodes follow)
    Xs = fl["X_spatial"]
    cz = np.linalg.lstsq(np.column_stack([(np.hypot(Xs[:, 0], Xs[:, 1]) / R) ** (2 * j) for j in range(5)]),
                         Xs[:, 2], rcond=None)[0]
    q = np.hypot(x[:, 0], x[:, 1]) / R
    z_t = np.column_stack([q ** (2 * j) for j in range(5)]) @ cz
    dz = np.column_stack([2 * j * q ** (2 * j - 1) if j else 0 * q for j in range(5)]) @ cz / R
    dev = (x[:, 2] - z_t) / np.sqrt(1.0 + dz ** 2)
    area = 0.5 * np.linalg.norm(np.cross(x[T[:, 1]] - x[T[:, 0]], x[T[:, 2]] - x[T[:, 0]]), axis=1)
    dd = dev[T]
    m2 = ((dd ** 2).sum(1) + dd[:, 0] * dd[:, 1] + dd[:, 1] * dd[:, 2] + dd[:, 2] * dd[:, 0]) / 6.0

    def aw(f):
        return float((area * f).sum() / area.sum())
    ne = int(fl["n_el_per_gore"])
    S = r["S"][:len(T), :2]
    Tt = fl["T"][np.arange(len(S)) % ne][:, :2]
    es = Xs[T[:ne] % len(Xs)] if ng_model == 1 else Xs[fl["tris"][:ne]]
    a_s = 0.5 * np.linalg.norm(np.cross(es[:, 1] - es[:, 0], es[:, 2] - es[:, 0]), axis=1)
    heff = float(np.sqrt(4.0 * a_s.sum() / (np.sqrt(3.0) * ne)))
    return dict(job=os.path.basename(npz).replace("_final.npz", ""), frames=int(w.sum()), heff=heff,
                apex_err=float(x[:, 2].max() - float(fl["apex_z"])), shape_rms=float(np.sqrt(aw(m2))),
                warp_rms=float(np.sqrt(aw((S[:, 0] / Tt[:, 0] - 1) ** 2))),
                weft_rms=float(np.sqrt(aw((S[:, 1] / Tt[:, 1] - 1) ** 2))))


def classify(h, y):
    """the rule of the docstring; returns (kind, fit params, se, limit, band)."""
    o = np.argsort(-h)
    h, y = h[o], y[o]
    d = np.diff(y)                                            # coarse -> fine changes
    v, se = fit(h, y)
    fine = np.abs(d[-3:])
    shrinking = bool(np.all(fine[1:] < fine[:-1]))
    resolved = bool(v[2] > 0.5 and se[2] < v[2])
    if resolved and shrinking:
        r = v[0] + v[1] * h ** v[2] - y
        return "converging", v, se, float(v[0]), float(np.hypot(se[0], np.sqrt(np.mean(r ** 2)))), d
    if np.all(d[1:] * d[:-1] < 0):                           # every change reverses the previous one
        return "flat", v, se, float(y.mean()), float(y.std(ddof=1)), d
    return "not converged", v, se, None, None, d


def main():
    rows = {t: metrics(os.path.join(SECTOR, t, "vares_dp")) for t in TAGS}
    rows = {t: m for t, m in rows.items() if m}
    full = {t: metrics(os.path.join(FULL, t, "vares_dp")) for t in ("h100", "h050")}
    full = {t: m for t, m in full.items() if m}
    for t, m in rows.items():
        print(f"sector {t}: {m['job']} ({m['frames']} frames) h_eff {m['heff']:.2f}  apex {m['apex_err']:+.4f} mm  "
              f"shape rms {m['shape_rms']:.4f} mm  warp {100 * m['warp_rms']:.3f} %  weft {100 * m['weft_rms']:.3f} %")
    for t, m in full.items():
        print(f"full   {t}: {m['job']} h_eff {m['heff']:.2f}  apex {m['apex_err']:+.4f}  shape rms {m['shape_rms']:.4f}  "
              f"warp {100 * m['warp_rms']:.3f} %  weft {100 * m['weft_rms']:.3f} %")
    h = np.array([m["heff"] for m in rows.values()])
    out = {"rule": __doc__.split("Asymptote rule")[1].split("Journal")[0].strip(),
           "sector": rows, "full_24_gore": full, "qoi": {}}

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True, "legend.frameon": False})
    W, H = 170.0, 150.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    F = lambda x, y, w, hh: [x / W, y / H, w / W, hh / H]    # noqa: E731
    pos = [F(24, 92, 58, 46), F(106, 92, 58, 46), F(24, 28, 58, 46), F(106, 28, 58, 46)]
    hh = np.linspace(0, 85, 200)
    kinds = set()
    for i, (k, lab, sc) in enumerate(QOIS):
        ax = fig.add_axes(pos[i])
        y = np.array([m[k] for m in rows.values()]) * sc
        kind, v, se, lim, band, d = classify(h, y)
        kinds.add(kind)
        out["qoi"][k] = dict(kind=kind, p=float(v[2]), p_se=float(se[2]), f_inf_fit=float(v[0]), f_inf_se=float(se[0]),
                             limit=lim, band=band, successive_changes=d.tolist(), scale=sc)
        print(f"{k:10s} {kind:14s} values {np.round(y, 5)}  changes {np.round(d, 5)}  fit p {v[2]:.2f} +- {se[2]:.2f}, "
              f"f_inf {v[0]:.5f} +- {se[0]:.5f}" + (f"  -> limit {lim:.5f} +- {band:.5f}" if lim is not None else ""))
        if kind == "converging":
            ax.axhspan(lim - band, lim + band, color="0.85", lw=0, zorder=0)
            ax.axhline(lim, color="0.35", lw=1.0, ls=(0, (4, 2)), zorder=1)
            ax.plot(hh, v[0] + v[1] * hh ** v[2], color=COL, lw=1.1, zorder=2)
        elif kind == "flat":
            ax.axhspan(lim - band, lim + band, color="0.85", lw=0, zorder=0)
            ax.axhline(lim, color="0.35", lw=1.0, ls=(0, (1, 1.5)), zorder=1)
        ax.plot(h, y, "o", color=COL, mec="k", mew=0.6, ms=6, zorder=3)
        if full:
            ax.plot([m["heff"] for m in full.values()], [m[k] * sc for m in full.values()], "o", mfc="none", mec="k",
                    mew=0.9, ms=9, zorder=4)
        ax.set_xlim(0, 85)
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_xlabel(r"Element size $h_{\mathrm{eff}}$ (mm)")
        ax.set_ylabel(lab)
        fig.text(pos[i][0], pos[i][1] + pos[i][3] + 2.5 / H, "(%s)" % "abcd"[i], fontsize=11, fontweight="bold")
    hd = [Line2D([], [], color=COL, marker="o", mec="k", mew=0.6, ms=6, lw=0, label="VaReS pattern, one-gore sector")]
    if full:
        hd.append(Line2D([], [], color="k", marker="o", mfc="none", mew=0.9, ms=9, lw=0, label="24-gore model"))
    if "converging" in kinds:
        hd += [Line2D([], [], color=COL, lw=1.1, label=r"fit $f_\infty + a\,h^p$"),
               Line2D([], [], color="0.35", lw=1.0, ls=(0, (4, 2)), label=r"extrapolated limit $f_\infty$")]
    if "flat" in kinds:
        hd.append(Line2D([], [], color="0.35", lw=1.0, ls=(0, (1, 1.5)), label="mean (no mesh trend)"))
    if kinds & {"converging", "flat"}:
        hd.append(Patch(color="0.85", label="uncertainty"))
    fig.legend(handles=hd, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0), handlelength=2.0)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig("%s.%s" % (OUT, ext), **kw)
    with open(os.path.join(HERE, "abaqus_sector_vares_asymptote.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
