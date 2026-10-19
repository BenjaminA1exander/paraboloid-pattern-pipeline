r"""plot_seamtip_vares_vs_uniform.py -- VaReS-optimised vs uniform-shrinkage cutting pattern, measured against the EXACT
(analytic) paraboloid membrane solution, on the seam- and tip-resolved Abaqus sector ladder.

Exact solution (closed form; paraboloid_profile.py):  surface z = f - r^2 / (4 f), f = 1250 mm, D = 5000 mm;
pressure p = 0.05 MPa; membrane (Laplace) resultants  N_phi = p f u,  N_theta = 2 p f u (1 - 1/(2 u^2)),
u = sqrt(1 + (r / 2f)^2).  warp = meridional (phi), weft = hoop (theta) -- the decks' material axes.  The decks'
form-found target sigma* = N*/t (t = 0.9 mm) agrees with the exact field to L1 0.09 / 0.07 % (warp / weft), > 1 % only
in the outer 2 % of the radius (checked 2026-10-16); its surface lies within 3 um of the paraboloid.
Per element (settled run; positions = mean of the last 30 s of frames, forces = last frame; abaqus_ladder_seamband.load
conventions):
  shape   normal distance of the inflated surface from the exact paraboloid, d = (z - z_p(r)) / u (first order), linear
          per element: L2 from the exact P1 integral of d^2, L1 from the exact integral of |d| (sign change split)
  stress  MEMBRANE RESULTANT deviation e = N / N_exact(r) - 1 at the element centroid radius, N = Abaqus section force
          SF (through-thickness integral; 2026-10-16 audit: the section-point MEAN stress differs from SF / t by up to
          30 % in ~1 % of the elements -- seam bending, one-face wrinkling -- while SF is what balances the pressure and is
          what Laplace's solution gives; thickness is not updated, median SF / S = 0.900 mm)
Norms over the whole gore, area-weighted (deformed areas): L1 = sum a |e| / sum a, L2 = sqrt(sum a e^2 / sum a).

Rungs: LEVELS (default L0-L3 both patterns).  --with-l4 adds h006: VaReS from the DelftBlue mirror, uniform from
UNIFORM_L4 (set it to the mirror of whichever Euler chain settles: abaqus_sector_seamtip_v4[_b] mirror).  Every rung's
measures and radial profiles are cached in <OUT>/vares_vs_uniform_cache.json (+ per-element maps of the finest rung in
.npz), so adding L4 loads only the new rungs.  The figures always use the finest rung present for BOTH patterns.

OUT results/abaqus_ladder_seamtip/vares_vs_uniform/
  fig_vu_profiles.{pdf,png}     (a) shape, (b) warp, (c) weft deviation vs r/R, finest rung: area-weighted mean
                                (line) and 5-95 % range across the gore (band), VaReS vs uniform
  fig_vu_maps.{pdf,png}         the same three fields on the full dish (sector repeated x24), VaReS | uniform
  fig_vu_convergence.{pdf,png}  warp and weft L1 / L2 vs number of elements (log), both patterns; pre-set asymptote
                                rule (classify) -> fit + extrapolated limit where "converging"; GCI_fine bar (Celik 2008)
  vu_summary.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_seamtip_asymptote_combined as AC                         # noqa: E402
from plot_abaqus_sector_vares_asymptote import classify              # noqa: E402
from plot_vares_shape_asymptote import plt, Line2D                   # noqa: E402

F_MM, P_MPA, T_MM, R_MM = 1250.0, 0.05, 0.9, 2500.0
WITH_L4 = "--with-l4" in sys.argv
UNIFORM_L4 = os.path.join(AC.TEMP, "abaqus_sector_seamtip_v4_mirror")       # fill in when the Euler chain settles
LEVELS = {"vares": ["h100", "h050", "h025", "h012"] + (["h006"] if WITH_L4 else []),
          "uniform": ["h100", "h050", "h025", "h012"] + (["h006"] if WITH_L4 else [])}
ROOT = {"vares": dict(AC.ROOT["vares"]), "uniform": dict(AC.ROOT["uniform"], h006=UNIFORM_L4)}
PAT = [("vares", "VaReS pattern", "#0072B2", "o", "-"), ("uniform", "Uniform shrinkage", "#D55E00", "s", (0, (5, 2)))]
OUTD = os.path.join(HERE, "results", "abaqus_ladder_seamtip", "vares_vs_uniform")
CACHE = os.path.join(OUTD, "vares_vs_uniform_cache.json")
RBINS = np.linspace(0.0, 1.0, 41)
# maps colour scale (2026-10-16, user picked from the options sheet colour_scale_options.png): magma -- perceptually
# uniform, colour-blind safe, lightness monotonic (dark = negative, light = positive), so the sign survives in greyscale;
# zero is a mid-tone, read off the colour bar
CMAP = plt.get_cmap("magma")


MEASURE_VERSION = 2          # 2: stress from section forces SF, shape L1 = exact integral of |d| (cache invalidation)


def exact_sigma(r):
    u = np.sqrt(1.0 + (r / (2.0 * F_MM)) ** 2)
    return np.c_[P_MPA * F_MM * u, 2.0 * P_MPA * F_MM * u * (1.0 - 1.0 / (2.0 * u ** 2))] / T_MM


abs_integral = AC.SB.abs_integral


def element_fields(v, t):
    jd = os.path.join(ROOT[v][t], t, f"{v}_dp")
    npz = AC.SB.last_job(jd, v)
    if npz is None:
        return None
    if t in AC.SETTLED.get(v, {}):
        k = int(os.path.basename(npz).split("_")[1][3:])
        assert k == AC.SETTLED[v][t], f"{jd}: last import stl{k}, settled one is stl{AC.SETTLED[v][t]}"
    r = np.load(npz, allow_pickle=True)
    fl = np.load(os.path.join(jd, "flower.npz"))
    T, ne, n = fl["tris"], int(fl["n_el_per_gore"]), int(fl["n_gore_nodes"])
    ft, fc = np.asarray(r["frame_t"], float), r["frame_coord"].astype(float)
    x = fc[ft >= ft[-1] - 30.0 - 1e-6].mean(0)[:n]
    rn = np.hypot(x[:, 0], x[:, 1])
    dev = (x[:, 2] - (F_MM - rn ** 2 / (4 * F_MM))) / np.sqrt(1.0 + (rn / (2 * F_MM)) ** 2)   # normal distance
    a = 0.5 * np.linalg.norm(np.cross(x[T[:, 1]] - x[T[:, 0]], x[T[:, 2]] - x[T[:, 0]]), axis=1)
    dd = dev[T]
    m2 = ((dd ** 2).sum(1) + dd[:, 0] * dd[:, 1] + dd[:, 1] * dd[:, 2] + dd[:, 2] * dd[:, 0]) / 6.0
    Xs = fl["X_spatial"]
    cs = Xs[T[:ne] % len(Xs)].mean(1)
    rc = np.hypot(cs[:, 0], cs[:, 1])
    sx = exact_sigma(rc)
    chk = fl["T"][:ne, :2] / sx                                     # deck target / exact: the thickness check
    assert np.all(np.abs(np.median(chk, 0) - 1.0) < 5e-3), f"target vs exact sigma: median ratio {np.median(chk, 0)}"
    Nx = sx * T_MM                                                  # exact membrane resultants (N/mm)
    SF = r["SF"][:ne, :2]                                           # Abaqus section forces, material (warp, weft) axes
    return dict(job=os.path.basename(npz).replace("_final.npz", ""), a=a, m2=m2, absd=abs_integral(dd, a),
                sdev=dd.mean(1), q=rc / R_MM, warp=SF[:, 0] / Nx[:, 0] - 1.0, weft=SF[:, 1] / Nx[:, 1] - 1.0, ne=ne,
                ng=int(fl["sector_gores"]), tri_xy=cs[:, :2], target_vs_exact=np.median(chk, 0).tolist())


def summarise(e):
    a = e["a"]
    A = a.sum()
    out = dict(job=e["job"], n_el=e["ne"] * e["ng"], target_vs_exact=e["target_vs_exact"], version=MEASURE_VERSION,
               shape_L1=float(e["absd"].sum() / A), shape_L2=float(np.sqrt((a * e["m2"]).sum() / A)))
    for k in ("warp", "weft"):
        out[f"{k}_L1"] = float((a * np.abs(e[k])).sum() / A)
        out[f"{k}_L2"] = float(np.sqrt((a * e[k] ** 2).sum() / A))
    b = np.clip(np.digitize(e["q"], RBINS) - 1, 0, len(RBINS) - 2)
    prof = {}
    for k, f in (("shape", e["sdev"]), ("warp", e["warp"]), ("weft", e["weft"])):
        mean, lo, hi = [], [], []
        for j in range(len(RBINS) - 1):
            m = b == j
            if not m.any():
                mean.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                continue
            o = np.argsort(f[m])
            c = np.cumsum(a[m][o]) / a[m].sum()
            mean.append(float((a[m] * f[m]).sum() / a[m].sum()))
            lo.append(float(f[m][o][np.searchsorted(c, 0.05)]))
            hi.append(float(f[m][o][min(np.searchsorted(c, 0.95), m.sum() - 1)]))
        prof[k] = dict(mean=mean, p05=lo, p95=hi)
    out["profile"] = prof
    return out


def collect():
    os.makedirs(OUTD, exist_ok=True)
    C = json.load(open(CACHE)) if os.path.exists(CACHE) and "--reload" not in sys.argv else {}
    for v, *_ in PAT:
        C.setdefault(v, {})
        for t in LEVELS[v]:
            mp = os.path.join(OUTD, f"map_{v}_{t}.npz")
            if t in C[v] and os.path.exists(mp) and C[v][t].get("version") == MEASURE_VERSION:
                continue
            e = element_fields(v, t)
            if e is None:
                raise SystemExit(f"[vu] {v} {t}: no settled run under {ROOT[v][t]} -- mirror it first")
            C[v][t] = summarise(e)
            np.savez(mp, xy=e["tri_xy"], sdev=e["sdev"], warp=e["warp"], weft=e["weft"])
            with open(CACHE, "w") as fh:
                json.dump(C, fh, indent=1)
            print(f"[vu] {v} {t}: {e['job']}  shape L1 {C[v][t]['shape_L1']:.4f} mm  warp L1 {100 * C[v][t]['warp_L1']:.3f} %"
                  f"  weft L1 {100 * C[v][t]['weft_L1']:.3f} %  (target/exact {np.round(e['target_vs_exact'], 4)})", flush=True)
    return C


def style(fs=9):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": fs,
                         "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
                         "ytick.right": True, "legend.frameon": False})


def finest_common():
    return [t for t in LEVELS["vares"] if t in LEVELS["uniform"]][-1]


def fig_profiles(C):
    style()
    t = finest_common()
    W, H = 180.0, 66.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    qc = 0.5 * (RBINS[1:] + RBINS[:-1])
    labs = [("shape", "Shape deviation (mm)", 1.0), ("warp", "Warp stress deviation (%)", 100.0),
            ("weft", "Weft stress deviation (%)", 100.0)]
    for i, (k, lab, sc) in enumerate(labs):
        ax = fig.add_axes([(17 + 58 * i) / W, 20 / H, 42 / W, 40 / H])
        for v, name, col, m, ls in PAT:
            p = C[v][t]["profile"][k]
            ax.fill_between(qc, np.array(p["p05"]) * sc, np.array(p["p95"]) * sc, color=col, alpha=0.18, lw=0)
            ax.plot(qc, np.array(p["mean"]) * sc, color=col, lw=1.3, ls=ls)
        ax.axhline(0.0, color="0.3", lw=0.7)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlabel("Radius $r/R$")
        ax.set_ylabel(lab)
        fig.text((17 + 58 * i - 13) / W, 62 / H, f"({'abc'[i]})", fontsize=10, fontweight="bold")
    hd = [Line2D([], [], color=c, lw=1.3, ls=ls, label=n) for _, n, c, _, ls in PAT]
    hd.append(plt.matplotlib.patches.Patch(color="0.6", alpha=0.35, label="5–95 % range over the gore"))
    fig.legend(handles=hd, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    o = os.path.join(OUTD, "fig_vu_profiles")
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(f"{o}.{ext}", **kw)
    print("[out]", o, f"(rung {t})")


def sector_image(xy, f, npx=700):
    """full-dish raster of a sector field: each pixel is rotated into the sector (cyclic symmetry, 24 gores) and takes
    the value of the nearest element centroid -- gap-free and fast."""
    from scipy.spatial import cKDTree
    wdg = 2 * np.pi / 24
    phic = np.arctan2(xy[:, 1].mean(), xy[:, 0].mean())
    g = np.linspace(-R_MM, R_MM, npx)
    X, Y = np.meshgrid(g, g)
    rr, ph = np.hypot(X, Y), np.arctan2(Y, X)
    ph = phic + np.mod(ph - phic + 0.5 * wdg, wdg) - 0.5 * wdg
    _, k = cKDTree(xy).query(np.c_[(rr * np.cos(ph)).ravel(), (rr * np.sin(ph)).ravel()])
    img = f[k].reshape(X.shape)
    return np.ma.masked_where(rr > R_MM, img)


def fig_maps(C):
    """maps (CMAP blue-white-orange, shared limits per field) + ONE compact convergence panel at the right: warp and weft L1 of
    both patterns, each normalised by its own extrapolated limit f_inf (pre-set rule; finest value if not converging).
    2026-10-17 (GCI audit): f_inf is the Richardson extrapolation of the three finest rungs (Celik 2008, the phi_ext of
    Table 5's GCI_fine), not the least-squares fit over all rungs -- that fit (p ~ 1, pulled by the pre-asymptotic h100
    rung) put f_inf outside the GCI band in 7 of 8 measures, and the VaReS h006 rung sides with the GCI.  Drawn (user,
    2026-10-17): coarsest -> 2nd coarsest as a straight line, then the GCI extrapolation f_inf + a h^p through the three
    finest rungs out to the limit, so the kink at the 2nd coarsest rung marks where asymptotic convergence begins."""
    style()
    t = finest_common()
    W, H = 180.0, 108.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    labs = [("sdev", "shape", "Shape deviation (mm)", 1.0), ("warp", "warp", "Warp stress deviation (%)", 100.0),
            ("weft", "weft", "Weft stress deviation (%)", 100.0)]
    M = {v: np.load(os.path.join(OUTD, f"map_{v}_{t}.npz")) for v, *_ in PAT}
    lims = {}
    for j, (k, kc, lab, sc) in enumerate(labs):
        allv = np.concatenate([np.abs(M[v][k] * sc) for v, *_ in PAT])
        lim = float(np.percentile(allv, 97.5))                    # contrast: saturate the top 2.5 % (apex spikes)
        lims[k] = lim
        for i, (v, name, *_) in enumerate(PAT):
            ax = fig.add_axes([(8 + 41 * j) / W, (62 - 41 * i) / H, 38 / W, 38 / H])
            img = sector_image(M[v]["xy"], M[v][k] * sc)
            ax.imshow(img, origin="lower", extent=(-R_MM, R_MM, -R_MM, R_MM), cmap=CMAP, vmin=-lim, vmax=lim,
                      interpolation="nearest")
            ax.set_aspect("equal")
            ax.axis("off")
            if j == 0:
                fig.text(3 / W, (62 - 41 * i + 19) / H, name, rotation=90, va="center", fontsize=8)
        cax = fig.add_axes([(11 + 41 * j) / W, 13 / H, 32 / W, 2.2 / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-lim, lim), cmap=CMAP), cax=cax,
                          orientation="horizontal", extend="both")
        cb.set_label(lab, fontsize=7.5, labelpad=1)
        cb.ax.tick_params(labelsize=7)
        fig.text((8 + 41 * j) / W, 102 / H, f"({'abc'[j]})", fontsize=10, fontweight="bold")
    # (d) convergence: warp and weft L1, both patterns, normalised by the extrapolated limit
    # right column spans the two map rows: axes top = map top (100 mm), legend centred beneath, ending near the map bottom
    AX = (143.5, 58.0, 32.0, 42.0)
    ax = fig.add_axes([AX[0] / W, AX[1] / H, AX[2] / W, AX[3] / H])
    QS = [("warp", "warp", "o", "-"), ("weft", "weft", "D", (0, (4, 2)))]
    hd = []
    for v, name, col, _, _ in PAT:
        for kc, klab, m, ls in QS:
            tags = LEVELS[v]
            h = np.array([AC.HNOM[tt] for tt in tags])
            N = np.array([C[v][tt]["n_el"] for tt in tags], float)
            f = np.array([C[v][tt][f"{kc}_L1"] for tt in tags])
            kind = classify(h, f)[0]                                # pre-set rule decides converging / not
            g = gci(f)                                              # limit: Richardson on the three finest (Table 5)
            ref = g["ext"] if kind == "converging" else f[-1]
            cN = float(N[0] * h[0] ** 2)
            if kind == "converging":
                ax.plot(N[:-2], f[:-2] / ref, color=col, lw=1.0, ls=ls, zorder=2)   # pre-asymptotic: straight line
                a = (f[-1] - g["ext"]) / h[-1] ** g["p"]            # GCI extrapolation f_inf + a h^p through the
                assert np.allclose(g["ext"] + a * h[-3:] ** g["p"], f[-3:], rtol=1e-9), f"{v} {kc}: curve misses rungs"
                hc = np.geomspace(5.0, h[-3], 200)                  # three finest rungs, out to the limit
                ax.plot(cN / hc ** 2, (g["ext"] + a * hc ** g["p"]) / ref, color=col, lw=1.0, ls=ls, zorder=2)
            else:
                ax.plot(N, f / ref, color=col, lw=1.0, ls=ls, zorder=2)
            mfc = col if kc == "warp" else "white"
            mec = "k" if kc == "warp" else col
            ax.plot(N, f / ref, m, color=col, mfc=mfc, mec=mec, mew=0.8, ms=5, ls="none", zorder=3)
            hd.append(Line2D([], [], color=col, marker=m, mfc=mfc, mec=mec, mew=0.9, ms=6, lw=1.2, ls=ls,
                             label=f"{name.split()[0]} {klab}"))
    ax.axhline(1.0, color="0.35", lw=0.8, ls=(0, (1, 1.5)), zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(cN / 120.0 ** 2, cN / 5.0 ** 2)
    ax.set_xlabel("Number of elements", fontsize=7.5)
    ax.set_ylabel(r"$L_1$ deviation $f/f_\infty$", fontsize=7.5, labelpad=1)
    ax.tick_params(labelsize=7)
    fig.text(131 / W, 102 / H, "(d)", fontsize=10, fontweight="bold")          # same baseline as (a)-(c)
    hd.append(Line2D([], [], color="0.35", lw=1.0, ls=(0, (1, 1.5)), label="extrapolated limit"))
    fig.legend(handles=hd, loc="upper center", ncol=1, fontsize=8, bbox_to_anchor=((AX[0] + AX[2] / 2) / W, 45.5 / H),
               handlelength=3.0, labelspacing=0.45, borderaxespad=0.0, borderpad=0.0, handletextpad=0.6)  # centred under (d)
    o = os.path.join(OUTD, "fig_vu_maps")
    for ext, kw in (("pdf", {"dpi": 400}), ("png", {"dpi": 400})):
        fig.savefig(f"{o}.{ext}", **kw)
    print("[out]", o, f"(rung {t}); colour limits (97.5th pct) " + ", ".join(f"{k} {x:.3g}" for k, x in lims.items()))


def gci(f):
    p3, p2, p1 = f[-3], f[-2], f[-1]
    ratio = (p3 - p2) / (p2 - p1)
    p = abs(np.log(abs(ratio))) / np.log(2.0)
    return dict(p=p, kind="monotonic" if ratio > 0 else "oscillatory", gci=1.25 * abs(p1 - p2) / abs(p1) / (2 ** p - 1),
                ext=(2 ** p * p1 - p2) / (2 ** p - 1))


def fig_convergence(C):
    style()
    W, H = 150.0, 118.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    out = {}
    hh = np.geomspace(5.0, 120.0, 300)
    for r_, (nk, nlab) in enumerate((("L1", r"$L_1$"), ("L2", r"$L_2$"))):
        for c_, (k, klab) in enumerate((("warp", "Warp stress"), ("weft", "Weft stress"))):
            ax = fig.add_axes([(20 + 68 * c_) / W, (68 - 50 * r_) / H, 50 / W, 38 / H])
            for v, name, col, m, ls in PAT:
                tags = LEVELS[v]
                h = np.array([AC.HNOM[t] for t in tags])
                N = np.array([C[v][t]["n_el"] for t in tags], float)
                f = np.array([C[v][t][f"{k}_{nk}"] for t in tags]) * 100
                kind, pf, se, lim, band, d = classify(h, f)
                g = gci(f)
                out.setdefault(v, {})[f"{k}_{nk}"] = dict(rungs=tags, values_pct=f.tolist(), kind=kind, limit_pct=lim,
                                                          p_fit=float(pf[2]), gci_fine=g["gci"], p_obs=g["p"],
                                                          convergence=g["kind"])
                cN = float(N[0] * h[0] ** 2)
                if kind == "converging":
                    ax.plot(cN / hh ** 2, pf[0] + pf[1] * hh ** pf[2], color=col, lw=1.0, ls=ls, zorder=2)
                    ax.axhline(lim, color=col, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
                else:
                    ax.plot(N, f, color=col, lw=0.8, ls=ls, zorder=2)
                ax.plot(N, f, m, color=col, mec="k", mew=0.5, ms=5, ls="none", zorder=3)
                ax.errorbar(N[-1], f[-1], yerr=g["gci"] * f[-1], fmt="none", ecolor="k", elinewidth=0.9, capsize=2.5,
                            zorder=4)
                print(f"  {v:8s} {k} {nk}: {np.round(f, 4)} %  rule {kind}" + (f", limit {lim:.4f} %" if lim else "") +
                      f"; GCI_fine {100 * g['gci']:.2f} % (p {g['p']:.2f}, {g['kind']})")
            ax.set_xscale("log")
            ax.set_xlim(cN / 120.0 ** 2, cN / 5.0 ** 2)
            if r_ == 1:
                ax.set_xlabel("Number of elements")
            ax.set_ylabel(f"{klab} deviation, {nlab} (%)")
            ax.set_ylim(0, None)
            fig.text((20 + 68 * c_ - 15) / W, (68 - 50 * r_ + 41) / H, f"({'abcd'[2 * r_ + c_]})", fontsize=10,
                     fontweight="bold")
    hd = [Line2D([], [], color=c, marker=m, mec="k", mew=0.5, ms=5, lw=1.0, ls=ls, label=n) for _, n, c, m, ls in PAT]
    hd += [Line2D([], [], color="0.4", lw=0.8, ls=(0, (1, 1.5)), label="extrapolated limit"),
           Line2D([], [], color="k", marker="|", ms=8, lw=0, label=r"GCI$_\mathrm{fine}$")]
    fig.legend(handles=hd, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    o = os.path.join(OUTD, "fig_vu_convergence")
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(f"{o}.{ext}", **kw)
    print("[out]", o)
    return out


def write_table(C, conv):
    """LaTeX table (elsarticle/cas booktabs) of the finest-rung norms + GCI_fine, regenerated with the figures."""
    t = finest_common()
    BS = "\\"
    rows = [("shape", r"Shape, $d$ (mm)", 1.0, "%.2f"), ("warp", r"Warp, $e_{\mathrm{w}}$ (\%)", 100.0, "%.1f"),
            ("weft", r"Weft, $e_{\mathrm{f}}$ (\%)", 100.0, "%.2f")]
    L = [r"\begin{tabular*}{\tblwidth}{@{}LCCCCC@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{VaReS} & \multicolumn{2}{c}{Uniform} & \\",
         r"Measure & $L_1$ & $L_2$ & $L_1$ & $L_2$ & Ratio $L_1$ \\", r"\midrule"]
    for k, lab, sc, fmt in rows:
        v1, v2 = C["vares"][t][f"{k}_L1"] * sc, C["vares"][t][f"{k}_L2"] * sc
        u1, u2 = C["uniform"][t][f"{k}_L1"] * sc, C["uniform"][t][f"{k}_L2"] * sc
        L.append(f"{lab} & {fmt % v1} & {fmt % v2} & {fmt % u1} & {fmt % u2} & {u1 / v1:.1f} " + BS * 2)
    L += [r"\midrule", r"\multicolumn{6}{@{}l}{$\mathrm{GCI}_{\mathrm{fine}}$ (\%), $L_1$ / $L_2$} \\"]
    for k, lab in (("warp", "Warp"), ("weft", "Weft")):
        g = lambda v, n: 100 * conv[v][f"{k}_{n}"]["gci_fine"]                 # noqa: E731
        L.append(f"{lab} & {BS}multicolumn{{2}}{{c}}{{{g('vares', 'L1'):.1f} / {g('vares', 'L2'):.1f}}} & "
                 f"{BS}multicolumn{{2}}{{c}}{{{g('uniform', 'L1'):.1f} / {g('uniform', 'L2'):.1f}}} & " + BS * 2)
    L += [r"\bottomrule", r"\end{tabular*}"]
    o = os.path.join(OUTD, "table_vu.tex")
    with open(o, "w") as fh:
        fh.write(f"% generated by plot_seamtip_vares_vs_uniform.py, finest rung {t} "
                 f"({C['vares'][t]['n_el']:,} elements, full dish) -- do not edit by hand\n" + "\n".join(L) + "\n")
    print("[out]", o)


def main():
    C = collect()
    fig_profiles(C)
    fig_maps(C)
    conv = fig_convergence(C)
    write_table(C, conv)
    t = finest_common()
    summ = {"reference": f"analytic Laplace paraboloid f {F_MM} mm, p {P_MPA} MPa, t {T_MM} mm", "levels": LEVELS,
            "finest_common": t, "convergence": conv,
            "finest": {v: {k: C[v][t][k] for k in ("job", "n_el", "shape_L1", "shape_L2", "warp_L1", "warp_L2",
                                                   "weft_L1", "weft_L2", "target_vs_exact")} for v, *_ in PAT}}
    with open(os.path.join(OUTD, "vu_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1, default=float)
    for v, *_ in PAT:
        s = summ["finest"][v]
        print(f"[{t}] {v:8s} shape L1 {s['shape_L1']:.3f} mm L2 {s['shape_L2']:.3f} mm | warp L1 {100 * s['warp_L1']:.2f} % "
              f"L2 {100 * s['warp_L2']:.2f} % | weft L1 {100 * s['weft_L1']:.2f} % L2 {100 * s['weft_L2']:.2f} %")


if __name__ == "__main__":
    main()
