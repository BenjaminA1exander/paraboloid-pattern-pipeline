r"""abaqus_ladder_seamband.py -- field measures of a settled Abaqus/Explicit VaReS run, over the whole gore and with fixed
physical bands along the seams excluded.

Definitions are those of plot_abaqus_sector_vares_asymptote.py (positions = mean over the last 30 s of field frames,
the settle gate's window; stresses = last frame, section-point mean; sigma* = element-centre target):
  apex_err   apex height minus the target apex height (mm) -- a point measure, the same with or without bands
  shape_rms  area-weighted rms normal deviation from the target surface (mm), exact P1 integral per element
  warp_rms, weft_rms   area-weighted rms of sigma / sigma* - 1 (fraction)
Seam band: an element is excluded when its centroid lies within BAND mm of the nearest seam, the distance measured
along the inflated surface's parallel circle, d = (1 - |s|) rho pi / n_gores (s = across-gore position, seams at +-1).
The band is a fixed physical width, the same at every rung, so the measure is one functional of the solution and the
ladder converges to it.  PAPER_BAND = 32 mm was set before the weft data were seen (plot_abaqus_weft_seam_diagnosis.py,
"same pre-set rule"); the appendix reports 0 / 8 / 16 / 32 / 64 mm.
Works for the one-gore sector (flower.npz has n_gore_nodes) and the 24-gore model.
"""
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMP = os.path.normpath(os.path.join(HERE, "..", "..", "temp"))
SECTOR = os.path.join(TEMP, "abaqus_sector_mirror")
FULL = os.path.join(TEMP, "abaqus_prod_tippolar210nest_mirror")
TAGS = ("h100", "h050", "h025", "h012", "h006")
PAPER_BAND = 32.0
BANDS = (0.0, 8.0, 16.0, 32.0, 64.0)


def abs_integral(dd, A):
    """exact integral of |f| over triangles, f linear with nodal values dd (n, 3), areas A: elements where f changes sign
    are split at the zero line; the part on the side of the lone vertex v0 integrates to A v0^3 / (3 (v0-v1)(v0-v2))
    (checked against Monte Carlo, 2026-10-16)."""
    F = A * dd.sum(1) / 3.0
    out = np.abs(F)
    pos = dd > 0
    npos = pos.sum(1)
    lone = np.where(npos == 1, np.argmax(pos, 1), np.argmax(~pos, 1))
    i = np.nonzero((npos == 1) | (npos == 2))[0]
    v0 = dd[i, lone[i]]
    o = np.array([[1, 2], [0, 2], [0, 1]])[lone[i]]
    v1, v2 = dd[i, o[:, 0]], dd[i, o[:, 1]]
    P = A[i] * v0 ** 3 / (3.0 * (v0 - v1) * (v0 - v2))
    out[i] = np.abs(P) + np.abs(F[i] - P)
    return out


def last_job(jd, v="vares"):
    fs = glob.glob(os.path.join(jd, f"cst_stl*_{v}_final.npz"))
    if not fs:
        return None
    return max(fs, key=lambda p: int(re.search(r"stl(\d+)_", os.path.basename(p)).group(1)))


def load(jd, v="vares", force=False):
    """per-element arrays of one settled run.  force=True (2026-10-16 audit; the seam-tip ladder uses it): warp / weft
    deviation from the membrane RESULTANT, SF / (sigma* t) - 1 with SF the Abaqus section force -- the section-point
    mean of S differs from SF / t by up to 30 % where the shell bends across a seam (one-face wrinkling); default False
    keeps the earlier ladders' figures reproducible."""
    npz = last_job(jd, v)
    if npz is None:
        return None
    r = np.load(npz, allow_pickle=True)
    fl = np.load(os.path.join(jd, "flower.npz"))
    sector = "n_gore_nodes" in fl.files
    ng = int(fl["sector_gores"]) if sector else int(fl["n_gores"])
    T = fl["tris"]
    R = float(fl["ring_r"])
    ft, fc = np.asarray(r["frame_t"], float), r["frame_coord"].astype(float)
    w = ft >= ft[-1] - 30.0 - 1e-6
    x = fc[w].mean(0)
    x = x[:int(fl["n_gore_nodes"])] if sector else x[:len(fl["nodes"])]
    Xs = fl["X_spatial"]
    # target surface: even polynomial in r/R through the form-found nodes
    cz = np.linalg.lstsq(np.column_stack([(np.hypot(Xs[:, 0], Xs[:, 1]) / R) ** (2 * j) for j in range(5)]),
                         Xs[:, 2], rcond=None)[0]
    q = np.hypot(x[:, 0], x[:, 1]) / R
    z_t = np.column_stack([q ** (2 * j) for j in range(5)]) @ cz
    dz = np.column_stack([2 * j * q ** (2 * j - 1) if j else 0 * q for j in range(5)]) @ cz / R
    dev = (x[:, 2] - z_t) / np.sqrt(1.0 + dz ** 2)
    area = 0.5 * np.linalg.norm(np.cross(x[T[:, 1]] - x[T[:, 0]], x[T[:, 2]] - x[T[:, 0]]), axis=1)
    dd = dev[T]
    m2 = ((dd ** 2).sum(1) + dd[:, 0] * dd[:, 1] + dd[:, 1] * dd[:, 2] + dd[:, 2] * dd[:, 0]) / 6.0
    ne = int(fl["n_el_per_gore"])
    S = r["S"][:len(T), :2]
    Tt = fl["T"][np.arange(len(T)) % ne][:, :2]
    if force:
        import paraboloid_dish_config as CFG
        S = r["SF"][:len(T), :2] / CFG.T_FABRIC                    # membrane resultant per reference thickness
    # distance of each element centroid from the nearest seam (any gore)
    phic = np.arctan2(Xs[:, 1].mean(), Xs[:, 0].mean())
    c = x[T].mean(1)
    wdg = 2.0 * np.pi / ng
    dl = np.mod(np.arctan2(c[:, 1], c[:, 0]) - phic + 0.5 * wdg, wdg) - 0.5 * wdg
    s = dl / (0.5 * wdg)
    rho = np.hypot(c[:, 0], c[:, 1])
    dseam = (1.0 - np.abs(s)) * rho * np.pi / ng
    es = Xs[T[:ne] % len(Xs)] if sector else Xs[T[:ne]]
    a_s = 0.5 * np.linalg.norm(np.cross(es[:, 1] - es[:, 0], es[:, 2] - es[:, 0]), axis=1)
    heff = float(np.sqrt(4.0 * a_s.sum() / (np.sqrt(3.0) * ne)))
    return dict(job=os.path.basename(npz).replace("_final.npz", ""), frames=int(w.sum()), heff=heff, area=area, m2=m2,
                absd=abs_integral(dd, area), force=bool(force),
                warp=S[:, 0] / Tt[:, 0] - 1.0, weft=S[:, 1] / Tt[:, 1] - 1.0, dseam=dseam, rr=rho / R,
                apex_err=float(x[:, 2].max() - float(fl["apex_z"])), sector=sector)


def measures(e, band):
    """the four measures with elements within `band` mm of a seam excluded (band 0 = whole gore)."""
    k = e["dseam"] >= band
    a = e["area"][k]

    def aw(f):
        return float((a * f[k]).sum() / a.sum())
    return dict(apex_err=e["apex_err"], shape_rms=float(np.sqrt(aw(e["m2"]))), warp_rms=float(np.sqrt(aw(e["warp"] ** 2))),
                weft_rms=float(np.sqrt(aw(e["weft"] ** 2))), area_share=float(a.sum() / e["area"].sum()))
