"""
paraboloid_profile.py

Bridge from the single-chamber paraboloid-dish form-finder
(``ParaboloidDish_pressure_JAXFDM.py``) to the VaReS cutting-pattern optimiser --
the pure-dish analogue of ``formfound_profile.py`` (whose loader is specific to
the two-chamber sphere-dish npz; this module reuses its ``MembraneProfile``
machinery unchanged on the paraboloid npz).

From the form-found equilibrium (vertices + per-edge force densities q) it
extracts, for the ONE membrane:

    1. the axisymmetric meridian r(s), z(s) (rings ordered rim -> apex);
    2. the in-membrane tension resultants N_phi(s) (meridional) and N_theta(s)
       (hoop) via the axisymmetric tributary-width rule (see formfound_profile);
    3. physical scaling: geometry to a DIAMETER_MM dish, tension so the
       Laplace-implied FDM pressure maps to P_PHYS -> sigma* = N/t for VaReS.

Self-checks (printed by __main__):
    * Laplace residual p(s) = N_phi/r_m + N_theta/r_t ~ constant along the
      meridian (single chamber carries the full pressure, so unlike the
      differential sphere-dish diaphragm this MUST hold here);
    * p_fdm (Laplace-implied) vs p_true = GAS_K / V(x*) from the saved gas
      constant (stage-1 npz amendment) -- agreement <= ~3%;
    * rim vertical statics N_phi(rim) ~ p*r/(2 sin(alpha)).

Pure numpy + scipy, anaconda3 base env (no jax).
"""
from __future__ import annotations

import os

import numpy as np

import paraboloid_dish_config as CFG
import formfound_profile as ff


# =============================================================================
# Loading
# =============================================================================

def load_paraboloid(npz_path: str = CFG.FORM_NPZ):
    """Load the paraboloid-dish npz into the data-dict layout MembraneProfile
    expects (single membrane: every node is a 'dish' node, env set empty)."""
    d = np.load(npz_path, allow_pickle=True)
    V = np.asarray(d["vertices"], float)
    E = np.asarray(d["edges"], int)
    cat_ids = np.asarray(d["cat_ids"], int)
    Q = np.asarray(d["q_categories"], float)[cat_ids]      # per-edge q
    faces_tri = np.asarray(d["faces_tri"], int)

    r = np.hypot(V[:, 0], V[:, 1])
    z = V[:, 2]
    if "ring_ids" in d.files:                              # saved by stage-1 edit
        rid = np.asarray(d["ring_ids"], int)
    else:                                                  # fallback: (r,z) rounding
        key = np.round(r, 5) + 1j * np.round(z, 5)
        _, rid = np.unique(key, return_inverse=True)
    if "n_theta" in d.files:
        n_theta = int(d["n_theta"])
    else:
        counts = np.bincount(rid)
        n_theta = int(np.median(counts[counts > 1]))

    u, v = E[:, 0], E[:, 1]
    L = np.linalg.norm(V[u] - V[v], axis=1)
    F = Q * L
    is_hoop = rid[u] == rid[v]

    all_nodes = np.arange(V.shape[0])
    return dict(V=V, E=E, Q=Q, L=L, F=F, r=r, z=z, rid=rid, n_theta=n_theta,
                is_hoop=is_hoop, u=u, v=v,
                env_nodes=np.array([], int), dish_nodes=all_nodes,
                dish_tri=faces_tri, faces_tri=faces_tri,
                R=float(d["R"]), DEPTH=float(d["DEPTH"]),
                GAS_K=float(d["GAS_K"]) if "GAS_K" in d.files else np.nan,
                pressure_fixed=(float(d["pressure_fixed"])
                                if "pressure_fixed" in d.files else np.nan))


def _enclosed_volume(coords, faces_tri, z_ref=0.0):
    """Dome volume over the rim plane z=z_ref (projected-area form) -- numpy
    copy of ParaboloidDish_pressure_JAXFDM.enclosed_volume so p_true = GAS_K/V
    uses the form-finder's own volume convention."""
    vi = coords[faces_tri[:, 0]]
    vj = coords[faces_tri[:, 1]]
    vk = coords[faces_tri[:, 2]]
    axy = 0.5 * ((vj[:, 0] - vi[:, 0]) * (vk[:, 1] - vi[:, 1])
                 - (vk[:, 0] - vi[:, 0]) * (vj[:, 1] - vi[:, 1]))
    zc = (vi[:, 2] + vj[:, 2] + vk[:, 2]) / 3.0 - z_ref
    return abs(float(np.sum(axy * zc)))


# =============================================================================
# Extraction factory
# =============================================================================

def extract_paraboloid(npz_path=CFG.FORM_NPZ, *,
                       target_diameter_mm=CFG.DIAMETER_MM, p_phys=CFG.P_PHYS,
                       thickness=CFG.T_FABRIC, trim=CFG.TRIM_DISH):
    """Return (prof, D, meta): the single MembraneProfile with physical scaling.

    tension_scale is calibrated on |p_fdm| (Laplace-implied central pressure of
    THIS membrane -- it carries the full chamber pressure) so it maps to p_phys.
    """
    D = load_paraboloid(npz_path)
    r_max = float(D["r"].max())
    geom_scale = target_diameter_mm / (2.0 * r_max)
    prof = ff.MembraneProfile("dish", D, D["dish_nodes"], D["dish_tri"],
                              thickness=thickness, trim_r_frac=trim)
    p_fdm = prof.implied_pressure_central()
    p_true = D["GAS_K"] / _enclosed_volume(D["V"], D["faces_tri"]) \
        if np.isfinite(D["GAS_K"]) else np.nan
    tension_scale = (p_phys / abs(p_fdm)) * geom_scale
    prof.geom_scale = geom_scale
    prof.tension_scale = tension_scale
    # VaReS target representation: each tension
    # component is a LINEAR closest-fit (least-squares straight line) over the
    # kept rings -- NOT a smoothing spline.  Boundary-condition tension is part
    # of the form and is fitted, not filtered.  Rings beyond S_FIT_MAX_MM are
    # excluded from the fit: corrupted by the pole singularity (the lines still
    # extrapolate to the apex).
    S_FIT_MAX_MM = 2500.0
    infit = prof.s * geom_scale <= S_FIT_MAX_MM
    ok = infit & np.isfinite(prof.Nphi_k)
    prof.spl_Nphi = np.poly1d(np.polyfit(prof.s[ok], prof.Nphi_k[ok], 1))
    ok = infit & np.isfinite(prof.Nt_k)
    prof.spl_Nt = np.poly1d(np.polyfit(prof.s[ok], prof.Nt_k[ok], 1))
    meta = dict(geom_scale=geom_scale, tension_scale=tension_scale,
                p_fdm=p_fdm, p_true=p_true, p_fixed_npz=D["pressure_fixed"],
                p_phys=p_phys, target_diameter_mm=target_diameter_mm,
                r_max_form=r_max,
                depth_mm=geom_scale * D["DEPTH"],
                f_mm=geom_scale * D["R"] ** 2 / (4.0 * D["DEPTH"]))
    return prof, D, meta


# =============================================================================
# Self-check + figure
# =============================================================================

def _laplace_profile(prof, n=60, frac=0.05):
    s = np.linspace(prof.s_lo + frac * (prof.s_hi - prof.s_lo),
                    prof.s_hi - frac * (prof.s_hi - prof.s_lo), n)
    r_t, r_m = prof.principal_radii_at_s(s)
    p = prof.spl_Nphi(s) / r_m + prof.spl_Nt(s) / r_t
    return s, p


def _figure(prof, meta, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[figure] skipped ({e})")
        return
    gs = meta["geom_scale"]
    # layout: left column = meridian profile (top) + residual (below), equal
    # width, combined height (with spacing) equal to the right tension panel.
    fig = plt.figure(figsize=(11.5, 5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.9, 1.0], hspace=0.32, wspace=0.24)
    ax = fig.add_subplot(grid[0, 0])
    ax_res = fig.add_subplot(grid[1, 0], sharex=ax)
    ax_t = fig.add_subplot(grid[:, 1])
    s, r, z = prof.meridian(300)
    # profile + residual presentation: the analytic curve and the form-found
    # rings coincide to sub-mm at this scale, so the agreement is shown
    # quantitatively in the residual panel below.
    rr = np.linspace(0, CFG.R_MM, 200)
    z_ana = lambda rmm: meta["depth_mm"] * (1.0 - (rmm / CFG.R_MM) ** 2)
    ax.plot(rr, z_ana(rr), "-", color="0.25", lw=1.3, label="analytic paraboloid")
    ax.plot(prof.r_k * gs, prof.z_k * gs, "o", mfc="none", mec="crimson", ms=5,
            mew=1.1, label="form-found rings")
    ax.set_ylabel("Height, $z$ (mm)", fontsize=11)
    ax.tick_params(labelsize=10, labelbottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.grid(alpha=0.22)
    ax.text(0.02, 0.06, "(a)", transform=ax.transAxes, fontsize=11)
    # residual: ring height minus analytic paraboloid, in mm
    res = prof.z_k * gs - z_ana(prof.r_k * gs)
    rms = float(np.sqrt(np.mean(res ** 2)))
    ax_res.axhline(0.0, color="0.25", lw=0.8)
    ax_res.plot(prof.r_k * gs, res, "o-", mfc="none", mec="crimson", color="crimson",
                ms=4, lw=0.9, mew=1.0,
                label=f"residual, rms {rms:.2f} mm")
    ax_res.set_xlabel("Radius, $r$ (mm)", fontsize=11)
    ax_res.set_ylabel(r"$z-z_{\rm par}$ (mm)", fontsize=11)
    ax_res.tick_params(labelsize=10)
    ax_res.spines["top"].set_visible(False)
    ax_res.spines["right"].set_visible(False)
    ax_res.legend(fontsize=9, loc="upper right", frameon=False)
    ax_res.grid(alpha=0.22)
    ax_res.text(0.02, 0.10, "(b)", transform=ax_res.transAxes, fontsize=11)

    ax = ax_t
    # plot TENSION resultants N [N/mm] (thickness-independent), not stress
    Nm = np.array([prof.N_at_s(si)[0] for si in s])
    Nt = np.array([prof.N_at_s(si)[1] for si in s])
    smm = s * gs
    ax.plot(smm, Nm, "-", color="navy", lw=2,
            label=r"$N_\phi$ (warp, meridional) linear fit")
    ax.plot(smm, Nt, "-", color="darkorange", lw=2,
            label=r"$N_\theta$ (weft, hoop) linear fit")
    ts = prof.tension_scale
    ax.plot(prof.s * gs, prof.Nphi_k * ts, ".", color="navy", ms=5, alpha=0.5)
    ax.plot(prof.s * gs, prof.Nt_k * ts, ".", color="darkorange", ms=5, alpha=0.5)
    # analytic membrane-theory (Laplace) field of the ideal paraboloid at the
    # nominal calibration pressure -- comparison overlay (user request; the
    # standalone version lives in plot_targets_vs_laplace.py)
    f_mm, p0 = meta["f_mm"], meta["p_phys"]
    r_mm = np.clip(prof.spl_r(s) * gs, 0.0, None)
    u = np.sqrt(1.0 + (r_mm / (2.0 * f_mm)) ** 2)
    ax.plot(smm, p0 * f_mm * u, "--", color="navy", lw=1.1, alpha=0.8,
            label=r"Laplace $N_\phi$")
    ax.plot(smm, p0 * 2.0 * f_mm * u * (1.0 - 1.0 / (2.0 * u ** 2)), "--",
            color="darkorange", lw=1.1, alpha=0.8,
            label=r"Laplace $N_\theta$")
    ax.set_xlabel("Meridian arc length, $s$ (mm)", fontsize=11)
    ax.set_ylabel("Tension, $N$ (N/mm)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.22)
    ax.text(0.02, 0.03, "(c)", transform=ax.transAxes, fontsize=11)
    ax.text(0.09, 0.03, "rim", transform=ax.transAxes, fontsize=9, color="0.45")
    ax.text(0.98, 0.03, "pole", transform=ax.transAxes, fontsize=9, color="0.45",
            ha="right")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.115)
    fig.savefig(out_png, dpi=200)
    print(f"[figure] saved {out_png}")


if __name__ == "__main__":
    prof, D, meta = extract_paraboloid()
    print(f"Loaded {CFG.FORM_NPZ}")
    print(f"  {D['V'].shape[0]} nodes, {D['E'].shape[0]} edges, "
          f"n_theta={D['n_theta']}, R={D['R']}, DEPTH={D['DEPTH']}")
    print(f"  geom_scale={meta['geom_scale']:.2f} mm/unit  "
          f"tension_scale={meta['tension_scale']:.3f} N/mm-per-unit")
    print(f"  physical: D={meta['target_diameter_mm']:.0f} mm  "
          f"depth={meta['depth_mm']:.0f} mm  f={meta['f_mm']:.0f} mm  "
          f"p={meta['p_phys']} MPa  t={prof.t} mm")
    # ---- target table (linear-fit tension) --------------------------------------
    ss = np.linspace(prof.s_lo, prof.s_hi, 13)
    print(f"\n  {'s_mm':>8s} {'r_mm':>8s} {'z_mm':>8s} {'N_phi':>8s} "
          f"{'N_theta':>8s} {'sig_m':>8s} {'sig_t':>8s}")
    for si in ss:
        Np, Nt = prof.N_at_s(si)
        sm_, st_, _ = prof.stress_at_s(si)
        print(f"  {si*prof.geom_scale:8.1f} {float(prof.spl_r(si))*prof.geom_scale:8.1f} "
              f"{float(prof.spl_z(si))*prof.geom_scale:8.1f} {Np:8.3f} {Nt:8.3f} "
              f"{sm_:8.2f} {st_:8.2f}")

    # ---- physical stress range --------------------------------------------------
    ss = np.linspace(prof.s_lo, prof.s_hi, 100)
    sig_m = np.array([prof.stress_at_s(si)[0] for si in ss])
    sig_t = np.array([prof.stress_at_s(si)[1] for si in ss])
    print(f"[info] sigma* range: warp/meridional [{sig_m.min():.1f}, "
          f"{sig_m.max():.1f}] MPa, weft/hoop [{sig_t.min():.1f}, "
          f"{sig_t.max():.1f}] MPa  (NPL regime ~50-100 MPa)")

    _figure(prof, meta, os.path.join(CFG.HERE, "paraboloid_targets_extract.png"))
