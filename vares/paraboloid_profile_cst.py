r"""paraboloid_profile_cst.py -- VaReS target taken DIRECTLY from the 4626-node form-found net:
self-stress-projected CST state + nodal virial + quadratic MLS (8-edge patch), cyclic-averaged
over the 24 gores and mirror-averaged about the gore centreline.

Drop-in for paraboloid_profile.extract_paraboloid(): extract_paraboloid_cst() returns
(prof, D, meta) where prof has the attributes the gore builder and the VaReS worker use
(spl_r, spl_z, s_lo, s_hi, geom_scale, t, name, build_gore_gmsh, s_of_pos, sigma_field).

SOURCE (handoff: temp/paraboloid_edge_constq_u4626_odt_qual1e-4_warm/HANDOFF_stress_extraction.md)
    STATE_NPZ  jaxfdm_paraboloid_dish_projected_cst.npz   (CST projection, n_negative = 0)
    tensor     local_stress_recovery.nodal_virial, then quadratic MLS ("mls2") with
               rho = 8 x median edge, Gaussian weight sigma = rho/2, interior nodes only --
               the estimator of local_stress_recovery.recover, evaluated at an arbitrary
               point instead of at a node (so the rim band is a one-sided fit, as at the
               rim nodes)
GEOMETRY   the gore meridian is an even polynomial z = sum c_k (r/R)^(2k), k = 0..4, least-squares
           fitted to ALL net nodes (the axisymmetric part of the form-found shape; the
           exact paraboloid is not used).  The gore runs to the FULL TIP (r = 0, no apex
           trim; user decision 2026-10-04) -- longer than the linear-fit gore, which stops at
           the CFG.TRIM_DISH ring (r = 117 mm).  CST_R_TIP_MM overrides (e.g. 117.155 to mesh
           the linear-fit leaf).
TARGET     on a fixed grid (s, eta), eta = gore-relative angle / half gore angle in [-1, 1]:
             sigma(s, eta) = mean over the 24 gores of (m.S.m, h.S.h, m.S.h)  at
                             phi = theta (2k + 1 + eta)
           then mirror-averaged: normals (f(eta) + f(-eta))/2, shear (f(eta) - f(-eta))/2.
           m = unit meridian tangent toward the apex (= leaf +x = warp), h = +phi (= leaf +y =
           weft).  Scaled to the physical pressure: sigma = N (P_PHYS / p_form) geom_scale / t,
           p_form = GAS_K / V(x*).  The grid is cached in temp/meshstudy_vares_cst/target_grid.npz
           and interpolated bicubically, so the target does not depend on the gore mesh.
Exact membrane statics is used ONLY for scoring (__main__), never in the target.

USAGE (base anaconda env)
    python paraboloid_profile_cst.py     # build/cache the grid, full-gore error check, figure
Outputs:
    temp/meshstudy_vares_cst/target_grid.npz
    fig_paraboloid_cst_target_check.{pdf,png}
"""
import os
import sys

import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paraboloid_dish_config as CFG                 # noqa: E402
import formfound_profile as ff                        # noqa: E402

STATE_NPZ = os.environ.get("CST_STATE_NPZ", os.path.join(
    CFG.TEMP_DIR, "paraboloid_edge_constq_u4626_odt_qual1e-4_warm",
    "jaxfdm_paraboloid_dish_projected_cst.npz"))
CACHE_DIR = os.path.join(CFG.TEMP_DIR, "meshstudy_vares_cst")
CACHE_NPZ = os.path.join(CACHE_DIR, "target_grid.npz")

RHO_EDGES = 8.0          # MLS patch radius in median edge lengths (handoff)
N_S, N_ETA = 241, 21     # target grid
POLY_DEG = 4             # meridian z(r^2) polynomial degree
# end of the gore: full tip (r = 0).  The linear-fit gore ends at spl_r(s_hi) = 0.140586 form
# units (117.155 mm), the polar-net ring kept by CFG.TRIM_DISH.
R_TIP_FORM = float(os.environ.get("CST_R_TIP_MM", "0")) / (CFG.DIAMETER_MM / 6.0)


# =============================================================================
# Net, tensors, MLS at arbitrary points
# =============================================================================

def _load_state():
    import paraboloid_profile as pp
    import local_stress_recovery as LSR
    st = np.load(STATE_NPZ, allow_pickle=True)
    V, T, E, fixed = (np.asarray(st[k]) for k in ("vertices", "faces_tri", "edges", "fixed"))
    q = np.asarray(st["q_categories"], float)[np.asarray(st["cat_ids"], int)]
    p_form = abs(float(st["GAS_K"]) / pp._enclosed_volume(V, T))
    S, Ai, nrm, h_med = LSR.nodal_virial(V, T, E, q, fixed)
    interior = np.ones(len(V), bool)
    interior[fixed] = False
    info = str(st["projection_info"])
    return dict(V=V, T=T, S=S, Ai=Ai, nrm=nrm, h_med=h_med, interior=interior,
                p_form=p_form, R=float(st["R"]), DEPTH=float(st["DEPTH"]), info=info)


def _mls2_at(net, mer, P):
    """Quadratic MLS tensor (3x3 global, tangent-projected) at planform points P (n, 2):
    the estimator of local_stress_recovery.recover('mls2') centred at P instead of a node.
    The query point's height and normal come from the fitted axisymmetric meridian (the
    surface the gore lies on); the rim of the net is a chord polygon, so points at r = R
    lie just outside its planform and cannot be interpolated from the net."""
    V, S, Ai, interior = net["V"], net["S"], net["Ai"], net["interior"]
    rho = RHO_EDGES * net["h_med"]
    sig = 0.5 * rho
    rq = np.hypot(P[:, 0], P[:, 1])
    zq = mer["z_of"](rq)
    dz = mer["dz_of"](rq)
    cq, sq = P[:, 0] / np.maximum(rq, 1e-12), P[:, 1] / np.maximum(rq, 1e-12)
    nq = np.column_stack([-dz * cq, -dz * sq, np.ones_like(rq)])
    nq /= np.linalg.norm(nq, axis=1)[:, None]
    idx = np.flatnonzero(interior)
    tree = cKDTree(V[idx, :2])
    comp = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    Y_all = np.column_stack([S[:, a, b] for a, b in comp])
    out = np.zeros((len(P), 3, 3))
    nbs = tree.query_ball_point(P, rho)
    for i, nb in enumerate(nbs):
        nb = idx[np.asarray(nb, int)]
        p3 = np.array([P[i, 0], P[i, 1], zq[i]])
        d = V[nb] - p3
        dist = np.linalg.norm(d, axis=1)
        w = Ai[nb] * np.exp(-0.5 * (dist / sig) ** 2)
        n = nq[i]
        e1 = np.cross(n, [0.0, 1.0, 0.0] if abs(n[1]) < 0.9 else [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        u, v = d @ e1, d @ e2
        A = np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
        if len(nb) < 2 * A.shape[1]:
            raise ValueError(f"MLS patch too small at {P[i]} ({len(nb)} nodes)")
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(A * sw[:, None], Y_all[nb] * sw[:, None], rcond=None)
        M = np.zeros((3, 3))
        for k, (a, b) in enumerate(comp):
            M[a, b] = M[b, a] = coef[0, k]
        Pm = np.eye(3) - np.outer(n, n)
        out[i] = Pm @ M @ Pm
    return out


# =============================================================================
# Meridian (form units, s from the rim)
# =============================================================================

def _meridian(net):
    V, R = net["V"], net["R"]
    r = np.hypot(V[:, 0], V[:, 1])
    A = np.column_stack([(r / R) ** (2 * k) for k in range(POLY_DEG + 1)])
    c, *_ = np.linalg.lstsq(A, V[:, 2], rcond=None)
    z_of = lambda rr: np.column_stack([(rr / R) ** (2 * k) for k in range(POLY_DEG + 1)]) @ c  # noqa: E731
    dz_of = lambda rr: np.column_stack(                                                        # noqa: E731
        [np.zeros_like(rr)] + [2 * k * rr ** (2 * k - 1) / R ** (2 * k)
                               for k in range(1, POLY_DEG + 1)]) @ c
    res = V[:, 2] - z_of(r)
    rr = np.linspace(R, R_TIP_FORM, 20001)                 # rim -> tip
    g = np.sqrt(1.0 + dz_of(rr) ** 2)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.abs(np.diff(rr)))])
    sub = slice(None, None, 5)
    spl_r = UnivariateSpline(s[sub], rr[sub], k=3, s=0)
    spl_z = UnivariateSpline(s[sub], z_of(rr[sub]), k=3, s=0)
    return dict(c=c, z_of=z_of, dz_of=dz_of, spl_r=spl_r, spl_z=spl_z, s_hi=float(s[-1]),
                s_tab=s, r_tab=rr, fit_rms=float(np.sqrt(np.mean(res ** 2))),
                fit_max=float(np.abs(res).max()))


# =============================================================================
# Cyclic + mirror averaged target grid
# =============================================================================

def build_grid(net, mer, verbose=True):
    gs = CFG.DIAMETER_MM / (2.0 * net["R"])
    theta = np.pi / CFG.N_GORES
    s_g = np.linspace(0.0, mer["s_hi"], N_S)
    eta_g = np.linspace(-1.0, 1.0, N_ETA)
    r_g = np.clip(mer["spl_r"](s_g), 0.0, None)
    r_g[0] = min(r_g[0], net["R"] * (1 - 1e-9))            # stay inside the planform
    drds = mer["spl_r"].derivative()(s_g)
    dzds = mer["spl_z"].derivative()(s_g)
    ks = np.arange(CFG.N_GORES)
    SS, EE, KK = np.meshgrid(np.arange(N_S), np.arange(N_ETA), ks, indexing="ij")
    phi = theta * (2 * KK + 1 + eta_g[EE])
    rr = r_g[SS]
    P = np.column_stack([(rr * np.cos(phi)).ravel(), (rr * np.sin(phi)).ravel()])
    if verbose:
        print(f"[cst] MLS2 at {len(P)} points ({N_S} s x {N_ETA} eta x {CFG.N_GORES} gores), "
              f"rho = {RHO_EDGES:g} x {net['h_med']:.4f}", flush=True)
    Sg = _mls2_at(net, mer, P).reshape(N_S, N_ETA, CFG.N_GORES, 3, 3)
    ph = phi
    m = np.stack([drds[SS] * np.cos(ph), drds[SS] * np.sin(ph), dzds[SS] * np.ones_like(ph)], -1)
    m /= np.linalg.norm(m, axis=-1, keepdims=True)
    h = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], -1)
    Nmm = np.einsum("...i,...ij,...j->...", m, Sg, m)
    Nhh = np.einsum("...i,...ij,...j->...", h, Sg, h)
    Nmh = np.einsum("...i,...ij,...j->...", m, Sg, h)
    per_gore = np.stack([Nmm, Nhh, Nmh], -1)               # (N_S, N_ETA, 24, 3), form units
    cyc = per_gore.mean(axis=2)
    mir = cyc[:, ::-1]
    avg = np.empty_like(cyc)
    avg[..., :2] = 0.5 * (cyc[..., :2] + mir[..., :2])
    avg[..., 2] = 0.5 * (cyc[..., 2] - mir[..., 2])
    to_mpa = (CFG.P_PHYS / net["p_form"]) * gs / CFG.T_FABRIC
    return dict(s_g=s_g, eta_g=eta_g, r_g=r_g, sigma=avg * to_mpa, per_gore=per_gore * to_mpa,
                cyc=cyc * to_mpa, to_mpa=to_mpa, p_form=net["p_form"], geom_scale=gs,
                h_med=net["h_med"], rho=RHO_EDGES * net["h_med"], mer_c=mer["c"],
                fit_rms=mer["fit_rms"], fit_max=mer["fit_max"], s_hi=mer["s_hi"],
                state_npz=STATE_NPZ, info=net["info"], r_tip_form=R_TIP_FORM)


def load_grid(rebuild=False):
    if os.path.exists(CACHE_NPZ) and not rebuild:
        g = dict(np.load(CACHE_NPZ, allow_pickle=True))
        if str(g["state_npz"]) == STATE_NPZ and float(g.get("r_tip_form", -1)) == R_TIP_FORM:
            return g
    net = _load_state()
    mer = _meridian(net)
    g = build_grid(net, mer)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CACHE_NPZ + ".tmp.npz"
    np.savez(tmp, **g)
    os.replace(tmp, CACHE_NPZ)
    return dict(np.load(CACHE_NPZ, allow_pickle=True))


# =============================================================================
# Profile object
# =============================================================================

class CSTProfile(ff.MembraneProfile):
    """MembraneProfile interface over the cached CST grid (no ring data needed)."""

    def __init__(self, grid):                              # noqa: D401 -- no super().__init__
        self.name = "dish"
        self.t = float(CFG.T_FABRIC)
        self.geom_scale = float(grid["geom_scale"])
        self.tension_scale = np.nan                        # not used: sigma comes from the grid
        R = CFG.DIAMETER_MM / 2.0 / self.geom_scale
        c = np.asarray(grid["mer_c"])
        z_of = lambda rr: np.column_stack([(rr / R) ** (2 * k) for k in range(len(c))]) @ c  # noqa: E731
        dz_of = lambda rr: np.column_stack(                                                  # noqa: E731
            [np.zeros_like(rr)] + [2 * k * rr ** (2 * k - 1) / R ** (2 * k)
                                   for k in range(1, len(c))]) @ c
        rr = np.linspace(R, float(grid["r_tip_form"]), 20001)
        g = np.sqrt(1.0 + dz_of(rr) ** 2)
        s = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.abs(np.diff(rr)))])
        sub = slice(None, None, 5)
        self.spl_r = UnivariateSpline(s[sub], rr[sub], k=3, s=0)
        self.spl_z = UnivariateSpline(s[sub], z_of(rr[sub]), k=3, s=0)
        self.s_lo, self.s_hi = 0.0, float(s[-1])
        self._r_inc, self._s_inc = rr[::-1], s[::-1]       # r increasing, for s_of_pos
        sg, eg, sig = grid["s_g"], grid["eta_g"], grid["sigma"]
        self._f = [RectBivariateSpline(sg, eg, sig[..., k], kx=3, ky=3) for k in range(3)]
        self._theta = np.pi / CFG.N_GORES
        self.grid = grid

    def s_of_pos(self, pos):
        rq = float(np.hypot(pos[0], pos[1])) / self.geom_scale
        return float(np.clip(np.interp(rq, self._r_inc, self._s_inc), self.s_lo, self.s_hi))

    def sigma_field(self, pos):
        s = self.s_of_pos(pos)
        phi = np.arctan2(pos[1], pos[0]) % (2 * self._theta)    # gore-relative, cyclic
        eta = float(np.clip(phi / self._theta - 1.0, -1.0, 1.0))
        return tuple(float(f(s, eta)[0, 0]) for f in self._f)

    def stress_at_s(self, s):
        return tuple(float(f(s, 0.0)[0, 0]) for f in self._f)

    def N_at_s(self, s):
        sm, st, _ = self.stress_at_s(s)
        return sm * self.t, st * self.t


def extract_paraboloid_cst(rebuild=False, gore=None):
    """gore=None: the cyclic + mirror averaged target (the "cst" study).  gore=k: the RAW field of
    gore k (no averaging; per_gore of the same cached grid), the "cstraw" target -- the gore must
    then be meshed with gore_index=k (sigma_field maps position -> (s, eta) of gore k)."""
    g = load_grid(rebuild)
    if gore is not None:
        g = dict(g)
        g["sigma"] = np.asarray(g["per_gore"])[:, :, int(gore), :]
    prof = CSTProfile(g)
    meta = dict(geom_scale=prof.geom_scale, tension_scale=np.nan, p_fdm=float(g["p_form"]),
                p_true=float(g["p_form"]), p_phys=CFG.P_PHYS,
                target="cst" if gore is None else "cstraw", raw_gore=-1 if gore is None else int(gore),
                state_npz=str(g["state_npz"]))
    return prof, None, meta


# =============================================================================
# Full-gore error check (exact statics used ONLY here)
# =============================================================================

def _exact_mpa(r_form, g):
    f = CFG.F_MM
    r_mm = r_form * float(g["geom_scale"])
    u = np.sqrt(1.0 + (r_mm / (2.0 * f)) ** 2)
    return (CFG.P_PHYS * f * u / CFG.T_FABRIC,
            CFG.P_PHYS * 2.0 * f * u * (1.0 - 1.0 / (2.0 * u ** 2)) / CFG.T_FABRIC)


def check(g, stem="fig_paraboloid_cst_target_check"):
    import paraboloid_profile as pp
    lin, _, _ = pp.extract_paraboloid()
    s_g, eta_g, r_g, sig = g["s_g"], g["eta_g"], g["r_g"], g["sigma"]
    gs = float(g["geom_scale"])
    exm, exh = _exact_mpa(r_g, g)
    em = sig[..., 0] / exm[:, None] - 1
    eh = sig[..., 1] / exh[:, None] - 1
    tau = sig[..., 2] / exm[:, None]
    pg = g["per_gore"]
    spread_m = pg[..., 0].std(axis=2) / exm[:, None]
    spread_h = pg[..., 1].std(axis=2) / exh[:, None]
    R = CFG.DIAMETER_MM / 2.0 / gs
    bands = [("rim  r > 0.96R", r_g > 0.96 * R), ("body 0.2-0.96R", (r_g >= 0.2 * R) & (r_g <= 0.96 * R)),
             ("apex r < 0.2R", r_g < 0.2 * R), ("whole gore", np.ones_like(r_g, bool))]
    rms = lambda a: 100 * float(np.sqrt(np.mean(a ** 2)))              # noqa: E731
    mx = lambda a: 100 * float(np.abs(a).max())                        # noqa: E731
    print(f"[cst] state {os.path.basename(str(g['state_npz']))}; {g['info']}")
    print(f"[cst] p_form {float(g['p_form']):.6f}, scale to {CFG.P_PHYS} MPa: x{float(g['to_mpa']):.4f}"
          f" MPa per form unit; meridian fit rms {float(g['fit_rms'])*gs:.3f} mm, "
          f"max {float(g['fit_max'])*gs:.3f} mm; gore s_hi {float(g['s_hi'])*gs:.1f} mm")
    print("  band              N_phi rms/max %    N_theta rms/max %   |tau|/N_phi rms %  "
          "gore spread phi/theta rms %")
    for name, b in bands:
        print(f"  {name:16s}  {rms(em[b]):6.3f} / {mx(em[b]):6.3f}    {rms(eh[b]):6.3f} / "
              f"{mx(eh[b]):6.3f}     {rms(tau[b]):7.4f}            "
              f"{rms(spread_m[b]):6.3f} / {rms(spread_h[b]):6.3f}")
    # the current linear-fit target on the same s (its own meridian), for reference
    sl = np.linspace(lin.s_lo, lin.s_hi, 400)
    rl = np.clip(lin.spl_r(sl), 0, None)
    exml, exhl = _exact_mpa(rl, g)
    Ll = np.array([lin.stress_at_s(x)[:2] for x in sl])
    print(f"  linear-fit target (current), whole gore: N_phi {rms(Ll[:, 0]/exml-1):.3f} / "
          f"{mx(Ll[:, 0]/exml-1):.3f}   N_theta {rms(Ll[:, 1]/exhl-1):.3f} / {mx(Ll[:, 1]/exhl-1):.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "mathtext.fontset": "stixsans", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    W = 190 / 25.4
    fig, ax = plt.subplots(1, 3, figsize=(W, W * 0.32))
    smm = s_g * gs / 1000
    ic = N_ETA // 2
    ax[0].plot(smm, exm, "-", color="0.2", lw=1.0, label="exact statics")
    ax[0].plot(smm, exh, "--", color="0.2", lw=1.0)
    ax[0].plot(smm, sig[:, ic, 0], "-", color="#1f4e79", lw=1.6, alpha=0.6, label="CST + MLS2 (avg.)")
    ax[0].plot(smm, sig[:, ic, 1], "--", color="#1f4e79", lw=1.6, alpha=0.6)
    ax[0].plot(sl * lin.geom_scale / 1000, Ll[:, 0], "-", color="#b35806", lw=0.9, label="linear fit (current)")
    ax[0].plot(sl * lin.geom_scale / 1000, Ll[:, 1], "--", color="#b35806", lw=0.9)
    ax[0].set_ylabel("Target stress (MPa)")
    ax[0].legend(loc="upper right")
    ax[0].annotate("solid meridional, dashed hoop", xy=(0.03, 0.04), xycoords="axes fraction",
                   fontsize=7, color="0.35")
    ax[1].axhline(0, color="0.3", lw=0.6)
    ax[1].plot(smm, 100 * em[:, ic], "-", color="#1f4e79", lw=1.0, label=r"$N_\phi$")
    ax[1].plot(smm, 100 * eh[:, ic], "--", color="#1f4e79", lw=1.0, label=r"$N_\theta$")
    ax[1].fill_between(smm, -100 * spread_h[:, ic], 100 * spread_h[:, ic], color="0.85",
                       lw=0, label=r"gore-to-gore spread ($N_\theta$, 1 sd)")
    ax[1].set_ylabel("CST target / exact $-$ 1 (%)")
    ax[1].legend(loc="upper left")
    for a in ax[:2]:
        a.set_xlabel("Meridian arc from rim, $s$ (m)")
        a.grid(alpha=0.22)
    im = ax[2].pcolormesh(eta_g, smm, 100 * np.maximum(np.abs(em), np.abs(eh)), cmap="cividis",
                          shading="auto")
    ax[2].set_xlabel(r"Across gore, $\eta$")
    ax[2].set_ylabel("Meridian arc from rim, $s$ (m)")
    cb = fig.colorbar(im, ax=ax[2])
    cb.set_label(r"max($|\varepsilon_{N_\phi}|, |\varepsilon_{N_\theta}|$) (%)")
    for a, l in zip(ax, "abc"):
        a.text(-0.02, 1.04, f"({l})", transform=a.transAxes, ha="right", fontsize=9)
    fig.tight_layout(w_pad=1.5)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        p = os.path.join(HERE, f"{stem}.{ext}")
        fig.savefig(p, **kw)
        print(f"[figure] saved {p}")


if __name__ == "__main__":
    g = load_grid(rebuild="--rebuild" in sys.argv)
    check(g)
    prof, _, meta = extract_paraboloid_cst()
    # interface self-checks
    X, Xl, seam, bnd, els = prof.build_gore_gmsh(200.0, CFG.N_GORES, gore_index=0)
    hw_err = 0.0
    for p, xl in zip(X, Xl):
        s = prof.s_of_pos(p)
        hw_err = max(hw_err, abs((s - 0.5 * (prof.s_lo + prof.s_hi)) * prof.geom_scale - xl[0]))
    print(f"[check] gore h=200: {len(X)} nodes, {len(els)} tris; max |s_of_pos - leaf x| "
          f"{hw_err:.2e} mm")
    sig = np.array([prof.sigma_field(p) for p in X])
    print(f"[check] sigma_field at gore nodes: warp {sig[:,0].min():.2f}..{sig[:,0].max():.2f}, "
          f"weft {sig[:,1].min():.2f}..{sig[:,1].max():.2f}, |tau| max {np.abs(sig[:,2]).max():.4f} MPa")
