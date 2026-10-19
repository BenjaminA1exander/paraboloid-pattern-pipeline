"""
formfound_profile.py

Bridge from the axisymmetric jax_fdm form-finder
(``SphereDishDifferential_pressure_JAXFDM.py``) to the VaReS cutting-pattern
optimiser -- the form-found analogue of ``npl_profile.py``.

It reads the form-finder's ``jaxfdm_spheredish.npz`` and, for each of the TWO
membranes of the inflatable reflector (they will be made of DIFFERENT
materials, so each gets its own cutting pattern):

    * "wall"  -- the outer envelope (base pole -> equator -> top pole),
    * "dish"  -- the internal concave-up diaphragm (rim -> vertex),

extracts, purely from the equilibrium the form-finder converged to:

    1. the axisymmetric MERIDIAN SHAPE  (r_i, z_i) per ring, ordered along the
       meridian, plus a smooth arc-length parametrisation r(s), z(s);
    2. the IN-MEMBRANE TENSION field, converted from the FDM force densities
       q_e (edge force F_e = q_e * L_e) to continuum stress resultants
           N_phi(s)  -- meridional  (force / hoop-length),
           N_theta(s)-- hoop        (force / meridional-length),
       via the axisymmetric tributary-width rule
           N_phi   = <q_m L_m> * n_theta / (2 pi r),
           N_theta = <q_h L_h> / w_meridional .

The membrane wall stresses handed to VaReS are sigma = N / t (t = per-membrane
thickness), exactly the (sigma_meridional=warp, sigma_hoop=weft, tau=0) triple
that ``npl_profile.npl_membrane_stress`` supplies -- so the optimised gore makes
the form-found shape a genuine pressure equilibrium and reproduces it on
inflation.

Because the surface is axisymmetric a single meridian defines everything; a gore
is a pizza-slice wedge of azimuth 2*theta_gore = 2*pi/N_GORES revolved from that
meridian.  ``MembraneProfile.build_gore`` returns the spatial target lune and an
equal-arc flat-gore seed in the same node ordering VaReS expects.

Validation (no pressure needs to be stored): the Laplace residual
    p_implied(s) = N_phi/r_m + N_theta/r_t
should be ~constant along each membrane (= that chamber's net pressure); this is
printed by ``__main__`` and is the self-consistency check on the extraction.

Units follow the form-finder (dimensionless jax_fdm units); scale with SCALE if a
physical size is wanted.  Pure numpy + scipy (runs in the anaconda3 base env,
like the rest of the VaReS pipeline -- no jax needed here).
"""

from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import UnivariateSpline


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NPZ = os.environ.get(
    "FORMFOUND_NPZ",
    os.path.join(os.environ.get("PIPELINE_DATA", os.path.join(HERE, "..", "data")), "jaxfdm_spheredish.npz"))

# Physical scaling (see extract_membranes): geometry is scaled to millimetres so
# the reflector has diameter TARGET_DIAMETER_MM, and the (dimensionless) FDM
# tension is converted to physical N/mm by a single tension scale calibrated so
# the wall's implied FDM pressure maps to P_PHYS_MPA.  Units then match the rest
# of the VaReS pipeline: N, mm, MPa.
TARGET_DIAMETER_MM = float(os.environ.get("FORMFOUND_DIAMETER_MM", "5000.0"))
P_PHYS_MPA = float(os.environ.get("FORMFOUND_P_PHYS", "0.05"))


# =============================================================================
# Low-level: load the npz and split it into the two membranes
# =============================================================================

def load_formfound(npz_path: str = DEFAULT_NPZ):
    """Load vertices (FORM units), edges, force densities and the per-membrane
    face sets.

    The npz stores faces_up = [uenv_tri, dish_down] and
    faces_lo = [lenv_tri, dish_tri], so the envelope and dish triangle sets are
    recovered by trimming the dish block off each.  Geometry is kept in the
    form-finder's own (dimensionless) units here; MembraneProfile applies the
    physical geom_scale/tension_scale at its output interfaces so the tension
    resultants (invariant to a uniform geometry rescale) are computed cleanly.
    """
    d = np.load(npz_path)
    V = np.asarray(d["vertices"], float)
    E = np.asarray(d["edges"], int)
    Q = np.asarray(d["q_opt"], float)
    dish_tri = np.asarray(d["dish_tri"], int)
    faces_up = np.asarray(d["faces_up"], int)
    faces_lo = np.asarray(d["faces_lo"], int)
    nd = dish_tri.shape[0]
    uenv_tri = faces_up[: faces_up.shape[0] - nd]
    lenv_tri = faces_lo[: faces_lo.shape[0] - nd]
    env_nodes = np.unique(np.concatenate([uenv_tri.ravel(), lenv_tri.ravel()]))
    dish_nodes = np.unique(dish_tri.ravel())

    r = np.hypot(V[:, 0], V[:, 1])
    z = V[:, 2]
    # ring id: nodes sharing (r, z) under axisymmetry form one hoop ring.
    key = np.round(r, 5) + 1j * np.round(z, 5)
    _, rid = np.unique(key, return_inverse=True)
    counts = np.bincount(rid)
    n_theta = int(np.median(counts[counts > 1]))

    u, v = E[:, 0], E[:, 1]
    L = np.linalg.norm(V[u] - V[v], axis=1)
    F = Q * L                                   # per-edge axial force
    is_hoop = rid[u] == rid[v]

    return dict(V=V, E=E, Q=Q, L=L, F=F, r=r, z=z, rid=rid, n_theta=n_theta,
                is_hoop=is_hoop, u=u, v=v,
                env_nodes=env_nodes, dish_nodes=dish_nodes,
                uenv_tri=uenv_tri, lenv_tri=lenv_tri, dish_tri=dish_tri)


# =============================================================================
# MembraneProfile -- one meridian + its tension field + a gore builder
# =============================================================================

class MembraneProfile:
    """One axisymmetric membrane: ordered rings, smooth meridian r(s)/z(s),
    smoothed tension resultants N_phi(s), N_theta(s), and a gore builder."""

    def __init__(self, name, data, node_set, tri_set, *, thickness,
                 trim_r_frac=0.06, spline_smooth=2e-3,
                 geom_scale=1.0, tension_scale=1.0):
        self.name = name
        self.t = float(thickness)                 # mm
        self.trim_r_frac = float(trim_r_frac)
        self.geom_scale = float(geom_scale)       # mm per form-unit
        self.tension_scale = float(tension_scale)  # N/mm per FDM-tension-unit
        D = data
        self.n_theta = D["n_theta"]
        node_set = np.asarray(sorted(int(n) for n in node_set), int)
        self._node_set = set(node_set.tolist())
        self._env_set = set(D["env_nodes"].tolist())
        self._dish_set = set(D["dish_nodes"].tolist())

        # -- order the rings along the meridian -------------------------------
        rid = D["rid"]; r = D["r"]; z = D["z"]
        rings = sorted({int(rid[n]) for n in node_set})
        rr = np.array([r[rid == g].mean() for g in rings])
        zz = np.array([z[rid == g].mean() for g in rings])
        # wall: z monotonic base->top; dish: r monotonic rim->vertex.
        order = np.argsort(zz) if name == "wall" else np.argsort(-rr)
        rings = [rings[i] for i in order]
        rr, zz = rr[order], zz[order]

        self.rings_all = rings
        self.r_all, self.z_all = rr, zz

        # -- tension resultants per ring / band -------------------------------
        s_all = self._arclen(rr, zz)
        Nt = self._hoop_resultant(D, rings, rr, s_all)          # per ring
        s_band, Nphi = self._merid_resultant(D, rings, rr, zz)  # per band
        Nphi_ring = np.interp(s_all, s_band, Nphi) if len(s_band) >= 2 \
            else np.full_like(s_all, np.nan)

        # -- trim the diverging pole/vertex neighbourhood ---------------------
        rmax = rr.max()
        keep = rr >= self.trim_r_frac * rmax
        if name == "wall":
            keep &= np.isfinite(Nt) & np.isfinite(Nphi_ring)
        self._keep = keep
        s_k = s_all[keep] - s_all[keep][0]
        r_k, z_k = rr[keep], zz[keep]
        Nt_k, Nphi_k = Nt[keep], Nphi_ring[keep]

        # -- smooth splines in arc length ------------------------------------
        # GEOMETRY: the (r,z) rings are exact under axisymmetry, so the geometry
        # spline should nearly INTERPOLATE them (a large smoothing rounds off the
        # belly/equator -- that was pulling the wall equator ~30mm below its true
        # radius and creating a spurious weld mismatch with the dish rim).
        sm = spline_smooth * len(s_k)
        geom_sm = 2e-5 * len(s_k)
        self.s = s_k
        self.spl_r = UnivariateSpline(s_k, r_k, k=3, s=geom_sm * (r_k.ptp() ** 2 or 1))
        self.spl_z = UnivariateSpline(s_k, z_k, k=3, s=geom_sm * (z_k.ptp() ** 2 or 1))
        # tension: robust smoothing (resultants are noisier near the trimmed ends)
        self.spl_Nt = UnivariateSpline(s_k, self._fillnan(Nt_k), k=3,
                                       s=3 * sm * (np.nanmax(Nt_k) ** 2 or 1))
        self.spl_Nphi = UnivariateSpline(s_k, self._fillnan(Nphi_k), k=3,
                                         s=3 * sm * (np.nanmax(Nphi_k) ** 2 or 1))
        self.s_lo, self.s_hi = float(s_k[0]), float(s_k[-1])
        self.r_k, self.z_k, self.Nt_k, self.Nphi_k = r_k, z_k, Nt_k, Nphi_k

        # dense meridian samples for pos->s inversion.  For a CONTINUOUS,
        # smooth target field we invert s from the meridian's MONOTONIC
        # coordinate (wall: z base->top; dish: r rim->vertex) by 1-D
        # interpolation -- not a nearest-sample lookup (which would quantise s
        # and stair-step the field).
        sd = np.linspace(self.s_lo, self.s_hi, 1200)
        self._sd = sd
        self._rd = self.spl_r(sd)
        self._zd = self.spl_z(sd)
        self._inv_coord = "z" if name == "wall" else "r"
        key = self._zd if self._inv_coord == "z" else self._rd
        order = np.argsort(key)
        self._inv_key = key[order]
        self._inv_s = sd[order]

    # ---- geometry helpers ---------------------------------------------------
    @staticmethod
    def _arclen(rr, zz):
        ds = np.sqrt(np.diff(rr) ** 2 + np.diff(zz) ** 2)
        return np.concatenate([[0.0], np.cumsum(ds)])

    @staticmethod
    def _fillnan(a):
        a = np.array(a, float)
        m = ~np.isfinite(a)
        if m.any():
            idx = np.arange(len(a))
            a[m] = np.interp(idx[m], idx[~m], a[~m])
        return a

    def _which(self, a, b):
        in_env = (a in self._env_set) and (b in self._env_set)
        in_dish = (a in self._dish_set) and (b in self._dish_set)
        return in_env if self.name == "wall" else in_dish

    def _hoop_resultant(self, D, rings, rr, s_all):
        rid, u, v, F, is_hoop = D["rid"], D["u"], D["v"], D["F"], D["is_hoop"]
        ds = np.diff(s_all)
        wm = np.empty(len(rings))
        for i in range(len(rings)):
            seg = []
            if i > 0:
                seg.append(ds[i - 1])
            if i < len(rings) - 1:
                seg.append(ds[i])
            wm[i] = 0.5 * sum(seg) if len(seg) == 2 else (seg[0] if seg else 1.0)
        Nt = np.full(len(rings), np.nan)
        for i, g in enumerate(rings):
            m = is_hoop & (rid[u] == g)
            if m.any() and rr[i] > 1e-9 and wm[i] > 0:
                Nt[i] = F[m].mean() / wm[i]
        return Nt

    def _merid_resultant(self, D, rings, rr, zz):
        rid, u, v, F, is_hoop, n_theta = (D["rid"], D["u"], D["v"], D["F"],
                                          D["is_hoop"], D["n_theta"])
        s_all = self._arclen(rr, zz)
        s_band, Nphi = [], []
        for i in range(len(rings) - 1):
            a, b = rings[i], rings[i + 1]
            m = (~is_hoop) & (((rid[u] == a) & (rid[v] == b)) |
                              ((rid[u] == b) & (rid[v] == a)))
            idx = np.where(m)[0]
            if idx.size == 0:
                continue
            idx = idx[[self._which(int(u[k]), int(v[k])) for k in idx]]
            if idx.size == 0:
                continue
            rbar = 0.5 * (rr[i] + rr[i + 1])
            rbar = rbar if rbar > 1e-9 else max(rr[i], rr[i + 1])
            if rbar <= 1e-9:
                continue
            Nphi.append(F[idx].mean() * n_theta / (2 * np.pi * rbar))
            s_band.append(0.5 * (s_all[i] + s_all[i + 1]))
        return np.asarray(s_band), np.asarray(Nphi)

    # ---- public sampling ----------------------------------------------------
    # Convention: `s` is FORM-unit meridian arc length (internal spline domain);
    # geometry outputs (r, z, gore coords) are in mm (x geom_scale); stresses are
    # physical MPa (N_fdm x tension_scale / t_mm).
    def meridian(self, n=200):
        s = np.linspace(self.s_lo, self.s_hi, n)
        return s, self.spl_r(s) * self.geom_scale, self.spl_z(s) * self.geom_scale

    def stress_at_s(self, s):
        s = np.clip(s, self.s_lo, self.s_hi)
        return (float(self.spl_Nphi(s)) * self.tension_scale / self.t,  # warp=merid
                float(self.spl_Nt(s)) * self.tension_scale / self.t,    # weft=hoop
                0.0)

    def N_at_s(self, s):
        """Physical membrane tension resultants (N_phi, N_theta) [N/mm]."""
        s = np.clip(s, self.s_lo, self.s_hi)
        return (float(self.spl_Nphi(s)) * self.tension_scale,
                float(self.spl_Nt(s)) * self.tension_scale)

    def s_of_pos(self, pos):
        """Continuous meridian arc length s at a 3D point on the gore, by 1-D
        interpolation from the meridian's monotonic coordinate -> smooth field."""
        rq = float(np.hypot(pos[0], pos[1])) / self.geom_scale
        zq = float(pos[2]) / self.geom_scale
        key = zq if self._inv_coord == "z" else rq
        return float(np.clip(np.interp(key, self._inv_key, self._inv_s),
                             self.s_lo, self.s_hi))

    def sigma_field(self, pos):
        """VaReS target: (sigma_warp, sigma_weft, tau) at a 3D IP on the gore."""
        return self.stress_at_s(self.s_of_pos(pos))

    def principal_radii_at_s(self, s):
        """(r_t, r_m) FORM units from the smoothed meridian; r_m signed."""
        rp = self.spl_r.derivative()(s); zp = self.spl_z.derivative()(s)
        rpp = self.spl_r.derivative(2)(s); zpp = self.spl_z.derivative(2)(s)
        speed = np.hypot(rp, zp)
        r_t = self.spl_r(s) / np.clip(np.abs(zp / speed), 1e-6, None)
        kappa = (rp * zpp - zp * rpp) / np.clip(speed ** 3, 1e-12, None)
        r_m = 1.0 / np.where(np.abs(kappa) < 1e-9, np.nan, kappa)
        return r_t, r_m

    def implied_pressure_central(self, frac=0.2):
        """FDM Laplace pressure N_phi/r_m + N_theta/r_t averaged over the central
        (1-2*frac) of the meridian (FORM units) -- ~constant = chamber net p."""
        smid = np.linspace(self.s_lo + frac * (self.s_hi - self.s_lo),
                           self.s_hi - frac * (self.s_hi - self.s_lo), 60)
        rt, rm = self.principal_radii_at_s(smid)
        p = self.spl_Nphi(smid) / rm + self.spl_Nt(smid) / rt
        return float(np.nanmean(p))

    # ---- gore builder -------------------------------------------------------
    def build_gore(self, n_axial, n_circ, n_gores, gore_index=0):
        """Return the spatial target lune + equal-arc flat-gore seed for one
        gore, in VaReS node ordering (flat index = j*(n_axial+1) + i).

        Returns X_spatial (N,3), X_flat0 (N,3, z=0), s_stations (n_axial+1,),
        seam_ids (list).
        """
        theta = np.pi / n_gores
        phi_c = gore_index * 2.0 * np.pi / n_gores + theta
        s_st = np.linspace(self.s_lo, self.s_hi, n_axial + 1)
        r_st = self.spl_r(s_st) * self.geom_scale
        z_st = self.spl_z(s_st) * self.geom_scale
        s_centred = (s_st - 0.5 * (s_st[0] + s_st[-1])) * self.geom_scale

        nn = (n_axial + 1) * (n_circ + 1)
        X_spatial = np.zeros((nn, 3))
        X_flat0 = np.zeros((nn, 3))
        seam_ids = []
        for j in range(n_circ + 1):
            uu = j / n_circ
            phi = phi_c - theta + uu * (2.0 * theta)
            for i in range(n_axial + 1):
                nid = j * (n_axial + 1) + i
                ri = r_st[i]
                X_spatial[nid] = (ri * np.cos(phi), ri * np.sin(phi), z_st[i])
                w = ri * theta                       # developed half-width
                X_flat0[nid] = (s_centred[i], -w + uu * (2.0 * w), 0.0)
            if j == 0 or j == n_circ:
                seam_ids.extend(j * (n_axial + 1) + i for i in range(n_axial + 1))
        return X_spatial, X_flat0, s_st, sorted(seam_ids)

    def build_gore_gmsh(self, mesh_size_mm, n_gores, gore_index=0, tip_grade=None, tip_affine=None):
        """Unstructured gmsh triangle gore (same method as the NPL chen
        optimiser).  The gore is meshed in its DEVELOPED equal-metric leaf
        (x = meridian arc length mm, |y| <= h(s) = r(s)*theta), a near-isometry
        of the surface-of-revolution lune, then each node is mapped back:
        x -> s -> (r(s), z(s)), phi = clip(y/r, -theta, theta).  Tip-graded so
        the narrowing pole/vertex end keeps ~2 elements across.

        tip_grade=None (default): the size near the tip is theta*(r_ref + d), d = distance from the
        tip -- INDEPENDENT of mesh_size_mm, so the tip zone is the same mesh on every rung of a
        ladder (~2 elements across) and is not refined by it.
        tip_grade=(R_mm, g_min): size = h * max(g_min, min(1, d / R_mm)) with h = mesh_size_mm, i.e.
        h times a FIXED grading function, so every region (tip included) refines with h; the tip
        zone d < R has ~2 theta R / h elements across.  Floor h * g_min.
        tip_affine=(g0, k_per_mm) (2026-10-10): size = h * min(1, g0 + k d), floor h * g0 -- the grading
        of the 25 mm remesh test (graded_tip_test.py: 2 mm + 0.15 d at h = 25 <=> g0 0.08, k 0.006/mm)
        scaled with h.  Mutually exclusive with tip_grade.

        Returns X_spatial (N,3 mm), X_leaf (N,3 mm, z=0 developed seed),
        seam_ids, boundary_ids, elements (list of (3,) int arrays).
        """
        import gmsh
        theta = np.pi / n_gores
        gs = self.geom_scale
        s_mid = 0.5 * (self.s_lo + self.s_hi)

        # dense s<->leaf tables (form units for the splines; mm for the leaf)
        sd = np.linspace(self.s_lo, self.s_hi, 2001)
        xd = (sd - s_mid) * gs                              # leaf x (mm)
        rd_mm = self.spl_r(sd) * gs
        # boundary samples (dense: the seam spline curves where the leaf narrows)
        n_bnd = 201
        sb = np.linspace(self.s_lo, self.s_hi, n_bnd)
        xb = (sb - s_mid) * gs
        hb = self.spl_r(sb) * gs * theta                   # developed half-width
        # closed (zero-width) tips: with no trim the gore runs to the pole, where
        # r -> 0 and the r(s) spline can overshoot slightly negative.  Detect a
        # closed end, taper its half-width linearly to EXACTLY zero at the end
        # sample, and build that end as a single shared apex point (triangular
        # tip) -- a zero-length closing line makes gmsh mesh no triangles.
        TIP_H = 1.0                                        # mm half-width => closed
        closed_lo = bool(hb[0] < TIP_H)
        closed_hi = bool(hb[-1] < TIP_H)
        hb = np.maximum(hb, 0.0)
        if closed_hi:
            it = int(np.nonzero(hb > TIP_H)[0][-1])
            hb[it:] = hb[it] * (xb[-1] - xb[it:]) / max(xb[-1] - xb[it], 1e-9)
        if closed_lo:
            it = int(np.nonzero(hb > TIP_H)[0][0])
            hb[:it + 1] = hb[it] * (xb[:it + 1] - xb[0]) / max(xb[it] - xb[0], 1e-9)
        # narrow ends (r small): add a grading tip point there
        r_end = self.spl_r(np.array([self.s_lo, self.s_hi])) * gs
        r_max = float(rd_mm.max())
        narrow = r_end < 0.5 * r_max
        r_ref = float(min(r_end[narrow])) if narrow.any() else r_max

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add(f"formfound_{self.name}_gore")
            geo = gmsh.model.geo
            tip_lo = geo.addPoint(xb[0], 0.0, 0.0, mesh_size_mm) if closed_lo else None
            tip_hi = geo.addPoint(xb[-1], 0.0, 0.0, mesh_size_mm) if closed_hi else None
            pts_top, pts_bot = [], []
            for i in range(n_bnd):
                if closed_lo and i == 0:
                    pts_top.append(tip_lo); pts_bot.append(tip_lo); continue
                if closed_hi and i == n_bnd - 1:
                    pts_top.append(tip_hi); pts_bot.append(tip_hi); continue
                pts_top.append(geo.addPoint(xb[i], +hb[i], 0.0, mesh_size_mm))
                pts_bot.append(geo.addPoint(xb[i], -hb[i], 0.0, mesh_size_mm))
            c_top = geo.addSpline(pts_top)
            c_bot = geo.addSpline(pts_bot)
            curves = [c_top]
            c_hi = c_lo = None
            if not closed_hi:
                c_hi = geo.addLine(pts_top[-1], pts_bot[-1])   # s_hi end
                curves.append(c_hi)
            curves.append(-c_bot)
            if not closed_lo:
                c_lo = geo.addLine(pts_bot[0], pts_top[0])     # s_lo end
                curves.append(c_lo)
            loop = geo.addCurveLoop(curves)
            surf = geo.addPlaneSurface([loop])
            tip_pts = []
            for k, is_narrow in enumerate(narrow):
                if is_narrow:
                    if k and closed_hi:
                        tip_pts.append(tip_hi)
                    elif (not k) and closed_lo:
                        tip_pts.append(tip_lo)
                    else:
                        tip_pts.append(geo.addPoint(xb[-1] if k else xb[0], 0.0, 0.0,
                                                    mesh_size_mm))
            geo.synchronize()

            lc_floor = max(8.0, 0.8 * theta * r_ref)
            if tip_grade is not None:
                if not tip_pts:
                    raise ValueError("tip_grade needs a narrow (tip) end")
                R_g, g_min = (float(v) for v in tip_grade)
                lc_floor = mesh_size_mm * g_min
            if tip_affine is not None:
                if tip_grade is not None:
                    raise ValueError("tip_grade and tip_affine are mutually exclusive")
                if not tip_pts:
                    raise ValueError("tip_affine needs a narrow (tip) end")
                g0, k_aff = (float(v) for v in tip_affine)
                lc_floor = mesh_size_mm * g0
            if tip_pts:
                dist_f = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist_f, "PointsList", tip_pts)
                gmsh.model.mesh.field.setNumber(dist_f, "Sampling", 100)
                math_f = gmsh.model.mesh.field.add("MathEval")
                if tip_affine is not None:
                    F = (f"Min({mesh_size_mm:.4f}, {mesh_size_mm * g0:.4f} + "
                         f"{mesh_size_mm * k_aff:.6f}*F{dist_f})")
                elif tip_grade is None:
                    F = (f"Max({lc_floor:.3f}, Min({mesh_size_mm:.3f}, "
                         f"{theta:.6f}*({r_ref:.1f} + F{dist_f})))")
                else:
                    F = (f"Max({lc_floor:.4f}, Min({mesh_size_mm:.4f}, "
                         f"{mesh_size_mm:.4f}*F{dist_f}/{R_g:.3f}))")
                gmsh.model.mesh.field.setString(math_f, "F", F)
                gmsh.model.mesh.field.setAsBackgroundMesh(math_f)
                gmsh.option.setNumber("Mesh.MeshSizeMin", lc_floor)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_mm)
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.Smoothing", 10)
            gmsh.model.mesh.generate(2)

            ntags, ncoords, _ = gmsh.model.mesh.getNodes()
            ncoords = np.asarray(ncoords, float).reshape(-1, 3)
            tag2xy = {int(t): ncoords[i, :2] for i, t in enumerate(ntags)}
            etypes, _, enodes = gmsh.model.mesh.getElements(2, surf)
            tris = None
            for et, en in zip(etypes, enodes):
                if et == 2:
                    tris = np.asarray(en, np.int64).reshape(-1, 3)
            if tris is None:
                raise RuntimeError(f"gmsh meshed no triangles ({self.name})")

            def _cnodes(tag):
                nt, _, _ = gmsh.model.mesh.getNodes(1, tag, includeBoundary=True)
                return {int(t) for t in nt}
            seam_tags = _cnodes(c_top) | _cnodes(c_bot)
            bnd_tags = set(seam_tags)
            if c_hi is not None:
                bnd_tags |= _cnodes(c_hi)
            if c_lo is not None:
                bnd_tags |= _cnodes(c_lo)

            used = np.unique(tris)
            remap = {int(t): i for i, t in enumerate(used)}
            elements = [np.array([remap[int(t)] for t in row], int) for row in tris]
            phi_c = 2.0 * theta * gore_index + theta
            X_spatial = np.zeros((len(used), 3))
            X_leaf = np.zeros((len(used), 3))
            for t in used:
                t = int(t)
                xl, yl = tag2xy[t]
                s = float(np.clip(xl / gs + s_mid, self.s_lo, self.s_hi))
                r = max(float(self.spl_r(s) * gs), 0.0)   # spline can dip <0 at a pole
                z = float(self.spl_z(s) * gs)
                if t in seam_tags and abs(yl) > 0.5 * max(r * theta, 1e-9):
                    phil = np.sign(yl) * theta
                else:
                    phil = float(np.clip(yl / max(r, 1e-9), -theta, theta))
                phi = phi_c + phil
                X_spatial[remap[t]] = (r * np.cos(phi), r * np.sin(phi), z)
                X_leaf[remap[t]] = (xl, yl, 0.0)
            seam_ids = sorted(remap[t] for t in seam_tags if t in remap)
            boundary_ids = sorted(remap[t] for t in bnd_tags if t in remap)
        finally:
            gmsh.finalize()
        return X_spatial, X_leaf, seam_ids, boundary_ids, elements


# =============================================================================
# Factory: build both membranes from an npz
# =============================================================================

def extract_membranes(npz_path=DEFAULT_NPZ, *,
                      target_diameter_mm=TARGET_DIAMETER_MM, p_phys=P_PHYS_MPA,
                      t_wall=0.9, t_dish=0.5, trim_wall=0.10, trim_dish=0.06):
    """Return (wall, dish, D, meta) with physical geom/tension scaling.

    geom_scale [mm/form-unit] makes the belly diameter = target_diameter_mm.
    tension_scale [N/mm per FDM unit] is calibrated so the wall's implied FDM
    pressure maps to p_phys (MPa); the SAME scale is applied to both membranes so
    the whole form-found equilibrium is converted consistently (the dish keeps
    its true, much smaller, differential-driven tension).
    """
    D = load_formfound(npz_path)
    r_max = float(D["r"].max())
    geom_scale = target_diameter_mm / (2.0 * r_max)
    wall = MembraneProfile("wall", D, D["env_nodes"], D["uenv_tri"],
                           thickness=t_wall, trim_r_frac=trim_wall)
    dish = MembraneProfile("dish", D, D["dish_nodes"], D["dish_tri"],
                           thickness=t_dish, trim_r_frac=trim_dish)
    p_fdm = wall.implied_pressure_central()
    tension_scale = (p_phys / p_fdm) * geom_scale
    for prof in (wall, dish):
        prof.geom_scale = geom_scale
        prof.tension_scale = tension_scale
    meta = dict(geom_scale=geom_scale, tension_scale=tension_scale, p_fdm=p_fdm,
                p_phys=p_phys, target_diameter_mm=target_diameter_mm,
                r_max_form=r_max)
    return wall, dish, D, meta


# =============================================================================
# Self-check + validation figure
# =============================================================================

def _report(prof: MembraneProfile):
    s, r, z = prof.meridian(24)                      # r, z in mm
    r_t, r_m = prof.principal_radii_at_s(s)          # form units (for p_impl)
    print(f"\n=== {prof.name.upper()} ===  t={prof.t} mm  "
          f"meridian s in [{prof.s_lo * prof.geom_scale:.1f}, "
          f"{prof.s_hi * prof.geom_scale:.1f}] mm  "
          f"(kept {prof._keep.sum()}/{len(prof.r_all)} rings)")
    print(f"  r in [{r.min():.1f}, {r.max():.1f}] mm  "
          f"z in [{z.min():.1f}, {z.max():.1f}] mm")
    print(f"  {'s_mm':>8s} {'r_mm':>8s} {'z_mm':>8s} {'N_phi':>8s} {'N_theta':>8s} "
          f"{'sig_m':>8s} {'sig_t':>8s} {'p_fdm':>7s}")
    for i in range(0, len(s), 2):
        Np, Nt = prof.N_at_s(s[i])                   # N/mm physical
        sm, st, _ = prof.stress_at_s(s[i])           # MPa
        p = float(prof.spl_Nphi(s[i])) / r_m[i] + float(prof.spl_Nt(s[i])) / r_t[i]
        print(f"  {s[i]*prof.geom_scale:8.1f} {r[i]:8.1f} {z[i]:8.1f} "
              f"{Np:8.3f} {Nt:8.3f} {sm:8.2f} {st:8.2f} {p:7.3f}")
    p0 = prof.implied_pressure_central()
    print(f"  [central 60%] FDM implied pressure ~{p0:+.4f} (form units; the "
          f"consistency check -- should be ~constant per membrane)")


def _figure(wall, dish, meta, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[figure] skipped ({e})")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    gs = meta["geom_scale"]
    for col, prof in enumerate((wall, dish)):
        s, r, z = prof.meridian(300)
        ax = axes[0, col]
        ax.plot(r, z, "-", color="0.3", lw=1.5, label="smoothed meridian")
        ax.plot(prof.r_all * gs, prof.z_all * gs, ".", color="0.75", ms=4,
                label="all rings")
        ax.plot(prof.r_k * gs, prof.z_k * gs, "o", color="crimson", ms=4,
                label="kept rings")
        ax.set_aspect("equal"); ax.set_xlabel("radius r [mm]"); ax.set_ylabel("z [mm]")
        ax.set_title(f"{prof.name} meridian (revolve axis = z)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax = axes[1, col]
        sm = np.array([prof.stress_at_s(si)[0] for si in s])
        st = np.array([prof.stress_at_s(si)[1] for si in s])
        smm = s * gs
        ax.plot(smm, sm, "-", color="navy", lw=2, label=r"$\sigma_m$ (warp)")
        ax.plot(smm, st, "-", color="darkorange", lw=2, label=r"$\sigma_t$ (weft)")
        ts = prof.tension_scale / prof.t
        ax.plot(prof.s * gs, prof.Nphi_k * ts, ".", color="navy", ms=5, alpha=0.5)
        ax.plot(prof.s * gs, prof.Nt_k * ts, ".", color="darkorange", ms=5, alpha=0.5)
        ax.set_xlabel("meridian arc length s [mm]")
        ax.set_ylabel("stress = N/t [MPa]")
        ax.set_title(f"{prof.name} in-membrane tension (VaReS target)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"Form-found sphere-dish -> VaReS targets  "
                 f"(diameter {meta['target_diameter_mm']/1000:.2f} m, "
                 f"p={meta['p_phys']} MPa; geom {gs:.1f} mm/unit)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120)
    print(f"[figure] saved {out_png}")


if __name__ == "__main__":
    wall, dish, D, meta = extract_membranes()
    print(f"Loaded {DEFAULT_NPZ}")
    print(f"  {D['V'].shape[0]} nodes, {D['E'].shape[0]} edges, n_theta={D['n_theta']}")
    print(f"  geom_scale={meta['geom_scale']:.2f} mm/unit  "
          f"tension_scale={meta['tension_scale']:.3f} N/mm-per-unit  "
          f"(p_fdm={meta['p_fdm']:.3f} -> p_phys={meta['p_phys']} MPa)")
    _report(wall)
    _report(dish)
    _figure(wall, dish, meta, os.path.join(HERE, "formfound_targets_extract.png"))
    Xs, Xf, sst, seam = wall.build_gore(24, 8, 12)
    print(f"\n[gore] wall 24x8 12-gore: X_spatial {Xs.shape}, seam nodes {len(seam)}, "
          f"span r[{np.hypot(Xs[:,0],Xs[:,1]).min():.0f},"
          f"{np.hypot(Xs[:,0],Xs[:,1]).max():.0f}] mm")
