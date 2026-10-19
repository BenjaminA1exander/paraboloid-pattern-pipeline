r"""cut_extraction.py -- stress recovery from a force-density net on the paraboloid by CUTS, version 2.

Three tools (form units; the target surface z = D(1 - r^2/R^2) supplies tangent frames):

segment_traction(V, E, F, p0, p1)
    Force per unit length transmitted across the straight (plan) segment p0->p1 lifted onto the
    surface: T = sum over crossing edges of F_e * t_e (t_e oriented to the + side) / L_3D.
    Returns T (3-vector), the cut's in-surface normal nu and tangent d.

rosette(V, E, F, x0, ell, n_dir=3, n_off=5)
    Local membrane tensor N (2x2, in the tangent frame e1 = lifted x, e2 = lifted y) at plan point x0:
    straight cuts of length ell in n_dir directions, each repeated at n_off parallel offsets spanning
    +-ell/2 (so the rosette averages over an ell x ell patch), then least squares on
        T . nu = nu^T N nu,   T . d = d^T N nu      (normal + shear, every cut).
    Valid at the apex, where meridional / hoop directions do not exist.

hoop_cumulative(V, E, F, s_eval, window, n_az=720)
    Hoop resultant by INTEGRATE-THEN-DIFFERENTIATE: on each radial half-line theta_j, the cumulative
    hoop force from the apex out to radius r, H_j(r) = sum_{crossings at r_c < r} F_e (t_e . h);
    H(s) = mean_j H_j, and N_theta(s) = dH/ds as the slope of a least-squares line through H over
    [s - window/2, s + window/2] (one-sided at the ends).  No crossing is ever binned into one
    narrow band, which is what made the ring-band hoop cut scatter; `window` is an explicit
    resolution / noise trade.

Run:  python cut_extraction.py   (self-check: uniform isotropic stress on a flat equilateral patch)
"""
import numpy as np

R_FORM, D_FORM = 3.0, 1.5
K = 2.0 * D_FORM / R_FORM ** 2


def surf_z(x, y):
    return D_FORM * (1.0 - (x * x + y * y) / R_FORM ** 2)


def lift_dir(x, y, d2):
    """Plan direction d2 at (x, y) lifted onto the paraboloid's tangent plane, unit."""
    gx, gy = -K * x, -K * y                       # dz/dx, dz/dy
    v = np.array([d2[0], d2[1], gx * d2[0] + gy * d2[1]])
    return v / np.linalg.norm(v)


def surf_normal(x, y):
    n = np.array([K * x, K * y, 1.0])
    return n / np.linalg.norm(n)


def tangent_frame(x, y):
    """e1 = lifted plan x, e2 = in-plane perpendicular (right-handed with the outward normal)."""
    e1 = lift_dir(x, y, (1.0, 0.0))
    n = surf_normal(x, y)
    e2 = np.cross(n, e1)
    return e1, e2 / np.linalg.norm(e2), n


def _lifted_length(p0, p1, n=8):
    s = np.linspace(0, 1, n + 1)
    P = p0[None] + s[:, None] * (p1 - p0)[None]
    Z = surf_z(P[:, 0], P[:, 1])
    Q = np.column_stack([P, Z])
    return float(np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1)))


class Net:
    """Pre-extracted edge data of a converged net (plan endpoints, unit 3-D edge vectors, forces)."""

    def __init__(self, V, E, F):
        self.V, self.E, self.F = V, E, F
        self.a2, self.b2 = V[E[:, 0], :2], V[E[:, 1], :2]
        d = V[E[:, 1]] - V[E[:, 0]]
        self.t = d / np.linalg.norm(d, axis=1)[:, None]
        mid = 0.5 * (self.a2 + self.b2)
        self.mid, self.half = mid, 0.5 * np.linalg.norm(self.b2 - self.a2, axis=1)

    def segment_traction(self, p0, p1):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d2 = p1 - p0
        Lp = np.linalg.norm(d2)
        c = 0.5 * (p0 + p1)
        # cheap prefilter: edges whose midpoint is near the segment
        near = np.linalg.norm(self.mid - c, axis=1) < 0.5 * Lp + self.half + 1e-12
        a, b = self.a2[near], self.b2[near]
        nrm2 = np.array([-d2[1], d2[0]])                          # + side in plan
        sa, sb = (a - p0) @ nrm2, (b - p0) @ nrm2
        crosses = np.sign(sa) != np.sign(sb)
        lam = np.where(crosses, sa / np.where(crosses, sa - sb, 1.0), 0.0)
        X = a + lam[:, None] * (b - a)
        u = ((X - p0) @ d2) / (Lp * Lp)
        sel = crosses & (u >= 0.0) & (u <= 1.0)
        idx = np.flatnonzero(near)[sel]
        sgn = np.sign(sb[sel] - sa[sel])                          # orient t_e from - to +
        Tvec = np.sum((self.F[idx] * sgn)[:, None] * self.t[idx], axis=0)
        L3 = _lifted_length(p0, p1)
        xm, ym = c
        dd = lift_dir(xm, ym, d2 / Lp)
        nu = np.cross(dd, surf_normal(xm, ym))                    # in-surface, toward + side?
        if nu @ lift_dir(xm, ym, nrm2 / Lp) < 0:
            nu = -nu
        return Tvec / L3, nu, dd, int(sel.sum())

    def rosette(self, x0, ell, n_dir=3, n_off=5, angle0=0.0):
        """2x2 tensor N in the frame (e1, e2) at x0, least squares over every cut."""
        x0 = np.asarray(x0, float)
        e1, e2, _ = tangent_frame(*x0)
        rows, rhs, ncross = [], [], 0
        for k in range(n_dir):
            al = angle0 + np.pi * k / n_dir
            dv = np.array([np.cos(al), np.sin(al)])
            pv = np.array([-dv[1], dv[0]])
            offs = np.linspace(-0.5, 0.5, n_off) * ell if n_off > 1 else [0.0]
            for o in offs:
                c = x0 + o * pv
                Tv, nu, dd, nc = self.segment_traction(c - 0.5 * ell * dv, c + 0.5 * ell * dv)
                ncross += nc
                n1, n2 = nu @ e1, nu @ e2                         # nu, d in the frame
                d1, d2_ = dd @ e1, dd @ e2
                # normal: nu^T N nu ; shear: d^T N nu ;  N = [[a, c], [c, b]]
                rows.append([n1 * n1, n2 * n2, 2 * n1 * n2]); rhs.append(Tv @ nu)
                rows.append([d1 * n1, d2_ * n2, d1 * n2 + d2_ * n1]); rhs.append(Tv @ dd)
        sol, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
        a, b, c = sol
        return np.array([[a, c], [c, b]]), e1, e2, ncross

    def hoop_cumulative(self, s_eval, window, n_az=720, r_max=None, deg=1):
        """N_theta at meridian arc s_eval (apex = 0) by dH/ds over `window`; returns (N, s_c, H_c)."""
        V, E = self.V, self.E
        rg = np.linspace(0, R_FORM, 4000)
        sg = np.concatenate([[0], np.cumsum(np.sqrt(1 + (K * 0.5 * (rg[1:] + rg[:-1])) ** 2) * np.diff(rg))])
        a, b = self.a2, self.b2
        rcs, fhs = [], []
        for th0 in np.linspace(0, 2 * np.pi, n_az, endpoint=False):
            er = np.array([np.cos(th0), np.sin(th0)])
            eh = np.array([-np.sin(th0), np.cos(th0)])
            da, db = a @ eh, b @ eh
            m = (np.sign(da) != np.sign(db)) & (da != db)
            lam = da[m] / (da[m] - db[m])
            X = a[m] + lam[:, None] * (b[m] - a[m])
            on = X @ er > 0
            idx = np.flatnonzero(m)[on]
            hh = np.array([-np.sin(th0), np.cos(th0), 0.0])
            fh = self.F[idx] * np.sign(db[idx] - da[idx]) * (self.t[idx] @ hh)
            rcs.append(np.linalg.norm(X[on], axis=1))
            fhs.append(fh)
        rc = np.concatenate(rcs)
        fh = np.concatenate(fhs) / n_az
        o = np.argsort(rc)
        s_c = np.interp(rc[o], rg, sg)
        H_c = np.cumsum(fh[o])
        if r_max is not None:
            keep = rc[o] <= r_max
            s_c, H_c = s_c[keep], H_c[keep]
        out = np.full(len(s_eval), np.nan)
        for i, s0 in enumerate(s_eval):
            lo, hi = s0 - 0.5 * window, s0 + 0.5 * window
            if lo < s_c[0]:
                lo, hi = s_c[0], s_c[0] + window
            if hi > s_c[-1]:
                lo, hi = s_c[-1] - window, s_c[-1]
            mm = (s_c >= lo) & (s_c <= hi)
            if mm.sum() > deg + 2:
                # local polynomial fit of H(s), slope taken AT s0.  deg=1 (default) measured best on the
                # 4626-node run: deg=2 fits the staircase noise (0.2R-0.96R 0.53 vs 0.50%,
                # inner ends 19% vs 6% at a 4-edge window).  The comments below describe deg=2:
                # removes the bias of a straight line through a curving H, and keeps the one-sided
                # windows at the apex / rim unbiased to first order
                cf = np.polyfit(s_c[mm] - s0, H_c[mm], deg)
                out[i] = cf[-2]
        return out, s_c, H_c


def to_mer_hoop(N, e1, e2, x, y):
    """(N_mm, N_hh) of a frame tensor at plan (x, y); NaN at the axis."""
    r = np.hypot(x, y)
    if r < 1e-12:
        return np.nan, np.nan
    m = lift_dir(x, y, (x / r, y / r))
    h = lift_dir(x, y, (-y / r, x / r))
    mv, hv = np.array([m @ e1, m @ e2]), np.array([h @ e1, h @ e2])
    mv /= np.linalg.norm(mv); hv /= np.linalg.norm(hv)
    return float(mv @ N @ mv), float(hv @ N @ hv)


if __name__ == "__main__":
    # self-check on a FLAT equilateral lattice with uniform q: exact N = sqrt(3) q I (the cap_A rule)
    K = 0.0                                       # module scope here: flattens the frame helpers
    q0, h = 1.0, 0.05
    ii, jj = np.meshgrid(np.arange(-40, 41), np.arange(-40, 41))
    P = np.column_stack([((ii + 0.5 * (jj % 2)) * h).ravel(), (jj * h * np.sqrt(3) / 2).ravel()])
    from scipy.spatial import Delaunay
    T = Delaunay(P).simplices
    Ed = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    L = np.linalg.norm(P[Ed[:, 0]] - P[Ed[:, 1]], axis=1)
    Ed = Ed[L < 1.01 * h]
    V = np.column_stack([P, np.zeros(len(P))])
    net = Net(V, Ed, q0 * h * np.ones(len(Ed)))
    for ell in (2 * h, 4 * h, 8 * h):
        N, *_ = net.rosette(np.array([0.013, -0.021]), ell)
        print(f"[self-check] flat equilateral lattice, ell={ell / h:.0f}h: N = {np.round(N, 4).tolist()}  "
              f"(exact sqrt3 q = {np.sqrt(3) * q0:.4f} I)")
