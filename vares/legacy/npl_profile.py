"""
npl_profile.py

Geometry and membrane pressure-equilibrium stress field of the NPL
airship envelope profile, shared by the gore cutting-pattern optimiser
(``optimise_cutting_pattern_npl_gore.py``) and its Abaqus validation.

Profile (a body of revolution about the +x axis)
------------------------------------------------
    r(x) = B * sqrt(1 - (x / a)^2),   a = A_REAR  (x < 0)
                                      a = A_NOSE  (x >= 0)

Two half-ellipses meet at the maximum-radius station x = 0 (r = B).  The
profile is C1 there (r' = 0 from both sides) but NOT C2 -- the meridional
curvature jumps because A_REAR != A_NOSE, so the equilibrium hoop stress
has a step at x = 0.  That is a genuine feature of the piecewise profile,
not a numerical artefact.

Membrane equilibrium (the optimiser target)
-------------------------------------------
For a surface of revolution under internal pressure p the two principal
radii of curvature are

    r_t = r * sqrt(1 + r'^2)                 (transverse / hoop)
    r_m = -(1 + r'^2)^{3/2} / r''            (meridional; r'' < 0 -> r_m > 0)

An axial "cut" balance of the cap beyond station x gives the meridional
resultant, and Laplace's law then gives the hoop resultant:

    N_m = p * r_t / 2
    N_t = p * r_t * (1 - r_t / (2 r_m))

The membrane wall stresses are sigma_m = N_m / t (warp, along the gore
length) and sigma_t = N_t / t (weft, across the gore).  These are what
``optimise_cutting_pattern_npl_gore`` feeds to VaReS as the spatially
varying target so the optimised gore reproduces the true NPL shape on
inflation.

Units: N, mm, MPa, t (tonne) -- consistent with the rest of the folder.
"""

from __future__ import annotations

import numpy as np


# ── Default NPL parameters (mm), matching FullEnvelope_NoComp.py ──────────────
A_REAR = 5928.8
A_NOSE = 4192.3
B      = 1259.8


def profile_radius(x, a_rear: float = A_REAR, a_nose: float = A_NOSE,
                   b: float = B):
    """Envelope radius r(x).  0 outside [-a_rear, a_nose].  Scalar or array."""
    x = np.asarray(x, dtype=float)
    a = np.where(x < 0.0, a_rear, a_nose)
    inside = (x > -a_rear) & (x < a_nose)
    val = np.zeros_like(x)
    u = np.clip(1.0 - (x / a) ** 2, 0.0, None)
    val = np.where(inside, b * np.sqrt(u), 0.0)
    return val if val.ndim else float(val)


def profile_derivs(x, a_rear: float = A_REAR, a_nose: float = A_NOSE,
                   b: float = B):
    """Return (r, r', r'') at x for the piecewise-ellipse profile.

    Analytic derivatives of r(x) = B sqrt(1 - (x/a)^2):
        r'  = -(B/a^2) x (1 - (x/a)^2)^{-1/2}
        r'' = -(B/a^2)   (1 - (x/a)^2)^{-3/2}
    Evaluated with the side-appropriate semi-axis a.  Returns floats for a
    scalar x, arrays otherwise.
    """
    x = np.asarray(x, dtype=float)
    scalar = (x.ndim == 0)
    x = np.atleast_1d(x)
    a = np.where(x < 0.0, a_rear, a_nose)
    u = 1.0 - (x / a) ** 2
    inside = u > 1.0e-12
    r   = np.zeros_like(x)
    dr  = np.zeros_like(x)
    d2r = np.zeros_like(x)
    us = np.where(inside, u, 1.0)                       # safe base for powers
    r   = np.where(inside, b * np.sqrt(us), 0.0)
    dr  = np.where(inside, -(b / a ** 2) * x * us ** -0.5, 0.0)
    d2r = np.where(inside, -(b / a ** 2) * us ** -1.5, 0.0)
    if scalar:
        return float(r[0]), float(dr[0]), float(d2r[0])
    return r, dr, d2r


def meridian_arclength(x, a_rear: float = A_REAR, a_nose: float = A_NOSE,
                       b: float = B, n_quad: int = 400):
    """Cumulative meridian arc length s(x) = int sqrt(1 + r'^2) dx, measured
    from the rear pole (x = -a_rear).  Returns an array aligned with x
    (which must be sorted ascending).  Used to lay the flat gore out at true
    meridional length instead of the equal-x approximation.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    # Dense integration grid from the rear pole to each station.
    x0 = -a_rear + 1.0e-6
    s = np.zeros_like(x)
    prev_x, prev_s = x0, 0.0
    order = np.argsort(x)
    for idx in order:
        xi = x[idx]
        xs = np.linspace(prev_x, xi, n_quad)
        _, dr, _ = profile_derivs(xs, a_rear, a_nose, b)
        integ = np.sqrt(1.0 + dr ** 2)
        prev_s += float(np.trapz(integ, xs))
        s[idx] = prev_s
        prev_x = xi
    return s


def principal_radii(x, a_rear: float = A_REAR, a_nose: float = A_NOSE,
                    b: float = B):
    """Return (r_t, r_m) transverse and meridional radii of curvature."""
    r, dr, d2r = profile_derivs(x, a_rear, a_nose, b)
    root = np.sqrt(1.0 + dr ** 2)
    r_t = r * root
    # r'' < 0 for the convex profile, so r_m = -(1+r'^2)^{3/2}/r'' > 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        r_m = -(root ** 3) / d2r
    return r_t, r_m


def npl_membrane_stress(x, pressure: float, thickness: float,
                        a_rear: float = A_REAR, a_nose: float = A_NOSE,
                        b: float = B):
    """Membrane wall stresses (sigma_meridional, sigma_hoop) at station x.

        N_m = p r_t / 2
        N_t = p r_t (1 - r_t / (2 r_m))
        sigma = N / t

    ``sigma_meridional`` is the warp (gore-length) target, ``sigma_hoop`` the
    weft (across-gore) target.  Returns floats for scalar x, arrays otherwise.
    """
    r_t, r_m = principal_radii(x, a_rear, a_nose, b)
    N_m = pressure * r_t / 2.0
    N_t = pressure * r_t * (1.0 - r_t / (2.0 * r_m))
    sig_m = N_m / thickness
    sig_t = N_t / thickness
    if np.ndim(x) == 0:
        return float(sig_m), float(sig_t)
    return sig_m, sig_t


# ── Self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p, t = 0.05, 0.9
    print("NPL profile membrane equilibrium check (p=%.3f MPa, t=%.2f mm)" % (p, t))
    print(f"{'x':>10s} {'r':>9s} {'sig_m':>9s} {'sig_t':>9s} "
          f"{'Laplace resid':>14s}")
    for x in [-5000, -3000, -1000, -1e-3, 1e-3, 1000, 2000, 3500]:
        r = profile_radius(x)
        sm, st = npl_membrane_stress(x, p, t)
        r_t, r_m = principal_radii(x)
        # Laplace: N_m/r_m + N_t/r_t = p  ->  should be ~0 residual
        Nm, Nt = sm * t, st * t
        resid = Nm / r_m + Nt / r_t - p
        print(f"{x:10.1f} {r:9.2f} {sm:9.2f} {st:9.2f} {resid:14.2e}")

    # Finite-difference check of the analytic derivatives.
    x0 = 1500.0
    h = 1.0e-2
    r_p = profile_radius(x0 + h); r_m0 = profile_radius(x0 - h)
    r, dr, d2r = profile_derivs(x0)
    dr_fd = (r_p - r_m0) / (2 * h)
    d2r_fd = (r_p - 2 * profile_radius(x0) + r_m0) / h ** 2
    print(f"\nDeriv check at x={x0}:  r'={dr:.6f} (fd {dr_fd:.6f}), "
          f"r''={d2r:.3e} (fd {d2r_fd:.3e})")
