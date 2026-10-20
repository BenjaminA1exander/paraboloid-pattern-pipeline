"""
vares_membrane.py

VaReS (Variation of Reference Strategy) cutting-pattern membrane element,
driving the INDUCED contravariant 2nd Piola-Kirchhoff stress to a
prescribed target by varying the planar material configuration X.

The formulation follows Ch. 5.2 of A. Widhammer, "Variation of Reference
Strategy: Generation of Optimized Cutting Patterns for Textile Fabrics",
Dissertation, Technische Universitaet Muenchen, 2015. This module is an
independent implementation written from that published description; no
code or material from TU Munich was used and no affiliation is claimed.

The element shape is dispatched per element from its node count
(element_types.element_type_of): 4-node bilinear quad (QUAD4) or
3-node linear / constant-strain triangle (TRI3).  A single connectivity
list may therefore mix the two; every assembly loop resolves the
integration rule and DOF count of each element individually.

Method (true VaReS variational residual-force)
----------------------------------------------
Warp / weft are FIXED unit vectors a, b in the global Cartesian frame.
At every integration point:

  1. Contravariant fibre components  a^i = a . G^i,  b^i = b . G^i.
  2. Engineering fibre strains (invariants, square-root-free):
         eps_x = a^i a^j E_{ij},   eps_y = b^i b^j E_{ij},
         gamma = 2 a^i b^j E_{ij}.
  3. Chen 2022 polynomial  ->  physical stress (sx, sy, tau) and the
     physical tangent D = d sigma / d eps.
  4. Contravariant 2nd PK stress (structural-tensor assembly):
         S^{ij} = sx a^i a^j + sy b^i b^j + tau (a^i b^j + a^j b^i).
  5. Residual force (eq 5.42):
         R_a = sum_IP  detJ  (S_chi - S_pre)^{ij} (dE/dX_a)_{ij}
     with the FULL tensor strain derivative (eq 5.34, Christoffel-
     corrected) from vares_kinematics, contracted against the
     contravariant stress.

The prescribed target S_pre^{ij} is the user's PHYSICAL stress
(sx*, sy*, tau*) pushed once into the contravariant basis at a fixed
reference configuration X_ref (Variation of Reference Strategy) and
held constant for all iterations.

CURRENT-FRAME target mode (additive alternative to the frozen S_pre):
pass ``sigma_star_all`` from ``build_phys_targets_field`` to the
assemblers instead of S_pre_all.  The mismatch is then rebuilt in the
CURRENT material frame at every evaluation,
    (S_chi - S_pre)^{ij} = (sx - sx*) a^i a^j + (sy - sy*) b^i b^j
                           + (tau - tau*) (a^i b^j + a^j b^i),
equivalent to updating the frozen S_pre at every iteration.  Because
sigma* is X-independent, the exact tangent only requires replacing the
structural-tensor weights sigma by (sigma - sigma*) in dS_con -- true
Newton is preserved, and no outer relinearisation is needed (the frozen
mode's relinearisation fixed point IS this formulation).

Consistent tangent
-------------------
    K_ab = dR_a / dX_b
         = sum_IP detJ [ (dS_chi/dX_b)^{ij} (dE/dX_a)_{ij}
                         + (S_chi - S_pre)^{ij} (d2E/dX_a dX_b)_{ij} ]
         ( + sum_IP (d detJ/dX_b)(S_chi - S_pre):(dE/dX_a)  if vary_detJ )

This is the EXACT Jacobian of R (true Newton -> quadratic convergence
in the convergence radius).  Note it needs only the first constitutive
tangent D (no d2 sigma / d eps2): R contains S, so dR/dX needs only
dS/dX.  No Gram-Schmidt, no square roots in the strain/stress chain.

The Psi * detJ reference-energy term of Widhammer's full residual
(5.42, second term) is omitted: the Chen 2022 law is non-hyperelastic
so no stored-energy density Psi exists.  The stress-matching residual
above is exact and well-posed without it.

Default integration uses the REFERENCE element area (detJ at X_ref,
constant) so the whole tangent is square-root-free and symmetric-ish;
vary_detJ=True re-adds the current-config detJ derivative (eq 5.40),
the faithful reference-area-varying VaReS at the cost of re-introducing
the |G1 x G2| square root.
"""

from __future__ import annotations

import numpy as np

from element_types import ElementType, QUAD4, element_type_of, elem_dof_ids
from element_types import quad_N as _quad_N
from chen2022_constitutive import chen_stress, chen_stress_linear
from vares_kinematics import (
    IPKinematics, bundle_at, d2E_tensor_pair, fibre_contra, dfibre_contra,
)


# Fixed global warp / weft directions in the flat-pattern (z = 0) plane.
WARP = np.array([1.0, 0.0, 0.0])
WEFT = np.array([0.0, 1.0, 0.0])


def _ddot(A: np.ndarray, B: np.ndarray) -> float:
    """Double contraction of two 2x2 arrays:  sum_ij A_ij B_ij."""
    return float(A[0, 0] * B[0, 0] + A[0, 1] * B[0, 1]
                 + A[1, 0] * B[1, 0] + A[1, 1] * B[1, 1])


# ============================================================
# Constitutive evaluation at one IP (invariant -> Chen -> S^{ij})
# ============================================================

def _stress_at_ip(k: IPKinematics, a_vec: np.ndarray, b_vec: np.ndarray,
                  *, linear_material: bool, sigma_pre=None):
    """Return everything the residual/tangent needs at one IP.

    sigma_pre : optional (3,) physical target (sx*, sy*, tau*).  When given,
        S_con / dS_con are returned as the MISMATCH tensor and its exact
        derivative: the structural-tensor weights become (sigma - sigma*),
        i.e. S_con = (S_chi - S_pre) with S_pre expressed in the CURRENT
        material frame -- equivalent to rebuilding the frozen target at
        every evaluation ("update S_pre each iteration").  sigma* itself is
        X-independent, so the tangent stays exact (true Newton).  With
        sigma_pre=None behaviour is unchanged (raw S_chi; callers subtract
        a frozen S_pre themselves).

    Returns
    -------
    S_con   : (2, 2) contravariant 2nd PK stress S^{ij} (or mismatch, above)
    dS_con  : (8, 2, 2) dS^{ij}/dX_c  (full, exact; of the mismatch if
              sigma_pre is given)
    eps     : (3,) engineering fibre strains (eps_x, eps_y, gamma)
    sigma   : (3,) physical stress (sx, sy, tau) -- always the RAW stress
    info    : Chen branch dict
    """
    E = k.E
    ai = fibre_contra(a_vec, k)                       # (2,)
    bi = fibre_contra(b_vec, k)                       # (2,)
    dai = dfibre_contra(a_vec, k)                     # (8, 2)
    dbi = dfibre_contra(b_vec, k)                     # (8, 2)

    # Engineering fibre strains (invariants)
    Ea = E @ ai
    Eb = E @ bi
    eps_x = float(ai @ Ea)
    eps_y = float(bi @ Eb)
    gamma = float(2.0 * (ai @ Eb))
    eps = np.array([eps_x, eps_y, gamma])

    # Chen polynomial on the engineering strains (passed as a tensor:
    # [0,0]=eps_x, [1,1]=eps_y, 2*[0,1]=gamma).
    eps_tensor = np.array([[eps_x,       0.5 * gamma],
                           [0.5 * gamma, eps_y]])
    stress_fn = chen_stress_linear if linear_material else chen_stress
    S_phys, D_voigt, info = stress_fn(eps_tensor)
    sx, sy, tau = S_phys[0, 0], S_phys[1, 1], S_phys[0, 1]
    sigma = np.array([sx, sy, tau])

    # Structural tensors and contravariant stress.  With sigma_pre given,
    # the weights are the physical mismatch (sigma - sigma*) so that S_con
    # and dS_con below become the current-frame residual tensor and its
    # exact derivative (sigma* does not depend on X).
    aa = np.outer(ai, ai)
    bb = np.outer(bi, bi)
    ab = np.outer(ai, bi) + np.outer(bi, ai)
    if sigma_pre is None:
        wx, wy, wt = sx, sy, tau
    else:
        wx = sx - float(sigma_pre[0])
        wy = sy - float(sigma_pre[1])
        wt = tau - float(sigma_pre[2])
    S_con = wx * aa + wy * bb + wt * ab

    # --- exact derivative dS^{ij}/dX_c -----------------------------
    # DOF count read from the bundle so this works for quads (8) and
    # triangles (6) alike.
    n_dofs = k.dGup.shape[0]
    dS_con = np.zeros((n_dofs, 2, 2))
    for c in range(n_dofs):
        dac = dai[c]                                  # (2,)
        dbc = dbi[c]
        Epc = k.E_part[c]                             # (2, 2)

        # d eps / dX_c  (scalar invariant product rule)
        deps_x = 2.0 * (dac @ Ea) + float(ai @ Epc @ ai)
        deps_y = 2.0 * (dbc @ Eb) + float(bi @ Epc @ bi)
        dgamma = 2.0 * (float(dac @ Eb) + float(ai @ Epc @ bi)
                        + float(ai @ E @ dbc))
        deps = np.array([deps_x, deps_y, dgamma])

        dsx, dsy, dtau = D_voigt @ deps               # d sigma / dX_c

        daa = np.outer(dac, ai) + np.outer(ai, dac)
        dbb = np.outer(dbc, bi) + np.outer(bi, dbc)
        dab = (np.outer(dac, bi) + np.outer(ai, dbc)
               + np.outer(dbc, ai) + np.outer(bi, dac))

        dS_con[c] = (dsx * aa + wx * daa
                     + dsy * bb + wy * dbb
                     + dtau * ab + wt * dab)

    return S_con, dS_con, eps, sigma, info


# ============================================================
# Fixed-reference contravariant target  S_pre^{ij}
# ============================================================

def physical_to_S_pre(sigma_phys: np.ndarray, k_ref: IPKinematics,
                      a_vec: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
    """Push a physical stress (sx, sy, tau) into the contravariant
    basis at the reference IP -- the constant VaReS target S_pre^{ij}.
    """
    ai = fibre_contra(a_vec, k_ref)
    bi = fibre_contra(b_vec, k_ref)
    sx, sy, tau = sigma_phys
    return (sx * np.outer(ai, ai) + sy * np.outer(bi, bi)
            + tau * (np.outer(ai, bi) + np.outer(bi, ai)))


def build_targets(X_ref: np.ndarray, elements, sigma_phys,
                  a_vec: np.ndarray = WARP, b_vec: np.ndarray = WEFT):
    """Precompute the constant per-(elem, IP) target S_pre and the
    reference detJ integration weight.

    Because a mixed mesh has a different number of integration points
    per element (5 for a quad, 1 for a triangle), the results are
    returned as PER-ELEMENT LISTS rather than dense rectangular arrays.

    Returns
    -------
    S_pre_all     : list of (n_ip_e, 2, 2) arrays, one per element
    detJ_ref_all  : list of (n_ip_e,)     arrays, one per element
    """
    sigma_phys = np.asarray(sigma_phys, dtype=float)
    S_pre_all = []
    detJ_ref_all = []
    # X_ref also serves as the spatial dummy here (strain unused for the
    # target build) -- bundle_at needs an X_sp but only the material
    # metric enters S_pre / detJ_ref.
    for elem in elements:
        etype = element_type_of(elem)
        Xr = X_ref[elem]
        S_pre_e = np.zeros((len(etype.ips), 2, 2))
        detJ_e = np.zeros(len(etype.ips))
        for j, (xi, eta) in enumerate(etype.ips):
            k_ref = bundle_at(Xr, Xr, xi, eta, etype=etype,
                              want_second=False)
            S_pre_e[j] = physical_to_S_pre(sigma_phys, k_ref, a_vec, b_vec)
            detJ_e[j] = k_ref.detJ
        S_pre_all.append(S_pre_e)
        detJ_ref_all.append(detJ_e)
    return S_pre_all, detJ_ref_all


def _ip_shape_values(etype: ElementType, xi: float, eta: float) -> np.ndarray:
    """Shape-function values N_k at (xi, eta) for the given element type.

    Used to interpolate the SPATIAL position of an integration point (for a
    spatially varying stress target).  element_types only exposes dN, so N is
    reconstructed here: bilinear form for the quad, barycentric for the tri.
    """
    if etype.n_nodes == 4:
        return _quad_N(xi, eta)
    return np.array([1.0 - xi - eta, xi, eta])          # linear triangle


def build_targets_field(X_ref: np.ndarray, X_spatial: np.ndarray, elements,
                        sigma_fn, a_vec: np.ndarray = WARP,
                        b_vec: np.ndarray = WEFT):
    """Like ``build_targets`` but with a SPATIALLY VARYING physical target.

    ``sigma_fn(pos)`` receives the 3D position of each integration point on the
    fixed spatial (target) surface and returns the physical stress
    (sx, sy, tau) to prescribe there.  The contravariant target S_pre is then
    built at the material reference bundle exactly as in ``build_targets`` (the
    Variation-of-Reference freeze), so only the target VALUE is position
    dependent -- the reference frame is still X_ref.

    Returns the same (S_pre_all, detJ_ref_all) per-element lists.
    """
    S_pre_all = []
    detJ_ref_all = []
    for elem in elements:
        etype = element_type_of(elem)
        Xr = X_ref[elem]
        Xs = X_spatial[elem]
        S_pre_e = np.zeros((len(etype.ips), 2, 2))
        detJ_e = np.zeros(len(etype.ips))
        for j, (xi, eta) in enumerate(etype.ips):
            k_ref = bundle_at(Xr, Xr, xi, eta, etype=etype,
                              want_second=False)
            pos = _ip_shape_values(etype, xi, eta) @ Xs      # IP on target surf
            sigma_phys = np.asarray(sigma_fn(pos), dtype=float)
            S_pre_e[j] = physical_to_S_pre(sigma_phys, k_ref, a_vec, b_vec)
            detJ_e[j] = k_ref.detJ
        S_pre_all.append(S_pre_e)
        detJ_ref_all.append(detJ_e)
    return S_pre_all, detJ_ref_all


def build_phys_targets_field(X_ref: np.ndarray, X_spatial: np.ndarray,
                             elements, sigma_fn):
    """Per-IP PHYSICAL target values for the CURRENT-FRAME target mode.

    Unlike ``build_targets_field``, the target is NOT pushed into a frozen
    contravariant basis: ``sigma_star_all`` holds the raw physical values
    (sx*, sy*, tau*) at each IP's position on the fixed spatial surface
    (X-independent).  Passing it to the assembly as ``sigma_star_all``
    rebuilds the mismatch (sigma - sigma*) . (structural tensors) in the
    CURRENT material frame at every evaluation -- equivalent to updating
    S_pre at every iteration, with the exact consistent tangent -- so no
    outer relinearisation is needed.

    detJ_ref_all is still frozen at X_ref (used only when vary_detJ=False).

    Returns (sigma_star_all, detJ_ref_all) per-element lists.
    """
    sigma_star_all = []
    detJ_ref_all = []
    for elem in elements:
        etype = element_type_of(elem)
        Xr = X_ref[elem]
        Xs = X_spatial[elem]
        sig_e = np.zeros((len(etype.ips), 3))
        detJ_e = np.zeros(len(etype.ips))
        for j, (xi, eta) in enumerate(etype.ips):
            k_ref = bundle_at(Xr, Xr, xi, eta, etype=etype,
                              want_second=False)
            pos = _ip_shape_values(etype, xi, eta) @ Xs      # IP on target surf
            sig_e[j] = np.asarray(sigma_fn(pos), dtype=float)
            detJ_e[j] = k_ref.detJ
        sigma_star_all.append(sig_e)
        detJ_ref_all.append(detJ_e)
    return sigma_star_all, detJ_ref_all


# ============================================================
# Element residual and tangent
# ============================================================

def elem_residual(X_mat_local: np.ndarray, X_sp_local: np.ndarray,
                  S_pre_ips: np.ndarray, detJ_w_ips: np.ndarray,
                  a_vec: np.ndarray = WARP, b_vec: np.ndarray = WEFT,
                  *, etype: ElementType = QUAD4,
                  linear_material: bool = False,
                  vary_detJ: bool = False,
                  sigma_star_ips: np.ndarray | None = None) -> np.ndarray:
    """Per-element residual force R_e (n_dofs,).  No tangent.

    With ``sigma_star_ips`` given ((n_ip, 3) physical targets), the mismatch
    is built in the CURRENT frame inside _stress_at_ip and S_pre_ips is
    ignored (may be None).
    """
    R = np.zeros(etype.n_dofs)
    for j, ((xi, eta), w_ip) in enumerate(zip(etype.ips, etype.wts)):
        k = bundle_at(X_mat_local, X_sp_local, xi, eta,
                      etype=etype, want_second=False)
        sp = None if sigma_star_ips is None else sigma_star_ips[j]
        S_con, _, _, _, _ = _stress_at_ip(
            k, a_vec, b_vec, linear_material=linear_material, sigma_pre=sp)
        Sres = S_con if sp is not None else S_con - S_pre_ips[j]
        W = w_ip * (k.detJ if vary_detJ else detJ_w_ips[j])
        for a in range(etype.n_dofs):
            R[a] += W * _ddot(Sres, k.dE_tens[a])
    return R


def elem_residual_tangent(X_mat_local: np.ndarray, X_sp_local: np.ndarray,
                          S_pre_ips: np.ndarray, detJ_w_ips: np.ndarray,
                          a_vec: np.ndarray = WARP, b_vec: np.ndarray = WEFT,
                          *, etype: ElementType = QUAD4,
                          linear_material: bool = False,
                          vary_detJ: bool = False,
                          sigma_star_ips: np.ndarray | None = None):
    """Per-element (R_e (n_dofs,), K_e (n_dofs, n_dofs)).

    K_e is the exact Jacobian dR_e/dX (true Newton tangent).  With
    ``sigma_star_ips`` given, _stress_at_ip returns the current-frame
    mismatch AND its exact derivative, so the same contraction formulas
    below remain the exact Jacobian; S_pre_ips is ignored (may be None).
    """
    n_dofs = etype.n_dofs
    R = np.zeros(n_dofs)
    K = np.zeros((n_dofs, n_dofs))
    for j, ((xi, eta), w_ip) in enumerate(zip(etype.ips, etype.wts)):
        k = bundle_at(X_mat_local, X_sp_local, xi, eta,
                      etype=etype, want_second=True)
        sp = None if sigma_star_ips is None else sigma_star_ips[j]
        S_con, dS_con, _, _, _ = _stress_at_ip(
            k, a_vec, b_vec, linear_material=linear_material, sigma_pre=sp)
        Sres = S_con if sp is not None else S_con - S_pre_ips[j]
        W = w_ip * (k.detJ if vary_detJ else detJ_w_ips[j])

        dE = k.dE_tens                                # (n_dofs, 2, 2)
        for a in range(n_dofs):
            R[a] += W * _ddot(Sres, dE[a])

        for a in range(n_dofs):
            for b in range(n_dofs):
                d2E_ab = d2E_tensor_pair(k, a, b)
                kab = (_ddot(dS_con[b], dE[a])
                       + _ddot(Sres, d2E_ab))
                K[a, b] += W * kab
                if vary_detJ:
                    K[a, b] += (w_ip * k.ddetJ[b]
                                * _ddot(Sres, dE[a]))
    return R, K


# ============================================================
# Global mesh assembly
# ============================================================

def assemble_residual(X_mat: np.ndarray, X_sp: np.ndarray,
                      elements,
                      S_pre_all, detJ_ref_all,
                      *, linear_material: bool = False,
                      vary_detJ: bool = False,
                      sigma_star_all=None) -> np.ndarray:
    """Global residual force R (2*N_nodes,).

    elements may mix quads and triangles; the element type is dispatched
    per element from its node count, and S_pre_all / detJ_ref_all are the
    per-element lists returned by build_targets.  For the current-frame
    target mode pass ``sigma_star_all`` (from build_phys_targets_field);
    S_pre_all is then unused and may be None.
    """
    N_g = 2 * X_mat.shape[0]
    R = np.zeros(N_g)
    for e, elem in enumerate(elements):
        etype = element_type_of(elem)
        R_e = elem_residual(
            X_mat[elem], X_sp[elem],
            None if S_pre_all is None else S_pre_all[e], detJ_ref_all[e],
            etype=etype, linear_material=linear_material,
            vary_detJ=vary_detJ,
            sigma_star_ips=(None if sigma_star_all is None
                            else sigma_star_all[e]))
        R[elem_dof_ids(elem)] += R_e
    return R


def assemble_residual_tangent(X_mat: np.ndarray, X_sp: np.ndarray,
                              elements,
                              S_pre_all, detJ_ref_all,
                              *, linear_material: bool = False,
                              vary_detJ: bool = False,
                              sigma_star_all=None):
    """Global (R (2N,), K (2N, 2N)).

    Handles mixed quad/triangle meshes: each element's type, local DOF
    count and integration rule are resolved from its node count.  For the
    current-frame target mode pass ``sigma_star_all``; S_pre_all is then
    unused and may be None.
    """
    N_g = 2 * X_mat.shape[0]
    R = np.zeros(N_g)
    K = np.zeros((N_g, N_g))
    for e, elem in enumerate(elements):
        etype = element_type_of(elem)
        R_e, K_e = elem_residual_tangent(
            X_mat[elem], X_sp[elem],
            None if S_pre_all is None else S_pre_all[e], detJ_ref_all[e],
            etype=etype, linear_material=linear_material,
            vary_detJ=vary_detJ,
            sigma_star_ips=(None if sigma_star_all is None
                            else sigma_star_all[e]))
        ids = elem_dof_ids(elem)
        R[ids] += R_e
        K[np.ix_(ids, ids)] += K_e
    return R, K


# ============================================================
# Post-processing: physical stress field at converged X
# ============================================================

def element_centre_physical_stress(X_mat: np.ndarray, X_sp: np.ndarray,
                                   elements,
                                   a_vec: np.ndarray = WARP,
                                   b_vec: np.ndarray = WEFT,
                                   *, linear_material: bool = False
                                   ) -> np.ndarray:
    """Per-element physical (sx, sy, tau) at the element's parametric
    centre (quad: xi=eta=0; triangle: centroid 1/3, 1/3), via the
    invariant fibre strains.  Returns (n_elem, 3).
    """
    out = np.zeros((len(elements), 3))
    for e, elem in enumerate(elements):
        etype = element_type_of(elem)
        xi_c, eta_c = etype.centre
        k = bundle_at(X_mat[elem], X_sp[elem], xi_c, eta_c,
                      etype=etype, want_second=False)
        _, _, _, sigma, _ = _stress_at_ip(
            k, a_vec, b_vec, linear_material=linear_material)
        out[e] = sigma
    return out


def element_centre_phys_residual(X_mat: np.ndarray, X_sp: np.ndarray,
                                 elements: np.ndarray,
                                 sigma_target,
                                 weights=(1.0, 1.0, 1.0),
                                 *, linear_material: bool = False
                                 ) -> np.ndarray:
    """sqrt(sum_k w_k (sigma_k - target_k)^2) at each element centre,
    in PHYSICAL stress -- the user-facing residual readout.
    """
    sigma_target = np.asarray(sigma_target, dtype=float)
    w = np.asarray(weights, dtype=float)
    S = element_centre_physical_stress(X_mat, X_sp, elements,
                                       linear_material=linear_material)
    return np.sqrt(((S - sigma_target) ** 2 * w).sum(axis=1))


def apply_step_mesh(X_mat: np.ndarray, dX_flat: np.ndarray,
                    alpha: float = 1.0) -> np.ndarray:
    """X_mat += alpha * dX_flat (the z column stays untouched).

    dX_flat has length 2 N_nodes and the same (x_0, y_0, x_1, y_1, ...)
    layout as the assembled gradient.
    """
    out = X_mat.copy()
    N_nodes = X_mat.shape[0]
    dX_xy = dX_flat.reshape(N_nodes, 2)
    out[:, :2] += alpha * dX_xy
    return out
