"""
bilinear_quad_lib.py

Pure functions for first- and second-derivative kinematics of a
4-node bilinear quad element, plus the Chen 2022 failure-factor
objective with full analytical gradient and Hessian, plus the
Newton-Raphson solver that uses them.

DESIGN
------
Every function here is pure: it takes all its inputs as explicit
arguments and returns a result. There is NO module-level state for
X_material, X_spatial, parametric point, FF_target, etc. -- all of
those live in the driver script (bilinear_quad_derivatives.py).

This keeps the library importable and testable from anywhere, and
makes the driver script read top-to-bottom as a configuration plus
a sequence of calls.

Conventions
-----------
- Material configuration is constrained to z = 0 -> 4 nodes x 2 DOFs
  = 8 design DOFs total. The z column of X_material is stored but
  never perturbed.
- DOF indexing: a in 0..7; node = a // 2, component = a % 2
  (0 = x, 1 = y).
- Voigt strain uses engineering shear: (E_11, E_22, gamma_xy) with
  gamma_xy = 2 * E_12. The factor of 2 is carried explicitly in the
  dE_voigt builders.
- Voigt stress is (S_11, S_22, S_12). The shear component has no
  factor of 2.

Contents
--------
0.  DOF helpers and small utilities
1.  Atomic first derivative  dG_i/dX_a
2.  Metric tensor first/second derivatives  dG_ij,  d^2 G_ij
3.  Cross-product first/second derivatives  d(G_1xG_2), d^2(G_1xG_2)
4.  Norm of cross product   d||c||, d^2||c||                (eq. 5.49)
5.  Strain tensor derivatives  dE_kl,  d^2 E_kl             (eqs. 5.35, 5.56)
6.  Contravariant metric / dual basis derivatives           (matrix-inverse identity)
7.  Voigt packers           dE_voigt_per_dof,  d2E_voigt_per_pair
8.  Reference objective     J = sum E_kl^2  (gradient, Hessian)
9.  Failure-factor objective    J_FF, grad_FF, hess_FF
10. Least-squares wrapper       J_sq = 0.5 (FF - FF_target)^2
11. Finite-difference helpers and max-rel-err
12. Newton helpers              apply_step,  modified_newton_direction
13. Generic Newton solver       newton_solve
"""

from __future__ import annotations

import numpy as np

from single_bilinear_quad_covariant import (
    dN, covariant_basis, covariant_metric, contravariant_basis,
)
from chen2022_constitutive import (
    chen_stress, chen_failure_factor,
    chen_stress_hessian, chen_FF_hessian,
    chen_stress_linear, chen_stress_hessian_linear,
    chen_stress_covariant,
    physical_target_to_covariant,
    UMAT_PROPS,
)


# Standard 2x2 Gauss product rule plus a centre body point. Same set
# used by the single-element driver in bilinear_quad_derivatives.py.
# The centre point is not part of any Gauss product rule; it is an
# extra objective sample that pins down the element interior in the
# stress-deviation least-squares fit.
_GAUSS_G = 1.0 / np.sqrt(3.0)
GAUSS_PTS_5 = [(-_GAUSS_G, -_GAUSS_G), ( _GAUSS_G, -_GAUSS_G),
               ( _GAUSS_G,  _GAUSS_G), (-_GAUSS_G,  _GAUSS_G),
               ( 0.0,       0.0      )]
GAUSS_WTS_5 = [1.0, 1.0, 1.0, 1.0, 1.0]


# ============================================================
# 0. DOF helpers
# ============================================================

N_DOFS = 8                          # 4 nodes x 2 (x, y) components


def dof_to_node_comp(a: int) -> tuple[int, int]:
    """Decode global DOF index a into (node_index, component_index).

    With the material restricted to the z = 0 plane, each node has
    two DOFs (x, y). Component 0 is x, component 1 is y.
    """
    if not 0 <= a < N_DOFS:
        raise ValueError(f"DOF index {a} outside valid range "
                         f"0..{N_DOFS - 1}.")
    return a // 2, a % 2


def perturb(X_nodes: np.ndarray, a: int, h: float) -> np.ndarray:
    """Return a copy of X_nodes with DOF a shifted by h.

    Only the in-plane (x, y) components are ever perturbed; the z
    column stays untouched.
    """
    Xp = X_nodes.copy()
    node, comp = dof_to_node_comp(a)
    Xp[node, comp] += h
    return Xp


# ============================================================
# 1. Atomic first derivative dG_i/dX_a
# ============================================================
#
# G_i(xi, eta) = sum_k (dN_k/dtheta^i) X_k. Differentiating w.r.t. the
# j-th component of node m:
#   dG_i/dX_a = (dN_m/dtheta^i) * e_j
# G_i is LINEAR in nodal coords, so d^2 G_i/dX_a dX_b = 0.

def dGi_dXa(a: int, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (dG_1/dX_a, dG_2/dX_a) as 3-vectors."""
    node, comp = dof_to_node_comp(a)
    dN_mat = dN(xi, eta)
    e = np.zeros(3); e[comp] = 1.0
    return dN_mat[node, 0] * e, dN_mat[node, 1] * e


# ============================================================
# 2. Metric tensor:  dG_ij/dX_a  and  d^2 G_ij/dX_a dX_b
# ============================================================

def dGij_dXa(X_nodes: np.ndarray, a: int,
             xi: float, eta: float) -> np.ndarray:
    G1, G2, _ = covariant_basis(X_nodes, xi, eta)
    dG1_a, dG2_a = dGi_dXa(a, xi, eta)
    M = np.zeros((2, 2))
    M[0, 0] = 2.0 * (dG1_a @ G1)
    M[1, 1] = 2.0 * (dG2_a @ G2)
    M[0, 1] = dG1_a @ G2 + G1 @ dG2_a
    M[1, 0] = M[0, 1]
    return M


def d2Gij_dXa_dXb(a: int, b: int, xi: float, eta: float) -> np.ndarray:
    """Independent of X_nodes since d^2 G_i / dX^2 = 0."""
    dG1_a, dG2_a = dGi_dXa(a, xi, eta)
    dG1_b, dG2_b = dGi_dXa(b, xi, eta)
    M = np.zeros((2, 2))
    M[0, 0] = 2.0 * (dG1_a @ dG1_b)
    M[1, 1] = 2.0 * (dG2_a @ dG2_b)
    M[0, 1] = dG1_a @ dG2_b + dG1_b @ dG2_a
    M[1, 0] = M[0, 1]
    return M


# ============================================================
# 3. Cross product:  d(G_1xG_2)/dX_a  and  d^2/dX_a dX_b
# ============================================================

def dcross_dXa(X_nodes: np.ndarray, a: int,
               xi: float, eta: float) -> np.ndarray:
    G1, G2, _ = covariant_basis(X_nodes, xi, eta)
    dG1_a, dG2_a = dGi_dXa(a, xi, eta)
    return np.cross(dG1_a, G2) + np.cross(G1, dG2_a)


def d2cross_dXa_dXb(a: int, b: int, xi: float, eta: float) -> np.ndarray:
    dG1_a, dG2_a = dGi_dXa(a, xi, eta)
    dG1_b, dG2_b = dGi_dXa(b, xi, eta)
    return np.cross(dG1_a, dG2_b) + np.cross(dG1_b, dG2_a)


# ============================================================
# 4. Norm of cross product   (eq. 5.49 of the thesis)
# ============================================================

def dnorm_cross_dXa(X_nodes: np.ndarray, a: int,
                    xi: float, eta: float) -> float:
    G1, G2, _ = covariant_basis(X_nodes, xi, eta)
    c = np.cross(G1, G2)
    c_a = dcross_dXa(X_nodes, a, xi, eta)
    return (c @ c_a) / np.linalg.norm(c)


def d2norm_cross_dXa_dXb(X_nodes: np.ndarray, a: int, b: int,
                         xi: float, eta: float) -> float:
    G1, G2, _ = covariant_basis(X_nodes, xi, eta)
    c   = np.cross(G1, G2)
    n   = np.linalg.norm(c)
    c_a = dcross_dXa(X_nodes, a, xi, eta)
    c_b = dcross_dXa(X_nodes, b, xi, eta)
    c_ab = d2cross_dXa_dXb(a, b, xi, eta)
    return (c_a @ c_b + c @ c_ab) / n  -  (c @ c_a) * (c @ c_b) / n**3


# ============================================================
# 5. Strain tensor:  dE_kl,  d^2 E_kl   (X_spatial is FIXED)
# ============================================================

def dEij_dXa(X_material: np.ndarray, a: int,
             xi: float, eta: float) -> np.ndarray:
    return -0.5 * dGij_dXa(X_material, a, xi, eta)


def d2Eij_dXa_dXb(a: int, b: int, xi: float, eta: float) -> np.ndarray:
    return -0.5 * d2Gij_dXa_dXb(a, b, xi, eta)


# ============================================================
# 6. Contravariant metric / dual basis derivatives
# ============================================================

def dGup_ij_dXa(X_nodes: np.ndarray, a: int,
                xi: float, eta: float) -> np.ndarray:
    G1, G2, _ = covariant_basis(X_nodes, xi, eta)
    _, _, _, Gup_ij = contravariant_basis(G1, G2)
    dGij = dGij_dXa(X_nodes, a, xi, eta)
    return -Gup_ij @ dGij @ Gup_ij


def dGup_i_dXa(X_nodes: np.ndarray, a: int,
               xi: float, eta: float) -> np.ndarray:
    """Return (dG^1/dX_a, dG^2/dX_a) as 3-vectors."""
    G1, G2, _ = covariant_basis(X_nodes, xi, eta)
    M_lower = np.stack([G1, G2])
    Gup_ij  = np.linalg.inv(M_lower @ M_lower.T)
    dG1_a, dG2_a = dGi_dXa(a, xi, eta)
    dM_lower = np.stack([dG1_a, dG2_a])
    dGup_ij  = dGup_ij_dXa(X_nodes, a, xi, eta)
    M_up_deriv = dGup_ij @ M_lower + Gup_ij @ dM_lower
    return M_up_deriv[0], M_up_deriv[1]


# ============================================================
# 7. Voigt packers (engineering shear convention)
# ============================================================

def strain_at(X_material: np.ndarray, X_spatial: np.ndarray,
              xi: float, eta: float) -> np.ndarray:
    """Green-Lagrange strain E_kl at parametric point (xi, eta)."""
    G1m, G2m, _ = covariant_basis(X_material, xi, eta)
    g1s, g2s, _ = covariant_basis(X_spatial,  xi, eta)
    return 0.5 * (covariant_metric(g1s, g2s)
                  - covariant_metric(G1m, G2m))


def dE_voigt_per_dof(X_material: np.ndarray,
                     xi: float, eta: float) -> np.ndarray:
    """(N_DOFS, 3) array of dE_voigt/dX_a; engineering shear."""
    out = np.zeros((N_DOFS, 3))
    for a in range(N_DOFS):
        dE = dEij_dXa(X_material, a, xi, eta)
        out[a] = [dE[0, 0], dE[1, 1], 2.0 * dE[0, 1]]
    return out


def d2E_voigt_per_pair(xi: float, eta: float) -> np.ndarray:
    """(N_DOFS, N_DOFS, 3) array of d^2 E_voigt/dX_a dX_b.

    Independent of X_material since G_i is linear in nodal coords.
    """
    out = np.zeros((N_DOFS, N_DOFS, 3))
    for a in range(N_DOFS):
        for b in range(N_DOFS):
            d2E = d2Eij_dXa_dXb(a, b, xi, eta)
            out[a, b] = [d2E[0, 0], d2E[1, 1], 2.0 * d2E[0, 1]]
    return out


# ============================================================
# 8. Reference objective J = sum E_kl^2 (gradient and Hessian)
# ============================================================

def J_sumE2(X_material: np.ndarray, X_spatial: np.ndarray,
            xi: float, eta: float) -> float:
    E = strain_at(X_material, X_spatial, xi, eta)
    return float(np.sum(E * E))


def grad_J_sumE2(X_material: np.ndarray, X_spatial: np.ndarray,
                 xi: float, eta: float) -> np.ndarray:
    E = strain_at(X_material, X_spatial, xi, eta)
    g = np.zeros(N_DOFS)
    for a in range(N_DOFS):
        dE = dEij_dXa(X_material, a, xi, eta)
        g[a] = 2.0 * np.sum(E * dE)
    return g


def hess_J_sumE2(X_material: np.ndarray, X_spatial: np.ndarray,
                 xi: float, eta: float) -> np.ndarray:
    E = strain_at(X_material, X_spatial, xi, eta)
    H = np.zeros((N_DOFS, N_DOFS))
    dE_all = [dEij_dXa(X_material, a, xi, eta) for a in range(N_DOFS)]
    for a in range(N_DOFS):
        for b in range(N_DOFS):
            d2E = d2Eij_dXa_dXb(a, b, xi, eta)
            H[a, b] = 2.0 * np.sum(dE_all[a] * dE_all[b] + E * d2E)
    return H


# ============================================================
# 9. Chen 2022 failure-factor objective and its derivatives
# ============================================================

def J_FF(X_material: np.ndarray, X_spatial: np.ndarray,
         xi: float, eta: float) -> float:
    """Raw failure factor at one IP."""
    E_kl       = strain_at(X_material, X_spatial, xi, eta)
    S_kl, _, _ = chen_stress(E_kl)
    return chen_failure_factor(S_kl)


def dFF_dS_voigt(S_kl: np.ndarray,
                 props: np.ndarray = UMAT_PROPS) -> np.ndarray:
    """Closed-form dFF/d(sx, sy, tau) -- linear in stress."""
    X_, Y_, S_sh_ = props[12], props[13], props[14]
    alpha_, beta_, gamma_ = props[15], props[16], props[17]
    xi_, eta_p_ = props[18], props[19]
    AX2  = (alpha_ * X_)**2
    BY2  = (beta_  * Y_)**2
    GXYF = (gamma_**2) * X_ * Y_

    sx, sy, tau = S_kl[0, 0], S_kl[1, 1], S_kl[0, 1]
    abs_tau = abs(tau)
    sgn_tau = 1.0 if tau >= 0.0 else -1.0

    dFF_dsx  = (2.0 * sx / AX2 - sy / GXYF
                + 1.0 / (xi_ * X_)  + abs_tau / (X_ * S_sh_))
    dFF_dsy  = (2.0 * sy / BY2 - sx / GXYF
                + 1.0 / (eta_p_ * Y_) + abs_tau / (Y_ * S_sh_))
    dFF_dtau = (sgn_tau * sx / (X_ * S_sh_)
                + sgn_tau * sy / (Y_ * S_sh_))
    return np.array([dFF_dsx, dFF_dsy, dFF_dtau])


def grad_FF(X_material: np.ndarray, X_spatial: np.ndarray,
            xi: float, eta: float) -> np.ndarray:
    """Chain rule  dJ/dX_a = dFF/dS_i * D_ij * dE_voigt[a, j]."""
    E_kl              = strain_at(X_material, X_spatial, xi, eta)
    S_kl, D_voigt, _  = chen_stress(E_kl)
    dFF_dS            = dFF_dS_voigt(S_kl)
    dE_voigt          = dE_voigt_per_dof(X_material, xi, eta)
    dS_voigt = dE_voigt @ D_voigt.T                     # (N_DOFS, 3)
    return dS_voigt @ dFF_dS                            # (N_DOFS,)


def hess_FF(X_material: np.ndarray, X_spatial: np.ndarray,
            xi: float, eta: float) -> np.ndarray:
    """Full analytical Hessian = termI + termII + termIII.

    See driver-script docstring section 9 for the chain-rule derivation.
    """
    E_kl                = strain_at(X_material, X_spatial, xi, eta)
    S_kl, D_voigt, _    = chen_stress(E_kl)
    dFF_dS              = dFF_dS_voigt(S_kl)
    H_FF                = chen_FF_hessian(S_kl)
    M                   = chen_stress_hessian(E_kl)
    dE_voigt            = dE_voigt_per_dof(X_material, xi, eta)
    d2E_voigt           = d2E_voigt_per_pair(xi, eta)

    dS_voigt = dE_voigt @ D_voigt.T                     # (N_DOFS, 3)

    # Term I -- FF curvature in stress space
    term_I = dS_voigt @ H_FF @ dS_voigt.T

    # Term II -- S(E) curvature
    C_jk = np.einsum('i,ijk->jk', dFF_dS, M)
    term_II = dE_voigt @ C_jk @ dE_voigt.T

    # Term III -- kinematic curvature
    v_j = dFF_dS @ D_voigt
    term_III = d2E_voigt @ v_j

    H = term_I + term_II + term_III
    return 0.5 * (H + H.T)


# ============================================================
# 10. Least-squares wrapper J_sq = 0.5 (FF - FF_target)^2
# ============================================================
#
# Convex-ish reformulation that's bounded below (min = 0 when
# FF == FF_target). Smooth, no log singularity, plays nicely with
# the Gauss-Newton structure of its Hessian.

def J_sq(X_material: np.ndarray, X_spatial: np.ndarray,
         xi: float, eta: float, FF_target: float) -> float:
    return 0.5 * (J_FF(X_material, X_spatial, xi, eta) - FF_target) ** 2


def grad_sq(X_material: np.ndarray, X_spatial: np.ndarray,
            xi: float, eta: float, FF_target: float) -> np.ndarray:
    ff   = J_FF(X_material, X_spatial, xi, eta)
    g_ff = grad_FF(X_material, X_spatial, xi, eta)
    return (ff - FF_target) * g_ff


def hess_sq(X_material: np.ndarray, X_spatial: np.ndarray,
            xi: float, eta: float, FF_target: float) -> np.ndarray:
    ff   = J_FF(X_material, X_spatial, xi, eta)
    g_ff = grad_FF(X_material, X_spatial, xi, eta)
    H_ff = hess_FF(X_material, X_spatial, xi, eta)
    # Full Hessian = outer(g_ff, g_ff) + (FF - FF_target) * H_ff.
    # Outer product is rank 1 and psd; correction term is signed.
    H = np.outer(g_ff, g_ff) + (ff - FF_target) * H_ff
    return 0.5 * (H + H.T)


# ============================================================
# 11. Finite-difference helpers
# ============================================================

def fd_first(f, X_nodes: np.ndarray, a: int,
             xi: float, eta: float, h: float = 1.0e-6) -> np.ndarray:
    """Central FD: (f(X + h e_a) - f(X - h e_a)) / (2h)."""
    f_p = np.asarray(f(perturb(X_nodes, a,  h), xi, eta), dtype=float)
    f_m = np.asarray(f(perturb(X_nodes, a, -h), xi, eta), dtype=float)
    return (f_p - f_m) / (2.0 * h)


def fd_second(f, X_nodes: np.ndarray, a: int, b: int,
              xi: float, eta: float, h: float = 1.0e-4) -> np.ndarray:
    """Central FD of the first derivative w.r.t. X_b."""
    def first_at(X):
        f_p = np.asarray(f(perturb(X, a,  h), xi, eta), dtype=float)
        f_m = np.asarray(f(perturb(X, a, -h), xi, eta), dtype=float)
        return (f_p - f_m) / (2.0 * h)
    return (first_at(perturb(X_nodes, b,  h))
            - first_at(perturb(X_nodes, b, -h))) / (2.0 * h)


def max_rel_err(analytical, numerical) -> float:
    """Largest relative error between two arrays, with a floor."""
    diff = np.linalg.norm(analytical - numerical)
    scale = max(np.linalg.norm(analytical),
                np.linalg.norm(numerical), 1.0e-12)
    return diff / scale


# ============================================================
# 12. Newton helpers
# ============================================================

def apply_step(X_mat: np.ndarray, dX_flat: np.ndarray,
               alpha: float = 1.0) -> np.ndarray:
    """Add alpha * dX_flat (length-N_DOFS) to the (x, y) DOFs of X_mat.

    The z column stays untouched -- this is what enforces the
    "material is planar" constraint structurally.
    """
    out = X_mat.copy()
    for a in range(N_DOFS):
        node, comp = dof_to_node_comp(a)
        out[node, comp] += alpha * dX_flat[a]
    return out


def modified_newton_direction(g: np.ndarray, H: np.ndarray,
                              eig_floor: float = 1.0e-6,
                              cond_cap: float = 1.0e6):
    """Solve (H + lambda I) dX = -g with two safeguards:

    1. Positive-definiteness: lambda >= -min(eigvals) + eig_floor.
    2. Condition-number cap: lambda raised so max_eig/min_eig <= cond_cap.

    Without (2), a heavily indefinite H gives a near-singular H_reg
    whose inverse amplifies the gradient into a wildly oversized step.
    """
    eigvals = np.linalg.eigvalsh(H)
    min_e = float(eigvals.min())
    max_e = float(eigvals.max())

    lam = max(0.0, eig_floor - min_e)
    shifted_min = min_e + lam
    shifted_max = max_e + lam
    target_min  = shifted_max / cond_cap
    if shifted_min < target_min:
        lam += (target_min - shifted_min)

    H_reg = H + lam * np.eye(H.shape[0])
    return np.linalg.solve(H_reg, -g), lam


# ============================================================
# 13. Generic Newton solver
# ============================================================
#
# Wraps modified_newton_direction + Armijo backtracking + a
# fixed step cap. Returns (X_final, history) where history is a list
# of dicts with the per-iteration diagnostics.
#
# obj_value, obj_grad, obj_hess: callables that take only X_material
# (the driver binds any extra arguments like X_spatial, xi, eta,
# FF_target via closures or functools.partial).

def newton_solve(X_initial: np.ndarray,
                 obj_value, obj_grad, obj_hess,
                 *, tol_g: float = 1.0e-9,
                 max_it: int = 50,
                 max_step: float = 0.05,
                 c1: float = 1.0e-4,
                 verbose: bool = False,
                 display_extra=None):
    """Damped Newton with Armijo backtracking and step cap.

    Parameters
    ----------
    X_initial : (4, 3) ndarray
        Starting material configuration. The z column is preserved
        identically by apply_step.
    obj_value, obj_grad, obj_hess : callables X -> scalar / vector / matrix
        Objective and its first/second derivatives. These must be
        callables of X only -- bind any other arguments in the driver.
    tol_g : float
        Stop when ||obj_grad|| falls below this.
    max_it : int
        Iteration cap.
    max_step : float
        Maximum L2 norm of the per-iteration Newton step (the step
        cap). Set higher for problems that need bigger jumps.
    c1 : float
        Armijo sufficient-decrease constant.
    verbose : bool
        Print a per-iteration row if True.
    display_extra : callable or None
        Optional callable X -> float printed alongside the objective
        in the verbose log (e.g. raw FF when minimising J_sq).

    Returns
    -------
    X_final : (4, 3) ndarray
    history : list of dict
        Per-iteration diagnostics.
    """
    X = X_initial.copy()
    history = []
    stall_counter = 0

    if verbose:
        extra_col = "  extra" if display_extra is not None else ""
        print(f"  {'iter':>4s}  {'J':>12s}{extra_col}  {'|g|':>10s}  "
              f"{'|dX|':>10s}  {'alpha':>6s}  {'lambda':>10s}  "
              f"{'min eig(H)':>12s}")
        print("  " + "-" * (78 + (8 if display_extra else 0)))

    for k in range(max_it):
        J_k  = obj_value(X)
        g_k  = obj_grad(X)
        H_k  = obj_hess(X)
        gnorm = float(np.linalg.norm(g_k))
        min_eig = float(np.linalg.eigvalsh(H_k).min())

        if gnorm < tol_g:
            if verbose:
                extra = (f"  {display_extra(X):8.4f}"
                         if display_extra is not None else "")
                print(f"  {k:>4d}  {J_k:12.4e}{extra}  {gnorm:10.3e}  "
                      f"{'-':>10s}  {'-':>6s}  {'-':>10s}  "
                      f"{min_eig:12.3e}    converged")
            history.append({"k": k, "J": J_k, "gnorm": gnorm,
                            "status": "converged"})
            return X, history

        dX, lam = modified_newton_direction(g_k, H_k)
        step_len = np.linalg.norm(dX)
        if step_len > max_step:
            dX = dX * (max_step / step_len)

        # Armijo backtracking on obj_value (same function as g, H).
        alpha = 1.0
        accepted = False
        for _ in range(40):
            X_trial = apply_step(X, dX, alpha)
            if obj_value(X_trial) <= J_k + c1 * alpha * (g_k @ dX):
                accepted = True
                break
            alpha *= 0.5

        eff_step = alpha * float(np.linalg.norm(dX))

        if verbose:
            extra = (f"  {display_extra(X):8.4f}"
                     if display_extra is not None else "")
            print(f"  {k:>4d}  {J_k:12.4e}{extra}  {gnorm:10.3e}  "
                  f"{eff_step:10.3e}  {alpha:6.3f}  "
                  f"{lam:10.3e}  {min_eig:12.3e}")

        history.append({"k": k, "J": J_k, "gnorm": gnorm,
                        "alpha": alpha, "step": eff_step,
                        "lambda": lam, "min_eig": min_eig,
                        "status": "accepted" if accepted else "stalled"})

        if accepted:
            X = X_trial
            stall_counter = 0
        else:
            stall_counter += 1
            # Last-ditch gradient direction
            grad_dir = -g_k / max(gnorm, 1e-30) * max_step
            for _ in range(20):
                X_trial = apply_step(X, grad_dir, alpha)
                if obj_value(X_trial) < J_k:
                    X = X_trial
                    break
                alpha *= 0.5

        if eff_step < 1.0e-10 and stall_counter >= 2:
            if verbose:
                print(f"  stalled (effective step < 1e-10 for "
                      f"{stall_counter} iters) -- exiting")
            history[-1]["status"] = "stalled_exit"
            break

    else:
        if verbose:
            print(f"  reached max_it = {max_it} without ||g|| < {tol_g}")

    return X, history


# ============================================================
# 14. Stress-deviation residuals (parameterised targets/weights)
# ============================================================
#
# The single-element driver hard-codes its own S_TARGET / S_WEIGHTS
# and inlines an equivalent helper. This version takes both as dict
# arguments so the mesh-assembly code below can be reused for any
# uniform / per-element target field without touching the lib.

def stress_residuals(S_kl: np.ndarray,
                     targets: dict, weights: dict) -> list:
    """Return per-functional residual data at one IP.

    For each scalar functional phi_k of the Voigt stress
    S_voigt = (sx, sy, txy) = (S_11, S_22, S_12) the result is

        (name, r_k, dr_k/dS, d2 r_k / dS dS, w_k)

    where r_k = phi_k(S) - targets[name] and w_k = weights[name].

    The four functionals are sx, sy, txy and ts = txy * (sx + sy).
    Only ts has non-zero stress-space curvature.
    """
    sx, sy, txy = S_kl[0, 0], S_kl[1, 1], S_kl[0, 1]
    ZERO3 = np.zeros((3, 3))
    d2ts = np.array([[0.0, 0.0, 1.0],
                     [0.0, 0.0, 1.0],
                     [1.0, 1.0, 0.0]])
    return [
        ("sx",  sx  - targets["sx"],
                np.array([1.0, 0.0, 0.0]),       ZERO3, weights["sx"]),
        ("sy",  sy  - targets["sy"],
                np.array([0.0, 1.0, 0.0]),       ZERO3, weights["sy"]),
        ("txy", txy - targets["txy"],
                np.array([0.0, 0.0, 1.0]),       ZERO3, weights["txy"]),
        ("ts",  txy * (sx + sy) - targets["ts"],
                np.array([txy, txy, sx + sy]),   d2ts,  weights["ts"]),
    ]


# ============================================================
# 14b. Physical-strain conversion (full Gram-Schmidt)  --
#      POST-PROCESSING ONLY for the mesh optimisation path
# ============================================================
#
# >>> IMPORTANT:  these helpers are NO LONGER part of the per-element
# >>> chain rule used by elem_obj_grad_hess / elem_obj_only /
# >>> elem_obj_grad below.  They survive as:
# >>>   - the single-element driver (bilinear_quad_derivatives.py)
# >>>     consumes them directly,
# >>>   - test_physical_strain_chain_rule.py validates them against FD,
# >>>   - compute_physical_stress_postprocess (section 17) calls
# >>>     strain_at_physical at the converged X to produce physical
# >>>     stresses for output and visualisation.
# >>>
# >>> Why they were dropped from the optimiser path: differentiating
# >>> T_E(G_cov) through the chain rule introduces 1/|G_1|^k singular
# >>> derivatives at the cylinder boundary (sin(theta) -> 0), driving
# >>> the Hessian condition number to ~1e23 and forcing the Levenberg-
# >>> Marquardt shift to dominate the Newton step.  The covariant
# >>> optimiser path keeps T_E as a per-IP CONSTANT (see section 15) so
# >>> the chain rule stays polynomial.
#
# chen2022_constitutive.chen_stress treats its inputs (ex, ey, gamma)
# as PHYSICAL engineering strains, calibrated against uniaxial test
# data. The covariant Green-Lagrange components returned by strain_at
# carry the parametric Jacobian (factor of (h/2)^2 from the basis
# magnitudes), so feeding them straight into Chen gives stresses that
# scale wrongly with element size.
#
# This block builds an ORTHONORMAL material frame at each IP via
# Gram-Schmidt on (G_1, G_2):
#
#     e_1 = G_1 / |G_1|
#     e_2 = (G_2 - (G_12/G_11) G_1) / |...|
#
# and expresses Green-Lagrange strain in that frame. The contravariant
# components of e_alpha in the (G_1, G_2) basis are
#
#     p := (e_1)^1 = 1 / sqrt(G_11)
#     q := (e_2)^1 = -G_12 / sqrt(G_11 J)
#     r := (e_2)^2 = sqrt(G_11 / J)
#
# where J = det(G) = G_11 G_22 - G_12^2.
#
# With A = [[p, 0], [q, r]] the physical strain tensor is
# E_phys = A E_cov A^T and the engineering-Voigt transform is
# eps_voigt_phys = T E_voigt_cov with
#
#     T = [[p^2,     0,    0  ],
#          [q^2,    r^2,   q r],
#          [2 p q,   0,    p r]]
#
# In the orthogonal limit G_12 -> 0 this reduces to the diagonal
# s = [1/G_11, 1/G_22, 1/sqrt(G_11 G_22)] scaling, so the formulation
# is back-compatible for rectangular elements but stays correct
# under skew.
#
# All chain-rule pieces -- T(X), dT/dX_a, d^2 T/dX_a dX_b -- are
# built by propagating (G_11, G_22, G_12, J) derivatives through
# (p, q, r) and then through T's polynomial entries.

def _metric_first_derivs(X_material: np.ndarray,
                         xi: float, eta: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dG_11/dX_a, dG_22/dX_a, dG_12/dX_a) packed as (N_DOFS,) arrays."""
    dG11 = np.zeros(N_DOFS)
    dG22 = np.zeros(N_DOFS)
    dG12 = np.zeros(N_DOFS)
    for a in range(N_DOFS):
        dGij = dGij_dXa(X_material, a, xi, eta)
        dG11[a] = dGij[0, 0]
        dG22[a] = dGij[1, 1]
        dG12[a] = dGij[0, 1]
    return dG11, dG22, dG12


def _metric_second_derivs(xi: float, eta: float
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(d^2 G_11, d^2 G_22, d^2 G_12) as (N_DOFS, N_DOFS) arrays.

    Independent of X_material (G_i is linear in nodal coords).
    """
    d2G11 = np.zeros((N_DOFS, N_DOFS))
    d2G22 = np.zeros((N_DOFS, N_DOFS))
    d2G12 = np.zeros((N_DOFS, N_DOFS))
    for a in range(N_DOFS):
        for b in range(N_DOFS):
            d2Gij = d2Gij_dXa_dXb(a, b, xi, eta)
            d2G11[a, b] = d2Gij[0, 0]
            d2G22[a, b] = d2Gij[1, 1]
            d2G12[a, b] = d2Gij[0, 1]
    return d2G11, d2G22, d2G12


def _pqr_and_derivs(G11: float, G22: float, G12: float,
                    dG11: np.ndarray, dG22: np.ndarray, dG12: np.ndarray,
                    d2G11: np.ndarray, d2G22: np.ndarray, d2G12: np.ndarray
                    ) -> tuple:
    """Compute (p, q, r) and their first and second derivatives.

    Returns
    -------
    (p, q, r)             : scalars
    (dp, dq, dr)          : (N_DOFS,) each
    (d2p, d2q, d2r)       : (N_DOFS, N_DOFS) each
    """
    # -- J = G_11 G_22 - G_12^2 and its derivatives -----------------
    J   = G11 * G22 - G12 * G12
    dJ  = dG11 * G22 + G11 * dG22 - 2.0 * G12 * dG12
    d2J = (d2G11 * G22
           + np.outer(dG11, dG22) + np.outer(dG22, dG11)
           + G11 * d2G22
           - 2.0 * np.outer(dG12, dG12)
           - 2.0 * G12 * d2G12)

    # -- p = G_11^(-1/2) --------------------------------------------
    p   = 1.0 / np.sqrt(G11)
    dp  = -0.5 * (p ** 3) * dG11
    d2p = (0.75 * (p ** 5) * np.outer(dG11, dG11)
           - 0.5  * (p ** 3) * d2G11)

    # -- r = (G_11 / J)^(1/2)  via  dr = 0.5 r K,  K = dG_11/G_11 - dJ/J
    r   = np.sqrt(G11 / J)
    K   = dG11 / G11 - dJ / J                                       # (N_DOFS,)
    dr  = 0.5 * r * K
    # dK/dX_b = d2G_11/G_11 - outer(dG_11, dG_11)/G_11^2
    #         - d2J/J + outer(dJ, dJ)/J^2
    dK  = (d2G11 / G11 - np.outer(dG11, dG11) / (G11 * G11)
           - d2J / J + np.outer(dJ, dJ) / (J * J))
    d2r = 0.25 * r * np.outer(K, K) + 0.5 * r * dK

    # -- q = -G_12 * h^(-1/2),  h = G_11 J --------------------------
    h      = G11 * J
    sqrt_h = np.sqrt(h)
    q      = -G12 / sqrt_h
    dh     = dG11 * J + G11 * dJ
    d2h    = (d2G11 * J
              + np.outer(dG11, dJ) + np.outer(dJ, dG11)
              + G11 * d2J)
    # dq/dX_a = -dG_12_a / sqrt(h)  +  0.5 G_12 dh_a / h^(3/2)
    dq = -dG12 / sqrt_h + 0.5 * G12 * dh / (h * sqrt_h)
    # d^2 q (see derivation in the docstring of section 14b)
    d2q = (-d2G12 / sqrt_h
           + 0.5 * (np.outer(dG12, dh) + np.outer(dh, dG12)) / (h * sqrt_h)
           + 0.5 * G12 * d2h / (h * sqrt_h)
           - 0.75 * G12 * np.outer(dh, dh) / (h * h * sqrt_h))

    return (p, q, r), (dp, dq, dr), (d2p, d2q, d2r)


def _T_and_derivs(p: float, q: float, r: float,
                  dp: np.ndarray, dq: np.ndarray, dr: np.ndarray,
                  d2p: np.ndarray, d2q: np.ndarray, d2r: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the Voigt transformation T (3x3), dT (N_DOFS, 3, 3), and
    d2T (N_DOFS, N_DOFS, 3, 3).

    T[i, j] is a polynomial in (p, q, r). Only six entries are
    non-zero: T[0,0], T[1,0], T[1,1], T[1,2], T[2,0], T[2,2]. The
    other three entries (T[0,1], T[0,2], T[2,1]) are identically zero
    by Gram-Schmidt construction (A[0,1] = 0).
    """
    N = N_DOFS

    T = np.zeros((3, 3))
    T[0, 0] = p * p
    T[1, 0] = q * q
    T[1, 1] = r * r
    T[1, 2] = q * r
    T[2, 0] = 2.0 * p * q
    T[2, 2] = p * r

    dT = np.zeros((N, 3, 3))
    dT[:, 0, 0] = 2.0 * p * dp
    dT[:, 1, 0] = 2.0 * q * dq
    dT[:, 1, 1] = 2.0 * r * dr
    dT[:, 1, 2] = dq * r + q * dr
    dT[:, 2, 0] = 2.0 * (dp * q + p * dq)
    dT[:, 2, 2] = dp * r + p * dr

    d2T = np.zeros((N, N, 3, 3))
    d2T[:, :, 0, 0] = 2.0 * (np.outer(dp, dp) + p * d2p)
    d2T[:, :, 1, 0] = 2.0 * (np.outer(dq, dq) + q * d2q)
    d2T[:, :, 1, 1] = 2.0 * (np.outer(dr, dr) + r * d2r)
    d2T[:, :, 1, 2] = (np.outer(dq, dr) + np.outer(dr, dq)
                       + q * d2r + r * d2q)
    d2T[:, :, 2, 0] = 2.0 * (np.outer(dp, dq) + np.outer(dq, dp)
                             + p * d2q + q * d2p)
    d2T[:, :, 2, 2] = (np.outer(dp, dr) + np.outer(dr, dp)
                       + p * d2r + r * d2p)

    return T, dT, d2T


def strain_at_physical(X_material: np.ndarray, X_spatial: np.ndarray,
                       xi: float, eta: float) -> np.ndarray:
    """Green-Lagrange strain in the Gram-Schmidt orthonormal material
    frame -- the engineering-strain form Chen 2022 was calibrated
    against.

    Returns the 2x2 physical-strain tensor:
        E_phys = A E_cov A^T   with A = [[p, 0], [q, r]].
    """
    E_cov = strain_at(X_material, X_spatial, xi, eta)
    G1, G2, _ = covariant_basis(X_material, xi, eta)
    G11, G22, G12 = G1 @ G1, G2 @ G2, G1 @ G2
    J = G11 * G22 - G12 * G12
    p = 1.0 / np.sqrt(G11)
    r = np.sqrt(G11 / J)
    q = -G12 / np.sqrt(G11 * J)
    A = np.array([[p, 0.0],
                  [q, r  ]])
    return A @ E_cov @ A.T


def dE_voigt_phys_per_dof(X_material: np.ndarray, X_spatial: np.ndarray,
                          xi: float, eta: float) -> np.ndarray:
    """(N_DOFS, 3) array of d(eps_voigt_phys)/dX_a.

    eps_voigt_phys = T(X) . E_voigt_cov(X),  so
        d/dX_a = dT_a . E_voigt_cov + T . dE_voigt_cov_a
    """
    E_cov = strain_at(X_material, X_spatial, xi, eta)
    E_voigt_cov = np.array([E_cov[0, 0], E_cov[1, 1], 2.0 * E_cov[0, 1]])

    G1, G2, _ = covariant_basis(X_material, xi, eta)
    G11, G22, G12 = G1 @ G1, G2 @ G2, G1 @ G2

    dG11,  dG22,  dG12  = _metric_first_derivs (X_material, xi, eta)
    d2G11, d2G22, d2G12 = _metric_second_derivs(xi, eta)

    (p, q, r), (dp, dq, dr), (d2p, d2q, d2r) = _pqr_and_derivs(
        G11, G22, G12,
        dG11, dG22, dG12,
        d2G11, d2G22, d2G12)

    T, dT, _ = _T_and_derivs(p, q, r, dp, dq, dr, d2p, d2q, d2r)

    dE_voigt_cov = dE_voigt_per_dof(X_material, xi, eta)

    # eps_voigt_phys[k] = T[k, m] E_voigt_cov[m]
    # d/dX_a:    sum_m  dT[a, k, m] E_voigt_cov[m] + T[k, m] dE_voigt_cov[a, m]
    return (dT @ E_voigt_cov                # (N_DOFS, 3)
            + dE_voigt_cov @ T.T)           # (N_DOFS, 3)


def d2E_voigt_phys_per_pair(X_material: np.ndarray, X_spatial: np.ndarray,
                            xi: float, eta: float) -> np.ndarray:
    """(N_DOFS, N_DOFS, 3) array.

    Full product rule on T(X) . E_voigt_cov(X):
        d^2 / dX_a dX_b =
              d2T_ab . E_voigt_cov
            + dT_a . dE_voigt_cov_b
            + dT_b . dE_voigt_cov_a
            + T . d2E_voigt_cov_ab
    """
    E_cov = strain_at(X_material, X_spatial, xi, eta)
    E_voigt_cov = np.array([E_cov[0, 0], E_cov[1, 1], 2.0 * E_cov[0, 1]])

    G1, G2, _ = covariant_basis(X_material, xi, eta)
    G11, G22, G12 = G1 @ G1, G2 @ G2, G1 @ G2

    dG11,  dG22,  dG12  = _metric_first_derivs (X_material, xi, eta)
    d2G11, d2G22, d2G12 = _metric_second_derivs(xi, eta)

    (p, q, r), (dp, dq, dr), (d2p, d2q, d2r) = _pqr_and_derivs(
        G11, G22, G12,
        dG11, dG22, dG12,
        d2G11, d2G22, d2G12)

    T, dT, d2T = _T_and_derivs(p, q, r, dp, dq, dr, d2p, d2q, d2r)

    dE_voigt_cov  = dE_voigt_per_dof(X_material, xi, eta)
    d2E_voigt_cov = d2E_voigt_per_pair(xi, eta)

    # Term 1:  d2T . E_voigt_cov           shape (N, N, 3)
    term1 = d2T @ E_voigt_cov

    # Term 2 + 3:  dT_a . dE_cov_b + dT_b . dE_cov_a
    # dT has shape (N, 3, 3), dE_cov has shape (N, 3).
    # We want out[a, b, k] = sum_m dT[a, k, m] dE_cov[b, m]
    #                     + sum_m dT[b, k, m] dE_cov[a, m]
    # einsum makes this explicit and avoids a transpose mistake.
    term23 = (np.einsum('akm,bm->abk', dT, dE_voigt_cov)
              + np.einsum('bkm,am->abk', dT, dE_voigt_cov))

    # Term 4:  T . d2E_cov_ab              shape (N, N, 3)
    term4 = d2E_voigt_cov @ T.T

    return term1 + term23 + term4


# ============================================================
# 15. Per-element stress-deviation J, grad, Hessian -- COVARIANT path
# ============================================================
#
# DESIGN (see section 14b for context).
#
# OLD optimiser path computed eps_phys = T_E(X) @ E_cov(X) and
# differentiated that product fully -- carrying d T_E / d X through
# every chain-rule term.  T_E goes as 1/sqrt(G_11), so its derivatives
# scale as 1/sqrt(G_11)^k and EXPLODE at the cylinder mesh boundary
# (sin(theta) -> 0).  Hessian condition number reached ~1e23; Newton
# degenerated into infinitesimal gradient descent under the Levenberg
# shift.
#
# NEW covariant path  (this section):
#
#   (a) Build covariant Green-Lagrange E_cov(X) and its first/second
#       derivatives via strain_at, dE_voigt_per_dof, d2E_voigt_per_pair
#       -- all polynomial in X.
#
#   (b) Evaluate the Chen polynomial DIRECTLY on E_cov via
#       chen_stress_covariant -- it returns S^{ij} (the CONTRAVARIANT
#       2nd PK in the curvilinear basis) and its tangent D, curvature
#       M.  The polynomial is applied without any Gram-Schmidt
#       pull-back, so D and M are the standard Chen tangents evaluated
#       at E_cov.  Polynomial in X.
#
#   (c) BRIDGE TO THE USER'S PHYSICAL TARGET.  The user specifies
#       targets in PHYSICAL stress units (sx = 5 MPa etc).  We map
#       that target into the curvilinear basis ONCE PER IP, using
#           S^{ij}_target_voigt  =  T_E(G_cov).T  @  sigma_phys_target_voigt
#       and feed it to stress_residuals.  T_E is rebuilt at the
#       CURRENT iterate's metric each Newton step -- so the covariant
#       target tracks the metric as X moves -- but T_E is NOT
#       differentiated through the chain rule.  This is the "frozen
#       per iteration" approximation: each Newton step minimises the
#       polynomial inner problem
#           J_inner(X)  =  sum_IP w_IP * 1/2 sum_k w_k r_k(X)^2
#           r_k         =  S^{ij}(X) - S^{ij}_target(X_iter_k)
#       which has polynomial conditioning.  The outer fixed-point
#       updates X_iter and the covariant target together; at the
#       fixed point the physical stress matches the physical target
#       exactly (since pushing S^{ij}* through T_E^{-T} reproduces
#       sigma_phys_target_voigt by construction).
#
#   (d) detJ MULTIPLIER NOT USED.  The VaReS shape-sensitivity terms
#       (psi * d detJ/dX, psi * d2 detJ/dX) are sign-indefinite and
#       made the Hessian indefinite under the high-strain initial
#       conditions of the semi-cylinder problem.  We integrate over
#       the PARENT (parametric) domain instead -- mathematically
#       equivalent up to a per-element area weighting, but without
#       the sign-indefinite cross terms in the Hessian.
#
# Net result: every chain-rule entry is a polynomial of polynomials.
# No 1/sqrt(G_11) anywhere in the derivatives.  T_E enters only as a
# per-iteration scalar on the TARGET side, not inside the chain rule.
# Hessian conditioning is bounded by the Chen polynomial curvature
# (~1e10) -- 13 orders of magnitude tighter than the old GS path.
#
# Public interface is unchanged: the X_*_local arrays are 4x3 slices of
# the global node arrays for one element; the returned gradient is an
# 8-vector and the Hessian an 8x8 matrix in DOF-flat order
# (N0.x, N0.y, N1.x, N1.y, ...).

_DEFAULT_RESIDUAL_KEYS = ("sx", "sy", "txy", "ts")


def _covariant_metric_at(X_mat_local: np.ndarray,
                         xi: float, eta: float
                         ) -> np.ndarray:
    """Convenience: returns G_cov (2x2 covariant metric) at the IP."""
    G1, G2, _ = covariant_basis(X_mat_local, xi, eta)
    return np.array([[G1 @ G1, G1 @ G2],
                     [G2 @ G1, G2 @ G2]])


def _physical_to_covariant_targets(targets_phys: dict,
                                   weights: dict,
                                   G_cov: np.ndarray) -> dict:
    """Convert per-IP physical-stress targets to their CONTRAVARIANT
    counterparts in the curvilinear basis.

    The (sx, sy, txy) entries are mapped via the linear push-back
        S^{ij}_target_voigt = T_E(G_cov).T @ sigma_phys_target_voigt
    (see chen2022_constitutive.physical_target_to_covariant).

    The "ts" residual functional is QUADRATIC in stress, so the
    physical-to-covariant target conversion is not a simple linear
    map.  This refactor leaves "ts" pass-through unchanged -- which is
    semantically meaningful ONLY when weights["ts"] == 0.  The
    optimiser's default S_WEIGHTS dictionary (in
    optimise_cutting_pattern_semicylinder.py) ships with ts=0, so
    this is fine for current use; raise a warning if the user opts in.
    """
    sigma_phys_voigt = np.array([targets_phys.get("sx",  0.0),
                                  targets_phys.get("sy",  0.0),
                                  targets_phys.get("txy", 0.0)])
    S_con_voigt = physical_target_to_covariant(sigma_phys_voigt, G_cov)
    out = {
        "sx":  float(S_con_voigt[0]),
        "sy":  float(S_con_voigt[1]),
        "txy": float(S_con_voigt[2]),
        "ts":  float(targets_phys.get("ts", 0.0)),
    }
    if weights.get("ts", 0.0) != 0.0:
        # The ts residual is quadratic in stress; a strictly correct
        # covariant translation involves a rank-3 polynomial in S^{ij}
        # not implemented here.  Currently nothing is done -- the
        # numerical value is passed straight through.  Optimisations
        # that require ts != 0 should add the proper conversion.
        out["_ts_warning"] = True
    return out


def elem_obj_grad_hess(X_mat_local: np.ndarray,
                       X_sp_local: np.ndarray,
                       targets: dict, weights: dict,
                       *, linear_material: bool = False,
                       frozen_target_metric_X: np.ndarray | None = None
                       ) -> tuple[float, np.ndarray, np.ndarray]:
    """Per-element (J, g, H) of the stress-deviation objective.

    Covariant path (see section 15 header).

        J_e  = sum_IP w_IP * 1/2 sum_k w_k r_k(X)^2
        r_k  = scalar functional of S^{ij}(X) - frozen covariant target

    where chen_stress_covariant evaluates the Chen polynomial DIRECTLY
    on the covariant Green-Lagrange components and returns the
    standard tangent D and curvature M (no T_E inside).

    Target frozen reference
    -----------------------
    The user's PHYSICAL stress target sigma_phys is converted to its
    covariant counterpart via T_E(G_cov(X_ref)).T @ sigma_phys_voigt,
    where X_ref = frozen_target_metric_X if provided, else X_mat_local.
    For the optimiser to MINIMISE A FIXED OBJECTIVE the frozen
    reference should be set ONCE (at the outer step start) and reused
    across all inner-Newton gradient/Hessian/line-search calls --
    otherwise we are chasing a moving target and divergence is
    possible.  See optimise_cutting_pattern_semicylinder.py for the
    intended usage.

    Chain rule downstream is purely polynomial:
        dS^{ij} / dX   =  D @ dE_cov_voigt / dX
        d^2 S^{ij}/dX^2 contributes via M @ (dE_cov/dX)^2
                        + D @ d^2 E_cov / dX^2

    linear_material=True is the LINEARISED HOMOTOPY PHASE:
      - swap in chen_stress_linear / chen_stress_hessian_linear via
        chen_stress_covariant (D constant, M = 0),
      - exclude the ts = txy * (sx + sy) residual entirely (it is a
        stress product, scales as eps^2 / eps^4, and is the dominant
        source of large J + indefinite Hessian even in the linear law),
      - in the Newton-correction block keep ONLY the kinematic
        curvature term_III = d2E_cov_voigt @ (dr/dS . D); term_I is
        zero (no ts -> no stress-space curvature) and term_II is zero
        (M = 0 in the linear law).
    """
    metric_X = (frozen_target_metric_X
                if frozen_target_metric_X is not None else X_mat_local)

    J_e = 0.0
    g_e = np.zeros(N_DOFS)
    H_e = np.zeros((N_DOFS, N_DOFS))

    for (xi, eta), w_ip in zip(GAUSS_PTS_5, GAUSS_WTS_5):
        # ---- COVARIANT kinematics (polynomial in X) ----------------
        E_cov     = strain_at(X_mat_local, X_sp_local, xi, eta)
        dE_v_cov  = dE_voigt_per_dof(X_mat_local, xi, eta)        # (8, 3)
        d2E_v_cov = d2E_voigt_per_pair(xi, eta)                   # (8, 8, 3)

        # ---- Frozen-reference target conversion (T_E enters HERE) --
        G_cov_t     = _covariant_metric_at(metric_X, xi, eta)
        targets_con = _physical_to_covariant_targets(
            targets, weights, G_cov_t)

        # ---- Chen polynomial on covariant strain -------------------
        S_kl, D_voigt, M_voigt, _ = chen_stress_covariant(
            E_cov, linear_material=linear_material)

        dS_v = dE_v_cov @ D_voigt.T                               # (8, 3)

        for name, r_k, dr_dS, d2r_dS_dS, w_k in stress_residuals(
                S_kl, targets_con, weights):
            if linear_material and name == "ts":
                continue

            J_e  += 0.5 * w_ip * w_k * r_k * r_k
            dr_dX = dS_v @ dr_dS                                  # (8,)

            # Gauss-Newton outer product (always PSD).
            g_e += w_ip * w_k * r_k * dr_dX
            H_e += w_ip * w_k * np.outer(dr_dX, dr_dX)

            # Newton correction: r_k * d^2 r_k / dX dX.
            v_j      = dr_dS @ D_voigt
            term_III = d2E_v_cov @ v_j
            if linear_material:
                H_e += w_ip * w_k * r_k * term_III
            else:
                term_I  = dS_v @ d2r_dS_dS @ dS_v.T
                C_jk    = np.einsum('i,ijk->jk', dr_dS, M_voigt)
                term_II = dE_v_cov @ C_jk @ dE_v_cov.T
                H_e += w_ip * w_k * r_k * (term_I + term_II + term_III)

    return J_e, g_e, 0.5 * (H_e + H_e.T)


def elem_obj_only(X_mat_local: np.ndarray,
                  X_sp_local: np.ndarray,
                  targets: dict, weights: dict,
                  *, linear_material: bool = False,
                  frozen_target_metric_X: np.ndarray | None = None
                  ) -> float:
    """Cheap J-only variant, used by the line-search inner loop.

    Mirrors elem_obj_grad_hess's per-IP path -- including the per-IP
    physical->covariant TARGET conversion -- so J reported here and
    g.dX in the Armijo test refer to the SAME inner objective.

    Parameters
    ----------
    frozen_target_metric_X : (4, 3) ndarray or None
        If provided, the covariant TARGET conversion uses
        G_cov(frozen_target_metric_X) instead of G_cov(X_mat_local).
        This matches the "frozen-per-iteration target" approximation
        the gradient/Hessian computation uses, so the Armijo Newton
        line search compares like-for-like.  Pass the OUTER Newton
        iterate here and use the trial point X_mat_local + alpha*dX
        for the strain evaluation.  None falls back to using X_mat_local
        for both (consistent at fixed points, off-step otherwise).

    linear_material must MATCH the flag used by the grad/Hess call
    feeding the line search -- otherwise the J reported here and the
    g.dX used in the Armijo test belong to different objectives.
    """
    metric_X = (frozen_target_metric_X
                if frozen_target_metric_X is not None else X_mat_local)

    J_e = 0.0
    for (xi, eta), w_ip in zip(GAUSS_PTS_5, GAUSS_WTS_5):
        E_cov       = strain_at(X_mat_local, X_sp_local, xi, eta)
        G_cov_t     = _covariant_metric_at(metric_X, xi, eta)
        targets_con = _physical_to_covariant_targets(
            targets, weights, G_cov_t)
        S_kl, _, _, _ = chen_stress_covariant(
            E_cov, linear_material=linear_material)

        for name, r_k, _, _, w_k in stress_residuals(
                S_kl, targets_con, weights):
            if linear_material and name == "ts":
                continue
            J_e += 0.5 * w_ip * w_k * r_k * r_k
    return J_e


def elem_obj_grad(X_mat_local: np.ndarray,
                  X_sp_local: np.ndarray,
                  targets: dict, weights: dict,
                  *, linear_material: bool = False,
                  frozen_target_metric_X: np.ndarray | None = None
                  ) -> tuple[float, np.ndarray]:
    """Per-element (J, g) without the Hessian assembly.

    Same covariant chain rule and frozen-reference target conversion
    as elem_obj_grad_hess but skipping the curvature terms M_voigt /
    d2E_voigt_per_pair.  Roughly 5-10x faster per iter for first-order
    optimisers (steepest descent, L-BFGS, conjugate gradient).

    See elem_obj_grad_hess for the linear_material and
    frozen_target_metric_X kwargs.
    """
    metric_X = (frozen_target_metric_X
                if frozen_target_metric_X is not None else X_mat_local)

    J_e = 0.0
    g_e = np.zeros(N_DOFS)
    for (xi, eta), w_ip in zip(GAUSS_PTS_5, GAUSS_WTS_5):
        E_cov       = strain_at(X_mat_local, X_sp_local, xi, eta)
        dE_v_cov    = dE_voigt_per_dof(X_mat_local, xi, eta)
        G_cov_t     = _covariant_metric_at(metric_X, xi, eta)
        targets_con = _physical_to_covariant_targets(
            targets, weights, G_cov_t)

        S_kl, D_voigt, _, _ = chen_stress_covariant(
            E_cov, linear_material=linear_material)
        dS_v = dE_v_cov @ D_voigt.T                               # (8, 3)

        for name, r_k, dr_dS, _, w_k in stress_residuals(
                S_kl, targets_con, weights):
            if linear_material and name == "ts":
                continue
            J_e += 0.5 * w_ip * w_k * r_k * r_k
            dr_dX = dS_v @ dr_dS                                  # (8,)
            g_e += w_ip * w_k * r_k * dr_dX
    return J_e, g_e


# ============================================================
# 16. Global assembly over a quad mesh
# ============================================================
#
# Element node ids are 0-based; DOF id = 2 * node + (0 for x, 1 for y).
# The z column of X_mat is preserved structurally because every step
# is applied through apply_step_mesh, which only writes to the (x, y)
# entries. Dense Hessian; the assembled matrix is at most 2N_nodes
# square -- for the 7x11 = 77-node validation mesh that is 154 x 154
# (~200 KB), so dense linear algebra is the simple choice.

def _elem_dof_ids(elem_node_ids: np.ndarray) -> np.ndarray:
    """Map 4 element-local node ids (0-based) to 8 global DOF ids
    in the layout (N0.x, N0.y, N1.x, N1.y, N2.x, N2.y, N3.x, N3.y)."""
    ids = np.empty(8, dtype=int)
    for i, n in enumerate(elem_node_ids):
        ids[2 * i    ] = 2 * int(n)
        ids[2 * i + 1] = 2 * int(n) + 1
    return ids


def assemble_mesh_obj(X_mat: np.ndarray, X_sp: np.ndarray,
                      elements: np.ndarray,
                      targets: dict, weights: dict,
                      *, linear_material: bool = False,
                      frozen_X_for_targets: np.ndarray | None = None
                      ) -> float:
    """Sum of per-element J across the whole mesh (J-only).

    frozen_X_for_targets : (N_nodes, 3) ndarray or None
        Optional fixed mesh used to compute the per-IP covariant target
        conversion -- pass the OUTER Newton iterate here so the Armijo
        line search uses the SAME (frozen) covariant target the
        gradient was built from.  None falls back to using X_mat for
        both target metric and strain.
    """
    total = 0.0
    for elem in elements:
        kw = {"linear_material": linear_material}
        if frozen_X_for_targets is not None:
            kw["frozen_target_metric_X"] = frozen_X_for_targets[elem]
        total += elem_obj_only(X_mat[elem], X_sp[elem], targets, weights, **kw)
    return total


def assemble_mesh_obj_grad(X_mat: np.ndarray, X_sp: np.ndarray,
                           elements: np.ndarray,
                           targets: dict, weights: dict,
                           *, linear_material: bool = False,
                           frozen_X_for_targets: np.ndarray | None = None
                           ) -> tuple[float, np.ndarray]:
    """Assemble global (J, g) over the mesh -- no Hessian.

    Companion to assemble_mesh_grad_hess for first-order optimisers.

    frozen_X_for_targets : (N_nodes, 3) ndarray or None
        Fixed mesh for the per-IP covariant target conversion.  See
        elem_obj_grad_hess docstring -- the optimiser should pass the
        SAME frozen X to grad / Hess / obj calls so the inner Newton
        minimises a single well-defined objective.  None falls back
        to X_mat (current state).
    """
    N_nodes = X_mat.shape[0]
    J = 0.0
    g = np.zeros(2 * N_nodes)
    for elem in elements:
        kw = {"linear_material": linear_material}
        if frozen_X_for_targets is not None:
            kw["frozen_target_metric_X"] = frozen_X_for_targets[elem]
        J_e, g_e = elem_obj_grad(X_mat[elem], X_sp[elem], targets, weights, **kw)
        dof_ids = _elem_dof_ids(elem)
        J += J_e
        g[dof_ids] += g_e
    return J, g


def assemble_mesh_grad_hess(X_mat: np.ndarray, X_sp: np.ndarray,
                            elements: np.ndarray,
                            targets: dict, weights: dict,
                            *, linear_material: bool = False,
                            frozen_X_for_targets: np.ndarray | None = None
                            ) -> tuple[float, np.ndarray, np.ndarray]:
    """Assemble global (J, g, H) over a quad mesh.

    Parameters
    ----------
    X_mat : (N_nodes, 3) ndarray
        Material configuration (planar, z = 0).
    X_sp : (N_nodes, 3) ndarray
        Spatial configuration (fixed 3D target).
    elements : (N_elem, 4) int array
        0-based node ids per quad (CCW winding from outside).
    targets, weights : dict
        Keys 'sx', 'sy', 'txy', 'ts'.
    frozen_X_for_targets : (N_nodes, 3) ndarray or None
        Fixed mesh for the per-IP covariant target conversion (see
        elem_obj_grad_hess docstring).  None -> use X_mat.

    Returns
    -------
    J : float
    g : (2 N_nodes,) ndarray, layout (x_0, y_0, x_1, y_1, ...)
    H : (2 N_nodes, 2 N_nodes) ndarray
    """
    N_nodes = X_mat.shape[0]
    N_g     = 2 * N_nodes
    J = 0.0
    g = np.zeros(N_g)
    H = np.zeros((N_g, N_g))
    for elem in elements:
        kw = {"linear_material": linear_material}
        if frozen_X_for_targets is not None:
            kw["frozen_target_metric_X"] = frozen_X_for_targets[elem]
        J_e, g_e, H_e = elem_obj_grad_hess(
            X_mat[elem], X_sp[elem], targets, weights, **kw)
        dof_ids = _elem_dof_ids(elem)
        J += J_e
        g[dof_ids] += g_e
        H[np.ix_(dof_ids, dof_ids)] += H_e
    return J, g, H


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


# ============================================================
# 17. Post-processing: physical stress / strain at converged X
# ============================================================
#
# The optimisation path above operates on covariant strain and uses a
# per-IP CONSTANT Gram-Schmidt transform.  These helpers do the
# converse: given a (presumably converged) X_material, evaluate the
# FULL Gram-Schmidt transform AT THAT X and return the physical
# Green-Lagrange strain / 2nd PK stress in the local orthonormal frame.
# No derivatives are taken here, so the conditioning issues that
# motivated the covariant refactor are irrelevant -- this is the ONLY
# place the optimisation pipeline ever evaluates strain_at_physical.

def compute_physical_stress_postprocess(
        X_mat: np.ndarray, X_sp: np.ndarray,
        elements: np.ndarray,
        *, ip_points: list | None = None,
        linear_material: bool = False
) -> dict:
    """Physical Green-Lagrange / 2nd PK stress field at converged X.

    Parameters
    ----------
    X_mat, X_sp : (N_nodes, 3) ndarray
        Converged material and (fixed) spatial configurations.
    elements : (N_elem, 4) int ndarray
        0-based quad connectivity.
    ip_points : list of (xi, eta) tuples or None
        IPs to sample.  Defaults to the element centre only
        [(0.0, 0.0)].  Pass GAUSS_PTS_5 for the full 5-point set.
    linear_material : bool
        If True, use chen_stress_linear (for the linear-homotopy
        diagnostic plots).

    Returns
    -------
    dict with keys
        E_phys     : (N_elem, N_ip, 3) physical Green-Lagrange in
                     engineering Voigt (E11, E22, gam12)
        S_phys     : (N_elem, N_ip, 3) physical 2nd PK in Voigt
                     (S11, S22, S12)
        ip_points  : list of (xi, eta) actually sampled
        info       : per-(elem, ip) constitutive-branch labels
    """
    if ip_points is None:
        ip_points = [(0.0, 0.0)]
    n_elem = len(elements)
    n_ip   = len(ip_points)

    E_phys_out = np.zeros((n_elem, n_ip, 3))
    S_phys_out = np.zeros((n_elem, n_ip, 3))
    info_grid  = [[None] * n_ip for _ in range(n_elem)]

    stress_fn = chen_stress_linear if linear_material else chen_stress

    for k, elem in enumerate(elements):
        Xm = X_mat[elem]
        Xs = X_sp[elem]
        for j, (xi, eta) in enumerate(ip_points):
            # FULL Gram-Schmidt at this X (post-processing path).
            E_phys = strain_at_physical(Xm, Xs, xi, eta)
            S_phys, _, info = stress_fn(E_phys)
            E_phys_out[k, j] = [E_phys[0, 0], E_phys[1, 1],
                                2.0 * E_phys[0, 1]]
            S_phys_out[k, j] = [S_phys[0, 0], S_phys[1, 1],
                                S_phys[0, 1]]
            info_grid[k][j] = info

    return {
        "E_phys":    E_phys_out,
        "S_phys":    S_phys_out,
        "ip_points": list(ip_points),
        "info":      info_grid,
    }


def element_centre_stresses(X_mat: np.ndarray, X_sp: np.ndarray,
                            elements: np.ndarray) -> np.ndarray:
    """Per-element physical (sx, sy, txy) at the centre IP (xi=eta=0).

    Returns (N_elem, 3).  Thin wrapper over
    compute_physical_stress_postprocess for plot face-coloring.
    """
    out = compute_physical_stress_postprocess(X_mat, X_sp, elements)
    return out["S_phys"][:, 0, :]


def element_centre_residual_magnitude(X_mat: np.ndarray, X_sp: np.ndarray,
                                      elements: np.ndarray,
                                      targets: dict,
                                      weights: dict) -> np.ndarray:
    """sqrt( sum_k w_k r_k^2 ) at the centre IP of each element.

    Per-element 'how far am I from the target stress' scalar -- a
    direct visual readout of the optimiser's objective density across
    the mesh.  Uses the physical-stress post-processing path so the
    residual matches the user-facing stress target in physical units.
    """
    S_centre = element_centre_stresses(X_mat, X_sp, elements)
    out = np.zeros(len(elements))
    for k in range(len(elements)):
        S_kl = np.array([[S_centre[k, 0], S_centre[k, 2]],
                         [S_centre[k, 2], S_centre[k, 1]]])
        s = 0.0
        for _, r_k, _, _, w_k in stress_residuals(S_kl, targets, weights):
            s += w_k * r_k * r_k
        out[k] = np.sqrt(s)
    return out