"""
element_kinematics.py

Element-type-aware membrane kinematics for the VaReS cutting-pattern
optimiser.  These are the *generic* counterparts of the atomic
first/second derivatives in bilinear_quad_lib (sections 1-7): each
function takes an ElementType (from element_types) and reads its
shape-function derivatives dN(xi, eta) and DOF count, so the very same
formulas serve the 4-node bilinear quad and the 3-node linear triangle.

Why a separate module (rather than editing bilinear_quad_lib)
-------------------------------------------------------------
bilinear_quad_lib's functions hard-code the bilinear dN and N_DOFS = 8
and are consumed (and finite-difference-validated) by the older
single-element / Gauss-Newton path.  Leaving them untouched keeps that
path bit-for-bit unchanged.  The VaReS residual path (vares_kinematics,
vares_membrane) is rebuilt on top of THIS module instead.  For QUAD4
the formulas here are identical to bilinear_quad_lib's -- a regression
test (test_element_kinematics_quad_equivalence) asserts that
numerically so the two cannot silently drift apart.

Conventions match bilinear_quad_lib:
  * material restricted to z = 0; DOF a -> node a // 2, component a % 2
    (0 = x, 1 = y);
  * Green-Lagrange E_kl are covariant components in the (G^k (x) G^l)
    dual basis;
  * the spatial configuration X_spatial is held fixed.
"""

from __future__ import annotations

import numpy as np

from element_types import ElementType, contravariant_basis, covariant_metric


# ============================================================
# Covariant base vectors (any element type)
# ============================================================

def covariant_basis(X_nodes: np.ndarray, xi: float, eta: float,
                    etype: ElementType):
    """Return (G_1, G_2, G_3) at parametric point (xi, eta).

    G_1, G_2 are tangent to the surface; G_3 is the unit normal.  Raises
    ValueError on a degenerate element (||G_1 x G_2|| ~ 0), mirroring the
    quad tutorial's guard.
    """
    X_nodes = np.asarray(X_nodes, dtype=float)
    if X_nodes.shape != (etype.n_nodes, 3):
        raise ValueError(
            f"X_nodes must have shape ({etype.n_nodes}, 3) for element "
            f"type '{etype.name}'; got {X_nodes.shape}.")
    grad = etype.dN(xi, eta)                        # (n_nodes, 2)
    G_1 = grad[:, 0] @ X_nodes                      # (3,)
    G_2 = grad[:, 1] @ X_nodes                      # (3,)
    cross = np.cross(G_1, G_2)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1.0e-12:
        raise ValueError(
            f"Degenerate {etype.name} element at (xi, eta) = ({xi}, {eta}): "
            f"||G_1 x G_2|| = {cross_norm:.3e}.  Likely a collapsed or "
            f"collinear element.\n{X_nodes}")
    G_3 = cross / cross_norm
    return G_1, G_2, G_3


# ============================================================
# 1. Atomic first derivative  dG_i/dX_a
# ============================================================
#
# G_i is LINEAR in nodal coords, so d^2 G_i / dX_a dX_b = 0.

def dGi_dXa(a: int, xi: float, eta: float, etype: ElementType):
    """Return (dG_1/dX_a, dG_2/dX_a) as 3-vectors."""
    node, comp = a // 2, a % 2
    dN_mat = etype.dN(xi, eta)
    e = np.zeros(3)
    e[comp] = 1.0
    return dN_mat[node, 0] * e, dN_mat[node, 1] * e


# ============================================================
# 2. Metric tensor:  dG_ij/dX_a  and  d^2 G_ij/dX_a dX_b
# ============================================================

def dGij_dXa(X_nodes: np.ndarray, a: int,
             xi: float, eta: float, etype: ElementType) -> np.ndarray:
    G1, G2, _ = covariant_basis(X_nodes, xi, eta, etype)
    dG1_a, dG2_a = dGi_dXa(a, xi, eta, etype)
    M = np.zeros((2, 2))
    M[0, 0] = 2.0 * (dG1_a @ G1)
    M[1, 1] = 2.0 * (dG2_a @ G2)
    M[0, 1] = dG1_a @ G2 + G1 @ dG2_a
    M[1, 0] = M[0, 1]
    return M


def d2Gij_dXa_dXb(a: int, b: int, xi: float, eta: float,
                  etype: ElementType) -> np.ndarray:
    """Independent of X_nodes since d^2 G_i / dX^2 = 0."""
    dG1_a, dG2_a = dGi_dXa(a, xi, eta, etype)
    dG1_b, dG2_b = dGi_dXa(b, xi, eta, etype)
    M = np.zeros((2, 2))
    M[0, 0] = 2.0 * (dG1_a @ dG1_b)
    M[1, 1] = 2.0 * (dG2_a @ dG2_b)
    M[0, 1] = dG1_a @ dG2_b + dG1_b @ dG2_a
    M[1, 0] = M[0, 1]
    return M


# ============================================================
# 3. Cross product:  d(G_1 x G_2)/dX_a
# ============================================================

def dcross_dXa(X_nodes: np.ndarray, a: int,
               xi: float, eta: float, etype: ElementType) -> np.ndarray:
    G1, G2, _ = covariant_basis(X_nodes, xi, eta, etype)
    dG1_a, dG2_a = dGi_dXa(a, xi, eta, etype)
    return np.cross(dG1_a, G2) + np.cross(G1, dG2_a)


# ============================================================
# 5. Strain tensor derivatives  (X_spatial is FIXED)
# ============================================================

def dEij_dXa(X_material: np.ndarray, a: int,
             xi: float, eta: float, etype: ElementType) -> np.ndarray:
    return -0.5 * dGij_dXa(X_material, a, xi, eta, etype)


def d2Eij_dXa_dXb(a: int, b: int, xi: float, eta: float,
                  etype: ElementType) -> np.ndarray:
    return -0.5 * d2Gij_dXa_dXb(a, b, xi, eta, etype)


# ============================================================
# 6. Contravariant metric / dual basis derivatives
# ============================================================

def dGup_ij_dXa(X_nodes: np.ndarray, a: int,
                xi: float, eta: float, etype: ElementType) -> np.ndarray:
    G1, G2, _ = covariant_basis(X_nodes, xi, eta, etype)
    _, _, _, Gup_ij = contravariant_basis(G1, G2)
    dGij = dGij_dXa(X_nodes, a, xi, eta, etype)
    return -Gup_ij @ dGij @ Gup_ij


def dGup_i_dXa(X_nodes: np.ndarray, a: int,
               xi: float, eta: float, etype: ElementType):
    """Return (dG^1/dX_a, dG^2/dX_a) as 3-vectors."""
    G1, G2, _ = covariant_basis(X_nodes, xi, eta, etype)
    M_lower = np.stack([G1, G2])
    Gup_ij = np.linalg.inv(M_lower @ M_lower.T)
    dG1_a, dG2_a = dGi_dXa(a, xi, eta, etype)
    dM_lower = np.stack([dG1_a, dG2_a])
    dGup_ij = dGup_ij_dXa(X_nodes, a, xi, eta, etype)
    M_up_deriv = dGup_ij @ M_lower + Gup_ij @ dM_lower
    return M_up_deriv[0], M_up_deriv[1]


# ============================================================
# 7. Green-Lagrange strain between two configurations
# ============================================================

def strain_at(X_material: np.ndarray, X_spatial: np.ndarray,
              xi: float, eta: float, etype: ElementType) -> np.ndarray:
    """Covariant Green-Lagrange strain E_kl at (xi, eta)."""
    G1m, G2m, _ = covariant_basis(X_material, xi, eta, etype)
    g1s, g2s, _ = covariant_basis(X_spatial,  xi, eta, etype)
    return 0.5 * (covariant_metric(g1s, g2s)
                  - covariant_metric(G1m, G2m))
