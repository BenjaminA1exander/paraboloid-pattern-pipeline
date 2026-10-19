"""
vares_kinematics.py

Curvilinear membrane kinematics for the VaReS (Variation of Reference
Strategy, Widhammer 2015, Ch. 5.2) cutting-pattern optimiser.  The
element shape is supplied as an ElementType (element_types) -- 4-node
bilinear quad or 3-node linear triangle -- so every quantity below is
built for the element passed to bundle_at.

This module provides the *exact* (un-frozen, square-root-free in the
strain/stress chain) tensor derivatives of the covariant Green-Lagrange
strain that the VaReS residual-force equation needs:

    R_a = sum_IP  detJ  (S_chi - S_pre)^{ij}  (dE/dX_a)_{ij}                  (5.42)

with the contravariant 2nd PK stress S^{ij} contracted against the FULL
tensor derivative of strain, eq. (5.34):

    (dE/dX_a)_{ij} = E_{ij,a} - E_{rj} Gamma^r_{ia} - E_{is} Gamma^s_{ja}

and its linearisation, eq. (5.55), for the consistent tangent.

Index convention
----------------
- In-plane indices i, j, k, l, r, s, m, n run over {0, 1} (the two
  parametric directions).  The out-of-plane (normal) contributions
  vanish identically here because the material configuration is planar
  (z = 0) and DOFs are in-plane, so dG_k/dX_a is perpendicular to the
  normal G^3 and every Christoffel symbol Gamma^3 = 0 (Widhammer
  Remark I).
- DOF index a in 0..n_dofs-1: node = a // 2, component = a % 2
  (0 = x, 1 = y).  n_dofs = 8 for the quad, 6 for the triangle.
- All "tensor" quantities are returned as 2x2 arrays of covariant
  components in the (G^k (x) G^l) dual basis, matching the covariant
  Green-Lagrange components E_{kl}.

Everything is built from the atomic derivatives already validated in
bilinear_quad_lib (dGi_dXa, dEij_dXa, d2Eij_dXa_dXb, dGup_i_dXa).
"""

from __future__ import annotations

import numpy as np

from single_bilinear_quad_covariant import contravariant_basis
from element_types import ElementType, QUAD4
from element_kinematics import (
    covariant_basis, dGi_dXa, dEij_dXa, d2Eij_dXa_dXb, dGup_i_dXa,
    strain_at, dcross_dXa,
)


# ============================================================
# Per-IP precomputed kinematic bundle
# ============================================================
#
# Assembling the residual/tangent at one integration point reuses the
# same handful of metric quantities many times.  bundle_at packs them
# once so the element loop in vares_membrane stays readable.

class IPKinematics:
    """All metric / strain derivatives at one (xi, eta) for one element.

    Attributes
    ----------
    G1, G2       : covariant base vectors (3-vectors)
    Gup          : list [G^1, G^2] contravariant (dual) base vectors
    E            : (2, 2) covariant Green-Lagrange strain E_{kl}
    dGk          : (8, 2, 3) dG_k/dX_a -- index [a][k] is a 3-vector
    dGup         : (8, 2, 3) dG^k/dX_a -- index [a][k] is a 3-vector
    E_part       : (8, 2, 2) covariant partial E_{kl,a}  (eq 5.35)
    E_part2      : (8, 8, 2, 2) covariant partial E_{kl,a,b}  (eq 5.56)
    Gamma        : (8, 2, 2) Gamma[a][r][k] = Gamma^r_{ka} = dG_k/dX_a . G^r
    dGamma       : (8, 8, 2, 2) dGamma[a][b][r][k] = Gamma^r_{ka,b}
                   = dG_k/dX_a . dG^r/dX_b   (eq 5.58a)
    dE_tens      : (8, 2, 2) full tensor (dE/dX_a)_{kl}      (eq 5.34)
    detJ         : scalar 0.5 ||G1 x G2|| (current config)
    ddetJ        : (8,) d detJ / dX_a                        (eq 5.40)
    """

    __slots__ = ("G1", "G2", "Gup", "E", "dGk", "dGup", "E_part",
                 "E_part2", "Gamma", "dGamma", "dE_tens", "detJ", "ddetJ")


def bundle_at(X_mat: np.ndarray, X_sp: np.ndarray,
              xi: float, eta: float,
              *, etype: ElementType = QUAD4,
              want_second: bool = True) -> IPKinematics:
    """Precompute every kinematic quantity needed at one IP.

    etype selects the element shape (QUAD4 / TRI3); every DOF loop and
    array dimension is driven by etype.n_dofs, and the shape-function
    derivatives come from etype.dN, so the same code serves quads and
    triangles.

    want_second=False skips the (n_dofs, n_dofs, ...) second-derivative
    arrays (E_part2, dGamma) -- used by the gradient-only / residual-only
    path.
    """
    n_dofs = etype.n_dofs
    k = IPKinematics()

    G1, G2, _ = covariant_basis(X_mat, xi, eta, etype)
    Gup1, Gup2, _, _ = contravariant_basis(G1, G2)
    k.G1, k.G2 = G1, G2
    k.Gup = [Gup1, Gup2]

    k.E = strain_at(X_mat, X_sp, xi, eta, etype)     # (2, 2)

    # dG_k/dX_a  (independent of X) and dG^k/dX_a (depends on X)
    dGk  = np.zeros((n_dofs, 2, 3))
    dGup = np.zeros((n_dofs, 2, 3))
    E_part = np.zeros((n_dofs, 2, 2))
    for a in range(n_dofs):
        dG1_a, dG2_a = dGi_dXa(a, xi, eta, etype)
        dGk[a, 0], dGk[a, 1] = dG1_a, dG2_a
        dGup1_a, dGup2_a = dGup_i_dXa(X_mat, a, xi, eta, etype)
        dGup[a, 0], dGup[a, 1] = dGup1_a, dGup2_a
        E_part[a] = dEij_dXa(X_mat, a, xi, eta, etype)
    k.dGk, k.dGup, k.E_part = dGk, dGup, E_part

    # Christoffel symbols Gamma^r_{ka} = dG_k/dX_a . G^r
    Gamma = np.zeros((n_dofs, 2, 2))                 # [a][r][k]
    for a in range(n_dofs):
        for r in range(2):
            for kk in range(2):
                Gamma[a, r, kk] = dGk[a, kk] @ k.Gup[r]
    k.Gamma = Gamma

    # Full tensor first derivative (eq 5.34)
    E = k.E
    dE_tens = np.zeros((n_dofs, 2, 2))
    for a in range(n_dofs):
        for i in range(2):
            for j in range(2):
                val = E_part[a, i, j]
                for r in range(2):
                    val -= E[r, j] * Gamma[a, r, i]
                for s in range(2):
                    val -= E[i, s] * Gamma[a, s, j]
                dE_tens[a, i, j] = val
    k.dE_tens = dE_tens

    # detJ and its first derivative (eq 5.40); reference-area default
    # never differentiates this, but the vary_detJ path uses it.
    cross = np.cross(G1, G2)
    norm_c = float(np.linalg.norm(cross))
    k.detJ = 0.5 * norm_c
    ddetJ = np.zeros(n_dofs)
    for a in range(n_dofs):
        c_a = dcross_dXa(X_mat, a, xi, eta, etype)
        ddetJ[a] = 0.5 * (cross @ c_a) / norm_c
    k.ddetJ = ddetJ

    if want_second:
        E_part2 = np.zeros((n_dofs, n_dofs, 2, 2))
        for a in range(n_dofs):
            for b in range(n_dofs):
                E_part2[a, b] = d2Eij_dXa_dXb(a, b, xi, eta, etype)
        k.E_part2 = E_part2

        # Gamma^r_{ka,b} = dG_k/dX_a . dG^r/dX_b  (eq 5.58a)
        dGamma = np.zeros((n_dofs, n_dofs, 2, 2))    # [a][b][r][k]
        for a in range(n_dofs):
            for b in range(n_dofs):
                for r in range(2):
                    for kk in range(2):
                        dGamma[a, b, r, kk] = dGk[a, kk] @ dGup[b, r]
        k.dGamma = dGamma
    else:
        k.E_part2 = None
        k.dGamma = None

    return k


def d2E_tensor_pair(k: IPKinematics, a: int, b: int) -> np.ndarray:
    """Ordinary partial derivative  d/dX_b [ (dE/dX_a)_{ij} ]  of the
    eq-5.34 strain-derivative *components*.

    This is what the consistent tangent of the VaReS residual needs:
    the residual is the plain product  R_a = sum_IP W S^{ij} (dE/dX_a)_{ij}
    of two component-valued functions of X, so dR_a/dX_b uses the
    ORDINARY derivative of (dE/dX_a)_{ij}, not the tensor second
    derivative (Widhammer eq 5.55).  The two differ by the basis-
    variation terms  -(dE/dX_a)_{mj} Gamma^m_{ib} - (dE/dX_a)_{in}
    Gamma^n_{jb}, which belong to the tensor d^2E but NOT to the
    ordinary component derivative; including them double-counts the
    basis change already carried by the contravariant stress S^{ij}.

    Concretely (differentiating eq 5.34, using d^2 G_i/dX^2 = 0 so
    Gamma^r_{ia,b} = dG_i/dX_a . dG^r/dX_b, eq 5.58a):

        d/dX_b (dE/dX_a)_{ij}
            = E_{ij,a,b}
              - E_{rj,b} Gamma^r_{ia} - E_{rj} Gamma^r_{ia,b}
              - E_{is,b} Gamma^s_{ja} - E_{is} Gamma^s_{ja,b}

    Requires a bundle built with want_second=True.
    """
    E = k.E
    E_part2 = k.E_part2
    E_part = k.E_part
    Gamma = k.Gamma
    dGamma = k.dGamma

    out = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            val = E_part2[a, b, i, j]
            # -(E_{rj,b} Gamma^r_{ia} + E_{rj} Gamma^r_{ia,b})
            for r in range(2):
                val -= (E_part[b, r, j] * Gamma[a, r, i]
                        + E[r, j] * dGamma[a, b, r, i])
            # -(E_{is,b} Gamma^s_{ja} + E_{is} Gamma^s_{ja,b})
            for s in range(2):
                val -= (E_part[b, i, s] * Gamma[a, s, j]
                        + E[i, s] * dGamma[a, b, s, j])
            out[i, j] = val
    return out


# ============================================================
# Fibre (warp / weft) contravariant components and derivatives
# ============================================================
#
# Warp / weft are FIXED unit vectors a, b in the global Cartesian frame
# (the flat-pattern plane).  Their CONTRAVARIANT components in the
# current material basis are a^i = a . G^i; these vary with X through
# the dual basis and are differentiated EXACTLY (no Gram-Schmidt, no
# freezing).  All rational -- no square roots.

def fibre_contra(a_vec: np.ndarray, k: IPKinematics) -> np.ndarray:
    """Contravariant components a^i = a . G^i  -> (2,) array."""
    return np.array([a_vec @ k.Gup[0], a_vec @ k.Gup[1]])


def dfibre_contra(a_vec: np.ndarray, k: IPKinematics) -> np.ndarray:
    """d a^i / dX_c = a . dG^i/dX_c  ->  (n_dofs, 2) array [c][i].

    The DOF count is read from the bundle (k.dGup) so this works for
    both quads (8) and triangles (6).
    """
    n_dofs = k.dGup.shape[0]
    out = np.zeros((n_dofs, 2))
    for c in range(n_dofs):
        out[c, 0] = a_vec @ k.dGup[c, 0]
        out[c, 1] = a_vec @ k.dGup[c, 1]
    return out
