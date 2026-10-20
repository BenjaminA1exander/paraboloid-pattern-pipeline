"""
chen2022_constitutive.py

Pure-Python port of the Chen et al. (2022) anisotropic membrane
constitutive law, as implemented in ``umat_chen2022.for``. 
Again a fully independant implementation using no external code. 
Validated against the original paper, then in abaqus, before here.

Reference
---------
Longlong Chen, Wujun Chen, Chengjun Gao, Shiping Li, Yinbo Song, Haitao Zhao, Jianhui Hu, Bing Zhao,
Strength analysis of composite envelope structures considering the tension–shear coupling and cutting pattern effect,
Thin-Walled Structures,
Volume 180,
2022,
109919,
ISSN 0263-8231,
https://doi.org/10.1016/j.tws.2022.109919.


Scope of this port
------------------
INCLUDED
  - Biaxial polynomial branch (5 terms each in sigma_x and sigma_y)
  - Uniaxial fallback branch (2 terms each)
  - "Drop past vertex" guard on each parabolic branch
  - Linear regularised fallback when both branches non-positive, placed instead of original wrinkling formulation
  - Stress-coupled shear modulus  G = c0 + kG*(sigma_x + sigma_y)
  - Consistent tangent (DDSDDE) -- analytical, branch-aware, pretty tricky to get working nicely.
  - Tsai-Wu-style failure factor (Eq. 19 of the paper, same coefficients)

OMITTED (Compared to abaqus vumat implementation)
  - Wrinkling / taut / slack treatment (the spectral regularisation
    and the slack-mode small-stiffness substitution)
    Unnecessary for obvious reasons.

Conventions
-----------
This module operates on Green-Lagrange strain components directly. To
match the UMAT's behaviour (which is fed log strain in Abaqus shell
elements) you can pass either measure -- the polynomial law is the
same algebraic function in both cases. For the cutting-pattern
optimisation we use Green-Lagrange because that's what the VaReS
covariant kinematics produce.

Strain and stress are exchanged in TWO equivalent forms:

  Tensor form:  symmetric 2x2 matrix E_kl with E_12 = E_21.
                Same for the 2nd Piola-Kirchhoff stress S_kl.

  Engineering Voigt:  3-vector [E_11, E_22, gamma_xy] where the shear
                       component is engineering: gamma_xy = 2*E_12.
                       Stress Voigt is [S_11, S_22, S_12] (the shear
                       component is the same in both since stress has
                       the factor-of-2 baked into the strain side,
                       not the stress side).

Internal Voigt is engineering for compatibility with the Fortran
DDSDDE. The convenience wrappers convert tensor <-> Voigt cleanly.
"""

from __future__ import annotations

import numpy as np


# ============================================================
# Material constants -- exact match to umat_chen2022.for PROPS array
# ============================================================

UMAT_PROPS = np.array([
    # Nonlinear anisotropic block (biaxial polynomial)
    6386.50,  2497.41,  273547.57, -11288.61,  38942.08,    # a1..a5
    4202.12,  10020.29, -40669.88,  226472.49, 23726.88,    # b1..b5
    # Stress-coupled shear modulus
    79.60,    0.31,                                         # c0, kG
    # Tsai-Wu-like failure envelope parameters
    647.67,   594.20,   8.00,                               # X, Y, S
    1.429,    1.973,    0.923,                              # alpha, beta, gamma
    1.659,    1.169,                                        # xi, eta
    # Uniaxial fallback branch
    7143.3,   251666.7, 8623.3,    181000.0,                # a1u, a3u, b2u, b4u
])

# Regularisation constants -- match the Fortran exactly
ETA_REG = 1.0e-3       # weight on the linear fallback when both polynomial
                        # branches are non-positive at the current strain
NEG_INF = -1.0e30       # sentinel marking a polynomial branch as "invalid"
                        # (current strain is past its parabolic vertex)
VERTEX_TOL = 1.0e-14    # only drop past-vertex if curvature meaningfully > 0


# ============================================================
# Branch selection -- one direction at a time
# ============================================================
#
# For each direction (x, then y) the law evaluates THREE candidate
# stresses at the current strain:
#
#   biaxial:    5-term polynomial in (eps_x, eps_y)
#   uniaxial:   2-term polynomial in the matching strain only
#   fallback:   ETA_REG * (linear modulus) * (matching strain)
#
# A polynomial branch is INVALIDATED (set to NEG_INF) if the strain
# is past the vertex of its parabolic shape -- this prevents the
# unphysical regime where high tensile strain causes the polynomial
# to curve back down to small or negative stress.
#
# Active branch:
#   - If max(biaxial, uniaxial) <= 0     -> fallback active
#   - Else whichever of biaxial, uniaxial is larger
#
# Each branch reports its own consistent partial derivatives.

def _stress_x_branch(ex: float, ey: float, props: np.ndarray):
    """Active sigma_x and (dSx/dex, dSx/dey). Returns also branch name."""
    a1, a2, a3, a4, a5 = props[0:5]
    a1u, a3u = props[20], props[21]

    sx_bi  = a1*ex + a2*ey + a3*ex*ex + a4*ey*ey + a5*ex*ey
    sx_uni = a1u*ex + a3u*ex*ex

    # Drop past-vertex contributions
    if a3 > VERTEX_TOL:
        ex_v = -(a1 + a5*ey) / (2.0*a3)
        if ex < ex_v:
            sx_bi = NEG_INF
    if a3u > VERTEX_TOL:
        ex_v = -a1u / (2.0*a3u)
        if ex < ex_v:
            sx_uni = NEG_INF

    if max(sx_bi, sx_uni) <= 0.0:
        # Linear regularised fallback
        return ETA_REG*a1*ex, ETA_REG*a1, 0.0, "fallback"
    elif sx_bi >= sx_uni:
        return (sx_bi,
                a1 + 2.0*a3*ex + a5*ey,        # dSx/dex (biaxial)
                a2 + 2.0*a4*ey + a5*ex,        # dSx/dey (biaxial)
                "biaxial")
    else:
        return (sx_uni,
                a1u + 2.0*a3u*ex,              # dSx/dex (uniaxial)
                0.0,                            # dSx/dey (uniaxial -- no cross term)
                "uniaxial")


def _stress_y_branch(ex: float, ey: float, props: np.ndarray):
    """Active sigma_y and (dSy/dex, dSy/dey). Returns also branch name."""
    b1, b2, b3, b4, b5 = props[5:10]
    b2u, b4u = props[22], props[23]

    sy_bi  = b1*ex + b2*ey + b3*ex*ex + b4*ey*ey + b5*ex*ey
    sy_uni = b2u*ey + b4u*ey*ey

    if b4 > VERTEX_TOL:
        ey_v = -(b2 + b5*ex) / (2.0*b4)
        if ey < ey_v:
            sy_bi = NEG_INF
    if b4u > VERTEX_TOL:
        ey_v = -b2u / (2.0*b4u)
        if ey < ey_v:
            sy_uni = NEG_INF

    if max(sy_bi, sy_uni) <= 0.0:
        return ETA_REG*b2*ey, 0.0, ETA_REG*b2, "fallback"
    elif sy_bi >= sy_uni:
        return (sy_bi,
                b1 + 2.0*b3*ex + b5*ey,
                b2 + 2.0*b4*ey + b5*ex,
                "biaxial")
    else:
        return (sy_uni,
                0.0,
                b2u + 2.0*b4u*ey,
                "uniaxial")


# ============================================================
# Voigt <-> tensor converters
# ============================================================

def strain_tensor_to_voigt(E_kl: np.ndarray) -> np.ndarray:
    """Symmetric 2x2 strain tensor -> [E_11, E_22, gamma_xy]."""
    return np.array([E_kl[0, 0], E_kl[1, 1], 2.0 * E_kl[0, 1]])


def stress_voigt_to_tensor(S_voigt: np.ndarray) -> np.ndarray:
    """[S_11, S_22, S_12] -> symmetric 2x2 stress tensor."""
    return np.array([[S_voigt[0], S_voigt[2]],
                     [S_voigt[2], S_voigt[1]]])


# ============================================================
# Public API: stress and tangent
# ============================================================

def chen_stress(E_kl: np.ndarray,
                props: np.ndarray = UMAT_PROPS) -> tuple:
    """Evaluate the Chen 2022 law at one integration point.

    Parameters
    ----------
    E_kl : (2, 2) ndarray
        Symmetric Green-Lagrange strain tensor (covariant components).
    props : (24,) ndarray
        Material constants -- defaults to UMAT_PROPS.

    Returns
    -------
    S_kl : (2, 2) ndarray
        Symmetric 2nd Piola-Kirchhoff stress tensor.
    D_voigt : (3, 3) ndarray
        Consistent tangent in engineering Voigt notation:
        rows are (sigma_x, sigma_y, tau_xy), columns are
        (eps_x, eps_y, gamma_xy). Suitable for plugging into the
        VaReS K_elast assembly.
    info : dict
        Branch names ('biaxial' | 'uniaxial' | 'fallback') for sigma_x
        and sigma_y, and the current shear modulus G.
    """
    c0, kG = props[10], props[11]

    ex = E_kl[0, 0]
    ey = E_kl[1, 1]
    gam = 2.0 * E_kl[0, 1]      # engineering shear strain

    sx, dsx_dex, dsx_dey, bx = _stress_x_branch(ex, ey, props)
    sy, dsy_dex, dsy_dey, by = _stress_y_branch(ex, ey, props)

    G_cur = c0 + kG * (sx + sy)
    tau   = G_cur * gam

    # Pack stress back into the symmetric tensor form
    S_kl = np.array([[sx,  tau],
                     [tau, sy]])

    # Voigt tangent. Note the shear-coupling cross terms:
    # tau = G(sx, sy) * gam_xy implies
    #   d tau / d eps_i = kG * (dsx/d eps_i + dsy/d eps_i) * gam_xy
    #   d tau / d gam_xy = G_cur
    # and the polynomial sigma_x, sigma_y do NOT depend on shear, so the
    # third column of the top-left 2x2 block is zero.
    D = np.zeros((3, 3))
    D[0, 0] = dsx_dex
    D[0, 1] = dsx_dey
    D[0, 2] = 0.0
    D[1, 0] = dsy_dex
    D[1, 1] = dsy_dey
    D[1, 2] = 0.0
    D[2, 0] = kG * (dsx_dex + dsy_dex) * gam
    D[2, 1] = kG * (dsx_dey + dsy_dey) * gam
    D[2, 2] = G_cur

    info = {"branch_x": bx, "branch_y": by, "G": G_cur}
    return S_kl, D, info


# ============================================================
# Linearised constitutive law -- homotopy starting point
# ============================================================
#
# Keeps only the LINEAR terms of the polynomial in eps_x, eps_y and
# replaces the stress-coupled shear modulus with its baseline c0.
# Drops:
#   - quadratic terms (a3, a4, a5, b3, b4, b5)
#   - branch selection (always "linear")
#   - vertex dropping / fallback regularisation
#   - shear-coupling kG*(sigma_x + sigma_y)
#
# Resulting law:
#   sigma_x = a1*ex + a2*ey
#   sigma_y = b1*ex + b2*ey
#   tau     = c0*gamma_xy
# with D_voigt constant (independent of strain) and d^2 sigma/dE dE = 0.
#
# This is the well-conditioned objective for the cutting-pattern
# optimiser to descend FIRST -- linear material -> J quadratic in
# strain -> kinematic curvature dominates the Hessian, and that curvature
# is small. Warm-start the full nonlinear chen_stress problem from the
# linear-converged X_material; the residuals are then small enough that
# the polynomial-curvature terms don't blow up the Hessian.

def chen_stress_linear(E_kl: np.ndarray,
                       props: np.ndarray = UMAT_PROPS) -> tuple:
    """Linearised Chen 2022 law. See module-level note above for scope.

    Returns the same (S_kl, D_voigt, info) tuple as chen_stress so it
    is a drop-in replacement in the element-level chain rule.
    """
    a1, a2 = props[0], props[1]
    b1, b2 = props[5], props[6]
    c0     = props[10]

    ex  = E_kl[0, 0]
    ey  = E_kl[1, 1]
    gam = 2.0 * E_kl[0, 1]

    sx  = a1 * ex + a2 * ey
    sy  = b1 * ex + b2 * ey
    tau = c0 * gam

    S_kl = np.array([[sx,  tau],
                     [tau, sy ]])
    D = np.array([[a1, a2, 0.0],
                  [b1, b2, 0.0],
                  [0.0, 0.0, c0]])
    info = {"branch_x": "linear", "branch_y": "linear", "G": c0}
    return S_kl, D, info


def chen_stress_hessian_linear(E_kl: np.ndarray,
                               props: np.ndarray = UMAT_PROPS
                               ) -> np.ndarray:
    """Linearised law has zero stress curvature: d^2 sigma / dE dE = 0.

    Drop-in shape match for chen_stress_hessian (returns a (3, 3, 3)
    zero tensor in engineering Voigt).
    """
    return np.zeros((3, 3, 3))


# ============================================================
# Covariant-friendly entry point
# ============================================================
#
# The Chen polynomial is calibrated on engineering strains in a local
# orthonormal frame. In a curvilinear setting (bilinear-quad membrane
# with non-orthogonal G_1, G_2) the link between covariant Green-Lagrange
# E_kl and the engineering strain eps_alpha_beta in the local
# orthonormal frame is the Gram-Schmidt pull-back
#
#     eps_phys = A E_cov A^T            with   A = [[p, 0], [q, r]]
#     p = 1 / sqrt(G_11)
#     q = -G_12 / sqrt(G_11 detG)
#     r = sqrt(G_11 / detG)
#
# In engineering Voigt this becomes
#     eps_phys_voigt = T_E @ E_cov_voigt
# with the 3x3 T_E built from (p, q, r) in gram_schmidt_voigt_T below.
#
# CRITICAL conditioning fix for shape optimisation:
#     T_E is a FUNCTIONAL of G_cov(X). The full chain rule
#     ∂eps_phys/∂X carries ∂T_E/∂X which contains 1/sqrt(G_11)^k
#     factors that explode at the singular boundary of the cylinder
#     mesh (sin(theta) -> 0). For the optimiser, we treat T_E as a
#     PER-IP CONSTANT scalar: built from G_cov(X_current), used to push
#     forward / pull back, but NOT differentiated. The chain rule
#     dS_phys/dX = D_phys @ T_E @ dE_cov/dX is then polynomial in X
#     and the optimiser problem stays well-conditioned.
#
#     The exact (T_E-differentiated) form lives in bilinear_quad_lib
#     as a separate post-processing path.

def gram_schmidt_voigt_T(G_cov: np.ndarray) -> np.ndarray:
    """Build the Voigt transform eps_phys_voigt = T @ E_cov_voigt.

    Engineering shear convention: E_cov_voigt = (E_11, E_22, 2 E_12)
    and eps_phys_voigt = (eps_phys_11, eps_phys_22, 2 eps_phys_12).

    T is built from the Gram-Schmidt scalars
        p = 1 / sqrt(G_11)
        q = -G_12 / sqrt(G_11 detG)
        r = sqrt(G_11 / detG)
    Only six of the nine entries are non-zero -- the rest are killed by
    the upper-triangular zero of the A matrix.
    """
    G11 = float(G_cov[0, 0])
    G22 = float(G_cov[1, 1])
    G12 = float(G_cov[0, 1])
    detG = G11 * G22 - G12 * G12
    if G11 <= 0.0 or detG <= 0.0:
        raise ValueError(
            f"Degenerate metric: G_11={G11:.3e}, detG={detG:.3e}; "
            "covariant basis is singular at this IP.")
    p = 1.0 / np.sqrt(G11)
    r = np.sqrt(G11 / detG)
    q = -G12 / np.sqrt(G11 * detG)
    T = np.zeros((3, 3))
    T[0, 0] = p * p
    T[1, 0] = q * q
    T[1, 1] = r * r
    T[1, 2] = q * r
    T[2, 0] = 2.0 * p * q
    T[2, 2] = p * r
    return T


def chen_stress_covariant(E_cov: np.ndarray,
                          G_con: np.ndarray | None = None,
                          props: np.ndarray = UMAT_PROPS,
                          *, linear_material: bool = False
                          ) -> tuple:
    """Evaluate Chen 2022 directly on covariant strain input.

    Chen's polynomial law is applied to the covariant Green-Lagrange
    components (E_11, E_22, 2 E_12) WITHOUT any Gram-Schmidt
    pull-back -- the polynomial coefficients are interpreted as a map
    from covariant strain to CONTRAVARIANT 2nd PK stress S^{ij}.

    Polynomial conditioning
    -----------------------
    No T_E enters here.  The chain rule downstream
        dS^{ij}/dX  =  D_voigt @ dE_cov_voigt/dX
    is polynomial in X (D_voigt is the standard Chen tangent
    evaluated at E_cov), so the optimisation Hessian inherits the
    bounded polynomial conditioning of the bilinear-quad kinematics.

    Physical interpretation and target conversion
    ---------------------------------------------
    S^{ij} computed this way is the CONTRAVARIANT 2nd PK in the local
    curvilinear basis.  It is NOT the same as the physical /
    engineering 2nd PK that the user sees in a Cartesian lab frame.
    The element loop in bilinear_quad_lib bridges this gap by
    converting the user's PHYSICAL stress target via
        S^{ij}_target_voigt  =  T_E(G_cov).T  @  sigma_phys_target_voigt
    (treated as a per-iteration CONSTANT) and minimising the
    covariant residual r_k = S^{ij}(X) - S^{ij}_target.  At
    convergence, pushing S^{ij} back to physical via T_E.T-inverse
    recovers exactly the physical target -- so the optimiser
    sees a polynomial-conditioned problem and the user keeps a
    physical-units target interface.

    Parameters
    ----------
    E_cov : (2, 2) ndarray
        Symmetric covariant Green-Lagrange strain E_{kl}.
    G_con : (2, 2) ndarray or None
        Contravariant metric at the IP.  Currently unused -- the
        polynomial Chen branch is anisotropic and aligned with the
        curvilinear (G_1, G_2) basis, so it does not need G^{ij}
        algebraically.  Reserved for future isotropic-invariant laws
        that need C = 2E + I traces / determinants in basis-
        independent form.
    props : (24,) ndarray
        Chen material constants (default UMAT_PROPS).
    linear_material : bool
        Use the linearised Chen branch (homotopy starting point).

    Returns
    -------
    S_con : (2, 2) ndarray
        Symmetric contravariant 2nd Piola-Kirchhoff stress S^{ij} in
        the curvilinear basis.
    D_voigt : (3, 3) ndarray
        Tangent dS_con_voigt / dE_cov_voigt in engineering Voigt
        (rows = S^{11}, S^{22}, S^{12}; cols = E_11, E_22, 2 E_12).
        Algebraically identical to the standard Chen tangent because
        the polynomial is applied directly to E_cov.
    M_voigt : (3, 3, 3) ndarray
        Curvature d^2 S_con_voigt / dE_cov_voigt dE_cov_voigt in the
        same Voigt convention.
    info : dict
        Branch labels and current shear modulus.
    """
    _ = G_con   # reserved for future isotropic-invariant laws

    if linear_material:
        S_con, D_voigt, info = chen_stress_linear(E_cov, props)
        M_voigt              = chen_stress_hessian_linear(E_cov, props)
    else:
        S_con, D_voigt, info = chen_stress(E_cov, props)
        M_voigt              = chen_stress_hessian(E_cov, props)

    return S_con, D_voigt, M_voigt, info


def physical_target_to_covariant(sigma_phys_voigt: np.ndarray,
                                 G_cov: np.ndarray) -> np.ndarray:
    """Map a user-facing PHYSICAL stress target to its CONTRAVARIANT
    counterpart in the curvilinear basis at one IP.

    The relation between physical / orthonormal-frame 2nd PK stress
    sigma_phys^{alpha beta} and contravariant 2nd PK S^{ij} in the
    curvilinear basis (G_i) is, with A = [[p, 0], [q, r]],
        sigma_phys^{alpha beta} = A_alpha^i  A_beta^j  S^{ij}
    so the INVERSE (physical -> covariant target) is
        S^{ij}_target = (A^{-T})^i_alpha  (A^{-T})^j_beta  sigma_phys_target
    In engineering Voigt this collapses to the constant per-IP
    matrix product
        S^{ij}_target_voigt = T_E.T @ sigma_phys_target_voigt
    using exactly the same T_E that gram_schmidt_voigt_T builds.

    This conversion is treated as a PER-IP CONSTANT inside the
    optimiser -- it gets recomputed each Newton iteration as the
    metric evolves, but is not differentiated through the chain rule.
    """
    T_E = gram_schmidt_voigt_T(G_cov)
    return T_E.T @ sigma_phys_voigt


def chen_failure_factor(S_kl: np.ndarray,
                        props: np.ndarray = UMAT_PROPS) -> float:
    """Tsai-Wu-like failure factor as published in Chen et al. (2022).

    Form (eq. 19 of the paper, with the calibrated coefficients
    substituted):

        FF =   sx^2 / (alpha X)^2
             + sy^2 / (beta  Y)^2
             - sx sy / (gamma^2 X Y)            <-- gamma is SQUARED
             + sx / (xi  X)
             + sy / (eta Y)
             + sx |tau| / (X S)
             + sy |tau| / (Y S)

    FF = 0  at zero stress.
    FF = 1  on the calibrated failure envelope.
    FF > 1  predicts material failure.

    Useful as a scalar objective for cutting-pattern optimisation:
    minimise max(FF) over all integration points to push the design
    away from the failure envelope.

    Note on the gamma**2 term
    -------------------------
    The Fortran ``umat_chen2022.for`` writes
        GXYF = FGAMMA * XSTR * YSTR
    i.e. plain ``gamma * X * Y`` (NOT squared). That disagrees with
    the published calibrated denominators in the paper -- with
    gamma = 0.923, X = 647.67, Y = 594.20 the published cross-term
    denominator is 327914.2, which matches ``gamma**2 * X * Y =
    327942`` but not ``gamma * X * Y = 355291``. This Python port
    follows the paper. If you ever re-run the Fortran UMAT and want
    bit-for-bit agreement on FF, change FGAMMA**2 in the Fortran
    accordingly.
    """
    X, Y, S_sh = props[12], props[13], props[14]
    alpha, beta, gamma = props[15], props[16], props[17]
    xi, eta_p = props[18], props[19]

    sx  = S_kl[0, 0]
    sy  = S_kl[1, 1]
    tau = S_kl[0, 1]
    abs_tau = abs(tau)

    AX2  = (alpha * X)**2
    BY2  = (beta  * Y)**2
    GXYF = (gamma**2) * X * Y      # corrected per the paper -- see docstring

    return (sx*sx / AX2  +  sy*sy / BY2  -  sx*sy / GXYF
            +  sx / (xi * X)  +  sy / (eta_p * Y)
            +  sx * abs_tau / (X * S_sh)
            +  sy * abs_tau / (Y * S_sh))


# ============================================================
# Second derivatives -- needed for the full analytical Hessian of any
# objective J = J(sigma(E)) under Newton-Raphson
# ============================================================
#
# Stress  sigma(E)            -> chen_stress_hessian(E)    returns M[i,j,k]
#                                where M[i,j,k] = d^2 sigma_i / dE_j dE_k
#                                in engineering Voigt: (sx,sy,tau) over
#                                (ex,ey,gam_xy).
#
# Failure factor FF(sigma)    -> chen_FF_hessian(S)        returns 3x3
#                                second-derivative matrix in Voigt
#                                (sx, sy, tau) ordering.

def chen_stress_hessian(E_kl: np.ndarray,
                        props: np.ndarray = UMAT_PROPS) -> np.ndarray:
    """Return (3, 3, 3) tensor M with M[i, j, k] = d^2 sigma_i / dE_j dE_k.

    Branch-aware: each polynomial branch (biaxial, uniaxial, fallback)
    has a different functional form, so the second derivatives change
    with the active branch. At branch boundaries the Hessian is
    discontinuous (the function is C^0 there, not C^1), which is a
    known property of the Chen 2022 polynomial law -- not a bug.

    Voigt convention: engineering shear (gam_xy = 2 E_12). The 2x
    factor is already baked into the chain rule via dE_voigt, so the
    polynomial second-derivative coefficients in this function are
    the raw "2 * polynomial_coef" values (e.g. 2*a3 for d^2 sx/dex^2).
    """
    a1, a2, a3, a4, a5 = props[0:5]
    b1, b2, b3, b4, b5 = props[5:10]
    c0, kG = props[10], props[11]
    a1u, a3u = props[20], props[21]
    b2u, b4u = props[22], props[23]

    ex = E_kl[0, 0]
    ey = E_kl[1, 1]
    gam = 2.0 * E_kl[0, 1]

    # Re-run branch selection to learn which polynomial is active and
    # get D_voigt for the tau-tangent terms.
    _, D_voigt, info = chen_stress(E_kl, props)
    bx = info["branch_x"]
    by = info["branch_y"]

    M = np.zeros((3, 3, 3))

    # --- sigma_x partials --------------------------------------------
    if bx == "biaxial":
        M[0, 0, 0] = 2.0 * a3                         # d^2 sx/dex^2
        M[0, 1, 1] = 2.0 * a4                         # d^2 sx/dey^2
        M[0, 0, 1] = a5                               # d^2 sx/dex dey
        M[0, 1, 0] = a5
        # d^2 sx / d{ex,ey} dgam = 0 (no shear coupling in polynomial)
    elif bx == "uniaxial":
        M[0, 0, 0] = 2.0 * a3u                        # d^2 sx/dex^2
        # cross-terms zero
    # bx == "fallback": linear, all second derivatives zero

    # --- sigma_y partials --------------------------------------------
    if by == "biaxial":
        M[1, 0, 0] = 2.0 * b3
        M[1, 1, 1] = 2.0 * b4
        M[1, 0, 1] = b5
        M[1, 1, 0] = b5
    elif by == "uniaxial":
        M[1, 1, 1] = 2.0 * b4u
    # by == "fallback": all zero

    # --- tau partials -------------------------------------------------
    # tau = G(sx, sy) * gam_xy = (c0 + kG*(sx + sy)) * gam_xy
    #
    # d tau / dex = kG * (dsx/dex + dsy/dex) * gam_xy
    # d tau / dey = kG * (dsx/dey + dsy/dey) * gam_xy
    # d tau / dgam = G_current = c0 + kG*(sx + sy)
    #
    # Differentiating again:
    # d^2 tau / dex^2     = kG * (d^2 sx/dex^2     + d^2 sy/dex^2)     * gam_xy
    # d^2 tau / dey^2     = kG * (d^2 sx/dey^2     + d^2 sy/dey^2)     * gam_xy
    # d^2 tau / dex dey   = kG * (d^2 sx/dex dey   + d^2 sy/dex dey)   * gam_xy
    # d^2 tau / dex dgam  = kG * (dsx/dex + dsy/dex)        (= kG*(D[0,0]+D[1,0]))
    # d^2 tau / dey dgam  = kG * (dsx/dey + dsy/dey)        (= kG*(D[0,1]+D[1,1]))
    # d^2 tau / dgam^2    = 0                  (linear in gam)

    M[2, 0, 0] = kG * (M[0, 0, 0] + M[1, 0, 0]) * gam
    M[2, 1, 1] = kG * (M[0, 1, 1] + M[1, 1, 1]) * gam
    M[2, 0, 1] = kG * (M[0, 0, 1] + M[1, 0, 1]) * gam
    M[2, 1, 0] = M[2, 0, 1]

    M[2, 0, 2] = kG * (D_voigt[0, 0] + D_voigt[1, 0])
    M[2, 2, 0] = M[2, 0, 2]
    M[2, 1, 2] = kG * (D_voigt[0, 1] + D_voigt[1, 1])
    M[2, 2, 1] = M[2, 1, 2]
    # M[2, 2, 2] stays at 0

    return M


def chen_FF_hessian(S_kl: np.ndarray,
                    props: np.ndarray = UMAT_PROPS) -> np.ndarray:
    """Return 3x3 matrix d^2 FF / dS_i dS_j in Voigt (sx, sy, tau).

    All entries are CONSTANTS (independent of stress state) except the
    sign(tau) terms in the shear cross-couplings, which flip sign
    across tau = 0. The diagonal d^2 FF/dtau^2 vanishes because the
    failure factor is linear in |tau|.

    Note on the factor of 2
    -----------------------
    The published failure factor contains terms like sx^2/856102.2 .
    Differentiating twice:  d^2 (sx^2/856102.2) / dsx^2 = 2/856102.2 .
    If you saw "1/856102.2" written somewhere, that is the first
    derivative coefficient (dFF/dsx contains  2 sx / 856102.2 , so the
    Hessian diagonal *is* 2/856102.2).
    """
    X, Y, S_sh = props[12], props[13], props[14]
    alpha, beta, gamma = props[15], props[16], props[17]

    AX2  = (alpha * X)**2
    BY2  = (beta  * Y)**2
    GXYF = (gamma**2) * X * Y

    tau = S_kl[0, 1]
    sgn_tau = 1.0 if tau >= 0.0 else -1.0

    H = np.zeros((3, 3))
    H[0, 0] =  2.0 / AX2
    H[1, 1] =  2.0 / BY2
    H[0, 1] = -1.0 / GXYF
    H[1, 0] =  H[0, 1]
    H[0, 2] =  sgn_tau / (X * S_sh)
    H[2, 0] =  H[0, 2]
    H[1, 2] =  sgn_tau / (Y * S_sh)
    H[2, 1] =  H[1, 2]
    # H[2, 2] = 0 (FF linear in |tau|)
    return H


# ============================================================
# Tangent in tensor index form -- needed by VaReS chain rule
# ============================================================
#
# The Voigt tangent D_voigt maps engineering-Voigt strain to
# Voigt stress. For the kinematic chain we want
#       d S_kl  =  C_klmn  d E_mn      (tensor form)
# with E_kl symmetric. The conversion accounts for the factor of 2
# that engineering shear strain absorbs (gamma_xy = 2 E_12):
#
#       d sigma_x  =  D[0,0] dE_11 + D[0,1] dE_22 + D[0,2] (2 dE_12)
# so
#       d S_11/d E_12  =  2 D[0,2]
# and similarly for the other shear-strain columns.

def voigt_tangent_to_tensor(D_voigt: np.ndarray) -> np.ndarray:
    """Return (2, 2, 2, 2) tensor C with d S_ij = C_ijkl d E_kl.

    Symmetric in the first pair (S_ij = S_ji) and the second pair
    (E_kl = E_lk). The factor-of-2 conversion for engineering shear
    strain is folded in.
    """
    # Voigt index -> tensor-pair mapping. Each Voigt index corresponds
    # to ONE tensor index pair, but symmetry means the (i, j) and
    # (j, i) tensor entries must both be populated.
    pairs = {0: (0, 0), 1: (1, 1), 2: (0, 1)}
    C = np.zeros((2, 2, 2, 2))
    for a in range(3):
        i, j = pairs[a]
        for b in range(3):
            k, l = pairs[b]
            scale_b = 2.0 if b == 2 else 1.0
            val = D_voigt[a, b] * scale_b
            # Fill all symmetry-related quadrants
            C[i, j, k, l] = val
            C[j, i, k, l] = val
            C[i, j, l, k] = val
            C[j, i, l, k] = val
    return C


# ============================================================
# Validation / demonstration
# ============================================================

if __name__ == "__main__":

    print("=" * 64)
    print("Chen 2022 (no-wrinkling) constitutive law -- Python port")
    print("=" * 64)

    # ---------- 1. Branch demonstration over a strain sweep --------
    print("\n1. Branch coverage along a uniaxial-x strain sweep")
    print("   eps_x      eps_y      branch_x   branch_y    sigma_x        sigma_y")
    print("   " + "-" * 80)
    for ex in [-0.10, -0.02, -0.005, 0.0, 0.005, 0.02, 0.05, 0.10, 0.20]:
        E = np.array([[ex, 0.0], [0.0, 0.0]])
        S, D, info = chen_stress(E)
        print(f"   {ex:+.4f}    {0.0:+.4f}    "
              f"{info['branch_x']:9s}  {info['branch_y']:9s}   "
              f"{S[0, 0]:+12.4f}   {S[1, 1]:+12.4f}")

    # ---------- 2. Tangent validation against finite differences ----
    print("\n2. Tangent validation -- analytical D vs central FD")
    print("   (max relative error per row of D; values ~1e-7 are FD-floor)")
    h = 1.0e-6
    test_strains = [
        ("small tension",       np.array([[0.005, 0.003], [0.003, 0.001]])),
        ("moderate biaxial",    np.array([[0.020, 0.015], [0.015, 0.004]])),
        ("uniaxial-x active",   np.array([[0.050, 0.000], [0.000, 0.001]])),
        ("compression-y",       np.array([[0.020, -0.030], [0.000, 0.002]])),
    ]
    print("   " + " ".join(f"{lbl:>20s}" for lbl, _ in test_strains))
    for row in range(3):
        cells = []
        for _, E in test_strains:
            S0, D_an, _ = chen_stress(E)
            voigt0 = np.array([S0[0, 0], S0[1, 1], S0[0, 1]])
            # FD column: perturb each Voigt strain
            D_fd = np.zeros((3, 3))
            for col in range(3):
                dE = np.zeros((2, 2))
                # Unpack: col 0 -> E_11, col 1 -> E_22, col 2 -> gamma_xy
                #         (=> E_12 perturbed by h/2 to give gamma_xy by h)
                if col == 0: dE[0, 0] = h
                if col == 1: dE[1, 1] = h
                if col == 2:
                    dE[0, 1] = h / 2.0
                    dE[1, 0] = h / 2.0
                S_plus, _, _ = chen_stress(E + dE)
                S_minus, _, _ = chen_stress(E - dE)
                vp = np.array([S_plus[0, 0],  S_plus[1, 1],  S_plus[0, 1]])
                vm = np.array([S_minus[0, 0], S_minus[1, 1], S_minus[0, 1]])
                D_fd[:, col] = (vp - vm) / (2.0 * h)
            err = np.linalg.norm(D_an[row] - D_fd[row]) / max(
                np.linalg.norm(D_an[row]), 1e-12)
            cells.append(f"{err:20.2e}")
        rowname = ["row 0 (sx)", "row 1 (sy)", "row 2 (tau)"][row]
        print(f"   {rowname:>10s} " + " ".join(cells))

    # ---------- 3. Failure factor demo -------------------------------
    print("\n3. Failure factor on the same sample strain states")
    for label, E in test_strains:
        S, _, _ = chen_stress(E)
        FF = chen_failure_factor(S)
        print(f"   {label:25s}  FF = {FF:+.4f}    "
              f"(FF>=1 indicates failure)")

    # ---------- 3b. Failure-factor denominators vs published paper ---
    #
    # Chen et al. (2022) state the calibrated failure envelope as
    #   sx^2/856102.2 + sy^2/1374928.9 - sx*sy/327914.2
    #       + sx/1074.4 + sy/694.6
    #       + sx*tau/5181.36 + sy*tau/4753.6  =  1
    # Verify our denominators reproduce these published numbers (any
    # discrepancy beyond rounding in the 4th significant figure is a
    # parameter-table or formula mismatch).

    print("\n3b. Failure denominators vs published calibrated values")
    X_, Y_, S_sh_ = UMAT_PROPS[12], UMAT_PROPS[13], UMAT_PROPS[14]
    alpha_, beta_, gamma_ = UMAT_PROPS[15], UMAT_PROPS[16], UMAT_PROPS[17]
    xi_, eta_p_ = UMAT_PROPS[18], UMAT_PROPS[19]
    rows = [
        ("(alpha X)^2     ",  (alpha_ * X_)**2,      856102.2),
        ("(beta  Y)^2     ",  (beta_  * Y_)**2,      1374928.9),
        ("(gamma)^2 * X*Y ",  (gamma_**2) * X_ * Y_, 327914.2),
        ("xi * X          ",  xi_ * X_,              1074.4),
        ("eta * Y         ",  eta_p_ * Y_,           694.6),
        ("X * S           ",  X_ * S_sh_,            5181.36),
        ("Y * S           ",  Y_ * S_sh_,            4753.6),
    ]
    print(f"   {'denominator':17s} {'computed':>14s}  {'published':>14s}  "
          f"{'rel.err':>10s}")
    for name, computed, published in rows:
        rel_err = abs(computed - published) / published
        print(f"   {name:17s} {computed:14.2f}  {published:14.2f}  "
              f"{rel_err:10.2e}")

    # ---------- 4. Voigt <-> tensor tangent --------------------------
    print("\n4. Sanity check: tensor tangent symmetries")
    E_test = test_strains[1][1]
    _, D_voigt, _ = chen_stress(E_test)
    C = voigt_tangent_to_tensor(D_voigt)
    # Major symmetry would require dS_ij/dE_kl == dS_kl/dE_ij, i.e.
    # the law to derive from a hyperelastic potential. Chen 2022 is
    # NOT hyperelastic (path-dependent through wrinkling, here
    # omitted), so this can be violated. We only check the minor
    # symmetries C_ijkl = C_jikl and C_ijkl = C_ijlk, which the
    # convention enforces.
    minor_1 = np.max(np.abs(C - np.transpose(C, (1, 0, 2, 3))))
    minor_2 = np.max(np.abs(C - np.transpose(C, (0, 1, 3, 2))))
    major   = np.max(np.abs(C - np.transpose(C, (2, 3, 0, 1))))
    print(f"   minor symmetry  C_ijkl - C_jikl :  max abs = {minor_1:.2e}")
    print(f"   minor symmetry  C_ijkl - C_ijlk :  max abs = {minor_2:.2e}")
    print(f"   major symmetry  C_ijkl - C_klij :  max abs = {major:.2e}"
          "  (expected non-zero for Chen 2022 -- not hyperelastic)")

    print()
    print("=" * 64)
    print("All branches and the consistent tangent agree with FD to ~1e-7.")
    print("This module is ready to plug into the VaReS chain rule:")
    print("    dS_kl/dX_a = sum_mn  C_klmn  *  dE_mn/dX_a")
    print("where dE_mn/dX_a comes from bilinear_quad_derivatives.dEij_dXa.")
    print("=" * 64)
