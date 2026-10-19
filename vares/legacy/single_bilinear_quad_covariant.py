"""
single_bilinear_quad_covariant.py

Tutorial: a single 4-node bilinear quad element described using
*covariant base vectors* in 3D space, with a side-by-side
visualisation of a material (reference) and a spatial (deformed)
configuration.

Why this file exists
--------------------
The VaReS thesis chapter (Section 5.2) writes the FE kinematics of a
membrane element in terms of covariant base vectors G_i = dR/dtheta^i
and their dual basis G^i. If you have only ever seen standard
Cartesian FE notation (with constant orthonormal axes), the covariant
machinery looks bizarre. This script makes every quantity concrete:
you can read the numbers, see the arrows in 3D, and verify the
duality relation G^i . G_j = delta numerically.

The script is deliberately self-contained -- it does not import
anything from the cutting-pattern modules in this folder. It only
needs numpy and matplotlib.

Pieces covered
--------------
1. Bilinear shape functions N_k(xi, eta) and their derivatives.
2. Covariant base vectors G_1, G_2 (tangent to the parameter lines)
   and G_3 (unit normal).
3. The covariant metric tensor G_ij = G_i . G_j.
4. The contravariant (dual) basis G^1, G^2 satisfying G^i . G_j = delta.
5. Green-Lagrange strain E_kl = (1/2)(g_kl - G_kl) between two
   configurations, evaluated at the element centre.
6. A 3D plot of both configurations with their local frames.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# Importing Axes3D used to be required as a side-effect; in modern
# matplotlib (>= 3.2) projection="3d" alone is sufficient.


# ============================================================
# 1. Shape functions on the parametric square [-1, 1] x [-1, 1]
# ============================================================
#
# Local node numbering (counter-clockwise from the bottom-left
# corner). The parametric coordinates are (xi, eta) -- the thesis
# calls them theta^1, theta^2.
#
#                    eta
#                     ^
#                     |
#       N_4 (-1, +1)  o----------o  N_3 (+1, +1)
#                     |          |
#                     |   * (0,0)|     * = element centre
#                     |          |       (used for all reports below)
#       N_1 (-1, -1)  o----------o  N_2 (+1, -1)  --> xi
#
# The shape function N_k(xi, eta) equals 1 at node k, 0 at the other
# three corners, and varies bilinearly in between. The standard
# closed form is:

def N(xi: float, eta: float) -> np.ndarray:
    """Shape functions N_1..N_4 at parametric point (xi, eta)."""
    return 0.25 * np.array([
        (1 - xi) * (1 - eta),       # N_1 -- corner (-1, -1)
        (1 + xi) * (1 - eta),       # N_2 -- corner (+1, -1)
        (1 + xi) * (1 + eta),       # N_3 -- corner (+1, +1)
        (1 - xi) * (1 + eta),       # N_4 -- corner (-1, +1)
    ])


def dN(xi: float, eta: float) -> np.ndarray:
    """Shape-function derivatives w.r.t. (xi, eta).

    Returns a (4, 2) array where dN[k, 0] = dN_k/dxi and
    dN[k, 1] = dN_k/deta. For a bilinear quad these are LINEAR in
    (xi, eta) -- not constant as they are for a linear triangle.
    """
    return 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],    # node 1
        [ (1 - eta), -(1 + xi)],    # node 2
        [ (1 + eta),  (1 + xi)],    # node 3
        [-(1 + eta),  (1 - xi)],    # node 4
    ])


# ============================================================
# 2. Covariant base vectors at a parametric point
# ============================================================
#
# The position of any point inside the element in physical 3D space
# is the bilinear interpolation of the 4 nodal positions:
#
#     R(xi, eta) = sum_k  N_k(xi, eta) * X_k          (1)
#
# Differentiating (1) w.r.t. the parametric coordinates gives the
# *covariant base vectors*:
#
#     G_1 = dR/dxi   = sum_k (dN_k/dxi)  X_k          (2a)
#     G_2 = dR/deta  = sum_k (dN_k/deta) X_k          (2b)
#
# Geometric meaning:
#   - G_1 is the 3-vector you'd traverse if you moved by 1 unit of
#     xi while holding eta fixed -- i.e. it's tangent to the
#     "xi parameter line" at that point.
#   - G_2 is the same idea for eta.
#   - Both lie in the element's tangent plane.
#
# Crucially, |G_1| and |G_2| do *not* have to be 1, and G_1 . G_2
# does *not* have to be 0. They are whatever the geometry dictates.

def covariant_basis(X_nodes: np.ndarray, xi: float, eta: float):
    """Return (G_1, G_2, G_3) at parametric point (xi, eta).

    X_nodes : (4, 3) ndarray of nodal positions in R^3
              (rows = nodes 1..4 in the order shown above).

    G_1, G_2 are tangent to the surface at (xi, eta).
    G_3 is the unit normal (G_1 x G_2 normalised).

    Raises ValueError if the quad is degenerate at (xi, eta) -- i.e.
    if ||G_1 x G_2|| is too small to define a unit normal. This is the
    common failure mode when you edit X_nodes carelessly (collinear
    points, duplicated nodes, zero-area quad).
    """
    X_nodes = np.asarray(X_nodes, dtype=float)
    if X_nodes.shape != (4, 3):
        raise ValueError(
            f"X_nodes must have shape (4, 3); got {X_nodes.shape}. "
            "Each row is one nodal position (x, y, z) in R^3.")

    grad = dN(xi, eta)                              # (4, 2)
    G_1 = grad[:, 0] @ X_nodes                      # (3,)
    G_2 = grad[:, 1] @ X_nodes                      # (3,)
    cross = np.cross(G_1, G_2)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1.0e-12:
        raise ValueError(
            "Degenerate quad at (xi, eta) = "
            f"({xi}, {eta}): ||G_1 x G_2|| = {cross_norm:.3e}. "
            "Likely cause: two or more nodes coincide, or all four "
            "lie on a straight line. Check X_nodes:\n"
            f"{X_nodes}")
    G_3 = cross / cross_norm
    return G_1, G_2, G_3


# ============================================================
# 3. Covariant metric tensor
# ============================================================
#
# G_ij = G_i . G_j     for i, j in {1, 2}.
#
# Geometric meaning:
#   - G_11 = |G_1|^2     "squared length per unit xi"
#   - G_22 = |G_2|^2     "squared length per unit eta"
#   - G_12 = G_1 . G_2   off-diagonal -- non-zero iff the basis is
#                        skewed (not orthogonal)
#
# In an orthonormal Cartesian basis G_ij = identity. Any departure
# from identity tells you the element's local frame is not Cartesian.

def covariant_metric(G_1: np.ndarray, G_2: np.ndarray) -> np.ndarray:
    """Return the 2x2 covariant metric tensor G_ij."""
    return np.array([[G_1 @ G_1, G_1 @ G_2],
                     [G_2 @ G_1, G_2 @ G_2]])


# ============================================================
# 4. Contravariant (dual) base vectors
# ============================================================
#
# Defined by the duality relation:
#     G^i . G_j = delta^i_j                          (3)
#
# Useful when you want to read off *components* of a vector v in the
# covariant basis: dotting v with G^i gives the i-th contravariant
# component. In an orthonormal basis G^i = G_i; otherwise they differ.
#
# Practical construction in code:
#   1. Stack G_1, G_2 as rows of a (2, 3) matrix M_lower.
#   2. The 2x2 covariant metric is G_ij = M_lower @ M_lower.T.
#   3. The 2x2 contravariant metric is G^ij = inv(G_ij).
#   4. The contravariant basis is G^i = G^ij G_j, i.e.
#      M_upper = G^ij @ M_lower (each row is a G^i vector in R^3).

def contravariant_basis(G_1: np.ndarray, G_2: np.ndarray):
    """Return (G^1, G^2, G_ij, G^ij).

    Raises ValueError if the covariant metric G_ij is (near-)singular,
    i.e. if G_1 and G_2 are linearly dependent. This corresponds to
    the same geometric pathology covariant_basis already rejects, but
    it can also surface independently if you ever call this function
    on hand-supplied G_1, G_2 vectors.
    """
    M_lower = np.stack([G_1, G_2])                  # (2, 3)
    G_ij = M_lower @ M_lower.T                      # (2, 2) covariant
    det = float(np.linalg.det(G_ij))
    if det < 1.0e-20:
        raise ValueError(
            f"Covariant metric is singular (det G_ij = {det:.3e}); "
            "G_1 and G_2 are linearly dependent so a dual basis "
            "doesn't exist. Inspect the input X_nodes for degenerate "
            "geometry (collinear / duplicated nodes).")
    G_up_ij = np.linalg.inv(G_ij)                   # (2, 2) contravariant
    M_upper = G_up_ij @ M_lower                     # (2, 3)
    return M_upper[0], M_upper[1], G_ij, G_up_ij


# ============================================================
# 5. Green-Lagrange strain between two configurations
# ============================================================
#
# Comparing the same parametric point (xi, eta) in two configs:
#   - Material (reference, X) has covariant metric G_ij.
#   - Spatial  (deformed,  x) has covariant metric g_ij.
#
# Green-Lagrange strain components (covariant indices):
#
#     E_kl = (1/2) (g_kl - G_kl)                     (4)
#
# These are *not* Cartesian-frame strains. They are components in
# the (G^k, G^l) contravariant basis. If you want Cartesian-frame
# strains you have to rotate -- but the point of working in the
# covariant frame is precisely that you don't have to.

def green_lagrange_strain(X_material: np.ndarray,
                          X_spatial : np.ndarray,
                          xi: float, eta: float):
    G1_m, G2_m, _ = covariant_basis(X_material, xi, eta)
    g1_s, g2_s, _ = covariant_basis(X_spatial,  xi, eta)
    G_ij = covariant_metric(G1_m, G2_m)
    g_ij = covariant_metric(g1_s, g2_s)
    return 0.5 * (g_ij - G_ij), G_ij, g_ij


# ============================================================
# 6. Define two configurations
# ============================================================
#
# Material (reference): a 2x2 square in the xy plane, centred on the
# origin. With this choice the parametric coordinates (xi, eta) in
# [-1, 1] coincide with the physical (x, y) coordinates -- so G_1
# and G_2 come out as unit Cartesian directions and G_ij = identity.
# This is the "Cartesian sanity-check" configuration -- everything
# behaves exactly as it would in a flat Cartesian frame.

def report_strain(X_mat: np.ndarray, X_sp: np.ndarray,
                  label: str) -> np.ndarray:
    """Compute strain at centre for one (material, spatial) pair."""
    try:
        E, _, _ = green_lagrange_strain(X_mat, X_sp, 0.0, 0.0)
    except ValueError as exc:
        print(f"  {label:42s}  --  REJECTED: {exc.args[0].splitlines()[0]}")
        return None
    print(f"  {label:42s}  E_11={E[0,0]:+.4f}  "
          f"E_22={E[1,1]:+.4f}  E_12={E[0,1]:+.4f}")
    return E

def sample_surface(X_nodes: np.ndarray, n: int = 15) -> np.ndarray:
    """Sample physical positions on an n x n parametric grid."""
    xi  = np.linspace(-1, 1, n)
    eta = np.linspace(-1, 1, n)
    surf = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            surf[i, j] = N(xi[i], eta[j]) @ X_nodes
    return surf


def plot_config(ax, X_nodes: np.ndarray, title: str):
    surf = sample_surface(X_nodes, n=11)

    # Parametric grid lines (lines of constant xi or constant eta)
    for i in range(surf.shape[0]):
        ax.plot(surf[i, :, 0], surf[i, :, 1], surf[i, :, 2],
                color='0.7', lw=0.6)
        ax.plot(surf[:, i, 0], surf[:, i, 1], surf[:, i, 2],
                color='0.7', lw=0.6)

    # Element edges (heavier)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        ax.plot(*zip(X_nodes[a], X_nodes[b]), 'k-', lw=2)

    # Nodes
    for k, p in enumerate(X_nodes, start=1):
        ax.scatter(*p, s=70, c='C0', zorder=5)
        ax.text(p[0] + 0.05, p[1] + 0.05, p[2] + 0.02,
                f'N{k}', fontsize=11, weight='bold')

    # Local frame at the centre (xi, eta) = (0, 0)
    centre = N(0.0, 0.0) @ X_nodes
    G_1, G_2, G_3 = covariant_basis(X_nodes, 0.0, 0.0)

    # Covariant base vectors (solid)
    ax.quiver(*centre, *G_1, color='C3', lw=2.5, arrow_length_ratio=0.15)
    ax.quiver(*centre, *G_2, color='C2', lw=2.5, arrow_length_ratio=0.15)
    ax.quiver(*centre, *(0.5 * G_3), color='C0', lw=2.5,
              arrow_length_ratio=0.2)

    # Contravariant (dual) basis (dashed lines -- 3D quiver doesn't
    # support linestyle, so we draw them as line segments)
    Gu_1, Gu_2, _, _ = contravariant_basis(G_1, G_2)
    end1 = centre + Gu_1
    end2 = centre + Gu_2
    ax.plot(*zip(centre, end1), color='C3', lw=1.4, linestyle='--')
    ax.plot(*zip(centre, end2), color='C2', lw=1.4, linestyle='--')

    # Annotations for the frame arrows
    ax.text(*(centre + G_1 * 1.05), '$G_1$', color='C3', fontsize=10)
    ax.text(*(centre + G_2 * 1.05), '$G_2$', color='C2', fontsize=10)
    ax.text(*(centre + 0.5 * G_3 * 1.10), '$G_3$', color='C0', fontsize=10)
    ax.text(*(end1 * 1.02), '$G^1$', color='C3', fontsize=9, style='italic')
    ax.text(*(end2 * 1.02), '$G^2$', color='C2', fontsize=9, style='italic')

    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title(title, fontsize=10)




def _tutorial():
    """The worked example of the module docstring: prints the covariant analysis of one element and writes
    quad_covariant_visualisation.png into the current directory. Runs only as a script."""
    X_material = np.array([
        [-1.0, -1.0, 0.0],          # N_1
        [ 1.0, -1.0, 0.0],          # N_2
        [ 1.0,  1.0, 0.0],          # N_3
        [-1.0,  1.0, 0.0],          # N_4
    ])

    # Spatial (deformed): the same parametric square mapped to a stretched
    # and tilted 3D quad. xi-direction is stretched 1.5x; corners are
    # lifted unevenly so the spatial element is non-planar. The g_ij at
    # the centre will NOT be identity -- non-zero shear, anisotropic
    # stretch.

    X_spatial = np.array([
        [-1.5, -1.0, 0.0],
        [ 1.5, -1.0, 0.3],          # right-bottom corner lifted slightly
        [ 1.5,  1.0, 1.0],          # right-top corner lifted a lot
        [-1.5,  1.0, 0.4],          # left-top corner lifted moderately
    ])


    # ============================================================
    # 7. Compute and report all quantities at the element centre
    # ============================================================
    #
    # Evaluating at (xi, eta) = (0, 0) -- the element's parametric
    # centre. For the chosen material configuration this is also the
    # physical centroid; for the spatial one it is the bilinear-
    # interpolated centre, generally NOT the geometric centroid.

    xi0, eta0 = 0.0, 0.0

    print("=" * 64)
    print("Single bilinear quad element -- covariant analysis at centre")
    print(f"Parametric point: (xi, eta) = ({xi0}, {eta0})")
    print("=" * 64)

    # Shape-function derivatives at the centre. For a bilinear quad on
    # [-1, 1]^2, all four dN_k/dxi at the centre equal +/- 0.25 in the
    # specific pattern below -- these constants drive every subsequent
    # derivative w.r.t. nodal coordinates X_a.
    dN_centre = dN(xi0, eta0)
    print("\nShape-function derivatives at centre (each row = node k):")
    print(f"  dN_k/dxi  : {dN_centre[:, 0]}")
    print(f"  dN_k/deta : {dN_centre[:, 1]}")

    # ---- Material ----------------------------------------------------
    G1_m, G2_m, G3_m = covariant_basis(X_material, xi0, eta0)
    Gup1_m, Gup2_m, G_ij_m, Gup_ij_m = contravariant_basis(G1_m, G2_m)

    print("\nMATERIAL config -- flat 2x2 square in xy plane:")
    print(f"  G_1 = {G1_m}    |G_1| = {np.linalg.norm(G1_m):.4f}")
    print(f"  G_2 = {G2_m}    |G_2| = {np.linalg.norm(G2_m):.4f}")
    print(f"  G_3 = {G3_m}    (unit normal)")
    print(f"  G_ij =")
    print(f"    {G_ij_m[0]}")
    print(f"    {G_ij_m[1]}")
    print("  --> G_ij = identity, so this basis IS Cartesian-orthonormal.")
    print(f"  G^1 = {Gup1_m}      (equals G_1 here because basis is orthonormal)")
    print(f"  G^2 = {Gup2_m}      (equals G_2)")
    print(f"  Duality check G^i . G_j (should equal Kronecker delta):")
    print(f"    G^1.G_1 = {Gup1_m @ G1_m:+.4f}     G^1.G_2 = {Gup1_m @ G2_m:+.4f}")
    print(f"    G^2.G_1 = {Gup2_m @ G1_m:+.4f}     G^2.G_2 = {Gup2_m @ G2_m:+.4f}")

    # ---- Spatial -----------------------------------------------------
    g1_s, g2_s, g3_s = covariant_basis(X_spatial, xi0, eta0)
    gup1_s, gup2_s, g_ij_s, gup_ij_s = contravariant_basis(g1_s, g2_s)

    print("\nSPATIAL config -- stretched, tilted, non-planar:")
    print(f"  g_1 = {g1_s}    |g_1| = {np.linalg.norm(g1_s):.4f}")
    print(f"  g_2 = {g2_s}    |g_2| = {np.linalg.norm(g2_s):.4f}")
    print(f"  g_3 = {g3_s}    (unit normal)")
    print(f"  g_ij =")
    print(f"    {g_ij_s[0]}")
    print(f"    {g_ij_s[1]}")
    print("  --> g_ij is NOT identity. Off-diagonal g_12 != 0 means the")
    print("      basis is skewed; diagonal entries != 1 means the basis")
    print("      is stretched relative to unit parameter steps.")
    print(f"  g^1 = {gup1_s}      (DIFFERS from g_1 because basis is non-orthonormal)")
    print(f"  g^2 = {gup2_s}      (DIFFERS from g_2)")
    print(f"  Duality check g^i . g_j:")
    print(f"    g^1.g_1 = {gup1_s @ g1_s:+.4f}     g^1.g_2 = {gup1_s @ g2_s:+.4f}")
    print(f"    g^2.g_1 = {gup2_s @ g1_s:+.4f}     g^2.g_2 = {gup2_s @ g2_s:+.4f}")

    # ---- Strain ------------------------------------------------------
    E_low, G_ij, g_ij = green_lagrange_strain(X_material, X_spatial, xi0, eta0)
    print("\nGREEN-LAGRANGE STRAIN at centre  (E_kl = 1/2 (g_kl - G_kl)):")
    print(f"  E_kl =")
    print(f"    {E_low[0]}")
    print(f"    {E_low[1]}")
    print(f"  E_11 = {E_low[0, 0]:+.4f}    (xi-direction normal strain)")
    print(f"  E_22 = {E_low[1, 1]:+.4f}    (eta-direction normal strain)")
    print(f"  E_12 = {E_low[0, 1]:+.4f}    (shear strain)")
    print()
    print("  Interpretation: E_11 > 0 means the spatial element is longer")
    print("  per unit xi than the material element -- a stretch in that")
    print("  parameter direction. E_22, E_12 likewise.")
    print("  These are NOT Cartesian strains; they are components in the")
    print("  (G^k, G^l) basis. The same physical state expressed in a")
    print("  local Cartesian frame would have different numerical values.")


    # ============================================================
    # 7b. VaReS-style sweep: vary X_material, hold X_spatial fixed
    # ============================================================
    #
    # This is the core VaReS premise: the SPATIAL configuration is given
    # and held FIXED (it's the target shape -- the draped cylinder, say).
    # The MATERIAL configuration is the design variable. As you vary
    # X_material, the strain field changes -- and the VaReS optimiser
    # would walk X_material until the strain (or some functional of it)
    # is minimised.
    #
    # Below we sweep through a handful of trial material configurations,
    # keeping X_spatial untouched, and report the resulting strain. You
    # should see:
    #   - The trivial "material == spatial projection" case gives small
    #     or zero strain (depending on the projection).
    #   - Stretching the material LARGER produces NEGATIVE strain (the
    #     material is bigger than the spatial state -> compressive).
    #   - Shrinking the material produces POSITIVE strain (tensile).
    #   - Distorting the material asymmetrically produces shear.

    print("\n" + "=" * 64)
    print("VaReS validation sweep: vary X_material, hold X_spatial fixed")
    print("=" * 64)
    print("Strain at centre  (X_spatial unchanged across all rows below)")
    print()

    # Reference row: original X_material defined at the top of the file
    report_strain(X_material, X_spatial, "original X_material (baseline)")

    # Same shape as spatial projected to z=0: the material that exactly
    # matches the spatial XY footprint. This kills xi/eta normal strain
    # from the in-plane stretch, leaves only the out-of-plane (z) lift
    # contribution.
    X_proj = X_spatial.copy()
    X_proj[:, 2] = 0.0
    report_strain(X_proj, X_spatial, "spatial projected to z=0")

    # Uniformly larger material (1.5x in xi direction)
    X_big_xi = X_material.copy()
    X_big_xi[:, 0] *= 1.5
    report_strain(X_big_xi, X_spatial, "X_material stretched 1.5x in x")

    # Uniformly smaller material
    X_small = X_material * 0.7
    report_strain(X_small, X_spatial, "X_material scaled 0.7x")

    # Shear the material -- shift top edge in +x
    X_sheared = X_material.copy()
    X_sheared[2, 0] += 0.6   # node 3 (top-right)
    X_sheared[3, 0] += 0.6   # node 4 (top-left)
    report_strain(X_sheared, X_spatial, "X_material sheared (+0.6 on top)")

    # Degenerate: collapse two nodes (should be rejected with clear error)
    X_bad = X_material.copy()
    X_bad[1] = X_bad[0]                       # node 2 == node 1
    report_strain(X_bad, X_spatial, "X_material with N2 collapsed onto N1")

    print()
    print("Observations to internalise:")
    print("  - X_spatial never changed -- only X_material varies row to row.")
    print("  - Strain DOES change in response. That is the VaReS variable")
    print("    of interest: pick the X_material that minimises some")
    print("    functional of E (energy, von Mises, target-stress match).")
    print("  - Degenerate material meshes are rejected up front, not")
    print("    silently producing NaN strain.")


    # ============================================================
    # 8. 3D visualisation
    # ============================================================
    #
    # Two panels side by side: material on the left, spatial on the
    # right. Each panel shows:
    #   - the 4 corner nodes (labelled)
    #   - the element edges
    #   - a parametric grid (lines of constant xi or eta inside the
    #     element), which shows how the parametric coordinates map onto
    #     physical space
    #   - the local frame (G_1, G_2, G_3) drawn at the centre as solid
    #     arrows
    #   - the dual basis (G^1, G^2) drawn at the centre as dashed lines

    fig = plt.figure(figsize=(13, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_config(ax1, X_material,
                "Material configuration\n"
                "G_ij = identity (orthonormal); G^i = G_i")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    g11, g22, g12 = g_ij_s[0, 0], g_ij_s[1, 1], g_ij_s[0, 1]
    ax2_title = ("Spatial configuration\n"
                 f"g_11={g11:.2f}  g_22={g22:.2f}  g_12={g12:.2f}  "
                 "(non-orthonormal)")
    plot_config(ax2, X_spatial, ax2_title)

    # Legend hint -- describe the colour scheme once, off to the side
    fig.text(0.5, 0.02,
             "Solid arrows: covariant base vectors  G_1 (red), G_2 (green), "
             "G_3 (blue, scaled 0.5x).   "
             "Dashed lines: contravariant duals G^1, G^2.   "
             "Grey grid: lines of constant xi or eta inside the element.",
             ha='center', fontsize=8.5)

    for ax in (ax1, ax2):
        ax.set_box_aspect((1, 1, 0.6))

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_png = "quad_covariant_visualisation.png"
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f"\nFigure written: {out_png}")

    # plt.show() opens an interactive window and BLOCKS until you close
    # it. If you press Ctrl+C in the terminal (or kill the window via the
    # OS) the script exits with KeyboardInterrupt -- that is not a bug,
    # it just means the GUI loop was cancelled. The PNG above was already
    # written to disk by the time show() ran.
    #
    # Set SHOW_INTERACTIVE = False below if you only want the PNG (e.g.
    # when iterating quickly on X_material without wanting a window each
    # run).  Defaults to True only when THIS file is run directly: this
    # module is imported (for N/dN/covariant_basis) by every VaReS
    # optimiser script, and a blocking window at import time would stall
    # those scripts before they even start.
    SHOW_INTERACTIVE = True
    if SHOW_INTERACTIVE:
        print("Opening interactive 3D window -- close it to exit cleanly.")
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    _tutorial()
