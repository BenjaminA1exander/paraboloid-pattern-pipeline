"""
flatten_semicylinder_validation.py

First geometric-validation stage of the cutting-pattern optimisation
pipeline for pressurised membrane / shell panels.

Pipeline context
----------------
The long-term workflow is:

    target 3D shape
        |
        v
    2D cutting panel  <-----+
        |                   |
        v                   |  Bayesian update of panel
    Abaqus/Standard         |  (panel parameters / node positions)
    inflation + Chen UMAT   |
        |                   |
        v                   |
    inflated 3D shape ------+
        |   compare against target
        v
    objective J

This script implements ONLY the first geometric step: take a simple
3D shell patch (a semi-cylinder), project it orthogonally to 2D to
obtain a raw flat panel, then apply Laplacian smoothing to regularise
the flat mesh.  The smoothed 2D mesh is the kind of "candidate cutting
panel" that will later be:
    * written to an Abaqus .inp (S4R quads),
    * inflated against pressure / form-found with the Chen UMAT,
    * compared to a target 3D shape,
    * and updated by the Bayesian optimiser.

No Abaqus call, no UMAT, no optimiser code lives here yet -- this
file is purely the geometric validation harness.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import least_squares


# ============================================================
# 1. Mesh generation
# ============================================================

def generate_semicylinder_mesh(R: float,
                               L: float,
                               n_circ: int,
                               n_axial: int):
    """Build a structured quad mesh on a semi-cylinder.

    Parameterisation (axis along +x):
        x     = axial coordinate
        y     = R cos(theta)
        z     = R sin(theta),     theta in [0, pi]

    Parameters
    ----------
    R : float
        Cylinder radius.
    L : float
        Axial length.
    n_circ : int
        Number of elements in the circumferential (theta) direction.
    n_axial : int
        Number of elements in the axial (x) direction.

    Returns
    -------
    nodes_3d : (N, 3) ndarray
        Node coordinates in row-major (j, i) order with
        j = theta index (0..n_circ), i = axial index (0..n_axial).
        Flat index = j * (n_axial + 1) + i.
    """
    nx = n_axial + 1       # nodes along x
    ny = n_circ  + 1       # nodes along theta

    xs = np.linspace(0.0, L, nx)
    thetas = np.linspace(0.0, math.pi, ny)

    nodes = np.zeros((nx * ny, 3), dtype=float)
    for j, th in enumerate(thetas):
        cy = R * math.cos(th)
        cz = R * math.sin(th)
        for i, x in enumerate(xs):
            nodes[j * nx + i] = (x, cy, cz)
    return nodes


def build_quad_connectivity(n_circ: int, n_axial: int) -> np.ndarray:
    """Quad connectivity for the structured (n_circ x n_axial) grid.

    Winding is counter-clockwise looking from +outside (i.e. from
    larger radius outward), which keeps face normals pointing out of
    the semi-cylinder.

    Returns
    -------
    elements : (n_circ * n_axial, 4) int ndarray
    """
    nx = n_axial + 1
    quads = []
    for j in range(n_circ):
        for i in range(n_axial):
            n1 =  j      * nx + i
            n2 =  j      * nx + i + 1
            n3 = (j + 1) * nx + i + 1
            n4 = (j + 1) * nx + i
            quads.append((n1, n2, n3, n4))
    return np.asarray(quads, dtype=int)


def build_tri_connectivity(n_circ: int, n_axial: int,
                           *, alternating: bool = True) -> np.ndarray:
    """Native linear-triangle connectivity for the structured grid.

    Each structured cell is split into two triangles directly on the
    semi-cylinder node grid (no intermediate quad object).  With
    ``alternating=True`` the splitting diagonal flips on a checkerboard
    of the (i, j) cell indices, which avoids the directional stiffness
    bias of an all-same-diagonal triangulation.  Winding is kept
    counter-clockwise from outside so triangle normals match the quad
    mesh.

    Cell corners (same labelling as build_quad_connectivity):
        n1 = bottom-left, n2 = bottom-right, n3 = top-right, n4 = top-left

        diagonal n1-n3 :  (n1, n2, n3), (n1, n3, n4)
        diagonal n2-n4 :  (n1, n2, n4), (n2, n3, n4)

    Returns
    -------
    elements : (2 * n_circ * n_axial, 3) int ndarray
    """
    nx = n_axial + 1
    tris = []
    for j in range(n_circ):
        for i in range(n_axial):
            n1 =  j      * nx + i
            n2 =  j      * nx + i + 1
            n3 = (j + 1) * nx + i + 1
            n4 = (j + 1) * nx + i
            if (not alternating) or ((i + j) % 2 == 0):
                tris.append((n1, n2, n3))
                tris.append((n1, n3, n4))
            else:
                tris.append((n1, n2, n4))
                tris.append((n2, n3, n4))
    return np.asarray(tris, dtype=int)


def build_mixed_connectivity(n_circ: int, n_axial: int):
    """Mixed quad/triangle connectivity for the structured grid.

    Cells on one colour of an (i, j) checkerboard are kept as 4-node
    quads; the others are split into two 3-node triangles.  This
    exercises the per-element dispatch in the VaReS assembly (the
    element type is resolved from each element's node count).

    Because the elements have different node counts they cannot be
    packed into a rectangular array, so this returns a PYTHON LIST of
    int ndarrays (length 4 for quads, 3 for triangles).
    """
    nx = n_axial + 1
    elems = []
    for j in range(n_circ):
        for i in range(n_axial):
            n1 =  j      * nx + i
            n2 =  j      * nx + i + 1
            n3 = (j + 1) * nx + i + 1
            n4 = (j + 1) * nx + i
            if (i + j) % 2 == 0:
                elems.append(np.array([n1, n2, n3, n4], dtype=int))
            else:
                elems.append(np.array([n1, n2, n3], dtype=int))
                elems.append(np.array([n1, n3, n4], dtype=int))
    return elems


def find_boundary_nodes(n_circ: int, n_axial: int):
    """Return (boundary_idx, interior_idx) as int ndarrays.

    The boundary of the structured grid is the four parametric edges:
        theta = 0,  theta = pi,  x = 0,  x = L.

    Under orthogonal z-projection these four edges become the rim of
    the 2D flat panel and must be held fixed by the smoother, both to
    keep the panel size invariant and because in the eventual Abaqus
    workflow they are the seam / clamp lines whose location is set by
    the optimiser.
    """
    nx = n_axial + 1
    ny = n_circ  + 1
    n_nodes = nx * ny

    boundary = set()
    boundary.update(range(0, nx))                              # theta = 0
    boundary.update(range((ny - 1) * nx, ny * nx))             # theta = pi
    boundary.update(j * nx        for j in range(ny))          # x = 0
    boundary.update(j * nx + nx-1 for j in range(ny))          # x = L

    boundary_idx = np.array(sorted(boundary), dtype=int)
    interior_idx = np.array(sorted(set(range(n_nodes)) - boundary),
                            dtype=int)
    return boundary_idx, interior_idx


def build_node_adjacency(elements: np.ndarray, n_nodes: int):
    """Edge-neighbour adjacency from quad connectivity.

    Two nodes are neighbours iff they share a quad edge (4 edges per
    quad, undirected).  Returns a list-of-sets indexed by node id.
    """
    nbrs = [set() for _ in range(n_nodes)]
    for q in elements:
        for a, b in ((q[0], q[1]), (q[1], q[2]),
                     (q[2], q[3]), (q[3], q[0])):
            nbrs[a].add(int(b))
            nbrs[b].add(int(a))
    return [sorted(s) for s in nbrs]


# ============================================================
# 2. Projection
# ============================================================

def project_mesh_to_xy(nodes_3d: np.ndarray) -> np.ndarray:
    """Orthogonal projection onto the xy plane (drop z).

    This is deliberately *not* a developable unrolling -- a
    semi-cylinder is developable but the straight-down projection
    compresses theta non-uniformly (cos(theta) spacing).  That
    distortion is exactly the signal the downstream optimiser will be
    asked to correct; smoothing here merely produces a sensible
    starting point.
    """
    return nodes_3d[:, :2].copy()


# ============================================================
# 3. Laplacian smoothing
# ============================================================

def build_unique_edges(elements) -> np.ndarray:
    """Unique undirected element edges, returned as an (E, 2) int array.

    Iterates the boundary edges of each element cyclically, so it works
    for quads (4 edges), triangles (3 edges) and mixed connectivity
    lists alike.  Sorting each pair (i, j) so i < j collapses the two
    directed occurrences of every shared interior edge into one entry,
    which is what the symmetric graph Laplacian wants.
    """
    pairs = set()
    for elem in elements:
        m = len(elem)
        for k in range(m):
            a = int(elem[k])
            b = int(elem[(k + 1) % m])
            pairs.add((min(a, b), max(a, b)))
    return np.asarray(sorted(pairs), dtype=np.int64)


def build_slide_mask(n_circ: int, n_axial: int) -> np.ndarray:
    """Per-axis free / pinned mask for an 'edge-fixed' rim.

    Returns an (N, 2) float array: 1.0 = free, 0.0 = pinned.

    The rim of the projected rectangle stays where it is in space,
    but rim nodes are free to slide *along* their own segment:

        theta = 0  row (y =  R) -> y pinned, x free
        theta = pi row (y = -R) -> y pinned, x free
        x     = 0  col          -> x pinned, y free
        x     = L  col          -> x pinned, y free
        corners (intersection of two segments) -> both axes pinned
        interior nodes                         -> both axes free
    """
    nx = n_axial + 1
    ny = n_circ  + 1
    mask = np.ones((nx * ny, 2), dtype=float)
    mask[0 * nx        : 1 * nx,  1] = 0.0     # theta = 0  row: pin y
    mask[(ny - 1) * nx : ny * nx, 1] = 0.0     # theta = pi row: pin y
    mask[0      :: nx, 0] = 0.0                # x = 0  col: pin x
    mask[nx - 1 :: nx, 0] = 0.0                # x = L  col: pin x
    return mask


def laplacian_smooth_2d(nodes_2d: np.ndarray,
                        elements: np.ndarray,
                        free_mask: np.ndarray) -> np.ndarray:
    """Edge-fixed Laplacian smoothing via direct harmonic solve.

    Smoothing each node toward its neighbours' centroid (Jacobi /
    Gauss-Seidel iteration) is just an iterative solver for the
    discrete harmonic Dirichlet problem

        L x = 0   with x prescribed on the pinned axis of each node,

    where L is the symmetric graph Laplacian
    (L[i,i] = deg(i), L[i,j] = -1 for each neighbour j).

    Instead of iterating until convergence, this function solves the
    system in one shot per axis with a sparse direct solve, returning
    the limit of infinite Laplacian smoothing sweeps.  Cost is O(N)
    for planar meshes (sparse Cholesky on a banded matrix) and runs
    in microseconds for the validation mesh, with comfortable
    headroom for the larger optimiser-driven panels.

    Parameters
    ----------
    nodes_2d : (N, 2) ndarray
        Initial 2D node positions.  Pinned-axis coordinates are used
        as Dirichlet data; free-axis coordinates are overwritten.
    elements : (M, 4) int ndarray
        Quad connectivity.
    free_mask : (N, 2) ndarray
        Per-axis 1.0 = free / 0.0 = pinned mask.  Use
        ``build_slide_mask(n_circ, n_axial)`` for the edge-fixed /
        slide-along-rim constraint.  Each axis must have at least one
        pinned node per connected mesh component or the linear system
        is singular.

    Returns
    -------
    nodes_smooth : (N, 2) ndarray
    """
    pts = nodes_2d.copy()
    n_nodes = pts.shape[0]
    edges = build_unique_edges(elements)
    n_edges = edges.shape[0]

    # Symmetric graph Laplacian L (N x N) assembled from COO triplets.
    # Each edge (i, j) contributes:  L[i,i] += 1, L[j,j] += 1,
    #                                L[i,j] -= 1, L[j,i] -= 1.
    i_idx = edges[:, 0]
    j_idx = edges[:, 1]
    rows = np.concatenate([i_idx, j_idx, i_idx, j_idx])
    cols = np.concatenate([i_idx, j_idx, j_idx, i_idx])
    ones = np.ones(n_edges)
    data = np.concatenate([ones, ones, -ones, -ones])
    L = coo_matrix((data, (rows, cols)),
                   shape=(n_nodes, n_nodes)).tocsr()

    out = pts.copy()
    for axis in range(2):
        free  = free_mask[:, axis] > 0.5
        fixed = ~free
        if not free.any():
            continue
        free_idx  = np.flatnonzero(free)
        fixed_idx = np.flatnonzero(fixed)
        L_ff = L[free_idx, :][:, free_idx].tocsc()
        L_fc = L[free_idx, :][:, fixed_idx]
        rhs  = -L_fc @ pts[fixed_idx, axis]
        out[free_idx, axis] = spsolve(L_ff, rhs)
    return out


# ============================================================
# 3b. Least-squares edge-length flattening
# ============================================================

def edge_lengths(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Euclidean length of every edge in ``edges`` on the point set
    ``nodes`` (works for 2D or 3D node arrays)."""
    d = nodes[edges[:, 0]] - nodes[edges[:, 1]]
    return np.linalg.norm(d, axis=1)


def _procrustes_align(X: np.ndarray, target: np.ndarray,
                      *, allow_reflection: bool = True) -> np.ndarray:
    """Rigid alignment (rotation + translation, optional reflection,
    NO scaling) of ``X`` onto ``target`` in the least-squares sense.

    Kabsch solution: with the centred clouds Xc, Pc and cross-covariance
    H = Xc^T Pc = U S V^T, the optimal orthogonal map is R = V U^T.
    Allowing a reflection (det R = -1) lets the flattened layout match
    the handedness of the reference projection; forbidding it forces a
    proper rotation by flipping the sign of the smallest singular axis.
    """
    cX = X.mean(axis=0)
    cP = target.mean(axis=0)
    Xc = X - cX
    Pc = target - cP
    H = Xc.T @ Pc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if not allow_reflection and np.linalg.det(R) < 0.0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return Xc @ R.T + cP


def least_squares_edge_flatten(nodes_3d: np.ndarray,
                               elements,
                               *,
                               init_2d: np.ndarray | None = None,
                               weights: np.ndarray | None = None,
                               anchor_to: np.ndarray | None = None,
                               allow_reflection: bool = True,
                               eps: float = 1.0e-12,
                               max_nfev: int = 400,
                               verbose: bool = False) -> np.ndarray:
    """Flatten a 3D mesh by least-squares edge-length preservation.

    Finds 2D node positions X that minimise the spring energy

        E(X) = 1/2 sum_e w_e (||x_i - x_j|| - L_e)^2,

    where L_e is the length of edge e measured on the ORIGINAL 3D mesh.
    This is the 'as-developable-as-possible' flat layout: for a
    genuinely developable surface (e.g. the semi-cylinder) it recovers
    the isometric unrolling (the L x pi*R rectangle) up to a planar
    rigid motion, rather than the distorted width-2R orthogonal
    projection produced by :func:`project_mesh_to_xy`.

    The residuals r_e = sqrt(w_e) (||x_i - x_j|| - L_e) are minimised
    with a sparse-Jacobian trust-region (scipy ``least_squares`` 'trf').
    The objective is invariant to planar rigid motion (a 3-dim null
    space; +1 more if reflections are allowed), so the raw solution
    floats freely.  When ``anchor_to`` is given the result is
    Procrustes-aligned onto that reference layout so it overlays the
    other starting meshes in the same frame.

    Parameters
    ----------
    nodes_3d : (N, 3) ndarray
        Original 3D node coordinates; edge lengths are taken from here.
    elements : (M, k) int ndarray or list of int ndarrays
        Connectivity used to derive the edge graph (via
        :func:`build_unique_edges`).  Pass a *triangulated* connectivity
        (e.g. ``build_tri_connectivity``) so the diagonal edges remove
        the planar shear/hourglass freedom of a quad-only graph.
    init_2d : (N, 2) or (N, 3) ndarray, optional
        Initial 2D guess.  Defaults to the orthogonal xy projection of
        ``nodes_3d``.  Only the first two columns are used.
    weights : (E,) ndarray, optional
        Per-edge weights w_e (default: all ones).
    anchor_to : (N, 2) or (N, 3) ndarray, optional
        Reference layout to Procrustes-align the result onto.  Pass the
        raw projection to keep the panel in the existing frame.
    allow_reflection : bool, default True
        Whether the Procrustes alignment may reflect (match handedness).
    eps : float
        Softening added under the square root so the Jacobian stays
        finite if an edge ever collapses; harmless at O(1) edge lengths.
    max_nfev : int
        Trust-region iteration budget.
    verbose : bool
        Forward scipy's iteration log when True.

    Returns
    -------
    nodes_2d : (N, 2) ndarray
    """
    edges = build_unique_edges(elements)
    n_nodes = nodes_3d.shape[0]
    n_edges = edges.shape[0]
    L0 = edge_lengths(nodes_3d, edges)

    w = np.ones(n_edges) if weights is None else np.asarray(weights, float)
    sw = np.sqrt(w)

    if init_2d is None:
        X0 = nodes_3d[:, :2].astype(float).copy()
    else:
        X0 = np.asarray(init_2d, float)[:, :2].copy()
    # The projection can put a whole rim row on one line, leaving the
    # Jacobian rank-deficient beyond the rigid null space; a tiny
    # deterministic jitter breaks those accidental degeneracies without
    # perturbing the converged layout.
    rng = np.random.default_rng(0)
    X0 = X0 + 1.0e-6 * rng.standard_normal(X0.shape)

    i_idx = edges[:, 0]
    j_idx = edges[:, 1]

    def residual(xflat):
        X = xflat.reshape(n_nodes, 2)
        d = X[i_idx] - X[j_idx]
        ell = np.sqrt((d * d).sum(axis=1) + eps)
        return sw * (ell - L0)

    # Sparse analytic Jacobian: each residual touches only its two end
    # nodes.  d r_e/d x_i =  sw_e * u,  d r_e/d x_j = -sw_e * u, with
    # u = (x_i - x_j)/||x_i - x_j|| the planar unit vector (4 nz/edge).
    jac_rows = np.repeat(np.arange(n_edges), 4)
    jac_cols = np.column_stack(
        [2 * i_idx, 2 * i_idx + 1, 2 * j_idx, 2 * j_idx + 1]).ravel()

    def jac(xflat):
        X = xflat.reshape(n_nodes, 2)
        d = X[i_idx] - X[j_idx]
        ell = np.sqrt((d * d).sum(axis=1) + eps)
        u = d / ell[:, None]
        vals = np.column_stack(
            [sw * u[:, 0], sw * u[:, 1], -sw * u[:, 0], -sw * u[:, 1]]).ravel()
        return coo_matrix((vals, (jac_rows, jac_cols)),
                          shape=(n_edges, 2 * n_nodes)).tocsr()

    sol = least_squares(residual, X0.ravel(), jac=jac, method='trf',
                        max_nfev=max_nfev, verbose=2 if verbose else 0)
    X = sol.x.reshape(n_nodes, 2)

    if anchor_to is not None:
        X = _procrustes_align(X, np.asarray(anchor_to, float)[:, :2],
                              allow_reflection=allow_reflection)
    return X


# ============================================================
# 4. Visualisation
# ============================================================

def _quad_face_list(elements: np.ndarray):
    return [list(map(int, q)) for q in elements]


def visualise_meshes(nodes_3d: np.ndarray,
                     nodes_2d_raw: np.ndarray,
                     nodes_2d_smooth: np.ndarray,
                     elements: np.ndarray,
                     gap: float | None = None) -> None:
    """Show the three meshes side by side.

    Tries COMPAS Viewer first (user's preferred renderer for this
    library); falls back to matplotlib if compas_viewer is not
    importable.  Each mesh is colour-coded so the three pipeline
    stages are visually distinct:

        * 3D semi-cylinder   : light blue faces
        * raw 2D projection  : pink faces
        * smoothed 2D panel  : light green faces
    """
    try:
        from compas.datastructures import Mesh
        from compas.colors import Color
        from compas_viewer import Viewer
    except ImportError:
        _visualise_matplotlib(nodes_3d, nodes_2d_raw,
                              nodes_2d_smooth, elements)
        return

    faces = _quad_face_list(elements)

    # 3D mesh -- leave where it is
    verts_3d = [list(map(float, p)) for p in nodes_3d]

    # 2D meshes -- lift to 3D (z = 0) and shift along +x so all three
    # appear side by side in the viewer.
    span_x = float(nodes_3d[:, 0].max() - nodes_3d[:, 0].min())
    if gap is None:
        gap = 0.5 * span_x + 1.0e-9
    dx_raw    = span_x + gap
    dx_smooth = 2.0 * (span_x + gap)

    verts_raw = [[float(p[0]) + dx_raw,    float(p[1]), 0.0]
                 for p in nodes_2d_raw]
    verts_sm  = [[float(p[0]) + dx_smooth, float(p[1]), 0.0]
                 for p in nodes_2d_smooth]

    mesh_3d  = Mesh.from_vertices_and_faces(verts_3d,  faces)
    mesh_raw = Mesh.from_vertices_and_faces(verts_raw, faces)
    mesh_sm  = Mesh.from_vertices_and_faces(verts_sm,  faces)

    viewer = Viewer()
    viewer.scene.add(mesh_3d,  show_faces=True, show_lines=True,
                     show_points=True,
                     facecolor=Color(0.55, 0.75, 1.00),
                     linecolor=Color(0.05, 0.05, 0.05),
                     pointcolor=Color(0.00, 0.00, 0.00))
    viewer.scene.add(mesh_raw, show_faces=True, show_lines=True,
                     show_points=True,
                     facecolor=Color(1.00, 0.65, 0.75),
                     linecolor=Color(0.05, 0.05, 0.05),
                     pointcolor=Color(0.00, 0.00, 0.00))
    viewer.scene.add(mesh_sm,  show_faces=True, show_lines=True,
                     show_points=True,
                     facecolor=Color(0.65, 0.95, 0.70),
                     linecolor=Color(0.05, 0.05, 0.05),
                     pointcolor=Color(0.00, 0.00, 0.00))
    viewer.show()


def _visualise_matplotlib(nodes_3d, nodes_2d_raw, nodes_2d_smooth,
                          elements) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.collections import PolyCollection

    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    polys3 = [[nodes_3d[v] for v in q] for q in elements]
    ax1.add_collection3d(Poly3DCollection(
        polys3, facecolors=(0.55, 0.75, 1.0, 0.85),
        edgecolors='k', linewidths=0.6))
    ax1.scatter(*nodes_3d.T, c='k', s=8)
    ax1.set_title("3D semi-cylinder")
    _equalise_3d(ax1, nodes_3d)

    ax2 = fig.add_subplot(1, 3, 2)
    polys_raw = [[nodes_2d_raw[v] for v in q] for q in elements]
    ax2.add_collection(PolyCollection(
        polys_raw, facecolors=(1.0, 0.65, 0.75, 0.85),
        edgecolors='k', linewidths=0.6))
    ax2.scatter(*nodes_2d_raw.T, c='k', s=8)
    ax2.set_aspect('equal')
    ax2.autoscale()
    ax2.set_title("Raw orthogonal projection (xy)")

    ax3 = fig.add_subplot(1, 3, 3)
    polys_sm = [[nodes_2d_smooth[v] for v in q] for q in elements]
    ax3.add_collection(PolyCollection(
        polys_sm, facecolors=(0.65, 0.95, 0.70, 0.85),
        edgecolors='k', linewidths=0.6))
    ax3.scatter(*nodes_2d_smooth.T, c='k', s=8)
    ax3.set_aspect('equal')
    ax3.autoscale()
    ax3.set_title("Laplacian-smoothed panel")

    plt.tight_layout()
    plt.show()


def _equalise_3d(ax, pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ctr = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins).max()
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)


# ============================================================
# 5. Validation
# ============================================================

def validate(nodes_3d, nodes_2d_raw, nodes_2d_smooth,
             elements, free_mask, n_circ, n_axial):
    """Cheap sanity checks; raises AssertionError on violation."""
    nx = n_axial + 1
    ny = n_circ  + 1
    expected_nodes = nx * ny
    expected_elems = n_circ * n_axial

    assert nodes_3d.shape       == (expected_nodes, 3), "wrong 3D node count"
    assert nodes_2d_raw.shape   == (expected_nodes, 2), "wrong raw 2D shape"
    assert nodes_2d_smooth.shape == (expected_nodes, 2), "wrong smooth 2D shape"
    assert elements.shape       == (expected_elems, 4), "wrong element count"

    assert np.isfinite(nodes_3d).all(),       "NaN/Inf in 3D nodes"
    assert np.isfinite(nodes_2d_raw).all(),   "NaN/Inf in raw 2D nodes"
    assert np.isfinite(nodes_2d_smooth).all(), "NaN/Inf in smooth 2D nodes"

    # Pinned-axis coordinates must be untouched by the harmonic solve.
    pinned = free_mask < 0.5
    diff = np.abs(nodes_2d_smooth - nodes_2d_raw)
    assert diff[pinned].max() < 1.0e-10, (
        f"pinned-axis coord moved during smoothing "
        f"(max {diff[pinned].max():.3e})")


# ============================================================
# 6. Main
# ============================================================

def main():
    # --- parameters --------------------------------------------------
    R       = 1.0     # cylinder radius
    L       = 2.0     # axial length
    n_circ  = 10      # elements in theta
    n_axial = 6       # elements in x

    # --- build & flatten --------------------------------------------
    nodes_3d = generate_semicylinder_mesh(R, L, n_circ, n_axial)
    elements = build_quad_connectivity(n_circ, n_axial)
    boundary_idx, interior_idx = find_boundary_nodes(n_circ, n_axial)
    free_mask = build_slide_mask(n_circ, n_axial)

    nodes_2d_raw    = project_mesh_to_xy(nodes_3d)
    # Edge-fixed Laplacian smoothing: rim outline preserved in space,
    # boundary nodes slide tangentially along their rim segment,
    # interior nodes go to the discrete-harmonic limit in one solve.
    nodes_2d_smooth = laplacian_smooth_2d(nodes_2d_raw, elements,
                                          free_mask=free_mask)

    validate(nodes_3d, nodes_2d_raw, nodes_2d_smooth,
             elements, free_mask, n_circ, n_axial)

    # --- summary -----------------------------------------------------
    disp_all     = np.linalg.norm(nodes_2d_smooth - nodes_2d_raw, axis=1)
    max_disp     = float(disp_all.max())
    max_disp_int = float(disp_all[interior_idx].max())
    max_disp_bnd = float(disp_all[boundary_idx].max())

    print("=" * 60)
    print("Semi-cylinder flattening validation")
    print("=" * 60)
    print(f"R, L                 : {R}, {L}")
    print(f"Mesh (n_axial x n_circ): {n_axial} x {n_circ}")
    print(f"Nodes                : {nodes_3d.shape[0]}")
    print(f"Elements             : {elements.shape[0]}")
    print(f"Boundary nodes       : {boundary_idx.size}")
    print(f"Interior nodes       : {interior_idx.size}")
    print("Smoothing            : edge-fixed harmonic solve (1-shot)")
    print(f"Max node disp.       : {max_disp:.6f}")
    print(f"  interior nodes     : {max_disp_int:.6f}")
    print(f"  boundary nodes     : {max_disp_bnd:.6f}")
    print()
    print("Next pipeline step (not in this script):")
    print("  -> write nodes_2d_smooth + elements to an Abaqus .inp,")
    print("     drive inflation with the Chen 2022 UMAT, compare the")
    print("     inflated shape to the target, and feed the residual to")
    print("     the Bayesian-optimiser updating the 2D panel.")

    # --- visualise ---------------------------------------------------
    visualise_meshes(nodes_3d, nodes_2d_raw, nodes_2d_smooth, elements)


if __name__ == "__main__":
    main()
