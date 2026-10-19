"""vares_flatten.py -- the least-squares edge-length flattening that seeds a VaReS rung.

Flat node positions X that minimise  E(X) = 1/2 sum_e w_e (|x_i - x_j| - L_e)^2  over the edges of the gore mesh,
L_e the edge lengths on the target surface: the "as developable as possible" layout, Procrustes-aligned onto the
developed leaf so that every rung starts in the same frame.  Sparse-Jacobian trust-region least squares.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix


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
