"""
element_types.py

Element-type registry for the VaReS cutting-pattern optimiser.

The VaReS residual/tangent machinery (vares_membrane, vares_kinematics,
element_kinematics) is written generically in terms of an ElementType
descriptor that carries everything the kinematics chain needs that is
NOT shared across element shapes:

  * the number of nodes / design DOFs,
  * the parametric shape-function derivatives dN(xi, eta),
  * the integration points and weights,
  * the parametric "centre" used for the single-sample stress readout,
  * the local edge connectivity (for mesh adjacency / plotting).

Two element types are provided:

  QUAD4  -- the original 4-node bilinear quad (8 DOFs).  Its dN, IPs and
            weights are taken verbatim from the existing quad library so
            the quad path reproduces the previous results bit for bit.
  TRI3   -- a 3-node linear (constant-strain) triangle (6 DOFs).  Its
            shape-function derivatives are CONSTANT, so the covariant
            basis, metric and Green-Lagrange strain are constant over
            the element and a single centroid integration point with
            weight 1.0 is exact.  With G_1 = X_2 - X_1, G_2 = X_3 - X_1
            the existing detJ = 0.5 ||G_1 x G_2|| is exactly the triangle
            area, so no special-casing is needed downstream.

Element type is dispatched purely on node count (3 -> TRI3, 4 -> QUAD4),
which is what lets a single connectivity list mix quads and triangles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np


# ============================================================
# Bilinear quad (QUAD4): shape functions, integration points
# ============================================================

def quad_N(xi: float, eta: float) -> np.ndarray:
    """Shape functions N_1..N_4 at parametric point (xi, eta)."""
    return 0.25 * np.array([
        (1 - xi) * (1 - eta),       # N_1 -- corner (-1, -1)
        (1 + xi) * (1 - eta),       # N_2 -- corner (+1, -1)
        (1 + xi) * (1 + eta),       # N_3 -- corner (+1, +1)
        (1 - xi) * (1 + eta),       # N_4 -- corner (-1, +1)
    ])


def quad_dN(xi: float, eta: float) -> np.ndarray:
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


# Integration points of the bilinear quad: the 2x2 Gauss points plus the centre.
_GAUSS_G = 1.0 / np.sqrt(3.0)
GAUSS_PTS_5 = [(-_GAUSS_G, -_GAUSS_G), ( _GAUSS_G, -_GAUSS_G),
               ( _GAUSS_G,  _GAUSS_G), (-_GAUSS_G,  _GAUSS_G),
               ( 0.0,       0.0      )]
GAUSS_WTS_5 = [1.0, 1.0, 1.0, 1.0, 1.0]

_quad_dN = quad_dN


# ============================================================
# Covariant / contravariant metric of a surface element
# ============================================================

def covariant_metric(G_1: np.ndarray, G_2: np.ndarray) -> np.ndarray:
    """Return the 2x2 covariant metric tensor G_ij."""
    return np.array([[G_1 @ G_1, G_1 @ G_2],
                     [G_2 @ G_1, G_2 @ G_2]])


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
# TRI3 shape-function derivatives (constant)
# ============================================================
#
# Reference triangle on (xi, eta) with corners
#     N_1 = (0, 0),  N_2 = (1, 0),  N_3 = (0, 1)
# and linear shape functions
#     N_1 = 1 - xi - eta,   N_2 = xi,   N_3 = eta.
# Their parametric derivatives are independent of (xi, eta):

_TRI_DN = np.array([[-1.0, -1.0],     # node 1: dN/dxi, dN/deta
                    [ 1.0,  0.0],     # node 2
                    [ 0.0,  1.0]])    # node 3


def _tri_dN(xi: float, eta: float) -> np.ndarray:
    """Linear-triangle shape-function derivatives (3, 2).

    Constant in (xi, eta) -- the arguments are accepted only so the
    signature matches the bilinear-quad dN and the kinematics chain can
    call etype.dN(xi, eta) uniformly.
    """
    return _TRI_DN.copy()


# ============================================================
# ElementType descriptor
# ============================================================

@dataclass(frozen=True)
class ElementType:
    """Everything the generic kinematics need that varies by shape.

    Attributes
    ----------
    name      : human label ("quad4" / "tri3")
    n_nodes   : nodes per element
    ips       : tuple of (xi, eta) integration points
    wts       : matching integration weights
    centre    : single parametric point for the centre-stress readout
    edges     : local node-id pairs forming the element boundary
    _dN_fn    : callable (xi, eta) -> (n_nodes, 2) shape-fn derivatives
    """

    name: str
    n_nodes: int
    ips: Tuple[Tuple[float, float], ...]
    wts: Tuple[float, ...]
    centre: Tuple[float, float]
    edges: Tuple[Tuple[int, int], ...]
    _dN_fn: Callable[[float, float], np.ndarray]

    @property
    def n_dofs(self) -> int:
        return 2 * self.n_nodes

    def dN(self, xi: float, eta: float) -> np.ndarray:
        return self._dN_fn(xi, eta)


QUAD4 = ElementType(
    name="quad4",
    n_nodes=4,
    ips=tuple(GAUSS_PTS_5),
    wts=tuple(GAUSS_WTS_5),
    centre=(0.0, 0.0),
    edges=((0, 1), (1, 2), (2, 3), (3, 0)),
    _dN_fn=_quad_dN,
)

TRI3 = ElementType(
    name="tri3",
    n_nodes=3,
    ips=((1.0 / 3.0, 1.0 / 3.0),),   # centroid: exact for constant strain
    wts=(1.0,),                       # detJ = 0.5||G1 x G2|| = triangle area
    centre=(1.0 / 3.0, 1.0 / 3.0),
    edges=((0, 1), (1, 2), (2, 0)),
    _dN_fn=_tri_dN,
)


_BY_NODE_COUNT = {3: TRI3, 4: QUAD4}


def element_type_for(n_nodes: int) -> ElementType:
    """Dispatch an ElementType from the node count of one element.

    This is what allows a connectivity list to mix quads (4 nodes) and
    triangles (3 nodes): every assembly loop asks for the element type
    of each element individually.
    """
    try:
        return _BY_NODE_COUNT[int(n_nodes)]
    except KeyError:
        raise ValueError(
            f"No element type registered for {n_nodes} nodes; "
            f"supported: {sorted(_BY_NODE_COUNT)} "
            "(3 = linear triangle, 4 = bilinear quad).")


def element_type_of(elem: Sequence[int]) -> ElementType:
    """ElementType for one element given its node-id sequence."""
    return element_type_for(len(elem))


# ============================================================
# Generic DOF-id mapping (any node count)
# ============================================================

def elem_dof_ids(elem: Sequence[int]) -> np.ndarray:
    """Map element-local node ids (0-based) to global DOF ids in the
    layout (N0.x, N0.y, N1.x, N1.y, ...).

    Generalises bilinear_quad_lib._elem_dof_ids (which hard-codes 8
    DOFs) to elements with any number of nodes, so triangles (6 DOFs)
    and quads (8 DOFs) scatter into the same global system.
    """
    elem = np.asarray(elem)
    n = elem.shape[0]
    ids = np.empty(2 * n, dtype=int)
    for i, node in enumerate(elem):
        ids[2 * i] = 2 * int(node)
        ids[2 * i + 1] = 2 * int(node) + 1
    return ids
