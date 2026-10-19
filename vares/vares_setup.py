"""vares_setup.py -- the gore-level set-up of a VaReS rung: pin set, element-centre targets, residual field.

    build_pin_dof_set_geom   the geometric 3-dof pin of the flat gore: the node nearest the centroid held in x
                             and y, the +x-most boundary node held in y (warp aligned with the gore axis);
                             the whole perimeter, seams included, is otherwise free (Sec. 3.4.2)
    element_centre_targets   target stress (warp, weft, shear) at the spatial centre of every element
    phys_residual_field      per-element magnitude of the weighted mismatch |sigma(eps) - sigma*| at the centre
"""
import numpy as np

from element_types import element_type_of
from vares_membrane import element_centre_physical_stress, _ip_shape_values
from vares_settings import RES_WEIGHTS


def build_pin_dof_set_geom(X_flat, boundary_ids):
    """Geometric 3-DOF pin (copy of optimise_cutting_pattern_formfound_gore's):
    node nearest the flat centroid pinned in x+y, +x-most boundary node pinned
    in y.  Whole perimeter otherwise free -- the optimiser chooses the seam."""
    xy = X_flat[:, :2]
    c = int(np.argmin(((xy - xy.mean(0)) ** 2).sum(1)))
    bnd = np.asarray(boundary_ids, int)
    b = int(bnd[int(np.argmax(xy[bnd, 0]))])
    return sorted({2 * c, 2 * c + 1, 2 * b + 1})


def element_centre_targets(X_spatial, elements, sigma_field):
    """Physical target (sx, sy, tau) at each element's centre spatial x."""
    out = np.zeros((len(elements), 3))
    for e, elem in enumerate(elements):
        etype = element_type_of(elem)
        xi_c, eta_c = etype.centre
        Nv = _ip_shape_values(etype, xi_c, eta_c)
        pos = Nv @ X_spatial[elem]
        out[e] = sigma_field(pos)
    return out


def phys_residual_field(X_mat, X_spatial, elements, sigma_field,
                        weights=RES_WEIGHTS):
    S = element_centre_physical_stress(X_mat, X_spatial, elements,
                                       linear_material=False)
    T = element_centre_targets(X_spatial, elements, sigma_field)
    w = np.asarray(weights, dtype=float)
    return np.sqrt((((S - T) ** 2) * w).sum(axis=1)), S, T
