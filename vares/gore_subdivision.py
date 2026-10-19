r"""gore_subdivision.py -- NESTED gore meshes by uniform (red) subdivision of a gmsh base mesh.

meshes that are subdivisions of the previous one.  Two interleaved families keep the sqrt(2) spacing:
    family A  50   -> 25   -> 12.5 mm        family B  70.7 -> 35.4 -> 17.7 mm
Each base is formfound_profile.build_gore_gmsh(h, tip_affine=(0.08, 0.006)) -- the tested tip grading, so
every level keeps size ~ h * min(1, 0.08 + 0.006 d) with h halved per level.

One level of subdivision, in the developed leaf (where gmsh meshed):
  * every triangle -> 4 (edge midpoints); coarse nodes keep their coordinates EXACTLY (nested);
  * a midpoint of a SEAM edge is put back on the seam curve y = +-hb(x) (the same boundary function as
    build_gore_gmsh: developed half-width r(s) theta, linear taper to zero over the last < 1 mm at a closed
    tip), so the boundary converges with the mesh instead of staying at the coarse chords;
    a midpoint of a RIM edge stays on the rim line (straight);
  * every new node is mapped to the surface with build_gore_gmsh's own node map
    (x -> s -> r(s), z(s); phi = clip(y / r, -theta, theta); seam nodes snapped to +-theta), never by
    averaging spatial coordinates.
seam_ids / boundary_ids follow the boundary edges.  Interior midpoints are exact leaf midpoints.
"""
import numpy as np


def _boundary_fn(prof, n_gores):
    """hb(x_leaf) -- developed half-width with build_gore_gmsh's closed-tip taper (dense table, linear interp)."""
    theta = np.pi / n_gores
    gs, s_mid = prof.geom_scale, 0.5 * (prof.s_lo + prof.s_hi)
    sb = np.linspace(prof.s_lo, prof.s_hi, 4001)
    xb = (sb - s_mid) * gs
    hb = np.maximum(prof.spl_r(sb) * gs * theta, 0.0)
    TIP_H = 1.0
    if hb[-1] < TIP_H:
        it = int(np.nonzero(hb > TIP_H)[0][-1])
        hb[it:] = hb[it] * (xb[-1] - xb[it:]) / max(xb[-1] - xb[it], 1e-9)
    if hb[0] < TIP_H:
        it = int(np.nonzero(hb > TIP_H)[0][0])
        hb[:it + 1] = hb[it] * (xb[:it + 1] - xb[0]) / max(xb[it] - xb[0], 1e-9)
    return lambda x: np.interp(x, xb, hb)


def leaf_to_spatial(prof, xl, yl, is_seam, n_gores, gore_index=0):
    """build_gore_gmsh's node map, vectorised."""
    theta = np.pi / n_gores
    gs, s_mid = prof.geom_scale, 0.5 * (prof.s_lo + prof.s_hi)
    s = np.clip(xl / gs + s_mid, prof.s_lo, prof.s_hi)
    r = np.maximum(prof.spl_r(s) * gs, 0.0)
    z = prof.spl_z(s) * gs
    phil = np.clip(yl / np.maximum(r, 1e-9), -theta, theta)
    snap = is_seam & (np.abs(yl) > 0.5 * np.maximum(r * theta, 1e-9))
    phil = np.where(snap, np.sign(yl) * theta, phil)
    phi = 2.0 * theta * gore_index + theta + phil
    return np.c_[r * np.cos(phi), r * np.sin(phi), z]


def subdivide(prof, X_leaf, elements, seam_ids, boundary_ids, n_gores, gore_index=0):
    """One level of red refinement.  Returns X_spatial, X_leaf, seam_ids, boundary_ids, elements (same
    conventions as build_gore_gmsh); nodes 0..N-1 are the coarse nodes, unchanged."""
    T = np.asarray(elements, int)
    XL = np.asarray(X_leaf, float)[:, :2]
    N = len(XL)
    seam, bnd = set(int(i) for i in seam_ids), set(int(i) for i in boundary_ids)
    E = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    uE, inv, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    on_bnd = cnt == 1                                          # boundary edge: in one triangle only
    mid = 0.5 * (XL[uE[:, 0]] + XL[uE[:, 1]])
    is_seam_e = on_bnd & np.array([a in seam and b in seam for a, b in uE])
    is_bnd_e = on_bnd & np.array([a in bnd and b in bnd for a, b in uE])
    hb = _boundary_fn(prof, n_gores)
    ys = np.sign(mid[is_seam_e, 1]) * hb(mid[is_seam_e, 0])   # midpoint back onto the seam curve
    mid[is_seam_e, 1] = ys
    m_id = N + np.arange(len(uE))
    ne = len(T)
    e01, e12, e20 = m_id[inv[:ne]], m_id[inv[ne:2 * ne]], m_id[inv[2 * ne:]]
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    Tn = np.vstack([np.c_[a, e01, e20], np.c_[e01, b, e12], np.c_[e20, e12, c], np.c_[e01, e12, e20]])
    XLn = np.r_[XL, mid]
    seam_n = sorted(seam | set(int(i) for i in m_id[is_seam_e]))
    bnd_n = sorted(bnd | set(int(i) for i in m_id[is_bnd_e]))
    is_seam_node = np.zeros(len(XLn), bool)
    is_seam_node[seam_n] = True
    Xs = leaf_to_spatial(prof, XLn[:, 0], XLn[:, 1], is_seam_node, n_gores, gore_index)
    # (nested_family overwrites rows 0..N-1 with the coarse spatial coordinates, bit for bit)
    return Xs, np.c_[XLn, np.zeros(len(XLn))], seam_n, bnd_n, [t for t in Tn]


def nested_family(prof, base_h, levels, n_gores, gore_index=0, tip_affine=(0.08, 0.006)):
    """[(h_nominal, X_spatial, X_leaf, seam_ids, boundary_ids, elements), ...] for the base gmsh mesh and
    `levels` subdivisions.  Coarse nodes keep their spatial AND leaf coordinates bit for bit."""
    Xs, Xl, seam, bnd, els = prof.build_gore_gmsh(base_h, n_gores, gore_index=gore_index, tip_affine=tip_affine)
    out = [(float(base_h), Xs, Xl, list(seam), list(bnd), list(els))]
    for k in range(levels):
        Xs_c = Xs
        Xs, Xl, seam, bnd, els = subdivide(prof, Xl, els, seam, bnd, n_gores, gore_index)
        Xs[:len(Xs_c)] = Xs_c                                  # nested: coarse nodes exactly unchanged
        out.append((base_h / 2 ** (k + 1), Xs, Xl, seam, bnd, els))
    return out


def check(mesh, prof, n_gores):
    """Quality / consistency numbers for one mesh."""
    h, Xs, Xl, seam, bnd, els = mesh
    T = np.asarray(els)
    e = Xl[T][:, :, :2]
    A = 0.5 * ((e[:, 1, 0] - e[:, 0, 0]) * (e[:, 2, 1] - e[:, 0, 1]) - (e[:, 2, 0] - e[:, 0, 0]) * (e[:, 1, 1] - e[:, 0, 1]))
    L2 = sum(((e[:, (i + 1) % 3] - e[:, i]) ** 2).sum(1) for i in range(3))
    q = 4 * np.sqrt(3) * np.abs(A) / L2
    es = Xs[T]
    As = 0.5 * np.linalg.norm(np.cross(es[:, 1] - es[:, 0], es[:, 2] - es[:, 0]), axis=1)
    hb = _boundary_fn(prof, n_gores)
    sid = np.asarray(seam)
    seam_err = np.abs(np.abs(Xl[sid, 1]) - hb(Xl[sid, 0]))
    return dict(h=h, n_nodes=len(Xl), n_el=len(T), h_eff=float(np.sqrt(4 * np.abs(A).mean() / np.sqrt(3))),
                inverted=int((np.sign(A) != np.sign(A[0])).sum()), q_mean=float(q.mean()), q_min=float(q.min()),
                spatial_area_ratio=(float((As / np.abs(A)).min()), float((As / np.abs(A)).max())),
                seam_off_curve_max_mm=float(seam_err.max()), n_seam=len(sid))
