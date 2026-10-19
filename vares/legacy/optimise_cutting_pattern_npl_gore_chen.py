"""
optimise_cutting_pattern_npl_gore_chen.py

VaReS cutting-pattern optimisation for ONE gore of the NPL airship envelope,
run as the EXACT hemisphere pipeline (optimise_cutting_pattern_hemisphere.py)
applied to the airship gore:

    1. generate the 3D target mesh from the NPL profile (the true lune, nodes
       on the ACTUAL circular arcs of the body of revolution) -- either the
       structured (x, phi) grid or an UNSTRUCTURED gmsh triangle mesh
       (MESH_BACKEND = "gmsh"): the gore is meshed in its developed
       equal-metric leaf (s = meridian arc length, w = developed transverse
       offset, |w| <= r*theta) with near-equilateral triangles and mapped to
       the lune by x(s), phi = w/r(x).  The leaf is a near-isometry of the
       lune, so element quality survives the map and the mesh grades cleanly
       into the narrow trimmed ends (no structured-grid slivers there),
    2. flatten it by orthogonal projection to z = 0 and smooth it with the
       edge-fixed Laplace (harmonic) solve  -- STARTING_MESH = "smoothed",
    3. optimise the flat pattern with the FULL NONLINEAR Chen (2022) law --
       the biaxial coupled polynomial branches PLUS the stress-coupled shear
       modulus G = c0 + kG*(sx+sy) -- driving the induced stress to the
       analytic membrane PRESSURE-EQUILIBRIUM field of the NPL profile.

Target frame (TARGET_FRAME)
---------------------------
    "current" (default): the target is kept as PHYSICAL values sigma*(x_IP)
        and the mismatch (sigma - sigma*).(structural tensors) is rebuilt in
        the CURRENT material frame at every residual/tangent evaluation --
        i.e. S_pre is updated at every iteration, with the exact consistent
        tangent (vares_membrane current-frame mode).  No outer
        relinearisation is needed: its fixed point IS this formulation.
    "frozen": classic VaReS -- S_pre pushed into the contravariant basis at
        the seed pattern and held, with OUTER_RELIN_MAX refreeze rounds.
        Kept for A/B comparison.

Difference from optimise_cutting_pattern_npl_gore.py
-----------------------------------------------------
That sibling keeps the inverse loop material-consistent with the linear
*Elastic, type=LAMINA forward deck (FullEnvelope_NoComp.py): it rebinds
vares_membrane.chen_stress_linear to a LAMINA Q-matrix and solves the LINEAR
path from an analytic equal-arc flat-gore seed.  THIS script is instead
material-consistent with the VUMAT/UMAT decks (vumat_chen2022.for): it runs
the true nonlinear chen_stress with the calibrated UMAT_PROPS
(linear_material=False), and takes its seed from projection + Laplace
smoothing exactly like the hemisphere optimiser.  Do NOT import the LAMINA
sibling into the same session -- it rebinds
vares_membrane.chen_stress_linear at import time, which would poison the
linear continuation warm-up used here.

Target stress (pressure equilibrium)
------------------------------------
sigma*(x) = (sigma_m(x), sigma_t(x), 0) from npl_profile.npl_membrane_stress:
N_m = p*r_t/2, N_t = p*r_t*(1 - r_t/2r_m), sigma = N/t.  Because that field
is in equilibrium with the internal pressure ON the NPL surface, the
optimised gore makes the NPL shape a genuine equilibrium of the inflated
assembly.  Spatially varying -> fed through vares_membrane.build_targets_field
(warp = meridional +x, weft = hoop +y).

Solve
-----
Default (MATERIAL-PURE): lsq_edge seed + CONTINUATION="off" -- the
as-isometric-as-possible seed carries only the ~1% curvature obstruction,
so direct Levenberg-damped Newton under the FULL Chen law converges in
~15 iterations with no material linearisation anywhere.  Alternative
(hemisphere pipeline): "smoothed" projection seed + CONTINUATION="linear"
warm-up.  See the recipe comments at STARTING_MESH / CONTINUATION for the
measured failure modes of the other combinations.  Armijo backtracking,
min-eigenvalue conditioning diagnostic, NEWTON_MIN_STEP floor exit (the
trimmed lune has a curvature-set residual floor, like the hemisphere's).

Output (chen-suffixed so the validated LAMINA results are not clobbered)
------------------------------------------------------------------------
    cutting_pattern_npl_gore_chen.npz               (same schema as sibling)
    cutting_pattern_npl_gore_chen_{initial,converged}.csv
    optimise_cutting_pattern_npl_gore_chen_result.png
    vares_npl_gore_chen_stress_fields.png

Run with the anaconda3 python (needs numpy / scipy / matplotlib):
    "%USERPROFILE%\\anaconda3\\python.exe" optimise_cutting_pattern_npl_gore_chen.py
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

import vares_membrane
from bilinear_quad_lib import apply_step_mesh
from flatten_semicylinder_validation import (
    build_tri_connectivity, build_quad_connectivity,
    laplacian_smooth_2d, least_squares_edge_flatten, project_mesh_to_xy,
)
from vares_membrane import (
    WARP, WEFT,
    build_targets_field,
    build_phys_targets_field,
    assemble_residual,
    assemble_residual_tangent,
    element_centre_physical_stress,
)
import npl_profile as npl


HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# NPL geometry / gore mesh
# ============================================================
A_REAR  = npl.A_REAR
A_NOSE  = npl.A_NOSE
B       = npl.B
N_GORES = 12

# Trim the lune where r >= R_MIN.  Lowered from 250 to 60 mm so the pattern
# is optimised nearly to the pole tips (the undesigned cap shrinks from
# ~120/85 mm axially to ~7/5 mm) -- the gmsh leaf mesher grades its element
# size toward the narrowing tips to support this.  NOT taken to r = 0: the
# 12-fold fabric junction at the pole point is a material singularity
# (adjacent gores' warp directions differ by 30 deg at one point), the
# Gauss-curvature obstruction diverges (~250x mid-body), and real envelopes
# terminate the tips with a cap piece anyway.
R_MIN   = 60.0

MESH_BACKEND   = "gmsh"      # "gmsh" (unstructured leaf) or "structured"
GMSH_MESH_SIZE = 150.0       # [gmsh] target edge length (mm) in the leaf
N_AXIAL = 34    # [structured] longitudinal elements along the gore
N_CIRC  = 10    # [structured] transverse elements (matches FullEnvelope NR)
ELEMENT_TYPE  = "tri"        # [structured] "tri" or "quad"

# Seed + continuation RECIPES (measured on this problem -- the two knobs
# are coupled, other combinations stall under the full Chen law):
#   STARTING_MESH="lsq_edge" + CONTINUATION="off"     MATERIAL-PURE default:
#       the as-isometric-as-possible seed carries only the ~1% curvature
#       obstruction, so direct Newton under the FULL Chen law converges in
#       ~15 iterations with no linearisation anywhere.
#   STARTING_MESH="smoothed" + CONTINUATION="linear"  hemisphere pipeline:
#       projection seed (~300% end strains) needs the linearised warm-up.
#   FAILING combos (kept selectable for experiments): smoothed+nonlinear /
#       smoothed+off (seed outside the Chen range -> indefinite tangents);
#       leaf+anything (the analytic leaf has ~25% taper shear at the end
#       corners, >> Chen shear strength S=8 MPa); lsq_edge+continuation
#       (morphed intermediate targets put a near-final seed in compression
#       -> wrinkling/slack branches).
STARTING_MESH = "lsq_edge"   # "lsq_edge", "smoothed", "direct", "leaf"

# Target frame: "current" = S_pre rebuilt in the current material frame at
# every evaluation (exact tangent, no relinearisation); "frozen" = classic
# VaReS freeze at the seed + outer relinearisation rounds.
TARGET_FRAME = "current"

# ============================================================
# Analysis (pressure equilibrium target; VUMAT-consistent material)
# ============================================================
PRESSURE  = 0.10    # MPa internal pressure -- MUST match
                    # validate_optimised_pattern_npl.py (guarded there)
THICKNESS = 0.9     # mm -- MUST match the validation deck (guarded there)

# Material: the full Chen 2022 law with the calibrated UMAT_PROPS, exactly as
# vares_membrane dispatches it for linear_material=False (chen_stress) and for
# the continuation warm-up (chen_stress_linear = the a1,a2,b1,b2,c0 tangent of
# the SAME law).  Nothing is rebound.

# ============================================================
# Newton / continuation knobs (hemisphere driver, mm-scale steps)
# ============================================================
NEWTON_MAX_IT   = 100
NEWTON_TOL_R    = 1.0e-7
NEWTON_MIN_STEP = 1.0e-6       # mm; below this the pattern has stopped moving
NEWTON_MAX_STEP = 500.0        # mm cap on ||dX|| (large geometry)
LAM_INIT        = 1.0e-3
LAM_MIN         = 1.0e-12
LAM_MAX         = 1.0e14
LAM_DOWN        = 1.0 / 3.0
LAM_UP          = 3.0
LS_C1           = 1.0e-4
LS_MAX_BT       = 30
VARY_DETJ       = True         # faithful reference-area-varying VaReS

# Phase-1 geometric continuation (flat -> lune homotopy over t) -- see the
# seed/continuation recipes above STARTING_MESH for which combos converge:
#   "off"       -- no continuation, direct Newton at t = 1.  The right mode
#                  for a near-isometric seed (lsq_edge): the full Chen law
#                  converges directly, nothing is ever linearised.
#   "linear"    -- classic warm-up: t-steps under the LINEARISED Chen tangent
#                  (a1,a2,b1,b2,c0), full law only at t = 1.  Needed by the
#                  far-from-isometric projection seed ("smoothed").
#   "nonlinear" -- t-steps under the FULL Chen law.  Measured to stall from
#                  BOTH seed types (far seed: strains outside the law's
#                  range; near seed: morphed targets induce compression) --
#                  kept selectable for experiments.
CONTINUATION = "off"
N_CONTINUATION_INTERMEDIATES = 3   # flat -> lune homotopy steps
OUTER_RELIN_MAX = 4                # [frozen target frame only]
OUTER_RELIN_TOL = 1.0e-3           # [frozen target frame only]

RES_WEIGHTS = (1.0, 1.0, 10.0)

SHOW_PLOTS = True
OUT_NPZ = os.path.join(HERE, "cutting_pattern_npl_gore_chen.npz")
OUT_PNG = os.path.join(HERE, "optimise_cutting_pattern_npl_gore_chen_result.png")
OUT_PNG_STRESS = os.path.join(HERE, "vares_npl_gore_chen_stress_fields.png")


# ============================================================
# Gore geometry
# ============================================================

def _body_x_range(r_min: float):
    """x stations where profile_radius == r_min (rear, nose)."""
    frac = np.sqrt(max(0.0, 1.0 - (r_min / B) ** 2))
    return -A_REAR * frac, A_NOSE * frac


def node_id(i_ax: int, j_tr: int) -> int:
    return j_tr * (N_AXIAL + 1) + i_ax


def build_connectivity():
    if ELEMENT_TYPE == "quad":
        return build_quad_connectivity(N_CIRC, N_AXIAL)
    if ELEMENT_TYPE == "tri":
        return build_tri_connectivity(N_CIRC, N_AXIAL)
    raise ValueError("ELEMENT_TYPE must be 'tri' or 'quad'")


def generate_npl_gore_mesh():
    """True NPL lune (spatial target), centred on the +z axis.

    The gore is laid out with its centreline at phi = 0 so the crown faces +z
    and the orthogonal drop-z projection gives a symmetric flat leaf -- the
    lune analogue of the hemisphere's disc projection.  (The target is a body
    of revolution, so which gore index this is only changes a rigid rotation.)

    Node ordering matches build_*_connectivity: flat index = j*(N_AXIAL+1)+i,
    i = 0..N_AXIAL longitudinal, j = 0..N_CIRC transverse (seam to seam).

    Returns
    -------
    X_spatial    : (N,3) true lune coordinates (mm)
    X_leaf       : (N,3) developed equal-metric leaf (s, w, 0) per node --
                   the near-isometric flat layout (STARTING_MESH="leaf")
    xs           : (N_AXIAL+1,) station x-coordinates
    seam_ids     : node ids on the two seams (j=0 and j=N_CIRC)
    boundary_ids : node ids on the full perimeter (seams + both end rows)
    """
    theta = np.pi / N_GORES
    x_lo, x_hi = _body_x_range(R_MIN)
    xs = np.linspace(x_lo, x_hi, N_AXIAL + 1)
    r = npl.profile_radius(xs)
    s = npl.meridian_arclength(xs)
    s = s - 0.5 * (s[0] + s[-1])                  # centre the leaf

    nn = (N_AXIAL + 1) * (N_CIRC + 1)
    X_spatial = np.zeros((nn, 3))
    X_leaf = np.zeros((nn, 3))
    seam_ids = []
    boundary_ids = set()
    for j in range(N_CIRC + 1):
        u = j / N_CIRC                            # 0..1 across the gore
        phi = -theta + u * (2.0 * theta)          # seam to seam, crown at +z
        for i in range(N_AXIAL + 1):
            nid = node_id(i, j)
            ri = r[i]
            X_spatial[nid] = (xs[i], ri * np.sin(phi), ri * np.cos(phi))
            X_leaf[nid] = (s[i], ri * phi, 0.0)   # developed arc offset
            if i == 0 or i == N_AXIAL:
                boundary_ids.add(nid)
        if j == 0 or j == N_CIRC:
            ids = [node_id(i, j) for i in range(N_AXIAL + 1)]
            seam_ids.extend(ids)
            boundary_ids.update(ids)
    return X_spatial, X_leaf, xs, sorted(seam_ids), sorted(boundary_ids)


def generate_npl_gore_mesh_gmsh(mesh_size: float):
    """Unstructured triangle mesh of the trimmed NPL lune via gmsh.

    The gore is meshed in its DEVELOPED equal-metric leaf coordinates
    (s = meridian arc length from the rear trim, centred; w = developed
    transverse offset, boundary w = +-h(s) = +-r(x(s))*theta) and each node
    is mapped onto the true lune by inverting s -> x and setting
    phi = w / r(x).  The leaf is a near-isometry of the lune (meridional
    lengths exact on the centreline, transverse developed arcs exact), so
    gmsh's near-equilateral leaf triangles stay near-equilateral ON the 3D
    surface and grade smoothly into the narrow trimmed ends -- unlike the
    structured (x, phi) grid whose end elements are ~20:1 slivers.

    Seam nodes are snapped exactly onto the seam planes phi = +-theta; the
    four boundary curves (two seam splines, two end lines) give the exact
    seam / boundary node sets.

    Returns
    -------
    X_spatial    : (N, 3) lune coordinates (mm)
    X_leaf       : (N, 3) developed equal-metric leaf (s, w, 0) per node --
                   the meshing domain itself, i.e. the near-isometric flat
                   layout (STARTING_MESH="leaf")
    xs           : boundary-sample station x-coordinates (for reporting)
    seam_ids     : node ids on the two seams
    boundary_ids : node ids on the full perimeter
    elements     : list of (3,) int arrays (compact 0-based ids)
    """
    try:
        import gmsh
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "MESH_BACKEND='gmsh' requires the gmsh python module "
            "(`pip install gmsh`).") from exc

    theta = np.pi / N_GORES
    x_lo, x_hi = _body_x_range(R_MIN)

    # Fine arclength table for the s <-> x inversion.
    xs_f = np.linspace(x_lo, x_hi, 2001)
    s_f = npl.meridian_arclength(xs_f)
    s_f = s_f - 0.5 * (s_f[0] + s_f[-1])              # centre the gore
    r_f = npl.profile_radius(xs_f)

    # Boundary samples of the leaf outline (dense: the seam splines curve
    # sharply where the leaf narrows toward the low-R_MIN tips).
    n_bnd = 161
    xb = np.linspace(x_lo, x_hi, n_bnd)
    sb = np.interp(xb, xs_f, s_f)
    hb = npl.profile_radius(xb) * theta

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)   # no FLTK / stdout spam
        gmsh.model.add("npl_gore_leaf")
        geo = gmsh.model.geo
        pts_top = [geo.addPoint(sb[i], +hb[i], 0.0, mesh_size)
                   for i in range(n_bnd)]
        pts_bot = [geo.addPoint(sb[i], -hb[i], 0.0, mesh_size)
                   for i in range(n_bnd)]
        c_top = geo.addSpline(pts_top)                 # rear -> nose, w = +h
        c_bot = geo.addSpline(pts_bot)                 # rear -> nose, w = -h
        c_nose = geo.addLine(pts_top[-1], pts_bot[-1])
        c_rear = geo.addLine(pts_bot[0], pts_top[0])
        loop = geo.addCurveLoop([c_top, c_nose, -c_bot, c_rear])
        surf = geo.addPlaneSurface([loop])
        # Free points at the leaf tip centres, used by the grading field.
        p_tip0 = geo.addPoint(sb[0], 0.0, 0.0, mesh_size)
        p_tip1 = geo.addPoint(sb[-1], 0.0, 0.0, mesh_size)
        geo.synchronize()

        # Tip-graded size field: the leaf narrows to ~2*R_MIN*theta at the
        # trimmed ends, so a uniform lc leaves a single element across
        # there.  Near the pole r ~ R_MIN + (meridian distance from the
        # trim), so lc ~ theta*(R_MIN + dist) keeps ~2 elements across the
        # local width everywhere, blending into the uniform body size.
        lc_floor = max(8.0, 0.8 * theta * R_MIN)
        dist_f = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_f, "PointsList",
                                         [p_tip0, p_tip1])
        gmsh.model.mesh.field.setNumber(dist_f, "Sampling", 100)
        math_f = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(
            math_f, "F",
            f"Max({lc_floor:.3f}, Min({mesh_size:.3f}, "
            f"{theta:.6f}*({R_MIN:.1f} + F{dist_f})))")
        gmsh.model.mesh.field.setAsBackgroundMesh(math_f)

        gmsh.option.setNumber("Mesh.MeshSizeMin", lc_floor)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.Algorithm", 6)     # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Smoothing", 10)
        gmsh.model.mesh.generate(2)

        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        ncoords = np.asarray(ncoords, dtype=float).reshape(-1, 3)
        tag2sw = {int(t): ncoords[i, :2] for i, t in enumerate(ntags)}

        etypes, _, enodes = gmsh.model.mesh.getElements(2, surf)
        tris = None
        for et, en in zip(etypes, enodes):
            if et == 2:                                # 3-node triangle
                tris = np.asarray(en, dtype=np.int64).reshape(-1, 3)
        if tris is None:
            raise RuntimeError("gmsh meshed no triangles on the gore leaf")

        def _curve_nodes(tag):
            nt, _, _ = gmsh.model.mesh.getNodes(1, tag, includeBoundary=True)
            return {int(t) for t in nt}

        seam_tags = _curve_nodes(c_top) | _curve_nodes(c_bot)
        bnd_tags = (seam_tags | _curve_nodes(c_nose) | _curve_nodes(c_rear))

        used = np.unique(tris)
        remap = {int(t): i for i, t in enumerate(used)}
        elements = [np.array([remap[int(t)] for t in row], dtype=int)
                    for row in tris]

        X_spatial = np.zeros((len(used), 3))
        X_leaf = np.zeros((len(used), 3))
        for t in used:
            t = int(t)
            s, w = tag2sw[t]
            x = float(np.interp(s, s_f, xs_f))
            r = float(np.interp(x, xs_f, r_f))
            if t in seam_tags and abs(w) > 0.5 * r * theta:
                phi = np.sign(w) * theta               # snap seams exactly
            else:
                phi = float(np.clip(w / r, -theta, theta))
            X_spatial[remap[t]] = (x, r * np.sin(phi), r * np.cos(phi))
            X_leaf[remap[t]] = (s, w, 0.0)             # the meshing domain

        seam_ids = sorted(remap[t] for t in seam_tags if t in remap)
        boundary_ids = sorted(remap[t] for t in bnd_tags if t in remap)
    finally:
        gmsh.finalize()
    return X_spatial, X_leaf, xb, seam_ids, boundary_ids, elements


# ============================================================
# Starting flat pattern: project to z=0, then Laplace-smooth
# ============================================================

def build_starting_pattern(X_spatial, X_leaf, elements, boundary_ids):
    """Flatten the lune exactly as the hemisphere optimiser flattens the dome.

    "smoothed" (default): orthogonal projection to z = 0, then pin the full
    perimeter to its projected position and harmonically relax the interior
    (edge-fixed Laplace solve).  "direct" skips the smoothing; "lsq_edge" is
    the as-developable-as-possible layout for comparison.

    "leaf": the developed equal-metric layout (s = meridian arc length,
    w = developed transverse arc offset) -- a NEAR-ISOMETRY of the lune, so
    the seed strain is ~1% (just the Gauss-curvature obstruction) instead of
    the projection's ~300% end foreshortening.  This is the seed that lets
    CONTINUATION="nonlinear" (or even "off") converge under the FULL Chen
    law with no material linearisation anywhere.
    """
    if STARTING_MESH == "leaf":
        return X_leaf.copy()

    nodes_2d_raw = project_mesh_to_xy(X_spatial)

    if STARTING_MESH == "smoothed":
        free_mask = np.ones((X_spatial.shape[0], 2))
        free_mask[boundary_ids, :] = 0.0
        nodes_2d = laplacian_smooth_2d(nodes_2d_raw, elements,
                                       free_mask=free_mask)
    elif STARTING_MESH == "direct":
        nodes_2d = nodes_2d_raw
    elif STARTING_MESH == "lsq_edge":
        tri_graph = (build_tri_connectivity(N_CIRC, N_AXIAL)
                     if ELEMENT_TYPE == "quad" else elements)
        nodes_2d = least_squares_edge_flatten(
            X_spatial, tri_graph, init_2d=nodes_2d_raw,
            anchor_to=nodes_2d_raw)
    else:
        raise ValueError(
            "STARTING_MESH must be 'smoothed', 'direct', 'lsq_edge' "
            "or 'leaf'")

    X_initial = np.zeros((nodes_2d.shape[0], 3))
    X_initial[:, :2] = nodes_2d
    return X_initial


# ============================================================
# Target stress field (membrane equilibrium of the NPL profile)
# ============================================================

def sigma_field(pos):
    """Physical target (sx_warp=sig_m, sy_weft=sig_t, tau=0) at a spatial IP."""
    sig_m, sig_t = npl.npl_membrane_stress(pos[0], PRESSURE, THICKNESS)
    return (sig_m, sig_t, 0.0)


# ============================================================
# Pin set (free edge: 3 DOFs to remove rigid-body motion)
# ============================================================

def build_pin_dof_set(X_spatial, boundary_ids):
    """Hemisphere-style 3-DOF pin set.

    The crown node (max z -- mid-gore centreline near x = 0) is pinned in x
    and y to kill the two in-plane translations; the boundary node furthest
    along +x (nose end) is pinned in y to kill the in-plane spin.  The whole
    perimeter is otherwise FREE -- the optimiser chooses the seam curve.
    """
    crown = int(np.argmax(X_spatial[:, 2]))
    bnd = np.asarray(boundary_ids, dtype=int)
    rb = int(bnd[int(np.argmax(X_spatial[bnd, 0]))])
    ids = {2 * crown, 2 * crown + 1, 2 * rb + 1}
    return sorted(ids), crown


# ============================================================
# Levenberg-damped Newton (one continuation step) -- hemisphere driver
# ============================================================

def run_inner_loop(X_start, X_spatial_t, elements, free_dofs,
                   S_pre_all, detJ_ref_all, step_label,
                   *, linear_material=False, sigma_star_all=None,
                   verbose=True):
    """Solve R(X) = 0 for one continuation step.  Returns
    (X_final, R_history, n_indef).

    With ``sigma_star_all`` given the mismatch is rebuilt in the current
    material frame at every assembly (S_pre updated at every evaluation);
    S_pre_all is then unused (may be None).
    """
    X = X_start.copy()
    free = np.asarray(free_dofs)
    free_ix = np.ix_(free, free)
    N_g = 2 * X.shape[0]
    lam = LAM_INIT
    R_history = []
    n_indef = 0
    stall = 0

    if verbose:
        print(f"  {'iter':>4s}  {'||R||':>12s}  {'|dX|':>10s}  "
              f"{'alpha':>6s}  {'lambda':>10s}  {'minEig(Ks)':>12s}  flag")
        print("  " + "-" * 78)

    for k in range(NEWTON_MAX_IT):
        R, K = assemble_residual_tangent(
            X, X_spatial_t, elements, S_pre_all, detJ_ref_all,
            linear_material=linear_material, vary_detJ=VARY_DETJ,
            sigma_star_all=sigma_star_all)
        R_free = R[free]
        K_free = K[free_ix]
        rn = float(np.linalg.norm(R_free))
        R_history.append(rn)

        # Conditioning diagnostic on the symmetric part.
        Ks = 0.5 * (K_free + K_free.T)
        min_eig = float(np.linalg.eigvalsh(Ks).min())
        if min_eig < 0.0:
            n_indef += 1

        if rn < NEWTON_TOL_R:
            if verbose:
                print(f"  {k:>4d}  {rn:12.4e}  {'-':>10s}  {'-':>6s}  "
                      f"{'-':>10s}  {min_eig:12.3e}  converged")
            break

        accepted = False
        alpha = 1.0
        while True:
            try:
                dX_free = np.linalg.solve(
                    K_free + lam * np.eye(K_free.shape[0]), -R_free)
            except np.linalg.LinAlgError:
                lam = min(lam * LAM_UP, LAM_MAX)
                if lam >= LAM_MAX:
                    break
                continue

            dX = np.zeros(N_g)
            dX[free] = dX_free
            step = float(np.linalg.norm(dX))
            if step > NEWTON_MAX_STEP:
                dX *= NEWTON_MAX_STEP / step

            alpha = 1.0
            ls_ok = False
            for _ in range(LS_MAX_BT):
                X_trial = apply_step_mesh(X, dX, alpha)
                try:
                    R_tr = assemble_residual(
                        X_trial, X_spatial_t, elements,
                        S_pre_all, detJ_ref_all,
                        linear_material=linear_material, vary_detJ=VARY_DETJ,
                        sigma_star_all=sigma_star_all)
                except (ValueError, np.linalg.LinAlgError):
                    alpha *= 0.5
                    continue
                if np.linalg.norm(R_tr[free]) <= (1.0 - LS_C1 * alpha) * rn:
                    ls_ok = True
                    break
                alpha *= 0.5

            if ls_ok:
                X = X_trial
                rn_new = float(np.linalg.norm(R_tr[free]))
                if alpha >= 1.0:
                    lam = max(lam * LAM_DOWN, LAM_MIN)
                else:
                    lam = min(lam * LAM_UP, LAM_MAX)
                stall = stall + 1 if (rn - rn_new) <= 1.0e-4 * rn else 0
                accepted = True
                break
            else:
                lam = min(lam * LAM_UP, LAM_MAX)
                if lam >= LAM_MAX:
                    break

        eff_step = alpha * float(np.linalg.norm(dX)) if accepted else 0.0
        if verbose:
            flag = "ok" if accepted else "STALL"
            print(f"  {k:>4d}  {rn:12.4e}  {eff_step:10.3e}  {alpha:6.3f}  "
                  f"{lam:10.3e}  {min_eig:12.3e}  {flag}")
        if not accepted:
            if verbose:
                print(f"  no descent (lambda hit {LAM_MAX:.0e}) -- exiting "
                      f"{step_label}")
            break
        if eff_step < NEWTON_MIN_STEP:
            if verbose:
                print(f"  step floor (|dX|<{NEWTON_MIN_STEP:.0e} mm) -- "
                      f"exiting {step_label}")
            break
        if stall >= 4:
            if verbose:
                print(f"  converged to a residual floor (||R|| stalled) "
                      f"-- exiting {step_label}")
            break
    else:
        if verbose:
            print(f"  reached NEWTON_MAX_IT = {NEWTON_MAX_IT} "
                  f"without ||R|| < {NEWTON_TOL_R}")

    return X, R_history, n_indef


# ============================================================
# Per-element target readout (spatially varying)
# ============================================================

def element_centre_targets(X_spatial, elements):
    """Physical target (sx, sy, tau) at each element's centre spatial x."""
    from element_types import element_type_of
    out = np.zeros((len(elements), 3))
    for e, elem in enumerate(elements):
        etype = element_type_of(elem)
        xi_c, eta_c = etype.centre
        Nv = vares_membrane._ip_shape_values(etype, xi_c, eta_c)
        pos = Nv @ X_spatial[elem]
        out[e] = sigma_field(pos)
    return out


def phys_residual_field(X_mat, X_spatial, elements, weights=RES_WEIGHTS):
    """sqrt(sum_k w_k (sigma_k - target_k(x))^2) at each element centre,
    with the FULL nonlinear Chen stress."""
    S = element_centre_physical_stress(X_mat, X_spatial, elements,
                                       linear_material=False)
    T = element_centre_targets(X_spatial, elements)
    w = np.asarray(weights, dtype=float)
    return np.sqrt((((S - T) ** 2) * w).sum(axis=1)), S, T


# ============================================================
# Driver
# ============================================================

def run_optimisation(*, verbose=True):
    if MESH_BACKEND == "structured":
        X_spatial, X_leaf, xs, seam_ids, boundary_ids = \
            generate_npl_gore_mesh()
        elements = build_connectivity()
        mesh_desc = f"structured {N_AXIAL} x {N_CIRC} {ELEMENT_TYPE}"
    elif MESH_BACKEND == "gmsh":
        X_spatial, X_leaf, xs, seam_ids, boundary_ids, elements = \
            generate_npl_gore_mesh_gmsh(GMSH_MESH_SIZE)
        mesh_desc = f"gmsh unstructured leaf (size {GMSH_MESH_SIZE:.0f} mm)"
    else:
        raise ValueError("MESH_BACKEND must be 'structured' or 'gmsh'")
    X_initial = build_starting_pattern(X_spatial, X_leaf, elements,
                                       boundary_ids)

    N_nodes = X_initial.shape[0]
    N_DOFS = 2 * N_nodes
    pinned, crown = build_pin_dof_set(X_spatial, boundary_ids)
    free_dofs = [a for a in range(N_DOFS) if a not in pinned]

    # Spatially varying pressure-equilibrium target.
    if TARGET_FRAME == "current":
        # Physical values per IP; the mismatch tensor is rebuilt in the
        # CURRENT material frame at every assembly (exact tangent).
        sigma_star_all, detJ_ref_all = build_phys_targets_field(
            X_initial, X_spatial, elements, sigma_field)
        S_pre_all = None
    elif TARGET_FRAME == "frozen":
        # Classic VaReS: contravariant target frozen at the seed pattern.
        S_pre_all, detJ_ref_all = build_targets_field(
            X_initial, X_spatial, elements, sigma_field, WARP, WEFT)
        sigma_star_all = None
    else:
        raise ValueError("TARGET_FRAME must be 'current' or 'frozen'")

    if verbose:
        sm0, st0 = npl.npl_membrane_stress(0.0, PRESSURE, THICKNESS)
        print("=" * 70)
        print("VaReS cutting-pattern optimiser -- NPL gore, full Chen law")
        print("=" * 70)
        print(f"Gore mesh   : {mesh_desc} ({len(elements)} elems, "
              f"{N_nodes} nodes, {len(free_dofs)} free DOFs)")
        print(f"Body x-range: [{xs[0]:.1f}, {xs[-1]:.1f}] mm  "
              f"(r >= {R_MIN} mm; r_max = {B})")
        seed_desc = {
            "smoothed": "project to z=0 + edge-fixed Laplace smooth",
            "direct": "raw z=0 projection",
            "lsq_edge": "as-developable-as-possible flatten",
            "leaf": "developed equal-metric leaf (near-isometric, ~1% seed "
                    "strain)",
        }.get(STARTING_MESH, "?")
        print(f"Start       : {STARTING_MESH} ({seed_desc})")
        print(f"Material    : Chen 2022 nonlinear (biaxial coupled + "
              f"G = c0 + kG*(sx+sy)), UMAT_PROPS")
        print(f"Target      : NPL membrane equilibrium (p={PRESSURE} MPa, "
              f"t={THICKNESS} mm); e.g. sig_m,sig_t(x=0) = "
              f"{sm0:.2f}, {st0:.2f} MPa")
        frame_desc = ("S_pre rebuilt every evaluation, no relinearisation"
                      if TARGET_FRAME == "current"
                      else "frozen at seed + outer relinearisation")
        print(f"Target frame: {TARGET_FRAME} ({frame_desc})")
        print(f"Continuation: {CONTINUATION} "
              f"({N_CONTINUATION_INTERMEDIATES} intermediates)")
        print(f"vary_detJ   : {VARY_DETJ}")
        R0 = assemble_residual(X_initial, X_spatial, elements,
                               S_pre_all, detJ_ref_all, linear_material=True,
                               sigma_star_all=sigma_star_all)
        print(f"Initial ||R|| (linear) : "
              f"{np.linalg.norm(R0[free_dofs]):.4e}")
        print()

    t_values = np.linspace(0.0, 1.0, N_CONTINUATION_INTERMEDIATES + 2)[1:]
    X = X_initial.copy()
    R_history = []
    step_boundaries = [0]

    def _do_step(t, linear, tag, S_pre, detJ_ref):
        nonlocal X
        X_sp_t = X_initial + t * (X_spatial - X_initial)
        if verbose:
            print("=" * 70)
            print(f"{tag}  t = {t:.4f}  "
                  f"({'linear' if linear else 'nonlinear'})")
            print("=" * 70)
        X, R_hist, n_indef = run_inner_loop(
            X, X_sp_t, elements, free_dofs, S_pre, detJ_ref, tag,
            linear_material=linear, sigma_star_all=sigma_star_all,
            verbose=verbose)
        R_history.extend(R_hist)
        step_boundaries.append(len(R_history))
        if verbose and R_hist:
            print(f"  -> {len(R_hist)} iters, ||R||_end = {R_hist[-1]:.4e}, "
                  f"indef on {n_indef} iters\n")

    # Phase 1: geometric continuation flat -> lune, either under the FULL
    # Chen law ("nonlinear": the material model is never linearised) or the
    # linearised warm-up tangent ("linear"); "off" skips straight to t = 1.
    if CONTINUATION not in ("nonlinear", "linear", "off"):
        raise ValueError("CONTINUATION must be 'nonlinear', 'linear' or 'off'")
    if CONTINUATION != "off":
        lin = (CONTINUATION == "linear")
        for i, t in enumerate(t_values, 1):
            _do_step(t, lin, f"P1.{i} ({CONTINUATION})",
                     S_pre_all, detJ_ref_all)

    if TARGET_FRAME == "current":
        # Phase 2: FULL nonlinear Chen law at the lune (t = 1).  The target
        # frame is always the current one, so self-consistency is built in
        # and no outer relinearisation is needed.
        _do_step(1.0, False, "P2 (nonlinear, t=1, current-frame)",
                 S_pre_all, detJ_ref_all)
        _, S_now, T_now = phys_residual_field(X, X_spatial, elements)
        tref = max(float(np.abs(T_now).max()), 1.0)
        err = float(np.abs(S_now - T_now).max() / tref)
        if verbose:
            print(f"  final max stress mismatch {err:.3e} "
                  f"(physical floor; no frame offset by construction)\n")
    else:
        # Phase 2 + outer relinearisation: FULL nonlinear Chen law at the
        # lune (t = 1), rebuilding the frozen target at the converged
        # pattern each round until self-consistent.
        for outer in range(OUTER_RELIN_MAX):
            _do_step(1.0, False, f"P2.{outer} (nonlinear, t=1)",
                     S_pre_all, detJ_ref_all)
            _, S_now, T_now = phys_residual_field(X, X_spatial, elements)
            tref = max(float(np.abs(T_now).max()), 1.0)
            err = float(np.abs(S_now - T_now).max() / tref)
            if verbose:
                print(f"  relinearisation {outer}: max stress mismatch "
                      f"{err:.3e} (tol {OUTER_RELIN_TOL:.0e})\n")
            if err < OUTER_RELIN_TOL:
                break
            S_pre_all, detJ_ref_all = build_targets_field(
                X, X_spatial, elements, sigma_field, WARP, WEFT)

    res_init, S_init, T = phys_residual_field(X_initial, X_spatial, elements)
    res_conv, S_conv, _ = phys_residual_field(X, X_spatial, elements)
    if verbose:
        print("Stress-mismatch residual (weighted, MPa, nonlinear Chen):")
        print(f"  initial (smoothed projection) : max {res_init.max():.4f}  "
              f"mean {res_init.mean():.4f}")
        print(f"  converged (optimised)         : max {res_conv.max():.4f}  "
              f"mean {res_conv.mean():.4f}")
        print(f"  improvement factor (mean)     : "
              f"{res_init.mean() / max(res_conv.mean(), 1e-12):.1f}x")
        print(f"  converged sx in [{S_conv[:,0].min():+.2f}, "
              f"{S_conv[:,0].max():+.2f}]  target sig_m in "
              f"[{T[:,0].min():.2f}, {T[:,0].max():.2f}] MPa")
        print(f"  converged sy in [{S_conv[:,1].min():+.2f}, "
              f"{S_conv[:,1].max():+.2f}]  target sig_t in "
              f"[{T[:,1].min():.2f}, {T[:,1].max():.2f}] MPa")
        print(f"  converged tau in [{S_conv[:,2].min():+.3f}, "
              f"{S_conv[:,2].max():+.3f}]  target 0 MPa")

    return {
        "X_initial": X_initial, "X_converged": X, "X_spatial": X_spatial,
        "elements": np.asarray(elements), "xs": xs, "seam_ids": seam_ids,
        "boundary_ids": boundary_ids,
        "R_history": R_history, "step_boundaries": step_boundaries,
        "res_init": res_init, "res_conv": res_conv,
        "S_init": S_init, "S_conv": S_conv, "T": T,
    }


# ============================================================
# Output
# ============================================================

def save_results(res):
    np.savez(
        OUT_NPZ,
        X_initial=res["X_initial"], X_converged=res["X_converged"],
        X_spatial=res["X_spatial"], elements=res["elements"],
        xs=res["xs"], seam_ids=np.asarray(res["seam_ids"]),
        boundary_ids=np.asarray(res["boundary_ids"]),
        R_history=np.asarray(res["R_history"], dtype=float),
        step_boundaries=np.asarray(res["step_boundaries"], dtype=int),
        res_init=res["res_init"], res_conv=res["res_conv"],
        S_init=res["S_init"], S_conv=res["S_conv"], T=res["T"],
        n_axial=N_AXIAL, n_circ=N_CIRC, n_gores=N_GORES,
        r_min=R_MIN, pressure=PRESSURE, thickness=THICKNESS,
        a_rear=A_REAR, a_nose=A_NOSE, b=B,
        element_type=ELEMENT_TYPE, starting_mesh=STARTING_MESH,
        material="chen2022_nonlinear",
        mesh_backend=MESH_BACKEND, gmsh_mesh_size=GMSH_MESH_SIZE,
        target_frame=TARGET_FRAME,
    )
    print(f"Saved: {OUT_NPZ}")

    seam_set = set(res["seam_ids"])
    for tag, key in (("initial", "X_initial"), ("converged", "X_converged")):
        path = os.path.join(HERE, f"cutting_pattern_npl_gore_chen_{tag}.csv")
        Xf = res[key]
        Xs = res["X_spatial"]
        with open(path, "w") as f:
            f.write(f"# meta: pressure={PRESSURE} thickness={THICKNESS} "
                    f"n_gores={N_GORES} r_min={R_MIN} "
                    f"material=chen2022_nonlinear start={STARTING_MESH} "
                    f"mesh={MESH_BACKEND} target_frame={TARGET_FRAME}\n")
            f.write("# node_id, flat_x, flat_y, flat_z, tgt_x, tgt_y, tgt_z, seam\n")
            for nid in range(Xf.shape[0]):
                fx, fy, fz = Xf[nid]
                tx, ty, tz = Xs[nid]
                f.write(f"{nid}, {fx:.6f}, {fy:.6f}, {fz:.6f}, "
                        f"{tx:.6f}, {ty:.6f}, {tz:.6f}, "
                        f"{1 if nid in seam_set else 0}\n")
        print(f"Saved: {path}")


# Stress-distribution plots: same convention as the hemisphere optimiser.
STRESS_SHARED_SCALE = False


def _lune_aspect(Xs):
    """Box aspect proportional to the target lune's data extents, floored at
    1/3 so the long thin lune stays legible (hemisphere-plot convention)."""
    ext = Xs.max(axis=0) - Xs.min(axis=0)
    ext = np.maximum(ext, 1e-9)
    return tuple(np.maximum(ext / ext.max(), 1.0 / 3.0))


def make_plots(res):
    """Hemisphere-optimiser format: 2x2 -- (1) 2D cutting pattern initial vs
    converged, (2) ||R|| convergence, (3, 4) 3D residual fields on the TRUE
    NPL target lune (initial vs converged, shared viridis scale)."""
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.cm as cm

    Xi = res["X_initial"]; Xc = res["X_converged"]; Xs = res["X_spatial"]
    elements = res["elements"]; R_hist = res["R_history"]
    sb = res["step_boundaries"]

    fig = plt.figure(figsize=(16, 11))

    # (0,0) 2D initial vs converged cutting pattern
    ax = fig.add_subplot(2, 2, 1)
    polys_i = [[(Xi[n, 0], Xi[n, 1]) for n in q] for q in elements]
    polys_c = [[(Xc[n, 0], Xc[n, 1]) for n in q] for q in elements]
    ax.add_collection(PolyCollection(polys_i, facecolor='none',
                                     edgecolor='tab:blue', lw=1.0, alpha=0.6))
    ax.add_collection(PolyCollection(polys_c, facecolor='none',
                                     edgecolor='tab:red', lw=1.3, alpha=0.9))
    allxy = np.vstack([Xi[:, :2], Xc[:, :2]])
    pad = 0.05 * (allxy.max() - allxy.min())
    ax.set_xlim(allxy[:, 0].min() - pad, allxy[:, 0].max() + pad)
    ax.set_ylim(allxy[:, 1].min() - pad, allxy[:, 1].max() + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Gore cutting pattern (z=0): initial (blue) vs converged (red)')
    ax.legend(handles=[Line2D([0], [0], color='tab:blue', label='initial'),
                       Line2D([0], [0], color='tab:red', label='converged')],
              loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) ||R|| convergence
    ax = fig.add_subplot(2, 2, 2)
    R_hist = np.asarray(R_hist, dtype=float)
    if R_hist.size:
        ax.semilogy(np.arange(R_hist.size), np.maximum(R_hist, 1e-30),
                    'o-', color='tab:green', lw=2, ms=4)
        for b in np.asarray(sb)[1:-1]:
            ax.axvline(b, color='gray', ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel('iteration'); ax.set_ylabel(r'$\|R\|$ (log)')
    ax.set_title(f'VaReS residual convergence ({R_hist.size} iters)')
    ax.grid(True, which='both', alpha=0.3)

    # (1,*) 3D residual fields on the TARGET lune
    r_i = res["res_init"]; r_c = res["res_conv"]
    vmax = float(max(r_i.max(), r_c.max(), 1e-12))
    cmap = cm.viridis
    aspect = _lune_aspect(Xs)

    def _plot3d(ax, scalar, title):
        polys = [[tuple(Xs[n]) for n in q] for q in elements]
        pc = Poly3DCollection(polys, edgecolor='k', lw=0.2, alpha=0.95)
        pc.set_facecolor(cmap(scalar / vmax))
        ax.add_collection3d(pc)
        ax.set_xlim(Xs[:, 0].min(), Xs[:, 0].max())
        ax.set_ylim(Xs[:, 1].min(), Xs[:, 1].max())
        ax.set_zlim(Xs[:, 2].min(), Xs[:, 2].max())
        ax.set_title(title); ax.set_box_aspect(aspect)

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    _plot3d(ax, r_i, f'INITIAL residual (max {r_i.max():.3e})')
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    _plot3d(ax, r_c, f'CONVERGED residual (max {r_c.max():.3e})')

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=fig.get_axes()[2:], shrink=0.7, location='right',
                 label=r'$\sqrt{\sum_k w_k (\sigma_k-\sigma^*_k)^2}$ (MPa)')
    fig.suptitle("VaReS cutting-pattern optimisation -- NPL gore, "
                 "full nonlinear Chen law "
                 "(target: NPL membrane-equilibrium stress field)",
                 fontsize=13)
    fig.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    print(f"Saved: {OUT_PNG}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def make_stress_plots(res):
    """Physical stress distributions (sx, sy, tau) on the TARGET NPL lune:
    INITIAL pattern (top row) vs CONVERGED pattern (bottom row) -- the same
    figure the hemisphere optimiser produces, evaluated with the full
    nonlinear Chen law.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    Xs = res["X_spatial"]
    elements = res["elements"]
    S_init = res["S_init"]
    S_conv = res["S_conv"]
    T = res["T"]

    polys = [[tuple(Xs[n]) for n in q] for q in elements]
    aspect = _lune_aspect(Xs)
    cmap = cm.RdBu_r
    comps = [(r'warp  $\sigma_x$', 0),
             (r'weft  $\sigma_y$', 1),
             (r'shear  $\tau$',    2)]

    def _sym(*arrays):
        v = max(float(np.abs(a).max()) for a in arrays)
        return Normalize(-max(v, 1e-9), max(v, 1e-9))

    def _panel(ax, vals, norm, title):
        pc = Poly3DCollection(polys, edgecolor='k', lw=0.15, alpha=0.97)
        pc.set_facecolor(cmap(norm(vals)))
        ax.add_collection3d(pc)
        ax.set_xlim(Xs[:, 0].min(), Xs[:, 0].max())
        ax.set_ylim(Xs[:, 1].min(), Xs[:, 1].max())
        ax.set_zlim(Xs[:, 2].min(), Xs[:, 2].max())
        ax.set_box_aspect(aspect); ax.set_title(title, fontsize=9)

    fig = plt.figure(figsize=(15, 9))
    for c, (label, idx) in enumerate(comps):
        si, sc = S_init[:, idx], S_conv[:, idx]
        if STRESS_SHARED_SCALE:
            norm_i = norm_c = _sym(si, sc)
        else:
            norm_i, norm_c = _sym(si), _sym(sc)

        ax_i = fig.add_subplot(2, 3, c + 1, projection='3d')
        ax_c = fig.add_subplot(2, 3, c + 4, projection='3d')
        _panel(ax_i, si, norm_i,
               f'INITIAL  {label}\n[{si.min():+.2f}, {si.max():+.2f}] MPa')
        _panel(ax_c, sc, norm_c,
               f'CONVERGED  {label}\n[{sc.min():+.2f}, {sc.max():+.2f}] MPa')

        if STRESS_SHARED_SCALE:
            sm = cm.ScalarMappable(cmap=cmap, norm=norm_i); sm.set_array([])
            fig.colorbar(sm, ax=[ax_i, ax_c], shrink=0.6, location='bottom',
                         pad=0.02, label=f'{label}  (MPa)')
        else:
            for ax, nrm in ((ax_i, norm_i), (ax_c, norm_c)):
                sm = cm.ScalarMappable(cmap=cmap, norm=nrm); sm.set_array([])
                fig.colorbar(sm, ax=ax, shrink=0.6, location='bottom', pad=0.02)

    note = ("init/converged share each component's scale"
            if STRESS_SHARED_SCALE else "every panel auto-scaled")
    fig.suptitle(
        "Physical stress on the NPL target lune -- full nonlinear Chen law  "
        f"(target field: $\\sigma_m$ [{T[:,0].min():.1f}, {T[:,0].max():.1f}], "
        f"$\\sigma_t$ [{T[:,1].min():.1f}, {T[:,1].max():.1f}], "
        f"$\\tau$ 0 MPa; {note})", fontsize=11)
    fig.savefig(OUT_PNG_STRESS, dpi=120, bbox_inches='tight')
    print(f"Saved: {OUT_PNG_STRESS}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def main():
    res = run_optimisation(verbose=True)
    save_results(res)
    make_plots(res)
    make_stress_plots(res)


if __name__ == "__main__":
    main()