r"""
ParaboloidDish_unstructured_JAXFDM.py -- unstructured-mesh form finder of the paraboloid dish (developed from an earlier structured-mesh solver, not included)
for the unstructured rungs of the paraboloid mesh study (MeshStudy_paraboloid_unstructured.py).

WHAT DIFFERS FROM THE POLAR ORIGINAL (everything else -- constant-mass coupling, GAS_K calibration,
Newton/SM adjoint, Sobolev L-BFGS, health gates, objective -- is the original code, unchanged):

  * Topology is READ from JAXFDM_MESH_NPZ (paraboloid_unstructured_mesh.py): a gmsh triangulation
    with every node on the target paraboloid, rim nodes pinned, one node at the apex.  The nodal
    targets (target_rho, target_z) are the seed positions, exactly as the polar builder sets them.
  * There are no rings, so "one q per hoop ring + one per meridional level" has no meaning.  Its
    replacement keeps the same design dimension and the same Sobolev structure:
        q_e = q_m(s_e) cos^2(a_e) + q_h(s_e) sin^2(a_e)
    q_m(s), q_h(s): piecewise-linear profiles on JAXFDM_N_STATIONS stations equally spaced in
    meridian arc length s (apex -> rim);  s_e: arc length at the edge midpoint;  a_e: angle between
    the edge and the meridian on the SEED surface.  (For a triangle net this is q_e = t^T M t with
    M = diag(q_m, q_h) in the local (meridional, hoop) frame -- the general form that lets a
    triangulated net carry an anisotropic membrane stress.)  Both are frozen on the seed, so q(theta)
    is a fixed linear map of the station values and the gradient flows through it exactly.
    Design variables are "mer@k" / "hoop@k", so the original H1 Sobolev preconditioner applies as is.
  * JAXFDM_OBJECTIVE=normal replaces the nodal (rho, z)-to-seed fit by the normal distance to the
    paraboloid.  On an irregular net smooth q leaves an in-plane nodal imbalance, the nodes slide
    along the surface, and a nodal objective then drives q into a station-to-station zig-zag to
    stop sliding that is not shape error (measured 2026-10-01: raw N_theta 15-35% vs 2-4% polar).
    Default stays "nodal" so the file reproduces the polar formulation when asked to.
  * JAXFDM_W_QUAL > 0 adds a smooth element-quality energy (area vs seed + shape ratio; see W_QUAL).
  * JAXFDM_W_MESH > 0 adds the BWB element-quality penalty (area / shape-ratio caps; see W_MESH).
  * The per-ring symmetry projectors (solver sweep + accepted state) are OFF: they need rings, and an
    unstructured net is not C_n symmetric anyway.  Nothing replaces them.

ORIGINAL HEADER FOLLOWS
-----------------------
Paraboloid-dish fitting of a pressure-inflated circular membrane with jax_fdm
(arpastrana/jax_fdm), using TRUE automatic differentiation (no finite differences).

PROBLEM
-------
A flat CIRCULAR boundary of radius R is pinned in the plane z = 0. A membrane
spanning the disk is inflated by a constant-mass follower pressure p = sign*GAS_K/V(x)
(ideal-gas coupling, dp/dV = -K/V^2). Left to itself a pinned inflated membrane bulges
into a spherical-cap-like dome; here we OPTIMISE the per-ring force densities so the
equilibrium surface instead matches a target PARABOLOID of revolution --

        z_target(r) = DEPTH * (1 - (r/R)^2)          (apex at the centre, rim at z=0)

i.e. an inflatable parabolic reflector / dish antenna (the concave underside is the
signalling surface; focal length f = R^2 / (4*DEPTH)).

SOLUTION STRATEGY  (identical stack to AirfoilWing_pressure_JAXFDM.py)
---------------------------------------------------------------------
  * Topology: a fan of concentric rings around a single apex; the OUTER ring is the
    fixed circular boundary. Node 0 = apex; ring i in [1, n_rings]; ring n_rings pinned.
  * Constant mass INSIDE jax: ConstantMassModelSparse computes p = sign*GAS_K/V(x) inside
    the differentiated fixed point, so ONE value_and_grad returns the EXACT constant-mass
    gradient (dp/dV feedback included).
  * AXISYM_Q: the ~n_edges per-edge q are reduced to one value per hoop RING + one per
    meridional LEVEL, so azimuthal q variation (the k>=1 lobed instabilities) is
    structurally impossible for a symmetric seed.
  * NEWTON solves, Picard checks: a sparse Sherman-Morrison Newton (frozen-pressure sparse
    tangent + rank-one mass feedback, fold deflated in the SM denominator) is the
    DIFFERENTIABLE forward -- Newton primal + SM-factored adjoint. It is
    contraction-independent, so it converges genuine equilibria even at/past the FDM
    constant-mass fold (lambda_sym -> 1), where the damped Picard critically slows. The
    damped Picard is demoted to a drift/stability check.
  * Sobolev-metric L-BFGS is the ONLY stepper: H0 = gamma*B^-1 (per-family screened
    Laplacian) at the centre of the two-loop recursion -> mesh-independent descent, sharp
    optimum features (rim/apex) stay reachable. Armijo line search on forward-only solves.
  * The FDM tangent's constant-mass fold is a GEOMETRIC diagnostic, never a constraint:
    FDM omits the always-positive material stiffness EA, and the k=0 fold mode is the pure
    membrane-stretching mode a real material resists most -- a TRUE equilibrium past the FDM
    fold is stable for a real material. lambda is reported, not gated (governor/fold-veto
    default OFF).

ENVIRONMENT
-----------
Run in the dedicated `jaxfdm` env (jax_fdm pins compas 1.x, includes the amended
ConstantMassModelSparse / global_load_fn hook):

    python ParaboloidDish_unstructured_JAXFDM.py            (jax environment)
"""

import os
import time

# 64-bit precision BEFORE importing jax (the near-fold constant-mass adjoint loses too many
# digits in float32; the validation runs float64 for exactly this reason).
os.environ.setdefault("JAX_ENABLE_X64", "True")

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import EquilibriumMeshStructureSparse
from jax_fdm.equilibrium import ConstantMassModelSparse
from jax_fdm.equilibrium.states import EquilibriumParametersState
from jax_fdm.equilibrium.solvers import solver_forward
from equinox.internal import while_loop

# --- gather/scatter connectivity patch (REVERSIBLE; JAXFDM_NO_SCATTER_PATCH=1 disables) ------
# Replaces jax_fdm's dense connectivity matvecs with matrix-free gather/scatter so the
# follower value_and_grad compiles cheaply (bit-identical equilibria + gradients). Must
# import AFTER the jax_fdm imports, before any solve.
if os.environ.get("JAXFDM_NO_SCATTER_PATCH") != "1":
    import jaxfdm_scatter_patch  # noqa: F401  (monkeypatches jax_fdm on import)
    print("[scatter-patch] dense connectivity -> matrix-free gather/scatter (active)", flush=True)

# =============================================================================
# USER CONTROLS — edit this block
# =============================================================================

# Unstructured mesh (paraboloid_unstructured_mesh.py output) and the number of q-profile stations.
MESH_NPZ = os.environ.get("JAXFDM_MESH_NPZ", "")
N_STATIONS = int(os.environ.get("JAXFDM_N_STATIONS", "25"))
# q parametrisation: "stations" (smooth q_m(s), q_h(s) profiles, see header) or "edge" (one free q per
# edge, as SphereDishFanBeam_unstructured_JAXFDM / cap_A_unstr).  Per-edge q gives every node the local
# freedom to balance the in-plane forces of an irregular net; a smooth profile cannot, so its nodes
# slide (2026-10-01).  "edge" is seeded with COTANGENT weights q_e = (cot a + cot b)/2 on the seed
# surface -- a uniform isotropic membrane stress, so the seed is mesh-consistent (uniform q is not:
# a triangle net under uniform q carries sqrt3 q) and carries no knowledge of the target stress.
Q_MODE = os.environ.get("JAXFDM_Q_MODE", "stations").strip().lower()
EDGE_Q_BOUNDS = (1e-3, 1e3)
# Per-edge seed: "cot" (cotangent weights, default) or "const" (every edge START_Q).
EDGE_SEED = os.environ.get("JAXFDM_EDGE_SEED", "cot").strip().lower()
assert EDGE_SEED in ("cot", "const"), EDGE_SEED
# Long-run monitoring (cap_A_unstr style: patience off, run to the mesh floor).  Every step appends
# step/objective/nodal fit/area-weighted surface error to history.csv in TEMP_DIR; every SNAP_EVERY
# accepted steps the BEST iterate is written to snap_latest.npz.  Creating TEMP_DIR/STOP ends the run
# cleanly (Newton polish + normal save).  SNAP_EVERY = 0 disables both.
SNAP_EVERY = int(os.environ.get("JAXFDM_SNAP_EVERY", "0"))
# Checkpoint / resume (ported from SphereDishFanBeam_unstructured_JAXFDM): every CKPT_EVERY accepted
# steps, ckpt.npz in TEMP_DIR holds the CURRENT iterate plus the full L-BFGS state (pairs, previous
# (x, g), skip flag), trust radius, patience counter, best state and histories -- taken after the
# step's bookkeeping and BEFORE its L-BFGS update.  JAXFDM_RESUME_NPZ=<ckpt.npz> re-enters the loop
# at exactly that point (the re-entry step's bookkeeping is not repeated), so a resumed run follows
# the uninterrupted one.  Step numbers are absolute across resumes.  0 = no checkpoints.
CKPT_EVERY = int(os.environ.get("JAXFDM_CKPT_EVERY", "0"))
RESUME_NPZ = os.environ.get("JAXFDM_RESUME_NPZ", "")
# Warm start (see optimise_native_mass): a finished run's npz, continued under a NEW objective.
WARM_NPZ = os.environ.get("JAXFDM_WARM_NPZ", "")
assert Q_MODE in ("stations", "edge"), Q_MODE
# Kept only so the saved npz has the same keys as the polar run (n_theta=0 marks "no sectors").
n_theta = 0
n_rings = N_STATIONS - 1

# Target paraboloid dish: pinned circular rim radius R at z=0; apex rise DEPTH at r=0.
#   z_target(r) = DEPTH * (1 - (r/R)^2);  focal length f = R^2 / (4*DEPTH).
# DEPTH/R = 0.5 here (f/D = 0.25, a "deep dish") makes the sphere-vs-paraboloid distinction
# a genuine optimisation task while staying comfortably below the inflation fold.
R     = float(os.environ.get("JAXFDM_R", "3.0"))
DEPTH = float(os.environ.get("JAXFDM_DEPTH", "1.5"))

# Every edge starts at this force density; bounds are role-specific (hoop / meridional / fan).
START_Q      = 1.0
FAN_Q_BOUNDS = (0.02, 500.0)     # apex spokes (meridional level 0)
MER_Q_BOUNDS = (0.010, 600.0)    # ring-to-ring meridional edges
HOOP_Q_BOUNDS = (0.050, 800.0)   # circumferential hoops

# Constant-mass follower pressure: GAS_K calibrated once (p = sign*GAS_K/V), then held FIXED
# by default; q does the fitting. PRESSURE is only the initial calibration magnitude.
PRESSURE = float(os.environ.get("JAXFDM_PRESSURE", "0.35"))

# Objective weights: radial (rho), axial (z), apex.
W_RAD   = float(os.environ.get("JAXFDM_W_RAD", "1.0"))
W_AXIAL = float(os.environ.get("JAXFDM_W_AXIAL", "1.0"))
W_APEX  = float(os.environ.get("JAXFDM_W_APEX", "1.0"))
# Unstructured copy: "nodal" (as the polar solver) or "normal" (distance to the paraboloid; see _loss).
OBJECTIVE = os.environ.get("JAXFDM_OBJECTIVE", "nodal").strip().lower()
W_NORMAL = float(os.environ.get("JAXFDM_W_NORMAL", "1.0"))
# Optional weight on TANGENTIAL drift from the seed (normal objective only). 0 = sliding is free;
# W_TAN = 1 with W_NORMAL = 1 is, to first order, the nodal objective again.
W_TAN = float(os.environ.get("JAXFDM_W_TAN", "0.0"))
# Element-quality penalty, ported verbatim from BWBCentrebody_v8_pressure_JAXFDM.Model.mesh_pen (the
# astigmatic reflector's f_A + f_S): one-sided Huber penalties on triangle area relative to its OWN
# seed area above MESH_CAP_A, and on the shape ratio SR = (l1^2+l2^2+l3^2)/(4 sqrt3 A) (1 =
# equilateral) above MESH_CAP_S.  Zero for every triangle inside both caps.  BWB defaults
# (W_MESH 0.1, caps 2.5 / 2.0); W_MESH = 0 (default here) = off, i.e. the polar formulation.
# NB as in the BWB there is no area FLOOR: a shrinking triangle is caught by SR (it goes sliver)
# or by its neighbours' area cap, not directly.
W_MESH = float(os.environ.get("JAXFDM_W_MESH", "0.0"))
# Smooth element-QUALITY energy (2026-10-04): with per-edge q the shape fixes only the NORMAL position
# of the nodes; their tangential position is spare freedom.  Use it for the mesh instead of letting
# nodes drift:  W_QUAL * [ mean (A/A_seed - 1)^2 + mean (SR - 1)^2 ]  over every triangle, A_seed the
# triangle's own seed area (absolute, so blind neither to shrinkage nor to drift), SR the shape ratio
# (1 = equilateral).  Smooth everywhere (no cap to rest on).  Pair with JAXFDM_OBJECTIVE=normal.
W_QUAL = float(os.environ.get("JAXFDM_W_QUAL", "0.0"))
# Curvature-sensitive shape term (2026-10-14, opt-in; 0 = off, identical to before).  Membrane stress follows
# the CURVATURE of the form, which a nodal offset d changes by ~ d/h^2: offsets the mean(d^2) term barely sees
# can still carry percent-level stress errors on fine meshes.  Adds W_CURV * mean(e_i^2) over the shape nodes,
# e_i = 4 (d_i - mean_{j~i} d_j) / (h_i^2 k0): the umbrella Laplacian of the normal offset field as a relative
# curvature error (h_i = mean seed edge length at i, k0 = 2 DEPTH / R^2).  Zero exactly on the target surface.
W_CURV = float(os.environ.get("JAXFDM_W_CURV", "0.0"))
# On JAXFDM_RESUME_NPZ under a CHANGED objective: re-score 'best' at the resumed iterate and drop the L-BFGS
# pairs (both belong to the old objective).  Iterate, step count and history carry on.  0 = resume as before.
RESUME_RESET = os.environ.get("JAXFDM_RESUME_RESET", "0") == "1"
# Keep the Sherman-Morrison (Newton) adjoint even if its one-off cross-check against the damped-Picard adjoint
# fails (see _newton_grad_checked).  0 = previous behaviour (fall back to the damped gradient).
NEWTON_GRAD_FORCE = os.environ.get("JAXFDM_NEWTON_GRAD_FORCE", "0") == "1"
MESH_CAP_A = float(os.environ.get("JAXFDM_MESH_CAP_A", "2.5"))
MESH_CAP_S = float(os.environ.get("JAXFDM_MESH_CAP_S", "2.0"))
assert OBJECTIVE in ("nodal", "normal"), OBJECTIVE

# ---- Optimiser (Sobolev-metric L-BFGS) --------------------------------------------------
N_OUTER     = int(os.environ.get("JAXFDM_N_OUTER", "400"))
LR          = float(os.environ.get("JAXFDM_LR", "0.02"))     # trust radius seed (log-q units)
PATIENCE    = int(os.environ.get("JAXFDM_PATIENCE", "60"))
# ABSOLUTE improvement threshold (see the acceptance test in the outer loop: obj < best - TOL).
# It is absolute, not relative, so it stops meaning "has it stopped improving?" and starts
# meaning "is it below 1e-7?" as soon as the objective itself falls under the tolerance -- a
# well-fitted dish reaches obj ~1e-8 (shape rms ~1e-4 in form units, squared), i.e. ten times
# BELOW this gate, after which no further improvement can ever be registered and the run
# terminates on patience with the line search still healthy.  A mesh-refinement study sees this
# as a fit quality that will not improve with resolution.  Default unchanged; lower it
# (JAXFDM_IMPROVE_TOL=1e-12) when the converged objective, not the budget, has to be the stop.
IMPROVE_TOL = float(os.environ.get("JAXFDM_IMPROVE_TOL", "1e-7"))

# ---- Native constant-mass solver tuning -------------------------------------------------
NATIVE_URS_TMAX = 500                                          # damped-Picard iteration cap
NATIVE_URS_ETA  = float(os.environ.get("JAXFDM_URS_ETA", "1e-4"))
NATIVE_MASS_BETA = float(os.environ.get("JAXFDM_MASS_BETA", "0.5"))  # damped under-relaxation

# GAS_K calibration ramp (creep up the stable branch to the target apex; safe far below fold).
NATIVE_CALIB_ITERS       = 80
NATIVE_CALIB_STEP        = 1.25
NATIVE_CALIB_STEP_SHRINK = 0.5
NATIVE_APEX_TOL_FRAC     = 0.03

# Health gates on the converged equilibrium (fractions of target apex / baseline volume).
NATIVE_GATE_FRAC        = 0.02
NATIVE_COLLAPSE_FRAC    = 0.5
NATIVE_VOL_FLOOR_FRAC   = 0.5
NATIVE_OVERINFLATE_FRAC = 1.5
NATIVE_VOL_CEIL_FRAC    = 5.0
NATIVE_ACCEPT_RES  = float(os.environ.get("JAXFDM_ACCEPT_RES", "1e-3"))
NATIVE_RUNAWAY_TOL = float(os.environ.get("JAXFDM_RUNAWAY_TOL", "1e-2"))
NATIVE_LR_RECOVER  = float(os.environ.get("JAXFDM_LR_RECOVER", "1.15"))

# ---- Optional gas amount as a design variable (OFF by default: the fit gradient de-pressurises
# straight across the fold onto the collapsed branch; calibrate once and let q fit). ---------
OPTIMISE_GAS_K = os.environ.get("JAXFDM_OPTIMISE_GASK", "0") not in ("0", "false", "False")
GK_LR_SCALE    = float(os.environ.get("JAXFDM_GK_LR_SCALE", "0.25"))
GK_MIN_FRAC    = 0.2
GK_MAX_FRAC    = 25.0

# ---- Axisymmetric q reduction: one q per hoop ring + one per meridional level ----------------
# Unstructured: the reduction is the station-profile map (always on); the ring projectors are off.
AXISYM_Q = True
AXISYM_STATE_PROJECT = False

# ---- Sobolev (H1) gradient preconditioner over the reduced q-profiles ------------------------
SOBOLEV_ELL = float(os.environ.get("JAXFDM_SOBOLEV_ELL", "2.0"))

# ---- L-BFGS knobs ----------------------------------------------------------------------------
LBFGS_M      = int(os.environ.get("JAXFDM_LBFGS_M", "10"))
LBFGS_LS_MAX = int(os.environ.get("JAXFDM_LBFGS_LS_MAX", "8"))
LBFGS_CAP    = float(os.environ.get("JAXFDM_LBFGS_CAP", "10.0"))
LBFGS_CURV   = float(os.environ.get("JAXFDM_LBFGS_CURV", "1e-4"))
LBFGS_GK_H0  = float(os.environ.get("JAXFDM_LBFGS_GK_H0", str(GK_LR_SCALE)))

# ---- Newton+SVD / Sherman-Morrison equilibrium refinement (validated stack) -----------------
NEWTON_MODE      = os.environ.get("JAXFDM_NEWTON", "on").strip().lower()  # on|auto|off
NEWTON_MAX_DOF   = int(os.environ.get("JAXFDM_NEWTON_MAX_DOF", "4500"))
NEWTON_LAMBDA    = float(os.environ.get("JAXFDM_NEWTON_LAMBDA", "0.0"))   # 0 = Newton always
NEWTON_TMAX_IT   = int(os.environ.get("JAXFDM_NEWTON_TMAX", "12"))
NEWTON_TOL       = float(os.environ.get("JAXFDM_NEWTON_TOL", "1e-10"))
NEWTON_JAC_CHUNK = int(os.environ.get("JAXFDM_NEWTON_JAC_CHUNK", "512"))
NEWTON_LINSOLVE  = os.environ.get("JAXFDM_NEWTON_LINSOLVE", "sm").strip().lower()  # sm|dense
NEWTON_DEN_TOL   = float(os.environ.get("JAXFDM_NEWTON_DEN_TOL", "1e-8"))
NEWTON_GRAD      = os.environ.get("JAXFDM_NEWTON_GRAD", "1") not in ("0", "false", "False")

# ---- Stability diagnostic (lambda_max power iteration; reported, not gated) ------------------
STAB_EVERY    = int(os.environ.get("JAXFDM_STAB_EVERY", "10"))   # measure lambda every N steps (0=off)
STAB_PI_ITERS = int(os.environ.get("JAXFDM_STAB_ITERS", "25"))
# Governor / fold-veto: OFF by default (lambda is a diagnostic, crossing the fold is correct).
STAB_GOVERN = os.environ.get("JAXFDM_STAB_GOVERN", "0") not in ("0", "false", "False")
LAMBDA_SAFE = float(os.environ.get("JAXFDM_LAMBDA_SAFE", "0.97"))
LAMBDA_GROW = float(os.environ.get("JAXFDM_LAMBDA_GROW", "0.85"))
LAMBDA_HARD = float(os.environ.get("JAXFDM_LAMBDA_HARD", "0.995"))
LAMBDA_HITS = int(os.environ.get("JAXFDM_LAMBDA_HITS", "3"))
FOLD_VETO   = os.environ.get("JAXFDM_FOLD_VETO", "0") not in ("0", "false", "False")
LAMBDA_VETO = float(os.environ.get("JAXFDM_LAMBDA_VETO", "1.0"))

# Output / viewer.
TEMP_DIR    = os.environ.get(
    "JAXFDM_TEMP_DIR", os.environ.get("PIPELINE_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
SHOW_VIEWER = os.environ.get("JAXFDM_NO_VIEWER") != "1"

# =============================================================================
# Derived configuration + design-variable metadata
# =============================================================================

target_height = DEPTH
FOCAL = R * R / (4.0 * DEPTH)

HOOP_RINGS = list(range(1, n_rings))          # interior hoops (rim ring is pinned)
HOOP_RING_SET = set(HOOP_RINGS)

# Filled by build_disk_topology().
CATEGORIES, CAT_ID = [], {}
CAT_Q0 = CAT_LO = CAT_HI = np.zeros(0)


def hoop_cat(ring):
    return f"hoop@{ring}"


def cat_spec(name):
    """Initial value and (min, max) bounds keyed by role name prefix."""
    if name.startswith("hoop@"):
        lo, hi = HOOP_Q_BOUNDS
    elif name == "fan":
        lo, hi = FAN_Q_BOUNDS
    else:                                      # "mer"
        lo, hi = MER_Q_BOUNDS
    return START_Q, lo, hi


# =============================================================================
# Unstructured topology (read from JAXFDM_MESH_NPZ) + frozen station-profile q map
# =============================================================================

EDGE_ROLE = {}        # (min(u,v), max(u,v)) -> "mer" | "hoop": dominant direction, labels only


def edge_role(u, v):
    """Dominant direction of the edge on the seed surface ('mer' if cos^2 a >= 1/2, else 'hoop').
    A LABEL for diagnostics only; the q map itself uses the exact cos^2 / sin^2 split."""
    return EDGE_ROLE[(min(u, v), max(u, v))]


def _newell_normal(pts):
    pts = np.asarray(pts, dtype=float)
    N = np.zeros(3)
    n = len(pts)
    for k in range(n):
        x0, y0, z0 = pts[k]
        x1, y1, z1 = pts[(k + 1) % n]
        N[0] += (y0 - y1) * (z0 + z1)
        N[1] += (z0 - z1) * (x0 + x1)
        N[2] += (x0 - x1) * (y0 + y1)
    return N


def paraboloid_z(r):
    """Target dome height at radius r: z = DEPTH*(1 - (r/R)^2)."""
    return DEPTH * (1.0 - (np.asarray(r) / R) ** 2)


def meridian_arclength_table(n=20001):
    """(r, s(r)): meridian arc length from the apex on the target paraboloid."""
    rg = np.linspace(0.0, R, n)
    ds = np.sqrt(1.0 + (2.0 * DEPTH * rg / R ** 2) ** 2)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(rg))])
    return rg, s


def build_station_map(vertices, edges):
    """Frozen linear map from the station values (q_m[k], q_h[k]) to per-edge q, built on the SEED
    surface.  Per edge: bracketing station k0 and hat weight w (profile value at s_e is
    (1-w) Q[k0] + w Q[k0+1]), and the direction split c2 = cos^2 a, s2 = sin^2 a, a the angle
    between the edge and the local meridian (both measured in the tangent plane)."""
    rg, sg = meridian_arclength_table()
    S = float(sg[-1])
    V = np.asarray(vertices, float)
    E = np.asarray(edges, int)
    a, b = V[E[:, 0]], V[E[:, 1]]
    mid = 0.5 * (a + b)
    r_mid = np.hypot(mid[:, 0], mid[:, 1])
    s_e = 0.5 * (np.interp(np.hypot(a[:, 0], a[:, 1]), rg, sg)
                 + np.interp(np.hypot(b[:, 0], b[:, 1]), rg, sg))
    th = np.arctan2(mid[:, 1], mid[:, 0])
    dzdr = -2.0 * DEPTH * r_mid / R ** 2
    m_hat = (np.column_stack([np.cos(th), np.sin(th), dzdr])
             / np.sqrt(1.0 + dzdr ** 2)[:, None])
    h_hat = np.column_stack([-np.sin(th), np.cos(th), np.zeros_like(th)])
    t = b - a
    tm = np.einsum("ij,ij->i", t, m_hat)
    tt = np.einsum("ij,ij->i", t, h_hat)
    c2 = tm ** 2 / np.maximum(tm ** 2 + tt ** 2, 1e-300)
    c2 = np.where(r_mid > 1e-12, c2, 1.0)
    x = np.clip(s_e / S * (N_STATIONS - 1), 0.0, N_STATIONS - 1 - 1e-12)
    k0 = np.floor(x).astype(int)
    return dict(k0=k0, w=x - k0, c2=c2, s2=1.0 - c2,
                s_st=np.linspace(0.0, S, N_STATIONS), s_e=s_e, S=S)


def build_disk_topology():
    """Read the unstructured paraboloid mesh.  Every node is ON the target paraboloid (seed),
    the rim nodes are pinned, `apex` is the node at r = 0.  Faces are wound with the outward
    (upward) normal: the dish is a graph z(r), so outward is simply n_z > 0."""
    global CATEGORIES, CAT_ID, CAT_Q0, CAT_LO, CAT_HI, EDGE_ROLE
    if not MESH_NPZ:
        raise SystemExit("JAXFDM_MESH_NPZ is required (paraboloid_unstructured_mesh.py output)")
    m = np.load(MESH_NPZ)
    if abs(float(m["R"]) - R) > 1e-12 or abs(float(m["DEPTH"]) - DEPTH) > 1e-12:
        raise SystemExit(f"mesh R/DEPTH {float(m['R'])}/{float(m['DEPTH'])} != solver {R}/{DEPTH}")
    vertices = np.asarray(m["vertices"], dtype=float)
    n_v = vertices.shape[0]
    apex = int(m["apex"])
    fixed = [int(v) for v in m["rim"]]

    faces, n_rev = [], 0
    for f in np.asarray(m["tris"], int):
        idx = [int(v) for v in f]
        if _newell_normal(vertices[idx])[2] < 0.0:
            idx = idx[::-1]
            n_rev += 1
        faces.append(idx)
    faces_tri = np.asarray(faces, dtype=int)
    assert all(_newell_normal(vertices[f])[2] > 0.0 for f in faces), "winding not coherent"
    print(f"[orient] re-wound {n_rev}/{len(faces)} triangles to the outward (n_z>0) side",
          flush=True)

    edge_set, edges = set(), []
    for f in faces:
        for a in range(3):
            u, v = f[a], f[(a + 1) % 3]
            key = (u, v) if u < v else (v, u)
            if key not in edge_set:
                edge_set.add(key)
                edges.append((u, v))

    smap = build_station_map(vertices, edges)
    EDGE_ROLE = {(min(u, v), max(u, v)): ("mer" if c >= 0.5 else "hoop")
                 for (u, v), c in zip(edges, smap["c2"])}
    edge_roles = [edge_role(u, v) for (u, v) in edges]
    CATEGORIES = [f"{role}::e{u}-{v}" for role, (u, v) in zip(edge_roles, edges)]
    CAT_ID = {c: i for i, c in enumerate(CATEGORIES)}
    _specs = [cat_spec(role) for role in edge_roles]
    CAT_Q0 = np.asarray([s[0] for s in _specs])
    CAT_LO = np.asarray([s[1] for s in _specs])
    CAT_HI = np.asarray([s[2] for s in _specs])

    fixed_set = set(fixed)
    free = [v for v in range(n_v) if v not in fixed_set]
    # nearest station of every node (bookkeeping only: there are no rings)
    rg, sg = meridian_arclength_table()
    s_v = np.interp(np.hypot(vertices[:, 0], vertices[:, 1]), rg, sg)
    ring_ids = np.rint(s_v / smap["S"] * (N_STATIONS - 1)).astype(int)

    return dict(vertices=vertices, faces_quad=faces, faces_tri=faces_tri,
                edges=edges, cat_ids=np.arange(len(edges)),
                fixed=fixed, free=free, ring_ids=ring_ids, apex=apex, anchor=None,
                target_rho=np.hypot(vertices[:, 0], vertices[:, 1]),
                target_z=vertices[:, 2].copy(),
                target_sx=1.0, target_sy=1.0, target_sz=1.0,
                target_cap_height=DEPTH, target_total_height=DEPTH,
                station_map=smap)


# =============================================================================
# Follower pressure -> nodal loads, enclosed volume, energy, RMS (numpy)
# =============================================================================

def follower_loads(coords, faces_tri, pressure):
    """Area-weighted OUTWARD pressure lumped to the nodes (numpy)."""
    loads = np.zeros_like(coords)
    centroid = coords.mean(axis=0)
    vi, vj, vk = coords[faces_tri[:, 0]], coords[faces_tri[:, 1]], coords[faces_tri[:, 2]]
    area_vec = 0.5 * np.cross(vj - vi, vk - vi)
    face_ctr = (vi + vj + vk) / 3.0
    flip = np.einsum("fi,fi->f", area_vec, face_ctr - centroid) < 0.0
    area_vec[flip] *= -1.0
    f_face = pressure * area_vec / 3.0
    for c in range(3):
        np.add.at(loads, faces_tri[:, c], f_face)
    return loads


def enclosed_volume(coords, faces_tri, z_ref=0.0):
    """Volume enclosed between the dome surface and the rim plane z=z_ref (signed
    projected-area form; the planar rim contributes 0)."""
    vi, vj, vk = coords[faces_tri[:, 0]], coords[faces_tri[:, 1]], coords[faces_tri[:, 2]]
    axy = 0.5 * ((vj[:, 0] - vi[:, 0]) * (vk[:, 1] - vi[:, 1])
                 - (vk[:, 0] - vi[:, 0]) * (vj[:, 1] - vi[:, 1]))
    zc = (vi[:, 2] + vj[:, 2] + vk[:, 2]) / 3.0 - z_ref
    return abs(float(np.sum(axy * zc)))


def total_energy(coords, q_edge, edges_uv, faces_tri, gas_k):
    """Potential Pi(x) = 0.5*sum_e q_e*||x_u-x_v||^2 - gas_k*ln V(x) of the constant-mass
    system. The tri-lumped follower load is exactly +(gas_k/V)*dV/dx, so equilibria are
    critical points of Pi and the damped Picard descends Pi in the K-metric."""
    p = np.asarray(coords)
    eu, ev = edges_uv
    d = p[eu] - p[ev]
    elastic = 0.5 * float(np.sum(np.asarray(q_edge) * np.einsum("ij,ij->i", d, d)))
    V = max(enclosed_volume(coords, faces_tri), 1e-12)
    return elastic - float(gas_k) * float(np.log(V))


_DUN_A, _DUN_B = 0.445948490915965, 0.091576213509771
_BARY = np.array([[_DUN_A, _DUN_A, 1 - 2 * _DUN_A], [_DUN_A, 1 - 2 * _DUN_A, _DUN_A],
                  [1 - 2 * _DUN_A, _DUN_A, _DUN_A], [_DUN_B, _DUN_B, 1 - 2 * _DUN_B],
                  [_DUN_B, 1 - 2 * _DUN_B, _DUN_B], [1 - 2 * _DUN_B, _DUN_B, _DUN_B]])
_BW = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


def surface_error_area(coords, faces_tri):
    """Area-weighted rms normal distance of the flat triangles to the target paraboloid (counts
    the sag between nodes; 6-point Dunavant).  Same as MeshStudy_paraboloid_unstructured.surface_error."""
    a, b, c = (coords[faces_tri[:, k]] for k in range(3))
    A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    P = (_BARY[None, :, 0, None] * a[:, None] + _BARY[None, :, 1, None] * b[:, None]
         + _BARY[None, :, 2, None] * c[:, None])
    r2 = P[..., 0] ** 2 + P[..., 1] ** 2
    k = 2.0 * DEPTH / R ** 2
    d = (P[..., 2] - DEPTH * (1.0 - r2 / R ** 2)) / np.sqrt(1.0 + k * k * r2)
    return float(np.sqrt(np.sum(A * (d ** 2 @ _BW)) / A.sum()))


def shape_nodes_of(topo):
    return [v for v in topo["free"] if v != topo["apex"]]


def surface_rms(coords, topo):
    """RMS deviation in the radial coordinate rho = sqrt(x^2 + y^2) from the target meridian
    radius (rho = target_rho describes the paraboloid). Returns (rms, max, rho, z, d)."""
    nodes = shape_nodes_of(topo)
    xf = coords[nodes]
    rho = np.sqrt(xf[:, 0] ** 2 + xf[:, 1] ** 2)
    d = rho - topo["target_rho"][nodes]
    return float(np.sqrt(np.mean(d ** 2))), float(np.abs(d).max()), rho, xf[:, 2], d


# =============================================================================
# Native constant-mass structure (EquilibriumMeshStructureSparse)
# =============================================================================

def build_mesh_structure(topo):
    """Build the native sparse structure + base params (skin q + unit face loads) and rewrite
    CATEGORIES/bounds into mesh-edge order. Returns (bp, st, n_skin)."""
    global CATEGORIES, CAT_ID, CAT_Q0, CAT_LO, CAT_HI

    verts = topo["vertices"]
    skin_faces = [list(map(int, f)) for f in topo["faces_quad"]]
    fixed = [int(k) for k in topo["fixed"]]

    mesh = FDMesh.from_vertices_and_faces(verts.tolist(), skin_faces)
    for k in fixed:
        mesh.vertex_support(k)
    for (u, v) in mesh.edges():
        mesh.edge_forcedensity((u, v), 1.0)
    mesh.faces_loads([0.0, 0.0, 1.0])

    bst = EquilibriumMeshStructureSparse.from_mesh(mesh)
    bp = EquilibriumParametersState.from_datastructure(mesh)

    skin = np.asarray(bst.edges)                                  # (n_skin, 2), mesh order
    vk = np.asarray(bst.vertices)
    supp = np.isin(vk, np.asarray(fixed)).astype(skin.dtype)
    st = EquilibriumMeshStructureSparse(vk, np.asarray(bst.faces),
                                        edges=skin, supports=supp)

    # Rebuild design-variable metadata in mesh-edge order so diagnostics / q line up.
    edges_list = [(int(u), int(v)) for (u, v) in skin]
    CATEGORIES, _q0, _lo, _hi = [], [], [], []
    for (u, v) in edges_list:
        role = edge_role(u, v)
        q0, lo, hi = cat_spec(role)
        CATEGORIES.append(f"{role}::e{u}-{v}")
        _q0.append(q0); _lo.append(lo); _hi.append(hi)
    CAT_ID = {c: i for i, c in enumerate(CATEGORIES)}
    CAT_Q0 = np.asarray(_q0)
    CAT_LO = np.asarray(_lo)
    CAT_HI = np.asarray(_hi)
    topo["edges"] = edges_list
    topo["cat_ids"] = np.arange(len(edges_list))
    return bp, st, len(skin)


# =============================================================================
# Damped-Picard solver (passed via the low-level solver= arg)
# =============================================================================

def _make_damped_picard(beta, project=None):
    """Fixed-point solver x <- (1-beta) x + beta f(x). `project` (optional, jnp-traceable) is
    applied to the iterate every sweep -- with the axisymmetric projector this makes k!=0
    growth structurally impossible WITHIN a solve. The implicit adjoint is registered on the
    residual map independent of this solver and project(x*) = x*, so the gradient is unchanged."""
    def solver_damped_picard(f, a, x_init, solver_config):
        tmax = solver_config["tmax"]
        eta = solver_config["eta"]

        def dist(x_prev, x):
            return jnp.mean(jnp.linalg.norm(x_prev - x, axis=1))

        def cond_fun(carry):
            x_prev, x = carry
            return dist(x_prev, x) > eta

        def step(x):
            x1 = (1.0 - beta) * x + beta * f(a, x)
            return project(x1) if project is not None else x1

        def body_fun(carry):
            _, x = carry
            return x, step(x)

        _, x_star = while_loop(cond_fun, body_fun, (x_init, step(x_init)),
                               max_steps=tmax, kind="checkpointed")
        return x_star

    return solver_damped_picard


_NATIVE_MASS_DAMPED = _make_damped_picard(NATIVE_MASS_BETA)


# =============================================================================
# NATIVE constant-mass optimiser (Newton SM-adjoint forward + Sobolev L-BFGS)
# =============================================================================

def optimise_native_mass(topo, q_initial=None, coords_initial=None, stage_label="",
                         controls=None):
    """Constant-mass INSIDE jax: p = sign*GAS_K/V computed in the differentiated damped fixed
    point (ConstantMassModelSparse). NEWTON (sparse Sherman-Morrison) is the differentiable
    forward via an SM-factored adjoint; damped Picard is the drift/stability check. Sobolev-
    metric L-BFGS is the stepper. Returns a 12-tuple compatible with diagnostics/view."""
    if controls is None:
        controls = _controls()
    N_OUTER_ = controls["N_OUTER"]
    LR_ = controls["LR"]
    PATIENCE_ = controls["PATIENCE"]
    IMPROVE_TOL_ = controls["IMPROVE_TOL"]

    bp, st, n_skin = build_mesh_structure(topo)
    qdtype = bp.q.dtype
    unit_faces = bp.loads.faces
    faces_tri = topo["faces_tri"]
    ft = jnp.asarray(faces_tri)
    n_v = topo["vertices"].shape[0]

    # ---- station-profile q map: design vector = [log q_m(s_k)] + [log q_h(s_k)] ---------------
    # Rebuilt HERE because build_mesh_structure has just re-ordered topo["edges"] to mesh order.
    smap = build_station_map(topo["vertices"], topo["edges"])
    topo["station_map"] = smap
    K = N_STATIONS
    GROUP_NAMES = [f"mer@{k}" for k in range(K)] + [f"hoop@{k}" for k in range(K)]
    LO_RED = np.asarray([MER_Q_BOUNDS[0]] * K + [HOOP_Q_BOUNDS[0]] * K)
    HI_RED = np.asarray([MER_Q_BOUNDS[1]] * K + [HOOP_Q_BOUNDS[1]] * K)
    _K0, _W = smap["k0"], smap["w"]
    _C2, _S2 = smap["c2"], smap["s2"]
    if Q_MODE == "edge":
        GROUP_NAMES = [f"e{u}-{v}" for (u, v) in topo["edges"]]
        LO_RED = np.full(len(GROUP_NAMES), EDGE_Q_BOUNDS[0])
        HI_RED = np.full(len(GROUP_NAMES), EDGE_Q_BOUNDS[1])
    print(f"[stations] q reparametrised: {len(topo['edges'])} edges -> {len(GROUP_NAMES)} "
          f"profile variables ({K} meridional + {K} hoop stations, ds={smap['S'] / (K - 1):.4f}); "
          f"edge direction split: mean cos^2 a = {_C2.mean():.3f}", flush=True)
    LOG_LO_NP, LOG_HI_NP = np.log(LO_RED), np.log(HI_RED)
    log_lo = jnp.asarray(LOG_LO_NP, dtype=qdtype)
    log_hi = jnp.asarray(LOG_HI_NP, dtype=qdtype)
    idx_free = np.asarray(st.indices_free)
    _K0j, _K1j = jnp.asarray(_K0), jnp.asarray(_K0 + 1)
    _Wj = jnp.asarray(_W, dtype=qdtype)
    _C2j = jnp.asarray(_C2, dtype=qdtype)
    _S2j = jnp.asarray(_S2, dtype=qdtype)

    def q_full_of(theta):
        qs = jnp.exp(jnp.clip(theta, log_lo, log_hi)).astype(qdtype)
        if Q_MODE == "edge":
            return qs
        qm, qh = qs[:K], qs[K:]
        return (_C2j * ((1.0 - _Wj) * qm[_K0j] + _Wj * qm[_K1j])
                + _S2j * ((1.0 - _Wj) * qh[_K0j] + _Wj * qh[_K1j]))

    def q_full_np(theta):
        qs = np.exp(np.clip(np.asarray(theta, dtype=float), LOG_LO_NP, LOG_HI_NP))
        if Q_MODE == "edge":
            return qs
        qm, qh = qs[:K], qs[K:]
        return (_C2 * ((1.0 - _W) * qm[_K0] + _W * qm[_K0 + 1])
                + _S2 * ((1.0 - _W) * qh[_K0] + _W * qh[_K0 + 1]))

    # ---- per-SWEEP symmetry projection inside the damped solver (jittable) -------------------
    sweep_project = None
    if AXISYM_Q and AXISYM_STATE_PROJECT:
        _vref_f = np.asarray(topo["vertices"], dtype=float)[idx_free]
        _thf = np.arctan2(_vref_f[:, 1], _vref_f[:, 0])
        _CTHF = jnp.asarray(np.cos(_thf), dtype=qdtype)
        _STHF = jnp.asarray(np.sin(_thf), dtype=qdtype)
        _rid_np = np.asarray(topo["ring_ids"])[idx_free]
        _RIDF = jnp.asarray(_rid_np, dtype=int)
        _nring = int(np.asarray(topo["ring_ids"]).max()) + 1
        _CNTF = jnp.asarray(np.maximum(np.bincount(_rid_np, minlength=_nring), 1),
                            dtype=qdtype)

        def sweep_project(xf):
            vr = xf[:, 0] * _CTHF + xf[:, 1] * _STHF
            vt = -xf[:, 0] * _STHF + xf[:, 1] * _CTHF
            mr = jnp.zeros(_nring, dtype=xf.dtype).at[_RIDF].add(vr) / _CNTF
            mt = jnp.zeros(_nring, dtype=xf.dtype).at[_RIDF].add(vt) / _CNTF
            mz = jnp.zeros(_nring, dtype=xf.dtype).at[_RIDF].add(xf[:, 2]) / _CNTF
            mr = mr.at[0].set(0.0)
            mt = mt.at[0].set(0.0)
            return jnp.stack([mr[_RIDF] * _CTHF - mt[_RIDF] * _STHF,
                              mr[_RIDF] * _STHF + mt[_RIDF] * _CTHF,
                              mz[_RIDF]], axis=1)

        print("[axisym] per-SWEEP solver projection ACTIVE: k!=0 content cannot grow "
              "within a solve", flush=True)

    _SOLVER = (_make_damped_picard(NATIVE_MASS_BETA, project=sweep_project)
               if sweep_project is not None else _NATIVE_MASS_DAMPED)

    # ---- Sobolev (H1) preconditioner ---------------------------------------------------------
    SOBOLEV_APPLY = None
    if AXISYM_Q and SOBOLEV_ELL > 0.0:
        def _fam_rows(fam):
            pairs = sorted((int(n.split("@")[1]), k) for k, n in enumerate(GROUP_NAMES)
                           if n.split("@")[0] == fam)
            return np.asarray([k for _, k in pairs], dtype=int)

        def _h1_inv(nn, ell):
            D2 = np.zeros((nn, nn))
            for i in range(nn):
                if i > 0:
                    D2[i, i - 1] = 1.0
                if i + 1 < nn:
                    D2[i, i + 1] = 1.0
                D2[i, i] = -(int(i > 0) + int(i + 1 < nn))    # Neumann ends
            return np.linalg.inv(np.eye(nn) - (ell ** 2) * D2)

        _fam_blocks = [(rows, _h1_inv(rows.size, SOBOLEV_ELL))
                       for rows in (_fam_rows("hoop"), _fam_rows("mer")) if rows.size >= 3]

        def SOBOLEV_APPLY(g_np):
            out = np.array(g_np, dtype=float, copy=True)
            for rows, a_inv in _fam_blocks:
                out[rows] = a_inv @ out[rows]
            return out

        print(f"[sobolev] H1 gradient preconditioner ACTIVE: ell={SOBOLEV_ELL:g} over "
              f"{len(_fam_blocks)} profile families "
              f"({'+'.join(str(rows.size) for rows, _ in _fam_blocks)} stations)", flush=True)

    # ---- L-BFGS two-loop in the Sobolev metric -----------------------------------------------
    def _sob_joint(v):
        out = np.array(v, dtype=float, copy=True)
        if OPTIMISE_GAS_K:
            if SOBOLEV_APPLY is not None:
                out[:-1] = SOBOLEV_APPLY(out[:-1])
            out[-1] *= LBFGS_GK_H0
        elif SOBOLEV_APPLY is not None:
            out = SOBOLEV_APPLY(out)
        return out

    def _lbfgs_direction(g_now, pairs, scale0):
        q_ = np.array(g_now, dtype=float, copy=True)
        alphas = []
        for s_i, y_i, rho_i in reversed(pairs):
            a_i = rho_i * float(s_i @ q_)
            alphas.append(a_i)
            q_ -= a_i * y_i
        r = _sob_joint(q_)
        if pairs:
            s_l, y_l, _ = pairs[-1]
            r *= float(s_l @ y_l) / max(float(y_l @ _sob_joint(y_l)), 1e-30)
        else:
            r *= scale0 / max(float(np.abs(r).max()), 1e-30)
        for (s_i, y_i, rho_i), a_i in zip(pairs, reversed(alphas)):
            b_i = rho_i * float(y_i @ r)
            r += (a_i - b_i) * s_i
        return -r

    def _ls_fail_txt(why):
        parts = [f"{k} x{v}" for k, v in why.items() if v]
        return ", ".join(parts) if parts else "no trials"

    apex = topo["apex"]
    shape_nodes = shape_nodes_of(topo)
    shape_rows = jnp.asarray(shape_nodes)
    target_rho = jnp.asarray(topo["target_rho"][shape_nodes], dtype=qdtype)
    target_z = jnp.asarray(topo["target_z"][shape_nodes], dtype=qdtype)
    target_rho_np = topo["target_rho"][shape_nodes]
    target_z_np = topo["target_z"][shape_nodes]

    # OBJECTIVE=normal: first-order normal distance to the paraboloid, d = F/|grad F| with
    # F = z - DEPTH(1 - r^2/R^2) -- the true shape error; sliding ALONG the surface is free.
    # OBJECTIVE=nodal: the polar solver's (rho, z) nodal fit to the seed position.
    _kp = 2.0 * DEPTH / R ** 2
    _seed_np = np.asarray(topo["vertices"], dtype=float)[shape_nodes]
    _nrm_np = np.column_stack([_kp * _seed_np[:, 0], _kp * _seed_np[:, 1],
                               np.ones(len(shape_nodes))])
    _nrm_np /= np.linalg.norm(_nrm_np, axis=1)[:, None]          # seed surface normal
    _seed_j = jnp.asarray(_seed_np, dtype=qdtype)
    _nrm_j = jnp.asarray(_nrm_np, dtype=qdtype)

    # ---- element-quality penalty (BWB mesh_pen; A* = seed area of each triangle) -----------------
    _Fj = jnp.asarray(faces_tri)
    _V0 = np.asarray(topo["vertices"], dtype=float)
    _A_star = jnp.asarray(0.5 * np.linalg.norm(np.cross(
        _V0[faces_tri[:, 1]] - _V0[faces_tri[:, 0]], _V0[faces_tri[:, 2]] - _V0[faces_tri[:, 0]]),
        axis=1), dtype=qdtype)

    def _huber(u):
        return jnp.where(u <= 1.0, 0.5 * u * u, u - 0.5)

    def mesh_pen(X):
        a, b, c = X[_Fj[:, 0]], X[_Fj[:, 1]], X[_Fj[:, 2]]
        A = 0.5 * jnp.linalg.norm(jnp.cross(b - a, c - a), axis=1)
        l2 = jnp.sum((b - a) ** 2 + (c - b) ** 2 + (a - c) ** 2, axis=1)
        SR = l2 / (4.0 * jnp.sqrt(3.0) * A)
        eA = jnp.maximum(A / _A_star - MESH_CAP_A, 0.0)
        eS = jnp.maximum(SR - MESH_CAP_S, 0.0)
        return jnp.mean(_huber(eA)) + jnp.mean(_huber(eS))

    _mesh_pen_np = jax.jit(mesh_pen)

    def qual_energy(X):
        a, b, c = X[_Fj[:, 0]], X[_Fj[:, 1]], X[_Fj[:, 2]]
        A = 0.5 * jnp.linalg.norm(jnp.cross(b - a, c - a), axis=1)
        l2 = jnp.sum((b - a) ** 2 + (c - b) ** 2 + (a - c) ** 2, axis=1)
        SR = l2 / (4.0 * jnp.sqrt(3.0) * A)
        return jnp.mean((A / _A_star - 1.0) ** 2) + jnp.mean((SR - 1.0) ** 2)

    _qual_np = jax.jit(qual_energy)

    # ---- curvature-sensitive shape term (W_CURV > 0 only) -------------------------------------------
    _ceu = np.asarray([u for (u, v) in topo["edges"]], dtype=int)
    _cev = np.asarray([v for (u, v) in topo["edges"]], dtype=int)
    _cdeg = np.bincount(np.concatenate([_ceu, _cev]), minlength=len(_V0)).astype(float)
    _cL = np.linalg.norm(_V0[_ceu] - _V0[_cev], axis=1)
    _ch = (np.bincount(_ceu, _cL, len(_V0)) + np.bincount(_cev, _cL, len(_V0))) / np.maximum(_cdeg, 1.0)
    _cscale = 4.0 / (np.maximum(_ch, 1e-12) ** 2 * (2.0 * DEPTH / R ** 2))
    _crows = np.asarray(shape_nodes, dtype=int)
    _ceu_j, _cev_j = jnp.asarray(_ceu), jnp.asarray(_cev)
    _cdeg_j, _cscale_j, _crows_j = (jnp.asarray(_cdeg, dtype=qdtype), jnp.asarray(_cscale, dtype=qdtype),
                                    jnp.asarray(_crows))

    def curv_term(X):
        r2 = X[:, 0] ** 2 + X[:, 1] ** 2
        d = (X[:, 2] - DEPTH * (1.0 - r2 / R ** 2)) / jnp.sqrt(1.0 + _kp ** 2 * r2)
        s = jnp.zeros_like(d).at[_ceu_j].add(d[_cev_j]).at[_cev_j].add(d[_ceu_j])
        e = (d - s / jnp.maximum(_cdeg_j, 1.0)) * _cscale_j
        return jnp.mean(e[_crows_j] ** 2)

    def curv_term_np(X):
        r2 = X[:, 0] ** 2 + X[:, 1] ** 2
        d = (X[:, 2] - DEPTH * (1.0 - r2 / R ** 2)) / np.sqrt(1.0 + _kp ** 2 * r2)
        s = np.bincount(_ceu, d[_cev], len(d)) + np.bincount(_cev, d[_ceu], len(d))
        e = (d - s / np.maximum(_cdeg, 1.0)) * _cscale
        return float(np.mean(e[_crows] ** 2))

    if W_CURV > 0.0:
        print(f"[curv] curvature-sensitive shape term ACTIVE: W_CURV={W_CURV:g} over {len(_crows)} shape nodes",
              flush=True)

    def _extra(X):
        out = 0.0
        if W_MESH > 0.0:
            out = out + W_MESH * mesh_pen(X)
        if W_QUAL > 0.0:
            out = out + W_QUAL * qual_energy(X)
        return out

    def _loss(xyz):
        ax = xyz[apex]
        j_apex = ax[0] ** 2 + ax[1] ** 2 + (ax[2] - target_height) ** 2
        if OBJECTIVE == "normal":
            x, y, z = xyz[shape_rows, 0], xyz[shape_rows, 1], xyz[shape_rows, 2]
            r2 = x ** 2 + y ** 2
            d = (z - DEPTH * (1.0 - r2 / R ** 2)) / jnp.sqrt(1.0 + _kp ** 2 * r2)
            u = xyz[shape_rows] - _seed_j
            ut = u - jnp.sum(u * _nrm_j, axis=1)[:, None] * _nrm_j
            out = (W_NORMAL * jnp.mean(d ** 2) + W_TAN * jnp.mean(jnp.sum(ut ** 2, axis=1))
                   + W_APEX * j_apex + _extra(xyz))
            if W_CURV > 0.0:
                out = out + W_CURV * curv_term(xyz)
            return out
        rho = jnp.sqrt(xyz[shape_rows, 0] ** 2 + xyz[shape_rows, 1] ** 2 + 1e-12)
        j_rad = jnp.mean((rho - target_rho) ** 2)
        j_ax = jnp.mean((xyz[shape_rows, 2] - target_z) ** 2)
        return (W_RAD * j_rad + W_AXIAL * j_ax + W_APEX * j_apex
                + _extra(xyz))

    def objective_np(coords):
        ax = coords[apex]
        j_apex = ax[0] ** 2 + ax[1] ** 2 + (ax[2] - target_height) ** 2
        xf = coords[shape_nodes]
        if OBJECTIVE == "normal":
            r2 = xf[:, 0] ** 2 + xf[:, 1] ** 2
            d = (xf[:, 2] - DEPTH * (1.0 - r2 / R ** 2)) / np.sqrt(1.0 + _kp ** 2 * r2)
            u = xf - _seed_np
            ut = u - np.sum(u * _nrm_np, axis=1)[:, None] * _nrm_np
            out = float(W_NORMAL * np.mean(d ** 2) + W_TAN * np.mean(np.sum(ut ** 2, axis=1))
                        + W_APEX * j_apex + _mpen_np(coords))
            if W_CURV > 0.0:
                out += W_CURV * curv_term_np(np.asarray(coords, dtype=float))
            return out
        rho = np.sqrt(xf[:, 0] ** 2 + xf[:, 1] ** 2)
        j_rad = np.mean((rho - target_rho_np) ** 2)
        j_ax = np.mean((xf[:, 2] - target_z_np) ** 2)
        return float(W_RAD * j_rad + W_AXIAL * j_ax + W_APEX * j_apex + _mpen_np(coords))

    def _mpen_np(coords):
        out = 0.0
        if W_MESH > 0.0:
            out += W_MESH * float(_mesh_pen_np(jnp.asarray(coords, dtype=qdtype)))
        if W_QUAL > 0.0:
            out += W_QUAL * float(_qual_np(jnp.asarray(coords, dtype=qdtype)))
        return out

    def vol(coords):
        return max(enclosed_volume(coords, faces_tri), 1e-9)

    def apex_z_of(c):
        return -np.inf if not np.isfinite(c).all() else float(c[apex, 2])

    def diverged(c):
        return (not np.isfinite(c).all()) or (np.abs(c).max() > 10.0 * target_height)

    def free_of(coords):
        return jnp.asarray(np.asarray(coords)[idx_free], dtype=qdtype)

    def make_model(gas_k, sgn):
        return ConstantMassModelSparse(
            tmax=NATIVE_URS_TMAX, eta=NATIVE_URS_ETA, is_load_local=True, implicit_diff=True,
            itersolve_fn=solver_forward, faces_volume=ft, gas_k=gas_k, sign=sgn,
            unit_faces=unit_faces)

    def mass_xyz(model_, theta, xf_init):
        q = q_full_of(theta)
        xf = model_.equilibrium_iterative_xyz(
            q, bp.xyz_fixed, bp.loads, st, xyz_free_init=xf_init,
            tmax=NATIVE_URS_TMAX, eta=NATIVE_URS_ETA, solver=_SOLVER, implicit_diff=True)
        return model_.nodes_positions(xf, bp.xyz_fixed, st)

    def solve_np(model_, theta, xf_init):
        try:
            return np.asarray(mass_xyz(model_, theta, xf_init))
        except Exception:                                    # noqa: BLE001
            return np.full((n_v, 3), np.inf)

    # ---- warm-start q ----------------------------------------------------------
    if q_initial is None:
        q0 = np.array(CAT_Q0, copy=True)
    elif isinstance(q_initial, dict):
        edge_keys = [(min(u, v), max(u, v)) for u, v in topo["edges"]]
        missing = [k for k in edge_keys if k not in q_initial]
        if missing:
            raise ValueError(f"Warm-start q is missing {len(missing)} edge(s)")
        q0 = np.clip(np.asarray([q_initial[k] for k in edge_keys], dtype=float), CAT_LO, CAT_HI)
    else:
        q0 = np.clip(np.asarray(q_initial, dtype=float), CAT_LO, CAT_HI)

    # Seed: every station at START_Q (q0 is per-edge and has no exact station preimage, so a
    # warm-start q is not supported here; the study never passes one).
    if q_initial is not None:
        raise NotImplementedError("warm-start q is not supported on the station-profile map")
    q0_red = np.clip(np.full(len(GROUP_NAMES), START_Q), LO_RED, HI_RED)
    if Q_MODE == "edge":
        _Vs = np.asarray(topo["vertices"], dtype=float)
        _cot = {}
        for f in np.asarray(faces_tri):
            for m_ in range(3):
                i_, j_, o_ = int(f[m_]), int(f[(m_ + 1) % 3]), int(f[(m_ + 2) % 3])
                u_, v_ = _Vs[i_] - _Vs[o_], _Vs[j_] - _Vs[o_]
                key_ = (min(i_, j_), max(i_, j_))
                _cot[key_] = _cot.get(key_, 0.0) + 0.5 * float(u_ @ v_) / np.linalg.norm(
                    np.cross(u_, v_))
        q0_red = np.clip(np.asarray([_cot[(min(u, v), max(u, v))] for (u, v) in topo["edges"]]),
                         LO_RED, HI_RED)
        if EDGE_SEED == "const":
            q0_red = np.clip(np.full(len(GROUP_NAMES), START_Q), LO_RED, HI_RED)
        print(f"[edge q] {len(q0_red)} per-edge variables, {EDGE_SEED} seed: min {q0_red.min():.3f} "
              f"median {np.median(q0_red):.3f} max {q0_red.max():.3f}", flush=True)
    theta0 = jnp.asarray(np.log(q0_red), dtype=qdtype)

    if coords_initial is None:
        seed_coords = np.array(topo["vertices"], copy=True)
    else:
        seed_coords = np.asarray(coords_initial, dtype=float).copy()
        seed_coords[np.asarray(topo["fixed"], dtype=int)] = \
            topo["vertices"][np.asarray(topo["fixed"], dtype=int)]

    if stage_label:
        print(f"\n{'=' * 78}\n[stage:native_mass] {stage_label}\n{'=' * 78}", flush=True)

    V0 = vol(seed_coords)
    xf_seed = free_of(seed_coords)

    # ---- GAS_K calibration: sign detection + light ramp to the target apex -------------------
    GAS_K = PRESSURE * V0
    c_p = solve_np(make_model(GAS_K, 1.0), theta0, xf_seed)
    c_m = solve_np(make_model(GAS_K, -1.0), theta0, xf_seed)
    sign = 1.0 if apex_z_of(c_p) >= apex_z_of(c_m) else -1.0
    coords0 = c_p if sign > 0 else c_m
    print(f"[native_mass setup] sign {int(sign):+d} inflates (apex +1:{apex_z_of(c_p):.3f} "
          f"-1:{apex_z_of(c_m):.3f}); target apex z={target_height:.3f}", flush=True)
    best_c, best_apex, best_gas = coords0, apex_z_of(coords0), GAS_K
    step = NATIVE_CALIB_STEP
    for _ in range(NATIVE_CALIB_ITERS):
        az = apex_z_of(coords0)
        if az > 1e-6 and abs(az - target_height) <= NATIVE_APEX_TOL_FRAC * target_height:
            break
        saved = GAS_K
        GAS_K *= (step if az < target_height else 1.0 / step)
        trial = solve_np(make_model(GAS_K, sign), theta0, xf_seed)
        if diverged(trial):
            GAS_K = saved
            step = 1.0 + (step - 1.0) * NATIVE_CALIB_STEP_SHRINK
            if step < 1.01:
                break
            continue
        coords0 = trial
        if apex_z_of(coords0) > best_apex:
            best_c, best_apex, best_gas = coords0, apex_z_of(coords0), GAS_K
    GAS_K, coords0 = best_gas, best_c
    print(f"[native_mass calibrate] GAS_K={GAS_K:.4f} sign={int(sign):+d} -> "
          f"apex_z={apex_z_of(coords0):.4f} V={vol(coords0):.4f} (target {target_height:.4f})",
          flush=True)

    # ---- WARM START from a finished run (JAXFDM_WARM_NPZ): its per-edge q and equilibrium as the
    # starting point under the CURRENT objective.  Not a resume: the L-BFGS memory belongs to the old
    # objective and is not carried over.  The gas constant must be the one that run used -- it is
    # recalibrated here deterministically from the same seed, so it is checked, not trusted.
    if WARM_NPZ:
        if Q_MODE != "edge":
            raise NotImplementedError("JAXFDM_WARM_NPZ needs JAXFDM_Q_MODE=edge")
        wz = np.load(WARM_NPZ, allow_pickle=True)
        if not np.array_equal(np.asarray(wz["edges"]), np.asarray(topo["edges"])):
            raise RuntimeError("warm start: edge ordering differs -- not the same mesh")
        gk_w = float(wz["GAS_K"])
        if abs(gk_w / float(GAS_K) - 1.0) > 1e-10:
            raise RuntimeError(f"warm start: GAS_K {gk_w!r} != calibrated {float(GAS_K)!r}")
        q_w = np.asarray(wz["q_categories"], float)[np.asarray(wz["cat_ids"], int)]
        theta0 = jnp.asarray(np.log(np.clip(q_w, LO_RED, HI_RED)), dtype=qdtype)
        q0 = q_w.copy()
        coords0 = np.asarray(wz["vertices"], float)
        print(f"[warm] {WARM_NPZ}: per-edge q + equilibrium adopted (GAS_K {gk_w:.6f} matches); "
              f"L-BFGS memory starts empty under objective '{OBJECTIVE}'", flush=True)

    if diverged(coords0):
        coords0 = seed_coords
    initial_ok = (not diverged(coords0)) and apex_z_of(coords0) > NATIVE_GATE_FRAC * target_height
    coords_start = np.array(coords0, copy=True)
    q_start = np.array(q0, copy=True)
    rms0, max0, *_ = surface_rms(coords0, topo)
    obj0 = objective_np(coords0)
    print(f"[native_mass baseline] {len(GROUP_NAMES)} design vars over {len(CATEGORIES)} edges "
          f"({n_skin} skin) -> obj={obj0:.6f} surfRMS={rms0:.5f} max|d|={max0:.5f} "
          f"apex_z={apex_z_of(coords0):.4f}", flush=True)

    # ---- differentiable constant-mass model (unit_faces=None, gas_k carried in loads) --------
    model = ConstantMassModelSparse(
        tmax=NATIVE_URS_TMAX, eta=NATIVE_URS_ETA, is_load_local=True, implicit_diff=True,
        itersolve_fn=solver_forward, faces_volume=ft, gas_k=1.0, sign=sign, unit_faces=None)

    @jax.jit
    def native_forward(q, gk, xf_init):
        loads = bp.loads._replace(faces=unit_faces * gk)
        xf = model.equilibrium_iterative_xyz(
            q, bp.xyz_fixed, loads, st, xyz_free_init=xf_init,
            tmax=NATIVE_URS_TMAX, eta=NATIVE_URS_ETA, solver=_SOLVER, implicit_diff=False)
        return model.nodes_positions(xf, bp.xyz_fixed, st)

    @jax.jit
    def native_nodal_loads(coords, gk):
        loads = bp.loads._replace(faces=unit_faces * gk)
        return model.nodes_load(coords, loads, st)

    @jax.jit
    def solve_frozen(q, loads_nodes):
        xf = model.nodes_equilibrium(q, bp.xyz_fixed, loads_nodes, st)
        return model.nodes_positions(xf, bp.xyz_fixed, st)

    # ---- Energy + Picard-stability diagnostics -----------------------------------------------
    _eu = np.asarray([u for (u, v) in topo["edges"]], dtype=int)
    _ev = np.asarray([v for (u, v) in topo["edges"]], dtype=int)
    _fixed_rows = np.asarray(topo["fixed"], dtype=int)

    def energy_of(coords, theta_arr, gk_val):
        return total_energy(coords, q_full_np(theta_arr), (_eu, _ev), faces_tri, gk_val)

    @jax.jit
    def _vjp_damped_map(q, gk, x0, v):
        def m(c):
            return ((1.0 - NATIVE_MASS_BETA) * c
                    + NATIVE_MASS_BETA * solve_frozen(q, native_nodal_loads(c, gk)))
        _, pull = jax.vjp(m, x0)
        return pull(v)[0]

    def lambda_max_at(theta_j, gk_float, coords_star, iters=STAB_PI_ITERS, project=None):
        """Dominant eigenvalue of the free-block Picard Jacobian J = K^-1 dF/dx at an
        equilibrium, via power iteration on the damped map's transpose. Returns (lambda, mu, v).
        lambda > 1 = FDM geometric energy-saddle (fold) -- reported, never a constraint."""
        if not np.isfinite(np.asarray(coords_star)).all():
            return np.nan, np.nan, None
        q = q_full_of(theta_j)
        gk = jnp.asarray(gk_float, dtype=qdtype)
        x0 = jnp.asarray(np.asarray(coords_star), dtype=qdtype)
        rng = np.random.default_rng(0)
        v = rng.standard_normal((n_v, 3))
        if project is not None:
            v = project(v)
        v[_fixed_rows] = 0.0
        v /= np.linalg.norm(v)
        mu = mu_prev = np.nan
        for it in range(1, iters + 1):
            try:
                w = np.array(_vjp_damped_map(q, gk, x0, jnp.asarray(v, dtype=qdtype)))
            except Exception:                            # noqa: BLE001
                return np.nan, np.nan, None
            if project is not None:
                w = project(w)
            w[_fixed_rows] = 0.0
            nw = float(np.linalg.norm(w))
            if not np.isfinite(nw) or nw < 1e-14:
                return np.nan, np.nan, None
            mu = float(np.dot(w.ravel(), v.ravel()))
            v = w / nw
            if it >= 5 and abs(mu - mu_prev) < 1e-3:
                break
            mu_prev = mu
        lam = (mu - (1.0 - NATIVE_MASS_BETA)) / NATIVE_MASS_BETA
        return lam, mu, v

    def make_sym_projector(coords_ref):
        """Orthogonal projector onto the axisymmetric (k=0) perturbation subspace."""
        c = np.asarray(coords_ref)
        th = np.arctan2(c[:, 1], c[:, 0])
        cth, sth = np.cos(th), np.sin(th)
        rid = np.asarray(topo["ring_ids"])
        ring_list = [int(i) for i in np.unique(rid)]

        def project(vfield):
            vfield = np.asarray(vfield)
            vr = vfield[:, 0] * cth + vfield[:, 1] * sth
            vt = -vfield[:, 0] * sth + vfield[:, 1] * cth
            vz = vfield[:, 2]
            out = np.zeros_like(vfield)
            for i in ring_list:
                m = rid == i
                if i <= 0:
                    out[m, 2] = vz[m].mean()
                    continue
                mr, mt, mz = vr[m].mean(), vt[m].mean(), vz[m].mean()
                out[m, 0] = mr * cth[m] - mt * sth[m]
                out[m, 1] = mr * sth[m] + mt * cth[m]
                out[m, 2] = mz
            return out

        return project

    log_gk_lo = jnp.asarray(np.log(GK_MIN_FRAC * GAS_K), dtype=qdtype)
    log_gk_hi = jnp.asarray(np.log(GK_MAX_FRAC * GAS_K), dtype=qdtype)

    def loss_and_xyz(theta, log_gk, xf):
        # Damped-Picard constant-mass solve to convergence, warm-started -> ONE value_and_grad
        # is the exact constant-mass gradient (dp/dV feedback included).
        q = q_full_of(theta)
        gk = jnp.exp(jnp.clip(log_gk, log_gk_lo, log_gk_hi)).astype(qdtype)
        loads = bp.loads._replace(faces=unit_faces * gk)
        xf2 = model.equilibrium_iterative_xyz(
            q, bp.xyz_fixed, loads, st, xyz_free_init=xf,
            tmax=NATIVE_URS_TMAX, eta=NATIVE_URS_ETA, solver=_SOLVER, implicit_diff=True)
        xyz = model.nodes_positions(xf2, bp.xyz_fixed, st)
        return _loss(xyz), xyz

    vg = jax.jit(jax.value_and_grad(loss_and_xyz, argnums=(0, 1), has_aux=True))
    loss_fwd = jax.jit(loss_and_xyz)

    @jax.jit
    def native_probe(theta, log_gk, xf):
        # A few extra damped-Picard steps: a settled fixed point barely moves; a drifting
        # (fold/tangle) transient keeps moving. Distinguishes genuine equilibria.
        q = q_full_of(theta)
        gk = jnp.exp(jnp.clip(log_gk, log_gk_lo, log_gk_hi)).astype(qdtype)
        loads = bp.loads._replace(faces=unit_faces * gk)
        xf2 = model.equilibrium_iterative_xyz(
            q, bp.xyz_fixed, loads, st, xyz_free_init=xf,
            tmax=3, eta=NATIVE_URS_ETA, solver=_SOLVER, implicit_diff=False)
        return model.nodes_positions(xf2, bp.xyz_fixed, st)

    # ---- Newton (sparse Sherman-Morrison) refinement + SM-factored adjoint -------------------
    def gk_of(lgk):
        return jnp.exp(jnp.clip(lgk, log_gk_lo, log_gk_hi)).astype(qdtype)

    _n_dof = 3 * idx_free.size
    _newton_wanted = NEWTON_MODE not in ("0", "off", "false", "no")
    NEWTON_ON = _newton_wanted and (NEWTON_MODE in ("1", "force", "on")
                                    or _n_dof <= NEWTON_MAX_DOF)
    newton_refine = None
    newton_grad = None
    if _newton_wanted and not NEWTON_ON:
        print(f"[newton] refinement AUTO-OFF: {_n_dof} dof > NEWTON_MAX_DOF={NEWTON_MAX_DOF} "
              f"(JAXFDM_NEWTON=force to override)", flush=True)
    if NEWTON_ON:
        _EU = jnp.asarray(_eu)
        _EV = jnp.asarray(_ev)
        _IDXF = jnp.asarray(idx_free)

        def _newton_residual(q, gk, xf):
            # Exact nodal force residual R(x) = sum_e q_e(x_i - x_j) - P(x) at the FREE nodes;
            # P includes the full constant-mass coupling p = sign*gk/V, so roots ARE fixed
            # points of the native coupled map.
            xyz = model.nodes_positions(xf, bp.xyz_fixed, st)
            t = q[:, None] * (xyz[_EU] - xyz[_EV])
            f_int = jnp.zeros_like(xyz).at[_EU].add(t).at[_EV].add(-t)
            return (f_int - native_nodal_loads(xyz, gk))[_IDXF]

        def _newton_res_flat(q, gk, xff):
            return _newton_residual(q, gk, xff.reshape(-1, 3)).ravel()

        _newton_res_flat_j = jax.jit(_newton_res_flat)

        @jax.jit
        def _newton_resnorm(q, gk, xf):
            return jnp.mean(jnp.linalg.norm(_newton_residual(q, gk, xf), axis=1))

        _jvp_cols = jax.jit(jax.vmap(
            lambda q, gk, xff, e: jax.jvp(
                lambda x: _newton_res_flat(q, gk, x), (xff,), (e,))[1],
            in_axes=(None, None, None, 0)))

        def _newton_jacobian(q, gk, xff):
            # Exact dense J = dR/dx via CHUNKED batched jvp.
            n = int(xff.size)
            rows = []
            for s in range(0, n, NEWTON_JAC_CHUNK):
                m = min(NEWTON_JAC_CHUNK, n - s)
                e = np.zeros((m, n))
                e[np.arange(m), s + np.arange(m)] = 1.0
                rows.append(np.asarray(_jvp_cols(q, gk, xff, jnp.asarray(e))))
            return np.concatenate(rows, axis=0).T

        def _newton_step_dense(q, gk, xf):
            xff = jnp.asarray(xf).ravel()
            r = np.asarray(_newton_res_flat_j(q, gk, xff))
            J = _newton_jacobian(q, gk, xff)
            delta = np.linalg.lstsq(J, -r, rcond=None)[0]
            return jnp.asarray((np.asarray(xff) + delta).reshape(-1, 3), dtype=qdtype)

        # ---- sparse Sherman-Morrison step (default) ------------------------------------------
        def _pfrozen_res_flat(q, gk, vbar, xff):
            # Residual with pressure FROZEN at pbar = sign*gk/vbar: the live native load
            # rescaled by V(x)/vbar equals pbar*A(x) exactly, so J_s is differentiated from the
            # model's own load assembly (never a hand-derived H_V).
            xyz = model.nodes_positions(xff.reshape(-1, 3), bp.xyz_fixed, st)
            t = q[:, None] * (xyz[_EU] - xyz[_EV])
            f_int = jnp.zeros_like(xyz).at[_EU].add(t).at[_EV].add(-t)
            load = (model.enclosed_volume(xyz) / vbar) * native_nodal_loads(xyz, gk)
            return (f_int - load)[_IDXF].ravel()

        _jvp_frozen = jax.jit(jax.vmap(
            lambda q, gk, vbar, xff, e: jax.jvp(
                lambda x: _pfrozen_res_flat(q, gk, vbar, x), (xff,), (e,))[1],
            in_axes=(None, None, None, None, 0)))

        @jax.jit
        def _vol_and_gradv(xf):
            def _v(xf_):
                return model.enclosed_volume(model.nodes_positions(xf_, bp.xyz_fixed, st))
            return jax.value_and_grad(_v)(xf)

        def _build_sm_pattern():
            # Coupling stencil of J_s: free nodes couple iff they share an EDGE or a FACE.
            # Distance-2 greedy colouring lets one jvp recover many columns at once.
            nfree = idx_free.size
            loc = -np.ones(n_v, dtype=int)
            loc[idx_free] = np.arange(nfree)
            adjn = [{k} for k in range(nfree)]
            for (uu, vv) in topo["edges"]:
                lu_, lv_ = loc[int(uu)], loc[int(vv)]
                if lu_ >= 0 and lv_ >= 0:
                    adjn[lu_].add(lv_)
                    adjn[lv_].add(lu_)
            for f in topo["faces_quad"]:
                ls = [loc[int(vv)] for vv in f if loc[int(vv)] >= 0]
                for a_ in ls:
                    adjn[a_].update(ls)
            order = sorted(range(nfree), key=lambda k: -len(adjn[k]))
            color = -np.ones(nfree, dtype=int)
            for k in order:
                used = {color[nb2] for nb in adjn[k] for nb2 in adjn[nb] if color[nb2] >= 0}
                c = 0
                while c in used:
                    c += 1
                color[k] = c
            ncol = int(color.max()) + 1
            indptr, indices, slot_ids = [0], [], []
            for nl in range(nfree):
                rows_nodes = np.asarray(sorted(adjn[nl]), dtype=int)
                rows_dof = (rows_nodes[:, None] * 3 + np.arange(3)[None, :]).ravel()
                for k3 in range(3):
                    indices.append(rows_dof)
                    slot_ids.append(np.full(rows_dof.size, color[nl] * 3 + k3, dtype=int))
                    indptr.append(indptr[-1] + rows_dof.size)
            tang = np.zeros((ncol * 3, nfree * 3))
            cols = np.arange(nfree) * 3
            for k3 in range(3):
                tang[color * 3 + k3, cols + k3] = 1.0
            return (ncol, jnp.asarray(tang), np.concatenate(indices),
                    np.asarray(indptr), np.concatenate(slot_ids))

        try:
            import scipy.sparse as _sp
            import scipy.sparse.linalg as _spl
            _have_scipy = True
        except Exception:                                # noqa: BLE001
            _have_scipy = False
        _SM = {"ok": _have_scipy and NEWTON_LINSOLVE != "dense",
               "pat": None, "verified": False}

        def _newton_step_sm(q, gk, xf):
            if _SM["pat"] is None:
                _SM["pat"] = _build_sm_pattern()
                print(f"[newton] sparse SM pattern: {_SM['pat'][0]} colours -> "
                      f"{3 * _SM['pat'][0]} jvp tangents + 1 splu per iteration", flush=True)
            ncol, tang, indices, indptr, slot_ids = _SM["pat"]
            xfj = jnp.asarray(xf)
            xff = xfj.ravel()
            r = np.asarray(_newton_res_flat_j(q, gk, xff))
            v_, gv = _vol_and_gradv(xfj)
            vbar = float(v_)
            pbar = float(sign) * float(gk) / vbar
            P_free = np.asarray(native_nodal_loads(
                model.nodes_positions(xfj, bp.xyz_fixed, st), gk))[idx_free].ravel()
            u_vec = P_free / pbar
            w_vec = (float(sign) * float(gk) / vbar ** 2) * np.asarray(gv).ravel()
            compressed = np.asarray(_jvp_frozen(q, gk, jnp.asarray(vbar, dtype=qdtype),
                                                xff, tang))
            Js = _sp.csc_matrix((compressed[slot_ids, indices], indices, indptr),
                                shape=(r.size, r.size))
            lu_ = _spl.splu(Js)
            if not _SM["verified"]:
                rng = np.random.default_rng(0)
                vt = rng.standard_normal(r.size)
                jv_full = np.asarray(_jvp_cols(q, gk, xff, jnp.asarray(vt[None, :])))[0]
                jv_sm = Js @ vt + u_vec * float(w_vec @ vt)
                rel = (np.linalg.norm(jv_sm - jv_full)
                       / max(np.linalg.norm(jv_full), 1e-30))
                ok = rel < 1e-8
                print(f"[newton] SM split verification: |J_sm v - J v|/|J v| = {rel:.2e} "
                      f"{'OK' if ok else '-- FALLING BACK TO DENSE'}", flush=True)
                if not ok:
                    _SM["ok"] = False
                    return _newton_step_dense(q, gk, xf)
                _SM["verified"] = True
            t1 = lu_.solve(-r)
            t2 = lu_.solve(u_vec)
            den = 1.0 + float(w_vec @ t2)
            den = NEWTON_DEN_TOL if den == 0.0 else np.sign(den) * max(abs(den), NEWTON_DEN_TOL)
            delta = t1 - (float(w_vec @ t1) / den) * t2
            if not np.isfinite(delta).all():
                return _newton_step_dense(q, gk, xf)
            return jnp.asarray((np.asarray(xff) + delta).reshape(-1, 3), dtype=qdtype)

        def _newton_step(q, gk, xf):
            if _SM["ok"]:
                try:
                    return _newton_step_sm(q, gk, xf)
                except Exception:                        # noqa: BLE001
                    _SM["ok"] = False
                    print("[newton] sparse SM step raised -- dense fallback for the rest "
                          "of the run", flush=True)
            return _newton_step_dense(q, gk, xf)

        def newton_refine(q, gk, xf, tol=NEWTON_TOL, tmax=NEWTON_TMAX_IT):
            """Newton-polish a near-equilibrium to machine precision. Contraction-independent:
            converges where the damped Picard cannot (lambda_sym >= 1). Returns (xf*, mean
            nodal residual, iterations). The CALLER must health-gate the result."""
            res = float(_newton_resnorm(q, gk, xf))
            if res < tol:
                return xf, res, 0
            for it in range(1, tmax + 1):
                xf_new = _newton_step(q, gk, xf)
                if not np.isfinite(np.asarray(xf_new)).all():
                    return xf, res, it - 1
                xf = xf_new
                res = float(_newton_resnorm(q, gk, xf))
                if res < tol:
                    return xf, res, it
            return xf, res, tmax

        # ---- SM-factored ADJOINT: Newton IS the differentiable forward (NEWTON_GRAD) ----------
        def _sm_build(q, gk, xff):
            if _SM["pat"] is None:
                _SM["pat"] = _build_sm_pattern()
            _, tang, indices, indptr, slot_ids = _SM["pat"]
            xfj = jnp.asarray(xff).reshape(-1, 3)
            v_, gv = _vol_and_gradv(xfj)
            vbar = float(v_)
            pbar = float(sign) * float(gk) / vbar
            P_free = np.asarray(native_nodal_loads(
                model.nodes_positions(xfj, bp.xyz_fixed, st), gk))[idx_free].ravel()
            u_vec = P_free / pbar
            w_vec = (float(sign) * float(gk) / vbar ** 2) * np.asarray(gv).ravel()
            compressed = np.asarray(_jvp_frozen(q, gk, jnp.asarray(vbar, dtype=qdtype),
                                                jnp.asarray(xff), tang))
            n = int(xff.size)
            Js = _sp.csc_matrix((compressed[slot_ids, indices], indices, indptr), shape=(n, n))
            return _spl.splu(Js), u_vec, w_vec

        def newton_value_and_grad(theta, log_gk, xf):
            """Newton (SM) primal + SM-factored adjoint -> ((loss, xyz), (dL/dtheta,
            dL/dlog_gk)). Drop-in replacement for the damped `vg` with a FOLD-ROBUST gradient."""
            q = q_full_of(theta)
            gk = gk_of(log_gk)
            xf_star, _res, _nit = newton_refine(q, gk, xf)
            xff = jnp.asarray(xf_star).ravel()
            xyz = model.nodes_positions(jnp.asarray(xf_star), bp.xyz_fixed, st)
            loss = _loss(xyz)
            dLdxf = np.asarray(jax.grad(
                lambda xf_: _loss(model.nodes_positions(xf_, bp.xyz_fixed, st)))(
                    jnp.asarray(xf_star))).ravel()
            lu_, u_vec, w_vec = _sm_build(q, gk, xff)
            s1 = lu_.solve(dLdxf, trans="T")
            s2 = lu_.solve(w_vec, trans="T")
            denT = 1.0 + float(u_vec @ s2)
            denT = (NEWTON_DEN_TOL if denT == 0.0
                    else np.sign(denT) * max(abs(denT), NEWTON_DEN_TOL))
            lam = s1 - (float(u_vec @ s1) / denT) * s2
            _, vjp_fn = jax.vjp(
                lambda th, lgk: _newton_res_flat(q_full_of(th), gk_of(lgk), xff),
                theta, log_gk)
            g_theta, g_gk = vjp_fn(jnp.asarray(-lam, dtype=qdtype))
            return (loss, xyz), (g_theta, g_gk)

        def newton_loss_fwd(theta, log_gk, xf):
            # Newton primal (exact loss for the Armijo test; no adjoint). Warm-started.
            q = q_full_of(theta)
            gk = gk_of(log_gk)
            xf_star, _res, _nit = newton_refine(q, gk, xf)
            xyz = model.nodes_positions(jnp.asarray(xf_star), bp.xyz_fixed, st)
            return _loss(xyz), xyz

        _NG = {"ok": bool(NEWTON_GRAD and _SM["ok"]), "verified": False}

        def _newton_grad_checked(theta, log_gk, xf):
            # First call cross-checks the SM adjoint vs the damped gradient on the (stable) seed.
            try:
                (loss, xyz), (g_th, g_gk) = newton_value_and_grad(theta, log_gk, xf)
                if not _NG["verified"]:
                    (_ld, _), (gd_th, gd_gk) = vg(theta, log_gk, xf)
                    gn = np.concatenate([np.asarray(g_th, float), [float(g_gk)]])
                    gd = np.concatenate([np.asarray(gd_th, float), [float(gd_gk)]])
                    cos = float(gn @ gd / (np.linalg.norm(gn) * np.linalg.norm(gd) + 1e-30))
                    ok = bool(np.isfinite(cos) and cos > 0.99)
                    # JAXFDM_NEWTON_GRAD_FORCE=1: keep the SM adjoint even when the cross-check fails.  The damped
                    # reference is the implicit adjoint of the Picard map, whose linear solve can stop unconverged
                    # (jax_fdm prints the tally at exit); near a nearly singular equilibrium it is the reference that
                    # is unreliable, and falling back to it stalls the optimisation (L2 stage 3, 2026-10-19).
                    print(f"[newton grad] SM adjoint cross-check vs damped (stable seed): "
                          f"cos={cos:.6f} {'OK' if ok else ('-- cross-check FAILED, SM adjoint KEPT (JAXFDM_NEWTON_GRAD_FORCE=1)' if NEWTON_GRAD_FORCE else '-- FALLING BACK to damped vg')}",
                          flush=True)
                    _NG["verified"] = True
                    if not ok and not NEWTON_GRAD_FORCE:
                        _NG["ok"] = False
                        return vg(theta, log_gk, xf)
                return (loss, xyz), (g_th, g_gk)
            except Exception as _exc:                          # noqa: BLE001
                print(f"[newton grad] SM adjoint raised ({_exc}) -- damped gradient fallback",
                      flush=True)
                _NG["ok"] = False
                return vg(theta, log_gk, xf)

        if _NG["ok"]:
            newton_grad = _newton_grad_checked
            loss_fwd = newton_loss_fwd

        print(f"[newton] Newton refinement ACTIVE ({_n_dof} dof, linsolve="
              f"{'sparse Sherman-Morrison' if _SM['ok'] else 'chunked-dense SVD'}, gradient="
              f"{'SM-adjoint (Newton IS the differentiable forward)' if _NG['ok'] else 'damped-Picard vg'})",
              flush=True)

    _psym_state = (make_sym_projector(topo["vertices"])
                   if (AXISYM_Q and AXISYM_STATE_PROJECT) else None)
    if _psym_state is not None:
        coords0 = _psym_state(coords0)
        print("[axisym] state projection ACTIVE: chained geometry re-symmetrised every "
              "accepted step", flush=True)
    theta = theta0
    log_gk = jnp.asarray(np.log(GAS_K), dtype=qdtype)
    best, best_theta, best_log_gk, coords_best = obj0, theta, log_gk, coords0
    history, shape_rms_history, shape_max_history = [obj0], [rms0], [max0]
    xf_cur = free_of(coords0)
    coords_cur = np.array(coords0, copy=True)
    V_ref = vol(coords0)
    V_tgt = max(enclosed_volume(np.asarray(topo["vertices"], dtype=float), faces_tri), 1e-9)
    vol_floor = NATIVE_VOL_FLOOR_FRAC * min(V_ref, V_tgt)
    lr_cur, last_improve, n_reject = LR_, 0, 0
    _stab_hits = 0
    _lb_pairs, _lb_prev, _lb_skip_pair = [], None, False
    _lb_lo = (np.concatenate([LOG_LO_NP, [float(np.asarray(log_gk_lo))]])
              if OPTIMISE_GAS_K else np.array(LOG_LO_NP, dtype=float))
    _lb_hi = (np.concatenate([LOG_HI_NP, [float(np.asarray(log_gk_hi))]])
              if OPTIMISE_GAS_K else np.array(LOG_HI_NP, dtype=float))
    print(f"[lbfgs] stepper: Sobolev-metric L-BFGS (H0=gamma*B^-1, m={LBFGS_M}); "
          f"empty history = pure Sobolev descent", flush=True)
    collapse_z = NATIVE_COLLAPSE_FRAC * target_height
    _trace = os.environ.get("JAXFDM_TRACE") == "1"
    if _trace:
        _roles = np.array([c.split("::")[0].split("@")[0] for c in GROUP_NAMES])
        _hoop = _roles == "hoop"

    if STAB_EVERY > 0:
        lam0, mu0, v0 = lambda_max_at(theta0, float(GAS_K), coords0)
        print(f"[stab] baseline: Pi={energy_of(coords0, theta0, float(GAS_K)):.4f} "
              f"lambda_full={lam0:.4f} mu_damped={mu0:.4f}", flush=True)
        if _psym_state is not None and v0 is not None and np.isfinite(lam0):
            lam_s0, mu_s0, _v = lambda_max_at(theta0, float(GAS_K), coords0,
                                              project=make_sym_projector(coords0))
            print(f"[stab] baseline SYMMETRIC-subspace: lambda_sym={lam_s0:.4f} "
                  f"(constant-mass fold family; <1 = below the geometric fold)", flush=True)

    # ---- resume from a checkpoint ---------------------------------------------------------------
    STEP0, _resumed = 0, False
    if RESUME_NPZ:
        ck = np.load(RESUME_NPZ, allow_pickle=True)
        if abs(float(ck["GAS_K"]) / float(GAS_K) - 1.0) > 1e-12:
            raise RuntimeError(f"resume: GAS_K {float(ck['GAS_K'])!r} != calibrated {float(GAS_K)!r}")
        if not np.array_equal(np.asarray(ck["edges"]), np.asarray(topo["edges"])):
            raise RuntimeError("resume: edge ordering differs from this mesh -- not the same run")
        theta = jnp.asarray(np.asarray(ck["theta"], float), dtype=qdtype)
        log_gk = jnp.asarray(float(ck["log_gk"]), dtype=qdtype)
        coords_cur = np.asarray(ck["coords_cur"], float)
        xf_cur = free_of(coords_cur)
        best = float(ck["best"])
        best_theta = jnp.asarray(np.asarray(ck["best_theta"], float), dtype=qdtype)
        best_log_gk = jnp.asarray(float(ck["best_log_gk"]), dtype=qdtype)
        coords_best = np.asarray(ck["coords_best"], float)
        history = [float(v) for v in ck["history"]]
        shape_rms_history = [float(v) for v in ck["shape_rms_history"]]
        shape_max_history = [float(v) for v in ck["shape_max_history"]]
        lr_cur = float(ck["lr_cur"])
        last_improve = 1 - int(ck["since_improve"])       # loop step 1 == the saved step
        n_reject, _stab_hits = int(ck["n_reject"]), int(ck["stab_hits"])
        _S, _Y, _RHO = np.asarray(ck["lbfgs_s"]), np.asarray(ck["lbfgs_y"]), np.asarray(ck["lbfgs_rho"])
        _lb_pairs = [(_S[i], _Y[i], float(_RHO[i])) for i in range(_S.shape[0])]
        _px, _pg = np.asarray(ck["prev_x"]), np.asarray(ck["prev_g"])
        _lb_prev = (_px, _pg) if _px.size else None
        _lb_skip_pair = bool(ck["skip"])
        STEP0, _resumed = int(ck["step"]) - 1, True
        print(f"[resume] {RESUME_NPZ}: step {int(ck['step'])}, {len(_lb_pairs)} L-BFGS pairs, "
              f"trust cap {lr_cur:.4g}, best {best:.6e}", flush=True)
        if RESUME_RESET:                                  # objective changed: old best/pairs do not apply
            best = float(objective_np(coords_cur))
            best_theta, best_log_gk, coords_best = theta, log_gk, np.array(coords_cur, copy=True)
            _lb_pairs, _lb_prev, _lb_skip_pair = [], None, False
            print(f"[resume] RESET under the current objective: best re-scored at the resumed iterate = "
                  f"{best:.6e}; L-BFGS pairs dropped", flush=True)

    def _save_ckpt(gabs_now, gstep_now):
        """CURRENT iterate + full L-BFGS state, written atomically (tmp + replace)."""
        nx = len(np.asarray(theta))
        S_ = np.asarray([p_[0] for p_ in _lb_pairs]).reshape(-1, nx)
        Y_ = np.asarray([p_[1] for p_ in _lb_pairs]).reshape(-1, nx)
        R_ = np.asarray([p_[2] for p_ in _lb_pairs], float)
        tmp = os.path.join(TEMP_DIR, "ckpt_tmp.npz")
        np.savez(tmp, step=gabs_now, theta=np.asarray(theta, float),
                 log_gk=float(np.asarray(log_gk)), GAS_K=float(GAS_K),
                 coords_cur=np.asarray(coords_cur), best=float(best),
                 best_theta=np.asarray(best_theta, float),
                 best_log_gk=float(np.asarray(best_log_gk)), coords_best=np.asarray(coords_best),
                 history=np.asarray(history), shape_rms_history=np.asarray(shape_rms_history),
                 shape_max_history=np.asarray(shape_max_history), lr_cur=float(lr_cur),
                 since_improve=int(gstep_now - last_improve), n_reject=int(n_reject),
                 stab_hits=int(_stab_hits), lbfgs_s=S_, lbfgs_y=Y_, lbfgs_rho=R_,
                 prev_x=(np.asarray(_lb_prev[0]) if _lb_prev is not None else np.zeros(0)),
                 prev_g=(np.asarray(_lb_prev[1]) if _lb_prev is not None else np.zeros(0)),
                 skip=bool(_lb_skip_pair), edges=np.asarray(topo["edges"]),
                 # for plotting straight from the checkpoint (plot_paraboloid_edge_cd.py)
                 vertices=np.asarray(coords_best), q=q_full_np(best_theta),
                 faces_tri=faces_tri, fixed=np.asarray(topo["fixed"], dtype=int),
                 seed_vertices=np.asarray(topo["vertices"]))
        os.replace(tmp, os.path.join(TEMP_DIR, "ckpt.npz"))

    _hist_f = None
    if SNAP_EVERY > 0 or CKPT_EVERY > 0:
        os.makedirs(TEMP_DIR, exist_ok=True)
        _floor = surface_error_area(np.asarray(topo["vertices"], float), faces_tri)
        _hp = os.path.join(TEMP_DIR, "history.csv")
        if _resumed and os.path.exists(_hp):                # keep only rows up to the saved step
            _rows = open(_hp).read().splitlines()
            _keep = [_rows[0]] + [r_ for r_ in _rows[1:] if int(r_.split(",")[0]) <= STEP0 + 1]
            _hist_f = open(_hp, "w")
            _hist_f.write("\n".join(_keep) + "\n")
        else:
            _hist_f = open(_hp, "w")
            _hist_f.write("step,obj,best,nodal_rms,surf_err,surf_floor,lr,time_s" + "\n")
    t_loop = time.perf_counter()
    gstep = 0
    for gstep in range(1, N_OUTER_ - STEP0 + 1):
        gabs = gstep + STEP0
        _reentry = _resumed and gstep == 1                  # same point as the saved step
        try:
            if newton_grad is not None:
                (loss_val, xyz), (g, g_gk) = newton_grad(theta, log_gk, xf_cur)
            else:
                (loss_val, xyz), (g, g_gk) = vg(theta, log_gk, xf_cur)
        except Exception:                                    # noqa: BLE001
            break
        coords = np.asarray(xyz)
        try:
            coords_probe = np.asarray(native_probe(theta, log_gk, free_of(coords)))
            conv_res = (np.linalg.norm(coords_probe - coords)
                        / max(np.linalg.norm(coords_probe), 1e-12))
        except Exception:                                    # noqa: BLE001
            conv_res = np.inf
        az_s, V_s = apex_z_of(coords), vol(coords)
        collapsed = (az_s < collapse_z) or (V_s < vol_floor)
        overinflated = (az_s > NATIVE_OVERINFLATE_FRAC * target_height) \
            or (V_s > NATIVE_VOL_CEIL_FRAC * V_ref)
        g_ok = np.isfinite(np.asarray(g)).all() and np.isfinite(float(g_gk))
        runaway = conv_res > NATIVE_RUNAWAY_TOL
        # Newton supersedes the movement-based runaway reject: near lambda=1 the damped probe
        # keeps moving even at a genuine equilibrium. Let Newton converge to a true fixed point.
        if (runaway and newton_refine is not None and np.isfinite(coords).all()
                and not collapsed and not overinflated):
            try:
                _xf_nr, _res_nr, _it_nr = newton_refine(
                    q_full_of(theta), gk_of(log_gk), free_of(coords))
                _c_nr = np.asarray(model.nodes_positions(_xf_nr, bp.xyz_fixed, st))
                _az, _V = apex_z_of(_c_nr), vol(_c_nr)
                if (_res_nr < 100.0 * NEWTON_TOL and np.isfinite(_c_nr).all()
                        and not diverged(_c_nr) and _az >= collapse_z and _V >= vol_floor
                        and _az <= NATIVE_OVERINFLATE_FRAC * target_height
                        and _V <= NATIVE_VOL_CEIL_FRAC * V_ref):
                    coords, az_s, V_s, runaway = _c_nr, _az, _V, False
                    collapsed = overinflated = False
                    print(f"[newton] step {gstep:3d}: runaway conv_res={conv_res:.1e} "
                          f"-> Newton converged to genuine equilibrium (res={_res_nr:.1e}, "
                          f"{_it_nr} iters); adopted", flush=True)
            except Exception:                            # noqa: BLE001
                pass
        bad = diverged(coords) or collapsed or overinflated or (not g_ok) or runaway
        if bad:
            n_reject += 1
            if _trace:
                why = ("diverged" if diverged(coords) else "runaway" if runaway
                       else "collapsed" if collapsed else "over-inflated" if overinflated
                       else "bad-grad")
                print(f"[step {gstep:3d}] REJECT {why} apex={az_s:.3f} V={V_s:.1f} "
                      f"conv_res={conv_res:.1e} -> shrink q step lr={lr_cur * 0.5:.5f}",
                      flush=True)
            if STAB_EVERY > 0:
                lam_b, _mu_b, _vb = lambda_max_at(
                    best_theta, float(np.exp(float(np.asarray(best_log_gk)))), coords_best)
                print(f"[stab] reject: banked-best lambda_max={lam_b:.4f} "
                      f"(1 = FDM geometric fold, not a real-material limit)", flush=True)
            theta, log_gk = best_theta, best_log_gk
            xf_cur, coords_cur = free_of(coords_best), np.array(coords_best, copy=True)
            lr_cur *= 0.5
            _lb_pairs, _lb_prev, _lb_skip_pair = [], None, False
            if lr_cur < 1e-4:
                break
            continue
        # ACCEPT: strip fresh k!=0 rounding content, then measure.
        if _psym_state is not None:
            coords = _psym_state(coords)
        lam_acc = lam_emp = lam_symc = np.nan
        _gov_shrunk = False
        if STAB_EVERY > 0 and gabs % STAB_EVERY == 0:
            gk_now = float(np.exp(float(np.asarray(log_gk))))
            lam_acc, _mu_acc, _va = lambda_max_at(theta, gk_now, coords)
            if _psym_state is not None:
                lam_symc, _mu_sc, _vsc = lambda_max_at(theta, gk_now, coords,
                                                       project=_psym_state)
            try:
                probe1 = np.asarray(native_probe(theta, log_gk, free_of(coords)))
                probe2 = np.asarray(native_probe(theta, log_gk, free_of(probe1)))
                d1 = float(np.linalg.norm(probe1 - coords))
                d2 = float(np.linalg.norm(probe2 - probe1))
                if d1 / max(float(np.linalg.norm(coords)), 1e-12) < 1e-12:
                    lam_emp = 0.0
                elif np.isfinite(d2):
                    mu_emp = (d2 / max(d1, 1e-15)) ** (1.0 / 3.0)
                    lam_emp = (mu_emp - (1.0 - NATIVE_MASS_BETA)) / NATIVE_MASS_BETA
            except Exception:                            # noqa: BLE001
                pass
            print(f"[stab] step {gstep:3d}: Pi={energy_of(coords, theta, gk_now):.4f} "
                  f"lambda_full={lam_acc:.4f} lambda_sym={lam_symc:.4f} "
                  f"lambda_excited={lam_emp:.4f}", flush=True)
        if _psym_state is not None:
            lam_ctl = lam_symc if np.isfinite(lam_symc) else lam_emp
        else:
            lam_ctl = lam_acc if np.isfinite(lam_acc) else lam_emp
        # FOLD VETO (default OFF: crossing the fold is correct; kept for experiments).
        if FOLD_VETO and np.isfinite(lam_ctl) and lam_ctl >= LAMBDA_VETO:
            n_reject += 1
            print(f"[stab] FOLD VETO step {gstep:3d}: lambda_ctl={lam_ctl:.4f} >= "
                  f"{LAMBDA_VETO:g} -- reject + revert (lr={lr_cur * 0.5:.5f})", flush=True)
            theta, log_gk = best_theta, best_log_gk
            xf_cur, coords_cur = free_of(coords_best), np.array(coords_best, copy=True)
            lr_cur *= 0.5
            _lb_pairs, _lb_prev, _lb_skip_pair = [], None, False
            if lr_cur < 1e-4:
                break
            continue
        if STAB_GOVERN and np.isfinite(lam_ctl) and lam_ctl > LAMBDA_HARD:
            _stab_hits = 0
            lr_cur *= 0.5
            _gov_shrunk = True
            print(f"[stab] governor: lambda_ctl={lam_ctl:.4f} > HARD {LAMBDA_HARD:g} -- "
                  f"shrink q step lr={lr_cur:.5f}", flush=True)
        elif STAB_GOVERN and np.isfinite(lam_ctl) and lam_ctl > LAMBDA_SAFE:
            _stab_hits += 1
            if _stab_hits >= LAMBDA_HITS:
                _stab_hits = 0
                lr_cur *= 0.5
                _gov_shrunk = True
                print(f"[stab] governor: lambda_ctl={lam_ctl:.4f} > {LAMBDA_SAFE:g} for "
                      f"{LAMBDA_HITS} consecutive steps -- shrink q step lr={lr_cur:.5f}",
                      flush=True)
        else:
            _stab_hits = 0
            if (not (STAB_GOVERN and np.isfinite(lam_ctl) and lam_ctl > LAMBDA_GROW)
                    and not _reentry):
                lr_cur = min(LR_, lr_cur * NATIVE_LR_RECOVER)
        xf_cur = free_of(coords)
        coords_cur = np.array(coords, copy=True)
        obj = objective_np(coords)
        if _trace:
            _hq = float(np.exp(np.clip(np.asarray(theta)[_hoop],
                                       LOG_LO_NP[_hoop], LOG_HI_NP[_hoop])).mean())
            _slow = "  [slow-conv]" if conv_res > NATIVE_ACCEPT_RES else ""
            print(f"[step {gstep:3d}] obj={obj:.6f} lr={lr_cur:.5f} hoop_q_mean={_hq:.4f} "
                  f"gas_k={float(np.exp(np.asarray(log_gk))):.2f} apex={apex_z_of(coords):.4f} "
                  f"conv_res={conv_res:.1e} best={best:.6f}{_slow}", flush=True)
        rms, mx, *_ = surface_rms(coords, topo)
        if not _reentry:
            history.append(obj)
            shape_rms_history.append(rms)
            shape_max_history.append(mx)
        if obj < best - IMPROVE_TOL_:
            best, best_theta, best_log_gk, coords_best, last_improve = \
                obj, theta, log_gk, coords, gstep
        _stop = os.path.exists(os.path.join(TEMP_DIR, "STOP"))
        if _hist_f is not None and not _reentry:
            _hist_f.write(f"{gabs},{obj:.10e},{best:.10e},{rms:.10e},"
                          f"{surface_error_area(coords, faces_tri):.10e},{_floor:.10e},{lr_cur:.4e},"
                          f"{time.perf_counter() - t_loop:.1f}" + "\n")
            _hist_f.flush()
        if SNAP_EVERY > 0 and not _reentry and gabs % SNAP_EVERY == 0:
            np.savez(os.path.join(TEMP_DIR, "snap_latest.npz"), step=gabs,
                     vertices=coords_best, q=q_full_np(best_theta),
                     GAS_K=float(np.exp(float(np.asarray(best_log_gk)))),
                     edges=np.asarray(topo["edges"]), faces_tri=faces_tri,
                     seed_vertices=np.asarray(topo["vertices"]),
                     fixed=np.asarray(topo["fixed"], dtype=int))
        if CKPT_EVERY > 0 and not _reentry and (gabs % CKPT_EVERY == 0 or _stop
                                                 or gstep == N_OUTER_ - STEP0):
            _save_ckpt(gabs, gstep)
        if _stop:
            print(f"[stop] STOP file found at step {gabs} -- finishing cleanly", flush=True)
            break
        if gstep - last_improve >= PATIENCE_:
            break
        # ---- parameter update: Sobolev-metric L-BFGS ----------------------------------------
        g_now = (np.concatenate([np.asarray(g, dtype=float), [float(g_gk)]])
                 if OPTIMISE_GAS_K else np.asarray(g, dtype=float).copy())
        x_now = (np.concatenate([np.asarray(theta, dtype=float),
                                 [float(np.asarray(log_gk))]])
                 if OPTIMISE_GAS_K else np.asarray(theta, dtype=float).copy())
        if _lb_prev is not None:
            s_p, y_p = x_now - _lb_prev[0], g_now - _lb_prev[1]
            sy = float(s_p @ y_p)
            if (not _lb_skip_pair) and sy > LBFGS_CURV * float(
                    np.linalg.norm(s_p) * np.linalg.norm(y_p)):
                _lb_pairs.append((s_p, y_p, 1.0 / sy))
                if len(_lb_pairs) > LBFGS_M:
                    _lb_pairs.pop(0)
        _lb_prev, _lb_skip_pair = (x_now, g_now), False
        d = _lbfgs_direction(g_now, _lb_pairs, lr_cur)
        alpha = min(1.0, LBFGS_CAP * lr_cur / max(float(np.abs(d).max()), 1e-30))
        f0, accepted, n_ls = float(loss_val), False, 0
        _ls_why = {"ascent": 0, "solve": 0, "health": 0, "armijo": 0, "unsettled": 0}
        for n_ls in range(1, LBFGS_LS_MAX + 1):
            x_t = np.clip(x_now + alpha * d, _lb_lo, _lb_hi)
            step_t = x_t - x_now
            slope = float(g_now @ step_t)
            if slope >= 0.0:
                _ls_why["ascent"] += 1
                alpha *= 0.5
                continue
            th_t = jnp.asarray(x_t[:-1] if OPTIMISE_GAS_K else x_t, dtype=qdtype)
            lgk_t = jnp.asarray(x_t[-1], dtype=qdtype) if OPTIMISE_GAS_K else log_gk
            try:
                f_t, xyz_t = loss_fwd(th_t, lgk_t, xf_cur)
                f_t, c_t = float(f_t), np.asarray(xyz_t)
            except Exception:                            # noqa: BLE001
                _ls_why["solve"] += 1
                alpha *= 0.5
                continue
            az_t, V_t = apex_z_of(c_t), vol(c_t)
            healthy = ((not diverged(c_t)) and az_t >= collapse_z and V_t >= vol_floor
                       and az_t <= NATIVE_OVERINFLATE_FRAC * target_height
                       and V_t <= NATIVE_VOL_CEIL_FRAC * V_ref)
            if not (healthy and np.isfinite(f_t)):
                _ls_why["health"] += 1
                alpha *= 0.5
                continue
            if f_t > f0 + 1e-4 * slope:
                _ls_why["armijo"] += 1
                alpha *= 0.5
                continue
            # Settledness gate: near lambda~1 a large trial can return an under-settled state
            # whose fit loss transiently beats the true equilibrium's.
            try:
                c_p = np.asarray(native_probe(th_t, lgk_t, free_of(c_t)))
                conv_t = (np.linalg.norm(c_p - c_t)
                          / max(np.linalg.norm(c_p), 1e-12))
            except Exception:                            # noqa: BLE001
                conv_t = np.inf
            if conv_t > NATIVE_ACCEPT_RES:
                # NEWTON RESCUE: Newton-polish the under-settled trial to the exact equilibrium
                # and re-judge health + Armijo on the true loss.
                if (newton_refine is not None and np.isfinite(lam_ctl)
                        and lam_ctl >= NEWTON_LAMBDA):
                    xf_r, res_r, it_r = newton_refine(q_full_of(th_t), gk_of(lgk_t),
                                                      free_of(c_t))
                    c_r = np.asarray(model.nodes_positions(xf_r, bp.xyz_fixed, st))
                    if res_r < 100.0 * NEWTON_TOL and np.isfinite(c_r).all():
                        f_r = objective_np(c_r)
                        az_r, V_r = apex_z_of(c_r), vol(c_r)
                        ok_r = ((not diverged(c_r)) and az_r >= collapse_z
                                and V_r >= vol_floor
                                and az_r <= NATIVE_OVERINFLATE_FRAC * target_height
                                and V_r <= NATIVE_VOL_CEIL_FRAC * V_ref)
                        if ok_r and f_r <= f0 + 1e-4 * slope:
                            f_t, c_t = f_r, c_r
                            print(f"[newton] step {gstep:3d}: rescued under-settled trial "
                                  f"(res={res_r:.1e} in {it_r} iters, loss {f_r:.6f})",
                                  flush=True)
                            accepted = True
                            break
                _ls_why["unsettled"] += 1
                alpha *= 0.5
                continue
            accepted = True
            break
        if accepted:
            if float(np.abs(x_now + alpha * d - x_t).max()) > 0.05 * float(
                    np.abs(step_t).max()):
                _lb_skip_pair = True
            theta = jnp.asarray(x_t[:-1] if OPTIMISE_GAS_K else x_t, dtype=qdtype)
            if OPTIMISE_GAS_K:
                log_gk = jnp.asarray(x_t[-1], dtype=qdtype)
            # Newton warm-start polish so the next gradient is taken at a genuine fixed point.
            if (newton_refine is not None and np.isfinite(lam_ctl)
                    and lam_ctl >= NEWTON_LAMBDA):
                xf_r, res_r, it_r = newton_refine(q_full_of(theta), gk_of(log_gk),
                                                  free_of(c_t))
                c_r = np.asarray(model.nodes_positions(xf_r, bp.xyz_fixed, st))
                if (res_r < 100.0 * NEWTON_TOL and np.isfinite(c_r).all()
                        and not diverged(c_r)):
                    c_t = c_r
                    if _trace:
                        print(f"[newton] step {gstep:3d}: warm start polished to "
                              f"res={res_r:.1e} ({it_r} iters)", flush=True)
            if _psym_state is not None:
                c_t = _psym_state(c_t)
            xf_cur = free_of(c_t)
            coords_cur = np.array(c_t, copy=True)
            if _trace:
                print(f"[lbfgs] step {gstep:3d}: pairs={len(_lb_pairs)} alpha={alpha:.3g} "
                      f"|d|inf={float(np.abs(d).max()):.3g} loss {f0:.6f} -> {f_t:.6f} "
                      f"({n_ls} probe{'s' if n_ls > 1 else ''})", flush=True)
        elif _lb_pairs:
            _lb_pairs = []
            print(f"[lbfgs] step {gstep:3d}: line search failed ({_ls_fail_txt(_ls_why)}) "
                  f"-- history cleared, next step falls back to pure Sobolev descent",
                  flush=True)
        else:
            if _gov_shrunk:
                print(f"[lbfgs] step {gstep:3d}: Sobolev-descent line search failed "
                      f"({_ls_fail_txt(_ls_why)}) -- governor already shrank lr, holding "
                      f"(lr={lr_cur:.5f})", flush=True)
            else:
                lr_cur *= 0.5
                print(f"[lbfgs] step {gstep:3d}: Sobolev-descent line search failed "
                      f"({_ls_fail_txt(_ls_why)}) -- trust cap halved (lr={lr_cur:.5f})",
                      flush=True)
                if lr_cur < 1e-4:
                    break

    coords_final = np.array(coords_best, copy=True)
    final_ok = not diverged(coords_final)
    # ---- Newton final polish: drive the banked best to an exact root -------------------------
    if final_ok and newton_refine is not None:
        try:
            xf_p, res_p, it_p = newton_refine(q_full_of(best_theta), gk_of(best_log_gk),
                                              free_of(coords_final))
            c_pol = np.asarray(model.nodes_positions(xf_p, bp.xyz_fixed, st))
            if (res_p < 100.0 * NEWTON_TOL and np.isfinite(c_pol).all()
                    and not diverged(c_pol)
                    and objective_np(c_pol) <= objective_np(coords_final) + IMPROVE_TOL_):
                print(f"[newton polish] best x*: force residual -> {res_p:.2e} in {it_p} "
                      f"Newton iters; obj {objective_np(coords_final):.6f} -> "
                      f"{objective_np(c_pol):.6f} apex {apex_z_of(c_pol):.4f}", flush=True)
                coords_final = c_pol
            else:
                print(f"[newton polish] NOT adopted (res={res_p:.1e}, health/objective gate)",
                      flush=True)
        except Exception:                                # noqa: BLE001
            print("[newton polish] refinement raised -- keeping damped best", flush=True)
    final_obj = objective_np(coords_final)
    final_rms, final_max, *_ = surface_rms(coords_final, topo)
    history.append(final_obj)
    shape_rms_history.append(final_rms)
    shape_max_history.append(final_max)
    q_opt = q_full_np(best_theta)
    topo["q_stations"] = np.full(2 * N_STATIONS, np.nan) if Q_MODE == "edge" else np.exp(np.clip(np.asarray(best_theta, dtype=float), LOG_LO_NP, LOG_HI_NP))
    GAS_K_opt = float(np.exp(np.asarray(best_log_gk)))
    # Fixed-point verification: warm-started AT the best equilibrium the coupled solve should
    # stay put -- confirms x* is a genuine native constant-mass fixed point.
    if final_ok:
        try:
            q_best = q_full_of(best_theta)
            c_chk = np.asarray(native_forward(
                q_best, jnp.asarray(GAS_K_opt, dtype=qdtype), free_of(coords_final)))
            drift = (np.linalg.norm(c_chk - coords_final)
                     / max(np.linalg.norm(c_chk), 1e-12))
            print(f"[native_mass verify] native coupled re-solve at best x*: drift={drift:.2e} "
                  f"apex_z={apex_z_of(c_chk):.4f} V={vol(c_chk):.4f} "
                  f"(settled apex_z={apex_z_of(coords_final):.4f} V={vol(coords_final):.4f})",
                  flush=True)
        except Exception:                                    # noqa: BLE001
            print("[native_mass verify] native coupled re-solve at best x* FAILED", flush=True)
    loop_dt = time.perf_counter() - t_loop
    print(f"[native_mass optimise] {gstep + STEP0} steps ({n_reject} rejects); best obj={best:.6f}, "
          f"final obj={final_obj:.6f}, surfRMS={final_rms:.5f}, max|d|={final_max:.5f}, "
          f"apex_z={apex_z_of(coords_final):.4f}", flush=True)
    if OPTIMISE_GAS_K:
        print(f"[native_mass gas_k] GAS_K {GAS_K:.4f} -> {GAS_K_opt:.4f} (optimised)", flush=True)
    else:
        print(f"[native_mass gas_k] GAS_K={GAS_K:.4f} calibrated then held FIXED", flush=True)
    print(f"[native_mass time] {loop_dt:.1f} s for {gstep} steps "
          f"({1000 * loop_dt / max(gstep, 1):.1f} ms/step incl. first-call compile)", flush=True)
    pressure_out = dict(mode="native_mass_inside", GAS_K=GAS_K_opt,
                        pressure_fixed=float(GAS_K_opt / V0), sign=float(sign))
    return (None, coords_start, q_start, initial_ok,
            coords_final, q_opt, final_ok, history, rms0,
            shape_rms_history, shape_max_history, pressure_out)


def _controls():
    return dict(N_OUTER=N_OUTER, LR=LR, PATIENCE=PATIENCE, IMPROVE_TOL=IMPROVE_TOL)


# =============================================================================
# Matplotlib backend
# =============================================================================

def setup_matplotlib():
    import matplotlib
    for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as _plt  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            continue
    matplotlib.use("Agg", force=True)
    return False


# =============================================================================
# Diagnostics / export / view
# =============================================================================

def diagnostics(coords, q_opt, history, rms0, shape_rms_history, shape_max_history, topo,
                stage_name="Paraboloid dish",
                artifact_stem="jaxfdm_paraboloid_dish", plot_stem="jaxfdm_paraboloid",
                pressure_state=None):
    os.makedirs(TEMP_DIR, exist_ok=True)
    rms, mx, rho, z, d = surface_rms(coords, topo)
    print(f"\n[result:{stage_name}] {len(CATEGORIES)} per-edge force-density variables, "
          "optimised q by role:")
    groups = {}
    for name, qf in zip(CATEGORIES, q_opt):
        g = name.split("::")[0].split("@")[0]
        groups.setdefault(g, []).append(float(qf))
    for g, vals in groups.items():
        v = np.asarray(vals)
        print(f"   {g:12s} n={len(v):4d}   q: min={v.min():7.3f}  "
              f"mean={v.mean():7.3f}  max={v.max():7.3f}")
    print(f"[result] surface RMS {rms0:.5f} -> {rms:.5f}   max|d|={mx:.5f}   "
          f"apex_z={coords[topo['apex'], 2]:.4f}  (target DEPTH={DEPTH}, f={FOCAL:.3f})")

    ps = pressure_state or {}
    np.savez(f"{TEMP_DIR}/{artifact_stem}.npz",
             vertices=coords, q_categories=q_opt, category_names=CATEGORIES,
             edges=np.asarray(topo["edges"]), cat_ids=topo["cat_ids"],
             target_rho=topo["target_rho"], target_z=topo["target_z"],
             R=R, DEPTH=DEPTH, faces_tri=topo["faces_tri"],
             shape_rms_history=np.asarray(shape_rms_history),
             shape_max_history=np.asarray(shape_max_history),
             GAS_K=float(ps.get("GAS_K", np.nan)),
             pressure_fixed=float(ps.get("pressure_fixed", np.nan)),
             pressure_sign=float(ps.get("sign", np.nan)),
             pressure_mode=str(ps.get("mode", "")),
             fixed=np.asarray(topo["fixed"], dtype=int),
             ring_ids=np.asarray(topo["ring_ids"], dtype=int),
             n_theta=n_theta, n_rings=n_rings, focal=FOCAL,
             mesh_kind="unstructured", mesh_npz=MESH_NPZ, objective=OBJECTIVE, q_mode=Q_MODE, apex=int(topo["apex"]),
             n_stations=N_STATIONS, s_stations=topo["station_map"]["s_st"],
             q_stations=topo.get("q_stations", np.full(2 * N_STATIONS, np.nan)),
             edge_c2=topo["station_map"]["c2"], edge_s=topo["station_map"]["s_e"],
             seed_vertices=np.asarray(topo["vertices"]))

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history, ".-")
        ax.set_xlabel("outer step (last point = final Newton polish)")
        ax.set_ylabel("objective")
        ax.set_title(f"{stage_name} fit (autodiff gradient descent)")
        fig.tight_layout()
        fig.savefig(f"{TEMP_DIR}/{plot_stem}_obj.png", dpi=110)
        plt.close(fig)

        steps = np.arange(len(shape_rms_history))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(steps, shape_rms_history, ".-", label="surface RMS deviation")
        ax.semilogy(steps, shape_max_history, ".-", label="maximum deviation")
        ax.set_xlabel("accepted iteration (0 = initial; last = final polish)")
        ax.set_ylabel("shape deviation")
        ax.set_title("Shape residual convergence")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{TEMP_DIR}/{plot_stem}_shape_residual_convergence.png", dpi=120)
        plt.close(fig)

        # Meridian section: node (rho, z) vs the analytic paraboloid.
        fig, ax = plt.subplots(figsize=(6, 5))
        rr = np.linspace(0, R, 200)
        ax.plot(rr, paraboloid_z(rr), "k-", lw=1.5, label="target paraboloid")
        rho_all = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
        ax.scatter(rho_all, coords[:, 2], s=6, c="crimson", alpha=0.5, label="form-found nodes")
        ax.set_xlabel("radius rho")
        ax.set_ylabel("z")
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(f"{stage_name}: meridian section")
        fig.tight_layout()
        fig.savefig(f"{TEMP_DIR}/{plot_stem}_meridian.png", dpi=110)
        plt.close(fig)
        print(f"[result] plots + npz saved to {TEMP_DIR}")
    except Exception as e:  # noqa: BLE001
        print(f"[result] plotting skipped ({e})")


def edge_tensions(coords, q_opt, topo):
    """Axial force in every edge: N = q * L."""
    q_edge = np.asarray(q_opt)[topo["cat_ids"]]
    p = np.asarray(coords)
    lengths = np.array([np.linalg.norm(p[v] - p[u]) for (u, v) in topo["edges"]])
    return q_edge * lengths, lengths


def _draw_target(ax, color=(0.6, 0.6, 0.6), lw=1.0, alpha=1.0,
                 n_rings_draw=6, n_meridians=24):
    """Wireframe of the target paraboloid dish."""
    th = np.linspace(0, 2 * np.pi, 80)
    for rr in np.linspace(R / n_rings_draw, R, n_rings_draw):
        ax.plot(rr * np.cos(th), rr * np.sin(th), float(paraboloid_z(rr)),
                color=color, lw=lw, ls="--", alpha=alpha)
    rr = np.linspace(0, R, 40)
    zr = paraboloid_z(rr)
    for t in np.linspace(0, 2 * np.pi, n_meridians + 1)[:-1]:
        ax.plot(rr * np.cos(t), rr * np.sin(t), zr, color=color, lw=lw * 0.85,
                ls="--", alpha=alpha)


def view(coords, q_opt, topo, interactive, stage="Optimised",
         basename="jaxfdm_paraboloid_shape3d"):
    """3D network vs the target paraboloid, edges coloured by axial tension."""
    if not SHOW_VIEWER:
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        tension, _ = edge_tensions(coords, q_opt, topo)
        segs = np.array([[coords[u], coords[v]] for (u, v) in topo["edges"]])
        tpos = tension[tension > 0]
        vmin = float(tpos.min()) if tpos.size else 1e-6
        vmax = float(tension.max()) if tension.size else 1.0
        use_log = (vmax / max(vmin, 1e-12)) > 50.0
        norm = (mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, vmin * 1.0001))
                if use_log else mcolors.Normalize(vmin=vmin, vmax=vmax))
        cmap = plt.cm.turbo
        colors = cmap(norm(np.clip(tension, vmin, vmax)))
        print(f"[view:{stage}] edge tension N=q*L: min={tension.min():.4g}  "
              f"mean={tension.mean():.4g}  max={tension.max():.4g}  "
              f"({'log' if use_log else 'linear'} colour scale)", flush=True)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Line3DCollection(segs, colors=colors, linewidths=1.0))
        _draw_target(ax)
        lim = R * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(0, DEPTH * 1.15)
        ax.set_box_aspect((2 * lim, 2 * lim, DEPTH * 1.15))
        ax.set_title(f"{stage} form vs paraboloid dish (f={FOCAL:.2f}) — axial tension N = q*L")

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(tension)
        fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1,
                     label=f"axial tension N = q*L ({'log' if use_log else 'linear'} scale)")
        png = f"{TEMP_DIR}/{basename}.png"
        fig.savefig(png, dpi=120)
        print(f"[view] 3D figure saved to {png}", flush=True)
        if interactive:
            plt.show()
        else:
            plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[view] skipped ({e})")


def view_comparison(coords, topo, interactive, stage="Final",
                    basename="jaxfdm_paraboloid_compare"):
    """Form-found net (crimson) overlaid on the target paraboloid wireframe (steel-blue)."""
    if not SHOW_VIEWER:
        return
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        rms, mx, *_ = surface_rms(coords, topo)
        segs = np.array([[coords[u], coords[v]] for (u, v) in topo["edges"]])
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Line3DCollection(segs, colors="crimson", linewidths=0.5,
                                             alpha=0.22))
        _draw_target(ax, color="steelblue", lw=2.2, alpha=1.0,
                     n_rings_draw=7, n_meridians=36)
        lim = R * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(0, DEPTH * 1.15)
        ax.set_box_aspect((2 * lim, 2 * lim, DEPTH * 1.15))
        ax.set_title(f"{stage} form vs target paraboloid  "
                     f"(surface RMS={rms:.5f}, max|d|={mx:.5f})")
        ax.legend(handles=[Line2D([0], [0], color="crimson", lw=1.5, label=stage),
                           Line2D([0], [0], color="steelblue", lw=1.5, ls="--",
                                  label="target")], loc="upper right")
        png = f"{TEMP_DIR}/{basename}.png"
        fig.savefig(png, dpi=120)
        print(f"[view] comparison figure saved to {png}", flush=True)
        if interactive:
            plt.show()
        else:
            plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"[view] comparison skipped ({e})")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    interactive = setup_matplotlib()
    topo = build_disk_topology()
    from collections import Counter
    roles = Counter(c.split("::")[0].split("@")[0] for c in CATEGORIES)
    print(f"Target: paraboloid dish  z = {DEPTH} * (1 - (r/{R})^2)  -> apex z={DEPTH}, "
          f"focal length f={FOCAL:.4f} (f/D={FOCAL / (2 * R):.3f})", flush=True)
    print(f"Mesh: {topo['vertices'].shape[0]} nodes, {len(topo['edges'])} edges, "
          f"{topo['faces_tri'].shape[0]} tris, {len(topo['free'])} free", flush=True)
    print(f"{len(CATEGORIES)} per-edge variables; edges by role: {dict(roles)}", flush=True)

    (network, coords_start, q_start, initial_ok,
     coords, q_opt, final_ok, history, rms0,
     shape_rms_history, shape_max_history, pressure_state) = optimise_native_mass(
         topo, stage_label="Paraboloid dish fit (native constant-mass Newton adjoint)",
         controls=_controls())

    diagnostics(coords, q_opt, history, rms0, shape_rms_history, shape_max_history, topo,
                pressure_state=pressure_state)
    view(coords, q_opt, topo, interactive, stage="Optimised")
    view_comparison(coords, topo, interactive, stage="Final")
    print("\n[done] paraboloid dish fit complete.", flush=True)
