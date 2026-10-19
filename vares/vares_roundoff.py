r"""vares_roundoff.py -- certify a VaReS rung converged at the double-precision ROUND-OFF FLOOR of its residual
(criterion (a), user decision 2026-10-13).

Why: the absolute tolerance NEWTON_TOL_R = 1e-7 on ||R|| over all free dofs lies at / below what double precision can
reach on the finest nested rungs.  h012 (25k dofs) stopped on the step floor at ||R|| 1.2165e-7: the linear solve was
exact (1.7e-12, iterative refinement changed nothing), the full Newton correction was 3e-8 mm, and every fraction of it
RAISED ||R|| (1.245 / 1.347 / 1.445e-7) -- the residual is round-off-dominated (diag_h012_refine.py).  The tolerance is
NOT changed; a rung that misses it is accepted only if it is PROVEN to sit on the floor:
  (1) ||R|| <= FLOOR_FACTOR * NEWTON_TOL_R           (near the tolerance -- never a genuine stall, e.g. ||R|| 7.4)
  (2) max over nodes |dX_newton,node| <= DX_MAX mm   (the remaining correction to the pattern is negligible)
      ): was the L2 norm of the whole dof vector, which grows with sqrt(#dofs) -- h006
      (98k dofs) failed it at 5.5e-6 mm with a largest nodal correction of 4.2e-8 mm.  The per-node maximum is
      mesh-size independent; DX_MAX unchanged.  Old version kept as vares_roundoff_v1_L2check.py.
  (3) no step fraction alpha in ALPHAS lowers ||R|| by more than MIN_GAIN (relative)   (round-off, not descent)
dX_newton is the Levenberg step (K_free + LAM I) dX = -R at LAM = LAM_CERT (tiny), solved with vares_sparse.
Recorded in pattern.npz as converged_criterion = "roundoff_floor" with the certificate numbers.
"""
import numpy as np

FLOOR_FACTOR = 10.0
DX_MAX = 1e-6           # mm
MIN_GAIN = 0.01
ALPHAS = (1.0, 0.5, 0.25, 0.125)
LAM_CERT = 1e-8


def certify(X, free, tol, residual_tangent, residual, step_fn):
    """residual_tangent(X) -> (R, K sparse), residual(X) -> R, step_fn(X, dX, alpha) -> X_trial (apply_step_mesh).
    Returns the certificate dict (ok True/False and every number)."""
    import vares_sparse as VS
    R, K = residual_tangent(X)
    Kf = VS.restrict(K, free)
    Rf = R[free]
    r0 = float(np.linalg.norm(Rf))
    dx = VS.solve_shifted(Kf, LAM_CERT, -Rf)
    dX = np.zeros(2 * X.shape[0])
    dX[free] = dx
    rs = [float(np.linalg.norm(residual(step_fn(X, dX, a))[free])) for a in ALPHAS]
    best = min(rs)
    c1 = r0 <= FLOOR_FACTOR * tol
    dn = float(np.linalg.norm(dX.reshape(-1, 2), axis=1).max())          # largest nodal correction (mm)
    c2 = dn <= DX_MAX
    c3 = best >= (1.0 - MIN_GAIN) * r0
    return dict(ok=bool(c1 and c2 and c3), R=r0, tol=tol, R_over_tol=r0 / tol, dx_newton_mm=float(np.linalg.norm(dx)),
                dx_newton_max_mm=float(np.abs(dx).max()),
                dx_newton_node_max_mm=dn, R_trial=rs, alphas=list(ALPHAS), best_gain=1.0 - best / r0,
                checks=dict(near_tol=bool(c1), small_correction=bool(c2), no_descent=bool(c3)),
                rule=f"||R|| <= {FLOOR_FACTOR:g} tol, max nodal |dX| <= {DX_MAX:g} mm, no alpha in {ALPHAS} lowers ||R|| by > {MIN_GAIN:g}")
