"""vares_settings.py -- the solver constants of the VaReS Newton loop (Sec. 3.4.2).

Levenberg-damped Newton on the stress-mismatch residual (Eq. 34): tolerance on the residual norm over the free
dofs, iteration budget, step caps, damping schedule and the Armijo line search.  NEWTON_MAX_IT is a budget that
the ladder worker raises (MESHSTUDY_MAX_IT); NEWTON_TOL_R is never changed -- a rung that cannot reach it in double
precision is accepted only with the round-off certificate of vares_roundoff.py.
"""
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

# weights of (warp, weft, shear) in the element-centre residual magnitude used for reporting
RES_WEIGHTS = (1.0, 1.0, 10.0)
