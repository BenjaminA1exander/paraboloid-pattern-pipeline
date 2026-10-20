r"""asymptote_fit.py -- the asymptote fit f(h) = f_inf + a h^p and the convergence rule shared by the figure scripts.

    fit       least-squares fit of f_inf, a, p (p in [0.5, 4]) with standard errors from the fit covariance
    limit     the extrapolated limit and its band: f_inf and sqrt(se(f_inf)^2 + rms residual^2) when the trend is
              resolved (se(p) < p and p > 0.51), otherwise the mean and standard deviation of the values
    classify  the rule of the Abaqus ladder figures, fixed before the data were seen: "converging" when the fit is
              resolved AND the successive changes shrink over the three finest steps; "flat" when every change
              reverses the previous one (scatter about a level); "not converged" otherwise
"""
import numpy as np
from scipy.optimize import least_squares


def fit(h, y):
    def res(v):
        return v[0] + v[1] * h ** v[2] - y
    s = least_squares(res, x0=[y[-1], (y[0] - y[-1]) / h[0] ** 2, 2.0],
                      bounds=([-np.inf, -np.inf, 0.5], [np.inf, np.inf, 4.0]))
    J = s.jac
    dof = max(len(y) - 3, 1)
    cov = np.linalg.pinv(J.T @ J) * (s.fun @ s.fun) / dof
    return s.x, np.sqrt(np.diag(cov))


def limit(h, y):
    """(fit parameters, their standard errors, limit, band, flat)."""
    v, se = fit(h, y)
    flat = not (se[2] < v[2]) or v[2] <= 0.51
    if flat:
        return v, se, float(y.mean()), float(y.std(ddof=1)), True
    r = v[0] + v[1] * h ** v[2] - y
    return v, se, float(v[0]), float(np.hypot(se[0], np.sqrt(np.mean(r ** 2)))), False


def classify(h, y):
    """(kind, fit parameters, standard errors, limit, band, successive changes coarse -> fine)."""
    o = np.argsort(-h)
    h, y = h[o], y[o]
    d = np.diff(y)                                            # coarse -> fine changes
    v, se = fit(h, y)
    fine = np.abs(d[-3:])
    shrinking = bool(np.all(fine[1:] < fine[:-1]))
    resolved = bool(v[2] > 0.5 and se[2] < v[2])
    if resolved and shrinking:
        r = v[0] + v[1] * h ** v[2] - y
        return "converging", v, se, float(v[0]), float(np.hypot(se[0], np.sqrt(np.mean(r ** 2)))), d
    if np.all(d[1:] * d[:-1] < 0):                           # every change reverses the previous one
        return "flat", v, se, float(y.mean()), float(y.std(ddof=1)), d
    return "not converged", v, se, None, None, d
