r"""vares_sparse.py -- sparse linear algebra for the VaReS Newton loop.

The checkpointed worker (MeshStudy_paraboloid_vares_ckpt.py) assembled the tangent as a DENSE (2N, 2N) array and
solved it with scipy.linalg.solve, plus a dense eigh for the min-eigenvalue diagnostic: memory ~3 (2N)^2 x 8 B,
time ~(2N)^3 -- 7.6 GB / ~minutes per iteration at 12.5 mm and ~120 GB at 6.25 mm.  Each triangle couples only its
three nodes, so the tangent is sparse.  This module changes ONLY the linear algebra:

  assemble_residual_tangent_sparse  the same per-element calls as vares_membrane.assemble_residual_tangent
                                    (elem_residual_tangent, element_type_of, elem_dof_ids, same arguments, same
                                    element order); R is accumulated identically (bit-identical); K is collected as
                                    COO triplets and summed into CSR (duplicate entries summed in a different order
                                    than the dense '+=' -> differences at round-off only);
  solve_shifted                     (K_free + lam I) dX = -R by SuperLU (scipy splu) -- the general, non-symmetric
                                    factorisation, as the exact tangent is non-symmetric; a singular factor raises
                                    np.linalg.LinAlgError like the dense solve, so the Levenberg logic is unchanged;
  min_eig_sym                       smallest eigenvalue of 0.5 (K + K^T) by Lanczos (ARPACK eigsh, 'SA') -- a log
                                    diagnostic only (the n_indef counter); the Newton path never reads it.
Tolerances, damping constants, line search and the residual are untouched.  Validated against the dense path in
validate_vares_sparse.py before any use.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from vares_membrane import element_type_of, elem_dof_ids, elem_residual_tangent

SPLU_ORDER = "COLAMD"                     # SuperLU column ordering (scipy default)


def assemble_residual_tangent_sparse(X_mat, X_sp, elements, S_pre_all, detJ_ref_all, *, linear_material=False,
                                     vary_detJ=False, sigma_star_all=None):
    """Global (R (2N,), K (2N, 2N) CSR) -- vares_membrane.assemble_residual_tangent with a sparse K."""
    N_g = 2 * X_mat.shape[0]
    R = np.zeros(N_g)
    rows, cols, vals = [], [], []
    for e, elem in enumerate(elements):
        etype = element_type_of(elem)
        R_e, K_e = elem_residual_tangent(
            X_mat[elem], X_sp[elem],
            None if S_pre_all is None else S_pre_all[e], detJ_ref_all[e],
            etype=etype, linear_material=linear_material,
            vary_detJ=vary_detJ,
            sigma_star_ips=(None if sigma_star_all is None else sigma_star_all[e]))
        ids = np.asarray(elem_dof_ids(elem))
        R[ids] += R_e
        n = len(ids)
        rows.append(np.repeat(ids, n))
        cols.append(np.tile(ids, n))
        vals.append(np.asarray(K_e).ravel())
    K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N_g, N_g)).tocsr()
    K.sum_duplicates()
    return R, K


# ------------------------------------------------------------------------------------------------ parallel assembly
# The per-element calls are pure Python (~10-20 ms per triangle): at 12.5 mm one tangent takes ~290 s and one
# line-search residual ~200 s on one core.  ParallelAssembler farms CONTIGUOUS element chunks out to a process pool
# and accumulates the returned element vectors / matrices in the parent IN ELEMENT ORDER, exactly as the serial loops
# do -- so R is bit-identical to vares_membrane.assemble_residual / assemble_residual_tangent and K bit-identical to
# assemble_residual_tangent_sparse (validated in validate_vares_sparse.py).  Children inherit the parent's priority.
_G = {}


def _init(X_sp, elements, S_pre_all, detJ_ref_all, sigma_star_all, kw):
    _G.update(X_sp=X_sp, elements=elements, S_pre_all=S_pre_all, detJ_ref_all=detJ_ref_all,
              sigma_star_all=sigma_star_all, kw=kw)


def _chunk(args):
    from vares_membrane import elem_residual
    X, lo, hi, want_K = args
    g = _G
    out = []
    for e in range(lo, hi):
        elem = g["elements"][e]
        etype = element_type_of(elem)
        a = (X[elem], g["X_sp"][elem], None if g["S_pre_all"] is None else g["S_pre_all"][e], g["detJ_ref_all"][e])
        kw = dict(etype=etype, sigma_star_ips=None if g["sigma_star_all"] is None else g["sigma_star_all"][e],
                  **g["kw"])
        out.append(elem_residual_tangent(*a, **kw) if want_K else elem_residual(*a, **kw))
    return out


class ParallelAssembler:
    def __init__(self, procs, X_sp, elements, S_pre_all, detJ_ref_all, *, linear_material=False, vary_detJ=False,
                 sigma_star_all=None, chunks_per_proc=4):
        import multiprocessing as mp
        self.elements = elements
        self.ids = [np.asarray(elem_dof_ids(el)) for el in elements]
        kw = dict(linear_material=linear_material, vary_detJ=vary_detJ)
        self.pool = mp.get_context("spawn").Pool(
            procs, initializer=_init, initargs=(X_sp, elements, S_pre_all, detJ_ref_all, sigma_star_all, kw))
        n = len(elements)
        edges = np.linspace(0, n, procs * chunks_per_proc + 1).astype(int)
        self.chunks = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        self.N_g = 2 * X_sp.shape[0]

    def _run(self, X, want_K):
        res = self.pool.map(_chunk, [(X, a, b, want_K) for a, b in self.chunks])
        return [r for part in res for r in part]                  # element order

    def residual_tangent(self, X):
        R = np.zeros(self.N_g)
        rows, cols, vals = [], [], []
        for ids, (R_e, K_e) in zip(self.ids, self._run(X, True)):
            R[ids] += R_e
            n = len(ids)
            rows.append(np.repeat(ids, n))
            cols.append(np.tile(ids, n))
            vals.append(np.asarray(K_e).ravel())
        K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.N_g, self.N_g)).tocsr()
        K.sum_duplicates()
        return R, K

    def residual(self, X):
        R = np.zeros(self.N_g)
        for ids, R_e in zip(self.ids, self._run(X, False)):
            R[ids] += R_e
        return R

    def close(self):
        self.pool.terminate()


def restrict(K, free):
    """K[free][:, free] as CSC (the factorisation format)."""
    return K[free][:, free].tocsc()


# Iterative refinement: h012 (25k dofs) stopped at ||R|| 1.217e-7 > NEWTON_TOL_R 1e-7 on the step floor
# -- near-singular tangent (min eig 1.8e-4) at lambda ~ 1e-6, the SuperLU solution loses accuracy and the Newton step
# stops being a descent direction.  REFINE correction passes x += LU^-1 (b - A x) reuse the factorisation (cheap) and
# restore the solve accuracy; the tolerance is untouched.  MESHSTUDY_REFINE=0 gives the previous behaviour.
import os as _os
REFINE = int(_os.environ.get("MESHSTUDY_REFINE", "2"))


def solve_shifted(K_free, lam, rhs):
    """(K_free + lam I)^-1 rhs by SuperLU + REFINE iterative-refinement passes; singular -> np.linalg.LinAlgError
    (as scipy.linalg.solve)."""
    A = (K_free + lam * sp.identity(K_free.shape[0], format="csc")).tocsc()
    try:
        lu = spla.splu(A, permc_spec=SPLU_ORDER)
    except RuntimeError as ex:                      # "Factor is exactly singular"
        raise np.linalg.LinAlgError(str(ex))
    x = lu.solve(rhs)
    nb = np.linalg.norm(rhs)
    for _ in range(REFINE):
        r = rhs - A @ x
        if not np.linalg.norm(r) > 1e-15 * nb:
            break
        x = x + lu.solve(r)
    if not np.all(np.isfinite(x)):
        raise np.linalg.LinAlgError("non-finite solve")
    return x


def n_negative(Ks, sigma):
    """Number of eigenvalues of the symmetric Ks below sigma (Sylvester inertia of Ks - sigma I): SuperLU in symmetric
    mode with a symmetric ordering and NO pivoting gives P A P^T = L U with U = D L^T, so sign(diag U) = inertia.
    None if the unpivoted factorisation breaks down (exactly zero pivot)."""
    A = (Ks - sigma * sp.identity(Ks.shape[0], format="csc")).tocsc()
    try:
        lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    except RuntimeError:
        return None
    if not (np.array_equal(lu.perm_r, lu.perm_c)):        # a row swap happened: inertia not readable
        return None
    return int(np.sum(lu.U.diagonal() < 0))


MINEIG_BUDGET_S = float(_os.environ.get("MESHSTUDY_MINEIG_BUDGET", "120"))


def min_eig_sym(K_free, tol=1e-8):
    """Smallest eigenvalue of the symmetric part 0.5 (K + K^T) -- a log diagnostic (the Newton path never reads it).
    Find a shift sigma BELOW the whole spectrum by inertia counts (n_negative == 0), then shift-invert Lanczos around
    it: the eigenvalue nearest sigma is then the minimum, and it converges in a few steps.  Falls back to plain
    Lanczos ('SA', slow near convergence) if an inertia count cannot be read.
    2026-10-13: h006 (98k dofs) hung > 3 h inside this diagnostic -> every stage is capped (MINEIG_BUDGET_S between
    stages, ARPACK maxiter 300); over budget or unconverged -> NaN ("not computed" -- the Newton path never reads it)."""
    import time as _t
    t0 = _t.time()
    Ks = (0.5 * (K_free + K_free.T)).tocsc()
    sigma, step = 0.0, 1.0
    for _ in range(60):
        if _t.time() - t0 > MINEIG_BUDGET_S:
            return float("nan")
        n = n_negative(Ks, sigma)
        if n is None:
            return _min_eig_lanczos(Ks)
        if n == 0:
            break
        sigma, step = sigma - step, step * 4.0
    else:
        return _min_eig_lanczos(Ks)
    sigma -= 1e-6 * max(1.0, abs(sigma))                 # strictly below, never on an eigenvalue
    try:
        return float(spla.eigsh(Ks, k=1, sigma=sigma, which="LM", tol=tol, maxiter=300, return_eigenvectors=False)[0])
    except (spla.ArpackNoConvergence, RuntimeError):
        return _min_eig_lanczos(Ks)


def _min_eig_lanczos(Ks, tol=1e-8, maxiter=300):
    Ks = Ks.tocsr()
    try:
        return float(spla.eigsh(Ks, k=1, which="SA", tol=tol, maxiter=maxiter, ncv=min(64, Ks.shape[0] - 1),
                                return_eigenvectors=False)[0])
    except spla.ArpackNoConvergence as ex:
        return float(ex.eigenvalues[0]) if len(ex.eigenvalues) else float("nan")
