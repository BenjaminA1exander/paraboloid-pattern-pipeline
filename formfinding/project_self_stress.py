r"""project_self_stress.py -- remove the arbitrary SELF-STRESS from a converged force-density net:
at the converged shape x*, find the smoothest q that holds exactly the same shape under exactly the same
loads.

At fixed x*, equilibrium of every free node is LINEAR in q:
    A(x*) q = P(x*),      A: 3 rows per free node, one column per edge (entries x_i - x_j)
P(x*) (follower pressure, constant-mass p = GAS_K/V(x*)) depends on the shape only.  The optimised q*
is an exact equilibrium (Newton residual ~1e-14), so P(x*) = A(x*) q* to machine precision -- the
solver's own load model, with nothing re-implemented.  Every solution is q* + (self-stress), the
self-stress being the null space of A (>= n_edges - 3 n_free states).
A pressurised dome with a pinned rim is STATICALLY INDETERMINATE: statics alone does not fix the
membrane stress, a material does (compatibility).  The projection therefore picks the member of minimum
COMPLEMENTARY ENERGY of a uniform membrane -- the stress a uniform elastic fabric would carry:
    minimise  sum_e F_e^2 L_e / EA_e = sum_e q_e^2 L_e^4 / A_e     subject to  A(x*) q = A(x*) q*
with EA_e proportional to A_e / L_e (A_e = edge tributary area, 1/3 of each adjacent triangle), the
triangle-lattice model of a uniform membrane.  One sparse KKT solve.
Measured on the 4626-node run: q changes 3.4% rms, all edges stay in tension, the local
(MLS 6-edge) recovery error drops 2.53/1.22% -> 1.13/0.72% (N_phi/N_theta), the axisymmetric part is
untouched.  Weights L^3 (uniform EA per edge) give 1.50/0.85%.  A smoothness criterion (graph Laplacian
of the nodal stress) FAILED: it barely penalises the long-wavelength self-stress (a smooth pattern has
a small Laplacian) and moved q 41% with 79 edges in compression for a 5% roughness gain.
Rim chords (both ends pinned) enter no free-node equation and are left as they are.
q > 0 (tension only) is checked; if violated the change is reported (bounds would be the next step).

Writes <out>.npz = the input npz with q_categories replaced (cat_ids = identity), plus q_orig, and the
algebraic checks.  Physical checks (solver residual, coupled re-solve, stability) run in the solver:
JAXFDM_VERIFY_NPZ=<out>.npz (see ParaboloidDish_unstructured_JAXFDM.py).

Run (base env): python project_self_stress.py <in.npz> <out.npz>
"""
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def build(V, T, E, fixed):
    N = len(V)
    free = np.setdiff1d(np.arange(N), fixed)
    fmask = np.zeros(N, bool)
    fmask[free] = True
    row_of = -np.ones(N, int)
    row_of[free] = np.arange(free.size)
    act = fmask[E[:, 0]] | fmask[E[:, 1]]                  # edges in at least one free equation
    cols = np.flatnonzero(act)
    # ---- equilibrium matrix A (3 nfree x ncols): node i gets +(x_i - x_j) q_e -------------------
    rr, cc, vv = [], [], []
    for k, e in enumerate(cols):
        i, j = E[e]
        d = V[i] - V[j]
        for node, sgn in ((i, 1.0), (j, -1.0)):
            if fmask[node]:
                for a in range(3):
                    rr.append(3 * row_of[node] + a); cc.append(k); vv.append(sgn * d[a])
    A = sp.csr_matrix((vv, (rr, cc)), shape=(3 * free.size, cols.size))
    # ---- nodal virial operator B (6 comps per interior node) --------------------------------------
    a_, b_, c_ = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    fn = np.cross(b_ - a_, c_ - a_)
    Ai = np.zeros(N)
    nrm = np.zeros((N, 3))
    for k in range(3):
        np.add.at(Ai, T[:, k], 0.5 * np.linalg.norm(fn, axis=1) / 3)
        np.add.at(nrm, T[:, k], 0.5 * fn)
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    comps = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    cw = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])   # Frobenius weights
    rr, cc, vv = [], [], []
    for k, e in enumerate(cols):
        i, j = E[e]
        d = V[j] - V[i]
        L2 = d @ d
        t = d / np.sqrt(L2)
        for node in (i, j):
            if not fmask[node]:
                continue
            P = np.eye(3) - np.outer(nrm[node], nrm[node])
            M = P @ np.outer(t, t) @ P * L2 / (2 * Ai[node])
            for ci, (p_, r_) in enumerate(comps):
                rr.append(6 * row_of[node] + ci); cc.append(k); vv.append(cw[ci] * M[p_, r_])
    B = sp.csr_matrix((vv, (rr, cc)), shape=(6 * free.size, cols.size))
    # ---- graph Laplacian over free nodes (uniform weights), lifted to 6 comps ----------------------
    nb = [[] for _ in range(N)]
    for i, j in E:
        nb[i].append(j); nb[j].append(i)
    lr, lc, lv = [], [], []
    for n_, i in enumerate(free):
        js = [j for j in nb[i] if fmask[j]]
        lr.append(n_); lc.append(n_); lv.append(1.0)
        for j in js:
            lr.append(n_); lc.append(row_of[j]); lv.append(-1.0 / len(js))
    Lg = sp.csr_matrix((lv, (lr, lc)), shape=(free.size, free.size))
    L6 = sp.kron(Lg, sp.identity(6), format="csr")
    return A, B, L6, cols, free


def edge_tributary_area(V, T, E):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    At = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    acc = {}
    for ti, t in enumerate(T):
        for k in range(3):
            key = (min(t[k], t[(k + 1) % 3]), max(t[k], t[(k + 1) % 3]))
            acc[key] = acc.get(key, 0.0) + At[ti] / 3.0
    return np.array([acc[(min(i, j), max(i, j))] for i, j in E])


def project(V, T, E, fixed, q, weights="L4A"):
    A, B, L6, cols, free = build(V, T, E, fixed)
    qa = q[cols]
    b = A @ qa
    L = np.linalg.norm(V[E[cols, 0]] - V[E[cols, 1]], axis=1)
    w = L ** 4 / edge_tributary_area(V, T, E)[cols] if weights == "L4A" else L ** 3
    n, m = A.shape[1], A.shape[0]
    KKT = sp.bmat([[sp.diags(w), A.T], [A, None]], format="csc")
    sol = spl.spsolve(KKT, np.concatenate([np.zeros(n), b]))
    dq = sol[:n] - qa
    M = (L6 @ B).tocsr()
    q_new = q.copy()
    q_new[cols] = qa + dq
    info = dict(n_edges_active=n, n_eq=m, eq_residual_rel=float(np.linalg.norm(A @ (qa + dq) - b) / np.linalg.norm(b)),
                rough_before=float(np.linalg.norm(M @ qa)), rough_after=float(np.linalg.norm(M @ (qa + dq))),
                dq_rel_rms=float(np.sqrt(np.mean((dq / qa) ** 2))), q_min=float(q_new.min()),
                n_negative=int((q_new <= 0).sum()))
    return q_new, info


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    d = dict(np.load(src, allow_pickle=True))
    V, T, E, fixed = (np.asarray(d[k]) for k in ("vertices", "faces_tri", "edges", "fixed"))
    q = np.asarray(d["q_categories"], float)[np.asarray(d["cat_ids"], int)]
    import time
    t0 = time.perf_counter()
    q_new, info = project(V, T, E, fixed, q)
    info["seconds"] = time.perf_counter() - t0
    for k, v in info.items():
        print(f"[project] {k}: {v}")
    d["q_orig"] = q
    d["q_categories"] = q_new
    d["cat_ids"] = np.arange(len(q_new))
    d["projection_info"] = np.array(str(info))
    np.savez(dst, **d)
    print(f"[project] saved {dst}")
