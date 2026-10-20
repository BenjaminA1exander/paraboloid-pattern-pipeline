r"""project_self_stress_cst.py -- remove the arbitrary self-stress of a converged force-density net with a
TRIANGLE-based (constant-stress-triangle, CST) complementary energy, the refinement of project_self_stress.py
(which lumps the fabric onto edges with weights L^4/A_e).

Model: every triangle T carries a uniform membrane stress sigma_T (3 components in its own plane). A uniform
stress in a triangle is statically equivalent to exactly three edge forces f_k^T (k = its edges): the virial
of the triangle,  sum_k f_k^T L_k (t_k (x) t_k) = A_T sigma_T,  is a 3x3 linear map (unique for a non-degenerate
triangle).  An edge's force is the sum over its (one or two) triangles:  F_e = sum_T f_e^T  ->  F = G sigma.

Projection, at the converged shape x* and loads (both unchanged):
    minimise   sum_T  A_T / 2 [ sxx^2 + syy^2 - 2 nu sxx syy + 2 (1 + nu) sxy^2 ]      (isotropic membrane, Et = 1)
    subject to A(x*) diag(1/L) G sigma = A(x*) q*                                          (exact net equilibrium)
one sparse KKT solve; then q = G sigma / L.  The equilibrium matrix A and the right-hand side are those of
project_self_stress.build (free-node equations; rim chords enter none and are reported, not constrained).
No symmetry and no exact solution are used.  nu (Poisson's ratio of the fabric) is the one material choice.

Run (base env): python project_self_stress_cst.py <in.npz> <out.npz> [nu]
"""
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import project_self_stress as PS


def edge_index(E):
    return {(min(a, b), max(a, b)): k for k, (a, b) in enumerate(E)}


def triangle_maps(V, T, E):
    """Sparse G (n_edges x 3 n_tri): edge forces from triangle stresses, and the per-triangle areas."""
    key = edge_index(E)
    rr, cc, vv = [], [], []
    At = np.zeros(len(T))
    for ti, (a, b, c) in enumerate(T):
        P = V[[a, b, c]]
        n = np.cross(P[1] - P[0], P[2] - P[0]); A = 0.5 * np.linalg.norm(n); n /= 2 * A
        e1 = (P[1] - P[0]) / np.linalg.norm(P[1] - P[0]); e2 = np.cross(n, e1)
        At[ti] = A
        M = np.zeros((3, 3)); ek = []
        for k, (i, j) in enumerate(((a, b), (b, c), (c, a))):
            d = V[j] - V[i]; L = np.linalg.norm(d); t = d / L
            u, v = t @ e1, t @ e2
            M[:, k] = L * np.array([u * u, v * v, u * v])          # virial columns [xx, yy, xy]
            ek.append(key[(min(i, j), max(i, j))])
        K = np.linalg.solve(M, A * np.eye(3))                      # f = K sigma, sigma = [sxx, syy, sxy]
        for k in range(3):
            for m in range(3):
                rr.append(ek[k]); cc.append(3 * ti + m); vv.append(K[k, m])
    G = sp.csr_matrix((vv, (rr, cc)), shape=(len(E), 3 * len(T)))
    return G, At


def project_cst(V, T, E, fixed, q, nu=0.3):
    A, _B, _L6, cols, free = PS.build(V, T, E, fixed)
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    G, At = triangle_maps(V, T, E)
    Bq = A @ sp.diags(1.0 / L[cols]) @ G[cols]                    # equilibrium in terms of sigma
    b = A @ q[cols]
    h = np.zeros((len(T), 3, 3))
    h[:, 0, 0] = h[:, 1, 1] = 1.0; h[:, 0, 1] = h[:, 1, 0] = -nu; h[:, 2, 2] = 2 * (1 + nu)
    H = sp.block_diag([At[t] * h[t] for t in range(len(T))], format="csr")
    m = Bq.shape[0]
    KKT = sp.bmat([[H, Bq.T], [Bq, -1e-13 * sp.identity(m)]], format="csc")
    sol = spl.spsolve(KKT, np.concatenate([np.zeros(H.shape[0]), b]))
    sig = sol[:H.shape[0]]
    F = G @ sig
    q_new = F / L
    info = dict(n_tri=len(T), n_sigma=H.shape[0], n_eq=m, nu=nu,
                eq_residual_rel=float(np.linalg.norm(A @ q_new[cols] - b) / np.linalg.norm(b)),
                dq_rel_rms=float(np.sqrt(np.mean((q_new[cols] / q[cols] - 1) ** 2))),
                q_min=float(q_new[cols].min()), n_negative=int((q_new[cols] <= 0).sum()),
                rim_chords=int(len(E) - len(cols)), sigma_tri=sig.reshape(-1, 3))   # per-triangle (s11, s22, s12)
    q_out = q.copy(); q_out[cols] = q_new[cols]                    # rim chords (reactions) left as they are
    return q_out, info


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    nu = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    d = dict(np.load(src, allow_pickle=True))
    V, T, E, fixed = (np.asarray(d[k]) for k in ("vertices", "faces_tri", "edges", "fixed"))
    q = np.asarray(d["q_categories"], float)[np.asarray(d["cat_ids"], int)]
    q_new, info = project_cst(V, T, E, fixed, q, nu)
    for k, v in info.items():
        print(f"[cst] {k}: {v}")
    d["q_orig"] = q; d["q_categories"] = q_new; d["cat_ids"] = np.arange(len(q_new))
    d["projection_info"] = np.array("CST " + str(info))
    np.savez(dst, **d)
    print(f"[cst] saved {dst}")
