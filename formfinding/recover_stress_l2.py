r"""recover_stress_l2.py -- membrane stress recovery by GLOBAL L2 (Galerkin) projection, no patch, no tuning knob.

Input, per rung: the unprojected form-found net (jaxfdm_paraboloid_dish.npz). The CST self-stress projection
(project_self_stress_cst) is re-run here only to obtain its per-triangle stresses sigma_T, which the saved
projected npz does not store. The q it returns is checked against the saved jaxfdm_paraboloid_dish_projected_cst.npz.

Recovery: find the continuous piecewise-linear tensor field N_h (P1 hat functions phi_i on the form-found
triangles, six Cartesian components) that is closest to the input field in L2 over the whole surface:
        sum_j M_ij N_j = b_i,   M_ij = int phi_i phi_j dA (consistent mass),   b_i = int phi_i S dA
then project each nodal tensor onto the nodal tangent plane.  No symmetry, no exact solution, no patch size;
every node is coupled to its neighbours through M.  The exact statics are used ONLY to score.

Variants scored (rms relative error over interior nodes with 0.2R < r < 0.96R, as the MLS figure):
  cstL2    sigma_T (CST, equilibrated)          -> consistent L2      (the method)
  cstLump  sigma_T                              -> lumped mass (area-weighted average of the adjacent triangles)
  cstRaw   sigma_T at the triangle centroids, no recovery (scored per triangle)
  virL2    edge-force equal split per triangle  -> consistent L2      (control: same projection, the old input)
  mls6/8   nodal virial + quadratic MLS, RHO 6 / 8 (the current figure)

Run (base env): python recover_stress_l2.py
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import project_self_stress as PS                      # noqa: E402
import project_self_stress_cst as PC                  # noqa: E402
import paraboloid_profile as pp                       # noqa: E402
import local_stress_recovery as LR                     # noqa: E402

import paraboloid_dish_config as CFG
TEMP = CFG.TEMP_DIR
RUNGS = [("73", "paraboloid_edge_constq_u0073_odt_qual1e-4_warm"),
         ("289", "paraboloid_edge_constq_u0289_odt_qual1e-4_warm"),
         ("1158", "paraboloid_edge_constq_u1158_odt_qual1e-4_warm"),
         ("4626", "paraboloid_edge_constq_u4626_odt_qual1e-4_warm")]
NU = 0.3
R, DEP = LR.R, LR.D
F_FOC = R * R / (4 * DEP)
MM = 2500.0 / R
COMP = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def cst_sigma(V, T, E, fixed, q, nu=NU):
    """project_self_stress_cst.project_cst, returning the triangle stresses too (same algebra, lines 57-77)."""
    A, _B, _L6, cols, free = PS.build(V, T, E, fixed)
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    G, At = PC.triangle_maps(V, T, E)
    Bq = A @ sp.diags(1.0 / L[cols]) @ G[cols]
    b = A @ q[cols]
    h = np.zeros((len(T), 3, 3))
    h[:, 0, 0] = h[:, 1, 1] = 1.0; h[:, 0, 1] = h[:, 1, 0] = -nu; h[:, 2, 2] = 2 * (1 + nu)
    H = sp.block_diag([At[t] * h[t] for t in range(len(T))], format="csr")
    m = Bq.shape[0]
    KKT = sp.bmat([[H, Bq.T], [Bq, -1e-13 * sp.identity(m)]], format="csc")
    sig = spl.spsolve(KKT, np.concatenate([np.zeros(H.shape[0]), b]))[:H.shape[0]]
    q_out = q.copy(); q_out[cols] = (G @ sig / L)[cols]
    return sig.reshape(-1, 3), q_out


def tri_frames(V, T):
    """Per-triangle (e1, e2, n, area, centroid) -- the same in-plane frame as PC.triangle_maps."""
    P0, P1, P2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(P1 - P0, P2 - P0); A = 0.5 * np.linalg.norm(n, axis=1); n /= (2 * A)[:, None]
    e1 = (P1 - P0) / np.linalg.norm(P1 - P0, axis=1)[:, None]
    e2 = np.cross(n, e1)
    return e1, e2, n, A, (P0 + P1 + P2) / 3


def to_global(s, e1, e2):
    """[sxx, syy, sxy] in (e1, e2) -> 3x3 Cartesian tensor."""
    o = lambda a, b: a[:, :, None] * b[:, None, :]   # noqa: E731
    return s[:, 0, None, None] * o(e1, e1) + s[:, 1, None, None] * o(e2, e2) \
        + s[:, 2, None, None] * (o(e1, e2) + o(e2, e1))


def virial_per_triangle(V, T, E, q, At):
    """Control input: each edge force F = qL split EQUALLY over its adjacent triangles, tensor = sum F L t t / A_T."""
    key = PC.edge_index(E)
    ntri = np.zeros(len(E))
    for a, b, c in T:
        for i, j in ((a, b), (b, c), (c, a)):
            ntri[key[(min(i, j), max(i, j))]] += 1
    S = np.zeros((len(T), 3, 3))
    for ti, (a, b, c) in enumerate(T):
        for i, j in ((a, b), (b, c), (c, a)):
            k = key[(min(i, j), max(i, j))]
            d = V[j] - V[i]; L = np.linalg.norm(d); t = d / L
            S[ti] += (q[k] * L / ntri[k]) * L * np.outer(t, t)
        S[ti] /= At[ti]
    return S


def l2_project(V, T, At, S_tri, lumped=False):
    """P1 L2 projection of a per-triangle tensor field: M N = b (six components), then tangent projection."""
    N = len(V)
    rows = np.repeat(T, 3, axis=1).ravel(); cols = np.tile(T, (1, 3)).ravel()
    loc = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], float) / 12.0
    vals = (At[:, None, None] * loc[None]).reshape(len(T), 9).ravel()
    M = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    w = np.zeros(N)
    for k in range(3):
        np.add.at(w, T[:, k], At / 3)
    out = np.zeros((N, 3, 3))
    lu = None if lumped else spl.splu(M.tocsc())
    for a_, b_ in COMP:
        bvec = np.zeros(N)
        for k in range(3):
            np.add.at(bvec, T[:, k], At / 3 * S_tri[:, a_, b_])
        x = bvec / w if lumped else lu.solve(bvec)
        out[:, a_, b_] = out[:, b_, a_] = x
    nrm = np.zeros((N, 3))
    for k in range(3):
        np.add.at(nrm, T[:, k], (np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])))
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    P = np.eye(3)[None] - nrm[:, :, None] * nrm[:, None, :]
    return P @ out @ P


def exact(p, r):
    u = np.sqrt(1 + (r / (2 * F_FOC)) ** 2)
    return p * F_FOC * u, 2 * p * F_FOC * u * (1 - 1 / (2 * u ** 2))


def score(Vpts, S, p):
    r = np.hypot(Vpts[:, 0], Vpts[:, 1])
    m, h = LR.mer_hoop(Vpts, S)
    em, eh = exact(p, r)
    return m / em - 1, h / eh - 1, r


rms = lambda e: 100 * float(np.sqrt(np.nanmean(np.asarray(e) ** 2)))     # noqa: E731


def analyse(d):
    st = np.load(f"{TEMP}/{d}/jaxfdm_paraboloid_dish.npz", allow_pickle=True)
    ref = np.load(f"{TEMP}/{d}/jaxfdm_paraboloid_dish_projected_cst.npz", allow_pickle=True)
    V, T, E, fx = (np.asarray(st[k]) for k in ("vertices", "faces_tri", "edges", "fixed"))
    q = np.asarray(st["q_categories"], float)[np.asarray(st["cat_ids"], int)]
    q_ref = np.asarray(ref["q_categories"], float)[np.asarray(ref["cat_ids"], int)]
    sig, q_cst = cst_sigma(V, T, E, fx, q)
    dq = float(np.max(np.abs(q_cst - q_ref)) / np.max(np.abs(q_ref)))
    p = abs(float(st["GAS_K"]) / pp._enclosed_volume(V, T))
    e1, e2, n, At, C = tri_frames(V, T)
    S_cst = to_global(sig, e1, e2)
    S_vir = virial_per_triangle(V, T, E, q_cst, At)
    inn = np.ones(len(V), bool); inn[fx] = False
    r = np.hypot(V[:, 0], V[:, 1])
    band = inn & (r > 0.2 * R) & (r < 0.96 * R)
    res = dict(n=len(V), dq=dq)
    for key, S_ in (("cstL2", l2_project(V, T, At, S_cst)),
                    ("cstLump", l2_project(V, T, At, S_cst, lumped=True)),
                    ("virL2", l2_project(V, T, At, S_vir))):
        em, eh, _ = score(V, S_, p)
        res[key] = (rms(em[band]), rms(eh[band]))
    em, eh, rc = score(C, S_cst, p)
    bt = (rc > 0.2 * R) & (rc < 0.96 * R)
    res["cstRaw"] = (rms(em[bt]), rms(eh[bt]))
    Sv, Ai, nrm, hmed = LR.nodal_virial(V, T, E, q_cst, fx)
    res["h"] = hmed * MM
    for rho in (6, 8):
        Sr = LR.recover(V, Sv, Ai, nrm, inn, rho * hmed, "mls2")
        em, eh, _ = score(V, Sr, p)
        ok = band & np.isfinite(em) & np.isfinite(eh)
        res[f"mls{rho}"] = (rms(em[ok]), rms(eh[ok]))
    return res


KEYS = ("cstL2", "cstLump", "cstRaw", "virL2", "mls6", "mls8")

if __name__ == "__main__":
    rows = []
    for lab, d in RUNGS:
        a = analyse(d)
        rows.append(a)
        print(f"[{lab:>4s}] h {a['h']:6.1f} mm  q check {a['dq']:.1e}  " +
              "  ".join(f"{k} {a[k][0]:.3f}/{a[k][1]:.3f}%" for k in KEYS), flush=True)
    hm = np.array([r_["h"] for r_ in rows])
    print("observed order N_phi | N_theta  (73->289, 289->1158, 1158->4626):")
    for k in KEYS:
        o = [np.log(np.array([r_[k][c] for r_ in rows])[:-1] / np.array([r_[k][c] for r_ in rows])[1:])
             / np.log(hm[:-1] / hm[1:]) for c in (0, 1)]
        print(f"   {k:8s} " + ", ".join(f"{x:5.2f}" for x in o[0]) + "  |  " + ", ".join(f"{x:5.2f}" for x in o[1]))
    np.savez(os.path.join(TEMP, "stress_l2_recovery_ladder.npz"), h=hm, n=[r_["n"] for r_ in rows],
             **{k: np.array([r_[k] for r_ in rows]) for k in KEYS})
    print("saved", os.path.join(TEMP, "stress_l2_recovery_ladder.npz"))
