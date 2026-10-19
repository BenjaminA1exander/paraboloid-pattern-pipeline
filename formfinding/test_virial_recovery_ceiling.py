r"""test_virial_recovery_ceiling.py -- separate the stress-error sources of the paraboloid study WITHOUT form finding,
on the nested meshes (temp/paraboloid_mesh_nested/mesh_L0..L3.npz, every node on the target surface).

(a) FORMULA test: a uniform isotropic membrane stress (sigma = identity in every triangle's plane) is turned into edge
    forces exactly as the CST projection does (project_self_stress_cst.triangle_maps: F = G sigma), q = F/L, and the
    nodal virial (local_stress_recovery.nodal_virial) is evaluated. Exact answer: the tangent identity at every node.
    Any error is the consistency error of the nodal virial formula itself.
(b) RECOVERY test: the EXACT paraboloid statics (N_phi m m + N_theta h h, p = 1) evaluated at every triangle centroid,
    projected onto the triangle plane, turned into edge forces the same way, then nodal virial + the paper's recovery
    (quadratic MLS, rho = k h_ref sqrt(h/h_ref), h_ref = 159 mm, k = 8; also k = 4) and scored against exact statics on
    0.2R..0.96R. The order of this error is the ceiling set by the extraction alone (zero form-finding error).

Run (base env): python test_virial_recovery_ceiling.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import project_self_stress_cst as PC                  # noqa: E402
import local_stress_recovery as LR                     # noqa: E402

import paraboloid_dish_config as CFG
TEMP = CFG.TEMP_DIR
NEST = f"{TEMP}/paraboloid_mesh_nested"
R, DEP = LR.R, LR.D
F_FOC = R * R / (4 * DEP)
K_CURV = 2.0 * DEP / R ** 2
MM = 2500.0 / R
H_REF = 159.0 / MM
LEVELS = [L for L in range(4) if os.path.exists(f"{NEST}/mesh_L{L}.npz")]


def frames(V, T):
    P0, P1, P2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    n = np.cross(P1 - P0, P2 - P0); n /= np.linalg.norm(n, axis=1)[:, None]
    e1 = (P1 - P0) / np.linalg.norm(P1 - P0, axis=1)[:, None]
    return e1, np.cross(n, e1), (P0 + P1 + P2) / 3


def exact_tensor(X, p=1.0):
    r = np.hypot(X[:, 0], X[:, 1]); th = np.arctan2(X[:, 1], X[:, 0])
    u = np.sqrt(1 + (r / (2 * F_FOC)) ** 2)
    Nm, Nh = p * F_FOC * u, 2 * p * F_FOC * u * (1 - 1 / (2 * u ** 2))
    m = np.column_stack([np.cos(th), np.sin(th), -K_CURV * r]) / np.sqrt(1 + (K_CURV * r) ** 2)[:, None]
    hv = np.column_stack([-np.sin(th), np.cos(th), np.zeros_like(r)])
    return Nm[:, None, None] * m[:, :, None] * m[:, None, :] + Nh[:, None, None] * hv[:, :, None] * hv[:, None, :], Nm, Nh


def rms(e):
    return float(np.sqrt(np.nanmean(np.asarray(e) ** 2)))


def order(v, h):
    v, h = np.asarray(v), np.asarray(h)
    return np.log(v[:-1] / v[1:]) / np.log(h[:-1] / h[1:])


rows = []
for L in LEVELS:
    m_ = np.load(f"{NEST}/mesh_L{L}.npz")
    V, T, rim = np.asarray(m_["vertices"]), np.asarray(m_["tris"]), np.asarray(m_["rim"])
    E = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    Lg = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    G, At = PC.triangle_maps(V, T, E)
    e1, e2, C = frames(V, T)
    inn = np.ones(len(V), bool); inn[rim] = False
    r = np.hypot(V[:, 0], V[:, 1])
    band = inn & (r > 0.2 * R) & (r < 0.96 * R)
    # (a) uniform isotropic stress
    sig = np.tile([1.0, 1.0, 0.0], len(T))
    S, Ai, nrm, h = LR.nodal_virial(V, T, E, G @ sig / Lg, rim)
    P = np.eye(3)[None] - nrm[:, :, None] * nrm[:, None, :]
    dS = S - P
    err_a = np.sqrt(np.einsum("nij,nij->n", dS, dS) / 2.0) * 100       # Frobenius error / ||I_2|| in %
    # (b) exact statics -> edge forces -> virial -> recovery
    Sx, _, _ = exact_tensor(C)
    s11 = np.einsum("ni,nij,nj->n", e1, Sx, e1); s22 = np.einsum("ni,nij,nj->n", e2, Sx, e2)
    s12 = np.einsum("ni,nij,nj->n", e1, Sx, e2)
    sigx = np.column_stack([s11, s22, s12]).ravel()
    Sv, Ai, nrm, h = LR.nodal_virial(V, T, E, G @ sigx / Lg, rim)
    _, Nm, Nh = exact_tensor(V)
    m0, h0 = LR.mer_hoop(V, Sv)
    out = dict(L=L, n=len(V), h=h * MM, a=rms(err_a[band]), a_max=float(np.max(err_a[band])),
               raw=(rms((m0 / Nm - 1)[band] * 100), rms((h0 / Nh - 1)[band] * 100)))
    for k in (8.0, 4.0):
        Sr = LR.recover(V, Sv, Ai, nrm, inn, k * H_REF * np.sqrt(h / H_REF), "mls2")
        mm, hh = LR.mer_hoop(V, Sr)
        ok = band & np.isfinite(mm) & np.isfinite(hh)
        out[f"k{k:g}"] = (rms((mm / Nm - 1)[ok] * 100), rms((hh / Nh - 1)[ok] * 100))
    rows.append(out)
    print(f"[L{L} {len(V):6d} nodes, h {h * MM:5.1f} mm] (a) uniform: virial error rms {out['a']:.3f}% max {out['a_max']:.2f}%"
          f" | (b) exact statics: raw virial {out['raw'][0]:.3f}/{out['raw'][1]:.3f}%  k=8 {out['k8'][0]:.4f}/"
          f"{out['k8'][1]:.4f}%  k=4 {out['k4'][0]:.4f}/{out['k4'][1]:.4f}%", flush=True)

hm = [r_["h"] for r_ in rows]
print("orders (N_phi | N_theta), level to level:")
print(f"  (a) uniform virial error      {np.round(order([r_['a'] for r_ in rows], hm), 2)}")
for key in ("raw", "k8", "k4"):
    print(f"  (b) {key:4s} N_phi {np.round(order([r_[key][0] for r_ in rows], hm), 2)}  "
          f"N_theta {np.round(order([r_[key][1] for r_ in rows], hm), 2)}")
np.savez(f"{NEST}/virial_recovery_ceiling.npz", h=hm, a=[r_["a"] for r_ in rows],
         **{key: np.array([r_[key] for r_ in rows]) for key in ("raw", "k8", "k4")})
