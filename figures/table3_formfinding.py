r"""table3_formfinding.py -- the error norms of Table 3 (Sec. 3.4.1) for every form-found state of the nested ladder.

For a state (jaxfdm_paraboloid_dish.npz) the recovery of the paper is applied: CST self-stress projection of the
force densities (project_self_stress_cst, Eq. 28), nodal virial, quadratic moving least squares with support
rho = 8 h_ref sqrt(h/h_ref), h_ref = 159 mm (local_stress_recovery, Eq. 29-30); then N_phi and N_theta at every
free node are compared with the exact membrane statics of the paraboloid (Eq. 31).  Reported, area-weighted over
all free nodes: the mean, the L1 norm and the L2 norm of N/N_exact - 1 for both resultants, the enclosed-volume
error and the rms normal distance of the free nodes to the paraboloid.

Table 3 of the paper lists L0-L2 after stage 2 and L3 after stage 3 (STATES_PAPER).  STAGE3=1 scores the
stage-3 state of every rung instead (docs/RECIPES.md, stage 1).  Scores are cached in
<data>/rim_onsurface_verify/table3_scores.json by state file and modification time; plot_paraboloid_journal_ff_best.py
reads panel (d) from the same cache.

RUN   python table3_formfinding.py            (base env, a few minutes for L3)
      STAGE3=1 python table3_formfinding.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import paraboloid_dish_config as CFG                   # noqa: E402
import project_self_stress_cst as PC                   # noqa: E402
import paraboloid_profile as pp                        # noqa: E402
import local_stress_recovery as LR                     # noqa: E402

TEMP = CFG.TEMP_DIR
CACHE = os.path.join(TEMP, "rim_onsurface_verify", "table3_scores.json")
R, DEP = LR.R, LR.D
F = R * R / (4 * DEP)
MM = 2500.0 / R
V_EX = np.pi * R ** 2 * DEP / 2
H_REF_MM = 159.0

STAGE2 = {n: f"paraboloid_edge_constq_nest{n}_odt_qual1e-4_warm/jaxfdm_paraboloid_dish.npz" for n in (289, 1105, 4321, 17089)}
STAGE3 = {289: "paraboloid_edge_constq_nest289_stage3_curv/jaxfdm_paraboloid_dish.npz",
          1105: "paraboloid_edge_constq_nest1105_stage3_curv/jaxfdm_paraboloid_dish.npz",
          4321: "paraboloid_edge_constq_nest4321_stage3_curv/jaxfdm_paraboloid_dish.npz",
          17089: "rim_onsurface_verify/curv_17089_step19250.npz"}
STATES_PAPER = [("L0", STAGE2[289]), ("L1", STAGE2[1105]), ("L2", STAGE2[4321]), ("L3", STAGE3[17089])]
STATES_STAGE3 = [("L0", STAGE3[289]), ("L1", STAGE3[1105]), ("L2", STAGE3[4321]), ("L3", STAGE3[17089])]


def score(path):
    st = np.load(path, allow_pickle=True)
    V, T, E, fx = (np.asarray(st[k]) for k in ("vertices", "faces_tri", "edges", "fixed"))
    q = np.asarray(st["q_categories"], float)[np.asarray(st["cat_ids"], int)]
    qc, _ = PC.project_cst(V, T, E, fx, q)
    Vol = abs(pp._enclosed_volume(V, T))
    p = abs(float(st["GAS_K"]) / Vol)
    S, Ai, nrm, h = LR.nodal_virial(V, T, E, qc, fx)
    inn = np.ones(len(V), bool); inn[fx] = False
    h_ref = H_REF_MM / MM
    Sr = LR.recover(V, S, Ai, nrm, inn, 8 * h_ref * np.sqrt(h / h_ref), "mls2")
    m, hh = LR.mer_hoop(V, Sr)
    r = np.hypot(V[:, 0], V[:, 1]); u = np.sqrt(1 + (r / (2 * F)) ** 2)
    em = m / (p * F * u) - 1; eh = hh / (2 * p * F * u * (1 - 1 / (2 * u ** 2))) - 1
    P_ = V[T]; ta = 0.5 * np.linalg.norm(np.cross(P_[:, 1] - P_[:, 0], P_[:, 2] - P_[:, 0]), axis=1)
    A = np.zeros(len(V)); np.add.at(A, T.ravel(), np.repeat(ta / 3, 3))
    k = inn & np.isfinite(em) & np.isfinite(eh); w = A[k] / A[k].sum()
    dn = (V[:, 2] - DEP * (1 - (r / R) ** 2)) / np.sqrt(1 + (2 * DEP * r / R ** 2) ** 2)   # normal distance
    return dict(ne=int(len(T)), ndof=int(3 * (len(V) - len(fx))), n_nodes=int(len(V)), n_pinned=int(len(fx)),
                h_mm=float(h * MM), vol=float(abs(Vol / V_EX - 1)),
                dn_rms=float(np.sqrt((A[inn] / A[inn].sum()) @ dn[inn] ** 2) / R),
                mphi=float(w @ em[k]), mth=float(w @ eh[k]),
                aphi=float(w @ np.abs(em[k])), ath=float(w @ np.abs(eh[k])),
                rphi=float(np.sqrt(w @ em[k] ** 2)), rth=float(np.sqrt(w @ eh[k] ** 2)))


def score_cached(path):
    """score(path) through the json cache (key: path relative to the data root, modification time)."""
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    rel = os.path.relpath(path, TEMP).replace("\\", "/")
    key = f"{rel}|{os.path.getmtime(path):.0f}"
    if key not in cache:
        cache[key] = score(path)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w") as fh:
            json.dump(cache, fh, indent=1)
    return dict(cache[key])


def main():
    states = STATES_STAGE3 if os.environ.get("STAGE3", "0") == "1" else STATES_PAPER
    print("rung  nodes pinned    dof  |V/Vex-1|  dn_rms/R   N_phi mean   L1     L2    N_theta mean   L1     L2   (%)")
    rows = []
    for lev, rel in states:
        c = score_cached(os.path.join(TEMP, rel))
        rows.append((lev, c))
        print(f"{lev}  {c['n_nodes']:6d} {c['n_pinned']:5d} {c['ndof']:7d}   {c['vol']:.2e}  {c['dn_rms']:.1e}   "
              f"{100 * c['mphi']:+.3f}  {100 * c['aphi']:.3f}  {100 * c['rphi']:.3f}      "
              f"{100 * c['mth']:+.3f}  {100 * c['ath']:.3f}  {100 * c['rth']:.3f}")
    dof = np.array([c["ndof"] for _, c in rows], float)
    for key, lab in (("aphi", "N_phi L1"), ("rphi", "N_phi L2"), ("ath", "N_theta L1"), ("rth", "N_theta L2")):
        v = np.array([c[key] for _, c in rows])
        o = np.log(v[:-1] / v[1:]) / np.log(np.sqrt(dof[1:] / dof[:-1]))
        print(f"observed order in h, {lab:11s}: " + "  ".join(f"{x:.1f}" for x in o))


if __name__ == "__main__":
    main()
