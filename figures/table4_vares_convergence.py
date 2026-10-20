"""table4_vares_convergence.py -- Table 4 of the paper (Sec. 3.4.2): the 200 mm nested chain (+ the 100 mm chain's 6.25 mm rung).
Per rung: triangles, nodes, N = 2n - 3, Newton its, final ||r||, h_eff, Delta (compensation vs developed leaf, rigid
removed, area weighted) in L1 and L2; successive pattern change (coarse P1 field at the fine nodes, rigid removed,
area weighted on the fine leaf) L1/L2 + observed orders; Celik GCI_fine on Delta_L2 and Delta_L1.
RUN   python table4_vares_convergence.py      (base env)      OUT  <data>/results/convergence_loglog/conv_table.json"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paraboloid_dish_config as CFG
import MeshStudy_paraboloid_vares_ckpt as CK
import vares_study as MS
D = 5000.0

def areas(P, T):
    e = P[T]
    return 0.5 * np.abs((e[:, 1, 0] - e[:, 0, 0]) * (e[:, 2, 1] - e[:, 0, 1]) - (e[:, 2, 0] - e[:, 0, 0]) * (e[:, 1, 1] - e[:, 0, 1]))

def lumped(P, T):
    A = areas(P, T); w = np.zeros(len(P)); np.add.at(w, T.ravel(), np.repeat(A / 3.0, 3)); return w, A

def align(Pm, Pf, w):                      # rigid Q Pm ~ Pf (area weighted), returns Q Pm
    ca, cb = (w[:, None] * Pm).sum(0) / w.sum(), (w[:, None] * Pf).sum(0) / w.sum()
    U, _, Vt = np.linalg.svd(((Pm - ca) * w[:, None]).T @ (Pf - cb))
    if np.linalg.det(U @ Vt) < 0: Vt[-1] *= -1
    return (Pm - ca) @ (U @ Vt) + cb

def norms(u, Xl, T):                       # area-weighted L1 (lumped) and L2 (exact P1) of the vector field u
    w, A = lumped(Xl, T)
    L1 = float((w * np.linalg.norm(u, axis=1)).sum() / w.sum())
    ue = u[T]
    m2 = ((ue ** 2).sum((1, 2)) + sum((ue[:, i] * ue[:, (i + 1) % 3]).sum(1) for i in range(3))) / 6.0
    return L1, float(np.sqrt((A * m2).sum() / A.sum()))

def rung(grading, h):
    CK.configure("cstraw", grading)
    d = np.load(MS.rung_npz(h), allow_pickle=True)
    T = np.asarray(d["elements"]); Xl = d["X_leaf"][:, :2]; P = d["X_converged"][:, :2]
    w, _ = lumped(Xl, T)
    u = align(P, Xl, w) - Xl
    e = d["X_spatial"][T]
    heff = float(np.sqrt(4.0 * 0.5 * np.linalg.norm(np.cross(e[:, 1] - e[:, 0], e[:, 2] - e[:, 0]), axis=1).sum() / (np.sqrt(3.0) * len(T))))
    return dict(tag=MS.tag(h), T=T, Xl=Xl, P=P, n=len(Xl), ne=len(T), N=2 * len(Xl) - 3, its=len(d["R_history"]),
                R=float(d["R_history"][-1]), crit=str(d["converged_criterion"]) if "converged_criterion" in d.files else "tol",
                heff=heff, Delta=norms(u, Xl, T))

rows = [rung("tippolar210c200nest", h) for h in (200.0, 100.0, 50.0, 25.0, 12.5)]
x6 = rung("tippolar210nest", 6.25)
x12 = rung("tippolar210nest", 12.5)
out = []
for i, r in enumerate(rows):
    s = dict(tag=r["tag"], tris=r["ne"], nodes=r["n"], N=r["N"], its=r["its"], R=r["R"], crit=r["crit"], heff=r["heff"],
             Delta_L1=r["Delta"][0], Delta_L2=r["Delta"][1])
    if i + 1 < len(rows):
        f = rows[i + 1]
        Pc = MS.p1_sample(r["Xl"], r["T"], r["P"], f["Xl"])     # coarse P1 pattern at the fine leaf nodes
        wf, _ = lumped(f["Xl"], f["T"])
        dd = f["P"] - align(Pc, f["P"], wf)
        s["chg_L1"], s["chg_L2"] = norms(dd, f["Xl"], f["T"])
    out.append(s)
for i in range(1, len(out) - 1):
    for k in ("chg_L1", "chg_L2"):
        out[i]["p_" + k] = float(np.log2(out[i - 1][k] / out[i][k]))
print("%-5s %6s %6s %6s %4s %9s %-14s %6s %9s %9s %10s %10s %5s %5s" % ("rung", "tris", "nodes", "N", "its", "||r||", "crit", "heff", "D_L1/D", "D_L2/D", "chg_L1/D", "chg_L2/D", "p1", "p2"))
for s in out:
    print("%-5s %6d %6d %6d %4d %9.2e %-14s %6.2f %9.5e %9.5e %10s %10s %5s %5s" % (
        s["tag"], s["tris"], s["nodes"], s["N"], s["its"], s["R"], s["crit"], s["heff"], s["Delta_L1"] / D, s["Delta_L2"] / D,
        "%.3e" % (s["chg_L1"] / D) if "chg_L1" in s else "-", "%.3e" % (s["chg_L2"] / D) if "chg_L2" in s else "-",
        "%.2f" % s["p_chg_L1"] if "p_chg_L1" in s else "-", "%.2f" % s["p_chg_L2"] if "p_chg_L2" in s else "-"))
print("100 mm chain h012: N %d, Delta L1 %.6f L2 %.6f mm ; h006: N %d its %d ||r|| %.2e heff %.2f Delta L1 %.6f L2 %.6f mm" % (
    x12["N"], *x12["Delta"], x6["N"], x6["its"], x6["R"], x6["heff"], *x6["Delta"]))
for j, nm in ((0, "L1"), (1, "L2")):
    f1, f2, f3 = (out[k]["Delta_" + nm] for k in (4, 3, 2))
    e21, e32 = f2 - f1, f3 - f2
    p = np.log(abs(e32 / e21)) / np.log(2.0)
    ext = (2 ** p * f1 - f2) / (2 ** p - 1)
    gci = 1.25 * abs(e21 / f1) / (2 ** p - 1)
    print("Delta_%s: phi(50,25,12.5) = %.6f %.6f %.6f mm; %s; p = %.3f; ext %.6f mm; GCI_fine %.3e (%.2e mm); 100-chain h006 %.6f (vs ext %+.2e mm)" % (
        nm, f3, f2, f1, "monotone" if e32 * e21 > 0 else "oscillatory", p, ext, gci, gci * f1, x6["Delta"][j], x6["Delta"][j] - ext))
    print("   relative change successive: %s" % ["%.2e" % (abs(out[k + 1]["Delta_" + nm] - out[k]["Delta_" + nm]) / out[k + 1]["Delta_" + nm]) for k in range(4)])
OUT = os.path.join(CFG.TEMP_DIR, "results", "convergence_loglog", "conv_table.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print("[saved]", OUT)
