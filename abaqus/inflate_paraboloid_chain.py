"""inflate_paraboloid_chain.py -- stage 4b: the two-run Abaqus chain per variant.

RUN 1  paraboloid_asm_{variant}   (Explicit)
    Flat flower of as-cut gores + cap (flower_paraboloid_{variant}.npz).
    Step Drag  : ramped displacement BCs drag ONLY seam/rim/tip/cap-edge nodes
                 to the joined configuration; interiors free.
    Step Settle: BCs held, kinetic energy bleeds off.  *Restart, write.

RUN 2  paraboloid_prs_{variant}   (Explicit, *IMPORT STATE=YES UPDATE=NO)
    Stress/strain/VUMAT state carried over; reference = as-cut panels.
    Drag BCs are NOT re-created (seams free).  *MPC TIE seals the coincident
    seam mates and the cap ring.  Rim held (velocity=0) = the rigid edge.
    Pressure in N_PRESS load-continuation rungs on the correct side per elset.

Both jobs share one dir paraboloid_chain_{variant}/ (import needs oldjob files).
Run:  builder -> this script -> run1.bat -> run2.bat  (ifx paths per session fix)
"""
from __future__ import annotations

import os

import numpy as np

import paraboloid_dish_config as CFG
from chen2022_constitutive import UMAT_PROPS

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA_DAMP = 40.0
DRAG_TIME, DRAG_RAMP, SETTLE_TIME, MAX_INC = 20.0, 16.0, 40.0, 3.0e-3
N_PRESS, P_STEP_TIME, P_RAMP = 4, 3.0, 2.0
TSS_K = (5.0 / 6.0) * UMAT_PROPS[10] * CFG.T_FABRIC


def mean_nz(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    return float(n[:, 2].mean())


def common_model(f, fz, P, F):
    """Nodes, elements, elsets, orientation, material, sections (both runs
    import run 1's model, so only run 1 writes this)."""
    ng_el = int(fz["n_gore_el"])
    n_el_per_gore = ng_el // int(fz["n_gores"])
    warp3d = np.asarray(fz["warp3d"], float)
    weft3d = np.asarray(fz["weft3d"], float)
    f.write("*Node\n")
    for k in range(len(P)):
        f.write(f" {k+1}, {P[k,0]:.5f}, {P[k,1]:.5f}, {P[k,2]:.5f}\n")
    f.write("*Element, type=S3, elset=GORES\n")
    for e, t in enumerate(F[:ng_el], 1):
        f.write(f" {e}, {t[0]+1}, {t[1]+1}, {t[2]+1}\n")
    f.write("*Element, type=S3, elset=CAP\n")
    for e, t in enumerate(F[ng_el:], ng_el + 1):
        f.write(f" {e}, {t[0]+1}, {t[1]+1}, {t[2]+1}\n")
    f.write("*Elset, elset=ALL_ELEMENTS\n GORES, CAP\n")

    def nset(name, ids):
        f.write(f"*Nset, nset={name}\n")
        ids = list(ids)
        for k in range(0, len(ids), 12):
            f.write(" " + ", ".join(str(i + 1) for i in ids[k:k+12]) + "\n")
    nset("RIM", sorted(int(r) for r in fz["rim_ids"]))

    # per-element fabric orientation for the gores (EMPIRICAL column order:
    # local-1 follows the SECOND vector -> write (weft, warp)).  Cap: default.
    f.write("**\n*Distribution Table, name=ORI_TAB\n COORD3D, COORD3D\n")
    f.write("*Distribution, name=ORI_DIST, location=ELEMENT, table=ORI_TAB\n")
    f.write(" , 0.0, 1.0, 0.0, 1.0, 0.0, 0.0\n")
    for e in range(ng_el):
        g = e // n_el_per_gore
        a, b = warp3d[g], weft3d[g]
        f.write(f" {e+1}, {b[0]:.7f}, {b[1]:.7f}, {b[2]:.7f}, "
                f"{a[0]:.7f}, {a[1]:.7f}, {a[2]:.7f}\n")
    f.write("*Orientation, name=FABRIC_ORI, system=RECTANGULAR, "
            "definition=COORDINATES\n ORI_DIST\n")

    f.write("**\n*Material, name=FABRIC\n")
    f.write(f"*Density\n {CFG.DENS}\n*Damping, alpha={ALPHA_DAMP}\n")
    f.write("*Depvar\n 6\n")
    f.write(f"*User Material, constants={len(UMAT_PROPS)}\n")
    for k in range(0, len(UMAT_PROPS), 8):
        f.write(" " + ", ".join(f"{p}" for p in UMAT_PROPS[k:k+8]) + "\n")
    f.write("**\n*Section Controls, name=BODY_CTRL, hourglass=ENHANCED\n"
            " 1.0, 1.0, 1.0\n")
    f.write("*Shell Section, elset=GORES, material=FABRIC, "
            "orientation=FABRIC_ORI, controls=BODY_CTRL\n"
            f" {CFG.T_FABRIC}, 5\n")
    f.write(f"*Transverse Shear Stiffness\n {TSS_K:.6f}, {TSS_K:.6f}\n")
    f.write("*Shell Section, elset=CAP, material=FABRIC, controls=BODY_CTRL\n"
            f" {CFG.T_FABRIC}, 5\n")
    f.write(f"*Transverse Shear Stiffness\n {TSS_K:.6f}, {TSS_K:.6f}\n")


OUT = ("*Output, field, time interval=%.2f\n*Node Output\n U, COORD\n"
       "*Element Output, directions=YES\n S, E, SDV\n"
       "*Output, history, time interval=0.05\n"
       "*Energy Output\n ALLKE, ALLIE, ALLAE, ALLWK, ETOTAL\n")


def write_run1(fz, path, variant):
    P = np.asarray(fz["nodes"], float)
    F = np.asarray(fz["tris"], int)
    dn = np.asarray(fz["drag_nodes"], int)
    dt = np.asarray(fz["drag_targets"], float)
    disp = {int(n): dt[k] - P[int(n)] for k, n in enumerate(dn)}

    with open(path, "w") as f:
        f.write(f"*Heading\n Paraboloid dish {variant}: RUN 1 assemble -- drag flat"
                f" flower edges to the joined configuration\n**\n")
        common_model(f, fz, P, F)
        f.write(f"*Amplitude, name=RAMP, definition=SMOOTH STEP\n"
                f" 0.0, 0.0, {DRAG_RAMP:.2f}, 1.0\n")

        def wbc(amp):
            f.write(f"*Boundary{amp}\n")
            for n in sorted(disp):
                u = disp[n]
                f.write(f" {n+1}, 1,1, {u[0]:.5f}\n {n+1}, 2,2, {u[1]:.5f}\n"
                        f" {n+1}, 3,3, {u[2]:.5f}\n")
        f.write(f"*Step, name=Drag\n*Dynamic, Explicit\n , {DRAG_TIME}, , {MAX_INC}\n")
        f.write(f"*Variable Mass Scaling, dt={MAX_INC}, type=below min, frequency=1\n")
        wbc(", amplitude=RAMP")
        f.write(OUT % 1.0 + "*End Step\n")
        f.write(f"*Step, name=Settle\n*Dynamic, Explicit\n , {SETTLE_TIME}, , {MAX_INC}\n")
        f.write(f"*Variable Mass Scaling, dt={MAX_INC}, type=below min, frequency=1\n")
        wbc("")
        f.write("*Restart, write, number interval=1\n")
        f.write(OUT % 0.5 + "*End Step\n")


def write_run2(fz, path, variant, job1):
    P = np.asarray(fz["nodes"], float)
    F = np.asarray(fz["tris"], int)
    ng_el = int(fz["n_gore_el"])
    seal = np.asarray(fz["seal_pairs"], int)
    # FE pressure: pattern-design value unless overridden (e.g. bias recal runs)
    press = float(os.environ.get("FORMFOUND_FE_PRESS", "0")) or float(fz["pressure"])
    # pressure must push +z (inflate the dome upward); the fold preserves the
    # sign of each element's normal z, so pick the side per elset from the FLAT
    # layout: positive pressure acts opposite the chosen face's normal.
    side_g = "SPOS" if mean_nz(P, F[:ng_el]) < 0 else "SNEG"
    side_c = "SPOS" if mean_nz(P, F[ng_el:]) < 0 else "SNEG"

    with open(path, "w") as f:
        f.write(f"*Heading\n Paraboloid dish {variant}: RUN 2 seal + pressurise"
                f" (import from {job1})\n**\n")
        f.write("*Import, state=yes, update=no\n ALL_ELEMENTS,\n")
        f.write("*Import Elset\n GORES, CAP, ALL_ELEMENTS\n")
        f.write("*Import Nset\n RIM\n")
        # SEAL: weld the coincident seam mates + cap ring (dep, indep)
        f.write("**\n*MPC\n")
        for a, b in seal:
            f.write(f" TIE, {int(a)+1}, {int(b)+1}\n")
        for k in range(N_PRESS):
            f0, f1 = k / N_PRESS, (k + 1) / N_PRESS
            f.write(f"*Amplitude, name=PR{k+1}, definition=SMOOTH STEP\n"
                    f" 0.0, {f0:.4f}, {P_RAMP:.2f}, {f1:.4f}\n")
        f.write(f"*Surface, name=SG, type=ELEMENT\n GORES, {side_g}\n")
        f.write(f"*Surface, name=SC, type=ELEMENT\n CAP, {side_c}\n")
        for k in range(N_PRESS):
            f.write(f"*Step, name=Press{k+1}\n*Dynamic, Explicit\n"
                    f" , {P_STEP_TIME}, , {MAX_INC}\n")
            f.write(f"*Variable Mass Scaling, dt={MAX_INC}, "
                    "type=below min, frequency=1\n")
            f.write("*Boundary, type=VELOCITY\n RIM, 1, 3, 0.0\n")
            f.write(f"*Dsload, amplitude=PR{k+1}\n SG, P, {press:.6f}\n"
                    f" SC, P, {press:.6f}\n")
            f.write(OUT % (1.0 if k < N_PRESS - 1 else 0.5) + "*End Step\n")


def validate(path, run2=False):
    low = open(path).read().lower()
    assert "*dynamic, explicit" in low and "*static" not in low
    assert "*variable mass scaling" in low
    if run2:
        assert "*import, state=yes, update=no" in low
        assert "*mpc" in low and " tie," in low
        assert "*dsload" in low and "type=velocity" in low
        assert "amplitude=ramp" not in low, "run 2 must not re-create drag BCs"
    else:
        assert "*user material, constants=24" in low and "*depvar" in low
        assert "*elastic" not in low and "*no compression" not in low
        assert "*transverse shear stiffness" in low
        assert "orientation=fabric_ori" in low
        assert "*restart, write" in low
        assert "*dsload" not in low, "no pressure in run 1"


def main():
    for variant in ("vares", "naive"):
        fz = np.load(os.path.join(HERE, f"flower_paraboloid_{variant}.npz"),
                     allow_pickle=True)
        jd = os.path.join(HERE, f"paraboloid_chain_{variant}")
        os.makedirs(jd, exist_ok=True)
        j1, j2 = f"paraboloid_asm_{variant}", f"paraboloid_prs_{variant}"
        write_run1(fz, os.path.join(jd, j1 + ".inp"), variant)
        write_run2(fz, os.path.join(jd, j2 + ".inp"), variant, j1)
        validate(os.path.join(jd, j1 + ".inp"))
        validate(os.path.join(jd, j2 + ".inp"), run2=True)
        for j, extra in ((j1, ""), (j2, f" oldjob={j1}")):
            with open(os.path.join(jd, f"run_{j}.bat"), "w") as f:
                f.write("@echo off\r\n")
                f.write(f"call \"{CFG.ABAQUS_BAT}\" job={j}{extra} "
                        f"user=..\\vumat_chen2022.for cpus=6 interactive "
                        f"ask_delete=OFF\r\n")
        print(f"[chain:{variant}] {jd}\\{j1}.inp + {j2}.inp  decks OK")


if __name__ == "__main__":
    main()
