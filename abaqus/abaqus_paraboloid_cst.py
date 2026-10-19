r"""abaqus_paraboloid_cst.py -- Abaqus/Explicit validation of a CST-target VaReS gore rung
(temp/meshstudy_vares_cst/hNNN/pattern.npz), three cutting patterns side by side:

    vares    X_converged             the converged VaReS pattern of that rung
    naive    X_leaf                  the developed leaf, no compensation (unoptimised)
    uniform  X_leaf x (1+c_m, 1+c_w) one constant warp and one constant weft shrinkage over the
                                     whole gore: the best guess WITHOUT an optimiser (below)

Recipe = the validated stage-4b chain (make_paraboloid_flower.py + inflate_paraboloid_chain.py,
corrected by the _p050 runs, memory paraboloid-fe-pressure-equilibrium):
    RUN 1  asm  flat flower of the 24 as-cut gores; Drag EVERY node onto its form-found position,
                Settle with only the seam/rim/tip nodes held (interior released); restart written.
    RUN 2  prs  *Import state; *MPC TIE the seam mates; rim held; design pressure 0.05 MPa in
                4 smooth rungs of 3 s, then a 120 s hold at constant pressure.
Same Chen VUMAT, S3 shells, fabric orientation, *Variable Mass Scaling dt = 3e-3.

WHAT DIFFERS FROM THE OLD CHAIN, AND WHY
  * Full tip, no cap.  The CST gore runs to r = 0 (one tip node per gore), so there is no pole
    cap: every tip node is dragged to the apex and gores 1..23 are TIEd to gore 0's tip.
  * Drag targets are the form-found positions themselves: gore g's node i goes to
    Rot(2 theta g) X_spatial[i] (gore 0 spans azimuth 0..2 theta).  The old interpolated fold
    reproduces exactly this for seam nodes; the two seam mates must coincide (asserted).
  * Whole-panel drag (user decision 2026-10-05).  The old chain dragged only the edges and let the
    interior follow by inertia.  At 12.5 mm that crushes the first element row at the rim (the
    rim moves ~350 mm INTO its own panel, 2850 -> 2500 mm, and fabric has no compressive
    stiffness): LE11 -7.9 there, added mass x800, blow-up at 3.1 s even with bulk viscosity.  Now
    the Drag step takes every node to Rot X_spatial and Settle (*Boundary, op=NEW) keeps only the
    edges, so the interior relaxes before pressure.  The as-cut panel is still the reference.
  * Seam / rim / tip sets are classified on X_LEAF only (topology), never on the pattern.
  * Damping scales with the mesh.  Variable mass scaling to a fixed dt makes every element's
    mass ~ E t dt^2 A / L_e^2, so the scaled mass grows ~1/h^2 (12.5 mm: ~44x the validated
    100 mm model).  The slow, over-damped relaxation that the 120 s hold has to outlast has
    tau ~ alpha M / K, so ALPHA = 40 x M_ref / M_this keeps tau at the validated ~7 s (and
    the damping force alpha M v unchanged).  M is predicted from the flat mesh with
    L_e = 2A / longest edge; only the RATIO is used.  Keeping alpha = 40 would stretch tau
    ~44x, far past the hold; shrinking dt instead would cost ~6.6x the run time.
  * Output thinned (field frames per step, *Restart OVERLAY): at 12.5 mm the old output
    settings would write ~20 GB per variant.

UNIFORM SHRINKAGE -- THE BEST GUESS
  Per gore element the Chen law is inverted at the target stress (sx*, sy*) from the CST field
  (pattern.npz "T"; shear < 0.04 MPa, ignored) to Green-Lagrange (Ex, Ey); the stretch is
  lam = sqrt(1 + 2E) and the local compensation 1/lam - 1.  The uniform value per direction is
  its area-weighted mean over the inflated gore (least-squares best single factor for local
  lengths).  It uses only the target and the material law, never the VaReS result; the VaReS
  means are printed beside it for comparison.  T is the physical (Cauchy) target and the law is
  2nd-PK / Green-Lagrange; at ~1 % strain the difference moves c by ~0.01 percentage points.

USAGE (base anaconda env)
    python abaqus_paraboloid_cst.py --build  [--rung 12.5]          # decks + bats, no solve
    python abaqus_paraboloid_cst.py --launch [--rung 12.5]          # detached sequential queue
    python abaqus_paraboloid_cst.py --run    [--rung 12.5]          # the queue itself (blocking)
    python abaqus_paraboloid_cst.py --check  [--rung 12.5]          # convergence + shape/stress
Layout:  temp/abaqus_vares_cst/hNNN/{vares,uniform,naive}/  (+ setup.json, queue.log)
The queue is resumable: finished jobs are skipped, a failed asm/prs is rerun.  After a variant
passes the equilibrium check its restart files are deleted (odb + extracted npz kept).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paraboloid_dish_config as CFG                  # noqa: E402
import inflate_paraboloid_chain as IC                 # noqa: E402  validated recipe constants
import MeshStudy_paraboloid_vares as MS               # noqa: E402
import MeshStudy_paraboloid_vares_ckpt as CK          # noqa: E402
from chen2022_constitutive import UMAT_PROPS, chen_stress   # noqa: E402

ROOT = os.path.normpath(os.path.join(CK.ROOTS["cst"], "..", "abaqus_vares_cst"))
VARIANTS = ("vares", "uniform", "naive")              # queue order: most important first
P_DESIGN = CFG.P_PHYS                                 # 0.05 MPa, what the targets were built at
HOLD_TIME = 120.0
ALPHA_REF = IC.ALPHA_DAMP                             # 40, validated on the reference flower
REF_FLOWER = os.path.join(HERE, "flower_paraboloid_vares.npz")
LINE_TOL = 1.0                                        # mm, as make_paraboloid_flower
CPUS = 6
MIN_FREE_GB = 8.0                                     # refuse to start a job below this
RF_OK = 0.98                                          # sum RF3 / applied needed to pass
RF_DRIFT_OK = 1e-3                                    # |RF3 change| / applied over the last 10 s
IE_DRIFT_OK = 5e-4                                    # |ALLIE change| / ALLIE over the last 10 s
CREEP_OK = 0.1                                        # mm: max node motion over the last 30 s hold frame
LIC_HOST, LIC_PORT = "licence-server.example.org", 1973
VUMAT = os.path.join(HERE, "vumat_chen2022.for")
ENV_LINES = (                                         # memory: abaqus-vumat-compiler-env
    'set "VS2022INSTALLDIR=C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools"',
    'call "C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat" --force > nul 2>&1',
    'set "PATH=C:\\Program Files (x86)\\Intel\\oneAPI\\2025.3\\bin;%PATH%"',
    'set "LIB=C:\\Program Files (x86)\\Intel\\oneAPI\\2025.3\\lib;%LIB%"',
    'set "INCLUDE=C:\\Program Files (x86)\\Intel\\oneAPI\\2025.3\\include;%INCLUDE%"',
)
RESTART_EXT = (".abq", ".stt", ".res", ".pac", ".mdl", ".sel", ".prt")
# STABILITY (2026-10-05, 12.5 mm): the stock deck blows up within ~13 increments whatever alpha,
# precision or dt.  Tests (temp/abaqus_vares_cst/stabtest): the VUMAT's start-up viscosity
# (CVISC = 0.01 s, stress += CVISC E de/DT, invisible to the stable-time estimate) is harmful --
# 0.001 gives NaN at increment 2, 0 delays the blow-up; Abaqus bulk viscosity, which IS in the
# stable-time estimate, is the lever: b1 0.2 fails at 0.06 s, b1 0.5 passes.  Both are rate terms
# only, so the converged (at-rest) state is unaffected.  The 100 mm rung ran with the stock values.
VUMAT_CVISC = 0.0
BULK_VISC = (0.5, 1.2)
VUMAT_RUN = "vumat_run.for"
VUMAT_STOCK = "vumat_stock.for"                       # unmodified copy (CVISC 0.01)
# naive's assembled state is ~half slack; with CVISC = 0 the packager rejects it ("zero or negative
# initial dilatational modulus") because the VUMAT's start-of-step probe returns no stiffness for
# wrinkled elements.  The stock VUMAT's start-up viscosity makes the probe positive and its prs ran
# fine at 12.5 mm (2026-10-05), so naive's prs uses it; every other job keeps VUMAT_RUN.
PRS_STOCK_VUMAT = ("naive",)
LADDER_UP = (12.5, 17.7, 25.0, 35.4, 50.0, 70.7, 100.0, 141.4, 200.0)   # fine -> coarse
# overrides for comparison runs (--dt / --alpha / --suffix); defaults reproduce the baseline
DT = IC.MAX_INC                                       # explicit target increment (mass scaling)
ALPHA_FIX = None                                      # fixed mass-proportional damping, else the rule
SUFFIX = ""                                           # variant folder suffix, e.g. "_a013"


def rung_dir(h):
    return os.path.join(ROOT, MS.tag(h))


def load_rung(h):
    p = os.path.join(CK.ROOTS["cst"], MS.tag(h), "pattern.npz")
    d = np.load(p, allow_pickle=True)
    if not bool(d["converged"]):
        raise SystemExit(f"{p} is not converged")
    if str(d["target"]) != "cst":
        raise SystemExit(f"{p} is not a CST-target rung")
    return d


# ---------------------------------------------------------------------------------------------
# uniform best-guess compensation
# ---------------------------------------------------------------------------------------------

def invert_chen(sx, sy, e0=(0.01, 0.01)):
    """Green-Lagrange (Ex, Ey) with chen_stress = (sx, sy), zero shear; Newton on the 2x2."""
    E = np.array(e0, float)
    for _ in range(50):
        S, D, _ = chen_stress(np.array([[E[0], 0.0], [0.0, E[1]]]))
        r = np.array([S[0, 0] - sx, S[1, 1] - sy])
        if np.abs(r).max() < 1e-10:
            return E
        E -= np.linalg.solve(D[:2, :2], r)
    raise RuntimeError(f"Chen inversion did not converge at ({sx}, {sy})")


def uniform_compensation(d):
    Xs, el, T = d["X_spatial"], d["elements"], d["T"]
    e = Xs[el]
    A = 0.5 * np.linalg.norm(np.cross(e[:, 1] - e[:, 0], e[:, 2] - e[:, 0]), axis=1)
    E = np.array([invert_chen(t[0], t[1]) for t in T])
    c = 1.0 / np.sqrt(1.0 + 2.0 * E) - 1.0                    # (warp, weft) per element
    c_m, c_w = (A[:, None] * c).sum(0) / A.sum()
    # for comparison only: one inversion at the area-mean stress, and the VaReS means
    Tm = (A[:, None] * T[:, :2]).sum(0) / A.sum()
    c_at_mean = 1.0 / np.sqrt(1.0 + 2.0 * invert_chen(*Tm)) - 1.0
    Xl, Xc, seam = d["X_leaf"][:, :2], d["X_converged"][:, :2], d["seam_ids"]
    rim = np.abs(Xl[:, 0] - Xl[:, 0].min()) < LINE_TOL
    L_leaf = Xl[:, 0].max() - Xl[:, 0].min()
    L_vares = Xc[0, 0] - Xc[rim].mean(0)[0]                   # node 0 = tip
    return dict(c_m=float(c_m), c_w=float(c_w),
                c_m_range=[float(c[:, 0].min()), float(c[:, 0].max())],
                c_w_range=[float(c[:, 1].min()), float(c[:, 1].max())],
                c_at_mean_stress=[float(c_at_mean[0]), float(c_at_mean[1])],
                mean_stress_mpa=[float(Tm[0]), float(Tm[1])],
                vares_length_ratio_minus_1=float(L_vares / L_leaf - 1.0),
                vares_rim_width_ratio_minus_1=float(
                    np.ptp(Xc[rim][:, 1]) / np.ptp(Xl[rim][:, 1]) - 1.0),
                n_seam=int(len(seam)))


# ---------------------------------------------------------------------------------------------
# flower: flat layout, drag targets, seals
# ---------------------------------------------------------------------------------------------

def topology(d):
    Xl = d["X_leaf"][:, :2]
    seam = np.asarray(d["seam_ids"], int)
    rim = np.where(np.abs(Xl[:, 0] - Xl[:, 0].min()) < LINE_TOL)[0]
    tip = np.where(np.abs(Xl[:, 0] - Xl[:, 0].max()) < LINE_TOL)[0]
    if len(tip) != 1:
        raise SystemExit(f"expected a pointed full-tip gore, found {len(tip)} tip nodes")
    sp = [int(i) for i in seam if Xl[i, 1] > 1e-9]
    sm = [int(i) for i in seam if Xl[i, 1] < -1e-9]
    key = {(round(Xl[j, 0], 6), round(-Xl[j, 1], 6)): j for j in sm}
    pair = {i: key[(round(Xl[i, 0], 6), round(Xl[i, 1], 6))] for i in sp}   # exact mirror
    return dict(rim=rim, tip=int(tip[0]), sp=sp, pair=pair)


def rot(g, th):
    a = 2.0 * th * g
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def build_flower(d, Xf, topo):
    ng = int(d["n_gores"])
    th = np.pi / ng
    Xs = np.asarray(d["X_spatial"], float)
    el = np.asarray(d["elements"], int)
    N = len(Xs)
    rimset = set(topo["rim"].tolist())
    fx, fy = Xf[:, 0], Xf[:, 1]
    tip = topo["tip"]

    nodes = np.zeros((ng * N, 3))
    warp3d, weft3d = np.zeros((ng, 3)), np.zeros((ng, 3))
    for g in range(ng):
        a = 2 * th * g + th
        ur = np.array([np.cos(a), np.sin(a), 0.0])
        ut = np.array([-np.sin(a), np.cos(a), 0.0])
        rho = fx[tip] - fx                                   # tip at the origin, rim outward
        nodes[g * N:(g + 1) * N] = rho[:, None] * ur + fy[:, None] * ut
        warp3d[g], weft3d[g] = -ur, ut                       # warp = meridional (rim -> tip)
    tris = np.vstack([el + g * N for g in range(ng)])

    tgt = lambda g, i: rot(g, th) @ Xs[i]                    # noqa: E731
    drag, seal, gap = {}, [], 0.0
    apex = np.array([0.0, 0.0, Xs[tip, 2]])
    for g in range(ng):
        gn = (g + 1) % ng
        for i in topo["sp"]:
            j = topo["pair"][i]
            a, b = tgt(g, i), tgt(gn, j)
            gap = max(gap, float(np.linalg.norm(a - b)))
            drag[g * N + i] = drag[gn * N + j] = 0.5 * (a + b)
            if i not in rimset:
                seal.append((g * N + i, gn * N + j))          # dep = +seam, indep = mate
        for i in topo["rim"]:
            drag.setdefault(g * N + int(i), tgt(g, int(i)))
        drag[g * N + tip] = apex
        if g:
            seal.append((g * N + tip, tip))                   # all tips onto gore 0's tip
    if gap > 1e-6:
        raise SystemExit(f"seam mates' form-found positions differ by {gap:.2e} mm")
    dep = [a for a, _ in seal]
    assert len(dep) == len(set(dep)), "duplicate dependent node in seal pairs"
    rim_ids = sorted(g * N + int(i) for g in range(ng) for i in topo["rim"])
    # Drag step moves EVERY node to its form-found position (edges keep the targets above);
    # Settle holds only the edges, so the interior relaxes freely before pressure
    drag_all = {g * N + i: tgt(g, i) for g in range(ng) for i in range(N)}
    drag_all.update(drag)
    return dict(nodes=nodes, tris=tris, n_el_per_gore=len(el), n_gores=ng, drag=drag,
                drag_all=drag_all,
                seal=np.asarray(seal, int), rim_ids=rim_ids, warp3d=warp3d, weft3d=weft3d,
                ring_r=float(np.hypot(*Xs[topo["rim"], :2].T).mean()),
                ring_z=float(Xs[topo["rim"], 2].mean()), apex_z=float(apex[2]))


def mass_index(P, F):
    """sum A / L_e^2 over the flat mesh, L_e = 2A / longest edge (scaled mass ~ this)."""
    e = P[F]
    A = 0.5 * np.linalg.norm(np.cross(e[:, 1] - e[:, 0], e[:, 2] - e[:, 0]), axis=1)
    lmax = np.max([np.linalg.norm(e[:, k] - e[:, (k + 1) % 3], axis=1) for k in range(3)], axis=0)
    return float(np.sum(A / (2.0 * A / lmax) ** 2))


# ---------------------------------------------------------------------------------------------
# decks
# ---------------------------------------------------------------------------------------------

def write_model(f, fl, alpha):
    P, F = fl["nodes"], fl["tris"]
    f.write("*Node\n")
    f.writelines(f" {k+1}, {p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f}\n" for k, p in enumerate(P))
    f.write("*Element, type=S3, elset=GORES\n")
    f.writelines(f" {e}, {t[0]+1}, {t[1]+1}, {t[2]+1}\n" for e, t in enumerate(F, 1))
    f.write("*Elset, elset=ALL_ELEMENTS\n GORES\n")
    ids = fl["rim_ids"]
    f.write("*Nset, nset=RIM\n")
    for k in range(0, len(ids), 12):
        f.write(" " + ", ".join(str(i + 1) for i in ids[k:k + 12]) + "\n")
    # per-element fabric orientation; EMPIRICAL column order (weft, warp) -- as the old chain
    f.write("**\n*Distribution Table, name=ORI_TAB\n COORD3D, COORD3D\n")
    f.write("*Distribution, name=ORI_DIST, location=ELEMENT, table=ORI_TAB\n")
    f.write(" , 0.0, 1.0, 0.0, 1.0, 0.0, 0.0\n")
    ne = fl["n_el_per_gore"]
    for e in range(len(F)):
        a, b = fl["warp3d"][e // ne], fl["weft3d"][e // ne]
        f.write(f" {e+1}, {b[0]:.7f}, {b[1]:.7f}, {b[2]:.7f}, "
                f"{a[0]:.7f}, {a[1]:.7f}, {a[2]:.7f}\n")
    f.write("*Orientation, name=FABRIC_ORI, system=RECTANGULAR, definition=COORDINATES\n"
            " ORI_DIST\n")
    f.write("**\n*Material, name=FABRIC\n")
    f.write(f"*Density\n {CFG.DENS}\n*Damping, alpha={alpha:.6g}\n*Depvar\n 6\n")
    f.write(f"*User Material, constants={len(UMAT_PROPS)}\n")
    for k in range(0, len(UMAT_PROPS), 8):
        f.write(" " + ", ".join(f"{p}" for p in UMAT_PROPS[k:k + 8]) + "\n")
    f.write("**\n*Section Controls, name=BODY_CTRL, hourglass=ENHANCED\n 1.0, 1.0, 1.0\n")
    f.write("*Shell Section, elset=GORES, material=FABRIC, orientation=FABRIC_ORI, "
            f"controls=BODY_CTRL\n {CFG.T_FABRIC}, 5\n")
    f.write(f"*Transverse Shear Stiffness\n {IC.TSS_K:.6f}, {IC.TSS_K:.6f}\n")


def out_block(field_dt, hist_dt, prs):
    s = (f"*Output, field, time interval={field_dt:.2f}\n*Node Output\n U, COORD, RF"
         + "\n*Element Output, directions=YES\n S, E, SDV"
         + (", SF" if prs else "") + "\n"
         f"*Output, history, time interval={hist_dt:.2f}\n*Energy Output\n"
         " ALLKE, ALLIE, ALLAE, ALLWK, ALLVD, ETOTAL\n")
    if prs:
        s += "*Node Output, nset=RIM\n RF1, RF2, RF3\n"
    return s


def mass_scaling():
    return (f"*Variable Mass Scaling, dt={DT}, type=below min, frequency=1\n"
            f"*Bulk Viscosity\n {BULK_VISC[0]}, {BULK_VISC[1]}\n")


def write_asm(path, fl, alpha, title):
    P = fl["nodes"]
    with open(path, "w") as f:
        f.write(f"*Heading\n {title}: RUN 1 assemble -- drag the whole flat flower onto the "
                f"form-found surface, then release the interior\n**\n")
        write_model(f, fl, alpha)
        f.write(f"*Amplitude, name=RAMP, definition=SMOOTH STEP\n 0.0, 0.0, {IC.DRAG_RAMP:.2f}, 1.0\n")

        def wbc(key, amp, op=""):
            f.write(f"*Boundary{amp}{op}\n")
            for n in sorted(fl[key]):
                u = fl[key][n] - P[n]
                f.write(f" {n+1}, 1,1, {u[0]:.5f}\n {n+1}, 2,2, {u[1]:.5f}\n"
                        f" {n+1}, 3,3, {u[2]:.5f}\n")
        f.write(f"*Step, name=Drag\n*Dynamic, Explicit\n , {IC.DRAG_TIME}, , {DT}\n")
        f.write(mass_scaling())
        wbc("drag_all", ", amplitude=RAMP")
        f.write(out_block(5.0, 0.1, False) + "*End Step\n")
        f.write(f"*Step, name=Settle\n*Dynamic, Explicit\n , {IC.SETTLE_TIME}, , {DT}\n")
        f.write(mass_scaling())
        wbc("drag", "", ", op=NEW")                      # edges only: the interior is released
        f.write("*Restart, write, number interval=1, overlay\n")
        f.write(out_block(IC.SETTLE_TIME / 2, 0.1, False) + "*End Step\n")


def write_prs(path, fl, title, job1):
    P, F = fl["nodes"], fl["tris"]
    side = "SPOS" if IC.mean_nz(P, F) < 0 else "SNEG"        # push +z, as the old chain
    with open(path, "w") as f:
        f.write(f"*Heading\n {title}: RUN 2 seal + pressurise, design pressure {P_DESIGN:.3f} MPa"
                f" + {HOLD_TIME:.0f} s hold (import from {job1})\n**\n")
        f.write("*Import, state=yes, update=no\n ALL_ELEMENTS,\n")
        f.write("*Import Elset\n GORES, ALL_ELEMENTS\n*Import Nset\n RIM\n")
        f.write("**\n*MPC\n")
        f.writelines(f" TIE, {a+1}, {b+1}\n" for a, b in fl["seal"])
        for k in range(IC.N_PRESS):
            f0, f1 = k / IC.N_PRESS, (k + 1) / IC.N_PRESS
            f.write(f"*Amplitude, name=PR{k+1}, definition=SMOOTH STEP\n"
                    f" 0.0, {f0:.4f}, {IC.P_RAMP:.2f}, {f1:.4f}\n")
        f.write(f"*Surface, name=SG, type=ELEMENT\n GORES, {side}\n")
        steps = [(f"Press{k+1}", IC.P_STEP_TIME, f"PR{k+1}") for k in range(IC.N_PRESS)]
        steps.append(("Hold", HOLD_TIME, None))
        for name, t, amp in steps:
            f.write(f"*Step, name={name}\n*Dynamic, Explicit\n , {t}, , {DT}\n")
            f.write(mass_scaling())
            f.write("*Boundary, type=VELOCITY\n RIM, 1, 3, 0.0\n")
            f.write(("*Dsload" + (f", amplitude={amp}" if amp else "")) + f"\n SG, P, {P_DESIGN:.6f}\n")
            f.write("*Restart, write, number interval=1, overlay\n")
            f.write(out_block(t if amp else 30.0, 0.05 if amp else 0.5, True) + "*End Step\n")


def validate(path, run2):
    low = open(path).read().lower()
    assert "*dynamic, explicit" in low and "*static" not in low
    assert "*variable mass scaling" in low and "overlay" in low
    assert low.count("*bulk viscosity") == low.count("*variable mass scaling")
    if run2:
        assert "*import, state=yes, update=no" in low and " tie," in low
        assert "*dsload" in low and "type=velocity" in low and "amplitude=ramp" not in low
        assert "name=hold" in low
    else:
        assert "*user material, constants=24" in low and "*depvar" in low
        assert "orientation=fabric_ori" in low and "*transverse shear stiffness" in low
        assert "*restart, write" in low and "*dsload" not in low
        assert low.count("op=new") == 1, "Settle must release the interior"


def write_bat(jd, job, oldjob=None, vumat=VUMAT_RUN):
    p = os.path.join(jd, f"run_{job}.bat")
    with open(p, "w", newline="\r\n") as f:
        f.write("@echo off\ncd /d \"%~dp0\"\n" + "\n".join(ENV_LINES) + "\n")
        f.write(f'call "{CFG.ABAQUS_BAT}" job={job}' + (f" oldjob={oldjob}" if oldjob else "")
                + f" user={vumat} cpus={CPUS} interactive ask_delete=OFF\n")
    return p


def jobs(variant):
    return f"cst_asm_{variant}", f"cst_prs_{variant}"


def build(h, variants=VARIANTS):
    d = load_rung(h)
    rd = rung_dir(h)
    os.makedirs(rd, exist_ok=True)
    comp = uniform_compensation(d)
    Xl = d["X_leaf"][:, :2].copy()
    pats = {"vares": d["X_converged"][:, :2].copy(), "naive": Xl,
            "uniform": Xl * np.array([1.0 + comp["c_m"], 1.0 + comp["c_w"]])}
    topo = topology(d)
    ref = np.load(REF_FLOWER)
    m_ref = mass_index(np.asarray(ref["nodes"], float), np.asarray(ref["tris"], int))
    setup = dict(vumat_cvisc=VUMAT_CVISC, bulk_viscosity=list(BULK_VISC), rung_h=h, pattern_npz=os.path.join(CK.ROOTS["cst"], MS.tag(h), "pattern.npz"),
                 pressure_mpa=P_DESIGN, hold_s=HOLD_TIME, cpus=CPUS, uniform=comp,
                 alpha_ref=ALPHA_REF, mass_index_ref=m_ref, dt=DT, alpha_fixed=ALPHA_FIX,
                 suffix=SUFFIX, variants={})
    for v in variants:
        fl = build_flower(d, pats[v], topo)
        m = mass_index(fl["nodes"], fl["tris"])
        alpha = ALPHA_FIX if ALPHA_FIX is not None else ALPHA_REF * m_ref / m
        jd = os.path.join(rd, v + SUFFIX)
        if finished(jd, jobs(v)[1]):
            print(f"[build:{v}] {MS.tag(h)} already finished -- left untouched")
            continue
        os.makedirs(jd, exist_ok=True)
        src = open(VUMAT).read()
        key = "PARAMETER (CVISC  = 0.01D0)"
        assert src.count(key) == 1, "VUMAT CVISC line changed"
        with open(os.path.join(jd, VUMAT_RUN), "w") as fo:
            fo.write(src.replace(key, f"PARAMETER (CVISC  = {VUMAT_CVISC:.6g}D0)"))
        shutil.copy2(VUMAT, os.path.join(jd, VUMAT_STOCK))
        j1, j2 = jobs(v)
        title = f"Paraboloid CST {MS.tag(h)} {v}"
        write_asm(os.path.join(jd, j1 + ".inp"), fl, alpha, title)
        write_prs(os.path.join(jd, j2 + ".inp"), fl, title, j1)
        validate(os.path.join(jd, j1 + ".inp"), False)
        validate(os.path.join(jd, j2 + ".inp"), True)
        write_bat(jd, j1)
        write_bat(jd, j2, j1, VUMAT_STOCK if v in PRS_STOCK_VUMAT else VUMAT_RUN)
        np.savez(os.path.join(jd, "flower.npz"), nodes=fl["nodes"], tris=fl["tris"],
                 n_el_per_gore=fl["n_el_per_gore"], n_gores=fl["n_gores"],
                 rim_ids=np.asarray(fl["rim_ids"], int), seal_pairs=fl["seal"],
                 pattern=pats[v], X_spatial=d["X_spatial"], T=d["T"],
                 ring_r=fl["ring_r"], ring_z=fl["ring_z"], apex_z=fl["apex_z"])
        dmax = max(np.linalg.norm(fl["drag"][n] - fl["nodes"][n]) for n in fl["drag"])
        setup["variants"][v] = dict(nodes=len(fl["nodes"]), elements=len(fl["tris"]),
                                    drag_nodes=len(fl["drag"]), seal_pairs=len(fl["seal"]),
                                    rim_nodes=len(fl["rim_ids"]), mass_index=m, alpha=alpha,
                                    max_drag_mm=float(dmax))
        print(f"[build:{v}] {len(fl['nodes'])} nodes, {len(fl['tris'])} S3 ; drag "
              f"{len(fl['drag'])} (max travel {dmax:.0f} mm) ; seal {len(fl['seal'])} ; "
              f"rim {len(fl['rim_ids'])} ; M/M_ref {m/m_ref:.2f} -> alpha {alpha:.3g}")
    sp = os.path.join(rd, f"setup{SUFFIX}.json")
    if os.path.exists(sp):                       # keep the record of variants not rebuilt now
        old = json.load(open(sp))
        setup["variants"] = dict(old.get("variants", {}), **setup["variants"])
    with open(sp, "w") as f:
        json.dump(setup, f, indent=1)
    c = comp
    print(f"[uniform] c_m (warp) {100*c['c_m']:+.4f} %   c_w (weft) {100*c['c_w']:+.4f} %   "
          f"(local range warp {100*c['c_m_range'][0]:+.3f}..{100*c['c_m_range'][1]:+.3f}, "
          f"weft {100*c['c_w_range'][0]:+.3f}..{100*c['c_w_range'][1]:+.3f})")
    print(f"[uniform] at the area-mean stress ({c['mean_stress_mpa'][0]:.1f}, "
          f"{c['mean_stress_mpa'][1]:.1f}) MPa: {100*c['c_at_mean_stress'][0]:+.4f} / "
          f"{100*c['c_at_mean_stress'][1]:+.4f} %")
    print(f"[uniform] VaReS pattern for comparison: length {100*c['vares_length_ratio_minus_1']:+.4f} %, "
          f"rim width {100*c['vares_rim_width_ratio_minus_1']:+.4f} %")
    print(f"[build] {rd}")


# ---------------------------------------------------------------------------------------------
# queue
# ---------------------------------------------------------------------------------------------

def log(rd, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    with open(os.path.join(rd, "queue.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def finished(jd, job):
    p = os.path.join(jd, job + ".sta")
    return os.path.exists(p) and "COMPLETED SUCCESSFULLY" in open(p, errors="ignore").read()


def free_gb(path):
    return shutil.disk_usage(path).free / 2 ** 30


def licence_up():
    try:
        with socket.create_connection((LIC_HOST, LIC_PORT), timeout=10):
            return True
    except OSError:
        return False


def run_job(rd, jd, job):
    if free_gb(jd) < MIN_FREE_GB:
        log(rd, f"{job}: only {free_gb(jd):.1f} GB free (< {MIN_FREE_GB}) -- queue stopped")
        return False
    while not licence_up():
        log(rd, f"{job}: licence server {LIC_HOST}:{LIC_PORT} unreachable (VPN?) -- retry in 5 min")
        time.sleep(300)
    log(rd, f"{job}: start ({free_gb(jd):.1f} GB free)")
    t0 = time.time()
    with open(os.path.join(jd, job + ".runlog"), "a") as out:
        subprocess.run(["cmd.exe", "/c", os.path.join(jd, f"run_{job}.bat")], cwd=jd, stdout=out,
                       stderr=subprocess.STDOUT)
    ok = finished(jd, job)
    log(rd, f"{job}: {'completed' if ok else 'FAILED'} after {(time.time()-t0)/60:.1f} min")
    return ok


def extract(rd, jd, job):
    out = os.path.join(jd, job + "_final.npz")
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(os.path.join(jd, job + ".odb")):
        return out
    with open(os.path.join(jd, "extract.log"), "a") as lo:
        subprocess.run(["cmd.exe", "/c", "call", CFG.ABAQUS_BAT, "python",
                        os.path.join(HERE, "extract_paraboloid_cst_odb.py"),
                        "--odb", job + ".odb", "--out", "."], cwd=jd, stdout=lo,
                       stderr=subprocess.STDOUT)
    if not os.path.exists(out):
        log(rd, f"{job}: extraction FAILED (see extract.log)")
        return None
    return out


def clean(rd, jd, job):
    gone = 0
    for ext in RESTART_EXT:
        p = os.path.join(jd, job + ext)
        if os.path.exists(p):
            gone += os.path.getsize(p)
            os.remove(p)
    log(rd, f"{job}: restart files removed ({gone/2**30:.2f} GB)")


def run_queue(h, variants):
    rd = rung_dir(h)
    if not os.path.exists(os.path.join(rd, f"setup{SUFFIX}.json")):
        raise SystemExit("run --build first")
    log(rd, f"queue start: {variants}")
    for v in variants:
        jd = os.path.join(rd, v + SUFFIX)
        j1, j2 = jobs(v)
        if not finished(jd, j2):
            if not finished(jd, j1) or not os.path.exists(os.path.join(jd, j1 + ".abq")):
                if not run_job(rd, jd, j1):
                    continue
            a1 = extract(rd, jd, j1)
            if a1 is None:
                continue
            c = convergence(a1, jd, j1, prs=False)
            log(rd, f"{j1}: KE/IE {c['ke_ie']:.2e}, viscous/IE {c['vd_ie']:.3f}, added mass "
                    f"{c['mass_change_pct']:.3g} %, drag reactions net {100*c['rf_net_rel']:.3f} %")
            if not c["ke_ie"] < ASM_KE_OK:
                if v in ASM_GATE_SKIP:
                    log(rd, f"{v}: assembly KE/IE {c['ke_ie']:.2e} >= {ASM_KE_OK} -- gate SKIPPED by user decision "
                            f"(2026-10-14, option 2: the naive gore carries ~no strain energy, so KE/IE is undefined; "
                            f"pressure starts from the dragged shape and the settle gate judges the final state)")
                else:
                    log(rd, f"{v}: assembly NOT settled (KE/IE >= {ASM_KE_OK}) -- prs not started")
                    continue
            if not run_job(rd, jd, j2):
                continue
            clean(rd, jd, j1)                    # prs is done; asm restart no longer needed
        npz = extract(rd, jd, j2)
        if npz is None:
            continue
        r = equilibrium(npz, jd)
        c = convergence(npz, jd, j2, prs=True)
        log(rd, f"{v}: sum RF3 {r['rf3']:.4e} N = {100*r['frac']:.2f} % of applied, last-10 s "
                f"drift {100*r['drift']:+.3f} %, KE/IE {c['ke_ie']:.2e}, viscous/IE {c['vd_ie']:.3f}, "
                f"|RFxy| {100*c['rf_xy_rel']:.3f} %, added mass {c['mass_change_pct']:.3g} %")
        # converged only if the rim carries the load AND nothing is still moving: on 2026-10-05
        # the vares run had RF3 100.03 % with zero drift while a ring at r ~ 0.3-0.9 R crept
        # 0.1 mm / 30 s (IE -0.4 % / 10 s) -- RF3 alone cannot see that
        # ... nor the apex: in double precision (2026-10-06) h200 uniform/naive passed RF3 + IE drift with
        # the apex still rising 2.2 / 4.2 mm per 30 s (little strain energy in that motion) -- so the
        # last hold frame's max node motion must be small too before the restart is thrown away
        cr = c["creep"][-1][2] if c.get("creep") else float("inf")
        if r["frac"] >= RF_OK and abs(r["drift"]) < RF_DRIFT_OK and abs(c["ie_change_late"]) < IE_DRIFT_OK \
                and cr < CREEP_OK:
            clean(rd, jd, j2)
        else:
            log(rd, f"{v}: NOT converged (RF3 {100*r['frac']:.2f} %, RF3 drift {100*r['drift']:+.3f} %, "
                    f"IE drift {100*c['ie_change_late']:+.3f} % / 10 s, last-30 s max|du| {cr:.3f} mm) "
                    f"-- prs restart kept")
    log(rd, "queue done")


def run_ladder(variants):
    """Every rung in LADDER_UP (fine -> coarse), one after another; finished jobs are skipped."""
    for h in LADDER_UP:
        if not os.path.exists(os.path.join(rung_dir(h), f"setup{SUFFIX}.json")):
            print(f"[ladder] {MS.tag(h)} not built -- skipped", flush=True)
            continue
        run_queue(h, variants)


def launch(h, variants, ladder=False):
    rd = ROOT if ladder else rung_dir(h)
    flags = 0x08000000 | 0x00000200          # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    py = CFG.PY_BASE
    me = os.path.abspath(__file__)
    extra = ((f" --dt {DT}" if DT != IC.MAX_INC else "")
             + (f" --alpha {ALPHA_FIX}" if ALPHA_FIX is not None else "")
             + (f" --suffix {SUFFIX}" if SUFFIX else ""))
    where = "--ladder" if ladder else f"--rung {h}"
    line = (f'"{py}" "{me}" --run {where} --variants {" ".join(variants)}{extra} '
            f'>> "{os.path.join(rd, "ladder.out" if ladder else "queue.out")}" 2>&1')
    subprocess.Popen(f'cmd.exe /s /c "{line}"', creationflags=flags, cwd=rd)
    pyw = os.path.join(os.path.dirname(py), "pythonw.exe")
    subprocess.Popen([pyw, os.path.join(HERE, "power_guard.py"), "--match",
                      os.path.basename(me), "--exe", "explicit_dp.exe", "--exe", "explicit.exe", "--exe", "pre.exe",
                      "--exe", "package.exe"], creationflags=flags, cwd=HERE)
    print(f"[launch] queue {variants} for {'the ladder ' + str(LADDER_UP) if ladder else MS.tag(h)} "
          f"started; per-rung logs hNNN/queue.log")


# ---------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------

def load_fraction(fl):
    """share of the dish load carried by the modelled part: 1 (full flower) or 1 / n_gores for a one-gore cyclic
    sector (sector_abaqus_ladder.py writes load_fraction into flower.npz, 2026-10-13)."""
    return float(fl["load_fraction"]) if "load_fraction" in fl.files else 1.0


def equilibrium(npz, jd):
    r = np.load(npz, allow_pickle=True)
    fl = np.load(os.path.join(jd, "flower.npz"))
    applied = P_DESIGN * np.pi * float(fl["ring_r"]) ** 2 * load_fraction(fl)
    t, rf = r["hist_t"], -r["hist_rf3"]                      # reaction is downward
    late = t >= t[-1] - 10.0
    return dict(rf3=float(rf[-1]), frac=float(rf[-1] / applied), applied=applied,
                drift=float((rf[-1] - rf[late][0]) / applied),
                ke_end=float(r["hist_ke"][-1]), ke_max=float(r["hist_ke"].max()))


ASM_KE_OK = 1e-3                                       # settle: KE / IE at the end of the asm job
ASM_GATE_SKIP = set()                                  # variants whose asm gate is skipped (runner --skip-asm-gate)


def sta_mass_change(jd, job):
    """Last 'PERCENT CHNG MASS' value of the .sta increment table (added mass, %)."""
    last = None
    for line in open(os.path.join(jd, job + ".sta"), errors="ignore"):
        p = line.split()
        if len(p) == 9 and p[0].isdigit() and ":" in p[3]:
            last = float(p[8])
    return last


def convergence(npz, jd, job, prs):
    """Energy, reaction and stress diagnostics of one extracted job; returns a dict and prints it."""
    r = np.load(npz, allow_pickle=True)
    fl = np.load(os.path.join(jd, "flower.npz"))
    t, ke, ie = r["hist_t"], r["hist_ke"], r["hist_allie"]
    vd, wk, et = r["hist_allvd"], r["hist_allwk"], r["hist_etotal"]
    late = t >= t[-1] - 10.0
    scale = max(abs(ie[-1]), abs(wk[-1]), 1e-30)
    out = dict(ke_ie=float(ke[-1] / max(ie[-1], 1e-30)), ke_max=float(ke.max()),
               ke_end=float(ke[-1]), vd_ie=float(vd[-1] / max(ie[-1], 1e-30)),
               vd_rate_late=float((vd[-1] - vd[late][0]) / max(t[-1] - t[late][0], 1e-30)),
               wk_rate_late=float((wk[-1] - wk[late][0]) / max(t[-1] - t[late][0], 1e-30)),
               ie_change_late=float((ie[-1] - ie[late][0]) / max(ie[-1], 1e-30)),
               etotal_rel=float(et[-1] / scale), mass_change_pct=sta_mass_change(jd, job))
    rf = r["rf"]
    if prs:
        rim = np.asarray(fl["rim_ids"], int)
        applied = P_DESIGN * np.pi * float(fl["ring_r"]) ** 2 * load_fraction(fl)
        F = rf[rim].sum(0)                                   # rim reactions vs -pressure resultant
        out.update(rf_sum=F.tolist(), rf3_frac=float(-F[2] / applied),
                   rf_xy_rel=float(np.hypot(F[0], F[1]) / applied),
                   rf_off_rim=float(np.abs(np.delete(rf, rim, 0)).sum() / applied))
        ne = int(fl["n_el_per_gore"])
        T = fl["T"][np.arange(len(r["S"])) % ne]
        rr = np.hypot(r["coord"][:, 0], r["coord"][:, 1])
        tri = fl["tris"]
        rc = rr[tri].mean(1) / float(fl["ring_r"])
        ratio = r["S"][:, :2] / T[:, :2]
        for nm, m in (("rim r>0.8", rc > 0.8), ("mid", (rc > 0.2) & (rc <= 0.8)), ("apex r<0.2", rc <= 0.2)):
            out[f"sig_ratio_{nm}"] = [float(ratio[m, 0].mean()), float(ratio[m, 1].mean()),
                                      float(np.sqrt(np.mean((ratio[m] - 1) ** 2)))]
    else:
        d = np.abs(rf).sum(1) > 0                            # dragged nodes carry the reactions
        F = rf[d].sum(0)
        out.update(rf_net=F.tolist(), rf_abs=float(np.abs(rf[d]).sum()),
                   rf_net_rel=float(np.linalg.norm(F) / max(np.abs(rf[d]).sum(), 1e-30)),
                   s_warp=[float(np.percentile(r["S"][:, 0], q)) for q in (1, 50, 99)],
                   s_weft=[float(np.percentile(r["S"][:, 1], q)) for q in (1, 50, 99)])
    tag = os.path.basename(npz)
    print(f"[conv] {tag}: KE/IE {out['ke_ie']:.2e} (KE end {out['ke_end']:.3g}, max {out['ke_max']:.3g}) ; "
          f"viscous/IE {out['vd_ie']:.3f} ; last 10 s: IE change {100*out['ie_change_late']:+.3f} %, "
          f"VD rate {out['vd_rate_late']:.3g} vs work rate {out['wk_rate_late']:.3g} ; "
          f"ETOTAL/scale {out['etotal_rel']:+.2e} ; added mass {out['mass_change_pct']:.3g} %")
    if prs:
        print(f"[conv] {tag}: rim RF sum {np.round(out['rf_sum'], 1).tolist()} N ; RF3 = "
              f"{100*out['rf3_frac']:.2f} % of applied ; |RFxy| {100*out['rf_xy_rel']:.3f} % ; "
              f"reactions off the rim {100*out['rf_off_rim']:.3f} %")
        for k in [k for k in out if k.startswith("sig_ratio")]:
            w, f_, rms = out[k]
            print(f"[conv] {tag}:   sigma/sigma* {k[10:]:>10}: warp {w:.4f}  weft {f_:.4f}  rms dev {rms:.4f}")
        if "frame_coord" in r.files:                         # creep between hold frames
            ft, fc = r["frame_t"], r["frame_coord"].astype(float)
            bands = ((0, .1), (.1, .3), (.3, .6), (.6, .9), (.9, 1.01))
            out["creep"] = []
            for a in range(1, len(ft) - 1):
                d = fc[a + 1] - fc[a]
                rb = np.hypot(fc[a + 1][:, 0], fc[a + 1][:, 1]) / float(fl["ring_r"])
                row = [float(ft[a]), float(ft[a + 1]), float(np.linalg.norm(d, axis=1).max())] + \
                      [float(d[(rb >= lo) & (rb < hi), 2].mean()) for lo, hi in bands]
                out["creep"].append(row)
                print(f"[conv] {tag}:   creep {row[0]:5.0f}->{row[1]:5.0f} s: max|du| {row[2]:.4f} mm ; "
                      f"mean dz by r/R band 0-.1/.1-.3/.3-.6/.6-.9/.9-1: "
                      + " ".join(f"{x:+.4f}" for x in row[3:]))
    else:
        print(f"[conv] {tag}: drag reactions net {np.round(out['rf_net'], 1).tolist()} N = "
              f"{100*out['rf_net_rel']:.3f} % of sum|RF| ; S warp 1/50/99 % "
              f"{np.round(out['s_warp'], 2).tolist()} weft {np.round(out['s_weft'], 2).tolist()} MPa")
    return out


def rung_metrics(h, v):
    """Headline numbers of one finished rung/variant (None if not extracted yet)."""
    jd = os.path.join(rung_dir(h), v + SUFFIX)
    npz = os.path.join(jd, jobs(v)[1] + "_final.npz")
    if not os.path.exists(npz):
        return None
    ext = None                                            # hold extended by restart (hold_ext.py)
    for k in range(6, 1, -1):
        if os.path.exists(os.path.join(jd, f"cst_px{k}_{v}_final.npz")):
            ext = npz = os.path.join(jd, f"cst_px{k}_{v}_final.npz")
            break
    r = np.load(npz, allow_pickle=True)
    fl = np.load(os.path.join(jd, "flower.npz"))
    eq = equilibrium(npz, jd)
    R = float(fl["ring_r"])
    Xs = fl["X_spatial"]
    rr = np.hypot(Xs[:, 0], Xs[:, 1]) / R
    cz = np.linalg.lstsq(np.column_stack([rr ** (2 * k) for k in range(5)]), Xs[:, 2], rcond=None)[0]
    x = r["coord"]
    q = np.hypot(x[:, 0], x[:, 1]) / R
    z_t = np.column_stack([q ** (2 * k) for k in range(5)]) @ cz
    dz = np.column_stack([2 * k * q ** (2 * k - 1) if k else 0 * q for k in range(5)]) @ cz / R
    dev = (x[:, 2] - z_t) / np.sqrt(1.0 + dz ** 2)
    ne = int(fl["n_el_per_gore"])
    ratio = r["S"][:, :2] / fl["T"][np.arange(len(r["S"])) % ne][:, :2]
    t, ie = r["hist_t"], r["hist_allie"]
    late = t >= t[-1] - 10.0
    out = dict(h=h, variant=v, elements=int(len(fl["tris"])), rf3_pct=100 * eq["frac"],
               rf3_drift_pct=100 * eq["drift"], ie_drift_pct=float(100 * (ie[-1] - ie[late][0]) / ie[-1]),
               shape_rms=float(np.sqrt(np.mean(dev ** 2))), shape_max=float(np.abs(dev).max()),
               apex_err=float(x[:, 2].max() - float(fl["apex_z"])),
               warp_mean=float(ratio[:, 0].mean()), warp_rms=float(np.sqrt(np.mean((ratio[:, 0] - 1) ** 2))),
               weft_mean=float(ratio[:, 1].mean()), weft_rms=float(np.sqrt(np.mean((ratio[:, 1] - 1) ** 2))),
               not_taut_pct=float(100 * np.mean(r["wrinkle_max"] > 0.5)), creep_last_max=None,
               hold_extended=bool(npz == ext))
    if "frame_coord" in r.files and len(r["frame_t"]) >= 3:
        fc = r["frame_coord"].astype(float)
        out["creep_last_max"] = float(np.linalg.norm(fc[-1] - fc[-2], axis=1).max())
    return out


def ladder_report(variants):
    rows = [m for h in LADDER_UP for v in variants if (m := rung_metrics(h, v))]
    if not rows:
        print("[ladder] no finished rungs yet")
        return
    print(f"{'rung':>6} {'variant':8} {'elems':>7} {'RF3 %':>7} {'IEdrift%':>8} {'creep mm':>8} "
          f"{'shape rms':>9} {'max':>7} {'apex':>7} {'warp/T':>7} {'weft/T':>7} {'w rms':>6} {'f rms':>6} {'slack%':>6}")
    for m in rows:
        cr = f"{m['creep_last_max']:.3f}" if m["creep_last_max"] is not None else "-"
        print(f"{m['h']:6.1f} {m['variant'] + ('+' if m['hold_extended'] else ''):8} {m['elements']:7d} {m['rf3_pct']:7.2f} {m['ie_drift_pct']:8.3f} "
              f"{cr:>8} {m['shape_rms']:9.3f} {m['shape_max']:7.2f} {m['apex_err']:+7.2f} "
              f"{m['warp_mean']:7.4f} {m['weft_mean']:7.4f} {m['warp_rms']:6.3f} {m['weft_rms']:6.3f} "
              f"{m['not_taut_pct']:6.2f}")
    with open(os.path.join(ROOT, f"ladder_summary{SUFFIX}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"[ladder] {len(rows)} rows -> {os.path.join(ROOT, f'ladder_summary{SUFFIX}.json')}")


def check(h):
    rd = rung_dir(h)
    rows = []
    for v in VARIANTS:
        jd = os.path.join(rd, v + SUFFIX)
        npz = os.path.join(jd, jobs(v)[1] + "_final.npz")
        if not os.path.exists(npz):
            print(f"[check:{v}] no result yet")
            continue
        r = np.load(npz, allow_pickle=True)
        fl = np.load(os.path.join(jd, "flower.npz"))
        eq = equilibrium(npz, jd)
        if "hist_allie" in r.files:
            convergence(npz, jd, jobs(v)[1], prs=True)
        # shape: first-order normal offset from the form-found meridian z(r) (even poly fit of
        # the target nodes, which lie on it), anchored on the held rim
        Xs = fl["X_spatial"]
        rr = np.hypot(Xs[:, 0], Xs[:, 1]) / float(fl["ring_r"])
        V = np.column_stack([rr ** (2 * k) for k in range(5)])
        cz = np.linalg.lstsq(V, Xs[:, 2], rcond=None)[0]
        x = r["coord"]
        q = np.hypot(x[:, 0], x[:, 1]) / float(fl["ring_r"])
        z_t = np.column_stack([q ** (2 * k) for k in range(5)]) @ cz
        dz = np.column_stack([2 * k * q ** (2 * k - 1) if k else 0 * q for k in range(5)]) @ cz \
            / float(fl["ring_r"])
        dev = (x[:, 2] - z_t) / np.sqrt(1.0 + dz ** 2)
        # stress: element membrane stress vs the VaReS target of the same gore element
        ne = int(fl["n_el_per_gore"])
        T = fl["T"][np.arange(len(r["S"])) % ne]
        S = r["S"][:, :2]
        ratio = S / T[:, :2]
        wr = r["wrinkle_max"]                                    # SDV6: 0 taut, 1 wrinkled, 2 slack
        rows.append(v)
        print(f"[check:{v}] t = {float(r['time']):.1f} s  sum RF3 {eq['rf3']:.4e} N = "
              f"{100*eq['frac']:.2f} % of {eq['applied']:.4e}  (last 10 s {100*eq['drift']:+.3f} %)"
              f"  KE end/max {eq['ke_end']:.3g}/{eq['ke_max']:.3g}")
        print(f"[check:{v}]   shape normal dev: mean {dev.mean():+.3f}  rms {np.sqrt(np.mean(dev**2)):.3f}"
              f"  max |dev| {np.abs(dev).max():.3f} mm ; apex z {x[:, 2].max():.2f} "
              f"(target {float(fl['apex_z']):.2f})")
        print(f"[check:{v}]   sigma/sigma*: warp mean {ratio[:, 0].mean():.4f} rms dev "
              f"{np.sqrt(np.mean((ratio[:, 0]-1)**2)):.4f} ; weft mean {ratio[:, 1].mean():.4f} "
              f"rms dev {np.sqrt(np.mean((ratio[:, 1]-1)**2)):.4f} ; not taut "
              f"{100*np.mean(wr > 0.5):.2f} % of elements")
    if not rows:
        print("[check] nothing to check")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=float, default=12.5)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    g = ap.add_mutually_exclusive_group(required=True)
    for k in ("build", "launch", "run", "check"):
        g.add_argument("--" + k, action="store_true")
    ap.add_argument("--dt", type=float, default=None, help="explicit target increment (default 3e-3)")
    ap.add_argument("--alpha", type=float, default=None, help="fixed damping alpha (default: mass rule)")
    ap.add_argument("--suffix", default="", help="variant folder suffix for comparison runs")
    ap.add_argument("--ladder", action="store_true",
                    help="act on every rung in LADDER_UP (fine -> coarse) instead of --rung")
    a = ap.parse_args()
    if a.dt is not None:
        DT = a.dt
    if a.alpha is not None:
        ALPHA_FIX = a.alpha
    SUFFIX = a.suffix
    h = min(MS.LADDER + CK.FINE, key=lambda x: abs(x - a.rung))
    if a.ladder:
        if a.build:
            for hh in LADDER_UP:
                build(hh, a.variants)
        elif a.launch:
            launch(h, a.variants, ladder=True)
        elif a.run:
            run_ladder(a.variants)
        else:
            ladder_report(a.variants)
    elif a.build:
        build(h, a.variants)
    elif a.launch:
        launch(h, a.variants)
    elif a.run:
        run_queue(h, a.variants)
    else:
        check(h)
