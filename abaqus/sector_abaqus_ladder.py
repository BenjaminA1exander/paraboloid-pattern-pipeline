r"""sector_abaqus_ladder.py -- ONE-GORE cyclic-symmetry version of the nested Abaqus ladder.  Builds decks and checks them in Python;
nothing here starts Abaqus.

Why: Abaqus 2023 on Euler crashes in the Explicit packager on every *Import job of the 25 mm and finer FULL models
(24 gores, 78k-1.2M nodes) unless they run on one domain (loop-parallel 8), which makes the 6.25 mm chains take days.
The dish is 24 congruent gores and the whole full-model pipeline is 24-fold symmetric (flat flower, drag targets, seam
ties, rim, pressure), so ONE gore with cyclic constraints on its two seam edges is the same problem 24x smaller: the
6.25 mm sector (49k nodes) is smaller than the 25 mm full model.

Equivalence with A.build_flower / A.write_asm / A.write_prs (the production decks), gore 0 of the full model:
  asm   identical: in the full model the gores do not interact during the assembly (no ties, no contact), so gore 0's
        asm deck IS the sector asm deck (same flat nodes, elements, orientation, material incl. alpha, drag targets --
        the seam targets use the same averaged expressions, the tip the apex; checked textually by
        compare_sector_full.py --decks).  alpha = ALPHA_REF * M_ref / M_full with M_full = n_gores * M_sector
        (congruent gores), i.e. the full model's alpha, NOT the mass rule applied to the sector.
  prs   the full model TIES (+seam node i of gore g) to (-seam node j = pair[i] of gore g+1), all active dofs.  Under
        cyclic symmetry gore g+1's node j is gore 0's node j rotated by 2 pi / n_gores about the dish axis, so the tie
        becomes the linear constraint  u_i = R u_j  (translations AND rotations, R = rotation by 2 pi / n_gores):
        *Equation with the rotation coefficients, dependent dof on the +seam node (no BC there), 6 equations per
        seam pair.  The 24 tied tip nodes become ONE tip node that may move only along the axis and rotate only about
        it: TIP dofs 1, 2, 4, 5 held (velocity BC, as the rim).  Rim held as before.  Pressure on the gore as before;
        the applied-load check scales by load_fraction = 1 / n_gores (A.equilibrium / A.convergence honour it).
  WHAT THE 100 mm VALIDATION MUST CONFIRM (never assumed): (1) Abaqus/Explicit enforces *Equation at the velocity /
        acceleration level after *Import, state=yes, so the constant flat-layout offset P_i - R P_j between the seam
        mates (the gap of the flat flower) is preserved exactly as the non-coincident MPC TIE preserves it -- if it
        were imposed on the TOTAL displacement the seam would snap by that gap and the comparison with the full model
        would show it at once; (2) *Equation / *Nset are accepted as model data of an import analysis (datacheck);
        (3) sector == gore 0 of the full model: vares to ~0.01 mm; uniform / naive only if their full-model wrinkles
        are cyclically symmetric (compare_sector_full.py --symmetry measures that on the finished full results).
  The sector FORCES every gore to deform identically.  vares (taut, path-independent) satisfies that; uniform and
  naive wrinkle and may not -- a sector result for them is the symmetric solution, which can differ in the tip zone.

Meshes / patterns: exactly nested_abaqus_ladder (tippolar210 store, tip = node 0, every rung the 6.25 mm VaReS pattern
restricted; uniform / naive as before).  Output root: NESTED_ABQ_SECTOR_ROOT or temp/abaqus_sector_cstraw_tippolar210nest,
same layout as the production root (<tag>/<variant>_dp/cst_asm_<v>.inp, cst_prs_<v>.inp, flower.npz, run_*.bat,
setup_dp.json), so euler_abaqus_runner.py / A.run_queue / the settle imports work unchanged (settle_deck splits the
prs deck at the end of the import block, both deck kinds).

USAGE  python sector_abaqus_ladder.py --check H                 # build in memory, print the checks, write nothing
       python sector_abaqus_ladder.py --build H V [V ...]       # write the decks (never overwrites a finished chain)
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abaqus_paraboloid_cst as A                 # noqa: E402
import nested_abaqus_ladder as NL                 # noqa: E402

CK, MS = A.CK, A.MS
ROOT = os.environ.get("NESTED_ABQ_SECTOR_ROOT",
                      os.path.join(A.CFG.TEMP_DIR, "abaqus_sector_cstraw_tippolar210nest"))
SUFFIX, LAPTOP_CPUS, LAPTOP_DOMAINS = "_dp", 2, 2
ROT_EQ = True            # also constrain the rotational dofs 4-6 across the seam (the TIE MPC ties all active dofs)
TIP_HELD_DOFS = ((1, 2), (4, 5))
# CORNER_ROT_EQ (2026-10-15, user: "do it, solution is logical"): the seam pair AT THE RIM (the two seam-rim corner nodes)
# was left out of the cyclic *Equation set because the rim holds its translations -- but nothing held its ROTATIONS, so
# the corner acted as a free double hinge (in the real dish the two corner nodes are one sewn point).  Harmless with
# large corner elements; on the seam+tip L4 mesh (0.39 mm seam elements) every uniform failure (prs x2, stl1 x2) was
# at the corner element 73 / 1035372.  True: write the ROTATION equations (dofs 4-6 only; translations stay rim-held)
# for the corner pairs.  Default False keeps every earlier deck reproducible; sector_seamtip_ladder.py sets it.
CORNER_ROT_EQ = False


def build_sector(d, Xf, topo):
    """One gore -- gore 0 of A.build_flower(d, Xf, topo) -- with the cyclic partner of every +seam node.
    Same keys as build_flower (nodes, tris, drag, drag_all, seal, rim_ids, ...) plus the sector fields."""
    ng = int(d["n_gores"])
    th = np.pi / ng
    Xs = np.asarray(d["X_spatial"], float)
    el = np.asarray(d["elements"], int)
    N = len(Xs)
    rimset = set(topo["rim"].tolist())
    fx, fy = Xf[:, 0], Xf[:, 1]
    tip = topo["tip"]
    a = th                                                   # gore 0 of the flat flower
    ur = np.array([np.cos(a), np.sin(a), 0.0])
    ut = np.array([-np.sin(a), np.cos(a), 0.0])
    rho = fx[tip] - fx
    nodes = rho[:, None] * ur + fy[:, None] * ut
    warp3d, weft3d = -ur[None, :], ut[None, :]
    tgt = lambda g, i: A.rot(g, th) @ Xs[i]                  # noqa: E731  (exactly as build_flower)
    apex = np.array([0.0, 0.0, Xs[tip, 2]])
    drag, pairs, gap, corner = {}, [], 0.0, []
    for i in topo["sp"]:
        j = topo["pair"][i]
        a0, b1 = tgt(0, i), tgt(1, j)                        # gore 0's +seam node i meets gore 1's -seam node j
        gap = max(gap, float(np.linalg.norm(a0 - b1)))
        drag[i] = 0.5 * (a0 + b1)
        drag[j] = 0.5 * (tgt(ng - 1, i) + tgt(0, j))         # gore 0's -seam node j meets gore ng-1's +seam node i
        if i not in rimset:
            pairs.append((i, j))                             # dependent = +seam i, independent = -seam j
        else:
            corner.append((i, j))                            # seam-rim corner pair (rim-held translations)
    for i in topo["rim"]:
        drag.setdefault(int(i), tgt(0, int(i)))
    drag[tip] = apex
    if gap > 1e-6:
        raise SystemExit(f"seam mates' form-found positions differ by {gap:.2e} mm")
    dep = [i for i, _ in pairs]
    assert len(dep) == len(set(dep)) and len(set(j for _, j in pairs)) == len(pairs), "seam pairing not one-to-one"
    assert tip not in dep and tip not in set(j for _, j in pairs) and tip not in rimset
    assert abs(Xs[tip, 0]) < 1e-6 and abs(Xs[tip, 1]) < 1e-6, "tip node is not on the dish axis"
    drag_all = {i: tgt(0, i) for i in range(N)}
    drag_all.update(drag)
    rim_ids = sorted(int(i) for i in topo["rim"])
    # Reference (dummy) node per cyclic pair, 2026-10-13 (first Euler test: Abaqus/Explicit imposes *Equation on the
    # TOTAL displacement, so u_i = R u_j yanked every +seam node by the flat-flower gap at import -- element distortion
    # at increment 0).  The constraint is inhomogeneous: u_i - R u_j = g_ij with g_ij = R P_j - P_i, the constant gap
    # between the flat positions of the mates (P = flat flower).  Dummy node d_k (point mass, no stiffness) is dragged
    # during the assembly to u_d = g_ij and held afterwards; the prs equation u_i - R u_j - u_d = 0 is then satisfied
    # exactly at import (p_i = R p_j at the drag targets) and keeps the seam together like the production TIE.  Its
    # flat position is chosen so that its final position coincides with the +seam node's target (harmless in metrics).
    Rm = A.rot(1, th)
    dummies, dnodes = [], []
    for k, (i, j) in enumerate(pairs):
        g = Rm @ nodes[j] - nodes[i]                         # flat-layout gap vector of the pair
        d = N + k
        dnodes.append(drag[i] - g)                           # flat position: final = P_d + g = drag target of i
        drag[d] = drag_all[d] = drag[i]
        dummies.append((d, g))
    nodes_all = np.vstack([nodes, np.asarray(dnodes).reshape(-1, 3)]) if dummies else nodes
    return dict(nodes=nodes_all, tris=el, n_el_per_gore=len(el), n_gores=1, drag=drag, drag_all=drag_all,
                seal=np.asarray(pairs, int), corner_pairs=np.asarray(corner, int).reshape(-1, 2), rim_ids=rim_ids, warp3d=warp3d, weft3d=weft3d,
                ring_r=float(np.hypot(*Xs[topo["rim"], :2].T).mean()), ring_z=float(Xs[topo["rim"], 2].mean()),
                apex_z=float(apex[2]), sector_gores=ng, cyclic_angle=2.0 * th, tip=int(tip), seam_gap=gap,
                n_gore_nodes=N, dummies=dummies)


DUMMY_MASS = 1e-6        # tonne: ~ a gore node's physical mass; the node is BC-held always, so the value is immaterial


def add_dummy_elements(path, fl):
    """post-process the asm deck written by A.write_asm: point-mass elements on the dummy nodes, elset DUMMY, part of
    ALL_ELEMENTS (so the import carries them); nset DUMMYN."""
    s = open(path).read()
    key = "*Elset, elset=ALL_ELEMENTS\n GORES\n"
    assert s.count(key) == 1, "ALL_ELEMENTS elset line"
    ne = len(fl["tris"])
    blk = "*Element, type=MASS, elset=DUMMY\n" + "".join(f" {ne + k + 1}, {d + 1}\n" for k, (d, _) in enumerate(fl["dummies"]))
    blk += f"*Mass, elset=DUMMY\n {DUMMY_MASS:g}\n*Nset, nset=DUMMYN\n"
    ids = [d + 1 for d, _ in fl["dummies"]]
    for k in range(0, len(ids), 12):
        blk += " " + ", ".join(str(i) for i in ids[k:k + 12]) + "\n"
    with open(path, "w") as f:
        f.write(s.replace(key, blk + "*Elset, elset=ALL_ELEMENTS\n GORES, DUMMY\n"))


def write_prs_sector(path, fl, title, job1):
    """RUN 2 of the sector: import the assembled gore, cyclic *Equation across the seam instead of the TIE MPCs, the
    tip on the axis, rim held, design pressure in N_PRESS ramps + hold.  Step blocks as A.write_prs."""
    P, F = fl["nodes"], fl["tris"]
    IC = A.IC
    side = "SPOS" if IC.mean_nz(P, F) < 0 else "SNEG"
    c, s = np.cos(fl["cyclic_angle"]), np.sin(fl["cyclic_angle"])
    ng, tip = fl["sector_gores"], fl["tip"]
    with open(path, "w") as f:
        f.write(f"*Heading\n {title}: RUN 2 seal + pressurise, design pressure {A.P_DESIGN:.3f} MPa + "
                f"{A.HOLD_TIME:.0f} s hold (import from {job1}) -- ONE-GORE CYCLIC SECTOR of {ng}\n**\n")
        f.write("*Import, state=yes, update=no\n ALL_ELEMENTS,\n")
        f.write("*Import Elset\n GORES, ALL_ELEMENTS, DUMMY\n*Import Nset\n RIM, DUMMYN\n")
        f.write(f"**\n*Nset, nset=TIP\n {tip + 1}\n")
        # translations: u_i - R u_j - u_d = 0 with R = [[c, -s], [s, c]] about the dish axis and u_d the held dummy
        # node of the pair (= the flat-layout gap, see build_sector):  u_i1 - c u_j1 + s u_j2 - u_d1 = 0 ;
        # u_i2 - s u_j1 - c u_j2 - u_d2 = 0 ; u_i3 - u_j3 - u_d3 = 0.  Rotations 4-6: no offset, phi_i = R phi_j.
        # First term = dependent dof (eliminated); it has no BC.
        f.write(f"** cyclic symmetry: +seam node = -seam node rotated by 2 pi / {ng} (dofs 1-3 with the pair's held "
                f"reference node carrying the flat-layout gap{', rotations 4-6' if ROT_EQ else ''})\n*Equation\n")
        dmy = dict((tuple(p), d) for p, (d, _) in zip(fl["seal"].tolist(), fl["dummies"]))
        for i, j in fl["seal"]:
            d = dmy[(int(i), int(j))]
            f.write(f"4\n {i + 1}, 1, 1.0, {j + 1}, 1, {-c:.15g}, {j + 1}, 2, {s:.15g}, {d + 1}, 1, -1.0\n")
            f.write(f"4\n {i + 1}, 2, 1.0, {j + 1}, 1, {-s:.15g}, {j + 1}, 2, {-c:.15g}, {d + 1}, 2, -1.0\n")
            f.write(f"3\n {i + 1}, 3, 1.0, {j + 1}, 3, -1.0, {d + 1}, 3, -1.0\n")
            if ROT_EQ:
                f.write(f"3\n {i + 1}, 4, 1.0, {j + 1}, 4, {-c:.15g}, {j + 1}, 5, {s:.15g}\n")
                f.write(f"3\n {i + 1}, 5, 1.0, {j + 1}, 4, {-s:.15g}, {j + 1}, 5, {-c:.15g}\n")
                f.write(f"2\n {i + 1}, 6, 1.0, {j + 1}, 6, -1.0\n")
        if CORNER_ROT_EQ:
            f.write("** seam-rim corner pair(s): rotations 4-6 tied (translations held by the rim)\n")
            for i, j in fl["corner_pairs"]:
                f.write(f"3\n {i + 1}, 4, 1.0, {j + 1}, 4, {-c:.15g}, {j + 1}, 5, {s:.15g}\n")
                f.write(f"3\n {i + 1}, 5, 1.0, {j + 1}, 4, {-s:.15g}, {j + 1}, 5, {-c:.15g}\n")
                f.write(f"2\n {i + 1}, 6, 1.0, {j + 1}, 6, -1.0\n")
        for k in range(IC.N_PRESS):
            f0, f1 = k / IC.N_PRESS, (k + 1) / IC.N_PRESS
            f.write(f"*Amplitude, name=PR{k+1}, definition=SMOOTH STEP\n 0.0, {f0:.4f}, {IC.P_RAMP:.2f}, {f1:.4f}\n")
        f.write(f"*Surface, name=SG, type=ELEMENT\n GORES, {side}\n")
        steps = [(f"Press{k+1}", IC.P_STEP_TIME, f"PR{k+1}") for k in range(IC.N_PRESS)]
        steps.append(("Hold", A.HOLD_TIME, None))
        for name, t, amp in steps:
            f.write(f"*Step, name={name}\n*Dynamic, Explicit\n , {t}, , {A.DT}\n")
            f.write(A.mass_scaling())
            f.write("*Boundary, type=VELOCITY\n RIM, 1, 3, 0.0\n DUMMYN, 1, 3, 0.0\n")
            for lo, hi in TIP_HELD_DOFS:
                f.write(f" TIP, {lo}, {hi}, 0.0\n")
            f.write(("*Dsload" + (f", amplitude={amp}" if amp else "")) + f"\n SG, P, {A.P_DESIGN:.6f}\n")
            f.write("*Restart, write, number interval=1, overlay\n")
            f.write(A.out_block(t if amp else 30.0, 0.05 if amp else 0.5, True) + "*End Step\n")


def validate_sector(path):
    low = open(path).read().lower()
    assert "*import, state=yes, update=no" in low and "*equation" in low and " tie," not in low
    assert "*nset, nset=tip" in low and " tip, 1, 2, 0.0" in low and " tip, 4, 5, 0.0" in low
    assert " dummyn, 1, 3, 0.0" in low and "gores, all_elements, dummy" in low
    assert "*dsload" in low and "type=velocity" in low and "amplitude=ramp" not in low and "name=hold" in low
    assert "*variable mass scaling" in low and "overlay" in low
    assert low.count("*bulk viscosity") == low.count("*variable mass scaling")


def setup(need_vares):
    NL.setup_A(need_vares)                                   # tip = node 0, rung data from the nested store + h006
    A.ROOT, A.SUFFIX = ROOT, SUFFIX


def patterns(d):
    comp = A.uniform_compensation(d)
    Xl = d["X_leaf"][:, :2].copy()
    return comp, {"vares": d["X_converged"][:, :2].copy(), "naive": Xl,
                  "uniform": Xl * np.array([1.0 + comp["c_m"], 1.0 + comp["c_w"]])}


def check(h):
    setup(False)
    d = A.load_rung(h)
    topo = A.topology(d)
    comp, pats = patterns(d)
    ref = np.load(A.REF_FLOWER)
    m_ref = A.mass_index(np.asarray(ref["nodes"], float), np.asarray(ref["tris"], int))
    ng = int(d["n_gores"])
    Rm = A.rot(1, np.pi / ng)
    print(f"{MS.tag(h)}: {len(d['X_leaf'])} nodes, {len(d['elements'])} S3 per gore ; {ng} gores ; "
          f"vares pattern: {d['pattern_src']}")
    for v, Xf in pats.items():
        fl = build_sector(d, Xf, topo)
        m_s = A.mass_index(fl["nodes"], fl["tris"])
        m_full = ng * m_s
        alpha = A.ALPHA_REF * m_ref / m_full
        # the drag targets of every cyclic pair satisfy p_i = R p_j (what the *Equation keeps after the import)
        dev = max(float(np.linalg.norm(fl["drag"][i] - Rm @ fl["drag"][j])) for i, j in fl["seal"])
        dmax = max(np.linalg.norm(fl["drag"][n] - fl["nodes"][n]) for n in fl["drag"])
        print(f"  [{v:7s}] cyclic pairs {len(fl['seal'])} ; rim {len(fl['rim_ids'])} ; tip node {fl['tip']} ; "
              f"seam target gap {fl['seam_gap']:.2e} mm ; max |p_i - R p_j| over the pairs {dev:.2e} mm ; "
              f"drag max travel {dmax:.0f} mm ; M_full/M_ref {m_full / m_ref:.4f} -> alpha {alpha:.6g} "
              f"(sector alone would give {A.ALPHA_REF * m_ref / m_s:.4g})")
    return comp


def build(h, variants, d_override=None, alpha_override=None, note=""):
    """d_override: rung data to use instead of the nested store's (seam_refine.py, 2026-10-14); alpha_override: the
    material damping to write instead of the mass rule (seam refinement keeps the BASE rung's alpha)."""
    setup("vares" in variants)
    d = d_override if d_override is not None else A.load_rung(h)
    rd = A.rung_dir(h)
    os.makedirs(rd, exist_ok=True)
    comp, pats = patterns(d)
    topo = A.topology(d)
    ref = np.load(A.REF_FLOWER)
    m_ref = A.mass_index(np.asarray(ref["nodes"], float), np.asarray(ref["tris"], int))
    ng = int(d["n_gores"])
    setup_rec = dict(sector=True, sector_gores=ng, cyclic_rotation_eqs=ROT_EQ, tip_held_dofs=TIP_HELD_DOFS,
                     vumat_cvisc=A.VUMAT_CVISC, bulk_viscosity=list(A.BULK_VISC), rung_h=h,
                     pattern_src=str(d["pattern_src"]), mesh_npz=str(d["mesh_npz"]), pressure_mpa=A.P_DESIGN,
                     hold_s=A.HOLD_TIME, cpus=LAPTOP_CPUS, uniform=comp, alpha_ref=A.ALPHA_REF,
                     mass_index_ref=m_ref, dt=A.DT, suffix=SUFFIX, variants={})
    for v in variants:
        fl = build_sector(d, pats[v], topo)
        m_s = A.mass_index(fl["nodes"], fl["tris"])
        m_full = ng * m_s                                    # congruent gores: the FULL flower's mass index
        alpha = A.ALPHA_REF * m_ref / m_full                 # the full model's damping, not the sector's mass rule
        if alpha_override is not None:
            alpha = float(alpha_override)
        jd = os.path.join(rd, v + SUFFIX)
        j1, j2 = A.jobs(v)
        if A.finished(jd, j2):
            print(f"[build:{v}] {MS.tag(h)} already finished -- left untouched")
            continue
        os.makedirs(jd, exist_ok=True)
        src = open(A.VUMAT).read()
        key = "PARAMETER (CVISC  = 0.01D0)"
        assert src.count(key) == 1, "VUMAT CVISC line changed"
        with open(os.path.join(jd, A.VUMAT_RUN), "w") as fo:
            fo.write(src.replace(key, f"PARAMETER (CVISC  = {A.VUMAT_CVISC:.6g}D0)"))
        shutil.copy2(A.VUMAT, os.path.join(jd, A.VUMAT_STOCK))
        title = f"Paraboloid CST {MS.tag(h)} {v} SECTOR"
        A.write_asm(os.path.join(jd, j1 + ".inp"), fl, alpha, title)
        add_dummy_elements(os.path.join(jd, j1 + ".inp"), fl)
        write_prs_sector(os.path.join(jd, j2 + ".inp"), fl, title, j1)
        A.validate(os.path.join(jd, j1 + ".inp"), False)
        validate_sector(os.path.join(jd, j2 + ".inp"))
        for job, old, vm in ((j1, None, A.VUMAT_RUN), (j2, j1, A.VUMAT_STOCK if v in A.PRS_STOCK_VUMAT else A.VUMAT_RUN)):
            p = A.write_bat(jd, job, old, vm)
            s = open(p).read()
            new = s.replace(f" cpus={A.CPUS} ", f" cpus={LAPTOP_CPUS} domains={LAPTOP_DOMAINS} double=both ")
            assert new != s, p
            with open(p, "w", newline="\r\n") as f:
                f.write(new.replace("\r\n", "\n"))
        np.savez(os.path.join(jd, "flower.npz"), nodes=fl["nodes"], tris=fl["tris"], n_el_per_gore=fl["n_el_per_gore"],
                 n_gores=1, rim_ids=np.asarray(fl["rim_ids"], int), seal_pairs=fl["seal"], pattern=pats[v],
                 X_spatial=d["X_spatial"], T=d["T"], ring_r=fl["ring_r"], ring_z=fl["ring_z"], apex_z=fl["apex_z"],
                 load_fraction=1.0 / ng, sector_gores=ng, cyclic_pairs=fl["seal"], cyclic_angle=fl["cyclic_angle"],
                 tip_node=fl["tip"], n_gore_nodes=fl["n_gore_nodes"],
                 dummy_nodes=np.asarray([d for d, _ in fl["dummies"]], int),
                 dummy_gap=np.asarray([g for _, g in fl["dummies"]], float).reshape(-1, 3))
        with open(os.path.join(jd, "pattern_source.txt"), "w") as f:
            f.write((str(d["pattern_src"]) if v == "vares" else
                     "uniform / naive: built from this rung's X_leaf and element-centre targets T") +
                    f"\nONE-GORE CYCLIC SECTOR of {ng} (sector_abaqus_ladder.py)\n")
        dmax = max(np.linalg.norm(fl["drag"][n] - fl["nodes"][n]) for n in fl["drag"])
        setup_rec["variants"][v] = dict(nodes=len(fl["nodes"]), elements=len(fl["tris"]), drag_nodes=len(fl["drag"]),
                                        cyclic_pairs=len(fl["seal"]), rim_nodes=len(fl["rim_ids"]), tip_node=fl["tip"],
                                        mass_index_sector=m_s, mass_index_full=m_full, alpha=alpha,
                                        max_drag_mm=float(dmax), seam_gap_mm=fl["seam_gap"])
        print(f"[build:{v}] SECTOR {len(fl['nodes'])} nodes, {len(fl['tris'])} S3 ; drag {len(fl['drag'])} "
              f"(max travel {dmax:.0f} mm) ; cyclic pairs {len(fl['seal'])} ; rim {len(fl['rim_ids'])} ; "
              f"M_full/M_ref {m_full / m_ref:.2f} -> alpha {alpha:.3g}")
    sp = os.path.join(rd, f"setup{SUFFIX}.json")
    if os.path.exists(sp):
        old = json.load(open(sp))
        setup_rec["variants"] = dict(old.get("variants", {}), **setup_rec["variants"])
    with open(sp, "w") as f:
        json.dump(setup_rec, f, indent=1)
    print(f"[build] {MS.tag(h)} {variants} -> {rd}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs="+")
    ap.add_argument("--check", type=float)
    a = ap.parse_args()
    if a.check:
        check(a.check)
    elif a.build:
        build(float(a.build[0]), a.build[1:])
