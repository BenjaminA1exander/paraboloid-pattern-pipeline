r"""nested_abaqus_ladder.py -- Abaqus/Explicit (double precision) ladder on the NESTED tip-polar meshes.

EVERY rung uses the 6.25 mm VaReS pattern, mapped; all three patterns as before (vares / uniform / naive).

Meshes: temp/gore_nested_cstraw_tippolar210/<tag>.npz (build_gore_nested_ladder.py --grading tippolar210 --bases 100
--levels 4): 100 -> 50 -> 25 -> 12.5 -> 6.25 mm, each a 1->4 subdivision of the previous one, coarse nodes kept as the
FIRST nodes with bit-identical leaf and spatial coordinates.
Patterns per rung (X = flat cutting pattern of gore 0):
  vares    X_converged of the 6.25 mm VaReS rung (tippolar210nest/h006) RESTRICTED to this rung's nodes: node i of a
           coarser rung IS node i of h006 (checked bit for bit), so the mapping is exact injection -- no interpolation;
  uniform  X_leaf x (1 + c_m, 1 + c_w), c from the Chen law inverted at this rung's element-centre targets (A.build);
  naive    X_leaf.
Decks: the UNCHANGED abaqus_paraboloid_cst builders (A.build: whole-panel drag, Settle releasing the interior, seam
TIE, rim held, 0.05 MPa in 4 ramps + 120 s hold, VUMAT with CVISC 0 + bulk viscosity, stock VUMAT for naive prs,
alpha = 40 M_ref / M), with A.ROOT -> temp/abaqus_vares_cstraw_tippolar210nest, SUFFIX _dp.  One change only:
A.topology takes the tip as NODE 0 (the gmsh tip point): from 25 mm down the graded tip puts more nodes than node 0
inside A.topology's 1 mm tip window.  Everything else (rim, seam mirror pairs, seam-mate gap assertion) unchanged.
Laptop .bat files get cpus=2 domains=6 double=both (as the earlier dp ladders); ETH Euler runs the same decks with
euler_abaqus_runner.py.

USAGE  python nested_abaqus_ladder.py --build H V [V ...]     # V in vares uniform naive; vares needs h006 converged
       python nested_abaqus_ladder.py --check H               # print the mapping / topology checks, build nothing
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import abaqus_paraboloid_cst as A                 # noqa: E402

CK, MS = A.CK, A.MS
TARGET, GRADING = "cstraw", "tippolar210nest"
RUNGS = (100.0, 50.0, 25.0, 12.5, 6.25)
FINEST = float(os.environ.get("NESTED_ABQ_VARES_FROM", "6.25"))    # TEST override only (production: 6.25)
ROOT = os.environ.get("NESTED_ABQ_ROOT", os.path.join(A.CFG.TEMP_DIR, "abaqus_vares_cstraw_tippolar210nest"))
SUFFIX, LAPTOP_CPUS, LAPTOP_DOMAINS = "_dp", 2, 6


def finest_pattern():
    p = os.path.join(CK.study_root(TARGET, GRADING), MS.tag(FINEST), "pattern.npz")
    if not os.path.exists(p):
        return None, p
    d = np.load(p)
    if not bool(d["converged"]):
        raise SystemExit(f"{p} is not converged -- no VaReS pattern to map")
    return d, p


def rung_data(h, need_vares):
    """pattern.npz-like dict for A.build: store mesh + targets + X_converged (h006 restricted, or X_leaf placeholder)."""
    nm, mp = CK.load_nested(TARGET, GRADING, h)
    n = len(nm["X_leaf"])
    d = dict(X_spatial=nm["X_spatial"], X_leaf=nm["X_leaf"], seam_ids=np.asarray(nm["seam_ids"], int),
             boundary_ids=np.asarray(nm["boundary_ids"], int), elements=np.asarray(nm["elements"], int), T=nm["T"],
             n_gores=int(nm["n_gores"]), converged=True, target=TARGET, mesh_npz=mp)
    f6, p6 = finest_pattern()
    if f6 is None:
        if need_vares:
            raise SystemExit(f"{p6} does not exist yet -- the vares variant needs the converged 6.25 mm pattern")
        d["X_converged"], d["pattern_src"] = nm["X_leaf"], "placeholder X_leaf (uniform / naive only)"
        return d
    # exact injection: the first n nodes of h006 are this rung's nodes, bit for bit (nested store)
    if not (np.array_equal(f6["X_leaf"][:n], nm["X_leaf"]) and np.array_equal(f6["X_spatial"][:n], nm["X_spatial"])):
        raise SystemExit(f"{MS.tag(h)}: its nodes are not the first {n} nodes of {MS.tag(FINEST)} -- mapping not exact")
    d["X_converged"] = f6["X_converged"][:n].copy()
    d["pattern_src"] = f"{p6} restricted to nodes 0..{n - 1} (exact)"
    return d


def topology_node0(d):
    """A.topology with the tip = node 0 (asserted to be the unique +x-most leaf node)."""
    Xl = d["X_leaf"][:, :2]
    if not (np.argmax(Xl[:, 0]) == 0 and np.sum(Xl[:, 0] >= Xl[0, 0] - 1e-9) == 1):
        raise SystemExit("node 0 is not the unique tip node")
    seam = np.asarray(d["seam_ids"], int)
    rim = np.where(np.abs(Xl[:, 0] - Xl[:, 0].min()) < A.LINE_TOL)[0]
    sp = [int(i) for i in seam if Xl[i, 1] > 1e-9]
    sm = [int(i) for i in seam if Xl[i, 1] < -1e-9]
    key = {(round(Xl[j, 0], 6), round(-Xl[j, 1], 6)): j for j in sm}
    pair = {i: key[(round(Xl[i, 0], 6), round(Xl[i, 1], 6))] for i in sp}     # exact mirror (KeyError if not)
    if len(sp) != len(sm):
        raise SystemExit(f"seam sides differ: {len(sp)} vs {len(sm)}")
    return dict(rim=rim, tip=0, sp=sp, pair=pair)


def setup_A(need_vares):
    A.ROOT = ROOT
    A.SUFFIX = SUFFIX
    A.topology = topology_node0
    A.load_rung = lambda h: rung_data(h, need_vares)


def check(h):
    setup_A(False)
    d = rung_data(h, False)
    topo = topology_node0(d)
    fl = A.build_flower(d, d["X_leaf"][:, :2], topo)              # asserts the seam-mate gap
    print(f"{MS.tag(h)}: {len(d['X_leaf'])} nodes, {len(d['elements'])} el ; tip node 0 ; rim {len(topo['rim'])} ; "
          f"seam pairs {len(topo['pair'])} ; flower {len(fl['nodes'])} nodes {len(fl['tris'])} S3 ; "
          f"vares pattern: {d['pattern_src']}")
    if not str(d["pattern_src"]).startswith("placeholder"):
        f6, _ = finest_pattern()
        own = os.path.join(CK.study_root(TARGET, GRADING), MS.tag(h), "pattern.npz")
        if os.path.exists(own):
            o = np.load(own)
            dx = np.linalg.norm(A.MS.rigid_fit(d["X_converged"][:, :2], o["X_converged"][:, :2]) -
                                o["X_converged"][:, :2], axis=1)
            print(f"   h006 pattern vs this rung's own VaReS pattern (rigid fit): max {dx.max():.4f} mm, "
                  f"rms {np.sqrt((dx ** 2).mean()):.4f} mm")


def build(h, variants):
    setup_A("vares" in variants)
    A.build(h, variants)
    for v in variants:
        jd = os.path.join(A.rung_dir(h), v + SUFFIX)
        for job in A.jobs(v):
            p = os.path.join(jd, f"run_{job}.bat")
            s = open(p).read()
            if "double=both" in s:
                continue
            new = s.replace(f" cpus={A.CPUS} ", f" cpus={LAPTOP_CPUS} domains={LAPTOP_DOMAINS} double=both ")
            assert new != s, p
            with open(p, "w", newline="\r\n") as f:
                f.write(new.replace("\r\n", "\n"))
        with open(os.path.join(jd, "pattern_source.txt"), "w") as f:
            f.write(rung_data(h, v == "vares")["pattern_src"] if v == "vares" else
                    "uniform / naive: built from this rung's X_leaf and element-centre targets T\n")
    print(f"[build] {MS.tag(h)} {variants} -> {A.rung_dir(h)}")


def chain(h, v):
    """Laptop chain of one rung/variant (the B3 version comparison): A.run_queue + hold_ext.extend, creep 0.1 mm,
    exactly as graded_ladder_abaqus.chain."""
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "temp", "abaqus_vares_cst")))
    import hold_ext as X
    setup_A(v == "vares")
    A.run_queue(h, [v])
    X.SUFFIX, X.CREEP_MAX, X.CPUS, X.DOMAINS = SUFFIX, 0.1, LAPTOP_CPUS, LAPTOP_DOMAINS
    print(f"[chain] {MS.tag(h)} {v}: extend -> {X.extend(h, v)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs="+")
    ap.add_argument("--check", type=float)
    ap.add_argument("--chain", nargs=2)
    a = ap.parse_args()
    if a.check:
        check(a.check)
    elif a.build:
        build(float(a.build[0]), a.build[1:])
    elif a.chain:
        chain(float(a.chain[0]), a.chain[1])
