r"""sector_seamtip_ladder.py -- one-gore cyclic SECTOR decks on the seam- and tip-resolved nested meshes of
build_seamtip_ladder.py, VaReS and uniform patterns.

Everything Abaqus is sector_abaqus_ladder.build, unchanged (asm / prs / cyclic *Equation with dummy nodes / settle
re-splice).  What this script sets:
  mesh      temp/gore_seamtip_s625/<tag>.npz (store of build_seamtip_ladder.py; tip = node 0, exact mirror seam pairs)
  vares     X_vares_flat of the store: the converged 6.25 mm VaReS pattern (meshstudy_vares_cstraw_tippolar210nestgrade/
            h006) mapped by its outline -- cubic spline per seam side and along the rim through the h006 boundary
            nodes, P1 inside; the SAME pattern on every rung (only its discretisation changes)
  uniform   X_leaf * (1 + c_m, 1 + c_w) with ONE (c_m, c_w) for the whole ladder: A.uniform_compensation on the finest
            rung (h006, 1.56 M elements), computed once (--uniform-c) into <store>/uniform_c.json; the old ladder
            re-derived c on every rung (spread ~1e-8 there)
  rim       nodes ON the rim line (A.LINE_TOL 1 mm -> 1e-6 mm): at 0.39-0.78 mm seam elements the 1 mm window would also
            take seam and interior nodes near the rim corners as rim (held)
  alpha     asm material damping = the old nested sector ladder's value at the SAME interior h (per variant, from its
            setup_dp.json on Euler, 2026-10-14), not the mass rule -- the mass-scaled seam band would cut the rule's
            alpha ~14x; precedent sector_seamref.py (seam refinement kept the base rung's alpha)
  asm settle  40 s; h006: lengthened to 160 s afterwards with extend_asm_settle.py (alpha T >= 20, as the old 6.25 mm)
Root: SEAMTIP_ABQ_ROOT or temp/abaqus_sector_seamtip, layout <tag>/<variant>_dp as the old sector root, so
euler_abaqus_runner.py works unchanged.
USAGE  python sector_seamtip_ladder.py --uniform-c                 # once: c from h006 -> <store>/uniform_c.json
       python sector_seamtip_ladder.py --rung 100 [--variants vares uniform] [--check]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sector_abaqus_ladder as SL                 # noqa: E402

A = SL.A
STORE = os.environ.get("SEAMTIP_STORE", os.path.join(A.CFG.TEMP_DIR, "gore_seamtip_s625"))
ROOT = os.environ.get("SEAMTIP_ABQ_ROOT", os.path.join(A.CFG.TEMP_DIR, "abaqus_sector_seamtip"))
RIM_TOL = 1e-6
# old nested sector ladder, asm alpha per interior h and variant (Euler .../abaqus_sector_tippolar210nest/<tag>/setup_dp.json)
ALPHA_OLD = {"h100": {"vares": 34.596168, "uniform": 34.75496}, "h050": {"vares": 8.653773, "uniform": 8.688876},
             "h025": {"vares": 2.163665, "uniform": 2.172227}, "h012": {"vares": 0.540931, "uniform": 0.543057},
             "h006": {"vares": 0.135234, "uniform": 0.135764}}


def rung(h):
    p = os.path.join(STORE, A.MS.tag(h) + ".npz")
    s = np.load(p, allow_pickle=True)
    if str(s["family"]) != "seamtip":
        raise SystemExit(f"{p} is not a seamtip store mesh")
    return dict(X_spatial=s["X_spatial"], X_leaf=s["X_leaf"], seam_ids=np.asarray(s["seam_ids"], int),
                boundary_ids=np.asarray(s["boundary_ids"], int), elements=np.asarray(s["elements"], int), T=s["T"],
                n_gores=int(s["n_gores"]), converged=True, target=str(s["target"]), mesh_npz=p,
                X_converged=s["X_vares_flat"],
                pattern_src=f"converged 6.25 mm VaReS pattern ({s['vares_source']}) by outline map, build_seamtip_ladder.py"), s


def uniform_c():
    d, _ = rung(6.25)
    A.LINE_TOL = RIM_TOL
    comp = A.uniform_compensation(d)
    comp.update(source=d["mesh_npz"], n_elements=int(len(d["elements"])))
    out = os.path.join(STORE, "uniform_c.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists -- not overwriting")
    with open(out, "w") as f:
        json.dump(comp, f, indent=1)
    print(f"[seamtip] uniform c_m {comp['c_m']:.10f}, c_w {comp['c_w']:.10f} (h006, {comp['n_elements']} elements) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uniform-c", action="store_true")
    ap.add_argument("--rung", type=float)
    ap.add_argument("--variants", nargs="+", default=["vares", "uniform"])
    ap.add_argument("--check", action="store_true", help="topology / pairing / alpha report, write nothing")
    a = ap.parse_args()
    if a.uniform_c:
        return uniform_c()
    tg = A.MS.tag(a.rung)
    comp = json.load(open(os.path.join(STORE, "uniform_c.json")))
    d, s = rung(a.rung)
    A.LINE_TOL = RIM_TOL                                         # read at call time by topology_node0
    SL.patterns = lambda dd: (comp, {"vares": dd["X_converged"][:, :2].copy(),
                                     "uniform": dd["X_leaf"][:, :2] * np.array([1.0 + comp["c_m"], 1.0 + comp["c_w"]])})
    SL.ROOT = ROOT
    SL.CORNER_ROT_EQ = os.environ.get("SEAMTIP_CORNER_ROT", "1") == "1"   # 2026-10-15: tie the seam-rim corner rotations
    SL.setup("vares" in a.variants)
    topo = A.topology(d)
    rim_store = np.asarray(d["boundary_ids"])[np.abs(d["X_leaf"][d["boundary_ids"], 0] - d["X_leaf"][:, 0].min()) < RIM_TOL]
    assert np.array_equal(np.sort(topo["rim"]), np.sort(rim_store)), "rim set differs from the store's rim edges"
    print(f"[seamtip] {tg}: {len(d['X_leaf'])} nodes, {len(d['elements'])} S3 ; level {int(s['level'])}, seam size "
          f"{float(s['seam_size']):.3f} mm ; rim {len(topo['rim'])} ; seam pairs {len(topo['pair'])} ; uniform c "
          f"{comp['c_m']:.8f} / {comp['c_w']:.8f}")
    if a.check:
        Rm = A.rot(1, np.pi / int(d["n_gores"]))
        _, pats = SL.patterns(d)
        for v in a.variants:
            fl = SL.build_sector(d, pats[v], topo)
            dev = max(float(np.linalg.norm(fl["drag"][i] - Rm @ fl["drag"][j])) for i, j in fl["seal"])
            print(f"  [{v}] cyclic pairs {len(fl['seal'])}, seam target gap {fl['seam_gap']:.1e} mm, max |p_i - R p_j| "
                  f"{dev:.1e} mm, alpha {ALPHA_OLD[tg][v]}")
        return
    for v in a.variants:
        SL.build(a.rung, [v], d_override=d, alpha_override=ALPHA_OLD[tg][v])
    rd = A.rung_dir(a.rung)
    with open(os.path.join(rd, "seamtip.json"), "w") as f:
        json.dump(dict(store=STORE, mesh=d["mesh_npz"], level=int(s["level"]), seam_size_mm=float(s["seam_size"]),
                       uniform_c=comp, alpha={v: ALPHA_OLD[tg][v] for v in a.variants}, rim_tol_mm=RIM_TOL,
                       vares_pattern=d["pattern_src"]), f, indent=1, default=float)
    print(f"[seamtip] decks -> {rd}")


if __name__ == "__main__":
    main()
