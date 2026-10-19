r"""certify_vares_rung.py -- apply criterion (a) (vares_roundoff.certify) to a FINISHED rung that stopped above the
tolerance.

If the certificate passes, the original pattern.npz is MOVED to pattern_stopped_above_tol.npz (never overwritten) and a
new pattern.npz is written with the same arrays plus converged=True, converged_criterion="roundoff_floor",
certificate=<json>; certificate.json beside it.  If it fails, nothing is changed.

USAGE  python certify_vares_rung.py --rung 12.5 [--target cstraw --grading tippolar210nest --procs 8]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=float, required=True)
    ap.add_argument("--target", default="cstraw")
    ap.add_argument("--grading", default="tippolar210nest")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    import MeshStudy_paraboloid_vares_ckpt as CK
    import MeshStudy_paraboloid_vares as MS
    import vares_sparse as VS
    import vares_roundoff as VR
    import vares_settings as S
    import vares_membrane as VM
    from vares_membrane import apply_step_mesh
    d = os.path.join(CK.study_root(a.target, a.grading), MS.tag(a.rung))
    p = os.path.join(d, "pattern.npz")
    pat = dict(np.load(p, allow_pickle=True))
    if bool(pat["converged"]):
        raise SystemExit(f"{p} is already converged ({pat.get('converged_criterion', 'tol')}) -- nothing to do")
    s = np.load(os.path.join(d, "setup.npz"))
    els = [t for t in np.asarray(pat["elements"])]
    Xs = pat["X_spatial"]
    kw = dict(linear_material=False, vary_detJ=S.VARY_DETJ, sigma_star_all=s["sigma_star_all"])
    PA = VS.ParallelAssembler(a.procs, Xs, els, None, s["detJ_ref_all"], **kw)
    try:
        cert = VR.certify(pat["X_converged"], s["free"], float(pat["newton_tol"]), PA.residual_tangent, PA.residual,
                          apply_step_mesh)
    finally:
        PA.close()
    cert.update(rung=MS.tag(a.rung), grading=a.grading, its=int(len(pat["R_history"])),
                R_history_last=float(pat["R_history"][-1]))
    print(json.dumps(cert, indent=1), flush=True)
    if not cert["ok"]:
        print("[certify] NOT certified -- pattern.npz unchanged", flush=True)
        return
    old = os.path.join(d, "pattern_stopped_above_tol.npz")
    if os.path.exists(old):
        raise SystemExit(f"{old} exists -- not overwriting")
    os.replace(p, old)
    pat.update(converged=True, converged_criterion="roundoff_floor", certificate=json.dumps(cert))
    np.savez(p, **pat)
    with open(os.path.join(d, "certificate.json"), "w") as f:
        json.dump(cert, f, indent=1)
    print(f"[certify] {MS.tag(a.rung)} certified (criterion (a)); original kept as {old}", flush=True)


if __name__ == "__main__":
    main()
