r"""validate_vares_sparse.py -- the sparse Newton linear algebra (vares_sparse.py) against the dense one.

PART 1  operator level, on the frozen setups of converged nested rungs (temp/meshstudy_vares_cstraw_tippolarnestgrade):
        at the seed X_initial and at the converged X: R dense vs sparse (expected bit-identical), K (round-off),
        one Levenberg step (K + lam I) dX = -R at lam 1 and 1e-3, the min-eigenvalue diagnostic, and timings.
        Plus sparse-only timings on the nested 12.5 mm mesh (dense would need 7.6 GB) at its prolongation seed.
PART 2  end to end (--compare): sparse re-runs written by the worker under VAL_ROOT (MESHSTUDY_SOLVER=sparse, same
        setup -> same seed, pins, targets) compared with the dense pattern.npz: iterations, ||R|| history, converged
        pattern difference, stress residual.
PART 3  (--parallel) vares_sparse.ParallelAssembler vs the serial loops: R, K and the line-search residual must be
        bit-identical; timings.
USAGE  python validate_vares_sparse.py             # part 1
       python validate_vares_sparse.py --compare   # part 2, after the sparse runs
       python validate_vares_sparse.py --parallel  # part 3
OUT    validate_vares_sparse.json (part 1), validate_vares_sparse_compare.json (part 2)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import MeshStudy_paraboloid_vares_ckpt as CK      # noqa: E402
import MeshStudy_paraboloid_vares as MS           # noqa: E402
import vares_sparse as VS                         # noqa: E402

TARGET, GRADING = "cstraw", "tippolarnest"
ROOT = CK.study_root(TARGET, GRADING)
VAL_ROOT = ROOT + "_sparseval"


def operators(h, X, Xs, elements, setup, free, chen, label):
    import scipy.linalg as sla
    kw = dict(linear_material=False, vary_detJ=chen.VARY_DETJ, sigma_star_all=setup["sigma_star_all"])
    t = time.time()
    Rd, Kd = chen.assemble_residual_tangent(X, Xs, elements, None, setup["detJ_ref_all"], **kw)
    t_ad = time.time() - t
    t = time.time()
    Rs, Ks = VS.assemble_residual_tangent_sparse(X, Xs, elements, None, setup["detJ_ref_all"], **kw)
    t_as = time.time() - t
    Kdf = Kd[np.ix_(free, free)]
    Ksf = VS.restrict(Ks, free)
    out = dict(rung=MS.tag(h), at=label, n_dof_free=int(len(free)), nnz=int(Ksf.nnz),
               R_max_abs_diff=float(np.abs(Rd - Rs).max()), R_bit_identical=bool(np.array_equal(Rd, Rs)),
               K_max_abs_diff=float(np.abs(Kd - Ks.toarray()).max()), K_max_abs=float(np.abs(Kd).max()),
               t_assemble_dense_s=t_ad, t_assemble_sparse_s=t_as)
    for lam in (1.0, 1e-3):
        A = Kdf.copy()
        A[np.diag_indices(A.shape[0])] += lam
        t = time.time()
        xd = sla.solve(A, -Rd[free], check_finite=False)
        t_sd = time.time() - t
        t = time.time()
        xs = VS.solve_shifted(Ksf, lam, -Rs[free])
        t_ss = time.time() - t
        out["step_lam%g" % lam] = dict(rel_diff=float(np.linalg.norm(xd - xs) / np.linalg.norm(xd)),
                                       norm_dX=float(np.linalg.norm(xd)), t_dense_s=t_sd, t_sparse_s=t_ss)
    t = time.time()
    Ksym = 0.5 * (Kdf + Kdf.T)
    ed = float(sla.eigh(Ksym, eigvals_only=True, subset_by_index=[0, 0], check_finite=False)[0])
    t_ed = time.time() - t
    t = time.time()
    es = VS.min_eig_sym(Ksf)
    t_es = time.time() - t
    out.update(min_eig_dense=ed, min_eig_sparse=es, t_eig_dense_s=t_ed, t_eig_sparse_s=t_es)
    print(json.dumps(out), flush=True)
    return out


def part1():
    import optimise_cutting_pattern_npl_gore_chen as chen
    res = []
    for h in (70.7, 25.0):
        d = os.path.join(ROOT, MS.tag(h))
        s = np.load(os.path.join(d, "setup.npz"))
        p = np.load(os.path.join(d, "pattern.npz"))
        elements = [t for t in np.asarray(p["elements"])]
        Xs = p["X_spatial"]
        for X, lab in ((s["X_initial"], "seed"), (p["X_converged"], "converged")):
            res.append(operators(h, X, Xs, elements, s, s["free"], chen, lab))
    # 12.5 mm, sparse only: prolongation seed of the converged h025 pattern, targets as the worker builds them
    from vares_membrane import build_phys_targets_field
    import optimise_cutting_pattern_paraboloid_gore_chen as stage3
    prof = CK.load_profile(TARGET)[0]
    nm, _ = CK.load_nested(TARGET, GRADING, 12.5)
    Xl = nm["X_leaf"]
    X0, desc, _ = CK.nested_seed(TARGET, GRADING, 12.5, nm, Xl.copy())
    elements = [t for t in np.asarray(nm["elements"])]
    pinned = stage3.build_pin_dof_set_geom(X0, [int(i) for i in nm["boundary_ids"]])
    free = np.asarray([q for q in range(2 * X0.shape[0]) if q not in pinned])
    sig, dref = build_phys_targets_field(X0, nm["X_spatial"], elements, prof.sigma_field)
    kw = dict(linear_material=False, vary_detJ=chen.VARY_DETJ, sigma_star_all=sig)
    t = time.time()
    R, K = VS.assemble_residual_tangent_sparse(X0, nm["X_spatial"], elements, None, dref, **kw)
    t_a = time.time() - t
    Kf = VS.restrict(K, free)
    t = time.time()
    x = VS.solve_shifted(Kf, 1.0, -R[free])
    t_s = time.time() - t
    t = time.time()
    e = VS.min_eig_sym(Kf)
    t_e = time.time() - t
    t = time.time()
    chen.assemble_residual(X0, nm["X_spatial"], elements, None, dref, **kw)
    t_r = time.time() - t
    o = dict(rung="h012", at="seed " + desc, n_dof_free=int(len(free)), nnz=int(Kf.nnz), R_norm=float(np.linalg.norm(R[free])),
             t_assemble_sparse_s=t_a, t_residual_only_s=t_r, t_solve_sparse_s=t_s, min_eig_sparse=e, t_eig_sparse_s=t_e,
             norm_dX=float(np.linalg.norm(x)))
    print(json.dumps(o), flush=True)
    res.append(o)
    with open(os.path.join(HERE, "validate_vares_sparse.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def part2():
    import optimise_cutting_pattern_paraboloid_gore_chen as stage3
    prof = CK.load_profile(TARGET)[0]
    res = []
    for tag in sorted(os.listdir(VAL_ROOT)):
        ps = os.path.join(VAL_ROOT, tag, "pattern.npz")
        pd = os.path.join(ROOT, tag, "pattern.npz")
        if not (os.path.exists(ps) and os.path.exists(pd)):
            continue
        s, d = np.load(ps), np.load(pd)
        assert np.array_equal(s["X_initial"], d["X_initial"]), tag + ": different seeds"
        Rs, Rd = s["R_history"], d["R_history"]
        n = min(len(Rs), len(Rd))
        dx = np.linalg.norm(s["X_converged"][:, :2] - d["X_converged"][:, :2], axis=1)
        o = dict(rung=tag, converged_sparse=bool(s["converged"]), converged_dense=bool(d["converged"]),
                 its_sparse=int(len(Rs)), its_dense=int(len(Rd)), R_final_sparse=float(Rs[-1]),
                 R_final_dense=float(Rd[-1]),
                 R_history_max_rel_diff_common=float(np.max(np.abs(Rs[:n] - Rd[:n]) / Rd[:n])),
                 first_it_rel_diff_above_1e6=int(np.argmax(np.abs(Rs[:n] - Rd[:n]) / Rd[:n] > 1e-6))
                 if np.any(np.abs(Rs[:n] - Rd[:n]) / Rd[:n] > 1e-6) else -1,
                 pattern_max_diff_mm=float(dx.max()), pattern_rms_diff_mm=float(np.sqrt((dx ** 2).mean())),
                 stress_res_max_diff_mpa=float(np.abs(s["res_conv"] - d["res_conv"]).max()),
                 wall_sparse_s=float(s["wall_s"]), wall_dense_s=float(d["wall_s"]))
        print(json.dumps(o), flush=True)
        res.append(o)
    with open(os.path.join(HERE, "validate_vares_sparse_compare.json"), "w") as fh:
        json.dump(res, fh, indent=1)




def part_parallel(procs=4):
    """ParallelAssembler vs the serial loops on h071 and h025 (seed + converged): must be bit-identical."""
    import optimise_cutting_pattern_npl_gore_chen as chen
    res = []
    for h in (70.7, 25.0):
        d = os.path.join(ROOT, MS.tag(h))
        s, p = np.load(os.path.join(d, "setup.npz")), np.load(os.path.join(d, "pattern.npz"))
        el = [t for t in np.asarray(p["elements"])]
        kw = dict(linear_material=False, vary_detJ=chen.VARY_DETJ, sigma_star_all=s["sigma_star_all"])
        PA = VS.ParallelAssembler(procs, p["X_spatial"], el, None, s["detJ_ref_all"], **kw)
        for X, lab in ((s["X_initial"], "seed"), (p["X_converged"], "converged")):
            t = time.time(); Rs, Ks = VS.assemble_residual_tangent_sparse(X, p["X_spatial"], el, None, s["detJ_ref_all"], **kw); ts = time.time() - t
            t = time.time(); Rp, Kp = PA.residual_tangent(X); tp = time.time() - t
            t = time.time(); r_s = chen.assemble_residual(X, p["X_spatial"], el, None, s["detJ_ref_all"], **kw); trs = time.time() - t
            t = time.time(); r_p = PA.residual(X); trp = time.time() - t
            o = dict(rung=MS.tag(h), at=lab, procs=procs, R_tangent_identical=bool(np.array_equal(Rs, Rp)),
                     K_identical=bool(np.array_equal(Ks.indptr, Kp.indptr) and np.array_equal(Ks.indices, Kp.indices)
                                      and np.array_equal(Ks.data, Kp.data)),
                     R_residual_identical=bool(np.array_equal(r_s, r_p)),
                     t_tangent_serial_s=ts, t_tangent_parallel_s=tp, t_residual_serial_s=trs, t_residual_parallel_s=trp)
            print(json.dumps(o), flush=True)
            res.append(o)
        PA.close()
    with open(os.path.join(HERE, "validate_vares_sparse_parallel.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    if "--compare" in sys.argv:
        part2()
    elif "--parallel" in sys.argv:
        part_parallel()
    else:
        part1()
