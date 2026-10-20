r"""MeshStudy_paraboloid_vares_ckpt.py -- checkpointed, resumable worker for further rungs of the
paraboloid VaReS mesh study (conventions in vares_study.py).

Same setup as the first, dense driver of the study (same imports in the same order, same mesher,
seed, pins, targets) and the same Newton loop as S.run_inner_loop, copied line for line
except that:
  * the full loop state (X, lam, stall, n_indef, R_history, k) is written atomically after
    EVERY iteration (ckpt.npz via tmp + os.replace, previous kept as ckpt_prev.npz), and
    the solve resumes from it, so a stop (kill, sleep, reboot) loses at most one iteration
    and the resumed trajectory is the uninterrupted one;
  * the setup (seed, pins, targets) is frozen in setup.npz on first start and reloaded;
  * memory: K is freed after slicing and K_free + lam*I is built by adding lam to the
    diagonal of a copy (bitwise the same matrix), factorised in place (scipy
    solve, overwrite_a: no hidden copy -- peak ~2 dense matrices, 3.8 GB at 12.5 mm).  The min-eigenvalue diagnostic uses
    scipy eigh(subset_by_index=[0, 0]) -- diagnostic only, it feeds nothing in the solve.
NEWTON_TOL_R is NOT touched; NEWTON_MAX_IT is the study's budget (MESHSTUDY_MAX_IT, 300).

On convergence it writes hNNN/pattern.npz with exactly the keys of run_one, so the analysis
and plot scripts read it like any other rung.  It refuses to start if pattern.npz exists.

TARGETS (--target)
    linearfit  paraboloid_profile.extract_paraboloid() -- the published study; root temp/meshstudy_vares
    cst        paraboloid_profile_cst.extract_paraboloid_cst() -- cyclic-averaged CST + quadratic MLS
               field of the 4626-node net; root temp/meshstudy_vares_cst

USAGE (base anaconda env)
    python MeshStudy_paraboloid_vares_ckpt.py --rung 17.68          # start or resume
    python MeshStudy_paraboloid_vares_ckpt.py --rung 17.68 --target cst
    python MeshStudy_paraboloid_vares_ckpt.py --launch --target cst
              # start (or resume) EVERY unfinished rung of the ladder detached, plus power_guard:
              # 12.5 mm (6 threads), 17.7 mm (3) and 25 mm (2) alone, the other six in one
              # sequential queue (1 thread), each appending to its hNNN/run.log.  Safe to repeat:
              # a rung with a live lock (running) or a pattern.npz (done) is skipped by the worker.
    python MeshStudy_paraboloid_vares_ckpt.py --rung 200 --root DIR --stop-after 10
                                         # (test) stop after 10 iterations, in another root
Outputs in temp/meshstudy_vares/hNNN/:  setup.npz, ckpt.npz, ckpt_prev.npz, run.log (appended),
    pattern.npz (on exit)
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import paraboloid_dish_config as CFG            # noqa: E402
import vares_study as MS         # noqa: E402

# rungs beyond MS.LADDER, continuing the sqrt(2) ladder
FINE = [17.7, 12.5]
ROOTS = {"linearfit": MS.ROOT, "cst": os.path.join(CFG.TEMP_DIR, "meshstudy_vares_cst"),
         # cstraw: the RAW extracted field of gore RAW_GORE, no 24-gore / mirror average --
         # every point of the target is a form-found value, checkable against exact statics
         "cstraw": os.path.join(CFG.TEMP_DIR, "meshstudy_vares_cstraw")}
RAW_GORE = 0
# graded rungs are seeded from the converged UNIFORM rung of this target's study (same h)
SEED_TARGET = {"linearfit": "linearfit", "cst": "cst", "cstraw": "cst"}
# Newton iteration BUDGET for --launch (not a tolerance; NEWTON_TOL_R stays 1e-7).  The cst gore
# runs to the full tip, where the 24 gore tips meet: h200 needs 156 its there (37 trimmed), a slow
# damped crawl on a soft tip mode, so the 300 budget of the published study is too tight.
MAX_IT_LAUNCH = {"linearfit": 300, "cst": 1000, "cstraw": 1000}
# Retries of the 2026-10-04 cst ladder.  The first attempts stay on disk
# as hNNN_<reason>_lsqseed/.  Both apply only to a rung's FIRST start (the seed is frozen in
# setup.npz), so a resume reproduces the same run.
#   SEED_FROM: seed = the converged pattern of the neighbouring rung, P1-sampled at this rung's leaf
#     nodes (same leaf, same target), instead of the lsq_edge flattening.  It changes the start,
#     not the answer: the result must still reach NEWTON_TOL_R.
#       12.5 <- 17.7: lsq seed exited on the stall heuristic at it 4, ||R|| 2.8e4 (still in the
#                     lambda ramp-up every rung goes through)
#       50   <- 35.4: lsq seed got stuck at ||R|| 283 (lambda -> 1e14, indefinite tangent)
#   NO_EARLY_STALL: the stall exit (4 accepted steps with < 1e-4 relative decrease) is NOT taken
#     while ||R|| > NEWTON_TOL_R -- it only ends a run as a failure; the budget still caps it.
SEED_FROM = {"cst": {12.5: 17.7, 50.0: 35.4}}
NO_EARLY_STALL = {"linearfit": False, "cst": True, "cstraw": True}
# Tip-graded ladder.  The default mesher sizes the tip zone theta*(r_ref + d), independent
# of h, so ~2 elements span the tip on EVERY rung and the tip is never refined (the cst ladder converges
# at order ~1 in the last 10 % of the gore, order ~2 mid-gore).  --grading tip meshes with
# size = h * max(G_MIN, min(1, d / R)): h times a fixed grading function, so the tip refines with h
# (2 theta R / h elements across).  Only the uniformly refined rungs (seam edge = h from 70.7 mm down).
# Seed: the converged UNIFORM rung of the same h, P1-sampled at the graded leaf -- it changes the start,
# not the answer (NEWTON_TOL_R unchanged).  Root: <target root>_tipgrade.
GRADINGS = {"none": None, "tip": (600.0, 0.1),
            # tiptest: the grading of the 25 mm Abaqus remesh test
            # (graded_tip_test.py, 2 mm + 0.15 d at h = 25) scaled with h: size = h * min(1, 0.08 + 0.006 d),
            # floor 0.08 h.  Passed as build_gore_gmsh(tip_affine=...); root <target root>_tiptestgrade.
            "tiptest": (0.08, 0.006)}
GRADING_KW = {"none": "tip_grade", "tip": "tip_grade", "tiptest": "tip_affine"}
# tippolar.  The
# tiptest slope (0.006 h per mm) outgrows the gore's opening (2 theta = 0.262 mm per mm), leaving 0.262 / (0.006 h)
# elements across the tip zone -- one strip of stretched triangles at a 50 mm base.  Same affine form, slope matched
# to the opening: k = 2 theta / C -> C / h elements across the tip zone at every distance, floor 0.02 h; k and g0 do
# not depend on h, so every rung is the same mesh design scaled by h.  C = 105 mm, not 150 / 210: dense peak 7.6 GB
# at the nested 12.5 rung on a 15.8 GB laptop (plot_tip_grading_options.py, fig_tip_grading_options.png).
TIPPOLAR_C = 105.0
GRADINGS["tippolar"] = (0.02, 2.0 * np.pi / CFG.N_GORES / TIPPOLAR_C)
GRADING_KW["tippolar"] = "tip_affine"
# NESTED gradings: the rung mesh is not meshed by gmsh here but read from the frozen store
# temp/gore_nested_<target>_<base grading>/<tag>.npz (build_gore_nested_ladder.py): a gmsh base at 70.7 / 50 mm and its
# red-refined subdivisions (gore_subdivision.py), families B 70.7 -> 35.35 -> 17.675 and A 50 -> 25 -> 12.5, so every
# rung contains its parent exactly.  Seed: nested_seed().  Run by vares_nested_chain.py, not --launch.
NESTED = {"tippolarnest": "tippolar"}
GRADINGS["tippolarnest"] = GRADINGS["tippolar"]
GRADING_KW["tippolarnest"] = "tip_affine"
# ONE nested family from a 100 mm base:
# 100 -> 50 -> 25 -> 12.5 -> 6.25, tip-polar grading with C = 210 mm (~2 elements across the tip zone at the 100 mm
# base, where C = 105 leaves one stretched strip); the sparse solver removes the RAM limit that set C = 105.
GRADINGS["tippolar210"] = (0.02, 2.0 * np.pi / CFG.N_GORES / 210.0)
GRADING_KW["tippolar210"] = "tip_affine"
NESTED["tippolar210nest"] = "tippolar210"
GRADINGS["tippolar210nest"] = GRADINGS["tippolar210"]
GRADING_KW["tippolar210nest"] = "tip_affine"
NEST_RUNGS = {"tippolarnest": [70.7, 50.0, 35.4, 25.0, 17.7, 12.5, 8.8, 6.25],
              "tippolar210nest": [100.0, 50.0, 25.0, 12.5, 6.25, 128.0, 64.0]}
# family D: a second
# nested family 128 -> 64 mm (same tip-polar C = 210 grading, same store, build_gore_nested_ladder.py --bases 128
# --levels 1 --family D); h_eff 98.4 / 49.2.  Not part of the 100 mm chain; plotted as an independent family.
# ONE nested family from a 200 mm base (2026-10-13, user: "rerun the whole ladder using a coarser starting mesh ... 200
# ... then the same graded tip as before avoiding bad elements at the nose"; preview plot_nested_family_coarse.py,
# fig_nested_family_coarse_200).  The C = 210 function scaled to h = 200 leaves ~1 element across the nose (slivers,
# q 0.33), so the TIP is the 100 mm chain's base tip and only the far field is coarser:
# size = min(200, 100 (0.02 + k d)) = 200 min(1, 0.01 + k/2 d).  200 -> 100 -> 50 -> 25 -> 12.5 (h_eff 99 .. 6.2).
GRADINGS["tippolar210c200"] = (0.02 * 100.0 / 200.0, GRADINGS["tippolar210"][1] * 100.0 / 200.0)
GRADING_KW["tippolar210c200"] = "tip_affine"
NESTED["tippolar210c200nest"] = "tippolar210c200"
GRADINGS["tippolar210c200nest"] = GRADINGS["tippolar210c200"]
GRADING_KW["tippolar210c200nest"] = "tip_affine"
NEST_RUNGS["tippolar210c200nest"] = [200.0, 100.0, 50.0, 25.0, 12.5]
NEST_BASE_SEED = "tiptest"           # base rungs start from the converged tiptest-graded rung of the same tag + target
# Linear algebra of the Newton loop: "dense" (default, unchanged: dense K, scipy.linalg.solve, dense
# eigh) or "sparse" (vares_sparse.py: same element calls, CSR K, SuperLU, Lanczos min-eig diagnostic) -- memory
# ~O(nnz) instead of 3 (2N)^2 x 8 B, so rungs below 12.5 mm fit.  Same tolerance, damping and line search.
# 4th nested level (2026-10-12, sparse solver): B3 8.84 mm (tag h009, from h018), A3 6.25 mm (h006, from h012)
NEST_EXTRA_RUNGS = [8.8, 6.25]
SOLVER = os.environ.get("MESHSTUDY_SOLVER", "dense")
if SOLVER not in ("dense", "sparse"):
    raise SystemExit(f"MESHSTUDY_SOLVER={SOLVER!r}: dense | sparse")
SPARSE = SOLVER == "sparse"
if SPARSE:
    import vares_sparse as VS                # noqa: E402
# Retry of a nested BASE rung from the converged base rung of the OTHER family (same nested study), P1-sampled -- the
# nested analogue of SEED_FROM.  50 <- 70.7: from the tiptest P1 seed, h050 stalled at ||R|| 7.53
# (lambda -> 6e4, indefinite tangent) with two elements within 2 mm of the tip blown up to flat/spatial area 1.6
# (0.985 elsewhere); first attempt kept as h050_stall_tipblowup_p1tiptest/.  First start only (setup.npz frozen).
# value: h (same nested study) or (grading, h).  100 <- converged nested tippolar h071 (no converged 100 mm rung of
# this target exists; P1 sampling at the leaf, start only).
NEST_SEED_FROM = {("cstraw", "tippolarnest"): {50.0: 70.7},
                  ("cstraw", "tippolar210nest"): {100.0: ("tippolarnest", 70.7), 128.0: 100.0},
                  # 200 mm base <- converged 100 mm chain base (P1 at the leaf, start only)
                  ("cstraw", "tippolar210c200nest"): {200.0: ("tippolar210nest", 100.0)}}


def nested_store(target, grading):
    return os.path.join(CFG.TEMP_DIR, f"gore_nested_{target}_{NESTED[grading]}")


def load_nested(target, grading, h):
    p = os.path.join(nested_store(target, grading), MS.tag(h) + ".npz")
    nm = np.load(p)
    if str(nm["target"]) != target or not np.array_equal(nm["tip_affine"], np.asarray(GRADINGS[grading], float)):
        raise SystemExit(f"{p}: target {nm['target']} / grading {nm['tip_affine']} do not match {target} {grading}")
    return nm, p


def nested_seed(target, grading, h, nm, X_leaf):
    """Seed of a nested rung -- it changes the start, not the answer (NEWTON_TOL_R unchanged).
    level 0: the converged NEST_BASE_SEED-graded rung of the same tag and target, P1-sampled at this leaf;
    level k: the converged PARENT pattern prolongated -- parent nodes (0..n-1, identical leaf) keep their converged
    positions, the node of parent edge (a, b) goes to the pattern midpoint of a, b plus its leaf offset from the leaf
    chord midpoint (non-zero only for seam midpoints, which sit on the seam curve)."""
    X0 = np.zeros_like(X_leaf)
    lvl = int(nm["level"])
    p1 = os.environ.get("MESHSTUDY_NEST_SEED_P1")
    if p1:
        # 2026-10-13 retry of the 200 mm chain's h100: prolongated from the converged h200 (far field 200 mm) it fell into
        # an indefinite branch at the nose (||R|| 168, minEig -414; 4 elements < 6 mm from the tip at flat/spatial area
        # 1.18-1.24 vs 0.983-0.991) -- the h050 tip blow-up of 2026-10-11 again.  MESHSTUDY_NEST_SEED_P1=<grading>:<h>
        # P1-samples a converged pattern of another study at this leaf instead (start only; tolerance unchanged).
        g_src, h_src = p1.split(":")[0], float(p1.split(":")[1])
        p = os.path.join(study_root(target, g_src), MS.tag(h_src), "pattern.npz")
        sd = np.load(p)
        if not bool(sd["converged"]):
            raise SystemExit(f"seed {p} is not converged")
        X0[:, :2] = MS.p1_sample(sd["X_leaf"][:, :2], sd["elements"], sd["X_converged"][:, :2], X_leaf[:, :2])
        return X0, f"p1:{g_src}/{MS.tag(h_src)}", h_src
    if lvl == 0:
        src = NEST_SEED_FROM.get((target, grading), {}).get(h)
        g_src, h_src = (src if isinstance(src, tuple) else (grading, src)) if src is not None else (NEST_BASE_SEED, h)
        p = os.path.join(study_root(target, g_src), MS.tag(h_src), "pattern.npz")
        sd = np.load(p)
        if not bool(sd["converged"]):
            raise SystemExit(f"base seed {p} is not converged")
        X0[:, :2] = MS.p1_sample(sd["X_leaf"][:, :2], sd["elements"], sd["X_converged"][:, :2], X_leaf[:, :2])
        return X0, f"p1:{g_src}/{MS.tag(h_src)}", h_src
    anc = os.environ.get("MESHSTUDY_NEST_SEED_ANCESTOR")
    if anc:
        # seed from a converged ANCESTOR rung, prolongated level
        # by level through the intermediate store meshes (the same exact midpoint rule as below, applied repeatedly).
        h_a = float(anc)
        p = os.path.join(study_root(target, grading), MS.tag(h_a), "pattern.npz")
        sd = np.load(p)
        if not bool(sd["converged"]):
            raise SystemExit(f"ancestor {p} is not converged")
        X = sd["X_converged"][:, :2].copy()
        hc = h_a
        while hc > h * 1.01:
            pm, _ = load_nested(target, grading, hc)
            cm, _ = load_nested(target, grading, hc / 2)
            n = len(pm["X_leaf"])
            if not np.array_equal(cm["X_leaf"][:n], pm["X_leaf"]):
                raise SystemExit(f"{MS.tag(hc / 2)} is not a subdivision of {MS.tag(hc)}")
            T = np.asarray(pm["elements"], int)
            E = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
            Xlp, Xlc = pm["X_leaf"][:, :2], cm["X_leaf"][:, :2]
            X = np.vstack([X, 0.5 * (X[E[:, 0]] + X[E[:, 1]]) + (Xlc[n:] - 0.5 * (Xlp[E[:, 0]] + Xlp[E[:, 1]]))])
            hc /= 2
        if len(X) != len(X_leaf) or not np.allclose(hc, h, rtol=0.02):
            raise SystemExit(f"ancestor prolongation {MS.tag(h_a)} -> {MS.tag(h)} failed")
        X0[:, :2] = X
        return X0, f"prolong{int(round(np.log2(h_a / h)))}:{MS.tag(h_a)}", h_a
    hp = float(nm["base_h"]) / 2 ** (lvl - 1)
    p = os.path.join(study_root(target, grading), MS.tag(hp), "pattern.npz")
    sd = np.load(p)
    if not bool(sd["converged"]):
        # 2026-10-13: h012 stopped at ||R|| 1.2165e-7 (tol 1e-7) on its double-precision residual floor -- the full
        # Newton correction there is 3e-8 mm and every fraction of it RAISES ||R|| (diag_h012_refine.py).  A seed only
        # sets the start, so MESHSTUDY_NEST_SEED_UNCONVERGED_OK=1 lets such a parent seed the child; the child's own
        # convergence is judged against the unchanged NEWTON_TOL_R.
        if os.environ.get("MESHSTUDY_NEST_SEED_UNCONVERGED_OK") != "1":
            raise SystemExit(f"parent {p} is not converged")
        print(f"[seed] parent {MS.tag(hp)} NOT converged (||R|| {float(sd['R_history'][-1]):.3e}) -- used as a seed "
              f"only (MESHSTUDY_NEST_SEED_UNCONVERGED_OK=1)", flush=True)
    n = int(nm["n_parent_nodes"])
    Xlp, Xc = sd["X_leaf"][:, :2], sd["X_converged"][:, :2]
    if len(Xlp) != n or not np.array_equal(sd["X_leaf"], X_leaf[:n]):
        raise SystemExit(f"parent {p} is not the parent mesh of {MS.tag(h)}")
    T = np.asarray(sd["elements"], int)
    E = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    if len(E) != len(X_leaf) - n:
        raise SystemExit(f"{MS.tag(h)}: {len(X_leaf) - n} new nodes for {len(E)} parent edges")
    X0[:n, :2] = Xc
    X0[n:, :2] = 0.5 * (Xc[E[:, 0]] + Xc[E[:, 1]]) + (X_leaf[n:, :2] - 0.5 * (Xlp[E[:, 0]] + Xlp[E[:, 1]]))
    return X0, f"prolong:{MS.tag(hp)}", hp


def mesh_kw(grading):
    """build_gore_gmsh keyword for a grading: {"tip_grade": ...} or {"tip_affine": ...}."""
    return {GRADING_KW[grading]: GRADINGS[grading]}


# graded-ladder seed: "uniform" (converged uniform rung, P1-sampled) or "lsq" (lsq_edge flattening, as
# every uniform rung started).  MESHSTUDY_LAM_INIT overrides S.LAM_INIT (initial Levenberg damping;
# a solver start setting, not a tolerance).  Both apply only to a rung's first start and are saved.
GRADED_SEED = os.environ.get("MESHSTUDY_GRADED_SEED", "uniform")
# MESHSTUDY_GRADED_SEED_FROM=<h> (2026-10-10, fallback for the graded 12.5 rung): seed from the CONVERGED graded
# pattern of rung <h> of the same study (same target + grading), P1-sampled at this leaf -- the graded analogue
# of SEED_FROM (user-approved 2026-10-05 for the uniform 12.5 <- 17.7 retry).  First start only; saved as seed_from.
GRADED_SEED_FROM = os.environ.get("MESHSTUDY_GRADED_SEED_FROM")
LAM_INIT_ENV = os.environ.get("MESHSTUDY_LAM_INIT")
# Graded rungs start at lambda 1, not S.LAM_INIT (1e-3): from the resampled uniform seed the
# undamped first step (146 mm at 70.7 mm) folded the pole element and Newton stalled (indefinite
# tangent, no descent); with lambda_init 1 the 70.7 mm graded rung converged in 16 its to 1.7e-8,
# tangent positive definite throughout, no inverted element (test 2026-10-08).
GRADED_LAM_INIT = 1.0


def lam_init(grading):
    if LAM_INIT_ENV:
        return float(LAM_INIT_ENV)
    return GRADED_LAM_INIT if grading != "none" else None
GRADED_RUNGS = [70.7, 50.0, 35.4, 25.0, 17.7, 12.5]
TARGET = os.environ.get("MESHSTUDY_TARGET", "linearfit")
GRADING = os.environ.get("MESHSTUDY_GRADING", "none")
SUFFIX = ("" if TARGET == "linearfit" else "_" + TARGET) + ("" if GRADING == "none" else "_" + GRADING + "grade")


def study_root(target, grading="none"):
    return ROOTS[target] + ("" if grading == "none" else "_" + grading + "grade")


def load_profile(target):
    if target == "linearfit":
        import paraboloid_profile as pp
        return pp.extract_paraboloid()
    if target == "cst":
        import paraboloid_profile_cst as pc
        return pc.extract_paraboloid_cst()
    if target == "cstraw":
        import paraboloid_profile_cst as pc
        return pc.extract_paraboloid_cst(gore=RAW_GORE)
    raise ValueError(target)


def configure(target=None, grading=None):
    """Point vares_study at a target's study (root + profile for leaf_half_width).
    Plot scripts call this; the target defaults to $MESHSTUDY_TARGET (linearfit), the grading to
    $MESHSTUDY_GRADING (none)."""
    global TARGET, SUFFIX, GRADING
    TARGET = target or TARGET
    GRADING = grading or GRADING
    SUFFIX = ("" if TARGET == "linearfit" else "_" + TARGET) + \
        ("" if GRADING == "none" else "_" + GRADING + "grade")
    MS.ROOT = study_root(TARGET, GRADING)
    if TARGET != "linearfit":
        MS._PROF = load_profile(TARGET)[0]
    return TARGET


def ladder():
    """MS.LADDER + FINE, coarse to fine, keeping only rungs that have a pattern.npz."""
    rungs = (MS.LADDER + FINE if GRADING == "none" else
             NEST_RUNGS[GRADING] if GRADING in NESTED else GRADED_RUNGS)
    return [h for h in sorted(rungs, reverse=True)
            if os.path.exists(MS.rung_npz(h))]


def _lock(d):
    """Refuse to run a rung that another live worker holds; take the lock otherwise."""
    import psutil
    lk = os.path.join(d, "lock")
    if os.path.exists(lk):
        try:
            pid = int(open(lk).read().split()[0])
            p = psutil.Process(pid)
            if p.is_running() and any("MeshStudy_paraboloid_vares_ckpt.py" in c
                                      for c in p.cmdline()):
                raise SystemExit(f"{d} is being solved by pid {pid} -- not starting a second copy")
        except (ValueError, IndexError, psutil.NoSuchProcess, psutil.AccessDenied):
            pass                                           # stale lock
    with open(lk, "w") as f:
        f.write(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def _free_commit_gb():
    """Free commit (RAM + pagefile) -- the limit for the dense solve, not free RAM."""
    import ctypes

    class _MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                   [(n, ctypes.c_ulonglong) for n in ("tp", "ap", "tf", "af", "tv", "av", "ax")]
    m = _MS()
    m.dwLength = ctypes.sizeof(_MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.af / 2 ** 30


def launch(target, grading="none", force_memory=False):
    """Start every unfinished rung detached (Windows), then power_guard if none is running."""
    import subprocess
    import psutil
    if grading in NESTED:
        raise SystemExit("nested gradings run parent -> child via vares_nested_chain.py, not --launch")
    root = study_root(target, grading)
    live = [q for q in psutil.process_iter(["pid", "cmdline"])
            if q.pid != os.getpid()
            and any("MeshStudy_paraboloid_vares_ckpt.py" in c for c in (q.info["cmdline"] or []))
            and "--rung" in (q.info["cmdline"] or [])
            and target == (q.info["cmdline"][q.info["cmdline"].index("--target") + 1]
                           if "--target" in (q.info["cmdline"] or []) else "linearfit")
            and grading == (q.info["cmdline"][q.info["cmdline"].index("--grading") + 1]
                            if "--grading" in (q.info["cmdline"] or []) else "none")]
    if live:
        print(f"[launch] {len(live)} {target} worker(s) already running (pids "
              f"{[q.pid for q in live]}) -- not launching; re-run after they have stopped")
        return
    py = sys.executable
    # CREATE_NO_WINDOW | NEW_PROCESS_GROUP: a hidden console.  NOT DETACHED_PROCESS -- with no
    # console the python children lose the cmd ">> run.log" redirect and log nothing.
    flags = 0x08000000 | 0x00000200
    rungs = sorted(MS.LADDER + FINE if grading == "none" else GRADED_RUNGS, reverse=True)
    todo = [h for h in rungs if not os.path.exists(os.path.join(root, MS.tag(h), "pattern.npz"))]
    # memory gate: the solve is dense, peak ~3 matrices of (2N)^2 doubles (measured 5.5 GB for
    # 2.0 GB/matrix at uniform 12.5 mm).  All "alone" rungs run at once, plus the largest queued one.
    if todo and not force_memory:
        prof = load_profile(target)[0]
        need = {}
        for h in todo:
            n = len(prof.build_gore_gmsh(h, CFG.N_GORES, gore_index=0, **mesh_kw(grading))[1])
            need[h] = 3 * (2 * n) ** 2 * 8 / 2 ** 30
        conc = sum(need[h] for h in todo if h in (12.5, 17.7, 25.0)) + \
            max([need[h] for h in todo if h not in (12.5, 17.7, 25.0)] or [0.0])
        free = _free_commit_gb()
        print("[launch] peak estimate %.1f GB (%s), free commit %.1f GB" % (
            conc, ", ".join("%s %.1f" % (MS.tag(h), need[h]) for h in todo), free))
        if conc + 2.0 > free:
            print("[launch] NOT launching: needs peak + 2 GB margin of free commit.  Close other jobs, "
                  "or pass --force-memory")
            return

    def cmd(h, th):
        d = os.path.join(root, MS.tag(h))
        os.makedirs(d, exist_ok=True)
        return (f'set OMP_NUM_THREADS={th}&& set MKL_NUM_THREADS={th}&& '
                f'set OPENBLAS_NUM_THREADS={th}&& set MPLBACKEND=Agg&& '
                f'set MESHSTUDY_MAX_IT={MAX_IT_LAUNCH[target]}&& '
                f'"{py}" -u "{os.path.abspath(__file__)}" --rung {h!r} --target {target} '
                f'--grading {grading} '
                f'>> "{os.path.join(d, "run.log")}" 2>&1')
    # threads: 6 + 3 + 2 + 1 = 12 logical CPUs; the coarse queue is assembly-bound (1 thread)
    alone = {12.5: 6, 17.7: 3, 25.0: 2}
    jobs = [[(h, alone[h])] for h in todo if h in alone]
    queue = [(h, 1) for h in todo if h not in alone]
    if queue:
        jobs.append(queue)
    for job in jobs:
        line = " & ".join(cmd(h, th) for h, th in job)     # '&': a failed rung does not stop the queue
        # one string with /s /c "...": a list would be re-quoted with \" escapes cmd cannot read
        p = subprocess.Popen(f'cmd.exe /s /c "{line}"', cwd=HERE, creationflags=flags,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        print(f"[launch] {', '.join(MS.tag(h) for h, _ in job)}  (cmd pid {p.pid})")
    if not todo:
        print("[launch] every rung has a pattern.npz -- nothing to do")
        return
    guards = [p for p in psutil.process_iter(["cmdline"])
              if any("power_guard.py" in c for c in (p.info["cmdline"] or []))]
    if guards:
        print(f"[launch] power_guard already running (pid {guards[0].pid})")
    else:
        time.sleep(20)                                     # let the workers appear first
        pyw = os.path.join(os.path.dirname(py), "pythonw.exe")
        g = subprocess.Popen([pyw if os.path.exists(pyw) else py,
                              os.path.join(HERE, "power_guard.py")], cwd=HERE,
                             creationflags=flags, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[launch] power_guard started (pid {g.pid})")


def _atomic_savez(path, **kw):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **kw)
    with open(tmp, "r+b") as f:                     # Windows fsync needs a writable handle
        os.fsync(f.fileno())
    if os.path.exists(path):
        os.replace(path, path.replace(".npz", "_prev.npz"))
    os.replace(tmp, path)


def _load_ckpt(d):
    for name in ("ckpt.npz", "ckpt_prev.npz"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            try:
                c = dict(np.load(p))
                return c, name
            except Exception as e:                       # truncated file -> try the older
                print(f"[ckpt] {name} unreadable ({e}); trying the previous one", flush=True)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=float)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--target", choices=sorted(ROOTS), default="linearfit")
    ap.add_argument("--grading", choices=sorted(GRADINGS), default="none",
                    help="tip: h-scaled tip grading, root <target root>_tipgrade (see GRADINGS)")
    ap.add_argument("--force-memory", action="store_true", help="--launch: skip the memory gate")
    ap.add_argument("--root", default=None, help="default: the target's study root")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="test only: exit (as if killed) after this many iterations")
    a = ap.parse_args()
    if a.launch:
        return launch(a.target, a.grading, a.force_memory)
    if a.rung is None:
        ap.error("--rung or --launch is required")
    h = a.rung
    rungs_ok = NEST_RUNGS[a.grading] if a.grading in NESTED else GRADED_RUNGS
    if a.grading != "none" and h not in rungs_ok:
        raise SystemExit(f"--grading {a.grading} runs only the rungs {rungs_ok}")
    d = os.path.join(a.root or study_root(a.target, a.grading), MS.tag(h))
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "pattern.npz")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists -- not overwriting")
    _lock(d)

    # same imports, same order as the first driver
    import paraboloid_profile as pp                 # noqa: F401 (import order as run_one)
    import vares_settings as S
    import vares_membrane as VM
    import vares_setup as stage3
    from vares_membrane import build_phys_targets_field
    from vares_flatten import least_squares_edge_flatten
    from vares_membrane import apply_step_mesh
    import scipy.linalg as sla

    S.NEWTON_MAX_IT = MS.MAX_IT
    prof, D, meta = load_profile(a.target)
    print(f"[rung {h}] target {a.target}", flush=True)
    print(f"[rung {h}] grading {a.grading} {GRADINGS[a.grading]}", flush=True)
    nest = {}
    if a.grading in NESTED:
        nm, nest_path = load_nested(a.target, a.grading, h)
        X_spatial, X_leaf = nm["X_spatial"], nm["X_leaf"]
        seam_ids, boundary_ids = [int(i) for i in nm["seam_ids"]], [int(i) for i in nm["boundary_ids"]]
        elements = [t for t in np.asarray(nm["elements"])]
        nest = dict(nest_mesh=nest_path, nest_family=str(nm["family"]), nest_level=int(nm["level"]),
                    nest_h_nominal=float(nm["h_nominal"]))
        print(f"[rung {h}] nested mesh {nest_path} (family {nm['family']}, level {int(nm['level'])}, "
              f"h {float(nm['h_nominal']):g})", flush=True)
    else:
        X_spatial, X_leaf, seam_ids, boundary_ids, elements = prof.build_gore_gmsh(
            h, CFG.N_GORES, gore_index=0, **mesh_kw(a.grading))
    els = np.asarray(elements)
    seed_desc = None

    sp = os.path.join(d, "setup.npz")
    if os.path.exists(sp):
        s = np.load(sp)
        assert np.array_equal(s["X_spatial"], X_spatial) and np.array_equal(s["elements"], els), \
            "mesher output differs from the frozen setup"
        X_initial, free = s["X_initial"], s["free"]
        sigma_star_all, detJ_ref_all = s["sigma_star_all"], s["detJ_ref_all"]
        if "seed_desc" in s.files:
            seed_desc, seed_h = str(s["seed_desc"]), float(s["seed_h"])
        print(f"[rung {h}] setup reloaded from setup.npz", flush=True)
    else:
        src = SEED_FROM.get(a.target, {}).get(h) if a.grading == "none" else None
        X_initial = np.zeros_like(X_leaf)
        if a.grading in NESTED:
            X_initial, seed_desc, seed_h = nested_seed(a.target, a.grading, h, nm, X_leaf)
            print(f"[rung {h}] seed: {seed_desc}", flush=True)
        elif a.grading != "none" and GRADED_SEED_FROM:
            src = float(GRADED_SEED_FROM)
            sd_ = np.load(os.path.join(study_root(a.target, a.grading), MS.tag(src), "pattern.npz"))
            if not bool(sd_["converged"]):
                raise SystemExit(f"graded seed rung {MS.tag(src)} is not converged")
            X_initial[:, :2] = MS.p1_sample(sd_["X_leaf"][:, :2], sd_["elements"],
                                            sd_["X_converged"][:, :2], X_leaf[:, :2])
            print(f"[rung {h}] seed: converged GRADED pattern of {MS.tag(src)} sampled at this leaf", flush=True)
        elif a.grading != "none" and GRADED_SEED == "uniform":
            sd_ = np.load(os.path.join(ROOTS[SEED_TARGET[a.target]], MS.tag(h), "pattern.npz"))
            if not bool(sd_["converged"]):
                raise SystemExit(f"uniform rung {MS.tag(h)} is not converged -- no seed")
            X_initial[:, :2] = MS.p1_sample(sd_["X_leaf"][:, :2], sd_["elements"],
                                            sd_["X_converged"][:, :2], X_leaf[:, :2])
            print(f"[rung {h}] seed: converged UNIFORM {SEED_TARGET[a.target]} pattern of {MS.tag(h)} "
                  f"sampled at the graded leaf", flush=True)
        elif src is not None:
            sd_ = np.load(os.path.join(ROOTS[a.target], MS.tag(src), "pattern.npz"))
            if not bool(sd_["converged"]):
                raise SystemExit(f"seed rung {MS.tag(src)} is not converged")
            X_initial[:, :2] = MS.p1_sample(sd_["X_leaf"][:, :2], sd_["elements"],
                                            sd_["X_converged"][:, :2], X_leaf[:, :2])
            print(f"[rung {h}] seed: converged pattern of {MS.tag(src)} sampled at this leaf",
                  flush=True)
        else:
            X_initial[:, :2] = least_squares_edge_flatten(
                X_spatial, elements, init_2d=X_leaf[:, :2], anchor_to=X_leaf[:, :2])
        pinned = stage3.build_pin_dof_set_geom(X_initial, boundary_ids)
        free = np.asarray([q for q in range(2 * X_initial.shape[0]) if q not in pinned])
        sigma_star_all, detJ_ref_all = build_phys_targets_field(
            X_initial, X_spatial, elements, prof.sigma_field)
        _atomic_savez(sp, X_initial=X_initial, free=free, sigma_star_all=sigma_star_all,
                      detJ_ref_all=detJ_ref_all, X_spatial=X_spatial, elements=els,
                      **({} if seed_desc is None else dict(seed_desc=seed_desc, seed_h=seed_h)))

    threads = os.environ.get("OMP_NUM_THREADS", "?")
    c, which = _load_ckpt(d)
    if c is not None:
        X = c["X"]
        lam = float(c["lam"])
        stall = int(c["stall"])
        n_indef = int(c["n_indef"])
        R_history = [float(v) for v in c["R_history"]]
        k0 = int(c["k"]) + 1
        wall_prev = float(c["wall_s"])
        if str(c["threads"]) != threads:
            print(f"[ckpt] WARNING: checkpoint written with {c['threads']} BLAS threads, "
                  f"now {threads}; resumed iterates may differ in the last bits", flush=True)
        print(f"[rung {h}] RESUMED from {which} at iteration {k0}, "
              f"||R|| {R_history[-1]:.4e}, lam {lam:.3e}", flush=True)
    else:
        X = X_initial.copy()
        lam = lam_init(a.grading) or S.LAM_INIT
        print(f"[rung {h}] lambda_init {lam:g}", flush=True)
        stall = 0
        n_indef = 0
        R_history = []
        k0 = 0
        wall_prev = 0.0
        print(f"[rung {h}] {len(els)} tris, {X_spatial.shape[0]} nodes, {len(free)} free dofs;"
              f" tol {S.NEWTON_TOL_R:g}, max_it {S.NEWTON_MAX_IT}, threads {threads}, solver {SOLVER}"
              + (f" (refine {VS.REFINE})" if SPARSE else ""),
              flush=True)

    t0 = time.time()
    N_g = 2 * X.shape[0]
    free_ix = np.ix_(free, free)
    kw = dict(linear_material=False, vary_detJ=S.VARY_DETJ, sigma_star_all=sigma_star_all)
    # sparse mode only: element loops on MESHSTUDY_ASSEMBLY_PROCS processes, accumulated in element order
    # (bit-identical R and K, vares_sparse.ParallelAssembler); 1 = the serial loops
    procs = int(os.environ.get("MESHSTUDY_ASSEMBLY_PROCS", "1")) if SPARSE else 1
    PA = VS.ParallelAssembler(procs, X_spatial, elements, None, detJ_ref_all, **kw) if procs > 1 else None
    if PA is not None:
        print(f"[rung {h}] parallel assembly on {procs} processes", flush=True)
    print(f"  {'iter':>4s}  {'||R||':>12s}  {'|dX|':>10s}  "
          f"{'alpha':>6s}  {'lambda':>10s}  {'minEig(Ks)':>12s}  flag", flush=True)
    exit_reason = "max_it"
    n_this = 0
    for k in range(k0, S.NEWTON_MAX_IT):
        if SPARSE:
            R, K = (PA.residual_tangent(X) if PA is not None else
                    VS.assemble_residual_tangent_sparse(X, X_spatial, elements, None, detJ_ref_all, **kw))
            R_free = R[free]
            K_free = VS.restrict(K, free)
        else:
            R, K = VM.assemble_residual_tangent(X, X_spatial, elements, None, detJ_ref_all, **kw)
            R_free = R[free]
            K_free = K[free_ix]
        del K
        rn = float(np.linalg.norm(R_free))
        if len(R_history) > k:               # iteration k was evaluated before a stop
            R_history = R_history[:k]
        R_history.append(rn)

        if SPARSE:
            min_eig = VS.min_eig_sym(K_free)
        else:
            Ks = 0.5 * (K_free + K_free.T)
            min_eig = float(sla.eigh(Ks, eigvals_only=True, subset_by_index=[0, 0],
                                     overwrite_a=True, check_finite=False)[0])
            del Ks
        if min_eig < 0.0:
            n_indef += 1

        if rn < S.NEWTON_TOL_R:
            print(f"  {k:>4d}  {rn:12.4e}  {'-':>10s}  {'-':>6s}  "
                  f"{'-':>10s}  {min_eig:12.3e}  converged", flush=True)
            exit_reason = "converged"
            break

        accepted = False
        alpha = 1.0
        diag = None if SPARSE else np.diag_indices(K_free.shape[0])
        while True:
            try:
                if SPARSE:
                    dX_free = VS.solve_shifted(K_free, lam, -R_free)
                else:
                    A = K_free.copy()
                    A[diag] += lam                    # == K_free + lam * eye, bitwise
                    dX_free = sla.solve(A, -R_free, overwrite_a=True, check_finite=False)
            except np.linalg.LinAlgError:
                A = None
                lam = min(lam * S.LAM_UP, S.LAM_MAX)
                if lam >= S.LAM_MAX:
                    break
                continue
            A = None

            dX = np.zeros(N_g)
            dX[free] = dX_free
            step = float(np.linalg.norm(dX))
            if step > S.NEWTON_MAX_STEP:
                dX *= S.NEWTON_MAX_STEP / step

            alpha = 1.0
            ls_ok = False
            for _ in range(S.LS_MAX_BT):
                X_trial = apply_step_mesh(X, dX, alpha)
                try:
                    R_tr = (PA.residual(X_trial) if PA is not None else
                            VM.assemble_residual(X_trial, X_spatial, elements, None, detJ_ref_all, **kw))
                except (ValueError, np.linalg.LinAlgError):
                    alpha *= 0.5
                    continue
                if np.linalg.norm(R_tr[free]) <= (1.0 - S.LS_C1 * alpha) * rn:
                    ls_ok = True
                    break
                alpha *= 0.5

            if ls_ok:
                X = X_trial
                rn_new = float(np.linalg.norm(R_tr[free]))
                if alpha >= 1.0:
                    lam = max(lam * S.LAM_DOWN, S.LAM_MIN)
                else:
                    lam = min(lam * S.LAM_UP, S.LAM_MAX)
                stall = stall + 1 if (rn - rn_new) <= 1.0e-4 * rn else 0
                accepted = True
                break
            else:
                lam = min(lam * S.LAM_UP, S.LAM_MAX)
                if lam >= S.LAM_MAX:
                    break
        del K_free

        eff_step = alpha * float(np.linalg.norm(dX)) if accepted else 0.0
        flag = "ok" if accepted else "STALL"
        print(f"  {k:>4d}  {rn:12.4e}  {eff_step:10.3e}  {alpha:6.3f}  "
              f"{lam:10.3e}  {min_eig:12.3e}  {flag}   [{time.time() - t0 + wall_prev:.0f} s]",
              flush=True)
        if not accepted:
            print(f"  no descent (lambda hit {S.LAM_MAX:.0e}) -- exiting", flush=True)
            exit_reason = "no_descent"
            break
        # state after iteration k: everything the next iteration reads
        _atomic_savez(os.path.join(d, "ckpt.npz"), X=X, lam=lam, stall=stall, n_indef=n_indef,
                      R_history=np.asarray(R_history), k=k,
                      wall_s=time.time() - t0 + wall_prev, threads=threads)
        n_this += 1
        if eff_step < S.NEWTON_MIN_STEP:
            print(f"  step floor (|dX|<{S.NEWTON_MIN_STEP:.0e} mm) -- exiting", flush=True)
            exit_reason = "step_floor"
            break
        if stall >= 4 and NO_EARLY_STALL[a.target]:
            if stall == 4:
                print("  (stall heuristic would exit here -- not taken, ||R|| above tol)",
                      flush=True)
        elif stall >= 4:
            print("  converged to a residual floor (||R|| stalled) -- exiting", flush=True)
            exit_reason = "stall"
            break
        if a.stop_after and n_this >= a.stop_after:
            print(f"[test] stopping after {n_this} iterations (simulated kill)", flush=True)
            sys.exit(3)
    else:
        print(f"  reached NEWTON_MAX_IT = {S.NEWTON_MAX_IT} "
              f"without ||R|| < {S.NEWTON_TOL_R}", flush=True)

    # criterion (a) (user 2026-10-13): a SPARSE run that stopped on the step floor above NEWTON_TOL_R is accepted only if
    # vares_roundoff.certify proves it sits on the double-precision residual floor (tolerance itself unchanged)
    cert = None
    if SPARSE and exit_reason == "step_floor" and not (R_history and R_history[-1] < S.NEWTON_TOL_R):
        import json as _json
        import vares_roundoff as VR
        rt = PA.residual_tangent if PA is not None else (
            lambda X_: VS.assemble_residual_tangent_sparse(X_, X_spatial, elements, None, detJ_ref_all, **kw))
        rr = PA.residual if PA is not None else (
            lambda X_: VM.assemble_residual(X_, X_spatial, elements, None, detJ_ref_all, **kw))
        cert = VR.certify(X, free, S.NEWTON_TOL_R, rt, rr, apply_step_mesh)
        print(f"[certify] round-off floor certificate: {_json.dumps(cert)}", flush=True)
    if PA is not None:
        PA.close()
    wall = time.time() - t0 + wall_prev
    R_hist = R_history
    res_init, S_init, T = stage3.phys_residual_field(X_initial, X_spatial, elements,
                                                     prof.sigma_field)
    res_conv, S_conv, _ = stage3.phys_residual_field(X, X_spatial, elements, prof.sigma_field)
    s_cen = np.array([prof.s_of_pos(p) for p in X_spatial[els].mean(axis=1)]) \
        * prof.geom_scale
    converged_tol = bool(R_hist[-1] < S.NEWTON_TOL_R)
    converged = converged_tol or bool(cert and cert["ok"])
    crit = "tol" if converged_tol else ("roundoff_floor" if converged else "none")
    capped = (not converged) and len(R_hist) >= S.NEWTON_MAX_IT
    np.savez(
        out,
        X_initial=X_initial, X_converged=X, X_spatial=X_spatial, X_leaf=X_leaf,
        elements=els, seam_ids=np.asarray(seam_ids), boundary_ids=np.asarray(boundary_ids),
        R_history=np.asarray(R_hist, float), n_indef=n_indef,
        res_init=res_init, res_conv=res_conv, S_init=S_init, S_conv=S_conv, T=T,
        s_cen=s_cen, gmsh_mesh_size=h, converged=converged, capped=capped,
        newton_tol=S.NEWTON_TOL_R, newton_max_it=S.NEWTON_MAX_IT, wall_s=wall,
        geom_scale=prof.geom_scale, s_mid_mm=0.5 * (prof.s_lo + prof.s_hi) * prof.geom_scale,
        s_lo_mm=prof.s_lo * prof.geom_scale, s_hi_mm=prof.s_hi * prof.geom_scale,
        n_gores=CFG.N_GORES, trim_dish=CFG.TRIM_DISH if a.target == "linearfit" else 0.0,
        p_fdm=meta["p_fdm"],
        seed_from=(SEED_FROM.get(a.target, {}).get(h, np.nan) if a.grading == "none" else
                   seed_h if seed_desc is not None else
                   (float(GRADED_SEED_FROM) if GRADED_SEED_FROM else h)),
        grading=a.grading, tip_grade=np.asarray(GRADINGS[a.grading] or (np.nan, np.nan), float),
        graded_seed=(seed_desc if seed_desc is not None else GRADED_SEED if a.grading != "none" else ""),
        **nest,
        lam_init=lam_init(a.grading) or S.LAM_INIT,
        no_early_stall=NO_EARLY_STALL[a.target],
        form_npz=CFG.FORM_NPZ if a.target == "linearfit" else meta.get("state_npz", ""),
        target=a.target, raw_gore=meta.get("raw_gore", -1),
        solver=SOLVER + (f"+refine{VS.REFINE}" if SPARSE else ""), converged_criterion=crit,
        **({} if cert is None else dict(certificate=__import__("json").dumps(cert))))
    print(f"[rung {h}] {len(R_hist)} its, ||R|| {R_hist[-1]:.3e} -> "
          f"{('CONVERGED' if converged_tol else 'CONVERGED (round-off floor, criterion a)') if converged else ('CAPPED' if capped else 'STOPPED ABOVE TOL')}"
          f" ({exit_reason}, {wall:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
