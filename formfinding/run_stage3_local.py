"""run_stage3_local.py -- STAGE 3 (curvature-sensitive shape objective) on a nested paraboloid rung, LAPTOP.
Same objective and stop rule as the L3 stage 3 (Euler noqual/nest17089 part 2): W_QUAL = 0, W_CURV = mean(d^2)/mean(e^2)
at the warm-start state, L-BFGS m = 50; warm start from the rung's stage-2 result; stop when over 1000 steps the surface
error changes < 0.2 %, the objective < 1 %, and the surface error <= 1.002 x faceting floor.  Then edge + CST projection.

Run (base env): python run_stage3_local.py <N> <W_CURV>      e.g.  python run_stage3_local.py 289 2.0710e-4
"""
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import paraboloid_dish_config as CFG
TEMP, HERE, PY_BASE, PY_JAX = CFG.TEMP_DIR, CFG.CODE_FORMFINDING, CFG.PY_BASE, CFG.PY_JAX
N, W_CURV = sys.argv[1], sys.argv[2]
SRC = f"{TEMP}/paraboloid_edge_constq_nest{N}_odt_qual1e-4_warm"
RUN = f"{TEMP}/paraboloid_edge_constq_nest{N}_stage3_curv"
WINDOW, TOL_S, TOL_O, TOL_FLOOR = 1000, 2e-3, 1e-2, 2e-3
ENV = dict(JAXFDM_CKPT_EVERY="25", JAXFDM_EDGE_SEED="const", JAXFDM_IMPROVE_TOL="0", JAXFDM_LBFGS_LS_MAX="24",
           JAXFDM_NEWTON="force", JAXFDM_NO_VIEWER="1", JAXFDM_N_OUTER="40000", JAXFDM_N_STATIONS="25",
           JAXFDM_OBJECTIVE="normal", JAXFDM_PATIENCE="1000000", JAXFDM_Q_MODE="edge", JAXFDM_SNAP_EVERY="50",
           JAXFDM_STAB_EVERY="100", JAXFDM_W_QUAL="0", JAXFDM_W_TAN="0", JAXFDM_W_CURV=W_CURV, JAXFDM_LBFGS_M="50",
           JAXFDM_SCATTER_COMPACT="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
           JAXFDM_MESH_NPZ=f"{RUN}/mesh.npz", JAXFDM_TEMP_DIR=RUN, JAXFDM_WARM_NPZ=f"{SRC}/jaxfdm_paraboloid_dish.npz")


def log(m):
    with open(f"{RUN}/chain.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}\n")


def converged(hist):
    try:
        h = np.genfromtxt(hist, delimiter=",", names=True)
    except Exception:                                     # noqa: BLE001
        return False, "unreadable"
    if h.size < WINDOW + 1:
        return False, "too early"
    se, ob = h["surf_err"], h["obj"]
    rs = abs(se[-1] - se[-1 - WINDOW]) / se[-1]; ro = abs(ob[-1] - ob[-1 - WINDOW]) / ob[-1]
    fl = se[-1] / h["surf_floor"][-1] - 1
    msg = (f"step {int(h['step'][-1])}: surf {1 + fl:.4f} x floor (change {100 * rs:.3f}%), "
           f"obj {ob[-1]:.3e} (change {100 * ro:.2f}%)")
    return (rs < TOL_S and ro < TOL_O and fl < TOL_FLOOR), msg


os.makedirs(RUN, exist_ok=False)                          # never reuse / overwrite a run dir
shutil.copy(f"{SRC}/mesh.npz", f"{RUN}/mesh.npz")
env = dict(os.environ, **ENV)
with open(f"{RUN}/run_env.txt", "w") as f:
    for k in sorted(ENV):
        f.write(f"{k}={ENV[k]}\n")
log(f"stage 3 start: warm from {SRC}, W_CURV {W_CURV}")
with open(f"{RUN}/run.log", "a") as out:
    p = subprocess.Popen([PY_JAX, "ParaboloidDish_unstructured_JAXFDM.py"], cwd=HERE, env=env, stdout=out,
                         stderr=subprocess.STDOUT)
    last = ""
    while p.poll() is None:
        time.sleep(10)
        ok, msg = converged(f"{RUN}/history.csv")
        if msg != last:
            open(f"{RUN}/converged_status.txt", "w").write(msg + "\n"); last = msg
        if ok and not os.path.exists(f"{RUN}/STOP"):
            open(f"{RUN}/STOP", "w").close(); log(f"converged -> STOP: {msg}")
    rc = p.wait()
log(f"solver exited {rc}; {last}")
npz = f"{RUN}/jaxfdm_paraboloid_dish.npz"
if not (os.path.exists(npz) and os.path.exists(f"{RUN}/STOP")):
    log("no converged result -- stopping for inspection"); sys.exit(1)
subprocess.run([PY_BASE, "project_self_stress.py", npz, f"{RUN}/jaxfdm_paraboloid_dish_projected.npz"], cwd=HERE)
rc = subprocess.run([PY_BASE, "project_self_stress_cst.py", npz, f"{RUN}/jaxfdm_paraboloid_dish_projected_cst.npz"],
                    cwd=HERE, stdout=open(f"{RUN}/projection_cst.log", "w"), stderr=subprocess.STDOUT).returncode
log(f"CST projection exited {rc}; DONE")
