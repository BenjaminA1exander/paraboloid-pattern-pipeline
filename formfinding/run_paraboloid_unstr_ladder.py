r"""run_paraboloid_unstr_ladder.py -- coarser rungs of the unstructured paraboloid convergence ladder, run with
EXACTLY the protocol of the 4626-node result (paraboloid_edge_constq_u4626_odt -> _qual1e-4_warm -> projection):

  mesh   : gmsh seed meshstudy_par_unstr_edge_nodal/u<seed>/mesh.npz -> paraboloid_mesh_odt.py --iters 100
           (option B, APEX_R0 default), written to paraboloid_mesh_odt/mesh_u<n>_odt_B_ladder.npz (new file)
  stage 1: every edge a design variable, constant q seed, nodal objective; stopped when the area-weighted
           surface error changes < 0.2% over 1000 accepted steps (the 4626 stage-1 rule, converged.py)
  stage 2: warm start from stage 1, normal objective + element quality W_QUAL=1e-4; stopped when over 1000
           steps the surface error changes < 0.2% AND the objective < 1% (the 4626 warm rule)
  proj.  : project_self_stress.py (minimum complementary energy) -> jaxfdm_paraboloid_dish_projected.npz

Nothing existing is overwritten: refuses to start if a target directory already exists. An EXISTING mesh
and an EXISTING converged stage 1 are reused (inventory first; nothing is rebuilt that already exists).
Run detached (one process per rung):
   python run_paraboloid_unstr_ladder.py <tag> <mesh_npz> [<existing converged stage-1 dir>]
   e.g. u0289 temp/paraboloid_mesh_odt/mesh_u0289_odt_B_ladder.npz
        u1158 temp/paraboloid_mesh_odt/mesh_u1158_odt_B.npz  temp/paraboloid_edge_constq_u1158_odt
"""
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import paraboloid_dish_config as CFG
TEMP, PY_BASE, PY_JAX = CFG.TEMP_DIR, CFG.PY_BASE, CFG.PY_JAX
SOLVER = "ParaboloidDish_unstructured_JAXFDM.py"
WINDOW, TOL_S, TOL_O = 1000, 2e-3, 1e-2

tag, MESH = sys.argv[1], sys.argv[2]
STAGE1_EXISTING = sys.argv[3] if len(sys.argv) > 3 else None
D1 = STAGE1_EXISTING or f"{TEMP}/paraboloid_edge_constq_{tag}_odt"
D2 = f"{TEMP}/paraboloid_edge_constq_{tag}_odt_qual1e-4_warm"
if not os.path.exists(MESH):
    sys.exit(f"[ladder] mesh {MESH} missing")
for p_ in ([D2] if STAGE1_EXISTING else [D1, D2]):
    if os.path.exists(p_):
        sys.exit(f"[ladder] {p_} exists -- refusing to overwrite")
if STAGE1_EXISTING:
    _m = np.load(f"{D1}/mesh.npz"); _r = np.load(MESH)
    assert np.array_equal(_m["vertices"], _r["vertices"]) and np.array_equal(_m["tris"], _r["tris"]),         "existing stage-1 run is not on this mesh"
    assert os.path.exists(f"{D1}/jaxfdm_paraboloid_dish.npz"), "existing stage 1 has no final npz"


def log(msg):
    print(f"[ladder {tag} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


COMMON = dict(JAXFDM_Q_MODE="edge", JAXFDM_EDGE_SEED="const", JAXFDM_N_STATIONS="25", JAXFDM_NEWTON="force",
              JAXFDM_PATIENCE="1000000", JAXFDM_IMPROVE_TOL="0", JAXFDM_LBFGS_LS_MAX="24",
              JAXFDM_STAB_EVERY="100", JAXFDM_SNAP_EVERY="50", JAXFDM_CKPT_EVERY="25",
              JAXFDM_NO_VIEWER="1", PYTHONUNBUFFERED="1")
STAGE1 = dict(JAXFDM_OBJECTIVE="nodal", JAXFDM_N_OUTER="50000")
STAGE2 = dict(JAXFDM_OBJECTIVE="normal", JAXFDM_W_TAN="0", JAXFDM_W_QUAL="1e-4", JAXFDM_N_OUTER="20000")


def converged(hist, warm):
    try:
        h = np.genfromtxt(hist, delimiter=",", names=True)
    except Exception:                                   # noqa: BLE001  (file being written)
        return False, "unreadable"
    if h.size < WINDOW + 1:
        return False, f"step {int(np.atleast_1d(h['step'])[-1]) if h.size else 0}: too early"
    se, ob = h["surf_err"], h["obj"]
    rs = abs(se[-1] - se[-1 - WINDOW]) / se[-1]
    ro = abs(ob[-1] - ob[-1 - WINDOW]) / ob[-1]
    msg = (f"step {int(h['step'][-1])}: surf {se[-1] / h['surf_floor'][-1]:.4f} x floor (change {100 * rs:.3f}%), "
           f"obj {ob[-1]:.3e} (change {100 * ro:.2f}%)")
    return (rs < TOL_S and (ro < TOL_O or not warm)), msg


def stage(d, extra, warm):
    os.makedirs(d)
    shutil.copy(MESH, f"{d}/mesh.npz")
    env = dict(os.environ, **COMMON, **extra, JAXFDM_MESH_NPZ=f"{d}/mesh.npz", JAXFDM_TEMP_DIR=d)
    with open(f"{d}/run_env.txt", "w") as f:            # the exact settings, for the record
        for k in sorted(set(COMMON) | set(extra) | {"JAXFDM_MESH_NPZ", "JAXFDM_TEMP_DIR"}):
            f.write(f"{k}={env[k]}\n")
    log(f"start {os.path.basename(d)}")
    with open(f"{d}/run.log", "w") as out:
        p = subprocess.Popen([PY_JAX, SOLVER], cwd=HERE, env=env, stdout=out, stderr=subprocess.STDOUT)
        last = ""
        while p.poll() is None:
            time.sleep(60)
            ok, msg = converged(f"{d}/history.csv", warm)
            if msg != last:
                with open(f"{d}/converged_status.txt", "w") as f:
                    f.write(msg + "\n")
                last = msg
            if ok and not os.path.exists(f"{d}/STOP"):
                open(f"{d}/STOP", "w").close()
                log(f"converged -> STOP: {msg}")
        rc = p.wait()
    with open(f"{d}/run.log", "a") as out:
        out.write(f"EXIT {rc}\n")
    log(f"{os.path.basename(d)} exited {rc}; {last}")
    if rc != 0 or not os.path.exists(f"{d}/jaxfdm_paraboloid_dish.npz"):
        sys.exit(f"[ladder] stage failed in {d}")


if STAGE1_EXISTING:
    log(f"reusing converged stage 1 {D1}")
else:
    stage(D1, STAGE1, warm=False)
stage(D2, dict(STAGE2, JAXFDM_WARM_NPZ=f"{D1}/jaxfdm_paraboloid_dish.npz"), warm=True)
log("projection")
subprocess.run([PY_BASE, "project_self_stress.py", f"{D2}/jaxfdm_paraboloid_dish.npz",
                f"{D2}/jaxfdm_paraboloid_dish_projected.npz"], cwd=HERE, check=True)
log("DONE")
