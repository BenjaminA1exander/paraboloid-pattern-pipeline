"""euler_nest_stage3.py -- STAGE 3 (curvature-sensitive shape objective) of a nested paraboloid rung on Euler.
Same objective, settings and stop rule as the laptop L0 run (temp/run_stage3_local.py) and the L3 Euler run
(nest17089_noqual_curv*): W_QUAL = 0, W_CURV = mean(d^2)/mean(e^2) at the stage-2 state (set in the sbatch),
L-BFGS m = 50, warm start from the rung's stage-2 result; stop when over 1000 steps the surface error changes < 0.2 %,
the objective < 1 % and the surface error <= 1.002 x faceting floor.  Then edge + CST projection.
If the solver stops on its own rule (e.g. the trust cap at an objective of round-off size, as L0 did) the projections
are run anyway and the reason is logged; a requeue resumes from the newest checkpoint.
"""
import os, subprocess, sys, time
import numpy as np

RUN = os.environ["RUN"]
CODE = os.environ["CODE"]
WINDOW, TOL_S, TOL_O, TOL_FLOOR = 1000, 2e-3, 1e-2, 2e-3


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


def log(m):
    with open(f"{RUN}/chain.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}\n")


npz = f"{RUN}/jaxfdm_paraboloid_dish.npz"
if not os.path.exists(npz):
    env = dict(os.environ)
    if os.path.exists(f"{RUN}/ckpt.npz"):                 # requeue: resume this stage's own newest checkpoint
        os.replace(f"{RUN}/ckpt.npz", f"{RUN}/ckpt_resume.npz")
    if os.path.exists(f"{RUN}/ckpt_resume.npz"):
        env["JAXFDM_RESUME_NPZ"] = f"{RUN}/ckpt_resume.npz"
    log(f"stage 3 start ({'resume' if 'JAXFDM_RESUME_NPZ' in env else 'warm from ' + env['JAXFDM_WARM_NPZ']}), "
        f"W_CURV {env['JAXFDM_W_CURV']}, on {os.uname().nodename}")
    with open(f"{RUN}/run.log", "a") as out:
        p = subprocess.Popen([sys.executable, "ParaboloidDish_unstructured_JAXFDM.py"], cwd=CODE, env=env,
                             stdout=out, stderr=subprocess.STDOUT)
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
    if not os.path.exists(npz):
        log("no result npz -- stopping for inspection"); sys.exit(1)
    if not os.path.exists(f"{RUN}/STOP"):
        log("solver ended on its own rule (not the stop rule); projections run anyway -- inspect run.log")
subprocess.run([sys.executable, "project_self_stress.py", npz, f"{RUN}/jaxfdm_paraboloid_dish_projected.npz"],
               cwd=CODE, stdout=open(f"{RUN}/projection_edge.log", "w"), stderr=subprocess.STDOUT)
rc = subprocess.run([sys.executable, "project_self_stress_cst.py", npz, f"{RUN}/jaxfdm_paraboloid_dish_projected_cst.npz"],
                    cwd=CODE, stdout=open(f"{RUN}/projection_cst.log", "w"), stderr=subprocess.STDOUT).returncode
log(f"CST projection exited {rc}; DONE")
