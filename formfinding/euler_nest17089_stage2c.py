"""euler_nest17089_stage2c.py -- (2c: W_QUAL scaled by (h_L3/h_L2)^4, WARM start from the stage-2b result; stop rule as 2b) continue L3 (nest17089) STAGE 2 on Euler from the laptop's ckpt.npz.
Same env and stop rule as temp/paraboloid_nest17089_resume.py (laptop); no memory watchdog (Slurm limit replaces it).
Then edge projection + CST projection, as the laptop driver. Success = result npz exists (rc may be != 0 if a
headless plot fails after the npz is written)."""
import os, subprocess, sys, time
import numpy as np

RUN = os.environ["RUN"]                                   # stage-2 run dir (ckpt.npz, history.csv, mesh.npz)
CODE = os.environ["CODE"]
WINDOW, TOL_S, TOL_O = 1000, 2e-3, 1e-2
TOL_FLOOR = 2e-3                                          # stage 2b: surf_err / surf_floor - 1 must also be < this (L2 reached 1.5e-3)


def converged(hist, warm=True):                           # identical to the laptop runner
    try:
        h = np.genfromtxt(hist, delimiter=",", names=True)
    except Exception:                                     # noqa: BLE001
        return False, "unreadable"
    if h.size < WINDOW + 1:
        return False, "too early"
    se, ob = h["surf_err"], h["obj"]
    rs = abs(se[-1] - se[-1 - WINDOW]) / se[-1]; ro = abs(ob[-1] - ob[-1 - WINDOW]) / ob[-1]
    msg = (f"step {int(h['step'][-1])}: surf {se[-1] / h['surf_floor'][-1]:.4f} x floor (change {100 * rs:.3f}%), "
           f"obj {ob[-1]:.3e} (change {100 * ro:.2f}%)")
    fl = se[-1] / h["surf_floor"][-1] - 1
    return (rs < TOL_S and (ro < TOL_O or not warm) and fl < TOL_FLOOR), msg


def log(m):
    with open(f"{RUN}/chain.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}\n")


npz = f"{RUN}/jaxfdm_paraboloid_dish.npz"
if not os.path.exists(npz):
    if os.path.exists(f"{RUN}/ckpt.npz"):                 # requeue: resume this stage's own newest ckpt
        os.replace(f"{RUN}/ckpt.npz", f"{RUN}/ckpt_resume.npz")
    env = dict(os.environ)
    if os.path.exists(f"{RUN}/ckpt_resume.npz"):
        env["JAXFDM_RESUME_NPZ"] = f"{RUN}/ckpt_resume.npz"
    log(f"solver start ({'resume' if 'JAXFDM_RESUME_NPZ' in env else 'warm start from ' + env['JAXFDM_WARM_NPZ']}) "
        f"on {os.uname().nodename}")
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
    if not (os.path.exists(npz) and os.path.exists(f"{RUN}/STOP")):
        log("no converged result npz -- stopping for inspection"); sys.exit(1)
subprocess.run([sys.executable, "project_self_stress.py", npz, f"{RUN}/jaxfdm_paraboloid_dish_projected.npz"], cwd=CODE)
rc = subprocess.run([sys.executable, "project_self_stress_cst.py", npz, f"{RUN}/jaxfdm_paraboloid_dish_projected_cst.npz"],
                    cwd=CODE, stdout=open(f"{RUN}/projection_cst.log", "w"), stderr=subprocess.STDOUT).returncode
log(f"CST projection exited {rc}; DONE")
