"""hold_ext.py -- continue a finished prs hold by restart.

Why: the naive apex was still inflating at the end of the 120 s hold on the coarser rungs
(17.7 mm: max|du| 15.7 -> 7.6 -> 3.9 mm per 30 s; 25 mm: 13.6 -> 9.3 -> 6.3), an order above the
~0.2 mm / 30 s soft-mode creep every variant shows.  Those shape numbers were not final.

Rule: while the last 30 s of the latest hold moved some node more than CREEP_MAX = 1 mm, continue
by restart for another EXT_TIME s, at most MAX_EXT times: cst_prs_<v> -> cst_px2_<v> -> cst_px3_<v>
-> ... cst_px6_<v>.  Each extension is one step 'HoldN' with the SAME increment, variable mass scaling,
bulk viscosity, rim BC, pressure, VUMAT and domain count as the prs job -- only the hold is longer.
New files only; earlier jobs and extracts are untouched.  Each is extracted and checked with the
same equilibrium / convergence printout as run_queue.

USAGE  python hold_ext.py --rung 17.7 --variant naive      # one rung/variant
       python hold_ext.py --queue                           # all ladder rungs, as their prs finish,
                                                            # until ladder_par.py logs 'all chains done'
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
BASIC = os.path.normpath(os.path.join(HERE, "..", "..", "01_tutorials", "basic_examples"))
sys.path.insert(0, BASIC)
os.chdir(BASIC)
import abaqus_paraboloid_cst as A   # noqa: E402

CPUS, DOMAINS = 2, 6
PRS_HOLD_STEP = 7                   # asm Drag, Settle = 1, 2 ; prs Press1-4 = 3-6, Hold = 7
CREEP_MAX = 1.0                     # mm per 30 s (last hold interval, any node)
EXT_TIME = 120.0
MAX_EXT = 5                         # raised from 3 at 04:43: the naive apex decays slower on coarser rungs
NIGHT_LOG = os.path.join(HERE, "night.log")
SUFFIX = ""                         # --suffix _dp: the double-precision ladder folders (06:40)


def chain(v):
    """[(job, step number of its last step)] -- prs, then the extensions."""
    return [(A.jobs(v)[1], PRS_HOLD_STEP)] + [(f"cst_px{k}_{v}", PRS_HOLD_STEP + k - 1)
                                             for k in range(2, MAX_EXT + 2)]


def ext_time(npz, t_default):
    """Hold time that the observed geometric decay of the last two 30 s intervals needs to bring the
    interval motion under 0.8 CREEP_MAX, clipped to [120, 600] s (05:36: licence waits of ~20 min per
    checkout made 120 s chunks the bottleneck).  Falls back to t_default if it is not decaying."""
    r = np.load(npz, allow_pickle=True)
    fc = r["frame_coord"].astype(float)
    if len(fc) < 3:
        return t_default
    d1 = float(np.linalg.norm(fc[-2] - fc[-3], axis=1).max())
    d2 = float(np.linalg.norm(fc[-1] - fc[-2], axis=1).max())
    q = d2 / max(d1, 1e-12)
    if not 0.0 < q < 0.98:
        return t_default
    n = int(np.ceil(np.log(d2 / (0.8 * CREEP_MAX)) / np.log(1.0 / q)))
    return float(min(600.0, max(120.0, 30.0 * n)))


def last_creep(npz):
    r = np.load(npz, allow_pickle=True)
    if "frame_coord" not in r.files or len(r["frame_t"]) < 2:
        return None
    fc = r["frame_coord"].astype(float)
    return float(np.linalg.norm(fc[-1] - fc[-2], axis=1).max())


def write_deck(jd, v, job, oldjob, step, t):
    with open(os.path.join(jd, job + ".inp"), "w") as f:
        f.write(f"*Heading\n Hold extension of {oldjob}: {t:.0f} s more at the design pressure, same "
                f"increment / mass scaling / damping / load (restart from the end of step {step})\n")
        f.write(f"*Restart, read, step={step}\n**\n")
        f.write(f"*Step, name=Hold{step - PRS_HOLD_STEP + 2}\n*Dynamic, Explicit\n , {t}, , {A.DT}\n")
        f.write(A.mass_scaling())
        f.write("*Boundary, type=VELOCITY\n RIM, 1, 3, 0.0\n")
        f.write(f"*Dsload\n SG, P, {A.P_DESIGN:.6f}\n")
        f.write("*Restart, write, number interval=1, overlay\n")
        f.write(A.out_block(30.0, 0.5, True) + "*End Step\n")
    vumat = A.VUMAT_STOCK if v in A.PRS_STOCK_VUMAT else A.VUMAT_RUN
    p = A.write_bat(jd, job, oldjob, vumat)
    dbl = " double=both" if "dp" in SUFFIX else ""
    s = open(p).read().replace(f" cpus={A.CPUS} ", f" cpus={CPUS} domains={DOMAINS}{dbl} ")
    with open(p, "w", newline="\r\n") as fo:
        fo.write(s)


def report(rd, jd, job):
    npz = A.extract(rd, jd, job)
    if npz is None:
        return None
    r = A.equilibrium(npz, jd)
    c = A.convergence(npz, jd, job, prs=True)
    cr = last_creep(npz)
    A.log(rd, f"{job}: sum RF3 {100*r['frac']:.2f} % of applied, last-10 s drift {100*r['drift']:+.3f} %, "
              f"KE/IE {c['ke_ie']:.2e}, IE drift {100*c['ie_change_late']:+.3f} % / 10 s, added mass "
              f"{c['mass_change_pct']:.3g} %, last-30 s max|du| {cr:.3f} mm")
    return cr


def claim(jd):
    """Exclusive claim of one rung/variant across queue workers (stale claims of dead pids are freed)."""
    p = os.path.join(jd, "hold_ext.claim")
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                with open(p) as f:
                    pid = int(f.read().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and psutil.pid_exists(pid):
                return False
            try:
                os.remove(p)
            except OSError:
                return False
    return False


def extend(h, v, t=EXT_TIME):
    """Run the extension chain of one rung/variant as far as the rule asks.  Returns 'done' or 'wait'."""
    rd = A.rung_dir(h)
    jd = os.path.join(rd, v + SUFFIX)
    if glob.glob(os.path.join(jd, "*.lck")):
        return "wait"                                    # a job of this variant is running
    if not claim(jd):
        return "wait"                                    # another queue worker has it
    try:
        return _extend(rd, jd, v, t)
    finally:
        release(jd)


def release(jd):
    """Remove the claim; another worker may be reading it this instant (Windows: WinError 32 killed two
    workers on 2026-10-06), so retry briefly."""
    p = os.path.join(jd, "hold_ext.claim")
    for _ in range(40):
        try:
            os.remove(p)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(0.25)
    A.log(A.ROOT, f"[hold_ext] could not remove {p}")


def _extend(rd, jd, v, t):
    jobs = chain(v)
    if not A.finished(jd, jobs[0][0]):
        return "wait"
    for k in range(len(jobs)):
        job, step = jobs[k]
        if not A.finished(jd, job):
            return "done"                                # never started: the rule stopped before it
        npz = os.path.join(jd, job + "_final.npz")
        if not os.path.exists(npz) and time.time() - os.path.getmtime(os.path.join(jd, job + ".odb")) < 180:
            return "wait"                                # its own driver is probably extracting it now
        cr = last_creep(npz) if os.path.exists(npz) else report(rd, jd, job)
        if cr is None:
            return "done"
        if cr <= CREEP_MAX:
            return "done"
        if k + 1 >= len(jobs):
            A.log(rd, f"{v}: still moving {cr:.2f} mm / 30 s after {MAX_EXT} extensions -- stopped")
            return "done"
        nxt = jobs[k + 1][0]
        if A.finished(jd, nxt):
            continue
        if not os.path.exists(os.path.join(jd, job + ".res")):
            A.log(rd, f"{nxt}: restart of {job} is gone -- cannot extend")
            return "done"
        tx = ext_time(npz, t)
        write_deck(jd, v, nxt, job, step, tx)
        A.log(rd, f"{nxt}: last 30 s of {job} moved {cr:.2f} mm (> {CREEP_MAX}) -- hold +{tx:.0f} s")
        if not A.run_job(rd, jd, nxt):
            return "done"
        report(rd, jd, nxt)
    return "done"


def queue():
    todo = [(h, v) for h in A.LADDER_UP for v in A.VARIANTS]
    A.log(A.ROOT, f"[hold_ext{SUFFIX}] queue start: {len(todo)} rung/variants, creep limit {CREEP_MAX} mm / 30 s")
    while todo:
        for hv in list(todo):
            # a rung the scheduler has not built yet is WAITED for, not dropped (2026-10-06: workers
            # started 06:37 dropped every rung <= 50 mm, so 35.4 mm naive sat unextended 09:47-10:56)
            if not os.path.exists(os.path.join(A.rung_dir(hv[0]), f"setup{SUFFIX}.json")):
                continue
            try:
                if extend(*hv) == "done":
                    todo.remove(hv)
            except Exception as e:                       # one bad item must not kill the worker
                A.log(A.ROOT, f"[hold_ext] {A.MS.tag(hv[0])} {hv[1]}: {e.__class__.__name__}: {e} -- retried later")
        if todo:
            marker = "[dp] all chains done" if "dp" in SUFFIX else "all chains done"
            ladder_done = os.path.exists(NIGHT_LOG) and marker in open(NIGHT_LOG).read()
            if ladder_done and all(not glob.glob(os.path.join(A.rung_dir(h), v + SUFFIX, "*.lck")) and
                                   not A.finished(os.path.join(A.rung_dir(h), v + SUFFIX), A.jobs(v)[1])
                                   for h, v in todo):
                break                                    # ladder over; the rest never got a prs
            time.sleep(60)
    A.log(A.ROOT, "[hold_ext] queue done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=float)
    ap.add_argument("--variant", choices=A.VARIANTS)
    ap.add_argument("--time", type=float, default=EXT_TIME)
    ap.add_argument("--queue", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--creep", type=float, default=CREEP_MAX)
    a = ap.parse_args()
    SUFFIX, CREEP_MAX = a.suffix, a.creep
    if a.queue:
        queue()
    else:
        h = min(A.MS.LADDER + A.CK.FINE, key=lambda x: abs(x - a.rung))
        extend(h, a.variant, a.time)
