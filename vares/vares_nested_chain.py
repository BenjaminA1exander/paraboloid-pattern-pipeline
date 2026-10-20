r"""vares_nested_chain.py -- runs the NESTED tip-polar VaReS ladder parent -> child.

Meshes: temp/gore_nested_cstraw_tippolar/<tag>.npz (build_gore_nested_ladder.py --grading tippolar), solved by the
unchanged checkpointed worker MeshStudy_paraboloid_vares_ckpt.py --target cstraw --grading tippolarnest
(NEWTON_TOL_R 1e-7, budget MESHSTUDY_MAX_IT 1000 as every cst ladder).

  * order: family B 70.7 -> 35.4 -> 17.7 and family A 50 -> 25 -> 12.5.  A rung starts once its parent's
    pattern.npz is CONVERGED (the worker seeds it by prolongating the parent pattern); base rungs seed from the
    converged tiptest rung of the same tag;
  * every worker runs at BELOW_NORMAL priority, so the running Normal-priority jobs keep the CPU;
  * RAM gate: a rung starts only when available physical RAM, minus what already-running workers of this chain have
    yet to allocate, is >= its dense peak estimate 3 (2N)^2 x 8 B (as --launch) + MARGIN_GB; else it waits;
  * RAM watch: if available RAM stays below CRIT_GB for CRIT_POLLS polls while a worker of this chain runs, the
    largest one is stopped (it loses < 1 Newton iteration: ckpt.npz is written every iteration) and waits for the
    gate again -- the other jobs keep their memory;
  * a worker that dies without a pattern.npz is restarted (resumes from ckpt.npz) up to MAX_RESTART times; a rung
    that ends NOT converged stops its branch (logged; the ramp driver is the manual fallback);
  * power_guard.py runs whenever a worker of this chain runs (suspend on battery, keep awake on AC).
: rungs run with MESHSTUDY_SOLVER=sparse
and MESHSTUDY_ASSEMBLY_PROCS (vares_sparse.py, validate_vares_sparse.py) -- memory ~O(nnz), so the 4th nested level
fits: B3 8.84 mm (h009, parent h018) and A3 6.25 mm (h006, parent h012).  Peak estimate for the RAM gate:
0.4 GB + 40 kB per node + 0.25 GB per assembly process.  Rungs already converged (dense) are not re-run.
Log <study root>/chain.log, state <study root>/chain_state.json.
USAGE  (all take --grading tippolar210nest [default] | tippolarnest)
       python vares_nested_chain.py --launch     # start the chain detached and return
       python vares_nested_chain.py --status
       python vares_nested_chain.py              # the chain itself (blocking)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import MeshStudy_paraboloid_vares_ckpt as CK      # noqa: E402
import vares_study as MS           # noqa: E402

TARGET = "cstraw"
# --grading tippolar210nest | tippolarnest (A/B)
GRADING = sys.argv[sys.argv.index("--grading") + 1] if "--grading" in sys.argv else "tippolar210nest"
ROOT = CK.study_root(TARGET, GRADING)
STORE = CK.nested_store(TARGET, GRADING)
WORKER = os.path.join(HERE, "MeshStudy_paraboloid_vares_ckpt.py")
# tag: (--rung value, parent tag)
RUNGS = {"tippolarnest": {"h071": (70.7, None), "h035": (35.4, "h071"), "h018": (17.7, "h035"), "h009": (8.8, "h018"),
                          "h050": (50.0, None), "h025": (25.0, "h050"), "h012": (12.5, "h025"),
                          "h006": (6.25, "h012")},
         "tippolar210nest": {"h100": (100.0, None), "h050": (50.0, "h100"), "h025": (25.0, "h050"),
                             "h012": (12.5, "h025"), "h006": (6.25, "h012")},
         # 2026-10-13: 200 mm base, the 100 mm chain's tip (CK.GRADINGS["tippolar210c200"])
         "tippolar210c200nest": {"h200": (200.0, None), "h100": (100.0, "h200"), "h050": (50.0, "h100"),
                                 "h025": (25.0, "h050"), "h012": (12.5, "h025")}}[GRADING]
SOLVER = "sparse"
PROCS = {12.5: 4, 8.8: 4, 6.25: 6}                # assembly processes (sparse); others 1
THREADS = {}                                      # BLAS threads: the sparse path is not BLAS-bound -> 1
MARGIN_GB, CRIT_GB, CRIT_POLLS = 2.0, 0.8, 3
MAX_RESTART, POLL = 3, 60
FLAGS = 0x08000000 | 0x00000200                   # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
BELOW_NORMAL = 0x00004000
LOG = os.path.join(ROOT, "chain.log")
STATE = os.path.join(ROOT, "chain_state.json")


def log(msg):
    os.makedirs(ROOT, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def peak_gb(tag):
    n = len(np.load(os.path.join(STORE, tag + ".npz"))["X_leaf"])
    if SOLVER == "sparse":
        return 0.4 + 40e-6 * n + 0.25 * PROCS.get(RUNGS[tag][0], 1)
    return 3 * (2 * n) ** 2 * 8 / 2 ** 30


def result(tag):
    p = os.path.join(ROOT, tag, "pattern.npz")
    if not os.path.exists(p):
        return None
    return "converged" if bool(np.load(p)["converged"]) else "failed"


def workers():
    """{tag: psutil.Process} of live workers of this ladder (found by command line, survives a chain restart)."""
    out = {}
    for p in psutil.process_iter(["cmdline"]):
        c = p.info["cmdline"] or []
        if any("MeshStudy_paraboloid_vares_ckpt.py" in x for x in c) and "--rung" in c and GRADING in c:
            out[MS.tag(float(c[c.index("--rung") + 1]))] = p
    return out


def rss_gb(p):
    try:
        return p.memory_info().rss / 2 ** 30
    except psutil.Error:
        return 0.0


def start(tag):
    h = RUNGS[tag][0]
    d = os.path.join(ROOT, tag)
    os.makedirs(d, exist_ok=True)
    th = str(THREADS.get(h, 1))
    env = dict(os.environ, OMP_NUM_THREADS=th, MKL_NUM_THREADS=th, OPENBLAS_NUM_THREADS=th, MPLBACKEND="Agg",
               MESHSTUDY_MAX_IT=str(CK.MAX_IT_LAUNCH[TARGET]), MESHSTUDY_SOLVER=SOLVER,
               MESHSTUDY_ASSEMBLY_PROCS=str(PROCS.get(h, 1)))
    with open(os.path.join(d, "run.log"), "a") as f:
        p = subprocess.Popen([sys.executable, "-u", WORKER, "--rung", repr(h), "--target", TARGET,
                              "--grading", GRADING], cwd=HERE, env=env, stdin=subprocess.DEVNULL, stdout=f,
                             stderr=subprocess.STDOUT, creationflags=FLAGS | BELOW_NORMAL)
    try:
        psutil.Process(p.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except psutil.Error:
        pass
    return p.pid


def ensure_guard():
    if any(any("power_guard.py" in x for x in (p.info["cmdline"] or []))
           for p in psutil.process_iter(["cmdline"])):
        return
    pyw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    g = subprocess.Popen([pyw if os.path.exists(pyw) else sys.executable, os.path.join(HERE, "power_guard.py")],
                         cwd=HERE, creationflags=FLAGS, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    log(f"power_guard started (pid {g.pid})")


def save_state(st):
    with open(STATE + ".tmp", "w") as f:
        json.dump(st, f, indent=1)
    os.replace(STATE + ".tmp", STATE)


def chain():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    for t in RUNGS:
        st.setdefault(t, dict(restarts=0, stops=0))
    log(f"chain start (pid {os.getpid()}), root {ROOT}, peaks " +
        ", ".join(f"{t} {peak_gb(t):.2f} GB" for t in RUNGS))
    low = 0
    while True:
        live = workers()
        res = {t: result(t) for t in RUNGS}
        for t in RUNGS:
            if t in live:
                st[t]["state"] = "running"
            elif res[t]:
                if st[t].get("state") != res[t]:
                    log(f"{t}: {res[t].upper()}")
                st[t]["state"] = res[t]
            elif st[t].get("state") == "running":           # died without a pattern.npz
                if st[t].get("stopped_for_ram"):
                    st[t]["state"], st[t]["stopped_for_ram"] = "waiting", False
                else:
                    st[t]["restarts"] += 1
                    st[t]["state"] = "waiting" if st[t]["restarts"] <= MAX_RESTART else "dead"
                    log(f"{t}: worker exited without pattern.npz (restart {st[t]['restarts']}/{MAX_RESTART}"
                        f"{'' if st[t]['state'] == 'waiting' else ' -- giving up'})")
            else:
                st[t].setdefault("state", "waiting")
        vm = psutil.virtual_memory()
        avail = vm.available / 2 ** 30
        # RAM watch: protect the other (higher-priority) jobs
        if live and avail < CRIT_GB:
            low += 1
            if low >= CRIT_POLLS:
                t = max(live, key=lambda k: rss_gb(live[k]))
                log(f"RAM: available {avail:.2f} GB < {CRIT_GB} for {low} polls -- stopping {t} "
                    f"(rss {rss_gb(live[t]):.2f} GB); it resumes from ckpt.npz when the gate allows")
                st[t]["stopped_for_ram"] = True
                st[t]["stops"] += 1
                try:
                    live[t].kill()
                except psutil.Error:
                    pass
                low = 0
        else:
            low = 0
        # start what is ready, smallest first, through the RAM gate
        reserved = sum(max(0.0, peak_gb(t) - rss_gb(p)) for t, p in live.items())
        ready = [t for t in RUNGS if st[t]["state"] == "waiting"
                 and (RUNGS[t][1] is None or res[RUNGS[t][1]] == "converged")]
        for t in sorted(ready, key=peak_gb):
            need = peak_gb(t) + min(MARGIN_GB, max(0.5, peak_gb(t)))   # margin scales with the rung, 0.5-2 GB
            free_commit = CK._free_commit_gb()
            if avail - reserved >= need and free_commit - reserved >= need:
                pid = start(t)
                st[t]["state"] = "running"
                log(f"{t}: started pid {pid} (BELOW_NORMAL, {SOLVER}, {PROCS.get(RUNGS[t][0], 1)} assembly procs); peak est "
                    f"{peak_gb(t):.2f} GB, available {avail:.1f} GB, reserved {reserved:.1f} GB, "
                    f"free commit {free_commit:.1f} GB")
                reserved += peak_gb(t)
                ensure_guard()
            elif st[t].get("gate_logged") != round(avail, 0):
                st[t]["gate_logged"] = round(avail, 0)
                log(f"{t}: waiting for RAM -- needs {need:.1f} GB, available {avail:.1f} GB "
                    f"(- {reserved:.1f} reserved), free commit {free_commit:.1f} GB")
        save_state(st)
        blocked = [t for t in RUNGS if st[t]["state"] == "waiting" and RUNGS[t][1]
                   and st[RUNGS[t][1]]["state"] in ("failed", "dead")]
        for t in blocked:
            st[t]["state"] = "blocked"
            log(f"{t}: blocked -- parent {RUNGS[t][1]} {st[RUNGS[t][1]]['state']}")
        if all(st[t]["state"] in ("converged", "failed", "dead", "blocked") for t in RUNGS):
            save_state(st)
            log("chain done: " + ", ".join(f"{t} {st[t]['state']}" for t in RUNGS))
            return
        time.sleep(POLL)


def status():
    live = workers()
    vm = psutil.virtual_memory()
    print(f"available RAM {vm.available / 2 ** 30:.1f} GB of {vm.total / 2 ** 30:.1f}")
    for t in RUNGS:
        line = f"{t}: {result(t) or ('running pid %d rss %.2f GB' % (live[t].pid, rss_gb(live[t])) if t in live else 'waiting')}"
        ck = os.path.join(ROOT, t, "ckpt.npz")
        if os.path.exists(ck) and not result(t):
            c = np.load(ck)
            line += f"   it {int(c['k']) + 1}, ||R|| {float(c['R_history'][-1]):.3e}"
        print(line)


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    elif "--launch" in sys.argv:
        live = [p for p in psutil.process_iter(["cmdline"])
                if (p.info["cmdline"] or [""])[0].lower().endswith(("python.exe", "pythonw.exe"))
                and any(x.endswith("vares_nested_chain.py") for x in p.info["cmdline"][1:])
                and "--launch" not in p.info["cmdline"] and "--status" not in p.info["cmdline"]
                and p.pid != os.getpid()]
        if live:
            raise SystemExit(f"chain already running (pid {live[0].pid})")
        os.makedirs(ROOT, exist_ok=True)
        with open(os.path.join(ROOT, "chain.out"), "a") as f:
            p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--grading", GRADING], cwd=HERE,
                                 stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                 creationflags=FLAGS | BELOW_NORMAL)
        print(f"chain launched (pid {p.pid}); log {LOG}")
    else:
        chain()
