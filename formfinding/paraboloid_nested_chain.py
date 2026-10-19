"""paraboloid_nested_chain.py -- form-find the NESTED paraboloid ladder (build_paraboloid_nested_meshes.py):
L0 = 289 nodes (identical to the existing u0289 rung -> reused, not re-run), L1 1105, L2 4321, L3 17089.

Per rung, exactly the protocol of the existing ladder: run_paraboloid_unstr_ladder.py (stage 1 nodal -> stage 2
normal + W_QUAL 1e-4 with the same stop rules -> edge projection), then project_self_stress_cst.py.
JAXFDM_SCATTER_COMPACT=1 (bit-identical, A/B-tested 2026-10-11).
Order: L1 and L2 in parallel, then L3 alone. Run dirs: temp/paraboloid_edge_constq_nest<N>_odt[_qual1e-4_warm]
(the "_odt" suffix is the driver's naming; these meshes are subdivided, not ODT-relaxed).

Watchdog: free commit < MIN_FREE_GB -> kill the paraboloid solver(s) (ckpt every 25 steps). Peak commit of every
solver process is logged (PeakPagedMemorySize64) for memory planning.
Log: temp/paraboloid_nested_chain.log
"""
import ctypes
import os
import subprocess
import threading
import time

import paraboloid_dish_config as CFG
TEMP = CFG.TEMP_DIR
HERE, PY_BASE = CFG.CODE_FORMFINDING, CFG.PY_BASE
MESH = f"{TEMP}/paraboloid_mesh_nested/mesh_L{{}}.npz"
RUNGS = {1: "nest1105", 2: "nest4321", 3: "nest17089"}
MIN_FREE_GB = 1.0
LOG = f"{TEMP}/paraboloid_nested_chain.log"
_lock = threading.Lock()


def log(msg):
    with _lock, open(LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


class MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_commit_gb():
    m = MS(); m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / 1e9


def solvers():
    """{pid: (temp_dir, peak_commit_GB)} for running paraboloid solvers."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like "
          "'*ParaboloidDish_unstructured_JAXFDM.py*' } | ForEach-Object { $p = Get-Process -Id $_.ProcessId "
          "-ErrorAction SilentlyContinue; if ($p) { '{0} {1}' -f $_.ProcessId, $p.PeakPagedMemorySize64 } }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    res = {}
    for line in out.split("\n"):
        f = line.split()
        if len(f) == 2 and f[0].isdigit():
            res[int(f[0])] = int(f[1]) / 1e9
    return res


def watchdog(stop):
    peaks = {}
    while not stop.is_set():
        fc = free_commit_gb()
        cur = solvers()
        for pid, pk in cur.items():
            if pk > peaks.get(pid, 0) + 0.25:
                log(f"solver pid {pid}: peak commit {pk:.2f} GB (system free commit {fc:.2f} GB)")
                peaks[pid] = pk
        if fc < MIN_FREE_GB:
            for pid in cur:
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
                log(f"WATCHDOG: free commit {fc:.2f} GB < {MIN_FREE_GB} -> killed solver pid {pid}")
            time.sleep(30)
        time.sleep(10)


def run_rung(L):
    tag = RUNGS[L]
    d2 = f"{TEMP}/paraboloid_edge_constq_{tag}_odt_qual1e-4_warm"
    env = dict(os.environ, JAXFDM_SCATTER_COMPACT="1", PYTHONUNBUFFERED="1")
    log(f"L{L} {tag}: start")
    with open(f"{TEMP}/paraboloid_{tag}_ladder.out", "a") as out:
        rc = subprocess.run([PY_BASE, "-u", "run_paraboloid_unstr_ladder.py", tag, MESH.format(L)], cwd=HERE,
                            env=env, stdout=out, stderr=subprocess.STDOUT).returncode
    log(f"L{L} {tag}: ladder driver exited {rc}")
    if rc != 0 or not os.path.exists(f"{d2}/jaxfdm_paraboloid_dish.npz"):
        return False
    with open(f"{d2}/projection_cst.log", "w") as out:
        rc2 = subprocess.run([PY_BASE, "project_self_stress_cst.py", f"{d2}/jaxfdm_paraboloid_dish.npz",
                              f"{d2}/jaxfdm_paraboloid_dish_projected_cst.npz"], cwd=HERE, stdout=out,
                             stderr=subprocess.STDOUT).returncode
    log(f"L{L} {tag}: CST projection exited {rc2}")
    return rc2 == 0


if __name__ == "__main__":
    log(f"chain start; free commit {free_commit_gb():.2f} GB; L0 = existing u0289 rung (reused)")
    stop = threading.Event()
    threading.Thread(target=watchdog, args=(stop,), daemon=True).start()
    res = {}
    th = [threading.Thread(target=lambda L=L: res.__setitem__(L, run_rung(L))) for L in (1, 2)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    log(f"L1/L2 done: {res}")
    res[3] = run_rung(3)
    log(f"L3 done: {res[3]}")
    stop.set(); time.sleep(12)
    log("DONE")
