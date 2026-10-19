r"""euler_abaqus_runner.py -- run ONE rung/variant chain of an abaqus_paraboloid_cst ladder on ETH Euler (Linux, Slurm),
with the laptop's own queue logic.

The chain is exactly the laptop's: A.run_queue (asm -> extract -> settle gate KE/IE < 1e-3 -> prs -> extract ->
equilibrium + creep gates; restart files deleted only after every gate passes) followed by hold_ext.extend (restart
extensions cst_px2..px6 while the last 30 s of the hold moved any node more than CREEP mm).  Only the three
Windows-bound calls are replaced:
  A.run_job   -> `abaqus job=J [oldjob=O] user=<vumat> cpus=N domains=N double=both mp_mode=threads interactive
                 ask_delete=OFF` (the VUMAT is compiled per job with gfortran: ABQ_USUB_GFORTRAN=1 -- ifort 2021.10
                 hits an internal compiler error on vumat_chen2022.for, 2026-10-12); the vumat per job follows the
                 laptop rule (VUMAT_RUN, the stock VUMAT for the naive prs and its extensions);
  A.extract   -> `abaqus python extract_paraboloid_cst_odb.py` (Python 2.7-compatible extractor, Abaqus 2023);
  A.write_bat -> a placeholder text file (hold_ext.write_deck edits it; the command is built in run_job).
Licence: the same server (1973@licence-server.example.org); A.licence_up is kept.  Environment (modules, gfortran) comes
from the Slurm script.  Gates and tolerances are A's and hold_ext's, unchanged; CREEP = 0.1 mm / 30 s (dp ladders).

USAGE (inside a Slurm job)  python euler_abaqus_runner.py --root <ladder root> --rung H --variant V --cpus N [--dt DT]
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "temp", "abaqus_vares_cst")))
import abaqus_paraboloid_cst as A                 # noqa: E402
import hold_ext as X                              # noqa: E402  (chdirs to basic_examples)

EXTRACTOR = os.path.join(HERE, "extract_paraboloid_cst_odb.py")
NCPU = 1


def job_spec(job):
    """(oldjob, vumat) of a job name, by the laptop's rules."""
    kind, v = job.split("_")[1], job.split("_", 2)[2]
    stock = v in A.PRS_STOCK_VUMAT
    if kind == "asm":
        return None, A.VUMAT_RUN
    if kind == "prs":
        return f"cst_asm_{v}", (A.VUMAT_STOCK if stock else A.VUMAT_RUN)
    if kind.startswith("stl"):                            # settle imports: oldjob given by settle_chain
        return None, (A.VUMAT_STOCK if stock else A.VUMAT_RUN)
    k = int(kind[2:])                                     # pxK
    return (f"cst_prs_{v}" if k == 2 else f"cst_px{k - 1}_{v}"), (A.VUMAT_STOCK if stock else A.VUMAT_RUN)


# 2026-10-13 production ladder: Abaqus 2023 (GA build 2022_09_28, the only version on Euler) crashes in the Explicit
# PACKAGER on *Import jobs (prs, stlK) of the 25 mm and finer meshes: "package_dp rank 0 terminated by signal 11" or
# "internal error" ~20-50 s in, phase pac_exe_AnalysisDomainParallel.  Imports at 100 mm (4 domains) and 50 mm (8)
# always ran.  Tests on the h025 settle import (pactest_* jobs, $SCRATCH/pactest): independent of the
# deck (no ties / no pressure / no mass scaling / update=yes all crash), of stack limits, allocator variables and the
# filesystem (node-local copies crash); 16/12/8/4 domains, MPI mode, 32 domains and a 1-thread packager with 16 domains
# all crash (29 of 30 at 16 domains); the packager threading cannot be switched off (pac_parallel: env-file keyword
# dropped, command line refused).  CORRECTION 2026-10-14: Abaqus/Explicit 2023 IGNORES parallel=loop -- every such
# run logs "The number of domains is adjusted to 8 to suit the given CPU parameter value" -- so the "loop-8" tests and
# all imports since ran as plain cpus=8 domains=8.  Empirically 8 domains package (h025 settle 4/4, h012 prs 2/2, every
# sector import since; one 8-domain failure in the first test round), 16+ domains almost never.  --import-loop is
# kept for reproducibility of the logged commands but has no effect; the operative setting is --import-cpus 8.  A job
# whose runlog shows this packager crash is still retried (earlier attempt stashed to failed/), at most PAC_RETRY times.
PAC_SERIAL = False
PAC_RETRY = 12
PAC_SEGV = ("package_dp rank 0 terminated by signal 11", "ERROR CATEGORY:  PACKAGE")
PAC_FAIL_ANY = "Abaqus/Explicit Packager exited with an error"
IMPORT_CPUS, IMPORT_LOOP = None, False                 # --import-cpus / --import-loop


def pac_serial_env(jd):
    p = os.path.join(jd, "abaqus_v6.env")
    s = open(p).read() if os.path.exists(p) else ""
    if "pac_parallel" not in s:
        with open(p, "a") as f:
            f.write("pac_parallel=OFF\n")


def stash_failed(rd, jd, job):
    """move the files of an earlier, FAILED attempt of job (all but its .inp) to failed/<job>_<time>/ -- never
    overwritten by the rerun."""
    import glob
    import shutil
    fs = [f for f in glob.glob(os.path.join(jd, job + ".*")) + glob.glob(os.path.join(jd, job + "_PAC.*"))
          if not f.endswith(".inp")]
    if not fs:
        return
    dst = os.path.join(jd, "failed", job + time.strftime("_%Y%m%d-%H%M%S"))
    os.makedirs(dst)
    for f in fs:
        shutil.move(f, dst)
    A.log(rd, f"{job}: earlier failed attempt moved to {os.path.relpath(dst, jd)} ({len(fs)} files)")


def abq_cmd(job, oldjob=None):
    """abaqus command of one job: asm domain-parallel on NCPU; imports (prs / stlK / pxK) on IMPORT_CPUS, with
    --import-loop as ONE domain + loop-level threads."""
    old_rule, vumat = job_spec(job)
    oldjob = oldjob or old_rule
    if job.split("_")[1] == "asm" or IMPORT_CPUS is None:
        par = [f"cpus={NCPU}", f"domains={NCPU}"]
    elif IMPORT_LOOP:
        par = [f"cpus={IMPORT_CPUS}", "domains=1", "parallel=loop"]
    else:
        par = [f"cpus={IMPORT_CPUS}", f"domains={IMPORT_CPUS}"]
    return ["abaqus", f"job={job}"] + ([f"oldjob={oldjob}"] if oldjob else []) + \
           [f"user={vumat}"] + par + ["double=both", "mp_mode=threads", "interactive", "ask_delete=OFF"]


def run_job(rd, jd, job, oldjob=None):
    if A.free_gb(jd) < A.MIN_FREE_GB:
        A.log(rd, f"{job}: only {A.free_gb(jd):.1f} GB free (< {A.MIN_FREE_GB}) -- queue stopped")
        return False
    while not A.licence_up():
        A.log(rd, f"{job}: licence server {A.LIC_HOST}:{A.LIC_PORT} unreachable -- retry in 5 min")
        time.sleep(300)
    cmd = abq_cmd(job, oldjob)
    env = dict(os.environ, ABQ_USUB_GFORTRAN="1")
    if PAC_SERIAL:
        pac_serial_env(jd)
        env["PAC_PARALLEL"] = "OFF"
    for attempt in range(1, PAC_RETRY + 1):
        if not A.finished(jd, job):
            stash_failed(rd, jd, job)
        A.log(rd, f"{job}: start ({A.free_gb(jd):.1f} GB free) : {' '.join(cmd)}"
                  f"{'' if attempt == 1 else f' [retry {attempt}/{PAC_RETRY} after a packager crash]'}")
        t0 = time.time()
        rl = os.path.join(jd, job + ".runlog")
        with open(rl, "a") as out:
            subprocess.run(cmd, cwd=jd, stdout=out, stderr=subprocess.STDOUT, env=env)
        ok = A.finished(jd, job)
        A.log(rd, f"{job}: {'completed' if ok else 'FAILED'} after {(time.time() - t0) / 60:.1f} min")
        if ok:
            return True
        txt = open(rl, errors="ignore").read() if os.path.exists(rl) else ""
        if not (all(s in txt for s in PAC_SEGV) or PAC_FAIL_ANY in txt):
            return False                                   # not the packager crash -- no retry
        A.log(rd, f"{job}: Abaqus 2023 packager crash (see failed/) -- retrying")
    return False


def extract(rd, jd, job):
    out = os.path.join(jd, job + "_final.npz")
    odb = os.path.join(jd, job + ".odb")
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(odb):
        return out
    with open(os.path.join(jd, "extract.log"), "a") as lo:
        subprocess.run(["abaqus", "python", EXTRACTOR, "--odb", job + ".odb", "--out", "."], cwd=jd, stdout=lo,
                       stderr=subprocess.STDOUT)
    if not os.path.exists(out):
        A.log(rd, f"{job}: extraction FAILED (see extract.log)")
        return None
    return out


def write_bat(jd, job, oldjob=None, vumat=None):
    p = os.path.join(jd, f"run_{job}.cmd.txt")
    with open(p, "w") as f:
        f.write(f"placeholder: {job} oldjob={oldjob} user={vumat} cpus={A.CPUS} (euler_abaqus_runner builds the command)\n")
    return p


def settle_measure(npz):
    """(b) settle gate measure, revised 2026-10-13 (time-averaged).  The single-node max motion never passes at a low
    settle alpha -- local flutter, mostly in the wrinkled apex zone, 0.04-0.2 mm / 30 s while the apex is flat to
    < 0.003 mm -- so the measure uses WINDOW MEANS of the 10 s frames: m1 = mean positions over the last 30 s, m0 over
    the 30 s before.  Returns the LARGEST of
      (i)   rms over nodes of |m1 - m0|                                   (whole-shape drift)
      (ii)  2 x |apex(m1) - apex(m0)|                                     (apex drift; gate 0.01 -> 0.005 mm / 30 s)
      (iii) remaining apex rise from a fit apex(t) = a - b exp(-t/tau) over the last 60 s (0 if the window drift is
            below 0.001 mm; inf if the fit fails while the apex still drifts)."""
    import numpy as np
    r = np.load(npz, allow_pickle=True)
    if "frame_coord" not in r.files or len(r["frame_t"]) < 4:
        return None
    ft, fc = np.asarray(r["frame_t"], float), r["frame_coord"].astype(float)
    w1 = ft >= ft[-1] - 30.0 - 1e-6
    w0 = (ft >= ft[-1] - 60.0 - 1e-6) & ~w1
    if not w0.any():
        return None
    m1, m0 = fc[w1].mean(0), fc[w0].mean(0)
    drift_rms = float(np.sqrt((np.linalg.norm(m1 - m0, axis=1) ** 2).mean()))
    ia = int(np.argmax(m1[:, 2]))
    apex_drift = float(abs(m1[ia, 2] - m0[ia, 2]))
    rem = 0.0
    if apex_drift > 1e-3:
        rem = float("inf")
        sel = ft >= ft[-1] - 60.0 - 1e-6
        try:
            from scipy.optimize import curve_fit
            ap = fc[sel, ia, 2]
            t = ft[sel]
            f = lambda tt, a, b, tau: a - b * np.exp(-(tt - t[-1]) / tau)      # noqa: E731
            (a, b, tau), _ = curve_fit(f, t, ap, p0=[ap[-1], ap[-1] - ap[0], 30.0], maxfev=20000)
            if tau > 0 and np.isfinite(a):
                rem = abs(a - ap[-1])
        except Exception:
            pass
    m = max(drift_rms, 2.0 * apex_drift, rem)
    print(f"[settle] {os.path.basename(npz)}: 30 s-mean drift rms {drift_rms:.4f} mm, apex {apex_drift:.4f} mm, "
          f"remaining apex {rem:.4f} mm -> measure {m:.4f}", flush=True)
    return m


# ------------------------------------------------------------------------------------------------ settle chain
# (a') 2026-10-12: *Global Damping is NOT available in Abaqus/Explicit and material damping cannot change on restart,
# so the low-damping final phase is a separate IMPORT analysis: cst_stl<k>_<v> imports the end state of the previous
# job (state=yes, update=no -- exactly as prs imports asm), REDEFINES the material with *Damping alpha = alpha_low
# (Abaqus warns "material redefined", tested 2026-10-12: apex 18.63 -> 19.64 mm in 30 s, the fitted limit), re-declares
# the prs model data (seam TIE MPCs, surface; same 1248 non-coincident-node TIE warnings as prs) and holds the design
# pressure with the rim held for SETTLE_T s, field frames every 10 s.  Repeated (stl2 imports stl1, ...) until the
# (b) gate settle_measure <= creep.  Mass-proportional damping shapes only the path, not the equilibrium.
SETTLE_T, SETTLE_MAX = 120.0, 6


def settle_deck(jd, v, job, alpha_low, t):
    rd_ = lambda f: open(os.path.join(jd, f), newline=None).read()              # noqa: E731
    prs, asm = rd_(f"cst_prs_{v}.inp"), rd_(f"cst_asm_{v}.inp")
    mat = asm[asm.index("*Material, name=FABRIC"):asm.index("**\n*Section Controls")]
    import re
    mat, n = re.subn(r"\*Damping, alpha=[0-9.eE+-]+", f"*Damping, alpha={alpha_low:.6g}", mat)
    # one *Damping line per material: 1 normally; 2 with apply_local_regularisation.py (FABRIC + FABRIC_CORNER,
    # 2026-10-15) -- every material gets the settle alpha
    assert n == mat.count("*Material, name="), "material damping line(s)"
    # split at the end of the import block: the model data that follows (full model: "**\n*MPC\n..." seam ties;
    # one-gore sector: "**\n*Nset, nset=TIP\n...*Equation..." cyclic constraints, 2026-10-13) is re-declared as is
    imp = "*Import Nset\n"
    k0 = prs.index("\n", prs.index(imp) + len(imp)) + 1       # end of the nset line (" RIM" or " RIM, DUMMYN")
    head, rest = prs[:k0], prs[k0:]
    assert rest.startswith("**\n*"), "prs deck: model data expected right after the import block"
    head = re.sub(r"\*Heading\n[^\n]*\n", f"*Heading\n {job}: settle import at material alpha {alpha_low:g} "
                                          f"(design pressure held, rim held)\n", head, count=1)
    model = rest[:rest.index("*Step, name=Press1")]
    hold = rest[rest.index("*Step, name=Hold"):]
    hold, k1 = re.subn(r"(\*Dynamic, Explicit\n , )[0-9.]+(, , )", r"\g<1>%g\2" % t, hold, count=1)
    hold, k2 = re.subn(r"\*Output, field, time interval=30\.00", "*Output, field, time interval=10.00", hold)
    assert k1 == 1 and k2 == 1, "hold step template"
    with open(os.path.join(jd, job + ".inp"), "w") as f:
        f.write(head + mat + model + hold.replace("*Step, name=Hold", "*Step, name=Settle", 1))


def settle_chain(rd, jd, v, alpha_low, creep, start_job):
    """stl1 imports start_job, stl2 imports stl1, ... until settle_measure <= creep (at most SETTLE_MAX)."""
    old = start_job
    for k in range(1, SETTLE_MAX + 1):
        job = f"cst_stl{k}_{v}"
        if not A.finished(jd, job):
            settle_deck(jd, v, job, alpha_low, SETTLE_T)
            A.log(rd, f"{job}: settle import of {old} at alpha {alpha_low:g}, {SETTLE_T:.0f} s")
            if not run_job(rd, jd, job, oldjob=old):
                return "failed"
        npz = extract(rd, jd, job)
        if npz is None:
            return "failed"
        r = A.equilibrium(npz, jd)
        m = settle_measure(npz)
        A.log(rd, f"{job}: RF3 {100 * r['frac']:.2f} % of applied, settle measure {m:.4f} mm (gate {creep})")
        if m is not None and m <= creep:
            return f"settled ({job})"
        old = job
    return f"NOT settled after {SETTLE_MAX} settle imports"


def last_hold_job(jd, v):
    for k in range(6, 1, -1):
        if A.finished(jd, f"cst_px{k}_{v}"):
            return f"cst_px{k}_{v}"
    return A.jobs(v)[1]


def main():
    global NCPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--rung", type=float, required=True)
    ap.add_argument("--variant", required=True, choices=A.VARIANTS)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--creep", type=float, default=0.1)
    ap.add_argument("--settle-alpha", type=float, default=None,
                    help="(a')+(b): after prs, settle imports at this material alpha until settle_measure <= --creep "
                         "(replaces hold_ext extensions)")
    ap.add_argument("--settle-max", type=int, default=6, help="max settle imports (fine rungs settle slower)")
    ap.add_argument("--asm-only", action="store_true",
                    help="pre-check: run the assemble job + extraction + its settle gate (KE/IE < A.ASM_KE_OK) only")
    ap.add_argument("--settle-from-last", action="store_true",
                    help="test: skip asm/prs, settle-import the last existing hold job (prs or pxK)")
    ap.add_argument("--dt", type=float, default=None,
                    help="sensitivity runs only: the decks' explicit increment / mass-scaling dt, so hold extensions match")
    ap.add_argument("--import-cpus", type=int, default=None,
                    help="cpus of the import jobs (prs, stlK, pxK); default --cpus (2026-10-13 packager crash: 8)")
    ap.add_argument("--import-loop", action="store_true",
                    help="import jobs with domains=1 parallel=loop (the only parallel setting the 2023 packager survives)")
    ap.add_argument("--stock-vumat-imports", action="store_true",
                    help="imports (prs, stlK) of this variant with the stock VUMAT (start-up viscosity), as naive's")
    ap.add_argument("--skip-asm-gate", action="store_true",
                    help="naive only (user decision 2026-10-14, option 2): start prs even if the asm KE/IE gate fails; "
                         "the settle gate still judges the final state")
    ap.add_argument("--no-asm", action="store_true",
                    help="import-chain job: never run the asm here (it ran in its own --asm-only job); exit 4 if missing")
    a = ap.parse_args()
    NCPU = a.cpus
    global IMPORT_CPUS, IMPORT_LOOP
    IMPORT_CPUS, IMPORT_LOOP = a.import_cpus, a.import_loop
    if a.stock_vumat_imports:
        # 2026-10-14 uniform tip test ($SCRATCH/uniformtest): the 12.5 mm uniform settle import blew
        # up at increment 2 with VUMAT_RUN (CVISC 0) in every repeat and with bulk viscosity x2, ran cleanly with the
        # stock VUMAT (start-up viscosity CVISC 0.01, a rate term: the at-rest state is unaffected) -- the remedy
        # naive's imports already use.  The asm keeps VUMAT_RUN.
        A.PRS_STOCK_VUMAT = tuple(sorted(set(A.PRS_STOCK_VUMAT) | {a.variant}))
    if a.skip_asm_gate:
        if a.variant != "naive":
            raise SystemExit("--skip-asm-gate is for naive only (user decision 2026-10-14)")
        A.ASM_GATE_SKIP.add(a.variant)
    global SETTLE_MAX
    SETTLE_MAX = a.settle_max
    A.ROOT, A.SUFFIX = a.root, "_dp"
    if a.dt is not None:
        A.DT = a.dt
    A.run_job, A.extract, A.write_bat = run_job, extract, write_bat
    X.SUFFIX, X.CREEP_MAX, X.CPUS, X.DOMAINS = "_dp", a.creep, a.cpus, a.cpus
    if a.asm_only:
        rd = A.rung_dir(a.rung)
        jd = os.path.join(rd, a.variant + "_dp")
        j1 = A.jobs(a.variant)[0]
        if not A.finished(jd, j1) and not run_job(rd, jd, j1):
            print("[runner] asm FAILED", flush=True)
            sys.exit(1)
        npz = extract(rd, jd, j1)
        if npz is None:
            sys.exit(1)
        c = A.convergence(npz, jd, j1, prs=False)
        ok = c["ke_ie"] < A.ASM_KE_OK
        A.log(rd, f"{j1}: asm pre-check KE/IE {c['ke_ie']:.2e} (gate {A.ASM_KE_OK}) -> {'PASS' if ok else 'FAIL'}, "
                  f"added mass {c['mass_change_pct']:.3g} %")
        if not ok:
            sys.exit(3)                                    # gate failed: the afterok import job never starts
        return
    if a.settle_alpha is not None:
        clean0 = A.clean

        def clean(rd, jd, job):                            # the settle imports need the prs restart
            if "_asm_" in job:
                clean0(rd, jd, job)
            else:
                A.log(rd, f"{job}: restart kept for the settle imports")
        A.clean = clean
        rd = A.rung_dir(a.rung)
        jd = os.path.join(rd, a.variant + "_dp")
        j1, j2 = A.jobs(a.variant)
        if a.no_asm and not (A.finished(jd, j2) or (A.finished(jd, j1) and os.path.exists(os.path.join(jd, j1 + ".abq")))):
            A.log(rd, f"{a.variant}: --no-asm but {j1} is not finished with its restart -- import chain not started")
            sys.exit(4)
        if not a.settle_from_last:
            A.run_queue(a.rung, [a.variant])
            if not A.finished(jd, A.jobs(a.variant)[1]):
                print("[runner] prs did not finish -- no settle", flush=True)
                sys.exit(1)                                # Slurm shows FAILED, not COMPLETED 0:0 (2026-10-13)
        start = last_hold_job(jd, a.variant)
        res = settle_chain(rd, jd, a.variant, a.settle_alpha, a.creep, start)
        A.log(rd, f"{a.variant}: settle chain from {start}: {res}")
        print(f"[runner] {A.MS.tag(a.rung)} {a.variant}: settle -> {res}", flush=True)
        if not res.startswith("settled"):
            sys.exit(2)
        return
    A.run_queue(a.rung, [a.variant])
    print(f"[runner] {A.MS.tag(a.rung)} {a.variant}: extend -> {X.extend(a.rung, a.variant)}", flush=True)


if __name__ == "__main__":
    main()
