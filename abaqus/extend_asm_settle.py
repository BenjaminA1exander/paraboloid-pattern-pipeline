r"""extend_asm_settle.py -- lengthen the assembly Settle step of chosen production rungs/variants.

Why: the asm gate KE/IE < 1e-3 (unchanged) failed at 6.25 mm (vares 1.18e-3, uniform 1.43e-3) and for 12.5 mm naive
(1.05e-3).  Under mass-proportional damping the kinetic energy decays as exp(-alpha t); alpha = 40 M_ref / M quarters
per nested level (12.5: 0.543, 6.25: 0.136), so the 40 s Settle gives alpha*T = 21.7 at 12.5 mm (vares / uniform
3e-11 / 1e-10) but only 5.4 at 6.25 mm.  Settle 160 s restores alpha*T = 21.8 at 6.25 mm; for the slack naive gore at
12.5 mm it gives alpha*T = 87, the value at which naive passed at 25 mm (1.6e-4).  Only the Settle step time changes;
damping, drag, boundary conditions, outputs (field every 20 s) and the gate are unchanged; prs imports the last state.

Per directory: every cst_asm_<v>.* file and cst_asm_<v>_final.npz is MOVED to asm_settle40/ (the original deck too --
nothing overwritten), then the deck is written back with the one changed line plus a comment.  Refuses if the Settle
line is not exactly the expected one, or if asm_settle40/ already exists.

USAGE  python extend_asm_settle.py ROOT 12.5:naive 6.25:vares 6.25:uniform 6.25:naive [--settle 160]
"""
import glob
import os
import shutil
import sys

OLD, TAG = 40.0, "asm_settle40"


def tag(h):
    return "h%03d" % int(h) if h >= 10 else "h%03d" % int(round(h))


def main():
    a = sys.argv[1:]
    new = float(a[a.index("--settle") + 1]) if "--settle" in a else 160.0
    a = [x for i, x in enumerate(a) if x != "--settle" and (i == 0 or a[i - 1] != "--settle")]
    root, items = a[0], a[1:]
    for it in items:
        h, v = it.split(":")
        jd = os.path.join(root, tag(float(h)), v + "_dp")
        inp = os.path.join(jd, f"cst_asm_{v}.inp")
        src = open(inp).read()
        key = f"*Step, name=Settle\n*Dynamic, Explicit\n , {OLD}, , 0.003\n"
        if src.count(key) != 1:
            raise SystemExit(f"{inp}: Settle line not found exactly once -- nothing changed")
        dst = os.path.join(jd, TAG)
        if os.path.exists(dst):
            raise SystemExit(f"{dst} exists -- not overwriting")
        os.makedirs(dst)
        fs = sorted(set(glob.glob(os.path.join(jd, f"cst_asm_{v}.*")) + glob.glob(os.path.join(jd, f"cst_asm_{v}_*"))))
        for f in fs:
            shutil.move(f, dst)
        out = src.replace(key, f"** 2026-10-13: Settle {OLD:g} -> {new:g} s (asm KE/IE gate unchanged; alpha*T >= 21.7)\n"
                               f"*Step, name=Settle\n*Dynamic, Explicit\n , {new}, , 0.003\n")
        with open(inp, "w") as f:
            f.write(out)
        print(f"{tag(float(h))} {v}: moved {len(fs)} files to {TAG}/, Settle {OLD:g} -> {new:g} s in {inp}", flush=True)


if __name__ == "__main__":
    main()
