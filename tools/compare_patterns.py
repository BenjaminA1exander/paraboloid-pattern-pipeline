"""compare_patterns.py -- compare two VaReS rung results (pattern.npz) array by array.

    python tools/compare_patterns.py results/regression/h200/pattern.npz \
        data/meshstudy_vares_cstraw_tippolar210c200nestgrade/h200/pattern.npz

Prints, for every numeric array the two files share, the largest absolute and relative
difference, and the Newton histories side by side. The converged pattern X_converged is
expected to agree to round-off (1e-9 mm or better on the same machine).
"""
import sys

import numpy as np

a, b = (np.load(p, allow_pickle=True) for p in sys.argv[1:3])
worst = 0.0
for k in sorted(set(a.files) & set(b.files)):
    x, y = a[k], b[k]
    if x.dtype.kind not in "fiu" or y.dtype.kind not in "fiu":
        continue
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape:
        print(f"{k:18s} shape {x.shape} vs {y.shape}")
        continue
    if x.size == 0:
        continue
    d = np.abs(x - y).max()
    s = np.abs(y).max()
    rel = d / s if s > 0 else d
    worst = max(worst, rel) if k not in ("wall_s",) else worst
    print(f"{k:18s} max |diff| {d:.3e}   relative {rel:.3e}")
print(f"\nNewton iterations: {len(a['R_history'])} vs {len(b['R_history'])}")
for i, (ra, rb) in enumerate(zip(a["R_history"], b["R_history"])):
    print(f"  {i:3d}  {ra:.6e}  {rb:.6e}")
print(f"\nworst relative difference (wall time excluded): {worst:.3e}")
