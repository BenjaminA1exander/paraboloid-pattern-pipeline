r"""build_gore_nested_ladder.py -- the NESTED, tip-graded gore meshes for the next VaReS ladder and the Abaqus
ladder after it.  Meshes, checks and figures only: nothing is solved, no deck is written.

subdivisions of each other and the singularity at the tip.  Use a tip-graded mesh (resolution increasing
towards the tip) and a structured subdivision scheme, for the VaReS ladder and the Abaqus ladder alike;
VaReS first, Abaqus only once VaReS has converged.

Scheme (gore_subdivision.py, unchanged)
  base     build_gore_gmsh(h0, tip_affine=(0.08, 0.006)) on the cstraw profile: element size
           h0 * min(1, 0.08 + 0.006 d), d = distance from the tip (mm) -- the tested tiptest grading;
  level k  every triangle -> 4 at its edge midpoints in the developed leaf; a seam-edge midpoint goes back
           onto the seam curve y = +-hb(x), a rim-edge midpoint stays on the straight rim; new nodes are
           mapped to the surface by build_gore_gmsh's own node map; coarse nodes keep their leaf AND spatial
           coordinates bit for bit, node 0 stays the tip.  Size h0/2^k * the same grading function.
  families A 50 -> 25 -> 12.5 mm and B 70.7 -> 35.35 -> 17.675 mm (tags h050 h025 h012 / h071 h035 h018):
           sqrt(2) spacing over the whole ladder, ratio 2 and exactly nested inside each family.
Abaqus: the unchanged abaqus_paraboloid_cst flower -- 24 rotated copies of the gore mesh, seam mates TIEd,
gore tips TIEd to gore 0's tip -- so every Abaqus rung is nested as well.  Its classification
(A.topology: one tip node, exact mirror seam pairs) and seam-gap assertion (A.build_flower) are run here.

OUT  temp/gore_nested_cstraw_tiptest/<tag>.npz      one frozen mesh per rung, read by both ladders
     temp/gore_nested_cstraw_tiptest/checks.json
     fig_nested_ladder_scheme.{png,pdf}  fig_nested_ladder_vares_meshes.{png,pdf}
     fig_nested_ladder_abaqus_meshes.{png,pdf}
Existing mesh files are never overwritten (the script stops before writing anything).

--grading tippolar: base size h * min(1, 0.02 + k d),
k = 2 theta / 105 mm (CK.GRADINGS["tippolar"]) -> 105 / h elements across the tip zone at every distance.  Store
temp/gore_nested_cstraw_tippolar/, figures *_tippolar.*.  The tip then has nodes within 1 mm of it on the finest
rungs, which A.topology's 1 mm tip window would count as tip nodes: the Abaqus check here (and the Abaqus driver
later) takes the tip as node 0 instead -- reported as abaqus_tip_window_nodes.
USAGE  python build_gore_nested_ladder.py [--grading tiptest|tippolar] [--levels N] [--bases 100] [--no-figs]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import MeshStudy_paraboloid_vares_ckpt as CK      # noqa: E402
import MeshStudy_paraboloid_vares as MS           # noqa: E402
import gore_subdivision as GS                     # noqa: E402
import abaqus_paraboloid_cst as A                 # noqa: E402

import matplotlib                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.lines import Line2D               # noqa: E402

TARGET = "cstraw"
GRADING = sys.argv[sys.argv.index("--grading") + 1] if "--grading" in sys.argv else "tiptest"
FAMILIES = {"A": 50.0, "B": 70.7}
if "--bases" in sys.argv:                         # e.g. --bases 100 : one family "C" from a 100 mm base
    FAMILIES = {chr(67 + i): float(b) for i, b in enumerate(sys.argv[sys.argv.index("--bases") + 1].split(","))}
# --family D: label of a
# single extra family, e.g. --bases 128 --levels 1 --family D -> 128 -> 64 into the same store, checks_D_L1.json
if "--family" in sys.argv:
    FAMILIES = {sys.argv[sys.argv.index("--family") + 1]: b for b in FAMILIES.values()}
LEVELS = int(sys.argv[sys.argv.index("--levels") + 1]) if "--levels" in sys.argv else 2
OUT_DIR = os.path.join(CK.CFG.TEMP_DIR, "gore_nested_%s_%s" % (TARGET, GRADING))
FIG_SUFFIX = "" if GRADING == "tiptest" else "_" + GRADING
OKABE = {"A": "#0072B2", "B": "#D55E00", "seam": "#E69F00", "grey": "0.55"}
LEVEL_MARK = ("o", "s", "^")


def tag(h):
    return MS.tag(h)


def h_eff(Xs, T):
    e = Xs[T]
    a = 0.5 * np.linalg.norm(np.cross(e[:, 1] - e[:, 0], e[:, 2] - e[:, 0]), axis=1)
    return float(np.sqrt(4.0 * a.sum() / (np.sqrt(3.0) * len(T)))), a


def edges(T):
    E = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    return np.unique(E, axis=0)


def size_vs_tip(Xl, T, Xs):
    """per element: distance of the centroid from the tip along the leaf, equilateral-equivalent size."""
    _, a = h_eff(Xs, T)
    d = Xl[:, 0].max() - Xl[T][:, :, 0].mean(1)
    return d, np.sqrt(4.0 * a / np.sqrt(3.0))


def build():
    import optimise_cutting_pattern_paraboloid_gore_chen as stage3
    prof = CK.load_profile(TARGET)[0]
    ng = CK.CFG.N_GORES
    hb = GS._boundary_fn(prof, ng)
    ref = np.load(A.REF_FLOWER)
    m_ref = A.mass_index(np.asarray(ref["nodes"], float), np.asarray(ref["tris"], int))
    rungs, report = [], {}
    for f, h0 in FAMILIES.items():
        fam = GS.nested_family(prof, h0, LEVELS, ng, tip_affine=CK.GRADINGS[GRADING])
        for k, (h, Xs, Xl, seam, bnd, els) in enumerate(fam):
            T = np.asarray(els, int)
            c = GS.check((h, Xs, Xl, seam, bnd, els), prof, ng)
            c.update(family=f, level=k, base_h=h0, tag=tag(h))
            c["h_eff"], _ = h_eff(Xs, T)
            c["tip_is_node0"] = bool(np.argmax(Xl[:, 0]) == 0 and np.sum(Xl[:, 0] > Xl[0, 0] - 1e-9) == 1)
            if k:
                n = len(fam[k - 1][2])
                c["n_parent_nodes"] = n
                c["nested_leaf_exact"] = bool(np.array_equal(Xl[:n], fam[k - 1][2]))
                c["nested_spatial_exact"] = bool(np.array_equal(Xs[:n], fam[k - 1][1]))
                # every parent edge (a, b) is split at its new node m = n + i into fine edges (a, m), (m, b)
                pe = edges(np.asarray(fam[k - 1][5], int))
                fe = set(map(tuple, edges(T)))
                mi = n + np.arange(len(pe))
                c["parent_edges_split_exact"] = bool(all(
                    (min(a_, m_), max(a_, m_)) in fe and (min(b_, m_), max(b_, m_)) in fe
                    for (a_, b_), m_ in zip(pe, mi)))
            d, s = size_vs_tip(Xl, T, Xs)
            c["tip_element_size_mm"] = float(s[np.argmin(d)])
            c["size_at_d"] = {str(dd): float(np.median(s[np.abs(d - dd) < max(0.25 * dd, 1.0)]))
                              for dd in (10, 50, 100, 300, 1000)}
            c["across_at_d"] = {k_: float(2 * hb(Xl[:, 0].max() - float(k_)) / v)
                                for k_, v in c["size_at_d"].items()}
            # Abaqus flower: classification + seam-mate assertion of the unchanged deck builder
            dd_ = dict(X_leaf=Xl, X_spatial=Xs, seam_ids=np.asarray(seam, int), elements=T, n_gores=ng)
            try:
                topo = A.topology(dd_)
                c["abaqus_tip_window_nodes"] = 1
            except SystemExit:
                # finer than the 1 mm tip window: same classification with the tip = node 0 (the gmsh tip point)
                Xl2 = Xl[:, :2]
                sm = [int(i) for i in seam if Xl2[i, 1] < -1e-9]
                key = {(round(Xl2[j, 0], 6), round(-Xl2[j, 1], 6)): j for j in sm}
                sp = [int(i) for i in seam if Xl2[i, 1] > 1e-9]
                topo = dict(rim=np.where(np.abs(Xl2[:, 0] - Xl2[:, 0].min()) < A.LINE_TOL)[0], tip=0, sp=sp,
                            pair={i: key[(round(Xl2[i, 0], 6), round(Xl2[i, 1], 6))] for i in sp})
                c["abaqus_tip_window_nodes"] = int(np.sum(np.abs(Xl2[:, 0] - Xl2[:, 0].max()) < A.LINE_TOL))
            fl = A.build_flower(dd_, Xl[:, :2], topo)
            c["abaqus"] = dict(tip_node=int(topo["tip"]), rim_nodes=int(len(topo["rim"])),
                               seam_pairs=int(len(topo["pair"])), plus_seam=int(len(topo["sp"])),
                               flower_nodes=int(len(fl["nodes"])), flower_S3=int(len(fl["tris"])),
                               seal_pairs=int(len(fl["seal"])),
                               alpha_rule=float(A.ALPHA_REF * m_ref / A.mass_index(fl["nodes"], fl["tris"])))
            c["vares_dense_peak_gb"] = float(3 * (2 * len(Xl)) ** 2 * 8 / 2 ** 30)
            old = os.path.join(CK.study_root(TARGET, GRADING), tag(h), "pattern.npz")
            if os.path.exists(old):
                o = np.load(old)
                c["graded_gmsh_same_tag"] = dict(nodes=int(len(o["X_leaf"])), elements=int(len(o["elements"])),
                                                 h_eff=h_eff(o["X_spatial"], o["elements"])[0])
            report[tag(h)] = c
            print("%s %s L%d h %6.3f  h_eff %5.2f  nodes %5d  el %5d  tip el %.2f mm  q_min %.3f  inv %d  "
                  "abaqus tip %d seam pairs %d  flower %d nodes %d S3" % (
                      tag(h), f, k, h, c["h_eff"], c["n_nodes"], c["n_el"], c["tip_element_size_mm"],
                      c["q_min"], c["inverted"], topo["tip"], len(topo["pair"]), len(fl["nodes"]),
                      len(fl["tris"])), flush=True)
            rungs.append(dict(family=f, level=k, h=h, base_h=h0, Xs=Xs, Xl=Xl, seam=np.asarray(seam, int),
                              bnd=np.asarray(bnd, int), T=T, check=c,
                              Tgt=stage3.element_centre_targets(Xs, T, prof.sigma_field)))
    return prof, hb, rungs, report


def save(rungs, report):
    """write the frozen meshes once; on a re-run (figures) verify the saved ones are bit-identical instead."""
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = [os.path.join(OUT_DIR, tag(r["h"]) + ".npz") for r in rungs] + [os.path.join(OUT_DIR, "checks.json")]
    have = [p for p in paths if os.path.exists(p)]
    if have and len(have) != len(paths):
        # deeper ladder (--levels 3, 2026-10-12): verify every existing mesh bit for bit, add only the missing ones,
        # checks to a new checks_L<levels>.json
        for r in rungs:
            f = os.path.join(OUT_DIR, tag(r["h"]) + ".npz")
            if os.path.exists(f):
                s = np.load(f)
                for k_, v in (("X_spatial", r["Xs"]), ("X_leaf", r["Xl"]), ("elements", r["T"]), ("T", r["Tgt"])):
                    if not np.array_equal(s[k_], v):
                        raise SystemExit("%s: saved %s differs from a fresh build" % (tag(r["h"]), k_))
        cj = os.path.join(OUT_DIR, ("checks_%s_L%d.json" % ("".join(FAMILIES), LEVELS)) if "--family" in sys.argv
                          else "checks_L%d.json" % LEVELS)
        if os.path.exists(cj):
            raise SystemExit("not overwriting " + cj)
        new = [r for r in rungs if not os.path.exists(os.path.join(OUT_DIR, tag(r["h"]) + ".npz"))]
        rungs = new
        paths = [cj]
        print("[out] existing meshes bit-identical; adding %s" % [tag(r["h"]) for r in new])
    elif have:
        for r in rungs:
            s = np.load(os.path.join(OUT_DIR, tag(r["h"]) + ".npz"))
            for k_, v in (("X_spatial", r["Xs"]), ("X_leaf", r["Xl"]), ("elements", r["T"]),
                          ("seam_ids", r["seam"]), ("boundary_ids", r["bnd"]), ("T", r["Tgt"])):
                if not np.array_equal(s[k_], v):
                    raise SystemExit("%s: saved %s differs from a fresh build" % (tag(r["h"]), k_))
        print("[out] %s exists -- saved meshes bit-identical to a fresh build, not rewritten" % OUT_DIR)
        return
    for r in rungs:
        np.savez(os.path.join(OUT_DIR, tag(r["h"]) + ".npz"),
                 X_spatial=r["Xs"], X_leaf=r["Xl"], seam_ids=r["seam"], boundary_ids=r["bnd"], elements=r["T"],
                 T=r["Tgt"], family=r["family"], level=r["level"], h_nominal=r["h"], base_h=r["base_h"],
                 n_parent_nodes=r["check"].get("n_parent_nodes", 0), target=TARGET, grading=GRADING + "_nest",
                 tip_affine=np.asarray(CK.GRADINGS[GRADING], float), n_gores=CK.CFG.N_GORES, gore_index=0,
                 note="nested tip-graded gore mesh (build_gore_nested_ladder.py): gmsh tiptest base + "
                      "gore_subdivision red refinement; nodes 0..n_parent_nodes-1 = the parent level, unchanged")
    with open(paths[-1], "w") as fh:
        json.dump(report, fh, indent=1)
    print("[out]", OUT_DIR)


# ------------------------------------------------------------------------------------------------- figures

def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})


def seg(X, E):
    return np.stack([X[E[:, 0], :2], X[E[:, 1], :2]], 1)


def fig_scheme(prof, hb, rungs):
    fig = plt.figure(figsize=(190 / 25.4, 150 / 25.4))
    W, H = 190.0, 150.0
    F = lambda x, y, w, h: [x / W, y / H, w / W, h / H]          # noqa: E731

    # (a) one subdivision step at a seam (schematic, curvature exaggerated)
    ax = fig.add_axes(F(10, 88, 80, 52))
    xs = np.linspace(-0.2, 3.2, 200)
    yc = lambda x: 1.75 + 0.10 * x - 0.16 * (x - 1.5) ** 2       # noqa: E731
    ax.plot(xs, yc(xs), color=OKABE["seam"], lw=2.2, zorder=1)
    P = {"a": (0.0, yc(0.0)), "b": (3.0, yc(3.0)), "c": (1.5, 0.0), "d": (-0.2, 0.0), "e": (3.2, 0.0)}
    tris = [("a", "c", "b"), ("a", "d", "c"), ("b", "c", "e")]
    for t in tris:
        q = np.array([P[v] for v in t] + [P[t[0]]])
        ax.plot(q[:, 0], q[:, 1], color="k", lw=1.5, zorder=3)
    a_, b_, c_ = (np.array(P[v]) for v in ("a", "b", "c"))
    m_chord = 0.5 * (a_ + b_)
    m_seam = np.array([m_chord[0], yc(m_chord[0])])
    m_ac, m_bc = 0.5 * (a_ + c_), 0.5 * (b_ + c_)
    for p_, q_ in ((m_seam, m_ac), (m_seam, m_bc), (m_ac, m_bc)):
        ax.plot([p_[0], q_[0]], [p_[1], q_[1]], color=OKABE["A"], lw=1.0, ls=(0, (3, 1.5)), zorder=2)
    for t in tris[1:]:
        q = np.array([P[v] for v in t])
        mids = [0.5 * (q[i] + q[(i + 1) % 3]) for i in range(3)]
        for i in range(3):
            j = (i + 1) % 3
            ax.plot([mids[i][0], mids[j][0]], [mids[i][1], mids[j][1]], color=OKABE["A"], lw=1.0,
                    ls=(0, (3, 1.5)), zorder=2)
    ax.plot(*m_chord, "o", mfc="white", mec="k", ms=5.5, mew=0.9, zorder=5)
    ax.annotate("", xy=m_seam, xytext=m_chord + [0, 0.02],
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="k", shrinkA=3, shrinkB=2), zorder=6)
    ax.plot(*m_seam, "o", color=OKABE["seam"], mec="k", ms=5.5, mew=0.6, zorder=6)
    ax.plot(*np.c_[m_ac, m_bc], "o", color=OKABE["A"], ms=4.0, ls="none", zorder=5)
    ax.plot(*np.c_[a_, b_, c_], "o", color="k", ms=4.5, ls="none", zorder=5)
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-0.25, 2.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(handles=[Line2D([], [], color="k", lw=1.5, marker="o", ms=4, label="coarse element, node kept"),
                       Line2D([], [], color=OKABE["A"], lw=1.0, ls=(0, (3, 1.5)), marker="o", ms=3.5,
                              label="new edge, edge midpoint"),
                       Line2D([], [], color=OKABE["seam"], lw=2.2, label="seam curve $y = \\pm h_b(x)$"),
                       Line2D([], [], color="k", lw=0, marker="o", mfc="white", ms=5, label="seam chord midpoint"),
                       Line2D([], [], color="k", lw=0, marker="o", mfc=OKABE["seam"], mew=0.6, ms=5,
                              label="moved onto the seam curve")],
              loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=7, handlelength=2.2,
              columnspacing=1.2)
    fig.text(3 / W, 140 / H, "(a)", fontsize=10, fontweight="bold")

    # (b) the tip of family A, the three levels overlaid (each coarse edge is two fine edges)
    ax = fig.add_axes(F(106, 100, 78, 38))
    fa = [r for r in rungs if r["family"] == "A"]
    xt = fa[0]["Xl"][:, 0].max()
    L = 32.0
    styles = [("k", 1.4), (OKABE["A"], 0.8), ("#56B4E9", 0.4)]
    for r, (col, lw) in reversed(list(zip(fa, styles))):
        Xl = r["Xl"] - [xt, 0, 0]
        E = edges(r["T"])
        S = seg(Xl, E)
        keep = (S[:, :, 0].max(1) > -L - 5)
        ax.add_collection(LineCollection(S[keep], colors=col, linewidths=lw))
    x_ = np.linspace(-L - 5, 0, 300)
    for sg in (1, -1):
        ax.plot(x_, sg * hb(x_ + xt), color=OKABE["seam"], lw=0.8, zorder=0)
    ax.set_xlim(-L, 1.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Distance from the tip (mm)", labelpad=1)
    ax.set_ylabel("$y$ (mm)", labelpad=1)
    ax.legend(handles=[Line2D([], [], color=c_, lw=l_, label="%g mm" % r["h"]) for r, (c_, l_) in zip(fa, styles)],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
    fig.text(97 / W, 140 / H, "(b)", fontsize=10, fontweight="bold")

    # (c) the ladder: two interleaved nested families
    ax = fig.add_axes(F(26, 14, 44, 50))
    for row, f in enumerate(("B", "A")):
        fr = [r for r in rungs if r["family"] == f]
        he = [r["check"]["h_eff"] for r in fr]
        ax.plot(he, [row] * len(he), color=OKABE[f], lw=1.0, zorder=1)
        for k, (r, x) in enumerate(zip(fr, he)):
            ax.plot(x, row, LEVEL_MARK[k], color=OKABE[f], mec="k", mew=0.5, ms=7, zorder=3)
            ax.text(x, row + 0.17, tag(r["h"]), ha="center", va="bottom", fontsize=7)
            ax.text(x, row - 0.17, "%d el." % r["check"]["n_el"], ha="center", va="top", fontsize=6.5, color="0.3")
        for x0, x1 in zip(he[:-1], he[1:]):
            ax.annotate("", xy=(x1 * 1.12, row), xytext=(x0 / 1.12, row),
                        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=OKABE[f], shrinkA=0, shrinkB=0))
    ax.set_xscale("log")
    ax.set_xlim(85, 9)
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["family B\n(base 70.7)", "family A\n(base 50)"])
    xt_ = [10, 20, 40, 80]
    ax.set_xticks(xt_)
    ax.set_xticklabels([str(t) for t in xt_])
    ax.minorticks_off()
    ax.set_xlabel(r"Element size $h_{\mathrm{eff}}$ (mm)")
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.text(3 / W, 70 / H, "(c)", fontsize=10, fontweight="bold")

    # (d) element size and (e) elements across the gore, against distance from the tip
    bins = np.geomspace(0.5, 3000, 32)
    stats = []
    for r in rungs:
        d, s = size_vs_tip(r["Xl"], r["T"], r["Xs"])
        idx = np.digitize(d, bins)
        xm, ym = [], []
        for i in range(1, len(bins)):
            m = idx == i
            if m.sum() >= 2:
                xm.append(np.sqrt(bins[i - 1] * bins[i]))
                ym.append(np.median(s[m]))
        xm, ym = np.array(xm), np.array(ym)
        stats.append((r, xm, ym, 2.0 * hb(r["Xl"][:, 0].max() - xm) / ym))
    ax = fig.add_axes(F(91, 14, 40, 50))
    dd = np.geomspace(0.5, 3000, 200)
    for r in rungs:
        g0_, k_ = CK.GRADINGS[GRADING]
        ax.loglog(dd, r["h"] * np.minimum(1, g0_ + k_ * dd), color="0.6", lw=0.6, ls=(0, (2, 1.5)), zorder=0)
    for r, xm, ym, _ in stats:
        ax.loglog(xm, ym, LEVEL_MARK[r["level"]] + "-", color=OKABE[r["family"]], ms=2.8, lw=0.9, mec="k",
                  mew=0.3)
    ax.set_xlim(0.5, 3000)
    ax.set_xlabel("Distance from the tip $d$ (mm)")
    ax.set_ylabel("Element size (mm)", labelpad=1)
    ax.legend(handles=[Line2D([], [], color="0.6", lw=0.6, ls=(0, (2, 1.5)),
                              label=r"$h\,\min(1,\ %g + %.4g\,d)$" % CK.GRADINGS[GRADING])],
              loc="lower right", fontsize=6.5, handlelength=1.8)
    fig.text(80 / W, 70 / H, "(d)", fontsize=10, fontweight="bold")
    ax = fig.add_axes(F(146, 14, 40, 50))
    ax.axhline(1.0, color="0.6", lw=0.6, ls=(0, (2, 1.5)), zorder=0)
    for r, xm, _, n in stats:
        ax.loglog(xm, n, LEVEL_MARK[r["level"]] + "-", color=OKABE[r["family"]], ms=2.8, lw=0.9, mec="k", mew=0.3)
    ax.set_xlim(0.5, 3000)
    ax.set_ylim(0.3, 60)
    ax.set_xlabel("Distance from the tip $d$ (mm)")
    ax.set_ylabel("Elements across the gore width", labelpad=1)
    ax.legend(handles=[Line2D([], [], color=OKABE["A"], marker="o", lw=0.9, ms=3, mec="k", mew=0.3,
                              label="family A"),
                       Line2D([], [], color=OKABE["B"], marker="o", lw=0.9, ms=3, mec="k", mew=0.3,
                              label="family B"),
                       Line2D([], [], color="0.4", marker="o", lw=0, ms=3, label="base"),
                       Line2D([], [], color="0.4", marker="s", lw=0, ms=3, label="1 subdivision"),
                       Line2D([], [], color="0.4", marker="^", lw=0, ms=3, label="2 subdivisions")],
              loc="upper left", fontsize=6.5, handlelength=1.8)
    fig.text(136 / W, 70 / H, "(e)", fontsize=10, fontweight="bold")
    out = os.path.join(HERE, "fig_nested_ladder_scheme" + FIG_SUFFIX)
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")
    plt.close(fig)
    print("[fig]", out)


def fig_vares(rungs):
    fig, ax = plt.subplots(4, 3, figsize=(190 / 25.4, 170 / 25.4),
                           gridspec_kw=dict(height_ratios=[1, 1.35, 1, 1.35], hspace=0.75, wspace=0.22))
    zoom = 200.0
    for row0, f in ((0, "A"), (2, "B")):
        fr = [r for r in rungs if r["family"] == f]
        for k, r in enumerate(fr):
            Xl, T, c = r["Xl"], r["T"], r["check"]
            a1, a2 = ax[row0, k], ax[row0 + 1, k]
            P = Xl[T][:, :, :2]
            a1.add_collection(PolyCollection(P / 1000.0, facecolor="none", edgecolor="0.15", lw=0.05))
            a1.plot(Xl[r["seam"], 0] / 1000, Xl[r["seam"], 1] / 1000, ".", ms=0.3, color=OKABE["seam"])
            a1.set_xlim(Xl[:, 0].min() / 1000 - 0.03, Xl[:, 0].max() / 1000 + 0.03)
            a1.set_ylim(-0.36, 0.36)
            a1.set_aspect("equal")
            a1.set_title("%s: %s, $h_{\\mathrm{eff}}$ %.1f mm\n%d nodes, %d elements" % (
                tag(r["h"]), "gmsh base" if k == 0 else "subdivision %d" % k, c["h_eff"], c["n_nodes"],
                c["n_el"]), fontsize=7.5)
            a1.set_xlabel("$x$ (m)", labelpad=1)
            xt = Xl[:, 0].max()
            S = seg(Xl - [xt, 0, 0], edges(T))
            S = S[S[:, :, 0].max(1) > -zoom * 1.05]
            a2.add_collection(LineCollection(S, colors="0.15", linewidths=0.18))
            if k:
                pr = fr[k - 1]
                Sp = seg(pr["Xl"] - [xt, 0, 0], edges(pr["T"]))
                Sp = Sp[Sp[:, :, 0].max(1) > -zoom * 1.05]
                a2.add_collection(LineCollection(Sp, colors=OKABE[f], linewidths=0.75))
            a2.set_xlim(-zoom, 3)
            a2.set_ylim(-0.29 * zoom, 0.29 * zoom)
            a2.set_aspect("equal")
            a2.set_xlabel("Distance from the tip (mm)", labelpad=1)
            if k == 0:
                a1.set_ylabel("$y$ (m)")
                a2.set_ylabel("$y$ (mm)")
                a1.text(-0.42, 0.5, "family %s" % f, transform=a1.transAxes, rotation=90, ha="center",
                        va="center", fontsize=9, fontweight="bold", color=OKABE[f])
    fig.legend(handles=[Line2D([], [], color="0.15", lw=0.6, label="this mesh"),
                        Line2D([], [], color=OKABE["A"], lw=1.2, label="parent mesh, family A"),
                        Line2D([], [], color=OKABE["B"], lw=1.2, label="parent mesh, family B"),
                        Line2D([], [], color=OKABE["seam"], lw=0, marker="o", ms=2.5, label="seam nodes")],
               loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    out = os.path.join(HERE, "fig_nested_ladder_vares_meshes" + FIG_SUFFIX)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig]", out)


def flower_xy(r, ng):
    th = np.pi / ng
    out = []
    for g in range(ng):
        out.append((A.rot(g, th) @ r["Xs"].T).T)
    return out


def fig_abaqus(rungs):
    ng = CK.CFG.N_GORES
    fig = plt.figure(figsize=(190 / 25.4, 128 / 25.4))
    W, H = 190.0, 128.0
    F = lambda x, y, w, h: [x / W, y / H, w / W, h / H]          # noqa: E731
    # full dish, coarsest rung, gore 0 shaded
    r0 = [r for r in rungs if r["family"] == "B"][0]
    ax = fig.add_axes(F(4, 18, 66, 96))
    for g, X in enumerate(flower_xy(r0, ng)):
        P = X[r0["T"]][:, :, :2] / 1000.0
        ax.add_collection(PolyCollection(P, facecolor=(OKABE["B"] + "40") if g == 0 else "none",
                                         edgecolor="0.25", lw=0.08))
        sp = r0["seam"][r0["Xl"][r0["seam"], 1] > 0]
        o = np.argsort(r0["Xl"][sp, 0])
        ax.plot(X[sp[o], 0] / 1000, X[sp[o], 1] / 1000, color=OKABE["seam"], lw=0.6)
    R = np.hypot(*r0["Xs"][:, :2].T).max() / 1000
    ax.set_xlim(-R * 1.03, R * 1.03)
    ax.set_ylim(-R * 1.03, R * 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (m)", labelpad=1)
    ax.set_ylabel("$y$ (m)", labelpad=1)
    ax.set_title("%s, 24 gores: %d nodes, %d S3" % (tag(r0["h"]), r0["check"]["abaqus"]["flower_nodes"],
                                                     r0["check"]["abaqus"]["flower_S3"]), fontsize=7.5)
    ax.legend(handles=[Line2D([], [], color=OKABE["seam"], lw=1.0, label="seam (TIE)"),
                       PolyCollection([], facecolor=OKABE["B"] + "40", edgecolor="0.25", lw=0.3,
                                      label="gore 0 (the VaReS gore)")],
              loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7)
    fig.text(2 / W, 121 / H, "(a)", fontsize=10, fontweight="bold")
    # apex close-ups, all six rungs
    Rz = 40.0
    for i, f in enumerate(("A", "B")):
        fr = [r for r in rungs if r["family"] == f]
        for k, r in enumerate(fr):
            ax = fig.add_axes(F(84 + k * 35.5, 70 - i * 52, 31, 31))
            for X in flower_xy(r, ng):
                P = X[r["T"]][:, :, :2]
                sel = np.hypot(P[:, :, 0], P[:, :, 1]).min(1) < Rz * 1.5
                ax.add_collection(PolyCollection(P[sel], facecolor="none", edgecolor="0.2", lw=0.12))
                sp = r["seam"][r["Xl"][r["seam"], 1] > 0]
                o = np.argsort(r["Xl"][sp, 0])
                ax.plot(X[sp[o], 0], X[sp[o], 1], color=OKABE["seam"], lw=0.7)
            ax.plot(0, 0, "o", ms=3.2, mfc="white", mec="k", mew=0.7, zorder=5)
            ax.set_xlim(-Rz, Rz)
            ax.set_ylim(-Rz, Rz)
            ax.set_aspect("equal")
            ax.set_xticks([-40, 0, 40])
            ax.set_yticks([-40, 0, 40])
            c = r["check"]
            ax.set_title("%s (%s%d): tip el. %.1f mm" % (tag(r["h"]), f, k, c["tip_element_size_mm"]),
                         fontsize=7, color=OKABE[f])
            if k == 0:
                ax.set_ylabel("$y$ (mm)", labelpad=0)
            if i == 1:
                ax.set_xlabel("$x$ (mm)", labelpad=1)
    fig.text(78 / W, 121 / H, "(b)", fontsize=10, fontweight="bold")
    fig.legend(handles=[Line2D([], [], color=OKABE["seam"], lw=1.0, label="seam (TIE)"),
                        Line2D([], [], color="k", lw=0, marker="o", mfc="white", ms=4,
                               label="apex: 24 gore tips TIEd to gore 0's tip")],
               loc="lower left", bbox_to_anchor=(84 / W, 0.0), ncol=2, fontsize=7)
    out = os.path.join(HERE, "fig_nested_ladder_abaqus_meshes" + FIG_SUFFIX)
    fig.savefig(out + ".png", dpi=300)
    fig.savefig(out + ".pdf")
    plt.close(fig)
    print("[fig]", out)


def main():
    prof, hb, rungs, report = build()
    save(rungs, report)
    if "--no-figs" in sys.argv:
        return
    style()
    fig_scheme(prof, hb, rungs)
    fig_vares(rungs)
    fig_abaqus(rungs)


if __name__ == "__main__":
    main()
