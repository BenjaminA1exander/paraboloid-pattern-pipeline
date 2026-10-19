"""
optimise_cutting_pattern_paraboloid_gore_chen.py

VaReS cutting-pattern optimisation for ONE gore of the form-found PARABOLOID
DISH (pure single-chamber, rigid rim), run with the FULL NONLINEAR Chen (2022)
law -- the exact recipe of the FINISHED optimise_cutting_pattern_npl_gore_chen.py
(current-frame target, lsq_edge seed, continuation OFF, Levenberg exact-tangent
Newton) with only the geometry + target source swapped:

    * geometry: the gore lune of the form-found paraboloid meridian, meshed
      unstructured in its developed equal-metric leaf by
      formfound_profile.MembraneProfile.build_gore_gmsh (N_GORES = 24);
    * target:   sigma*(x) = (sigma_m, sigma_t, 0) = the form-found FDM tension
      resultants N/t sampled at the meridian station of x
      (paraboloid_profile.extract_paraboloid), NOT an analytic profile field.

Material-consistency: the SAME Chen UMAT_PROPS drive this inverse solve and the
Abaqus VUMAT deck (vumat_chen2022.for) that validates it -- the NPL project's
shared-material loop.  Do NOT import the LAMINA sibling
optimise_cutting_pattern_npl_gore.py into the same session (it rebinds
vares_membrane.chen_stress_linear at import time).

Output:
    cutting_pattern_paraboloid_chen.npz          (chen npz schema + dish meta)
    optimise_cutting_pattern_paraboloid_result.png
    vares_paraboloid_stress_fields.png

Run with the anaconda3 base python:
    "%USERPROFILE%\\anaconda3\\python.exe" optimise_cutting_pattern_paraboloid_gore_chen.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paraboloid_dish_config as CFG
import paraboloid_profile as pp
# Generic Levenberg-Newton driver + tuned knobs (imports cleanly: __main__-guarded,
# full-chen module -- nothing rebound).
import optimise_cutting_pattern_npl_gore_chen as chen
from vares_membrane import (
    build_phys_targets_field,
    element_centre_physical_stress,
    _ip_shape_values,
)
from element_types import element_type_of
from flatten_semicylinder_validation import least_squares_edge_flatten


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_NPZ = CFG.PAT_NPZ
OUT_PNG = os.path.join(HERE, "optimise_cutting_pattern_paraboloid_result.png")
OUT_PNG_STRESS = os.path.join(HERE, "vares_paraboloid_stress_fields.png")
RES_WEIGHTS = chen.RES_WEIGHTS


def build_pin_dof_set_geom(X_flat, boundary_ids):
    """Geometric 3-DOF pin (copy of optimise_cutting_pattern_formfound_gore's):
    node nearest the flat centroid pinned in x+y, +x-most boundary node pinned
    in y.  Whole perimeter otherwise free -- the optimiser chooses the seam."""
    xy = X_flat[:, :2]
    c = int(np.argmin(((xy - xy.mean(0)) ** 2).sum(1)))
    bnd = np.asarray(boundary_ids, int)
    b = int(bnd[int(np.argmax(xy[bnd, 0]))])
    return sorted({2 * c, 2 * c + 1, 2 * b + 1})


def element_centre_targets(X_spatial, elements, sigma_field):
    """Physical target (sx, sy, tau) at each element's centre spatial x."""
    out = np.zeros((len(elements), 3))
    for e, elem in enumerate(elements):
        etype = element_type_of(elem)
        xi_c, eta_c = etype.centre
        Nv = _ip_shape_values(etype, xi_c, eta_c)
        pos = Nv @ X_spatial[elem]
        out[e] = sigma_field(pos)
    return out


def phys_residual_field(X_mat, X_spatial, elements, sigma_field,
                        weights=RES_WEIGHTS):
    S = element_centre_physical_stress(X_mat, X_spatial, elements,
                                       linear_material=False)
    T = element_centre_targets(X_spatial, elements, sigma_field)
    w = np.asarray(weights, dtype=float)
    return np.sqrt((((S - T) ** 2) * w).sum(axis=1)), S, T


def main():
    prof, D, meta = pp.extract_paraboloid()
    print("=" * 70)
    print("VaReS cutting-pattern optimiser -- paraboloid dish gore, full Chen law")
    print("=" * 70)
    print(f"Dish: D={meta['target_diameter_mm']:.0f} mm  depth={meta['depth_mm']:.0f} mm  "
          f"f={meta['f_mm']:.0f} mm  p={meta['p_phys']} MPa  t={prof.t} mm  "
          f"gores={CFG.N_GORES}")

    X_spatial, X_leaf, seam_ids, boundary_ids, elements = prof.build_gore_gmsh(
        CFG.GMSH_SIZE, CFG.N_GORES, gore_index=0)
    print(f"Gore mesh: gmsh leaf size {CFG.GMSH_SIZE:.0f} mm -> "
          f"{len(elements)} tris, {X_spatial.shape[0]} nodes")

    # lsq_edge seed anchored to the near-isometric developed leaf (NPL recipe).
    nodes_2d = least_squares_edge_flatten(
        X_spatial, elements, init_2d=X_leaf[:, :2], anchor_to=X_leaf[:, :2])
    X_initial = np.zeros_like(X_leaf)
    X_initial[:, :2] = nodes_2d

    pinned = build_pin_dof_set_geom(X_initial, boundary_ids)
    free_dofs = [a for a in range(2 * X_initial.shape[0]) if a not in pinned]

    # Current-frame physical target field from the form-found tensions.
    sigma_star_all, detJ_ref_all = build_phys_targets_field(
        X_initial, X_spatial, elements, prof.sigma_field)

    sm_rim = prof.sigma_field(X_spatial[np.argmax(np.hypot(X_spatial[:, 0],
                                                           X_spatial[:, 1]))])
    print(f"Target sigma* at rim: warp {sm_rim[0]:.1f} / weft {sm_rim[1]:.1f} MPa; "
          f"TARGET_FRAME=current, seed=lsq_edge, continuation=off\n")

    X, R_hist, n_indef = chen.run_inner_loop(
        X_initial.copy(), X_spatial, elements, free_dofs, None, detJ_ref_all,
        "P2 (nonlinear, t=1, current-frame)",
        linear_material=False, sigma_star_all=sigma_star_all, verbose=True)
    print(f"  -> {len(R_hist)} iters, ||R||_end = {R_hist[-1]:.4e}, "
          f"indef on {n_indef} iters\n")

    res_init, S_init, T = phys_residual_field(X_initial, X_spatial, elements,
                                              prof.sigma_field)
    res_conv, S_conv, _ = phys_residual_field(X, X_spatial, elements,
                                              prof.sigma_field)
    print("Stress-mismatch residual (weighted, MPa, nonlinear Chen):")
    print(f"  initial (lsq_edge seed) : max {res_init.max():.4f}  "
          f"mean {res_init.mean():.4f}")
    print(f"  converged (optimised)   : max {res_conv.max():.4f}  "
          f"mean {res_conv.mean():.4f}")
    print(f"  improvement factor      : "
          f"{res_init.mean() / max(res_conv.mean(), 1e-12):.1f}x")
    print(f"  converged sx in [{S_conv[:, 0].min():+.2f}, {S_conv[:, 0].max():+.2f}]"
          f"  target sig_m in [{T[:, 0].min():.2f}, {T[:, 0].max():.2f}] MPa")
    print(f"  converged sy in [{S_conv[:, 1].min():+.2f}, {S_conv[:, 1].max():+.2f}]"
          f"  target sig_t in [{T[:, 1].min():.2f}, {T[:, 1].max():.2f}] MPa")

    # Pattern-shrink diagnostic: converged vs seed edge-length ratio.
    els = np.asarray(elements)
    e01 = np.unique(np.sort(np.concatenate(
        [els[:, [0, 1]], els[:, [1, 2]], els[:, [2, 0]]]), axis=1), axis=0)
    L_ini = np.linalg.norm(X_initial[e01[:, 0], :2] - X_initial[e01[:, 1], :2], axis=1)
    L_con = np.linalg.norm(X[e01[:, 0], :2] - X[e01[:, 1], :2], axis=1)
    ratio = L_con / L_ini
    print(f"  pattern edge shrink converged/seed: mean {ratio.mean():.4f}  "
          f"[{ratio.min():.4f}, {ratio.max():.4f}]  (expected ~1% below 1)")

    np.savez(
        OUT_NPZ,
        X_initial=X_initial, X_converged=X, X_spatial=X_spatial,
        X_leaf=X_leaf, elements=els,
        seam_ids=np.asarray(seam_ids), boundary_ids=np.asarray(boundary_ids),
        R_history=np.asarray(R_hist, float),
        res_init=res_init, res_conv=res_conv,
        S_init=S_init, S_conv=S_conv, T=T,
        n_gores=CFG.N_GORES, pressure=CFG.P_PHYS, thickness=CFG.T_FABRIC,
        material="chen2022_nonlinear", mesh_backend="gmsh",
        gmsh_mesh_size=CFG.GMSH_SIZE, target_frame="current",
        starting_mesh="lsq_edge",
        geom_scale=meta["geom_scale"], depth_mm=meta["depth_mm"],
        f_mm=meta["f_mm"], diameter_mm=meta["target_diameter_mm"],
        trim_dish=CFG.TRIM_DISH, p_fdm=meta["p_fdm"],
        tension_scale=meta["tension_scale"])
    print(f"[saved] {OUT_NPZ}")

    # ---- figures ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    ax = axes[0]
    for els_x, col, lab in ((X_initial, "0.65", "seed (lsq_edge)"),
                            (X, "crimson", "converged")):
        for tri in els:
            p = els_x[np.r_[tri, tri[0]], :2]
            ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=0.3)
        ax.plot([], [], "-", color=col, label=lab)
    ax.set_aspect("equal")
    ax.set_title(f"flat gore pattern ({CFG.N_GORES} gores)")
    ax.set_xlabel("s [mm]")
    ax.set_ylabel("w [mm]")
    ax.legend(fontsize=8)

    ax = axes[1]
    cen = X[els].mean(axis=1)
    sc = ax.scatter(cen[:, 0], cen[:, 1], c=res_conv, s=14, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_title(f"converged stress mismatch (mean {res_conv.mean():.2f} MPa)")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="|sigma - sigma*| weighted [MPa]")

    ax = axes[2]
    cen_sp = X_spatial[els].mean(axis=1)
    s_cen = np.array([prof.s_of_pos(p) for p in cen_sp]) * prof.geom_scale
    o = np.argsort(s_cen)
    ax.plot(s_cen[o], S_conv[o, 0], ".", color="navy", ms=3, label="sx (warp) FEM-frame")
    ax.plot(s_cen[o], T[o, 0], "-", color="navy", lw=1.5, label="sig_m* target")
    ax.plot(s_cen[o], S_conv[o, 1], ".", color="darkorange", ms=3, label="sy (weft)")
    ax.plot(s_cen[o], T[o, 1], "-", color="darkorange", lw=1.5, label="sig_t* target")
    ax.set_xlabel("meridian s [mm] (rim -> apex)")
    ax.set_ylabel("stress [MPa]")
    ax.set_title("induced vs target stress")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("VaReS paraboloid-dish gore -- full Chen law, current-frame target",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PNG, dpi=120)
    print(f"[saved] {OUT_PNG}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(R_hist, ".-")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("||R||")
    ax.set_title("VaReS residual convergence (paraboloid gore, Chen law)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_PNG_STRESS, dpi=110)
    print(f"[saved] {OUT_PNG_STRESS}")


if __name__ == "__main__":
    main()
