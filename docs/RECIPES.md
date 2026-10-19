# Recipes: the exact settings of the published runs

Units: mm, N, MPa, tonne, s in the VaReS and Abaqus stages; the form finding runs in its own
units (R = 3, depth 1.5, q = 1) and is scaled by `geom_scale = 2500/3` afterwards.

## Stage 1: form finding of the paraboloid (Sec. 3.4.1, Table 3, Fig. 7)

Target: z = DEPTH (1 - r^2/R^2) with R = 3, DEPTH = 1.5 (f/D = 0.25). Rim pinned. Gas-law
pressure p = K_gas / V, K_gas calibrated once on the uniform-q seed to the target apex height
(apex tolerance 3 %) and then fixed (`JAXFDM_OPTIMISE_GASK` unset).

Meshes: `data/paraboloid_mesh_nested/mesh_L{0,1,2,3}.npz`, 289 / 1105 / 4321 / 17089 nodes
(48 / 96 / 192 / 384 rim nodes), nested by 1 -> 4 subdivision of the 289-node ODT base.

Three stages per mesh, all `ParaboloidDish_unstructured_JAXFDM.py` with
`JAXFDM_Q_MODE=edge JAXFDM_EDGE_SEED=const JAXFDM_NEWTON=force JAXFDM_SCATTER_COMPACT=1 JAXFDM_N_STATIONS=25`:

| stage | objective | settings | stop rule |
|---|---|---|---|
| 1 | nodal (rho, z) fit, constant-q seed | `JAXFDM_OBJECTIVE=nodal` | surface error changes < 0.2 % over 1000 accepted steps |
| 2 | normal distance + element quality | `JAXFDM_OBJECTIVE=normal JAXFDM_W_QUAL=1e-4 JAXFDM_W_TAN=0`, warm start from stage 1 (`JAXFDM_WARM_NPZ`) | over 1000 steps: surface error changes < 0.2 %, objective < 1 %, and surface error <= 1.002 x faceting floor |
| 3 | normal distance + curvature term (Eq. 27) | `JAXFDM_W_QUAL=0 JAXFDM_W_CURV=<w> JAXFDM_LBFGS_M=50`, warm start from stage 2 | as stage 2 |

`w = mean(d^2) / mean(e^2)` at the stage-2 state (d the normal offset, e the curvature measure
of Eq. 27 over the free nodes): L0 2.0710e-4, L1 5.5440e-6, L2 7.4416e-7, L3 1.237e-7
(L3 at the resumed iterate). Full environment: `data/paraboloid_edge_constq_nest289_stage3_curv/run_env.txt`
(laptop) and `formfinding/nest4321_stage3.sbatch` (Euler). The stage-2 environment of every
rung is in `data/paraboloid_edge_constq_nest*/run_env.txt`.

Gradient path: the solver differentiates its Newton equilibrium through a Sherman-Morrison adjoint and
cross-checks it once against the implicit adjoint of the damped Picard map; when the cosine is below 0.99 it
falls back to the Picard adjoint. On the near-singular stage-2 states of L2 and L3 the Picard adjoint does
not converge (jax_fdm reports the unconverged solves at exit), and on L2 the fallback made no progress in
1000 steps. `JAXFDM_NEWTON_GRAD_FORCE=1` keeps the Sherman-Morrison adjoint; the L2 stage-3 run uses it.

Drivers: `run_paraboloid_unstr_ladder.py` (stages 1-2), `run_stage3_local.py` (stage 3,
laptop), `euler_nest_stage3.py` with `nest4321_stage3.sbatch` (stage 3, Euler), followed by
`project_self_stress.py` and `project_self_stress_cst.py` (Eq. 28, nu = 0.3).

Recovery (Eq. 29-30): nodal virial from the CST-projected edge forces, quadratic moving least
squares with Gaussian weights, support rho = 8 h_ref sqrt(h/h_ref), h_ref = 159 mm
(c = 8 sqrt(159/2500) = 2.02 in the units of Eq. 30), interior nodes; area-weighted L1 and L2
of N/N_exact - 1 over all free nodes against Eq. 31 (`figures/plot_paraboloid_journal_ff_best.py`).

## Stage 2: VaReS cutting pattern (Sec. 3.4.2, Table 4, Fig. 8)

Target stress sigma* (`vares/paraboloid_profile_cst.py`, target `cstraw`): the CST-projected,
MLS-recovered tension field of the 4626-node form-found net
(`data/paraboloid_edge_constq_u4626_odt_qual1e-4_warm/jaxfdm_paraboloid_dish_projected_cst.npz`),
gore 0 without cyclic averaging, scaled to p = 0.05 MPa and divided by t = 0.9 mm; cached as a
(s, eta) grid in `data/meshstudy_vares_cst/target_grid.npz`. Meridional and hoop components
are assigned to warp and weft (exact on the gore centreline, 1.6 % L2 difference towards the
seams, Sec. 3.4.2).

Gore meshes (`vares/build_gore_nested_ladder.py --grading tippolar --bases 200`): one gore in
its developed leaf, size h min(1, 0.02 + k d), k = 2 theta / 210 mm (105/h elements across
the tip zone), then red refinement; G0-G4 = 234 / 936 / 3744 / 14976 / 59904 triangles
(`data/gore_nested_cstraw_tippolar210c200/h{200,100,050,025,012}.npz`).

Solver (`vares/MeshStudy_paraboloid_vares_ckpt.py --target cstraw --grading tippolar210c200nest`,
`MESHSTUDY_SOLVER=sparse`): residual of Eq. 34 with the full Chen law, exact non-symmetric
tangent, Levenberg damping lambda_init = 1, lambda x3 / /3, Armijo backtracking (c1 = 1e-4,
30 halvings), step cap 500 mm, `NEWTON_TOL_R = 1e-7` N/mm on the free dofs, budget 1000
iterations, SuperLU with 2 refinement passes. One node fixed and one dof aligning the warp
with the gore axis (`build_pin_dof_set_geom`); seam and rim free. Seeds: the base rung from
the converged 100 mm rung of the 100 mm chain (P1-sampled), every finer rung by prolongation
of its parent's converged pattern. A rung that stops on the step floor above the tolerance is
accepted only with the round-off certificate of `vares_roundoff.py` (G4: ||R|| 1.9e-7,
certified).

Table 4 (`figures/table4_vares_convergence.py`): Delta = area-weighted RMS displacement of
the pattern from the developed leaf after the best rigid fit (L1, L2); delta X = change of
the pattern between successive rungs (coarse P1 field at the fine nodes); Celik GCI on the
three finest rungs; e_sigma = area-weighted RMS of |sigma(eps) - sigma*| / |sigma*| over warp,
weft and shear.

## Stage 3: Abaqus/Explicit verification (Sec. 3.4.3, Table 5, Fig. 9)

Meshes (`abaqus/build_seamtip_ladder.py`): one gore, size min(h, h (0.02 + k d_tip),
6.25 + 0.4 d_seam) with h = 100 mm, structured apex strip to 20 mm, exact mirror seam pairs,
red refinement to level 4: h100 / h050 / h025 / h012 / h006 = 1.5e5 ... 9.4e6 elements over
the 24 gores (store `gore_seamtip_s625`, 144 MB, not shipped).

Decks (`abaqus/sector_seamtip_ladder.py`, `sector_abaqus_ladder.py`, `abaqus_paraboloid_cst.py`):
S3 shells, t = 0.9 mm, `vumat_chen2022.for` with the mixed stress-strain wrinkling criterion,
fabric orientation warp = meridian. Run 1 `asm`: the flat gore (VaReS pattern = the converged
6.25 mm pattern mapped by its outline on every rung; uniform = developed leaf x (1 + c_m,
1 + c_w), c from the finest rung, -0.769 % / -0.696 %) is dragged to its form-found position
and settled with only the seam, rim and tip held. Run 2 `prs`: import, cyclic `*Equation` on
the seam pairs (u_i = R u_j), tip on the axis, rim held, pressure to 0.05 MPa in 4 smooth
3 s ramps, then held. Variable mass scaling to dt = 3e-3 s; mass-proportional damping
alpha = alpha_ref M_ref / M (same time constant on every rung); double precision.

Running (`abaqus/euler_abaqus_runner.py`, Euler, Abaqus 2023, gfortran VUMAT, 6-8 cpus): the
hold is extended in stages of length T until the rim reaction matches p pi R^2 / 24 within
0.2 %, the kinetic energy is below 4e-7 of the internal energy, and no node moves more than
2e-6 D over the last T/4 (one or two stages for VaReS, up to seven for uniform shrinkage).

Measures (`abaqus/abaqus_ladder_seamband.py`, `figures/plot_seamtip_vares_vs_uniform.py`):
positions = mean of the last 30 s of frames, resultants = section forces SF of the last frame;
d = (z - z_p(r)) / u, e_w = N_w / N_phi(r) - 1, e_f = N_f / N_theta(r) - 1 against Eq. 31;
area-weighted L1 and L2 over the whole gore; Celik GCI on the three finest rungs
(`data/results/abaqus_ladder_seamtip/vares_vs_uniform/vu_summary.json`).

## Where the large stores live

| store | location |
|---|---|
| form-found L3 state and checkpoints | ETH Euler `$SCRATCH/jaxfdm/noqual/nest17089`, laptop `temp/rim_onsurface_verify/curv_17089_step19250.npz` |
| 6.25 mm VaReS pattern | laptop `temp/meshstudy_vares_cstraw_tippolar210nestgrade/h006/pattern.npz` (15 MB) |
| seam/tip Abaqus meshes | laptop `temp/gore_seamtip_s625` (144 MB) |
| Abaqus results | Euler `$SCRATCH/abaqus_sector_seamtip*`, mirrors `temp/abaqus_sector_seamtip*_mirror` |
