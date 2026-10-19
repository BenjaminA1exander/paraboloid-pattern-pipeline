# paraboloid-pattern-pipeline

The materialisation chain of Sec. 3.4 of

> B. A. O'Regan, *Differentiable design of pneumatic membranes*, submitted to
> Computer Methods in Applied Mechanics and Engineering (2026):

an axisymmetric parabolic reflector (aperture 5 m, f/D = 0.25, 24 gores, woven Vectran
fabric of Chen et al. 2022) carried from form finding to a cutting pattern and an
independent finite-element check.

```
 1  form finding            force-density net on nested meshes L0-L3, gas-law pressure, L-BFGS with
    (formfinding/)          adjoint gradients; self-stress projection; membrane resultants N_phi, N_theta
                            by quadratic moving least squares                         -> Fig. 7, Table 3
 2  VaReS cutting pattern   one gore; target stress from the form-found net; Variation of Reference
    (vares/)                Strategy with the non-linear Chen law, stress-matching residual (Eq. 34),
                            Levenberg-damped Newton, nested tip-graded gore meshes     -> Fig. 8, Table 4
 3  Abaqus/Explicit         one-gore cyclic sector, flat gores assembled, seamed and inflated to 0.05 MPa
    (abaqus/)               with the same law as a VUMAT; VaReS pattern against uniform shrinkage, seam-
                            and tip-resolved nested ladder 1.5e5-9.4e6 elements         -> Fig. 9, Table 5
```

The companion repository `pneumatic-dfdm` holds the gas-law closure itself and the
pneumatic-cap verification (Sec. 2 and 3.1).

## Layout

```
paraboloid_dish_config.py   the one configuration module: dish geometry, pressure, fabric, gores,
                            data root (PIPELINE_DATA), interpreters, Abaqus command
env.sh / env.ps1            put the stage folders on PYTHONPATH and set the data root
formfinding/                stage 1: meshes, the JAX-FDM solver, the stage drivers, projections, recovery
vares/                      stage 2: material law, VaReS membrane element, sparse Newton, gore meshes,
                            the ladder worker and its certification, the nested-chain driver
abaqus/                     stage 3: seam/tip-resolved meshes, decks, VUMAT, the Euler runner, ODB
                            extraction, field measures
figures/                    the figure and table scripts of Sec. 3.4 and the log-log convergence summary
data/                       the small data the chain is built on (meshes, targets, converged patterns,
                            run records, result caches); large stores stay outside (see "Data")
docs/RECIPES.md             the exact settings of every published run
results/                    outputs of re-runs (not tracked)
```

Every script keeps its own docstring with its inputs, outputs and usage. The scripts import
each other by bare module name, so the stage folders must be on `PYTHONPATH`: run
`source env.sh` (Git Bash, Linux) or `. .\env.ps1` (PowerShell) from the repository root
first. `PIPELINE_DATA` points at the data root (default `data/`).

## Environments

| stage | environment |
|---|---|
| form finding | the `jaxfdm` environment of `pneumatic-dfdm` (Python 3.10, jax 0.4.30, patched jax_fdm 0.10.0) |
| meshes, VaReS, Abaqus decks, figures | Python 3.11 with numpy 1.24, scipy 1.15, matplotlib, gmsh 4.15, psutil |
| Abaqus | Abaqus/Explicit 2024 (laptop) or 2023 (ETH Euler) with a Fortran compiler for `vumat_chen2022.for` (ifx on Windows, gfortran on Euler) |

`paraboloid_dish_config.PY_BASE` / `PY_JAX` name the two interpreters for the drivers that
start the solver as a subprocess (`PIPELINE_PY_BASE`, `PIPELINE_PY_JAX`).

## Stage 1: form finding (`formfinding/`)

| step | script | notes |
|---|---|---|
| seed mesh | `paraboloid_unstructured_mesh.py` -> `paraboloid_mesh_odt.py` | gmsh frontal-Delaunay in the conformal plane, then ODT smoothing (option B) |
| nested ladder | `build_paraboloid_nested_meshes.py` | 1 -> 4 subdivision of the 289-node base in the conformal plane: L0-L3 = 289 / 1105 / 4321 / 17089 nodes (`data/paraboloid_mesh_nested`) |
| solver | `ParaboloidDish_unstructured_JAXFDM.py` (+ `jaxfdm_scatter_patch.py`) | per-edge design variables theta = ln q, gas-law pressure p = K_gas/V with K_gas calibrated once on the uniform seed, L-BFGS with the adjoint of the Newton equilibrium; every setting is an environment variable `JAXFDM_*` |
| stages 1-2 | `run_paraboloid_unstr_ladder.py`, `paraboloid_nested_chain.py` | stage 1 nodal objective, constant-q seed; stage 2 warm start, normal-distance objective + element-quality term W_QUAL = 1e-4 |
| stage 3 | `run_stage3_local.py` (laptop), `euler_nest_stage3.py` + `*.sbatch` (Euler) | curvature term of Eq. 27, W_QUAL = 0, W_CURV = mean(d^2)/mean(e^2) at the stage-2 state |
| self-stress projection | `project_self_stress.py`, `project_self_stress_cst.py` | Eq. 28: minimum complementary energy with constant-stress triangles |
| recovery | `local_stress_recovery.py` (+ `cut_extraction.py`) | nodal virial, quadratic MLS, support rho = 8 h_ref sqrt(h/h_ref) (Eq. 30) |
| checks | `test_virial_recovery_ceiling.py`, `recover_stress_l2.py`, `plot_paraboloid_optimised_vs_solved_nested.py` | extraction ceiling with exact inputs, global L2 recovery (fails, documented), optimised state against the exact equilibrium on the target |

Figure 7 and Table 3: `figures/plot_paraboloid_journal_ff_best.py`; the table values are in
`data/rim_onsurface_verify/optimised_vs_solved_mean_rms.json`.

## Stage 2: VaReS cutting pattern (`vares/`)

| layer | modules |
|---|---|
| material | `chen2022_constitutive.py`: the biaxial / uniaxial polynomial law with tension-coupled shear (Eq. 33), its consistent tangent and failure factor; the same 24 constants as the VUMAT |
| element | `vares_membrane.py` (residual and exact non-symmetric tangent of the stress mismatch, Eq. 34), `vares_kinematics.py`, `element_kinematics.py`, `element_types.py` (3-node triangle; the 4-node quad is kept for generality) |
| linear algebra | `vares_sparse.py`: sparse assembly (bit-identical to the dense path), SuperLU solve with two refinement passes, parallel element assembly; `vares_roundoff.py`: the round-off-floor certificate of a rung whose residual cannot reach the tolerance in double precision |
| target and gore | `paraboloid_profile_cst.py` (target stress from the form-found net, raw gore 0), `formfound_profile.py` and `paraboloid_profile.py` (meridian, gore lune, developed leaf, gmsh gore mesher), `gore_subdivision.py` and `build_gore_nested_ladder.py` (nested tip-graded gore meshes) |
| driver | `MeshStudy_paraboloid_vares_ckpt.py`: the checkpointed, resumable worker for one rung (seed by prolongation of the parent pattern, geometric 3-dof pin, Levenberg-damped Newton to ||R|| < 1e-7 N/mm); `vares_nested_chain.py` runs a chain parent -> child; `certify_vares_rung.py` applies the certificate afterwards; `validate_vares_sparse.py` and `plot_vares_tangent_check.py` are the A/B checks of the sparse path and of the tangent |
| seeds and set-up | `vares_settings.py` (Newton tolerance, budget, Levenberg schedule, line search), `vares_flatten.py` (least-squares edge-length flattening of the gore, the seed of a base rung), `vares_setup.py` (the geometric 3-dof pin, element-centre targets, residual field) |
| earlier code | `legacy/`: the airship-envelope implementation these modules were consolidated from; not imported by the pipeline (only the dense loop of `MeshStudy_paraboloid_vares.run_one` still uses it) |

Figure 8 and Table 4: `figures/plot_vares_journal_combined_flat.py` and
`figures/table4_vares_convergence.py` (`data/results/convergence_loglog/conv_table.json`).

### Regression test

The 200 mm rung of the nested chain (234 triangles, 11 Newton iterations, a minute)
re-solves from the shipped data and must reproduce `data/meshstudy_vares_cstraw_tippolar210c200nestgrade/h200/pattern.npz`:

```
source env.sh
MESHSTUDY_SOLVER=sparse python vares/MeshStudy_paraboloid_vares_ckpt.py --rung 200 --target cstraw \
    --grading tippolar210c200nest --root results/regression
python tools/compare_patterns.py results/regression/h200/pattern.npz \
    data/meshstudy_vares_cstraw_tippolar210c200nestgrade/h200/pattern.npz
```

Run on 2026-10-19 from this repository: 11 Newton iterations as in the published run, identical residual
history to 1.5e-13 relative, converged pattern identical to 2.3e-13 mm.

## Stage 3: Abaqus/Explicit verification (`abaqus/`)

| step | script | notes |
|---|---|---|
| meshes | `build_seamtip_ladder.py` | one gore in the developed leaf: 6.25 mm seam elements, tip-polar grading, structured apex strip, exact mirror seam pairs; levels by red refinement (h100 ... h006 = 1.5e5 ... 9.4e6 elements over the dish) |
| decks | `sector_seamtip_ladder.py` -> `sector_abaqus_ladder.py` -> `abaqus_paraboloid_cst.py` (+ `nested_abaqus_ladder.py`, `inflate_paraboloid_chain.py`) | run 1 `asm`: flat gore dragged to its form-found position and settled; run 2 `prs`: import, cyclic `*Equation` on the seam pairs, rim held, pressure ramped in 4 smooth steps and held; VaReS pattern (the converged 6.25 mm pattern mapped on every rung) against uniform shrinkage (Eq. 35) |
| material | `vumat_chen2022.for` | the Chen law as a VUMAT with the mixed stress-strain wrinkling criterion |
| running | `euler_abaqus_runner.py` (+ `hold_ext.py`, `extend_asm_settle.py`) | the settle gates: rim reaction within 0.2 % of p pi R^2, kinetic/internal energy below 4e-7, no node moving more than 2e-6 D over the last T/4; double precision; mass-proportional damping scaled with the mesh mass |
| extraction | `extract_paraboloid_cst_odb.py` | run under Abaqus Python: coordinates, section forces SF, wrinkle state, energy histories |
| measures | `abaqus_ladder_seamband.py`, `figures/plot_abaqus_sector_vares_asymptote.py`, `figures/plot_seamtip_asymptote_combined.py` | area-weighted L1 and L2 norms of the normal deviation and of N/N_exact - 1 over the whole gore |

Figure 9 and Table 5: `figures/plot_seamtip_vares_vs_uniform.py` and
`figures/plot_seamtip_cmame_main.py` (`data/results/abaqus_ladder_seamtip/vares_vs_uniform/vu_summary.json`).

## Convergence summary

`figures/plot_convergence_loglog_all.py` draws every mesh-refinement study of the paper
(Sec. 3.1, 3.4.1, 3.4.2, 3.4.3) on log-log axes against the number of degrees of freedom or
elements, with O(h) and O(h^2) slope guides, and prints the observed orders
(`data/results/convergence_loglog/orders.txt`).

## Data

`data/` (26 MB) holds what the chain is built on and what the tables are read from:

| folder | content |
|---|---|
| `paraboloid_mesh_odt`, `paraboloid_mesh_nested` | the 289-node base mesh and the nested L0-L3 meshes |
| `paraboloid_edge_constq_nest*` | stage-2 run records (mesh, history, environment) of L0-L2 and the L0 stage-3 record |
| `paraboloid_edge_constq_u4626_odt_qual1e-4_warm` | the CST-projected form-found net the VaReS target is extracted from |
| `meshstudy_vares_cst/target_grid.npz` | the cached target stress grid (s, eta) |
| `gore_nested_cstraw_tippolar210c200` | the nested gore meshes of the 200 mm chain (G0-G4 of Table 4) |
| `meshstudy_vares_cstraw_tippolar210c200nestgrade` | the converged patterns of that chain; `..._tippolar210nestgrade/h100` the seed of its base rung |
| `rim_onsurface_verify`, `results/` | the json caches behind Tables 3, 4 and 5 |

Not shipped (sizes): the form-found L3 state and its checkpoints, the 6.25 mm VaReS pattern
(15 MB), the seam/tip Abaqus meshes (144 MB) and the Abaqus results (hundreds of GB on
Euler). `docs/RECIPES.md` says where each lives and how it was produced.

## State of the code

This repository is the organised, path-independent copy of the scripts that produced the
published results. The numerical code is unchanged: the VaReS core (`element_types.py`,
`element_kinematics.py`, `vares_kinematics.py`, `vares_membrane.py`, `vares_sparse.py`,
`vares_roundoff.py`) now also carries the shape functions, metrics, solver constants, pin
rule and flattening seed that the worker used to import from the earlier airship-envelope
modules (now in `vares/legacy/`, unused by the pipeline). The consolidation is protected
by the regression test above (same 11 Newton iterations, converged pattern identical to
round-off) and by `vares/validate_vares_sparse.py`.
