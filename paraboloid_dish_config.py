"""paraboloid_dish_config.py -- the one place where the paraboloid-dish pipeline is configured.

Single-chamber paraboloid dish, rigid (pinned) rim, Chen-2022 woven fabric, 24 gores that enter
the Abaqus model flat and are dragged to shape (Sec. 3.4 of the paper).

Units: N / mm / MPa / tonne (density tonne/mm^3), time s.

Pipeline (README.md):
  1. formfinding/   ParaboloidDish_unstructured_JAXFDM.py (jax env)   -> form-found net, CST projection
  2. vares/         paraboloid_profile_cst.py, MeshStudy_paraboloid_vares_ckpt.py (base env) -> cutting pattern
  3. abaqus/        sector_seamtip_ladder.py, euler_abaqus_runner.py, abaqus_ladder_seamband.py -> verification
  4. figures/       the figure and table scripts of the paper
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- physical target ---------------------------------------------------------------------
DIAMETER_MM = 5000.0          # dish aperture diameter; rim circle radius R_MM at z=0
R_MM        = 2500.0
DEPTH_MM    = 1250.0          # DEPTH/R = 0.5 (deep dish, f/D = 0.25)
F_MM        = 1250.0          # focal length = R^2/(4*DEPTH)

# ---- pressure ----------------------------------------------------------------------------
# NPL-validated regime: sigma ~ p*R_curv/(2t) with R_curv_apex = 2f = 2500 mm
#   -> N ~ 62.5 N/mm -> sigma ~ 69 MPa at t=0.9 (NPL p=0.10 target ~70 MPa, FF~0.47).
# Strain ~1%: large enough that the VaReS correction is meaningful (at 0.003 MPa this
# fabric strains ~0.1% and naive == optimised to within noise).
P_PHYS = 0.05                 # MPa

# ---- material: Chen 2022 NPL woven fabric (shared inverse<->FEA law) ---------------------
# Python law: chen2022_constitutive.UMAT_PROPS (24 constants), used by VaReS (stage 3).
# Abaqus law: vumat_chen2022.for (same 24 constants), used by the Explicit decks (stage 4).
T_FABRIC = 0.9                # mm
DENS     = 1.45e-9            # tonne/mm^3
VUMAT    = os.path.join(HERE, "abaqus", "vumat_chen2022.for")

# ---- gores -------------------------------------------------------------------------------
N_GORES   = 24                # "relatively high" per user; 2 form-finder sectors per gore
TRIM_DISH = 0.04              # gore ends on the cap circle r~120 mm (pole cap replaces the singularity)
GMSH_SIZE = 100.0             # mm target edge length for the gore mesh (24 gores -> slim gore)

# ---- form-finder reference (stage 1 runs in its own form units; do not change there) -----
FORM_R, FORM_DEPTH = 3.0, 1.5                     # -> geom_scale = R_MM/FORM_R = 833.33
# Data root of the whole pipeline (meshes, runs, patterns, decks, results).  Set PIPELINE_DATA (or the older
# VARES_TEMP_DIR) to point at it; default <repository>/data.
TEMP_DIR = os.environ.get("PIPELINE_DATA", os.environ.get("VARES_TEMP_DIR", os.path.join(HERE, "data")))
# Python interpreters of the two environments (jax for the form finding, base for VaReS / Abaqus decks)
PY_BASE = os.environ.get("PIPELINE_PY_BASE", sys.executable)
PY_JAX = os.environ.get("PIPELINE_PY_JAX", sys.executable)
# source folders of the pipeline, for drivers that start the solver as a subprocess
CODE_FORMFINDING = os.path.join(HERE, "formfinding")
FORM_NPZ = os.path.join(TEMP_DIR, "jaxfdm_paraboloid_dish.npz")

# ---- artifacts ---------------------------------------------------------------------------
ABAQUS_BAT = os.environ.get("ABAQUS_CMD", r"C:\SIMULIA\Commands\abaqus.bat")   # Abaqus launcher on this machine
