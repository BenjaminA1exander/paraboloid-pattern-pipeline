"""paraboloid_dish_config.py -- single source of truth for the paraboloid-dish pipeline.

Pure single-chamber paraboloid dish, rigid (pinned) tensioned rim, Chen-2022 NPL
woven fabric, 24 gores, FLAT-facet Abaqus assembly (gores enter flat and
unmodified; joining is a physical Assemble step).

Units: N / mm / MPa / tonne (density tonne/mm^3), time s.

Pipeline stages (see PARABOLOID_DISH_PIPELINE.md):
  1. ParaboloidDish_pressure_JAXFDM.py        (jaxfdm env)  -> FORM_NPZ
  2. paraboloid_profile.py                    (base env)    -> targets N_phi/N_theta
  3. optimise_cutting_pattern_paraboloid_gore_chen.py (base) -> PAT_NPZ
  4. make_paraboloid_facets.py + inflate_facet_paraboloid.py -> job decks
  5. extract_paraboloid_odb.py (abaqus python) + compare_paraboloid_dish.py
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
PAT_NPZ    = os.path.join(TEMP_DIR, "cutting_pattern_paraboloid_chen.npz")
FACET_NPZ  = os.path.join(TEMP_DIR, "facet_paraboloid_{variant}.npz")
JOB_STEM   = "paraboloid_dish"                    # job dirs: paraboloid_dish_{vares,naive}
ABAQUS_BAT = r"C:\SIMULIA\Commands\abaqus.bat"

# NOTE (import hazard, documented in VARES_NPL_PROJECT_HANDOFF.md 9): never co-import the
# LAMINA sibling optimise_cutting_pattern_npl_gore.py with any chen script -- it rebinds
# vares_membrane.chen_stress_linear at import time.
