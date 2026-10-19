#!/bin/bash
#SBATCH --job-name=nest17089_noqual_curv3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --time=24:00:00
#SBATCH --output=$SCRATCH/jaxfdm/noqual/%x_%j.out
# Paraboloid nested 17089 NO-QUALITY test: W_QUAL = 0 (pure normal-distance shape objective),
# warm-started from the converged stage-2 result; stop rule as 2b (incl. surf <= 1.002 x floor).
# PART 2 (2026-10-14): resumed from its own ckpt (step 1300) with the curvature term W_CURV = mean(d^2)/mean(e^2)
# at that iterate = 1.237e-7, L-BFGS memory m = 50, best/pairs reset (JAXFDM_RESUME_RESET=1).
# PART 3 (2026-10-16): part 2 hit the 24 h wall time at step 19266; plain resume (same objective, no reset).
# Solver env = laptop run_env.txt of temp/paraboloid_edge_constq_nest17089_odt_qual1e-4_warm; only paths differ.
module load stack/2024-06 python/3.10.13
source $HOME/venv_jaxfdm/bin/activate
set -u
export CODE=$HOME/jaxfdm/01_tutorials/basic_examples
export RUN=$SCRATCH/jaxfdm/noqual/nest17089
# Euler additions (not solver settings)
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# laptop env, paths translated
export JAXFDM_CKPT_EVERY=25 JAXFDM_EDGE_SEED=const JAXFDM_IMPROVE_TOL=0 JAXFDM_LBFGS_LS_MAX=24
export JAXFDM_NEWTON=force JAXFDM_NO_VIEWER=1 JAXFDM_N_OUTER=20000 JAXFDM_N_STATIONS=25
export JAXFDM_OBJECTIVE=normal JAXFDM_PATIENCE=1000000 JAXFDM_Q_MODE=edge JAXFDM_SCATTER_COMPACT=1
export JAXFDM_SNAP_EVERY=50 JAXFDM_STAB_EVERY=100 JAXFDM_W_QUAL=0 JAXFDM_W_TAN=0
export JAXFDM_W_CURV=1.237e-7 JAXFDM_LBFGS_M=50
export JAXFDM_MESH_NPZ=${RUN}/mesh.npz JAXFDM_TEMP_DIR=${RUN}
export JAXFDM_WARM_NPZ=$SCRATCH/jaxfdm/nest17089/stage2b/jaxfdm_paraboloid_dish.npz
echo "[$(date "+%Y-%m-%d %H:%M:%S")] job ${SLURM_JOB_ID} on $(hostname) ($(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs))" >> ${RUN}/chain.log
env | sort > ${RUN}/run_env_euler.txt
python $SCRATCH/jaxfdm/nest17089/euler_nest17089_stage2c.py
