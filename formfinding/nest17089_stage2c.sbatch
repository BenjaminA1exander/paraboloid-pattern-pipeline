#!/bin/bash
#SBATCH --job-name=nest17089_s2c
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --time=24:00:00
#SBATCH --output=$SCRATCH/jaxfdm/nest17089/slurm/%x_%j.out
# Paraboloid nested L3 stage 2c: W_QUAL = 1e-4 * (h_L3/h_L2)^4 = 6.25e-6 (keeps the L2 shape:quality balance),
# warm-started from the converged stage-2b result (step 12251); stop rule as 2b.
# Solver env = laptop run_env.txt of temp/paraboloid_edge_constq_nest17089_odt_qual1e-4_warm; only paths differ.
module load stack/2024-06 python/3.10.13
source $HOME/venv_jaxfdm/bin/activate
set -u
export CODE=$HOME/jaxfdm/01_tutorials/basic_examples
export RUN=$SCRATCH/jaxfdm/nest17089/stage2c
# Euler additions (not solver settings)
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# laptop env, paths translated
export JAXFDM_CKPT_EVERY=25 JAXFDM_EDGE_SEED=const JAXFDM_IMPROVE_TOL=0 JAXFDM_LBFGS_LS_MAX=24
export JAXFDM_NEWTON=force JAXFDM_NO_VIEWER=1 JAXFDM_N_OUTER=20000 JAXFDM_N_STATIONS=25
export JAXFDM_OBJECTIVE=normal JAXFDM_PATIENCE=1000000 JAXFDM_Q_MODE=edge JAXFDM_SCATTER_COMPACT=1
export JAXFDM_SNAP_EVERY=50 JAXFDM_STAB_EVERY=100 JAXFDM_W_QUAL=6.25e-6 JAXFDM_W_TAN=0
export JAXFDM_MESH_NPZ=${RUN}/mesh.npz JAXFDM_TEMP_DIR=${RUN}
export JAXFDM_WARM_NPZ=$SCRATCH/jaxfdm/nest17089/stage2b/jaxfdm_paraboloid_dish.npz
echo "[$(date "+%Y-%m-%d %H:%M:%S")] job ${SLURM_JOB_ID} on $(hostname) ($(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs))" >> ${RUN}/chain.log
env | sort > ${RUN}/run_env_euler.txt
python ${RUN}/../euler_nest17089_stage2c.py
