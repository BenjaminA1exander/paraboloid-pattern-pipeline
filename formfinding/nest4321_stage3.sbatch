#!/bin/bash
#SBATCH --job-name=nest4321_s3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=$SCRATCH/jaxfdm/stage3/%x_%j.out
# Paraboloid nested L4321 STAGE 3 (2026-10-19): W_QUAL = 0, curvature term W_CURV = mean(d^2)/mean(e^2) at the
# stage-2 state = 7.4416e-7 (computed on the laptop from the stage-2 npz; the same rule gave the L0 value 2.0710e-4
# and the L3 value 1.237e-7), L-BFGS m = 50, warm start from the converged stage-2 result; stop rule of the L0/L3
# stage-3 runs.  Backs the sentence of Sec. 3.4.1 "applied to the coarser meshes, it reaches the same solution".
# Solver env = temp/run_stage3_local.py (laptop L0 run); only paths differ.
module load stack/2024-06 python/3.10.13
source $HOME/venv_jaxfdm/bin/activate
set -u
export CODE=$HOME/jaxfdm/01_tutorials/basic_examples
export RUN=$SCRATCH/jaxfdm/stage3/nest4321
export SRC=$SCRATCH/jaxfdm/nest4321/stage2
mkdir -p $RUN
[ -f $RUN/mesh.npz ] || cp $SRC/mesh.npz $RUN/mesh.npz
# Euler additions (not solver settings)
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=True MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# laptop run_env.txt of the L0 stage-3 run, paths translated
export JAXFDM_CKPT_EVERY=25 JAXFDM_EDGE_SEED=const JAXFDM_IMPROVE_TOL=0 JAXFDM_LBFGS_LS_MAX=24
export JAXFDM_NEWTON=force JAXFDM_NO_VIEWER=1 JAXFDM_N_OUTER=40000 JAXFDM_N_STATIONS=25
export JAXFDM_OBJECTIVE=normal JAXFDM_PATIENCE=1000000 JAXFDM_Q_MODE=edge JAXFDM_SCATTER_COMPACT=1
export JAXFDM_SNAP_EVERY=50 JAXFDM_STAB_EVERY=100 JAXFDM_W_QUAL=0 JAXFDM_W_TAN=0
export JAXFDM_W_CURV=7.4416e-7 JAXFDM_LBFGS_M=50
export JAXFDM_MESH_NPZ=${RUN}/mesh.npz JAXFDM_TEMP_DIR=${RUN}
export JAXFDM_WARM_NPZ=${SRC}/jaxfdm_paraboloid_dish.npz
echo "[$(date "+%Y-%m-%d %H:%M:%S")] job ${SLURM_JOB_ID} on $(hostname) ($(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs))" >> ${RUN}/chain.log
env | sort > ${RUN}/run_env_euler.txt
python $SCRATCH/jaxfdm/stage3/euler_nest_stage3.py
