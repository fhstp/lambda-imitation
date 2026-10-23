#!/bin/bash
# One cluster node of the POPGym benchmark: claims up to one task per GPU via
# W&B, trains, and exits when they are done or paused before the time limit.
# Submit as many as you like, in any queue; they coordinate through W&B only.
#
#   sbatch scripts/popgym_bench_slurm.sh                       # 24 h queue
#   sbatch --partition=fast --time=05:55:00 scripts/popgym_bench_slurm.sh
#
#SBATCH --job-name=popgym-bench
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=23:55:00
#SBATCH --signal=B:TERM@1500
#SBATCH --output=popgym-bench-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# The time limit is read from SLURM (squeue %L); SIGTERM 25 min before the end
# also makes every worker checkpoint, upload to W&B and pause.
exec python examples/popgym_bench.py node "$@"
