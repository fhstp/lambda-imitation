#!/bin/bash
# One cluster node of the POPGym benchmark: claims up to one task per GPU via
# W&B, trains, and exits when they are done or paused before the time limit.
# Submit as many as you like, in any queue; they coordinate through W&B only.
#
#   sbatch scripts/popgym_bench_slurm.sh                       # 24 h queue
#   sbatch --partition=fast --time=05:55:00 scripts/popgym_bench_slurm.sh
#
# Local files (checkpoints, logs, W&B cache/staging) go to node-local scratch:
# $POPGYM_SCRATCH if set, else $TMPDIR, else /tmp. Nothing large goes to $HOME.
# They are deleted when the job ends (results/checkpoints are in W&B); set
# POPGYM_KEEP_OUTPUT=1 to keep them. POPGYM_VENV: venv path (default ./.venv).
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
source "${POPGYM_VENV:-.venv}/bin/activate"
base="${POPGYM_SCRATCH:-${TMPDIR:-/tmp}}"
out="$base/popgym-bench-${SLURM_JOB_ID:-$$}"
mkdir -p "$out/wandb-cache" "$out/wandb-data"
export POPGYM_BENCH_OUTPUT="$out" WANDB_DIR="$out" WANDB_CACHE_DIR="$out/wandb-cache" WANDB_DATA_DIR="$out/wandb-data"
export WANDB_ARTIFACT_DIR="$out/wandb-cache/artifacts" TMPDIR="$out" XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Local output: $out ($(df -h "$base" | tail -1 | awk '{print $4}') free)"
# The time limit is read from SLURM (squeue %L). SLURM's SIGTERM 25 min before
# the end is forwarded: every worker checkpoints, uploads to W&B and pauses.
python examples/popgym_bench.py node "$@" &
pid=$!
trap 'kill -TERM "$pid" 2>/dev/null' TERM
set +e
wait "$pid"; status=$?
while kill -0 "$pid" 2>/dev/null; do wait "$pid"; status=$?; done  # a trapped SIGTERM interrupts wait
set -e
if [ -z "${POPGYM_KEEP_OUTPUT:-}" ]; then rm -rf "$out"; fi
exit "$status"
