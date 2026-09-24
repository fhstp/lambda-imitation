#!/usr/bin/env bash
# Run from the repo root, in the activated Python environment. One agent per GPU.
#   export MEMORY_GAMES_OUTPUT_DIR=/path/to/scratch/minesweeper-sweeps
#   CUDA_VISIBLE_DEVICES=0 bash examples/lambda-envs/memory_games_sweep_agent.sh \
#       <entity>/<project>/<sweep-id> --count 20
# Use separate terminals/tmux panes for additional GPUs and the other sweep.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    printf 'Usage: bash %s <entity/project/sweep-id> [wandb agent options]\n' "$0" >&2
    exit 2
fi
if [[ ! -f examples/lambda-envs/memory_games_sac.py ]]; then
    printf 'Run this command from the lambda-imitation repository root.\n' >&2
    exit 2
fi
: "${MEMORY_GAMES_OUTPUT_DIR:?Set MEMORY_GAMES_OUTPUT_DIR to a scratch/project directory}"
if [[ "$MEMORY_GAMES_OUTPUT_DIR" != /* ]]; then
    printf 'MEMORY_GAMES_OUTPUT_DIR must be an absolute path.\n' >&2
    exit 2
fi
sweep="$1"
shift
runtime="$MEMORY_GAMES_OUTPUT_DIR/runtime/${HOSTNAME:-host}-agent-$$"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TMPDIR="$runtime/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$runtime/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export JAX_COMPILATION_CACHE_DIR="$XDG_CACHE_HOME/jax"
export WANDB_DIR="$MEMORY_GAMES_OUTPUT_DIR/wandb"
export WANDB_MODE=online
export WANDB_QUIET=true
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_CONFIG_DIR="$runtime/config/wandb"
export WANDB_DATA_DIR="$runtime/wandb-data"
# A sweep controller assigns a fresh run ID/configuration to every trial.
unset WANDB_RUN_ID WANDB_RESUME WANDB_SWEEP_ID WANDB_RUN_GROUP WANDB_NAME
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR" \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"
printf 'Sweep: %s\nGPU: %s\nOutputs: %s\n' \
    "$sweep" "${CUDA_VISIBLE_DEVICES:-all visible devices}" "$MEMORY_GAMES_OUTPUT_DIR"
exec wandb agent "$@" "$sweep"
