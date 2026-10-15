#!/usr/bin/env bash
# Eight independent GPU agents;each TRIAL uses a real3/5/10-seed vmap on ONE GPU.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 ENTITY/PROJECT/SWEEP_ID [agents=8] [assignments-per-agent=256]" >&2
  exit 2
fi
sweep=$1
agents=${2:-8}
count=${3:-256}
if ! [[ $agents =~ ^[1-9][0-9]*$ && $count =~ ^[1-9][0-9]*$ ]]; then
  echo "Positive integer agent/trial counts required" >&2
  exit 2
fi
mkdir -p "${BATTLESHIP_LOGS:-logs/battleship-agents}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
pids=()
cleanup() {
  # Only this launcher's own agent processes;never broad pkill/service management.
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM EXIT
for ((gpu=0; gpu<agents; gpu++)); do
  CUDA_VISIBLE_DEVICES=$gpu wandb agent --count "$count" "$sweep" >"${BATTLESHIP_LOGS:-logs/battleship-agents}/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
result=0
for pid in "${pids[@]}"; do wait "$pid" || result=1; done
trap - INT TERM EXIT
exit "$result"
