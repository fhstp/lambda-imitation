#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${HISTORY_B200_PYTHON:-$ROOT/.venv-b200/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo 'Run bash deployment/b200/setup.sh first (Python 3.12, isolated environment).' >&2
  exit 1
fi
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if (( $# == 0 )); then set -- start; fi
exec "$PYTHON" -u -B "$ROOT/deployment/b200/worker_queue.py" "$@"
