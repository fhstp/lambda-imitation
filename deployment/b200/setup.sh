#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BASE="${HISTORY_B200_BASE_PYTHON:-python3.12}"
"$BASE" -B deployment/b200/worker_queue.py install-check
"$BASE" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"'
if [[ ! -d .venv-b200 ]]; then "$BASE" -m venv .venv-b200; fi
.venv-b200/bin/python -m pip install -r deployment/b200/requirements.txt
echo 'Setup complete. Authenticate persistently with: .venv-b200/bin/wandb login'
echo 'Then, inside tmux: bash deployment/b200/run.sh start'
