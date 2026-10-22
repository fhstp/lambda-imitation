#!/usr/bin/env bash
set -euo pipefail
# Run from repository root,using Python3.11 or3.12. Never modifies system Python.
python=${PYTHON:-python3.11}
"$python" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-b200.txt -e '.[sweep,popgym-jax,popgym,dev]' cython
.venv/bin/python - <<'PY'
import jax
print('JAX',jax.__version__)
print('DEVICES',[(str(d),d.device_kind) for d in jax.devices()])
assert jax.default_backend()=='gpu','CUDA failed;do not silently use CPU'
assert all('B200' in d.device_kind for d in jax.devices()),'Expected the leased B200 host;inspect hardware/runtime before launch'
PY
.venv/bin/python -m pip freeze > b200-environment.txt
echo 'Install complete. Activate .venv;run the native and diagnostics gates BEFORE starting agents.'
