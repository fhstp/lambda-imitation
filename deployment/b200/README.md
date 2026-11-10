# B200 history-window workers (no OpenCode required)

This deployment is for **one B200 server only**. It owns eight already-reserved
second five-seed blocks, one for each history `0,1,2,5,10,15,20,25`. The original
server keeps the first blocks. Do not run this assignment on two servers or
change IDs. Reservations have no automatic expiry or local fallback.

## 1. Get the branch

New checkout:

```bash
git clone --branch rebuttal/history-windows-2026-11-09 git@github.com:fhstp/lambda-imitation.git lambda-history
cd lambda-history
```

Existing **clean, inactive** checkout:

```bash
git fetch origin
git switch --track origin/rebuttal/history-windows-2026-11-09
git pull --ff-only
```

If the branch already exists locally, use `git switch rebuttal/history-windows-2026-11-09`
instead of creating a tracking branch. Never discard local changes to get here.
Never `git pull` or run setup while this deployment's workers are active.

## 2. One-time setup and authentication

Requirements: Linux, `git`, `tmux`, Python **3.12** with venv support, a B200-capable
NVIDIA driver, eight full idle B200 cards, at least **96 GiB available host RAM**,
and **20 GiB free work storage after installing dependencies**. Plan extra disk
for the isolated CUDA environment. No root/systemd/OpenCode is needed.

```bash
bash deployment/b200/setup.sh
.venv-b200/bin/wandb login
```

Setup only installs into `.venv-b200`; it does not change shared packages. To
choose an installed Python 3.12 executable, set `HISTORY_B200_BASE_PYTHON` before
setup. Authenticate persistently using `wandb login` (saved credentials), not an
API key pasted into a command or an OpenCode conversation. W&B must permit
uploads to the `fhstp-data-intelligence-research-group` entity.

The server needs outbound access to GitHub/package indexes during setup, and
to W&B/API/artifact storage during execution. The custom VPN need not be
reachable from the original server: results synchronize through W&B.

## 3. Start everything with one queue command

```bash
tmux new -s history-b200-queue
bash deployment/b200/run.sh
```

Detach with **Ctrl-b, then d**. The queue creates the eight per-GPU worker tmux
sessions automatically (`rr-history-h00-b2` through `rr-history-h25-b2`). You do
not need to create them manually. Each worker sees only its assigned GPU.

Default mapping: GPU `0,1,2,3,4,5,6,7` gets H `0,1,2,5,10,15,20,25` respectively.
For another order, pass `start --gpus 7,6,5,4,3,2,1,0` to `run.sh`.

**Before the first start only**, optionally select durable scratch storage:

```bash
export HISTORY_B200_STATE=/your/persistent/scratch/history-b200-state
export HISTORY_B200_WORK=/your/persistent/scratch/history-b200-work
bash deployment/b200/run.sh
```

Default state is `~/.local/state/lambda-history-b200/<lease-id>/`, with `work/`
inside it. Keep the same state path on every later invocation; never choose a
new state directory to bypass an error. Do not use RAM-backed or purgeable
scratch storage. The selected paths are captured in `settings.json`.

The queue checks source hashes, runtime versions, free GPUs/resources, runs CPU
tests and a W&B upload/download canary, then starts each worker. Each worker
first performs the original full-dimensional 64-update native check on its
B200, discards that validation state, and starts its reserved production block
from fresh initialization. Any validation failure blocks that worker, without
changing the estimator, seeds, budget, or another worker's running training.

**There is no live B200 validation result yet at packaging time.** Driver,
network, clean-environment installation, GPU compatibility and actual speed
must pass these gates on the remote server; packaging tests do not certify them.

## 4. Inspect / reconnect

```bash
bash deployment/b200/run.sh status
tmux attach -t history-b200-queue
tmux attach -t rr-history-h25-b2
```

Per-worker logs/status/native proofs and final receipts are under the state
directory's `jobs/` and `native/`. W&B-managed training files are under the work
directory's `runs/wandb/run-*/files/training/`, not session-specific output paths.
Checkpoints are uploaded at the original cadence; their own local staging file
is released only after remote commit/size/MD5/SHA metadata verification. Failure
checkpoint files are retained. Small records remain on persistent storage.

If only the **queue terminal** exits, recreate/attach a queue tmux session and
run the same command with the **same state path**. Existing worker tickets are
not resubmitted. Administrative W&B logging may resume; policy training never
does. A lost/failed worker is shown as failed/worker-lost, not silently retried.
Do not kill worker sessions to reconnect the observer.

If training exited successfully but final result assembly failed, the explicit
read-only repair is:

```bash
bash deployment/b200/run.sh finalize --job battleship-retrace-history25-block2-5seeds
bash deployment/b200/run.sh
```

This requires a recorded exit code zero. It cannot launch or resume a learner.
For any training failure, preserve all files and contact the original agent;
do not remove tickets, change W&B IDs, replace a seed, or load a checkpoint.

## 5. Results return without the VPN

Scientific project:
https://wandb.ai/fhstp-data-intelligence-research-group/retrace-rebuttal-battleship-history-length-20261109

The queue publishes one hash-indexed result bundle per completed block to the
validation project's coordination run. The original host's read-only sync
observer downloads these, verifies config/source/run IDs and remote checkpoint
manifests, then marks the reserved logical block complete. Original reporting
still requires **all 80 planned seed instances**, with no survivor-only results
and no duplicates counted as extra seeds. If the observer is temporarily down,
the bundles remain in W&B and can be imported later.

## 6. Optional OpenCode / GPT handoff

Training does not depend on OpenCode. If its use is permitted, start a new
session from the checkout with `opencode`, select an available GPT model in
your installation, and paste the contents of `deployment/b200/AGENT_PROMPT.txt`.
Do not import private credentials or the original OpenCode database. The
repository's `AGENTS.md` points to the full scientific/operational handoff.

If no agent is allowed, use the commands above; the queue, monitoring, logging
and result uploads operate without one. Automated monitoring is not an AI
repair service, and no worker is automatically restarted after failure.
