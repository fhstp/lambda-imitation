# Full POPGym benchmark on the 8×B200 host

One command trains every discrete POPGym task, logs to a new W&B project and
serves a live dashboard that puts the results next to the literature and our
earlier runs. The Battleship Bayes sweep files are untouched. This is a new
study with its own entry point.

## What runs

| | |
|---|---|
| Tasks | All 42 discrete POPGym tasks (14 families × Easy/Medium/Hard). Pendulums are continuous and excluded. |
| Environments | 24 audited POPJym JAX ports. NEW JAX ports of Autoencode, CountRecall, MineSweeper and VelocityOnlyCartPole (`examples/popgym_bench_envs.py`), checked step by step against NumPy POPGym 1.0.7 (`tests/test_popgym_bench_envs.py`). Labyrinth Escape/Explore use the original Gymnasium envs on the host, with acting/replay/updates still vmapped on the GPU. |
| Agent | FlashSAC-style residual RMS heads (actor 128, critics 256, 2 expansion-4 blocks), critic OFN, all six critics 51-atom categorical on [-1.5, 1.5] (dense deterministic projection), GRU512 + 128-unit prev-action projection, complete-episode BPTT, reward-only projected main law, two-hot scalar-Retrace λ heads, detached-head scalar λ-discrepancy, actor-to-FE stopped. |
| Settings | flash-combined defaults: β=10, all LRs 1e-4, α 1→.01 geometric over 500k then fixed, τ=.005, γ=.99, λ1=.05, **λ2=.99**, replay 100k, 12.8k random prefill to a natural episode boundary. |
| Batch | Constant ~8.3k transition slots per update: `max(8, 8320 // (H+1))` complete episodes (Battleship Easy keeps 128×65). |
| Exploration | **Correlated**: unit-Gaussian logit bias held for k decisions, P(k)∝k⁻², k≤16, fresh categorical draw every step. The exact conditional behaviour probability softmax(logits+bias)[a] is stored for Retrace. Bias resets at real episode ends, with its own RNG. Evaluations use the unbiased actor. |
| Seeds | 3 per task (42–44), genuinely `jax.vmap`-batched: training, prefill, replay, optimizer, rollout and RNG are per lane. |
| Budget | 1M online steps per task. Evaluate every 20k (50 stochastic + 50 greedy episodes per seed). Final holdout: 200 episodes per seed × stochastic/greedy/critic-greedy/random. |
| Arms | `primary` (β=10) for all 42 tasks first. Only after **every** primary task has finished: `control` (β=0, otherwise identical). |

## Setup (separate checkout, so running sweep agents keep their sources)

```bash
git fetch origin
git worktree add ../popgym-b200-bench origin/feat/popgym-b200-bench   # or a fresh clone
cd ../popgym-b200-bench
PYTHON=python3.11 bash scripts/setup_b200.sh      # adds popgym[navigation] + cython for Labyrinth
source .venv/bin/activate
wandb login
```

The GPUs must be free: stop/pause the Battleship sweep agents first, if they
are still running.

## Run (one command)

```bash
tmux new -s popgym
source .venv/bin/activate
python examples/popgym_bench.py --output /SCRATCH/popgym-bench
```

The supervisor does the following:

1. Writes `study.json`, then starts the dashboard on `0.0.0.0:8765`. Open
   http://<b200-vpn-address>:8765 (the host is only reachable via the VPN).
   Use `--dashboard-host 127.0.0.1` to restrict it to the machine itself.
2. Runs the launch gate on GPU 0 (`validation.json`): full-size same-input
   repeatability, seed-lane isolation and host round-trip resume, exact bytes,
   on Battleship Easy and Autoencode Easy (~5 min). On failure nothing is launched.
3. Starts one worker per GPU (`CUDA_VISIBLE_DEVICES`), longest expected task
   first. Two workers per GPU were measured to be *slower* in aggregate.
4. Starts the β=0 control arm only once no primary task is pending or running.

W&B: entity `fhstp-data-intelligence-research-group` (or `$WANDB_ENTITY`),
project `popgym-b200-flash-bench` (or `$WANDB_PROJECT`). There is one run per
task × arm, grouped by task, tagged `primary`/`control`, with per-seed returns.
A W&B outage never stops training; errors go to the task's `wandb-errors.log`.

**Restarting** the same command resumes everything: complete tasks are skipped,
and unfinished ones continue from their last checkpoint (every 100k steps,
including replay, optimizer and RNG; Labyrinth also pickles its host envs). A
crashed worker is resumed automatically once (`--max-resumes`). Use a new
`--output` for any change of settings. The config is pinned in `study.json`.

Useful flags: `--gpus 0 1 2 …`, `--envs <ids>`, `--skip-control`,
`--dashboard-port 0`, `--no-wandb`. Standalone dashboard:
`python examples/popgym_bench.py dashboard --output /SCRATCH/popgym-bench`.

## Expected time (measured on local H100s, B200 not yet measured)

Per task, 1M steps × 3 seeds: CartPole ≈ 11 h, Battleship Easy ≈ 16 h,
Battleship Hard ≈ 25 h on one H100. The total is ≈ 650 H100-GPU-hours for the
primary arm, so on 8 GPUs ≈ 3.4 H100-days. That is ~1.4–2.3 days on B200 at
an assumed 1.5–2.5× speed-up, and the control arm needs the same again. The
dashboard shows the live estimate after the first evaluations. The run is
compute-bound (fp32 native precision, unchanged).

## Outputs

`OUTPUT/{primary,control}/<env_id>/`: `status.json`, `history.json` (all
evaluations, per-seed values), `config.json` (config, task metadata, source
hashes, versions), `latest.pkl`, `wandb-errors.log`. Also
`OUTPUT/scheduler.json`, `OUTPUT/validation.json`, and `OUTPUT/logs/` (one log
per worker).

## Dashboard from W&B (any other machine)

The B200 host is hard to reach, so the same page can be rebuilt from W&B alone:

```bash
python examples/popgym_bench_wandb.py --web-dir /path/served/popgym-bench --watch 120
```

It writes a static `index.html` + `data.json` into the web directory every 2
minutes. Workers log per-seed returns, an explicit `run_state`, and a
`supervisor` run carrying the launch gate and scheduler state, so the page matches
the on-host dashboard. On the development host it runs as
`lambda-popgym-bench-dashboard.service` (user unit), published at
http://10.203.222.228:8000/flash-combined/popgym-bench/.

## Comparing with other work (dashboard)

References are in `examples/popgym_bench_references.json`:

- POPGym PPO best of 13 models (Morad et al. 2023)
- POPJym JAX PPO (Lu et al. 2023)
- our earlier weekend suite (SAC and SAC+λ, 250k; many tasks have only 1 seed)
- the best-α runs of our α follow-up (5 tasks)
- the Battleship Bayes sweep best

The normalized score is 0 = random, 1 = best published PPO. "Best during
training" is an MMER analogue (per seed, max over evaluations of the mean
return), not PPO's training-episode MMER. Literature used original NumPy POPGym
and 15M steps. Task versions differ for POPJym ports (e.g. Battleship's 1-bit
observation). Treat comparisons as context, not a controlled benchmark claim.
