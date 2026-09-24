# Deadline-friendly memory-game experiments

`memory_games_sac.py` uses the existing gymnax/JAX learner for two independent
implementations of POPGym games. No additional environment package is needed.

## Parallel cluster pilot

From the repository root on the cluster (with these new files present):

```bash
mkdir -p /project/home/p201442/u104493/slurm-logs
sbatch examples/lambda-envs/memory_games_pilot.slrm
```

This submits **two array tasks, four GPUs each**: Concentration (52 cards, 13
ranks) and Minesweeper (6x6, six mines). Each GPU trains three seeds together for
100k steps. The four conditions have matched seeds and hyperparameters:

1. `sac`: no lambda branches.
2. `critics`: both lambda value-regression losses, discrepancy coefficient zero.
3. `ld0.01`: same critics plus discrepancy coefficient 0.01.
4. `ld0.1`: same critics plus discrepancy coefficient 0.1.

For four available GPUs, use `sbatch --array=0-1%1 ...` to run the games in two
waves. `--array=0` selects Concentration alone; `--array=1` selects Minesweeper.

Outputs: `/project/home/p201442/$USER/runs/memory-games-<job-id>/<env>/<condition>/`.
W&B project: `offline-lambda-memory-games-results`. Every round logs locally too.
The Slurm working directory is project storage. W&B files, temporary files,
CUDA/JAX/Matplotlib caches and W&B settings are explicitly redirected beneath
the run directory; Python bytecode writes are disabled to avoid creating cache
files in the home checkout or its environment.
`ALPHA=0.003` or `ALPHA=0.03` can test exploration sensitivity with the same
four conditions. The default is 0.01; the games have normalized reward scales.

## Longer run and resume

Run ten seeds per condition for 1M steps, using a fresh matched seed set:

```bash
sbatch --time=12:00:00 \
  --export=ALL,ROUNDS=200,NUM_SEEDS=10,BASE_SEED=100,EVAL_EPISODES=128,CHECKPOINT_EVERY=10 \
  examples/lambda-envs/memory_games_pilot.slrm
```

The script defaults to a six-hour allocation for the pilot; increase it explicitly
for longer runs. `ROUNDS=400` sets 2M steps. A longer configured budget is not an
assertion of convergence: inspect the late learning curves and per-seed outcomes.

Continue the **same** pilot seeds to 1M, retaining replay and optimizer state:

```bash
sbatch --time=12:00:00 \
  --export=ALL,ROUNDS=200,RESUME_ROOT=/project/home/p201442/$USER/runs/memory-games-123456 \
  examples/lambda-envs/memory_games_pilot.slrm
```

Replace `123456` with the pilot array job ID. Keep its original seed count,
base seed, alpha and training configuration. `--rounds` is the total target,
not an additional number. The runner checks configuration compatibility.
Resume writes to the new job directory, so the source checkpoint is preserved.

Checkpoints include agent/target parameters, optimizer states, replay, RNG,
environment state and recurrent carry. Atomic replacement retains only the latest
checkpoint. With 10 seeds, Concentration checkpoints are several GB: the default
pilot saves every 25k steps; the long-run command above saves every 50k.

## First pilot and diagnostic follow-ups

Pilot `5251683` completed on 2026-09-24, using three matched seeds and 100k
training steps per condition. Mean greedy returns over the final five
evaluations (80k–100k), with across-seed standard errors:

| Condition | Concentration | Minesweeper |
|---|---:|---:|
| SAC | -0.940 ± 0.002 | -0.404 ± 0.010 |
| Critics only | -0.957 ± 0.002 | -0.399 ± 0.003 |
| LD 0.01 | -0.985 ± 0.003 | -0.369 ± 0.016 |
| LD 0.1 | -0.993 ± 0.003 | -0.311 ± 0.003 |

Minesweeper LD 0.1 beat both controls in all three seeds, with known-safe choice
rate 34% vs SAC's 17%. Its sampled return also improved (-0.345 vs -0.399).
No condition cleared a board; the history-only scripted reference scored -0.058.
Policy entropy differed strongly (2.60 vs 0.52 nats), motivating baseline tuning
before attributing the gap specifically to memory. Concentration was below its
random reference (-0.819), with badly drifting critics, so it needs stability
and history-exposed controls before a long run. These are pilot findings, not
convergence results.

The four Minesweeper checkpoints are being continued to 1M steps (job
`5251757`, output root `memory-games-5251757`, three original seeds, 128 eval
episodes every 10k steps). Short fresh-initialization checks run with:

```bash
sbatch examples/lambda-envs/memory_games_followup.slrm
```

This is a **three-task array capped at one active task**, four GPUs per task:

| Task | Game | Four conditions |
|---|---|---|
| 0 | Minesweeper | SAC / critics-only × α={0.03, 0.1}; FE lr=1e-4 |
| 1 | Minesweeper | SAC / critics-only × α={0.01, 0.03}; FE lr=1e-5 |
| 2 | Concentration | partial GRU / history-exposed identity SAC × α={0.01, 0.03}; FE lr=1e-5 |

All run three seeds for 100k steps, using the original tuning seed set and
unchanged head learning rates (1e-4). They save resumable checkpoints. Default
evaluation budget is 64 episodes; all runtime artifacts are under project
storage at `runs/memory-games-followup-<job-id>/<env>/<cohort>/<condition>`.
Together with the four-GPU continuation, this keeps peak usage at eight GPUs.
Use `--array=0-1%1` for the Minesweeper checks only or `--array=2` for Concentration.
Select settings using these tuning runs, then use fresh seeds for confirmation.

### Targeted LD comparison on the improved baseline configuration

The completed lower-FE-rate pilots reached -0.327 (SAC) and -0.315
(critics-only) at 100k, using FE lr=1e-5 and α=0.01. The latter is close to the
original LD 0.1 result (-0.311). Apply LD to this better baseline configuration
before interpreting the original gap as a discrepancy-specific improvement.

Optional task **3** of the follow-up launcher compares:

| Display label | Method | Discrepancy weight | Initialization |
|---|---|---:|---|
| `baseline` | SAC | 0 | Resume the low-FE-rate SAC checkpoint |
| `extra-critics` | Critics only | 0 | Resume the low-FE-rate critics checkpoint |
| `LD-small` | LD | 0.03 | Fresh |
| `LD-large` | LD | 0.1 | Fresh |

All use **FE lr=1e-5, α=0.01, three matched seeds, and 500k total steps**.
Compare equal training-step windows; the resumed controls begin at 100k, while
the LD runs begin at zero. Their first 100k baseline histories remain in the
source job. The W&B group includes `matched-ld`, making these four lines easy
to filter together. This task is excluded from the default three-wave array.

```bash
sbatch --array=3 --dependency=afterany:5251797 \
  --export=ALL,ROUNDS=100,BASELINE_ROOT=/project/home/p201442/$USER/runs/memory-games-followup-5251797/minesweeper/low-fe-lr \
  examples/lambda-envs/memory_games_followup.slrm
```

The dependency lets the Concentration diagnostic finish first, keeping total
usage at eight GPUs while the original 1M Minesweeper comparison runs. Expected
training time is about 45 minutes after allocation, plus startup/evaluation.

### Continue all four checkpoints and the same W&B curves

`finished` means a process reached its configured step budget, not that learning
necessarily converged. The matched baselines reached 500k before the new LD runs
because they resumed at 100k; the plain baseline is also faster per update.

Optional task **4** resumes every matched condition to **1M total steps**.
Set `MATCHED_RESUME_ROOT` to the completed cohort directory. `WANDB_RUN_IDS`
contains IDs in the order baseline, extra-critics, LD-small, LD-large. Keep the
old W&B group as well. Set the comma-containing ID list in the shell environment
so Slurm does not split it as separate `--export` entries:

```bash
export WANDB_RUN_IDS=c0v11yvi,rt4nayvx,ykga1pca,j8mkrf1g
sbatch --array=4 --job-name=minesweeper-matched-1M \
  --export=ALL,ROUNDS=200,CHECKPOINT_EVERY=10,MATCHED_RESUME_ROOT=/project/home/p201442/$USER/runs/memory-games-followup-5251952/minesweeper/matched-ld,WANDB_GROUP_OVERRIDE=memory-games-minesweeper-matched-ld-5251952 \
  examples/lambda-envs/memory_games_followup.slrm
```

Wait for the source job's writers to finish (or use `--dependency=afterok:5251952`)
before running this command. Only one process may write each W&B run at a time.
Training resumes from checkpoint state, while `WANDB_RESUME=must` attaches logging
to the existing run IDs. The runner updates the larger budget in W&B's config
without resetting its history. New local artifacts go to the new job's project
directory; source checkpoints and summaries remain available.

## Runtime measurement

Measured on an RTX 3090, JAX 0.7.1 / Flax 0.11.1, with the actual default model:
128-unit GRU, 128-wide projection and heads, batch 32, sequence 32, trailing
look-ahead 32, full-episode burn-in (104 / 30). Measurements use `method=ld`,
500-update rounds, three rounds, and synchronize the GPU before timing.

| Game | Concurrent seeds | Updates/s **per seed** | 100k steps | 1M steps |
|---|---:|---:|---:|---:|
| Concentration | 3 | 127 | 13 min | 2.2 h |
| Minesweeper | 3 | 169 | 10 min | 1.6 h |
| Concentration | 10 | 55 | 30 min | 5.1 h |
| Minesweeper | 10 | 76 | 22 min | 3.7 h |

These are **training-only projections for an entire seed group on one GPU**, not
times to multiply by the seed count. Compilation was about 18 seconds per
configuration. Evaluation, checkpoints, cluster performance and queueing add
overhead. The runner prints and logs updated projections on the actual device.
Eight GPUs run all eight configurations concurrently; four GPUs require waves.

Reproduce a short benchmark:

```bash
python examples/lambda-envs/memory_games_sac.py \
  --env concentration --method ld --rounds 3 --train-steps 500 \
  --num-seeds 3 --eval-episodes 32 --checkpoint-every 0 \
  --output-dir /tmp/concentration-timing
```

Switch `--env minesweeper` for the other game. Use a new output directory for
each fresh run. Short benchmark runs establish throughput and finite updates,
**not** learning success or a discrepancy advantage.

## What to inspect

- `agg/return/mean`: greedy evaluation return. `eval/sampled/return/mean` logs
  the stochastic policy as well; early greedy policies can repeatedly select
  one cell/card, so inspect both.
- `agg/progress/mean`: fraction of card pairs matched / safe cells revealed.
- `agg/known_choice_rate/mean`: matches selected when a matching location was
  already seen; or unviewed safe neighbors chosen when a zero clue guarantees
  one exists. This is undefined when there were no opportunities, not zero.
- `agg/opportunities/mean` and `agg/invalid_fraction/mean` distinguish missing
  opportunities, forgetting, and repeated invalid actions.
- Random and history-only scripted reference policies are evaluated at startup.
  They read no unobserved labels/mines and are reference policies, not optimal
  upper bounds.
- `env_interactions` **includes random prefill**; `train_env_steps` counts
  subsequent one-update-per-interaction learning. The same prefill budget and
  sequence windows are used across all four ablations.

For a history-exposed control, add `--remember --memory-type identity` to a
standalone `--method sac` run. It exposes only previously observed information,
not the private deck/mine board. A failure even with this control suggests
action-value/inference/exploration difficulties rather than only memory learning.

## Environment and replay details

- Concentration matches the original rank-matching rules: 52 positions with
  four cards per rank, a 104-action cap, +1/26 for a pair, -1/104 per failed
  attempted flip. Successful pairs stay face up. Failed pairs remain visible
  until the next action; the transient observation is cached in the state.
- Minesweeper has one clue per action, no flood fill and no first-click
  protection. It rewards new safe cells, penalizes repeats, and terminates on
  a mine, completion or the 30-action limit. Its clue one-hot has all possible
  counts plus a distinct reset marker. This reset marker is an explicit minor
  observation difference from original POPGym's ambiguous reset zero.
- Neither game masks actions: invalid choices receive the reference penalties.
- Previous executed action is part of the stored observation; its adapter and
  the carried recurrent state preserve history across training rounds. Reset
  occurs at actual episode boundaries. It enters the projection, not the memory
  carry, with no duplicated previous-action input.
- Full-episode burn-in is the default. Shortening it can remove clues from a
  replay window and change what the memory-learning objective can train.

Sources:
[Concentration](https://github.com/proroklab/popgym/blob/master/popgym/envs/concentration.py),
[Minesweeper](https://github.com/proroklab/popgym/blob/master/popgym/envs/minesweeper.py).

Checks:

```bash
JAX_PLATFORMS=cpu mamba run -n lambda pytest \
  tests/test_concentration_env.py tests/test_minesweeper_env.py \
  tests/test_memory_games_runner.py -q
bash -n examples/lambda-envs/memory_games_pilot.slrm
```
