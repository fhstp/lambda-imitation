# Online memory-learning experiments

Run from the repository root after `pip install -e ".[experiments]"`.
The two entry points share the same training, evaluation, checkpoint and
optional probe implementation. `--help` lists the complete supported interface;
unknown flags and unknown sweep parameters are errors.

## Methods

- `--method ac`: entropy-regularised actor plus task critic.
- `--method critics`: add both auxiliary Retrace critics, with discrepancy
  coefficient zero.
- `--method ld`: add both critics and the squared discrepancy objective.

Auxiliary regression uses **Huber loss with threshold 1**. The task critic and
discrepancy use squared error. Uniform replay, reward-only critic backups,
zero-start burn-in, and stopped actor gradients into recurrent memory are fixed
parts of the protocol. The identity-memory control retains actor gradients into
its feed-forward embedding.

## Battleship: paper configurations

The default task is 5×5 with ships of lengths three and two. Only the latest
shot's hit/miss bit and previous action reach the network. A separate action
mask prevents firing twice at the same cell. Clearing the board in N shots
gives return `26 - N`.

These commands reproduce the three selected off-policy **configurations**.
Each runs ten seeds with the recorded base seed plus 0 through 9:

```bash
# Recurrent AC + Retrace discrepancy
python examples/battleship.py --method ld --num-seeds 10 --seed 7451238 \
  --rounds 50 --train-steps 10000 --memory-hidden-dim 512 \
  --fe-lr 1e-5 --actor-lr 1e-4 --critic-lr 1e-4 \
  --batch-size 128 --burn-in-length 10 --sequence-length 40 \
  --lambda-truncation 50 --lambda1 0.05 --lambda2 0.85 --lambda-coef 0.01 \
  --eval-episodes 50 --output-dir outputs/battleship-ld

# Recurrent AC baseline
python examples/battleship.py --method ac --num-seeds 10 --seed 842482347 \
  --rounds 50 --train-steps 10000 --memory-hidden-dim 512 \
  --fe-lr 1e-5 --batch-size 128 --burn-in-length 15 --sequence-length 28 \
  --eval-episodes 50 --output-dir outputs/battleship-ac

# Memoryless AC control
python examples/battleship.py --method ac --memory-type identity \
  --num-seeds 10 --seed 42 --rounds 50 --train-steps 10000 \
  --fe-lr 1e-4 --burn-in-length 10 --sequence-length 50 \
  --eval-episodes 50 --output-dir outputs/battleship-memoryless
```

All use γ=0.99, α=0.1, τ=0.005, replay capacity 100,000 and the
Allen et al. Battleship skip-connection embedding. Recurrent agents use a
512-unit GRU and one-hidden-layer heads of width 512, with critic LayerNorm.
The memoryless control substitutes a third embedding layer for the GRU.

| Configuration | Burn-in / learning / tail | Random prefill per seed |
|---|---|---:|
| Recurrent AC | 15 / 28 / 50 | 11,904 |
| Recurrent AC + discrepancy | 10 / 40 / 50 | 12,800 |
| Memoryless AC | 10 / 50 / 50 | 14,080 |

The paper analyses checkpoints through **500,000 total interactions**, including
prefill. Filter on `env_interactions <= 500000`; fifty 10,000-step rounds also
produce checkpoints beyond that analysis window. Original runs used a longer
training budget. Their result snapshots and PPO comparison/figure scripts are
in the companion paper repository. PPO is the Allen et al. reference
implementation, not a method implemented by this learner.

The shared runner preserves the learner's loss, architecture and replay
conventions, but reorganises RNG streams and checkpoint storage. It is not a
bitwise replay of legacy run trajectories. Historical checkpoint pickles are
not compatible with the publication schema.

`reference/reference/*` evaluates the Bayes-density heuristic using its own
shot history; `reference/random/*` evaluates a uniformly random legal policy.
The reference is evaluation-only. The paper's exact reference expectation is
computed by the paper repository's enumeration script; the runner reports a
sampled reference evaluation.

## Minesweeper

```bash
python examples/minesweeper.py --method ld --num-seeds 5 \
  --output-dir outputs/minesweeper-ld
python examples/minesweeper.py --method ac --num-seeds 5 \
  --output-dir outputs/minesweeper-ac
python examples/minesweeper.py --method critics --num-seeds 5 \
  --output-dir outputs/minesweeper-critics
```

The task is a partial 6×6 board with six mines. Actions reveal one square; there
is no flood fill, action masking or protected first move. Observations contain
only the latest clue, so previous-action input identifies its location.

Defaults: GRU/embedding/heads of width 128, batch 32, full-episode burn-in 30,
learning length 32, tail 32, α=0.01, λ=(0,0.95), discrepancy coefficient 0.01,
40 rounds × 5,000 interactions, and 3,008 random prefill transitions per seed.
The history-only reference selects unqueried neighbours of previously observed
zero clues, falling back to unqueried cells. It never consults unseen mines.
`known_safe_choice` and `repeat_fraction` report history-based decision
diagnostics. An evaluation with no known-safe opportunities records JSON null.

Minesweeper is an additional experiment, not yet a paper result.

### Matched sweeps

```bash
wandb sweep examples/sweeps/minesweeper_baseline.yaml
wandb sweep examples/sweeps/minesweeper_ld.yaml
EXPERIMENT_OUTPUT_DIR=/path/to/results wandb agent <entity/project/sweep-id>
```

Both searches use five fixed seeds, 200,000 post-prefill interactions per seed,
the same shared search spaces and 40-trial caps. The objective is
`final/return_smoothed/mean`: average each seed's final five greedy evaluations,
then average seeds. The discrepancy search additionally tunes its coefficient.
Every sweep run gets a separate `run_<id>` output subdirectory.

## Evaluation, logging and budgets

`--num-seeds` runs seeds concurrently using `jax.vmap`; choose a count that fits
your device memory. Replay and optimizer states are separate per seed.

Each evaluation records both **greedy and sampled** policy returns. Metrics
include each seed plus their mean and sample standard deviation. Training,
evaluation and optional probes use separate RNG streams. Training time is
device-synchronised; the first round's timing explicitly includes compilation.

- `train_env_steps`: online interactions after prefill, per seed.
- `env_interactions`: `prefill_steps + train_env_steps`, per seed.
- Evaluation, reference and probe rollouts are excluded from the training budget.
- By default, prefill size is `batch_size × (burn-in + learning + tail)`.
  `--prefill-steps` can override it and must fit the buffer.

All environments reset memory and previous-action input at episode boundaries.
For the recorded Battleship protocol, rollout history additionally
starts at zero at each training round; the environment state continues.
Minesweeper retains rollout history across rounds, matching its original
pilot protocol. These choices are fixed by the environment specification and
recorded in `config.json`.

Outputs:

```text
config.json                 resolved environment/training configuration
metrics.jsonl               per-round metrics and reference/probe evaluations
summary.json                final-window greedy-return summary
checkpoint_<round>.pkl      optional periodic resumable checkpoints
checkpoint_final.pkl        final resumable checkpoint
probe_<round>/              optional per-seed datasets/predictions and figures
```

Add `--wandb` to log the same metrics; `--wandb-project` and
`--wandb-run-name` select their destination. Local outputs are always written.
The final summary averages available final evaluations if fewer than
`--final-return-window` were recorded; it does not interpolate missing points.

## Checkpoint continuation

```bash
python examples/battleship.py --rounds 100 \
  --resume-from outputs/battleship-ld/checkpoint_final.pkl \
  --output-dir outputs/battleship-continued \
  --method ld --num-seeds 10 --seed 7451238
```

Supply the same training settings as the saved run, including any nondefault
ones. `--rounds` is the total target budget. Checkpoints contain networks,
targets, optimizer states, replay, training RNG, environment state, memory,
previous actions and metric history. Incompatible training configurations are
rejected; evaluation/probe cadence, logging, output location and total rounds
may change. Resuming restores metric history from the checkpoint.

## Optional memory decoding

Add `--probe` for final decoding and `--probe-every N` for periodic decoding.
Configure its budget with `--probe-collect-steps`, `--probe-steps`,
`--probe-hidden-dim`, `--probe-batch-size`, and `--probe-lr`.

Probes fit a two-hidden-layer MLP on independent sampled-policy training and
test rollouts. They never train the agent or provide privileged input to it.

- **Battleship:** binary ship occupancy. Observed-cell scores measure retention;
  unobserved-cell scores measure inference.
- **Minesweeper:** categorical neighbour counts. Observed-cell scores measure
  clue retention; unobserved-cell scores measure clue inference. This is not a
  mine-location oracle for the policy.

Outputs include balanced accuracy, per-class recalls, errors per state, exact
map accuracy and cross-entropy in bits; binary probes also report tie-corrected
AUROC. Missing-class statistics are null. Recency buckets use time since first
observation. Figures outline observed cells rather than replacing
predictions with known truth. Probe scores should be interpreted alongside
class balance and untrained-memory controls, not raw accuracy alone.
