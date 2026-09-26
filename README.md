# Retracing the λ-Discrepancy

JAX implementation for **Retracing the λ-Discrepancy: Memory Learning from
Off-Policy Data**. The code learns recurrent representations from replay using
the disagreement between two Retrace value estimates as an auxiliary loss.

The publication implementation supports **discrete actions**, a recurrent
actor–critic baseline, auxiliary-critic regression, and Retrace discrepancy.
Battleship, Minesweeper, and the paper's T-maze reproduction code ship
in this repository.

## Installation

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[experiments,dev]"
```

For GPU training, install the appropriate [JAX accelerator
build](https://docs.jax.dev/en/latest/installation.html) for your machine.
The package itself is still installed/imported as `lambda-imitation` /
`lambda_imitation`.

Core learning needs JAX, Flax NNX, Optax and NumPy. The `experiments` extra adds
Gymnax, Matplotlib and optional W&B logging.
No separate `lambda-envs` checkout is needed. A smaller installation for
Battleship/Minesweeper without plotting is `pip install -e ".[gymnax]"`.

## Quick start

```bash
# Small end-to-end smoke run, including evaluation and a resumable checkpoint.
python examples/battleship.py \
  --rounds 1 --train-steps 10 --memory-hidden-dim 16 \
  --batch-size 4 --burn-in-length 2 --sequence-length 4 \
  --lambda-truncation 4 --online-buffer-size 1000 \
  --eval-episodes 4 --output-dir outputs/battleship-smoke

# The two main runners share a CLI.
python examples/battleship.py --help
python examples/minesweeper.py --help
```

See **[examples/README.md](examples/README.md)** for the paper's Battleship
configurations, the other environments, multi-seed runs, checkpoints, probes
and sweeps. **[examples/tmaze/README.md](examples/tmaze/README.md)** covers
exact detection, finite-data critics, observation-channel controls and audits.

## Learning objective

All heads share a feature extractor: an observation/previous-action projection
followed by an identity, RNN, GRU or LSTM memory. The task and auxiliary critics
are independent twin pairs.

- The **actor** maximises expected task Q plus `alpha × entropy`. Its gradient
  stops at recurrent memory; the memoryless control trains its embedding.
- The **task critic** fits a reward-only, one-step Bellman target with MSE.
- Each **auxiliary critic** fits a detached Retrace target using **Huber
  regression, threshold 1**. Both twins regress separately; their minimum is
  used for bootstrapping.
- The **discrepancy** is the mean squared difference between the two auxiliary
  twin-min values at replayed actions. Its gradient trains the feature
  extractor, with auxiliary-head parameters held fixed.

Retrace uses `c[t] = lambda × min(1, pi[t] / mu[t])`, with logged collection
probabilities in the denominator and the EMA target actor/feature extractor in
the numerator. The current TD residual always has coefficient one. All target
networks are updated by EMA.

Replay is uniform over contiguous windows. Burn-in starts from zero, is
detached before the learning unroll, and resets at episode boundaries. A
look-ahead tail is excluded from auxiliary regression. As in the reported
implementation, actor learning uses the entire post-burn-in window, while task
regression and discrepancy exclude only its last step. These conventions are
documented in `Hyperparameters` and tested independently of the experiment CLI.

| `--method` | Task critic | Auxiliary regression | Discrepancy |
|---|---|---|---|
| `ac` | MSE | — | — |
| `critics` | MSE | Huber | zero coefficient |
| `ld` | MSE | Huber | `--lambda-coef` |

## Code layout

```text
src/lambda_imitation/
    actor_critic.py       # functional learner, networks, Retrace targets
    buffer.py             # immutable circular replay and uniform sampling
    utils.py              # projections, recurrent cells and Gymnax factory
    envs/                 # bundled Battleship, Minesweeper and T-maze
examples/
    battleship.py
    minesweeper.py
    _common.py            # shared training/evaluation/checkpoint runner
    _probes.py            # optional post-hoc memory decoding
    sweeps/               # matched Minesweeper searches
    tmaze/                # paper reproduction and numerical audits
tests/
```

`create_actor_critic_from_env(EnvSpec(...), ...)` returns `(state, functions)`;
there is no required dataset or placeholder buffer. `functions.predict`
accepts a memory carry and previous-action encoding separately.
`functions.prefill_buffer` collects uniform-random data, and
`functions.train_unrolled` performs online collection and updates while
returning both rollout-history components. The functions are compatible with
`jax.jit` and `jax.vmap`. See their docstrings for signatures.

## Verification

```bash
pytest tests/
python -m build
```

Tests cover independent Retrace calculations, Huber regression and gradient
routing, recurrent/previous-action semantics, action masks, environment rules,
exact checkpoint continuation, and the T-maze population/empirical operators.

## Attribution

The bundled Battleship and T-maze environments and the reference
network architectures build on **Allen et al. (NeurIPS 2024),
*Mitigating Partial Observability in Sequential Decision Processes via the
Lambda Discrepancy***:
<https://github.com/brownirl/lambda_discrepancy>.
Minesweeper independently implements the POPGym task rules.

See **[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)** for source revisions,
modifications, authors and the included Apache-2.0 license, and the
[environment README](src/lambda_imitation/envs/README.md) for task semantics.
