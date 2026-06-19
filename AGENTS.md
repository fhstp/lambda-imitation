# AGENTS.md

## Overview

IQ-Learn (Inverse Q-Learning) imitation learning implementation in **Python** using **JAX + Flax NNX + Optax**. Currently on the `fresh_start` branch. The project is structured as an installable Python package (`lambda-imitation`) with a `src/` layout.

## Repository layout

```
src/
    lambda_imitation/
        __init__.py       ← public API re-exports
        buffer.py         ← generic circular replay buffer
        iqlearn.py        ← SAC-style actor-critic with IQ-Learn reward recovery
        utils.py          ← environment-interface adapters and high-level factory
tests/
    __init__.py
    test_buffer.py        ← 26 tests
    test_iqlearn.py       ← 90 tests
    test_utils.py         ← 45 tests
examples/
    mountain_car_discrete.py     ← MountainCar-v0 demo (discrete, gymnasium, IQ-Learn)
    mountain_car_continuous.py   ← MountainCarContinuous-v0 demo (continuous, gymnasium, IQ-Learn)
    cartpole_sac.py              ← CartPole-v1 demo (discrete, gymnax, pure SAC)
    pendulum_sac.py              ← Pendulum-v1 demo (continuous, gymnax, pure SAC)
pyproject.toml            ← hatchling build, optional extras, pytest config
```

## Architecture

- `buffer.py` -- Generic circular replay buffer using JAX arrays and NamedTuples
- `iqlearn.py` -- SAC-style actor-critic with IQ-Learn reward recovery (configurable networks, continuous and discrete actions)
- `utils.py` -- Environment-interface adapters: extracts `EnvSpec` from gymnasium / gymnax / jumanji environments, plus `create_iqlearn_from_env` high-level factory

**Design pattern:** Purely functional. Both modules use a factory pattern (`create_buffer`, `create_iqlearn`) that returns `(state_namedtuple, functions_namedtuple)`. No mutable class state. Flax NNX models are split into graph definition + state via `nnx.split`/`nnx.merge` for functional updates inside `jax.jit`.

### Network design (iqlearn.py)

Networks are split into a **feature extractor** and a **head**:

- `MLPFeatureExtractor(input_dim, hidden_dims, rngs)` -- configurable MLP backbone; flattens input, applies ReLU layers. `hidden_dims=(256, 256)` by default.
- `Head(feature_dim, hidden_dims, output_dim, *, rngs)` -- generic head: ReLU on hidden layers, linear output. No flattening. Used for all four roles:
  - continuous actor: `output_dim = 2 * action_dim` (mean + log_std)
  - discrete actor: `output_dim = num_actions` (categorical logits)
  - continuous critic Q1/Q2: `output_dim = 1` (actions concatenated to features before the head)
  - discrete critic Q1/Q2: `output_dim = num_actions` (per-action Q-values)

FE + head state are stored together in a `NetworkState(fe, head)` NamedTuple. The twin-Q critic uses `TwinCriticState(q1: NetworkState, q2: NetworkState)` — both branches are fully independent (separate FE and head) and a single optimizer operates on the whole pytree.

`create_iqlearn` takes three separate FE instances (`actor_feature_extractor`, `critic_q1_feature_extractor`, `critic_q2_feature_extractor`). It infers each feature dim by running a dummy forward pass before splitting. Use plain Python `list` (not `nnx.List`) for collections of sub-modules inside an `nnx.Module` — `nnx.List` was removed in Flax 0.10.7.

Graph definitions are stored in `NetworkGraphs(fe, head)` NamedTuples, grouped into `IQLearnGraphs(actor, critic_q1, critic_q2)`.

### Action space support

Both **continuous** (Box-style) and **discrete** (categorical) action spaces are supported. Pass `is_discrete=True` to `create_iqlearn` / `create_iqlearn_from_env` to activate the discrete path:

- Discrete actor: categorical policy; `predict` returns a `float32` scalar action index (e.g. `0.0`, `1.0`).
- Discrete critic: all-actions Q-values; V(s) computed as exact `Σ_a π(a|s)·Q(s,a)` without sampling.
- Default `target_entropy` for discrete: `0.98 * log(num_actions)` (Christodoulou 2019).
- `MultiDiscreteArray` (jumanji) is not supported — raises `ValueError`.

## Dependencies

Declared in `pyproject.toml`. Install with:

```
pip install -e ".[dev]"          # core + pytest
pip install -e ".[dev,gymnasium]"  # add gymnasium support
```

Core: `jax>=0.4.30`, `jaxlib>=0.4.30`, `flax>=0.9.0`, `optax>=0.2.0`, `numpy>=1.26`  
Optional extras: `[gymnasium]`, `[gymnax]`, `[jumanji]`, `[dev]` (pytest)  
Requires Python 3.10+ (uses `X | Y` union type syntax).

## Commands

- Install: `pip install -e ".[dev]"`
- Run all tests: `pytest tests/`
- Run buffer tests only: `pytest tests/test_buffer.py`
- Run iqlearn tests only: `pytest tests/test_iqlearn.py`
- Run utils tests only: `pytest tests/test_utils.py`
- Record demos (discrete): `python examples/mountain_car_discrete.py --record`
- Train + visualise (discrete): `python examples/mountain_car_discrete.py`
- Record demos (continuous): `python examples/mountain_car_continuous.py --record`
- Train + visualise (continuous): `python examples/mountain_car_continuous.py`
- Train + visualise CartPole SAC (gymnax): `python examples/cartpole_sac.py`
- Train + visualise Pendulum SAC (gymnax): `python examples/pendulum_sac.py`
- Train PocMan SAC+λ (lambda-envs): `python examples/lambda-envs/pocman_sac_mc.py`
- Train Battleship SAC+λ, action-masked (lambda-envs): `python examples/lambda-envs/battleship_sac_mc.py`
- Pellet probe + visualisation (PocMan): `python examples/lambda-envs/pocman_pellet_probe.py`
- Board probe + visualisation (Battleship): `python examples/lambda-envs/battleship_board_probe.py`

## Conventions

- All state is immutable NamedTuples; updates return new instances.
- Buffer entries are keyed by string names in `info: dict[str, jax.Array]`, not fixed fields.
- Hyperparameters are a NamedTuple with defaults, not a dataclass.
- `jax.lax.scan` is used for multi-step training loops; avoid Python for-loops inside JIT-compiled code.
- Actor/Critic FEs flatten input via `x.reshape(x.shape[0], -1)` to handle any obs shape.
- `utils.py` extractors do lazy imports (each library only imported when its extractor is called); only the library you use needs to be installed.
- `utils.py` supports both continuous (Box-style) and discrete action spaces; discrete actions in expert data are stored as float32 indices of shape `(N, 1)`.
- Mocking gymnax/jumanji in tests: use `ModuleType` objects (not `MagicMock`) for the full parent-module chain and wire `.spaces`/`.specs` attributes explicitly, so CPython's attribute-traversal in dotted imports resolves to the mock classes rather than auto-generated MagicMock attributes.
- All imports within the package use relative imports (`from .buffer import ...`).
- The twin-Q `jnp.min` gradient flow: with continuous actions, only the branch producing the smaller Q receives gradient in a given step. Tests for per-branch parameter changes therefore check the full `TwinCriticState` pytree (not individual branches), and verify branch independence by checking initial parameter divergence rather than per-step updates.
- **Gymnax auto-reset**: the gymnax base class `Environment.step()` always auto-resets — it runs `reset_env` on every call and returns the reset state/obs when `done=True` (via `jax.lax.select`). Do NOT add a manual `env.reset` call after `env.step` in gymnax code; `run_env_step` in `iqlearn.py` relies on this. Mock gymnax environments used in SAC tests must replicate this behaviour in their `step()` method.
- **Simulating absent installed packages in tests**: deleting a package from `sys.modules` is not sufficient when the package is actually installed — Python will re-import it from disk. Use `patch.dict(sys.modules, {key: None, ...})` (setting entries to `None`) to block re-import and trigger `ImportError`, as `None` entries are treated as import blockers by the import machinery.
- **`fns.train_sac()` metric keys**: `"q"`, `"entropy"`, `"v"`, `"critic_loss"`, `"target_q"`, `"alpha"`. Note the SAC critic loss key is `"critic_loss"` (same as in `update_step`), not `"sac_critic_loss"`.
- **Pure SAC examples with `create_iqlearn_from_env`**: when only `fns.train_sac()` is used (no imitation learning), a placeholder expert buffer is required. Use 1 all-zeros transition with `buffer_size=1` and `batch_size=1` in `Hyperparameters` — this is structurally valid and never sampled.
- **Prev-action input (`use_prev_action`)**: the FE call convention is `feature_extractor(carry, obs, prev_action=None) -> (new_carry, y)`. The carry is the **memory state only** — the prev-action is *not* packed into it. When `create_iqlearn(_from_env)` is built with `use_prev_action=True` the previously executed action (one-hot for discrete, squashed action values for continuous) is fed to the **projection** as an explicit input and threaded *next to* the carry: `run_env_step`/`train_unrolled` thread an `env_prev_action` (internally, so their public signatures are unchanged), `calculate_latent` threads `enc(a_{t-1})` in its local scan carry, and both reset to zero at episode boundaries. The default `LinearProjection` reproduces the old behaviour exactly (concat `[flatten(obs), prev_action]` then one `Linear` — matching `ActionConcatWrapper` in the original lambda-discrepancy code, action through the embedding never the cell). `carry_dim` is memory-only — hand-rolled `CARRY_DIM` formulas in examples must **not** add `action_dim`; thread a separate `zero_prev_action()` instead. `predict` gains an optional keyword `prev_action=` and returns the memory carry only; manual rollout loops compute the next prev-action from the *executed* action via `fns.encode_action(...)` (this transparently handles epsilon-greedy overrides — encode whatever was executed). The **projection is configurable** (`projection=` on `create_iqlearn_from_env`, `RecurrentFeatureExtractor`): an `int` (default `LinearProjection`), `None` (no-embedding concat), or a builder `(obs_shape, prev_action_dim, rngs) -> nnx.Module` with contract `module(obs, prev_action) -> z` receiving raw (unflattened) obs — see `BattleshipProjection` in `lambda-envs/battleship_sac_mc.py` (`--skip-projection`) for a spatial skip-connection example. With `burn_in_from_stored_carry` the prev-action input is stored in the buffer under `prev_actions` (alongside `carries`) and seeds the burn-in's first step. Tests: `tests/test_prev_action.py` (includes the obs-concat equivalence test and a custom-projection regression).
- **Action masking (`obs_fn` / `mask_fn`)**: discrete only. The buffer always stores the *full* observation; `obs_fn` (default identity) selects what the FE sees, and `mask_fn` (default `None`) derives a legal-action mask from the same full observation. The mask is applied wherever logits become a categorical distribution: `predict`, `get_v`, `get_entropy`, `get_importance_ratios`, and the random prefill policy (which then samples uniformly over legal actions with `behaviour_prob = 1/num_legal`). Masked logits are filled with a large *finite* negative (`-1e9`), **not** `-inf`: `-inf` makes the `probs·log_probs` entropy term back-propagate `NaN` (the `jnp.where`-with-`inf` trap) into the shared FE; `-1e9` keeps masked probabilities at ~0 with finite gradients. Fully-masked rows (zero-padded burn-in obs) are left unmasked to keep softmax well defined.

## Examples

Scripts in `examples/` are standalone demos — not collected by pytest.

- **`mountain_car_discrete.py` / `mountain_car_continuous.py`**: IQ-Learn imitation learning on gymnasium MountainCar environments. Require `gymnasium` and `imitation-gym-wrappers`. Two-step workflow: `--record` to capture expert demos, then train and visualise.
- **`cartpole_sac.py` / `pendulum_sac.py`**: Pure online SAC (no expert data) on gymnax CartPole-v1 and Pendulum-v1. Require `gymnax` for training; `gymnasium[classic-control]` is optional and only needed for the post-training visualisation window. CLI flags: `--rounds`, `--train-steps`, `--seed`.
- **`lambda-envs/pocman_sac_mc.py` / `lambda-envs/battleship_sac_mc.py`**: Pure online SAC + λ-discrepancy on `lambda-envs` POMDPs (require `lambda-envs`). Multi-seed, vmapped/jitted training with a recurrent FE (`--memory-type`). **Battleship needs action masking**: the env packs a legal-action mask into the observation tail, and the demo splits it via `obs_fn`/`mask_fn` passed to `create_iqlearn_from_env` — `obs_fn=obs[...,:1]` (only the last-shot bit reaches the FE), `mask_fn=obs[...,1:]` (the mask applied to the policy). `mask_fn` defaults to `None` (no masking) everywhere else.
- **`lambda-envs/pocman_pellet_probe.py` / `lambda-envs/battleship_board_probe.py`**: 4-phase probe analysis (train agent → collect hidden states + ground truth → train MLP probe → visualise) testing whether the recurrent memory encodes the hidden state. Battleship probes the ship board from the carry; because its raw obs carries no position, the probe agent feeds `[last_hit_miss, prev-action one-hot]` to the FE via an `ActionHistoryWrapper` (stores the last action in the state so it reaches `get_obs`), keeping the mask out. Phases are skippable (`--skip-train/-collect/-probe`); `--vis-only` renders from saved pkls with just jax+matplotlib. The probe accuracy is split into **fired cells = retention** (the memory directly observed them; tests long-term recall) vs **unfired cells = inference** (never observed; tests reconstructing the hidden board). **Use balanced accuracy / per-class recall, not raw accuracy** — ships are only ~14% of a 10×10 board, so "predict water everywhere" already scores ~86% and makes raw accuracy look flat/high regardless of memory. `board_probe_retention.png` plots per-class recall (hit→ship, miss→water) and balanced vs steps-since-fired — the retention horizon, which shrinks as `--memory-hidden-dim` is reduced. The episode/mp4 renders draw shots as **outlines** (not fills) so the probe's P(ship) belief stays visible under fired cells (filling them would just copy the ground-truth shot result into both panels).
