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
    test_iqlearn.py       ← 60 tests
    test_utils.py         ← 45 tests
examples/
    mountain_car_discrete.py     ← MountainCar-v0 demo (discrete)
    mountain_car_continuous.py   ← MountainCarContinuous-v0 demo (continuous)
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

## Examples

Scripts in `examples/` are standalone demos — not collected by pytest. They require `gymnasium` and `imitation-gym-wrappers`. Each has `--record` / `--steps` / `--episodes` / `--file` CLI flags; run with `--help` for the full two-step workflow.
