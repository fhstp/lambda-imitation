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
    test_iqlearn.py       ← 37 tests
    test_utils.py         ← 37 tests
pyproject.toml            ← hatchling build, optional extras, pytest config
```

## Architecture

- `buffer.py` -- Generic circular replay buffer using JAX arrays and NamedTuples
- `iqlearn.py` -- SAC-style actor-critic with IQ-Learn reward recovery (configurable networks, continuous actions)
- `utils.py` -- Environment-interface adapters: extracts `EnvSpec` from gymnasium / gymnax / jumanji environments, plus `create_iqlearn_from_env` high-level factory

**Design pattern:** Purely functional. Both modules use a factory pattern (`create_buffer`, `create_iqlearn`) that returns `(state_namedtuple, functions_namedtuple)`. No mutable class state. Flax NNX models are split into graph definition + state via `nnx.split`/`nnx.merge` for functional updates inside `jax.jit`.

### Network design (iqlearn.py)

Networks are split into a **feature extractor** and a **head**:

- `MLPFeatureExtractor(input_dim, hidden_dims, rngs)` -- configurable MLP backbone; flattens input, applies relu layers. `hidden_dims=(256, 256)` by default.
- `ActorHead(feature_dim, action_dim, hidden_dims, rngs)` -- outputs `2*action_dim` (mean + log_std). `hidden_dims=()` by default (direct linear projection).
- `CriticHead(feature_dim, action_dim, hidden_dims, rngs)` -- concats features and actions, outputs twin Q-values. `hidden_dims=(256, 256)` by default.

Both are stored together in a `NetworkState(fe, head)` NamedTuple, which is a JAX pytree. Actor and critic each receive **separate** feature extractor instances, so they can differ in architecture and output dimension.

`create_iqlearn` infers the feature dim by running a dummy forward pass on the provided feature extractor before splitting it. Use `nnx.List` (not plain `list`) for any collection of sub-modules inside an `nnx.Module`.

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

## Conventions

- All state is immutable NamedTuples; updates return new instances.
- Buffer entries are keyed by string names in `info: dict[str, jax.Array]`, not fixed fields.
- Hyperparameters are a NamedTuple with defaults, not a dataclass.
- `jax.lax.scan` is used for multi-step training loops; avoid Python for-loops inside JIT-compiled code.
- Actor/Critic are MLP networks; input is flattened via `x.reshape(x.shape[0], -1)` to handle any obs shape.
- `utils.py` extractors do lazy imports (each library only imported when its extractor is called); only the library you use needs to be installed.
- `utils.py` only supports continuous (Box-style) action spaces; discrete action spaces raise `ValueError`.
- Mocking gymnax/jumanji in tests: use `ModuleType` objects (not `MagicMock`) for the full parent-module chain and wire `.spaces`/`.specs` attributes explicitly, so CPython's attribute-traversal in dotted imports resolves to the mock classes rather than auto-generated MagicMock attributes.
- All imports within the package use relative imports (`from .buffer import ...`).
