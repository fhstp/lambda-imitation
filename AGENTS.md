# AGENTS.md

## Overview

IQ-Learn (Inverse Q-Learning) imitation learning implementation in **Python** using **JAX + Flax NNX + Optax**. Currently on the `fresh_start` branch -- an active rewrite with two source files.

## Architecture

- `buffer.py` -- Generic circular replay buffer using JAX arrays and NamedTuples
- `iqlearn.py` -- SAC-style actor-critic with IQ-Learn reward recovery (configurable networks, continuous actions)

**Design pattern:** Purely functional. Both modules use a factory pattern (`create_buffer`, `create_iqlearn`) that returns `(state_namedtuple, functions_namedtuple)`. No mutable class state. Flax NNX models are split into graph definition + state via `nnx.split`/`nnx.merge` for functional updates inside `jax.jit`.

### Network design (iqlearn.py)

Networks are split into a **feature extractor** and a **head**:

- `MLPFeatureExtractor(input_dim, hidden_dims, rngs)` -- configurable MLP backbone; flattens input, applies relu layers. `hidden_dims=(256, 256)` by default.
- `ActorHead(feature_dim, action_dim, hidden_dims, rngs)` -- outputs `2*action_dim` (mean + log_std). `hidden_dims=()` by default (direct linear projection).
- `CriticHead(feature_dim, action_dim, hidden_dims, rngs)` -- concats features and actions, outputs twin Q-values. `hidden_dims=(256, 256)` by default.

Both are stored together in a `NetworkState(fe, head)` NamedTuple, which is a JAX pytree. Actor and critic each receive **separate** feature extractor instances, so they can differ in architecture and output dimension.

`create_iqlearn` infers the feature dim by running a dummy forward pass on the provided feature extractor before splitting it. Use `nnx.List` (not plain `list`) for any collection of sub-modules inside an `nnx.Module`.

## Dependencies (no manifest exists)

Inferred from imports: `jax`, `jaxlib`, `flax` (NNX API), `optax`, `numpy`, `pytest`. Install manually; there is no `requirements.txt` or `pyproject.toml`.

## Commands

No build system, CI, linting, or formatting is configured.

- Run all tests: `pytest test_buffer.py test_iqlearn.py`
- Run buffer tests only: `pytest test_buffer.py`
- Run iqlearn tests only: `pytest test_iqlearn.py`

## Conventions

- All state is immutable NamedTuples; updates return new instances.
- Buffer entries are keyed by string names in `info: dict[str, jax.Array]`, not fixed fields.
- Hyperparameters are a NamedTuple with defaults, not a dataclass.
- `jax.lax.scan` is used for multi-step training loops; avoid Python for-loops inside JIT-compiled code.
- Actor/Critic are MLP networks; input is flattened via `x.reshape(x.shape[0], -1)` to handle any obs shape.
