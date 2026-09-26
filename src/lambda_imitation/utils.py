"""Recurrent networks and the Gymnax factory for discrete actor–critic agents."""

import math
from typing import Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from .actor_critic import Hyperparameters, create_actor_critic


class EnvSpec(NamedTuple):
    """Shape of one observation and number of categorical actions."""

    obs_shape: tuple[int, ...]
    action_dim: int


MemoryType = Literal["identity", "rnn", "gru", "lstm"]


class LinearProjection(nnx.Module):
    """Flatten observations, append previous-action one-hot, apply one Linear."""

    def __init__(self, flat_in, prev_action_dim, out_dim, *, rngs):
        self.linear = nnx.Linear(flat_in + prev_action_dim, out_dim, rngs=rngs)

    def __call__(self, obs, prev_action):
        x = obs.reshape(obs.shape[0], -1)
        if prev_action is not None and prev_action.shape[-1]:
            x = jnp.concatenate([x, prev_action], axis=-1)
        return self.linear(x)


class _ConcatProjection(nnx.Module):
    def __call__(self, obs, prev_action):
        x = obs.reshape(obs.shape[0], -1)
        if prev_action is not None and prev_action.shape[-1]:
            x = jnp.concatenate([x, prev_action], axis=-1)
        return x


class BattleshipProjection(nnx.Module):
    """Allen et al.'s Battleship embedding, with a skip connection for the hit bit.

    Architecture follows ``BattleShipActorCriticRNN`` in
    https://github.com/brownirl/lambda_discrepancy (``lamb/models.py``).
    Our previous-action input enters before the first dense layer. The memoryless
    control adds a third dense layer in place of the recurrent cell. See
    ``THIRD_PARTY_NOTICES.md`` for attribution and the upstream license.
    """

    def __init__(self, hidden_size, flat_in, prev_action_dim, *,
                 extra_layer=False, rngs):
        kernel_init = nnx.initializers.orthogonal(math.sqrt(2))
        bias_init = nnx.initializers.constant(0.0)
        self.dense1 = nnx.Linear(
            flat_in + prev_action_dim, 2 * hidden_size,
            kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        self.dense2 = nnx.Linear(
            2 * hidden_size + 1, hidden_size,
            kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        self.dense3 = (nnx.Linear(
            hidden_size, hidden_size, kernel_init=kernel_init, bias_init=bias_init,
            rngs=rngs) if extra_layer else None)

    def __call__(self, obs, prev_action):
        x = obs.reshape(obs.shape[0], -1)
        hit = x[..., :1]
        if prev_action is not None and prev_action.shape[-1]:
            x = jnp.concatenate([x, prev_action], axis=-1)
        e = nnx.relu(self.dense1(x))
        e = nnx.relu(self.dense2(jnp.concatenate([hit, e], axis=-1)))
        return nnx.relu(self.dense3(e)) if self.dense3 is not None else e


def battleship_projection(hidden_size, extra_layer=False):
    return lambda shape, pa_dim, rngs: BattleshipProjection(
        hidden_size, math.prod(shape), pa_dim, extra_layer=extra_layer, rngs=rngs)


class RecurrentFeatureExtractor(nnx.Module):
    """Projection followed by optional RNN, GRU or LSTM memory.

    ``(carry, obs, prev_action=None) -> (new_carry, features)`` consumes batched
    observations. Memory and previous action are separate inputs; both must
    reset at episode boundaries. LSTM carry is flattened as ``[c, h]``.

    ``projection`` is an integer width (Linear), ``None`` (concatenation), or a
    builder ``(obs_shape, prev_action_dim, rngs) -> module``. Custom modules
    receive raw batched observations and the previous-action encoding.
    """

    def __init__(self, input_shape, projection=256, memory_type="identity",
                 memory_hidden_dim=128, prev_action_dim=0, *, rngs):
        if memory_type not in ("identity", "rnn", "gru", "lstm"):
            raise ValueError(f"unknown memory_type {memory_type!r}")
        if prev_action_dim < 0:
            raise ValueError("prev_action_dim must be nonnegative")
        self.input_shape = (input_shape,) if isinstance(input_shape, int) else tuple(input_shape)
        self.memory_type = memory_type
        self._memory_hidden_dim = memory_hidden_dim
        self.prev_action_dim = prev_action_dim
        if projection is None:
            self.projection = _ConcatProjection()
        elif isinstance(projection, int):
            self.projection = LinearProjection(
                math.prod(self.input_shape), prev_action_dim, projection, rngs=rngs)
        else:
            self.projection = projection(self.input_shape, prev_action_dim, rngs)
        dummy = self.projection(jnp.zeros((1, *self.input_shape), jnp.float32),
                                jnp.zeros((1, prev_action_dim), jnp.float32))
        width = dummy.shape[-1]
        if memory_type == "identity":
            self.cell = None
            self.output_dim = width
        else:
            cell = {"rnn": nnx.SimpleCell, "gru": nnx.GRUCell, "lstm": nnx.LSTMCell}[memory_type]
            self.cell = cell(in_features=width, hidden_features=memory_hidden_dim, rngs=rngs)
            self.output_dim = memory_hidden_dim
            # Older Flax cells retain RNG state, which must not enter jax.grad.
            if hasattr(self.cell, "rngs"):
                del self.cell.rngs

    def __call__(self, carry, obs, prev_action=None):
        if self.prev_action_dim and prev_action is None:
            prev_action = jnp.zeros((obs.shape[0], self.prev_action_dim), jnp.float32)
        z = self.projection(obs, prev_action)
        if self.cell is None:
            return carry, z
        if self.memory_type == "lstm":
            c, h = jnp.split(carry, 2, axis=-1)
            (nc, nh), y = self.cell((c, h), z)
            return jnp.concatenate([nc, nh], axis=-1), y
        return self.cell(carry, z)

    @property
    def carry_dim(self):
        if self.memory_type == "identity":
            return 0
        return self._memory_hidden_dim * (2 if self.memory_type == "lstm" else 1)

    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, self.carry_dim), jnp.float32)

    def initialize_prev_action(self, batch_size):
        return jnp.zeros((batch_size, self.prev_action_dim), jnp.float32)


def env_spec_from_gymnax(env, params):
    """Extract dimensions, rejecting non-categorical action spaces."""
    from gymnax.environments import spaces

    action_space = env.action_space(params)
    if not isinstance(action_space, spaces.Discrete):
        raise ValueError("only discrete (categorical) action spaces are supported")
    return EnvSpec(tuple(env.observation_space(params).shape), int(action_space.n))


def create_actor_critic_from_env(
    env_spec: EnvSpec,
    *,
    hp: Hyperparameters | None = None,
    projection: int | None | Callable = 256,
    memory_type: MemoryType = "identity",
    memory_hidden_dim=128,
    use_prev_action=False,
    actor_dims=(),
    critic_dims=(256, 256),
    lambda1_critic_dims=(256, 256),
    lambda2_critic_dims=(256, 256),
    train_steps=1000,
    approximate_lambda=True,
    critic_layer_norm=False,
    obs_fn=lambda obs: obs,
    mask_fn=None,
    debug=False,
    seed=0,
):
    """Build an agent from observation/action dimensions; no dataset is needed.

    A single shared feature extractor feeds the actor and all twin critics.
    ``obs_fn`` controls which observation channels reach the feature extractor;
    ``mask_fn`` acts on the full observation stored in replay.
    """
    hp = Hyperparameters() if hp is None else hp
    shape = jax.eval_shape(obs_fn, jax.ShapeDtypeStruct(env_spec.obs_shape, jnp.float32)).shape
    key_fe, key_heads = jax.random.split(jax.random.key(seed))
    fe = RecurrentFeatureExtractor(
        shape, projection, memory_type, memory_hidden_dim,
        env_spec.action_dim if use_prev_action else 0, rngs=nnx.Rngs(key_fe))
    return create_actor_critic(
        hp, env_spec.obs_shape, env_spec.action_dim, fe, key_heads,
        train_steps=train_steps, actor_dims=actor_dims, critic_dims=critic_dims,
        lambda1_critic_dims=lambda1_critic_dims, lambda2_critic_dims=lambda2_critic_dims,
        approximate_lambda=approximate_lambda, use_prev_action=use_prev_action,
        critic_layer_norm=critic_layer_norm, obs_fn=obs_fn, mask_fn=mask_fn, debug=debug)
