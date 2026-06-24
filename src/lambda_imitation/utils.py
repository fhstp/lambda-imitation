"""Environment-interface utilities for IQ-Learn.

Provides low-level helpers that extract the observation shape, action
dimensionality, and action bounds from gymnasium, gymnax, and jumanji
environments into a common :class:`EnvSpec` NamedTuple, and a high-level
convenience function :func:`create_iqlearn_from_env` that builds a ready-to-use
IQ-Learn agent from an ``EnvSpec`` and a dict of expert transitions.

Also exposes :class:`RecurrentFeatureExtractor`, an ``nnx.Module`` that
combines a configurable projection module with an optional recurrent cell
(identity, vanilla RNN, GRU or LSTM) and exposes the
``feature_extractor(carry, obs, prev_action=None) -> (new_carry, y)`` calling
convention that :func:`lambda_imitation.iqlearn.create_iqlearn` expects.

All three environment libraries are imported lazily, so only the library you
actually use needs to be installed.

Both continuous (``Box``-style) and discrete action spaces are supported.
For discrete spaces, ``action_low`` and ``action_high`` in the returned
:class:`EnvSpec` are ``None``; ``action_dim`` holds the number of actions.
Observation specs must be flat (single array); nested jumanji observation
specs raise :exc:`ValueError` -- flatten the observation manually before using
this module.

Typical usage (gymnax, GRU memory)::

    import gymnax
    from lambda_imitation.utils import (
        env_spec_from_gymnax, create_iqlearn_from_env,
    )

    env, params = gymnax.make("CartPole-v1")
    spec        = env_spec_from_gymnax(env, params)
    state, fns  = create_iqlearn_from_env(
        spec, expert_data,
        memory_type="gru", memory_hidden_dim=128,
    )
"""

import math
from typing import Callable, Literal, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from .buffer import create_buffer
from .iqlearn import (
    DebugFunctions,
    Hyperparameters,
    SACFunctions,
    SACState,
    create_iqlearn,
)

# ---------------------------------------------------------------------------
# Common spec container
# ---------------------------------------------------------------------------


class EnvSpec(NamedTuple):
    """Environment dimensions extracted from any supported interface.

    Attributes:
        obs_shape: Shape of a single observation, e.g. ``(11,)`` or
            ``(84, 84, 3)``.  The feature extractor receives batches of this
            shape and is responsible for flattening if necessary.
        action_dim: Number of continuous action dimensions, or number of
            discrete actions when ``is_discrete=True``.
        is_discrete: ``True`` for discrete (categorical) action spaces,
            ``False`` for continuous (Box) action spaces.
        action_low: Lower bound per action dimension, shape ``(action_dim,)``.
            ``None`` for discrete action spaces.
        action_high: Upper bound per action dimension, shape ``(action_dim,)``.
            ``None`` for discrete action spaces.
    """

    obs_shape: tuple[int, ...]
    action_dim: int
    is_discrete: bool
    action_low: jax.Array | None = None
    action_high: jax.Array | None = None


# ---------------------------------------------------------------------------
# Recurrent feature extractor
# ---------------------------------------------------------------------------


MemoryType = Literal["identity", "rnn", "gru", "lstm"]


class LinearProjection(nnx.Module):
    """Default projection: flatten obs, concat the prev-action, single Linear.

    Reproduces the embedding layer of standard recurrent baselines (R2D2,
    DRQN): a plain ``nnx.Linear`` with **no activation**.  ``prev_action`` is
    concatenated to the flattened observation before the linear map (matching
    the ``ActionConcatWrapper`` of the original lambda-discrepancy code, where
    the action enters through the embedding rather than the cell).  A ``None``
    or zero-width ``prev_action`` is the observation-only case.

    Args:
        flat_in: Flattened observation width (product of ``obs_shape``).
        prev_action_dim: Width of the prev-action encoding (``0`` = obs only).
        out_dim: Output (embedding) width.
        rngs: Flax NNX RNG container.
    """

    def __init__(
        self, flat_in: int, prev_action_dim: int, out_dim: int, *, rngs: nnx.Rngs
    ):
        self.linear = nnx.Linear(flat_in + prev_action_dim, out_dim, rngs=rngs)

    def __call__(self, obs: jax.Array, prev_action: jax.Array | None) -> jax.Array:
        x = obs.reshape(obs.shape[0], -1)
        if prev_action is not None and prev_action.shape[-1]:
            x = jnp.concatenate([x, prev_action], axis=-1)
        return self.linear(x)


class _ConcatProjection(nnx.Module):
    """Identity projection (``projection=None``): flatten + concat, no params.

    Feeds ``concat([flatten(obs), prev_action])`` straight into the memory
    cell (or out, for ``memory_type="identity"``).
    """

    def __call__(self, obs: jax.Array, prev_action: jax.Array | None) -> jax.Array:
        x = obs.reshape(obs.shape[0], -1)
        if prev_action is not None and prev_action.shape[-1]:
            x = jnp.concatenate([x, prev_action], axis=-1)
        return x


class BattleshipProjection(nnx.Module):
    """Pre-RNN embedding from the original lambda-discrepancy Battleship net.

    Reproduces the embedding of ``BattleShipActorCriticRNN`` in the original
    lambda-discrepancy code (``lamb/models.py``).  Channel 0 of the
    observation is the *hit* bit (did the last shot hit a ship), and it is
    re-injected via a skip connection after the first dense layer so the
    recurrent memory never loses the raw hit signal under the projection::

        x = concat([obs, prev_action])          # [hit | …obs… | a_{t-1}]
        e = relu(Dense(2H)(x))
        e = concat([hit, e])                    # skip-connect the raw hit bit
        e = relu(Dense(H)(e))                   # -> recurrent cell
        e = relu(Dense(H)(e))                   # only if extra_layer=True

    The observation fed in must carry the hit bit at channel 0; the battleship
    probe's ``obs_fn`` already strips the action mask, leaving ``[hit]`` (or
    ``[hit | …]`` under ``--full-obs``).  ``prev_action`` is concatenated
    before the first layer — the probe threads the previously-fired cell's
    one-hot here, which the paper's mask-only variant omits (pass a zero-width
    array / ``prev_action_dim == 0`` to reproduce that).  Dense layers use the
    paper's orthogonal(√2) kernel init with zero bias.

    The original code has two networks: ``BattleShipActorCriticRNN`` feeds the
    two-layer embedding straight into a GRU, while the memoryless
    ``BattleShipActorCritic`` inserts a *third* ``Dense(H)→relu`` in place of
    the recurrent cell (to keep depth comparable).  Set ``extra_layer=True``
    to reproduce that third layer — do so exactly when the feature extractor
    has no recurrent cell (``memory_type="identity"``).

    Args:
        hidden_size: Width ``H``.  The first dense layer is ``2H`` wide, the
            second (cell-input) layer ``H``.  Tie this to the recurrent cell's
            ``memory_hidden_dim``.
        flat_in: Flattened observation width (product of ``obs_shape``).
        prev_action_dim: Width of the prev-action encoding (``0`` = obs only).
        extra_layer: If True, append a third ``Dense(H)→relu`` (the
            ``BattleShipActorCritic`` memoryless variant).  Leave False when a
            recurrent cell follows the projection.
        rngs: Flax NNX RNG container.
    """

    def __init__(
        self, hidden_size: int, flat_in: int, prev_action_dim: int,
        *, extra_layer: bool = False, rngs: nnx.Rngs,
    ):
        kernel_init = nnx.initializers.orthogonal(math.sqrt(2))
        bias_init = nnx.initializers.constant(0.0)
        self.dense1 = nnx.Linear(
            flat_in + prev_action_dim, 2 * hidden_size,
            kernel_init=kernel_init, bias_init=bias_init, rngs=rngs,
        )
        # +1 input feature for the re-injected raw hit bit.
        self.dense2 = nnx.Linear(
            2 * hidden_size + 1, hidden_size,
            kernel_init=kernel_init, bias_init=bias_init, rngs=rngs,
        )
        # Memoryless variant: a third dense replaces the recurrent cell.
        self.dense3 = (
            nnx.Linear(
                hidden_size, hidden_size,
                kernel_init=kernel_init, bias_init=bias_init, rngs=rngs,
            )
            if extra_layer else None
        )

    def __call__(self, obs: jax.Array, prev_action: jax.Array | None) -> jax.Array:
        x = obs.reshape(obs.shape[0], -1)
        hit = x[..., 0:1]
        if prev_action is not None and prev_action.shape[-1]:
            x = jnp.concatenate([x, prev_action], axis=-1)
        e = nnx.relu(self.dense1(x))
        e = jnp.concatenate([hit, e], axis=-1)
        e = nnx.relu(self.dense2(e))
        if self.dense3 is not None:
            e = nnx.relu(self.dense3(e))
        return e


def battleship_projection(
    hidden_size: int, extra_layer: bool = False
) -> Callable[..., nnx.Module]:
    """Projection builder for the original Battleship architecture.

    Returns a callable with the ``(obs_shape, prev_action_dim, rngs) ->
    nnx.Module`` signature :class:`RecurrentFeatureExtractor` expects, building
    a :class:`BattleshipProjection` of width ``hidden_size``.  Pair it with
    ``memory_type="gru"``, ``memory_hidden_dim=hidden_size`` and single-hidden
    actor/critic heads of width ``hidden_size`` to reproduce
    ``BattleShipActorCriticRNN``.

    Args:
        hidden_size: Embedding / cell width ``H`` (see
            :class:`BattleshipProjection`).
        extra_layer: Forwarded to :class:`BattleshipProjection`.  Set True for
            ``memory_type="identity"`` to add the third ``Dense(H)→relu`` of the
            memoryless ``BattleShipActorCritic`` (which has no recurrent cell to
            provide that depth).
    """

    def build(obs_shape, prev_action_dim, rngs):
        flat_in = math.prod(obs_shape)
        return BattleshipProjection(
            hidden_size, flat_in, prev_action_dim,
            extra_layer=extra_layer, rngs=rngs,
        )

    return build


class RecurrentFeatureExtractor(nnx.Module):
    """Projection module followed by an optional recurrent memory cell.

    The module exposes the
    ``feature_extractor(carry, obs, prev_action=None) -> (new_carry, y)``
    calling convention required by
    :func:`lambda_imitation.iqlearn.calculate_latent`.

    Layout::

        (obs, prev_action) --[ projection module ]--> z --[ memory cell ]--> y
                                                                  |
                                                                  +---> new_carry

    The **carry is the memory state only** — the previous action is *not*
    packed into it.  Instead ``prev_action`` (the encoding of the previously
    executed action: one-hot for discrete, raw values for continuous) is an
    explicit input to the projection module, threaded by the caller as a
    sibling of the carry and reset to zeros at episode boundaries.  When the
    prev-action input is disabled (``prev_action_dim == 0``) callers may pass
    ``None``; it is normalised internally to a zero-width array so the
    projection / scan structure is identical either way.

    The projection is configurable:

    * ``int`` -> a :class:`LinearProjection` of that output width (the default,
      reproduces the R2D2/DRQN embedding: flatten obs, concat prev-action,
      one ``Linear`` with no activation).
    * ``None`` -> a :class:`_ConcatProjection` (no embedding; feed
      ``concat([flatten(obs), prev_action])`` straight to the cell).
    * a builder callable ``(obs_shape, prev_action_dim, rngs) -> nnx.Module``
      -> full control (skip connections, convolutions, separate encoders,
      ...).  The returned module must implement
      ``module(obs, prev_action) -> z`` and receives ``obs`` with its raw
      (unflattened) shape, e.g. ``(B, 10, 10)`` for a battleship board.

    The cell, if any, is a Flax NNX ``SimpleCell`` (vanilla RNN), ``GRUCell``
    or ``LSTMCell`` with ``hidden_features=memory_hidden_dim``.  For
    ``"identity"`` no cell is used and the carry is passed through untouched.

    Attributes:
        output_dim: Width of the produced feature vector ``y``.  Equals
            ``memory_hidden_dim`` for recurrent cells, otherwise the
            projection's output width.
        memory_type: One of ``"identity"``, ``"rnn"``, ``"gru"``, ``"lstm"``.
        prev_action_dim: Width of the prev-action encoding fed to the
            projection (``0`` disables the action input entirely).
        input_shape: Raw (unflattened) observation shape the projection sees.

    Args:
        input_shape: Shape of a single observation the projection receives
            (post ``obs_fn``), e.g. ``(10, 10)`` or a flat ``int``.
        projection: ``int`` (default ``256``) for a :class:`LinearProjection`,
            ``None`` for the no-embedding concat projection, or a builder
            callable ``(obs_shape, prev_action_dim, rngs) -> nnx.Module``.
        memory_type: ``"identity"`` (no recurrency), ``"rnn"``, ``"gru"`` or
            ``"lstm"``.
        memory_hidden_dim: Hidden-state width for the recurrent cell.
            Ignored when ``memory_type="identity"``.
        prev_action_dim: Width of the prev-action encoding.  ``0`` (default)
            reproduces the observation-only behaviour exactly.
        rngs: Flax NNX RNG container used to initialise parameters.
    """

    def __init__(
        self,
        input_shape: int | tuple[int, ...],
        projection: int | None | Callable[..., nnx.Module] = 256,
        memory_type: MemoryType = "identity",
        memory_hidden_dim: int = 128,
        prev_action_dim: int = 0,
        *,
        rngs: nnx.Rngs,
    ):
        if memory_type not in ("identity", "rnn", "gru", "lstm"):
            raise ValueError(
                f"memory_type must be one of 'identity', 'rnn', 'gru', 'lstm'; "
                f"got {memory_type!r}."
            )
        if prev_action_dim < 0:
            raise ValueError(f"prev_action_dim must be >= 0; got {prev_action_dim}.")

        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        self.input_shape = tuple(input_shape)
        flat_in = math.prod(self.input_shape)

        self.memory_type = memory_type
        self._memory_hidden_dim = memory_hidden_dim
        self.prev_action_dim = prev_action_dim

        if projection is None:
            self.projection = _ConcatProjection()
        elif isinstance(projection, int):
            self.projection = LinearProjection(
                flat_in, prev_action_dim, projection, rngs=rngs
            )
        else:  # builder callable (obs_shape, prev_action_dim, rngs) -> Module
            self.projection = projection(self.input_shape, prev_action_dim, rngs)

        # Infer the projection's output width (= cell input width) with one
        # dummy forward, so custom modules need not expose an out_dim.
        dummy_obs = jnp.zeros((1, *self.input_shape), dtype=jnp.float32)
        dummy_pa = jnp.zeros((1, prev_action_dim), dtype=jnp.float32)
        cell_in_dim = int(self.projection(dummy_obs, dummy_pa).shape[-1])

        if memory_type == "identity":
            self.cell = None
            self.output_dim = cell_in_dim
        elif memory_type == "rnn":
            self.cell = nnx.SimpleCell(
                in_features=cell_in_dim,
                hidden_features=memory_hidden_dim,
                rngs=rngs,
            )
            self.output_dim = memory_hidden_dim
        elif memory_type == "gru":
            self.cell = nnx.GRUCell(
                in_features=cell_in_dim,
                hidden_features=memory_hidden_dim,
                rngs=rngs,
            )
            self.output_dim = memory_hidden_dim
        else:  # "lstm"
            self.cell = nnx.LSTMCell(
                in_features=cell_in_dim,
                hidden_features=memory_hidden_dim,
                rngs=rngs,
            )
            self.output_dim = memory_hidden_dim

    def __call__(
        self,
        carry: jax.Array,
        obs: jax.Array,
        prev_action: jax.Array | None = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """One recurrent step.

        The carry is the memory state only, a single flat array of shape
        ``(batch, carry_dim)``.  For LSTM, the cell's ``(c, h)`` tuple is
        concatenated along the last axis on output and split back inside this
        method on input, so callers never see the tuple structure.

        Args:
            carry: Memory state from the previous step, shape
                ``(batch, carry_dim)``.  Use :meth:`initialize_carry` to build
                a zero carry of the right shape.
            obs: Observation batch of shape ``(batch, *obs_shape)``, passed to
                the projection module with its raw shape (the default
                projection flattens it internally).
            prev_action: Encoding of the previously executed action, shape
                ``(batch, prev_action_dim)`` (zeros = episode start).  ``None``
                is allowed and normalised to a zero array; when
                ``prev_action_dim == 0`` it is ignored entirely.

        Returns:
            ``(new_carry, y)`` where both are flat arrays;
            ``y`` has shape ``(batch, output_dim)``.
        """
        if self.prev_action_dim and prev_action is None:
            prev_action = jnp.zeros(
                (obs.shape[0], self.prev_action_dim), dtype=jnp.float32
            )
        z = self.projection(obs, prev_action)
        if self.cell is None:
            return carry, z
        if self.memory_type == "lstm":
            c, h = jnp.split(carry, 2, axis=-1)
            (new_c, new_h), out = self.cell((c, h), z)
            return jnp.concatenate([new_c, new_h], axis=-1), out
        new_carry, out = self.cell(carry, z)
        return new_carry, out

    @property
    def carry_dim(self) -> int:
        """Width of the flat memory carry vector.

        ``0`` for identity, ``memory_hidden_dim`` for RNN/GRU, and
        ``2 * memory_hidden_dim`` for LSTM (concatenated ``[c, h]``).  The
        prev-action encoding is *not* part of the carry.
        """
        if self.memory_type == "identity":
            return 0
        if self.memory_type == "lstm":
            return 2 * self._memory_hidden_dim
        return self._memory_hidden_dim

    def initialize_carry(self, batch_size: int) -> jax.Array:
        """Construct a zero memory carry matching this extractor's cell.

        Always returns a single flat array of shape
        ``(batch_size, carry_dim)``; LSTM's ``(c, h)`` split is handled
        internally by :meth:`__call__`.

        Args:
            batch_size: Leading dim of the carry (one carry per env / row in
                a sample batch).

        Returns:
            Zero array of shape ``(batch_size, carry_dim)`` (``0``-width for
            identity, ``memory_hidden_dim`` for RNN/GRU, ``2 *
            memory_hidden_dim`` for LSTM).
        """
        return jnp.zeros((batch_size, self.carry_dim), dtype=jnp.float32)

    def initialize_prev_action(self, batch_size: int) -> jax.Array:
        """Construct a zero prev-action input (``"no previous action"``).

        Returns a ``(batch_size, prev_action_dim)`` zero array — width ``0``
        when the prev-action input is disabled.
        """
        return jnp.zeros((batch_size, self.prev_action_dim), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Low-level extractors
# ---------------------------------------------------------------------------


def env_spec_from_gymnasium(env) -> EnvSpec:
    """Extract an :class:`EnvSpec` from a ``gymnasium.Env``.

    Reads ``env.observation_space`` and ``env.action_space`` directly.  The
    observation space must be ``gymnasium.spaces.Box``.  The action space may
    be either ``Box`` (continuous) or ``Discrete`` (discrete).

    Args:
        env: A ``gymnasium.Env`` instance (or any object exposing
             ``observation_space`` and ``action_space`` attributes).

    Returns:
        An :class:`EnvSpec` populated from the environment's spaces.
        ``is_discrete=True`` and ``action_low=action_high=None`` for discrete
        action spaces.

    Raises:
        ImportError: If ``gymnasium`` is not installed.
        ValueError: If the observation space is not a ``gymnasium.spaces.Box``,
            or if the action space is neither ``Box`` nor ``Discrete``.
    """
    try:
        import gymnasium.spaces as spaces
    except ImportError as exc:
        raise ImportError(
            "gymnasium is required for env_spec_from_gymnasium. "
            "Install it with: pip install gymnasium"
        ) from exc

    obs_space = env.observation_space
    act_space = env.action_space

    if not isinstance(obs_space, spaces.Box):
        raise ValueError(
            f"env_spec_from_gymnasium requires a Box observation space, "
            f"got {type(obs_space).__name__}."
        )

    if isinstance(act_space, spaces.Discrete):
        return EnvSpec(
            obs_shape=tuple(obs_space.shape),
            action_dim=int(act_space.n),
            is_discrete=True,
        )

    if isinstance(act_space, spaces.Box):
        action_dim = math.prod(act_space.shape)
        return EnvSpec(
            obs_shape=tuple(obs_space.shape),
            action_dim=action_dim,
            is_discrete=False,
            action_low=jnp.asarray(
                act_space.low.reshape(action_dim), dtype=jnp.float32
            ),
            action_high=jnp.asarray(
                act_space.high.reshape(action_dim), dtype=jnp.float32
            ),
        )

    raise ValueError(
        f"env_spec_from_gymnasium requires a Box or Discrete action space, "
        f"got {type(act_space).__name__}."
    )


def env_spec_from_gymnax(env, params) -> EnvSpec:
    """Extract an :class:`EnvSpec` from a ``gymnax`` environment.

    Calls ``env.observation_space(params)`` and ``env.action_space(params)``
    to obtain space objects.  The observation space must be
    ``gymnax.environments.spaces.Box``; the action space may be either
    ``Box`` (continuous) or ``Discrete`` (discrete).

    Args:
        env: A ``gymnax`` environment instance (returned by ``gymnax.make``).
        params: The corresponding ``EnvParams`` instance (also returned by
            ``gymnax.make``).

    Returns:
        An :class:`EnvSpec` populated from the environment's spaces.

    Raises:
        ImportError: If ``gymnax`` is not installed.
        ValueError: If the observation space is not a ``Box``, or if the
            action space is neither ``Box`` nor ``Discrete``.
    """
    try:
        import gymnax.environments.spaces as spaces
    except ImportError as exc:
        raise ImportError(
            "gymnax is required for env_spec_from_gymnax. "
            "Install it with: pip install gymnax"
        ) from exc

    obs_space = env.observation_space(params)
    act_space = env.action_space(params)

    if not isinstance(obs_space, spaces.Box):
        raise ValueError(
            f"env_spec_from_gymnax requires a Box observation space, "
            f"got {type(obs_space).__name__}."
        )

    if isinstance(act_space, spaces.Discrete):
        return EnvSpec(
            obs_shape=tuple(obs_space.shape),
            action_dim=int(act_space.n),
            is_discrete=True,
        )

    if isinstance(act_space, spaces.Box):
        action_dim = math.prod(act_space.shape)
        low = jnp.broadcast_to(
            jnp.asarray(act_space.low, dtype=jnp.float32), (action_dim,)
        )
        high = jnp.broadcast_to(
            jnp.asarray(act_space.high, dtype=jnp.float32), (action_dim,)
        )
        return EnvSpec(
            obs_shape=tuple(obs_space.shape),
            action_dim=action_dim,
            is_discrete=False,
            action_low=low,
            action_high=high,
        )

    raise ValueError(
        f"env_spec_from_gymnax requires a Box or Discrete action space, "
        f"got {type(act_space).__name__}."
    )


def env_spec_from_jumanji(env) -> EnvSpec:
    """Extract an :class:`EnvSpec` from a ``jumanji`` environment.

    Reads ``env.observation_spec`` and ``env.action_spec``; the observation
    spec must be a flat ``jumanji.specs.Array`` or ``BoundedArray`` (not a
    nested ``Spec`` container).  The action spec may be a
    ``jumanji.specs.BoundedArray`` (continuous) or
    ``jumanji.specs.DiscreteArray`` (discrete).
    ``MultiDiscreteArray`` is not supported.

    Args:
        env: A ``jumanji`` environment instance (returned by ``jumanji.make``).

    Returns:
        An :class:`EnvSpec` populated from the environment's specs.

    Raises:
        ImportError: If ``jumanji`` is not installed.
        ValueError: If the observation spec is nested, if the action spec is a
            ``MultiDiscreteArray``, or if the action spec is neither a
            ``BoundedArray`` nor a ``DiscreteArray``.
    """
    try:
        import jumanji.specs as specs
    except ImportError as exc:
        raise ImportError(
            "jumanji is required for env_spec_from_jumanji. "
            "Install it with: pip install jumanji"
        ) from exc

    obs_spec = env.observation_spec
    act_spec = env.action_spec

    if not isinstance(obs_spec, specs.Array):
        raise ValueError(
            f"env_spec_from_jumanji requires a flat Array observation spec, "
            f"got {type(obs_spec).__name__}. Flatten the observation manually "
            f"before using this utility."
        )

    if isinstance(act_spec, specs.MultiDiscreteArray):
        raise ValueError(
            f"env_spec_from_jumanji does not support MultiDiscreteArray action "
            f"specs. Flatten or handle this action space manually."
        )

    if isinstance(act_spec, specs.DiscreteArray):
        return EnvSpec(
            obs_shape=tuple(obs_spec.shape),
            action_dim=int(act_spec.num_values),
            is_discrete=True,
        )

    if isinstance(act_spec, specs.BoundedArray):
        action_dim = math.prod(act_spec.shape)
        low = jnp.broadcast_to(
            jnp.asarray(act_spec.minimum, dtype=jnp.float32), (action_dim,)
        )
        high = jnp.broadcast_to(
            jnp.asarray(act_spec.maximum, dtype=jnp.float32), (action_dim,)
        )
        return EnvSpec(
            obs_shape=tuple(obs_spec.shape),
            action_dim=action_dim,
            is_discrete=False,
            action_low=low,
            action_high=high,
        )

    raise ValueError(
        f"env_spec_from_jumanji requires a BoundedArray or DiscreteArray action "
        f"spec, got {type(act_spec).__name__}."
    )


# ---------------------------------------------------------------------------
# High-level convenience factory
# ---------------------------------------------------------------------------


def create_iqlearn_from_env(
    env_spec: EnvSpec,
    expert_data: dict[str, jax.Array],
    buffer_size: int = 10_000,
    hp: Hyperparameters | None = None,
    projection: int | None | Callable[..., nnx.Module] = 256,
    memory_type: MemoryType = "identity",
    memory_hidden_dim: int = 128,
    use_prev_action: bool = False,
    actor_dims: tuple[int, ...] = (),
    critic_dims: tuple[int, ...] = (256, 256),
    lambda1_critic_dims: tuple[int, ...] = (256, 256),
    lambda2_critic_dims: tuple[int, ...] = (256, 256),
    train_steps: int = 1000,
    obs_key: str = "observations",
    action_key: str = "actions",
    approximate_lambda: bool = False,
    critic_layer_norm: bool = False,
    obs_fn: Callable[[jax.Array], jax.Array] = lambda obs: obs,
    mask_fn: Callable[[jax.Array], jax.Array] | None = None,
    burn_in_from_stored_carry: bool = False,
    use_gvd: bool = False,
    gvd_feature_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    gvd_sf_dims: tuple[int, ...] = (128,),
    debug: bool = False,
    seed: int = 0,
    use_sac: bool = True,
) -> (
    "Tuple[SACState, SACFunctions] | "
    "Tuple[SACState, SACFunctions, DebugFunctions]"
):
    """Build a ready-to-use IQ-Learn agent from an :class:`EnvSpec`.

    Thin wrapper around :func:`lambda_imitation.iqlearn.create_iqlearn` that
    handles:

    * Computing ``action_scale`` / ``action_bias`` (continuous spaces) so the
      actor's ``tanh`` output maps into the environment's real action range,
      and setting a sensible default ``target_entropy``.
    * Creating and pre-filling a replay :class:`Buffer` from ``expert_data``.
    * Constructing a single :class:`RecurrentFeatureExtractor` whose memory
      type and hidden width are configurable.

    The expert data is stored as-is; no normalisation is applied.  If
    ``expert_data`` contains more entries than ``buffer_size``, the last
    ``buffer_size`` transitions are kept (the buffer wraps circularly).

    Args:
        env_spec: Environment dimensions from one of the ``env_spec_from_*``
            extractors.
        expert_data: Dict mapping string keys to arrays of shape
            ``(N, *item_shape)``.  Must contain at least the keys ``obs_key``
            and ``action_key``.  For discrete spaces, actions should be
            stored as ``float32`` indices of shape ``(N, 1)``.
        buffer_size: Maximum number of transitions stored in the expert
            replay buffer.
        hp: :class:`Hyperparameters` instance.  When ``None``, defaults are
            used with ``target_entropy`` chosen automatically for the action
            space type.
        projection: Projection applied before the recurrent cell.  An ``int``
            (default ``256``) builds a :class:`LinearProjection` of that width
            (flatten obs, concat prev-action, one ``Linear``, no activation);
            ``None`` skips the embedding and feeds
            ``concat([flatten(obs), prev_action])`` straight into the memory
            cell; a builder callable ``(obs_shape, prev_action_dim, rngs) ->
            nnx.Module`` gives full control (skip connections, convolutions,
            ...).  Custom modules receive ``obs`` with its raw (unflattened)
            ``obs_fn`` shape and must implement
            ``module(obs, prev_action) -> z``.
        memory_type: ``"identity"`` (no recurrency), ``"rnn"``, ``"gru"`` or
            ``"lstm"``.  Selects the recurrent cell that follows the linear
            projection.
        memory_hidden_dim: Hidden-state width for the recurrent cell.
            Ignored when ``memory_type="identity"``.
        use_prev_action: If True, feed the previously executed action into
            the feature extractor's projection alongside the observation
            (one-hot for discrete, raw values for continuous), mirroring the
            ``ActionConcatWrapper`` of the original lambda-discrepancy code.
            The FE is built with ``prev_action_dim == env_spec.action_dim``;
            the prev-action is threaded next to the carry (not packed into
            it) and reset to zero at episode start.
        actor_dims: Hidden widths of the actor head.  ``()`` is a direct
            linear projection.
        critic_dims: Hidden widths of each SAC critic head.
        lambda1_critic_dims: Hidden widths of the λ1 critic heads (twin-Q).
            Only used when ``approximate_lambda=True``.
        lambda2_critic_dims: Hidden widths of the λ2 critic heads (twin-Q).
            Only used when ``approximate_lambda=True``.
        train_steps: Number of gradient steps per
            :func:`SACFunctions.train` call.
        obs_key: Key under which observations are stored in ``expert_data``
            and in the buffer.
        action_key: Key under which actions are stored.
        approximate_lambda: If True, enable the λ-discrepancy critic branches.
        critic_layer_norm: If True, apply LayerNorm in the hidden layers of
            all critic heads (SAC twin + λ-critics).  Recommended whenever
            ``approximate_lambda=True``: the λ-discrepancy regulariser trains
            the shared FE only, and the resulting representation drift can
            otherwise blow up the one-step-bootstrap SAC critic.
        obs_fn: Pure function mapping the raw observation to the part fed to
            the feature extractor (defaults to the identity).  The FE is sized
            from ``obs_fn``'s output; the full raw observation is still stored
            in the buffer.  Use together with ``mask_fn`` to keep an
            action-mask channel out of the network's observation.
        mask_fn: Optional pure function mapping the raw observation to a boolean
            action mask (discrete only).  When given, illegal actions are
            removed from the policy everywhere (action selection, soft value,
            entropy, importance ratios, random pre-fill).  ``None`` disables
            masking.  Forwarded verbatim to :func:`create_iqlearn`.
        burn_in_from_stored_carry: If True, store the recurrent carry the
            online policy consumed at every step in the online buffer and
            initialise the burn-in of each sampled training sequence from it
            instead of zeros (R2D2 "stored state"; allows a much shorter
            ``burn_in_length``).  Costs ``carry_dim * online_buffer_size *
            4`` bytes of extra buffer memory per agent.  Forwarded verbatim
            to :func:`create_iqlearn`.
        use_gvd: If True, enable the General Value Discrepancy branches: two
            successor-feature V-heads trained on ``gvd_feature_fn(obs,
            prev_action)`` cumulants whose squared discrepancy regularises the
            shared FE (reward-free).  Discrete-only.  Forwarded verbatim to
            :func:`create_iqlearn`.
        gvd_feature_fn: Pure function ``(obs, prev_action) -> features`` mapping
            the **raw** observation ``o_t`` (before ``obs_fn`` — same
            convention as ``mask_fn``) and the previous action ``a_{t-1}``
            (float32 index ``(*batch, 1)``) to a feature vector ``(*batch,
            n_features)``.  Required when ``use_gvd=True``.  See
            :func:`create_iqlearn` for the alignment / hit-gating contract.
        gvd_sf_dims: Hidden widths of each SF head.
        debug: If True, return a 3-tuple whose last element is a
            :class:`DebugFunctions` named tuple.
        seed: Integer seed for the ``nnx.Rngs`` used to initialise the
            feature extractor.

    Returns:
        ``(SACState, SACFunctions)``, or
        ``(SACState, SACFunctions, DebugFunctions)`` when ``debug=True``.

    Raises:
        ValueError: If ``expert_data`` is missing ``obs_key`` or
            ``action_key``, or contains zero transitions.
    """
    if obs_key not in expert_data:
        raise ValueError(
            f"expert_data must contain the key '{obs_key}' "
            f"(got keys: {sorted(expert_data)})."
        )
    if action_key not in expert_data:
        raise ValueError(
            f"expert_data must contain the key '{action_key}' "
            f"(got keys: {sorted(expert_data)})."
        )

    n_transitions = expert_data[obs_key].shape[0]
    if n_transitions == 0:
        raise ValueError("expert_data arrays must have at least one transition.")

    if env_spec.is_discrete:
        if hp is None:
            hp = Hyperparameters(target_entropy=0.2)
        action_scale = 1.0
        action_bias = 0.0
    else:
        if hp is None:
            hp = Hyperparameters(target_entropy=float(-env_spec.action_dim))
        # action = tanh(raw) * scale + bias, scale=(high-low)/2, bias=(high+low)/2
        action_scale = (env_spec.action_high - env_spec.action_low) / 2.0
        action_bias = (env_spec.action_high + env_spec.action_low) / 2.0

    shapes = {k: v.shape[1:] for k, v in expert_data.items()}

    buffer, buf_fns = create_buffer(
        shapes=shapes,
        size=buffer_size,
        sampling_size=hp.batch_size,
        this_step_infos=[obs_key, action_key],
        next_step_infos=[obs_key],
    )

    # Fill the buffer with expert transitions (non-terminal throughout; the
    # last transition is left non-terminal so the next-obs slot wraps cleanly)
    for i in range(n_transitions):
        step = {k: v[i] for k, v in expert_data.items()}
        terminated = bool(i == n_transitions - 1)
        buffer = buf_fns.add(buffer, step, terminated)

    # Size the FE from the (possibly reduced) observation ``obs_fn`` produces,
    # not the raw obs — when ``obs_fn`` strips an action-mask channel the FE
    # must only see the remaining features.
    fe_obs_shape = jax.eval_shape(
        obs_fn, jax.ShapeDtypeStruct(env_spec.obs_shape, jnp.float32)
    ).shape
    key = jax.random.key(seed)
    key_fe, key_heads = jax.random.split(key)
    feature_extractor = RecurrentFeatureExtractor(
        input_shape=fe_obs_shape,
        projection=projection,
        memory_type=memory_type,
        memory_hidden_dim=memory_hidden_dim,
        prev_action_dim=env_spec.action_dim if use_prev_action else 0,
        rngs=nnx.Rngs(key_fe),
    )

    return create_iqlearn(
        params=hp,
        buffer=buffer,
        action_dim=env_spec.action_dim,
        feature_extractor=feature_extractor,
        key=key_heads,
        obs_key=obs_key,
        action_key=action_key,
        action_scale=action_scale,
        action_bias=action_bias,
        train_steps=train_steps,
        actor_dims=actor_dims,
        critic_dims=critic_dims,
        lambda1_critic_dims=lambda1_critic_dims,
        lambda2_critic_dims=lambda2_critic_dims,
        is_discrete=env_spec.is_discrete,
        approximate_lambda=approximate_lambda,
        use_prev_action=use_prev_action,
        critic_layer_norm=critic_layer_norm,
        obs_fn=obs_fn,
        mask_fn=mask_fn,
        burn_in_from_stored_carry=burn_in_from_stored_carry,
        use_gvd=use_gvd,
        gvd_feature_fn=gvd_feature_fn,
        gvd_sf_dims=gvd_sf_dims,
        debug=debug,
        use_sac=use_sac,
    )
