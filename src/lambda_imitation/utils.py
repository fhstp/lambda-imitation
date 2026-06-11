"""Environment-interface utilities for IQ-Learn.

Provides low-level helpers that extract the observation shape, action
dimensionality, and action bounds from gymnasium, gymnax, and jumanji
environments into a common :class:`EnvSpec` NamedTuple, and a high-level
convenience function :func:`create_iqlearn_from_env` that builds a ready-to-use
IQ-Learn agent from an ``EnvSpec`` and a dict of expert transitions.

Also exposes :class:`RecurrentFeatureExtractor`, an ``nnx.Module`` that
combines an MLP backbone with an optional recurrent cell (identity, vanilla
RNN, GRU or LSTM) and exposes the
``feature_extractor(carry, obs) -> (new_carry, y)`` calling convention that
:func:`lambda_imitation.iqlearn.create_iqlearn` expects.

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


class RecurrentFeatureExtractor(nnx.Module):
    """Linear projection followed by an optional recurrent memory cell.

    The module exposes the
    ``feature_extractor(carry, obs) -> (new_carry, y)`` calling convention
    required by :func:`lambda_imitation.iqlearn.calculate_latent`.

    Layout::

        [obs | prev_action] --[ Linear(projection_dim) ]--> z --[ memory cell ]--> y
                                                                       |
                                                                       +---> new_carry

    The projection is a plain ``nnx.Linear`` with **no activation**, mirroring
    the embedding layer used in standard recurrent baselines (e.g. R2D2,
    DRQN).  Set ``projection_dim=None`` to skip it and feed the flattened raw
    observation straight into the recurrent cell.  The cell, if any, is a
    Flax NNX ``SimpleCell`` (vanilla RNN), ``GRUCell`` or ``LSTMCell`` with
    ``hidden_features=memory_hidden_dim``.  For ``"identity"`` no cell is
    used and the carry is passed through untouched.

    When ``prev_action_dim > 0`` the carry is the full agent recurrent state

        carry = [ memory_carry | prev_action_encoding ]

    where the tail holds the encoding of the previously executed action
    (one-hot for discrete, raw values for continuous).  The tail is split off
    on input, concatenated with the flattened observation *before* the
    projection (matching the ``ActionConcatWrapper`` of the original
    lambda-discrepancy code, where the action enters through the embedding
    rather than the cell), and re-appended unchanged on output.  Writing the
    new action into the tail is the caller's responsibility (see
    :meth:`write_prev_action`); a zero tail means "no previous action", i.e.
    episode start — so any code that zeroes the carry at resets also resets
    the action input for free.

    Attributes:
        output_dim: Width of the produced feature vector ``y``.  Equals
            ``memory_hidden_dim`` for recurrent cells, otherwise
            ``projection_dim`` (or ``input_dim + prev_action_dim`` when
            ``projection_dim`` is ``None``).
        memory_type: One of ``"identity"``, ``"rnn"``, ``"gru"``, ``"lstm"``.
        prev_action_dim: Width of the prev-action tail of the carry
            (``0`` disables the action input entirely).

    Args:
        input_dim: Flat size of a single observation (product of all dims).
        projection_dim: Width of the linear embedding applied to the
            flattened observation before the memory cell.  ``None`` skips
            the projection.
        memory_type: ``"identity"`` (no recurrency), ``"rnn"``, ``"gru"`` or
            ``"lstm"``.
        memory_hidden_dim: Hidden-state width for the recurrent cell.
            Ignored when ``memory_type="identity"``.
        prev_action_dim: If ``> 0``, the carry grows by this many trailing
            entries holding the previous action's encoding, which is
            concatenated with the flattened observation before the
            projection.  ``0`` (default) reproduces the observation-only
            behaviour exactly.
        rngs: Flax NNX RNG container used to initialise parameters.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int | None = 256,
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

        self.memory_type = memory_type
        self._memory_hidden_dim = memory_hidden_dim
        self.prev_action_dim = prev_action_dim

        net_in_dim = input_dim + prev_action_dim
        if projection_dim is None:
            self.projection = None
            cell_in_dim = net_in_dim
        else:
            self.projection = nnx.Linear(net_in_dim, projection_dim, rngs=rngs)
            cell_in_dim = projection_dim

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

    def __call__(self, carry: jax.Array, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """One recurrent step.

        The carry is always a single flat array of shape
        ``(batch, carry_dim)``.  For LSTM, the cell's ``(c, h)`` tuple is
        concatenated along the last axis on output and split back inside this
        method on input, so callers never see the tuple structure.

        Args:
            carry: Hidden state from the previous step, shape
                ``(batch, carry_dim)``.  Use :meth:`initialize_carry` to build
                a zero carry of the right shape.  When ``prev_action_dim > 0``
                the trailing ``prev_action_dim`` entries hold the previous
                action's encoding (zeros = episode start); they are consumed
                as network input and passed through to ``new_carry``
                unchanged — overwriting them with the *new* action is the
                caller's job (:meth:`write_prev_action`).
            obs: Observation batch of shape ``(batch, *obs_shape)``.  Any
                dims beyond the batch axis are flattened before the linear
                projection.

        Returns:
            ``(new_carry, y)`` where both are flat arrays;
            ``y`` has shape ``(batch, output_dim)``.
        """
        x = obs.reshape(obs.shape[0], -1)
        if self.prev_action_dim:
            memory_carry = carry[..., : -self.prev_action_dim]
            prev_action = carry[..., -self.prev_action_dim :]
            x = jnp.concatenate([x, prev_action], axis=-1)
        else:
            memory_carry = carry
        if self.projection is not None:
            x = self.projection(x)
        if self.cell is None:
            return carry, x
        if self.memory_type == "lstm":
            c, h = jnp.split(memory_carry, 2, axis=-1)
            (new_c, new_h), out = self.cell((c, h), x)
            new_memory_carry = jnp.concatenate([new_c, new_h], axis=-1)
        else:
            new_memory_carry, out = self.cell(memory_carry, x)
        if self.prev_action_dim:
            return jnp.concatenate([new_memory_carry, prev_action], axis=-1), out
        return new_memory_carry, out

    @property
    def carry_dim(self) -> int:
        """Width of the flat carry vector.

        ``0`` for identity, ``memory_hidden_dim`` for RNN/GRU, and
        ``2 * memory_hidden_dim`` for LSTM (concatenated ``[c, h]``), plus
        ``prev_action_dim`` trailing entries for the prev-action encoding
        when enabled.
        """
        if self.memory_type == "identity":
            memory_dim = 0
        elif self.memory_type == "lstm":
            memory_dim = 2 * self._memory_hidden_dim
        else:
            memory_dim = self._memory_hidden_dim
        return memory_dim + self.prev_action_dim

    def write_prev_action(
        self, carry: jax.Array, encoded_action: jax.Array
    ) -> jax.Array:
        """Overwrite the prev-action tail of a carry.

        Use after every executed action so the next :meth:`__call__` sees it.
        Callers that execute a *different* action than the one used to build
        the carry (e.g. epsilon-greedy overrides on top of ``predict``) must
        call this with the actually-executed action's encoding.

        Args:
            carry: Carry of shape ``(..., carry_dim)``.
            encoded_action: Encoding of the executed action, shape
                ``(..., prev_action_dim)`` (one-hot for discrete, raw values
                for continuous).

        Returns:
            Carry with the trailing ``prev_action_dim`` entries replaced.
            No-op when ``prev_action_dim == 0``.
        """
        if not self.prev_action_dim:
            return carry
        return carry.at[..., -self.prev_action_dim :].set(encoded_action)

    def initialize_carry(self, batch_size: int) -> jax.Array:
        """Construct a zero carry matching this extractor's memory cell.

        Always returns a single flat array of shape
        ``(batch_size, carry_dim)``; LSTM's ``(c, h)`` split is handled
        internally by :meth:`__call__`.

        Args:
            batch_size: Leading dim of the carry (one carry per env / row in
                a sample batch).

        Returns:
            Zero array of shape ``(batch_size, carry_dim)``.
            ``carry_dim`` is ``0`` for identity, ``memory_hidden_dim`` for
            RNN/GRU, and ``2 * memory_hidden_dim`` for LSTM, plus
            ``prev_action_dim`` when the prev-action input is enabled (the
            zero tail encodes "no previous action").
        """
        return jnp.zeros((batch_size, self.carry_dim), dtype=jnp.float32)


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
    projection_dim: int | None = 256,
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
    use_gvd: bool = False,
    gvd_feature_fn: Callable[[jax.Array], jax.Array] | None = None,
    gvd_sf_dims: tuple[int, ...] = (128,),
    debug: bool = False,
    seed: int = 0,
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
        projection_dim: Width of the linear embedding applied to the
            flattened observation before the recurrent cell (no activation).
            Pass ``None`` to skip the projection and feed the raw flattened
            observation straight into the memory cell.
        memory_type: ``"identity"`` (no recurrency), ``"rnn"``, ``"gru"`` or
            ``"lstm"``.  Selects the recurrent cell that follows the linear
            projection.
        memory_hidden_dim: Hidden-state width for the recurrent cell.
            Ignored when ``memory_type="identity"``.
        use_prev_action: If True, feed the previously executed action into
            the feature extractor alongside the observation (one-hot for
            discrete, raw values for continuous), mirroring the
            ``ActionConcatWrapper`` of the original lambda-discrepancy code.
            The carry grows by ``env_spec.action_dim`` trailing entries that
            hold the action encoding (zeros at episode start).
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
        use_gvd: If True, enable the General Value Discrepancy branches: two
            successor-feature V-heads trained on ``gvd_feature_fn``-difference
            cumulants whose squared discrepancy regularises the shared FE
            (reward-free).  Discrete-only.  Forwarded verbatim to
            :func:`create_iqlearn`.
        gvd_feature_fn: Pure function mapping the **raw** observation (before
            ``obs_fn`` — same convention as ``mask_fn``) to a feature vector
            ``(*batch, n_features)``.  Required when ``use_gvd=True``.
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
    input_dim = math.prod(fe_obs_shape)
    key = jax.random.key(seed)
    key_fe, key_heads = jax.random.split(key)
    feature_extractor = RecurrentFeatureExtractor(
        input_dim=input_dim,
        projection_dim=projection_dim,
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
        use_gvd=use_gvd,
        gvd_feature_fn=gvd_feature_fn,
        gvd_sf_dims=gvd_sf_dims,
        debug=debug,
    )
