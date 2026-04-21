"""Environment-interface utilities for IQ-Learn.

Provides low-level helpers that extract the observation shape, action
dimensionality, and action bounds from gymnasium, gymnax, and jumanji
environments into a common :class:`EnvSpec` NamedTuple, and a high-level
convenience function :func:`create_iqlearn_from_env` that builds a ready-to-use
IQ-Learn agent from an ``EnvSpec`` and a dict of expert transitions.

All three environment libraries are imported lazily, so only the library you
actually use needs to be installed.

Both continuous (``Box``-style) and discrete action spaces are supported.
For discrete spaces, ``action_low`` and ``action_high`` in the returned
:class:`EnvSpec` are ``None``; ``action_dim`` holds the number of actions.
Observation specs must be flat (single array); nested jumanji observation
specs raise :exc:`ValueError` — flatten the observation manually before using
this module.

Typical usage (gymnasium, continuous)::

    import gymnasium as gym
    from lambda_imitation.utils import env_spec_from_gymnasium, create_iqlearn_from_env

    env      = gym.make("HalfCheetah-v5")
    spec     = env_spec_from_gymnasium(env)
    state, fns, graphs = create_iqlearn_from_env(spec, expert_data)

Typical usage (gymnasium, discrete)::

    env  = gym.make("CartPole-v1")
    spec = env_spec_from_gymnasium(env)   # is_discrete=True
    state, fns, graphs = create_iqlearn_from_env(spec, expert_data)

Typical usage (gymnax)::

    import gymnax
    from lambda_imitation.utils import env_spec_from_gymnax, create_iqlearn_from_env

    env, params = gymnax.make("Pendulum-v1")
    spec        = env_spec_from_gymnax(env, params)
    state, fns, graphs = create_iqlearn_from_env(spec, expert_data)

Typical usage (jumanji)::

    import jumanji
    from lambda_imitation.utils import env_spec_from_jumanji, create_iqlearn_from_env

    env  = jumanji.make("Ant-v1")
    spec = env_spec_from_jumanji(env)
    state, fns, graphs = create_iqlearn_from_env(spec, expert_data)
"""

import math
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from .buffer import Buffer, create_buffer
from .iqlearn import (
    Hyperparameters,
    IQLearnFunctions,
    IQLearnGraphs,
    IQLearnState,
    MLPFeatureExtractor,
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
            action_low=jnp.asarray(act_space.low.reshape(action_dim), dtype=jnp.float32),
            action_high=jnp.asarray(act_space.high.reshape(action_dim), dtype=jnp.float32),
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

    # Reject nested/composite observation specs
    if not isinstance(obs_spec, specs.Array):
        raise ValueError(
            f"env_spec_from_jumanji requires a flat Array observation spec, "
            f"got {type(obs_spec).__name__}. Flatten the observation manually "
            f"before using this utility."
        )

    # MultiDiscreteArray is unsupported (factored categorical needs a different policy)
    if isinstance(act_spec, specs.MultiDiscreteArray):
        raise ValueError(
            f"env_spec_from_jumanji does not support MultiDiscreteArray action "
            f"specs. Flatten or handle this action space manually."
        )

    # DiscreteArray (subclass of BoundedArray — check before BoundedArray)
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
    hp: Hyperparameters = None,
    fe_hidden_dims: tuple[int, ...] = (256, 256),
    actor_head_dims: tuple[int, ...] = (),
    critic_head_dims: tuple[int, ...] = (256, 256),
    train_steps: int = 1000,
    obs_key: str = "observations",
    action_key: str = "actions",
    seed: int = 0,
) -> Tuple[IQLearnState, IQLearnFunctions, IQLearnGraphs]:
    """Build a ready-to-use IQ-Learn agent from an :class:`EnvSpec`.

    This is a high-level convenience wrapper around :func:`~lambda_imitation.iqlearn.create_iqlearn`
    that handles:

    * For **continuous** spaces: computing ``action_scale`` / ``action_bias``
      to map the actor's ``[-1, 1]`` tanh output to the environment's actual
      action range; setting ``target_entropy`` to ``-action_dim`` by default.
    * For **discrete** spaces: setting ``target_entropy`` to
      ``0.98 * log(num_actions)`` by default (Christodoulou 2019);
      ``action_scale`` and ``action_bias`` are not used.
    * Creating and pre-filling a replay :class:`~lambda_imitation.buffer.Buffer` from
      ``expert_data``.
    * Constructing three :class:`~lambda_imitation.iqlearn.MLPFeatureExtractor` instances
      (one for the actor and one per critic Q-branch) with identical architecture.

    The expert data is stored as-is; no normalisation is applied.  If
    ``expert_data`` contains more entries than ``buffer_size``, the last
    ``buffer_size`` transitions are kept (the buffer wraps circularly).

    Args:
        env_spec: Environment dimensions produced by one of the
            ``env_spec_from_*`` extractors.
        expert_data: Dict mapping string keys to arrays of shape
            ``(N, *item_shape)``, where ``N`` is the number of expert
            transitions.  Must contain at least the keys ``obs_key`` and
            ``action_key``.  For discrete spaces, actions should be stored as
            float32 scalars (shape ``(N, 1)``), with action indices as floats
            (e.g. ``0.0``, ``1.0``, ``2.0``).
        buffer_size: Maximum number of transitions the replay buffer can hold.
        hp: :class:`~lambda_imitation.iqlearn.Hyperparameters` instance.  When ``None``,
            defaults are used with ``target_entropy`` set appropriately for
            the action space type.
        fe_hidden_dims: Hidden layer widths for the actor and critic
            feature extractors.  All three share the same architecture;
            construct them manually and call
            :func:`~lambda_imitation.iqlearn.create_iqlearn` directly if you need
            different backbones.
        actor_head_dims: Hidden layer widths for the actor head.  Defaults to
            ``()`` (direct linear projection).
        critic_head_dims: Hidden layer widths for the critic head.  Defaults
            to ``(256, 256)``.
        train_steps: Number of gradient steps per :func:`~lambda_imitation.iqlearn.IQLearnFunctions.train`
            call.
        obs_key: Key in ``expert_data`` (and in the buffer) that holds
            observations.
        action_key: Key in ``expert_data`` (and in the buffer) that holds
            actions.
        seed: Integer seed passed to ``nnx.Rngs`` when constructing the
            feature extractors.

    Returns:
        A ``(IQLearnState, IQLearnFunctions, IQLearnGraphs)`` triple ready for
        training; see :func:`~lambda_imitation.iqlearn.create_iqlearn` for details.

    Raises:
        ValueError: If ``expert_data`` does not contain ``obs_key`` or
            ``action_key``.
        ValueError: If any expert data array has a leading dimension of zero.
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
        # Default target_entropy = 0.98 * log(num_actions) (Christodoulou 2019)
        if hp is None:
            hp = Hyperparameters(
                target_entropy=float(0.98 * math.log(env_spec.action_dim))
            )
        action_scale = 1.0
        action_bias = 0.0
    else:
        # Default target_entropy = -action_dim for continuous
        if hp is None:
            hp = Hyperparameters(target_entropy=float(-env_spec.action_dim))
        # action_scale / action_bias map tanh output in [-1, 1] to [low, high]:
        #   action = tanh(raw) * scale + bias
        #   scale  = (high - low) / 2
        #   bias   = (high + low) / 2
        action_scale = (env_spec.action_high - env_spec.action_low) / 2.0
        action_bias  = (env_spec.action_high + env_spec.action_low) / 2.0

    # Build shapes dict from expert_data (strip leading N dimension)
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

    # Feature extractors — identical architecture, independent weights
    input_dim = math.prod(env_spec.obs_shape)
    rngs = nnx.Rngs(seed)
    actor_fe    = MLPFeatureExtractor(input_dim, fe_hidden_dims, rngs=rngs)
    critic_q1_fe = MLPFeatureExtractor(input_dim, fe_hidden_dims, rngs=rngs)
    critic_q2_fe = MLPFeatureExtractor(input_dim, fe_hidden_dims, rngs=rngs)

    return create_iqlearn(
        params=hp,
        buffer=buffer,
        action_dim=env_spec.action_dim,
        actor_feature_extractor=actor_fe,
        critic_q1_feature_extractor=critic_q1_fe,
        critic_q2_feature_extractor=critic_q2_fe,
        obs_key=obs_key,
        action_key=action_key,
        action_scale=action_scale,
        action_bias=action_bias,
        train_steps=train_steps,
        actor_head_dims=actor_head_dims,
        critic_head_dims=critic_head_dims,
        is_discrete=env_spec.is_discrete,
    )
