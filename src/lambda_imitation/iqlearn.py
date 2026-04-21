"""IQ-Learn (Inverse Q-Learning) imitation learning.

Implements a SAC-style actor-critic whose reward signal is recovered from
expert demonstrations via the IQ-Learn objective (Garg et al., 2021).
Networks are split into a user-supplied *feature extractor* and an
internally-managed *head*, so the observation-processing backbone can be
swapped freely (MLP, CNN, transformer, …) without touching any IQ-Learn logic.

Both **continuous** and **discrete** action spaces are supported.  Pass
``is_discrete=True`` to :func:`create_iqlearn` to activate the discrete path,
which uses categorical distributions and an all-actions critic (no action
input to the critic; V(s) is computed as the exact inner product
``Σ_a π(a|s)·Q(s,a)`` rather than via Monte-Carlo sampling).

The twin-Q critic is implemented as two fully independent
``(FeatureExtractor, Head)`` pairs — one per Q-branch — grouped in a
:class:`TwinCriticState` NamedTuple.  Because ``TwinCriticState`` is a JAX
pytree, a single optimizer operates on both branches transparently.

All state is held in immutable NamedTuples and the functional design
(``create_iqlearn`` factory + pure ``train``/``predict`` closures) keeps the
implementation compatible with ``jax.jit`` and ``jax.lax.scan``.

Typical usage (continuous)::

    rngs = nnx.Rngs(0)
    actor_fe   = MLPFeatureExtractor(obs_dim, (256, 256), rngs=rngs)
    critic_q1_fe = MLPFeatureExtractor(obs_dim, (256, 256), rngs=rngs)
    critic_q2_fe = MLPFeatureExtractor(obs_dim, (256, 256), rngs=rngs)

    state, fns, graphs = create_iqlearn(
        Hyperparameters(), buffer, action_dim,
        actor_fe, critic_q1_fe, critic_q2_fe,
    )
    state, metrics = fns.train(state, jax.random.key(0))
    action = fns.predict(state, obs, deterministic=True)

Typical usage (discrete)::

    state, fns, graphs = create_iqlearn(
        Hyperparameters(), buffer, num_actions,
        actor_fe, critic_q1_fe, critic_q2_fe,
        is_discrete=True,
    )
    action = fns.predict(state, obs, deterministic=True)  # float32 scalar index
"""

from functools import partial
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .buffer import Buffer, BufferSample, create_sample

# Bounds for the squashed log-standard-deviation of the policy distribution.
# The raw output is tanh-squashed and then rescaled into this range to keep
# the distribution numerically stable while remaining expressive.
LOG_STD_MIN = -5
LOG_STD_MAX = 2


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------


class MLPFeatureExtractor(nnx.Module):
    """Configurable MLP observation encoder.

    Flattens the input (to handle arbitrary observation shapes) and passes it
    through a sequence of linear layers with ReLU activations.  The output
    dimension equals ``hidden_dims[-1]``.

    Args:
        input_dim: Flat size of one observation (product of all spatial dims).
        hidden_dims: Width of each hidden layer.  Defaults to ``(256, 256)``.
            An empty tuple would make this a no-op identity, which is only
            meaningful if the downstream head is large enough.
        rngs: Flax NNX RNG container used to initialise parameters.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        *,
        rngs: nnx.Rngs,
    ):
        dims = [input_dim] + list(hidden_dims)
        self.layers = [
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        """Encode a batch of observations.

        Args:
            x: Observation batch of shape ``(batch, *obs_shape)``.  Any shape
                beyond the batch dimension is flattened automatically.

        Returns:
            Feature tensor of shape ``(batch, hidden_dims[-1])``.
        """
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = nnx.relu(layer(x))
        return x


class Head(nnx.Module):
    """Generic MLP head: ReLU on hidden layers, linear output.

    Takes a pre-computed feature vector (no flattening) and maps it to an
    output of the desired dimensionality.  Hidden layers use ReLU activations;
    the final layer is a plain linear projection.

    This single class replaces the former ``ActorHead``, ``CriticHead``,
    ``DiscreteActorHead``, and ``DiscreteCriticHead`` specialisations.  The
    caller controls the role by choosing the appropriate ``output_dim``:

    * **Continuous actor**: ``output_dim = 2 * action_dim``
      (mean + log-std of a squashed Gaussian).
    * **Discrete actor**: ``output_dim = num_actions`` (categorical logits).
    * **Continuous critic Q1 / Q2**: ``output_dim = 1``; the caller
      concatenates features and actions *before* passing them in.
    * **Discrete critic Q1 / Q2**: ``output_dim = num_actions``
      (per-action Q-values).

    Args:
        feature_dim: Dimensionality of the input feature vector.
        hidden_dims: Widths of optional hidden layers.  Use ``()`` for a
            direct linear projection from features to output.
        output_dim: Dimensionality of the output.
        rngs: Flax NNX RNG container used to initialise parameters.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        dims = [feature_dim] + list(hidden_dims) + [output_dim]
        self.layers = [
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        """Map a feature batch to the output space.

        Args:
            x: Feature batch of shape ``(batch, feature_dim)``.

        Returns:
            Output array of shape ``(batch, output_dim)``.
        """
        for layer in self.layers[:-1]:
            x = nnx.relu(layer(x))
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# State / function / graph containers
# ---------------------------------------------------------------------------


class NetworkState(NamedTuple):
    """Flax NNX graph states for a feature-extractor + head pair.

    Both fields are ``nnx.GraphState`` objects produced by ``nnx.split``.
    Together they form a JAX pytree, so optimizer updates and EMA target
    updates work on them transparently via ``jax.tree.map``.

    Attributes:
        fe: Graph state of the feature extractor module.
        head: Graph state of the task-specific head module.
    """

    fe: nnx.GraphState
    head: nnx.GraphState


class TwinCriticState(NamedTuple):
    """Paired network states for the two independent Q-branches.

    Grouping both branches in a single NamedTuple (which is a JAX pytree)
    lets a single optimizer and a single ``jax.grad`` call operate over both
    branches simultaneously without any changes to the loss/update logic.

    Attributes:
        q1: Network state (FE + head) for the first Q-branch.
        q2: Network state (FE + head) for the second Q-branch.
    """

    q1: NetworkState
    q2: NetworkState


class NetworkGraphs(NamedTuple):
    """Flax NNX graph definitions for a feature-extractor + head pair.

    These are the static (non-parameter) graph descriptions produced by
    ``nnx.split`` and consumed by ``nnx.merge`` to reconstruct live modules
    during forward passes.

    Attributes:
        fe: Graph definition of the feature extractor.
        head: Graph definition of the head.
    """

    fe: nnx.GraphDef
    head: nnx.GraphDef


class IQLearnState(NamedTuple):
    """Complete, serialisable state of one IQ-Learn agent.

    All fields are JAX pytrees, so the entire state can be checkpointed,
    passed through ``jax.jit``, or stacked for vectorised environments.

    Attributes:
        actor: Online actor network state (feature extractor + head).
        critic: Online twin-critic state (two independent Q-branches).
        actor_target: EMA-smoothed copy of the actor, used as a stable target
            during critic updates.
        critic_target: EMA-smoothed copy of the critic, used for bootstrapping
            next-state values.
        actor_optimizer_state: Optax state for the actor Adam optimiser.
        critic_optimizer_state: Optax state for the critic Adam optimiser;
            operates on the full :class:`TwinCriticState` pytree.
        alpha_optimizer_state: Optax state for the entropy temperature optimiser.
        alpha: Current entropy temperature (``exp(log_alpha)``).
        log_alpha: Log-space entropy temperature; directly optimised to avoid
            a positivity constraint.
    """

    actor: NetworkState
    critic: TwinCriticState
    actor_target: NetworkState
    critic_target: TwinCriticState
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    alpha: jax.Array
    log_alpha: jax.Array
    online_buffer: Buffer
    online_buffer_functions: BufferFunctions


class IQLearnFunctions(NamedTuple):
    """Pure functions returned by :func:`create_iqlearn`.

    Attributes:
        predict: ``(state, obs, key, deterministic) -> action`` -- sample or
            compute a deterministic action for a single observation.
        train: ``(state, key) -> (state, metrics)`` -- run ``train_steps``
            update iterations via ``jax.lax.scan`` and return averaged metrics.
    """

    predict: Callable
    train: Callable


class IQLearnGraphs(NamedTuple):
    """Flax NNX graph definitions for all network modules.

    These are the static (non-parameter) descriptions produced by
    ``nnx.split`` and consumed by ``nnx.merge`` to reconstruct live modules
    during forward passes.  Returned by :func:`create_iqlearn` for callers
    that need direct access to the graph structure (e.g. for inspection or
    custom inference code).

    Attributes:
        actor: Graph definitions for the actor (FE + head).
        critic_q1: Graph definitions for the first critic Q-branch (FE + head).
        critic_q2: Graph definitions for the second critic Q-branch (FE + head).
    """

    actor: NetworkGraphs
    critic_q1: NetworkGraphs
    critic_q2: NetworkGraphs


class Hyperparameters(NamedTuple):
    """Training hyperparameters for IQ-Learn.

    All fields have sensible defaults so callers only need to override what
    differs from the standard SAC/IQ-Learn setup.

    Attributes:
        actor_lr: Learning rate for the actor Adam optimiser.
        critic_lr: Learning rate for the critic Adam optimiser.
        alpha_lr: Learning rate for the entropy temperature Adam optimiser.
        alpha: Initial entropy temperature.  Ignored when ``autotune_alpha``
            is True after the first update.
        autotune_alpha: If True, alpha is continuously adjusted to match
            ``target_entropy``.  If False, alpha is held fixed at its initial
            value throughout training.
        batch_size: Number of transitions sampled per gradient step.
        gamma: Discount factor for future rewards.
        regularizer_coef: Weight of the IQ-Learn soft-regularisation term
            (``1/40`` in the original paper).
        target_entropy: Desired policy entropy used by the alpha loss.  For
            continuous spaces a common heuristic is ``-action_dim``; for
            discrete spaces ``0.98 * log(num_actions)`` (Christodoulou 2019).
        tau: Soft update coefficient for EMA target networks.  A value of
            ``0.005`` means targets lag significantly behind online weights.
    """

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-3
    alpha: float = 1.0
    autotune_alpha: bool = True
    batch_size: int = 256
    gamma: float = 0.99
    regularizer_coef: float = 1 / 40
    target_entropy: float = -1
    tau: float = 0.005


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_iqlearn(
    params: Hyperparameters,
    buffer: Buffer,
    action_dim: int,
    actor_feature_extractor: nnx.Module,
    critic_q1_feature_extractor: nnx.Module,
    critic_q2_feature_extractor: nnx.Module,
    obs_key: str = "observations",
    action_key: str = "actions",
    action_scale: float | jax.Array = 1,
    action_bias: float | jax.Array = 0,
    train_steps: int = 1000,
    actor_head_dims: tuple[int, ...] = (),
    critic_head_dims: tuple[int, ...] = (256, 256),
    is_discrete: bool = False,
) -> Tuple[IQLearnState, IQLearnFunctions, IQLearnGraphs]:
    """Construct an IQ-Learn agent from a pre-filled buffer and user-supplied FEs.

    The feature extractors are taken as-is (already initialised by the caller),
    split into graph definition + parameter state via ``nnx.split``, and frozen
    inside the returned closures.  Actor and critic heads are created internally
    from ``actor_head_dims``/``critic_head_dims``; their input dimension is
    inferred automatically by running a dummy forward pass through each feature
    extractor.

    The twin-Q critic is implemented as two fully independent
    ``(FeatureExtractor, Head)`` pairs.  Pass separate, independently-seeded
    feature extractors as ``critic_q1_feature_extractor`` and
    ``critic_q2_feature_extractor`` so that the two Q-branches diverge from
    the very first gradient step.

    The returned ``train`` function runs ``train_steps`` gradient steps per call
    using ``jax.lax.scan``, keeping the whole loop JIT-compiled after the first
    invocation.

    Args:
        params: Hyperparameters controlling learning rates, discount, alpha, etc.
        buffer: Filled (or partially filled) replay buffer.  Must contain at
            least ``params.batch_size`` sampleable slots before ``train`` is
            called.
        action_dim: Number of continuous action dimensions, or number of
            discrete actions when ``is_discrete=True``.
        actor_feature_extractor: Initialised ``nnx.Module`` that maps
            ``(batch, *obs_shape) -> (batch, actor_feature_dim)``.  Ownership
            is transferred; the module is split and should not be used directly
            afterwards.
        critic_q1_feature_extractor: Same contract as ``actor_feature_extractor``.
            Used exclusively by the first Q-branch.
        critic_q2_feature_extractor: Same contract as ``actor_feature_extractor``.
            Used exclusively by the second Q-branch.  Should be initialised with
            a different seed from ``critic_q1_feature_extractor`` to ensure the
            two branches start with different weights.
        obs_key: Key in ``buffer.info`` that holds observations.
        action_key: Key in ``buffer.info`` that holds actions.
        action_scale: Per-dimension scale applied after the tanh squashing
            (continuous only).  Scalar or array of shape ``(action_dim,)``.
        action_bias: Per-dimension offset applied after the tanh squashing
            (continuous only).  Scalar or array of shape ``(action_dim,)``.
        train_steps: Number of gradient steps executed per ``train`` call.
        actor_head_dims: Hidden layer widths for the actor head.  Defaults to
            ``()`` (direct linear projection from features to outputs).
        critic_head_dims: Hidden layer widths for each critic head.  Defaults to
            ``(256, 256)``.  Applied identically to both Q-branches.
        is_discrete: If True, use a categorical actor and an all-actions critic.
            The soft value V(s) is computed as the exact closed-form inner
            product ``Σ_a π(a|s)·(Q(s,a) − α·log π(a|s))`` without sampling.
            ``predict`` returns the action index as a ``float32`` scalar.
            If False (default), use a squashed-Gaussian actor and a continuous
            critic that takes actions as additional input.

    Returns:
        A ``(IQLearnState, IQLearnFunctions, IQLearnGraphs)`` triple.

        - ``IQLearnState``: initial agent state with online and target networks
          set to the same weights.
        - ``IQLearnFunctions``: named tuple of ``predict`` and ``train`` closures.
        - ``IQLearnGraphs``: static NNX graph definitions for all network modules,
          useful for inspection or custom inference.
    """
    buffer_sample = create_sample(
        buffer.size,
        params.batch_size,
        this_keys=[obs_key, action_key],
        next_keys=[obs_key],
    )

    # Infer feature dims via dummy forward pass (before split)
    dummy_obs = jnp.zeros((1,) + buffer.info[obs_key].shape[1:])
    actor_feature_dim = actor_feature_extractor(dummy_obs).shape[-1]
    critic_q1_feature_dim = critic_q1_feature_extractor(dummy_obs).shape[-1]
    critic_q2_feature_dim = critic_q2_feature_extractor(dummy_obs).shape[-1]

    # Create heads — discrete and continuous differ only in output_dim and
    # whether actions are concatenated to features before the head.
    rngs = nnx.Rngs(0)
    if is_discrete:
        actor_head_model = Head(
            actor_feature_dim,
            actor_head_dims,
            action_dim,
            rngs=rngs,
        )
        critic_q1_head_model = Head(
            critic_q1_feature_dim,
            critic_head_dims,
            action_dim,
            rngs=rngs,
        )
        critic_q2_head_model = Head(
            critic_q2_feature_dim,
            critic_head_dims,
            action_dim,
            rngs=rngs,
        )
    else:
        actor_head_model = Head(
            actor_feature_dim,
            actor_head_dims,
            2 * action_dim,
            rngs=rngs,
        )
        # For continuous critics, features and actions are concatenated before
        # the head, so input_dim = feature_dim + action_dim, output_dim = 1.
        critic_q1_head_model = Head(
            critic_q1_feature_dim + action_dim,
            critic_head_dims,
            1,
            rngs=rngs,
        )
        critic_q2_head_model = Head(
            critic_q2_feature_dim + action_dim,
            critic_head_dims,
            1,
            rngs=rngs,
        )

    # Split all six modules into (graph_def, state)
    actor_fe_graph, actor_fe_st = nnx.split(actor_feature_extractor)
    actor_head_graph, actor_head_st = nnx.split(actor_head_model)
    critic_q1_fe_graph, critic_q1_fe_st = nnx.split(critic_q1_feature_extractor)
    critic_q1_head_graph, critic_q1_head_st = nnx.split(critic_q1_head_model)
    critic_q2_fe_graph, critic_q2_fe_st = nnx.split(critic_q2_feature_extractor)
    critic_q2_head_graph, critic_q2_head_st = nnx.split(critic_q2_head_model)

    actor_state = NetworkState(actor_fe_st, actor_head_st)
    critic_q1_state = NetworkState(critic_q1_fe_st, critic_q1_head_st)
    critic_q2_state = NetworkState(critic_q2_fe_st, critic_q2_head_st)
    critic_state = TwinCriticState(critic_q1_state, critic_q2_state)

    # Optimizers: actor operates on NetworkState; critic on TwinCriticState.
    actor_optimizer = optax.adam(params.actor_lr)
    critic_optimizer = optax.adam(params.critic_lr)
    alpha_optimizer = optax.adam(params.alpha_lr)

    log_alpha = jnp.array(jnp.log(params.alpha))
    actor_optimizer_state = actor_optimizer.init(actor_state)
    critic_optimizer_state = critic_optimizer.init(critic_state)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    def remove_weak_types(state):
        """Materialise JAX weak dtypes to concrete dtypes to avoid JIT issues."""
        return jax.tree.map(
            lambda x: jnp.array(x, dtype=x.dtype) if hasattr(x, "dtype") else x,
            state,
        )

    iqlearn = IQLearnState(
        remove_weak_types(actor_state),
        remove_weak_types(critic_state),
        remove_weak_types(actor_state),  # targets start equal to online weights
        remove_weak_types(critic_state),
        remove_weak_types(actor_optimizer_state),
        remove_weak_types(critic_optimizer_state),
        remove_weak_types(alpha_optimizer_state),
        remove_weak_types(jnp.exp(log_alpha)),
        remove_weak_types(log_alpha),
    )

    # ------------------------------------------------------------------
    # Shared helper: actor forward pass (same for both action space types)
    # ------------------------------------------------------------------

    def run_actor(actor: NetworkState, x: jax.Array) -> jax.Array:
        """Reconstruct and run the actor (FE then head) on observation batch x."""
        fe = nnx.merge(actor_fe_graph, actor.fe)
        head = nnx.merge(actor_head_graph, actor.head)
        return head(fe(x))

    # ------------------------------------------------------------------
    # Action-space-specific helpers
    # ------------------------------------------------------------------

    if is_discrete:

        def run_critic(critic: TwinCriticState, x: jax.Array) -> jax.Array:
            """Reconstruct and run both discrete critic branches.

            Returns:
                Array of shape ``(batch, num_actions, 2)`` where the last axis
                indexes the two independent Q estimates.
            """
            fe1 = nnx.merge(critic_q1_fe_graph, critic.q1.fe)
            head1 = nnx.merge(critic_q1_head_graph, critic.q1.head)
            fe2 = nnx.merge(critic_q2_fe_graph, critic.q2.fe)
            head2 = nnx.merge(critic_q2_head_graph, critic.q2.head)
            q1 = head1(fe1(x))  # (batch, num_actions)
            q2 = head2(fe2(x))  # (batch, num_actions)
            return jnp.stack([q1, q2], axis=-1)  # (batch, num_actions, 2)

        def get_q(
            critic: TwinCriticState, x: jax.Array, expert_actions: jax.Array
        ) -> jax.Array:
            """Conservative Q-value for each expert transition.

            Args:
                critic: Twin-critic network state.
                x: Observation batch.
                expert_actions: Float32 array of shape ``(batch, 1)`` holding
                    action indices stored as floats (e.g. 0.0, 1.0, 2.0).

            Returns:
                Per-transition Q-value of shape ``(batch,)``.
            """
            q_twin = run_critic(critic, x)  # (batch, num_actions, 2)
            q_min = jnp.minimum(q_twin[..., 0], q_twin[..., 1])  # (batch, num_actions)
            indices = jnp.round(expert_actions.reshape(-1)).astype(jnp.int32)
            return q_min[jnp.arange(q_min.shape[0]), indices]  # (batch,)

        def get_v(
            actor: NetworkState,
            critic: TwinCriticState,
            alpha: jax.Array,
            x: jax.Array,
            key: jax.Array,
            include_entropy: bool = True,
            include_log: bool = False,
        ) -> jax.Array | Tuple[jax.Array, dict]:
            """Compute the soft value V(x) = Σ_a π(a|x)·(Q(x,a) − α·log π(a|x)).

            Uses the exact closed-form inner product over all actions; no PRNG
            sampling is required.  The ``key`` argument is accepted for API
            compatibility with the continuous path but is not used.

            Args:
                actor: Actor network state.
                critic: Twin-critic network state.
                alpha: Current entropy temperature.
                x: Observation batch.
                key: Unused PRNG key (present for API compatibility).
                include_entropy: If True, add the entropy term ``α·H(π)`` to
                    the expected Q to obtain the soft value.
                include_log: If True, also return a metrics dict with scalar
                    summaries ``{"q": ..., "entropy": ...}``.  Only meaningful
                    when ``include_entropy`` is True.

            Returns:
                When ``include_log=False``: value array of shape ``(batch,)``.
                When ``include_log=True``: ``(values, metrics_dict)``.
            """
            logits = run_actor(actor, x)  # (batch, num_actions)
            probs = jax.nn.softmax(logits)  # (batch, num_actions)
            log_probs = jax.nn.log_softmax(logits)  # (batch, num_actions)
            q_twin = run_critic(critic, x)  # (batch, num_actions, 2)
            q_min = jnp.minimum(q_twin[..., 0], q_twin[..., 1])  # (batch, num_actions)
            entropy = -(probs * log_probs).sum(-1)  # (batch,) — exact H(π)
            if include_entropy:
                v = (probs * q_min).sum(-1) + alpha * entropy
                if include_log:
                    return v, {
                        "q": (probs * q_min).sum(-1).mean(),
                        "entropy": entropy.mean(),
                    }
                return v
            else:
                return (probs * q_min).sum(-1)

        @partial(jax.jit, static_argnames=["deterministic"])
        def predict(
            iqlearn: IQLearnState,
            obs: jax.Array,
            key: jax.Array = jnp.array(0),
            deterministic: bool = False,
        ) -> jax.Array:
            """Compute a discrete action for a single observation.

            Args:
                iqlearn: Current agent state.
                obs: Single observation of shape ``(*obs_shape,)`` (no batch dim).
                key: JAX PRNG key, used only when ``deterministic=False``.
                deterministic: If True, return the greedy (argmax) action.
                    If False, sample from the categorical policy.

            Returns:
                Action index as a ``float32`` scalar.
            """
            obs_batch = jnp.expand_dims(obs, 0)
            logits = run_actor(iqlearn.actor, obs_batch)[0]  # (num_actions,)
            if deterministic:
                return jnp.argmax(logits).astype(jnp.float32)
            else:
                return jax.random.categorical(key, logits).astype(jnp.float32)

    else:
        # ------------------------------------------------------------------
        # Continuous helpers
        # ------------------------------------------------------------------

        def run_critic(
            critic: TwinCriticState, x: jax.Array, actions: jax.Array
        ) -> jax.Array:
            """Reconstruct and run both continuous critic branches.

            Features and actions are concatenated before each head so each
            branch has a fully independent view of the (obs, action) pair.

            Returns:
                Array of shape ``(batch, 2)`` containing two independent Q estimates.
            """
            fe1 = nnx.merge(critic_q1_fe_graph, critic.q1.fe)
            head1 = nnx.merge(critic_q1_head_graph, critic.q1.head)
            fe2 = nnx.merge(critic_q2_fe_graph, critic.q2.fe)
            head2 = nnx.merge(critic_q2_head_graph, critic.q2.head)
            q1 = head1(jnp.concat((fe1(x), actions), axis=-1))  # (batch, 1)
            q2 = head2(jnp.concat((fe2(x), actions), axis=-1))  # (batch, 1)
            return jnp.concat([q1, q2], axis=-1)  # (batch, 2)

        def get_q(
            critic: TwinCriticState, x: jax.Array, actions: jax.Array
        ) -> jax.Array:
            """Return the conservative (min over twin) Q-value for each transition."""
            return jnp.min(run_critic(critic, x, actions), axis=-1)

        def get_dist_params(
            actor: NetworkState, x: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
            """Extract mean and std of the squashed Gaussian policy.

            The raw log-std output is tanh-squashed and rescaled into
            ``[LOG_STD_MIN, LOG_STD_MAX]`` for numerical stability.

            Returns:
                ``(mean, std)`` each of shape ``(batch, action_dim)``.
            """
            dist_params = run_actor(actor, x)
            mean, log_std = (
                dist_params[..., :action_dim],
                dist_params[..., action_dim:],
            )
            log_std = jnp.tanh(log_std)
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
            std = jnp.exp(log_std)
            return mean, std

        def sample_action_logprob(
            actor: NetworkState, x: jax.Array, key: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
            """Sample actions and compute their log-probabilities under the policy.

            Uses the reparameterisation trick: samples from N(mean, std) then
            applies tanh squashing followed by optional affine rescaling.  The
            log-probability accounts for the tanh change-of-variables (Appendix C
            of the SAC paper).

            Args:
                actor: Current actor network state.
                x: Observation batch of shape ``(batch, *obs_shape)``.
                key: JAX PRNG key for sampling.

            Returns:
                ``(actions, log_probs)`` where ``actions`` has shape
                ``(batch, action_dim)`` and ``log_probs`` has shape ``(batch,)``.
            """
            mean, std = get_dist_params(actor, x)
            unsquashed_action = jax.random.normal(key, mean.shape) * std + mean
            y_t = jnp.tanh(unsquashed_action)
            action = y_t * action_scale + action_bias
            log_prob = (
                -((unsquashed_action - mean) ** 2) / (2 * std**2)
                - 0.5 * jnp.log(2 * jnp.pi)
                - jnp.log(std)
                - jnp.log(action_scale * (1 - y_t**2) + 1e-6)
            )
            return action, log_prob.sum(axis=-1)

        def get_v(
            actor: NetworkState,
            critic: TwinCriticState,
            alpha: jax.Array,
            x: jax.Array,
            key: jax.Array,
            include_entropy: bool = True,
            include_log: bool = False,
        ) -> jax.Array | Tuple[jax.Array, dict]:
            """Compute the soft state-value V(x) = E_π[Q(x,a) - α log π(a|x)].

            Args:
                actor: Actor network state used to sample actions.
                critic: Twin-critic network state used to evaluate Q-values.
                alpha: Current entropy temperature.
                x: Observation batch.
                key: JAX PRNG key for action sampling.
                include_entropy: If True, subtract the entropy term ``α log π``
                    from Q to obtain the soft value.  If False, return raw Q.
                include_log: If True, also return a metrics dict with scalar
                    summaries ``{"q": ..., "entropy": ...}``.  Only meaningful
                    when ``include_entropy`` is True.

            Returns:
                When ``include_log=False``: value array of shape ``(batch,)``.
                When ``include_log=True``: ``(values, metrics_dict)``.
            """
            action, logprob = sample_action_logprob(actor, x, key)
            q = get_q(critic, x, action)
            if include_entropy:
                if include_log:
                    return q - alpha * logprob, {
                        "q": q.mean(),
                        "entropy": -logprob.mean(),
                    }
                else:
                    return q - alpha * logprob
            else:
                return q

        @partial(jax.jit, static_argnames=["deterministic"])
        def predict(
            iqlearn: IQLearnState,
            obs: jax.Array,
            key: jax.Array = jnp.array(0),
            deterministic: bool = False,
        ) -> jax.Array:
            """Compute an action for a single observation.

            Args:
                iqlearn: Current agent state.
                obs: Single observation of shape ``(*obs_shape,)`` (no batch dim).
                key: JAX PRNG key, used only when ``deterministic=False``.
                deterministic: If True, return the tanh-squashed policy mean
                    (no sampling noise).  If False, sample from the full
                    Gaussian policy.

            Returns:
                Action array of shape ``(action_dim,)``, scaled and shifted by
                ``action_scale`` and ``action_bias``.
            """
            obs = jnp.expand_dims(obs, 0)
            mean, std = get_dist_params(iqlearn.actor, obs)
            if deterministic:
                unsquashed_action = mean
            else:
                unsquashed_action = jax.random.normal(key, mean.shape) * std + mean
            y_t = jnp.tanh(unsquashed_action)
            action = y_t * action_scale + action_bias
            return action[0]

    # ------------------------------------------------------------------
    # Loss functions — structurally identical for both action space types;
    # differences are absorbed by the action-space-specific helpers above.
    # ------------------------------------------------------------------

    def loss_alpha(log_alpha: jax.Array, log_pi: jax.Array) -> jax.Array:
        """Entropy temperature loss.

        Minimising this pushes alpha so that the expected policy entropy
        matches ``params.target_entropy``.

        Args:
            log_alpha: Current log-temperature scalar.
            log_pi: Mean log-probability of the current policy (scalar).
                For continuous: sampled log-prob.  For discrete: expected
                log-prob ``Σ_a π(a|s) log π(a|s)``.

        Returns:
            Scalar loss value.
        """
        alpha_loss = -jnp.exp(log_alpha) * (log_pi + params.target_entropy)
        return alpha_loss

    def loss_actor(
        actor: NetworkState,
        critic: TwinCriticState,
        buffer: Buffer,
        buffer_sample: Callable[[Buffer, jax.Array], Tuple[BufferSample, Tuple[int]]],
        alpha: jax.Array,
        key: jax.Array,
    ) -> Tuple[jax.Array, dict]:
        """Actor loss: maximise the soft state value V(s) = Q(s,a) - α log π(a|s).

        Samples a fresh batch from the buffer and computes the negative mean
        soft value (to be minimised via gradient descent).

        Returns:
            ``(scalar_loss, metrics)`` where metrics contains ``"q"``,
            ``"entropy"``, and ``"v"``.
        """
        key_sample, key_v = jax.random.split(key, 2)
        sample, _ = buffer_sample(buffer, key_sample)
        v, metrics = get_v(
            actor,
            critic,
            alpha,
            sample.this_info[obs_key],
            key_v,
            include_entropy=True,
            include_log=True,
        )
        metrics.update({"v": v.mean()})
        return -v.mean(), metrics

    def loss_critic(
        actor_target: NetworkState,
        critic: TwinCriticState,
        critic_target: TwinCriticState,
        buffer: Buffer,
        buffer_sample: Callable[[Buffer, jax.Array], Tuple[BufferSample, Tuple[int]]],
        alpha: jax.Array,
        key: jax.Array,
    ) -> Tuple[jax.Array, dict]:
        """IQ-Learn critic loss.

        Implements the IQ-Learn objective:

        ``loss = -mean(Q(s,a) - γV(s') - V(s) + γV(s') - λ(Q² + V²))``

        where Q is evaluated on expert actions from the buffer, V uses
        the target actor for sampling, and λ is the soft regulariser
        coefficient.  The gradient is taken only w.r.t. the online critic.

        Returns:
            ``(scalar_loss, metrics)`` where metrics contains
            ``"demonstration_loss"``, ``"mixed_loss"``,
            ``"regularizer_loss"``, and ``"critic_loss"``.
        """
        key_sample, key_v, key_next_v = jax.random.split(key, 3)
        sample, _ = buffer_sample(buffer, key_sample)

        q_values = get_q(
            critic, sample.this_info[obs_key], sample.this_info[action_key]
        )
        v_values = get_v(
            actor_target,
            critic,
            alpha,
            sample.this_info[obs_key],
            key_v,
            include_entropy=True,
        )
        next_v_values = get_v(
            actor_target,
            critic_target,
            alpha,
            sample.next_info[obs_key],
            key_next_v,
            include_entropy=True,
        )

        demonstration_loss = q_values - params.gamma * next_v_values  # type: ignore
        mixed_loss = v_values - params.gamma * next_v_values  # type: ignore
        regularizer_loss = params.regularizer_coef * (
            demonstration_loss**2 + mixed_loss**2
        )

        loss = -(demonstration_loss - mixed_loss - regularizer_loss).mean()

        return loss, {
            "demonstration_loss": demonstration_loss.mean(),
            "mixed_loss": mixed_loss.mean(),
            "regularizer_loss": regularizer_loss.mean(),
            "critic_loss": loss,
        }

    def update_step(iqlearn: IQLearnState, key: jax.Array) -> Tuple[IQLearnState, dict]:
        """Execute one full SAC-style update (actor + critic + alpha + EMA targets).

        Computes gradients for the actor and critic independently, applies
        Adam updates, optionally updates alpha, and soft-updates both target
        networks via exponential moving average with coefficient ``params.tau``.

        Args:
            iqlearn: Current agent state.
            key: JAX PRNG key; split internally for actor and critic updates.

        Returns:
            ``(new_state, metrics)`` where metrics is the union of actor and
            critic metric dicts, plus ``"alpha"`` when autotune is enabled.
        """
        print("compiling...")
        key_actor, key_critic = jax.random.split(key, 2)

        # actor gradients
        grads_actor, metrics = jax.grad(loss_actor, has_aux=True)(
            iqlearn.actor,
            iqlearn.critic_target,
            buffer,
            buffer_sample,
            iqlearn.alpha,
            key_actor,
        )
        # critic gradients — grad w.r.t. TwinCriticState (both branches jointly)
        grads_critic, metrics_critic = jax.grad(loss_critic, argnums=1, has_aux=True)(
            iqlearn.actor_target,
            iqlearn.critic,
            iqlearn.critic_target,
            buffer,
            buffer_sample,
            iqlearn.alpha,
            key_critic,
        )

        metrics.update(metrics_critic)

        # update actor (fe + head jointly)
        updates, new_actor_optimizer_state = actor_optimizer.update(
            grads_actor, iqlearn.actor_optimizer_state
        )
        new_actor = optax.apply_updates(iqlearn.actor, updates)  # type: ignore

        # update critic (both Q-branches jointly via TwinCriticState pytree)
        updates, new_critic_optimizer_state = critic_optimizer.update(
            grads_critic, iqlearn.critic_optimizer_state
        )
        new_critic = optax.apply_updates(iqlearn.critic, updates)  # type: ignore

        # update alpha
        if params.autotune_alpha:
            grads_alpha = jax.grad(loss_alpha)(iqlearn.log_alpha, -metrics["entropy"])
            updates, new_alpha_optimizer_state = alpha_optimizer.update(
                grads_alpha, iqlearn.alpha_optimizer_state
            )
            new_log_alpha = optax.apply_updates(iqlearn.log_alpha, updates)  # type: ignore
            new_alpha = jnp.exp(new_log_alpha)  # type: ignore
            metrics.update({"alpha": new_alpha})
        else:
            new_alpha_optimizer_state = iqlearn.alpha_optimizer_state
            new_log_alpha = iqlearn.log_alpha
            new_alpha = iqlearn.alpha

        # EMA target update: target = (1 - tau) * target + tau * online
        new_actor_target = jax.tree.map(
            lambda x, y: (1 - params.tau) * x + params.tau * y,
            iqlearn.actor_target,
            new_actor,
        )
        new_critic_target = jax.tree.map(
            lambda x, y: (1 - params.tau) * x + params.tau * y,
            iqlearn.critic_target,
            new_critic,
        )

        return (
            IQLearnState(
                new_actor,  # type: ignore
                new_critic,  # type: ignore
                new_actor_target,
                new_critic_target,
                new_actor_optimizer_state,
                new_critic_optimizer_state,
                new_alpha_optimizer_state,  # type: ignore
                new_alpha,
                new_log_alpha,  # type: ignore
            ),
            metrics,
        )

    @jax.jit
    def train(
        iqlearn: IQLearnState, key: jax.Array
    ) -> Tuple[IQLearnState, dict]:
        """Run ``train_steps`` gradient updates and return averaged metrics.

        The loop is implemented with ``jax.lax.scan`` so the entire sequence
        is compiled as a single XLA program after the first call.

        Args:
            iqlearn: Current agent state.
            key: JAX PRNG key; split internally across all steps.

        Returns:
            ``(new_state, metrics)`` where each metric scalar is the mean over
            all ``train_steps`` steps.
        """

        def scan_fun(carry, x):
            iqlearn, key = carry
            key, next_key = jax.random.split(key)
            next_iqlearn, metrics = update_step(iqlearn, key)
            return (next_iqlearn, next_key), metrics

        (iqlearn, _), metrics = jax.lax.scan(
            scan_fun, (iqlearn, key), length=train_steps
        )
        metrics = jax.tree.map(lambda x: x.mean(), metrics)
        return iqlearn, metrics

    graphs = IQLearnGraphs(
        actor=NetworkGraphs(actor_fe_graph, actor_head_graph),
        critic_q1=NetworkGraphs(critic_q1_fe_graph, critic_q1_head_graph),
        critic_q2=NetworkGraphs(critic_q2_fe_graph, critic_q2_head_graph),
    )
    return iqlearn, IQLearnFunctions(predict, train), graphs
