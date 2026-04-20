"""IQ-Learn (Inverse Q-Learning) imitation learning.

Implements a SAC-style actor-critic whose reward signal is recovered from
expert demonstrations via the IQ-Learn objective (Garg et al., 2021).
Networks are split into a user-supplied *feature extractor* and an
internally-managed *head*, so the observation-processing backbone can be
swapped freely (MLP, CNN, transformer, …) without touching any IQ-Learn logic.

All state is held in immutable NamedTuples and the functional design
(``create_iqlearn`` factory + pure ``train``/``predict`` closures) keeps the
implementation compatible with ``jax.jit`` and ``jax.lax.scan``.

Typical usage::

    rngs = nnx.Rngs(0)
    actor_fe  = MLPFeatureExtractor(obs_dim, (256, 256), rngs=rngs)
    critic_fe = MLPFeatureExtractor(obs_dim, (256, 256), rngs=rngs)

    state, fns, graphs = create_iqlearn(
        Hyperparameters(), buffer, action_dim, actor_fe, critic_fe,
    )
    state, metrics = fns.train(state, jax.random.key(0))
    action = fns.predict(state, obs, deterministic=True)
"""

from functools import partial
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from buffer import Buffer, BufferSample, create_sample

# Bounds for the squashed log-standard-deviation of the policy distribution.
# The raw output is tanh-squashed and then rescaled into this range to keep
# the distribution numerically stable while remaining expressive.
LOG_STD_MIN = -5
LOG_STD_MAX = 2


# ---------------------------------------------------------------------------
# Network building blocks
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
        self.layers = nnx.List([
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
            for i in range(len(dims) - 1)
        ])

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


class ActorHead(nnx.Module):
    """Maps feature vectors to Gaussian distribution parameters.

    Outputs a single tensor of shape ``(batch, 2 * action_dim)`` whose first
    half is the mean and second half is the (pre-squashed) log-standard-deviation
    of a diagonal Gaussian policy.

    Args:
        feature_dim: Dimensionality of the incoming feature vector.
        action_dim: Number of action dimensions.
        hidden_dims: Optional hidden layers between features and output.
            Defaults to ``()`` (single linear projection from features to
            distribution parameters).
        rngs: Flax NNX RNG container used to initialise parameters.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (),
        *,
        rngs: nnx.Rngs,
    ):
        dims = [feature_dim] + list(hidden_dims) + [2 * action_dim]
        self.layers = nnx.List([
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
            for i in range(len(dims) - 1)
        ])

    def __call__(self, features: jax.Array) -> jax.Array:
        """Compute distribution parameters from features.

        Args:
            features: Feature batch of shape ``(batch, feature_dim)``.

        Returns:
            Array of shape ``(batch, 2 * action_dim)``.  Slice ``[..., :action_dim]``
            is the mean; ``[..., action_dim:]`` is the raw log-std (before
            tanh-squashing and rescaling).
        """
        x = features
        for layer in self.layers[:-1]:
            x = nnx.relu(layer(x))
        return self.layers[-1](x)


class CriticHead(nnx.Module):
    """Maps (feature, action) pairs to twin Q-values.

    Concatenates features and actions along the last axis, passes the result
    through hidden layers, and produces two Q-value estimates simultaneously
    (double-Q trick to reduce overestimation bias).

    Args:
        feature_dim: Dimensionality of the incoming feature vector.
        action_dim: Number of action dimensions.
        hidden_dims: Hidden layer widths after the feature-action concatenation.
            Defaults to ``(256, 256)``.
        rngs: Flax NNX RNG container used to initialise parameters.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        *,
        rngs: nnx.Rngs,
    ):
        dims = [feature_dim + action_dim] + list(hidden_dims) + [2]
        self.layers = nnx.List([
            nnx.Linear(dims[i], dims[i + 1], rngs=rngs)
            for i in range(len(dims) - 1)
        ])

    def __call__(self, features: jax.Array, actions: jax.Array) -> jax.Array:
        """Estimate twin Q-values for a batch of (feature, action) pairs.

        Args:
            features: Feature batch of shape ``(batch, feature_dim)``.
            actions: Action batch of shape ``(batch, action_dim)``.

        Returns:
            Array of shape ``(batch, 2)`` containing two independent Q estimates.
            Downstream callers take the element-wise minimum to form a
            conservative value estimate.
        """
        x = jnp.concat((features, actions), axis=-1)
        for layer in self.layers[:-1]:
            x = nnx.relu(layer(x))
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# State / function / graph containers
# ---------------------------------------------------------------------------


class IQLearnState(NamedTuple):
    """Complete, serialisable state of one IQ-Learn agent.

    All fields are JAX pytrees, so the entire state can be checkpointed,
    passed through ``jax.jit``, or stacked for vectorised environments.

    Attributes:
        actor: Online actor network state (feature extractor + head).
        critic: Online critic network state (feature extractor + head).
        actor_target: EMA-smoothed copy of the actor, used as a stable target
            during critic updates.
        critic_target: EMA-smoothed copy of the critic, used for bootstrapping
            next-state values.
        actor_optimizer_state: Optax state for the actor Adam optimiser.
        critic_optimizer_state: Optax state for the critic Adam optimiser.
        alpha_optimizer_state: Optax state for the entropy temperature optimiser.
        alpha: Current entropy temperature (``exp(log_alpha)``).
        log_alpha: Log-space entropy temperature; directly optimised to avoid
            a positivity constraint.
    """

    actor: NetworkState
    critic: NetworkState
    actor_target: NetworkState
    critic_target: NetworkState
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    alpha: jax.Array
    log_alpha: jax.Array


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
    """Flax NNX graph definitions for all four network modules.

    These are the static (non-parameter) descriptions produced by
    ``nnx.split`` and consumed by ``nnx.merge`` to reconstruct live modules
    during forward passes.  Returned by :func:`create_iqlearn` for callers
    that need direct access to the graph structure (e.g. for inspection or
    custom inference code).

    Attributes:
        actor_fe: Graph definition of the actor feature extractor.
        actor_head: Graph definition of the actor head.
        critic_fe: Graph definition of the critic feature extractor.
        critic_head: Graph definition of the critic head.
    """

    actor_fe: nnx.GraphDef
    actor_head: nnx.GraphDef
    critic_fe: nnx.GraphDef
    critic_head: nnx.GraphDef


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
        target_entropy: Desired policy entropy used by the alpha loss.  A
            common heuristic is ``-action_dim``.
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
    critic_feature_extractor: nnx.Module,
    obs_key: str = "observations",
    action_key: str = "actions",
    action_scale: float | jax.Array = 1,
    action_bias: float | jax.Array = 0,
    train_steps: int = 1000,
    actor_head_dims: tuple[int, ...] = (),
    critic_head_dims: tuple[int, ...] = (256, 256),
) -> Tuple[IQLearnState, IQLearnFunctions, IQLearnGraphs]:
    """Construct an IQ-Learn agent from a pre-filled buffer and user-supplied FEs.

    The feature extractors are taken as-is (already initialised by the caller),
    split into graph definition + parameter state via ``nnx.split``, and frozen
    inside the returned closures.  Actor and critic heads are created internally
    from ``actor_head_dims``/``critic_head_dims``; their input dimension is
    inferred automatically by running a dummy forward pass through each feature
    extractor.

    The returned ``train`` function runs ``train_steps`` gradient steps per call
    using ``jax.lax.scan``, keeping the whole loop JIT-compiled after the first
    invocation.

    Args:
        params: Hyperparameters controlling learning rates, discount, alpha, etc.
        buffer: Filled (or partially filled) replay buffer.  Must contain at
            least ``params.batch_size`` sampleable slots before ``train`` is
            called.
        action_dim: Number of continuous action dimensions.
        actor_feature_extractor: Initialised ``nnx.Module`` that maps
            ``(batch, *obs_shape) -> (batch, actor_feature_dim)``.  Ownership
            is transferred; the module is split and should not be used directly
            afterwards.
        critic_feature_extractor: Same contract as ``actor_feature_extractor``.
            Actor and critic may have different architectures and output widths.
        obs_key: Key in ``buffer.info`` that holds observations.
        action_key: Key in ``buffer.info`` that holds actions.
        action_scale: Per-dimension scale applied after the tanh squashing.
            Scalar or array of shape ``(action_dim,)``.
        action_bias: Per-dimension offset applied after the tanh squashing.
            Scalar or array of shape ``(action_dim,)``.
        train_steps: Number of gradient steps executed per ``train`` call.
        actor_head_dims: Hidden layer widths for the actor head.  Defaults to
            ``()`` (direct linear projection from features to distribution params).
        critic_head_dims: Hidden layer widths for the critic head.  Defaults to
            ``(256, 256)``.

    Returns:
        A ``(IQLearnState, IQLearnFunctions, IQLearnGraphs)`` triple.

        - ``IQLearnState``: initial agent state with online and target networks
          set to the same weights.
        - ``IQLearnFunctions``: named tuple of ``predict`` and ``train`` closures.
        - ``IQLearnGraphs``: static NNX graph definitions for all four modules,
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
    critic_feature_dim = critic_feature_extractor(dummy_obs).shape[-1]

    # Create heads
    rngs = nnx.Rngs(0)
    actor_head_model = ActorHead(
        actor_feature_dim, action_dim, actor_head_dims, rngs=rngs,
    )
    critic_head_model = CriticHead(
        critic_feature_dim, action_dim, critic_head_dims, rngs=rngs,
    )

    # Split all four modules into (graph_def, state)
    actor_fe_graph, actor_fe = nnx.split(actor_feature_extractor)
    actor_head_graph, actor_head = nnx.split(actor_head_model)
    critic_fe_graph, critic_fe = nnx.split(critic_feature_extractor)
    critic_head_graph, critic_head = nnx.split(critic_head_model)

    actor_state = NetworkState(actor_fe, actor_head)
    critic_state = NetworkState(critic_fe, critic_head)

    # Optimizers operate on NetworkState pytrees (fe + head jointly)
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
        remove_weak_types(actor_state),   # targets start equal to online weights
        remove_weak_types(critic_state),
        remove_weak_types(actor_optimizer_state),
        remove_weak_types(critic_optimizer_state),
        remove_weak_types(alpha_optimizer_state),
        remove_weak_types(jnp.exp(log_alpha)),
        remove_weak_types(log_alpha),
    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def run_actor(actor: NetworkState, x: jax.Array) -> jax.Array:
        """Reconstruct and run the actor (FE then head) on observation batch x."""
        fe = nnx.merge(actor_fe_graph, actor.fe)
        head = nnx.merge(actor_head_graph, actor.head)
        return head(fe(x))

    def run_critic(
        critic: NetworkState, x: jax.Array, actions: jax.Array
    ) -> jax.Array:
        """Reconstruct and run the critic (FE then head) returning twin Q-values."""
        fe = nnx.merge(critic_fe_graph, critic.fe)
        head = nnx.merge(critic_head_graph, critic.head)
        return head(fe(x), actions)

    def get_q(
        critic: NetworkState, x: jax.Array, actions: jax.Array
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
        critic: NetworkState,
        alpha: jax.Array,
        x: jax.Array,
        key: jax.Array,
        include_entropy: bool = True,
        include_log: bool = False,
    ) -> jax.Array | Tuple[jax.Array, dict]:
        """Compute the soft state-value V(x) = E_π[Q(x,a) - α log π(a|x)].

        Args:
            actor: Actor network state used to sample actions.
            critic: Critic network state used to evaluate Q-values.
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
                return q - alpha * logprob, {"q": q.mean(), "entropy": -logprob.mean()}
            else:
                return q - alpha * logprob
        else:
            return q

    # ------------------------------------------------------------------
    # Public functions
    # ------------------------------------------------------------------

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

    def loss_alpha(log_alpha: jax.Array, log_pi: jax.Array) -> jax.Array:
        """Entropy temperature loss.

        Minimising this pushes alpha so that the expected policy entropy
        matches ``params.target_entropy``.

        Args:
            log_alpha: Current log-temperature scalar.
            log_pi: Mean log-probability of the current policy (scalar).

        Returns:
            Scalar loss value.
        """
        alpha_loss = -jnp.exp(log_alpha) * (log_pi + params.target_entropy)
        return alpha_loss

    def loss_actor(
        actor: NetworkState,
        critic: NetworkState,
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
        critic: NetworkState,
        critic_target: NetworkState,
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

    def update_step(
        iqlearn: IQLearnState, key: jax.Array
    ) -> Tuple[IQLearnState, dict]:
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
        # critic gradients
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

        # update critic (fe + head jointly)
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

    graphs = IQLearnGraphs(actor_fe_graph, actor_head_graph, critic_fe_graph, critic_head_graph)
    return iqlearn, IQLearnFunctions(predict, train), graphs


# Example usage:
#
# obs_dim, action_dim = 10, 2
# rngs = nnx.Rngs(0)
# buffer, buffer_fns = create_buffer(
#     shapes={"observations": (obs_dim,), "actions": (action_dim,)},
#     size=10000, sampling_size=256,
#     this_step_infos=["observations", "actions"],
#     next_step_infos=["observations"],
# )
# actor_fe = MLPFeatureExtractor(obs_dim, (256, 256), rngs=rngs)
# critic_fe = MLPFeatureExtractor(obs_dim, (256, 256), rngs=rngs)
# iqlearn, functions, _ = create_iqlearn(
#     Hyperparameters(), buffer, action_dim, actor_fe, critic_fe,
# )
# iqlearn, metrics = functions.train(iqlearn, jax.random.key(0))
# print(metrics)
