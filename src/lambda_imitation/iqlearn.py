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
from jax.tree_util import Partial

from .buffer import Buffer, BufferFunctions, BufferSample, create_buffer, create_sample

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
        self.layers = nnx.List(
            [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]
        )

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
        self.layers = nnx.List(
            [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]
        )

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


class ExtractorState(NamedTuple):
    """Flax NNX graph states for a feature-extractor + head pair.

    Both fields are ``nnx.GraphState`` objects produced by ``nnx.split``.
    Together they form a JAX pytree, so optimizer updates and EMA target
    updates work on them transparently via ``jax.tree.map``.

    Attributes:
        fe: Graph state of the feature extractor module.
        head: Graph state of the task-specific head module.
    """

    fe: nnx.GraphState
    memory: nnx.GraphState


class TwinCriticState(NamedTuple):
    """Paired network states for the two independent Q-branches.

    Grouping both branches in a single NamedTuple (which is a JAX pytree)
    lets a single optimizer and a single ``jax.grad`` call operate over both
    branches simultaneously without any changes to the loss/update logic.

    Attributes:
        q1: Network state (FE + head) for the first Q-branch.
        q2: Network state (FE + head) for the second Q-branch.
    """

    q1: nnx.GraphState
    q2: nnx.GraphState


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
    memory: nnx.GraphDef


class TwinCriticGraph(NamedTuple):
    """Flax NNX graph definitions for a feature-extractor + head pair.

    These are the static (non-parameter) graph descriptions produced by
    ``nnx.split`` and consumed by ``nnx.merge`` to reconstruct live modules
    during forward passes.

    Attributes:
        fe: Graph definition of the feature extractor.
        head: Graph definition of the head.
    """

    q1: nnx.GraphDef
    q2: nnx.GraphDef


class SACState(NamedTuple):
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

    feature_extractor: ExtractorState
    actor: nnx.GraphState
    critic: TwinCriticState
    lambda1_critic: TwinCriticState
    lambda2_critic: TwinCriticState
    feature_extractor_target: ExtractorState
    actor_target: nnx.GraphState
    critic_target: TwinCriticState
    lambda1_critic_target: TwinCriticState
    lambda2_critic_target: TwinCriticState
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    lambda1_critic_optimizer_state: optax.OptState
    lambda2_critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    alpha: jax.Array
    log_alpha: jax.Array
    online_buffer: Buffer


class SACFunctions(NamedTuple):
    """Pure functions returned by :func:`create_iqlearn`.

    Attributes:
        predict: ``(state, obs, key, deterministic) -> action`` -- sample or
            compute a deterministic action for a single observation.
        train: ``(state, key) -> (state, metrics)`` -- run ``train_steps``
            IQ-Learn update iterations via ``jax.lax.scan`` and return
            averaged metrics.
        train_sac: ``(state, env, env_params, env_state, key) ->
            (state, env_state, metrics)`` -- collect online transitions from a
            gymnax-compatible environment and run ``train_steps`` SAC gradient
            updates.  ``env`` is a static (non-traced) Python object;
            ``env_params`` and ``env_state`` are JAX pytrees.  Returns the
            updated agent state, the new gymnax environment state (including
            auto-resets on episode termination), and averaged metrics.
        get_importance_ratios: ``(actor, obs, actions, behaviour_probs) ->
            ratios`` -- compute per-transition importance ratios
            ``π(a|s) / b(a|s)`` for a batch of ``(obs, action)`` pairs
            collected under a behaviour policy ``b``.  For discrete spaces,
            ``actions`` are float32 integer indices.  For continuous spaces,
            ``actions`` are the **unsquashed** (pre-tanh) values stored under
            ``unsquashed_action_key`` in the online buffer.
            ``behaviour_probs`` are probabilities (discrete) or probability
            densities (continuous) under ``b``, shape ``(batch,)``.
        prefill_buffer: ``(state, env, env_params, env_state, n_steps, key) ->
            (state, env_state)`` -- collect ``n_steps`` transitions using a
            uniform random policy and write them into the online buffer.
            Discrete: action drawn uniformly from ``{0, …, action_dim-1}``,
            ``behaviour_key = 1/action_dim``.  Continuous: ``u ~ N(0, I)``,
            action squashed through tanh, ``behaviour_key = exp(log_prob)``
            using the same change-of-variables as ``get_importance_ratios``.
            The last step is force-terminated so that all ``n_steps`` written
            slots are immediately sampleable.  Called automatically by
            ``train_sac`` when the online buffer has fewer than
            ``params.online_batch_size`` sampleable transitions.
    """

    predict: Callable
    train: Callable
    get_importance_ratios: Callable
    prefill_buffer: Callable


class DebugFunctions(NamedTuple):
    """Optional debug helpers returned by :func:`create_iqlearn` when ``debug=True``.

    These functions expose internal computations that are useful for
    introspection and unit testing but are not required for normal training.

    Attributes:
        calculate_td_lambda: ``(actor, mc_critic_target, online_buffer, indices) ->
            td_returns`` -- compute a V-trace TD(λ) return estimate for each
            index in ``indices`` (vmapped).  ``actor`` is the current
            :class:`NetworkState` from :class:`SACState`.
            ``mc_critic_target`` is the EMA target for the MC critic
            (i.e. ``state.mc_critic_target``); the function uses it to
            bootstrap Q-values.  ``online_buffer`` is the online replay
            buffer (e.g. ``state.online_buffer``); it must contain at least
            ``params.lambda_truncation + 1`` filled slots starting from the
            smallest index requested.  ``indices`` is a 1-D integer array of
            buffer start positions.  Returns a float32 array of shape
            ``(len(indices),)``.
        get_q: ``(critic, obs, actions, use_mc) -> q_values`` -- evaluate the
            conservative (min over twin) Q-values for a batch of
            ``(obs, actions)`` pairs.  Pass ``state.critic_target`` with
            ``use_mc=False`` for the regular SAC critic; pass
            ``state.mc_critic_target`` with ``use_mc=True`` for the MC
            critic.  ``obs`` has shape ``(batch, *obs_shape)``; ``actions``
            has shape ``(batch, action_dim)`` for continuous spaces or
            ``(batch, 1)`` (float32 action indices) for discrete spaces.
            Returns a float32 array of shape ``(batch,)``.
    """

    calculate_td_lambda: Callable
    get_q: Callable
    get_entropy: Callable


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
        online_buffer_size: Capacity of the circular online replay buffer used
            by :func:`train_sac`.  Older transitions are overwritten once the
            buffer is full.
        online_batch_size: Number of transitions sampled per SAC gradient step.
            :func:`create_iqlearn` pre-fills the online buffer with this many
            random transitions so that :func:`train_sac` can update from the
            very first call.
        tau: Soft update coefficient for EMA target networks.  A value of
            ``0.005`` means targets lag significantly behind online weights.
        lam: TD(λ) mixing coefficient for the Monte-Carlo critic target.
            ``lam=0`` collapses to a single one-step bootstrap; ``lam→1``
            approaches a full Monte-Carlo return.  (Named ``lam`` because
            ``lambda`` is a Python keyword.)
        lambda_truncation: Number of look-ahead steps used when computing the
            TD(λ) return in :func:`calculate_td_lambda`.  The function reads
            ``lambda_truncation + 1`` consecutive slots from the online buffer
            per target estimate, so the online buffer must hold at least that
            many transitions.
    """

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    lambda_critic_lr: float = 1e-2
    alpha_lr: float = 1e-3
    alpha: float = 1.0
    autotune_alpha: bool = True
    batch_size: int = 256
    gamma: float = 0.99
    regularizer_coef: float = 1 / 40
    target_entropy: float = -1
    online_buffer_size: int = 10_000
    online_batch_size: int = 256
    tau: float = 0.005
    lambda1: float = 0.1
    lambda2: float = 0.9
    lambda1_truncation: int = 15
    lambda2_truncation: int = 15


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def extract_buffer_shapes(buffer: Buffer) -> dict[str, tuple[int, ...]]:
    """Extract per-key item shapes from an existing buffer.

    Useful for creating a second buffer (e.g. the online buffer) with the
    same data schema as an existing one.

    Args:
        buffer: A :class:`Buffer` whose ``info`` arrays determine the item
            shapes.  Only the shape is read; the contents are not used.

    Returns:
        Dict mapping each key to its per-item shape (i.e.
        ``buffer.info[k].shape[1:]`` for every key ``k``).
    """
    return {k: v.shape[1:] for k, v in buffer.info.items()}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

unsquashed_action_key = "unsquashed_actions"
reward_key = "rewards"
terminated_key = "terminated"
return_key = "returns"
behaviour_key = "behaviour_weight"
carry_key = "carry"


def create_iqlearn(
    params: Hyperparameters,
    buffer: Buffer,
    action_dim: int,
    feature_extractor: nnx.Module,
    obs_key: str = "observations",
    action_key: str = "actions",
    action_scale: float | jax.Array = 1,
    action_bias: float | jax.Array = 0,
    train_steps: int = 1000,
    actor_dims: tuple[int, ...] = (),
    critic_dims: tuple[int, ...] = (256, 256),
    lambda1_critic_dims: tuple[int, ...] = (256, 256),
    lambda2_critic_dims: tuple[int, ...] = (256, 256),
    is_discrete: bool = False,
    approximate_lambda: bool = False,
    debug: bool = False,
) -> (
    "Tuple[SACState, SACFunctions, IQLearnGraphs] | "
    "Tuple[SACState, SACFunctions, IQLearnGraphs, DebugFunctions]"
):
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
        reward_key: Key used to store per-step scalar rewards in the online
            buffer populated by :func:`train_sac`.  Must not clash with any
            key already in ``buffer.info``.
        terminated_key: Key used to store per-step episode-termination flags
            (``float32`` 0/1) in the online buffer.  Must not clash with any
            key already in ``buffer.info``.
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
        debug: If True, return a 4-tuple whose fourth element is a
            :class:`DebugFunctions` named tuple exposing internal computations
            (currently: ``calculate_td_lambda`` and ``get_q``).  If False
            (default), return the usual 3-tuple and the debug helpers are not
            constructed.

    Returns:
        When ``debug=False`` (default): a
        ``(SACState, SACFunctions, IQLearnGraphs)`` triple.

        When ``debug=True``: a
        ``(SACState, SACFunctions, IQLearnGraphs, DebugFunctions)``
        4-tuple.

        - ``SACState``: initial agent state with online and target networks
          set to the same weights.
        - ``SACFunctions``: named tuple of ``predict`` and ``train`` closures.
        - ``IQLearnGraphs``: static NNX graph definitions for all network modules,
          useful for inspection or custom inference.
        - ``DebugFunctions`` *(only when debug=True)*: named tuple of internal
          helper functions, including ``calculate_td_lambda`` and ``get_q``.
    """
    assert (
        not approximate_lambda or mc_critic_q1_feature_extractor is not None
    ), "If approximate_lambda is set, feature extractors have to be set as well"
    assert (
        not approximate_lambda or mc_critic_q2_feature_extractor is not None
    ), "If approximate_lambda is set, feature extractors have to be set as well"
    this_keys = [obs_key, action_key]
    next_keys = [obs_key]
    buffer_sample = create_sample(
        buffer.size,
        params.batch_size,
        this_keys=this_keys,
        next_keys=next_keys,
    )
    # Online buffer: same obs/action shapes as the expert buffer, plus scalar
    # reward and terminated fields written by run_env_step / train_sac.
    online_shapes = {
        **extract_buffer_shapes(buffer),
        reward_key: (),
        return_key: (),
        terminated_key: (),
        behaviour_key: (),
    }
    if not is_discrete:
        online_shapes[unsquashed_action_key] = online_shapes[action_key]
    online_this_keys = [obs_key, action_key, reward_key, terminated_key]
    online_next_keys = [obs_key]
    online_buffer, online_buffer_functions = create_buffer(
        online_shapes,
        params.online_buffer_size,
        params.online_batch_size,
        online_this_keys,
        online_next_keys,
    )
    online_buffer_sample = online_buffer_functions.sample
    if approximate_lambda:
        online_mc_this_keys = [obs_key, action_key, return_key, behaviour_key]
        online_mc_next_keys = []
        online_buffer_mc_sample = create_sequence_sample(
            online_buffer.size,
            params.online_batch_size,
            online_mc_this_keys,
            online_mc_next_keys,
        )
        # Pre-initialise every behaviour_key slot to 1.0 so that slots which
        # have never been written by run_env_step (pre-fill and as-yet-unwritten
        # slots) contribute an IS denominator of 1.0 rather than 0.0.
        # Division by zero would otherwise produce NaN in calculate_td_lambda.
        # Real transitions overwrite their slot with the true policy probability.
        online_buffer = online_buffer._replace(
            info={
                **online_buffer.info,
                behaviour_key: jnp.ones_like(online_buffer.info[behaviour_key]),
            }
        )

    # Infer feature dims via dummy forward pass (before split)
    dummy_obs = jnp.zeros((1,) + buffer.info[obs_key].shape[1:])
    feature_dim = feature_extractor(dummy_obs).shape[-1]

    # Create heads — discrete and continuous differ only in output_dim and
    # whether actions are concatenated to features before the head.
    rngs = nnx.Rngs(0)
    if is_discrete:
        actor_model = Head(
            feature_dim,
            actor_dims,
            action_dim,
            rngs=rngs,
        )
        critic_q1_model = Head(
            feature_dim,
            critic_dims,
            action_dim,
            rngs=rngs,
        )
        critic_q2_model = Head(
            feature_dim,
            critic_dims,
            action_dim,
            rngs=rngs,
        )
        if approximate_lambda:
            lamda1_q1_critic_model = Head(
                feature_dim,
                lambda1_critic_dims,
                action_dim,
                rngs=rngs,
            )
            lambda1_q2_critic_model = Head(
                feature_dim,
                lambda1_critic_dims,
                action_dim,
                rngs=rngs,
            )
            lambda2_q1_critic_model = Head(
                feature_dim,
                lambda2_critic_dims,
                action_dim,
                rngs=rngs,
            )
            lambda2_q2_critic_model = Head(
                feature_dim,
                lambda2_critic_dims,
                action_dim,
                rngs=rngs,
            )
    else:
        actor_model = Head(
            feature_dim,
            actor_dims,
            2 * action_dim,
            rngs=rngs,
        )
        # For continuous critics, features and actions are concatenated before
        # the head, so input_dim = feature_dim + action_dim, output_dim = 1.
        critic_q1_model = Head(
            feature_dim + action_dim,
            critic_dims,
            1,
            rngs=rngs,
        )
        critic_q2_model = Head(
            feature_dim + action_dim,
            critic_dims,
            1,
            rngs=rngs,
        )
        if approximate_lambda:
            lambda1_q1_critic_model = Head(
                feature_dim + action_dim,
                lambda1_critic_dims,
                1,
                rngs=rngs,
            )
            lambda1_q2_critic_model = Head(
                feature_dim + action_dim,
                lambda1_critic_dims,
                1,
                rngs=rngs,
            )
            lambda2_q1_critic_model = Head(
                feature_dim + action_dim,
                lambda2_critic_dims,
                1,
                rngs=rngs,
            )
            lambda2_q2_critic_model = Head(
                feature_dim + action_dim,
                lambda2_critic_dims,
                1,
                rngs=rngs,
            )

    # Split all six modules into (graph_def, state)
    feature_extractor_graph, feature_extractor_state = nnx.split(actor_model)
    actor_graph, actor_state = nnx.split(actor_model)
    critic_q1_graph, critic_q1_state = nnx.split(critic_q1_model)
    critic_q2_graph, critic_q2_state = nnx.split(critic_q2_model)
    if approximate_lambda:
        lambda1_q1_critic_graph, lambda1_q1_critic_state = nnx.split(
            lambda1_q1_critic_model
        )
        lambda1_q2_critic_graph, lambda1_q2_critic_state = nnx.split(
            lambda1_q2_critic_model
        )
        lambda2_q1_critic_graph, lambda2_q1_critic_state = nnx.split(
            lambda2_q1_critic_model
        )
        lambda2_q2_critic_graph, lambda2_q2_critic_state = nnx.split(
            lambda2_q2_critic_model
        )

    critic_state = TwinCriticState(critic_q1_state, critic_q2_state)
    critic_graph = TwinCriticGraph(critic_q1_graph, critic_q2_graph)
    if approximate_lambda:
        lambda1_critic_state = TwinCriticState(
            lambda1_q1_critic_state, lambda1_q2_critic_state
        )
        lambda1_critic_graph = TwinCriticGraph(
            lambda1_q1_critic_graph, lambda1_q2_critic_graph
        )
        lambda2_critic_state = TwinCriticState(
            lambda2_q1_critic_state, lambda2_q2_critic_state
        )
        lambda2_critic_graph = TwinCriticGraph(
            lambda2_q1_critic_graph, lambda2_q2_critic_graph
        )

    # Optimizers: actor operates on NetworkState; critic on TwinCriticState.
    actor_optimizer = optax.adam(params.actor_lr)
    critic_optimizer = optax.adam(params.critic_lr)
    if approximate_lambda:
        lambda1_critic_optimizer = optax.adam(params.lambda_critic_lr)
        lambda2_critic_optimizer = optax.adam(params.lambda_critic_lr)
    alpha_optimizer = optax.adam(params.alpha_lr)

    log_alpha = jnp.array(jnp.log(params.alpha))
    actor_optimizer_state = actor_optimizer.init(actor_state)
    critic_optimizer_state = critic_optimizer.init(critic_state)
    if approximate_lambda:
        lambda1_critic_optimizer_state = lambda1_critic_optimizer.init(
            lambda1_critic_state
        )
        lambda2_critic_optimizer_state = lambda2_critic_optimizer.init(
            lambda2_critic_state
        )
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    def remove_weak_types(state):
        """Materialise JAX weak dtypes to concrete dtypes to avoid JIT issues."""
        return jax.tree.map(
            lambda x: jnp.array(x, dtype=x.dtype) if hasattr(x, "dtype") else x,
            state,
        )

    iqlearn = SACState(
        remove_weak_types(feature_extractor_state),
        remove_weak_types(actor_state),
        remove_weak_types(critic_state),
        remove_weak_types(lambda1_critic_state) if approximate_lambda else None,
        remove_weak_types(lambda2_critic_state) if approximate_lambda else None,
        remove_weak_types(feature_extractor_state),
        remove_weak_types(actor_state),  # targets start equal to online weights
        remove_weak_types(critic_state),
        remove_weak_types(lambda1_critic_state) if approximate_lambda else None,
        remove_weak_types(lambda2_critic_state) if approximate_lambda else None,
        remove_weak_types(actor_optimizer_state),
        remove_weak_types(critic_optimizer_state),
        (
            remove_weak_types(lambda1_critic_optimizer_state)
            if approximate_lambda
            else None
        ),
        (
            remove_weak_types(lambda2_critic_optimizer_state)
            if approximate_lambda
            else None
        ),
        remove_weak_types(alpha_optimizer_state),
        remove_weak_types(jnp.exp(log_alpha)),
        remove_weak_types(log_alpha),
        remove_weak_types(online_buffer),
    )

    # ------------------------------------------------------------------
    # Shared helper: actor forward pass (same for both action space types)
    # ------------------------------------------------------------------

    def run_actor(actor: nnx.GraphState, x: jax.Array) -> jax.Array:
        """Reconstruct and run the actor (FE then head) on observation batch x."""
        head = nnx.merge(actor_graph, actor.head)
        return head(x)

    # ------------------------------------------------------------------
    # Action-space-specific helpers
    # ------------------------------------------------------------------

    if is_discrete:

        def run_critic(
            critic: TwinCriticState, graph: TwinCriticGraph, x: jax.Array
        ) -> jax.Array:
            """Reconstruct and run both discrete critic branches.

            Returns:
                Array of shape ``(batch, num_actions, 2)`` where the last axis
                indexes the two independent Q estimates.
            """
            head1 = nnx.merge(graph.q1, critic.q1)
            head2 = nnx.merge(graph.q2, critic.q2)
            q1 = head1(x)  # (batch, num_actions)
            q2 = head2(x)  # (batch, num_actions)
            return jnp.stack([q1, q2], axis=-1)  # (batch, num_actions, 2)

        def get_q_both(
            critic: TwinCriticState,
            graph: TwinCriticGraph,
            x: jax.Array,
            actions: jax.Array,
        ) -> Tuple[jax.Array, jax.Array]:
            """Per-branch Q-values for the taken action in each expert transition.

            Args:
                critic: Twin-critic network state.
                x: Observation batch.
                expert_actions: Float32 array of shape ``(batch, 1)`` holding
                    action indices stored as floats (e.g. 0.0, 1.0, 2.0).

            Returns:
                ``(q1, q2)`` each of shape ``(batch,)``.
            """
            q_twin = run_critic(critic, graph, x)  # (batch, num_actions, 2)
            indices = jnp.round(actions.reshape(-1)).astype(jnp.int32)
            batch = q_twin.shape[0]
            return (
                q_twin[jnp.arange(batch), indices, 0],  # (batch,)
                q_twin[jnp.arange(batch), indices, 1],
            )  # (batch,)

        def get_q(
            critic: TwinCriticState,
            graph: TwinCriticGraph,
            x: jax.Array,
            actions: jax.Array,
        ) -> jax.Array:
            """Conservative (min over twin) Q-value for each expert transition.

            Args:
                critic: Twin-critic network state.
                x: Observation batch.
                expert_actions: Float32 array of shape ``(batch, 1)`` holding
                    action indices stored as floats (e.g. 0.0, 1.0, 2.0).

            Returns:
                Per-transition Q-value of shape ``(batch,)``.
            """
            q1, q2 = get_q_both(critic, graph, x, actions)
            return jnp.minimum(q1, q2)

        def get_v(
            actor: NetworkState,
            critic: TwinCriticState,
            graph: TwinCriticGraph,
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
            q_twin = run_critic(critic, graph, x)  # (batch, num_actions, 2)
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

        def get_entropy(actor, x, _key):
            logits = run_actor(actor, x)  # (batch, num_actions)
            probs = jax.nn.softmax(logits)  # (batch, num_actions)
            log_probs = jax.nn.log_softmax(logits)  # (batch, num_actions)
            entropy = -(probs * log_probs).sum(-1)  # (batch,) — exact H(π)
            return entropy

        @partial(jax.jit, static_argnames=["deterministic", "return_prob"])
        def predict(
            iqlearn: SACState,
            obs: jax.Array,
            carry: jax.Array,
            key: jax.Array = jnp.array(0),
            deterministic: bool = False,
            return_prob: bool = False,
        ) -> jax.Array | Tuple[jax.Array, jax.Array]:
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
            carry_batch = jnp.expand_dims(carry, 0)
            new_carry, x = nnx.merge(feature_extractor_graph, feature_extractor_state)(
                obs_batch, carry_batch
            )
            logits = run_actor(iqlearn.actor, x)[0]  # (num_actions,)
            if deterministic:
                action = jnp.argmax(logits)
            else:
                action = jax.random.categorical(key, logits)
            if return_prob:
                return (
                    action.astype(jnp.float32),
                    new_carry,
                    jax.nn.softmax(logits)[action],
                )
            return (
                action.astype(jnp.float32),
                new_carry,
            )

        @jax.jit
        def get_importance_ratios(
            actor: nnx.GraphState,
            x: jax.Array,
            actions: jax.Array,
            behaviour_probs: jax.Array,
        ) -> jax.Array:
            """Compute importance ratios ``π(a|s) / b(a|s)`` for a batch.

            Args:
                actor: Current actor network state.
                obs: Observation batch of shape ``(batch, *obs_shape)``.
                actions: Integer action indices stored as float32, shape
                    ``(batch,)`` or ``(batch, 1)``.
                behaviour_probs: Per-transition probabilities under the
                    behaviour policy, shape ``(batch,)``.

            Returns:
                Importance ratios of shape ``(batch,)``.
            """
            logits = run_actor(actor, x)  # (batch, num_actions)
            probs = jax.nn.softmax(logits)  # (batch, num_actions)
            idx = actions.reshape(-1).astype(jnp.int32)  # (batch,)
            pi_a = probs[jnp.arange(idx.shape[0]), idx]  # (batch,)
            return pi_a / behaviour_probs

    else:
        # ------------------------------------------------------------------
        # Continuous helpers
        # ------------------------------------------------------------------

        def run_critic(
            critic: TwinCriticState,
            graph: TwinCriticGraph,
            x: jax.Array,
            actions: jax.Array,
        ) -> jax.Array:
            """Reconstruct and run both continuous critic branches.

            Features and actions are concatenated before each head so each
            branch has a fully independent view of the (obs, action) pair.

            Returns:
                Array of shape ``(batch, 2)`` containing two independent Q estimates.
            """
            head1 = nnx.merge(graph.q1, critic.q1)
            head2 = nnx.merge(graph.q2, critic.q2)
            q1 = head1(jnp.concat((x, actions), axis=-1))  # (batch, 1)
            q2 = head2(jnp.concat((x, actions), axis=-1))  # (batch, 1)
            return jnp.concat([q1, q2], axis=-1)  # (batch, 2)

        def get_q_both(
            critic: TwinCriticState,
            graph: TwinCriticGraph,
            x: jax.Array,
            actions: jax.Array,
        ) -> Tuple[jax.Array, jax.Array]:
            """Per-branch Q-values for continuous actions.

            Returns:
                ``(q1, q2)`` each of shape ``(batch,)``.
            """
            q = run_critic(critic, graph, x, actions)  # (batch, 2)
            return q[:, 0], q[:, 1]

        def get_q(
            critic: TwinCriticState,
            graph: TwinCriticGraph,
            x: jax.Array,
            actions: jax.Array,
        ) -> jax.Array:
            """Return the conservative (min over twin) Q-value for each transition."""
            q1, q2 = get_q_both(critic, graph, x, actions)
            return jnp.minimum(q1, q2)

        def get_dist_params(
            actor: nnx.GraphState, x: jax.Array
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
            actor: nnx.GraphState, x: jax.Array, key: jax.Array
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
            return action, unsquashed_action, log_prob.sum(axis=-1)

        @jax.jit
        def get_importance_ratios(
            actor: nnx.GraphState,
            x: jax.Array,
            actions: jax.Array,
            behaviour_probs: jax.Array,
        ) -> jax.Array:
            """Compute importance ratios ``π(a|s) / b(a|s)`` for a batch.

            Args:
                actor: Current actor network state.
                obs: Observation batch of shape ``(batch, *obs_shape)``.
                actions: **Unsquashed** (pre-tanh) action values as stored
                    under ``unsquashed_action_key`` in the online buffer,
                    shape ``(batch, action_dim)``.
                behaviour_probs: Per-transition probability densities under
                    the behaviour policy, shape ``(batch,)``.

            Returns:
                Importance ratios of shape ``(batch,)``.
            """
            mean, std = get_dist_params(actor, x)  # (batch, action_dim)
            u = actions  # pre-tanh, no inversion needed
            y_t = jnp.tanh(u)
            log_prob = (
                -((u - mean) ** 2) / (2 * std**2)
                - 0.5 * jnp.log(2 * jnp.pi)
                - jnp.log(std)
                - jnp.log(action_scale * (1 - y_t**2) + 1e-6)
            ).sum(
                axis=-1
            )  # (batch,)
            return jnp.exp(log_prob) / behaviour_probs

        def get_v(
            actor: nnx.GraphState,
            critic: TwinCriticState,
            graph: TwinCriticGraph,
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
            action, _unsquashed, logprob = sample_action_logprob(actor, x, key)
            q = get_q(critic, graph, x, action)
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

        def get_entropy(actor, obs, key):
            _action, _unsquashed, logprob = sample_action_logprob(actor, obs, key)
            return -logprob

        @partial(jax.jit, static_argnames=["deterministic", "return_unsquashed"])
        def predict(
            iqlearn: SACState,
            obs: jax.Array,
            carry: jax.Array,
            key: jax.Array = jnp.array(0),
            deterministic: bool = False,
            return_unsquashed: bool = False,
        ) -> jax.Array | Tuple[jax.Array, jax.Array]:
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
            carry = jnp.expand_dims(carry, 0)
            new_carry, x = nnx.merge(
                feature_extractor_graph, iqlearn.feature_extractor
            )(obs, carry)
            mean, std = get_dist_params(iqlearn.actor, x)
            if deterministic:
                unsquashed_action = mean
            else:
                unsquashed_action = jax.random.normal(key, mean.shape) * std + mean
            y_t = jnp.tanh(unsquashed_action)
            action = y_t * action_scale + action_bias
            if return_unsquashed:
                return action[0], new_carry, unsquashed_action[0]
            return action[0], new_carry

    # ------------------------------------------------------------------
    # Loss functions and other helpers which are structurally identical
    # for both action space types; differences are absorbed by the
    # action-space-specific helpers above.
    # ------------------------------------------------------------------

    @Partial(jax.vmap, in_axes=[None, None, None, None, None, 0])
    def calculate_td_lambda(
        actor, critic, alpha, key, online_buffer: Buffer, index: int
    ):
        """Compute a V-trace TD(λ) return estimate starting at buffer position ``index``.

        This function is vmapped over ``index`` via ``@Partial(jax.vmap, ...)``;
        it operates on a single start index and produces a single scalar return.

        **Algorithm overview**

        1. Collect ``lambda_truncation + 1`` consecutive buffer slots:
           ``indices = index + jnp.arange(lambda_truncation + 1)``.

        2. Compute per-step importance ratios
           ``c_k = π(a_k|s_k) / b(a_k|s_k)`` via :func:`get_importance_ratios`.

        3. Compute V-trace truncated cumulative IS weights:
           ``p_k = clip(∏_{j≤k} c_j, max=1.0)``.
           Clipping to 1 prevents variance explosion from large off-policy
           corrections (V-trace truncation level ``ρ̄ = 1``).

        4. Compute the done mask: ``done_mask[k] = 1`` if any step
           ``j < k`` was terminal, else 0 (first step is always 0).
           Rewards at and after a terminal are zeroed out by this mask.

        5. Accumulate partial n-step returns:
           ``rn[n] = Σ_{k=0}^{n-1} γ^k p_k r_k (1−done_mask[k])
                    + γ^{n} p_{n} Q(s_n, a_n)``.

        6. Mix with geometric weights:
           ``TD(λ) = (1 − λ) Σ_{n=0}^{lambda_truncation-1} λ^n rn[n]``.

        Args:
            actor: Current actor network state (:class:`NetworkState`).
            critic: MC critic target state (:class:`TwinCriticState`), i.e.
                ``state.mc_critic_target``.  The function calls ``get_q``
                with ``use_mc=True``, which selects the MC critic graph
                definitions for reconstruction.
            online_buffer: Online replay buffer containing at least
                ``lambda_truncation + 1`` filled transitions from the start
                index onward.  Must include keys ``obs_key``, ``action_key``,
                ``behaviour_key``, ``reward_key``, and ``terminated_key``.
            index: Scalar integer start position in the buffer (before vmap).

        Returns:
            Scalar ``float32`` TD(λ) return estimate for the transition at
            ``index``.
        """
        # Equivalent to the original ``index + jnp.arange(index, index + n + 1)``
        # (which produces ``index + [0, 1, ..., n]``), but uses a concrete
        # length so the expression is valid under jax.vmap.
        indices = index + jnp.arange(params.lambda_truncation + 1)
        key_ent, key_v = jax.random.split(key)

        c_k = get_importance_ratios(
            actor,
            online_buffer.info[obs_key][indices],
            online_buffer.info[action_key][indices],
            online_buffer.info[behaviour_key][indices],
        )
        p_k = jnp.clip(jnp.prod(c_k), max=1.0)
        gammas = params.gamma ** jnp.arange(params.lambda_truncation + 1)
        lambdas = params.lam ** jnp.arange(params.lambda_truncation)
        done_mask = 1 - jnp.concatenate(
            (
                jnp.array([0]),
                jnp.maximum.accumulate(
                    online_buffer.info[terminated_key][indices[:-1]]
                ),
            ),
            dtype=jnp.float32,
        )
        entropies = get_entropy(actor, online_buffer.info[obs_key][indices], key_ent)
        rn = jnp.cumsum(
            gammas[:-1]
            * (online_buffer.info[reward_key][indices[:-1]] + alpha * entropies[:-1])
            * done_mask[:-1]
        )
        rn = rn + gammas[1:] * done_mask[1:] * get_v(  # type: ignore
            actor,
            critic,
            alpha,
            online_buffer.info[obs_key][indices][1:],
            key_v,
            include_entropy=True,
        )
        return (1 - params.lam) * jnp.sum(lambdas * rn), p_k

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
        online_buf: Buffer,
        alpha: jax.Array,
        key: jax.Array,
    ) -> Tuple[jax.Array, dict]:
        """SAC Bellman MSE loss for the twin-critic (continuous and discrete).

        Computes independent TD errors for both Q-branches against the shared
        target ``r + γ(1−done)·V(s')``.  ``V(s')`` is computed under the
        target actor and critic; for discrete spaces this is the exact
        closed-form inner product, for continuous spaces it uses a sampled
        action.

        Args:
            actor_target: EMA-smoothed actor used to compute ``V(s')``.
            critic: Online twin-critic being optimised.
            critic_target: EMA-smoothed critic used inside ``V(s')``.
            online_buf: The online replay buffer.
            alpha: Current entropy temperature.
            key: JAX PRNG key.

        Returns:
            ``(scalar_loss, metrics)`` where metrics contains
            ``"critic_loss"`` and ``"target_q"``.
        """
        key_sample, key_v = jax.random.split(key, 2)
        sample, _ = online_buffer_sample(online_buf, key_sample)
        obs = sample.this_info[obs_key]
        actions = sample.this_info[action_key]
        rewards = sample.this_info[reward_key].reshape(-1)
        terminated = sample.this_info[terminated_key].reshape(-1)

        next_v = get_v(
            actor_target,
            critic_target,
            alpha,
            sample.next_info[obs_key],
            key_v,
            include_entropy=True,
        )
        target_q = jax.lax.stop_gradient(
            rewards + params.gamma * (1.0 - terminated) * next_v
        )

        q1, q2 = get_q_both(critic, obs, actions)
        loss = 0.5 * (jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2))
        return loss, {"critic_loss": loss, "target_q": target_q.mean()}

    def loss_mc_critic_sac(
        actor: NetworkState,
        mc_critic: TwinCriticState,
        critic_target: TwinCriticState,
        online_buf: Buffer,
        alpha: jax.Array,
        key: jax.Array,
    ) -> Tuple[jax.Array, dict]:
        key_sample, key_td_lambda = jax.random.split(key, 2)
        sample, indices = online_buffer_sample(online_buf, key_sample)
        obs = sample.this_info[obs_key]
        actions = sample.this_info[action_key]

        target_q, coefs = jax.lax.stop_gradient(
            calculate_td_lambda(actor, critic_target, alpha, key_td_lambda, online_buf, indices)  # type: ignore
        )

        q1, q2 = get_q_both(mc_critic, obs, actions, use_mc=True)
        loss = 0.5 * (
            jnp.mean(coefs * (q1 - target_q) ** 2)
            + jnp.mean(coefs * (q2 - target_q) ** 2)
        )
        return loss, {"mc_critic_loss": loss, "target_q": target_q.mean()}

    # ------------------------------------------------------------------
    # Online helpers: environment interaction and SAC update
    # ------------------------------------------------------------------

    def run_env_step(sac: SACState, env, env_params, env_state, key: jax.Array):
        """Collect one transition from a gymnax environment into the online buffer.

        Calls ``env.get_obs`` to read the current observation, queries the
        actor policy for an action, steps the environment, and writes the
        ``(obs, action, reward, terminated)`` transition into
        ``sac.online_buffer``.  Gymnax's base ``step()`` already performs an
        automatic reset when the episode ends, so the returned state is always
        ready for the next step without any additional handling.

        Args:
            sac: Current agent state.  Only ``sac.online_buffer`` is mutated.
            env: Gymnax environment object (static — not traced by JAX).
            env_params: Gymnax environment parameters pytree.
            env_state: Current gymnax environment state pytree.
            key: JAX PRNG key; split internally for action sampling and env step.

        Returns:
            ``(new_sac, new_env_state)`` where ``new_sac`` has an updated
            ``online_buffer`` and ``new_env_state`` is the post-step gymnax
            state (already reset if the episode ended).
        """
        key_act, key_step = jax.random.split(key, 2)
        obs = env.get_obs(env_state, env_params)
        if is_discrete:
            action, prob = predict(sac, obs, key_act, return_prob=True)
            env_action = jnp.round(action).astype(jnp.int32)
        else:
            if approximate_lambda:
                action, unsquashed_action, logprob = sample_action_logprob(
                    sac.actor, obs, key_act
                )
                prob = jnp.exp(logprob)
            else:
                action = predict(sac, obs, key_act)
            env_action = action
        _next_obs, new_env_state, reward, done, _ = env.step(
            key_step, env_state, env_action, env_params
        )
        transition = {
            obs_key: obs,
            action_key: jnp.atleast_1d(action),
            reward_key: jnp.asarray(reward, dtype=jnp.float32),
            terminated_key: jnp.asarray(done, dtype=jnp.float32),
        }
        if approximate_lambda:
            transition[behaviour_key] = prob
            if not is_discrete:
                transition[unsquashed_action_key] = jnp.atleast_1d(unsquashed_action)  # type: ignore
        new_online_buffer = online_buffer_functions.add(
            sac.online_buffer, transition, terminated=done
        )
        return sac._replace(online_buffer=new_online_buffer), new_env_state

    def update_step(sac: SACState, key: jax.Array) -> Tuple[SACState, dict]:
        """Execute one SAC update step using the online replay buffer.

        Uses the standard SAC Bellman MSE objective for the critic (with real
        environment rewards) and the same soft-value actor objective as
        IQ-Learn.  The online buffer must already hold at least
        ``params.online_batch_size`` sampleable transitions before this
        function is called (guaranteed by :func:`prefill_buffer`, which
        :func:`train_sac` calls automatically when the buffer is cold).

        Args:
            sac: Current agent state.
            key: JAX PRNG key; split internally for actor/critic updates and
                optional alpha update.

        Returns:
            ``(new_state, metrics)`` where metrics contains ``"q"``,
            ``"entropy"``, ``"v"``, ``"critic_loss"``, ``"target_q"``,
            and ``"alpha"`` (when ``params.autotune_alpha`` is True).
        """
        key_actor, key_critic, key_mc_critic = jax.random.split(key, 3)

        # Actor gradient: maximise soft value V(s) = Q(s,a) - α log π(a|s)
        grads_actor, metrics = jax.grad(loss_actor, has_aux=True)(
            sac.actor,
            sac.critic_target,
            sac.online_buffer,
            online_buffer_sample,
            sac.alpha,
            key_actor,
        )
        # Critic gradient: minimise SAC Bellman MSE
        grads_critic, metrics_critic = jax.grad(loss_critic, argnums=1, has_aux=True)(
            sac.actor_target,
            sac.critic,
            sac.critic_target,
            sac.online_buffer,
            sac.alpha,
            key_critic,
        )
        metrics.update(metrics_critic)

        if approximate_lambda:
            grads_mc_critic, metrics_mc_critic = jax.grad(
                loss_mc_critic_sac, argnums=1, has_aux=True
            )(
                sac.actor,
                sac.mc_critic,
                sac.critic_target,
                sac.online_buffer,
                sac.alpha,
                key_mc_critic,
            )
            metrics.update(metrics_mc_critic)

        updates, new_actor_opt = actor_optimizer.update(
            grads_actor, sac.actor_optimizer_state
        )
        new_actor = optax.apply_updates(sac.actor, updates)  # type: ignore

        updates, new_critic_opt = critic_optimizer.update(
            grads_critic, sac.critic_optimizer_state
        )
        new_critic = optax.apply_updates(sac.critic, updates)  # type: ignore

        if approximate_lambda:
            updates, new_mc_critic_opt = mc_critic_optimizer.update(
                grads_mc_critic, sac.mc_critic_optimizer_state
            )
            new_mc_critic = optax.apply_updates(sac.mc_critic, updates)  # type: ignore

        if params.autotune_alpha:
            grads_alpha = jax.grad(loss_alpha)(sac.log_alpha, -metrics["entropy"])
            updates, new_alpha_opt = alpha_optimizer.update(
                grads_alpha, sac.alpha_optimizer_state
            )
            new_log_alpha = optax.apply_updates(sac.log_alpha, updates)  # type: ignore
            new_alpha = jnp.exp(new_log_alpha)  # type: ignore
            metrics.update({"alpha": new_alpha})
        else:
            new_alpha_opt = sac.alpha_optimizer_state
            new_log_alpha = sac.log_alpha
            new_alpha = sac.alpha

        new_actor_target = jax.tree.map(
            lambda x, y: (1 - params.tau) * x + params.tau * y,
            sac.actor_target,
            new_actor,
        )
        new_critic_target = jax.tree.map(
            lambda x, y: (1 - params.tau) * x + params.tau * y,
            sac.critic_target,
            new_critic,
        )
        if approximate_lambda:
            new_mc_critic_target = jax.tree.map(
                lambda x, y: (1 - params.tau) * x + params.tau * y,
                sac.mc_critic_target,
                new_mc_critic,
            )
        return (
            SACState(
                new_actor,  # type: ignore
                new_critic,  # type: ignore
                new_mc_critic if approximate_lambda else None,
                new_actor_target,
                new_critic_target,
                new_mc_critic_target if approximate_lambda else None,
                new_actor_opt,
                new_critic_opt,
                new_mc_critic_opt if approximate_lambda else None,
                new_alpha_opt,  # type: ignore
                new_alpha,
                new_log_alpha,  # type: ignore
                sac.online_buffer,
            ),
            metrics,
        )

    @partial(jax.jit, static_argnames=["env"])
    def _train_jit(
        sac: SACState,
        env,
        env_params,
        env_state,
        key: jax.Array,
    ) -> Tuple[SACState, any, dict]:
        print("compiling...")

        def scan_fun(carry, _):
            sac, env_state, key = carry
            key, next_key, env_key, update_key = jax.random.split(key, 4)
            sac, env_state = run_env_step(sac, env, env_params, env_state, env_key)
            sac, metrics = update_step_sac(sac, update_key)
            return (sac, env_state, next_key), metrics

        (sac, env_state, _), metrics = jax.lax.scan(
            scan_fun, (sac, env_state, key), length=train_steps
        )
        metrics = jax.tree.map(lambda x: x.mean(), metrics)
        return sac, env_state, metrics

    def train(
        sac: SACState,
        env,
        env_params,
        env_state,
        key: jax.Array,
    ) -> Tuple[SACState, any, dict]:
        """Collect online experience and run SAC gradient updates.

        Each step of the inner scan loop:

        1. Calls ``env.get_obs`` to obtain the current observation.
        2. Samples an action from the current policy.
        3. Steps the gymnax environment and writes the transition
           ``(obs, action, reward, terminated)`` into ``sac.online_buffer``.
           Gymnax's base ``step()`` automatically resets the environment state
           when the episode terminates, so no separate reset call is needed.
        4. Runs one SAC gradient update via :func:`update_step_sac`.

        The entire loop is compiled as a single XLA program after the first
        invocation (via the ``_train_sac_jit`` inner function).

        A Python-level check is performed on every call to ensure the online
        buffer is warm (at least ``params.online_batch_size`` sampleable
        transitions).  If the buffer is cold, :func:`prefill_buffer` is called
        automatically with the provided environment before the JIT-compiled
        training loop runs.  This happens transparently on the first call when
        starting from a freshly constructed agent.

        Args:
            sac: Current agent state.
            env: Gymnax environment object.  Treated as a static (non-traced)
                Python object; passed as a ``static_argnames`` argument to the
                inner JIT.
            env_params: Gymnax environment parameters pytree.
            env_state: Current gymnax environment state pytree.  Updated by
                each environment step and returned.
            key: JAX PRNG key; split internally across all steps.

        Returns:
            ``(new_sac, new_env_state, metrics)`` where each metric scalar is
            the mean over all ``train_steps`` steps.
        """
        n_ok = int(sac.online_buffer.sampling_ok.sum())
        if n_ok < params.online_batch_size:
            key, prefill_key = jax.random.split(key)
            sac, env_state = prefill_buffer(
                sac, env, env_params, env_state, params.online_batch_size, prefill_key
            )
        return _train_jit(sac, env, env_params, env_state, key)

    def prefill_buffer(
        sac: SACState,
        env,
        env_params,
        env_state,
        n_steps: int,
        key: jax.Array,
    ):
        """Pre-fill the online buffer with real environment interactions.

        Runs ``n_steps`` steps using a uniform random policy and writes the
        resulting transitions into ``sac.online_buffer``.  The last step is
        force-terminated so that all ``n_steps`` written slots are immediately
        sampleable by :func:`update_step_sac`.

        For discrete action spaces the action is drawn uniformly from
        ``{0, …, action_dim-1}`` and ``behaviour_key`` is set to
        ``1/action_dim``.  For continuous spaces ``u ~ N(0, I)`` is sampled,
        squashed through tanh with optional affine rescaling, and
        ``behaviour_key`` is set to ``exp(log_prob)`` using the same
        change-of-variables formula as :func:`get_importance_ratios`.

        Args:
            sac: Current agent state.  Only ``sac.online_buffer`` is mutated.
            env: Gymnax environment object (static — not traced by JAX).
            env_params: Gymnax environment parameters pytree.
            env_state: Current gymnax environment state pytree.
            n_steps: Number of transitions to collect.
            key: JAX PRNG key; split internally for each action sample and
                environment step.

        Returns:
            ``(new_sac, new_env_state)`` where ``new_sac`` has an updated
            ``online_buffer`` and ``new_env_state`` is the post-step gymnax
            state.
        """
        for i in range(n_steps):
            key, key_act, key_step = jax.random.split(key, 3)
            obs = env.get_obs(env_state, env_params)
            if is_discrete:
                action_idx = jax.random.randint(key_act, (), 0, action_dim)
                action = action_idx.astype(jnp.float32)
                prob = jnp.float32(1.0 / action_dim)
                env_action = action_idx
            else:
                u = jax.random.normal(key_act, (action_dim,))
                y_t = jnp.tanh(u)
                action = y_t * action_scale + action_bias
                log_prob = (
                    -(u**2) / 2
                    - 0.5 * jnp.log(2 * jnp.pi)
                    - jnp.log(action_scale * (1 - y_t**2) + 1e-6)
                ).sum()
                prob = jnp.exp(log_prob)
                env_action = action
            _next_obs, env_state, reward, done, _ = env.step(
                key_step, env_state, env_action, env_params
            )
            # Force-terminate the last step so that all n_steps slots are
            # immediately sampleable (terminal transitions need no successor).
            terminated = bool(done) | (i == n_steps - 1)
            transition = {
                obs_key: obs,
                action_key: jnp.atleast_1d(action),
                reward_key: jnp.asarray(reward, dtype=jnp.float32),
                terminated_key: jnp.asarray(terminated, dtype=jnp.float32),
            }
            if approximate_lambda:
                transition[behaviour_key] = prob
                if not is_discrete:
                    transition[unsquashed_action_key] = jnp.atleast_1d(u)
            sac = sac._replace(
                online_buffer=online_buffer_functions.add(
                    sac.online_buffer, transition, terminated=terminated
                )
            )
        return sac, env_state

    fns = SACFunctions(predict, train, get_importance_ratios, prefill_buffer)
    if debug:
        return (
            iqlearn,
            fns,
            DebugFunctions(calculate_td_lambda, get_q, get_entropy),
        )
    return (iqlearn, fns)
